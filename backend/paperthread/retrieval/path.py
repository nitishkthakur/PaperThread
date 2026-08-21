"""The learning-path pipeline — stages 1 through 5, in order.

    1  retrieve      multi-provider search, dedup, RRF          search.py
    2  expand        citation graph: ancestors and later work   expansion.py
    3  score         age-rescaled PageRank, communities         graph.py
    -  select        quota-based, foundation + surface          selection.py
    4  judge         roles, prerequisites, explanations         judgment.py
    5  order         DAG under the citation constraint          ordering.py

Stage 0 (topic decomposition into subtopics *before* searching) and Stage 6
(personalization against reading history) are not built. Subtopics here are induced from
the citation graph of the results rather than proposed up front, and the path is built for
the topic rather than for a particular reader — both are stated in the returned notes, and
neither is papered over.

**Each stage degrades independently.** No provider with a citation graph, no API key, no
edges in the subgraph — each of those subtracts a capability and says so, and none of them
turns into an error. The floor is stage 1 alone, which is a ranked candidate set honestly
labelled as one.
"""

from __future__ import annotations

import logging

from ..config import Config
from ..domain.path import LearningPath, PaperSignals, PathStep
from ..llm.registry import LLMClient
from ..providers.http_cache import HTTPCache
from .expansion import CitationExpansionService
from .graph import analyze
from .judgment import JudgmentService
from .ordering import order_path
from .search import TopicSearchService
from .selection import candidate_pairs, select_path_papers

logger = logging.getLogger(__name__)

# How many candidate prerequisite pairs are worth judging. The cap is a budget, not a
# quality threshold — see selection.py.
MAX_JUDGED_PAIRS = 60


class LearningPathService:
    def __init__(self, config: Config, llm_client: LLMClient | None = None) -> None:
        self.config = config
        # One cache shared by both provider stages, so a paper fetched during search is
        # not fetched again during expansion.
        self.cache = HTTPCache(
            config.provider_cache.cache_dir, enabled=config.provider_cache.enabled
        )
        self.search_service = TopicSearchService(config, cache=self.cache)
        self.expansion_service = CitationExpansionService(config, cache=self.cache)
        self.judgment_service = JudgmentService(llm_client or LLMClient(config))

    async def build(self, topic: str, limit: int | None = None) -> LearningPath:
        topic = topic.strip()
        if not topic:
            raise ValueError("topic must not be empty")

        path = LearningPath(topic=topic, layers_used=self._layers())

        # -- Stage 1 --------------------------------------------------------------------
        search = await self.search_service.search(
            topic, limit=self.config.retrieval.max_candidates, standalone=False
        )
        path.stages_run.append("retrieve")
        path.notes.extend(search.notes)
        path.degraded = search.degraded
        if not search.papers:
            path.notes.append("No candidates matched this topic, so there is no path to build.")
            return path

        lexical = {ranked.paper.canonical_id: ranked.score for ranked in search.papers}

        # -- Stage 2 --------------------------------------------------------------------
        corpus = await self.expansion_service.expand(topic, search.papers)
        path.notes.extend(corpus.notes)
        if corpus.ran:
            path.stages_run.append("expand")
        if any(not outcome.ok for outcome in corpus.outcomes):
            path.degraded = True

        # -- Stage 3 --------------------------------------------------------------------
        years = {paper_id: paper.year for paper_id, paper in corpus.papers.items()}
        analysis = analyze(
            list(corpus.papers), corpus.edge_pairs(), years, self.config.retrieval.graph
        )
        path.stages_run.append("score")
        path.notes.extend(analysis.notes)

        # -- Selection ------------------------------------------------------------------
        budget = limit or self.config.retrieval.graph.max_path_papers
        selection = select_path_papers(
            corpus, analysis, budget, self.config.retrieval.expansion.min_co_citations
        )
        path.notes.extend(selection.notes)
        if not selection.paper_ids:
            path.notes.append("Nothing survived selection; returning an empty path.")
            return path

        pairs = candidate_pairs(
            selection.paper_ids,
            corpus.edge_pairs(),
            corpus.papers,
            corpus.co_citations,
            analysis,
            MAX_JUDGED_PAIRS,
        )

        # -- Stage 4 --------------------------------------------------------------------
        assessment = await self.judgment_service.assess(
            topic=topic,
            paper_ids=selection.paper_ids,
            corpus=corpus,
            analysis=analysis,
            pairs=pairs,
            reader=self._reader_description(topic),
        )
        path.stages_run.append("judge")
        path.notes.extend(assessment.notes)
        if assessment.used_llm:
            path.stages_run.append("judge:llm")
        else:
            path.degraded = True

        # -- Stage 5 --------------------------------------------------------------------
        def rank_key(paper_id: str):
            # Within a level nothing forces an order, so present the most structurally
            # central first — that is the paper most worth reading if the reader stops.
            return (
                -analysis.age_rescaled.get(paper_id, 0.0),
                -corpus.co_citations.get(paper_id, 0),
                corpus.papers[paper_id].year or 9999,
            )

        selected = set(selection.paper_ids)
        ordering = order_path(
            selection.paper_ids,
            {(a, b) for a, b in corpus.edge_pairs() if a in selected and b in selected},
            {
                (edge.prerequisite_id, edge.dependent_id)
                for edge in assessment.prerequisite_edges
            },
            years,
            rank_key,
        )
        path.stages_run.append("order")
        path.notes.extend(ordering.notes)

        # -- Assemble -------------------------------------------------------------------
        steps: list[PathStep] = []
        for paper_id in selection.paper_ids:
            paper = corpus.papers[paper_id]
            steps.append(
                PathStep(
                    order=ordering.order[paper_id],
                    level=ordering.levels[paper_id],
                    paper=paper,
                    role=assessment.roles[paper_id],
                    signals=PaperSignals(
                        co_citations=corpus.co_citations.get(paper_id, 0),
                        pagerank=analysis.pagerank.get(paper_id, 0.0),
                        age_rescaled_pagerank=analysis.age_rescaled.get(paper_id, 0.0),
                        lexical_score=lexical.get(paper_id, 0.0),
                        in_degree=analysis.in_degree.get(paper_id, 0),
                        out_degree=analysis.out_degree.get(paper_id, 0),
                        discovered_by_expansion=paper_id in corpus.discovered_ids,
                    ),
                    explanation=assessment.explanations[paper_id],
                    subtopic_id=assessment.subtopic_of.get(paper_id),
                    prerequisite_ids=ordering.direct_prerequisites.get(paper_id, ()),
                )
            )
        steps.sort(key=lambda step: step.order)

        path.steps = steps
        path.subtopics = assessment.subtopics
        # Only edges between papers that made the path — an edge to a paper the reader
        # cannot see is not an explanation, it is a dangling reference.
        path.edges = [
            edge
            for edge in assessment.prerequisite_edges
            if edge.prerequisite_id in selected and edge.dependent_id in selected
        ]
        path.notes.append(self._summary(path, corpus, assessment.used_llm))
        return path

    def _reader_description(self, topic: str) -> str:
        """What Stage 4 is told about the reader.

        Stage 6 does not exist, so this states the absence rather than inventing a persona.
        An LLM given "the reader is an expert" will confidently tailor to a fiction, and
        §5's "why for this user" would become the least trustworthy field on the page.
        """
        return (
            f"A reader who has entered the topic {topic!r} and has no recorded reading "
            f"history in this system yet. Do not assume prior familiarity beyond general "
            f"background in the area, and do not claim to know what they have read."
        )

    def _summary(self, path: LearningPath, corpus, used_llm: bool) -> str:
        expanded = sum(1 for step in path.steps if step.signals.discovered_by_expansion)
        edges = len(path.edges)
        if used_llm:
            basis = "prerequisites judged by an LLM over citation candidates"
        else:
            basis = "prerequisites inferred from shared citations, not reasoned"
        return (
            f"{len(path.steps)} papers across {path.levels} level(s); {expanded} reached by "
            f"citation expansion rather than search; {edges} prerequisite edge(s); {basis}."
        )

    def _layers(self) -> list[str]:
        layers = self.config.retrieval.layers
        return [
            name
            for name, enabled in (
                ("lexical", layers.lexical),
                ("local_nlp", layers.local_nlp),
                ("embeddings", layers.embeddings),
                ("reranking", layers.reranking),
                ("llm", layers.llm),
            )
            if enabled
        ]
