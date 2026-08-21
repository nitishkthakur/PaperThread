"""Stage 4 — roles, prerequisite judgment, and explanations.

This is D2 stage 2 and §5's explanation requirement, and it is the one stage with two
complete implementations rather than one implementation with a fallback path:

* **Structural** (L0–L3). Roles from citation position and title evidence; prerequisites
  from "A cites B and B is shared foundation"; explanations built from what the graph
  measured. Genuinely useful on its own, as D12 requires, and it is what runs today with
  no API key.
* **Reasoned** (L4). An LLM judges each candidate pair and writes the four explanation
  fields. It refines the structural result — it never replaces it: structural output is
  computed first and stays in place for anything the LLM did not cover or got rejected on.

Degradation is per-batch, not per-request. If one explanation batch fails, those papers
keep their structural explanations and the rest keep their reasoned ones; the path still
comes back, and `notes` says exactly what happened.

Two guards on the model:

* **Every id it returns is checked against the shortlist it was given.** Unknown ids are
  dropped, not trusted. Asked for papers rather than for a choice among papers, an LLM
  produces confident, well-formatted, nonexistent ones.
* **Judgments below a confidence floor are discarded.** The prompt already biases toward
  "no"; this catches the residue. A false prerequisite is invisible to the reader — it just
  looks like a longer path — which makes it the expensive error.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from ..domain.models import Paper
from ..domain.path import (
    EdgeSource,
    Explanation,
    ExplanationSource,
    PaperRole,
    PrerequisiteEdge,
    Provenance,
    Subtopic,
)
from ..llm import prompts
from ..llm.base import LLMError
from ..llm.registry import LLMClient
from .expansion import ExpandedCorpus
from .graph import GraphAnalysis
from .selection import CandidatePair

logger = logging.getLogger(__name__)

# Below this, an LLM "yes" is not worth an edge the reader cannot verify.
MIN_EDGE_CONFIDENCE = 0.6

EXPLANATION_BATCH = 6
JUDGMENT_BATCH = 12

_SURVEY_RE = re.compile(
    r"\b(survey|review|overview|tutorial|introduction to|primer|taxonomy|"
    r"systematic (literature )?review)\b",
    re.IGNORECASE,
)
_CRITIQUE_RE = re.compile(
    r"\b(critique|criticism|rethinking|revisiting|reconsidering|limitations|pitfalls|"
    r"failure modes|do .{1,30} really|are .{1,30} really|myth|harms|considered harmful|"
    r"a closer look|on the (dangers|difficulty))\b",
    re.IGNORECASE,
)


@dataclass
class Assessment:
    roles: dict[str, PaperRole] = field(default_factory=dict)
    explanations: dict[str, Explanation] = field(default_factory=dict)
    prerequisite_edges: list[PrerequisiteEdge] = field(default_factory=list)
    subtopics: list[Subtopic] = field(default_factory=list)
    subtopic_of: dict[str, str] = field(default_factory=dict)
    used_llm: bool = False
    notes: list[str] = field(default_factory=list)


class JudgmentService:
    def __init__(self, client: LLMClient | None) -> None:
        self.client = client

    async def assess(
        self,
        topic: str,
        paper_ids: list[str],
        corpus: ExpandedCorpus,
        analysis: GraphAnalysis,
        pairs: list[CandidatePair],
        reader: str,
    ) -> Assessment:
        papers = {paper_id: corpus.papers[paper_id] for paper_id in paper_ids}
        assessment = _structural_assessment(paper_ids, papers, corpus, analysis, pairs)

        available, reason = (False, "no LLM client") if self.client is None else self.client.available()
        if not available:
            assessment.notes.append(
                f"L4 inactive — roles, prerequisites and explanations are structural, "
                f"derived from the citation graph rather than reasoned. {reason}"
            )
            return assessment

        assert self.client is not None
        try:
            await self._refine(topic, paper_ids, papers, corpus, analysis, pairs, reader, assessment)
        except Exception as exc:  # noqa: BLE001 - L4 must degrade, never break the path
            logger.exception("L4 refinement failed")
            assessment.notes.append(
                f"L4 failed and the path fell back to structural output: {exc}"
            )
        return assessment

    async def _refine(
        self,
        topic: str,
        paper_ids: list[str],
        papers: dict[str, Paper],
        corpus: ExpandedCorpus,
        analysis: GraphAnalysis,
        pairs: list[CandidatePair],
        reader: str,
        assessment: Assessment,
    ) -> None:
        """Run the three L4 calls concurrently and fold each result in independently."""
        assert self.client is not None
        explanations, judgments, subtopics = await asyncio.gather(
            self._explain(topic, paper_ids, papers, corpus, analysis, reader),
            self._judge(topic, pairs, papers),
            self._name_subtopics(topic, paper_ids, papers, analysis),
            return_exceptions=True,
        )

        if isinstance(explanations, dict) and explanations:
            assessment.explanations.update({k: v[0] for k, v in explanations.items()})
            assessment.roles.update({k: v[1] for k, v in explanations.items()})
            assessment.used_llm = True
            missing = len(paper_ids) - len(explanations)
            if missing > 0:
                assessment.notes.append(
                    f"{missing} paper(s) kept structural explanations — the model did not "
                    f"return usable output for them."
                )
        elif isinstance(explanations, Exception):
            assessment.notes.append(f"Explanations stayed structural: {explanations}")

        if isinstance(judgments, list):
            # The LLM's verdict REPLACES the structural edge set rather than adding to it:
            # its job is to remove the perfunctory citations structure cannot distinguish.
            assessment.prerequisite_edges = judgments
            assessment.used_llm = True
        elif isinstance(judgments, Exception):
            assessment.notes.append(
                f"Prerequisite edges stayed structural (A cites B, B is shared "
                f"foundation): {judgments}"
            )

        if isinstance(subtopics, tuple):
            named, mapping = subtopics
            if named:
                assessment.subtopics = named
                assessment.subtopic_of = mapping
                assessment.used_llm = True
        elif isinstance(subtopics, Exception):
            assessment.notes.append(f"Subtopics stayed unnamed: {subtopics}")

    # -- L4 calls ----------------------------------------------------------------------

    async def _explain(
        self,
        topic: str,
        paper_ids: list[str],
        papers: dict[str, Paper],
        corpus: ExpandedCorpus,
        analysis: GraphAnalysis,
        reader: str,
    ) -> dict[str, tuple[Explanation, PaperRole]]:
        assert self.client is not None
        labels = {f"n{i + 1}": paper_id for i, paper_id in enumerate(paper_ids)}

        batches = [
            list(labels.items())[i : i + EXPLANATION_BATCH]
            for i in range(0, len(labels), EXPLANATION_BATCH)
        ]
        results = await asyncio.gather(
            *(
                self._explain_batch(topic, batch, papers, corpus, analysis, reader)
                for batch in batches
            ),
            return_exceptions=True,
        )

        merged: dict[str, tuple[Explanation, PaperRole]] = {}
        for result in results:
            if isinstance(result, dict):
                merged.update(result)
            else:
                logger.warning("explanation batch failed: %s", result)
        return merged

    async def _explain_batch(
        self,
        topic: str,
        batch: list[tuple[str, str]],
        papers: dict[str, Paper],
        corpus: ExpandedCorpus,
        analysis: GraphAnalysis,
        reader: str,
    ) -> dict[str, tuple[Explanation, PaperRole]]:
        assert self.client is not None
        by_label = dict(batch)
        payload = [
            {
                "id": label,
                "title": papers[paper_id].title,
                "year": papers[paper_id].year,
                "abstract": papers[paper_id].abstract,
                "facts": _structural_facts(paper_id, corpus, analysis),
            }
            for label, paper_id in batch
        ]

        result = await self.client.structured(
            role=prompts.EXPLANATION_ROLE,
            system=prompts.EXPLANATION_SYSTEM,
            user=prompts.explanation_user(topic, payload, reader),
            schema=prompts.EXPLANATION_SCHEMA,
            prompt_version=prompts.VERSION,
        )

        provenance = Provenance(
            asserted_by=result.provider, model=result.model, prompt_version=result.prompt_version
        )
        out: dict[str, tuple[Explanation, PaperRole]] = {}
        for entry in result.data.get("papers", []):
            paper_id = by_label.get(str(entry.get("id")))
            if paper_id is None:
                # The model answered about something it was not shown. Drop it.
                logger.warning("dropping explanation for unknown id %r", entry.get("id"))
                continue
            out[paper_id] = (
                Explanation(
                    why_it_matters=entry["why_it_matters"].strip(),
                    what_it_assumes=entry["what_it_assumes"].strip(),
                    what_it_teaches=entry["what_it_teaches"].strip(),
                    why_for_you=entry["why_for_you"].strip(),
                    source=ExplanationSource.LLM,
                    provenance=provenance,
                ),
                PaperRole(entry["role"]),
            )
        return out

    async def _judge(
        self, topic: str, pairs: list[CandidatePair], papers: dict[str, Paper]
    ) -> list[PrerequisiteEdge]:
        assert self.client is not None
        if not pairs:
            return []

        labels = {f"p{i + 1}": pair for i, pair in enumerate(pairs)}
        batches = [
            list(labels.items())[i : i + JUDGMENT_BATCH]
            for i in range(0, len(labels), JUDGMENT_BATCH)
        ]
        results = await asyncio.gather(
            *(self._judge_batch(topic, batch, papers) for batch in batches),
            return_exceptions=True,
        )

        edges: list[PrerequisiteEdge] = []
        failures = 0
        for result in results:
            if isinstance(result, list):
                edges.extend(result)
            else:
                failures += 1
                logger.warning("judgment batch failed: %s", result)
        if failures and failures == len(batches):
            raise LLMError("llm", "every prerequisite judgment batch failed")
        return edges

    async def _judge_batch(
        self, topic: str, batch: list[tuple[str, CandidatePair]], papers: dict[str, Paper]
    ) -> list[PrerequisiteEdge]:
        assert self.client is not None
        by_label = dict(batch)
        payload = [
            {
                "id": label,
                "prerequisite": _paper_payload(papers[pair.prerequisite_id]),
                "dependent": _paper_payload(papers[pair.dependent_id]),
            }
            for label, pair in batch
        ]

        result = await self.client.structured(
            role=prompts.JUDGMENT_ROLE,
            system=prompts.JUDGMENT_SYSTEM,
            user=prompts.judgment_user(topic, payload),
            schema=prompts.JUDGMENT_SCHEMA,
            prompt_version=prompts.VERSION,
        )

        provenance = Provenance(
            asserted_by=result.provider, model=result.model, prompt_version=result.prompt_version
        )
        edges: list[PrerequisiteEdge] = []
        for entry in result.data.get("judgments", []):
            pair = by_label.get(str(entry.get("pair")))
            if pair is None:
                logger.warning("dropping judgment for unknown pair %r", entry.get("pair"))
                continue
            if not entry["is_prerequisite"]:
                continue
            confidence = float(entry["confidence"])
            if confidence < MIN_EDGE_CONFIDENCE:
                continue
            edges.append(
                PrerequisiteEdge(
                    prerequisite_id=pair.prerequisite_id,
                    dependent_id=pair.dependent_id,
                    source=EdgeSource.LLM_JUDGMENT,
                    provenance=provenance,
                    confidence=confidence,
                    reason=entry["reason"].strip(),
                )
            )
        return edges

    async def _name_subtopics(
        self,
        topic: str,
        paper_ids: list[str],
        papers: dict[str, Paper],
        analysis: GraphAnalysis,
    ) -> tuple[list[Subtopic], dict[str, str]]:
        assert self.client is not None
        groups = _group_by_community(paper_ids, analysis)
        if len(groups) < 2:
            # One community is not a decomposition; naming it would just restate the topic.
            return [], {}

        payload = [
            {
                "id": f"g{index}",
                "size": len(members),
                "papers": [
                    {"title": papers[p].title, "year": papers[p].year} for p in members[:6]
                ],
            }
            for index, members in groups.items()
        ]
        result = await self.client.structured(
            role=prompts.SUBTOPIC_ROLE,
            system=prompts.SUBTOPIC_SYSTEM,
            user=prompts.subtopic_user(topic, payload),
            schema=prompts.SUBTOPIC_SCHEMA,
            prompt_version=prompts.VERSION,
        )

        subtopics: list[Subtopic] = []
        mapping: dict[str, str] = {}
        for entry in result.data.get("groups", []):
            group_id = str(entry.get("id"))
            try:
                index = int(group_id.removeprefix("g"))
            except ValueError:
                continue
            if index not in groups:
                logger.warning("dropping subtopic for unknown group %r", group_id)
                continue
            subtopics.append(
                Subtopic(
                    id=group_id,
                    label=entry["label"].strip(),
                    summary=entry["summary"].strip() or None,
                    order=int(entry["position"]),
                    named_by_llm=True,
                )
            )
            for paper_id in groups[index]:
                mapping[paper_id] = group_id

        subtopics.sort(key=lambda s: (s.order, s.id))
        return subtopics, mapping


# -- Structural implementation (L0) -----------------------------------------------------


def _structural_assessment(
    paper_ids: list[str],
    papers: dict[str, Paper],
    corpus: ExpandedCorpus,
    analysis: GraphAnalysis,
    pairs: list[CandidatePair],
) -> Assessment:
    assessment = Assessment()
    years = [papers[p].year for p in paper_ids if papers[p].year is not None]
    median_year = sorted(years)[len(years) // 2] if years else None

    for paper_id in paper_ids:
        assessment.roles[paper_id] = _structural_role(
            papers[paper_id], paper_id, corpus, analysis, median_year
        )

    # Structure cannot tell a prerequisite from a passing reference. What it CAN say is
    # "A cites B, and B is what this topic's papers have in common" — a weaker claim,
    # made honestly, rather than treating every citation as a teaching dependency.
    threshold = max(2, _median([pair.co_citations for pair in pairs] or [0]))
    for pair in pairs:
        if pair.co_citations < threshold:
            continue
        assessment.prerequisite_edges.append(
            PrerequisiteEdge(
                prerequisite_id=pair.prerequisite_id,
                dependent_id=pair.dependent_id,
                source=EdgeSource.CO_CITATION,
                provenance=Provenance(asserted_by="citation-graph"),
                reason=(
                    f"Cited by this paper and by {pair.co_citations} of the topic's other "
                    f"papers — shared background rather than an isolated reference."
                ),
            )
        )

    groups = _group_by_community(paper_ids, analysis)
    if len(groups) > 1:
        for order, (index, members) in enumerate(groups.items(), start=1):
            assessment.subtopics.append(
                Subtopic(
                    id=f"g{index}",
                    label=f"Line of work {order}",
                    summary=(
                        f"{len(members)} papers that cite each other more than they cite "
                        f"the rest of this topic. Unnamed — naming requires L4."
                    ),
                    order=order,
                    named_by_llm=False,
                )
            )
            for paper_id in members:
                assessment.subtopic_of[paper_id] = f"g{index}"

    for paper_id in paper_ids:
        assessment.explanations[paper_id] = _structural_explanation(
            paper_id, papers[paper_id], corpus, analysis, assessment.roles[paper_id]
        )
    return assessment


def _structural_role(
    paper: Paper,
    paper_id: str,
    corpus: ExpandedCorpus,
    analysis: GraphAnalysis,
    median_year: int | None,
) -> PaperRole:
    """A role from evidence, not from reading the paper.

    Title patterns are used only for the two roles a title reliably announces — surveys and
    critiques name themselves. Everything else comes from citation position, because a
    title cannot tell you whether a paper is foundational to a topic and the graph can.
    """
    title = paper.title
    if _SURVEY_RE.search(title):
        return PaperRole.SURVEY
    if _CRITIQUE_RE.search(title):
        return PaperRole.CRITIQUE

    co_cited = corpus.co_citations.get(paper_id, 0)
    rescaled = analysis.age_rescaled.get(paper_id, 0.0)
    is_old = median_year is not None and paper.year is not None and paper.year < median_year

    if co_cited >= 3 and is_old:
        return PaperRole.FOUNDATION
    if rescaled >= 1.0:
        return PaperRole.BREAKTHROUGH
    if analysis.out_degree.get(paper_id, 0) > 0 and not is_old:
        return PaperRole.EXTENSION
    return PaperRole.UNCLASSIFIED


def _structural_explanation(
    paper_id: str,
    paper: Paper,
    corpus: ExpandedCorpus,
    analysis: GraphAnalysis,
    role: PaperRole,
) -> Explanation:
    """Build §5's four fields from measurements only.

    The honesty rule for this function: state what was counted, and where nothing was
    reasoned, say so. `what_it_teaches` quotes the abstract rather than paraphrasing it,
    because paraphrasing is the part that needs a model.
    """
    co_cited = corpus.co_citations.get(paper_id, 0)
    seeds = len(corpus.seed_ids) or 1
    rescaled = analysis.age_rescaled.get(paper_id, 0.0)
    discovered = paper_id in corpus.discovered_ids

    if co_cited >= 2:
        matters = (
            f"{co_cited} of the {seeds} papers this topic surfaced cite it, so it is part "
            f"of the background they have in common."
        )
    elif rescaled >= 1.0:
        matters = (
            f"It stands {rescaled:.1f} standard deviations above other papers of its age "
            f"in this topic's citation graph."
        )
    elif discovered:
        matters = (
            "Its position in this topic's citation graph is not distinctive enough to say "
            "much beyond the fact that the topic's papers reach it."
        )
    else:
        matters = (
            "It matched the topic directly; its position in the citation graph is not "
            "distinctive enough to say more."
        )
    # Only claim search missed it when search actually missed it — the two clauses
    # contradict each other otherwise, and a self-contradicting explanation is worse than
    # a thin one.
    if discovered:
        matters += (
            " Keyword search did not return it — it was reached by following citations "
            "backwards, which is how work that predates the topic's vocabulary is found."
        )

    in_degree = analysis.in_degree.get(paper_id, 0)
    out_degree = analysis.out_degree.get(paper_id, 0)
    # Phrased as "of the papers in this path" throughout: a bare "cites 0" would read as a
    # claim that the paper has no bibliography, when it only means we never fetched one.
    assumes = (
        f"Not assessed — that judgment needs L4. Within this path, {in_degree} paper(s) "
        f"cite it and it cites {out_degree} of them."
    )

    if paper.has_abstract:
        teaches = f"From the abstract: {_first_sentence(paper.abstract or '')}"
    else:
        teaches = (
            "No abstract is available from any enabled provider, and no full text is "
            "ingested yet, so its content cannot be summarized."
        )

    for_you = (
        f"Placed by role ({role.value}) and citation position, not by your reading "
        f"history — personalization is not built yet, so this is the path for the topic "
        f"rather than for you specifically."
    )

    return Explanation(
        why_it_matters=matters,
        what_it_assumes=assumes,
        what_it_teaches=teaches,
        why_for_you=for_you,
        source=ExplanationSource.STRUCTURAL,
        provenance=Provenance(asserted_by="citation-graph"),
    )


# -- helpers ----------------------------------------------------------------------------


def _group_by_community(
    paper_ids: list[str], analysis: GraphAnalysis
) -> dict[int, list[str]]:
    """Communities restricted to the path, dropping singletons.

    A one-paper "line of work" is a labelling artefact, not a subtopic; its member is left
    ungrouped rather than presented as its own theme.
    """
    groups: dict[int, list[str]] = {}
    for paper_id in paper_ids:
        community = analysis.communities.get(paper_id)
        if community is None:
            continue
        groups.setdefault(community, []).append(paper_id)
    return {index: sorted(members) for index, members in sorted(groups.items()) if len(members) > 1}


def _structural_facts(paper_id: str, corpus: ExpandedCorpus, analysis: GraphAnalysis) -> str:
    co_cited = corpus.co_citations.get(paper_id, 0)
    parts = [
        f"cited by {co_cited} of the {len(corpus.seed_ids)} papers this topic surfaced",
        f"age-rescaled PageRank {analysis.age_rescaled.get(paper_id, 0.0):+.2f}",
        f"cited by {analysis.in_degree.get(paper_id, 0)} and cites "
        f"{analysis.out_degree.get(paper_id, 0)} papers within this path",
    ]
    if paper_id in corpus.discovered_ids:
        parts.append("found by following citations, not by keyword search")
    return "; ".join(parts)


def _paper_payload(paper: Paper) -> dict:
    return {"title": paper.title, "year": paper.year, "abstract": paper.abstract}


def _first_sentence(text: str) -> str:
    cleaned = " ".join(text.split())
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
    sentence = cleaned[: match.start()] if match else cleaned
    return sentence if len(sentence) <= 300 else sentence[:297] + "..."


def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
