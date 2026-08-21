"""Learning-path construction strategies — the tunable part of the system.

Centrality answers "which papers matter in this literature". A learner asks a different
question: "what do I read, in what order, to understand this". Those diverge sharply. For
dropout, centrality surfaces ImageNet and ResNet — genuinely central to the surrounding
literature, and no help at all to someone learning dropout. The pedagogical ordering is a
judgment about *teaching*, and it is the one thing citation structure cannot supply.

So the algorithm is not fixed. Each strategy here is a different answer to "how do you
build a path", and which one wins is decided by evaluation, not by argument:

| Strategy | Idea | Grounding | Fails when |
|---|---|---|---|
| `syllabus` | LLM plans the whole sequence, then each step is resolved to a real paper | model knowledge | the topic postdates training, or titles resolve poorly |
| `anchor` | LLM names the anchor; its REAL references supply the prerequisites | citation graph | the anchor is unresolvable, or has no useful references |
| `rerank` | Retrieve + expand as before, then an LLM reorders into a teaching sequence | corpus | the right paper never made it into the candidate set |
| `hybrid` | `syllabus` skeleton, with `anchor`'s citation graph filling what did not resolve | both | — |
| `structural` | The pre-LLM pipeline. Baseline. | citation graph | pedagogy was never modelled |

Every strategy returns a `LearningPath`, so the API and UI are unchanged by which one runs.

**Nothing an LLM names is trusted.** Titles go through `resolver.PaperResolver`, which
refuses to substitute a near-miss, and unresolved steps are reported in the path's notes
rather than silently dropped.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..config import Config
from ..domain.identity import title_fingerprint
from ..domain.models import Paper
from ..domain.path import (
    EdgeSource,
    Explanation,
    ExplanationSource,
    LearningPath,
    PaperRole,
    PaperSignals,
    PathStep,
    PrerequisiteEdge,
    Provenance,
)
from ..llm import curriculum_prompts as cp
from ..llm.base import LLMError
from ..llm.registry import LLMClient
from ..providers.base import Capability
from ..providers.http_cache import HTTPCache
from ..providers.registry import build_paper_providers
from .expansion import CitationExpansionService
from .graph import analyze
from .resolver import PaperResolver
from .search import TopicSearchService

logger = logging.getLogger(__name__)

# A step's place in the arc of the path. Distinct from PaperRole, which describes what a
# paper IS; this describes what it is doing HERE.
STAGE_PREREQUISITE = "prerequisite"
STAGE_ANCHOR = "anchor"
STAGE_FOLLOWUP = "followup"

_STAGE_ROLE = {
    STAGE_PREREQUISITE: PaperRole.FOUNDATION,
    STAGE_ANCHOR: PaperRole.BREAKTHROUGH,
    STAGE_FOLLOWUP: PaperRole.EXTENSION,
}


@dataclass
class PlannedStep:
    """One step before it becomes a `PathStep`."""

    paper: Paper
    concept: str
    stage: str
    why_here: str
    position: int


@dataclass
class BuildContext:
    """Shared services, built once so caches and rate limiters are actually shared."""

    config: Config
    client: LLMClient
    cache: HTTPCache
    resolver: PaperResolver
    notes: list[str] = field(default_factory=list)
    models_used: set[str] = field(default_factory=set)

    @classmethod
    def create(cls, config: Config, client: LLMClient | None = None) -> BuildContext:
        cache = HTTPCache(
            config.provider_cache.cache_dir, enabled=config.provider_cache.enabled
        )
        return cls(
            config=config,
            client=client or LLMClient(config),
            cache=cache,
            resolver=PaperResolver(config, cache=cache),
        )

    def record(self, result) -> None:
        self.models_used.add(result.model)
        if result.fell_back:
            self.notes.append(f"Primary model unavailable; {result.model} answered instead.")


class CurriculumStrategy:
    """Base strategy. Subclasses implement `plan`."""

    name = "base"

    def __init__(self, context: BuildContext) -> None:
        self.context = context
        self.config = context.config
        self.client = context.client

    async def plan(self, topic: str, budget: int) -> list[PlannedStep]:
        raise NotImplementedError

    async def build(self, topic: str, budget: int) -> LearningPath:
        path = LearningPath(topic=topic, layers_used=_layers(self.config))
        path.stages_run.append(f"strategy:{self.name}")

        try:
            steps = await self.plan(topic, budget)
        except LLMError as exc:
            path.degraded = True
            path.notes.append(f"Strategy {self.name!r} could not run: {exc}")
            return path

        path.notes.extend(self.context.notes)
        if not steps:
            path.degraded = True
            path.notes.append(f"Strategy {self.name!r} produced no usable steps.")
            return path

        _assemble(path, steps, self.context)
        return path


# ======================================================================================
# syllabus — the LLM plans the whole sequence, then reality checks every step
# ======================================================================================


class SyllabusStrategy(CurriculumStrategy):
    """Ask for the teaching sequence directly, then ground each step in a real paper.

    The bet: a model that has read the literature knows the pedagogical order, and the
    thing it cannot be trusted with — whether a specific paper exists — is exactly what a
    bibliographic lookup can check. Planning and verification are separated so neither has
    to do the other's job.
    """

    name = "syllabus"

    async def plan(self, topic: str, budget: int) -> list[PlannedStep]:
        result = await self.client.structured(
            role="curriculum_planning",
            system=cp.SYLLABUS_SYSTEM_V1,
            user=cp.syllabus_user(topic, budget),
            schema=cp.SYLLABUS_SCHEMA,
            prompt_version=cp.VERSION,
            max_tokens=12000,
        )
        self.context.record(result)

        proposed = sorted(
            result.data.get("steps", []), key=lambda s: int(s.get("position", 0))
        )
        resolutions = await self.context.resolver.resolve_many(
            [(step["title"], step.get("year")) for step in proposed]
        )

        steps: list[PlannedStep] = []
        unresolved: list[str] = []
        seen_ids: set[str] = set()
        seen_titles: set[str] = set()
        for step, resolution in zip(proposed, resolutions, strict=True):
            if not resolution.ok:
                unresolved.append(step["title"])
                continue
            assert resolution.paper is not None
            paper = resolution.paper
            # Deduplicate on the TITLE as well as the id. Canonical identity merges on a
            # year window, so the same work indexed under two different years survives as
            # two nodes — which shipped Batch Normalization twice in one path, dated 2015
            # and 2024.
            fingerprint = title_fingerprint(paper.title)
            if paper.canonical_id in seen_ids or fingerprint in seen_titles:
                continue
            seen_ids.add(paper.canonical_id)
            seen_titles.add(fingerprint)

            concept, why_here = step["concept"], step["why_here"]
            if not resolution.is_confident:
                # The lookup returned something close but not the paper that was planned,
                # so the plan's prose describes a DIFFERENT paper. Presenting it here is
                # how "Is Space-Time Attention All You Need for Video Understanding?" came
                # to be labelled "the paper that defines the Transformer".
                self.context.notes.append(
                    f"Planned {step['title'][:60]!r} resolved to {paper.title[:60]!r} "
                    f"({resolution.similarity:.2f}); its rationale was discarded rather "
                    f"than reattached to a different paper."
                )
                concept = f"Closest available match for: {step['concept']}"
                why_here = (
                    f"The plan asked for {step['title']!r} at this position; this is the "
                    f"nearest paper actually found, so treat the placement as approximate."
                )

            steps.append(
                PlannedStep(
                    paper=paper,
                    concept=concept,
                    stage=step.get("stage", STAGE_FOLLOWUP),
                    why_here=why_here,
                    position=len(steps),
                )
            )

        if unresolved:
            self.context.notes.append(
                f"{len(unresolved)} planned step(s) named a paper that could not be found "
                f"and were dropped rather than substituted: "
                + "; ".join(f"{t[:70]!r}" for t in unresolved[:4])
                + ("…" if len(unresolved) > 4 else "")
            )
        return steps[:budget]


# ======================================================================================
# anchor — name the target, then take its real references as the prerequisites
# ======================================================================================


class AnchorFirstStrategy(CurriculumStrategy):
    """Identify the paper that IS the topic, then read backwards from it.

    Where `syllabus` trusts the model for the whole sequence, this trusts it for one
    judgment — which paper is the target — and derives the prerequisites from that paper's
    actual bibliography. The candidates are therefore real by construction, and the model
    is only asked to select and order among works it did not have to recall.

    That distinction matters most where `syllabus` is weakest: a topic whose literature
    postdates the model's training still has a resolvable anchor with a real reference list.
    """

    name = "anchor"

    async def plan(self, topic: str, budget: int) -> list[PlannedStep]:
        anchors = await self._find_anchors(topic)
        if not anchors:
            self.context.notes.append(
                "No anchor paper for this topic could be identified and resolved, so the "
                "path has no target to build towards."
            )
            return []

        primary = anchors[0]
        references = await self._references(primary)
        if not references:
            self.context.notes.append(
                f"Anchor {primary.title[:60]!r} resolved, but no provider returned its "
                f"reference list, so prerequisites could not be derived from it."
            )

        prerequisite_budget = max(2, int(budget * 0.5))
        prerequisites = (
            await self._select_prerequisites(topic, primary, references, prerequisite_budget)
            if references
            else []
        )
        followups = await self._followups(topic, primary, max(1, budget - len(prerequisites) - len(anchors)))

        steps: list[PlannedStep] = []
        seen: set[str] = set()

        def add(paper: Paper, concept: str, stage: str, why: str) -> None:
            if paper.canonical_id in seen or len(steps) >= budget:
                return
            seen.add(paper.canonical_id)
            steps.append(PlannedStep(paper, concept, stage, why, len(steps)))

        for paper, concept, why in prerequisites:
            add(paper, concept, STAGE_PREREQUISITE, why)
        for paper in anchors:
            add(
                paper,
                f"{topic} itself",
                STAGE_ANCHOR,
                "The paper that introduced the topic you asked about — everything before "
                "this exists to make it readable.",
            )
        for paper, concept, why in followups:
            add(paper, concept, STAGE_FOLLOWUP, why)
        return steps

    async def _find_anchors(self, topic: str) -> list[Paper]:
        result = await self.client.structured(
            role="curriculum_planning",
            system=cp.ANCHOR_SYSTEM_V1,
            user=cp.anchor_user(topic),
            schema=cp.ANCHOR_SCHEMA,
            prompt_version=cp.VERSION,
            max_tokens=6000,
        )
        self.context.record(result)

        named = result.data.get("anchors", [])[:2]
        resolutions = await self.context.resolver.resolve_many(
            [(entry["title"], entry.get("year")) for entry in named]
        )
        resolved = [r.paper for r in resolutions if r.ok and r.paper is not None]
        for entry, resolution in zip(named, resolutions, strict=True):
            if not resolution.ok:
                self.context.notes.append(
                    f"Proposed anchor {entry['title'][:70]!r} did not resolve to a real "
                    f"paper and was discarded."
                )
        return resolved

    async def _references(self, anchor: Paper) -> list[Paper]:
        providers = build_paper_providers(
            self.config, Capability.CITATIONS, cache=self.context.cache
        )
        if not providers:
            return []
        limit = self.config.retrieval.expansion.references_per_paper

        async def fetch(provider):
            try:
                return await provider.references(anchor, limit)
            except Exception as exc:  # noqa: BLE001 - degrade, never break
                logger.warning("anchor references via %s failed: %s", provider.name, exc)
                return []

        groups = await asyncio.gather(*(fetch(p) for p in providers))
        # Deduplicate across providers; their reference lists overlap and disagree.
        merged: dict[str, Paper] = {}
        for paper in (p for group in groups for p in group):
            merged.setdefault(paper.canonical_id, paper)
        return list(merged.values())

    async def _select_prerequisites(
        self, topic: str, anchor: Paper, references: list[Paper], want: int
    ) -> list[tuple[Paper, str, str]]:
        # Send the most-cited references: a paper's bibliography runs to dozens of entries,
        # and the influential ones are where genuine prerequisites concentrate.
        ranked = sorted(references, key=lambda p: -(p.citation_count or 0))[:40]
        labels = {f"r{i + 1}": paper for i, paper in enumerate(ranked)}
        payload = [
            {
                "id": label,
                "title": paper.title,
                "year": paper.year,
                "abstract": paper.abstract,
            }
            for label, paper in labels.items()
        ]

        result = await self.client.structured(
            role="prerequisite_judgment",
            system=cp.PREREQ_SYSTEM_V1,
            user=cp.prereq_user(topic, anchor.title, payload, want),
            schema=cp.PREREQ_SCHEMA,
            prompt_version=cp.VERSION,
            max_tokens=10000,
        )
        self.context.record(result)

        chosen = sorted(
            result.data.get("selected", []), key=lambda s: int(s.get("position", 0))
        )
        out: list[tuple[Paper, str, str]] = []
        for entry in chosen[:want]:
            paper = labels.get(str(entry.get("id")))
            if paper is None:
                logger.warning("prereq selection named unknown id %r", entry.get("id"))
                continue
            out.append((paper, entry["concept"], entry["why_needed"]))
        return out

    async def _followups(
        self, topic: str, anchor: Paper, want: int
    ) -> list[tuple[Paper, str, str]]:
        """Later work, chosen from papers that actually cite the anchor.

        An earlier version sorted the citing papers by citation count and pasted the same
        sentence under each — "builds on the anchor paper and is itself widely cited". A
        learner reviewing that output found the claim was usually false: batch
        normalization does not build on dropout, and adversarial examples do not build on
        conditional GANs. Citing a paper is not building on it, and popularity among the
        citers says nothing about which of them teach the reader anything.

        So the LLM selects, with the citation edge as the candidate filter rather than as
        the justification. With no LLM there is no honest way to make this call, and the
        stage is dropped rather than filled with boilerplate.
        """
        if want <= 0:
            return []
        providers = build_paper_providers(
            self.config, Capability.CITATIONS, cache=self.context.cache
        )
        if not providers:
            return []

        async def fetch(provider):
            try:
                return await provider.citations(
                    anchor, self.config.retrieval.expansion.citations_per_paper, topic
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("anchor citations via %s failed: %s", provider.name, exc)
                return []

        groups = await asyncio.gather(*(fetch(p) for p in providers))
        merged: dict[str, Paper] = {}
        for paper in (p for group in groups for p in group):
            merged.setdefault(paper.canonical_id, paper)
        if not merged:
            return []

        # Citation count decides who is SHOWN to the model, not who is selected.
        ranked = sorted(merged.values(), key=lambda p: -(p.citation_count or 0))[:30]
        labels = {f"c{i + 1}": paper for i, paper in enumerate(ranked)}
        payload = [
            {"id": label, "title": p.title, "year": p.year, "abstract": p.abstract}
            for label, p in labels.items()
        ]

        try:
            result = await self.client.structured(
                role="pedagogical_rerank",
                system=cp.FOLLOWUP_SYSTEM_V1,
                user=cp.followup_user(topic, anchor.title, payload, want),
                schema=cp.FOLLOWUP_SCHEMA,
                prompt_version=cp.VERSION,
                max_tokens=10000,
            )
        except LLMError as exc:
            self.context.notes.append(
                f"Follow-up work could not be selected ({exc}), so the path stops at the "
                f"anchor rather than listing papers that merely cite it."
            )
            return []
        self.context.record(result)

        chosen = sorted(
            result.data.get("selected", []), key=lambda s: int(s.get("position", 0))
        )
        out: list[tuple[Paper, str, str]] = []
        for entry in chosen[:want]:
            paper = labels.get(str(entry.get("id")))
            if paper is None:
                logger.warning("follow-up selection named unknown id %r", entry.get("id"))
                continue
            out.append((paper, entry["concept"], f"({entry['kind']}) {entry['why_after']}"))
        return out


# ======================================================================================
# rerank — retrieve as before, then let the LLM impose a teaching order
# ======================================================================================


class RerankStrategy(CurriculumStrategy):
    """Keep the existing retrieval, replace the ordering policy.

    This is the minimal intervention: the corpus is whatever search and citation expansion
    found, and the LLM only decides which of it forms a path. It cannot surface a paper
    retrieval missed — which is precisely the hypothesis being tested, since the structural
    pipeline's failure might be ranking rather than recall.
    """

    name = "rerank"

    async def plan(self, topic: str, budget: int) -> list[PlannedStep]:
        search_service = TopicSearchService(self.config, cache=self.context.cache)
        expansion_service = CitationExpansionService(self.config, cache=self.context.cache)

        search = await search_service.search(
            topic, limit=self.config.retrieval.max_candidates, standalone=False
        )
        if not search.papers:
            return []
        corpus = await expansion_service.expand(topic, search.papers)
        analysis = analyze(
            list(corpus.papers),
            corpus.edge_pairs(),
            {pid: p.year for pid, p in corpus.papers.items()},
            self.config.retrieval.graph,
        )

        shortlist = self._shortlist(corpus, analysis, limit=40)
        labels = {f"n{i + 1}": pid for i, pid in enumerate(shortlist)}
        payload = [
            {
                "id": label,
                "title": corpus.papers[pid].title,
                "year": corpus.papers[pid].year,
                "abstract": corpus.papers[pid].abstract,
                "facts": (
                    f"cited by {corpus.co_citations.get(pid, 0)} of this topic's papers; "
                    f"{'reached by following citations' if pid in corpus.discovered_ids else 'matched the query text'}"
                ),
            }
            for label, pid in labels.items()
        ]

        result = await self.client.structured(
            role="pedagogical_rerank",
            system=cp.RERANK_SYSTEM_V1,
            user=cp.rerank_user(topic, payload, budget),
            schema=cp.RERANK_SCHEMA,
            prompt_version=cp.VERSION,
            max_tokens=12000,
        )
        self.context.record(result)
        if reason := result.data.get("rejected_reason"):
            self.context.notes.append(f"Reranker: {reason}")

        sequence = sorted(
            result.data.get("sequence", []), key=lambda s: int(s.get("position", 0))
        )
        steps: list[PlannedStep] = []
        for entry in sequence[:budget]:
            paper_id = labels.get(str(entry.get("id")))
            if paper_id is None:
                logger.warning("rerank named unknown id %r", entry.get("id"))
                continue
            steps.append(
                PlannedStep(
                    paper=corpus.papers[paper_id],
                    concept=entry["concept"],
                    stage=entry.get("stage", STAGE_FOLLOWUP),
                    why_here=entry["why_here"],
                    position=len(steps),
                )
            )
        return steps

    def _shortlist(self, corpus, analysis, limit: int) -> list[str]:
        """Candidates worth showing the reranker: the topic's surface plus its ancestry.

        Balanced deliberately. Handing over the top-N by centrality alone would present the
        reranker with the same biased pool that produced the problem, and it cannot promote
        a foundational paper it was never shown.
        """
        seeds = corpus.seed_ids[: limit // 2]
        ancestors = sorted(
            (pid for pid in corpus.papers if corpus.co_citations.get(pid, 0) >= 2),
            key=lambda pid: (
                -corpus.co_citations.get(pid, 0),
                -analysis.age_rescaled.get(pid, 0.0),
                pid,
            ),
        )
        out: list[str] = []
        for pid in [*ancestors, *seeds]:
            if pid not in out:
                out.append(pid)
            if len(out) >= limit:
                break
        return out


# ======================================================================================
# hybrid — syllabus for pedagogy, anchor's citation graph for what did not resolve
# ======================================================================================


class HybridStrategy(CurriculumStrategy):
    """Plan with `syllabus`; when too much of it fails to resolve, fall through to `anchor`.

    The two strategies fail in different places — `syllabus` on recall of exact titles,
    `anchor` on topics with no single definitive paper — so the combination covers more
    than either. The threshold is on *coverage*, not on errors: a syllabus that resolved
    two of eight steps did not fail loudly, it just quietly stopped being a path.
    """

    name = "hybrid"
    MIN_STEPS = 4

    async def plan(self, topic: str, budget: int) -> list[PlannedStep]:
        syllabus = SyllabusStrategy(self.context)
        steps = await syllabus.plan(topic, budget)
        has_anchor = any(s.stage == STAGE_ANCHOR for s in steps)

        if len(steps) >= self.MIN_STEPS and has_anchor:
            return steps

        self.context.notes.append(
            f"Syllabus planning yielded {len(steps)} resolvable step(s)"
            f"{' with no anchor' if not has_anchor else ''}; rebuilt from the anchor "
            f"paper's own citation graph instead."
        )
        fallback = await AnchorFirstStrategy(self.context).plan(topic, budget)
        return fallback or steps


STRATEGIES: dict[str, type[CurriculumStrategy]] = {
    "syllabus": SyllabusStrategy,
    "anchor": AnchorFirstStrategy,
    "rerank": RerankStrategy,
    "hybrid": HybridStrategy,
}


def build_strategy(name: str, context: BuildContext) -> CurriculumStrategy:
    try:
        return STRATEGIES[name](context)
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}. Known: {', '.join(sorted(STRATEGIES))}"
        ) from None


# ======================================================================================
# assembly
# ======================================================================================


def _assemble(path: LearningPath, steps: list[PlannedStep], context: BuildContext) -> None:
    """Turn planned steps into a `LearningPath`.

    Levels come from the planned order rather than from a DAG here: the sequence IS the
    judgment being made, and re-deriving it from citation edges would discard it. The
    citation constraint is still checked — a step placed before something it cites is
    reported, because that is a real contradiction between the plan and the record.
    """
    provenance = Provenance(
        asserted_by="+".join(sorted(context.models_used)) or "llm",
        model="+".join(sorted(context.models_used)) or None,
        prompt_version=cp.VERSION,
    )

    for index, step in enumerate(steps):
        paper = step.paper
        previous = steps[index - 1].paper.title if index else None
        path.steps.append(
            PathStep(
                order=index,
                level=index,
                paper=paper,
                role=_STAGE_ROLE.get(step.stage, PaperRole.UNCLASSIFIED),
                signals=PaperSignals(
                    co_citations=0,
                    pagerank=0.0,
                    age_rescaled_pagerank=0.0,
                    lexical_score=0.0,
                    in_degree=0,
                    out_degree=0,
                    discovered_by_expansion=step.stage == STAGE_PREREQUISITE,
                ),
                explanation=Explanation(
                    why_it_matters=step.why_here,
                    what_it_assumes=(
                        f"Assumes step {index}: {previous}."
                        if previous
                        else "Nothing earlier in this path — this is the entry point."
                    ),
                    what_it_teaches=step.concept,
                    why_for_you=_why_for_you(step, index, len(steps)),
                    source=ExplanationSource.LLM,
                    provenance=provenance,
                ),
                subtopic_id=step.stage,
                prerequisite_ids=(steps[index - 1].paper.canonical_id,) if index else (),
            )
        )
        if index:
            path.edges.append(
                PrerequisiteEdge(
                    prerequisite_id=steps[index - 1].paper.canonical_id,
                    dependent_id=paper.canonical_id,
                    source=EdgeSource.LLM_JUDGMENT,
                    provenance=provenance,
                    reason=step.why_here,
                )
            )

    counts = {stage: sum(1 for s in steps if s.stage == stage) for stage in
              (STAGE_PREREQUISITE, STAGE_ANCHOR, STAGE_FOLLOWUP)}
    path.notes.append(
        f"{len(steps)} steps: {counts[STAGE_PREREQUISITE]} prerequisite, "
        f"{counts[STAGE_ANCHOR]} anchor, {counts[STAGE_FOLLOWUP]} follow-up. "
        f"Planned by {', '.join(sorted(context.models_used)) or 'no model'}."
    )
    if not counts[STAGE_ANCHOR]:
        path.degraded = True
        # Deliberately the FIRST note. A path with no anchor is not a short path, it is a
        # path about the neighbourhood of the topic that never arrives at the topic — and
        # a learner reading prerequisites to nothing has no way to notice that from the
        # steps alone.
        path.notes.insert(
            0,
            "INCOMPLETE: this path never reaches the topic itself. The paper that would "
            "be the destination could not be identified or could not be found, so what "
            "follows is background only. Treat it as a partial result.",
        )

    _check_chronology(path, steps)
    _score_confidence(path, steps, context)


def _score_confidence(path: LearningPath, steps: list[PlannedStep], context: BuildContext) -> None:
    """Rate the system's belief in its own path, and say what moved the number.

    Deliberately built from things that are *checkable without knowing the right answer* —
    did we reach the topic, did the plan survive lookup, is the shape a path — rather than
    from anything the model asserts about its own quality. A model's stated confidence is
    the least reliable signal available here; whether its plan resolved to real papers is
    among the most reliable.
    """
    reasons: list[str] = []
    score = 1.0

    has_anchor = any(s.stage == STAGE_ANCHOR for s in steps)
    if not has_anchor:
        score -= 0.45
        reasons.append("never reaches the topic itself (no anchor paper)")

    prerequisites = sum(1 for s in steps if s.stage == STAGE_PREREQUISITE)
    if not prerequisites:
        score -= 0.20
        reasons.append("no prerequisites — this is a reading list, not a path")

    # Steps whose rationale had to be discarded because the lookup returned a near-miss.
    approximate = sum(1 for s in steps if s.concept.startswith("Closest available match"))
    if approximate:
        penalty = min(0.30, 0.10 * approximate)
        score -= penalty
        reasons.append(f"{approximate} step(s) are an approximate match for what was planned")

    dropped = sum(1 for note in context.notes if "could not be found" in note)
    if dropped:
        score -= 0.15
        reasons.append("part of the plan could not be found and was dropped")

    if len(steps) < 5:
        score -= 0.15
        reasons.append(f"only {len(steps)} steps survived")

    if any("backwards in time" in note for note in path.notes):
        score -= 0.10
        reasons.append("the sequence is largely reverse-chronological")

    path.confidence = max(0.0, min(1.0, score))
    path.confidence_reasons = reasons
    if not reasons:
        path.confidence_reasons = ["reaches the topic, with prerequisites, all steps verified"]


def _check_chronology(path: LearningPath, steps: list[PlannedStep]) -> None:
    """Report steps that are out of publication order.

    Not an error on its own — a later paper can genuinely be the better entry point to an
    earlier idea — but a sequence that is mostly backwards in time is usually a plan that
    ignored dependency, so it is surfaced rather than hidden.
    """
    years = [(i, s.paper.year) for i, s in enumerate(steps) if s.paper.year]
    inversions = sum(
        1
        for (i, yi), (j, yj) in zip(years, years[1:], strict=False)
        if yj < yi - 2
    )
    if inversions and inversions >= len(years) // 3:
        path.notes.append(
            f"{inversions} of {max(1, len(years) - 1)} consecutive steps go backwards in "
            f"time by more than two years — the sequence may be ordered by topic rather "
            f"than by dependency."
        )


def _why_for_you(step: PlannedStep, index: int, total: int) -> str:
    if step.stage == STAGE_ANCHOR:
        return (
            f"This is what you came for. It is step {index + 1} of {total} because the "
            f"{index} paper(s) before it supply what it assumes you already know."
        )
    if step.stage == STAGE_PREREQUISITE:
        return (
            f"Read this before the main paper. It is here at position {index + 1} because "
            f"the topic builds directly on it, not because it matches your search words."
        )
    return (
        f"Read after the main paper. It extends or reassesses the core idea, and will not "
        f"make much sense before it."
    )


def _layers(config: Config) -> list[str]:
    layers = config.retrieval.layers
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
