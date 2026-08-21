"""Stage 2 — citation-graph expansion.

**The foundational papers of a topic are almost never the best keyword matches for it.**
Search "diffusion models" and you get papers that use the phrase; you do not get the work
underneath, because the vocabulary was invented afterwards. Those papers are found by
asking what the candidates *cite in common*: if 20 of 25 candidates cite the same paper,
it is foundational to the topic, and no model had to have an opinion about it.

That is the highest-value signal in the system, and it runs at L0 — no model weights, no
LLM, network only for the providers themselves.

Three passes, in cost order:

1. **Backward from candidates** — references of the top candidates. Yields ancestors, and
   for free, the candidate→candidate edges that make the subgraph dense enough to rank.
2. **Backward from ancestors** — so ancestors can be ordered against *each other*. Without
   this pass every ancestor lands at depth 0 in an undifferentiated heap, which is a list,
   not a path.
3. **Forward, topic-constrained** — later developments, extensions, critiques (§4).

Identity is the correctness risk here, not the network. References arrive as fresh
provider records with no knowledge of the candidate set, so the same work will arrive as
an arXiv preprint from one path and an OpenAlex record from another. Left unreconciled,
one ancestor becomes three nodes and its co-citation count — the entire ancestor signal —
splits three ways (PROVIDER_NOTES C4). Everything is therefore deduplicated as one
population, and every edge is remapped onto survivors.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..config import Config
from ..domain.identity import deduplicate
from ..domain.models import Paper, RankedPaper
from ..providers.base import BasePaperProvider, Capability, ProviderError
from ..providers.http_cache import HTTPCache
from ..providers.registry import build_paper_providers

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CitationEdge:
    """`citing` cites `cited`. Provenance is recorded because provider citation graphs
    disagree with each other and none of them is complete (PROVIDER_NOTES C7)."""

    citing_id: str
    cited_id: str
    provider: str


@dataclass
class ExpansionOutcome:
    provider: str
    ok: bool
    requests: int = 0
    papers_added: int = 0
    error: str | None = None


@dataclass
class ExpandedCorpus:
    """The induced subgraph for one topic: nodes, edges, and how each node got here."""

    papers: dict[str, Paper] = field(default_factory=dict)
    edges: list[CitationEdge] = field(default_factory=list)
    # Stage 1 candidates, in fused order. These are the topic's "surface".
    seed_ids: list[str] = field(default_factory=list)
    # Reached only by following citations — keyword search could not have found them.
    discovered_ids: set[str] = field(default_factory=set)
    # canonical_id -> how many distinct seeds cite it. The ancestor signal.
    co_citations: dict[str, int] = field(default_factory=dict)
    outcomes: list[ExpansionOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ran: bool = False

    def edge_pairs(self) -> set[tuple[str, str]]:
        return {(edge.citing_id, edge.cited_id) for edge in self.edges}


class CitationExpansionService:
    """Stage 2. Degrades to a no-op when no provider offers the `citations` capability."""

    def __init__(
        self,
        config: Config,
        providers: list[BasePaperProvider] | None = None,
        cache: HTTPCache | None = None,
    ) -> None:
        self.config = config
        self.settings = config.retrieval.expansion
        self.cache = cache or HTTPCache(
            config.provider_cache.cache_dir, enabled=config.provider_cache.enabled
        )
        # Built once and reused: each adapter owns its own rate limiter, so rebuilding
        # per request would throttle nothing and get us blocked.
        self.providers = (
            providers
            if providers is not None
            else build_paper_providers(config, Capability.CITATIONS, cache=self.cache)
        )

    async def expand(self, topic: str, candidates: list[RankedPaper]) -> ExpandedCorpus:
        corpus = ExpandedCorpus(
            seed_ids=[c.paper.canonical_id for c in candidates],
            papers={c.paper.canonical_id: c.paper for c in candidates},
        )

        if not self.settings.enabled:
            corpus.notes.append("Citation-graph expansion is disabled in config.")
            return corpus
        if not candidates:
            return corpus
        if not self.providers:
            corpus.notes.append(
                "No provider with the 'citations' capability is enabled, so ancestors "
                "cannot be found. arXiv has no citation graph; enable OpenAlex or "
                "Semantic Scholar."
            )
            return corpus

        collector = _Collector(self.providers)
        seeds = [c.paper for c in candidates][: self.settings.seed_papers]

        # Pass 1 — backward from the candidates.
        await collector.gather_references(seeds, self.settings.references_per_paper)

        # Ancestors are decided on the RAW references before deduplication only to pick
        # who to expand next; the authoritative counts are recomputed after merging.
        ancestors = collector.provisional_ancestors(
            seed_ids={p.canonical_id for p in seeds},
            minimum=self.settings.min_co_citations,
            limit=self.settings.max_ancestors,
        )

        # Pass 2 — backward from the ancestors, so they can be ordered against each other.
        if self.settings.expand_ancestors and ancestors:
            await collector.gather_references(
                ancestors[: self.settings.ancestor_seeds], self.settings.references_per_paper
            )

        # Pass 3 — forward, constrained to the topic.
        if self.settings.forward_enabled and ancestors:
            await collector.gather_citations(
                ancestors[: self.settings.forward_seeds],
                self.settings.citations_per_paper,
                topic,
            )

        self._assemble(corpus, collector)
        return corpus

    def _assemble(self, corpus: ExpandedCorpus, collector: _Collector) -> None:
        """Reconcile everything onto canonical papers and remap the edges.

        Deduplication runs over candidates AND everything expansion found, as one
        population. Doing it per-pass would leave a preprint from pass 1 and its published
        version from pass 3 as separate nodes.
        """
        seeds_before = list(corpus.papers.values())
        originals = seeds_before + collector.discovered
        survivors = deduplicate(originals)

        by_alias: dict[str, Paper] = {}
        for paper in survivors:
            by_alias[paper.canonical_id] = paper
            for external in paper.external_ids:
                by_alias.setdefault(str(external), paper)

        def resolve(paper: Paper) -> Paper:
            if found := by_alias.get(paper.canonical_id):
                return found
            for external in paper.external_ids:
                if found := by_alias.get(str(external)):
                    return found
            return paper

        corpus.papers = {paper.canonical_id: paper for paper in survivors}
        corpus.seed_ids = _dedupe_preserving_order(
            resolve(paper).canonical_id for paper in seeds_before
        )
        seed_set = set(corpus.seed_ids)
        corpus.discovered_ids = set(corpus.papers) - seed_set

        # Remap edges onto survivors. A self-edge here is a preprint citing its own
        # published version after merging — dropping it is what keeps the DAG acyclic.
        seen: set[tuple[str, str, str]] = set()
        edges: list[CitationEdge] = []
        # Distinct SEEDS citing a paper, not distinct citations: two seeds that merged
        # into one work must count once, or a preprint/published pair double-votes.
        citers: dict[str, set[str]] = {}
        for citing, cited, provider in collector.edges:
            citing_id = resolve(citing).canonical_id
            cited_id = resolve(cited).canonical_id
            if citing_id == cited_id:
                continue
            key = (citing_id, cited_id, provider)
            if key in seen:
                continue
            seen.add(key)
            edges.append(CitationEdge(citing_id, cited_id, provider))
            if citing_id in seed_set:
                citers.setdefault(cited_id, set()).add(citing_id)

        corpus.edges = edges
        corpus.co_citations = {paper_id: len(who) for paper_id, who in citers.items()}
        corpus.outcomes = collector.outcomes()
        corpus.ran = True

        discovered = len(corpus.discovered_ids)
        ancestors = sum(
            1
            for paper_id in corpus.discovered_ids
            if corpus.co_citations.get(paper_id, 0) >= self.settings.min_co_citations
        )
        corpus.notes.append(
            f"Citation expansion added {discovered} papers and {len(edges)} edges; "
            f"{ancestors} are cited by {self.settings.min_co_citations}+ of the topic's "
            f"own candidates."
        )
        if failed := [o.provider for o in corpus.outcomes if not o.ok]:
            corpus.notes.append(
                f"Degraded: citation provider(s) failed — {', '.join(sorted(set(failed)))}."
            )


class _Collector:
    """Accumulates raw provider records and edges before any reconciliation.

    Kept deliberately dumb: it does not know about canonical identity, because identity
    can only be resolved once every pass has contributed its records.
    """

    def __init__(self, providers: list[BasePaperProvider]) -> None:
        self.providers = providers
        self.discovered: list[Paper] = []
        self.edges: list[tuple[Paper, Paper, str]] = []
        self._stats: dict[str, ExpansionOutcome] = {
            provider.name: ExpansionOutcome(provider.name, ok=True) for provider in providers
        }

    def outcomes(self) -> list[ExpansionOutcome]:
        return list(self._stats.values())

    async def gather_references(self, papers: list[Paper], limit: int) -> None:
        await self._gather(papers, limit, forward=False, query=None)

    async def gather_citations(self, papers: list[Paper], limit: int, query: str) -> None:
        await self._gather(papers, limit, forward=True, query=query)

    async def _gather(
        self, papers: list[Paper], limit: int, forward: bool, query: str | None
    ) -> None:
        """Fan out across papers AND providers.

        Every provider is asked about every paper: providers answering the same capability
        at once is the normal case (D11), and their citation graphs genuinely differ, so
        the union is more complete than any one of them.
        """
        tasks = [
            self._fetch(provider, paper, limit, forward, query)
            for paper in papers
            for provider in self.providers
        ]
        for source, found, error, provider_name in await asyncio.gather(*tasks):
            stats = self._stats[provider_name]
            stats.requests += 1
            if error is not None:
                stats.ok = False
                stats.error = error
                continue
            stats.papers_added += len(found)
            for other in found:
                self.discovered.append(other)
                # Direction matters and is the easiest thing to invert: a REFERENCE of
                # `source` is cited BY it; a CITATION of `source` cites it.
                if forward:
                    self.edges.append((other, source, provider_name))
                else:
                    self.edges.append((source, other, provider_name))

    async def _fetch(
        self,
        provider: BasePaperProvider,
        paper: Paper,
        limit: int,
        forward: bool,
        query: str | None,
    ) -> tuple[Paper, list[Paper], str | None, str]:
        """One provider request. Never raises — a failure degrades the graph, not the run."""
        try:
            if forward:
                found = await provider.citations(paper, limit, query)  # type: ignore[attr-defined]
            else:
                found = await provider.references(paper, limit)  # type: ignore[attr-defined]
            return paper, found, None, provider.name
        except ProviderError as exc:
            logger.warning("expansion: %s failed on %s: %s", provider.name, paper.canonical_id, exc)
            return paper, [], str(exc), provider.name
        except Exception as exc:  # noqa: BLE001 - an adapter bug must not take down the path
            logger.exception("expansion: %s raised on %s", provider.name, paper.canonical_id)
            return paper, [], f"[{provider.name}] unexpected error: {exc}", provider.name

    def provisional_ancestors(
        self, seed_ids: set[str], minimum: int, limit: int
    ) -> list[Paper]:
        """Most co-cited references, used only to choose what to expand next.

        Counts here are pre-deduplication and therefore conservative — a work split across
        a preprint and a published record undercounts. That is acceptable for choosing
        *who to expand*; it is not acceptable for the numbers shown to a user, which are
        recomputed in `_assemble` after merging.
        """
        counts: dict[str, set[str]] = {}
        examples: dict[str, Paper] = {}
        for citing, cited, _ in self.edges:
            if citing.canonical_id not in seed_ids:
                continue
            counts.setdefault(cited.canonical_id, set()).add(citing.canonical_id)
            examples.setdefault(cited.canonical_id, cited)

        ranked = sorted(
            (
                (len(citers), paper_id)
                for paper_id, citers in counts.items()
                if len(citers) >= minimum and paper_id not in seed_ids
            ),
            # Ties broken by id so the choice is deterministic run to run, which §6's
            # incremental path updates depend on.
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [examples[paper_id] for _, paper_id in ranked[:limit]]


def _dedupe_preserving_order(values) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
