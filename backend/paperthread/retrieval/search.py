"""Topic search — Stage 1 of the recommendation pipeline.

Scope of the POC: a user types a topic, providers fan out, results reconcile onto
canonical papers, and a fused ranking comes back. This is candidate *retrieval* only.

What is deliberately NOT here yet, in pipeline order:
  Stage 0  topic decomposition into subtopics (needs L4)
  Stage 2  citation-graph expansion — backward co-citation for ancestors, forward for
           later work. This is where the actual product lives: foundational papers are
           found because many candidates cite them in common, not because they match the
           query text.
  Stage 3  age-rescaled PageRank + co-citation scoring on the induced subgraph
  Stage 4  LLM role assignment, prerequisite edges, explanations
  Stage 5  DAG ordering under the citation constraint (if A cites B, B precedes A)
  Stage 6  personalization against reading history
See docs/RETRIEVAL_NOTES.md.

Layer discipline (D12): this module runs at L0 — no model weights, no LLM, network only
for the providers themselves. Every higher layer refines this result; none replaces it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from ..config import Config
from ..domain.models import RankedPaper, SearchHit
from ..providers.base import BasePaperProvider, Capability, ProviderError
from ..providers.http_cache import HTTPCache
from ..providers.registry import build_paper_providers
from .fusion import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


@dataclass
class ProviderOutcome:
    """Per-provider result, reported to the caller.

    Failures are surfaced rather than swallowed: a degraded result must be visibly
    degraded (D12), and "0 results because a provider was rate limited" must never be
    presented as "0 results because nothing matched".
    """

    provider: str
    ok: bool
    count: int = 0
    error: str | None = None


@dataclass
class SearchResult:
    topic: str
    papers: list[RankedPaper]
    providers: list[ProviderOutcome]
    layers_used: list[str] = field(default_factory=list)
    degraded: bool = False
    notes: list[str] = field(default_factory=list)


class TopicSearchService:
    def __init__(self, config: Config, cache: HTTPCache | None = None) -> None:
        self.config = config
        self.cache = cache or HTTPCache(
            config.provider_cache.cache_dir, enabled=config.provider_cache.enabled
        )

    async def search(
        self, topic: str, limit: int | None = None, standalone: bool = True
    ) -> SearchResult:
        """Stage 1.

        `standalone` says whether this result is going straight to a user. When it is a
        stage of the path pipeline it is not, and the "L0 only" caveat below would be a
        lie — expansion and ordering run immediately afterwards. Callers that embed this
        stage own the honesty reporting for the whole pipeline.
        """
        topic = topic.strip()
        if not topic:
            raise ValueError("topic must not be empty")

        providers = build_paper_providers(self.config, Capability.SEARCH, cache=self.cache)
        if not providers:
            return SearchResult(
                topic=topic,
                papers=[],
                providers=[],
                degraded=True,
                notes=["No paper provider with the 'search' capability is enabled in config."],
            )

        per_provider = self.config.retrieval.candidates_per_provider
        results = await asyncio.gather(
            *(self._search_one(p, topic, per_provider) for p in providers)
        )

        hit_lists: list[list[SearchHit]] = []
        outcomes: list[ProviderOutcome] = []
        for provider, (hits, error) in zip(providers, results, strict=True):
            if error is None:
                hit_lists.append(hits)
                outcomes.append(ProviderOutcome(provider.name, ok=True, count=len(hits)))
            else:
                outcomes.append(ProviderOutcome(provider.name, ok=False, error=error))

        fused = reciprocal_rank_fusion(hit_lists, k=self.config.retrieval.rrf_k)
        capped = fused[: (limit or self.config.retrieval.max_candidates)]

        notes: list[str] = []
        failed = [o.provider for o in outcomes if not o.ok]
        if failed:
            notes.append(f"Degraded: provider(s) failed — {', '.join(failed)}.")

        # Papers with no abstract are kept, never filtered — abstract coverage is worst
        # for older papers, which are exactly the foundational ones (PROVIDER_NOTES C2).
        missing_abstract = sum(1 for r in capped if not r.paper.has_abstract)
        if missing_abstract:
            notes.append(
                f"{missing_abstract} of {len(capped)} results have no abstract; retained deliberately."
            )

        layers = self._active_layers()
        if standalone and layers == ["lexical"]:
            notes.append(
                "L0 only: provider search + RRF fusion. Citation-graph expansion, "
                "reranking and LLM ordering are not active."
            )

        return SearchResult(
            topic=topic,
            papers=capped,
            providers=outcomes,
            layers_used=layers,
            degraded=bool(failed),
            notes=notes,
        )

    async def _search_one(
        self, provider: BasePaperProvider, topic: str, limit: int
    ) -> tuple[list[SearchHit], str | None]:
        """One provider's search. Never raises — a failure degrades, it does not break."""
        try:
            return await provider.search(topic, limit), None
        except ProviderError as exc:
            logger.warning("provider %s failed: %s", provider.name, exc)
            return [], str(exc)
        except Exception as exc:  # noqa: BLE001 - an adapter bug must not take down search
            logger.exception("provider %s raised unexpectedly", provider.name)
            return [], f"[{provider.name}] unexpected error: {exc}"

    def _active_layers(self) -> list[str]:
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
