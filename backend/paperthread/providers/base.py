"""Provider ports, split by CAPABILITY rather than by vendor.

No single provider offers citations + abstracts + free full text together, so the provider
that supplies a paper's citations is usually NOT the one that supplies its full text
(PROVIDER_NOTES §1.1). A design where one provider owns a paper end-to-end breaks the
moment D8's full-text phase arrives.

Each adapter implements only the capabilities it can, and DECLARES them. Callers ask the
registry rather than assuming feature parity.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Protocol, runtime_checkable

from ..config import PaperProviderConfig
from ..domain.models import Paper, SearchHit
from .http_cache import HTTPCache


class Capability(str, Enum):
    SEARCH = "search"
    CITATIONS = "citations"
    FULLTEXT = "fulltext"
    ID_RESOLVE = "id_resolve"


class ProviderError(RuntimeError):
    """A provider failed. Never fatal — the pipeline degrades to the providers that worked."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


@runtime_checkable
class PaperSearchPort(Protocol):
    """Find candidate papers for a free-text query."""

    name: str

    async def search(self, query: str, limit: int) -> list[SearchHit]: ...


@runtime_checkable
class CitationGraphPort(Protocol):
    """Fetch references (backward) and citations (forward) for a paper.

    Edges carry provenance because provider citation graphs disagree and none is complete
    (PROVIDER_NOTES C7).
    """

    name: str

    async def references(self, paper: Paper, limit: int) -> list[Paper]: ...

    async def citations(self, paper: Paper, limit: int, query: str | None = None) -> list[Paper]:
        """Papers citing this one, optionally narrowed to those matching `query`.

        The narrowing is not a convenience. A seminal paper's forward set is tens of
        thousands of papers spanning every field that ever borrowed the idea; unfiltered,
        forward expansion adds far more noise than signal. Providers that cannot filter
        MUST ignore `query` rather than fail — callers degrade, they do not branch on
        vendor (D9).
        """
        ...


class RateLimiter:
    """Per-provider rate limiting.

    Provider limits differ by orders of magnitude — arXiv asks for ~1 request per 3
    seconds while OpenAlex is generous (PROVIDER_NOTES C6) — so this is per-provider
    config, never a global constant.
    """

    def __init__(self, per_second: float) -> None:
        self._min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        async with self._lock:
            wait = self._last + self._min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


# Rate limiters are shared per provider NAME, process-wide.
#
# The limit belongs to the remote service, not to an object we happened to construct. Each
# adapter instance owning a private limiter means the effective request rate scales with
# how many services we built — and it does: search, expansion, and resolution each build
# their own adapters, and running several topics concurrently multiplies that again. That
# produced real HTTP 429s from OpenAlex, which degraded silently to "arXiv only" and let
# arXiv's weak title search resolve papers to the wrong work.
_LIMITERS: dict[str, RateLimiter] = {}


def limiter_for(provider: str, per_second: float) -> RateLimiter:
    limiter = _LIMITERS.get(provider)
    if limiter is None:
        limiter = _LIMITERS[provider] = RateLimiter(per_second)
    return limiter


class BasePaperProvider:
    """Shared adapter plumbing: identity, declared capabilities, rate limiting."""

    name: str = "base"

    def __init__(
        self, config: PaperProviderConfig, cache: HTTPCache | None = None
    ) -> None:
        self.config = config
        self.name = config.name
        self.capabilities = frozenset(Capability(c) for c in config.capabilities)
        self._limiter = limiter_for(config.name, config.rate_limit_per_sec)
        self.cache = cache

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    async def _throttle(self) -> None:
        """Wait for this provider's rate limit — but only when a call is actually made.

        Cache lookups must NOT throttle. Throttling a cache hit would make a warm run as
        slow as a cold one, which defeats the point of having the cache.
        """
        await self._limiter.acquire()

    async def find_by_title(self, title: str, limit: int = 5) -> list[Paper]:
        """Look up a KNOWN paper by its title. Distinct from `search`, deliberately.

        Searching for a topic and looking up a named paper are different operations, and
        conflating them was a real defect: relevance search for "Attention Is All You Need"
        returns papers *about* attention — "Do You Even Need Attention?", "GAN Vocoder:
        Multi-Resolution Discriminator Is All You Need" — and never the paper itself,
        because relevance ranking optimises for topical match, not for identity. A
        title-field query returns it as the first hit.

        Providers that offer no title-field query inherit this fallback to `search`, which
        is worse but never wrong to attempt.
        """
        return [hit.paper for hit in await self.search(title, limit)]  # type: ignore[attr-defined]

    def cache_get(self, url: str, params: dict | None = None):
        return self.cache.get(self.cache.key(self.name, url, params)) if self.cache else None

    def cache_put(self, url: str, params: dict | None, payload) -> None:
        if self.cache:
            self.cache.put(self.cache.key(self.name, url, params), payload)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        caps = ",".join(sorted(c.value for c in self.capabilities))
        return f"<{type(self).__name__} name={self.name} capabilities={caps}>"
