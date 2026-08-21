"""OpenAlex adapter.

Capabilities: search, citations, id_resolve. Broadest coverage (all fields), CC0 licensed
and therefore safe to cache and redistribute — the only provider here for which that is
unambiguously true (PROVIDER_NOTES C9).

Two caveats drive this file:
  * Abstracts arrive as an INVERTED INDEX, not plaintext, for legal reasons inherited from
    Microsoft Academic Graph. They must be reconstructed (C1).
  * Abstract coverage is partial and worst for older papers — which are exactly the
    foundational ones we exist to surface. Papers without abstracts are RETAINED (C2).
"""

from __future__ import annotations

import asyncio
import logging
import re

import httpx

from ...domain.identity import canonical_id_for, normalize_external_id
from ...domain.models import (
    ContentDepth,
    ExternalId,
    FieldProvenance,
    IdNamespace,
    Paper,
    SearchHit,
)
from ..base import BasePaperProvider, ProviderError

logger = logging.getLogger(__name__)

_API_URL = "https://api.openalex.org/works"
_MAX_429_RETRIES = 4


# Beyond this, a 429 is a spent quota rather than a burst, and waiting is not an option a
# request can take. Retrying is then pure waste — and worse than waste, because each retry
# is itself a metered call.
_MAX_SENSIBLE_WAIT_SECONDS = 60.0


def _retry_after(response, attempt: int) -> float | None:
    """Seconds to wait, or None when the wait is too long to be worth attempting.

    OpenAlex became a metered API in 2026 and now returns a credit-exhaustion 429 with
    `Retry-After` measured in **hours** (observed: 48806 seconds). Treating that as a
    transient burst — capping the wait at 30s and retrying four times — burns four more
    metered calls against a quota that has already run out, and then reports a generic
    failure. Distinguishing the two cases is the difference between "slow down" and "stop".
    """
    header = response.headers.get("Retry-After")
    if header:
        try:
            wait = float(header)
        except ValueError:
            wait = 0.0
        if wait > _MAX_SENSIBLE_WAIT_SECONDS:
            return None
        if wait > 0:
            return wait
    return min(30.0, 2.0 * (2**attempt))
_LICENSE = "CC0"
# `select` is free — widening it costs nothing and supplies the fields needed to reject
# non-papers (type, language, is_paratext, is_retracted) and to verify prerequisite edges
# without a second call (referenced_works).
_FIELDS = (
    "id,doi,title,display_name,publication_year,cited_by_count,"
    "abstract_inverted_index,authorships,primary_location,best_oa_location,ids,"
    "type,language,is_paratext,is_retracted,referenced_works_count,primary_topic"
)


class OpenAlexProvider(BasePaperProvider):
    name = "openalex"

    def _params(self, extra: dict[str, object]) -> dict[str, object]:
        params: dict[str, object] = {"select": _FIELDS, **extra}
        # Identifying yourself puts requests in the faster, more reliable "polite pool".
        if mailto := self.config.options.get("mailto"):
            params["mailto"] = mailto
        # A free key raises the daily allowance roughly tenfold. Absent, requests still
        # work on the anonymous tier — just with far less credit (PROVIDER_NOTES C12).
        if key := self.config.api_key:
            params["api_key"] = key
        return params

    async def _get(self, params: dict[str, object]) -> dict:
        # Checked before throttling: a cache hit must not pay the rate limit.
        if (cached := self.cache_get(_API_URL, params)) is not None:
            return cached

        # A 429 here used to degrade the whole pipeline to "arXiv only" — and arXiv's
        # relevance search cannot do exact-title lookup, so resolution then matched papers
        # to the wrong work. Backing off is much cheaper than that failure.
        payload = None
        for attempt in range(_MAX_429_RETRIES + 1):
            await self._throttle()
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                    response = await client.get(_API_URL, params=params)
                    if response.status_code == 429:
                        delay = (
                            _retry_after(response, attempt)
                            if attempt < _MAX_429_RETRIES
                            else None
                        )
                        if delay is None:
                            remaining = response.headers.get("x-ratelimit-remaining-usd")
                            reset = response.headers.get("x-ratelimit-reset", "?")
                            raise ProviderError(
                                self.name,
                                f"daily quota exhausted (remaining ${remaining}, resets in "
                                f"{reset}s). OpenAlex is metered: searches cost credits, "
                                f"lookups by ID are free. Set OPENALEX_API_KEY for a "
                                f"larger allowance.",
                            )
                        logger.warning(
                            "openalex rate limited; retrying in %.1fs (attempt %d/%d)",
                            delay, attempt + 1, _MAX_429_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    break
            except httpx.HTTPError as exc:
                raise ProviderError(self.name, f"request failed: {exc}") from exc
            except ValueError as exc:
                raise ProviderError(self.name, f"malformed JSON: {exc}") from exc

        if payload is None:
            raise ProviderError(self.name, "rate limited (HTTP 429) after retries")

        self.cache_put(_API_URL, params, payload)
        return payload

    async def find_by_title(self, title: str, limit: int = 5) -> list[Paper]:
        """Title-field lookup. Costs the same as a search, so try free providers first."""
        payload = await self._get(
            self._params(
                {
                    "filter": f"title.search:{sanitize_query(title)}",
                    "per_page": min(limit, 50),
                    "sort": "cited_by_count:desc",
                }
            )
        )
        return [p for w in payload.get("results", []) if (p := self._parse_work(w))]

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = await self._get(
            self._params({"search": sanitize_query(query), "per_page": min(limit, 200)})
        )
        hits: list[SearchHit] = []
        for rank, work in enumerate(payload.get("results", []), start=1):
            paper = self._parse_work(work)
            if paper is not None:
                hits.append(SearchHit(paper=paper, rank=rank, provider=self.name))
        return hits

    async def references(self, paper: Paper, limit: int) -> list[Paper]:
        """Backward expansion — the papers this one cites.

        This is how foundational work is reached: papers cited in common by many
        candidates are the topic's ancestors, and keyword search structurally cannot find
        them because they predate the topic's modern vocabulary.
        """
        openalex_id = paper.id_for(IdNamespace.OPENALEX)
        if not openalex_id:
            return []
        payload = await self._get(
            self._params({"filter": f"cited_by:{openalex_id}", "per_page": min(limit, 200)})
        )
        return [p for w in payload.get("results", []) if (p := self._parse_work(w))]

    async def citations(self, paper: Paper, limit: int, query: str | None = None) -> list[Paper]:
        """Forward expansion — later developments, surveys, critiques.

        `query` is applied as a full-text filter alongside `cites:`. Without it the forward
        set of a foundational paper is unusable: "Attention Is All You Need" has six
        figures' worth of citing papers spanning genomics, weather, and audio, and none of
        that belongs in a reading path about the topic the user asked for.
        """
        openalex_id = paper.id_for(IdNamespace.OPENALEX)
        if not openalex_id:
            return []
        params: dict[str, object] = {
            "filter": f"cites:{openalex_id}",
            "per_page": min(limit, 200),
            # Most-cited first: among on-topic descendants, the influential ones are the
            # extensions and critiques a curriculum should actually include.
            "sort": "cited_by_count:desc",
        }
        if query:
            params["search"] = query
        payload = await self._get(self._params(params))
        return [p for w in payload.get("results", []) if (p := self._parse_work(w))]

    def _parse_work(self, work: dict) -> Paper | None:
        title = work.get("display_name") or work.get("title")
        raw_id = work.get("id")
        if not title or not raw_id:
            return None

        external_ids: set[ExternalId] = {normalize_external_id(IdNamespace.OPENALEX, raw_id)}
        ids = work.get("ids") or {}
        if doi := (work.get("doi") or ids.get("doi")):
            external_ids.add(normalize_external_id(IdNamespace.DOI, doi))
        if pmid := ids.get("pmid"):
            external_ids.add(ExternalId(IdNamespace.PMID, str(pmid).rsplit("/", 1)[-1]))

        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        authors = [
            name
            for authorship in (work.get("authorships") or [])
            if (name := (authorship.get("author") or {}).get("display_name"))
        ]

        best_oa = work.get("best_oa_location") or {}
        primary = work.get("primary_location") or {}
        venue = ((primary.get("source") or {}) or {}).get("display_name")

        provenance = FieldProvenance(provider=self.name, license=_LICENSE)
        paper = Paper(
            canonical_id="",
            title=title,
            external_ids=external_ids,
            abstract=abstract,
            year=work.get("publication_year"),
            authors=authors,
            venue=venue,
            citation_count=work.get("cited_by_count"),
            # PDF only when the work is open access — unlike arXiv, this is not guaranteed.
            pdf_url=best_oa.get("pdf_url"),
            landing_url=primary.get("landing_page_url") or raw_id,
            depth=ContentDepth.ABSTRACT if abstract else ContentDepth.METADATA,
            provenance={
                field: provenance
                for field in ("title", "abstract", "year", "authors", "citation_count")
            },
        )
        paper.canonical_id = canonical_id_for(paper)
        return paper


_UNSAFE_SEARCH_CHARS = re.compile(r"[?!&|]+")


def sanitize_query(query: str) -> str:
    """Strip characters OpenAlex's `search` parameter rejects.

    Verified 2026-08-16: a query ending in `?` returns **HTTP 400**, while `:` and `!` are
    fine — the API treats some punctuation as query syntax. This matters far more than it
    looks, because paper titles are full of question marks ("How Does Batch Normalization
    Help Optimization?", "Do ImageNet Classifiers Generalize to ImageNet?"), and those are
    exactly the titles a curriculum planner names. Left unsanitized, the papers hardest to
    look up are the ones silently missing from every path.

    Stripping is safe: OpenAlex's search is token-based, so removing punctuation changes
    nothing about what matches.
    """
    cleaned = _UNSAFE_SEARCH_CHARS.sub(" ", query)
    return " ".join(cleaned.split())


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """Rebuild plaintext from OpenAlex's `abstract_inverted_index`.

    OpenAlex ships {word: [positions]} rather than text, inherited from MAG's legal
    constraints. Reconstruction is lossy on punctuation and whitespace — good enough for
    retrieval and for feeding an LLM, but it is NOT a verbatim abstract, and it should not
    be presented to a user as one.
    """
    if not inverted_index:
        return None

    positioned: list[tuple[int, str]] = [
        (position, word)
        for word, positions in inverted_index.items()
        for position in positions
    ]
    if not positioned:
        return None
    positioned.sort()
    return " ".join(word for _, word in positioned)
