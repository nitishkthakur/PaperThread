"""arXiv adapter.

Capabilities: search, fulltext. **No citation graph** — arXiv is the only provider that
hands over complete PDFs and LaTeX source for free with no open-access gate, which makes
it the natural backend for D8's full-text phase, but it exposes no references or citations
at all (PROVIDER_NOTES C3).

Rate limit: arXiv asks for roughly 1 request per 3 seconds. Set in config, not here.
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET

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

_API_URL = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_LICENSE = "arXiv non-exclusive license (varies per paper)"

logger = logging.getLogger(__name__)

# arXiv returns 429 with no Retry-After, so the schedule is ours to choose. Generous,
# because the alternative is an empty path.
_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 4
_BASE_BACKOFF = 5.0


class ArxivProvider(BasePaperProvider):
    name = "arxiv"

    async def find_by_title(self, title: str, limit: int = 5) -> list[Paper]:
        """Exact-ish lookup by title, using arXiv's title field.

        This is the difference between finding a paper and finding papers about it.
        Measured: `all:A Simple Framework for Contrastive Learning of Visual
        Representations` returns, as its top hit, *"Observation of the rare B⁰ₛ→μ⁺μ⁻
        decay"* — a particle-physics paper — because only the first token binds to the
        `all:` field and the rest becomes unfielded noise that the relevance ranker then
        does its best with. The same string as `ti:"…"` returns the paper itself, alone.

        That one malformed query is why physics and chemistry papers appeared in ML
        learning paths, and why every post-2017 arXiv-native paper failed to resolve once
        OpenAlex was rate-limited and arXiv was the only provider left.
        """
        hits = await self._query({"search_query": f'ti:"{title}"', "start": 0,
                                  "max_results": limit})
        return [hit.paper for hit in hits]

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        params = {
            # Quoted so the phrase binds to the field as a unit. Unquoted, `all:` applies
            # to the first token only.
            "search_query": f'all:"{query}"',
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        return await self._query(params)

    async def _query(self, params: dict) -> list[SearchHit]:
        # arXiv asks for one request per three seconds, so a cache hit here is worth more
        # than anywhere else in the system. Checked before throttling, deliberately.
        raw = self.cache_get(_API_URL, params)
        fetched = raw is None
        if fetched:
            raw = await self._fetch(params)

        try:
            # Atom, not JSON — the cache stores the raw document so a parser change does
            # not require refetching under the rate limit.
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise ProviderError(self.name, f"malformed Atom response: {exc}") from exc

        hits: list[SearchHit] = []
        for rank, entry in enumerate(root.findall(f"{_ATOM}entry"), start=1):
            paper = self._parse_entry(entry)
            if paper is not None:
                hits.append(SearchHit(paper=paper, rank=rank, provider=self.name))

        # An EMPTY result is never cached, and that asymmetry is deliberate. Under load
        # arXiv returns HTTP 200 with an empty feed rather than an error, so caching zero
        # results makes a transient throttle permanent: 17 such entries were written during
        # a throttled sweep, and every later run then "found" nothing for those titles
        # instantly and without touching the network. The cost of not caching emptiness is
        # one wasted request for a genuinely missing paper; the cost of caching it is a
        # silent, sticky wrong answer.
        if fetched and hits:
            self.cache_put(_API_URL, params, raw)
        return hits

    async def _fetch(self, params: dict) -> str:
        """One arXiv request, retried through throttling.

        arXiv 429s under sustained load even at its documented 1-request-per-3-seconds,
        and it has no `Retry-After` header. Without a retry here a single throttle wipes
        out a whole learning path — which is exactly what happened once OpenAlex's metered
        quota ran out and arXiv became the only provider left: half the topics returned
        zero steps, not because the papers were missing but because the lookups were
        refused. A provider being the last one standing is precisely when it must not fail
        on the first refusal.
        """
        last: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            await self._throttle()
            try:
                # follow_redirects: export.arxiv.org 301s http -> https.
                async with httpx.AsyncClient(
                    timeout=self.config.timeout_seconds, follow_redirects=True
                ) as client:
                    response = await client.get(_API_URL, params=params)
                    if response.status_code in _RETRYABLE and attempt < _MAX_RETRIES:
                        delay = _BASE_BACKOFF * (2**attempt)
                        logger.warning(
                            "arxiv %s; backing off %.0fs (attempt %d/%d)",
                            response.status_code, delay, attempt + 1, _MAX_RETRIES,
                        )
                        await asyncio.sleep(delay)
                        continue
                    response.raise_for_status()
                    return response.text
            except httpx.HTTPError as exc:
                last = exc
                if attempt >= _MAX_RETRIES:
                    break
                await asyncio.sleep(_BASE_BACKOFF * (2**attempt))
        raise ProviderError(self.name, f"search failed after retries: {last}")

    def _parse_entry(self, entry: ET.Element) -> Paper | None:
        raw_id = _text(entry, f"{_ATOM}id")
        title = _text(entry, f"{_ATOM}title")
        if not raw_id or not title:
            return None

        external_ids: set[ExternalId] = {
            normalize_external_id(IdNamespace.ARXIV, raw_id),
            ExternalId(IdNamespace.URL, raw_id.strip()),
        }
        doi = _text(entry, f"{_ATOM}doi") or _text(entry, "{http://arxiv.org/schemas/atom}doi")
        if doi:
            external_ids.add(normalize_external_id(IdNamespace.DOI, doi))

        abstract = _text(entry, f"{_ATOM}summary")
        published = _text(entry, f"{_ATOM}published")
        year = int(published[:4]) if published and published[:4].isdigit() else None

        pdf_url = None
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                break

        authors = [
            name
            for author in entry.findall(f"{_ATOM}author")
            if (name := _text(author, f"{_ATOM}name"))
        ]

        provenance = FieldProvenance(provider=self.name, license=_LICENSE)
        paper = Paper(
            canonical_id="",
            title=_collapse(title),
            external_ids=external_ids,
            abstract=_collapse(abstract) if abstract else None,
            year=year,
            authors=authors,
            venue="arXiv",
            pdf_url=pdf_url,
            landing_url=raw_id.strip(),
            # arXiv always gives a real abstract; full text is available but not fetched
            # in v1 (D8 — metadata + abstract now, full text later).
            depth=ContentDepth.ABSTRACT if abstract else ContentDepth.METADATA,
            provenance={
                field: provenance
                for field in ("title", "abstract", "year", "authors", "pdf_url")
            },
        )
        paper.canonical_id = canonical_id_for(paper)
        return paper


def _text(element: ET.Element, path: str) -> str | None:
    found = element.find(path)
    return found.text if found is not None and found.text else None


def _collapse(value: str) -> str:
    return " ".join(value.split())
