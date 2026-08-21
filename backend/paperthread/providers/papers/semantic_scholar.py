"""Semantic Scholar adapter.

Capabilities: search, citations. The best citation graph of the three, and the only
provider that exposes **per-edge citation intent** — `intents`, `contextsWithIntent`, and
`isInfluential` — for free.

That matters more than it looks. "A cites B" does not mean "B is a prerequisite for A":
the research behind S2's influential-citation feature found only **14.6% of citations are
'important'** rather than incidental. Getting intent labels from the API means we can
filter perfunctory citations without hosting a classifier (RETRIEVAL_NOTES, Finding 4).

Caveats: abstracts are missing for a meaningful fraction of papers due to licensing, PDFs
are available only for open-access works, and influential-citation detection requires the
citing paper's full text — so its coverage correlates with the open-access gap.
Unauthenticated rate limits are punishing; set SEMANTIC_SCHOLAR_API_KEY.
"""

from __future__ import annotations

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

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_LICENSE = "Semantic Scholar API terms (redistribution restricted)"
_PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,citationCount,"
    "influentialCitationCount,authors,openAccessPdf,url"
)


class SemanticScholarProvider(BasePaperProvider):
    name = "semantic_scholar"

    def _headers(self) -> dict[str, str]:
        key = self.config.api_key
        return {"x-api-key": key} if key else {}

    async def _get(self, path: str, params: dict[str, object]) -> dict:
        url = f"{_BASE_URL}{path}"
        # Checked before throttling: unauthenticated S2 limits are punishing, so a cache
        # hit must not queue behind the rate limiter.
        if (cached := self.cache_get(url, params)) is not None:
            return cached

        await self._throttle()
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.get(
                    url, params=params, headers=self._headers()
                )
                if response.status_code == 429:
                    raise ProviderError(
                        self.name,
                        "rate limited (HTTP 429). Set SEMANTIC_SCHOLAR_API_KEY or lower "
                        "rate_limit_per_sec in config.",
                    )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"request failed: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(self.name, f"malformed JSON: {exc}") from exc

        self.cache_put(url, params, payload)
        return payload

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = await self._get(
            "/paper/search", {"query": query, "limit": min(limit, 100), "fields": _PAPER_FIELDS}
        )
        hits: list[SearchHit] = []
        for rank, item in enumerate(payload.get("data", []), start=1):
            paper = self._parse_paper(item)
            if paper is not None:
                hits.append(SearchHit(paper=paper, rank=rank, provider=self.name))
        return hits

    async def references(self, paper: Paper, limit: int) -> list[Paper]:
        return await self._edges(paper, "references", limit)

    async def citations(self, paper: Paper, limit: int, query: str | None = None) -> list[Paper]:
        # S2's citations endpoint has no full-text filter, so `query` is accepted and
        # ignored — the port's contract is that a provider degrades rather than fails.
        return await self._edges(paper, "citations", limit)

    async def _edges(self, paper: Paper, direction: str, limit: int) -> list[Paper]:
        paper_id = self._api_id(paper)
        if not paper_id:
            return []
        # `intents` and `isInfluential` come free on the edge — this is what lets us
        # distinguish a genuine prerequisite from a perfunctory background citation.
        payload = await self._get(
            f"/paper/{paper_id}/{direction}",
            {"limit": min(limit, 100), "fields": f"{_PAPER_FIELDS},intents,isInfluential"},
        )
        key = "citedPaper" if direction == "references" else "citingPaper"
        results: list[Paper] = []
        for edge in payload.get("data", []):
            parsed = self._parse_paper(edge.get(key) or {})
            if parsed is not None:
                results.append(parsed)
        return results

    def _api_id(self, paper: Paper) -> str | None:
        """S2 accepts several ID forms; use the strongest one this paper carries."""
        if s2_id := paper.id_for(IdNamespace.S2):
            return s2_id
        if doi := paper.id_for(IdNamespace.DOI):
            return f"DOI:{doi}"
        if arxiv_id := paper.id_for(IdNamespace.ARXIV):
            return f"arXiv:{arxiv_id}"
        return None

    def _parse_paper(self, item: dict) -> Paper | None:
        title = item.get("title")
        paper_id = item.get("paperId")
        if not title or not paper_id:
            return None

        external_ids: set[ExternalId] = {ExternalId(IdNamespace.S2, paper_id)}
        for key, namespace in (
            ("DOI", IdNamespace.DOI),
            ("ArXiv", IdNamespace.ARXIV),
            ("PubMed", IdNamespace.PMID),
        ):
            if value := (item.get("externalIds") or {}).get(key):
                external_ids.add(normalize_external_id(namespace, str(value)))

        abstract = item.get("abstract")
        authors = [
            name for author in (item.get("authors") or []) if (name := author.get("name"))
        ]

        provenance = FieldProvenance(provider=self.name, license=_LICENSE)
        paper = Paper(
            canonical_id="",
            title=title,
            external_ids=external_ids,
            abstract=abstract,
            year=item.get("year"),
            authors=authors,
            venue=item.get("venue") or None,
            citation_count=item.get("citationCount"),
            pdf_url=(item.get("openAccessPdf") or {}).get("url"),
            landing_url=item.get("url"),
            depth=ContentDepth.ABSTRACT if abstract else ContentDepth.METADATA,
            provenance={
                field: provenance
                for field in ("title", "abstract", "year", "authors", "citation_count")
            },
        )
        paper.canonical_id = canonical_id_for(paper)
        return paper
