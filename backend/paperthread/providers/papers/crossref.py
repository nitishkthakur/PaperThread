"""Crossref adapter.

Capabilities: search, id_resolve. **No citation graph** (Crossref's reference data is
opt-in per publisher and patchy), and no abstracts for most records.

It exists to close a coverage gap that is not incidental to this product — it is the gap
that hurts most. arXiv began in 1991 and indexes preprints, so the foundational papers a
learning path needs are structurally absent from it: Rumelhart's backpropagation paper
(Nature, 1986), Breiman's bagging paper (1996), Tibshirani's lasso paper (1996),
Hochreiter & Schmidhuber on LSTM (1997). Those are exactly the prerequisites that make a
sequence a *path* rather than a list of recent work, and with OpenAlex now metered they
were unreachable whenever its daily credit ran out.

Crossref has all of them, is free, needs no key, and is not metered.

Division of labour across the three search providers:

    arXiv      modern preprints, exact title lookup, free full text
    Crossref   the published record, especially pre-2010 journal literature
    OpenAlex   broadest coverage and the citation graph, but metered

Use `query.title`, not `query.bibliographic`: measured on the same four classics, the
bibliographic query returned figure components and unrelated 2024 preprints, while the
title query returned all four correctly at rank 1.
"""

from __future__ import annotations

import logging

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

_API_URL = "https://api.crossref.org/works"
_LICENSE = "Crossref metadata (CC0 for bibliographic metadata)"
_SELECT = "title,author,issued,DOI,type,container-title,abstract,is-referenced-by-count,URL"

# Crossref indexes figures, datasets, corrections, and whole books alongside papers. A
# `component` is a figure inside a paper; a learner cannot read one.
_READABLE_TYPES = frozenset(
    {
        "journal-article",
        "proceedings-article",
        "posted-content",
        "book-chapter",
        "report",
        "dissertation",
    }
)


class CrossrefProvider(BasePaperProvider):
    name = "crossref"

    def _params(self, extra: dict[str, object]) -> dict[str, object]:
        params: dict[str, object] = {"select": _SELECT, **extra}
        # Crossref's polite pool; the same courtesy convention as OpenAlex.
        if mailto := self.config.options.get("mailto"):
            params["mailto"] = mailto
        return params

    async def _get(self, params: dict[str, object]) -> dict:
        if (cached := self.cache_get(_API_URL, params)) is not None:
            return cached

        await self._throttle()
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.get(_API_URL, params=params)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"request failed: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(self.name, f"malformed JSON: {exc}") from exc

        # As in the arXiv adapter: an empty result is not cached, so a transient bad day
        # cannot become a permanent "this paper does not exist".
        if payload.get("message", {}).get("items"):
            self.cache_put(_API_URL, params, payload)
        return payload

    async def find_by_title(self, title: str, limit: int = 5) -> list[Paper]:
        payload = await self._get(
            self._params({"query.title": title, "rows": min(limit, 20)})
        )
        return self._parse(payload)

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        payload = await self._get(
            self._params({"query.bibliographic": query, "rows": min(limit, 20)})
        )
        return [
            SearchHit(paper=paper, rank=rank, provider=self.name)
            for rank, paper in enumerate(self._parse(payload), start=1)
        ]

    def _parse(self, payload: dict) -> list[Paper]:
        items = (payload.get("message") or {}).get("items") or []
        return [paper for item in items if (paper := self._parse_item(item))]

    def _parse_item(self, item: dict) -> Paper | None:
        titles = item.get("title") or []
        doi = item.get("DOI")
        if not titles or not doi:
            return None
        if item.get("type") not in _READABLE_TYPES:
            return None

        external_ids: set[ExternalId] = {normalize_external_id(IdNamespace.DOI, doi)}

        parts = (item.get("issued") or {}).get("date-parts") or [[None]]
        year = parts[0][0] if parts and parts[0] else None

        authors = [
            name
            for author in (item.get("author") or [])
            if (name := " ".join(filter(None, [author.get("given"), author.get("family")])))
        ]

        venues = item.get("container-title") or []
        # Crossref abstracts arrive as JATS XML when they arrive at all; most records have
        # none, so depth stays METADATA unless one is actually present.
        abstract = item.get("abstract")

        provenance = FieldProvenance(provider=self.name, license=_LICENSE)
        paper = Paper(
            canonical_id="",
            title=titles[0],
            external_ids=external_ids,
            abstract=_strip_jats(abstract) if abstract else None,
            year=year,
            authors=authors,
            venue=venues[0] if venues else None,
            citation_count=item.get("is-referenced-by-count"),
            landing_url=item.get("URL") or f"https://doi.org/{doi}",
            depth=ContentDepth.ABSTRACT if abstract else ContentDepth.METADATA,
            provenance={
                field: provenance
                for field in ("title", "year", "authors", "venue", "citation_count")
            },
        )
        paper.canonical_id = canonical_id_for(paper)
        return paper


def _strip_jats(abstract: str) -> str:
    """Crossref abstracts are JATS XML fragments; the tags are noise to every consumer."""
    import re

    text = re.sub(r"<[^>]+>", " ", abstract)
    return " ".join(text.split())
