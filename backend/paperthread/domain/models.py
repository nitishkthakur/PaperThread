"""Core domain types.

Nothing provider-shaped lives here. Adapters translate into these types; the rest of the
application only ever sees these. See REQUIREMENTS.md §11 D1/D9/D11.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IdNamespace(str, Enum):
    """External identifier namespaces.

    Deliberately NOT a primary key. Many arXiv preprints have no DOI, so DOI cannot be
    assumed present (PROVIDER_NOTES C5). PaperThread owns canonical identity; these are
    aliases pointing at it.
    """

    DOI = "doi"
    ARXIV = "arxiv"
    OPENALEX = "openalex"
    S2 = "s2"
    PMID = "pmid"
    URL = "url"


@dataclass(frozen=True, slots=True)
class ExternalId:
    namespace: IdNamespace
    value: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.namespace.value}:{self.value}"


class ContentDepth(int, Enum):
    """How much of a paper we actually hold.

    Depth is explicit and ordered so it can INCREASE later without a schema rewrite
    (D8: metadata + abstract now, full text later). Consumers ask for the depth they
    need rather than assuming abstract-only.
    """

    METADATA = 1
    ABSTRACT = 2
    FULLTEXT = 3


@dataclass(frozen=True, slots=True)
class FieldProvenance:
    """Which provider supplied a field, and under what terms.

    Required because providers disagree and their licences differ — notably, whether a
    cached abstract may be served to another user once deployed (PROVIDER_NOTES C9).
    """

    provider: str
    license: str | None = None


@dataclass
class Paper:
    """A canonical paper. One logical paper, possibly assembled from several providers."""

    canonical_id: str
    title: str
    external_ids: set[ExternalId] = field(default_factory=set)
    abstract: str | None = None
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    venue: str | None = None
    citation_count: int | None = None
    pdf_url: str | None = None
    landing_url: str | None = None
    depth: ContentDepth = ContentDepth.METADATA
    provenance: dict[str, FieldProvenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Titles arrive with embedded newlines and runs of spaces (arXiv Atom wraps them,
        # and some OpenAlex records carry them through). Left alone they render as a
        # literal "\n" in the UI and, worse, break title matching during resolution.
        self.title = " ".join(self.title.split())

    @property
    def has_abstract(self) -> bool:
        return bool(self.abstract and self.abstract.strip())

    def id_for(self, namespace: IdNamespace) -> str | None:
        for ext in self.external_ids:
            if ext.namespace is namespace:
                return ext.value
        return None

    def merge_from(self, other: Paper) -> None:
        """Fold another provider's view of the same paper into this one.

        First writer wins per field, so provider order in config expresses preference.
        Papers with no abstract are RETAINED, never dropped — abstract coverage is worst
        for older papers, which are exactly the foundational ones we exist to surface
        (PROVIDER_NOTES C2).
        """
        self.external_ids |= other.external_ids

        for attr in ("abstract", "venue", "pdf_url", "landing_url"):
            if getattr(self, attr) is None and getattr(other, attr) is not None:
                setattr(self, attr, getattr(other, attr))
                if attr in other.provenance:
                    self.provenance[attr] = other.provenance[attr]

        # Year takes the EARLIEST of the merged records, not the first seen. Providers
        # report the year of whichever version they indexed, so AlexNet arrives as 2017
        # from a reprint and Vaswani as 2025 from a re-issue. For a product that orders
        # papers chronologically the year is a sort key, and "when did this idea first
        # appear" is always the earlier date.
        if other.year is not None and (self.year is None or other.year < self.year):
            self.year = other.year
            if "year" in other.provenance:
                self.provenance["year"] = other.provenance["year"]

        if not self.authors and other.authors:
            self.authors = other.authors

        # Citation counts are split across preprint/published versions and differ by
        # provider; the maximum is the least-wrong single number (PROVIDER_NOTES C4/C7).
        if other.citation_count is not None:
            self.citation_count = max(self.citation_count or 0, other.citation_count)

        if other.depth > self.depth:
            self.depth = other.depth


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One provider's ranked result. `rank` is 1-based within that provider's list."""

    paper: Paper
    rank: int
    provider: str
    score: float | None = None


@dataclass(frozen=True, slots=True)
class RankedPaper:
    """A paper after cross-provider fusion."""

    paper: Paper
    score: float
    found_by: list[str]
    ranks: dict[str, int]
