"""Canonical paper identity and cross-provider deduplication.

This is not a cleanup step — it is core correctness. The same work exists as an arXiv
preprint AND a published paper, with different IDs, dates, and *separate citation counts*.
Left unreconciled that produces duplicate nodes in a learning path, corrupted centrality
scoring, and false prerequisite edges between a paper and its own preprint
(PROVIDER_NOTES C4). Batagelj names arXiv preprint duplication as the pathological case
for citation-graph algorithms specifically.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from .models import ExternalId, IdNamespace, Paper

# Identifier priority for choosing a canonical id: strongest, most stable first.
_ID_PRIORITY = (
    IdNamespace.DOI,
    IdNamespace.ARXIV,
    IdNamespace.OPENALEX,
    IdNamespace.S2,
    IdNamespace.PMID,
)

_ARXIV_VERSION_RE = re.compile(r"v\d+$")
_DOI_PREFIX_RE = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:)", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_doi(raw: str) -> str:
    return _DOI_PREFIX_RE.sub("", raw.strip()).lower()


def normalize_arxiv_id(raw: str) -> str:
    """Strip URL wrapper and version suffix.

    `2301.12345v3` and `2301.12345` are the same work; treating them separately is the
    single easiest way to get duplicate nodes.
    """
    value = raw.strip()
    value = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    if value.lower().startswith("arxiv:"):
        value = value[6:]
    return _ARXIV_VERSION_RE.sub("", value.strip())


def normalize_openalex_id(raw: str) -> str:
    return raw.strip().rsplit("/", 1)[-1].upper()


def normalize_external_id(namespace: IdNamespace, value: str) -> ExternalId:
    match namespace:
        case IdNamespace.DOI:
            return ExternalId(namespace, normalize_doi(value))
        case IdNamespace.ARXIV:
            return ExternalId(namespace, normalize_arxiv_id(value))
        case IdNamespace.OPENALEX:
            return ExternalId(namespace, normalize_openalex_id(value))
        case _:
            return ExternalId(namespace, value.strip())


def title_fingerprint(title: str) -> str:
    """Aggressive title normalization, used only as a last-resort matcher.

    Lowercase, strip accents and all non-alphanumerics. Deliberately lossy: it must match
    "Attention Is All You Need" against "Attention is all you need." Never used alone to
    merge when a shared strong identifier is available.
    """
    folded = unicodedata.normalize("NFKD", title)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return _NON_ALNUM_RE.sub("", folded.lower())


def canonical_id_for(paper: Paper) -> str:
    """Pick a stable canonical id.

    Prefers the strongest available external identifier. Falls back to a title+year hash
    so that a paper with NO usable identifier still gets stable identity rather than being
    dropped — many arXiv preprints have no DOI (PROVIDER_NOTES C5).
    """
    by_ns = {ext.namespace: ext.value for ext in paper.external_ids}
    for namespace in _ID_PRIORITY:
        if namespace in by_ns:
            return f"{namespace.value}:{by_ns[namespace]}"

    seed = f"{title_fingerprint(paper.title)}|{paper.year or ''}"
    return "sig:" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


# A preprint and its published version are usually 0-2 years apart. Wider than this and
# genuinely distinct works start colliding — e.g. two different surveys sharing a title.
_YEAR_WINDOW = 1
_MIN_FINGERPRINT_LEN = 12  # shorter titles are not distinctive enough to merge on


def merge_keys(paper: Paper) -> list[str]:
    """All keys under which this paper should be discoverable for merging.

    Two papers merge if they share ANY key. Strong identifiers are exact. The title key is
    widened over a small year window, because the preprint/published pair usually shares
    no identifier at all AND is often a year apart — matching on title+exact-year would
    miss precisely the case this exists for.

    Papers with no year are handled separately in `deduplicate`, not here: giving them a
    year-agnostic title key would transitively merge every same-titled paper regardless of
    era.
    """
    keys = [f"{ext.namespace.value}:{ext.value}" for ext in paper.external_ids]
    fingerprint = title_fingerprint(paper.title)
    if len(fingerprint) >= _MIN_FINGERPRINT_LEN and paper.year is not None:
        for year in range(paper.year - _YEAR_WINDOW, paper.year + _YEAR_WINDOW + 1):
            keys.append(f"title:{fingerprint}|{year}")
    return keys


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Merge papers that refer to the same work, preserving input order.

    Union-find over shared merge keys, so A~B and B~C collapses A, B and C together even
    when A and C share nothing directly — which is the common case for
    arXiv-preprint / OpenAlex-record / S2-record triples.
    """
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        while parent.setdefault(i, i) != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)  # keep the earliest index as root

    key_owner: dict[str, int] = {}
    for index, paper in enumerate(papers):
        find(index)
        for key in merge_keys(paper):
            if key in key_owner:
                union(key_owner[key], index)
            else:
                key_owner[key] = index

    _absorb_undated(papers, find, union)

    merged: dict[int, Paper] = {}
    order: list[int] = []
    for index, paper in enumerate(papers):
        root = find(index)
        if root not in merged:
            merged[root] = paper
            order.append(root)
        elif merged[root] is not paper:
            merged[root].merge_from(paper)

    result = [merged[root] for root in order]
    for paper in result:
        paper.canonical_id = canonical_id_for(paper)
    return result


def _absorb_undated(papers: list[Paper], find, union) -> None:
    """Attach year-less papers to a dated group with the same title, when unambiguous.

    Providers do sometimes omit a publication year. Such a record should still merge with
    the same work from another provider — but only when there is exactly ONE candidate
    group, otherwise we would be guessing which era it belongs to.
    """
    groups_by_title: dict[str, set[int]] = {}
    undated: list[tuple[int, str]] = []

    for index, paper in enumerate(papers):
        fingerprint = title_fingerprint(paper.title)
        if len(fingerprint) < _MIN_FINGERPRINT_LEN:
            continue
        if paper.year is None:
            undated.append((index, fingerprint))
        else:
            groups_by_title.setdefault(fingerprint, set()).add(find(index))

    first_undated_of_title: dict[str, int] = {}
    for index, fingerprint in undated:
        candidates = groups_by_title.get(fingerprint)
        if candidates:
            # A dated group exists: join it only if there is exactly one, otherwise we
            # would be guessing which era this record belongs to.
            if len({find(root) for root in candidates}) == 1:
                union(next(iter(candidates)), index)
        elif fingerprint in first_undated_of_title:
            # No dated group at all — undated records of the same title are each other's
            # only evidence, and two providers both omitting the year is common.
            union(first_undated_of_title[fingerprint], index)
        else:
            first_undated_of_title[fingerprint] = index
