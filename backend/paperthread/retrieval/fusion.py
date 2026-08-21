"""Reciprocal Rank Fusion.

Combines ranked lists from providers (and later, from retrieval layers) without needing
their scores to be comparable — which they are not, since every provider scores
differently and none documents its scale.

Measured value: fusion buys roughly +5 pp Recall@5 over the best single retriever. Worth
having, and notably less than reranking buys on top of it (RETRIEVAL_NOTES L3).
"""

from __future__ import annotations

from ..domain.identity import deduplicate
from ..domain.models import Paper, RankedPaper, SearchHit

DEFAULT_K = 60


def reciprocal_rank_fusion(
    hit_lists: list[list[SearchHit]], k: int = DEFAULT_K
) -> list[RankedPaper]:
    """Fuse per-provider ranked lists into one ranking.

    score(paper) = sum over providers of 1 / (k + rank). `k` damps the influence of the
    very top ranks so a single provider cannot dominate; 60 is the constant from the
    original paper.

    Deduplication happens FIRST: the same work returned by three providers must score as
    one paper found three times, not three papers found once. Skipping this would let
    preprint/published duplicates split their own score (PROVIDER_NOTES C4).
    """
    flat: list[SearchHit] = [hit for hits in hit_lists for hit in hits]
    if not flat:
        return []

    canonical = deduplicate([hit.paper for hit in flat])

    # deduplicate() may rewrite canonical_id, so map every pre-merge id to its survivor.
    survivor_by_id: dict[str, Paper] = {}
    for paper in canonical:
        survivor_by_id[paper.canonical_id] = paper
        for external in paper.external_ids:
            survivor_by_id.setdefault(str(external), paper)

    scores: dict[str, float] = {}
    found_by: dict[str, list[str]] = {}
    ranks: dict[str, dict[str, int]] = {}
    papers: dict[str, Paper] = {}

    for hit in flat:
        survivor = _resolve(hit.paper, survivor_by_id)
        key = survivor.canonical_id
        papers.setdefault(key, survivor)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + hit.rank)
        ranks.setdefault(key, {})
        # A provider may return the same work twice (e.g. preprint + published). Keep its
        # best rank so the duplicate does not inflate the score.
        previous = ranks[key].get(hit.provider)
        if previous is None or hit.rank < previous:
            ranks[key][hit.provider] = hit.rank
        if hit.provider not in found_by.setdefault(key, []):
            found_by[key].append(hit.provider)

    # Recompute from best-per-provider ranks so within-provider duplicates don't double count.
    fused = [
        RankedPaper(
            paper=papers[key],
            score=sum(1.0 / (k + rank) for rank in provider_ranks.values()),
            found_by=found_by[key],
            ranks=provider_ranks,
        )
        for key, provider_ranks in ranks.items()
    ]
    # Ties broken by provider agreement, then title, so ordering is deterministic.
    fused.sort(key=lambda r: (-r.score, -len(r.found_by), r.paper.title.lower()))
    return fused


def _resolve(paper: Paper, survivor_by_id: dict[str, Paper]) -> Paper:
    if survivor := survivor_by_id.get(paper.canonical_id):
        return survivor
    for external in paper.external_ids:
        if survivor := survivor_by_id.get(str(external)):
            return survivor
    return paper
