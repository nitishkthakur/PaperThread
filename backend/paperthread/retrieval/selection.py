"""Choosing which papers make the path, and which pairs are worth an LLM call.

Two selection problems, both of them budget problems.

**Which papers?** A topic's subgraph runs to a few hundred nodes; a reading path is a
couple of dozen. Selecting purely by structural centrality would return a path of famous
old papers that never reaches the work the user actually asked about. Selecting purely by
search rank returns the current-vocabulary papers and none of the foundation — which is the
failure the whole citation-expansion stage exists to fix. So selection is **quota-based and
explicit**: guaranteed slots for the topic's shared foundation, guaranteed slots for its
current surface, and the remainder to whatever scores highest structurally. A blended
scalar score would hide this trade-off inside weights nobody can interpret.

**Which pairs?** Every citation between selected papers is a candidate prerequisite, which
is far more pairs than are worth judging. The reranking evidence frames the real constraint:
the scarce resource is the LLM judgment budget and the integrity of the persisted edge set,
not recall (RETRIEVAL_NOTES L3). So pairs are ranked by how much the *prerequisite* looks
like shared foundation, and the tail is dropped rather than judged cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.models import Paper
from .expansion import ExpandedCorpus
from .graph import GraphAnalysis


@dataclass
class Selection:
    paper_ids: list[str] = field(default_factory=list)
    # Why each paper is in the path — surfaced so the choice stays auditable.
    reasons: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def select_path_papers(
    corpus: ExpandedCorpus,
    analysis: GraphAnalysis,
    budget: int,
    min_co_citations: int,
) -> Selection:
    """Pick the papers that form the path.

    Quotas are fractions of the budget rather than fixed counts so that a smaller path
    stays balanced instead of degenerating into whichever category is listed first.
    """
    selection = Selection()
    if budget <= 0 or not corpus.papers:
        return selection

    seed_rank = {paper_id: i for i, paper_id in enumerate(corpus.seed_ids)}
    rescaled = analysis.age_rescaled
    co_cited = corpus.co_citations

    def structural(paper_id: str) -> float:
        return rescaled.get(paper_id, 0.0)

    # The shared foundation: papers many of the topic's own candidates cite. These are the
    # ones keyword search structurally cannot find, so they get first claim on the budget.
    foundation = sorted(
        (p for p in corpus.papers if co_cited.get(p, 0) >= min_co_citations),
        key=lambda p: (-co_cited.get(p, 0), -structural(p), p),
    )
    # The topic's current surface: what the user's words actually matched.
    surface = sorted(
        (p for p in corpus.papers if p in seed_rank), key=lambda p: (seed_rank[p], p)
    )

    chosen: list[str] = []
    taken: set[str] = set()

    def take(paper_ids: list[str], limit: int, reason: str) -> None:
        added = 0
        for paper_id in paper_ids:
            if added >= limit or len(chosen) >= budget:
                break
            if paper_id in taken:
                continue
            taken.add(paper_id)
            chosen.append(paper_id)
            selection.reasons[paper_id] = reason
            added += 1

    take(foundation, max(1, int(budget * 0.4)), "cited in common by the topic's own papers")
    take(surface, max(1, int(budget * 0.4)), "matched the topic directly")
    # Whatever is left goes to structural standing, which is how mid-graph bridging papers
    # — cited by the surface, citing the foundation — get in at all.
    #
    # Filtered by `_has_topic_evidence` first, and that filter is load-bearing. Age
    # rescaling divides by the spread of a paper's age cohort, and our subgraph has very
    # few papers in any given pre-2000 year, so a lone old paper in a sparse cohort scores
    # spectacularly on nothing at all. Without this guard a search for "diffusion models"
    # promotes a 1997 paper about Italian WordNet into the reading path, purely because
    # some candidate cited it once and it had no age peers to be compared against.
    remaining = sorted(
        (p for p in corpus.papers if _has_topic_evidence(p, corpus, analysis)),
        key=lambda p: (-structural(p), p),
    )
    take(remaining, budget, "central in this topic's citation graph")

    selection.paper_ids = chosen
    foundation_count = sum(
        1 for p in chosen if selection.reasons[p] == "cited in common by the topic's own papers"
    )
    if not foundation_count:
        selection.notes.append(
            f"No paper is cited by {min_co_citations}+ of this topic's candidates, so the "
            f"path has no evidenced foundation layer — it is ordered current work. This is "
            f"normal for a very new or very narrow topic."
        )
    return selection


def _has_topic_evidence(
    paper_id: str, corpus: ExpandedCorpus, analysis: GraphAnalysis
) -> bool:
    """Is there evidence this paper belongs to the topic at all?

    A single citation from a single candidate is not evidence — papers cite tools,
    datasets, and asides from unrelated fields. Two independent citations within the
    topic's own subgraph is the weakest claim worth acting on.

    One is genuinely not enough, and the failure is not hypothetical: a lexical false
    positive that reaches the candidate set drags its whole ancestry in behind it. A search
    for "diffusion models" matches a 1940 chemical-kinetics paper, expansion then follows
    its references, and three 1960s papers on osmotic flow through cellulose acetate
    membranes arrive with exactly one citation each — enough to pass a threshold of one,
    and each scoring well because almost nothing else in the subgraph shares their decade.

    Deep ancestors reached in the second expansion round legitimately have zero
    *co-citations*, because that counter only counts citations from Stage 1 candidates.
    In-degree is what keeps them eligible.
    """
    return (
        corpus.co_citations.get(paper_id, 0) >= 2
        or analysis.in_degree.get(paper_id, 0) >= 2
    )


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """A citation that *might* be a prerequisite relationship. Most are not."""

    prerequisite_id: str
    dependent_id: str
    # How many of the topic's candidates cite the prerequisite. The strongest cheap signal
    # that it is shared background rather than an incidental reference.
    co_citations: int


def candidate_pairs(
    paper_ids: list[str],
    citation_edges: set[tuple[str, str]],
    papers: dict[str, Paper],
    co_citations: dict[str, int],
    analysis: GraphAnalysis,
    limit: int,
) -> list[CandidatePair]:
    """Citations between selected papers, ranked by how likely they are to matter.

    A pair is excluded when the cited paper is *newer* than the citing one. That is a data
    error — a paper cannot cite its own future — and it comes from providers reporting the
    published year of one version against the preprint year of another.
    """
    selected = set(paper_ids)
    pairs: list[CandidatePair] = []
    for citing, cited in citation_edges:
        if citing not in selected or cited not in selected or citing == cited:
            continue
        citing_year = papers[citing].year
        cited_year = papers[cited].year
        if citing_year is not None and cited_year is not None and cited_year > citing_year:
            continue
        pairs.append(CandidatePair(cited, citing, co_citations.get(cited, 0)))

    pairs.sort(
        key=lambda pair: (
            -pair.co_citations,
            -analysis.age_rescaled.get(pair.prerequisite_id, 0.0),
            pair.prerequisite_id,
            pair.dependent_id,
        )
    )
    return pairs[:limit]
