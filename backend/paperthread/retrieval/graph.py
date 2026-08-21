"""Stage 3 — structural scoring of the candidate subgraph.

Everything here is pure: it takes nodes and edges and returns numbers. No network, no
model weights, no LLM. This is the layer D12 requires to be genuinely useful on its own,
and it is where the product's actual signal lives.

Three decisions are settled by evidence and should not be relitigated casually
(docs/RETRIEVAL_NOTES.md, "Citation-graph algorithms"):

* **Age-rescaled PageRank, not PageRank, and never HITS.** Raw PageRank "completely fails
  to identify recent milestone papers"; CiteRank over-corrects and then misses the old
  ones. Age-rescaling — scoring a paper against others of its own age — beats citation
  count at *every* paper age, and HITS is measurably terrible on citation networks
  (authority identification rates of 0.14/0.12/0.05 across three corpora).
* **Damping d = 0.5**, from Chen et al. (2007), not the web's 0.85. Roughly half of a
  bibliography's references cite each other, so reference-following paths are short. At
  d = 0.9 PageRank degenerates into citation count and stops telling you anything new.
* **Centrality is computed within the candidate subgraph**, never globally. §3 asks for
  educational value for *this topic*, not for fame.

The payoff case is Slater 1929: 114 citations, citation rank 1853rd, yet a PageRank within
a factor of 2.2 of the top paper — a contribution so foundational that people stopped
citing the original. Citation count cannot see that paper. This can.

Communities come from the citation graph rather than from text similarity, because text
embeddings demonstrably fail to reproduce citation partitions even when trained on
citations, and §4 is about intellectual lineage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import GraphConfig


@dataclass
class GraphAnalysis:
    """Per-node structural signals, keyed by canonical paper id."""

    pagerank: dict[str, float] = field(default_factory=dict)
    # PageRank expressed as a z-score against papers of a similar age. THIS is the ranking
    # signal; `pagerank` is retained only so the rescaling stays auditable.
    age_rescaled: dict[str, float] = field(default_factory=dict)
    communities: dict[str, int] = field(default_factory=dict)
    in_degree: dict[str, int] = field(default_factory=dict)
    out_degree: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def analyze(
    node_ids: list[str],
    edges: set[tuple[str, str]],
    years: dict[str, int | None],
    config: GraphConfig,
) -> GraphAnalysis:
    """Score one topic's induced subgraph. `edges` are (citing, cited) pairs."""
    nodes = sorted(set(node_ids))
    analysis = GraphAnalysis()
    if not nodes:
        return analysis

    # Ignore edges pointing outside the node set; they carry no information about the
    # relative standing of papers we are actually ranking.
    known = set(nodes)
    kept = {(a, b) for a, b in edges if a in known and b in known and a != b}

    analysis.in_degree = {node: 0 for node in nodes}
    analysis.out_degree = {node: 0 for node in nodes}
    for citing, cited in kept:
        analysis.out_degree[citing] += 1
        analysis.in_degree[cited] += 1

    analysis.pagerank = pagerank(
        nodes,
        kept,
        damping=config.pagerank_damping,
        iterations=config.pagerank_iterations,
        tolerance=config.pagerank_tolerance,
    )
    analysis.age_rescaled = age_rescale(
        analysis.pagerank,
        years,
        cohort_years=config.age_cohort_years,
        min_cohort=config.min_cohort_size,
    )
    analysis.communities = louvain(nodes, kept, resolution=config.community_resolution)

    if not kept:
        analysis.notes.append(
            "No citation edges within the candidate set — structural ranking is inactive "
            "and ordering falls back to publication date."
        )
    undated = sum(1 for node in nodes if years.get(node) is None)
    if undated:
        analysis.notes.append(
            f"{undated} paper(s) have no publication year and could not be age-rescaled."
        )
    return analysis


def pagerank(
    nodes: list[str],
    edges: set[tuple[str, str]],
    damping: float,
    iterations: int,
    tolerance: float,
) -> dict[str, float]:
    """PageRank over a citation graph, where an edge (A, B) means "A cites B".

    Importance therefore flows from citing paper to cited paper, which is the direction
    that makes a heavily-cited-by-important-papers node score highly.

    Dangling nodes are the common case here, not an edge case: most nodes are papers whose
    own reference lists were never fetched, so they cite nothing on record. Their mass is
    redistributed uniformly each iteration. Skipping that would leak probability mass and
    quietly deflate every score toward the teleport floor.
    """
    count = len(nodes)
    if count == 0:
        return {}

    index = {node: i for i, node in enumerate(nodes)}
    out_links: list[list[int]] = [[] for _ in range(count)]
    for citing, cited in sorted(edges):
        out_links[index[citing]].append(index[cited])

    out_degree = [len(links) for links in out_links]
    dangling = [i for i in range(count) if out_degree[i] == 0]

    scores = [1.0 / count] * count
    teleport = (1.0 - damping) / count

    for _ in range(iterations):
        incoming = [0.0] * count
        dangling_mass = sum(scores[i] for i in dangling) / count
        for source, links in enumerate(out_links):
            if not links:
                continue
            share = scores[source] / out_degree[source]
            for target in links:
                incoming[target] += share

        updated = [teleport + damping * (incoming[i] + dangling_mass) for i in range(count)]
        delta = sum(abs(updated[i] - scores[i]) for i in range(count))
        scores = updated
        if delta < tolerance:
            break

    return {node: scores[index[node]] for node in nodes}


def age_rescale(
    scores: dict[str, float],
    years: dict[str, int | None],
    cohort_years: int,
    min_cohort: int,
) -> dict[str, float]:
    """Express each score as a z-score against papers of a similar age.

    A 2017 paper and a 2024 paper cannot be compared on raw PageRank: the older one has
    had seven more years to accumulate the citations that PageRank feeds on. Mariani, Medo
    & Zhang (2016) showed that rescaling within an age cohort fixes this at every age,
    where exponential recency discounting (CiteRank) over-corrects and buries the
    foundational work.

    Cohorts widen when a year is sparsely populated, because our subgraph is a few hundred
    papers, not the 450k-paper APS corpus the method was validated on. A cohort of one
    would give every isolated year a z-score of 0.

    Papers with no year score 0.0 — explicitly neutral rather than best or worst, since we
    have no basis for either. They are counted in `GraphAnalysis.notes`.
    """
    dated = sorted(
        ((year, node) for node, year in years.items() if year is not None and node in scores)
    )
    if not dated:
        return {node: 0.0 for node in scores}

    rescaled: dict[str, float] = {}
    for node in scores:
        year = years.get(node)
        if year is None:
            rescaled[node] = 0.0
            continue

        cohort = [other for other_year, other in dated if abs(other_year - year) <= cohort_years]
        if len(cohort) < min_cohort:
            # Widen to the nearest-in-time papers rather than the nearest in years: what
            # matters is having enough of a comparison group, not the window's width.
            nearest = sorted(dated, key=lambda pair: (abs(pair[0] - year), pair[0], pair[1]))
            cohort = [other for _, other in nearest[: max(min_cohort, len(cohort))]]

        values = [scores[other] for other in cohort if other in scores]
        if len(values) < 2:
            rescaled[node] = 0.0
            continue

        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        deviation = math.sqrt(variance)
        # A flat cohort carries no information about who stands out within it.
        rescaled[node] = 0.0 if deviation == 0 else (scores[node] - mean) / deviation

    return rescaled


def louvain(
    nodes: list[str], edges: set[tuple[str, str]], resolution: float = 1.0
) -> dict[str, int]:
    """Louvain community detection on the undirected projection of the citation graph.

    Direction is dropped deliberately: for "which papers form a line of work", A citing B
    and B being cited by A are the same evidence of association.

    **This implementation is deterministic.** The reference algorithm visits nodes in
    random order, which would give the same topic a different curriculum on every run —
    fatal for §6, where a path must update incrementally rather than reshuffle. Nodes are
    visited in sorted order and gain ties break toward the lowest community id.
    """
    if not nodes:
        return {}

    weights: dict[tuple[int, int], float] = {}
    index = {node: i for i, node in enumerate(sorted(nodes))}
    for citing, cited in edges:
        a, b = index[citing], index[cited]
        key = (min(a, b), max(a, b))
        weights[key] = weights.get(key, 0.0) + 1.0

    # No edges: every paper is its own community. Honest — we have no grouping evidence.
    if not weights:
        return {node: i for node, i in index.items()}

    # Communities are tracked over the ORIGINAL nodes and rewritten after each aggregation
    # level, so the returned labels always refer to real papers.
    membership = list(range(len(index)))
    graph_nodes = list(range(len(index)))
    graph_weights = dict(weights)

    while True:
        local = _louvain_level(graph_nodes, graph_weights, resolution)
        if len(set(local.values())) == len(graph_nodes):
            break  # nothing merged; further levels cannot improve modularity
        membership = [local[community] for community in membership]
        graph_nodes, graph_weights = _aggregate(graph_nodes, graph_weights, local)
        if len(graph_nodes) <= 1:
            break

    # Relabel to a dense, deterministic 0..k-1 ordered by smallest member index, so
    # community 0 is always the one containing the first node.
    order: dict[int, int] = {}
    for community in membership:
        order.setdefault(community, len(order))

    reverse = {i: node for node, i in index.items()}
    return {reverse[i]: order[community] for i, community in enumerate(membership)}


def _louvain_level(
    nodes: list[int], weights: dict[tuple[int, int], float], resolution: float
) -> dict[int, int]:
    """One pass of local modularity optimization. Returns node -> community."""
    total = sum(weights.values())
    if total <= 0:
        return {node: node for node in nodes}

    neighbors: dict[int, dict[int, float]] = {node: {} for node in nodes}
    self_loops: dict[int, float] = {node: 0.0 for node in nodes}
    for (a, b), weight in weights.items():
        if a == b:
            self_loops[a] += weight
            continue
        neighbors[a][b] = neighbors[a].get(b, 0.0) + weight
        neighbors[b][a] = neighbors[b].get(a, 0.0) + weight

    # Degree counts a self-loop twice, as in the standard modularity formulation.
    degree = {
        node: sum(neighbors[node].values()) + 2 * self_loops[node] for node in nodes
    }
    community = {node: node for node in nodes}
    community_degree = {node: degree[node] for node in nodes}

    improved = True
    while improved:
        improved = False
        for node in sorted(nodes):
            current = community[node]
            community_degree[current] -= degree[node]

            # Weight from this node into each neighbouring community.
            into: dict[int, float] = {}
            for neighbor, weight in neighbors[node].items():
                if neighbor != node:
                    target = community[neighbor]
                    into[target] = into.get(target, 0.0) + weight
            into.setdefault(current, 0.0)

            def quality(target: int) -> float:
                # ΔQ for placing `node` in `target`, dropping terms that are constant
                # across candidates. Compared against staying put, NOT against zero — a
                # move can improve modularity while both options score negative.
                return into[target] - resolution * community_degree[target] * degree[node] / (
                    2.0 * total
                )

            best_community = current
            best_gain = quality(current)
            for target in sorted(into):
                gain = quality(target)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_community = target

            community[node] = best_community
            community_degree[best_community] += degree[node]
            if best_community != current:
                improved = True

    return community


def _aggregate(
    nodes: list[int], weights: dict[tuple[int, int], float], community: dict[int, int]
) -> tuple[list[int], dict[tuple[int, int], float]]:
    """Collapse each community into a single node, preserving edge weights as self-loops."""
    merged: dict[tuple[int, int], float] = {}
    for (a, b), weight in weights.items():
        ca, cb = community[a], community[b]
        key = (min(ca, cb), max(ca, cb))
        merged[key] = merged.get(key, 0.0) + weight
    return sorted({community[node] for node in nodes}), merged
