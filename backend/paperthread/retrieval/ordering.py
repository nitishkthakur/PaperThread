"""Stage 5 — turning a scored set of papers into a sequence.

**The hard constraint is free and unfalsifiable: if A cites B, B precedes A.** A's authors
had read B; B existed first. Enforcing that as a structural invariant rather than asking a
model to respect it eliminates an entire class of ordering error at zero cost, and it holds
whether or not L4 ran.

Two things this module deliberately does not do:

* **It does not produce a single chain.** A path is a *partial* order. Papers at the same
  level have no dependency between them and may be read in any order; flattening them into
  a line would invent constraints the evidence does not support. Levels are the output.
* **It is not Main Path Analysis.** MPA is the obvious-looking tool and the evidence is
  against it: formally biased toward long paths, greedy traversal into dead branches, and
  it breaks on cycles — where our corpus is the pathological case, since arXiv
  preprint/published duplication is the canonical producer of strongly-connected components
  (RETRIEVAL_NOTES Finding 3, PROVIDER_NOTES C4).

Cycles are still possible even after deduplication — providers report inconsistent edges,
and a v2 preprint can cite something that cites its v1. They are broken here rather than
allowed to fail, because a user asking for a reading path is entitled to one even when the
citation data disagrees with itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Ordering:
    """The result of ordering: a level per paper, and a global reading position."""

    levels: dict[str, int] = field(default_factory=dict)
    order: dict[str, int] = field(default_factory=dict)
    # Direct prerequisites kept after transitive reduction — what the UI draws as arrows.
    direct_prerequisites: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Edges dropped to make the graph acyclic. Reported, never silently discarded.
    broken_edges: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def order_path(
    paper_ids: list[str],
    citation_edges: set[tuple[str, str]],
    prerequisite_edges: set[tuple[str, str]],
    years: dict[str, int | None],
    rank_key,
) -> Ordering:
    """Order `paper_ids` under the citation constraint.

    `citation_edges` are (citing, cited) pairs — the hard ordering constraint.
    `prerequisite_edges` are (prerequisite, dependent) pairs — the *claimed* teaching
    dependencies, from structure or from L4. They are what the reader is shown, and they
    are a subset of the ordering constraints, not a replacement for them.

    `rank_key` maps a paper id to a sort key used to break ties within a level. Injected
    rather than computed here so this module stays free of scoring policy.
    """
    nodes = sorted(set(paper_ids))
    result = Ordering()
    if not nodes:
        return result

    known = set(nodes)
    # (prerequisite, dependent): B must precede A. Note the inversion from citation order.
    constraints = {(cited, citing) for citing, cited in citation_edges if citing in known and cited in known}
    constraints |= {(a, b) for a, b in prerequisite_edges if a in known and b in known}
    constraints = {(a, b) for a, b in constraints if a != b}

    acyclic, broken = _break_cycles(nodes, constraints, years)
    result.broken_edges = broken
    if broken:
        result.notes.append(
            f"{len(broken)} citation edge(s) formed cycles and were dropped to make the "
            f"path orderable — providers disagree, and preprint/published pairs can cite "
            f"each other."
        )

    result.levels = _assign_levels(nodes, acyclic)

    # Only show prerequisites that survived cycle-breaking, and only direct ones.
    claimed = {(a, b) for a, b in prerequisite_edges if (a, b) in acyclic}
    reduced = _transitive_reduction(nodes, claimed)
    direct: dict[str, list[str]] = {node: [] for node in nodes}
    for prerequisite, dependent in reduced:
        direct[dependent].append(prerequisite)
    result.direct_prerequisites = {
        node: tuple(sorted(prerequisites, key=lambda p: (result.levels[p], p)))
        for node, prerequisites in direct.items()
    }

    ordered = sorted(nodes, key=lambda node: (result.levels[node], rank_key(node), node))
    result.order = {node: i for i, node in enumerate(ordered)}
    return result


def _break_cycles(
    nodes: list[str], edges: set[tuple[str, str]], years: dict[str, int | None]
) -> tuple[set[tuple[str, str]], list[tuple[str, str]]]:
    """Remove back edges found by a deterministic DFS.

    Nodes are visited oldest-first, so when a cycle must be broken the edge that gets
    dropped is the one pointing backwards in time — the one more likely to be a data error.

    Undated papers are placed at the median year of the dated ones rather than at an
    extreme. Sinking them to the end would make every undated ancestor look like the most
    recent work in the topic, and undated papers skew old, not new (PROVIDER_NOTES C2).
    """
    dated = sorted(year for year in (years.get(node) for node in nodes) if year is not None)
    fallback = dated[len(dated) // 2] if dated else 0

    def when(node: str) -> int:
        year = years.get(node)
        return year if year is not None else fallback

    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in sorted(edges):
        outgoing[source].append(target)

    WHITE, GREY, BLACK = 0, 1, 2
    colour = {node: WHITE for node in nodes}
    broken: list[tuple[str, str]] = []
    kept = set(edges)

    for root in sorted(nodes, key=lambda node: (when(node), node)):
        if colour[root] != WHITE:
            continue
        # Iterative DFS: a topic's subgraph can be deep enough to blow the recursion limit,
        # and a stack overflow here would be an outage, not a bad ordering.
        stack: list[tuple[str, int]] = [(root, 0)]
        colour[root] = GREY
        while stack:
            node, index = stack[-1]
            if index < len(outgoing[node]):
                stack[-1] = (node, index + 1)
                target = outgoing[node][index]
                if colour[target] == GREY:
                    broken.append((node, target))  # back edge: closes a cycle
                    kept.discard((node, target))
                elif colour[target] == WHITE:
                    colour[target] = GREY
                    stack.append((target, 0))
            else:
                colour[node] = BLACK
                stack.pop()

    return kept, sorted(broken)


def _assign_levels(nodes: list[str], edges: set[tuple[str, str]]) -> dict[str, int]:
    """Longest-path layering: level(x) = 1 + max(level of x's prerequisites).

    Longest path rather than shortest, so a paper never appears before something it depends
    on — the whole point of the exercise. Requires a DAG, which `_break_cycles` guarantees.
    """
    dependents: dict[str, list[str]] = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for prerequisite, dependent in edges:
        dependents[prerequisite].append(dependent)
        indegree[dependent] += 1

    levels = {node: 0 for node in nodes}
    ready = sorted(node for node in nodes if indegree[node] == 0)
    while ready:
        node = ready.pop(0)
        for dependent in sorted(dependents[node]):
            levels[dependent] = max(levels[dependent], levels[node] + 1)
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                # Keep the frontier sorted so the traversal is deterministic.
                ready.append(dependent)
                ready.sort()
    return levels


def _transitive_reduction(
    nodes: list[str], edges: set[tuple[str, str]]
) -> set[tuple[str, str]]:
    """Drop edges implied by a longer path.

    If A → B → C and also A → C, the direct A → C edge tells the reader nothing and adds a
    line to the diagram. On a citation graph this matters a lot: papers cite their
    grandparents as well as their parents, so without reduction almost every early paper
    connects to almost every later one and the "path" becomes an unreadable mesh.
    """
    successors: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in edges:
        successors[source].add(target)

    # Reachability excluding the direct hop, computed over a topological order.
    order = _topological_order(nodes, edges)
    reachable: dict[str, set[str]] = {node: set() for node in nodes}
    for node in reversed(order):
        for target in successors[node]:
            reachable[node].add(target)
            reachable[node] |= reachable[target]

    return {
        (source, target)
        for source, target in edges
        if not any(
            target in reachable[intermediate]
            for intermediate in successors[source]
            if intermediate != target
        )
    }


def _topological_order(nodes: list[str], edges: set[tuple[str, str]]) -> list[str]:
    dependents: dict[str, list[str]] = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for source, target in edges:
        dependents[source].append(target)
        indegree[target] += 1

    ready = sorted(node for node in nodes if indegree[node] == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for target in sorted(dependents[node]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    # A cycle would leave nodes unvisited; callers pass a DAG, but never silently lose one.
    order.extend(sorted(set(nodes) - set(order)))
    return order
