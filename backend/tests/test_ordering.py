"""Stage 5 — sequencing.

The invariant under test is the one the product rests on: **if A cites B, B precedes A**.
It is free, unfalsifiable, and it eliminates a whole class of ordering error — but only if
it actually holds for every pair, including when the citation data contradicts itself.
"""

from paperthread.retrieval.ordering import order_path

FLAT = lambda node: 0  # noqa: E731 - tie-break is irrelevant to most of these


def assert_respects_citations(ordering, citations, ids):
    """Every cited paper must come strictly before the paper citing it."""
    for citing, cited in citations:
        if citing in ids and cited in ids and (cited, citing) not in ordering.broken_edges:
            assert ordering.order[cited] < ordering.order[citing], f"{cited} !< {citing}"


class TestOrdering:
    def test_empty(self):
        result = order_path([], set(), set(), {}, FLAT)
        assert result.levels == {} and result.order == {}

    def test_chain_is_ordered_oldest_first(self):
        citations = {("b", "a"), ("c", "b")}
        result = order_path(["a", "b", "c"], citations, set(), {"a": 2010, "b": 2015, "c": 2020}, FLAT)
        assert result.levels == {"a": 0, "b": 1, "c": 2}
        assert_respects_citations(result, citations, {"a", "b", "c"})

    def test_papers_with_no_dependency_share_a_level(self):
        """A path is a PARTIAL order. Forcing unrelated papers into a line would invent
        constraints the citation evidence does not support."""
        result = order_path(["x", "y"], set(), set(), {"x": 2019, "y": 2021}, FLAT)
        assert result.levels == {"x": 0, "y": 0}

    def test_level_is_longest_path_not_shortest(self):
        # d cites both a (direct) and c; c cites b cites a. d must sit above c.
        citations = {("b", "a"), ("c", "b"), ("d", "c"), ("d", "a")}
        result = order_path(
            ["a", "b", "c", "d"], citations, set(), {"a": 2000, "b": 2005, "c": 2010, "d": 2015}, FLAT
        )
        assert result.levels["d"] == 3
        assert_respects_citations(result, citations, {"a", "b", "c", "d"})

    def test_within_level_order_uses_the_injected_rank_key(self):
        result = order_path(
            ["p", "q", "r"], set(), set(), {"p": 2020, "q": 2019, "r": 2021},
            lambda n: {"p": 1, "q": 0, "r": 2}[n],
        )
        assert [result.order[k] for k in ("q", "p", "r")] == [0, 1, 2]


class TestCycleBreaking:
    def test_mutual_citation_still_yields_an_order(self):
        """arXiv preprint/published duplication is the canonical cycle producer. A user
        asking for a reading path gets one even when the provider data disagrees."""
        result = order_path(["x", "y"], {("x", "y"), ("y", "x")}, set(), {"x": 2019, "y": 2021}, FLAT)
        assert len(result.broken_edges) == 1
        assert set(result.levels.values()) == {0, 1}
        assert result.notes

    def test_breaks_the_edge_pointing_backwards_in_time(self):
        # y (2021) cites x (2019) is plausible; x citing y is the data error.
        result = order_path(["x", "y"], {("x", "y"), ("y", "x")}, set(), {"x": 2019, "y": 2021}, FLAT)
        assert result.order["x"] < result.order["y"]

    def test_three_cycle_is_resolved(self):
        result = order_path(
            ["a", "b", "c"], {("a", "b"), ("b", "c"), ("c", "a")}, set(),
            {"a": 2020, "b": 2018, "c": 2019}, FLAT,
        )
        assert len(result.order) == 3
        assert len(set(result.order.values())) == 3

    def test_undated_papers_do_not_sink_to_the_end(self):
        """Undated papers skew OLD, not new (abstract/metadata coverage is worst for early
        work), so treating a missing year as "most recent" is backwards."""
        result = order_path(
            ["old", "mid", "new"], set(), set(),
            {"old": 1995, "mid": None, "new": 2024}, FLAT,
        )
        assert result.levels == {"old": 0, "mid": 0, "new": 0}


class TestTransitiveReduction:
    def test_redundant_grandparent_edge_is_hidden(self):
        """Papers cite their grandparents as well as their parents. Without reduction the
        diagram becomes a mesh in which nothing is legibly "next"."""
        prerequisites = {("a", "b"), ("b", "c"), ("a", "c")}
        result = order_path(
            ["a", "b", "c"], set(), prerequisites, {"a": 2010, "b": 2015, "c": 2020}, FLAT
        )
        assert result.direct_prerequisites["c"] == ("b",)
        assert result.direct_prerequisites["b"] == ("a",)

    def test_independent_prerequisites_are_both_kept(self):
        prerequisites = {("a", "c"), ("b", "c")}
        result = order_path(
            ["a", "b", "c"], set(), prerequisites, {"a": 2010, "b": 2011, "c": 2020}, FLAT
        )
        assert set(result.direct_prerequisites["c"]) == {"a", "b"}

    def test_prerequisites_pointing_outside_the_path_are_dropped(self):
        result = order_path(["a"], set(), {("ghost", "a")}, {"a": 2020}, FLAT)
        assert result.direct_prerequisites["a"] == ()
