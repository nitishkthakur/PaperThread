"""Stage 3 — structural scoring.

These are the numbers every downstream stage trusts: selection, ordering, and the roles
and explanations built on top of them. A regression here does not raise — it reorders a
learning path, which nobody notices until the path is wrong.
"""

from paperthread.config import GraphConfig
from paperthread.retrieval.graph import age_rescale, analyze, louvain, pagerank


def config(**overrides) -> GraphConfig:
    return GraphConfig(**overrides)


class TestPageRank:
    def test_empty_graph(self):
        assert pagerank([], set(), 0.5, 50, 1e-9) == {}

    def test_probability_mass_is_conserved(self):
        """Dangling nodes are the common case, not an edge case.

        Most nodes are papers whose own reference lists were never fetched, so they cite
        nothing on record. If their mass is not redistributed the scores leak toward the
        teleport floor and every comparison between them becomes meaningless.
        """
        nodes = ["a", "b", "c", "d"]
        edges = {("a", "d"), ("b", "d"), ("c", "d")}  # d is dangling
        scores = pagerank(nodes, edges, 0.5, 200, 1e-12)
        assert abs(sum(scores.values()) - 1.0) < 1e-9

    def test_heavily_cited_paper_outranks_its_citers(self):
        nodes = ["old", "p1", "p2", "p3"]
        edges = {("p1", "old"), ("p2", "old"), ("p3", "old")}
        scores = pagerank(nodes, edges, 0.5, 200, 1e-12)
        assert scores["old"] > max(scores["p1"], scores["p2"], scores["p3"])

    def test_isolated_nodes_share_score_equally(self):
        scores = pagerank(["a", "b", "c"], set(), 0.5, 50, 1e-12)
        assert len(set(round(v, 9) for v in scores.values())) == 1

    def test_deterministic_across_runs(self):
        nodes = [f"n{i}" for i in range(12)]
        edges = {(f"n{i}", f"n{(i * 5) % 12}") for i in range(12)}
        first = pagerank(nodes, edges, 0.5, 100, 1e-12)
        second = pagerank(list(reversed(nodes)), edges, 0.5, 100, 1e-12)
        assert first == second


class TestAgeRescale:
    def test_old_paper_is_compared_against_its_own_cohort(self):
        """The whole point: a 1990 paper must not be scored against 2024 papers.

        Raw PageRank rewards age, because older papers have had longer to accumulate the
        citations it feeds on.
        """
        scores = {"old": 0.4, "new1": 0.1, "new2": 0.1, "new3": 0.1, "new4": 0.1}
        years = {"old": 1990, "new1": 2022, "new2": 2022, "new3": 2023, "new4": 2023}
        rescaled = age_rescale(scores, years, cohort_years=3, min_cohort=5)
        assert rescaled["old"] > 0
        assert all(rescaled[k] < 0 for k in ("new1", "new2", "new3", "new4"))

    def test_undated_papers_are_neutral_not_extreme(self):
        """No year is not evidence of being good or bad. Zero is the only honest answer."""
        scores = {"a": 0.5, "b": 0.1, "c": 0.1, "d": 0.1, "e": 0.1, "undated": 0.9}
        years = {"a": 2020, "b": 2020, "c": 2021, "d": 2021, "e": 2022, "undated": None}
        rescaled = age_rescale(scores, years, cohort_years=3, min_cohort=3)
        assert rescaled["undated"] == 0.0

    def test_flat_cohort_yields_zero_not_a_division_error(self):
        scores = {f"p{i}": 0.2 for i in range(5)}
        years = {f"p{i}": 2020 for i in range(5)}
        rescaled = age_rescale(scores, years, cohort_years=3, min_cohort=3)
        assert set(rescaled.values()) == {0.0}

    def test_no_dated_papers_at_all(self):
        rescaled = age_rescale({"a": 0.5, "b": 0.5}, {"a": None, "b": None}, 3, 3)
        assert rescaled == {"a": 0.0, "b": 0.0}


class TestLouvain:
    def test_two_cliques_joined_by_one_edge_split(self):
        nodes = list("abcdef")
        edges = {("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d"), ("c", "d")}
        communities = louvain(nodes, edges)
        assert communities["a"] == communities["b"] == communities["c"]
        assert communities["d"] == communities["e"] == communities["f"]
        assert communities["a"] != communities["d"]

    def test_no_edges_means_no_grouping_evidence(self):
        communities = louvain(["x", "y", "z"], set())
        assert len(set(communities.values())) == 3

    def test_deterministic_regardless_of_input_order(self):
        """Stochastic community detection would give the same topic a different curriculum
        on every run, which breaks §6's incremental path updates outright."""
        nodes = list("abcdef")
        edges = {("a", "b"), ("b", "c"), ("c", "a"), ("d", "e"), ("e", "f"), ("f", "d"), ("c", "d")}
        first = louvain(nodes, edges)
        second = louvain(list(reversed(nodes)), edges)
        assert first == second


class TestAnalyze:
    def test_empty_subgraph(self):
        result = analyze([], set(), {}, config())
        assert result.pagerank == {} and result.communities == {}

    def test_edges_outside_the_node_set_are_ignored(self):
        result = analyze(["a", "b"], {("a", "b"), ("a", "ghost")}, {"a": 2020, "b": 2019}, config())
        assert result.out_degree["a"] == 1
        assert "ghost" not in result.pagerank

    def test_reports_when_there_is_no_structure_to_rank_on(self):
        result = analyze(["a", "b"], set(), {"a": 2020, "b": 2021}, config())
        assert any("No citation edges" in note for note in result.notes)

    def test_counts_undated_papers_in_notes(self):
        result = analyze(["a", "b"], {("a", "b")}, {"a": None, "b": 2019}, config())
        assert any("no publication year" in note for note in result.notes)

    def test_self_citation_is_dropped(self):
        result = analyze(["a"], {("a", "a")}, {"a": 2020}, config())
        assert result.in_degree["a"] == 0
