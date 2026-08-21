"""Deduplication is core correctness, not cleanup — so it gets real tests.

The preprint/published pair is the case that matters: unreconciled it produces duplicate
nodes in a learning path and splits citation counts, which corrupts the centrality scoring
that ranking depends on (PROVIDER_NOTES C4).
"""

from paperthread.domain.identity import (
    canonical_id_for,
    deduplicate,
    normalize_arxiv_id,
    normalize_doi,
    title_fingerprint,
)
from paperthread.domain.models import ExternalId, IdNamespace, Paper


def make_paper(title, *, ids=(), year=None, abstract=None, citations=None):
    paper = Paper(
        canonical_id="",
        title=title,
        external_ids={ExternalId(ns, v) for ns, v in ids},
        year=year,
        abstract=abstract,
        citation_count=citations,
    )
    paper.canonical_id = canonical_id_for(paper)
    return paper


class TestNormalization:
    def test_arxiv_version_suffix_is_stripped(self):
        assert normalize_arxiv_id("2301.12345v3") == "2301.12345"
        assert normalize_arxiv_id("arXiv:2301.12345") == "2301.12345"
        assert normalize_arxiv_id("https://arxiv.org/abs/2301.12345v2") == "2301.12345"
        assert normalize_arxiv_id("https://arxiv.org/pdf/2301.12345.pdf") == "2301.12345"

    def test_doi_prefixes_are_stripped_and_lowercased(self):
        expected = "10.1000/abc"
        assert normalize_doi("https://doi.org/10.1000/ABC") == expected
        assert normalize_doi("doi:10.1000/abc") == expected
        assert normalize_doi("  10.1000/Abc  ") == expected

    def test_title_fingerprint_ignores_case_punctuation_and_accents(self):
        assert title_fingerprint("Attention Is All You Need") == title_fingerprint(
            "attention is all you need."
        )
        assert title_fingerprint("Schrödinger") == title_fingerprint("Schrodinger")


class TestDeduplicate:
    def test_same_doi_merges(self):
        a = make_paper("A Paper", ids=[(IdNamespace.DOI, "10.1/x")], abstract=None)
        b = make_paper("A Paper", ids=[(IdNamespace.DOI, "10.1/x")], abstract="hello")
        merged = deduplicate([a, b])
        assert len(merged) == 1
        assert merged[0].abstract == "hello"

    def test_arxiv_versions_merge(self):
        from paperthread.domain.identity import normalize_external_id

        a = Paper(canonical_id="", title="Same Work Here", external_ids={
            normalize_external_id(IdNamespace.ARXIV, "2301.12345v1")})
        b = Paper(canonical_id="", title="Same Work Here", external_ids={
            normalize_external_id(IdNamespace.ARXIV, "2301.12345v4")})
        a.canonical_id, b.canonical_id = canonical_id_for(a), canonical_id_for(b)
        assert len(deduplicate([a, b])) == 1

    def test_preprint_and_published_merge_on_title_despite_no_shared_id(self):
        """The critical case: no shared identifier at all."""
        preprint = make_paper(
            "Denoising Diffusion Probabilistic Models",
            ids=[(IdNamespace.ARXIV, "2006.11239")],
            year=2020,
            citations=100,
        )
        published = make_paper(
            "Denoising diffusion probabilistic models.",
            ids=[(IdNamespace.DOI, "10.5555/xyz")],
            year=2020,
            citations=8000,
        )
        merged = deduplicate([preprint, published])
        assert len(merged) == 1
        # Citation counts are split across versions; keeping the max avoids under-ranking
        # a foundational paper because its influence is spread over two records.
        assert merged[0].citation_count == 8000
        assert len(merged[0].external_ids) == 2

    def test_transitive_merge_across_three_providers(self):
        """A~B via DOI, B~C via title: all three must collapse."""
        a = make_paper("Some Title Long Enough", ids=[(IdNamespace.DOI, "10.1/z")], year=2020)
        b = make_paper(
            "Some Title Long Enough",
            ids=[(IdNamespace.DOI, "10.1/z"), (IdNamespace.OPENALEX, "W1")],
            year=2020,
        )
        c = make_paper("Some Title Long Enough", ids=[(IdNamespace.S2, "s2abc")], year=2020)
        assert len(deduplicate([a, b, c])) == 1

    def test_different_papers_stay_separate(self):
        a = make_paper("Attention Is All You Need", ids=[(IdNamespace.ARXIV, "1706.03762")])
        b = make_paper("Deep Residual Learning", ids=[(IdNamespace.ARXIV, "1512.03385")])
        assert len(deduplicate([a, b])) == 2

    def test_same_title_different_year_does_not_merge_without_shared_id(self):
        a = make_paper("A Survey of Methods", year=2018)
        b = make_paper("A Survey of Methods", year=2024)
        assert len(deduplicate([a, b])) == 2

    def test_paper_with_no_identifier_still_gets_stable_id(self):
        a = make_paper("Some Old Paper With No Identifiers", year=1958)
        b = make_paper("Some Old Paper With No Identifiers", year=1958)
        assert a.canonical_id == b.canonical_id
        assert a.canonical_id.startswith("sig:")

    def test_input_order_is_preserved(self):
        a = make_paper("First Paper Title Here", ids=[(IdNamespace.DOI, "10.1/a")])
        b = make_paper("Second Paper Title Here", ids=[(IdNamespace.DOI, "10.1/b")])
        assert [p.title for p in deduplicate([a, b])] == [a.title, b.title]

    def test_papers_without_abstracts_are_retained(self):
        """Never filter on missing abstract — coverage is worst for older, foundational
        papers, which are exactly the ones §5 exists to surface (PROVIDER_NOTES C2)."""
        with_abstract = make_paper("Recent Paper Title", ids=[(IdNamespace.DOI, "10.1/new")],
                                   abstract="text")
        without = make_paper("Ancient Foundational Paper", ids=[(IdNamespace.DOI, "10.1/old")])
        merged = deduplicate([with_abstract, without])
        assert len(merged) == 2
        assert any(not p.has_abstract for p in merged)


class TestYearWindow:
    """Regression tests for the merge-key year window."""

    def test_preprint_and_published_one_year_apart_merge(self):
        preprint = make_paper("Denoising Diffusion Probabilistic Models",
                              ids=[(IdNamespace.ARXIV, "2006.11239")], year=2020)
        published = make_paper("Denoising Diffusion Probabilistic Models",
                               ids=[(IdNamespace.DOI, "10.5555/x")], year=2021)
        assert len(deduplicate([preprint, published])) == 1

    def test_six_years_apart_does_not_merge(self):
        a = make_paper("A Survey of Methods In Things", year=2018)
        b = make_paper("A Survey of Methods In Things", year=2024)
        assert len(deduplicate([a, b])) == 2

    def test_undated_record_absorbed_when_unambiguous(self):
        dated = make_paper("Some Distinctive Paper Title", ids=[(IdNamespace.DOI, "10.1/a")],
                           year=2020, abstract="body")
        undated = make_paper("Some distinctive paper title", ids=[(IdNamespace.S2, "s2x")])
        merged = deduplicate([dated, undated])
        assert len(merged) == 1
        assert len(merged[0].external_ids) == 2

    def test_undated_record_left_alone_when_ambiguous(self):
        """Two eras with the same title: we must not guess which one it belongs to."""
        old = make_paper("A Survey of Methods In Things", ids=[(IdNamespace.DOI, "10.1/o")],
                         year=2010)
        new = make_paper("A Survey of Methods In Things", ids=[(IdNamespace.DOI, "10.1/n")],
                         year=2024)
        undated = make_paper("A survey of methods in things", ids=[(IdNamespace.S2, "s2y")])
        assert len(deduplicate([old, new, undated])) == 3
