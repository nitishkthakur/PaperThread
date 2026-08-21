from paperthread.domain.identity import canonical_id_for
from paperthread.domain.models import ExternalId, IdNamespace, Paper, SearchHit
from paperthread.providers.papers.openalex import reconstruct_abstract
from paperthread.retrieval.fusion import reciprocal_rank_fusion


def paper(title, doi=None, arxiv=None):
    ids = set()
    if doi:
        ids.add(ExternalId(IdNamespace.DOI, doi))
    if arxiv:
        ids.add(ExternalId(IdNamespace.ARXIV, arxiv))
    p = Paper(canonical_id="", title=title, external_ids=ids)
    p.canonical_id = canonical_id_for(p)
    return p


def hits(provider, *papers):
    return [SearchHit(paper=p, rank=i, provider=provider) for i, p in enumerate(papers, 1)]


class TestReciprocalRankFusion:
    def test_empty_input(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list_preserves_order(self):
        a, b, c = paper("Alpha One", doi="1"), paper("Beta Two", doi="2"), paper("Gamma", doi="3")
        result = reciprocal_rank_fusion([hits("p1", a, b, c)])
        assert [r.paper.title for r in result] == ["Alpha One", "Beta Two", "Gamma"]

    def test_agreement_across_providers_wins(self):
        """A paper ranked mid-list by both providers should beat one ranked #1 by one."""
        shared = paper("Shared Paper Title", doi="10.1/shared")
        only_a = paper("Only In A List", doi="10.1/a")
        only_b = paper("Only In B List", doi="10.1/b")

        result = reciprocal_rank_fusion(
            [hits("a", only_a, shared), hits("b", only_b, shared)], k=1
        )
        assert result[0].paper.title == "Shared Paper Title"
        assert sorted(result[0].found_by) == ["a", "b"]
        assert result[0].ranks == {"a": 2, "b": 2}

    def test_duplicates_are_merged_before_scoring(self):
        """Same work from two providers must score as one paper found twice."""
        from_a = paper("Denoising Diffusion Models", arxiv="2006.11239")
        from_b = paper("Denoising diffusion models", doi="10.5555/x")
        result = reciprocal_rank_fusion([hits("a", from_a), hits("b", from_b)])
        assert len(result) == 1
        assert sorted(result[0].found_by) == ["a", "b"]

    def test_within_provider_duplicate_does_not_double_count(self):
        """A provider returning preprint + published must not inflate that paper's score."""
        preprint = paper("Same Work Title Here", arxiv="2006.11239")
        published = paper("Same work title here", doi="10.5555/y")
        other = paper("A Different Paper", doi="10.5555/z")

        dup = reciprocal_rank_fusion([hits("a", preprint, published)], k=60)
        single = reciprocal_rank_fusion([hits("a", preprint, other)], k=60)
        # Best rank (1) is kept in both cases, so the merged paper scores identically.
        assert dup[0].score == single[0].score
        assert dup[0].ranks == {"a": 1}

    def test_ordering_is_deterministic_for_equal_scores(self):
        a, b = paper("Zebra Paper", doi="1"), paper("Alpha Paper", doi="2")
        first = reciprocal_rank_fusion([hits("p", a), hits("q", b)])
        second = reciprocal_rank_fusion([hits("p", a), hits("q", b)])
        assert [r.paper.title for r in first] == [r.paper.title for r in second]

    def test_k_damps_top_rank_dominance(self):
        top = paper("Ranked First", doi="1")
        second = paper("Ranked Second", doi="2")
        small_k = reciprocal_rank_fusion([hits("p", top, second)], k=1)
        large_k = reciprocal_rank_fusion([hits("p", top, second)], k=1000)
        assert small_k[0].score / small_k[1].score > large_k[0].score / large_k[1].score


class TestOpenAlexAbstractReconstruction:
    def test_reconstructs_word_order(self):
        inverted = {"Diffusion": [0], "models": [1, 4], "beat": [2], "other": [3]}
        assert reconstruct_abstract(inverted) == "Diffusion models beat other models"

    def test_missing_or_empty_returns_none(self):
        assert reconstruct_abstract(None) is None
        assert reconstruct_abstract({}) is None
