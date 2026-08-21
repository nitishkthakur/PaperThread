"""Selection quotas and the structural (L0) half of Stage 4.

Both files under test decide what a user sees, and both had real bugs during the build that
these lock down:

* selection promoted 1960s papers on osmotic flow through cellulose acetate membranes into
  a path about diffusion models, because a lexical false positive dragged its ancestry in
  and age-rescaling flattered anything with no age peers;
* the structural explanation claimed a paper "matched the topic directly" and that "keyword
  search did not return it" in the same paragraph.

Neither raised. Both are only visible by reading the output, which is exactly why they need
tests.
"""

from paperthread.config import GraphConfig
from paperthread.domain.identity import canonical_id_for
from paperthread.domain.models import ExternalId, IdNamespace, Paper
from paperthread.domain.path import EdgeSource, ExplanationSource, PaperRole
from paperthread.retrieval.expansion import CitationEdge, ExpandedCorpus
from paperthread.retrieval.graph import analyze
from paperthread.retrieval.judgment import JudgmentService
from paperthread.retrieval.selection import candidate_pairs, select_path_papers


def make_paper(title, year=None, abstract=None) -> Paper:
    paper = Paper(
        canonical_id="",
        title=title,
        external_ids={ExternalId(IdNamespace.DOI, title.lower().replace(" ", "-"))},
        year=year,
        abstract=abstract,
    )
    paper.canonical_id = canonical_id_for(paper)
    return paper


def build_corpus(papers, edges=(), seeds=(), co_citations=None) -> ExpandedCorpus:
    by_id = {p.canonical_id: p for p in papers}
    seed_ids = [p.canonical_id for p in seeds]
    return ExpandedCorpus(
        papers=by_id,
        # Real edge objects rather than a stubbed `edge_pairs`, so these tests exercise the
        # same accessor the pipeline does.
        edges=[CitationEdge(citing, cited, "test") for citing, cited in edges],
        seed_ids=seed_ids,
        discovered_ids=set(by_id) - set(seed_ids),
        co_citations=co_citations or {},
        ran=True,
    )


class TestSelection:
    def test_empty_corpus(self):
        corpus = build_corpus([])
        result = select_path_papers(corpus, analyze([], set(), {}, GraphConfig()), 10, 2)
        assert result.paper_ids == []

    def test_zero_budget(self):
        seed = make_paper("Seed", 2020)
        corpus = build_corpus([seed], seeds=[seed])
        result = select_path_papers(corpus, analyze([seed.canonical_id], set(), {}, GraphConfig()), 0, 2)
        assert result.paper_ids == []

    def test_foundation_and_surface_both_get_slots(self):
        """Selecting purely by centrality returns famous old papers and never reaches what
        the user asked about; selecting purely by search rank returns no foundation at all.
        """
        ancestor = make_paper("Shared Ancestor", 2010)
        seeds = [make_paper(f"Recent Work {i}", 2023) for i in range(8)]
        papers = [ancestor, *seeds]
        edges = {(s.canonical_id, ancestor.canonical_id) for s in seeds}
        corpus = build_corpus(
            papers, edges, seeds, co_citations={ancestor.canonical_id: len(seeds)}
        )
        analysis = analyze(
            [p.canonical_id for p in papers], edges, {p.canonical_id: p.year for p in papers},
            GraphConfig(),
        )

        result = select_path_papers(corpus, analysis, 6, 2)

        assert ancestor.canonical_id in result.paper_ids
        assert any(s.canonical_id in result.paper_ids for s in seeds)

    def test_weakly_connected_noise_is_excluded(self):
        """The osmotic-membrane regression. A paper reached by a single citation from a
        single candidate, with no age peers to be compared against, must not be promoted
        into the path on the strength of an inflated z-score.
        """
        seed = make_paper("Lexical False Positive", 1940)
        noise = make_paper("Transport Properties Of Cellulose Acetate", 1965)
        edges = {(seed.canonical_id, noise.canonical_id)}
        corpus = build_corpus(
            [seed, noise], edges, [seed], co_citations={noise.canonical_id: 1}
        )
        analysis = analyze(
            [seed.canonical_id, noise.canonical_id], edges,
            {seed.canonical_id: 1940, noise.canonical_id: 1965}, GraphConfig(),
        )

        result = select_path_papers(corpus, analysis, 10, 3)

        assert noise.canonical_id not in result.paper_ids
        assert seed.canonical_id in result.paper_ids, "the seed itself still matched the query"

    def test_deep_ancestor_with_no_co_citations_survives_on_in_degree(self):
        """Papers reached in the second expansion round legitimately have zero
        co-citations, because that counter only counts citations from Stage 1 candidates.
        """
        deep = make_paper("Deep Foundation", 1986)
        mids = [make_paper(f"Mid Paper {i}", 2015) for i in range(3)]
        seed = make_paper("Surface Paper", 2023)
        papers = [deep, *mids, seed]
        edges = {(m.canonical_id, deep.canonical_id) for m in mids}
        corpus = build_corpus(papers, edges, [seed], co_citations={})
        analysis = analyze(
            [p.canonical_id for p in papers], edges,
            {p.canonical_id: p.year for p in papers}, GraphConfig(),
        )

        result = select_path_papers(corpus, analysis, 10, 3)

        assert deep.canonical_id in result.paper_ids

    def test_reports_when_the_topic_has_no_evidenced_foundation(self):
        seed = make_paper("Only Paper", 2024)
        corpus = build_corpus([seed], seeds=[seed])
        analysis = analyze([seed.canonical_id], set(), {seed.canonical_id: 2024}, GraphConfig())
        result = select_path_papers(corpus, analysis, 10, 3)
        assert any("no evidenced foundation" in note for note in result.notes)


class TestCandidatePairs:
    def test_pair_where_the_cited_paper_is_newer_is_dropped(self):
        """A paper cannot cite its own future. This is provider data mixing a published
        year against a preprint year, not a real edge."""
        old, new = make_paper("Older", 2015), make_paper("Newer", 2022)
        papers = {p.canonical_id: p for p in (old, new)}
        edges = {(old.canonical_id, new.canonical_id)}  # 2015 paper "cites" a 2022 paper
        analysis = analyze(list(papers), edges, {p: papers[p].year for p in papers}, GraphConfig())

        pairs = candidate_pairs(list(papers), edges, papers, {}, analysis, 10)

        assert pairs == []

    def test_ranked_by_how_much_the_prerequisite_looks_like_shared_foundation(self):
        shared = make_paper("Widely Cited", 2010)
        incidental = make_paper("Cited Once", 2010)
        citer = make_paper("Citing Paper", 2022)
        papers = {p.canonical_id: p for p in (shared, incidental, citer)}
        edges = {
            (citer.canonical_id, shared.canonical_id),
            (citer.canonical_id, incidental.canonical_id),
        }
        analysis = analyze(list(papers), edges, {p: papers[p].year for p in papers}, GraphConfig())

        pairs = candidate_pairs(
            list(papers), edges, papers,
            {shared.canonical_id: 9, incidental.canonical_id: 1}, analysis, 10,
        )

        assert pairs[0].prerequisite_id == shared.canonical_id

    def test_limit_is_respected(self):
        papers = {p.canonical_id: p for p in (make_paper(f"P{i}", 2000 + i) for i in range(6))}
        ids = sorted(papers)
        edges = {(ids[-1], other) for other in ids[:-1]}
        analysis = analyze(ids, edges, {p: papers[p].year for p in papers}, GraphConfig())
        assert len(candidate_pairs(ids, edges, papers, {}, analysis, 2)) == 2


class TestStructuralJudgment:
    async def build(self, papers, edges, seeds, co_citations):
        corpus = build_corpus(papers, edges, seeds, co_citations)
        ids = [p.canonical_id for p in papers]
        analysis = analyze(ids, edges, {p.canonical_id: p.year for p in papers}, GraphConfig())
        pairs = candidate_pairs(ids, edges, corpus.papers, co_citations, analysis, 20)
        assessment = await JudgmentService(client=None).assess(
            "topic", ids, corpus, analysis, pairs, "a reader"
        )
        return assessment, corpus, analysis

    async def test_every_paper_gets_all_four_explanation_fields(self):
        """§5 is a hard requirement, not a best effort. Structural output must satisfy it."""
        papers = [make_paper("Alpha Paper", 2015, "First sentence. Second sentence."),
                  make_paper("Beta Paper", 2022)]
        assessment, _, _ = await self.build(papers, set(), papers, {})

        for paper in papers:
            explanation = assessment.explanations[paper.canonical_id]
            assert explanation.why_it_matters
            assert explanation.what_it_assumes
            assert explanation.what_it_teaches
            assert explanation.why_for_you
            assert explanation.source is ExplanationSource.STRUCTURAL

    async def test_explanation_does_not_contradict_itself(self):
        """Regression: a paper reached by expansion was told it "matched the topic
        directly" and that "keyword search did not return it" in the same breath."""
        seed = make_paper("Searched Paper", 2023)
        expanded = make_paper("Expanded Paper", 2018)
        edges = {(seed.canonical_id, expanded.canonical_id)}
        assessment, _, _ = await self.build([seed, expanded], edges, [seed], {})

        text = assessment.explanations[expanded.canonical_id].why_it_matters
        assert not ("matched the topic directly" in text and "did not return it" in text)

    async def test_expanded_paper_says_search_missed_it(self):
        seed = make_paper("Searched Paper", 2023)
        expanded = make_paper("Expanded Paper", 2018)
        edges = {(seed.canonical_id, expanded.canonical_id)}
        assessment, _, _ = await self.build([seed, expanded], edges, [seed], {})
        assert "did not return it" in assessment.explanations[expanded.canonical_id].why_it_matters

    async def test_surveys_and_critiques_are_recognised_from_their_titles(self):
        survey = make_paper("Diffusion Models: A Comprehensive Survey", 2022)
        critique = make_paper("Rethinking the Value of Network Pruning", 2019)
        assessment, _, _ = await self.build([survey, critique], set(), [survey, critique], {})
        assert assessment.roles[survey.canonical_id] is PaperRole.SURVEY
        assert assessment.roles[critique.canonical_id] is PaperRole.CRITIQUE

    async def test_structural_edges_are_labelled_as_co_citation_not_judgment(self):
        """A structural edge is a weaker claim than a judged one and must not masquerade
        as it — the provenance is what lets the UI say which it is showing."""
        ancestor = make_paper("Shared Ancestor", 2010)
        seeds = [make_paper(f"Seed {i}", 2022) for i in range(4)]
        edges = {(s.canonical_id, ancestor.canonical_id) for s in seeds}
        assessment, _, _ = await self.build(
            [ancestor, *seeds], edges, seeds, {ancestor.canonical_id: 4}
        )
        assert assessment.prerequisite_edges
        assert all(e.source is EdgeSource.CO_CITATION for e in assessment.prerequisite_edges)
        assert all(e.confidence is None for e in assessment.prerequisite_edges)

    async def test_reports_that_l4_did_not_run(self):
        paper = make_paper("Only Paper", 2020)
        assessment, _, _ = await self.build([paper], set(), [paper], {})
        assert not assessment.used_llm
        assert any("L4 inactive" in note for note in assessment.notes)

    async def test_paper_with_no_abstract_says_so_rather_than_inventing_content(self):
        paper = make_paper("No Abstract Paper", 1995)
        assessment, _, _ = await self.build([paper], set(), [paper], {})
        assert "No abstract" in assessment.explanations[paper.canonical_id].what_it_teaches
