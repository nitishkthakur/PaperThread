"""Stage 2 — citation-graph expansion.

The network is the boring part; **identity is the correctness risk**. References arrive as
fresh provider records that know nothing about the candidate set, so the same work turns up
as a preprint down one path and a published record down another. Unreconciled, one ancestor
becomes three nodes and its co-citation count — the entire ancestor signal — splits three
ways (PROVIDER_NOTES C4).

No test here touches the network: providers are fakes implementing the same port.
"""

import pytest

from paperthread.config import (
    Config,
    EmbeddingsConfig,
    ExpansionConfig,
    GraphConfig,
    LayerConfig,
    LLMConfig,
    PaperProviderConfig,
    RetrievalConfig,
)
from paperthread.domain.identity import canonical_id_for
from paperthread.domain.models import ExternalId, IdNamespace, Paper, RankedPaper
from paperthread.providers.base import BasePaperProvider, ProviderError
from paperthread.retrieval.expansion import CitationExpansionService
from pathlib import Path


def make_paper(title, *, doi=None, arxiv=None, openalex=None, year=None) -> Paper:
    ids = set()
    if doi:
        ids.add(ExternalId(IdNamespace.DOI, doi))
    if arxiv:
        ids.add(ExternalId(IdNamespace.ARXIV, arxiv))
    if openalex:
        ids.add(ExternalId(IdNamespace.OPENALEX, openalex))
    paper = Paper(canonical_id="", title=title, external_ids=ids, year=year)
    paper.canonical_id = canonical_id_for(paper)
    return paper


def ranked(*papers) -> list[RankedPaper]:
    return [RankedPaper(paper=p, score=1.0, found_by=["fake"], ranks={}) for p in papers]


def make_config(**expansion) -> Config:
    settings = {
        "seed_papers": 10,
        "references_per_paper": 10,
        "min_co_citations": 2,
        "max_ancestors": 5,
        "expand_ancestors": False,
        "forward_enabled": False,
        **expansion,
    }
    return Config(
        default_user_id="test",
        paper_providers=(),
        llm=LLMConfig(provider="none", roles={}, providers={}, cache_dir=Path("/tmp/nope")),
        embeddings=EmbeddingsConfig(False, "", "", 0, "main"),
        retrieval=RetrievalConfig(
            candidates_per_provider=10,
            max_candidates=50,
            layers=LayerConfig(),
            rrf_k=60,
            expansion=ExpansionConfig(**settings),
            graph=GraphConfig(),
        ),
        source_path=Path("test.toml"),
    )


class FakeCitationProvider(BasePaperProvider):
    """Implements the `citations` capability from a canned reference table."""

    def __init__(self, name, references, fail=False):
        super().__init__(
            PaperProviderConfig(
                name=name, enabled=True, capabilities=frozenset({"citations"}),
                rate_limit_per_sec=0.0,
            )
        )
        self._references = references
        self._fail = fail

    async def references(self, paper, limit):
        if self._fail:
            raise ProviderError(self.name, "boom")
        return list(self._references.get(paper.canonical_id, []))[:limit]

    async def citations(self, paper, limit, query=None):
        return []


class TestExpansion:
    async def test_disabled_is_a_no_op_that_says_so(self):
        service = CitationExpansionService(make_config(enabled=False), providers=[])
        corpus = await service.expand("topic", ranked(make_paper("A", doi="1")))
        assert not corpus.ran
        assert any("disabled" in note for note in corpus.notes)

    async def test_no_citation_provider_degrades_with_an_explanation(self):
        service = CitationExpansionService(make_config(), providers=[])
        corpus = await service.expand("topic", ranked(make_paper("A", doi="1")))
        assert not corpus.ran
        assert any("citations" in note for note in corpus.notes)

    async def test_co_citation_counts_distinct_seeds(self):
        ancestor = make_paper("Shared Ancestor Paper", doi="anc", year=2010)
        s1 = make_paper("Seed One Title", doi="s1", year=2020)
        s2 = make_paper("Seed Two Title", doi="s2", year=2021)
        provider = FakeCitationProvider(
            "fake", {s1.canonical_id: [ancestor], s2.canonical_id: [ancestor]}
        )
        service = CitationExpansionService(make_config(), providers=[provider])

        corpus = await service.expand("topic", ranked(s1, s2))

        assert corpus.co_citations[ancestor.canonical_id] == 2
        assert ancestor.canonical_id in corpus.discovered_ids

    async def test_preprint_and_published_ancestor_merge_into_one_node(self):
        """The signal-splitting case. Two seeds cite what is really the same work, one via
        its arXiv preprint and one via its published record. If they do not merge, the
        ancestor scores 1 + 1 instead of 2 and drops below the threshold that surfaces it.
        """
        preprint = make_paper("Attention Is All You Need", arxiv="1706.03762", year=2017)
        published = make_paper("Attention is all you need", doi="10.5555/xyz", year=2017)
        s1 = make_paper("Seed One Title", doi="s1", year=2020)
        s2 = make_paper("Seed Two Title", doi="s2", year=2021)
        provider = FakeCitationProvider(
            "fake", {s1.canonical_id: [preprint], s2.canonical_id: [published]}
        )
        service = CitationExpansionService(make_config(), providers=[provider])

        corpus = await service.expand("topic", ranked(s1, s2))

        ancestors = [p for p in corpus.papers if p in corpus.discovered_ids]
        assert len(ancestors) == 1, "preprint and published record must reconcile"
        assert corpus.co_citations[ancestors[0]] == 2

    async def test_self_edge_after_merging_is_dropped(self):
        """A seed's reference list containing its own preprint becomes a self-edge once
        they merge — and a self-edge is a one-node cycle."""
        seed = make_paper("Some Distinctive Title", arxiv="2301.00001", year=2023)
        same_work = make_paper("Some Distinctive Title", doi="10.1/abc", year=2023)
        provider = FakeCitationProvider("fake", {seed.canonical_id: [same_work]})
        service = CitationExpansionService(make_config(), providers=[provider])

        corpus = await service.expand("topic", ranked(seed))

        assert all(a != b for a, b in corpus.edge_pairs())

    async def test_provider_failure_degrades_rather_than_raising(self):
        seed = make_paper("Seed One Title", doi="s1", year=2020)
        good = FakeCitationProvider("good", {seed.canonical_id: [make_paper("Ref", doi="r1")]})
        bad = FakeCitationProvider("bad", {}, fail=True)
        service = CitationExpansionService(make_config(), providers=[good, bad])

        corpus = await service.expand("topic", ranked(seed))

        assert corpus.ran
        assert {o.provider: o.ok for o in corpus.outcomes} == {"good": True, "bad": False}
        assert any("Degraded" in note for note in corpus.notes)

    async def test_seed_ids_survive_reconciliation(self):
        seed = make_paper("Seed One Title", doi="s1", year=2020)
        provider = FakeCitationProvider("fake", {seed.canonical_id: []})
        service = CitationExpansionService(make_config(), providers=[provider])

        corpus = await service.expand("topic", ranked(seed))

        assert corpus.seed_ids and all(sid in corpus.papers for sid in corpus.seed_ids)

    async def test_empty_candidate_set(self):
        provider = FakeCitationProvider("fake", {})
        service = CitationExpansionService(make_config(), providers=[provider])
        corpus = await service.expand("topic", [])
        assert corpus.papers == {}
