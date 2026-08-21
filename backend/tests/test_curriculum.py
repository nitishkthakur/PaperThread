"""LLM-planned learning paths: assembly, honesty signals, and confidence.

The strategies themselves need a network and a model, so what is tested here is everything
*around* the model call — the parts that decide whether a wrong answer is visible. Those
are the parts that failed in review: the paths were not merely imperfect, they were
imperfect in a way a reader could not detect.
"""

import pytest

from paperthread.config import (
    Config,
    EmbeddingsConfig,
    ExpansionConfig,
    GraphConfig,
    LayerConfig,
    LLMConfig,
    ProviderCacheConfig,
    RetrievalConfig,
)
from paperthread.domain.identity import canonical_id_for
from paperthread.domain.models import ExternalId, IdNamespace, Paper
from paperthread.domain.path import ExplanationSource, PaperRole
from paperthread.retrieval.curriculum import (
    STAGE_ANCHOR,
    STAGE_FOLLOWUP,
    STAGE_PREREQUISITE,
    BuildContext,
    PlannedStep,
    _assemble,
    build_strategy,
)
from paperthread.domain.path import LearningPath
from pathlib import Path as FsPath


def make_config() -> Config:
    return Config(
        default_user_id="test",
        paper_providers=(),
        llm=LLMConfig(provider="none", roles={}, providers={}, cache_dir=FsPath("/tmp/x")),
        embeddings=EmbeddingsConfig(False, "", "", 0, "main"),
        retrieval=RetrievalConfig(
            candidates_per_provider=5,
            max_candidates=20,
            layers=LayerConfig(llm=True),
            rrf_k=60,
            expansion=ExpansionConfig(),
            graph=GraphConfig(),
        ),
        source_path=FsPath("test.toml"),
        provider_cache=ProviderCacheConfig(enabled=False),
    )


def make_paper(title: str, year: int | None = None) -> Paper:
    paper = Paper(
        canonical_id="",
        title=title,
        external_ids={ExternalId(IdNamespace.DOI, title.lower().replace(" ", "-"))},
        year=year,
    )
    paper.canonical_id = canonical_id_for(paper)
    return paper


def step(title: str, stage: str, year: int | None = 2020, concept: str = "a concept"):
    return PlannedStep(
        paper=make_paper(title, year),
        concept=concept,
        stage=stage,
        why_here="because it comes here",
        position=0,
    )


def assemble(steps, notes=None) -> LearningPath:
    context = BuildContext.create(make_config())
    context.notes = list(notes or [])
    context.models_used = {"test-model"}
    path = LearningPath(topic="a topic")
    _assemble(path, steps, context)
    return path


class TestAssembly:
    def test_steps_are_ordered_and_chained(self):
        path = assemble(
            [
                step("First Paper", STAGE_PREREQUISITE, 2010),
                step("Second Paper", STAGE_ANCHOR, 2015),
                step("Third Paper", STAGE_FOLLOWUP, 2020),
            ]
        )
        assert [s.order for s in path.steps] == [0, 1, 2]
        assert [s.level for s in path.steps] == [0, 1, 2]
        # Each step names its predecessor as prerequisite; the first names nothing.
        assert path.steps[0].prerequisite_ids == ()
        assert path.steps[1].prerequisite_ids == (path.steps[0].paper.canonical_id,)

    def test_stage_maps_to_role(self):
        path = assemble(
            [step("A Paper", STAGE_PREREQUISITE), step("B Paper", STAGE_ANCHOR)]
        )
        assert path.steps[0].role is PaperRole.FOUNDATION
        assert path.steps[1].role is PaperRole.BREAKTHROUGH

    def test_explanations_are_marked_as_reasoned(self):
        path = assemble([step("A Paper", STAGE_ANCHOR)])
        assert path.steps[0].explanation.source is ExplanationSource.LLM
        assert "test-model" in path.steps[0].explanation.provenance.stamp()

    def test_edges_are_created_between_consecutive_steps(self):
        path = assemble(
            [step("A Paper", STAGE_PREREQUISITE), step("B Paper", STAGE_ANCHOR)]
        )
        assert len(path.edges) == 1
        assert path.edges[0].prerequisite_id == path.steps[0].paper.canonical_id


class TestAnchorLoss:
    """A path with no anchor never reaches the topic. That must be the first thing said."""

    def test_missing_anchor_is_reported_first_and_degrades(self):
        path = assemble(
            [step("A Paper", STAGE_PREREQUISITE), step("B Paper", STAGE_PREREQUISITE)]
        )
        assert path.degraded
        assert path.notes[0].startswith("INCOMPLETE")

    def test_present_anchor_does_not_degrade_on_that_basis(self):
        path = assemble(
            [step("A Paper", STAGE_PREREQUISITE), step("B Paper", STAGE_ANCHOR)]
        )
        assert not any(n.startswith("INCOMPLETE") for n in path.notes)


class TestConfidence:
    """Confidence must fall for reasons a reader would agree with, and be built only from
    checkable facts — never from the model's opinion of itself."""

    def full_path(self):
        return [
            step("Prereq One", STAGE_PREREQUISITE, 2010),
            step("Prereq Two", STAGE_PREREQUISITE, 2012),
            step("The Anchor", STAGE_ANCHOR, 2015),
            step("Follow One", STAGE_FOLLOWUP, 2018),
            step("Follow Two", STAGE_FOLLOWUP, 2020),
        ]

    def test_complete_path_scores_high(self):
        path = assemble(self.full_path())
        assert path.confidence >= 0.8
        assert path.confidence_reasons

    def test_missing_anchor_is_the_largest_penalty(self):
        steps = self.full_path()
        steps[2] = step("Not The Anchor", STAGE_FOLLOWUP, 2015)
        path = assemble(steps)
        assert path.confidence < 0.6
        assert any("never reaches the topic" in r for r in path.confidence_reasons)

    def test_no_prerequisites_is_penalised(self):
        path = assemble(
            [step(f"Paper {i}", STAGE_FOLLOWUP, 2015 + i) for i in range(4)]
            + [step("The Anchor", STAGE_ANCHOR, 2014)]
        )
        assert any("reading list, not a path" in r for r in path.confidence_reasons)

    def test_dropped_plan_steps_lower_confidence(self):
        clean = assemble(self.full_path())
        lossy = assemble(
            self.full_path(), notes=["3 planned step(s) named a paper that could not be found"]
        )
        assert lossy.confidence < clean.confidence
        assert any("could not be found" in r for r in lossy.confidence_reasons)

    def test_approximate_matches_lower_confidence(self):
        steps = self.full_path()
        steps[0] = step(
            "Some Paper", STAGE_PREREQUISITE, 2010, concept="Closest available match for: X"
        )
        path = assemble(steps)
        assert any("approximate match" in r for r in path.confidence_reasons)

    def test_short_path_lowers_confidence(self):
        short = assemble(
            [step("Prereq", STAGE_PREREQUISITE, 2010), step("Anchor", STAGE_ANCHOR, 2015)]
        )
        assert any("only 2 steps" in r for r in short.confidence_reasons)

    def test_confidence_stays_in_range(self):
        worst = assemble(
            [step("Only", STAGE_FOLLOWUP, 2020)],
            notes=["planned step(s) named a paper that could not be found"],
        )
        assert 0.0 <= worst.confidence <= 1.0


class TestChronology:
    def test_reverse_chronological_sequence_is_flagged(self):
        path = assemble(
            [
                step("Newest", STAGE_PREREQUISITE, 2022),
                step("Older", STAGE_PREREQUISITE, 2012),
                step("Oldest", STAGE_ANCHOR, 2002),
            ]
        )
        assert any("backwards in time" in n for n in path.notes)

    def test_forward_sequence_is_not_flagged(self):
        path = assemble(
            [
                step("Oldest", STAGE_PREREQUISITE, 2002),
                step("Older", STAGE_PREREQUISITE, 2012),
                step("Newest", STAGE_ANCHOR, 2022),
            ]
        )
        assert not any("backwards in time" in n for n in path.notes)


class TestStrategyRegistry:
    @pytest.mark.parametrize("name", ["syllabus", "anchor", "rerank", "hybrid"])
    def test_known_strategies_build(self, name):
        assert build_strategy(name, BuildContext.create(make_config())).name

    def test_unknown_strategy_names_the_alternatives(self):
        with pytest.raises(ValueError, match="syllabus"):
            build_strategy("nonsense", BuildContext.create(make_config()))
