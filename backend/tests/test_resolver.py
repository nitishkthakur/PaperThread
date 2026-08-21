"""Grounding a paper an LLM named in a paper that exists.

Every test here is a regression for a failure that **shipped to a learner**, and none of
them raised an exception. That is the shape of the whole problem: a wrong paper and a right
paper are the same type, render identically, and read plausibly.

The two failure directions pull against each other, which is why the cases are kept
together — tightening the matcher to stop one has twice now broken the other:

* **Substitution** — a different paper accepted in place of the named one. TimeSformer
  shipped as the anchor of a "transformers" path, carrying the sentence "this is the paper
  that defines the Transformer".
* **Non-resolution** — the real paper rejected. Tightening the floor to stop TimeSformer
  also rejected Goodfellow's GAN paper, which OpenAlex indexes as "…Nets" against a query
  saying "…Networks".

No test here touches the network.
"""

import pytest

from paperthread.domain.models import Paper
from paperthread.retrieval.resolver import (
    MIN_COVERAGE,
    MIN_SIMILARITY,
    looks_like_a_paper,
    main_title,
    title_similarity,
    tokenize,
)


def accepts(query: str, candidate: str) -> bool:
    """The resolver's actual acceptance rule, both gates."""
    from paperthread.retrieval.resolver import _coverage

    return (
        title_similarity(query, candidate) >= MIN_SIMILARITY
        and _coverage(query, candidate) >= MIN_COVERAGE
    )


class TestAcceptsTheRightPaper:
    def test_exact_match(self):
        assert accepts("Attention Is All You Need", "Attention is All you Need")

    def test_paper_named_by_its_main_clause(self):
        """A planner naming a paper without its subtitle is naming the same paper."""
        assert accepts(
            "Dropout", "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
        )

    def test_morphological_variant(self):
        """OpenAlex indexes Goodfellow as "Nets"; everyone calls it "Networks".

        Regression: tightening the floor to stop substitution rejected the real paper.
        """
        assert accepts("Generative Adversarial Networks", "Generative Adversarial Nets")

    def test_trailing_annotation(self):
        assert accepts(
            "Deep Residual Learning for Image Recognition",
            "Deep Residual Learning for Image Recognition (Extended Abstract)",
        )

    def test_query_naming_a_later_clause_of_the_real_title(self):
        assert accepts(
            "Preventing co-adaptation of feature detectors",
            "Improving neural networks by preventing co-adaptation of feature detectors",
        )


class TestRejectsTheWrongPaper:
    def test_wrapped_query_is_a_different_paper(self):
        """THE regression. Every content word of Vaswani's title is inside TimeSformer's.

        Containment cannot separate these; what does is that the extra words are wrapped
        around the query rather than following a colon.
        """
        assert not accepts(
            "Attention Is All You Need",
            "Is Space-Time Attention All You Need for Video Understanding?",
        )

    @pytest.mark.parametrize(
        "candidate",
        [
            "Conditional Generative Adversarial Nets",
            "Wasserstein Generative Adversarial Networks",
            "Deep Convolutional Generative Adversarial Networks",
        ],
    )
    def test_qualifier_prefix_names_a_different_paper(self, candidate):
        """In this literature a one-word qualifier marks a variant, not the original."""
        assert not accepts("Generative Adversarial Networks", candidate)

    def test_missing_distinctive_words(self):
        assert not accepts(
            "Training language models to follow instructions with human feedback",
            "InstructPatentGPT: Training patent language models to follow instructions",
        )

    def test_shared_prefix_different_paper(self):
        assert not accepts(
            "Deep Reinforcement Learning from Human Preferences",
            "Deep Reinforcement Learning: An Overview",
        )

    def test_same_main_title_different_subtitle_is_a_different_paper(self):
        """Both sides carry a subtitle, so the subtitle is the discriminator.

        Regression: the subtitle rule matched on main title alone and accepted a 2025
        paper as Srivastava 2014 at 0.95 confidence. Eleven years and two research groups
        apart, identical up to the colon.
        """
        assert not accepts(
            "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "Dropout: An Effective Approach to Prevent Neural Networks from Overfitting",
        )

    def test_discriminating_adjectives_are_not_stopwords(self):
        """"simple", "effective" and "approach" look like filler and are not.

        Deleting them as low-signal made the two Dropout papers above identical.
        """
        assert tokenize("A Simple Way") != tokenize("An Effective Approach")

    def test_unrelated_titles(self):
        assert not accepts(
            "Attention Is All You Need",
            "Neural Machine Translation by Jointly Learning to Align and Translate",
        )


class TestNonPapers:
    """Providers index things alongside papers that a learner cannot read."""

    def make(self, title: str, authors=("A. Author",)) -> Paper:
        return Paper(canonical_id="x", title=title, authors=list(authors))

    @pytest.mark.parametrize(
        "title",
        [
            "Convolutional Neural Networks On Graphs (Gdl Seminar)",
            "Lecture Notes on Deep Learning",
            "Reading Group: Attention Mechanisms",
            "Corrigendum to Deep Learning",
        ],
    )
    def test_rejects_non_papers(self, title):
        assert not looks_like_a_paper(self.make(title))

    def test_rejects_records_with_no_authors(self):
        assert not looks_like_a_paper(self.make("Deep Residual Learning", authors=()))

    def test_keeps_real_papers(self):
        assert looks_like_a_paper(self.make("Attention Is All You Need"))

    def test_keeps_a_paper_whose_title_merely_mentions_a_talk_topic(self):
        """`presentation` in a technical sense must not be filtered as a talk record."""
        assert looks_like_a_paper(self.make("Learning Distributed Representations"))


class TestHelpers:
    def test_main_title_strips_subtitle(self):
        assert main_title("Dropout: A Simple Way to Prevent Overfitting") == "Dropout"

    def test_main_title_without_colon_is_the_whole_title(self):
        assert main_title("Attention Is All You Need") == "Attention Is All You Need"

    def test_tokenize_drops_stopwords_and_normalises_variants(self):
        assert tokenize("Generative Adversarial Nets") == tokenize(
            "Generative Adversarial Networks"
        )

    def test_tokenize_keeps_stopwords_when_nothing_else_remains(self):
        """A title made entirely of stopwords must not tokenize to the empty set, or it
        would match everything."""
        assert tokenize("The Of And") != set()

    def test_similarity_is_bounded(self):
        assert 0.0 <= title_similarity("a b c", "d e f") <= 1.0
        assert title_similarity("", "anything") == 0.0
