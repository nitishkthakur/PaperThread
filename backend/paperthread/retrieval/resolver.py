"""Turning a paper an LLM *named* into a paper that actually exists.

This module is the safety rail for every LLM-planned strategy. Ask a model for "the
foundational papers on X" and it returns confident, well-formatted, plausible titles — some
of which are real, some of which are near-misses of real papers, and some of which do not
exist at all. The failure is silent: a fabricated citation looks exactly like a correct one
in the output.

So nothing an LLM names reaches the user unverified. Every proposed title is searched for
against the real providers, and the best hit is accepted **only if the titles genuinely
match**. Everything else is reported as unresolved rather than quietly dropped or, worse,
quietly substituted with whatever the search happened to return first.

The substitution risk is the subtle one. A search for a hallucinated title still returns
results — the provider is a relevance engine, not an oracle — so accepting the top hit
unconditionally would convert every hallucination into a real-looking but wrong paper. The
similarity floor exists to make that impossible.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from ..config import Config
from ..domain.models import Paper
from ..providers.base import BasePaperProvider, Capability, ProviderError
from ..providers.http_cache import HTTPCache
from ..providers.registry import build_paper_providers

logger = logging.getLogger(__name__)

# Below this, we say "not found" rather than accept the provider's best guess. Tuned on
# real misses: "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" against
# "Improving neural networks by preventing co-adaptation of feature detectors" scores well
# under 0.5, and those are two genuinely different papers that a looser floor would fuse.
MIN_SIMILARITY = 0.75

# At least this fraction of the query's content words must appear in the candidate. F1
# already penalises missing words, but this states the rule the error message needs to
# explain: we asked for a specific paper and did not get most of its title back.
MIN_COVERAGE = 0.80

# Function words only. An earlier version also dropped "simple", "effective", "approach",
# "novel", "method" and "learning" as low-signal — which is true of a topic query and false
# of a title, where those words are often the whole difference between two papers.
# "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014) and "Dropout:
# An Effective Approach to Prevent Neural Networks from Overfitting" (2025) become
# indistinguishable once you delete "simple", "effective" and "approach".
_STOPWORDS = frozenset(
    """a an the of for and or to in on with without via using by from as at is are be
    that this these those its their our it""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Morphological variants that name the same concept. Titles are cited inconsistently:
# Goodfellow's paper is indexed as "Generative Adversarial Nets" and universally called
# "Generative Adversarial Networks", and a planner naming the latter must still find the
# former. Without this, tightening the match floor to stop wrong-paper substitution
# rejects the correct paper instead — a fix that trades one failure for another.
_EQUIVALENT = {
    "nets": "network", "net": "network", "networks": "network",
    "representations": "representation", "models": "model", "embeddings": "embedding",
    "transformers": "transformer", "features": "feature", "machines": "machine",
    "algorithms": "algorithm", "predictors": "predictor", "detectors": "detector",
}


def _canonical(token: str) -> str:
    return _EQUIVALENT.get(token, token)

# Above this, the resolved paper is the planned paper and the planner's rationale about it
# can be trusted. Between MIN_SIMILARITY and this, it resolved but the caller should not
# present the plan's prose as describing it.
CONFIDENT_MATCH = 0.90

# Records that are not papers. OpenAlex indexes reading-group recordings, lecture slides and
# seminar notes alongside papers, and they surface as plausible search hits: a path for GANs
# opened on a Japanese-language reading-group record, and one for word embeddings included a
# item whose title literally ends "(Gdl Seminar)". A learner cannot read these.
_NON_PAPER_RE = re.compile(
    r"\b(seminar|slides?|lecture notes?|reading group|talk|webinar|tutorial session|"
    r"presentation|poster|erratum|corrigendum|retracted|table of contents|front matter|"
    r"editorial board)\b",
    re.IGNORECASE,
)


def looks_like_a_paper(paper) -> bool:
    """Reject records that are indexed alongside papers but cannot be read as one."""
    if _NON_PAPER_RE.search(paper.title):
        return False
    # A record with no authors at all is usually a table-of-contents entry or a stub.
    return bool(paper.authors)


@dataclass(frozen=True, slots=True)
class Resolution:
    """One attempt to ground a named paper in reality."""

    query_title: str
    paper: Paper | None
    similarity: float
    note: str

    @property
    def ok(self) -> bool:
        return self.paper is not None

    @property
    def is_confident(self) -> bool:
        """Resolved, and close enough that the plan's prose still describes this paper.

        A resolution can clear the acceptance floor while still being a slightly different
        record — a companion paper, an extended abstract, a different edition. The planner
        wrote its rationale about the paper it NAMED, so below this bar that rationale must
        not be presented as describing what came back.
        """
        return self.ok and self.similarity >= CONFIDENT_MATCH


def tokenize(title: str) -> set[str]:
    tokens = {
        _canonical(t) for t in _TOKEN_RE.findall(title.lower()) if t not in _STOPWORDS
    }
    # Keep stopwords if that is all there was, rather than returning an empty set that
    # would make every short title match every other.
    return tokens or set(_TOKEN_RE.findall(title.lower()))


def main_title(title: str) -> str:
    """The part before the first colon — a paper's name without its subtitle."""
    return title.split(":", 1)[0].strip() or title.strip()


def token_f1(query: str, candidate: str) -> float:
    """Symmetric F1 over content-word sets."""
    tq, tc = tokenize(query), tokenize(candidate)
    if not tq or not tc:
        return 0.0
    overlap = len(tq & tc)
    if not overlap:
        return 0.0
    precision = overlap / len(tq)
    recall = overlap / len(tc)
    return 2 * precision * recall / (precision + recall)


def title_similarity(query: str, candidate: str) -> float:
    """How well `candidate` matches the paper `query` names, in [0, 1].

    **Symmetric F1, with one narrow exception for subtitles.** The measure has to separate
    two cases that look structurally identical — in both, the query's words all appear
    inside a longer candidate title:

    * **Accept.** "Dropout" vs "Dropout: A Simple Way to Prevent Neural Networks from
      Overfitting". Same paper, named by its main clause.
    * **Reject.** "Attention Is All You Need" vs "Is Space-Time Attention All You Need for
      Video Understanding?". Different papers. TimeSformer contains every content word of
      Vaswani's title, and shipped as the anchor of a "transformers" path — carrying the
      planner's sentence "this is the paper that defines the Transformer".

    Containment cannot tell these apart; both are 100% contained. What separates them is
    **where the extra words are**. In the first, the candidate's main title *is* the query
    and the extra words are a subtitle after a colon. In the second, the extra words are
    wrapped around the query — prepended ("Is Space-Time") and appended ("for Video
    Understanding") — which is what makes it a different work.

    So the subtitle exception is deliberately strict: it fires only when the candidate's
    main title has *exactly* the query's content words, not merely most of them. Everywhere
    else, symmetric F1 applies and extra candidate words cost score, as they should.
    """
    tq = tokenize(query)
    if not tq:
        return 0.0

    # The candidate's main title IS the whole query: the query named the paper by its main
    # clause and the extra words are a subtitle.
    #
    # Restricted to a query with NO subtitle of its own. Matching on main title alone when
    # BOTH sides have subtitles is unsound: "Dropout: A Simple Way to Prevent Neural
    # Networks from Overfitting" and "Dropout: An Effective Approach to Prevent Neural
    # Networks from Overfitting" share the main title "Dropout" and are eleven years and
    # two research groups apart. When both carry a subtitle, the subtitle IS the
    # discriminator, so fall through to full-title F1 and let it do its job.
    if ":" not in query and tokenize(main_title(candidate)) == tq:
        return 1.0

    score = token_f1(query, candidate)
    if _has_qualifier_prefix(query, candidate):
        score = min(score, QUALIFIER_PENALTY_CAP)
    return score


# A candidate with a qualifier bolted onto the front of the query is capped below the
# acceptance floor, so it can still be reported with a score but can never be substituted.
QUALIFIER_PENALTY_CAP = 0.70

# More leading extras than this and the pattern is not a qualifier — it is the query naming
# a later clause of a longer real title, which usually IS the same paper.
_MAX_QUALIFIER_WORDS = 2


def _has_qualifier_prefix(query: str, candidate: str) -> bool:
    """Does the candidate prepend a qualifier to the query's title?

    A one- or two-word prefix in front of an otherwise-matching title almost always names a
    *variant*, and in this literature the variant is a different paper: "Generative
    Adversarial Nets" (Goodfellow) versus "**Conditional** Generative Adversarial Nets"
    (Mirza), which differ by one word and by authorship, year, and content. F1 scores that
    pair at 0.86 — comfortably over any threshold that still accepts legitimate matches —
    so word count alone cannot separate them and position has to.

    This is a heuristic and it has a known cost: a planner that names a paper by a later
    clause of its real title ("Preventing co-adaptation of feature detectors" for
    "**Improving neural networks by** preventing co-adaptation of feature detectors") looks
    the same from the front. The three-word prefix there is what keeps it out of this rule,
    and the residual risk is accepted deliberately — a learner told the critique that a
    substituted paper is worse than a missing one, because a missing paper sends you
    elsewhere and a substituted one does not.
    """
    tq = tokenize(query)
    # Canonicalised the same way as `tokenize`, in document order. Comparing raw surface
    # forms against a canonicalised set silently drops any word that was normalised
    # ("networks" -> "network"), which shifts what counts as the prefix and made this rule
    # fire on titles it was never meant to touch.
    candidate_tokens = [
        _canonical(t)
        for t in _TOKEN_RE.findall(candidate.lower())
        if t not in _STOPWORDS
    ]
    leading: list[str] = []
    for token in candidate_tokens:
        if token in tq:
            break
        leading.append(token)
    else:
        return False  # no query token found at all; F1 will already be near zero
    return 0 < len(leading) <= _MAX_QUALIFIER_WORDS


def _coverage(query: str, candidate: str) -> float:
    """Fraction of the query's content words present in the candidate."""
    tq, tc = tokenize(query), tokenize(candidate)
    return len(tq & tc) / len(tq) if tq else 0.0


class PaperResolver:
    """Resolves named papers against the enabled search providers."""

    def __init__(
        self,
        config: Config,
        cache: HTTPCache | None = None,
        providers: list[BasePaperProvider] | None = None,
    ) -> None:
        self.config = config
        self.cache = cache or HTTPCache(
            config.provider_cache.cache_dir, enabled=config.provider_cache.enabled
        )
        self.providers = (
            providers
            if providers is not None
            else build_paper_providers(config, Capability.SEARCH, cache=self.cache)
        )

    async def resolve(
        self, title: str, year: int | None = None, limit: int = 8
    ) -> Resolution:
        title = (title or "").strip()
        if not title:
            return Resolution(title, None, 0.0, "empty title")
        if not self.providers:
            return Resolution(title, None, 0.0, "no search provider enabled")

        results = await asyncio.gather(
            *(self._search(provider, title, limit) for provider in self.providers)
        )
        candidates = [p for group in results for p in group if looks_like_a_paper(p)]
        if not candidates:
            return Resolution(title, None, 0.0, "no provider returned a readable paper")

        scored = [(self._score(title, year, paper), paper) for paper in candidates]
        # Ties broken by citation count then id, so resolution is deterministic.
        scored.sort(key=lambda pair: (-pair[0], -(pair[1].citation_count or 0), pair[1].canonical_id))
        best_score, best = scored[0]

        coverage = _coverage(title, best.title)
        if best_score < MIN_SIMILARITY or coverage < MIN_COVERAGE:
            missing = sorted(tokenize(title) - tokenize(best.title))
            return Resolution(
                title,
                None,
                best_score,
                f"best match {best.title!r} scored {best_score:.2f} with "
                f"{coverage:.0%} of the query's words present"
                + (f" (missing: {', '.join(missing[:5])})" if missing else "")
                + " — treating as not found rather than substituting it",
            )
        return Resolution(title, best, best_score, f"matched {best.title!r} at {best_score:.2f}")

    # -- convenience -------------------------------------------------------------------

    async def resolve_many(
        self, requests: list[tuple[str, int | None]]
    ) -> list[Resolution]:
        """Resolve several named papers. Order is preserved."""
        return list(await asyncio.gather(*(self.resolve(t, y) for t, y in requests)))

    def _score(self, title: str, year: int | None, paper: Paper) -> float:
        """Title match, nudged by year agreement — but never penalised by disagreement.

        The penalty was removed after it was measured selecting the wrong paper. Asked for
        "Attention Is All You Need" (2017), the correct record scored 0.85 and the wrong
        one — TimeSformer — scored 0.92, because OpenAlex carries a duplicate of Vaswani
        dated **2025**, which tripped the "different era" penalty, while TimeSformer's
        2021 escaped it. Provider years are unreliable enough (AlexNet as 2017, BatchNorm
        as 2024) that disagreement is weak evidence about identity, whereas agreement is
        still mild evidence for it. Keep the bonus, drop the penalty.
        """
        score = title_similarity(title, paper.title)
        if year is not None and paper.year is not None and abs(paper.year - year) <= 1:
            score += 0.05
        return max(0.0, min(1.0, score))

    async def _search(
        self, provider: BasePaperProvider, title: str, limit: int
    ) -> list[Paper]:
        """One provider's search. Never raises — an unresolvable step degrades the path."""
        try:
            found = await provider.find_by_title(title, limit)
        except ProviderError as exc:
            logger.warning("resolver: %s failed on %r: %s", provider.name, title[:60], exc)
            return []
        except Exception as exc:  # noqa: BLE001 - an adapter bug must not break planning
            logger.exception("resolver: %s raised on %r", provider.name, title[:60])
            return []
        return found
