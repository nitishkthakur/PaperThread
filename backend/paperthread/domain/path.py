"""Learning-path domain types.

A learning path is a **sequence** (A → B → C) with prerequisite edges, not a ranked list
(REQUIREMENTS.md §4). Anything that flattens it into an unordered list is a bug.

Two things are modelled here that are easy to mistake for presentation detail:

* **`Explanation` is part of the data model, not UI garnish** (§5). Every recommendation
  must say why the paper matters, what it assumes, what it teaches, and why it suits this
  user. It also records HOW it was produced, because a heuristic explanation and an LLM
  one are not interchangeable and the UI must be able to tell a user which it is reading.
* **Every edge carries provenance.** Provider citation graphs disagree and none is
  complete (PROVIDER_NOTES C7), and an LLM judgment is only interpretable alongside the
  `{provider, model, prompt_version}` that produced it (D10). An edge with no provenance
  cannot be invalidated later, and D2 edges are *persisted data*.

Nothing here does I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .models import Paper


class PaperRole(str, Enum):
    """What a paper does for the learner, which is not the same as how important it is.

    Deliberately field-agnostic (D7): none of these names presume machine learning, or
    even computer science.
    """

    FOUNDATION = "foundation"
    BREAKTHROUGH = "breakthrough"
    ALTERNATIVE = "alternative"
    EXTENSION = "extension"
    CRITIQUE = "critique"
    SURVEY = "survey"
    APPLICATION = "application"
    UNCLASSIFIED = "unclassified"


class EdgeSource(str, Enum):
    """How a prerequisite edge came to exist.

    `CITATION` edges are unfalsifiable and free — if A cites B, B existed first and A's
    authors read it. `LLM_JUDGMENT` edges are opinions and are treated as such: they carry
    a confidence and a model stamp, and they can be withdrawn when the model changes.
    """

    CITATION = "citation"
    CO_CITATION = "co_citation"
    LLM_JUDGMENT = "llm_judgment"


class ExplanationSource(str, Enum):
    """Whether an explanation was reasoned or derived from graph structure.

    A structural explanation is a real explanation — it is grounded in what the corpus
    actually did — but it is not the LLM's, and §5's "why for this user" is weaker in it.
    Never present one as the other.
    """

    LLM = "llm"
    STRUCTURAL = "structural"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Who asserted something, and under what model.

    `asserted_by` is a provider name for citation edges and a provider/model pair for LLM
    judgments. `prompt_version` is null for anything that did not come from a prompt.
    """

    asserted_by: str
    model: str | None = None
    prompt_version: str | None = None

    def stamp(self) -> str:
        parts = [self.asserted_by]
        if self.model:
            parts.append(self.model)
        if self.prompt_version:
            parts.append(f"p{self.prompt_version}")
        return "/".join(parts)


@dataclass(frozen=True, slots=True)
class PrerequisiteEdge:
    """`prerequisite` should be read before `dependent`.

    Direction is stated explicitly rather than implied by tuple order, because the citation
    edge points the other way (A cites B means B is the prerequisite) and that inversion is
    the single easiest thing to get backwards in this codebase.
    """

    prerequisite_id: str
    dependent_id: str
    source: EdgeSource
    provenance: Provenance
    # Only meaningful for LLM_JUDGMENT edges; structural edges are not probabilistic.
    confidence: float | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Explanation:
    """§5's four required questions, kept as four fields rather than one blob.

    Separate fields because the UI must be able to show "what it assumes" next to the
    prerequisite edges, and because a later evaluation pass needs to score them
    independently.
    """

    why_it_matters: str
    what_it_assumes: str
    what_it_teaches: str
    why_for_you: str
    source: ExplanationSource
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class Subtopic:
    """A cluster of the topic's literature, in reading order.

    Derived from citation-graph community structure, not from text similarity — text
    embeddings demonstrably fail to reproduce citation partitions even when trained on
    citations, and §4 cares about intellectual lineage (RETRIEVAL_NOTES, L1).
    """

    id: str
    label: str
    summary: str | None = None
    order: int = 0
    # True when an LLM named the cluster; otherwise the label is a structural placeholder.
    named_by_llm: bool = False


@dataclass(frozen=True, slots=True)
class PaperSignals:
    """The structural evidence behind a paper's placement.

    Surfaced rather than hidden so a recommendation stays explainable and reproducible
    (D12): a user — and a future evaluation harness — can see that a paper is present
    because 23 candidates cite it, not because a model liked it.
    """

    # How many of the topic's candidate papers cite this one. The ancestor signal.
    co_citations: int = 0
    # Raw PageRank within the candidate subgraph.
    pagerank: float = 0.0
    # PageRank expressed relative to papers of the same age. This is the ranking signal;
    # raw PageRank "completely fails to identify recent milestone papers".
    age_rescaled_pagerank: float = 0.0
    # Rank-fusion score from Stage 1, kept so lexical relevance stays visible.
    lexical_score: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    # True when the paper was reached by citation expansion rather than by search — i.e.
    # keyword search could not have found it. Exactly the §5 knowledge-gap case.
    discovered_by_expansion: bool = False


@dataclass(frozen=True, slots=True)
class PathStep:
    """One paper at one position in the path.

    `level` is the DAG depth: every paper at level N has all its prerequisites at levels
    < N. Steps sharing a level may be read in any order relative to each other, which is
    real information — a path is a partial order, and pretending it is a strict line
    invents constraints the evidence does not support.
    """

    order: int
    level: int
    paper: Paper
    role: PaperRole
    signals: PaperSignals
    explanation: Explanation
    subtopic_id: str | None = None
    prerequisite_ids: tuple[str, ...] = ()
    # Papers the user has already read are kept in the path and marked, not removed —
    # §6 requires completed work to be visibly marked, and removing it would silently
    # break the prerequisite chain for everything downstream.
    already_read: bool = False


@dataclass
class LearningPath:
    """The product's actual output.

    `stages_run` and `notes` are not diagnostics — they are the honesty contract. A path
    built without LLM judgment is a different object from one built with it, and the user
    is entitled to know which they are looking at (D12).
    """

    topic: str
    steps: list[PathStep] = field(default_factory=list)
    subtopics: list[Subtopic] = field(default_factory=list)
    edges: list[PrerequisiteEdge] = field(default_factory=list)
    layers_used: list[str] = field(default_factory=list)
    stages_run: list[str] = field(default_factory=list)
    degraded: bool = False
    notes: list[str] = field(default_factory=list)
    # How much the system believes its own answer, in [0, 1], with the reasons.
    #
    # This exists because of a specific finding: a path scored 22/25 by a learner and one
    # scored 9/25 rendered IDENTICALLY — same confident stage labels, same fluent
    # rationales — and the 9/25 one described a video-understanding paper as "the paper
    # that defines the Transformer". When the system is sometimes wrong, its output must
    # carry the difference; a learner has no other way to tell the two apart.
    confidence: float = 0.0
    confidence_reasons: list[str] = field(default_factory=list)

    @property
    def levels(self) -> int:
        return max((step.level for step in self.steps), default=-1) + 1

    def steps_at(self, level: int) -> list[PathStep]:
        return [step for step in self.steps if step.level == level]
