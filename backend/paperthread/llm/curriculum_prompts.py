"""Prompts for LLM-planned learning paths, kept as named variants.

Separate from `prompts.py` — and with its own `VERSION` — so that iterating on curriculum
wording does not invalidate the structural pipeline's cached judgments, and so the tuning
loop can select a variant by name without editing code.

**The design problem these exist to solve.** Ranking papers by citation-graph centrality
produces the *important* papers for a topic, which is not the same thing as a path through
it. Asked about dropout, centrality returns ImageNet and ResNet — genuinely central to the
literature dropout lives in, and useless to someone trying to understand dropout. A
learning path is a teaching order: what you must understand first, then the thing itself,
then where it went. That is a pedagogical judgment, and it is the one thing in this system
that structure cannot supply.

Rules carried from `prompts.py` and still binding here:

* **No prompt presumes the field.** D7 restricts the corpus, not the code.
* **Reply with JSON only**, because Ollama ignores schema enforcement (see the config
  notes) — the instruction in the prompt is the only thing actually producing JSON.
* **Named papers are never trusted.** Everything a model names here is resolved against a
  real provider by `retrieval/resolver.py`, and unresolvable titles are reported, not
  substituted.
"""

from __future__ import annotations

VERSION = "c1"

_JSON_ONLY = (
    "Reply with ONLY a JSON object. No prose before or after it, no markdown fences, "
    "no commentary."
)


# ======================================================================================
# Syllabus: plan the whole teaching sequence up front
# ======================================================================================

SYLLABUS_SCHEMA = {
    "type": "object",
    "required": ["anchor_title", "steps"],
    "properties": {
        "anchor_title": {"type": "string", "minLength": 3},
        "steps": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "object",
                "required": ["position", "concept", "title", "stage", "why_here"],
                "properties": {
                    "position": {"type": "integer", "minimum": 1},
                    "concept": {"type": "string", "minLength": 3},
                    "title": {"type": "string", "minLength": 3},
                    "authors": {"type": "string"},
                    "year": {"type": "integer"},
                    "stage": {
                        "type": "string",
                        "enum": ["prerequisite", "anchor", "followup"],
                    },
                    "why_here": {"type": "string", "minLength": 10},
                },
            },
        },
    },
}

SYLLABUS_SYSTEM_V1 = f"""\
You design reading sequences that teach a research topic from the ground up.

Given a topic, produce the ordered list of papers someone should read to understand it,
starting from what they must know first and ending with where the topic went next.

Structure every sequence in three stages:

1. "prerequisite" — the ideas the topic is built ON. These usually come from an EARLIER and
   often BROADER area than the topic itself, and their titles usually do not contain the
   topic's name. This is the stage that makes it a learning path rather than a reading
   list, and it is the stage most people get wrong by skipping.
2. "anchor" — the paper that IS the topic. The one a knowledgeable person would name if
   asked "what should I read to learn this?". Usually exactly one, occasionally two.
3. "followup" — what came after: major extensions, competing approaches, critiques, or the
   analysis that explained why it works.

Hard requirements:

- Order by CONCEPTUAL DEPENDENCY, not by date and not by fame. Step N must be
  understandable to someone who has read steps 1..N-1 and nothing else.
- The anchor must NOT be first. If it is, you have skipped the prerequisites.
- Name REAL papers, with their exact published titles. Every title you give is checked
  against a bibliographic database; an invented or misremembered title is dropped and the
  step is lost. If you are not confident a paper exists under that exact title, choose a
  different paper you are sure of.
- Prefer the paper that INTRODUCED an idea over a later survey of it, unless the survey is
  genuinely the better entry point.
- Do not pad. A tight sequence of 6 strong papers beats 12 loose ones.
- "concept" is what the reader learns at that step, in plain words — not the paper's title
  restated.
- "why_here" says why this step comes at this position, referring to what precedes it.

{_JSON_ONLY}"""


def syllabus_user(topic: str, budget: int) -> str:
    return (
        f'Topic the learner typed: "{topic}"\n\n'
        f"Design the reading sequence. Use at most {budget} steps.\n"
        f"Aim for roughly: 40% prerequisite, one or two anchor papers, the rest followup.\n\n"
        'Return JSON shaped as: {"anchor_title": "<exact title of the anchor paper>", '
        '"steps": [{"position": 1, "concept": "...", "title": "<exact paper title>", '
        '"authors": "<first author surname>", "year": 1998, '
        '"stage": "prerequisite|anchor|followup", "why_here": "..."}]}'
    )


# ======================================================================================
# Anchor: identify the paper that IS the topic
# ======================================================================================

ANCHOR_SCHEMA = {
    "type": "object",
    "required": ["anchors"],
    "properties": {
        "anchors": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["title", "why"],
                "properties": {
                    "title": {"type": "string", "minLength": 3},
                    "authors": {"type": "string"},
                    "year": {"type": "integer"},
                    "why": {"type": "string", "minLength": 10},
                },
            },
        },
        "reading_goal": {"type": "string"},
    },
}

ANCHOR_SYSTEM_V1 = f"""\
You identify the definitive paper for a research topic.

Given a topic, name the paper — or at most two — that a knowledgeable person would point to
and say "that is the paper on this". The work that introduced the idea, or the one that
established the form everyone now uses.

Rules:
- Give the EXACT published title. It is checked against a bibliographic database, and a
  misremembered title resolves to nothing.
- Prefer the paper that INTRODUCED the idea over later surveys or textbook treatments.
- If the topic names a broad area rather than a specific contribution, choose the single
  work most responsible for the area existing in its current form.
- Do not name more than two.

{_JSON_ONLY}"""


def anchor_user(topic: str) -> str:
    return (
        f'Topic the learner typed: "{topic}"\n\n'
        'Return JSON: {"anchors": [{"title": "<exact title>", "authors": "<first author '
        'surname>", "year": 2014, "why": "why this is the definitive paper"}], '
        '"reading_goal": "what the learner is trying to understand, in one sentence"}'
    )


# ======================================================================================
# Prerequisite selection from a paper's REAL reference list
# ======================================================================================

PREREQ_SCHEMA = {
    "type": "object",
    "required": ["selected"],
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "position", "concept", "why_needed"],
                "properties": {
                    "id": {"type": "string"},
                    "position": {"type": "integer", "minimum": 1},
                    "concept": {"type": "string", "minLength": 3},
                    "why_needed": {"type": "string", "minLength": 10},
                },
            },
        }
    },
}

PREREQ_SYSTEM_V1 = f"""\
You choose which of a paper's own references a learner must read BEFORE it, and in what
order.

You are given a target paper and a numbered list of works it actually cites. Every
candidate is real — these are its genuine references — so your job is selection and
ordering, not recall.

Select a reference when reading it first genuinely changes whether the learner can follow
the target: it introduces a method the target builds on, a problem the target solves, or a
concept the target assumes without explaining.

Reject a reference when it is cited for context, comparison, a dataset, an experimental
detail, or as one item in a list of related work. Most references are of this kind — expect
to select only a handful.

Then order what you selected so that each one is understandable given the ones before it,
starting from the most general or earliest idea.

Use ONLY the ids given to you. Do not invent references and do not rename them.

{_JSON_ONLY}"""


def prereq_user(topic: str, anchor_title: str, candidates: list[dict], want: int) -> str:
    lines = [
        f'The learner wants to understand: "{topic}"',
        f"Target paper: {anchor_title}",
        "",
        f"Works the target cites (choose at most {want}):",
    ]
    for candidate in candidates:
        year = candidate.get("year") or "n.d."
        lines.append(f"\n[{candidate['id']}] ({year}) {candidate['title']}")
        if abstract := (candidate.get("abstract") or "").strip():
            lines.append(f"    {abstract[:400]}")
    lines.append(
        '\nReturn JSON: {"selected": [{"id": "r3", "position": 1, "concept": "what the '
        'reader learns here", "why_needed": "why the target is hard to follow without it"}]}'
    )
    return "\n".join(lines)


# ======================================================================================
# Follow-up selection from papers that actually cite the anchor
# ======================================================================================

FOLLOWUP_SCHEMA = {
    "type": "object",
    "required": ["selected"],
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "position", "kind", "concept", "why_after"],
                "properties": {
                    "id": {"type": "string"},
                    "position": {"type": "integer", "minimum": 1},
                    "kind": {
                        "type": "string",
                        "enum": ["extension", "alternative", "critique", "analysis", "application"],
                    },
                    "concept": {"type": "string", "minLength": 3},
                    "why_after": {"type": "string", "minLength": 10},
                },
            },
        }
    },
}

FOLLOWUP_SYSTEM_V1 = f"""\
You choose what a learner should read AFTER the main paper on a topic, from among papers
that actually cite it.

Every candidate genuinely cites the target, so citing it is not a reason to select one.
Select a paper only if it changes what the reader understands about the target:

- "extension" — takes the target's idea further in the same direction.
- "alternative" — solves the same problem a different way, so the reader can compare.
- "critique" — shows where the target is wrong, overstated, or misunderstood.
- "analysis" — explains WHY the target works, which the target itself may not have.
- "application" — only when applying the idea reveals something about the idea.

Reject a paper when it merely USES the target as a component in unrelated work. A paper
that cites the target once in its related-work section and then does something else
entirely teaches the reader nothing about the topic, no matter how highly cited it is.
Most candidates are of this kind. Expect to select few.

Do not select papers about a different problem domain than the target — a paper applying
the idea to a distant field is not a next step for someone learning the idea.

Use ONLY the ids given to you.

{_JSON_ONLY}"""


def followup_user(topic: str, anchor_title: str, candidates: list[dict], want: int) -> str:
    lines = [
        f'The learner has just read the main paper on: "{topic}"',
        f"Main paper: {anchor_title}",
        "",
        f"Papers that cite it (choose at most {want}, in reading order):",
    ]
    for candidate in candidates:
        year = candidate.get("year") or "n.d."
        lines.append(f"\n[{candidate['id']}] ({year}) {candidate['title']}")
        if abstract := (candidate.get("abstract") or "").strip():
            lines.append(f"    {abstract[:400]}")
    lines.append(
        '\nReturn JSON: {"selected": [{"id": "c2", "position": 1, "kind": '
        '"extension|alternative|critique|analysis|application", "concept": "...", '
        '"why_after": "what reading this adds once the main paper is understood"}]}'
    )
    return "\n".join(lines)


# ======================================================================================
# Pedagogical rerank: order a retrieved candidate set into a teaching sequence
# ======================================================================================

RERANK_SCHEMA = {
    "type": "object",
    "required": ["sequence"],
    "properties": {
        "sequence": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "position", "stage", "concept", "why_here"],
                "properties": {
                    "id": {"type": "string"},
                    "position": {"type": "integer", "minimum": 1},
                    "stage": {
                        "type": "string",
                        "enum": ["prerequisite", "anchor", "followup"],
                    },
                    "concept": {"type": "string", "minLength": 3},
                    "why_here": {"type": "string", "minLength": 10},
                },
            },
        },
        "rejected_reason": {"type": "string"},
    },
}

RERANK_SYSTEM_V1 = f"""\
You turn a pile of retrieved papers into a teaching sequence.

You are given a topic and a numbered list of real papers found for it. Choose the subset
that forms a coherent learning path and put it in reading order.

A learning path is not a relevance ranking. The papers must build on each other: someone
who reads them in your order should be able to understand each one when they reach it.

- Mark as "prerequisite" the papers that teach what the topic assumes. These are often
  older and often do not mention the topic by name.
- Mark as "anchor" the paper that IS the topic, if it is present in the list.
- Mark as "followup" what extends, competes with, critiques, or analyses the anchor.
- LEAVE OUT papers that are merely related, merely highly cited, or that apply the topic to
  an unrelated domain. A shorter coherent path beats a longer padded one. Say what you
  excluded and why in "rejected_reason".
- If the anchor paper is missing from the list, say so in "rejected_reason".

Use ONLY the ids given to you.

{_JSON_ONLY}"""


def rerank_user(topic: str, candidates: list[dict], budget: int) -> str:
    lines = [
        f'Topic the learner typed: "{topic}"',
        "",
        f"Retrieved papers (choose at most {budget}, in teaching order):",
    ]
    for candidate in candidates:
        year = candidate.get("year") or "n.d."
        facts = candidate.get("facts")
        lines.append(f"\n[{candidate['id']}] ({year}) {candidate['title']}")
        if facts:
            lines.append(f"    graph: {facts}")
        if abstract := (candidate.get("abstract") or "").strip():
            lines.append(f"    {abstract[:350]}")
    lines.append(
        '\nReturn JSON: {"sequence": [{"id": "n7", "position": 1, "stage": '
        '"prerequisite|anchor|followup", "concept": "...", "why_here": "..."}], '
        '"rejected_reason": "what you left out and why"}'
    )
    return "\n".join(lines)
