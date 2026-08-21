"""Versioned prompts and their output schemas.

Three rules govern everything in this file.

1. **No prompt may presume the field.** D7 restricts the *corpus* to CS/ML; it does not let
   the code assume it. Nothing here says "machine learning", names an ML concept, or gives
   an ML example, so widening the corpus stays a configuration change.
2. **The model chooses among supplied ids, never names papers.** Every prompt hands the
   model a numbered shortlist and requires it to answer with those labels. Ask an LLM for
   "the foundational papers on X" and it will produce plausible, well-formatted,
   nonexistent citations. The shortlist makes that structurally impossible, and the caller
   re-checks every returned label anyway.
3. **`VERSION` is bumped when a prompt's meaning changes.** It is stored with every cached
   artifact, so a bump is how old judgments get retired (D10).

The judgment prompt is deliberately biased toward saying *no*. A learning path's failure
mode is not a missing edge — it is a false prerequisite, which pushes a paper the reader
does not need in front of the one they came for, and which the reader cannot detect.
"""

from __future__ import annotations

VERSION = "1"

# --------------------------------------------------------------------------------------
# Subtopic naming and ordering
# --------------------------------------------------------------------------------------

SUBTOPIC_ROLE = "topic_decomposition"

SUBTOPIC_SYSTEM = """\
You organize research literature into a teaching sequence.

You will be given a topic and several groups of papers. The groups were derived from the \
citation graph — papers in a group cite each other more than they cite outside it — so \
they represent genuine lines of work, not keyword clusters.

Your job is to name each group and put the groups in the order a newcomer should study \
them: what must be understood first comes first.

Rules:
- Use ONLY the group ids you are given. Do not invent, merge, or drop groups.
- A name is a short noun phrase describing what the group is ABOUT, in the vocabulary of \
the papers themselves. Not "Group 1", not "Foundational Work".
- Order by conceptual dependency, not by date. A recent paper can be a prerequisite for an \
older one if it is the clearer entry point, and an old paper can belong late.
- If two groups have no dependency between them, order them by which is the more natural \
starting point for someone new to the topic.
- Reply with JSON only."""

SUBTOPIC_SCHEMA = {
    "type": "object",
    "required": ["groups"],
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "label", "summary", "position"],
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string", "minLength": 2},
                    "summary": {"type": "string"},
                    "position": {"type": "integer", "minimum": 1},
                },
            },
        }
    },
}


def subtopic_user(topic: str, groups: list[dict]) -> str:
    lines = [f"Topic: {topic}", "", "Groups:"]
    for group in groups:
        lines.append(f"\n[{group['id']}] {group['size']} papers")
        for paper in group["papers"]:
            year = paper.get("year") or "n.d."
            lines.append(f"  - ({year}) {paper['title']}")
    lines.append(
        "\nName each group and assign it a position, 1 = read first. "
        "Return every group id exactly once."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Per-paper role and explanation (§5)
# --------------------------------------------------------------------------------------

EXPLANATION_ROLE = "prerequisite_judgment"

EXPLANATION_SYSTEM = """\
You explain why a specific research paper belongs in a specific reader's study path.

For each paper you will be given its title, year, abstract where available, and structural \
facts measured from the citation graph of this topic — including how many of the topic's \
own papers cite it. Those numbers are evidence about the paper's place in the literature; \
use them, and do not contradict them.

For each paper produce:
- role: what the paper DOES for a learner, chosen from the allowed values.
- why_it_matters: what changed because this paper exists. One or two sentences.
- what_it_assumes: what a reader needs to already understand. If little, say so plainly.
- what_it_teaches: the specific idea or result the reader takes away.
- why_for_you: why this paper at this point in THIS reader's path, referring to what they \
have already read and where the paper sits in the sequence.

Rules:
- Ground every claim in the supplied title, abstract, and structural facts. If the abstract \
is missing, say what can be inferred from the title and citation position and no more.
- Never invent results, numbers, author names, or claims not present in the input.
- Be specific and plain. No marketing language, no "seminal", no "groundbreaking".
- Two sentences maximum per field.
- Answer for every paper id given, using those exact ids.
- Reply with JSON only."""

_ROLES = [
    "foundation",
    "breakthrough",
    "alternative",
    "extension",
    "critique",
    "survey",
    "application",
]

EXPLANATION_SCHEMA = {
    "type": "object",
    "required": ["papers"],
    "properties": {
        "papers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "id",
                    "role",
                    "why_it_matters",
                    "what_it_assumes",
                    "what_it_teaches",
                    "why_for_you",
                ],
                "properties": {
                    "id": {"type": "string"},
                    "role": {"type": "string", "enum": _ROLES},
                    "why_it_matters": {"type": "string", "minLength": 10},
                    "what_it_assumes": {"type": "string", "minLength": 5},
                    "what_it_teaches": {"type": "string", "minLength": 10},
                    "why_for_you": {"type": "string", "minLength": 10},
                },
            },
        }
    },
}


def explanation_user(topic: str, papers: list[dict], reader: str) -> str:
    lines = [f"Topic the reader asked about: {topic}", "", f"Reader: {reader}", "", "Papers:"]
    for paper in papers:
        lines.append(f"\n[{paper['id']}] ({paper.get('year') or 'n.d.'}) {paper['title']}")
        facts = paper.get("facts")
        if facts:
            lines.append(f"  Position in this topic's citation graph: {facts}")
        abstract = (paper.get("abstract") or "").strip()
        if abstract:
            lines.append(f"  Abstract: {abstract[:1200]}")
        else:
            lines.append("  Abstract: not available from any provider.")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Prerequisite judgment (D2 stage 2)
# --------------------------------------------------------------------------------------

JUDGMENT_ROLE = "prerequisite_judgment"

JUDGMENT_SYSTEM = """\
You decide whether one paper is genuinely a prerequisite for understanding another.

You will be given pairs. In every pair, B was published before A and A cites B, so the \
citation direction is already established. That is NOT the question.

The question is: must a reader understand B in order to understand A?

Answer NO when:
- A cites B only for context, motivation, comparison, or as one item in a list of related \
work.
- B is background a reader would already have, or could pick up from A itself.
- A is self-contained enough that reading B first would not change comprehension.

Answer YES only when a reader who skipped B would misunderstand A, or would be unable to \
follow its central argument, method, or result.

Most citations are not prerequisites. Across the literature only a small minority of \
citations are substantively important; the rest are perfunctory. Expect to answer NO more \
often than YES, and when genuinely unsure, answer NO — a false prerequisite puts a paper \
the reader does not need in front of the one they came for, and they have no way to tell \
that it was a mistake.

Give confidence as the probability that YES is correct. Reply with JSON only."""

JUDGMENT_SCHEMA = {
    "type": "object",
    "required": ["judgments"],
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["pair", "is_prerequisite", "confidence", "reason"],
                "properties": {
                    "pair": {"type": "string"},
                    "is_prerequisite": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "minLength": 5},
                },
            },
        }
    },
}


def judgment_user(topic: str, pairs: list[dict]) -> str:
    lines = [f"Topic: {topic}", "", "Pairs to judge:"]
    for pair in pairs:
        prerequisite, dependent = pair["prerequisite"], pair["dependent"]
        lines.append(f"\n[{pair['id']}]")
        lines.append(
            f"  B (earlier, cited): ({prerequisite.get('year') or 'n.d.'}) {prerequisite['title']}"
        )
        if abstract := (prerequisite.get("abstract") or "").strip():
            lines.append(f"     {abstract[:600]}")
        lines.append(
            f"  A (later, citing):  ({dependent.get('year') or 'n.d.'}) {dependent['title']}"
        )
        if abstract := (dependent.get("abstract") or "").strip():
            lines.append(f"     {abstract[:600]}")
        lines.append("  Question: must a reader understand B to understand A?")
    lines.append("\nReturn one judgment per pair id, using those exact ids.")
    return "\n".join(lines)
