# Reading Notes — Challenges Behind the Reading Workspace

Working notes for `REQUIREMENTS.md` §7 (progressive reading passes), §8 (notes, conversational
assistant) and §11 D13. Records **what has to be solved before the reading surface can be built**,
so the hard parts are met as decisions rather than as surprises.

**Why this file exists separately from `REQUIREMENTS.md`:** most of what follows is not yet a
requirement. It is the set of problems that will *force* requirements. Several of them have to be
answered before the first line of schema is written, because they are not reversible once a user
has stored a note.

**Status:** analysis only, 2026-08-17. No implementation, no benchmarks run. Every number below is
an order-of-magnitude estimate and is marked as such.

---

## The shape of the phase

Today PaperThread ends at the recommendation: *here is the path, here is why each step is on it*.
The user then leaves for a PDF viewer and never comes back, and the system never learns whether
they read anything. The reading workspace closes that loop — a place to open the paper, read it in
passes, ask questions grounded in the text, and keep notes.

That framing matters for scope. **The reading surface exists to make the next paper choice better,
not to be a reader.** §10 explicitly rules out "a PDF reader/annotation product in v1", and this
phase walks right up to that line. The test for any feature here is §9's guardrail: does it improve
the model of the user's journey through the literature? Notes that feed personalization pass.
Highlight colours, export formats and citation managers fail.

The strongest argument for building it is not the reading experience at all — it is that
**reading events and notes are the only real signal the recommender has ever been missing.** §2's
reading history, §3's personalization, §5's gap detection and pipeline stage 6 are all blocked on
data that does not exist yet, and this is where it comes from.

---

## Part 1 — Getting the text at all

### 1.1 Full text availability is a licensing fact, not an engineering one

`PROVIDER_NOTES.md` §1.1 already records that no provider supplies citations + abstracts + free
full text together. For full text the gap is sharper than for abstracts:

| Provider | Full text | Notes |
|---|---|---|
| arXiv | **Yes** — PDF, and often LaTeX source | CS/ML preprints only, 1991→. The best case by a wide margin. |
| Crossref | No | DOIs and metadata. Never had the text. |
| OpenAlex | Indirect | `best_oa_location` points at a PDF hosted elsewhere; success rate varies by publisher. |
| Semantic Scholar | Partial | S2ORC has parsed full text for a subset; needs a key and has its own terms. |

Consequence: **coverage of the reading surface is a function of open-access status, and nothing we
build changes that.** For a CS/ML corpus (D7) arXiv coverage is high, which is the single largest
reason the corpus choice was correct. It will not hold if the corpus ever widens.

### 1.2 The hole-in-the-path problem

A path is *ordered*. If steps 1, 2, 4 and 5 open in the workspace and step 3 does not, the product's
central promise has a gap exactly where the user is trying to walk. Two bad ways to respond:

- **Prefer OA papers when building the path.** This corrupts the pedagogy. D2 says the best
  prerequisite is the best prerequisite; letting a licence decide the reading order is the same
  class of error as letting recency decide it. **Rejected.**
- **Hide unavailable steps.** This is the failure mode D12 exists to prevent — a missing thing
  must never look like an absent thing.

The rule that already exists one level down is the right one: *papers with no abstract are retained
and flagged, never filtered out* (D11). Same principle here — **availability is reported, never
selected on**, and an unavailable step still renders with its explanation, its position and a link
out to the publisher.

### 1.3 Fetching is a per-user act once deployed

Q15 is already open for abstracts. Full text makes it sharper: caching a PDF and serving it to a
second user is redistribution, and the licence that permits the first act often does not permit the
second. Publisher OA licences vary per paper; arXiv's are per-submission and include
non-redistributable options.

The conservative posture that is probably forced: **cache derived structure freely, never re-serve
the source PDF to a user who did not fetch it.** Derived structure (section offsets, extracted
plain text, embeddings, summaries) is a different artefact from the document. That posture costs a
re-extraction per user, which is only tolerable because extraction is cacheable per (paper, user)
and rare.

---

## Part 2 — Turning a PDF into something §7 can page through

### 2.1 §7 requires *logical structure*, not text

The progressive passes are abstract → conclusion → introduction → figures → deeper sections → full
paper. That is a **section-level** contract. A PDF does not contain sections; it contains positioned
glyphs in one or two columns with floats, running headers, footnotes and math. Everything §7
promises depends on recovering structure that the format threw away.

Candidate approaches, in rough order of output quality:

| Approach | Gives | Cost |
|---|---|---|
| **arXiv LaTeX source** | Real `\section`, real math, real figure captions | Free and exact — but arXiv only, and only when source is available |
| **GROBID** | TEI XML with sections, references, figures | Java service to run alongside; heaviest operational dependency in the repo |
| **marker / Nougat** | Markdown with structure and math | ML models; slow on CPU, effectively wants a GPU |
| **PyMuPDF** | Fast raw text | No logical structure at all — cannot satisfy §7 |

The likely answer is two-tier: **LaTeX source when arXiv has it, a PDF parser otherwise**, with the
extraction method recorded per paper the same way provider provenance is recorded per field (D11).
A paper read from LaTeX and a paper read from a PDF parse are not the same quality of object and
the system should know which it is holding.

### 2.2 Math and figures degrade silently, and that is the dangerous part

For an ML paper, meaning frequently lives in an equation or in Figure 2. PDF text extraction turns
equations into scattered glyph soup and drops figures entirely. Two failure modes:

1. The user sees a mangled section and distrusts the product. Recoverable.
2. **The mangled text is fed to the LLM, which fluently explains a corrupted formula.** Not
   recoverable — it is confidently wrong about the one thing the user came to understand.

This argues for extraction confidence being a first-class, *per-block* property, and for the
assistant refusing to explain blocks below a threshold rather than trying. It also argues for
rendering page images for figure-heavy passes instead of pretending text is enough.

### 2.3 Block identity must be stable before anything anchors to it

D8 already requires that ingestion be **re-runnable at greater depth** without losing paper
identity or edges. Reading adds a much harder version of the same rule: notes, highlights and every
grounded assistant citation anchor into extracted text. If a better extractor later renumbers or
re-segments the blocks, **every note and every citation dangles.**

Character offsets into a specific extraction are the tempting choice and the wrong one. Something
more durable is needed — content-hashed block IDs, or anchors stored as quoted text plus fuzzy
re-location on re-extraction, or both, with a documented repair path for anchors that cannot be
found. **This is a schema decision that must be made before the first note is stored**, because
there is no migration for an anchor whose target no longer exists.

---

## Part 3 — The assistant

### 3.1 Grounding is the whole problem

PaperThread already learned this lesson expensively: two paths scored 22/25 and 9/25, and the
9/25 one — which called a video-understanding paper "the paper that defines the Transformer" —
rendered with exactly the same confident labels and fluent prose as the good one. That is why
`Confidence` sits above the path in the UI and why judged and inferred edges look different.

The reading assistant raises the stakes in both directions. The user is now looking at the source
while the model talks about it, so every hallucination is immediately checkable — which is
excellent for calibration and fatal for trust when it goes wrong.

The rule that follows directly from the existing design: **every assistant claim about the paper
carries an anchor into the paper.** Not "as described in Section 3" — an actual block reference the
UI can scroll to and highlight. A claim that cannot be anchored is rendered as model inference, not
as paper content, and it looks visibly different. This is the same claim-provenance distinction the
frontend already draws with the solid/dashed thread, and it should look the same, because it *is*
the same: *did something verify this, or is it inferred?*

### 3.2 Personalized explanations need reading history that does not exist yet

§7's "personalized explanations grounded in the user's prior reading" is the most valuable feature
in the phase and the most blocked. It needs §2's reading history to be real, which needs this phase
to have shipped — a genuine ordering constraint, not a chicken-and-egg paradox: **v1 of the
workspace produces the data; v2 consumes it.** Building the personalization prompt before there is
history to put in it produces a prompt that flatters the model and tells the user nothing.

### 3.3 Context economics

A full ML paper is roughly 8k–20k tokens, more with appendices. Naive per-turn stuffing is
expensive and degrades as the conversation grows.

§7's passes are the natural answer, and they were already the right product decision for
independent reasons: **the passes are the chunk boundaries.** Summarize per pass, cache the
summaries, retrieve over the paper's own blocks for specific questions, and only load full sections
when the question actually reaches into one.

This also needs new entries in `[llm.roles]`, and the existing per-role split is exactly right
here: pass summarization is high-volume, cacheable, batchable and belongs on a cheap model;
interactive question-answering is latency-sensitive and low-volume and can afford a strong one.
One global model prices both wrong — the same argument already recorded in `paperthread.toml`.

### 3.4 Prompt injection from the document

A paper is untrusted input. Once its text goes into a prompt whose output the user acts on, the
document can attempt to steer the assistant — and adversarial text in preprints is a real, if
uncommon, phenomenon. Extracted paper content must be handled as **data, never as instructions**,
with the boundary made explicit in the prompt and the assistant's tools scoped so that a document
cannot cause an action outside the current reading session.

---

## Part 4 — What it costs the rest of the system

### 4.1 The first user-owned mutable data

Everything PaperThread stores today is *derived and disposable*: delete `.cache/` and nothing of the
user's is lost. Notes invert that. Losing them is unrecoverable and unforgivable, and it changes
several standing assumptions at once:

- Alembic is chosen (D5) but no schema is owned yet — migrations become real rather than theoretical.
- Export and backup become obligations, not features.
- The local-first stance (D3) means the durable store is a SQLite file on a laptop with no backup.

### 4.2 "Read" stops being a boolean

§7's passes mean a user can have read the abstract, the abstract and conclusion, or the whole
paper — and the recommender must treat those differently. Skimming an abstract is weak evidence of
understanding; finishing a paper and writing three notes about it is strong. Reading events need a
depth, and §5's gap detection should read that depth rather than a flag. Related: Q14 (does a
history entry have to be a formal paper?) becomes load-bearing here.

### 4.3 Latency moves to the foreground

Path building is already slow — many provider calls plus LLM planning — and users tolerate it
because it happens once, behind a progress state. Reading assistance is *interactive*: a typed
question expects a first token in under two seconds.

Two things the repo does not have follow from that. **Streaming**: the API is currently one
request, one JSON response. And **background work**: extraction and pass summarization must happen
*before* the click, which means a job queue — awkward under D3's local-first stance, where there
may be no worker process at all. A plausible compromise is to extract the next 1–2 steps of the
path speculatively while the user reads the current one.

### 4.4 The frontend gains a second layout

The UI is a single scrolling thread on one page with no routing. A workspace is a persistent
multi-pane layout (path · paper · assistant/notes) with per-paper URLs and state that survives
navigation. Two design constraints carry forward from `frontend/AGENTS.md` and must not be broken:
**era-as-colour keeps meaning era only**, and the grounded/inferred distinction in the assistant
should reuse the solid/dashed vocabulary the thread already established, since it is the same claim.

### 4.5 Evaluation has no equivalent yet

`backend/tools/eval_paths.py` and its rubric score paths. Nothing scores a summary. Faithfulness is
measurable — *does the cited span actually support the claim?* — and, consistent with the
evidence-first posture of `RETRIEVAL_NOTES.md`, that harness should exist **before** the assistant
ships, not after it disappoints someone.

---

## Consolidated: what must be decided before implementation

| # | Decision | Blocks |
|---|---|---|
| 1 | Anchor scheme for notes and citations, durable across re-extraction | The schema. Everything else. |
| 2 | Extraction stack — LaTeX-first with which PDF fallback | §7 passes, all assistant grounding |
| 3 | Caching and redistribution posture for fetched full text | Deployment (D3), Q15 |
| 4 | Reading-event model: depth, not a boolean | §2, §3, §5, stage 6 |
| 5 | Where background extraction runs under local-first | §7 latency |
| 6 | Faithfulness eval harness | Whether the assistant may ship |

## Deliberately deferred

- Collaborative or shared notes — social features are a §10 non-goal.
- Annotation export / citation-manager integration — fails the §9 guardrail test.
- Non-CS corpora, where the OA coverage assumption in §1.1 collapses.
