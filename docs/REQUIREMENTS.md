# PaperThread — Project Requirements

> **Status:** Draft v0.1 — initial requirements captured 2026-08-10.
> **Authority:** This document is the single source of truth for what PaperThread must do.
> Do not change it without confirming with the project owner. See [Technical Decisions](#11-technical-decisions),
> [Open Questions](#12-open-questions), and [Changelog](#13-changelog).

## 1. Product Goal

PaperThread is a personalized research-paper learning platform designed to help users decide
**what research paper to read next and why**. The system should go beyond conventional paper
search or similarity-based recommendation by constructing a learning path based on the user's
interests, reading history, and likely knowledge gaps.

## 2. User Profile and Reading History

Each user must have a persistent profile containing:

- papers they have read
- papers they are currently reading
- active learning paths
- topic interests
- previous searches

Users must be able to mark any paper as **Read**. This action must update their profile and
influence future recommendations.

Initially, reading a paper is treated as evidence of exposure to the concepts within it.
Future versions may support richer states such as *familiar*, *understood*, *mastered*, or
*revisit*.

## 3. Personalized Paper Recommendations

Two distinct recommendation modes:

1. **Papers You May Like** — papers likely to interest the user based on previous reading,
   topics, and related research.
2. **Papers You Should Read Next** — papers selected because they fill an important
   prerequisite, conceptual gap, historical gap, or logical next step in the user's learning
   journey.

Recommendations must prioritize **educational value** rather than only citation count,
popularity, recency, or textual similarity.

## 4. Topic-Based Learning Paths

Users must be able to enter a topic such as *Transformers*, *Regularization*, or
*Diffusion Models*. The system must identify meaningful subtopics and construct a sequential
research-paper curriculum.

A learning path should include, when relevant:

- foundational papers
- prerequisites
- major breakthroughs
- alternative approaches
- extensions
- critiques
- later developments

**Ordering matters.** The output must represent sequences such as **A → B → C**, not merely an
unordered list of related papers.

## 5. Knowledge-Gap Detection

The system must compare the recommended topic curriculum against the user's reading history.
If a user has read an advanced paper but skipped an important predecessor, PaperThread must be
able to recommend the older paper as a knowledge gap.

Every recommendation must include a short explanation covering:

- why the paper matters
- what it assumes (prerequisites)
- what it teaches
- why it is appropriate **for this user**

## 6. Dynamic Learning Paths

When a user marks a recommended paper as read, the path must update automatically:

- completed papers are visually marked
- redundant recommendations may disappear
- new downstream papers may become available

Users must also be able to request additional papers within any subtopic. **"Show more"** must
continue the learning sequence rather than return loosely related results.

## 7. Paper Reading Experience *(later phase)*

A later product phase introduces **progressive reading passes**. A paper may first show its
abstract and conclusion, then its introduction and figures, then deeper sections, and finally
the complete paper.

Premium LLM features may:

- summarize individual reading passes
- explain concepts
- compare papers
- provide personalized explanations grounded in the user's prior reading

> Constraints for building this phase are recorded in D13; the problems behind them are worked
> through in [`READING_NOTES.md`](READING_NOTES.md).

## 8. Future Scope *(non-blocking)*

A conversational paper assistant, user notes, and discussion of personal interpretations may be
added later. These features remain **secondary** and **must not determine the initial
architecture**.

> D13 folds the assistant and notes into the §7 phase rather than leaving them open-ended. They
> stay secondary in the sense meant here — they must not reshape the initial architecture — but
> they are no longer undated, because §3's personalization has no input without them.

## 9. Core Focus (Guardrail)

The core product must remain focused on **modeling a user's journey through research literature
and identifying the most valuable next paper to read**. Any proposed feature should be evaluated
against this statement first.

## 10. Non-Goals (Initial Version)

Derived from §7–§9; listed explicitly so scope stays bounded.

- Not a general paper search engine.
- Not a citation-graph browser for its own sake.
- Not a social/discussion platform.
- Not a PDF reader/annotation product in v1.

## 11. Technical Decisions

Decisions made by the project owner. These are binding; changing one requires the same
confirmation as changing a requirement.

Each decision records **Now** (what v1 does) and **Later** (where it is headed). The *Later* is
not speculation — it is stated intent, and **v1 must not close it off**. When a decision says a
capability arrives later, the v1 design must leave a seam for it rather than assume it away.

### D1 — Paper metadata: adapter layer, backends deferred

No single metadata provider is committed to yet. Build a **provider-agnostic paper source
interface** (search, fetch by ID, fetch references/citations, fetch abstract/full text) so
concrete backends — Semantic Scholar, arXiv, OpenAlex, Crossref — can be plugged in, combined,
and swapped without touching recommendation logic.

**Now:** one or two providers wired up behind the interface, enough to populate a CS/ML corpus.
**Later:** multiple providers combined and reconciled, with provider choice per field.

Implications: no provider-specific ID leaks into the domain model; PaperThread owns its own
canonical paper identity with external IDs stored as aliases. Ingestion results are cached
locally so provider rate limits never sit on a user request path.

### D2 — Path construction: hybrid citation graph + LLM

Prerequisite and ordering edges (A → B → C) are derived in two stages:

1. **Candidate generation (algorithmic).** Citation/reference edges, publication dates, and
   influence signals produce candidate prerequisite pairs and a rough topological ordering.
2. **Judgment and explanation (LLM).** An LLM evaluates each candidate — *is A genuinely a
   prerequisite for B?* — and produces the per-recommendation explanation required by §5
   (why it matters, what it assumes, what it teaches, why for this user).

Stage 2 output must be **cached and persisted**, not recomputed per request. Edges are data,
not transient model output. This pipeline is the core intellectual property of the product.

**Now:** LLM judgment works from titles, abstracts, and citation context (see D8).
**Later:** the same judgment step reads full paper text, which should raise the quality of both
the prerequisite calls and the explanations. The judgment interface must therefore accept a
richer paper representation without changing its callers.

### D3 — Deployment: web app, local-first now, multi-user later

PaperThread is a **web application from day one** — never a CLI, notebook, or desktop-only tool.

**Now:** runs locally, single user, SQLite.
**Later:** deployed and multi-user.

This constrains the initial build — the local-first phase must not create a rewrite later:

- Every user-owned entity (reading history, paths, interests, searches) is keyed by a
  `user_id` from the start, even while only one user exists.
- Authentication sits behind an interface with a trivial local/single-user implementation;
  adding real auth must not touch domain logic.
- No process-global or module-level mutable user state.
- The database layer must be portable from local (SQLite) to a server database (Postgres).
- Frontend talks to the backend over HTTP only — no local-only shortcuts that a hosted
  deployment couldn't support.

### D4 — Stack: Python backend, React frontend

- **Backend:** Python. Keeps embedding/ML and graph work in-process.
- **Frontend:** React, communicating with the backend over an HTTP API.

Concrete choices (D5) must satisfy the portability constraints in D3.

### D5 — Frameworks, database, migrations

- **API:** FastAPI.
- **ORM:** SQLAlchemy, so the storage layer is engine-portable.
- **Database:** SQLite now → PostgreSQL later. No SQLite-only SQL, and no raw SQL that would not
  run on both.
- **Migrations:** Alembic from the first schema, so the local database and a future deployed one
  share one migration history.
- **Frontend:** React + Vite + TypeScript.
- **Vector store:** deferred. When embeddings are needed, prefer something that works locally and
  maps to `pgvector` on Postgres. Keep retrieval behind an interface so this stays swappable.

### D6 — Cold start: topic entry now, library seeding later

**Now:** the entry point for a new user is **entering a topic** (§4). No onboarding flow, no
import step — type *Transformers*, get an ordered curriculum.

**Later:** the user seeds their profile by **selecting papers and articles they have already
read**. This is stated intent, not a maybe. It implies v1 must:

- allow reading history to be populated by any means, not only by marking papers Read inside a
  path — the profile is not a byproduct of in-app activity;
- support bulk/batch additions to reading history without an active learning path;
- keep gap detection (§5) working against a history that arrived all at once, with no in-app
  reading sequence and no timestamps implying order.

Beyond that, "articles" signals that a read item may eventually be something other than a formal
paper. Do not assume every history entry has a DOI, a citation graph position, or a venue.

### D7 — Corpus: CS/ML first, no domain assumptions in code

**Now:** the ingested corpus is restricted to computer science and machine learning, where the
citation graph is dense and LLM prerequisite judgment is most reliable.

**Later:** other fields. Widening the corpus must be a **configuration change, not a rewrite**.

Therefore no field-specific logic may be hardcoded: no CS/ML-only taxonomies baked into the
schema, no arXiv-category assumptions in the domain model, no prompt that presumes the topic is
machine learning. The restriction lives in ingestion configuration, not in the model.

### D8 — Ingestion depth: metadata + abstract now, full text later

**Now:** ingest **metadata and abstracts only** — title, authors, date, venue, abstract,
references/citations. This is sufficient for path construction (D2), gap detection (§5), and
recommendation explanations. No PDF parsing in v1.

**Later:** ingest **full paper contents**, which is also what §7's progressive reading passes
require (abstract → conclusion → introduction → figures → deeper sections → full paper).

The v1 model must leave room for this:

- the paper entity carries content at a declared depth, so depth can increase later without a
  schema rewrite or a backfill of every consumer;
- anything consuming paper content asks for what it needs rather than assuming abstract-only;
- ingestion is re-runnable at a greater depth for papers already stored — enriching an existing
  paper must not mean re-creating it and losing its identity or its edges.

### D9 — Every external boundary is multi-provider, selected by config

**This is a general rule, not a per-feature one.** Any section of the system that talks to an
external service must be written so additional providers can be added by dropping in an adapter
and editing config — never by changing calling code.

Requirements:

- One **config file** is the single place a user selects providers and models. No provider
  choice is hardcoded, and none is read from scattered environment lookups.
- Provider ports are grouped by capability in their own directory, each with its own
  `AGENTS.md`.
- **Multiple providers may be active at once** — this is the normal case, not an edge case.
  Results from several providers reconcile onto canonical entities.
- Every provider **declares its capabilities**; callers query the registry and degrade
  gracefully instead of assuming feature parity.
- No provider-shaped type reaches domain logic.

Providers are **not feature-equivalent**, and the asymmetries are load-bearing. They are
documented in [`PROVIDER_NOTES.md`](PROVIDER_NOTES.md), which must be read before adding or
modifying any adapter and updated whenever a gap is found.

### D10 — LLM providers: Ollama Cloud by default

**Now:** **Ollama Cloud** is the default LLM provider (`https://ollama.com`, `OLLAMA_API_KEY`,
OpenAI-compatible endpoints; local Ollama is the same adapter with a different base URL).
**Later:** OpenRouter, Anthropic, and others, selected by config.

Consequences carried from `PROVIDER_NOTES.md` Part 2:

- **`LLMProvider` and `EmbeddingProvider` are separate ports**, configured independently.
  Verified 2026-08-10: **Ollama Cloud offers no embedding models at all** (Ollama embeddings are
  local-only), and **Anthropic has no embeddings API**. Our default LLM provider therefore cannot
  serve embeddings, and neither can the most likely later one — coupling these ports would break
  immediately, not eventually.
- **Embeddings are computed in-process** (sentence-transformers / ONNX), not via a provider API.
  Ollama applies no model-required prefixes for embeddings, degrades on batches ≥16, and changes
  embedding values across versions — any of which silently corrupts an index. Every stored vector
  records its embedding model and version; a change means a rebuild.
- Provider **and model are selectable per task role** (topic decomposition vs. prerequisite
  judgment have different cost/quality needs), not one global model.
- Structured output sits behind one method with a per-provider strategy, schema validation, and
  bounded retry. Raw model text is never trusted.
- Every cached LLM artifact records `{provider, model, prompt_version}`, so changing the default
  model cannot silently corrupt stored judgments (D2 edges are persisted data).

### D11 — Paper providers: capability ports, many active at once

**Now:** a small number of sources enabled by config. **Later:** more, including non-paper
sources.

Because no provider supplies citations, abstracts, and free full text together
(`PROVIDER_NOTES.md` Part 1), paper access is split by **capability, not vendor**:
`PaperSearch`, `CitationGraph`, `FullText`, `IdResolver`. A provider implements whichever it
supports, and a single logical paper is assembled from several.

This makes the following mandatory rather than optional:

- **Canonical paper identity with external IDs as aliases** (D1), because the same work appears
  as a preprint and a published paper with separate IDs and *split citation counts* — which
  would otherwise corrupt the centrality scoring that ranking depends on.
- **Field-level provenance**: which provider supplied each field, when, under what license.
- **Edge provenance**: citation graphs from different providers disagree and neither is
  complete, so edges record who asserted them.
- **Papers with no abstract are retained and flagged, never filtered out** — abstract coverage
  is worst for older papers, which are exactly the foundational ones §5 exists to surface.
- **Per-provider rate limiting and caching**; no provider call sits on a user request path
  un-cached.

### D12 — Retrieval and analysis are layered, and each layer stands alone

Recommendation is built as **independent layers of increasing capability and cost**, not as one
pipeline with an LLM bolted on. Each layer must produce a usable result **by itself**, and each
higher layer refines the one below rather than replacing it.

| Layer | Depends on | Must work when… |
|---|---|---|
| **L0 — Lexical / statistical** | nothing external | fully offline, no network, no model weights |
| **L1 — Local NLP** | downloadable model files | offline after a one-time download |
| **L2 — Embeddings** | embedding model (local or hosted) | offline if the model is local |
| **L3 — Reranking** | reranker model | offline if the model is local |
| **L4 — LLM reasoning** | an LLM provider (D10) | network/provider available |

Binding rules:

- **L0 must be genuinely useful on its own**, not a degraded placeholder. A user with no network
  access and no model weights must still be able to enter a topic and get a sensible ordered
  result.
- **Graceful degradation is a requirement, not a fallback path.** Disabling any layer by config
  must leave a working system, with quality reduced and clearly reported — never an error.
- **Layers are separately configurable and separately cacheable**, and each records which layer
  produced a given signal, so results remain explainable and reproducible.
- **The layer composition is not fixed by intuition.** Which techniques occupy each layer is
  decided by evidence (benchmark numbers, measured latency), not by convention or familiarity.
  Findings are recorded in [`RETRIEVAL_NOTES.md`](RETRIEVAL_NOTES.md).

This also protects the product against provider risk: an LLM outage degrades PaperThread's
quality but never takes it down.

### D13 — Reading workspace: the path is consumed in-product, and reading feeds the path

§7 and §8 describe the next phase — open a paper, read it in progressive passes, ask an LLM about
it, keep notes. This decision records the constraints that phase must be built under. The problems
behind them are worked through in [`READING_NOTES.md`](READING_NOTES.md).

**Why it is a phase and not a feature:** every personalization requirement in the product (§2
reading history, §3 recommendations, §5 gap detection, retrieval stage 6) is blocked on data that
does not exist, because today the user leaves for a PDF viewer and the system never learns whether
they read anything. **The reading workspace is where that data comes from.** It is justified by
§9's guardrail on that basis, not as a reading product in itself — §10's "not a PDF
reader/annotation product" still holds.

**Now (unchanged):** metadata and abstracts only (D8). Nothing in this decision is implemented.

**Later — binding constraints when it is built:**

- **Availability is reported, never selected on.** Full text is available only for open-access
  papers, so some steps of a path will not open in the workspace. Path construction must not prefer
  papers by licence — that would let a publisher decide the teaching order. An unavailable step
  still renders with its explanation and position, and says it is unavailable. This is D11's
  "papers with no abstract are retained and flagged, never filtered out", one level up.
- **Every assistant claim about a paper is anchored into that paper.** A claim that can be traced
  to a span of the text renders as grounded and links to it; a claim that cannot renders as model
  inference and looks different. This is the same distinction the path already draws between judged
  and inferred edges, and it exists for the same reason: when a system is sometimes wrong, the
  output must carry the difference.
- **Extraction provenance is recorded per paper, and quality per block.** A paper parsed from LaTeX
  source and one parsed from a PDF are not equivalent objects. Where extraction confidence is low —
  equations and figures are the usual casualties — the assistant declines to explain rather than
  explaining a corrupted formula fluently.
- **Note and citation anchors survive re-extraction.** D8 already requires ingestion to be
  re-runnable at greater depth without losing paper identity; anchors inherit that requirement.
  Raw character offsets into one extraction do not satisfy it. **This is settled before the first
  note is stored**, because a dangling anchor has no migration.
- **Notes are the first user-owned durable data in the system.** Everything stored today is derived
  and disposable. Notes are neither. Migrations, export and backup become obligations at that point.
- **Reading is recorded with a depth, not as a boolean.** §7's passes mean "read the abstract" and
  "read the paper" are different events, and §3 and §5 must be able to tell them apart.
- **Paper text is untrusted input.** Extracted content is data, never instructions, and a document
  must not be able to steer the assistant or reach anything outside the current reading session.
- **Faithfulness is evaluated before the assistant ships**, on the same evidence-first basis as
  D12's layer choices. Paths have `eval_paths.py` and a rubric; summaries need an equivalent.

## 12. Open Questions

Unresolved decisions that shape the architecture. When one is answered, fold the answer into the
relevant section above *and* mark the row resolved here.

| # | Question | Status |
|---|----------|--------|
| Q1 | Paper metadata source(s) — which providers, which is primary? | **Deferred** — adapter layer first (D1) |
| Q2 | Corpus scope — CS/ML only at first, or all of science? | **Resolved** — CS/ML corpus, field-agnostic code (D7) |
| Q3 | How are prerequisite/ordering edges derived? | **Resolved** — hybrid (D2) |
| Q4 | Accounts & auth — real multi-user auth in v1, or single local user? | **Resolved** — single local user, multi-user-ready design (D3) |
| Q5 | Deployment target? | **Resolved** — web app, local-first, deployable later (D3) |
| Q6 | Concrete backend framework, database, migrations | **Resolved** — FastAPI + SQLAlchemy + SQLite→Postgres + Alembic (D5) |
| Q7 | LLM provider, and which calls run offline (batch) vs. per-request | Open |
| Q8 | Is full paper text ingested in v1, or metadata + abstract only? | **Resolved** — metadata + abstract now, full text later (D8) |
| Q9 | Cold start — how do we build a profile for a brand-new user with no reading history? | **Resolved** — topic entry now, library seeding later (D6) |
| Q10 | "Premium" tier in §7 — a real billing concept, or just a feature-flag label? | Open |
| Q11 | Scale expectation once deployed — small group or public product? | Open |
| Q12 | Offline/caching — must the app work without network access to paper APIs? | Open |
| Q13 | Vector store choice, once embeddings are actually needed | Open (deferred by D5) |
| Q14 | Does a reading-history entry have to be a formal paper, or can it be any article? (raised by D6) | Open |
| Q15 | May cached abstracts be served to other users once deployed? Licensing differs by provider (PROVIDER_NOTES C9) | Open |
| Q16 | Concrete technique per D12 layer (L0–L4) — pending research, recorded in RETRIEVAL_NOTES.md | Open |
| Q17 | Anchor scheme for notes and assistant citations that survives re-extraction (raised by D13) | Open — **blocks the schema** |
| Q18 | Extraction stack — arXiv LaTeX source first, and which PDF parser as fallback (GROBID / marker / other)? | Open (READING_NOTES 2.1) |
| Q19 | May fetched full text be cached and served to a second user, or must it be fetched per user? Sharper than Q15 | Open |
| Q20 | Where does background extraction and pass summarization run, given local-first deployment (D3)? | Open |
| Q21 | Does the API need streaming for the reading assistant, and does that change the path endpoint too? | Open |
| Q22 | Reading-event depth model — what counts as having read a paper, for §3 and §5? | Open |

## 13. Changelog

| Date | Change |
|------|--------|
| 2026-08-10 | Initial requirements captured from project owner. Sections 1–9 are the owner's original text, lightly reformatted. Sections 10–13 added as scaffolding. |
| 2026-08-10 | Added §11 Technical Decisions (D1–D4) from owner: metadata adapter layer, hybrid citation-graph + LLM path construction, web app that is local-first but multi-user-ready, Python backend + React frontend. Resolved Q3–Q5, deferred Q1. |
| 2026-08-10 | Added D5–D8 from owner: FastAPI/SQLAlchemy/SQLite→Postgres/Alembic/React+Vite+TS; cold start via topic entry now with library seeding later; CS/ML corpus with field-agnostic code; metadata + abstract ingestion now with full text later. Restructured §11 so every decision records **Now** and **Later**, at owner's instruction that stated future intent belongs in the requirements. Resolved Q2, Q6, Q8, Q9; added Q13–Q14. |
| 2026-08-17 | Added D13 (reading workspace) at owner's request, covering the §7/§8 phase: full-text availability reported but never selected on, anchored assistant claims, extraction provenance, durable note anchors, reading depth, untrusted document text, and faithfulness evaluation before ship. §7 and §8 keep their original wording and gain pointers only. Added `READING_NOTES.md` with the underlying analysis; added Q17–Q22. |
