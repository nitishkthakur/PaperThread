# PaperThread — Agent Guide (root)

PaperThread is a personalized research-paper learning platform: it answers **"what paper should
I read next, and why?"** using the user's reading history, topic interests, and detected
knowledge gaps — not similarity search.

## Read this first

1. **[`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) is the source of truth for what this product
   must do.** Read it before designing, planning, or implementing anything. §9 (Core Focus) and
   §10 (Non-Goals) are the scope guardrails, §11 records binding technical decisions, and §12
   lists questions that are still unanswered.
2. **Requirements are owner-controlled.** If the user asks for a new feature or a behavior
   change, *ask whether it should be added to `docs/REQUIREMENTS.md`*, then update the doc and
   its changelog (§12) once confirmed. Never edit requirements silently, and never treat a
   passing remark in conversation as an approved requirement.
3. When an open question in §11 gets answered, move the answer into the relevant section and
   mark the row resolved in the same change.

## Progressive disclosure convention (important)

Documentation for agents is **distributed, not centralized**.

- **Every key subfolder has its own `AGENTS.md`** — backend, frontend, MCP servers, tools,
  utilities, data/ingestion, tests, infra, and any other meaningful module directory.
- **Before working in any folder, read that folder's `AGENTS.md` first.** It is the fastest way
  to learn what lives there and which file to open for which purpose.
- **This root file stays a map, not an encyclopedia.** Folder-specific detail belongs in the
  folder's own `AGENTS.md`. Do not migrate it up here.
- **Creating a significant new directory means creating its `AGENTS.md` in the same change.**
  Changing what a folder does means updating that folder's `AGENTS.md` in the same change.

A subfolder `AGENTS.md` should cover, concisely:

- what the folder is responsible for (and what it is *not*)
- key files, and which file to read for which purpose
- conventions/patterns local to that folder
- commands relevant to that folder (that actually run)
- pointers to child folders that have their own `AGENTS.md`

Keep them factual. Document what exists, not what is planned.

## Repository map

| Path | Contents | Has `AGENTS.md` |
|------|----------|-----------------|
| `docs/` | Requirements, provider caveats, retrieval evidence | yes |
| `config/` | `paperthread.toml` — the one place providers and models are selected | yes |
| `backend/` | Python: FastAPI, domain model, providers, retrieval pipeline | yes |
| `backend/paperthread/domain/` | Papers, canonical identity, deduplication. No I/O. | yes |
| `backend/paperthread/providers/` | External-service adapters, grouped by capability | yes |
| `backend/paperthread/providers/papers/` | arXiv, OpenAlex, Semantic Scholar | yes |
| `backend/paperthread/llm/` | LLM port, structured output, judgment cache | yes |
| `backend/paperthread/retrieval/` | The recommendation pipeline, and the path strategies | yes |
| `backend/tools/` | Evaluation harness and the learner critique rubric | — |
| `backend/paperthread/api/` | FastAPI routes and wire types | yes |
| `backend/tests/` | pytest, fully offline | yes |
| `frontend/` | React + Vite + TypeScript UI | yes |
| `README.md` | Public-facing project description | — |

## Documentation map

| Read this | Before |
|---|---|
| [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) | Any design or implementation work |
| [`docs/PROVIDER_NOTES.md`](docs/PROVIDER_NOTES.md) | Writing or changing a provider adapter |
| [`docs/RETRIEVAL_NOTES.md`](docs/RETRIEVAL_NOTES.md) | Changing retrieval, ranking, or the pipeline |
| [`docs/READING_NOTES.md`](docs/READING_NOTES.md) | Starting the reading-workspace phase (§7/§8, D13) — full text, notes, in-paper assistant |

## Running it

Two processes. The backend needs no API keys — arXiv and OpenAlex are enabled and keyless.

```bash
# terminal 1
cd backend
/opt/homebrew/bin/python3.11 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
./.venv/bin/python -m uvicorn paperthread.api.main:app --reload --port 8000

# terminal 2
cd frontend && npm install && npm run dev     # http://localhost:5173
```

Tests: `cd backend && ./.venv/bin/python -m pytest -q` — 128 tests, fully offline.

**Turning on L4** (LLM-planned paths, reasoned prerequisites, explanations): flip
`llm = true` under `[retrieval.layers]`. The default provider is the **local Ollama daemon**,
which reaches Ollama Cloud models when you have run `ollama signin` — no API key, no key file.
Without L4 the pipeline runs its structural implementation and says so; it does not error.

Optional but recommended: `OPENALEX_API_KEY` (free) for roughly ten times the anonymous daily
allowance — OpenAlex is metered now, and one evaluation sweep exhausts the anonymous tier.

## Current state

**This produces ordered learning paths.** `GET /api/path?topic=…` runs pipeline stages 1–5:
multi-provider search → cross-provider deduplication and RRF → citation-graph expansion →
age-rescaled PageRank and Louvain communities → roles, prerequisite edges and §5 explanations →
DAG ordering under the citation constraint.

The core claim works without any model: a path for "transformers" reaches LSTM, ResNet and
backpropagation, and one for "regularization" reaches Lasso (1996) and Elastic Net (2005) —
none of which are lexical matches for the query. They are found because the topic's own papers
cite them in common.

Stage 4 has **two full implementations**: structural (citation-graph evidence, always available)
and reasoned (L4, when an LLM is configured). The UI distinguishes them everywhere rather than
presenting a measured explanation as a reasoned one.

Not built: **persistence and user profiles** — no database, no reading history, no mark-as-read,
so §5's "why for *this user*" is honestly reported as absent. Also unbuilt: stage 0 (topic
decomposition before searching; subtopics are currently induced from the results' citation graph
instead) and stage 6 (personalization).

### Two families of path builder

`retrieval/path.py` builds paths from **citation-graph structure**; `retrieval/curriculum.py`
builds them from an **LLM-planned teaching sequence**, grounded by resolving every named paper
against a real provider. Both produce a `LearningPath`, so the API and UI do not change with the
choice.

Measured against `backend/tools/CRITIQUE_RUBRIC.md` over ten topics, scored by a learner
role-playing someone studying each topic: **structural 1.5/25, LLM-planned syllabus 15.2/25.**
Centrality finds the important papers; it does not find a teaching order. For "dropout" the
structural pipeline surfaces ImageNet and ResNet, while the planned path is backpropagation →
weight decay → bagging → **Dropout** → analyses → extensions.

The structural pipeline stays: D12 requires a usable result with no LLM, and its citation graph
is what grounds the planned paths rather than competing with them.

### Tuning loop

`backend/tools/eval_paths.py` runs a topic set through a (strategy, model) combination and writes
transcripts; a critique agent scores them against the rubric. Strategy, model chain, and prompts
are all configuration, so the ranking algorithm is tunable without code changes.

**Known upstream defects, both verified and both live:** OpenAlex conflates distinct works into
one record (`PROVIDER_NOTES.md` C11) and is now a **metered API** where search costs credit and
ID lookup is free (C12). arXiv's `all:` prefix binds only the first token, which put physics
papers into ML reading paths until it was quoted (C13).

## Working agreements

- **Don't invent product scope.** If a behavior isn't in `docs/REQUIREMENTS.md` and wasn't asked
  for, raise it rather than building it.
- **Ordering is a first-class product concept.** Learning paths are sequences (A → B → C) with
  prerequisite edges; anything that flattens them into an unordered list is a bug, not a
  simplification.
- **Every recommendation carries an explanation** (why it matters, what it assumes, what it
  teaches, why *this* user). Treat the explanation as part of the data model, not UI garnish.
- **Local-first is a deployment mode, not an architecture.** This is a web app that will become
  multi-user. Key every user-owned row by `user_id`, keep auth behind an interface, keep the DB
  layer portable, and never add a shortcut a hosted deployment couldn't support. See §11 D3.
- **No provider-specific paper IDs in the domain model.** Metadata sources sit behind an adapter
  (§11 D1); PaperThread owns canonical paper identity and stores external IDs as aliases.
- **Every decision in §11 has a *Now* and a *Later*, and the *Later* is binding intent.** Build
  the *Now*, but leave the seam — don't let a v1 shortcut make the stated later version a
  rewrite. The three live ones:
  - v1 stores **metadata + abstracts only**, but full paper text is coming (D8) — paper content
    is depth-tagged and ingestion is re-runnable at greater depth.
  - v1 onboards by **topic entry**, but users will later seed history by picking papers they've
    already read (D6) — reading history must be populatable in bulk, outside any learning path.
  - v1 ingests **CS/ML only**, but the code must be field-agnostic (D7) — the restriction lives
    in ingestion config, never in the schema, domain model, or prompts.

## Commands

| Command | From | Does |
|---|---|---|
| `./.venv/bin/python -m pytest -q` | `backend/` | 128 tests, fully offline |
| `./.venv/bin/python -m uvicorn paperthread.api.main:app --reload --port 8000` | `backend/` | API on :8000, docs at `/docs` |
| `npm run dev` | `frontend/` | UI on :5173, proxies `/api` |
| `npm run build` | `frontend/` | `tsc -b && vite build` |
| `curl 'localhost:8000/api/health'` | anywhere | Config path, enabled providers, whether L4 is available and why not |
| `./.venv/bin/python tools/eval_paths.py --strategy syllabus --run r1` | `backend/` | Build paths for the topic set and write scoreable transcripts |

Building a path for a cold topic takes a while — stage 2 makes tens of provider requests under
per-provider rate limits. LLM judgments are cached, so a repeat topic pays only the provider
round-trips.
