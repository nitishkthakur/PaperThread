# `backend/` — Agent Guide

Python backend: FastAPI + SQLAlchemy target, SQLite now → Postgres later (§11 D5).
Currently a retrieval POC — **no database or persistence layer exists yet.**

## Setup and commands

```bash
cd backend
/opt/homebrew/bin/python3.11 -m venv .venv       # 3.11+ required
./.venv/bin/pip install -e ".[dev]"

./.venv/bin/python -m pytest -q                  # 104 tests, all offline
./.venv/bin/python -m uvicorn paperthread.api.main:app --reload --port 8000
```

Configuration is read from `../config/paperthread.toml`. Nothing needs an API key to run —
arXiv and OpenAlex are enabled and keyless by default.

## Layout — read the folder's own `AGENTS.md` before working in it

| Path | Responsibility |
|---|---|
| `paperthread/config.py` | Loads `config/paperthread.toml`. The only place provider/model selection is resolved. |
| `paperthread/domain/` | Papers, identity, deduplication. No I/O, no provider types. |
| `paperthread/providers/` | Adapters for external services, grouped by capability. |
| `paperthread/llm/` | LLM provider port, structured output, prompt/judgment cache. |
| `paperthread/retrieval/` | The recommendation pipeline. Stages 1–5 built. |
| `paperthread/api/` | FastAPI routes and wire types. |
| `tests/` | pytest. All offline — no test hits a network. |

## Conventions that matter here

- **No provider-shaped type escapes `providers/`.** Adapters translate into `domain/` types;
  everything above only sees those.
- **Nothing reads environment variables directly for configuration.** Config names an
  `api_key_env`; only secrets come from the environment.
- **A provider failure degrades, it never breaks.** `TopicSearchService._search_one` catches
  everything and reports the failure in the response, because "0 results because a provider was
  rate-limited" must never look like "0 results because nothing matched" (§11 D12).
- **`user_id` keys every user-owned concept** even though only one user exists (§11 D3).
- Python 3.11+ syntax is fine (`X | Y`, `match`). The system Python 3.9 will not run this.

## Not built yet

Pipeline stage 0 (topic decomposition before searching) and stage 6 (personalization),
persistence, the user profile, and mark-as-read. See `paperthread/retrieval/AGENTS.md` and
`docs/RETRIEVAL_NOTES.md`.

**Persistence is the load-bearing gap.** D2 requires prerequisite edges and explanations to be
*stored*, not recomputed per request. `llm/cache.py` honours the key discipline — every entry
stamped `{provider, model, prompt_version}` — but it is a file cache standing in for a table.
The discipline ports unchanged to SQLAlchemy; the storage medium does not.

Nothing needs an API key to run. Set `OLLAMA_API_KEY` and flip `[retrieval.layers] llm = true`
to turn on L4 (reasoned prerequisites and §5 explanations); without it the pipeline runs its
structural implementation and reports that it did.
