# `api/` — Agent Guide

FastAPI routes and wire types. Thin by design: parse, delegate to `retrieval/`, serialize.

| File | Contents |
|---|---|
| `main.py` | `create_app()`, the `/api` router, and the Pydantic response models |

## Endpoints

| Route | Purpose |
|---|---|
| `GET /api/health` | Config path, enabled vs known providers, active layers. Use this first when debugging. |
| `GET /api/search?topic=…&limit=…` | Stage 1 only. Ranked candidates plus per-provider outcomes. |
| `GET /api/path?topic=…&limit=…` | **The product's endpoint.** An ordered path with levels, roles, prerequisite edges and §5 explanations. Slow on a cold topic — stage 2 makes tens of rate-limited provider requests. |

Run it:

```bash
./.venv/bin/python -m uvicorn paperthread.api.main:app --reload --port 8000
```

Interactive docs at `http://127.0.0.1:8000/docs`.

## Conventions

- **Domain types never go on the wire.** `PaperOut` / `SearchOut` are the contract; converting
  happens in `_to_paper_out` / `_to_search_out`. Changing `domain/models.py` must not silently
  change the API.
- **The response reports honesty, not just data.** `providers`, `layers_used`, `degraded`, and
  `notes` are part of the contract — the UI shows them, and a failed provider must be
  distinguishable from an empty result set. Do not trim them for tidiness.
- **HTTP only, no local-only shortcuts** (§11 D3). Nothing here may assume the frontend shares a
  filesystem or process with the backend.
- **Config is cached** via `lru_cache` on `get_config`. Restart the server after editing
  `config/paperthread.toml`.
- **`PaperOut.found_by` is provenance, not an explanation.** The §5 explanation lives in
  `ExplanationOut`, on every `StepOut`.
- **`ExplanationOut.source` must survive to the client.** `structural` and `llm` explanations are
  not interchangeable, and presenting a measured one as reasoned is the most misleading thing
  this API could do. Same for `EdgeOut.source` and `asserted_by`.
- **`stages_run` is part of the contract**, including `judge:llm`, which is how a caller knows
  whether prerequisites were judged or inferred.

## Before deployment

CORS is currently open to `localhost:5173` for dev convenience. Tighten it, and add
authentication behind an interface, before this is exposed (§11 D3). No auth exists yet; the
single local user comes from `config.default_user_id`.
