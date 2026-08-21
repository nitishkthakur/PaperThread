# `config/` — Agent Guide

| File | Purpose |
|---|---|
| `paperthread.toml` | **The one place providers and models are selected** (§11 D9) |

## The rule this folder enforces

No provider choice is hardcoded anywhere in the application, and none is read from scattered
environment lookups. If you are adding an `os.environ.get("SOME_PROVIDER_URL")` in application
code, it belongs here instead.

Environment variables are for **secrets only**, and always through the `api_key_env` indirection —
config names the variable, `config.py` reads it. No key is ever written in this file.

Loaded by `backend/paperthread/config.py`. The server caches it, so **restart after editing**.

## Sections

| Section | Selects |
|---|---|
| `[[paper_providers]]` | Paper sources, each with a declared `capabilities` list |
| `[llm]` | LLM provider, plus **model per task role** |
| `[embeddings]` | Embedding provider and model — deliberately separate from `[llm]` |
| `[retrieval]` | Candidate limits, RRF `k`, and the per-layer on/off switches |
| `[retrieval.expansion]` | Stage 2 request budget — every knob costs provider calls |
| `[retrieval.graph]` | Stage 3 parameters: PageRank damping, age-cohort width, path size |
| `[llm.cache]` | Where judged edges and explanations persist, and whether caching is on |

## Things that look odd but are deliberate

- **`capabilities` is per-provider, not implied by the provider's name.** arXiv has `fulltext` but
  not `citations`; OpenAlex has `citations` but its abstracts need reconstruction. The pipeline
  asks for a capability and gets whoever can serve it.
- **`[embeddings]` is separate from `[llm]`, with its own provider field.** Ollama Cloud has no
  embedding models at all, and Anthropic has no embeddings API — so the LLM provider cannot serve
  embeddings. Merging these two sections would break on the first provider switch.
- **`[llm.roles]` maps role → an ORDERED CHAIN, not one global model.** The first entry is the
  primary; the rest are fallbacks, tried when a model is missing, unreachable, or cannot produce
  usable output. A bare string still works and means "no fallback". Topic decomposition is cheap
  and frequent; curriculum planning is expensive and quality-critical, so one global model prices
  one of them wrong.
- **`structured_output` per LLM provider is not cosmetic.** Ollama accepts `json_schema` and
  silently ignores it, so a wrong value here does not fail — it just burns a repair retry on every
  call. Ollama needs `json_object`; Anthropic needs `tool`.
- **`api_key_env` is absent for `ollama_local` on purpose.** The daemon authenticates to Ollama
  Cloud with credentials `ollama signin` stored, so naming a variable would make a missing
  environment variable look like a missing capability.
- **OpenAlex is metered.** Set `OPENALEX_API_KEY` (free) for roughly ten times the anonymous
  allowance; without it, ~100 searches a day. Lookups by ID are free either way
  (`PROVIDER_NOTES.md` C12).
- **`[retrieval.layers]` defaults to lexical-only.** Every other layer is off until its dependency
  is configured, and the system must remain fully usable that way (§11 D12).
- **`embeddings.model_revision` is not cosmetic.** It is stamped onto every stored vector;
  changing it means the index must be rebuilt.
- **`retrieval.graph.pagerank_damping = 0.5`, not 0.85.** Citation networks are not the web:
  ~42–51% of a bibliography's references cite each other, so reference-following paths are
  short. At 0.9 PageRank degenerates into citation count. Evidence in `docs/RETRIEVAL_NOTES.md`.
- **`retrieval.expansion.min_co_citations` is a precision knob, not a recall one.** Lowering it
  to 1 admits every reference of every candidate, which swamps PageRank with the ancestry of
  whatever lexical false positives got through.
- **An unknown key in `[retrieval.expansion]` or `[retrieval.graph]` raises `ConfigError`.** A
  typo'd tuning knob that silently leaves the default in place looks exactly like "the setting
  had no effect", which is the most expensive kind of config bug to chase.
- **`[llm.cache]` is standing in for a database table.** D2 requires judged edges to be stored,
  not recomputed. Deleting the cache directory is safe; it costs API calls, not correctness.

## Turning on L4

```bash
export OLLAMA_API_KEY=...          # https://ollama.com/settings/keys
# then set llm = true under [retrieval.layers]
```

Without both, the pipeline runs its structural implementation and reports that it did — it does
not error. `GET /api/health` reports `llm_available` and the reason when false.

## Adding a provider

Add a block here **and** register the adapter in
`backend/paperthread/providers/registry.py` (paper providers) or
`backend/paperthread/llm/registry.py` (LLM kinds). A provider named here with no registered
adapter raises `UnknownProviderError` at search time, listing what is registered.

Record any new capability gap or licensing constraint in `docs/PROVIDER_NOTES.md`.
