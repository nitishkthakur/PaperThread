# `llm/` — Agent Guide

The LLM provider port (§11 D10) and everything that makes model output safe to store.

| File | Read it for |
|---|---|
| `base.py` | `LLMRequest`/`LLMResult`, the validate-and-repair loop, `extract_json`, `validate` |
| `registry.py` | `LLMClient` — the seam callers use. Model-per-role, caching, concurrency. |
| `openai_compatible.py` | Ollama Cloud, local Ollama, OpenRouter — one adapter, three base URLs |
| `anthropic.py` | Anthropic Messages API, structured output via a forced tool call |
| `cache.py` | Content-addressed cache stamped with `{provider, model, prompt_version}` |
| `prompts.py` | Versioned prompts and schemas for the structural pipeline |
| `curriculum_prompts.py` | Prompts for LLM-planned learning paths, versioned separately |

## The rules here, and why each exists

**Schema enforcement does not work on Ollama, and failing to know that costs a retry per
call.** Verified 2026-08-16 against Ollama 0.32.12 with cloud models: `response_format` of
type `json_schema` is **accepted and silently ignored** — the model returns prose and the
endpoint returns 200. Ollama's native `format: <schema>` behaves identically. Only
`json_object` plus an explicit "reply with JSON" instruction actually produces JSON. The
mode is therefore **configured per provider** (`structured_output`), not negotiated on
error, because the error never arrives.

**Models are chains, not names.** `[llm.roles]` maps a role to `[primary, *fallbacks]`, and
`LLMClient.structured` walks the chain when a model is missing, unreachable, rate limited,
or cannot produce valid output. Every model in the chain is checked against the cache
*before* any call, so a path already built by the second choice is not rebuilt when the
first comes back. `LLMResult.fell_back` records what happened.

**Reasoning models leave `content` empty.** The cloud models put their thinking in
`reasoning`/`thinking`, and when the token budget runs out mid-thought they return an empty
`content` with `finish_reason: length` — a truncation that looks exactly like a refusal.
Budget generously (8k+) and read the adapter's error message before blaming the prompt.

**Raw model text is never trusted.** Every call declares a JSON schema; the response is parsed,
validated, and on failure fed back to the model with the specific validation error for a bounded
number of retries. The repair turn matters more than the retry count — at temperature 0 a blind
retry reproduces the same mistake exactly.

**`LLMProvider` and `EmbeddingProvider` are separate ports.** Not tidiness: Ollama Cloud has **no
embedding models at all**, and Anthropic has **no embeddings API**. A single "AI provider"
abstraction breaks on the first two providers we tried. Embeddings are computed in-process; never
route them through here.

**Model is selected per task role**, via `[llm.roles]`. Topic decomposition is cheap and frequent;
prerequisite judgment is expensive and quality-critical. One global model prices one of them
wrong.

**Every cached artifact records `{provider, model, prompt_version}`, and that stamp is part of the
cache key.** Changing the default model must *miss*, not silently serve judgments a different
model made. D2 edges are persisted data, and this is what keeps them attributable.

**Callers ask `available()` before running, they do not catch exceptions to find out.** A missing
API key is a known state, not an error — the pipeline must be able to build a path without L4 and
say so (§11 D12).

## Curriculum prompts

`curriculum_prompts.py` has its **own `VERSION`** so that iterating on learning-path wording
does not invalidate the structural pipeline's cached judgments. Prompts are the primary
tuning surface for path quality — treat a prompt edit as an experiment with a measurement,
not as a wording preference. `tools/eval_paths.py` plus `tools/CRITIQUE_RUBRIC.md` is how
that gets measured.

## Prompt conventions

- **No prompt may presume the field.** D7 restricts the *corpus* to CS/ML; it does not let the
  code assume it. Nothing in `prompts.py` says "machine learning" or gives an ML example, so
  widening the corpus stays a configuration change.
- **The model chooses among supplied ids; it never names papers.** Every prompt passes a numbered
  shortlist and requires answers using those labels, and `judgment.py` re-checks every returned id
  against the shortlist. Asked for "the foundational papers on X", an LLM produces confident,
  well-formatted, nonexistent citations.
- **The judgment prompt is deliberately biased toward "no".** Only a small minority of citations
  are substantively important; the rest are perfunctory. A false prerequisite is invisible to the
  reader — it just looks like a longer path — which makes it the expensive error.
- **Bump `VERSION` when a prompt's meaning changes.** The content hash catches edits you forget to
  version, but the version is what makes stored provenance readable.

## Adding a provider

Write an adapter subclassing `BaseLLMProvider` (implement `_complete` only — the validate-and-
repair loop is inherited), register it in `registry.py` under its `kind`, and add a
`[llm.providers.<name>]` block to `config/paperthread.toml`. No calling code changes.

The registry is keyed by **`kind`, not vendor name**, because three vendors already share one wire
format. Record any capability asymmetry in `docs/PROVIDER_NOTES.md` Part 2.

## Testing

`../../tests/test_llm.py`, fully offline. No test may make a real API call — test `extract_json`,
`validate`, and `LLMCache` directly, and fake the provider for anything above them.
