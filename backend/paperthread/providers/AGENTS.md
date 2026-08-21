# `providers/` — Agent Guide

Adapters for external services. **Read `docs/PROVIDER_NOTES.md` before writing or changing any
adapter** — it records where providers are *not* feature-equivalent, which is the whole reason
this folder is shaped the way it is.

| File | Read it for |
|---|---|
| `base.py` | `Capability`, the port protocols, `BasePaperProvider`, `RateLimiter`, `ProviderError` |
| `registry.py` | config name → adapter class. Add new providers here. |
| `papers/` | Paper metadata adapters — has its own `AGENTS.md` |

## Ports are grouped by capability, not by vendor

`SEARCH`, `CITATIONS`, `FULLTEXT`, `ID_RESOLVE`. A provider implements whichever it can and
**declares** them; callers query the registry rather than assuming parity.

This is not over-engineering. **No provider gives citations, abstracts, and free full text
together** — arXiv has free PDFs and LaTeX source but *no citation graph at all*; Semantic Scholar
has the best graph but PDFs only when open-access; OpenAlex ships abstracts as an inverted index
rather than text. A single logical paper is assembled from several providers, so a design where
one provider owns a paper end-to-end breaks the moment full-text ingestion lands.

**Multiple providers active at once is the normal case**, not an edge case.

## Adding a provider

1. Subclass `BasePaperProvider` in `papers/`, implementing only the capabilities you support.
2. Register it in `registry.py`.
3. Add a `[[paper_providers]]` block to `config/paperthread.toml` with its capabilities.
4. Record any new capability gap or licensing constraint in `docs/PROVIDER_NOTES.md`.

No calling code changes. If you find yourself editing `retrieval/` to add a provider, the
abstraction has leaked.

## Rules

- **Raise `ProviderError`, never let a raw `httpx` exception escape.** Callers degrade on failure.
- **Rate limits are per-provider config**, never a global constant — arXiv wants ~1 request per
  3 seconds while OpenAlex is generous. Use `self._throttle()`.
- **Every field gets `FieldProvenance`** (which provider, under what licence). Needed because
  providers disagree, and because whether a cached abstract may be served to another user after
  deployment depends on its source.
- **Set `depth` honestly.** `ABSTRACT` only when an abstract is actually present.
- **Never drop a paper for missing an abstract.** Return it with `depth=METADATA`.

## LLM and embedding providers

Not built yet. When they are: **`LLMProvider` and `EmbeddingProvider` are separate ports.**
Ollama Cloud has no embedding models at all and Anthropic has no embeddings API, so the default
LLM provider cannot serve embeddings. Embeddings are computed in-process, not via an API — see
`docs/RETRIEVAL_NOTES.md` L2 for why routing them through Ollama silently corrupts an index.

## Cost and query-shape traps — both measured, both were live bugs

**OpenAlex is metered (2026).** Searches cost credit; **fetching a work by ID is free**. Anonymous
access is ~100 searches/day before a 13-hour lockout. Set `OPENALEX_API_KEY` for ~10x. A
credit-exhaustion 429 is terminal, not transient — its `Retry-After` is in hours, and retrying
spends more metered calls against an empty quota. See `docs/PROVIDER_NOTES.md` C12.

**`search` and `find_by_title` are different operations.** `search` ranks by relevance to a topic;
`find_by_title` looks up a paper you can already name. Conflating them was the single largest
defect in the system: arXiv's `all:{title}` binds the field prefix to the first token only, so the
top hit for a contrastive-learning title was a particle-physics paper on B⁰ₛ→μ⁺μ⁻ decay. Quote the
phrase (`all:"…"`), and use `ti:"…"` when the paper is known. See C13.

**Rate limiters are shared per provider name, process-wide** (`base.limiter_for`). The limit
belongs to the remote service, not to an adapter instance — and search, expansion and resolution
each build their own adapters, so per-instance limiters multiplied the real request rate and
produced 429s.

**`http_cache.py` caches the request, not the parsed result**, so a parser change does not mean
refetching under a rate limit. Cache hits must never throttle — that is why `cache_get` is checked
before `_throttle`.
