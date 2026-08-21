# `providers/papers/` — Agent Guide

Paper metadata adapters. Each translates one external API into `domain/` types.

| Provider | Capabilities | Why it's here | Key caveat |
|---|---|---|---|
| `arxiv.py` | search, fulltext | **Only source of free full PDFs + LaTeX for every paper**, no OA gate | **No citation graph at all** |
| `openalex.py` | search, citations, id_resolve | Broadest coverage, **CC0** so caching is unambiguously safe | Abstracts are an **inverted index**, not text |
| `semantic_scholar.py` | search, citations | Best citation graph; **per-edge citation intent for free** | Needs an API key in practice; abstract gaps |

## Per-file notes

**`arxiv.py`** — Atom XML, not JSON. Uses `follow_redirects=True` because `export.arxiv.org`
301s http→https. Rate limit ~0.34/s, set in config. Enabled by default, no key.

**`openalex.py`** — `reconstruct_abstract()` rebuilds plaintext from `abstract_inverted_index`,
which OpenAlex ships instead of text for legal reasons inherited from MAG. **Reconstruction is
lossy on punctuation and whitespace**: fine for retrieval and for feeding an LLM, but it is not a
verbatim abstract and must not be shown to a user as one. Coverage is partial and worst for
pre-2000 work. Set `mailto` in config to get the faster "polite pool".

**`semantic_scholar.py`** — the reason to enable this is `intents`, `contextsWithIntent`, and
`isInfluential` on citation/reference edges. **"A cites B" does not mean "B is a prerequisite for
A"** — only ~14.6% of citations are substantive rather than incidental — and S2 hands us intent
labels with no model to host. Disabled by default because unauthenticated rate limits are
punishing; set `SEMANTIC_SCHOLAR_API_KEY` and flip `enabled = true`.

## Writing a new adapter

Follow `openalex.py` as the reference. Required of every adapter:

- `await self._throttle()` before every outbound request.
- Wrap transport and parse errors in `ProviderError` — never let `httpx` or JSON errors escape.
- Normalize every external ID through `domain.identity.normalize_external_id`. Raw arXiv IDs
  carry version suffixes (`2301.12345v3`) that will silently create duplicate papers.
- Call `canonical_id_for(paper)` before returning.
- Attach `FieldProvenance` with the provider's actual licence, not a placeholder.
- Return papers with `depth=METADATA` rather than dropping them when an abstract is missing.
- 1-based `rank` on `SearchHit` — RRF depends on it.

## Testing

Adapter parsing is tested through pure functions (e.g. `reconstruct_abstract` in
`tests/test_fusion.py`). **No test hits the network.** If you add an adapter, extract its parsing
into a testable function rather than testing against a live API.
