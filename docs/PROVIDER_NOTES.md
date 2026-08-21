# Provider Notes — Caveats, Asymmetries, Side Effects

Working notes on the pluggable-provider design (see `REQUIREMENTS.md` §11 D1, D9, D10).

**Why this file exists:** every provider port in PaperThread must support *multiple* providers,
selected by config. The hard part is not the interface — it is that **providers are not
feature-equivalent**. This file records where they differ, so future expansion is a config
change and not a discovery of a broken assumption.

Keep this updated whenever a provider is added, or a capability gap is found.

---

## Part 1 — Paper / metadata providers

### 1.1 The core asymmetry

No single provider gives all three of the things PaperThread needs:

| Provider | Citation graph | Abstracts | Free full text / PDF | Coverage |
|---|---|---|---|---|
| **arXiv** | ✗ none | ✓ plaintext | ✓ **PDF + LaTeX source, all of it** | Preprints only; CS/ML dense |
| **Semantic Scholar** | ✓✓ best (incl. influential-citation flags) | ~ often present, gaps from licensing | ~ `openAccessPdf` only when OA | Broad, strongest in CS |
| **OpenAlex** | ✓ good | ⚠ **inverted index, not plaintext** | ~ `best_oa_location` when OA | Broadest (all fields), CC0 |
| **Crossref** | ~ references sometimes restricted | ✗ unreliable | ✗ | DOI-registered works |
| **Unpaywall** | ✗ | ✗ | ✓ OA PDF locator by DOI | Resolver, not a source |

**Consequence:** the "get citations" provider and the "get full text" provider will usually be
*different providers for the same paper*. The architecture must allow a single logical paper to
be assembled from several providers. A design where one provider owns a paper end-to-end will
break the moment full-text ingestion arrives (D8 Later).

### 1.2 Specific caveats

**C1 — OpenAlex abstracts are inverted indexes, not text.**
OpenAlex ships `abstract_inverted_index` (word → positions) rather than plaintext, for legal
reasons inherited from Microsoft Academic Graph. It must be reconstructed before it can be fed
to an LLM. Reconstruction is lossy on punctuation and whitespace. Coverage is also partial —
roughly 60% of recent works have abstract data, and materially less for pre-2000 work.
→ *Abstract retrieval must be a fallback chain across providers, not a single call.*

**C2 — Abstract coverage gaps are worst exactly where PaperThread needs them most.**
Older papers have the poorest abstract coverage — and older papers are disproportionately the
*foundational* ones the product exists to surface (§5 knowledge gaps). A pipeline that silently
drops papers with no abstract will systematically delete the most important nodes in the path.
→ *Papers with no abstract must be retained and flagged, never filtered out.*

**C3 — arXiv gives free full text but no citation graph.**
This is the sharpest trade. arXiv is the only provider that hands over complete PDFs *and* LaTeX
source for free with no OA gate — which makes it the natural backend for D8's full-text phase in
CS/ML — but it has no references/citations, which D2's candidate generation depends on entirely.
→ *Full-text depth and graph depth are independent axes. Model them separately.*

**C4 — Preprint/published duplication is the biggest data-quality risk.**
The same work exists as an arXiv preprint *and* a published paper, with different IDs, dates,
titles (sometimes), and separate citation counts. Unreconciled, this produces:
- duplicate nodes in a learning path (the same paper twice, at different positions);
- **split citation counts**, which corrupts the centrality scoring that ranking depends on;
- false prerequisite edges between a paper and its own preprint.
→ *Deduplication is not a cleanup step; it is core correctness. Canonical paper identity with
external IDs as aliases (D1) exists precisely for this.*

**C5 — ID spaces don't line up.** DOI, arXiv ID, S2 `corpusId`, S2 `paperId`, OpenAlex `W…`,
legacy MAG ID, PMID. Not all papers have a DOI (many arXiv preprints don't). Reconciliation must
tolerate a missing DOI rather than using it as the primary key.

**C6 — Rate limits differ by orders of magnitude.** arXiv asks ~1 request per 3 seconds;
Semantic Scholar is tight without a key and still modest with one; OpenAlex is generous with a
polite-pool email. → *Rate limiting is per-provider config, and no provider call may sit on a
user request path un-cached.*

**C7 — Citation graphs disagree.** Reference lists for the same paper differ between S2 and
OpenAlex; neither is complete. Querying two providers and unioning improves recall but produces
conflicting edge sets. → *Edges need provenance (which provider asserted them), so conflicts can
be resolved rather than silently last-write-wins.*

**C8 — Indexing lag differs.** arXiv is immediate; aggregators lag by days to weeks. A "latest
developments" section will look different by provider.

**C9 — Redistribution terms differ.** OpenAlex is CC0 and safe to cache freely. Others permit
API access but restrict redistribution of abstracts. Caching for personal use is fine; if
PaperThread becomes multi-user and deployed (D3 Later), **serving cached abstracts to other
users is a different legal question.** → *Flag before deployment; store the source and license
per field.*

**C10 — Non-paper items have no provider at all** (Q14). Blog posts, Distill articles, lecture
notes, textbook chapters. If reading history may contain "articles" (D6 Later), some entries
will have no DOI, no citation-graph position, and no provider. → *A manual/URL-based source is a
first-class provider, not an escape hatch.*

**C11 — OpenAlex records can conflate two distinct works, and the conflation is invisible
within a single provider.** Verified 2026-08-15 against the live API:
[`W4385245566`](https://api.openalex.org/works/W4385245566) is titled *"Exploiting Generative AI
to Scale up Intelligent Tutoring Systems"*, carries the DOI `10.4230/lipics.itp.2023.19` (a
Dagstuhl LIPIcs theorem-proving paper) and lists automated-reasoning authors — while reporting
**`cited_by_count` of 78,979** and being cited by 13 of the 26 candidates a search for
"transformers" returns. The citation *edges* belong to a heavily-cited transformer paper; the
title, DOI, and authorship do not. `W2896457183` shows the same pattern (health-supplement
title, `cosit.2022.18` DOI, 46,035 citations).

This is worse than a missing field, because every internal consistency check passes: the record
agrees with itself, and the co-citation signal built on it is arithmetically correct. It surfaces
as a nonsense title at the *top* of a learning path — precisely where the strongest structural
evidence puts it.
→ *Single-provider structural signal cannot detect this. Cross-provider corroboration of
high-influence nodes is the only real defence, which makes enabling a second `citations` provider
a data-quality requirement rather than a recall optimisation. Until then it is a known, visible
failure mode, not a mystery.*

**C12 — OpenAlex is a METERED API, and `search` is the expensive operation.**
Verified 2026-08-16 from live response headers: `x-ratelimit-limit-usd: 0.1`,
`x-ratelimit-remaining-usd: 0.0001`, `retry-after: 48806` (13.7 hours). Anonymous access
carries ~$0.10/day of credit and a search costs $0.001, so roughly **100 searches per day**
before a half-day lockout — which a single evaluation sweep exhausts. A free API key raises
the allowance about tenfold (`OPENALEX_API_KEY`).

The price structure inverts an assumption this codebase was built on:

| Call | Cost |
|---|---|
| Fetch one work by DOI or OpenAlex ID | **free, unmetered** |
| List/filter (`filter=cites:…`, `filter=cited_by:…`) | $0.0001 |
| Search (`search=…` and `filter=title.search:…`) | **$0.001** |

So **identity lookup is free and topical search is scarce**, which is the opposite of how a
retrieval system is normally budgeted. Two consequences: prefer IDs over strings wherever a
paper is already known, and treat a credit-exhaustion 429 as terminal rather than
transient — its `Retry-After` is measured in hours, and retrying spends more metered calls
against a quota that has already run out.

This also strengthens the citation-graph argument in `RETRIEVAL_NOTES.md`: graph traversal
is ten times cheaper than search, and free when done by ID.

**C13 — arXiv's `all:` field prefix binds only the first token.**
`search_query=all:A Simple Framework for Contrastive Learning of Visual Representations`
returns, as its top hit, a particle-physics paper on the rare B⁰ₛ→μ⁺μ⁻ decay: everything
after the first word becomes unfielded text that the relevance ranker interprets freely.
The phrase must be quoted — `all:"…"` — and for looking up a *known* paper the title field
`ti:"…"` returns it as the sole hit.

→ *Searching for a topic and looking up a named paper are different operations. Conflating
them put chemistry and physics papers into machine-learning reading paths, and made every
arXiv-native paper unresolvable whenever OpenAlex was unavailable.*

### 1.3 Design implications

1. Ports split by **capability**, not by vendor: `PaperSearch`, `CitationGraph`, `FullText`,
   `IdResolver`. A provider implements whichever it can. Config binds capability → provider(s).
2. Each provider **declares its capabilities**; the pipeline queries the registry and degrades
   gracefully rather than assuming.
3. **Fan-out + merge is the default**, not single-provider dispatch: several providers answer,
   results reconcile onto canonical papers.
4. **Field-level provenance** — every field records which provider supplied it, when, and under
   what license.
5. **Content depth is explicit** (`metadata` → `abstract` → `fulltext`) per D8, and independent
   of which provider supplied it.

---

## Part 2 — LLM providers

**Default: Ollama Cloud.** Base URL `https://ollama.com/api` (native) or `https://ollama.com/v1`
(OpenAI-compatible: `/chat/completions`, `/embeddings`, `/models`), authenticated with
`OLLAMA_API_KEY`. It hosts larger open-weight models than local Ollama can run, and the local
Ollama daemon at `http://localhost:11434` speaks the same API — so local and cloud are the same
adapter with a different base URL.

Later, per owner: OpenRouter, Anthropic, others.

### 2.1 Capability asymmetries

**L1 — Embeddings are not universal.** Verified 2026-08-10:

- **Ollama Cloud has NO embedding models.** `ollama.com/search?c=cloud&c=embedding` returns "No
  models found" — every cloud model is generative. Ollama embeddings are **local-only**.
- **Anthropic has no embeddings API at all.**

So *our default LLM provider cannot serve embeddings*, and neither can the most likely later one.
→ *`LLMProvider` and `EmbeddingProvider` are **separate ports**, configured independently. This
is the single most important consequence in this file.*

**L1a — Prefer in-process embedding over any provider API.** Even local Ollama is the wrong path:
it applies **no prefix or template** for embeddings (verified in `server/routes.go` —
`EmbedHandler` processes raw input, unlike the chat handlers), while nearly every embedding model
requires one (`search_document:`, `Represent this sentence for searching…`, `query:`/`passage:`).
Blog claims that Ollama handles this automatically are false, and getting it wrong is a **silent**
quality regression. Ollama also has an open batch-quality bug at batch ≥16 and **changes embedding
values across versions**, which can silently invalidate an index. See `RETRIEVAL_NOTES.md` L2.
→ *Store the embedding model identity and version with every vector; treat a change as a rebuild.*

**L2 — Structured output support varies, and D2 depends on it.** The LLM judge must return
machine-readable verdicts (prerequisite yes/no, role, confidence, explanation). Mechanisms
differ: Ollama's native `format` accepts a JSON schema; the OpenAI-compatible path uses
`response_format`; Anthropic is strongest via tool-use. Open-weight models honor schemas less
reliably than frontier models.
→ *Structured output goes behind one method with per-provider strategy, plus schema validation
and a bounded retry. Never trust raw model text.*

**L3 — Prompt caching and batch APIs are provider-specific.** Anthropic has both (batch is
substantially cheaper); Ollama Cloud has neither. D2's judgment step is a large batch job, so
its cost profile changes by provider.
→ *Batching is an adapter-level concern; the pipeline requests "judge these N pairs" and the
adapter decides how.*

**L4 — Context windows and tool-calling reliability vary widely** across the open-weight models
Ollama Cloud hosts. → *Model capabilities (context length, tool use, JSON mode) belong in
config, declared per model, not assumed from the provider.*

**L5 — Determinism and reproducibility.** Cached LLM judgments (D2) must record **provider +
model + prompt version**. Swapping the default model silently invalidates stored edges; without
that stamp, a path becomes an unreproducible mix of two models' opinions.

**L6 — Cost/quality asymmetry is the point of the abstraction.** Topic decomposition is cheap and
frequent; prerequisite judgment is expensive and quality-critical. → *Provider/model is selectable
**per task**, not globally — e.g. a small model for decomposition, a large one for judgment.*

### 2.2 Design implications

- Ports: `LLMProvider` (chat + structured output), `EmbeddingProvider`. Separate. Per L1.
- Config selects provider **and model per task role**, not one global model.
- Every cached LLM artifact carries `{provider, model, prompt_version}`.
- Adapters normalize: streaming, token accounting, retries, rate limits, structured-output
  strategy. The pipeline never sees provider-shaped types.

---

## Part 3 — Open items

- Which paper providers to enable first (Q1).
- Whether cached abstracts may be served to other users after deployment (C9, ties to D3 Later).
- Vector store choice, once embeddings are needed (Q13).
- Whether reading-history entries must be formal papers (Q14, ties to C10).
