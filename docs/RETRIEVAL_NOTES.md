# Retrieval Notes — Evidence for the Layered Design

Evidence behind `REQUIREMENTS.md` §11 D12 (layers L0–L4). Records **which technique sits at each
layer and why**, with sources, so the choices can be re-examined rather than inherited.

**Status:** in progress. Reranking (L3) and classical NLP (L1) sections are complete. Lexical
retrieval (L0), embeddings (L2), and citation-graph/curriculum algorithms are pending research.

**Standing caveat on all latency numbers here:** published CPU throughput for the *same model*
varies by an order of magnitude across sources depending on batch size, sequence length, thread
count, and quantization. Every number below needs one afternoon of verification on the actual
target laptop with actual abstracts before it becomes a commitment.

**Standing caveat on all quality numbers here:** reranker and embedding vendors each publish
benchmark figures from their own harness with their own first-stage retriever. Those are **not
comparable across vendors** — changing the first stage moves the score by several points. Below,
single-harness comparisons are marked as such; vendor self-reports are marked and must only be
read in isolation.

---

## L3 — Reranking

### Decision

**`cross-encoder/ettin-reranker-32m-v1` as default, `ettin-reranker-68m-v1` as the quality
config.** ONNX/int8-quantized, batched, over ~150–200 candidates. Apache-2.0, ~130 MB on disk.

### Why

The [Ettin reranker release](https://huggingface.co/blog/ettin-reranker) is the only
apples-to-apples table found — 13 public rerankers plus 6 new ones, **one harness, one metric**
(NDCG@10, MTEB eng v2 Retrieval), three hardware tiers:

| Model | Params | NDCG@10 |
|---|---|---|
| Qwen3-Reranker-4B | 4B | 0.6367 |
| mxbai-rerank-large-v2 | 1.54B | 0.6115 |
| ettin-reranker-1b | 1.0B | 0.6114 |
| ettin-reranker-400m | 402M | 0.6091 |
| **ettin-reranker-68m** | **69M** | **0.5915** |
| **ettin-reranker-32m** | **33M** | **0.5779** |
| **bge-reranker-v2-m3** | **568M** | **0.5526** |
| ms-marco-MiniLM-L12-v2 | 33M | 0.5066 |

**A 69M model beats a 568M `bge-reranker-v2-m3` by +3.9 points.** `bge-reranker-v2-m3` is the
default in essentially every RAG tutorial written before 2025 and is now obsolete for English
work — its remaining justification is 100+ language coverage, which D7 (English CS/ML) does not
need.

CPU latency (Ettin harness, i7-13700K, max_length 512), adjusted for our case — a laptop is
~1.5–3× slower than that desktop, but title+abstract is ~200–350 tokens rather than 512, which
is ~1.5–2× faster. These roughly cancel:

| Model | pairs/sec | ~200 candidates |
|---|---|---|
| ettin-reranker-17m | 267 | ~0.8 s |
| **ettin-reranker-32m** | **93** | **~2 s** |
| ettin-reranker-68m | 31 | ~6 s |
| ettin-reranker-150m | 14 | ~14 s |
| bge-reranker-v2-m3 | 6.0 | ~33 s |

Anything ≥500M on CPU is a background-job technique, not a request-path one.

### What reranking is actually worth

The cleanest four-stage ablation found ([arXiv 2604.01733](https://arxiv.org/html/2604.01733v1),
23,088 queries):

| Stage | Recall@5 | nDCG@10 | MRR@3 |
|---|---|---|---|
| BM25 | 0.644 | 0.515 | 0.411 |
| Dense | 0.587 | 0.466 | 0.351 |
| Hybrid RRF | 0.695 | 0.551 | 0.433 |
| **Hybrid + cross-encoder** | **0.816** | **0.683** | **0.605** |

Read the deltas: fusion buys **+5.1 pp** Recall@5 over the best single retriever; the reranker
buys a further **+12.1 pp on top of fusion** — roughly **2.5× what fusion is worth**, and the
largest single intervention in the pipeline. Note also that **BM25 beat dense retrieval here**
(0.644 vs 0.587). Sparse retrieval is not legacy.

### The reason it earns its place in *this* product

The standard objection is real: a reranker only reorders, so it is bounded by the retriever's
Recall@k, and the usual heuristic is to skip it when recall is already >90%.

That objection doesn't apply here, for a product-specific reason. **The reranker's job in
PaperThread is not to improve a result list a human reads — it is to decide which ~20 of ~300
candidates are worth spending an L4 LLM call on, and to keep D2's *persisted* edge set from
filling with junk.** The scarce resource is the LLM judgment budget and the integrity of the
cached edge graph, not recall. A +20 pp Hit@1 at the top of that funnel converts directly into
fewer wasted LLM calls.

Rerank the **fused** candidate list — reranking a hybrid list beats reranking a BM25-only or
dense-only list, so L3 does not replace the L0+L2 fusion.

### Rejected at L3

| Rejected | Reason |
|---|---|
| `bge-reranker-v2-m3` | 568M params, *worse* than a 69M Ettin model, ~33 s vs ~6 s for 200 candidates on CPU. Tutorial default, obsolete for English. |
| **jina-reranker-v3 / v3.5** | Current BEIR SOTA (61.94 / 63.20) but **CC-BY-NC-4.0** — unshippable under D3's deployed multi-user future. Don't benchmark what can't be deployed. |
| Qwen3-Reranker, Nemotron, any causal-LM reranker | Autoregressive decoding, not parameter count, is the cost: >1 s for 100 candidates **on an H100**. Non-starter on a laptop CPU. |
| monoT5 / monoBERT | 2020–2022 baselines; monoT5-3B is beaten on both axes by a 69M encoder. |
| Multilingual rerankers generally | D7 is English CS/ML; paying capacity for 100+ unused languages. |

**Upgrade path:** `gte-reranker-modernbert-base` (149M, Apache-2.0, 8192 context) becomes the
right choice when D8's full-text phase lands and long-context scoring matters — but at ~20–40 s
per 200 candidates on CPU it belongs in a precompute path, not the request path.

### ColBERT / late interaction — rejected

Quality is real (`answerai-colbert-small-v1`, 33M, BEIR 0.533, beating bge-base at 3× its size;
`mxbai-edge-colbert-v0` 17M beating full ColBERTv2). **Storage kills it:** 10,000 documents costs
275–366 MB in fp16 for the *edge-optimized* variants, versus ~30 MB for a 768-dim fp32 dense
index — roughly **10×**. At a realistic 200k–500k paper CS/ML corpus that is **5.5–18 GB of
multi-vector index**, and pgvector has no first-class MaxSim, so it means maintaining a second
index system alongside the database — directly against D5.

It solves a latency problem we don't have (sub-100 ms over 100k+ candidates); our candidate sets
are hundreds and L4 already puts seconds on the clock. JaColBERT is Japanese — irrelevant under
D7.

### LLM-as-reranker — already have one, don't build a second

[arXiv 2508.16757](https://arxiv.org/html/2508.16757v1) (EMNLP 2025, 22 methods / 40 variants) is
the rigorous evaluation. On familiar benchmarks listwise LLM rerankers win (RankGPT-GPT-4 75.59
nDCG@10 on TREC DL19). On **FutureQueryEval** — queries about content postdating training —
everything drops **12–15 points**, and ListT5 collapses to 9.72. Cross-encoders degrade more
gracefully. Named failure modes: listwise positional bias, pairwise inconsistency, hallucinated
justifications.

> ⚠️ **This is the most important finding in this section.** PaperThread's value proposition is
> surfacing literature the model hasn't memorized. An LLM reranker's benchmark score is partly a
> memorization score, and our operating regime is exactly where it degrades.

Also: **D2 stage 2 already is an LLM ranking a shortlist.** A separate RankGPT/RankZephyr pass
would be a third ranking pass over the same candidates for the same money. RankZephyr is 7B —
minutes per query on a laptop, violating D12's offline rule outright. If LLM ranking is ever
wanted, **Setwise** (SIGIR 2024, [code](https://github.com/ielab/llm-rankers)) is the efficient
formulation.

---

## L1 — Classical / local NLP

### spaCy: keep, but narrowly. scispaCy: reject.

**Reject scispaCy outright.** Every high-value component is **biomedical**: NER for
genes/proteins/chemicals/diseases, and five knowledge bases (UMLS, MeSH, RxNorm, GO, HPO) that
contain **none** of "diffusion model", "attention", "LoRA", "contrastive learning", or "RLHF".
Linking CS/ML abstracts to UMLS is actively harmful — it will link "transformer", "attention",
and "convolution" to unrelated medical concepts, a documented failure mode requiring
disambiguation post-processing. ~1 GB of downloads for negative value under D7.

The one domain-agnostic thing it wraps is Schwartz–Hearst abbreviation detection — ~100 lines to
vendor directly. Don't import a biomedical NLP stack for one function.

**Where spaCy is right:** tokenization + lemmatization for the L0 BM25 index (`en_core_web_sm`,
~12 MB, deterministic, offline), and noun-chunk extraction as cheap candidate-term generation.
Load it as `spacy.load("en_core_web_sm", disable=["parser","ner"])`.

**Where spaCy is wasted effort:**

- **NER on ML text** — no CS/ML entity model exists; `en_core_web_sm` gives PERSON/ORG/GPE, which
  is noise. Training one needs annotation data we don't have.
- **Sentence segmentation** — spaCy scores **52.1%** on the pySBD Golden Rule Set (60.4% with the
  parser) versus **pySBD 97.9%** and syntok 70.8%. Scientific text is the adversarial case: "et
  al.", "Fig. 3", "i.e.", inline citations, decimals. Under D8 (abstracts only) it's barely
  needed; when full text lands, use pySBD or `wtpsplit`, not spaCy.
- **Dependency parsing** — nothing consumes parse trees and it's the slowest component.

### Acronym expansion: yes, but at L0 only

Schwartz–Hearst is mature (F ≈ 91.4% on the Ab3P corpus) and **domain-agnostic** — it keys on the
`Long Form (SHORT)` pattern, which ML abstracts are saturated with.

The frequently-cited "acronym expansion almost always improves retrieval" result is **TREC
Genomics-era, lexical-retrieval, biomedical** — pre-BERT. Modern dense embeddings already place
"DDPM" near "denoising diffusion probabilistic models". **So it will not meaningfully move dense
retrieval or reranking metrics.**

It earns its keep in exactly two places:

1. **The L0 BM25 index** — to BM25, "DDPM" and "denoising diffusion probabilistic models" are
   unrelated strings. Expanding at index time *and* query time makes them collide. ~100 LOC, no
   model, no download, deterministic — improving precisely the layer D12 requires to be useful
   with zero external dependencies.
2. **Topic-entry normalization at cold start (D6)** — users type `GAN`, `RLHF`, `VAE`, `MoE`.

### Subtopic decomposition (§4) — cannot be done well without an LLM

This is the most consequential finding, because §4 requires subtopics **with A→B→C ordering**.

**Keyphrase extractors are the wrong shape of tool.** YAKE/RAKE/TextRank/TF-IDF return phrases
that literally occur in the text — `"denoising diffusion probabilistic"`, `"reverse process"`,
`"sampling steps"` — unordered, redundant, no hierarchy, no prerequisite relation. §4 needs
**taxonomy induction**, a categorically different task. Quality is also poor in absolute terms:
KeyBERT(MMR) leads the family, **YAKE scores lowest**, and unsupervised KPE broadly sits around
F1@10 ≈ 0.25–0.40.

**Topic modeling is closer but wrong at our scale.** BERTopic has `hierarchical_topics()` and
beats LDA/NMF on coherence, but: (a) UMAP+HDBSCAN needs thousands of documents — at the ~300
candidates a topic query returns, HDBSCAN labels most as outliers; (b) **UMAP is stochastic**, so
the same topic yields a different curriculum per run, fatal for §6's incremental path updates;
(c) labels are word bags — BERTopic's own recommended fix is an LLM representation model.

Direct evidence on this exact task: hierarchical taxonomy generation over scientific papers
([arXiv 2509.19125](https://arxiv.org/pdf/2509.19125)) and its comparators (CHIME, GoalEx,
TnT-LLM, TaxoAdapt) are **all LLM-guided**. Nobody is publishing competitive purely-unsupervised
scientific taxonomy induction in 2026.

**The design that follows — grounded decomposition, not either/or:**

1. **L2** — retrieve ~300 papers (hybrid + RRF).
2. **L1** — cluster abstract embeddings with **deterministic agglomerative or fixed-seed k-means**
   (*not* HDBSCAN), extract per-cluster discriminative terms via **c-TF-IDF** (cheap,
   deterministic, no new dependency).
3. **L1, better** — run **Louvain/Leiden community detection on the citation subgraph** of the
   candidates. This graph is already built for D2. Supporting evidence: [Topic Is Not Agenda
   (arXiv 2605.07158)](https://arxiv.org/html/2605.07158v1) finds **SPECTER, SPECTER2, and SciNCL
   cosine neighborhoods do *not* match a well-constructed citation partition** — despite being
   trained on citations. Text similarity and intellectual lineage are different structures, and
   §4/§5 care about lineage.
4. **L4** — hand the LLM the clusters, their terms, and representative papers, and ask it to
   **name and order** them. One cheap call, grounded in our corpus, so it cannot invent a subtopic
   with no papers behind it.

This satisfies D12 exactly: L1 alone yields a real (if unnamed) subtopic partition; L4 refines
rather than replaces it.

**For L0, with no model weights at all:** don't reach for YAKE. Use **already-ingested metadata**
(arXiv categories, author keywords, ACM CCS concepts) plus **citation-graph community structure
and in-degree centrality within the candidate set**. That is a genuinely useful offline result. A
YAKE keyword list is not.

### Rejected at L1

| Rejected | Reason |
|---|---|
| scispaCy | Entirely biomedical; ~1 GB for zero-to-negative value under D7. |
| spaCy NER / parser | No CS/ML entity model; nothing consumes parse trees; slowest component. |
| spaCy sentence segmentation | 52.1% vs pySBD's 97.9% on the Golden Rule Set. |
| YAKE / RAKE / TextRank as subtopic engine | Surface phrases, not taxonomies — wrong output shape for §4. |
| LDA / NMF / Top2Vec | Beaten by BERTopic, itself beaten by LLM labeling. |
| BERTopic HDBSCAN at query time | Needs thousands of docs; mostly outliers at n≈300; stochastic → unstable paths, breaking §6. |
| `spacy-llm` | Second provider-orchestration layer; D10 already owns that. |

---

## Cross-cutting finding

> **The strongest signal in this system is not NLP at all — it is citation-graph structure.**
> Community detection and centrality on the candidate subgraph outperform text-based approaches
> for finding lineage, and text embeddings demonstrably fail to reproduce citation partitions
> even when trained on citations. This is the product's moat, and it runs at L0/L1 with no model
> weights and no network.

---

---

## L2 — Embeddings

### 🚩 Blocking finding: Ollama Cloud has no embedding models

`ollama.com/search?c=cloud&c=embedding` returns **"No models found."** Every Ollama Cloud model
is generative (`gpt-oss:120b`, `deepseek-v3.1:671b`, `kimi-k2:1t`, `qwen3-coder:480b`, …).
**Embeddings via Ollama are local-only.**

This is the same structural point as `PROVIDER_NOTES.md` L1 (Anthropic has no embeddings API),
arriving from the other direction, and it confirms the design: **`LLMProvider` and
`EmbeddingProvider` must be separately configured.** Our default LLM provider cannot serve
embeddings at all. See `REQUIREMENTS.md` §11 D10.

### Decision: embed in-process, not through Ollama

Even for *local* Ollama, route embeddings through `sentence-transformers` / ONNX in-process
instead. Three concrete defects, all source-verified:

1. **Ollama applies no prefix or template for embeddings** — verified by reading `EmbedHandler`
   in `server/routes.go`, which processes raw input with no template rendering, unlike
   `GenerateHandler`/`ChatHandler`. But nearly every model in the library *requires* prefixes:
   `nomic-embed-text` needs `search_document:` / `search_query:`; `mxbai-embed-large` needs
   `Represent this sentence for searching relevant passages:`; BGE, E5, EmbeddingGemma, and
   Qwen3 all have their own. Several blog posts claim Ollama handles this automatically —
   **that claim is false.** Getting it wrong is a *silent* quality regression, not an error.
2. **Batch embeddings degrade at batch ≥16** ([#6262](https://github.com/ollama/ollama/issues/6262),
   open) — `bge-large` cosine similarity vs single-item drops from ~0.9999 to ~0.95. There is
   also a race condition zeroing parts of the output array
   ([#8713](https://github.com/ollama/ollama/issues/8713), seen on Apple M1 and Intel CPU).
3. **Embeddings change across Ollama versions**
   ([#14449](https://github.com/ollama/ollama/issues/14449),
   [#3777](https://github.com/ollama/ollama/issues/3777)) — nomic-embed-text values differ
   between v0.4.6 and v0.17.0, with divergence growing on longer inputs. **An Ollama upgrade can
   silently invalidate the whole index.**

> Defect 3 generalizes into a rule: **store the embedding model identity and version alongside
> every vector**, and treat a change as an index rebuild. This is the same discipline D10 already
> requires for cached LLM judgments (`{provider, model, prompt_version}`).

In-process also unlocks the ONNX int8 backend, real tensor batching, and explicit pooling
control.

### Model choice

Corrected specs (primary sources — HF cards, config.json, safetensors param counts):

| Model | Params | Dim | Ctx | License | MTEB |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 22.7M | 384 | 512 | Apache-2.0 | — |
| **bge-small-en-v1.5** | **33.4M** | **384** | 512 | MIT | — |
| bge-base-en-v1.5 | ~110M | 768 | 512 | MIT | — |
| **gte-modernbert-base** | **149M** | **768** | **8192** | **Apache-2.0** | **64.38** (v1, 56-task) |
| nomic-embed-text-v1.5 | 137M | 768 (MRL) | 8192 | Apache-2.0 | 62.28 |
| EmbeddingGemma-300m | 300M | 768 (MRL) | 2048 | Gemma | 69.67 (Eng v2) |
| bge-large-en-v1.5 | 335M | 1024 | 512 | MIT | 64.23 (v1) |
| Qwen3-Embedding-0.6B | 596M | 1024 (MRL 32–1024) | 32K | Apache-2.0 | 70.70 (Eng v2) |
| Qwen3-Embedding-8B | 8B | 4096 | 32K | Apache-2.0 | 75.22 (Eng v2) |

**Start with `bge-small-en-v1.5` (384d), move to `gte-modernbert-base` if quality demands.**
Rationale: at POC scale the difference is unlikely to be the bottleneck, and `gte-modernbert-base`
is the natural upgrade — Apache-2.0, 8192 context (matters when D8's full text lands), 768d, and
ModernBERT-based.

⚠️ **Do not mix MTEB numbers across versions.** BGE/GTE cards report **MTEB v1 English 56-task**;
Qwen3 and EmbeddingGemma report **MTEB Eng v2**. They are not comparable. Several widely-quoted
figures (e5-mistral 66.63, mE5-large-instruct 64.41, bge-en-icl 71.67) appear on **neither** the
model cards nor the paper abstracts.

**Excluded on licence** — the same trap as the rerankers: `jina-embeddings-v3` and the entire
**v5 family (v5-text-small/nano, v5-omni-small/nano) are CC-BY-NC-4.0**, unusable under D3's
deployed multi-user future, despite being current SOTA (v5-text-small: MTEB Eng v2 71.7 at 677M).
`jina-embeddings-v4` is Qwen Research License. `bge-multilingual-gemma2` is under Gemma terms.

**Excluded on cost:** 7–8B embedders (e5-mistral, Qwen3-Embedding-8B, bge-en-icl) are ~100–200×
MiniLM's CPU cost — days to index 100k abstracts on a laptop. Not viable under D12's offline rule.

**Excluded on regression:** `nomic-embed-text-v2-moe` caps at **512 tokens**, down from v1.5's
8192.

### CPU feasibility and storage

⚠️ Published CPU throughput at ~250-token abstract length essentially does not exist; the figures
available derive from an unmethodologied sentence-transformers table. **The ratios are the usable
part, not the absolute rates:** 22–33M models are roughly 1 hour per 100k abstracts, 110–150M
around 3–6 hours, 300M around 7–14 hours, and 7B-class is 10+ days. Measure before committing.

Acceleration (sentence-transformers official benchmarks, averaged over 4 models): ONNX fp32
**1.39×** on CPU, OpenVINO fp32 1.29×, **ONNX int8 3.08×** — but the docs warn these are
*short-text* best cases and "may perform worse than PyTorch on longer texts." Note `avx512_vnni`
is x86-only, so **the 3.08× int8 figure does not apply on Apple Silicon**.

**Storage is a non-issue at our scale.** 100k vectors: 1024d fp32 = 410 MB, 768d fp32 = 307 MB,
384d fp32 = 154 MB. **Do not quantize vectors at POC scale** — it buys nothing and adds risk.

For later reference, when the index grows: int8 retains ~97–99% NDCG@10 with rescoring and is
~3.7× faster; binary is ~24× faster but **model-dependent** — mxbai retains 96.5% while e5-base-v2
collapses to **74.8%**. Never assume binary quantization is safe for a model not trained for it.

---

## Citation-graph algorithms — the core of the product

This is where the strongest evidence landed, and it changes several defaults.

### Finding 1 — co-citation is the right tool, and the asymmetry is structural

Co-citation (Small 1973) surfaces the **intellectual base**; bibliographic coupling (Kessler 1963)
surfaces the **research front**. The asymmetry is structural, not statistical: a coupling edge
exists at publication time and never changes, while a co-citation edge cannot exist until a third
paper cites both. Klavans & Boyack (2017) put it directly — *"bibliographic coupling obliterates
history when it is based on a short time window."*

This validates Stage 2 of the pipeline. **Important nuance:** in a bibliographic-coupling graph
the foundational papers are not nodes at all — they are the *shared references* that induce the
edges. So BC defines the topic set, and the ancestors fall out of the reference side. Both are
worth computing; they answer different questions.

⚠️ **Do not import the clustering-benchmark verdicts.** Several studies (Waltman et al. 2020,
Ahlgren et al. 2020) rank co-citation *worst* — but they evaluate **taxonomy/clustering quality**,
not "did you recover the seminal ancestors of topic X." Co-citation's poor showing there does not
transfer to our task.

### Finding 2 — use **age-rescaled PageRank**, not PageRank, and never HITS

The cleanest evidence is Mariani, Medo & Zhang (2016), evaluated against **87 PRL Milestone
Letters** over **449,935 APS papers**, and Xu et al. (2020), which benchmarked **17 metrics across
3 corpora** (APS 595k papers; INSPIRE-HEP 830k; US patents) against expert-curated milestone lists:

- **Raw PageRank "completely fails to identify recent milestone papers"** due to temporal bias.
- **CiteRank** (exponential age discount) fixes recency but "markedly underperforms in identifying
  old milestone papers" — it over-corrects.
- **Age-rescaled PageRank** (rank within an age cohort) is best at **every** paper age, and
  "indicators based on simple citation count are outperformed by rescaled PageRank for papers of
  every age." On the age-bias-corrected metric, rescaled PageRank and rescaled LeaderRank score
  **0.98**, far ahead of the field.
- **HITS is catastrophically bad on citation networks** — authority-score identification rates of
  **0.143 / 0.116 / 0.054** across the three corpora, so poor it was omitted from the figures. The
  stated reason is sound: HITS assumes a citation from a well-referenced but little-cited review
  is more indicative of impact than one from a high-impact paper with few references, which is
  backwards for science.

> **Decision: age-rescaled PageRank on the candidate subgraph is the L0 ranking signal.** Not
> citation count, not raw PageRank, not HITS.

Damping: Chen et al. (2007) use **d = 0.5**, not the web-standard 0.15, justified empirically —
~42–51% of references in a paper's bibliography cite each other, so reference-following paths are
short. At d = 0.9, PageRank degenerates toward raw citation count. Their headline case is
**Slater 1929**: only **114 citations**, citation rank **1853rd**, yet PageRank within a factor
2.2 of the top paper — the "obliteration effect", where a contribution is so foundational nobody
cites the original any more. **That is exactly the paper PaperThread exists to surface.**

PageRank vs citation count correlate at Pearson **0.82** but Spearman drops to **0.59** among
papers with >10 citations — the extra information lives precisely in the well-cited regime.

### Finding 3 — do NOT use Main Path Analysis

It is the obvious-looking tool for "ordered path through a citation graph," and the evidence is
against it:

- **Formally biased toward long paths.** Šubelj, Waltman, Traag & van Eck (2020, *R. Soc. Open
  Sci.*) show this conceptually and empirically, and conclude **intermediacy** "offers a more
  principled approach."
- **Greedy traversal walks into dead branches** — demonstrated in Batagelj's own SOM example,
  where the main path follows a branch that dies out while the live line of development is missed.
- **A single thin chain misses parallel developments.** In a patent study (PLoS ONE 2017), the
  standard multiple-main-path method produced **1821 nodes / 1729 links** while containing only
  **44 of 58** dominant nodes; a better method got all 58 in **159 nodes** — simultaneously ~10×
  bulkier *and* less complete.
- **Cycles break it**, and our corpus is the pathological case: Batagelj explicitly names arXiv
  preprint/published duplication as producing large strongly-connected components. This is
  `PROVIDER_NOTES.md` **C4** arriving from the algorithmic side.
- **There is no production-grade Python implementation.** The maintained ones are Pajek (Windows
  GUI) and the R package `mpaR`. Python has only low-star unmaintained repos.

### Finding 4 — citation intent is available for free, and it matters enormously

The central problem: **"A cites B" does not mean "B is a prerequisite for A."** Valenzuela, Ha &
Etzioni (2015) — the paper behind Semantic Scholar's influential-citation feature — found that
**only 14.6% of citations are "important"** rather than incidental.

**Semantic Scholar's Graph API exposes per-edge `intents`, `contexts`, and `contextsWithIntent`
fields, plus `isInfluential` on citation/reference edges.** That means citation-intent labels
require **no model hosting at all** — a substantial saving, and it directly attacks the
perfunctory-citation problem in Stage 2.

If a local classifier is ever needed, SciBERT fine-tuned on `allenai/scicite` is the well-trodden
~85 macro-F1 baseline (SciCite: SciBERT 85.22; best published 89.46. ACL-ARC is much harder: best
75.57).

⚠️ **Accuracy correction:** the Valenzuela paper reports **AUC 0.80**, precision ≈0.65 at recall
0.9, on **465 annotated examples**. A figure of "0.91 AUC" circulates in search snippets and is
wrong. S2 has never published accuracy for the *deployed* model, and the FAQ notes influential
citations are missed where full text is unavailable — a coverage gap that correlates with the
open-access gaps in `PROVIDER_NOTES.md` §1.

### Finding 5 — no benchmark exists for our actual task

There is **no established benchmark for "given topic X, recover its foundational papers."** What
exists is corpus-wide milestone identification (APS PRL Milestone Letters, the INSPIRE-HEP
particle-physics chronology, F1000Prime expert tags) — not topic-conditional. And Xu et al. (2020)
showed even those are confounded: on HEP, 74% of seminal nodes fall in the oldest age group, so
an age-biased metric gets *rewarded* for sharing the ground truth's bias.

Practical consequence: **build evaluation in from the start, using survey-paper reference sections
and curated reading lists as proxy ground truth**, and don't expect a benchmark number to validate
the approach. This makes Stage 4's stored provenance (D2/D10) more important, not less.

### Also worth knowing

- **Sleeping beauties are common, not exceptional.** Ke et al. (2015, PNAS, 22M papers) found "a
  continuous spectrum of delayed recognition" and concluded there is "empirical evidence against
  the use of short-term citation metrics." Another argument against recency-weighted ranking.
- **The disruption / CD index is contested** — Petersen et al. (2024) show the reported decline is
  "an artifact" of citation inflation, and convergent-validity studies found the *original* DI1 is
  not the best variant (DI5 is). Use as a feature, never as a headline ranking.
- **RPYS** (Reference Publication Year Spectroscopy) is purpose-built for finding historical roots
  and is worth revisiting later.

---

## Consolidated layer assignment

| Layer | Technique | Status |
|---|---|---|
| **L0** | BM25 over title+abstract (fielded), spaCy tokenize/lemmatize, Schwartz–Hearst acronym expansion at index+query time | Implementation choice pending |
| **L0** | Co-citation / reference co-occurrence expansion; **age-rescaled PageRank** on the candidate subgraph; Louvain/Leiden communities for subtopics | **Decided** |
| **L1** | Deterministic clustering + c-TF-IDF terms; citation-intent filtering via S2 `intents` | **Decided** |
| **L2** | `bge-small-en-v1.5` → `gte-modernbert-base`, in-process via sentence-transformers/ONNX, **never through Ollama** | **Decided** |
| **L2** | RRF fusion of BM25 + dense | **Decided** (+5.1 pp R@5) |
| **L3** | `cross-encoder/ettin-reranker-32m-v1` | **Decided** (+12.1 pp R@5 over fusion) |
| **L4** | LLM role assignment, prerequisite judgment, explanations, subtopic naming — constrained to shortlist IDs | **Decided** |

## Still open

- **L0 lexical implementation** — SQLite FTS5 vs `bm25s` vs Tantivy vs Postgres FTS. Constraint
  from D5: must work on SQLite now and Postgres later. Research agent hit session limits.
- **RM3 / pseudo-relevance feedback** — whether offline query expansion helps on short topical
  queries. Unresearched.
- **SPECTER2 / SciNCL vs general-purpose embeddings** for scientific retrieval specifically —
  partially answered by the citation-partition finding above, but the head-to-head is unresearched.
- **Vector store** (Q13) — deferred until embeddings are actually needed.

## Sources

Ettin: [blog](https://huggingface.co/blog/ettin-reranker) ·
[32m card](https://huggingface.co/cross-encoder/ettin-reranker-32m-v1) ·
[gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) ·
[Qwen3 Embedding/Reranker](https://qwenlm.github.io/blog/qwen3-embedding/) ·
[mxbai-rerank-v2](https://www.mixedbread.com/blog/mxbai-rerank-v2) ·
[mxbai-edge-colbert (2510.14880)](https://arxiv.org/html/2510.14880v1) ·
[jina-reranker-v3 (2509.25085)](https://arxiv.org/abs/2509.25085) ·
[four-stage ablation (2604.01733)](https://arxiv.org/html/2604.01733v1) ·
[LLM rerankers (2508.16757)](https://arxiv.org/html/2508.16757v1) ·
[Setwise / llm-rankers](https://github.com/ielab/llm-rankers) ·
[answerai-colbert-small](https://www.answer.ai/posts/2024-08-13-small-but-mighty-colbert.html) ·
[PLAID (2205.09707)](https://arxiv.org/abs/2205.09707) ·
[scispaCy](https://github.com/allenai/scispacy) ·
[PySBD](https://www.researchgate.net/publication/347234862_PySBD_Pragmatic_Sentence_Boundary_Disambiguation) ·
[wtpsplit](https://github.com/segment-any-text/wtpsplit) ·
[BERTopic (2203.05794)](https://arxiv.org/pdf/2203.05794) ·
[LLM topic labeling (2502.18469)](https://arxiv.org/abs/2502.18469) ·
[scientific taxonomy generation (2509.19125)](https://arxiv.org/pdf/2509.19125) ·
[Topic Is Not Agenda (2605.07158)](https://arxiv.org/html/2605.07158v1) ·
[BLAR / Ab3P](https://aclanthology.org/2021.bionlp-1.14.pdf) ·
[reranker benchmark](https://aimultiple.com/rerankers)
