# `retrieval/` — Agent Guide

The recommendation pipeline. This is the product's core; **read `docs/RETRIEVAL_NOTES.md` before
changing anything here** — the technique at each layer was chosen from benchmark evidence, and
the notes record what was rejected and why.

| File | Read it for |
|---|---|
| `path.py` | `LearningPathService` — orchestrates stages 1–5. **Entry point.** |
| `search.py` | Stage 1: multi-provider fan-out, dedup, RRF |
| `fusion.py` | Reciprocal Rank Fusion across providers/layers |
| `expansion.py` | Stage 2: citation-graph expansion and co-citation |
| `graph.py` | Stage 3: age-rescaled PageRank, Louvain communities |
| `selection.py` | Which papers make the path, and which pairs are worth an LLM call |
| `judgment.py` | Stage 4: roles, prerequisite edges, §5 explanations — structural **and** LLM |
| `ordering.py` | Stage 5: DAG ordering under the citation constraint |
| `curriculum.py` | **LLM-planned learning paths.** The strategies, and the tunable part. |
| `resolver.py` | Grounding a paper an LLM *named* in a paper that exists |

## The pipeline, and what exists

| Stage | What it does | Status |
|---|---|---|
| 0 | Decompose topic into subtopics *before* searching | **not built** |
| 1 | Multi-provider candidate retrieval, dedup, RRF | **built** — `search.py` |
| 2 | Citation-graph expansion: backward for ancestors, forward for later work | **built** — `expansion.py` |
| 3 | Age-rescaled PageRank + co-citation + communities | **built** — `graph.py` |
| 4 | Roles, prerequisite edges, explanations | **built** — `judgment.py` (structural always; LLM when L4 is on) |
| 5 | DAG ordering under the citation constraint | **built** — `ordering.py` |
| 6 | Personalization against reading history | **not built** — no persistence layer yet |

Subtopics are induced from the citation graph of the *results* (stage 3 communities), not
proposed up front. That is not stage 0, and `path.py` says so in its notes rather than implying
otherwise.

## The idea Stage 2 exists to implement

**The foundational papers of a topic are almost never the best keyword matches for it.** Search
"diffusion models" and you get recent papers that use the phrase — you do not get the 2015 work
underneath, because the terminology was invented after it.

Ancestors are found by looking at what the candidates **cite in common**. If 20 of 25 candidates
cite the same paper, it is foundational to the topic, and no model had to have an opinion about
it. That is the highest-value signal in the system and it runs offline.

It works: a path for "transformers" reaches LSTM, ResNet and backpropagation; one for
"regularization" reaches Lasso (1996) and Elastic Net (2005). None of those are lexical matches.

## Decisions already made from evidence — do not relitigate casually

- **Age-rescaled PageRank on the candidate subgraph**, not citation count, not raw PageRank, and
  **never HITS** (HITS scores 0.14 identification rate on citation networks — it is measurably
  bad here). Raw PageRank "completely fails to identify recent milestone papers"; age-rescaling
  beats citation count at every paper age.
- **Damping `d = 0.5`** (Chen et al. 2007), not the web-standard 0.85. At 0.9 PageRank degenerates
  toward raw citation count.
- **Do not use Main Path Analysis.** Formally biased toward long paths, walks into dead branches,
  misses dominant nodes, breaks on cycles — and arXiv preprint duplication is the canonical
  cycle-producing case. Use intermediacy if a path primitive is needed.
- **Centrality is computed within the candidate subgraph**, not globally. §3 requires educational
  value, not popularity.
- **Stage 5's hard constraint: if A cites B, B precedes A.** Free, unfalsifiable, and it kills a
  whole class of LLM ordering error.
- **Stage 4 is constrained to shortlist IDs**, or the LLM will invent plausible papers that do not
  exist. `judgment.py` re-checks every returned id.
- **Do not add an LLM reranker.** Stage 4 already is one, and LLM rerankers lose 12–15 points on
  queries about content postdating training — precisely this product's operating regime.
- **Community detection must stay deterministic.** The reference Louvain visits nodes in random
  order, which would give the same topic a different curriculum per run — fatal for §6's
  incremental updates. `graph.louvain` sorts and breaks ties by lowest community id.

## Traps this code already fell into

Each of these has a regression test. They look like over-specification until you change the
surrounding logic.

- **Expansion must deduplicate candidates and discovered papers as ONE population.** References
  arrive as fresh provider records; a preprint from pass 1 and its published version from pass 3
  are the same work. Unmerged, an ancestor's co-citation count splits and it drops below the
  threshold that would have surfaced it.
- **Self-edges appear only after merging** — a seed's reference list containing its own preprint.
  They are one-node cycles and must be dropped.
- **`_has_topic_evidence` needs a threshold of 2, not 1.** A lexical false positive drags its
  whole ancestry in behind it, and age-rescaling flatters anything with no age peers. At a
  threshold of 1, "diffusion models" returned three 1960s papers on osmotic flow through cellulose
  acetate membranes.
- **Structural explanations must not contradict themselves.** "It matched the topic directly" and
  "keyword search did not return it" were once emitted in the same paragraph.
- **`search.py` takes `standalone`.** Its "L0 only" caveat is a lie when it runs as stage 1 of the
  path pipeline, because expansion and ordering run immediately afterwards.

## Layer discipline (§11 D12)

Everything except `judgment.py`'s LLM half runs at **L0**: no model weights, no LLM, network only
for the providers themselves.

- **Every layer must be independently disableable**, and disabling one must degrade quality
  without breaking the system. `LayerConfig` drives this.
- **Degradation must be visible.** `LearningPath.degraded`, `.notes` and `.stages_run` exist so a
  caller can tell "a provider failed" from "nothing matched" from "L4 is off". Never swallow a
  failure.
- **L0 must be genuinely useful alone**, not a placeholder. Stage 4 therefore has two full
  implementations — structural and reasoned — not one implementation with an error path.
- **Degradation is per-batch inside stage 4.** One failed explanation batch leaves those papers
  with structural explanations and the rest reasoned.

## `fusion.py` notes

RRF combines ranked lists without needing scores to be comparable — they are not, and no provider
documents its scale. `k=60` is the constant from the original paper.

**Deduplication happens before scoring.** The same work from three providers must score as one
paper found three times, not three papers found once — otherwise preprint/published duplicates
split their own score. Within-provider duplicates keep the best rank so they cannot inflate it.
Ordering is deterministic (score → provider agreement → title), which §6's incremental path
updates depend on.

## Two families of path builder, and why both exist

`path.py` (structural) and `curriculum.py` (LLM-planned) answer different questions.

**Centrality is not pedagogy, and this is measured, not argued.** A learner scored the two
approaches over ten topics against `tools/CRITIQUE_RUBRIC.md`: the structural pipeline
averaged **1.5/25**, an LLM-planned syllabus **15.2/25**. For "dropout", centrality surfaces
ImageNet and ResNet — central to the surrounding literature, useless to someone learning
dropout. The LLM plans backprop -> weight decay -> bagging -> Dropout, which is a path.

Keep the structural pipeline anyway: D12 requires a usable result with no LLM, and its
citation-graph machinery is what grounds the LLM strategies rather than competing with them.

### Strategies (`curriculum.py`)

| Strategy | Idea | Fails when |
|---|---|---|
| `syllabus` | LLM plans the sequence; each step is resolved to a real paper | titles resolve poorly |
| `anchor` | LLM names the target; its REAL references supply prerequisites | no single definitive paper exists |
| `rerank` | Existing retrieval, LLM imposes teaching order | the right paper was never retrieved |
| `hybrid` | `syllabus`, falling through to `anchor` when too little resolves | — |

`anchor` scored worse overall (9.0) but beat `syllabus` on *sourcing* wherever its anchor
resolved — a real bibliography surfaces precursors nobody recalls from memory. The
medium-term design is `syllabus`'s plan with `anchor`'s reference-list grounding.

## Traps specific to LLM-planned paths

- **Never let a substituted paper inherit the planner's rationale.** The resolver returned
  TimeSformer for "Attention Is All You Need", and the path shipped it carrying the
  sentence "this is the paper that defines the Transformer". `Resolution.is_confident`
  gates this; below the bar the rationale is discarded, not reattached.
- **A path with no anchor is not a short path.** It is a path about the neighbourhood of a
  topic that never arrives. It must say so first, not in a footnote.
- **Paths must carry confidence.** A 22/25 path and a 9/25 path rendered identically, which
  a learner called the shipping blocker — the good ones were indistinguishable from the
  ones confidently describing the wrong paper. `_score_confidence` is built only from
  checkable facts (did we reach the topic, did the plan survive lookup), never from
  anything the model says about itself.

## Known upstream defect

OpenAlex sometimes conflates two distinct works into one record: correct citation edges, wrong
title and DOI. It surfaces as a nonsense title at the *top* of a path, because the structural
evidence is genuinely strong. Single-provider structure cannot detect it — see
`docs/PROVIDER_NOTES.md` **C11**. Do not "fix" it with a title-versus-topic heuristic; that would
suppress exactly the foundational papers whose vocabulary predates the topic.
