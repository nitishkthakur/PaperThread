# `domain/` — Agent Guide

Core types and identity logic. **No I/O, no HTTP, no provider-specific anything.** If you are
about to import `httpx` here, you are in the wrong folder.

| File | Read it for |
|---|---|
| `models.py` | `Paper`, `ExternalId`, `ContentDepth`, `FieldProvenance`, `SearchHit`, `RankedPaper` |
| `identity.py` | Canonical IDs, ID normalization, and cross-provider deduplication |
| `path.py` | `LearningPath`, `PathStep`, `PrerequisiteEdge`, `Explanation`, `PaperRole` |

## The two ideas that shape everything here

**1. PaperThread owns canonical paper identity; external IDs are aliases.**
Not DOI-keyed — many arXiv preprints have no DOI. `canonical_id_for()` prefers the strongest
available identifier and falls back to a title+year hash so a paper with no usable ID still gets
stable identity instead of being dropped.

**2. Content depth is explicit and ordered** (`METADATA` < `ABSTRACT` < `FULLTEXT`).
v1 stores metadata + abstracts, but full text is coming (§11 D8). Consumers ask for the depth
they need; depth can increase later without a schema rewrite. Do not write code that assumes
abstract-only.

## `identity.py` is correctness-critical — read before editing

The same work exists as an arXiv preprint *and* a published paper, with different IDs, different
years, and **separate citation counts**. Unreconciled, that produces duplicate nodes in a learning
path, corrupted centrality scoring, and false prerequisite edges between a paper and its own
preprint. Batagelj names arXiv preprint duplication as the pathological case for citation-graph
algorithms specifically.

`deduplicate()` is union-find over shared merge keys, so A~B and B~C collapse A, B and C together
— the common case for an arXiv/OpenAlex/S2 triple that shares no single identifier across all
three.

Three subtleties that already have regression tests, so don't "simplify" them away:

- **The title merge key spans a ±1 year window** (`_YEAR_WINDOW`). Exact-year matching misses
  preprint/published pairs a year apart, which is the case the key exists for. Widening it further
  starts merging genuinely distinct same-titled surveys.
- **Undated papers are handled separately** (`_absorb_undated`), not given a year-agnostic title
  key — that would transitively merge every same-titled paper across all eras. They join a dated
  group only when exactly one candidate exists.
- **`merge_from` keeps the maximum citation count**, because counts are split across versions and
  by provider. Taking the first or last would under-rank foundational papers.

**Papers with no abstract are retained, never filtered.** Abstract coverage is worst for older
papers — exactly the foundational ones the product exists to surface.

Run `../../.venv/bin/python -m pytest tests/test_identity.py -q` after any change here.
