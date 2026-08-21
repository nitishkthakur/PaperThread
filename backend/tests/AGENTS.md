# `tests/` — Agent Guide

pytest. **No test touches the network** — run them anywhere, offline, in under a second.

```bash
./.venv/bin/python -m pytest -q            # from backend/
```

| File | Covers |
|---|---|
| `test_identity.py` | ID normalization, canonical identity, deduplication |
| `test_fusion.py` | Reciprocal Rank Fusion, OpenAlex abstract reconstruction |
| `test_graph.py` | PageRank mass conservation, age-rescaling, deterministic Louvain |
| `test_ordering.py` | The citation constraint, cycle breaking, transitive reduction |
| `test_expansion.py` | Stage 2 identity reconciliation, co-citation counting, degradation |
| `test_selection_and_judgment.py` | Selection quotas, pair ranking, structural §5 explanations |
| `test_llm.py` | JSON extraction, schema validation, cache-key stamping |
| `test_resolver.py` | Grounding named papers: accepting the right one, rejecting the wrong one |
| `test_curriculum.py` | Path assembly, anchor-loss reporting, confidence scoring |

## What is worth testing here

Deduplication and fusion, because they are **correctness-critical and easy to break silently**.
A dedup regression does not throw — it quietly produces duplicate nodes in a learning path and
splits citation counts, corrupting the ranking that depends on them.

The tests in `TestYearWindow` and `test_within_provider_duplicate_does_not_double_count` are
regressions for real bugs found during the POC build. Both look like over-specification until you
change the merge logic; then they are the reason you find out immediately.

## Conventions

- **Never add a test that hits a provider API.** Extract parsing into a pure function and test
  that, as `reconstruct_abstract` is tested.
- Use the local `make_paper` / `paper` helpers rather than constructing `Paper` inline — they
  handle `canonical_id` assignment.
- Name tests after the behaviour and its reason, not the function
  (`test_preprint_and_published_merge_on_title_despite_no_shared_id`, not `test_dedup_2`).

## Regressions locked down here

Every one of these was a real bug that **did not raise** — it silently produced a wrong path.
That is the failure mode this suite exists for.

- A preprint and its published version failing to merge, splitting an ancestor's co-citation
  count so it drops below the threshold that would have surfaced it.
- `_has_topic_evidence` at a threshold of 1 promoting 1960s osmotic-membrane papers into a path
  about diffusion models.
- A structural explanation claiming both "matched the topic directly" and "keyword search did
  not return it".
- `bool` passing an `isinstance(x, int)` check, so a JSON `true` became a confidence of 1.0.
- Louvain returning different communities for the same graph depending on input order.
- **The resolver substituting a different paper and keeping the planner's prose.** TimeSformer
  shipped as the anchor of a "transformers" path described as "the paper that defines the
  Transformer". Every content word of Vaswani's title is inside TimeSformer's, so containment
  cannot separate them.
- **Tightening the matcher to stop that, and thereby rejecting Goodfellow's real paper** (indexed
  as "Nets" against a query saying "Networks"). `test_resolver.py` keeps both directions in one
  file precisely because fixing either one alone breaks the other.

## Gaps worth filling

- `TopicSearchService` degradation paths. `test_expansion.py` now has a fake-provider pattern
  (`FakeCitationProvider`) worth reusing for it.
- `LLMClient` end to end against a fake provider — that a cache hit skips the call, and that a
  schema failure retries with the error fed back before giving up.
- `LearningPathService` wiring, with every stage faked. The stage-by-stage degradation matrix is
  the product's honesty contract and nothing tests it as a whole.
- Config loading and `ConfigError` on a provider named in config with no registered adapter, and
  on an unknown key in `[retrieval.expansion]`.
- Adapter parsing against recorded fixture payloads.
