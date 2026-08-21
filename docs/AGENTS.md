# `docs/` — Agent Guide

Project documentation. No code, no build step.

## Files

| File | Purpose |
|------|---------|
| `REQUIREMENTS.md` | **Source of truth** for product scope and behavior. Read before any design or implementation work. |
| `PROVIDER_NOTES.md` | How external providers differ (LLM and paper sources) and what those asymmetries force in the design. Read before writing or changing any provider adapter. |
| `RETRIEVAL_NOTES.md` | Evidence behind the layered retrieval design (§11 D12) — what technique sits at each layer and why. Read before changing retrieval or ranking. |
| `READING_NOTES.md` | Challenges behind the reading workspace (§7, §8, §11 D13) — full-text acquisition, PDF structure, grounding, durable note anchors. Analysis only; nothing here is built. Read before starting that phase. |

## Conventions

- `REQUIREMENTS.md` is **owner-controlled**. Ask the user before adding, removing, or reinterpreting
  a requirement; record every accepted change in its changelog (§12) with the date.
- Open questions live in §11 as a table. When one is answered, fold the answer into the relevant
  section *and* mark the row resolved in the same change — don't leave the answer only in the table.
- Requirement sections 1–9 are the owner's original wording. Preserve intent when reformatting.
- Design docs, ADRs, and API docs added later belong here too; list them in the table above.

See the root [`AGENTS.md`](../AGENTS.md) for the repository map and the progressive-disclosure
convention.
