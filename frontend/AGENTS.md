# `frontend/` — Agent Guide

React + Vite + TypeScript (§11 D5). Single page: enter a topic, get an ordered learning path.

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173, proxies /api -> 127.0.0.1:8000
npm run build    # tsc -b && vite build
```

The backend must be running separately (see `backend/AGENTS.md`).

| File | Contents |
|---|---|
| `src/api.ts` | Wire types mirroring `backend/paperthread/api/main.py`, `fetchPath`, and the derivations the UI reads the response through (`actOf`, `edgesJudged`, `hasMeasuredSignals`) |
| `src/App.tsx` | The whole UI: hero, search, confidence, the acted path, and the build log |
| `src/styles.css` | Design tokens and all styling |
| `index.html` | Page metadata for a public deployment |
| `vite.config.ts` | Dev server and the `/api` proxy |

## Two design rules, both content-true

**1. The thread reports what the system actually did.**
The vertical rule down the path is the page's signature element. It is **solid** when a model
has judged every prerequisite edge to be a real teaching dependency, and **dashed**
(`.thread-inferred`) while the edges are merely inferred from shared citations. Order and
edge-quality are two different claims and the UI distinguishes them.

**Read that state from the edges, not from the stage list.** `edgesJudged()` checks that every
`PathEdge.source` is `llm_judgment`. Keying off `stages_run.includes("judge:llm")` is wrong:
the structural pipeline records that stage but the planned strategies record only
`strategy:<name>`, so a fully model-judged syllabus path reported itself as unjudged.

The same rule governs `.explain-structural`: a dashed border and a "measured, not reasoned"
badge. **Never render a structural explanation identically to an LLM one.**

**2. Era is colour.** Ochre marks pre-2015 papers (`ANCESTOR_CUTOFF`), blue marks recent ones.
The product's core claim is that ancestors and entry points are different things, so the palette
carries that distinction instead of decorating. Don't repurpose these two colours for anything
else — in particular the **acts are distinguished by weight and label, never by hue**, precisely
so they cannot be mistaken for era. Each node resolves `--era` once; the marker, the card rail
and the year all read it, so they cannot drift apart and say different things.

No external fonts or assets are loaded — the UI stays fully offline, matching the L0 retrieval
layer. Personality comes from the type scale and from pairing the bibliographic serif against
the UI sans, not from a downloaded face.

## Structure: acts, not levels

A planned path (`retrieval/curriculum.py`) arrives as **one paper per level**, so `level` carries
no grouping and a heading per level means a heading per paper. The real structure is the act,
which the backend puts in `subtopic_id` as one of `prerequisite` / `anchor` / `followup` —
mirrored by `ACTS` in `src/api.ts`, and read by `actOf()`.

- A path **with** acts renders three sections: *Before the paper*, *The paper*, *After the paper*.
- A path **without** acts (structural, where levels genuinely hold several papers each) falls back
  to level grouping.

`groupIntoActs()` picks between the two. Keep `ACTS` keys in sync with `STAGE_*` in
`curriculum.py`; a mismatch silently drops the grouping and the UI falls back to levels.

The **anchor** is the one place this page raises its voice: a wider rail, a deeper shadow, a
larger title and a bullseye marker. Spend boldness there and nowhere else.

## Conventions

- **Show degradation, don't hide it.** The build log renders every layer including the ones that
  are OFF, every stage including the ones that were never built, and all backend `notes`. A
  failed provider returning nothing must never look like nothing matching. Do not "clean up" this
  section. It sits *below* the path rather than above it — the reader came for the reading list —
  but nothing in it is hidden or collapsed, and `Gaps` states the unbuilt capabilities at the top
  in the reader's language with a link straight down to it.
- **The §5 explanation is data, not garnish.** All four fields (why it matters / what it assumes /
  what it teaches / why for you) render for every paper, along with `source` and `asserted_by`.
  Don't collapse them into a summary — they are separately meaningful and separately scorable.
- **Notes are classified per note, not per response** (`isWarning`). `degraded` is true whenever
  the pipeline ran below full capability, which with L4 off is always; colouring every note as an
  alert on that basis makes the notes that ARE failures invisible.
- **Confidence is rendered before the path, not after it.** `Confidence` states how much the
  system trusts its own answer and why. This exists because a learner scored two paths 22/25 and
  9/25 and could not tell them apart: both rendered with the same confident stage labels and
  fluent rationales, and the 9/25 one called a video-understanding paper "the paper that defines
  the Transformer". When a system is sometimes wrong, the output must carry the difference.
- **An absent measurement is not a zero.** The planned strategies leave `signals` at 0 because
  they never ran the citation graph, so `hasMeasuredSignals()` gates the signal chips. Rendering
  "age-rescaled +0.00" on every card states a measurement that was never taken. For the same
  reason `discovered_by_expansion` only renders when the `expand` stage actually ran.
- **Don't restate what the order already says.** A prerequisite that is simply the previous step
  is not rendered; only a dependency reaching further back is. Likewise `role` is suppressed on
  acted paths, where it is a pure function of the act.
- **Keep `src/api.ts` in sync with the backend's Pydantic models.** There is no codegen; a
  mismatch fails at runtime, not compile time.
- Requests are abortable and superseded searches are cancelled — keep that when adding calls.
- Quality floor: keyboard focus is visible, `prefers-reduced-motion` respected, responsive to
  mobile, light and dark both first-class. Don't regress these.

## Not built yet

Mark-as-read, user profile, and "show more" within a subtopic. All wait on backend persistence
and pipeline stage 6 — see `backend/paperthread/retrieval/AGENTS.md`. The landing page says so
in its own words rather than leaving the reader to infer it.

Subtopic labels read "Line of work N" until L4 names them; that placeholder is deliberate and
`SubtopicLegend` says why in the UI. Note that the planned strategies return no `subtopics` at
all, so that legend only appears on structural paths.
