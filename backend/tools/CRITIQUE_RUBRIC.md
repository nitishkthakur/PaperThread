# Learner critique rubric

The scoring instrument for the ranking-algorithm tuning loop. It exists because the thing
being optimised — *does this feel like a learning path?* — has no automatic metric, and the
metric it is most easily confused with (are these the important papers?) is precisely the
one that produced the problem.

## Who you are when you score

**You are not a reviewer of research quality. You are someone trying to learn the topic.**

You know how to read a paper and you have general background in the field, but you do not
know this specific topic. You have a weekend. You are going to read these papers in the
order given, and you will not reorder them or skip around, because you do not yet know
enough to know what is safe to skip.

Score what actually happens to you as you go down the list.

## The five dimensions (0–5 each, 25 total)

### 1. Entry point
Can you start? Step 1 must be readable by someone who does not yet know the topic.

- **5** — Step 1 is genuinely introductory and assumes only general background.
- **3** — Step 1 is readable but drops you closer to the middle than the beginning.
- **0** — Step 1 already assumes the thing you are trying to learn.

### 2. Anchor presence and placement
Is the paper that *is* the topic actually here, and in the right place?

- **5** — The definitive paper is present, and positioned after its prerequisites.
- **3** — Present but misplaced (first, or buried at the very end).
- **0** — Absent. You could read this entire path and never meet the topic itself.

A path about dropout that never reaches Srivastava et al. has failed, no matter how good
the other papers are.

### 3. Incremental build
Does each step depend on the ones before it?

- **5** — Every step is understandable given the previous ones, and each adds one new idea.
- **3** — Broadly sensible, with a jump or two you would have to look something up for.
- **0** — The order is arbitrary; shuffling it would lose nothing.

Test each adjacent pair: *does step N become easier because I read step N-1?* If the answer
is "no, they are just both about the topic", that is not a path.

### 4. Signal, not noise
Is anything here that a learner should not be reading?

- **5** — Every paper earns its place.
- **3** — One or two loosely-related papers that waste an afternoon.
- **0** — Padded with famous-but-irrelevant papers, or papers from a different field that
  share a word with the topic.

The characteristic failure: highly-cited papers from the surrounding literature that are
central to the *field* but teach nothing about the *topic*. ImageNet is not a step on the
way to understanding dropout.

### 5. Coverage of the arc
Does it go somewhere — foundations, then the thing, then what came of it?

- **5** — Prerequisites, anchor, and meaningful follow-up work, in proportion.
- **3** — Two of the three stages are real, one is thin or missing.
- **0** — Flat. A list of same-era papers with no arc.

## Required output per topic

- The five sub-scores and the total out of 25.
- **The one change that would most improve this path.** Be specific: name the paper that
  should not be there, or the concept the sequence skips over.
- **Would you actually follow this?** yes / no / partly. Answer as the learner, not as a
  critic being generous.

## Rules

- **Judge the path, not the papers.** A list of excellent papers in an order that teaches
  nothing scores low. That is the entire point of this exercise.
- **The `stage` labels are claims, not facts.** A step labelled `prerequisite` that is not
  one is worse than an unlabelled step, because it misleads. Check them.
- **Do not reward length.** Six coherent steps beat twelve padded ones.
- **Penalise wrong-field intrusions hard.** A paper about electrical transformers in a path
  about attention, or chemical diffusion in a path about generative models, is not a small
  blemish — it is evidence the system does not understand the query.
- **Be a harsh grader.** A 25/25 means you would hand this to a colleague without
  editing it. Most first attempts should not score above 15.
