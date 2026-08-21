# Learner critique — iteration 1

Mean scores (out of 25): **syllabus 15.2, anchor 9.0, structural 1.5**
(structural had 8/10 empty from a provider outage, so its number is not a fair baseline.)

## Verdict
Ship `syllabus`. But `anchor` is not the loser it looks like: where its anchor resolved, the
first half of its paths beat syllabus on sourcing (Raiko/Wiesler for batchnorm, "Nightmare at
test time" for dropout) because a real reference list surfaces precursors no one recalls from
memory. Its score is destroyed by two fixable things that are NOT the reference-list idea.

## Failures, ranked by damage to a learner

1. **Silent wrong-paper substitution with the planner's prose left attached.** `syllabus/transformers`
   step 4 is "Is Space-Time Attention All You Need for Video Understanding?" (TimeSformer) carrying
   the text "This is the paper that defines the Transformer." Same in `syllabus/GANs`: Mirza's
   Conditional GAN carrying Goodfellow's abstract. Worse than an empty path — it tells the learner
   authoritatively that they have read the founding paper when they have not.
2. **Whole paths built toward a wrong anchor.** Both strategies chose TimeSformer for "transformers"
   and Conditional GAN for "GANs" — the SAME two wrong papers, so this is a shared deterministic
   component (the resolver), not two independent hallucinations. In `anchor/transformers` the error
   cascades: 7 of 9 steps become video understanding.
3. **Zero/one-step paths from title-resolution failure, not missing papers.** SimCLR, Christiano,
   InstructGPT, DPO all failed to resolve. Evidence it is the resolver: `anchor/transformers`
   resolved Vaswani fine when it arrived as a reference-list ID, while `syllabus/transformers`
   failed on the same paper as a title string. Everything that resolves reliably is an older
   journal-indexed classic; everything that fails is arXiv-native post-2017 ML — i.e. the system is
   weakest exactly where users search most.
4. **The `followup` stage is a citation-count sort wearing a pedagogy label** (anchor, 8/10 topics).
   Byte-identical boilerplate under every entry. BatchNorm does not "build on" dropout.
5. **Wrong-field intrusions.** Polynomial-chaos physics in a dropout path; a Japanese-language
   seminar record as the entry point for GANs; a "(Gdl Seminar)" recording in word embeddings.
6. **Famous-but-useless intrusions.** ImageNet benchmark paper as step 1 for batch normalization;
   AlexNet labelled `prerequisite` for GANs; a 350-page textbook as one "step".
7. **Sequences where step N does not depend on step N-1** — `anchor/word embeddings` steps 2-9
   entirely; `syllabus/CNNs` backprop(1986) -> Neocognitron(1980).
8. **Stage labels that are lies.** `anchor/transformers` labels *Attention Is All You Need* as
   `prerequisite` and TimeSformer as `anchor` — the taxonomy is exactly inverted. `anchor/regularization`
   has TWO steps labelled `anchor`.
9. **Metadata corruption.** AlexNet dated 2017, Vaswani dated 2025, BatchNorm dated 2024 and shipped
   TWICE in one path (dedup missed it because the year differed). Literal `\n` inside titles.
   Mangled authors ("BengioYoshua, DucharmeRejean").
10. **Paths that stop at the anchor** — `anchor/CNNs` and `anchor/regularization` have zero follow-up.

## Highest-value fix
Make the resolver verify what it returns, and **never let a substituted paper inherit the planner's
rationale**. Prefer reference-list IDs over title strings wherever available. Fail loudly when the
anchor is lost rather than shipping a short path with the bad news in a collapsed block.

## Shipping blocker
Nothing in the output distinguishes the 22/25 paths from the 9/25 ones. They render identically,
with the same confident stage labels and fluent rationales. A learner cannot tell them apart.
