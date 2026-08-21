# Learning paths — strategy `syllabus`, planning model `deepseek-v4-pro:cloud`

Each block is what a user sees after typing the topic. Read it as a learner:
**would following these papers in this order teach you the topic?**

`stage` is the path's own claim about each step — `prerequisite` (read before the main idea), `anchor` (the paper that IS the topic), `followup` (what came after).

---

## dropout in neural networks

*7 steps, 15s — self-rated confidence **LOW** (0.40)*

> never reaches the topic itself (no anchor paper); part of the plan could not be found and was dropped

1. **[prerequisite]** (1986) *Learning representations by back-propagating errors* — David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
   - teaches: Fundamentals of training neural networks using backpropagation
   - why here: Establishes the basic training procedure for neural networks, which all later regularization methods modify.
2. **[prerequisite]** (1996) *Bagging predictors* — Leo Breiman
   - teaches: Ensemble averaging of multiple models to reduce variance
   - why here: Dropout is motivated by model averaging; understanding bagging provides the ensemble perspective needed to see dropout as an implicit ensemble.
3. **[prerequisite]** (2008) *Extracting and composing robust features with denoising autoencoders* — Pascal Vincent, Hugo Larochelle, Yoshua Bengio
   - teaches: Injecting noise into inputs to learn robust features
   - why here: Shows that stochastic corruption during training can improve generalization, a direct precursor to dropout's random omission of hidden units.
4. **[followup]** (2013) *Understanding Dropout: Training Multi-Layer Perceptrons with Auxiliary Independent Stochastic Neurons* — Kyunghyun Cho
   - teaches: Theoretical analysis of dropout as ensemble averaging and adaptive regularization
   - why here: Provides a formal analysis of why dropout works, connecting it to the ensemble and regularization ideas from the prerequisites.
5. **[followup]** (2015) *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning* — Yarin Gal, Zoubin Ghahramani
   - teaches: Dropout as approximate Bayesian inference for uncertainty estimation
   - why here: Extends dropout to quantify model uncertainty, a major application beyond regularization.
6. **[followup]** (2015) *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks* — Yarin Gal, Zoubin Ghahramani
   - teaches: Applying dropout to recurrent neural networks with theoretical grounding
   - why here: Shows how to correctly use dropout in RNNs, addressing a limitation of the original method.
7. **[followup]** (2017) *Concrete Dropout* — Yarin Gal, Jiri Hron, Alex Kendall
   - teaches: Automatic tuning of dropout rates via continuous relaxation
   - why here: Improves dropout by learning per-layer dropout rates, reducing the need for manual tuning.

<details><summary>pipeline notes</summary>

- INCOMPLETE: this path never reaches the topic itself. The paper that would be the destination could not be identified or could not be found, so what follows is background only. Treat it as a partial result.
- 2 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'A Simple Weight Decay Can Improve Generalization'; 'Dropout: A Simple Way to Prevent Neural Networks from Overfitting'
- 7 steps: 3 prerequisite, 0 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## contrastive learning

*8 steps, 15s — self-rated confidence **HIGH** (0.85)*

> part of the plan could not be found and was dropped

1. **[prerequisite]** (1993) *SIGNATURE VERIFICATION USING A “SIAMESE” TIME DELAY NEURAL NETWORK* — JANE BROMLEY, JAMES W. BENTZ, LÉON BOTTOU
   - teaches: Learning a similarity function between pairs of inputs using a shared neural network.
   - why here: Introduces the siamese architecture, the foundational idea of comparing pairs to learn representations, which contrastive learning builds upon.
2. **[prerequisite]** (n.d.) *Dimensionality Reduction by Learning an Invariant Mapping* — R. Hadsell, S. Chopra, Y. LeCun
   - teaches: A loss function that pulls embeddings of similar pairs together and pushes dissimilar pairs apart.
   - why here: Defines the contrastive loss, directly preceding modern contrastive learning by formalizing the push-pull objective.
3. **[prerequisite]** (2013) *Distributed Representations of Words and Phrases and their Compositionality* — Tomas Mikolov, Ilya Sutskever, Kai Chen
   - teaches: Learning word embeddings efficiently using negative sampling, a form of contrastive estimation.
   - why here: Shows how contrastive ideas scale to large representation learning, influencing later self-supervised visual methods.
4. **[anchor]** (2020) *A Simple Framework for Contrastive Learning of Visual Representations* — Ting Chen, Simon Kornblith, Mohammad Norouzi
   - teaches: A simple framework for contrastive learning of visual representations using data augmentation, a projection head, and large batch training.
   - why here: The anchor paper that popularized contrastive learning in computer vision, synthesizing the prerequisite ideas into a highly effective method.
5. **[followup]** (2020) *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere* — Tongzhou Wang, Phillip Isola
   - teaches: Theoretical analysis showing contrastive learning optimizes alignment of positive pairs and uniformity of the representation distribution.
   - why here: Provides a principled understanding of why the anchor method works, immediately following it.
6. **[followup]** (2020) *Bootstrap your own latent: A new approach to self-supervised Learning* — Jean-Bastien Grill, Florian Strub, Florent Altché
   - teaches: A self-supervised method that removes negative pairs by using a momentum encoder and a predictor, achieving strong results.
   - why here: Challenges a core assumption of the anchor (need for negatives), representing a major extension of contrastive learning.
7. **[followup]** (2020) *Supervised Contrastive Learning* — Prannay Khosla, Piotr Teterwak, Chen Wang
   - teaches: Extending contrastive learning to supervised settings by leveraging label information to define positive sets.
   - why here: Shows how the contrastive framework can be adapted beyond self-supervision, building on the anchor.
8. **[followup]** (2020) *What Makes for Good Views for Contrastive Learning?* — Yonglong Tian, Chen Sun, Ben Poole
   - teaches: Analysis of the role of data augmentation in contrastive learning, showing that view selection is crucial for performance.
   - why here: Provides practical insights into a key component of the anchor method, rounding out the sequence with a focus on implementation details.

<details><summary>pipeline notes</summary>

- 1 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Noise-contrastive estimation: A new estimation principle for unnormali'
- 8 steps: 3 prerequisite, 1 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## transformers

*7 steps, 16s — self-rated confidence **HIGH** (1.00)*

> reaches the topic, with prerequisites, all steps verified

1. **[prerequisite]** (2014) *Sequence to Sequence Learning with Neural Networks* — Ilya Sutskever, Oriol Vinyals, Quoc V. Le
   - teaches: Sequence-to-sequence learning with recurrent neural networks, where an encoder compresses the input into a fixed vector and a decoder generates the output.
   - why here: Introduces the encoder-decoder framework for sequence transduction, which the Transformer later replaces the recurrent components of.
2. **[prerequisite]** (2014) *Neural Machine Translation by Jointly Learning to Align and Translate* — Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
   - teaches: Attention mechanism that allows the decoder to focus on relevant parts of the input sequence, avoiding the fixed-vector bottleneck.
   - why here: Introduces attention, the core idea that the Transformer extends to self-attention across the entire sequence.
3. **[prerequisite]** (2017) *Convolutional Sequence to Sequence Learning* — Jonas Gehring, Michael Auli, David Grangier
   - teaches: Non-recurrent sequence modeling using convolutional networks and positional embeddings, showing that recurrence is not necessary for sequence tasks.
   - why here: Demonstrates a fully parallelizable sequence model with positional information, directly motivating the Transformer's design.
4. **[anchor]** (2025) *Attention Is All You Need* — Ashish Vaswani, Noam Shazeer, Niki Parmar
   - teaches: The Transformer architecture: multi-head self-attention, positional encoding, and feed-forward layers, entirely without recurrence or convolution.
   - why here: This is the paper that defines the Transformer, the topic the learner asked about.
5. **[followup]** (2018) *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — Jacob Devlin, Ming-Wei Chang, Kenton Lee
   - teaches: Pre-training a deep bidirectional Transformer on large text corpora and fine-tuning for downstream tasks, establishing the dominant NLP paradigm.
   - why here: Shows how the Transformer encoder can be pre-trained to capture rich contextual representations, leading to state-of-the-art results across NLP.
6. **[followup]** (2020) *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov
   - teaches: Applying a pure Transformer encoder to image patches, demonstrating that the architecture generalizes beyond text to vision.
   - why here: Extends the Transformer to computer vision, proving its versatility and sparking the ViT line of research.
7. **[followup]** (2019) *Transformer-XL: Attentive Language Models beyond a Fixed-Length Context* — Zihang Dai, Zhilin Yang, Yiming Yang
   - teaches: Introducing recurrence into Transformers to handle longer contexts by caching hidden states, overcoming the fixed-length limitation.
   - why here: Addresses a key limitation of the original Transformer (fixed context window) and enables modeling of long-range dependencies.

<details><summary>pipeline notes</summary>

- 7 steps: 3 prerequisite, 1 anchor, 3 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## reinforcement learning from human feedback

*7 steps, 25s — self-rated confidence **MEDIUM** (0.75)*

> 1 step(s) are an approximate match for what was planned; part of the plan could not be found and was dropped

1. **[prerequisite]** (2017) *Policy Gradient Methods for Reinforcement Learning with Function Approximation and Action-Dependent Baselines* — Philip S. Thomas, Emma Brunskill
   - teaches: Closest available match for: Fundamentals of policy optimization in reinforcement learning, including the policy gradient theorem.
   - why here: The plan asked for 'Policy Gradient Methods for Reinforcement Learning with Function Approximation' at this position; this is the nearest paper actually found, so treat the placement as approximate.
2. **[prerequisite]** (2012) *Preference-based reinforcement learning: a formal framework and a policy iteration algorithm* — Johannes Fürnkranz, Eyke Hüllermeier, Weiwei Cheng
   - teaches: Formalizing RL with preferences instead of numerical rewards, and an algorithm for policy iteration.
   - why here: This is the direct precursor to deep RLHF, establishing the framework of learning from pairwise preferences.
3. **[anchor]** (2017) *Deep reinforcement learning from human preferences* — Paul Christiano, Jan Leike, Tom B. Brown
   - teaches: Scaling preference-based RL to deep neural networks by training a reward model from human comparisons and optimizing a policy with PPO.
   - why here: This is the seminal paper that introduced RLHF as we know it, combining deep RL with human preference learning.
4. **[followup]** (2019) *Fine-Tuning Language Models from Human Preferences* — Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu
   - teaches: Applying RLHF to language models for text continuation tasks, using a reward model trained on human preferences.
   - why here: Extends the anchor's method to language generation, showing its applicability beyond simulated environments.
5. **[followup]** (2022) *Training Language Models to Follow Instructions with Human Feedback* — Long Ouyang, Jeffrey Wu, Xu Jiang
   - teaches: Large-scale application of RLHF to instruction-following language models (InstructGPT), demonstrating alignment improvements.
   - why here: A landmark application that popularized RLHF for aligning LLMs with human intent.
6. **[followup]** (2023) *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — Rafael Rafailov, Archit Sharma, Eric Mitchell
   - teaches: A simpler alternative to RLHF that optimizes the policy directly from preferences without a separate reward model.
   - why here: Addresses limitations of RLHF (complexity, instability) by reparameterizing the reward model, representing a major recent development.
7. **[followup]** (2022) *Scaling Laws for Reward Model Overoptimization* — Leo Gao, John Schulman, Jacob Hilton
   - teaches: Analyzing how optimizing too much against a learned reward model leads to overoptimization and degradation of true performance.
   - why here: Provides a critical analysis of a key failure mode in RLHF, informing best practices.

<details><summary>pipeline notes</summary>

- Planned 'Policy Gradient Methods for Reinforcement Learning with Func' resolved to 'Policy Gradient Methods for Reinforcement Learning with Func' (0.82); its rationale was discarded rather than reattached to a different paper.
- 1 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Algorithms for Inverse Reinforcement Learning'
- 7 steps: 2 prerequisite, 1 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---
