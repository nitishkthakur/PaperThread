# Learning paths — strategy `syllabus`, planning model `deepseek-v4-pro:cloud`

Each block is what a user sees after typing the topic. Read it as a learner:
**would following these papers in this order teach you the topic?**

`stage` is the path's own claim about each step — `prerequisite` (read before the main idea), `anchor` (the paper that IS the topic), `followup` (what came after).

---

## dropout in neural networks

*4 steps, 24s — self-rated confidence **LOW** (0.05)*

> never reaches the topic itself (no anchor paper); no prerequisites — this is a reading list, not a path; part of the plan could not be found and was dropped; only 4 steps survived

1. **[followup]** (2013) *Understanding Dropout: Training Multi-Layer Perceptrons with Auxiliary Independent Stochastic Neurons* — Kyunghyun Cho
   - teaches: Theoretical analysis of dropout as ensemble averaging and adaptive regularization
   - why here: Provides a formal analysis of why dropout works, connecting it to the ensemble and regularization ideas from the prerequisites.
2. **[followup]** (2015) *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning* — Yarin Gal, Zoubin Ghahramani
   - teaches: Dropout as approximate Bayesian inference for uncertainty estimation
   - why here: Extends dropout to quantify model uncertainty, a major application beyond regularization.
3. **[followup]** (2015) *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks* — Yarin Gal, Zoubin Ghahramani
   - teaches: Applying dropout to recurrent neural networks with theoretical grounding
   - why here: Shows how to correctly use dropout in RNNs, addressing a limitation of the original method.
4. **[followup]** (2017) *Concrete Dropout* — Yarin Gal, Jiri Hron, Alex Kendall
   - teaches: Automatic tuning of dropout rates via continuous relaxation
   - why here: Improves dropout by learning per-layer dropout rates, reducing the need for manual tuning.

<details><summary>pipeline notes</summary>

- INCOMPLETE: this path never reaches the topic itself. The paper that would be the destination could not be identified or could not be found, so what follows is background only. Treat it as a partial result.
- 5 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Learning representations by back-propagating errors'; 'A Simple Weight Decay Can Improve Generalization'; 'Bagging predictors'; 'Extracting and Composing Robust Features with Denoising Autoencoders'…
- 4 steps: 0 prerequisite, 0 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## batch normalization

*5 steps, 50s — self-rated confidence **MEDIUM** (0.65)*

> no prerequisites — this is a reading list, not a path; part of the plan could not be found and was dropped

1. **[anchor]** (2015) *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* — Sergey Ioffe, Christian Szegedy
   - teaches: Batch normalization: normalizing layer inputs to reduce internal covariate shift, enabling higher learning rates and less dependence on initialization.
   - why here: This is the anchor paper that introduces the topic.
2. **[followup]** (2017) *Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models* — Sergey Ioffe
   - teaches: Batch renormalization, which reduces the dependence on minibatch statistics, making batch norm more robust to small batch sizes.
   - why here: Direct extension of batch norm addressing a key limitation.
3. **[followup]** (2016) *Layer Normalization* — Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
   - teaches: Layer normalization, which normalizes across features instead of batch, suitable for recurrent networks and small batches.
   - why here: An alternative normalization method that overcomes batch norm's limitations in certain settings.
4. **[followup]** (2016) *Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks* — Tim Salimans, Diederik P. Kingma
   - teaches: Weight normalization, which reparameterizes weights to decouple length and direction, improving training.
   - why here: Another alternative that achieves similar benefits without batch statistics.
5. **[followup]** (2018) *How Does Batch Normalization Help Optimization?* — Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas
   - teaches: Analysis showing that batch norm's success is not primarily due to reducing internal covariate shift, but rather due to smoothing the optimization landscape.
   - why here: Provides a deeper understanding and challenges the original motivation, representing a key followup analysis.

<details><summary>pipeline notes</summary>

- 4 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Efficient BackProp'; 'Understanding the difficulty of training deep feedforward neural netwo'; 'Improving predictive inference under covariate shift by weighting the '; 'On the importance of initialization and momentum in deep learning'
- 5 steps: 0 prerequisite, 1 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## transformers

*3 steps, 44s — self-rated confidence **MEDIUM** (0.70)*

> part of the plan could not be found and was dropped; only 3 steps survived

1. **[prerequisite]** (2014) *Sequence to Sequence Learning with Neural Networks* — Ilya Sutskever, Oriol Vinyals, Quoc V. Le
   - teaches: Sequence-to-sequence learning with recurrent neural networks, where an encoder compresses the input into a fixed vector and a decoder generates the output.
   - why here: Introduces the encoder-decoder framework for sequence transduction, which the Transformer later replaces the recurrent components of.
2. **[prerequisite]** (2014) *Neural Machine Translation by Jointly Learning to Align and Translate* — Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
   - teaches: Attention mechanism that allows the decoder to focus on relevant parts of the input sequence, avoiding the fixed-vector bottleneck.
   - why here: Introduces attention, the core idea that the Transformer extends to self-attention across the entire sequence.
3. **[anchor]** (2017) *Attention Is All You Need* — Ashish Vaswani, Noam Shazeer, Niki Parmar
   - teaches: The Transformer architecture: multi-head self-attention, positional encoding, and feed-forward layers, entirely without recurrence or convolution.
   - why here: This is the paper that defines the Transformer, the topic the learner asked about.

<details><summary>pipeline notes</summary>

- 4 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Convolutional Sequence to Sequence Learning'; 'BERT: Pre-training of Deep Bidirectional Transformers for Language Und'; 'An Image is Worth 16x16 Words: Transformers for Image Recognition at S'; 'Transformer-XL: Attentive Language Models Beyond a Fixed-Length Contex'
- 3 steps: 2 prerequisite, 1 anchor, 0 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## diffusion models

*0 steps, 44s — self-rated confidence **LOW** (0.00)*

**Empty path.**

<details><summary>pipeline notes</summary>

- 9 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Auto-Encoding Variational Bayes'; 'Deep Unsupervised Learning using Nonequilibrium Thermodynamics'; 'Denoising Diffusion Probabilistic Models'; 'Generative Modeling by Estimating Gradients of the Data Distribution'…
- Strategy 'syllabus' produced no usable steps.

</details>

---

## generative adversarial networks

*0 steps, 53s — self-rated confidence **LOW** (0.00)*

**Empty path.**

<details><summary>pipeline notes</summary>

- 9 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'A fast learning algorithm for deep belief nets'; 'Noise-contrastive estimation: A new estimation principle for unnormali'; 'ImageNet Classification with Deep Convolutional Neural Networks'; 'Generative Adversarial Nets'…
- Strategy 'syllabus' produced no usable steps.

</details>

---

## word embeddings

*0 steps, 53s — self-rated confidence **LOW** (0.00)*

**Empty path.**

<details><summary>pipeline notes</summary>

- 9 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Indexing by Latent Semantic Analysis'; 'A Neural Probabilistic Language Model'; 'Natural Language Processing (almost) from Scratch'; 'Efficient Estimation of Word Representations in Vector Space'…
- Strategy 'syllabus' produced no usable steps.

</details>

---

## convolutional neural networks

*0 steps, 44s — self-rated confidence **LOW** (0.00)*

**Empty path.**

<details><summary>pipeline notes</summary>

- 7 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Receptive fields, binocular interaction and functional architecture in'; 'Learning representations by back-propagating errors'; 'Neocognitron: A self-organizing neural network model for a mechanism o'; 'Gradient-Based Learning Applied to Document Recognition'…
- Strategy 'syllabus' produced no usable steps.

</details>

---

## reinforcement learning from human feedback

*3 steps, 32s — self-rated confidence **LOW** (0.50)*

> no prerequisites — this is a reading list, not a path; part of the plan could not be found and was dropped; only 3 steps survived

1. **[anchor]** (2017) *Deep reinforcement learning from human preferences* — Paul Christiano, Jan Leike, Tom B. Brown
   - teaches: Scaling preference-based RL to deep neural networks by training a reward model from human comparisons and optimizing a policy with PPO.
   - why here: This is the seminal paper that introduced RLHF as we know it, combining deep RL with human preference learning.
2. **[followup]** (2022) *Training language models to follow instructions with human feedback* — Long Ouyang, Jeff Wu, Xu Jiang
   - teaches: Large-scale application of RLHF to instruction-following language models (InstructGPT), demonstrating alignment improvements.
   - why here: A landmark application that popularized RLHF for aligning LLMs with human intent.
3. **[followup]** (2023) *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — Rafael Rafailov, Archit Sharma, Eric Mitchell
   - teaches: A simpler alternative to RLHF that optimizes the policy directly from preferences without a separate reward model.
   - why here: Addresses limitations of RLHF (complexity, instability) by reparameterizing the reward model, representing a major recent development.

<details><summary>pipeline notes</summary>

- 5 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Policy Gradient Methods for Reinforcement Learning with Function Appro'; 'Algorithms for Inverse Reinforcement Learning'; 'Preference-based Reinforcement Learning: A Formal Framework and a Poli'; 'Fine-Tuning Language Models from Human Preferences'…
- 3 steps: 0 prerequisite, 1 anchor, 2 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## regularization in machine learning

*0 steps, 38s — self-rated confidence **LOW** (0.00)*

**Empty path.**

<details><summary>pipeline notes</summary>

- 9 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Neural Networks and the Bias/Variance Dilemma'; 'Inadmissibility of the Usual Estimator for the Mean of a Multivariate '; 'Ridge Regression: Biased Estimation for Nonorthogonal Problems'; 'A Practical Bayesian Framework for Backpropagation Networks'…
- Strategy 'syllabus' produced no usable steps.

</details>

---

## contrastive learning

*1 steps, 47s — self-rated confidence **LOW** (0.50)*

> no prerequisites — this is a reading list, not a path; part of the plan could not be found and was dropped; only 1 steps survived

1. **[anchor]** (2020) *A Simple Framework for Contrastive Learning of Visual Representations* — Ting Chen, Simon Kornblith, Mohammad Norouzi
   - teaches: A simple framework for contrastive learning of visual representations using data augmentation, a projection head, and large batch training.
   - why here: The anchor paper that popularized contrastive learning in computer vision, synthesizing the prerequisite ideas into a highly effective method.

<details><summary>pipeline notes</summary>

- 8 planned step(s) named a paper that could not be found and were dropped rather than substituted: "Signature Verification using a 'Siamese' Time Delay Neural Network"; 'Dimensionality Reduction by Learning an Invariant Mapping'; 'Noise-contrastive estimation: A new estimation principle for unnormali'; 'Distributed Representations of Words and Phrases and their Composition'…
- 1 steps: 0 prerequisite, 1 anchor, 0 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---
