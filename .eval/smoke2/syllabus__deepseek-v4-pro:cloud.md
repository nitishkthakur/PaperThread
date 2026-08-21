# Learning paths — strategy `syllabus`, planning model `deepseek-v4-pro:cloud`

Each block is what a user sees after typing the topic. Read it as a learner:
**would following these papers in this order teach you the topic?**

`stage` is the path's own claim about each step — `prerequisite` (read before the main idea), `anchor` (the paper that IS the topic), `followup` (what came after).

---

## dropout in neural networks

*8 steps, 0s — self-rated confidence **HIGH** (1.00)*

> reaches the topic, with prerequisites, all steps verified

1. **[prerequisite]** (1986) *Learning representations by back-propagating errors* — David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
   - teaches: Training neural networks by gradient descent via backpropagation
   - why here: Dropout is a regularization technique applied during neural network training; the reader must first understand how networks are trained with backpropagation.
2. **[prerequisite]** (1992) *A Practical Bayesian Framework for Backpropagation Networks* — David Mackay
   - teaches: Overfitting and regularization in neural networks
   - why here: Dropout addresses overfitting, so this step introduces the problem of overfitting and classical regularization approaches, building on backpropagation.
3. **[prerequisite]** (1996) *Bagging Predictors* — Leo Breiman
   - teaches: Ensemble averaging to reduce variance
   - why here: Dropout approximates an ensemble of subnetworks; bagging provides the conceptual foundation for why averaging multiple models improves generalization.
4. **[anchor]** (2014) *Dropout: a simple way to prevent neural networks from overfitting* — Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky
   - teaches: Randomly dropping units during training to prevent co-adaptation
   - why here: This is the core paper that introduces dropout, combining the regularization goal from step 2 with the ensemble idea from step 3.
5. **[followup]** (2013) *Understanding Dropout* — Pierre Baldi, Peter Sadowski
   - teaches: Analysis of dropout as ensemble averaging and adaptive regularization
   - why here: After the anchor, this paper provides a theoretical analysis of why dropout works, explicitly connecting it to the bagging concept from step 3.
6. **[followup]** (2015) *Dropout as a Bayesian Approximation: Representing Model Uncertainty in\n Deep Learning* — Yarin Gal, Zoubin Ghahramani
   - teaches: Dropout as approximate Bayesian inference
   - why here: Building on the empirical understanding from step 5, this paper gives a Bayesian interpretation of dropout, linking it to model uncertainty.
7. **[followup]** (2015) *Variational Dropout and the Local Reparameterization Trick* — Diederik P. Kingma, Tim Salimans, Max Welling
   - teaches: Learning per-weight dropout rates via variational inference
   - why here: Extends the Bayesian view from step 6 by making dropout rates learnable parameters, improving regularization adaptivity.
8. **[followup]** (2017) *Concrete Dropout* — Yarin Gal, Jiri Hron, Alex Kendall
   - teaches: Automatic tuning of dropout rate with continuous relaxation
   - why here: Further automates dropout rate selection, building on the variational dropout framework from step 7 to provide a practical adaptive method.

<details><summary>pipeline notes</summary>

- 8 steps: 3 prerequisite, 1 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## transformers

*6 steps, 147s — self-rated confidence **LOW** (0.40)*

> never reaches the topic itself (no anchor paper); part of the plan could not be found and was dropped

1. **[prerequisite]** (2014) *Sequence to Sequence Learning with Neural Networks* — Ilya Sutskever, Oriol Vinyals, Quoc V. Le
   - teaches: The encoder-decoder framework for sequence-to-sequence learning using recurrent neural networks, where an input sequence is encoded into a fixed-length vector and then decoded into an output sequence.
   - why here: This is the foundational architecture for sequence transduction tasks like machine translation, which the Transformer later improves upon. It establishes the problem setting and the baseline approach.
2. **[prerequisite]** (2014) *Neural Machine Translation by Jointly Learning to Align and Translate* — Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
   - teaches: The attention mechanism, which allows the decoder to focus on different parts of the input sequence at each step, rather than relying on a single fixed-length context vector.
   - why here: Attention is the key idea that the Transformer generalizes and builds upon. Understanding attention in the context of RNNs is essential before seeing how it can replace recurrence entirely.
3. **[prerequisite]** (2017) *Convolutional Sequence to Sequence Learning* — Jonas Gehring, Michael Auli, David Grangier
   - teaches: A fully convolutional sequence-to-sequence model that removes recurrence, enabling parallel computation over the input sequence, and uses position embeddings to retain order information.
   - why here: This paper demonstrates that sequence transduction can be done without recurrent networks, paving the way for the Transformer's non-recurrent architecture. It also introduces the idea of positional information being added explicitly, which the Transformer adopts.
4. **[followup]** (2018) *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — Jacob Devlin, Ming-Wei Chang, Kenton Lee
   - teaches: Pre-training a deep bidirectional Transformer encoder on a large text corpus using masked language modeling and next sentence prediction, then fine-tuning on downstream tasks, achieving state-of-the-art results across many NLP benchmarks.
   - why here: BERT is one of the most influential follow-ups, showing how the Transformer encoder can be pre-trained as a general language understanding model, leading to a paradigm shift in NLP.
5. **[followup]** (2019) *Transformer-XL: Attentive Language Models beyond a Fixed-Length Context* — Zihang Dai, Zhilin Yang, Yiming Yang
   - teaches: An extension of the Transformer that introduces a segment-level recurrence mechanism and a novel positional encoding scheme, allowing the model to capture longer-term dependencies beyond a fixed-length context.
   - why here: Transformer-XL addresses a key limitation of the original Transformer—its fixed context window—and is important for tasks requiring long-range memory, such as language modeling and text generation.
6. **[followup]** (2020) *Reformer: The Efficient Transformer* — Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya
   - teaches: Techniques to reduce the memory and computational complexity of the Transformer, including locality-sensitive hashing for attention and reversible residual layers, enabling the model to handle much longer sequences efficiently.
   - why here: Reformer tackles the scalability issues of the Transformer, making it practical for very long sequences and large-scale applications, and represents a major line of work on efficient Transformers.

<details><summary>pipeline notes</summary>

- INCOMPLETE: this path never reaches the topic itself. The paper that would be the destination could not be identified or could not be found, so what follows is background only. Treat it as a partial result.
- 2 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Attention is All You Need'; 'Improving Language Understanding by Generative Pre-Training'
- 6 steps: 3 prerequisite, 0 anchor, 3 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---
