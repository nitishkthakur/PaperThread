# Learning paths — strategy `structural`, planning model `deepseek-v4-pro:cloud`

Each block is what a user sees after typing the topic. Read it as a learner:
**would following these papers in this order teach you the topic?**

`stage` is the path's own claim about each step — `prerequisite` (read before the main idea), `anchor` (the paper that IS the topic), `followup` (what came after).

---

## dropout in neural networks

*9 steps, 730s*

1. **[g5]** (2006) *Reducing the Dimensionality of Data with Neural Networks* — Geoffrey E. Hinton, Ruslan Salakhutdinov
   - teaches: Layer-wise pretraining of autoencoders produces better low-dimensional codes than PCA.
   - why here: Showed that deep autoencoders can outperform PCA for dimensionality reduction when initialized properly, enabling later deep network training.
2. **[g5]** (1998) *Gradient-based learning applied to document recognition* — Yann LeCun, Léon Bottou, Yoshua Bengio
   - teaches: From the abstract: Multilayer neural networks trained with the back-propagation algorithm constitute the best example of a successful gradient based learning technique.
   - why here: 7 of the 47 papers this topic surfaced cite it, so it is part of the background they have in common. Keyword search did not return it — it was reached by following citations backwards, which is how work that predates the topic's vocabulary is found.
3. **[-]** (2023) *The Deep Arbitrary Polynomial Chaos Neural Network or how Deep Artificial Neural Networks could benefit from Data-Driven Homogeneous Chaos Theory* — Sergey Oladyshkin, Timothy Praditia, Ilja Kröker
   - teaches: From the abstract: Artificial Intelligence and Machine learning have been widely used in various fields of mathematical computing, physical modeling, computational science, communication science, and stochastic analysis.
   - why here: It matched the topic directly; its position in the citation graph is not distinctive enough to say more.
4. **[-]** (2011) *Acoustic Modeling Using Deep Belief Networks* — Abdelrahman Mohamed, George E. Dahl, Geoffrey E. Hinton
   - teaches: Pretraining a deep belief network on spectral features then fine-tuning with backpropagation improves phone recognition on TIMIT.
   - why here: Demonstrated that deep neural networks pretrained as generative models can replace GMMs in speech recognition, showing practical value of deep architectures.
5. **[-]** (2010) *Rectified Linear Units Improve Restricted Boltzmann Machines* — Vinod Nair, Geoffrey E. Hinton
   - teaches: ReLUs can be derived as infinite copies of binary units with shifted biases and learn better features for vision tasks.
   - why here: Introduced rectified linear units as an alternative to binary hidden units in RBMs, improving feature learning and preserving intensity information.
6. **[g14]** (2012) *Improving neural networks by preventing co-adaptation of feature detectors* — Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky
   - teaches: From the abstract: When a large feedforward neural network is trained on a small training set, it typically performs poorly on held-out test data.
   - why here: 8 of the 47 papers this topic surfaced cite it, so it is part of the background they have in common.
7. **[-]** (2017) *ImageNet classification with deep convolutional neural networks* — Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
   - teaches: From the abstract: We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes.
   - why here: 7 of the 47 papers this topic surfaced cite it, so it is part of the background they have in common.
8. **[-]** (2014) *Dropout: a simple way to prevent neural networks from overfitting* — Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky
   - teaches: From the abstract: Deep neural nets with a large number of parameters are very powerful machine learning systems.
   - why here: 4 of the 47 papers this topic surfaced cite it, so it is part of the background they have in common.
9. **[g14]** (2015) *Towards Dropout Training for Convolutional Neural Networks* — Haibing Wu, Xiaodong Gu
   - teaches: From the abstract: Recently, dropout has seen increasing use in deep learning.
   - why here: It matched the topic directly; its position in the citation graph is not distinctive enough to say more.

<details><summary>pipeline notes</summary>

- 2 of 47 results have no abstract; retained deliberately.
- Citation expansion added 572 papers and 793 edges; 20 are cited by 3+ of the topic's own candidates.
- 6 paper(s) kept structural explanations — the model did not return usable output for them.
- Prerequisite edges stayed structural (A cites B, B is shared foundation): [llm] every prerequisite judgment batch failed
- Subtopics stayed unnamed: [ollama_local] every model for role 'topic_decomposition' failed — deepseek-v4-flash:cloud: [ollama_local] invalid structured output after 3 attempts: $.groups[0]: missing required field 'summary'; glm-5.1:cloud: [ollama_local] invalid structured output after 3 attempts: $.groups[0]: missing required field 'label'; minimax-m2.7:cloud: [ollama_local] invalid structured output after 3 attempts: $.groups[0]: missing required field 'summary'
- 9 papers across 6 level(s); 4 reached by citation expansion rather than search; 14 prerequisite edge(s); prerequisites judged by an LLM over citation candidates.

</details>

---

## batch normalization

*9 steps, 566s*

1. **[-]** (2015) *ImageNet Large Scale Visual Recognition Challenge* — Olga Russakovsky, Jia Deng, Hao Su
   - teaches: The evaluation metrics and dataset challenges of the ImageNet competition.
   - why here: Defined the large-scale visual recognition benchmark that became the primary testing ground for deep learning techniques.
2. **[-]** (2014) *Dropout: a simple way to prevent neural networks from overfitting* — Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky
   - teaches: From the abstract: Deep neural nets with a large number of parameters are very powerful machine learning systems.
   - why here: 6 of the 47 papers this topic surfaced cite it, so it is part of the background they have in common. Keyword search did not return it — it was reached by following citations backwards, which is how work that predates the topic's vocabulary is found.
3. **[-]** (2000) *Improving predictive inference under covariate shift by weighting the log-likelihood function* — Hidetoshi Shimodaira
   - teaches: The statistical definition of covariate shift and how importance weighting can correct for distribution changes.
   - why here: Introduced the statistical concept of covariate shift, which later inspired the formulation of internal covariate shift in deep networks.
4. **[-]** (2015) *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* — Sergey Ioffe, Christian Szegedy
   - teaches: From the abstract: Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change.
   - why here: 5 of the 47 papers this topic surfaced cite it, so it is part of the background they have in common.
5. **[-]** (2015) *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* — Kaiming He, Xiangyu Zhang, Shaoqing Ren
   - teaches: The Parametric Rectified Linear Unit (PReLU) and a robust initialization method for deep rectifier models.
   - why here: Demonstrated that very deep rectifier networks could be trained from scratch using proper initialization and learnable activations.
6. **[-]** (2016) *Rethinking the Inception Architecture for Computer Vision* — Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe
   - teaches: From the abstract: Convolutional networks are at the core of most state of-the-art computer vision solutions for a wide variety of tasks.
   - why here: 7 of the 47 papers this topic surfaced cite it, so it is part of the background they have in common. Keyword search did not return it — it was reached by following citations backwards, which is how work that predates the topic's vocabulary is found.
7. **[-]** (2016) *Deep Residual Learning for Image Recognition* — Kaiming He, Xiangyu Zhang, Shaoqing Ren
   - teaches: From the abstract: Deeper neural networks are more difficult to train.
   - why here: 9 of the 47 papers this topic surfaced cite it, so it is part of the background they have in common. Keyword search did not return it — it was reached by following citations backwards, which is how work that predates the topic's vocabulary is found.
8. **[-]** (2018) *Decorrelated Batch Normalization* — Lei Huang, Dawei Yang, Bo Lang
   - teaches: From the abstract: Batch Normalization (BN) is capable of accelerating the training of deep models by centering and scaling activations within mini-batches.
   - why here: It matched the topic directly; its position in the citation graph is not distinctive enough to say more.
9. **[-]** (2020) *Cross-Iteration Batch Normalization* — Zhuliang Yao, Yue Cao, Shuxin Zheng
   - teaches: From the abstract: A well-known issue of Batch Normalization is its significantly reduced effectiveness in the case of small mini-batch sizes.
   - why here: It matched the topic directly; its position in the citation graph is not distinctive enough to say more.

<details><summary>pipeline notes</summary>

- 7 of 47 results have no abstract; retained deliberately.
- Citation expansion added 533 papers and 815 edges; 24 are cited by 3+ of the topic's own candidates.
- 6 paper(s) kept structural explanations — the model did not return usable output for them.
- Prerequisite edges stayed structural (A cites B, B is shared foundation): [llm] every prerequisite judgment batch failed
- 1 citation edge(s) formed cycles and were dropped to make the path orderable — providers disagree, and preprint/published pairs can cite each other.
- 9 papers across 5 level(s); 6 reached by citation expansion rather than search; 14 prerequisite edge(s); prerequisites judged by an LLM over citation candidates.

</details>

---

## transformers

*0 steps, 0s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Degraded: provider(s) failed — arxiv, openalex.
- No candidates matched this topic, so there is no path to build.

</details>

---

## diffusion models

*0 steps, 0s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Degraded: provider(s) failed — arxiv, openalex.
- No candidates matched this topic, so there is no path to build.

</details>

---

## generative adversarial networks

*0 steps, 0s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Degraded: provider(s) failed — arxiv, openalex.
- No candidates matched this topic, so there is no path to build.

</details>

---

## word embeddings

*0 steps, 0s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Degraded: provider(s) failed — arxiv, openalex.
- No candidates matched this topic, so there is no path to build.

</details>

---

## convolutional neural networks

*0 steps, 0s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Degraded: provider(s) failed — arxiv, openalex.
- No candidates matched this topic, so there is no path to build.

</details>

---

## reinforcement learning from human feedback

*0 steps, 0s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Degraded: provider(s) failed — arxiv, openalex.
- No candidates matched this topic, so there is no path to build.

</details>

---

## regularization in machine learning

*0 steps, 0s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Degraded: provider(s) failed — arxiv, openalex.
- No candidates matched this topic, so there is no path to build.

</details>

---

## contrastive learning

*0 steps, 0s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Degraded: provider(s) failed — arxiv, openalex.
- No candidates matched this topic, so there is no path to build.

</details>

---
