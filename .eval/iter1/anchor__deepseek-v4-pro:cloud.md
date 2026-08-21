# Learning paths — strategy `anchor`, planning model `deepseek-v4-pro:cloud`

Each block is what a user sees after typing the topic. Read it as a learner:
**would following these papers in this order teach you the topic?**

`stage` is the path's own claim about each step — `prerequisite` (read before the main idea), `anchor` (the paper that IS the topic), `followup` (what came after).

---

## dropout in neural networks

*9 steps, 66s*

1. **[prerequisite]** (1992) *Simplifying Neural Networks by Soft Weight-Sharing* — Steven J. Nowlan, Geoffrey E. Hinton
   - teaches: Regularization in neural networks by adding penalty terms to the error function to prevent overfitting
   - why here: Dropout is a regularization technique; understanding prior regularization methods helps grasp the problem dropout solves.
2. **[prerequisite]** (2006) *Nightmare at test time* — Amir Globerson, Sam T. Roweis
   - teaches: Robustness to missing or corrupted features at test time
   - why here: Dropout randomly drops units during training to simulate missing features, so this paper introduces the problem of feature deletion robustness.
3. **[prerequisite]** (2008) *Extracting and composing robust features with denoising autoencoders* — Pascal Vincent, Hugo Larochelle, Yoshua Bengio
   - teaches: Training neural networks with corrupted inputs to learn robust representations (denoising autoencoders)
   - why here: Dropout extends the idea of corrupting inputs to corrupting hidden units, so this paper introduces the core concept of noise injection for robustness.
4. **[anchor]** (2014) *Dropout: a simple way to prevent neural networks from overfitting* — Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky
   - teaches: dropout in neural networks itself
   - why here: The paper that introduced the topic you asked about — everything before this exists to make it readable.
5. **[followup]** (2015) *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* — Sergey Ioffe, Christian Szegedy
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
6. **[followup]** (2024) *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* — Sergey Ioffe
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
7. **[followup]** (2019) *A survey on Image Data Augmentation for Deep Learning* — Connor Shorten, Taghi M. Khoshgoftaar
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
8. **[followup]** (2014) *Explaining and Harnessing Adversarial Examples* — Ian Goodfellow, Jonathon Shlens, Christian Szegedy
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
9. **[followup]** (2021) *Review of deep learning: concepts, CNN architectures, challenges, applications, future directions* — Laith Alzubaidi, Jinglan Zhang, Amjad J. Humaidi
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.

<details><summary>pipeline notes</summary>

- 9 steps: 3 prerequisite, 1 anchor, 5 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## batch normalization

*9 steps, 39s*

1. **[prerequisite]** (2000) *Improving predictive inference under covariate shift by weighting the log-likelihood function* — Hidetoshi Shimodaira
   - teaches: covariate shift in statistical learning
   - why here: The target paper defines internal covariate shift by analogy; without this concept, the core problem is unclear.
2. **[prerequisite]** (2010) *Understanding the difficulty of training deep feedforward neural networks* — Xavier Glorot, Yoshua Bengio
   - teaches: why deep networks are hard to train: vanishing/exploding gradients, saturation, initialization sensitivity
   - why here: The target paper argues batch normalization mitigates these difficulties; the reader needs to know the baseline problems.
3. **[prerequisite]** (2012) *Deep Learning Made Easier by Linear Transformations in Perceptrons* — Tapani Raiko, Harri Valpola, Yann LeCun
   - teaches: normalizing hidden unit outputs to zero mean and unit slope with shortcut connections
   - why here: The target paper builds on this idea of transforming hidden activations; it is a direct precursor to batch normalization.
4. **[prerequisite]** (2014) *Mean-normalized stochastic gradient for large-scale deep learning* — Simon Wiesler, Alexander Richard, Ralf Schlüter
   - teaches: mean normalization of features improves SGD convergence
   - why here: The target paper cites mean subtraction as prior work and shows why full batch normalization is needed; the reader should know this partial solution.
5. **[anchor]** (2015) *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* — Sergey Ioffe, Christian Szegedy
   - teaches: batch normalization itself
   - why here: The paper that introduced the topic you asked about — everything before this exists to make it readable.
6. **[followup]** (2017) *Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning* — Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
7. **[followup]** (2017) *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising* — Kai Zhang, Wangmeng Zuo, Yunjin Chen
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
8. **[followup]** (2021) *A Survey of Convolutional Neural Networks: Analysis, Applications, and Prospects* — Zewen Li, Fan Liu, Wenjie Yang
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
9. **[followup]** (2015) *Deep Residual Learning for Image Recognition* — He, Kaiming, Xiangyu Zhang, Shaoqing Ren
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.

<details><summary>pipeline notes</summary>

- 9 steps: 4 prerequisite, 1 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## transformers

*9 steps, 89s*

1. **[prerequisite]** (2025) *Attention Is All You Need* — Ashish Vaswani, Noam Shazeer, Niki Parmar
   - teaches: The Transformer architecture and self-attention mechanism, including multi-head attention and positional encodings.
   - why here: The target paper builds a video model entirely from self-attention, so the reader must first understand the core Transformer design.
2. **[prerequisite]** (2020) *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* — Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov
   - teaches: Vision Transformer (ViT): how to apply a Transformer to images by splitting them into patches and treating them as tokens.
   - why here: The target paper adapts ViT to video by treating spatiotemporal patches as tokens, so understanding ViT is essential.
3. **[prerequisite]** (2018) *Non-local Neural Networks* — Xiaolong Wang, Ross Girshick, Abhinav Gupta
   - teaches: Non-local neural networks: using self-attention to capture long-range dependencies in video, as an alternative to 3D convolutions.
   - why here: The target paper contrasts its pure attention approach with prior non-local blocks that augment CNNs, so this reference clarifies the baseline the target improves upon.
4. **[prerequisite]** (2019) *Axial Attention in Multidimensional Transformers* — Jonathan Ho, Nal Kalchbrenner, Dirk Weissenborn
   - teaches: Axial attention: applying attention along separate axes of a tensor to reduce computational cost while capturing global dependencies.
   - why here: The target paper's divided space-time attention applies temporal and spatial attention separately, which is a form of axial attention; understanding this concept explains the key design choice.
5. **[anchor]** (2021) *Is Space-Time Attention All You Need for Video Understanding?* — Gedas Bertasius, Heng Wang, Lorenzo Torresani
   - teaches: transformers itself
   - why here: The paper that introduced the topic you asked about — everything before this exists to make it readable.
6. **[followup]** (2022) *Video Swin Transformer* — Ze Liu, Ning Jia, Yue Cao
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
7. **[followup]** (2022) *MViTv2: Improved Multiscale Vision Transformers for Classification and Detection* — Yanghao Li, Chao-Yuan Wu, Haoqi Fan
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
8. **[followup]** (2022) *Human Action Recognition From Various Data Modalities: A Review* — Zehua Sun, Qiuhong Ke, Hossein Rahmani
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
9. **[followup]** (2023) *UniFormer: Unifying Convolution and Self-Attention for Visual Recognition* — Kunchang Li, Yali Wang, Junhao Zhang
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.

<details><summary>pipeline notes</summary>

- 9 steps: 4 prerequisite, 1 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## diffusion models

*9 steps, 100s*

1. **[prerequisite]** (2011) *A Connection Between Score Matching and Denoising Autoencoders* — Pascal Vincent
   - teaches: Equivalence between score matching and training denoising autoencoders, introducing denoising score matching.
   - why here: The target paper's simplified training objective is a form of denoising score matching; understanding this equivalence is necessary to see why the loss works.
2. **[prerequisite]** (2024) *Deep unsupervised learning using nonequilibrium thermodynamics* — Jascha Sohl‐Dickstein
   - teaches: Diffusion probabilistic models: defining a forward diffusion process that gradually destroys data and learning a reverse process to generate data, trained via variational inference.
   - why here: The target paper directly builds on this framework, extending it with a simplified loss and improved architecture; without it, the core idea of diffusion models is missing.
3. **[prerequisite]** (2019) *Generative Modeling by Estimating Gradients of the Data Distribution* — Yang Song, Stefano Ermon
   - teaches: Score-based generative modeling using multiple noise levels and annealed Langevin dynamics, estimating score functions with denoising score matching.
   - why here: The target paper connects its diffusion model to score-based models and uses similar multi-scale noise; this reference clarifies the score perspective and sampling procedure.
4. **[prerequisite]** (2015) *U-Net: Convolutional Networks for Biomedical Image Segmentation* — Olaf Ronneberger, Philipp Fischer, Thomas Brox
   - teaches: U-Net architecture: an encoder-decoder with skip connections for dense prediction, widely used for image-to-image tasks.
   - why here: The target paper uses a modified U-Net as the neural network for the reverse diffusion process; without knowing U-Net, the architecture details are opaque.
5. **[anchor]** (2020) *Denoising Diffusion Probabilistic Models* — Ho, Jonathan, Ajay N. Jain, Pieter Abbeel
   - teaches: diffusion models itself
   - why here: The paper that introduced the topic you asked about — everything before this exists to make it readable.
6. **[followup]** (2022) *High-Resolution Image Synthesis with Latent Diffusion Models* — Robin Rombach, Andreas Blattmann, Dominik Lorenz
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
7. **[followup]** (2021) *Diffusion Models Beat GANs on Image Synthesis* — Prafulla Dhariwal, Alex Nichol
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
8. **[followup]** (2023) *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation* — Nataniel Ruiz, Yuanzhen Li, Varun Jampani
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
9. **[followup]** (2023) *Diffusion Models in Vision: A Survey* — Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.

<details><summary>pipeline notes</summary>

- 9 steps: 4 prerequisite, 1 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## generative adversarial networks

*9 steps, 80s*

1. **[prerequisite]** (2017) *GAN（Generative Adversarial Nets）* — 柴田 淳司
   - teaches: adversarial training framework for generative models
   - why here: the target paper extends GANs to conditional setting; understanding the base GAN is necessary to follow the conditional modification
2. **[anchor]** (2014) *Conditional Generative Adversarial Nets* — Mehdi Mirza, Simon Osindero
   - teaches: generative adversarial networks itself
   - why here: The paper that introduced the topic you asked about — everything before this exists to make it readable.
3. **[followup]** (2017) *Least Squares Generative Adversarial Networks* — Xudong Mao, Qing Li, Haoran Xie
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
4. **[followup]** (2017) *Adversarial Discriminative Domain Adaptation* — Eric Tzeng, Judy Hoffman, Kate Saenko
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
5. **[followup]** (2018) *Generative Adversarial Networks: An Overview* — Antonia Creswell, Tom White, Vincent Dumoulin
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
6. **[followup]** (2018) *High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs* — Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
7. **[followup]** (2018) *Spectral Normalization for Generative Adversarial Networks* — Takeru Miyato, Toshiki Kataoka, Masanori Koyama
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
8. **[followup]** (2017) *DualGAN: Unsupervised Dual Learning for Image-to-Image Translation* — Zili Yi, Hao Zhang, Ping Tan
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
9. **[followup]** (2016) *Conditional Image Synthesis With Auxiliary Classifier GANs* — Augustus Odena, Christopher Olah, Jonathon Shlens
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.

<details><summary>pipeline notes</summary>

- 9 steps: 1 prerequisite, 1 anchor, 7 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## word embeddings

*9 steps, 39s*

1. **[anchor]** (2013) *Efficient Estimation of Word Representations in Vector Space* — Tomáš Mikolov, Kai Chen, Greg S. Corrado
   - teaches: word embeddings itself
   - why here: The paper that introduced the topic you asked about — everything before this exists to make it readable.
2. **[followup]** (2016) *Convolutional Neural Networks On Graphs With Fast Localized Spectral Filtering (Gdl Seminar)* — Michaël Defferrard
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
3. **[followup]** (2016) *Neural Architectures for Named Entity Recognition* — Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
4. **[followup]** (2015) *Learning Entity and Relation Embeddings for Knowledge Graph Completion* — Yankai Lin, Zhiyuan Liu, Maosong Sun
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
5. **[followup]** (2017) *Semantics derived automatically from language corpora contain human-like biases* — Aylin Caliskan, Joanna J. Bryson, Arvind Narayanan
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
6. **[followup]** (2017) *ConceptNet 5.5: An Open Multilingual Graph of General Knowledge* — Robert E. Speer, Joshua Chin, Catherine Havasi
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
7. **[followup]** (2018) *Deep learning for sentiment analysis: A survey* — Lei Zhang, Shuai Wang, Bing Liu
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
8. **[followup]** (2021) *Knowledge Graphs* — Aidan Hogan, Eva Blomqvist, Michael Cochez
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.
9. **[followup]** (2014) *Neural Word Embedding as Implicit Matrix Factorization* — Omer Levy, Yoav Goldberg
   - teaches: Where the idea went next
   - why here: Builds on the anchor paper and is itself widely cited, so it is the natural continuation once the anchor is understood.

<details><summary>pipeline notes</summary>

- Anchor 'Efficient Estimation of Word Representations in Vector Space' resolved, but no provider returned its reference list, so prerequisites could not be derived from it.
- 9 steps: 0 prerequisite, 1 anchor, 8 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## convolutional neural networks

*5 steps, 67s*

1. **[prerequisite]** (1989) *Handwritten Digit Recognition with a Back-Propagation Network* — Yann LeCun, Bernhard E. Boser, John S. Denker
   - teaches: Convolutional neural networks and backpropagation applied to image recognition, establishing the basic architecture of convolutional layers, pooling, and fully connected layers.
   - why here: The target paper builds upon this foundational CNN architecture; without it, the learner cannot understand the convolutional and pooling operations that are central to the model.
2. **[prerequisite]** (2010) *Convolutional networks and applications in vision* — Yann LeCun, Koray Kavukcuoglu, Clément Farabet
   - teaches: A comprehensive overview of convolutional networks, their components, and applications in vision, including invariance and feature learning.
   - why here: The target paper assumes familiarity with CNN design choices and terminology; this review provides the necessary background to follow the architecture and training discussion.
3. **[prerequisite]** (2010) *Rectified Linear Units Improve Restricted Boltzmann Machines* — Vinod Nair, Geoffrey E. Hinton
   - teaches: Rectified linear units (ReLUs) as activation functions that improve training of neural networks.
   - why here: The target paper uses ReLUs as a key component and claims they accelerate training; understanding ReLUs is necessary to grasp this design choice and its impact.
4. **[prerequisite]** (2012) *Improving neural networks by preventing co-adaptation of feature detectors* — Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky
   - teaches: Dropout as a regularization technique that prevents co-adaptation of feature detectors by randomly omitting units during training.
   - why here: The target paper employs dropout to reduce overfitting; without this background, the learner cannot understand why dropout is used and how it works.
5. **[anchor]** (2017) *ImageNet classification with deep convolutional neural networks* — Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
   - teaches: convolutional neural networks itself
   - why here: The paper that introduced the topic you asked about — everything before this exists to make it readable.

<details><summary>pipeline notes</summary>

- 5 steps: 4 prerequisite, 1 anchor, 0 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## reinforcement learning from human feedback

*0 steps, 31s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Proposed anchor 'Deep reinforcement learning from human preferences' did not resolve to a real paper and was discarded.
- No anchor paper for this topic could be identified and resolved, so the path has no target to build towards.
- Strategy 'anchor' produced no usable steps.

</details>

---

## regularization in machine learning

*6 steps, 81s*

1. **[prerequisite]** (1963) *The Logic of Least Squares* — G. A. Barnard
   - teaches: Foundations of least squares estimation, including mean square error and unbiasedness.
   - why here: The target paper contrasts ridge regression with ordinary least squares; without understanding least squares logic, the bias-variance tradeoff and motivation for biased estimation are unclear.
2. **[prerequisite]** (1955) *Solving systems of linear equations with a positive definite, symmetric, but possibly ill-conditioned matrix* — James D. Riley
   - teaches: Numerical difficulties in solving ill-conditioned positive definite linear systems, such as those arising from nonorthogonal predictors.
   - why here: Ridge regression adds a constant to the diagonal of X'X to improve conditioning; this reference explains why ill-conditioning makes least squares estimates unstable.
3. **[prerequisite]** (1962) *Applied Statistical Decision Theory.* — Harry V. Roberts, Howard Raiffa, Robert Schlaifer
   - teaches: Decision-theoretic framework for estimation, including loss functions and Bayes rules.
   - why here: The target paper evaluates estimators by mean squared error and advocates biased estimation; decision theory provides the formal basis for comparing biased and unbiased estimators.
4. **[prerequisite]** (1962) *Confidence Sets for the Mean of a Multivariate Normal Distribution* — C. Stein
   - teaches: Inadmissibility of the sample mean in high dimensions and improved confidence sets via shrinkage.
   - why here: This is a key theoretical result showing that biased (shrinkage) estimators can dominate unbiased ones, directly motivating ridge regression's biased approach.
5. **[anchor]** (1970) *Ridge Regression: Biased Estimation for Nonorthogonal Problems* — Arthur E. Hoerl, Robert W. Kennard
   - teaches: regularization in machine learning itself
   - why here: The paper that introduced the topic you asked about — everything before this exists to make it readable.
6. **[anchor]** (1996) *Regression Shrinkage and Selection Via the Lasso* — Robert Tibshirani
   - teaches: regularization in machine learning itself
   - why here: The paper that introduced the topic you asked about — everything before this exists to make it readable.

<details><summary>pipeline notes</summary>

- 6 steps: 4 prerequisite, 2 anchor, 0 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## contrastive learning

*0 steps, 30s*

**Empty path.**

<details><summary>pipeline notes</summary>

- Proposed anchor 'A Simple Framework for Contrastive Learning of Visual Representations' did not resolve to a real paper and was discarded.
- No anchor paper for this topic could be identified and resolved, so the path has no target to build towards.
- Strategy 'anchor' produced no usable steps.

</details>

---
