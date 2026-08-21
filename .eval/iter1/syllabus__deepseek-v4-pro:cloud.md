# Learning paths — strategy `syllabus`, planning model `deepseek-v4-pro:cloud`

Each block is what a user sees after typing the topic. Read it as a learner:
**would following these papers in this order teach you the topic?**

`stage` is the path's own claim about each step — `prerequisite` (read before the main idea), `anchor` (the paper that IS the topic), `followup` (what came after).

---

## dropout in neural networks

*9 steps, 67s*

1. **[prerequisite]** (1986) *Learning representations by back-propagating errors* — David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
   - teaches: Fundamentals of training neural networks using backpropagation
   - why here: Establishes the basic training procedure for neural networks, which all later regularization methods modify.
2. **[prerequisite]** (1991) *A Simple Weight Decay Can Improve Generalization* — Anders Krogh, John Hertz
   - teaches: Regularization via weight decay to reduce overfitting
   - why here: Introduces the problem of overfitting and a standard regularization technique, setting the stage for more sophisticated methods like dropout.
3. **[prerequisite]** (1996) *Bagging Predictors* — Leo Breiman
   - teaches: Ensemble averaging of multiple models to reduce variance
   - why here: Dropout is motivated by model averaging; understanding bagging provides the ensemble perspective needed to see dropout as an implicit ensemble.
4. **[prerequisite]** (2008) *Extracting and composing robust features with denoising autoencoders* — Pascal Vincent, Hugo Larochelle, Yoshua Bengio
   - teaches: Injecting noise into inputs to learn robust features
   - why here: Shows that stochastic corruption during training can improve generalization, a direct precursor to dropout's random omission of hidden units.
5. **[anchor]** (2014) *Dropout: a simple way to prevent neural networks from overfitting* — Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky
   - teaches: Dropout: randomly dropping units during training to prevent co-adaptation
   - why here: The definitive paper on dropout, presenting the method, its motivation, and extensive empirical results.
6. **[followup]** (2013) *Understanding Dropout* — Pierre Baldi, Peter Sadowski
   - teaches: Theoretical analysis of dropout as ensemble averaging and adaptive regularization
   - why here: Provides a formal analysis of why dropout works, connecting it to the ensemble and regularization ideas from the prerequisites.
7. **[followup]** (2015) *Dropout as a Bayesian Approximation: Representing Model Uncertainty in\n Deep Learning* — Yarin Gal, Zoubin Ghahramani
   - teaches: Dropout as approximate Bayesian inference for uncertainty estimation
   - why here: Extends dropout to quantify model uncertainty, a major application beyond regularization.
8. **[followup]** (2016) *A theoretically grounded application of dropout in recurrent neural networks* — Yarin Gal, Zoubin Ghahramani
   - teaches: Applying dropout to recurrent neural networks with theoretical grounding
   - why here: Shows how to correctly use dropout in RNNs, addressing a limitation of the original method.
9. **[followup]** (2017) *Concrete Dropout* — Yarin Gal, Jiri Hron, Alex Kendall
   - teaches: Automatic tuning of dropout rates via continuous relaxation
   - why here: Improves dropout by learning per-layer dropout rates, reducing the need for manual tuning.

<details><summary>pipeline notes</summary>

- 9 steps: 4 prerequisite, 1 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## batch normalization

*8 steps, 85s*

1. **[prerequisite]** (1998) *Efficient BackProp* — Yann LeCun, Léon Bottou, Genevieve Orr
   - teaches: Practical techniques for training neural networks, including input normalization and centering, which improve convergence.
   - why here: Introduces the idea that normalizing inputs helps optimization, a foundational concept for batch normalization.
2. **[prerequisite]** (2010) *Understanding the difficulty of training deep feedforward neural networks* — Xavier Glorot, Yoshua Bengio
   - teaches: The vanishing/exploding gradient problem in deep networks and the importance of proper initialization.
   - why here: Explains why deep networks are hard to train, motivating the need for techniques like batch normalization.
3. **[prerequisite]** (2000) *Improving predictive inference under covariate shift by weighting the log-likelihood function* — Hidetoshi Shimodaira
   - teaches: Covariate shift, where the input distribution changes between training and test, and its impact on learning.
   - why here: Provides the theoretical background for the concept of internal covariate shift, which batch normalization aims to reduce.
4. **[prerequisite]** (2013) *On the importance of initialization and momentum in deep learning* — Ilya Sutskever, James Martens, George E. Dahl
   - teaches: The critical role of initialization and optimization choices in training deep networks.
   - why here: Further establishes the training difficulties that batch normalization addresses, showing that careful initialization is not enough.
5. **[anchor]** (2015) *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* — Sergey Ioffe, Christian Szegedy
   - teaches: Batch normalization: normalizing layer inputs to reduce internal covariate shift, enabling higher learning rates and less dependence on initialization.
   - why here: This is the anchor paper that introduces the topic.
6. **[followup]** (2017) *Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models* — Sergey Ioffe
   - teaches: Batch renormalization, which reduces the dependence on minibatch statistics, making batch norm more robust to small batch sizes.
   - why here: Direct extension of batch norm addressing a key limitation.
7. **[followup]** (2016) *Layer Normalization* — Jimmy Ba, Jamie Kiros, Geoffrey E. Hinton
   - teaches: Layer normalization, which normalizes across features instead of batch, suitable for recurrent networks and small batches.
   - why here: An alternative normalization method that overcomes batch norm's limitations in certain settings.
8. **[followup]** (2016) *Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks* — Tim Salimans, Diederik P. Kingma
   - teaches: Weight normalization, which reparameterizes weights to decouple length and direction, improving training.
   - why here: Another alternative that achieves similar benefits without batch statistics.

<details><summary>pipeline notes</summary>

- 1 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'How Does Batch Normalization Help Optimization?'
- 8 steps: 4 prerequisite, 1 anchor, 3 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## transformers

*7 steps, 79s*

1. **[prerequisite]** (2014) *Sequence to Sequence Learning with Neural Networks* — Ilya Sutskever, Oriol Vinyals, Quoc V. Le
   - teaches: Sequence-to-sequence learning with recurrent neural networks, where an encoder compresses the input into a fixed vector and a decoder generates the output.
   - why here: Introduces the encoder-decoder framework for sequence transduction, which the Transformer later replaces the recurrent components of.
2. **[prerequisite]** (2014) *Neural Machine Translation by Jointly Learning to Align and Translate* — Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
   - teaches: Attention mechanism that allows the decoder to focus on relevant parts of the input sequence, avoiding the fixed-vector bottleneck.
   - why here: Introduces attention, the core idea that the Transformer extends to self-attention across the entire sequence.
3. **[prerequisite]** (2018) *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* — Shaojie Bai, J. Zico Kolter, Vladlen Koltun
   - teaches: Non-recurrent sequence modeling using convolutional networks and positional embeddings, showing that recurrence is not necessary for sequence tasks.
   - why here: Demonstrates a fully parallelizable sequence model with positional information, directly motivating the Transformer's design.
4. **[anchor]** (2021) *Is Space-Time Attention All You Need for Video Understanding?* — Gedas Bertasius, Heng Wang, Lorenzo Torresani
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

## diffusion models

*9 steps, 82s*

1. **[prerequisite]** (2013) *Auto-Encoding Variational Bayes* — Diederik P. Kingma, Max Welling
   - teaches: Variational inference and the reparameterization trick for training latent variable models.
   - why here: Diffusion models are trained by optimizing a variational lower bound, so understanding variational inference is essential before seeing how it is applied to diffusion processes.
2. **[prerequisite]** (2015) *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* — Jascha Sohl‐Dickstein, Eric A. Weiss, Niru Maheswaranathan
   - teaches: The original formulation of diffusion models as a reverse diffusion process trained with a variational bound.
   - why here: This paper introduces the core idea of gradually destroying and then reconstructing data, which is the foundation that the anchor paper simplifies and scales.
3. **[anchor]** (2020) *Denoising Diffusion Probabilistic Models* — Ho, Jonathan, Ajay N. Jain, Pieter Abbeel
   - teaches: The modern denoising diffusion probabilistic model with a simplified training objective and high-quality image generation.
   - why here: This is the anchor paper that made diffusion models practical and popular; it builds directly on the variational bound from step 1 and the diffusion process from step 2.
4. **[followup]** (2019) *Generative Modeling by Estimating Gradients of the Data Distribution* — Yang Song, Stefano Ermon
   - teaches: Estimating the score (gradient of the log-density) and sampling with Langevin dynamics as an alternative generative approach.
   - why here: This parallel line of work is introduced after the anchor to show a different perspective that will later be unified with diffusion models.
5. **[followup]** (2020) *Score-Based Generative Modeling through Stochastic Differential Equations* — Yang Song, Jascha Sohl‐Dickstein, Diederik P. Kingma
   - teaches: A continuous-time framework that unifies diffusion models and score-based models through stochastic differential equations.
   - why here: Having seen both DDPM and score-based models, this paper provides the theoretical connection and generalizes both.
6. **[followup]** (2021) *Improved Denoising Diffusion Probabilistic Models* — Alex Nichol, Prafulla Dhariwal
   - teaches: Practical improvements to DDPM, including learned variances and a cosine noise schedule.
   - why here: This follow-up refines the anchor, showing how to achieve better likelihoods and sample quality.
7. **[followup]** (2021) *Diffusion Models Beat GANs on Image Synthesis* — Prafulla Dhariwal, Alex Nichol
   - teaches: Scaling up diffusion models with classifier guidance to surpass GANs on image synthesis.
   - why here: This paper demonstrates the competitiveness of diffusion models, building on the improvements from step 6.
8. **[followup]** (2022) *High-Resolution Image Synthesis with Latent Diffusion Models* — Robin Rombach, Andreas Blattmann, Dominik Lorenz
   - teaches: Applying diffusion in a learned latent space to enable high-resolution image synthesis efficiently.
   - why here: This follow-up addresses the computational cost of pixel-space diffusion, using the VAE idea from step 1 to compress data before diffusing.
9. **[followup]** (2022) *Elucidating the Design Space of Diffusion-Based Generative Models* — Tero Karras, Miika Aittala, Timo Aila
   - teaches: A systematic analysis of diffusion model design choices, leading to state-of-the-art sampling speed and quality.
   - why here: This paper consolidates the lessons from previous steps and provides a modern best-practice recipe.

<details><summary>pipeline notes</summary>

- 9 steps: 2 prerequisite, 1 anchor, 6 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## generative adversarial networks

*9 steps, 84s*

1. **[prerequisite]** (2006) *A Fast Learning Algorithm for Deep Belief Nets* — Geoffrey E. Hinton, Simon Osindero, Yee‐Whye Teh
   - teaches: Deep generative models and unsupervised learning via layer-wise pretraining of restricted Boltzmann machines.
   - why here: Introduces the idea of learning a generative model of data using neural networks, which is the foundation for all generative models including GANs. It establishes the deep learning paradigm that GANs later build upon.
2. **[prerequisite]** (2010) *Noise-contrastive estimation: A new estimation principle for unnormalized statistical models* — Michael U. Gutmann, Aapo Hyvärinen
   - teaches: Training a model by contrasting data samples with noise samples, using a logistic regression classifier to estimate density ratios.
   - why here: Presents the core idea of training a model to distinguish real data from generated noise, which is a direct precursor to the adversarial training framework of GANs. It provides the statistical estimation principle that GANs generalize.
3. **[prerequisite]** (2017) *ImageNet classification with deep convolutional neural networks* — Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
   - teaches: Deep convolutional neural networks for large-scale image classification, demonstrating the power of CNNs for visual data.
   - why here: Provides the architectural foundation (CNNs) that later GAN variants like DCGAN use for image generation. Understanding CNNs is necessary to follow the followup papers that apply GANs to images.
4. **[anchor]** (2014) *Conditional Generative Adversarial Nets* — Mehdi Mirza, Simon Osindero
   - teaches: The GAN framework: a generator and discriminator play a minimax game, where the generator learns to produce samples indistinguishable from real data.
   - why here: This is the anchor paper that introduces generative adversarial networks. It builds on the previous ideas of generative modeling, noise-contrastive estimation, and deep learning to propose a novel adversarial training procedure.
5. **[followup]** (2015) *Unsupervised Representation Learning with Deep Convolutional Generative\n Adversarial Networks* — Alec Radford, Luke Metz, Soumith Chintala
   - teaches: Applying convolutional architectures to GANs (DCGAN) to improve image generation quality and stability, and showing the learned representations are useful for downstream tasks.
   - why here: Extends the GAN framework to image data by using CNNs, addressing the instability of the original GAN on images. It relies on the CNN knowledge from step 3 and the GAN framework from step 4.
6. **[followup]** (2016) *Improved Techniques for Training GANs* — Tim Salimans, Ian Goodfellow, Wojciech Zaremba
   - teaches: A set of practical techniques (feature matching, minibatch discrimination, historical averaging, etc.) to stabilize GAN training and improve sample quality.
   - why here: Addresses the training difficulties of GANs identified in the anchor paper. It provides empirical solutions that are widely used, and it sets the stage for later theoretical analyses.
7. **[followup]** (2017) *Wasserstein GAN* — Martín Arjovsky, Soumith Chintala, Léon Bottou
   - teaches: Wasserstein distance as a more meaningful loss metric for GANs, leading to more stable training and better correlation with sample quality.
   - why here: Provides a theoretical grounding for GAN training by replacing the Jensen-Shannon divergence with the Wasserstein distance. It explains why the original GAN training is unstable and offers a principled alternative, building on the empirical observations from step 6.
8. **[followup]** (2017) *Progressive Growing of GANs for Improved Quality, Stability, and\n Variation* — Tero Karras, Timo Aila, Samuli Laine
   - teaches: Progressive training of GANs, starting from low-resolution images and gradually increasing resolution, to generate high-quality, high-resolution images.
   - why here: Builds on the stability improvements from steps 6 and 7 to enable generation of high-resolution images. It introduces a training curriculum that is now standard for high-quality image synthesis.
9. **[followup]** (2019) *A Style-Based Generator Architecture for Generative Adversarial Networks* — Tero Karras, Samuli Laine, Timo Aila
   - teaches: Style-based generator architecture that separates high-level attributes from stochastic variation, enabling control over the generated image style and improving quality.
   - why here: Represents a major architectural advance in GANs, building on the progressive growing approach from step 8. It introduces the style-based generator that has become the basis for state-of-the-art image generation.

<details><summary>pipeline notes</summary>

- 9 steps: 3 prerequisite, 1 anchor, 5 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## word embeddings

*9 steps, 76s*

1. **[prerequisite]** (1990) *Indexing by latent semantic analysis* — Scott Deerwester, Susan Dumais, George W. Furnas
   - teaches: Words can be represented as vectors in a reduced-dimensional space derived from co-occurrence statistics, capturing semantic similarity.
   - why here: Introduces the foundational idea of representing word meaning as vectors in a continuous space, which all later embedding methods build upon.
2. **[prerequisite]** (2003) *A neural probabilistic language model* — BengioYoshua, DucharmeRéjean, VincentPascal
   - teaches: Neural networks can learn distributed representations of words as part of a language model, where similar words get similar vectors.
   - why here: Shows that word representations can be learned automatically from data using neural networks, a direct precursor to word2vec's training approach.
3. **[prerequisite]** (2011) *Natural Language Processing (almost) from Scratch* — Ronan Collobert, Jason Weston, Léon Bottou
   - teaches: Pre-trained word embeddings can be used as features to improve performance across many NLP tasks, demonstrating their general utility.
   - why here: Establishes the practical value of learned word representations for downstream tasks, motivating the need for efficient training methods like word2vec.
4. **[anchor]** (2013) *Efficient Estimation of Word Representations in Vector Space* — Tomáš Mikolov, Kai Chen, Greg S. Corrado
   - teaches: Efficient training of word embeddings using simple neural architectures (CBOW and skip-gram) on large corpora, producing vectors that capture rich semantic and syntactic relationships.
   - why here: This is the anchor paper that introduced word2vec, the method that popularized word embeddings and is the central topic of this sequence.
5. **[followup]** (2014) *Glove: Global Vectors for Word Representation* — Jeffrey Pennington, Richard Socher, Christopher D. Manning
   - teaches: Word embeddings can also be obtained by factorizing a global word-word co-occurrence matrix, offering an alternative to local context prediction.
   - why here: Presents a competing approach to word2vec that uses global statistics, showing that the core idea can be implemented differently and sparking comparisons.
6. **[followup]** (2015) *Improving Distributional Similarity with Lessons Learned from Word Embeddings* — Omer Levy, Yoav Goldberg, Ido Dagan
   - teaches: The success of word2vec can be explained by its connection to traditional distributional similarity methods, and hyperparameters play a crucial role.
   - why here: Analyzes why word2vec works, linking it back to earlier distributional semantics and providing practical guidance, deepening understanding of the anchor method.
7. **[followup]** (2017) *Enriching Word Vectors with Subword Information* — Piotr Bojanowski, Édouard Grave, Armand Joulin
   - teaches: Word embeddings can be improved by incorporating subword information (character n-grams), which helps with rare words and morphology.
   - why here: Extends word2vec to handle subword units, addressing a limitation of the original method and showing how embeddings can be enhanced.
8. **[followup]** (2018) *Deep Contextualized Word Representations* — Matthew E. Peters, Mark E Neumann, Mohit Iyyer
   - teaches: Word representations can be contextual, meaning the vector for a word depends on its surrounding sentence, capturing polysemy and context-specific meaning.
   - why here: Introduces a major shift from static embeddings to contextual ones, building on the idea of learned representations but making them dynamic.
9. **[followup]** (2018) *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* — Jacob Devlin, Ming-Wei Chang, Kenton Lee
   - teaches: Large-scale pre-training of deep bidirectional transformers produces highly effective contextual embeddings that can be fine-tuned for many tasks, becoming the new standard.
   - why here: Represents the culmination of the contextual embedding trend, showing how the ideas from word2vec evolved into powerful pre-trained language models.

<details><summary>pipeline notes</summary>

- 9 steps: 3 prerequisite, 1 anchor, 5 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## convolutional neural networks

*7 steps, 74s*

1. **[prerequisite]** (1962) *Receptive fields, binocular interaction and functional architecture in the cat's visual cortex* — David H. Hubel, T. N. Wiesel
   - teaches: Biological visual cortex uses local receptive fields and hierarchical processing, inspiring local connectivity in artificial networks.
   - why here: Provides the biological motivation for local connectivity and hierarchical feature extraction, which are core to CNNs; this is the foundational idea from neuroscience that CNNs build upon.
2. **[prerequisite]** (1986) *Learning representations by back-propagating errors* — David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
   - teaches: Backpropagation algorithm for training multi-layer neural networks by gradient descent.
   - why here: CNNs are trained with backpropagation; this paper introduces the learning algorithm that makes training deep networks possible, a necessary tool before any convolutional architecture can be learned.
3. **[prerequisite]** (1980) *Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position* — Kunihiko Fukushima
   - teaches: Neocognitron: an early neural network with local receptive fields and weight sharing, robust to shifts in position.
   - why here: Translates the biological idea into a computational model with convolutional-like layers and weight sharing, directly preceding the modern CNN; it shows how local connectivity and shared weights can be implemented in a trainable network (though trained differently).
4. **[anchor]** (1998) *Gradient-based learning applied to document recognition* — Yann LeCun, Léon Bottou, Yoshua Bengio
   - teaches: Convolutional neural networks (LeNet-5) trained by backpropagation for document recognition, combining local receptive fields, weight sharing, and subsampling.
   - why here: This is the anchor paper that defines the modern CNN: it integrates the prerequisites (local connectivity from Hubel & Wiesel, backprop from Rumelhart, and the neocognitron's architecture) into a practical, end-to-end trainable system, establishing the standard CNN architecture.
5. **[followup]** (2017) *ImageNet classification with deep convolutional neural networks* — Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
   - teaches: Deep CNNs (AlexNet) with ReLU activations, dropout, and GPU training achieve breakthrough performance on large-scale image classification.
   - why here: Shows that scaling up the CNN architecture from LeNet with more layers, data, and compute leads to dramatic improvements, reigniting interest in deep learning; it builds directly on the anchor's architecture but demonstrates its power on a much larger problem.
6. **[followup]** (2014) *Visualizing and Understanding Convolutional Networks* — Matthew D. Zeiler, Rob Fergus
   - teaches: Visualization techniques (deconvolutional networks) reveal what features CNNs learn at each layer, aiding understanding and architecture improvement.
   - why here: After seeing the success of AlexNet, this paper provides insight into why deep CNNs work by visualizing learned features, which helps diagnose and improve architectures; it is a natural followup to understand the internal representations of the deep network introduced in the previous step.
7. **[followup]** (2016) *Deep Residual Learning for Image Recognition* — Kaiming He, Xiangyu Zhang, Shaoqing Ren
   - teaches: Residual connections (skip connections) enable training of very deep networks (ResNet) by addressing degradation, achieving state-of-the-art results.
   - why here: Building on the understanding from visualization and the trend toward deeper networks, this paper introduces a key architectural innovation that allows training hundreds of layers, overcoming limitations of plain deep CNNs; it represents a major extension of the CNN paradigm.

<details><summary>pipeline notes</summary>

- 7 steps: 3 prerequisite, 1 anchor, 3 follow-up. Planned by deepseek-v4-pro:cloud.
- 2 of 6 consecutive steps go backwards in time by more than two years — the sequence may be ordered by topic rather than by dependency.

</details>

---

## reinforcement learning from human feedback

*3 steps, 94s*

1. **[prerequisite]** (1999) *Policy Gradient Methods for Reinforcement Learning with Function Approximation* — Richard S. Sutton, David McAllester, Satinder Singh
   - teaches: Fundamentals of policy optimization in reinforcement learning, including the policy gradient theorem.
   - why here: RLHF optimizes a policy using gradient-based methods; this paper provides the necessary RL foundation.
2. **[prerequisite]** (2014) *Inverse Reinforcement Learning algorithms and features for robot navigation in crowds: An experimental comparison* — Dizan Vasquez, Billy Okal, Kai O. Arras
   - teaches: Learning a reward function from observed behavior, a precursor to learning from human feedback.
   - why here: RLHF learns a reward model from human preferences, which is conceptually similar to IRL; this paper introduces the idea of inferring rewards from external signals.
3. **[prerequisite]** (2012) *Preference-based reinforcement learning: a formal framework and a policy iteration algorithm* — Johannes Fürnkranz, Eyke Hüllermeier, Weiwei Cheng
   - teaches: Formalizing RL with preferences instead of numerical rewards, and an algorithm for policy iteration.
   - why here: This is the direct precursor to deep RLHF, establishing the framework of learning from pairwise preferences.

<details><summary>pipeline notes</summary>

- 5 planned step(s) named a paper that could not be found and were dropped rather than substituted: 'Deep Reinforcement Learning from Human Preferences'; 'Fine-Tuning Language Models from Human Preferences'; 'Training language models to follow instructions with human feedback'; 'Direct Preference Optimization: Your Language Model is Secretly a Rewa'…
- 3 steps: 3 prerequisite, 0 anchor, 0 follow-up. Planned by deepseek-v4-pro:cloud.
- No anchor paper in this path — nothing here is the topic itself, only work around it. That is a weaker result than it looks.

</details>

---

## regularization in machine learning

*9 steps, 83s*

1. **[prerequisite]** (1992) *Neural Networks and the Bias/Variance Dilemma* — Stuart Geman, Elie Bienenstock, René Doursat
   - teaches: Bias-variance tradeoff and overfitting
   - why here: Establishes the bias-variance tradeoff, the core reason regularization is needed; no prior knowledge assumed.
2. **[prerequisite]** (1956) *INADMISSIBILITY OF THE USUAL ESTIMATOR FOR THE MEAN OF A MULTIVARIATE NORMAL DISTRIBUTION* — Charles Stein
   - teaches: Shrinkage can improve estimation
   - why here: Shows that shrinking estimates toward a common value can reduce total error, providing theoretical motivation for penalization introduced next.
3. **[prerequisite]** (1970) *Ridge Regression: Biased Estimation for Nonorthogonal Problems* — Arthur E. Hoerl, Robert W. Kennard
   - teaches: L2 penalty in linear regression
   - why here: Applies shrinkage via an L2 penalty to linear regression, giving the first widely used regularization technique.
4. **[prerequisite]** (1992) *A Practical Bayesian Framework for Backpropagation Networks* — David Mackay
   - teaches: Regularization as Bayesian prior
   - why here: Connects regularization to Bayesian priors, showing weight decay as maximum a posteriori estimation, which deepens understanding before sparse penalties.
5. **[anchor]** (1996) *Regression Shrinkage and Selection Via the Lasso* — Robert Tibshirani
   - teaches: L1 penalty and variable selection
   - why here: Introduces L1 regularization, the anchor method that extends shrinkage to variable selection and defines modern regularization.
6. **[followup]** (2004) *Least angle regression* — Bradley Efron, Trevor Hastie, Iain M. Johnstone
   - teaches: Efficient algorithm for lasso
   - why here: Provides an efficient algorithm for computing lasso solutions, making the anchor method practical for larger problems.
7. **[followup]** (2005) *Regularization and Variable Selection Via the Elastic Net* — Hui Zou, Trevor Hastie
   - teaches: Combining L1 and L2 penalties
   - why here: Combines L1 and L2 penalties to address lasso limitations with correlated features, a direct extension of the anchor.
8. **[followup]** (2014) *Dropout: a simple way to prevent neural networks from overfitting* — Nitish Srivastava, Geoffrey E. Hinton, Alex Krizhevsky
   - teaches: Stochastic regularization for neural networks
   - why here: Transfers regularization to neural networks via stochastic noise injection, showing how the idea generalizes beyond linear models.
9. **[followup]** (2016) *Understanding deep learning requires rethinking generalization* — Chiyuan Zhang, Samy Bengio, Moritz Hardt
   - teaches: Implicit regularization in deep learning
   - why here: Challenges classical bias-variance explanations for deep networks, pointing to implicit regularization and motivating current research.

<details><summary>pipeline notes</summary>

- 9 steps: 4 prerequisite, 1 anchor, 4 follow-up. Planned by deepseek-v4-pro:cloud.

</details>

---

## contrastive learning

*1 steps, 78s*

1. **[prerequisite]** (2010) *Noise-contrastive estimation: A new estimation principle for unnormalized statistical models* — Michael U. Gutmann, Aapo Hyvärinen
   - teaches: Estimating model parameters by contrasting data samples with noise samples, avoiding normalization.
   - why here: Introduces the principle of contrasting positive and negative samples, which underlies the use of negative examples in contrastive learning.

<details><summary>pipeline notes</summary>

- 8 planned step(s) named a paper that could not be found and were dropped rather than substituted: "Signature Verification using a 'Siamese' Time Delay Neural Network"; 'Dimensionality Reduction by Learning an Invariant Mapping'; 'Distributed Representations of Words and Phrases and their Composition'; 'A Simple Framework for Contrastive Learning of Visual Representations'…
- 1 steps: 1 prerequisite, 0 anchor, 0 follow-up. Planned by deepseek-v4-pro:cloud.
- No anchor paper in this path — nothing here is the topic itself, only work around it. That is a weaker result than it looks.

</details>

---
