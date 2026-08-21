# Learning paths — strategy `syllabus`, planning model `deepseek-v4-pro:cloud`

Each block is what a user sees after typing the topic. Read it as a learner:
**would following these papers in this order teach you the topic?**

`stage` is the path's own claim about each step — `prerequisite` (read before the main idea), `anchor` (the paper that IS the topic), `followup` (what came after).

---

## dropout in neural networks

*8 steps, 37s*

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
