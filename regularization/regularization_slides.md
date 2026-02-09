---
marp: true
theme: default
paginate: true
backgroundColor: #fff
math: katex
style: |
  section {
    font-size: 28px;
  }
  h1 {
    color: #2c3e50;
  }
  h2 {
    color: #3498db;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
---

# Mathematics for Machine Learning

## MAD-B3-2526-S2-MAT0611

# Regularization via Bayesian Priors

---

# Connecting to Last Session

**Last time**: Bayesian inference framework

- Prior beliefs $P(\theta)$
- Likelihood $P(D|\theta)$
- Posterior $P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$

**Today**: Apply Bayesian thinking to **linear regression**

- What if we have **prior beliefs** about coefficients?
- Different priors → different regularization methods
- Bridge between Bayesian inference and optimization

**Flow**: Bayesian priors → MAP estimation → Ridge/LASSO/Elastic Net

---

# Agenda

1. **Bayesian Linear Regression & MAP Estimation**
2. **Gaussian Prior → Ridge (L2) Regularization**
   - Derivation from Bayesian principles
   - Matrix perspective & properties
3. **Laplace Prior → LASSO (L1) Regularization**
   - Why sparsity? Comparison with Gaussian
   - Derivation & properties
4. **Mixture Prior → Elastic Net**
   - Combining both priors for best of both worlds
5. **Decision Framework: When to Use Each Method**

---

# Bayesian Linear Regression: The Setup

**Standard linear model:**
$$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim N(0, \sigma^2\mathbf{I})$$

**Likelihood** (same as before):
$$P(\mathbf{Y}|\boldsymbol{\beta}, \sigma^2) = N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I})$$

**NEW: Add prior beliefs about $\boldsymbol{\beta}$**
$$P(\boldsymbol{\beta}) = \text{some distribution}$$

**Posterior** (Bayes' rule):
$$P(\boldsymbol{\beta}|\mathbf{Y}) \propto \underbrace{P(\mathbf{Y}|\boldsymbol{\beta})}_{\text{Likelihood}} \cdot \underbrace{P(\boldsymbol{\beta})}_{\text{Prior}}$$

---

# What Does Bayesian Linear Regression Do?

**Three key benefits:**

1. **Incorporates prior knowledge**
   - E.g., "coefficients should be small" → prevents overfitting
   - E.g., "most coefficients are zero" → automatic feature selection

2. **Provides full posterior distribution**
   - Not just point estimate $\hat{\boldsymbol{\beta}}$
   - Get uncertainty: $P(\boldsymbol{\beta}|\mathbf{Y})$

3. **Natural regularization**
   - Prior acts as "soft constraint"
   - Controls model complexity

**Today's focus**: How different priors lead to different regularization methods

---

# Maximum A Posteriori (MAP) Estimation

Computing the entire posterior distribution $P(\boldsymbol{\beta}|\mathbf{Y})$ is often complex

**MAP simplification**: Finding the mode (most probable value) of the posterior
$$\hat{\boldsymbol{\beta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\beta}} P(\boldsymbol{\beta}|\mathbf{Y})$$

**Taking logs** (easier to optimize):
$$\hat{\boldsymbol{\beta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\beta}} \left\{ \log P(\mathbf{Y}|\boldsymbol{\beta}) + \log P(\boldsymbol{\beta}) \right\}$$

**Or minimize negative log**:
$$\hat{\boldsymbol{\beta}}_{\text{MAP}} = \arg\min_{\boldsymbol{\beta}} \left\{ \underbrace{-\log P(\mathbf{Y}|\boldsymbol{\beta})}_{\text{Loss term}} + \underbrace{(-\log P(\boldsymbol{\beta}))}_{\text{Regularization}} \right\}$$

---

# Gaussian Prior on Coefficients

**Why Gaussian prior?** To keep coefficients small and avoid overfitting

**Prior specification:**
$$\beta_j \sim N(0, \tau^2) \quad \text{independently for each } j$$

or matricially:

$$
\boldsymbol{\beta} \sim N(\mathbf{0}, \tau^2 \mathbf{I})
$$

where $
\boldsymbol{\beta}$ is the vector of coefficients and $N(\mathbf{0}, \tau^2 \mathbf{I})$ is a multivariate normal distribution with mean vector $\mathbf{0}$ and covariance matrix $\tau^2 \mathbf{I}$.

---

# Gaussian Prior on Coefficients

**Properties:**

- Most probability mass near zero
- Favours small coefficients
- Allows non-zero values (soft constraint)
- Variance $\tau^2$ controls strength: small $\tau$ → strong prior

**When to use:**

- Believe all features somewhat relevant
- Want to reduce overfitting
- Have many correlated predictors

---

# Deriving the MAP with Gaussian Prior

**Start with Gaussian prior:**
$$P(\boldsymbol{\beta}) = \prod_{j=1}^p N(\beta_j | 0, \tau^2) \propto \exp\left(-\frac{1}{2\tau^2}\sum_{j=1}^p \beta_j^2\right)$$

**Gaussian likelihood** (assuming known $\sigma^2$):
$$P(\mathbf{Y}|\boldsymbol{\beta}) \propto \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2\right)$$

**MAP estimation** (negative log):
$$\hat{\boldsymbol{\beta}}_{\text{MAP}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2\sigma^2}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \frac{1}{2\tau^2}\sum_{j=1}^p \beta_j^2 \right\}$$

---

# Ridge regression

$$\hat{\boldsymbol{\beta}}_{\text{MAP}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2\sigma^2}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \frac{1}{2\tau^2}\sum_{j=1}^p \beta_j^2 \right\}$$

**Setting $\lambda = \sigma^2/\tau^2$** (scaling constants):
$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\}$$

A Gaussian prior is equivalent to an L2 penalty to a linear regression, also known as **Ridge regression**

---

# Ridge Regression: Matrix Perspective

**Ridge objective:**
$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_2^2 \right\}$$

**Closed-form solution** (take derivative, set to zero):
$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{Y}$$

$\lambda\mathbf{I}$ "regularizes" the covariance matrix. If $\lambda=0$, then we get the classic OLS estimator:

$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$$

---

# Ridge Regression: Properties

Ridge regression is useful to reduce overfitting and handle multicollinearity, which typically causes instability in OLS estimates and very large coefficients.

- Always invertible: $\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I}$ is positive definite even if $\mathbf{X}^T\mathbf{X}$ is singular

- Shrinks all coefficients toward zero, but doesn't set any exactly to zero

- Control via λ:
  - $\lambda = 0$ → OLS (no regularization)
  - $\lambda \to \infty$ → $\boldsymbol{\beta} \to \mathbf{0}$

If we want a regularization techinque that sets some coefficients exactly to zero (so we can remove the associated variables from the model), we need a different prior.

---

# Laplace Prior on Coefficients

**Why Laplace prior?** To encourage sparsity and perform feature selection

**Prior specification:**
$$P(\beta_j) = \frac{1}{2b}\exp\left(-\frac{|\beta_j|}{b}\right) \quad \text{independently}$$

**Properties:**

- **Sharp peak at zero** → strong preference for exactly zero
- Heavy tails → allows some large values
- Parameter $b$ controls scale

---
<center>

<img src="https://upload.wikimedia.org/wikipedia/commons/8/89/Laplace_distribution_pdf.png" width="775">

CC BY-SA 3.0, <https://commons.wikimedia.org/w/index.php?curid=75502>

</center>

---

# Comparison with Gaussian

| Prior | Shape | Effect |
|-------|-------|--------|
| Gaussian | Smooth bell | Continuous shrinkage |
| Laplace | Sharp peak | Sparsity |

**When to use:**

- Believe many coefficients are exactly zero
- Want automatic feature selection
- Need interpretable model (few features)

---

# Deriving MAP with Laplace Prior

**Start with Laplace prior:**
$$P(\boldsymbol{\beta}) = \prod_{j=1}^p \frac{1}{2b}\exp\left(-\frac{|\beta_j|}{b}\right) \propto \exp\left(-\frac{1}{b}\sum_{j=1}^p |\beta_j|\right)$$

**Gaussian likelihood** (assuming known $\sigma^2$):
$$P(\mathbf{Y}|\boldsymbol{\beta}) \propto \exp\left(-\frac{1}{2\sigma^2}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2\right)$$

**MAP estimation** (negative log):
$$\hat{\boldsymbol{\beta}}_{\text{MAP}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2\sigma^2}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \frac{1}{b}\sum_{j=1}^p |\beta_j| \right\}$$

---

# LASSO Regression

$$\hat{\boldsymbol{\beta}}_{\text{MAP}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2\sigma^2}\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \frac{1}{b}\sum_{j=1}^p |\beta_j| \right\}$$

**Setting $\lambda = 2\sigma^2/b$**:
$$\hat{\boldsymbol{\beta}}_{\text{LASSO}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \sum_{j=1}^p |\beta_j| \right\}$$

A Laplace prior is equivalent to an L1 penalty to a linear regression, also known as LASSO (Least Absolute Shrinkage and Selection Operator) regression

---

# LASSO Regression: Properties

**Objective:**
$$\hat{\boldsymbol{\beta}}_{\text{LASSO}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1 \right\}$$

**Properties:**

- No closed-form solution (requires iterative optimization methods)
- Produces sparse models
  - Sets some coefficients **exactly to zero** → automatic feature selection
- As $\lambda$ increases, more coefficients become zero
  - Least important variables get zeroed out first as $\lambda$ increases

---

# Why not both?: Laplace + Gaussian combined prior

- Ridge: Handles correlated features well, stable
- LASSO: Sparse solutions, feature selection

**Combined prior** (unnormalized):
$$P(\beta_j) \propto \underbrace{\exp\left(-\frac{\beta_j^2}{2\tau^2}\right)}_{\text{Gaussian}} \cdot \underbrace{\exp\left(-\frac{|\beta_j|}{b}\right)}_{\text{Laplace}}$$

Taking negative log like before:
$$-\log P(\boldsymbol{\beta}) \propto \frac{1}{2\tau^2}\|\boldsymbol{\beta}\|_2^2 + \frac{1}{b}\|\boldsymbol{\beta}\|_1$$

**This naturally leads to Elastic Net!**

---

# Elastic Net Regression

**Objective** (combines L1 and L2):
$$\hat{\boldsymbol{\beta}}_{\text{ElasticNet}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \lambda_2 \|\boldsymbol{\beta}\|_2^2 \right\}$$

**Alternative parameterization** (sklearn uses this):
$$\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \left(\alpha \|\boldsymbol{\beta}\|_1 + \frac{1-\alpha}{2} \|\boldsymbol{\beta}\|_2^2\right) \right\}$$

- $\alpha \in [0,1]$: mixing parameter ($\alpha=1$ → LASSO, $\alpha=0$ → Ridge)
- $\lambda > 0$: overall regularization strength

**Best of both worlds:**

- Sparsity from L1 (feature selection)
- Stability from L2 (grouped selection for correlated features)

---

# When to Use Each Method?

| Method | Prior | Use When | Benefits |
|--------|-------|----------|----------|
| **Ridge** | Gaussian | All features relevant, many correlated | Stable, closed-form |
| **LASSO** | Laplace | Many irrelevant features | Sparse, interpretable |
| **Elastic Net** | Mixture | High correlation + need sparsity | Balanced |

**Decision tree:**

1. Need sparsity/interpretability? → **LASSO or Elastic Net**
2. $p > n$ (more features than samples)? → **Ridge or Elastic Net**
3. Features highly correlated? → **Ridge or Elastic Net**
4. Few features matter? → **LASSO**
5. Uncertain? → **Elastic Net** (robust choice)

---

# Summary: Bayesian perspective unifies regularization

1. **Prior choice determines regularization type**
   - Gaussian prior → Ridge (L2 penalty)
   - Laplace prior → LASSO (L1 penalty)
   - Combined prior → Elastic Net (L1 + L2)

2. **MAP estimation = penalized optimization**
   - Stronger prior (smaller variance) = more regularization (higher $\lambda$)

3. **Each method serves different goals**
   - Ridge: stability, handles correlation
   - LASSO: sparsity, feature selection
   - Elastic Net: balanced approach

---

# References & Further Reading

**Core Textbooks:**

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
  - Chapter 3.4: Shrinkage Methods
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
  - Chapter 7.5: Ridge Regression, Chapter 13.3: Sparse Linear Models
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
  - Chapter 6: Linear Model Selection and Regularization
