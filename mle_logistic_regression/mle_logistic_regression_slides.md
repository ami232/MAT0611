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

# Fitting Logistic Regression via MLE

---

# Linear Regression Recap

## The Model

<div class="columns">
<div>

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon$$

where:

- $Y$: response variable (continuous)
- $X_1, \ldots, X_p$: predictor variables
- $\beta_0, \beta_1, \ldots, \beta_p$: coefficients (parameters)
- $\epsilon \sim N(0, \sigma^2)$: error term

</div>
<div>

![width:500px](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)

</div>
</div>

**Goal**: Estimate coefficients to predict $Y$ from $\mathbf{X}$

---

# Linear Regression: Matrix Notation

$$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

$$\mathbf{Y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}; \quad \mathbf{X} = \begin{pmatrix} 1 & x_{11} & \cdots & x_{1p} \\ 1 & x_{21} & \cdots & x_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{pmatrix}; \quad \boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{pmatrix}$$

---

# Linear Regression: Ordinary Least Squares (OLS)

## Objective

Minimize the sum of squared residuals:
$$\text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \mathbf{x}_i^T\boldsymbol{\beta})^2$$

In matrix form:
$$\text{RSS} = (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})$$

**Solution** (closed-form):
$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$$

We'll look into matrix algebra in future lectures.

---

# Logistic Regression Recap

## Binary Classification

$$P(Y = 1 | \mathbf{X}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)}} = \sigma(\beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p)$$

**Logit form**:
$$\log\left(\frac{P(Y=1|\mathbf{X})}{1-P(Y=1|\mathbf{X})}\right) = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p$$

**Interpretation**:

- Linear relationship between predictors and log-odds
- $\beta_j$: change in log-odds for one unit increase in $X_j$

---

# Logistic Regression: Estimation

## Maximum Likelihood Estimation

**Likelihood** for observations $(y_i, \mathbf{x}_i)$:
$$\begin{aligned}
L(\boldsymbol{\beta}) &= \prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i} \\
&= \prod_{i=1}^n \sigma(\mathbf{x}_i^T\boldsymbol{\beta})^{y_i}(1-\sigma(\mathbf{x}_i^T\boldsymbol{\beta}))^{1-y_i}
\end{aligned}$$

Remember $p_i$ is a function of $\boldsymbol{\beta}$:

$$p_i = P(Y=1|\mathbf{x}_i, \boldsymbol{\beta}) = \sigma(\mathbf{x}_i^T\boldsymbol{\beta}) = \frac{1}{1 + e^{-\mathbf{x}_i^T\boldsymbol{\beta}}}$$

---

# Logistic Regression: Estimation

**Log-likelihood**:
$$\begin{aligned}\ell(\boldsymbol{\beta}) &= \sum_{i=1}^n [y_i \log(p_i) + (1-y_i)\log(1-p_i)] \\
&= \sum_{i=1}^n [y_i \log(\sigma(\mathbf{x}_i^T\boldsymbol{\beta})) + (1-y_i)\log(1-\sigma(\mathbf{x}_i^T\boldsymbol{\beta}))] \\
&= \sum_{i=1}^n y_i \log{\left( \frac{1}{1 + e^{-\mathbf{x}_i^T\boldsymbol{\beta}}}\right)} + (1-y_i)\log{\left(1 - \frac{1}{1 + e^{-\mathbf{x}_i^T\boldsymbol{\beta}}}\right)} \\
\end{aligned}$$

---

# Logistic Regression: MLE Solution

**No closed-form solution!**

Use **iterative optimization**:

- Newton-Raphson method
- Iteratively Reweighted Least Squares (IRLS)
- Gradient descent variants

---
# Logistic Regression: Newton-Raphson Method

- If $\beta_0, ..., \beta_p$  maximize $\ell(\boldsymbol{\beta})$, they also make  $\nabla\ell(\boldsymbol{\beta}) = 0$.
- Newton-Raphson is an iterative algorithm to find roots (zeroes) of real-valued functions.
- Therefore, we can use it to find $\hat{\boldsymbol{\beta}}$ such that $\nabla\ell(\hat{\boldsymbol{\beta}}) = 0$, we have MLE candidates
- It starts with an initial guess (e.g., $\boldsymbol{\beta}^{(0)} = \mathbf{0}$).
- The guess is updated iteratively until $\nabla\ell(\boldsymbol{\beta}^{(k)}) \approx \mathbf{0} \Leftrightarrow \|\nabla\ell(\boldsymbol{\beta}^{(k)})\| < \epsilon$.
- The $k$-th guess update is given by:
  $$\boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} - \left[ \mathbf{H}(\boldsymbol{\beta}^{(k)})\right] ^{-1}\nabla\ell(\boldsymbol{\beta}^{(k)})$$
where:

- $\nabla\ell$: gradient (score vector)
- $\mathbf{H}$: Hessian matrix (second derivatives of log-likelihood)

---

# Logistic Regression: Gradient (Score Vector)

**Log-likelihood**:
$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^n [y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

We can use the **chain rule** to find the **gradient** (remember $p_i$ depends on $\boldsymbol{\beta}$):
$$\frac{\partial\ell}{\partial\beta_j} =  \sum_{i=1}^n \frac{\partial\ell}{\partial p_i} \cdot \frac{\partial p_i}{\partial\beta_j}$$

**Key derivatives**:

<div class="columns">
<div>

$$\frac{\partial\ell}{\partial p_i} = \frac{y_i}{p_i} - \frac{1-y_i}{1-p_i} = \frac{y_i - p_i}{p_i(1-p_i)}$$

</div>
<div>

$$\frac{\partial p_i}{\partial\beta_j} = p_i(1-p_i)x_{ij}$$

</div>
</div>

<center>

Remember that $p_i = \sigma(\mathbf{x}_i^T\boldsymbol{\beta})$ and $\frac{d\sigma(z)}{dz} = \sigma(z)(1-\sigma(z))$.

</center>

---

# Logistic Regression: Gradient (Simplified)

Combining the derivatives:
$$\frac{\partial\ell}{\partial\beta_j} = \sum_{i=1}^n \frac{y_i - p_i}{p_i(1-p_i)} \cdot p_i(1-p_i)x_{ij} = \sum_{i=1}^n (y_i - p_i)x_{ij}$$

**Gradient vector** (score):
$$\nabla\ell = \mathbf{X}^T(\mathbf{y} - \mathbf{p})$$

where:

- $\mathbf{X}$: design matrix $(n \times (p+1))$
- $\mathbf{y}$: outcome vector $(n \times 1)$
- $\mathbf{p}$: predicted probabilities $(n \times 1)$

---

# Logistic Regression: Hessian Matrix

Following a similar approach, we compute the **Hessian** (matrix of second derivatives):

$$\boldsymbol{H}_{j,k} = \frac{\partial^2\ell}{\partial\beta_j\partial\beta_k} = \sum_{i=1}^n \frac{\partial}{\partial\beta_k}[(y_i - p_i)x_{ij}]$$

Remember $y_i$ and $x_i$ don't depend on $\beta_k$, only $p_i$ does:

$$\frac{\partial^2\ell}{\partial\beta_j\partial\beta_k} = -\sum_{i=1}^n x_{ij} \cdot \frac{\partial p_i}{\partial\beta_k} = -\sum_{i=1}^n x_{ij} \cdot p_i(1-p_i)x_{ik}$$

After simplifying, we get:
$$\boldsymbol{H}_{j,k} = \frac{\partial^2\ell}{\partial\beta_j\partial\beta_k} = -\sum_{i=1}^n p_i(1-p_i)x_{ij}x_{ik}$$

---

# Logistic Regression: Hessian (Matrix Form)

We can express the Hessian in matrix form:
$$\mathbf{H} = -\mathbf{X}^T\mathbf{W}\mathbf{X}$$

where $\mathbf{W}$ is a diagonal matrix with weights:

$$\mathbf{W} = \text{diag}(w_1, w_2, \ldots, w_n), \quad w_i = p_i(1-p_i)$$

**Exercise:** Show that $\left[\mathbf{X}^T\mathbf{W}\mathbf{X}\right]_{ij} = \sum_{i=1}^n p_i(1-p_i)x_{ij}x_{ik}$$

**Properties**:

- **Negative definite**: $\mathbf{H} \preceq 0$ (log-likelihood is concave)
- **Unique maximum**: Guarantees convergence to global optimum
- **Fisher Information**: $\mathcal{I}(\boldsymbol{\beta}) = -\mathbb{E}[\mathbf{H}] = \mathbf{X}^T\mathbf{W}\mathbf{X}$

---

# Newton-Raphson Update

**Update formula**:
$$\boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} - \mathbf{H}^{-1}\nabla\ell$$

**Substituting** gradient and Hessian:
$$\boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} + (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{y} - \mathbf{p})$$

**Algorithm**:

1. Initialize $\boldsymbol{\beta}^{(0)}$ (often $\mathbf{0}$)
2. Compute $\mathbf{p}$ using current $\boldsymbol{\beta}$
3. Compute $\mathbf{W} = \text{diag}(p_i(1-p_i))$
4. Update $\boldsymbol{\beta}$
5. If convergence $\|\nabla\ell\| < \epsilon$, stop, otherwise repeat steps 2-5
