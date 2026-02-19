
# Mock Exam

**Course:** MAT0611 - Mathematics for Machine Learning  
**Duration:** 120 minutes  
**Total Points:** 100 points  
**Instructions:**

- Show all your work. Partial credit will be awarded.
- You may use a calculator.
- Distribution formulas and critical values are provided where needed.
- Answer all questions clearly and completely.

---

## Question 1 (20 points)

**Poisson Distribution** $X \sim \text{Poisson}(\lambda)$

$$P(X = x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0, 1, 2, \ldots$$

$$E[X] = \lambda, \quad \text{Var}(X) = \lambda$$

An e-commerce website tracks the number of customer orders received per hour during peak times. Over a sample of 10 randomly selected hours, the following counts were observed:

**Data:** 2, 3, 3, 3, 4, 4, 5, 5, 5, 6

**a)** Explain why a Poisson distribution is appropriate for modeling the number of customer orders per hour. **(5 points)**

**b)** Produce the log-likelihood function and simplify. **(5 points)**

**c)** Using your log-likelihood function, derive the MLE estimator $\hat{\lambda}_{MLE}$. **(5 points)**

**d)** Calculate $\hat{\lambda}_{MLE}$ from the sample above and use it to estimate the probability of receiving exactly 3 orders in an hour. **(5 points)**

---

## Question 2 (20 points)

**Binomial Distribution** $X \sim \text{Bin}(n, p)$

$$P(X = x) = \binom{n}{x}p^x(1-p)^{n-x}, \quad x = 0, 1, \ldots, n$$

$$E[X] = np, \quad \text{Var}(X) = np(1-p)$$

**Beta Distribution** $X \sim \text{Beta}(\alpha, \beta)$

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0, 1]$$

$$E[X] = \frac{\alpha}{\alpha + \beta}, \quad \text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function.

A marketing team wants to estimate the conversion rate $p$ (probability that a website visitor makes a purchase). Based on industry experience, they use a Beta(4, 6) prior for $p$.

**Data:** Out of 20 visitors, 9 made a purchase

**a)** State Bayes' rule and explain how it is used in Bayesian inference. **(5 points)**

**b)** Use an appropriate distribution for the likelihood and calculate the posterior distribution parameters $\alpha_{\text{post}}$ and $\beta_{\text{post}}$. **(10 points)**

**c)** Calculate the posterior mean and compare it to the MLE estimate $\hat{p}_{MLE}$ (the sample mean) **(5 points)**

---

## Question 3 (20 points)

An e-commerce company claims that the average session duration on their website is 8 minutes. A data analyst tests this claim with a random sample of 30 user sessions.

**Sample statistics:**

- Sample size: $n = 30$
- Sample mean: $\bar{x} = 7.2$ minutes
- Sample standard deviation: $s = 2.1$ minutes

**Distribution quantiles you might need:**

- $t_{30; 0.025} = -2.045$
- $t_{30; 0.05} = -1.699$
- $t_{29; 0.025} = -2.042$
- $t_{29; 0.05} = -1.697$

**a)** State and explain the type of test you should use and the null and alternative hypotheses. Is it a unilateral or bilateral test? **(5 points)**

**b)** Calculate the test statistic. **(5 points)**

**c)** Make a statistical decision at $\alpha = 0.05$ and interpret your conclusion in context. **(5 points)**

**d)** Explain what a p-value is. If you were told that $p\text{-value} = 0.023$, interpret what this means specifically for this test. **(5 points)**

---

## Question 4 (20 points)

An e-commerce analyst fits a simple linear regression model to predict total purchase amount (in euros) from the number of items viewed during a session, using a sample of 35 customers. The model is: $\text{Purchase} = \beta_0 + \beta_1 \cdot \text{ItemsViewed} + \epsilon$

**Regression output:**

```text
Coefficient estimates:
β₀ (Intercept)     = 12.50,  SE = 3.80,  t = 3.29,  p = 0.002
β₁ (ItemsViewed)   = 4.25,   SE = 0.65,  t = 6.54,  p < 0.001

R² = 0.564
Residual standard error: 8.5 on 33 degrees of freedom
95% CI for β₁: [2.93, 5.57]
```

**Critical value:** $t_{33}(0.025) = -2.035$

**a)** State three assumptions of the linear regression model. **(5 points)**

**b)** Interpret the estimated slope coefficient $\hat{\beta}_1$. Is the relationship between items viewed and purchase amount statistically significant at $\alpha = 0.05$? Justify your answer using two different pieces of evidence from the output. **(5 points)**

**c)** Calculate a 95% confidence interval for the intercept $\beta_0$ using the  standard error and the critical value provided. **(5 points)**

**d)** Suppose the analyst adds more predictors and the model starts to overfit. Explain how ridge regression can be used to address overfitting, and how the formula for the classic OLS estimator $ \hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y} $ is modified in ridge regression. **(5 points)**

---

## Question 5 (20 points)

An analyst wants to predict whether a website visitor will make a purchase (Y = 1) or not (Y = 0) based on their session duration and number of pages visited.

**a)** Explain why logistic regression is more appropriate than linear regression when the outcome is binary. You may discuss why the logistic function $ \sigma(z) = \frac{1}{1 + e^{-z}}$ is used. **(5 points)**

**b)** Logistic regression does not have a closed-form solution for the maximum likelihood estimators. Explain how gradient descent can be used to find the MLE. **(5 points)**

**c)** The Newton-Raphson method is commonly used to find the MLE in logistic regression using the following update formula:
$$\boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} + (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{y} - \mathbf{p})$$
Identify the gradient and the Hessian in the formula, explain how it works and mention one advantage of Newton-Raphson over simple gradient descent for logistic regression. **(10 points)**

---

## Distribution Reference Sheet

### Discrete Distributions

**Bernoulli Distribution** $X \sim \text{Ber}(p)$

$$P(X = x) = p^x(1-p)^{1-x}, \quad x \in \{0, 1\}$$

$$E[X] = p, \quad \text{Var}(X) = p(1-p)$$

**Binomial Distribution** $X \sim \text{Bin}(n, p)$

$$P(X = x) = \binom{n}{x}p^x(1-p)^{n-x}, \quad x = 0, 1, \ldots, n$$

$$E[X] = np, \quad \text{Var}(X) = np(1-p)$$

**Poisson Distribution** $X \sim \text{Poisson}(\lambda)$

$$P(X = x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0, 1, 2, \ldots$$

$$E[X] = \lambda, \quad \text{Var}(X) = \lambda$$

### Continuous Distributions

**Normal (Gaussian) Distribution** $X \sim N(\mu, \sigma^2)$

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \quad x \in \mathbb{R}$$

$$E[X] = \mu, \quad \text{Var}(X) = \sigma^2$$

**Exponential Distribution** $X \sim \text{Exp}(\lambda)$

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

$$E[X] = \frac{1}{\lambda}, \quad \text{Var}(X) = \frac{1}{\lambda^2}$$

**Beta Distribution** $X \sim \text{Beta}(\alpha, \beta)$

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0, 1]$$

$$E[X] = \frac{\alpha}{\alpha + \beta}, \quad \text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function.

**Laplace Distribution** $X \sim \text{Laplace}(\mu, b)$

$$f(x) = \frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right), \quad x \in \mathbb{R}$$

$$E[X] = \mu, \quad \text{Var}(X) = 2b^2$$

where $\mu$ is the location parameter and $b > 0$ is the scale parameter.

**Gamma Distribution** $X \sim \text{Gamma}(\alpha, \beta)$

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, \quad x > 0$$

$$E[X] = \frac{\alpha}{\beta}, \quad \text{Var}(X) = \frac{\alpha}{\beta^2}$$

where $\alpha > 0$ is the shape parameter and $\beta > 0$ is the rate parameter.

**Note:** The Chi-Square distribution $\chi^2_n$ is a special case of the Gamma distribution with shape parameter $\alpha = n/2$ and rate parameter $\beta = 1/2$, i.e., $\chi^2_n = \text{Gamma}(n/2, 1/2)$.
