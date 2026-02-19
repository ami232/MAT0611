# Mock Exam - SOLUTIONS

**Course:** MAT0611 - Mathematics for Machine Learning  
**Duration:** 120 minutes  
**Total Points:** 100 points

---

## Question 1 (20 points)

An e-commerce website tracks the number of customer orders received per hour during peak times. Over a sample of 10 randomly selected hours, the following counts were observed:

**Data:** 2, 3, 3, 3, 4, 4, 5, 5, 5, 6

### **a) Explain why a Poisson distribution is appropriate for modeling the number of customer orders per hour. (5 points)**

**Solution:**

The Poisson distribution is appropriate for this scenario because:

1. **Counting events in fixed intervals:** We are counting discrete events (customer orders) occurring in fixed time intervals (per hour).

2. **Independence:** Customer orders can be assumed to occur independently of each other - one customer placing an order doesn't directly affect whether another customer places an order.

3. **Constant average rate:** During peak times, we can assume a relatively stable average rate of orders per hour (λ).

4. **Rare events:** Individual orders are relatively rare events in the continuous time interval, and we're counting how many occur.

These characteristics match the assumptions of the Poisson distribution, making it a suitable model for count data of this nature.

---

### **b) Produce the log-likelihood function and simplify. (5 points)**

**Solution:**

The Poisson probability mass function is:
$$P(X = x \mid \lambda) = \frac{\lambda^x e^{-\lambda}}{x!}$$

For a sample of $n$ observations $x_1, x_2, \ldots, x_n$, the likelihood function is:
$$L(\lambda) = \prod_{i=1}^{n} \frac{\lambda^{x_i} e^{-\lambda}}{x_i!}$$

Taking the natural logarithm:
$$\ell(\lambda) = \log L(\lambda) = \sum_{i=1}^{n} \log\left(\frac{\lambda^{x_i} e^{-\lambda}}{x_i!}\right)$$

$$= \sum_{i=1}^{n} \left[x_i \log(\lambda) - \lambda - \log(x_i!)\right]$$

$$= \log(\lambda) \sum_{i=1}^{n} x_i - n\lambda - \sum_{i=1}^{n} \log(x_i!)$$

**Simplified log-likelihood:**
$$\boxed{\ell(\lambda) = \left(\sum_{i=1}^{n} x_i\right) \log(\lambda) - n\lambda - \sum_{i=1}^{n} \log(x_i!)}$$

---

### **c) Using your log-likelihood function, derive the MLE estimator $\hat{\lambda}_{MLE}$. (5 points)**

**Solution:**

To find the MLE, we maximize the log-likelihood by taking the derivative with respect to $\lambda$ and setting it equal to zero:

$$\frac{d\ell(\lambda)}{d\lambda} = \frac{1}{\lambda}\sum_{i=1}^{n} x_i - n = 0$$

Solving for $\lambda$:
$$\frac{\sum_{i=1}^{n} x_i}{\lambda} = n$$

$$\lambda = \frac{\sum_{i=1}^{n} x_i}{n} = \bar{x}$$

To verify this is a maximum, check the second derivative:
$$\frac{d^2\ell(\lambda)}{d\lambda^2} = -\frac{1}{\lambda^2}\sum_{i=1}^{n} x_i < 0$$

Since the second derivative is negative, this is indeed a maximum.

**MLE estimator:**
$$\boxed{\hat{\lambda}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i}$$

The MLE for the Poisson parameter is simply the sample mean.

---

### **d) Calculate $\hat{\lambda}_{MLE}$ from the sample above and use it to estimate the probability of receiving exactly 3 orders in an hour. (5 points)**

**Solution:**

First, calculate the sample mean:
$$\hat{\lambda}_{MLE} = \frac{2 + 3 + 3 + 3 + 4 + 4 + 5 + 5 + 5 + 6}{10} = \frac{40}{10} = 4$$

Now, estimate the probability of exactly 3 orders using the Poisson PMF with $\lambda = 4$:
$$P(X = 3 \mid \lambda = 4) = \frac{4^3 e^{-4}}{3!} = \frac{64 \cdot e^{-4}}{6}$$

$$= \frac{64}{6} \cdot e^{-4} = \frac{32}{3} \cdot e^{-4}$$

Computing numerically:
$$= 10.667 \times 0.01832 \approx 0.1954$$

**Answer:**
$$\boxed{\hat{\lambda}_{MLE} = 4}$$
$$\boxed{P(X = 3) \approx 0.195 \text{ or } 19.5\%}$$

---

## Question 2 (20 points)

A marketing team wants to estimate the conversion rate $p$ (probability that a website visitor makes a purchase). Based on industry experience, they use a Beta(4, 6) prior for $p$.

**Data:** Out of 20 visitors, 9 made a purchase

### **a) State Bayes' rule and explain how it is used in Bayesian inference. (5 points)**

**Solution:**

**Bayes' Rule:**
$$P(\theta \mid D) = \frac{P(D \mid \theta) \cdot P(\theta)}{P(D)}$$

Or in more intuitive notation:
$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

**Explanation in Bayesian Inference:**

- **Prior $P(\theta)$:** Represents our initial beliefs about the parameter $\theta$ before observing any data. It incorporates domain knowledge or previous experience.

- **Likelihood $P(D \mid \theta)$:** The probability of observing the data $D$ given a particular parameter value $\theta$. This comes from the statistical model.

- **Evidence $P(D)$:** The marginal probability of the data, which serves as a normalizing constant to ensure the posterior is a valid probability distribution. Often calculated as $P(D) = \int P(D \mid \theta)P(\theta)d\theta$.

- **Posterior $P(\theta \mid D)$:** Our updated beliefs about $\theta$ after observing the data. It combines prior knowledge with observed evidence.

**Usage:** Bayesian inference updates our beliefs about parameters by combining prior information with new data. The posterior can then be used for estimation, prediction, or as the prior for future analyses.

---

### **b) Use an appropriate distribution for the likelihood and calculate the posterior distribution parameters $\alpha_{\text{post}}$ and $\beta_{\text{post}}$. (10 points)**

**Solution:**

**Likelihood:**
Since we're observing binary outcomes (purchase or no purchase), the appropriate likelihood is the **Binomial distribution**:
$$P(X = x \mid n, p) = \binom{n}{x} p^x (1-p)^{n-x}$$

where $n = 20$ (number of visitors) and $x = 9$ (number of purchases).

**Prior:**
$$p \sim \text{Beta}(\alpha_0, \beta_0) = \text{Beta}(4, 6)$$

with density:
$$f(p) \propto p^{\alpha_0 - 1}(1-p)^{\beta_0 - 1} = p^{3}(1-p)^{5}$$

**Posterior Calculation:**
The Beta distribution is the conjugate prior for the Binomial likelihood, so the posterior is also Beta distributed.

Using Bayes' rule:
$$f(p \mid D) \propto P(D \mid p) \cdot f(p)$$

$$\propto p^x(1-p)^{n-x} \cdot p^{\alpha_0 - 1}(1-p)^{\beta_0 - 1}$$

$$= p^{x + \alpha_0 - 1}(1-p)^{n - x + \beta_0 - 1}$$

This is the kernel of a Beta distribution with parameters:
$$\alpha_{\text{post}} = \alpha_0 + x = 4 + 9 = 13$$
$$\beta_{\text{post}} = \beta_0 + (n - x) = 6 + (20 - 9) = 6 + 11 = 17$$

**Posterior distribution:**
$$\boxed{p \mid D \sim \text{Beta}(13, 17)}$$
$$\boxed{\alpha_{\text{post}} = 13, \quad \beta_{\text{post}} = 17}$$

---

### **c) Calculate the posterior mean and compare it to the MLE estimate $\hat{p}_{MLE}$ (the sample mean) (5 points)**

**Solution:**

**Posterior Mean:**
For a Beta distribution with parameters $\alpha$ and $\beta$, the mean is:
$$E[p \mid D] = \frac{\alpha_{\text{post}}}{\alpha_{\text{post}} + \beta_{\text{post}}} = \frac{13}{13 + 17} = \frac{13}{30} \approx 0.433$$

**MLE (Sample Mean):**
$$\hat{p}_{MLE} = \frac{x}{n} = \frac{9}{20} = 0.45$$

**Comparison:**
$$\boxed{\text{Posterior Mean} = 0.433 \quad \text{vs.} \quad \hat{p}_{MLE} = 0.450}$$

**Interpretation:**
The posterior mean (0.433) is slightly lower than the MLE (0.450). This is because the prior Beta(4, 6) has a mean of $\frac{4}{10} = 0.4$, which pulls the posterior estimate slightly below the sample proportion.

The Bayesian estimate is a weighted average between the prior belief and the observed data. With a total of 10 prior "pseudo-observations" (4 successes, 6 failures) and 20 actual observations, the data has more influence, but the prior still pulls the estimate slightly toward 0.4.

As we collect more data, the posterior mean will converge toward the MLE, since the likelihood will dominate the prior.

---

## Question 3 (20 points)

An e-commerce company claims that the average session duration on their website is 8 minutes. A data analyst tests this claim with a random sample of 30 user sessions.

**Sample statistics:**

- Sample size: $n = 30$
- Sample mean: $\bar{x} = 7.2$ minutes
- Sample standard deviation: $s = 2.1$ minutes

### **a) State and explain the type of test you should use and the null and alternative hypotheses. Is it a unilateral or bilateral test? (5 points)**

**Solution:**

**Type of Test:**
We should use a **one-sample t-test** because:

1. We're comparing a sample mean to a known population value (8 minutes)
2. The population standard deviation is unknown (we only have sample standard deviation $s$)
3. Sample size is reasonable ($n = 30$) for the t-distribution to be appropriate

**Hypotheses:**

The analyst is testing whether the average session duration differs from the company's claim of 8 minutes. Without indication that we're only interested in one direction, this is a **two-tailed (bilateral) test**.

$$H_0: \mu = 8 \text{ minutes (the company's claim is correct)}$$
$$H_1: \mu \neq 8 \text{ minutes (the average differs from 8 minutes)}$$

**Type of Test:**
$$\boxed{\text{Bilateral (two-tailed) one-sample t-test}}$$

**Explanation:** We're testing if the true mean differs from 8 in either direction (could be higher or lower), not just in one specific direction.

---

### **b) Calculate the test statistic. (5 points)**

**Solution:**

The t-test statistic for a one-sample test is:
$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

where:

- $\bar{x} = 7.2$ (sample mean)
- $\mu_0 = 8$ (hypothesized population mean)
- $s = 2.1$ (sample standard deviation)
- $n = 30$ (sample size)

Computing the standard error:
$$SE = \frac{s}{\sqrt{n}} = \frac{2.1}{\sqrt{30}} = \frac{2.1}{5.477} \approx 0.3834$$

Computing the t-statistic:
$$t = \frac{7.2 - 8}{0.3834} = \frac{-0.8}{0.3834} \approx -2.086$$

**Answer:**
$$\boxed{t \approx -2.086}$$

The negative value indicates the sample mean is below the hypothesized mean, with a magnitude of about 2.09 standard errors.

---

### **c) Make a statistical decision at $\alpha = 0.05$ and interpret your conclusion in context. (5 points)**

**Solution:**

For a two-tailed test with $\alpha = 0.05$ and degrees of freedom $df = n - 1 = 29$:

**Critical values:** $\pm t_{29, 0.025} = \pm 2.042$

**Decision rule:** Reject $H_0$ if $|t| > 2.042$

**Our test statistic:** $|t| = |-2.086| = 2.086$

**Comparison:** $2.086 > 2.042$

**Statistical Decision:**
$$\boxed{\text{Reject } H_0 \text{ at } \alpha = 0.05}$$

**Interpretation in Context:**

At the 5% significance level, we have sufficient evidence to reject the company's claim that the average session duration is 8 minutes. The sample data suggests that the true average session duration is significantly different from 8 minutes.

Specifically, the sample mean of 7.2 minutes is significantly lower than the claimed 8 minutes. This difference of 0.8 minutes is unlikely to have occurred by random chance alone (probability less than 5%).

**Practical Interpretation:** The company should investigate why actual session durations appear to be shorter than claimed. This could have implications for user engagement strategies or marketing materials.

---

### **d) Explain what a p-value is. If you were told that $p\text{-value} = 0.023$, interpret what this means specifically for this test. (5 points)**

**Solution:**

**What is a p-value:**

The p-value is the probability of observing a test statistic as extreme as, or more extreme than, the one calculated from the sample data, assuming the null hypothesis is true.

More formally:

- It measures the strength of evidence against the null hypothesis
- Smaller p-values indicate stronger evidence against $H_0$
- It is NOT the probability that $H_0$ is true

**Interpretation of p-value = 0.023 for this test:**

If the p-value is 0.023, this means:

1. **Probabilistic interpretation:** If the true average session duration were actually 8 minutes (as the company claims), there would be only a 2.3% chance of observing a sample mean as far from 8 minutes as 7.2 minutes (or farther) in a random sample of 30 sessions.

2. **Evidence strength:** The p-value of 0.023 is less than our significance level $\alpha = 0.05$, providing significant evidence against the company's claim.

3. **Decision:** Since $p = 0.023 < 0.05$, we reject $H_0$. This confirms our decision from part (c).

4. **Practical meaning:** There's strong evidence (97.7% confidence, roughly speaking) that the true average session duration differs from 8 minutes. The observed difference is unlikely to be due to random sampling variation alone.

**Note:** The p-value being 0.023 means this result would occur less than 1 in 40 times by random chance if the company's claim were true, making it a statistically significant finding.

---

## Question 4 (20 points)

An e-commerce analyst fits a simple linear regression model to predict total purchase amount (in euros) from the number of items viewed during a session, using a sample of 35 customers.

**Model:** $\text{Purchase} = \beta_0 + \beta_1 \cdot \text{ItemsViewed} + \epsilon$

### **a) State three assumptions of the linear regression model. (5 points)**

**Solution:**

The classical linear regression model relies on several key assumptions:

**1. Linearity:**
The relationship between the predictors and the response variable is linear. That is, the expected value of $Y$ is a linear function of $X$:
$$E[Y \mid X] = \beta_0 + \beta_1 X$$

**2. Independence:**
The observations (and their error terms) are independent of each other. The error $\epsilon_i$ for one observation does not depend on the error for another observation. This is often expressed as:
$$\text{Cov}(\epsilon_i, \epsilon_j) = 0 \text{ for } i \neq j$$

**3. Homoscedasticity (Constant Variance):**
The variance of the error terms is constant across all levels of the predictor variables:
$$\text{Var}(\epsilon_i) = \sigma^2 \text{ for all } i$$

The error terms should not exhibit heteroscedasticity (varying spread).

**4. Normality:**
The error terms are normally distributed:
$$\epsilon_i \sim N(0, \sigma^2)$$

This is especially important for making valid inferences (confidence intervals, hypothesis tests) in small samples.

**5. No Multicollinearity (for multiple regression):**
In multiple regression, the predictor variables should not be perfectly (or highly) correlated with each other.

**6. Exogeneity:**
The predictor variables are assumed to be uncorrelated with the error term:
$$E[\epsilon \mid X] = 0$$

---

### **b) Interpret the estimated slope coefficient $\hat{\beta}_1$. Is the relationship between items viewed and purchase amount statistically significant at $\alpha = 0.05$? Justify your answer using two different pieces of evidence from the output. (5 points)**

**Solution:**

**Interpretation of $\hat{\beta}_1 = 4.25$:**

For each additional item viewed during a session, the predicted total purchase amount increases by **€4.25**, on average, holding all else constant.

In other words, viewing one more item is associated with a €4.25 increase in the expected purchase amount.

**Statistical Significance:**

$$\boxed{\text{YES, the relationship is statistically significant at } \alpha = 0.05}$$

**Evidence 1 - p-value:**
The p-value for $\beta_1$ is $p < 0.001$, which is much smaller than $\alpha = 0.05$.
This indicates very strong evidence that the slope is different from zero, meaning there is a significant relationship between items viewed and purchase amount.

**Evidence 2 - 95% Confidence Interval:**
The 95% confidence interval for $\beta_1$ is [2.93, 5.57], which **does not contain zero**.
Since zero is not in the confidence interval, we can reject the null hypothesis $H_0: \beta_1 = 0$ at the 5% significance level.

**Alternative Evidence 2 - t-statistic:**
The t-statistic is $t = 6.54$, and the critical value is approximately $t_{33}(0.025) = 2.035$ for a two-tailed test.
Since $|6.54| > 2.035$, we reject the null hypothesis that $\beta_1 = 0$.

**Conclusion:** The relationship between items viewed and purchase amount is highly statistically significant, with very strong evidence that viewing more items leads to higher purchase amounts.

---

### **c) Calculate a 95% confidence interval for the intercept $\beta_0$ using the standard error and the critical value provided. (5 points)**

**Solution:**

The formula for a 95% confidence interval for a regression coefficient is:
$$\hat{\beta} \pm t_{\alpha/2, df} \cdot SE(\hat{\beta})$$

**Given information:**

- $\hat{\beta}_0 = 12.50$ (point estimate)
- $SE(\hat{\beta}_0) = 3.80$ (standard error)
- $t_{33}(0.025) = 2.035$ (critical value for two-tailed test)

**Calculation:**

Margin of error:
$$ME = t_{33}(0.025) \times SE(\hat{\beta}_0) = 2.035 \times 3.80 = 7.733$$

Lower bound:
$$\hat{\beta}_0 - ME = 12.50 - 7.733 = 4.767$$

Upper bound:
$$\hat{\beta}_0 + ME = 12.50 + 7.733 = 20.233$$

**95% Confidence Interval for $\beta_0$:**
$$\boxed{[4.77, 20.23] \text{ euros}}$$

**Interpretation:**
We are 95% confident that the true intercept $\beta_0$ (the expected purchase amount when zero items are viewed) lies between €4.77 and €20.23.

Note: Since this interval does not contain zero and is entirely positive, the intercept is statistically significant, suggesting there's a baseline purchase amount even with minimal browsing.

---

### **d) Suppose the analyst adds more predictors and the model starts to overfit. Explain how ridge regression can be used to address overfitting, and how the formula for the classic OLS estimator $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$ is modified in ridge regression. (5 points)**

**Solution:**

**How Ridge Regression Addresses Overfitting:**

Ridge regression is a regularization technique that adds a penalty term to the loss function to prevent overfitting. Specifically:

1. **Penalizes large coefficients:** Ridge adds an L2 penalty (sum of squared coefficients) to the ordinary least squares objective function. This discourages the model from fitting large coefficient values.

2. **Bias-variance tradeoff:** By accepting a small amount of bias (shrinking coefficients toward zero), ridge regression significantly reduces variance, leading to better prediction performance on new data.

3. **Handles multicollinearity:** When predictors are highly correlated, OLS estimates can become unstable with large magnitudes. Ridge regression stabilizes these estimates by shrinking them.

4. **Controls model complexity:** The penalty prevents the model from becoming too complex and fitting noise in the training data.

**Modified Formula:**

**OLS objective:** Minimize $||\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}||^2$

**Ridge objective:** Minimize $||\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}||^2 + \lambda ||\boldsymbol{\beta}||^2$

where $\lambda \geq 0$ is the regularization parameter that controls the strength of the penalty.

**OLS estimator:**
$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$$

**Ridge regression estimator:**
$$\boxed{\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{Y}}$$

where $\mathbf{I}$ is the identity matrix.

**Key modification:** The term $\lambda \mathbf{I}$ is added to $\mathbf{X}^T\mathbf{X}$ before inversion.

**Effect of the modification:**

- When $\lambda = 0$: Ridge equals OLS
- When $\lambda > 0$: Coefficients are shrunk toward zero
- As $\lambda \to \infty$: Coefficients approach zero
- The optimal $\lambda$ is typically chosen by cross-validation

**Additional benefit:** Adding $\lambda \mathbf{I}$ also ensures the matrix is invertible even when $\mathbf{X}^T\mathbf{X}$ is singular or nearly singular, which can occur with multicollinearity or when $p > n$.

---

## Question 5 (20 points)

An analyst wants to predict whether a website visitor will make a purchase (Y = 1) or not (Y = 0) based on their session duration and number of pages visited.

### **a) Explain why logistic regression is more appropriate than linear regression when the outcome is binary. You may discuss why the logistic function $\sigma(z) = \frac{1}{1 + e^{-z}}$ is used. (5 points)**

**Solution:**

**Why Logistic Regression is More Appropriate:**

**1. Bounded probabilities:**

- Linear regression can predict values outside [0, 1], which is nonsensical for probabilities
- Logistic regression guarantees predictions between 0 and 1 by using the logistic function
- Example: Linear regression might predict $P(Y=1) = 1.3$ or $-0.2$, which are invalid probabilities

**2. Appropriate error distribution:**

- Binary outcomes follow a Bernoulli distribution, not a normal distribution
- Linear regression assumes normally distributed errors, which is violated with binary data
- Logistic regression uses the appropriate binomial/Bernoulli likelihood

**3. Non-linear relationship:**

- The relationship between predictors and probability of success is typically non-linear for binary outcomes
- As predictors increase/decrease, the effect on probability should diminish near 0 and 1
- Linear models don't capture this S-shaped relationship

**Why the Logistic Function $\sigma(z) = \frac{1}{1 + e^{-z}}$ is Used:**

The logistic (sigmoid) function has several desirable properties:

**1. Bounds:** $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$, ensuring valid probability predictions

**2. S-shaped curve:**

- Captures the natural relationship between predictors and probability
- Small changes in $z$ near extreme values have little effect
- Changes near $z = 0$ have the largest effect

**3. Smooth and differentiable:** Allows for gradient-based optimization methods

**4. Interpretation via odds and log-odds:**
$$\log\left(\frac{p}{1-p}\right) = z = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p$$

The log-odds (logit) transform creates a linear relationship with predictors, while probabilities remain bounded.

**5. Mathematical convenience:** The derivative has a simple form: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

---

### **b) Logistic regression does not have a closed-form solution for the maximum likelihood estimators. Explain how gradient descent can be used to find the MLE. (5 points)**

**Solution:**

**Why No Closed-Form Solution:**

In logistic regression, the log-likelihood involves non-linear functions (logarithms and exponentials) of the parameters. Setting the gradient to zero does not yield a closed-form expression that can be algebraically solved for $\boldsymbol{\beta}$.

**Gradient Descent for MLE:**

Gradient descent is an iterative optimization algorithm that finds the MLE by repeatedly updating parameters in the direction that increases the log-likelihood (or decreases the negative log-likelihood).

**Algorithm:**

1. **Initialize:** Start with initial parameter values $\boldsymbol{\beta}^{(0)}$ (commonly zeros or random values)

2. **Compute gradient:** Calculate the gradient of the log-likelihood with respect to $\boldsymbol{\beta}$:

   For logistic regression, the gradient is:
   $$\nabla \ell(\boldsymbol{\beta}) = \mathbf{X}^T(\mathbf{y} - \mathbf{p})$$

   where $\mathbf{p}$ is the vector of predicted probabilities: $p_i = \sigma(\mathbf{x}_i^T\boldsymbol{\beta})$

3. **Update parameters:** Move in the direction of the gradient (since we're maximizing):
   $$\boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} + \alpha \nabla \ell(\boldsymbol{\beta}^{(k)})$$

   Or equivalently, minimize the negative log-likelihood:
   $$\boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} - \alpha \nabla(-\ell)(\boldsymbol{\beta}^{(k)})$$

   where $\alpha > 0$ is the learning rate (step size)

4. **Iterate:** Repeat steps 2-3 until convergence (when the change in parameters or log-likelihood becomes very small)

**Key Considerations:**

- **Learning rate $\alpha$:** Must be chosen carefully. Too large causes overshooting; too small causes slow convergence
- **Convergence criterion:** Stop when $||\boldsymbol{\beta}^{(k+1)} - \boldsymbol{\beta}^{(k)}|| < \epsilon$ or when log-likelihood change is negligible
- **Convexity:** The log-likelihood for logistic regression is concave (negative log-likelihood is convex), guaranteeing convergence to the global optimum

**Advantages:** Simple to implement, works for large datasets, generalizes to complex models

---

### **c) The Newton-Raphson method is commonly used to find the MLE in logistic regression using the following update formula: (10 points)**

$$\boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} + (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{y} - \mathbf{p})$$

**Identify the gradient and the Hessian in the formula, explain how it works and mention one advantage of Newton-Raphson over simple gradient descent for logistic regression.**

**Solution:**

**Identifying Components:**

**Gradient:**
$$\nabla \ell(\boldsymbol{\beta}) = \mathbf{X}^T(\mathbf{y} - \mathbf{p})$$

This is the vector of first partial derivatives of the log-likelihood with respect to each parameter $\beta_j$. It appears explicitly on the right side of the update formula.

**Hessian:**
$$\mathbf{H} = -\mathbf{X}^T\mathbf{W}\mathbf{X}$$

This is the matrix of second partial derivatives. The term $(\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}$ is the inverse of the negative Hessian. Here, $\mathbf{W}$ is a diagonal matrix with entries:
$$W_{ii} = p_i(1 - p_i) = \sigma(\mathbf{x}_i^T\boldsymbol{\beta})(1 - \sigma(\mathbf{x}_i^T\boldsymbol{\beta}))$$

**How Newton-Raphson Works:**

**1. Second-order approximation:**
Newton-Raphson uses a second-order Taylor expansion of the log-likelihood around the current point $\boldsymbol{\beta}^{(k)}$:
$$\ell(\boldsymbol{\beta}) \approx \ell(\boldsymbol{\beta}^{(k)}) + \nabla\ell^T(\boldsymbol{\beta} - \boldsymbol{\beta}^{(k)}) + \frac{1}{2}(\boldsymbol{\beta} - \boldsymbol{\beta}^{(k)})^T\mathbf{H}(\boldsymbol{\beta} - \boldsymbol{\beta}^{(k)})$$

**2. Optimization:**
Maximize this quadratic approximation with respect to $\boldsymbol{\beta}$ by taking the derivative and setting to zero:
$$\nabla\ell + \mathbf{H}(\boldsymbol{\beta} - \boldsymbol{\beta}^{(k)}) = 0$$

**3. Update:**
Solving for $\boldsymbol{\beta}$ gives:
$$\boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} - \mathbf{H}^{-1}\nabla\ell$$

Since $\mathbf{H} = -\mathbf{X}^T\mathbf{W}\mathbf{X}$ (negative), we have:
$$\boldsymbol{\beta}^{(k+1)} = \boldsymbol{\beta}^{(k)} + (\mathbf{X}^T\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{y} - \mathbf{p})$$

**Intuition:**

- The Hessian captures the curvature of the log-likelihood surface
- Newton-Raphson adaptively adjusts the step size and direction based on local curvature
- It's like gradient descent with an adaptive, data-driven learning rate

**Advantage of Newton-Raphson over Gradient Descent:**

**Faster convergence:** Newton-Raphson typically achieves **quadratic convergence** near the optimum, meaning the number of correct digits approximately doubles with each iteration. In contrast, gradient descent has linear convergence.

**Mathematically:** If $\boldsymbol{\beta}^*$ is the optimum:

- Gradient descent: $||\boldsymbol{\beta}^{(k+1)} - \boldsymbol{\beta}^*|| \approx c \cdot ||\boldsymbol{\beta}^{(k)} - \boldsymbol{\beta}^*||$ (linear)
- Newton-Raphson: $||\boldsymbol{\beta}^{(k+1)} - \boldsymbol{\beta}^*|| \approx c \cdot ||\boldsymbol{\beta}^{(k)} - \boldsymbol{\beta}^*||^2$ (quadratic)

**Practical impact:** Newton-Raphson often converges in 5-10 iterations, while gradient descent might require hundreds or thousands of iterations to achieve similar accuracy.

**Additional advantages:**

- No need to manually tune a learning rate
- Better for ill-conditioned problems
- Naturally provides standard errors for inference (from the Hessian)

**Trade-off:** Newton-Raphson requires computing and inverting the Hessian, which can be expensive for large datasets (O($p^3$) complexity). Gradient descent only requires computing the gradient (O($p$) per iteration).
