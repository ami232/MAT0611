---
marp: true
theme: default
paginate: true
math: katex
---

# Mathematics for Machine Learning

## MAD-B3-2526-S2-MAT0611

# Bayesian Statistics and Inference

---

# Statistical Paradigms

**Frequentist Statistics**

- Parameters are fixed but unknown
- Probability = long-run frequency
- Inference based on hypothetical repeated sampling
- Confidence intervals, p-values, hypothesis tests

**Bayesian Statistics**

- Parameters are random variables with distributions
- Probability = degree of belief
- Inference updates beliefs with data
- Prior → Posterior, credible intervals

---

# Bayes' Theorem (Review)

For events $A$ and $B$:

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

**In terms of parameters and data:**

$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

Where:

- $\theta$ = parameter(s) of interest
- $D$ = observed data

---

# The Bayesian Framework

$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

**Components:**

- **Prior** $P(\theta)$: Our belief about $\theta$ before seeing data
- **Likelihood** $P(D|\theta)$: Probability of data given parameter
- **Posterior** $P(\theta|D)$: Updated belief after seeing data
- **Evidence** $P(D) = \int P(D|\theta)P(\theta)d\theta$: Normalizing constant

---

# Why the Evidence Often Doesn't Matter

$$P(\theta|D) = \frac{P(D|\theta) \cdot P(\theta)}{P(D)}$$

Since $P(D)$ doesn't depend on $\theta$, we can write:

$$P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$$

**"Posterior is proportional to Likelihood times Prior"**

We can often work without calculating $P(D)$ and normalize at the end

---

# Bayesian Inference Process

1. **Choose a prior distribution** $P(\theta)$
   - Represents initial beliefs about the parameter

2. **Collect data** $D$
   - Observations from the process

3. **Specify the likelihood** $P(D|\theta)$
   - Probability model for the data

4. **Calculate the posterior** $P(\theta|D)$
   - Updated beliefs combining prior and data

5. **Make inference** from the posterior distribution

---

# Example: Coin Flipping

**Problem:** Is a coin fair? We flip it 10 times and get 7 heads.

**Prior:** Let $\theta$ = probability of heads

- We start with $\theta \sim \text{Beta}(2, 2)$ (slightly favors fair coins)

**Likelihood:** Given $\theta$, the data follows:

- $D|\theta \sim \text{Binomial}(10, \theta)$
- $P(D|\theta) = \binom{10}{7}\theta^7(1-\theta)^3$

**Posterior:** (we'll derive this next)

---

# Beta Distribution (Quick Review)

$$\text{Beta}(\alpha, \beta): \quad f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)} \quad \text{for } x \in [0,1]$$

where $B(\alpha,\beta)$ is the Beta function (normalizing constant).

**Properties:**

- Mean: $\frac{\alpha}{\alpha+\beta}$
- Mode: $\frac{\alpha-1}{\alpha+\beta-2}$ (for $\alpha, \beta > 1$)
- Flexible shapes depending on $\alpha, \beta$

**Useful for modeling probabilities**

---

# Coin Example: Posterior Calculation

**Prior:** $\theta \sim \text{Beta}(2, 2)$ so $P(\theta) \propto \theta^{2-1}(1-\theta)^{2-1} = \theta(1-\theta)$

**Likelihood:** $P(D|\theta) \propto \theta^7(1-\theta)^3$

**Posterior:**
$$P(\theta|D) \propto P(D|\theta) \cdot P(\theta) \propto \theta^7(1-\theta)^3 \cdot \theta(1-\theta)$$
$$P(\theta|D) \propto \theta^{8}(1-\theta)^{4} = \theta^{9-1}(1-\theta)^{5-1}$$

Which is proportional to the density of $\text{Beta}(9, 5)$

**Posterior:** $\theta|D \sim \text{Beta}(9, 5)$

---

# Coin Example: Visualization

Prior: $\text{Beta}(2, 2)$ → broad, centered at 0.5
Data: 7 heads out of 10 flips
Posterior: $\text{Beta}(9, 5)$ → narrower, centered around 0.64

The posterior:

- Pulls toward the data (7/10 = 0.70)
- But is influenced by the prior (which favored 0.5)
- Has less uncertainty than the prior (more peaked)

**The data updated our beliefs**

---

# Conjugate Priors

A prior is **conjugate** to a likelihood if the posterior has the same distributional form as the prior.

**Benefits:**

- Analytical solutions (no numerical integration)
- Easy interpretation
- Efficient computation

---

# Common Conjugate Pairs

|Likelihood|Parameter|Prior (and Posterior)|
|---|---|---|
|Binomial|$p$|Beta|
|Normal|$\mu$|Normal|
|Normal|$\sigma^2$|Inverse-Gamma|
|Poisson|$\lambda$|Gamma|
|Exponential|$\lambda$|Gamma|
|Categorical|$\mathbf{p}$|Dirichlet|

[More examples of conjugate priors (Wikipedia)](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions)

---

# Conjugate Prior: Beta-Binomial

**Setup:**

- Data: $X \sim \text{Binomial}(n, \theta)$ (observe $x$ successes)
- Prior: $\theta \sim \text{Beta}(\alpha, \beta)$

**Posterior:**
$$\theta|X \sim \text{Beta}(\alpha + x, \beta + n - x)$$

**Interpretation:**

- $\alpha$ = prior "pseudo-successes"
- $\beta$ = prior "pseudo-failures"
- Data adds $x$ successes and $n-x$ failures

---

# Conjugate Prior: Normal-Normal (Known Variance)

**Setup:**

- Data: $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$ with $\sigma^2$ known
- Prior: $\mu \sim N(\mu_0, \sigma_0^2)$

**Posterior:**
$$\mu|X \sim N(\mu_n, \sigma_n^2)$$

Where:

$$\mu_n = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}\left(\frac{\mu_0}{\sigma_0^2} + \frac{\sum_{i=1}^n x_i}{\sigma^2}\right),
\sigma_n^2 = \left(\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}\right)^{-1}$$

Weighted average of prior mean and sample mean

---

# Prior Selection

**Types of Priors:**

1. **Informative Prior:** Strong beliefs based on previous knowledge
   - Example: $\theta \sim \text{Beta}(20, 20)$ for a coin

2. **Weakly Informative Prior:** Mild constraints, allows data to dominate
   - Example: $\theta \sim \text{Beta}(2, 2)$

3. **Non-informative (Flat) Prior:** Minimal prior knowledge
   - Example: $\theta \sim \text{Beta}(1, 1) = \text{Uniform}(0, 1)$

---

# Impact of Prior Choice

**Weak Data + Strong Prior:** Posterior ≈ Prior

**Weak Data + Weak Prior:** Posterior still has high uncertainty

**Strong Data + Strong Prior:** Battle between prior and data

**Strong Data + Weak Prior:** Posterior ≈ Likelihood

**As $n \to \infty$, the prior influence vanishes** (with proper priors)

---

# Bayesian Point Estimates

From the posterior $P(\theta|D)$, we can extract point estimates:

1. **Posterior Mean:** $E[\theta|D]$
   - Minimizes squared error loss

2. **Posterior Median:** 50th percentile
   - Minimizes absolute error loss

3. **MAP (Maximum A Posteriori):** Mode of posterior
   - Most probable value
   - $\arg\max_\theta P(\theta|D)$

---

# Credible Intervals

A **95% credible interval** is an interval $[a, b]$ such that:

$$P(a \leq \theta \leq b | D) = 0.95$$

**Interpretation:** Given the data, there's a 95% probability that $\theta$ lies in $[a, b]$.

**Types:**

- **Equal-tailed:** 2.5% in each tail
- **Highest Posterior Density (HPD):** Shortest interval with 95% probability

---

# Credible vs Confidence Intervals

**Frequentist Confidence Interval (95%):**

- "In repeated sampling, 95% of such intervals contain the true parameter"
- The interval is random; the parameter is fixed
- Cannot say "95% probability $\theta$ is in this interval"

**Bayesian Credible Interval (95%):**

- "Given this data, 95% probability $\theta$ is in this interval"
- The parameter is random; the interval is fixed (given data)
- Direct probability statement about the parameter

---

# Example: Normal Mean with Known Variance

**Problem:** Estimate mean IQ $\mu$ from a sample.

- Known: $\sigma = 15$ (population SD)
- Data: $n = 25$ students, $\bar{x} = 108$
- Prior: $\mu \sim N(100, 10^2)$ (mild belief around 100)

**Posterior:**
$$\sigma_n^2 = \frac{1}{\frac{1}{100} + \frac{25}{225}} = \frac{1}{0.01 + 0.111} = 8.26$$

$$\mu_n = \frac{\frac{100}{100} + \frac{25 \cdot 108}{225}}{\frac{1}{100} + \frac{25}{225}} = \frac{1 + 12}{0.121} = 107.4$$

**Posterior:** $\mu|D \sim N(107.4, 8.26)$

---

# Example: IQ Inference

**Posterior:** $\mu|D \sim N(107.4, 8.26)$, so $SD = 2.87$

**Point Estimate:** $E[\mu|D] = 107.4$

**95% Credible Interval:**
$$107.4 \pm 1.96 \times 2.87 = [101.8, 113.0]$$

**Interpretation:** Given our data and prior beliefs, there's a 95% probability that the true mean IQ is between 101.8 and 113.0.

---

# Bayesian Hypothesis Testing

**Question:** Is a parameter equal to a specific value?

**Bayesian Approach:** Calculate posterior probability directly

**Example:** Is the coin fair? ($\theta = 0.5$?)

With posterior $\theta|D \sim \text{Beta}(9, 5)$:

- Calculate $P(\theta > 0.5 | D)$
- Calculate $P(0.45 < \theta < 0.55 | D)$ (region around fair)

No p-values, no significance levels, just probability

---

# Bayes Factor

**Compare two hypotheses:** $H_1$ vs $H_0$

**Bayes Factor:**
$$BF = \frac{P(D|H_1)}{P(D|H_0)} = \frac{P(H_1|D)/P(H_1)}{P(H_0|D)/P(H_0)}$$

**Interpretation:**

- $BF > 1$: Data favors $H_1$
- $BF < 1$: Data favors $H_0$
- $BF = 3$ to 10: Moderate evidence
- $BF > 10$: Strong evidence
- $BF > 100$: Decisive evidence

---

# Bayesian Prediction

**Posterior Predictive Distribution:** Predict new data $\tilde{X}$

$$P(\tilde{X}|D) = \int P(\tilde{X}|\theta) P(\theta|D) d\theta$$

**Interpretation:**

- Average predictions over all possible $\theta$ values
- Weighted by posterior probability $P(\theta|D)$
- Accounts for parameter uncertainty more directly than point estimates

---

# Example: Posterior Prediction

**Coin with** $\theta|D \sim \text{Beta}(9, 5)$

**Question:** What's the probability of heads on the next flip?

$$P(\tilde{X} = 1 | D) = \int_0^1 \theta \cdot P(\theta|D) d\theta = E[\theta|D]$$

For Beta$(9, 5)$: $E[\theta|D] = \frac{9}{9+5} = \frac{9}{14} \approx 0.643$

**Prediction:** About 64.3% chance of heads on next flip.

---

# Comparing Paradigms

**Problem:** Estimate coin bias from 7 heads in 10 flips

**Frequentist:**

- Point estimate: $\hat{\theta} = 0.7$
- 95% CI: $[0.40, 0.90]$ (approximate)
- "If repeated many times, 95% of such intervals contain true $\theta$"

**Bayesian (with Beta(2,2) prior):**

- Posterior: $\text{Beta}(9, 5)$
- Point estimate: $E[\theta|D] = 0.64$
- 95% credible: $[0.38, 0.86]$
- "95% probability $\theta \in [0.38, 0.86]$ given the data"

---

# When to Use Bayesian Methods?

**Bayesian methods excel when:**

- You have meaningful prior information
- Sample sizes are small
- You need probability statements about parameters
- Complex hierarchical or multilevel structures
- Sequential data collection (update as data arrives)
- Need to quantify uncertainty in predictions

**Both approaches are valuable and valid; choose based on the problem**

---

# Example: Medical Diagnosis

**Scenario:** Test for a rare disease (1% prevalence)

- Test sensitivity: 95% (true positive rate)
- Test specificity: 90% (true negative rate)
- You test positive. What's the probability you have the disease?

**Frequentist:** Sensitivity and specificity only

**Bayesian:** Use Bayes' theorem with prevalence as prior

---

# Medical Diagnosis Solution

Let $D$ = has disease, $T$ = tests positive

- $P(D) = 0.01$ (prior/prevalence)
- $P(T|D) = 0.95$ (true positive rate/sensitivity/recall)
- $P(T|no\ D) = 0.10$ (false positive rate/1 - specificity)

**Bayes' Theorem:**
$$\begin{aligned} P(D|T) &= \frac{P(T|D) \cdot P(D)}{P(T|D) \cdot P(D) + P(T|no\ D) \cdot P(no\ D)} \\
&= \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} = \frac{0.0095}{0.1085} \approx 0.0876\end{aligned}$$

**Only 8.76% chance you have the disease despite testing positive**

---

# Hierarchical Bayesian Models

**Structure:** Parameters have hyperparameters with their own priors

**Example:** Students in multiple schools

- Student scores: $y_{ij} \sim N(\mu_j, \sigma^2)$
- School means: $\mu_j \sim N(\mu, \tau^2)$
- Hyperprior: $\mu \sim N(m, s^2)$

**Benefits:**

- Share information across groups
- Partial pooling (balance between no pooling and complete pooling)
- Better estimates for small groups

---

# Bayesian Computation

**When posteriors are complex (no conjugacy):**

**Markov Chain Monte Carlo (MCMC):**

- Gibbs Sampling
- Metropolis-Hastings
- Hamiltonian Monte Carlo (HMC)

**Software Tools:**

- **Stan:** Modern probabilistic programming language
- **PyMC:** Python library for Bayesian modeling
- **JAGS:** Just Another Gibbs Sampler
- **INLA:** Fast approximate inference
