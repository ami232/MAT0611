# Assignment: Hypothesis Testing with Simulations

## Objective

Understand the behavior of hypothesis tests through simulation, including Type I error rates, p-values, and the relationship between test statistics and critical values.

## Task Description

You will implement two hypothesis testing procedures from scratch and analyze their behavior through simulation:

1. A **bilateral t-test** for the mean of a normal distribution
2. A **Kolmogorov-Smirnov (KS) test** for goodness of fit to a normal distribution

## Deliverable

Submit a **single Python script** named `hypothesis_testing_simulation.py` that completes all tasks below and includes a **detailed discussion as a comment block at the top of the file**.

---

## Part 1: Bilateral t-Test for the Mean

### Task 1.1: Single Sample Test

1. **Simulate a sample** from a standard normal distribution with sample size `n = 10`
2. **Calculate the t-statistic** for testing $H_0: \mu = 0$ vs $H_1: \mu \neq 0$
   - Use the formula: $t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$ where $s$ is the sample standard deviation
3. **Determine the critical value** for $\alpha = 0.05$ (two-tailed test)
   - Use the appropriate degrees of freedom
4. **Calculate the p-value** for your test statistic
5. **Print the results**:
   - Sample mean
   - Test statistic
   - Critical value(s)
   - p-value
   - Decision (reject or fail to reject $H_0$)

### Task 1.2: Simulation Study (1000 replications)

1. **Simulate 1000 independent samples** from a standard normal distribution, each with `n = 10`
2. For each sample, calculate the t-statistic
3. **Analyze the results**:
   - Count how many times the test statistic exceeds the critical value (in absolute value)
   - Count how many times the test statistic is more extreme than the one calculated in Task 1.1
   - Calculate the empirical Type I error rate (proportion of rejections)
4. **Create visualizations**:
   - Histogram of the 1000 test statistics
   - Overlay the theoretical t-distribution
   - Mark the critical values
   - Mark the test statistic from Task 1.1

---

## Part 2: Kolmogorov-Smirnov Test

### Task 2.1: Single Sample KS Test

1. **Use the same sample** from Task 1.1 (or generate a new one with `n = 10`)
2. **Calculate the KS statistic** for testing whether the data comes from a standard normal distribution
   - The KS statistic is: $D = \max_i |F_n(x_i) - F_0(x_i)|$
   - Where $F_n$ is the empirical CDF and $F_0$ is the theoretical CDF (standard normal)
3. **Determine the critical value** for $\alpha = 0.05$
   - Use the formula or lookup table for the KS critical value
4. **Calculate the p-value** (if possible with available functions)
5. **Print the results**:
   - KS statistic
   - Critical value
   - p-value (if calculated)
   - Decision (reject or fail to reject $H_0$)

### Task 2.2: Simulation Study (1000 replications)

1. **Simulate 1000 independent samples** from a standard normal distribution, each with `n = 10`
2. For each sample, calculate the KS statistic
3. **Analyze the results**:
   - Count how many times the KS statistic exceeds the critical value
   - Count how many times the KS statistic is more extreme than the one calculated in Task 2.1
   - Calculate the empirical Type I error rate (proportion of rejections)
4. **Create visualizations**:
   - Histogram of the 1000 KS statistics
   - Mark the critical value
   - Mark the KS statistic from Task 2.1
   - Compare with the theoretical distribution (if available)

---

## Required Discussion (Comment Block)

Your comment block must address the following points:

### 1. Implementation Explanation

- Explain how you calculated the t-statistic and KS statistic
- Describe any functions or libraries you used and why
- Document how you determined critical values

### 2. Simulation Results Analysis

For the t-test:

- What proportion of your 1000 simulations resulted in rejection of $H_0$?
- How does this compare to the theoretical Type I error rate ($\alpha = 0.05$)?
- What proportion of test statistics were more extreme than your first sample's statistic?

For the KS test:

- What proportion of your 1000 simulations resulted in rejection of $H_0$?
- How does this compare to the expected rate?
- What differences did you observe compared to the t-test?

### 3. Theoretical Understanding

- Explain what Type I error means in the context of this simulation
- Why should approximately 5% of tests reject $H_0$ when $H_0$ is true?
- What does the p-value represent? How does it relate to the test statistic and critical value?
- Discuss the difference between the t-test and KS test in terms of what they test

### 4. Critical Value vs. P-Value

- Explain the relationship between comparing the test statistic to the critical value and comparing the p-value to $\alpha$
- Which approach did you find more intuitive? Why?

### Optional Bonus: Power Analysis

- Duplicate and modify part 1 to sample from $N(0.5, 1)$ instead of $N(0, 1)$
- Calculate the empirical power (proportion of rejections when $H_0$ is false (i.e., type II errors))
- Discuss the relationship between effect size, sample size, and power

---

## Technical Requirements

### Code Structure

Your script should:

- Set a random seed for reproducibility (e.g., `np.random.seed(42)`)
- Use clear variable names
- Include docstrings for any functions you define
- Generate all required plots with proper labels, titles, and legends
- Print results in a clear, formatted manner

### Required Libraries

You may use:

- `numpy` for numerical computations and random sampling
- `scipy.stats` for statistical distributions and tests (but try to implement the core calculations yourself)
- `matplotlib` for plotting

### Suggested Functions to Use

- `np.random.normal()` for sampling
- `np.mean()`, `np.std()` for sample statistics
- `scipy.stats.t.ppf()` for critical values
- `scipy.stats.t.cdf()` for p-values
- `scipy.stats.kstest()` to verify your KS implementation (but implement the calculation yourself first)

## AI Usage Policy

**AI tools are permitted for auxiliary tasks only:**

**Allowed:**

- Understanding error messages
- Code formatting and style improvements
- Clarifying statistical concepts you've already learned
- Syntax help for plotting

**Not allowed:**

- Generating the core test statistic calculations without understanding
- Copy-pasting simulation loops without comprehension
- Having AI write your discussion/analysis

### Disclosure Requirement

**You must include an AI disclosure section in your comment block if you used AI tools.** State:

- Which AI tool(s) you used
- Specifically what tasks you used them for
- What you learned from the interaction

If no disclosure is provided, it will be assumed that AI was not used.

### Critical Reminder

**You must be able to explain every line of code and every concept in your submission.** Be prepared to:

- Walk through your simulation logic
- Explain the formulas for test statistics
- Interpret your results and visualizations
- Discuss the statistical theory behind hypothesis testing

---

## Submission Guidelines

1. Submit a single Python file: `hypothesis_testing_simulation.py`
2. The file should run without errors from start to finish
3. Include all required outputs (printed results and plots)
4. Ensure your code is well-commented and readable
5. Include the discussion comment block at the top of the file

## Grading Rubric

- **Part 1 (t-test)**: 40%
  - Single sample test (15%)
  - Simulation study (25%)
- **Part 2 (KS test)**: 40%
  - Single sample test (15%)
  - Simulation study (25%)
- **Discussion and Analysis**: 15%
- **Code Quality and Documentation**: 5%
- **Bonus Points**: Up to +10%
