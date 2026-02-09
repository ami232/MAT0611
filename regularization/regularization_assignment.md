# Assignment: Polynomial Ridge Regression and Bayesian Inference

## Objective

Implement Ridge regression for polynomial features using two approaches: scikit-learn's RidgeCV (for finding the optimal regularization parameter) and a closed-form matrix solution. Optionally, explore a Bayesian approach using PyMC for bonus credit.

## Task Description

You will generate synthetic data with a non-linear relationship, expand the features using polynomial terms, and solve the regression problem using Ridge regression. The main focus is understanding how regularization works and implementing the matrix-based solution.

## Deliverable

Submit a **single Python script** named `polynomial_ridge_regression.py` that:

1. Generates the required synthetic dataset
2. Implements RidgeCV and the closed-form matrix solution
3. Compares the two methods
4. Includes a **discussion as a comment block at the top of the file**
5. Produces at least one visualization showing results

---

## Data Generation Requirements

Generate a synthetic dataset with the following specifications:

1. **Sample size**: 1000 observations
2. **Predictors**:
   - `X1`: uniformly distributed in the interval (0, 1)
   - `X2`: uniformly distributed in the interval (0, 1)
3. **Response variable**:
   - $Y = e^{2X_1} + e^{X_2} - 2e^{X_1 \cdot X_2} + \epsilon$
   - Where $\epsilon \sim N(0, 1)$ (standard normal distribution)
4. **Feature expansion**:
   - Use `sklearn.preprocessing.PolynomialFeatures` to create polynomial features up to **degree 5**
   - This will generate interaction terms and polynomial combinations

**Important**: Set a random seed for reproducibility (e.g., `np.random.seed(42)`)

---

## Required Implementations

### Part 1: Ridge Regression with Cross-Validation (30%)

1. Use `sklearn.linear_model.RidgeCV` to find the optimal regularization parameter $\lambda$
2. Fit the model on the polynomial features
3. Print and record:
   - Optimal $\lambda$ value
   - Training R² score and MSE
   - First 5 coefficient estimates

**Hint:** Use `alphas=np.logspace(-3, 3, 100)` to test a range of lambda values

### Part 2: Closed-Form Ridge Regression Solution (50%)

Using the optimal $\lambda$ from Part 1:

1. **Implement the closed-form solution** using matrix operations:
   $$\hat{\beta}_{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T y$$

2. **Requirements**:
   - Use NumPy matrix operations: `@` for matrix multiplication, `np.linalg.solve()` or `np.linalg.inv()`
   - No loops over observations or features
   - Calculate predictions: `y_pred = X_poly @ beta_ridge`
   - Calculate MSE and R² score

3. **Verify** that your matrix-based solution produces coefficients very close to `RidgeCV`
   - Print the maximum difference between coefficients
   - Differences should be < 0.01 (if larger, check your implementation)

**Hint:** Use `np.linalg.solve(A, b)` instead of `np.linalg.inv(A) @ b` for better numerical stability

Your comment block should address the following points (can be brief, 1-2 paragraphs total is fine):

### 1. Feature Engineering (2-3 sentences)

- How many features do you get from PolynomialFeatures with degree 5 and 2 input variables?
- Why do we need regularization when using high-degree polynomials?

### 2. Cross-Validation Results (3-4 sentences)

- What optimal $\lambda$ did RidgeCV find?
- Is it closer to 0 or to a large value? What does this tell you?
- What were your final R² and MSE values?

### 3. Matrix Implementation (3-4 sentences)

- Did your matrix solution match sklearn's RidgeCV? (Report max difference)
- Which numpy functions did you use for the matrix operations?
- Were there any challenges in implementing this?

### 4. Comparison and Visualization (2-3 sentences)

- Do both methods give similar predictions?
- What does your visualization show?
- Which method would you prefer to use in practice and why?

---

## Technical Requirements

### Code Structure

Your script must:

- Set a random seed at the beginning for reproducibility
- Use clear, descriptive variable names
- Include docstrings for any functions you define
- Produce well-labeled plots with titles and legends
- Print results in a clear, organized format

### Required Libraries

You should use:

- `numpy` for numerical computations and matrix operations
- `sklearn.preprocessing.PolynomialFeatures` for feature expansion
- `sklearn.linear_model.RidgeCV` for cross-validated ridge regression
- `sklearn.metrics` for model evaluation
- `pymc` for Bayesian inference
- `matplotlib` or `seaborn` for visualization

### Expected Outputs

Your script should produce:

1. **Printed output**:
   - Number of features after polynomial expansion
   - Optimal λ from cross-validation
   - Performance metrics (MSE, R²) for both methods
   - First 5 coefficient estimates from both methods
   - Maximum difference between sklearn and matrix solution coefficients

2. **Visualization** (at least one):
   - Scatter plot of true vs. predicted values, OR
   - Bar plot comparing first 10 coefficients from both methods, OR
   - Residual plot

   (Feel free to create multiple plots if you want! They will help you understand the results better.)

---

## Optional: Bayesian Approach with PyMC (Bonus +20%)

**Only attempt this after completing Parts 1 and 2 successfully!**

Implement a Bayesian Ridge regression using PyMC:

1. **Define the model**:
   - Likelihood: `y_obs = pm.Normal('y_obs', mu=X @ beta, sigma=sigma, observed=y)`
   - Prior: `beta = pm.Normal('beta', mu=0, sigma=tau, shape=p)`
   - Prior: `sigma = pm.HalfNormal('sigma', sigma=2)`

2. **Sample from posterior**:
   - Use `pm.sample(draws=1000, tune=500, chains=2)` (keep it simple)
   - Extract posterior means: `trace.posterior['beta'].mean(dim=['chain', 'draw'])`

3. **Compare with Ridge**:
   - Do the posterior means match your Ridge estimates?
   - Create a simple plot comparing coefficients

4. **Brief discussion** (3-4 sentences):
   - What value of `tau` (prior std) did you choose and why?
   - How do the Bayesian estimates compare to Ridge?
   - What advantage does the Bayesian approach give you?

---

## AI Usage Policy

**AI tools are permitted for auxiliary tasks only:**

**Allowed:**

- Code formatting and style improvements
- Understanding error messages and debugging syntax errors
- Clarifying documentation for libraries (sklearn, PyMC)
- Proofreading comments and discussion
- Help with plotting syntax

**Not allowed:**

- Generating the core algorithm implementations without understanding them
- Copy-pasting matrix operations code you cannot explain
- Having AI write your statistical analysis or discussion
- Using AI to implement the closed-form solution without understanding the math

### Disclosure Requirement

**You must include an AI disclosure section in your comment block if you used AI tools.** State:

- Which AI tool(s) you used
- Specifically what tasks you used them for
- What you learned from the interaction

If no disclosure is provided, it will be assumed that AI was not used.

### Critical Reminder

**You must be able to explain every line of code and every concept in your submission.** If questioned about your work, you should be able to:

- Explain the closed-form Ridge regression formula
- Walk through your matrix operations step by step
- Explain why regularization helps prevent overfitting
- Discuss how the two methods (RidgeCV and matrix solution) compare
- If you did the bonus: explain the basic idea behind the Bayesian approach

Submissions that cannot be explained by their author may be subject to further review.

---

## Submission Format

- Single Python file: `regularization_assignment.py` uploaded to Blackboard
- Comment block at the top with discussion
- Code should run from start to finish without errors
- Use clear variable names and include comments explaining key steps

---

## Evaluation Criteria

| Criterion | Weight |
| --------- | ------ |
| **Part 1**: RidgeCV implementation | 30% |
| **Part 2**: Closed-form matrix solution | 50% |
| **Discussion**: Answers to required questions | 15% |
| **Code quality & visualization**: Clean code, comments, plots | 5% |
| **Bonus**: Bayesian PyMC implementation | Up to +20% |
