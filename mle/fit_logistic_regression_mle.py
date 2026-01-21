"""
Logistic Regression with Maximum Likelihood Estimation
Newton-Raphson Algorithm Implementation
Following the lecture slides
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n = 200
X = np.random.randn(n)  # Single predictor
z = -0.5 + 2.0 * X  # Linear combination
p = 1 / (1 + np.exp(-z))  # Logistic function
y = np.random.binomial(1, p)  # Binary outcomes

print("=" * 70)
print("Logistic Regression via Maximum Likelihood Estimation")
print("Newton-Raphson Algorithm")
print("=" * 70)
print(f"Sample size: {n}")
print(f"Positive outcomes: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Negative outcomes: {n - y.sum()} ({(1-y.mean())*100:.1f}%)")
print()


def logistic_function(z):
    """
    Compute the logistic (sigmoid) function.

    Following slides: σ(z) = 1/(1+e^(-z))

    Parameters:
    -----------
    z : array-like
        Linear combination of predictors

    Returns:
    --------
    p : array-like
        Probabilities in [0, 1]
    """
    # TODO: Implement and return the logistic (sigmoid) function
    # Hint: σ(z) = 1/(1+e^(-z))
    pass


def log_likelihood(beta, X, y):
    """
    Compute log-likelihood for logistic regression.

    Following slides: l(β) = Σ[y_i log(p_i) + (1-y_i)log(1-p_i)]

    Parameters:
    -----------
    beta : array-like, shape (2,)
        Coefficients [intercept, slope]
    X : array-like, shape (n,)
        Predictor variable
    y : array-like, shape (n,)
        Binary outcome (0 or 1)

    Returns:
    --------
    ll : float
        Log-likelihood
    """
    beta0, beta1 = beta
    z = beta0 + beta1 * X
    p = logistic_function(z)

    # Avoid log(0) by clipping probabilities
    p = np.clip(p, 1e-10, 1 - 1e-10)

    # TODO: Implement log-likelihood calculation
    # Hint: l(β) = Σ[y_i log(p_i) + (1-y_i)log(1-p_i)]
    pass

    # return log_likelihood_value


def gradient(beta, X, y):
    """
    Compute gradient (score vector) of log-likelihood.

    Following slides: ∇ℓ = X^T(y - p)

    Parameters:
    -----------
    beta : array-like, shape (2,)
        Coefficients [intercept, slope]
    X : array-like, shape (n,)
        Predictor variable
    y : array-like, shape (n,)
        Binary outcome (0 or 1)

    Returns:
    --------
    grad : array-like, shape (2,)
        Gradient vector (for log-likelihood, not negative)
    """
    # TODO: Implement and return the gradient computation
    # Hint: ∇ℓ = X^T(y - p)
    pass


def compute_weights(beta, X):
    """
    Compute weight matrix W for Newton-Raphson.

    Following slides: W = diag(p_i(1-p_i))

    Parameters:
    -----------
    beta : array-like, shape (2,)
        Coefficients [intercept, slope]
    X : array-like, shape (n,)
        Predictor variable

    Returns:
    --------
    W : array-like, shape (n,)
        Diagonal weights w_i = p_i(1-p_i)
    """
    beta0, beta1 = beta
    z = beta0 + beta1 * X
    p = logistic_function(z)

    # Weights: w_i = p_i(1-p_i)
    w = p * (1 - p)

    return w


def hessian(beta, X, y):
    """
    Compute Hessian matrix (second derivatives).

    Following slides: H = -X^T W X

    Parameters:
    -----------
    beta : array-like, shape (2,)
        Coefficients [intercept, slope]
    X : array-like, shape (n,)
        Predictor variable
    y : array-like, shape (n,)
        Binary outcome (0 or 1)

    Returns:
    --------
    H : array-like, shape (2, 2)
        Hessian matrix (negative definite)
    """
    # TODO: Implement Hessian computation
    # Hint: H = -X^T W X, use compute_weights function
    pass


def newton_raphson(X, y, beta_init, max_iter=100, tol=1e-8, verbose=True):
    """
    Newton-Raphson algorithm for logistic regression MLE.

    Algorithm from slides:
    1. Initialize β^(0) (often 0)
    2. Compute p using current β
    3. Compute W = diag(p_i(1-p_i))
    4. Update β^(k+1) = β^(k) - H^(-1)∇ℓ
    5. Check convergence: ||∇ℓ|| < ε
    6. Repeat steps 2-5

    Parameters:
    -----------
    X : array-like, shape (n,)
        Predictor variable
    y : array-like, shape (n,)
        Binary outcome (0 or 1)
    beta_init : array-like, shape (2,)
        Initial guess for coefficients
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    verbose : bool
        Print iteration details

    Returns:
    --------
    beta : array-like, shape (2,)
        Estimated coefficients
    converged : bool
        Whether algorithm converged
    iterations : int
        Number of iterations performed
    history : list
        History of [iteration, beta, log_likelihood, norm_gradient]
    """
    # TODO: Implement Newton-Raphson optimization algorithm
    # Follow the algorithm steps outlined in the docstring above
    # Use the gradient and hessian functions you implemented
    # Remember to check for convergence using gradient norm
    pass


# Step 1: Initialize β^(0) (often zeros)
beta_init = np.array([0.0, 0.0])

print("\n" + "=" * 70)
print("STEP 1: Initialize Parameters")
print("=" * 70)
print(f"β^(0) = {beta_init}")
print()

# Run Newton-Raphson algorithm
print("=" * 70)
print("STEPS 2-6: Iterative Optimization")
print("=" * 70)
print()

beta_hat, converged, n_iter, history = newton_raphson(
    X, y, beta_init, max_iter=50, tol=1e-8, verbose=True
)

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Converged: {converged}")
print(f"Number of iterations: {n_iter}")
print(f"Final log-likelihood: {log_likelihood(beta_hat, X, y):.4f}")
print(f"Final β₀ (intercept): {beta_hat[0]:.6f}")
print(f"Final β₁ (slope): {beta_hat[1]:.6f}")
print()

# Use estimated parameters for inference
beta_final = beta_hat

# Compute standard errors from Fisher Information
print("=" * 70)
print("INFERENCE: Standard Errors and Hypothesis Tests")
print("=" * 70)
print()

# Following slides: Var(β̂) = I(β̂)^(-1) = (X^T W X)^(-1)
H = hessian(beta_final, X, y)
fisher_info = -H  # I(β) = -H (for logistic regression)
cov_matrix = np.linalg.inv(fisher_info)
std_errors = np.sqrt(np.diag(cov_matrix))

print("Covariance Matrix Var(β̂):")
print(cov_matrix)
print()

# Compute z-statistics and p-values (Wald tests)
# Following slides: z_j = β̂_j / SE(β̂_j) ~ N(0,1) under H₀: β_j = 0
z_stats = beta_final / std_errors
p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))

# Display results
print("Coefficient Estimates with Hypothesis Tests:")
print("-" * 70)
print(
    f"{'Parameter':<15} {'Estimate':>10} {'Std Error':>10} {'z-value':>10} {'p-value':>10}"
)
print("-" * 70)
print(
    f"{'Intercept (β₀)':<15} {beta_final[0]:>10.4f} {std_errors[0]:>10.4f} {z_stats[0]:>10.4f} {p_values[0]:>10.4f}"
)
print(
    f"{'Slope (β₁)':<15} {beta_final[1]:>10.4f} {std_errors[1]:>10.4f} {z_stats[1]:>10.4f} {p_values[1]:>10.4f}"
)
print("-" * 70)
print()

# Compute confidence intervals
alpha = 0.05
z_crit = norm.ppf(1 - alpha / 2)
ci_lower = beta_final - z_crit * std_errors
ci_upper = beta_final + z_crit * std_errors

print("95% Confidence Intervals:")
print("-" * 70)
print(f"{'Parameter':<15} {'Lower':>10} {'Upper':>10}")
print("-" * 70)
print(f"{'Intercept (β₀)':<15} {ci_lower[0]:>10.4f} {ci_upper[0]:>10.4f}")
print(f"{'Slope (β₁)':<15} {ci_lower[1]:>10.4f} {ci_upper[1]:>10.4f}")
print("-" * 70)
print()

# Odds ratio interpretation
odds_ratio = np.exp(beta_final[1])
print("Interpretation:")
print(f"  Odds Ratio for X: exp(β₁) = {odds_ratio:.4f}")
print(f"  One unit increase in X multiplies the odds by {odds_ratio:.2f}")
print()

# Compare with statsmodels to verify our implementation
print("=" * 70)
print("VALIDATION: Comparison with Statsmodels")
print("=" * 70)
print()

X_design = sm.add_constant(X)
logit_model = sm.Logit(y, X_design).fit(disp=0)

print("Statsmodels Results:")
print(logit_model.summary())
print()

print("Parameter Comparison (Our Implementation vs Statsmodels):")
print("-" * 70)
print(f"{'Parameter':<15} {'Our Estimate':>15} {'Statsmodels':>15} {'Difference':>15}")
print("-" * 70)
print(
    f"{'Intercept (β₀)':<15} {beta_final[0]:>15.6f} {logit_model.params[0]:>15.6f} {abs(beta_final[0] - logit_model.params[0]):>15.2e}"
)
print(
    f"{'Slope (β₁)':<15} {beta_final[1]:>15.6f} {logit_model.params[1]:>15.6f} {abs(beta_final[1] - logit_model.params[1]):>15.2e}"
)
print("-" * 70)
print()

print("Standard Error Comparison:")
print("-" * 70)
print(f"{'Parameter':<15} {'Our SE':>15} {'Statsmodels SE':>15} {'Difference':>15}")
print("-" * 70)
print(
    f"{'Intercept (β₀)':<15} {std_errors[0]:>15.6f} {logit_model.bse[0]:>15.6f} {abs(std_errors[0] - logit_model.bse[0]):>15.2e}"
)
print(
    f"{'Slope (β₁)':<15} {std_errors[1]:>15.6f} {logit_model.bse[1]:>15.6f} {abs(std_errors[1] - logit_model.bse[1]):>15.2e}"
)
print("-" * 70)
print()

if np.allclose(beta_final, logit_model.params, atol=1e-6):
    print("Our implementation matches Statsmodels")
else:
    print("Warning: Differences detected between implementations")
print()

# Visualization
print("=" * 70)
print("VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data and fitted logistic curve
ax1 = axes[0, 0]
X_sorted = np.sort(X)
z_pred = beta_final[0] + beta_final[1] * X_sorted
p_pred = logistic_function(z_pred)

ax1.scatter(X, y, alpha=0.5, label="Observed data", s=30)
ax1.plot(X_sorted, p_pred, "r-", linewidth=2.5, label="Fitted curve P(Y=1|X)")
ax1.axhline(
    y=0.5,
    color="gray",
    linestyle="--",
    alpha=0.5,
    linewidth=1.5,
    label="p=0.5 boundary",
)
ax1.set_xlabel("X", fontsize=11)
ax1.set_ylabel("Probability / Outcome", fontsize=11)
ax1.set_title("Logistic Regression Fit", fontsize=12, fontweight="bold")
ax1.legend(loc="best")
ax1.grid(True, alpha=0.3)

# Plot 2: Predicted probabilities histogram
ax2 = axes[0, 1]
p_fitted = logistic_function(beta_final[0] + beta_final[1] * X)
ax2.hist(
    p_fitted[y == 0], bins=20, alpha=0.6, label="Y=0", color="blue", edgecolor="black"
)
ax2.hist(
    p_fitted[y == 1], bins=20, alpha=0.6, label="Y=1", color="red", edgecolor="black"
)
ax2.axvline(x=0.5, color="black", linestyle="--", linewidth=2, label="Threshold p=0.5")
ax2.set_xlabel("Predicted Probability", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)
ax2.set_title("Distribution of Predicted Probabilities", fontsize=12, fontweight="bold")
ax2.legend(loc="best")
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: Newton-Raphson convergence (Log-Likelihood)
ax3 = axes[1, 0]
iterations = [h[0] for h in history]
lls = [h[2] for h in history]
ax3.plot(iterations, lls, "o-", linewidth=2, markersize=7, color="darkblue")
ax3.set_xlabel("Iteration", fontsize=11)
ax3.set_ylabel("Log-Likelihood ℓ(β)", fontsize=11)
ax3.set_title("Convergence: Log-Likelihood", fontsize=12, fontweight="bold")
ax3.grid(True, alpha=0.3)

# Plot 4: Gradient norm convergence
ax4 = axes[1, 1]
grad_norms = [h[3] for h in history]
ax4.semilogy(iterations, grad_norms, "o-", linewidth=2, markersize=7, color="green")
ax4.axhline(y=1e-8, color="red", linestyle="--", linewidth=2, label="Tolerance ε=1e-8")
ax4.set_xlabel("Iteration", fontsize=11)
ax4.set_ylabel("||∇ℓ|| (log scale)", fontsize=11)
ax4.set_title("Convergence: Gradient Norm", fontsize=12, fontweight="bold")
ax4.legend(loc="best")
ax4.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("logistic_regression_mle_results.png", dpi=300, bbox_inches="tight")
print("Plots saved as 'logistic_regression_mle_results.png'")
plt.show()
