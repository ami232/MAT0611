# Assignment: Logistic Regression with Gradient Descent

## Objective

Implement a logistic regression classifier using **gradient descent** optimization instead of the Newton-Raphson method covered in class.

## Task Description

Starting from [mle_logistic_regression_exercises.py](mle_logistic_regression_exercises.py), create a new Python script that replaces the Newton-Raphson optimization algorithm with gradient descent while maintaining the same maximum likelihood estimation framework.

## Deliverable

Submit a **single Python script** named `mle_logistic_regression_gradient_descent.py` that:

1. Implements gradient descent optimization for logistic regression MLE
2. Includes a **detailed discussion as a comment block at the top of the file**
3. Produces working code that successfully trains a logistic regression model

## Required Discussion (Comment Block)

Your comment block must address the following points:

### 1. Implementation Explanation

- Describe your gradient descent algorithm
- Explain key differences from Newton-Raphson in terms of update rules
- Document your choice of stopping criteria

### 2. Convergence Comparison

- Compare convergence behavior between gradient descent and Newton-Raphson
- Discuss convergence speed (number of iterations required)
- Analyze convergence stability

### 3. Learning Rate Analysis

- Explain how different learning rates impact convergence
- Document what happens with learning rates that are too small or too large
- Justify your final choice of learning rate(s)

### 4. Algorithmic Differences

- Computational complexity per iteration
- Memory requirements
- When to prefer gradient descent vs. Newton-Raphson

**Points awarded for:**

1. **Matrix Operations** (+30%)
   - Implement using vectorized matrix operations whenever possible
   - Minimize explicit loops over data points
   - Demonstrate computational efficiency

2. **Generalizable Code** (+20%)
   - Design your script to handle arbitrary numbers of input variables (regressors)
   - Generate a synthetic datasets with more than one predictor run the algorithm

**Important:** A working implementation with loops is preferred over a non-working matrix-based implementation. Get it working first, then optimize.

## Technical Requirements

- Your script must run without errors
- Produce convergence plots similar to the Newton-Raphson version (you may reuse plotting code)
- Include comparison of final parameter estimates with `statsmodels` or the solved Newton-Raphson script
- Use clear variable names and include docstrings for functions

## AI Usage Policy

**AI tools are permitted for auxiliary tasks only:**

**Allowed:**

- Code formatting and style improvements
- Interpreting error messages and stack traces
- Proofreading comments and documentation
- Syntax clarification

**Not allowed:**

- Generating the core algorithm implementation without understanding it
- Copy-pasting code you cannot explain

### Disclosure Requirement

**You must include an AI disclosure section in your comment block if you used AI tools.** State:

- Which AI tool(s) you used
- Specifically what tasks you used them for
- What you learned from the interaction

If no disclosure is provided, it will be assumed that AI was not used.

### Critical Reminder

**You must be able to explain every line of code and every concept in your submission.** If questioned about your work, you should be able to:

- Walk through your algorithm step by step
- Justify your implementation decisions
- Explain the mathematical foundations
- Discuss trade-offs and alternatives

Submissions that cannot be explained by their author may be subject to further review.

## Submission Format

- Single Python file: `mle_logistic_regression_gradient_descent.py` uploaded to Blackboard
- Comment block at the top with required discussion
- Clean, readable, well-documented code
- Test with the same synthetic data generation as in class

## Evaluation Criteria

| Criterion | Weight |
| --------- | ------ |
| Code: working implementation and code quality | 30% |
| Discussion: quality of analysis and depth | 20% |
| Matrix operations | 30% |
| Generalization to more variables | 20% |
