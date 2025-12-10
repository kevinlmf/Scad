# scadLLA: SCAD Penalized Regression via Local Quadratic Approximation

![R-CMD-check](https://github.com/username/scadLLA/actions/workflows/R-CMD-check.yaml/badge.svg)

The `scadLLA` package implements SCAD (Smoothly Clipped Absolute Deviation) penalized regression using the Local Quadratic Approximation (LQA) algorithm. It provides both a standard implementation and an improved, numerically stable version using QR/SVD decompositions and adaptive ridge regularization for high-dimensional data.

## Features

*   **Standard LQA**: Solves SCAD via iterative weighted Ridge regression.
*   **Improved Stability**: `lqa_scad_improved` uses QR or SVD to handle multicollinearity and $p > n$ cases.
*   **Full Documentation**: Compatible with `roxygen2`.

## Installation

You can install the development version of scadLLA from [GitHub](https://github.com/) with:

```r
# install.packages("devtools")
devtools::install_github("kevinlmf/scadLLA")
```

*(Note: Replace `kevinlmf/scadLLA` with the actual repository user/name)*

## Example

Here is a basic example of how to use `lqa_scad` to fit a SCAD model:

```r
library(scadLLA)

# 1. Generate synthetic data
set.seed(123)
n <- 100
p <- 20
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(3, 1.5, 0, 0, 2, rep(0, p - 5))
y <- X %*% beta_true + rnorm(n)

# 2. Fit SCAD model
# lambda is the tuning parameter, a is the SCAD shape parameter (usually 3.7)
fit <- lqa_scad(y, X, lambda = 0.5, a = 3.7)

# 3. Print results
print(paste("Converged:", fit$converged))
print(paste("Iterations:", fit$iterations))

# Check estimated coefficients of the first few variables
print("Estimated coefficients (first 8):")
print(round(fit$beta[1:8], 3))
print("Actual coefficients (first 8):")
print(beta_true[1:8])
```

## GenAI Tutorial

This package was developed with the assistance of Generative AI tools. 
See the full **[GenAI Tutorial](GenAI_Tutorial.md)** for details on the development process, prompts used, and workflow.

## License

MIT
