# Scad: Stabilized SCAD Penalized Regression for High-Dimensional Factor Models


The **Scad** package implements SCAD (Smoothly Clipped Absolute Deviation) penalized regression for **high-dimensional estimation**, focused on **factor-based portfolio construction**. It includes both classical **Local Quadratic Approximation (LQA)** and an enhanced **Stabilized LQA** using QR/SVD solvers, sparse warm starts, and hard-thresholding. The package also features **Dynamic SCAD** for time-varying coefficient regression with temporal smoothness constraints.

---
## Introduction and Motivation

### The Optimization Problem

**Mean-Variance Portfolio Optimization** (Markowitz, 1952) aims to find portfolio weights that maximize expected return while minimizing risk:

$$\max_w \quad w^T \mu - \frac{\gamma}{2} w^T \Sigma w$$

subject to:
- $\sum_{i=1}^n w_i = 1$ (fully invested)
- $w_i \geq 0$ for all $i$ (long-only constraint)

where:
- $w \in \mathbb{R}^n$: Portfolio weights
- $\mu \in \mathbb{R}^n$: Expected returns
- $\Sigma \in \mathbb{R}^{n \times n}$: Covariance matrix
- $\gamma > 0$: Risk aversion parameter

The optimal solution is:

$$w^* = \frac{\Sigma^{-1} \mu}{1^T \Sigma^{-1} \mu}$$
---
### Factor Model Approach

Directly estimating the full mean vector $\mu$ and covariance matrix $\Sigma$ from asset returns can be noisy and unstable—especially when the number of assets $n$ is large compared to the available sample size.

To obtain more reliable inputs for portfolio optimization, we adopt a **factor model**:

$$r_{i,t} = \sum_{j=1}^p \beta_{i,j} F_{j,t} + \varepsilon_{i,t}$$

where:
- $F_{j,t}$: Common factor returns (e.g., market, size, value factors)
- $\beta_{i,j}$: Asset-specific factor loadings (sensitivities)
- $\varepsilon_{i,t}$: Idiosyncratic noise (asset-specific risk)

The factor model greatly improves estimation stability by explaining most return variation through a small number of systematic factors ($p \ll n$).

This leads to more robust estimates of the key portfolio inputs:

**Expected returns:**

$$\hat{\mu}_i = \sum_{j=1}^p \hat{\beta}_{i,j} E[F_j]$$

**Covariance matrix:**

$$\hat{\Sigma} = B F B^T + D$$

where:
- $B \in \mathbb{R}^{n \times p}$: Loading matrix (each row is an asset's factor loadings)
- $F \in \mathbb{R}^{p \times p}$: Factor covariance matrix
- $D \in \mathbb{R}^{n \times n}$: Diagonal matrix of idiosyncratic variances

By decomposing returns into **systematic versus idiosyncratic** components, the factor model yields:
- More stable mean estimates
- Lower-dimensional covariance structures ($p \ll n$)
- Better-conditioned inputs for Markowitz portfolio optimization

---
## Why SCAD

Accurate portfolio modeling depends on estimating:

- **Factor loadings** ($\boldsymbol{\beta}$)
- **Covariance matrices** ($\boldsymbol{\Sigma}$)
- **Expected returns** ($\boldsymbol{\mu}$)

OLS and LASSO struggle when predictors are highly correlated or when **$p \geq n$**. SCAD reduces bias and improves feature recovery, though classical LQA can be unstable under collinearity or high dimensionality.

---



SCAD is well-suited for finance because it provides:

- **Low bias** for strong coefficients
- **Oracle variable selection** property (Fan & Li, 2001)
- **Better interpretability** in factor models


---
## Installation

You can install the development version from [GitHub](https://github.com/) with:

```r
# install.packages("devtools")
devtools::install_github("kevinlmf/Scad")
```

Or install from source:

```r
# Build the package first
# R CMD build .

# Then install
install.packages("Scad_0.1.0.tar.gz", repos = NULL, type = "source")
```


---


## Available Functions

| Function | Description |
|----------|-------------|
| `lqa_scad()` | Standard LQA algorithm for SCAD |
| `lqa_scad_improved()` | Stabilized LQA with QR/SVD decomposition |
| `dynamic_scad()` | Dynamic SCAD for time-varying coefficients |

---
### Rcpp Compilation

The package includes C++ code for performance-critical functions. After installation, Rcpp functions are automatically available. To verify:

```r
library(Scad)

# Check if Rcpp functions are available
exists("solve_dynamic_lqa_admm_cpp", mode = "function")  # Should return TRUE
exists("generate_X_cpp", mode = "function")              # Should return TRUE
```

If Rcpp functions are not available, or if you've modified C++ source files, recompile using one of the following methods:

#### Method 1: Using R CMD (Recommended for source installation)

```bash

# Step 1: Compile Rcpp attributes
Rscript -e 'Rcpp::compileAttributes()'

# Step 2: Build the package
R CMD build .

# Step 3: Install the package
R CMD INSTALL Scad_0.1.0.tar.gz
```

#### Method 2: Using devtools (Easier, recommended)

```r
# From R console, navigate to package directory or use full path
devtools::install("path/to/Scad", reload = TRUE)
```

This automatically handles Rcpp compilation and installation.


## Rcpp Performance Acceleration

The package includes C++ implementations for performance-critical functions, providing **5-20x speedup** compared to pure R implementations.

### Available Rcpp Functions

- **Data Generation**:
  - `generate_X_cpp(n, p, rho)` - Generate correlated design matrix
  - `generate_epsilon_cpp(n, sigma, rho_eps)` - Generate correlated errors
  - `generate_time_series_data_cpp(...)` - Generate time-varying data

- **Optimization**:
  - `solve_dynamic_lqa_admm_cpp(...)` - ADMM solver for Dynamic SCAD (10-20x faster)
  - `soft_threshold_cpp(x, lambda)` - Soft thresholding operator

- **Evaluation Metrics**:
  - `compute_beta_error_cpp(beta_est, beta_true)` - Beta estimation error
  - `compute_prediction_mse_cpp(...)` - Prediction MSE
  - `compute_covariance_error_cpp(...)` - Covariance estimation error
  - `compute_mv_return_cpp(...)` - Mean-Variance portfolio returns





---


## Citation

If you use this package in your research, please cite:

```bibtex
@software{Scad,
  title = {Scad: SCAD Penalized Regression with LQA and Dynamic Models},
  author = {Long, Mengfan},
  year = {2025},
  url = {https://github.com/kevinlmf/Scad}
}
```
---

## References

- Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized likelihood and its oracle properties. *Journal of the American Statistical Association*, 96(456), 1348-1360.
- Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77-91.

---

## GenAI Tutorial

This package was developed with the assistance of Generative AI tools. 
See the full **[GenAI Tutorial](AI%20tutorial/GenAI_Tutorial.md)** for details on the development process, p**_)**

You may also refer to the accompanying PDF:  
📄 **[Biostats 615 Notes (PDF)](AI%20tutorial/Biostats_615.pdf)**

---
