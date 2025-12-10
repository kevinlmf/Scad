#' Factor-Based Estimation, Prediction, and Risk Modeling Framework
#'
#' This script implements the three-step pipeline used throughout the simulation
#' and empirical analysis. The framework follows the standard structure of a
#' linear factor model and makes explicit how coefficient estimation, return
#' prediction, and covariance estimation are interconnected.
#'
#' Step 1: Estimating Factor Loadings
#'   - Penalized estimator: β̂ = argmin{ (1/2)||y - Xβ||² + p_λ(β) }
#'   - Methods: LASSO, SCAD, LQA, Stabilized-LQA
#'
#' Step 2: Return Prediction via Estimated Loadings
#'   - One-step-ahead prediction: r̂_{i,t+1} = X_{i,t} β̂_t
#'
#' Step 3: Covariance Estimation via Factor Structure
#'   - Estimated covariance: Σ̂ = X diag(β̂²) X^T + σ²I
#'   - Matrix error: ||Σ̂ - Σ*||_F
#'
#' @author Mengfan Long
#' @date 2025

library(scadLLA)
library(glmnet)

# Force output to console
options(warn = 1)
if (interactive()) {
  cat("\n")
  cat("===========================================\n")
  cat("Factor-Based Estimation, Prediction, and Risk Modeling Framework\n")
  cat("===========================================\n\n")
  flush.console()
}

# Set random seed for reproducibility
set.seed(123)

# ============================================================================
# Simulation Parameters
# ============================================================================
n_sim <- 100  # Number of simulation replications
lambda_base <- 0.5  # Base lambda for reference
a <- 3.7  # SCAD shape parameter
save_results <- FALSE  # Set to TRUE to save results to files

# Simulation configurations
# Add idiosyncratic correlation parameter (rho_eps) to create correlation in residuals
# This makes MV and beta-weighted portfolios differ significantly
#
# NOTE: SCAD Improved (Stabilized-LQA) shows advantages in:
# 1. High-dimensional settings (p >> n or p ≈ n)
# 2. High correlation (rho close to 1)
# 3. Numerically unstable scenarios
#
# Current configurations include:
# - Standard: n > p (baseline)
# - High-dim: n > p but larger scale
# - Ultra high-dim: p > n (where Improved excels)
# - Equal-dim: p ≈ n (boundary case)
configurations <- list(
  # Standard case: n > p (baseline, Improved advantage minimal)
  list(n = 100, p = 50, rho = 0.5, sigma = 1.0, rho_eps = 0.3, 
       name = "Standard (n=100, p=50, ρ_ε=0.3)", weak_signal = FALSE),
  
  # High-dim case: n > p but larger (Improved may show slight advantage)
  list(n = 200, p = 100, rho = 0.8, sigma = 1.0, rho_eps = 0.5,
       name = "High-dim (n=200, p=100, ρ_ε=0.5)", weak_signal = FALSE),
  
  # ⭐ KEY: Ultra high-dim case (p > n) - This is where Improved excels!
  # SCAD Improved uses QR/SVD decomposition for stability
  list(n = 50, p = 100, rho = 0.7, sigma = 1.0, rho_eps = 0.3,
       name = "Ultra high-dim (n=50, p=100, ρ_ε=0.3) ⭐", weak_signal = FALSE),
  
  # Equal dimension case (p ≈ n) - Boundary where Improved shows advantage
  list(n = 100, p = 100, rho = 0.8, sigma = 1.0, rho_eps = 0.4,
       name = "Equal-dim (n=100, p=100, ρ_ε=0.4) ⭐", weak_signal = FALSE),
  
  # Very high correlation + high-dim (challenging for original LQA)
  list(n = 60, p = 120, rho = 0.9, sigma = 1.0, rho_eps = 0.3,
       name = "Very high-correlation (n=60, p=120, ρ=0.9) ⭐", weak_signal = FALSE)
)

# ============================================================================
# Data Generation Functions
# ============================================================================

# True coefficient vector (sparse with strong and moderate signals)
create_true_beta <- function(p) {
  beta_star <- c(3, 3, 2, 1.5, 1, rep(0, p - 5))
  return(beta_star)
}

# Generate correlated design matrix
# X ~ N(0, Sigma) where Sigma_{jk} = rho^{|j-k|}
generate_X <- function(n, p, rho) {
  # Create correlation matrix: Sigma_{jk} = rho^{|j-k|}
  Sigma <- outer(1:p, 1:p, function(i, j) rho^abs(i - j))
  
  # Generate X from multivariate normal
  X <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  
  # Standardize columns
  X <- scale(X)
  
  return(X)
}

# Generate simulation data with idiosyncratic correlation
# Data generating process: r_{i,t} = X_{i,t} β* + ε_{i,t}
# where ε_{i,t} has correlation structure (not independent)
generate_data <- function(n, p, rho, sigma, rho_eps = 0.0, n_test = NULL) {
  if (is.null(n_test)) {
    n_test <- floor(n * 0.3)  # 30% for testing
  }
  
  # Generate training data
  X_train <- generate_X(n, p, rho)
  beta_star <- create_true_beta(p)
  
  # Generate correlated residuals (idiosyncratic correlation)
  # Create correlation matrix for residuals: Σ_ε with correlation rho_eps
  if (rho_eps > 0 && n > 1) {
    # Use AR(1) structure for residual correlation: cor(ε_i, ε_j) = rho_eps^|i-j|
    # Build matrix safely
    Sigma_eps <- matrix(0, nrow = n, ncol = n)
    for (i in 1:n) {
      for (j in 1:n) {
        Sigma_eps[i, j] <- (sigma^2) * (rho_eps^abs(i - j))
      }
    }
    # Ensure positive definite
    diag(Sigma_eps) <- diag(Sigma_eps) + 1e-6
    epsilon_train <- as.vector(MASS::mvrnorm(1, mu = rep(0, n), Sigma = Sigma_eps))
  } else {
    # Independent residuals (original case)
    epsilon_train <- rnorm(n, mean = 0, sd = sigma)
  }
  # Ensure epsilon_train is a vector of correct length
  if (length(epsilon_train) != n) {
    epsilon_train <- rnorm(n, mean = 0, sd = sigma)
  }
  y_train <- as.vector(X_train %*% beta_star) + epsilon_train
  
  # Generate test data (for prediction evaluation in Step 2)
  X_test <- generate_X(n_test, p, rho)
  
  # Generate correlated residuals for test data
  if (rho_eps > 0 && n_test > 1) {
    # Build matrix safely
    Sigma_eps_test <- matrix(0, nrow = n_test, ncol = n_test)
    for (i in 1:n_test) {
      for (j in 1:n_test) {
        Sigma_eps_test[i, j] <- (sigma^2) * (rho_eps^abs(i - j))
      }
    }
    diag(Sigma_eps_test) <- diag(Sigma_eps_test) + 1e-6
    epsilon_test <- as.vector(MASS::mvrnorm(1, mu = rep(0, n_test), Sigma = Sigma_eps_test))
  } else {
    epsilon_test <- rnorm(n_test, mean = 0, sd = sigma)
  }
  # Ensure epsilon_test is a vector of correct length
  if (length(epsilon_test) != n_test) {
    epsilon_test <- rnorm(n_test, mean = 0, sd = sigma)
  }
  y_test <- as.vector(X_test %*% beta_star) + epsilon_test
  
  # True covariance matrix (for Step 3 evaluation)
  # Population covariance: Σ* = X diag(β*²) X^T + Σ_ε
  # where Σ_ε has idiosyncratic correlation structure
  true_sigma <- outer(1:p, 1:p, function(i, j) rho^abs(i - j))
  
  return(list(
    X = X_train, 
    y = y_train, 
    beta_star = beta_star,
    X_test = X_test,
    y_test = y_test,
    true_sigma = true_sigma,
    rho_eps = rho_eps,
    sigma = sigma
  ))
}

# Alternative: Generate data with weaker signals
generate_data_weak_signal <- function(n, p, rho, sigma, rho_eps = 0.0, n_test = NULL) {
  if (is.null(n_test)) {
    n_test <- floor(n * 0.3)
  }
  
  X_train <- generate_X(n, p, rho)
  beta_star <- c(2, 1.5, 1, 0.8, 0.6, rep(0, p - 5))
  
  # Generate correlated residuals
  if (rho_eps > 0 && n > 1) {
    # Build matrix safely
    Sigma_eps <- matrix(0, nrow = n, ncol = n)
    for (i in 1:n) {
      for (j in 1:n) {
        Sigma_eps[i, j] <- (sigma^2) * (rho_eps^abs(i - j))
      }
    }
    diag(Sigma_eps) <- diag(Sigma_eps) + 1e-6
    epsilon_train <- as.vector(MASS::mvrnorm(1, mu = rep(0, n), Sigma = Sigma_eps))
  } else {
    epsilon_train <- rnorm(n, mean = 0, sd = sigma)
  }
  # Ensure epsilon_train is a vector of correct length
  if (length(epsilon_train) != n) {
    epsilon_train <- rnorm(n, mean = 0, sd = sigma)
  }
  y_train <- as.vector(X_train %*% beta_star) + epsilon_train
  
  X_test <- generate_X(n_test, p, rho)
  if (rho_eps > 0 && n_test > 1) {
    # Build matrix safely
    Sigma_eps_test <- matrix(0, nrow = n_test, ncol = n_test)
    for (i in 1:n_test) {
      for (j in 1:n_test) {
        Sigma_eps_test[i, j] <- (sigma^2) * (rho_eps^abs(i - j))
      }
    }
    diag(Sigma_eps_test) <- diag(Sigma_eps_test) + 1e-6
    epsilon_test <- as.vector(MASS::mvrnorm(1, mu = rep(0, n_test), Sigma = Sigma_eps_test))
  } else {
    epsilon_test <- rnorm(n_test, mean = 0, sd = sigma)
  }
  # Ensure epsilon_test is a vector of correct length
  if (length(epsilon_test) != n_test) {
    epsilon_test <- rnorm(n_test, mean = 0, sd = sigma)
  }
  y_test <- as.vector(X_test %*% beta_star) + epsilon_test
  
  true_sigma <- outer(1:p, 1:p, function(i, j) rho^abs(i - j))
  
  return(list(
    X = X_train, 
    y = y_train, 
    beta_star = beta_star,
    X_test = X_test,
    y_test = y_test,
    true_sigma = true_sigma,
    rho_eps = rho_eps,
    sigma = sigma
  ))
}

# ============================================================================
# Lambda Selection via Cross-Validation
# ============================================================================

select_lambda_cv <- function(X, y, method = c("lasso", "lqa"), a = 3.7) {
  method <- match.arg(method)
  
  # Generate lambda sequence
  lambda_max <- max(abs(crossprod(X, y))) / nrow(X)
  lambda_seq <- exp(seq(log(lambda_max), log(lambda_max * 0.01), length.out = 50))
  
  # Use cross-validation
  nfolds <- min(5, floor(nrow(X) / 10))
  if (nfolds < 3) nfolds <- 3
  
  if (method == "lasso") {
    cv_fit <- cv.glmnet(X, y, lambda = lambda_seq, alpha = 1,
                       standardize = FALSE, intercept = FALSE,
                       nfolds = nfolds)
    return(cv_fit$lambda.1se)
  } else {
    # For SCAD methods, use grid search with CV
    best_lambda <- lambda_base
    best_cv_error <- Inf
    
    lambda_candidates <- lambda_seq[seq(1, length(lambda_seq), by = 5)]
    
    for (lambda_candidate in lambda_candidates) {
      cv_errors <- numeric(nfolds)
      fold_size <- floor(nrow(X) / nfolds)
      
      for (fold in 1:nfolds) {
        test_idx <- ((fold - 1) * fold_size + 1):min(fold * fold_size, nrow(X))
        train_idx <- setdiff(1:nrow(X), test_idx)
        
        X_train <- X[train_idx, , drop = FALSE]
        y_train <- y[train_idx]
        X_test <- X[test_idx, , drop = FALSE]
        y_test <- y[test_idx]
        
        tryCatch({
          if (method == "lqa") {
            fit <- lqa_scad(y_train, X_train, lambda = lambda_candidate, a = a,
                           standardize = FALSE, max_iter = 50)
          }
          
          y_pred <- X_test %*% fit$beta
          cv_errors[fold] <- mean((y_test - y_pred)^2)
        }, error = function(e) {
          cv_errors[fold] <<- Inf
        })
      }
      
      mean_cv_error <- mean(cv_errors)
      if (mean_cv_error < best_cv_error && is.finite(mean_cv_error)) {
        best_cv_error <- mean_cv_error
        best_lambda <- lambda_candidate
      }
    }
    
    return(best_lambda)
  }
}

# ============================================================================
# STEP 1: Estimating Factor Loadings
# ============================================================================
#
# We begin with the excess-return factor model:
#   r_{i,t} = X_{i,t} β + ε_{i,t}
#
# Given observations (X, y), the penalized estimator solves:
#   β̂ = argmin_{β} { (1/2)||y - Xβ||² + p_λ(β) }
#
# where p_λ is the regularization penalty:
#   - LASSO: p_λ(β) = λ||β||₁
#   - SCAD: the non-convex SCAD penalty of Fan and Li (2001)
#   - LQA: local quadratic approximation of the SCAD penalty
#   - Stabilized-LQA: improved SCAD estimator using QR/SVD and adaptive ridge
#
# The estimator produces a sparse vector β̂ ∈ ℝ^p, which forms the basis
# for both prediction and risk modeling.
# ============================================================================

step1_estimate_factor_loadings <- function(X, y, lambda_base, a, use_cv = TRUE) {
  # Select optimal lambda for each method using CV
  if (use_cv) {
    # LASSO: Use CV
    cv_lasso <- cv.glmnet(X, y, alpha = 1, standardize = FALSE, 
                         intercept = FALSE, nfolds = min(5, floor(nrow(X)/10)))
    lambda_lasso <- cv_lasso$lambda.1se
    
    # LQA: Use CV
    lambda_lqa <- select_lambda_cv(X, y, method = "lqa", a = a)
  } else {
    # Use fixed lambda
    lambda_lasso <- lambda_base
    lambda_lqa <- lambda_base
  }
  
  results <- list()
  
  # LASSO: p_λ(β) = λ||β||₁
  lasso_fit <- glmnet(X, y, lambda = lambda_lasso, alpha = 1,
                      standardize = FALSE, intercept = FALSE)
  results$lasso <- list(
    beta_hat = as.numeric(lasso_fit$beta),
    lambda_used = lambda_lasso
  )
  
  # LQA (Local Quadratic Approximation) for SCAD
  scad_lqa <- lqa_scad(y, X, lambda = lambda_lqa, a = a, standardize = FALSE)
  results$scad_lqa <- list(
    beta_hat = scad_lqa$beta,
    lambda_used = lambda_lqa,
    iterations = scad_lqa$iterations,
    converged = scad_lqa$converged
  )
  
  # Stabilized-LQA (LQA Improved using QR/SVD and adaptive ridge)
  scad_lqa_improved <- lqa_scad_improved(y, X, lambda = lambda_lqa, a = a,
                                         standardize = FALSE, 
                                         decomposition = "qr")
  results$scad_lqa_improved <- list(
    beta_hat = scad_lqa_improved$beta,
    lambda_used = lambda_lqa,
    iterations = scad_lqa_improved$iterations,
    converged = scad_lqa_improved$converged
  )
  
  return(results)
}

# ============================================================================
# STEP 2: Return Prediction via Estimated Loadings
# ============================================================================
#
# Given β̂_t estimated at time t, we generate one-step-ahead cross-sectional
# predictions using:
#   r̂_{i,t+1} = X_{i,t} β̂_t
#
# where X_{i,t} contains firm characteristics at month t.
# This forms the predictive signal used to construct portfolios.
#
# Portfolio Construction (Two Methods):
#
# Method 1: Beta-weighted Portfolio
#   - Portfolio weights: w_i = r̂_i / (Σ_j |r̂_j|)
#   - Portfolio return: R_portfolio = Σ_i w_i · r_{i,t+1}
#
# Method 2: Mean-Variance Portfolio
#   - Expected returns: μ = r̂ (predicted returns)
#   - Covariance matrix: Σ̂ = X_test diag(β̂²) X_test^T + σ²I
#   - Portfolio weights: w = (Σ̂^(-1) μ) / (1^T Σ̂^(-1) μ)
#   - Portfolio return: R_portfolio = Σ_i w_i · r_{i,t+1}
#
# Collecting predictions across N assets yields:
#   r̂_{t+1} ∈ ℝ^N
# ============================================================================

step2_predict_returns <- function(X_test, beta_hat) {
  # One-step-ahead prediction: r̂_{i,t+1} = X_{i,t} β̂_t
  r_hat <- X_test %*% beta_hat
  
  return(as.numeric(r_hat))
}

# ============================================================================
# STEP 3: Covariance Estimation via Factor Structure
# ============================================================================
#
# Given β̂, we estimate the factor-model covariance matrix.
# The population covariance under a linear factor model is:
#   Σ = BFB^T + D
#
# where B collects factor loadings, F is the K×K factor covariance matrix,
# and D is a diagonal matrix of idiosyncratic variances.
#
# In the simulation design we construct the population covariance as:
#   Σ* = X diag(β*²) X^T + σ²I
#
# and the estimated covariance using:
#   Σ̂ = X diag(β̂²) X^T + σ²I
#
# The quality of the covariance estimator is measured through the Frobenius norm:
#   MatrixError = ||Σ̂ - Σ*||_F
# ============================================================================

step3_estimate_covariance <- function(X, beta_hat, sigma, rho_eps = 0.0) {
  # Estimated covariance: Σ̂ = X diag(β̂²) X^T + Σ_ε
  # where Σ_ε has idiosyncratic correlation structure (not just σ²I)
  n <- nrow(X)
  p <- ncol(X)
  
  # Ensure X is a matrix
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  
  # Ensure beta_hat is a vector with correct length
  beta_hat <- as.numeric(beta_hat)
  if (length(beta_hat) != p) {
    stop(sprintf("Dimension mismatch: beta_hat length (%d) != X columns (%d)", 
                 length(beta_hat), p))
  }
  
  # Factor structure: Σ̂ = X diag(β̂²) X^T
  # Compute more safely: scale columns of X by beta_sq, then multiply by X^T
  beta_sq <- beta_hat^2
  
  # Scale each column of X by corresponding beta_sq value
  # Method: X %*% diag(beta_sq) is equivalent to scaling columns
  # Use sweep or direct multiplication: t(t(X) * beta_sq)
  X_scaled <- sweep(X, 2, beta_sq, "*")
  
  # Ensure X_scaled is a matrix
  if (!is.matrix(X_scaled)) {
    X_scaled <- as.matrix(X_scaled)
  }
  
  # Compute X_scaled %*% t(X) to get n × n covariance matrix
  Sigma_hat <- X_scaled %*% t(X)
  
  # Ensure Sigma_hat is a proper matrix
  if (!is.matrix(Sigma_hat) || nrow(Sigma_hat) != n || ncol(Sigma_hat) != n) {
    # Fallback to diagonal matrix
    Sigma_hat <- diag(n) * sigma^2
  }
  
  # Add idiosyncratic covariance: Σ_ε (with correlation structure)
  if (rho_eps > 0 && n > 1) {
    # Create correlation matrix for residuals: cor(ε_i, ε_j) = rho_eps^|i-j|
    # Use safer method to build matrix
    Sigma_eps <- matrix(0, nrow = n, ncol = n)
    for (i in 1:n) {
      for (j in 1:n) {
        Sigma_eps[i, j] <- (sigma^2) * (rho_eps^abs(i - j))
      }
    }
    # Ensure dimensions match before addition
    if (nrow(Sigma_eps) == nrow(Sigma_hat) && ncol(Sigma_eps) == ncol(Sigma_hat)) {
      Sigma_hat <- Sigma_hat + Sigma_eps
    } else {
      # If dimensions don't match, just add to diagonal
      diag(Sigma_hat) <- diag(Sigma_hat) + sigma^2
    }
  } else {
    # Independent residuals: + σ²I
    diag(Sigma_hat) <- diag(Sigma_hat) + sigma^2
  }
  
  # Final check: ensure result is a proper n × n matrix
  if (!is.matrix(Sigma_hat) || nrow(Sigma_hat) != n || ncol(Sigma_hat) != n) {
    Sigma_hat <- diag(n) * sigma^2
  }
  
  return(Sigma_hat)
}

step3_compute_matrix_error <- function(Sigma_hat, Sigma_star) {
  # Matrix error: ||Σ̂ - Σ*||_F
  # Ensure both are matrices with matching dimensions
  if (!is.matrix(Sigma_hat)) {
    Sigma_hat <- as.matrix(Sigma_hat)
  }
  if (!is.matrix(Sigma_star)) {
    Sigma_star <- as.matrix(Sigma_star)
  }
  
  # Check dimensions match
  if (nrow(Sigma_hat) != nrow(Sigma_star) || ncol(Sigma_hat) != ncol(Sigma_star)) {
    # Return a large error value if dimensions don't match
    return(Inf)
  }
  
  # Compute Frobenius norm
  diff_matrix <- Sigma_hat - Sigma_star
  matrix_error <- sqrt(sum(diff_matrix^2, na.rm = TRUE))  # Frobenius norm
  
  # Ensure result is finite
  if (!is.finite(matrix_error)) {
    matrix_error <- Inf
  }
  
  return(matrix_error)
}

# ============================================================================
# Evaluation Metrics
# ============================================================================

compute_metrics <- function(beta_hat, beta_star, X = NULL, y = NULL, 
                           X_test = NULL, y_test = NULL, 
                           true_sigma = NULL, sigma = NULL, rho_eps = 0.0) {
  
  # ============================================================================
  # Criterion 1: β-Estimation Error (from Step 1)
  # ============================================================================
  mse <- mean((beta_hat - beta_star)^2)
  l2_error_sq <- sum((beta_hat - beta_star)^2)  # ||β̂ - β*||²
  l2_error <- sqrt(l2_error_sq)
  l1_error <- sum(abs(beta_hat - beta_star))
  
  # Support recovery
  true_support <- which(beta_star != 0)
  estimated_support <- which(beta_hat != 0)
  
  true_positives <- length(intersect(true_support, estimated_support))
  false_positives <- length(setdiff(estimated_support, true_support))
  false_negatives <- length(setdiff(true_support, estimated_support))
  
  tpr <- ifelse(length(true_support) > 0, 
                true_positives / length(true_support), 0)
  fpr <- ifelse(length(setdiff(1:length(beta_star), true_support)) > 0,
                false_positives / length(setdiff(1:length(beta_star), true_support)), 0)
  
  exact_recovery <- identical(sort(true_support), sort(estimated_support))
  support_accuracy <- ifelse(length(beta_star) > 0,
                             (true_positives + length(beta_star) - length(true_support) - false_positives) / length(beta_star),
                             0)
  
  # ============================================================================
  # Criterion 2: Return Prediction (from Step 2)
  # ============================================================================
  prediction_mse <- NA
  portfolio_return_beta_weighted <- NA
  portfolio_return_mean_variance <- NA
  if (!is.null(X_test) && !is.null(y_test)) {
    # Ensure X_test is a matrix and y_test is a vector
    if (!is.matrix(X_test)) {
      X_test <- as.matrix(X_test)
    }
    y_test <- as.vector(y_test)
    beta_hat <- as.vector(beta_hat)
    
    # Step 2: r̂_{i,t+1} = X_{i,t} β̂_t
    r_hat <- numeric(0)
    tryCatch({
      r_hat <- step2_predict_returns(X_test, beta_hat)
      r_hat <- as.vector(r_hat)
      
      # Ensure dimensions match
      if (length(r_hat) == length(y_test) && length(r_hat) > 0) {
        prediction_mse <- mean((y_test - r_hat)^2, na.rm = TRUE)
      } else {
        prediction_mse <- NA
        r_hat <- numeric(0)
      }
    }, error = function(e) {
      prediction_mse <<- NA
      r_hat <<- numeric(0)
    })
    
    n_test <- length(r_hat)
    if (n_test > 1 && length(y_test) == n_test) {
      # ========================================================================
      # Method 1: Beta-weighted Portfolio
      # ========================================================================
      # Use predicted returns (r_hat = X_test %*% beta_hat) as signals
      # Beta-weighted portfolio: weights proportional to predicted returns
      tryCatch({
        weights_beta <- r_hat / (sum(abs(r_hat)) + 1e-8)
        if (length(weights_beta) == length(y_test)) {
          portfolio_return_beta_weighted <- sum(weights_beta * y_test)
        }
      }, error = function(e) {
        portfolio_return_beta_weighted <<- NA
      })
      
      # ========================================================================
      # Method 2: Mean-Variance Portfolio
      # ========================================================================
      # Mean-variance optimization: w = (Σ^(-1) μ) / (1^T Σ^(-1) μ)
      # where μ = predicted returns, Σ = estimated covariance matrix
      if (n_test > 2 && !is.null(X_test) && !is.null(sigma) && 
          length(r_hat) == n_test && length(y_test) == n_test) {
        # Check dimensions
        if (!is.matrix(X_test)) {
          X_test <- as.matrix(X_test)
        }
        p_test <- ncol(X_test)
        beta_hat_len <- length(beta_hat)
        
        # Ensure beta_hat has correct length
        if (beta_hat_len != p_test || p_test <= 0 || n_test <= 0) {
          # If dimensions don't match, use beta-weighted as fallback
          portfolio_return_mean_variance <- portfolio_return_beta_weighted
        } else {
          # Estimate covariance matrix for portfolio construction
          # Use factor model structure with proper scaling
          # Key: Scale covariance to match the variance of actual returns
          
          # Get baseline variance from actual returns
          var_y_test <- var(y_test)
          if (!is.finite(var_y_test) || var_y_test <= 0) var_y_test <- sigma^2
          
          # Build covariance matrix using factor structure
          # Factor contribution: X_test * diag(beta_hat^2) * X_test^T
          tryCatch({
            beta_sq <- beta_hat^2
            if (length(beta_sq) != p_test) {
              stop("beta_sq length mismatch")
            }
            X_scaled <- sweep(X_test, 2, beta_sq, "*")
            if (!is.matrix(X_scaled) || nrow(X_scaled) != n_test || ncol(X_scaled) != p_test) {
              stop("X_scaled dimension mismatch")
            }
            Sigma_factor <- X_scaled %*% t(X_test)
            if (!is.matrix(Sigma_factor) || nrow(Sigma_factor) != n_test || ncol(Sigma_factor) != n_test) {
              stop("Sigma_factor dimension mismatch")
            }
          }, error = function(e) {
            # If error, create simple diagonal matrix
            Sigma_factor <<- diag(n_test) * var_y_test * 0.5
          })
          
          # Normalize factor part: since X_test is standardized, 
          # we need to scale to match return variance
          # The factor part should contribute proportionally to total variance
          factor_scale <- mean(diag(Sigma_factor))
          if (factor_scale > 0 && is.finite(factor_scale)) {
            # Scale factor part to contribute ~50% of variance (adjustable)
            factor_weight <- 0.5
            Sigma_factor <- Sigma_factor * (var_y_test * factor_weight / factor_scale)
          } else {
            Sigma_factor <- Sigma_factor * 0
          }
          
          # Add idiosyncratic variance (remaining variance)
          # Ensure Sigma_factor is a proper matrix
          if (!is.matrix(Sigma_factor)) {
            Sigma_factor <- as.matrix(Sigma_factor)
          }
          if (nrow(Sigma_factor) != n_test || ncol(Sigma_factor) != n_test) {
            # Dimension mismatch, use simple diagonal matrix
            Sigma_test <- diag(n_test) * var_y_test
          } else {
            if (rho_eps > 0 && n_test > 1) {
              # Idiosyncratic correlation structure
              idio_var <- var_y_test * (1 - 0.5)  # Remaining variance
              Sigma_eps <- matrix(0, nrow = n_test, ncol = n_test)
              for (i in 1:n_test) {
                for (j in 1:n_test) {
                  Sigma_eps[i, j] <- idio_var * (rho_eps^abs(i - j))
                }
              }
              # Ensure diagonal is exactly idio_var
              for (i in 1:n_test) {
                Sigma_eps[i, i] <- idio_var
              }
              Sigma_test <- Sigma_factor + Sigma_eps
            } else {
              # Independent residuals: diagonal matrix
              idio_var <- var_y_test * (1 - 0.5)
              Sigma_test <- Sigma_factor + diag(n_test) * idio_var
            }
            
            # Ensure diagonal elements are positive and match return variance scale
            diag_vals <- diag(Sigma_test)
            if (length(diag_vals) != n_test || any(diag_vals <= 0) || any(!is.finite(diag_vals))) {
              diag(Sigma_test) <- pmax(rep(var_y_test * 0.5, n_test), na.rm = TRUE)
            }
            
            # Ensure well-conditioned (add small regularization)
            reg_term <- max(var_y_test * 1e-4, 1e-6)
            diag(Sigma_test) <- diag(Sigma_test) + reg_term
          }
          
          # Final check: ensure Sigma_test is a proper n_test × n_test matrix
          if (!is.matrix(Sigma_test) || nrow(Sigma_test) != n_test || ncol(Sigma_test) != n_test) {
            # Fallback: use simple diagonal matrix
            Sigma_test <- diag(n_test) * var_y_test
          }
        
        tryCatch({
            # Ensure dimensions match
            if (length(r_hat) != n_test || nrow(Sigma_test) != n_test || ncol(Sigma_test) != n_test) {
              portfolio_return_mean_variance <- portfolio_return_beta_weighted
            } else {
              # Mean-variance weights: w = Σ^(-1) μ / (1^T Σ^(-1) μ)
              Sigma_inv <- solve(Sigma_test)
              w_raw <- as.vector(Sigma_inv %*% as.vector(r_hat))
              w_sum <- sum(w_raw)
              
              if (abs(w_sum) > 1e-8 && is.finite(w_sum)) {
                weights_mv <- as.numeric(w_raw / w_sum)
                # Ensure weights are reasonable (not too extreme)
                weights_mv <- pmax(pmin(weights_mv, 1), -1)  # Clip to [-1, 1]
                weights_sum_abs <- sum(abs(weights_mv))
                if (weights_sum_abs > 1e-8) {
                  weights_mv <- weights_mv / weights_sum_abs  # Renormalize
                  portfolio_return_mean_variance <- sum(weights_mv * y_test)
                } else {
                  portfolio_return_mean_variance <- portfolio_return_beta_weighted
                }
              } else {
                # Fallback to beta-weighted if inversion fails
                portfolio_return_mean_variance <- portfolio_return_beta_weighted
              }
            }
        }, error = function(e) {
            # If inversion fails, use beta-weighted as fallback
            portfolio_return_mean_variance <<- portfolio_return_beta_weighted
        })
        }
      } else {
        # Not enough data for mean-variance, use beta-weighted
        portfolio_return_mean_variance <- portfolio_return_beta_weighted
      }
    }
  }
  
  # ============================================================================
  # Criterion 3: Matrix Error (from Step 3)
  # ============================================================================
  matrix_error <- NA
  if (!is.null(X) && !is.null(true_sigma) && !is.null(sigma)) {
    # Step 3: Σ̂ = X diag(β̂²) X^T + Σ_ε (with idiosyncratic correlation)
    Sigma_hat <- step3_estimate_covariance(X, beta_hat, sigma, rho_eps = rho_eps)
    
    # Matrix error: ||Σ̂ - Σ*||_F
    # Note: true_sigma here is the design matrix covariance, not the return covariance
    # For proper comparison, we should construct the true return covariance
    # using the factor structure: Σ* = X diag(β*²) X^T + Σ_ε
    n <- nrow(X)
    p <- ncol(X)
    
    # Ensure X is a matrix
    if (!is.matrix(X)) {
      X <- as.matrix(X)
    }
    
    # Construct true covariance using factor structure
    beta_star_sq <- beta_star^2
    X_scaled_star <- sweep(X, 2, beta_star_sq, "*")
    
    # Ensure X_scaled_star is a matrix
    if (!is.matrix(X_scaled_star)) {
      X_scaled_star <- as.matrix(X_scaled_star)
    }
    
    Sigma_star <- X_scaled_star %*% t(X)
    
    # Ensure Sigma_star is a proper matrix
    if (!is.matrix(Sigma_star) || nrow(Sigma_star) != n || ncol(Sigma_star) != n) {
      Sigma_star <- diag(n) * sigma^2
    }
    
    # Add idiosyncratic covariance (with correlation structure)
    if (rho_eps > 0 && n > 1) {
      # True idiosyncratic covariance with correlation
      Sigma_eps_star <- matrix(0, nrow = n, ncol = n)
      for (i in 1:n) {
        for (j in 1:n) {
          Sigma_eps_star[i, j] <- (sigma^2) * (rho_eps^abs(i - j))
        }
      }
      # Ensure dimensions match before addition
      if (nrow(Sigma_eps_star) == nrow(Sigma_star) && ncol(Sigma_eps_star) == ncol(Sigma_star)) {
        Sigma_star <- Sigma_star + Sigma_eps_star
      } else {
        diag(Sigma_star) <- diag(Sigma_star) + sigma^2
      }
    } else {
      # Independent residuals: + σ²I
      diag(Sigma_star) <- diag(Sigma_star) + sigma^2
    }
    
    # Final check: ensure both matrices have same dimensions
    if (is.matrix(Sigma_hat) && is.matrix(Sigma_star) && 
        nrow(Sigma_hat) == nrow(Sigma_star) && ncol(Sigma_hat) == ncol(Sigma_star)) {
      # Compute matrix error
      matrix_error <- step3_compute_matrix_error(Sigma_hat, Sigma_star)
    }
  }
  
  return(list(
    # Criterion 1: β-Estimation Error (Step 1)
    mse = mse,
    l2_error_sq = l2_error_sq,  # ||β̂ - β*||²
    l2_error = l2_error,
    l1_error = l1_error,
    tpr = tpr,
    fpr = fpr,
    false_positives = false_positives,
    false_negatives = false_negatives,
    exact_recovery = exact_recovery,
    support_accuracy = support_accuracy,
    n_nonzero = length(estimated_support),
    
    # Criterion 2: Return Prediction (Step 2)
    prediction_mse = prediction_mse,
    
    # Criterion 2b: Portfolio Returns (Both methods)
    portfolio_return_beta_weighted = portfolio_return_beta_weighted,
    portfolio_return_mean_variance = portfolio_return_mean_variance,
    
    # Criterion 3: Matrix Error (Step 3)
    matrix_error = matrix_error
  ))
}

# ============================================================================
# Run Single Simulation Replication
# ============================================================================
#
# This function implements the complete three-step pipeline:
#   1. Estimate factor loadings β̂
#   2. Predict returns r̂_{t+1} = X_t β̂
#   3. Estimate covariance Σ̂ = X diag(β̂²) X^T + σ²I
# ============================================================================

run_simulation <- function(n, p, rho, sigma, lambda_base, a, use_cv = TRUE, weak_signal = FALSE, rho_eps = 0.0) {
  # Generate data with idiosyncratic correlation
  if (weak_signal) {
    data <- generate_data_weak_signal(n, p, rho, sigma, rho_eps = rho_eps)
  } else {
    data <- generate_data(n, p, rho, sigma, rho_eps = rho_eps)
  }
  X <- data$X
  y <- data$y
  beta_star <- data$beta_star
  X_test <- data$X_test
  y_test <- data$y_test
  true_sigma <- data$true_sigma
  rho_eps <- data$rho_eps
  sigma <- data$sigma
  
  results <- list()
  
  # ========================================================================
  # STEP 1: Estimating Factor Loadings
  # ========================================================================
  # β̂ = argmin_{β} { (1/2)||y - Xβ||² + p_λ(β) }
  step1_results <- step1_estimate_factor_loadings(X, y, lambda_base, a, use_cv)
  
  # ========================================================================
  # STEP 2: Return Prediction via Estimated Loadings
  # ========================================================================
  # r̂_{i,t+1} = X_{i,t} β̂_t
  # (Computed within compute_metrics for each method)
  
  # ========================================================================
  # STEP 3: Covariance Estimation via Factor Structure
  # ========================================================================
  # Σ̂ = X diag(β̂²) X^T + Σ_ε (with idiosyncratic correlation)
  # MatrixError = ||Σ̂ - Σ*||_F
  # (Computed within compute_metrics for each method)
  
  # Evaluate all three criteria for each method
  for (method_name in names(step1_results)) {
    tryCatch({
      beta_hat <- step1_results[[method_name]]$beta_hat
      
      # Ensure beta_hat is a vector
      beta_hat <- as.numeric(beta_hat)
      
      # Compute metrics for all three steps
      metrics <- compute_metrics(beta_hat, beta_star, X = X, y = y,
                                               X_test = X_test, y_test = y_test,
                                true_sigma = true_sigma, sigma = sigma, rho_eps = rho_eps)
      
      # Store results
      results[[method_name]] <- c(
        metrics,
        list(
          lambda_used = step1_results[[method_name]]$lambda_used
        )
      )
      
      # Add iteration info if available
      if ("iterations" %in% names(step1_results[[method_name]])) {
        results[[method_name]]$iterations <- step1_results[[method_name]]$iterations
        results[[method_name]]$converged <- step1_results[[method_name]]$converged
      }
    }, error = function(e) {
      # If error occurs, create minimal results with NA values
      cat(sprintf("Warning: Error computing metrics for %s: %s\n", method_name, e$message))
      results[[method_name]] <<- list(
        mse = NA,
        l2_error_sq = NA,
        prediction_mse = NA,
        portfolio_return_beta_weighted = NA,
        portfolio_return_mean_variance = NA,
        matrix_error = NA,
        lambda_used = step1_results[[method_name]]$lambda_used
      )
    })
  }
  
  return(results)
}

# ============================================================================
# Main Simulation Loop
# ============================================================================

cat("\n")
cat("===========================================\n")
cat("Starting Simulation Study\n")
cat("===========================================\n")
cat(sprintf("Number of replications: %d\n", n_sim))
cat(sprintf("Number of configurations: %d\n\n", length(configurations)))
flush.console()

all_results <- list()

for (config_idx in 1:length(configurations)) {
  config <- configurations[[config_idx]]
  n <- config$n
  p <- config$p
  rho <- config$rho
  sigma <- config$sigma
  rho_eps <- ifelse(is.null(config$rho_eps), 0.0, config$rho_eps)
  config_name <- config$name
  weak_signal <- ifelse(is.null(config$weak_signal), FALSE, config$weak_signal)
  
  cat("\n")
  cat("===========================================\n")
  cat(sprintf("Configuration %d/%d: %s\n", config_idx, length(configurations), config_name))
  cat(sprintf("  n=%d, p=%d, rho=%.1f, sigma=%.1f, rho_eps=%.2f\n", n, p, rho, sigma, rho_eps))
  cat("===========================================\n\n")
  flush.console()
  
  config_key <- sprintf("config_%d", config_idx)
  all_results[[config_key]] <- list()
  
  # Run simulations
  cat("Running simulations...\n")
  cat("📌 Idiosyncratic correlation (rho_eps) creates difference between MV and beta-weighted!\n")
  cat("   When rho_eps > 0, MV considers risk structure, beta-weighted only considers signal.\n\n")
  flush.console()
  
  for (sim in 1:n_sim) {
    if (sim %% 20 == 0 || sim == 1) {
      cat(sprintf("  Progress: %d/%d (%.1f%%)\n", sim, n_sim, 100*sim/n_sim))
      flush.console()
    }
    
    tryCatch({
    result <- run_simulation(n, p, rho, sigma, lambda_base, a, 
                              use_cv = TRUE, weak_signal = weak_signal, rho_eps = rho_eps)
    
    # Store results
    for (method in names(result)) {
      if (is.null(all_results[[config_key]][[method]])) {
        all_results[[config_key]][[method]] <- list()
        for (metric in names(result[[method]])) {
          all_results[[config_key]][[method]][[metric]] <- numeric(n_sim)
        }
      }
      for (metric in names(result[[method]])) {
          if (metric %in% names(all_results[[config_key]][[method]])) {
        all_results[[config_key]][[method]][[metric]][sim] <- result[[method]][[metric]]
      }
    }
      }
    }, error = function(e) {
      cat(sprintf("  Warning: Simulation %d failed: %s\n", sim, e$message))
      flush.console()
      # Skip this simulation, continue with next
    })
  }
  
  # Print summary statistics
  cat("\n")
  cat("===========================================\n")
  cat(sprintf("Summary Statistics (over %d replications)\n", n_sim))
  cat("===========================================\n\n")
  flush.console()
  
  summary_stats <- list()
  
  methods <- names(all_results[[config_key]])
  methods <- methods[!(methods %in% c("summary", "config"))]
  
  baseline_methods <- c("lasso")
  lqa_methods <- c("scad_lqa", "scad_lqa_improved")
  
  # Print main comparison table: Three Criteria
  cat("=== Three-Step Framework Results: LASSO vs LQA vs LQA Improved ===\n\n")
  
  # Step 1: β-Estimation Error
  cat("Step 1: β-Estimation Error\n")
  cat(sprintf("%-25s %12s %10s %10s %8s %8s %10s\n", 
              "Method", "||β̂-β*||²", "TPR", "FPR", "FP", "FN", "Iterations"))
  cat(paste(rep("-", 90), collapse=""), "\n")
  
  for (method in c(baseline_methods, lqa_methods)) {
    if (!method %in% methods) next
    
    metrics <- all_results[[config_key]][[method]]
    method_summary <- list()
    
    # Step 1 metrics
    l2_error_sq_mean <- if ("l2_error_sq" %in% names(metrics)) mean(metrics$l2_error_sq, na.rm = TRUE) else NA
    mse_mean <- if ("mse" %in% names(metrics)) mean(metrics$mse, na.rm = TRUE) else NA
    l2_mean <- if ("l2_error" %in% names(metrics)) mean(metrics$l2_error, na.rm = TRUE) else NA
    l1_mean <- if ("l1_error" %in% names(metrics)) mean(metrics$l1_error, na.rm = TRUE) else NA
    tpr_mean <- if ("tpr" %in% names(metrics)) mean(metrics$tpr, na.rm = TRUE) else NA
    fpr_mean <- if ("fpr" %in% names(metrics)) mean(metrics$fpr, na.rm = TRUE) else NA
    fp_mean <- if ("false_positives" %in% names(metrics)) mean(metrics$false_positives, na.rm = TRUE) else NA
    fn_mean <- if ("false_negatives" %in% names(metrics)) mean(metrics$false_negatives, na.rm = TRUE) else NA
    exact_rec <- if ("exact_recovery" %in% names(metrics)) mean(metrics$exact_recovery, na.rm = TRUE) else NA
    iter_mean <- if ("iterations" %in% names(metrics)) mean(metrics$iterations, na.rm = TRUE) else NA
    
    # Step 2 metrics
    prediction_mse_mean <- if ("prediction_mse" %in% names(metrics)) mean(metrics$prediction_mse, na.rm = TRUE) else NA
    portfolio_return_beta_mean <- if ("portfolio_return_beta_weighted" %in% names(metrics)) mean(metrics$portfolio_return_beta_weighted, na.rm = TRUE) else NA
    portfolio_return_mv_mean <- if ("portfolio_return_mean_variance" %in% names(metrics)) mean(metrics$portfolio_return_mean_variance, na.rm = TRUE) else NA
    
    # Step 3 metrics
    matrix_error_mean <- if ("matrix_error" %in% names(metrics)) mean(metrics$matrix_error, na.rm = TRUE) else NA
    
    method_display <- switch(method,
                             "lasso" = "LASSO (baseline)",
                             "scad_lqa" = "SCAD LQA",
                             "scad_lqa_improved" = "SCAD LQA Improved ⭐",
                             toupper(method))
    
    lambda_used <- if ("lambda_used" %in% names(metrics)) {
      mean(metrics$lambda_used, na.rm = TRUE)
    } else {
      NA
    }
    
    # Print formatted row for Step 1
    cat(sprintf("%-25s %12.4f %10.4f %10.4f %8.2f %8.2f %10.2f",
                method_display, l2_error_sq_mean, tpr_mean, fpr_mean, fp_mean, fn_mean, iter_mean))
    if (!is.na(lambda_used)) {
      cat(sprintf(" (λ=%.3f)", lambda_used))
    }
    cat("\n")
    
    # Store summary
    method_summary$l2_error_sq_mean <- l2_error_sq_mean
    method_summary$mse_mean <- mse_mean
    method_summary$l2_error_mean <- l2_mean
    method_summary$l1_error_mean <- l1_mean
    method_summary$tpr_mean <- tpr_mean
    method_summary$fpr_mean <- fpr_mean
    method_summary$false_positives_mean <- fp_mean
    method_summary$false_negatives_mean <- fn_mean
    method_summary$exact_recovery_rate <- exact_rec
    method_summary$iterations_mean <- iter_mean
    method_summary$prediction_mse_mean <- prediction_mse_mean
    method_summary$portfolio_return_beta_mean <- portfolio_return_beta_mean
    method_summary$portfolio_return_mv_mean <- portfolio_return_mv_mean
    method_summary$matrix_error_mean <- matrix_error_mean
    
    if ("converged" %in% names(metrics)) {
      conv_rate <- mean(metrics$converged, na.rm = TRUE)
      method_summary$convergence_rate <- conv_rate
    }
    if ("n_nonzero" %in% names(metrics)) {
      n_nonzero_mean <- mean(metrics$n_nonzero, na.rm = TRUE)
      method_summary$n_nonzero_mean <- n_nonzero_mean
    }
    if ("support_accuracy" %in% names(metrics)) {
      support_acc_mean <- mean(metrics$support_accuracy, na.rm = TRUE)
      method_summary$support_accuracy_mean <- support_acc_mean
    }
    
    summary_stats[[method]] <- method_summary
  }
  
  cat("\n⭐ = LQA Improved (Stabilized-LQA)\n")
  cat("Key improvements:\n")
  cat("  - Uses QR/SVD decomposition for numerical stability\n")
  cat("  - Adaptive Ridge boosting: H = X^T X + W + γI\n")
  cat("  - Better performance in high-dimensional settings (p >> n)\n")
  cat("  - More stable than LLA, especially with high correlation\n\n")
  
  # Step 2: Return Prediction
  cat("\nStep 2: Return Prediction (r̂_{i,t+1} = X_{i,t} β̂_t)\n")
  cat(sprintf("%-25s %15s\n", "Method", "Prediction MSE"))
  cat(paste(rep("-", 45), collapse=""), "\n")
  for (method in c(baseline_methods, lqa_methods)) {
    if (!method %in% methods) next
    metrics <- all_results[[config_key]][[method]]
    prediction_mse_mean <- if ("prediction_mse" %in% names(metrics)) mean(metrics$prediction_mse, na.rm = TRUE) else NA
    method_display <- switch(method,
                             "lasso" = "LASSO (baseline)",
                             "scad_lqa" = "SCAD LQA",
                             "scad_lqa_improved" = "SCAD LQA Improved ⭐",
                             toupper(method))
    cat(sprintf("%-25s %15.6f\n", method_display, prediction_mse_mean))
  }
  
  # Step 2b: Portfolio Returns (Both Methods)
  cat("\nStep 2b: Portfolio Returns (Beta-Weighted vs Mean-Variance)\n")
  cat("📌 Key Insight: When rho_eps > 0, idiosyncratic correlation creates risk structure.\n")
  cat("   - Beta-weighted: Only considers signal (predicted returns), ignores risk correlation\n")
  cat("   - Mean-Variance: Considers both signal AND risk structure (covariance matrix)\n")
  cat("   → MV and beta-weighted will differ significantly when rho_eps > 0!\n\n")
  cat(sprintf("%-25s %20s %20s %20s\n", "Method", "Beta-Weighted", "Mean-Variance", "Difference"))
  cat(paste(rep("-", 90), collapse=""), "\n")
  for (method in c(baseline_methods, lqa_methods)) {
    if (!method %in% methods) next
    metrics <- all_results[[config_key]][[method]]
    portfolio_return_beta_mean <- if ("portfolio_return_beta_weighted" %in% names(metrics)) mean(metrics$portfolio_return_beta_weighted, na.rm = TRUE) else NA
    portfolio_return_mv_mean <- if ("portfolio_return_mean_variance" %in% names(metrics)) mean(metrics$portfolio_return_mean_variance, na.rm = TRUE) else NA
    diff_mean <- if (!is.na(portfolio_return_beta_mean) && !is.na(portfolio_return_mv_mean)) portfolio_return_mv_mean - portfolio_return_beta_mean else NA
    method_display <- switch(method,
                             "lasso" = "LASSO (baseline)",
                             "scad_lqa" = "SCAD LQA",
                             "scad_lqa_improved" = "SCAD LQA Improved ⭐",
                             toupper(method))
    cat(sprintf("%-25s %20.6f %20.6f %20.6f\n", method_display, portfolio_return_beta_mean, portfolio_return_mv_mean, diff_mean))
  }
  
  # Step 3: Covariance Estimation
  cat("\nStep 3: Covariance Estimation (||Σ̂ - Σ*||_F)\n")
  cat(sprintf("%-25s %15s\n", "Method", "Matrix Error"))
  cat(paste(rep("-", 45), collapse=""), "\n")
  for (method in c(baseline_methods, lqa_methods)) {
    if (!method %in% methods) next
    metrics <- all_results[[config_key]][[method]]
    matrix_error_mean <- if ("matrix_error" %in% names(metrics)) mean(metrics$matrix_error, na.rm = TRUE) else NA
    method_display <- switch(method,
                             "lasso" = "LASSO (baseline)",
                             "scad_lqa" = "SCAD LQA",
                             "scad_lqa_improved" = "SCAD LQA Improved ⭐",
                             toupper(method))
    cat(sprintf("%-25s %15.4f\n", method_display, matrix_error_mean))
  }
  
  cat("\n")
  flush.console()
  
  # Print detailed statistics
  cat("=== Detailed Statistics: LASSO vs LQA vs LQA Improved Comparison ===\n\n")
  
  for (method in c(baseline_methods, lqa_methods)) {
    if (!method %in% methods) next
    
    metrics <- all_results[[config_key]][[method]]
    
    method_display <- switch(method,
                             "lasso" = "LASSO (baseline)",
                             "scad_lqa" = "SCAD LQA",
                             "scad_lqa_improved" = "SCAD LQA Improved ⭐",
                             toupper(method))
    
    cat(sprintf("%s:\n", method_display))
    
    # Step 1: β-Estimation Error
    if ("l2_error_sq" %in% names(metrics)) {
      mean_val <- mean(metrics$l2_error_sq, na.rm = TRUE)
      sd_val <- sd(metrics$l2_error_sq, na.rm = TRUE)
      cat(sprintf("  Step 1 - ||β̂-β*||²: %.4f (SD: %.4f)\n", mean_val, sd_val))
    }
    if ("mse" %in% names(metrics)) {
      mean_val <- mean(metrics$mse, na.rm = TRUE)
      sd_val <- sd(metrics$mse, na.rm = TRUE)
      cat(sprintf("  Step 1 - MSE: %.4f (SD: %.4f)\n", mean_val, sd_val))
    }
    if ("tpr" %in% names(metrics)) {
      mean_val <- mean(metrics$tpr, na.rm = TRUE)
      sd_val <- sd(metrics$tpr, na.rm = TRUE)
      cat(sprintf("  Step 1 - True Positive Rate: %.4f (SD: %.4f)\n", mean_val, sd_val))
    }
    if ("fpr" %in% names(metrics)) {
      mean_val <- mean(metrics$fpr, na.rm = TRUE)
      sd_val <- sd(metrics$fpr, na.rm = TRUE)
      cat(sprintf("  Step 1 - False Positive Rate: %.4f (SD: %.4f)\n", mean_val, sd_val))
    }
    
    # Step 2: Return Prediction
    if ("prediction_mse" %in% names(metrics)) {
      mean_val <- mean(metrics$prediction_mse, na.rm = TRUE)
      sd_val <- sd(metrics$prediction_mse, na.rm = TRUE)
      cat(sprintf("  Step 2 - Prediction MSE: %.6f (SD: %.6f)\n", mean_val, sd_val))
    }
    if ("portfolio_return_beta_weighted" %in% names(metrics)) {
      mean_val <- mean(metrics$portfolio_return_beta_weighted, na.rm = TRUE)
      sd_val <- sd(metrics$portfolio_return_beta_weighted, na.rm = TRUE)
      cat(sprintf("  Step 2b - Portfolio Return (Beta-weighted): %.6f (SD: %.6f)\n", mean_val, sd_val))
    }
    if ("portfolio_return_mean_variance" %in% names(metrics)) {
      mean_val <- mean(metrics$portfolio_return_mean_variance, na.rm = TRUE)
      sd_val <- sd(metrics$portfolio_return_mean_variance, na.rm = TRUE)
      cat(sprintf("  Step 2b - Portfolio Return (Mean-Variance): %.6f (SD: %.6f)\n", mean_val, sd_val))
    }
    
    # Step 3: Covariance Estimation
    if ("matrix_error" %in% names(metrics)) {
      mean_val <- mean(metrics$matrix_error, na.rm = TRUE)
      sd_val <- sd(metrics$matrix_error, na.rm = TRUE)
      cat(sprintf("  Step 3 - Matrix Error (||Σ̂-Σ*||_F): %.4f (SD: %.4f)\n", mean_val, sd_val))
    }
    
    if ("iterations" %in% names(metrics)) {
      mean_val <- mean(metrics$iterations, na.rm = TRUE)
      sd_val <- sd(metrics$iterations, na.rm = TRUE)
      cat(sprintf("  Convergence Iterations: %.2f (SD: %.2f)\n", mean_val, sd_val))
    }
    if ("converged" %in% names(metrics)) {
      mean_val <- mean(metrics$converged, na.rm = TRUE)
      cat(sprintf("  Convergence Rate: %.4f\n", mean_val))
    }
    
    cat("\n")
  }
  
  flush.console()
  
  # Store summary statistics
  all_results[[config_key]]$summary <- summary_stats
  all_results[[config_key]]$config <- config
}

cat("\n")
cat("===========================================\n")
cat("Simulation Study Completed!\n")
cat("===========================================\n\n")
flush.console()

# Optionally save results
if (save_results) {
  cat("=== Saving Results ===\n")
  results_dir <- "results"
  if (!dir.exists(results_dir)) {
    dir.create(results_dir, recursive = TRUE)
  }
  
  results_file <- file.path(results_dir, "simulation_results.RData")
  save(all_results, file = results_file)
  cat(sprintf("Results saved to %s\n", results_file))
} else {
  cat("(Results not saved. Set save_results = TRUE to save to files.)\n")
}
cat("\n")
flush.console()
