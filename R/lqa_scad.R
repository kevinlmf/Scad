#' Local Quadratic Approximation (LQA) for SCAD Penalized Regression
#'
#' Implements the LQA algorithm for SCAD penalized regression.
#' LQA uses quadratic approximation instead of linear approximation (LLA).
#'
#' Algorithm:
#' At iteration t, approximate SCAD penalty by quadratic:
#' p_SCAD(|beta_j|) ≈ p_SCAD(|beta_j^(t)|) + p'_SCAD(|beta_j^(t)|) / (2|beta_j^(t)|) * (beta_j^2 - beta_j^(t)^2)
#'
#' This leads to solving weighted Ridge regression at each iteration:
#' argmin ||y - X*beta||^2 + lambda * sum(w_j * beta_j^2)
#' where w_j = p'_SCAD(|beta_j^(t)|) / |beta_j^(t)|
#'
#' @param y Response vector of length n
#' @param X Design matrix of dimension n x p
#' @param lambda Tuning parameter for SCAD penalty (scalar)
#' @param a Shape parameter for SCAD penalty, must be > 2 (default: 3.7)
#' @param beta_init Initial coefficient vector of length p (default: LASSO solution)
#' @param max_iter Maximum number of iterations (default: 100)
#' @param tol Convergence tolerance (default: 1e-6)
#' @param standardize Logical, whether to standardize X (default: TRUE)
#'
#' @return A list containing:
#' \item{beta}{Estimated coefficient vector}
#' \item{iterations}{Number of iterations until convergence}
#' \item{converged}{Logical indicating convergence}
#' \item{objective}{Final objective value}
#' \item{objective_history}{History of objective values}
#'
#' @export
lqa_scad <- function(y, X, lambda, a = 3.7, beta_init = NULL, 
                     max_iter = 100, tol = 1e-6, standardize = TRUE) {
  
  # Input validation
  n <- nrow(X)
  p <- ncol(X)
  
  if (length(y) != n) {
    stop("Length of y must equal number of rows in X")
  }
  if (a <= 2) {
    stop("Parameter a must be > 2")
  }
  if (lambda <= 0) {
    stop("Lambda must be positive")
  }
  
  # Standardize X if requested
  if (standardize) {
    X_mean <- colMeans(X)
    X_sd <- apply(X, 2, sd)
    X_sd[X_sd == 0] <- 1  # Avoid division by zero
    X <- scale(X, center = X_mean, scale = X_sd)
  } else {
    X_mean <- rep(0, p)
    X_sd <- rep(1, p)
  }
  
  # Initialize beta (use LASSO as initial estimate)
  if (is.null(beta_init)) {
    lasso_fit <- glmnet::glmnet(X, y, lambda = lambda, alpha = 1, 
                                standardize = FALSE, intercept = FALSE)
    beta <- as.numeric(lasso_fit$beta)
  } else {
    beta <- beta_init
  }
  
  # SCAD penalty derivative function
  scad_derivative <- function(theta, lambda, a) {
    abs_theta <- abs(theta)
    result <- numeric(length(theta))
    result[abs_theta <= lambda] <- lambda
    idx <- (abs_theta > lambda) & (abs_theta <= a * lambda)
    result[idx] <- (a * lambda - abs_theta[idx]) / (a - 1)
    result[abs_theta > a * lambda] <- 0
    return(result)
  }
  
  # SCAD penalty function
  scad_penalty <- function(theta, lambda, a) {
    abs_theta <- abs(theta)
    result <- numeric(length(theta))
    result[abs_theta <= lambda] <- lambda * abs_theta[abs_theta <= lambda]
    idx <- (abs_theta > lambda) & (abs_theta <= a * lambda)
    result[idx] <- lambda * (abs_theta[idx] + lambda) / 2 + 
      (a * lambda - abs_theta[idx])^2 / (2 * (a - 1))
    result[abs_theta > a * lambda] <- lambda * (a + 1) * lambda / 2
    return(result)
  }
  
  # Objective function
  scad_objective <- function(beta, y, X, lambda, a) {
    n <- length(y)
    residuals <- y - X %*% beta
    rss <- sum(residuals^2) / (2 * n)
    penalty <- sum(scad_penalty(beta, lambda, a))
    return(rss + penalty)
  }
  
  # Solve weighted Ridge regression
  # argmin ||y - X*beta||^2 + lambda * sum(w_j * beta_j^2)
  solve_weighted_ridge <- function(X, y, w, lambda) {
    p <- ncol(X)
    n <- nrow(X)
    
    # Weighted Ridge: (X^T X + lambda * diag(w))^(-1) X^T y
    # Use glmnet with alpha=0 (Ridge) and penalty factors
    penalty_factor <- w / lambda
    penalty_factor[penalty_factor == 0] <- 1e-10  # Avoid zero weights
    
    ridge_fit <- glmnet::glmnet(X, y, lambda = lambda, alpha = 0,
                                penalty.factor = penalty_factor,
                                standardize = FALSE, intercept = FALSE)
    beta <- as.numeric(ridge_fit$beta)
    return(beta)
  }
  
  # Main LQA iteration
  converged <- FALSE
  objective_history <- numeric(max_iter)
  
  for (iter in 1:max_iter) {
    beta_old <- beta
    f_old <- scad_objective(beta, y, X, lambda, a)
    objective_history[iter] <- f_old
    
    # Compute SCAD penalty derivative
    w_prime <- scad_derivative(beta, lambda, a)
    
    # Compute LQA weights: w_j = p'_SCAD(|beta_j|) / |beta_j|
    # Key: For very small coefficients, set large penalty to drive to zero
    abs_beta <- abs(beta)
    epsilon <- 1e-6
    
    # For coefficients smaller than epsilon, use large penalty to encourage sparsity
    # For larger coefficients, use standard LQA weight
    w <- numeric(p)
    small_idx <- abs_beta < epsilon
    large_idx <- !small_idx
    
    # Small coefficients: use large penalty (encourage sparsity)
    w[small_idx] <- lambda / epsilon
    
    # Large coefficients: use LQA weight
    w[large_idx] <- w_prime[large_idx] / abs_beta[large_idx]
    
    # Cap weights to avoid numerical issues
    w <- pmin(w, lambda / epsilon)
    
    # Solve weighted Ridge regression
    beta <- solve_weighted_ridge(X, y, w, lambda)
    
    # Apply thresholding for sparsity (LQA uses Ridge which doesn't produce sparse solutions)
    # Strategy: Use hard thresholding based on lambda
    # This is necessary because Ridge regression doesn't naturally produce sparse solutions
    threshold <- lambda * 0.5  # More aggressive threshold
    beta[abs(beta) < threshold] <- 0
    
    # Additional sparsity: If too many coefficients remain, keep only top k
    # where k is based on expected sparsity level (e.g., similar to LASSO)
    n_nonzero <- sum(beta != 0)
    if (n_nonzero > p * 0.3) {  # If more than 30% are non-zero
      # Keep only top coefficients by absolute value
      abs_beta_sorted <- sort(abs(beta), decreasing = TRUE)
      # Estimate sparsity level from LASSO (roughly)
      expected_sparsity <- min(p * 0.2, sum(abs(beta) > lambda))
      if (expected_sparsity > 0 && expected_sparsity < n_nonzero) {
        threshold_value <- abs_beta_sorted[expected_sparsity]
        beta[abs(beta) < threshold_value] <- 0
      }
    }
    
    # Check convergence
    if (max(abs(beta - beta_old)) < tol) {
      converged <- TRUE
      break
    }
  }
  
  # Unstandardize coefficients if needed
  if (standardize) {
    beta <- beta / X_sd
  }
  
  return(list(
    beta = beta,
    iterations = iter,
    converged = converged,
    objective = objective_history[iter],
    objective_history = objective_history[1:iter]
  ))
}

