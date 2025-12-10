#' Solve Dynamic LQA System using ADMM
#'
#' Solves the block tridiagonal system arising from Dynamic SCAD with fused penalty
#' using Alternating Direction Method of Multipliers (ADMM).
#'
#' @section Problem Formulation:
#'
#' After LQA approximation, we need to solve:
#' \deqn{
#' \min_{\boldsymbol{\beta}} \left\{
#' \sum_{t=1}^T \|\mathbf{y}_t - \mathbf{X}_t \boldsymbol{\beta}_t\|_2^2
#' + \sum_{t=1}^T \sum_{j=1}^p w_{j,t} \beta_{j,t}^2
#' + \tau \sum_{t=2}^T \sum_{j=1}^p |\beta_{j,t} - \beta_{j,t-1}|
#' \right\}
#' }
#'
#' Using ADMM, we introduce auxiliary variables \eqn{\mathbf{z}_t} and solve:
#' \deqn{
#' \min_{\boldsymbol{\beta}, \mathbf{z}} \left\{
#' \sum_{t=1}^T \|\mathbf{y}_t - \mathbf{X}_t \boldsymbol{\beta}_t\|_2^2
#' + \sum_{t=1}^T \sum_{j=1}^p w_{j,t} \beta_{j,t}^2
#' + \tau \sum_{t=2}^T \sum_{j=1}^p |z_{j,t}|
#' \right\}
#' }
#' subject to \eqn{\beta_{j,t} - \beta_{j,t-1} = z_{j,t}} for \eqn{t = 2, \ldots, T}.
#'
#' @param y_list List of response vectors
#' @param X_list List of design matrices
#' @param w_mat Weight matrix of dimension \eqn{p \times T} for LQA weights
#' @param lambda SCAD tuning parameter
#' @param tau Fused penalty tuning parameter
#' @param rho ADMM penalty parameter
#' @param max_iter Maximum ADMM iterations
#' @param tol Convergence tolerance
#'
#' @return Coefficient matrix \eqn{p \times T}
#'
#' @keywords internal
solve_dynamic_lqa_admm <- function(y_list, X_list, w_mat, lambda, tau,
                                   rho = 1.0, max_iter = 100, tol = 1e-6) {
  
  T <- length(y_list)
  p <- ncol(X_list[[1]])
  
  # Initialize
  beta <- matrix(0, p, T)
  z <- matrix(0, p, T - 1)  # z_{j,t} = beta_{j,t+1} - beta_{j,t}
  z_old <- z
  u <- matrix(0, p, T - 1)  # ADMM dual variable
  
  # Precompute X^T X and X^T y for each time point
  XtX_list <- list()
  Xty_list <- list()
  
  for (t in 1:T) {
    XtX_list[[t]] <- crossprod(X_list[[t]])
    Xty_list[[t]] <- crossprod(X_list[[t]], y_list[[t]])
  }
  
  # ADMM iterations
  for (admm_iter in 1:max_iter) {
    beta_old <- beta
    
    # Update beta: solve for each time point with fused penalty constraint
    # For each time point t, solve weighted ridge regression with ADMM penalty
    # Use beta_old for constraints to ensure convergence
    for (t in 1:T) {
      # Build augmented system: (X_t^T X_t + diag(w_{.,t}) + rho*penalty) beta_t = rhs
      # The penalty term comes from ADMM constraints: beta_{j,t+1} - beta_{j,t} = z_{j,t}
      
      # Base system: X_t^T X_t + diag(w_{.,t})
      H_t <- XtX_list[[t]]
      diag(H_t) <- diag(H_t) + w_mat[, t]
      
      # Add ADMM penalty terms
      if (t == 1) {
        # Only constraint with t+1
        diag(H_t) <- diag(H_t) + rho
        rhs_t <- Xty_list[[t]] + rho * (beta_old[, t + 1] - z[, t] + u[, t])
      } else if (t == T) {
        # Only constraint with t-1
        diag(H_t) <- diag(H_t) + rho
        rhs_t <- Xty_list[[t]] + rho * (beta_old[, t - 1] + z[, t - 1] - u[, t - 1])
      } else {
        # Constraints with both t-1 and t+1
        diag(H_t) <- diag(H_t) + 2 * rho
        rhs_t <- Xty_list[[t]] + 
          rho * (beta_old[, t - 1] + z[, t - 1] - u[, t - 1] +
                 beta_old[, t + 1] - z[, t] + u[, t])
      }
      
      # Solve: H_t beta_t = rhs_t
      beta[, t] <- solve(H_t, rhs_t)
    }
    
    # Update z: soft thresholding
    # z_{j,t} = S_{tau/rho}(beta_{j,t+1} - beta_{j,t} + u_{j,t})
    for (t in 1:(T - 1)) {
      diff <- beta[, t + 1] - beta[, t] + u[, t]
      z[, t] <- soft_threshold(diff, tau / rho)
    }
    
    # Update u: dual variable
    for (t in 1:(T - 1)) {
      u[, t] <- u[, t] + (beta[, t + 1] - beta[, t] - z[, t])
    }
    
    # Check convergence
    primal_residual <- 0
    dual_residual <- 0
    
    for (t in 1:(T - 1)) {
      primal_residual <- max(primal_residual, 
                            max(abs(beta[, t + 1] - beta[, t] - z[, t])))
      dual_residual <- max(dual_residual,
                          max(abs(rho * (z[, t] - z_old[, t]))))
    }
    
    z_old <- z
    
    if (max(primal_residual, dual_residual) < tol) {
      break
    }
  }
  
  return(beta)
}

#' Solve Tridiagonal System using Thomas Algorithm
#'
#' Solves \eqn{A \mathbf{x} = \mathbf{b}} where A is tridiagonal.
#'
#' @param diag_main Main diagonal (length n)
#' @param diag_sub Subdiagonal (length n-1)
#' @param diag_super Superdiagonal (length n-1)
#' @param rhs Right-hand side vector (length n)
#'
#' @return Solution vector \eqn{\mathbf{x}}
#'
#' @keywords internal
solve_tridiagonal_system <- function(diag_main, diag_sub, diag_super, rhs) {
  n <- length(diag_main)
  
  if (n == 1) {
    return(rhs / diag_main)
  }
  
  # Forward elimination
  c_prime <- numeric(n - 1)
  d_prime <- numeric(n)
  
  c_prime[1] <- diag_super[1] / diag_main[1]
  d_prime[1] <- rhs[1] / diag_main[1]
  
  for (i in 2:(n - 1)) {
    denom <- diag_main[i] - diag_sub[i - 1] * c_prime[i - 1]
    c_prime[i] <- diag_super[i] / denom
    d_prime[i] <- (rhs[i] - diag_sub[i - 1] * d_prime[i - 1]) / denom
  }
  
  # Last row
  denom <- diag_main[n] - diag_sub[n - 1] * c_prime[n - 1]
  d_prime[n] <- (rhs[n] - diag_sub[n - 1] * d_prime[n - 1]) / denom
  
  # Backward substitution
  x <- numeric(n)
  x[n] <- d_prime[n]
  
  for (i in (n - 1):1) {
    x[i] <- d_prime[i] - c_prime[i] * x[i + 1]
  }
  
  return(x)
}

#' Soft Thresholding Operator
#'
#' \eqn{S_\lambda(x) = \text{sign}(x) \max(|x| - \lambda, 0)}
#'
#' @param x Input vector
#' @param lambda Threshold parameter
#'
#' @return Soft-thresholded vector
#'
#' @keywords internal
soft_threshold <- function(x, lambda) {
  sign(x) * pmax(abs(x) - lambda, 0)
}

