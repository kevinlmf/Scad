#' Dynamic SCAD Penalized Regression
#'
#' Implements Dynamic SCAD model for time-varying coefficient regression with
#' SCAD sparsity penalty and temporal smoothness constraint.
#'
#' @section Model Formulation:
#'
#' The Dynamic SCAD model estimates time-varying regression coefficients
#' \eqn{\{\boldsymbol{\beta}_t\}_{t=1}^T} from time series data
#' \eqn{\{(y_t, \mathbf{X}_t)\}_{t=1}^T} by solving:
#'
#' \deqn{
#' \min_{\boldsymbol{\beta}_1, \ldots, \boldsymbol{\beta}_T} \left\{
#' \sum_{t=1}^T \|\mathbf{y}_t - \mathbf{X}_t \boldsymbol{\beta}_t\|_2^2
#' + \lambda \sum_{t=1}^T \sum_{j=1}^p \text{SCAD}(\beta_{j,t})
#' + \tau \sum_{t=2}^T \sum_{j=1}^p |\beta_{j,t} - \beta_{j,t-1}|
#' \right\}
#' }
#'
#' where:
#' \itemize{
#'   \item \eqn{\mathbf{y}_t \in \mathbb{R}^n} or \eqn{\mathbb{R}} is the response at time \eqn{t}
#'   \item \eqn{\mathbf{X}_t \in \mathbb{R}^{n \times p}} is the design matrix at time \eqn{t}
#'   \item \eqn{\boldsymbol{\beta}_t \in \mathbb{R}^p} are time-varying coefficients
#'   \item \eqn{\lambda > 0} controls SCAD sparsity penalty
#'   \item \eqn{\tau > 0} controls temporal smoothness (fused penalty)
#'   \item \eqn{\text{SCAD}(\beta)} is the SCAD penalty function (Fan & Li, 2001)
#' }
#'
#' @section Model Components:
#'
#' \strong{1. Time-Varying Regression Loss:}
#' \deqn{L(\boldsymbol{\beta}) = \sum_{t=1}^T \|\mathbf{y}_t - \mathbf{X}_t \boldsymbol{\beta}_t\|_2^2}
#' Allows coefficients to vary over time, capturing regime shifts and structural breaks.
#'
#' \strong{2. SCAD Sparsity Penalty:}
#' \deqn{\lambda \sum_{t=1}^T \sum_{j=1}^p \text{SCAD}(\beta_{j,t})}
#' Provides oracle property (Fan & Li, 2001), enabling time-varying variable selection.
#' A factor can be active in some periods and inactive in others.
#'
#' \strong{3. Temporal Smoothness (Fused Penalty):}
#' \deqn{\tau \sum_{t=2}^T \sum_{j=1}^p |\beta_{j,t} - \beta_{j,t-1}|}
#' Encourages smooth coefficient paths and automatically detects structural breaks.
#' Large values of \eqn{|\beta_{j,t} - \beta_{j,t-1}|} indicate regime shifts.
#'
#' @section Algorithm:
#'
#' The estimation uses Dynamic LQA (Local Quadratic Approximation) algorithm:
#' \enumerate{
#'   \item Approximate SCAD penalty using quadratic approximation at current iterate
#'   \item Solve block tridiagonal system using ADMM for fused penalty
#'   \item Iterate until convergence
#' }
#'
#' The resulting system has block tridiagonal structure:
#' \deqn{\mathbf{A} \boldsymbol{\beta} = \mathbf{b}}
#' where \eqn{\boldsymbol{\beta} = (\boldsymbol{\beta}_1^T, \ldots, \boldsymbol{\beta}_T^T)^T}
#' and \eqn{\mathbf{A}} has tridiagonal blocks connecting adjacent time points.
#'
#' @param y_list List of length T, where each element is a response vector
#'   \eqn{\mathbf{y}_t} (can be scalar or vector)
#' @param X_list List of length T, where each element is a design matrix
#'   \eqn{\mathbf{X}_t} of dimension \eqn{n_t \times p}
#' @param lambda Tuning parameter for SCAD penalty (scalar, > 0)
#' @param tau Tuning parameter for temporal smoothness/fused penalty (scalar, >= 0)
#' @param a Shape parameter for SCAD penalty, must be > 2 (default: 3.7)
#' @param beta_init Initial coefficient matrix of dimension \eqn{p \times T}
#'   (default: NULL, uses LASSO initialization)
#' @param max_iter Maximum number of iterations (default: 100)
#' @param tol Convergence tolerance (default: 1e-6)
#' @param standardize Logical, whether to standardize each \eqn{\mathbf{X}_t}
#'   (default: TRUE)
#' @param admm_rho ADMM penalty parameter for fused penalty (default: 1.0)
#' @param admm_max_iter Maximum ADMM iterations per LQA step (default: 100)
#' @param verbose Logical, whether to print progress (default: FALSE)
#'
#' @return A list containing:
#' \item{beta}{Estimated coefficient matrix of dimension \eqn{p \times T},
#'   where column \eqn{t} is \eqn{\boldsymbol{\beta}_t}}
#' \item{beta_path}{List of coefficient matrices at each iteration}
#' \item{iterations}{Number of iterations until convergence}
#' \item{converged}{Logical indicating convergence}
#' \item{objective}{Final objective value}
#' \item{objective_history}{History of objective values}
#' \item{scad_penalty}{Total SCAD penalty value}
#' \item{fused_penalty}{Total fused penalty value}
#' \item{regression_loss}{Total regression loss}
#'
#' @references
#' Fan, J., & Li, R. (2001). Variable selection via nonconcave penalized
#' likelihood and its oracle properties. \emph{Journal of the American
#' Statistical Association}, 96(456), 1348-1360.
#'
#' @examples
#' \dontrun{
#' # Generate time-varying data
#' set.seed(123)
#' T <- 50
#' p <- 10
#' n <- 30
#' 
#' # Create time-varying coefficients with structural break
#' beta_true <- matrix(0, p, T)
#' beta_true[1, 1:25] <- 2.0
#' beta_true[1, 26:T] <- -1.5
#' beta_true[2, 1:T] <- 1.5
#' beta_true[3, 15:35] <- 1.0
#' 
#' # Generate data
#' y_list <- list()
#' X_list <- list()
#' for (t in 1:T) {
#'   X_t <- matrix(rnorm(n * p), n, p)
#'   y_t <- X_t %*% beta_true[, t] + rnorm(n, sd = 0.5)
#'   y_list[[t]] <- y_t
#'   X_list[[t]] <- X_t
#' }
#' 
#' # Fit Dynamic SCAD
#' fit <- dynamic_scad(
#'   y_list = y_list,
#'   X_list = X_list,
#'   lambda = 0.5,
#'   tau = 0.3,
#'   a = 3.7
#' )
#' 
#' # Check convergence
#' print(fit$converged)
#' print(fit$iterations)
#' 
#' # Plot coefficient paths
#' matplot(t(fit$beta), type = "l", xlab = "Time", ylab = "Coefficient")
#' }
#'
#' @export
dynamic_scad <- function(y_list, X_list, lambda, tau = 0.1, a = 3.7,
                         beta_init = NULL, max_iter = 100, tol = 1e-6,
                         standardize = TRUE, admm_rho = 1.0,
                         admm_max_iter = 100, verbose = FALSE) {
  
  # Input validation
  T <- length(y_list)
  if (T != length(X_list)) {
    stop("y_list and X_list must have the same length")
  }
  if (T < 2) {
    stop("Need at least T >= 2 time points")
  }
  
  # Check dimensions
  p <- ncol(X_list[[1]])
  for (t in 1:T) {
    if (ncol(X_list[[t]]) != p) {
      stop(sprintf("X_list[[%d]] must have %d columns", t, p))
    }
    if (length(y_list[[t]]) != nrow(X_list[[t]])) {
      stop(sprintf("y_list[[%d]] length must equal nrow(X_list[[%d]])", t, t))
    }
  }
  
  if (lambda <= 0) {
    stop("lambda must be positive")
  }
  if (tau < 0) {
    stop("tau must be non-negative")
  }
  if (a <= 2) {
    stop("Parameter a must be > 2")
  }
  
  # Standardize each X_t if requested
  X_standardized <- X_list
  X_mean_list <- list()
  X_sd_list <- list()
  
  if (standardize) {
    for (t in 1:T) {
      X_mean <- colMeans(X_list[[t]])
      X_sd <- apply(X_list[[t]], 2, sd)
      X_sd[X_sd == 0] <- 1
      X_standardized[[t]] <- scale(X_list[[t]], center = X_mean, scale = X_sd)
      X_mean_list[[t]] <- X_mean
      X_sd_list[[t]] <- X_sd
    }
  } else {
    for (t in 1:T) {
      X_mean_list[[t]] <- rep(0, p)
      X_sd_list[[t]] <- rep(1, p)
    }
  }
  
  # Initialize beta (p x T matrix)
  if (is.null(beta_init)) {
    beta <- matrix(0, p, T)
    for (t in 1:T) {
      # Use LASSO for initialization
      y_t <- y_list[[t]]
      X_t <- X_standardized[[t]]
      lasso_fit <- glmnet::glmnet(X_t, y_t, lambda = lambda, alpha = 1,
                                  standardize = FALSE, intercept = FALSE)
      beta[, t] <- as.numeric(lasso_fit$beta)
    }
  } else {
    if (nrow(beta_init) != p || ncol(beta_init) != T) {
      stop(sprintf("beta_init must be %d x %d matrix", p, T))
    }
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
  compute_objective <- function(beta_mat, y_list, X_list, lambda, tau, a) {
    # Regression loss
    reg_loss <- 0
    for (t in 1:T) {
      residuals <- y_list[[t]] - X_list[[t]] %*% beta_mat[, t]
      reg_loss <- reg_loss + sum(residuals^2)
    }
    
    # SCAD penalty
    scad_pen <- 0
    for (t in 1:T) {
      scad_pen <- scad_pen + sum(scad_penalty(beta_mat[, t], lambda, a))
    }
    
    # Fused penalty
    fused_pen <- 0
    if (tau > 0 && T > 1) {
      for (t in 2:T) {
        fused_pen <- fused_pen + sum(abs(beta_mat[, t] - beta_mat[, t - 1]))
      }
    }
    
    return(list(
      regression_loss = reg_loss / 2,
      scad_penalty = lambda * scad_pen,
      fused_penalty = tau * fused_pen,
      total = reg_loss / 2 + lambda * scad_pen + tau * fused_pen
    ))
  }
  
  # Main Dynamic LQA iteration
  converged <- FALSE
  objective_history <- numeric(max_iter)
  beta_path <- list()
  
  for (iter in 1:max_iter) {
    beta_old <- beta
    obj_vals <- compute_objective(beta, y_list, X_standardized, lambda, tau, a)
    objective_history[iter] <- obj_vals$total
    
    if (verbose && iter %% 10 == 0) {
      cat(sprintf("Iteration %d: Objective = %.6f\n", iter, obj_vals$total))
    }
    
    # Store beta path
    beta_path[[iter]] <- beta
    
    # Compute LQA weights for SCAD penalty
    # w_{j,t} = SCAD'(|beta_{j,t}|) / |beta_{j,t}|
    w_mat <- matrix(0, p, T)
    epsilon <- 1e-6
    
    for (t in 1:T) {
      w_prime <- scad_derivative(beta[, t], lambda, a)
      abs_beta <- abs(beta[, t])
      
      # Small coefficients: use large penalty
      small_idx <- abs_beta < epsilon
      w_mat[small_idx, t] <- lambda / epsilon
      
      # Large coefficients: use LQA weight
      large_idx <- !small_idx
      w_mat[large_idx, t] <- w_prime[large_idx] / abs_beta[large_idx]
      
      # Cap weights
      w_mat[, t] <- pmin(w_mat[, t], lambda / epsilon)
    }
    
    # Solve block tridiagonal system using ADMM
    # This solves the quadratic approximation with fused penalty
    # Use Rcpp version if available (much faster), otherwise fall back to R version
    use_rcpp <- tryCatch({
      # Check if Rcpp function is available
      if (requireNamespace("Rcpp", quietly = TRUE)) {
        # Try to call Rcpp function
        test_call <- tryCatch({
          get("solve_dynamic_lqa_admm_cpp", envir = asNamespace("scadLLA"), inherits = FALSE)
          TRUE
        }, error = function(e) FALSE)
        test_call
      } else {
        FALSE
      }
    }, error = function(e) FALSE)
    
    if (use_rcpp) {
      # Fast C++ implementation
      beta <- solve_dynamic_lqa_admm_cpp(
        y_list = y_list,
        X_list = X_standardized,
        w_mat = w_mat,
        lambda = lambda,
        tau = tau,
        rho = admm_rho,
        max_iter = admm_max_iter,
        tol = tol / 10
      )
    } else {
      # R implementation (slower but always available)
      beta <- solve_dynamic_lqa_admm(
        y_list = y_list,
        X_list = X_standardized,
        w_mat = w_mat,
        lambda = lambda,
        tau = tau,
        rho = admm_rho,
        max_iter = admm_max_iter,
        tol = tol / 10
      )
    }
    
    # Check convergence
    max_change <- max(abs(beta - beta_old))
    if (max_change < tol) {
      converged <- TRUE
      break
    }
  }
  
  # Unstandardize coefficients if needed
  if (standardize) {
    for (t in 1:T) {
      beta[, t] <- beta[, t] / X_sd_list[[t]]
    }
  }
  
  # Final objective
  final_obj <- compute_objective(beta, y_list, X_list, lambda, tau, a)
  
  return(list(
    beta = beta,
    beta_path = beta_path,
    iterations = iter,
    converged = converged,
    objective = final_obj$total,
    objective_history = objective_history[1:iter],
    scad_penalty = final_obj$scad_penalty,
    fused_penalty = final_obj$fused_penalty,
    regression_loss = final_obj$regression_loss
  ))
}


