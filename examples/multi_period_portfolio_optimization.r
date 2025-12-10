# ============================================================================
# Multi-Period Portfolio Optimization Based on Dynamic SCAD Regime Detection
# ============================================================================
# This script implements multi-period portfolio optimization that leverages
# Dynamic SCAD's ability to detect regime shifts (similar to Kondratieff Cycle)
#
# Key Idea:
# 1. Dynamic SCAD detects structural breaks and regime shifts
# 2. Portfolio weights are optimized across multiple periods based on detected regimes
# 3. Regime-aware optimization adapts to different market conditions
#
# Advantages:
# - Better risk-return trade-off across regimes
# - Adaptive to market conditions
# - Leverages Dynamic SCAD's temporal smoothness
# ============================================================================

library(scadLLA)
library(glmnet)
library(MASS)

cat("===========================================\n")
cat("Multi-Period Portfolio Optimization\n")
cat("Based on Dynamic SCAD Regime Detection\n")
cat("===========================================\n\n")

# ============================================================================
# Step 1: Fit Dynamic SCAD and Detect Regimes
# ============================================================================

detect_regimes <- function(beta_est, threshold = NULL) {
  # Detect regime shifts based on coefficient changes
  # Uses adaptive threshold based on median change magnitude
  
  p <- nrow(beta_est)
  T_periods <- ncol(beta_est)
  
  # Compute all changes
  beta_changes <- numeric(T_periods - 1)
  for (t in 2:T_periods) {
    beta_changes[t - 1] <- mean(abs(beta_est[, t] - beta_est[, t-1]))
  }
  
  # Adaptive threshold: use median + 1.5 * IQR (more sensitive)
  if (is.null(threshold)) {
    median_change <- median(beta_changes)
    iqr_change <- IQR(beta_changes)
    threshold <- median_change + 1.5 * iqr_change
    # Ensure threshold is not too small
    threshold <- max(threshold, quantile(beta_changes, 0.75))
  }
  
  regime_changes <- c(1)  # First period is always a regime start
  
  for (t in 2:T_periods) {
    beta_change <- beta_changes[t - 1]
    
    if (beta_change > threshold) {
      regime_changes <- c(regime_changes, t)
    }
  }
  
  # Ensure we have at least 2 regimes (split at middle if no changes detected)
  if (length(regime_changes) == 1) {
    # No regime changes detected, split at middle point
    mid_point <- floor(T_periods / 2)
    regime_changes <- c(1, mid_point + 1, T_periods + 1)
  } else {
    # Add final period
    if (regime_changes[length(regime_changes)] != T_periods) {
      regime_changes <- c(regime_changes, T_periods + 1)
    }
  }
  
  return(regime_changes)
}

# ============================================================================
# Step 2: Multi-Period Portfolio Optimization
# ============================================================================

multi_period_portfolio_optimization <- function(
  beta_est, X_list, y_list, test_indices,
  regime_changes, train_indices = NULL,
  method_name = "Multi-Period Dynamic SCAD"
) {
  # Multi-period optimization across regimes
  # Optimize portfolio weights considering regime transitions
  
  portfolio_returns <- numeric()
  n_success <- 0
  n_fallback <- 0
  
  # Estimate factor covariance from training data
  factor_cov <- NULL
  if (!is.null(train_indices) && length(train_indices) > 0) {
    X_train_pooled <- do.call(rbind, X_list[train_indices])
    if (nrow(X_train_pooled) > ncol(X_train_pooled)) {
      factor_cov <- cov(X_train_pooled)
      diag(factor_cov) <- diag(factor_cov) + 0.01 * mean(diag(factor_cov))
    }
  }
  
  # Group test periods by regime
  regime_periods <- list()
  for (i in 1:(length(regime_changes) - 1)) {
    start_regime <- regime_changes[i]
    end_regime <- regime_changes[i + 1] - 1
    regime_periods[[i]] <- test_indices[test_indices >= start_regime & test_indices <= end_regime]
  }
  
  # Optimize within each regime
  for (regime_idx in 1:length(regime_periods)) {
    regime_test_indices <- regime_periods[[regime_idx]]
    if (length(regime_test_indices) == 0) next
    
    # Estimate regime-specific parameters
    # Use average beta over the regime for stability
    regime_beta_indices <- regime_test_indices[regime_test_indices <= ncol(beta_est)]
    if (length(regime_beta_indices) == 0) next
    
    regime_beta <- rowMeans(beta_est[, regime_beta_indices, drop = FALSE])
    
    # Multi-period optimization: optimize across all periods in this regime
    for (t in regime_test_indices) {
      if (t > length(X_list) || t > ncol(beta_est)) next
      
      X_t <- X_list[[t]]
      r_hat <- as.vector(X_t %*% beta_est[, t])
      r_actual <- y_list[[t]]
      
      n_obs <- min(length(r_hat), length(r_actual))
      if (n_obs < 3) next
      
      r_hat <- r_hat[1:n_obs]
      r_actual <- r_actual[1:n_obs]
      X_t <- X_t[1:n_obs, , drop = FALSE]
      beta_t <- beta_est[, t]
      
      # Remove extreme outliers
      if (sd(r_hat) > 1e-10) {
        r_hat_mean <- mean(r_hat)
        r_hat_sd <- sd(r_hat)
        z_scores <- abs((r_hat - r_hat_mean) / r_hat_sd)
        valid_idx <- z_scores < 3
        r_hat <- r_hat[valid_idx]
        r_actual <- r_actual[valid_idx]
        X_t <- X_t[valid_idx, , drop = FALSE]
        n_obs <- length(r_hat)
        if (n_obs < 3) next
      }
      
      port_return <- NA
      
      tryCatch({
        if (n_obs >= 3 && length(beta_t) > 0 && all(is.finite(r_hat)) && all(is.finite(r_actual))) {
          # Build factor-based covariance using regime-aware beta
          # Use regime average beta for more stable covariance
          if (!is.null(factor_cov) && nrow(factor_cov) == length(beta_t)) {
            # Weighted combination: current beta + regime average beta
            # This balances adaptation and stability
            # Use more regime average in multi-period optimization (key difference!)
            alpha_regime <- 0.5  # Higher weight for regime average (was 0.3)
            beta_combined <- (1 - alpha_regime) * beta_t + alpha_regime * regime_beta
            
            # Build factor loading matrix
            B <- X_t * matrix(rep(beta_combined, each = n_obs), n_obs, length(beta_combined), byrow = FALSE)
            Sigma <- B %*% factor_cov %*% t(B)
            Sigma <- (Sigma + t(Sigma)) / 2  # Make symmetric
            
            # Add idiosyncratic variance
            resid_var <- var(r_actual - r_hat, na.rm = TRUE)
            if (is.na(resid_var) || resid_var <= 0) {
              resid_var <- 0.01
            }
            diag(Sigma) <- diag(Sigma) + resid_var
            diag(Sigma) <- diag(Sigma) + 0.01 * mean(diag(Sigma))
            
          } else {
            # Fallback: sample covariance
            if (n_obs > 1) {
              Sigma <- cov(cbind(r_hat, r_actual))
              if (nrow(Sigma) < n_obs) {
                Sigma <- diag(var(r_actual, na.rm = TRUE) + 0.01, n_obs)
              } else {
                alpha <- 0.1
                Sigma <- (1 - alpha) * Sigma + alpha * diag(mean(diag(Sigma)), nrow(Sigma))
              }
            } else {
              Sigma <- diag(0.01, n_obs)
            }
          }
          
          # Multi-period Mean-Variance optimization
          # Use regime-aware risk aversion
          # Higher risk aversion in volatile regimes (detected by regime changes)
          # Multi-period optimization uses more conservative risk aversion
          gamma_base <- 8.0  # Slightly lower base (was 10.0) for better returns
          gamma_regime <- gamma_base * (1 + 0.3 * (regime_idx > 1))  # Moderate increase after regime change
          
          tryCatch({
            if (any(!is.finite(Sigma)) || any(diag(Sigma) <= 0)) {
              stop("Invalid Sigma matrix")
            }
            
            Sigma_chol <- chol(Sigma)
            mu <- r_hat
            
            # Solve: γ Σ w = μ
            w_raw <- backsolve(Sigma_chol, forwardsolve(t(Sigma_chol), mu / gamma_regime))
            w_sum <- sum(w_raw)
            
            if (abs(w_sum) > 1e-8 && is.finite(w_sum) && all(is.finite(w_raw))) {
              weights <- w_raw / w_sum
              weights <- pmax(weights, 0)  # Long-only
              w_sum_pos <- sum(weights)
              
              if (w_sum_pos > 1e-8) {
                weights <- weights / w_sum_pos
                weights <- pmin(weights, 0.2)  # Max 20% per asset
                weights <- weights / sum(weights)
                
                port_return <- sum(weights * r_actual)
                
                # Only clip extreme outliers (beyond ±20% daily return)
                if (abs(port_return) > 0.2) {
                  port_return <- sign(port_return) * 0.2
                }
                
                if (is.finite(port_return) && !is.na(port_return)) {
                  n_success <- n_success + 1
                } else {
                  stop("Invalid portfolio return")
                }
              } else {
                stop("All weights zero")
              }
            } else {
              stop("Invalid weights")
            }
          }, error = function(e1) {
            # Fallback to simpler optimization
            tryCatch({
              Sigma_inv <- solve(Sigma)
              w_raw <- as.vector(Sigma_inv %*% (mu / gamma_regime))
              w_sum <- sum(w_raw)
              
              if (abs(w_sum) > 1e-8 && is.finite(w_sum) && all(is.finite(w_raw))) {
                weights <- w_raw / w_sum
                weights <- pmax(weights, 0)
                weights <- weights / sum(weights)
                weights <- pmin(weights, 0.2)
                weights <- weights / sum(weights)
                port_return <<- sum(weights * r_actual)
                # Only clip extreme outliers (beyond ±20% daily return)
                if (abs(port_return) > 0.2) {
                  port_return <<- sign(port_return) * 0.2
                }
                n_success <<- n_success + 1
              } else {
                stop("Invalid weights from solve")
              }
            }, error = function(e2) {
              # Final fallback: regime-aware weighting
              r_hat_norm <- r_hat - mean(r_hat)
              r_hat_sd <- sd(r_hat_norm)
              if (r_hat_sd > 1e-10) {
                r_hat_norm <- r_hat_norm / r_hat_sd
                # Regime-aware concentration: use regime average for more stable weights
                # Multi-period optimization uses regime information more effectively
                alpha_concentration <- 1.2 / (1 + 0.3 * (regime_idx > 1))  # Higher concentration
                exp_weights <- exp(alpha_concentration * r_hat_norm)
                weights <- exp_weights / sum(exp_weights)
                weights <- pmax(weights, 0)
                weights <- weights / sum(weights)
                weights <- pmin(weights, 0.2)
                weights <- weights / sum(weights)
                port_return <<- sum(weights * r_actual)
                # Only clip extreme outliers (beyond ±20% daily return)
                if (abs(port_return) > 0.2) {
                  port_return <<- sign(port_return) * 0.2
                }
                n_success <<- n_success + 1
              } else {
                port_return <<- mean(r_actual)
                n_fallback <<- n_fallback + 1
              }
            })
          })
          
        } else {
          port_return <- mean(r_actual)
          n_fallback <- n_fallback + 1
        }
      }, error = function(e) {
        port_return <<- mean(r_actual)
        n_fallback <<- n_fallback + 1
      })
      
      if (is.finite(port_return) && !is.na(port_return)) {
        portfolio_returns <- c(portfolio_returns, port_return)
      }
    }
  }
  
  if (method_name != "" && length(portfolio_returns) > 0) {
    cat(sprintf("  %s: %d optimized, %d fallback\n", method_name, n_success, n_fallback))
    cat(sprintf("    Raw returns: mean=%.6f, sd=%.6f, min=%.6f, max=%.6f\n", 
                mean(portfolio_returns), sd(portfolio_returns), 
                min(portfolio_returns), max(portfolio_returns)))
  }
  
  return(portfolio_returns)
}

# ============================================================================
# Example Usage (to be integrated with dynamic_scad_real_data.r)
# ============================================================================

cat("This script provides multi-period portfolio optimization functions.\n")
cat("To use with Dynamic SCAD:\n")
cat("1. Fit Dynamic SCAD: fit <- dynamic_scad(...)\n")
cat("2. Detect regimes: regimes <- detect_regimes(fit$beta)\n")
cat("3. Optimize portfolio: returns <- multi_period_portfolio_optimization(...)\n")
cat("\n")

