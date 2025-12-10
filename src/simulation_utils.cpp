#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Generate correlated design matrix: X ~ N(0, Σ), Σ_jk = ρ^{|j-k|}
// [[Rcpp::export]]
arma::mat generate_X_cpp(int n, int p, double rho) {
  // Build correlation matrix: Σ_jk = ρ^{|j-k|}
  arma::mat Sigma(p, p);
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < p; j++) {
      Sigma(i, j) = std::pow(rho, std::abs(i - j));
    }
  }
  
  // Generate X from multivariate normal
  arma::mat X = arma::randn(n, p);
  arma::mat chol_Sigma;
  bool chol_success = arma::chol(chol_Sigma, Sigma);
  if (!chol_success) {
    // If Cholesky fails, use eigendecomposition
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, Sigma);
    eigval = arma::max(eigval, arma::zeros<vec>(eigval.n_elem)); // Ensure non-negative
    arma::mat sqrt_Sigma = eigvec * arma::diagmat(arma::sqrt(eigval)) * eigvec.t();
    X = X * sqrt_Sigma.t();
  } else {
    X = X * chol_Sigma.t();
  }
  
  // Standardize columns
  for (int j = 0; j < p; j++) {
    double col_mean = arma::mean(X.col(j));
    double col_std = arma::stddev(X.col(j));
    if (col_std > 1e-10) {
      X.col(j) = (X.col(j) - col_mean) / col_std;
    }
  }
  
  return X;
}

// Generate correlated noise: ε ~ N(0, Σ_ε), (Σ_ε)_ij = σ² ρ_ε^{|i-j|}
// [[Rcpp::export]]
arma::vec generate_epsilon_cpp(int n, double sigma, double rho_eps) {
  if (rho_eps > 0 && n > 1) {
    // Build correlation matrix
    arma::mat Sigma_eps(n, n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        Sigma_eps(i, j) = (sigma * sigma) * std::pow(rho_eps, std::abs(i - j));
      }
    }
    // Ensure positive definite
    Sigma_eps.diag() += 1e-6;
    
    // Generate epsilon
    arma::mat epsilon_mat = arma::randn(1, n);
    arma::mat chol_Sigma;
    bool chol_success = arma::chol(chol_Sigma, Sigma_eps);
    if (!chol_success) {
      // If Cholesky fails, use eigendecomposition
      arma::vec eigval;
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, Sigma_eps);
      eigval = arma::max(eigval, arma::zeros<vec>(eigval.n_elem));
      arma::mat sqrt_Sigma = eigvec * arma::diagmat(arma::sqrt(eigval)) * eigvec.t();
      epsilon_mat = epsilon_mat * sqrt_Sigma.t();
    } else {
      epsilon_mat = epsilon_mat * chol_Sigma.t();
    }
    return epsilon_mat.t();
  } else {
    return sigma * arma::randn(n);
  }
}

// Compute beta error: ||β̂ - β*||₂ averaged across time points
// [[Rcpp::export]]
double compute_beta_error_cpp(const arma::mat& beta_est, const arma::mat& beta_true) {
  int T_periods = beta_est.n_cols;
  double total_error = 0.0;
  
  for (int t = 0; t < T_periods; t++) {
    arma::vec diff = beta_est.col(t) - beta_true.col(t);
    total_error += arma::norm(diff, 2);
  }
  
  return total_error / T_periods;
}

// Compute prediction MSE
// [[Rcpp::export]]
double compute_prediction_mse_cpp(const arma::mat& beta_est, 
                                   const List& X_list, const List& y_list) {
  int T_periods = y_list.size();
  double total_mse = 0.0;
  int count = 0;
  
  for (int t = 0; t < T_periods && t < (int)beta_est.n_cols; t++) {
    arma::mat X_t = as<arma::mat>(X_list[t]);
    arma::vec y_t = as<arma::vec>(y_list[t]);
    arma::vec beta_t = beta_est.col(t);
    
    arma::vec y_pred = X_t * beta_t;
    arma::vec residuals = y_t - y_pred;
    total_mse += arma::mean(arma::square(residuals));
    count++;
  }
  
  return count > 0 ? total_mse / count : 0.0;
}

// Compute covariance error: ||Σ̂ - Σ*||_F
// [[Rcpp::export]]
double compute_covariance_error_cpp(const arma::mat& beta_est, const List& X_list,
                                     const arma::mat& beta_true, double sigma, double rho_eps) {
  int T_periods = X_list.size();
  double total_error = 0.0;
  int count = 0;
  
  for (int t = 0; t < T_periods && t < (int)beta_est.n_cols; t++) {
    arma::mat X_t = as<arma::mat>(X_list[t]);
    int n = X_t.n_rows;
    int p = X_t.n_cols;
    
    // Estimated covariance: Σ̂ = X diag(β̂²) X^T + Σ_ε
    arma::vec beta_sq_est = arma::square(beta_est.col(t));
    arma::mat X_scaled_est = X_t;
    for (int j = 0; j < p; j++) {
      X_scaled_est.col(j) *= beta_sq_est(j);
    }
    arma::mat Sigma_hat = X_scaled_est * X_t.t();
    
    // True covariance: Σ* = X diag(β*²) X^T + Σ_ε
    arma::vec beta_sq_true = arma::square(beta_true.col(t));
    arma::mat X_scaled_true = X_t;
    for (int j = 0; j < p; j++) {
      X_scaled_true.col(j) *= beta_sq_true(j);
    }
    arma::mat Sigma_star = X_scaled_true * X_t.t();
    
    // Add idiosyncratic covariance
    if (rho_eps > 0 && n > 1) {
      arma::mat Sigma_eps(n, n);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          Sigma_eps(i, j) = (sigma * sigma) * std::pow(rho_eps, std::abs(i - j));
        }
      }
      Sigma_hat += Sigma_eps;
      Sigma_star += Sigma_eps;
    } else {
      Sigma_hat.diag() += sigma * sigma;
      Sigma_star.diag() += sigma * sigma;
    }
    
    // Frobenius norm error
    arma::mat diff = Sigma_hat - Sigma_star;
    total_error += arma::norm(diff, "fro");
    count++;
  }
  
  return count > 0 ? total_error / count : 0.0;
}

// Compute Mean-Variance portfolio return
// [[Rcpp::export]]
double compute_mv_return_cpp(const arma::mat& beta_est, const List& X_list,
                            const List& y_list, double sigma, double rho_eps) {
  int T_periods = y_list.size();
  arma::vec returns(T_periods);
  returns.fill(arma::datum::nan);
  
  for (int t = 0; t < T_periods && t < (int)beta_est.n_cols; t++) {
    arma::mat X_t = as<arma::mat>(X_list[t]);
    arma::vec y_t = as<arma::vec>(y_list[t]);
    int n = y_t.n_elem;
    
    if (n < 3) {
      continue;
    }
    
    // Predicted returns: r_hat = X * beta
    arma::vec beta_t = beta_est.col(t);
    arma::vec r_hat = X_t * beta_t;
    
    // Estimate covariance
    arma::vec beta_sq = arma::square(beta_t);
    arma::mat X_scaled = X_t;
    for (int j = 0; j < (int)beta_sq.n_elem; j++) {
      X_scaled.col(j) *= beta_sq(j);
    }
    arma::mat Sigma = X_scaled * X_t.t();
    
    // Add idiosyncratic covariance
    if (rho_eps > 0 && n > 1) {
      arma::mat Sigma_eps(n, n);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          Sigma_eps(i, j) = (sigma * sigma) * std::pow(rho_eps, std::abs(i - j));
        }
      }
      Sigma += Sigma_eps;
    } else {
      Sigma.diag() += sigma * sigma;
    }
    
    // Mean-variance weights: w = (Σ^(-1) μ) / (1^T Σ^(-1) μ)
    try {
      arma::mat Sigma_inv = arma::inv(Sigma);
      arma::vec w_raw = Sigma_inv * r_hat;
      double w_sum = arma::sum(w_raw);
      
      if (std::abs(w_sum) > 1e-8 && std::isfinite(w_sum)) {
        arma::vec weights = w_raw / w_sum;
        // Long-only constraint
        weights = arma::max(weights, arma::zeros<vec>(n));
        double w_sum_pos = arma::sum(weights);
        if (w_sum_pos > 1e-8) {
          weights = weights / w_sum_pos;
          returns(t) = arma::dot(weights, y_t);
        }
      }
    } catch (...) {
      // If inversion fails, skip this time point
    }
  }
  
  // Return mean of valid returns
  arma::vec valid_returns = returns.elem(arma::find_finite(returns));
  return valid_returns.n_elem > 0 ? arma::mean(valid_returns) : 0.0;
}

// Generate time series data (full function in C++)
// [[Rcpp::export]]
List generate_time_series_data_cpp(int T_periods, int n_per_period, int p, 
                                   double rho, double rho_eps, double sigma) {
  // Create true beta base: (3, 3, 2, 1.5, 1, 0, ..., 0)
  arma::vec beta_star_base(p);
  beta_star_base.zeros();
  if (p >= 1) beta_star_base(0) = 3.0;
  if (p >= 2) beta_star_base(1) = 3.0;
  if (p >= 3) beta_star_base(2) = 2.0;
  if (p >= 4) beta_star_base(3) = 1.5;
  if (p >= 5) beta_star_base(4) = 1.0;
  
  // Create time-varying coefficients with structural break
  arma::mat beta_true(p, T_periods);
  int break_point = T_periods / 2;
  
  for (int t = 0; t < T_periods; t++) {
    if (t < break_point) {
      beta_true.col(t) = beta_star_base;
    } else {
      // Structural break: coefficients change more dramatically
      // First two factors become smaller, others become zero
      arma::vec beta_break(p);
      beta_break.zeros();
      if (p >= 1) beta_break(0) = 1.0;
      if (p >= 2) beta_break(1) = 1.0;
      beta_true.col(t) = beta_break;
    }
  }
  
  // Generate time series data
  List y_list(T_periods);
  List X_list(T_periods);
  
  for (int t = 0; t < T_periods; t++) {
    arma::mat X_t = generate_X_cpp(n_per_period, p, rho);
    arma::vec epsilon_t = generate_epsilon_cpp(n_per_period, sigma, rho_eps);
    arma::vec y_t = X_t * beta_true.col(t) + epsilon_t;
    
    y_list[t] = y_t;
    X_list[t] = X_t;
  }
  
  return List::create(
    Named("y_list") = y_list,
    Named("X_list") = X_list,
    Named("beta_true") = beta_true
  );
}

