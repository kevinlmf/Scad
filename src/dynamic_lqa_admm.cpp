#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// Soft thresholding operator: S_lambda(x) = sign(x) * max(|x| - lambda, 0)
// [[Rcpp::export]]
arma::vec soft_threshold_cpp(const arma::vec& x, double lambda) {
  arma::vec result = arma::sign(x) % arma::max(arma::abs(x) - lambda, arma::zeros<vec>(x.n_elem));
  return result;
}

// Fast ADMM solver for Dynamic SCAD
// This is the computationally intensive part that benefits from Rcpp
// [[Rcpp::export]]
arma::mat solve_dynamic_lqa_admm_cpp(const List& y_list, const List& X_list, 
                                     const arma::mat& w_mat, double lambda, 
                                     double tau, double rho, int max_iter, 
                                     double tol) {
  
  int T = y_list.size();
  if (T == 0) {
    return arma::mat();
  }
  
  arma::mat X0 = as<arma::mat>(X_list[0]);
  int p = X0.n_cols;
  
  // Initialize
  arma::mat beta = arma::zeros<arma::mat>(p, T);
  arma::mat z = arma::zeros<arma::mat>(p, T - 1);
  arma::mat z_old = z;
  arma::mat u = arma::zeros<arma::mat>(p, T - 1);
  
  // Precompute X^T X and X^T y for each time point (much faster in C++)
  std::vector<arma::mat> XtX_list(T);
  std::vector<arma::vec> Xty_list(T);
  
  for (int t = 0; t < T; t++) {
    arma::mat X_t = as<arma::mat>(X_list[t]);
    arma::vec y_t = as<arma::vec>(y_list[t]);
    XtX_list[t] = X_t.t() * X_t;
    Xty_list[t] = X_t.t() * y_t;
  }
  
  // ADMM iterations
  for (int admm_iter = 0; admm_iter < max_iter; admm_iter++) {
    arma::mat beta_old = beta;
    
    // Update beta: solve for each time point
    for (int t = 0; t < T; t++) {
      // Build augmented system: (X_t^T X_t + diag(w_{.,t}) + rho*penalty) beta_t = rhs
      arma::mat H_t = XtX_list[t];
      arma::vec w_t = w_mat.col(t);
      
      // Add diagonal weights
      H_t.diag() += w_t;
      
      // Build right-hand side
      arma::vec rhs_t = Xty_list[t];
      
      // Add ADMM penalty terms
      if (t == 0) {
        // First time point: only constraint with t+1
        H_t.diag() += rho;
        rhs_t += rho * (beta_old.col(t + 1) - z.col(t) + u.col(t));
      } else if (t == T - 1) {
        // Last time point: only constraint with t-1
        H_t.diag() += rho;
        rhs_t += rho * (beta_old.col(t - 1) + z.col(t - 1) - u.col(t - 1));
      } else {
        // Middle time points: constraints with both t-1 and t+1
        H_t.diag() += 2.0 * rho;
        rhs_t += rho * (beta_old.col(t - 1) + z.col(t - 1) - u.col(t - 1) +
                       beta_old.col(t + 1) - z.col(t) + u.col(t));
      }
      
      // Solve: H_t beta_t = rhs_t (using Cholesky decomposition for speed)
      beta.col(t) = arma::solve(H_t, rhs_t, arma::solve_opts::fast);
    }
    
    // Update z: soft thresholding (vectorized)
    for (int t = 0; t < T - 1; t++) {
      arma::vec diff = beta.col(t + 1) - beta.col(t) + u.col(t);
      z.col(t) = soft_threshold_cpp(diff, tau / rho);
    }
    
    // Update u: dual variable (vectorized)
    for (int t = 0; t < T - 1; t++) {
      u.col(t) += (beta.col(t + 1) - beta.col(t) - z.col(t));
    }
    
    // Check convergence (vectorized)
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    
    for (int t = 0; t < T - 1; t++) {
      arma::vec primal_diff = beta.col(t + 1) - beta.col(t) - z.col(t);
      double primal_max = arma::max(arma::abs(primal_diff));
      if (primal_max > primal_residual) {
        primal_residual = primal_max;
      }
      
      arma::vec dual_diff = rho * (z.col(t) - z_old.col(t));
      double dual_max = arma::max(arma::abs(dual_diff));
      if (dual_max > dual_residual) {
        dual_residual = dual_max;
      }
    }
    
    z_old = z;
    
    if (std::max(primal_residual, dual_residual) < tol) {
      break;
    }
  }
  
  return beta;
}

