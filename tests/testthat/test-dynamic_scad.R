# Test Dynamic SCAD Model

test_that("dynamic_scad basic functionality", {
  # Generate simple test data
  set.seed(123)
  T <- 10
  p <- 5
  n <- 20
  
  beta_true <- matrix(0, p, T)
  beta_true[1, 1:T] <- 1.0
  beta_true[2, 1:T] <- 0.5
  
  y_list <- list()
  X_list <- list()
  
  for (t in 1:T) {
    X_t <- matrix(rnorm(n * p), n, p)
    y_t <- X_t %*% beta_true[, t] + rnorm(n, sd = 0.3)
    y_list[[t]] <- y_t
    X_list[[t]] <- X_t
  }
  
  # Fit model
  fit <- dynamic_scad(
    y_list = y_list,
    X_list = X_list,
    lambda = 0.3,
    tau = 0.1,
    a = 3.7,
    max_iter = 20
  )
  
  # Check output structure
  expect_true(is.list(fit))
  expect_true("beta" %in% names(fit))
  expect_true("converged" %in% names(fit))
  expect_true("iterations" %in% names(fit))
  expect_true("objective" %in% names(fit))
  
  # Check dimensions
  expect_equal(nrow(fit$beta), p)
  expect_equal(ncol(fit$beta), T)
  
  # Check convergence
  expect_true(is.logical(fit$converged))
  expect_true(fit$iterations > 0)
  expect_true(fit$iterations <= 20)
})

test_that("dynamic_scad handles tau = 0 (no temporal smoothness)", {
  set.seed(456)
  T <- 5
  p <- 3
  n <- 15
  
  y_list <- list()
  X_list <- list()
  
  for (t in 1:T) {
    X_t <- matrix(rnorm(n * p), n, p)
    y_t <- rnorm(n)
    y_list[[t]] <- y_t
    X_list[[t]] <- X_t
  }
  
  # Fit with tau = 0
  fit <- dynamic_scad(
    y_list = y_list,
    X_list = X_list,
    lambda = 0.2,
    tau = 0,  # No temporal smoothness
    a = 3.7,
    max_iter = 15
  )
  
  expect_true(fit$converged || fit$iterations == 15)
  expect_equal(nrow(fit$beta), p)
  expect_equal(ncol(fit$beta), T)
})

test_that("dynamic_scad input validation", {
  set.seed(789)
  T <- 3
  p <- 2
  n <- 10
  
  y_list <- list()
  X_list <- list()
  
  for (t in 1:T) {
    X_t <- matrix(rnorm(n * p), n, p)
    y_t <- rnorm(n)
    y_list[[t]] <- y_t
    X_list[[t]] <- X_t
  }
  
  # Test dimension mismatch
  y_list_wrong <- y_list
  y_list_wrong[[1]] <- c(y_list[[1]], 1)  # Wrong length
  
  expect_error(
    dynamic_scad(y_list_wrong, X_list, lambda = 0.2, tau = 0.1),
    "y_list\\[\\[1\\]\\] length must equal"
  )
  
  # Test lambda <= 0
  expect_error(
    dynamic_scad(y_list, X_list, lambda = -0.1, tau = 0.1),
    "lambda must be positive"
  )
  
  # Test tau < 0
  expect_error(
    dynamic_scad(y_list, X_list, lambda = 0.2, tau = -0.1),
    "tau must be non-negative"
  )
  
  # Test a <= 2
  expect_error(
    dynamic_scad(y_list, X_list, lambda = 0.2, tau = 0.1, a = 1.5),
    "Parameter a must be > 2"
  )
  
  # Test T < 2
  expect_error(
    dynamic_scad(y_list[1], X_list[1], lambda = 0.2, tau = 0.1),
    "Need at least T >= 2"
  )
})

test_that("dynamic_scad with custom initialization", {
  set.seed(321)
  T <- 5
  p <- 4
  n <- 12
  
  y_list <- list()
  X_list <- list()
  
  for (t in 1:T) {
    X_t <- matrix(rnorm(n * p), n, p)
    y_t <- rnorm(n)
    y_list[[t]] <- y_t
    X_list[[t]] <- X_t
  }
  
  # Custom initialization
  beta_init <- matrix(rnorm(p * T), p, T)
  
  fit <- dynamic_scad(
    y_list = y_list,
    X_list = X_list,
    lambda = 0.3,
    tau = 0.15,
    beta_init = beta_init,
    max_iter = 15
  )
  
  expect_equal(nrow(fit$beta), p)
  expect_equal(ncol(fit$beta), T)
})


