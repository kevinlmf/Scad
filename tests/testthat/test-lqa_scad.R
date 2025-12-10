test_that("lqa_scad works on simple data", {
  set.seed(123)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(3, 2, 0, 0, 1, rep(0, p - 5))
  y <- X %*% beta_true + rnorm(n)
  
  fit <- lqa_scad(y, X, lambda = 0.5)
  
  expect_true(is.list(fit))
  expect_equal(length(fit$beta), p)
  expect_true(fit$converged)
  # Check if it picked up the non-zero coefficients (roughly)
  expect_true(abs(fit$beta[1]) > 0.1)
})

test_that("lqa_scad_improved works", {
  set.seed(456)
  n <- 50
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- rnorm(n) # Random noise
  
  fit <- lqa_scad_improved(y, X, lambda = 1.0, decomposition = "qr")
  
  expect_true(fit$converged)
  expect_equal(length(fit$beta), p)
})
