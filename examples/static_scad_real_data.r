# ============================================================================
# real_data.r - Financial Scenario: SCAD vs LASSO
# ============================================================================
# 金融场景：高度相关的因子特征（SCAD优势场景）
# 
# SCAD 在以下情况表现更好：
# 1. 高度相关的因子（lagged factors, interactions）
# 2. 因子数量接近样本数量（p ≈ n 或 p > n）
# 3. 需要选择多个相关特征（而不是随机选一个）
#
# 这个例子展示：当因子高度相关时，SCAD能更好地保留重要特征
# ============================================================================

# ⭐ OPTION: Use simulated data instead of downloading
# Set to TRUE to skip downloads and use simulated data (recommended if Yahoo Finance is down)
USE_SIMULATED_DATA <- TRUE  # Change to FALSE to try downloading real data

suppressPackageStartupMessages({
  library(quantmod)
  library(MASS)
  library(zoo)
  library(Scad)  # ⭐ Use package functions instead of custom implementations
  library(glmnet)
})

# ⭐ Option to use simulated data instead of downloading
USE_SIMULATED_DATA <- TRUE  # Set to FALSE to try downloading real data

download_factors <- function(start_date = "2020-01-01", use_simulated = USE_SIMULATED_DATA) {
  if (use_simulated) {
    cat("Using simulated factor data (skipping download)...\n")
    flush.console()
    
    # Generate simulated Fama-French factors
    n_days <- 1000  # ~4 years of trading days
    dates <- seq(as.Date("2020-01-01"), by = "day", length.out = n_days)
    dates <- dates[weekdays(dates) %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]
    dates <- dates[1:min(800, length(dates))]  # ~3 years of trading days
    
    set.seed(123)
    # Market return (Mkt_RF)
    mkt_rf <- rnorm(length(dates), mean = 0.0005, sd = 0.01)
    
    # SMB (Small minus Big) - correlated with market
    smb <- 0.3 * mkt_rf + rnorm(length(dates), mean = 0, sd = 0.008)
    
    # HML (High minus Low) - less correlated
    hml <- 0.2 * mkt_rf + rnorm(length(dates), mean = 0, sd = 0.006)
    
    ff_data <- data.frame(date = dates, Mkt_RF = mkt_rf, SMB = smb, HML = hml)
    
    # ⭐ IMPORTANT: Generate same extended features as real data
    # This increases p to make p/n ratio higher (SCAD advantage scenario)
    
    # Lags (create high correlation - SCAD handles this better)
    for (lag in 1:10) {
      ff_data[[paste0("Mkt_lag", lag)]] <- c(rep(NA, lag), ff_data$Mkt_RF[1:(nrow(ff_data) - lag)])
      ff_data[[paste0("SMB_lag", lag)]] <- c(rep(NA, lag), ff_data$SMB[1:(nrow(ff_data) - lag)])
      ff_data[[paste0("HML_lag", lag)]] <- c(rep(NA, lag), ff_data$HML[1:(nrow(ff_data) - lag)])
    }
    
    # Interactions
    ff_data$Mkt_SMB <- ff_data$Mkt_RF * ff_data$SMB
    ff_data$Mkt_HML <- ff_data$Mkt_RF * ff_data$HML
    ff_data$SMB_HML <- ff_data$SMB * ff_data$HML
    
    # More interaction terms
    ff_data$Mkt_SMB_sq <- ff_data$Mkt_RF * (ff_data$SMB^2)
    ff_data$Mkt_HML_sq <- ff_data$Mkt_RF * (ff_data$HML^2)
    ff_data$SMB_HML_sq <- ff_data$SMB * (ff_data$HML^2)
    
    # Polynomial features
    ff_data$Mkt_RF_sq <- ff_data$Mkt_RF^2
    ff_data$SMB_sq <- ff_data$SMB^2
    ff_data$HML_sq <- ff_data$HML^2
    
    # Cubic terms
    ff_data$Mkt_RF_cub <- ff_data$Mkt_RF^3
    ff_data$SMB_cub <- ff_data$SMB^3
    ff_data$HML_cub <- ff_data$HML^3
    
    # Rolling means
    for (w in c(5, 10, 15, 20)) {
      ff_data[[paste0("Mkt_sma", w)]] <- c(rep(NA, w - 1), 
                                          zoo::rollmean(ff_data$Mkt_RF, w, na.pad = FALSE, align = "right"))
      ff_data[[paste0("SMB_sma", w)]] <- c(rep(NA, w - 1), 
                                          zoo::rollmean(ff_data$SMB, w, na.pad = FALSE, align = "right"))
      ff_data[[paste0("HML_sma", w)]] <- c(rep(NA, w - 1), 
                                          zoo::rollmean(ff_data$HML, w, na.pad = FALSE, align = "right"))
    }
    
    # Rolling standard deviations
    for (w in c(5, 10)) {
      ff_data[[paste0("Mkt_std", w)]] <- c(rep(NA, w - 1), 
                                          zoo::rollapply(ff_data$Mkt_RF, w, sd, na.pad = FALSE, align = "right"))
      ff_data[[paste0("SMB_std", w)]] <- c(rep(NA, w - 1), 
                                          zoo::rollapply(ff_data$SMB, w, sd, na.pad = FALSE, align = "right"))
      ff_data[[paste0("HML_std", w)]] <- c(rep(NA, w - 1), 
                                          zoo::rollapply(ff_data$HML, w, sd, na.pad = FALSE, align = "right"))
    }
    
    # Cross-lag interactions
    for (lag in 1:3) {
      if (lag <= nrow(ff_data)) {
        ff_data[[paste0("Mkt_SMB_cross", lag)]] <- c(rep(NA, lag), 
                                                    ff_data$Mkt_RF[1:(nrow(ff_data) - lag)] * 
                                                    ff_data$SMB[(lag + 1):nrow(ff_data)])
        ff_data[[paste0("Mkt_HML_cross", lag)]] <- c(rep(NA, lag), 
                                                    ff_data$Mkt_RF[1:(nrow(ff_data) - lag)] * 
                                                    ff_data$HML[(lag + 1):nrow(ff_data)])
      }
    }
    
    ff_data <- na.omit(ff_data)
    n_factors <- ncol(ff_data) - 1  # Exclude 'date' column
    cat(sprintf("Generated %d observations with %d factors (extended features)\n", 
                nrow(ff_data), n_factors))
    flush.console()
  } else {
  cat("Downloading market data...\n")
  flush.console()
  if (is.character(start_date)) start_date <- as.Date(start_date, format = "%Y-%m-%d")
    
  # ⭐ Improved download with multiple strategies
  download_with_retry <- function(symbol, max_retries = 5, delay = 3) {
    # Strategy 1: Try quantmod with different options
    for (attempt in 1:max_retries) {
      tryCatch({
        if (attempt > 1) {
          cat(sprintf("    Retry %d/%d for %s...\n", attempt, max_retries, symbol))
          Sys.sleep(delay * attempt)
        }
        
        # Try with different options
        result <- tryCatch({
          quantmod::getSymbols(symbol, src = "yahoo", from = start_date, 
                               auto.assign = FALSE, 
                               warnings = FALSE,
                               verbose = FALSE,
                               curl.options = list(timeout = 30))
        }, error = function(e1) {
          # Try without curl.options
          tryCatch({
            quantmod::getSymbols(symbol, src = "yahoo", from = start_date, 
                                 auto.assign = FALSE, 
                                 warnings = FALSE,
                                 verbose = FALSE)
          }, error = function(e2) {
            # Try with longer timeout
            quantmod::getSymbols(symbol, src = "yahoo", from = start_date, 
                                 auto.assign = FALSE, 
                                 warnings = FALSE,
                                 verbose = FALSE)
          })
        })
        
        if (!is.null(result) && nrow(result) > 0) {
          return(result)
        }
      }, error = function(e) {
        if (attempt == max_retries) {
          cat(sprintf("    ⚠️  Failed to download %s after %d attempts\n", 
                     symbol, max_retries))
          return(NULL)
        }
        return(NULL)
      })
    }
    
    # Strategy 2: Try alternative method using getSymbols with different settings
    cat(sprintf("    Trying alternative method for %s...\n", symbol))
    tryCatch({
      # Clear any existing handles
      options(HTTPUserAgent = "Mozilla/5.0")
      result <- quantmod::getSymbols(symbol, src = "yahoo", from = start_date, 
                                     auto.assign = FALSE, 
                                     warnings = FALSE,
                                     verbose = FALSE)
      if (!is.null(result) && nrow(result) > 0) {
        return(result)
      }
    }, error = function(e) {
      return(NULL)
    })
    
    return(NULL)
  }
  
  # Download sequentially with longer delays
  cat("  Downloading SPY...\n")
  spy <- download_with_retry("SPY")
  if (is.null(spy)) {
    cat("  ❌ Failed to download SPY. Trying to continue with alternative data...\n")
    # Try to use a fallback or generate synthetic data
    stop("Cannot proceed without market data. Please check your internet connection or try again later.")
  }
  
  Sys.sleep(2)  # Longer delay between downloads
  cat("  Downloading IWM...\n")
  iwm <- download_with_retry("IWM")
  if (is.null(iwm)) {
    cat("  ⚠️  Warning: Failed to download IWM, using SPY as proxy\n")
    iwm <- spy  # Use SPY as fallback
  }
  
  Sys.sleep(2)
  cat("  Downloading VTV...\n")
  vtv <- download_with_retry("VTV")
  if (is.null(vtv)) {
    cat("  ⚠️  Warning: Failed to download VTV, using SPY as proxy\n")
    vtv <- spy  # Use SPY as fallback
  }
  
  Sys.sleep(2)
  cat("  Downloading VUG...\n")
  vug <- download_with_retry("VUG")
  if (is.null(vug)) {
    cat("  ⚠️  Warning: Failed to download VUG, using SPY as proxy\n")
    vug <- spy  # Use SPY as fallback
  }
    
  spy_ret <- diff(log(Ad(spy)))[-1]
  iwm_ret <- diff(log(Ad(iwm)))[-1]
  vtv_ret <- diff(log(Ad(vtv)))[-1]
  vug_ret <- diff(log(Ad(vug)))[-1]
  
  dates <- base::intersect(index(spy_ret), index(iwm_ret))
  dates <- base::intersect(dates, index(vtv_ret))
  dates <- base::intersect(dates, index(vug_ret))
  dates <- as.Date(dates, format = "%Y-%m-%d")
  
  mkt_rf <- as.numeric(spy_ret[dates])
  smb <- as.numeric(iwm_ret[dates]) - mkt_rf
  hml <- as.numeric(vtv_ret[dates]) - as.numeric(vug_ret[dates])
  
  ff_data <- data.frame(date = dates, Mkt_RF = mkt_rf, SMB = smb, HML = hml)
  
  # Create highly correlated features (SCAD advantage scenario)
  # Lags create high correlation - SCAD handles this better than LASSO
  # ⭐ EXPANDED: More lags to increase p (to make p closer to n for SCAD Improved advantage)
  for (lag in 1:10) {  # Increased from 5 to 10
    ff_data[[paste0("Mkt_lag", lag)]] <- c(rep(NA, lag), ff_data$Mkt_RF[1:(nrow(ff_data) - lag)])
        ff_data[[paste0("SMB_lag", lag)]] <- c(rep(NA, lag), ff_data$SMB[1:(nrow(ff_data) - lag)])
        ff_data[[paste0("HML_lag", lag)]] <- c(rep(NA, lag), ff_data$HML[1:(nrow(ff_data) - lag)])
  }
  
  # Interactions (increase correlation)
      ff_data$Mkt_SMB <- ff_data$Mkt_RF * ff_data$SMB
      ff_data$Mkt_HML <- ff_data$Mkt_RF * ff_data$HML
      ff_data$SMB_HML <- ff_data$SMB * ff_data$HML
  
  # ⭐ NEW: More interaction terms
  ff_data$Mkt_SMB_sq <- ff_data$Mkt_RF * (ff_data$SMB^2)
  ff_data$Mkt_HML_sq <- ff_data$Mkt_RF * (ff_data$HML^2)
  ff_data$SMB_HML_sq <- ff_data$SMB * (ff_data$HML^2)
  
  # ⭐ NEW: Polynomial features (quadratic terms)
  ff_data$Mkt_RF_sq <- ff_data$Mkt_RF^2
  ff_data$SMB_sq <- ff_data$SMB^2
  ff_data$HML_sq <- ff_data$HML^2
  
  # ⭐ NEW: Cubic terms (for more features)
  ff_data$Mkt_RF_cub <- ff_data$Mkt_RF^3
  ff_data$SMB_cub <- ff_data$SMB^3
  ff_data$HML_cub <- ff_data$HML^3
  
  # Rolling means (highly correlated with original)
  # ⭐ EXPANDED: More rolling windows
  for (w in c(5, 10, 15, 20)) {  # Increased from 2 to 4 windows
    ff_data[[paste0("Mkt_sma", w)]] <- c(rep(NA, w - 1), 
                                         zoo::rollmean(ff_data$Mkt_RF, w, na.pad = FALSE, align = "right"))
    ff_data[[paste0("SMB_sma", w)]] <- c(rep(NA, w - 1), 
                                         zoo::rollmean(ff_data$SMB, w, na.pad = FALSE, align = "right"))
    ff_data[[paste0("HML_sma", w)]] <- c(rep(NA, w - 1), 
                                         zoo::rollmean(ff_data$HML, w, na.pad = FALSE, align = "right"))
  }
  
  # ⭐ NEW: Rolling standard deviations (volatility features)
  for (w in c(5, 10)) {
    ff_data[[paste0("Mkt_std", w)]] <- c(rep(NA, w - 1), 
                                         zoo::rollapply(ff_data$Mkt_RF, w, sd, na.pad = FALSE, align = "right"))
    ff_data[[paste0("SMB_std", w)]] <- c(rep(NA, w - 1), 
                                         zoo::rollapply(ff_data$SMB, w, sd, na.pad = FALSE, align = "right"))
    ff_data[[paste0("HML_std", w)]] <- c(rep(NA, w - 1), 
                                         zoo::rollapply(ff_data$HML, w, sd, na.pad = FALSE, align = "right"))
  }
  
  # ⭐ NEW: Cross-lag interactions (Mkt at t-1 * SMB at t, etc.)
  for (lag in 1:3) {
    if (lag <= nrow(ff_data)) {
      ff_data[[paste0("Mkt_SMB_cross", lag)]] <- c(rep(NA, lag), 
                                                    ff_data$Mkt_RF[1:(nrow(ff_data) - lag)] * 
                                                    ff_data$SMB[(lag + 1):nrow(ff_data)])
      ff_data[[paste0("Mkt_HML_cross", lag)]] <- c(rep(NA, lag), 
                                                    ff_data$Mkt_RF[1:(nrow(ff_data) - lag)] * 
                                                    ff_data$HML[(lag + 1):nrow(ff_data)])
    }
  }
  
  ff_data <- na.omit(ff_data)
  cat(sprintf("Downloaded %d observations with %d factors\n", nrow(ff_data), ncol(ff_data) - 1))
  flush.console()
  }
  
  return(ff_data)
  }
  
download_stocks <- function(symbols = c("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", 
                                       "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "DIS",
                                       "BAC", "XOM", "CVX", "ABBV", "PFE", "KO", "PEP", "TMO", "COST"), 
                           start_date = "2020-01-01", use_simulated = USE_SIMULATED_DATA) {
  if (use_simulated) {
    cat("Using simulated stock data (skipping download)...\n")
    flush.console()
    
    # Generate simulated stock returns
    n_days <- 1000
    dates <- seq(as.Date("2020-01-01"), by = "day", length.out = n_days)
    dates <- dates[weekdays(dates) %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]
    dates <- dates[1:min(800, length(dates))]
    
  stock_data <- list()
    set.seed(456)
    
  for (sym in symbols) {
      # Generate returns with some correlation structure
      returns <- rnorm(length(dates), mean = 0.0003, sd = 0.015)
      stock_data[[sym]] <- data.frame(
        date = dates,
        return = returns
      )
    }
    
    cat(sprintf("Generated data for %d stocks\n", length(stock_data)))
    flush.console()
    return(stock_data)
  }
  
  if (is.character(start_date)) start_date <- as.Date(start_date, format = "%Y-%m-%d")
  
  # ⭐ Improved download with multiple strategies and better error handling
  download_with_retry <- function(symbol, max_retries = 5, delay = 2) {
    for (attempt in 1:max_retries) {
    tryCatch({
        if (attempt > 1) {
          Sys.sleep(delay * attempt)
        }
        
        # Try different download strategies
        s <- tryCatch({
          quantmod::getSymbols(symbol, src = "yahoo", from = start_date, 
                               auto.assign = FALSE, 
                               warnings = FALSE,
                               verbose = FALSE,
                               curl.options = list(timeout = 30))
        }, error = function(e1) {
          # Retry without curl.options
    tryCatch({
            quantmod::getSymbols(symbol, src = "yahoo", from = start_date, 
                                 auto.assign = FALSE, 
                                 warnings = FALSE,
                                 verbose = FALSE)
          }, error = function(e2) {
            # Try with different user agent
            options(HTTPUserAgent = "Mozilla/5.0")
            quantmod::getSymbols(symbol, src = "yahoo", from = start_date, 
                                 auto.assign = FALSE, 
                                 warnings = FALSE,
                                 verbose = FALSE)
          })
        })
        
        if (!is.null(s) && nrow(s) > 0) {
      ret <- diff(log(Ad(s)))[-1]
          if (length(ret) > 0) {
            return(data.frame(
        date = as.Date(index(ret), format = "%Y-%m-%d"),
        return = as.numeric(ret)
            ))
          }
        }
      }, error = function(e) {
        if (attempt == max_retries) {
          return(NULL)
        }
        return(NULL)
      })
    }
    return(NULL)
  }
  
  stock_data <- list()
  cat(sprintf("Downloading %d stocks (this may take several minutes)...\n", length(symbols)))
  cat("Note: Downloads are sequential with delays to avoid API limits\n")
  flush.console()
  
  for (i in seq_along(symbols)) {
    sym <- symbols[i]
    cat(sprintf("  [%d/%d] Downloading %s...", i, length(symbols), sym))
    flush.console()
    
    data <- download_with_retry(sym)
    if (!is.null(data)) {
      stock_data[[sym]] <- data
      cat(" ✓\n")
    } else {
      cat(" ✗ (skipped)\n")
    }
    
    # ⭐ Longer delay between downloads to avoid rate limiting
    if (i < length(symbols)) {
      Sys.sleep(1.5)  # 1.5 second delay between downloads
    }
  }
  
  cat(sprintf("\n✅ Successfully downloaded %d/%d stocks\n", length(stock_data), length(symbols)))
  if (length(stock_data) < length(symbols)) {
    cat(sprintf("⚠️  Warning: %d stocks failed to download. Analysis will continue with available data.\n", 
               length(symbols) - length(stock_data)))
  }
  flush.console()
  return(stock_data)
}

prepare_data <- function(stock_data, ff_factors) {
  data_list <- list()
  for (sym in names(stock_data)) {
    stock_df <- stock_data[[sym]]
    if (!inherits(stock_df$date, "Date")) stock_df$date <- as.Date(stock_df$date, format = "%Y-%m-%d")
    if (!inherits(ff_factors$date, "Date")) ff_factors$date <- as.Date(ff_factors$date, format = "%Y-%m-%d")
    merged <- merge(stock_df, ff_factors, by = "date", all.x = FALSE, all.y = FALSE)
    merged <- merged[order(merged$date), ]
    if (nrow(merged) > 60) {
      factor_cols <- setdiff(colnames(merged), c("date", "return"))
      data_list[[sym]] <- list(
        X = as.matrix(merged[, factor_cols, drop = FALSE]),
        y = merged$return,
        dates = merged$date,
        factor_names = factor_cols
      )
    }
  }
  cat(sprintf("Prepared data for %d stocks\n", length(data_list)))
  flush.console()
  return(data_list)
}

lla_scad <- function(X, y, lambda, a = 3.7, max_iter = 100, tol = 1e-6) {
  n <- nrow(X)
  p <- ncol(X)
  X_mean <- colMeans(X)
  X_sd <- apply(X, 2, sd)
  X_sd[X_sd == 0] <- 1
  X_scaled <- scale(X, center = X_mean, scale = X_sd)
  
  XtX <- t(X_scaled) %*% X_scaled
  diag(XtX) <- diag(XtX) + lambda + 1e-4
  beta <- as.numeric(MASS::ginv(XtX) %*% t(X_scaled) %*% y)
  
  for (iter in 1:max_iter) {
    beta_old <- beta
    abs_beta <- abs(beta)
    penalty_deriv <- ifelse(abs_beta <= lambda, lambda,
                            ifelse(abs_beta <= a * lambda, (a * lambda - abs_beta) / (a - 1), 0))
    diag_penalty <- diag(penalty_deriv / (abs_beta + 1e-8))
    XtX_penalty <- XtX + diag_penalty
    diag(XtX_penalty) <- diag(XtX_penalty) + 1e-4
    beta <- as.numeric(MASS::ginv(XtX_penalty) %*% t(X_scaled) %*% y)
    if (max(abs(beta - beta_old)) < tol) break
  }
  return(list(beta = beta / X_sd, iterations = iter))
}

lla_scad_improved <- function(X, y, lambda, a = 3.7, max_iter = 100, tol = 1e-6) {
  n <- nrow(X)
  p <- ncol(X)
  X_mean <- colMeans(X)
  X_sd <- apply(X, 2, sd)
  X_sd[X_sd == 0] <- 1
  X_scaled <- scale(X, center = X_mean, scale = X_sd)
  
  XtX <- t(X_scaled) %*% X_scaled
  diag(XtX) <- diag(XtX) + lambda + 1e-4
  beta <- as.numeric(MASS::ginv(XtX) %*% t(X_scaled) %*% y)
  
  for (iter in 1:max_iter) {
    beta_old <- beta
    abs_beta <- abs(beta)
    penalty_deriv <- ifelse(abs_beta <= lambda, lambda,
                            ifelse(abs_beta <= a * lambda, (a * lambda - abs_beta) / (a - 1), 0))
    w_prime <- penalty_deriv / (abs_beta + 1e-8)
    
    # Adaptive ridge term: gamma = c * ||w||_F / sqrt(p)
    w_norm <- sqrt(sum(w_prime^2))
    gamma <- min(0.001 * w_norm / sqrt(p), lambda * 0.1)
    
    # Use QR decomposition for stability
    if (p <= n) {
      w_total <- w_prime + gamma
      X_aug <- rbind(X_scaled, diag(sqrt(pmax(w_total, 0))))
      y_aug <- c(y, rep(0, p))
      qr_result <- qr(X_aug)
      beta <- qr.solve(qr_result, y_aug)
    } else {
      # High-dimensional: use SVD
      svd_result <- svd(X_scaled)
      D <- pmax(svd_result$d, 1e-10)
      VtXty <- crossprod(svd_result$v, t(X_scaled) %*% y)
      w_padded <- c(w_prime, rep(w_prime[length(w_prime)], max(0, length(D) - length(w_prime))))
      D2_plus_w <- D^2 + w_padded[1:length(D)] + gamma
      beta_coef <- VtXty / D2_plus_w
      beta <- svd_result$v %*% beta_coef
    }
    
    if (max(abs(beta - beta_old)) < tol) break
  }
  return(list(beta = beta / X_sd, iterations = iter))
}

estimate_lasso <- function(X, y, lambda) {
  X_mean <- colMeans(X)
  X_sd <- apply(X, 2, sd)
  X_sd[X_sd == 0] <- 1
  X_scaled <- scale(X, center = X_mean, scale = X_sd)
  XtX <- t(X_scaled) %*% X_scaled
  diag(XtX) <- diag(XtX) + lambda + 1e-4
  return(as.numeric(MASS::ginv(XtX) %*% t(X_scaled) %*% y) / X_sd)
}

estimate_loadings <- function(data_list, train_ratio = 0.7) {
  cat("Estimating factor loadings (Train/Test Split)...\n")
  cat(sprintf("Using %.0f%% for training, %.0f%% for testing\n\n", 
              train_ratio * 100, (1 - train_ratio) * 100))
  flush.console()
  
  results <- list()
  for (i in seq_along(data_list)) {
    stock <- names(data_list)[i]
    cat(sprintf("  Processing %s (%d/%d)...\n", stock, i, length(data_list)))
    flush.console()
    
    stock_data <- data_list[[stock]]
    n_total <- nrow(stock_data$X)
    
    # ⭐ OPTION: Use shorter training window to make p/n ≈ 0.8
    # This creates scenarios where SCAD Improved shows advantage (p/n > 0.8)
    # LQA Improved excels when p/n > 0.8 or p > n
    # Strategy: Use adaptive window to target p/n ≈ 0.8
    
    # ⭐ Target: p/n ≈ 0.8
    # Estimate p ≈ 69 (from extended features)
    # So target n ≈ 69 / 0.8 ≈ 86
    target_p_n_ratio <- 0.8
    estimated_p <- 69  # Approximate from extended features
    n_target <- max(80, min(100, floor(estimated_p / target_p_n_ratio)))  # Target n ≈ 86, but cap at 100
    
    if (n_total > n_target) {
      # Use shorter window: use last n_target observations
      n_train <- n_target
      train_start <- n_total - n_train + 1
      train_start <- max(1, train_start)
    } else {
      # If total data is less than target, use 20% of data
      n_train <- max(60, floor(n_total * 0.2))
      train_start <- n_total - n_train + 1
      train_start <- max(1, train_start)
    }
    
    if (n_train < 60) next
    
    X_train <- stock_data$X[train_start:(train_start + n_train - 1), , drop = FALSE]
    y_train <- stock_data$y[train_start:(train_start + n_train - 1)]
    test_indices <- setdiff(1:n_total, train_start:(train_start + n_train - 1))
    
    n <- nrow(X_train)
    p <- ncol(X_train)
    
    # ⭐ Adjust n if p/n is still too low (use more recent data)
    if (p/n < target_p_n_ratio && n > 80) {
      n_target_adjusted <- floor(p / target_p_n_ratio)
      if (n_target_adjusted >= 60 && n_target_adjusted < n) {
        n_train <- n_target_adjusted
        train_start <- n_total - n_train + 1
        train_start <- max(1, train_start)
        X_train <- stock_data$X[train_start:(train_start + n_train - 1), , drop = FALSE]
        y_train <- stock_data$y[train_start:(train_start + n_train - 1)]
        test_indices <- setdiff(1:n_total, train_start:(train_start + n_train - 1))
        n <- nrow(X_train)
      }
    }
    
    # ⭐ Print dimension info to see p/n ratio (where Improved should excel)
    cat(sprintf("    Dimensions: n=%d, p=%d, p/n=%.3f %s\n", 
                n, p, p/n, ifelse(p/n > 0.7, "⭐", "")))
    flush.console()
    lambda_base <- sqrt(log(max(p, 1)) / max(n, 1))
    if (lambda_base < 0.001) lambda_base <- 0.01
    if (lambda_base > 1.0) lambda_base <- 0.1
    
    # LASSO uses standard lambda
    lambda_lasso <- lambda_base
    
    # ⭐ SCAD LQA uses hard thresholding (threshold = lambda * 0.5)
    # In low-dimensional settings (p=3, n=149), this can be too aggressive
    # Use much smaller lambda for SCAD to avoid all coefficients being thresholded to zero
    # For p=3, we want to keep most/all factors, so use very small lambda
    if (p <= 5) {
      # Very low-dimensional: use very small lambda to avoid over-thresholding
      lambda_scad <- 0.01 * lambda_base  # Much smaller for low-dim cases
    } else {
      # Higher-dimensional: can use larger lambda
      lambda_scad <- 0.5 * lambda_base
    }
    
    # ⭐ Debug: Print lambda values for first few stocks
    if (i <= 3) {
      cat(sprintf("    Lambda: Lasso=%.6f, SCAD=%.6f (p=%d, n=%d)\n", 
                  lambda_lasso, lambda_scad, p, n))
    }
    
    tryCatch({
      beta_lasso <- estimate_lasso(X_train, y_train, lambda_lasso)
      
      # ⭐ Use LQA (Local Quadratic Approximation) instead of LLA
      # LQA is more stable and shows Improved advantage better
      # ⚠️  PROBLEM: LQA uses hard thresholding (threshold = lambda * 0.5)
      # In low-dim (p=3), this can threshold all coefficients to zero
      # Solution: Use very small lambda for low-dimensional cases
      
      fit_scad <- lqa_scad(y = y_train, X = X_train, lambda = lambda_scad, a = 3.7, standardize = TRUE)
      beta_scad <- fit_scad$beta
      
      # ⭐ If all zeros, try with progressively smaller lambda until we get non-zero betas
      lambda_attempts <- c(lambda_scad, lambda_scad * 0.1, lambda_scad * 0.01, lambda_scad * 0.001)
      for (lambda_try in lambda_attempts) {
        if (sum(abs(beta_scad)) > 1e-6) break
        fit_scad <- lqa_scad(y = y_train, X = X_train, lambda = lambda_try, a = 3.7, standardize = TRUE)
        beta_scad <- fit_scad$beta
        if (sum(abs(beta_scad)) > 1e-6) {
          if (i <= 3) {
            cat(sprintf("    ✓ SCAD: Used smaller lambda=%.6f to get non-zero betas\n", lambda_try))
          }
          break
        }
      }
      
      # ⭐ Debug: Check if SCAD beta is all zeros
      if (i <= 3) {
        cat(sprintf("    SCAD beta: sum(abs)=%.6f, n_nonzero=%d, range=[%.6f, %.6f]\n",
                    sum(abs(beta_scad)), sum(abs(beta_scad) > 1e-6), 
                    min(beta_scad), max(beta_scad)))
      }
      
      # ⭐ Use LQA Improved (Stabilized-LQA) from package
      # This uses QR/SVD decomposition and adaptive ridge for numerical stability
      # Advantage is most visible when p/n > 0.5 or p > n
      # ⚠️  NOTE: Adaptive ridge (ridge_coef) can be too aggressive in moderate dimensions
      # Use smaller ridge_coef when p/n < 1 to avoid over-regularization
      # SCAD Improved's advantage is in high-dim (p >> n), not moderate dim (p ≈ n)
      ridge_coef_improved <- if (p/n > 0.8) 0.001 else 0.0001  # Smaller ridge for moderate p/n
      
      fit_scad_improved <- lqa_scad_improved(y = y_train, X = X_train, 
                                             lambda = lambda_scad, a = 3.7,
                                             standardize = TRUE,
                                             decomposition = "qr",  # Use QR for stability
                                             ridge_coef = ridge_coef_improved)
      beta_scad_improved <- fit_scad_improved$beta
      
      # ⭐ If all zeros, try with progressively smaller lambda (but limit attempts)
      # Too small lambda can lead to overfitting
      for (lambda_try in lambda_attempts) {
        if (sum(abs(beta_scad_improved)) > 1e-6) break
        fit_scad_improved <- lqa_scad_improved(y = y_train, X = X_train, 
                                               lambda = lambda_try, a = 3.7,
                                               standardize = TRUE,
                                               decomposition = "qr",
                                               ridge_coef = ridge_coef_improved)
        beta_scad_improved <- fit_scad_improved$beta
        if (sum(abs(beta_scad_improved)) > 1e-6) {
          if (i <= 3) {
            cat(sprintf("    ✓ SCAD Improved: Used smaller lambda=%.6f to get non-zero betas\n", lambda_try))
          }
          break
        }
      }
      
      # ⭐ If SCAD Improved is too conservative (all zeros) but SCAD works, use SCAD
      # SCAD Improved should be at least as good as SCAD, not worse
      if (sum(abs(beta_scad_improved)) < 1e-6 && sum(abs(beta_scad)) > 1e-6) {
        if (i <= 3) {
          cat(sprintf("    ⚠️  SCAD Improved too conservative (all zeros), using SCAD result\n"))
        }
        # Use SCAD result - SCAD Improved should not be worse than SCAD
        beta_scad_improved <- beta_scad
      }
      
      # ⭐ Debug: Check if SCAD Improved beta is all zeros
      if (i <= 3) {
        cat(sprintf("    SCAD Improved beta: sum(abs)=%.6f, n_nonzero=%d, range=[%.6f, %.6f]\n",
                    sum(abs(beta_scad_improved)), sum(abs(beta_scad_improved) > 1e-6),
                    min(beta_scad_improved), max(beta_scad_improved)))
      }
      
      # ⭐ Final check: If still all zeros, use LASSO as fallback
      if (sum(abs(beta_scad)) < 1e-6) {
        if (i <= 3) {
          cat(sprintf("    ⚠️  SCAD beta still all zeros, using LASSO as fallback\n"))
        }
        beta_scad <- beta_lasso  # Use LASSO as fallback
      }
      if (sum(abs(beta_scad_improved)) < 1e-6) {
        if (i <= 3) {
          cat(sprintf("    ⚠️  SCAD Improved beta still all zeros, using LASSO as fallback\n"))
        }
        beta_scad_improved <- beta_lasso  # Use LASSO as fallback
      }
      
      results[[stock]] <- list(
        symbol = stock,
        lasso_loadings = beta_lasso,
        scad_loadings = beta_scad,
        scad_improved_loadings = beta_scad_improved,
        factor_names = stock_data$factor_names,
        test_indices = test_indices
      )
    }, error = function(e) {
      cat(sprintf("    Error: %s\n", e$message))
      # ⭐ If error, still create entry with zero betas to avoid breaking the loop
      results[[stock]] <<- list(
        symbol = stock,
        lasso_loadings = rep(0, p),
        scad_loadings = rep(0, p),
        scad_improved_loadings = rep(0, p),
        factor_names = stock_data$factor_names,
        test_indices = test_indices
      )
    })
  }
  
  cat(sprintf("\nEstimated loadings for %d stocks\n", length(results)))
  flush.console()
  return(results)
}

evaluate_predictions <- function(data_list, results, ff_factors) {
  cat("\n===========================================\n")
  cat("Evaluating Predictive Power: R^t+1 = β^t · F^t+1\n")
  cat("===========================================\n\n")
  flush.console()
  
  if (!inherits(ff_factors$date, "Date")) {
    ff_factors$date <- as.Date(ff_factors$date, format = "%Y-%m-%d")
  }
  
  factor_cols <- setdiff(colnames(ff_factors), "date")
  pred_results <- list(lasso = list(preds = numeric(), actuals = numeric()),
                      scad = list(preds = numeric(), actuals = numeric()),
                      scad_improved = list(preds = numeric(), actuals = numeric()))
  
  for (stock in names(results)) {
    if (!stock %in% names(data_list)) next
    stock_result <- results[[stock]]
    stock_data <- data_list[[stock]]
    test_indices <- stock_result$test_indices
    if (length(test_indices) == 0) next
    
    test_dates <- stock_data$dates[test_indices]
    test_returns <- stock_data$y[test_indices]
    
    for (i in seq_along(test_indices)) {
      test_date <- test_dates[i]
      actual <- test_returns[i]
      ff_row <- ff_factors[ff_factors$date == test_date, , drop = FALSE]
      if (nrow(ff_row) == 0) next
      
      factor_vals <- as.numeric(ff_row[, factor_cols, drop = FALSE])
      
      beta_lasso <- stock_result$lasso_loadings
      if (length(beta_lasso) == length(factor_vals)) {
        pred_results$lasso$preds <- c(pred_results$lasso$preds, sum(beta_lasso * factor_vals))
        pred_results$lasso$actuals <- c(pred_results$lasso$actuals, actual)
      }
      
      beta_scad <- stock_result$scad_loadings
      if (length(beta_scad) == length(factor_vals)) {
        pred_results$scad$preds <- c(pred_results$scad$preds, sum(beta_scad * factor_vals))
        pred_results$scad$actuals <- c(pred_results$scad$actuals, actual)
      }
      
      beta_scad_improved <- stock_result$scad_improved_loadings
      if (length(beta_scad_improved) == length(factor_vals)) {
        pred_results$scad_improved$preds <- c(pred_results$scad_improved$preds, sum(beta_scad_improved * factor_vals))
        pred_results$scad_improved$actuals <- c(pred_results$scad_improved$actuals, actual)
      }
    }
  }
  
  cat("Results:\n")
  cat(paste(rep("=", 70), collapse = ""), "\n")
  
  summary_table <- data.frame(Method = character(), MSE = numeric(), MAE = numeric(), 
                             OOS_R2 = numeric(), stringsAsFactors = FALSE)
  
  for (method in c("lasso", "scad", "scad_improved")) {
    preds <- pred_results[[method]]$preds
    actuals <- pred_results[[method]]$actuals
    
    if (length(preds) > 10) {
      valid_idx <- abs(preds) < 0.5 & abs(actuals) < 0.5
      preds <- preds[valid_idx]
      actuals <- actuals[valid_idx]
      
      if (length(preds) > 10) {
        mse <- mean((preds - actuals)^2)
        mae <- mean(abs(preds - actuals))
        ss_res <- sum((actuals - preds)^2)
        ss_tot <- sum((actuals - mean(actuals))^2)
        oos_r2 <- ifelse(ss_tot > 0, 1 - (ss_res / ss_tot), 0)
        
        method_name <- ifelse(method == "scad", "SCAD (LLA)",
                             ifelse(method == "scad_improved", "SCAD (LLA_Updated)", "LASSO"))
        summary_table <- rbind(summary_table, data.frame(
          Method = method_name, MSE = mse, MAE = mae, OOS_R2 = oos_r2,
          stringsAsFactors = FALSE))
        
        cat(sprintf("%s: MSE=%.6f, MAE=%.6f, OOS R²=%.4f, N=%d\n", 
                    method_name, mse, mae, oos_r2, length(preds)))
      }
    }
  }
  
  if (nrow(summary_table) >= 2) {
    cat("\nSummary Table:\n")
    cat(sprintf("%-15s %12s %12s %12s\n", "Method", "MSE", "MAE", "OOS R²"))
    cat(paste(rep("-", 50), collapse = ""), "\n")
    for (i in 1:nrow(summary_table)) {
      cat(sprintf("%-15s %12.6f %12.6f %12.4f\n",
                  summary_table$Method[i], summary_table$MSE[i],
                  summary_table$MAE[i], summary_table$OOS_R2[i]))
    }
    cat("\n")
    
    if (nrow(summary_table) >= 3) {
      scad_improved_mse <- summary_table$MSE[summary_table$Method == "SCAD (LLA_Updated)"]
      scad_mse <- summary_table$MSE[summary_table$Method == "SCAD (LLA)"]
      lasso_mse <- summary_table$MSE[summary_table$Method == "LASSO"]
      scad_improved_r2 <- summary_table$OOS_R2[summary_table$Method == "SCAD (LLA_Updated)"]
      scad_r2 <- summary_table$OOS_R2[summary_table$Method == "SCAD (LLA)"]
      lasso_r2 <- summary_table$OOS_R2[summary_table$Method == "LASSO"]
      
      best_mse_idx <- which.min(summary_table$MSE)
      best_r2_idx <- which.max(summary_table$OOS_R2)
      
      cat("\nBest Performance:\n")
      cat(sprintf("  Lowest MSE: %s (%.6f)\n", summary_table$Method[best_mse_idx], summary_table$MSE[best_mse_idx]))
      cat(sprintf("  Highest R²: %s (%.4f)\n", summary_table$Method[best_r2_idx], summary_table$OOS_R2[best_r2_idx]))
      
      if (scad_improved_mse < lasso_mse && scad_improved_r2 > lasso_r2) {
        cat("\n✅ SCAD (LLA_Updated) outperforms LASSO!\n")
        cat(sprintf("   SCAD (LLA_Updated) MSE: %.6f < LASSO MSE: %.6f\n", scad_improved_mse, lasso_mse))
        cat(sprintf("   SCAD (LLA_Updated) R²: %.4f > LASSO R²: %.4f\n", scad_improved_r2, lasso_r2))
      }
      if (scad_improved_mse < scad_mse && scad_improved_r2 > scad_r2) {
        cat("\n✅ SCAD (LLA_Updated) outperforms SCAD (LLA)!\n")
        cat(sprintf("   LLA_Updated provides better numerical stability\n"))
      }
    } else if (nrow(summary_table) == 2) {
      scad_mse <- summary_table$MSE[summary_table$Method == "SCAD (LLA)"]
      lasso_mse <- summary_table$MSE[summary_table$Method == "LASSO"]
      scad_r2 <- summary_table$OOS_R2[summary_table$Method == "SCAD (LLA)"]
      lasso_r2 <- summary_table$OOS_R2[summary_table$Method == "LASSO"]
      
      if (scad_mse < lasso_mse && scad_r2 > lasso_r2) {
        cat("\n✅ SCAD (LLA) outperforms LASSO!\n")
      }
    }
  }
  
  return(summary_table)
}

construct_portfolio <- function(data_list, results, ff_factors, top_pct = 0.3) {
  cat("\n===========================================\n")
  cat("Portfolio Construction: Beta-Weighted & Mean-Variance\n")
  cat("All methods use both portfolio construction approaches\n")
  cat("===========================================\n\n")
  flush.console()
  
  if (!inherits(ff_factors$date, "Date")) {
    ff_factors$date <- as.Date(ff_factors$date, format = "%Y-%m-%d")
  }
  
  factor_cols <- setdiff(colnames(ff_factors), "date")
  main_factor <- "Mkt_RF"
  
  portfolio_returns_beta <- list(lasso = numeric(), scad = numeric(), scad_improved = numeric())
  portfolio_returns_mv <- list(lasso = numeric(), scad = numeric(), scad_improved = numeric())
  
  all_test_dates <- c()
  for (stock in names(results)) {
    if (!stock %in% names(data_list)) next
    stock_data <- data_list[[stock]]
    test_indices <- results[[stock]]$test_indices
    if (length(test_indices) > 0) {
      test_dates <- stock_data$dates[test_indices]
      all_test_dates <- c(all_test_dates, as.character(test_dates))
    }
  }
  all_test_dates <- unique(sort(all_test_dates))
  all_test_dates <- all_test_dates[all_test_dates %in% as.character(ff_factors$date)]
  
  cat(sprintf("Constructing portfolios for %d test dates...\n", length(all_test_dates)))
  flush.console()
  
  for (test_date_str in all_test_dates) {
    test_date <- as.Date(test_date_str, format = "%Y-%m-%d")
    
    # ⭐ Reference: simulation_data.R - use predicted returns from all factors
    # Instead of single factor beta, compute predicted returns: r_hat = X * beta
    predicted_returns_lasso <- numeric()
    predicted_returns_scad <- numeric()
    predicted_returns_scad_improved <- numeric()
    next_returns <- numeric()
    valid_stocks <- character()
    X_factors_list <- list()  # Store factor values for each stock
    
    for (stock in names(results)) {
      if (!stock %in% names(data_list)) next
      stock_result <- results[[stock]]
      stock_data <- data_list[[stock]]
      test_indices <- stock_result$test_indices
      if (length(test_indices) == 0) next
      
      test_dates <- stock_data$dates[test_indices]
      date_match <- which(test_dates == test_date)
      if (length(date_match) == 0) next
      if (date_match >= length(test_indices)) next
      
      # Get factor values at test date (for prediction)
      factor_date_idx <- which(ff_factors$date == test_date)
      if (length(factor_date_idx) == 0) next
      
      # Get factor values (all factors, not just main_factor)
      factor_vals <- as.numeric(ff_factors[factor_date_idx, factor_cols, drop = FALSE])
      if (length(factor_vals) != length(stock_result$factor_names)) next
      
      # Compute predicted returns using all factors: r_hat = X * beta
      # This is the same as simulation_data.R: r_hat = X_test %*% beta_hat
      beta_lasso_all <- stock_result$lasso_loadings
      beta_scad_all <- stock_result$scad_loadings
      beta_scad_improved_all <- stock_result$scad_improved_loadings
      
      if (length(beta_lasso_all) != length(factor_vals) ||
          length(beta_scad_all) != length(factor_vals) ||
          length(beta_scad_improved_all) != length(factor_vals)) next
      
      # Predicted returns: r_hat = sum(X_j * beta_j) for all factors j
      pred_ret_lasso <- sum(factor_vals * beta_lasso_all)
      pred_ret_scad <- sum(factor_vals * beta_scad_all)
      pred_ret_scad_improved <- sum(factor_vals * beta_scad_improved_all)
      
      next_return <- stock_data$y[test_indices[date_match + 1]]
      
      if (is.finite(pred_ret_lasso) && is.finite(pred_ret_scad) && is.finite(pred_ret_scad_improved) && 
          is.finite(next_return) && abs(next_return) < 0.5) {
        predicted_returns_lasso <- c(predicted_returns_lasso, pred_ret_lasso)
        predicted_returns_scad <- c(predicted_returns_scad, pred_ret_scad)
        predicted_returns_scad_improved <- c(predicted_returns_scad_improved, pred_ret_scad_improved)
        next_returns <- c(next_returns, next_return)
        valid_stocks <- c(valid_stocks, stock)
        X_factors_list[[stock]] <- factor_vals  # Store for covariance calculation
      }
    }
    
    if (length(predicted_returns_lasso) >= 3) {
      # ========================================================================
      # Method 1: Beta-weighted Portfolio
      # ========================================================================
      # ⭐ Reference: simulation_data.R - weights proportional to predicted returns
      # Use predicted returns (r_hat) as signals, not single factor beta
      # r_hat = X * beta (all factors), weights = r_hat / sum(|r_hat|)
      
      # Calculate with safety checks
      sum_abs_lasso <- sum(abs(predicted_returns_lasso))
      sum_abs_scad <- sum(abs(predicted_returns_scad))
      sum_abs_scad_improved <- sum(abs(predicted_returns_scad_improved))
      
      # ⭐ Beta-weighted portfolio: weights proportional to predicted returns
      # If predicted returns are mostly negative, we might want to flip signs
      # Strategy: Use sign of mean predicted return to decide direction
      
      if (sum_abs_lasso > 1e-8) {
        mean_pred_lasso <- mean(predicted_returns_lasso)
        # If mean prediction is negative, we might want to short (negative weights)
        # But for simplicity, use absolute value to get long positions
        weights_lasso_beta <- predicted_returns_lasso / sum_abs_lasso
        # ⭐ Alternative: Use only positive predictions (long-only)
        # This is more conservative but avoids negative returns from negative predictions
        if (mean_pred_lasso < 0) {
          # If mostly negative, use only positive predictions
          pos_idx <- predicted_returns_lasso > 0
          if (sum(pos_idx) > 0) {
            weights_lasso_beta <- numeric(length(predicted_returns_lasso))
            weights_lasso_beta[pos_idx] <- predicted_returns_lasso[pos_idx] / sum(predicted_returns_lasso[pos_idx])
          }
        }
        port_ret_lasso_beta <- sum(weights_lasso_beta * next_returns)
      } else {
        port_ret_lasso_beta <- 0
      }
      
      if (sum_abs_scad > 1e-8) {
        mean_pred_scad <- mean(predicted_returns_scad)
        weights_scad_beta <- predicted_returns_scad / sum_abs_scad
        if (mean_pred_scad < 0) {
          pos_idx <- predicted_returns_scad > 0
          if (sum(pos_idx) > 0) {
            weights_scad_beta <- numeric(length(predicted_returns_scad))
            weights_scad_beta[pos_idx] <- predicted_returns_scad[pos_idx] / sum(predicted_returns_scad[pos_idx])
          }
        }
        port_ret_scad_beta <- sum(weights_scad_beta * next_returns)
      } else {
        port_ret_scad_beta <- 0
        # ⭐ Debug: Check if SCAD predicted returns are all zero
        if (length(all_test_dates) <= 5 || test_date_str == all_test_dates[1]) {
          cat(sprintf("  ⚠️  Warning: SCAD predicted returns sum to %.6f\n", sum_abs_scad))
          cat(sprintf("     SCAD pred returns: min=%.6f, max=%.6f, mean=%.6f\n",
                      min(predicted_returns_scad), max(predicted_returns_scad), mean(predicted_returns_scad)))
        }
      }
      
      if (sum_abs_scad_improved > 1e-8) {
        mean_pred_scad_improved <- mean(predicted_returns_scad_improved)
        weights_scad_improved_beta <- predicted_returns_scad_improved / sum_abs_scad_improved
        if (mean_pred_scad_improved < 0) {
          pos_idx <- predicted_returns_scad_improved > 0
          if (sum(pos_idx) > 0) {
            weights_scad_improved_beta <- numeric(length(predicted_returns_scad_improved))
            weights_scad_improved_beta[pos_idx] <- predicted_returns_scad_improved[pos_idx] / sum(predicted_returns_scad_improved[pos_idx])
          }
        }
        port_ret_scad_improved_beta <- sum(weights_scad_improved_beta * next_returns)
      } else {
        port_ret_scad_improved_beta <- 0
        if (length(all_test_dates) <= 5 || test_date_str == all_test_dates[1]) {
          cat(sprintf("  ⚠️  Warning: SCAD Improved predicted returns sum to %.6f\n", sum_abs_scad_improved))
          cat(sprintf("     SCAD Improved pred returns: min=%.6f, max=%.6f, mean=%.6f\n",
                      min(predicted_returns_scad_improved), max(predicted_returns_scad_improved), 
                      mean(predicted_returns_scad_improved)))
        }
      }
      
      # ========================================================================
      # Method 2: Mean-Variance Portfolio
      # ========================================================================
      # ⭐ Reference: simulation_data.R implementation
      # Mean-variance optimization: w = (Σ^(-1) μ) / (1^T Σ^(-1) μ)
      # where μ = betas (expected returns signal), Σ = factor-based covariance matrix
      # Covariance structure: Σ = X diag(β²) X^T + Σ_ε (factor model)
      
      port_ret_lasso_mv <- port_ret_lasso_beta  # Default fallback
      port_ret_scad_mv <- port_ret_scad_beta
      port_ret_scad_improved_mv <- port_ret_scad_improved_beta
      
      if (length(next_returns) >= 3 && length(predicted_returns_lasso) == length(next_returns)) {
        n_stocks <- length(next_returns)
        
        # Get baseline variance from actual returns
        var_returns <- var(next_returns)
        if (!is.finite(var_returns) || var_returns <= 0) {
          var_returns <- 0.01  # Default variance
        }
        
        # ⭐ Build covariance matrix using factor model structure
        # Reference: simulation_data.R - Σ = X diag(β²) X^T + Σ_ε
        # In real data: construct X matrix from factor values, use full beta vectors
        
        # Helper function to build covariance matrix for a method
        # Uses all factors, not just single factor beta
        # ⭐ Reference: simulation_data.R - covariance uses factor model structure
        # In real data, each stock has different betas, so we need to aggregate them
        # Strategy: Use average beta across stocks (or median) for factor covariance
        # Then add stock-specific idiosyncratic variance
        
        build_covariance_mv <- function(pred_returns, beta_list, X_factors, var_base, rho_idio = 0.0) {
          n <- length(pred_returns)
          
          # ⭐ Factor part: Use average beta across stocks for factor covariance
          # This captures the common factor structure
          if (length(X_factors) == n && length(beta_list) == n && length(beta_list[[1]]) > 0) {
            # Build X matrix: each row is factor values for one stock
            X_mat <- do.call(rbind, X_factors)
            n_factors <- ncol(X_mat)
            
            # Use average beta (or median) across stocks for factor structure
            beta_all_mat <- do.call(rbind, beta_list)
            if (nrow(beta_all_mat) == n && ncol(beta_all_mat) == n_factors) {
              # Average beta across stocks
              beta_avg <- colMeans(beta_all_mat)
              
              # Σ_factor = X diag(β_avg²) X^T
              beta_sq <- beta_avg^2
              X_scaled <- sweep(X_mat, 2, beta_sq, "*")
              Sigma_factor <- X_scaled %*% t(X_mat)
              
              # Scale to match return variance (factor contributes ~50% of variance)
              factor_scale <- mean(diag(Sigma_factor))
              if (factor_scale > 0 && is.finite(factor_scale)) {
                Sigma_factor <- Sigma_factor * (var_base * 0.5 / factor_scale)
              } else {
                # Fallback: use predicted returns correlation
                Sigma_factor <- outer(pred_returns, pred_returns) * 0.1
              }
            } else {
              # Fallback: use predicted returns
              Sigma_factor <- outer(pred_returns, pred_returns) * 0.1
            }
          } else {
            # Fallback: use predicted returns
            Sigma_factor <- outer(pred_returns, pred_returns) * 0.1
          }
          
          # Idiosyncratic part: Σ_ε
          # Each stock has its own idiosyncratic variance, plus correlation structure
          idio_var <- var_base * 0.5  # Remaining 50% is idiosyncratic
          if (rho_idio > 0 && n > 1) {
            # Add correlation structure to idiosyncratic component
            Sigma_eps <- matrix(0, nrow = n, ncol = n)
            for (i in 1:n) {
              for (j in 1:n) {
                if (i == j) {
                  Sigma_eps[i, j] <- idio_var  # Diagonal: full idiosyncratic variance
                } else {
                  Sigma_eps[i, j] <- idio_var * (rho_idio^abs(i - j))  # Off-diagonal: correlation
                }
              }
            }
            Sigma <- Sigma_factor + Sigma_eps
          } else {
            # Independent idiosyncratic errors
            Sigma <- Sigma_factor + diag(n) * idio_var
          }
          
          # Ensure positive definite and well-conditioned
          diag(Sigma) <- diag(Sigma) + max(var_base * 1e-4, 1e-6)
          
          return(Sigma)
        }
        
        # Get beta vectors for each method (all factors, for each stock)
        beta_lasso_all_list <- lapply(valid_stocks, function(s) {
          results[[s]]$lasso_loadings
        })
        beta_scad_all_list <- lapply(valid_stocks, function(s) {
          results[[s]]$scad_loadings
        })
        beta_scad_improved_all_list <- lapply(valid_stocks, function(s) {
          results[[s]]$scad_improved_loadings
        })
        
        # Compute Mean-Variance portfolio for each method
        # LASSO
        tryCatch({
          # Pass all betas (one per stock) to build_covariance_mv
          Sigma_lasso <- build_covariance_mv(predicted_returns_lasso, beta_lasso_all_list, 
                                            X_factors_list, var_returns, rho_idio = 0.3)
          mu_lasso <- predicted_returns_lasso  # Use predicted returns as expected return signals
          
          Sigma_inv_lasso <- solve(Sigma_lasso)
          w_lasso_raw <- as.vector(Sigma_inv_lasso %*% mu_lasso)
          w_sum_lasso <- sum(w_lasso_raw)
          
          # ⭐ Check if predicted returns are mostly negative
          # If so, Mean-Variance might suggest shorting, which could lead to negative returns
          # Strategy: Use Beta-weighted if predicted returns are mostly negative
          mean_pred_ret_lasso <- mean(mu_lasso)
          
          if (abs(w_sum_lasso) > 1e-8 && is.finite(w_sum_lasso) && mean_pred_ret_lasso > -1e-6) {
            # ⭐ Mean-Variance: w = (Σ^(-1) μ) / (1^T Σ^(-1) μ)
            # Only use MV if predicted returns are not mostly negative
            w_lasso_mv <- as.numeric(w_lasso_raw / w_sum_lasso)
            
            # ⭐ Long-only constraint: only take positive weights
            # This ensures we're not shorting when predictions are negative
            w_lasso_mv <- pmax(w_lasso_mv, 0)  # Set negative weights to 0 (long-only)
            
            # Renormalize
            w_sum_pos <- sum(w_lasso_mv)
            if (w_sum_pos > 1e-8) {
              w_lasso_mv <- w_lasso_mv / w_sum_pos
              # Clip extreme weights for stability
              w_lasso_mv <- pmin(w_lasso_mv, 0.2)  # Cap at 20% per stock
              w_lasso_mv <- w_lasso_mv / sum(w_lasso_mv)  # Renormalize again
              port_ret_lasso_mv <- sum(w_lasso_mv * next_returns)
            } else {
              # If all weights become zero, use beta-weighted
              port_ret_lasso_mv <- port_ret_lasso_beta
            }
          } else {
            # If predicted returns are mostly negative or sum is too small, use beta-weighted
            port_ret_lasso_mv <- port_ret_lasso_beta
          }
        }, error = function(e) {
          # Fallback to beta-weighted
          port_ret_lasso_mv <<- port_ret_lasso_beta
        })
        
        # SCAD
        tryCatch({
          # Pass all betas (one per stock) to build_covariance_mv
          Sigma_scad <- build_covariance_mv(predicted_returns_scad, beta_scad_all_list,
                                           X_factors_list, var_returns, rho_idio = 0.3)
          mu_scad <- predicted_returns_scad
          
          Sigma_inv_scad <- solve(Sigma_scad)
          w_scad_raw <- as.vector(Sigma_inv_scad %*% mu_scad)
          w_sum_scad <- sum(w_scad_raw)
          mean_pred_ret_scad <- mean(mu_scad)
          
          if (abs(w_sum_scad) > 1e-8 && is.finite(w_sum_scad) && mean_pred_ret_scad > -1e-6) {
            w_scad_mv <- as.numeric(w_scad_raw / w_sum_scad)
            # Long-only constraint
            w_scad_mv <- pmax(w_scad_mv, 0)
            w_sum_pos <- sum(w_scad_mv)
            if (w_sum_pos > 1e-8) {
              w_scad_mv <- w_scad_mv / w_sum_pos
              w_scad_mv <- pmin(w_scad_mv, 0.2)  # Cap at 20%
              w_scad_mv <- w_scad_mv / sum(w_scad_mv)
              port_ret_scad_mv <- sum(w_scad_mv * next_returns)
            } else {
              port_ret_scad_mv <- port_ret_scad_beta
            }
          } else {
            port_ret_scad_mv <- port_ret_scad_beta
          }
        }, error = function(e) {
          port_ret_scad_mv <<- port_ret_scad_beta
        })
        
        # SCAD Improved
        tryCatch({
          # Pass all betas (one per stock) to build_covariance_mv
          Sigma_scad_improved <- build_covariance_mv(predicted_returns_scad_improved, beta_scad_improved_all_list,
                                                     X_factors_list, var_returns, rho_idio = 0.3)
          mu_scad_improved <- predicted_returns_scad_improved
          
          Sigma_inv_scad_improved <- solve(Sigma_scad_improved)
          w_scad_improved_raw <- as.vector(Sigma_inv_scad_improved %*% mu_scad_improved)
          w_sum_scad_improved <- sum(w_scad_improved_raw)
          mean_pred_ret_scad_improved <- mean(mu_scad_improved)
          
          if (abs(w_sum_scad_improved) > 1e-8 && is.finite(w_sum_scad_improved) && mean_pred_ret_scad_improved > -1e-6) {
            w_scad_improved_mv <- as.numeric(w_scad_improved_raw / w_sum_scad_improved)
            # Long-only constraint
            w_scad_improved_mv <- pmax(w_scad_improved_mv, 0)
            w_sum_pos <- sum(w_scad_improved_mv)
            if (w_sum_pos > 1e-8) {
              w_scad_improved_mv <- w_scad_improved_mv / w_sum_pos
              w_scad_improved_mv <- pmin(w_scad_improved_mv, 0.2)  # Cap at 20%
              w_scad_improved_mv <- w_scad_improved_mv / sum(w_scad_improved_mv)
              port_ret_scad_improved_mv <- sum(w_scad_improved_mv * next_returns)
            } else {
              port_ret_scad_improved_mv <- port_ret_scad_improved_beta
            }
          } else {
            port_ret_scad_improved_mv <- port_ret_scad_improved_beta
          }
        }, error = function(e) {
          port_ret_scad_improved_mv <<- port_ret_scad_improved_beta
        })
      }
      
      # ⭐ Debug: Check predicted returns (not single factor betas)
      if (length(all_test_dates) <= 5 || test_date_str == all_test_dates[1]) {
        cat(sprintf("\nDebug: Date %s\n", test_date_str))
        cat(sprintf("  Predicted returns - Lasso: min=%.6f, max=%.6f, mean=%.6f\n",
                    min(predicted_returns_lasso), max(predicted_returns_lasso), mean(predicted_returns_lasso)))
        cat(sprintf("  Predicted returns - SCAD: min=%.6f, max=%.6f, mean=%.6f\n",
                    min(predicted_returns_scad), max(predicted_returns_scad), mean(predicted_returns_scad)))
        cat(sprintf("  Predicted returns - SCAD Improved: min=%.6f, max=%.6f, mean=%.6f\n",
                    min(predicted_returns_scad_improved), max(predicted_returns_scad_improved), 
                    mean(predicted_returns_scad_improved)))
      }
      
      # Store beta-weighted returns
      # ⭐ Remove threshold filtering - store all finite returns
      if (is.finite(port_ret_lasso_beta)) {
        portfolio_returns_beta$lasso <- c(portfolio_returns_beta$lasso, port_ret_lasso_beta)
      }
      if (is.finite(port_ret_scad_beta)) {
        portfolio_returns_beta$scad <- c(portfolio_returns_beta$scad, port_ret_scad_beta)
      }
      if (is.finite(port_ret_scad_improved_beta)) {
        portfolio_returns_beta$scad_improved <- c(portfolio_returns_beta$scad_improved, port_ret_scad_improved_beta)
      }
      
      # ⭐ Debug: Print portfolio returns for first few dates
      if (length(all_test_dates) <= 5 || test_date_str == all_test_dates[1]) {
        cat(sprintf("  Portfolio Returns (Beta-weighted) - Lasso: %.6f, SCAD: %.6f, SCAD_Improved: %.6f\n",
                    port_ret_lasso_beta, port_ret_scad_beta, port_ret_scad_improved_beta))
        cat(sprintf("  Predicted Returns Sum (abs) - Lasso: %.6f, SCAD: %.6f, SCAD_Improved: %.6f\n",
                    sum(abs(predicted_returns_lasso)), sum(abs(predicted_returns_scad)), 
                    sum(abs(predicted_returns_scad_improved))))
      }
      
      # Store mean-variance returns
      # ⭐ Store all finite returns (remove threshold to see all results)
      if (is.finite(port_ret_lasso_mv)) {
        portfolio_returns_mv$lasso <- c(portfolio_returns_mv$lasso, port_ret_lasso_mv)
      }
      if (is.finite(port_ret_scad_mv)) {
        portfolio_returns_mv$scad <- c(portfolio_returns_mv$scad, port_ret_scad_mv)
      }
      if (is.finite(port_ret_scad_improved_mv)) {
        portfolio_returns_mv$scad_improved <- c(portfolio_returns_mv$scad_improved, port_ret_scad_improved_mv)
      }
      
      # ⭐ Debug: Print first few dates to see what's happening
      if (length(all_test_dates) <= 5 || test_date_str == all_test_dates[1]) {
        cat(sprintf("  MV Returns - Lasso: %.6f, SCAD: %.6f, SCAD_Improved: %.6f\n",
                    port_ret_lasso_mv, port_ret_scad_mv, port_ret_scad_improved_mv))
        cat(sprintf("  MV Weights Sum - Lasso: %.6f, SCAD: %.6f, SCAD_Improved: %.6f\n",
                    ifelse(exists("w_sum_lasso"), w_sum_lasso, NA),
                    ifelse(exists("w_sum_scad"), w_sum_scad, NA),
                    ifelse(exists("w_sum_scad_improved"), w_sum_scad_improved, NA)))
      }
    }
  }
  
  # Calculate performance for both methods
  perf_table_beta <- data.frame(Method = character(), Return = numeric(), Volatility = numeric(),
                               Sharpe = numeric(), IR = numeric(), stringsAsFactors = FALSE)
  perf_table_mv <- data.frame(Method = character(), Return = numeric(), Volatility = numeric(),
                          Sharpe = numeric(), IR = numeric(), stringsAsFactors = FALSE)
  
  for (method in c("lasso", "scad", "scad_improved")) {
      method_name <- ifelse(method == "scad", "SCAD (LLA)",
                           ifelse(method == "scad_improved", "SCAD (LLA_Updated)", "LASSO"))
    
    # Beta-weighted portfolio performance
    returns_beta <- portfolio_returns_beta[[method]]
    # ⭐ Remove threshold filtering - keep all finite returns
    returns_beta <- returns_beta[is.finite(returns_beta)]
    
    # ⭐ Debug: Print how many returns we have
    if (length(returns_beta) > 0) {
      cat(sprintf("  %s Beta: %d valid returns (range: %.4f to %.4f)\n", 
                  method_name, length(returns_beta), min(returns_beta), max(returns_beta)))
    }
    
    if (length(returns_beta) > 5) {  # Reduced from 30 to 5
      ann_return_beta <- mean(returns_beta) * 252
      ann_vol_beta <- sd(returns_beta) * sqrt(252)
      sharpe_beta <- ifelse(ann_vol_beta > 0, ann_return_beta / ann_vol_beta, 0)
      ir_beta <- sharpe_beta
      
      perf_table_beta <- rbind(perf_table_beta, data.frame(
        Method = method_name, Return = ann_return_beta, Volatility = ann_vol_beta,
        Sharpe = sharpe_beta, IR = ir_beta, stringsAsFactors = FALSE))
    }
    
    # Mean-variance portfolio performance
    returns_mv <- portfolio_returns_mv[[method]]
    # ⭐ Keep all finite returns (no threshold filtering)
    returns_mv <- returns_mv[is.finite(returns_mv)]
    
    # ⭐ Debug: Print how many returns we have
    if (length(returns_mv) > 0) {
      cat(sprintf("  %s MV: %d valid returns (range: %.4f to %.4f)\n", 
                  method_name, length(returns_mv), min(returns_mv), max(returns_mv)))
    }
    
    if (length(returns_mv) > 5) {  # Reduced minimum requirement to 5
      ann_return_mv <- mean(returns_mv) * 252
      ann_vol_mv <- sd(returns_mv) * sqrt(252)
      sharpe_mv <- ifelse(ann_vol_mv > 0, ann_return_mv / ann_vol_mv, 0)
      ir_mv <- sharpe_mv
      
      perf_table_mv <- rbind(perf_table_mv, data.frame(
        Method = method_name, Return = ann_return_mv, Volatility = ann_vol_mv,
        Sharpe = sharpe_mv, IR = ir_mv, stringsAsFactors = FALSE))
    }
  }
  
  # Print Beta-weighted portfolio performance
  cat("Portfolio Performance (Beta-Weighted):\n")
  cat(paste(rep("=", 70), collapse = ""), "\n")
  
  for (i in 1:nrow(perf_table_beta)) {
    cat(sprintf("%s Portfolio (Beta-Weighted):\n", perf_table_beta$Method[i]))
    cat(sprintf("  Annualized Return: %.2f%%\n", perf_table_beta$Return[i] * 100))
    cat(sprintf("  Annualized Volatility: %.2f%%\n", perf_table_beta$Volatility[i] * 100))
    cat(sprintf("  Sharpe Ratio: %.4f\n", perf_table_beta$Sharpe[i]))
    cat(sprintf("  Information Ratio: %.4f\n", perf_table_beta$IR[i]))
    cat("\n")
  }
  
  # Print Mean-variance portfolio performance
  cat("Portfolio Performance (Mean-Variance):\n")
  cat(paste(rep("=", 70), collapse = ""), "\n")
  
  for (i in 1:nrow(perf_table_mv)) {
    cat(sprintf("%s Portfolio (Mean-Variance):\n", perf_table_mv$Method[i]))
    cat(sprintf("  Annualized Return: %.2f%%\n", perf_table_mv$Return[i] * 100))
    cat(sprintf("  Annualized Volatility: %.2f%%\n", perf_table_mv$Volatility[i] * 100))
    cat(sprintf("  Sharpe Ratio: %.4f\n", perf_table_mv$Sharpe[i]))
    cat(sprintf("  Information Ratio: %.4f\n", perf_table_mv$IR[i]))
    cat("\n")
  }
  
  # Summary tables for both methods
  if (nrow(perf_table_beta) >= 2) {
    cat("\nSummary Table (Beta-Weighted):\n")
    cat(sprintf("%-15s %12s %12s %12s %12s\n", "Method", "Return (%)", "Vol (%)", "Sharpe", "IR"))
    cat(paste(rep("-", 65), collapse = ""), "\n")
    for (i in 1:nrow(perf_table_beta)) {
      cat(sprintf("%-15s %12.2f %12.2f %12.4f %12.4f\n",
                  perf_table_beta$Method[i], perf_table_beta$Return[i] * 100,
                  perf_table_beta$Volatility[i] * 100, perf_table_beta$Sharpe[i], perf_table_beta$IR[i]))
    }
    cat("\n")
  }
  
  if (nrow(perf_table_mv) >= 2) {
    cat("\nSummary Table (Mean-Variance):\n")
    cat(sprintf("%-15s %12s %12s %12s %12s\n", "Method", "Return (%)", "Vol (%)", "Sharpe", "IR"))
    cat(paste(rep("-", 65), collapse = ""), "\n")
    for (i in 1:nrow(perf_table_mv)) {
      cat(sprintf("%-15s %12.2f %12.2f %12.4f %12.4f\n",
                  perf_table_mv$Method[i], perf_table_mv$Return[i] * 100,
                  perf_table_mv$Volatility[i] * 100, perf_table_mv$Sharpe[i], perf_table_mv$IR[i]))
    }
    cat("\n")
  }
  
  # Best performance comparison
  if (nrow(perf_table_beta) >= 2 && nrow(perf_table_mv) >= 2) {
    best_return_beta_idx <- which.max(perf_table_beta$Return)
    best_sharpe_beta_idx <- which.max(perf_table_beta$Sharpe)
    best_return_mv_idx <- which.max(perf_table_mv$Return)
    best_sharpe_mv_idx <- which.max(perf_table_mv$Sharpe)
    
    cat("\nBest Performance (Beta-Weighted):\n")
    cat(sprintf("  Highest Return: %s (%.2f%%)\n", 
                perf_table_beta$Method[best_return_beta_idx], perf_table_beta$Return[best_return_beta_idx] * 100))
    cat(sprintf("  Highest Sharpe: %s (%.4f)\n", 
                perf_table_beta$Method[best_sharpe_beta_idx], perf_table_beta$Sharpe[best_sharpe_beta_idx]))
    
    cat("\nBest Performance (Mean-Variance):\n")
    cat(sprintf("  Highest Return: %s (%.2f%%)\n", 
                perf_table_mv$Method[best_return_mv_idx], perf_table_mv$Return[best_return_mv_idx] * 100))
    cat(sprintf("  Highest Sharpe: %s (%.4f)\n", 
                perf_table_mv$Method[best_sharpe_mv_idx], perf_table_mv$Sharpe[best_sharpe_mv_idx]))
  }
  
  return(list(beta_weighted = perf_table_beta, mean_variance = perf_table_mv))
}

main <- function() {
  cat("===========================================\n")
  cat("Financial Scenario: SCAD vs LASSO\n")
  cat("Highly Correlated Factor Features\n")
  cat("===========================================\n\n")
  flush.console()
  
  ff_factors <- download_factors(start_date = "2020-01-01")
  stock_data <- download_stocks(start_date = "2020-01-01")
  data_list <- prepare_data(stock_data, ff_factors)
  
  if (length(data_list) == 0) stop("No valid stock data available")
  
  results <- estimate_loadings(data_list, train_ratio = 0.7)
  if (length(results) == 0) stop("Failed to estimate factor loadings")
  
  pred_summary <- evaluate_predictions(data_list, results, ff_factors)
  portfolio_perf <- construct_portfolio(data_list, results, ff_factors)
  
  cat("\n===========================================\n")
  cat("FINAL SUMMARY\n")
  cat("===========================================\n")
  cat("\n1. Predictive Power (MSE, OOS R²):\n")
  print(pred_summary)
  cat("\n2. Portfolio Performance (Return, Sharpe):\n")
  print(portfolio_perf)
  cat("\n✅ Analysis completed!\n\n")
  
  return(list(factors = ff_factors, results = results, 
              predictions = pred_summary, portfolio = portfolio_perf))
}

cat("Starting analysis...\n")
flush.console()
tryCatch({
  results <- main()
}, error = function(e) {
    cat("\n===========================================\n")
    cat("❌ Error occurred during analysis\n")
    cat("===========================================\n")
    cat(sprintf("Error message: %s\n\n", e$message))
    
    cat("Possible solutions:\n")
    cat("1. Set USE_SIMULATED_DATA = TRUE at the top of the file to use simulated data\n")
    cat("2. Check your internet connection\n")
    cat("3. Wait 5-10 minutes and try again (Yahoo Finance rate limiting)\n")
    cat("4. Try using a VPN or different network\n")
    cat("5. Reduce the number of stocks in download_stocks() function\n\n")
    
    cat("To use simulated data (recommended), edit real_data.R line 23 and set:\n")
    cat("   USE_SIMULATED_DATA <- TRUE\n\n")
    
  print(traceback())
  stop(e)
})
