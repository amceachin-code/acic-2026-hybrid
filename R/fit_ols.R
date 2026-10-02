# ===========================================================================
# R/fit_ols.R — OLS-based CATE estimation for multi-arm treatment
#
# Provides three OLS estimators for 5-arm treatment (z = a,b,c,d,e):
#   OLS 1: Y ~ X + D              (constant CATEs = ATE only)
#   OLS 2: Y ~ X + D + X:D        (full heterogeneous CATEs)
#   OLS 3: Stepwise AIC from OLS 2 (selected heterogeneous CATEs)
#
# All functions accept (train_df, test_df) raw data frames and return:
#   list(tau_hat, tau_L, tau_U)  — each N_test × 4 matrix
#   Columns correspond to contrasts: b-a, c-a, d-a, e-a
#
# Shared utility:
#   build_linear_data(df) — constructs design matrices from raw data
#
# Dependencies: sandwich (for HC1 robust SEs), prepare_covariates.R
#
# Usage:
#   source("R/prepare_covariates.R")
#   source("R/fit_ols.R")
#   result <- fit_ols1(train_df, test_df)
# ===========================================================================


# ---------------------------------------------------------------------------
# build_linear_data(df)
#
# Constructs all the pieces needed for OLS regression from a raw data frame:
#   Y         — n-vector of outcomes
#   X         — n × 54 covariate matrix (from prepare_covariates)
#   D_mat     — n × 4 treatment dummy matrix (D_b, D_c, D_d, D_e)
#   X_D_inter — n × 216 interaction matrix (each of 54 X cols × 4 D dummies)
#   col_map   — named list mapping each contrast to its coefficient indices
#               in the full OLS 2 model (intercept + 54 X + 4 D + 216 X:D)
#
# The col_map is critical for extracting CATEs via the delta method.
# For contrast w, CATE = gamma_w + x_i' * delta_w, where:
#   gamma_w = coefficient on D_w (the main effect of treatment arm w)
#   delta_w = coefficients on X:D_w interactions (54 coefficients)
# ---------------------------------------------------------------------------
build_linear_data <- function(df) {
  # Outcome vector
  Y <- df$y

  # Covariate matrix via the shared encoder (N × 54)
  X <- prepare_covariates(df)
  p_x <- ncol(X)  # Should be 54

  # Treatment dummies: D_b, D_c, D_d, D_e (reference = 'a')
  arms <- c("b", "c", "d", "e")
  D_mat <- sapply(arms, function(w) as.numeric(df$z == w))
  colnames(D_mat) <- paste0("D_", arms)

  # Interaction columns: X_j * D_w for all j in 1:54, w in {b,c,d,e}
  # Layout: first 54 cols are X:D_b, next 54 are X:D_c, etc.
  # Vectorized: sweep each D_w across all X columns simultaneously
  inter_parts <- lapply(seq_along(arms), function(w_idx) {
    block <- sweep(X, 1, D_mat[, w_idx], "*")
    colnames(block) <- paste0(colnames(X), ":D_", arms[w_idx])
    block
  })
  X_D_inter <- do.call(cbind, inter_parts)

  # Build column map for the FULL OLS 2 model:
  # Columns: [intercept] [X_1 ... X_54] [D_b D_c D_d D_e] [X:D_b ... X:D_e]
  # Indices: 1           2:55            56:59              60:275
  #
  # For contrast w (w_idx = 1,2,3,4 for b,c,d,e):
  #   gamma_w is at index 55 + w_idx (i.e., 56, 57, 58, 59)
  #   delta_w is at indices 59 + (w_idx-1)*54 + 1:54
  col_map <- list()
  for (w_idx in seq_along(arms)) {
    w <- arms[w_idx]
    gamma_idx <- 55 + w_idx                           # D_w main effect
    delta_idx <- 59 + (w_idx - 1) * p_x + (1:p_x)    # X:D_w interactions
    col_map[[w]] <- list(gamma = gamma_idx, delta = delta_idx)
  }

  list(
    Y = Y,
    X = X,
    D_mat = D_mat,
    X_D_inter = X_D_inter,
    col_map = col_map
  )
}


# ---------------------------------------------------------------------------
# fit_ols1(train_df, test_df)
#
# OLS 1: Main effects only — Y ~ X + D
#   - 59 parameters: intercept + 54 covariates + 4 treatment dummies
#   - CATE is CONSTANT for all individuals: tau_w(x) = gamma_w (the ATE)
#   - CIs use HC1 robust standard errors from sandwich::vcovHC
#   - The same CI applies to every test observation (no heterogeneity)
#
# Returns: list(tau_hat, tau_L, tau_U) — each N_test × 4
# ---------------------------------------------------------------------------
fit_ols1 <- function(train_df, test_df) {
  # Build design matrices for training data
  train_data <- build_linear_data(train_df)

  # Construct design matrix and fit OLS via lm (needed for sandwich::vcovHC)
  X_design <- cbind(train_data$X, train_data$D_mat)
  df_train <- data.frame(y = train_data$Y, X_design)
  fit_lm <- lm(y ~ ., data = df_train)

  # Extract coefficients for the 4 treatment dummies
  # In the lm formula y ~ ., the intercept is position 1, X_1...X_54 at 2:55,
  # D_b at 56, D_c at 57, D_d at 58, D_e at 59
  n_x <- ncol(train_data$X)  # 54
  gamma_idx <- (n_x + 2):(n_x + 5)  # columns 56:59
  gamma_hat <- coef(fit_lm)[gamma_idx]

  # HC1 robust standard errors for the treatment dummy coefficients
  V <- sandwich::vcovHC(fit_lm, type = "HC1")
  gamma_se <- sqrt(diag(V)[gamma_idx])

  # CATEs are constant: tau_w(x_i) = gamma_w for all i
  n_test <- nrow(test_df)
  tau_hat <- matrix(rep(gamma_hat, each = n_test), nrow = n_test, ncol = 4)
  tau_L   <- matrix(rep(gamma_hat - 1.96 * gamma_se, each = n_test),
                    nrow = n_test, ncol = 4)
  tau_U   <- matrix(rep(gamma_hat + 1.96 * gamma_se, each = n_test),
                    nrow = n_test, ncol = 4)

  colnames(tau_hat) <- colnames(tau_L) <- colnames(tau_U) <- c("b", "c", "d", "e")

  list(tau_hat = tau_hat, tau_L = tau_L, tau_U = tau_U)
}


# ---------------------------------------------------------------------------
# fit_ols2(train_df, test_df)
#
# OLS 2: Full interactions — Y ~ X + D + X:D
#   - 275 parameters: intercept + 54 X + 4 D + 216 X:D
#   - n/p ratio = 1000/275 ≈ 3.6 (borderline but estimable)
#   - CATE: tau_w(x_i) = gamma_w + x_i' * delta_w (heterogeneous)
#   - CIs via delta method with HC1 sandwich covariance:
#       For each test obs x_i and contrast w:
#         c_i = contrast vector (1 at gamma_w, x_ij at delta_wj, 0 elsewhere)
#         se_i = sqrt(c_i' V c_i)
#     Vectorized: build C matrix (N_test × p), compute C %*% V, then
#     rowSums of elementwise product with C to get variances.
#
# Returns: list(tau_hat, tau_L, tau_U) — each N_test × 4
# ---------------------------------------------------------------------------
fit_ols2 <- function(train_df, test_df) {
  # Build design matrices
  train_data <- build_linear_data(train_df)
  X_test <- prepare_covariates(test_df)

  # Full design matrix: [intercept, X, D, X:D] — 275 columns
  X_design <- cbind(1, train_data$X, train_data$D_mat, train_data$X_D_inter)
  colnames(X_design)[1] <- "intercept"
  p <- ncol(X_design)  # Should be 275

  # Fit OLS
  df_train <- data.frame(y = train_data$Y, X_design[, -1, drop = FALSE])
  fit_lm <- lm(y ~ ., data = df_train)
  beta <- coef(fit_lm)

  # HC1 sandwich covariance matrix (275 × 275)
  V <- sandwich::vcovHC(fit_lm, type = "HC1")

  # Extract CATEs and CIs for each contrast
  n_test <- nrow(X_test)
  arms <- c("b", "c", "d", "e")
  tau_hat <- tau_L <- tau_U <- matrix(NA, n_test, 4)
  colnames(tau_hat) <- colnames(tau_L) <- colnames(tau_U) <- arms

  for (w_idx in 1:4) {
    w <- arms[w_idx]
    gamma_idx <- train_data$col_map[[w]]$gamma   # scalar index
    delta_idx <- train_data$col_map[[w]]$delta    # 54 indices

    # Point estimates: tau_w(x_i) = gamma_w + x_i' * delta_w
    gamma_w <- beta[gamma_idx]
    delta_w <- beta[delta_idx]
    tau_hat[, w_idx] <- gamma_w + X_test %*% delta_w

    # Delta method CIs:
    # Build the N_test × p contrast matrix C where:
    #   C[i, gamma_idx] = 1
    #   C[i, delta_idx] = X_test[i, ]
    #   C[i, everything else] = 0
    C <- matrix(0, nrow = n_test, ncol = p)
    C[, gamma_idx] <- 1
    C[, delta_idx] <- X_test

    # Variance: diag(C %*% V %*% t(C)) computed efficiently as
    # rowSums((C %*% V) * C)
    CV <- C %*% V
    var_tau <- rowSums(CV * C)
    se_tau <- sqrt(pmax(var_tau, 0))  # Guard against numerical negatives

    tau_L[, w_idx] <- tau_hat[, w_idx] - 1.96 * se_tau
    tau_U[, w_idx] <- tau_hat[, w_idx] + 1.96 * se_tau
  }

  list(tau_hat = tau_hat, tau_L = tau_L, tau_U = tau_U)
}


# ---------------------------------------------------------------------------
# fit_ols3(train_df, test_df)
#
# OLS 3: Stepwise AIC from OLS 2
#   - Starts from the full OLS 2 model (275 params)
#   - Uses step() with direction="backward" and AIC criterion (k=2)
#   - Lower scope: main effects only (Y ~ X + D, 59 params)
#   - Only X:D interaction terms can be dropped
#   - max_steps = 100 to bound runtime
#   - CATE extraction: same delta method as OLS 2, but only for
#     interaction terms that survived selection
#
# Returns: list(tau_hat, tau_L, tau_U) — each N_test × 4
# ---------------------------------------------------------------------------
fit_ols3 <- function(train_df, test_df) {
  # Build design matrices
  train_data <- build_linear_data(train_df)
  X_test <- prepare_covariates(test_df)

  # Full design matrix: [intercept, X, D, X:D] — 275 columns
  X_full <- cbind(1, train_data$X, train_data$D_mat, train_data$X_D_inter)
  colnames(X_full)[1] <- "intercept"

  # Create data frames for lm() — need formula interface for step()
  train_df_lm <- data.frame(y = train_data$Y, X_full[, -1, drop = FALSE])

  # Sanitize column names for formula compatibility
  # (colons in interaction names need to be replaced)
  orig_names <- colnames(train_df_lm)[-1]  # skip 'y'
  safe_names <- gsub(":", "_X_", orig_names)
  safe_names <- gsub(" ", "_", safe_names)
  colnames(train_df_lm)[-1] <- safe_names

  # Identify which columns are main effects vs interactions
  # Main effects: columns 1:58 (54 X + 4 D) after removing intercept
  # Interactions: columns 59:274 (216 X:D terms)
  n_x <- ncol(train_data$X)       # 54
  n_d <- ncol(train_data$D_mat)   # 4
  n_main <- n_x + n_d             # 58

  main_vars <- safe_names[1:n_main]
  inter_vars <- safe_names[(n_main + 1):length(safe_names)]

  # Full model formula
  full_formula <- as.formula(paste("y ~", paste(safe_names, collapse = " + ")))

  # Lower scope: main effects only (these are always kept)
  lower_formula <- as.formula(paste("y ~", paste(main_vars, collapse = " + ")))

  # Fit full model
  fit_full <- lm(full_formula, data = train_df_lm)

  # Stepwise selection: backward elimination with AIC criterion
  # direction="backward" is much faster than "both" (no re-addition scans)
  # since we have 216 candidate interaction terms to evaluate per step.
  # Suppress output with trace=0, bound iterations with steps=100
  fit_step <- step(fit_full,
                   direction = "backward",
                   scope = list(lower = lower_formula),
                   trace = 0,
                   k = 2,
                   steps = 100)

  # HC1 robust covariance for the selected model
  V_step <- sandwich::vcovHC(fit_step, type = "HC1")
  beta_step <- coef(fit_step)
  selected_names <- names(beta_step)  # includes "(Intercept)"

  # Extract CATEs using delta method on the selected model
  # For each contrast w:
  #   tau_w(x_i) = gamma_w + sum over surviving X:D_w interactions
  #   The contrast vector c_i has:
  #     1 at the gamma_w position (if D_w survived — it always does, it's main effect)
  #     x_ij at each surviving X_j:D_w position
  #     0 everywhere else
  n_test <- nrow(X_test)
  arms <- c("b", "c", "d", "e")
  p_step <- length(beta_step)

  tau_hat <- tau_L <- tau_U <- matrix(NA, n_test, 4)
  colnames(tau_hat) <- colnames(tau_L) <- colnames(tau_U) <- arms

  # Build mapping from safe column names to X column indices
  x_colnames <- colnames(train_data$X)
  safe_x_names <- gsub(" ", "_", x_colnames)

  for (w_idx in 1:4) {
    w <- arms[w_idx]
    d_name <- paste0("D_", w)

    # Find gamma_w position in the selected model's coefficient vector
    gamma_pos <- which(selected_names == d_name)
    if (length(gamma_pos) == 0) {
      # Treatment main effect was somehow dropped (shouldn't happen with lower scope)
      # Fall back to zero CATE with infinite CI
      tau_hat[, w_idx] <- 0
      tau_L[, w_idx] <- -Inf
      tau_U[, w_idx] <- Inf
      next
    }

    gamma_w <- beta_step[gamma_pos]

    # Find surviving interaction terms for this contrast
    # They have names like "X_colname_X_D_w" (where _X_ replaced :)
    inter_pattern <- paste0("_X_D_", w, "$")
    inter_selected <- grep(inter_pattern, selected_names, value = TRUE)
    inter_pos <- match(inter_selected, selected_names)

    # Map each surviving interaction back to its X column index
    # inter name format: "{safe_x_name}_X_D_{w}"
    # Strip the "_X_D_{w}" suffix to get the safe X name
    suffix <- paste0("_X_D_", w)
    x_safe_for_inter <- sub(paste0(suffix, "$"), "", inter_selected)
    x_col_idx <- match(x_safe_for_inter, safe_x_names)

    # Build contrast matrix C: N_test × p_step
    C <- matrix(0, nrow = n_test, ncol = p_step)
    C[, gamma_pos] <- 1

    # Point estimate: tau_w(x_i) = gamma_w + sum_j(delta_wj * x_ij) for surviving j
    tau_hat[, w_idx] <- gamma_w
    if (length(inter_pos) > 0) {
      delta_w <- beta_step[inter_pos]
      X_sub <- X_test[, x_col_idx, drop = FALSE]
      tau_hat[, w_idx] <- tau_hat[, w_idx] + X_sub %*% delta_w
      C[, inter_pos] <- X_sub
    }

    # Delta method variance
    CV <- C %*% V_step
    var_tau <- rowSums(CV * C)
    se_tau <- sqrt(pmax(var_tau, 0))

    tau_L[, w_idx] <- tau_hat[, w_idx] - 1.96 * se_tau
    tau_U[, w_idx] <- tau_hat[, w_idx] + 1.96 * se_tau
  }

  list(tau_hat = tau_hat, tau_L = tau_L, tau_U = tau_U)
}
