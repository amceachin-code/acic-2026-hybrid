# ===========================================================================
# R/fit_horseshoe.R — Horseshoe prior CATE estimation
#
# Implements Bayesian linear regression with the horseshoe prior
# (Carvalho, Polson & Scott, 2010) on the full interaction model
# Y ~ X + D + X:D (274 predictors, no explicit intercept).
#
# Uses a custom Gibbs sampler (horseshoe_gibbs.R) that supports
# per-parameter penalization control. By default:
#   - Covariate main effects (X, 54 cols): UNPENALIZED (flat prior)
#   - Treatment dummies (D, 4 cols):       UNPENALIZED (flat prior)
#   - Treatment × covariate interactions (X:D, 216 cols): PENALIZED (horseshoe)
#
# This means the treatment main effects (ATEs) are estimated without
# any shrinkage bias, while the heterogeneous interaction terms are
# regularized by the horseshoe to avoid overfitting in the n/p ≈ 3.6 regime.
#
# Dependencies: horseshoe_gibbs.R, prepare_covariates.R, fit_ols.R
#               (for build_linear_data)
#
# Usage:
#   source("R/prepare_covariates.R")
#   source("R/fit_ols.R")
#   source("R/horseshoe_gibbs.R")
#   source("R/fit_horseshoe.R")
#   result <- fit_horseshoe(train_df, test_df)
# ===========================================================================


# ---------------------------------------------------------------------------
# fit_horseshoe(train_df, test_df, n_mcmc, burnin, penalize_X, penalize_D)
#
# Fits the horseshoe prior via the custom Gibbs sampler on the full
# interaction design matrix (X + D + X:D = 274 columns after centering
# the outcome to remove the need for an explicit intercept).
#
# The `penalize_X` and `penalize_D` arguments control whether covariate
# main effects and treatment dummies are shrunk. The X:D interactions
# are ALWAYS penalized (that's the whole point — regularize the
# heterogeneous effects).
#
# CATE extraction from posterior draws:
#   For each MCMC draw s and contrast w:
#     tau_w^(s)(x_i) = beta_{gamma_w}^(s) + X_test[i,] %*% beta_{delta_w}^(s)
#   Vectorized: tau_draws is n_mcmc × N_test
#
# Point estimates: posterior mean
# CIs: 2nd and 98th percentiles of the posterior (96% credible intervals)
#
# Parameters:
#   train_df    — training data frame (with y, z, x1-x40)
#   test_df     — test data frame (with x1-x40)
#   n_mcmc      — number of post-burnin MCMC samples (default 1000)
#   burnin      — number of burnin samples to discard (default 500)
#   thin        — thinning interval for MCMC chain (default 1)
#   penalize_X  — logical, should covariate main effects (54 cols) be
#                 penalized? Default FALSE (flat prior on X).
#   penalize_D  — logical, should treatment dummies (4 cols) be
#                 penalized? Default FALSE (flat prior on D).
#   penalized    — optional logical vector of length 274 for full manual
#                  control. Overrides penalize_X and penalize_D if provided.
#   lambda_scale — scale of the half-Cauchy on local shrinkage lambda_j.
#                  Larger = wider local tails = signals escape easier.
#                  Default 1 (standard horseshoe).
#   tau_scale    — scale of the half-Cauchy on global shrinkage tau.
#                  Smaller = tighter global shrinkage. Default 1.
#   return_draws — logical, if TRUE include the full n_test × n_mcmc × 4
#                  array of posterior CATE draws in the return list.
#                  Used by compute_p_best() for P(best) treatment selection.
#                  Default FALSE (backward compatible: draws are discarded).
#   verbose      — logical, print MCMC progress (default TRUE)
#
# Returns: list(tau_hat, tau_L, tau_U) — each N_test × 4, or NAs if error
#          When return_draws = TRUE, also includes tau_draws_all (N_test × n_mcmc × 4)
# ---------------------------------------------------------------------------
fit_horseshoe <- function(train_df, test_df,
                          n_mcmc = 1000, burnin = 500, thin = 1,
                          penalize_X = FALSE, penalize_D = FALSE,
                          penalized = NULL,
                          lambda_scale = 1, tau_scale = 1,
                          return_draws = FALSE,
                          verbose = TRUE) {
  n_test <- nrow(test_df)
  arms <- c("b", "c", "d", "e")

  # Build design matrices using the shared builder from fit_ols.R
  train_data <- build_linear_data(train_df)
  X_test <- prepare_covariates(test_df)

  # Design matrix: [X, D, X:D] — 274 columns (no intercept column;
  # we center Y to remove the need for an intercept)
  X_design <- cbind(train_data$X, train_data$D_mat, train_data$X_D_inter)

  n_x     <- ncol(train_data$X)          # 54 covariate columns
  n_d     <- ncol(train_data$D_mat)      # 4 treatment dummy columns
  n_inter <- ncol(train_data$X_D_inter)  # 216 interaction columns
  p_total <- n_x + n_d + n_inter         # 274

  # -----------------------------------------------------------------------
  # Build the penalized vector
  #
  # Column layout of X_design:
  #   Columns 1:54    = X (covariate main effects)
  #   Columns 55:58   = D (treatment dummies: D_b, D_c, D_d, D_e)
  #   Columns 59:274  = X:D (interactions)
  #
  # Default: penalize only the X:D interactions. The user can override
  # this via penalize_X, penalize_D, or the full penalized vector.
  # -----------------------------------------------------------------------
  if (is.null(penalized)) {
    penalized <- c(
      rep(penalize_X, n_x),      # covariate main effects
      rep(penalize_D, n_d),      # treatment dummies
      rep(TRUE, n_inter)         # interactions — always penalized
    )
  }

  if (length(penalized) != p_total) {
    stop("penalized vector length (", length(penalized),
         ") != number of design columns (", p_total, ")")
  }

  # Log what's being penalized for transparency
  n_pen <- sum(penalized)
  n_free <- sum(!penalized)
  if (verbose) {
    cat(sprintf("  [Horseshoe] Design: %d cols (%d penalized, %d free)\n",
                p_total, n_pen, n_free))
    cat(sprintf("  [Horseshoe]   X main effects (%d cols): %s\n",
                n_x, ifelse(penalize_X || (!is.null(penalized) && any(penalized[1:n_x])),
                             "PENALIZED", "UNPENALIZED")))
    cat(sprintf("  [Horseshoe]   D treatment (%d cols):    %s\n",
                n_d, ifelse(penalize_D || (!is.null(penalized) && any(penalized[(n_x+1):(n_x+n_d)])),
                             "PENALIZED", "UNPENALIZED")))
    cat(sprintf("  [Horseshoe]   X:D interactions (%d cols): PENALIZED\n", n_inter))
  }

  # Center Y for the no-intercept model
  Y_mean <- mean(train_data$Y)
  Y_centered <- train_data$Y - Y_mean

  # Scale columns of X_design for better MCMC mixing
  # Store the scaling factors to transform posterior draws back
  X_sd <- apply(X_design, 2, sd)
  # Avoid division by zero for constant columns (e.g., rare dummies)
  X_sd[X_sd < 1e-10] <- 1
  X_scaled <- sweep(X_design, 2, X_sd, "/")

  # -----------------------------------------------------------------------
  # Run the custom Gibbs sampler
  # -----------------------------------------------------------------------
  if (verbose) {
    cat(sprintf("  [Horseshoe]   lambda_scale = %.2f, tau_scale = %.4f\n",
                lambda_scale, tau_scale))
  }

  hs_fit <- horseshoe_gibbs(
    y            = Y_centered,
    X            = X_scaled,
    penalized    = penalized,
    lambda_scale = lambda_scale,
    tau_scale    = tau_scale,
    n_mcmc       = n_mcmc,
    burnin       = burnin,
    thin         = thin,
    verbose      = verbose
  )

  # -----------------------------------------------------------------------
  # Rescale posterior draws back to original (unscaled) parameterization
  #
  # If X_scaled = X_design / diag(X_sd), then beta_orig = beta_scaled / X_sd
  # because y = X_scaled %*% beta_scaled = X_design %*% (beta_scaled / X_sd)
  # -----------------------------------------------------------------------
  beta_draws <- sweep(hs_fit$beta_draws, 1, X_sd, "/")  # p × n_mcmc

  # -----------------------------------------------------------------------
  # Extract CATEs from posterior draws
  #
  # For contrast w (vs reference arm 'a'):
  #   tau_w^(s)(x_i) = gamma_w^(s) + x_i' %*% delta_w^(s)
  #
  # gamma_w = coefficient on D_w (treatment dummy main effect)
  # delta_w = coefficients on X:D_w (54 interaction terms for arm w)
  #
  # In our no-intercept design [X(54), D(4), X:D(216)]:
  #   D_w is at column n_x + w_idx
  #   X:D_w interactions are at columns n_x + n_d + (w_idx-1)*n_x + (1:n_x)
  # -----------------------------------------------------------------------
  tau_hat <- tau_L <- tau_U <- matrix(NA, n_test, 4)
  colnames(tau_hat) <- colnames(tau_L) <- colnames(tau_U) <- arms

  # Pre-allocate array to store full posterior draws when requested.
  # Dimensions: n_test x n_mcmc x 4 (one slice per contrast).
  # This array is populated inside the loop below and returned when
  # return_draws = TRUE; otherwise it is discarded after summarization.
  n_mcmc_actual <- ncol(beta_draws)  # actual number of post-burnin draws
  if (return_draws) {
    tau_draws_all <- array(NA, dim = c(n_test, n_mcmc_actual, 4))
  }

  for (w_idx in 1:4) {
    w <- arms[w_idx]

    # Column indices in the design matrix
    gamma_col <- n_x + w_idx                              # D_w position
    delta_cols <- n_x + n_d + (w_idx - 1) * n_x + (1:n_x)  # X:D_w positions

    # Extract posterior draws for gamma_w and delta_w
    gamma_draws <- beta_draws[gamma_col, ]       # n_mcmc vector
    delta_draws <- beta_draws[delta_cols, ]       # n_x × n_mcmc matrix

    # Compute CATE draws for each test observation:
    # tau_w^(s)(x_i) = gamma_w^(s) + x_i' %*% delta_w^(s)
    # Result: N_test × n_mcmc matrix
    tau_draws <- sweep(X_test %*% delta_draws, 2, gamma_draws, "+")

    # Store draws in the 3D array (k-th contrast slice)
    if (return_draws) {
      tau_draws_all[, , w_idx] <- tau_draws
    }

    # Point estimate: posterior mean across draws
    tau_hat[, w_idx] <- rowMeans(tau_draws)

    # 96% credible intervals: 2nd and 98th percentiles
    # Moderately widened from 95% to compensate for center mismatch between
    # Horseshoe point estimates and the ensemble average. More conservative
    # than 98% CIs which over-corrected in validation (99.9% coverage).
    tau_L[, w_idx] <- apply(tau_draws, 1, quantile, probs = 0.02)
    tau_U[, w_idx] <- apply(tau_draws, 1, quantile, probs = 0.98)
  }

  result <- list(tau_hat = tau_hat, tau_L = tau_L, tau_U = tau_U)

  # When return_draws = TRUE, include the full n_test x n_mcmc x 4 array
  # of posterior CATE draws. Used by compute_p_best() for P(best) selection.
  # Also include beta_draws (p x n_mcmc rescaled coefficient matrix) for
  # PATE computation via compute_pate_from_hs_posterior(). Memory: ~6.6 MB.
  if (return_draws) {
    result$tau_draws_all <- tau_draws_all
    result$beta_draws    <- beta_draws  # p(274) x n_mcmc, rescaled to original scale
  }

  result
}
