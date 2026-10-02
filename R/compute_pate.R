# ===========================================================================
# R/compute_pate.R -- PATE computation from Horseshoe posterior draws
#
# Computes the Population Average Treatment Effect (PATE) by leveraging
# the linearity of the Horseshoe CATE model:
#
#   CATE_w(x) = gamma_w + x' * delta_w
#
# Therefore:
#   PATE_w = E_X[CATE_w(X)] = gamma_w + E[X]' * delta_w
#          = gamma_w + x_bar_pop' * delta_w
#
# For each MCMC draw s, computing PATE is a single dot product. This makes
# the HS PATE essentially free (274 x 3000 multiplies vs. the full matrix
# multiplication that would be needed for a nonlinear model).
#
# Dependencies: None (standalone function, requires beta_draws from
#               fit_horseshoe.R with return_draws = TRUE)
#
# Usage:
#   source("R/compute_pate.R")
#   pate <- compute_pate_from_hs_posterior(beta_draws, x_bar_pop)
# ===========================================================================


# ---------------------------------------------------------------------------
# compute_pate_from_hs_posterior(beta_draws, x_bar_pop)
#
# Compute PATE for all 4 contrasts (b, c, d, e vs a) from the rescaled
# posterior draws of the Horseshoe regression coefficients.
#
# Design matrix layout (hardcoded, fixed by ACIC design):
#   Columns 1:54    = X (covariate main effects)
#   Columns 55:58   = D (treatment dummies: D_b, D_c, D_d, D_e)
#   Columns 59:274  = X:D (interactions, 54 per arm x 4 arms)
#
# For contrast w (w_idx = 1..4):
#   gamma_col  = 54 + w_idx          (treatment dummy main effect)
#   delta_cols = 58 + (w_idx-1)*54 + (1:54)  (interaction coefficients)
#   PATE_w^(s) = gamma_w^(s) + x_bar_pop' * delta_w^(s)
#
# Arguments:
#   beta_draws -- p x n_mcmc matrix of rescaled posterior draws from
#                 fit_horseshoe.R (p = 274 = 54 + 4 + 216)
#   x_bar_pop  -- length-54 vector of population covariate means
#                 (from build_x_population.R)
#
# Returns: list with elements for each contrast (b, c, d, e):
#   pate_est   -- length-4 vector of PATE point estimates (posterior means)
#   pate_se    -- length-4 vector of PATE standard errors (posterior SDs)
#   pate_L95   -- length-4 vector of lower 95% CI bounds (2.5th percentile)
#   pate_U95   -- length-4 vector of upper 95% CI bounds (97.5th percentile)
#   pate_draws -- 4 x n_mcmc matrix of PATE posterior draws (row per contrast)
# ---------------------------------------------------------------------------
compute_pate_from_hs_posterior <- function(beta_draws, x_bar_pop) {
  # Hardcoded layout constants. These are fixed by the ACIC 2026 design:
  # 54 covariates (after one-hot encoding), 4 treatment contrasts.
  # Hardcoding avoids threading these through multiple function signatures.
  n_x <- 54L
  n_d <- 4L
  p_expected <- n_x + n_d + n_x * n_d  # 274

  # Assertions: verify dimensions match the expected design

  stopifnot(
    is.matrix(beta_draws),
    nrow(beta_draws) == p_expected,
    length(x_bar_pop) == n_x
  )

  arms <- c("b", "c", "d", "e")
  n_mcmc <- ncol(beta_draws)

  # Pre-allocate output
  pate_est <- pate_se <- numeric(n_d)
  names(pate_est) <- names(pate_se) <- arms
  pate_draws_mat <- matrix(NA_real_, nrow = n_d, ncol = n_mcmc)
  rownames(pate_draws_mat) <- arms

  for (w_idx in 1:n_d) {
    # Column indices in the beta_draws matrix (same layout as fit_horseshoe.R)
    gamma_col  <- n_x + w_idx
    delta_cols <- n_x + n_d + (w_idx - 1L) * n_x + (1:n_x)

    # For each MCMC draw s:
    #   PATE_w^(s) = gamma_w^(s) + x_bar_pop' * delta_w^(s)
    #
    # gamma_draws: length n_mcmc vector
    # delta_draws: n_x x n_mcmc matrix
    # x_bar_pop' %*% delta_draws: 1 x n_mcmc matrix (dot product per draw)
    gamma_draws <- beta_draws[gamma_col, ]
    delta_draws <- beta_draws[delta_cols, , drop = FALSE]

    # Vectorized dot product across all MCMC draws
    pate_draws_w <- gamma_draws + as.numeric(x_bar_pop %*% delta_draws)

    pate_draws_mat[w_idx, ] <- pate_draws_w
    pate_est[w_idx] <- mean(pate_draws_w)
    pate_se[w_idx]  <- sd(pate_draws_w)
  }

  # CIs from posterior quantiles (raw, before z-multiplier adjustment).
  # The caller (fit_hybrid.R) applies PATE_Z_MULT for the final output.
  pate_L95 <- apply(pate_draws_mat, 1, quantile, probs = 0.025)
  pate_U95 <- apply(pate_draws_mat, 1, quantile, probs = 0.975)
  names(pate_L95) <- names(pate_U95) <- arms

  list(
    pate_est   = pate_est,
    pate_se    = pate_se,
    pate_L95   = pate_L95,
    pate_U95   = pate_U95,
    pate_draws = pate_draws_mat
  )
}


# ---------------------------------------------------------------------------
# compute_hybrid_pate(pate_hs, scate_grf, scate_se,
#                     w_hs_rmse, w_grf_rmse)
#
# Combine HS PATE and GRF sCATE (as GRF PATE proxy) into a hybrid PATE
# using inverse-RMSE weights. Shared by fit_hybrid.R and experiment scripts
# to avoid duplicating the weighting logic.
#
# Arguments:
#   pate_hs    -- list from compute_pate_from_hs_posterior()
#                 (must contain $pate_est and $pate_se)
#   scate_grf  -- length-4 vector: GRF sCATE (proxy for GRF PATE)
#   scate_se   -- length-4 vector: sCATE SE (proxy for GRF PATE SE)
#   w_hs_rmse  -- Horseshoe RMSE for inverse weighting
#   w_grf_rmse -- GRF RMSE for inverse weighting
#
# Returns: list(pate_est, pate_se) -- each length-4 vectors
# ---------------------------------------------------------------------------
compute_hybrid_pate <- function(pate_hs, scate_grf, scate_se,
                                 w_hs_rmse, w_grf_rmse) {
  w_hs  <- 1 / w_hs_rmse
  w_grf <- 1 / w_grf_rmse
  w_sum <- w_hs + w_grf

  pate_est <- (w_hs * pate_hs$pate_est + w_grf * scate_grf) / w_sum

  # Propagate both HS and GRF variance through ensemble weights.
  # HS variance: from posterior draws of population-averaged CATE.
  # GRF variance: approximate with sCATE SE (since GRF PATE ≈ GRF sCATE).
  pate_se <- sqrt((w_hs / w_sum)^2 * pate_hs$pate_se^2 +
                  (w_grf / w_sum)^2 * scate_se^2)

  list(pate_est = pate_est, pate_se = pate_se)
}
