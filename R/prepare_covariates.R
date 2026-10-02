# ===========================================================================
# R/prepare_covariates.R — Shared covariate preparation
#
# Provides the canonical covariate encoding used by ALL methods:
#   - Categorical (x1-x6): one-hot encode, drop level 1 as reference
#   - Binary (x7-x25): as numeric 0/1
#   - Continuous (x26-x40): as numeric
#
# Result: N × 54 numeric matrix (20 one-hot + 19 binary + 15 continuous)
#
# Usage:
#   source("R/prepare_covariates.R")
#   X <- prepare_covariates(df)
# ===========================================================================

# Categorical covariate specifications (name -> number of levels)
cat_specs <- list(
  x1 = 3, x2 = 4, x3 = 3, x4 = 5, x5 = 4, x6 = 7
)

# Binary and continuous covariate names
binary_cols <- paste0("x", 7:25)
continuous_cols <- paste0("x", 26:40)

# Treatment contrasts (all vs control 'a')
contrasts <- c("b", "c", "d", "e")


#' Prepare numeric covariate matrix from raw data frame
#'
#' @param df Data frame with columns x1-x40
#' @return Numeric matrix (N × 54) suitable for all causal methods
prepare_covariates <- function(df) {
  parts <- list()

  # Categorical: one-hot encode, drop level 1 as reference
  for (var_name in names(cat_specs)) {
    n_levels <- cat_specs[[var_name]]
    for (lev in 2:n_levels) {
      col_name <- paste0(var_name, "_level", lev)
      parts[[col_name]] <- as.numeric(df[[var_name]] == paste("level", lev))
    }
  }

  # Binary covariates (x7-x25)
  for (col in binary_cols) {
    parts[[col]] <- as.numeric(df[[col]])
  }

  # Continuous covariates (x26-x40)
  for (col in continuous_cols) {
    parts[[col]] <- as.numeric(df[[col]])
  }

  X <- do.call(cbind, parts)
  return(X)
}


#' Save iCATE, aggregates, and best-treatment results in standard format
#'
#' @param df Original data frame (for ID and x12 columns)
#' @param tau_hat N×4 matrix of iCATE point estimates
#' @param tau_L95 N×4 matrix of iCATE lower 95% CI bounds
#' @param tau_U95 N×4 matrix of iCATE upper 95% CI bounds
#' @param scate_est Length-4 vector of sCATE estimates
#' @param scate_se Length-4 vector of sCATE standard errors
#' @param dataset_id Integer dataset ID
#' @param output_dir Path to method output directory
#' @param t_elapsed Numeric runtime in seconds
#' @return List with timing row data
save_standard_output <- function(df, tau_hat, tau_L95, tau_U95,
                                  scate_est, scate_se,
                                  dataset_id, output_dir, t_elapsed,
                                  p_best_result = NULL,
                                  hs_draws = NULL,
                                  pate_est = NULL,
                                  pate_se = NULL) {
  n <- nrow(df)

  # sCATE CIs: use tuned z-multiplier (SCATE_Z_MULT = 2.0, ~95.3% coverage)
  scate_L95 <- scate_est - SCATE_Z_MULT * scate_se
  scate_U95 <- scate_est + SCATE_Z_MULT * scate_se

  # subCATE: mean of iCATE within x12 strata
  x12_vals <- df$x12
  mask_0 <- which(x12_vals == 0)
  mask_1 <- which(x12_vals == 1)

  subcate_0_est <- colMeans(tau_hat[mask_0, , drop = FALSE])
  subcate_1_est <- colMeans(tau_hat[mask_1, , drop = FALSE])

  # subCATE SE from HS posterior draws (Approach 2) when available.
  # Uses the contest n-denominator formula: subCATE(z,x) = n^{-1} * sum(iCATE * 1(x12==x))
  # Falls back to Approach 1 (from iCATE CIs) if HS draws are unavailable.
  if (!is.null(hs_draws)) {
    subcate_0_se <- compute_aggregate_se_from_draws(
      hs_draws, mask = mask_0, n_total = n, aggregation = "subcate_n")$se
    subcate_1_se <- compute_aggregate_se_from_draws(
      hs_draws, mask = mask_1, n_total = n, aggregation = "subcate_n")$se
  } else {
    subcate_0_se <- compute_aggregate_se_from_ci(tau_L95, tau_U95, mask = mask_0,
                                                  n_total = n, aggregation = "subcate_n")
    subcate_1_se <- compute_aggregate_se_from_ci(tau_L95, tau_U95, mask = mask_1,
                                                  n_total = n, aggregation = "subcate_n")
  }

  # subCATE CIs: use tuned z-multiplier (SUBCATE_Z_MULT = 2.5, ~95.4% coverage)
  subcate_0_L95 <- subcate_0_est - SUBCATE_Z_MULT * subcate_0_se
  subcate_0_U95 <- subcate_0_est + SUBCATE_Z_MULT * subcate_0_se

  subcate_1_L95 <- subcate_1_est - SUBCATE_Z_MULT * subcate_1_se
  subcate_1_U95 <- subcate_1_est + SUBCATE_Z_MULT * subcate_1_se

  # PATE: use provided population-level estimates if available.
  # When pate_est/pate_se are NULL, fall back to sCATE (the old behavior).
  # The proper PATE is computed from HS posterior draws in fit_hybrid.R using
  # compute_pate_from_hs_posterior(), which exploits the linearity of the
  # HS model to evaluate E_X[CATE(X)] = gamma + x_bar_pop' * delta.
  if (is.null(pate_est)) {
    pate_est <- scate_est
    pate_se  <- scate_se
  }
  pate_L95 <- pate_est - PATE_Z_MULT * pate_se
  pate_U95 <- pate_est + PATE_Z_MULT * pate_se

  # Best treatment rankings
  # If max CATE > 0, assign to the best treatment arm; otherwise control 'a'
  # is optimal (all treatments worse than control). This matches the logic
  # in evaluate_multi.R for treatment assignment accuracy and regret.
  best_icates <- apply(tau_hat, 1, function(row) {
    if (max(row) > 0) contrasts[which.max(row)] else "a"
  })
  best_scate <- if (max(scate_est) > 0) contrasts[which.max(scate_est)] else "a"
  best_subcate_0 <- if (max(subcate_0_est) > 0) contrasts[which.max(subcate_0_est)] else "a"
  best_subcate_1 <- if (max(subcate_1_est) > 0) contrasts[which.max(subcate_1_est)] else "a"
  best_pate <- if (max(pate_est) > 0) contrasts[which.max(pate_est)] else "a"

  # Mean CI width for iCATEs (diagnostic for meta-learner)
  ci_widths <- tau_U95 - tau_L95
  mean_ci_width <- mean(ci_widths)

  # --- Save iCATE CSV ---
  icates_df <- data.frame(
    ID = df$ID,
    tau_b = tau_hat[, 1], tau_b_L95 = tau_L95[, 1], tau_b_U95 = tau_U95[, 1],
    tau_c = tau_hat[, 2], tau_c_L95 = tau_L95[, 2], tau_c_U95 = tau_U95[, 2],
    tau_d = tau_hat[, 3], tau_d_L95 = tau_L95[, 3], tau_d_U95 = tau_U95[, 3],
    tau_e = tau_hat[, 4], tau_e_L95 = tau_L95[, 4], tau_e_U95 = tau_U95[, 4]
  )
  write.csv(icates_df,
            file.path(output_dir, sprintf("dataset_%d_icates.csv", dataset_id)),
            row.names = FALSE)

  # --- Save aggregates CSV ---
  agg_rows <- data.frame(
    estimand = c(
      rep("sCATE", 4), rep("subCATE", 8), rep("PATE", 4),
      "BEST_sCATE", "BEST_subCATE", "BEST_subCATE", "BEST_PATE"
    ),
    z = c(
      contrasts, rep(contrasts, 2), contrasts,
      best_scate, best_subcate_0, best_subcate_1, best_pate
    ),
    x = c(
      rep("", 4),
      rep(0, 4), rep(1, 4),
      rep("", 4),
      "", 0, 1, ""
    ),
    Estimate = c(
      scate_est,
      subcate_0_est, subcate_1_est,
      pate_est,
      NA, NA, NA, NA
    ),
    L95 = c(
      scate_L95,
      subcate_0_L95, subcate_1_L95,
      pate_L95,
      NA, NA, NA, NA
    ),
    U95 = c(
      scate_U95,
      subcate_0_U95, subcate_1_U95,
      pate_U95,
      NA, NA, NA, NA
    ),
    stringsAsFactors = FALSE
  )
  write.csv(agg_rows,
            file.path(output_dir, sprintf("dataset_%d_aggregates.csv", dataset_id)),
            row.names = FALSE)

  # --- Save best iCATE per individual (argmax selection) ---
  best_df <- data.frame(ID = df$ID, best_z = best_icates, stringsAsFactors = FALSE)
  write.csv(best_df,
            file.path(output_dir, sprintf("dataset_%d_best_icates.csv", dataset_id)),
            row.names = FALSE)

  # --- Save P(best) results when available ---
  # P(best) provides a second treatment selection method based on the posterior
  # probability of each arm being optimal, computed from Horseshoe MCMC draws.
  # Two files are written:
  #   1. best_icates_pbest.csv: best treatment per individual (P(best) selection)
  #   2. pbest_probs.csv: full N x 5 probability matrix for downstream analysis
  if (!is.null(p_best_result)) {
    # P(best) treatment selection
    pbest_df <- data.frame(
      ID     = df$ID,
      best_z = p_best_result$best_pbest,
      stringsAsFactors = FALSE
    )
    write.csv(pbest_df,
              file.path(output_dir, sprintf("dataset_%d_best_icates_pbest.csv", dataset_id)),
              row.names = FALSE)

    # Full probability matrix
    probs_df <- data.frame(
      ID  = df$ID,
      P_a = p_best_result$P_best[, "a"],
      P_b = p_best_result$P_best[, "b"],
      P_c = p_best_result$P_best[, "c"],
      P_d = p_best_result$P_best[, "d"],
      P_e = p_best_result$P_best[, "e"],
      stringsAsFactors = FALSE
    )
    write.csv(probs_df,
              file.path(output_dir, sprintf("dataset_%d_pbest_probs.csv", dataset_id)),
              row.names = FALSE)
  }

  # --- Return timing row ---
  return(data.frame(
    dataset_id = dataset_id,
    runtime_seconds = t_elapsed,
    scate_b = scate_est[1],
    scate_c = scate_est[2],
    scate_d = scate_est[3],
    scate_e = scate_est[4],
    mean_ci_width_icates = mean_ci_width,
    stringsAsFactors = FALSE
  ))
}
