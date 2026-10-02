# ===========================================================================
# R/fit_hybrid.R -- Hybrid ensemble: Horseshoe + GRF (2-component)
#
# Production wrapper for the Hybrid method.
#
# Combination rule:
#   Point estimates: tau_hat = inverse-RMSE weighted average of HS and GRF
#   Confidence intervals: weighted blend of HS and GRF CIs, conformally
#     calibrated to target 95% coverage.
#
# History: Originally a 3-component ensemble (HS+GRF+BCF, later HS+GRF+XL).
# benchmark_2comp_vs_3comp.R (500 mirror DGP v2 datasets) showed the
# 2-component version (HS+GRF) significantly beats 3-component:
#   RMSE:      0.376 vs 0.394 (p<0.0001), 2-comp wins 79% of datasets
#   Bias:     -0.004 vs +0.002 (p=0.0002)
#   Assign:    0.875 vs 0.870 (p=0.015)
#   Regret:    0.042 vs 0.045 (p=0.005)
# Adding XL (standalone RMSE 0.539) dilutes the two stronger estimators
# (HS=0.419, GRF=0.381), hurting all metrics.
#
# Component methods:
#   1. Horseshoe: Bayesian linear model with horseshoe prior on X:D
#      interactions. Provides both point estimates and posterior CIs.
#      Source: fit_horseshoe.R + horseshoe_gibbs.R
#
#   2. GRF: Multi-arm causal forest (grf::multi_arm_causal_forest).
#      Provides point estimates and variance-based CIs.
#      Source: grf package directly (no helper needed)
#
# Error handling:
#   - If Horseshoe fails: use GRF alone with GRF CIs
#   - If GRF fails:       return NAs (should never happen)
#
# Dependencies (sourced by dispatcher before this file):
#   - prepare_covariates.R (for prepare_covariates(), build_linear_data(),
#     save_standard_output(), contrasts)
#   - fit_ols.R (for build_linear_data())
#   - horseshoe_gibbs.R (Gibbs sampler)
#   - fit_horseshoe.R (horseshoe wrapper)
#
# Libraries (loaded by dispatcher):
#   - grf
#
# Usage:
#   timing_row <- fit_hybrid_dataset(df, dataset_id, output_dir)
# ===========================================================================


# ---------------------------------------------------------------------------
# Production parameters (2-component ensemble: HS + GRF)
# ---------------------------------------------------------------------------

# GRF parameters
HYBRID_GRF_NUM_TREES     <- 2000  # Number of trees in multi-arm causal forest
HYBRID_GRF_MIN_NODE_SIZE <- 20    # Minimum leaf size. Changed from GRF default (5)
                                   # to 20 based on 2x2x2 factorial experiment
                                   # (tune_grf_bcf_factorial.R, 2026-02-28).
                                   # Paired t-test on 100 mirror DGP v2 datasets:
                                   #   MNS=20 vs MNS=5 main effect: p = 0.015
                                   #   GRF standalone RMSE: 0.364 vs 0.390 (8% gain)
                                   #   Hybrid RMSE: ~0.004 reduction (significant)

# Horseshoe parameters
HYBRID_HS_N_MCMC  <- 3000  # Post-burnin MCMC samples (3x base for stable 2%/98% quantiles)
HYBRID_HS_BURNIN  <- 1000  # Burn-in samples to discard

# Inverse-RMSE weights for the 2-component ensemble.
# Higher-performing methods (lower RMSE) get proportionally more weight.
#
# Calibrated from OOS evaluation on 500 mirror DGP v2 datasets
# (benchmark_hybrid_xl_swap.R, 2026-03-01).
# 2-component shares: GRF=52.4%, HS=47.6%.
HYBRID_W_HS_RMSE  <- 0.419   # Horseshoe OOS RMSE on mirror DGP v2 (500 datasets)
HYBRID_W_GRF_RMSE <- 0.381   # GRF OOS RMSE on mirror DGP v2 (500 datasets)

# Conformal calibration multiplier for CI half-widths.
# Calibrated for the 2-component ensemble (HS+GRF) on 500 mirror DGP v2
# datasets (benchmark_2comp_vs_3comp.R, 2026-03-01).
# 2,000,000 conformity scores; 95th percentile = 1.0450.
#
# Bumped from 1.0450 to 1.10 as insurance against CATE-type distribution
# mismatch (methods_reviewer_5analyses.R, analysis 5e). Coverage is
# heterogeneous across CATE types: constant=98.8%, interaction=87.1%,
# smooth_nonlinear=76.4%. If the target data has fewer constant CATEs
# than the mirror DGP (70%), the 1.0450 multiplier under-covers.
# At 1.10, the ensemble achieves ~96% coverage at 70% constant and
# ~93-94% at 55% constant, with only 5.3% wider CIs.
# Source: methods_reviewer_5analyses.R, analysis 5e (private research code)
CONFORMAL_MULTIPLIER <- 1.10

# z-multipliers for aggregate CI construction (sCATE/subCATE).
# The HS posterior SE is used as the uncertainty measure (Approach 2).
# Production CIs use: est +/- z_mult * se.
#
# Tuned on 100 mirror DGP v3 datasets (tune_scate_se_approach2.R, 2026-03-18):
#   SCATE_Z_MULT = 2.0:  95.3% sCATE coverage, width ~1.22
#   SUBCATE_Z_MULT = 2.5: 95.4% subCATE coverage (averaged across x12 strata)
#
# The subCATE needs a wider multiplier than sCATE because the posterior for
# subgroup means is slightly underdispersed relative to the full-sample mean
# (fewer observations per stratum).
SCATE_Z_MULT   <- 2.0
SUBCATE_Z_MULT <- 2.5

# PATE z-multiplier for CI construction: est +/- z * se.
# Initial value 2.0 (same as sCATE). Tune via tune_pate_z_mult.R on mirror
# DGP v3 datasets and update this value with the calibrated result.
PATE_Z_MULT    <- 2.0


# ---------------------------------------------------------------------------
# Legacy parameters (kept for backward compatibility in OOS experiments;
# not used by the production 2-component pipeline)
# ---------------------------------------------------------------------------

# BCF tuning (used when third_method="bcf" in fit_hybrid_oos.R)
HYBRID_BCF_NUM_GFR    <- 10    # GFR (grow-from-root) warmstart iterations
HYBRID_BCF_NUM_BURNIN <- 0     # MCMC burn-in (0 because GFR handles warmup)
HYBRID_BCF_NUM_MCMC   <- 100   # MCMC posterior draws

# XL-mns40 tuning (used when third_method="xlearner" in fit_hybrid_oos.R)
HYBRID_XL_N_TREES  <- 2000
HYBRID_XL_MIN_NODE <- 40
HYBRID_XL_N_FOLDS  <- 5

# Legacy inverse-RMSE weights for 3rd components
HYBRID_W_BCF_RMSE <- 0.577   # BCF OOS RMSE on mirror DGP v2
HYBRID_W_XL_RMSE  <- 0.539   # XL-mns40 OOS RMSE on mirror DGP v2


# ---------------------------------------------------------------------------
# fit_grf_component(X, Y, Z)
#
# Fit multi-arm causal forest and return iCATE predictions with CIs.
# Uses in-sample (OOB) predictions for production (train == predict data).
#
# Arguments:
#   X  -- N x 54 covariate matrix
#   Y  -- length-N outcome vector
#   Z  -- length-N treatment vector (character: a,b,c,d,e)
#
# Returns: list(tau_hat, tau_L, tau_U) -- each N x 4 matrices
# ---------------------------------------------------------------------------
fit_grf_component <- function(X, Y, Z) {
  result <- tryCatch({
    W <- factor(Z, levels = c("a", "b", "c", "d", "e"))

    cf <- multi_arm_causal_forest(
      X = X, Y = Y, W = W,
      num.trees = HYBRID_GRF_NUM_TREES,
      min.node.size = HYBRID_GRF_MIN_NODE_SIZE,
      num.threads = 1,
      seed = 42
    )

    # OOB predictions (no newdata argument = in-sample OOB)
    preds <- predict(cf, estimate.variance = TRUE)
    tau_hat <- preds$predictions[, , 1, drop = TRUE]  # N x 4
    tau_se  <- sqrt(preds$variance.estimates)

    # Shape assertion: grf should return N x 4 matrices for 5-arm forest
    stopifnot(is.matrix(tau_se), ncol(tau_se) == 4,
              nrow(tau_se) == nrow(X))

    tau_L <- tau_hat - 1.96 * tau_se
    tau_U <- tau_hat + 1.96 * tau_se

    colnames(tau_hat) <- colnames(tau_L) <- colnames(tau_U) <- c("b", "c", "d", "e")

    # Return variance estimates for use by compute_aggregate_se_combined().
    # Backward compatible: existing callers only access $tau_hat/$tau_L/$tau_U.
    variance_estimates <- preds$variance.estimates
    colnames(variance_estimates) <- c("b", "c", "d", "e")

    list(tau_hat = tau_hat, tau_L = tau_L, tau_U = tau_U,
         variance_estimates = variance_estimates)
  },
  error = function(e) {
    cat(sprintf("    [GRF] Failed: %s\n", e$message))
    n <- nrow(X)
    na_mat <- matrix(NA, n, 4)
    colnames(na_mat) <- c("b", "c", "d", "e")
    list(tau_hat = na_mat, tau_L = na_mat, tau_U = na_mat,
         variance_estimates = na_mat)
  })

  result
}


# ---------------------------------------------------------------------------
# fit_bcf_binary(Y, D, X, X_full, num_gfr, num_burnin, num_mcmc,
#                num_trees_tau)
#
# Fit stochtree::bcf for one binary contrast. Returns point estimates and
# posterior credible intervals. In-sample predictions (no X_test).
#
# Arguments:
#   Y             -- length-n outcome vector (subset to z in {a, w})
#   D             -- length-n binary treatment (0 = arm a, 1 = arm w)
#   X             -- n x 54 covariate matrix (subset)
#   X_full        -- N x 54 covariate matrix (all obs, for prediction)
#   num_gfr       -- GFR warmstart iterations
#   num_burnin    -- MCMC burn-in
#   num_mcmc      -- MCMC posterior draws
#   num_trees_tau -- (optional) number of trees in the treatment effect forest;
#                     NULL uses stochtree default (50)
#
# Returns: list(tau_hat, tau_L, tau_U) -- each length-N vectors
# ---------------------------------------------------------------------------
fit_bcf_binary <- function(Y, D, X, X_full = NULL,
                            num_gfr = HYBRID_BCF_NUM_GFR,
                            num_burnin = HYBRID_BCF_NUM_BURNIN,
                            num_mcmc = HYBRID_BCF_NUM_MCMC,
                            num_trees_tau = NULL) {
  # If X_full provided, predict OOS on full sample; otherwise in-sample
  is_oos <- !is.null(X_full)

  # Build treatment_effect_forest_params only when num_trees_tau is specified
  tau_forest_params <- if (!is.null(num_trees_tau)) {
    list(num_trees = num_trees_tau)
  } else {
    list()  # empty list = stochtree defaults
  }

  result <- tryCatch({
    X_mat <- as.matrix(X)
    set.seed(42)  # Seed for stochtree reproducibility

    if (is_oos) {
      X_pred_mat <- as.matrix(X_full)
      n_pred <- nrow(X_pred_mat)
      Z_test <- rep(1, n_pred)  # predict under treatment = 1

      bcf_fit <- stochtree::bcf(
        X_train = X_mat, Z_train = D, y_train = Y,
        X_test = X_pred_mat, Z_test = Z_test,
        num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc,
        treatment_effect_forest_params = tau_forest_params
      )
      tau_draws <- bcf_fit$tau_hat_test
    } else {
      bcf_fit <- stochtree::bcf(
        X_train = X_mat, Z_train = D, y_train = Y,
        num_gfr = num_gfr, num_burnin = num_burnin, num_mcmc = num_mcmc,
        treatment_effect_forest_params = tau_forest_params
      )
      tau_draws <- bcf_fit$tau_hat_train
    }

    tau_hat <- rowMeans(tau_draws)
    tau_L   <- apply(tau_draws, 1, quantile, probs = 0.025)
    tau_U   <- apply(tau_draws, 1, quantile, probs = 0.975)

    list(tau_hat = tau_hat, tau_L = tau_L, tau_U = tau_U)
  },
  error = function(e) {
    n_pred <- if (is_oos) nrow(X_full) else length(Y)
    cat(sprintf("    [BCF] stochtree failed: %s\n", e$message))
    list(tau_hat = rep(NA, n_pred), tau_L = rep(NA, n_pred),
         tau_U = rep(NA, n_pred))
  })

  result
}


# ---------------------------------------------------------------------------
# fit_bcf_component(X, Y, Z)
#
# Fit BCF for all 4 contrasts (b,c,d,e vs a) via binary subsetting.
# Each contrast is fit independently on the z in {a, w} subset, then
# predicted on the full N-obs sample.
#
# Arguments:
#   X  -- N x 54 covariate matrix (all observations)
#   Y  -- length-N outcome vector
#   Z  -- length-N treatment vector (character: a,b,c,d,e)
#
# Returns: list(tau_hat, tau_L, tau_U) -- each N x 4 matrices
# ---------------------------------------------------------------------------
fit_bcf_component <- function(X, Y, Z) {
  arms <- c("b", "c", "d", "e")
  n <- nrow(X)

  tau_hat <- tau_L <- tau_U <- matrix(NA, n, 4)
  colnames(tau_hat) <- colnames(tau_L) <- colnames(tau_U) <- arms

  for (j in 1:4) {
    w <- arms[j]
    mask <- Z %in% c("a", w)
    D_sub <- as.integer(Z[mask] == w)

    bcf_j <- fit_bcf_binary(
      Y    = Y[mask],
      D    = D_sub,
      X    = X[mask, , drop = FALSE],
      X_full = X  # predict on all N observations
    )

    tau_hat[, j] <- bcf_j$tau_hat
    tau_L[, j]   <- bcf_j$tau_L
    tau_U[, j]   <- bcf_j$tau_U
  }

  list(tau_hat = tau_hat, tau_L = tau_L, tau_U = tau_U)
}


# ---------------------------------------------------------------------------
# fit_hs_component(df)
#
# Fit Horseshoe on the dataset (in-sample: train on df, predict on df).
# Wrapper around fit_horseshoe() from fit_horseshoe.R.
#
# Arguments:
#   df  -- data frame with y, z, x1-x40
#
# Returns: list(tau_hat, tau_L, tau_U) -- each N x 4 matrices, or NAs
# ---------------------------------------------------------------------------
fit_hs_component <- function(df) {
  result <- tryCatch({
    fit_horseshoe(
      train_df     = df,
      test_df      = df,
      n_mcmc       = HYBRID_HS_N_MCMC,
      burnin       = HYBRID_HS_BURNIN,
      return_draws = TRUE,   # Request full posterior draws for P(best)
      verbose      = FALSE
    )
  },
  error = function(e) {
    cat(sprintf("    [Horseshoe] Failed: %s\n", e$message))
    n <- nrow(df)
    list(
      tau_hat       = matrix(NA, n, 4),
      tau_L         = matrix(NA, n, 4),
      tau_U         = matrix(NA, n, 4),
      tau_draws_all = NULL,  # No draws available on failure
      beta_draws    = NULL   # No coefficient draws on failure (used by PATE)
    )
  })

  # Ensure column names are set
  arms <- c("b", "c", "d", "e")
  colnames(result$tau_hat) <- colnames(result$tau_L) <- colnames(result$tau_U) <- arms

  result
}


# ---------------------------------------------------------------------------
# ensemble_hybrid(hs, grf, ...)
#
# Combine Horseshoe and GRF into the 2-component Hybrid ensemble.
#
# Point estimates: inverse-RMSE weighted average of HS and GRF.
# CIs: inverse-RMSE weighted blend of HS and GRF CIs, then conformally
#       calibrated (CONFORMAL_MULTIPLIER targets 95% coverage).
#
# Handles missing components gracefully (NAs from failed fits).
#
# An optional third component can still be passed for backward compatibility
# with OOS experiments; pass third=NULL (default) for production 2-component.
#
# Arguments:
#   hs            -- list(tau_hat, tau_L, tau_U) from Horseshoe (or NAs)
#   grf           -- list(tau_hat, tau_L, tau_U) from GRF
#   third         -- list(tau_hat, tau_L, tau_U) from optional third component
#                    (default: NULL, meaning 2-component mode)
#   w_hs_rmse     -- Horseshoe RMSE for inverse weighting (default: HYBRID_W_HS_RMSE)
#   w_grf_rmse    -- GRF RMSE for inverse weighting (default: HYBRID_W_GRF_RMSE)
#   w_third_rmse  -- Third component RMSE for inverse weighting (default: HYBRID_W_XL_RMSE)
#   conformal_mult -- Conformal calibration multiplier (default: CONFORMAL_MULTIPLIER)
#
# Returns: list(tau_hat, tau_L, tau_U) -- each N x 4 matrices
# ---------------------------------------------------------------------------
ensemble_hybrid <- function(hs, grf, third = NULL,
                             w_hs_rmse = HYBRID_W_HS_RMSE,
                             w_grf_rmse = HYBRID_W_GRF_RMSE,
                             w_third_rmse = NULL,
                             conformal_mult = CONFORMAL_MULTIPLIER) {
  arms <- c("b", "c", "d", "e")
  n <- nrow(grf$tau_hat)  # GRF should always succeed

  # Determine which components are available (non-NA)
  hs_ok    <- !all(is.na(hs$tau_hat))
  third_ok <- !is.null(third) && !all(is.na(third$tau_hat))
  grf_ok   <- !all(is.na(grf$tau_hat))

  if (!grf_ok) {
    # GRF should never fail, but handle gracefully
    cat("    [Hybrid] WARNING: GRF failed. Returning NAs.\n")
    return(list(
      tau_hat = matrix(NA, n, 4),
      tau_L   = matrix(NA, n, 4),
      tau_U   = matrix(NA, n, 4)
    ))
  }

  # Build point estimates from available components
  # Default w_third_rmse to XL RMSE if NULL and a third component is present
  if (is.null(w_third_rmse) && third_ok) w_third_rmse <- HYBRID_W_XL_RMSE

  if (hs_ok && third_ok) {
    # Legacy 3-method ensemble (for OOS experiments requesting a third component).
    # WARNING: The default CONFORMAL_MULTIPLIER (1.10) was calibrated for the
    # 2-component ensemble. Callers using 3-component mode should pass their own
    # conformal_mult (e.g., 0.8111 for HS+GRF+XL). Using the wrong multiplier
    # produces miscalibrated coverage.
    if (conformal_mult == CONFORMAL_MULTIPLIER) {
      warning("ensemble_hybrid: 3-component mode with default conformal_mult (",
              CONFORMAL_MULTIPLIER, ") which was calibrated for 2-component. ",
              "Pass an appropriate conformal_mult for 3-component CIs.",
              call. = FALSE)
    }

    w_hs    <- 1 / w_hs_rmse
    w_grf   <- 1 / w_grf_rmse
    w_third <- 1 / w_third_rmse
    w_sum   <- w_hs + w_grf + w_third
    tau_hat <- (w_hs * hs$tau_hat + w_grf * grf$tau_hat + w_third * third$tau_hat) / w_sum

    # 3-component CIs: start from HS posterior intervals (not blended),
    # then widen by inter-method disagreement. This differs from the
    # 2-component path (which uses weighted blending) because the 3-comp
    # blend was found to undercover in early experiments.
    tau_L <- hs$tau_L
    tau_U <- hs$tau_U

    # Adaptive CI widening based on 3-method disagreement
    spread <- pmax(hs$tau_hat, grf$tau_hat, third$tau_hat) -
              pmin(hs$tau_hat, grf$tau_hat, third$tau_hat)
    baseline_spread <- max(median(spread), 1e-10)
    widen_factor <- pmin(1.5, pmax(1.0, spread / baseline_spread))
    ci_center <- (tau_L + tau_U) / 2
    ci_half   <- (tau_U - tau_L) / 2
    tau_L <- ci_center - ci_half * widen_factor
    tau_U <- ci_center + ci_half * widen_factor

    cat(sprintf("    [Hybrid] 3-method ensemble (HS=%.1f%%, GRF=%.1f%%, 3rd=%.1f%%)\n",
                100 * w_hs / w_sum, 100 * w_grf / w_sum, 100 * w_third / w_sum))
  } else if (hs_ok) {
    # Production 2-component ensemble: HS + GRF with inverse-RMSE weights
    w_hs  <- 1 / w_hs_rmse
    w_grf <- 1 / w_grf_rmse
    w_sum <- w_hs + w_grf
    tau_hat <- (w_hs * hs$tau_hat + w_grf * grf$tau_hat) / w_sum

    # CIs: weighted blend of HS and GRF CIs (same weights as point estimates)
    tau_L <- (w_hs * hs$tau_L + w_grf * grf$tau_L) / w_sum
    tau_U <- (w_hs * hs$tau_U + w_grf * grf$tau_U) / w_sum

    cat(sprintf("    [Hybrid] 2-method ensemble (HS=%.1f%%, GRF=%.1f%%)\n",
                100 * w_hs / w_sum, 100 * w_grf / w_sum))
  } else {
    # Horseshoe failed: GRF alone with GRF CIs
    tau_hat <- grf$tau_hat
    tau_L <- grf$tau_L
    tau_U <- grf$tau_U
    cat("    [Hybrid] FALLBACK: GRF only\n")
  }

  colnames(tau_hat) <- colnames(tau_L) <- colnames(tau_U) <- arms

  # -----------------------------------------------------------------------
  # Conformal calibration multiplier
  #
  # Final CI adjustment based on empirical coverage from mirror DGP.
  # conformal_mult = 1.0 means no adjustment.
  # Values > 1.0 widen CIs; values < 1.0 narrow them.
  # Production value (1.10) targets 95% coverage for the 2-component
  # ensemble, conservatively bumped from empirical 1.045 (see header).
  # -----------------------------------------------------------------------
  if (conformal_mult != 1.0) {
    ci_center <- (tau_L + tau_U) / 2
    ci_half   <- (tau_U - tau_L) / 2
    tau_L <- ci_center - ci_half * conformal_mult
    tau_U <- ci_center + ci_half * conformal_mult
    cat(sprintf("    [Hybrid] Conformal multiplier applied: %.4f\n",
                conformal_mult))
  }

  list(tau_hat = tau_hat, tau_L = tau_L, tau_U = tau_U)
}


# ---------------------------------------------------------------------------
# fit_hybrid_dataset(df, dataset_id, output_dir)
#
# Production entry point. Fits Horseshoe + GRF (2-component ensemble),
# combines into Hybrid ensemble, computes sCATE/subCATE/PATE aggregates,
# and saves standard output files via save_standard_output().
#
# Arguments:
#   df         -- data frame with ID, y, z, x1-x40 (N=1000)
#   dataset_id -- integer dataset ID
#   output_dir -- path for output CSVs
#
# Returns: data frame with one timing row (same format as other methods)
# ---------------------------------------------------------------------------
fit_hybrid_dataset <- function(df, dataset_id, output_dir, x_bar_pop = NULL) {
  t_start <- Sys.time()
  cat(sprintf("  [Hybrid] Dataset %d: fitting 2 components...\n", dataset_id))

  n <- nrow(df)
  X <- prepare_covariates(df)
  Y <- df$y
  Z <- df$z

  # ----- Component 1: Horseshoe -----
  cat("    [Hybrid] Fitting Horseshoe...\n")
  t1 <- Sys.time()
  hs_preds <- fit_hs_component(df)
  cat(sprintf("    [Hybrid] Horseshoe: %.1fs\n",
              as.numeric(difftime(Sys.time(), t1, units = "secs"))))

  # ----- Component 2: GRF -----
  cat("    [Hybrid] Fitting GRF...\n")
  t2 <- Sys.time()
  grf_preds <- fit_grf_component(X, Y, Z)
  cat(sprintf("    [Hybrid] GRF: %.1fs\n",
              as.numeric(difftime(Sys.time(), t2, units = "secs"))))

  # ----- Ensemble combination (2-component, no third argument) -----
  hybrid <- ensemble_hybrid(hs_preds, grf_preds)

  # ----- P(best) treatment selection from Horseshoe posterior draws -----
  # The Horseshoe is the only component with a proper joint posterior across
  # all 4 contrasts (shared MCMC chain), so it is the correct source for
  # computing the probability that each arm is best for each individual.
  p_best_result <- NULL
  if (!is.null(hs_preds$tau_draws_all)) {
    p_best_result <- compute_p_best(hs_preds$tau_draws_all)
    cat(sprintf("    [Hybrid] P(best) computed: %d individuals, %d MCMC draws\n",
                nrow(p_best_result$P_best), dim(hs_preds$tau_draws_all)[2]))
  } else {
    cat("    [Hybrid] P(best) skipped: Horseshoe draws not available\n")
  }

  # ----- sCATE: sample mean of iCATEs with SE -----
  # SE from Horseshoe posterior draws (Approach 2). For each posterior draw,
  # compute the sCATE, then take SD across draws. This directly captures
  # posterior uncertainty about the aggregate.
  #
  # Tuned on 100 mirror DGP v3 datasets (tune_scate_se_approach2.R):
  #   z=2.0 gives 95.3% sCATE coverage (width ~1.22)
  #   z=2.5 gives 95.4% subCATE coverage
  #
  # Falls back to Approach 1 (from iCATE CIs) if HS draws unavailable.
  scate_est <- colMeans(hybrid$tau_hat)
  if (!is.null(hs_preds$tau_draws_all)) {
    scate_draws_result <- compute_aggregate_se_from_draws(
      hs_preds$tau_draws_all, aggregation = "scate")
    scate_se <- scate_draws_result$se
  } else {
    scate_se <- compute_aggregate_se_from_ci(hybrid$tau_L, hybrid$tau_U,
                                              aggregation = "scate")
  }

  # ----- PATE: population average treatment effect -----
  # When x_bar_pop is provided and beta_draws are available from the HS fit,
  # compute proper PATE using the linearity of the HS model:
  #   PATE_w = gamma_w + x_bar_pop' * delta_w
  #
  # The hybrid PATE combines:
  #   - HS component: proper population-averaged PATE from posterior draws
  #   - GRF component: approximate PATE with sCATE (since X is drawn from
  #     the same distribution, GRF sCATE ≈ GRF PATE)
  # Both weighted by inverse-RMSE, same as the iCATE ensemble.
  #
  # Falls back to PATE = sCATE when x_bar_pop is NULL (backward compatible).
  pate_est_out <- NULL
  pate_se_out  <- NULL

  if (!is.null(x_bar_pop) && !is.null(hs_preds$beta_draws)) {
    pate_hs <- compute_pate_from_hs_posterior(hs_preds$beta_draws, x_bar_pop)

    # GRF PATE approximation: use GRF sCATE (sample mean of GRF iCATEs).
    # Since X is drawn from the same distribution across populations,
    # the sample mean is an unbiased estimator of the population mean.
    scate_grf <- colMeans(grf_preds$tau_hat)

    # Hybrid PATE via shared helper (inverse-RMSE weighted HS + GRF)
    hybrid_pate <- compute_hybrid_pate(
      pate_hs, scate_grf, scate_se,
      w_hs_rmse = HYBRID_W_HS_RMSE, w_grf_rmse = HYBRID_W_GRF_RMSE)
    pate_est_out <- hybrid_pate$pate_est
    pate_se_out  <- hybrid_pate$pate_se

    cat(sprintf("    [Hybrid] PATE computed: HS=%.4f/%.4f/%.4f/%.4f, ",
                pate_hs$pate_est[1], pate_hs$pate_est[2],
                pate_hs$pate_est[3], pate_hs$pate_est[4]))
    cat(sprintf("Hybrid=%.4f/%.4f/%.4f/%.4f\n",
                pate_est_out[1], pate_est_out[2],
                pate_est_out[3], pate_est_out[4]))
  } else {
    if (is.null(x_bar_pop)) {
      cat("    [Hybrid] PATE: falling back to sCATE (x_bar_pop not provided)\n")
    } else {
      cat("    [Hybrid] PATE: falling back to sCATE (beta_draws not available)\n")
    }
  }

  t_elapsed <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
  cat(sprintf("  [Hybrid] Dataset %d: total %.1fs\n", dataset_id, t_elapsed))

  # ----- Save standard output -----
  timing_row <- save_standard_output(
    df, hybrid$tau_hat, hybrid$tau_L, hybrid$tau_U,
    scate_est, scate_se,
    dataset_id, output_dir, t_elapsed,
    p_best_result = p_best_result,
    hs_draws = hs_preds$tau_draws_all,
    pate_est = pate_est_out,
    pate_se  = pate_se_out
  )

  return(timing_row)
}
