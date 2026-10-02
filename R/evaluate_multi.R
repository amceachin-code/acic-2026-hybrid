# ===========================================================================
# R/evaluate_multi.R — Shared evaluation metrics for multi-arm CATEs
#
# Provides:
#   evaluate_multi()        — iCATE-level metrics (RMSE, bias, coverage, etc.)
#   na_metrics()            — NA placeholder matching evaluate_multi shape
#   evaluate_aggregates()   — sCATE/subCATE metrics (both n_k and n definitions)
#   na_aggregate_metrics()  — NA placeholder matching evaluate_aggregates shape
#
# Canonical version — do NOT copy-paste these into experiment scripts.
#
# Usage:
#   source("R/evaluate_multi.R")
#   metrics <- evaluate_multi(tau_hat, tau_L, tau_U, tau_true)
#   na_row  <- na_metrics()
#   agg     <- evaluate_aggregates(tau_hat, tau_true, x12_vals)
#   na_agg  <- na_aggregate_metrics()
# ===========================================================================


# ---------------------------------------------------------------------------
# na_metrics()
#
# Returns a named list of NA values matching the output shape of
# evaluate_multi(). Used when a method fails and returns all-NA predictions,
# to avoid passing NAs into evaluate_multi() which would produce NaN results
# via arithmetic on NA values.
# ---------------------------------------------------------------------------
na_metrics <- function() {
  arms <- c("b", "c", "d", "e")
  nms <- c("rmse", "bias", "coverage", "width", "assign_acc", "regret",
           paste0("rmse_", arms), paste0("bias_", arms),
           paste0("coverage_", arms), paste0("width_", arms))
  result <- as.list(rep(NA_real_, length(nms)))
  names(result) <- nms
  result
}


#' Evaluate iCATE estimates against ground truth (multi-arm)
#'
#' All inputs are N x 4 matrices (columns = b-a, c-a, d-a, e-a contrasts).
#' Returns averaged metrics AND per-contrast metrics.
#' Also computes treatment assignment accuracy and regret.
#'
#' @param tau_hat N×4 matrix of iCATE point estimates
#' @param tau_L N×4 matrix of lower 95% CI bounds
#' @param tau_U N×4 matrix of upper 95% CI bounds
#' @param tau_true N×4 matrix of ground truth CATEs
#' @return Named list with rmse, bias, coverage, width, assign_acc, regret,
#'         and per-contrast versions (rmse_b, rmse_c, ..., width_e)
evaluate_multi <- function(tau_hat, tau_L, tau_U, tau_true) {
  stopifnot(is.matrix(tau_hat), ncol(tau_hat) == 4, ncol(tau_true) == 4)
  contrast_names <- c("b", "c", "d", "e")
  n <- nrow(tau_hat)

  # Per-contrast metrics
  per_contrast <- lapply(1:4, function(j) {
    rmse_j     <- sqrt(mean((tau_hat[, j] - tau_true[, j])^2))
    bias_j     <- mean(tau_hat[, j] - tau_true[, j])
    coverage_j <- mean(tau_true[, j] >= tau_L[, j] & tau_true[, j] <= tau_U[, j])
    width_j    <- mean(tau_U[, j] - tau_L[, j])
    list(rmse = rmse_j, bias = bias_j, coverage = coverage_j, width = width_j)
  })

  # Treatment assignment accuracy (control 'a' is a candidate)
  # If max(tau) > 0, assign to the best treatment arm; otherwise assign to 'a'
  best_true <- apply(tau_true, 1, function(row) {
    if (max(row) > 0) which.max(row) else 0L
  })
  best_hat <- apply(tau_hat, 1, function(row) {
    if (max(row) > 0) which.max(row) else 0L
  })
  assign_acc <- mean(best_true == best_hat)

  # Regret: expected outcome loss from following estimated vs true optimal rule
  regret <- mean(sapply(1:n, function(i) {
    val_true <- if (best_true[i] == 0) 0 else tau_true[i, best_true[i]]
    val_hat  <- if (best_hat[i] == 0) 0 else tau_true[i, best_hat[i]]
    val_true - val_hat
  }))

  # Build result: averaged metrics + per-contrast columns
  result <- list(
    rmse       = mean(sapply(per_contrast, `[[`, "rmse")),
    bias       = mean(sapply(per_contrast, `[[`, "bias")),
    coverage   = mean(sapply(per_contrast, `[[`, "coverage")),
    width      = mean(sapply(per_contrast, `[[`, "width")),
    assign_acc = assign_acc,
    regret     = regret
  )

  for (j in 1:4) {
    w <- contrast_names[j]
    result[[paste0("rmse_", w)]]     <- per_contrast[[j]]$rmse
    result[[paste0("bias_", w)]]     <- per_contrast[[j]]$bias
    result[[paste0("coverage_", w)]] <- per_contrast[[j]]$coverage
    result[[paste0("width_", w)]]    <- per_contrast[[j]]$width
  }

  result
}


# ---------------------------------------------------------------------------
# na_aggregate_metrics()
#
# Returns a named list of NA values matching the output shape of
# evaluate_aggregates(). Used when a method fails and we need a
# placeholder row that won't break rbind().
# ---------------------------------------------------------------------------
na_aggregate_metrics <- function() {
  arms <- c("b", "c", "d", "e")
  strata <- c("0", "1")

  nms <- c(
    # sCATE metrics (averaged across contrasts)
    "scate_rmse", "scate_bias", "scate_coverage", "scate_width",
    "scate_best_arm_acc",
    # sCATE per-contrast
    paste0("scate_rmse_", arms), paste0("scate_bias_", arms),
    paste0("scate_coverage_", arms), paste0("scate_width_", arms),
    # subCATE (n_k denominator) per stratum per contrast
    paste0("subcate_nk_rmse_x12_", rep(strata, each = 4), "_", arms),
    paste0("subcate_nk_bias_x12_", rep(strata, each = 4), "_", arms),
    paste0("subcate_nk_coverage_x12_", rep(strata, each = 4), "_", arms),
    paste0("subcate_nk_width_x12_", rep(strata, each = 4), "_", arms),
    # subCATE (n_k denominator) best-arm accuracy per stratum
    "subcate_nk_best_arm_acc_x12_0", "subcate_nk_best_arm_acc_x12_1",
    # subCATE (n denominator) per stratum per contrast
    paste0("subcate_n_rmse_x12_", rep(strata, each = 4), "_", arms),
    paste0("subcate_n_bias_x12_", rep(strata, each = 4), "_", arms),
    paste0("subcate_n_coverage_x12_", rep(strata, each = 4), "_", arms),
    paste0("subcate_n_width_x12_", rep(strata, each = 4), "_", arms),
    # subCATE (n denominator) best-arm accuracy per stratum
    "subcate_n_best_arm_acc_x12_0", "subcate_n_best_arm_acc_x12_1",
    # Heterogeneity gap metrics (subCATE_1 - subCATE_0)
    "het_gap_nk_rmse", "het_gap_nk_bias",
    paste0("het_gap_nk_rmse_", arms), paste0("het_gap_nk_bias_", arms),
    "het_gap_n_rmse", "het_gap_n_bias",
    paste0("het_gap_n_rmse_", arms), paste0("het_gap_n_bias_", arms)
  )

  result <- as.list(rep(NA_real_, length(nms)))
  names(result) <- nms
  result
}


# ---------------------------------------------------------------------------
# evaluate_aggregates(tau_hat, tau_true, x12_vals, tau_L, tau_U, se_method)
#
# Computes sCATE, subCATE, and heterogeneity gap evaluation metrics.
#
# Two subCATE definitions are computed:
#   - "n_k" (subgroup mean): colMeans(tau[mask, ]) -- the standard definition
#   - "n" (contest formula): sum(tau[mask, ]) / n  -- uses full sample denominator
#
# The contest formula literally uses n^{-1} with an indicator, while the
# standard approach uses colMeans within strata. We compute both so the
# choice can be deferred to submission time.
#
# Relationship: subcate_n = (n_k / n) * subcate_nk
#
# SE computation:
#   Priority order for SE method:
#   1. "from_draws" (Approach 2): if hs_draws provided, use HS posterior SE.
#      Uses SCATE_Z_MULT / SUBCATE_Z_MULT for CI construction.
#   2. "from_ci" (Approach 1): if tau_L/tau_U provided, derive SE from iCATE CIs.
#   3. "legacy": fall back to sd/sqrt(n) with a warning.
#
# Args:
#   tau_hat   -- N x 4 matrix of iCATE point estimates
#   tau_true  -- N x 4 matrix of ground truth iCATEs
#   x12_vals  -- length-N integer/numeric vector of x12 values (0 or 1)
#   tau_L     -- N x 4 matrix of iCATE lower 95% CI bounds (optional)
#   tau_U     -- N x 4 matrix of iCATE upper 95% CI bounds (optional)
#   hs_draws  -- n x n_mcmc x 4 array of HS posterior draws (optional)
#   se_method -- character: "from_draws" (default), "from_ci", or "legacy"
#
# Returns: named list with all aggregate metrics (see na_aggregate_metrics
#          for the full field list)
# ---------------------------------------------------------------------------
evaluate_aggregates <- function(tau_hat, tau_true, x12_vals,
                                 tau_L = NULL, tau_U = NULL,
                                 hs_draws = NULL,
                                 se_method = "from_draws") {
  stopifnot(is.matrix(tau_hat), ncol(tau_hat) == 4,
            is.matrix(tau_true), ncol(tau_true) == 4,
            length(x12_vals) == nrow(tau_hat))

  arms <- c("b", "c", "d", "e")
  n <- nrow(tau_hat)

  # ---- sCATE: sample average treatment effect per contrast ----
  # Estimated sCATE = colMeans(tau_hat), True sCATE = colMeans(tau_true)
  scate_hat  <- colMeans(tau_hat)
  scate_true <- colMeans(tau_true)

  # sCATE SE: determine which method to use based on available data.
  # Priority: HS draws (Approach 2) > iCATE CIs (Approach 1) > legacy sd/sqrt(n)
  use_draws <- !is.null(hs_draws) && se_method == "from_draws"
  use_ci_se <- !use_draws && !is.null(tau_L) && !is.null(tau_U) &&
               se_method %in% c("from_ci", "from_draws")

  if (!use_draws && !use_ci_se) {
    if (is.null(tau_L) && is.null(hs_draws)) {
      warning("evaluate_aggregates: no tau_L/tau_U or hs_draws; using legacy sd/sqrt(n) SE. ",
              "Pass iCATE CI bounds or HS draws for correct aggregate CIs.", call. = FALSE)
    }
  }

  if (use_draws) {
    scate_draws_result <- compute_aggregate_se_from_draws(hs_draws, aggregation = "scate")
    scate_se <- scate_draws_result$se
    scate_z  <- SCATE_Z_MULT
  } else if (use_ci_se) {
    scate_se <- compute_aggregate_se_from_ci(tau_L, tau_U, aggregation = "scate")
    scate_z  <- 1.96
  } else {
    # Legacy fallback (backward compatible, but produces miscalibrated CIs)
    scate_se <- apply(tau_hat, 2, sd) / sqrt(n)
    scate_z  <- 1.96
  }
  scate_L  <- scate_hat - scate_z * scate_se
  scate_U  <- scate_hat + scate_z * scate_se

  # Per-contrast sCATE metrics
  # For a single dataset, sCATE RMSE per contrast = |error| (one scalar).
  # Across datasets these get averaged to produce a proper RMSE.
  scate_rmse_j     <- abs(scate_hat - scate_true)
  scate_bias_j     <- scate_hat - scate_true
  scate_coverage_j <- as.numeric(scate_true >= scate_L & scate_true <= scate_U)
  scate_width_j    <- scate_U - scate_L

  # sCATE best-arm accuracy (does the best estimated arm match the best true arm?)
  best_scate_true <- if (max(scate_true) > 0) which.max(scate_true) else 0L
  best_scate_hat  <- if (max(scate_hat) > 0)  which.max(scate_hat)  else 0L
  scate_best_arm_acc <- as.numeric(best_scate_true == best_scate_hat)

  # ---- subCATE: stratified by x12 ----
  mask_0 <- which(x12_vals == 0)
  mask_1 <- which(x12_vals == 1)
  n_0 <- length(mask_0)
  n_1 <- length(mask_1)

  # Guard: both strata must have >= 2 observations for sd() to be defined.
  # With n=1000 and typical Bernoulli probs this never fires, but prevents
  # silent NaN propagation in edge cases or future DGP variants.
  if (n_0 < 2 || n_1 < 2) {
    warning("evaluate_aggregates: x12 stratum has < 2 obs; returning NA")
    return(na_aggregate_metrics())
  }

  # Helper: compute subCATE metrics for one stratum under one definition.
  #
  # Args:
  #   mask  -- integer vector of row indices for this stratum
  #   n_k   -- stratum sample size (length of mask)
  #   denom -- denominator for the subCATE definition:
  #            n_k for standard subgroup mean, n for contest formula
  compute_subcate_metrics <- function(mask, n_k, denom) {
    # Estimated subCATE: sum of tau_hat within stratum, divided by denom
    est   <- colSums(tau_hat[mask, , drop = FALSE]) / denom
    # True subCATE: sum of tau_true within stratum, divided by denom
    truth <- colSums(tau_true[mask, , drop = FALSE]) / denom

    # SE computation: use HS draws (Approach 2) when available,
    # otherwise derive from iCATE CIs (Approach 1), otherwise legacy.
    agg_type <- if (denom == n_k) "subcate_nk" else "subcate_n"

    if (use_draws) {
      draws_result <- compute_aggregate_se_from_draws(
        hs_draws, mask = mask, n_total = n, aggregation = agg_type)
      se <- draws_result$se
      subcate_z <- SUBCATE_Z_MULT
    } else if (use_ci_se) {
      se <- compute_aggregate_se_from_ci(tau_L, tau_U, mask = mask,
                                          n_total = n, aggregation = agg_type)
      subcate_z <- 1.96
    } else {
      # Legacy fallback
      se_nk <- apply(tau_hat[mask, , drop = FALSE], 2, sd) / sqrt(n_k)
      se    <- se_nk * (n_k / denom)
      subcate_z <- 1.96
    }

    L <- est - subcate_z * se
    U <- est + subcate_z * se

    # Per-contrast metrics
    rmse_j     <- abs(est - truth)
    bias_j     <- est - truth
    coverage_j <- as.numeric(truth >= L & truth <= U)
    width_j    <- U - L

    # Best-arm accuracy for this stratum
    best_true <- if (max(truth) > 0) which.max(truth) else 0L
    best_hat  <- if (max(est) > 0)   which.max(est)   else 0L
    best_acc  <- as.numeric(best_true == best_hat)

    list(est = est, truth = truth,
         rmse = rmse_j, bias = bias_j,
         coverage = coverage_j, width = width_j,
         best_acc = best_acc)
  }

  # n_k denominator (standard subgroup mean)
  sub_nk_0 <- compute_subcate_metrics(mask_0, n_0, n_0)
  sub_nk_1 <- compute_subcate_metrics(mask_1, n_1, n_1)

  # n denominator (contest formula: sum/n)
  sub_n_0 <- compute_subcate_metrics(mask_0, n_0, n)
  sub_n_1 <- compute_subcate_metrics(mask_1, n_1, n)

  # ---- Heterogeneity gap: subCATE(x12=1) - subCATE(x12=0) ----
  # Measures ability to detect differential treatment effects across x12 strata.
  # n_k version
  het_gap_nk_hat  <- sub_nk_1$est - sub_nk_0$est
  het_gap_nk_true <- sub_nk_1$truth - sub_nk_0$truth
  het_gap_nk_rmse_j <- abs(het_gap_nk_hat - het_gap_nk_true)
  het_gap_nk_bias_j <- het_gap_nk_hat - het_gap_nk_true

  # n version
  het_gap_n_hat  <- sub_n_1$est - sub_n_0$est
  het_gap_n_true <- sub_n_1$truth - sub_n_0$truth
  het_gap_n_rmse_j <- abs(het_gap_n_hat - het_gap_n_true)
  het_gap_n_bias_j <- het_gap_n_hat - het_gap_n_true

  # ---- Assemble result ----
  result <- list(
    # sCATE averaged across contrasts
    scate_rmse         = mean(scate_rmse_j),
    scate_bias         = mean(scate_bias_j),
    scate_coverage     = mean(scate_coverage_j),
    scate_width        = mean(scate_width_j),
    scate_best_arm_acc = scate_best_arm_acc
  )

  # sCATE per-contrast
  for (j in 1:4) {
    w <- arms[j]
    result[[paste0("scate_rmse_", w)]]     <- scate_rmse_j[j]
    result[[paste0("scate_bias_", w)]]     <- scate_bias_j[j]
    result[[paste0("scate_coverage_", w)]] <- scate_coverage_j[j]
    result[[paste0("scate_width_", w)]]    <- scate_width_j[j]
  }

  # subCATE per stratum per contrast (both definitions in a single loop)
  # Iterates over (definition_prefix, stratum_0_metrics, stratum_1_metrics)
  subcate_defs <- list(
    list(pfx = "subcate_nk_", s0 = sub_nk_0, s1 = sub_nk_1),
    list(pfx = "subcate_n_",  s0 = sub_n_0,  s1 = sub_n_1)
  )
  for (def in subcate_defs) {
    for (s_label in c("0", "1")) {
      sub <- if (s_label == "0") def$s0 else def$s1
      for (j in 1:4) {
        sfx <- paste0("_x12_", s_label, "_", arms[j])
        result[[paste0(def$pfx, "rmse", sfx)]]     <- sub$rmse[j]
        result[[paste0(def$pfx, "bias", sfx)]]     <- sub$bias[j]
        result[[paste0(def$pfx, "coverage", sfx)]] <- sub$coverage[j]
        result[[paste0(def$pfx, "width", sfx)]]    <- sub$width[j]
      }
      result[[paste0(def$pfx, "best_arm_acc_x12_", s_label)]] <- sub$best_acc
    }
  }

  # Heterogeneity gap (both definitions in a single loop)
  het_gap_defs <- list(
    list(pfx = "het_gap_nk_", rmse = het_gap_nk_rmse_j, bias = het_gap_nk_bias_j),
    list(pfx = "het_gap_n_",  rmse = het_gap_n_rmse_j,  bias = het_gap_n_bias_j)
  )
  for (hg in het_gap_defs) {
    result[[paste0(hg$pfx, "rmse")]] <- mean(hg$rmse)
    result[[paste0(hg$pfx, "bias")]] <- mean(hg$bias)
    for (j in 1:4) {
      result[[paste0(hg$pfx, "rmse_", arms[j])]] <- hg$rmse[j]
      result[[paste0(hg$pfx, "bias_", arms[j])]] <- hg$bias[j]
    }
  }

  result
}
