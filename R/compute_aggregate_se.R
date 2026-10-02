# ===========================================================================
# R/compute_aggregate_se.R -- Standard errors for aggregate estimands
#
# Provides three approaches for computing sCATE/subCATE standard errors,
# replacing the broken sd(tau_hat)/sqrt(n) formula that measured heterogeneity
# of the estimated CATE surface rather than estimation uncertainty.
#
# The old formula produced catastrophically miscalibrated CIs (1.7% coverage
# instead of 95%) because dividing by sqrt(n) shrinks the SE to nearly zero
# regardless of how noisy each individual iCATE estimate is.
#
# Three approaches:
#   1. from_ci:   Extract per-unit variance from calibrated iCATE CIs
#   2. from_draws: Posterior SE from Horseshoe MCMC draws
#   3. combined:  Weight HS posterior + GRF variance by ensemble shares
#
# Aggregation modes:
#   - "scate":      sCATE = mean of all iCATEs; SE = sqrt(mean(var_i) / n)
#   - "subcate_nk": subCATE with n_k denominator (subgroup mean)
#   - "subcate_n":  subCATE with n denominator (contest formula)
#
# Dependencies: none (standalone helper)
#
# Usage:
#   source("R/compute_aggregate_se.R")
#   se <- compute_aggregate_se_from_ci(tau_L, tau_U, aggregation = "scate")
# ===========================================================================


# ---------------------------------------------------------------------------
# compute_aggregate_se_from_ci(tau_L, tau_U, mask, n_total, aggregation)
#
# Approach 1: Extract per-unit variance from calibrated iCATE CIs.
#
# The iCATE CIs are already conformally calibrated (~95% coverage). We can
# back out the implied per-unit variance as:
#   var_i = ((tau_U_i - tau_L_i) / (2 * 1.96))^2
#
# Then propagate to the aggregate SE:
#   sCATE:      SE = sqrt(mean(var_i) / n)
#   subCATE_nk: SE = sqrt(mean(var_i[mask]) / n_k)
#   subCATE_n:  SE = sqrt(sum(var_i[mask]) / n^2)
#
# This approach requires no pipeline changes; it works with the existing
# iCATE CI bounds that are already computed and saved.
#
# Arguments:
#   tau_L       -- N x 4 matrix of iCATE lower 95% CI bounds
#   tau_U       -- N x 4 matrix of iCATE upper 95% CI bounds
#   mask        -- integer vector of row indices for subCATE (NULL for sCATE)
#   n_total     -- total sample size (needed for subcate_n denominator)
#   aggregation -- one of "scate", "subcate_nk", "subcate_n"
#
# Returns: length-4 named SE vector (one per contrast: b, c, d, e)
# ---------------------------------------------------------------------------
compute_aggregate_se_from_ci <- function(tau_L, tau_U,
                                          mask = NULL,
                                          n_total = NULL,
                                          aggregation = "scate") {
  stopifnot(is.matrix(tau_L), is.matrix(tau_U),
            ncol(tau_L) == 4, ncol(tau_U) == 4,
            nrow(tau_L) == nrow(tau_U))

  arms <- c("b", "c", "d", "e")

  # Per-unit implied variance from calibrated CIs.
  # CI half-width = 1.96 * se_i, so se_i = half_width / 1.96,
  # and var_i = se_i^2 = (half_width / 1.96)^2.
  icatevar <- ((tau_U - tau_L) / (2 * 1.96))^2  # N x 4

  se <- switch(aggregation,
    "scate" = {
      # sCATE = (1/n) * sum(tau_i), so Var(sCATE) = (1/n^2) * sum(var_i)
      #       = mean(var_i) / n
      n <- nrow(tau_L)
      sqrt(colMeans(icatevar) / n)
    },
    "subcate_nk" = {
      # subCATE_nk = (1/n_k) * sum(tau_i for i in mask)
      # Var = mean(var_i[mask]) / n_k
      stopifnot(!is.null(mask), length(mask) > 0)
      n_k <- length(mask)
      sqrt(colMeans(icatevar[mask, , drop = FALSE]) / n_k)
    },
    "subcate_n" = {
      # subCATE_n = (1/n) * sum(tau_i * 1(x12_i == x))
      # Var = sum(var_i[mask]) / n^2
      stopifnot(!is.null(mask), !is.null(n_total))
      sqrt(colSums(icatevar[mask, , drop = FALSE]) / n_total^2)
    },
    stop(sprintf("Unknown aggregation: '%s'. Expected: scate, subcate_nk, subcate_n",
                 aggregation))
  )

  names(se) <- arms
  se
}


# ---------------------------------------------------------------------------
# compute_aggregate_se_from_draws(tau_draws_all, mask, n_total, aggregation)
#
# Approach 2: Posterior SE from Horseshoe MCMC draws.
#
# For each posterior draw m, compute the aggregate (e.g., mean of iCATEs),
# then take the SD across draws. This directly captures posterior uncertainty
# about the aggregate estimand.
#
# Also returns quantile-based CIs (2.5th/97.5th percentiles of the
# aggregate draws), which may be more accurate than normal-approximation CIs
# when the posterior is skewed.
#
# Arguments:
#   tau_draws_all -- n_test x n_mcmc x 4 array of posterior iCATE draws
#   mask          -- integer vector of row indices for subCATE (NULL for sCATE)
#   n_total       -- total sample size (needed for subcate_n denominator)
#   aggregation   -- one of "scate", "subcate_nk", "subcate_n"
#
# Returns: list with:
#   se   -- length-4 named SE vector
#   L95  -- length-4 lower 2.5% quantile of aggregate draws
#   U95  -- length-4 upper 97.5% quantile of aggregate draws
#   est  -- length-4 posterior mean of aggregate draws
# ---------------------------------------------------------------------------
compute_aggregate_se_from_draws <- function(tau_draws_all,
                                             mask = NULL,
                                             n_total = NULL,
                                             aggregation = "scate") {
  stopifnot(is.array(tau_draws_all), length(dim(tau_draws_all)) == 3,
            dim(tau_draws_all)[3] == 4)

  arms <- c("b", "c", "d", "e")
  n_mcmc <- dim(tau_draws_all)[2]

  se  <- numeric(4)
  L95 <- numeric(4)
  U95 <- numeric(4)
  est <- numeric(4)

  for (j in 1:4) {
    # Extract the n_test x n_mcmc matrix for contrast j
    draws_j <- tau_draws_all[, , j]  # n_test x n_mcmc

    # Compute the aggregate for each posterior draw
    agg_draws <- switch(aggregation,
      "scate" = {
        # sCATE draw m = mean of all iCATE draws for draw m
        colMeans(draws_j)  # length n_mcmc
      },
      "subcate_nk" = {
        stopifnot(!is.null(mask), length(mask) > 0)
        colMeans(draws_j[mask, , drop = FALSE])
      },
      "subcate_n" = {
        stopifnot(!is.null(mask), !is.null(n_total))
        colSums(draws_j[mask, , drop = FALSE]) / n_total
      },
      stop(sprintf("Unknown aggregation: '%s'", aggregation))
    )

    # Posterior SE = SD across draws
    se[j]  <- sd(agg_draws)
    L95[j] <- quantile(agg_draws, probs = 0.025)
    U95[j] <- quantile(agg_draws, probs = 0.975)
    est[j] <- mean(agg_draws)
  }

  names(se) <- names(L95) <- names(U95) <- names(est) <- arms

  list(se = se, L95 = L95, U95 = U95, est = est)
}


# ---------------------------------------------------------------------------
# compute_aggregate_se_combined(tau_draws_all, grf_variance, w_hs, w_grf,
#                                mask, n_total, aggregation)
#
# Approach 3: Combined Horseshoe posterior + GRF variance, weighted by
# ensemble shares.
#
# The Hybrid ensemble combines HS and GRF with inverse-RMSE weights.
# This approach computes the variance of the aggregate under each component
# separately, then combines them using the squared normalized weights:
#   Var(aggregate) = w_hs_norm^2 * Var_HS + w_grf_norm^2 * Var_GRF
#
# HS variance: from posterior draws (same as Approach 2)
# GRF variance: from grf's variance.estimates (mean of per-unit variance / n)
#
# Arguments:
#   tau_draws_all -- n_test x n_mcmc x 4 array of HS posterior draws
#   grf_variance  -- n_test x 4 matrix of GRF variance estimates
#   w_hs          -- inverse-RMSE weight for Horseshoe (e.g., 1/0.419)
#   w_grf         -- inverse-RMSE weight for GRF (e.g., 1/0.381)
#   mask          -- integer vector of row indices for subCATE (NULL for sCATE)
#   n_total       -- total sample size (needed for subcate_n denominator)
#   aggregation   -- one of "scate", "subcate_nk", "subcate_n"
#
# Returns: length-4 named SE vector
# ---------------------------------------------------------------------------
compute_aggregate_se_combined <- function(tau_draws_all, grf_variance,
                                           w_hs, w_grf,
                                           mask = NULL,
                                           n_total = NULL,
                                           aggregation = "scate") {
  stopifnot(is.array(tau_draws_all), length(dim(tau_draws_all)) == 3,
            dim(tau_draws_all)[3] == 4,
            is.matrix(grf_variance), ncol(grf_variance) == 4)

  arms <- c("b", "c", "d", "e")

  # Normalize ensemble weights so they sum to 1
  w_sum <- w_hs + w_grf
  w_hs_norm  <- w_hs / w_sum
  w_grf_norm <- w_grf / w_sum

  se <- numeric(4)

  for (j in 1:4) {
    # HS variance: variance of the aggregate across posterior draws
    draws_j <- tau_draws_all[, , j]  # n_test x n_mcmc

    hs_agg_draws <- switch(aggregation,
      "scate" = colMeans(draws_j),
      "subcate_nk" = {
        stopifnot(!is.null(mask))
        colMeans(draws_j[mask, , drop = FALSE])
      },
      "subcate_n" = {
        stopifnot(!is.null(mask), !is.null(n_total))
        colSums(draws_j[mask, , drop = FALSE]) / n_total
      },
      stop(sprintf("Unknown aggregation: '%s'", aggregation))
    )
    var_hs <- var(hs_agg_draws)

    # GRF variance: propagate per-unit variance estimates to the aggregate
    grf_var_j <- grf_variance[, j]

    var_grf <- switch(aggregation,
      "scate" = {
        # Var(mean) = mean(var_i) / n
        mean(grf_var_j) / length(grf_var_j)
      },
      "subcate_nk" = {
        stopifnot(!is.null(mask))
        mean(grf_var_j[mask]) / length(mask)
      },
      "subcate_n" = {
        stopifnot(!is.null(mask), !is.null(n_total))
        sum(grf_var_j[mask]) / n_total^2
      }
    )

    # Combined variance under independence assumption
    se[j] <- sqrt(w_hs_norm^2 * var_hs + w_grf_norm^2 * var_grf)
  }

  names(se) <- arms
  se
}


# ---------------------------------------------------------------------------
# compute_aggregate_se(method, ...)
#
# Dispatcher: routes to the appropriate SE computation function.
#
# Arguments:
#   method -- one of "from_ci", "from_draws", "combined"
#   ...    -- arguments passed to the underlying function
#
# Returns: depends on method (see individual function docs)
# ---------------------------------------------------------------------------
compute_aggregate_se <- function(method, ...) {
  switch(method,
    "from_ci"    = compute_aggregate_se_from_ci(...),
    "from_draws" = compute_aggregate_se_from_draws(...),
    "combined"   = compute_aggregate_se_combined(...),
    stop(sprintf("Unknown SE method: '%s'. Expected: from_ci, from_draws, combined",
                 method))
  )
}
