# ===========================================================================
# R/compute_p_best.R -- Probability of being best treatment (P(best))
#
# Computes the posterior probability that each treatment arm is optimal for
# each individual, using the full joint posterior draws from the Horseshoe
# model. The Horseshoe is the only Hybrid component with a proper joint
# posterior across all 4 contrasts (shared MCMC chain), making it the
# correct source for this calculation.
#
# For each MCMC draw m and individual i, we compare:
#   CATE_b^m(x_i), CATE_c^m(x_i), CATE_d^m(x_i), CATE_e^m(x_i), and 0
#
# The "0" represents control arm 'a': if all 4 contrast CATEs are negative
# in draw m, then arm 'a' is best (no treatment beats control). P(best)
# for each arm is the fraction of draws where that arm wins.
#
# This provides a decision-theoretic alternative to argmax of point
# estimates: a treatment with a large-but-imprecise estimate may lose to
# one with a smaller-but-precisely-positive estimate, because the latter
# wins more often across posterior draws.
#
# Dependencies: none (standalone function)
#
# Usage:
#   source("R/compute_p_best.R")
#   result <- compute_p_best(tau_draws_array)
#   result$P_best      # N x 5 matrix (columns: a, b, c, d, e)
#   result$best_pbest   # character vector of length N
# ===========================================================================


# ---------------------------------------------------------------------------
# compute_p_best(tau_draws_array)
#
# Compute posterior probability that each treatment arm is optimal for
# each individual, using Horseshoe CATE posterior draws.
#
# Parameters:
#   tau_draws_array -- 3D array of dimension (n, n_mcmc, 4)
#                      Slice [,,k] contains N x n_mcmc draws for contrast k
#                      (k=1: b vs a, k=2: c vs a, k=3: d vs a, k=4: e vs a)
#
# Returns: list with
#   P_best     -- N x 5 matrix of P(best) probabilities
#                 Columns named "a", "b", "c", "d", "e"
#                 Each row sums to 1.0
#   best_pbest -- character vector of length N
#                 Treatment arm with highest P(best) for each individual
#                 Ties broken by first occurrence (favors control 'a')
# ---------------------------------------------------------------------------
compute_p_best <- function(tau_draws_array) {
  # Validate input dimensions
  stopifnot(is.array(tau_draws_array), length(dim(tau_draws_array)) == 3)
  n      <- dim(tau_draws_array)[1]  # number of individuals
  n_mcmc <- dim(tau_draws_array)[2]  # number of posterior draws
  n_k    <- dim(tau_draws_array)[3]  # number of contrasts (should be 4)
  stopifnot(n_k == 4)

  arms <- c("a", "b", "c", "d", "e")

  # -------------------------------------------------------------------------
  # For each draw m and individual i, find which arm is best.
  #
  # We construct an n x n_mcmc x 5 array where the 5th slice is all zeros
  # (representing arm 'a' = control). Then for each (i, m) we find the
  # argmax across the 5 arms.
  #
  # Vectorized approach: reshape to (n * n_mcmc) x 5 matrix, compute
  # row-wise argmax, then tabulate per individual.
  # -------------------------------------------------------------------------

  # Build the 5-arm matrix: n*n_mcmc rows, 5 columns
  # Column order: a (=0), b, c, d, e
  tau_all <- matrix(0, nrow = n * n_mcmc, ncol = 5)
  colnames(tau_all) <- arms

  # Column 1 (arm 'a') stays 0 -- control baseline
  for (k in 1:4) {
    # tau_draws_array[,,k] is n x n_mcmc
    # Flatten column-major (R default): draw 1 all individuals, draw 2 all individuals, etc.
    # The reshape on line 90 also uses column-major (byrow=FALSE), so the mapping is consistent.
    tau_all[, k + 1] <- as.vector(tau_draws_array[, , k])
  }

  # Find the winning arm for each (individual, draw) pair
  # max.col breaks ties at random by default; use "first" to favor control
  winner_idx <- max.col(tau_all, ties.method = "first")  # length n * n_mcmc

  # Reshape winner indices into n x n_mcmc matrix
  winner_mat <- matrix(winner_idx, nrow = n, ncol = n_mcmc, byrow = FALSE)

  # -------------------------------------------------------------------------
  # Tabulate P(best) for each individual
  # For each row, count how often each arm index (1-5) appears, divide by n_mcmc
  # -------------------------------------------------------------------------
  P_best <- matrix(0, nrow = n, ncol = 5)
  colnames(P_best) <- arms

  for (i in 1:n) {
    counts <- tabulate(winner_mat[i, ], nbins = 5)
    P_best[i, ] <- counts / n_mcmc
  }

  # Select the arm with highest P(best) for each individual
  # Ties broken by "first" (favors 'a' since it's column 1)
  best_idx <- max.col(P_best, ties.method = "first")
  best_pbest <- arms[best_idx]

  list(P_best = P_best, best_pbest = best_pbest)
}
