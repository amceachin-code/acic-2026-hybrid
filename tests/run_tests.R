# ===========================================================================
# tests/run_tests.R: Self-contained checks for the Hybrid ensemble repository
#
# Usage:
#   Rscript tests/run_tests.R          # fast unit checks (~5 seconds)
#   Rscript tests/run_tests.R --full   # also runs one end-to-end fit (~20 s)
#
# Uses base R only (stopifnot), so no test framework is needed. Each check
# prints PASS or stops with an error. Exit status is non-zero on failure.
#
# What is covered:
#   1. Simulator: output shape, ACIC file layout, reproducibility by seed,
#      and train/test pairs sharing one DGP
#   2. Covariate encoding: 54 columns, drop-first one-hot, values preserved
#   3. Population means: exact formula used by dgp/generate_data.R agrees
#      with a large Monte Carlo draw
#   4. PATE from posterior draws: recovers a known linear CATE exactly
#   5. P(best): probabilities are valid and pick the obviously best arm
#   6. (--full) End-to-end: R/run_hybrid.R writes well-formed outputs
# ===========================================================================

get_repo_root <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(dirname(normalizePath(sub("--file=", "", file_arg[1])))))
  }
  normalizePath(getwd())
}
REPO_ROOT <- get_repo_root()
FULL <- "--full" %in% commandArgs(trailingOnly = TRUE)

source(file.path(REPO_ROOT, "dgp", "mirror_dgp.R"))
source(file.path(REPO_ROOT, "R", "prepare_covariates.R"))
source(file.path(REPO_ROOT, "R", "compute_pate.R"))
source(file.path(REPO_ROOT, "R", "compute_p_best.R"))

n_pass <- 0
check <- function(name, expr) {
  force(expr)
  n_pass <<- n_pass + 1
  cat(sprintf("PASS  %s\n", name))
}

# ---------------------------------------------------------------------------
# 1. Simulator
# ---------------------------------------------------------------------------
params <- draw_acic_params_v2(seed = 1)
sim_a  <- generate_acic_mirror_v2(n = 500, params = params, seed = 2)
sim_b  <- generate_acic_mirror_v2(n = 500, params = params, seed = 2)
sim_c  <- generate_acic_mirror_v2(n = 500, params = params, seed = 3)
df     <- format_acic_csv(sim_a)

check("simulator: ACIC file layout (ID, y, z, x1-x40)", {
  stopifnot(identical(names(df), c("ID", "y", "z", paste0("x", 1:40))),
            nrow(df) == 500,
            all(df$z %in% c("a", "b", "c", "d", "e")),
            all(grepl("^level [0-9]$", df$x1)),
            all(df$x7 %in% c(0, 1)),
            all(df$x26 >= 0 & df$x26 <= 1))
})
check("simulator: true CATE matrix is N x 4 and finite", {
  stopifnot(identical(dim(sim_a$tau_true), c(500L, 4L)),
            all(is.finite(sim_a$tau_true)))
})
check("simulator: same seed reproduces the same data", {
  stopifnot(identical(sim_a$Y, sim_b$Y), identical(sim_a$Z, sim_b$Z))
})
check("simulator: different seed gives new rows from the same DGP", {
  stopifnot(!identical(sim_a$Y, sim_c$Y),
            identical(sim_a$cate_types, sim_c$cate_types))
})
check("simulator: parametric defaults stay in range", {
  draws <- lapply(1:200, function(s) draw_acic_params_v2(seed = s))
  r2 <- sapply(draws, `[[`, "target_r2")
  sd <- sapply(draws, `[[`, "target_y_sd")
  stopifnot(all(r2 > 0.05 & r2 < 0.85), all(sd >= 0.5))
})

# ---------------------------------------------------------------------------
# 2. Covariate encoding
# ---------------------------------------------------------------------------
X <- prepare_covariates(df)
check("encoding: 54 numeric columns, one per row", {
  stopifnot(is.matrix(X), ncol(X) == 54, nrow(X) == nrow(df), is.numeric(X))
})
check("encoding: matches the simulator's internal 54-column matrix", {
  stopifnot(max(abs(unname(X) - unname(sim_a$X))) < 1e-12)
})

# ---------------------------------------------------------------------------
# 3. Population covariate means (formula from dgp/generate_data.R)
# ---------------------------------------------------------------------------
check("population means: exact formula agrees with Monte Carlo", {
  n_levels  <- c(3, 4, 3, 5, 4, 7)
  cat_means <- unlist(lapply(n_levels, function(J) rep(1 / J, J - 1)))
  exact <- c(cat_means, params$bin_probs, rep(0.5, 12),
             (1 - params$zero_fracs) * 0.5)
  set.seed(10)
  mc <- colMeans(generate_acic_covariates(200000, params$bin_probs,
                                          params$zero_fracs)$X)
  stopifnot(length(exact) == 54, max(abs(mc - exact)) < 0.01)
})

# ---------------------------------------------------------------------------
# 4. PATE from Horseshoe posterior draws
# ---------------------------------------------------------------------------
check("PATE: recovers gamma + x_bar' delta for a known linear CATE", {
  set.seed(4)
  p <- 54 + 4 + 54 * 4
  n_draws <- 200
  beta <- matrix(rnorm(p * n_draws, sd = 0.01), p, n_draws)
  gamma <- c(1, -0.5, 0.25, 2)
  delta <- matrix(rnorm(54 * 4), 54, 4)
  for (w in 1:4) {
    beta[54 + w, ] <- beta[54 + w, ] + gamma[w]
    rows <- 58 + (w - 1) * 54 + (1:54)
    beta[rows, ] <- beta[rows, ] + delta[, w]
  }
  x_bar <- runif(54)
  res <- compute_pate_from_hs_posterior(beta, x_bar)
  truth <- gamma + as.vector(crossprod(delta, x_bar))
  stopifnot(max(abs(res$pate_est - truth)) < 0.05,
            all(res$pate_L95 < res$pate_est), all(res$pate_U95 > res$pate_est))
})

# ---------------------------------------------------------------------------
# 5. P(best arm)
# ---------------------------------------------------------------------------
check("P(best): rows sum to 1 and the dominant arm is chosen", {
  set.seed(5)
  n <- 20; s <- 300
  draws <- array(rnorm(n * s * 4, sd = 0.1), dim = c(n, s, 4))
  draws[, , 3] <- draws[, , 3] + 2          # arm d is clearly best
  res <- compute_p_best(draws)
  stopifnot(max(abs(rowSums(res$P_best) - 1)) < 1e-12,
            all(res$best_pbest == "d"))
})

# ---------------------------------------------------------------------------
# 6. End-to-end (optional, slower)
# ---------------------------------------------------------------------------
if (FULL) {
  check("end-to-end: run_hybrid.R writes well-formed outputs", {
    tmp <- tempfile("hybrid_test_")
    dir.create(tmp)
    sim <- generate_acic_mirror_v2(n = 1000, params = params, seed = 7)
    write.csv(format_acic_csv(sim), file.path(tmp, "data_7.csv"), row.names = FALSE)
    status <- system2("Rscript",
                      c(file.path(REPO_ROOT, "R", "run_hybrid.R"),
                        file.path(tmp, "data_7.csv"), tmp),
                      stdout = FALSE, stderr = FALSE)
    stopifnot(status == 0)
    ic  <- read.csv(file.path(tmp, "dataset_7_icates.csv"))
    agg <- read.csv(file.path(tmp, "dataset_7_aggregates.csv"))
    pb  <- read.csv(file.path(tmp, "dataset_7_pbest_probs.csv"))
    stopifnot(nrow(ic) == 1000, all(is.finite(as.matrix(ic))),
              all(ic$tau_b_L95 <= ic$tau_b & ic$tau_b <= ic$tau_b_U95),
              all(c("sCATE", "subCATE", "PATE") %in% agg$estimand),
              max(abs(rowSums(pb[, paste0("P_", c("a", "b", "c", "d", "e"))]) - 1)) < 1e-8)
    unlink(tmp, recursive = TRUE)
  })
}

cat(sprintf("\nAll %d checks passed.\n", n_pass))
