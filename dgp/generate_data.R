# =============================================================================
# dgp/generate_data.R: Generate simulated train/test datasets with ground truth
#
# Purpose:
#   Builds a collection of simulated datasets for trying out, testing, and
#   benchmarking the Hybrid ensemble. Each dataset "pair" shares one set of
#   DGP parameters (CATE functions, arm probabilities, outcome scale, R²) but
#   has independent observations for train and test, so the pair supports
#   both in-sample and out-of-sample evaluation.
#
#   Everything here is simulated. No contest data is read.
#
# Usage:
#   Rscript dgp/generate_data.R [out_dir] [n_datasets] [preset]
#
# Arguments:
#   out_dir     Output directory (default: example/data)
#   n_datasets  Number of train/test pairs (default: 50)
#   preset      CATE type mix (default: "mostly_constant")
#                 mostly_constant: constant=70%, sparse_linear=10%,
#                   smooth_nonlinear=2%, step_function=2%, interaction=16%
#                   (the mix used to tune the ensemble)
#                 balanced: constant=20%, sparse_linear=25%,
#                   smooth_nonlinear=25%, step_function=15%, interaction=15%
#
# Outputs (inside out_dir):
#   train/data_<i>.csv          Training data (ID, y, z, x1-x40), n = 1000
#   test/data_<i>.csv           Test data from the same DGP, n = 1000
#   truth/train_truth_<i>.csv   True iCATEs for the training rows (ID, tau_b..tau_e)
#   truth/test_truth_<i>.csv    True iCATEs for the test rows
#   pop/x_pop_colmeans_<i>.csv  Population means of the 54 encoded covariates
#                               (exact, from the DGP), used for PATE estimation
#   metadata.csv                One row per dataset: DGP parameters, CATE
#                               types, and the true PATE for each contrast
#
# Design notes:
#   - Population covariate means are computed analytically from the DGP
#     parameters, so the PATE target is exact rather than estimated:
#       categorical level k of a J-level variable: 1/J (uniform levels)
#       binary x7-x25: the dataset's Bernoulli probability
#       continuous x26-x37: 0.5 (Uniform(0, 1))
#       zero-inflated x38-x40: (1 - zero_frac) * 0.5
#   - The true PATE needs E[tau(X)] over the population. The simulator scales
#     the raw CATE surface by a factor computed on each sample, so we use the
#     TRAINING sample's tau_scale and average the raw surface over a large
#     Monte Carlo draw (N_POP_MC rows) of population covariates.
#
# Runtime: about 1 to 2 seconds per pair.
# =============================================================================

# ---------------------------------------------------------------------------
# Repo root detection: works when called via Rscript from any directory
# ---------------------------------------------------------------------------
get_repo_root <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("--file=", "", file_arg[1]))
    return(dirname(dirname(script_path)))  # up from dgp/ to repo root
  }
  normalizePath(getwd())
}
REPO_ROOT <- get_repo_root()

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
OUT_DIR    <- if (length(args) >= 1) args[1] else file.path(REPO_ROOT, "example", "data")
N_DATASETS <- if (length(args) >= 2) as.integer(args[2]) else 50L
PRESET     <- if (length(args) >= 3) args[3] else "mostly_constant"

CATE_PRESETS <- list(
  mostly_constant = c(0.70, 0.10, 0.02, 0.02, 0.16),
  balanced        = c(0.20, 0.25, 0.25, 0.15, 0.15)
)
if (!PRESET %in% names(CATE_PRESETS)) {
  stop(sprintf("Unknown preset '%s'. Use one of: %s",
               PRESET, paste(names(CATE_PRESETS), collapse = ", ")))
}
CATE_PROBS <- CATE_PRESETS[[PRESET]]
stopifnot(N_DATASETS >= 1, abs(sum(CATE_PROBS) - 1) < 1e-12)

source(file.path(REPO_ROOT, "dgp", "mirror_dgp.R"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_OBS       <- 1000      # Rows per dataset
N_POP_MC    <- 200000    # Monte Carlo rows for the true PATE
SEED_PARAMS <- 1000L     # Param seed for dataset i:  1000 + i
SEED_TRAIN  <- 100000L   # Train seed for dataset i:  100000 + i
SEED_TEST   <- 200000L   # Test seed for dataset i:   200000 + i
SEED_POP    <- 300000L   # Population MC seed:        300000 + i

for (sub in c("train", "test", "truth", "pop")) {
  dir.create(file.path(OUT_DIR, sub), recursive = TRUE, showWarnings = FALSE)
}

cat(sprintf("Generating %d simulated train/test pairs (preset: %s)\n",
            N_DATASETS, PRESET))
cat(sprintf("Output: %s\n", normalizePath(OUT_DIR)))

# ---------------------------------------------------------------------------
# population_colmeans(params): exact means of the 54 encoded covariates
#
# Column order matches prepare_covariates() in R/prepare_covariates.R and
# generate_acic_covariates() in dgp/mirror_dgp.R: 20 one-hot categorical
# (level 1 dropped), 19 binary, 12 continuous, 3 zero-inflated.
# ---------------------------------------------------------------------------
population_colmeans <- function(params) {
  n_levels <- c(3, 4, 3, 5, 4, 7)                       # x1-x6
  cat_means <- unlist(lapply(n_levels, function(J) rep(1 / J, J - 1)))
  means <- c(cat_means,                                 # 20 columns
             params$bin_probs,                          # 19 columns
             rep(0.5, 12),                              # 12 columns
             (1 - params$zero_fracs) * 0.5)             #  3 columns
  stopifnot(length(means) == 54)
  means
}

# Helper: write a truth file for one sample
write_truth <- function(tau, path) {
  write.csv(data.frame(ID = seq_len(nrow(tau)),
                       tau_b = tau[, "b"], tau_c = tau[, "c"],
                       tau_d = tau[, "d"], tau_e = tau[, "e"]),
            path, row.names = FALSE)
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
metadata_rows <- vector("list", N_DATASETS)
t_start <- proc.time()

for (i in seq_len(N_DATASETS)) {
  # 1. DGP structure (once per pair), using the parametric defaults
  params <- draw_acic_params_v2(seed = SEED_PARAMS + i, cate_probs = CATE_PROBS)

  # 2. Observations: train and test share params, differ in randomness
  sim_train <- generate_acic_mirror_v2(n = N_OBS, params = params, seed = SEED_TRAIN + i)
  sim_test  <- generate_acic_mirror_v2(n = N_OBS, params = params, seed = SEED_TEST + i)

  write.csv(format_acic_csv(sim_train),
            file.path(OUT_DIR, "train", sprintf("data_%d.csv", i)), row.names = FALSE)
  write.csv(format_acic_csv(sim_test),
            file.path(OUT_DIR, "test", sprintf("data_%d.csv", i)), row.names = FALSE)
  write_truth(sim_train$tau_true, file.path(OUT_DIR, "truth", sprintf("train_truth_%d.csv", i)))
  write_truth(sim_test$tau_true,  file.path(OUT_DIR, "truth", sprintf("test_truth_%d.csv", i)))

  # 3. Exact population covariate means (input to the PATE estimator)
  x_pop <- population_colmeans(params)
  write.csv(data.frame(column = seq_along(x_pop), mean = x_pop),
            file.path(OUT_DIR, "pop", sprintf("x_pop_colmeans_%d.csv", i)),
            row.names = FALSE)

  # 4. True PATE for the training DGP: raw surface averaged over a large
  #    population draw, times the training sample's scale factor
  set.seed(SEED_POP + i)
  X_pop <- generate_acic_covariates(N_POP_MC, params$bin_probs, params$zero_fracs)$X
  pate_true <- colMeans(compute_cate_surface(X_pop, params)) * sim_train$tau_scale
  names(pate_true) <- c("b", "c", "d", "e")

  # Sanity check: the Monte Carlo covariate means should match the exact ones
  stopifnot(max(abs(colMeans(X_pop) - x_pop)) < 0.01)

  metadata_rows[[i]] <- data.frame(
    dataset_id    = i,
    target_r2     = params$target_r2,
    target_y_mean = params$target_y_mean,
    target_y_sd   = params$target_y_sd,
    arm_prob_a    = params$arm_probs[1], arm_prob_b = params$arm_probs[2],
    arm_prob_c    = params$arm_probs[3], arm_prob_d = params$arm_probs[4],
    arm_prob_e    = params$arm_probs[5],
    cate_type_b   = params$cate_types["b"], cate_type_c = params$cate_types["c"],
    cate_type_d   = params$cate_types["d"], cate_type_e = params$cate_types["e"],
    pate_true_b   = pate_true["b"], pate_true_c = pate_true["c"],
    pate_true_d   = pate_true["d"], pate_true_e = pate_true["e"],
    tau_scale     = sim_train$tau_scale,
    noise_sd      = sim_train$noise_sd,
    stringsAsFactors = FALSE, row.names = NULL
  )

  if (i %% 10 == 0 || i == N_DATASETS) {
    cat(sprintf("  [%d/%d] %.0f s elapsed\n", i, N_DATASETS, (proc.time() - t_start)[3]))
  }
}

metadata <- do.call(rbind, metadata_rows)
write.csv(metadata, file.path(OUT_DIR, "metadata.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# Summary and final checks
# ---------------------------------------------------------------------------
stopifnot(
  nrow(metadata) == N_DATASETS,
  length(list.files(file.path(OUT_DIR, "train"), pattern = "^data_\\d+\\.csv$")) >= N_DATASETS,
  length(list.files(file.path(OUT_DIR, "pop"),   pattern = "\\.csv$")) >= N_DATASETS
)
types <- table(unlist(metadata[, paste0("cate_type_", c("b", "c", "d", "e"))]))
cat("\nRealized CATE type mix across all contrasts:\n")
print(round(100 * types / sum(types), 1))
cat(sprintf("Done: %d pairs written to %s\n", N_DATASETS, OUT_DIR))
