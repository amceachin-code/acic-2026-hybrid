# ===========================================================================
# R/run_hybrid.R: Run the Hybrid ensemble (Horseshoe + GRF) on one dataset
#
# Purpose:
#   Command-line entry point. Reads one CSV in the ACIC 2026 layout
#   (ID, y, z, x1-x40), fits both components, combines them, and writes
#   iCATE, aggregate (sCATE, subCATE, PATE), and treatment-choice outputs.
#
# Usage:
#   Rscript R/run_hybrid.R <data_csv> <output_dir> [pop_means_csv] [seed]
#
# Arguments:
#   data_csv       Input file. If named data_<id>.csv, <id> is used for
#                  output file names and as the default seed; otherwise 1.
#   output_dir     Directory for output CSVs (created if missing).
#   pop_means_csv  Optional. Population means of the 54 encoded covariates
#                  (CSV with a 'mean' column, as written by
#                  dgp/generate_data.R). Enables the model-based PATE.
#                  Without it, PATE falls back to the sCATE.
#   seed           Optional integer RNG seed (default: dataset id).
#
# Example:
#   Rscript R/run_hybrid.R example/data/train/data_1.csv example/output \
#     example/data/pop/x_pop_colmeans_1.csv
#
# Outputs (in output_dir, <id> taken from the file name):
#   dataset_<id>_icates.csv            iCATE estimates and 95% CIs (wide)
#   dataset_<id>_aggregates.csv        sCATE, subCATE (by x12), PATE, best arms
#   dataset_<id>_best_icates.csv       Best arm per unit (argmax of estimates)
#   dataset_<id>_best_icates_pbest.csv Best arm per unit (posterior P(best))
#   dataset_<id>_pbest_probs.csv       P(each arm is best) per unit
#   A "TIMING|<id>|hybrid|<seconds>|<mean CI width>" line on stdout
#
# Dependencies: grf, ranger, sandwich (install via setup.sh)
# ===========================================================================

# ---------------------------------------------------------------------------
# Repo root detection: works when called via Rscript from any directory
# ---------------------------------------------------------------------------
get_repo_root <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("--file=", "", file_arg[1]))
    return(dirname(dirname(script_path)))  # up from R/ to repo root
  }
  normalizePath(getwd())
}
REPO_ROOT <- get_repo_root()

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript R/run_hybrid.R <data_csv> <output_dir> [pop_means_csv] [seed]\n")
  quit(status = 1)
}
data_csv   <- args[1]
output_dir <- args[2]
pop_csv    <- if (length(args) >= 3 && nchar(args[3]) > 0) args[3] else NULL

if (!file.exists(data_csv)) {
  cat(sprintf("ERROR: Data file not found: %s\n", data_csv))
  quit(status = 1)
}

# Dataset id from the file name (data_<id>.csv), else 1
id_match   <- regmatches(basename(data_csv), regexec("^data_(\\d+)\\.csv$", basename(data_csv)))[[1]]
dataset_id <- if (length(id_match) == 2) as.integer(id_match[2]) else 1L
seed       <- if (length(args) >= 4) as.integer(args[4]) else dataset_id

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Libraries and method code
# ---------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(grf)
  library(ranger)
  library(sandwich)  # HC1 robust SEs (used by fit_ols.R)
})

for (f in c("prepare_covariates.R", "fit_ols.R", "horseshoe_gibbs.R",
            "fit_horseshoe.R", "compute_p_best.R", "compute_aggregate_se.R",
            "compute_pate.R", "fit_hybrid.R")) {
  source(file.path(REPO_ROOT, "R", f))
}

# ---------------------------------------------------------------------------
# Data, seed, and optional population means
# ---------------------------------------------------------------------------
df <- read.csv(data_csv, stringsAsFactors = FALSE)
required_cols <- c("ID", "y", "z", paste0("x", 1:40))
missing_cols  <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  cat(sprintf("ERROR: Missing columns: %s\n", paste(missing_cols, collapse = ", ")))
  quit(status = 1)
}

# A deterministic per-dataset seed makes results identical across runs,
# regardless of process scheduling or parallelism order.
set.seed(seed)

x_bar_pop <- NULL
if (!is.null(pop_csv)) {
  x_bar_pop <- read.csv(pop_csv)$mean
  stopifnot(length(x_bar_pop) == 54)
  cat(sprintf("  [PATE] Loaded population covariate means from %s\n", pop_csv))
} else {
  cat("  [PATE] No population means supplied; PATE will fall back to sCATE\n")
}

# ---------------------------------------------------------------------------
# Fit and save
# ---------------------------------------------------------------------------
timing_row <- fit_hybrid_dataset(df, dataset_id, output_dir, x_bar_pop = x_bar_pop)

cat(sprintf("TIMING|%d|hybrid|%.3f|%.6f\n",
            dataset_id, timing_row$runtime_seconds, timing_row$mean_ci_width_icates))
