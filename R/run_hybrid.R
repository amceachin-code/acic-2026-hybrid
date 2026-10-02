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
#                  output file names and as the default seed; otherwise the
#                  id is 1 (with a warning, since outputs could collide).
#   output_dir     Directory for output CSVs (created if missing).
#   pop_means_csv  Optional. Population means of the 54 encoded covariates
#                  (CSV with a 'mean' column, as written by
#                  dgp/generate_data.R). Enables the model-based PATE.
#                  Without it, PATE falls back to the sCATE. Pass "" to
#                  skip it while still giving a seed.
#   seed           Optional integer seed for the Horseshoe Gibbs sampler
#                  (default: dataset id). GRF always uses its own fixed
#                  seed (42, set in R/fit_hybrid.R).
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
# Dependencies: grf (install via setup.sh)
# ===========================================================================

# ---------------------------------------------------------------------------
# Repo root: the parent of this script's folder (works from any directory)
# ---------------------------------------------------------------------------
script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
REPO_ROOT <- if (length(script_file) > 0) {
  normalizePath(file.path(dirname(script_file[1]), ".."))
} else {
  normalizePath(getwd())
}

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
id_match <- regmatches(basename(data_csv), regexec("^data_(\\d+)\\.csv$", basename(data_csv)))[[1]]
if (length(id_match) == 2) {
  dataset_id <- as.integer(id_match[2])
} else {
  dataset_id <- 1L
  warning(sprintf("'%s' is not named data_<id>.csv; using id 1 for output names. ",
                  basename(data_csv)),
          "Outputs from other such files in the same directory would be overwritten.",
          call. = FALSE)
}
seed <- if (length(args) >= 4) suppressWarnings(as.integer(args[4])) else dataset_id
if (is.na(seed)) {
  cat(sprintf("ERROR: seed must be an integer (got '%s')\n", args[4]))
  quit(status = 1)
}
if (!is.null(pop_csv) && !file.exists(pop_csv)) {
  cat(sprintf("ERROR: Population means file not found: %s\n", pop_csv))
  quit(status = 1)
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------------------
# Libraries and method code
# ---------------------------------------------------------------------------
# Only grf is needed. fit_ols.R is sourced for build_linear_data(), which the
# Horseshoe uses; its OLS estimators (which need sandwich) are not called.
suppressPackageStartupMessages(library(grf))

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

# A deterministic per-dataset seed makes the Horseshoe draws identical across
# runs, regardless of process scheduling or parallelism order.
set.seed(seed)

x_bar_pop <- NULL
if (!is.null(pop_csv)) {
  pop_df <- read.csv(pop_csv)
  if (!"mean" %in% names(pop_df) || nrow(pop_df) != 54) {
    cat(sprintf("ERROR: %s must have a 'mean' column with 54 rows (got %d rows)\n",
                pop_csv, nrow(pop_df)))
    quit(status = 1)
  }
  x_bar_pop <- pop_df$mean
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
