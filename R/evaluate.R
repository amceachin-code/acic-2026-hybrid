# ===========================================================================
# R/evaluate.R: Score Hybrid output against simulated ground truth
#
# Purpose:
#   Compares pipeline output with the true treatment effects written by
#   dgp/generate_data.R, and reports accuracy and interval coverage.
#
#   The pipeline fits and predicts on the same file (as in the ACIC task).
#   Predictions made on <split>/data_<i>.csv are scored against
#   truth/<split>_truth_<i>.csv, so pass the split you ran on.
#
# Usage:
#   Rscript R/evaluate.R <data_dir> <pred_dir> [split] [max_id]
#
# Arguments:
#   data_dir  Directory written by dgp/generate_data.R (needs truth/ and
#             metadata.csv)
#   pred_dir  Directory with dataset_<i>_icates.csv and
#             dataset_<i>_aggregates.csv files from R/run_hybrid.R
#   split     "train" (default) or "test": which files the run used
#   max_id    Only score datasets 1..max_id (default: all in metadata.csv)
#
# Metrics (per dataset, then summarized over datasets):
#   iCATE:    RMSE, bias, 95% CI coverage, CI width (averaged over the 4
#             contrasts), plus per-unit treatment assignment accuracy and
#             regret (from R/evaluate_multi.R)
#   sCATE:    absolute error and 95% CI coverage vs the true sample mean CATE
#   subCATE:  absolute error and 95% CI coverage vs the true mean CATE within
#             each x12 stratum (the quantity the point estimate targets)
#   PATE:     absolute error and 95% CI coverage vs the true population ATE.
#             The PATE target is defined for the training DGP scale, so PATE
#             is only scored for split = "train".
#
# Outputs:
#   <pred_dir>/evaluation_by_dataset.csv  One row per evaluated dataset
#   Summary table on stdout
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
source(file.path(REPO_ROOT, "R", "evaluate_multi.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript R/evaluate.R <data_dir> <pred_dir> [split] [max_id]\n")
  quit(status = 1)
}
data_dir <- args[1]
pred_dir <- args[2]
split    <- if (length(args) >= 3) args[3] else "train"
if (!split %in% c("train", "test")) {
  cat(sprintf("ERROR: split must be 'train' or 'test' (got '%s')\n", split))
  quit(status = 1)
}

arms     <- c("b", "c", "d", "e")
metadata <- read.csv(file.path(data_dir, "metadata.csv"), stringsAsFactors = FALSE)
ids      <- metadata$dataset_id
if (length(args) >= 4) ids <- ids[ids <= as.integer(args[4])]

# ---------------------------------------------------------------------------
# read_icates(): wide iCATE output -> list of N x 4 matrices (hat, L, U)
# ---------------------------------------------------------------------------
read_icates <- function(path) {
  p <- read.csv(path)
  p <- p[order(p$ID), ]
  grab <- function(suffix) {
    m <- as.matrix(p[, paste0("tau_", arms, suffix)])
    colnames(m) <- arms
    m
  }
  list(ID = p$ID, hat = grab(""), L = grab("_L95"), U = grab("_U95"))
}

# Helper: rows of the aggregates file for one estimand, ordered b, c, d, e
agg_rows <- function(agg, estimand) {
  a <- agg[agg$estimand == estimand, ]
  a[match(arms, a$z), ]
}

# ---------------------------------------------------------------------------
# Evaluate each dataset that has predictions
# ---------------------------------------------------------------------------
results <- list()
missing <- integer(0)
for (i in ids) {
  icate_file <- file.path(pred_dir, sprintf("dataset_%d_icates.csv", i))
  agg_file   <- file.path(pred_dir, sprintf("dataset_%d_aggregates.csv", i))
  if (!file.exists(icate_file) || !file.exists(agg_file)) {
    missing <- c(missing, i)
    next
  }

  truth <- read.csv(file.path(data_dir, "truth", sprintf("%s_truth_%d.csv", split, i)))
  truth <- truth[order(truth$ID), ]
  tau_true <- as.matrix(truth[, paste0("tau_", arms)])
  colnames(tau_true) <- arms

  pred <- read_icates(icate_file)
  stopifnot(identical(as.integer(pred$ID), as.integer(truth$ID)))

  # iCATE metrics
  m <- evaluate_multi(pred$hat, pred$L, pred$U, tau_true)

  # sCATE: true value is the mean of the true iCATEs in this sample
  agg   <- read.csv(agg_file, stringsAsFactors = FALSE)
  sc    <- agg_rows(agg, "sCATE")
  sc_tr <- colMeans(tau_true)

  # subCATE: true value is the mean true iCATE within each x12 stratum
  x12 <- read.csv(file.path(data_dir, split, sprintf("data_%d.csv", i)))$x12
  sub_err <- sub_cov <- c()
  for (x in c(0, 1)) {
    su    <- agg[agg$estimand == "subCATE" & agg$x == x, ]
    su    <- su[match(arms, su$z), ]
    su_tr <- colMeans(tau_true[x12 == x, , drop = FALSE])
    sub_err <- c(sub_err, abs(su$Estimate - su_tr))
    sub_cov <- c(sub_cov, su_tr >= su$L95 & su_tr <= su$U95)
  }

  # PATE: true value is the population mean CATE from the training DGP
  pa    <- agg_rows(agg, "PATE")
  meta  <- metadata[metadata$dataset_id == i, ]
  pa_tr <- unlist(meta[, paste0("pate_true_", arms)])
  if (split != "train") pa_tr[] <- NA

  results[[length(results) + 1]] <- data.frame(
    dataset_id     = i,
    icate_rmse     = m$rmse,
    icate_bias     = m$bias,
    icate_coverage = m$coverage,
    icate_width    = m$width,
    assign_acc     = m$assign_acc,
    regret         = m$regret,
    scate_abs_err  = mean(abs(sc$Estimate - sc_tr)),
    scate_coverage = mean(sc_tr >= sc$L95 & sc_tr <= sc$U95),
    subcate_abs_err  = mean(sub_err),
    subcate_coverage = mean(sub_cov),
    pate_abs_err   = mean(abs(pa$Estimate - pa_tr)),
    pate_coverage  = mean(pa_tr >= pa$L95 & pa_tr <= pa$U95)
  )
}

if (length(missing) > 0) {
  cat(sprintf("WARNING: %d of %d datasets have no predictions and were skipped: %s\n",
              length(missing), length(ids), paste(head(missing, 20), collapse = ", ")))
}
if (length(results) == 0) {
  cat(sprintf("ERROR: no predictions found in %s\n", pred_dir))
  quit(status = 1)
}

by_dataset <- do.call(rbind, results)
out_file   <- file.path(pred_dir, "evaluation_by_dataset.csv")
write.csv(by_dataset, out_file, row.names = FALSE)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat(sprintf("\n=== Hybrid ensemble on %d simulated datasets (%s split) ===\n",
            nrow(by_dataset), split))
cat(sprintf("  %-16s %8s %8s\n", "Metric", "Mean", "Median"))
cat(sprintf("  %s\n", strrep("-", 34)))
for (col in setdiff(names(by_dataset), "dataset_id")) {
  if (all(is.na(by_dataset[[col]]))) next   # e.g. PATE on the test split
  cat(sprintf("  %-16s %8.3f %8.3f\n", col,
              mean(by_dataset[[col]]), median(by_dataset[[col]])))
}
cat(sprintf("\nPer-dataset results: %s\n", out_file))
