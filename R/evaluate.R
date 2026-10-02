# ===========================================================================
# R/evaluate.R: Score Hybrid output against simulated ground truth
#
# Purpose:
#   Compares pipeline output with the true treatment effects written by
#   dgp/generate_data.R, and reports accuracy and interval coverage.
#
#   The pipeline fits and predicts on the same file (as in the ACIC task), so
#   predictions made on train/data_<i>.csv are scored against
#   truth/train_truth_<i>.csv.
#
# Usage:
#   Rscript R/evaluate.R <data_dir> <pred_dir>
#
# Arguments:
#   data_dir  Directory written by dgp/generate_data.R (needs truth/ and
#             metadata.csv)
#   pred_dir  Directory with dataset_<i>_icates.csv and
#             dataset_<i>_aggregates.csv files from R/run_hybrid.R
#
# Metrics (averaged over the 4 contrasts, then over datasets):
#   iCATE:  RMSE, bias, 95% CI coverage, CI width, treatment assignment
#           accuracy, and regret (from R/evaluate_multi.R)
#   sCATE:  absolute error and 95% CI coverage vs the true sample mean CATE
#   PATE:   absolute error and 95% CI coverage vs the true population ATE
#
# Outputs:
#   <pred_dir>/evaluation_by_dataset.csv  One row per evaluated dataset
#   Summary table on stdout
# ===========================================================================

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
source(file.path(REPO_ROOT, "R", "evaluate_multi.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript R/evaluate.R <data_dir> <pred_dir>\n")
  quit(status = 1)
}
data_dir <- args[1]
pred_dir <- args[2]

arms     <- c("b", "c", "d", "e")
metadata <- read.csv(file.path(data_dir, "metadata.csv"), stringsAsFactors = FALSE)

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
for (i in metadata$dataset_id) {
  icate_file <- file.path(pred_dir, sprintf("dataset_%d_icates.csv", i))
  agg_file   <- file.path(pred_dir, sprintf("dataset_%d_aggregates.csv", i))
  if (!file.exists(icate_file) || !file.exists(agg_file)) next

  truth <- read.csv(file.path(data_dir, "truth", sprintf("train_truth_%d.csv", i)))
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

  # PATE: true value is the population mean CATE from the DGP
  pa    <- agg_rows(agg, "PATE")
  meta  <- metadata[metadata$dataset_id == i, ]
  pa_tr <- unlist(meta[, paste0("pate_true_", arms)])

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
    pate_abs_err   = mean(abs(pa$Estimate - pa_tr)),
    pate_coverage  = mean(pa_tr >= pa$L95 & pa_tr <= pa$U95)
  )
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
cat(sprintf("\n=== Hybrid ensemble on %d simulated datasets ===\n", nrow(by_dataset)))
cat(sprintf("  %-16s %8s %8s\n", "Metric", "Mean", "Median"))
cat(sprintf("  %s\n", strrep("-", 34)))
for (col in setdiff(names(by_dataset), "dataset_id")) {
  cat(sprintf("  %-16s %8.3f %8.3f\n", col,
              mean(by_dataset[[col]]), median(by_dataset[[col]])))
}
cat(sprintf("\nPer-dataset results: %s\n", out_file))
