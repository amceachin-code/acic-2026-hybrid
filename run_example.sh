#!/usr/bin/env bash
# =============================================================================
# run_example.sh: Run the Hybrid ensemble on the bundled simulated datasets
#
# Usage:
#   bash run_example.sh [n_datasets] [workers]
#
# Arguments:
#   n_datasets  How many of the example datasets to run (default: 50, all)
#   workers     Parallel R processes (default: 4)
#
# What it does:
#   1. Fits the Hybrid ensemble on example/data/train/data_<i>.csv, passing
#      the matching population covariate means so PATE is model-based.
#   2. Writes outputs to example/output/ (skips datasets already done).
#   3. Scores the outputs against the true effects (R/evaluate.R).
#
# Runtime: about 15 to 20 seconds per dataset per worker. With 4 workers,
# all 50 datasets take roughly 4 minutes.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
N_DATASETS="${1:-50}"
WORKERS="${2:-4}"

DATA_DIR="$REPO_ROOT/example/data"
OUT_DIR="$REPO_ROOT/example/output"
mkdir -p "$OUT_DIR"

if [[ ! -f "$DATA_DIR/metadata.csv" ]]; then
    echo "ERROR: $DATA_DIR/metadata.csv not found."
    echo "Regenerate the example data with: Rscript dgp/generate_data.R"
    exit 1
fi

# Build the list of datasets that still need to run
TODO=()
for id in $(seq 1 "$N_DATASETS"); do
    [[ -f "$OUT_DIR/dataset_${id}_icates.csv" ]] || TODO+=("$id")
done

echo "=== Hybrid ensemble: example run ==="
echo "  Datasets: $N_DATASETS (${#TODO[@]} to run, $((N_DATASETS - ${#TODO[@]})) already done)"
echo "  Workers:  $WORKERS"
echo "  Output:   $OUT_DIR"
echo ""

if [[ ${#TODO[@]} -gt 0 ]]; then
    LOG="$OUT_DIR/run.log"
    : > "$LOG"
    export REPO_ROOT DATA_DIR OUT_DIR LOG
    # Each id is passed as $1 (xargs -I has a 255-byte limit on macOS)
    printf '%s\n' "${TODO[@]}" | xargs -n 1 -P "$WORKERS" bash -c '
        id="$1"
        if Rscript "$REPO_ROOT/R/run_hybrid.R" \
             "$DATA_DIR/train/data_${id}.csv" "$OUT_DIR" \
             "$DATA_DIR/pop/x_pop_colmeans_${id}.csv" >> "$LOG" 2>&1; then
            echo "  dataset $id done"
        else
            echo "  dataset $id FAILED (see $LOG)"
        fi
    ' _
fi

Rscript "$REPO_ROOT/R/evaluate.R" "$DATA_DIR" "$OUT_DIR"
