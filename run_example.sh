#!/usr/bin/env bash
# =============================================================================
# run_example.sh: Run the Hybrid ensemble on the bundled simulated datasets
#
# Usage:
#   bash run_example.sh [n_datasets] [workers]
#
# Arguments:
#   n_datasets  Run and score example datasets 1..n_datasets (default: 50, all)
#   workers     Parallel R processes (default: 4)
#
# What it does:
#   1. Fits the Hybrid ensemble on example/data/train/data_<i>.csv, passing
#      the matching population covariate means so PATE is model-based.
#   2. Writes outputs to example/output/. A dataset counts as done only when
#      its last output file (dataset_<i>_pbest_probs.csv) exists, so runs
#      interrupted partway are redone.
#   3. Scores datasets 1..n_datasets against the true effects (R/evaluate.R).
#   Exits non-zero if any dataset failed.
#
# Runtime: about 15 to 20 seconds per dataset per worker. With 4 workers,
# all 50 datasets take about 4 to 5 minutes.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
N_DATASETS="${1:-50}"
WORKERS="${2:-4}"

DATA_DIR="$REPO_ROOT/example/data"
OUT_DIR="$REPO_ROOT/example/output"

if [[ ! -f "$DATA_DIR/metadata.csv" ]]; then
    echo "ERROR: $DATA_DIR/metadata.csv not found."
    echo "Regenerate the example data with: Rscript dgp/generate_data.R"
    exit 1
fi

# Validate arguments (BSD seq counts down for "seq 1 0", so check explicitly)
N_AVAILABLE=$(( $(wc -l < "$DATA_DIR/metadata.csv") - 1 ))
if ! [[ "$N_DATASETS" =~ ^[0-9]+$ ]] || (( N_DATASETS < 1 || N_DATASETS > N_AVAILABLE )); then
    echo "ERROR: n_datasets must be an integer from 1 to $N_AVAILABLE (got '$N_DATASETS')"
    exit 1
fi
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || (( WORKERS < 1 )); then
    echo "ERROR: workers must be a positive integer (got '$WORKERS')"
    exit 1
fi

mkdir -p "$OUT_DIR"

# Build the list of datasets that still need to run
TODO=()
for id in $(seq 1 "$N_DATASETS"); do
    [[ -f "$OUT_DIR/dataset_${id}_pbest_probs.csv" ]] || TODO+=("$id")
done

echo "=== Hybrid ensemble: example run ==="
echo "  Datasets: $N_DATASETS (${#TODO[@]} to run, $((N_DATASETS - ${#TODO[@]})) already done)"
echo "  Workers:  $WORKERS"
echo "  Output:   $OUT_DIR"
echo ""

RUN_STATUS=0
if [[ ${#TODO[@]} -gt 0 ]]; then
    LOG="$OUT_DIR/run.log"
    : > "$LOG"
    export REPO_ROOT DATA_DIR OUT_DIR LOG
    # Each id is passed as $1 (xargs -I has a 255-byte limit on macOS).
    # A failing worker exits 1, so xargs exits non-zero (123) at the end.
    printf '%s\n' "${TODO[@]}" | xargs -n 1 -P "$WORKERS" bash -c '
        id="$1"
        if Rscript "$REPO_ROOT/R/run_hybrid.R" \
             "$DATA_DIR/train/data_${id}.csv" "$OUT_DIR" \
             "$DATA_DIR/pop/x_pop_colmeans_${id}.csv" >> "$LOG" 2>&1; then
            echo "  dataset $id done"
        else
            echo "  dataset $id FAILED (see $LOG)"
            exit 1
        fi
    ' _ || RUN_STATUS=$?
fi

Rscript "$REPO_ROOT/R/evaluate.R" "$DATA_DIR" "$OUT_DIR" train "$N_DATASETS"

if [[ $RUN_STATUS -ne 0 ]]; then
    echo ""
    echo "ERROR: one or more datasets failed; see $OUT_DIR/run.log"
    exit 1
fi
