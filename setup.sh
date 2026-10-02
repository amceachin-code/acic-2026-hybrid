#!/usr/bin/env bash
# ===========================================================================
# setup.sh -- Install R package dependencies for the Hybrid ensemble
#
# The pipeline requires one R package:
#   - grf       (multi-arm causal forests)
# The Horseshoe sampler and the simulator use base R only.
#
# Usage:
#   bash setup.sh
# ===========================================================================

set -euo pipefail

echo "=== ACIC 2026 Hybrid Ensemble: Setup ==="
echo ""

# ---------------------------------------------------------------------------
# Check R is available
# ---------------------------------------------------------------------------
if ! command -v Rscript &> /dev/null; then
    echo "ERROR: Rscript not found. Please install R (>= 4.1) first."
    echo "  macOS:   brew install r"
    echo "  Ubuntu:  sudo apt-get install r-base"
    exit 1
fi

R_VERSION=$(Rscript --version 2>&1 | head -1)
echo "Found R: $R_VERSION"
echo ""

# ---------------------------------------------------------------------------
# Install packages (idempotent: skips if already installed)
# ---------------------------------------------------------------------------
echo "Installing R package: grf..."
echo "This may take a few minutes on first run."
echo ""

Rscript -e '
pkgs <- c("grf")
installed <- rownames(installed.packages())
to_install <- setdiff(pkgs, installed)

if (length(to_install) == 0) {
  cat("All packages already installed.\n")
} else {
  cat(sprintf("Installing: %s\n", paste(to_install, collapse = ", ")))
  install.packages(to_install, repos = "https://cloud.r-project.org", quiet = TRUE)
}

# Verify all packages load
for (pkg in pkgs) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  cat(sprintf("  %-10s v%s  OK\n", pkg, packageVersion(pkg)))
}
cat("\nAll dependencies satisfied.\n")
'

echo ""
echo "Setup complete."
