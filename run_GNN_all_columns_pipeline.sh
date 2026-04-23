#!/bin/bash
# Full-fly (all_columns=True) YT-only CV pipeline.
#
# Runs two Python scripts in sequence:
#   1. run_generate_holdout_data_all_columns.py — pre-generate 40 datasets
#      (8 conditions × 5 folds) at 45669 neurons / 1513231 edges each.
#   2. run_GNN_unified_all_columns.py     — train 40 GNNs (uniform HPs,
#      ~1–5 h per GNN on a100) and emit the TeX table.
#
# Dataset folders created (40):
#   <DATA_ROOT>/graphs_data/fly/<base>_yt_all_cv<i:02d>/
#
# Config files created (40):
#   <DATA_ROOT>/config/fly/<base>_yt_all_unified_cv<i:02d>.yaml
#
# TeX output:
#   <DATA_ROOT>/log/cv_yt_all_unified_rows.tex
#
# Submit to cluster (reserves one a100 slot; inner pipeline dispatches
# 40 child LSF jobs itself):
#   bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is bash run_GNN_all_columns_pipeline.sh
#
# Or run interactively:
#   bash run_GNN_all_columns_pipeline.sh

set -euo pipefail

# Resolve repo dir robustly (works under `bash script.sh`, `bash < script.sh`,
# and `bsub -Is bash script.sh`).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
if [ -z "$REPO_DIR" ] || [ ! -f "$REPO_DIR/run_GNN_unified_all_columns.py" ]; then
    # stdin fallback
    REPO_DIR="$(cd "$(pwd)" && pwd)"
    if [ ! -f "$REPO_DIR/run_GNN_unified_all_columns.py" ]; then
        echo "ERROR: cannot locate repo root (missing run_GNN_unified_all_columns.py)" >&2
        exit 1
    fi
fi
cd "$REPO_DIR"
echo "repo dir: $REPO_DIR"

# Optional: respect $GNN_OUTPUT_ROOT if set, else rely on data_paths.json
if [ -n "${GNN_OUTPUT_ROOT:-}" ]; then
    echo "GNN_OUTPUT_ROOT=$GNN_OUTPUT_ROOT"
fi

echo
echo "============================================================"
echo "Step 1/2: pre-generate 40 all-columns YT datasets"
echo "============================================================"
python run_generate_holdout_data_all_columns.py

echo
echo "============================================================"
echo "Step 2/2: train 40 GNNs (unified HPs) + emit TeX table"
echo "============================================================"
python run_GNN_unified_all_columns.py

echo
echo "============================================================"
echo "DONE — see log/cv_yt_all_unified_rows.tex"
echo "============================================================"
