#!/bin/bash
# Blank-50 YT-only CV pipeline — V_rest recovery test.
#
# Runs two Python scripts in sequence:
#   1. run_generate_YT_data_blank50.py — pre-generate 15 datasets
#      (3 conditions × 5 folds) with `simulation.blank_prefix_fraction: 0.50`.
#      First 50% of each YT video sequence is zero-stimulus — supplies
#      the V_rest training signal missing from the default YT CV.
#   2. run_GNN_unified_blank50.py     — train 15 GNNs (uniform HPs,
#      ~1 h per GNN on a100) and emit the TeX table.
#
# Conditions (first 3 rows of the YT CV table):
#   flyvis_noise_free
#   flyvis_noise_005
#   flyvis_noise_05
#
# Dataset folders created (15):
#   <DATA_ROOT>/graphs_data/fly/<base>_yt_blank50_cv<i:02d>/
#
# Config files created (15):
#   <DATA_ROOT>/config/fly/<base>_yt_blank50_unified_cv<i:02d>.yaml
#
# TeX output:
#   <DATA_ROOT>/log/cv_yt_blank50_unified_rows.tex
#
# Submit to cluster:
#   bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is bash run_GNN_blank50_pipeline.sh
#
# Or run interactively:
#   bash run_GNN_blank50_pipeline.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
if [ -z "$REPO_DIR" ] || [ ! -f "$REPO_DIR/run_GNN_unified_blank50.py" ]; then
    REPO_DIR="$(cd "$(pwd)" && pwd)"
    if [ ! -f "$REPO_DIR/run_GNN_unified_blank50.py" ]; then
        echo "ERROR: cannot locate repo root (missing run_GNN_unified_blank50.py)" >&2
        exit 1
    fi
fi
cd "$REPO_DIR"
echo "repo dir: $REPO_DIR"

if [ -n "${GNN_OUTPUT_ROOT:-}" ]; then
    echo "GNN_OUTPUT_ROOT=$GNN_OUTPUT_ROOT"
fi

echo
echo "============================================================"
echo "Step 1/2: pre-generate 15 blank-50 YT datasets"
echo "============================================================"
python run_generate_YT_data_blank50.py

echo
echo "============================================================"
echo "Step 2/2: train 15 GNNs (unified HPs) + emit TeX table"
echo "============================================================"
python run_GNN_unified_blank50.py

echo
echo "============================================================"
echo "DONE — see log/cv_yt_blank50_unified_rows.tex"
echo "============================================================"
