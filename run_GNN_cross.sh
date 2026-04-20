#!/bin/bash
# GNN cross-check with a SINGLE uniform agentic HP set — the winner HPs
# from the flyvis_noise_005_null_edges_pc_400 exploration — applied across
# all 8 conditions. Twin of run_GNN_conditions.sh; only the HP source
# differs.
#
# For each of 8 conditions, trains a fresh GNN on YouTube-VOS data using
# the SHARED null-edges HPs, then cross-rolls it out on DAVIS held-out
# test data. The 6-column table reports:
#
#   prediction columns (one-step r, rollout r) : YT-trained model rolled
#       out on DAVIS held-out test data.
#   parameter-recovery columns (W, tau, V_rest, cluster) : YT-trained
#       model's own learned parameters (from results/metrics.txt).
#
# Caching (per condition):
#   data_generate skipped if <graphs_data>/x_list_train exists
#   data_train    skipped if models/best_model_with_*.pt    exists
#   data_test     skipped if results_rollout_on_<base>.log  exists
#   data_plot     skipped if results/metrics.txt            exists
#
# Use  --force_test  to redo only the test + plot steps (generate + train
# remain cached).
#
# Output:
#   <DATA_ROOT>/log/cv_yt_cross_rows.tex
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is < run_GNN_cross.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
SUFFIX="yt_cross"
HP_YAML="flyvis_noise_005_null_edges_pc_400_winner"

FORCE_FLAG=""
if [ "${1:-}" = "--force_test" ]; then
    FORCE_FLAG="--force_test"
fi

echo "============================================================"
echo "GNN cross-check (uniform ${HP_YAML} HPs) — $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Suffix:    ${SUFFIX}"
echo "HP yaml:   ${HP_YAML}"
echo "Force:     ${FORCE_FLAG:-<none>}"
echo "============================================================"

python "${REPO_DIR}/scripts/write_cross_yt_configs.py" \
    --hp_source uniform \
    --hp_yaml   "${HP_YAML}" \
    --suffix    "${SUFFIX}" \
    --n_folds   5

python "${REPO_DIR}/scripts/run_cross_yt_parallel.py" \
    --suffix "${SUFFIX}" \
    --output_root "${DATA_ROOT}" \
    --n_folds 5 \
    --emit_tex cv_yt_cross_rows.tex \
    ${FORCE_FLAG}

# Final emit (also written after each condition by --emit_tex above).
python "${REPO_DIR}/scripts/emit_cross_table_rows.py" \
    --suffix "${SUFFIX}" \
    --output_tex cv_yt_cross_rows.tex \
    --output_root "${DATA_ROOT}" \
    --n_folds 5

echo ""
echo "============================================================"
echo "GNN cross-check complete — $(date)"
echo "============================================================"
