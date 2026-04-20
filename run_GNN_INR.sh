#!/bin/bash
# Full CV pipeline for the joint GNN+INR (SIREN) model.
#
# Trains N seeds x 2 conditions of GNN+INR and emits the two TeX rows of
# tab:cv_inr (DAVIS and YouTube-VOS).  Replaces run_INR_cv_batch.sh and
# run_INR_DAVIS_cv_batch.sh.
#
# Directory layout produced per seed i in 0..N-1:
#   log/fly/<CFG_NAME>_davis_cv<i>/   DAVIS stimuli   -> Row 1
#   log/fly/<CFG_NAME>_yt_cv<i>/      YouTube-VOS     -> Row 2
#
# Caching — nothing is re-done when its output already exists (per fold):
#   data_generate skipped if <graphs_data>/x_list_train exists
#   data_train    skipped if models/best_model_with_*.pt   exists
#   data_test     skipped if results_rollout.log           exists
#   data_plot     skipped if results/metrics.txt           exists
# So the script is idempotent: interrupt and re-run freely.
#
# Output:
#   TeX rows for tab:cv_inr are written to
#     <DATA_ROOT>/log/fly/<CFG_NAME>/results/cv_inr_table_rows.tex
#   (also echoed to stdout at the end of the run).
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is < run_GNN_INR.sh
#
# Or run interactively:
#   bash run_GNN_INR.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
CFG_NAME="flyvis_noise_005_INR"
N_SEEDS=5

echo "============================================================"
echo "GNN+INR full CV — $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Config:    ${CFG_NAME}   Seeds: ${N_SEEDS}"
echo "Folds:     ${CFG_NAME}_{davis,yt}_cv00..$((N_SEEDS-1))"
echo "============================================================"

python "${REPO_DIR}/scripts/run_inr_cv.py" \
    --config "${CFG_NAME}" \
    --output_root "${DATA_ROOT}" \
    --n_seeds "${N_SEEDS}" \
    --conditions davis yt

echo ""
echo "============================================================"
echo "Emitting TeX rows for tab:cv_inr — $(date)"
echo "============================================================"
python "${REPO_DIR}/scripts/emit_inr_table_rows.py" \
    --config "${CFG_NAME}" \
    --output_root "${DATA_ROOT}" \
    --n_seeds "${N_SEEDS}"

echo ""
echo "============================================================"
echo "GNN+INR full CV complete — $(date)"
echo "============================================================"
