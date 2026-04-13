#!/bin/bash
# CV pipeline for the joint GNN+SIREN (INR) config.
#
# Tests the joint GNN+SIREN champion trained on noise_005 DAVIS data.
# Phase 2 tests zero-shot transfer to YouTube-VOS (auto-skipped if no
# pre-trained model found). Phase 3 retrains on YouTube-VOS and extracts
# connectivity + field parameters.
#
#   Config                        Noise   conn_R2    field_R2
#   ─────────────────────────────────────────────────────────
#   flyvis_noise_005_INR          0.05    0.942      0.709
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is < run_INR_cv_batch.sh
#
# Or run interactively:
#   bash run_INR_cv_batch.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
N_SEEDS=5

CONFIGS=(
    "${CFG_DIR}/flyvis_noise_005_INR"
)

echo "============================================================"
echo "INR CV batch — $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Configs:   ${#CONFIGS[@]}   Seeds: ${N_SEEDS}"
echo "============================================================"

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Starting CV: $(basename ${cfg})"
    echo "Start time: $(date)"
    echo "------------------------------------------------------------"
    python "${REPO_DIR}/GNN_Main.py" \
        -o cv "${cfg}" \
        --n_seeds "${N_SEEDS}" \
        --output_root "${DATA_ROOT}"
    echo "Done: $(basename ${cfg})  ($(date))"
done

echo ""
echo "============================================================"
echo "All INR CV runs complete — $(date)"
echo "============================================================"
