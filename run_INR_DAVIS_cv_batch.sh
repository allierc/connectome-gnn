#!/bin/bash
# CV pipeline for the joint GNN+SIREN (INR) model with DAVIS pre-training.
#
# Three-phase pipeline:
#   Phase 1 — Train on DAVIS data → saves best_model_*.pt
#   Phase 2 — Zero-shot: DAVIS-trained model tested on held-out YouTube-VOS folds
#   Phase 3 — Re-train on each YouTube-VOS fold → parameter recovery
#
# Phase 1 is run explicitly here (-o train) so that the DAVIS model is available
# for the zero-shot generalisation test in phase 2 of the CV run (-o cv).
# Without phase 1, -o cv auto-skips phase 2 ("no pre-trained DAVIS model found").
#
#   Config                        Noise   conn_R2    field_R2
#   ─────────────────────────────────────────────────────────
#   flyvis_noise_005_INR          0.05    0.942      0.709
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is < run_INR_DAVIS_cv_batch.sh
#
# Or run interactively:
#   bash run_INR_DAVIS_cv_batch.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
N_SEEDS=5

CONFIGS=(
    "${CFG_DIR}/flyvis_noise_005_INR"
)

echo "============================================================"
echo "INR DAVIS+CV batch — $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Configs:   ${#CONFIGS[@]}   Seeds: ${N_SEEDS}"
echo "============================================================"

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Phase 1 — DAVIS training: $(basename ${cfg})"
    echo "Start: $(date)"
    echo "------------------------------------------------------------"
    python "${REPO_DIR}/GNN_Main.py" \
        -o train "${cfg}" \
        --output_root "${DATA_ROOT}"

    echo ""
    echo "------------------------------------------------------------"
    echo "Phase 2+3 — CV (zero-shot + re-train): $(basename ${cfg})"
    echo "Start: $(date)"
    echo "------------------------------------------------------------"
    python "${REPO_DIR}/GNN_Main.py" \
        -o cv "${cfg}" \
        --n_seeds "${N_SEEDS}" \
        --output_root "${DATA_ROOT}"

    echo "Done: $(basename ${cfg})  ($(date))"
done

echo ""
echo "============================================================"
echo "All INR DAVIS+CV runs complete — $(date)"
echo "============================================================"
