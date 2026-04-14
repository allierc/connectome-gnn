#!/bin/bash
# CV pipeline for all 6 known_ode winner configs.
#
# Each config is the result of LLM hyperparameter exploration on the
# flyvis_known_ode model (linear ODE: learns W, tau, V_rest from voltage traces).
#
#   Config                                          Noise   Edges             Peak conn_R2
#   ──────────────────────────────────────────────────────────────────────────────────────
#   flyvis_noise_free_known_ode_winner              0.0     434 112           0.9776 ± 0.0001 (8-seed)
#   flyvis_noise_005_known_ode_winner               0.05    434 112           0.9884 ± 0.0001 (12-seed)
#   flyvis_noise_005_010_known_ode_winner           0.05+   434 112           0.7838 (4-seed CV=0.38%)
#   flyvis_noise_05_known_ode_winner                0.5     434 112           0.9996 ± 0.0000 (12-seed)
#   flyvis_noise_005_null_edges_pc_400_known_ode_winner  0.05  434 112+1.7M null  0.8934 / 0.8722 mean (4-seed)
#   flyvis_noise_005_removed_pc_20_known_ode_winner 0.05    347 000 (−20%)    0.9682 / 0.9542 mean (4-seed)
#
# Phase 2 (zero-shot DAVIS→YouTube test) runs automatically when a pre-trained
# DAVIS model is found; skipped automatically if no model exists.
#
# Submit to cluster:
#   bsub -n 2 -gpu "num=1" -q gpu_a100 -W 600 -Is   < run_known_ode_cv_batch.sh
#
# Or run interactively:
#   bash run_known_ode_cv_batch.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
N_SEEDS=5

CONFIGS=(
    # "${CFG_DIR}/flyvis_noise_free_known_ode_winner"
    # "${CFG_DIR}/flyvis_noise_005_known_ode_winner"
    # "${CFG_DIR}/flyvis_noise_05_known_ode_winner"
    # "${CFG_DIR}/flyvis_noise_005_010_known_ode_winner"
    # "${CFG_DIR}/flyvis_noise_005_null_edges_pc_400_known_ode_winner"
    "${CFG_DIR}/flyvis_noise_005_removed_pc_20_known_ode_winner"
)

echo "============================================================"
echo "known_ode CV batch — $(date)"
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
echo "All known_ode CV runs complete — $(date)"
echo "============================================================"
