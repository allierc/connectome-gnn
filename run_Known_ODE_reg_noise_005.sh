#!/bin/bash
# CV pipeline for ONE Known-ODE-reg condition: low intrinsic noise (sigma=0.05).
#
# Config: flyvis_noise_005_known_ode_reg_winner
#   L1/L2 bio-parameter regularization (tau, V_rest). The agentic-loop
#   winner set all reg coefficients to 0 — this config effectively acts as
#   a control for the regularization study.
#
# `-o cv` runs Phase 1 (YouTube-VOS data gen, skipped if present),
# Phase 2 (auto-train DAVIS base model if missing, then zero-shot test on
# each YT fold), and Phase 3 (5-fold YT retrain + parameter extraction).
# Skipped entirely if cv_summary.txt already exists.
#
# Output:
#   <DATA_ROOT>/log/cv_known_ode_reg_noise_005_rows.tex
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is < run_Known_ODE_reg_noise_005.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
CFG_NAME="flyvis_noise_005_known_ode_reg_winner"
N_SEEDS=5

echo "============================================================"
echo "Known-ODE-reg (low intrinsic noise) CV — $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Config:    ${CFG_NAME}   Seeds: ${N_SEEDS}"
echo "============================================================"

SUMMARY="${DATA_ROOT}/log/fly/${CFG_NAME}/results/cv_summary.txt"
if [ -f "${SUMMARY}" ]; then
    echo "[skip] cv_summary.txt already exists at ${SUMMARY}"
else
    echo "[run ] generate + train + test + plot + CV via -o cv"
    python "${REPO_DIR}/GNN_Main.py" \
        -o cv "${CFG_DIR}/${CFG_NAME}" \
        --n_seeds "${N_SEEDS}" \
        --output_root "${DATA_ROOT}"
    echo "Done: ${CFG_NAME}  ($(date))"
fi

echo ""
echo "============================================================"
echo "Emitting TeX row — $(date)"
echo "============================================================"
python "${REPO_DIR}/scripts/emit_Known_ODE_reg_noise_005_rows.py" \
    --output_root "${DATA_ROOT}"

echo ""
echo "============================================================"
echo "Known-ODE-reg (low intrinsic noise) CV complete — $(date)"
echo "============================================================"
