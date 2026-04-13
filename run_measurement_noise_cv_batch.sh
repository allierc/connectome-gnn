#!/bin/bash
# CV pipeline for measurement noise configs (GNN + Known-ODE).
#
# Covers the "measurement noise NEW" result in the paper (Table 1).
# Both models trained on DAVIS data; Phase 2 tests zero-shot transfer
# to YouTube-VOS; Phase 3 retrains on YouTube-VOS and extracts parameters.
#
#   Config                                noise_dyn  noise_meas  conn_R2 ceiling
#   ─────────────────────────────────────────────────────────────────────────────
#   flyvis_noise_005_010_winner           0.05       0.10        0.745±0.004 (4-seed)
#   flyvis_noise_005_010_known_ode_winner 0.05       0.10        0.784 (4-seed CV=0.38%)
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 36000 < run_measurement_noise_cv_batch.sh
#
# Or run interactively:
#   bash run_measurement_noise_cv_batch.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
N_SEEDS=5

CONFIGS=(
    "${CFG_DIR}/flyvis_noise_005_010_winner"
    "${CFG_DIR}/flyvis_noise_005_010_known_ode_winner"
)

echo "============================================================"
echo "Measurement noise CV batch — $(date)"
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
echo "All measurement noise CV runs complete — $(date)"
echo "============================================================"
