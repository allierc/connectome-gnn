#!/bin/bash
# Sequential CV pipeline for multiple flyvis configs.
#
# Submit to cluster (adjust walltime as needed):
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 36000 < run_cv_batch.sh
#
# Or run interactively:
#   bash run_cv_batch.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS=(
    "${REPO_DIR}/config/fly/flyvis_noise_free"
    "${REPO_DIR}/config/fly/flyvis_noise_005"
    "${REPO_DIR}/config/fly/flyvis_noise_05"
    "${REPO_DIR}/config/fly/flyvis_noise_005_null_edges_pc_400"
    "${REPO_DIR}/config/fly/flyvis_noise_005_removed_pc_20"
)
N_SEEDS=5

echo "============================================================"
echo "CV batch — $(date)"
echo "Repo: ${REPO_DIR}"
echo "Configs: ${#CONFIGS[@]}"
echo "Seeds per config: ${N_SEEDS}"
echo "============================================================"

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Starting CV: ${cfg}"
    echo "Start time: $(date)"
    echo "------------------------------------------------------------"
    python "${REPO_DIR}/GNN_Main.py" -o cv "${cfg}" --n_seeds "${N_SEEDS}"
    echo "Done: ${cfg}  ($(date))"
done

echo ""
echo "============================================================"
echo "All CV runs complete — $(date)"
echo "============================================================"
