#!/bin/bash
# Sequential CV pipeline for the three flyvis LLM-exploration winner configs.
#
# Submit to cluster (adjust walltime as needed):
#   bsub -n 2 -gpu "num=1" -q gpu_a100 -W 600 -Is  < run_winner_cv_batch.sh
#
# Or run interactively:
#   bash run_winner_cv_batch.sh

set -euo pipefail

# Hardcoded cluster path — avoids LS_SUBCWD/BASH_SOURCE issues when LSF
# copies the script to ~/.lsbatch/ or bsub is run from a different directory.
REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CONFIGS=(
    # "${REPO_DIR}/config/fly/flyvis_noise_free_winner"
    "${REPO_DIR}/config/fly/flyvis_noise_005_winner"
    "${REPO_DIR}/config/fly/flyvis_noise_05_winner"
)
N_SEEDS=5

echo "============================================================"
echo "Winner CV batch — $(date)"
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
    python "${REPO_DIR}/GNN_Main.py" -o cv "${cfg}" --n_seeds "${N_SEEDS}" --output_root "${DATA_ROOT}"
    echo "Done: ${cfg}  ($(date))"
done

echo ""
echo "============================================================"
echo "All winner CV runs complete — $(date)"
echo "============================================================"
