#!/bin/bash
# Sequential CV pipeline for multiple flyvis configs.
#
# Submit to cluster (adjust walltime as needed):
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 36000 < run_cv_batch.sh
#
# Or run interactively:
#   bash run_cv_batch.sh

set -euo pipefail

# When submitted via "bsub < run_cv_batch.sh", LSF copies the script to
# ~/.lsbatch/ and runs it there, so BASH_SOURCE[0] points to the wrong place.
# LS_SUBCWD is set by LSF to the directory where bsub was invoked — use that
# as the primary source, with a fallback for direct bash execution.
REPO_DIR="${LS_SUBCWD:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
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
