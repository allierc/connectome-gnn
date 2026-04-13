#!/bin/bash
# Cross-noise robustness test for flyvis_noise_005_null_edges_pc_400.
#
# Purpose: zero-shot test the trained null_edges_pc_400 model on DAVIS data
# regenerated at three noise levels (noise_free, noise_005, noise_05).
# No retraining — purely evaluates whether the real-edge W weights learned
# with null-edge regularization transfer to standard (no null edge) data.
#
# Results written to:
#   {data_root}/log/fly/flyvis_noise_005_null_edges_pc_400/
#       results_test_on_noise_005_null_edges_pc_400_cross_noise_free.log
#       results_test_on_noise_005_null_edges_pc_400_cross_noise_005.log
#       results_test_on_noise_005_null_edges_pc_400_cross_noise_05.log
#
# This is DIFFERENT from run_cv_batch.sh which tests DAVIS→YouTube-VOS
# generalization. This script tests noise-level robustness.
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 3000 < run_cross_noise_batch.sh
#
# Or run interactively:
#   bash run_cross_noise_batch.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
CFG_DIR="${REPO_DIR}/config/fly"

# Model to evaluate (must already be trained)
MODEL_CONFIG="${CFG_DIR}/flyvis_noise_005_null_edges_pc_400"

# Cross-noise test configs (no null edges, different noise levels)
CROSS_CONFIGS=(
    "${CFG_DIR}/flyvis_noise_005_null_edges_pc_400_cross_noise_free"
    "${CFG_DIR}/flyvis_noise_005_null_edges_pc_400_cross_noise_005"
    "${CFG_DIR}/flyvis_noise_005_null_edges_pc_400_cross_noise_05"
)

echo "============================================================"
echo "Cross-noise robustness test — $(date)"
echo "Model: ${MODEL_CONFIG}"
echo "Cross configs: ${#CROSS_CONFIGS[@]}"
echo "============================================================"

for cross_cfg in "${CROSS_CONFIGS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Testing: $(basename ${cross_cfg})"
    echo "Start time: $(date)"
    echo "------------------------------------------------------------"
    python "${REPO_DIR}/GNN_Main.py" -o test "${MODEL_CONFIG}" best "${cross_cfg}"
    echo "Done: $(basename ${cross_cfg})  ($(date))"
done

echo ""
echo "============================================================"
echo "All cross-noise tests complete — $(date)"
echo "============================================================"
