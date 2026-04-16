#!/bin/bash
# CV pipeline for partial-data configs (1/5 frames + 10% hidden neurons).
#
# Covers the three "partial data NEW" results in the paper (Table 1).
# Trains/evaluates GNN under two partial-data conditions:
#   1. Temporal stride-5 (observe every 5th frame, BPTT through 5 steps)
#   2. 10% of non-retinal neurons hidden (voltage unobserved)
#
#   Config                                       Type        Key result
#   ─────────────────────────────────────────────────────────────────────────────
#   flyvis_noise_005_stride_5_winner             DAVIS/GNN   conn_R2=0.387±0.051 (8-seed)
#   flyvis_noise_005_stride_5_yt_winner          YouTube/GNN conn_R2~0.38 (ported DAVIS HPs)
#   flyvis_noise_005_hidden_010_ngp_winner       DAVIS/GNN   conn_R2~0.82 (NGP hidden INR)
#   flyvis_noise_005_hidden_010_siren_winner     DAVIS/GNN   conn_R2~0.77 (SIREN — neg. result)
#
# Notes:
#   - stride_5 hard ceiling: spectral radius ~1.72 limits conn_R2 ≤ 0.43 regardless of HPs
#   - stride_5_yt: no dedicated LLM exploration; DAVIS winner HPs ported to YouTube-VOS
#   - hidden_010_siren: NEGATIVE RESULT — hidden_nnr_R2 never > 0 across 128 iterations
#   - hidden_010_ngp: NEGATIVE RESULT — hidden_nnr_R2 never > 0 across 128 iterations
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is < run_partial_data_cv_batch.sh
#
# Or run interactively:
#   bash run_partial_data_cv_batch.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
N_SEEDS=5

CONFIGS=(
    "${CFG_DIR}/flyvis_noise_005_stride_5_winner"
    "${CFG_DIR}/flyvis_noise_005_stride_5_yt_winner"
    "${CFG_DIR}/flyvis_noise_005_hidden_010_ngp_winner"
    "${CFG_DIR}/flyvis_noise_005_hidden_010_siren_winner"
)

echo "============================================================"
echo "Partial data CV batch — $(date)"
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
echo "All partial data CV runs complete — $(date)"
echo "============================================================"
