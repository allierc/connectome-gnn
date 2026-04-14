#!/bin/bash
# Controlled CV comparison across noise / connectivity conditions.
#
# All 5 conditions use IDENTICAL training hyperparameters (from
# flyvis_noise_005_null_edges_pc_400) and only differ in their data settings:
#
#   Condition               noise   null_edges  edges
#   ─────────────────────────────────────────────────────────────
#   flyvis_cmp_noise_free    0.0    none        434 112  (full graph)
#   flyvis_cmp_noise_005     0.05   none        434 112  (full graph)
#   flyvis_cmp_noise_05      0.5    none        434 112  (full graph)
#   null_edges_pc_400        0.05   +400%       434 112 + 1 736 448
#   flyvis_cmp_removed_pc_20  0.05  none        347 290  (20% removed)
#
# Phase 2 (zero-shot DAVIS→YouTube test) runs automatically when a pre-trained
# DAVIS model is found; it is skipped gracefully otherwise (e.g. cmp_* configs
# that have not yet been trained on DAVIS data).
#
# At the end a comparison table is printed and saved to
#   {DATA_ROOT}/log/cv_comparison_table.txt
#
# Submit to cluster:
#   bsub -n 2 -gpu "num=1" -q gpu_a100 -W 600 -Is < run_cross_noise_batch.sh
#
# Or run interactively:
#   bash run_cross_noise_batch.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
N_SEEDS=5

# Parallel arrays: display label | absolute config path (no .yaml)
LABELS=( "noise_free" "noise_005" "noise_05" "null_edges_pc_400" "removed_pc_20" )
CONFIGS=(
    # "${CFG_DIR}/flyvis_cmp_noise_free"
    # "${CFG_DIR}/flyvis_cmp_noise_005"
    # "${CFG_DIR}/flyvis_cmp_noise_05"
    # "${CFG_DIR}/flyvis_noise_005_null_edges_pc_400"
    "${CFG_DIR}/flyvis_cmp_removed_pc_20"
)
# Base config names (for the comparison table — must match log/fly/<name>)
BASE_NAMES=(
    "flyvis_cmp_noise_free"
    "flyvis_cmp_noise_005"
    "flyvis_cmp_noise_05"
    "flyvis_noise_005_null_edges_pc_400"
    "flyvis_cmp_removed_pc_20"
)

echo "============================================================"
echo "CV comparison — $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Conditions: ${#CONFIGS[@]}   Seeds per condition: ${N_SEEDS}"
echo "============================================================"

for i in "${!CONFIGS[@]}"; do
    label="${LABELS[$i]}"
    cfg="${CONFIGS[$i]}"
    echo ""
    echo "------------------------------------------------------------"
    echo "Condition: ${label}   ($(basename ${cfg}))"
    echo "Start: $(date)"
    echo "------------------------------------------------------------"
    python "${REPO_DIR}/GNN_Main.py" \
        -o cv "${cfg}" \
        --n_seeds "${N_SEEDS}" \
        --output_root "${DATA_ROOT}"
    echo "Done: ${label}  ($(date))"
done

echo ""
echo "============================================================"
echo "All CV runs complete — printing comparison table"
echo "============================================================"

python "${REPO_DIR}/scripts/print_cv_comparison.py" \
    --output_root "${DATA_ROOT}" \
    --labels  "${LABELS[@]}" \
    --configs "${BASE_NAMES[@]}"

echo ""
echo "============================================================"
echo "Done — $(date)"
echo "============================================================"
