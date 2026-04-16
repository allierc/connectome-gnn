#!/bin/bash
# Controlled CV comparison across noise / connectivity conditions.
#
# All 5 conditions use IDENTICAL training hyperparameters (from
# flyvis_noise_005_null_edges_pc_400) and only differ in their data settings:
#
#   Condition                    noise_dyn  noise_meas  null_edges  edges
#   ─────────────────────────────────────────────────────────────────────────
#   flyvis_cmp_noise_free         0.0       0.0         none        434 112  (full graph)  DONE
#   flyvis_cmp_noise_005          0.05      0.0         none        434 112  (full graph)  DONE
#   flyvis_cmp_noise_05           0.5       0.0         none        434 112  (full graph)  DONE
#   null_edges_pc_400             0.05      0.0         +400%       434 112 + 1 736 448    DONE
#   flyvis_cmp_removed_pc_20      0.05      0.0         none        347 290  (20% removed) DONE
#   flyvis_noise_005_010_rc_winner 0.05     0.10        none        434 112  (meas noise, RC GNN)
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
LABELS=( "noise_free" "noise_005" "noise_05" "null_edges_pc_400" "removed_pc_20" "noise_005_010_rc" )
CONFIGS=(
    # "${CFG_DIR}/flyvis_cmp_noise_free"          # DONE
    # "${CFG_DIR}/flyvis_cmp_noise_005"            # DONE
    # "${CFG_DIR}/flyvis_cmp_noise_05"             # DONE
    # "${CFG_DIR}/flyvis_noise_005_null_edges_pc_400" # DONE
    # "${CFG_DIR}/flyvis_cmp_removed_pc_20"        # DONE
    # "${CFG_DIR}/flyvis_noise_005_010_rc_winner"             # DONE
    "${CFG_DIR}/flyvis_noise_005_010_winner"
)
# Base config names (for the comparison table — must match log/fly/<name>)
BASE_NAMES=(
    "flyvis_cmp_noise_free"
    "flyvis_cmp_noise_005"
    "flyvis_cmp_noise_05"
    "flyvis_noise_005_null_edges_pc_400"
    "flyvis_cmp_removed_pc_20"
    "flyvis_noise_005_010_rc_winner"
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
