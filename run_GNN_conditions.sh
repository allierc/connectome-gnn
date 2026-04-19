#!/bin/bash
# Full CV pipeline for the 8 conditions of tab:cv_per_condition.
#
# Replaces: run_cv_batch.sh, run_partial_data_cv_batch.sh,
#           run_measurement_cross_batch.sh, run_measurement_noise_cv_batch.sh,
#           run_winner_cv_batch.sh.
#
# For each condition, runs generate / train / test / plot / CV via
# `-o cv` — which auto-skips Phase 1 YT data generation when the data is
# already present and auto-trains the DAVIS base model only if it is
# missing. Conditions whose cv_summary.txt is already present are skipped
# entirely (no retrain).
#
# Table rows produced (labels map to configs below):
#
#   Condition                  sigma  gamma   edges       config
#   ──────────────────────────────────────────────────────────────────────────
#   noise-free                 0      0       434 112     flyvis_noise_free_winner
#   low intrinsic noise        0.05   0       434 112     flyvis_noise_005_winner
#   high intrinsic noise       0.5    0       434 112     flyvis_noise_05_winner
#   low meas. noise            0.05   0.1     434 112     flyvis_noise_005_010_winner
#   +400% null edges           0.05   0       2 170 560   flyvis_noise_005_null_edges_pc_400
#   -20% edges removed         0.05   0       347 290     flyvis_noise_005_removed_pc_20_winner
#   1/5 frames                 0.05   0       434 112     flyvis_noise_005_stride_5_winner
#   10% hidden                 0.05   0       434 112     flyvis_noise_005_hidden_010_ngp_winner
#
# Prediction columns (one-step r, rollout r) come from yt_one_step_r /
# yt_rollout_r (DAVIS-trained model tested on held-out YouTube-VOS folds).
# Parameter recovery (W, tau, V_rest, cluster) is mean±SD over 5 YT-retrained
# folds per condition.
#
# Output:
#   TeX rows for tab:cv_per_condition are written to
#     <DATA_ROOT>/log/cv_per_condition_rows.tex
#   (and also echoed to stdout at the end of the run).
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is < run_GNN_conditions.sh
#
# Or run interactively:
#   bash run_GNN_conditions.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"
N_SEEDS=5

CONFIGS=(
    "flyvis_noise_free_winner"
    "flyvis_noise_005_winner"
    "flyvis_noise_05_winner"
    "flyvis_noise_005_010_winner"
    "flyvis_noise_005_null_edges_pc_400"
    "flyvis_noise_005_removed_pc_20_winner"
    "flyvis_noise_005_stride_5_winner"
    "flyvis_noise_005_hidden_010_ngp_winner"
)

echo "============================================================"
echo "GNN per-condition CV — $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Configs:   ${#CONFIGS[@]}   Seeds per config: ${N_SEEDS}"
echo "============================================================"

for cfg in "${CONFIGS[@]}"; do
    SUMMARY="${DATA_ROOT}/log/fly/${cfg}/results/cv_summary.txt"
    echo ""
    echo "------------------------------------------------------------"
    echo "Condition: ${cfg}"
    echo "------------------------------------------------------------"
    if [ -f "${SUMMARY}" ]; then
        echo "[skip] cv_summary.txt already exists at ${SUMMARY}"
        continue
    fi
    echo "[run ] generate + train + test + plot + CV via -o cv"
    python "${REPO_DIR}/GNN_Main.py" \
        -o cv "${CFG_DIR}/${cfg}" \
        --n_seeds "${N_SEEDS}" \
        --output_root "${DATA_ROOT}"
    echo "Done: ${cfg}  ($(date))"
done

echo ""
echo "============================================================"
echo "Emitting TeX rows for tab:cv_per_condition — $(date)"
echo "============================================================"
python "${REPO_DIR}/scripts/emit_conditions_table_rows.py" \
    --output_root "${DATA_ROOT}"

echo ""
echo "============================================================"
echo "GNN per-condition CV complete — $(date)"
echo "============================================================"
