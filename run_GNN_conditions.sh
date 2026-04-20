#!/bin/bash
# GNN cross-check with PER-CONDITION agentic HPs.
#
# For each of 8 conditions, trains a fresh GNN on YouTube-VOS data using
# THAT condition's own `_winner.yaml` hyperparameters — including the
# recent `flyvis_noise_005_null_edges_pc_400_winner.yaml` for the
# "$+400\%$ null edges" row — then cross-rolls it out on DAVIS held-out
# test data. The 6-column table reports:
#
#   prediction columns (one-step r, rollout r) : YT-trained model rolled
#       out on DAVIS held-out test data.
#   parameter-recovery columns (W, tau, V_rest, cluster) : YT-trained
#       model's own learned parameters (from results/metrics.txt).
#
# Caching (per condition):
#   data_generate skipped if <graphs_data>/x_list_train exists
#   data_train    skipped if models/best_model_with_*.pt    exists
#   data_test     skipped if results_rollout_on_<base>.log  exists
#   data_plot     skipped if results/metrics.txt            exists
#
# Use  --force_test  when resubmitting to redo only the test + plot steps
# (generate + train remain cached). That's the idiom for "I don't trust the
# current test results, redo them" without losing the trained models.
#
# Output:
#   <DATA_ROOT>/log/cv_yt_per_cond_rows.tex
#
# Submit to cluster:
#   bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is < run_GNN_conditions.sh
#
# Or run interactively:
#   bash run_GNN_conditions.sh                   # cache-respecting
#   bash run_GNN_conditions.sh --force_test       # redo tests + plots

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
SUFFIX="yt_per_cond"

# Forward an optional --force_test passed on the command line.
FORCE_FLAG=""
if [ "${1:-}" = "--force_test" ]; then
    FORCE_FLAG="--force_test"
fi

echo "============================================================"
echo "GNN conditions (per-condition winner HPs) — $(date)"
echo "Repo:      ${REPO_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Suffix:    ${SUFFIX}"
echo "Force:     ${FORCE_FLAG:-<none>}"
echo "============================================================"

# Step 1: emit 5 per-fold YT training YAMLs per condition, per-condition
# HPs (+400% null-edges row picks up flyvis_noise_005_null_edges_pc_400_winner).
python "${REPO_DIR}/scripts/write_cross_yt_configs.py" \
    --hp_source per_condition \
    --suffix "${SUFFIX}" \
    --n_folds 5

# Step 2: orchestrate parallel cluster training (5 bsubs per condition,
# wave-waited before next condition). Local: data_generate, data_test
# (YT-model -> DAVIS held-out), data_plot. TeX re-emitted after each
# condition finishes.
python "${REPO_DIR}/scripts/run_cross_yt_parallel.py" \
    --suffix "${SUFFIX}" \
    --output_root "${DATA_ROOT}" \
    --n_folds 5 \
    --emit_tex cv_yt_per_cond_rows.tex \
    ${FORCE_FLAG}

# Final emit (also written after each condition by --emit_tex above).
python "${REPO_DIR}/scripts/emit_cross_table_rows.py" \
    --suffix "${SUFFIX}" \
    --output_tex cv_yt_per_cond_rows.tex \
    --output_root "${DATA_ROOT}" \
    --n_folds 5

echo ""
echo "============================================================"
echo "GNN conditions complete — $(date)"
echo "============================================================"
