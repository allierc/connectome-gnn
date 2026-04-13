#!/bin/bash
# Causal invariance cross-tests: DAVIS-trained models tested on structurally
# perturbed simulations (edges removed), WITHOUT any retraining.
#
# Scientific question:
#   Can a GNN trained on the original full network predict the dynamics of a
#   circuit with 50% of its connections removed? If rollout Pearson r stays
#   high, the GNN captured mechanistic causal relationships — not just
#   statistical regularities of its training data.
#
# Experiment design:
#   Source model:  flyvis_noise_005_winner (DAVIS-trained, multiple seeds)
#   Ablation data: flyvis_noise_005_removed_pc_50 (50% edges removed, new sim)
#   Metric:        rollout Pearson r  (written to results_rollout_on_*.log)
#
# The -o test <config> best <test_config> command:
#   1. Loads best trained model from log/fly/<config>/models/best_model_*.pt
#   2. Generates test data from <test_config> (if not already present)
#   3. Runs zero-noise rollout and writes results_rollout_on_<test>.log
#   No retraining occurs.
#
# Reference (CoSyne 2026, old model, 50% pruning):
#   RMSE = 0.15 ± 0.23,  Pearson r = 0.91 ± 0.20  (± over neurons, single seed)
#
# Submit to cluster:
#   bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is < run_causal_invariance_batch.sh
#
# Or run interactively:
#   bash run_causal_invariance_batch.sh

set -euo pipefail

REPO_DIR="/groups/saalfeld/home/allierc/Graph/connectome-gnn"
DATA_ROOT="/groups/saalfeld/home/allierc/GraphData"
CFG_DIR="${REPO_DIR}/config/fly"

# Winner config used as training base (multiple CV seeds already trained by run_cv_batch.sh)
TRAIN_CONFIG="${CFG_DIR}/flyvis_noise_005_winner"

# Ablation test targets (trained model tested on each without retraining)
ABLATION_CONFIGS=(
    "${CFG_DIR}/flyvis_noise_005_removed_pc_50"   # causal invariance — 50% pruned
    "${CFG_DIR}/flyvis_noise_005_removed_pc_20"   # lighter ablation — 20% pruned
)

# Seeds must match what was used in run_cv_batch.sh for flyvis_noise_005_winner
N_SEEDS=5
SEEDS=($(seq 0 $((N_SEEDS - 1))))

echo "============================================================"
echo "Causal invariance cross-tests — $(date)"
echo "Train config: $(basename ${TRAIN_CONFIG})"
echo "Ablation targets: ${#ABLATION_CONFIGS[@]}"
echo "Seeds: ${N_SEEDS} (cv00 .. cv$(printf '%02d' $((N_SEEDS-1))))"
echo "============================================================"

for abl_cfg in "${ABLATION_CONFIGS[@]}"; do
    abl_name=$(basename "${abl_cfg}")
    echo ""
    echo "------------------------------------------------------------"
    echo "Ablation target: ${abl_name}"
    echo "Start: $(date)"
    echo "------------------------------------------------------------"

    for i in "${SEEDS[@]}"; do
        fold=$(printf "cv%02d" "${i}")
        fold_config="${TRAIN_CONFIG}_${fold}"
        echo "  Testing fold=${fold} on ${abl_name} ..."
        python "${REPO_DIR}/GNN_Main.py" \
            -o test "${fold_config}" best "${abl_cfg}" \
            --output_root "${DATA_ROOT}"
        echo "  Done fold=${fold}  ($(date))"
    done

    echo "All seeds done for ${abl_name}"
done

echo ""
echo "============================================================"
echo "All causal invariance tests complete — $(date)"
echo ""
echo "Results are in:"
echo "  ${DATA_ROOT}/log/fly/<fold_name>/results_rollout_on_*.log"
echo ""
echo "To collect summary statistics across seeds:"
echo "  grep -h 'Pearson r' \\"
echo "    ${DATA_ROOT}/log/fly/flyvis_noise_005_winner_cv*/results_rollout_on_*removed_pc_50*.log"
echo "============================================================"
