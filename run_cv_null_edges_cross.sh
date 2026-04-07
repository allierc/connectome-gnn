#!/bin/bash

# Cross-dataset CV: train on flyvis_noise_005_null_edges_pc_400, test on 4 datasets
# Results: log/fly/<config_name>/results/cv_summary.txt  +  cv_barplot.png
# Each config runs 5 seeds (42..46) sequentially on one GPU node.
#
# Submit all 4 in parallel:
#   bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -o logs/cv_cross.out -e logs/cv_cross.err bash run_cv_null_edges_cross.sh

OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData
N_SEEDS=5

configs=(
    flyvis_noise_005_null_edges_pc_400_cross
    flyvis_noise_005_null_edges_pc_400_cross_removed_pc_20
    flyvis_noise_005_null_edges_pc_400_cross_noise_05
    flyvis_noise_005_null_edges_pc_400_cross_noise_free
)

for cfg in "${configs[@]}"; do
    echo ""
    echo "=============================="
    echo "CV: $cfg"
    echo "=============================="
    python GNN_Main.py -o cv "$cfg" --n_seeds "$N_SEEDS" --output_root "$OUTPUT_ROOT"
done
