#!/bin/bash

# Batch run of known_ode LLM explorations on cuda:1
# Run 2 known_ode configurations sequentially with iterations=84, --cluster, and --resume
# Device: cuda:1

python GNN_LLM.py -o generate_train_test_plot_Claude \
  --batch-configs \
    flyvis_noise_005_removed_pc_20_known_ode
  --batch-iterations 64 \
  --cluster \
  --resume \
  --skip-confirm \
  --device cuda:1
