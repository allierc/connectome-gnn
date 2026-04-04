#!/bin/bash

# Batch run of known_ode LLM explorations
# Run 6 known_ode configurations sequentially with iterations=84, --cluster, and --resume

python GNN_LLM.py -o generate_train_test_plot_Claude \
  --batch-configs \
    flyvis_noise_005_removed_pc_20_known_ode \
    flyvis_noise_005_null_edges_pc_400_known_ode \
  --batch-iterations 84 \
  --cluster \
  --resume \
  --skip-confirm
