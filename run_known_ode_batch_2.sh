#!/bin/bash

# Batch run of known_ode LLM explorations on cuda:0
# Run 2 known_ode configurations sequentially with iterations=84, --cluster, and --resume
# Device: cuda:0

python GNN_LLM.py -o generate_train_test_plot_Claude \
  --batch-configs \
    flyvis_noise_005_known_ode \
    flyvis_noise_005_010_known_ode \
  --batch-iterations 84 \
  --cluster \
  --resume \
  --skip-confirm \
  --device cuda:0
