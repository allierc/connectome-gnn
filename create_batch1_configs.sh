#!/bin/bash

# Read base config into variable
BASE=$(cat config/fly/flyvis_noise_005_removed_pc_10_known_ode.yaml)

# Slot 0: baseline (seeds: 1000, 1500)
echo "$BASE" | sed 's/seed: 42$/seed: 1000/' | sed 's/seed: 43$/seed: 1500/' > config/fly/flyvis_noise_005_removed_pc_10_known_ode_Claude_00.yaml
echo "Created Slot 0 (baseline)"

# Slot 1: lr_W=0.0001 (seeds: 2001, 2501)
echo "$BASE" | sed 's/seed: 42$/seed: 2001/' | sed 's/seed: 43$/seed: 2501/' | sed 's/lr_W: 0.0009/lr_W: 0.0001/' > config/fly/flyvis_noise_005_removed_pc_10_known_ode_Claude_01.yaml
echo "Created Slot 1 (lr_W=0.0001)"

# Slot 2: lr_W=0.0005 (seeds: 3002, 3502)
echo "$BASE" | sed 's/seed: 42$/seed: 3002/' | sed 's/seed: 43$/seed: 3502/' | sed 's/lr_W: 0.0009/lr_W: 0.0005/' > config/fly/flyvis_noise_005_removed_pc_10_known_ode_Claude_02.yaml
echo "Created Slot 2 (lr_W=0.0005)"

# Slot 3: lr=0.005 (seeds: 4003, 4503)
echo "$BASE" | sed 's/seed: 42$/seed: 4003/' | sed 's/seed: 43$/seed: 4503/' | sed 's/lr: 0.0018/lr: 0.005/' > config/fly/flyvis_noise_005_removed_pc_10_known_ode_Claude_03.yaml
echo "Created Slot 3 (lr=0.005)"

echo "All 4 configs created successfully"
