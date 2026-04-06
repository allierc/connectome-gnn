#!/bin/bash

# Comprehensive batch run of all larva LLM explorations
# Includes GNN variants (clean, noise_005, noise_05) and known_ode variants
# Runs on cuda:1 with clustering and resume capability

echo "=========================================="
echo "Larva LLM Exploration Suite"
echo "=========================================="
echo "Running all 12 larva configurations"
echo "Device: cuda:1"
echo "Clustering enabled, Resume enabled"
echo ""

# ============================================
# PHASE 1: GNN Base Variants (3 configs)
# ============================================
echo "PHASE 1: GNN Base Variants"
echo "============================"

python GNN_LLM.py -o generate_train_test_plot_Claude \
  --batch-configs \
    larva_noise_free \
    larva_noise_005 \
    larva_noise_05 \
  --batch-iterations 84 \
  --cluster \
  --resume \
  --skip-confirm \
  --device cuda:1

echo ""
echo "✓ PHASE 1 Complete"
echo ""

# ============================================
# PHASE 2: GNN GT Edges Variants (3 configs)
# ============================================
echo "PHASE 2: GNN GT Edges Variants"
echo "==============================="

python GNN_LLM.py -o generate_train_test_plot_Claude \
  --batch-configs \
    larva_gt_edges_noise_free \
    larva_gt_edges_noise_005 \
    larva_gt_edges_noise_05 \
  --batch-iterations 84 \
  --cluster \
  --resume \
  --skip-confirm \
  --device cuda:1

echo ""
echo "✓ PHASE 2 Complete"
echo ""

# ============================================
# PHASE 3: Known ODE Base Variants (3 configs)
# ============================================
echo "PHASE 3: Known ODE Base Variants"
echo "=================================="

python GNN_LLM.py -o generate_train_test_plot_Claude \
  --batch-configs \
    larva_known_ode_noise_free \
    larva_known_ode_noise_005 \
    larva_known_ode_noise_05 \
  --batch-iterations 84 \
  --cluster \
  --resume \
  --skip-confirm \
  --device cuda:1

echo ""
echo "✓ PHASE 3 Complete"
echo ""

# ============================================
# PHASE 4: Known ODE GT Edges Variants (3 configs)
# ============================================
echo "PHASE 4: Known ODE GT Edges Variants"
echo "====================================="

python GNN_LLM.py -o generate_train_test_plot_Claude \
  --batch-configs \
    larva_known_ode_gt_edges_noise_free \
    larva_known_ode_gt_edges_noise_005 \
    larva_known_ode_gt_edges_noise_05 \
  --batch-iterations 84 \
  --cluster \
  --resume \
  --skip-confirm \
  --device cuda:1

echo ""
echo "✓ PHASE 4 Complete"
echo ""

# ============================================
# Summary
# ============================================
echo "=========================================="
echo "✓✓✓ All Larva Explorations Complete!"
echo "=========================================="
echo ""
echo "Summary of executed configurations:"
echo "  • GNN Base: 3 variants (clean, 0.5%, 5% noise)"
echo "  • GNN GT Edges: 3 variants (clean, 0.5%, 5% noise)"
echo "  • Known ODE Base: 3 variants (clean, 0.5%, 5% noise)"
echo "  • Known ODE GT Edges: 3 variants (clean, 0.5%, 5% noise)"
echo "Total: 12 configurations, 1,008 iterations (84 per config)"
echo ""
echo "Results available in:"
echo "  • log/Claude_exploration/LLM_larva*/"
echo "  • graphs_data/larva/"
echo ""
