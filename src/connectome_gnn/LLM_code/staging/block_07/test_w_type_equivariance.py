"""Test script for Block 07 — W type-equivariance regularizer.

Tests the four PASS conditions from the hypothesis:
  (1) Random W with 500 edges, 5 types: loss > 0.
  (2) Block-constant W (same value per type pair): loss < 1e-10.
  (3) Gradient norm w.r.t. W > 0 (differentiable).
  (4) All outputs finite (no NaN/Inf).

Runs on CPU. No GPU required.
"""

import sys
import torch

sys.path.insert(0, '/workspace/connectome-gnn/src')

from connectome_gnn.LLM_code.staging.block_07.w_type_equivariance import (
    w_type_equivariance_loss,
)


def main():
    torch.manual_seed(42)
    device = torch.device('cpu')

    # --- Setup: synthetic graph with known structure ---
    n_neurons = 50
    n_types = 5
    n_edges = 500

    # Assign random types to neurons
    type_ids = torch.randint(0, n_types, (n_neurons,), device=device)

    # Random edges (source, dest)
    edge_src = torch.randint(0, n_neurons, (n_edges,), device=device)
    edge_dst = torch.randint(0, n_neurons, (n_edges,), device=device)
    edges = torch.stack([edge_src, edge_dst], dim=0)

    # ===== TEST 1: Random W => loss > 0 =====
    W_random = torch.randn(n_edges, 1, device=device, requires_grad=False)

    with torch.no_grad():
        loss_random = w_type_equivariance_loss(W_random, edges, type_ids, n_types)

    if not torch.isfinite(loss_random):
        print(f"FAIL: loss is not finite for random W: {loss_random.item()}")
        sys.exit(1)

    if loss_random.item() <= 0:
        print(f"FAIL: random W should have loss > 0, got {loss_random.item()}")
        sys.exit(1)

    print(f"  [1/4] random W loss = {loss_random.item():.6f} > 0  OK")

    # ===== TEST 2: Block-constant W => loss ~ 0 =====
    # For each edge, set W to a value that depends only on (src_type, dst_type)
    src_types = type_ids[edge_src]
    dst_types = type_ids[edge_dst]
    pair_ids = src_types * n_types + dst_types

    # Assign a deterministic value per type pair
    torch.manual_seed(123)
    pair_values = torch.randn(n_types * n_types, device=device)
    W_block = pair_values[pair_ids].unsqueeze(-1)  # (E, 1)

    with torch.no_grad():
        loss_block = w_type_equivariance_loss(W_block, edges, type_ids, n_types)

    if not torch.isfinite(loss_block):
        print(f"FAIL: loss is not finite for block-constant W: {loss_block.item()}")
        sys.exit(1)

    if loss_block.item() > 1e-10:
        print(f"FAIL: block-constant W should have loss < 1e-10, got {loss_block.item():.2e}")
        sys.exit(1)

    print(f"  [2/4] block-constant W loss = {loss_block.item():.2e} < 1e-10  OK")

    # ===== TEST 3: Gradient flows through W =====
    W_grad = torch.randn(n_edges, 1, device=device, requires_grad=True)
    loss_grad = w_type_equivariance_loss(W_grad, edges, type_ids, n_types)
    loss_grad.backward()

    grad_norm = W_grad.grad.norm().item()
    if grad_norm <= 0:
        print(f"FAIL: gradient norm w.r.t. W = 0 (not differentiable)")
        sys.exit(1)

    print(f"  [3/4] grad norm = {grad_norm:.4e} > 0  OK")

    # ===== TEST 4: All outputs finite across scales =====
    for scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
        W_test = torch.randn(n_edges, 1, device=device) * scale
        with torch.no_grad():
            l = w_type_equivariance_loss(W_test, edges, type_ids, n_types)
        if not torch.isfinite(l):
            print(f"FAIL: loss not finite at scale={scale}: {l.item()}")
            sys.exit(1)

    print(f"  [4/4] all outputs finite across W scales [0.01..100]  OK")

    # ===== Bonus: scaling test — loss scales as variance =====
    # If all W are shifted by a constant, within-group variance unchanged
    W_base = torch.randn(n_edges, 1, device=device)
    with torch.no_grad():
        loss_base = w_type_equivariance_loss(W_base, edges, type_ids, n_types)
        # Shift all W by 10 — within-pair variance should not change
        loss_shifted = w_type_equivariance_loss(W_base + 10.0, edges, type_ids, n_types)

    # Not exactly equal due to the shift affecting pair means differently,
    # but the within-pair deviations should be identical
    if abs(loss_base.item() - loss_shifted.item()) > 1e-5:
        print(f"  [bonus] WARNING: shift invariance not exact: base={loss_base.item():.6f}, shifted={loss_shifted.item():.6f}")
    else:
        print(f"  [bonus] shift-invariant: base={loss_base.item():.6f}, shifted={loss_shifted.item():.6f}  OK")

    # ===== Bonus: realistic flyvis scale =====
    n_neurons_fly = 13741
    n_types_fly = 65
    n_edges_fly = 434112
    type_ids_fly = torch.randint(0, n_types_fly, (n_neurons_fly,), device=device)
    edge_src_fly = torch.randint(0, n_neurons_fly, (n_edges_fly,), device=device)
    edge_dst_fly = torch.randint(0, n_neurons_fly, (n_edges_fly,), device=device)
    edges_fly = torch.stack([edge_src_fly, edge_dst_fly], dim=0)
    W_fly = torch.randn(n_edges_fly, 1, device=device)

    with torch.no_grad():
        loss_fly = w_type_equivariance_loss(W_fly, edges_fly, type_ids_fly, n_types_fly)

    if not torch.isfinite(loss_fly):
        print(f"FAIL: loss not finite at flyvis scale (434K edges, 65 types)")
        sys.exit(1)

    print(f"  [bonus] flyvis-scale (434K edges, 65 types) loss = {loss_fly.item():.6f}  OK")

    print(f"PASS: w_type_equivariance_loss functional — random_loss={loss_random.item():.4f}, block_loss={loss_block.item():.2e}, grad_norm={grad_norm:.4e}, all finite")


if __name__ == '__main__':
    main()
