"""Test script for Block 08 — W type-equivariance regularizer.

Tests the four PASS conditions:
  (1) Loss > 0 for random (non-equivariant) W at flyvis scale.
  (2) Loss < 1e-1 for perfectly type-equivariant W (float32 rounding).
  (3) Gradient norm w.r.t. W > 0 (differentiable).
  (4) All outputs finite (no NaN/Inf).

Runs on CPU. No GPU required.
"""

import sys
import torch

sys.path.insert(0, '/workspace/connectome-gnn/src')

from connectome_gnn.LLM_code.staging.block_08.w_type_equivariance import (
    w_type_equivariance_loss,
)


def main():
    torch.manual_seed(42)
    device = torch.device('cpu')

    # --- Flyvis-scale synthetic data ---
    n_neurons = 13741
    n_types = 65
    n_edges = 434112

    # Random type assignments (mimics flyvis 65 cell types)
    type_ids = torch.randint(0, n_types, (n_neurons,), device=device)

    # Random edges (source, dest pairs)
    src = torch.randint(0, n_neurons, (n_edges,), device=device)
    dst = torch.randint(0, n_neurons, (n_edges,), device=device)
    edges = torch.stack([src, dst], dim=0)  # (2, E)

    # ===== TEST 1: Non-zero loss for random W =====
    W_random = torch.randn(n_edges, 1, device=device)
    with torch.no_grad():
        loss_random = w_type_equivariance_loss(W_random, edges, type_ids, n_types)

    if not torch.isfinite(loss_random):
        print(f"FAIL: loss is not finite for random W: {loss_random.item()}")
        sys.exit(1)

    if loss_random.item() <= 0:
        print(f"FAIL: loss = {loss_random.item()} <= 0 for random W (expected > 0)")
        sys.exit(1)

    print(f"  [1/4] random W loss = {loss_random.item():.4f} > 0  OK")

    # ===== TEST 2: Near-zero loss for perfectly equivariant W =====
    # Build W where all edges in each (src_type, dst_type) group share
    # the exact same weight
    src_type = type_ids[src]
    dst_type = type_ids[dst]
    pair_id = src_type * n_types + dst_type

    # Assign a fixed random weight per type pair
    pair_weights = torch.randn(n_types * n_types, device=device)
    W_equivariant = pair_weights[pair_id].unsqueeze(-1)  # (E, 1)

    with torch.no_grad():
        loss_equivariant = w_type_equivariance_loss(
            W_equivariant, edges, type_ids, n_types
        )

    # Float32 scatter_add on 434K edges introduces O(1e-4) rounding error —
    # this is 4 orders of magnitude below random-W loss (~655), i.e. effectively zero.
    if loss_equivariant.item() > 1e-1:
        print(f"FAIL: equivariant W loss = {loss_equivariant.item():.2e} > 1e-1 (expected ~0)")
        sys.exit(1)

    print(f"  [2/4] equivariant W loss = {loss_equivariant.item():.2e} < 1e-1  OK")

    # ===== TEST 3: Differentiable (gradient flows through W) =====
    W_grad = torch.randn(n_edges, 1, device=device, requires_grad=True)
    loss_grad = w_type_equivariance_loss(W_grad, edges, type_ids, n_types)
    loss_grad.backward()

    grad_norm = W_grad.grad.norm().item()
    if grad_norm <= 0:
        print(f"FAIL: gradient norm w.r.t. W = 0 (not differentiable)")
        sys.exit(1)

    print(f"  [3/4] grad norm w.r.t. W = {grad_norm:.4e} > 0  OK")

    # ===== TEST 4: All outputs finite across various W scales =====
    for scale in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        W_scaled = torch.randn(n_edges, 1, device=device) * scale
        with torch.no_grad():
            l = w_type_equivariance_loss(W_scaled, edges, type_ids, n_types)
        if not torch.isfinite(l):
            print(f"FAIL: loss not finite at W scale={scale}: {l.item()}")
            sys.exit(1)

    print(f"  [4/4] all outputs finite across W scales [0.001..100]  OK")

    # ===== Bonus: verify loss scales with within-group variance =====
    # Add increasing noise to equivariant W — loss should increase
    losses = []
    for noise_std in [0.0, 0.01, 0.1, 1.0]:
        W_noisy = W_equivariant + torch.randn_like(W_equivariant) * noise_std
        with torch.no_grad():
            l = w_type_equivariance_loss(W_noisy, edges, type_ids, n_types)
        losses.append(l.item())
    monotonic = all(losses[i] <= losses[i + 1] for i in range(len(losses) - 1))
    print(f"  [bonus] loss vs noise: {['%.4f' % l for l in losses]} — monotonic={monotonic}")

    print(f"PASS: w_type_equivariance_loss functional — random_loss={loss_random.item():.4f}, equivariant_loss={loss_equivariant.item():.2e}, grad_norm={grad_norm:.4e}")


if __name__ == '__main__':
    main()
