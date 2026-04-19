"""Test script for Block 06 — f_theta msg curvature penalty.

Tests the three PASS conditions from the Phase-R hypothesis:
  (1) A randomly initialised f_theta MLP (3-layer tanh, input_size=4,
      hidden=64) exhibits measurable curvature: mean |curvature| > 1e-6
      at 1000 synthetic operating points.
  (2) Gradient norm w.r.t. f_theta parameters > 0 (differentiable).
  (3) All outputs finite (no NaN/Inf).

Runs on CPU. No GPU required.
"""

import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, '/workspace/connectome-gnn/src')

from connectome_gnn.LLM_code.staging.block_06.f_theta_msg_curvature import (
    f_theta_msg_curvature_loss,
)
from connectome_gnn.models.MLP import MLP


def main():
    torch.manual_seed(42)
    device = torch.device('cpu')

    # --- Build a mock model with a 3-layer tanh MLP as f_theta ---
    # in_features layout: [v(1), embedding(2), msg(1), excitation(0)] = input_size=4
    # (No excitation dim to keep it minimal; the test only cares about msg column)
    embedding_dim = 2
    input_size = 4  # v + embedding_dim + msg
    hidden_dim = 64
    n_layers = 3

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.f_theta = MLP(
                input_size=input_size,
                output_size=1,
                nlayers=n_layers,
                hidden_size=hidden_dim,
                activation='tanh',
                device=device,
            )

    model = MockModel()
    model.to(device)

    # --- Synthetic operating points ---
    N = 1000
    # Realistic range: v ~ N(0,1), embedding ~ N(0,0.3), msg ~ N(0,0.5)
    in_features = torch.randn(N, input_size, device=device)
    in_features[:, 0] *= 1.0        # v
    in_features[:, 1:1+embedding_dim] *= 0.3  # embedding
    in_features[:, embedding_dim + 1] *= 0.5  # msg

    ids_batch = torch.arange(N, device=device)
    delta_msg = 0.05

    # ===== TEST 1: Measurable curvature =====
    with torch.no_grad():
        loss_val = f_theta_msg_curvature_loss(
            model, in_features, ids_batch, embedding_dim, delta_msg
        )

    if not torch.isfinite(loss_val):
        print(f"FAIL: curvature loss is not finite: {loss_val.item()}")
        sys.exit(1)

    # Also compute mean absolute curvature directly for the threshold check
    msg_col = embedding_dim + 1
    with torch.no_grad():
        base = in_features.clone()
        pred_c = model.f_theta(base)

        in_p = in_features.clone()
        in_p[:, msg_col] += delta_msg
        pred_p = model.f_theta(in_p)

        in_m = in_features.clone()
        in_m[:, msg_col] -= delta_msg
        pred_m = model.f_theta(in_m)

        curvature = pred_p + pred_m - 2.0 * pred_c
        mean_abs_curvature = curvature.abs().mean().item()

    if mean_abs_curvature <= 1e-6:
        print(f"FAIL: mean |curvature| = {mean_abs_curvature:.2e} <= 1e-6 (no measurable curvature)")
        sys.exit(1)

    print(f"  [1/3] mean |curvature| = {mean_abs_curvature:.4e} > 1e-6  OK")

    # ===== TEST 2: Differentiable (gradient norm > 0) =====
    model.zero_grad()
    in_features_grad = in_features.clone().detach()  # fresh tensor
    loss_grad = f_theta_msg_curvature_loss(
        model, in_features_grad, ids_batch, embedding_dim, delta_msg
    )
    loss_grad.backward()

    total_grad_norm = 0.0
    for p in model.f_theta.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    if total_grad_norm <= 0:
        print(f"FAIL: gradient norm w.r.t. f_theta parameters = 0 (not differentiable)")
        sys.exit(1)

    print(f"  [2/3] grad norm = {total_grad_norm:.4e} > 0  OK")

    # ===== TEST 3: All outputs finite =====
    if not torch.isfinite(loss_grad):
        print(f"FAIL: loss is not finite after backward: {loss_grad.item()}")
        sys.exit(1)

    # Check with a range of delta values
    for delta in [0.01, 0.05, 0.1, 0.5]:
        with torch.no_grad():
            l = f_theta_msg_curvature_loss(
                model, in_features, ids_batch, embedding_dim, delta
            )
        if not torch.isfinite(l):
            print(f"FAIL: loss not finite at delta={delta}: {l.item()}")
            sys.exit(1)

    print(f"  [3/3] all outputs finite across delta=[0.01, 0.05, 0.1, 0.5]  OK")

    # ===== Additional: verify curvature -> 0 for a linear-only MLP =====
    class LinearMockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.f_theta = MLP(
                input_size=input_size,
                output_size=1,
                nlayers=2,  # 2-layer = single hidden layer
                hidden_size=hidden_dim,
                activation='none',  # linear activation => fully linear
                device=device,
            )

    linear_model = LinearMockModel()
    with torch.no_grad():
        linear_loss = f_theta_msg_curvature_loss(
            linear_model, in_features, ids_batch, embedding_dim, delta_msg
        )

    print(f"  [bonus] linear MLP curvature loss = {linear_loss.item():.2e} (expect ~0)")

    print(f"PASS: f_theta_msg_curvature_loss functional — curvature={mean_abs_curvature:.4e}, grad_norm={total_grad_norm:.4e}, all finite")


if __name__ == '__main__':
    main()
