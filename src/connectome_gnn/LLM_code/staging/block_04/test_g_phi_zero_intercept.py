"""Standalone test for g_phi_zero_intercept_loss (block 04).

Constructs a minimal model with g_phi MLP + embedding, exercises the loss
function, and checks the three PASS conditions:
  (1) g_phi(v=0, emb) has nonzero mean |output| (> 1e-4) for a randomly
      initialized model -- confirms the loss targets a real degree of freedom.
  (2) Loss gradient norm w.r.t. g_phi parameters > 0 -- differentiable.
  (3) All outputs finite (no NaN/Inf).

Usage:
    python test_g_phi_zero_intercept.py
    Last line: PASS: ... or FAIL: ...
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn


class _FakeModel(nn.Module):
    """Minimal model matching NeuralGNN's g_phi + embedding interface."""

    def __init__(self, n_neurons: int = 200, emb_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        input_size = 1 + emb_dim  # voltage + embedding (flyvis_A layout)
        self.g_phi = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Learnable node embedding
        self.a = nn.Parameter(torch.randn(n_neurons, emb_dim) * 0.1)

    def forward(self):
        raise NotImplementedError


def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    n_neurons = 200
    emb_dim = 2

    model = _FakeModel(n_neurons=n_neurons, emb_dim=emb_dim).to(device)

    # Build in_features_edge: [voltage, embedding] -- same layout as
    # get_in_features_g_phi produces for flyvis_A
    voltage = torch.randn(n_neurons, 1, device=device) * 0.5
    in_features_edge = torch.cat([voltage, model.a.detach()], dim=1)  # (N, 1+emb_dim)

    ids = torch.arange(n_neurons, device=device)

    # Import the function under test
    from connectome_gnn.LLM_code.staging.block_04.g_phi_zero_intercept import (
        g_phi_zero_intercept_loss,
    )

    # --- Test both g_phi_positive=True and g_phi_positive=False ---
    for g_phi_positive in [True, False]:
        tag = f"g_phi_positive={g_phi_positive}"

        # Zero gradients
        model.zero_grad()

        loss = g_phi_zero_intercept_loss(
            model=model,
            in_features_edge=in_features_edge,
            ids=ids,
            g_phi_positive=g_phi_positive,
        )

        # (3) Finite check
        if not torch.isfinite(loss):
            print(f"FAIL: loss is not finite ({loss.item()}) with {tag}")
            sys.exit(1)

        # (1) Nonzero output check -- the loss should be > 1e-4 for a random model
        # (if zero, the regularizer has nothing to push against)
        loss_val = loss.item()
        if loss_val < 1e-4:
            print(f"FAIL: loss too small ({loss_val:.6e}) with {tag} "
                  f"-- g_phi(v=0, emb) is already near zero, no degree of freedom to regularize")
            sys.exit(1)

        # (2) Gradient check
        loss.backward()
        grad_norm = sum(
            p.grad.norm().item() for p in model.g_phi.parameters() if p.grad is not None
        )
        if grad_norm <= 0:
            print(f"FAIL: gradient norm is zero with {tag} -- loss is not differentiable")
            sys.exit(1)

        # Check that gradients are finite
        for name, p in model.g_phi.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"FAIL: non-finite gradient in g_phi.{name} with {tag}")
                sys.exit(1)

    # All checks passed for both g_phi_positive modes
    print(f"PASS: g_phi_zero_intercept_loss -- nonzero output ({loss_val:.4f}), "
          f"grad_norm > 0 ({grad_norm:.4f}), all finite, both g_phi_positive modes OK")


if __name__ == "__main__":
    main()
