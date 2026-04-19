"""Standalone test for g_phi_identity_loss (block 04).

Constructs a minimal model with g_phi MLP + embedding, exercises the loss
function, and checks the four PASS conditions:
  (1) Mean |g_phi(v_k) - v_k|^2 across k=10 voltages > 1e-4.
  (2) RMS at 8 interior points >= 50% of RMS at all 10 points.
  (3) Gradient norm w.r.t. g_phi parameters > 0.
  (4) All outputs finite (no NaN/Inf).

Usage:
    python test_g_phi_identity_loss.py
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
    xnorm = 0.5  # typical flyvis xnorm
    n_pts = 10

    model = _FakeModel(n_neurons=n_neurons, emb_dim=emb_dim).to(device)

    # Build in_features_edge: [voltage, embedding] -- same layout as
    # get_in_features_g_phi produces for flyvis_A
    voltage = torch.randn(n_neurons, 1, device=device) * 0.5
    in_features_edge = torch.cat([voltage, model.a.detach()], dim=1)  # (N, 1+emb_dim)

    ids = torch.arange(n_neurons, device=device)

    from connectome_gnn.LLM_code.staging.block_04.g_phi_identity_loss import (
        g_phi_identity_loss,
    )

    # --- Test both g_phi_positive=True and g_phi_positive=False ---
    for g_phi_positive in [True, False]:
        tag = f"g_phi_positive={g_phi_positive}"

        model.zero_grad()

        loss = g_phi_identity_loss(
            model=model,
            in_features_edge=in_features_edge,
            ids=ids,
            xnorm=xnorm,
            g_phi_positive=g_phi_positive,
            n_pts=n_pts,
        )

        # (4) Finite check
        if not torch.isfinite(loss):
            print(f"FAIL: loss is not finite ({loss.item()}) with {tag}")
            sys.exit(1)

        # (1) Nonzero output check -- the loss should be > 1e-4 for a random model
        loss_val = loss.item()
        if loss_val < 1e-4:
            print(f"FAIL: loss too small ({loss_val:.6e}) with {tag} "
                  f"-- g_phi already near identity, no degree of freedom to regularize")
            sys.exit(1)

        # (2) Interior vs all-points contribution check
        # Compute per-point deviations to verify interior points contribute
        v_grid = torch.linspace(0.0, 2.0 * xnorm, n_pts, device=device)
        per_point_rms = []
        for v_k in v_grid:
            inp = in_features_edge[ids].clone().detach()
            inp[:, 0] = v_k.item()
            out = model.g_phi(inp.float())
            if g_phi_positive:
                out = out ** 2
            deviation = ((out - v_k) ** 2).mean().item()
            per_point_rms.append(deviation)

        all_rms = (sum(per_point_rms) / len(per_point_rms)) ** 0.5
        # Interior = points 1 through n_pts-2 (exclude endpoints 0 and n_pts-1)
        interior_rms = (sum(per_point_rms[1:-1]) / (n_pts - 2)) ** 0.5

        if all_rms < 1e-8:
            print(f"FAIL: all-points RMS is effectively zero ({all_rms:.6e}) with {tag}")
            sys.exit(1)

        ratio = interior_rms / all_rms
        if ratio < 0.5:
            print(f"FAIL: interior RMS ratio {ratio:.4f} < 0.50 with {tag} "
                  f"-- interior points don't contribute enough signal "
                  f"(interior_rms={interior_rms:.6e}, all_rms={all_rms:.6e})")
            sys.exit(1)

        # (3) Gradient check
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
    print(f"PASS: g_phi_identity_loss -- loss={loss_val:.4f} > 1e-4, "
          f"interior/all ratio={ratio:.2f} >= 0.50, "
          f"grad_norm={grad_norm:.4f} > 0, all finite, both modes OK")


if __name__ == "__main__":
    main()
