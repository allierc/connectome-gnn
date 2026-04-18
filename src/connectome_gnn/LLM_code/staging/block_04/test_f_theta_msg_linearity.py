#!/usr/bin/env python
"""Phase-S test for f_theta_msg_linearity (block 04 — best-of-combination).

Creates a random f_theta MLP and verifies that the msg linearity loss
exhibits measurable nonlinearity, produces gradients, and stays finite.
Uses real flyvis voltage data to derive realistic per-neuron mu/sigma.

PASS CONDITIONS (all must hold):
  (1) Random MLP msg residual_MSE / output_variance > 0.01
  (2) Gradient norm w.r.t. f_theta params > 0
  (3) All values finite (no NaN/Inf)
"""

from __future__ import annotations

import sys
import types


def main():
    import numpy as np
    import torch
    import torch.nn as nn

    from connectome_gnn.LLM_code.scratchpad import load_full_voltage
    from connectome_gnn.LLM_code.staging.block_04.f_theta_msg_linearity import (
        f_theta_msg_linearity_loss,
    )

    # ================================================================ #
    #  Load real voltage data for realistic mu/sigma
    # ================================================================ #
    print("Loading flyvis voltage data (gamma=0.10)...")
    v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", 0.10)
    print(f"  v_clean shape: {v_clean.shape}, v_noisy shape: {v_noisy.shape}")

    # Compute per-neuron stats from clean voltage (full population)
    N_total = v_clean.shape[1]
    mu_full = v_clean.mean(dim=0).numpy().astype(np.float32)      # (N_total,)
    sigma_full = v_clean.std(dim=0).numpy().astype(np.float32)    # (N_total,)

    # Sample 200 neurons with diverse stats (stride across the population)
    N = 200
    rng = np.random.default_rng(42)
    idx = rng.choice(N_total, size=N, replace=False)
    mu = mu_full[idx]
    sigma = sigma_full[idx]
    print(f"  using N={N} neurons (sampled), mu range=[{mu.min():.3f}, {mu.max():.3f}], "
          f"sigma range=[{sigma.min():.4f}, {sigma.max():.4f}]")

    # ================================================================ #
    #  Build a minimal dummy model with f_theta MLP and embedding
    # ================================================================ #
    emb_dim = 2
    in_dim = 1 + emb_dim + 1 + 1  # v + emb + msg + exc = 5

    f_theta = nn.Sequential(
        nn.Linear(in_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    model = types.SimpleNamespace(
        f_theta=f_theta,
        a=nn.Parameter(torch.randn(N, emb_dim) * 0.1),
    )

    device = torch.device("cpu")

    # ================================================================ #
    #  Compute the loss
    # ================================================================ #
    loss = f_theta_msg_linearity_loss(model, N, mu, sigma, device, n_pts=50)
    loss_val = loss.item()
    print(f"\n  msg linearity loss = {loss_val:.6f}")

    # ================================================================ #
    #  Condition (1): residual_MSE / output_variance > 0.01
    #  Compute per-neuron: for each neuron, ratio of residual variance
    #  to total output variance along the msg axis. Then average.
    #  This avoids diluting msg-nonlinearity with inter-neuron variance.
    # ================================================================ #
    with torch.no_grad():
        from connectome_gnn.LLM_code.staging.block_04.f_theta_msg_linearity import (
            _vectorized_linspace,
            _torch_linear_fit,
        )
        sigma_safe = np.maximum(sigma, 1e-6)
        msg_grid = _vectorized_linspace(
            mu - 2.0 * sigma_safe, mu + 2.0 * sigma_safe, 50, device
        )
        mu_t = torch.as_tensor(mu, dtype=torch.float32)
        v_flat = mu_t[:, None].expand(-1, 50).reshape(-1, 1)
        emb_flat = model.a[:N, None, :].expand(-1, 50, -1).reshape(-1, emb_dim)
        msg_flat = msg_grid.reshape(-1, 1)
        exc_flat = torch.zeros_like(msg_flat)
        feats = torch.cat([v_flat, emb_flat, msg_flat, exc_flat], dim=1)
        out = model.f_theta(feats.float()).squeeze(-1).reshape(N, 50)

        # Per-neuron: fit line, compute residual fraction
        slopes, offsets = _torch_linear_fit(msg_grid, out)
        linear_pred = slopes[:, None] * msg_grid + offsets[:, None]
        residual = out - linear_pred
        per_neuron_resid_var = (residual ** 2).mean(dim=1)   # (N,)
        per_neuron_out_var = out.var(dim=1)                  # (N,)
        # Only count neurons with non-trivial output variation
        valid = per_neuron_out_var > 1e-12
        if valid.sum() > 0:
            nonlin_frac = (per_neuron_resid_var[valid] / per_neuron_out_var[valid]).mean().item()
        else:
            nonlin_frac = 0.0

    print(f"  per-neuron nonlinearity fraction = {nonlin_frac:.4f} (threshold > 0.01)")
    print(f"  neurons with valid output variance: {valid.sum().item()}/{N}")
    cond1 = nonlin_frac > 0.01

    # ================================================================ #
    #  Condition (2): gradient norm > 0
    # ================================================================ #
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in f_theta.parameters() if p.grad is not None
    )
    print(f"  gradient norm = {grad_norm:.6f} (must be > 0)")
    cond2 = grad_norm > 0

    # ================================================================ #
    #  Condition (3): all outputs finite
    # ================================================================ #
    cond3 = np.isfinite(loss_val) and np.isfinite(grad_norm)
    print(f"  finite check: loss={np.isfinite(loss_val)}, grad={np.isfinite(grad_norm)}")

    # ================================================================ #
    #  Verdict
    # ================================================================ #
    all_pass = cond1 and cond2 and cond3

    if all_pass:
        print(
            f"\nPASS: f_theta msg linearity — nonlinearity fraction={nonlin_frac:.4f}, "
            f"grad_norm={grad_norm:.4f}, all finite"
        )
    else:
        failures = []
        if not cond1:
            failures.append(f"nonlinearity fraction {nonlin_frac:.4f} <= 0.01")
        if not cond2:
            failures.append(f"gradient norm {grad_norm} == 0")
        if not cond3:
            failures.append("non-finite values detected")
        print(f"\nFAIL: {'; '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
