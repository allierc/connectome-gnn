#!/usr/bin/env python
"""Phase-S test for combined_msg_linearity_snr (block 04 — best-of-combination).

Tests both components of the combined mechanism:
  Component A (msg linearity): creates a random f_theta MLP and verifies
    that it exhibits measurable msg nonlinearity (residual > 1% of variance)
    and that gradients flow.
  Component B (derivative SNR): loads real flyvis voltage data and verifies
    that SNR weights discriminate between high- and low-derivative timesteps.

PASS CONDITIONS (all must hold):
  (1) Random MLP msg residual_MSE / output_variance > 0.01
  (2) Gradient norm w.r.t. f_theta params > 0
  (3) Top-quartile derivative timesteps have higher mean SNR weight than
      bottom-quartile
  (4) All values finite (no NaN/Inf)
"""

from __future__ import annotations

import sys
import types


def main():
    import numpy as np
    import torch
    import torch.nn as nn

    from connectome_gnn.LLM_code.scratchpad import load_full_voltage
    from connectome_gnn.LLM_code.staging.block_04.combined_msg_linearity_snr import (
        derivative_snr_weights,
        f_theta_msg_linearity_loss,
    )

    # ================================================================ #
    #  Component A: f_theta msg linearity loss
    # ================================================================ #
    print("=== Component A: f_theta msg linearity loss ===")

    # Create a minimal dummy model with f_theta MLP and embedding a
    N = 200  # small subset for CPU test
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

    # Synthetic per-neuron stats (realistic range for flyvis)
    rng = np.random.default_rng(42)
    mu = rng.standard_normal(N).astype(np.float32) * 0.5
    sigma = np.abs(rng.standard_normal(N).astype(np.float32)) * 0.3 + 0.1

    device = torch.device("cpu")

    # Compute the loss
    loss = f_theta_msg_linearity_loss(model, N, mu, sigma, device, n_pts=50)
    loss_val = loss.item()
    print(f"  msg linearity loss = {loss_val:.6f}")

    # Condition (1): residual_MSE / output_variance > 0.01
    # Evaluate f_theta to get output variance for comparison
    with torch.no_grad():
        mu_t = torch.as_tensor(mu, dtype=torch.float32)
        msg_grid = mu_t[:, None] + torch.linspace(-1, 1, 50)[None, :] * (
            torch.as_tensor(sigma)[:, None].clamp(min=1e-6) * 2.0
        )
        v_flat = mu_t[:, None].expand(-1, 50).reshape(-1, 1)
        emb_flat = model.a[:N, None, :].expand(-1, 50, -1).reshape(-1, emb_dim)
        msg_flat = msg_grid.reshape(-1, 1)
        exc_flat = torch.zeros_like(msg_flat)
        feats = torch.cat([v_flat, emb_flat, msg_flat, exc_flat], dim=1)
        out = model.f_theta(feats.float()).squeeze(-1).reshape(N, 50)
        output_var = out.var().item()

    nonlin_frac = loss_val / max(output_var, 1e-12)
    print(f"  output variance = {output_var:.6f}")
    print(f"  nonlinearity fraction = {nonlin_frac:.4f} (threshold > 0.01)")
    cond1 = nonlin_frac > 0.01

    # Condition (2): gradient norm > 0
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in f_theta.parameters() if p.grad is not None
    )
    print(f"  gradient norm = {grad_norm:.6f} (must be > 0)")
    cond2 = grad_norm > 0

    # Condition (4a): finite values
    cond4a = np.isfinite(loss_val) and np.isfinite(grad_norm)
    print(f"  finite check: loss={np.isfinite(loss_val)}, grad={np.isfinite(grad_norm)}")

    # ================================================================ #
    #  Component B: derivative SNR weights
    # ================================================================ #
    print("\n=== Component B: derivative SNR weights ===")

    print("Loading flyvis voltage data (T=64000, N=13741, gamma=0.10)...")
    v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", 0.10)
    print(f"  v_clean shape: {v_clean.shape}, v_noisy shape: {v_noisy.shape}")

    # Compute derivatives (stride 1)
    dt = 1.0
    gamma = 0.10
    derivs_noisy = (v_noisy[1:] - v_noisy[:-1]) / dt  # (T-1, N)
    derivs_clean = (v_clean[1:] - v_clean[:-1]) / dt

    # Compute SNR weights on noisy derivatives
    weights = derivative_snr_weights(derivs_noisy, gamma=gamma, dt=dt, min_weight=0.1)
    print(f"  weights shape: {weights.shape}")
    print(f"  weights range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
    print(f"  weights mean: {weights.mean().item():.4f}")

    # Condition (3): top-quartile vs bottom-quartile discrimination
    # Use per-timestep mean absolute derivative across neurons
    deriv_mag = derivs_noisy.abs().mean(dim=1)  # (T-1,)
    q25 = torch.quantile(deriv_mag, 0.25).item()
    q75 = torch.quantile(deriv_mag, 0.75).item()

    w_mean = weights.mean(dim=1)  # per-timestep mean weight
    bottom_q_mask = deriv_mag <= q25
    top_q_mask = deriv_mag >= q75

    mean_w_bottom = w_mean[bottom_q_mask].mean().item()
    mean_w_top = w_mean[top_q_mask].mean().item()

    print(f"  bottom-quartile mean weight: {mean_w_bottom:.4f}")
    print(f"  top-quartile mean weight: {mean_w_top:.4f}")
    print(f"  discrimination: top > bottom = {mean_w_top > mean_w_bottom}")
    cond3 = mean_w_top > mean_w_bottom

    # Condition (4b): finite values
    cond4b = bool(torch.isfinite(weights).all().item())
    print(f"  all weights finite: {cond4b}")

    # ================================================================ #
    #  Verdict
    # ================================================================ #
    cond4 = cond4a and cond4b
    all_pass = cond1 and cond2 and cond3 and cond4

    if all_pass:
        print(
            f"\nPASS: combined mechanism — msg nonlinearity fraction={nonlin_frac:.4f}, "
            f"grad_norm={grad_norm:.4f}, SNR discrimination top/bottom="
            f"{mean_w_top:.4f}/{mean_w_bottom:.4f}"
        )
    else:
        failures = []
        if not cond1:
            failures.append(
                f"msg nonlinearity fraction {nonlin_frac:.4f} <= 0.01"
            )
        if not cond2:
            failures.append(f"gradient norm {grad_norm} == 0")
        if not cond3:
            failures.append(
                f"SNR discrimination failed: top {mean_w_top:.4f} <= bottom {mean_w_bottom:.4f}"
            )
        if not cond4:
            failures.append("non-finite values detected")
        print(f"\nFAIL: {'; '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
