#!/usr/bin/env python
"""Phase-S test for temporal_voltage_denoise (Block 01).

Loads the full flyvis dataset (T=64000, N=13741) with γ=0.10 measurement noise,
applies the window=3 moving-average denoiser, and checks the three PASS conditions:
  (1) MSE ratio ≤ 0.70
  (2) mean per-neuron Pearson-r improves
  (3) ≥ 75% of neurons have lower MSE after denoising
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    from connectome_gnn.LLM_code.scratchpad import load_full_voltage
    from connectome_gnn.LLM_code.staging.block_01.temporal_voltage_denoise import (
        temporal_voltage_denoise,
    )

    print("Loading flyvis data (T=64000, N=13741, γ=0.10) ...")
    v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", 0.10)
    T, N = v_clean.shape
    print(f"  shape: T={T}, N={N}")

    print("Applying temporal_voltage_denoise(window=3) ...")
    v_denoised = temporal_voltage_denoise(v_noisy, window=3)
    assert v_denoised.shape == (T, N), f"shape mismatch: {v_denoised.shape}"

    # --- Condition 1: global MSE ratio ---
    mse_noisy = ((v_noisy - v_clean) ** 2).mean().item()
    mse_denoised = ((v_denoised - v_clean) ** 2).mean().item()
    mse_ratio = mse_denoised / mse_noisy
    print(f"  MSE noisy:    {mse_noisy:.6f}")
    print(f"  MSE denoised: {mse_denoised:.6f}")
    print(f"  MSE ratio:    {mse_ratio:.4f}  (need ≤ 0.70)")
    cond1 = mse_ratio <= 0.70

    # --- Condition 2: mean per-neuron Pearson-r improves ---
    def pearson_per_neuron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-column Pearson correlation between (T, N) tensors."""
        a_c = a - a.mean(dim=0, keepdim=True)
        b_c = b - b.mean(dim=0, keepdim=True)
        num = (a_c * b_c).sum(dim=0)
        den = torch.sqrt((a_c ** 2).sum(dim=0) * (b_c ** 2).sum(dim=0)).clamp(min=1e-12)
        return num / den

    r_noisy = pearson_per_neuron(v_noisy, v_clean)
    r_denoised = pearson_per_neuron(v_denoised, v_clean)
    mean_r_noisy = r_noisy.mean().item()
    mean_r_denoised = r_denoised.mean().item()
    print(f"  Mean Pearson-r noisy:    {mean_r_noisy:.4f}")
    print(f"  Mean Pearson-r denoised: {mean_r_denoised:.4f}")
    cond2 = mean_r_denoised > mean_r_noisy

    # --- Condition 3: ≥ 75% of neurons benefit ---
    mse_per_neuron_noisy = ((v_noisy - v_clean) ** 2).mean(dim=0)
    mse_per_neuron_denoised = ((v_denoised - v_clean) ** 2).mean(dim=0)
    frac_better = (mse_per_neuron_denoised < mse_per_neuron_noisy).float().mean().item()
    print(f"  Fraction of neurons with lower MSE: {frac_better:.4f}  (need ≥ 0.75)")
    cond3 = frac_better >= 0.75

    # --- Verdict ---
    print()
    if cond1 and cond2 and cond3:
        print(
            f"PASS: window=3 denoiser — MSE ratio {mse_ratio:.3f}, "
            f"Pearson-r {mean_r_noisy:.3f}→{mean_r_denoised:.3f}, "
            f"{frac_better*100:.1f}% neurons benefit"
        )
        return 0
    else:
        reasons = []
        if not cond1:
            reasons.append(f"MSE ratio {mse_ratio:.3f} > 0.70")
        if not cond2:
            reasons.append(
                f"Pearson-r did not improve ({mean_r_noisy:.4f} → {mean_r_denoised:.4f})"
            )
        if not cond3:
            reasons.append(f"only {frac_better*100:.1f}% neurons benefit (need 75%)")
        print(f"FAIL: {'; '.join(reasons)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
