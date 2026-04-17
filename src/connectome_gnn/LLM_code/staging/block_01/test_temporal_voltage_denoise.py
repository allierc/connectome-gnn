#!/usr/bin/env python
"""Phase-S test for temporal_voltage_denoise (block 01).

Loads full flyvis data (T=64000, N=13741, γ=0.10), applies the denoiser
with window=3, and checks the three PASS conditions:
  (1) MSE(denoised, clean) / MSE(noisy, clean) ≤ 0.70
  (2) mean per-neuron Pearson-r(denoised, clean) > Pearson-r(noisy, clean)
  (3) ≥ 75% of neurons have lower MSE after denoising
"""

from __future__ import annotations

import sys


def pearson_per_neuron(a, b):
    """Per-neuron Pearson correlation between (T, N) tensors. Returns (N,)."""
    a_m = a - a.mean(dim=0, keepdim=True)
    b_m = b - b.mean(dim=0, keepdim=True)
    num = (a_m * b_m).sum(dim=0)
    den = (a_m.norm(dim=0) * b_m.norm(dim=0)).clamp(min=1e-12)
    return num / den


def main():
    import torch
    from connectome_gnn.LLM_code.scratchpad import load_full_voltage
    from connectome_gnn.LLM_code.staging.block_01.temporal_voltage_denoise import (
        temporal_voltage_denoise,
    )

    print("Loading flyvis voltage data (T=64000, N=13741, γ=0.10)...")
    v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", 0.10)
    print(f"  v_clean shape: {v_clean.shape}, v_noisy shape: {v_noisy.shape}")

    print("Applying temporal_voltage_denoise(window=3)...")
    v_denoised = temporal_voltage_denoise(v_noisy, window=3)
    print(f"  v_denoised shape: {v_denoised.shape}")

    # --- Condition (1): MSE ratio ≤ 0.70 ---
    mse_noisy = ((v_noisy - v_clean) ** 2).mean().item()
    mse_denoised = ((v_denoised - v_clean) ** 2).mean().item()
    mse_ratio = mse_denoised / mse_noisy
    print(f"\nCondition 1 — MSE ratio: {mse_ratio:.4f} (threshold ≤ 0.70)")
    cond1 = mse_ratio <= 0.70

    # --- Condition (2): mean Pearson-r improves ---
    r_noisy = pearson_per_neuron(v_noisy, v_clean)
    r_denoised = pearson_per_neuron(v_denoised, v_clean)
    mean_r_noisy = r_noisy.mean().item()
    mean_r_denoised = r_denoised.mean().item()
    print(
        f"Condition 2 — mean Pearson-r: noisy={mean_r_noisy:.4f}, "
        f"denoised={mean_r_denoised:.4f} (must improve)"
    )
    cond2 = mean_r_denoised > mean_r_noisy

    # --- Condition (3): ≥ 75% neurons have lower MSE ---
    mse_per_neuron_noisy = ((v_noisy - v_clean) ** 2).mean(dim=0)
    mse_per_neuron_denoised = ((v_denoised - v_clean) ** 2).mean(dim=0)
    frac_improved = (mse_per_neuron_denoised < mse_per_neuron_noisy).float().mean().item()
    print(
        f"Condition 3 — fraction of neurons with lower MSE: {frac_improved:.4f} "
        f"(threshold ≥ 0.75)"
    )
    cond3 = frac_improved >= 0.75

    # --- Verdict ---
    all_pass = cond1 and cond2 and cond3
    if all_pass:
        print(
            f"\nPASS: window=3 denoising — MSE ratio={mse_ratio:.4f}, "
            f"Δr=+{mean_r_denoised - mean_r_noisy:.4f}, "
            f"{frac_improved*100:.1f}% neurons improved"
        )
    else:
        failures = []
        if not cond1:
            failures.append(f"MSE ratio {mse_ratio:.4f} > 0.70")
        if not cond2:
            failures.append(f"Pearson-r did not improve ({mean_r_denoised:.4f} ≤ {mean_r_noisy:.4f})")
        if not cond3:
            failures.append(f"only {frac_improved*100:.1f}% neurons improved (< 75%)")
        print(f"\nFAIL: {'; '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
