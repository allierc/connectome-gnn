#!/usr/bin/env python3
"""Phase-R analysis: calibrate temporal-average denoising on flyvis voltage.

Measures MSE reduction and correlation preservation for various window sizes
to determine the optimal window and set a realistic PASS condition for Phase S.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/connectome-gnn/src')
from connectome_gnn.LLM_code.scratchpad import load_full_voltage


def temporal_moving_average(v: torch.Tensor, window: int) -> torch.Tensor:
    """Apply symmetric moving-average filter along time axis.

    Args:
        v: (T, N) voltage tensor
        window: odd window size (will be forced odd)
    Returns:
        (T, N) smoothed voltage
    """
    if window <= 1:
        return v.clone()
    if window % 2 == 0:
        window += 1
    T, N = v.shape
    w = window // 2
    # Conv1d along time axis: (N, 1, T) layout
    v_t = v.T.unsqueeze(1).float()  # (N, 1, T)
    v_padded = F.pad(v_t, (w, w), mode='reflect')
    kernel = torch.ones(1, 1, window, dtype=torch.float32) / window
    v_smooth = F.conv1d(v_padded, kernel)  # (N, 1, T)
    return v_smooth.squeeze(1).T  # (T, N)


def per_neuron_pearson_r(a, b):
    """Compute mean per-neuron Pearson correlation.

    Args:
        a, b: (T, N) tensors
    Returns:
        mean correlation across neurons
    """
    a_f = a.float()
    b_f = b.float()
    a_m = a_f - a_f.mean(0, keepdim=True)
    b_m = b_f - b_f.mean(0, keepdim=True)
    num = (a_m * b_m).sum(0)
    den = a_m.norm(dim=0) * b_m.norm(dim=0) + 1e-12
    return (num / den).mean().item()


def main():
    print("Loading flyvis voltage data...")
    v_clean, v_noisy = load_full_voltage('fly/flyvis_noise_free', 0.10)
    T, N = v_clean.shape
    print(f"Shape: T={T}, N={N}")

    mse_noisy = ((v_noisy - v_clean).float() ** 2).mean().item()
    r_noisy = per_neuron_pearson_r(v_noisy, v_clean)

    print(f"\nBaseline (no denoising):")
    print(f"  MSE(noisy, clean) = {mse_noisy:.6f}")
    print(f"  mean per-neuron Pearson-r(noisy, clean) = {r_noisy:.6f}")
    print(f"  noise std (gamma) = {(v_noisy - v_clean).float().std().item():.4f}")
    print(f"  signal std = {v_clean.float().std().item():.4f}")
    print(f"  SNR = {v_clean.float().std().item() / (v_noisy - v_clean).float().std().item():.2f}")

    print(f"\nTemporal moving-average denoising:")
    print(f"{'window':>8} {'MSE_ratio':>10} {'MSE_denoised':>13} {'r_denoised':>11} {'delta_r':>10}")

    for window in [3, 5, 7, 9, 11, 15, 21]:
        v_denoised = temporal_moving_average(v_noisy, window)
        mse_denoised = ((v_denoised - v_clean).float() ** 2).mean().item()
        r_denoised = per_neuron_pearson_r(v_denoised, v_clean)
        ratio = mse_denoised / mse_noisy
        r_gain = r_denoised - r_noisy
        print(f"{window:>8d} {ratio:>10.4f} {mse_denoised:>13.6f} {r_denoised:>11.6f} {r_gain:>+10.6f}")

    # Per-neuron temporal autocorrelation
    print(f"\nSignal temporal autocorrelation (per-neuron, averaged):")
    v_c = v_clean.float()
    for lag in [1, 2, 3, 5, 10]:
        r_lag = per_neuron_pearson_r(v_c[:-lag], v_c[lag:])
        print(f"  lag={lag}: r={r_lag:.6f}")

    # Also check: what fraction of neurons benefit from window=3?
    print(f"\nPer-neuron MSE breakdown for window=3:")
    v_denoised_3 = temporal_moving_average(v_noisy, 3)
    mse_per_neuron_noisy = ((v_noisy - v_clean).float() ** 2).mean(0)
    mse_per_neuron_denoised = ((v_denoised_3 - v_clean).float() ** 2).mean(0)
    frac_better = (mse_per_neuron_denoised < mse_per_neuron_noisy).float().mean().item()
    print(f"  Fraction of neurons with lower MSE: {frac_better:.4f}")
    ratio_per_neuron = (mse_per_neuron_denoised / (mse_per_neuron_noisy + 1e-12))
    print(f"  Per-neuron MSE ratio: mean={ratio_per_neuron.mean().item():.4f}, "
          f"median={ratio_per_neuron.median().item():.4f}, "
          f"std={ratio_per_neuron.std().item():.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
