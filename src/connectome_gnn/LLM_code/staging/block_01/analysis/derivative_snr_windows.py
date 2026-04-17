"""Derivative SNR analysis under temporal moving-average pre-filters.

Quantifies how much temporal smoothing improves the finite-difference
derivative signal (what f_theta learns) vs raw noisy voltage.
This informs the optimal window size for the Block 01 denoising mechanism.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/workspace/connectome-gnn/src")
from connectome_gnn.LLM_code.scratchpad import load_full_voltage


def temporal_ma(v: torch.Tensor, window: int) -> torch.Tensor:
    """Symmetric moving average along time axis.  v: (T, N) -> (T, N)."""
    if window <= 1:
        return v.clone()
    pad = window // 2
    # (N, 1, T) for conv1d
    v_t = v.T.unsqueeze(1).float()
    kernel = torch.ones(1, 1, window, device=v.device, dtype=v.dtype) / window
    v_padded = F.pad(v_t, (pad, pad), mode="reflect")
    v_smooth = F.conv1d(v_padded, kernel)
    return v_smooth.squeeze(1).T  # (T, N)


def per_neuron_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-neuron Pearson r.  a, b: (T, N)."""
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    num = (a_c * b_c).sum(dim=0)
    den = a_c.norm(dim=0) * b_c.norm(dim=0) + 1e-12
    return (num / den).mean().item()


def main():
    print("Loading flyvis voltage (T~64k, N~13.7k) ...")
    v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", 0.10)
    T, N = v_clean.shape
    print(f"Shape: T={T}, N={N}")

    # --- Voltage-level baseline ---
    mse_raw_v = ((v_noisy - v_clean) ** 2).mean().item()
    corr_raw_v = per_neuron_pearson(v_noisy, v_clean)
    print(f"\n=== Voltage level ===")
    print(f"{'raw noisy':>12s}  MSE={mse_raw_v:.6f}  corr={corr_raw_v:.4f}")

    for w in [3, 5, 7, 9, 11]:
        v_s = temporal_ma(v_noisy, w)
        mse = ((v_s - v_clean) ** 2).mean().item()
        corr = per_neuron_pearson(v_s, v_clean)
        red = (1 - mse / mse_raw_v) * 100
        print(f"  window={w:2d}   MSE={mse:.6f}  reduction={red:5.1f}%  corr={corr:.4f}")
        del v_s

    # --- Stride-1 derivative (what f_theta learns in 1-step mode) ---
    dv_clean = v_clean[1:] - v_clean[:-1]
    dv_noisy = v_noisy[1:] - v_noisy[:-1]
    mse_raw_d1 = ((dv_noisy - dv_clean) ** 2).mean().item()
    corr_raw_d1 = per_neuron_pearson(dv_noisy, dv_clean)
    print(f"\n=== Stride-1 derivative ===")
    print(f"{'raw noisy':>12s}  MSE={mse_raw_d1:.6f}  corr={corr_raw_d1:.4f}")

    for w in [3, 5, 7, 9, 11]:
        v_s = temporal_ma(v_noisy, w)
        dv_s = v_s[1:] - v_s[:-1]
        mse = ((dv_s - dv_clean) ** 2).mean().item()
        corr = per_neuron_pearson(dv_s, dv_clean)
        red = (1 - mse / mse_raw_d1) * 100
        print(f"  window={w:2d}   MSE={mse:.6f}  reduction={red:5.1f}%  corr={corr:.4f}")
        del v_s, dv_s

    # --- Stride-5 derivative (what recurrent time_step=5 targets) ---
    dv5_clean = v_clean[5:] - v_clean[:-5]
    dv5_noisy = v_noisy[5:] - v_noisy[:-5]
    mse_raw_d5 = ((dv5_noisy - dv5_clean) ** 2).mean().item()
    corr_raw_d5 = per_neuron_pearson(dv5_noisy, dv5_clean)
    print(f"\n=== Stride-5 derivative (recurrent target) ===")
    print(f"{'raw noisy':>12s}  MSE={mse_raw_d5:.6f}  corr={corr_raw_d5:.4f}")

    for w in [3, 5, 7, 9, 11]:
        v_s = temporal_ma(v_noisy, w)
        dv5_s = v_s[5:] - v_s[:-5]
        mse = ((dv5_s - dv5_clean) ** 2).mean().item()
        corr = per_neuron_pearson(dv5_s, dv5_clean)
        red = (1 - mse / mse_raw_d5) * 100
        print(f"  window={w:2d}   MSE={mse:.6f}  reduction={red:5.1f}%  corr={corr:.4f}")
        del v_s, dv5_s

    # --- Signal variance vs noise variance (SNR) ---
    signal_var_d1 = dv_clean.var().item()
    noise_var_d1 = ((dv_noisy - dv_clean) ** 2).mean().item()
    print(f"\n=== SNR summary ===")
    print(f"Stride-1 derivative: signal_var={signal_var_d1:.6f}  noise_var={noise_var_d1:.6f}  SNR={signal_var_d1/noise_var_d1:.2f}")
    signal_var_d5 = dv5_clean.var().item()
    noise_var_d5 = ((dv5_noisy - dv5_clean) ** 2).mean().item()
    print(f"Stride-5 derivative: signal_var={signal_var_d5:.6f}  noise_var={noise_var_d5:.6f}  SNR={signal_var_d5/noise_var_d5:.2f}")


if __name__ == "__main__":
    main()
