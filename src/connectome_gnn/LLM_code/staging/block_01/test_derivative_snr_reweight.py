#!/usr/bin/env python
"""Phase-S test for derivative_snr_reweight (block 01).

Loads full flyvis data (T~64000, N=13741, gamma=0.10), computes stride-1
derivatives (clean and noisy), applies Wiener SNR weights, and checks:
  (1) Signal concentration: weighted mean(dy_clean^2) / unweighted >= 1.5
  (2) Effective SNR gain >= 1.5 (weighted gradient has 50%+ better SNR)
  (3) Kish effective sample fraction >= 0.3 (not overly aggressive)
"""

from __future__ import annotations

import sys


def main():
    import numpy as np
    import torch
    from connectome_gnn.LLM_code.scratchpad import load_full_voltage
    from connectome_gnn.LLM_code.staging.block_01.derivative_snr_reweight import (
        derivative_snr_weights,
    )

    gamma = 0.10
    print(f"Loading flyvis voltage data (gamma={gamma}) ...")
    v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", gamma)
    T, N = v_clean.shape
    print(f"  T={T}, N={N}")

    # Stride-1 derivatives
    dy_clean = v_clean[1:] - v_clean[:-1]  # (T-1, N)
    dy_noisy = v_noisy[1:] - v_noisy[:-1]  # (T-1, N)

    print("Computing SNR weights ...")
    weights = derivative_snr_weights(dy_noisy, gamma, floor=0.1)
    w = weights.numpy()
    print(f"  weight stats: mean={w.mean():.4f}, std={w.std():.4f}, "
          f"min={w.min():.4f}, max={w.max():.4f}")

    dy_c = dy_clean.numpy()
    noise_var = 2.0 * gamma ** 2

    # --- Condition (1): Signal concentration >= 1.5 ---
    # The weighted mean of dy_clean^2 should be higher than the unweighted
    # mean, because the weights upweight samples where |y_noisy| is large,
    # which on average means |y_clean| is also large.
    unweighted_signal = (dy_c ** 2).mean()
    weighted_signal = (w * dy_c ** 2).sum() / w.sum()
    signal_concentration = weighted_signal / unweighted_signal
    print(f"\nCondition 1 — signal concentration: {signal_concentration:.4f} (threshold >= 1.5)")
    cond1 = signal_concentration >= 1.5

    # --- Condition (2): Effective SNR gain >= 1.5 ---
    # SNR = mean_signal_power / noise_variance
    # Weighted: weighted_mean(dy_clean^2) / noise_var
    # Unweighted: mean(dy_clean^2) / noise_var
    # Gain = weighted / unweighted = signal_concentration (same ratio)
    snr_unweighted = unweighted_signal / noise_var
    snr_weighted = weighted_signal / noise_var
    snr_gain = snr_weighted / snr_unweighted
    print(f"Condition 2 — SNR gain: {snr_gain:.4f} (threshold >= 1.5)")
    print(f"  unweighted SNR: {snr_unweighted:.4f}, weighted SNR: {snr_weighted:.4f}")
    cond2 = snr_gain >= 1.5

    # --- Condition (3): Kish effective sample fraction >= 0.3 ---
    eff_n = (w.sum() ** 2) / (w ** 2).sum()
    eff_frac = eff_n / w.size
    print(f"Condition 3 — Kish effective sample fraction: {eff_frac:.4f} (threshold >= 0.3)")
    cond3 = eff_frac >= 0.3

    # --- Additional diagnostics ---
    frac_at_floor = (w <= 0.1 + 1e-6).mean()
    print(f"\nDiagnostics:")
    print(f"  fraction at floor (w~0.1): {frac_at_floor:.4f}")
    print(f"  fraction w > 0.5: {(w > 0.5).mean():.4f}")
    print(f"  noise_var (2*gamma^2): {noise_var:.6f}")

    # --- Verdict ---
    all_pass = cond1 and cond2 and cond3
    if all_pass:
        print(
            f"\nPASS: SNR reweighting — signal_conc={signal_concentration:.3f}, "
            f"SNR_gain={snr_gain:.3f}, Kish={eff_frac:.3f}"
        )
    else:
        failures = []
        if not cond1:
            failures.append(f"signal_conc {signal_concentration:.4f} < 1.5")
        if not cond2:
            failures.append(f"SNR_gain {snr_gain:.4f} < 1.5")
        if not cond3:
            failures.append(f"Kish {eff_frac:.4f} < 0.3")
        print(f"\nFAIL: {'; '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
