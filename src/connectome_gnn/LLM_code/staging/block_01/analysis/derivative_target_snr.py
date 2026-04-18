"""Analyse the SNR distribution of derivative targets under γ=0.10 measurement noise.

Motivation: the previous Block 01 denoised INPUT voltage (g_phi side), but the
DERIVATIVE TARGETS remain noisy. This analysis quantifies what fraction of
training samples are noise-dominated and whether SNR-based loss reweighting
would meaningfully change the training signal.

Derivative target: y[k] = (v[k+1] - v[k]) / dt
Noise component:   (ε[k+1] - ε[k]) * γ / dt  with var = 2γ²/dt²
"""
from __future__ import annotations

import sys
import numpy as np
import torch

# ── load data ──────────────────────────────────────────────────────────
sys.path.insert(0, "/workspace/connectome-gnn/src")
from connectome_gnn.LLM_code.scratchpad import load_full_voltage

gamma = 0.10
v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", gamma)
T, N = v_clean.shape
print(f"Loaded voltage: T={T}, N={N}, γ={gamma}")

# ── compute derivatives ───────────────────────────────────────────────
dy_clean = (v_clean[1:] - v_clean[:-1]).numpy()   # (T-1, N), dt=1
dy_noisy = (v_noisy[1:] - v_noisy[:-1]).numpy()

noise_var = 2 * gamma**2   # variance of derivative noise per sample

# ── per-sample SNR weight ─────────────────────────────────────────────
# Wiener-style weight: w = |signal|² / (|signal|² + noise_var)
# We approximate |signal|² ≈ |y_noisy|² - noise_var (bias-corrected)
y2 = dy_noisy**2
signal_est = np.maximum(y2 - noise_var, 0.0)  # clamp to non-negative
weights = signal_est / (signal_est + noise_var + 1e-12)

# ── distribution statistics ───────────────────────────────────────────
print(f"\n=== Derivative target statistics ===")
print(f"  dy_clean:  mean={dy_clean.mean():.4f}, std={dy_clean.std():.4f}")
print(f"  dy_noisy:  mean={dy_noisy.mean():.4f}, std={dy_noisy.std():.4f}")
print(f"  noise_std_theory = {np.sqrt(noise_var):.4f}")

print(f"\n=== SNR weight distribution ===")
print(f"  mean weight:   {weights.mean():.4f}")
print(f"  std weight:    {weights.std():.4f}")
print(f"  median weight: {np.median(weights):.4f}")
for thr in [0.1, 0.3, 0.5, 0.7, 0.9]:
    frac = (weights < thr).mean()
    print(f"  fraction w < {thr}: {frac:.4f}")

# ── correlation with true SNR ─────────────────────────────────────────
true_snr = dy_clean**2 / noise_var
# flatten for correlation
flat_w = weights.ravel()
flat_true_snr = true_snr.ravel()

# subsample for speed (10M samples)
rng = np.random.default_rng(42)
n_total = flat_w.shape[0]
n_sub = min(10_000_000, n_total)
idx = rng.choice(n_total, size=n_sub, replace=False)
r_corr = np.corrcoef(flat_w[idx], flat_true_snr[idx])[0, 1]
print(f"\n=== Validation ===")
print(f"  Pearson r(estimated weight, true SNR) = {r_corr:.4f}")

# ── effect on effective sample size ───────────────────────────────────
eff_n = (weights.sum()**2) / (weights**2).sum()
eff_frac = eff_n / weights.size
print(f"  effective sample fraction (Kish): {eff_frac:.4f}")
print(f"  effective N / total N: {eff_n:.0f} / {weights.size}")

# ── MSE improvement under weighting ──────────────────────────────────
# weighted MSE between noisy and clean derivatives
mse_uniform = ((dy_noisy - dy_clean)**2).mean()
mse_weighted = (weights * (dy_noisy - dy_clean)**2).sum() / weights.sum()
print(f"\n=== Expected loss quality ===")
print(f"  uniform MSE(noisy, clean derivative):   {mse_uniform:.6f}")
print(f"  weighted MSE(noisy, clean derivative):   {mse_weighted:.6f}")
print(f"  ratio (weighted/uniform):               {mse_weighted/mse_uniform:.4f}")
print(f"  → {'IMPROVEMENT' if mse_weighted < mse_uniform else 'NO IMPROVEMENT'}: "
      f"weighted loss focuses on samples where noisy≈clean")

# ── per-neuron statistics ─────────────────────────────────────────────
neuron_mean_w = weights.mean(axis=0)
print(f"\n=== Per-neuron weight stats ===")
print(f"  mean(neuron_mean_w): {neuron_mean_w.mean():.4f}")
print(f"  std(neuron_mean_w):  {neuron_mean_w.std():.4f}")
print(f"  min(neuron_mean_w):  {neuron_mean_w.min():.4f}")
print(f"  max(neuron_mean_w):  {neuron_mean_w.max():.4f}")
n_low = (neuron_mean_w < 0.3).sum()
print(f"  neurons with mean w < 0.3: {n_low}/{N}")

print("\nDONE")
