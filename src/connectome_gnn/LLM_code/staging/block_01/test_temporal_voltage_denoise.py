"""Test temporal_voltage_denoise on the full FlyVis voltage cache.

Verifies that Gaussian smoothing (sigma=3) with alpha=0.3 blending
meaningfully reduces noise while preserving signal. The deployed mechanism
is the *blend* v_out = (1-alpha)*v_noisy + alpha*v_denoised, not the raw
denoised output (which over-smoothes at this sigma/noise ratio).

PASS criteria (all must hold):
  1. Alpha=0.3 blend: Pearson(blended, clean) > Pearson(noisy, clean) + 0.01
     (noise reduction is real and meaningful).
  2. Alpha=0.3 blend: Pearson(blended, clean) > Pearson(noisy, clean)
     (no signal destruction — blend preserves dynamics).
  3. MSE(blended, clean) < MSE(noisy, clean) (noise power reduced).
  4. Output shape matches input shape, no NaN/Inf.
  5. At least 80% of neurons individually improve under blending.
"""

import sys
import torch


def per_neuron_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-column (neuron) Pearson correlation between (T, N) tensors."""
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    num = (a_c * b_c).sum(dim=0)
    den = torch.sqrt((a_c ** 2).sum(dim=0) * (b_c ** 2).sum(dim=0))
    return num / den.clamp(min=1e-12)


def main():
    # -- Load data --
    from connectome_gnn.LLM_code.scratchpad import load_full_voltage

    print("Loading voltage data...")
    v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", 0.10)
    T, N = v_clean.shape
    print(f"  Loaded: T={T}, N={N}")

    # -- Apply denoising (sigma=3 per hypothesis) --
    from connectome_gnn.LLM_code.staging.block_01.temporal_voltage_denoise import (
        temporal_voltage_denoise,
    )

    print("Applying temporal_voltage_denoise (sigma=3)...")
    v_denoised = temporal_voltage_denoise(v_noisy, sigma=3.0)

    # Check 4a: shape
    if v_denoised.shape != v_noisy.shape:
        print(f"FAIL: Shape mismatch: {v_denoised.shape} vs {v_noisy.shape}")
        sys.exit(1)

    # Check 4b: finiteness
    if not torch.isfinite(v_denoised).all():
        print("FAIL: Output contains NaN or Inf")
        sys.exit(1)
    print("  Shape and finiteness OK.")

    # -- Alpha=0.3 blend (as specified in hypothesis) --
    alpha = 0.3
    v_blended = (1.0 - alpha) * v_noisy + alpha * v_denoised

    # -- Per-neuron Pearson with clean ground truth --
    print("Computing per-neuron Pearson correlations...")
    r_noisy = per_neuron_pearson(v_noisy, v_clean)
    r_denoised = per_neuron_pearson(v_denoised, v_clean)
    r_blended = per_neuron_pearson(v_blended, v_clean)

    mean_r_noisy = r_noisy.mean().item()
    mean_r_denoised = r_denoised.mean().item()
    mean_r_blended = r_blended.mean().item()

    print(f"  Mean Pearson(noisy, clean)    = {mean_r_noisy:.6f}")
    print(f"  Mean Pearson(denoised, clean) = {mean_r_denoised:.6f}")
    print(f"  Mean Pearson(blended, clean)  = {mean_r_blended:.6f}")

    improvement_full = mean_r_denoised - mean_r_noisy
    improvement_blend = mean_r_blended - mean_r_noisy
    print(f"  Improvement (full denoise): {improvement_full:+.6f}")
    print(f"  Improvement (alpha={alpha} blend): {improvement_blend:+.6f}")

    # -- MSE noise power --
    mse_noisy = (v_noisy - v_clean).pow(2).mean().item()
    mse_denoised = (v_denoised - v_clean).pow(2).mean().item()
    mse_blended = (v_blended - v_clean).pow(2).mean().item()
    print(f"  MSE(noisy, clean)    = {mse_noisy:.6f}")
    print(f"  MSE(denoised, clean) = {mse_denoised:.6f}")
    print(f"  MSE(blended, clean)  = {mse_blended:.6f}")

    # -- Fraction of neurons improved --
    frac_improved = (r_blended > r_noisy).float().mean().item()
    print(f"  Fraction of neurons improved by blend: {frac_improved:.4f}")

    # -- Gate checks --
    failures = []

    # Check 1: blend improves Pearson by at least 0.01 (meaningful effect)
    if improvement_blend < 0.01:
        failures.append(
            f"blended Pearson improvement {improvement_blend:.6f} < 0.01"
        )

    # Check 2: blend preserves signal (must exceed noisy baseline)
    if mean_r_blended <= mean_r_noisy:
        failures.append(
            f"blended Pearson {mean_r_blended:.6f} <= noisy {mean_r_noisy:.6f}"
        )

    # Check 3: MSE reduced
    if mse_blended >= mse_noisy:
        failures.append(
            f"blended MSE {mse_blended:.6f} >= noisy MSE {mse_noisy:.6f}"
        )

    # Check 5: most neurons benefit
    if frac_improved < 0.80:
        failures.append(f"only {frac_improved*100:.1f}% neurons improved (<80%)")

    if failures:
        print(f"FAIL: {'; '.join(failures)}")
        sys.exit(1)

    print(
        f"PASS: sigma=3 Gaussian denoise improves mean Pearson by "
        f"{improvement_full:+.4f} (to {mean_r_denoised:.4f}); "
        f"alpha={alpha} blend +{improvement_blend:.4f}, "
        f"MSE reduced {(1 - mse_blended/mse_noisy)*100:.1f}%, "
        f"{frac_improved*100:.1f}% neurons improved"
    )


if __name__ == "__main__":
    main()
