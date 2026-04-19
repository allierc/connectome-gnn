"""Analyse the 'dead coefficient' problem in epoch-based regularization annealing.

With n_epochs=1 and regul_annealing_rate=0.5 (winner config), the annealing
formula `coeff * (1 - exp(-rate * epoch))` evaluates to zero for ALL annealed
regularizers throughout training. This analysis:

1. Confirms which coefficients are annealed vs non-annealed
2. Computes effective values under both epoch-based and proposed step-based schemes
3. Quantifies the regularization gap at different DAL values
"""
from __future__ import annotations
import numpy as np


def epoch_anneal(coeff: float, rate: float, epoch: int) -> float:
    """Current epoch-based annealing (from regularizer.py line 152)."""
    return float(coeff * (1 - np.exp(-rate * epoch))) if rate > 0 else float(coeff)


def step_anneal(coeff: float, rate: float, epoch: int, iter_in_epoch: int,
                Niter: int, n_epochs: int) -> float:
    """Proposed step-based annealing using fractional training progress."""
    if rate <= 0:
        return float(coeff)
    progress = (epoch * Niter + iter_in_epoch) / (n_epochs * Niter)
    effective_epoch = progress * n_epochs
    return float(coeff * (1 - np.exp(-rate * effective_epoch)))


def main():
    # Winner config parameters
    n_epochs = 1
    rate = 0.5
    batch_size = 6
    n_frames = 64000

    # Annealed coefficients from winner config
    annealed_coeffs = {
        'W_L1': 0.00015,
        'W_L2': 1.5e-6,
        'g_phi_weight_L1': 0.28,
        'g_phi_weight_L2': 0.0,
        'f_theta_weight_L1': 0.05,
        'f_theta_weight_L2': 0.001,
    }

    # Non-annealed coefficients (always active)
    non_annealed = {
        'g_phi_diff': 2000,
        'g_phi_norm': 0.9,
        'f_theta_zero': 0,
        'f_theta_diff': 0,
        'f_theta_msg_diff': 0,
    }

    print("=" * 70)
    print("DEAD COEFFICIENT ANALYSIS — n_epochs=1, regul_annealing_rate=0.5")
    print("=" * 70)

    # 1. Epoch-based: all annealed coefficients at epoch 0
    print("\n--- Epoch-based annealing at epoch=0 (current winner behavior) ---")
    for name, configured_val in annealed_coeffs.items():
        effective = epoch_anneal(configured_val, rate, epoch=0)
        print(f"  {name:25s} configured={configured_val:.6f}  effective={effective:.6f}  "
              f"{'DEAD' if effective == 0 else 'ACTIVE'}")

    print("\n--- Non-annealed coefficients (always active) ---")
    for name, val in non_annealed.items():
        status = 'ACTIVE' if val > 0 else 'inactive (configured=0)'
        print(f"  {name:25s} value={val:.6f}  {status}")

    # 2. Compare schedules at different DAL values
    dal_values = [20, 80, 110]
    print(f"\n{'=' * 70}")
    print("STEP-BASED vs EPOCH-BASED COMPARISON")
    print(f"{'=' * 70}")

    for dal in dal_values:
        Niter = int(n_frames * dal // batch_size * 0.2)
        training_min = Niter * 0.085 / 60  # rough: ~85ms per iter on A100

        print(f"\n--- DAL={dal} (Niter={Niter}, ~{training_min:.0f} min) ---")

        checkpoints = [0, Niter // 4, Niter // 2, 3 * Niter // 4, Niter - 1]
        frac_labels = ['0%', '25%', '50%', '75%', '100%']

        for name, configured_val in [('W_L1', 0.00015), ('g_phi_weight_L1', 0.28)]:
            print(f"\n  {name} (configured={configured_val}):")
            print(f"  {'Progress':>10s}  {'Epoch-based':>12s}  {'Step-based':>12s}  {'Ratio':>8s}")
            for ck, label in zip(checkpoints, frac_labels):
                e_val = epoch_anneal(configured_val, rate, epoch=0)
                s_val = step_anneal(configured_val, rate, 0, ck, Niter, n_epochs)
                ratio = s_val / configured_val if configured_val > 0 else 0
                print(f"  {label:>10s}  {e_val:12.8f}  {s_val:12.8f}  {ratio:7.1%}")

    # 3. DAL invariance check: final value should be the same
    print(f"\n{'=' * 70}")
    print("DAL INVARIANCE — final step-based W_L1 value at different DAL")
    print(f"{'=' * 70}")
    for dal in [20, 40, 80, 110, 200]:
        Niter = int(n_frames * dal // batch_size * 0.2)
        final_val = step_anneal(0.00015, rate, 0, Niter - 1, Niter, n_epochs)
        frac = final_val / 0.00015
        print(f"  DAL={dal:>4d}  Niter={Niter:>7d}  final W_L1={final_val:.8f}  ({frac:.3%} of configured)")

    # 4. Multi-epoch backward compatibility
    print(f"\n{'=' * 70}")
    print("BACKWARD COMPATIBILITY — n_epochs=2, step-based at epoch boundaries")
    print(f"{'=' * 70}")
    Niter_2e = int(n_frames * 20 // batch_size * 0.2)
    for ep in range(2):
        e_val = epoch_anneal(0.00015, rate, epoch=ep)
        s_val = step_anneal(0.00015, rate, ep, 0, Niter_2e, 2)
        match = abs(e_val - s_val) < 1e-12
        print(f"  epoch={ep}, iter=0: epoch-based={e_val:.8f}  step-based={s_val:.8f}  "
              f"{'MATCH' if match else 'MISMATCH'}")

    # 5. Quantify total regularization dose
    print(f"\n{'=' * 70}")
    print("TOTAL REGULARIZATION DOSE (integral of effective coeff over training)")
    print(f"{'=' * 70}")
    for dal in [20, 110]:
        Niter = int(n_frames * dal // batch_size * 0.2)
        epoch_dose = sum(epoch_anneal(0.00015, rate, 0) for _ in range(Niter))
        step_dose = sum(step_anneal(0.00015, rate, 0, k, Niter, n_epochs) for k in range(Niter))
        print(f"  DAL={dal:>3d}: epoch-based dose = {epoch_dose:.6f}  "
              f"step-based dose = {step_dose:.6f}  ratio = {step_dose / max(epoch_dose, 1e-20):.1f}x")

    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")
    print("With n_epochs=1 and regul_annealing_rate=0.5:")
    print("  - Epoch-based: W_L1, W_L2, g_phi_weight_L1, f_theta_weight_L1 are ALL ZERO")
    print("  - Step-based: these coefficients ramp from 0 to ~39.3% of configured values")
    print("  - Step-based final value is DAL-invariant (same fraction regardless of DAL)")
    print("  - Step-based matches epoch-based at epoch boundaries (backward compatible)")


if __name__ == '__main__':
    main()
