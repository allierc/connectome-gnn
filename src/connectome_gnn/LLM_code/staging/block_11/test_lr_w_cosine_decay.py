#!/usr/bin/env python3
"""Test: lr_W cosine decay schedule for flyvis noise=0.05 from-zero exploration.

PASS CONDITION (all must hold):
  1. Mathematical correctness: lr at step=0 equals initial_lr, lr at step=total
     equals initial_lr * min_lr_factor, and lr at step=total/2 equals the midpoint
     (initial_lr * (1 + min_lr_factor) / 2) within 1e-7 tolerance.
  2. Optimizer integration: applying the schedule to a real torch.optim.Adam
     optimizer correctly updates the param_group lr at every step.
  3. Monotonicity: lr is non-increasing over the full schedule (no bumps).
  4. Regression-prevention evidence: in the flyvis DAL=35 trajectory data,
     seeds that regress (peak - final > 0.03) show their peak conn_R2 occurring
     in the first 50% of training steps, meaning the cosine decay would have
     reduced lr by >30% during the regression phase. At least 60% of regressing
     seeds must have this property.
"""

from __future__ import annotations

import csv
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from connectome_gnn.LLM_code.staging.block_11.lr_w_cosine_decay import apply_lr_w_cosine_decay


def test_mathematical_correctness() -> bool:
    """Verify schedule values at key points."""
    import torch

    initial_lr = 0.0006
    min_lr_factor = 0.1
    total_steps = 1000

    # Create a minimal optimizer
    param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([param], lr=initial_lr)

    # Step 0: should be initial_lr
    apply_lr_w_cosine_decay(optimizer, 0, total_steps, initial_lr, min_lr_factor)
    lr_0 = optimizer.param_groups[0]["lr"]
    expected_0 = initial_lr  # cos(0) = 1 → factor = min + (1-min)*1.0 = 1.0
    if abs(lr_0 - expected_0) > 1e-10:
        print(f"FAIL: lr at step=0: expected {expected_0}, got {lr_0}")
        return False

    # Step = total: should be initial_lr * min_lr_factor
    apply_lr_w_cosine_decay(optimizer, total_steps, total_steps, initial_lr, min_lr_factor)
    lr_end = optimizer.param_groups[0]["lr"]
    expected_end = initial_lr * min_lr_factor  # cos(pi) = -1 → factor = min + (1-min)*0 = min
    if abs(lr_end - expected_end) > 1e-10:
        print(f"FAIL: lr at step=total: expected {expected_end}, got {lr_end}")
        return False

    # Step = total/2: should be midpoint
    apply_lr_w_cosine_decay(optimizer, total_steps // 2, total_steps, initial_lr, min_lr_factor)
    lr_mid = optimizer.param_groups[0]["lr"]
    expected_mid = initial_lr * (min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * 0.5)))
    # cos(pi/2) = 0 → factor = min + (1-min)*0.5
    if abs(lr_mid - expected_mid) > 1e-7:
        print(f"FAIL: lr at step=total/2: expected {expected_mid:.10f}, got {lr_mid:.10f}")
        return False

    print(f"  Mathematical correctness: lr_0={lr_0:.6f}, lr_mid={lr_mid:.6f}, lr_end={lr_end:.7f} ✓")
    return True


def test_optimizer_integration() -> bool:
    """Verify that schedule correctly updates lr in optimizer param_groups."""
    import torch

    initial_lr = 0.001
    min_lr_factor = 0.1
    total_steps = 100

    # Multi-param-group optimizer
    p1 = torch.zeros(10, requires_grad=True)
    p2 = torch.zeros(5, requires_grad=True)
    optimizer = torch.optim.Adam([
        {"params": [p1], "lr": initial_lr},
        {"params": [p2], "lr": initial_lr},
    ])

    # Apply at step 75 (3/4 through training)
    apply_lr_w_cosine_decay(optimizer, 75, total_steps, initial_lr, min_lr_factor)

    expected = initial_lr * (min_lr_factor + (1 - min_lr_factor) * 0.5 * (1 + math.cos(math.pi * 0.75)))
    for i, pg in enumerate(optimizer.param_groups):
        if abs(pg["lr"] - expected) > 1e-10:
            print(f"FAIL: param_group[{i}] lr={pg['lr']}, expected {expected}")
            return False

    print(f"  Optimizer integration: all param_groups updated correctly at step 75/100 ✓")
    return True


def test_monotonicity() -> bool:
    """Verify lr is non-increasing over full schedule."""
    import torch

    initial_lr = 0.0006
    min_lr_factor = 0.1
    total_steps = 2000

    param = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([param], lr=initial_lr)

    prev_lr = float("inf")
    violations = 0
    for step in range(total_steps + 1):
        apply_lr_w_cosine_decay(optimizer, step, total_steps, initial_lr, min_lr_factor)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr > prev_lr + 1e-15:  # small tolerance for float
            violations += 1
        prev_lr = current_lr

    if violations > 0:
        print(f"FAIL: monotonicity violated {violations} times over {total_steps} steps")
        return False

    print(f"  Monotonicity: lr is non-increasing over all {total_steps} steps ✓")
    return True


def test_regression_prevention_evidence() -> bool:
    """Analyze trajectory data to verify cosine decay targets the regression phase.

    For seeds that show regression (peak - final > 0.03), check that the peak
    occurs early enough that cosine decay would have meaningfully reduced lr
    during the regression phase.
    """
    traj_dir = os.path.join(
        REPO_ROOT,
        "log", "remote", "Claude_exploration",
        "LLM_flyvis_noise_005_from_zero", "r2_trajectory",
    )
    if not os.path.isdir(traj_dir):
        print(f"FAIL: trajectory dir not found: {traj_dir}")
        return False

    # Load all trajectories
    regression_threshold = 0.03  # peak - final > this = regression
    regressing_seeds = []
    total_seeds = 0

    for fname in sorted(os.listdir(traj_dir)):
        if not fname.endswith(".log"):
            continue
        path = os.path.join(traj_dir, fname)
        with open(path) as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header
            rows = []
            for r in reader:
                try:
                    rows.append((int(r[0]), float(r[1])))
                except (ValueError, IndexError):
                    continue

        if len(rows) < 3:
            continue
        total_seeds += 1

        peak_r2 = max(r[1] for r in rows)
        final_r2 = rows[-1][1]
        drop = peak_r2 - final_r2

        if drop > regression_threshold:
            # Find where peak occurs (as fraction of total steps)
            total_training_steps = rows[-1][0]
            peak_step = max(rows, key=lambda x: x[1])[0]
            peak_fraction = peak_step / total_training_steps if total_training_steps > 0 else 0

            # Compute what cosine decay lr would be at peak step
            progress_at_peak = peak_fraction
            cosine_at_peak = 0.5 * (1 + math.cos(math.pi * progress_at_peak))
            min_lr_factor = 0.1
            lr_factor_at_peak = min_lr_factor + (1 - min_lr_factor) * cosine_at_peak

            # After the peak, lr continues to decay. Compute average lr reduction
            # in the post-peak phase relative to constant lr
            # At peak: lr_factor_at_peak; at end: 0.1; average post-peak factor:
            avg_post_peak_factor = (lr_factor_at_peak + min_lr_factor) / 2
            lr_reduction_pct = (1.0 - avg_post_peak_factor) * 100

            regressing_seeds.append({
                "file": fname,
                "peak_r2": peak_r2,
                "final_r2": final_r2,
                "drop": drop,
                "peak_fraction": peak_fraction,
                "lr_reduction_pct": lr_reduction_pct,
            })

    if total_seeds < 20:
        print(f"FAIL: only {total_seeds} trajectories available, need >= 20")
        return False

    if len(regressing_seeds) < 3:
        print(f"FAIL: only {len(regressing_seeds)} regressing seeds found (need >= 3 for statistical evidence)")
        return False

    # Check: at least 60% of regressing seeds have peak in first 50% of training
    # (meaning lr decay would have been > 30% during regression phase)
    early_peak_count = sum(1 for s in regressing_seeds if s["peak_fraction"] <= 0.50)
    early_peak_frac = early_peak_count / len(regressing_seeds)

    # Also check average lr reduction in post-peak phase
    avg_lr_reduction = sum(s["lr_reduction_pct"] for s in regressing_seeds) / len(regressing_seeds)

    print(f"  Regression analysis: {len(regressing_seeds)}/{total_seeds} seeds regress (drop>{regression_threshold})")
    print(f"  Seeds with peak in first 50%: {early_peak_count}/{len(regressing_seeds)} ({early_peak_frac:.1%})")
    print(f"  Average post-peak lr reduction with cosine decay: {avg_lr_reduction:.1f}%")

    # Show top 5 regressing seeds
    regressing_seeds.sort(key=lambda x: -x["drop"])
    for s in regressing_seeds[:5]:
        print(f"    {s['file']}: peak={s['peak_r2']:.4f} final={s['final_r2']:.4f} "
              f"drop={s['drop']:.4f} peak@{s['peak_fraction']:.1%} lr_red={s['lr_reduction_pct']:.0f}%")

    if early_peak_frac < 0.60:
        print(f"FAIL: only {early_peak_frac:.1%} of regressing seeds have early peaks (need >= 60%)")
        return False

    if avg_lr_reduction < 30:
        print(f"FAIL: average lr reduction {avg_lr_reduction:.1f}% < 30% (insufficient decay in regression phase)")
        return False

    print(f"  Regression prevention evidence: cosine decay would reduce lr by {avg_lr_reduction:.1f}% "
          f"during regression phase ✓")
    return True


def main() -> int:
    print("=" * 60)
    print("Test: lr_W cosine decay schedule")
    print("=" * 60)

    tests = [
        ("Mathematical correctness", test_mathematical_correctness),
        ("Optimizer integration", test_optimizer_integration),
        ("Monotonicity", test_monotonicity),
        ("Regression prevention evidence", test_regression_prevention_evidence),
    ]

    all_passed = True
    for name, test_fn in tests:
        print(f"\n[{name}]")
        if not test_fn():
            all_passed = False
            # Don't short-circuit; run all tests for diagnostics

    print("\n" + "=" * 60)
    if all_passed:
        print("PASS: lr_W cosine decay is mathematically correct, monotonic, integrates with "
              "PyTorch optimizer, and trajectory evidence confirms regression occurs in the "
              "phase where cosine decay would reduce lr by >30%")
        return 0
    else:
        print("FAIL: one or more sub-tests failed (see above)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
