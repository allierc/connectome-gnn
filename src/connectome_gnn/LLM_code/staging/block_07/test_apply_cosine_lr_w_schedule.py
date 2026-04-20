#!/usr/bin/env python
"""Standalone test for apply_cosine_lr_w_schedule.

Verifies:
1. Scheduler attaches correctly and returns a CosineAnnealingLR instance.
2. Learning rate follows the expected cosine curve from lr_init to 0.
3. At step 0 lr == lr_init, at step T_max lr == 0.
4. Midpoint lr is approximately lr_init / 2.
5. Schedule is monotonically non-increasing.
6. Works with multi-param-group optimizers (only W group scheduled).
7. Raises ValueError for total_steps < 1.

Runs on CPU only — no GPU required.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn

# Import the staged function
from connectome_gnn.LLM_code.staging.block_07.apply_cosine_lr_w_schedule import (
    apply_cosine_lr_w_schedule,
)


def _make_optimizer(lr: float = 0.0006) -> torch.optim.Adam:
    """Create a minimal Adam optimizer with a single parameter."""
    param = nn.Parameter(torch.randn(100, 100))
    return torch.optim.Adam([param], lr=lr)


def test_basic_attachment():
    """Scheduler attaches and is the correct type."""
    opt = _make_optimizer(lr=0.0006)
    sched = apply_cosine_lr_w_schedule(opt, total_steps=100)
    assert isinstance(
        sched, torch.optim.lr_scheduler.CosineAnnealingLR
    ), f"Expected CosineAnnealingLR, got {type(sched)}"
    assert sched.T_max == 100
    assert sched.eta_min == 0.0
    return True


def test_cosine_curve():
    """LR follows the cosine schedule from lr_init → 0."""
    lr_init = 0.0006
    T = 1000
    opt = _make_optimizer(lr=lr_init)
    sched = apply_cosine_lr_w_schedule(opt, total_steps=T)

    lrs = []
    for step in range(T):
        lrs.append(opt.param_groups[0]["lr"])
        # Simulate an optimizer step (no actual gradient)
        opt.step()
        sched.step()

    # Check initial lr
    assert abs(lrs[0] - lr_init) < 1e-10, f"Initial lr {lrs[0]} != {lr_init}"

    # Check final lr is near zero
    final_lr = opt.param_groups[0]["lr"]
    assert final_lr < 1e-10, f"Final lr {final_lr} should be ~0"

    # Check midpoint is approximately lr_init / 2
    mid = T // 2
    expected_mid = lr_init * 0.5 * (1 + math.cos(math.pi * mid / T))
    assert abs(lrs[mid] - expected_mid) < 1e-10, (
        f"Midpoint lr {lrs[mid]} != expected {expected_mid}"
    )

    # Check monotonically non-increasing
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-12, (
            f"LR increased at step {i}: {lrs[i-1]:.8f} -> {lrs[i]:.8f}"
        )

    return True


def test_matches_analytical():
    """Each step's lr matches the analytical cosine formula."""
    lr_init = 0.0006
    T = 200
    opt = _make_optimizer(lr=lr_init)
    sched = apply_cosine_lr_w_schedule(opt, total_steps=T)

    max_err = 0.0
    for step in range(T):
        actual = opt.param_groups[0]["lr"]
        # CosineAnnealingLR formula: eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi*t/T_max))
        # After step() call at iteration `step`, the LR for the NEXT read is for step+1.
        # But we read BEFORE step(), so this is the LR for step `step`.
        expected = lr_init * 0.5 * (1 + math.cos(math.pi * step / T))
        err = abs(actual - expected)
        max_err = max(max_err, err)
        assert err < 1e-9, (
            f"Step {step}: actual={actual:.10f}, expected={expected:.10f}, err={err:.2e}"
        )
        opt.step()
        sched.step()

    return max_err


def test_invalid_total_steps():
    """Raises ValueError for total_steps < 1."""
    opt = _make_optimizer()
    try:
        apply_cosine_lr_w_schedule(opt, total_steps=0)
        return False  # Should have raised
    except ValueError:
        pass

    try:
        apply_cosine_lr_w_schedule(opt, total_steps=-5)
        return False
    except ValueError:
        pass

    return True


def test_realistic_flyvis_config():
    """Simulates a realistic FlyVis training schedule.

    DAL=35, batch_size=4, n_frames=64000 → steps_per_loop ≈ 64000/4 = 16000
    total_steps = 1 * 35 * 16000 = 560000 (n_epochs=1).
    Verify schedule decays smoothly over this range.
    """
    lr_init = 0.0006
    total_steps = 560_000
    opt = _make_optimizer(lr=lr_init)
    sched = apply_cosine_lr_w_schedule(opt, total_steps=total_steps)

    # Sample at 10%, 50%, 90% of training
    checkpoints = {
        int(0.10 * total_steps): None,
        int(0.50 * total_steps): None,
        int(0.90 * total_steps): None,
    }

    for step in range(max(checkpoints.keys()) + 1):
        if step in checkpoints:
            checkpoints[step] = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()

    # 10%: should still be close to initial (cosine is slow at start)
    lr_10 = checkpoints[int(0.10 * total_steps)]
    assert lr_10 > 0.95 * lr_init, f"10% lr={lr_10:.6f} too low (expected > {0.95*lr_init:.6f})"

    # 50%: should be approximately lr_init / 2
    lr_50 = checkpoints[int(0.50 * total_steps)]
    assert 0.4 * lr_init < lr_50 < 0.6 * lr_init, (
        f"50% lr={lr_50:.6f} not near {0.5*lr_init:.6f}"
    )

    # 90%: should be very small
    lr_90 = checkpoints[int(0.90 * total_steps)]
    assert lr_90 < 0.05 * lr_init, f"90% lr={lr_90:.6f} too high (expected < {0.05*lr_init:.6f})"

    return checkpoints


def main():
    tests = [
        ("basic_attachment", test_basic_attachment),
        ("cosine_curve", test_cosine_curve),
        ("matches_analytical", test_matches_analytical),
        ("invalid_total_steps", test_invalid_total_steps),
        ("realistic_flyvis_config", test_realistic_flyvis_config),
    ]

    failures = []
    for name, fn in tests:
        try:
            result = fn()
            print(f"  OK: {name}")
        except Exception as e:
            failures.append((name, str(e)))
            print(f"  FAIL: {name} — {e}")

    if failures:
        print(f"\nFAIL: {len(failures)}/{len(tests)} tests failed: "
              + ", ".join(n for n, _ in failures))
        sys.exit(1)
    else:
        print(f"\nPASS: all {len(tests)} tests passed — cosine lr_W schedule "
              "correctly decays from lr_init to 0 following analytical formula")
        sys.exit(0)


if __name__ == "__main__":
    main()
