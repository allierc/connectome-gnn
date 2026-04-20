"""Test for apply_differential_warmup.

Verifies:
1. W groups maintain constant LR multiplier = 1.0 at all steps.
2. MLP/embedding groups start at warmup_start_fraction and reach 1.0 at warmup_steps.
3. Scheduler integrates correctly with a real optimizer on FlyVis-scale parameters.
4. The warmup creates a meaningful LR differential between W and MLP groups during early steps.
"""

import sys
import torch

from connectome_gnn.LLM_code.staging.block_06.differential_warmup import (
    apply_differential_warmup,
)


def build_mock_optimizer():
    """Build an optimizer with param groups mimicking the FlyVis model structure."""
    # Simulate realistic parameter shapes for flyvis (13741 neurons, 434112 edges)
    params_W = torch.randn(434112, requires_grad=True)
    params_g_phi = torch.randn(80, 80, requires_grad=True)
    params_f_theta = torch.randn(80, 80, requires_grad=True)
    params_embedding = torch.randn(13741, 2, requires_grad=True)

    param_groups = [
        {"params": [params_W], "lr": 0.0006, "name": "W", "base_lr": 0.0006},
        {"params": [params_g_phi], "lr": 0.0012, "name": "g_phi", "base_lr": 0.0012},
        {"params": [params_f_theta], "lr": 0.0012, "name": "f_theta", "base_lr": 0.0012},
        {"params": [params_embedding], "lr": 0.00155, "name": "embedding", "base_lr": 0.00155},
    ]
    optimizer = torch.optim.Adam(param_groups)
    return optimizer


def test_constant_W_lr():
    """W groups must have multiplier 1.0 at all steps."""
    optimizer = build_mock_optimizer()
    scheduler = apply_differential_warmup(optimizer, config=None, warmup_steps=1000, warmup_start_fraction=0.01)

    for step in [0, 1, 100, 500, 999, 1000, 2000]:
        # Reset scheduler state to test specific step
        scheduler.last_epoch = step
        lrs = scheduler.get_lr()
        # Group 0 is W — should always be base_lr * 1.0
        w_lr = lrs[0]
        expected = 0.0006
        if abs(w_lr - expected) > 1e-10:
            print(f"FAIL: W lr at step {step} is {w_lr}, expected {expected}")
            sys.exit(1)


def test_warmup_start():
    """MLP/embedding groups start at warmup_start_fraction of their base LR."""
    optimizer = build_mock_optimizer()
    scheduler = apply_differential_warmup(optimizer, config=None, warmup_steps=1000, warmup_start_fraction=0.01)

    # At step 0, the multiplier should be 0.01
    scheduler.last_epoch = 0
    lrs = scheduler.get_lr()
    # Groups 1,2,3 are g_phi, f_theta, embedding
    for i, (name, base) in enumerate(
        [("g_phi", 0.0012), ("f_theta", 0.0012), ("embedding", 0.00155)], start=1
    ):
        expected = base * 0.01
        actual = lrs[i]
        if abs(actual - expected) / expected > 0.01:
            print(f"FAIL: {name} lr at step 0 is {actual:.8f}, expected {expected:.8f}")
            sys.exit(1)


def test_warmup_end():
    """MLP/embedding groups reach full LR at warmup_steps."""
    optimizer = build_mock_optimizer()
    scheduler = apply_differential_warmup(optimizer, config=None, warmup_steps=1000, warmup_start_fraction=0.01)

    scheduler.last_epoch = 1000
    lrs = scheduler.get_lr()
    for i, (name, base) in enumerate(
        [("g_phi", 0.0012), ("f_theta", 0.0012), ("embedding", 0.00155)], start=1
    ):
        actual = lrs[i]
        if abs(actual - base) / base > 1e-6:
            print(f"FAIL: {name} lr at step 1000 is {actual:.8f}, expected {base:.8f}")
            sys.exit(1)


def test_warmup_midpoint():
    """At step 500 (midpoint), multiplier should be ~0.505."""
    optimizer = build_mock_optimizer()
    scheduler = apply_differential_warmup(optimizer, config=None, warmup_steps=1000, warmup_start_fraction=0.01)

    scheduler.last_epoch = 500
    lrs = scheduler.get_lr()
    # Expected multiplier at step 500: 0.01 + 0.99 * (500/1000) = 0.505
    expected_mult = 0.01 + 0.99 * 0.5
    for i, (name, base) in enumerate(
        [("g_phi", 0.0012), ("f_theta", 0.0012), ("embedding", 0.00155)], start=1
    ):
        expected = base * expected_mult
        actual = lrs[i]
        if abs(actual - expected) / expected > 0.01:
            print(f"FAIL: {name} lr at step 500 is {actual:.8f}, expected {expected:.8f}")
            sys.exit(1)


def test_differential_ratio():
    """During warmup, effective W lr should be much higher than MLP lr (the whole point).

    At step 0: W effective = 0.0006, g_phi effective = 0.0012 * 0.01 = 0.000012
    Ratio W/g_phi = 50x. This is the "head start" for W.
    At step 1000+: W = 0.0006, g_phi = 0.0012. Ratio = 0.5x (normal).
    """
    optimizer = build_mock_optimizer()
    scheduler = apply_differential_warmup(optimizer, config=None, warmup_steps=1000, warmup_start_fraction=0.01)

    # Step 0: W should dominate
    scheduler.last_epoch = 0
    lrs = scheduler.get_lr()
    w_lr = lrs[0]
    gphi_lr = lrs[1]
    ratio_start = w_lr / gphi_lr
    if ratio_start < 40:
        print(f"FAIL: W/g_phi ratio at step 0 is {ratio_start:.1f}, expected >= 40")
        sys.exit(1)

    # Step 1000: normal ratio restored
    scheduler.last_epoch = 1000
    lrs = scheduler.get_lr()
    w_lr = lrs[0]
    gphi_lr = lrs[1]
    ratio_end = w_lr / gphi_lr
    if abs(ratio_end - 0.5) > 0.01:
        print(f"FAIL: W/g_phi ratio at step 1000 is {ratio_end:.3f}, expected 0.5")
        sys.exit(1)


def test_step_integration():
    """Test that scheduler.step() correctly advances LR through warmup."""
    optimizer = build_mock_optimizer()
    scheduler = apply_differential_warmup(optimizer, config=None, warmup_steps=100, warmup_start_fraction=0.01)

    # Collect effective LRs at various steps
    w_lrs = []
    gphi_lrs = []

    for step in range(150):
        # Record current LRs
        w_lrs.append(optimizer.param_groups[0]["lr"])
        gphi_lrs.append(optimizer.param_groups[1]["lr"])
        # Simulate optimizer step (dummy backward)
        optimizer.zero_grad()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                p.grad = torch.zeros_like(p)
        optimizer.step()
        scheduler.step()

    # W should stay constant throughout
    w_base = 0.0006
    for i, lr in enumerate(w_lrs):
        if abs(lr - w_base) / w_base > 0.01:
            print(f"FAIL: W lr drifted at step {i}: {lr:.8f} vs expected {w_base:.8f}")
            sys.exit(1)

    # g_phi should start low and increase
    if gphi_lrs[0] > 0.0012 * 0.05:
        print(f"FAIL: g_phi lr at step 0 too high: {gphi_lrs[0]:.8f}")
        sys.exit(1)
    if gphi_lrs[100] < 0.0012 * 0.95:
        print(f"FAIL: g_phi lr at step 100 too low: {gphi_lrs[100]:.8f}")
        sys.exit(1)


def test_post_warmup_constant():
    """After warmup_steps, MLP LRs should stay at 1.0 multiplier (no decay)."""
    optimizer = build_mock_optimizer()
    scheduler = apply_differential_warmup(optimizer, config=None, warmup_steps=100, warmup_start_fraction=0.01)

    # Advance well past warmup
    for _ in range(200):
        optimizer.zero_grad()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                p.grad = torch.zeros_like(p)
        optimizer.step()
        scheduler.step()

    # At step 200, multiplier should be 1.0 (not decaying)
    scheduler_lrs = scheduler.get_lr()
    for i, (name, base) in enumerate(
        [("W", 0.0006), ("g_phi", 0.0012), ("f_theta", 0.0012), ("embedding", 0.00155)]
    ):
        expected = base
        actual = scheduler_lrs[i]
        if abs(actual - expected) / expected > 0.02:
            print(f"FAIL: {name} lr at step 200 is {actual:.8f}, expected {expected:.8f} (no decay)")
            sys.exit(1)


def test_invalid_args():
    """Verify input validation."""
    optimizer = build_mock_optimizer()

    try:
        apply_differential_warmup(optimizer, config=None, warmup_steps=0)
        print("FAIL: Should have raised ValueError for warmup_steps=0")
        sys.exit(1)
    except ValueError:
        pass

    try:
        apply_differential_warmup(optimizer, config=None, warmup_start_fraction=1.5)
        print("FAIL: Should have raised ValueError for warmup_start_fraction=1.5")
        sys.exit(1)
    except ValueError:
        pass


if __name__ == "__main__":
    test_constant_W_lr()
    test_warmup_start()
    test_warmup_end()
    test_warmup_midpoint()
    test_differential_ratio()
    test_step_integration()
    test_post_warmup_constant()
    test_invalid_args()
    print("PASS: Differential warmup scheduler correctly holds W at full LR while ramping MLP/embedding from 1% to 100% over warmup window")
