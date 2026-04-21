"""Standalone test for cosine_lr_w_schedule.

Verifies that the cosine LR schedule:
1. Decays the W param group from initial lr to ~0 following a cosine curve
2. Leaves all other param groups (g_phi, f_theta, embedding) at constant LR
3. Reaches the correct value at key checkpoints (25%, 50%, 75%, 100%)
4. The schedule shape is monotonically non-increasing for W
"""

import math
import sys
import types

import torch


def _make_mock_optimizer():
    """Create a mock optimizer with named param groups matching the real
    pipeline (set_trainable_parameters in models/utils.py)."""
    groups = [
        {"name": "g_phi", "lr": 0.0012},
        {"name": "f_theta", "lr": 0.0012},
        {"name": "W", "lr": 0.0006},
        {"name": "embedding", "lr": 0.00155},
    ]
    # Create tiny dummy params for each group
    for g in groups:
        p = torch.nn.Parameter(torch.zeros(1))
        g["params"] = [p]

    return torch.optim.Adam(groups)


def _make_mock_config(n_frames=64000, dal=35, batch_size=4):
    """Create a minimal config namespace matching what the function reads."""
    config = types.SimpleNamespace()
    config.training = types.SimpleNamespace(
        data_augmentation_loop=dal,
        batch_size=batch_size,
    )
    config.simulation = types.SimpleNamespace(n_frames=n_frames)
    return config


def main():
    from connectome_gnn.LLM_code.staging.block_08.cosine_lr_w_schedule import (
        apply_cosine_lr_w_schedule,
    )

    # --- Setup ---
    optimizer = _make_mock_optimizer()
    config = _make_mock_config(n_frames=64000, dal=35, batch_size=4)
    total_steps = int(64000 * 35 // 4 * 0.2)  # = 112000

    scheduler = apply_cosine_lr_w_schedule(optimizer, config)

    # Identify param group indices by name
    group_idx = {g["name"]: i for i, g in enumerate(optimizer.param_groups)}
    w_idx = group_idx["W"]
    initial_lr_w = 0.0006
    other_groups = {name: idx for name, idx in group_idx.items() if name != "W"}
    initial_lrs = {name: optimizer.param_groups[idx]["lr"] for name, idx in other_groups.items()}

    # --- Test 1: checkpoint values ---
    # Step through and record W lr at key points
    checkpoints = {
        0: 1.0,                          # cos(0) = 1.0 → multiplier = 1.0
        total_steps // 4: 0.5 * (1.0 + math.cos(math.pi * 0.25)),   # ~0.854
        total_steps // 2: 0.5 * (1.0 + math.cos(math.pi * 0.5)),    # = 0.5
        3 * total_steps // 4: 0.5 * (1.0 + math.cos(math.pi * 0.75)),  # ~0.146
        total_steps: 0.0,                 # cos(pi) = -1 → multiplier = 0.0
    }

    # We'll step through in chunks to check key points
    # LambdaLR starts at step 0 (after first .step() it becomes step 1)
    # Before any .step() call, the LR is already set to lambda(0)
    current_step = 0
    sorted_checkpoints = sorted(checkpoints.items())

    tol = 1e-6

    for target_step, expected_mult in sorted_checkpoints:
        while current_step < target_step:
            scheduler.step()
            current_step += 1

        actual_lr_w = optimizer.param_groups[w_idx]["lr"]
        expected_lr_w = initial_lr_w * expected_mult

        if abs(actual_lr_w - expected_lr_w) > tol:
            print(
                f"FAIL: W lr at step {target_step}: expected {expected_lr_w:.8f}, "
                f"got {actual_lr_w:.8f} (mult expected={expected_mult:.4f})"
            )
            sys.exit(1)

    # --- Test 2: other groups stay constant ---
    for name, idx in other_groups.items():
        actual = optimizer.param_groups[idx]["lr"]
        expected = initial_lrs[name]
        if abs(actual - expected) > tol:
            print(
                f"FAIL: {name} lr changed from {expected} to {actual} "
                f"after {total_steps} steps (should be constant)"
            )
            sys.exit(1)

    # --- Test 3: monotonically non-increasing W lr ---
    # Reset and step through, sampling every 1000 steps
    optimizer2 = _make_mock_optimizer()
    scheduler2 = apply_cosine_lr_w_schedule(optimizer2, config)
    w_idx2 = next(
        i for i, g in enumerate(optimizer2.param_groups) if g["name"] == "W"
    )

    prev_lr = optimizer2.param_groups[w_idx2]["lr"]
    sample_every = 1000
    for step in range(1, total_steps + 1):
        scheduler2.step()
        if step % sample_every == 0:
            cur_lr = optimizer2.param_groups[w_idx2]["lr"]
            if cur_lr > prev_lr + tol:
                print(
                    f"FAIL: W lr increased from {prev_lr:.8f} to {cur_lr:.8f} "
                    f"at step {step} (should be monotonically non-increasing)"
                )
                sys.exit(1)
            prev_lr = cur_lr

    # --- Test 4: final W lr is effectively 0 ---
    final_lr = optimizer2.param_groups[w_idx2]["lr"]
    if final_lr > 1e-10:
        print(f"FAIL: final W lr = {final_lr:.2e}, expected ~0")
        sys.exit(1)

    # --- Test 5: explicit total_steps override ---
    optimizer3 = _make_mock_optimizer()
    custom_steps = 500
    scheduler3 = apply_cosine_lr_w_schedule(
        optimizer3, config, total_steps=custom_steps
    )
    for _ in range(custom_steps):
        scheduler3.step()
    final_lr3 = optimizer3.param_groups[
        next(i for i, g in enumerate(optimizer3.param_groups) if g["name"] == "W")
    ]["lr"]
    if final_lr3 > 1e-10:
        print(f"FAIL: W lr with custom total_steps={custom_steps}: {final_lr3:.2e}")
        sys.exit(1)

    # --- Test 6: ValueError on bad total_steps ---
    try:
        apply_cosine_lr_w_schedule(_make_mock_optimizer(), config, total_steps=0)
        print("FAIL: should raise ValueError for total_steps=0")
        sys.exit(1)
    except ValueError:
        pass

    print(
        f"PASS: cosine lr_W schedule decays W from {initial_lr_w} to 0 over "
        f"{total_steps} steps; other groups constant; monotonically non-increasing"
    )


if __name__ == "__main__":
    main()
