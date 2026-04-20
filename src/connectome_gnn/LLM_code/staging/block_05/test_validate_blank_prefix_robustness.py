"""Test validate_blank_prefix_robustness on flyvis data.

Tests both modes:
1. Precondition mode: uses full flyvis voltage/stimulus data to verify
   that blanking the first 10% creates a W-dominated training regime.
2. Post-training mode: uses synthetic trajectory files to verify the
   parsing and PASS-condition logic works correctly.

PASS: both the mechanistic precondition and the validation logic are sound.
FAIL: either the precondition doesn't hold or validation logic is broken.
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, "/workspace/connectome-gnn/src")

from connectome_gnn.LLM_code.staging.block_05.validate_blank_prefix_robustness import (
    validate_blank_prefix_robustness,
    _has_early_decline,
    _validate_post_training,
)
import numpy as np


def test_precondition():
    """Validate mechanistic precondition using real flyvis voltage data."""
    # Use a non-existent log_dir to trigger precondition mode
    result = validate_blank_prefix_robustness(
        log_dir="/tmp/nonexistent_blank_prefix_test",
        n_seeds=8,
        dal=35,
        blank_prefix_fraction=0.1,
    )

    assert result["mode"] == "precondition", f"Expected precondition mode, got {result['mode']}"
    details = result["details"]

    # Check that voltage dynamics exist during blank period
    if not details.get("cond_a_dynamics_exist"):
        return False, (
            f"Voltage has no variance during blank period "
            f"(var={details.get('mean_voltage_var_blank_period', 0):.2e})"
        )

    # Check that active dynamics (dv/dt) are non-trivial
    if not details.get("cond_b_active_dynamics"):
        return False, (
            f"No active dynamics during blank period "
            f"(dv/dt RMS={details.get('dv_dt_rms_blank', 0):.2e})"
        )

    # Verify the data dimensions match expected flyvis
    if details.get("N") != 13741:
        return False, f"Unexpected neuron count: {details.get('N')} (expected 13741)"
    if details.get("T") != 64000:
        return False, f"Unexpected timestep count: {details.get('T')} (expected 64000)"
    if details.get("blank_end_frame") != 6400:
        return False, f"Unexpected blank end: {details.get('blank_end_frame')} (expected 6400)"

    return True, details["interpretation"]


def test_post_training_pass():
    """Test that validation logic correctly identifies a PASS case."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_dir = os.path.join(tmpdir, "r2_trajectory")
        os.makedirs(traj_dir)

        # Create 8 synthetic trajectory files that PASS all conditions:
        # - All final conn_R2 >= 0.93 (no catastrophic)
        # - Mean >= 0.93
        # - No early decline
        for i in range(8):
            path = os.path.join(traj_dir, f"seed_{i:03d}.log")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["iteration", "connectivity_r2", "vrest_r2", "tau_r2"])
                writer.writeheader()
                # Monotonically increasing trajectory: 0.5 -> 0.95
                for step in range(0, 100001, 10000):
                    r2 = 0.50 + 0.45 * (step / 100000) + np.random.uniform(-0.005, 0.005)
                    writer.writerow({
                        "iteration": step,
                        "connectivity_r2": f"{r2:.6f}",
                        "vrest_r2": "0.0",
                        "tau_r2": "0.0",
                    })

        result = _validate_post_training(traj_dir, n_seeds=8)

        if not result["passed"]:
            return False, f"Pass case failed: {result}"
        if result["n_catastrophic"] != 0:
            return False, f"Expected 0 catastrophic, got {result['n_catastrophic']}"
        if result["mean_conn_r2"] < 0.93:
            return False, f"Mean R2 too low: {result['mean_conn_r2']}"

    return True, "Post-training PASS case validated correctly"


def test_post_training_fail_catastrophic():
    """Test that validation logic correctly identifies catastrophic failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_dir = os.path.join(tmpdir, "r2_trajectory")
        os.makedirs(traj_dir)

        for i in range(8):
            path = os.path.join(traj_dir, f"seed_{i:03d}.log")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["iteration", "connectivity_r2", "vrest_r2", "tau_r2"])
                writer.writeheader()
                # Seed 0: catastrophic collapse (peaks then crashes)
                if i == 0:
                    for step_idx, step in enumerate(range(0, 100001, 10000)):
                        if step <= 30000:
                            r2 = 0.5 + 0.45 * (step / 30000)
                        else:
                            r2 = 0.95 - 0.6 * ((step - 30000) / 70000)  # collapse to ~0.35
                        writer.writerow({
                            "iteration": step,
                            "connectivity_r2": f"{r2:.6f}",
                            "vrest_r2": "0.0",
                            "tau_r2": "0.0",
                        })
                else:
                    # Good seeds
                    for step in range(0, 100001, 10000):
                        r2 = 0.50 + 0.45 * (step / 100000)
                        writer.writerow({
                            "iteration": step,
                            "connectivity_r2": f"{r2:.6f}",
                            "vrest_r2": "0.0",
                            "tau_r2": "0.0",
                        })

        result = _validate_post_training(traj_dir, n_seeds=8)

        if result["passed"]:
            return False, "Should have FAILED due to catastrophic seed but passed"
        if result["n_catastrophic"] != 1:
            return False, f"Expected 1 catastrophic, got {result['n_catastrophic']}"

    return True, "Catastrophic failure detection works correctly"


def test_post_training_fail_early_decline():
    """Test that validation logic detects early decline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_dir = os.path.join(tmpdir, "r2_trajectory")
        os.makedirs(traj_dir)

        for i in range(8):
            path = os.path.join(traj_dir, f"seed_{i:03d}.log")
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["iteration", "connectivity_r2", "vrest_r2", "tau_r2"])
                writer.writeheader()
                if i == 2:
                    # Seed with early decline: drops > 0.02 before step 60k
                    trajectory = [
                        (0, 0.50), (10000, 0.75), (20000, 0.85),
                        (30000, 0.90), (40000, 0.87),  # decline of 0.03 at step 40k
                        (50000, 0.91), (60000, 0.93),
                        (70000, 0.94), (80000, 0.95), (90000, 0.95), (100000, 0.95),
                    ]
                    for step, r2 in trajectory:
                        writer.writerow({
                            "iteration": step,
                            "connectivity_r2": f"{r2:.6f}",
                            "vrest_r2": "0.0",
                            "tau_r2": "0.0",
                        })
                else:
                    # Good monotonic seeds
                    for step in range(0, 100001, 10000):
                        r2 = 0.50 + 0.45 * (step / 100000)
                        writer.writerow({
                            "iteration": step,
                            "connectivity_r2": f"{r2:.6f}",
                            "vrest_r2": "0.0",
                            "tau_r2": "0.0",
                        })

        result = _validate_post_training(traj_dir, n_seeds=8)

        if result["passed"]:
            return False, "Should have FAILED due to early decline but passed"
        if result["n_early_declines"] < 1:
            return False, f"Expected at least 1 early decline, got {result['n_early_declines']}"

    return True, "Early decline detection works correctly"


def test_early_decline_helper():
    """Unit test the _has_early_decline helper."""
    # Case 1: No decline
    steps = np.array([0, 10000, 20000, 30000, 40000, 50000])
    r2 = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.90])
    has, _, _ = _has_early_decline(steps, r2)
    if has:
        return False, "False positive: monotonic trajectory flagged as decline"

    # Case 2: Decline before 60k
    steps = np.array([0, 10000, 20000, 30000, 40000, 50000])
    r2 = np.array([0.5, 0.7, 0.85, 0.82, 0.88, 0.90])  # drop of 0.03 at step 30k
    has, step, mag = _has_early_decline(steps, r2)
    if not has:
        return False, "Missed decline of 0.03 at step 30k"
    if step != 30000:
        return False, f"Wrong decline step: {step} (expected 30000)"

    # Case 3: Decline AFTER 60k (should not flag)
    steps = np.array([0, 20000, 40000, 60000, 80000, 100000])
    r2 = np.array([0.5, 0.7, 0.85, 0.90, 0.87, 0.85])  # decline at 80k
    has, _, _ = _has_early_decline(steps, r2)
    if has:
        return False, "False positive: decline after 60k flagged"

    return True, "Early decline helper works correctly"


def main():
    tests = [
        ("early_decline_helper", test_early_decline_helper),
        ("post_training_pass", test_post_training_pass),
        ("post_training_fail_catastrophic", test_post_training_fail_catastrophic),
        ("post_training_fail_early_decline", test_post_training_fail_early_decline),
        ("precondition", test_precondition),
    ]

    all_passed = True
    summaries = []

    for name, test_fn in tests:
        try:
            passed, msg = test_fn()
            status = "OK" if passed else "FAILED"
            if not passed:
                all_passed = False
            summaries.append(f"  [{status}] {name}: {msg}")
        except Exception as e:
            all_passed = False
            summaries.append(f"  [ERROR] {name}: {type(e).__name__}: {e}")

    print("Test results:")
    for s in summaries:
        print(s)
    print()

    if all_passed:
        print("PASS: blank_prefix_fraction=0.1 precondition validated — voltage dynamics exist "
              "during blank period (W-dominated regime) and validation logic is correct")
    else:
        failed = [s for s in summaries if "[FAILED]" in s or "[ERROR]" in s]
        print(f"FAIL: {len(failed)} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
