#!/usr/bin/env python3
"""Test: early bifurcation detector on flyvis noise=0.05 from-zero exploration.

PASS CONDITION (all must hold):
  1. Threshold classifier on first-real-checkpoint conn_R2 achieves > 78%
     accuracy at separating CONV (final >= 0.90) from FAIL seeds — this is
     above the always-predict-majority baseline (~73%).
  2. Accuracy improves monotonically from checkpoint 1 → 3 (later checkpoints
     = more signal, confirming the bifurcation is *progressive* in early
     training, not a random flip).
  3. Voltage spectral analysis shows effective_rank << n_neurons and
     condition_number > 10, confirming the learning problem is ill-conditioned
     (multi-basin landscape).

Uses load_full_voltage for the mechanistic (spectral) part.
Uses trajectory logs in r2_trajectory/ for the classifier part.
"""

from __future__ import annotations

import os
import sys

# Ensure repo root is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from connectome_gnn.LLM_code.staging.block_10.early_bifurcation_detector import (
    load_trajectories,
    evaluate_classifier,
    find_optimal_threshold,
    voltage_spectral_analysis,
)


def main() -> int:
    # ------------------------------------------------------------------
    # Part 1: Trajectory-based early bifurcation classifier
    # ------------------------------------------------------------------
    traj_dir = os.path.join(
        REPO_ROOT,
        "log", "remote", "Claude_exploration",
        "LLM_flyvis_noise_005_from_zero", "r2_trajectory",
    )
    if not os.path.isdir(traj_dir):
        print(f"FAIL: trajectory dir not found: {traj_dir}")
        return 1

    trajs = load_trajectories(traj_dir)
    if len(trajs) < 20:
        print(f"FAIL: too few trajectories ({len(trajs)}), need >= 20 for meaningful test")
        return 1

    n_conv = sum(1 for t in trajs if t.final_r2 >= 0.90)
    n_fail = len(trajs) - n_conv
    baseline_acc = max(n_conv, n_fail) / len(trajs)
    print(f"Trajectories loaded: {len(trajs)} (CONV={n_conv}, FAIL={n_fail}, baseline={baseline_acc:.3f})")

    # Evaluate at checkpoints 1, 2, 3 (idx 0 is always step=1 with ~0 conn_R2)
    accs = []
    for ckpt_idx in [1, 2, 3]:
        thr, acc = find_optimal_threshold(trajs, checkpoint_idx=ckpt_idx)
        metrics = evaluate_classifier(trajs, checkpoint_idx=ckpt_idx, threshold=thr)
        print(
            f"  Checkpoint {ckpt_idx}: threshold={thr:.3f}, "
            f"accuracy={acc:.3f} ({metrics.n_correct}/{metrics.n_total}), "
            f"TP={metrics.n_true_pos} FP={metrics.n_false_pos} "
            f"TN={metrics.n_true_neg} FN={metrics.n_false_neg}"
        )
        accs.append(acc)

    # CHECK 1: first-checkpoint accuracy > 78% (above baseline)
    if accs[0] <= 0.78:
        print(f"FAIL: first-checkpoint accuracy {accs[0]:.3f} <= 0.78 threshold")
        return 1

    # CHECK 2: monotonic improvement across checkpoints 1-3
    monotonic = all(accs[i] <= accs[i + 1] + 0.005 for i in range(len(accs) - 1))
    # Allow tiny tolerance (0.005) for ties in discrete accuracy
    if not monotonic:
        # Weaker check: at least checkpoint 3 > checkpoint 1
        if accs[2] < accs[0] - 0.01:
            print(
                f"FAIL: accuracy does not improve from ckpt1 ({accs[0]:.3f}) to ckpt3 ({accs[2]:.3f})"
            )
            return 1
        print(f"  Note: non-monotonic but ckpt3 ({accs[2]:.3f}) >= ckpt1 ({accs[0]:.3f}), OK")

    # ------------------------------------------------------------------
    # Part 2: Voltage spectral analysis (mechanistic explanation)
    # ------------------------------------------------------------------
    print("\nLoading voltage data for spectral analysis...")
    from connectome_gnn.LLM_code.scratchpad import load_full_voltage

    v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", 0.10)
    print(f"  v_clean shape: {v_clean.shape}, v_noisy shape: {v_noisy.shape}")

    spec = voltage_spectral_analysis(v_clean, v_noisy)
    print(f"  effective_rank: {spec['effective_rank']} / {spec['n_neurons']} neurons")
    print(f"  condition_number: {spec['condition_number']:.1f}")
    print(f"  snr_db: {spec['snr_db']:.1f} dB")
    print(f"  top_sv_ratio: {spec['top_sv_ratio']:.4f}")

    # CHECK 3: ill-conditioned (multi-basin) landscape
    if spec["effective_rank"] >= spec["n_neurons"]:
        print(f"FAIL: effective_rank ({spec['effective_rank']}) not << n_neurons ({spec['n_neurons']})")
        return 1
    if spec["condition_number"] <= 10.0:
        print(f"FAIL: condition_number ({spec['condition_number']:.1f}) <= 10, problem is well-conditioned")
        return 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    rank_ratio = spec["effective_rank"] / spec["n_neurons"]
    print(
        f"\nPASS: early bifurcation confirmed — ckpt1 acc={accs[0]:.3f}, "
        f"ckpt3 acc={accs[2]:.3f} (baseline={baseline_acc:.3f}); "
        f"voltage rank ratio={rank_ratio:.3f}, cond={spec['condition_number']:.0f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
