#!/usr/bin/env python
"""Global SVD-based null space analysis (Step 1).

Computes the effective rank and null space dimension using population-level SVD.

For the entire population activity matrix H (T × N), we:
1. Subsample neurons (every 14th) and timesteps (every 8th)
2. Compute full SVD: H_sub = U Sigma V^T
3. Find effective rank r at threshold θ (99.5%, 99%, 99.9% variance)
4. Estimate per-neuron null space: dim(ker(H_i)) ≈ max(0, d_i - r)

This gives a coarse upper bound on null space, used across all neurons.
Loops over multiple noise conditions to reveal how noise affects identifiability.
"""

import os
import sys
import json
import numpy as np
import torch
import zarr
from tqdm import tqdm


def load_training_activity(data_dir, subsample_t=8):
    """Load voltage from x_list_train/voltage.zarr (full 64K training data, subsampled).

    Uses training data for robust rank estimation (64,000 frames is better than
    test's 8,528 frames for determining true population-level correlations).
    """
    from pathlib import Path
    data_path = Path(data_dir) / "x_list_train" / "voltage.zarr"
    print(f"  Loading training data from {data_path}...")

    voltage_zarr = zarr.open_array(str(data_path), mode='r')
    print(f"    Full shape: {voltage_zarr.shape}")

    # Subsample timesteps for tractable SVD (64000 / 8 = 8000)
    voltage = np.array(voltage_zarr[::subsample_t, :])
    print(f"    Subsampled (every {subsample_t}th frame): {voltage.shape}")

    return voltage


def compute_global_svd(h, edge_index, variance_thresholds=(0.995, 0.99, 0.999)):
    """Compute global SVD effective rank and estimate per-neuron null space."""
    T, N = h.shape
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    print(f"  Full activity matrix shape: {h.shape}")

    # Subsample: every 14th neuron, keep all timepoints
    neuron_indices = np.arange(0, N, 14)
    h_sub = h[:, neuron_indices]
    print(f"  Subsampled activity matrix: {h_sub.shape} (every 14th neuron)")

    # Compute full SVD
    print(f"  Computing SVD of {h_sub.shape[0]} × {h_sub.shape[1]} matrix...")
    U, sigma, Vt = np.linalg.svd(h_sub, full_matrices=False)

    # Total variance
    total_var = np.sum(sigma ** 2)
    print(f"  Total variance: {total_var:.2e}")
    print(f"  Singular values range: [{sigma.min():.4f}, {sigma.max():.4f}]")

    # Compute effective ranks at each threshold
    cumsum_var = np.cumsum(sigma ** 2) / total_var
    results = {
        "variance_thresholds": variance_thresholds,
        "effective_ranks": {},
        "singular_values": sigma,
    }

    for theta in variance_thresholds:
        r = np.searchsorted(cumsum_var, theta) + 1
        r = min(r, len(sigma))
        results["effective_ranks"][theta] = r
        print(f"    {theta:.1%} variance: rank = {r}")

    return results, sigma, cumsum_var


def estimate_null_space_from_global_rank(r, in_degrees, E):
    """Estimate per-neuron null space using global rank upper bound."""
    null_dims = np.maximum(0, in_degrees - r)
    return null_dims


def analyze_noise_condition(noise_label, ode_path, data_dir):
    """Analyze global SVD for a single noise condition.

    Uses training data (x_list_train/voltage.zarr) with 64,000 frames subsampled to 8,000,
    which is more robust than test data (8,528 frames only).
    """

    if not os.path.exists(ode_path):
        print(f"ERROR: ODE params not found at {ode_path}")
        return None

    print(f"\n{'='*80}")
    print(f"NOISE CONDITION: {noise_label}")
    print(f"{'='*80}")
    print(f"Loading ODE params from {ode_path}")

    state = torch.load(ode_path, map_location="cpu", weights_only=True)
    print(f"  W shape: {state['W'].shape}")
    print(f"  edge_index shape: {state['edge_index'].shape}")

    N = len(state['tau_i'])
    E = len(state['W'])

    # Load training voltage traces (64K frames subsampled to 8K)
    print(f"\nLoading voltage traces from x_list_train/voltage.zarr...")
    v = load_training_activity(data_dir, subsample_t=8)

    print(f"  ✓ Voltage matrix v shape: {v.shape}")
    print(f"    v: mean={v.mean():.4f}, std={v.std():.4f}, range=[{v.min():.4f}, {v.max():.4f}]")

    # Compute activity: h(t) = ReLU(v(t))
    print(f"\nComputing presynaptic activity h = ReLU(v)...")
    h = np.maximum(0, v)  # ReLU
    print(f"  ✓ Activity matrix h shape: {h.shape}")
    print(f"    h: mean={h.mean():.4f}, std={h.std():.4f}, sparsity={100*(h==0).mean():.1f}%")

    # Compute global SVD
    print(f"\nComputing global SVD...")
    results, sigma, cumsum_var = compute_global_svd(
        h, state['edge_index'], variance_thresholds=(0.90, 0.95, 0.99, 0.995, 0.999)
    )

    # Get in-degree distribution
    src, dst = state['edge_index'][0].numpy(), state['edge_index'][1].numpy()
    in_degree = np.zeros(N, dtype=int)
    for s, d in zip(src, dst):
        in_degree[int(d)] += 1

    # Filter neurons with in-degree > 0
    mask_postsynaptic = in_degree > 0
    in_degrees_post = in_degree[mask_postsynaptic]
    n_post = mask_postsynaptic.sum()

    print(f"  ✓ Postsynaptic neurons: {n_post:,d}")
    print(f"    In-degree: mean={in_degrees_post.mean():.1f}, median={np.median(in_degrees_post):.0f}, "
          f"min={in_degrees_post.min()}, max={in_degrees_post.max()}")

    return results, in_degrees_post, sigma, cumsum_var, E, N, n_post, state




def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define noise conditions to analyze
    noise_conditions = {
        "noise-free": {
            "ode_path": "../graphs_data/fly/flyvis_noise_free/ode_params.pt",
            "data_dir": "../graphs_data/fly/flyvis_noise_free"
        },
        "noise-0.05": {
            "ode_path": "../graphs_data/fly/flyvis_noise_005/ode_params.pt",
            "data_dir": "../graphs_data/fly/flyvis_noise_005"
        },
        "noise-0.5": {
            "ode_path": "../graphs_data/fly/flyvis_noise_05/ode_params.pt",
            "data_dir": "../graphs_data/fly/flyvis_noise_05"
        },
    }

    all_results = {}

    # Analyze each noise condition
    for noise_label, paths in noise_conditions.items():
        ode_path = os.path.join(script_dir, paths["ode_path"])
        data_dir = os.path.join(script_dir, paths["data_dir"])

        result = analyze_noise_condition(noise_label, ode_path, data_dir)
        if result is not None:
            all_results[noise_label] = result

    # Report results across all noise conditions
    if not all_results:
        print("ERROR: No results generated")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"CROSS-NOISE COMPARISON: GLOBAL SVD EFFECTIVE RANKS")
    print(f"{'='*80}")

    # Create comparison table
    print(f"\n{'Noise Condition':>16} {'Threshold':>10} {'Global rank':>12} {'Degree of':>15}")
    print(f"{'':>16} {'':>10} {'':>12} {'degeneracy':>15}")
    print(f"{'-'*80}")

    for noise_label in ["noise-free", "noise-0.05", "noise-0.5"]:
        if noise_label not in all_results:
            continue

        results, in_deg, sigma, cumsum_var, E, N, n_post, state = all_results[noise_label]

        for theta in [0.90, 0.95, 0.99, 0.995, 0.999]:
            if theta not in results["effective_ranks"]:
                continue

            r = results["effective_ranks"][theta]
            null_dims = estimate_null_space_from_global_rank(r, in_deg, E)
            total_null = int(null_dims.sum())
            degree_degen = 100 * total_null / E

            print(f"{noise_label:>16} {theta:10.1%} {r:12d} {degree_degen:14.1f}%")

    # Per-noise summary with all thresholds
    print(f"\n{'='*80}")
    print(f"DETAILED SUMMARY: EFFECTIVE RANKS AT ALL VARIANCE THRESHOLDS")
    print(f"{'='*80}")

    for noise_label in ["noise-free", "noise-0.05", "noise-0.5"]:
        if noise_label not in all_results:
            continue

        results, in_deg, sigma, cumsum_var, E, N, n_post, state = all_results[noise_label]

        print(f"\n{noise_label.upper()}")
        print(f"{'─'*60}")
        print(f"  Total edges: {E:,d}")
        print(f"  Postsynaptic neurons: {n_post:,d}")
        print(f"\n  Effective ranks across variance thresholds:")
        print(f"  {'Threshold':>12} {'Rank':>8} {'Null dims':>12} {'Degeneracy':>12}")
        print(f"  {'-'*50}")

        for theta in [0.90, 0.95, 0.99, 0.995, 0.999]:
            if theta in results["effective_ranks"]:
                r = results["effective_ranks"][theta]
                null_dims = estimate_null_space_from_global_rank(r, in_deg, E)
                total_null = int(null_dims.sum())
                degree_degen = 100 * total_null / E
                print(f"  {theta:11.1%} {r:8d} {total_null:12,d} {degree_degen:11.1f}%")

        # Use 99% for in-degree breakdown
        r99 = results["effective_ranks"][0.99]
        null99 = estimate_null_space_from_global_rank(r99, in_deg, E)

        n_fully_id = int((null99 == 0).sum())
        print(f"\n  Fully identifiable neurons (at 99%): {n_fully_id:,d} ({100*n_fully_id/n_post:.1f}%)")

        # In-degree breakdown
        print(f"\n  Null space by in-degree (at 99%, rank {r99}):")
        for lo, hi in [(1, 10), (11, 20), (21, 45), (46, 100), (101, 208)]:
            mask = (in_deg >= lo) & (in_deg <= hi)
            count = int(mask.sum())
            if count > 0:
                contribution = int(null99[mask].sum())
                print(f"    {lo:4d}--{hi:4d}: {count:6d} neurons → {contribution:8,d} null dims")

    print(f"\n{'='*80}\n")

    # Write results to JSON file
    results_file = os.path.join(script_dir, "results_global_svd.json")
    write_results_json(all_results, results_file)

    return all_results


def write_results_json(all_results, output_file):
    """Write results to JSON file for tex auto-update."""
    results_dict = {}

    for noise_label in ["noise-free", "noise-0.05", "noise-0.5"]:
        if noise_label not in all_results:
            continue

        results, in_deg, sigma, cumsum_var, E, N, n_post, state = all_results[noise_label]

        results_dict[noise_label] = {}
        for theta in [0.90, 0.95, 0.99, 0.995, 0.999]:
            if theta not in results["effective_ranks"]:
                continue

            r = results["effective_ranks"][theta]
            null_dims = estimate_null_space_from_global_rank(r, in_deg, E)
            total_null = int(null_dims.sum())

            results_dict[noise_label][f"{theta:.1%}"] = {
                "global_rank": int(r),
                "null_space_dim": total_null,
                "degree_of_degeneracy": f"{100*total_null/E:.1f}",
            }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✓ Results written to {output_file}")


if __name__ == "__main__":
    results = main()
