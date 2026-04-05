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
import numpy as np
import torch
import zarr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def load_saved_activity(test_zarr_path, n_frames=None):
    """Load voltage traces from saved test.zarr."""
    print(f"  Loading from {test_zarr_path}...")
    zarr_root = zarr.open(test_zarr_path, mode='r')

    if hasattr(zarr_root, 'shape'):
        v = zarr_root[:]
    else:
        chunks = []
        keys = sorted(zarr_root.keys(), key=lambda x: int(x.split('.')[0]))
        for key in tqdm(keys, desc="  Loading chunks", ncols=100, leave=False):
            chunk = zarr_root[key][:]
            chunks.append(chunk)
        v = np.concatenate(chunks, axis=0)

    if n_frames is not None:
        v = v[:n_frames]

    return v


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


def analyze_noise_condition(noise_label, ode_path, test_zarr_path):
    """Analyze global SVD for a single noise condition."""

    if not os.path.exists(ode_path):
        print(f"ERROR: ODE params not found at {ode_path}")
        return None

    if not os.path.exists(test_zarr_path):
        print(f"ERROR: Test data not found at {test_zarr_path}")
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

    # Load saved voltage traces (test set = 8K frames)
    print(f"\nLoading voltage traces from test.zarr (8,000 frames)...")
    v = load_saved_activity(test_zarr_path, n_frames=8000)

    if v.ndim > 2:
        v = v.squeeze()

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
        h, state['edge_index'], variance_thresholds=(0.995, 0.99, 0.999)
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


def visualize_global_svd(all_results, output_dir="./svg_global_svd_plots"):
    """Create visualizations of global SVD analysis."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating visualizations...")

    # Figure 1: Scree plot (singular values by threshold)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (noise_label, (results, in_deg, sigma, cumsum_var, E, N, n_post, state)) in enumerate(all_results.items()):
        ax = axes[idx]

        # Plot singular values
        ax.semilogy(range(len(sigma)), sigma, 'o-', markersize=4, linewidth=1.5, alpha=0.7)

        # Mark thresholds
        thresholds = [0.995, 0.99, 0.999]
        colors = ['red', 'blue', 'green']
        for theta, color in zip(thresholds, colors):
            r = results["effective_ranks"][theta]
            ax.axvline(r, color=color, linestyle='--', alpha=0.5, label=f'{theta:.1%}: r={r}')

        ax.set_xlabel('Singular value index', fontweight='bold')
        ax.set_ylabel('Singular value (log scale)', fontweight='bold')
        ax.set_title(f'{noise_label}\n(Subsampled: 982 neurons × {len(sigma)} singular values)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_scree_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Cumulative variance explained
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (noise_label, (results, in_deg, sigma, cumsum_var, E, N, n_post, state)) in enumerate(all_results.items()):
        ax = axes[idx]

        ax.plot(range(len(cumsum_var)), cumsum_var, 'b-', linewidth=2, label='Cumulative variance')
        ax.fill_between(range(len(cumsum_var)), 0, cumsum_var, alpha=0.3)

        # Mark thresholds
        thresholds = [0.995, 0.99, 0.999]
        colors = ['red', 'blue', 'green']
        for theta, color in zip(thresholds, colors):
            r = results["effective_ranks"][theta]
            ax.axhline(theta, color=color, linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(r, color=color, linestyle='--', alpha=0.5, linewidth=1.5, label=f'{theta:.1%}: r={r}')

        ax.set_xlabel('Singular value index', fontweight='bold')
        ax.set_ylabel('Cumulative variance fraction', fontweight='bold')
        ax.set_ylim([0.8, 1.0])
        ax.set_title(f'{noise_label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_cumulative_variance.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Null space estimate vs in-degree
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (noise_label, (results, in_deg, sigma, cumsum_var, E, N, n_post, state)) in enumerate(all_results.items()):
        ax = axes[idx]

        r_99 = results["effective_ranks"][0.99]
        null_dims = estimate_null_space_from_global_rank(r_99, in_deg, E)

        ax.scatter(in_deg, null_dims, alpha=0.4, s=20)
        ax.plot([0, in_deg.max()], [0, in_deg.max() - r_99], 'r--', linewidth=2,
                label=f'null = d_i - {r_99} (at 99%)')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('In-degree ($d_i$)', fontweight='bold')
        ax.set_ylabel('Estimated null space dim', fontweight='bold')
        ax.set_title(f'{noise_label}\nGlobal rank: {r_99}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_null_vs_degree.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved 3 visualization PNG files to {output_dir}/")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define noise conditions to analyze
    noise_conditions = {
        "noise-free": {
            "ode_path": "../graphs_data/fly/flyvis_noise_free/ode_params.pt",
            "test_zarr_path": "../graphs_data/fly/flyvis_noise_free/y_list_test.zarr"
        },
        "noise-0.05": {
            "ode_path": "../graphs_data/fly/flyvis_noise_005/ode_params.pt",
            "test_zarr_path": "../graphs_data/fly/flyvis_noise_005/y_list_test.zarr"
        },
        "noise-0.5": {
            "ode_path": "../graphs_data/fly/flyvis_noise_05/ode_params.pt",
            "test_zarr_path": "../graphs_data/fly/flyvis_noise_05/y_list_test.zarr"
        },
    }

    all_results = {}

    # Analyze each noise condition
    for noise_label, paths in noise_conditions.items():
        ode_path = os.path.join(script_dir, paths["ode_path"])
        test_zarr_path = os.path.join(script_dir, paths["test_zarr_path"])

        result = analyze_noise_condition(noise_label, ode_path, test_zarr_path)
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

        for theta in [0.995, 0.99, 0.999]:
            if theta not in results["effective_ranks"]:
                continue

            r = results["effective_ranks"][theta]
            null_dims = estimate_null_space_from_global_rank(r, in_deg, E)
            total_null = int(null_dims.sum())
            degree_degen = 100 * total_null / E

            print(f"{noise_label:>16} {theta:10.1%} {r:12d} {degree_degen:14.1f}%")

    # Per-noise summary at 99%
    print(f"\n{'='*80}")
    print(f"DETAILED SUMMARY AT 99% VARIANCE THRESHOLD")
    print(f"{'='*80}")

    for noise_label in ["noise-free", "noise-0.05", "noise-0.5"]:
        if noise_label not in all_results:
            continue

        results, in_deg, sigma, cumsum_var, E, N, n_post, state = all_results[noise_label]

        print(f"\n{noise_label.upper()}")
        print(f"{'─'*60}")

        r99 = results["effective_ranks"][0.99]
        null99 = estimate_null_space_from_global_rank(r99, in_deg, E)

        print(f"  Global effective rank (99%): {r99}")
        print(f"  Total edges: {E:,d}")
        print(f"  Postsynaptic neurons: {n_post:,d}")
        print(f"  Null space dimension: {null99.sum():,.0f}")
        print(f"  Degree of degeneracy: {100*null99.sum()/E:.1f}%")

        n_fully_id = int((null99 == 0).sum())
        print(f"  Fully identifiable neurons: {n_fully_id:,d} ({100*n_fully_id/n_post:.1f}%)")

        # In-degree breakdown
        print(f"\n  Null space by in-degree (using global rank {r99}):")
        for lo, hi in [(1, 10), (11, 20), (21, 45), (46, 100), (101, 208)]:
            mask = (in_deg >= lo) & (in_deg <= hi)
            count = int(mask.sum())
            if count > 0:
                contribution = int(null99[mask].sum())
                print(f"    {lo:4d}--{hi:4d}: {count:6d} neurons → {contribution:8,d} null dims")

    print(f"\n{'='*80}\n")

    # Generate visualizations
    visualize_global_svd(all_results)

    return all_results


if __name__ == "__main__":
    results = main()
