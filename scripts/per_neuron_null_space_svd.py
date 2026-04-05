#!/usr/bin/env python
"""Per-neuron SVD-based null space analysis (Step 2).

Computes the effective rank and null space dimension for each neuron's
incoming activity matrix using SVD with a variable variance threshold.

For each postsynaptic neuron i, we:
1. Extract H_i = activity matrix of its d_i presynaptic partners
2. Compute SVD: H_i = U Sigma V^T
3. Find effective rank r_i at threshold θ (99.5%, 99%, 99.9% variance)
4. Deduce null space: dim(ker(H_i)) ≈ d_i - r_i

Loops over multiple noise conditions to reveal how noise affects identifiability.
"""

import os
import sys
import numpy as np
from collections import defaultdict
import torch
import zarr
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns


def load_saved_activity(test_zarr_path, n_frames=None):
    """Load voltage traces from saved test.zarr."""
    print(f"  Loading from {test_zarr_path}...")
    zarr_root = zarr.open(test_zarr_path, mode='r')

    # zarr_root is an Array, load directly
    if hasattr(zarr_root, 'shape'):
        # It's an array
        v = zarr_root[:]
    else:
        # It's a group, load chunks
        chunks = []
        keys = sorted(zarr_root.keys(), key=lambda x: int(x.split('.')[0]))
        for key in tqdm(keys, desc="  Loading chunks", ncols=100, leave=False):
            chunk = zarr_root[key][:]
            chunks.append(chunk)
        v = np.concatenate(chunks, axis=0)

    if n_frames is not None:
        v = v[:n_frames]

    return v


def compute_per_neuron_svd(h, edge_index, tau_i, v_rest, variance_thresholds=(0.995, 0.99, 0.999)):
    """Compute per-neuron effective rank and null space dimension."""
    T, N = h.shape
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    # Build incoming edge lists per neuron
    in_edges = defaultdict(list)
    for e_idx in range(len(src)):
        in_edges[int(dst[e_idx])].append(int(src[e_idx]))

    # Store per-neuron results
    results = {
        "neuron_id": [],
        "in_degree": [],
        "effective_rank": {θ: [] for θ in variance_thresholds},
        "null_dim": {θ: [] for θ in variance_thresholds},
    }

    # For each postsynaptic neuron
    for i in tqdm(range(N), desc="Computing per-neuron SVD", ncols=100):
        if i not in in_edges:
            continue

        # Extract presynaptic partners
        presynaptic = np.array(in_edges[i])
        d_i = len(presynaptic)

        # Extract activity matrix H_i (T x d_i)
        H_i = h[:, presynaptic]

        # Compute SVD
        U, sigma, Vt = np.linalg.svd(H_i, full_matrices=False)

        # Compute explained variance ratio
        total_var = np.sum(sigma ** 2)

        results["neuron_id"].append(i)
        results["in_degree"].append(d_i)

        # For each variance threshold
        for theta in variance_thresholds:
            # Find effective rank
            cumsum_var = np.cumsum(sigma ** 2) / total_var
            r_i = np.searchsorted(cumsum_var, theta) + 1
            r_i = min(r_i, d_i)

            # Null space dimension
            null_dim_i = max(0, d_i - r_i)

            results["effective_rank"][theta].append(r_i)
            results["null_dim"][theta].append(null_dim_i)

    return results


def analyze_noise_condition(noise_label, ode_path, test_zarr_path):
    """Analyze per-neuron SVD for a single noise condition."""

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

    # Compute per-neuron SVD
    print(f"\nComputing per-neuron SVD...")
    results = compute_per_neuron_svd(
        h, state['edge_index'], state['tau_i'], state['V_i_rest'],
        variance_thresholds=(0.995, 0.99, 0.999)
    )

    n_post = len(results["neuron_id"])
    print(f"  ✓ Processed {n_post:,d} postsynaptic neurons")

    degrees = np.array(results["in_degree"])
    print(f"    In-degree: mean={degrees.mean():.1f}, median={np.median(degrees):.0f}, "
          f"min={degrees.min()}, max={degrees.max()}")

    return results, degrees, E, N, n_post, state


def load_connectome_types(script_dir):
    """Load cell type names and mapping from bio connectome."""
    sys.path.insert(0, os.path.join(script_dir, "../src"))
    try:
        from connectome_gnn.generators.connconstr_data import load_connectome_bio
        bio_data = load_connectome_bio("flyvis")
        cell_types = bio_data["cell_type_names"]
        neuron_type_labels = bio_data["neuron_type_labels"]
        return cell_types, neuron_type_labels
    except Exception as e:
        print(f"  Warning: Could not load cell types: {e}")
        return None, None


def visualize_per_neuron_svd(all_results, script_dir, output_dir="./svg_per_neuron_plots"):
    """Create novel visualizations of per-neuron rank-nullity analysis for all noise levels."""
    os.makedirs(output_dir, exist_ok=True)

    # Load cell type information
    cell_types, neuron_type_labels = load_connectome_types(script_dir)

    if cell_types is None:
        print("  Skipping visualizations: cell type data unavailable")
        return

    print(f"\nGenerating visualizations for all noise conditions...")

    viz_count = 0

    # Process each noise condition
    for noise_label in ["noise-free", "noise-0.05", "noise-0.5"]:
        if noise_label not in all_results:
            print(f"  Warning: {noise_label} results not available, skipping")
            continue

        results, degrees, E, N, n_post, state = all_results[noise_label]

        # Extract per-neuron data
        neuron_ids = np.array(results["neuron_id"])
        in_degrees = np.array(results["in_degree"])
        r99 = np.array(results["effective_rank"][0.99])
        null99 = np.array(results["null_dim"][0.99])

        # Map neurons to cell types
        neuron_types = neuron_type_labels[neuron_ids]

        # Per-type aggregation
        type_stats = {}
        for type_id, type_name in enumerate(cell_types):
            mask = neuron_types == type_id
            if mask.sum() == 0:
                continue

            type_stats[type_name] = {
                "n_neurons": int(mask.sum()),
                "mean_rank": r99[mask].mean(),
                "mean_nullity": null99[mask].mean(),
                "mean_in_degree": in_degrees[mask].mean(),
                "n_fully_id": int((null99[mask] == 0).sum()),
                "pct_fully_id": 100 * (null99[mask] == 0).sum() / mask.sum(),
                "total_null": int(null99[mask].sum()),
            }

        # Sort by total null space contribution
        sorted_types = sorted(type_stats.items(), key=lambda x: x[1]["total_null"], reverse=True)

        # Create visualizations for this noise condition
        _create_noise_condition_plots(sorted_types, noise_label, output_dir, E)
        viz_count += 1

    if viz_count > 0:
        print(f"  ✓ Generated {viz_count * 4} visualization PNG files in {output_dir}/")


def _create_noise_condition_plots(sorted_types, noise_label, output_dir, E):

    # Noise label for filename
    noise_key = noise_label.replace("-", "_").replace(".", "_")

    # =========================================================================
    # Figure 1: Per-type identifiability landscape (heatmap)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 10))

    type_names = [t[0] for t in sorted_types]
    metrics = ["mean_in_degree", "mean_rank", "mean_nullity", "pct_fully_id"]
    metric_labels = ["Mean In-Degree", "Mean Rank", "Mean Nullity", "% Fully ID"]

    data_matrix = np.zeros((len(type_names), len(metrics)))
    for i, (tname, tdata) in enumerate(sorted_types):
        data_matrix[i, 0] = tdata["mean_in_degree"]
        data_matrix[i, 1] = tdata["mean_rank"]
        data_matrix[i, 2] = tdata["mean_nullity"]
        data_matrix[i, 3] = tdata["pct_fully_id"]

    # Normalize columns for visualization
    data_norm = data_matrix.copy()
    for col in range(data_matrix.shape[1]):
        vmax = data_matrix[:, col].max()
        if vmax > 0:
            data_norm[:, col] = data_matrix[:, col] / vmax

    sns.heatmap(data_norm, annot=data_matrix.astype(int), fmt='d',
                xticklabels=metric_labels, yticklabels=type_names,
                cmap='YlOrRd', cbar_kws={'label': 'Normalized value'}, ax=ax)
    ax.set_title(f"Per-Type Rank-Nullity Landscape (99% threshold, {noise_label})",
                 fontsize=14, fontweight='bold')
    ax.set_ylabel("Cell Type (sorted by null space contribution)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"01_heatmap_{noise_key}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 2: Diverging bar chart - Identifiable vs Degenerate per type
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 10))

    type_names_short = [t[0][:15] for t in sorted_types[:30]]  # Top 30 types
    identified = np.array([sorted_types[i][1]["total_null"] for i in range(min(30, len(sorted_types)))])
    unidentified = np.array([E - identified[i] for i in range(len(identified))])

    y_pos = np.arange(len(type_names_short))
    ax.barh(y_pos, -identified, label='Null dims (degenerate)', color='#d62728', alpha=0.8)
    ax.barh(y_pos, unidentified, label='Identifiable dims', color='#2ca02c', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(type_names_short, fontsize=9)
    ax.set_xlabel('Cumulative edge dimensions', fontweight='bold')
    ax.set_title(f'Per-Type Identifiability: {noise_label} (Top 30 types)',
                 fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"02_diverging_{noise_key}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 3: Rank vs In-Degree scatter colored by % fully identifiable
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))

    type_names_list = [t[0] for t in sorted_types]
    mean_ranks = [t[1]["mean_rank"] for t in sorted_types]
    mean_degrees = [t[1]["mean_in_degree"] for t in sorted_types]
    pct_fully_id = [t[1]["pct_fully_id"] for t in sorted_types]
    n_neurons = [t[1]["n_neurons"] for t in sorted_types]

    scatter = ax.scatter(mean_degrees, mean_ranks, c=pct_fully_id, s=np.array(n_neurons)*3,
                        cmap='RdYlGn', alpha=0.6, edgecolors='black', linewidth=0.5)

    # Diagonal line: rank = in-degree (fully identifiable limit)
    max_deg = max(mean_degrees) if mean_degrees else 1
    ax.plot([0, max_deg], [0, max_deg], 'k--', alpha=0.3, label='rank = in-degree (full ID)')

    for i, (name, rank, deg) in enumerate(zip(type_names_list[:15], mean_ranks[:15], mean_degrees[:15])):
        ax.annotate(name[:10], (deg, rank), fontsize=7, alpha=0.7)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('% Fully Identifiable', fontweight='bold')
    ax.set_xlabel('Mean In-Degree', fontweight='bold', fontsize=11)
    ax.set_ylabel('Mean Effective Rank (99%)', fontweight='bold', fontsize=11)
    ax.set_title(f'Rank-Nullity Phase Space: {noise_label}',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"03_scatter_{noise_key}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Figure 4: Waterfall - Cumulative null space by cell type
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))

    type_names_short = [t[0][:12] for t in sorted_types[:25]]
    null_contribs = [sorted_types[i][1]["total_null"] for i in range(min(25, len(sorted_types)))]
    cumsum = np.cumsum(null_contribs)

    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(type_names_short)))

    for i, (name, val, cs) in enumerate(zip(type_names_short, null_contribs, cumsum)):
        ax.bar(i, val, bottom=cs-val, color=colors[i], edgecolor='black', linewidth=0.5, label=name if i < 5 else '')
        if val > 5000:  # Label significant contributors
            ax.text(i, cs - val/2, f'{int(val/1000)}k', ha='center', va='center', fontsize=8, fontweight='bold')

    ax.axhline(E/2, color='red', linestyle='--', alpha=0.5, label='50% of edges')
    ax.set_ylabel('Cumulative null space dimension', fontweight='bold', fontsize=11)
    ax.set_xlabel('Cell Type (sorted by contribution)', fontweight='bold', fontsize=11)
    ax.set_xticks(range(len(type_names_short)))
    ax.set_xticklabels(type_names_short, rotation=45, ha='right', fontsize=9)
    ax.set_title(f'Cumulative Null Space by Type: {noise_label}',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0, E * 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"04_waterfall_{noise_key}.png"), dpi=150, bbox_inches='tight')
    plt.close()



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
    print(f"CROSS-NOISE COMPARISON: PER-NEURON SVD AT EACH VARIANCE THRESHOLD")
    print(f"{'='*80}")

    # Create comparison table
    print(f"\n{'Noise Condition':>16} {'Threshold':>10} {'Eff.Rank':>10} {'Null dim':>12} {'% edges':>10} {'Fully ID':>10}")
    print(f"{'-'*80}")

    for noise_label in ["noise-free", "noise-0.05", "noise-0.5"]:
        if noise_label not in all_results:
            continue

        results, degrees, E, N, n_post, state = all_results[noise_label]

        for theta in [0.995, 0.99, 0.999]:
            if theta not in results["effective_rank"]:
                continue

            r = np.array(results["effective_rank"][theta])
            null_dims = np.array(results["null_dim"][theta])
            total_null = int(null_dims.sum())
            n_fully_id = int((null_dims == 0).sum())

            print(f"{noise_label:>16} {theta:10.1%} {r.mean():10.1f} {total_null:12,d} "
                  f"{100*total_null/E:9.1f}% {n_fully_id:10d}")

    # Per-noise summary at 99%
    print(f"\n{'='*80}")
    print(f"DETAILED SUMMARY AT 99% VARIANCE THRESHOLD")
    print(f"{'='*80}")

    for noise_label in ["noise-free", "noise-0.05", "noise-0.5"]:
        if noise_label not in all_results:
            continue

        results, degrees, E, N, n_post, state = all_results[noise_label]

        print(f"\n{noise_label.upper()}")
        print(f"{'─'*60}")

        r99 = np.array(results['effective_rank'][0.99])
        null99 = np.array(results['null_dim'][0.99])
        fully_id_99 = int((null99 == 0).sum())
        partial_degen = int(((null99 > 0) & (null99 < 20)).sum())
        heavy_degen = int((null99 >= 20).sum())

        print(f"  Total edges: {E:,d}")
        print(f"  Postsynaptic neurons: {n_post:,d}")
        print(f"  Effective rank (mean ± std): {r99.mean():.1f} ± {r99.std():.1f}")
        print(f"  Null space dimension: {null99.sum():,.0f} ({100*null99.sum()/E:.1f}%)")
        print(f"  Fully identifiable neurons: {fully_id_99:,d} ({100*fully_id_99/n_post:.1f}%)")
        print(f"  Partially degenerate (1-19): {partial_degen:,d} ({100*partial_degen/n_post:.1f}%)")
        print(f"  Heavily degenerate (≥20): {heavy_degen:,d} ({100*heavy_degen/n_post:.1f}%)")

        # In-degree breakdown
        print(f"\n  Null space by in-degree:")
        for lo, hi in [(1, 10), (11, 20), (21, 45), (46, 100), (101, 208)]:
            mask = (degrees >= lo) & (degrees <= hi)
            count = int(mask.sum())
            if count > 0:
                contribution = int(null99[mask].sum())
                print(f"    {lo:4d}--{hi:4d}: {count:6d} neurons → {contribution:8,d} null dims")

    print(f"\n{'='*80}\n")

    # Generate visualizations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    visualize_per_neuron_svd(all_results, script_dir)

    return all_results


if __name__ == "__main__":
    results = main()
