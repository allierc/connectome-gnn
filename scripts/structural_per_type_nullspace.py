#!/usr/bin/env python
"""Structural per-type null space analysis (Step 3).

Counts the null space dimension by analyzing the columnar connectivity structure.

For each postsynaptic neuron i and each presynaptic cell type α with k_iα > 1
incoming edges, the null space gains k_iα - 1 free dimensions (sum constraint).

Summing over all neurons and types:
    dim(ker(H)) = Σ_i Σ_α:(k_iα > 1) (k_iα - 1)

This mechanistic approach precisely measures within-type redundancy.
"""

import os
import sys
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


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


def compute_structural_null_space(edge_index, neuron_type_labels, cell_types):
    """Compute null space dimension by per-type counting."""
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    N = len(neuron_type_labels)

    # For each (postsynaptic neuron, presynaptic type) pair, count edges
    type_edges = defaultdict(lambda: defaultdict(list))
    for s, d in zip(src, dst):
        src_type = int(neuron_type_labels[int(s)])
        dst_neuron = int(d)
        type_edges[dst_neuron][src_type].append(int(s))

    # Count null space dimensions
    null_dim_per_neuron = {}
    null_dim_by_type = defaultdict(int)
    degenerate_groups = 0

    for dst_neuron, type_dict in type_edges.items():
        null_dim_neuron = 0
        for src_type, presynaptic_neurons in type_dict.items():
            k = len(presynaptic_neurons)
            if k > 1:
                # Each group of k same-type edges contributes k-1 null dimensions
                null_dim_neuron += (k - 1)
                null_dim_by_type[src_type] += (k - 1)
                degenerate_groups += 1

        if null_dim_neuron > 0:
            null_dim_per_neuron[dst_neuron] = null_dim_neuron

    total_null = sum(null_dim_per_neuron.values())

    return {
        "null_dim_per_neuron": null_dim_per_neuron,
        "null_dim_by_type": null_dim_by_type,
        "total_null": total_null,
        "degenerate_groups": degenerate_groups,
    }


def analyze_null_space(ode_path, cell_types, neuron_type_labels, noise_label):
    """Analyze structural null space for one noise condition."""

    if not os.path.exists(ode_path):
        print(f"ERROR: ODE params not found at {ode_path}")
        return None

    print(f"\n{'='*80}")
    print(f"NOISE CONDITION: {noise_label}")
    print(f"{'='*80}")
    print(f"Loading ODE params from {ode_path}")

    state = torch.load(ode_path, map_location="cpu", weights_only=True)
    edge_index = state["edge_index"]

    print(f"  edge_index shape: {edge_index.shape}")
    E = edge_index.shape[1]
    N = len(neuron_type_labels)
    print(f"  Total neurons: {N}")
    print(f"  Total edges: {E}")

    # Compute structural null space
    print(f"\nComputing structural per-type null space...")
    results = compute_structural_null_space(edge_index, neuron_type_labels, cell_types)

    # Get in-degree distribution
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    in_degree = np.zeros(N, dtype=int)
    for s, d in zip(src, dst):
        in_degree[int(d)] += 1

    mask_postsynaptic = in_degree > 0
    in_degrees_post = in_degree[mask_postsynaptic]
    n_post = mask_postsynaptic.sum()

    print(f"  ✓ Postsynaptic neurons: {n_post:,d}")
    print(f"    In-degree: mean={in_degrees_post.mean():.1f}, median={np.median(in_degrees_post):.0f}, "
          f"min={in_degrees_post.min()}, max={in_degrees_post.max()}")

    return results, in_degrees_post, E, N, n_post, edge_index


def visualize_structural_nullspace(all_results, cell_types, output_dir="./svg_structural_plots"):
    """Create visualizations of structural null space analysis."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating visualizations...")

    # Figure 1: Per-type null space contribution (bar chart)
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for idx, (noise_label, (results, in_deg, E, N, n_post, edge_index)) in enumerate(all_results.items()):
        ax = axes[idx]

        null_by_type = results["null_dim_by_type"]

        if not null_by_type:
            ax.text(0.5, 0.5, 'No degenerate types found', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{noise_label}')
            continue

        # Sort by contribution
        sorted_types = sorted(null_by_type.items(), key=lambda x: x[1], reverse=True)[:20]
        type_names = [cell_types[t[0]][:15] for t in sorted_types]
        type_contribs = [t[1] for t in sorted_types]

        ax.barh(range(len(type_names)), type_contribs, color=plt.cm.viridis(
            np.linspace(0, 1, len(type_names))))
        ax.set_yticks(range(len(type_names)))
        ax.set_yticklabels(type_names, fontsize=9)
        ax.set_xlabel('Null space contribution (# free weights)', fontweight='bold')
        ax.set_title(f'{noise_label}\n(Top 20 types)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_per_type_contribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2: Null space per neuron vs in-degree
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (noise_label, (results, in_deg, E, N, n_post, edge_index)) in enumerate(all_results.items()):
        ax = axes[idx]

        # Reconstruct null_dim array
        null_dim_per_neuron = results["null_dim_per_neuron"]
        neuron_ids = sorted(null_dim_per_neuron.keys())
        null_dims = np.array([null_dim_per_neuron[nid] for nid in neuron_ids])

        # Get corresponding in-degrees
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        in_deg_neurons = np.zeros(N, dtype=int)
        for s, d in zip(src, dst):
            in_deg_neurons[int(d)] += 1
        in_deg_degenerate = in_deg_neurons[neuron_ids]

        ax.scatter(in_deg_degenerate, null_dims, alpha=0.5, s=30)
        ax.set_xlabel('In-degree ($d_i$)', fontweight='bold')
        ax.set_ylabel('Null space dimension (from same-type constraint)', fontweight='bold')
        ax.set_title(f'{noise_label}\n({len(neuron_ids)} degenerate neurons)',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_nullspace_vs_degree.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3: Distribution of null space dimensions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (noise_label, (results, in_deg, E, N, n_post, edge_index)) in enumerate(all_results.items()):
        ax = axes[idx]

        null_dim_per_neuron = results["null_dim_per_neuron"]
        null_dims_all = np.array(list(null_dim_per_neuron.values()))

        if len(null_dims_all) > 0:
            ax.hist(null_dims_all, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Null space dimension per neuron', fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')
            ax.set_title(f'{noise_label}\n(n={len(null_dims_all)} degenerate neurons)',
                        fontsize=11, fontweight='bold')
            ax.axvline(null_dims_all.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'mean={null_dims_all.mean():.1f}')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_null_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved 3 visualization PNG files to {output_dir}/")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load cell type information first
    print("Loading cell type information...")
    cell_types, neuron_type_labels = load_connectome_types(script_dir)
    if cell_types is None or neuron_type_labels is None:
        print("ERROR: Could not load cell type data")
        sys.exit(1)
    print(f"  ✓ Loaded {len(cell_types)} cell types, {len(neuron_type_labels)} neurons")

    # Define noise conditions to analyze
    noise_conditions = {
        "noise-free": {
            "ode_path": "../graphs_data/fly/flyvis_noise_free/ode_params.pt",
        },
        "noise-0.05": {
            "ode_path": "../graphs_data/fly/flyvis_noise_005/ode_params.pt",
        },
        "noise-0.5": {
            "ode_path": "../graphs_data/fly/flyvis_noise_05/ode_params.pt",
        },
    }

    all_results = {}

    # Analyze each noise condition
    for noise_label, paths in noise_conditions.items():
        ode_path = os.path.join(script_dir, paths["ode_path"])

        result = analyze_null_space(ode_path, cell_types, neuron_type_labels, noise_label)
        if result is not None:
            all_results[noise_label] = result

    # Report results across all noise conditions
    if not all_results:
        print("ERROR: No results generated")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"CROSS-NOISE COMPARISON: STRUCTURAL PER-TYPE NULL SPACE")
    print(f"{'='*80}\n")

    # Create comparison table
    print(f"{'Noise Condition':>16} {'Degenerate':>15} {'Null space':>15} {'Degree of':>15}")
    print(f"{'':>16} {'groups':>15} {'dimension':>15} {'degeneracy':>15}")
    print(f"{'-'*80}")

    for noise_label in ["noise-free", "noise-0.05", "noise-0.5"]:
        if noise_label not in all_results:
            continue

        results, in_deg, E, N, n_post, edge_index = all_results[noise_label]

        total_null = results["total_null"]
        degenerate_groups = results["degenerate_groups"]
        degree_degen = 100 * total_null / E

        print(f"{noise_label:>16} {degenerate_groups:15,d} {total_null:15,d} {degree_degen:14.1f}%")

    # Per-noise summary
    print(f"\n{'='*80}")
    print(f"DETAILED STRUCTURAL ANALYSIS")
    print(f"{'='*80}")

    for noise_label in ["noise-free", "noise-0.05", "noise-0.5"]:
        if noise_label not in all_results:
            continue

        results, in_deg, E, N, n_post, edge_index = all_results[noise_label]

        print(f"\n{noise_label.upper()}")
        print(f"{'─'*60}")

        total_null = results["total_null"]
        degenerate_groups = results["degenerate_groups"]
        null_by_type = results["null_dim_by_type"]

        print(f"  Total edges: {E:,d}")
        print(f"  Postsynaptic neurons: {n_post:,d}")
        print(f"  Degenerate groups (same-type with k>1): {degenerate_groups:,d}")
        print(f"  Total null space dimension: {total_null:,d}")
        print(f"  Degree of degeneracy: {100*total_null/E:.1f}%")

        # Top contributing types
        print(f"\n  Top 10 cell types by null space contribution:")
        sorted_types = sorted(null_by_type.items(), key=lambda x: x[1], reverse=True)[:10]
        for type_id, null_contrib in sorted_types:
            type_name = cell_types[type_id]
            print(f"    {type_name:>12s}: {null_contrib:8,d} null dimensions")

    print(f"\n{'='*80}\n")

    # Generate visualizations
    visualize_structural_nullspace(all_results, cell_types)

    return all_results


if __name__ == "__main__":
    results = main()
