#!/usr/bin/env python3
"""Analyze within-type activity correlations across noise levels (flyvis).

For each noise condition (noise-free, σ=0.05, σ=0.5), compute:
1. Mean pairwise Pearson correlation among same-type neurons
2. Effective rank of same-type activity submatrix (at 99% variance)
3. Estimated null space dimension per type: max(0, k - eff_rank)

This quantifies how noise breaks within-type degeneracy and explains
why σ=0.5 yields R²_W=0.997 vs 0.959 noise-free.

Usage:
    python analyze_noise_correlation.py
"""

import sys
from pathlib import Path

import numpy as np
import zarr

INDEX_TO_NAME = {
    0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)',
    5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
    10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10', 14: 'Mi11',
    15: 'Mi12', 16: 'Mi13', 17: 'Mi14', 18: 'Mi15', 19: 'Mi2',
    20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2',
    25: 'R3', 26: 'R4', 27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8',
    31: 'T1', 32: 'T2', 33: 'T2a', 34: 'T3', 35: 'T4a',
    36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b',
    41: 'T5c', 42: 'T5d', 43: 'Tm1', 44: 'Tm16', 45: 'Tm2',
    46: 'Tm20', 47: 'Tm28', 48: 'Tm3', 49: 'Tm30', 50: 'Tm4',
    51: 'Tm5Y', 52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9',
    56: 'TmY10', 57: 'TmY13', 58: 'TmY14', 59: 'TmY15',
    60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9',
}

DATA_ROOT = Path(__file__).parent.parent / "graphs_data" / "fly"
DATASETS = {
    'σ=0 (noise-free)': DATA_ROOT / 'flyvis_noise_free',
    'σ=0.05': DATA_ROOT / 'flyvis_noise_005',
    'σ=0.5': DATA_ROOT / 'flyvis_noise_05',
}


def load_activity(data_dir, subsample_t=8):
    """Load voltage (T, N) and neuron_type (N,) from zarr.

    Subsamples timesteps by factor `subsample_t` to keep memory manageable.
    64000 / 8 = 8000 frames — enough for correlation and rank estimates.
    """
    voltage_zarr = zarr.open_array(
        str(data_dir / "x_list_train" / "voltage.zarr"), mode='r')
    voltage = np.array(voltage_zarr[::subsample_t, :])  # (T/sub, N)
    neuron_type = np.array(zarr.open_array(
        str(data_dir / "x_list_train" / "neuron_type.zarr"), mode='r'))
    return voltage, neuron_type


def relu(x):
    return np.maximum(x, 0.0)


def analyze_type(h_type):
    """Analyze a (T, k) activity matrix for one cell type.

    Returns: mean_corr, min_corr, eff_rank_99, null_dim
    """
    T, k = h_type.shape

    # --- Pairwise correlation ---
    h_centered = h_type - h_type.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(h_centered, axis=0, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    h_normed = h_centered / norms
    corr_matrix = (h_normed.T @ h_normed) / T  # (k, k)
    triu_idx = np.triu_indices(k, k=1)
    pairwise = corr_matrix[triu_idx]
    mean_corr = float(np.mean(pairwise))
    min_corr = float(np.min(pairwise))

    # --- Effective rank via SVD ---
    s = np.linalg.svd(h_type, compute_uv=False)
    cumvar = np.cumsum(s ** 2)
    total_var = cumvar[-1]
    if total_var == 0:
        eff_rank = 0
    else:
        eff_rank = int(np.searchsorted(cumvar / total_var, 0.99) + 1)

    null_dim = max(0, k - eff_rank)

    return mean_corr, min_corr, eff_rank, null_dim


def global_effective_rank(voltage, max_components=100, subsample_n=14):
    """Compute effective rank of the full population activity.

    Subsamples neurons by factor `subsample_n` so that full SVD is feasible
    and the total variance is exact (not truncated).
    """
    h = relu(voltage)
    T, N = h.shape

    # Subsample neurons for tractable full SVD
    h_sub = h[:, ::subsample_n]
    T_sub, N_sub = h_sub.shape
    print(f"    Global SVD: {T_sub} × {N_sub} (subsampled {subsample_n}× from {N} neurons)")

    # Full SVD on subsampled matrix — total variance is exact
    s = np.linalg.svd(h_sub.astype(np.float64), compute_uv=False)
    cumvar = np.cumsum(s ** 2)
    total = cumvar[-1]  # exact total variance (no truncation)
    rank_90 = int(np.searchsorted(cumvar / total, 0.90) + 1)
    rank_99 = int(np.searchsorted(cumvar / total, 0.99) + 1)
    return rank_90, rank_99


def load_in_degrees(data_dir):
    """Load in-degree per neuron from edge_index.pt."""
    import torch
    edge_index = torch.load(
        str(data_dir / "edge_index.pt"), weights_only=False, map_location='cpu').numpy()
    dst = edge_index[1]
    N = 13741
    return np.bincount(dst, minlength=N)


def per_neuron_null(in_degrees, rank):
    """Compute Σ max(0, d_i - rank) over all neurons."""
    null_dims = np.maximum(0, in_degrees - rank)
    return int(null_dims.sum())


def main():
    print("=" * 90)
    print("  Within-type correlation & rank analysis: flyvis (13,741 neurons, 65 types)")
    print("=" * 90)

    all_results = {}

    # Load in-degrees once (same graph for all noise levels)
    first_dir = next(d for d in DATASETS.values() if d.exists())
    in_degrees = load_in_degrees(first_dir)
    E = int(in_degrees.sum())
    print(f"\n  Edge count: {E}, max in-degree: {in_degrees.max()}, "
          f"mean: {in_degrees.mean():.1f}")

    for label, data_dir in DATASETS.items():
        if not data_dir.exists():
            print(f"\n  SKIP {label}: {data_dir} not found")
            continue

        print(f"\n{'─' * 90}")
        print(f"  {label}  ({data_dir.name})")
        print(f"{'─' * 90}")

        voltage, neuron_type = load_activity(data_dir)
        h = relu(voltage)
        T, N = h.shape
        n_types = int(neuron_type.max()) + 1
        print(f"  Shape: {T} timesteps × {N} neurons (subsampled from 64,000), {n_types} types\n")

        # Global effective rank
        print(f"  Computing global SVD (may take a minute)...")
        rank_90, rank_99 = global_effective_rank(voltage)
        null_global = per_neuron_null(in_degrees, rank_99)
        pct_ident = 100 * (E - null_global) / E
        print(f"  Global effective rank: {rank_90} (90% var), {rank_99} (99% var)")
        print(f"  Per-neuron null space (Σ max(0, d_i - {rank_99})): {null_global}")
        print(f"  Identifiable: {E - null_global} / {E} ({pct_ident:.1f}%)\n")

        # Per-type analysis
        type_results = {}
        print(f"  {'Type':<12s} {'k':>4s} {'mean_corr':>10s} {'min_corr':>10s}"
              f" {'rank_99':>8s} {'null_dim':>9s}")
        print(f"  {'─' * 12} {'─' * 4} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 9}")

        total_null = 0
        for tid in range(n_types):
            mask = neuron_type == tid
            k = int(mask.sum())
            if k < 2:
                continue

            h_type = h[:, mask]  # (T, k)
            mean_corr, min_corr, eff_rank, null_dim = analyze_type(h_type)
            total_null += null_dim

            name = INDEX_TO_NAME.get(tid, f'type_{tid}')
            type_results[tid] = {
                'k': k, 'mean_corr': mean_corr, 'min_corr': min_corr,
                'eff_rank': eff_rank, 'null_dim': null_dim,
            }

            print(f"  {name:<12s} {k:4d} {mean_corr:10.4f} {min_corr:10.4f}"
                  f" {eff_rank:8d} {null_dim:9d}")

        mean_corr_all = np.mean([r['mean_corr'] for r in type_results.values()])
        print(f"\n  Total within-type null dimensions: {total_null}")
        print(f"  Mean within-type correlation: {mean_corr_all:.4f}")

        all_results[label] = {
            'global_rank_90': rank_90,
            'global_rank_99': rank_99,
            'null_global': null_global,
            'pct_ident': pct_ident,
            'types': type_results,
            'total_null': total_null,
            'mean_corr': mean_corr_all,
        }

    # Summary comparison
    if len(all_results) > 1:
        print(f"\n\n{'=' * 90}")
        print(f"  SUMMARY: How noise breaks within-type degeneracy")
        print(f"{'=' * 90}\n")

        print(f"  {'Condition':<25s} {'Rank (99%)':>11s} {'Null (per-n)':>13s}"
              f" {'% ident':>8s} {'Null (per-type)':>16s}")
        print(f"  {'─' * 25} {'─' * 11} {'─' * 13} {'─' * 8} {'─' * 16}")

        for label, res in all_results.items():
            print(f"  {label:<25s} {res['global_rank_99']:>11d} "
                  f"{res['null_global']:>13d} {res['pct_ident']:>7.1f}% "
                  f"{res['total_null']:>16d}")

        print()
        print("  Noise → lower correlation → higher rank → smaller null space")
        print("  → more weights identifiable → higher R²_W")
        print()

        # Per-type comparison for top degenerate types
        print(f"  Top degenerate types — correlation change with noise:")
        print(f"  {'Type':<12s}", end="")
        for label in all_results:
            short = label.split('(')[0].strip()
            print(f" {short:>14s}", end="")
        print()
        print(f"  {'─' * 12}", end="")
        for _ in all_results:
            print(f" {'─' * 14}", end="")
        print()

        # Find types with highest null_dim in noise-free
        first_key = list(all_results.keys())[0]
        types_sorted = sorted(
            all_results[first_key]['types'].items(),
            key=lambda x: x[1]['null_dim'], reverse=True
        )

        for tid, _ in types_sorted[:15]:
            name = INDEX_TO_NAME.get(tid, f'type_{tid}')
            print(f"  {name:<12s}", end="")
            for label, res in all_results.items():
                tr = res['types'].get(tid, {})
                mc = tr.get('mean_corr', float('nan'))
                nd = tr.get('null_dim', 0)
                print(f"  {mc:.3f} ({nd:3d})", end="")
            print()


if __name__ == '__main__':
    main()
