#!/usr/bin/env python3
"""Compute the identifiability table: structural mean-k vs CV-seed SNR per cell type.

For each presynaptic cell type α, compute:
  - structural: mean_k_α = mean k_{i,α} across all outgoing edges (k=1 → identifiable)
  - empirical:  SNR(α)   = mean |W̄_e| / σ_e across CV seeds (high → seeds agree)

Then print and save a combined table for use in the degeneracy_identifiability.tex paper.

Usage:
    python scripts/compute_type_identifiability_table.py [--log_dir LOG_DIR] [--out OUT_CSV]

The script requires:
    - graphs_data/fly/flyvis_noise_005/ode_params.pt        (edge_index)
    - graphs_data/fly/flyvis_noise_005/x_list_train/neuron_type.zarr
    - log/fly/flyvis_noise_005_cvXX/corrected_W.pt          (5 CV seeds)

Uses INDEX_TO_NAME from connectome_gnn.metrics as the canonical type mapping.
"""

import os
import sys
import argparse
import numpy as np
import glob
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

import torch
import zarr
from connectome_gnn.metrics import INDEX_TO_NAME
from connectome_gnn.utils import graphs_data_path, log_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_cv_seeds(base_name='flyvis_noise_005', n_seeds=5, log_dir=None):
    """Return list of corrected_W.pt paths for CV seeds cv00..cv0N-1.

    Searches in order:
      1. log_dir/<fold_name>/results/corrected_W.pt
      2. log_dir/<fold_name>/corrected_W.pt
      3. log_path(fly/<fold_name>, corrected_W.pt) [repo-relative]
    """
    paths = []
    for i in range(n_seeds):
        fold_name = f'{base_name}_cv{i:02d}'
        candidates = []
        if log_dir:
            candidates.append(os.path.join(log_dir, fold_name, 'results', 'corrected_W.pt'))
            candidates.append(os.path.join(log_dir, fold_name, 'corrected_W.pt'))
        candidates.append(log_path(f'fly/{fold_name}', 'results/corrected_W.pt'))
        candidates.append(log_path(f'fly/{fold_name}', 'corrected_W.pt'))

        found = None
        for c in candidates:
            if os.path.isfile(c):
                found = c
                break
        if found:
            paths.append(found)
        else:
            print(f"  Warning: corrected_W.pt not found for {fold_name} (tried: {candidates[:2]})")
    return paths


def load_w_stack(cv_paths):
    """Load list of corrected_W.pt files → (K, E) numpy array."""
    ws = []
    for p in cv_paths:
        cw = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(cw, torch.Tensor):
            ws.append(cw.detach().squeeze().numpy())
        else:
            raise TypeError(f"Unexpected type in {p}: {type(cw)}")
    assert len(set(w.shape for w in ws)) == 1, "W tensors have different shapes"
    return np.stack(ws, axis=0)   # (K, E)


def build_type_groups(edge_index_np, ntype_np, n_edges):
    """Build (dst_neuron, src_type_id) → [edge_indices] groups."""
    src = edge_index_np[0]
    dst = edge_index_np[1]
    groups = defaultdict(list)
    for e in range(n_edges):
        key = (int(dst[e]), int(ntype_np[src[e]]))
        groups[key].append(e)
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_path', default=None,
                        help='Path to flyvis_noise_005 graphs_data dir')
    parser.add_argument('--log_dir', default=None,
                        help='Root log dir containing flyvis_noise_005_cvXX/ subdirs')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--out', default=None,
                        help='Output CSV path (default: scripts/type_identifiability.csv)')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Locate dataset
    # -----------------------------------------------------------------------
    CANDIDATES = [
        '/groups/saalfeld/home/allierc/GraphData/graphs_data/fly/flyvis_noise_005',
        '/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_005',
    ]
    dataset_path = args.dataset_path
    if dataset_path is None:
        for c in CANDIDATES:
            if os.path.isdir(c):
                dataset_path = c
                break
    assert dataset_path and os.path.isdir(dataset_path), \
        f"Dataset not found. Pass --dataset_path. Tried: {CANDIDATES}"
    print(f"Dataset:  {dataset_path}")

    # -----------------------------------------------------------------------
    # 2. Load edge_index and neuron_type
    # -----------------------------------------------------------------------
    ode = torch.load(os.path.join(dataset_path, 'ode_params.pt'),
                     map_location='cpu', weights_only=False)
    edge_index = ode['edge_index'].numpy()   # (2, E)
    src_nodes  = edge_index[0]
    dst_nodes  = edge_index[1]
    n_edges    = edge_index.shape[1]
    print(f"Edges:    {n_edges:,}")

    ntype_path = os.path.join(dataset_path, 'x_list_train', 'neuron_type.zarr')
    ntype = np.array(zarr.open_array(ntype_path, mode='r'))   # (N,) int32, canonical INDEX_TO_NAME
    print(f"Neurons:  {len(ntype):,}  unique types: {len(np.unique(ntype))}")

    # Verify mapping is sane
    assert INDEX_TO_NAME[23] == 'R1', "INDEX_TO_NAME sanity check failed"

    # -----------------------------------------------------------------------
    # 3. Build (dst, src_type) groups → structural k per edge
    # -----------------------------------------------------------------------
    print("Building (dst, src_type) groups ...")
    groups = build_type_groups(edge_index, ntype, n_edges)

    # For each presynaptic type, collect k values of all its outgoing edges
    pre_type_k = defaultdict(list)    # type_id → list of k values
    pre_type_edges = defaultdict(list)  # type_id → list of edge indices
    for (dst, src_type_id), edge_list in groups.items():
        k = len(edge_list)
        for e in edge_list:
            pre_type_k[src_type_id].append(k)
            pre_type_edges[src_type_id].append(e)

    print(f"Groups:   {len(groups):,} (dst, src_type) pairs")

    # -----------------------------------------------------------------------
    # 4. Load CV seed W stack → empirical SNR per edge
    # -----------------------------------------------------------------------
    cv_paths = find_cv_seeds('flyvis_noise_005', args.n_seeds, args.log_dir)
    print(f"CV seeds: found {len(cv_paths)} / {args.n_seeds}")

    if len(cv_paths) >= 2:
        W_stack = load_w_stack(cv_paths)  # (K, E)
        print(f"W_stack:  {W_stack.shape}")
        W_mean = W_stack.mean(axis=0)     # (E,)
        W_std  = W_stack.std(axis=0)      # (E,)
        W_std  = np.maximum(W_std, 1e-12)  # avoid div/0
        snr_per_edge = np.abs(W_mean) / W_std
    else:
        print("  Warning: not enough CV seeds for SNR; SNR column will be NaN")
        snr_per_edge = None

    # -----------------------------------------------------------------------
    # 5. Aggregate by presynaptic type
    # -----------------------------------------------------------------------
    print("\n=== Per presynaptic type: structural mean_k and empirical SNR ===")
    print(f"{'Type':15s} {'mean_k':>8s} {'f_ident':>8s} {'n_edges':>8s} {'SNR(mean)':>10s} {'SNR(med)':>9s}")

    rows = []
    for type_id in sorted(pre_type_k.keys()):
        name    = INDEX_TO_NAME.get(type_id, f'type{type_id}')
        k_arr   = np.array(pre_type_k[type_id])
        mean_k  = float(np.mean(k_arr))
        f_ident = float(np.mean(k_arr == 1))
        n_e     = len(k_arr)

        if snr_per_edge is not None:
            edge_idx = np.array(pre_type_edges[type_id])
            snr_mean = float(np.mean(snr_per_edge[edge_idx]))
            snr_med  = float(np.median(snr_per_edge[edge_idx]))
        else:
            snr_mean = float('nan')
            snr_med  = float('nan')

        print(f"{name:15s} {mean_k:>8.2f} {f_ident:>8.3f} {n_e:>8d} {snr_mean:>10.2f} {snr_med:>9.2f}")
        rows.append({
            'type_id': type_id,
            'name': name,
            'mean_k': mean_k,
            'f_ident': f_ident,
            'n_edges': n_e,
            'snr_mean': snr_mean,
            'snr_med': snr_med,
        })

    # -----------------------------------------------------------------------
    # 6. Save CSV
    # -----------------------------------------------------------------------
    out_csv = args.out or os.path.join(SCRIPT_DIR, 'type_identifiability.csv')
    import csv
    fieldnames = ['type_id', 'name', 'mean_k', 'f_ident', 'n_edges', 'snr_mean', 'snr_med']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")

    # -----------------------------------------------------------------------
    # 7. Print the key table for the paper (sorted by SNR descending)
    # -----------------------------------------------------------------------
    if snr_per_edge is not None:
        sorted_rows = sorted(rows, key=lambda r: -r['snr_mean'])
        print("\n=== Most identifiable (seeds agree, high SNR) ===")
        for r in sorted_rows[:12]:
            print(f"  {r['name']:15s} mean_k={r['mean_k']:.1f}  f_ident={r['f_ident']:.2f}  SNR={r['snr_mean']:.1f}")
        print("\n=== Least identifiable (seeds disagree, low SNR) ===")
        for r in sorted_rows[-12:]:
            print(f"  {r['name']:15s} mean_k={r['mean_k']:.1f}  f_ident={r['f_ident']:.2f}  SNR={r['snr_mean']:.3f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
