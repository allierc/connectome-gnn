#!/usr/bin/env python3
"""Validate the identifiable/duplicate-type edge partition at the edge level.

The theory predicts that the identifiable/duplicate-type classification is a
property of each *edge* W_{ij}:
  - Identifiable (k_{i,α}=1): edge is uniquely determined by dynamics → seeds agree
  - Duplicate-type (k_{i,α}>1): k-1 free redistributive directions → seeds disagree

This script tests that prediction directly by comparing the empirical cross-seed
SNR = |W̄_e| / σ_e for k=1 edges vs k>1 edges, at the edge level.

Key outputs:
  1. SNR distribution by k-class (identifiable vs duplicate)
  2. σ_e distribution (raw inter-seed std, independent of weight magnitude)
  3. Within-type comparison (same source type α, same weight scale)
  4. Cohen's d and Mann-Whitney U p-value for the two-group comparison
  5. Summary CSV: edge_partition_snr.csv

Usage:
    python scripts/edge_partition_snr.py [--log_dir LOG_DIR] [--out OUT_CSV]
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

import torch
import zarr
from connectome_gnn.metrics import INDEX_TO_NAME
from connectome_gnn.utils import graphs_data_path, log_path


# ---------------------------------------------------------------------------
# Helpers (shared with compute_type_identifiability_table.py)
# ---------------------------------------------------------------------------

def find_cv_seeds(base_name='flyvis_noise_005', n_seeds=5, log_dir=None):
    paths = []
    for i in range(n_seeds):
        fold_name = f'{base_name}_cv{i:02d}'
        candidates = []
        if log_dir:
            candidates.append(os.path.join(log_dir, fold_name, 'results', 'corrected_W.pt'))
            candidates.append(os.path.join(log_dir, fold_name, 'corrected_W.pt'))
        candidates.append(log_path(f'fly/{fold_name}', 'results/corrected_W.pt'))
        candidates.append(log_path(f'fly/{fold_name}', 'corrected_W.pt'))
        found = next((c for c in candidates if os.path.isfile(c)), None)
        if found:
            paths.append(found)
        else:
            print(f"  Warning: corrected_W.pt not found for {fold_name}")
    return paths


def load_w_stack(cv_paths):
    ws = []
    for p in cv_paths:
        cw = torch.load(p, map_location='cpu', weights_only=False)
        if isinstance(cw, torch.Tensor):
            ws.append(cw.detach().squeeze().numpy())
        else:
            raise TypeError(f"Unexpected type in {p}: {type(cw)}")
    assert len(set(w.shape for w in ws)) == 1, "W tensors have different shapes"
    return np.stack(ws, axis=0)   # (K, E)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_path', default=None)
    parser.add_argument('--log_dir',      default=None)
    parser.add_argument('--n_seeds',      type=int, default=5)
    parser.add_argument('--out',          default=None)
    args = parser.parse_args()

    # 1. Dataset ---------------------------------------------------------------
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

    # 2. Load edge_index + neuron types ----------------------------------------
    ode = torch.load(os.path.join(dataset_path, 'ode_params.pt'),
                     map_location='cpu', weights_only=False)
    edge_index = ode['edge_index'].numpy()     # (2, E)
    src_nodes  = edge_index[0]
    dst_nodes  = edge_index[1]
    n_edges    = edge_index.shape[1]
    print(f"Edges:    {n_edges:,}")

    ntype_path = os.path.join(dataset_path, 'x_list_train', 'neuron_type.zarr')
    ntype = np.array(zarr.open_array(ntype_path, mode='r'))
    print(f"Neurons:  {len(ntype):,}  types: {len(np.unique(ntype))}")
    assert INDEX_TO_NAME[23] == 'R1', "INDEX_TO_NAME sanity check failed"

    # 3. Build (dst, src_type) groups → k per edge -----------------------------
    print("Building groups ...")
    groups = defaultdict(list)
    for e in range(n_edges):
        key = (int(dst_nodes[e]), int(ntype[src_nodes[e]]))
        groups[key].append(e)

    k_per_edge = np.ones(n_edges, dtype=np.int32)     # k_{i,α} for each edge
    src_type_per_edge = ntype[src_nodes].astype(np.int32)

    for (dst, src_type_id), edge_list in groups.items():
        k = len(edge_list)
        for e in edge_list:
            k_per_edge[e] = k

    n_ident   = int((k_per_edge == 1).sum())
    n_dup     = int((k_per_edge >  1).sum())
    print(f"Identifiable edges (k=1):    {n_ident:,}  ({100*n_ident/n_edges:.1f}%)")
    print(f"Duplicate-type edges (k>1):  {n_dup:,}  ({100*n_dup/n_edges:.1f}%)")

    # 4. Load CV seed W stack --------------------------------------------------
    cv_paths = find_cv_seeds('flyvis_noise_005', args.n_seeds, args.log_dir)
    print(f"CV seeds: found {len(cv_paths)} / {args.n_seeds}")
    assert len(cv_paths) >= 2, "Need at least 2 CV seeds for SNR"

    W_stack = load_w_stack(cv_paths)   # (K, E)
    W_mean  = W_stack.mean(axis=0)    # (E,)
    W_std   = W_stack.std(axis=0)     # (E,)
    W_std   = np.maximum(W_std, 1e-12)
    snr     = np.abs(W_mean) / W_std   # (E,)

    # 5. Edge-level comparison: k=1 vs k>1 ------------------------------------
    mask_ident = (k_per_edge == 1)
    mask_dup   = (k_per_edge >  1)

    snr_ident = snr[mask_ident]
    snr_dup   = snr[mask_dup]
    std_ident = W_std[mask_ident]
    std_dup   = W_std[mask_dup]
    wabs_ident = np.abs(W_mean[mask_ident])
    wabs_dup   = np.abs(W_mean[mask_dup])

    def pct(arr, p): return np.percentile(arr, p)

    print("\n=== Edge-level SNR: identifiable (k=1) vs duplicate-type (k>1) ===")
    print(f"{'Metric':30s}  {'Ident (k=1)':>14s}  {'Dup (k>1)':>14s}")
    print("-"*62)
    print(f"{'SNR median':30s}  {pct(snr_ident,50):>14.3f}  {pct(snr_dup,50):>14.3f}")
    print(f"{'SNR mean':30s}  {snr_ident.mean():>14.3f}  {snr_dup.mean():>14.3f}")
    print(f"{'SNR 25th pct':30s}  {pct(snr_ident,25):>14.3f}  {pct(snr_dup,25):>14.3f}")
    print(f"{'SNR 75th pct':30s}  {pct(snr_ident,75):>14.3f}  {pct(snr_dup,75):>14.3f}")
    print(f"{'|W| median':30s}  {pct(wabs_ident,50):>14.4f}  {pct(wabs_dup,50):>14.4f}")
    print(f"{'σ_e median (inter-seed std)':30s}  {pct(std_ident,50):>14.4f}  {pct(std_dup,50):>14.4f}")

    # Mann-Whitney U test (one-sided: ident > dup)
    stat, p_mw = stats.mannwhitneyu(snr_ident, snr_dup, alternative='greater')
    n1, n2 = len(snr_ident), len(snr_dup)
    auroc = stat / (n1 * n2)   # = P(SNR_ident > SNR_dup)
    print(f"\nMann-Whitney U (SNR_ident > SNR_dup):")
    print(f"  AUROC = P(SNR_ident > SNR_dup) = {auroc:.4f}")
    print(f"  p-value                        = {p_mw:.2e}")

    # Cohen's d on log(SNR) (more normal after log transform)
    log_snr_ident = np.log1p(snr_ident)
    log_snr_dup   = np.log1p(snr_dup)
    pooled_std = np.sqrt((log_snr_ident.var() + log_snr_dup.var()) / 2)
    cohens_d = (log_snr_ident.mean() - log_snr_dup.mean()) / (pooled_std + 1e-12)
    print(f"  Cohen's d (log SNR)            = {cohens_d:.3f}")

    # 6. Within-type comparison ------------------------------------------------
    # For each source type, compare k=1 edges vs k>1 edges
    print("\n=== Within-type: ident vs dup edges per source type ===")
    print(f"{'Type':15s}  {'n_ident':>8s}  {'n_dup':>8s}  {'SNR_ident':>10s}  {'SNR_dup':>8s}  {'ratio':>7s}")

    within_type_rows = []
    for type_id in sorted(np.unique(src_type_per_edge)):
        name = INDEX_TO_NAME.get(int(type_id), f'type{type_id}')
        type_mask = (src_type_per_edge == type_id)
        mask_i = type_mask & mask_ident
        mask_d = type_mask & mask_dup
        ni, nd = mask_i.sum(), mask_d.sum()
        if ni == 0 or nd == 0:
            continue   # skip pure types for within-type comparison
        si = snr[mask_i].mean()
        sd = snr[mask_d].mean()
        ratio = si / (sd + 1e-9)
        print(f"  {name:13s}  {ni:>8d}  {nd:>8d}  {si:>10.2f}  {sd:>8.2f}  {ratio:>7.2f}x")
        within_type_rows.append({
            'type_id': int(type_id), 'name': name,
            'n_ident': int(ni), 'n_dup': int(nd),
            'snr_ident_mean': float(si), 'snr_dup_mean': float(sd),
            'ratio': float(ratio),
        })

    n_higher = sum(1 for r in within_type_rows if r['snr_ident_mean'] > r['snr_dup_mean'])
    n_mixed  = len(within_type_rows)
    print(f"\n  k=1 edges have higher SNR than k>1 edges in {n_higher}/{n_mixed} mixed types")

    # 7. Breakdown by k value --------------------------------------------------
    print("\n=== SNR vs k value (group size) ===")
    print(f"{'k':>5s}  {'n_edges':>9s}  {'SNR_mean':>10s}  {'SNR_med':>9s}  {'σ_e_med':>9s}")
    for k_val in sorted(np.unique(k_per_edge)):
        m = (k_per_edge == k_val)
        ne = m.sum()
        if ne < 10:
            continue
        print(f"  {k_val:>3d}  {ne:>9,d}  {snr[m].mean():>10.3f}  {np.median(snr[m]):>9.3f}  {np.median(W_std[m]):>9.4f}")

    # 8. Save CSV --------------------------------------------------------------
    out_csv = args.out or os.path.join(SCRIPT_DIR, 'edge_partition_snr.csv')
    import csv
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k_class', 'n_edges', 'snr_mean', 'snr_median',
                         'snr_p25', 'snr_p75', 'w_abs_median', 'sigma_median'])
        writer.writerow([
            'identifiable', n_ident,
            f"{snr_ident.mean():.4f}", f"{np.median(snr_ident):.4f}",
            f"{pct(snr_ident,25):.4f}", f"{pct(snr_ident,75):.4f}",
            f"{np.median(wabs_ident):.4f}", f"{np.median(std_ident):.4f}",
        ])
        writer.writerow([
            'duplicate_type', n_dup,
            f"{snr_dup.mean():.4f}", f"{np.median(snr_dup):.4f}",
            f"{pct(snr_dup,25):.4f}", f"{pct(snr_dup,75):.4f}",
            f"{np.median(wabs_dup):.4f}", f"{np.median(std_dup):.4f}",
        ])
    print(f"\nSaved: {out_csv}")
    print("Done.")


if __name__ == '__main__':
    main()
