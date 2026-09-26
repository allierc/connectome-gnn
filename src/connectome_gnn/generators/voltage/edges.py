"""Edits of the ground-truth connectivity: null edges, ablation, removal after generation.

The choice of WHICH edges is a pure function of explicit inputs here:
``sample_null_edges`` (global stdlib ``random``), ``ablation_mask`` and
``removal_kept_indices`` (private ``np.random.RandomState``s, seeded from the
config, so they never touch the global streams).

Applying an edit MUTATES the ODE parameters in place, exactly as legacy did:
``ode_params`` is owned by the chain (a linear resource) and nothing else
holds its tensors at that point. Copying instead would be equal in value, but
``torch.save`` writes a tensor's whole storage, so a copy is only
byte-identical while every saved tensor owns exactly its storage; keeping
legacy's in-place writes does not depend on that.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from connectome_gnn.log import get_logger

logger = get_logger(__name__)

GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'


def sample_null_edges(edge_index, edges) -> list:
    """(source, target) pairs to add as zero-weight edges: not existing, not self-loops.

    Sources and targets range over ``edges.n_neurons_config`` (``sim.n_neurons``,
    the CONFIG value, not the network's size). Draws from the GLOBAL stdlib
    ``random``, which generation never seeds (see QUIRKS: reproducible only
    because the DAVIS dataset reseeds it).
    """
    import random

    n_neurons = edges.n_neurons_config
    src_np = edge_index[0].cpu().numpy()
    dst_np = edge_index[1].cpu().numpy()
    existing_edges = set(zip(src_np, dst_np))
    extra_edges = []

    if edges.null_edges_mode == "per_column":
        # Per pre-synaptic neuron: add a proportional number of false targets
        # Compute out-degree per source neuron
        from collections import Counter

        out_degree = Counter(src_np.tolist())
        total_real = edge_index.shape[1]
        ratio = edges.n_extra_null_edges / total_real

        # Build per-neuron target sets for fast lookup
        targets_by_source = {}
        for s, d in zip(src_np, dst_np):
            targets_by_source.setdefault(int(s), set()).add(int(d))

        all_neurons = list(range(n_neurons))
        for source in range(n_neurons):
            deg = out_degree.get(source, 0)
            if deg == 0:
                continue
            n_false = max(1, int(round(deg * ratio)))
            existing_targets = targets_by_source.get(source, set())
            # Sample false targets not already connected and not self
            candidates = [t for t in all_neurons if t != source and t not in existing_targets]
            if len(candidates) <= n_false:
                chosen = candidates
            else:
                chosen = random.sample(candidates, n_false)
            for t in chosen:
                extra_edges.append([source, t])
                existing_targets.add(t)

        logger.info(
            f"per_column: added {len(extra_edges)} false edges "
            f"(requested ratio {ratio:.2f}, effective {len(extra_edges) / total_real:.2f})"
        )
    else:
        # Random: sample uniformly across the full matrix
        max_attempts = edges.n_extra_null_edges * 10
        attempts = 0
        while len(extra_edges) < edges.n_extra_null_edges and attempts < max_attempts:
            source = random.randint(0, n_neurons - 1)
            target = random.randint(0, n_neurons - 1)
            if (source, target) not in existing_edges and source != target:
                extra_edges.append([source, target])
                existing_edges.add((source, target))
            attempts += 1
    return extra_edges


def add_null_edges(ode_params, edge_index, edges, device):
    """Append ``sample_null_edges`` with weight 0 to ``ode_params`` (in place); return the new edge_index."""
    logger.info(f"adding {edges.n_extra_null_edges} extra null edges (mode={edges.null_edges_mode})...")
    extra_edges = sample_null_edges(edge_index, edges)
    if extra_edges:
        extra_edge_index = torch.tensor(extra_edges, dtype=torch.long, device=device).t()
        edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
        ode_params.edge_index = edge_index
        ode_params.W = torch.cat([ode_params.W, torch.zeros(len(extra_edges), device=device)])
        logger.info(f"Total extra edges added: {len(extra_edges)}")
    return edge_index


def ablation_mask(n_edges: int, ratio: float, seed: int, device):
    """(keep mask, n ablated): ``round(n_edges * ratio)`` edges chosen by RandomState(seed)."""
    rng = np.random.RandomState(seed)
    n_ablate = int(np.round(n_edges * ratio))
    ablate_indices = rng.choice(n_edges, size=n_ablate, replace=False)
    mask = torch.ones(n_edges, dtype=torch.bool, device=device)
    mask[ablate_indices] = False
    return mask, n_ablate


def removal_kept_indices(edge_index, edges):
    """Indices of the edges the pruned ode_params keeps.

    From ``edge_mask_path`` when that file exists; otherwise drawn with
    RandomState(edge_removal_seed): per_column removes round(ratio * out-degree)
    edges of every source (at least 1, at most out-degree - 1), random keeps
    int(n * (1 - ratio)) edges uniformly.
    """
    n_total = edge_index.shape[1]
    edge_mask_path = edges.edge_mask_path
    if edge_mask_path and os.path.exists(edge_mask_path):
        kept_indices = torch.load(edge_mask_path, weights_only=True)
        print(f"{GREEN}[GENERATE] mask loaded from {edge_mask_path}: "
              f"{len(kept_indices)}/{n_total} edges kept{RESET}")
        return kept_indices
    if edge_mask_path:
        print(f"{RED}[GENERATE] edge_mask_path set but NOT FOUND: {edge_mask_path} "
              f"— computing new mask{RESET}")
    else:
        print(f"{GREEN}[GENERATE] no edge_mask_path — computing new mask "
              f"(mode={edges.edge_removal_mode}, "
              f"ratio={edges.edge_removal_ratio}){RESET}")
    rng_rm = np.random.RandomState(edges.edge_removal_seed)
    if edges.edge_removal_mode == 'per_column':
        src_np = edge_index[0].cpu().numpy()
        keep_mask = np.ones(n_total, dtype=bool)
        for source in np.unique(src_np):
            source_edges = np.where(src_np == source)[0]
            n_remove = max(1, int(round(len(source_edges) * edges.edge_removal_ratio)))
            if n_remove >= len(source_edges):
                n_remove = len(source_edges) - 1
            remove_idx = rng_rm.choice(source_edges, n_remove, replace=False)
            keep_mask[remove_idx] = False
        return np.where(keep_mask)[0]
    n_keep = int(n_total * (1 - edges.edge_removal_ratio))
    return np.sort(rng_rm.choice(n_total, n_keep, replace=False))


def remove_edges(ode_params, edge_index, edges, store, save, device):
    """Prune ``ode_params`` to ``removal_kept_indices`` (in place); return the pruned edge_index.

    Applied AFTER both splits are generated: the activity comes from the full
    connectome and only the connectivity the GNN is given is incomplete. With
    ``save`` the full W and edge_index are written first (weights_full.pt,
    edge_index_full.pt) and the kept indices after (kept_edge_indices.pt). The
    printout is red when the removed share misses the requested one by 2 % or more.
    """
    # --- Edge removal: applied AFTER activity generation ---
    # Activity data (x_list, y_list) was generated with the full connectome above.
    # Now prune ode_params so the GNN only sees the incomplete adjacency matrix.
    print(f"{GREEN}[GENERATE] activity generated with full connectivity: "
          f"edge_index={edge_index.shape}  W={ode_params.W.shape}{RESET}")
    if not edges.edge_removal_ratio > 0:
        print(f"{GREEN}[GENERATE] no edge removal (ratio=0){RESET}")
        return edge_index
    if save:
        torch.save(ode_params.W.clone(), store.path("weights_full.pt"))
        torch.save(edge_index.clone(), store.path("edge_index_full.pt"))

    n_total = edge_index.shape[1]
    kept_indices = removal_kept_indices(edge_index, edges)
    edge_index = edge_index[:, kept_indices]
    ode_params.edge_index = edge_index
    ode_params.W = ode_params.W[kept_indices]
    pct_removed = (1 - len(kept_indices) / n_total) * 100
    expected_pct = edges.edge_removal_ratio * 100
    color = GREEN if abs(pct_removed - expected_pct) < 2 else RED
    print(f"{color}[GENERATE] ode_params pruned: edge_index={edge_index.shape}  "
          f"W={ode_params.W.shape}  removed={pct_removed:.1f}% "
          f"(expected {expected_pct:.0f}%){RESET}")
    if save:
        torch.save(torch.tensor(kept_indices, device=device),
                   store.path("kept_edge_indices.pt"))
    return edge_index
