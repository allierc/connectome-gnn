"""Block 08 — W type-equivariance regularizer (anchor class 4).

Penalizes per-edge weight deviation from the mean weight of its
(presynaptic_type, postsynaptic_type) group. This injects connectome
structural information directly into W: biologically, synapses between
the same cell-type pair carry similar weights.

This breaks the W<->f_theta scale degeneracy because the constraint
operates directly on W, orthogonal to f_theta's capacity. If f_theta
absorbs a scale factor k on the msg branch, W must shift by 1/k — but
this shifts ALL type-pair group means simultaneously, while the
*within-group variance* remains fixed by the data. The regularizer
penalizes exactly that variance, making scale absorption costly.

Uses scatter_add for O(E) computation — safe at flyvis scale (434K edges).
No Pydantic access, no numpy, no new imports beyond torch.
torch.compile-safe: all tensors are GPU, no Python data-dependent control flow.
"""

import torch


def w_type_equivariance_loss(
    W: torch.Tensor,
    edges: torch.Tensor,
    type_ids: torch.Tensor,
    n_types: int,
) -> torch.Tensor:
    """Penalize W deviation from (src_type, dst_type) group means.

    For each edge (i -> j), computes the mean weight of all edges
    sharing the same (type[i], type[j]) pair, then returns the L2 norm
    of deviations from those group means.

    Args:
        W: (n_edges, 1) or (n_edges,) learnable edge weights.
        edges: (2, n_edges) long tensor — [source_neurons, dest_neurons].
        type_ids: (n_neurons,) long tensor of cell-type indices.
        n_types: Number of distinct cell types.

    Returns:
        Scalar tensor: L2 norm of per-edge deviations from type-pair
        group means. Zero iff W is perfectly type-equivariant.

    PASS CONDITION:
      (1) Loss > 0 for random (non-equivariant) W at flyvis scale
          (434K edges, 65 types).
      (2) Loss < 1e-1 for perfectly type-equivariant W (all edges in
          a group share the same weight). Nonzero due to float32
          scatter_add rounding on 434K edges — 4 OoM below random W.
      (3) Gradient norm w.r.t. W > 0 (differentiable).
      (4) All outputs finite (no NaN/Inf).
    """
    w = W.squeeze()  # (E,)
    src = edges[0]   # (E,) long
    dst = edges[1]   # (E,) long

    # Map each edge to its (src_type, dst_type) pair index
    src_type = type_ids[src]  # (E,)
    dst_type = type_ids[dst]  # (E,)
    pair_id = src_type * n_types + dst_type  # (E,) — unique per type pair

    n_pairs = n_types * n_types

    # Compute group means via scatter_add
    group_sum = torch.zeros(n_pairs, device=w.device, dtype=w.dtype)
    group_count = torch.zeros(n_pairs, device=w.device, dtype=w.dtype)
    group_sum.scatter_add_(0, pair_id, w)
    group_count.scatter_add_(0, pair_id, torch.ones_like(w))
    group_mean = group_sum / group_count.clamp(min=1)  # (n_pairs,)

    # Per-edge deviation from its group mean
    edge_mean = group_mean[pair_id]  # (E,)
    deviation = w - edge_mean

    return deviation.norm(2)
