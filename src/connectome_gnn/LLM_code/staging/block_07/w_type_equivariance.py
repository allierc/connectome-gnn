"""Block 07 — W type-equivariance regularizer (anchor class 4).

Penalizes within-type-pair weight variance: for each (src_type, dst_type)
pair, all edges in that group should have similar W values.

Biological motivation: in the connectome, connection strength is primarily
determined by cell type (65 types in flyvis), not individual neuron identity.
The per-edge W has 434K degrees of freedom but the true structure lives in a
~4225-dimensional type-pair space. Penalizing within-group variance injects
this prior, reducing the effective DoF and restricting the W↔f_theta scale
degeneracy to a single global factor (which existing L1 already constrains).

Implementation: scatter-based, fully differentiable, torch.compile-safe.
No Pydantic access, no numpy, no attribute reads on config objects.
"""

import torch


def w_type_equivariance_loss(
    model_W: torch.Tensor,
    edges: torch.Tensor,
    type_ids: torch.Tensor,
    n_neuron_types: int,
) -> torch.Tensor:
    """Penalize within-type-pair weight variance.

    For each (source_type, target_type) cell-type pair, computes the
    mean-squared deviation of edge weights from the pair mean. Returns
    the average over all edges.

    PASS CONDITION:
      (1) Random W with 500 edges, 5 types: loss > 0.
      (2) Block-constant W (same value per type pair): loss < 1e-10.
      (3) Gradient norm w.r.t. W > 0 (differentiable).
      (4) All outputs finite (no NaN/Inf).

    Args:
        model_W: (E_total, 1) or (E_total,) edge weight tensor (learnable).
            Only the first edges.shape[1] entries are used.
        edges: (2, E) long tensor — [source_neurons, dest_neurons].
        type_ids: (N,) long tensor — cell-type index per neuron.
        n_neuron_types: Number of distinct cell types.

    Returns:
        Scalar tensor: mean squared deviation from per-pair means.
        Gradient flows through model_W.
    """
    n_edges = edges.shape[1]
    W = model_W[:n_edges].squeeze(-1)  # (E,)

    # Map each edge to its (src_type, dst_type) pair index
    src_types = type_ids[edges[0]]  # (E,)
    dst_types = type_ids[edges[1]]  # (E,)
    pair_ids = src_types * n_neuron_types + dst_types  # (E,) unique pair id

    n_pairs = n_neuron_types * n_neuron_types

    # Per-pair sum and count via scatter_add
    pair_sum = torch.zeros(n_pairs, device=W.device, dtype=W.dtype)
    pair_sum.scatter_add_(0, pair_ids, W)

    pair_count = torch.zeros(n_pairs, device=W.device, dtype=W.dtype)
    pair_count.scatter_add_(0, pair_ids, torch.ones_like(W))
    pair_count = pair_count.clamp(min=1.0)

    # Per-pair mean
    pair_mean = pair_sum / pair_count  # (n_pairs,)

    # Per-edge deviation from its pair mean
    edge_mean = pair_mean[pair_ids]  # (E,)
    deviation = W - edge_mean  # (E,)

    # Mean squared deviation
    loss = (deviation ** 2).mean()

    return loss
