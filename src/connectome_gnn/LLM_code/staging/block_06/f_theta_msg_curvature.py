"""Block 06 — f_theta msg curvature penalty (anchor class 1).

Penalizes the discrete second derivative of f_theta w.r.t. the msg column
of in_features. Zero curvature <==> f_theta is locally linear in msg at
the operating point, which collapses the W-f_theta scale degeneracy to a
1D linear scale that the optimizer can resolve via existing anchors.

Uses the identical compile-safe pattern as f_theta_msg_diff: clone().detach()
+ column perturbation + model.f_theta() forward. No Pydantic access, no
numpy, no new imports beyond torch.
"""

import torch


def f_theta_msg_curvature_loss(
    model,
    in_features: torch.Tensor,
    ids_batch: torch.Tensor,
    embedding_dim: int,
    delta_msg: float,
) -> torch.Tensor:
    """Penalize f_theta nonlinearity in msg via 3-point discrete curvature.

    Evaluates f_theta at (msg-delta, msg, msg+delta) using the current
    training batch's in_features, computes the discrete second derivative
    [f(msg+d) + f(msg-d) - 2*f(msg)], and returns its squared L2 norm.
    Zero curvature <==> f_theta is linear in msg at the operating point.

    Follows the exact torch.compile-safe pattern of f_theta_msg_diff
    (clone().detach() + column perturbation + f_theta forward).

    Args:
        model: Neural GNN model with a .f_theta MLP attribute.
        in_features: (N, D) tensor from model forward pass. Layout:
            [v (col 0), embedding (cols 1..embedding_dim), msg (col embedding_dim+1), excitation].
        ids_batch: (B,) indices selecting the batch neurons from in_features.
        embedding_dim: Number of embedding dimensions (to locate the msg column).
        delta_msg: Perturbation magnitude for the 3-point stencil.

    Returns:
        Scalar tensor: squared L2 norm of the discrete second derivative,
        summed over batch neurons.

    PASS CONDITION:
      (1) A randomly initialised f_theta MLP (3-layer tanh, input_size=4,
          hidden=64) exhibits measurable curvature: mean |curvature| > 1e-6
          at 1000 synthetic operating points.
      (2) Gradient norm w.r.t. f_theta parameters > 0 (differentiable).
      (3) All outputs finite (no NaN/Inf).
    """
    msg_col = embedding_dim + 1

    # f(msg) — the center point
    base = in_features.clone().detach()
    pred_center = model.f_theta(base)

    # f(msg + delta)
    in_plus = in_features.clone().detach()
    in_plus[:, msg_col] = in_plus[:, msg_col] + delta_msg
    pred_plus = model.f_theta(in_plus)

    # f(msg - delta)
    in_minus = in_features.clone().detach()
    in_minus[:, msg_col] = in_minus[:, msg_col] - delta_msg
    pred_minus = model.f_theta(in_minus)

    # Discrete second derivative: f(msg+d) + f(msg-d) - 2*f(msg)
    curvature = pred_plus[ids_batch] + pred_minus[ids_batch] - 2.0 * pred_center[ids_batch]

    # Return squared L2 norm (mean over batch for scale invariance)
    return (curvature ** 2).mean()
