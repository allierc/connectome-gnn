"""Multi-point g_phi identity loss for block 04 -- best-of-combination.

Penalizes |g_phi(v_k) - v_k|^2 at k uniformly-spaced voltages in [0, 2*xnorm],
forcing g_phi toward the identity function across the full operating range.

The existing regularizers constrain only two points:
  - g_phi_zero_intercept: g_phi(0) -> 0
  - g_phi_norm: g_phi(2*xnorm) -> 2*xnorm

A 64-hidden-unit MLP can easily interpolate through 2 points while having
arbitrary curvature (and thus non-unit gain) at intermediate voltages. This
lets g_phi absorb a scale factor from W, producing W slope bias (~0.51).

This loss fills k-2 interior points on the identity line, constraining
g_phi's gain across the full voltage range and removing the scale-absorption
pathway.
"""

from __future__ import annotations

import torch


def g_phi_identity_loss(
    model,
    in_features_edge: torch.Tensor,
    ids: torch.Tensor,
    xnorm: float,
    g_phi_positive: bool = False,
    n_pts: int = 10,
) -> torch.Tensor:
    """Multi-point g_phi identity loss: penalizes g_phi(v) != v at k uniformly-spaced voltages.

    For each of n_pts voltages v_k in [0, 2*xnorm], builds a copy of
    in_features_edge with voltage column set to v_k, evaluates g_phi,
    and penalizes (g_phi(v_k) - v_k)^2.

    When g_phi_positive=True the model uses h(v) = g_phi(v)^2, so the
    identity target becomes g_phi(v_k)^2 = v_k, i.e. g_phi(v_k) = sqrt(v_k).

    PASS CONDITION:
      (1) On a randomly initialized flyvis model, mean |g_phi(v_k) - v_k|^2
          across k=10 voltages in [0, 2*xnorm] exceeds 1e-4, confirming the
          loss targets a real degree of freedom.
      (2) The RMS identity deviation at the 8 intermediate points (excluding
          v=0 and v=2*xnorm) is >= 50% of the RMS at all 10 points,
          confirming the interior points contribute meaningful signal beyond
          the existing 2-point constraint.
      (3) Gradient norm w.r.t. g_phi parameters > 0 (differentiable).
      (4) All outputs finite (no NaN/Inf).

    Args:
        model: Object with .g_phi (nn.Module) -- the edge message MLP.
        in_features_edge: (N, 1+emb_dim) tensor -- precomputed g_phi input
            features (voltage in col 0, embeddings in cols 1:).
        ids: (M,) index tensor -- which neurons to evaluate.
        xnorm: Float -- voltage normalization constant.
        g_phi_positive: If True, square g_phi output (matches h(v)=g_phi(v)^2
            convention) and target becomes sqrt(v_k).
        n_pts: Number of uniformly-spaced voltage points (default 10).

    Returns:
        Scalar MSE-based loss with gradient through g_phi parameters.
    """
    device = in_features_edge.device

    # k uniformly-spaced voltages in [0, 2*xnorm]
    v_grid = torch.linspace(0.0, 2.0 * xnorm, n_pts, device=device)  # (n_pts,)

    total_loss = torch.zeros(1, device=device)

    for v_k in v_grid:
        # Clone and overwrite voltage column
        inp = in_features_edge[ids].clone().detach()
        inp[:, 0] = v_k.item()

        out = model.g_phi(inp.float())  # (M, 1)

        if g_phi_positive:
            out = out ** 2
            # Target: h(v) = g_phi(v)^2 = v  =>  we penalize (g_phi(v)^2 - v)^2
            target = v_k
        else:
            # Target: g_phi(v) = v
            target = v_k

        total_loss = total_loss + ((out - target) ** 2).mean()

    # Average over grid points
    return total_loss / n_pts
