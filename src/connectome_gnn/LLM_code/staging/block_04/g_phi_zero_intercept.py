"""g_phi zero-intercept regularizer for block 04 -- best-of-combination.

Penalizes g_phi output at zero voltage input (v=0), forcing the effective
edge function h(v) = g_phi(v, a)^2 through the origin.  Combined with the
existing g_phi_norm single-point constraint at v = 2*xnorm, this creates a
two-point identity anchor that breaks the g_phi-W scale degeneracy.

The winner baseline has W slope ~0.51 (model under-estimates weight
magnitudes ~2x).  The free intercept at v=0 lets g_phi absorb a constant
offset that shifts W recovery.  Penalizing g_phi(0, a) -> 0 removes this
degree of freedom.
"""

from __future__ import annotations

import torch


def g_phi_zero_intercept_loss(
    model,
    in_features_edge: torch.Tensor,
    ids: torch.Tensor,
    g_phi_positive: bool = False,
) -> torch.Tensor:
    """Penalize g_phi output at zero voltage input (origin anchor).

    Sets the voltage column (col 0) of in_features_edge to 0, evaluates
    g_phi, applies squaring if g_phi_positive, and returns the L2 norm.
    Forces g_phi(v=0, emb) ~ 0 for all neurons.

    PASS CONDITION:
      (1) g_phi(v=0, emb) has nonzero mean |output| (> 1e-4) for a
          randomly initialized model, confirming the loss targets a
          real degree of freedom.
      (2) Loss gradient norm w.r.t. g_phi parameters > 0.
      (3) All outputs finite (no NaN/Inf).

    Args:
        model: Object with .g_phi (nn.Module) -- the edge message MLP.
        in_features_edge: (N, 1+emb_dim) tensor -- precomputed g_phi input
            features (voltage in col 0, embeddings in cols 1:).  Already
            built by get_in_features_g_phi in the regularizer's compute().
        ids: (M,) index tensor -- which neurons to evaluate (subset for
            efficiency, same as used by g_phi_diff/g_phi_norm).
        g_phi_positive: If True, square the g_phi output before penalizing
            (matches the h(v) = g_phi(v)^2 convention).

    Returns:
        Scalar L2-norm loss with gradient through g_phi parameters.
    """
    # Clone and zero the voltage column -- embeddings stay as-is
    in_zero = in_features_edge[ids].clone().detach()
    in_zero[:, 0] = 0.0

    out = model.g_phi(in_zero.float())  # (M, 1)
    if g_phi_positive:
        out = out ** 2

    return out.norm(2)
