"""f_theta msg linearity loss for block 04 — best-of-combination.

Penalizes f_theta nonlinearity in the synaptic input (msg) dimension.
The true ODE is dv/dt = -(v - V_rest)/tau + msg/tau, so dv/dt is
LINEAR in msg. An unconstrained MLP can compress or saturate the msg
response, absorbing W magnitude into f_theta and causing systematic
slope bias (~0.51 in the winner baseline). This loss forces proportional
msg response, breaking the W-vs-f_theta scale degeneracy.

Extracted from combined_msg_linearity_snr.py, stripping the SNR
component (falsified in block 01).
"""

from __future__ import annotations

import numpy as np
import torch


# ------------------------------------------------------------------ #
#  Utility: vectorized linspace + differentiable OLS
# ------------------------------------------------------------------ #

def _vectorized_linspace(
    starts: np.ndarray, ends: np.ndarray, n_pts: int, device: torch.device
) -> torch.Tensor:
    """Create (N, n_pts) tensor where row n spans [starts[n], ends[n]]."""
    t = torch.linspace(0, 1, n_pts, device=device)
    starts_t = torch.as_tensor(starts, dtype=torch.float32, device=device)
    ends_t = torch.as_tensor(ends, dtype=torch.float32, device=device)
    return starts_t[:, None] + t[None, :] * (ends_t - starts_t)[:, None]


def _torch_linear_fit(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable OLS: y approx slope * x + offset. Grad flows through y."""
    n_pts = x.shape[1]
    sx = x.sum(dim=1)
    sy = y.sum(dim=1)
    sxy = (x * y).sum(dim=1)
    sxx = (x * x).sum(dim=1)
    denom = n_pts * sxx - sx * sx
    slopes = (n_pts * sxy - sx * sy) / (denom + 1e-12)
    offsets = (sy - slopes * sx) / n_pts
    return slopes, offsets


# ------------------------------------------------------------------ #
#  Main mechanism: f_theta msg linearity loss
# ------------------------------------------------------------------ #

def f_theta_msg_linearity_loss(
    model,
    n_neurons: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    device: torch.device,
    n_pts: int = 50,
) -> torch.Tensor:
    """Penalize f_theta nonlinearity in the message (synaptic input) dimension.

    Evaluates f_theta at each neuron's mean voltage with msg swept over a
    data-driven range (+-2 sigma), fits a differentiable OLS line through
    the msg->output curve, and penalizes the squared residual.

    PASS CONDITION:
      (1) A randomly initialised f_theta MLP exhibits measurable msg
          nonlinearity: residual_MSE / total_output_variance > 0.01.
      (2) Gradient norm w.r.t. f_theta parameters > 0 (differentiable).
      (3) All outputs finite (no NaN/Inf) for N=200 synthetic neurons.

    Args:
        model: Object with .f_theta (nn.Module) and .a (N, emb_dim) embedding.
        n_neurons: Number of neurons to evaluate.
        mu: (N,) numpy array -- per-neuron mean voltage (used to fix v).
        sigma: (N,) numpy array -- per-neuron voltage std (used to scale msg range).
        device: Torch device.
        n_pts: Number of msg grid points per neuron (default 50).

    Returns:
        Scalar mean-squared residual loss with gradient through f_theta.
    """
    N = n_neurons
    emb_dim = model.a.shape[1]

    sigma_safe = np.maximum(sigma[:N], 1e-6)
    msg_starts = mu[:N] - 2.0 * sigma_safe
    msg_ends = mu[:N] + 2.0 * sigma_safe
    msg_grid = _vectorized_linspace(msg_starts, msg_ends, n_pts, device)  # (N, n_pts)

    mu_t = torch.as_tensor(mu[:N], dtype=torch.float32, device=device)  # (N,)
    a_detached = model.a[:N].detach()  # (N, emb_dim), block grad to embeddings

    # Build input features: v = mu (fixed), emb = a (detached), msg = grid, exc = 0
    v_flat = mu_t[:, None].expand(-1, n_pts).reshape(-1, 1)         # (N*n_pts, 1)
    emb_flat = a_detached[:, None, :].expand(-1, n_pts, -1).reshape(-1, emb_dim)
    msg_flat = msg_grid.reshape(-1, 1)                                # (N*n_pts, 1)
    exc_flat = torch.zeros_like(msg_flat)                             # (N*n_pts, 1)

    in_features = torch.cat([v_flat, emb_flat, msg_flat, exc_flat], dim=1)

    # Forward through f_theta — gradient flows through f_theta weights
    out = model.f_theta(in_features.float())                          # (N*n_pts, 1)
    func = out.squeeze(-1).reshape(N, n_pts)                          # (N, n_pts)

    # Differentiable OLS: fit a line through msg->output
    slopes, offsets = _torch_linear_fit(msg_grid, func)

    # Linear prediction: what f_theta WOULD output if linear in msg
    linear_pred = slopes[:, None] * msg_grid + offsets[:, None]       # (N, n_pts)

    # Residual: the non-linear component
    residual = func - linear_pred                                     # (N, n_pts)

    # Mean squared residual across all neurons and points
    loss = (residual ** 2).mean()

    return loss
