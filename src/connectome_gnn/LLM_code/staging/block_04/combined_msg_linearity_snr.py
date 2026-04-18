"""Combined regularization for block 04 — best-of-combination.

Combines two complementary mechanisms from reverted blocks 01 and 03.
Individually each was insufficient to pass the verdict, but they target
orthogonal aspects of the inverse problem:

1. **f_theta msg linearity loss** (block 03 theme: identifiability) —
   penalizes f_theta nonlinearity in the synaptic input (msg) dimension.
   The true ODE is dv/dt = -(v - V_rest)/tau + msg/tau, so dv/dt is
   LINEAR in msg. An unconstrained MLP can compress or saturate the msg
   response, absorbing W magnitude into f_theta and causing systematic
   slope bias (~0.51 in the winner baseline). This loss forces
   proportional msg response, breaking the W-vs-f_theta scale degeneracy.

2. **derivative SNR weights** (block 01 theme: denoising) — computes
   per-timestep Wiener-style weights that down-weight derivative targets
   whose magnitude is below the measurement-noise floor. This focuses
   the gradient on timesteps where the true dynamics dominate noise,
   reducing noise-driven weight updates that bias W toward zero.

Wire-up plan:
- msg linearity: new COMPONENT in regularizer.py (one entry).
- SNR weights: one-liner modification at prediction-loss site in
  graph_trainer.py (not a COMPONENT, just a reweighting of existing loss).
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
#  Mechanism 1: f_theta msg linearity loss
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
    the msg->output curve, and penalizes the squared residual (non-linear
    component).

    Physical motivation: the true ODE is dv/dt = -(v - V_rest)/tau + msg/tau,
    so dv/dt is LINEAR in msg. An unconstrained MLP can compress or saturate
    the msg response, absorbing W magnitude into f_theta and causing
    systematic slope bias in recovered W. This loss forces proportional msg
    response, breaking the W-vs-f_theta scale degeneracy.

    PASS CONDITION (combined with derivative_snr_weights):
      (1) A randomly initialised f_theta MLP exhibits measurable msg
          nonlinearity: residual_MSE / total_output_variance > 0.01,
          confirming the loss targets a real degree of freedom.
      (2) Gradient norm w.r.t. f_theta parameters > 0 (differentiable).
      (3) Derivative SNR weights discriminate: top-quartile derivative
          timesteps have higher mean weight than bottom-quartile.
      (4) All outputs are finite (no NaN/Inf).

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
    emb_flat = a_detached[:, None, :].expand(-1, n_pts, -1).reshape(-1, emb_dim)  # (N*n_pts, emb_dim)
    msg_flat = msg_grid.reshape(-1, 1)                                # (N*n_pts, 1)
    exc_flat = torch.zeros_like(msg_flat)                             # (N*n_pts, 1)

    in_features = torch.cat([v_flat, emb_flat, msg_flat, exc_flat], dim=1)  # (N*n_pts, D)

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


# ------------------------------------------------------------------ #
#  Mechanism 2: derivative SNR weights
# ------------------------------------------------------------------ #

def derivative_snr_weights(
    derivatives: torch.Tensor,
    gamma: float,
    dt: float = 1.0,
    min_weight: float = 0.1,
) -> torch.Tensor:
    """Wiener-style per-timestep weights that down-weight noisy derivatives.

    Under measurement noise gamma, the discrete derivative dv/dt has noise
    variance 2*gamma^2/dt^2 (from differencing two independent noise samples).
    The Wiener weight assigns each timestep a weight proportional to its
    signal-to-noise ratio:

        w = |deriv|^2 / (|deriv|^2 + noise_var)

    Timesteps where |deriv| >> noise floor get w ~ 1 (trust the signal).
    Timesteps where |deriv| ~ noise floor get w ~ 0.5 (ambiguous).
    Timesteps where |deriv| << noise floor get w ~ min_weight (discard).

    Args:
        derivatives: (T-1, N) or (T-1,) tensor of discrete derivatives.
        gamma: Measurement noise standard deviation.
        dt: Time step size.
        min_weight: Floor on weights to prevent complete zeroing.

    Returns:
        Weights tensor, same shape as derivatives, in [min_weight, 1].
    """
    noise_var = 2.0 * gamma ** 2 / (dt ** 2)
    deriv_sq = derivatives ** 2
    w = deriv_sq / (deriv_sq + noise_var)
    return w.clamp(min=min_weight)
