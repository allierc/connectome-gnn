"""f_theta message-linearity regularization loss.

Penalizes f_theta nonlinearity in the message (synaptic input) dimension.
The true ODE is dv/dt = -(v - V_rest)/tau + msg/tau, so dv/dt is LINEAR
in msg. An unconstrained MLP can compress or saturate the msg response,
absorbing W magnitude into f_theta and causing systematic slope bias in
recovered W. This loss forces proportional msg response, breaking the
W-vs-f_theta scale degeneracy.
"""

import numpy as np
import torch


def _vectorized_linspace(starts, ends, n_pts, device):
    """Create (N, n_pts) tensor where row n spans [starts[n], ends[n]]."""
    t = torch.linspace(0, 1, n_pts, device=device)
    starts_t = torch.as_tensor(starts, dtype=torch.float32, device=device)
    ends_t = torch.as_tensor(ends, dtype=torch.float32, device=device)
    return starts_t[:, None] + t[None, :] * (ends_t - starts_t)[:, None]


def _torch_linear_fit(x, y):
    """Differentiable OLS fit: y ≈ slope * x + offset. Grad flows through y."""
    n_pts = x.shape[1]
    sx = x.sum(dim=1)
    sy = y.sum(dim=1)
    sxy = (x * y).sum(dim=1)
    sxx = (x * x).sum(dim=1)
    denom = n_pts * sxx - sx * sx
    slopes = (n_pts * sxy - sx * sy) / (denom + 1e-12)
    offsets = (sy - slopes * sx) / n_pts
    return slopes, offsets


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
    data-driven range, fits a differentiable OLS line through the msg->output
    curve, and penalizes the squared residual (non-linear component).

    Physical motivation: the true ODE is dv/dt = -(v - V_rest)/tau + msg/tau,
    so dv/dt is LINEAR in msg. An unconstrained MLP can compress or saturate
    the msg response, absorbing W magnitude into f_theta and causing systematic
    slope bias in recovered W. This loss forces proportional msg response,
    breaking the W-vs-f_theta scale degeneracy.

    PASS CONDITION:
      (1) A randomly initialised f_theta MLP exhibits measurable msg
          nonlinearity: residual_MSE / total_output_variance > 0.01,
          confirming the loss targets a real degree of freedom.
      (2) Gradient norm w.r.t. f_theta parameters > 0 (differentiable).
      (3) The loss is scale-invariant: using per-neuron sigma-scaled msg
          range produces finite, non-degenerate values for all neurons.

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

    # --- msg sweep range: [-2*sigma, +2*sigma] per neuron ---
    # msg = sum(W_ij * v_j); at the neuron's operating point the msg
    # magnitude scales with sigma_v. Using +-2*sigma covers the typical
    # operating range without extrapolation artifacts.
    sigma_safe = np.maximum(sigma[:N], 1e-6)  # avoid zero-width range
    msg_starts = -2.0 * sigma_safe
    msg_ends = 2.0 * sigma_safe
    msg_grid = _vectorized_linspace(msg_starts, msg_ends, n_pts, device)  # (N, n_pts)

    # --- build f_theta inputs: [v=mu, embedding, msg=sweep, exc=0] ---
    mu_t = torch.as_tensor(mu[:N], dtype=torch.float32, device=device)  # (N,)
    a_detached = model.a[:N].detach()  # (N, emb_dim) -- block grad to embeddings

    # Expand v and embedding to match (N, n_pts)
    v_flat = mu_t[:, None].expand(-1, n_pts).reshape(-1, 1)             # (N*n_pts, 1)
    emb_flat = (a_detached[:, None, :]
                .expand(-1, n_pts, -1)
                .reshape(-1, emb_dim))                                   # (N*n_pts, emb_dim)
    msg_flat = msg_grid.reshape(-1, 1)                                   # (N*n_pts, 1)
    exc_flat = torch.zeros_like(msg_flat)                                # (N*n_pts, 1)

    in_features = torch.cat([v_flat, emb_flat, msg_flat, exc_flat], dim=1)  # (N*n_pts, D)

    # --- evaluate f_theta WITH gradient tracking ---
    out = model.f_theta(in_features.float())  # (N*n_pts, 1)
    func = out.squeeze(-1).reshape(N, n_pts)  # (N, n_pts)

    # --- differentiable OLS fit: output ≈ slope * msg + offset ---
    slopes, offsets = _torch_linear_fit(msg_grid, func)

    # --- penalize non-linear component ---
    linear_pred = slopes[:, None] * msg_grid + offsets[:, None]  # (N, n_pts)
    residual = func - linear_pred                                 # (N, n_pts)
    loss = (residual ** 2).mean()

    return loss
