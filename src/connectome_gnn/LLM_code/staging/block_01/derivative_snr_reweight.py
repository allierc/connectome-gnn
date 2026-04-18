"""Wiener-style SNR reweighting for derivative targets under measurement noise.

Block 01 — denoising theme.

The derivative target y[k] = (v[k+1] - v[k]) / dt inherits doubled
measurement noise: var(y_noise) = 2 * gamma^2 / dt^2.  When |y[k]| is
small relative to the noise floor, the training gradient is dominated by
noise.  A Wiener-style weight w = signal_est / (signal_est + noise_var)
suppresses those noise-dominated samples, letting the model focus on
timesteps with genuine dynamics.
"""

from __future__ import annotations

import torch


def derivative_snr_weights(
    y: torch.Tensor,
    gamma: float,
    *,
    dt: float = 1.0,
    floor: float = 0.1,
) -> torch.Tensor:
    """Compute per-sample Wiener-style SNR weights for derivative targets.

    PASS CONDITION: On flyvis data (T=64000, N=13741) with gamma=0.10:
      (1) Signal concentration: weighted mean(dy_clean^2) / unweighted mean(dy_clean^2) >= 1.5
          (weights select higher-signal samples)
      (2) Effective SNR gain: weighted_mean_signal / noise_var > unweighted by >= 50%
      (3) Effective sample fraction (Kish) >= 0.3 (not too aggressive)

    Parameters
    ----------
    y : torch.Tensor
        Derivative targets (any shape).  Typically (T-1, N) or (batch*N, 1).
    gamma : float
        Measurement noise standard deviation.
    dt : float
        Timestep size (default 1.0, matching the pipeline's convention).
    floor : float
        Minimum weight to avoid fully zeroing any sample (default 0.1).

    Returns
    -------
    torch.Tensor
        Same shape as *y*, values in [floor, 1.0].
    """
    noise_var = 2.0 * (gamma / dt) ** 2
    y_sq = y.detach() ** 2
    # Bias-corrected signal power estimate (clamp to non-negative)
    signal_est = (y_sq - noise_var).clamp(min=0.0)
    # Wiener weight: 1 when signal >> noise, 0 when signal << noise
    w = signal_est / (signal_est + noise_var + 1e-12)
    # Apply floor
    w = w.clamp(min=floor)
    return w
