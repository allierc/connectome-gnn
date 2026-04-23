"""Temporal Gaussian smoothing of voltage traces for GNN input denoising.

Applies a 1D Gaussian filter along the time axis to reduce ODE integration
noise (noise_model_level * randn per step) while preserving neural dynamics
whose timescales are much longer than the smoothing window.
"""

import math
import torch
import torch.nn.functional as F


def temporal_voltage_denoise(
    voltage: torch.Tensor,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Apply 1D Gaussian temporal smoothing (sigma=3) to voltage traces (n_frames, n_neurons).

    PASS CONDITION: With coeff_voltage_denoise_alpha=0.3, 4-seed mean
    hidden_rollout_pearson >= Iter0_mean + 0.03 (absolute improvement over
    alpha=0 baseline).

    Args:
        voltage: Tensor of shape (T, N) — time series of neuron voltages.
        sigma: Standard deviation of the Gaussian kernel in frames.
            Default 3.0 gives an effective window of ~7 frames (3*sigma
            truncation on each side), reducing iid noise by ~sqrt(7) ≈ 2.6x.

    Returns:
        Smoothed voltage tensor of the same shape (T, N).
    """
    if voltage.ndim != 2:
        raise ValueError(f"Expected (T, N) tensor, got shape {voltage.shape}")
    if sigma <= 0:
        return voltage.clone()

    T, N = voltage.shape

    # Build 1D Gaussian kernel, truncated at 3*sigma
    radius = int(math.ceil(3 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=voltage.dtype, device=voltage.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()  # normalize

    # Conv1d expects (batch, channels, length) — treat neurons as batch
    # voltage: (T, N) -> (N, 1, T)
    v = voltage.t().unsqueeze(1)  # (N, 1, T)

    # Reflect-pad to preserve boundary values (avoids edge decay)
    v_padded = F.pad(v, (radius, radius), mode="reflect")

    # Reshape kernel for conv1d: (out_channels=1, in_channels=1, kernel_size)
    w = kernel.reshape(1, 1, -1)
    smoothed = F.conv1d(v_padded, w)  # (N, 1, T)

    return smoothed.squeeze(1).t()  # (T, N)
