"""Temporal moving-average denoiser for voltage traces.

Block 01 — denoising theme.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_voltage_denoise(v_noisy: torch.Tensor, window: int = 3) -> torch.Tensor:
    """Temporal moving-average denoiser for voltage traces.

    Applies a symmetric moving-average filter of width `window` along the
    time axis (dim 0) of a (T, N) voltage tensor, using reflect padding
    at boundaries.

    PASS CONDITION: On flyvis data (T=64000, N=13741) with γ=0.10
    and window=3:
      (1) MSE(denoised, clean) / MSE(noisy, clean) ≤ 0.70
      (2) mean per-neuron Pearson-r(denoised, clean) > Pearson-r(noisy, clean)
      (3) ≥ 75% of neurons have lower MSE after denoising

    Parameters
    ----------
    v_noisy : torch.Tensor
        (T, N) voltage tensor with measurement noise.
    window : int
        Width of the symmetric moving-average kernel (must be odd, ≥ 3).

    Returns
    -------
    torch.Tensor
        (T, N) denoised voltage tensor, same dtype and device.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and ≥ 1, got {window}")
    if window == 1:
        return v_noisy.clone()

    T, N = v_noisy.shape
    pad = window // 2

    # Reshape to (N, 1, T) for conv1d: batch=N, channels=1, length=T
    x = v_noisy.T.unsqueeze(1)  # (N, 1, T)

    # Reflect-pad along time axis
    x_padded = F.pad(x, (pad, pad), mode="reflect")  # (N, 1, T + 2*pad)

    # Uniform averaging kernel
    kernel = torch.ones(1, 1, window, device=v_noisy.device, dtype=v_noisy.dtype) / window

    # Apply depthwise conv1d (groups=1 since channel=1, applied per-batch-element)
    out = F.conv1d(x_padded, kernel)  # (N, 1, T)

    return out.squeeze(1).T  # (T, N)
