"""Heading-bin ablation utilities.

Shared between the trainer (`graph_trainer.py`) and the rollout / snapshot
helpers (`bump_attractor_eval.py`) so the bin convention is defined ONCE.

Convention
----------
K bins tile the circle (-π, π] uniformly. Bin k is centred at
    θ_k = -π + (k + 0.5) * (2π / K),    k ∈ {0, …, K-1}
so the first bin straddles -π/+π and the K/2-th bin is at 0. Bin assignment
is the standard `floor((θ + π) / (2π/K))`, with θ wrapped to (-π, π] first.

Encoding the input cue (t=0 only)
---------------------------------
The on-disk dataset stores the heading cue as a 2-channel impulse at t=0:
    u[:, 0, -2] = cos θ₀,    u[:, 0, -1] = sin θ₀
In bins mode the trainer drops those two channels and writes a K-channel
one-hot bump at t=0 instead:
    u_bin[:, 0, -K + bin(θ₀)] = 1

Decoding the K-bin readout
--------------------------
The model emits K-dim logits per frame. The decoded heading is the circular
mean of the softmax over bin centres — equivalent to the population-vector
average of K basis vectors weighted by p_k. Argmax is a valid alternative
but is non-differentiable and step-quantised at 2π/K resolution; the
circular mean is smooth (matters for the smoothness of the evolution-plot
trace) and biases toward the dominant mode when the distribution is peaked.

Both numpy and torch implementations are provided so the trainer (torch)
and the rollout helpers (numpy on already-detached tensors) share the same
bin convention without any back-and-forth.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = (
    "bin_centers_np",
    "bin_centers_torch",
    "theta_to_bin_np",
    "theta_to_bin_torch",
    "cos_sin_to_bin_torch",
    "convert_cos_sin_input_to_bin_cue_torch",
    "convert_cos_sin_target_to_bin_labels_torch",
    "softmax_logits_to_decoded_theta_np",
    "softmax_logits_to_cos_sin_np",
)


# ----- bin centres ---------------------------------------------------------

def bin_centers_np(n_bins: int) -> np.ndarray:
    """Return shape (K,) array of bin-centre angles in (-π, π]."""
    K = int(n_bins)
    return ((np.arange(K, dtype=np.float64) + 0.5) * (2.0 * np.pi / K)
            - np.pi).astype(np.float32)


def bin_centers_torch(n_bins: int, *, device=None,
                      dtype=torch.float32) -> torch.Tensor:
    K = int(n_bins)
    return (torch.arange(K, device=device, dtype=dtype) + 0.5) \
        * (2.0 * np.pi / K) - np.pi


# ----- θ → bin index -------------------------------------------------------

def _wrap_pi_np(theta: np.ndarray) -> np.ndarray:
    """Wrap to (-π, π]. (θ + π) mod 2π − π gives [-π, π); shift the
    -π edge to +π so the half-open interval matches the convention used by
    the bin centres above."""
    out = (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi
    out = np.where(out == -np.pi, np.pi, out)
    return out


def theta_to_bin_np(theta, n_bins: int) -> np.ndarray:
    """θ ∈ R (any shape) → bin index ∈ [0, K) of the same shape, int64."""
    K = int(n_bins)
    th = _wrap_pi_np(np.asarray(theta))
    idx = np.floor((th + np.pi) * K / (2.0 * np.pi)).astype(np.int64)
    return np.clip(idx, 0, K - 1)


def theta_to_bin_torch(theta: torch.Tensor, n_bins: int) -> torch.Tensor:
    """θ ∈ R (any shape) → bin index ∈ [0, K), torch long."""
    K = int(n_bins)
    th = torch.atan2(torch.sin(theta), torch.cos(theta))  # wrap to (-π, π]
    idx = torch.floor((th + np.pi) * K / (2.0 * np.pi)).long()
    return idx.clamp_(0, K - 1)


def cos_sin_to_bin_torch(cos_t: torch.Tensor, sin_t: torch.Tensor,
                         n_bins: int) -> torch.Tensor:
    """(cos θ, sin θ) → bin index. Same shape as cos_t, torch long."""
    return theta_to_bin_torch(torch.atan2(sin_t, cos_t), n_bins)


# ----- on-the-fly input/target conversion (torch, batched) -----------------

def convert_cos_sin_input_to_bin_cue_torch(
    u: torch.Tensor, n_bins: int, *, cue_first: bool = False,
) -> torch.Tensor:
    """Replace the trailing (cos θ₀, sin θ₀) cue pair with a K-bin one-hot
    bump at t=0.

    u: (B, T, n_drive + 2) — last 2 channels are the cos/sin cue, which is
       non-zero ONLY at t=0 by construction (see the swim-integration data
       generator). All channels before that are passed through unchanged
       (driver channels: ω, optionally v_fwd, optionally ω_proprio).

    Returns: (B, T, n_drive + K) with one-hot only at frame 0 and zeros
       elsewhere on the K cue channels.
    """
    K = int(n_bins)
    if u.ndim != 3 or u.shape[-1] < 2:
        raise ValueError(
            f"convert_cos_sin_input_to_bin_cue_torch expects (B, T, ≥2); "
            f"got shape {tuple(u.shape)}."
        )
    B, T, n_in = u.shape
    n_drive = n_in - 2
    # θ₀ from the t=0 impulse of the cos/sin cue columns.
    cos0 = u[:, 0, n_drive]
    sin0 = u[:, 0, n_drive + 1]
    bin_idx = cos_sin_to_bin_torch(cos0, sin0, K)              # (B,) long
    u_bin = u.new_zeros((B, T, n_drive + K))
    if n_drive > 0:
        u_bin[:, :, :n_drive] = u[:, :, :n_drive]
    # one-hot scatter at t=0 only
    cue = u.new_zeros((B, K))
    cue.scatter_(1, bin_idx.unsqueeze(1), 1.0)
    u_bin[:, 0, n_drive:] = cue
    return u_bin


def convert_cos_sin_target_to_bin_labels_torch(
    y: torch.Tensor, n_bins: int,
) -> torch.Tensor:
    """(B, T, 2) cos/sin target → (B, T) long bin indices."""
    if y.ndim != 3 or y.shape[-1] != 2:
        raise ValueError(
            f"convert_cos_sin_target_to_bin_labels_torch expects (B, T, 2); "
            f"got shape {tuple(y.shape)}."
        )
    return cos_sin_to_bin_torch(y[..., 0], y[..., 1], n_bins)


# ----- decode K-bin logits → angle / cos-sin pair (numpy) ------------------

def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=axis, keepdims=True)
    return x


def softmax_logits_to_cos_sin_np(logits: np.ndarray, n_bins: int,
                                 axis: int = -1) -> np.ndarray:
    """(..., K) logits → (..., 2) circular-mean (cos θ̂, sin θ̂).

    The L2 magnitude of (cos θ̂, sin θ̂) lies in (0, 1] and tracks
    distribution sharpness: 1 means a single-bin peak, 0 means uniform.
    Downstream plot helpers only use the angle, so the magnitude leak is
    harmless — they atan2 the pair.
    """
    K = int(n_bins)
    p = _softmax_np(logits, axis=axis)
    centres = bin_centers_np(K)
    # Broadcast centres into the logits shape over `axis`.
    shape = [1] * p.ndim
    shape[axis] = K
    c = np.cos(centres).reshape(shape)
    s = np.sin(centres).reshape(shape)
    cos_dec = (p * c).sum(axis=axis)
    sin_dec = (p * s).sum(axis=axis)
    return np.stack([cos_dec, sin_dec], axis=-1).astype(np.float32)


def softmax_logits_to_decoded_theta_np(logits: np.ndarray, n_bins: int,
                                       axis: int = -1) -> np.ndarray:
    """(..., K) logits → (...) decoded angle in (-π, π]."""
    cs = softmax_logits_to_cos_sin_np(logits, n_bins, axis=axis)
    return np.arctan2(cs[..., 1], cs[..., 0]).astype(np.float32)
