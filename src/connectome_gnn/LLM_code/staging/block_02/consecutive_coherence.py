"""Consecutive-frame temporal coherence loss for noise-robust training.

Block 02 — recurrent_schemes theme.

When training with consecutive_batch=True, the model produces predictions
for B adjacent time frames. Under measurement noise γ=0.10, each frame's
noisy voltage input causes independent prediction jitter. The underlying
dynamics are smooth, so predictions at adjacent frames should change
gradually. This loss penalises sharp frame-to-frame prediction changes,
selectively damping noise-induced variability without requiring multi-step
backpropagation through time.

Wire-up: add as a COMPONENT in regularizer.py or as a one-liner loss term
in graph_trainer.py after the batched forward pass when consecutive_batch
is active.
"""

from __future__ import annotations

import torch


def consecutive_coherence_loss(
    predictions: torch.Tensor,
    batch_size: int,
    n_neurons: int,
) -> torch.Tensor:
    """Temporal coherence loss for predictions on consecutive frames.

    Reshapes the flat batched prediction tensor into (B, N) and penalises
    the squared difference between predictions at adjacent frames.

    PASS CONDITION (Phase S test on cached flyvis data, γ=0.10):
      (1) Noisy stride-1 derivatives (proxy for predictions) have ≥ 3×
          higher mean temporal variation than clean derivatives, confirming
          the loss targets noise-induced jitter.
      (2) Absolute coherence value on noisy data, normalised per neuron, is
          in [1e-3, 1e3] — ensuring meaningful gradient at typical
          coefficient scales (0.001–1.0).
      (3) Function is differentiable: gradient norm w.r.t. predictions > 0.

    Parameters
    ----------
    predictions : torch.Tensor
        (B*N, 1) batched model predictions from B consecutive frames, each
        with N neurons. Entries [0:N] correspond to frame k, [N:2N] to
        frame k+1, etc.
    batch_size : int
        Number of consecutive frames B.
    n_neurons : int
        Number of neurons N per frame.

    Returns
    -------
    torch.Tensor
        Scalar loss (mean squared temporal difference across adjacent
        frames), normalised by (B-1)*N so the magnitude is per-neuron
        per-adjacent-pair.
    """
    if batch_size < 2:
        return torch.zeros((), device=predictions.device, dtype=predictions.dtype)

    # Reshape: (B*N, 1) → (B, N)
    pred_2d = predictions.squeeze(-1).view(batch_size, n_neurons)

    # Temporal differences: (B-1, N)
    diffs = pred_2d[1:] - pred_2d[:-1]

    # MSE normalised per neuron per pair
    return (diffs ** 2).sum() / ((batch_size - 1) * n_neurons)
