"""Cosine-annealing learning rate schedule for the W optimizer.

Hypothesis: decaying lr_W from initial to initial/10 over training prevents
regression-type failures (seeds that peak conn_R2>=0.95 then deteriorate <0.90)
which dominate the DAL=35 baseline failure mode at 57% of degenerate-basin entries.
"""

from __future__ import annotations

import math


def apply_lr_w_cosine_decay(
    optimizer_W,
    current_step: int,
    total_steps: int,
    initial_lr: float,
    min_lr_factor: float = 0.1,
) -> None:
    """Set lr_W = initial_lr * (min_lr_factor + (1-min_lr_factor) * 0.5*(1+cos(pi*step/total)))

    This implements a cosine-annealing schedule that decays lr_W from initial_lr
    down to initial_lr * min_lr_factor over the course of training. The decay is
    smooth and monotonic in the effective learning rate.

    Args:
        optimizer_W: PyTorch optimizer for the W (connectivity) matrix.
        current_step: Current training step (0-indexed).
        total_steps: Total number of training steps.
        initial_lr: The initial (maximum) learning rate.
        min_lr_factor: Minimum LR as a fraction of initial_lr (default 0.1 = lr/10).

    PASS CONDITION: 8-seed robustness shows >=7/8 convergence (>=87.5%) with
    converged-seed mean conn_R2 >= 0.95, measured at DAL=35 baseline + cosine decay.
    Convergence defined as conn_R2 >= 0.90.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if current_step < 0:
        raise ValueError(f"current_step must be >= 0, got {current_step}")

    # Clamp to [0, total_steps] for safety at boundaries
    progress = min(current_step / total_steps, 1.0)

    # Cosine annealing: starts at 1.0, ends at min_lr_factor
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = initial_lr * (min_lr_factor + (1.0 - min_lr_factor) * cosine_factor)

    for param_group in optimizer_W.param_groups:
        param_group["lr"] = lr
