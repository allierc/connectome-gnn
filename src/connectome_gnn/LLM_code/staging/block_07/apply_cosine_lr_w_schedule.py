"""Cosine-annealing lr_W schedule for FlyVis GNN training.

Attaches a CosineAnnealingLR scheduler to the W optimizer so that lr_W
decays from its initial value to zero over `total_steps` optimizer steps.
This prevents late-training conn_R2 regression caused by oversized W updates
after the network has nearly converged.
"""

from __future__ import annotations

import torch.optim.lr_scheduler as lr_scheduler


def apply_cosine_lr_w_schedule(optimizer_W, total_steps: int) -> lr_scheduler.CosineAnnealingLR:
    """Attach a CosineAnnealingLR scheduler to the W optimizer.

    Parameters
    ----------
    optimizer_W : torch.optim.Optimizer
        The optimizer controlling the W (synaptic weight) matrix.
    total_steps : int
        Total number of optimizer steps (T_max for the cosine schedule).
        Typically n_epochs * data_augmentation_loop * steps_per_loop.

    Returns
    -------
    scheduler : CosineAnnealingLR
        The attached scheduler. Caller must call scheduler.step() after
        each optimizer_W.step().

    PASS CONDITION: On >=8 DAL=35 seeds, convergence rate
    (conn_R2 >= 0.95) >= 87.5% AND mean converged conn_R2 >= 0.96
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer_W,
        T_max=total_steps,
        eta_min=0.0,
    )
    return scheduler
