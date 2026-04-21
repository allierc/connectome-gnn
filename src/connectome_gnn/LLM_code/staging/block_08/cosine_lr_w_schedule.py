"""Cosine annealing LR schedule applied ONLY to the W parameter group.

Decays lr_W from its initial value to 0 over total training steps following a
cosine curve, while keeping all other parameter groups (g_phi, f_theta,
embedding) at their constant initial LR.

Integration point: replaces the lr_scheduler returned by build_lr_scheduler()
in graph_trainer.py, same pattern as block_06/differential_warmup.py.
"""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


# Param group names that get cosine decay
_W_GROUP_NAMES = frozenset({"W"})


def apply_cosine_lr_w_schedule(
    optimizer: torch.optim.Optimizer,
    config: object,
    total_steps: int | None = None,
) -> LambdaLR:
    """Create per-param-group LR scheduler: W groups get cosine decay to 0,
    all other groups keep constant LR.

    PASS CONDITION: convergence rate (conn_R2 >= 0.90) >= 85% across >= 8 seeds
    at DAL=35, AND converged mean conn_R2 >= 0.95 (no ceiling regression).

    Args:
        optimizer: Adam optimizer with named param groups (as built by
            set_trainable_parameters). Each group must have a 'name' key.
        config: Training config object. Used to compute total_steps if not
            provided: Niter = n_frames * DAL // batch_size * 0.2.
        total_steps: Override for total optimizer steps per epoch. If None,
            computed from config.simulation.n_frames, config.training
            .data_augmentation_loop, and config.training.batch_size.

    Returns:
        LambdaLR scheduler with per-group lambda functions. Call
        scheduler.step() after each optimizer.step().
    """
    if total_steps is None:
        tc = config.training
        sim = config.simulation
        total_steps = int(
            sim.n_frames * tc.data_augmentation_loop // tc.batch_size * 0.2
        )

    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")

    lambdas = []
    for group in optimizer.param_groups:
        group_name = group.get("name", "")
        if group_name in _W_GROUP_NAMES:
            # W group: cosine decay from 1.0 to 0.0 over total_steps
            def _make_cosine_lambda(T=total_steps):
                def _lambda(step):
                    if step >= T:
                        return 0.0
                    return 0.5 * (1.0 + math.cos(math.pi * step / T))
                return _lambda
            lambdas.append(_make_cosine_lambda())
        else:
            # All other groups: constant multiplier = 1.0
            lambdas.append(lambda step: 1.0)

    return LambdaLR(optimizer, lr_lambda=lambdas)
