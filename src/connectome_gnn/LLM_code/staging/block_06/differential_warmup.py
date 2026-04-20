"""Differential learning rate warmup scheduler.

Keeps W param groups at full LR from step 0 while linearly ramping
MLP/embedding groups from warmup_start_fraction to 1.0 over warmup_steps.
"""

import torch
from torch.optim.lr_scheduler import LambdaLR


# Param group names that should NOT be warmed up (stay at full LR)
_W_GROUP_NAMES = frozenset({"W"})


def apply_differential_warmup(
    optimizer: torch.optim.Optimizer,
    config: object,
    warmup_steps: int = 1000,
    warmup_start_fraction: float = 0.01,
) -> LambdaLR:
    """Create per-param-group LR scheduler: W groups get constant LR, MLP/embedding groups
    get linear warmup from warmup_start_fraction to 1.0 over warmup_steps.

    PASS CONDITION: Over 8+ seeds at DAL=35, convergence rate (conn_R2 >= 0.95) increases
    from baseline 86% (6/7) to >= 95% (at least 8/8 or 15/16), AND mean conn_R2 of
    converged seeds remains >= 0.965 (no degradation of ceiling).

    Args:
        optimizer: Adam optimizer with named param groups (as built by set_trainable_parameters).
            Each group must have a 'name' key (e.g. 'W', 'g_phi', 'f_theta', 'embedding').
        config: Training config object (unused currently, reserved for future per-config tuning).
        warmup_steps: Number of optimizer steps over which MLP/embedding LRs ramp up.
        warmup_start_fraction: Starting LR multiplier for warmed-up groups (e.g. 0.01 = 1%).

    Returns:
        LambdaLR scheduler with per-group lambda functions. Call scheduler.step() after each
        optimizer.step().
    """
    if warmup_steps <= 0:
        raise ValueError(f"warmup_steps must be positive, got {warmup_steps}")
    if not (0.0 < warmup_start_fraction < 1.0):
        raise ValueError(
            f"warmup_start_fraction must be in (0, 1), got {warmup_start_fraction}"
        )

    lambdas = []
    for group in optimizer.param_groups:
        group_name = group.get("name", "")
        if group_name in _W_GROUP_NAMES:
            # W groups: constant multiplier = 1.0 (full LR from step 0)
            lambdas.append(lambda step: 1.0)
        else:
            # MLP/embedding groups: linear ramp from warmup_start_fraction to 1.0
            # Using default-arg capture to avoid late-binding closure issue
            def _make_warmup_lambda(start_frac=warmup_start_fraction, steps=warmup_steps):
                def _lambda(step):
                    if step >= steps:
                        return 1.0
                    # Linear interpolation: start_frac + (1.0 - start_frac) * (step / steps)
                    return start_frac + (1.0 - start_frac) * (step / steps)
                return _lambda
            lambdas.append(_make_warmup_lambda())

    return LambdaLR(optimizer, lr_lambda=lambdas)
