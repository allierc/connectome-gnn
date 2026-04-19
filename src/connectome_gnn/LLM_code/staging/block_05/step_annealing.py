"""Step-based regularization annealing for single-epoch training.

Block 05 — robustness theme.

The epoch-based annealing formula `coeff * (1 - exp(-rate * epoch))` is
exactly 0.0 for ALL iterations when n_epochs=1, because `epoch` stays at 0
throughout.  This module provides a drop-in replacement that uses fractional
training progress so that weight regularizers ramp up smoothly even within
the first (and only) epoch.
"""

import math


def step_annealing(
    epoch: int,
    iter_in_epoch: int,
    Niter: int,
    n_epochs: int,
    rate: float,
    coeff: float,
) -> float:
    """Compute annealed coefficient using fractional training progress.

    Replaces the epoch-based formula `coeff * (1 - exp(-rate * epoch))`
    (which is ZERO at epoch 0) with
    `coeff * (1 - exp(-rate * effective_epoch))` where
    effective_epoch = (epoch * Niter + iter_in_epoch) / (n_epochs * Niter) * n_epochs.

    PASS CONDITION (n_epochs=1, rate=0.5, Niter=42666, coeff=0.00015):
    (1) Epoch-based annealing produces 0.0 for ALL 42666 iterations.
    (2) Step-based annealing produces >0 for >99% of iterations (all except iter=0).
    (3) Step-based final value = coeff * (1-exp(-rate)) = 0.393*coeff +/- 1e-6.
    (4) Schedule is monotonically non-decreasing.
    (5) At integer epoch boundaries (n_epochs=2), step-based matches epoch-based.

    Parameters
    ----------
    epoch : int
        Current epoch index (0-based).
    iter_in_epoch : int
        Current iteration within the epoch (0-based).
    Niter : int
        Total iterations per epoch.
    n_epochs : int
        Total number of epochs in this training run.
    rate : float
        Annealing rate (same semantics as regul_annealing_rate).
        If rate <= 0, returns coeff unchanged (no annealing).
    coeff : float
        The nominal coefficient value (e.g. coeff_W_L1).

    Returns
    -------
    float
        The annealed coefficient for this training step.
    """
    if rate <= 0:
        return float(coeff)

    total_iters = n_epochs * Niter
    if total_iters <= 0:
        return 0.0

    global_iter = epoch * Niter + iter_in_epoch
    # effective_epoch maps [0, total_iters] -> [0, n_epochs]
    effective_epoch = (global_iter / total_iters) * n_epochs
    return float(coeff * (1.0 - math.exp(-rate * effective_epoch)))
