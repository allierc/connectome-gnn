"""Yang multitask trial adapter — Trial -> torch tensors + per-trial sampling.

Subset of `papers/multi-tasks/src/NeuralGraph/data_loaders/multi_task_data.py`
that touches data generation only (training-side helpers BlockTopology /
PyG packing / loss metrics are intentionally not ported here — keep this file
about turning a `Trial` into (x_in, y_tgt, c_mask) numpy/torch arrays).
"""
from __future__ import annotations

import numpy as np
import torch

from connectome_gnn.generators.cortex_task import generate_trials, get_default_hp


def trial_to_numpy(trial, b: int):
    """Extract one batch element of a Yang Trial as float32 numpy arrays.

    Returns:
        x_in:   (tdim, N_i)   input drive
        y_tgt:  (tdim, N_o)   target output
        c_mask: (tdim, N_o)   loss weight (Yang's c_mask, un-flattened from (tdim*B, N_o))
    """
    x_in  = np.asarray(trial.x[:, b, :], dtype=np.float32)
    y_tgt = np.asarray(trial.y[:, b, :], dtype=np.float32)
    cm_full = trial.c_mask.reshape(trial.tdim, trial.batch_size, -1)
    c_mask = np.asarray(cm_full[:, b, :], dtype=np.float32)
    return x_in, y_tgt, c_mask


def trial_to_tensors(trial, b: int, device):
    """Same as trial_to_numpy but returns torch tensors on `device`."""
    x_in, y_tgt, c_mask = trial_to_numpy(trial, b)
    return (
        torch.from_numpy(x_in).to(device),
        torch.from_numpy(y_tgt).to(device),
        torch.from_numpy(c_mask).to(device),
    )


def make_multi_task_batch(rules, weights, hp, batch_size, device,
                          mode: str = 'random'):
    """Per-trial random task sampling over `rules`.

    Each of the batch_size returned items has its rule drawn independently from
    `rules` (uniform if `weights` is empty, otherwise weighted). Returns
        items:        list of (x_in, y_tgt, c_mask) torch tensors, length batch_size
        rule_choices: list[str] of the rule selected for each item
    """
    rng = hp.get('rng', np.random.RandomState())
    weights = list(weights) if weights else None
    if weights and len(weights) != len(rules):
        raise ValueError(
            f'rule_weights length {len(weights)} != rules length {len(rules)}'
        )
    if weights:
        s = float(sum(weights))
        weights = [w / s for w in weights]

    items = []
    rule_choices = []
    for _ in range(batch_size):
        idx = rng.choice(len(rules), p=weights) if weights else rng.choice(len(rules))
        r = rules[idx]
        trial = generate_trials(r, hp, mode=mode, batch_size=1)
        items.append(trial_to_tensors(trial, 0, device))
        rule_choices.append(r)
    return items, rule_choices


def probe_task_shapes(rule: str, hp: dict | None = None, seed: int = 0):
    """Generate a tiny trial to discover N_i, N_o, tdim, dt.

    Returns (N_i, N_o, tdim, dt). Does NOT mutate the caller's hp.
    """
    if hp is None:
        hp = get_default_hp('all')
    hp = dict(hp)
    hp['rng'] = np.random.RandomState(seed=seed)
    t = generate_trials(rule, hp, mode='random', batch_size=2)
    N_i = t.x.shape[2]
    N_o = t.y.shape[2]
    tdim = t.x.shape[0]
    dt = float(hp.get('dt', 20.0)) / 1000.0
    return N_i, N_o, tdim, dt
