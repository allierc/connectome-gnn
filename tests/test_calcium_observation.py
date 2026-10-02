"""The latent-calcium indicator (models/calcium_observation.py)."""
import types

import numpy as np
import torch
import torch.nn as nn

from connectome_gnn.models.calcium_observation import (
    CalciumIndicator, attach_calcium_indicator, calcium_history)
from connectome_gnn.models.gcamp import create_gcamp

DT = 0.02


def test_cascade_is_the_gcamp6f_kernel():
    """Two unit-gain stages reproduce the convolution the datasets were built
    with to well under the 1% of SD(c) that 40 dB of noise adds."""
    rng = np.random.default_rng(0)
    v = np.cumsum(rng.normal(size=(4000, 8)), axis=0) * 0.05
    k = create_gcamp("double_exp", tau_rise=0.075, tau_decay=0.4, length_s=2.4).kernel(DT).numpy()
    conv = np.stack([np.convolve(v[:, i], k)[:len(v)] for i in range(v.shape[1])], 1)
    ind = CalciumIndicator(DT, 0.075, 0.4)
    vt = torch.as_tensor(v, dtype=torch.float64)
    r = vt[0].clone(); c = torch.as_tensor(conv[0])
    out = [c]
    with torch.no_grad():
        for t in range(len(v) - 1):
            r, c = ind.step(vt[t], r, c)
            out.append(c)
    out = torch.stack(out).numpy()
    rel = np.sqrt(((out[500:] - conv[500:]) ** 2).mean()) / conv[500:].std()
    assert rel < 0.005, rel


def test_taus_round_trip_and_kernel_given_is_frozen():
    ind = CalciumIndicator(DT, 0.075, 0.4, learn_kernel=False)
    assert torch.allclose(ind.taus(), torch.tensor([0.075, 0.4]), rtol=1e-5)
    assert [n for n, _ in ind.named_parameters()] == ["taps"]
    learned = CalciumIndicator(DT, 0.075, 0.4, learn_kernel=True)
    assert sorted(n for n, _ in learned.named_parameters()) == ["rate", "taps"]


def test_taps_start_at_the_newest_frame():
    ind = CalciumIndicator(DT, 0.075, 0.4, n_taps=6)
    c = torch.arange(20.0).unsqueeze(1).repeat(1, 3)        # c(t) = t
    hist = calcium_history(c, 10, 6)
    assert hist[:, 0].tolist() == [10, 9, 8, 7, 6, 5]
    v, r = ind.lift(hist)
    assert torch.equal(v, c[10]) and torch.equal(r, c[10])
    assert calcium_history(c, 1, 6)[:, 0].tolist() == [1, 0, 0, 0, 0, 0]


def test_voltage_observable_adds_nothing():
    """The default leaves the model's parameters exactly as they were."""
    model = nn.Linear(3, 2)
    before = list(model.state_dict())
    cfg = types.SimpleNamespace(training=types.SimpleNamespace(observable="voltage"),
                                simulation=types.SimpleNamespace(delta_t=DT))
    attach_calcium_indicator(model, cfg)
    assert list(model.state_dict()) == before and not hasattr(model, "calcium")
    cfg.training = types.SimpleNamespace(
        observable="calcium", calcium_tau_rise=0.075, calcium_tau_decay=0.4,
        calcium_n_taps=6, calcium_kernel_learned=False)
    attach_calcium_indicator(model, cfg)
    assert "calcium.taps" in model.state_dict()
