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


# ----------------------------------------------------------------------------- the rollout loss
from connectome_gnn.models.calcium_observation import calcium_rollout_loss  # noqa: E402
from connectome_gnn.neuron_state import NeuronState  # noqa: E402


class _Recording:
    """x_ts stand-in: a calcium recording and a stimulus, frame(k) -> NeuronState."""

    def __init__(self, n_frames=80, n=5, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.voltage = torch.rand(n_frames, n, generator=g)
        self.stimulus = torch.randn(n_frames, n, generator=g)
        self.n_frames = n_frames

    def frame(self, k):
        return NeuronState(index=torch.arange(self.voltage.shape[1]),
                           voltage=self.voltage[k].clone(), stimulus=self.stimulus[k].clone())


class _Net(nn.Module):
    """dv/dt = -v + W tanh(v + stimulus), on the batched state, as the GNN returns it.
    The stimulus enters NONLINEARLY, as in the GNN's update MLP: added outside the
    nonlinearity it would leave dL/dW unchanged by a stale stimulus, and the
    checkpoint test below could not see the bug it is there for."""

    def __init__(self, n=5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n, n, generator=torch.Generator().manual_seed(1)) * 0.3)
        self.n = n

    def forward(self, state, edges, data_id=None, return_all=False):
        v = state.voltage.view(-1, self.n)
        s = state.stimulus.view(-1, self.n)
        pred = (-v + torch.tanh(v + s) @ self.W.T).reshape(-1, 1)
        return pred, pred, None


class _NoRegul:
    def reset_iteration(self, device=None): pass
    def sample_g_phi_perm(self, device): return None
    def compute(self, **kw): return torch.zeros(())
    def compute_update_regul(self, *a): return torch.zeros(())


def _loss_and_grad(stride, ckpt, K=12):
    rec = _Recording()
    model = _Net()
    model.calcium = CalciumIndicator(DT, 0.075, 0.4, n_taps=3, stride=stride)
    tc = types.SimpleNamespace(batch_size=3, noise_recurrent_level=0.0, fit_reduction="norm2",
                               integration_method="euler", calcium_checkpoint_steps=ckpt)
    sim = types.SimpleNamespace(delta_t=DT)
    frames = torch.tensor([20, 35, 50])
    loss, _ = calcium_rollout_loss(model, rec, torch.zeros(2, 0, dtype=torch.long),
                                   torch.arange(5), frames, 0, K, sim, tc, "cpu", None, _NoRegul(),
                                   False)
    loss.backward()
    return loss.detach(), model.W.grad.clone(), model.calcium.taps.grad.clone()


def test_checkpointed_rollout_has_the_stored_gradient():
    """Recomputing each step in the backward pass gives the stored path's loss
    and gradients, dense and with a stride -- the check that catches a
    recompute reading the state objects after they moved on (Plexus exp17's
    checkpoint bug), since every step overwrites them."""
    for stride in (1, 3):
        a, b = _loss_and_grad(stride, False), _loss_and_grad(stride, True)
        assert torch.equal(a[0], b[0]), stride
        for ga, gb in zip(a[1:], b[1:]):
            assert torch.allclose(ga, gb, rtol=1e-6, atol=1e-9), (stride, (ga - gb).abs().max())
        assert a[1].abs().max() > 0


def test_stride_scores_only_the_observed_frames():
    """With stride m the loss reads only c(k+m), c(k+2m), ...: changing an
    unobserved frame of the recording changes nothing."""
    base = _loss_and_grad(3, False)[0]
    rec_frames = _Recording()
    for unobserved in (21, 22, 24):                    # k = 20: observed are 23, 26, ... and taps 20, 17, 14
        torch.manual_seed(0)
        model = _Net(); model.calcium = CalciumIndicator(DT, 0.075, 0.4, n_taps=3, stride=3)
        rec = _Recording(); rec.voltage[unobserved] += 10.0
        tc = types.SimpleNamespace(batch_size=3, noise_recurrent_level=0.0, fit_reduction="norm2",
                                   integration_method="euler", calcium_checkpoint_steps=False)
        loss, _ = calcium_rollout_loss(model, rec, torch.zeros(2, 0, dtype=torch.long), torch.arange(5),
                                       torch.tensor([20, 35, 50]), 0, 12, types.SimpleNamespace(delta_t=DT),
                                       tc, "cpu", None, _NoRegul(), False)
        assert torch.equal(loss.detach(), base), unobserved
    assert calcium_history(rec_frames.voltage, 20, 3, 3).shape == (3, 5)
    assert torch.equal(calcium_history(rec_frames.voltage, 20, 3, 3), rec_frames.voltage[[20, 17, 14]])
