"""msg_i of a GNN is read through f_theta and is gauge-free.

A GNN can scale its aggregate message by any per-neuron gamma_i that f_theta
undoes; the raw aggregate then scores gamma_i, which never reaches the
trajectory. _msg_through_f_theta returns tau_i * [f(v, msg) - f(v, 0)] with
tau_i from the model's own leak slope, which is the physical message whatever
gamma_i is.
"""
import numpy as np
import torch
import pytest

from connectome_gnn.metrics import _msg_through_f_theta, _msg_i_through_f_theta, msg_i_estimator


class _Leak(torch.nn.Module):
    """f_theta(v, a, msg', exc) = (-(v - V_rest) + msg' / gamma + exc) / tau, per neuron."""
    def __init__(self, tau, vrest, gamma):
        super().__init__()
        self.tau, self.vrest, self.gamma = tau, vrest, gamma

    def forward(self, x):
        v, msg, exc = x[:, 0], x[:, 2], x[:, 3]
        return ((-(v - self.vrest) + msg / self.gamma + exc) / self.tau).unsqueeze(1)


class _FakeGNN(torch.nn.Module):
    def __init__(self, n, gamma):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.a = torch.rand(n, 1, generator=g)
        self.tau = torch.rand(n, generator=g) * 0.3 + 0.05
        self.vrest = torch.rand(n, generator=g)
        self.f_theta = _Leak(self.tau, self.vrest, gamma)

    def _run_mlp(self, mlp, x):
        return mlp(x)


def test_gauge_free_and_exact_on_a_linear_leak():
    n = 500
    gamma = torch.rand(n) * 4 + 0.25          # per-neuron gain the model hides in g_phi
    m = _FakeGNN(n, gamma)
    v = torch.randn(n); msg = torch.randn(n) * 2; exc = torch.randn(n) * 0.1
    msg_prime = gamma * msg                    # what return_all would hand back
    x = torch.stack([v, m.a[:, 0], msg_prime, exc], dim=1)
    pred = m.f_theta(x)
    out = _msg_through_f_theta(m, pred, x, n)
    assert torch.allclose(out, msg, rtol=1e-3, atol=1e-3)         # gamma is gone (float32)
    raw_r2 = 1 - ((msg_prime - msg) ** 2).sum() / ((msg - msg.mean()) ** 2).sum()
    assert raw_r2 < 0.5                                             # the raw aggregate would not score


def test_non_leak_neurons_read_nan():
    n = 10
    m = _FakeGNN(n, torch.ones(n))
    m.f_theta.tau = -m.f_theta.tau                                  # positive dv/dv: not a leak
    x = torch.zeros(n, 4); x[:, 2] = 1.0
    out = _msg_through_f_theta(m, m.f_theta(x), x, n)
    assert torch.isnan(out).all()


def test_estimator_tag():
    assert msg_i_estimator(_FakeGNN(3, torch.ones(3))) == "through_f_theta"
    class KnownODE: pass
    assert msg_i_estimator(KnownODE()) == "forward"
