"""The message-gain anchor: df/dmsg must equal -df/dv, whatever tau is.

A GNN's message has no absolute scale -- msg/c with a gain of c through f_theta
is the same trajectory -- and that freedom is what let a sigma=0 run shrink its
whole weight vector to 1e-8 while f_theta amplified the message back. The
generator has no such freedom: msg and -v_i sit in one bracket over one tau, so
one volt of message moves dv/dt exactly as far as one volt of depolarisation
moves it back. This term is that identity, and it must be blind to tau.
"""
import torch

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.regularizer import LossRegularizer


EMB = 2
N = 16


class _FTheta(torch.nn.Module):
    """dv/dt = T * (V - v) + gain * T * msg, an affine update with a knob.

    `gain` is the model's message gain relative to the leak: 1.0 is the
    generator's, anything else is the gauge this term exists to close.
    """

    def __init__(self, T=11.1086, V=2.9096, gain=1.0):
        super().__init__()
        self.T, self.V = T, V
        self.gain = torch.nn.Parameter(torch.tensor(float(gain)))

    def forward(self, x):
        v, msg = x[:, 0:1], x[:, 1 + EMB:2 + EMB]
        return self.T * (self.V - v) + self.gain * self.T * msg


class _Model:
    def __init__(self, **kw):
        self.f_theta = _FTheta(**kw)


def _cfg(coeff):
    """A real config, because LossRegularizer reads annealing and a dozen other
    fields off it; only the one coefficient under test is non-zero."""
    return NeuralGraphConfig(
        dataset="test_dataset",
        simulation={"params": [[1.0, 1.0]], "n_neurons": N, "n_input_neurons": 2,
                    "n_neuron_types": 2, "n_edges": 8, "n_frames": 100,
                    "delta_t": 0.02, "seed": 42},
        graph_model={"signal_model_name": "flyvis_conductance", "aggr_type": "add",
                     "embedding_dim": EMB, "input_size": 6, "output_size": 1,
                     "hidden_dim": 8, "n_layers": 2, "input_size_update": 5,
                     "hidden_dim_update": 8, "n_layers_update": 2,
                     "output_size_update": 1, "g_phi_positive": False,
                     "w_squared": True},
        training={"n_epochs": 1, "batch_size": 1, "coeff_f_theta_msg_gain": coeff},
        plotting={"colormap": "tab20", "arrow_length": 1,
                  "xlim": [-1, 1], "ylim": [-1, 1]})


def _reg(coeff):
    cfg = _cfg(coeff)
    reg = LossRegularizer(cfg.training, cfg.graph_model, activity_column=3,
                          plot_frequency=1, n_neurons=N, trainer_type="flyvis")
    reg.reset_iteration(device=torch.device("cpu"))
    return reg


def _features(seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(N, 1, generator=g)
    a = torch.zeros(N, EMB)
    msg = torch.randn(N, 1, generator=g)
    stim = torch.zeros(N, 1)
    return torch.cat([v, a, msg, stim], dim=1)


def _term(model, coeff=1.0, seed=0):
    reg = _reg(coeff)
    ids = torch.arange(N)
    total = reg.compute_update_regul(model, _features(seed), ids,
                                     torch.device("cpu"), xnorm=1.0)
    return float(total.detach())


def test_zero_when_the_message_enters_like_the_leak():
    assert _term(_Model(gain=1.0)) < 1e-4


def test_blind_to_tau():
    """The same gain at a ten times shorter time constant is the same loss: the
    residual is divided by the batch's own leak scale, so tau cancels."""
    slow = _term(_Model(T=11.1086, gain=4.0))
    fast = _term(_Model(T=111.086, gain=4.0))
    assert abs(slow - fast) / max(slow, 1e-12) < 1e-3


def test_grows_with_the_gauge_error():
    """A message c times too small needs a gain of c, and the term charges it."""
    terms = [_term(_Model(gain=g)) for g in (1.0, 2.0, 5.0, 25.0)]
    assert terms == sorted(terms)
    assert terms[-1] > 10 * max(terms[1], 1e-12)


def test_descending_on_it_drives_the_gain_to_one():
    model = _Model(gain=6.0)
    opt = torch.optim.Adam(model.f_theta.parameters(), lr=0.2)
    for _ in range(400):
        opt.zero_grad()
        reg = _reg(1.0)
        loss = reg.compute_update_regul(model, _features(), torch.arange(N),
                                        torch.device("cpu"), xnorm=1.0)
        loss.backward()
        opt.step()
    assert abs(float(model.f_theta.gain.detach()) - 1.0) < 0.05


def test_off_by_default():
    assert _term(_Model(gain=6.0), coeff=0.0) == 0.0
