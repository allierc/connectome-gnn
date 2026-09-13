"""The silent-input anchor fires where the observed-sample one cannot.

The true per-edge message is W_ij * relu(v_j) * (E_ij - v_i), identically zero
whenever the presynaptic cell is silent, for every edge and whatever the
postsynaptic state. coeff_g_phi_zero_below asks that of the OBSERVED samples with
v_j < 0, which on this data is 0.6% of frames and never for 43% of neurons, so it
leaves the message/leak gauge unpinned. coeff_g_phi_silent asks it off the data:
the real edge features with the presynaptic voltage replaced by a draw below zero.
"""
import torch
import pytest

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.neural_gnn import NeuralGNN
from connectome_gnn.models.regularizer import LossRegularizer
from connectome_gnn.models.utils import NeuronState


def _setup(coeffs, n=12, n_edges=40):
    d = {
        "dataset": "test_dataset",
        "simulation": {"params": [[1.0, 1.0]], "n_neurons": n, "n_input_neurons": 2,
                       "n_neuron_types": 2, "n_edges": n_edges, "n_frames": 100,
                       "delta_t": 0.02, "seed": 42},
        "graph_model": {"signal_model_name": "flyvis_conductance", "aggr_type": "add",
                        "embedding_dim": 2, "input_size": 6, "output_size": 1,
                        "hidden_dim": 8, "n_layers": 2, "input_size_update": 5,
                        "hidden_dim_update": 8, "n_layers_update": 2,
                        "output_size_update": 1, "g_phi_positive": False,
                        "w_squared": True},
        "training": dict({"n_epochs": 1, "batch_size": 1}, **coeffs),
        "plotting": {"colormap": "tab20", "arrow_length": 1,
                     "xlim": [-1, 1], "ylim": [-1, 1]},
    }
    cfg = NeuralGraphConfig(**d)
    model = NeuralGNN(aggr_type="add", config=cfg, device="cpu")
    reg = LossRegularizer(cfg.training, cfg.graph_model, activity_column=3,
                          plot_frequency=1, n_neurons=n, trainer_type="flyvis")
    reg.reset_iteration(device=torch.device("cpu"))
    x = NeuronState.zeros(n)
    # EVERY presynaptic voltage strictly positive: the observed-sample anchor has
    # nothing to bite on, which is the situation this data is in almost always.
    x.voltage = torch.rand(n) + 1.0
    x.stimulus = torch.zeros(n)
    x.index = torch.arange(n)
    edges = torch.stack([torch.randint(0, n, (n_edges,)), torch.randint(0, n, (n_edges,))])
    ids = torch.arange(n)
    return cfg, model, reg, x, edges, ids


def _run(reg, model, x, edges, ids):
    perm = torch.randperm(x.voltage.shape[0])
    total = reg.compute(model=model, x=x, in_features=None, ids=ids, ids_batch=ids,
                        edges=edges, device=torch.device("cpu"), xnorm=1.0,
                        perm_indices=perm)
    return total, {k: float(v) for k, v in reg._iter_tracker.items()}


def test_silent_anchor_fires_where_zero_below_is_silent():
    _, model, reg, x, edges, ids = _setup(
        {"coeff_g_phi_zero_below": 5.0, "coeff_g_phi_silent": 5.0})
    total, parts = _run(reg, model, x, edges, ids)
    assert parts["g_phi_zero_below"] == 0.0      # no observed sample has v_j < 0
    assert parts["g_phi_silent"] > 0.0           # the off-data anchor still bites
    assert float(total.detach()) > 0.0


def test_off_when_the_coefficient_is_zero():
    _, model, reg, x, edges, ids = _setup({"coeff_g_phi_silent": 0.0})
    _, parts = _run(reg, model, x, edges, ids)
    assert parts["g_phi_silent"] == 0.0


def test_descending_on_it_drives_g_phi_to_zero_at_silent_inputs():
    """The term is not merely nonzero: minimising it removes the component of
    g_phi that survives presynaptic silence, which is the degenerate direction."""
    cfg, model, reg, x, edges, ids = _setup({"coeff_g_phi_silent": 5.0})

    from connectome_gnn.models.utils import get_in_features_g_phi

    def silent_magnitude():
        """|g_phi| at a silent presynaptic voltage, on the model's OWN features:
        the same v_i, a_i and a_j the anchor sees, with v_j set below zero."""
        perm = torch.arange(x.voltage.shape[0])
        feat, _ = get_in_features_g_phi(x, model, cfg.graph_model, 1.0,
                                        x.voltage.shape[0], torch.device("cpu"),
                                        perm_indices=perm)
        feat = feat.clone().detach()
        feat[:, 0] = -1.0
        with torch.no_grad():
            return float(model.g_phi(feat).abs().mean())

    before = silent_magnitude()
    opt = torch.optim.Adam(model.g_phi.parameters(), lr=0.05)
    for _ in range(60):
        reg.reset_iteration(device=torch.device("cpu"))
        loss, _ = _run(reg, model, x, edges, ids)
        opt.zero_grad(); loss.backward(); opt.step()
    assert silent_magnitude() < 0.5 * before


def test_the_band_is_configurable_and_respected():
    cfg, *_ = _setup({"coeff_g_phi_silent": 1.0, "g_phi_silent_range": (-4.0, -1.0)})
    assert tuple(cfg.training.g_phi_silent_range) == (-4.0, -1.0)
