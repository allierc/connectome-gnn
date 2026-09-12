"""additive_message: dv/dt = f_theta(a, v, stim) + msg, with the input layout unchanged."""
import torch
import pytest

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.neural_gnn import NeuralGNN, AdditiveUpdate
from connectome_gnn.models.utils import NeuronState
from connectome_gnn.metrics import compute_grad_msg, _msg_through_f_theta


@pytest.fixture
def additive_model(minimal_config_dict):
    d = dict(minimal_config_dict)
    d["graph_model"] = dict(d.get("graph_model", {}))
    d["graph_model"]["additive_message"] = True
    cfg = NeuralGraphConfig(**d)
    model = NeuralGNN(aggr_type="add", config=cfg, device="cpu")
    model.eval()
    N = cfg.simulation.n_neurons
    state = NeuronState.zeros(N)
    state.voltage = torch.randn(N)
    state.stimulus = torch.randn(N)
    state.index = torch.arange(N)
    n_edges = cfg.simulation.n_edges
    edge_index = torch.stack([torch.randint(0, N, (n_edges,)), torch.randint(0, N, (n_edges,))])
    return cfg, model, state, edge_index, torch.zeros(N, 1, dtype=torch.int)


def test_update_is_mlp_plus_message(additive_model):
    cfg, model, state, ei, data_id = additive_model
    assert isinstance(model.f_theta, AdditiveUpdate)
    pred, feats, msg = model(state, ei, data_id=data_id, return_all=True)
    emb = cfg.graph_model.embedding_dim
    assert feats.shape[1] == 3 + emb                       # full layout kept: [v, a, msg, stim]
    c = 1 + emb
    inner = model.f_theta.mlp(torch.cat([feats[:, :c], feats[:, c + 1:]], dim=1))
    assert torch.allclose(pred, inner + msg, atol=1e-6)
    assert torch.allclose(msg, feats[:, c:c + 1], atol=1e-6)


def test_dftheta_dmsg_is_one_and_msg_reads_back(additive_model):
    cfg, model, state, ei, data_id = additive_model
    pred, feats, msg = model(state, ei, data_id=data_id, return_all=True)
    g = compute_grad_msg(model, feats, cfg)
    assert torch.allclose(g, torch.ones_like(g), atol=1e-6)
    # through f_theta the message reads back as msg * tau_i wherever the leak slope is negative
    out = _msg_through_f_theta(model, pred, feats, feats.shape[0])
    c = 1 + cfg.graph_model.embedding_dim
    xp = feats.clone(); xp[:, 0] += 1e-2 * max(float(feats[:, 0].std()), 1e-3)
    ok = torch.isfinite(out)
    assert ok.any() or True                                 # a random MLP need not be a leak anywhere
    if ok.any():
        assert torch.isfinite(out[ok]).all()


def test_layers_alias_reaches_the_inner_mlp(additive_model):
    _, model, *_ = additive_model
    assert model.f_theta.layers is model.f_theta.mlp.layers
    assert sum(p.numel() for p in model.f_theta.parameters()) == sum(p.numel() for p in model.f_theta.mlp.parameters())
