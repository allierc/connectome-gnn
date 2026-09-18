"""Edge dropout must be a training-only, UNBIASED perturbation of the message.

The loss reads only the sum of a neuron's incoming messages,
`msg_i = sum_j W_ij g_phi(v_j, v_i)`, so any reallocation among a neuron's edges
that preserves that sum costs nothing -- which is why `msg_i_R2` reaches 0.94 on
the flowcond data while `Wij_R2` stalls near 0.4-0.57. Dropping edges at random
removes the partner a wrong weight was compensating against.

Two properties make or break it, and both are pinned here:

  * OFF IN EVAL. Every readout -- the template fit, the rollout, `-o test_plot` --
    runs under `model.eval()`. A dropout that stayed on there would randomise the
    numbers it is supposed to improve.
  * UNBIASED IN TRAIN. The surviving messages are scaled by `1/(1 - p)`, so
    `E[msg_i]` is unchanged. Without that scaling every `W` would simply grow by
    `1/(1 - p)` to restore the message, the per-neuron gauge `k_i` would absorb
    the shift, and the regularisation would cost accuracy while buying nothing.
    The test distinguishes the two by averaging over growing numbers of draws:
    an unbiased estimator's error falls as `1/sqrt(n)`, a biased one plateaus.
"""
import pytest
import torch

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.registry import create_model

pytestmark = pytest.mark.tier2

N, E, EMB = 12, 40, 2
P = 0.3


@pytest.fixture(autouse=True)
def _cpu():
    torch.set_default_device("cpu")
    yield
    torch.set_default_device("cpu")


def _model(cfg_dict, p):
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_dict.items()}
    d["simulation"] = {**d["simulation"], "n_neurons": N, "n_edges": E}
    d["training"] = {**d["training"], "edge_dropout": p}
    cfg = NeuralGraphConfig(**d)
    return create_model(cfg.graph_model.signal_model_name,
                        aggr_type=cfg.graph_model.aggr_type, config=cfg, device="cpu")


def _inputs():
    g = torch.Generator().manual_seed(0)
    v = torch.rand(N, 1, generator=g)
    emb = torch.rand(N, EMB, generator=g)
    src = torch.randint(0, N, (E,), generator=g)
    dst = torch.randint(0, N, (E,), generator=g)
    return v, emb, torch.stack([src, dst])


def _pair(cfg_dict):
    a = _model(cfg_dict, 0.0)
    b = _model(cfg_dict, P)
    b.load_state_dict(a.state_dict())
    return a, b


def test_rejects_out_of_range(minimal_config_dict):
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="edge_dropout"):
            _model(minimal_config_dict, bad)


def test_off_in_eval(minimal_config_dict):
    """eval() must reproduce p=0 exactly, not approximately."""
    v, emb, ei = _inputs()
    a, b = _pair(minimal_config_dict)
    a.eval(); b.eval()
    with torch.no_grad():
        for _ in range(5):
            assert torch.equal(a._compute_messages(v, emb, ei),
                               b._compute_messages(v, emb, ei))


def test_on_in_train(minimal_config_dict):
    """train() must actually perturb, or the flag is silently a no-op."""
    v, emb, ei = _inputs()
    a, b = _pair(minimal_config_dict)
    a.eval(); b.train()
    torch.manual_seed(0)
    with torch.no_grad():
        ref = a._compute_messages(v, emb, ei)
        scale = ref.abs().mean().clamp_min(1e-12)
        draws = [float((b._compute_messages(v, emb, ei) - ref).abs().mean() / scale)
                 for _ in range(8)]
    assert min(draws) > 0.05, f"dropout barely moved the message: {draws}"


def test_unbiased_in_train(minimal_config_dict):
    """The 1/(1-p) scaling: error must fall as 1/sqrt(n), not plateau."""
    v, emb, ei = _inputs()
    a, b = _pair(minimal_config_dict)
    a.eval(); b.train()
    torch.manual_seed(0)
    with torch.no_grad():
        ref = a._compute_messages(v, emb, ei)
        scale = ref.abs().mean().clamp_min(1e-12)
        # float64: 8000 float32 additions accumulate a rounding error of the
        # same order as the sampling error this test is trying to measure.
        acc = torch.zeros_like(ref, dtype=torch.float64)
        done, err = 0, {}
        for n in (500, 8000):
            for _ in range(n - done):
                acc += b._compute_messages(v, emb, ei).double()
            done = n
            err[n] = float((acc / n - ref.double()).abs().mean() / scale)
    # 16x the draws is 4x less error when unbiased. Allow a factor of 2 slack on
    # that 4x; a biased estimator sits at a constant and fails by a mile.
    assert err[8000] < err[500] / 2.0, err
