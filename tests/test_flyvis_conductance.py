"""Tests for the conductance-based flyvis twin: ODE, params, fit, and GNN.

None of these need flyvis: the twin's parameters, its integrator and the fit
stages are all flyvis-gnn's own. flyvis enters only when deriving a twin from
the pretrained model, which is covered by
``connectome_gnn.generators.verify_flyvis_conductance``.
"""
import builtins
import copy

import pytest
import torch

from connectome_gnn.generators.flyvis_conductance_fit import (
    ConductanceIndex,
    fit_synaptic_currents,
    group_conductances,
    initial_conductances,
    rollout,
    stimulus_from_movie,
)
from connectome_gnn.generators.flyvis_conductance_ode import FlyVisConductanceODE
from connectome_gnn.generators.ode_params import (
    FlyVisConductanceODEParams,
    get_ode_params_class,
)
from connectome_gnn.models.flyvis_conductance_gnn import FlyVisConductanceGNN
from connectome_gnn.models.registry import create_model, list_models
from connectome_gnn.neuron_state import NeuronState

pytestmark = pytest.mark.tier3


# --- ODE ---------------------------------------------------------------------


@pytest.fixture
def tiny_params():
    """Three neurons, two synapses: one excitatory, one inhibitory."""
    return FlyVisConductanceODEParams(
        tau_i=torch.tensor([0.05, 0.05, 0.05]),
        V_i_rest=torch.tensor([0.5, 0.5, 0.5]),
        edge_index=torch.tensor([[0, 1], [2, 2]]),
        G=torch.tensor([0.3, 0.2]),
        E_rev=torch.tensor([3.0, -2.0]),
        input_index=torch.tensor([[0, 1]]),
        reversal_exc=3.0,
        reversal_inh=-2.0,
    )


@pytest.fixture
def tiny_index(tiny_params):
    """Matching bookkeeping: both synapses form their own parameter group."""
    return ConductanceIndex(
        source_index=tiny_params.edge_index[0],
        target_index=tiny_params.edge_index[1],
        edge_group=torch.tensor([0, 1]),
        edge_syn_count=torch.tensor([2.0, 5.0]),
        edge_weight=torch.tensor([0.6, -1.0]),
        group_reversal=torch.tensor([3.0, -2.0]),
        n_groups=2,
        node_type=torch.tensor([0, 1, 2]),
        n_nodes=3,
        cell_types=["A", "B", "C"],
        type_nodes=[torch.tensor([0]), torch.tensor([1]), torch.tensor([2])],
        type_groups=[
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.tensor([0, 1]),
        ],
        type_pairs=[
            torch.zeros(1, 0, dtype=torch.long),
            torch.zeros(1, 0, dtype=torch.long),
            torch.tensor([[0, 1]]),
        ],
        pair_index=torch.tensor([0, 1]),
        n_pairs=2,
        input_index=torch.tensor([[0, 1]]),
    )


def _state(voltage, stimulus=None):
    voltage = torch.as_tensor(voltage, dtype=torch.float32)
    return NeuronState(
        voltage=voltage,
        stimulus=torch.zeros_like(voltage) if stimulus is None else stimulus,
        index=torch.arange(len(voltage)),
    )


def test_ode_matches_the_closed_form(tiny_params):
    """dv/dt must equal (-(v - v_rest) + sum g (E - v)) / tau."""
    ode = FlyVisConductanceODE(ode_params=tiny_params, device="cpu")
    v = torch.tensor([1.0, 2.0, 0.4])
    dv = ode(_state(v), tiny_params.edge_index)

    g_exc = 0.3 * torch.relu(v[0])
    g_inh = 0.2 * torch.relu(v[1])
    expected = (
        -(v[2] - 0.5) + g_exc * (3.0 - v[2]) + g_inh * (-2.0 - v[2])
    ) / 0.05
    assert dv[2, 0] == pytest.approx(float(expected), rel=1e-5)


def test_silent_presynaptic_cell_leaves_only_the_leak(tiny_params):
    """relu gates release: negative presynaptic voltage delivers nothing."""
    ode = FlyVisConductanceODE(ode_params=tiny_params, device="cpu")
    v = torch.tensor([-1.0, -1.0, 0.9])
    dv = ode(_state(v), tiny_params.edge_index)
    assert dv[2, 0] == pytest.approx((-(0.9 - 0.5)) / 0.05, rel=1e-5)


def test_time_constant_floor(tiny_params):
    """tau/(1+G) is floored at delta_t so forward Euler cannot overshoot."""
    strong = tiny_params.clone()
    strong.G = torch.tensor([50.0, 0.0])
    v = torch.tensor([1.0, 0.0, 0.0])
    delta_t = 0.01

    floored = FlyVisConductanceODE(ode_params=strong, device="cpu", delta_t=delta_t)
    dv = floored(_state(v), strong.edge_index)
    unfloored = FlyVisConductanceODE(ode_params=strong, device="cpu", delta_t=None)
    dv_unfloored = unfloored(_state(v), strong.edge_index)
    assert abs(float(dv_unfloored[2, 0])) > abs(float(dv[2, 0]))
    # a full floored step lands on the fixed point at most, never past it
    v_next = float(v[2]) + delta_t * float(dv[2, 0])
    v_inf = float(v[2]) + float(dv_unfloored[2, 0]) * strong.tau_i[2] / (
        1 + 50.0 * float(torch.relu(v[0]))
    )
    assert abs(v_next - float(v[2])) <= abs(v_inf - float(v[2])) + 1e-5


def test_voltage_stays_between_reversal_potentials(tiny_params):
    """Free rollout of the undriven network cannot leave the reversal hull."""
    ode = FlyVisConductanceODE(ode_params=tiny_params, device="cpu", delta_t=0.01)
    state = _state(torch.tensor([1.0, 1.0, 0.5]))
    for _ in range(500):
        dv = ode(state, tiny_params.edge_index)
        state.voltage = state.voltage + 0.01 * dv.squeeze(-1)
    assert float(state.voltage[2]) < tiny_params.reversal_exc
    assert float(state.voltage[2]) > tiny_params.reversal_inh
    assert torch.isfinite(state.voltage).all()


def test_params_roundtrip(tmp_path, tiny_params):
    tiny_params.save(str(tmp_path))
    loaded = FlyVisConductanceODEParams.load(str(tmp_path))
    for name in ("tau_i", "V_i_rest", "edge_index", "G", "E_rev", "input_index"):
        assert torch.equal(getattr(loaded, name), getattr(tiny_params, name))
    assert loaded.reversal_exc == tiny_params.reversal_exc


def test_params_are_registered():
    assert get_ode_params_class("flyvis_conductance") is FlyVisConductanceODEParams


def test_simulator_runs_without_flyvis(tmp_path, tiny_params, monkeypatch):
    """The shipped simulator must not need flyvis to be importable."""
    tiny_params.save(str(tmp_path))

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "flyvis" or name.startswith("flyvis."):
            raise ImportError(f"flyvis is unavailable in this test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError):
        __import__("flyvis")

    loaded = FlyVisConductanceODEParams.load(str(tmp_path))
    ode = FlyVisConductanceODE(ode_params=loaded, device="cpu", delta_t=0.01)
    stimulus = torch.zeros(5, 3)
    voltages = rollout(ode, loaded, stimulus, loaded.V_i_rest.clone(), 0.01)
    assert voltages.shape == (5, 3)
    assert torch.isfinite(voltages).all()


# --- fit ---------------------------------------------------------------------


def test_stimulus_from_movie_writes_only_photoreceptors():
    movie = torch.tensor([[[0.2, 0.7]], [[0.4, 0.9]]])  # (2 frames, 1, 2 hexals)
    input_index = torch.tensor([[0, 1], [2, 3]])        # two input cell types
    stimulus = stimulus_from_movie(movie, input_index, n_nodes=6)
    assert stimulus.shape == (2, 6)
    assert torch.equal(stimulus[0, :4], torch.tensor([0.2, 0.7, 0.2, 0.7]))
    assert torch.equal(stimulus[:, 4:], torch.zeros(2, 2))


def test_initial_conductances_reproduce_the_teacher_at_the_mean(tiny_index):
    """G (E - mean_v) must equal the teacher's signed weight at the mean voltage."""
    mean_voltage = torch.tensor([0.5, 0.5, 0.3])
    conductance = initial_conductances(tiny_index, mean_voltage)
    reversal = tiny_index.group_reversal[tiny_index.edge_group]
    recovered = conductance * (reversal - mean_voltage[tiny_index.target_index])
    assert torch.allclose(recovered, tiny_index.edge_weight, rtol=1e-5)
    assert (conductance > 0).all()


def test_group_conductances_factor_out_the_synapse_count(tiny_index):
    edge_conductance = tiny_index.edge_syn_count * torch.tensor([0.7, 0.3])
    assert torch.allclose(
        group_conductances(tiny_index, edge_conductance), torch.tensor([0.7, 0.3])
    )


def test_convex_fit_recovers_conductances_it_can_represent(tiny_index):
    """With a driving force the teacher's currents are exactly representable.

    Constructing the teacher's weights *from* a known conductance at a fixed
    postsynaptic voltage makes the regression consistent, so stage 2 must
    recover that conductance and leave no unexplained variance.
    """
    truth = torch.tensor([0.7, 0.4])
    postsynaptic = 0.25
    reversal = tiny_index.group_reversal
    index = copy.copy(tiny_index)
    index.edge_weight = (
        truth[index.edge_group]
        * index.edge_syn_count
        * (reversal[index.edge_group] - postsynaptic)
    )

    # every frame holds the postsynaptic cell at the voltage the weights assume
    voltages = torch.stack([
        torch.tensor([1.0, 2.0, postsynaptic]),
        torch.tensor([0.5, 1.5, postsynaptic]),
        torch.tensor([2.0, 0.5, postsynaptic]),
    ])
    prior = torch.ones(2)
    conductance, diagnostics = fit_synaptic_currents(
        index, [voltages], prior=prior, time_chunk=2, ridge=1e-12
    )
    assert diagnostics["unexplained_current_fraction"] < 1e-9
    recovered = group_conductances(index, conductance)
    assert torch.allclose(recovered, truth, rtol=1e-3)


def test_convex_fit_returns_non_negative_conductances(tiny_index):
    voltages = torch.stack([
        torch.tensor([1.0, 2.0, 0.4]),
        torch.tensor([0.5, 1.5, 0.9]),
    ])
    conductance, _ = fit_synaptic_currents(
        tiny_index, [voltages], prior=torch.ones(2), time_chunk=2
    )
    assert (conductance >= 0).all()


# --- GNN ---------------------------------------------------------------------


def _conductance_config(minimal_config_dict, name):
    config_dict = copy.deepcopy(minimal_config_dict)
    config_dict["graph_model"]["signal_model_name"] = name
    embedding_dim = config_dict["graph_model"]["embedding_dim"]
    config_dict["graph_model"]["input_size"] = (
        1 + embedding_dim if "factorized" in name else 2 + 2 * embedding_dim
    )
    from connectome_gnn.config import NeuralGraphConfig

    return NeuralGraphConfig(**config_dict)


@pytest.fixture(params=["flyvis_conductance", "flyvis_conductance_factorized"])
def gnn_and_inputs(request, minimal_config_dict):
    config = _conductance_config(minimal_config_dict, request.param)
    model = FlyVisConductanceGNN(aggr_type="add", config=config, device="cpu")
    model.eval()

    n_neurons = config.simulation.n_neurons
    state = NeuronState.zeros(n_neurons)
    state.voltage = torch.randn(n_neurons)
    state.stimulus = torch.randn(n_neurons)
    state.index = torch.arange(n_neurons)

    n_edges = config.simulation.n_edges
    edge_index = torch.stack([
        torch.randint(0, n_neurons, (n_edges,)),
        torch.randint(0, n_neurons, (n_edges,)),
    ])
    return model, state, edge_index, torch.zeros(n_neurons, 1, dtype=torch.int), config


def test_gnn_output_shape(gnn_and_inputs):
    model, state, edge_index, data_id, _ = gnn_and_inputs
    pred = model(state, edge_index, data_id=data_id)
    assert pred.shape == (state.n_neurons, 1)
    assert torch.isfinite(pred).all()


def test_gnn_types_are_registered():
    assert "flyvis_conductance" in list_models()
    assert "flyvis_conductance_factorized" in list_models()


def test_gnn_created_through_the_registry(minimal_config_dict):
    config = _conductance_config(minimal_config_dict, "flyvis_conductance")
    model = create_model("flyvis_conductance", config=config, device="cpu")
    assert isinstance(model, FlyVisConductanceGNN)


def test_message_depends_on_postsynaptic_voltage(gnn_and_inputs):
    """The point of this model type: changing v_i must change the message."""
    model, state, _, _, config = gnn_and_inputs
    n_neurons = config.simulation.n_neurons
    n_edges = config.simulation.n_edges
    embedding = model.a[state.index.long()].squeeze()
    v = state.voltage.unsqueeze(-1)

    # every edge targets node 0, which sends nothing, so perturbing v_0 leaves
    # all presynaptic inputs untouched
    sink = 0
    sources = torch.arange(1, n_edges + 1) % (n_neurons - 1) + 1
    edge_index = torch.stack([sources, torch.full((n_edges,), sink)])

    baseline = model._compute_messages(v, embedding, edge_index)
    perturbed_v = v.clone()
    perturbed_v[sink] += 1.0
    perturbed = model._compute_messages(perturbed_v, embedding, edge_index)
    assert not torch.allclose(baseline[sink], perturbed[sink])


def test_wrong_input_size_is_rejected(minimal_config_dict):
    config = _conductance_config(minimal_config_dict, "flyvis_conductance")
    config.graph_model.input_size = 3  # the flyvis_A value
    with pytest.raises(ValueError, match="input_size"):
        FlyVisConductanceGNN(config=config, device="cpu")


def test_factorized_exposes_conductance_and_reversal(minimal_config_dict):
    config = _conductance_config(minimal_config_dict, "flyvis_conductance_factorized")
    model = FlyVisConductanceGNN(config=config, device="cpu")
    n_neurons = config.simulation.n_neurons
    n_edges = config.simulation.n_edges
    v = torch.randn(n_neurons, 1)
    embedding = model.a[torch.arange(n_neurons)].squeeze()
    edge_index = torch.stack([
        torch.randint(0, n_neurons, (n_edges,)),
        torch.randint(0, n_neurons, (n_edges,)),
    ])
    conductance, reversal = model.edge_conductance(v, embedding, edge_index)
    assert conductance.shape == (n_edges, 1)
    assert reversal.shape == (n_edges, 1)
    assert (conductance >= 0).all()


def test_unfactorized_cannot_separate_conductance(minimal_config_dict):
    config = _conductance_config(minimal_config_dict, "flyvis_conductance")
    model = FlyVisConductanceGNN(config=config, device="cpu")
    with pytest.raises(NotImplementedError):
        model.edge_conductance(
            torch.randn(config.simulation.n_neurons, 1),
            model.a,
            torch.zeros(2, 4, dtype=torch.long),
        )


# --- identifiability metric --------------------------------------------------


class _FakeTimeSeries:
    """Minimal stand-in for NeuronTimeSeries: compute_activity_stats needs .voltage."""

    def __init__(self, voltage):
        self.voltage = voltage


def _dataset_with_twin(tmp_path, monkeypatch, n_neurons, n_edges, truth_G, truth_E,
                       edge_index):
    """Write a conductance ground truth where the metric will look for it."""
    twin = FlyVisConductanceODEParams(
        tau_i=torch.full((n_neurons,), 0.05),
        V_i_rest=torch.full((n_neurons,), 0.5),
        edge_index=edge_index,
        G=truth_G,
        E_rev=truth_E,
        reversal_exc=3.0,
        reversal_inh=-2.0,
    )
    twin.save(str(tmp_path), filename="conductance_params.pt")
    monkeypatch.setattr(
        "connectome_gnn.metrics.graphs_data_path", lambda *parts: str(tmp_path)
    )
    return twin


def test_conductance_r2_scores_a_factorized_model(tmp_path, monkeypatch,
                                                  minimal_config_dict):
    from connectome_gnn.metrics import compute_conductance_r2

    config = _conductance_config(minimal_config_dict, "flyvis_conductance_factorized")
    model = FlyVisConductanceGNN(config=config, device="cpu")
    n_neurons = config.simulation.n_neurons
    n_edges = config.simulation.n_edges
    edge_index = torch.stack([
        torch.arange(n_edges) % n_neurons,
        (torch.arange(n_edges) + 1) % n_neurons,
    ])
    _dataset_with_twin(
        tmp_path, monkeypatch, n_neurons, n_edges,
        truth_G=torch.rand(n_edges) + 0.1,
        truth_E=torch.where(
            torch.arange(n_edges) % 2 == 0,
            torch.full((n_edges,), 3.0),
            torch.full((n_edges,), -2.0),
        ),
        edge_index=edge_index,
    )

    x_ts = _FakeTimeSeries(torch.randn(20, n_neurons))
    scores = compute_conductance_r2(model, x_ts, config, "cpu", n_neurons)
    conductance_r2 = scores["conductance_r2"]
    reversal_r2 = scores["reversal_r2"]
    # G varies across edges through g_phi(v_j), so it is comparable ...
    assert conductance_r2 == conductance_r2  # not nan
    assert -10 < conductance_r2 <= 1
    # ... while an untrained model's embeddings are identical on every edge, so
    # its reversal potential is flat and the metric reports nan rather than a
    # spurious number fitted through rounding noise
    assert reversal_r2 != reversal_r2


def test_conductance_r2_is_nan_without_a_conductance_ground_truth(
        tmp_path, monkeypatch, minimal_config_dict):
    from connectome_gnn.metrics import compute_conductance_r2

    config = _conductance_config(minimal_config_dict, "flyvis_conductance_factorized")
    model = FlyVisConductanceGNN(config=config, device="cpu")
    monkeypatch.setattr(
        "connectome_gnn.metrics.graphs_data_path", lambda *parts: str(tmp_path)
    )
    x_ts = _FakeTimeSeries(torch.randn(5, config.simulation.n_neurons))
    scores = compute_conductance_r2(
        model, x_ts, config, "cpu", config.simulation.n_neurons
    )
    assert all(v != v for v in scores.values())  # nan, nan


def test_conductance_r2_is_nan_for_a_model_that_cannot_factorize(
        tmp_path, monkeypatch, minimal_config_dict):
    """NeuralGNN has no edge_conductance; the unfactorized variant raises."""
    from connectome_gnn.metrics import compute_conductance_r2
    from connectome_gnn.models.neural_gnn import NeuralGNN

    n_neurons = minimal_config_dict["simulation"]["n_neurons"]
    n_edges = minimal_config_dict["simulation"]["n_edges"]
    edge_index = torch.stack([
        torch.arange(n_edges) % n_neurons,
        (torch.arange(n_edges) + 1) % n_neurons,
    ])
    _dataset_with_twin(
        tmp_path, monkeypatch, n_neurons, n_edges,
        truth_G=torch.rand(n_edges) + 0.1,
        truth_E=torch.full((n_edges,), 3.0),
        edge_index=edge_index,
    )
    x_ts = _FakeTimeSeries(torch.randn(5, n_neurons))

    from connectome_gnn.config import NeuralGraphConfig
    old_config = NeuralGraphConfig(**minimal_config_dict)
    old_model = NeuralGNN(config=old_config, device="cpu")
    scores = compute_conductance_r2(old_model, x_ts, old_config, "cpu", n_neurons)
    assert all(v != v for v in scores.values())  # nan, nan

    unfactorized = FlyVisConductanceGNN(
        config=_conductance_config(minimal_config_dict, "flyvis_conductance"),
        device="cpu",
    )
    scores = compute_conductance_r2(unfactorized, x_ts, old_config, "cpu", n_neurons)
    assert all(v != v for v in scores.values())


def test_conductance_r2_recovers_a_perfect_match(tmp_path, monkeypatch,
                                                 minimal_config_dict):
    """If the ground truth is read back out of the model, R2 must be ~1."""
    from connectome_gnn.metrics import compute_conductance_r2

    config = _conductance_config(minimal_config_dict, "flyvis_conductance_factorized")
    model = FlyVisConductanceGNN(config=config, device="cpu")
    n_neurons = config.simulation.n_neurons
    n_edges = config.simulation.n_edges
    edge_index = torch.stack([
        torch.arange(n_edges) % n_neurons,
        (torch.arange(n_edges) + 1) % n_neurons,
    ])
    x_ts = _FakeTimeSeries(torch.randn(20, n_neurons))

    # embeddings start identical for every neuron, which makes e_psi — and so
    # the reversal potential — the same on every edge; differentiate them so the
    # comparison has any variance to explain
    with torch.no_grad():
        model.a.copy_(torch.randn_like(model.a))

    # take the model's own conductance and reversal as the ground truth
    voltage = x_ts.voltage.mean(0)[:, None]
    embedding = model.a[torch.arange(n_neurons)].squeeze()
    with torch.no_grad():
        conductance, reversal = model.edge_conductance(voltage, embedding, edge_index)
    _dataset_with_twin(
        tmp_path, monkeypatch, n_neurons, n_edges,
        truth_G=conductance.squeeze().clone(),
        truth_E=reversal.squeeze().clone(),
        edge_index=edge_index,
    )

    scores = compute_conductance_r2(model, x_ts, config, "cpu", n_neurons)
    assert scores["conductance_r2"] > 0.99
    assert scores["reversal_r2"] > 0.99
