"""LossRegularizer.compute must survive torch.compile(fullgraph=True).

graph_trainer compiles it with fullgraph=True, so a single untraceable op in it
disables torch.compile for the whole run. That is exactly what happened: a
torch.randperm building g_phi's random (a_i, a_j) partner pairs forced every
flyvis_conductance config to carry torch_compile: false. The permutation is now
drawn by sample_g_phi_perm() just outside the compiled call and passed in.

Three properties are pinned here, because breaking any of them is silent:
  1. compute() compiles under fullgraph=True for a conductance model.
  2. The permutation is still resampled every call — a "fix" that froze it to a
     constant would compile fine while enforcing the g_phi priors on one fixed
     pairing out of n_neurons! possible ones.
  3. Non-conductance models draw NOTHING, so the RNG stream (and hence the
     bit-reproducibility of every flyvis_A reference run) is untouched.
"""
import pytest
import torch

from connectome_gnn.config import GraphModelConfig, TrainingConfig
from connectome_gnn.models.MLP import MLP
from connectome_gnn.models.regularizer import LossRegularizer
from connectome_gnn.neuron_state import NeuronState

pytestmark = pytest.mark.tier2

N, E, EMB = 64, 200, 2
DEV = torch.device("cpu")


def _model_config(signal_model_name):
    return GraphModelConfig(
        signal_model_name=signal_model_name, prediction="first_derivative",
        input_size=3, n_layers=2, hidden_dim=16, output_size=1,
        input_size_update=5, n_layers_update=2, hidden_dim_update=16,
        output_size_update=1, aggr_type="add", embedding_dim=EMB,
        update_type="generic", g_phi_positive=True,
    )


def _train_config():
    # Only the two coefficients that reach get_in_features_g_phi are on.
    return TrainingConfig(n_epochs=1, batch_size=1, coeff_g_phi_diff=1.0,
                          coeff_g_phi_norm=1.0)


class _FakeModel(torch.nn.Module):
    """Only what regularizer.compute touches: W, a, g_phi, f_theta."""

    def __init__(self, mc):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(E))
        self.a = torch.nn.Parameter(torch.randn(N, EMB))
        n_in = 2 + 2 * EMB if mc.signal_model_name == "flyvis_conductance" else 1 + EMB
        self.g_phi = MLP(input_size=n_in, output_size=1, nlayers=2, hidden_size=16, device="cpu")
        self.f_theta = MLP(input_size=mc.input_size_update, output_size=1,
                           nlayers=2, hidden_size=16, device="cpu")


def _build(signal_model_name):
    mc = _model_config(signal_model_name)
    torch.manual_seed(0)
    model = _FakeModel(mc)
    reg = LossRegularizer(train_config=_train_config(), model_config=mc,
                          activity_column=3, plot_frequency=1, n_neurons=N,
                          trainer_type="flyvis", n_neuron_types=0)
    reg.set_epoch(0)
    return model, reg


def _call(reg_fn, model, perm):
    x = NeuronState(index=torch.arange(N), voltage=torch.zeros(N))
    return reg_fn(model=model, x=x, in_features=None, ids=torch.arange(N),
                  ids_batch=None, edges=torch.randint(0, N, (2, E)),
                  device=DEV, xnorm=1.0, perm_indices=perm)


class TestGPhiPermHoist:
    def test_compiles_fullgraph_for_conductance(self):
        model, reg = _build("flyvis_conductance")
        compiled = torch.compile(reg.compute, fullgraph=True)
        reg.reset_iteration(device=DEV)
        out = _call(compiled, model, reg.sample_g_phi_perm(DEV))
        assert torch.isfinite(out)

    def test_permutation_is_resampled_every_call(self):
        _, reg = _build("flyvis_conductance")
        reg.reset_iteration(device=DEV)
        perms = [reg.sample_g_phi_perm(DEV) for _ in range(5)]
        assert all(p is not None and p.shape == (N,) for p in perms)
        # A frozen permutation would still compile but would pin the g_phi priors
        # to a single pairing of embeddings.
        assert not all(torch.equal(perms[0], p) for p in perms[1:])

    def test_compiled_matches_eager(self):
        model, reg = _build("flyvis_conductance")
        reg.reset_iteration(device=DEV)
        perm = reg.sample_g_phi_perm(DEV)
        eager = _call(reg.compute, model, perm)
        compiled = _call(torch.compile(reg.compute, fullgraph=True), model, perm)
        torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("signal_model_name", ["flyvis_A", "flyvis_C", "flyvis_D"])
    def test_non_conductance_consumes_no_rng(self, signal_model_name):
        """Guards bit-reproducibility of every existing flyvis_A reference run."""
        _, reg = _build(signal_model_name)
        reg.reset_iteration(device=DEV)
        before = torch.random.get_rng_state()
        assert reg.sample_g_phi_perm(DEV) is None
        assert torch.equal(before, torch.random.get_rng_state())

    def test_no_perm_when_g_phi_priors_are_off(self):
        """Mirrors the old guard: randperm only ran when a g_phi prior was active."""
        mc = _model_config("flyvis_conductance")
        reg = LossRegularizer(train_config=TrainingConfig(n_epochs=1, batch_size=1),
                              model_config=mc, activity_column=3, plot_frequency=1,
                              n_neurons=N, trainer_type="flyvis", n_neuron_types=0)
        reg.set_epoch(0)
        reg.reset_iteration(device=DEV)
        assert not reg.needs_g_phi_perm()
        assert reg.sample_g_phi_perm(DEV) is None
