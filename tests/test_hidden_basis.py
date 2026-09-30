"""The basis-mixture hidden generator: starts at zero-silencing, mixes the right
columns, and picks a basis from the visible traces alone."""
import numpy as np
import torch

from connectome_gnn.models.hidden_basis import BasisMixHidden, select_basis


def test_it_starts_as_zero_silencing():
    g = BasisMixHidden(n_hidden=3, n_basis=4)
    g.basis_ids.copy_(torch.tensor([0, 2, 5, 7]))
    assert torch.equal(g(torch.randn(10)), torch.zeros(3))


def test_it_mixes_the_basis_columns_of_any_batch_shape():
    g = BasisMixHidden(n_hidden=2, n_basis=2)
    g.basis_ids.copy_(torch.tensor([1, 3]))
    with torch.no_grad():
        g.weight.copy_(torch.tensor([[1.0, 0.0], [0.5, 0.5]]))
        g.bias.copy_(torch.tensor([0.0, 1.0]))
    v = torch.arange(5.0)                                   # voltages 0..4
    assert torch.allclose(g(v), torch.tensor([1.0, 3.0]))   # v1 ; 0.5*v1 + 0.5*v3 + 1
    assert g(torch.stack([v, v])).shape == (2, 2)


def test_the_basis_spans_a_low_rank_population():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(2000, 5))
    traces = latent @ rng.normal(size=(5, 40)) + 1e-3 * rng.normal(size=(2000, 40))
    cols, explained = select_basis(traces, n_basis=5, n_frames=1000)
    assert len(set(cols.tolist())) == 5 and explained > 0.999


def test_gradients_reach_the_weights():
    g = BasisMixHidden(n_hidden=2, n_basis=3)
    g.basis_ids.copy_(torch.tensor([0, 1, 2]))
    g(torch.ones(4)).sum().backward()
    assert g.weight.grad.abs().sum() > 0 and g.bias.grad.abs().sum() > 0
