"""Hidden-neuron voltages as a learned mixture of a few visible neurons' voltages.

WHY. A trace network of time (SIREN, NGP-T) gives each hidden neuron a free
number per frame; trained only through the visible neurons it projects to, it
absorbs their noise and recovers nothing (experiment 9: trace r ~0). Neurons
of a circuit are strongly correlated, so a hidden voltage is far better
described as a weighted sum of observed ones:

    h_i(t) = sum_m W_im v_{b_m}(t) + c_i

over a BASIS of M visible neurons b_1..b_M. It is differentiable in W and c,
bounded to the span of real traces, and a function of the CURRENT state, so it
runs on held-out videos and in a free-running rollout alike.

BLIND. The basis is chosen from the visible neurons' traces alone (pivoted QR
on their standardised training traces: the M columns that best span the
others); nothing about the hidden neurons is used. W and c start at 0, so
training starts exactly at the zero-silenced baseline.
"""
import numpy as np
import torch
import torch.nn as nn


class BasisMixHidden(nn.Module):
    """h = v[basis] @ W.T + c, one row of W per hidden neuron (hidden_ids order)."""

    def __init__(self, n_hidden: int, n_basis: int):
        super().__init__()
        self.register_buffer("basis_ids", torch.zeros(n_basis, dtype=torch.long))
        self.weight = nn.Parameter(torch.zeros(n_hidden, n_basis))
        self.bias = nn.Parameter(torch.zeros(n_hidden))

    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        """voltage (..., N) -> hidden voltages (..., n_hidden)."""
        return voltage[..., self.basis_ids] @ self.weight.T + self.bias


def select_basis(voltage: np.ndarray, n_basis: int, n_frames: int = 4000, seed: int = 0):
    """The n_basis columns of `voltage` (frames x candidates) that best span the rest.

    Pivoted QR on the standardised traces of a random subset of frames. Returns
    (column indices in pivot order, fraction of the candidates' total variance
    the chosen columns explain by least squares).
    """
    from scipy.linalg import qr, lstsq
    v = np.asarray(voltage, dtype=np.float64)
    rng = np.random.default_rng(seed)
    fr = np.sort(rng.choice(v.shape[0], min(n_frames, v.shape[0]), replace=False))
    z = v[fr]
    z = (z - z.mean(0)) / (z.std(0) + 1e-9)
    _, _, piv = qr(z, mode="economic", pivoting=True)
    cols = np.asarray(piv[:n_basis], dtype=np.int64)
    coef = lstsq(z[:, cols], z)[0]
    resid = z - z[:, cols] @ coef
    explained = 1.0 - float((resid ** 2).sum() / (z ** 2).sum())
    return cols, explained


def init_hidden_basis(model, hn, x_ts, config, log_dir):
    """Choose the basis of a `basis_mix` hidden generator from the visible traces.

    A no-op for any other generator, when there are no hidden neurons, or when
    the basis is already set (a resumed run: the ids are in the checkpoint).
    Saves `hidden_basis_ids.pt` beside `hidden_neuron_ids.pt`.
    """
    import os
    core = getattr(model, "_orig_mod", model)
    gen = getattr(core, "NNR_hidden", None)
    if not isinstance(gen, BasisMixHidden) or not hn.has_hidden:
        return
    path = os.path.join(log_dir, "hidden_basis_ids.pt")
    if os.path.exists(path):
        gen.basis_ids.copy_(torch.load(path, map_location=gen.basis_ids.device))
        return
    visible = hn.visible_ids.detach().cpu().numpy()
    v = x_ts.voltage
    v = (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))[:, visible]
    cols, explained = select_basis(v, gen.basis_ids.numel(),
                                   seed=int(getattr(config.training, "seed", 0) or 0))
    ids = torch.as_tensor(visible[cols], dtype=torch.long, device=gen.basis_ids.device)
    gen.basis_ids.copy_(ids)
    torch.save(ids.cpu(), path)
    print(f"hidden basis: {ids.numel()} of {visible.size} visible neurons, "
          f"explaining {100 * explained:.1f}% of the visible variance")
