"""training.rollout_burn_in leaves the first B rollout steps unscored, and keeps
the gradient flowing back through them.

WHY IT EXISTS. Under measurement noise the dense rollout loss is biased in its
early steps: at step 0 the input is v + eta while the observed-difference
target carries -eta/dt, and that eta lingers in the state for steps 1-4 while
those targets hold only fresh noise. On exp03 at meas 0.20 the trained GNN
scores 5.9% below the TRUE generator on the dense loss, all of it in steps 0-8.
A known-ODE test lifted R2_W 0.921 -> 0.988 by leaving 8 steps unscored.

TWO PROPERTIES, AND THE SECOND IS THE ONE THAT MATTERS. The mask is easy: steps
0 .. B-1 carry no weight. But the same known-ODE test measured a burn-in that
DETACHED the state at the boundary at 0.753 -- worse than no burn-in. So these
tests run the production _dense_rollout_loss and check that the gradient through
a burn-in matches a finite difference, and differs from the detached variant.
The stride's off-by-one survived thirty cluster runs because every test of it
checked a list of floats and none ran the loss; this file does not repeat that.
"""
import types

import numpy as np
import pytest
import torch

from connectome_gnn.models.recurrent_step import (
    _dense_rollout_loss, _rollout_step_weights, validate_rollout_masks)
from connectome_gnn.neuron_state import NeuronTimeSeries


def scored(weights):
    """0-indexed steps with weight: step s holds the state at frame k+s."""
    return [s for s, w in enumerate(weights) if w > 0.0]


# --------------------------------------------------------------------------- #
#  the mask                                                                    #
# --------------------------------------------------------------------------- #
def test_burn_in_scores_from_step_b():
    assert scored(_rollout_step_weights("uniform", 20, 0.9, burn_in=8)) == list(range(8, 20))


def test_zero_burn_in_changes_nothing():
    for w in ("uniform", "discount", "linear_decay", "last"):
        assert (_rollout_step_weights(w, 12, 0.9, burn_in=0)
                == _rollout_step_weights(w, 12, 0.9))


def test_a_scored_step_keeps_its_dense_weight():
    """A mask, not a reweighting: the discount survives on the scored steps."""
    dense = _rollout_step_weights("discount", 20, 0.9)
    burned = _rollout_step_weights("discount", 20, 0.9, burn_in=8)
    assert burned[8:] == dense[8:]
    assert burned[:8] == [0.0] * 8


def test_burn_in_composes_with_the_stride():
    """Stride 5 over horizon 21 scores 0, 5, 10, 15, 20; burn-in 8 removes 0 and 5."""
    assert scored(_rollout_step_weights("uniform", 21, 0.9, loss_stride=5, burn_in=8)) == [10, 15, 20]


def test_last_weighting_is_a_burn_in_of_k_minus_one():
    """'last' already scores only step K-1, so a burn-in below it changes nothing."""
    assert (_rollout_step_weights("last", 20, 0.9, burn_in=8)
            == _rollout_step_weights("last", 20, 0.9))


def test_a_horizon_inside_the_burn_in_scores_nothing():
    for k in (1, 5, 8):
        assert scored(_rollout_step_weights("uniform", k, 0.9, burn_in=8)) == []
    assert scored(_rollout_step_weights("uniform", 9, 0.9, burn_in=8)) == [8]


# --------------------------------------------------------------------------- #
#  the guards                                                                  #
# --------------------------------------------------------------------------- #
def _training(**kw):
    base = dict(rollout_horizon_schedule=[], rollout_loss_stride=0,
                rollout_burn_in=0, rollout_step_weighting="uniform",
                rollout_discount=0.9)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_guard_accepts_a_schedule_past_the_burn_in():
    validate_rollout_masks(_training(rollout_horizon_schedule=list(range(9, 21)),
                                     rollout_burn_in=8))


def test_guard_refuses_a_horizon_inside_the_burn_in():
    """exp03's [1..20] ramp with burn-in 8: epochs 0-7 would train on nothing."""
    with pytest.raises(ValueError, match=r"\(0, 1\), \(1, 2\).*\(7, 8\)\]"):
        validate_rollout_masks(_training(rollout_horizon_schedule=list(range(1, 21)),
                                         rollout_burn_in=8))


def test_guard_refuses_a_burn_in_with_no_schedule():
    """Without a schedule the legacy path runs and would ignore the burn-in."""
    with pytest.raises(ValueError, match="silently"):
        validate_rollout_masks(_training(rollout_burn_in=8))


def test_guard_refuses_a_stride_with_no_schedule():
    with pytest.raises(ValueError, match="silently"):
        validate_rollout_masks(_training(rollout_loss_stride=5))


def test_guard_refuses_a_stride_horizon_that_reaches_only_the_anchor():
    with pytest.raises(ValueError, match=r"\[5\]"):
        validate_rollout_masks(_training(rollout_horizon_schedule=[5, 11],
                                         rollout_loss_stride=5))


def test_guard_sees_stride_and_burn_in_together():
    """Stride 5 at horizon 6 scores 0 and 5; burn-in 8 removes both."""
    with pytest.raises(ValueError, match=r"\[\(0, 6\)\]"):
        validate_rollout_masks(_training(rollout_horizon_schedule=[6, 11],
                                         rollout_loss_stride=5, rollout_burn_in=8))


def test_guard_is_silent_when_nothing_is_set():
    validate_rollout_masks(_training())
    validate_rollout_masks(_training(rollout_horizon_schedule=[1, 2, 3]))


# --------------------------------------------------------------------------- #
#  the production loss                                                         #
# --------------------------------------------------------------------------- #
N, T, DT = 6, 40, 0.02


class _LinearModel(torch.nn.Module):
    """dv/dt = a*v + b*stimulus + c. Three scalars are enough for a chain: a
    scored late step depends on `a` both directly and through every earlier
    integration, and only the second part dies if something detaches."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(-0.7, dtype=torch.float64))
        self.b = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
        self.c = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float64))

    def forward(self, state, edges, data_id=None, return_all=False):
        v = state.voltage.to(torch.float64).unsqueeze(-1)
        s = state.stimulus.to(torch.float64).unsqueeze(-1)
        pred = self.a * v + self.b * s + self.c
        return (pred, None, None) if return_all else pred


class _NoRegularizer:
    """The loss calls these four; returning zero isolates the fit term."""

    def reset_iteration(self, device):
        pass

    def compute(self, **kwargs):
        return torch.zeros((), dtype=torch.float64)

    def sample_g_phi_perm(self, device):
        return None

    def compute_update_regul(self, *args, **kwargs):
        return torch.zeros((), dtype=torch.float64)


def _data(seed=0):
    rng = np.random.RandomState(seed)
    arr = np.zeros((T, N, 9), dtype=np.float32)
    arr[:, :, 0] = np.arange(N)[None, :]
    arr[:, :, 3] = rng.randn(T, N)
    arr[:, :, 4] = rng.rand(T, N)
    return NeuronTimeSeries.from_numpy(arr), torch.tensor(rng.randn(T, N, 1))


def _loss(model, x_ts, y_ts, n_steps, burn_in=0, bptt_window=0, batch=2):
    tc = types.SimpleNamespace(
        batch_size=batch, fit_reduction="norm2", fit_huber_delta=1.0,
        rollout_step_weighting="uniform", rollout_step_reduction="mean",
        rollout_discount=0.9, rollout_bptt_window=bptt_window,
        rollout_shooting_stride=0, rollout_loss_stride=0, rollout_burn_in=burn_in,
        integration_method="euler", n_rollout_substeps=1, noise_recurrent_level=0.0)
    sim = types.SimpleNamespace(n_neurons=N, measurement_noise_level=0.0, delta_t=DT)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    loss, _ = _dense_rollout_loss(
        model, x_ts, y_ts, edges, torch.arange(N), np.array([3, 11]), 0,
        n_steps, sim, tc, "cpu", 1.0, 1.0, _NoRegularizer(), False)
    return loss


def _grad(model, *args, **kw):
    model.zero_grad()
    _loss(model, *args, **kw).backward()
    return torch.stack([p.grad.detach().clone() for p in model.parameters()])


def _finite_difference(model, *args, eps=1e-6, **kw):
    out = []
    with torch.no_grad():
        for p in model.parameters():
            p += eps
            hi = _loss(model, *args, **kw).item()
            p -= 2 * eps
            lo = _loss(model, *args, **kw).item()
            p += eps
            out.append((hi - lo) / (2 * eps))
    return torch.tensor(out, dtype=torch.float64)


def test_burn_in_gradient_matches_a_finite_difference():
    """The loss with burn-in 8 over horizon 20 is differentiated exactly.

    If anything cut the graph at the boundary, autograd would still return
    SOME gradient -- the scored steps' direct term -- and it would disagree
    with the finite difference, which sees the whole chain.
    """
    x_ts, y_ts = _data()
    model = _LinearModel()
    g = _grad(model, x_ts, y_ts, 20, burn_in=8)
    fd = _finite_difference(model, x_ts, y_ts, 20, burn_in=8)
    assert torch.allclose(g, fd, rtol=1e-5, atol=1e-7), (g, fd)


def test_burn_in_does_not_detach_at_the_boundary():
    """Detaching the state after step 7 (rollout_bptt_window 8) changes the
    gradient; the burn-in alone must not. That difference IS the chain term
    the known-ODE test showed a detached burn-in loses."""
    x_ts, y_ts = _data()
    model = _LinearModel()
    kept = _grad(model, x_ts, y_ts, 20, burn_in=8)
    cut = _grad(model, x_ts, y_ts, 20, burn_in=8, bptt_window=8)
    assert not torch.allclose(kept, cut, rtol=1e-3), (kept, cut)


def test_burn_in_targets_do_not_enter_the_loss():
    """Changing the target at a burned-in frame leaves the loss unchanged;
    changing it at a scored frame does not."""
    x_ts, y_ts = _data()
    model = _LinearModel()
    base = _loss(model, x_ts, y_ts, 20, burn_in=8).item()
    # starts are frames 3 and 11, so frame 3 + 5 is burned in and 3 + 12 is scored
    y_burned = y_ts.clone(); y_burned[3 + 5] += 100.0
    y_scored = y_ts.clone(); y_scored[3 + 12] += 100.0
    assert _loss(model, x_ts, y_burned, 20, burn_in=8).item() == pytest.approx(base, rel=1e-12)
    assert _loss(model, x_ts, y_scored, 20, burn_in=8).item() != pytest.approx(base, rel=1e-6)


def test_zero_burn_in_is_the_existing_objective():
    x_ts, y_ts = _data()
    model = _LinearModel()
    tc_default = _loss(model, x_ts, y_ts, 20).item()
    assert _loss(model, x_ts, y_ts, 20, burn_in=0).item() == tc_default


# --------------------------------------------------------------------------- #
#  the warm start: rollout_burn_in_start_epoch                                 #
# --------------------------------------------------------------------------- #
from connectome_gnn.models.recurrent_step import burn_in_at_epoch, recurrent_loss  # noqa: E402


def test_burn_in_is_off_during_the_warm_start():
    tr = _training(rollout_burn_in=8, rollout_burn_in_start_epoch=2)
    assert [burn_in_at_epoch(tr, e) for e in range(5)] == [0, 0, 8, 8, 8]


def test_no_warm_start_means_burn_in_from_epoch_zero():
    tr = _training(rollout_burn_in=8)
    assert burn_in_at_epoch(tr, 0) == 8


def test_guard_lets_warm_up_epochs_use_short_horizons():
    """The cold-start fix: epoch 0 at horizon 9 scores every step, including
    step 0, and only the burned-in epochs have to reach past step 7."""
    validate_rollout_masks(_training(
        n_epochs=20, rollout_horizon_schedule=[1] + list(range(9, 21)),
        rollout_burn_in=8, rollout_burn_in_start_epoch=1))


def test_guard_still_checks_the_burned_in_epochs():
    with pytest.raises(ValueError, match=r"\(1, 5\)"):
        validate_rollout_masks(_training(
            n_epochs=4, rollout_horizon_schedule=[1, 5, 9, 9],
            rollout_burn_in=8, rollout_burn_in_start_epoch=1))


def test_guard_checks_the_padded_tail():
    """The trainer pads a short schedule with its last value; so does the guard."""
    with pytest.raises(ValueError, match=r"\(3, 5\)"):
        validate_rollout_masks(_training(
            n_epochs=4, rollout_horizon_schedule=[9, 5],
            rollout_burn_in=8, rollout_burn_in_start_epoch=2))


def test_guard_refuses_a_warm_start_without_a_burn_in():
    with pytest.raises(ValueError, match="would do nothing"):
        validate_rollout_masks(_training(
            n_epochs=4, rollout_horizon_schedule=[9], rollout_burn_in_start_epoch=1))


def test_guard_refuses_a_warm_start_that_never_ends():
    with pytest.raises(ValueError, match="never apply"):
        validate_rollout_masks(_training(
            n_epochs=4, rollout_horizon_schedule=[9], rollout_burn_in=8,
            rollout_burn_in_start_epoch=4))


def test_an_explicit_burn_in_overrides_the_configured_one():
    """What the trainer does in a warm-up epoch: config says 8, the epoch says 0,
    and the loss must be the dense one."""
    x_ts, y_ts = _data()
    model = _LinearModel()
    dense = _loss(model, x_ts, y_ts, 20, burn_in=0).item()
    tc = types.SimpleNamespace(
        batch_size=2, fit_reduction="norm2", fit_huber_delta=1.0,
        rollout_step_weighting="uniform", rollout_step_reduction="mean",
        rollout_discount=0.9, rollout_bptt_window=0, rollout_shooting_stride=0,
        rollout_loss_stride=0, rollout_burn_in=8, integration_method="euler",
        n_rollout_substeps=1, noise_recurrent_level=0.0)
    sim = types.SimpleNamespace(n_neurons=N, measurement_noise_level=0.0, delta_t=DT)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    args = (model, x_ts, y_ts, edges, torch.arange(N), np.array([3, 11]), 0,
            20, sim, tc, "cpu", 1.0, 1.0, _NoRegularizer(), False)
    warm, _ = _dense_rollout_loss(*args, burn_in=0)
    burned, _ = _dense_rollout_loss(*args)          # None -> the configured 8
    assert warm.item() == pytest.approx(dense, rel=1e-12)
    assert burned.item() == pytest.approx(_loss(model, x_ts, y_ts, 20, burn_in=8).item(), rel=1e-12)
    assert warm.item() != pytest.approx(burned.item(), rel=1e-6)


def test_recurrent_loss_forwards_the_epoch_burn_in():
    """The dispatcher is the one hop between the trainer and the loss."""
    x_ts, y_ts = _data()
    model = _LinearModel()
    tc = types.SimpleNamespace(
        batch_size=2, fit_reduction="norm2", fit_huber_delta=1.0,
        rollout_step_weighting="uniform", rollout_step_reduction="mean",
        rollout_discount=0.9, rollout_bptt_window=0, rollout_shooting_stride=0,
        rollout_loss_stride=0, rollout_burn_in=8, integration_method="euler",
        n_rollout_substeps=1, noise_recurrent_level=0.0)
    sim = types.SimpleNamespace(n_neurons=N, measurement_noise_level=0.0, delta_t=DT)
    config = types.SimpleNamespace(simulation=sim, training=tc)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    kw = dict(model=model, x_ts=x_ts, y_ts=y_ts, edges=edges, ids=torch.arange(N),
              frame_indices=np.array([3, 11]), iter_idx=0, config=config,
              device="cpu", xnorm=1.0, ynorm=1.0, regularizer=_NoRegularizer(),
              n_steps=20)
    warm, _ = recurrent_loss(**kw, burn_in=0)
    assert warm.item() == pytest.approx(_loss(model, x_ts, y_ts, 20, burn_in=0).item(), rel=1e-12)
