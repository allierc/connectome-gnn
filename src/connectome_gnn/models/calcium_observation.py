r"""Latent-calcium forward model (experiment 13): train on the calcium an
indicator reports, not on a voltage recovered from it.

Experiment 12 deconvolved the calcium first and trained on the estimate, so the
deconvolution's errors became the target. Here the dataset's voltage.zarr IS
the calcium recording c, the voltage v is a latent the GNN integrates, and the
loss compares the calcium that latent would produce with the recorded one --
the batch-6 design of Plexus exp17 (operator `calcium_indicator`), with the
GCaMP6f kernel in place of its one-stage indicator.

THE INDICATOR, two leaky stages of unit gain, a rise state r then the calcium:

    r(t+1) = r(t) + k_r (v(t)   - r(t)),     k_r = 1 - exp(-dt / tau_rise)
    c(t+1) = c(t) + k_d (r(t+1) - c(t)),     k_d = 1 - exp(-dt / tau_decay)

Its impulse response is the unit-area difference of exponentials the datasets
were built with (rise 75 ms, decay 400 ms): run on the simulated voltage of
flyvis_unified_blank50_kernel it reproduces that dataset's calcium.zarr to
0.075% of SD(c) (2026-10-02), against a measurement noise of 1% of SD(c) at
40 dB. The order matters: with r updated after c the same check gives 0.95%.
Unit gain means v is in the units of c and no gain can trade against W.

THE LATENT AT THE START of a rollout is not observed. It is read from the H
recorded frames the rollout starts on, by taps shared by every neuron:

    v(k) = sum_j a_j c(k - j),     r(k) = sum_j b_j c(k - j),     j = 0 .. H-1

a and b start at v = r = c(k) (a_0 = b_0 = 1, the rest 0) and are learned by
the rollout loss, which chooses how much to invert the kernel against how much
noise that amplifies. The exact inverse amplifies white noise by about
1 / (k_r k_d) = 87x at dt = 20 ms, so it is not the start.

THE LOSS over a horizon of K frames scores c(k+1) .. c(k+K). c(k+s+1) depends on
v(k+s), which took s GNN steps, so K frames use K-1 GNN calls and the GNN gets
no gradient at K = 1 -- hence every horizon >= 2. Each frame's residual is
divided by dt, so a calcium error reads in V/s like the derivative the one-step
loss scores, and the regulariser coefficients keep their meaning to first order.
"""

import math

import torch
import torch.nn as nn

from connectome_gnn.models.utils import _batch_frames, fit_residual_loss


class CalciumIndicator(nn.Module):
    """The indicator and the start-of-rollout taps. Attached to the model as
    `model.calcium` only when training.observable is 'calcium', so every other
    run builds the same parameters, in the same order, as before."""

    def __init__(self, dt, tau_rise, tau_decay, n_taps=6, learn_kernel=False):
        super().__init__()
        self.dt = float(dt)
        k = torch.tensor([1.0 - math.exp(-dt / tau_rise), 1.0 - math.exp(-dt / tau_decay)])
        rate = torch.log(k / (1.0 - k))              # k = sigmoid(rate) stays in (0, 1)
        if learn_kernel:
            self.rate = nn.Parameter(rate)
        else:
            self.register_buffer("rate", rate)
        taps = torch.zeros(2, int(n_taps))           # row 0 -> v(k), row 1 -> r(k)
        taps[:, 0] = 1.0
        self.taps = nn.Parameter(taps)

    def k(self):
        """(k_r, k_d), the fraction of the gap each stage closes per frame."""
        return torch.sigmoid(self.rate)

    def taus(self):
        """(tau_rise, tau_decay) in seconds, for logging."""
        return -self.dt / torch.log1p(-self.k())

    def lift(self, c_hist):
        """c_hist (H, M), newest frame first -> (v, r), each (M,)."""
        vr = self.taps @ c_hist
        return vr[0], vr[1]

    def step(self, v, r, c):
        """One frame of the indicator: (r(t), c(t)) and v(t) -> (r(t+1), c(t+1))."""
        k_r, k_d = self.k()
        r = r + k_r * (v - r)
        c = c + k_d * (r - c)
        return r, c


def attach_calcium_indicator(model, config):
    """Give the model its indicator when the run observes calcium; no-op otherwise."""
    tc = config.training
    if getattr(tc, "observable", "voltage") != "calcium":
        return
    model.calcium = CalciumIndicator(
        config.simulation.delta_t, tc.calcium_tau_rise, tc.calcium_tau_decay,
        n_taps=tc.calcium_n_taps, learn_kernel=tc.calcium_kernel_learned,
    ).to(next(model.parameters()).device)


def calcium_history(voltage, k, n_taps):
    """Recorded frames c(k), c(k-1), ..., c(k-H+1) as (H, N); frames before 0
    repeat frame 0 (the test rollout starts at k = 0)."""
    idx = [max(k - j, 0) for j in range(n_taps)]
    return voltage[idx]


def latent_start(model, x_ts, k):
    """The latent voltage at frame k read from the recording, for the test
    rollout. Returns None when the model has no indicator."""
    ind = getattr(model, "calcium", None)
    if ind is None:
        return None
    v, _ = ind.lift(calcium_history(x_ts.voltage, k, ind.taps.shape[1]))
    return v


def calcium_rollout_loss(
    model, x_ts, edges, ids, frame_indices, iter_idx,
    n_steps, sim, tc, device, xnorm, regularizer, has_visual_field, hn=None,
):
    """The dense rollout of recurrent_step._dense_rollout_loss, scored on
    calcium. Returns (loss including regularisation, regularisation value)."""
    if n_steps is None or int(n_steps) < 2:
        raise ValueError("observable: calcium needs rollout_horizon_schedule with every "
                         f"horizon >= 2 (got {n_steps}): the GNN reaches c(k+s+1) only "
                         "through s steps, so a horizon of 1 trains nothing")
    if has_visual_field or (hn is not None and hn.has_hidden):
        raise NotImplementedError("observable: calcium supports neither a learned visual "
                                  "field nor hidden neurons")
    if getattr(tc, "integration_method", "euler") != "euler":
        raise NotImplementedError("observable: calcium integrates with euler only")
    ind = model.calcium
    n_steps = int(n_steps)
    n_taps = ind.taps.shape[1]
    dt = sim.delta_t

    state_batch, ids_list, k_list = [], [], []
    ids_index = 0
    for b in range(tc.batch_size):
        k = int(frame_indices[iter_idx * tc.batch_size + b])
        # c(k-H+1) .. c(k+K) must all be recorded.
        if k < n_taps - 1 or k + n_steps >= x_ts.n_frames:
            continue
        x = x_ts.frame(k)
        if torch.isnan(x_ts.voltage[k - n_taps + 1:k + n_steps + 1]).any():
            continue
        state_batch.append(x)
        ids_list.append(ids + ids_index)
        k_list.append(k)
        ids_index += x.n_neurons
    if not state_batch:
        return torch.zeros(1, device=device, requires_grad=True), 0.0

    ids_batch = torch.cat(ids_list, dim=0)
    data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)
    n_per = state_batch[0].n_neurons

    regularizer.reset_iteration(device=device)
    regul_loss = regularizer.compute(
        model=model, x=state_batch[0], in_features=None,
        ids=ids, ids_batch=None, edges=edges, device=device, xnorm=xnorm,
        perm_indices=regularizer.sample_g_phi_perm(device),
    )
    loss = regul_loss.clone()
    regul_value = regul_loss.item()

    reduction = getattr(tc, "fit_reduction", "norm2")
    huber_delta = getattr(tc, "fit_huber_delta", 1.0)

    def recorded(s):
        """c(k+s) for every sample, (B*N,)."""
        return torch.cat([x_ts.voltage[kb + s] for kb in k_list], dim=0)

    hist = torch.cat([calcium_history(x_ts.voltage, kb, n_taps) for kb in k_list], dim=1)
    v, r = ind.lift(hist)
    c = recorded(0)
    fit_loss = torch.zeros((), device=device)
    for s in range(n_steps):
        if s > 0:
            # v(k+s) = v(k+s-1) + dt f(v(k+s-1)), with the stimulus of frame k+s-1.
            for b_idx, st in enumerate(state_batch):
                st.voltage = v[b_idx * n_per:(b_idx + 1) * n_per]
                st.stimulus = x_ts.stimulus[k_list[b_idx] + s - 1]
            bs, be = _batch_frames(state_batch, edges)
            pred, in_features, _ = model(bs, be, data_id=data_id, return_all=True)
            if s == 1:
                loss = loss + regularizer.compute_update_regul(model, in_features, ids_batch, device)
            v = v + dt * pred.squeeze(-1)
            if tc.noise_recurrent_level > 0:
                v = v + tc.noise_recurrent_level * torch.randn_like(v)
        r, c = ind.step(v, r, c)                       # c(k+s+1), from v(k+s)
        target = recorded(s + 1)
        fit_loss = fit_loss + fit_residual_loss(
            ((c - target) / dt)[ids_batch], reduction,
            target=(target / dt)[ids_batch], huber_delta=huber_delta)
    return loss + fit_loss / n_steps, regul_value
