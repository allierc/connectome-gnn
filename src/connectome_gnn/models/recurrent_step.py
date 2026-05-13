"""Recurrent multi-step training loss for GNN.

Overview of recurrent / noise-aware training strategies
-------------------------------------------------------
All strategies load a pretrained one-step model and fine-tune it.
The goal is to improve robustness to observation noise (process +
measurement) without sacrificing connectivity R².

1. **Standard recurrent** (``recurrent_training=True``):
   Pick one random frame k, unroll time_step forward using the model's
   own predictions, compare predicted voltage at k+time_step to the
   observed (noisy) target. Forces the model to be self-consistent
   over multiple steps, but gradient flows through a long noisy chain.
   Config: ``recurrent_training: true, time_step: N``

2. **Multi-start recurrent** (``multi_start_recurrent=True``):
   For a target frame T, launch time_step parallel rollouts from
   T-time_step, T-time_step+1, ..., T-1 (lengths time_step down to 1).
   All predictions target the same observed v(T). Each start has
   independent noise on its initial voltage, so gradient noise from
   different starts partially cancels. Short paths (1-step) anchor the
   gradient while long paths enforce trajectory consistency.
   Config: ``recurrent_training: true, multi_start_recurrent: true, time_step: N``

3. **Consecutive batch** (``consecutive_batch=True``):
   Instead of sampling batch_size random frames, pick one random start k
   and use frames k, k+1, ..., k+batch_size-1. Each frame gets a
   standard one-step prediction (no unrolling). Consecutive frames share
   the same local dynamics but have independent noise realisations, so
   the gradient over the batch naturally averages out noise. Simplest
   approach: no extra memory, no multi-step backprop, just a sampling
   change.
   Config: ``consecutive_batch: true, batch_size: N``
   (no recurrent_training needed)

Modes 1 and 2 are implemented in this module. Mode 3 is a sampling
change in graph_trainer.py (no dedicated function needed).
"""

from dataclasses import fields as dc_fields

import torch

from connectome_gnn.models.utils import _batch_frames
from connectome_gnn.neuron_state import DYNAMIC_FIELDS, NeuronState


def recurrent_loss(
    model,
    x_ts,
    y_ts,
    edges,
    ids,
    frame_indices,
    iter_idx,
    config,
    device,
    xnorm,
    ynorm,
    regularizer,
    has_visual_field=False,
    hidden_ids=None,
):
    """Compute one training iteration of recurrent (possibly multi-start) loss.

    Returns:
        loss: scalar tensor (already includes regularisation)
        regul_value: float, regularisation component for logging
    """
    sim = config.simulation
    tc = config.training
    time_step = tc.time_step
    n_neurons = sim.n_neurons
    multi_start = tc.multi_start_recurrent

    if multi_start:
        return _multi_start_loss(
            model, x_ts, edges, ids, frame_indices, iter_idx,
            time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
            hidden_ids=hidden_ids,
        )
    else:
        return _standard_recurrent_loss(
            model, x_ts, edges, ids, frame_indices, iter_idx,
            time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
            hidden_ids=hidden_ids,
        )


# ------------------------------------------------------------------ #
#  Standard recurrent: single start, unroll time_step forward         #
# ------------------------------------------------------------------ #

def _standard_recurrent_loss(
    model, x_ts, edges, ids, frame_indices, iter_idx,
    time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
    hidden_ids=None,
):
    """Vectorized standard recurrent rollout loss.

    Behaviour matches the previous batch-loop version:
        - voltage target = x_ts.voltage[k + 1] (subsampled or not)
        - per-step Gaussian noise injection (tc.noise_recurrent_level)
        - optional per-step hidden-voltage MSE (coeff_hidden_voltage)
        - regularizer.compute/compute_update_regul integration
    The Python loop over batch elements is replaced by gathers on a (B,)
    frame-index tensor, and `_batch_frames` is called only implicitly when
    the initial state is built — the unroll mutates `batched_state` in
    place via batched indexing.
    """
    B = tc.batch_size
    N = sim.n_neurons
    coeff_hidden = getattr(tc, 'coeff_hidden_voltage', 0.0)
    use_hidden_loss = (coeff_hidden > 0.0) and (hidden_ids is not None)

    # Per-sample start frame (B,)
    k_per_sample = torch.as_tensor(
        frame_indices[iter_idx * B : iter_idx * B + B],
        device=device, dtype=torch.long,
    )

    # Batch-flat helper indices
    _b_off_N = (torch.arange(B, device=device) * N).view(-1, 1)         # (B, 1)
    ids_batch = (_b_off_N + ids.view(1, -1)).reshape(-1)                # (B*|ids|,)
    if use_hidden_loss:
        hidden_ids_batch = (_b_off_N + hidden_ids.view(1, -1)).reshape(-1)
    else:
        hidden_ids_batch = None

    data_id = torch.zeros((B * N, 1), dtype=torch.int, device=device)

    # Build batched_state with the same layout _batch_frames produces:
    # dynamic fields are (B*N,) — frame-gathered then flattened; static
    # fields are tiled B times. No Python loop over b.
    def _tile_static(val):
        if val is None:
            return None
        return val.repeat(B) if val.dim() == 1 else val.repeat(B, *([1] * (val.dim() - 1)))

    kwargs = {}
    for f in dc_fields(NeuronState):
        val_ts = getattr(x_ts, f.name, None)
        if val_ts is None:
            kwargs[f.name] = None
        elif f.name in DYNAMIC_FIELDS:
            kwargs[f.name] = val_ts[k_per_sample].reshape(-1).clone()
        else:
            kwargs[f.name] = _tile_static(val_ts)
    batched_state = NeuronState(**kwargs)
    batched_edges = torch.cat([edges + i * N for i in range(B)], dim=1)

    # Measurement noise on the observed voltage (now batched)
    if batched_state.noise is not None and sim.measurement_noise_level > 0:
        batched_state.voltage = batched_state.voltage + batched_state.noise

    # Hidden-neuron injection at the initial frame
    if hidden_ids is not None:
        _hidden_flat = (_b_off_N + hidden_ids.view(1, -1)).reshape(-1)
        if model.NNR_hidden is not None:
            # Batched query (only ngp_t / siren_t variants support this;
            # siren_txy is not exercised by current configs).
            hidden_pred = model.forward_hidden_batched(k_per_sample, hidden_ids=hidden_ids)
            batched_state.voltage[_hidden_flat] = hidden_pred.reshape(-1)
        else:
            batched_state.voltage[_hidden_flat] = 0.0

    # Visual field at the initial frame (B is small; per-b inner loop)
    if has_visual_field:
        n_input = model.n_input_neurons
        stim_buf = torch.zeros(B, N, device=device, dtype=batched_state.stimulus.dtype)
        for b in range(B):
            k_b = int(k_per_sample[b].item())
            x_b = x_ts.frame(k_b)
            vi = model.forward_visual(x_b, k_b)
            stim_buf[b, :n_input] = vi.squeeze(-1)
        batched_state.stimulus = stim_buf.flatten()

    # Regularisation — runs once on the initial state. compute() only
    # reads from `model`, not from `x`, so passing batched_state is fine.
    regularizer.reset_iteration(device=device)
    regul_loss = regularizer.compute(
        model=model, x=batched_state, in_features=None,
        ids=ids, ids_batch=None, edges=edges, device=device, xnorm=xnorm,
    )
    loss = regul_loss.clone()
    regul_value = regul_loss.item()

    # First forward + Euler step (k → k+1)
    pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)
    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
    loss = loss + update_regul
    pred_x = (
        batched_state.voltage.unsqueeze(-1)
        + sim.delta_t * pred
        + tc.noise_recurrent_level * torch.randn_like(pred)
    )

    # Per-step loss accumulator (visible neurons). After each Euler step
    # pred_x is at frame k + s + 1; compare against voltage[k + s + 1].
    target = x_ts.voltage[k_per_sample + 1].reshape(-1, 1)              # (B*N, 1)
    loss_steps = (pred_x[ids_batch] - target[ids_batch]).norm(2)
    if use_hidden_loss:
        gt_hidden = x_ts.voltage[k_per_sample + 1][:, hidden_ids].reshape(-1, 1)
        loss = loss + coeff_hidden * (pred_x[hidden_ids_batch] - gt_hidden).norm(2)

    # Unroll: each iteration advances pred_x from k+step+1 to k+step+2.
    for step in range(time_step - 1):
        # Roll observable forward — flat assignment, no _batch_frames re-pack.
        batched_state.voltage = pred_x.squeeze(-1)
        if hidden_ids is not None:
            k_now = k_per_sample + step + 1
            if model.NNR_hidden is not None:
                hidden_pred = model.forward_hidden_batched(k_now, hidden_ids=hidden_ids)
                batched_state.voltage[hidden_ids_batch] = hidden_pred.reshape(-1)
            else:
                batched_state.voltage[hidden_ids_batch] = 0.0

        # Update stimulus to the current frame (now that x_ts is full-length,
        # intermediate-frame stimuli are available).
        k_now = k_per_sample + step + 1
        if has_visual_field:
            n_input = model.n_input_neurons
            stim_buf = torch.zeros(B, N, device=device, dtype=batched_state.stimulus.dtype)
            for b in range(B):
                k_b = int(k_now[b].item())
                x_b = x_ts.frame(k_b)
                vi = model.forward_visual(x_b, k_b)
                stim_buf[b, :n_input] = vi.squeeze(-1)
            batched_state.stimulus = stim_buf.flatten()
        else:
            batched_state.stimulus = x_ts.stimulus[k_now].reshape(-1)
            if x_ts.stimulus_calcium is not None:
                batched_state.stimulus_calcium = x_ts.stimulus_calcium[k_now].reshape(-1)
            if x_ts.optogenetics_stimulus is not None:
                batched_state.optogenetics_stimulus = x_ts.optogenetics_stimulus[k_now].reshape(-1)

        # Forward + Euler step → pred_x now at frame k + step + 2.
        pred, _, _ = model(batched_state, batched_edges, data_id=data_id, return_all=True)
        pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

        # Per-step loss against voltage[k + step + 2].
        target = x_ts.voltage[k_per_sample + step + 2].reshape(-1, 1)
        loss_steps = loss_steps + (pred_x[ids_batch] - target[ids_batch]).norm(2)
        if use_hidden_loss:
            gt_hidden = x_ts.voltage[k_per_sample + step + 2][:, hidden_ids].reshape(-1, 1)
            loss = loss + coeff_hidden * (pred_x[hidden_ids_batch] - gt_hidden).norm(2)

    # Average per-step contributions; (dt * time_step) keeps the magnitude
    # comparable to the previous endpoint-only formulation.
    loss = loss + (loss_steps / time_step) / (sim.delta_t * time_step)
    return loss, regul_value


# ------------------------------------------------------------------ #
#  Multi-start recurrent: time_step starts all targeting frame T      #
# ------------------------------------------------------------------ #

def _multi_start_loss(
    model, x_ts, edges, ids, frame_indices, iter_idx,
    time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
    hidden_ids=None,
):
    """Launch time_step rollouts of decreasing length, all targeting frame T.

    Start frames: T - time_step, T - time_step + 1, ..., T - 1
    Rollout lengths: time_step, time_step - 1, ..., 1
    Target: observed v(T) for all.
    """
    n_neurons = sim.n_neurons

    # Pick target frame T (one per iteration, use first frame index)
    k_raw = int(frame_indices[iter_idx * time_step])  # batch_size == time_step
    T = max(time_step, k_raw)  # ensure we have enough history
    T = min(T, x_ts.n_frames - 1)  # stay in bounds

    # Target voltage at T (same for all starts)
    y_target = x_ts.voltage[T].unsqueeze(-1)
    if torch.isnan(y_target).any():
        return torch.zeros(1, device=device, requires_grad=True), 0.0

    # Regularisation (compute once)
    x0 = x_ts.frame(T - time_step)
    if x0.noise is not None and sim.measurement_noise_level > 0:
        x0.voltage = x0.voltage + x0.noise
    regularizer.reset_iteration(device=device)
    regul_loss = regularizer.compute(
        model=model, x=x0, in_features=None,
        ids=ids, ids_batch=None, edges=edges, device=device, xnorm=xnorm,
    )
    regul_value = regul_loss.item()
    loss = regul_loss.clone()

    # Launch each start independently
    for s in range(time_step):
        start_k = T - time_step + s  # start frame
        n_steps = time_step - s       # rollout length

        x = x_ts.frame(start_k)
        if x.noise is not None and sim.measurement_noise_level > 0:
            x.voltage = x.voltage + x.noise
        if hidden_ids is not None:
            if model.NNR_hidden is not None:
                x.voltage[hidden_ids] = model.forward_hidden(x, start_k, hidden_ids)
            else:
                x.voltage[hidden_ids] = 0.0

        if torch.isnan(x.voltage).any():
            continue

        if has_visual_field:
            vi = model.forward_visual(x, start_k)
            x.stimulus[:model.n_input_neurons] = vi.squeeze(-1)
            x.stimulus[model.n_input_neurons:] = 0

        data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

        # Unroll n_steps forward
        for step in range(n_steps):
            batched_state, batched_edges = _batch_frames([x], edges)
            pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

            if s == 0 and step == 0:
                update_regul = regularizer.compute_update_regul(model, in_features, ids, device)
                loss = loss + update_regul

            x.voltage = (x.voltage.unsqueeze(-1) + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)).squeeze(-1)
            if hidden_ids is not None:
                k_cur = start_k + step + 1
                if model.NNR_hidden is not None:
                    x.voltage[hidden_ids] = model.forward_hidden(x, k_cur, hidden_ids)
                else:
                    x.voltage[hidden_ids] = 0.0

            # Update stimulus for next step
            k_next = start_k + step + 1
            if k_next < x_ts.n_frames:
                if has_visual_field:
                    vi = model.forward_visual(x, k_next)
                    x.stimulus[:model.n_input_neurons] = vi.squeeze(-1)
                    x.stimulus[model.n_input_neurons:] = 0
                else:
                    pass  # stimulus held constant during unroll (subsampled x_ts; intermediate frames not available)

        # Loss: predicted voltage vs target at T
        pred_v = x.voltage.unsqueeze(-1)
        loss = loss + ((pred_v[ids] - y_target[ids]) / (sim.delta_t * time_step)).norm(2)

    # Average over the time_step starts
    loss = loss / time_step
    return loss, regul_value
