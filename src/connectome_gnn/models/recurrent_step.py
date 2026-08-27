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

4. **Dense-supervision horizon curriculum** (``rollout_horizon_schedule``):
   Epoch e unrolls ``rollout_horizon_schedule[e]`` steps from frame k and
   scores EVERY intermediate state against the observed voltage, with the
   real stimulus advanced at each step. Requires ``time_step: 1`` so the
   dataset is not decimated and every intermediate frame exists. This is
   the scheme both trainers that work in this repo use — the task trainer
   ``_data_train_task_pi`` (per-frame loss over a horizon grown by
   ``n_steps_schedule``) and the oculomotor prototype ``train_eyeG.py``
   (per-frame loss over a horizon grown by its ``sched`` list).
   Config: ``recurrent_training: true, time_step: 1,
   rollout_horizon_schedule: [1, 2, 3, ...]``

Modes 1, 2 and 4 are implemented in this module. Mode 3 is a sampling
change in graph_trainer.py (no dedicated function needed).

Note on 1 vs 4: mode 1 was built for the stride-5 regime, where the
observable is given only every ``time_step`` steps, so it can only score
the endpoint. Mode 4 is for the dense regime — observable at every step,
longer and longer trajectories.
"""

import torch

from connectome_gnn.models.utils import _batch_frames, fit_residual_loss


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
    hn=None,
    n_steps=None,
):
    """Compute one training iteration of recurrent (possibly multi-start) loss.

    hn: HiddenNeuronHandler (see training_utils.py), or None if the model
    has no hidden neurons.

    n_steps: when given (the per-epoch rollout-horizon curriculum, mode 4 below),
    unroll this many steps with DENSE supervision on an unstrided dataset instead
    of the legacy endpoint-only stride-subsampled scheme. None = legacy behaviour.

    Returns:
        loss: scalar tensor (already includes regularisation)
        regul_value: float, regularisation component for logging
    """
    sim = config.simulation
    tc = config.training
    time_step = tc.time_step
    n_neurons = sim.n_neurons
    multi_start = tc.multi_start_recurrent

    if n_steps is not None:
        return _dense_rollout_loss(
            model, x_ts, y_ts, edges, ids, frame_indices, iter_idx,
            int(n_steps), sim, tc, device, xnorm, ynorm, regularizer, has_visual_field,
            hn=hn,
        )
    elif multi_start:
        return _multi_start_loss(
            model, x_ts, edges, ids, frame_indices, iter_idx,
            time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
            hn=hn,
        )
    else:
        return _standard_recurrent_loss(
            model, x_ts, edges, ids, frame_indices, iter_idx,
            time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
            hn=hn,
        )


# ------------------------------------------------------------------ #
#  4. Dense-supervision rollout with a per-epoch horizon curriculum   #
# ------------------------------------------------------------------ #

def _dense_rollout_loss(
    model, x_ts, y_ts, edges, ids, frame_indices, iter_idx,
    n_steps, sim, tc, device, xnorm, ynorm, regularizer, has_visual_field,
    hn=None,
):
    """Unroll n_steps from frame k, supervising EVERY step against the observed
    dynamics — the scheme both successful trainers in this repo use.

    Three deliberate departures from _standard_recurrent_loss, each chosen to
    match _data_train_task_pi (graph_trainer.py) and prototype train_eyeG.py:

    1. DENSE supervision. The legacy path scores one state per rollout
       (recurrent_step.py:201), leaving the intermediate trajectory completely
       unconstrained — the model can take a wrong path that happens to land near
       the endpoint. Here every intermediate state is scored (see 3), giving
       n_steps times more constraints per unit compute.
       Requires observations at every step, i.e. an UNSTRIDED dataset
       (time_step == 1, so init_training_data's stride is 1).

    2. LIVE stimulus. The legacy path freezes the stimulus for the whole unroll
       ("intermediate frames not available" — they were decimated away by the
       stride). At stride 1 they exist, so the exogenous drive is advanced every step,
       removing a systematically-signed residual no parameter setting could absorb.

    3. IDENTICAL TO THE NOMINAL OBJECTIVE AT n_steps=1. Each step scores the
       model's DERIVATIVE against the precomputed derivative target
       `y_ts[k+s] / ynorm`, reduced with `.norm(2)` — exactly the nominal path's
       term (run_nominal_train_step). Averaged over steps, so n_steps=1 reduces
       to nominal *term for term* and the curriculum is a strict extension.

       This matters more than it looks. An earlier version scored the integrated
       voltage against `x_ts.voltage[k+s]` with `.pow(2).mean()` and no ynorm.
       `.norm(2)` is sqrt(sum r^2) over ~55k elements while `.mean()` is mean(r^2),
       a 3-5 ORDER OF MAGNITUDE difference at the same residual — and since the
       regularisers are unchanged parameter norms, the fit/regulariser balance
       flipped (prediction fell from ~80% of the loss to ~12%). The W penalties
       then dominated and drove R^2_W to ~0 while nominal reached ~0.98. Any
       change to this reduction must keep n_steps=1 equal to nominal.
    """
    batch_size = tc.batch_size
    n_neurons = sim.n_neurons

    state_batch = []
    ids_list = []
    k_list = []
    ids_index = 0

    for b in range(batch_size):
        k = int(frame_indices[iter_idx * batch_size + b])

        # Need observations at k+1 .. k+n_steps.
        if k + n_steps >= x_ts.n_frames:
            continue

        x = x_ts.frame(k)
        if x.noise is not None and sim.measurement_noise_level > 0:
            x.voltage = x.voltage + x.noise
        if hn is not None:
            hn.inject_hidden(model, x, k, True)

        if has_visual_field:
            visual_input = model.forward_visual(x, k)
            x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
            x.stimulus[model.n_input_neurons:] = 0

        if torch.isnan(x.voltage).any():
            continue

        state_batch.append(x)
        ids_list.append(ids + ids_index)
        k_list.append(k)
        ids_index += x.n_neurons

    if not state_batch:
        return torch.zeros(1, device=device, requires_grad=True), 0.0

    ids_batch = torch.cat(ids_list, dim=0)
    data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)

    # Regularisation (computed once on the initial state, as in the legacy path)
    regularizer.reset_iteration(device=device)
    regul_loss = regularizer.compute(
        model=model, x=state_batch[0], in_features=None,
        ids=ids, ids_batch=None, edges=edges, device=device, xnorm=xnorm,
    )
    loss = regul_loss.clone()
    regul_value = regul_loss.item()

    neurons_per_sample = state_batch[0].n_neurons
    fit_loss = torch.zeros((), device=device)
    n_scored = 0

    batched_state, batched_edges = _batch_frames(state_batch, edges)
    pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
    loss = loss + update_regul

    for step in range(n_steps):
        # --- score the DERIVATIVE at the current state, exactly as the nominal
        # path does: target y_ts[k+step] / ynorm, reduced with .norm(2). At
        # step 0 the state is the observed v(k), so n_steps=1 is term-for-term
        # the nominal objective. ---
        # No unsqueeze: y_ts rows already carry the trailing dim that `pred` has,
        # exactly as the nominal path uses `y_ts_gpu[k] / ynorm` directly. Adding
        # one made the residual broadcast to (N*B, N*B, 1) — an 11 GiB square
        # matrix whose norm is not the loss at all, which both OOMed and drove
        # R^2_W negative.
        gt_parts = [
            y_ts[k_list[b_idx] + step] / ynorm
            for b_idx in range(len(state_batch))
        ]
        y_step = torch.cat(gt_parts, dim=0)
        if not torch.isnan(y_step).any():
            fit_loss = fit_loss + fit_residual_loss(
                pred[ids_batch] - y_step[ids_batch],
                getattr(tc, "fit_reduction", "norm2"),
            )
            n_scored += 1

        if step == n_steps - 1:
            break

        # --- integrate one step to get the next state ---
        if step == 0:
            pred_x = (batched_state.voltage.unsqueeze(-1) + sim.delta_t * pred
                      + tc.noise_recurrent_level * torch.randn_like(pred))
        else:
            pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

        # --- advance the state, with the LIVE stimulus for the next frame ---
        for b_idx in range(len(state_batch)):
            s, e = b_idx * neurons_per_sample, (b_idx + 1) * neurons_per_sample
            state_batch[b_idx].voltage = pred_x[s:e].squeeze()
            k_current = k_list[b_idx] + step + 1
            if hn is not None:
                hn.inject_hidden(model, state_batch[b_idx], k_current, True)
            if has_visual_field:
                vi = model.forward_visual(state_batch[b_idx], k_current)
                state_batch[b_idx].stimulus[:model.n_input_neurons] = vi.squeeze(-1)
                state_batch[b_idx].stimulus[model.n_input_neurons:] = 0
            else:
                # stride == 1, so the intermediate stimulus frame exists
                state_batch[b_idx].stimulus = x_ts.stimulus[k_current]

        batched_state, batched_edges = _batch_frames(state_batch, edges)
        pred, _, _ = model(batched_state, batched_edges, data_id=data_id, return_all=True)


    if n_scored == 0:
        return torch.zeros(1, device=device, requires_grad=True), regul_value

    # Average over scored steps -> objective scale is horizon-independent.
    loss = loss + fit_loss / n_scored
    return loss, regul_value


# ------------------------------------------------------------------ #
#  Standard recurrent: single start, unroll time_step forward         #
# ------------------------------------------------------------------ #

def _standard_recurrent_loss(
    model, x_ts, edges, ids, frame_indices, iter_idx,
    time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
    hn=None,
):
    batch_size = tc.batch_size
    n_neurons = sim.n_neurons
    data_id = torch.zeros((n_neurons * batch_size, 1), dtype=torch.int, device=device)

    state_batch = []
    y_list = []
    ids_list = []
    hidden_ids_list = []
    k_list = []
    ids_index = 0

    coeff_hidden = getattr(tc, 'coeff_hidden_voltage', 0.0)
    use_hidden_loss = (coeff_hidden > 0.0) and hn is not None and hn.has_hidden

    for b in range(batch_size):
        k = int(frame_indices[iter_idx * batch_size + b])

        x = x_ts.frame(k)
        if x.noise is not None and sim.measurement_noise_level > 0:
            x.voltage = x.voltage + x.noise
        if hn is not None:
            hn.inject_hidden(model, x, k, True)

        if has_visual_field:
            visual_input = model.forward_visual(x, k)
            x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
            x.stimulus[model.n_input_neurons:] = 0

        if torch.isnan(x.voltage).any():
            continue

        y = x_ts.voltage[k + (1 if time_step > 1 else time_step)].unsqueeze(-1)
        if torch.isnan(y).any():
            continue

        state_batch.append(x)
        y_list.append(y)
        ids_list.append(ids + ids_index)
        if use_hidden_loss:
            hidden_ids_list.append(hn.hidden_ids + ids_index)
        k_list.append(k)
        ids_index += x.n_neurons

    if not state_batch:
        return torch.zeros(1, device=device, requires_grad=True), 0.0

    y_batch = torch.cat(y_list, dim=0)
    ids_batch = torch.cat(ids_list, dim=0)
    hidden_ids_batch = torch.cat(hidden_ids_list, dim=0) if use_hidden_loss else None
    data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)

    # Regularisation (computed once on initial state)
    regularizer.reset_iteration(device=device)
    regul_loss = regularizer.compute(
        model=model, x=state_batch[0], in_features=None,
        ids=ids, ids_batch=None, edges=edges, device=device, xnorm=xnorm,
    )
    loss = regul_loss.clone()
    regul_value = regul_loss.item()

    # Forward pass + unroll
    batched_state, batched_edges = _batch_frames(state_batch, edges)
    pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
    loss = loss + update_regul

    pred_x = batched_state.voltage.unsqueeze(-1) + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

    for step in range(time_step - 1):
        # Hidden neuron loss at this intermediate step (before overwriting with NGP)
        # Gradient path: loss → pred_x[hidden] → v_hidden(k) via -v/tau term → NGP(k)
        if use_hidden_loss:
            neurons_per_sample = state_batch[0].n_neurons
            gt_hidden_parts = []
            for b_idx in range(len(state_batch)):
                k_h = k_list[b_idx] + step + 1
                if k_h < x_ts.n_frames:
                    gt_hidden_parts.append(x_ts.voltage[k_h, hn.hidden_ids].unsqueeze(-1))
            if gt_hidden_parts:
                gt_hidden_batch = torch.cat(gt_hidden_parts, dim=0)
                loss = loss + coeff_hidden * (pred_x[hidden_ids_batch] - gt_hidden_batch).norm(2)

        neurons_per_sample = state_batch[0].n_neurons
        for b_idx in range(len(state_batch)):
            s, e = b_idx * neurons_per_sample, (b_idx + 1) * neurons_per_sample
            state_batch[b_idx].voltage = pred_x[s:e].squeeze()
            if hn is not None:
                k_current_h = k_list[b_idx] + step + 1
                hn.inject_hidden(model, state_batch[b_idx], k_current_h, True)
            k_current = k_list[b_idx] + step + 1
            if has_visual_field:
                vi = model.forward_visual(state_batch[b_idx], k_current)
                state_batch[b_idx].stimulus[:model.n_input_neurons] = vi.squeeze(-1)
                state_batch[b_idx].stimulus[model.n_input_neurons:] = 0
            else:
                pass  # stimulus held constant during unroll (subsampled x_ts; intermediate frames not available)

        batched_state, batched_edges = _batch_frames(state_batch, edges)
        pred, _, _ = model(batched_state, batched_edges, data_id=data_id, return_all=True)
        pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

    loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (sim.delta_t * time_step)).norm(2)
    return loss, regul_value


# ------------------------------------------------------------------ #
#  Multi-start recurrent: time_step starts all targeting frame T      #
# ------------------------------------------------------------------ #

def _multi_start_loss(
    model, x_ts, edges, ids, frame_indices, iter_idx,
    time_step, sim, tc, device, xnorm, regularizer, has_visual_field,
    hn=None,
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
        if hn is not None:
            hn.inject_hidden(model, x, start_k, True)

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
            if hn is not None:
                k_cur = start_k + step + 1
                hn.inject_hidden(model, x, k_cur, True)

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
