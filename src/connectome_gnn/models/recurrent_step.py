r"""Recurrent (multi-step) training losses.

THREE MODES, one dispatcher. `recurrent_loss` picks between them:

    n_steps is not None      -> _dense_rollout_loss     ("ROLLOUT", below)
    multi_start_recurrent    -> _multi_start_loss       (untested)
    otherwise                -> _standard_recurrent_loss (legacy, untested)

A fourth strategy, `consecutive_batch`, is a *sampling* change and lives in
training_utils.py, not here. One-step (t+1) training does not enter this module
at all -- it goes through run_nominal_train_step.


ROLLOUT  (`_dense_rollout_loss`)
--------------------------------
Activated by `rollout_horizon_schedule: [1, 2, 3, ...]`; epoch e unrolls K =
schedule[e] steps from a sampled frame k and scores EVERY step. The only mode
that has been benchmarked. 

Requires `time_step: 1` (unstrided data, so every intermediate frame exists) and
advances the real stimulus at each step.

Each step scores the DERIVATIVE against `y_ts[k+s] / ynorm`. At K=1 this is
term-for-term the one-step objective, so the curriculum is a strict extension of
it -- and every knob below is a no-op at K=1, which keeps that equality true
whatever they are set to.

TWO REDUCTIONS, do not confuse them. The loss is

    loss = REDUCE_s  w_s * fit_reduction( pred_s - y_{k+s} )   over s = 0..K-1
                            \____________________________/
                             collapses ONE step over n_visible * batch_size

  * `fit_reduction` (shared with one-step training) collapses a single step's
    residual over neurons AND batch. 'norm2' = ||r||_2 grows as sqrt(batch_size);
    `regul_batch_scaling: sqrt` is what cancels that, so the regulariser/fit ratio
    no longer depends on the batch size. 'mean' removes the coupling at the source
    instead, and the two are mutually exclusive (config.py rejects the pair).
  * `rollout_step_reduction` collapses the K steps: 'mean' (default) divides by
    sum_s w_s, 'sum' does not. Only this one is horizon-related; it has nothing to
    do with batch size.

    KNOB                        VALUE          BENCHMARK ARM
    (default = plain rollout)                  "uniform"
    rollout_step_weighting      "discount"     "discount"   w_s = gamma^s
                                "last"         "last"       only the final step
    rollout_step_reduction      "sum"          --           no 1/sum_s w_s
    rollout_bptt_window         1              "pushforward"  detach every step
    rollout_shooting_stride     1              "shoot1"     re-anchor every step
                                2              "shoot2"     re-anchor every 2nd

MODE 1  (`_standard_recurrent_loss`, the default when neither of the above is set)
---------------------------------------------------------------------------------
Legacy. One start, unroll `time_step`, score ONLY the endpoint, on a
stride-subsampled dataset with the stimulus frozen through the unroll. Built for
the stride-5 regime where intermediate observations do not exist.
NOT YET BENCHMARKED -- note this is NOT the same as the "last" arm above, which is
endpoint-only on the dense unstrided grid with a live stimulus.

MODE 2  (`_multi_start_loss`, `multi_start_recurrent: true`)
-----------------------------------------------------------
For a target frame T, launch `time_step` rollouts from T-time_step ... T-1, all
predicting the same observed v(T). Short paths anchor the gradient, long ones
enforce trajectory consistency, and independent start noise partially cancels.
NOT YET BENCHMARKED.


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
    target_weight=None,
):
    """Dispatch to one of the three modes. See the module docstring.

    n_steps: the epoch's rollout horizon K. Given -> ROLLOUT; None -> mode 1 or 2.
    hn: HiddenNeuronHandler, or None if the model has no hidden neurons.

    Returns (loss including regularisation, regularisation value for logging).
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
            hn=hn, target_weight=target_weight,
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
#  ROLLOUT: unroll K steps, score every one                          #
# ------------------------------------------------------------------ #

def _rollout_step_weights(weighting, n_steps, gamma):
    """Per-step weights: "uniform" | "discount" | "linear_decay" | "last".

    Unnormalised -- the caller divides by the weight actually applied, so any
    positive scaling is equivalent. All schemes return [1.0] at K=1, which is what
    keeps the K=1 objective identical to one-step training.
    """
    if weighting == "uniform":
        return [1.0] * n_steps
    if weighting == "discount":
        return [gamma ** s for s in range(n_steps)]
    if weighting == "linear_decay":
        return [(n_steps - s) / n_steps for s in range(n_steps)]
    if weighting == "last":
        return [0.0] * (n_steps - 1) + [1.0]
    raise ValueError(
        f"unknown rollout_step_weighting {weighting!r} "
        "(expected 'uniform', 'discount', 'linear_decay' or 'last')")


def _dense_rollout_loss(
    model, x_ts, y_ts, edges, ids, frame_indices, iter_idx,
    n_steps, sim, tc, device, xnorm, ynorm, regularizer, has_visual_field,
    hn=None, target_weight=None,
):
    """ROLLOUT: unroll K = n_steps from frame k, scoring every step.

    Three things distinguish this from the legacy mode 1, all chosen to match the
    two trainers that work in this repo (_data_train_task_pi, train_eyeG.py):

    1. DENSE supervision -- every intermediate state is scored, not just the
       endpoint, so the trajectory cannot wander and still land correctly.
       Needs an unstrided dataset (time_step == 1).
    2. LIVE stimulus -- advanced each step rather than frozen through the unroll.
    3. K=1 EQUALS one-step training, term for term. Each step scores the
       derivative against y_ts[k+s]/ynorm with the same reduction the one-step
       path uses, averaged over K.

       Do not break 3. An earlier version scored integrated VOLTAGE against
       x_ts.voltage[k+s] with .pow(2).mean() and no ynorm. norm2 = sqrt(sum r^2)
       and mean(r^2) differ by orders of magnitude at the same residual, so the
       fit/regulariser balance flipped, the W penalties took over, and connectivity
       recovery collapsed.

    Knobs -- all no-ops at K=1, so 3 holds whatever they are set to. Benchmark arm
    names in brackets; see the module docstring for the full table.

      rollout_step_weighting   ["uniform"|"discount"|"last"]  how steps are weighted
      rollout_step_reduction   ["mean"|"sum"]  how the K weighted terms combine
      rollout_bptt_window      [1 = "pushforward"]  detach the state every m steps
      rollout_shooting_stride  [1 = "shoot1", 2 = "shoot2"]  re-anchor on data
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

    # Regularisation: once per iteration, not once per rollout step -- it
    # penalises parameters, which do not change within the unroll.
    regularizer.reset_iteration(device=device)
    regul_loss = regularizer.compute(
        model=model, x=state_batch[0], in_features=None,
        ids=ids, ids_batch=None, edges=edges, device=device, xnorm=xnorm,
        perm_indices=regularizer.sample_g_phi_perm(device),
    )
    loss = regul_loss.clone()
    regul_value = regul_loss.item()

    neurons_per_sample = state_batch[0].n_neurons
    fit_loss = torch.zeros((), device=device)
    weight_scored = 0.0

    reduction = getattr(tc, "fit_reduction", "norm2")
    huber_delta = getattr(tc, "fit_huber_delta", 1.0)
    weighting = getattr(tc, "rollout_step_weighting", "uniform")
    step_reduction = getattr(tc, "rollout_step_reduction", "mean")
    gamma = getattr(tc, "rollout_discount", 0.9)
    bptt_window = int(getattr(tc, "rollout_bptt_window", 0) or 0)
    shooting_stride = int(getattr(tc, "rollout_shooting_stride", 0) or 0)
    step_weights = _rollout_step_weights(weighting, n_steps, gamma)

    batched_state, batched_edges = _batch_frames(state_batch, edges)
    pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
    loss = loss + update_regul

    for step in range(n_steps):
        # Score the derivative, exactly as the one-step path does. At step 0 the
        # state is the observed v(k), so K=1 is term-for-term one-step training.
        # No unsqueeze: y_ts rows already carry pred's trailing dim. Adding one
        # broadcasts the residual to (N*B, N*B, 1) -- an 11 GiB matrix whose norm
        # is not the loss (OOM, and R^2_W went negative).
        gt_parts = [
            y_ts[k_list[b_idx] + step] / ynorm
            for b_idx in range(len(state_batch))
        ]
        y_step = torch.cat(gt_parts, dim=0)
        w = step_weights[step]
        if w > 0.0 and not torch.isnan(y_step).any():
            fit_loss = fit_loss + w * fit_residual_loss(
                pred[ids_batch] - y_step[ids_batch],
                reduction,
                target=y_step[ids_batch],
                huber_delta=huber_delta,
                weight=None if target_weight is None else target_weight[ids_batch],
            )
            weight_scored += w

        if step == n_steps - 1:
            break

        # --- integrate one step to get the next state ---
        if step == 0:
            pred_x = (batched_state.voltage.unsqueeze(-1) + sim.delta_t * pred
                      + tc.noise_recurrent_level * torch.randn_like(pred))
        else:
            pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

        # "pushforward" (bptt_window=1): cut the gradient every m steps so no
        # chain is longer than m. The model still SEES its drifted state -- that is
        # the point -- it just is not differentiated through.
        if bptt_window > 0 and (step + 1) % bptt_window == 0:
            pred_x = pred_x.detach()

        # "shoot1"/"shoot2" (shooting_stride): re-anchor on the observation every
        # m steps, starting a fresh segment from an exact initial condition instead
        # of continuing the free run. Overwrites the state, so it also cuts the
        # gradient.
        if shooting_stride > 0 and (step + 1) % shooting_stride == 0:
            # Use the same observation the setup loop uses at step 0 (noisy when
            # the dataset has measurement noise), so an anchor is exactly as
            # informative as the rollout's own start.
            noisy = x_ts.noise is not None and sim.measurement_noise_level > 0
            anchor = []
            for b_idx in range(len(state_batch)):
                kc = k_list[b_idx] + step + 1
                v = x_ts.voltage[kc]
                anchor.append(v + x_ts.noise[kc] if noisy else v)
            pred_x = torch.cat(anchor, dim=0).unsqueeze(-1)

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


    if weight_scored == 0.0:
        return torch.zeros(1, device=device, requires_grad=True), regul_value

    # Collapse the K steps. 'mean' divides by the weight actually applied (not by
    # K), so the objective scale is independent of both horizon and weighting and
    # no coeff_* needs rescaling when either changes. 'sum' leaves the weighted sum
    # alone, so the fit grows with the horizon while the regulariser does not.
    # This is the K-step reduction; fit_reduction already collapsed each step over
    # n_visible * batch_size.
    if step_reduction == "sum":
        loss = loss + fit_loss
    else:
        loss = loss + fit_loss / weight_scored
    return loss, regul_value


# ------------------------------------------------------------------ #
#  MODE 1 (legacy): single start, unroll time_step, score the endpoint #
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
        perm_indices=regularizer.sample_g_phi_perm(device),
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
#  MODE 2: time_step starts, all targeting frame T                    #
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
        perm_indices=regularizer.sample_g_phi_perm(device),
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
