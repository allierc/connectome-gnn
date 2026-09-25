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


def _scatter_voltage(state_batch, pred_x, neurons_per_sample):
    """Write a batched (N*B, 1) voltage back onto the per-sample states.

    The sub-step path needs the model re-run on the intermediate state, and the
    model takes per-sample states, so the batched column has to go back first.
    Same slicing the frame advance below uses.
    """
    for b_idx in range(len(state_batch)):
        s_, e_ = b_idx * neurons_per_sample, (b_idx + 1) * neurons_per_sample
        state_batch[b_idx].voltage = pred_x[s_:e_].squeeze()


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
    burn_in=None,
):
    """Dispatch to one of the three modes. See the module docstring.

    n_steps: the epoch's rollout horizon K. Given -> ROLLOUT; None -> mode 1 or 2.
    burn_in: the epoch's burn-in (burn_in_at_epoch). None falls back to
        training.rollout_burn_in, ignoring any warm start.
    hn: HiddenNeuronHandler, or None if the model has no hidden neurons.

    Returns (loss including regularisation, regularisation value for logging).
    """
    sim = config.simulation
    tc = config.training
    n_neurons = sim.n_neurons

    if n_steps is not None:
        return _dense_rollout_loss(
            model, x_ts, y_ts, edges, ids, frame_indices, iter_idx,
            int(n_steps), sim, tc, device, xnorm, ynorm, regularizer, has_visual_field,
            hn=hn, target_weight=target_weight, burn_in=burn_in,
        )
    return _standard_recurrent_loss(
        model, x_ts, edges, ids, frame_indices, iter_idx,
        sim, tc, device, xnorm, regularizer, has_visual_field,
        hn=hn,
    )


# ------------------------------------------------------------------ #
#  ROLLOUT: unroll K steps, score every one                          #
# ------------------------------------------------------------------ #

def _rollout_step_weights(weighting, n_steps, gamma, loss_stride=0, burn_in=0):
    """Per-step weights: "uniform" | "discount" | "linear_decay" | "last".

    Unnormalised -- the caller divides by the weight actually applied, so any
    positive scaling is equivalent. All schemes return [1.0] at K=1, which is what
    keeps the K=1 objective identical to one-step training.

    `loss_stride` m > 0 then ZEROES every step but 0, m, 2m, ..., which is
    partial temporal sampling: the model still integrates each intermediate
    frame, it is simply not scored there. The mask is applied AFTER the
    weighting, so "one frame in five, discounted" is expressible and the two
    knobs stay orthogonal.

    STEP 0 IS SCORED AND THE INDEX IS NOT SHIFTED, and both halves of that
    sentence were wrong until 2026-09-24. `_dense_rollout_loss` runs step s with
    the state at frame k+s against target y_ts[k+s] -- step 0 being the
    un-integrated observed frame k itself. The mask used to read
    `(s + 1) % m == 0`, which selects s = m-1, 2m-1, ..., so with m = 5 the loss
    landed on frames k+4, k+9, k+14, k+19 while a 1-in-5 recording anchored at k
    observes k, k+5, k+10, k+15, k+20: NO OVERLAP. It scored exactly the frames
    the knob exists to skip, and it dropped step 0, the one term whose state is
    an observation with zero integration drift.

    That cost experiment 5 all thirty of its runs. Its conductance arm collapsed
    to R2_W -0.012 -- the strided fit gradient on g_phi's first layer fell to
    0.69-6.1 against a constant group-lasso pull of 200, which annihilated the
    v_j input column -- and its current arm was degraded to 0.753 against 0.956
    for unstrided training on the same data, with R2_tau 0.655 against 0.979.

    A HORIZON OF K THEREFORE SCORES floor((K-1)/m) + 1 STEPS, and reaching step
    s = m needs K >= m + 1: "one observed interval" is horizon m+1, not m.

    `burn_in` B > 0 then zeroes steps 0 .. B-1 as well. It is a MASK, like the
    stride, and deliberately not a detach: the loop still integrates the
    burn-in steps and autograd still runs back through them, so a scored step
    at s >= B keeps its whole chain to the parameters. Under measurement noise
    the early steps are where the loss is biased (see TrainingConfig.
    rollout_burn_in), and detaching at the boundary measured worse than not
    burning in at all.

    Applied last, after the weighting and the stride, so the three compose: a
    step is scored when its dense weight is non-zero, it sits on the stride's
    grid, and s >= B.
    """
    w = _rollout_step_weights_dense(weighting, n_steps, gamma)
    if loss_stride and loss_stride > 0:
        # s is 0-indexed and the state at step s is frame k+s, so the observed
        # grid k, k+m, k+2m, ... is exactly s % m == 0.
        w = [x if s % loss_stride == 0 else 0.0 for s, x in enumerate(w)]
    if burn_in and burn_in > 0:
        w = [x if s >= burn_in else 0.0 for s, x in enumerate(w)]
    return w


def burn_in_at_epoch(training, epoch):
    """The rollout burn-in in force at `epoch`: 0 during the warm-up epochs
    (epoch < rollout_burn_in_start_epoch), rollout_burn_in after.

    One function, called by the trainer to set each epoch's mask and by
    validate_rollout_masks to check the schedule, so the two cannot disagree
    about which epochs are burned in.
    """
    burn = int(getattr(training, "rollout_burn_in", 0) or 0)
    start = int(getattr(training, "rollout_burn_in_start_epoch", 0) or 0)
    return burn if epoch >= start else 0


def validate_rollout_masks(training):
    """Refuse a rollout configuration whose step masks would silently misbehave.

    Lives here, beside the weights it reasons about, rather than inline in
    graph_trainer's training function where no test could reach it -- the
    stride audit found both of its sibling guards untested. graph_trainer calls
    this once, before the schedule is expanded. Raises ValueError; returns None.

    Three refusals:

    NO SCHEDULE. rollout_loss_stride and rollout_burn_in are read by
    _dense_rollout_loss alone; without rollout_horizon_schedule the dispatcher
    takes the legacy endpoint path, which never looks at them, so either knob
    would be accepted and do nothing.

    A STRIDE HORIZON THAT REACHES ONLY THE ANCHOR. With stride m the scored steps
    are 0, m, 2m, ..., so a horizon K < m + 1 scores step 0 alone -- one-step
    training wearing a rollout's clothes.

    A HORIZON THAT SCORES NOTHING. A burn-in B zeroes steps 0 .. B-1, so a
    horizon K <= B -- or one whose stride-grid steps all fall inside the burn-in
    -- contributes a loss of exactly zero and trains on nothing while looking
    busy. Decided by calling _rollout_step_weights itself, with this run's
    weighting, stride and burn-in, so the check and the loss cannot drift apart.
    """
    horizons = [int(h) for h in (getattr(training, "rollout_horizon_schedule", []) or [])]
    stride = int(getattr(training, "rollout_loss_stride", 0) or 0)
    burn = int(getattr(training, "rollout_burn_in", 0) or 0)
    start = int(getattr(training, "rollout_burn_in_start_epoch", 0) or 0)
    n_epochs = int(getattr(training, "n_epochs", len(horizons)) or len(horizons))


    for knob, value in (("rollout_loss_stride", stride), ("rollout_burn_in", burn)):
        if value > 0 and not horizons:
            raise ValueError(
                f"{knob} is only read by the dense rollout, which needs "
                f"rollout_horizon_schedule; without one it would be silently "
                f"ignored. Set a schedule, or drop {knob}.")
    if not horizons:
        return

    # A WARM START WITH NOTHING TO WARM UP FOR, OR THAT NEVER ENDS. Either would
    # be accepted and silently change nothing.
    if start > 0 and burn == 0:
        raise ValueError(
            f"rollout_burn_in_start_epoch {start} delays a burn-in that is not "
            f"set (rollout_burn_in 0); it would do nothing. Set rollout_burn_in "
            f"or drop the start epoch.")
    if burn > 0 and start >= n_epochs:
        raise ValueError(
            f"rollout_burn_in_start_epoch {start} is not before n_epochs "
            f"{n_epochs}, so the burn-in would never apply.")

    if stride > 1:
        short = sorted({h for h in horizons if h < stride + 1})
        if short:
            raise ValueError(
                f"rollout_loss_stride: {stride} scores steps 0, {stride}, "
                f"{2 * stride}, ..., and step s holds the state at frame k+s, so "
                f"a horizon must reach {stride + 1} for the first observed frame "
                f"after the anchor to be in the rollout at all; "
                f"rollout_horizon_schedule contains {short}, whose epochs would "
                f"score the anchor and nothing else. Use {stride + 1}, "
                f"{2 * stride + 1}, ...")

    if burn > 0:
        weighting = getattr(training, "rollout_step_weighting", "uniform")
        gamma = getattr(training, "rollout_discount", 0.9)
        # The schedule as the trainer will run it: padded with its last value to
        # n_epochs. Only the epochs where the burn-in is in force are checked;
        # the warm-up epochs score step 0 and may use any horizon.
        per_epoch = (horizons + [horizons[-1]] * max(0, n_epochs - len(horizons)))[:n_epochs]
        empty = sorted({(e, h) for e, h in enumerate(per_epoch)
                        if not any(w > 0.0 for w in _rollout_step_weights(
                            weighting, h, gamma, loss_stride=stride,
                            burn_in=burn_in_at_epoch(training, e)))})
        if empty:
            raise ValueError(
                f"rollout_burn_in: {burn} leaves steps 0..{burn - 1} unscored "
                f"from epoch {start}, and with rollout_step_weighting "
                f"'{weighting}'"
                + (f" and rollout_loss_stride {stride}" if stride > 1 else "")
                + f" these (epoch, horizon) pairs score no step at all: "
                f"{empty}. Every horizon from epoch {start} on must reach past "
                f"the burn-in, e.g. {burn + 1}.")


def _rollout_step_weights_dense(weighting, n_steps, gamma):
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
    hn=None, target_weight=None, burn_in=None,
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
    step_weights = _rollout_step_weights(
        weighting, n_steps, gamma,
        loss_stride=int(getattr(tc, "rollout_loss_stride", 0) or 0),
        # The trainer passes the epoch's burn-in (burn_in_at_epoch), which is 0
        # during a warm start. None -- a direct caller with no epoch -- falls back
        # to the configured value.
        burn_in=(int(getattr(tc, "rollout_burn_in", 0) or 0)
                 if burn_in is None else int(burn_in)))

    batched_state, batched_edges = _batch_frames(state_batch, edges)
    pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

    update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
    loss = loss + update_regul

    _int_method = getattr(tc, "integration_method", "euler")
    _n_sub = max(1, int(getattr(tc, "n_rollout_substeps", 5)))

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

        # --- integrate one OBSERVED FRAME to get the next state ---
        # `euler` is one step of delta_t and is what this has always done.
        # `multi_substeps` crosses the same delta_t as M steps of delta_t/M with
        # the message RECOMPUTED at each: the generator uses exponential Euler,
        # and forward Euler at delta_t contracts only while
        # z = (delta_t/tau_i)(1 + G_i) < 2 while the generating network reaches
        # 4.4 -- so a model that learned the generator exactly would diverge
        # here. Sub-stepping divides z by M. Process noise is added once per
        # observed frame in both paths, so the two differ only in the integrator.
        _v0 = (batched_state.voltage.unsqueeze(-1) if step == 0 else pred_x)
        if _int_method == "multi_substeps" and _n_sub > 1:
            _h = sim.delta_t / _n_sub
            pred_x = _v0 + _h * pred
            for _sub in range(_n_sub - 1):
                _scatter_voltage(state_batch, pred_x, neurons_per_sample)
                _bs, _be = _batch_frames(state_batch, edges)
                _p, _, _ = model(_bs, _be, data_id=data_id, return_all=True)
                pred_x = pred_x + _h * _p
        else:
            pred_x = _v0 + sim.delta_t * pred
        pred_x = pred_x + tc.noise_recurrent_level * torch.randn_like(pred_x)

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
    sim, tc, device, xnorm, regularizer, has_visual_field,
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

        y = x_ts.voltage[k + 1].unsqueeze(-1)
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

    # No inner unroll: the horizon lives in rollout_horizon_schedule now.
    for step in range(0):
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

    loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / sim.delta_t).norm(2)
    return loss, regul_value


# ------------------------------------------------------------------ #
#  MODE 2: time_step starts, all targeting frame T                    #
# ------------------------------------------------------------------ #
