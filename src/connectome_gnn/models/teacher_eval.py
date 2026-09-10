"""Rollout evaluation during training, for teacher-student distillation.

WHY THIS EXISTS. When training.train_on_teacher is set, the run is fitting a
STUDENT (e.g. flyvis_conductance_known_ode) to a teacher's recorded activity. R2_W is
meaningless there -- the teacher is current-based and carries no conductance
ground truth to recover -- so the number that decides whether the student is
usable is the ROLLOUT: does it run free and stay on the teacher's trajectory.

WHY IN-PROCESS AND NOT `GNN_Main.py -o test` IN A THREAD, which was the obvious
alternative. Four reasons, and none of them is style:

  * LSF grants the GPU as mode=exclusive_process, so a second process cannot open
    the device at all.
  * `-o test` writes results_rollout.log, results_rollout_by_step.csv and results/
    into the SAME log dir the trainer is checkpointing into -- it would clobber the
    final test output and race with torch.save.
  * it reloads the 8 GB dataset every call, minutes per checkpoint, for a rollout
    that takes about a second.
  * it needs thread management, arg plumbing, a separate log dir and
    checkpoint-consistency handling. More code than this file, not less.

The trainer already holds the model, x_ts, edges and device at the checkpoint, so
everything the rollout needs is in memory.

IN SAMPLE, AND THAT IS THE POINT OF IT. The trainer loads the TRAIN split, so this
rolls out over the very frames the derivative loss is fitted on. It is a training
DIAGNOSTIC -- it answers "is the student still on the teacher's trajectory or has
it started to drift", per checkpoint, for free. It is NOT the acceptance test.
The held-out number comes from `GNN_Main.py -o test`, which rolls out on
x_list_test (graph_tester only falls back to training frames when a field INR was
learned, which these models do not have). Quote the -o test number in anything
that leaves this repo; quote this one as train-split.

WHAT IT WRITES, both under tmp_training/ and NEITHER touching metrics.log:
  rollout_r.log        iteration,pearson_r,rmse,n_frames   -- TRAIN split
  traces/<iter>.png    green ground truth, black rollout, red stimulus

metrics.log is deliberately left alone. Adding a column there means editing the
header and all four write sites, and plot.py reads that file by POSITIONAL index
(`_f(parts, idx)`), so a new column shifts everything after it. A separate file
costs nothing and cannot break the existing readers.
"""
import os

import numpy as np
import torch

from connectome_gnn.utils import fisher_pool


def score_rollout(true, pred):
    """(r_fisher, r_pooled, rmse) for a (T, N) rollout against its truth.

    TWO CORRELATIONS, AND THE GAP BETWEEN THEM IS A DIAGNOSTIC.

    r_fisher   per-neuron Pearson over time, pooled in Fisher-z space -- the same
               statistic `-o test` reports as "Pearson r (Fisher-z pooled over
               neurons)", with the same validity rule (both stds > 1e-8, else the
               neuron is excluded). THIS IS THE ROLLOUT NUMBER.
    r_pooled   one Pearson over every (frame, neuron) pair flattened together.
               Rewards getting each neuron's mean LEVEL right, not its dynamics:
               with 13,741 neurons whose rest and drive levels differ far more
               than any one of them moves in time, the between-neuron spread
               dominates the variance and a flat line at each neuron's correct
               level scores high. Measured on a sigma-0.05 GNN checkpoint whose
               trace panel showed flat predictions on every non-input neuron:
               r_pooled 0.84, r_fisher 0.50. Kept in the log precisely so that
               gap is visible, never as the headline.

    Non-finite anywhere in a neuron's trace excludes that neuron. A free-run that
    produces a NaN has diverged, and there is no partial credit for the frames
    before it did.
    """
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ok = np.isfinite(true) & np.isfinite(pred)
    if ok.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    a, b = true[ok].ravel(), pred[ok].ravel()
    r_pooled = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    # per-neuron r over time, vectorised; neurons with any non-finite sample or
    # a flat trace on either side drop out as nan and fisher_pool ignores them.
    col_ok = ok.all(axis=0)
    t = np.where(col_ok, true, 0.0); p = np.where(col_ok, pred, 0.0)
    tc = t - t.mean(axis=0, keepdims=True)
    pc = p - p.mean(axis=0, keepdims=True)
    st = np.sqrt((tc ** 2).mean(axis=0)); sp = np.sqrt((pc ** 2).mean(axis=0))
    good = col_ok & (st > 1e-8) & (sp > 1e-8)
    num = (tc * pc).mean(axis=0)
    r_i = np.full(true.shape[1], np.nan)
    r_i[good] = num[good] / (st[good] * sp[good])
    r_fisher = float(fisher_pool(r_i)["r_mean"])
    return r_fisher, r_pooled, rmse

# Green ground truth, black prediction -- the repo's GT-vs-predicted convention.
COLOR_TRUE, COLOR_PRED, COLOR_STIM = "tab:green", "black", "tab:red"


@torch.no_grad()
def teacher_rollout(model, x_ts, edges, sim, device, n_frames=1000, start=0,
                    has_visual_field=False, hn=None):
    """Free-run the student from one observed frame and score it against the teacher.

    Free-run means x.voltage is NEVER reset from ground truth after the first
    frame -- only the stimulus is fed from data, exactly as graph_tester's rollout
    does. A version that re-anchored the voltage would report the one-step error
    and call it a rollout.

    Returns (r_fisher, rmse, true (T,N), pred (T,N), stim (T,), r_pooled). The
    first is the per-neuron Fisher-pooled r that `-o test` reports; the last is
    the flattened-pair r kept only to expose level-fitting -- see score_rollout.
    """
    from connectome_gnn.utils import to_numpy

    n_frames = int(min(n_frames, x_ts.n_frames - start - 1))
    if n_frames < 2:
        return float("nan"), float("nan"), None, None, None, float("nan")

    x = x_ts.frame(start)
    x.voltage = x.voltage.clone()
    data_id = torch.zeros((x.n_neurons, 1), dtype=torch.int, device=device)

    true_l, pred_l, stim_l = [], [], []
    for k in range(start, start + n_frames):
        pred_l.append(to_numpy(x.voltage))
        true_l.append(to_numpy(x_ts.frame(k).voltage))

        frame_k = x_ts.frame(k)
        x.stimulus = frame_k.stimulus.clone()
        stim_l.append(float(x.stimulus[0]))
        if has_visual_field:
            vi = model.forward_visual(x, k)
            x.stimulus[: model.n_input_neurons] = vi.squeeze(-1)
            x.stimulus[model.n_input_neurons:] = 0

        y = model(x, edges, data_id=data_id, return_all=False)
        x.voltage = x.voltage + sim.delta_t * y.squeeze(-1)
        if hn is not None:
            hn.inject_hidden(model, x, k + 1, True)

    true = np.asarray(true_l)
    pred = np.asarray(pred_l)
    r_fisher, r_pooled, rmse = score_rollout(true, pred)
    return r_fisher, rmse, true, pred, np.asarray(stim_l), r_pooled


def save_trace_figure(path, true, pred, stim, delta_t, r, n_traces=12,
                      type_names=None, type_list=None, r_pooled=None):
    """Supplementary-Figure-6 style: stacked traces, green truth, black rollout, red stimulus.

    Traces are baseline-subtracted and offset so that a shared y-scale does not let
    the loudest neuron flatten every other row.

    `pred` and `r` may be None. That is the DATA-ONLY case: the same figure drawn
    straight off a generated dataset, where there is no student to roll out and so
    no rollout trace and no rollout-vs-truth correlation to annotate. Only the
    green ground truth and the red stimulus are drawn then. graph_data_generator
    calls it that way for <dataset>/activity.png so the generated data and the
    training-time rollouts are read on the same axes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, N = true.shape
    idx = np.linspace(0, N - 1, min(n_traces, N)).astype(int)
    t_ms = np.arange(T) * delta_t * 1e3

    tr = true[:, idx].T
    pr = pred[:, idx].T if pred is not None else None
    bl = tr.mean(axis=1, keepdims=True)
    step = float(np.nanpercentile(np.abs(tr - bl), 99)) * 2.5 or 1.0

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
    for i in range(len(idx)):
        ax.plot(t_ms, (tr[i] - bl[i, 0]) + i * step, color=COLOR_TRUE, lw=0.7)
        if pr is not None:
            ax.plot(t_ms, (pr[i] - bl[i, 0]) + i * step, color=COLOR_PRED, lw=0.6)
    if stim is not None and np.isfinite(stim).any():
        s = stim - np.nanmean(stim)
        sc = step / (np.nanmax(np.abs(s)) or 1.0)
        ax.plot(t_ms, s * sc - step, color=COLOR_STIM, lw=0.6)

    labels = []
    for j in idx:
        if type_names is not None and type_list is not None:
            try:
                labels.append(str(type_names[int(type_list[j])]))
                continue
            except Exception:
                pass
        labels.append(str(int(j)))
    ax.set_yticks([i * step for i in range(len(idx))] + [-step])
    ax.set_yticklabels(labels + ["stim"], fontsize=7)
    ax.set_xlabel("time (ms)", fontsize=9)
    ax.set_ylabel("neurons", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    if r is not None and np.isfinite(r):
        ax.text(0.01, 0.99,
                (f"r = {r:.4f}" if r_pooled is None else
                 f"r = {r:.4f}  (per-neuron, Fisher-z)      pooled r = {r_pooled:.4f}"),
                transform=ax.transAxes, va="top",
                fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def evaluate_teacher_rollout(model, x_ts, edges, sim, device, log_dir, iteration,
                             n_frames=1000, has_visual_field=False, hn=None,
                             type_names=None, type_list=None, make_figure=True):
    """One checkpoint's worth: score the rollout, log it, draw the traces.

    `make_figure` separates the two. The rollout itself always runs and always
    appends to rollout_r.log, because r and rmse are the trajectory metric and
    are what gets trended; the stacked-trace figure is ~1 MB and only useful to
    flip through, so the trainer draws it on its panel cadence rather than on
    every metric evaluation.
    """
    was_training = model.training
    model.eval()
    try:
        r, rmse, true, pred, stim, r_pooled = teacher_rollout(
            model, x_ts, edges, sim, device, n_frames=n_frames,
            has_visual_field=has_visual_field, hn=hn)
    finally:
        if was_training:
            model.train()

    tmp = os.path.join(log_dir, "tmp_training")
    os.makedirs(tmp, exist_ok=True)
    # Columns: iteration, r (per-neuron Fisher-pooled -- the same statistic as
    # `-o test`), rmse, n_frames, r_pooled (flattened pairs; level-fitting
    # detector, see score_rollout). The fifth column is new; the first four are
    # unchanged so any reader of the old layout still works.
    with open(os.path.join(tmp, "rollout_r.log"), "a") as f:
        f.write(f"{iteration},{r:.6f},{rmse:.6f},"
                f"{0 if true is None else true.shape[0]},{r_pooled:.6f}\n")

    if make_figure and true is not None:
        save_trace_figure(
            os.path.join(tmp, "traces", f"rollout_{iteration:08d}.png"),
            true, pred, stim, sim.delta_t, r,
            type_names=type_names, type_list=type_list, r_pooled=r_pooled)
    return r, rmse
