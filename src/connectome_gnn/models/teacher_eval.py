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

    WHO SCORES ZERO AND WHO IS EXCLUDED -- this is the rule that decides whether a
    diverged rollout can look perfect, so it is spelled out:

      * truth flat (std <= 1e-8): r is UNDEFINED for that neuron, and it is
        excluded. There is nothing to track.
      * truth varies, prediction non-finite, overflowed, or flat: r = 0. The
        prediction captured none of the variance, and that is a measurement, not
        a missing value. Excluding these is how a rollout that explodes to 1e34 at
        3.5 s scored 0.9995 -- every exploded neuron went nan in the arithmetic,
        fisher_pool dropped it, and the 1,736 stimulus-clamped input neurons
        scored alone. Likewise a model that predicts a flat line for 87% of the
        network must not be graded on the 13% it moved.

    `n_diverged` counts neurons with a non-finite or overflowed prediction;
    `n_scored` counts neurons that entered the pool. Both go to the log so a
    reader can see how much of the network the r describes.
    """
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    n_neurons = true.shape[1]
    ok = np.isfinite(true) & np.isfinite(pred)
    if ok.sum() < 2:
        return dict(r_fisher=float("nan"), r_pooled=float("nan"), rmse=float("nan"),
                    n_scored=0, n_diverged=n_neurons, n_neurons=n_neurons)
    a, b = true[ok].ravel(), pred[ok].ravel()
    r_pooled = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    # A prediction is "overflowed" when it leaves the truth's range by a factor
    # that no honest model reaches; 1e6x the truth's own extent is well past any
    # gauge ambiguity and well short of float overflow.
    extent = float(np.nanmax(np.abs(true[np.isfinite(true)])) or 1.0)
    blown = (~np.isfinite(pred)) | (np.abs(pred) > 1e6 * extent)
    pred_ok = ~blown.any(axis=0)
    t = np.where(np.isfinite(true), true, 0.0)
    p = np.where(blown, 0.0, pred)
    tc = t - t.mean(axis=0, keepdims=True)
    pc = p - p.mean(axis=0, keepdims=True)
    st = np.sqrt((tc ** 2).mean(axis=0)); sp = np.sqrt((pc ** 2).mean(axis=0))
    truth_varies = st > 1e-8
    scorable = truth_varies & pred_ok & (sp > 1e-8)
    r_i = np.full(n_neurons, np.nan)
    num = (tc * pc).mean(axis=0)
    r_i[scorable] = num[scorable] / (st[scorable] * sp[scorable])
    r_i[truth_varies & ~scorable] = 0.0          # captured nothing: scored, as zero
    r_fisher = float(fisher_pool(r_i)["r_mean"])
    return dict(r_fisher=r_fisher, r_pooled=r_pooled, rmse=rmse,
                n_scored=int(truth_varies.sum()), n_diverged=int((~pred_ok).sum()),
                n_neurons=n_neurons)

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

    Returns (r_fisher, rmse, true (T,N), pred (T,N), stim (T,), score) where
    score is score_rollout's dict (or None when no frames ran). r_fisher is the
    per-neuron Fisher-pooled r that `-o test` reports, with diverged and flat
    predictions scored as zero rather than excluded -- see score_rollout.
    """
    from connectome_gnn.utils import to_numpy

    n_frames = int(min(n_frames, x_ts.n_frames - start - 1))
    if n_frames < 2:
        return float("nan"), float("nan"), None, None, None, None

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
    s = score_rollout(true, pred)
    return s["r_fisher"], s["rmse"], true, pred, np.asarray(stim_l), s


def save_trace_figure(path, true, pred, stim, delta_t, r, n_traces=12,
                      type_names=None, type_list=None, r_pooled=None,
                      n_diverged=0, n_neurons=None):
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
                 f"r = {r:.4f}  (per-neuron, Fisher-z)      pooled r = {r_pooled:.4f}")
                + (f"      DIVERGED: {n_diverged}/{n_neurons} neurons" if n_diverged else ""),
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
        r, rmse, true, pred, stim, score = teacher_rollout(
            model, x_ts, edges, sim, device, n_frames=n_frames,
            has_visual_field=has_visual_field, hn=hn)
    finally:
        if was_training:
            model.train()

    r_pooled = float("nan") if score is None else score["r_pooled"]
    n_div = 0 if score is None else score["n_diverged"]
    n_sc = 0 if score is None else score["n_scored"]
    tmp = os.path.join(log_dir, "tmp_training")
    os.makedirs(tmp, exist_ok=True)
    # Columns: iteration, r (per-neuron Fisher-pooled, diverged/flat scored as 0
    # -- the `-o test` statistic), rmse, n_frames, r_pooled (flattened pairs;
    # level-fitting detector), n_diverged (neurons whose prediction went
    # non-finite or overflowed), n_scored (neurons whose truth varied and so
    # entered the pool). The first four columns are the original layout.
    with open(os.path.join(tmp, "rollout_r.log"), "a") as f:
        f.write(f"{iteration},{r:.6f},{rmse:.6g},"
                f"{0 if true is None else true.shape[0]},{r_pooled:.6f},{n_div},{n_sc}\n")

    if make_figure and true is not None:
        save_trace_figure(
            os.path.join(tmp, "traces", f"rollout_{iteration:08d}.png"),
            true, pred, stim, sim.delta_t, r,
            type_names=type_names, type_list=type_list, r_pooled=r_pooled,
            n_diverged=n_div, n_neurons=true.shape[1])
    return r, rmse
