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

    Returns (pearson_r, rmse, true (T,N), pred (T,N), stim (T,)).
    """
    from connectome_gnn.utils import to_numpy

    n_frames = int(min(n_frames, x_ts.n_frames - start - 1))
    if n_frames < 2:
        return float("nan"), float("nan"), None, None, None

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
    ok = np.isfinite(true) & np.isfinite(pred)
    if ok.sum() < 2:
        return float("nan"), float("nan"), true, pred, np.asarray(stim_l)
    a, b = true[ok].ravel(), pred[ok].ravel()
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    return r, rmse, true, pred, np.asarray(stim_l)


def save_trace_figure(path, true, pred, stim, delta_t, r, n_traces=12,
                      type_names=None, type_list=None):
    """Supplementary-Figure-6 style: stacked traces, green truth, black rollout, red stimulus.

    Traces are baseline-subtracted and offset so that a shared y-scale does not let
    the loudest neuron flatten every other row.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, N = true.shape
    idx = np.linspace(0, N - 1, min(n_traces, N)).astype(int)
    t_ms = np.arange(T) * delta_t * 1e3

    tr, pr = true[:, idx].T, pred[:, idx].T
    bl = tr.mean(axis=1, keepdims=True)
    step = float(np.nanpercentile(np.abs(tr - bl), 99)) * 2.5 or 1.0

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=150)
    for i in range(len(idx)):
        ax.plot(t_ms, (tr[i] - bl[i, 0]) + i * step, color=COLOR_TRUE, lw=0.7)
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
    ax.text(0.01, 0.99, f"r = {r:.4f}", transform=ax.transAxes, va="top",
            fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def evaluate_teacher_rollout(model, x_ts, edges, sim, device, log_dir, iteration,
                             n_frames=1000, has_visual_field=False, hn=None,
                             type_names=None, type_list=None, make_figure=True):
    """One checkpoint's worth: score the rollout, log it, draw the traces."""
    was_training = model.training
    model.eval()
    try:
        r, rmse, true, pred, stim = teacher_rollout(
            model, x_ts, edges, sim, device, n_frames=n_frames,
            has_visual_field=has_visual_field, hn=hn)
    finally:
        if was_training:
            model.train()

    tmp = os.path.join(log_dir, "tmp_training")
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "rollout_r.log"), "a") as f:
        f.write(f"{iteration},{r:.6f},{rmse:.6f},"
                f"{0 if true is None else true.shape[0]}\n")

    if make_figure and true is not None:
        save_trace_figure(
            os.path.join(tmp, "traces", f"rollout_{iteration:08d}.png"),
            true, pred, stim, sim.delta_t, r,
            type_names=type_names, type_list=type_list)
    return r, rmse
