"""Cortex (Yang 2019) task evaluation + training snapshot.

Sibling of `cx_eval.py` (which serves the path-integration / drosophila_cx_pi
trainer). Provides:

    compute_cortex_task_metrics(preds, tgts, cmasks) -> dict
    save_cortex_training_snapshot(...)               -> writes 8-panel PNG

Both are ports of `papers/multi-tasks/src/NeuralGraph/data_loaders/multi_task_data.py`
(`compute_task_metrics`) and `papers/multi-tasks/notebooks/multi_task/analyze_gnn.ipynb`
(cell 7 "combined figure"), kept identical so the cortex trainer's snapshots
match the multi-tasks reference visualisation.

Conventions (same as Yang/gyyang/multitask):
    pred / target shape:  (T, N_o)   N_o = 1 fixation + 32-ch motor ring
    c_mask shape:         (T, N_o)   weighting for masked-MSE loss
    channel 0:            fixation
    channels 1..32:       motor ring (1 of 32 active per trial)
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Scalar metrics (direct port of multi_task_data.compute_task_metrics)
# ---------------------------------------------------------------------------

def compute_cortex_task_metrics(
    pred_list: Sequence[torch.Tensor],
    target_list: Sequence[torch.Tensor],
    cmask_list: Sequence[torch.Tensor],
) -> dict:
    """Aggregate per-trial diagnostic metrics across a list of trials.

    Each input is a list (length n_trials) of tensors of shape (T, N_o).

    Returns dict with: loss, motor_max, motor_peak_mean, direction_acc.
    """
    losses, motor_maxes, peaks = [], [], []
    correct = 0
    for pred, y_tgt, c_mask in zip(pred_list, target_list, cmask_list):
        losses.append(float(((pred - y_tgt) ** 2 * c_mask).mean().item()))
        motor_pred = pred[:, 1:].detach().cpu().numpy()
        motor_tgt = y_tgt[:, 1:].detach().cpu().numpy()
        motor_maxes.append(float(motor_pred.max()))
        peaks.append(float(motor_pred.max(axis=0).max()))
        if int(motor_tgt.max(axis=0).argmax()) == int(motor_pred.max(axis=0).argmax()):
            correct += 1
    n = max(len(losses), 1)
    return {
        "loss": float(np.mean(losses)),
        "motor_max": float(np.mean(motor_maxes)),
        "motor_peak_mean": float(np.mean(peaks)),
        "direction_acc": correct / n,
    }


# ---------------------------------------------------------------------------
# 8-panel training snapshot (port of analyze_gnn.ipynb cell 7)
# ---------------------------------------------------------------------------

def save_cortex_training_snapshot(
    preds: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    cmasks: Sequence[torch.Tensor],
    *,
    output_path: str,
    step: int,
    rule_name: str = "delaygo",
) -> dict:
    """Write the 8-panel multi-tasks figure to `output_path` and return metrics.

    The figure layout (matches papers/multi-tasks/notebooks analyze_gnn cell 7):
        row 0 col 0:1  fixation channel trajectory  (target vs pred)
        row 1 col 0:1  motor channel trajectory     (target vs pred + distractors)
        row 0 col 2:3  motor peak amplitude histogram (n trials)
        row 1 col 2:3  scatter target_active_ch vs pred_active_ch + diagonal
        row 2 col 0..3 target heatmaps for 4 example trials
        row 3 col 0..3 pred heatmaps   for the same 4 trials

    Args:
        preds:   list of N>=4 prediction tensors (T, N_o). The first is used
                 for the trajectory plot; the first 4 for the heatmaps; all
                 of them for the histogram + scatter + scalar metrics.
        targets: list of N target tensors, same shape, aligned with `preds`.
        cmasks:  list of N c_mask tensors, same shape, aligned with `preds`.
        output_path: full destination .png path.
        step:    current training iteration (shown in title).
        rule_name: cortex rule name for title (e.g. "delaygo").

    Returns the metrics dict from compute_cortex_task_metrics.
    """
    n_trials = len(preds)
    assert n_trials == len(targets) == len(cmasks), \
        f"preds/targets/cmasks length mismatch: {n_trials}/{len(targets)}/{len(cmasks)}"
    assert n_trials >= 4, f"need >=4 trials for snapshot, got {n_trials}"

    # --- trial 0: full-trajectory plot ---
    pred_np = preds[0].detach().cpu().numpy()
    tgt_np = targets[0].detach().cpu().numpy()
    cm_np = cmasks[0].detach().cpu().numpy()
    active = int(tgt_np[:, 1:].max(axis=0).argmax()) + 1
    # response window = where c_mask on a motor channel is up-weighted (>4 in
    # Yang's lsq formulation; falls back to >mean if scaling differs)
    cm_motor_max = cm_np[:, 1:].max(axis=1)
    thresh = max(4.0, cm_motor_max.mean() * 1.5)
    rw = np.where(cm_motor_max > thresh)[0]

    # --- aggregate stats across all trials ---
    metrics = compute_cortex_task_metrics(preds, targets, cmasks)
    peaks_pred = np.array([p[:, 1:].detach().cpu().numpy().max() for p in preds])
    peaks_tgt = np.array([t[:, 1:].detach().cpu().numpy().max() for t in targets])
    pred_active = np.array([int(p[:, 1:].detach().cpu().numpy().max(axis=0).argmax())
                            for p in preds])
    tgt_active = np.array([int(t[:, 1:].detach().cpu().numpy().max(axis=0).argmax())
                           for t in targets])
    n_motor_ch = preds[0].shape[1] - 1

    # --- build figure ---
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.35,
                  height_ratios=[1, 1, 1.4, 1.4])

    # row 0/1 col 0:1 — fixation + motor trajectory
    ax_fix = fig.add_subplot(gs[0, 0:2])
    ax_fix.plot(tgt_np[:, 0], "k-", lw=1.5, label="target")
    ax_fix.plot(pred_np[:, 0], "r--", label="pred")
    ax_fix.set_ylabel("fix (ch 0)")
    ax_fix.legend(fontsize=8)
    ax_fix.set_title(f"trajectory  (iter {step})  rule={rule_name}")

    ax_mot = fig.add_subplot(gs[1, 0:2], sharex=ax_fix)
    ax_mot.plot(tgt_np[:, active], "k-", lw=1.5, label=f"target ch {active}")
    ax_mot.plot(pred_np[:, active], "r--", label=f"pred ch {active}")
    for ch_off in (3, 8, 15):
        other = ((active + ch_off - 1) % n_motor_ch) + 1
        ax_mot.plot(pred_np[:, other], "-", alpha=0.25,
                    label=f"pred ch {other}")
    if len(rw):
        ax_mot.axvspan(rw[0], rw[-1], color="blue", alpha=0.08)
    ax_mot.set_xlabel("t")
    ax_mot.set_ylabel("motor")
    ax_mot.legend(fontsize=7, ncol=2)

    # row 0 col 2:3 — peak distribution
    ax_hist = fig.add_subplot(gs[0, 2:4])
    ax_hist.hist(peaks_pred, bins=30, alpha=0.7, label="pred")
    ax_hist.hist(peaks_tgt, bins=30, alpha=0.7, label="target")
    ax_hist.set_xlabel("motor peak amplitude")
    ax_hist.set_ylabel("count")
    ax_hist.legend(fontsize=8)
    ax_hist.set_title(f"peak distribution (n={n_trials})")

    # row 1 col 2:3 — scatter of active channel
    ax_sct = fig.add_subplot(gs[1, 2:4])
    ax_sct.scatter(tgt_active, pred_active, alpha=0.3, s=14)
    ax_sct.plot([0, n_motor_ch - 1], [0, n_motor_ch - 1], "k--", alpha=0.3)
    ax_sct.set_xlabel("target active ch")
    ax_sct.set_ylabel("pred active ch")
    ax_sct.set_title(f"direction_acc = {metrics['direction_acc']:.2f}")

    # row 2/3 — 4-trial heatmaps
    vmax = 0.9
    for b in range(4):
        tgt_b = targets[b][:, 1:].detach().cpu().numpy().T
        pred_b = preds[b][:, 1:].detach().cpu().numpy().T
        ax_t = fig.add_subplot(gs[2, b])
        ax_t.imshow(tgt_b, aspect="auto", cmap="hot", vmin=0, vmax=vmax)
        ax_t.set_title(f"target (trial {b})", fontsize=10)
        ax_t.set_xlabel("t"); ax_t.set_ylabel("motor ch")
        ax_p = fig.add_subplot(gs[3, b])
        ax_p.imshow(pred_b, aspect="auto", cmap="hot", vmin=0, vmax=vmax)
        ax_p.set_title("pred", fontsize=10)
        ax_p.set_xlabel("t"); ax_p.set_ylabel("motor ch")

    fig.suptitle(f"GNN on {rule_name}  —  iter {step}", fontsize=12, y=0.995)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return metrics
