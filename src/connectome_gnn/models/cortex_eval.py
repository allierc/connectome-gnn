"""Cortex (Yang 2019) task evaluation + training snapshot.

Sibling of `cx_eval.py` (which serves the path-integration / drosophila_cx_pi
trainer). Provides:

    compute_cortex_task_metrics(preds, tgts, cmasks) -> dict
    save_cortex_training_snapshot(...)               -> writes 8-panel PNG
    save_cortex_matrix_snapshot(W_rec, ...)          -> writes W_rec heatmap
    save_cortex_test_kinograph(...)                  -> writes 10-trial GT/pred kinograph

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

    Returns dict with: loss, motor_max, motor_peak_mean, direction_acc, r2.

    r2 is the coefficient of determination of motor channels (1..N_o)
    against the target, computed on supervised frames only (c_mask > 0),
    same formula as `models/graph_tester.py` stimuli_R2:
        r2 = 1 - SS_res / SS_tot
    where SS_res / SS_tot are summed across the pooled (trial, time,
    channel) array of supervised entries.
    """
    losses, motor_maxes, peaks = [], [], []
    correct = 0
    # Pool supervised motor (pred, target) entries across all trials for R².
    pred_pool, tgt_pool = [], []
    for pred, y_tgt, c_mask in zip(pred_list, target_list, cmask_list):
        losses.append(float(((pred - y_tgt) ** 2 * c_mask).mean().item()))
        motor_pred = pred[:, 1:].detach().cpu().numpy()
        motor_tgt = y_tgt[:, 1:].detach().cpu().numpy()
        motor_maxes.append(float(motor_pred.max()))
        peaks.append(float(motor_pred.max(axis=0).max()))
        if int(motor_tgt.max(axis=0).argmax()) == int(motor_pred.max(axis=0).argmax()):
            correct += 1
        # Supervised mask on motor channels (c_mask > 0).
        cm_motor = c_mask[:, 1:].detach().cpu().numpy()
        mask = cm_motor > 0
        if mask.any():
            pred_pool.append(motor_pred[mask])
            tgt_pool.append(motor_tgt[mask])

    n = max(len(losses), 1)
    if pred_pool:
        pred_flat = np.concatenate(pred_pool)
        tgt_flat = np.concatenate(tgt_pool)
        ss_res = float(np.sum((tgt_flat - pred_flat) ** 2))
        ss_tot = float(np.sum((tgt_flat - tgt_flat.mean()) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + 1e-16))
    else:
        r2 = float("nan")

    return {
        "loss": float(np.mean(losses)),
        "motor_max": float(np.mean(motor_maxes)),
        "motor_peak_mean": float(np.mean(peaks)),
        "direction_acc": correct / n,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# 8-panel training snapshot (port of analyze_gnn.ipynb cell 7)
# ---------------------------------------------------------------------------

def save_cortex_training_snapshot(
    stimuli: Sequence[torch.Tensor],
    preds: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    cmasks: Sequence[torch.Tensor],
    *,
    output_path: str,
    step: int,
    rule_name: str = "delaygo",
    n_show: int = 5,
    n_eachring: int = 32,
    show_title: bool = True,
) -> dict:
    """Write a multi-trial training snapshot — data-gen kinograph + pred row.

    Layout (mirrors `plot_task_cortex_traces` from connectome_gnn.plot with an
    added prediction row + two stat panels on the right):

        row 0, cols 0..n_show-1 : stimulus heatmap   (T, N_i)
        row 1, cols 0..n_show-1 : target  motor + fix (T, N_o)
        row 2, cols 0..n_show-1 : pred    motor + fix (T, N_o)
        right column (spans all 3 rows): two stacked panels
          - top: motor peak amplitude histogram   (n trials, pred vs target)
          - bot: target_active_ch vs pred_active_ch scatter + diagonal

    The aggregate stats use ALL n_trials passed in `preds` (typically 64), not
    just the n_show=5 trials shown in the heatmaps.

    Args:
        stimuli, preds, targets, cmasks: lists of N>=n_show tensors. The
            first n_show are rendered in the heatmap grid; all N are used
            for the histogram + scatter + scalar metrics.
        output_path:   destination .png.
        step:          training iteration (shown in title).
        rule_name:     cortex rule (e.g. "delaygo").
        n_show:        number of trial columns in the heatmap grid.
        n_eachring:    motor ring channel count (32 for Yang's default).

    Returns the metrics dict from compute_cortex_task_metrics.
    """
    n_trials = len(preds)
    assert n_trials == len(targets) == len(cmasks) == len(stimuli), \
        (f"stimuli/preds/targets/cmasks length mismatch: "
         f"{len(stimuli)}/{n_trials}/{len(targets)}/{len(cmasks)}")
    n_show = min(int(n_show), n_trials)
    assert n_show >= 1, f"need >=1 trial for snapshot, got n_show={n_show}"

    # Consistent font size everywhere on this figure.
    FS = 9

    # --- aggregate stats across ALL trials, restricted to the response
    # window (c_mask > 4 in Yang's lsq formulation) so the peak isn't
    # contaminated by the unsupervised post-trial tail. ---
    def _trial_peak(motor_np, cm_np):
        resp = (cm_np[:, 1:] > 4).any(axis=1)
        if not resp.any():
            return float(motor_np.max())
        return float(motor_np[resp].max())

    def _trial_argmax_ch(motor_np, cm_np):
        resp = (cm_np[:, 1:] > 4).any(axis=1)
        if not resp.any():
            return int(motor_np.max(axis=0).argmax())
        return int(motor_np[resp].max(axis=0).argmax())

    metrics = compute_cortex_task_metrics(preds, targets, cmasks)
    p_np = [p[:, 1:].detach().cpu().numpy() for p in preds]
    t_np = [t[:, 1:].detach().cpu().numpy() for t in targets]
    c_np = [c.detach().cpu().numpy() for c in cmasks]
    pred_active = np.array([_trial_argmax_ch(m, c)  for m, c in zip(p_np, c_np)])
    tgt_active  = np.array([_trial_argmax_ch(m, c)  for m, c in zip(t_np, c_np)])
    n_motor_ch = preds[0].shape[1] - 1

    # --- Bump profile for trial 0: motor amplitude vs channel at the
    # target's response peak frame. Tests whether the prediction has the
    # right bump width AND amplitude AT THE RIGHT TIME. ---
    i = 0
    mot_p0, mot_t0, cm0 = p_np[i], t_np[i], c_np[i]
    resp0 = (cm0[:, 1:] > 4).any(axis=1)
    if resp0.any():
        # peak frame within the response window
        resp_idx = np.where(resp0)[0]
        t_peak = int(resp_idx[mot_t0[resp_idx].max(axis=1).argmax()])
    else:
        t_peak = int(mot_t0.max(axis=1).argmax())
    profile_tgt = mot_t0[t_peak]
    profile_prd = mot_p0[t_peak]

    # --- figure: 3 rows × (n_show + 1) cols.
    # Use a real spacer column between the trial grid and the right stat
    # panels so the stat-panel y-axis labels don't crash into the rightmost
    # trial column.
    fig = plt.figure(figsize=(2.4 * n_show + 6.0, 8.0))
    gs = fig.add_gridspec(
        3, n_show + 2,
        width_ratios=[1.0] * n_show + [0.3, 1.8],
        hspace=0.55, wspace=0.20,
    )

    # Channel-block boundary for input (fix | mod1 | mod2 | rule).
    b1, b2 = 0.5, n_eachring + 0.5
    b3 = 2 * n_eachring + 0.5

    for col in range(n_show):
        u_b = stimuli[col].detach().cpu().numpy()     # (T, N_i)
        tgt_b = targets[col].detach().cpu().numpy()   # (T, N_o)
        prd_b = preds[col].detach().cpu().numpy()
        cm_b = cmasks[col].detach().cpu().numpy()
        nz = np.where(cm_b.sum(axis=-1) > 0)[0]
        real_T = int(nz.max() + 1) if nz.size else u_b.shape[0]
        T = u_b.shape[0]
        N_i = u_b.shape[1]

        ax_in = fig.add_subplot(gs[0, col])
        ax_in.imshow(u_b.T, aspect="auto", cmap="hot",
                     vmin=0.0, vmax=1.0, interpolation="nearest")
        for boundary in (b1, b2, b3):
            if boundary < N_i:
                ax_in.axhline(boundary, color="cyan", lw=0.6, alpha=0.7)
        if real_T < T:
            ax_in.axvspan(real_T, T, color="0.92", alpha=0.4, lw=0)
        ax_in.set_title(f"trial {col} (T={real_T})", fontsize=FS)
        ax_in.tick_params(axis="x", labelbottom=False)
        ax_in.tick_params(axis="y", labelsize=FS)
        if col == 0:
            ax_in.set_ylabel(f"input ({N_i})", fontsize=FS)
        else:
            ax_in.tick_params(axis="y", labelleft=False)

        ax_tgt = fig.add_subplot(gs[1, col], sharex=ax_in)
        ax_tgt.imshow(tgt_b.T, aspect="auto", cmap="hot",
                      vmin=0.0, vmax=0.9, interpolation="nearest")
        ax_tgt.axhline(0.5, color="cyan", lw=0.6, alpha=0.7)
        if real_T < T:
            ax_tgt.axvspan(real_T, T, color="0.92", alpha=0.4, lw=0)
        ax_tgt.tick_params(axis="x", labelbottom=False)
        ax_tgt.tick_params(axis="y", labelsize=FS)
        if col == 0:
            ax_tgt.set_ylabel(f"target ({tgt_b.shape[1]})", fontsize=FS)
        else:
            ax_tgt.tick_params(axis="y", labelleft=False)

        ax_prd = fig.add_subplot(gs[2, col], sharex=ax_in)
        ax_prd.imshow(prd_b.T, aspect="auto", cmap="hot",
                      vmin=0.0, vmax=0.9, interpolation="nearest")
        ax_prd.axhline(0.5, color="cyan", lw=0.6, alpha=0.7)
        if real_T < T:
            ax_prd.axvspan(real_T, T, color="0.92", alpha=0.4, lw=0)
        ax_prd.tick_params(axis="x", labelsize=FS)
        ax_prd.tick_params(axis="y", labelsize=FS)
        if col == 0:
            ax_prd.set_ylabel(f"pred ({prd_b.shape[1]})", fontsize=FS)
            ax_prd.set_xlabel("time", fontsize=FS)
        else:
            ax_prd.tick_params(axis="y", labelleft=False)

    # --- right column: 2 stat panels stacked, separated from trial grid by
    # the spacer column gs[:, n_show]. ---
    right_gs = gs[:, n_show + 1].subgridspec(2, 1, hspace=0.40)
    ax_prof = fig.add_subplot(right_gs[0])
    ax_prof.plot(profile_tgt, "k-", lw=1.5, label="target")
    ax_prof.plot(profile_prd, "r--", lw=1.5, label="pred")
    ax_prof.set_xlabel("motor channel", fontsize=FS)
    ax_prof.set_ylabel("amplitude", fontsize=FS)
    ax_prof.tick_params(labelsize=FS)
    ax_prof.legend(fontsize=FS)
    ax_prof.set_title(f"trial {i} bump profile at t={t_peak}", fontsize=FS)

    ax_sct = fig.add_subplot(right_gs[1])
    ax_sct.scatter(tgt_active, pred_active, alpha=0.3, s=14)
    ax_sct.plot([0, n_motor_ch - 1], [0, n_motor_ch - 1], "k--", alpha=0.3,
                label="y=x")
    # Linear fit: pred_active ≈ a · tgt_active + b. Use the same R² formula
    # as graph_tester.stimuli_R2 (1 − SS_res/SS_tot on the corrected fit).
    if len(tgt_active) >= 2 and tgt_active.std() > 0:
        A_fit = np.vstack([tgt_active.astype(np.float64),
                           np.ones_like(tgt_active, dtype=np.float64)]).T
        a_coeff, b_coeff = np.linalg.lstsq(
            A_fit, pred_active.astype(np.float64), rcond=None,
        )[0]
        y_fit = a_coeff * tgt_active + b_coeff
        ss_res = float(np.sum((pred_active - y_fit) ** 2))
        ss_tot = float(np.sum((pred_active - pred_active.mean()) ** 2))
        r2_sct = 1.0 - ss_res / (ss_tot + 1e-16)
        xs = np.array([0, n_motor_ch - 1], dtype=np.float64)
        ax_sct.plot(xs, a_coeff * xs + b_coeff, "r-", lw=1.0, alpha=0.7,
                    label="lin fit")
        slope = float(a_coeff)
    else:
        r2_sct = float("nan")
        slope = float("nan")
    ax_sct.set_xlabel("target active ch", fontsize=FS)
    ax_sct.set_ylabel("pred active ch", fontsize=FS)
    ax_sct.tick_params(labelsize=FS)
    ax_sct.legend(fontsize=FS - 1, loc="upper left")
    ax_sct.set_title(
        f"dir_acc={metrics['direction_acc']:.2f}  "
        f"R²={r2_sct:.3f}  slope={slope:.2f}",
        fontsize=FS,
    )

    if show_title:
        title_step = f"iter {step}" if step >= 0 else "test"
        fig.suptitle(
            f"cortex/{rule_name} — {title_step}  "
            f"(top: input, middle: target, bottom: prediction)",
            fontsize=FS + 1, y=0.995,
        )
        fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    else:
        fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return metrics


# ---------------------------------------------------------------------------
# W_rec matrix snapshot (saved each snapshot interval during training)
# ---------------------------------------------------------------------------

def save_cortex_matrix_snapshot(
    W_rec: torch.Tensor,
    *,
    output_path: str,
    step: int,
    title_suffix: str = "",
) -> None:
    """Write a W_rec heatmap to `output_path`.

    Two-panel figure:
      - left:  signed W_rec heatmap (red/blue, vmax = |W|.max())
      - right: log|W_rec| heatmap (for sparsity / structure inspection)
    Title shows iteration step, ‖W‖_F, max|W|, and density of |W| > 1e-3.
    """
    W = W_rec.detach().cpu().numpy()
    vmax = 0.2  # fixed range so snapshots are comparable across iters / runs
    frob = float(np.linalg.norm(W))
    density = float((np.abs(W) > 1e-3).mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(W, cmap="bwr", vmin=-vmax, vmax=vmax, aspect="equal")
    # New convention (after transpose removal): W_rec[j, i] = weight from
    # presynaptic neuron j onto postsynaptic neuron i.
    ax.set_xlabel("postsynaptic")
    ax.set_ylabel("presynaptic")
    fig.colorbar(im, ax=ax, shrink=0.85)
    # frob, density, title_suffix kept available in scope but no longer drawn
    # on the figure (per user request: matrix plot, no title, single panel).
    del frob, density, title_suffix

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 10-trial GT vs prediction kinograph (used by data_test_cortex_task_gnn)
# ---------------------------------------------------------------------------

def save_cortex_test_kinograph(
    stimuli: Sequence[torch.Tensor],
    preds: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    cmasks: Sequence[torch.Tensor],
    *,
    output_path: str,
    rule_name: str = "delaygo",
    n_trials: int = 10,
    n_eachring: int = 32,
) -> dict:
    """Write the test-time kinograph: same template as the training snapshot
    (3 rows: input, target, pred) but for `n_trials` consecutive test trials
    plus the same two right-side stat panels.

    Args:
        stimuli, preds, targets, cmasks: lists of >= n_trials trial tensors.
        output_path: destination .png.
        rule_name:   cortex rule name for the title.
        n_trials:    number of consecutive trials to show (default 10).

    Returns the metrics dict from compute_cortex_task_metrics over all
    `n_trials` trials.
    """
    n = min(int(n_trials), len(preds), len(targets), len(cmasks), len(stimuli))
    assert n >= 1, f"need >=1 trial, got {n}"
    return save_cortex_training_snapshot(
        stimuli[:n], preds[:n], targets[:n], cmasks[:n],
        output_path=output_path, step=-1,  # -1 signals test (no iter)
        rule_name=rule_name, n_show=n, n_eachring=n_eachring,
        show_title=False,
    )
