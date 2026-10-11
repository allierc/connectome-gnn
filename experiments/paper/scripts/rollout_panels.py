"""Shared drawing for the three rollout figures (one script each):

    fig_rollout_3col.py         held-out stimuli, general-form GNN    (Supp. Fig.)
    fig_rollout_ablation50.py   50% edge ablation, same GNN           (Supp. Fig.)
    fig_rollout_known_ode.py    the Known-ODE oracle (experiment 15)  (Supp. Fig.)

Each column: 12 representative cell types over frames 500-1500 (10-30 s), green
ground truth, black rollout, red the photoreceptor row's visual input; below,
rollout against ground-truth voltage over all (neuron, frame) pairs as a
log-density image, range [-10, 10]. The r printed in each column is the
TESTER's -- "Pearson r: <mean> +/- <sd>" of the rollout log -o test_plot
wrote (per-neuron Pearson r pooled by utils.fisher_pool) -- not recomputed.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from paper_style import (CM, COLOR_GT, COLOR_PRED, COLOR_STIM, FIG_DIR, FS_ANNOT, FS_TICK,
                         column_titles, panel_labels, save, scatter_density, tester_rollout_r)

TITLES = ["noise-free ($\\sigma = 0$)", "low model noise ($\\sigma = 0.05$)", "high model noise ($\\sigma = 0.5$)"]
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]
T0, T1, DT_MS = 500, 1500, 20.0
V_RANGE = (-10.0, 10.0)
LW_GT, LW_PRED, LW_STIM = 1.0, 0.4, 0.6
WIDTH_CM, COL_W, TRACE_H, SCAT_S = 18.0, 5.0, 5.0, 4.0
LEFT, GAP_COL, GAP_ROW, BOTTOM, TOP_PAD = 1.1, 0.6, 1.4, 0.9, 0.9


def load_bundle(path):
    b = np.load(path, allow_pickle=True)
    return {k: b[k] for k in ("activity_true", "activity_pred", "stimulus", "type_ids", "type_names")}


def draw_traces(ax, b, idx, labels, step, header, show_labels):
    """Traces stacked bottom-up as in the published figure (R1 at the bottom,
    the stimulus trace below it)."""
    t = np.arange(T0, T1 + 1) * DT_MS
    n = len(idx)
    for k, i in enumerate(idx):
        off = (n - 1 - k) * step
        tr, pr = b["activity_true"][i, T0:T1 + 1], b["activity_pred"][i, T0:T1 + 1]
        base = tr.mean()
        ax.plot(t, tr - base + off, color=COLOR_GT, lw=LW_GT)
        ax.plot(t, pr - base + off, color=COLOR_PRED, lw=LW_PRED)
    st = b["stimulus"][idx[0], T0:T1 + 1]          # the photoreceptor row's input
    ax.plot(t, (st - st.mean()) / (st.std() + 1e-9) * 0.2 * step - step, color=COLOR_STIM, lw=LW_STIM)
    ax.set_yticks([(n - 1 - k) * step for k in range(n)])
    ax.set_yticklabels(labels if show_labels else [""] * n, fontsize=FS_TICK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0], t[-1]); ax.set_xticks([T0 * DT_MS, (T0 + T1) / 2 * DT_MS, T1 * DT_MS])
    # time axis labelled on the first column only, as in the published figure
    ax.set_xticklabels([f"{v:.0f}" for v in ax.get_xticks()] if show_labels else [])
    if show_labels:
        ax.set_xlabel("time (ms)"); ax.set_ylabel("neurons", labelpad=14)
    ax.set_ylim(-1.7 * step, n * step)
    ax.text(0.02, 1.0, header, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_ANNOT)


def draw_figure(bundle_paths, rollout_logs, out_name, truth="noise-free"):
    """bundle_paths / rollout_logs: one per column (sigma 0, 0.05, 0.5)."""
    bundles = [load_bundle(p) for p in bundle_paths]
    stats = [tester_rollout_r(p) for p in rollout_logs]
    types = bundles[0]["type_ids"]; names = list(bundles[0]["type_names"])
    idx, labels = [], []
    for tp in SELECTED_TYPES:
        ids = np.where(types == tp)[0]
        if len(ids):
            idx.append(int(ids[0])); labels.append(names[tp])
    step = 3.0 * max(float(np.std(b["activity_true"][idx, T0:T1])) for b in bundles)

    height = BOTTOM + SCAT_S + GAP_ROW + TRACE_H + TOP_PAD
    fig = plt.figure(figsize=(WIDTH_CM * CM, height * CM))

    def rect(x, y, w, h):
        return [x / WIDTH_CM, y / height, w / WIDTH_CM, h / height]

    top, bot = [], []
    for k, (b, (r, sd)) in enumerate(zip(bundles, stats)):
        x = LEFT + k * (COL_W + GAP_COL)
        header = ("" if k == 0 else f"vs {truth}, ") + f"$r$ = {r:.2f} $\\pm$ {sd:.2f}"
        ax = fig.add_axes(rect(x, BOTTOM + SCAT_S + GAP_ROW, COL_W, TRACE_H))
        draw_traces(ax, b, idx, labels, step, header, show_labels=(k == 0)); top.append(ax)
        ax = fig.add_axes(rect(x, BOTTOM, SCAT_S, SCAT_S))
        scatter_density(ax, b["activity_true"], b["activity_pred"], *V_RANGE)
        ax.text(0.04, 0.96, f"$r$ = {r:.2f} $\\pm$ {sd:.2f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_ANNOT)
        if k == 0:
            ax.set_xlabel(f"{truth} voltage"); ax.set_ylabel("rollout voltage")
        else:
            ax.set_title(f"vs {truth}", fontsize=FS_TICK, pad=2)
        bot.append(ax)
        print(f"column {k}: tester r {r:.3f} +- {sd:.3f}, {b['activity_true'].shape[1]} frames")
    column_titles(fig, [[a] for a in top], TITLES, dy_pt=12)
    panel_labels(fig, top + bot, dy_pt=4, y_from="tight")
    save(fig, os.path.join(FIG_DIR, out_name))
