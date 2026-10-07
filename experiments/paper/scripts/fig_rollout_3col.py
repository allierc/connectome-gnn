"""Supp. Fig.: 8,000-frame autoregressive rollout of the general-form GNN
(group lasso 25) at the three model-noise levels, fold cv00, against the
noise-free simulation of the same held-out stimuli.

    python scripts/fig_rollout_3col.py

Top row: 12 representative cell types over a 20 s window (frames 500-1500 at
20 ms). Green: noise-free ground truth; black: rollout; red: the visual input
of the one photoreceptor row. Bottom row: rollout voltage against the
noise-free voltage, every (neuron, frame) pair subsampled to 300,000 points,
with the per-neuron Pearson r pooled in Fisher z over all 8,000 frames.
Data: <GNN_OUTPUT_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_condl25_cv00/
results/rollout_bundle.npz (activity_true is the noise-free twin's trajectory).

Output: figures/fig_rollout_3col_noise_comparison.{pdf,png}
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (CM, COLOR_GT, COLOR_PRED, COLOR_STIM, FIG_DIR, FS_ANNOT, FS_TICK, LOG_ROOT,  # noqa: E402
                         column_titles, panel_labels, pretty_ticks, save, trim_axis)

BLOCKS = [
    ("noise-free ($\\sigma = 0$)", "flyvis_noise_free_blank50_condl25_cv00"),
    ("low model noise ($\\sigma = 0.05$)", "flyvis_noise_005_blank50_condl25_cv00"),
    ("high model noise ($\\sigma = 0.5$)", "flyvis_noise_05_blank50_condl25_cv00"),
]
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]   # as the published figure
T0, T1, DT_MS = 500, 1500, 20.0
N_SCATTER = 300_000
LW_GT, LW_PRED, LW_STIM = 1.0, 0.4, 0.5
# geometry in cm: three columns of width COL_W; traces TRACE_H high, scatters square SCAT_S
WIDTH_CM, COL_W, TRACE_H, SCAT_S = 18.0, 5.0, 4.8, 3.4
LEFT, GAP_COL, GAP_ROW, BOTTOM, TOP_PAD = 1.1, 0.6, 1.3, 0.9, 0.6


def load_bundle(run):
    b = np.load(os.path.join(LOG_ROOT, run, "results", "rollout_bundle.npz"), allow_pickle=True)
    return {k: b[k] for k in ("activity_true", "activity_pred", "stimulus", "type_ids", "type_names")}


def fisher_r(true, pred):
    """Per-neuron Pearson r pooled in Fisher z (the tables' recipe); returns (r, sd)."""
    t = true - true.mean(1, keepdims=True); p = pred - pred.mean(1, keepdims=True)
    st, sp = t.std(1), p.std(1)
    ok = (st > 1e-8) & (sp > 1e-8)
    r = (t[ok] * p[ok]).mean(1) / (st[ok] * sp[ok])
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return float(np.tanh(z.mean())), float((np.tanh(z.mean() + z.std()) - np.tanh(z.mean() - z.std())) / 2)


def draw_traces(ax, b, idx, labels, step, show_labels):
    t = np.arange(T0, T1 + 1) * DT_MS / 1000.0
    for k, i in enumerate(idx):
        off = -k * step
        tr, pr = b["activity_true"][i, T0:T1 + 1], b["activity_pred"][i, T0:T1 + 1]
        base = tr.mean()
        ax.plot(t, tr - base + off, color=COLOR_GT, lw=LW_GT)
        ax.plot(t, pr - base + off, color=COLOR_PRED, lw=LW_PRED)
        st = b["stimulus"][i, T0:T1 + 1]
        if np.abs(st).max() > 0:
            ax.plot(t, (st - st.mean()) / (st.std() + 1e-9) * 0.25 * step + off, color=COLOR_STIM, lw=LW_STIM)
    ax.set_yticks([-k * step for k in range(len(idx))])
    ax.set_yticklabels(labels if show_labels else [""] * len(idx), fontsize=FS_TICK)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks(pretty_ticks(t[0], t[-1], 5))
    ax.set_xlabel("time (s)")
    ax.spines["bottom"].set_bounds(t[0], t[-1])
    ax.set_ylim(-(len(idx) - 0.5) * step, 1.0 * step)


def draw_scatter(ax, b, show_ylabel):
    x = b["activity_true"].ravel(); y = b["activity_pred"].ravel()
    rng = np.random.default_rng(0)
    sel = rng.choice(len(x), N_SCATTER, replace=False)
    lo, hi = np.percentile(x, [0.01, 99.99])
    lo, hi = np.floor(lo), np.ceil(hi)
    ax.plot([lo, hi], [lo, hi], color="0.35", lw=0.5, ls="--")
    ax.scatter(x[sel], y[sel], s=0.3, alpha=0.25, color="k", lw=0, rasterized=True)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    tk = pretty_ticks(lo, hi, 3)
    ax.set_xticks(tk); ax.set_yticks(tk)
    ax.set_xlabel("noise-free voltage")
    if show_ylabel:
        ax.set_ylabel("rollout voltage")
    r, sd = fisher_r(b["activity_true"], b["activity_pred"])
    ax.text(0.03, 0.97, f"$r$ = {r:.2f} $\\pm$ {sd:.2f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_ANNOT)
    trim_axis(ax)


def main():
    bundles = [load_bundle(r) for _, r in BLOCKS]
    types = bundles[0]["type_ids"]; names = list(bundles[0]["type_names"])
    idx, labels = [], []
    for tp in SELECTED_TYPES:
        ids = np.where(types == tp)[0]
        if len(ids):
            idx.append(int(ids[0])); labels.append(names[tp])
    step = 3.0 * max(float(np.std(b["activity_true"][idx, T0:T1])) for b in bundles)

    height = BOTTOM + SCAT_S + GAP_ROW + TRACE_H + TOP_PAD
    fig = plt.figure(figsize=(WIDTH_CM * CM, height * CM))

    def rect(x_cm, y_cm, w_cm, h_cm):
        return [x_cm / WIDTH_CM, y_cm / height, w_cm / WIDTH_CM, h_cm / height]

    top, bot = [], []
    for k, (b, (title, _)) in enumerate(zip(bundles, BLOCKS)):
        x = LEFT + k * (COL_W + GAP_COL)
        ax = fig.add_axes(rect(x, BOTTOM + SCAT_S + GAP_ROW, COL_W, TRACE_H))
        draw_traces(ax, b, idx, labels, step, show_labels=(k == 0)); top.append(ax)
        ax = fig.add_axes(rect(x, BOTTOM, SCAT_S, SCAT_S))
        draw_scatter(ax, b, show_ylabel=(k == 0)); bot.append(ax)
    column_titles(fig, [[a] for a in top], [t for t, _ in BLOCKS], dy_pt=4)
    panel_labels(fig, top + bot)
    save(fig, os.path.join(FIG_DIR, "fig_rollout_3col_noise_comparison"))


if __name__ == "__main__":
    main()
