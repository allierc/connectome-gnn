"""Supp. Fig.: 8,000-frame autoregressive rollout of the general-form GNN
(group lasso 25) at the three model-noise levels, fold cv00, against the
noise-free simulation of the same held-out stimuli (published layout).

    python scripts/fig_rollout_3col.py                 # held-out stimuli, full connectome
    python scripts/fig_rollout_3col.py --ablation50    # 50% edge ablation (Supp. Fig.)

Top row: 12 representative cell types over frames 500-1500 (10-30 s), green
noise-free ground truth, black rollout, red the photoreceptor row's visual
input; the Fisher-z pooled per-neuron Pearson r over all 8,000 frames in the
header. Bottom row: rollout against noise-free voltage over all (neuron, frame)
pairs as a log-density image, range [-10, 10].
With --ablation50 the bundles are results/rollout_bundle_on_noise_free_mask_50.npz:
the tester zeroed the masked half of the learned W (ablation_mask.pt of
flyvis_noise_free_mask_50) and rolled out on that dataset's noise-free
simulation of the ablated circuit.
Data: <GNN_OUTPUT_ROOT>/log/fly/flyvis_noise_{free,005,05}_blank50_condl25_cv00/results/

Output: figures/fig_rollout_3col_noise_comparison[_ablation50].{pdf,png}
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (CM, COLOR_GT, COLOR_PRED, COLOR_STIM, FIG_DIR, FS_ANNOT, FS_TICK, LOG_ROOT,  # noqa: E402
                         column_titles, fisher_r, panel_labels, save, scatter_density)

BLOCKS = [
    ("noise-free ($\\sigma = 0$)", "flyvis_noise_free_blank50_condl25_cv00"),
    ("low model noise ($\\sigma = 0.05$)", "flyvis_noise_005_blank50_condl25_cv00"),
    ("high model noise ($\\sigma = 0.5$)", "flyvis_noise_05_blank50_condl25_cv00"),
]
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]
T0, T1, DT_MS = 500, 1500, 20.0
V_RANGE = (-10.0, 10.0)
LW_GT, LW_PRED, LW_STIM = 1.0, 0.4, 0.6
BUNDLE = {"": "rollout_bundle.npz", "ablation50": "rollout_bundle_on_noise_free_mask_50.npz"}
WIDTH_CM, COL_W, TRACE_H, SCAT_S = 18.0, 5.0, 5.0, 4.0
LEFT, GAP_COL, GAP_ROW, BOTTOM, TOP_PAD = 1.1, 0.6, 1.4, 0.9, 0.9


def load_bundle(run, variant=""):
    # the ablation bundles are kept in <run>/ablation50/: the tester clears results/*_on_* at each new test
    path = os.path.join(LOG_ROOT, run, "ablation50" if variant else "results", BUNDLE[variant])
    if not os.path.exists(path):
        path = os.path.join(LOG_ROOT, run, "results", BUNDLE[variant])
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
    ax.set_xticklabels([f"{v:.0f}" for v in ax.get_xticks()])
    if show_labels:
        ax.set_xlabel("time (ms)"); ax.set_ylabel("neurons", labelpad=14)
    ax.set_ylim(-1.7 * step, n * step)
    ax.text(0.02, 1.0, header, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_ANNOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation50", action="store_true")
    variant = "ablation50" if ap.parse_args().ablation50 else ""
    bundles = [load_bundle(r, variant) for _, r in BLOCKS]
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
    truth = "noise-free ablated" if variant else "noise-free"
    for k, (b, (title, _)) in enumerate(zip(bundles, BLOCKS)):
        x = LEFT + k * (COL_W + GAP_COL)
        r, sd = fisher_r(b["activity_true"], b["activity_pred"])
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
    column_titles(fig, [[a] for a in top], [t for t, _ in BLOCKS], dy_pt=12)
    panel_labels(fig, top + bot, dy_pt=4)
    save(fig, os.path.join(FIG_DIR, "fig_rollout_3col_noise_comparison" + ("_" + variant if variant else "")))


if __name__ == "__main__":
    main()
