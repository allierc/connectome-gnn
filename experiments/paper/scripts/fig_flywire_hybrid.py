"""Fig. 2: the hybrid Flyvis-FlyWire connectome figure, with panels e and f
redrawn from the general-form GNN (group lasso 25) of experiment 7.

    python scripts/fig_flywire_hybrid.py

Panels a-d are the published rendering (figures/fig_flywire_hybrid.png: the
3-D coregistration, the per-column FlyWire kernels and the FlyWire-eye vs
Flyvis simulation rollouts have no script in this repository) cropped from
that PNG. Panels e and f are drawn here: the GNN's 8,000-frame rollout
(black) against the simulated voltage (green) for the six example cell types
of the published figure over a 20 s window, on the FlyWire eye (e) and on the
FlyWire eye with proximal null edges (f), fold cv00 of experiment 7; the
Fisher-z pooled Pearson r over all neurons and frames is printed.
Data: <GNN_OUTPUT_ROOT>/log/fly/full_eye_flywireRF[_proximal_nulls]_noise_005_blank50_condl25_cv00/results/rollout_bundle.npz

Output: figures/fig_flywire_hybrid_new.{pdf,png}
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (CM, COLOR_GT, COLOR_PRED, FIG_DIR, FS_ANNOT, FS_LABEL, FS_PANEL, FS_TICK, LOG_ROOT,  # noqa: E402
                         fisher_r, save)
from fig_rollout_3col import T0, T1, DT_MS  # noqa: E402

PUBLISHED = os.path.join(FIG_DIR, "fig_flywire_hybrid.png")
PANELS = [
    ("e", "FlyWire eye model inferred", "full_eye_flywireRF_noise_005_blank50_condl25_cv00",
     "total: 50,412 neurons, 1,266,378 edges"),
    ("f", "FlyWire eye + null edges (n.e.)", "full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00",
     "total: 50,412 neurons, 9,642,335 edges"),
]
TYPES = ["T5a", "Mi9", "Mi1", "Tm9", "T4c", "Tm3"]          # the published panels' examples, top to bottom
ROW_SPLIT = 0.675                                            # fraction of the PNG height where row d-f starts
D_SPLIT = 0.335                                              # fraction of the PNG width panel d spans
WIDTH = 18.0


# fonts of the published PNG once scaled to the page: its titles are about 4.5 pt,
# tick labels 4 pt, letters 7 pt bold -- the new panels use the same sizes
FS_T, FS_K, FS_L = 4.5, 4.0, 10.0   # letters: the published 'd' is 0.27 cm tall at page width, i.e. 10 pt bold


def traces(ax, b, idx, labels, step, title, footer, r, sd):
    t = np.arange(T0, T1 + 1) * DT_MS
    n = len(idx)
    for k, i in enumerate(idx):
        off = (n - 1 - k) * step
        tr, pr = b["activity_true"][i, T0:T1 + 1], b["activity_pred"][i, T0:T1 + 1]
        base = tr.mean()
        ax.plot(t, tr - base + off, color=COLOR_GT, lw=0.8)
        ax.plot(t, pr - base + off, color=COLOR_PRED, lw=0.35)
    ax.set_yticks([(n - 1 - k) * step for k in range(n)]); ax.set_yticklabels(labels, fontsize=FS_K)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0], t[-1]); ax.set_xticks([T0 * DT_MS, (T0 + T1) / 2 * DT_MS, T1 * DT_MS])
    ax.set_xticklabels([f"{v:.0f}" for v in ax.get_xticks()], fontsize=FS_K)
    ax.tick_params(axis="x", length=1.5, width=0.4, pad=1)
    ax.set_xlabel("time (ms)", fontsize=FS_K, labelpad=1)
    ax.set_ylim(-0.8 * step, (n - 0.2) * step)
    ax.set_title(title, fontsize=FS_T, pad=2, loc="left", x=0.12)
    ax.text(0.0, -0.36, footer, transform=ax.transAxes, ha="left", va="top", fontsize=FS_K)
    ax.text(1.0, 1.0, f"$r$ = {r:.2f} $\\pm$ {sd:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_K)


def main():
    img = np.asarray(Image.open(PUBLISHED).convert("RGB"))
    H, W = img.shape[:2]
    top = img[: int(ROW_SPLIT * H)]
    d_panel = img[int(ROW_SPLIT * H):, : int(D_SPLIT * W)]
    top_h = WIDTH * top.shape[0] / top.shape[1]                 # cm, at full width
    d_w = WIDTH * D_SPLIT
    row_h = d_w * d_panel.shape[0] / d_panel.shape[1]
    gap, bottom = 0.4, 0.9
    height = bottom + row_h + gap + top_h + 0.2
    fig = plt.figure(figsize=(WIDTH * CM, height * CM))

    def rect(x, y, w, h):
        return [x / WIDTH, y / height, w / WIDTH, h / height]

    ax = fig.add_axes(rect(0, height - 0.2 - top_h, WIDTH, top_h)); ax.imshow(top); ax.set_axis_off()
    ax = fig.add_axes(rect(0, bottom, d_w, row_h)); ax.imshow(d_panel); ax.set_axis_off()
    x = d_w + 0.9
    pw = (WIDTH - x - 0.3 - 1.1) / 2
    for k, (letter, title, run, footer) in enumerate(PANELS):
        b = np.load(os.path.join(LOG_ROOT, run, "results", "rollout_bundle.npz"), allow_pickle=True)
        names = list(b["type_names"]); types = b["type_ids"]
        idx = [int(np.where(types == names.index(nm))[0][0]) for nm in TYPES]
        step = 2.5 * float(np.std(b["activity_true"][idx, T0:T1]))
        r, sd = fisher_r(b["activity_true"], b["activity_pred"])
        axp = fig.add_axes(rect(x + k * (pw + 1.1), bottom + 0.15, pw, row_h - 0.6))
        traces(axp, b, idx, TYPES, step, title, footer, r, sd)
        fig.text((x + k * (pw + 1.1) - 0.6) / WIDTH, (bottom + row_h - 0.05) / height, letter, fontsize=FS_L,
                 fontweight="bold", ha="right", va="top")
        print(f"{letter} {run}: r {r:.3f} +- {sd:.3f}")
    # the "simulated / inferred" legend under each of e and f, right-aligned on the footer line
    for k in range(len(PANELS)):
        xr = (x + k * (pw + 1.1) + pw) / WIDTH
        fig.text(xr, (bottom - 0.6) / height, "inferred", color=COLOR_PRED, ha="right", va="top", fontsize=FS_K)
        fig.text(xr - 0.045, (bottom - 0.6) / height, "simulated", color=COLOR_GT, ha="right", va="top", fontsize=FS_K)
    save(fig, os.path.join(FIG_DIR, "fig_flywire_hybrid_new"))


if __name__ == "__main__":
    main()
