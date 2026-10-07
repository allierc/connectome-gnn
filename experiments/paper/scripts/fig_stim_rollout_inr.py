"""Supp. Fig.: joint GNN + visual SIREN (unknown stimulus) with the
general-form GNN (group lasso 25), one-step training, sigma = 0.05, fold cv00
(experiment 11, run flyvis_noise_005_INR_davis_blank50_condl251s_cv00).

    python scripts/fig_stim_rollout_inr.py

a  the visual stimulus on the 217-column photoreceptor lattice (R1), every
   80 ms: true (top), SIREN (middle), residual (bottom), z-scored per frame
b  SIREN stimulus against the true one for 12 photoreceptors, 20 s
c  GNN voltage rollout against the noise-free voltage, 12 cell types, same window
d  SIREN against true stimulus, all (photoreceptor, frame) pairs (subsampled)
e  rollout against noise-free voltage, all (neuron, frame) pairs (subsampled)
The rollout runs over the first 8,000 TRAINING frames (the SIREN is a function
of time and cannot be queried outside the frames it was trained on). The SIREN
output is shown after the affine gauge correction stored by the tester
(stimulus_input_pred_corrected: the raw output is defined up to the sign and
scale that the GNN's input weights absorb).

Output: figures/fig_stim_rollout_inr.{pdf,png}
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (CM, COLOR_GT, COLOR_PRED, DATA_ROOT, FIG_DIR, FS_ANNOT, FS_TICK, LOG_ROOT,  # noqa: E402
                         panel_labels, pretty_ticks, save, trim_axis)
from fig_rollout_3col import SELECTED_TYPES, T0, T1, DT_MS, fisher_r  # noqa: E402

RUN = "flyvis_noise_005_INR_davis_blank50_condl251s_cv00"
DATASET = "flyvis_noise_005_INR_davis_blank50_cv00"
R1_TYPE = 23                       # photoreceptor R1: 217 columns, one per lattice point
HEX_STEP, HEX_N = 4, 6               # six frames 80 ms apart, from the first 2-s stretch
HEX_MIN_SD = 0.15                     # of the window in which every frame has stimulus contrast
N_SCATTER = 300_000
WIDTH = 18.0


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def hex_map(ax, xy, vals, vmax=2.5):
    ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=9, marker="h", lw=0)
    ax.set_aspect("equal"); ax.set_axis_off()


def traces(ax, true, pred, labels, t, step, title_y=True):
    for k in range(len(labels)):
        base = true[k].mean(); off = -k * step
        ax.plot(t, true[k] - base + off, color=COLOR_GT, lw=1.0)
        ax.plot(t, pred[k] - base + off, color=COLOR_PRED, lw=0.4)
    ax.set_yticks([-k * step for k in range(len(labels))]); ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0], t[-1]); ax.set_xticks(pretty_ticks(t[0], t[-1], 5)); ax.spines["bottom"].set_bounds(t[0], t[-1])
    ax.set_xlabel("time (s)"); ax.set_ylim(-(len(labels) - 0.5) * step, step)


def scatter(ax, x, y, xlabel, ylabel, annot):
    rng = np.random.default_rng(0)
    sel = rng.choice(x.size, min(N_SCATTER, x.size), replace=False)
    lo, hi = np.floor(np.percentile(x, 0.01)), np.ceil(np.percentile(x, 99.99))
    ax.plot([lo, hi], [lo, hi], color="0.35", lw=0.5, ls="--")
    ax.scatter(x.ravel()[sel], y.ravel()[sel], s=0.3, alpha=0.25, color="k", lw=0, rasterized=True)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); tk = pretty_ticks(lo, hi, 3); ax.set_xticks(tk); ax.set_yticks(tk)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.text(0.97, 0.03, annot, transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_ANNOT)
    trim_axis(ax)


def main():
    b = np.load(os.path.join(LOG_ROOT, RUN, "results", "rollout_bundle.npz"), allow_pickle=True)
    types = b["type_ids"]; names = list(b["type_names"])
    st, sp = b["stimulus_input_true"], b["stimulus_input_pred_corrected"]        # (T, 1736)
    pos = np.asarray(zarr.open(os.path.join(DATA_ROOT, DATASET, "x_list_train", "pos.zarr"), mode="r")[:])
    r1 = np.where(types[:1736] == R1_TYPE)[0]                                  # input neurons are the first 1736
    assert len(r1) == 217, len(r1)
    xy = pos[r1]
    t = np.arange(T0, T1 + 1) * DT_MS / 1000.0
    # ---- geometry (cm) ----
    hex_h, tr_h, sc_s = 3.3, 4.6, 3.4
    left, gap_r, bottom, top_pad = 1.1, 1.3, 0.9, 0.5
    height = bottom + sc_s + gap_r + tr_h + gap_r + hex_h + top_pad
    fig = plt.figure(figsize=(WIDTH * CM, height * CM))

    def rect(x, y, w, h):
        return [x / WIDTH, y / height, w / WIDTH, h / height]

    # a: hex frames, 3 rows x HEX_N columns, drawn in one axes region each
    y_hex = bottom + sc_s + gap_r + tr_h + gap_r
    cell = (WIDTH - left - 0.4) / HEX_N
    sd = st[:, r1].std(1)
    f0 = next(f for f in range(T0, T1) if (sd[f:f + HEX_STEP * HEX_N:HEX_STEP] > HEX_MIN_SD).all())
    frames = [f0 + k * HEX_STEP for k in range(HEX_N)]
    hex_axes = []
    row_labels = ["true", "SIREN", "residual"]
    for r in range(3):
        for k, f in enumerate(frames):
            ax = fig.add_axes(rect(left + k * cell, y_hex + (2 - r) * hex_h / 3, cell * 0.95, hex_h / 3 * 0.95))
            tv, pv = st[f, r1], sp[f, r1]
            v = [zscore(tv), zscore(pv), zscore(tv) - zscore(pv)][r]
            hex_map(ax, xy, v)
            if r == 0:
                ax.set_title(f"{f * DT_MS / 1000:.2f} s", fontsize=FS_TICK, pad=1)
            if k == 0:
                ax.text(-0.05, 0.5, row_labels[r], transform=ax.transAxes, ha="right", va="center", fontsize=FS_TICK)
            hex_axes.append(ax)
    # b, c: traces
    y_tr = bottom + sc_s + gap_r
    col_w = (WIDTH - left - 0.4 - 1.6) / 2
    ax_b = fig.add_axes(rect(left, y_tr, col_w, tr_h))
    sel12 = r1[np.linspace(0, len(r1) - 1, 12).astype(int)]
    step_s = 3.0 * float(np.std(st[T0:T1, sel12]))
    traces(ax_b, st[T0:T1 + 1, sel12].T, sp[T0:T1 + 1, sel12].T, [f"R1 c{int(i)}" for i in range(12)], t, step_s)
    idx, labels = [], []
    for tp in SELECTED_TYPES:
        ids = np.where(types == tp)[0]
        if len(ids):
            idx.append(int(ids[0])); labels.append(names[tp])
    ax_c = fig.add_axes(rect(left + col_w + 1.6, y_tr, col_w, tr_h))
    step_v = 3.0 * float(np.std(b["activity_true"][idx, T0:T1]))
    traces(ax_c, b["activity_true"][idx, T0:T1 + 1], b["activity_pred"][idx, T0:T1 + 1], labels, t, step_v)
    # d, e: scatters
    ax_d = fig.add_axes(rect(left, bottom, sc_s, sc_s))
    r_s = np.corrcoef(st.ravel(), sp.ravel())[0, 1]
    scatter(ax_d, st, sp, "true stimulus", "SIREN stimulus", f"$r$ = {r_s:.2f}")
    ax_e = fig.add_axes(rect(left + col_w + 1.6, bottom, sc_s, sc_s))
    r, sd = fisher_r(b["activity_true"], b["activity_pred"])
    scatter(ax_e, b["activity_true"], b["activity_pred"], "noise-free voltage", "rollout voltage",
            f"$r$ = {r:.2f} $\\pm$ {sd:.2f}")
    panel_labels(fig, [hex_axes[0], ax_b, ax_c, ax_d, ax_e], align_rows=False)
    save(fig, os.path.join(FIG_DIR, "fig_stim_rollout_inr"))


if __name__ == "__main__":
    main()
