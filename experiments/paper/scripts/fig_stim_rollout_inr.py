"""Supp. Fig.: joint GNN + visual SIREN (unknown stimulus) with the
general-form GNN (group lasso 25), one-step training, sigma = 0.05, fold cv00
(experiment 11, run flyvis_noise_005_INR_davis_blank50_condl251s_cv00), in the
published layout.

    python scripts/fig_stim_rollout_inr.py

a  the visual stimulus on the 217-column R1 lattice, ten frames 80 ms apart
   from 10,000 ms: ground truth, learned (SIREN), residual; z-scored per frame
b  SIREN against true stimulus for 12 photoreceptors (R1-R8 of one column, R1-R4
   of a second), 10-30 s; header: per-photoreceptor Pearson r, Fisher-z pooled, mean +- SD
c  GNN voltage rollout against the noise-free voltage, 12 cell types; same r over all neurons
The rollout runs over the first 8,000 TRAINING frames (the SIREN cannot be
queried outside the frames it was trained on); the SIREN output is shown with its mean and SD
matched to the true stimulus, sign from the correlation (its raw output is
defined up to the sign and scale the GNN's input weights absorb; the tester's
least-squares correction would compress the range by the factor r).

Output: figures/fig_stim_rollout_inr.{pdf,png}
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import (CM, COLOR_GT, COLOR_PRED, DATA_ROOT, FIG_DIR, FS_ANNOT, FS_LABEL, FS_PANEL, FS_TICK, LOG_ROOT,  # noqa: E402
                         fisher_r, panel_labels, save, scatter_density)
from fig_rollout_3col import SELECTED_TYPES, T0, T1, DT_MS  # noqa: E402

RUN = "flyvis_noise_005_INR_davis_blank50_condl251s_cv00"
DATASET = "flyvis_noise_005_INR_davis_blank50_cv00"
R_TYPES = list(range(23, 31))                  # R1..R8
HEX_T0, HEX_STEP, HEX_N, HEX_VMAX = 500, 4, 10, 3.0
WIDTH = 18.0


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-9)


def hex_map(ax, xy, vals):
    ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap="RdBu_r", vmin=-HEX_VMAX, vmax=HEX_VMAX, s=7, marker="h",
               edgecolors="0.6", linewidths=0.15)
    ax.set_aspect("equal"); ax.set_axis_off()


def traces(ax, true, pred, labels, step, header, xlabel):
    t = np.arange(T0, T1 + 1) * DT_MS
    n = len(labels)
    for k in range(n):
        base = true[k].mean(); off = (n - 1 - k) * step
        ax.plot(t, true[k] - base + off, color=COLOR_GT, lw=1.0)
        ax.plot(t, pred[k] - base + off, color=COLOR_PRED, lw=0.4)
    ax.set_yticks([(n - 1 - k) * step for k in range(n)]); ax.set_yticklabels(labels, fontsize=FS_TICK)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(t[0], t[-1]); ax.set_xticks([T0 * DT_MS, (T0 + T1) / 2 * DT_MS, T1 * DT_MS])
    ax.set_xticklabels([f"{v:.0f}" for v in ax.get_xticks()]); ax.set_xlabel(xlabel)
    ax.set_ylim(-0.8 * step, n * step)
    ax.text(0.02, 1.0, header, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_ANNOT)


def main():
    b = np.load(os.path.join(LOG_ROOT, RUN, "results", "rollout_bundle.npz"), allow_pickle=True)
    types = b["type_ids"]; names = list(b["type_names"])
    st, raw = b["stimulus_input_true"], b["stimulus_input_pred"]                  # (T, 1736)
    # THE GAUGE: the SIREN output is defined up to the sign and scale the GNN's input
    # weights absorb. The tester's stimulus_input_pred_corrected is the least-squares
    # fit of true on raw, whose slope is r x SD(true)/SD(raw): at r = 0.90 it compresses
    # the learned range to 0.68 of the true one (the flat top of the old panel c). Shown
    # instead with its mean and SD matched to the true stimulus, sign from r; r itself
    # is unchanged by either choice.
    sign = np.sign(np.corrcoef(st.ravel(), raw.ravel())[0, 1])
    sp = st.mean() + sign * (raw - raw.mean()) * (st.std() / raw.std())
    pos = np.asarray(zarr.open(os.path.join(DATA_ROOT, DATASET, "x_list_train", "pos.zarr"), mode="r")[:])
    in_types = types[:1736]
    r1 = np.where(in_types == R_TYPES[0])[0]
    xy = pos[r1]
    r_stim = float(np.corrcoef(st.ravel(), sp.ravel())[0, 1])
    r_v, sd_v = fisher_r(b["activity_true"], b["activity_pred"])
    # 12 photoreceptors: R1..R8 of the first column, R1..R4 of a second, as the published figure
    cols = {tp: np.where(in_types == tp)[0] for tp in R_TYPES}
    sel = [cols[tp][0] for tp in R_TYPES] + [cols[tp][len(r1) // 2] for tp in R_TYPES[:4]]
    sel_lab = [names[tp] for tp in R_TYPES] + [names[tp] for tp in R_TYPES[:4]]
    idx, labels = [], []
    for tp in SELECTED_TYPES:
        ids = np.where(types == tp)[0]
        if len(ids):
            idx.append(int(ids[0])); labels.append(names[tp])

    # ---- geometry (cm) ----
    hex_row, hex_title, tr_h, sc_s = 1.5, 0.45, 3.6, 3.6
    left, right_w, gap, gap_tr, bottom, top_pad = 1.0, sc_s, 0.9, 1.5, 0.8, 0.4
    tr_w = WIDTH - left - 0.4
    height = bottom + 2 * tr_h + gap_tr + 1.3 + 3 * (hex_row + hex_title) + top_pad
    fig = plt.figure(figsize=(WIDTH * CM, height * CM))

    def rect(x, y, w, h):
        return [x / WIDTH, y / height, w / WIDTH, h / height]

    # a: hex frames
    y_hex0 = bottom + 2 * tr_h + gap_tr + 1.3
    cell = (WIDTH - left - 1.6) / HEX_N
    frames = [HEX_T0 + k * HEX_STEP for k in range(HEX_N)]
    row_titles = ["ground truth visual stimulus", "learned visual stimulus", "residual (learned $-$ ground truth)"]
    hex_axes = []
    for r in range(3):
        y = y_hex0 + (2 - r) * (hex_row + hex_title)
        fig.text(left / WIDTH, (y + hex_row + 0.12) / height, row_titles[r], ha="left", va="bottom", fontsize=FS_LABEL)
        for k, f in enumerate(frames):
            ax = fig.add_axes(rect(left + k * cell, y, cell * 0.96, hex_row * 0.92))
            tv, pv = zscore(st[f, r1]), zscore(sp[f, r1])
            hex_map(ax, xy, [tv, pv, pv - tv][r])
            if r == 2 and k in (0, HEX_N - 1):     # time stamps under the last row, clear of the row titles
                ax.text(0.5, -0.04, f"t = {f * DT_MS:.0f} ms", transform=ax.transAxes, ha="center", va="top",
                        fontsize=FS_TICK)
            hex_axes.append(ax)
    cax = fig.add_axes(rect(WIDTH - 1.3, y_hex0 + 0.7 * hex_row, 0.18, 1.6 * (hex_row + hex_title)))
    plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(-HEX_VMAX, HEX_VMAX), cmap="RdBu_r"), cax=cax,
                 label="voltage (z-score)")
    cax.tick_params(labelsize=FS_TICK)
    # b, c: the stimulus and voltage traces at full width (no density panels). Both headers give the
    # per-cell Pearson r, Fisher-z pooled, mean +- SD: over the 1,736 photoreceptors for b, over all
    # 13,741 neurons for c.
    r_s, sd_s = fisher_r(st.T, sp.T)
    step_s = 3.0 * float(np.std(st[T0:T1, sel]))
    ax_b = fig.add_axes(rect(left, bottom + tr_h + gap_tr, tr_w, tr_h))
    traces(ax_b, st[T0:T1 + 1, sel].T, sp[T0:T1 + 1, sel].T, sel_lab, step_s,
           f"stimulus, INR vs true, $r$ = {r_s:.2f} $\\pm$ {sd_s:.2f}", "time (ms)")
    step_v = 3.0 * float(np.std(b["activity_true"][idx, T0:T1]))
    ax_c = fig.add_axes(rect(left, bottom, tr_w, tr_h))
    traces(ax_c, b["activity_true"][idx, T0:T1 + 1], b["activity_pred"][idx, T0:T1 + 1], labels, step_v,
           f"voltage, GNN rollout vs noise-free, $r$ = {r_v:.2f} $\\pm$ {sd_v:.2f}", "time (ms)")
    print(f"stimulus r {r_s:.3f} +- {sd_s:.3f}; voltage r {r_v:.3f} +- {sd_v:.3f}")
    panel_labels(fig, [ax_b, ax_c], letters="bc", align_rows=False, dy_pt=10, y_from="tight")
    fig.text(left / WIDTH - 0.02, (y_hex0 + 3 * (hex_row + hex_title) + 0.05) / height, "a", fontsize=FS_PANEL,
             fontweight="bold", ha="right", va="bottom")
    save(fig, os.path.join(FIG_DIR, "fig_stim_rollout_inr"))


if __name__ == "__main__":
    main()
