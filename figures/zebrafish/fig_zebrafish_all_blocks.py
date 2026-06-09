"""All-blocks inventory: the full ~2 h ZAPBench recording, every locally-
available drive/behaviour over the 9 task blocks, above the bump-pool ΔF/F.

Extends the rotation-only covariate figure (the paper's Fig. 14) to the whole
session, to motivate a single multi-task model: each block excites a different
axis of the latent code, so their union constrains a far richer circuit than
Rotation alone. Rows (shared time axis, 7870 frames ≈ 2 h):

  1. stim-condition code (stim_info_frames)   — the per-frame categorical drive
  2. forward swim (bilateral tail-EMG)         — self-motion / translational
  3. turning  (L/R tail-EMG asymmetry)         — heading / optomotor
  4. bump-pool real ΔF/F (n≈300, rastermap)    — the calcium target, all blocks

9 blocks (onsets from fishfuncem) are separated and labelled:
  GAIN DOTS FLASH TAXIS TURNING POSITION OPEN-LOOP ROTATION DARK

  python figures/zebrafish/fig_zebrafish_all_blocks.py
writes figures/zebrafish/fig_zebrafish_all_blocks.png
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (os.path.join(_REPO, "src"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

CONNECTOME = os.path.join(_HERE, "zebrafish_connectome_HD_IPN12")
CIRCUIT = "zebrafish_HD_IPN12_839_v1"
FUNC = os.path.join(CONNECTOME, "functional")
FISHFUNC = os.path.join(_REPO, "papers", "fishFuncEM", "data", "functional")
SRC_DT = 0.915

# block onsets (imaging frames) + labels — fishfuncem ff.onsets_frames.
ONSETS = [0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279]
TASKS = ["GAIN", "DOTS", "FLASH", "TAXIS", "TURNING",
         "POSITION", "OPEN LOOP", "ROTATION", "DARK"]
VMIN, VMAX = -2.0, 3.0


def _z(x):
    x = np.asarray(x, np.float64)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-9 else 1.0)


def _zscore_cols(M):
    mu = M.mean(0, keepdims=True)
    sd = M.std(0, keepdims=True)
    return (M - mu) / np.where(sd > 1e-6, sd, 1.0)


def load_full_recording():
    import zebrafish_functional_traces_panel as panel
    npz = np.load(os.path.join(FUNC, "circuit_functional_traces.npz"),
                  allow_pickle=True)
    traces = np.asarray(npz["traces"], np.float32)            # (T, 481)
    T = traces.shape[0]

    # bump-pool rows in the SAME rastermap order as Fig. 12/14
    rows, _ = panel.build_rows(CONNECTOME, CIRCUIT)
    rows = panel.sort_rows_rastermap(rows)
    rows = rows[rows["matched"]].reset_index(drop=True)
    pos = {int(b): i for i, b in enumerate(np.asarray(npz["bodyId"], np.int64))}
    ix = np.array([pos[int(b)] for b in rows["bodyId"].to_numpy()
                   if int(b) in pos], np.int64)
    calcium = traces[:, ix]                                   # (T, n_bump)

    stim = np.load(os.path.join(FISHFUNC, "stim_info_frames.npz"))[
        "stim_info_frames"][:T]
    beh = np.load(os.path.join(FISHFUNC, "forward_turning_from_neurons1201.npz"))
    fwd, turn = beh["forward"][:T], beh["turning"][:T]

    # The three velocity-gate afferent classes (Fig. 1 colour code): their
    # recorded ΔF/F over the same frames, matched cells only.
    import zebrafish_afferent_kino as aff
    afferents = [(name, colour, tr[:T])
                 for name, colour, tr in aff.load_afferent_traces(CONNECTOME)]
    return dict(T=T, t=np.arange(T) * SRC_DT, stim=stim,
                forward=fwd, turning=turn, calcium=calcium,
                n_bump=calcium.shape[1], afferents=afferents)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load_full_recording()
    t, T = d["t"], d["T"]
    edges = ONSETS + [T]
    mids = [(edges[i] + edges[i + 1]) / 2 * SRC_DT for i in range(len(TASKS))]
    seps = [e * SRC_DT for e in ONSETS[1:]]

    # stim | forward | turning | ARTR | pt-IPN1 | motor_efferent | bump-pool
    aff_rows = d["afferents"]
    ratios = [0.8, 1.0, 1.0] + [1.4] * len(aff_rows) + [3.2]
    fig, axs = plt.subplots(len(ratios), 1, sharex=True,
                            figsize=(15, 0.95 * sum(ratios) + 1.0),
                            gridspec_kw=dict(height_ratios=ratios))

    def _seps(ax, lab=False):
        for s in seps:
            ax.axvline(s, color="0.6", lw=0.6, ls="-")
        if lab:
            for m, name in zip(mids, TASKS):
                ax.text(m, 1.02, name, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                        rotation=0)

    axs[0].plot(t, d["stim"], color="0.25", lw=0.6, drawstyle="steps-mid")
    axs[0].set_ylabel("stim\ncode"); _seps(axs[0], lab=True)
    axs[0].set_title("")

    axs[1].plot(t, d["forward"], color="tab:purple", lw=0.5)
    axs[1].set_ylabel("forward\n(a.u.)"); _seps(axs[1])
    axs[1].axhline(0, color="0.85", lw=0.4)

    axs[2].plot(t, d["turning"], color="tab:olive", lw=0.5)
    axs[2].set_ylabel("turning\n(a.u.)"); _seps(axs[2])
    axs[2].axhline(0, color="0.85", lw=0.4)

    # ── three afferent-class kinographs (recorded ΔF/F, Fig. 1 colours) ──
    # ARTR (ω), pt-IPN1 (v_ext), motor_efferent (v_prop). Same viridis LUT
    # and z-score as the bump-pool kinograph; class name labelled in its
    # Fig. 1 colour.
    for j, (name, colour, tr) in enumerate(aff_rows):
        ax = axs[3 + j]
        na = tr.shape[1]
        ax.imshow(_zscore_cols(tr).T, aspect="auto", cmap="viridis",
                  vmin=VMIN, vmax=VMAX, extent=[t[0], t[-1], na, 0],
                  interpolation="nearest")
        ax.set_ylabel(f"{name}\n(n={na})", color=colour, fontweight="bold")
        ax.set_yticks([])
        for s in seps:
            ax.axvline(s, color="white", lw=0.7)

    n = d["n_bump"]
    ax_bump = axs[3 + len(aff_rows)]
    ax_bump.imshow(_zscore_cols(d["calcium"]).T, aspect="auto", cmap="viridis",
                   vmin=VMIN, vmax=VMAX, extent=[t[0], t[-1], n, 0],
                   interpolation="nearest")
    ax_bump.set_ylabel(f"bump-pool\nneuron (n={n})")
    for s in seps:
        ax_bump.axvline(s, color="white", lw=0.7)
    ax_bump.set_xlabel("time (s)")

    for ax in axs:
        ax.set_xlim(t[0], t[-1])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.tight_layout()
    out = os.path.join(_HERE, "fig_zebrafish_all_blocks.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[fig] wrote {out}  (n_bump={n}, T={T})")


if __name__ == "__main__":
    main()
