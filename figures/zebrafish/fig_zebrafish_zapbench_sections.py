"""Overview of the whole ZAPBench recording, split into its 9 task sections, to
ask: could a SINGLE model be trained across the battery, with different blocks
supervising different latent variables?

The rotation block (Fig. 14) drives only heading (+ weak forward). But the
recording has 9 consecutive task blocks, each exciting a different variable.
This figure lays the full 7870-frame recording out with the block boundaries so
the per-section availability of each signal is visible at a glance:

  stim code | forward swim | turning | bump-pool ΔF/F (same rows as Fig. 12/14).

Per-block summaries (forward drive, turning, calcium activity) are printed by the
companion characterisation; here we just show the time course. Visual-velocity /
raw-EMG are NOT drawn (ch8 scaling in stimuli_and_ephys.10chFlt is unverified;
forward/turning come from the validated swim npz).

    python figures/zebrafish/fig_zebrafish_zapbench_sections.py
writes figures/zebrafish/fig_zebrafish_zapbench_sections.png
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

CONNECTOME = os.path.join(_REPO, "figures", "zebrafish",
                          "zebrafish_connectome_HD_IPN12")
CIRCUIT = "zebrafish_HD_IPN12_839_v1"
FUNC = os.path.join(CONNECTOME, "functional")
FF = os.path.join(_REPO, "papers", "fishFuncEM", "data", "functional")
SRC_DT = 0.915

TASKS = ['Gain\nAdapt', 'Random\nDots', 'Light\nDark', 'Phototaxis', 'OMR\nTurning',
         'Position\n(OL)', 'Giving-Up\n(OL)', 'Rotation', 'Darkness']
# which latent each block most strongly drives (from the characterisation)
DRIVES = ['fwd', '–', 'turn', 'turn', 'HD+turn+fwd', '–', 'state', 'HD+turn', 'state']


def _z_cols(M):
    mu = M.mean(0, keepdims=True); sd = M.std(0, keepdims=True)
    return (M - mu) / np.where(sd > 1e-6, sd, 1.0)


def make_figure(out_png=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zebrafish_functional_traces_panel as panel

    od = np.load(os.path.join(FF, "onsets_processed.npz"),
                 allow_pickle=True)["onsets_dict"].item()
    onj = list(od["onsets_img"])
    T_full = 7870
    bounds = onj + [T_full]

    beh = np.load(os.path.join(FF, "forward_turning_from_neurons1201.npz"))
    fwd, turn = beh["forward"][:T_full], beh["turning"][:T_full]
    si = np.load(os.path.join(FF, "stim_info_frames.npz"))["stim_info_frames"][:T_full]

    # bump-pool calcium in the Fig.12/14 rastermap row order
    rows, _ = panel.build_rows(CONNECTOME, CIRCUIT)
    rows = panel.sort_rows_rastermap(rows)
    rows = rows[rows["matched"]].reset_index(drop=True)
    npz = np.load(os.path.join(FUNC, "circuit_functional_traces.npz"), allow_pickle=True)
    obs_pos = {int(b): i for i, b in enumerate(npz["bodyId"].astype(np.int64))}
    obs_ix = np.array([obs_pos[int(b)] for b in rows["bodyId"].to_numpy()], np.int64)
    ca = np.asarray(npz["traces"], np.float32)[:T_full][:, obs_ix]      # (7870, n)
    n = ca.shape[1]

    t = np.arange(T_full) * SRC_DT
    bt = [b * SRC_DT for b in bounds]

    ratios = [0.5, 0.8, 1.0, 1.0, 2.8]
    fig, axs = plt.subplots(len(ratios), 1, sharex=True,
                            figsize=(15, 0.95 * sum(ratios)),
                            gridspec_kw=dict(height_ratios=ratios), squeeze=True)

    # (0) block label strip
    axs[0].set_ylim(0, 1); axs[0].set_yticks([])
    for b in range(9):
        x0, x1 = bt[b], bt[b + 1]
        axs[0].axvspan(x0, x1, color=("0.92" if b % 2 else "0.85"))
        axs[0].text((x0 + x1) / 2, 0.62, TASKS[b], ha="center", va="center",
                    fontsize=7.5)
        axs[0].text((x0 + x1) / 2, 0.18, DRIVES[b], ha="center", va="center",
                    fontsize=7, style="italic", color="tab:blue")
    axs[0].set_title("ZAPBench task sections — italic = latent variable each block "
                     "most strongly drives", fontsize=10)

    # (1) stim code
    axs[1].plot(t, si, color="0.4", lw=0.5)
    axs[1].set_ylabel("stim\ncode")

    # (2) forward swim
    axs[2].plot(t, fwd, color="tab:purple", lw=0.5)
    axs[2].axhline(0, color="0.8", lw=0.4)
    axs[2].set_ylabel("forward\n(a.u.)")

    # (3) turning
    axs[3].plot(t, turn, color="tab:olive", lw=0.5)
    axs[3].axhline(0, color="0.8", lw=0.4)
    axs[3].set_ylabel("turning\n(a.u.)")

    # (4) bump-pool ΔF/F kinograph
    axs[4].imshow(_z_cols(ca).T, aspect="auto", cmap="viridis", vmin=-2, vmax=3,
                  extent=[t[0], t[-1], n, 0], interpolation="nearest")
    axs[4].set_ylabel(f"bump-pool\nneuron (n={n})")
    axs[4].set_xlabel("time (s)")

    for ax in axs:
        for b in bt[1:-1]:
            ax.axvline(b, color="k", lw=0.6, alpha=0.5)
        ax.set_xlim(t[0], t[-1])

    fig.suptitle("Whole ZAPBench recording across its 9 task sections: "
                 "different blocks excite different variables "
                 "(forward: Gain/OMR/Rotation; turning: Rotation/OMR/Phototaxis; "
                 "state: Giving-Up/Darkness)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if out_png is None:
        out_png = os.path.join(_HERE, "fig_zebrafish_zapbench_sections.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[fig] wrote {out_png}")
    return out_png


if __name__ == "__main__":
    make_figure()
