"""Figure 12 — recorded vs task-only learned calcium (one script, one figure).

Every panel (a–e) is rendered natively in a single matplotlib figure from
the companion ``.npz`` files dumped by ``fig_functional_panel.py``. There
is no PNG read-back, no compositing of pre-rendered subfigures: panels a,
b, c, d (kinographs + heading strip) and e (power-spectrum overlay) all
share the same font sizes, line widths, and spine style.

    panel a (real,    full block): kinograph + ω + true/decoded HD
    panel b (model,   full block): same layout for the model rollout
    panel c (real,    first 100 s zoom): same as a, x-limit cropped
    panel d (model,   first 100 s zoom): same as b, x-limit cropped
    panel e (full block): per-neuron power spectrum, real vs model
                          (median ± IQR, log–log)

Usage:
    # regenerates the .npz files first, then builds the figure
    python figures/zebrafish/fig_zebrafish_calcium_baseline.py
    # reuse existing .npz (skip the rollout)
    python figures/zebrafish/fig_zebrafish_calcium_baseline.py --skip-generate
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpecFromSubplotSpec

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_PANEL_PY = os.path.join(_HERE, "fig_functional_panel.py")
_DEFAULT_CONFIG = "zebrafish/zebrafish_hd_si_ipn12_artr_pt1_selfmotion_rotation"
_DEFAULT_OUT = os.path.join(_HERE, "fig_zebrafish_calcium_baseline.png")
_PAIRS_T_S = 200.0   # time window (s) for the example-pairs panel (e)

# Shared typography — every panel reads from this so a/b/c/d match e.
FS_LABEL  = 16
FS_TICK   = 13
FS_LEGEND = 13
FS_PANEL  = 22
LW_TRACE  = 1.4
LW_SPEC   = 2.0
GT_COLOR   = "#4daf4a"
PRED_COLOR = "black"


def _run_panel(extra_args, out_dir):
    cmd = [sys.executable, _PANEL_PY, "--no-title", "--out", out_dir, *extra_args]
    print("[gen]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_REPO)


def _wrap_deg(rad):
    """Wrap heading to (-180, 180] degrees, like the original render."""
    d = np.degrees(rad)
    return (d + 180.0) % 360.0 - 180.0


def _kinograph_block(fig, outer_ss, npz, label, *, t_window=None):
    """Render one (kinograph + ω + HD) block into the given outer subplotspec.

    Reads kino / omega / theta / decoded / t_sec from `npz`. Uses the
    same height ratios as the original `panel.render(show_swim_panels=
    False)` but with the manuscript fonts/spines (no bounding box,
    consistent fontsize across all panels).
    """
    kino    = np.asarray(npz["kino"])
    t       = np.asarray(npz["t_sec"])
    omega   = np.asarray(npz["omega"])
    theta   = np.asarray(npz["theta"])
    decoded = np.asarray(npz["decoded"])

    x_lo, x_hi_full = float(t[0]), float(t[-1])
    x_hi = x_hi_full if t_window is None else min(x_hi_full,
                                                  x_lo + float(t_window))
    # decimate the line panels (kino keeps full res)
    s = max(1, len(t) // 4000)
    td = t[::s]

    gs = GridSpecFromSubplotSpec(
        3, 1, subplot_spec=outer_ss,
        height_ratios=[4.2, 1.0, 1.2], hspace=0.10)

    # ── kinograph ────────────────────────────────────────────────────
    ax_k = fig.add_subplot(gs[0])
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("black")
    finite = kino[np.isfinite(kino)]
    vmin = float(np.percentile(finite, 2.0))  if finite.size else -1.0
    vmax = float(np.percentile(finite, 99.5)) if finite.size else  4.0
    ax_k.imshow(kino, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                extent=[x_lo, x_hi_full, kino.shape[0], 0],
                interpolation="nearest")
    ax_k.set_yticks([])
    ax_k.set_xlim(x_lo, x_hi)
    ax_k.tick_params(axis="x", labelbottom=False, length=3)
    ax_k.set_ylabel(f"bump-pool neuron  (n={kino.shape[0]})",
                    fontsize=FS_LABEL, labelpad=6)
    for sp in ax_k.spines.values():
        sp.set_visible(False)
    # panel letter — pushed left of the ylabel so it does not overlap
    # the rotated "bump-pool neuron (n=...)" text.
    ax_k.text(-0.07, 1.04, label, transform=ax_k.transAxes,
              ha="right", va="bottom",
              fontsize=FS_PANEL, fontweight="bold")

    # ── ω ───────────────────────────────────────────────────────────
    ax_o = fig.add_subplot(gs[1])
    ax_o.plot(td, omega[::s], color=GT_COLOR, lw=LW_TRACE)
    ax_o.axhline(0, color="0.35", lw=0.4)
    oa = max(np.abs(omega).max(), 1.0)
    ax_o.set_ylim(-oa * 1.15, oa * 1.15)
    ax_o.set_xlim(x_lo, x_hi)
    ax_o.set_ylabel(r"$\omega$ (°/s)", fontsize=FS_LABEL, labelpad=6)
    ax_o.tick_params(axis="y", labelsize=FS_TICK, length=3)
    ax_o.tick_params(axis="x", labelbottom=False, length=3)
    for sp in ax_o.spines.values():
        sp.set_visible(False)

    # ── HD (true vs decoded) ────────────────────────────────────────
    ax_h = fig.add_subplot(gs[2])
    ax_h.plot(td, _wrap_deg(theta[::s]),   color=GT_COLOR,   lw=LW_TRACE,
              label="true")
    ax_h.plot(td, _wrap_deg(decoded[::s]), color=PRED_COLOR, lw=LW_TRACE * 0.85,
              label="decoded")
    ax_h.set_ylim(-180, 180)
    ax_h.set_yticks([-180, 0, 180])
    ax_h.set_xlim(x_lo, x_hi)
    ax_h.set_xlabel("time (s)", fontsize=FS_LABEL)
    ax_h.set_ylabel("HD (°)", fontsize=FS_LABEL, labelpad=6)
    ax_h.tick_params(axis="both", labelsize=FS_TICK, length=3)
    for sp in ax_h.spines.values():
        sp.set_visible(False)
    ax_h.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right",
                ncol=2, handlelength=1.4)


def _spectrum_panel(ax, freqs, p_real, p_modl, n_real, n_modl, label):
    """Median + IQR power-spectrum overlay (panel e)."""
    f = freqs[1:]
    p_r = p_real[:, 1:]
    p_m = p_modl[:, 1:]
    for population, color, lbl, n in [
        (p_r, GT_COLOR,   "recorded", n_real),
        (p_m, PRED_COLOR, "modelled", n_modl),
    ]:
        med = np.median(population, axis=0)
        q25 = np.percentile(population, 25, axis=0)
        q75 = np.percentile(population, 75, axis=0)
        ax.fill_between(f, q25, q75, color=color, alpha=0.18, lw=0)
        ax.plot(f, med, color=color, lw=LW_SPEC,
                label=f"{lbl}  (n={n})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)", fontsize=FS_LABEL)
    ax.set_ylabel(r"$|X(f)|^{2}$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK, length=3)
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper right")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.07, 1.04, label, transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=FS_PANEL, fontweight="bold")


def _compute_spectrum(real_npz, model_npz):
    """Return (freqs, p_real, p_modl, n_real, n_modl) from the two npz files."""
    from connectome_gnn.metrics import fft_power_spectrum
    real = np.load(real_npz); modl = np.load(model_npz)
    T = min(real["kino"].shape[-1], modl["kino"].shape[-1])

    def _clean(k):
        k = k[:, :T]
        return k[np.isfinite(k).all(axis=1)]

    real_k = _clean(real["kino"])
    modl_k = _clean(modl["kino"])
    dt = float(real["dt_sec"])
    freqs, p_real = fft_power_spectrum(real_k, dt=dt)
    _,    p_modl = fft_power_spectrum(modl_k, dt=dt)
    return freqs, p_real, p_modl, int(real_k.shape[0]), int(modl_k.shape[0])


def _pair_panel(ax, real_npz, model_npz, t_window_s, label):
    """For every modelled bump-pool neuron, find the recorded neuron
    (within the 300-cell anatomy-matched pool) whose power spectrum is
    closest in cosine distance, and overlay the 15 best/median/worst
    pairs on the first `t_window_s` seconds.
    """
    from connectome_gnn.metrics import best_observed_match
    real = np.load(real_npz); modl = np.load(model_npz)
    T = min(real["kino"].shape[-1], modl["kino"].shape[-1])
    real_k = real["kino"][:, :T]
    modl_k = modl["kino"][:, :T]
    real_k = real_k[np.isfinite(real_k).all(axis=1)]
    modl_k = modl_k[np.isfinite(modl_k).all(axis=1)]
    dt = float(real["dt_sec"])
    t_sec = np.asarray(real.get("t_sec"))[:T] if "t_sec" in real.files \
        else np.arange(T) * dt
    t_sec = t_sec[:T]
    n_show = int(np.searchsorted(t_sec, t_window_s, side="right"))
    n_show = max(n_show, 1)

    best_idx, best_score = best_observed_match(
        real_k, modl_k, dt=dt, metric="cosine")
    order = np.argsort(best_score)
    n_each = min(5, modl_k.shape[0] // 3)
    picks = np.concatenate([
        order[:n_each],
        order[len(order)//2 - n_each//2: len(order)//2 + n_each - n_each//2],
        order[-n_each:],
    ])
    spacing = 4.5
    for k, mi in enumerate(picks):
        oi = int(best_idx[mi])
        offset = -k * spacing
        m = modl_k[mi, :n_show]; o = real_k[oi, :n_show]
        ax.plot(t_sec[:n_show], m + offset, color="black", lw=0.9)
        ax.plot(t_sec[:n_show], o + offset, color="#4daf4a", lw=0.9)
    # group labels on the right edge
    for j, name in enumerate(("best", "median", "worst")):
        ymid = -(j * n_each + (n_each - 1) / 2.0) * spacing
        ax.text(t_sec[n_show - 1] * 1.01, ymid, name,
                ha="left", va="center", fontsize=FS_TICK, color="0.4")
    ax.set_yticks([])
    ax.set_xlim(float(t_sec[0]), float(t_sec[n_show - 1]))
    ax.set_xlabel("time (s)", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK, length=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(-0.04, 1.02, label, transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=FS_PANEL, fontweight="bold")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=_DEFAULT_CONFIG,
                   help="model config for panels b/d (default: task-only baseline)")
    p.add_argument("--gcamp", default="gcamp7f")
    p.add_argument("--out", default=_DEFAULT_OUT, help="output PNG path")
    p.add_argument("--panel-dir", default=None,
                   help="dir for the kinograph .npz companions "
                        "(default: alongside --out)")
    p.add_argument("--skip-generate", action="store_true",
                   help="reuse already-generated .npz files in --panel-dir")
    p.add_argument("--zoom-s", type=float, default=100.0,
                   help="seconds for the right-hand zoom column "
                        "(first T s); pass <=0 to drop the zoom column")
    args = p.parse_args()

    panel_dir = args.panel_dir or os.path.dirname(os.path.abspath(args.out))
    os.makedirs(panel_dir, exist_ok=True)
    stem = args.config.rsplit("/", 1)[-1]
    zoom = args.zoom_s if args.zoom_s and args.zoom_s > 0 else None
    real_npz_p = os.path.join(panel_dir, "functional_panel_real_rotation.npz")
    model_npz_p = os.path.join(
        panel_dir, f"functional_panel_{stem}_{args.gcamp}.npz")

    if not args.skip_generate:
        # The kinograph PNGs are no longer consumed by this script, but
        # fig_functional_panel.py still writes the companion .npz that
        # we read below. Drive it once for real, once for model.
        _run_panel(["--sequence", "rotation"], panel_dir)
        _run_panel(["--config", args.config, "--gcamp", args.gcamp], panel_dir)

    for path in (real_npz_p, model_npz_p):
        if not os.path.isfile(path):
            sys.exit(f"missing companion: {path}\n"
                     f"  rerun without --skip-generate to regenerate it")

    real_npz = np.load(real_npz_p)
    model_npz = np.load(model_npz_p)

    # ── outer layout ────────────────────────────────────────────────
    # 3 outer rows × 2 outer cols. Rows 1–2 each hold a kinograph
    # block (3 nested sub-rows: kino + ω + HD); row 3 col-0 holds the
    # power-spectrum panel. Same outer column widths so a/c and b/d
    # are exactly aligned, and panel e sits in column 0 of row 3 so
    # its width matches a + b above.
    # Layout: 3 outer rows × 2 cols. Rows 1-2 hold the kinographs
    # (a,b full block; c,d zoom). Row 3 holds the power-spectrum
    # overlay (panel e) in column 0 only, width-matched to a + b.
    fig = plt.figure(figsize=(20, 16.5))
    outer = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 1.0, 0.85],
        hspace=0.32, wspace=0.12,
        left=0.085, right=0.985, top=0.955, bottom=0.045,
    )

    _kinograph_block(fig, outer[0, 0], real_npz,  "a")
    _kinograph_block(fig, outer[1, 0], model_npz, "b")
    _kinograph_block(fig, outer[0, 1], real_npz,  "c", t_window=zoom)
    _kinograph_block(fig, outer[1, 1], model_npz, "d", t_window=zoom)

    freqs, p_real, p_modl, n_real, n_modl = _compute_spectrum(
        real_npz_p, model_npz_p)
    ax_e = fig.add_subplot(outer[2, 0])
    _spectrum_panel(ax_e, freqs, p_real, p_modl, n_real, n_modl, "e")

    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
