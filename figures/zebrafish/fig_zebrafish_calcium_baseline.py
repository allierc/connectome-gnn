"""Figure 10 — real vs task-only learned calcium kinographs (self-generating).

Rather than stitching pre-made PNGs, this drives
``figures/zebrafish/fig_functional_panel.py`` to (re)generate both panels from
source, then composes them into one labelled a/b figure with no per-panel
titles:

  panel a (real):    fig_functional_panel.py --sequence rotation --no-title
  panel b (learned): fig_functional_panel.py --config <cfg> --no-title

`<cfg>` defaults to the task-only baseline (no observation loss),
``zebrafish/zebrafish_hd_si_ipn12_v1_cv0``. Both panels are written into the
same output dir so the montage picks them up deterministically.

Usage:
    python figures/zebrafish/fig_zebrafish_calcium_baseline.py
    python figures/zebrafish/fig_zebrafish_calcium_baseline.py --config zebrafish/<other> --skip-generate
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
# Make the in-repo metrics module importable for the spectrum panel.
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_PANEL_PY = os.path.join(_HERE, "fig_functional_panel.py")
_DEFAULT_CONFIG = "zebrafish/zebrafish_hd_si_ipn12_artr_pt1_selfmotion_rotation"
_DEFAULT_OUT = os.path.join(_HERE, "fig_zebrafish_calcium_baseline.png")


def _run_panel(extra_args, out_dir):
    cmd = [sys.executable, _PANEL_PY, "--no-title", "--out", out_dir, *extra_args]
    print("[gen]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=_REPO)


def _panel(ax, png, label):
    if not os.path.isfile(png):
        ax.text(0.5, 0.5, f"missing:\n{os.path.basename(png)}", ha="center",
                va="center", transform=ax.transAxes, fontsize=9, color="0.5")
        ax.axis("off")
    else:
        ax.imshow(mpimg.imread(png), interpolation="bilinear")
        ax.axis("off")
    ax.text(-0.01, 1.0, label, transform=ax.transAxes, ha="right", va="top",
            fontsize=16, fontweight="bold")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=_DEFAULT_CONFIG,
                   help="model config for panel b (default: task-only baseline)")
    p.add_argument("--gcamp", default="gcamp7f")
    p.add_argument("--out", default=_DEFAULT_OUT, help="montage PNG path")
    p.add_argument("--panel-dir", default=None,
                   help="dir for the two intermediate panels "
                        "(default: alongside --out)")
    p.add_argument("--skip-generate", action="store_true",
                   help="reuse already-generated panels in --panel-dir")
    p.add_argument("--zoom-s", type=float, default=100.0,
                   help="seconds for the right-hand zoom column (first T s); "
                        "set <=0 to drop the zoom column")
    args = p.parse_args()

    panel_dir = args.panel_dir or os.path.dirname(os.path.abspath(args.out))
    os.makedirs(panel_dir, exist_ok=True)
    stem = args.config.rsplit("/", 1)[-1]
    zoom = args.zoom_s if args.zoom_s and args.zoom_s > 0 else None
    zs = "" if zoom is None else f"_z{int(zoom)}s"
    real_png = os.path.join(panel_dir, "functional_panel_real_rotation.png")
    model_png = os.path.join(panel_dir, f"functional_panel_{stem}_{args.gcamp}.png")
    real_zoom = os.path.join(panel_dir, f"functional_panel_real_rotation{zs}.png")
    model_zoom = os.path.join(panel_dir, f"functional_panel_{stem}_{args.gcamp}{zs}.png")

    if not args.skip_generate:
        _run_panel(["--sequence", "rotation"], panel_dir)            # panel a (real)
        _run_panel(["--config", args.config, "--gcamp", args.gcamp], panel_dir)  # b (model)
        if zoom is not None:                                         # right zoom column
            zarg = ["--t-window", str(zoom)]
            _run_panel(["--sequence", "rotation", *zarg], panel_dir)                 # c
            _run_panel(["--config", args.config, "--gcamp", args.gcamp, *zarg],
                       panel_dir)                                                    # d

    # ---- frequency-analysis panel (e): per-neuron power spectrum,
    # recorded vs.\ modelled, median + IQR. Reads the companion .npz
    # files dumped by fig_functional_panel.py. Silently skips if
    # missing (the figure still works with just the kinograph rows).
    spectrum_data = _load_spectrum_pair(
        real_png.replace(".png", ".npz"),
        model_png.replace(".png", ".npz"),
    )

    # Layout: kinographs + (optional zoom) on rows 1–2, the shared
    # spectrum panel as a wide single axes on row 3.
    has_spec = spectrum_data is not None
    if zoom is None:
        n_rows = 2 + (1 if has_spec else 0)
        height = 6 + (5 if has_spec else 0)
        fig, axs = plt.subplots(n_rows, 1, figsize=(11, height))
        _panel(axs[0], real_png, "a")
        _panel(axs[1], model_png, "b")
        if has_spec:
            _spectrum_panel(axs[2], spectrum_data, "c")
    else:
        # rows = recorded / model, columns = full block / first-`zoom`-s zoom.
        # Both panel PNGs share the same 15:9 canvas, so equal columns render
        # the zoom at the same size as the full block (a ~6x temporal zoom).
        n_rows = 2 + (1 if has_spec else 0)
        height = 11 + (4.5 if has_spec else 0)
        fig = plt.figure(figsize=(20, height))
        gs = fig.add_gridspec(
            n_rows, 2,
            height_ratios=([1, 1, 0.7] if has_spec else [1, 1]),
        )
        axs = np.empty((n_rows, 2), dtype=object)
        for i in range(n_rows):
            for j in range(2):
                axs[i, j] = fig.add_subplot(gs[i, j])
        _panel(axs[0, 0], real_png, "a")
        _panel(axs[1, 0], model_png, "b")
        _panel(axs[0, 1], real_zoom, "c")
        _panel(axs[1, 1], model_zoom, "d")
        if has_spec:
            ax_spec = fig.add_subplot(gs[2, :])
            for j in range(2):
                axs[2, j].axis("off")
            _spectrum_panel(ax_spec, spectrum_data, "e")
        axs[0, 0].set_title("full Rotations block", fontsize=12)
        axs[0, 1].set_title(f"first {int(zoom)} s (zoom)", fontsize=12)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.04,
                        hspace=0.12, wspace=0.04)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {args.out}")


def _load_spectrum_pair(real_npz, model_npz):
    import os
    if not (os.path.isfile(real_npz) and os.path.isfile(model_npz)):
        print(f"[spec] SKIP — missing {real_npz} or {model_npz}; "
              f"rerun panels to dump the companion .npz")
        return None
    try:
        import numpy as np
        from connectome_gnn.metrics import fft_power_spectrum
        real = np.load(real_npz)
        modl = np.load(model_npz)
        # kino arrays: (N_rows, T) z-scored ΔF/F. Match T (recording
        # is sometimes longer than the rollout); align to the shorter.
        T = min(real["kino"].shape[-1], modl["kino"].shape[-1])
        # Drop NaN rows (unmatched bodyId fills) before spectrum.
        def _clean(k):
            k = k[:, :T]
            mask = np.isfinite(k).all(axis=1)
            return k[mask]
        real_k = _clean(real["kino"])
        modl_k = _clean(modl["kino"])
        dt = float(real["dt_sec"])
        freqs, p_real = fft_power_spectrum(real_k, dt=dt)
        _,    p_modl = fft_power_spectrum(modl_k, dt=dt)
        return dict(freqs=freqs, p_real=p_real, p_modl=p_modl,
                     n_real=int(real_k.shape[0]),
                     n_modl=int(modl_k.shape[0]))
    except Exception as e:
        print(f"[spec] SKIP — {type(e).__name__}: {e}")
        return None


def _spectrum_panel(ax, sd, label):
    """Median + IQR power-spectrum overlay: green = recorded, black = model."""
    import numpy as np
    freqs = sd["freqs"][1:]                # drop the DC bin
    p_real = sd["p_real"][:, 1:]
    p_modl = sd["p_modl"][:, 1:]
    GT_COLOR = "#4daf4a"; PRED_COLOR = "black"
    for population, color, lbl, n in [
        (p_real, GT_COLOR,   "recorded", sd["n_real"]),
        (p_modl, PRED_COLOR, "modelled", sd["n_modl"]),
    ]:
        med = np.median(population, axis=0)
        q25 = np.percentile(population, 25, axis=0)
        q75 = np.percentile(population, 75, axis=0)
        ax.fill_between(freqs, q25, q75, color=color, alpha=0.18, lw=0)
        ax.plot(freqs, med, color=color, lw=1.6,
                label=f"{lbl}  (n={n})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)", fontsize=12)
    ax.set_ylabel(r"$|X(f)|^{2}$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=11, frameon=False, loc="upper right")
    ax.text(-0.01, 1.02, label, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=16, fontweight="bold")


if __name__ == "__main__":
    main()
