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

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_PANEL_PY = os.path.join(_HERE, "fig_functional_panel.py")
_DEFAULT_CONFIG = "zebrafish/zebrafish_hd_si_ipn12_v1_cv0"
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
    args = p.parse_args()

    panel_dir = args.panel_dir or os.path.dirname(os.path.abspath(args.out))
    os.makedirs(panel_dir, exist_ok=True)
    stem = args.config.rsplit("/", 1)[-1]
    real_png = os.path.join(panel_dir, "functional_panel_real_rotation.png")
    model_png = os.path.join(panel_dir, f"functional_panel_{stem}_{args.gcamp}.png")

    if not args.skip_generate:
        _run_panel(["--sequence", "rotation"], panel_dir)            # panel a
        _run_panel(["--config", args.config, "--gcamp", args.gcamp], panel_dir)  # panel b

    fig, axs = plt.subplots(2, 1, figsize=(11, 11))
    _panel(axs[0], real_png, "a")
    _panel(axs[1], model_png, "b")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.01, hspace=0.05)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
