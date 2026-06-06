"""Per-model test-result composite: random trials + deterministic sweep.

For each trained ARTR / pt-IPN1 variant, stacks the two test PNGs
produced by ``GNN_Main.py -o test``
(``results/test_random_trials.png`` and the appropriate
``results/test_deterministic_*.png``) into a single composite
``figures/zebrafish/fig_test_<run>.png`` referenced by the manuscript.

Composite layout (top → bottom):
    row 1:  test_random_trials.png        (held-out trials)
    row 2:  test_deterministic_<mode>.png (constant-input rollout)

The deterministic plot picked per variant:
    rotation         → test_deterministic_sweep.png         (ω-sweep)
    translation      → test_deterministic_v_fwd_sweep.png   (v_fwd-sweep)
    translation_leaky → same
    both             → test_deterministic_sweep.png         (ω-sweep)
    both_leaky       → same
    position_2d      → test_deterministic_2d_sweep.png      ((x,y) sweep)
    position_2d_leaky → same
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))

PREFIX = "zebrafish_hd_si_ipn12_artr_pt1_"
# (suffix, deterministic-png basename, short label)
VARIANTS = [
    ("selfmotion_rotation",
     "test_deterministic_sweep.png",
     "rotation only"),
    ("selfmotion_translation",
     "test_deterministic_v_fwd_sweep.png",
     "translation only (perfect d)"),
    ("selfmotion_translation_leaky",
     "test_deterministic_v_fwd_sweep.png",
     "translation only (leaky d)"),
    ("selfmotion_both",
     "test_deterministic_sweep.png",
     "rotation + scalar d (perfect)"),
    ("selfmotion_both_leaky",
     "test_deterministic_sweep.png",
     "rotation + scalar d (leaky)"),
    ("position_2d",
     "test_deterministic_2d_sweep.png",
     "rotation + 2D path (perfect)"),
    ("position_2d_leaky",
     "test_deterministic_2d_sweep.png",
     "rotation + 2D path (leaky)"),
]


def _compose(random_png: str, det_png: str, out_path: str, label: str) -> None:
    if not os.path.isfile(random_png):
        print(f"  SKIP {os.path.basename(out_path)}: missing {random_png}")
        return
    if not os.path.isfile(det_png):
        print(f"  SKIP {os.path.basename(out_path)}: missing {det_png}")
        return

    img_top = mpimg.imread(random_png)
    img_bot = mpimg.imread(det_png)

    # Match widths by scaling heights to preserve aspect; matplotlib
    # imshow handles the unequal source widths via the data extent below.
    h_top, w_top = img_top.shape[:2]
    h_bot, w_bot = img_bot.shape[:2]
    # Display each row at the same on-figure width; height ratio
    # preserves the original aspect of each input.
    target_w = max(w_top, w_bot)
    ar_top = h_top / w_top
    ar_bot = h_bot / w_bot
    fig_w = 13.0
    fig_h = fig_w * (ar_top + ar_bot) + 0.6   # 0.6" for the row labels
    fig, (ax_t, ax_b) = plt.subplots(
        2, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [ar_top, ar_bot]},
    )
    ax_t.imshow(img_top); ax_t.set_axis_off()
    ax_b.imshow(img_bot); ax_b.set_axis_off()
    ax_t.set_title("Random held-out trials", fontsize=12, loc="left",
                    fontweight="bold")
    ax_b.set_title("Constant-input deterministic sweep", fontsize=12,
                    loc="left", fontweight="bold")
    fig.suptitle(label, fontsize=13, y=1.0)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01,
                         hspace=0.02)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root",
                   default="/groups/saalfeld/home/allierc/GraphData")
    p.add_argument("--out_dir", default=HERE)
    p.add_argument("--only", nargs="+", default=None,
                   help="render only these suffixes (e.g. selfmotion_rotation)")
    args = p.parse_args()

    wanted = set(args.only) if args.only else None
    for suffix, det_name, label in VARIANTS:
        if wanted is not None and suffix not in wanted:
            continue
        run = PREFIX + suffix
        results = os.path.join(args.data_root, "log", "zebrafish",
                                run, "results")
        random_png = os.path.join(results, "test_random_trials.png")
        det_png    = os.path.join(results, det_name)
        out_path   = os.path.join(args.out_dir, f"fig_test_{run}.png")
        _compose(random_png, det_png, out_path, f"{label} — {suffix}")


if __name__ == "__main__":
    main()
