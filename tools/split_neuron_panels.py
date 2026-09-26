#!/usr/bin/env python
"""Cut a neuron-panel figure into two slide-sized halves.

WHY. `analyse_neurons` draws six panels on one sheet: the traces a, b, c and the
tall synapse stack d down the left, the update fit e and the per-synapse fits f
down the right. On a 16:9 slide that whole sheet is height-limited to about
0.72 of the text height, and panel f's per-synapse numbers -- the ones that say
what the readout recovered for each edge -- come out around 2 pt. Splitting the
sheet across two slides doubles the linear scale of everything on it.

WHERE IT CUTS. At the widest fully-blank row band in the middle of the figure,
not at a hard-coded fraction: the sheet's height depends on how many synapses
neuron i has, so 42% on one neuron is the wrong place on another. On the
11-synapse neuron 2895 sheet the band is rows 1239-1267 of 2977, which puts
a, b, c and e above and d and f below -- the split asked for.

    python tools/split_neuron_panels.py <png> [<png> ...] [--window LO HI]

Writes <stem>_top.png and <stem>_bot.png beside each input and prints them.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image


def find_cut(gray, lo_frac, hi_frac):
    """Row index to cut at: the middle of the widest blank band in the window.

    Returns None when the figure has no blank band there, which is a figure
    this tool should not be guessing about -- the caller says so and skips it
    rather than cutting through a panel.
    """
    blank = (gray > 250).all(axis=1)
    runs, start = [], None
    for i, b in enumerate(blank):
        if b and start is None:
            start = i
        elif not b and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(blank)))
    h = gray.shape[0]
    inside = [(a, b) for a, b in runs if lo_frac < (a + b) / 2 / h < hi_frac]
    if not inside:
        return None
    a, b = max(inside, key=lambda r: r[1] - r[0])
    return (a + b) // 2


def split(path, lo_frac, hi_frac):
    img = Image.open(path)
    gray = np.asarray(img.convert("L"))
    cut = find_cut(gray, lo_frac, hi_frac)
    if cut is None:
        print(f"{path}: no blank band in {lo_frac:.0%}-{hi_frac:.0%}, skipped")
        return []
    stem, ext = os.path.splitext(path)
    out = []
    for name, box in (("top", (0, 0, img.width, cut)),
                      ("bot", (0, cut, img.width, img.height))):
        # Trim the blank margin the cut exposes, so the half that keeps a short
        # panel does not waste a third of the slide on white.
        half = img.crop(box)
        g = np.asarray(half.convert("L"))
        rows = np.where(~(g > 250).all(axis=1))[0]
        if rows.size:
            half = half.crop((0, max(0, rows[0] - 20), half.width,
                              min(half.height, rows[-1] + 20)))
        p = f"{stem}_{name}{ext}"
        half.save(p)
        out.append(p)
        print(f"wrote {p}  {half.width}x{half.height}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pngs", nargs="+")
    ap.add_argument("--window", nargs=2, type=float, default=(0.35, 0.55),
                    metavar=("LO", "HI"),
                    help="fraction-of-height band to look for the cut in "
                         "(default 0.35 0.55)")
    a = ap.parse_args()
    n = 0
    for p in a.pngs:
        if not os.path.exists(p):
            print(f"{p}: not found", file=sys.stderr)
            continue
        n += len(split(p, *a.window))
    print(f"{n} halves written")


if __name__ == "__main__":
    main()
