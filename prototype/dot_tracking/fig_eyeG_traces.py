#!/usr/bin/env python
"""fig_eyeG_traces -- what the movies show, as three curves on a page.

    python fig_eyeG_traces.py [--tag eyeG_deep]

The movies are the only record of how the controller behaves over time, and an mp4
is not something a note can quote. This draws the same rollout -- the same
checkpoint, the same sequences, the same seed, so the curves are frame-for-frame
what the movies play -- as three horizontal panels, one per regime:

  (a) smooth pursuit, from the corpus sequence;
  (b) saccades at the slowest rate the ramp still tests;
  (c) saccades at the fastest, where the half-period is the eye's own delay.

Each panel plots the horizontal angle only, because that is the axis the saccades
move and the one the reader can compare across panels: the target
$\\theta^{\\star}$, the command $\\theta_\\infty$ of Eq. (12) --- where the eye would
settle if the drives froze --- and the gaze $\\theta$ of Eq. (13), where it actually
is. The three-way gap between them is the whole subject of sections 4.1 and 5.2.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_eyeG as TG                                      # noqa: E402
import test_eyeG as TE                                       # noqa: E402
import fig_pulse_step as FP                                  # noqa: E402

C_TGT, C_CMD, C_GAZE = "black", "#cf222e", "#1f6feb"
LBL = dict(fontsize=17, fontweight="bold", va="top", ha="left")


def roll(tag, saccade, duration, dt_seed=0):
    mdl, eye, scale, dt = FP.load(tag)
    if saccade:
        v, xy, cuts = TE.saccade_sequence(duration, dt, scale)
        labels = [f"{f:g} Hz" for f in TE.SACCADE_HZ]
    else:
        v, xy, cuts = TE.sequence(duration, dt, scale, seed=dt_seed)
        labels = [(f"circle {'ccw' if mo.endswith('ccw') else 'cw'}"
                   if mo.startswith("circle") else mo.replace("_", "-"))
                  for mo, _ in TE.PHASES]
    x, m, R, cmd = FP.roll(mdl, eye, v)
    return dict(t=np.arange(len(x)) * dt, tgt=xy * scale, cmd=cmd, x=x,
                cuts=list(cuts), labels=labels, dt=dt)


def panel(ax, d, lo, hi, letter, title, legend=False):
    t = d["t"][lo:hi] - d["t"][lo]
    ax.plot(t, d["cmd"][lo:hi, 0], "-", color=C_CMD, lw=1.1, alpha=0.55,
            label=r"command  $\theta_\infty$  (12)")
    ax.plot(t, d["tgt"][lo:hi, 0], "-", color=C_TGT, lw=1.7,
            label=r"target  $\theta^{\star}$")
    ax.plot(t, d["x"][lo:hi, 0], "-", color=C_GAZE, lw=2.0,
            label=r"gaze  $\theta$  (13)")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="black", labelsize=12)
    ax.set_xlabel("time (s)", fontsize=13.5)
    ax.text(-0.11, 1.13, letter, transform=ax.transAxes, color="black", **LBL)
    ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=12.5, va="bottom")
    # Scale to the target and the gaze, not to the command. The command's pulse at
    # each reversal is an order of magnitude larger than the excursion -- that is
    # section 5.2's whole point -- and letting it set the axis flattens the two
    # curves the panel is about. It leaves the panel instead, and by how much is
    # printed rather than drawn.
    span = np.abs(np.concatenate([d["tgt"][lo:hi, 0], d["x"][lo:hi, 0]])).max()
    ax.set_ylim(-1.9 * span, 1.9 * span)
    peak = np.abs(d["cmd"][lo:hi, 0]).max()
    if peak > 1.9 * span:
        ax.text(0.985, 0.955, f"command peaks at $\\pm${peak:.0f}$^\\circ$",
                transform=ax.transAxes, ha="right", va="top", fontsize=11,
                color=C_CMD)
    e = np.linalg.norm(d["x"][lo:hi, :2] - d["tgt"][lo:hi], axis=1).mean()
    ax.text(0.985, 0.04, f"mean |err| {e:.2f}$^\\circ$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11.5)
    if legend:
        ax.legend(frameon=False, fontsize=11, loc="upper left",
                  bbox_to_anchor=(0.0, 0.99), ncol=1)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="eyeG_deep")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--window", type=float, default=4.0, help="seconds per panel")
    p.add_argument("--out", default=os.path.join(TG.MODELS, "fig_eyeG_traces.png"))
    a = p.parse_args()

    pur = roll(a.tag, False, a.duration)
    sac = roll(a.tag, True, a.duration)
    n = int(a.window / pur["dt"])

    # (a) the middle of the smooth-pursuit phase; (b) and (c) the slowest and
    # fastest saccade rates, each taken from the middle of its own phase
    b0 = [0] + sac["cuts"] + [len(sac["t"])]
    pb = [0] + pur["cuts"] + [len(pur["t"])]
    k_pur = 2                                   # continue, fast
    mid = lambda lo, hi: (lo + hi) // 2 - n // 2

    fig, ax = plt.subplots(1, 3, figsize=(16.0, 4.4), facecolor="white")
    fig.subplots_adjust(wspace=0.20, left=0.055, right=0.99, top=0.84, bottom=0.16)
    s = mid(pb[k_pur], pb[k_pur + 1])
    panel(ax[0], pur, s, s + n, "a",
          f"smooth pursuit  ({pur['labels'][k_pur]})", legend=True)
    for j, (k, L) in enumerate(((0, "b"), (len(sac["labels"]) - 1, "c"))):
        s = mid(b0[k], b0[k + 1])
        panel(ax[j + 1], sac, s, s + n, L, f"saccades  {sac['labels'][k]}")
    ax[0].set_ylabel(r"horizontal angle  $\theta$  (deg)", fontsize=13.5)

    fig.savefig(a.out, dpi=170, facecolor="white", bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
