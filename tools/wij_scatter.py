"""Learned conductance against true, for every edge, with one neuron marked.

THE QUESTION THIS ANSWERS. Panel f of neuron2895_panels.png shows eight synapses
whose conductance comes back within a factor of 1.1-2 of the generator's, while
the population Wij_R2 is +0.27. Both are true of the same model, and the gap is
not a contradiction but a sampling statement: 2895 is a well-driven neuron whose
synapses the model kept, and most edges are not like that.

Four things are drawn, and the point is the third:

  left    every fitted edge, learned against true, on log axes -- the bulk
  middle  the same on linear axes over the top decile of true conductance, where
          the recovery either works or does not
  right   the per-neuron median ratio learned/true against how strongly that
          neuron is driven, which is the variable the gap is really about
  marked  neuron 2895's own edges, in red, on all three

    python tools/wij_scatter.py <config> [--neuron 2895] [--gauge-tau model|true]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.metrics import extract_template_params      # noqa: E402
from connectome_gnn.utils import to_numpy                       # noqa: E402

from pysr_recovery import load_run                              # noqa: E402


def _per_neuron_ratio(w_true, w_learn, dst, n_neurons, floor):
    """Median learned/true per postsynaptic neuron, over its non-negligible edges."""
    ratio = np.full(n_neurons, np.nan)
    drive = np.full(n_neurons, np.nan)
    ok = np.isfinite(w_learn) & (w_true > floor)
    for i in np.unique(dst[ok]):
        m = ok & (dst == i)
        if m.sum() >= 3:
            ratio[i] = np.median(w_learn[m] / w_true[m])
            drive[i] = float(np.sum(w_true[m]))
    return ratio, drive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--neuron", type=int, default=2895)
    ap.add_argument("--gauge-tau", choices=("model", "true"), default="model")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg, data, model, log_dir = load_run(args.config, args.device)
    n = int(data.n_neurons)
    edges = data.edges.to(args.device)
    rec = extract_template_params(model, data.ode_params, config=cfg, edges=edges,
                                  x_ts=data.x_ts, device=args.device, n_neurons=n,
                                  gauge_tau=args.gauge_tau)
    w_learn = np.asarray(rec.diagnostics["_W_learned_full"], dtype=float)
    w_true = np.asarray(data.ode_params.effective_true_weights(
        to_numpy(data.ode_params.W), to_numpy(edges), n), dtype=float).ravel()
    dst = np.asarray(to_numpy(edges)).reshape(2, -1)[1]
    fitted = np.isfinite(w_learn) & np.isfinite(w_true)
    mine = dst == args.neuron

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # --- every edge, log-log. Only the positive pairs can be drawn; the count of
    # what falls off each axis is printed, because a log plot silently hides
    # exactly the edges the model killed.
    ax = axes[0]
    pos = fitted & (w_true > 0) & (w_learn > 0)
    ax.loglog(w_true[pos & ~mine], w_learn[pos & ~mine], ".", ms=1, alpha=0.15,
              color="0.4", rasterized=True)
    ax.loglog(w_true[pos & mine], w_learn[pos & mine], "o", ms=5, color="tab:red")
    lo, hi = 1e-12, max(w_true[pos].max(), w_learn[pos].max()) * 2
    ax.plot([lo, hi], [lo, hi], color="tab:green", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("true W"); ax.set_ylabel("learned W")
    _neg = int((fitted & ((w_true <= 0) | (w_learn <= 0))).sum())
    ax.text(0.02, 1.02, f"a   every fitted edge, log-log   n = {int(fitted.sum())}"
                        f"   ({_neg} not drawn: zero or negative)",
            transform=ax.transAxes, va="bottom", fontsize=10)

    # --- the top decile of true conductance, linear. This is the set a claim
    # about recovering a connectome actually rests on.
    ax = axes[1]
    thr = np.quantile(w_true[fitted], 0.9)
    top = fitted & (w_true >= thr)
    ax.plot(w_true[top & ~mine], w_learn[top & ~mine], ".", ms=2, alpha=0.25,
            color="0.4", rasterized=True)
    ax.plot(w_true[top & mine], w_learn[top & mine], "o", ms=6, color="tab:red")
    _hi = float(np.quantile(w_true[top], 0.999))
    ax.plot([0, _hi], [0, _hi], color="tab:green", lw=1)
    ax.set_xlim(0, _hi); ax.set_ylim(0, _hi)
    ax.set_xlabel("true W"); ax.set_ylabel("learned W")
    _r = np.median(w_learn[top] / np.maximum(w_true[top], 1e-30))
    ax.text(0.02, 1.02, f"b   top decile of true W (>= {thr:.4g})   "
                        f"median learned/true = {_r:.2f}",
            transform=ax.transAxes, va="bottom", fontsize=10)

    # --- per neuron: how far off is this neuron's whole row, against how much
    # conductance it receives. A row is scaled by its own k_i, so a neuron whose
    # tau the model missed has every synapse wrong by one common factor -- which
    # is a different failure from getting the wiring wrong.
    ax = axes[2]
    ratio, drive = _per_neuron_ratio(w_true, w_learn, dst, n, floor=thr * 0.1)
    good = np.isfinite(ratio) & np.isfinite(drive) & (ratio > 0)
    ax.semilogy(drive[good], ratio[good], ".", ms=2, alpha=0.3, color="0.4",
                rasterized=True)
    if good[args.neuron]:
        ax.semilogy([drive[args.neuron]], [ratio[args.neuron]], "o", ms=7,
                    color="tab:red")
    ax.axhline(1.0, color="tab:green", lw=1)
    ax.set_xlabel("total true conductance onto the neuron")
    ax.set_ylabel("median learned/true over its edges")
    ax.text(0.02, 1.02, f"c   per postsynaptic neuron, n = {int(good.sum())}   "
                        f"median {np.median(ratio[good]):.2f}, "
                        f"IQR {np.subtract(*np.percentile(ratio[good], [75, 25])):.2f}"
                        f"   (red: neuron {args.neuron})",
            transform=ax.transAxes, va="bottom", fontsize=10)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(log_dir, "results", f"Wij_scatter_{args.gauge_tau}_tau.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"-> {out}")

    # The numbers the figure is there to make believable.
    m = fitted & mine
    print(f"\nneuron {args.neuron}: {int(m.sum())} fitted edges, "
          f"median learned/true = {np.median(w_learn[m] / np.maximum(w_true[m], 1e-30)):.3f}")
    print(f"top decile       : median learned/true = {_r:.3f}")
    print(f"all fitted edges : median learned/true = "
          f"{np.median(w_learn[fitted] / np.maximum(w_true[fitted], 1e-30)):.3g}")
    print(f"per-neuron ratio : median {np.median(ratio[good]):.3f}, "
          f"10-90% {np.percentile(ratio[good], 10):.3f}-{np.percentile(ratio[good], 90):.3f}")


if __name__ == "__main__":
    main()
