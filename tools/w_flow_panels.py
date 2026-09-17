"""Nine weight distributions on one pair of axes: eight flow runs and the twin.

WHAT IS PLOTTED. For every trained network, the EFFECTIVE weight of each of the
1,513,231 edges -- `syn_count * syn_strength`, the quantity the message actually
multiplies -- as |w| on a log axis, against the `randn_scaled` initialisation
drawn in red on top of each panel. The ninth panel is the conductance twin that
generated the connectome-gnn campaign's data (a known-ODE student, not a flow
run), on the same axes, so the two lines of work are read against each other.

Shared axes throughout: nine histograms with nine auto-ranges are nine pictures,
and the question here is whether these distributions differ from one another and
from the initialisation, which only one pair of axes can answer.

Usage: python tools/w_flow_panels.py [--out figures/w_flow_panels.png]
"""

import argparse
import glob
import logging
import math
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLOW_RUNS = [(f"flow/1000/{m:03d}", "current") for m in range(6)] + \
            [(f"flow/2000/{m:03d}", "conductance") for m in (1, 2)]
# THE NINTH PANEL IS A TARGET, NOT A RUN. `ode_params.pt` sits beside the data
# the generator wrote, holding the W every connectome-gnn run on that dataset is
# scored against -- so putting it on these axes says whether a flow-trained
# weight distribution looks anything like the one the campaign has to recover.
GT_DATASETS = [("flyvis_noise_005_blank50_cv00", "target current"),
               ("flyvis_conductance_ion_sub_noise_005_blank50_cv01", "target conductance")]
TWIN = "flyvis_current_noise_free_conductance_ion_sub_cv00"
GRAPHS = os.environ.get(
    "GNN_GRAPHS_ROOT",
    "/groups/saalfeld/home/allierc/GraphData/graphs_data/fly")


def flow_weights(name):
    """`syn_count * syn_strength` per edge, from the run's selected checkpoint."""
    import flyvis_conductance_optical_flow  # noqa: F401  registers the dynamics
    from flyvis.network.network_view import NetworkView

    nv = NetworkView(name)
    network = nv.init_network()
    network.clamp()
    params = network._param_api()
    w = (params.edges.syn_count * params.edges.syn_strength)
    return np.abs(np.asarray(w.detach().cpu(), dtype=float).ravel())


def twin_weights(config_name, device):
    """The generating twin's learned conductances, from its own checkpoint."""
    sys.path.insert(0, "src")
    from connectome_gnn.models.utils import load_run_config
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.utils import log_path, migrate_state_dict, to_numpy

    cfg, _ = load_run_config(config_name, False, "plot")
    log_dir = log_path(cfg.config_file)
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type, config=cfg,
                         device=device).to(device)
    ck = sorted(glob.glob(os.path.join(log_dir, "models", "best_model_with_*.pt")))
    sd = torch.load(ck[-1], map_location=device, weights_only=False)
    migrate_state_dict(sd)
    model.load_state_dict(sd.get("model_state_dict", sd), strict=False)
    # The conductance class holds the square root of the conductance.
    return np.abs(to_numpy(model.W).ravel() ** 2)


def gt_weights(dataset):
    """The generator's own W for a dataset, out of the file the runs score against."""
    path = os.path.join(GRAPHS, dataset, "ode_params.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    w = d["W"] if isinstance(d, dict) else getattr(d, "W")
    w = w.detach().cpu() if hasattr(w, "detach") else torch.as_tensor(w)
    return np.abs(np.asarray(w, dtype=float).ravel())


def init_draw(n_edges, scale, seed=42):
    """The randn_scaled initialisation these runs start from."""
    # DEVICE EXPLICIT: importing flyvis sets the default device to cuda, and a
    # cpu generator against a cuda default raises.
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    draw = torch.randn(n_edges, generator=g, device="cpu")
    return np.abs(np.asarray(draw * (scale / math.sqrt(n_edges))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures/w_flow_panels.png")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="w_init_scale of the red reference draw")
    args = ap.parse_args()
    logging.getLogger().setLevel(logging.ERROR)

    panels = []
    for name, family in FLOW_RUNS:
        try:
            w = flow_weights(name)
            panels.append((name.replace("flow/", ""), family, w))
            print(f"{name:<16} {family:<12} {w.size:>9} edges  "
                  f"|w| median {np.median(w[w > 0]):.4g}")
        except Exception as exc:
            print(f"{name}: {type(exc).__name__}: {exc}")
    # THE TWIN'S OWN CHECKPOINT IS NOT DRAWN. Its learned conductances are, to
    # the digit, the ground truth of the dataset it wrote -- median |W| 2.383e-08
    # in both -- so the conductance target panel already shows them, and a second
    # copy invited the reading that two independent things agreed.
    for ds, fam in GT_DATASETS:
        w_gt = gt_weights(ds)
        panels.append((f"{ds}\nground truth the runs are scored against", fam, w_gt))
        print(f"{ds:<52} {fam:<20} {w_gt.size:>9} edges  "
              f"|w| median {np.median(w_gt[w_gt > 0]):.4g}")

    allw = np.concatenate([np.abs(w[w > 0]) for _, _, w in panels])
    lo, hi = np.percentile(allw, 0.05), allw.max()
    bins = np.logspace(np.log10(max(lo, 1e-12)), np.log10(hi * 1.5), 80)

    # THE COLOUR IS THE KIND OF THING, not decoration: the six current flow
    # members, the two conductance ones, the twin that generated the campaign's
    # data, and the two generator targets those runs are scored against.
    COLOUR = {"current": "darkblue", "conductance": "tab:blue",
              "target current": "tab:green", "target conductance": "darkgreen"}
    n = len(panels)
    ncol = 4
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(21, 4.6 * nrow),
                             sharex=True, sharey=True)
    for ax, (label, family, w) in zip(axes.ravel(), panels):
        ref = init_draw(w.size, args.scale)
        target = family.startswith("target")
        ax.hist(w[w > 0], bins=bins, color=COLOUR.get(family, "tab:blue"), alpha=0.8,
                label="generator's $|W|$" if target else "trained $|W|$")
        ax.hist(ref[ref > 0], bins=bins, histtype="step", color="tab:red", lw=1.6,
                label=f"randn-scaled init, scale {args.scale:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.text(0.0, 1.02, f"{label}   ({family})", transform=ax.transAxes,
                va="bottom", fontsize=10)
        if target:
            ax.legend(fontsize=9, frameon=False, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes.ravel()[max(n - ncol, 0):n]:
        ax.set_xlabel("|W|", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("edges", fontsize=12)
    axes[0, 0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
