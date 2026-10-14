"""The parameter fit's W with and without the activation floor on the presynaptic drive.

    PYTHONPATH=src python tools/fit_floor_compare.py [run] [--device cuda:0]

The per-edge fit keeps, on each edge, only the frames where the sender's relu(v_j) is above a
floor: the median of the positive activations over all (edge, frame) pairs (`vj_quantile` 0.5
in extract_template_params). This runs the fit twice on the same checkpoint and frames, with
the floor (vj_quantile 0.5) and without it (vj_quantile 0: every frame with a positive drive is
kept), and draws the learned W against the true W for both, in the style of Figure 1 panel a.
Prints, per variant, the number of fitted edges (8 kept frames or more) and R2_W with its
outlier percentage as score_recovery computes them. Writes
presentation/Fig/meeting_fit_floor_<fold>.{pdf,png} and the numbers to
<run>/results/extras/fit_floor_compare.json.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "experiments", "paper", "scripts"))
from clustering_template_vs_chain import ROOT, load                # noqa: E402
from connectome_gnn.generators.ode_params import load_ode_params_for_run  # noqa: E402
from connectome_gnn.metrics import extract_template_params, score_recovery  # noqa: E402
from connectome_gnn.utils import set_data_root                     # noqa: E402
from paper_style import CM, FS_ANNOT, FS_LABEL, FS_TICK            # noqa: E402

RUN = "flyvis_noise_005_blank50_condl25_cv00"
W_RANGE = (-1.0, 2.0)          # Figure 1 panel a's axes


def main(run=RUN, device="cuda:0"):
    set_data_root(ROOT)
    dev = torch.device(device)
    log_dir, config, x_ts, edges, model = load(run, dev)
    n = int(x_ts.n_neurons)
    op = load_ode_params_for_run(config, device="cpu")
    out, pairs = {}, {}
    for name, q in (("floor", 0.5), ("no_floor", 0.0)):
        rec = extract_template_params(model, op, config=config, edges=edges, x_ts=x_ts,
                                      device=dev, n_neurons=n, vj_quantile=q)
        sc = score_recovery(rec, config)
        out[name] = {"vj_quantile": q, "vj_floor": float(rec.diagnostics["tmpl_vj_floor"]),
                     "edges_fitted": int(rec.diagnostics["tmpl_n_edges_fitted"]),
                     "edges_total": int(rec.diagnostics["tmpl_n_edges"]),
                     "Wij_R2": float(sc["Wij_R2"]), "Wij_R2_all": float(sc["Wij_R2_all"]),
                     "Wij_pct_outliers": float(sc["Wij_pct_outliers"]),
                     "Wij_slope": float(sc["Wij_slope"])}
        pairs[name] = rec.pairs["W"]
        print(name, json.dumps(out[name]), flush=True)
    json.dump(out, open(os.path.join(log_dir, "results", "extras", "fit_floor_compare.json"), "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(11 * CM, 5.6 * CM))
    rng = np.random.default_rng(0)
    lo, hi = W_RANGE
    for ax, (name, title) in zip(axes, (("floor", "with the filter (0.52)"), ("no_floor", "without the filter"))):
        t, l = pairs[name]
        idx = rng.choice(len(t), min(len(t), 150_000), replace=False)
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, zorder=1)
        ax.scatter(t[idx], l[idx], c="k", s=0.1, alpha=0.1, lw=0, rasterized=True, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([-1, 0.5, 2]); ax.set_yticks([-1, 0.5, 2]); ax.tick_params(labelsize=FS_TICK)
        o = out[name]
        ax.text(0.04, 0.96, f"{title}\nR²: {o['Wij_R2']:.3f}\nslope: {o['Wij_slope']:.2f}\n"
                f"outliers: {o['Wij_pct_outliers']:.1f}%\n{o['edges_fitted']:,} of {o['edges_total']:,} edges",
                transform=ax.transAxes, va="top", fontsize=FS_ANNOT)
        ax.set_xlabel("true $W_{ij}$", fontsize=FS_LABEL)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("learned $\\hat W_{ij}$", fontsize=FS_LABEL)
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.22, 1.04, letter, transform=ax.transAxes, fontsize=FS_LABEL, fontweight="bold")
    fig.tight_layout()
    base = os.path.join(REPO, "presentation", "Fig", f"meeting_fit_floor_{run.split('_')[-1]}")
    fig.savefig(base + ".pdf", dpi=300); fig.savefig(base + ".png", dpi=300)
    print("wrote", base + ".{pdf,png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=RUN)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    main(args.run, args.device)
