"""Bar plot for the meeting deck: R2 of W, tau and V_rest at model noise sigma = 0, 0.05, 0.5, for
least squares, the Known-ODE and the GNN; mean +- SD over the five folds (ddof 0, as the tables).

    python tools/noise_recovery_bars.py

Known-ODE and GNN: experiments/paper/tables/table_numbers.json (Table 1's aggregates, outlier-
corrected). Least squares: GraphData/lstsq_noise/lstsq_noise_table.json (tools/lstsq_noise_table.py,
the same outlier bands). Writes presentation/Fig/meeting_noise_recovery_bars.{pdf,png}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE = ("noise_free", "noise_005", "noise_05")
SIGMA = ("0", "0.05", "0.5")
COLORS = {"least squares": "#8c8c8c", "Known-ODE": "#4c72b0", "GNN": "#dd8452"}


def main():
    tn = json.load(open(os.path.join(REPO, "experiments", "paper", "tables", "table_numbers.json")))
    ls = json.load(open("/groups/saalfeld/home/allierc/GraphData/lstsq_noise/lstsq_noise_table.json"))
    src = {"least squares": (ls, {"W": "W_R2", "tau": "tau_R2", "V_rest": "V_rest_R2"}),
           "Known-ODE": (tn["table1_known_ode"], {"W": "Wij_R2", "tau": "tau_R2", "V_rest": "V_rest_R2"}),
           "GNN": (tn["table1_gnn"], {"W": "Wij_R2", "tau": "tau_R2", "V_rest": "V_rest_R2"})}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    x = np.arange(len(NOISE)); w = 0.26
    for ax, (q, lab) in zip(axes, (("W", r"$R^2_{\hat W}$"), ("tau", r"$R^2_{\hat\tau}$"),
                                     ("V_rest", r"$R^2_{\hat V^{rest}}$"))):
        for k, (name, (d, keys)) in enumerate(src.items()):
            m = [d[n][keys[q]][0] for n in NOISE]; s = [d[n][keys[q]][1] for n in NOISE]
            ax.bar(x + (k - 1) * w, m, w, yerr=s, capsize=2, color=COLORS[name], label=name,
                   error_kw={"elinewidth": 0.8})
        ax.set_xticks(x); ax.set_xticklabels([rf"$\sigma$ = {v}" for v in SIGMA], fontsize=10)
        ax.set_ylim(0.7, 1.02)
        ax.set_title(lab, fontsize=12, loc="left")
        ax.tick_params(labelsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("R² (mean ± SD, 5 folds)", fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=10, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    base = os.path.join(REPO, "presentation", "Fig", "meeting_noise_recovery_bars")
    fig.savefig(base + ".pdf", bbox_inches="tight"); fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    print("wrote", base + ".{pdf,png}")


if __name__ == "__main__":
    main()
