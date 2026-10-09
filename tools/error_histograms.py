"""Recovery-error histograms for the experiment report (presentation/Fig/expNN_errors[_cvNN].png).

    python tools/error_histograms.py 15            # pooled over the five folds + one per fold

One panel per recovered quantity (W, tau, V_rest), one row per group of runs: a
step histogram, log count, of learned - true for every edge (W) or neuron (tau,
V_rest), one colour per run of the row. The pairs are the ones the plot pass
scored -- results/extras/recovered_pairs.npz written by `-o test_plot`
(recovery_figures.write_recovered_pairs) -- so the histograms show the errors
behind the R2 values of metrics.txt, not a re-extraction. Values beyond an
axis are counted in its end bins, so no outlier disappears from the counts.
Dotted lines: the outlier thresholds of the R2 (tau 0.1 s, V_rest 0.2).

A new experiment adds one entry to LAYOUTS: its rows, each a list of
(run pattern with {fold}, legend label).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(os.environ.get("GNN_OUTPUT_ROOT", "/groups/saalfeld/home/allierc/GraphData"), "log", "fly")
FIG_DIR = os.path.join(ROOT, "presentation", "Fig")
FOLDS = [f"cv{i:02d}" for i in range(5)]
COLORS = ["tab:blue", "tab:green", "tab:red", "tab:purple"]
QUANT = [   # (key in recovered_pairs.npz, x label, x range, outlier threshold of the R2)
    ("W", r"$\hat W_{ij} - W_{ij}$", 1.0, None),
    ("tau", r"$\hat\tau_i - \tau_i$  (s)", 0.5, 0.1),
    ("V_rest", r"$\hat V^{\mathrm{rest}}_i - V^{\mathrm{rest}}_i$", 2.0, 0.2),
]
N_BINS = 121

LAYOUTS = {
    15: [
        ("model noise", [("flyvis_noise_free_blank50_kode217_{fold}", r"$\sigma = 0$"),
                         ("flyvis_noise_005_blank50_kode217_{fold}", r"$\sigma = 0.05$"),
                         ("flyvis_noise_05_blank50_kode217_{fold}", r"$\sigma = 0.5$")]),
        (r"measurement noise ($\sigma = 0.05$)",
         [("flyvis_noise_005_blank50_kode217_{fold}", r"$\gamma = 0$"),
          ("flyvis_noise_005_010_blank50_kode217_{fold}", r"$\gamma = 0.1$"),
          ("flyvis_noise_005_020_blank50_kode217_{fold}", r"$\gamma = 0.2$")]),
        (r"connectivity ($\sigma = 0.05$)",
         [("flyvis_noise_005_blank50_kode217_{fold}", "true edges"),
          ("flyvis_noise_005_null_edges_pc_400_blank50_kode217_{fold}", "+400% null edges"),
          ("flyvis_noise_005_removed_pc_20_blank50_kode217_{fold}", "-20% edges removed"),
          ("flyvis_noise_005_removed_pc_50_blank50_kode217_{fold}", "-50% edges removed")]),
    ],
}


def errors(pattern, folds, key):
    out = []
    for f in folds:
        r = np.load(os.path.join(LOG_ROOT, pattern.format(fold=f), "results", "extras", "recovered_pairs.npz"))
        out.append(r[f"{key}_learned"].astype(np.float64) - r[f"{key}_true"].astype(np.float64))
    return np.concatenate(out)


def draw(number, folds, title, out):
    rows = LAYOUTS[number]
    fig, axes = plt.subplots(len(rows), len(QUANT), figsize=(11.5, 2.9 * len(rows)), squeeze=False)
    for r, (row_label, runs) in enumerate(rows):
        for c, (key, xlabel, lim, thr) in enumerate(QUANT):
            ax = axes[r, c]
            bins = np.linspace(-lim, lim, N_BINS)
            for k, (pattern, label) in enumerate(runs):
                d = np.clip(errors(pattern, folds, key), -lim, lim)   # beyond the axis -> the end bins
                ax.hist(d, bins=bins, histtype="step", color=COLORS[k], lw=1.1, label=label)
            ax.axvline(0, color="0.6", lw=0.6, zorder=0)
            if thr:
                for s in (-thr, thr):
                    ax.axvline(s, color="0.6", lw=0.6, ls=":", zorder=0)
            ax.set_yscale("log"); ax.set_xlim(-lim, lim)
            ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel("count")
                ax.set_title(row_label, loc="left", fontsize=9)
            if c == 1:
                ax.legend(fontsize=7, frameon=False, loc="upper right")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    number = int(sys.argv[1])
    draw(number, FOLDS, f"Experiment {number}: learned - true, five folds pooled",
         os.path.join(FIG_DIR, f"exp{number:02d}_errors.png"))
    for f in FOLDS:
        draw(number, [f], f"Experiment {number}: learned - true, fold {f}",
             os.path.join(FIG_DIR, f"exp{number:02d}_errors_{f}.png"))


if __name__ == "__main__":
    main()
