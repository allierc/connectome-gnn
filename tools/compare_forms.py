"""Compare one run's driving-force gain against a run that cannot have one.

THE ONLY TEST HERE WHOSE NULL IS NOT FALSE BY CONSTRUCTION. Within a run, the
two forms are nested and paired -- the current form is the conductance form
minus the u*v_i column -- so its residual can only be larger and a test of "are
these two distributions different" rejects on any sample big enough. Across
runs the comparison is honest: take the per-edge R2 gain of the run in question
and the same quantity from a model whose message provably has no driving force
(a current GNN on current-generated data), and ask whether they could be draws
from one distribution. That is a two-sample Kolmogorov-Smirnov test on
independent samples with a null that could be true.

Reads results/form_comparison.npz, written by `-o plot`.

Usage:
    python tools/compare_forms.py <run> --null <run-with-no-driving-force>
    python tools/compare_forms.py <run> --null <run> --figure
"""

import argparse
import os

import numpy as np

LOG_ROOT = os.environ.get("GNN_LOG_ROOT",
                          f"{os.environ['GNN_OUTPUT_ROOT']}/log/fly")


def load(run):
    path = (run if run.endswith(".npz")
            else os.path.join(LOG_ROOT, run, "results", "form_comparison.npz"))
    if not os.path.exists(path):
        raise SystemExit(f"no form_comparison.npz for {run}\n  looked in {path}\n"
                         f"  it is written by `-o plot`; re-run the plot pass")
    z = np.load(path)
    gain = z["conductance_form_r2"] - z["current_form_r2"]
    return dict(gain=gain[np.isfinite(gain)], t=z["t_driving_force"],
                frames=z["frames_per_edge"], path=path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--null", required=True,
                    help="a run whose message has no driving force; the empirical null")
    ap.add_argument("--figure", action="store_true",
                    help="write form_comparison_vs_null.png beside the run's results")
    args = ap.parse_args()

    from scipy.stats import ks_2samp, mannwhitneyu

    a, b = load(args.run), load(args.null)
    ks = ks_2samp(a["gain"], b["gain"])
    mw = mannwhitneyu(a["gain"], b["gain"], alternative="greater")

    def q(x):
        return " / ".join(f"{v:+.4f}" for v in np.quantile(x, [0.1, 0.5, 0.9]))

    print(f"run   {args.run}")
    print(f"  per-edge R² gain from the driving-force column, [10/50/90]: {q(a['gain'])}"
          f"   mean {a['gain'].mean():+.4f} ± {a['gain'].std(ddof=1):.4f}"
          f"   ({a['gain'].size:,} edges)")
    print(f"null  {args.null}")
    print(f"  the same quantity where there is no driving force, [10/50/90]: {q(b['gain'])}"
          f"   mean {b['gain'].mean():+.4f} ± {b['gain'].std(ddof=1):.4f}"
          f"   ({b['gain'].size:,} edges)")
    # D is the largest vertical gap between the two cumulative distributions:
    # 0 means one curve lies on the other, 1 that they do not overlap at all.
    print(f"Kolmogorov-Smirnov: D = {ks.statistic:.4f}, p = {ks.pvalue:.3g}")
    print(f"Mann-Whitney (run > null): U p = {mw.pvalue:.3g}")
    # With hundreds of thousands of edges any p-value is a formality; the size
    # of the separation is the claim, so it is printed beside them.
    _ratio = (a["gain"].mean() / b["gain"].mean()) if b["gain"].mean() else float("inf")
    print(f"the run's mean gain is {_ratio:.0f}x the null's, and D = {ks.statistic:.2f} "
          f"of a possible 1.00 -- with {a['gain'].size:,} edges the p-values are a "
          f"formality and the separation is the result")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        for x, c, lab in ((a["gain"], "tab:blue", args.run),
                          (b["gain"], "tab:red", args.null)):
            xs = np.sort(x)
            ax.plot(xs, np.arange(1, xs.size + 1) / xs.size, color=c, lw=2, label=lab)
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlabel("per-edge $R^2$ gained by the driving-force column", fontsize=13)
        ax.set_ylabel("fraction of edges at or below", fontsize=13)
        ax.legend(fontsize=9, frameon=False, loc="lower right")
        ax.text(0.004, 1.02, f"KS D = {ks.statistic:.3f}", transform=ax.transAxes,
                va="bottom", fontsize=12, fontweight="bold")
        out = os.path.join(os.path.dirname(a["path"]), "form_comparison_vs_null.png")
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
