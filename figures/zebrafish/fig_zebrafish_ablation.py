"""Per-cell-type ablation summary for the zebrafish HD circuit (v1), on the
TRIAL-based test metrics (GNN_Main -o test_plot) with a 10-fold seed band.

Each ``zebrafish_hd_si_ipn12_v1_ablate_<token>`` knockout (structural lesion
held through training) is scored by its per-trial heading RMSE over the
held-out test trials (512/run) and its integration-gain linearity. Significance
is judged against the 10 seed replicas (cv0..cv9 of the unablated v1): a lesion
matters only if it leaves the +-2 SD fold band.

  (a) ranked trial RMSE per knockout, coloured by family, with the fold band.
  (b) integration-gain linearity (|gain-1|) per knockout, with its fold band.

Run (after `run_GNN_zebrafish_hd_si_ipn12_ablation.py --mode test_plot` and the
10-fold test_plot have populated the logs):
  python figures/zebrafish/fig_zebrafish_ablation.py
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np

_REPO = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG = os.environ.get("GNN_ABLATION_LOG_ROOT",
                     "/groups/saalfeld/home/allierc/GraphData/log/zebrafish")
SUMMARY = os.path.join(LOG, "zebrafish_ipn12_ablation_runner", "test_summary.csv")
FAM_COLOR = {"IPN12": "#7b3fa0", "IPNd": "#d62728", "IPNds": "#ff7f0e",
             "RIPN": "#1f77b4", "pt-IPN": "#2ca02c"}


def _family(tok: str) -> str:
    if tok.startswith("IPN12"):
        return "IPN12"
    if tok.startswith("IPNds"):
        return "IPNds"
    if tok.startswith("IPNd"):
        return "IPNd"
    if tok.startswith("RIPN"):
        return "RIPN"
    if tok.startswith("pt"):
        return "pt-IPN"
    return "other"


def _fold_metrics():
    """Per-fold (cv0..cv9) trial RMSE + mean |gain-1| from the test logs."""
    rmse, gdev = [], []
    for i in range(10):
        p = os.path.join(LOG, f"zebrafish_hd_si_ipn12_v1_cv{i}",
                         "results_path_integration.log")
        if not os.path.isfile(p):
            continue
        gains, sec = [], None
        rm = np.nan
        for ln in open(p):
            ln = ln.strip()
            if ln.startswith("mean_trial_rmse_deg"):
                rm = float(re.search(r":\s*([0-9.]+)", ln).group(1))
            elif ln.startswith("# Integration gain"):
                sec = "g"
            elif ln.startswith("#"):
                sec = None
            elif sec == "g" and ln and not ln[0].isalpha():
                p2 = ln.split(",")
                if len(p2) >= 3:
                    try:
                        gains.append(float(p2[2]))
                    except ValueError:
                        pass
        rmse.append(rm)
        gdev.append(float(np.mean(np.abs(np.array(gains) - 1.0))) if gains else np.nan)
    return np.array(rmse), np.array(gdev)


def main():
    rows = {r["token"]: r for r in csv.DictReader(open(SUMMARY))}
    base = rows.pop("none", None)

    fold_rmse, fold_gdev = _fold_metrics()
    rb_m, rb_s = float(np.nanmean(fold_rmse)), float(np.nanstd(fold_rmse, ddof=1))
    gb_m, gb_s = float(np.nanmean(fold_gdev)), float(np.nanstd(fold_gdev, ddof=1))

    toks = sorted(rows, key=lambda t: float(rows[t]["trial_rmse_deg"]))
    rmse = np.array([float(rows[t]["trial_rmse_deg"]) for t in toks])
    gdev = np.array([float(rows[t]["mean_abs_gain_minus_1"]) for t in toks])
    cols = [FAM_COLOR.get(_family(t), "0.5") for t in toks]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 9),
                                   gridspec_kw=dict(width_ratios=[1.5, 1]))

    def _plabel(ax, lab):
        ax.text(-0.02, 1.02, lab, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="bottom", ha="right")

    y = np.arange(len(toks))
    # (a) trial RMSE; grey band = 10-fold +-2 SD seed band.
    axa.axvspan(rb_m - 2 * rb_s, rb_m + 2 * rb_s, color="0.88", zorder=0)
    axa.barh(y, rmse, color=cols, height=0.78, zorder=2)
    axa.axvline(rb_m, color="0.4", lw=1.0, zorder=3)
    if base is not None:
        axa.axvline(float(base["trial_rmse_deg"]), color="black", ls="--",
                    lw=1.2, zorder=3)
        axa.text(float(base["trial_rmse_deg"]), 0.1,
                 f" baseline {float(base['trial_rmse_deg']):.1f}°",
                 fontsize=8, va="bottom", ha="left", color="0.25")
    axa.text(rb_m + 2 * rb_s, len(toks) - 0.6,
             f" 10-fold seed band {rb_m:.1f}$\\pm${2 * rb_s:.1f}°",
             fontsize=8, va="top", ha="left", color="0.35")
    axa.set_yticks(y); axa.set_yticklabels(toks, fontsize=7)
    axa.set_xlabel("per-trial heading RMSE (°)  —  512 test trials/run")
    axa.set_ylim(-0.6, len(toks) - 0.4)
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)
    axa.legend(handles=[Patch(color=c, label=f) for f, c in FAM_COLOR.items()],
               fontsize=8, loc="lower right", title="cell family")
    _plabel(axa, "a")

    # (b) integration-gain linearity |gain-1|; grey band = 10-fold +-2 SD.
    order = np.argsort(gdev)
    yb = np.arange(len(toks))
    axb.axvspan(max(0, gb_m - 2 * gb_s), gb_m + 2 * gb_s, color="0.88", zorder=0)
    axb.barh(yb, gdev[order], color=[cols[i] for i in order], height=0.78, zorder=2)
    axb.axvline(gb_m, color="0.4", lw=1.0, zorder=3)
    axb.set_yticks(yb); axb.set_yticklabels([toks[i] for i in order], fontsize=7)
    axb.set_xlabel(r"integration-gain error  $\langle|g-1|\rangle$")
    axb.set_ylim(-0.6, len(toks) - 0.4)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    _plabel(axb, "b")

    fig.tight_layout()
    out = os.path.join(_REPO, "figures", "zebrafish", "fig_zebrafish_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {out}")

    # self-contained ranked table for the paper (high RMSE first)
    csv_out = os.path.join(_REPO, "figures", "zebrafish",
                           "fig_zebrafish_ablation.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "family", "trial_rmse_deg", "pi_acc",
                    "abs_gain_minus_1", "in_fold_band"])
        if base is not None:
            w.writerow(["none", "-", f"{float(base['trial_rmse_deg']):.3f}",
                        f"{float(base['pi_acc']):.4f}",
                        f"{float(base['mean_abs_gain_minus_1']):.3f}", "baseline"])
        for t in sorted(rows, key=lambda x: -float(rows[x]["trial_rmse_deg"])):
            v = rows[t]; rr = float(v["trial_rmse_deg"])
            inb = "no" if (rr > rb_m + 2 * rb_s or rr < rb_m - 2 * rb_s) else "yes"
            w.writerow([t, _family(t), f"{rr:.3f}", f"{float(v['pi_acc']):.4f}",
                        f"{float(v['mean_abs_gain_minus_1']):.3f}", inb])
    print(f"[csv ] wrote {csv_out}")
    n_sig = int(np.sum((rmse > rb_m + 2 * rb_s) | (rmse < rb_m - 2 * rb_s)))
    print(f"[stat] fold band trial-RMSE {rb_m:.2f}+-{rb_s:.2f}° (2SD {2 * rb_s:.2f}); "
          f"{n_sig}/{len(toks)} knockouts outside it")


if __name__ == "__main__":
    main()
