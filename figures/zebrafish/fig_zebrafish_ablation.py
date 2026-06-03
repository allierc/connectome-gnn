"""Per-cell-type ablation summary for the zebrafish HD circuit (v1).

Reads the final-epoch training metrics of every
``zebrafish_hd_si_ipn12_v1_ablate_<token>`` run (structural knockout held
through training) and ranks the knockouts by how much they degrade the
swim-integration rollout relative to the unablated ``ablate_none`` baseline.

  (a) ranked rollout RMSE per knockout, coloured by cell family, baseline marked.
  (b) impact (Delta RMSE vs baseline) against ablated-population size.

Run:
  /workspace/.conda_envs/neural-graph-linux/bin/python \
      figures/zebrafish/fig_zebrafish_ablation.py
"""
from __future__ import annotations

import csv
import glob
import os

import numpy as np

_REPO = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
LOG_ROOT = os.environ.get(
    "GNN_ABLATION_LOG_ROOT",
    "/groups/saalfeld/home/allierc/GraphData/log/zebrafish")
PREFIX = "zebrafish_hd_si_ipn12_v1_ablate_"
MAP_CSV = os.path.join(_REPO, "figures", "zebrafish",
                       "zebrafish_connectome_HD_IPN12", "functional",
                       "bodyid_zapbench_map.csv")

# family -> colour (matches the connectome-summary palette)
FAM_COLOR = {"IPN12": "#7b3fa0", "IPNd": "#d62728", "IPNds": "#ff7f0e",
             "RIPN": "#1f77b4", "pt-IPN": "#2ca02c"}


def _family(token: str) -> str:
    if token.startswith("IPN12"):
        return "IPN12"
    if token.startswith("IPNds"):
        return "IPNds"
    if token.startswith("IPNd"):
        return "IPNd"
    if token.startswith("RIPN"):
        return "RIPN"
    if token.startswith("pt"):
        return "pt-IPN"
    return "other"


def _final_metrics(token: str):
    mlog = os.path.join(LOG_ROOT, PREFIX + token, "tmp_training", "metrics.log")
    if not os.path.isfile(mlog):
        return None
    rows = list(csv.DictReader(open(mlog)))
    if not rows:
        return None
    r = rows[-1]
    return dict(rmse=float(r["rmse_roll_deg"]), pi=float(r["pi_acc"]),
                r1k=float(r["r_roll_1k"]), fwhm=float(r["fwhm_deg"]),
                epoch=int(r["epoch"]))


def _noise_band() -> float:
    """2-sigma within-run RMSE noise: 2 x median over runs of the late-epoch
    (>=4) rollout-RMSE std. Deltas inside +-band are within training noise and
    are not interpretable as a real ablation effect (single seed per run)."""
    sp = []
    for d in glob.glob(os.path.join(LOG_ROOT, PREFIX + "*")):
        r = list(csv.DictReader(open(os.path.join(d, "tmp_training", "metrics.log"))))
        rm = [float(x["rmse_roll_deg"]) for x in r if int(x["epoch"]) >= 4]
        if len(rm) >= 3:
            sp.append(float(np.std(rm)))
    return 2.0 * float(np.median(sp)) if sp else 0.0


def _pop_sizes():
    """token -> number of cells, from the 839-cell connectome map. The ablation
    token matches the `type` column (pt_IPN1 <-> pt-IPN1); a bare-family token
    (e.g. 'IPNd', 'IPNds') groups every subtype with that prefix."""
    import pandas as pd
    m = pd.read_csv(MAP_CSV)
    t = m["type"].astype(str)
    sizes = {}

    def count(tok):
        name = tok.replace("pt_IPN", "pt-IPN")
        exact = int((t == name).sum())
        if exact:
            return exact
        return int(t.str.startswith(name).sum())   # bare-family fallback
    return count


def main():
    tokens = sorted(os.path.basename(d)[len(PREFIX):]
                    for d in glob.glob(os.path.join(LOG_ROOT, PREFIX + "*")))
    data = {tok: _final_metrics(tok) for tok in tokens}
    data = {k: v for k, v in data.items() if v is not None}
    base = data.pop("none")
    count = _pop_sizes()

    items = sorted(data.items(), key=lambda kv: kv[1]["rmse"])
    names = [k for k, _ in items]
    rmse = np.array([v["rmse"] for _, v in items])
    fam = [_family(k) for k in names]
    cols = [FAM_COLOR.get(f, "0.5") for f in fam]
    sizes = np.array([count(k) for k in names], dtype=float)
    band = _noise_band()                       # 2-sigma within-run noise (deg)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(13, 9), gridspec_kw=dict(width_ratios=[1.5, 1]))

    def _panel_label(ax, lab):
        ax.text(-0.02, 1.02, lab, transform=ax.transAxes, fontsize=16,
                fontweight="bold", va="bottom", ha="right")

    # (a) ranked horizontal bars; grey band = within-run training noise.
    y = np.arange(len(names))
    axa.axvspan(base["rmse"] - band, base["rmse"] + band, color="0.88", zorder=0)
    axa.barh(y, rmse, color=cols, height=0.78, zorder=2)
    axa.axvline(base["rmse"], color="black", ls="--", lw=1.2, zorder=3)
    axa.text(base["rmse"] + band, 0.2,
             f"  baseline {base['rmse']:.1f}° $\\pm$ {band:.1f}° (noise)",
             fontsize=8, va="bottom", ha="left", color="0.25")
    axa.set_yticks(y)
    axa.set_yticklabels(names, fontsize=7)
    axa.set_xlabel("swim-integration rollout RMSE (°)  —  higher = more damaging")
    axa.set_ylim(-0.6, len(names) - 0.4)
    for spine in ("top", "right"):
        axa.spines[spine].set_visible(False)
    axa.legend(handles=[Patch(color=c, label=f) for f, c in FAM_COLOR.items()],
               fontsize=8, loc="lower right", title="cell family")
    _panel_label(axa, "a")

    # (b) impact vs population size; grey band = +-noise around zero.
    dr = rmse - base["rmse"]
    axb.axhspan(-band, band, color="0.88", zorder=0)
    axb.axhline(0, color="0.6", lw=0.8, zorder=1)
    axb.scatter(sizes, dr, c=cols, s=45, edgecolors="none", zorder=2)
    for k, s, d in zip(names, sizes, dr):
        if d > band:                           # only the significant knockouts
            axb.annotate(k, (s, d), fontsize=9, xytext=(4, 2),
                         textcoords="offset points")
    axb.set_xlabel("ablated population size (cells)")
    axb.set_ylabel("Δ rollout RMSE vs baseline (°)")
    for spine in ("top", "right"):
        axb.spines[spine].set_visible(False)
    _panel_label(axb, "b")

    fig.tight_layout()
    out = os.path.join(_REPO, "figures", "zebrafish", "fig_zebrafish_ablation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] wrote {out}")
    # also dump a sorted CSV for the paper table
    csv_out = os.path.join(_REPO, "figures", "zebrafish",
                           "fig_zebrafish_ablation.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "family", "n_cells", "rmse_deg", "delta_rmse",
                    "pi_acc", "r_roll_1k", "fwhm_deg"])
        w.writerow(["none", "-", "-", f"{base['rmse']:.2f}", "0.00",
                    f"{base['pi']:.4f}", f"{base['r1k']:.4f}", f"{base['fwhm']:.1f}"])
        for k, v in sorted(data.items(), key=lambda kv: -kv[1]["rmse"]):
            w.writerow([k, _family(k), int(count(k)), f"{v['rmse']:.2f}",
                        f"{v['rmse']-base['rmse']:+.2f}", f"{v['pi']:.4f}",
                        f"{v['r1k']:.4f}", f"{v['fwhm']:.1f}"])
    print(f"[csv ] wrote {csv_out}")


if __name__ == "__main__":
    main()
