"""Per-block encoding diagnostic: within each ZAPBench block, how much of the
bump-pool ΔF/F is explained by the stimulus vs the fish's own swim, and where
does forward-swim carry unique variance? Tells us — before any retraining —
which blocks are worth wiring in as tasks.

Companion to the rotation-only encoding diagnostic
(fig_zebrafish_covariate_encoding.py). Same blocked-CV ridge + circular-shift
surrogate, run independently in each of the 9 blocks. Covariates available in
every block: the per-frame stimulus code (one-hot of its block levels) and the
two swim behaviours forward / turning (both lagged). Nested models per block:

  stim     one-hot(stim code) + lags
  +turn    + turning
  +fwd     + forward
  FULL     + turning + forward

Reported per block: held-out R²(FULL) (how explainable the block's bump
activity is at all) and ΔR²(forward | stim) with a circular-shift null
(does forward add *aligned* variance beyond the stimulus).

  python figures/zebrafish/fig_zebrafish_block_encoding.py
writes figures/zebrafish/fig_zebrafish_block_encoding.png
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (os.path.join(_REPO, "src"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import fig_zebrafish_all_blocks as ab            # loader + onsets  # noqa: E402
from fig_zebrafish_covariate_encoding import (   # regression machinery  # noqa: E402
    _z, _lagged, blocked_folds, cv_r2)

N_SURR = 100
RNG = np.random.default_rng(0)
CACHE = os.path.join(_HERE, "fig_zebrafish_block_encoding.json")


def _save_cache(res, path=CACHE):
    import json
    ser = {n: (None if r is None else {
        "r2": {k: float(v) for k, v in r["r2"].items()},
        "dr2_fwd": float(r["dr2_fwd"]), "null_mu": float(r["null_mu"]),
        "null_sd": float(r["null_sd"]), "p": float(r["p"])})
        for n, r in res.items()}
    with open(path, "w") as fh:
        json.dump(ser, fh, indent=2)


def _load_cache(path=CACHE):
    import json
    with open(path) as fh:
        return json.load(fh)


def _onehot(code):
    """One-hot of a categorical per-frame code (drop one level as reference)."""
    u = np.unique(code)
    if u.size <= 1:
        return np.zeros((len(code), 0))
    cols = [(_z((code == v).astype(float))) for v in u[1:]]
    return np.column_stack(cols)


def block_diag(Y, stim, fwd, turn):
    """Return dict of held-out R² (stim/+turn/+fwd/FULL) and the forward
    ΔR² surrogate stats for ONE block. Y:(T,N) z-scored per neuron."""
    T = Y.shape[0]
    if T < 40:
        return None
    folds = blocked_folds(T, 4)
    Xs = _onehot(stim)
    Xt, Xf = _lagged(_z(turn)), _lagged(_z(fwd))
    if Xs.shape[1] == 0:                      # constant stimulus (e.g. DARK)
        Xs = np.zeros((T, 1))                 # intercept-only stimulus model

    def lag_oh(X):                            # add 0/1 lag to the one-hot block
        return np.column_stack([X, np.vstack([X[:1], X[:-1]])]) if X.shape[1] else X
    Xs = lag_oh(Xs)
    r2 = {
        "stim": cv_r2(Xs, Y, folds),
        "+turn": cv_r2(np.column_stack([Xs, Xt]), Y, folds),
        "+fwd": cv_r2(np.column_stack([Xs, Xf]), Y, folds),
        "FULL": cv_r2(np.column_stack([Xs, Xt, Xf]), Y, folds),
    }
    dr2 = (r2["+fwd"] - r2["stim"]).mean()
    fz = _z(fwd)
    null = np.empty(N_SURR)
    for i in range(N_SURR):
        sh = int(RNG.integers(10, T - 10))
        null[i] = (cv_r2(np.column_stack([Xs, _lagged(np.roll(fz, sh))]),
                         Y, folds) - r2["stim"]).mean()
    p = (1 + np.sum(null >= dr2)) / (1 + N_SURR)
    return dict(r2={k: v.mean() for k, v in r2.items()},
                dr2_fwd=dr2, null_mu=null.mean(), null_sd=null.std(), p=p)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--redraw", action="store_true",
                    help="skip the (slow) recompute; reload cached per-block "
                         "stats and only redraw the figure")
    args = ap.parse_args()
    if args.redraw and os.path.isfile(CACHE):
        print(f"[redraw] from {CACHE}")
        _figure(_load_cache())
        return

    d = ab.load_full_recording()
    Yfull = np.asarray(d["calcium"], np.float64)
    edges = ab.ONSETS + [d["T"]]

    res = {}
    for i, name in enumerate(ab.TASKS):
        a, b = edges[i], edges[i + 1]
        Y = Yfull[a:b]
        keep = np.isfinite(Y).all(0) & (Y.std(0) > 1e-6)
        Y = Y[:, keep]
        Y = (Y - Y.mean(0)) / Y.std(0)
        r = block_diag(Y, d["stim"][a:b], d["forward"][a:b], d["turning"][a:b])
        res[name] = r
        if r:
            print(f"[{name:9s}] R2 stim={r['r2']['stim']:+.3f} "
                  f"+turn={r['r2']['+turn']:+.3f} +fwd={r['r2']['+fwd']:+.3f} "
                  f"FULL={r['r2']['FULL']:+.3f}  ΔR2_fwd={r['dr2_fwd']:+.3f} "
                  f"(null {r['null_mu']:+.3f}±{r['null_sd']:.3f}, p={r['p']:.3f})")

    _save_cache(res)
    _figure(res)


def _figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ab.TASKS if res[n]]
    x = np.arange(len(names))
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    # (a) how explainable each block is: stim vs FULL
    stim = [res[n]["r2"]["stim"] for n in names]
    full = [res[n]["r2"]["FULL"] for n in names]
    axA.bar(x - 0.2, stim, 0.4, color="0.6", label="stimulus only")
    axA.bar(x + 0.2, full, 0.4, color="tab:blue", label="full (stim+turn+fwd)")
    axA.set_ylabel("held-out $R^2$ (bump pool)")
    axA.set_ylim(-0.2, 0.6)
    axA.set_xticks(x)
    axA.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axA.tick_params(labelbottom=True)
    axA.text(-0.06, 1.0, "a", transform=axA.transAxes, ha="right", va="top",
             fontsize=16, fontweight="bold")
    axA.legend(fontsize=8, loc="upper right")
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)

    # (b) unique forward variance per block, vs circular-shift null
    dr2 = np.array([res[n]["dr2_fwd"] for n in names])
    mu = np.array([res[n]["null_mu"] for n in names])
    sd = np.array([res[n]["null_sd"] for n in names])
    sig = np.array([res[n]["p"] < 0.05 for n in names])
    axB.bar(x, dr2, 0.55, color=np.where(sig, "tab:purple", "0.7"))
    axB.errorbar(x, mu, yerr=2 * sd, fmt="none", ecolor="0.3", elinewidth=1,
                 capsize=3, label="surrogate null (±2 SD)")
    for xi, (v, s) in enumerate(zip(dr2, sig)):
        if s:
            axB.text(xi, v + 0.003, "*", ha="center", fontsize=12,
                     color="tab:purple")
    axB.axhline(0, color="0.4", lw=0.6)
    axB.set_ylabel("$\\Delta R^2$(forward $\\mid$ stim)")
    axB.set_ylim(-0.2, 0.6)
    axB.set_xticks(x); axB.set_xticklabels(names, rotation=30, ha="right",
                                           fontsize=9)
    axB.text(-0.06, 1.0, "b", transform=axB.transAxes, ha="right", va="top",
             fontsize=16, fontweight="bold")
    axB.legend(fontsize=8, loc="upper right")
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)

    fig.tight_layout()
    out = os.path.join(_HERE, "fig_zebrafish_block_encoding.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {out}")


if __name__ == "__main__":
    main()
