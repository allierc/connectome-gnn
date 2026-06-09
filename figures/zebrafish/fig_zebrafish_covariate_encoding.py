"""Encoding diagnostic: how much of the real bump-pool ΔF/F is explained by the
locally-available covariates, and does *forward-swim* add variance beyond heading?

This is the ENCODING complement to the paper's Fig. 11 (a DECODING analysis:
activity -> heading, which fails at per-frame r=0.14). Here we run the reverse
regression --- covariates -> each bump neuron's ΔF/F --- over the ZAPBench
rotation block, to ask whether there is a *second coded variable* hiding in the
non-periodic part of the activity.

Covariates (frame grid, 656 fr ≈ 600 s; same block/rows as Fig. 12/14):
  HD / stimulus (PERIODIC):  cosθ, sinθ, ω, |ω|
  behaviour:                 turning (L/R tail-EMG asym; tracks ω, r≈0.76)
                             forward (bilateral swim amplitude; slow, ω-indep)
  (raw EMG ch0/1 are added automatically if the .10chFlt is on disk.)

Nested ridge models, blocked 5-fold CV (contiguous folds — random folds leak on
autocorrelated calcium), held-out R² per neuron:
  HD            cosθ,sinθ,ω,|ω|
  HD+turn       + turning
  HD+fwd        + forward
  FULL          + turning + forward
  HD+slow       + a generic slow cosine basis (ceiling for "any slow drift")

Decisive test for a second variable:
  ΔR²(forward | HD) = R²(HD+fwd) − R²(HD), vs a circular-shift surrogate null
  for forward (same spectrum, scrambled phase). Real >> surrogate ⇒ forward's
  *temporal alignment* with the activity is real, not a generic slow nuisance.

  python figures/zebrafish/fig_zebrafish_covariate_encoding.py
writes figures/zebrafish/fig_zebrafish_covariate_encoding.png
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

from sklearn.linear_model import Ridge  # noqa: E402

import fig_zebrafish_rotation_covariates as cov  # reuse load_rotation_block  # noqa: E402

SRC_DT = 0.915
N_LAGS = 5                 # 0..4 frames (≈0–3.7 s) to absorb the GCaMP delay
N_SURROGATE = 200          # circular-shift surrogates for the null
ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- #
# covariate / design construction
# ----------------------------------------------------------------------------- #
def _z(x):
    x = np.asarray(x, np.float64)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-9 else 1.0)


def _lagged(col, n_lags=N_LAGS):
    """Stack lags 0..n_lags-1 of a 1-D covariate (edge-padded)."""
    col = np.asarray(col, np.float64)
    return np.stack([np.concatenate([np.full(L, col[0]), col[:len(col) - L]])
                     for L in range(n_lags)], axis=1)          # (T, n_lags)


def build_covariates(d):
    """Return dict of named (T, k) design blocks on the 656-frame grid."""
    T = d["n_fr"]
    theta = np.deg2rad(np.asarray(d["heading_deg"], np.float64))   # cumulative
    # ω at the frame grid (signed) from the cumulative frame heading.
    omega_fr = np.gradient(np.asarray(d["heading_deg"], np.float64), SRC_DT)

    blocks = {
        "HD": np.column_stack([
            _lagged(np.cos(theta)), _lagged(np.sin(theta)),
            _lagged(_z(omega_fr)), _lagged(_z(np.abs(omega_fr)))]),
        "turn": _lagged(_z(d["turning"])),
        "fwd": _lagged(_z(d["forward"])),
    }
    if d.get("emg_L") is not None:
        eL = np.nan_to_num(d["emg_L"], nan=np.nanmean(d["emg_L"]))
        eR = np.nan_to_num(d["emg_R"], nan=np.nanmean(d["emg_R"]))
        blocks["emg"] = np.column_stack([_lagged(_z(eL)), _lagged(_z(eR))])

    # generic slow basis: half/one/.../3 cycles over the block — a fair "any
    # slow drift" control with a handful of df.
    t = np.arange(T) / T
    slow = []
    for k in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        slow.append(np.cos(2 * np.pi * k * t))
        slow.append(np.sin(2 * np.pi * k * t))
    blocks["slow"] = np.column_stack(slow)
    return blocks


# ----------------------------------------------------------------------------- #
# blocked-CV ridge, held-out R² per neuron
# ----------------------------------------------------------------------------- #
def blocked_folds(T, k=5):
    edges = np.linspace(0, T, k + 1).astype(int)
    return [(np.r_[0:edges[i], edges[i + 1]:T], np.arange(edges[i], edges[i + 1]))
            for i in range(k)]


def cv_r2(X, Y, folds):
    """Pooled held-out R² per neuron. Y:(T,N) z-scored per neuron. α picked per
    fold by a quick inner search on a 80/20 split of the train block."""
    T, N = Y.shape
    pred = np.zeros_like(Y)
    for tr, te in folds:
        # inner alpha pick (single split, scored on a tail of the train block)
        cut = int(len(tr) * 0.8)
        itr, iva = tr[:cut], tr[cut:]
        best_a, best = ALPHAS[0], -np.inf
        for a in ALPHAS:
            m = Ridge(alpha=a).fit(X[itr], Y[itr])
            s = -np.mean((m.predict(X[iva]) - Y[iva]) ** 2)
            if s > best:
                best, best_a = s, a
        pred[te] = Ridge(alpha=best_a).fit(X[tr], Y[tr]).predict(X[te])
    ss_res = np.sum((Y - pred) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(0)) ** 2, axis=0)
    return 1.0 - ss_res / np.where(ss_tot > 1e-12, ss_tot, 1.0)


def main():
    out_png = os.path.join(_HERE, "fig_zebrafish_covariate_encoding.png")
    d = cov.load_rotation_block()
    blocks = build_covariates(d)
    T = d["n_fr"]

    # target: per-neuron z-scored ΔF/F over the block (drop dead/flat neurons)
    Y = np.asarray(d["calcium"], np.float64)                  # (T, N)
    keep = np.isfinite(Y).all(0) & (Y.std(0) > 1e-6)
    Y = Y[:, keep]
    Y = (Y - Y.mean(0)) / Y.std(0)
    N = Y.shape[1]
    folds = blocked_folds(T, 5)
    print(f"[data] {T} frames, {N} bump neurons; covariate blocks: "
          f"{ {k: v.shape[1] for k, v in blocks.items()} }")

    def design(names):
        return np.column_stack([blocks[n] for n in names])

    have_emg = "emg" in blocks
    full_names = ["HD", "turn", "fwd"] + (["emg"] if have_emg else [])
    models = {
        "HD": ["HD"],
        "HD+turn": ["HD", "turn"],
        "HD+fwd": ["HD", "fwd"],
        "HD+slow": ["HD", "slow"],
        "FULL": full_names,
    }
    r2 = {name: cv_r2(design(ns), Y, folds) for name, ns in models.items()}
    for name in models:
        print(f"[R2] {name:9s} mean={r2[name].mean():+.3f}  "
              f"median={np.median(r2[name]):+.3f}  "
              f"frac>0.05={np.mean(r2[name] > 0.05):.2f}")

    # unique variance of forward beyond HD, per neuron
    dr2_fwd = r2["HD+fwd"] - r2["HD"]
    dr2_turn = r2["HD+turn"] - r2["HD"]
    dr2_slow = r2["HD+slow"] - r2["HD"]
    print(f"[ΔR²|HD] forward mean={dr2_fwd.mean():+.4f}  "
          f"turning mean={dr2_turn.mean():+.4f}  "
          f"slowbasis mean={dr2_slow.mean():+.4f}")

    # circular-shift surrogate null for forward (same spectrum, scrambled phase)
    base = design(["HD"])
    fwd_raw = _z(d["forward"])
    null = np.empty(N_SURROGATE)
    for i in range(N_SURROGATE):
        sh = int(RNG.integers(20, T - 20))
        Xs = np.column_stack([base, _lagged(np.roll(fwd_raw, sh))])
        r2s = cv_r2(Xs, Y, folds)
        null[i] = (r2s - r2["HD"]).mean()
    p = (1 + np.sum(null >= dr2_fwd.mean())) / (1 + N_SURROGATE)
    z = (dr2_fwd.mean() - null.mean()) / (null.std() + 1e-12)
    print(f"[surrogate] forward ΔR² real={dr2_fwd.mean():+.4f}  "
          f"null={null.mean():+.4f}±{null.std():.4f}  z={z:+.1f}  p={p:.4f}")

    _figure(out_png, d, blocks, Y, folds, r2, dr2_fwd, dr2_turn, dr2_slow,
            null, z, p, design, models)
    print(f"[fig] wrote {out_png}")


# ----------------------------------------------------------------------------- #
def _figure(out_png, d, blocks, Y, folds, r2, dr2_fwd, dr2_turn, dr2_slow,
            null, z, p, design, models):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = d["n_fr"]; t = d["t_frame"]
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.34)

    # (a) population held-out R² per model
    axa = fig.add_subplot(gs[0, 0])
    names = list(models.keys())
    means = [r2[n].mean() for n in names]
    axa.bar(range(len(names)), means, color="0.6")
    axa.set_xticks(range(len(names))); axa.set_xticklabels(names, rotation=30,
                                                           ha="right", fontsize=8)
    axa.set_ylabel("mean held-out $R^2$")
    axa.set_title("(a) encoding $R^2$ per model\n(bump-pool, blocked 5-fold CV)",
                  fontsize=9)
    for i, m in enumerate(means):
        axa.text(i, m + 0.002, f"{m:.3f}", ha="center", fontsize=7)
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)

    # (b) unique ΔR² of forward beyond HD: per-neuron, vs surrogate null
    axb = fig.add_subplot(gs[0, 1])
    axb.hist(dr2_fwd, bins=30, color="tab:purple", alpha=0.8,
             label=f"forward (mean {dr2_fwd.mean():+.3f})")
    axb.axvline(dr2_fwd.mean(), color="tab:purple", lw=1.5)
    lo, hi = np.percentile(null, [2.5, 97.5])
    axb.axvspan(lo, hi, color="0.7", alpha=0.5, label="surrogate null 95%")
    axb.axvline(0, color="0.4", lw=0.6)
    axb.set_xlabel("$\\Delta R^2$(forward $\\mid$ HD), per neuron")
    axb.set_ylabel("neurons")
    axb.set_title(f"(b) unique forward variance\nz={z:+.1f}, p={p:.3f}",
                  fontsize=9)
    axb.legend(fontsize=7, loc="upper right")
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    # (c) ΔR² forward vs turning vs slow-basis (per-neuron means)
    axc = fig.add_subplot(gs[0, 2])
    data = [dr2_fwd, dr2_turn, dr2_slow]
    labs = ["forward", "turning", "slow\nbasis"]
    parts = axc.violinplot(data, showmeans=True, showextrema=False)
    for pc, c in zip(parts["bodies"], ("tab:purple", "tab:olive", "0.6")):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    axc.axhline(0, color="0.4", lw=0.6)
    axc.set_xticks([1, 2, 3]); axc.set_xticklabels(labs, fontsize=8)
    axc.set_ylabel("$\\Delta R^2 \\mid$ HD")
    axc.set_title("(c) unique variance by covariate", fontsize=9)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)

    # (d-e) example neurons: observed vs HD-only vs HD+forward prediction
    order = np.argsort(dr2_fwd)[::-1]
    Xhd, Xhf = design(["HD"]), design(["HD", "fwd"])

    def fold_pred(X):
        pr = np.zeros_like(Y)
        for tr, te in folds:
            pr[te] = Ridge(alpha=10.0).fit(X[tr], Y[tr]).predict(X[te])
        return pr
    Phd, Phf = fold_pred(Xhd), fold_pred(Xhf)
    for j, ax_i in enumerate((gs[1, 0], gs[1, 1])):
        ax = fig.add_subplot(ax_i)
        nidx = order[j]
        ax.plot(t, Y[:, nidx], color="0.3", lw=0.8, label="observed ΔF/F (z)")
        ax.plot(t, Phd[:, nidx], color="tab:green", lw=1.0,
                label=f"HD fit ($R^2$={r2['HD'][nidx]:.2f})")
        ax.plot(t, Phf[:, nidx], color="tab:purple", lw=1.0,
                label=f"HD+fwd ($R^2$={r2['HD+fwd'][nidx]:.2f})")
        ax.set_title(f"(neuron #{nidx}: ΔR²_fwd={dr2_fwd[nidx]:+.2f})", fontsize=9)
        ax.set_xlabel("time (s)"); ax.set_ylabel("z ΔF/F")
        ax.legend(fontsize=6, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # (f) overlay forward covariate for reference
    axf = fig.add_subplot(gs[1, 2])
    axf.plot(t, _z(d["forward"]), color="tab:purple", lw=0.8, label="forward swim")
    axf.plot(t, _z(np.abs(np.gradient(np.asarray(d["heading_deg"], np.float64),
                                      SRC_DT))), color="0.5", lw=0.6,
             label="|ω| (periodic)")
    axf.set_title("(f) forward swim vs |ω|", fontsize=9)
    axf.set_xlabel("time (s)"); axf.set_ylabel("z")
    axf.legend(fontsize=7, loc="upper right")
    for sp in ("top", "right"):
        axf.spines[sp].set_visible(False)

    fig.suptitle("Encoding diagnostic — covariates → real bump-pool ΔF/F "
                 "(ZAPBench rotation block): is there a second coded variable?",
                 fontsize=11)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
