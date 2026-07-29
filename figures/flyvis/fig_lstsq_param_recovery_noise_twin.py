#!/usr/bin/env python
"""Twin of Fig. 12 across PROCESS-noise levels sigma in {0, 0.05, 0.5}, plus the
eigendecomposition spectrum that underlies the null/sloppy classification.

Rebuttal target
---------------
The reviewer flags an apparent internal inconsistency in Appendix C:

  * the NUMERICAL eigendecomposition reports ~6.4% of edge directions null and
    ~4.0% sloppy at sigma=0 (this analysis / Fig. 12);
  * Eq. (11) derives an EXACT null-space of dim 308,160 = 71% of edges from the
    columnar-repeat argument;
  * the empirical rollouts show only APPROXIMATE dynamical equivalence
    (r ~ 0.94-0.98).

and asks how process noise can "improve recovery" if the mechanism rests on an
exact kernel. This script does NOT touch Eq. (11). It answers the empirical
question directly: it MEASURES the numerical eigendecomposition and shows how the
null/sloppy fractions move as process noise rises. The mechanism ("noise raises
the small singular values of the per-neuron design matrix A_i") stops being an
assertion and becomes a measurement. It never required the singular values to be
exactly zero -- "sigma_k rises from eps to O(1)" is a continuous statement -- so
the correction strengthens the noise interpretation rather than undermining it.

This is a self-contained script (its own loader + solver, an exact copy of the
Fig. 12 solver from fig_lstsq_param_recovery.py plus the pooled relative-eigenvalue
spectrum). It imports nothing from the Fig. 12 script.

Outputs (into --out-dir, default: this script's directory)
----------------------------------------------------------
  * fig_lstsq_param_recovery_noise_twin.{pdf,png}
        rows = sigma in {0, 0.05, 0.5}, cols = (tau, V_rest, W). Same
        red=null / orange=sloppy / black=identifiable convention as Fig. 12,
        one representative fold (cv00) per sigma.
  * fig_lstsq_eigenspectrum_noise.{pdf,png}
        Phi_sigma(eps) = cumulative fraction of per-neuron directions whose
        relative eigenvalue w/w_max <= eps, one curve per sigma, pooled over the
        5 CV folds. Vertical lines at the null (1e-22) and sloppy (1e-12)
        tolerances: the sigma=0 curve reads ~6.4% / ~10% there, and the noise
        story is the whole curve sliding left/down.
  * fig_lstsq_param_recovery_noise_twin_fractions.{txt,tex}
        null/sloppy fractions (edge/neuron level, matching Fig. 12) and
        direction-level Phi, mean +/- SD across cv00..cv04.

Usage
-----
    # one process, all datasets, both GPUs available:
    python fig_lstsq_param_recovery_noise_twin.py --roots ROOT1 ROOT2 ...

    # split over the two GPUs (cache is shared, resumable), then assemble:
    CUDA_VISIBLE_DEVICES=0 python ... --compute-only --roots <half A> &
    CUDA_VISIBLE_DEVICES=1 python ... --compute-only --roots <half B> &
    wait
    python ... --assemble-only --roots <all>

NOTE (same caveat as Fig. 12): the finite-difference dv/dt column amplifies
voltage noise by ~1/dt, so the recovery SCATTER R^2 at sigma>0 is biased and is
NOT the trained R^2_W of Table 1. The load-bearing quantity here is the
null/sloppy spectrum (the conditioning of A_i), which is exactly what the
reviewer's 6.4%/4.0% refer to.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]          # repo root (has src/, figures/)
sys.path.insert(0, str(REPO / "src"))

from connectome_gnn.metrics import compute_r_squared_NSE

# ---------------------------------------------------------------------------
# Style (matches fig_lstsq_param_recovery.py so the twin reads as Fig. 12)
# ---------------------------------------------------------------------------
matplotlib.rc_file(str(SCRIPT_DIR / "janne.matplotlibrc"))
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi":  150,
})

_AXIS_LABEL_FS = 44
_TICK_LABEL_FS = 24
_ANNOT_FS      = 30
_LEGEND_FS     = 26
_PANEL_LBL_FS  = 40

# Two-tier eigenvalue tolerances -- identical to the Fig. 12 defaults so the
# fractions are directly comparable.
NULL_EIG_TOL   = 1e-22
SLOPPY_EIG_TOL = 1e-12
NULL_COMP_TOL  = 1e-3

# Sigma -> line color for the eigenspectrum figure. sigma=0 black baseline;
# 0.05 blue, 0.5 red (two distinct noise sources -> red/blue per house style).
SIGMA_COLORS = {0.0: "#000000", 0.05: "#1f77b4", 0.5: "#d62728"}


# ===========================================================================
# Data loading + per-neuron least-squares solve
# (self-contained copy of fig_lstsq_param_recovery.py's load_data /
#  build_in_edges / solve, plus the pooled relative-eigenvalue spectrum)
# ===========================================================================
def _load_cell_type_labels(data_root: Path, N: int):
    """Per-neuron cell-type label (str), best-effort from flyvis; falls back to
    stringified int ids."""
    nt_path = data_root / "x_list_train" / "neuron_type.zarr"
    if not nt_path.exists():
        return np.array([""] * N)
    nt = np.array(zarr.open(str(nt_path), mode="r"))
    try:
        from flyvis.network import NetworkView
        nv = NetworkView("flow/0000/000")
        types_full = np.array(
            [t.decode() if isinstance(t, bytes) else str(t)
             for t in nv.connectome.nodes["type"][:]]
        )
        names = np.unique(types_full)
        if int(nt.max()) < len(names):
            return names[nt]
    except Exception as e:
        print(f"[cell_types] flyvis lookup failed ({e}); using int ids")
    return np.array([str(int(x)) for x in nt])


def load_data(data_root: Path, dt: float):
    params = torch.load(data_root / "ode_params.pt", map_location="cpu", weights_only=False)
    edge_index = params["edge_index"].numpy()
    W_true = params["W"].numpy()
    tau_true = params["tau_i"].numpy()
    vrest_true = params["V_i_rest"].numpy()

    voltage = np.array(zarr.open(str(data_root / "x_list_train" / "voltage.zarr"), mode="r"))
    stimulus = np.array(zarr.open(str(data_root / "x_list_train" / "stimulus.zarr"), mode="r"))

    T, N = voltage.shape
    E = edge_index.shape[1]

    dv = (voltage[1:] - voltage[:-1]) / dt
    relu_v = np.maximum(voltage[:-1], 0.0)
    rhs = stimulus[:-1] - voltage[:-1]

    cell_type = _load_cell_type_labels(data_root, N)

    return dict(
        edge_index=edge_index,
        W_true=W_true, tau_true=tau_true, vrest_true=vrest_true,
        dv=dv, relu_v=relu_v, rhs=rhs,
        T=T, N=N, E=E,
        cell_type=cell_type,
    )


def build_in_edges(edge_index: np.ndarray, N: int):
    src, dst = edge_index[0], edge_index[1]
    order = np.argsort(dst)
    src_sorted = src[order]
    dst_sorted = dst[order]
    boundaries = np.searchsorted(dst_sorted, np.arange(N + 1))
    in_src = [src_sorted[boundaries[i]:boundaries[i + 1]] for i in range(N)]
    in_eidx = [order[boundaries[i]:boundaries[i + 1]] for i in range(N)]
    deg_in = np.array([len(s) for s in in_src])
    return in_src, in_eidx, deg_in


def solve(data, in_src, in_eidx, deg_in, device,
          null_eig_tol=NULL_EIG_TOL, sloppy_eig_tol=SLOPPY_EIG_TOL,
          null_comp_tol=NULL_COMP_TOL):
    """Per-neuron min-norm lstsq + two-tier degeneracy flagging. Additionally
    returns the pooled per-neuron relative eigenvalue spectrum `rel_eig`
    (concatenated w/w_max over all active neurons)."""
    N, E, T = data["N"], data["E"], data["T"]
    active_idx = np.where(deg_in > 0)[0]

    dv_d     = torch.from_numpy(data["dv"]).float().to(device)
    relu_v_d = torch.from_numpy(data["relu_v"]).float().to(device)
    rhs_d    = torch.from_numpy(data["rhs"]).float().to(device)
    ones_col = -torch.ones(T - 1, 1, device=device, dtype=torch.float64)

    K_max = int(deg_in.max())
    A_buf = torch.empty(T - 1, K_max + 2, device=device, dtype=torch.float64)
    A_buf[:, 1:2] = ones_col

    tau_lstsq   = np.full(N, np.nan, dtype=np.float64)
    vrest_lstsq = np.full(N, np.nan, dtype=np.float64)
    W_lstsq     = np.full(E, np.nan, dtype=np.float64)
    tau_null      = np.zeros(N, dtype=bool)
    tau_sloppy    = np.zeros(N, dtype=bool)
    vrest_null    = np.zeros(N, dtype=bool)
    vrest_sloppy  = np.zeros(N, dtype=bool)
    W_null        = np.zeros(E, dtype=bool)
    W_sloppy      = np.zeros(E, dtype=bool)

    # Neurons with no incoming edges are entirely unidentifiable.
    no_in = deg_in == 0
    tau_null[no_in] = True
    vrest_null[no_in] = True

    # Pooled per-neuron relative eigenvalues (w / w_max) across active neurons.
    n_dirs_total = int((deg_in[active_idx] + 2).sum())
    rel_eig_buf = torch.empty(n_dirs_total, device=device, dtype=torch.float64)
    rel_off = 0

    t0 = time.time()
    for i in tqdm(active_idx, desc="lstsq", unit="neuron"):
        K_i = len(in_src[i])
        A = A_buf[:, :K_i + 2]
        A[:, 0:1].copy_(dv_d[:, i:i + 1])
        A[:, 2:].copy_(relu_v_d[:, in_src[i]])
        A[:, 2:].neg_()
        b = rhs_d[:, i].double()

        s = A.norm(dim=0)
        s = torch.where(s > 0, s, torch.ones_like(s))
        A_s = A / s

        G = A_s.T @ A_s
        w, V = torch.linalg.eigh(G)
        w_max = w[-1]
        rel = w / w_max

        k_dirs = rel.numel()
        rel_eig_buf[rel_off:rel_off + k_dirs] = rel
        rel_off += k_dirs

        null_mask   = rel <= null_eig_tol
        sloppy_mask = (rel > null_eig_tol) & (rel <= sloppy_eig_tol)

        c = A_s.T @ b
        keep = ~(null_mask | sloppy_mask)
        inv_w = torch.where(keep, 1.0 / w, torch.zeros_like(w))
        theta_i = (V @ (inv_w * (V.T @ c))) / s

        th = theta_i.cpu().numpy()
        tau_lstsq[i]   = th[0]
        vrest_lstsq[i] = th[1]
        W_lstsq[in_eidx[i]] = th[2:]

        def _flag(mask_t, tau_arr, vrest_arr, W_arr):
            if int(mask_t.sum().item()) == 0:
                return
            V_theta = V[:, mask_t] / s.unsqueeze(1)
            V_theta = V_theta / V_theta.norm(dim=0, keepdim=True).clamp_min(1e-300)
            V_null = V_theta.abs()
            V_null = V_null / V_null.amax(dim=0, keepdim=True).clamp_min(1e-300)
            part = V_null.amax(dim=1).cpu().numpy()
            if part[0] > null_comp_tol:
                tau_arr[i] = True
            if part[1] > null_comp_tol:
                vrest_arr[i] = True
            edge_flag = part[2:] > null_comp_tol
            if edge_flag.any():
                W_arr[in_eidx[i][edge_flag]] = True

        _flag(null_mask,   tau_null,   vrest_null,   W_null)
        _flag(sloppy_mask, tau_sloppy, vrest_sloppy, W_sloppy)

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"solve time: {time.time() - t0:.1f}s   "
          f"null W={W_null.sum()}/{E}  sloppy W={W_sloppy.sum()}/{E}")

    return dict(
        tau_lstsq=tau_lstsq, vrest_lstsq=vrest_lstsq, W_lstsq=W_lstsq,
        tau_null=tau_null,     vrest_null=vrest_null,     W_null=W_null,
        tau_sloppy=tau_sloppy, vrest_sloppy=vrest_sloppy, W_sloppy=W_sloppy,
        rel_eig=rel_eig_buf[:rel_off].cpu().numpy(),
    )


# ===========================================================================
# Per-dataset compute (cached)
# ===========================================================================
def read_noise_levels(root: Path):
    """(sigma, gamma) = (noise_model_level, measurement_noise_level) from the
    dataset's generation_log.txt."""
    log = (root / "generation_log.txt").read_text()

    def _get(key):
        m = re.search(rf"^{re.escape(key)}:\s*([-\d.eE]+)", log, re.M)
        return float(m.group(1)) if m else float("nan")

    return _get("noise_model_level"), _get("measurement_noise_level")


def _frac(pred, null, sloppy):
    """null / sloppy counts over the finite-prediction subset, matching the
    Fig. 12 legend convention (is_sloppy excludes is_null)."""
    m = np.isfinite(pred)
    n = int(m.sum())
    n_null = int((m & null).sum())
    n_sloppy = int((m & sloppy & ~null).sum())
    return dict(n=n, n_null=n_null, n_sloppy=n_sloppy)


def compute_one(root: Path, dt: float, device: torch.device):
    sigma, gamma = read_noise_levels(root)
    data = load_data(root, dt)
    print(f"  {root.name}: sigma={sigma} gamma={gamma}  T={data['T']} N={data['N']} E={data['E']}")
    in_src, in_eidx, deg_in = build_in_edges(data["edge_index"], data["N"])
    out = solve(data, in_src, in_eidx, deg_in, device)

    def _f32(a):
        return np.asarray(a, dtype=np.float32)

    result = dict(
        name=root.name, sigma=sigma, gamma=gamma,
        N=data["N"], E=data["E"], dt=dt,
        frac_tau=_frac(out["tau_lstsq"], out["tau_null"], out["tau_sloppy"]),
        frac_vrest=_frac(out["vrest_lstsq"], out["vrest_null"], out["vrest_sloppy"]),
        frac_W=_frac(out["W_lstsq"], out["W_null"], out["W_sloppy"]),
        rel_eig=_f32(out["rel_eig"]),
        scatter=dict(
            tau_true=_f32(data["tau_true"]),   tau_pred=_f32(out["tau_lstsq"]),
            tau_null=out["tau_null"],          tau_sloppy=out["tau_sloppy"],
            vrest_true=_f32(data["vrest_true"]), vrest_pred=_f32(out["vrest_lstsq"]),
            vrest_null=out["vrest_null"],      vrest_sloppy=out["vrest_sloppy"],
            W_true=_f32(data["W_true"]),       W_pred=_f32(out["W_lstsq"]),
            W_null=out["W_null"],              W_sloppy=out["W_sloppy"],
            cell_type=np.asarray(data["cell_type"]),
        ),
    )
    del data, out  # drop the big (T x N) trajectory arrays before the next dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def cache_path(cache_dir: Path, root: Path):
    return cache_dir / f"{root.name}.pt"


def get_result(root: Path, cache_dir: Path, dt: float, device, force: bool):
    cp = cache_path(cache_dir, root)
    if cp.exists() and not force:
        return torch.load(cp, map_location="cpu", weights_only=False)
    res = compute_one(root, dt, device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(res, cp)
    print(f"  cached -> {cp}")
    return res


# ===========================================================================
# Figures
# ===========================================================================
def _pct(n, d):
    return (100.0 * n / d) if d else 0.0


def scatter_panel(ax, true, pred, null, sloppy, xlabel, ylabel):
    """One recovery panel, Fig. 12 conventions: red=null, orange=sloppy,
    black=identifiable, y=x dashed, R^2_ok (R^2_all) + slope annotation."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.isfinite(pred)
    is_null   = m & null
    is_sloppy = m & sloppy & ~null
    ok        = m & ~null & ~sloppy

    r2_all, _ = compute_r_squared_NSE(true[m], pred[m])
    if ok.any():
        r2_ok, slope_ok = compute_r_squared_NSE(true[ok], pred[ok])
    else:
        r2_ok, slope_ok = float("nan"), float("nan")

    n_total = int(m.sum())
    pct_null   = _pct(int(is_null.sum()), n_total)
    pct_sloppy = _pct(int(is_sloppy.sum()), n_total)

    def _fmt(pct):
        return "0%" if pct == 0 else ("<0.1%" if pct < 0.1 else f"{pct:.1f}%")

    ax.scatter(true[is_null],   pred[is_null],   s=6, alpha=0.35, color="red",
               label=f"null ({_fmt(pct_null)})", rasterized=True)
    ax.scatter(true[is_sloppy], pred[is_sloppy], s=6, alpha=0.4, color="orange",
               label=f"sloppy ({_fmt(pct_sloppy)})", rasterized=True)
    ax.scatter(true[ok],        pred[ok],        s=4, alpha=0.7, color="k",
               rasterized=True)

    lo, hi = float(true[m].min()), float(true[m].max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, alpha=0.6)

    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.set_xlim(lo - pad, hi + pad)
    y_lo, y_hi = np.percentile(pred[m], [0.5, 99.5])
    y_lo = min(y_lo, lo - pad)
    y_hi = max(y_hi, hi + pad)
    y_pad = 0.1 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.text(0.05, 0.95, f"R²: {r2_ok:.2f} ({r2_all:.2f})\nslope: {slope_ok:.2f}",
            transform=ax.transAxes, verticalalignment="top", fontsize=_ANNOT_FS)
    ax.set_xlabel(xlabel, fontsize=_AXIS_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_LABEL_FS)
    ax.legend(loc="upper right", fontsize=_LEGEND_FS, markerscale=4)


def make_twin_figure(by_sigma, sigmas, rep_fold, out_base):
    """Rows = sigma, cols = (tau, V_rest, W). Representative fold per sigma."""
    nrows = len(sigmas)
    fig, axes = plt.subplots(nrows, 3, figsize=(30, 9 * nrows),
                             constrained_layout=True, squeeze=False)
    letters = iter("abcdefghijklmnopqrstuvwxyz")

    for r, sigma in enumerate(sigmas):
        folds = by_sigma[sigma]
        rep = next((f for f in folds if f["name"].endswith(rep_fold)), folds[0])
        sc = rep["scatter"]
        sig_lbl = rf"$\sigma={sigma:g}$"
        panels = [
            (sc["tau_true"],   sc["tau_pred"],   sc["tau_null"],   sc["tau_sloppy"],
             r"true $\tau$",       f"{sig_lbl}" + "\n\n" + r"learned $\tau$"),
            (sc["vrest_true"], sc["vrest_pred"], sc["vrest_null"], sc["vrest_sloppy"],
             r"true $V_{rest}$",   r"learned $V_{rest}$"),
            (sc["W_true"],     sc["W_pred"],     sc["W_null"],     sc["W_sloppy"],
             r"true $W_{ij}$",     r"learned $W_{ij}$"),
        ]
        for c, (tr, pr, nu, sl, xl, yl) in enumerate(panels):
            ax = axes[r, c]
            scatter_panel(ax, tr, pr, nu, sl, xl, yl)
            ax.text(-0.02, 1.06, next(letters), transform=ax.transAxes,
                    fontsize=_PANEL_LBL_FS, fontweight="bold", va="bottom", ha="right")

    for ext in ("png", "pdf"):
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def make_eigenspectrum_figure(by_sigma, sigmas, out_base):
    """Phi_sigma(eps) = cumulative fraction of per-neuron directions with
    w/w_max <= eps, pooled over folds; one curve per sigma."""
    eps = np.logspace(-30, 0, 600)
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)

    annot = {}
    for sigma in sigmas:
        rel = np.concatenate([f["rel_eig"].astype(np.float64) for f in by_sigma[sigma]])
        rel_sorted = np.sort(rel)
        phi = 100.0 * np.searchsorted(rel_sorted, eps, side="right") / rel_sorted.size
        color = SIGMA_COLORS.get(round(sigma, 4), None)
        ax.plot(eps, phi, lw=2.6, color=color, label=rf"$\sigma={sigma:g}$")
        annot[sigma] = dict(
            at_null=100.0 * np.mean(rel <= NULL_EIG_TOL),
            at_sloppy=100.0 * np.mean(rel <= SLOPPY_EIG_TOL),
            n=rel_sorted.size,
        )

    for tol, name in [(NULL_EIG_TOL, "null"), (SLOPPY_EIG_TOL, "sloppy")]:
        ax.axvline(tol, color="gray", ls=":", lw=1.3)
        ax.text(tol, ax.get_ylim()[1], f" {name}\n $\\epsilon={tol:g}$",
                fontsize=13, va="top", ha="left", color="gray")

    ax.set_xscale("log")
    ax.set_xlim(1e-30, 1e0)
    ax.set_xlabel(r"relative eigenvalue tolerance $\epsilon$  ($w/w_{\max}$)", fontsize=16)
    ax.set_ylabel(r"cumulative % of directions $\leq \epsilon$   ($\Phi_\sigma(\epsilon)$)",
                  fontsize=16)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=15, loc="upper left", frameon=False)
    ax.text(0.02, 1.02, "d", transform=ax.transAxes, fontsize=22,
            fontweight="bold", va="bottom", ha="left")

    for ext in ("png", "pdf"):
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)
    return annot


# ===========================================================================
# Fractions table (mean +/- SD across folds)
# ===========================================================================
def _agg(vals):
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=0))


def write_fractions_table(by_sigma, sigmas, spectrum_annot, out_base):
    """Console + .txt + .tex: edge/neuron-level null% and sloppy% (matching
    Fig. 12) and direction-level Phi at the two tolerances, mean +/- SD over folds."""
    rows = []
    for sigma in sigmas:
        folds = by_sigma[sigma]
        rec = {"sigma": sigma, "nfolds": len(folds)}
        for key, label in [("frac_W", "W"), ("frac_tau", "tau"), ("frac_vrest", "Vrest")]:
            null_p = [_pct(f[key]["n_null"], f[key]["n"]) for f in folds]
            slop_p = [_pct(f[key]["n_sloppy"], f[key]["n"]) for f in folds]
            rec[f"{label}_null"] = _agg(null_p)
            rec[f"{label}_sloppy"] = _agg(slop_p)
        rows.append(rec)

    def fmt(mean_sd):
        return f"{mean_sd[0]:5.2f} ± {mean_sd[1]:.2f}"

    lines = []
    lines.append("Numerical eigendecomposition of the per-neuron Gram matrix "
                 "A_i^T A_i (Fig. 12 solver).")
    lines.append("Fractions = % of finite entries flagged; mean ± SD across CV folds "
                 "(cv00..cv04).\n")
    header = (f"{'sigma':>6} {'folds':>5} | "
              f"{'W null%':>14} {'W sloppy%':>14} | "
              f"{'tau null%':>14} {'Vrest null%':>14} | {'dir Phi(1e-22)%':>16} {'dir Phi(1e-12)%':>16}")
    lines.append(header)
    lines.append("-" * len(header))
    for rec in rows:
        s = rec["sigma"]
        ann = spectrum_annot.get(s, {"at_null": float("nan"), "at_sloppy": float("nan")})
        lines.append(
            f"{s:6g} {rec['nfolds']:5d} | "
            f"{fmt(rec['W_null']):>14} {fmt(rec['W_sloppy']):>14} | "
            f"{fmt(rec['tau_null']):>14} {fmt(rec['Vrest_null']):>14} | "
            f"{ann['at_null']:16.2f} {ann['at_sloppy']:16.2f}"
        )
    txt = "\n".join(lines)
    print("\n" + txt + "\n")
    (out_base.with_suffix(".txt")).write_text(txt + "\n")
    print(f"wrote {out_base.with_suffix('.txt')}")

    # LaTeX (edge-level W, the number the reviewer cites)
    tex = [
        r"% Auto-generated by fig_lstsq_param_recovery_noise_twin.py",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$\sigma$ & W null (\%) & W sloppy (\%) & W null+sloppy (\%) \\",
        r"\midrule",
    ]
    for rec in rows:
        wn, ws = rec["W_null"], rec["W_sloppy"]
        tot = (wn[0] + ws[0], (wn[1] ** 2 + ws[1] ** 2) ** 0.5)
        tex.append(
            rf"{rec['sigma']:g} & ${wn[0]:.2f}\pm{wn[1]:.2f}$ & "
            rf"${ws[0]:.2f}\pm{ws[1]:.2f}$ & ${tot[0]:.2f}\pm{tot[1]:.2f}$ \\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out_base.with_suffix(".tex")).write_text("\n".join(tex) + "\n")
    print(f"wrote {out_base.with_suffix('.tex')}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", type=Path, nargs="+", required=True,
                   help="dataset dirs (each with ode_params.pt, x_list_train/, generation_log.txt)")
    p.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "_noise_twin_cache")
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    p.add_argument("--dt", type=float, default=0.020)
    p.add_argument("--rep-fold", type=str, default="cv00",
                   help="fold name suffix used for the scatter rows")
    p.add_argument("--compute-only", action="store_true", help="compute+cache, no figures")
    p.add_argument("--assemble-only", action="store_true", help="figures from cache only")
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')})")
    print(f"cache_dir: {args.cache_dir}")

    # ---- compute / load ----
    results = []
    for root in args.roots:
        if args.assemble_only:
            cp = cache_path(args.cache_dir, root)
            if not cp.exists():
                print(f"  [assemble-only] MISSING cache for {root.name}; skipping")
                continue
            results.append(torch.load(cp, map_location="cpu", weights_only=False))
        else:
            results.append(get_result(root, args.cache_dir, args.dt, device, args.force))

    if args.compute_only:
        print(f"compute-only done: {len(results)} datasets cached.")
        return

    # ---- group by sigma ----
    by_sigma = {}
    for r in results:
        by_sigma.setdefault(round(float(r["sigma"]), 4), []).append(r)
    sigmas = sorted(by_sigma)
    print("sigmas:", {s: len(by_sigma[s]) for s in sigmas})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    twin_base = args.out_dir / "fig_lstsq_param_recovery_noise_twin"
    eig_base = args.out_dir / "fig_lstsq_eigenspectrum_noise"
    frac_base = args.out_dir / "fig_lstsq_param_recovery_noise_twin_fractions"

    make_twin_figure(by_sigma, sigmas, args.rep_fold, twin_base)
    spectrum_annot = make_eigenspectrum_figure(by_sigma, sigmas, eig_base)
    write_fractions_table(by_sigma, sigmas, spectrum_annot, frac_base)


if __name__ == "__main__":
    main()
