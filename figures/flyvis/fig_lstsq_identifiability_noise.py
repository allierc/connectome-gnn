#!/usr/bin/env python
"""Estimator-free identifiability sweep across process noise sigma in {0,0.05,0.5},
the quotable answer to meta-review concern #1 (the 6.4% / 71% "inconsistency").

Why this and not a twin of Fig. 12
----------------------------------
The claim "process noise lifts the columnar degeneracy" is a statement about
spec(A_i), NOT about any least-squares solution. So we measure the spectrum
directly -- no estimator, no bias to argue (Gauss-Markov exogeneity is moot when
there is no beta-hat). This sidesteps the endogeneity objection entirely rather
than answering it.

The four things needed to make the figure quotable (all implemented here):

  1. Axis convention. We take the singular values of A_i directly, computed by
     QR then SVD of the small R factor -- NOT the eigendecomposition of the Gram
     A_i^T A_i. Forming the Gram squares the condition number, so in float64
     nothing below lambda_k/lmax ~ 1e-16 survives; the SVD is backward-stable and
     resolves sigma_k/sigma_1 to ~1e-16, so lambda_k/lmax = (sigma_k/sigma_1)^2
     is meaningful down to ~1e-32. We index the sweep by
     lambda/lambda_max = (sigma_k/sigma_1)^2 and print sigma_k/sigma_1 =
     sqrt(lambda/lambda_max) on a twin axis, so the null (1e-22) and sloppy
     (1e-12) thresholds are both above the resolution floor and unambiguous.

  2. 6.4% and 71% on ONE curve. Phi^edge_sigma(eps) = fraction of edges with
     support on a direction of relative eigenvalue <= eps. On the sigma=0 trace,
     eps=null_tol gives the numerical 6.4%, and there is an eps at which it
     reaches Eq.(11)'s 71% -- both marked. Two tolerances, one monotone curve;
     the "inconsistency" is a definitional artifact.

  3. Columnar vs generic decomposition. Each direction's energy is split into the
     part living in the same-type sum-zero subspace (columnar copies -- the paper's
     mechanism) and the rest: Phi = Phi_col + Phi_other. If Phi_col collapses with
     sigma while Phi_other barely moves, the columnar mechanism is MEASURED, not a
     generic "noise adds a floor everywhere" effect.

  4. Fixed horizon. sigma_k ~ sqrt(T); all datasets share T (asserted at load),
     and lambda/lambda_max is T-invariant by construction (column equilibration +
     relative eigenvalue).

Bonus -- unbiased recovery scatter. Rather than only caveating the tau panel of
Fig. 12 (endogenous dv/dt column, attenuation bias), we recover via the INCREMENT
regression, whose regressors are all time-t and share no noise with the residual:
    Delta v_i(t) = alpha_i(-v_i+I_i) + beta_i + sum_j gamma_ij ReLU(v_j) + noise
    tau=dt/alpha, V_rest=beta/alpha, W=gamma/alpha.
Endogeneity removed => unbiased tau, V_rest, W. Same red/orange/black degeneracy
coloring (from the A-spectrum) as Fig. 12.

Self-contained (own loader + solver). Imports only connectome_gnn.metrics.

Outputs (--out-dir, default: this script's directory)
  * fig_lstsq_eigenspectrum_noise.{pdf,png}        (requirements 1-3)
  * fig_lstsq_param_recovery_noise_twin_incr.{pdf,png}  (unbiased twin scatter)
  * fig_lstsq_identifiability_noise_fractions.{txt,tex}

Usage
  CUDA_VISIBLE_DEVICES=0 python fig_lstsq_identifiability_noise.py --compute-only --roots <half A> &
  CUDA_VISIBLE_DEVICES=1 python fig_lstsq_identifiability_noise.py --compute-only --roots <half B> &
  wait
  python fig_lstsq_identifiability_noise.py --assemble-only --roots <all>
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
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from connectome_gnn.metrics import compute_r_squared_NSE

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

NULL_EIG_TOL   = 1e-22   # lambda/lambda_max ; == (sigma/sigma_1)^2 = 1e-11
SLOPPY_EIG_TOL = 1e-12   # lambda/lambda_max ; == (sigma/sigma_1)^2 = 1e-6
NULL_COMP_TOL  = 1e-3
EQ11_FRACTION  = 308160 / 434112   # 0.7099 -- Eq. (11) exact-null edge fraction
# Independent anchor: Supp. Tab. 5 same-type inputs have correlation r~0.94; a
# time-shifted pair with correlation r has relative Gram eigenvalue
# lambda/lambda_max = (1-r)/(1+r). r=0.94 -> 0.031. Set with NO reference to the
# spectral sweep, so marking it on rho(eps) is not a chosen threshold.
ANCHOR_EPS     = 0.031             # lambda/lambda_max ; sigma_k/sigma_1 = sqrt = 0.176
ANCHOR_R       = 0.94

SIGMA_COLORS = {0.0: "#000000", 0.05: "#1f77b4", 0.5: "#d62728"}


# ===========================================================================
# Data loading
# ===========================================================================
def _load_cell_type_codes(data_root: Path, N: int):
    """Integer per-neuron cell-type code (for same-type grouping)."""
    nt_path = data_root / "x_list_train" / "neuron_type.zarr"
    if not nt_path.exists():
        return np.zeros(N, dtype=np.int64), 1
    nt = np.asarray(zarr.open(str(nt_path), mode="r")).astype(np.int64)
    return nt, int(nt.max()) + 1


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

    dv = (voltage[1:] - voltage[:-1]) / dt       # Delta v / dt
    relu_v = np.maximum(voltage[:-1], 0.0)
    rhs = stimulus[:-1] - voltage[:-1]           # I - v  (== -v + I)

    type_code, n_types = _load_cell_type_codes(data_root, N)
    return dict(edge_index=edge_index, W_true=W_true, tau_true=tau_true,
                vrest_true=vrest_true, dv=dv, relu_v=relu_v, rhs=rhs,
                T=T, N=N, E=E, type_code=type_code, n_types=n_types)


def build_in_edges(edge_index, N):
    src, dst = edge_index[0], edge_index[1]
    order = np.argsort(dst)
    src_sorted, dst_sorted = src[order], dst[order]
    boundaries = np.searchsorted(dst_sorted, np.arange(N + 1))
    in_src = [src_sorted[boundaries[i]:boundaries[i + 1]] for i in range(N)]
    in_eidx = [order[boundaries[i]:boundaries[i + 1]] for i in range(N)]
    deg_in = np.array([len(s) for s in in_src])
    return in_src, in_eidx, deg_in


# ===========================================================================
# Enhanced per-neuron solve
#   A = [dv/dt, -1, -ReLU(v_j)]   -> spectrum + degeneracy (matches Fig. 12)
#   X = [-v+I ,  1,  ReLU(v_j)]   -> unbiased INCREMENT recovery of theta
# ===========================================================================
def solve_enhanced(data, in_src, in_eidx, deg_in, device, dt,
                   null_eig_tol=NULL_EIG_TOL, sloppy_eig_tol=SLOPPY_EIG_TOL,
                   null_comp_tol=NULL_COMP_TOL):
    N, E, T = data["N"], data["E"], data["T"]
    n_types = data["n_types"]
    active_idx = np.where(deg_in > 0)[0]

    dv_d     = torch.from_numpy(data["dv"]).float().to(device)
    relu_v_d = torch.from_numpy(data["relu_v"]).float().to(device)
    rhs_d    = torch.from_numpy(data["rhs"]).float().to(device)
    type_d   = torch.from_numpy(data["type_code"]).to(device)

    K_max = int(deg_in.max())
    A_buf = torch.empty(T - 1, K_max + 2, device=device, dtype=torch.float64)
    A_buf[:, 1:2] = -1.0
    X_buf = torch.empty(T - 1, K_max + 2, device=device, dtype=torch.float64)
    X_buf[:, 1:2] = 1.0

    # spectrum (pooled over directions): relative eigenvalue + columnar energy frac
    n_dirs = int((deg_in[active_idx] + 2).sum())
    rel_buf  = torch.empty(n_dirs, device=device, dtype=torch.float64)
    fcol_buf = torch.empty(n_dirs, device=device, dtype=torch.float64)
    off = 0

    # per-edge / per-neuron onset tolerance eps_alpha = min rel over supporting dirs
    W_onset     = np.full(E, np.inf, dtype=np.float64)
    tau_onset   = np.full(N, np.inf, dtype=np.float64)
    vrest_onset = np.full(N, np.inf, dtype=np.float64)
    no_in = deg_in == 0
    tau_onset[no_in] = 0.0
    vrest_onset[no_in] = 0.0

    # unbiased increment recovery
    tau_inc   = np.full(N, np.nan, dtype=np.float64)
    vrest_inc = np.full(N, np.nan, dtype=np.float64)
    W_inc     = np.full(E, np.nan, dtype=np.float64)

    t0 = time.time()
    for i in tqdm(active_idx, desc="solve", unit="neuron"):
        K = len(in_src[i])
        cols = in_src[i]
        eidx = in_eidx[i]

        # -------- A : spectrum & degeneracy --------
        A = A_buf[:, :K + 2]
        A[:, 0:1].copy_(dv_d[:, i:i + 1])
        A[:, 2:].copy_(relu_v_d[:, cols]); A[:, 2:].neg_()
        sA = A.norm(dim=0); sA = torch.where(sA > 0, sA, torch.ones_like(sA))
        As = A / sA
        # Right singular vectors via QR, NOT the Gram A^T A: forming the Gram
        # squares the condition number, so in float64 nothing below
        # lambda_k/lmax ~ 1e-16 survives. QR is backward-stable and does not
        # square it; svd of the small R then resolves sigma_k/sigma_1 to ~1e-16,
        # so lambda_k/lmax = (sigma_k/sigma_1)^2 is meaningful to ~1e-32.
        R = torch.linalg.qr(As, mode="r").R                # (K+2, K+2)
        _, Sv, Vh = torch.linalg.svd(R)                    # Sv descending
        relA = (Sv / Sv[0]).clamp_min(0) ** 2              # lambda_k/lmax
        VA = Vh.transpose(-2, -1)                          # right sing. vecs as cols

        # right singular vectors in theta-space, unit-normalized columns
        Vt = VA / sA.unsqueeze(1)
        Vt = Vt / Vt.norm(dim=0, keepdim=True).clamp_min(1e-300)
        Wrows = Vt[2:, :]                                   # (K, K+2)

        # columnar energy: project W-rows onto same-type sum-zero subspace
        codes = type_d[cols]                               # (K,)
        gsum = torch.zeros(n_types, K + 2, device=device, dtype=torch.float64)
        gsum.index_add_(0, codes, Wrows)
        gcnt = torch.bincount(codes, minlength=n_types).clamp_min(1).unsqueeze(1)
        proj = Wrows - (gsum / gcnt)[codes]                # sum-zero within each type
        fcol = proj.pow(2).sum(dim=0).sqrt().clamp(0, 1)   # per direction (Vt cols unit)

        rel_buf[off:off + K + 2]  = relA
        fcol_buf[off:off + K + 2] = fcol
        off += K + 2

        # onset tolerance per parameter (min rel over directions that support it)
        big = torch.where(Vt.abs() > null_comp_tol,
                          relA.unsqueeze(0).expand_as(Vt),
                          torch.full_like(Vt, float("inf")))
        onset = big.min(dim=1).values                      # (K+2,)
        tau_onset[i]   = float(onset[0])
        vrest_onset[i] = float(onset[1])
        W_onset[eidx]  = onset[2:].cpu().numpy()

        # -------- X : unbiased increment recovery --------
        X = X_buf[:, :K + 2]
        X[:, 0:1].copy_(rhs_d[:, i:i + 1])
        X[:, 2:].copy_(relu_v_d[:, cols])
        yv = (dv_d[:, i] * dt).double()
        sX = X.norm(dim=0); sX = torch.where(sX > 0, sX, torch.ones_like(sX))
        Xs = X / sX
        wX, VX = torch.linalg.eigh(Xs.T @ Xs)
        relX = wX / wX[-1]
        keepX = relX > sloppy_eig_tol                      # min-norm: drop null+sloppy
        inv = torch.where(keepX, 1.0 / wX, torch.zeros_like(wX))
        c = Xs.T @ yv
        thX = (VX @ (inv * (VX.T @ c))) / sX
        alpha = thX[0]
        if torch.abs(alpha) > 1e-12:
            tau_inc[i]   = float(dt / alpha)
            vrest_inc[i] = float(thX[1] / alpha)
            W_inc[eidx]  = (thX[2:] / alpha).cpu().numpy()

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"solve time: {time.time() - t0:.1f}s")

    return dict(
        rel_eig=rel_buf[:off].cpu().numpy(),
        frac_col=fcol_buf[:off].cpu().numpy(),
        W_onset=W_onset, tau_onset=tau_onset, vrest_onset=vrest_onset,
        tau_inc=tau_inc, vrest_inc=vrest_inc, W_inc=W_inc,
    )


# ===========================================================================
# Per-dataset compute (cached)
# ===========================================================================
def read_noise_levels(root: Path):
    log = (root / "generation_log.txt").read_text()
    def g(k):
        m = re.search(rf"^{re.escape(k)}:\s*([-\d.eE]+)", log, re.M)
        return float(m.group(1)) if m else float("nan")
    return g("noise_model_level"), g("measurement_noise_level")


def compute_one(root: Path, dt: float, device):
    sigma, gamma = read_noise_levels(root)
    data = load_data(root, dt)
    print(f"  {root.name}: sigma={sigma} gamma={gamma}  T={data['T']} N={data['N']} E={data['E']}")
    in_src, in_eidx, deg_in = build_in_edges(data["edge_index"], data["N"])
    out = solve_enhanced(data, in_src, in_eidx, deg_in, device, dt)
    f32 = lambda a: np.asarray(a, dtype=np.float32)

    # degeneracy fractions (edge/neuron level, matching Fig. 12) from onset tolerances
    def edge_fr(onset):
        return dict(n=int(onset.size),
                    n_null=int((onset <= NULL_EIG_TOL).sum()),
                    n_sloppy=int(((onset > NULL_EIG_TOL) & (onset <= SLOPPY_EIG_TOL)).sum()))

    res = dict(
        name=root.name, sigma=sigma, gamma=gamma, N=data["N"], E=data["E"], T=data["T"], dt=dt,
        rel_eig=f32(out["rel_eig"]), frac_col=f32(out["frac_col"]),
        W_onset=f32(out["W_onset"]), tau_onset=f32(out["tau_onset"]),
        vrest_onset=f32(out["vrest_onset"]),
        frac_W=edge_fr(out["W_onset"]), frac_tau=edge_fr(out["tau_onset"]),
        frac_vrest=edge_fr(out["vrest_onset"]),
        scatter=dict(
            tau_true=f32(data["tau_true"]),   tau_pred=f32(out["tau_inc"]),   tau_onset=f32(out["tau_onset"]),
            vrest_true=f32(data["vrest_true"]), vrest_pred=f32(out["vrest_inc"]), vrest_onset=f32(out["vrest_onset"]),
            W_true=f32(data["W_true"]),       W_pred=f32(out["W_inc"]),       W_onset=f32(out["W_onset"]),
        ),
    )
    del data, out
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def cache_path(cache_dir, root):
    return cache_dir / f"{root.name}.pt"


def get_result(root, cache_dir, dt, device, force):
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
def _cum_fraction(values, eps_grid, weights=None):
    """fraction (or weighted fraction) of `values` <= each eps, normalized by
    the total count (so weighted variants sum to the unweighted total)."""
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(v)
    v_sorted = v[order]
    if weights is None:
        cum = np.searchsorted(v_sorted, eps_grid, side="right")
        return 100.0 * cum / v.size
    w_sorted = np.asarray(weights, dtype=np.float64)[order]
    w_cum = np.concatenate([[0.0], np.cumsum(w_sorted)])
    idx = np.searchsorted(v_sorted, eps_grid, side="right")
    return 100.0 * w_cum[idx] / v.size


def make_eigenspectrum_figure(by_sigma, sigmas, out_base):
    eps = np.logspace(-30, 0, 600)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    # ---- Panel a: edge-level Phi_W(eps), 6.4% and 71% on the sigma=0 curve ----
    edge_curves = {}
    for s in sigmas:
        onset = np.concatenate([f["W_onset"].astype(np.float64) for f in by_sigma[s]])
        phi = _cum_fraction(onset, eps)                       # % of edges
        edge_curves[s] = (onset, phi)
        axL.plot(eps, phi, lw=2.8, color=SIGMA_COLORS.get(round(s, 4)),
                 label=rf"$\sigma={s:g}$")

    onset0, phi0 = edge_curves[sigmas[0]]
    def phi0_at(e):  # sigma=0 edge fraction at tolerance e
        return 100.0 * np.mean(onset0 <= e)
    def eps_at_phi0(target):  # eps where sigma=0 curve hits target%
        oc = np.sort(onset0)
        k = int(np.ceil(target / 100.0 * oc.size)) - 1
        k = min(max(k, 0), oc.size - 1)
        return oc[k]

    p_null = phi0_at(NULL_EIG_TOL)
    eps_71 = eps_at_phi0(100 * EQ11_FRACTION)
    for e, txt, yoff in [
        (NULL_EIG_TOL, "numerical null\n6.4%  (Fig. 12)", 4),
        (eps_71, f"Eq. (11): {100*EQ11_FRACTION:.0f}%", -10),
    ]:
        axL.axvline(e, color="gray", ls=":", lw=1.2)
        y = phi0_at(e)
        axL.plot([e], [y], "o", color="k", ms=7, zorder=5)
        axL.annotate(txt, xy=(e, y), xytext=(e * 10, y + yoff),
                     fontsize=18, color="k",
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    axL.axvline(SLOPPY_EIG_TOL, color="gray", ls=":", lw=0.9, alpha=0.6)
    axL.set_xscale("log"); axL.set_xlim(1e-30, 1e0); axL.set_ylim(0, 100)
    axL.set_xlabel(r"tolerance $\epsilon = \lambda_k/\lambda_{\max}\;=\;(\sigma_k/\sigma_1)^2$",
                   fontsize=22)
    axL.set_ylabel(r"% of edges degenerate at $\epsilon$   ($\Phi^{\mathrm{edge}}_\sigma$)",
                   fontsize=22)
    axL.tick_params(labelsize=18)
    axL.legend(fontsize=20, loc="upper left", frameon=False, title="process noise", title_fontsize=20)
    axL.text(-0.02, 1.04, "a", transform=axL.transAxes, fontsize=30, fontweight="bold",
             va="bottom", ha="right")
    # twin top axis in sigma_k/sigma_1 = sqrt(eps)
    axT = axL.secondary_xaxis("top", functions=(lambda x: np.sqrt(x), lambda x: x**2))
    axT.set_xlabel(r"singular-value ratio  $\sigma_k/\sigma_1$", fontsize=20)
    axT.tick_params(labelsize=16)

    # ---- Panel b: rho_sigma(eps) = columnar energy share among directions <= eps.
    #      Independent anchor (Supp. Tab. 5, r~0.94) marked -- NOT a chosen threshold.
    eps_r = np.logspace(-24, -0.3, 240)
    for s in sigmas:
        rel = np.concatenate([f["rel_eig"].astype(np.float64) for f in by_sigma[s]])
        fcol = np.concatenate([f["frac_col"].astype(np.float64) for f in by_sigma[s]])
        order = np.argsort(rel); rel_s = rel[order]; w_s = (fcol[order] ** 2)
        w_cum = np.concatenate([[0.0], np.cumsum(w_s)])
        cnt = np.searchsorted(rel_s, eps_r, side="right")
        # rho is a share of the directions below eps; where too few directions
        # exist it is 0/0 (undefined), not 0. Mask the unreliable low-count tail
        # (e.g. sigma=0.5 has almost no degenerate directions) so the line breaks
        # rather than dropping to zero.
        min_cnt = max(50, int(5e-4 * rel.size))
        rho = np.divide(100.0 * w_cum[cnt], cnt,
                        out=np.full_like(eps_r, np.nan), where=cnt >= min_cnt)
        axR.plot(eps_r, rho, lw=2.6, color=SIGMA_COLORS.get(round(s, 4)),
                 label=rf"$\sigma={s:g}$")
    axR.axvline(ANCHOR_EPS, color="k", ls="--", lw=1.4)
    axR.annotate(f"Supp. Tab. 5 anchor\n$r={ANCHOR_R}\\Rightarrow\\lambda/\\lambda_{{\\max}}={ANCHOR_EPS:.3f}$",
                 xy=(ANCHOR_EPS, 40), xytext=(1.5e-5, 13),
                 fontsize=18, ha="center",
                 arrowprops=dict(arrowstyle="->", color="k", lw=1))
    axR.text(0.02, 92, "columnar\nredundancy", fontsize=17, ha="center", color="0.35")
    axR.set_xscale("log"); axR.set_xlim(1e-24, 0.5); axR.set_ylim(0, 100)
    axR.set_xlabel(r"tolerance $\epsilon = \lambda_k/\lambda_{\max}$", fontsize=22)
    axR.set_ylabel(r"columnar share of directions $\leq\epsilon$   ($\rho_\sigma$, %)", fontsize=22)
    axR.tick_params(labelsize=18)
    axR.legend(fontsize=20, loc="lower left", frameon=False, title="process noise", title_fontsize=20)
    axR.text(-0.02, 1.04, "b", transform=axR.transAxes, fontsize=30, fontweight="bold",
             va="bottom", ha="right")

    for ext in ("png", "pdf"):
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight"); print(f"wrote {p}")
    plt.close(fig)
    return {s: dict(phi_null=phi0_at(NULL_EIG_TOL) if s == sigmas[0] else None) for s in sigmas}, eps_71


def scatter_panel(ax, true, pred, onset, xlabel, ylabel):
    true = np.asarray(true, np.float64); pred = np.asarray(pred, np.float64)
    onset = np.asarray(onset, np.float64)
    m = np.isfinite(pred)
    is_null   = m & (onset <= NULL_EIG_TOL)
    is_sloppy = m & (onset > NULL_EIG_TOL) & (onset <= SLOPPY_EIG_TOL)
    ok        = m & (onset > SLOPPY_EIG_TOL)
    r2_all, _ = compute_r_squared_NSE(true[m], pred[m])
    r2_ok, slope_ok = compute_r_squared_NSE(true[ok], pred[ok]) if ok.any() else (np.nan, np.nan)
    n = int(m.sum())
    pn = 100 * int(is_null.sum()) / n if n else 0
    ps = 100 * int(is_sloppy.sum()) / n if n else 0
    fpn = "0%" if pn == 0 else ("<0.1%" if pn < 0.1 else f"{pn:.1f}%")
    fps = "0%" if ps == 0 else ("<0.1%" if ps < 0.1 else f"{ps:.1f}%")
    ax.scatter(true[is_null], pred[is_null], s=6, alpha=.35, color="red",
               label=f"null ({fpn})", rasterized=True)
    ax.scatter(true[is_sloppy], pred[is_sloppy], s=6, alpha=.4, color="orange",
               label=f"sloppy ({fps})", rasterized=True)
    ax.scatter(true[ok], pred[ok], s=4, alpha=.7, color="k", rasterized=True)
    lo, hi = float(true[m].min()), float(true[m].max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, alpha=.6)
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.set_xlim(lo - pad, hi + pad)
    ylo, yhi = np.percentile(pred[m], [0.5, 99.5])
    ylo = min(ylo, lo - pad); yhi = max(yhi, hi + pad)
    ypad = 0.1 * (yhi - ylo) if yhi > ylo else 1.0
    ax.set_ylim(ylo - ypad, yhi + ypad)
    ax.text(.05, .95, f"R²: {r2_ok:.2f} ({r2_all:.2f})\nslope: {slope_ok:.2f}",
            transform=ax.transAxes, va="top", fontsize=_ANNOT_FS)
    ax.set_xlabel(xlabel, fontsize=_AXIS_LABEL_FS); ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_FS)
    ax.tick_params(labelsize=_TICK_LABEL_FS)
    ax.legend(loc="upper right", fontsize=_LEGEND_FS, markerscale=4)


def make_twin_scatter(by_sigma, sigmas, rep_fold, out_base):
    nrows = len(sigmas)
    fig, axes = plt.subplots(nrows, 3, figsize=(30, 9 * nrows),
                             constrained_layout=True, squeeze=False)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for r, s in enumerate(sigmas):
        folds = by_sigma[s]
        rep = next((f for f in folds if f["name"].endswith(rep_fold)), folds[0])
        sc = rep["scatter"]
        sig = rf"$\sigma={s:g}$"
        cols = [
            (sc["tau_true"], sc["tau_pred"], sc["tau_onset"], r"true $\tau$",
             f"{sig}" + "\n\n" + r"learned $\tau$"),
            (sc["vrest_true"], sc["vrest_pred"], sc["vrest_onset"], r"true $V_{rest}$", r"learned $V_{rest}$"),
            (sc["W_true"], sc["W_pred"], sc["W_onset"], r"true $W_{ij}$", r"learned $W_{ij}$"),
        ]
        for c, (tr, pr, on, xl, yl) in enumerate(cols):
            scatter_panel(axes[r, c], tr, pr, on, xl, yl)
            axes[r, c].text(-0.02, 1.06, next(letters), transform=axes[r, c].transAxes,
                            fontsize=_PANEL_LBL_FS, fontweight="bold", va="bottom", ha="right")
    for ext in ("png", "pdf"):
        p = out_base.with_suffix(f".{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight"); print(f"wrote {p}")
    plt.close(fig)


# ===========================================================================
# Fractions table
# ===========================================================================
def _ms(a):
    a = np.asarray(a, float); return a.mean(), a.std()


def write_table(by_sigma, sigmas, eps_71, out_base):
    rows = []
    for s in sigmas:
        folds = by_sigma[s]
        d = {"sigma": s, "n": len(folds)}
        for key, lab in [("frac_W", "W"), ("frac_tau", "tau"), ("frac_vrest", "Vrest")]:
            d[lab + "_null"] = _ms([100 * f[key]["n_null"] / f[key]["n"] for f in folds])
            d[lab + "_slop"] = _ms([100 * f[key]["n_sloppy"] / f[key]["n"] for f in folds])
        # columnar share of the degenerate (rel<=sloppy) directions
        col_share = []
        for f in folds:
            rel = f["rel_eig"].astype(np.float64); fc = f["frac_col"].astype(np.float64)
            deg = rel <= SLOPPY_EIG_TOL
            col_share.append(100 * (fc[deg] ** 2).sum() / max(deg.sum(), 1))
        d["col_share"] = _ms(col_share)
        rows.append(d)

    fmt = lambda t: f"{t[0]:5.2f}±{t[1]:.2f}"
    L = []
    L.append("Estimator-free identifiability sweep -- edge/neuron degeneracy vs process noise sigma.")
    L.append(f"Horizon T fixed across sigma; eps in lambda/lambda_max; Eq.(11) 71% reached at eps={eps_71:.2e}.")
    L.append("Mean +/- SD across CV folds cv00..cv04.\n")
    hdr = (f"{'sigma':>6}{'n':>3} | {'W null%':>12}{'W sloppy%':>12} | "
           f"{'tau null%':>12}{'Vrest null%':>12} | {'columnar % of degenerate':>26}")
    L.append(hdr); L.append("-" * len(hdr))
    for d in rows:
        L.append(f"{d['sigma']:6g}{d['n']:3d} | {fmt(d['W_null']):>12}{fmt(d['W_slop']):>12} | "
                 f"{fmt(d['tau_null']):>12}{fmt(d['Vrest_null']):>12} | {fmt(d['col_share']):>26}")
    txt = "\n".join(L)
    print("\n" + txt + "\n")
    out_base.with_suffix(".txt").write_text(txt + "\n"); print(f"wrote {out_base.with_suffix('.txt')}")

    tex = [r"% auto-generated by fig_lstsq_identifiability_noise.py",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r"$\sigma$ & W null (\%) & W sloppy (\%) & $\tau$ null (\%) & columnar \% of deg. \\",
           r"\midrule"]
    for d in rows:
        tex.append(rf"{d['sigma']:g} & ${d['W_null'][0]:.2f}\pm{d['W_null'][1]:.2f}$ & "
                   rf"${d['W_slop'][0]:.2f}\pm{d['W_slop'][1]:.2f}$ & "
                   rf"${d['tau_null'][0]:.2f}\pm{d['tau_null'][1]:.2f}$ & "
                   rf"${d['col_share'][0]:.1f}\pm{d['col_share'][1]:.1f}$ \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    out_base.with_suffix(".tex").write_text("\n".join(tex) + "\n"); print(f"wrote {out_base.with_suffix('.tex')}")


# ===========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", type=Path, nargs="+", required=True)
    p.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "_identif_noise_cache")
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR)
    p.add_argument("--dt", type=float, default=0.020)
    p.add_argument("--rep-fold", type=str, default="cv00")
    p.add_argument("--compute-only", action="store_true")
    p.add_argument("--assemble-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')})")

    results = []
    for root in args.roots:
        if args.assemble_only:
            cp = cache_path(args.cache_dir, root)
            if not cp.exists():
                print(f"  [assemble-only] MISSING {root.name}; skip"); continue
            results.append(torch.load(cp, map_location="cpu", weights_only=False))
        else:
            results.append(get_result(root, args.cache_dir, args.dt, device, args.force))
    if args.compute_only:
        print(f"compute-only done: {len(results)} cached."); return

    Ts = {r["T"] for r in results}
    if len(Ts) > 1:
        print(f"WARNING: horizons differ across datasets: {Ts} (req #4 -- sigma_k ~ sqrt(T))")

    by_sigma = {}
    for r in results:
        by_sigma.setdefault(round(float(r["sigma"]), 4), []).append(r)
    sigmas = sorted(by_sigma)
    print("sigmas:", {s: len(by_sigma[s]) for s in sigmas}, " T:", Ts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, eps_71 = make_eigenspectrum_figure(by_sigma, sigmas, args.out_dir / "fig_lstsq_eigenspectrum_noise")
    make_twin_scatter(by_sigma, sigmas, args.rep_fold,
                      args.out_dir / "fig_lstsq_param_recovery_noise_twin_incr")
    write_table(by_sigma, sigmas, eps_71,
                args.out_dir / "fig_lstsq_identifiability_noise_fractions")


if __name__ == "__main__":
    main()
