#!/usr/bin/env python
"""Recover ODE parameters (tau, V_rest, W) per neuron via min-norm least squares.

Solves the linear system A_i @ theta_i = b_i for each neuron, where columns of
A_i are [dv_i/dt, -1, -ReLU(v_j) for j in N_i] and b_i = I_i - v_i. Uses
column-equilibrated normal equations on GPU, with the eigendecomposition of the
small Gram matrix giving both the min-norm solution and a flag for parameters
that lie in the null space (degenerate / unidentifiable).

Outputs a 3-panel scatter (tau, V_rest, W) of recovered vs. ground-truth values,
with degenerate parameters colored red, plus a `<basename>_degenerate_mask.pt`
sidecar tensor used by downstream training.

Output: figures/fig_lstsq_param_recovery_<basename>.{pdf,png}

Usage:
    python figures/fig_lstsq_param_recovery.py DATA_ROOT [--dt DT]

NOTE: noise-free data only. For noisy SDE data (sigma > 0) recovery is biased
toward zero — particularly for tau, since the dv/dt column is computed by finite
differences which amplifies voltage noise by ~1/dt. Two fundamental obstacles:

  1. Errors-in-variables bias: noise enters BOTH A and b (not just b). Standard
     OLS gives attenuation bias on every coefficient with a noisy regressor;
     this bias does not vanish with more data.
  2. Correlated noise across columns: the same SDE noise on v_i(t) appears in
     dv/dt(t), dv/dt(t-1), and b(t) simultaneously. Vanilla TLS assumes column-
     wise independent noise and produces wild outputs in this regime.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from connectome_gnn.metrics import compute_r_squared_NSE


# ---------------------------------------------------------------------------
# Style: Janne base + scatter overrides per figures/INSTRUCTIONS.md
# ---------------------------------------------------------------------------
matplotlib.rc_file(str(REPO / "figures" / "janne.matplotlibrc"))
plt.rcParams.update({
    # GNN_PlotFigure scatter convention: keep spines, use Nimbus Sans family.
    "font.family":     "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi":  150,
})

# Scatter-content fonts (figsize=(30,9), each panel 10 in wide -> _S = 1.0).
_AXIS_LABEL_FS = 48
_TICK_LABEL_FS = 24
_ANNOT_FS      = 32
_LEGEND_FS     = 28
_PANEL_LBL_FS  = 40


def _load_cell_type_labels(data_root: Path, N: int):
    """Per-neuron cell-type label (str). Best-effort: tries to resolve int ids
    in neuron_type.zarr to flyvis type names (alphabetically sorted unique
    types from the connectome). Falls back to stringified int ids."""
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
        names = np.unique(types_full)  # alphabetical — matches np.unique(...,return_inverse=True) ordering used at gen time
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
    in_src = [src_sorted[boundaries[i]:boundaries[i+1]] for i in range(N)]
    in_eidx = [order[boundaries[i]:boundaries[i+1]] for i in range(N)]
    deg_in = np.array([len(s) for s in in_src])
    return in_src, in_eidx, deg_in


def solve(
    data: dict,
    in_src: list,
    in_eidx: list,
    deg_in: np.ndarray,
    device: torch.device,
    null_eig_tol: float = 1e-22,
    sloppy_eig_tol: float = 1e-12,
    null_comp_tol: float = 1e-3,
):
    """Per-neuron min-norm lstsq + degeneracy flagging.

    Two-tier classification of directions by relative eigenvalue w/w_max of
    A_tilde^T A_tilde:
        w/w_max <= null_eig_tol     -> exact null (numerical zero, structural)
        w/w_max <= sloppy_eig_tol   -> sloppy / weakly identifiable
                                       ("sloppy" in Sethna et al. terminology)
        w/w_max >  sloppy_eig_tol   -> identifiable

    Both null and sloppy directions are zeroed in the pseudoinverse (so the
    solve doesn't amplify noise on near-null directions) and used to flag
    degenerate parameters. The flag distinguishes the two: tau_null vs
    tau_sloppy, etc., letting plots use different colors.

    null_comp_tol — min |v_alpha| on a null/sloppy direction (after
                    normalization) to flag parameter alpha.
    """
    N, E, T = data["N"], data["E"], data["T"]
    active_idx = np.where(deg_in > 0)[0]

    # By default the full (T-1, N) matrices are uploaded to the GPU once. For
    # large N (e.g. flyvis full eye, ~25 GB per matrix at float64) this OOMs on
    # a single GPU; --lowmem keeps them on (pinned) host memory and ships only
    # per-neuron column slices each iteration. Slower per iter but fits.
    # Storage: float32 (halves memory + bandwidth). Per-neuron design matrix A
    # and target b are upcast to float64 inside the loop, since the eigh /
    # pseudoinverse step is sensitive to the conditioning of A^T A.
    dv_d     = torch.from_numpy(data["dv"]).float().to(device)
    relu_v_d = torch.from_numpy(data["relu_v"]).float().to(device)
    rhs_d    = torch.from_numpy(data["rhs"]).float().to(device)
    ones_col = -torch.ones(T - 1, 1, device=device, dtype=torch.float64)

    # Preallocate one (T-1, K_max+2) double buffer reused across all iterations.
    # Column 0 = dv_i, column 1 = -1, columns 2..2+K_i-1 = -ReLU(v_j) for j in
    # in_src[i]. A is a view into A_buf trimmed to the current K_i+2 width, so
    # there's no per-iteration allocation for A and no torch.cat call.
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
    # Cramér–Rao std-error per parameter (in original parameter scale): smaller
    # = more identifiable. sigma_alpha = sqrt( sum_k v_{k,alpha}^2 / lambda_k ) / s_alpha
    # where (lambda, V) is the eigendecomposition of A_s^T A_s. Captures both
    # null directions (huge sigma) and sloppy directions (large sigma) on a
    # continuous scale. Null lambdas are floored at null_eig_tol*lambda_max so
    # the sum stays finite; the resulting sigma is then large but not inf.
    tau_score   = np.full(N, np.nan, dtype=np.float64)
    vrest_score = np.full(N, np.nan, dtype=np.float64)
    W_score     = np.full(E, np.nan, dtype=np.float64)

    # Neurons with no incoming edges are entirely unidentifiable: tau, V_rest jointly null.
    no_in = deg_in == 0
    tau_null[no_in] = True
    vrest_null[no_in] = True

    t0 = time.time()
    for i in tqdm(active_idx, desc="lstsq", unit="neuron"):
        # Fill A_buf in place: copy_ handles the float32 -> float64 cast.
        # Column 1 (-1) was set once outside the loop and is never touched.
        K_i = len(in_src[i])
        A = A_buf[:, :K_i + 2]
        A[:, 0:1].copy_(dv_d[:, i:i+1])
        A[:, 2:].copy_(relu_v_d[:, in_src[i]])
        A[:, 2:].neg_()
        b = rhs_d[:, i].double()

        s = A.norm(dim=0)
        s = torch.where(s > 0, s, torch.ones_like(s))
        A_s = A / s

        # Null/sloppy detection always uses A's own Gram (independent of solver).
        G = A_s.T @ A_s
        w, V = torch.linalg.eigh(G)
        w_max = w[-1]
        rel = w / w_max
        null_mask   = rel <= null_eig_tol
        sloppy_mask = (rel > null_eig_tol) & (rel <= sloppy_eig_tol)

        # OLS via column-equilibrated normal equations + pseudoinverse.
        c = A_s.T @ b
        keep = ~(null_mask | sloppy_mask)
        inv_w = torch.where(keep, 1.0 / w, torch.zeros_like(w))
        theta_i = (V @ (inv_w * (V.T @ c))) / s

        # Cramér–Rao std error in original parameter coordinates.
        w_floor = w.clamp_min(null_eig_tol * w_max)
        sigma_tilde = torch.sqrt((V**2 / w_floor).sum(dim=1))   # in equilibrated space
        sigma = sigma_tilde / s                                 # back to original scale
        sg = sigma.cpu().numpy()

        th = theta_i.cpu().numpy()

        tau_lstsq[i]   = th[0]
        vrest_lstsq[i] = th[1]
        W_lstsq[in_eidx[i]] = th[2:]

        tau_score[i]   = sg[0]
        vrest_score[i] = sg[1]
        W_score[in_eidx[i]] = sg[2:]

        def _flag(mask_t, tau_arr, vrest_arr, W_arr):
            if int(mask_t.sum().item()) == 0:
                return
            V_null = (V[:, mask_t] / s.unsqueeze(1)).abs()
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
    print(f"solve time: {time.time()-t0:.1f}s")
    print(f"null   : tau={tau_null.sum()}  V_rest={vrest_null.sum()}  W={W_null.sum()}/{E}")
    print(f"sloppy : tau={tau_sloppy.sum()}  V_rest={vrest_sloppy.sum()}  W={W_sloppy.sum()}/{E}")

    return dict(
        tau_lstsq=tau_lstsq, vrest_lstsq=vrest_lstsq, W_lstsq=W_lstsq,
        tau_null=tau_null,     vrest_null=vrest_null,     W_null=W_null,
        tau_sloppy=tau_sloppy, vrest_sloppy=vrest_sloppy, W_sloppy=W_sloppy,
        tau_score=tau_score,   vrest_score=vrest_score,   W_score=W_score,
    )


def _add_panel_labels(fig, axes_flat, labels, fontsize=_PANEL_LBL_FS):
    """Place labels at top-left of each panel's outer (tight) bbox, aligned to a
    shared y. Per figures/INSTRUCTIONS.md."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bboxes = [ax.get_tightbbox(renderer) for ax in axes_flat]
    y1_max = max(inv.transform((bb.x0, bb.y1))[1] for bb in bboxes)
    for bb, lbl in zip(bboxes, labels):
        x0 = inv.transform((bb.x0, bb.y1))[0]
        fig.text(x0, y1_max, lbl, fontsize=fontsize, fontweight="bold",
                 va="bottom", ha="left", color="black", transform=fig.transFigure)


def plot(data: dict, out: dict, out_base: Path):
    fig, axes = plt.subplots(1, 3, figsize=(30, 9), constrained_layout=True)

    def _panel(ax, true, pred, null, sloppy, xlabel, ylabel, labels=None, label_x_min=None):
        m = np.isfinite(pred)   # excludes nan and inf
        is_null   = m & null
        is_sloppy = m & sloppy & ~null
        ok        = m & ~null & ~sloppy

        r2_all, _ = compute_r_squared_NSE(true[m], pred[m])
        if ok.any():
            r2_ok, slope_ok = compute_r_squared_NSE(true[ok], pred[ok])
        else:
            r2_ok, slope_ok = float('nan'), float('nan')

        n_total = int(m.sum())
        n_null   = int(is_null.sum())
        n_sloppy = int(is_sloppy.sum())
        pct_null   = (100.0 * n_null   / n_total) if n_total else 0.0
        pct_sloppy = (100.0 * n_sloppy / n_total) if n_total else 0.0

        n_ok = int(ok.sum())
        pct_ok = (100.0 * n_ok / n_total) if n_total else 0.0

        def _fmt(n, pct):
            if n == 0:
                return "0%"
            return f"<0.1%" if pct < 0.1 else f"{pct:.1f}%"

        # Layered: null (red) at the back, sloppy (orange) above, ok (black) on top.
        # rasterized=True keeps PDF size small and lets Overleaf render quickly
        # (otherwise tens of thousands of vector points choke the renderer).
        ax.scatter(true[is_null],   pred[is_null],   s=6, alpha=0.35, color="red",
                   label=f"null ({_fmt(n_null, pct_null)})", rasterized=True)
        ax.scatter(true[is_sloppy], pred[is_sloppy], s=6, alpha=0.4, color="orange",
                   label=f"sloppy ({_fmt(n_sloppy, pct_sloppy)})", rasterized=True)
        ax.scatter(true[ok],        pred[ok],        s=4, alpha=0.7, color="k",
                   rasterized=True)
        lo, hi = float(true[m].min()), float(true[m].max())
        ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1, alpha=0.6)

        # Robust axis limits: x from `true` range, y from percentiles of pred.
        # Prevents extreme outliers (e.g. 1e30 from unstable TLS) from breaking
        # matplotlib's tick computation while still letting them appear at the
        # plot edges.
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.set_xlim(lo - pad, hi + pad)
        if m.any():
            y_lo, y_hi = np.percentile(pred[m], [0.5, 99.5])
            y_lo = min(y_lo, lo - pad)
            y_hi = max(y_hi, hi + pad)
            y_pad = 0.1 * (y_hi - y_lo) if y_hi > y_lo else 1.0
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        # One label per cell type at the centroid of that type's null|sloppy
        # subset. Overlapping labels get nudged apart by a simple iterative
        # repulsion in axis-fraction coordinates so each tag stays readable.
        if labels is not None:
            labels_arr = np.asarray(labels)
            label_mask = is_null | is_sloppy
            true_arr = np.asarray(true)
            pred_arr = np.asarray(pred)
            anchors = []
            # Only label a type if its null/sloppy centroid sits noticeably off
            # the y=x line — otherwise the recovery is fine and labeling is
            # noise. Threshold = 5% of the y-axis span.
            ylo_t, yhi_t = ax.get_ylim()
            off_thresh = 0.05 * (yhi_t - ylo_t)
            for _t in np.unique(labels_arr[label_mask]):
                if not _t:
                    continue
                _m = label_mask & (labels_arr == _t)
                if not _m.any():
                    continue
                _x = float(true_arr[_m].mean())
                _y = float(pred_arr[_m].mean())
                if abs(_y - _x) < off_thresh:
                    continue
                if label_x_min is not None and _x < label_x_min:
                    continue
                anchors.append((_x, _y, str(_t)))
            if anchors:
                xlo, xhi = ax.get_xlim()
                ylo, yhi = ax.get_ylim()
                xr = xhi - xlo or 1.0
                yr = yhi - ylo or 1.0
                # Work in normalized coords so x/y radii are comparable.
                pts = np.array([[(a[0]-xlo)/xr, (a[1]-ylo)/yr] for a in anchors])
                pos = pts.copy()
                min_d = 0.05  # ~5% of axis span between label centers
                for _ in range(80):
                    moved = False
                    for i in range(len(pos)):
                        for j in range(i+1, len(pos)):
                            d = pos[j] - pos[i]
                            n = float(np.hypot(*d))
                            if n < min_d:
                                if n < 1e-9:
                                    d = np.array([1e-3, 1e-3])
                                    n = float(np.hypot(*d))
                                push = (min_d - n) / 2 * d / n
                                pos[i] -= push
                                pos[j] += push
                                moved = True
                    if not moved:
                        break
                for (ax_, ay_, txt), (px, py) in zip(anchors, pos):
                    lx = xlo + px * xr
                    ly = ylo + py * yr
                    if (lx, ly) != (ax_, ay_):
                        ax.plot([ax_, lx], [ay_, ly], color='gray',
                                linewidth=0.4, alpha=0.5, zorder=2)
                    ax.text(lx, ly, txt,
                            fontsize=11, ha='center', va='center',
                            color='black', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                      edgecolor='gray', alpha=0.75, linewidth=0.5),
                            zorder=3)

        ax.text(0.05, 0.95,
                f'R²: {r2_ok:.2f} ({r2_all:.2f})\nslope: {slope_ok:.2f}',
                transform=ax.transAxes, verticalalignment='top', fontsize=_ANNOT_FS)
        ax.set_xlabel(xlabel, fontsize=_AXIS_LABEL_FS)
        ax.set_ylabel(ylabel, fontsize=_AXIS_LABEL_FS)
        ax.tick_params(axis='both', labelsize=_TICK_LABEL_FS)
        # INSTRUCTIONS: legend top-right inside the data area.
        ax.legend(loc='upper right', fontsize=_LEGEND_FS, markerscale=4)

    cell_type = data.get("cell_type")

    _panel(axes[0], data["tau_true"],   out["tau_lstsq"],
           out["tau_null"],   out["tau_sloppy"],
           r'true $\tau$',      r'learned $\tau$',
           labels=cell_type, label_x_min=0.05)
    _panel(axes[1], data["vrest_true"], out["vrest_lstsq"],
           out["vrest_null"], out["vrest_sloppy"],
           r'true $V_{rest}$',  r'learned $V_{rest}$',
           labels=cell_type, label_x_min=0.05)
    _panel(axes[2], data["W_true"],     out["W_lstsq"],
           out["W_null"],     out["W_sloppy"],
           r'true $W_{ij}$',    r'learned $W_{ij}$')

    _add_panel_labels(fig, list(axes), ['A', 'B', 'C'])

    out_png = out_base.with_suffix('.png')
    out_pdf = out_base.with_suffix('.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png.name}, {out_pdf.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_root", type=Path, help="dir containing ode_params.pt and x_list_train/")
    p.add_argument("--out", type=Path, default=None,
                   help="output figure stem (default: figures/fig_lstsq_param_recovery_<basename>)")
    p.add_argument("--mask-out", type=Path, default=None,
                   help="output path for degeneracy mask .pt (default: alongside figure)")
    p.add_argument("--dt", type=float, default=0.020, help="simulation timestep in seconds")
    p.add_argument("--null-eig-tol", type=float, default=1e-22,
                   help="relative eigenvalue cutoff for STRICT null space (red)")
    p.add_argument("--sloppy-eig-tol", type=float, default=1e-12,
                   help="relative eigenvalue cutoff for sloppy directions (orange); "
                        "weakly identifiable, also zeroed in pseudoinverse")
    p.add_argument("--null-comp-tol", type=float, default=1e-3,
                   help="min |v_alpha| on a null/sloppy direction to flag parameter alpha")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = p.parse_args()

    out_base = args.out or REPO / "figures" / f"fig_lstsq_param_recovery_{args.data_root.name}"
    mask_path = args.mask_out or out_base.with_name(
        f"{args.data_root.name}_degenerate_mask.pt"
    )
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}  data_root: {args.data_root}")

    data = load_data(args.data_root, args.dt)
    print(f"T={data['T']}  N={data['N']}  E={data['E']}")
    in_src, in_eidx, deg_in = build_in_edges(data["edge_index"], data["N"])
    out = solve(data, in_src, in_eidx, deg_in, device,
                null_eig_tol=args.null_eig_tol,
                sloppy_eig_tol=args.sloppy_eig_tol,
                null_comp_tol=args.null_comp_tol)
    plot(data, out, out_base)

    torch.save({
        "tau":           torch.from_numpy(out["tau_null"]   | out["tau_sloppy"]),
        "V_rest":        torch.from_numpy(out["vrest_null"] | out["vrest_sloppy"]),
        "W":             torch.from_numpy(out["W_null"]     | out["W_sloppy"]),
        "tau_null":      torch.from_numpy(out["tau_null"]),
        "tau_sloppy":    torch.from_numpy(out["tau_sloppy"]),
        "V_rest_null":   torch.from_numpy(out["vrest_null"]),
        "V_rest_sloppy": torch.from_numpy(out["vrest_sloppy"]),
        "W_null":        torch.from_numpy(out["W_null"]),
        "W_sloppy":      torch.from_numpy(out["W_sloppy"]),
        # Continuous Cramér–Rao std-error per parameter (smaller = more identifiable).
        "tau_score":     torch.from_numpy(out["tau_score"]),
        "V_rest_score":  torch.from_numpy(out["vrest_score"]),
        "W_score":       torch.from_numpy(out["W_score"]),
    }, mask_path)
    print(f"wrote {mask_path}")


if __name__ == "__main__":
    main()
