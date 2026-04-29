#!/usr/bin/env python
"""Recover ODE parameters (tau, V_rest, W) per neuron via min-norm least squares.

Solves the linear system A_i @ theta_i = b_i for each neuron, where columns of
A_i are [dv_i/dt, -1, -ReLU(v_j) for j in N_i] and b_i = I_i - v_i. Uses
column-equilibrated normal equations on GPU, with the eigendecomposition of the
small Gram matrix giving both the min-norm solution and a flag for parameters
that lie in the null space (degenerate / unidentifiable).

Outputs a 3-panel scatter plot (tau, V_rest, W) of recovered vs. ground-truth
values, with degenerate parameters colored red.

Usage:
    python recover_ode_params_lstsq.py DATA_ROOT [--out OUT_PATH] [--dt DT]
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from tqdm.auto import tqdm


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

    return dict(
        edge_index=edge_index,
        W_true=W_true, tau_true=tau_true, vrest_true=vrest_true,
        dv=dv, relu_v=relu_v, rhs=rhs,
        T=T, N=N, E=E,
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
    neq_tol: float = 1e-12,
    null_comp_tol: float = 1e-3,
):
    N, E, T = data["N"], data["E"], data["T"]
    active_idx = np.where(deg_in > 0)[0]

    dv_d     = torch.from_numpy(data["dv"]).to(device).double()
    relu_v_d = torch.from_numpy(data["relu_v"]).to(device).double()
    rhs_d    = torch.from_numpy(data["rhs"]).to(device).double()
    ones_col = -torch.ones(T - 1, 1, device=device, dtype=torch.float64)

    tau_lstsq   = np.full(N, np.nan, dtype=np.float64)
    vrest_lstsq = np.full(N, np.nan, dtype=np.float64)
    W_lstsq     = np.full(E, np.nan, dtype=np.float64)
    tau_deg   = np.zeros(N, dtype=bool)
    vrest_deg = np.zeros(N, dtype=bool)
    W_deg     = np.zeros(E, dtype=bool)

    t0 = time.time()
    for i in tqdm(active_idx, desc="lstsq", unit="neuron"):
        A = torch.cat([
            dv_d[:, i:i+1],
            ones_col,
            -relu_v_d[:, in_src[i]],
        ], dim=1)
        b = rhs_d[:, i]

        s = A.norm(dim=0)
        s = torch.where(s > 0, s, torch.ones_like(s))
        A_s = A / s

        G = A_s.T @ A_s
        c = A_s.T @ b

        w, V = torch.linalg.eigh(G)
        w_max = w[-1]
        keep = w > neq_tol * w_max
        inv_w = torch.where(keep, 1.0 / w, torch.zeros_like(w))
        theta_i = (V @ (inv_w * (V.T @ c))) / s
        th = theta_i.cpu().numpy()

        tau_lstsq[i]   = th[0]
        vrest_lstsq[i] = th[1]
        W_lstsq[in_eidx[i]] = th[2:]

        null_mask = ~keep
        if int(null_mask.sum().item()) > 0:
            null_V = (V[:, null_mask] / s.unsqueeze(1)).abs()
            null_V = null_V / null_V.amax(dim=0, keepdim=True).clamp_min(1e-300)
            part = null_V.amax(dim=1).cpu().numpy()
            if part[0] > null_comp_tol:
                tau_deg[i] = True
            if part[1] > null_comp_tol:
                vrest_deg[i] = True
            edge_deg = part[2:] > null_comp_tol
            if edge_deg.any():
                W_deg[in_eidx[i][edge_deg]] = True

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"solve time: {time.time()-t0:.1f}s")
    print(f"degenerate: tau={tau_deg.sum()}  V_rest={vrest_deg.sum()}  W={W_deg.sum()}/{E}")

    return dict(
        tau_lstsq=tau_lstsq, vrest_lstsq=vrest_lstsq, W_lstsq=W_lstsq,
        tau_deg=tau_deg, vrest_deg=vrest_deg, W_deg=W_deg,
    )


def plot(data: dict, out: dict, fig_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    def _panel(ax, true, pred, deg, title):
        m = ~np.isnan(pred)
        ok = m & ~deg
        bad = m & deg
        ax.scatter(true[ok], pred[ok], s=3, alpha=0.4, color="k", label=f"ok ({ok.sum()})")
        ax.scatter(true[bad], pred[bad], s=6, alpha=0.6, color="r", label=f"degenerate ({bad.sum()})")
        lo, hi = true[m].min(), true[m].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel(f"{title} true"); ax.set_ylabel(f"{title} lstsq")
        ax.set_title(title); ax.legend(loc="best", fontsize=8)

    _panel(axes[0], data["tau_true"],   out["tau_lstsq"],   out["tau_deg"],   "tau")
    _panel(axes[1], data["vrest_true"], out["vrest_lstsq"], out["vrest_deg"], "V_rest")
    _panel(axes[2], data["W_true"],     out["W_lstsq"],     out["W_deg"],     "W")

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"wrote {fig_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_root", type=Path, help="dir containing ode_params.pt and x_list_train/")
    p.add_argument("--out", type=Path, default=None, help="output figure path (default: data_root/lstsq_recovery.png)")
    p.add_argument("--dt", type=float, default=0.020, help="simulation timestep in seconds")
    p.add_argument("--neq-tol", type=float, default=1e-12)
    p.add_argument("--null-comp-tol", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    args = p.parse_args()

    fig_path = args.out or (args.data_root / "lstsq_recovery.png")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}  data_root: {args.data_root}")

    data = load_data(args.data_root, args.dt)
    print(f"T={data['T']}  N={data['N']}  E={data['E']}")
    in_src, in_eidx, deg_in = build_in_edges(data["edge_index"], data["N"])
    out = solve(data, in_src, in_eidx, deg_in, device,
                neq_tol=args.neq_tol, null_comp_tol=args.null_comp_tol)
    plot(data, out, fig_path)


if __name__ == "__main__":
    main()
