"""Is the sigma=0.05 blow-up a property of the data or of the truncation?

Re-solves the increment system of fig_lstsq_identifiability_noise.py two ways --
the original Gram path (eigh of Xs^T Xs, which squares cond(X)) and a QR+SVD path
that never forms the Gram -- across a sweep of truncation tolerances, at each
process-noise level.

If the exploding edges at sigma=0.05 are physics, they survive both solvers at
every tolerance. If they are a truncation artifact, they move with the cut and
shrink when the condition number is not squared.

Each neuron is decomposed once (one eigh, one QR+SVD) and every tolerance is
evaluated from that decomposition, so the sweep costs no more than two solves.

    PYTHONPATH=src python neurips_review/lstsq_tol_sweep.py --sigma 0.05
"""

import os
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures" / "flyvis"))
from fig_lstsq_identifiability_noise import build_in_edges, load_data  # noqa: E402

from connectome_gnn.metrics import compute_r_squared_NSE  # noqa: E402

ROOT = Path(f"{os.environ['GNN_OUTPUT_ROOT']}/graphs_data/fly")
RUNS = {"0": "flyvis_noise_free_blank50_cv04",
        "0.05": "flyvis_noise_005_blank50_cv04",
        "0.5": "flyvis_noise_05_blank50_cv04"}
TOLS = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]
BLOWUP = 5.0     # |W_hat - W| above this counts as an exploded edge
ALPHA_GUARD = 1e-12


def solve_all_tols(data, in_src, in_eidx, deg_in, device, dt, tols):
    """Return {(mode, tol): W_hat}. One eigh + one QR/SVD per neuron."""
    dv = torch.from_numpy(data["dv"]).float().to(device)
    relu_v = torch.from_numpy(data["relu_v"]).float().to(device)
    rhs = torch.from_numpy(data["rhs"]).float().to(device)
    E = data["E"]
    out = {(m, t): np.zeros(E, dtype=np.float64)
           for m in ("gram", "svd") for t in tols}

    for i in np.where(deg_in > 0)[0]:
        cols, eidx = in_src[i], in_eidx[i]
        K = len(cols)
        X = torch.empty((dv.shape[0], K + 2), dtype=torch.float64, device=device)
        X[:, 0:1] = rhs[:, i:i + 1].double()
        X[:, 1] = 1.0
        X[:, 2:] = relu_v[:, cols].double()
        y = (dv[:, i] * dt).double()

        # Column equilibration, exactly as the original: zero-norm columns -> 1.
        sX = X.norm(dim=0)
        sX = torch.where(sX > 0, sX, torch.ones_like(sX))
        Xs = X / sX
        Xty = Xs.T @ y

        # --- Gram path (original) ---
        w, V = torch.linalg.eigh(Xs.T @ Xs)
        rel_g = w / w[-1]
        Vtc = V.T @ Xty

        # --- QR + SVD path (cond(X) never squared) ---
        # Q is needed explicitly: recovering Q^T y as R^-T (Xs^T y) would invert
        # R, which is unstable exactly when R is ill-conditioned -- the case of
        # interest. mode='reduced' costs one extra (T x n) buffer.
        Q, R = torch.linalg.qr(Xs, mode="reduced")
        U, S, Vh = torch.linalg.svd(R)
        rel_s = (S / S[0]).clamp_min(0) ** 2          # lambda_k / lambda_max
        Utc = U.T @ (Q.T @ y)

        for tol in tols:
            inv = torch.where(rel_g > tol, 1.0 / w, torch.zeros_like(w))
            th = (V @ (inv * Vtc)) / sX
            if torch.abs(th[0]) > ALPHA_GUARD:
                out[("gram", tol)][eidx] = (th[2:] / th[0]).cpu().numpy()

            sinv = torch.where(rel_s > tol, 1.0 / S, torch.zeros_like(S))
            th = (Vh.T @ (sinv * Utc)) / sX
            if torch.abs(th[0]) > ALPHA_GUARD:
                out[("svd", tol)][eidx] = (th[2:] / th[0]).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", nargs="+", default=list(RUNS))
    ap.add_argument("--dt", type=float, default=0.020)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")

    for sig in a.sigma:
        data = load_data(ROOT / RUNS[sig], a.dt)
        in_src, in_eidx, deg_in = build_in_edges(data["edge_index"], data["N"])
        Wt = data["W_true"]
        res = solve_all_tols(data, in_src, in_eidx, deg_in, dev, a.dt, TOLS)
        print(f"\n=== sigma = {sig}  ({RUNS[sig]}, {data['E']} edges) ===", flush=True)
        print(f"{'tol':>8} {'solver':>6} {'R2 all':>12} {'R2 no-blowup':>13} "
              f"{'n blown':>8} {'max|err|':>10} {'med|err|':>9}")
        for tol in TOLS:
            for mode in ("gram", "svd"):
                Wh = res[(mode, tol)]
                err = np.abs(Wh - Wt)
                keep = err <= BLOWUP
                print(f"{tol:8.0e} {mode:>6} "
                      f"{compute_r_squared_NSE(Wt, Wh)[0]:12.4f} "
                      f"{compute_r_squared_NSE(Wt[keep], Wh[keep])[0]:13.4f} "
                      f"{int((~keep).sum()):8d} {err.max():10.2f} "
                      f"{np.median(err):9.5f}", flush=True)


if __name__ == "__main__":
    main()
