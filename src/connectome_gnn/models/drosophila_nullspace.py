#!/usr/bin/env python
"""
drosophila_nullspace.py
=======================
Structural null-space analysis for the drosophila CX RNN.

Companion to ``flyvis_nullspace.py`` (Flyvis equivalent). Same overall
structure, applied to the trained ``drosophila_cx_pi`` DrosophilaCxTaskRNN using
noise-free voltage traces produced by
``GNN_Main.py -o generate drosophila_cx_pi_voltage_noise_free.yaml``.

Scientific context
------------------
The CX RNN dynamics for postsynaptic neuron i:

    tau * dh_i/dt = -h_i + sum_j W_rec_ij * sigma(h_j(t)) + stim_i(t) + b_i

where tau = 0.1 s (global), sigma = sigmoid, stim_i is the already-projected
per-neuron input (W_in folded in), and b_i is the learned bias.

With (tau, W_in, b) held fixed at GT, recovery of W_rec from voltage traces
is a linear per-neuron least-squares problem:

    H_i w_i = beta_i
    H_i in R^{T x d_i}  with columns sigma(h_j(t)) for j in N_i
    beta_i in R^T       = tau * dh_i/dt + h_i(t) - b_i - stim_i(t)

This script:
  STEP 1 — Load voltage / stimulus / ode_params for noise_free dataset.
  STEP 2 — Build degenerate groups by (post-neuron, pre-type) [Flyvis convention].
  STEP 3 — Structural null dim = sum_{i,alpha: k>1} (k_{i,alpha} - 1).
  STEP 4 — Per-neuron LS recovery of W_rec + scatter figure
           (recovered vs GT W, coloured null/sloppy/well-conditioned).
  STEP 5 — Single-type sum-zero variants at lambda in {0.5, 1, 2, 4, 8};
           rollout 1000 frames with DrosophilaCxTaskRNN Euler step; per-type Conn R^2
           and rollout Pearson r table (LaTeX).
  STEP 6 — JSON dump of all numeric results.
  STEP 7 — Sparse-W collapse: zero (k - 1) edges per group, keep one
           representative; two variants (sum-preserving and calibrated).

Outputs (written under figures/):
    structural_nullspace_cx.json
    tab_lambda_1_cx.tex
    fig_lstsq_param_recovery_cx.pdf / .png

Run with:
    conda activate neural-graph-linux
    python src/connectome_gnn/models/drosophila_nullspace.py
"""

import json
import os
import sys
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures", "drosophila")

DATA_DIR = ("/groups/saalfeld/home/allierc/GraphData/graphs_data/"
            "drosophila_cx/drosophila_cx_pi_voltage_noise_free")
ODE_PARAMS_PATH = os.path.join(DATA_DIR, "ode_params.pt")

# Rollout config (DrosophilaCxTaskRNN uses dt = 0.01 s, tau = 0.1 s).
DT          = 0.01
N_ROLLOUT   = 1_000     # frames for variant rollouts
WARMUP      = 10        # frames dropped from the head of the per-neuron LS

# Variant amplitudes (mirror flyvis_nullspace.py).
PERTURBATION_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)
SEED                = 42


# ===========================================================================
# STEP 1 — Data loading
# ===========================================================================

def load_ground_truth():
    """Load GT W, edge_index, tau, b from ode_params.pt.

    The CX generator stores edge_index in standard (src=pre, dst=post)
    convention (already transposed at save). W is signed (sign baked in,
    not magnitude).

    Returns:
        W_gt        (E,)     signed edge weights at edge_index positions
        edge_index  (2, E)   row 0 = src=pre, row 1 = dst=post
        tau         scalar   global membrane time constant
        b           (N,)     per-neuron bias (DrosophilaCxTaskRNN.b)
        type_names  list[str]  cell-type names (7 entries for CX)
        state       dict     raw state dict (for later reconstruction)
    """
    state      = torch.load(ODE_PARAMS_PATH, map_location="cpu",
                            weights_only=False)
    W_gt       = state["W"].numpy().copy()        # (E,) float32 signed
    edge_index = state["edge_index"].numpy()      # (2, E) int64 (pre, post)
    tau_i      = state["tau_i"].numpy()           # (N,) uniform
    tau        = float(tau_i[0])                  # scalar
    b          = state["V_i_rest"].numpy()        # (N,) bias
    type_names = list(state.get("type_names", []))
    return W_gt, edge_index, tau, b, type_names, state


def load_neuron_instances():
    """Per-neuron hemibrain `instance` string, in the same order as voltage.zarr.

    Replicates the row selection used by
    ``connectome_gnn.generators.connconstr_data.load_drosophila_cx_connectome``
    (CSV sort by ``instance``; subselect EPG / PEN / Delta7 / PEG / ER6;
    EPG topological-ring reordering) without importing the loader (which
    depends on h5py, not available in every env). Returns a list of 156
    instance strings.

    Two same-type neurons share the same instance string iff they sit in the
    same hemibrain "computational unit" (PB glomerulus for EPG/EPGt/PEG/PEN,
    spanned-glomerulus set for Delta7, hemisphere for ER6). This is the
    natural "clone-group" partition for the CX in the sense of Hulse 2024.
    """
    import pandas as pd
    csv_path = os.path.join(
        REPO_ROOT,
        "papers/Code_NN/Code_NN/Data/Figure5/"
        "exported-traced-adjacencies-v1.2/traced-neurons.csv",
    )
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=["instance"], ignore_index=True)
    types = df["type"].astype(str).to_numpy()
    instances = df["instance"].astype(str).to_numpy()

    def getsubtype(t):
        return np.array(
            [i for i, x in enumerate(types) if t in x], dtype=int
        )

    epg    = getsubtype("EPG")
    pen    = getsubtype("PEN")
    peg    = getsubtype("PEG")
    delta7 = getsubtype("Delta7")
    allcx = np.concatenate((epg, pen, delta7, peg))
    # EPG topological-ring reorder (matches connconstr_data lines 134-139).
    allcx[0:46] = allcx[[
        23, 24,  0,  1, 42, 43, 44, 45,  2,  3, 39, 40, 41,  4,  5,  6,
        36, 37, 38,  7,  8,  9, 33, 34, 35, 10, 11, 12,
        30, 31, 32, 13, 14, 15, 27, 28, 29, 16, 17, 18,
        25, 26, 19, 20, 21, 22,
    ]]
    er6 = np.array([i for i, t in enumerate(types) if t == "ER6"], dtype=int)
    if er6.size:
        allcx = np.concatenate((allcx, er6))
    return instances[allcx].tolist()


def build_unit_groups(edge_index, neuron_type, instances, n_edges):
    """Group edges by (post_neuron, pre_type, pre_instance).

    Replaces the Flyvis-style (post, pre-type) partition with a finer one
    that respects the hemibrain computational-unit structure. Two same-type
    presynaptic neurons share a group iff they share the same hemibrain
    instance string, i.e., they are true clones (same heading tuning).
    """
    src, dst = edge_index[0], edge_index[1]
    raw = defaultdict(list)
    for e in range(n_edges):
        s = int(src[e]); d = int(dst[e])
        key = (d, int(neuron_type[s]), instances[s])
        raw[key].append(e)
    groups = {k: np.array(v, dtype=np.int64)
              for k, v in raw.items() if len(v) >= 2}
    return groups


def load_voltage_stim_types():
    """Load voltage(T,N), stimulus(T,N), neuron_type(N,) from the train split.

    Voltage is the subthreshold h_i(t) (DrosophilaCxTaskRNN convention). Stimulus is
    already per-neuron projected (Σ_l W_in[i,l] u_l(t)). neuron_type maps
    neuron index to cell-type ID.
    """
    voltage  = np.asarray(zarr.open(
        os.path.join(DATA_DIR, "x_list_train", "voltage.zarr"), mode="r"))
    stimulus = np.asarray(zarr.open(
        os.path.join(DATA_DIR, "x_list_train", "stimulus.zarr"), mode="r"))
    ntype    = np.asarray(zarr.open(
        os.path.join(DATA_DIR, "x_list_train", "neuron_type.zarr"), mode="r"),
        dtype=np.int64)
    return voltage.astype(np.float32), stimulus.astype(np.float32), ntype


def load_trained_decoder(
    ckpt_path="/groups/saalfeld/home/allierc/GraphData/log/"
              "drosophila_cx/drosophila_cx_pi/models/"
              "best_model_with_0_graphs_9.pt",
):
    """Return the trained DrosophilaCxTaskRNN's (W_out, b_out) for decoding HD."""
    sd = torch.load(ckpt_path, map_location="cpu",
                    weights_only=False)["model_state_dict"]
    W_out = sd["W_out"].cpu().numpy().astype(np.float32)   # (2, N)
    b_out = sd["b_out"].cpu().numpy().astype(np.float32)   # (2,)
    return W_out, b_out


def load_W_con(
    ckpt_path="/groups/saalfeld/home/allierc/GraphData/log/"
              "drosophila_cx/drosophila_cx_pi/models/"
              "best_model_with_0_graphs_9.pt",
):
    """Return the baseline connectome template W_con (N, N) from the
    DrosophilaCxTaskRNN buffer. This is the hemibrain J_effective used as the
    structural prior at training time.
    """
    sd = torch.load(ckpt_path, map_location="cpu",
                    weights_only=False)["model_state_dict"]
    return sd["W_con"].cpu().numpy().astype(np.float32)   # (N, N)


def load_initial_voltage():
    """Load h(0) — initial subthreshold state for variant rollouts."""
    v_path = os.path.join(DATA_DIR, "x_list_train", "voltage.zarr")
    z = zarr.open(v_path, mode="r")
    return torch.from_numpy(z[0].astype(np.float32))


def load_test_stimulus(n_frames):
    """Load the first `n_frames` of the test-split stimulus for rollouts."""
    s_path = os.path.join(DATA_DIR, "x_list_test", "stimulus.zarr")
    z = zarr.open(s_path, mode="r")
    return torch.from_numpy(z[:n_frames].astype(np.float32))


def load_test_voltage(n_frames):
    """Load the first `n_frames` of the test-split voltage for GT reference."""
    v_path = os.path.join(DATA_DIR, "x_list_test", "voltage.zarr")
    z = zarr.open(v_path, mode="r")
    return torch.from_numpy(z[:n_frames].astype(np.float32))


# ===========================================================================
# STEP 2 — Degenerate-group identification
# ===========================================================================

def build_degenerate_groups(edge_index, neuron_type, n_edges):
    """Mirror of flyvis_nullspace.build_degenerate_groups for the CX.

    DEGENERATE GROUP definition (Flyvis convention):
        All edges (src -> dst) where src has the same cell type alpha AND
        the same postsynaptic neuron dst. Key: (dst, src_type).

    If a group has k >= 2 edges:
        - The k - 1 redistributive directions are free (in ker(H_dst)).
        - The cell type alpha is marked as "has degeneracy".

    Returns:
        groups, type_has_degeneracy, null_dim_per_type, src_types_seen
    """
    src, dst = edge_index[0], edge_index[1]
    raw = defaultdict(list)
    for e in range(n_edges):
        key = (int(dst[e]), int(neuron_type[src[e]]))
        raw[key].append(e)

    src_types_seen = set(int(neuron_type[src[e]]) for e in range(n_edges))

    groups              = {}
    type_has_degeneracy = {}
    null_dim_per_type   = {}

    for key, edge_list in raw.items():
        _, src_type = key
        k = len(edge_list)
        if k >= 2:
            groups[key] = np.array(edge_list, dtype=np.int64)
            type_has_degeneracy[src_type] = True
            null_dim_per_type[src_type] = (
                null_dim_per_type.get(src_type, 0) + (k - 1)
            )
        else:
            if src_type not in type_has_degeneracy:
                type_has_degeneracy[src_type] = False

    return groups, type_has_degeneracy, null_dim_per_type, src_types_seen


def report_type_degeneracy(type_has_degeneracy, null_dim_per_type,
                            type_names, src_types_seen):
    """Print classification table and return sorted type lists."""
    all_seen = sorted(type_has_degeneracy.keys())
    degenerate_types   = [t for t in all_seen if     type_has_degeneracy[t]]
    identifiable_types = [t for t in all_seen if not type_has_degeneracy[t]]
    all_type_ids       = set(range(len(type_names)))
    no_outgoing_types  = sorted(all_type_ids - src_types_seen)

    def tname(t):
        return type_names[t] if t < len(type_names) else f"type_{t}"

    print(f"\n{'='*60}")
    print(f"STEP 2 — CELL TYPE DEGENERACY CLASSIFICATION")
    print(f"{'='*60}")
    print(f"  Total cell types (CX):              {len(type_names)}")
    print(f"  Types appearing as presynaptic:     {len(all_seen)}")
    print(f"  Types WITH degenerate groups:       {len(degenerate_types)}")
    print(f"  Types WITHOUT degenerate groups:    {len(identifiable_types)}")
    print(f"  Types with NO outgoing edges:       {len(no_outgoing_types)}"
          f"  ({', '.join(tname(t) for t in no_outgoing_types)})")

    if identifiable_types:
        print(f"\n  Cell types WITHOUT degenerate groups (identifiable):")
        for t in identifiable_types:
            print(f"    type {t:2d}: {tname(t)}")

    print(f"\n  Cell types WITH degenerate groups (null_dim, descending):")
    print(f"    {'Type':>4}  {'Name':<12}  {'null_dim':>10}")
    for t in sorted(degenerate_types, key=lambda i: -null_dim_per_type.get(i, 0)):
        nd = null_dim_per_type.get(t, 0)
        print(f"    {t:4d}  {tname(t):<12}  {nd:10d}")

    return degenerate_types, identifiable_types, no_outgoing_types


# ===========================================================================
# STEP 3 — Structural null space dimension
# ===========================================================================

def compute_null_dim(groups, n_edges):
    """dim ker(H) = sum_groups (k - 1) over all degenerate groups."""
    null_dim = sum(len(idx) - 1 for idx in groups.values())
    pct      = 100.0 * null_dim / n_edges
    return null_dim, pct


def global_svd_null_dim(voltage, edge_index, n_neurons, n_edges,
                         thresholds=(0.90, 0.95, 0.99, 0.995, 0.999),
                         t_subsample=8000):
    """Coarse upper-bound null-space dim via SVD of the global activity matrix.

    Mirrors degeneracy_analysis.tex §Global SVD approach: build H \in
    R^{T_sub x N} from sigma(h(t)) (CX firing rates), SVD it, find the
    effective rank r at each cumulative-variance threshold. Each per-neuron
    activity matrix H_i is a column-submatrix of H, so rank(H_i) <= r and
    dim ker(H_i) >= max(0, d_i - r). Summing over neurons gives a coarse
    upper bound on the total null-space dimension.

    Returns a list of dicts with keys: threshold, rank, null_dim, pct_edges.
    """
    # Subsample frames uniformly to keep the SVD cheap (mirrors Flyvis: 8000
    # frames out of 64000). Apply sigmoid -- the LS system uses sigma(h) as
    # the columns of H_i.
    T = voltage.shape[0]
    step = max(1, T // t_subsample)
    h_sub = voltage[::step]                          # (T_sub, N)
    r_sub = 1.0 / (1.0 + np.exp(-h_sub.astype(np.float64)))   # firing rates

    # Per-neuron in-degree
    dst = edge_index[1]
    d_i = np.bincount(dst, minlength=n_neurons)      # (N,)

    # SVD of (T_sub, N)
    s = np.linalg.svd(r_sub, compute_uv=False)       # (min(T_sub, N),)
    var = s ** 2
    cumvar = np.cumsum(var) / var.sum()

    results = []
    for thr in thresholds:
        rank = int(np.searchsorted(cumvar, thr) + 1)
        rank = min(rank, len(s))
        null_dim = int(np.sum(np.maximum(0, d_i - rank)))
        pct = 100.0 * null_dim / n_edges
        results.append({
            "threshold": float(thr),
            "rank": rank,
            "null_dim": null_dim,
            "pct_edges": pct,
        })
    return results, d_i


# ===========================================================================
# STEP 4 — Per-neuron least-squares recovery of W_rec
# ===========================================================================

def per_neuron_lstsq_recovery(
    voltage, stimulus, edge_index, tau, b,
    null_eig_tol=1e-22, sloppy_eig_tol=1e-12, null_comp_tol=1e-3,
    device=None,
):
    """Recover per-neuron W_rec via min-norm LS with degeneracy flagging.

    Per-neuron system:
        H_i w_i = beta_i
        H_i[t, j_k] = sigma(h_{j_k}(t))     for j_k in N_i
        beta_i[t]   = tau * dh_i/dt + h_i(t) - b_i - stim_i(t)

    Direction classification (rel = sigma_k / sigma_max of H_i^T H_i):
        rel <= null_eig_tol   -> null
        rel <= sloppy_eig_tol -> sloppy
        else                  -> well-conditioned

    Returns dict with W_lstsq (E,), W_null (E,) bool, W_sloppy (E,) bool,
    plus per-neuron R^2 statistics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T_full, N = voltage.shape
    E = edge_index.shape[1]
    src, dst = edge_index[0], edge_index[1]

    # In-edges per postsynaptic neuron
    order = np.argsort(dst)
    src_sorted = src[order]
    dst_sorted = dst[order]
    boundaries = np.searchsorted(dst_sorted, np.arange(N + 1))
    in_src  = [src_sorted[boundaries[i]:boundaries[i+1]] for i in range(N)]
    in_eidx = [order[boundaries[i]:boundaries[i+1]] for i in range(N)]
    deg_in  = np.array([len(s) for s in in_src])
    active  = np.where(deg_in > 0)[0]

    # dh/dt by finite difference. Drop the head warmup window.
    dh = (voltage[1:] - voltage[:-1]) / DT                  # (T-1, N)
    h  = voltage[:-1]                                       # (T-1, N) aligned to dh
    r  = 1.0 / (1.0 + np.exp(-h))                           # sigma(h)
    beta = tau * dh + h - b[None, :] - stimulus[:-1]        # (T-1, N) residual

    # Drop warmup rows (head of trajectory has transients).
    dh   = dh[WARMUP:]
    h    = h[WARMUP:]
    r    = r[WARMUP:]
    beta = beta[WARMUP:]
    T = beta.shape[0]

    # Move to device, float32 for storage, upcast to float64 inside the loop.
    r_d    = torch.from_numpy(r).float().to(device)
    beta_d = torch.from_numpy(beta).float().to(device)

    # Outputs
    W_lstsq  = np.full(E, np.nan, dtype=np.float64)
    W_null   = np.zeros(E, dtype=bool)
    W_sloppy = np.zeros(E, dtype=bool)
    per_neuron_r2 = np.full(N, np.nan, dtype=np.float64)

    t0 = time.time()
    for i in tqdm(active, desc="STEP 4 lstsq", ncols=80, unit="neuron"):
        d_i = len(in_src[i])
        # A = r_d[:, in_src[i]] in float64
        A = r_d[:, in_src[i]].double()                   # (T, d_i)
        bvec = beta_d[:, i].double()                     # (T,)

        # Column normalise
        s = A.norm(dim=0)
        s = torch.where(s > 0, s, torch.ones_like(s))
        A_s = A / s

        # Eigendecomposition of A_s^T A_s
        G = A_s.T @ A_s
        w, V = torch.linalg.eigh(G)
        w_max = w[-1]
        rel = w / w_max
        null_mask   = rel <= null_eig_tol
        sloppy_mask = (rel > null_eig_tol) & (rel <= sloppy_eig_tol)
        keep        = ~(null_mask | sloppy_mask)
        inv_w = torch.where(keep, 1.0 / w, torch.zeros_like(w))

        # OLS via column-equilibrated normal equations
        c = A_s.T @ bvec
        theta_i = (V @ (inv_w * (V.T @ c))) / s
        w_i_hat = theta_i.cpu().numpy()
        W_lstsq[in_eidx[i]] = w_i_hat

        # Per-neuron R^2: how well does H_i ŵ_i reproduce beta_i?
        pred = (A @ theta_i).cpu().numpy()
        bn   = bvec.cpu().numpy()
        ss_res = float(np.sum((bn - pred) ** 2))
        ss_tot = float(np.sum((bn - bn.mean()) ** 2))
        per_neuron_r2[i] = 1.0 - ss_res / max(ss_tot, 1e-30)

        # Flag null/sloppy directions in W-space
        for mask_t, arr in ((null_mask, W_null), (sloppy_mask, W_sloppy)):
            if int(mask_t.sum().item()) == 0:
                continue
            V_theta = V[:, mask_t] / s.unsqueeze(1)
            V_theta = V_theta / V_theta.norm(dim=0, keepdim=True).clamp_min(1e-300)
            V_null = V_theta.abs()
            V_null = V_null / V_null.amax(dim=0, keepdim=True).clamp_min(1e-300)
            part = V_null.amax(dim=1).cpu().numpy()
            edge_flag = part > null_comp_tol
            if edge_flag.any():
                arr[in_eidx[i][edge_flag]] = True

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  solve time: {time.time()-t0:.1f}s")
    print(f"  null   : {W_null.sum()}/{E} edges")
    print(f"  sloppy : {W_sloppy.sum()}/{E} edges")

    return dict(
        W_lstsq=W_lstsq,
        W_null=W_null,
        W_sloppy=W_sloppy,
        per_neuron_r2=per_neuron_r2,
        deg_in=deg_in,
    )


def plot_lstsq_recovery(W_gt, lstsq_out, out_path):
    """Single-panel scatter of recovered vs GT W, coloured by direction class.

    Mirrors the W panel of the Flyvis fig_lstsq_param_recovery figure.
    Black = well-conditioned; orange = sloppy; red = null.
    R^2 (well-conditioned) + slope reported on-panel.
    """
    W_hat = lstsq_out["W_lstsq"]
    null   = lstsq_out["W_null"]
    sloppy = lstsq_out["W_sloppy"]
    well   = ~(null | sloppy) & np.isfinite(W_hat)

    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = float(W_gt.min()), float(W_gt.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--", label="y = x")

    if well.sum():
        ax.scatter(W_gt[well], W_hat[well], s=6, color="black",
                    alpha=0.45, edgecolors="none", label="well-conditioned")
    if sloppy.sum():
        ax.scatter(W_gt[sloppy], W_hat[sloppy], s=10, color="#d29922",
                    alpha=0.65, edgecolors="none", label="sloppy")
    if null.sum():
        ax.scatter(W_gt[null], W_hat[null], s=10, color="#cf222e",
                    alpha=0.75, edgecolors="none", label="null")

    if well.sum() >= 2:
        x = W_gt[well]; y = W_hat[well]
        slope, intercept = np.polyfit(x, y, 1)
        ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        all_finite = np.isfinite(W_hat)
        x_all = W_gt[all_finite]; y_all = W_hat[all_finite]
        ss_res_all = float(np.sum((y_all - x_all) ** 2))
        ss_tot_all = float(np.sum((x_all - x_all.mean()) ** 2))
        r2_all = 1.0 - ss_res_all / max(ss_tot_all, 1e-30)
        ax.text(0.03, 0.97,
                f"R$^2$ (well) = {r2:.3f}    slope = {slope:.3f}\n"
                f"R$^2$ (all)  = {r2_all:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=11,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                            boxstyle="round,pad=0.3"))

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"GT $W_{ij}$", fontsize=14)
    ax.set_ylabel(r"recovered $\hat W_{ij}$", fontsize=14)
    ax.set_title("Per-neuron least-squares recovery of $W_{\\mathrm{rec}}$",
                 fontsize=13)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  scatter written to {out_path}")


# ===========================================================================
# CX Euler rollout (mirrors DrosophilaCxTaskRNN.forward; used by STEPS 5 + 7)
# ===========================================================================

def _build_W_rec_dense(W_edge, edge_index, N):
    """Build dense (N, N) W_rec[post, pre] from edge weights."""
    W = np.zeros((N, N), dtype=np.float32)
    src = edge_index[0]; dst = edge_index[1]
    # edge_index is (pre, post) — fill W[post, pre]
    W[dst, src] = W_edge
    return W


def cx_rollout(W_edge, edge_index, b, tau, stim, v0, n_steps, device):
    """Roll the CX RNN forward with a custom W_edge.

    Implements (mirrors DrosophilaCxTaskRNN.forward):
        r       = sigmoid(h)                          # (B=1, N)
        rec     = r @ W_rec.T                         # W_rec[post, pre]
        h       = h + (dt/tau) * (-h + rec + stim + b)

    Returns voltage trace (n_steps, N) on CPU.
    """
    N = v0.shape[0]
    W_rec = torch.from_numpy(
        _build_W_rec_dense(W_edge, edge_index, N)
    ).to(device)
    b_d    = torch.from_numpy(b).to(device)
    stim_d = stim.to(device)
    h      = v0.clone().to(device)
    dt_over_tau = DT / tau

    out = torch.empty(n_steps, N, dtype=torch.float32)
    with torch.no_grad():
        for t in range(n_steps):
            out[t] = h.cpu()
            r = torch.sigmoid(h)
            rec = r @ W_rec.T
            h = h + dt_over_tau * (-h + rec + stim_d[t] + b_d)
    return out


def rollout_and_metrics(W_variant, W_gt, edge_index, b, tau, stim, v0, v_gt,
                        device):
    """Run a variant rollout; return connectivity R^2 + mean Pearson r vs GT."""
    diff   = W_variant - W_gt
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((W_gt - W_gt.mean()) ** 2))
    conn_r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    v_var = cx_rollout(W_variant, edge_index, b, tau, stim, v0,
                       n_steps=v_gt.shape[0], device=device)
    v_var_np = v_var.numpy()
    v_gt_np  = v_gt.numpy()

    pearson_t = np.zeros(v_gt_np.shape[0], dtype=np.float64)
    for t in range(v_gt_np.shape[0]):
        a = v_var_np[t] - v_var_np[t].mean()
        c = v_gt_np[t] - v_gt_np[t].mean()
        denom = np.sqrt(np.sum(a * a) * np.sum(c * c))
        pearson_t[t] = float(np.sum(a * c) / denom) if denom > 1e-12 else 0.0

    return conn_r2, float(np.mean(pearson_t))


# ===========================================================================
# STEP 5 — Single-type variants
# ===========================================================================

def sum_zero_vector(k, rng):
    """Random unit vector of length k that sums to zero."""
    v = rng.randn(k).astype(np.float64)
    v -= v.mean()
    norm = np.linalg.norm(v)
    return (v / norm) if norm > 1e-12 else v


def make_single_type_variant(W_gt, groups, type_id, scale, rng):
    """Sum-zero perturbation along the null space of one cell type."""
    W_var = W_gt.copy()
    for (dst_n, src_t), edge_idx in groups.items():
        if src_t != type_id:
            continue
        delta_unit = sum_zero_vector(len(edge_idx), rng)
        amplitude  = scale * float(np.mean(np.abs(W_gt[edge_idx])))
        W_var[edge_idx] += amplitude * delta_unit
    return W_var


# ===========================================================================
# STEP 7 — Sparse-W collapse
# ===========================================================================

def sparse_sum_preserving(W_gt, groups):
    """Collapse each (post, pre-type) group onto one representative edge,
    setting its weight to the group sum and zeroing the rest. Exact
    null-space member when same-type pre-synaptic activities are identical.
    """
    W_sparse = W_gt.copy()
    n_zeroed = 0
    for edge_idx in groups.values():
        k = len(edge_idx)
        rep_pos = int(np.argmax(np.abs(W_gt[edge_idx])))
        rep_e = edge_idx[rep_pos]
        group_sum = float(W_gt[edge_idx].sum())
        W_sparse[edge_idx] = 0.0
        W_sparse[rep_e] = group_sum
        n_zeroed += k - 1
    ss_res = float(np.sum((W_gt - W_sparse) ** 2))
    ss_tot = float(np.sum((W_gt - W_gt.mean()) ** 2))
    conn_r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return W_sparse, {
        "method": "sum_preserving",
        "n_edges_zeroed": n_zeroed,
        "n_edges_total": len(W_gt),
        "frac_zeroed": n_zeroed / len(W_gt),
        "conn_r2": conn_r2,
    }


def sparse_calibrated(W_gt, groups, edge_index, voltage_np):
    """Per-group OLS scalar fit for the representative-edge weight.

    Same collapse as sum_preserving, but W_rep solves
        W_rep = <h_rep, target> / <h_rep, h_rep>
    where target(t) = sum_{j in group} W_j * sigma(h_j(t)) (the group's GT
    contribution to the postsynaptic drive) and h_rep(t) = sigma(h_rep_neuron(t)).
    """
    src_np = edge_index[0]
    step = max(1, voltage_np.shape[0] // 1000)
    # CX uses sigmoid (not ReLU) as the firing-rate non-linearity.
    h = 1.0 / (1.0 + np.exp(-voltage_np[::step].astype(np.float64)))  # (T', N)

    W_calib = W_gt.copy()
    n_zeroed = 0
    for edge_idx in groups.values():
        k = len(edge_idx)
        rep_pos = int(np.argmax(np.abs(W_gt[edge_idx])))
        rep_e = edge_idx[rep_pos]
        rep_src = int(src_np[rep_e])

        target = np.zeros(h.shape[0], dtype=np.float64)
        for e in edge_idx:
            target += float(W_gt[e]) * h[:, int(src_np[e])]

        h_rep = h[:, rep_src]
        denom = float(np.sum(h_rep ** 2))
        if denom < 1e-12:
            W_rep = float(W_gt[edge_idx].sum())
        else:
            W_rep = float(np.sum(h_rep * target) / denom)

        W_calib[edge_idx] = 0.0
        W_calib[rep_e] = W_rep
        n_zeroed += k - 1

    ss_res = float(np.sum((W_gt - W_calib) ** 2))
    ss_tot = float(np.sum((W_gt - W_gt.mean()) ** 2))
    conn_r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return W_calib, {
        "method": "calibrated",
        "n_edges_zeroed": n_zeroed,
        "n_edges_total": len(W_gt),
        "frac_zeroed": n_zeroed / len(W_gt),
        "conn_r2": conn_r2,
    }


# ===========================================================================
# STEP 6 — LaTeX table writer
# ===========================================================================

def spectral_order(W_gt, edge_index, n_neurons):
    """Fiedler-vector reordering of the recurrent matrix.

    Build the symmetric adjacency $A = |W| + |W|^\top$, form the
    graph Laplacian $L = D - A$, and order neurons by the value of the
    second smallest eigenvector (the Fiedler vector). Strongly connected
    neurons land near each other along the diagonal.
    """
    from scipy.sparse.linalg import eigsh
    src, dst = edge_index[0], edge_index[1]
    M = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    M[dst, src] = np.abs(W_gt)
    A = M + M.T
    d = A.sum(axis=1)
    L = np.diag(d) - A
    # Compute the two smallest eigenvalues / vectors; Fiedler is the second.
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    return np.argsort(fiedler)


def rcm_order(W_gt, edge_index, n_neurons):
    """Reverse Cuthill-McKee ordering on $|W|$ to minimise bandwidth."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    src, dst = edge_index[0], edge_index[1]
    M = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    M[dst, src] = np.abs(W_gt)
    A = M + M.T
    return np.asarray(reverse_cuthill_mckee(csr_matrix(A), symmetric_mode=True))


def two_level_order(neuron_types, instances):
    """Lexicographic sort: cell type first, instance second.

    Same-type neurons are kept contiguous (preserving the seven type-blocks);
    within each block, neurons are grouped by hemibrain instance so the
    per-(i,a,instance) "computational unit" sub-blocks become visible.
    """
    nt = np.asarray(neuron_types)
    keys = list(zip(nt.tolist(), instances))
    return np.argsort(np.array(keys, dtype=[("t", int), ("i", "U64")]),
                        order=["t", "i"])


def plot_wcon_orderings(W_con_dense, neuron_types, instances, type_names,
                          out_path):
    """Two-panel rendering of the baseline connectome $W^{\\mathrm{con}}$.

    Both panels show the same matrix, z-scored on its own non-zero
    entries; only the row/column ordering differs:
        (a) cell-type sort
        (b) cell type, then hemibrain instance.
    The (b) ordering exposes the canonical PEN-mediated phase-shift "X"
    motif in the EPG <-> PEN sub-quadrants, the diagonal local
    excitation in EPG-EPG, and the multi-diagonal structure of the
    Delta7 column.
    """
    N = neuron_types.size
    nz = W_con_dense[W_con_dense != 0]
    mu, sigma = float(nz.mean()), float(nz.std())
    sigma = max(sigma, 1e-8)
    def zscore(M):
        return np.clip(np.where(M != 0, (M - mu) / sigma, 0.0), -3.0, 3.0)

    nt = np.asarray(neuron_types)
    order_type = np.argsort(nt, kind="stable")
    order_two  = two_level_order(neuron_types, instances)
    panels = [
        ("cell-type sort",            order_type),
        ("cell type, then instance",  order_two),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8),
                              gridspec_kw=dict(wspace=0.35))
    for ax, (title, order) in zip(axes, panels):
        Z = zscore(W_con_dense[order, :][:, order])
        im = ax.imshow(Z, cmap="RdBu_r", vmin=-3.0, vmax=3.0,
                        interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("presynaptic", fontsize=9)
        ax.set_ylabel("postsynaptic", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        nt_ord = nt[order]
        bnd = np.where(np.diff(nt_ord) != 0)[0] + 0.5
        for x in bnd:
            ax.axvline(x, color="k", lw=0.4, alpha=0.5)
            ax.axhline(x, color="k", lw=0.4, alpha=0.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=8)
    for ax, letter in zip(axes, ["a", "b"]):
        ax.text(-0.07, 1.04, letter, transform=ax.transAxes,
                 fontsize=16, fontweight="bold", va="bottom", ha="right")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if out_path.endswith(".pdf"):
        fig.savefig(out_path.replace(".pdf", ".png"), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  W_con ordering figure written to {out_path}")


def plot_sparsify_unit_orderings(W_gt, W_sparse_unit, edge_index,
                                    neuron_types, instances, type_names,
                                    out_path):
    """Four-panel comparison of orderings on two matrices.

    Top row: the LEARNED W_rec under (a) cell-type sort, (b) cell type then instance.
    Bottom row: the per-(i,a,instance)-collapsed W under (c), (d) the same two sorts.
    Same z-score colour scale across all four panels.
    """
    N = neuron_types.size
    src = edge_index[0]; dst = edge_index[1]

    def edges_to_dense(W_edge):
        M = np.zeros((N, N), dtype=np.float32)
        M[dst, src] = W_edge
        return M

    M_gt   = edges_to_dense(W_gt)
    M_unit = edges_to_dense(W_sparse_unit)

    nz = W_gt[W_gt != 0]
    mu, sigma = float(nz.mean()), float(nz.std())
    sigma = max(sigma, 1e-8)
    def zscore(M):
        return np.clip(np.where(M != 0, (M - mu) / sigma, 0.0), -3.0, 3.0)

    nt = np.asarray(neuron_types)
    order_type = np.argsort(nt, kind="stable")
    order_two  = two_level_order(neuron_types, instances)

    panels = [
        ("learned $\\hat W^{\\mathrm{rec}}$ -- cell-type sort",  M_gt,   order_type),
        ("learned $\\hat W^{\\mathrm{rec}}$ -- cell type, then instance",
                                                                M_gt,   order_two),
        ("per-$(i,\\alpha,\\mathrm{instance})$ collapse -- cell-type sort",
                                                                M_unit, order_type),
        ("per-$(i,\\alpha,\\mathrm{instance})$ collapse -- cell type, then instance",
                                                                M_unit, order_two),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 13),
                              gridspec_kw=dict(wspace=0.32, hspace=0.32))
    axes_flat = axes.flatten()
    for ax, (title, M, order) in zip(axes_flat, panels):
        Z = zscore(M[order, :][:, order])
        im = ax.imshow(Z, cmap="RdBu_r", vmin=-3.0, vmax=3.0,
                        interpolation="nearest", aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("presynaptic", fontsize=9)
        ax.set_ylabel("postsynaptic", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        nt_ord = nt[order]
        bnd = np.where(np.diff(nt_ord) != 0)[0] + 0.5
        for x in bnd:
            ax.axvline(x, color="k", lw=0.4, alpha=0.5)
            ax.axhline(x, color="k", lw=0.4, alpha=0.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=8)
    for ax, letter in zip(axes_flat, ["a", "b", "c", "d"]):
        ax.text(-0.07, 1.04, letter, transform=ax.transAxes,
                 fontsize=16, fontweight="bold", va="bottom", ha="right")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if out_path.endswith(".pdf"):
        fig.savefig(out_path.replace(".pdf", ".png"), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  ordering-comparison figure written to {out_path}")


def plot_sparsify_figure(W_gt, W_sparse_type, W_sparse_unit,
                           edge_index, neuron_types, type_names,
                           out_path,
                           pct_type, pct_unit, r_type, r_unit,
                           hd_trial=None, b=None, tau=None, device=None):
    """2x3 figure: top row = matrices, bottom row = HD on one OU test trial.

    All three matrix panels share a z-score colour scale derived from the
    GT non-zero entries. Each bottom-row HD panel shows the same OU test
    trial decoded by the corresponding W_rec; true HD in light green,
    decoded HD in black, both wrapped to (-pi, pi].
    """
    N = neuron_types.size
    src = edge_index[0]; dst = edge_index[1]

    def edges_to_dense(W_edge):
        M = np.zeros((N, N), dtype=np.float32)
        M[dst, src] = W_edge
        return M

    M_gt   = edges_to_dense(W_gt)
    M_type = edges_to_dense(W_sparse_type)
    M_unit = edges_to_dense(W_sparse_unit)

    nz = W_gt[W_gt != 0]
    mu, sigma = float(nz.mean()), float(nz.std())
    sigma = max(sigma, 1e-8)
    def zscore(M):
        return np.clip(np.where(M != 0, (M - mu) / sigma, 0.0), -3.0, 3.0)

    nt = np.asarray(neuron_types)
    order = np.argsort(nt, kind="stable")
    nt_sorted = nt[order]
    bnd = np.where(np.diff(nt_sorted) != 0)[0] + 0.5
    bounds = np.concatenate([[0], bnd + 0.5, [N]])
    centres = (bounds[:-1] + bounds[1:]) / 2 - 0.5
    tick_labels = [type_names[int(nt_sorted[int(c)])] for c in centres]

    # Explicit gridspec with a dedicated colorbar column so the colorbar
    # doesn't shrink the top-row axes (which would misalign them against
    # the bottom row of HD panels). Bottom row has an empty cell in the
    # colorbar column.
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.035],
        height_ratios=[1.0, 0.50],
        wspace=0.32, hspace=0.32,
        left=0.06, right=0.96, top=0.94, bottom=0.08,
    )
    ax_top = [fig.add_subplot(gs[0, j]) for j in range(3)]
    cax    = fig.add_subplot(gs[0, 3])
    ax_bot = [fig.add_subplot(gs[1, j]) for j in range(3)]
    axes = np.array([ax_top, ax_bot])

    matrix_panels = [
        (axes[0, 0], zscore(M_gt[order, :][:, order]),
         "learned $\\hat W^{\\mathrm{rec}}$"),
        (axes[0, 1], zscore(M_type[order, :][:, order]),
         f"per-$(i,\\alpha)$ collapse"),
        (axes[0, 2], zscore(M_unit[order, :][:, order]),
         f"per-$(i,\\alpha,\\mathrm{{instance}})$ collapse"),
    ]
    for ax, Z, title in matrix_panels:
        # aspect='auto' so the matrix fills its gridspec cell (cell is
        # already proportioned to match the matrix's 1:1 data aspect).
        # Mixing aspect='equal' with a constrained gridspec was causing
        # the per-axes bbox to shrink unpredictably.
        im = ax.imshow(Z, cmap="RdBu_r", vmin=-3.0, vmax=3.0,
                        interpolation="nearest", aspect="auto")
        for x in bnd:
            ax.axvline(x, color="k", lw=0.3, alpha=0.5)
            ax.axhline(x, color="k", lw=0.3, alpha=0.5)
        ax.set_xticks(centres)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(centres)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("presynaptic", fontsize=8)
        ax.set_ylabel("postsynaptic", fontsize=8)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)

    # --- Bottom row: HD on one OU test trial under each W_rec --------------
    if hd_trial is not None:
        y_true  = hd_trial["y_true"]            # (T, 2)
        dt      = float(hd_trial["dt"])
        T_full  = y_true.shape[0]
        t_axis  = np.arange(T_full) * dt
        true_hd = np.arctan2(y_true[:, 1], y_true[:, 0])
        true_wrap = np.angle(np.exp(1j * true_hd))

        stim = hd_trial["stim"]                 # (T, N) per-neuron projected
        v0 = hd_trial["v0"]                     # (N,) initial voltage
        stim_t = torch.from_numpy(stim).float()
        v0_t = torch.from_numpy(v0).float()

        hd_panels = [
            (axes[1, 0], W_gt,
             "learned $\\hat W^{\\mathrm{rec}}$"),
            (axes[1, 1], W_sparse_type,
             f"$r = {r_type:.2f}$ (sum-preserving, $91.6\\%$ zeroed)"),
            (axes[1, 2], W_sparse_unit,
             f"$r = {r_unit:.2f}$ (sum-preserving, $50.2\\%$ zeroed)"),
        ]
        for ax, W_edge, title in hd_panels:
            v = cx_rollout(W_edge, edge_index, b, tau,
                           stim_t, v0_t, n_steps=T_full, device=device)
            # Decode HD from the readout: project sigmoid(h) onto cos/sin.
            # We don't have W_out here, but the user trained DrosophilaCxTaskRNN where
            # y = W_out * sigmoid(h). Use the y_true projection direction
            # approximated by taking the same readout from the trained model
            # which is implicit in the DrosophilaCxTaskRNN. As a simpler proxy that
            # works without W_out: project the EPG-bump population vector.
            # Here we use the cosine of the per-frame bump-direction angle
            # via the readout stored in the rollout (y_pred), which we
            # compute by reading W_out from the hd_trial dict.
            W_out  = hd_trial["W_out"]
            b_out  = hd_trial["b_out"]
            r_sig  = torch.sigmoid(v).cpu().numpy()
            y_pred = r_sig @ W_out.T + b_out[None, :]
            dec_hd = np.arctan2(y_pred[:, 1], y_pred[:, 0])
            dec_wrap = np.angle(np.exp(1j * dec_hd))

            ax.plot(t_axis, true_wrap, color="#4daf4a", lw=0.0,
                    marker=".", ms=4.0, alpha=0.9)
            ax.plot(t_axis, dec_wrap, color="black", lw=0.0,
                    marker=".", ms=1.0)
            ax.set_yticks([-np.pi, 0, np.pi])
            ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=7)
            ax.set_ylim(-np.pi - 0.15, np.pi + 0.15)
            ax.set_xlabel("time (s)", fontsize=8)
            ax.set_ylabel("HD (rad)", fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=7)

    # Corner panel labels (a-f) — matches the convention in
    # docs/figure/fig_evolution.py.
    for ax, letter in zip(
        [axes[0, 0], axes[0, 1], axes[0, 2],
         axes[1, 0], axes[1, 1], axes[1, 2]],
        ["a", "b", "c", "d", "e", "f"],
    ):
        ax.text(-0.12, 1.02, letter, transform=ax.transAxes,
                 fontsize=16, fontweight="bold", va="bottom", ha="right")

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if out_path.endswith(".pdf"):
        fig.savefig(out_path.replace(".pdf", ".png"), dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  sparsify figure written to {out_path}")


def write_tab_global_svd(svd_results, n_edges, out_path):
    """Variance-threshold sweep table (CX analogue of degeneracy_analysis.tex
    §Global SVD approach).
    """
    lines = [
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        (r"Variance threshold & Effective rank $r$ & "
         r"Null space dim & \% unconstrained \\"),
        r"\midrule",
    ]
    for row in svd_results:
        thr_pct = row["threshold"] * 100.0
        # Bold the 99% row to match the Flyvis convention.
        if abs(row["threshold"] - 0.99) < 1e-9:
            lines.append(
                f"\\textbf{{{thr_pct:g}\\%}} & "
                f"\\textbf{{{row['rank']}}} & "
                f"\\textbf{{{row['null_dim']:,}}} & "
                f"\\textbf{{{row['pct_edges']:.1f}\\%}} \\\\"
            )
        else:
            lines.append(
                f"{thr_pct:g}\\% & {row['rank']} & "
                f"{row['null_dim']:,} & {row['pct_edges']:.1f}\\% \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX table written to {out_path}")


def write_tab_variants(type_results_by_scale, type_names, null_dim_per_type,
                         out_path, scales=(1.0, 2.0, 4.0)):
    """Per-type table across amplitude scales (CX analogue of Flyvis Tab 5).

    Columns: cell type | dim ker | (R^2, r) at each scale. No cell colouring.
    """
    def fmt_name(t):
        n = type_names[t] if t < len(type_names) else f"type_{t}"
        return n.replace("_", r"\_")

    base = type_results_by_scale[1.0]
    type_ids = sorted(
        [t for t, r in base.items()
         if r.get("ok", True) and r.get("pearson_r") is not None]
    )

    col_pairs = " & ".join("$R^2_{\\mathbf{W}}$ & $r$" for _ in scales)
    col_spec = "lr" + ("rr" * len(scales))
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        r"Cell Type & $\dim\ker$ & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\lambda = {sc:g}$}}"
                       for sc in scales) + r" \\",
        r" & & " + col_pairs + r" \\",
        r"\midrule",
    ]
    for t in type_ids:
        name = fmt_name(t)
        nd = int(null_dim_per_type.get(t, 0))
        cells = [f"{name:<14}", f"{nd:>4d}"]
        for sc in scales:
            r = type_results_by_scale[sc][t]
            cells.append(f"{r['conn_r2']:.3f}")
            cells.append(f"{r['pearson_r']:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX table written to {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng    = np.random.RandomState(SEED)
    print(f"Device: {device}   SEED={SEED}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 1 — Load
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nSTEP 1 — Loading data\n{'='*60}")
    W_gt, edge_index, tau, b, type_names, state = load_ground_truth()
    voltage, stimulus, ntype = load_voltage_stim_types()
    N = b.size
    E = W_gt.size
    print(f"  Data dir: {DATA_DIR}")
    print(f"  N={N}  E={E}  tau={tau:.4f}  T={voltage.shape[0]}")
    print(f"  W_gt: mean={W_gt.mean():.4f}  std={W_gt.std():.4f}  "
          f"range=[{W_gt.min():.4f}, {W_gt.max():.4f}]")
    print(f"  Cell types ({len(type_names)}): {type_names}")

    # -----------------------------------------------------------------------
    # STEP 2 — Degenerate groups
    # -----------------------------------------------------------------------
    groups, type_has_deg, null_dim_per_type, src_types_seen = (
        build_degenerate_groups(edge_index, ntype, E)
    )
    degenerate_types, identifiable_types, no_outgoing_types = (
        report_type_degeneracy(type_has_deg, null_dim_per_type, type_names,
                               src_types_seen)
    )

    # -----------------------------------------------------------------------
    # STEP 3 — Structural null dim
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nSTEP 3 — Structural null space dimension\n{'='*60}")
    null_dim, pct_edges = compute_null_dim(groups, E)
    print(f"  Per-type   (Eq 4): dim ker(H) = {null_dim:,}  "
          f"({pct_edges:.1f}% of {E:,} edges)")

    print(f"\n  Global-SVD coarse bound (rank vs in-degree):")
    svd_results, d_i = global_svd_null_dim(voltage, edge_index, N, E)
    print(f"  {'Variance':>10}  {'rank':>5}  {'null_dim':>10}  "
          f"{'% edges':>9}")
    for row in svd_results:
        print(f"  {row['threshold']*100:9.1f}%  {row['rank']:>5d}  "
              f"{row['null_dim']:>10,}  {row['pct_edges']:>8.1f}%")
    write_tab_global_svd(svd_results, E,
                          os.path.join(OUTPUT_DIR, "tab_global_svd_cx.tex"))

    # --- Finer (post, pre-type, pre-instance) partition --------------------
    # The hemibrain `instance` string identifies the computational unit
    # (Hulse 2024); two same-type neurons share an instance iff they are
    # true clones with the same heading tuning.
    print(f"\n  Per-(type, instance) partition:")
    instances = load_neuron_instances()
    assert len(instances) == N, (
        f"instance list len {len(instances)} != N {N}; "
        f"loader mirror is out of sync.")
    unit_groups = build_unit_groups(edge_index, ntype, instances, E)
    unit_null_dim = sum(len(idx) - 1 for idx in unit_groups.values())
    unit_pct = 100.0 * unit_null_dim / E
    print(f"  Per-(type, instance) (Hulse units):  "
          f"dim ker(H) = {unit_null_dim:,}  ({unit_pct:.1f}% of {E:,} edges)")
    # Count groups per cell type (for the prose).
    unit_count_by_type = defaultdict(int)
    unit_null_dim_by_type = defaultdict(int)
    for (_, src_t, _), idx in unit_groups.items():
        unit_count_by_type[src_t] += 1
        unit_null_dim_by_type[src_t] += len(idx) - 1
    print(f"  Per-type breakdown of unit groups (Hulse partition):")
    print(f"    {'Type':>4}  {'Name':<14}  {'n_groups':>8}  {'null_dim':>10}")
    for t in sorted(unit_null_dim_by_type.keys(),
                     key=lambda i: -unit_null_dim_by_type[i]):
        name = type_names[t] if t < len(type_names) else f"type_{t}"
        print(f"    {t:4d}  {name:<14}  {unit_count_by_type[t]:>8d}  "
              f"{unit_null_dim_by_type[t]:>10d}")

    # -----------------------------------------------------------------------
    # STEP 4 — Per-neuron LS recovery + figure
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nSTEP 4 — Per-neuron LS recovery\n{'='*60}")
    lstsq_out = per_neuron_lstsq_recovery(
        voltage, stimulus, edge_index, tau, b, device=device,
    )
    # Global R^2 stats
    finite = np.isfinite(lstsq_out["W_lstsq"])
    well = finite & ~lstsq_out["W_null"] & ~lstsq_out["W_sloppy"]
    if well.sum():
        x = W_gt[well]; y = lstsq_out["W_lstsq"][well]
        slope, intercept = np.polyfit(x, y, 1)
        ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_well = 1.0 - ss_res / max(ss_tot, 1e-30)
    else:
        slope, r2_well = float("nan"), float("nan")
    if finite.sum():
        x = W_gt[finite]; y = lstsq_out["W_lstsq"][finite]
        ss_res = float(np.sum((y - x) ** 2))
        ss_tot = float(np.sum((x - x.mean()) ** 2))
        r2_all = 1.0 - ss_res / max(ss_tot, 1e-30)
    else:
        r2_all = float("nan")
    print(f"  R^2_W (well-conditioned subset): {r2_well:.4f}  slope={slope:.4f}")
    print(f"  R^2_W (all edges):               {r2_all:.4f}")
    plot_lstsq_recovery(W_gt, lstsq_out,
                         os.path.join(OUTPUT_DIR,
                                       "fig_lstsq_param_recovery_cx.pdf"))

    # -----------------------------------------------------------------------
    # Pre-roll GT on test split (reference trajectory for variants).
    # -----------------------------------------------------------------------
    print(f"\n  Pre-rolling GT on test split for {N_ROLLOUT} frames ...")
    stim_test = load_test_stimulus(N_ROLLOUT)
    v0_test_full = load_test_voltage(1)
    v0_test = v0_test_full[0]
    v_gt = cx_rollout(W_gt, edge_index, b, tau, stim_test, v0_test,
                      n_steps=N_ROLLOUT, device=device)

    # -----------------------------------------------------------------------
    # STEP 5 — Single-type variants
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nSTEP 5 — Single-type variants  "
          f"(scales={PERTURBATION_SCALES})\n{'='*60}")
    type_results_by_scale = {sc: {} for sc in PERTURBATION_SCALES}
    type_results = {}
    for t in tqdm(degenerate_types, desc="  type variants", ncols=80):
        for sc in PERTURBATION_SCALES:
            W_var = make_single_type_variant(W_gt, groups, t, sc, rng)
            try:
                conn_r2, pearson_r = rollout_and_metrics(
                    W_var, W_gt, edge_index, b, tau,
                    stim_test, v0_test, v_gt, device,
                )
                rec = {"conn_r2": conn_r2, "pearson_r": pearson_r, "ok": True}
            except Exception as ex:
                print(f"\n    WARNING: type {t} ({type_names[t]}) scale={sc} "
                      f"rollout failed: {ex}")
                rec = {"conn_r2": None, "pearson_r": None, "ok": False}
            type_results_by_scale[sc][t] = rec
        # lambda = 1.0 result for the table
        type_results[t] = type_results_by_scale[1.0][t]

    # Console summary (sorted by R^2_W ascending; failed last)
    print(f"\n  Single-type variant summary at lambda=1.0:")
    print(f"  {'Type':>4}  {'Name':<12}  {'R^2_W':>7}  {'Pearson r':>9}")
    print(f"  {'-'*40}")
    sorted_types = sorted(
        degenerate_types,
        key=lambda t: (not type_results[t]["ok"],
                       type_results[t]["conn_r2"] if type_results[t]["ok"]
                       else float("inf")),
    )
    for t in sorted_types:
        r = type_results[t]
        name = type_names[t] if t < len(type_names) else f"type_{t}"
        if r["ok"]:
            print(f"  {t:4d}  {name:<12}  {r['conn_r2']:7.4f}  "
                  f"{r['pearson_r']:9.4f}")
        else:
            print(f"  {t:4d}  {name:<12}  {'FAILED':>7}  {'FAILED':>9}")

    # -----------------------------------------------------------------------
    # STEP 7 — Sparse W variants
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nSTEP 7 — Sparse W variants\n{'='*60}")

    print(f"  [7a] Sum-preserving sparse W ...")
    W_sparse, sp_stats = sparse_sum_preserving(W_gt, groups)
    print(f"       n_edges_zeroed={sp_stats['n_edges_zeroed']:,}  "
          f"({sp_stats['frac_zeroed']*100:.1f}%)  "
          f"conn_r2={sp_stats['conn_r2']:.4f}")
    sp_r2, sp_pearson = rollout_and_metrics(
        W_sparse, W_gt, edge_index, b, tau, stim_test, v0_test, v_gt, device,
    )
    sp_stats.update({"rollout_pearson_r": sp_pearson, "rollout_conn_r2": sp_r2})
    print(f"       rollout Pearson r={sp_pearson:.4f}  "
          f"conn_r2_rollout={sp_r2:.4f}")

    print(f"  [7b] Calibrated sparse W ...")
    W_calib, cal_stats = sparse_calibrated(W_gt, groups, edge_index, voltage)
    print(f"       n_edges_zeroed={cal_stats['n_edges_zeroed']:,}  "
          f"({cal_stats['frac_zeroed']*100:.1f}%)  "
          f"conn_r2={cal_stats['conn_r2']:.4f}")
    cal_r2, cal_pearson = rollout_and_metrics(
        W_calib, W_gt, edge_index, b, tau, stim_test, v0_test, v_gt, device,
    )
    cal_stats.update({"rollout_pearson_r": cal_pearson,
                      "rollout_conn_r2": cal_r2})
    print(f"       rollout Pearson r={cal_pearson:.4f}  "
          f"conn_r2_rollout={cal_r2:.4f}")

    # --- Sparse-W collapse on the FINER (type, instance) partition ----------
    # Same algorithm, smaller groups -- only true Hulse clone groups are
    # collapsed. If the per-instance partition tracks the genuine null
    # space, this should preserve the rollout much better.
    print(f"\n  [7c] Sum-preserving sparse W (per-(type, instance)) ...")
    W_sparse_u, sp_u_stats = sparse_sum_preserving(W_gt, unit_groups)
    print(f"       n_edges_zeroed={sp_u_stats['n_edges_zeroed']:,}  "
          f"({sp_u_stats['frac_zeroed']*100:.1f}%)  "
          f"conn_r2={sp_u_stats['conn_r2']:.4f}")
    sp_u_r2, sp_u_pearson = rollout_and_metrics(
        W_sparse_u, W_gt, edge_index, b, tau, stim_test, v0_test, v_gt, device,
    )
    sp_u_stats.update({"rollout_pearson_r": sp_u_pearson,
                        "rollout_conn_r2": sp_u_r2})
    print(f"       rollout Pearson r={sp_u_pearson:.4f}  "
          f"conn_r2_rollout={sp_u_r2:.4f}")

    print(f"  [7d] Calibrated sparse W (per-(type, instance)) ...")
    W_calib_u, cal_u_stats = sparse_calibrated(
        W_gt, unit_groups, edge_index, voltage,
    )
    print(f"       n_edges_zeroed={cal_u_stats['n_edges_zeroed']:,}  "
          f"({cal_u_stats['frac_zeroed']*100:.1f}%)  "
          f"conn_r2={cal_u_stats['conn_r2']:.4f}")
    cal_u_r2, cal_u_pearson = rollout_and_metrics(
        W_calib_u, W_gt, edge_index, b, tau, stim_test, v0_test, v_gt, device,
    )
    cal_u_stats.update({"rollout_pearson_r": cal_u_pearson,
                         "rollout_conn_r2": cal_u_r2})
    print(f"       rollout Pearson r={cal_u_pearson:.4f}  "
          f"conn_r2_rollout={cal_u_r2:.4f}")

    # --- Three-panel sparsification figure --------------------------------
    # Load the trained decoder (W_out, b_out) so we can decode HD from the
    # variant rollouts in the bottom row.
    W_out_dec, b_out_dec = load_trained_decoder()
    # Pick a fixed OU test trial. y_test = target (cos theta, sin theta).
    y_test_np = np.asarray(zarr.open(
        os.path.join(DATA_DIR, "x_list_test", "voltage.zarr"), mode="r"))[
        :N_ROLLOUT]
    # Use the same test-split stimulus and v0 already loaded above.
    hd_trial = dict(
        u=None,                       # raw u not needed
        y_true=None,                  # filled below
        stim=stim_test.numpy(),       # (T, N) per-neuron projected
        v0=v0_test.numpy(),
        dt=DT,
        W_out=W_out_dec,
        b_out=b_out_dec,
    )
    # Compute y_true from the GT voltage trace via the decoder.
    voltage_test = np.asarray(zarr.open(
        os.path.join(DATA_DIR, "x_list_test", "voltage.zarr"), mode="r"))[
        :N_ROLLOUT]
    r_gt = 1.0 / (1.0 + np.exp(-voltage_test))
    y_gt = r_gt @ W_out_dec.T + b_out_dec[None, :]
    hd_trial["y_true"] = y_gt.astype(np.float32)

    plot_sparsify_figure(
        W_gt=W_gt,
        W_sparse_type=W_sparse,
        W_sparse_unit=W_sparse_u,
        edge_index=edge_index,
        neuron_types=ntype,
        type_names=type_names,
        out_path=os.path.join(OUTPUT_DIR, "fig_sparsify_cx.pdf"),
        pct_type=sp_stats["frac_zeroed"] * 100.0,
        pct_unit=sp_u_stats["frac_zeroed"] * 100.0,
        r_type=sp_pearson,
        r_unit=sp_u_pearson,
        hd_trial=hd_trial, b=b, tau=tau, device=device,
    )

    plot_sparsify_unit_orderings(
        W_gt=W_gt,
        W_sparse_unit=W_sparse_u,
        edge_index=edge_index,
        neuron_types=ntype,
        instances=instances,
        type_names=type_names,
        out_path=os.path.join(OUTPUT_DIR, "fig_sparsify_orderings_cx.pdf"),
    )

    # Baseline W_con (hemibrain template) under the same two orderings.
    W_con_dense = load_W_con()
    plot_wcon_orderings(
        W_con_dense=W_con_dense,
        neuron_types=ntype,
        instances=instances,
        type_names=type_names,
        out_path=os.path.join(OUTPUT_DIR, "fig_wcon_orderings_cx.pdf"),
    )

    # -----------------------------------------------------------------------
    # STEP 6 — Write outputs
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}\nSTEP 6 — Writing outputs\n{'='*60}")
    null_dim_per_type_named = {
        (type_names[t] if t < len(type_names) else f"type_{t}"):
            int(null_dim_per_type.get(t, 0))
        for t in degenerate_types
    }
    type_results_serialised = {
        str(t): {
            "name": type_names[t] if t < len(type_names) else f"type_{t}",
            "null_dim": int(null_dim_per_type.get(t, 0)),
            **type_results[t],
            "by_scale": {
                str(sc): type_results_by_scale[sc][t]
                for sc in PERTURBATION_SCALES
            },
        }
        for t in degenerate_types
    }
    results = {
        "data_dir": DATA_DIR,
        "n_neurons": N,
        "n_edges": E,
        "tau": tau,
        "null_dim": null_dim,
        "pct_edges": pct_edges,
        "global_svd_null_dim": svd_results,
        "n_degenerate_types": len(degenerate_types),
        "n_identifiable_types": len(identifiable_types),
        "n_no_outgoing_types": len(no_outgoing_types),
        "identifiable_type_names": [
            type_names[t] if t < len(type_names) else f"type_{t}"
            for t in identifiable_types
        ],
        "no_outgoing_type_names": [
            type_names[t] if t < len(type_names) else f"type_{t}"
            for t in no_outgoing_types
        ],
        "null_dim_per_type": null_dim_per_type_named,
        "lstsq_recovery": {
            "r2_well_conditioned": r2_well,
            "r2_all_edges": r2_all,
            "slope_well_conditioned": slope,
            "n_null_edges": int(lstsq_out["W_null"].sum()),
            "n_sloppy_edges": int(lstsq_out["W_sloppy"].sum()),
        },
        "type_results": type_results_serialised,
        "sparse_sum_preserving": sp_stats,
        "sparse_calibrated":     cal_stats,
        "unit_partition": {
            "null_dim": unit_null_dim,
            "pct_edges": unit_pct,
            "n_groups": len(unit_groups),
            "null_dim_per_type": {
                (type_names[t] if t < len(type_names) else f"type_{t}"):
                    int(unit_null_dim_by_type[t])
                for t in unit_null_dim_by_type
            },
            "sparse_sum_preserving": sp_u_stats,
            "sparse_calibrated":     cal_u_stats,
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "structural_nullspace_cx.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved to {json_path}")

    tex_path = os.path.join(OUTPUT_DIR, "tab_variants_cx.tex")
    write_tab_variants(type_results_by_scale, type_names, null_dim_per_type,
                        tex_path)

    # Paragraph statistics for the appendix prose
    print(f"\n  --- PARAGRAPH STATISTICS ---")
    print(f"  dim ker(H) = {null_dim:,}  ({pct_edges:.0f}% of {E:,} edges)")
    print(f"  Per-neuron LS: R^2_W (well) = {r2_well:.4f}  "
          f"(all) = {r2_all:.4f}")
    ok = [(t, r) for t, r in type_results.items() if r["ok"]]
    if ok:
        r2_vals = [r["conn_r2"] for _, r in ok]
        pr_vals = [r["pearson_r"] for _, r in ok]
        print(f"  Single-type variants ({len(ok)} types, lambda=1.0):")
        print(f"    R^2_W range:   [{min(r2_vals):.4f}, {max(r2_vals):.4f}]")
        print(f"    Pearson r min: {min(pr_vals):.4f}")
    print(f"  Sparse sum-preserving: R^2_W={sp_stats['conn_r2']:.4f}  "
          f"Pearson r={sp_pearson:.4f}")
    print(f"  Sparse calibrated:     R^2_W={cal_stats['conn_r2']:.4f}  "
          f"Pearson r={cal_pearson:.4f}")
    n_z = sp_stats["n_edges_zeroed"]
    print(f"  Both zero {n_z:,} edges ({n_z / E * 100:.0f}% of {E:,})")


if __name__ == "__main__":
    main()
