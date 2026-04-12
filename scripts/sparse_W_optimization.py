#!/usr/bin/env python3
"""Find the minimally sparse W that still preserves neural dynamics.

===========================================================================
SCIENTIFIC MOTIVATION
===========================================================================
The null-space manifold of the inverse problem contains INFINITELY many Ws
that produce identical (or near-identical) dynamics.  Among all these
equivalent solutions, most are dense — they spread weight over all edges.

A sparsity-inducing penalty collapses the null space to its minimum-support
representative:

    W_sparse = argmin ||W||_0   s.t.   dynamics_loss(W) ≤ ε

The L0 "norm" is NP-hard to minimize directly.  This script provides two
complementary approaches:

  METHOD S: Structural null-space sparsification  (fast, exact)
  --------
  Directly exploits the flyvis null-space structure without any optimization.

  Within each degenerate group — the set of k same-type presynaptic neurons
  projecting to the same target — the k-1 sum-preserving null-space directions
  let us redistribute weight arbitrarily while the GROUP SUM is preserved:

      sum_{j in group} W_ij  is invariant under null-space perturbations

  The minimum-support (maximum-zeros) point is to collapse all group weight
  onto a SINGLE REPRESENTATIVE EDGE (the one with the maximum GT magnitude)
  and zero the remaining k-1 edges.

  Number of edges zeroed = Σ_groups (k - 1) = structural null dimension ≈ 121K

  This is not an approximation: within the exact linear system H_i * w_i = b_i,
  if all neurons in the group have IDENTICAL activity (exact degeneracy), the
  rollout is EXACTLY preserved.  If activity is correlated but not identical
  (approximate degeneracy), the rollout is nearly preserved — validated by ODE.

  METHOD L: Per-neuron LASSO  (general, slower)
  --------
  Exploit the per-neuron separability of the linear system H_i * w_i = b_i.
  For each postsynaptic neuron i:

      min ||w_i||_1  s.t.  ||H_i w_i - b_i||_2 / ||b_i||_2 ≤ δ

  The LASSO regularization path gives the trade-off between sparsity and
  local dynamics accuracy.  Unlike Method S, this can zero edges that are
  not in a same-type group and potentially achieve greater sparsity.

  Uses sklearn.linear_model.lasso_path for efficient path computation.
  Feature normalization is applied to ensure convergence.

RELATIONSHIP TO LITERATURE
---------------------------
  • Draye et al. MPI (sparse attention): GECO constrained optimization for
    sparsification — Method L is the per-neuron analogue.
  • Gao et al. OpenAI (sparse transformers): L0-constrained circuits for
    interpretability — same goal, here for biological connectomes.
  • Basis pursuit / compressed sensing: sparse recovery from underdetermined
    system H_i w_i = b_i via L1 relaxation — exactly Method L.
  • Group LASSO / structured sparsity: Method S exploits the null-space group
    structure for exact, analytic sparsification.

===========================================================================
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

_hpc_root = "/groups/saalfeld/home/allierc/GraphData"
_local_root = os.path.join(REPO_ROOT, "..", "flyvis-gnn")

if os.path.exists(os.path.join(_local_root, "graphs_data", "fly", "flyvis_noise_005")):
    DATA_ROOT = os.path.join(_local_root, "graphs_data")
else:
    DATA_ROOT = os.path.join(_hpc_root, "graphs_data")

SOURCE_DATASET = "fly/flyvis_noise_005"
OUTPUT_DIR = os.path.join(REPO_ROOT, "scripts", "sparse_W_results")

DT = 0.02
N_ROLLOUT = 1000     # timesteps for ODE rollout verification
LASSO_SUBSAMPLE = 64  # take every 64th frame → 1000 frames from 64K

from connectome_gnn.generators.flyvis_ode import FlyVisODE
from connectome_gnn.generators.ode_params import FlyVisODEParams
from connectome_gnn.neuron_state import NeuronState


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ode_params(dataset_path):
    pt_path = os.path.join(dataset_path, "ode_params.pt")
    assert os.path.exists(pt_path), f"ode_params.pt not found at {pt_path}"
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    return FlyVisODEParams(**state), state


def load_neuron_types(dataset_path):
    """Load neuron_type.zarr (one integer per neuron, 0..64)."""
    import zarr
    nt_path = os.path.join(dataset_path, "x_list_train", "neuron_type.zarr")
    assert os.path.exists(nt_path)
    return torch.tensor(np.array(zarr.open_array(nt_path, mode="r")), dtype=torch.long)


# ---------------------------------------------------------------------------
# METHOD S: Structural null-space sparsification
# ---------------------------------------------------------------------------

def build_type_groups(edge_index_np, neuron_type_np, n_neurons, n_edges):
    """Build (dst, src_type) → [edge_indices] groups.

    Each group captures edges from same-type presynaptic neurons to the same
    target — the null-space degenerate groups.

    Returns:
        groups: dict (dst_id, src_type_id) -> np.array of global edge indices
    """
    src, dst = edge_index_np[0], edge_index_np[1]
    groups = defaultdict(list)
    for e in range(n_edges):
        key = (int(dst[e]), int(neuron_type_np[src[e]]))
        groups[key].append(e)

    # Convert to arrays
    groups = {k: np.array(v, dtype=np.int64) for k, v in groups.items()}
    return groups


def structural_sparsification_calibrated(gt_W_np, groups, voltage_np, edge_index_np,
                                          subsample=64, strategy="max_weight"):
    """Structural sparsification with calibrated representative weights.

    Instead of W_rep = sum(W_group), find the optimal scalar W_rep that
    minimises the per-group dynamics residual:

        W_rep* = <h_rep(t), sum_j W_j * h_j(t)> / <h_rep(t), h_rep(t)>

    where h_rep(t) = ReLU(v_rep(t)) is the activation of the chosen
    representative neuron, and the target is the GT group contribution.

    This is a single scalar least-squares problem per group — fast and exact.
    """
    src, dst = edge_index_np[0], edge_index_np[1]
    T = voltage_np.shape[0]

    # Subsampled activation
    step = max(1, T // 1000)  # use ~1000 frames for calibration
    h = np.maximum(voltage_np[::step], 0.0)  # (T', N)

    W_sparse = gt_W_np.copy()
    n_groups_degen = 0
    n_edges_zeroed = 0

    for key, edge_idx in groups.items():
        k = len(edge_idx)
        if k <= 1:
            continue
        # Choose representative edge
        if strategy == "max_weight":
            rel_idx = int(np.argmax(np.abs(gt_W_np[edge_idx])))
        else:
            rel_idx = 0
        rep_edge = edge_idx[rel_idx]
        rep_src  = int(src[rep_edge])

        # GT group contribution over time: target(t) = sum_j W_j * h_j(t)
        target = np.zeros(h.shape[0])
        for e in edge_idx:
            target += gt_W_np[e] * h[:, int(src[e])]

        # Representative activation
        h_rep = h[:, rep_src]
        denom = np.sum(h_rep ** 2)
        if denom < 1e-12:
            # Fallback: use sum
            W_rep = gt_W_np[edge_idx].sum()
        else:
            W_rep = float(np.sum(h_rep * target) / denom)

        # Collapse
        W_sparse[edge_idx] = 0.0
        W_sparse[rep_edge] = W_rep

        n_groups_degen += 1
        n_edges_zeroed += k - 1

    # Connectivity R² vs GT
    ss_res = np.sum((gt_W_np - W_sparse) ** 2)
    ss_tot = np.sum((gt_W_np - gt_W_np.mean()) ** 2)
    conn_r2 = 1.0 - ss_res / ss_tot

    mask_zero = np.abs(W_sparse) < 1e-8
    stats = {
        "method": "structural_calibrated",
        "n_degenerate_groups": n_groups_degen,
        "n_edges_zeroed": int(n_edges_zeroed),
        "n_edges_total": len(gt_W_np),
        "frac_zero": float(mask_zero.mean()),
        "connectivity_r2_vs_gt": float(conn_r2),
    }
    return W_sparse, stats


def structural_sparsification(gt_W_np, groups, strategy="max_weight"):
    """Collapse each degenerate group onto a single representative edge.

    For each (dst, src_type) group of size k:
      - Representative edge: the one with the highest GT |weight|
      - Set representative weight = group weight sum
      - Zero all other k-1 edges

    Args:
        gt_W_np: (E,) ground-truth weights
        groups:  dict of (dst, src_type) -> edge_index array
        strategy: 'max_weight' (pick heaviest) or 'first' (pick first)

    Returns:
        W_sparse:  (E,) sparse weights
        stats:     summary dict
    """
    W_sparse = gt_W_np.copy()
    n_groups_degenerate = 0
    n_edges_zeroed = 0

    for key, edge_idx in groups.items():
        k = len(edge_idx)
        if k <= 1:
            continue  # No degeneracy for singleton groups

        # Pick representative edge
        if strategy == "max_weight":
            rel_idx = int(np.argmax(np.abs(gt_W_np[edge_idx])))
        else:
            rel_idx = 0
        rep_edge = edge_idx[rel_idx]

        # Group sum
        group_sum = gt_W_np[edge_idx].sum()

        # Collapse: set representative = sum, zero others
        W_sparse[edge_idx] = 0.0
        W_sparse[rep_edge] = group_sum

        n_groups_degenerate += 1
        n_edges_zeroed += k - 1

    # Compute connectivity R² vs GT
    ss_res = np.sum((gt_W_np - W_sparse) ** 2)
    ss_tot = np.sum((gt_W_np - gt_W_np.mean()) ** 2)
    conn_r2 = 1.0 - ss_res / ss_tot

    E = len(gt_W_np)
    mask_zero = np.abs(W_sparse) < 1e-8
    stats = {
        "method": "structural_sparsification",
        "n_degenerate_groups": n_groups_degenerate,
        "n_edges_zeroed": int(n_edges_zeroed),
        "n_edges_total": E,
        "frac_zero": float(mask_zero.mean()),
        "connectivity_r2_vs_gt": float(conn_r2),
        "W_nonzero_mean": float(np.abs(W_sparse[~mask_zero]).mean()),
        "W_nonzero_std":  float(np.abs(W_sparse[~mask_zero]).std()),
    }
    return W_sparse, stats


# ---------------------------------------------------------------------------
# METHOD L: Per-neuron LASSO (with regularization path)
# ---------------------------------------------------------------------------

def build_per_neuron_system(dataset_path, ode_params, edge_index_np,
                            subsample=LASSO_SUBSAMPLE):
    """Build H_i * w_i = b_i for each neuron from subsampled voltage.

    H_i[t, j] = ReLU(v_j(t))  for presynaptic j
    b_i[t]    = tau_i * dv_i/dt + v_i - V_rest_i - e_i

    Returns dict: neuron_id -> (H_i, b_i, edge_global_indices)
    """
    import zarr
    v_zarr = zarr.open_array(
        os.path.join(dataset_path, "x_list_train", "voltage.zarr"), mode="r")
    s_zarr = zarr.open_array(
        os.path.join(dataset_path, "x_list_train", "stimulus.zarr"), mode="r")

    voltage  = np.array(v_zarr[::subsample], dtype=np.float32)
    stimulus = np.array(s_zarr[::subsample], dtype=np.float32)
    T, N = voltage.shape
    print(f"  Voltage subsampled: T={T}, N={N}")

    src, dst = edge_index_np[0], edge_index_np[1]
    tau   = ode_params.tau_i.numpy()
    vrest = ode_params.V_i_rest.numpy()

    h_tm1    = np.maximum(voltage[:-1], 0.0)
    dv_dt    = (voltage[1:] - voltage[:-1]) / DT
    v_tm1    = voltage[:-1]
    stim_tm1 = stimulus[:-1]

    in_edges = defaultdict(list)
    for e in range(len(src)):
        in_edges[int(dst[e])].append((int(src[e]), e))

    systems = {}
    for i in range(N):
        if i not in in_edges:
            continue
        partners = in_edges[i]
        pre_idx  = np.array([p[0] for p in partners], dtype=np.int64)
        e_global = np.array([p[1] for p in partners], dtype=np.int64)
        H_i = h_tm1[:, pre_idx].astype(np.float64)
        b_i = (tau[i] * dv_dt[:, i]
               + v_tm1[:, i]
               - vrest[i]
               - stim_tm1[:, i]).astype(np.float64)
        systems[i] = (H_i, b_i, e_global)

    print(f"  Built {len(systems)} per-neuron systems")
    return systems


def lasso_per_neuron(per_neuron_systems, gt_W_np, target_frac_error=0.02):
    """Per-neuron LASSO sparsification via regularization path.

    For each neuron: use sklearn.linear_model.lasso_path to get the full
    path, find the maximum alpha where normalized residual ≤ target.

    Features are normalized per-column before LASSO (essential for convergence)
    and the solution is un-normalized back to original scale.

    Returns:
        W_sparse:    (E,) sparse weights
        neuron_stats: dict neuron_id -> stats
    """
    from sklearn.linear_model import lasso_path
    from sklearn.preprocessing import StandardScaler

    E = len(gt_W_np)
    W_sparse = gt_W_np.copy()
    neuron_stats = {}

    for i, (H_i, b_i, e_global) in tqdm(
            per_neuron_systems.items(), desc="LASSO path per neuron", ncols=100):

        d_i = H_i.shape[1]
        b_norm = np.linalg.norm(b_i)
        if b_norm < 1e-10 or d_i == 0:
            W_sparse[e_global] = 0.0
            continue

        # Feature normalization — critical for LASSO convergence
        col_scales = np.linalg.norm(H_i, axis=0) + 1e-8
        H_norm = H_i / col_scales[None, :]  # (T, d_i)

        # Lasso path: returns (alphas, coef_path, _)
        # coef_path is (d_i, n_alphas) along the regularization path
        try:
            alphas, coefs, _ = lasso_path(
                H_norm, b_i,
                fit_intercept=False,
                max_iter=500,
            )
        except Exception:
            # Fall back to least-squares if lasso_path fails
            w_ls, _, _, _ = np.linalg.lstsq(H_i, b_i, rcond=None)
            W_sparse[e_global] = w_ls
            continue

        # Find maximum alpha (most sparse) where residual ≤ target
        best_w = None
        best_alpha = None
        best_n_zeros = -1

        for k_alpha in range(len(alphas)):
            w_norm = coefs[:, k_alpha]       # (d_i,) in normalized space
            w_orig = w_norm / col_scales     # back to original scale
            residual = np.linalg.norm(H_i @ w_orig - b_i) / b_norm
            n_zeros = int(np.sum(np.abs(w_orig) < 1e-8))

            if residual <= target_frac_error:
                if n_zeros > best_n_zeros:
                    best_w = w_orig.copy()
                    best_alpha = float(alphas[k_alpha])
                    best_n_zeros = n_zeros
                break   # path goes from sparse to dense; first hit is sparsest

        if best_w is None:
            # No alpha met tolerance — use least-squares (alpha→0 limit)
            w_ls, _, _, _ = np.linalg.lstsq(H_i, b_i, rcond=None)
            best_w = w_ls
            best_alpha = 0.0
            best_n_zeros = int(np.sum(np.abs(best_w) < 1e-8))

        W_sparse[e_global] = best_w
        residual_final = float(np.linalg.norm(H_i @ best_w - b_i) / b_norm)
        b_mean = b_i.mean()
        ss_res = np.sum((H_i @ best_w - b_i) ** 2)
        ss_tot = np.sum((b_i - b_mean) ** 2)
        r2_local = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

        neuron_stats[i] = {
            "d_i": d_i,
            "n_zeros": best_n_zeros,
            "n_nonzero": d_i - best_n_zeros,
            "frac_zero": float(best_n_zeros / d_i),
            "alpha": best_alpha,
            "residual": residual_final,
            "r2_local": r2_local,
        }

    print(f"  Processed {len(neuron_stats)} neurons")
    return W_sparse, neuron_stats


# ---------------------------------------------------------------------------
# ODE rollout verification
# ---------------------------------------------------------------------------

def run_rollout_compare(ode_params_sparse_state, ode_params_gt_dev, stim_rollout,
                        neuron_types, v0, n_frames, device):
    """Run GT and sparse ODE rollouts, compute R² and Pearson r over time."""
    N = int(ode_params_gt_dev.tau_i.shape[0])
    T = min(n_frames, stim_rollout.shape[0])
    model_type = "flyvis_A"
    n_types = int(neuron_types.max().item()) + 1

    # Build sparse ode_params on device
    ode_params_sp = FlyVisODEParams(**ode_params_sparse_state).to(device)

    def make_ode(p):
        return FlyVisODE(ode_params=p, g_phi=torch.nn.functional.relu,
                         params=[], model_type=model_type,
                         n_neuron_types=n_types, device=device)

    def make_state(p, v_init):
        return NeuronState(
            index=torch.arange(N, dtype=torch.long, device=device),
            pos=torch.zeros(N, 2, device=device),
            voltage=v_init.clone().to(device),
            stimulus=torch.zeros(N, device=device),
            group_type=torch.zeros(N, dtype=torch.long, device=device),
            neuron_type=neuron_types.to(device),
            calcium=torch.zeros(N, device=device),
            fluorescence=torch.zeros(N, device=device),
            noise=torch.zeros(N, device=device),
        )

    pde_gt = make_ode(ode_params_gt_dev)
    pde_sp = make_ode(ode_params_sp)
    x_gt   = make_state(ode_params_gt_dev, v0)
    x_sp   = make_state(ode_params_sp, v0)

    ei_gt = ode_params_gt_dev.edge_index.to(device)
    ei_sp = ode_params_sp.edge_index.to(device)
    stim_t = stim_rollout[:T].to(device)

    r2_t = np.zeros(T)
    pearson_t = np.zeros(T)

    with torch.no_grad():
        for t in range(T):
            x_gt.stimulus[:] = stim_t[t]
            x_sp.stimulus[:] = stim_t[t]

            vgt = x_gt.voltage.cpu().numpy()
            vsp = x_sp.voltage.cpu().numpy()
            diff = vsp - vgt
            ss_res = np.sum(diff ** 2)
            ss_tot = np.sum((vgt - vgt.mean()) ** 2)
            r2_t[t] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

            vgt_c = vgt - vgt.mean()
            vsp_c = vsp - vsp.mean()
            d = np.sqrt(np.sum(vgt_c**2) * np.sum(vsp_c**2))
            pearson_t[t] = float(np.sum(vgt_c * vsp_c) / d) if d > 0 else 1.0

            x_gt.voltage = x_gt.voltage + DT * pde_gt(x_gt, ei_gt).squeeze()
            x_sp.voltage = x_sp.voltage + DT * pde_sp(x_sp, ei_sp).squeeze()

    return r2_t, pearson_t


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(W_sparse, gt_W, method_name, r2_t, pearson_t,
                 neuron_stats, output_dir):
    """Four-panel figure: scatter, rollout, per-neuron sparsity, weight distribution."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    mask_zero = np.abs(W_sparse) < 1e-8
    frac_zero = mask_zero.mean()
    ss_res = np.sum((gt_W - W_sparse) ** 2)
    ss_tot = np.sum((gt_W - gt_W.mean()) ** 2)
    conn_r2 = 1.0 - ss_res / ss_tot

    # (0,0) Scatter GT vs sparse W
    ax = axes[0, 0]
    rng = np.random.default_rng(0)
    n_plot = min(20000, len(gt_W))
    idx = rng.choice(len(gt_W), size=n_plot, replace=False)
    idx_nz = idx[~mask_zero[idx]]
    idx_z  = idx[ mask_zero[idx]]
    if len(idx_nz) > 0:
        ax.scatter(gt_W[idx_nz], W_sparse[idx_nz], s=1, alpha=0.4,
                   color="steelblue", label="non-zero")
    if len(idx_z) > 0:
        ax.scatter(gt_W[idx_z], W_sparse[idx_z], s=1, alpha=0.2,
                   color="red", label="zeroed")
    lo = gt_W.min(); hi = gt_W.max()
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"Sparse W vs GT  ({method_name})\n"
                 f"R²={conn_r2:.4f}  {frac_zero*100:.1f}% edges zeroed", fontsize=12)
    ax.set_xlabel(r"GT $W_{ij}$"); ax.set_ylabel(r"Sparse $W_{ij}$")
    ax.legend(fontsize=8, markerscale=5)

    # (0,1) Rollout R²(t) and Pearson r(t)
    ax = axes[0, 1]
    ax.plot(r2_t, color="steelblue", linewidth=1.5, label="R²(t)")
    ax.plot(pearson_t, color="tomato", linewidth=1.5, linestyle="--", label="Pearson r(t)")
    ax.axhline(0.99, color="green", linestyle=":", linewidth=1.0, label="target 0.99")
    ax.set_xlabel("Timestep"); ax.set_ylabel("R² / Pearson r vs GT")
    ax.set_title(f"ODE rollout  final R²={r2_t[-1]:.4f}  Pearson={pearson_t[-1]:.4f}",
                 fontsize=12)
    ax.legend(fontsize=9); ax.set_ylim(-0.05, 1.05)

    # (1,0) Per-neuron sparsity (if available)
    ax = axes[1, 0]
    if neuron_stats:
        fz = [s["frac_zero"] for s in neuron_stats.values()]
        d_i = [s["d_i"] for s in neuron_stats.values()]
        ax.scatter(d_i, fz, s=3, alpha=0.5, color="purple")
        ax.set_xlabel("In-degree $d_i$"); ax.set_ylabel("Fraction zeroed")
        ax.set_title("Per-neuron sparsity vs in-degree", fontsize=12)
    else:
        # For structural sparsification: show group sizes
        ax.text(0.5, 0.5, "Structural method:\nno per-neuron stats",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)

    # (1,1) Weight distributions
    ax = axes[1, 1]
    nz = W_sparse[~mask_zero]
    if len(nz) > 0:
        ax.hist(nz, bins=100, color="steelblue", alpha=0.7, label="sparse W (non-zero)")
    ax.hist(gt_W, bins=100, color="tomato", alpha=0.4, label="GT W")
    ax.set_xlabel("Weight value"); ax.set_ylabel("Count")
    ax.set_title("Weight distributions", fontsize=12)
    ax.legend(fontsize=9)

    plt.suptitle(f"Minimal Sparse W — {method_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"sparse_W_{method_name}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  Saved {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="structural",
                        choices=["structural", "lasso", "both"],
                        help="Sparsification method: structural (fast), lasso (slow), or both")
    parser.add_argument("--lasso_tolerance", type=float, default=0.02,
                        help="LASSO: max fractional residual per neuron (default 0.02)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Minimal Sparse W Optimization")
    print(f"{'='*70}")
    print(f"Device:  {device}")
    print(f"Method:  {args.method}")
    print(f"Data:    {os.path.join(DATA_ROOT, SOURCE_DATASET)}")
    print(f"Output:  {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset_path = os.path.join(DATA_ROOT, SOURCE_DATASET)

    # ------------------------------------------------------------------
    # 1. Load GT params and neuron types
    # ------------------------------------------------------------------
    print(f"\n[1] Loading GT ODE params ...")
    ode_params_gt, raw_state = load_ode_params(dataset_path)
    ode_params_gt_dev = ode_params_gt.to(device)

    gt_W_np = raw_state["W"].numpy().copy()
    edge_index_np = raw_state["edge_index"].numpy()
    N = int(ode_params_gt.tau_i.shape[0])
    E = len(gt_W_np)
    print(f"  N={N} neurons,  E={E} edges")
    print(f"  GT W: mean={gt_W_np.mean():.4f}  std={gt_W_np.std():.4f}  "
          f"non-zero={int((np.abs(gt_W_np) > 1e-8).sum()):,}")

    neuron_types = load_neuron_types(dataset_path)
    neuron_type_np = neuron_types.numpy()

    # ------------------------------------------------------------------
    # 2. Build type groups
    # ------------------------------------------------------------------
    print(f"\n[2] Building (dst, src_type) degenerate groups ...")
    t0 = time.time()
    groups = build_type_groups(edge_index_np, neuron_type_np, N, E)
    n_degen_groups = sum(1 for v in groups.values() if len(v) > 1)
    n_singleton    = sum(1 for v in groups.values() if len(v) == 1)
    structural_null_dim = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    print(f"  Total groups: {len(groups):,}")
    print(f"  Degenerate groups (k>1): {n_degen_groups:,}")
    print(f"  Singleton groups:        {n_singleton:,}")
    print(f"  Structural null dim:     {structural_null_dim:,}  "
          f"({structural_null_dim/E*100:.1f}% of edges)")
    print(f"  Done in {time.time()-t0:.2f}s")

    # ------------------------------------------------------------------
    # 3. Load rollout stimulus / initial state (shared by all methods)
    # ------------------------------------------------------------------
    print(f"\n[3] Loading rollout data ...")
    import zarr
    stim_zarr = zarr.open_array(
        os.path.join(dataset_path, "x_list_train", "stimulus.zarr"), mode="r")
    stim_rollout = torch.tensor(
        np.array(stim_zarr[:N_ROLLOUT], dtype=np.float32))
    v_zarr = zarr.open_array(
        os.path.join(dataset_path, "x_list_train", "voltage.zarr"), mode="r")
    v0 = torch.tensor(np.array(v_zarr[0], dtype=np.float32))
    print(f"  Stimulus: {stim_rollout.shape},  v0: {v0.shape}")

    all_summaries = {}

    # ------------------------------------------------------------------
    # METHOD S: Structural sparsification
    # ------------------------------------------------------------------
    if args.method in ("structural", "both"):
        print(f"\n{'='*60}")
        print(f"METHOD S: Structural null-space sparsification")
        print(f"{'='*60}")

        # Load voltage for calibration (subsampled)
        import zarr as _zarr
        v_zarr_c = _zarr.open_array(
            os.path.join(dataset_path, "x_list_train", "voltage.zarr"), mode="r")
        # Use first 8000 frames subsampled by 8 for calibration
        voltage_calib = np.array(v_zarr_c[::8], dtype=np.float32)
        print(f"  Calibration voltage: {voltage_calib.shape}")

        t0 = time.time()
        W_s, stats_s = structural_sparsification(gt_W_np, groups)
        print(f"  Sum-preserving done in {time.time()-t0:.2f}s")
        print(f"  Edges zeroed:    {stats_s['n_edges_zeroed']:,} / {E:,}  "
              f"({stats_s['frac_zero']*100:.1f}%)")
        print(f"  Connectivity R²: {stats_s['connectivity_r2_vs_gt']:.4f}")

        # ODE rollout
        print(f"  Running ODE rollout ({N_ROLLOUT} frames) ...")
        sparse_state_s = dict(raw_state)
        sparse_state_s["W"] = torch.tensor(W_s, dtype=torch.float32)
        t0 = time.time()
        r2_t_s, pearson_t_s = run_rollout_compare(
            sparse_state_s, ode_params_gt_dev,
            stim_rollout, neuron_types, v0, N_ROLLOUT, device)
        print(f"  Rollout done in {time.time()-t0:.1f}s")
        print(f"  Rollout R² (final):     {r2_t_s[-1]:.6f}")
        print(f"  Rollout Pearson (final): {pearson_t_s[-1]:.6f}")
        met_s = bool(r2_t_s[-1] >= 0.99)
        print(f"  Target (r²≥0.99): {'✓ MET' if met_s else '✗ NOT MET'}")

        # Save
        torch.save(torch.tensor(W_s, dtype=torch.float32),
                   os.path.join(OUTPUT_DIR, "W_sparse_structural.pt"))
        torch.save(sparse_state_s,
                   os.path.join(OUTPUT_DIR, "ode_params_sparse_structural.pt"))
        np.save(os.path.join(OUTPUT_DIR, "rollout_r2_structural.npy"), r2_t_s)
        np.save(os.path.join(OUTPUT_DIR, "rollout_pearson_structural.npy"), pearson_t_s)

        plot_results(W_s, gt_W_np, "structural", r2_t_s, pearson_t_s,
                     neuron_stats=None, output_dir=OUTPUT_DIR)

        stats_s.update({
            "rollout_r2_final": float(r2_t_s[-1]),
            "rollout_pearson_final": float(pearson_t_s[-1]),
            "rollout_r2_min": float(r2_t_s.min()),
            "target_met": met_s,
        })
        all_summaries["structural"] = stats_s

        # --- Calibrated variant ---
        print(f"\n  Calibrated variant (optimal W_rep per group) ...")
        t0 = time.time()
        W_sc, stats_sc = structural_sparsification_calibrated(
            gt_W_np, groups, voltage_calib, edge_index_np)
        print(f"  Calibration done in {time.time()-t0:.1f}s")
        print(f"  Edges zeroed:    {stats_sc['n_edges_zeroed']:,} / {E:,}  "
              f"({stats_sc['frac_zero']*100:.1f}%)")
        print(f"  Connectivity R²: {stats_sc['connectivity_r2_vs_gt']:.4f}")

        sparse_state_sc = dict(raw_state)
        sparse_state_sc["W"] = torch.tensor(W_sc, dtype=torch.float32)
        print(f"  Running calibrated ODE rollout ...")
        r2_t_sc, pearson_t_sc = run_rollout_compare(
            sparse_state_sc, ode_params_gt_dev,
            stim_rollout, neuron_types, v0, N_ROLLOUT, device)
        print(f"  Rollout R² (final):     {r2_t_sc[-1]:.6f}")
        print(f"  Rollout Pearson (final): {pearson_t_sc[-1]:.6f}")
        met_sc = bool(r2_t_sc[-1] >= 0.99)
        print(f"  Target (r²≥0.99): {'✓ MET' if met_sc else '✗ NOT MET'}")

        torch.save(torch.tensor(W_sc, dtype=torch.float32),
                   os.path.join(OUTPUT_DIR, "W_sparse_structural_calibrated.pt"))
        np.save(os.path.join(OUTPUT_DIR, "rollout_r2_structural_calibrated.npy"), r2_t_sc)
        np.save(os.path.join(OUTPUT_DIR, "rollout_pearson_structural_calibrated.npy"), pearson_t_sc)
        plot_results(W_sc, gt_W_np, "structural_calibrated", r2_t_sc, pearson_t_sc,
                     neuron_stats=None, output_dir=OUTPUT_DIR)

        stats_sc.update({
            "rollout_r2_final": float(r2_t_sc[-1]),
            "rollout_pearson_final": float(pearson_t_sc[-1]),
            "rollout_r2_min": float(r2_t_sc.min()),
            "target_met": met_sc,
        })
        all_summaries["structural_calibrated"] = stats_sc

    # ------------------------------------------------------------------
    # METHOD L: Per-neuron LASSO
    # ------------------------------------------------------------------
    if args.method in ("lasso", "both"):
        print(f"\n{'='*60}")
        print(f"METHOD L: Per-neuron LASSO (tol={args.lasso_tolerance:.1%})")
        print(f"{'='*60}")

        print(f"  Building per-neuron systems (subsample={LASSO_SUBSAMPLE}) ...")
        t0 = time.time()
        per_neuron = build_per_neuron_system(
            dataset_path, ode_params_gt, edge_index_np,
            subsample=LASSO_SUBSAMPLE)
        print(f"  System build done in {time.time()-t0:.1f}s")

        print(f"  Running per-neuron LASSO ...")
        t0 = time.time()
        W_l, nstats_l = lasso_per_neuron(per_neuron, gt_W_np,
                                          target_frac_error=args.lasso_tolerance)
        print(f"  LASSO done in {time.time()-t0:.1f}s")

        mask_z_l = np.abs(W_l) < 1e-8
        fz_l = mask_z_l.mean()
        ss_r = np.sum((gt_W_np - W_l) ** 2)
        ss_t = np.sum((gt_W_np - gt_W_np.mean()) ** 2)
        cr_l = 1.0 - ss_r / ss_t
        print(f"  Edges zeroed:    {mask_z_l.sum():,} / {E:,}  ({fz_l*100:.1f}%)")
        print(f"  Connectivity R²: {cr_l:.4f}")

        # ODE rollout
        print(f"  Running ODE rollout ({N_ROLLOUT} frames) ...")
        sparse_state_l = dict(raw_state)
        sparse_state_l["W"] = torch.tensor(W_l, dtype=torch.float32)
        t0 = time.time()
        r2_t_l, pearson_t_l = run_rollout_compare(
            sparse_state_l, ode_params_gt_dev,
            stim_rollout, neuron_types, v0, N_ROLLOUT, device)
        print(f"  Rollout done in {time.time()-t0:.1f}s")
        print(f"  Rollout R² (final):     {r2_t_l[-1]:.6f}")
        print(f"  Rollout Pearson (final): {pearson_t_l[-1]:.6f}")
        met_l = bool(r2_t_l[-1] >= 0.99)
        print(f"  Target (r²≥0.99): {'✓ MET' if met_l else '✗ NOT MET'}")

        # Save
        torch.save(torch.tensor(W_l, dtype=torch.float32),
                   os.path.join(OUTPUT_DIR, "W_sparse_lasso.pt"))
        torch.save(sparse_state_l,
                   os.path.join(OUTPUT_DIR, "ode_params_sparse_lasso.pt"))
        np.save(os.path.join(OUTPUT_DIR, "rollout_r2_lasso.npy"), r2_t_l)
        np.save(os.path.join(OUTPUT_DIR, "rollout_pearson_lasso.npy"), pearson_t_l)

        json_path = os.path.join(OUTPUT_DIR, "neuron_stats_lasso.json")
        with open(json_path, "w") as f:
            json.dump({str(k): v for k, v in nstats_l.items()}, f, indent=2)

        plot_results(W_l, gt_W_np, "lasso", r2_t_l, pearson_t_l,
                     neuron_stats=nstats_l, output_dir=OUTPUT_DIR)

        frac_zeros_all = [s["frac_zero"] for s in nstats_l.values()]
        r2_loc_all     = [s["r2_local"] for s in nstats_l.values()]
        all_summaries["lasso"] = {
            "method": "lasso",
            "n_edges_zeroed": int(mask_z_l.sum()),
            "n_edges_total": E,
            "frac_zero": float(fz_l),
            "connectivity_r2_vs_gt": float(cr_l),
            "rollout_r2_final": float(r2_t_l[-1]),
            "rollout_pearson_final": float(pearson_t_l[-1]),
            "rollout_r2_min": float(r2_t_l.min()),
            "target_met": met_l,
            "per_neuron_mean_frac_zero": float(np.mean(frac_zeros_all)),
            "per_neuron_mean_r2_local": float(np.mean(r2_loc_all)),
            "tolerance": args.lasso_tolerance,
        }

    # ------------------------------------------------------------------
    # Save joint summary
    # ------------------------------------------------------------------
    sum_path = os.path.join(OUTPUT_DIR, "sparse_W_summary.json")
    with open(sum_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n  Summary saved: {sum_path}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SUMMARY — Minimal Sparse W")
    print(f"{'='*70}")
    print(f"  GT edges: {E:,}   structural null dim: {structural_null_dim:,}  "
          f"({structural_null_dim/E*100:.1f}%)")
    for method, s in all_summaries.items():
        print(f"\n  [{method.upper()}]")
        print(f"    Edges zeroed:    {s['n_edges_zeroed']:,} / {E:,}  "
              f"({s['frac_zero']*100:.1f}%)")
        print(f"    Connectivity R²: {s['connectivity_r2_vs_gt']:.4f}")
        print(f"    Rollout R²:      {s['rollout_r2_final']:.6f}")
        print(f"    Rollout Pearson: {s['rollout_pearson_final']:.6f}")
        print(f"    Target met:      {'YES' if s['target_met'] else 'NO'}")
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
