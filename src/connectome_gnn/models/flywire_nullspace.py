#!/usr/bin/env python
"""
flywire_nullspace.py
=====================
Structural null-space analysis for the hybrid-flywire (full-eye) connectome.

Twin of flyvis_nullspace.py — same six-step pipeline, applied to the
full-eye flywire dataset at full_eye_flywireRF_noise_005_blank50_cv00.

Scientific context
------------------
The flyvis ODE for postsynaptic neuron i reads (Eq. 1 in the paper):

    tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * ReLU(v_j(t)) + e_i(t)

Rearranging and stacking T timesteps gives a per-neuron linear system:

    H_i * w_i = b_i

where:
    H_i  (T x d_i) = presynaptic ReLU-activity matrix (columns = presynaptic neurons)
    w_i  (d_i,)    = edge weights for all presynaptic partners of neuron i
    b_i  (T,)      = tau_i * dv_i/dt + v_i - V_rest_i - e_i(t)

Any delta_w in ker(H_i) can be added to w_i without changing the dynamics.
This script characterises the dominant source of that kernel.

WITHIN-TYPE COLUMNAR DEGENERACY (the mechanism)
------------------------------------------------
The full-eye flywire connectome has 796 retinotopic columns (50,412 neurons).
For each postsynaptic neuron i and each presynaptic cell type alpha with
k_{i,alpha} >= 2 same-type presynaptic partners, those k_{i,alpha} columns in
H_i are nearly linearly dependent (same type = correlated activity across
columns). Any sum-zero weight redistribution within the group:

    delta_W,  sum_{j in group} delta_W_ij = 0

leaves H_i * w_i unchanged => (k-1) free directions per group.

Summing over all (i, alpha) groups:
    dim ker(H) = sum_i sum_{alpha: k_{i,alpha} >= 2} (k_{i,alpha} - 1)

WHAT THE SCRIPT COMPUTES
--------------------------
STEP 1 — Load ground truth and cell type assignments
STEP 2 — Find all degenerate groups; classify the 65 cell types
          (the foundational computation everything else depends on)
STEP 3 — Compute structural null space dimension
STEP 4 — Generate one single-type variant per type (max perturbation scale),
          run 1000-frame ODE rollout
STEP 5 — Sparse W: collapse degenerate groups onto one representative edge,
          two variants (sum-preserving, calibrated); run rollouts
STEP 6 — Write JSON results + LaTeX table files

Outputs (written to figures/):
    structural_nullspace_table_flywire.json  — all numeric results
    tab_lambda_8_flywire.tex                 — LaTeX table
"""

import os
import sys
import json
import numpy as np
import torch
import zarr
from collections import defaultdict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This module lives at src/connectome_gnn/models/. REPO_ROOT is three levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))
# Output directory: repo-root/figures (figures and analytical artifacts live together).
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures")

# Ground-truth ODE params and dataset directory.
# No noise-free flywire counterpart exists, so we use the cv00 ode_params.pt
# for both GT W and stimulus/voltage. The W stored here is the noise-free
# ground truth (noise is applied to trajectories, not to the weights).
DATA_DIR = "/groups/saalfeld/home/allierc/GraphData/graphs_data/fly/full_eye_flywireRF_noise_005_blank50_cv00"
ODE_PARAMS_PATH = os.path.join(DATA_DIR, "ode_params.pt")

# Type-name source of truth: connectome_gnn.metrics.INDEX_TO_NAME (same 65-name
# canonical mapping used by neuron_type.zarr in both flyvis and flywire datasets).

# Single-type variant settings.
# Amplitude of each sum-zero perturbation = SCALE * mean(|W_group|).
# We use the maximum scale to show the worst-case connectivity distortion
# while still checking that rollout dynamics are preserved.
PERTURBATION_SCALE = 8.0

# All computations use this seed for reproducibility.
SEED = 42

# ODE rollout settings (must match dataset generation).
N_ROLLOUT  = 1_000   # timesteps to simulate
DT         = 0.02    # integration timestep [s]
MODEL_TYPE = "flyvis_A"  # graded-potential model: ReLU activation, no tanh

from connectome_gnn.generators.flyvis_ode  import FlyVisODE
from connectome_gnn.generators.ode_params  import FlyVisODEParams
from connectome_gnn.neuron_state           import NeuronState


# ===========================================================================
# STEP 1 — Data loading
# ===========================================================================

def load_ground_truth():
    """Load ground-truth W, edge_index, tau, V_rest from ode_params.pt."""
    state      = torch.load(ODE_PARAMS_PATH, map_location="cpu", weights_only=True)
    W_gt       = state["W"].numpy().copy()        # (E,) float32
    edge_index = state["edge_index"].numpy()      # (2, E) int64
    tau        = state["tau_i"].numpy()           # (N,)
    V_rest     = state["V_i_rest"].numpy()        # (N,)
    return W_gt, edge_index, tau, V_rest, state


def load_cell_type_names():
    """Return the canonical 65 cell type names indexed by integer type ID."""
    from connectome_gnn.metrics import INDEX_TO_NAME
    return [INDEX_TO_NAME[i] for i in range(len(INDEX_TO_NAME))]


def load_neuron_type_ids():
    """Load the per-neuron cell type assignment from neuron_type.zarr.

    Each of the 50,412 flywire neurons is assigned one type ID in 0..64.
    Used to group presynaptic neurons by cell type when building degenerate
    groups.
    """
    nt_path = os.path.join(DATA_DIR, "x_list_train", "neuron_type.zarr")
    return np.array(zarr.open_array(nt_path, mode="r"), dtype=np.int64)


def load_stimulus():
    """Load the first N_ROLLOUT frames of stimulus for ODE rollout."""
    s_path = os.path.join(DATA_DIR, "x_list_train", "stimulus.zarr")
    z = zarr.open_array(s_path, mode="r")
    return torch.tensor(z[:N_ROLLOUT].astype(np.float32))


def load_initial_voltage():
    """Load v(0): the initial neuron voltages used in every rollout."""
    v_path = os.path.join(DATA_DIR, "x_list_train", "voltage.zarr")
    z = zarr.open_array(v_path, mode="r")
    return torch.tensor(z[0].astype(np.float32))


def load_voltage_for_calibration(n_frames=14_000):
    """Load voltage traces for the calibrated sparse-W computation.

    We subsample to ~1000 frames inside `sparse_calibrated`, so 14k frames is
    plenty. Lowered from the flyvis default of 64k to keep RAM under ~3 GB
    on the flywire connectome (50k neurons × 14k frames × 4 bytes ≈ 2.8 GB).
    """
    v_path = os.path.join(DATA_DIR, "x_list_train", "voltage.zarr")
    z = zarr.open_array(v_path, mode="r")
    T = min(n_frames, z.shape[0])
    return z[:T].astype(np.float32)


# ===========================================================================
# STEP 2 — Degenerate group identification (foundational computation)
# ===========================================================================

def build_degenerate_groups(edge_index, neuron_type, n_edges):
    """Find all degenerate edge groups and classify each cell type.

    DEGENERATE GROUP definition:
        All edges (src -> dst) where src has the same cell type alpha AND
        the same postsynaptic neuron dst.  Key: (dst, src_type).

    If a group has k >= 2 edges:
        - The k-1 redistributive directions are free (in ker(H_dst)).
        - The cell type alpha is marked as "has degeneracy".

    If all groups containing src_type alpha have k = 1:
        - The type is uniquely identifiable from dynamics alone.
        - It is listed as a non-degenerate ("identifiable") type.
    """
    src, dst = edge_index[0], edge_index[1]

    # --- Pass 1: accumulate edges by (postsynaptic neuron, presynaptic type) ---
    raw = defaultdict(list)
    for e in range(n_edges):
        key = (int(dst[e]), int(neuron_type[src[e]]))
        raw[key].append(e)

    # --- Pass 1b: record which type IDs appear as presynaptic at all ---
    src_types_seen = set(int(neuron_type[src[e]]) for e in range(n_edges))

    # --- Pass 2: classify groups and compute null-space dimensions ---
    groups             = {}
    type_has_degeneracy = {}
    null_dim_per_type   = {}

    for key, edge_list in raw.items():
        _, src_type = key
        k = len(edge_list)

        if k >= 2:
            groups[key] = np.array(edge_list, dtype=np.int64)
            type_has_degeneracy[src_type] = True
            null_dim_per_type[src_type] = null_dim_per_type.get(src_type, 0) + (k - 1)
        else:
            if src_type not in type_has_degeneracy:
                type_has_degeneracy[src_type] = False

    return groups, type_has_degeneracy, null_dim_per_type, src_types_seen


def report_type_degeneracy(type_has_degeneracy, null_dim_per_type, type_names,
                           src_types_seen):
    """Print classification table and return sorted type lists."""
    all_seen = sorted(type_has_degeneracy.keys())
    degenerate_types   = [t for t in all_seen if     type_has_degeneracy[t]]
    identifiable_types = [t for t in all_seen if not type_has_degeneracy[t]]

    all_type_ids     = set(range(len(type_names)))
    no_outgoing_types = sorted(all_type_ids - src_types_seen)

    def tname(t):
        return type_names[t] if t < len(type_names) else f"type_{t}"

    print(f"\n{'='*60}")
    print(f"STEP 2 — CELL TYPE DEGENERACY CLASSIFICATION")
    print(f"{'='*60}")
    print(f"  Total cell types (flywire):         {len(type_names)}")
    print(f"  Types appearing as presynaptic:     {len(all_seen)}")
    print(f"  Types WITH degenerate groups:       {len(degenerate_types)}")
    print(f"  Types WITHOUT degenerate groups:    {len(identifiable_types)}")
    print(f"  Types with NO outgoing edges:       {len(no_outgoing_types)}"
          f"  ({', '.join(tname(t) for t in no_outgoing_types)})")
    assert len(degenerate_types) + len(identifiable_types) + len(no_outgoing_types) == len(type_names), (
        f"Category counts do not sum to {len(type_names)}: "
        f"{len(degenerate_types)} + {len(identifiable_types)} + {len(no_outgoing_types)}"
    )

    print(f"\n  Cell types WITHOUT degenerate groups (identifiable from dynamics):")
    for t in identifiable_types:
        print(f"    type {t:2d}: {tname(t)}")

    print(f"\n  Cell types WITH degenerate groups (null_dim contribution, descending):")
    print(f"    {'Type':>4}  {'Name':<12}  {'null_dim':>10}")
    for t in sorted(degenerate_types, key=lambda i: -null_dim_per_type.get(i, 0)):
        nd = null_dim_per_type.get(t, 0)
        print(f"    {t:4d}  {tname(t):<12}  {nd:10d}")

    return degenerate_types, identifiable_types, no_outgoing_types


# ===========================================================================
# STEP 3 — Structural null space dimension
# ===========================================================================

def compute_null_dim(groups, n_edges):
    """Compute dim ker(H) = sum_groups (k - 1) over all degenerate groups."""
    null_dim = sum(len(idx) - 1 for idx in groups.values())
    pct      = 100.0 * null_dim / n_edges
    return null_dim, pct


# ===========================================================================
# STEP 4 — ODE rollout utilities
# ===========================================================================

def _make_ode_and_state(W_np, state, neuron_types, v0, device):
    """Build FlyVisODE + NeuronState for a given weight matrix W_np."""
    s = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in state.items()}
    s["W"] = torch.tensor(W_np, dtype=torch.float32)
    params = FlyVisODEParams(**s).to(device)

    n_neurons = params.tau_i.shape[0]
    n_types   = int(neuron_types.max().item()) + 1
    ode = FlyVisODE(
        ode_params    = params,
        g_phi         = torch.nn.functional.relu,
        params        = [],
        model_type    = MODEL_TYPE,
        n_neuron_types= n_types,
        device        = device,
    )

    ns = NeuronState(
        index      = torch.arange(n_neurons, dtype=torch.long,    device=device),
        pos        = torch.zeros(n_neurons, 2, dtype=torch.float32, device=device),
        voltage    = v0.clone().to(device),
        stimulus   = torch.zeros(n_neurons, dtype=torch.float32, device=device),
        group_type = torch.zeros(n_neurons, dtype=torch.long,    device=device),
        neuron_type= neuron_types.to(device),
        calcium    = torch.zeros(n_neurons, dtype=torch.float32, device=device),
        fluorescence=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        noise      = torch.zeros(n_neurons, dtype=torch.float32, device=device),
    )
    return ode, ns, params.edge_index.to(device)


def run_gt_rollout(state, neuron_types, stim, v0, device):
    """Run the ground-truth ODE for N_ROLLOUT frames."""
    W_gt  = state["W"].numpy()
    ode, x, ei = _make_ode_and_state(W_gt, state, neuron_types, v0, device)
    stim_dev = stim.to(device)

    v_gt = torch.zeros(N_ROLLOUT, x.voltage.shape[0], dtype=torch.float32)
    with torch.no_grad():
        for t in tqdm(range(N_ROLLOUT), desc="GT rollout", ncols=80):
            x.stimulus[:] = stim_dev[t]
            v_gt[t] = x.voltage.cpu()
            x.voltage = x.voltage + DT * ode(x, ei).squeeze()
    return v_gt


def rollout_and_metrics(W_variant, state, neuron_types, stim, v0, v_gt, device):
    """Run a variant ODE rollout; return connectivity R² and Pearson r."""
    W_gt = state["W"].numpy()

    # Connectivity R² over all E edge weights
    diff    = W_variant - W_gt
    ss_res  = float(np.sum(diff ** 2))
    ss_tot  = float(np.sum((W_gt - W_gt.mean()) ** 2))
    conn_r2 = 1.0 - ss_res / ss_tot

    # ODE rollout with W_variant; compute per-frame Pearson r on the fly.
    # Off-by-one fix: read x.voltage BEFORE stepping so that variant[t] and
    # v_gt[t] refer to the same time point (matches run_gt_rollout's order).
    ode, x, ei = _make_ode_and_state(W_variant, state, neuron_types, v0, device)
    stim_dev = stim.to(device)
    pearson_t = np.zeros(N_ROLLOUT, dtype=np.float64)
    with torch.no_grad():
        for t in range(N_ROLLOUT):
            x.stimulus[:] = stim_dev[t]

            v_var = x.voltage.cpu().numpy()
            v_ref = v_gt[t].numpy()
            vc    = v_var - v_var.mean()
            rc    = v_ref - v_ref.mean()
            denom = np.sqrt(np.sum(vc ** 2) * np.sum(rc ** 2))
            pearson_t[t] = float(np.sum(vc * rc) / denom) if denom > 1e-12 else 0.0

            x.voltage = x.voltage + DT * ode(x, ei).squeeze()

    return conn_r2, float(np.mean(pearson_t))


# ===========================================================================
# STEP 4 (continued) — Single-type variants
# ===========================================================================

def sum_zero_vector(k, rng):
    """Sample a random unit vector of length k that sums to zero."""
    v    = rng.randn(k).astype(np.float64)
    v   -= v.mean()
    norm = np.linalg.norm(v)
    return (v / norm) if norm > 1e-12 else v


def make_single_type_variant(W_gt, groups, type_id, scale, rng):
    """Perturb ground-truth W along the null space of one cell type."""
    W_var = W_gt.copy()
    for (dst_n, src_t), edge_idx in groups.items():
        if src_t != type_id:
            continue
        delta_unit = sum_zero_vector(len(edge_idx), rng)
        amplitude  = scale * float(np.mean(np.abs(W_gt[edge_idx])))
        W_var[edge_idx] += amplitude * delta_unit
    return W_var


# ===========================================================================
# STEP 5 — Sparse W variants
# ===========================================================================

def sparse_sum_preserving(W_gt, groups):
    """Collapse each degenerate group onto one representative edge.

    For each degenerate group (k edges):
        1. Choose the REPRESENTATIVE edge = the one with max |W_gt|.
        2. Set its weight to the GROUP SUM: W_rep = sum_j W_j.
        3. Zero all other k-1 edges.
    """
    W_sparse = W_gt.copy()
    n_zeroed = 0

    for (dst_n, src_t), edge_idx in groups.items():
        k         = len(edge_idx)
        rep_pos   = int(np.argmax(np.abs(W_gt[edge_idx])))
        rep_e     = edge_idx[rep_pos]
        group_sum = float(W_gt[edge_idx].sum())

        W_sparse[edge_idx] = 0.0
        W_sparse[rep_e]    = group_sum
        n_zeroed += k - 1

    ss_res  = float(np.sum((W_gt - W_sparse) ** 2))
    ss_tot  = float(np.sum((W_gt - W_gt.mean()) ** 2))
    conn_r2 = 1.0 - ss_res / ss_tot

    return W_sparse, {
        "method":         "sum_preserving",
        "n_edges_zeroed": n_zeroed,
        "n_edges_total":  len(W_gt),
        "frac_zeroed":    n_zeroed / len(W_gt),
        "conn_r2":        conn_r2,
    }


def sparse_calibrated(W_gt, groups, edge_index, voltage_np):
    """Collapse each group onto one edge with a CALIBRATED (OLS) weight.

    Same collapse structure as sum_preserving, but the representative weight
    is fitted to reproduce the group's contribution to postsynaptic voltage:
        W_rep = <h_rep, target> / <h_rep, h_rep>            [closed-form OLS]
    """
    src_np = edge_index[0]

    step = max(1, voltage_np.shape[0] // 1000)
    h = np.maximum(voltage_np[::step], 0.0)

    W_calib  = W_gt.copy()
    n_zeroed = 0

    for (dst_n, src_t), edge_idx in groups.items():
        k = len(edge_idx)

        rep_pos = int(np.argmax(np.abs(W_gt[edge_idx])))
        rep_e   = edge_idx[rep_pos]
        rep_src = int(src_np[rep_e])

        target = np.zeros(h.shape[0], dtype=np.float64)
        for e in edge_idx:
            target += float(W_gt[e]) * h[:, int(src_np[e])].astype(np.float64)

        h_rep  = h[:, rep_src].astype(np.float64)
        denom  = float(np.sum(h_rep ** 2))
        if denom < 1e-12:
            W_rep = float(W_gt[edge_idx].sum())
        else:
            W_rep = float(np.sum(h_rep * target) / denom)

        W_calib[edge_idx] = 0.0
        W_calib[rep_e]    = W_rep
        n_zeroed += k - 1

    ss_res  = float(np.sum((W_gt - W_calib) ** 2))
    ss_tot  = float(np.sum((W_gt - W_gt.mean()) ** 2))
    conn_r2 = 1.0 - ss_res / ss_tot

    return W_calib, {
        "method":         "calibrated",
        "n_edges_zeroed": n_zeroed,
        "n_edges_total":  len(W_gt),
        "frac_zeroed":    n_zeroed / len(W_gt),
        "conn_r2":        conn_r2,
    }


# ===========================================================================
# STEP 6 — LaTeX output
# ===========================================================================

def write_tab_lambda_8(type_results, type_names, out_path):
    """Write the tab:lambda_8 LaTeX table.

    The table shows, for each cell type with degenerate groups, the
    connectivity R² and rollout Pearson r achieved by the max-scale
    single-type variant.

    Layout: 3-column triplet (cell type | R²_W | Pearson r) to fit the page.
    Types with broken rollouts (diverged ODE) are omitted.
    """
    rows = sorted(
        [(t, r["conn_r2"], r["pearson_r"]) for t, r in type_results.items()
         if r.get("ok", True) and r.get("pearson_r") is not None and r["pearson_r"] > 0.5],
        key=lambda x: x[0]
    )

    def fmt_name(t):
        return type_names[t] if t < len(type_names) else f"type {t}"

    lines = [
        r"\begin{tabular}{lrr@{\quad}lrr@{\quad}lrr}",
        r"\toprule",
        (r"Cell Type & $R^2_{\mathbf{W}}$ & rollout $r$ &"
         r"Cell Type & $R^2_{\mathbf{W}}$ & rollout $r$ & "
         r"Cell Type & $R^2_{\mathbf{W}}$ & rollout $r$ \\"),
        r"\midrule",
    ]

    for i in range(0, len(rows), 3):
        triple = rows[i:i+3]
        parts  = []
        for t, r2, pr in triple:
            parts.append(f"{fmt_name(t):<10} & {r2:.2f} & {pr:.2f}")
        while len(parts) < 3:
            parts.append(" & & ")
        lines.append(" & ".join(parts) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]

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

    # -----------------------------------------------------------------------
    # STEP 1 — Load data
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 1 — Loading data (flywire)")
    print(f"{'='*60}")
    print(f"  ODE_PARAMS_PATH: {ODE_PARAMS_PATH}")
    print(f"  DATA_DIR:        {DATA_DIR}")
    W_gt, edge_index, tau, V_rest, state = load_ground_truth()
    type_names  = load_cell_type_names()
    neuron_type = load_neuron_type_ids()
    N           = len(tau)
    E           = len(W_gt)
    print(f"  N={N} neurons,  E={E} edges")
    print(f"  W_gt: mean={W_gt.mean():.4f}  std={W_gt.std():.4f}  "
          f"range=[{W_gt.min():.4f}, {W_gt.max():.4f}]")
    print(f"  Cell types loaded:  {len(type_names)}")
    print(f"  Neuron type range:  {neuron_type.min()} .. {neuron_type.max()}")

    stim = load_stimulus()
    v0   = load_initial_voltage()
    print(f"  Stimulus shape: {stim.shape}  v0 range=[{v0.min():.4f}, {v0.max():.4f}]")

    neuron_types_tensor = torch.tensor(neuron_type, dtype=torch.long)

    # -----------------------------------------------------------------------
    # STEP 2 — Degenerate groups (foundational computation)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 2 — Degenerate group identification")
    print(f"{'='*60}")
    groups, type_has_degeneracy, null_dim_per_type, src_types_seen = build_degenerate_groups(
        edge_index, neuron_type, E
    )
    degenerate_types, identifiable_types, no_outgoing_types = report_type_degeneracy(
        type_has_degeneracy, null_dim_per_type, type_names, src_types_seen
    )

    # -----------------------------------------------------------------------
    # STEP 3 — Structural null space dimension
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 3 — Structural null space dimension")
    print(f"{'='*60}")
    null_dim, pct_edges = compute_null_dim(groups, E)
    print(f"  dim ker(H) = {null_dim:,}  ({pct_edges:.1f}% of {E:,} edges)")
    print(f"  sum_i sum_alpha (k_i_alpha - 1)")

    # -----------------------------------------------------------------------
    # STEP 4 — Single-type variants + rollouts
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 4 — Single-type variants  (scale={PERTURBATION_SCALE})")
    print(f"{'='*60}")
    print(f"  Types to process: {len(degenerate_types)}")
    print(f"  Running GT rollout first ...")
    v_gt = run_gt_rollout(state, neuron_types_tensor, stim, v0, device)

    type_results = {}
    print(f"  Running single-type variant rollouts ...")
    for t in tqdm(degenerate_types, desc="  type variants", ncols=80):
        W_var = make_single_type_variant(W_gt, groups, t, PERTURBATION_SCALE, rng)
        try:
            conn_r2, pearson_r = rollout_and_metrics(
                W_var, state, neuron_types_tensor, stim, v0, v_gt, device
            )
            type_results[t] = {
                "conn_r2":   conn_r2,
                "pearson_r": pearson_r,
                "ok":        True,
            }
        except Exception as ex:
            print(f"\n    WARNING: type {t} ({type_names[t]}) rollout failed: {ex}")
            type_results[t] = {"conn_r2": None, "pearson_r": None, "ok": False}

    print(f"\n  {'Type':>4}  {'Name':<12}  {'R²_W':>6}  {'Pearson r':>9}")
    print(f"  {'-'*40}")
    sorted_types = sorted(
        degenerate_types,
        key=lambda t: (not type_results[t]["ok"],
                       type_results[t]["conn_r2"] if type_results[t]["ok"] else float("inf")),
    )
    for t in sorted_types:
        r = type_results[t]
        name = type_names[t] if t < len(type_names) else f"type_{t}"
        if r["ok"]:
            print(f"  {t:4d}  {name:<12}  {r['conn_r2']:6.4f}  {r['pearson_r']:9.4f}")
        else:
            print(f"  {t:4d}  {name:<12}  {'FAILED':>6}  {'FAILED':>9}")

    # -----------------------------------------------------------------------
    # STEP 5 — Sparse W (sum-preserving and calibrated)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 5 — Sparse W variants")
    print(f"{'='*60}")

    print(f"  [5a] Building sum-preserving sparse W ...")
    W_sparse, sp_stats = sparse_sum_preserving(W_gt, groups)
    print(f"       n_edges_zeroed={sp_stats['n_edges_zeroed']:,}  "
          f"({sp_stats['frac_zeroed']*100:.1f}%)  "
          f"conn_r2={sp_stats['conn_r2']:.4f}")
    print(f"  [5a] Running rollout ...")
    sp_r2, sp_pearson = rollout_and_metrics(
        W_sparse, state, neuron_types_tensor, stim, v0, v_gt, device
    )
    sp_stats.update({"rollout_pearson_r": sp_pearson, "rollout_conn_r2": sp_r2})
    print(f"       rollout Pearson r={sp_pearson:.4f}  conn_r2_rollout={sp_r2:.4f}")

    print(f"  [5b] Loading voltage for calibration ...")
    voltage_np = load_voltage_for_calibration()
    print(f"       voltage shape={voltage_np.shape}")

    print(f"  [5b] Building calibrated sparse W ...")
    W_calib, cal_stats = sparse_calibrated(W_gt, groups, edge_index, voltage_np)
    print(f"       n_edges_zeroed={cal_stats['n_edges_zeroed']:,}  "
          f"({cal_stats['frac_zeroed']*100:.1f}%)  "
          f"conn_r2={cal_stats['conn_r2']:.4f}")
    print(f"  [5b] Running rollout ...")
    cal_r2, cal_pearson = rollout_and_metrics(
        W_calib, state, neuron_types_tensor, stim, v0, v_gt, device
    )
    cal_stats.update({"rollout_pearson_r": cal_pearson, "rollout_conn_r2": cal_r2})
    print(f"       rollout Pearson r={cal_pearson:.4f}  conn_r2_rollout={cal_r2:.4f}")

    # -----------------------------------------------------------------------
    # STEP 6 — Write outputs
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 6 — Writing outputs")
    print(f"{'='*60}")

    null_dim_per_type_named = {
        type_names[t] if t < len(type_names) else f"type_{t}":
            int(null_dim_per_type.get(t, 0))
        for t in degenerate_types
    }
    results = {
        "null_dim": null_dim,
        "pct_edges": pct_edges,
        "n_edges": E,
        "n_neurons": N,
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
        "type_results": {
            str(t): {
                "name": type_names[t] if t < len(type_names) else f"type_{t}",
                "null_dim": int(null_dim_per_type.get(t, 0)),
                **type_results[t],
            }
            for t in degenerate_types
        },
        "sparse_sum_preserving": sp_stats,
        "sparse_calibrated":     cal_stats,
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "structural_nullspace_table_flywire.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved to {json_path}")

    tex_path = os.path.join(OUTPUT_DIR, "tab_lambda_8_flywire.tex")
    write_tab_lambda_8(type_results, type_names, tex_path)

    print(f"\n  --- PARAGRAPH STATISTICS ---")
    print(f"  dim ker(H) = {null_dim:,}  ({pct_edges:.0f}% of {E:,} edges)")
    ok = [(t, r) for t, r in type_results.items() if r["ok"]]
    r2_vals = [r["conn_r2"] for _, r in ok]
    pr_vals = [r["pearson_r"] for _, r in ok]
    print(f"  Single-type variants ({len(ok)} types, scale={PERTURBATION_SCALE}):")
    print(f"    R²_W range:  [{min(r2_vals):.2f}, {max(r2_vals):.2f}]")
    print(f"    Pearson r min: {min(pr_vals):.4f}")
    print(f"  Sparse sum-preserving: R²_W={sp_stats['conn_r2']:.2f}  Pearson r={sp_pearson:.2f}")
    print(f"  Sparse calibrated:     R²_W={cal_stats['conn_r2']:.2f}  Pearson r={cal_pearson:.2f}")
    n_zeroed = sp_stats["n_edges_zeroed"]
    print(f"  Both zero {n_zeroed:,} edges ({n_zeroed/E*100:.0f}% of {E:,})")


if __name__ == "__main__":
    main()
