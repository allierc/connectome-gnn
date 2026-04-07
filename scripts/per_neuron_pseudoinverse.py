#!/usr/bin/env python
"""Per-neuron pseudoinverse weight recovery and connectivity R² analysis.

For each neuron i, we:
1. Extract H_i (activity matrix of presynaptic partners)
2. Compute b_i (right-hand side: tau_i * dv/dt + v - V_rest - stimulus)
3. Use SVD pseudoinverse: w_i = H_i^+ b_i
4. Compare recovered w_i against ground-truth W_i
5. Compute global connectivity R² and per-neuron breakdown

This directly measures how well the linear system H_i w_i = b_i
can recover ground-truth synaptic weights.
"""

import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
import zarr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def load_training_activity(data_dir, subsample_t=8):
    """Load voltage from x_list_train/voltage.zarr."""
    data_path = Path(data_dir) / "x_list_train" / "voltage.zarr"
    voltage_zarr = zarr.open_array(str(data_path), mode='r')
    voltage = np.array(voltage_zarr[::subsample_t, :])
    return voltage


def load_training_derivatives(data_dir, subsample_t=8):
    """Load dv/dt from y_list_train.zarr."""
    data_path = Path(data_dir) / "y_list_train.zarr"
    y_zarr = zarr.open_array(str(data_path), mode='r')
    # Shape is (T, N, 1), extract the derivative
    dv_dt = np.array(y_zarr[::subsample_t, :, 0])
    return dv_dt


def load_stimulus(data_dir, subsample_t=8):
    """Load stimulus from x_list_train/stimulus.zarr."""
    data_path = Path(data_dir) / "x_list_train" / "stimulus.zarr"
    stim_zarr = zarr.open_array(str(data_path), mode='r')
    stim = np.array(stim_zarr[::subsample_t, :])
    return stim


def compute_rhs(v, dv_dt, stim, tau_i, V_rest):
    """Compute right-hand side: b_i(t) = tau_i * dv/dt + v - V_rest - stim.

    Args:
        v: (T, N) voltage traces
        dv_dt: (T, N) derivatives
        stim: (T, N) stimulus
        tau_i: (N,) time constants
        V_rest: (N,) resting potentials

    Returns:
        b: (T, N) right-hand side
    """
    T, N = v.shape
    assert dv_dt.shape == (T, N), f"dv_dt shape {dv_dt.shape} doesn't match v {v.shape}"
    assert stim.shape == (T, N), f"stim shape {stim.shape} doesn't match v {v.shape}"
    assert tau_i.shape == (N,), f"tau_i shape {tau_i.shape} != (N,) = ({N},)"
    assert V_rest.shape == (N,), f"V_rest shape {V_rest.shape} != (N,) = ({N},)"

    # Compute: b_i(t) = tau_i * dv/dt[t, i] + v[t, i] - V_rest[i] - stim[t, i]
    b = dv_dt * tau_i[None, :] + v - V_rest[None, :] - stim

    return b


def recover_weights_pseudoinverse(h, b, edge_index, variance_threshold=1.0):
    """Recover weights w_i = H_i^+ b_i for each neuron using SVD pseudoinverse.

    Args:
        h: (T, N) presynaptic activity matrix (ReLU(v))
        b: (T, N) right-hand side (computed from ODE rearrangement)
        edge_index: (2, E) edge list [src, dst]
        variance_threshold: float in [0, 1], truncation level for SVD regularization

    Returns:
        w_recovered: (E,) recovered weight vector (same indexing as state['W'])
        per_neuron_info: dict with per-neuron results (rank, null_dim, r2, etc.)
    """
    T, N = h.shape
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    E = len(src)

    # Build incoming edge lists per neuron
    in_edges = defaultdict(list)
    edge_indices = defaultdict(list)  # Track which edge indices belong to each neuron
    for e_idx in range(E):
        dst_neuron = int(dst[e_idx])
        in_edges[dst_neuron].append(int(src[e_idx]))
        edge_indices[dst_neuron].append(e_idx)

    w_recovered = np.zeros(E)
    per_neuron_info = {
        "neuron_id": [],
        "in_degree": [],
        "effective_rank": [],
        "null_dim": [],
        "neuron_r2": [],
    }

    print(f"\nRecovering weights via pseudoinverse for {len(in_edges)} postsynaptic neurons...")

    for i in tqdm(sorted(in_edges.keys()), desc="Per-neuron pseudoinverse", ncols=100):
        presynaptic = np.array(in_edges[i])
        edge_idxs = np.array(edge_indices[i])
        d_i = len(presynaptic)

        # Extract activity matrix H_i (T x d_i)
        H_i = h[:, presynaptic]

        # Extract RHS b_i (T,)
        b_i = b[:, i]

        # Compute SVD
        U, sigma, Vt = np.linalg.svd(H_i, full_matrices=False)

        # Find effective rank at variance threshold
        total_var = np.sum(sigma ** 2)
        if total_var > 1e-16:  # Avoid division by zero for inactive neurons
            cumsum_var = np.cumsum(sigma ** 2) / total_var
            r_i = np.searchsorted(cumsum_var, variance_threshold) + 1
            r_i = min(r_i, len(sigma))
        else:
            # Neuron has zero activity, can't determine rank
            r_i = 1  # Fallback to rank 1

        null_dim_i = max(0, d_i - r_i)

        # Compute pseudoinverse (truncated at effective rank)
        U_r = U[:, :r_i]
        sigma_r = sigma[:r_i]
        Vt_r = Vt[:r_i, :]

        # w_i = V_r @ Sigma_r^{-1} @ U_r^T @ b_i
        # Use reciprocal with regularization to avoid division by very small values
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma_inv = np.where(sigma_r > 1e-12, 1.0 / sigma_r, 0.0)
        w_i = Vt_r.T @ (np.diag(sigma_inv) @ (U_r.T @ b_i))

        # Store recovered weights
        w_recovered[edge_idxs] = w_i

        # Store per-neuron info
        per_neuron_info["neuron_id"].append(i)
        per_neuron_info["in_degree"].append(d_i)
        per_neuron_info["effective_rank"].append(r_i)
        per_neuron_info["null_dim"].append(null_dim_i)
        per_neuron_info["neuron_r2"].append(None)  # Will fill after GT comparison

    return w_recovered, per_neuron_info


def compute_connectivity_r2(w_gt, w_recovered):
    """Compute connectivity R² (known_ode formula without linear rescaling).

    Formula: R² = 1 - SS_res / SS_tot
    where SS_res = sum((w_gt - w_recovered)²), SS_tot = sum((w_gt - mean(w_gt))²)

    Verified: w_gt and w_recovered are both shape (E,), no transpose needed.

    Args:
        w_gt: (E,) ground-truth weights
        w_recovered: (E,) recovered weights

    Returns:
        r2: float, connectivity R²
    """
    ss_res = np.sum((w_gt - w_recovered) ** 2)
    ss_tot = np.sum((w_gt - w_gt.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-16))
    return r2


def compute_per_neuron_r2(w_gt, w_recovered, edge_index, per_neuron_info):
    """Compute R² for each neuron's incoming weights.

    Args:
        w_gt: (E,) ground-truth weights
        w_recovered: (E,) recovered weights
        edge_index: (2, E) edge list
        per_neuron_info: dict with neuron indices

    Returns:
        Updated per_neuron_info with neuron_r2 filled in
    """
    dst = edge_index[1].numpy()

    # Group edges by destination neuron
    in_edges = defaultdict(list)
    for e_idx in range(len(w_gt)):
        dst_neuron = int(dst[e_idx])
        in_edges[dst_neuron].append(e_idx)

    neuron_r2_list = []
    for neuron_id in per_neuron_info["neuron_id"]:
        edge_idxs = in_edges[neuron_id]
        w_gt_i = w_gt[edge_idxs]
        w_rec_i = w_recovered[edge_idxs]

        ss_res = np.sum((w_gt_i - w_rec_i) ** 2)
        ss_tot = np.sum((w_gt_i - w_gt_i.mean()) ** 2)
        r2_i = float(1.0 - ss_res / (ss_tot + 1e-16)) if ss_tot > 1e-16 else 0.0
        neuron_r2_list.append(r2_i)

    per_neuron_info["neuron_r2"] = neuron_r2_list
    return per_neuron_info


def analyze_pseudoinverse(noise_label, ode_path, data_dir):
    """Complete analysis: load, recover, and evaluate pseudoinverse solution."""

    if not os.path.exists(ode_path):
        print(f"ERROR: ODE params not found at {ode_path}")
        return None

    print(f"pseudoinverse analysis: {noise_label}")
    print(f"loading ODE params from {ode_path}")

    state = torch.load(ode_path, map_location="cpu", weights_only=True)
    N = len(state['tau_i'])
    E = len(state['W'])
    print(f"  neurons: {N:,d}, edges: {E:,d}")

    # Load voltage
    print(f"loading voltage traces...")
    v = load_training_activity(data_dir, subsample_t=8)

    # Load dv/dt
    print(f"loading derivatives...")
    dv_dt = load_training_derivatives(data_dir, subsample_t=8)

    # Load stimulus
    print(f"loading stimulus...")
    stim = load_stimulus(data_dir, subsample_t=8)

    # Compute activity: h(t) = ReLU(v(t))
    print(f"computing presynaptic activity h = ReLU(v)...")
    h = np.maximum(0, v)
    print(f"  activity matrix h: {h.shape}, sparsity {100*(h==0).mean():.1f}%")

    # Compute RHS: b_i(t) = tau_i * dv/dt + v - V_rest - stim
    print(f"computing right-hand side b_i(t)...")
    tau_i = state['tau_i'].numpy()
    V_rest = state['V_i_rest'].numpy()
    b = compute_rhs(v, dv_dt, stim, tau_i, V_rest)
    print(f"  rhs matrix b: {b.shape}, mean={b.mean():.4f}, std={b.std():.4f}, range=[{b.min():.4f}, {b.max():.4f}]")

    # Recover weights via pseudoinverse
    print(f"recovering weights using pseudoinverse...")
    edge_index = state['edge_index']
    w_recovered, per_neuron_info = recover_weights_pseudoinverse(
        h, b, edge_index, variance_threshold=0.999
    )

    # Ground truth weights
    w_gt = state['W'].numpy()

    # Compute global connectivity R²
    print(f"computing connectivity R²...")
    global_r2 = compute_connectivity_r2(w_gt, w_recovered)
    print(f"  global connectivity R²: {global_r2:.6f}")

    # Compute per-neuron R²
    print(f"computing per-neuron R²...")
    per_neuron_info = compute_per_neuron_r2(w_gt, w_recovered, edge_index, per_neuron_info)
    neuron_r2_array = np.array(per_neuron_info["neuron_r2"])
    print(f"  per-neuron R² (mean ± std): {neuron_r2_array.mean():.6f} ± {neuron_r2_array.std():.6f}")
    print(f"  per-neuron R² (min, max): {neuron_r2_array.min():.6f}, {neuron_r2_array.max():.6f}")

    # Breakdown by in-degree
    print(f"connectivity R² breakdown by in-degree:")
    in_degrees = np.array(per_neuron_info["in_degree"])
    for lo, hi in [(1, 5), (6, 15), (16, 30), (31, 60), (61, 208)]:
        mask = (in_degrees >= lo) & (in_degrees <= hi)
        if mask.sum() > 0:
            r2_subset = neuron_r2_array[mask].mean()
            print(f"  {lo:3d}--{hi:3d}: {mask.sum():4d} neurons, R²={r2_subset:.6f}")

    # Breakdown by null space dimension
    print(f"connectivity R² breakdown by null space dimension:")
    null_dims = np.array(per_neuron_info["null_dim"])
    for threshold in [0, 1, 5, 10, 20]:
        if threshold == 0:
            mask = (null_dims == 0)
            label = "fully identifiable"
        else:
            mask = (null_dims >= threshold)
            label = f"null_dim ≥ {threshold}"
        if mask.sum() > 0:
            r2_subset = neuron_r2_array[mask].mean()
            print(f"  {label:25s}: {mask.sum():4d} neurons, R²={r2_subset:.6f}")

    return {
        "noise_label": noise_label,
        "global_r2": global_r2,
        "per_neuron_r2": neuron_r2_array,
        "per_neuron_info": per_neuron_info,
        "w_gt": w_gt,
        "w_recovered": w_recovered,
        "edge_index": edge_index,
    }


def create_scatter_plot(results, output_dir="./pseudoinverse_plots"):
    """Create scatter plot: GT weights vs recovered weights, colored by null space dim."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"creating visualizations in {output_dir}...")

    w_gt = results["w_gt"]
    w_recovered = results["w_recovered"]
    edge_index = results["edge_index"]
    per_neuron_info = results["per_neuron_info"]

    # Map each edge to its destination neuron's null space dimension
    dst = edge_index[1].numpy()
    null_dim_per_edge = np.zeros(len(w_gt))

    in_edges = defaultdict(list)
    for e_idx in range(len(w_gt)):
        dst_neuron = int(dst[e_idx])
        in_edges[dst_neuron].append(e_idx)

    for i, neuron_id in enumerate(per_neuron_info["neuron_id"]):
        null_dim = per_neuron_info["null_dim"][i]
        for e_idx in in_edges[neuron_id]:
            null_dim_per_edge[e_idx] = null_dim

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(w_gt, w_recovered, c=null_dim_per_edge, s=10, alpha=0.5,
                        cmap='viridis', edgecolors='none')

    # Diagonal line (perfect recovery)
    lim = max(np.abs(w_gt).max(), np.abs(w_recovered).max())
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.3, label='perfect recovery')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Null space dimension', fontweight='bold')

    ax.set_xlabel('Ground truth W', fontweight='bold', fontsize=11)
    ax.set_ylabel('Recovered W (pseudoinverse)', fontweight='bold', fontsize=11)
    ax.set_title(f'Pseudoinverse Weight Recovery: {results["noise_label"]}\nGlobal R² = {results["global_r2"]:.6f}',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_pseudoinverse.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ scatter plot saved to {output_dir}/scatter_pseudoinverse.png")


def create_r2_histogram(results, output_dir="./pseudoinverse_plots"):
    """Create histogram of per-neuron R² values."""
    os.makedirs(output_dir, exist_ok=True)

    neuron_r2 = results["per_neuron_r2"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(neuron_r2, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(neuron_r2.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean = {neuron_r2.mean():.4f}')
    ax.axvline(np.median(neuron_r2), color='green', linestyle='--', linewidth=2, label=f'Median = {np.median(neuron_r2):.4f}')

    ax.set_xlabel('Per-neuron connectivity R²', fontweight='bold', fontsize=11)
    ax.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax.set_title(f'Distribution of Per-Neuron R²: {results["noise_label"]}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_neuron_r2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ histogram saved to {output_dir}/histogram_neuron_r2.png")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Analyze all three noise conditions sequentially
    noise_conditions = {
        "noise-free": {
            "ode_path": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_free/ode_params.pt",
            "data_dir": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_free"
        },
        "noise-005": {
            "ode_path": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_005/ode_params.pt",
            "data_dir": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_005"
        },
        "noise-05": {
            "ode_path": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_05/ode_params.pt",
            "data_dir": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_05"
        },
    }

    results_list = []

    for noise_label, paths in noise_conditions.items():
        ode_path = os.path.join(script_dir, paths["ode_path"])
        data_dir = os.path.join(script_dir, paths["data_dir"])

        result = analyze_pseudoinverse(noise_label, ode_path, data_dir)
        if result is not None:
            results_list.append(result)

            # Generate visualizations
            create_scatter_plot(result)
            create_r2_histogram(result)

    if results_list:
        print(f"pseudoinverse connectivity R² comparison (full pseudoinverse, no truncation):")
        for result in results_list:
            print(f"  {result['noise_label']:12s}: global R² = {result['global_r2']:.6f}, per-neuron R² = {result['per_neuron_r2'].mean():.6f} ± {result['per_neuron_r2'].std():.6f}")

    return results_list


if __name__ == "__main__":
    results = main()
