#!/usr/bin/env python
"""Rollout verification for degenerate connectivity matrices.

===========================================================================
PURPOSE
===========================================================================
The generate_degenerate_W.py script creates connectivity matrices W that
differ from ground truth but are *predicted* to produce identical dynamics
(perturbations lie in the null space of the per-neuron activity matrix H_i).

This prediction rests on one assumption: presynaptic neurons of the same
cell type projecting to the same target have IDENTICAL activity.  In reality,
flyvis neurons of the same type in different hex columns receive different
spatial input — their activity is correlated but not perfectly identical.

This script tests the assumption by running the actual ODE forward and
measuring divergence.  Three outcomes are possible:

  1. RMSE stays at machine precision
     → within-type degeneracy is exact; the inverse problem is truly
       ill-posed with dim(null) = 121,100+ free parameters

  2. RMSE grows slowly (linearly or sub-linearly)
     → approximate degeneracy; long observation times can in principle
       disambiguate, but practical training lengths may not suffice

  3. RMSE grows exponentially
     → the system amplifies small perturbations; the inverse problem is
       better-posed than the linear analysis suggests, because nonlinear
       dynamics break the null-space structure over time

PROTOCOL
--------
  1. Load stimulus from existing zarr (first 10,000 frames)
  2. Run ground-truth ODE, save full voltage trajectory to disk
  3. For each degenerate variant: run ODE with same stimulus & v(0),
     compute RMSE(t) and per-type RMSE(t) online (no full trajectory saved)
  4. Plot RMSE(t) curves + per-type breakdown
  5. Save metrics to graphs_data/degenerate_matrix/rollout_results/

===========================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

SOURCE_DATASET = "graphs_data/fly/flyvis_noise_005"  # For stimulus/voltage zarr files (network structure is identical to noise_free)
DEGENERATE_ROOT = "graphs_data/degenerate_matrix"
OUTPUT_DIR = "graphs_data/degenerate_matrix/rollout_results"

N_FRAMES = 1_000        # rollout length
DT = 0.02               # flyvis integration timestep
MODEL_TYPE = "flyvis_A"  # standard graded model (no tanh)

# ---------------------------------------------------------------------------
# Imports from connectome_gnn
# ---------------------------------------------------------------------------
from connectome_gnn.neuron_state import NeuronState
from connectome_gnn.generators.flyvis_ode import FlyVisODE
from connectome_gnn.generators.ode_params import FlyVisODEParams


def load_stimulus_from_zarr(dataset_path, n_frames):
    """Load the first n_frames of stimulus from x_list_train/stimulus.zarr."""
    import zarr
    stim_path = os.path.join(dataset_path, "x_list_train", "stimulus.zarr")
    assert os.path.exists(stim_path), f"stimulus zarr not found at {stim_path}"
    stim_zarr = zarr.open(stim_path, mode="r")
    T_avail = stim_zarr.shape[0]
    T = min(n_frames, T_avail)
    print(f"  Loading stimulus: {stim_path}  shape={stim_zarr.shape}  using first {T} frames")
    stim = stim_zarr[:T]  # (T, N) numpy float32
    return torch.tensor(stim, dtype=torch.float32)


def load_initial_voltage_from_zarr(dataset_path):
    """Load v(0) from x_list_train/voltage.zarr (first frame)."""
    import zarr
    v_path = os.path.join(dataset_path, "x_list_train", "voltage.zarr")
    assert os.path.exists(v_path), f"voltage zarr not found at {v_path}"
    v_zarr = zarr.open(v_path, mode="r")
    v0 = v_zarr[0]  # (N,) first frame
    print(f"  Loading initial voltage: {v_path}  shape={v_zarr.shape}  "
          f"v(0) range=[{v0.min():.4f}, {v0.max():.4f}]")
    return torch.tensor(v0, dtype=torch.float32)


def load_neuron_types_from_zarr(dataset_path):
    """Load neuron_type array from x_list_train/neuron_type.zarr."""
    import zarr
    nt_path = os.path.join(dataset_path, "x_list_train", "neuron_type.zarr")
    assert os.path.exists(nt_path), f"neuron_type zarr not found at {nt_path}"
    nt_zarr = zarr.open(nt_path, mode="r")
    return torch.tensor(np.array(nt_zarr), dtype=torch.long)


def create_ode(ode_params, neuron_types, device):
    """Instantiate FlyVisODE for the standard graded model."""
    n_neuron_types = int(neuron_types.max().item()) + 1
    return FlyVisODE(
        ode_params=ode_params,
        g_phi=torch.nn.functional.relu,
        params=[],
        model_type=MODEL_TYPE,
        n_neuron_types=n_neuron_types,
        device=device,
    )


def create_neuron_state(n_neurons, neuron_types, v0, device):
    """Create NeuronState initialized with v0."""
    return NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=torch.zeros(n_neurons, 2, dtype=torch.float32, device=device),
        voltage=v0.clone().to(device),
        stimulus=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        group_type=torch.zeros(n_neurons, dtype=torch.long, device=device),
        neuron_type=neuron_types.to(device),
        calcium=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        fluorescence=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        noise=torch.zeros(n_neurons, dtype=torch.float32, device=device),
    )


def run_rollout_gt(ode_params, stim_all, neuron_types, v0, device):
    """Run ground-truth rollout.  Returns full voltage trajectory (T, N) on CPU."""
    n_neurons = ode_params.tau_i.shape[0]
    T = stim_all.shape[0]

    pde = create_ode(ode_params, neuron_types, device)
    x = create_neuron_state(n_neurons, neuron_types, v0, device)
    edge_index = ode_params.edge_index.to(device)
    stim_all = stim_all.to(device)

    voltage_history = torch.zeros(T, n_neurons, dtype=torch.float32)  # on CPU

    with torch.no_grad():
        for t in tqdm(range(T), desc="GT rollout", ncols=100):
            x.stimulus[:] = stim_all[t]
            voltage_history[t] = x.voltage.cpu()

            dv = pde(x, edge_index)
            x.voltage = x.voltage + DT * dv.squeeze()

    return voltage_history


def run_rollout_variant(ode_params, stim_all, neuron_types, v0, v_gt, device):
    """Run variant rollout.  Computes RMSE(t) and per-type RMSE(t) online.

    Args:
        v0:  (N,) initial voltage (same for all variants)
        v_gt: (T, N) ground-truth voltage on CPU

    Returns:
        rmse_t:      (T,) global RMSE at each timestep
        per_type_rmse: dict[int, (T,)] per-type RMSE
        pearson_t:   (T,) Pearson r at each timestep
    """
    n_neurons = ode_params.tau_i.shape[0]
    T = stim_all.shape[0]

    pde = create_ode(ode_params, neuron_types, device)
    x = create_neuron_state(n_neurons, neuron_types, v0, device)
    edge_index = ode_params.edge_index.to(device)
    stim_all = stim_all.to(device)

    # Prepare per-type masks
    nt_np = neuron_types.numpy()
    unique_types = np.unique(nt_np)
    type_masks = {int(t): (nt_np == t) for t in unique_types}

    rmse_t = np.zeros(T, dtype=np.float64)
    pearson_t = np.zeros(T, dtype=np.float64)
    r2_t = np.zeros(T, dtype=np.float64)
    per_type_rmse = {int(t): np.zeros(T, dtype=np.float64) for t in unique_types}

    with torch.no_grad():
        for t in range(T):
            x.stimulus[:] = stim_all[t]

            # Compare with GT at this timestep
            v_var = x.voltage.cpu().numpy()
            v_ref = v_gt[t].numpy()
            diff = v_var - v_ref

            rmse_t[t] = np.sqrt(np.mean(diff ** 2))

            # R2: 1 - SS_res / SS_tot (over all neurons at this timestep)
            ss_res = np.sum(diff ** 2)
            ss_tot = np.sum((v_ref - v_ref.mean()) ** 2)
            r2_t[t] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

            # Pearson correlation
            v_var_c = v_var - v_var.mean()
            v_ref_c = v_ref - v_ref.mean()
            denom = np.sqrt(np.sum(v_var_c**2) * np.sum(v_ref_c**2))
            pearson_t[t] = np.sum(v_var_c * v_ref_c) / denom if denom > 0 else 0.0

            # Per-type RMSE
            for ti, mask in type_masks.items():
                per_type_rmse[ti][t] = np.sqrt(np.mean(diff[mask] ** 2))

            # Euler step
            dv = pde(x, edge_index)
            x.voltage = x.voltage + DT * dv.squeeze()

    return rmse_t, per_type_rmse, pearson_t, r2_t


def plot_results(all_results, neuron_types, output_dir):
    """Plot RMSE(t) and R2(t) curves, grouped by perturbed type."""
    os.makedirs(output_dir, exist_ok=True)

    # Group results by perturbed type
    from collections import defaultdict
    by_type = defaultdict(list)  # type_id -> [(scale, result)]
    for name, res in all_results.items():
        meta = res["metadata"]
        t_id = meta.get("perturbed_type", -1)
        scale = meta.get("scale_factor", -1)
        by_type[t_id].append((scale, name, res))

    # --- Fig 1: 2x2 overview ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) RMSE(t) — one line per variant, color by perturbed type
    ax = axes[0, 0]
    cmap = plt.cm.tab20
    type_ids = sorted(by_type.keys())
    for ci, t_id in enumerate(type_ids):
        color = cmap(ci % 20)
        for scale, name, res in by_type[t_id]:
            alpha = 0.3 + 0.7 * (scale / max(s for s, _, _ in by_type[t_id]))
            ax.plot(res["rmse_t"], color=color, alpha=alpha, linewidth=0.4)
    ax.set_xlabel("timestep")
    ax.set_ylabel("RMSE vs ground truth")
    ax.set_title("Voltage divergence (all 780 variants)")
    ax.set_yscale("log")

    # (0,1) R2(t) — same grouping
    ax = axes[0, 1]
    for ci, t_id in enumerate(type_ids):
        color = cmap(ci % 20)
        for scale, name, res in by_type[t_id]:
            alpha = 0.3 + 0.7 * (scale / max(s for s, _, _ in by_type[t_id]))
            ax.plot(res["r2_t"], color=color, alpha=alpha, linewidth=0.4)
    ax.set_xlabel("timestep")
    ax.set_ylabel("R² vs ground truth")
    ax.set_title("Rollout R² (all variants)")

    # (1,0) Final R2 vs connectivity R2 — scatter, one point per variant
    ax = axes[1, 0]
    for ci, t_id in enumerate(type_ids):
        color = cmap(ci % 20)
        conn_r2s = [res["metadata"].get("connectivity_R2_vs_gt", 1.0)
                    for _, _, res in by_type[t_id]]
        rollout_r2s = [res["r2_t"][-1] for _, _, res in by_type[t_id]]
        ax.scatter(conn_r2s, rollout_r2s, c=[color], s=12, alpha=0.7,
                   label=f"type {t_id}" if ci < 15 else None)
    ax.set_xlabel("Connectivity R² (W vs GT)")
    ax.set_ylabel("Rollout R² (v(T) vs GT)")
    ax.set_title("Connectivity R² vs Rollout R²")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=5, ncol=3, loc="lower left")

    # (1,1) Final RMSE vs scale — per type, lines
    ax = axes[1, 1]
    for ci, t_id in enumerate(type_ids):
        color = cmap(ci % 20)
        sorted_entries = sorted(by_type[t_id], key=lambda x: x[0])
        scales = [s for s, _, _ in sorted_entries]
        final_rmse = [res["rmse_t"][-1] for _, _, res in sorted_entries]
        ax.plot(scales, final_rmse, color=color, linewidth=0.8, alpha=0.7,
                label=f"type {t_id}" if ci < 15 else None)
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Final RMSE")
    ax.set_title("Final RMSE vs perturbation scale")
    ax.set_yscale("log")
    ax.legend(fontsize=5, ncol=3, loc="upper left")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "rollout_rmse.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  Saved {fig_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_path = os.path.join(REPO_ROOT, SOURCE_DATASET)
    degenerate_path = os.path.join(REPO_ROOT, DEGENERATE_ROOT)
    output_dir = os.path.join(REPO_ROOT, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load stimulus and neuron types from zarr
    # ------------------------------------------------------------------
    print("\n[1/4] Loading stimulus, initial voltage, and neuron types from zarr...")
    stim_all = load_stimulus_from_zarr(dataset_path, N_FRAMES)
    v0 = load_initial_voltage_from_zarr(dataset_path)
    neuron_types = load_neuron_types_from_zarr(dataset_path)
    T = stim_all.shape[0]
    N = stim_all.shape[1]
    print(f"  T={T}, N={N}, neuron_types: {neuron_types.unique().shape[0]} types")

    # ------------------------------------------------------------------
    # Step 2: Ground-truth rollout — save full trajectory
    # ------------------------------------------------------------------
    print("\n[2/4] Running ground-truth rollout...")
    gt_state = torch.load(
        os.path.join(degenerate_path, "variant_00_ground_truth", "ode_params.pt"),
        map_location="cpu", weights_only=True,
    )
    gt_params = FlyVisODEParams(**gt_state).to(device)

    v_gt = run_rollout_gt(gt_params, stim_all, neuron_types, v0, device)
    # Save GT trajectory
    gt_traj_path = os.path.join(output_dir, "v_gt.pt")
    torch.save(v_gt, gt_traj_path)
    print(f"  Saved GT trajectory: {gt_traj_path}  shape={v_gt.shape}")

    # ------------------------------------------------------------------
    # Step 3: Variant rollouts — compute RMSE(t) online
    # ------------------------------------------------------------------
    print("\n[3/4] Running variant rollouts...")
    variant_dirs = sorted([
        d for d in os.listdir(degenerate_path)
        if (d.startswith("type_") or d.startswith("mixed_types_var_"))
        and os.path.isdir(os.path.join(degenerate_path, d))
        and os.path.exists(os.path.join(degenerate_path, d, "ode_params.pt"))
    ])
    print(f"  Found {len(variant_dirs)} variants (single-type + mixed-type)")

    all_results = {}
    for vi, vdir in enumerate(variant_dirs):
        vpath = os.path.join(degenerate_path, vdir, "ode_params.pt")
        vstate = torch.load(vpath, map_location="cpu", weights_only=True)
        v_params = FlyVisODEParams(**vstate).to(device)

        # Load metadata
        meta_path = os.path.join(degenerate_path, vdir, "metadata.pt")
        meta = torch.load(meta_path, map_location="cpu", weights_only=True) if os.path.exists(meta_path) else {}

        rmse_t, per_type_rmse, pearson_t, r2_t = run_rollout_variant(
            v_params, stim_all, neuron_types, v0, v_gt, device
        )

        all_results[vdir] = {
            "rmse_t": rmse_t,
            "per_type_rmse": per_type_rmse,
            "pearson_t": pearson_t,
            "r2_t": r2_t,
            "metadata": meta,
        }

        # Print progress every 50 variants + always on last
        if (vi + 1) % 50 == 0 or vi == len(variant_dirs) - 1:
            print(f"  [{vi+1}/{len(variant_dirs)}] {vdir}: "
                  f"R2_final={r2_t[-1]:.4f}  RMSE_final={rmse_t[-1]:.2e}")

    # ------------------------------------------------------------------
    # Step 4: Plot and save results
    # ------------------------------------------------------------------
    print("\n[4/4] Plotting results...")
    plot_results(all_results, neuron_types, output_dir)

    # Save numeric results
    summary = {}
    for vdir, res in all_results.items():
        summary[vdir] = {
            "rmse_final": float(res["rmse_t"][-1]),
            "rmse_max": float(res["rmse_t"].max()),
            "rmse_mean": float(res["rmse_t"].mean()),
            "r2_final": float(res["r2_t"][-1]),
            "r2_min": float(res["r2_t"].min()),
            "pearson_final": float(res["pearson_t"][-1]),
            "pearson_min": float(res["pearson_t"].min()),
            "scale_factor": float(res["metadata"].get("scale_factor", -1)),
            "connectivity_R2_vs_gt": float(res["metadata"].get("connectivity_R2_vs_gt", -1)),
        }

    summary_path = os.path.join(output_dir, "rollout_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {summary_path}")

    # Save raw RMSE curves
    rmse_path = os.path.join(output_dir, "rmse_curves.pt")
    torch.save(
        {vdir: torch.tensor(res["rmse_t"]) for vdir, res in all_results.items()},
        rmse_path,
    )
    print(f"  Saved {rmse_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    # Summary table to console and text file
    header = (f"{'Variant':<35} {'Type':>8} {'R2(W)':>7} {'RMSE_final':>11} "
              f"{'R2_final':>9} {'R2_min':>9}")
    sep = "=" * 90
    lines = [sep, header, "-" * 90]
    for vdir, s in summary.items():
        variant_type = "single" if vdir.startswith("type_") else "mixed"
        lines.append(
            f"{vdir:<35} {variant_type:>8} {s['connectivity_R2_vs_gt']:7.4f} "
            f"{s['rmse_final']:11.2e} {s['r2_final']:9.6f} {s['r2_min']:9.6f}"
        )
    lines.append(sep)

    table_str = "\n".join(lines)
    print(f"\n{table_str}")

    # Write full results to text file
    report_path = os.path.join(output_dir, "rollout_report.txt")
    with open(report_path, "w") as f:
        f.write("DEGENERATE CONNECTIVITY ROLLOUT VERIFICATION\n")
        f.write(f"Frames: {N_FRAMES}   dt: {DT}   model: {MODEL_TYPE}\n")
        f.write(f"Source dataset: {SOURCE_DATASET}\n")
        f.write(f"Device: {device}\n\n")
        f.write(table_str + "\n\n")
        f.write("Per-variant details:\n")
        f.write("-" * 60 + "\n")
        for vdir, res in all_results.items():
            r = res["rmse_t"]
            r2 = res["r2_t"]
            p = res["pearson_t"]
            f.write(f"\n{vdir}:\n")
            f.write(f"  scale_factor: {res['metadata'].get('scale_factor', '?')}\n")
            f.write(f"  connectivity_R2: {res['metadata'].get('connectivity_R2_vs_gt', '?')}\n")
            f.write(f"  RMSE:    t=0 {r[0]:.2e}  t={T//4} {r[T//4]:.2e}  "
                    f"t={T//2} {r[T//2]:.2e}  t={T-1} {r[-1]:.2e}\n")
            f.write(f"  R2:      t=0 {r2[0]:.6f}  t={T//4} {r2[T//4]:.6f}  "
                    f"t={T//2} {r2[T//2]:.6f}  t={T-1} {r2[-1]:.6f}\n")
            f.write(f"  Pearson: t=0 {p[0]:.6f}  t={T//4} {p[T//4]:.6f}  "
                    f"t={T//2} {p[T//2]:.6f}  t={T-1} {p[-1]:.6f}\n")
            # Top 5 most divergent neuron types
            pt = res["per_type_rmse"]
            final_by_type = {t: vals[-1] for t, vals in pt.items()}
            top5 = sorted(final_by_type, key=final_by_type.get, reverse=True)[:5]
            f.write(f"  Top divergent types: ")
            f.write(", ".join(f"type {t} (RMSE={final_by_type[t]:.2e})" for t in top5))
            f.write("\n")
    print(f"  Saved {report_path}")


if __name__ == "__main__":
    main()
