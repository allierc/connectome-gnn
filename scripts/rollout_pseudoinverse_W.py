#!/usr/bin/env python
"""Rollout verification for pseudoinverse-recovered connectivity matrices.

For each noise condition and each recovery method:
  1. Recover W via per-neuron pseudoinverse (truncated_svd / ridge / cv_ridge)
  2. Build FlyVisODEParams with recovered W, keeping all other params
  3. Run 1000-frame ODE rollout vs ground-truth W
  4. Print final comparison table: method × condition → conn R², rollout R²
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from connectome_gnn.generators.flyvis_ode import FlyVisODE
from connectome_gnn.generators.ode_params import FlyVisODEParams
from connectome_gnn.neuron_state import NeuronState

from per_neuron_pseudoinverse import analyze_pseudoinverse_multimethod, METHODS_DEFAULT
from rollout_degenerate_W import (
    load_stimulus_from_zarr,
    load_initial_voltage_from_zarr,
    load_neuron_types_from_zarr,
    run_rollout_gt,
    run_rollout_variant,
    DT,
    MODEL_TYPE,
)

N_FRAMES  = 1_000
OUTPUT_DIR = os.path.join(REPO_ROOT, "pseudoinverse_plots", "rollout")

NOISE_CONDITIONS = {
    "noise-free": {
        "ode_path": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_free/ode_params.pt",
        "data_dir": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_free",
    },
    "noise-005": {
        "ode_path": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_005/ode_params.pt",
        "data_dir": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_005",
    },
    "noise-05": {
        "ode_path": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_05/ode_params.pt",
        "data_dir": "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_05",
    },
}

METHODS = list(METHODS_DEFAULT)   # ["truncated_svd", "ridge", "cv_ridge"]


def build_variant_params(state, w_recovered, device):
    state_var = {k: v.clone() if isinstance(v, torch.Tensor) else v
                 for k, v in state.items()}
    state_var['W'] = torch.tensor(w_recovered, dtype=torch.float32)
    return FlyVisODEParams(**state_var).to(device)


def run_rollouts_for_condition(results, state, data_dir, gt_params, device):
    """Run rollouts for all methods in a condition; GT rollout computed once."""
    stim   = load_stimulus_from_zarr(data_dir, N_FRAMES)
    v0     = load_initial_voltage_from_zarr(data_dir)
    ntypes = load_neuron_types_from_zarr(data_dir)
    v_gt   = run_rollout_gt(gt_params, stim, ntypes, v0, device)

    out = {}
    for method, res in results.items():
        var_params = build_variant_params(state, res["w_recovered"], device)
        rmse_t, _, _, r2_t = run_rollout_variant(var_params, stim, ntypes, v0, v_gt, device)
        out[method] = (float(rmse_t[-1]), float(r2_t[-1]))
    return out


def plot_rollout_curves(all_rollouts, output_dir):
    """all_rollouts: dict[(method, noise_label)] -> (rmse_t, r2_t)"""
    os.makedirs(output_dir, exist_ok=True)
    method_colors = {
        "truncated_svd": {"noise-free": "#d62728", "noise-005": "#ff7f0e", "noise-05": "#bcbd22"},
        "ridge":         {"noise-free": "#1f77b4", "noise-005": "#17becf", "noise-05": "#9467bd"},
        "cv_ridge":      {"noise-free": "#2ca02c", "noise-005": "#8c564b", "noise-05": "#e377c2"},
    }
    t_axis = np.arange(N_FRAMES) * DT
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for (method, noise_label), (rmse_t, r2_t) in all_rollouts.items():
        label = f"{method}/{noise_label}"
        c = method_colors.get(method, {}).get(noise_label, "gray")
        ls = "--" if "noise-free" in noise_label else ("-" if "noise-05" == noise_label else ":")
        axes[0].plot(t_axis, rmse_t, label=label, color=c, linewidth=1.2, linestyle=ls)
        axes[1].plot(t_axis, r2_t,  label=label, color=c, linewidth=1.2, linestyle=ls)
    for ax, ylabel, title in [
        (axes[0], "RMSE(t)", "Voltage RMSE vs ground truth"),
        (axes[1], "R²(t)",   "Rollout R² vs ground truth"),
    ]:
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.05)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "rollout_pseudoinverse_W.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\033[92m  saved: {path}\033[0m")


def print_comparison_table(table):
    """table: dict[(method, noise_label)] -> {"conn_r2", "rmse_T", "r2_T"}"""
    noise_labels = list(NOISE_CONDITIONS.keys())
    col_w = 22

    # Header
    header = f"{'Method':<18}"
    for nl in noise_labels:
        header += f"  {nl:^{col_w}}"
    print("\n" + "=" * (18 + len(noise_labels) * (col_w + 2)))
    print(header)
    sub = " " * 18
    for _ in noise_labels:
        sub += f"  {'conn R²':>7}  {'RMSE(T)':>8}  {'R²(T)':>6}"
    print(sub)
    print("-" * (18 + len(noise_labels) * (col_w + 2)))

    for method in METHODS:
        row = f"{method:<18}"
        for nl in noise_labels:
            key = (method, nl)
            if key in table:
                d = table[key]
                row += f"  {d['conn_r2']:7.4f}  {d['rmse_T']:8.2e}  {d['r2_T']:6.4f}"
            else:
                row += f"  {'N/A':>7}  {'N/A':>8}  {'N/A':>6}"
        print(row)
    print("=" * (18 + len(noise_labels) * (col_w + 2)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Methods: {METHODS}")

    table = {}   # (method, noise_label) -> summary dict

    for noise_label, paths in NOISE_CONDITIONS.items():
        ode_path = paths["ode_path"]
        data_dir = paths["data_dir"]

        print(f"\n\033[94m{'='*60}\033[0m")
        print(f"\033[94m{noise_label}\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # --- Weight recovery for all methods ---
        print("\033[96m[1/2] Weight recovery (all methods) ...\033[0m")
        results = analyze_pseudoinverse_multimethod(
            noise_label, ode_path, data_dir, methods=METHODS
        )
        if results is None:
            continue

        # --- Rollouts (GT computed once, shared across methods) ---
        print("\033[96m[2/2] Running rollouts ...\033[0m")
        state = torch.load(ode_path, map_location="cpu", weights_only=True)
        gt_params = FlyVisODEParams(**state).to(device)

        rollout_out = run_rollouts_for_condition(results, state, data_dir, gt_params, device)
        for method, (rmse_T, r2_T) in rollout_out.items():
            table[(method, noise_label)] = {
                "conn_r2": results[method]["global_r2"],
                "rmse_T":  rmse_T,
                "r2_T":    r2_T,
            }
            print(f"  [{method}] conn R²={results[method]['global_r2']:.4f}  "
                  f"RMSE(T)={rmse_T:.2e}  R²(T)={r2_T:.6f}")

    # --- Final comparison table ---
    print_comparison_table(table)


if __name__ == "__main__":
    main()
