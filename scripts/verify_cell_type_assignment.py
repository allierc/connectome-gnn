#!/usr/bin/env python
"""Verify that flyvis cell type assignment matches (tau, V_rest) inference.

This script checks:
1. Load flyvis cell type assignment (actual ground truth from biomodel)
2. Infer cell types from (tau, V_rest) from ode_params
3. Compare the two assignments - should be the same if no bug
4. Show differences if they exist
"""

import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

SOURCE_ODE_PARAMS = "../graphs_data/fly/flyvis_noise_free/ode_params.pt"

try:
    from flyvis import Network
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config
    FLYVIS_AVAILABLE = True
except Exception as e:
    print(f"DEBUG: Flyvis import failed: {e}", file=sys.stderr)
    FLYVIS_AVAILABLE = False


def infer_types_from_tau_vrest(tau, vrest, decimals=6):
    """Infer type IDs from (tau, V_rest) - OLD approach."""
    N = len(tau)
    type_map = {}
    type_id = 0
    neuron_types = np.zeros(N, dtype=np.int64)
    type_params = {}
    for i in range(N):
        key = (round(float(tau[i]), decimals), round(float(vrest[i]), decimals))
        if key not in type_map:
            type_map[key] = type_id
            type_params[type_id] = key
            type_id += 1
        neuron_types[i] = type_map[key]
    return neuron_types, type_params


def load_flyvis_types():
    """Load actual cell type assignment from flyvis network - NEW approach."""
    if not FLYVIS_AVAILABLE:
        return None, None, None

    config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    net = Network(**config_net)

    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]

    unique_types, neuron_to_type_id = np.unique(node_types, return_inverse=True)
    unique_type_names = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in unique_types]

    return neuron_to_type_id, unique_type_names, node_types_str


def main():
    print("=" * 80)
    print("CELL TYPE ASSIGNMENT VERIFICATION")
    print("=" * 80)

    # Load ODE params
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, SOURCE_ODE_PARAMS)

    print(f"\nLoading ODE params from {source_path}")
    state = torch.load(source_path, map_location="cpu", weights_only=True)
    tau = state["tau_i"].numpy()
    vrest = state["V_i_rest"].numpy()
    N = len(tau)
    print(f"  N = {N} neurons")

    # Approach 1: Infer from (tau, V_rest)
    print("\n[1/3] Inferring types from (tau, V_rest)...")
    tau_vrest_types, tau_vrest_params = infer_types_from_tau_vrest(tau, vrest)
    n_tau_vrest_types = len(tau_vrest_params)
    print(f"  ✓ Found {n_tau_vrest_types} unique (tau, V_rest) pairs")

    # Approach 2: Load from flyvis
    print("\n[2/3] Loading types from flyvis network...")
    if FLYVIS_AVAILABLE:
        flyvis_types, unique_type_names, node_types_str = load_flyvis_types()
        n_flyvis_types = len(unique_type_names)
        print(f"  ✓ Loaded {n_flyvis_types} cell types from flyvis")
        print(f"  Cell types: {unique_type_names}")
    else:
        print("  ✗ Flyvis not available - cannot verify!")
        return

    # Compare
    print("\n[3/3] Comparing assignments...")

    # Check if they match
    matches = (tau_vrest_types == flyvis_types).sum()
    mismatches = (tau_vrest_types != flyvis_types).sum()

    print(f"  Neurons with matching assignment: {matches} / {N}")
    print(f"  Neurons with mismatched assignment: {mismatches} / {N}")

    if mismatches > 0:
        print(f"\n  ⚠️  Found {mismatches} mismatches!")
        print(f"  This means (tau, V_rest) and flyvis disagree on neuron types.")

        # Show which neurons have mismatches
        mismatch_idx = np.where(tau_vrest_types != flyvis_types)[0][:20]  # First 20
        print(f"\n  First 20 mismatches (neuron_idx: tau_vrest_type → flyvis_type):")
        for idx in mismatch_idx:
            tau_type_id = tau_vrest_types[idx]
            flyvis_type_id = flyvis_types[idx]
            tau_vrest_pair = tau_vrest_params[tau_type_id]
            flyvis_name = unique_type_names[flyvis_type_id]
            print(f"    {idx:5d}: type {tau_type_id:2d} (tau={tau_vrest_pair[0]:.6f}, vrest={tau_vrest_pair[1]:.6f}) → {flyvis_name}")
    else:
        print(f"\n  ✓ All assignments match! (tau, V_rest) correctly identifies cell types.")

    # Compare type distributions
    print(f"\n  Type distribution comparison:")
    print(f"  {'Approach':>20} {'# Types':>10} {'Types w/ degenerate groups':>30}")

    # Count degenerate groups for tau_vrest approach
    from collections import defaultdict
    edge_index = state["edge_index"].numpy()
    src, dst = edge_index[0], edge_index[1]

    # tau_vrest groups
    tau_vrest_groups = defaultdict(list)
    for e_idx in range(len(src)):
        s_type = tau_vrest_types[src[e_idx]]
        tau_vrest_groups[(int(dst[e_idx]), int(s_type))].append(e_idx)

    tau_vrest_types_with_degen = set()
    for (dst_n, src_t), edges in tau_vrest_groups.items():
        if len(edges) > 1:
            tau_vrest_types_with_degen.add(src_t)

    # flyvis groups
    flyvis_groups = defaultdict(list)
    for e_idx in range(len(src)):
        s_type = flyvis_types[src[e_idx]]
        flyvis_groups[(int(dst[e_idx]), int(s_type))].append(e_idx)

    flyvis_types_with_degen = set()
    for (dst_n, src_t), edges in flyvis_groups.items():
        if len(edges) > 1:
            flyvis_types_with_degen.add(src_t)

    print(f"  {'(tau, V_rest)':>20} {n_tau_vrest_types:>10} {len(tau_vrest_types_with_degen):>30}")
    print(f"  {'flyvis network':>20} {n_flyvis_types:>10} {len(flyvis_types_with_degen):>30}")

    # Summary
    print("\n" + "=" * 80)
    if mismatches == 0:
        print("✓ CONCLUSION: Cell type assignments are CONSISTENT")
        print("  Both approaches identify the same types for each neuron.")
        print("  Using flyvis network assignment is safe and correct.")
    else:
        print("✗ CONCLUSION: Cell type assignments DIFFER")
        print("  The two approaches identify different types for some neurons.")
        print("  Need to investigate the source of the difference.")
    print("=" * 80)


if __name__ == "__main__":
    main()
