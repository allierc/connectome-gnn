#!/usr/bin/env python
"""Generate degenerate connectivity matrices for the flyvis model.

===========================================================================
THEORETICAL BACKGROUND — ILL-POSEDNESS OF THE NOISE-FREE INVERSE PROBLEM
===========================================================================

The flyvis ODE is:

    tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * ReLU(v_j) + e_i(t)

Given observed trajectories v(t), recovering (W, tau, V_rest) is an inverse
problem.  Rearranging for each postsynaptic neuron i, at each time t:

    sum_j W_ij * h_j(t) = tau_i * dv_i/dt + v_i - V_rest_i - e_i(t)

where h_j(t) = ReLU(v_j(t)).  Stacking over T timesteps gives a linear
system  H_i * w_i = b_i, where H_i is the (T x d_i) activity matrix
restricted to the d_i presynaptic partners of neuron i.

NULL SPACE AND DEGENERACY
-------------------------
Any perturbation  delta_w  in  null(H_i)  produces *identical* trajectories.
The null space dimension is  d_i - rank(H_i).

WITHIN-TYPE DEGENERACY (the dominant mechanism)
-----------------------------------------------
In flyvis, neurons of the same cell type that project to the same target
have correlated (often nearly identical) activity — their columns in H_i
are (nearly) linearly dependent.  If k presynaptic neurons of the same type
all connect to neuron i, the null space includes (k-1) contrast directions:
perturbations that *redistribute* weight among those k edges while keeping
the sum constant:

    sum_{j in group} delta_W_ij = 0

This script exploits this structure.  The sparsity pattern (which neurons
connect) is FIXED — we never create new edges or remove existing ones.
We only redistribute weight among existing edges from the same presynaptic
type to the same postsynaptic neuron.

WHAT THE SCRIPT DOES
--------------------
SECTION 1: SINGLE-TYPE VARIANTS (all types × 15 scales)
1. Load ground-truth ode_params (W, edge_index, tau, V_rest)
2. Infer neuron types from (tau, V_rest) — neurons with identical
   intrinsic params belong to the same type (including retina)
3. For each type with degenerate groups, apply perturbations at 15 fixed scales:
   scale = {0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, ...} × mean(|W_group|)
4. Generate 15 variants per type (one scale per variant)

SECTION 2: MIXED-TYPE VARIANTS (top 10 types, 1000 variants)
5. Identify top 10 types by null space dimension
6. Generate 1000 variants perturbing all 10 types simultaneously
   - Each variant: random null-space directions + random amplitude per type
   - Tests whether degeneracy is robust to coordinated multi-type perturbations

OUTPUT STRUCTURE
----------------
    graphs_data/degenerate_matrix/
        variant_00_ground_truth/       # reference
        type_XX_scale_01/              # type XX, scale 1 (single-type)
        type_XX_scale_15/              # type XX, scale 15 (single-type)
        mixed_types_var_0001/          # mixed (top 10 types), variant 1
        mixed_types_var_1000/          # mixed (top 10 types), variant 1000

Total: 975 single-type + 1000 mixed-type = 1975 variants

Each variant perturbs one type OR all top 10 types, isolating/combining
their contributions to the dynamics divergence.

===========================================================================
"""

import os
import sys
import numpy as np

import torch

# Try to import flyvis for cell type names (optional)
try:
    from flyvis import NetworkView
    FLYVIS_AVAILABLE = True
except ImportError:
    FLYVIS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_ODE_PARAMS = "../graphs_data/fly/flyvis_noise_free/ode_params.pt"
OUTPUT_ROOT = "../graphs_data/degenerate_matrix"
N_SCALES = 15
SCALE_FACTORS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
N_MIXED_VARIANTS = 1000
N_TOP_TYPES_MIXED = 10
AMPLITUDE_RANGE = (0.05, 8.0)  # For mixed variants: log-uniform sampling
SEED = 42

# Hardcoded mapping from (tau, V_rest) to flyvis cell type names
# This is derived from the flyvis biomodel
FLYVIS_TYPE_NAMES = {
    # Photoreceptors (R cells)
    (0.01, -0.05): "R1-R6",  # Approximate
    (0.02, -0.05): "R7",
    (0.02, -0.05): "R8",
    # Lamina (L cells)
    (0.01, -0.05): "L1-L5",
    # Medulla (Mi cells)
    (0.01, -0.05): "Mi1-Mi4",
    (0.01, -0.05): "Mi9-Mi12",
    (0.01, -0.05): "Mi13-Mi15",
    # Medulla (Tm cells)
    (0.01, -0.05): "Tm1-Tm4",
    (0.01, -0.05): "Tm5",
    (0.01, -0.05): "Tm9/Tm16/Tm20",
    (0.01, -0.05): "Tm28/Tm30",
    (0.01, -0.05): "TmY",
    # Motion (T cells)
    (0.01, -0.05): "T4",
    (0.01, -0.05): "T5",
    (0.01, -0.05): "T1-T3",
    (0.01, -0.05): "Lawf",
}


def load_flyvis_cell_type_mapping():
    """Load actual flyvis cell type names from connectome JSON file.

    The flyvis connectome.json contains 65 cell types with their names.
    We create a mapping from type ID (0-64) to cell type name.

    Returns:
        unique_type_names: list of 65 cell type names (indices 0-64)
        neuron_to_type_id: (13741,) array mapping neuron index to type ID
                           loaded from neuron_type.zarr
        (empty lists if not available)
    """
    import json
    import os

    try:
        # First, try to load from flyvis package data
        # The connectome JSON file contains all 65 cell types with their names
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # Try multiple possible locations for the connectome JSON
        possible_paths = [
            "/workspace/.conda_envs/neural-graph-linux/lib/python3.12/site-packages/flyvis/connectome/fib25-fib19_v2.2.json",
            os.path.expanduser("~/.conda/envs/*/lib/python3*/site-packages/flyvis/connectome/fib25-fib19_v2.2.json"),
        ]

        # Also check if flyvis is importable and get config path
        if FLYVIS_AVAILABLE:
            try:
                from flyvis.utils.config_utils import CONFIG_PATH
                possible_paths.insert(0, os.path.join(CONFIG_PATH, "..", "..", "connectome", "fib25-fib19_v2.2.json"))
            except:
                pass

        connectome_json_path = None
        for path_pattern in possible_paths:
            if "*" in path_pattern:
                import glob
                matches = glob.glob(path_pattern)
                if matches:
                    connectome_json_path = matches[0]
                    break
            elif os.path.exists(path_pattern):
                connectome_json_path = path_pattern
                break

        if not connectome_json_path:
            # Try to find any fib25-fib19_v2.2.json in the system
            import subprocess
            result = subprocess.run(
                ["find", "/workspace", "-name", "fib25-fib19_v2.2.json", "-type", "f"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                connectome_json_path = result.stdout.strip().split('\n')[0]

        if not connectome_json_path or not os.path.exists(connectome_json_path):
            return [], np.array([])

        # Load connectome JSON to get cell type names
        with open(connectome_json_path, 'r') as f:
            connectome_data = json.load(f)

        # Extract cell type names in order (indices 0-64)
        unique_type_names = [node['name'] for node in connectome_data['nodes']]

        # Now load neuron-to-type mapping from zarr file (all neuron_type.zarr are identical)
        # We use the zarr from one of the simulated datasets since flyvis_noise_free doesn't have it
        import zarr

        zarr_paths = [
            "graphs_data/fly/flyvis_hodgkin_huxley/x_list_train/neuron_type.zarr",
            "graphs_data/fly/flyvis_adex_coba/x_list_train/neuron_type.zarr",
            "graphs_data/fly/flyvis_noise_free/neuron_type.zarr",  # In case it's added later
        ]

        neuron_to_type_id = None
        for zarr_path in zarr_paths:
            if os.path.exists(zarr_path):
                try:
                    z = zarr.open_array(zarr_path, mode='r')
                    neuron_to_type_id = z[:]
                    break
                except:
                    continue

        if neuron_to_type_id is None:
            print(f"  Warning: Could not load neuron_type from zarr files")
            return unique_type_names, np.array([])

        # Verify we have the right size
        if len(neuron_to_type_id) != 13741:
            print(f"  Warning: neuron_to_type_id size {len(neuron_to_type_id)} != 13741")
            return unique_type_names, neuron_to_type_id

        return unique_type_names, np.array(neuron_to_type_id, dtype=np.int64)

    except Exception as e:
        print(f"  Note: Could not load flyvis cell type mapping ({e})")
        return [], np.array([])


def infer_neuron_types(tau, vrest, decimals=6):
    """Infer neuron type indices from (tau, V_rest) pairs.

    Neurons with identical intrinsic parameters belong to the same type.
    Returns (neuron_types, type_params) where type_params maps type_id
    to (tau_val, vrest_val).
    """
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


def find_input_neurons(src, dst, N):
    """Neurons that appear as source but never as destination (photoreceptors)."""
    dst_set = set(dst)
    src_set = set(src)
    return set(i for i in range(N) if i not in dst_set and i in src_set)


def rank_types_by_fan_out(src, dst, neuron_types, exclude_types):
    """Rank neuron types by null space dimension (degeneracy).

    Higher null space dimension means more free parameters.
    """
    from collections import defaultdict

    type_stats = {}
    unique_types = np.unique(neuron_types)

    for t in unique_types:
        if t in exclude_types:
            continue
        mask = neuron_types[src] == t
        n_edges = mask.sum()
        if n_edges == 0:
            continue
        dst_of_type = dst[mask]
        n_targets = len(set(dst_of_type))

        from collections import Counter
        target_counts = Counter(dst_of_type.tolist())
        n_degenerate_groups = sum(1 for c in target_counts.values() if c > 1)
        total_degenerate_edges = sum(c for c in target_counts.values() if c > 1)
        null_dim = sum(c - 1 for c in target_counts.values() if c > 1)

        type_stats[int(t)] = {
            "n_neurons": int((neuron_types == t).sum()),
            "n_edges": int(n_edges),
            "n_targets": n_targets,
            "avg_fan_out": float(n_edges / max(n_targets, 1)),
            "n_degenerate_groups": n_degenerate_groups,
            "total_degenerate_edges": total_degenerate_edges,
            "null_dim": null_dim,
        }

    ranked = sorted(type_stats.items(), key=lambda x: x[1]["null_dim"], reverse=True)
    return ranked


def color_r2(r2_value):
    """Return R2 with ANSI color codes: green >0.95, orange >0.5, red otherwise."""
    if r2_value > 0.95:
        return f"\033[92m{r2_value:.4f}\033[0m"  # Green
    elif r2_value > 0.50:
        return f"\033[93m{r2_value:.4f}\033[0m"  # Orange
    else:
        return f"\033[91m{r2_value:.4f}\033[0m"  # Red


def generate_sum_preserving_perturbation(group_size, rng):
    """Generate a random vector of given size that sums to zero.

    Uses: sample (group_size - 1) iid Gaussians, set last element to
    minus the sum — then normalize to unit L2 norm.
    """
    delta = rng.randn(group_size)
    delta -= delta.mean()
    norm = np.linalg.norm(delta)
    if norm > 0:
        delta /= norm
    return delta


def main():
    rng = np.random.RandomState(SEED)

    # Load ground-truth ODE params
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, SOURCE_ODE_PARAMS)
    if not os.path.exists(source_path):
        print(f"ERROR: source ode_params not found at {source_path}")
        sys.exit(1)

    print(f"Loading ground-truth ODE params from {source_path}")
    state = torch.load(source_path, map_location="cpu", weights_only=True)
    W_true = state["W"].numpy().copy()
    edge_index = state["edge_index"].numpy()
    tau = state["tau_i"].numpy()
    vrest = state["V_i_rest"].numpy()

    N = len(tau)
    E = len(W_true)
    src, dst = edge_index[0], edge_index[1]

    print(f"  N = {N} neurons,  E = {E} edges")
    print(f"  W: mean={W_true.mean():.4f}, std={W_true.std():.4f}, "
          f"range=[{W_true.min():.4f}, {W_true.max():.4f}]")

    # Load flyvis cell type mapping (proper way — same as in graph_data_generator.py)
    print("Loading flyvis cell type mapping...")
    unique_type_names, neuron_to_type_id = load_flyvis_cell_type_mapping()

    # For diagnostic purposes, also infer from (tau, V_rest) to compare
    neuron_types_tau_vrest, type_params_tau_vrest = infer_neuron_types(tau, vrest)

    if len(unique_type_names) > 0:
        print(f"  ✓ Loaded {len(unique_type_names)} cell types from flyvis network")
        # For compatibility, create a simple mapping from type ID to name
        type_id_to_name = {i: name for i, name in enumerate(unique_type_names)}

        # Diagnostic: check if flyvis and tau/vrest assignments agree
        mismatches = (neuron_to_type_id != neuron_types_tau_vrest).sum()
        print(f"  Diagnostic: {mismatches} neurons have different assignment in tau/vrest vs flyvis")
        if mismatches > 0:
            print(f"  ⚠️  Cell type assignments DIFFER between flyvis and tau/vrest approaches")
    else:
        print("  (will use type IDs as identifiers)")
        type_id_to_name = {}
        # Fallback: infer types from (tau, V_rest) if flyvis not available
        n_types = len(type_params_tau_vrest)
        print(f"  Inferred {n_types} neuron types from (tau, V_rest) (fallback)")
        neuron_to_type_id = neuron_types_tau_vrest

    # Identify input neurons (retina) — now included in variant generation
    input_neurons = find_input_neurons(src, dst, N)
    input_types = set(neuron_to_type_id[list(input_neurons)])
    print(f"  Input-only neurons (retina): {len(input_neurons)} "
          f"(types: {sorted(input_types)})")
    print(f"  Note: Retina types are NOW INCLUDED in variant generation")

    # Rank types by null space dimension (including retina types)
    ranked_types = rank_types_by_fan_out(src, dst, neuron_to_type_id, exclude_types=set())

    # Build degenerate groups per type (using proper neuron-to-type mapping from flyvis)
    from collections import defaultdict
    all_groups = defaultdict(list)
    for e_idx in range(E):
        s_type = neuron_to_type_id[src[e_idx]]
        all_groups[(int(dst[e_idx]), int(s_type))].append(e_idx)

    type_groups = {}
    for t, _ in ranked_types:
        tg = {k: v for k, v in all_groups.items() if k[1] == t and len(v) > 1}
        if tg:
            type_groups[t] = tg

    # Create output directory and save ground truth
    output_root = os.path.join(script_dir, OUTPUT_ROOT)
    os.makedirs(output_root, exist_ok=True)

    gt_dir = os.path.join(output_root, "variant_00_ground_truth")
    os.makedirs(gt_dir, exist_ok=True)
    torch.save(state, os.path.join(gt_dir, "ode_params.pt"))
    print(f"\n  Saved ground truth -> {gt_dir}")

    ss_tot = np.sum((W_true - W_true.mean()) ** 2)

    # =========================================================================
    # SECTION 1: SINGLE-TYPE VARIANTS (65 types × 15 scales)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"SECTION 1: SINGLE-TYPE VARIANTS")
    print(f"{'='*70}")
    print(f"  Types with degenerate groups: {len(type_groups)}")
    print(f"  Scales per type: {N_SCALES}")
    print(f"  Total single-type variants: {len(type_groups)} types × {N_SCALES} = {len(type_groups) * N_SCALES}")

    n_saved_single = 0

    for t in sorted(type_groups.keys()):
        tg = type_groups[t]
        null_dim_t = sum(len(v) - 1 for v in tg.values())

        for si, scale in enumerate(SCALE_FACTORS):
            variant_name = f"type_{t:02d}_scale_{si+1:02d}"
            variant_dir = os.path.join(output_root, variant_name)
            os.makedirs(variant_dir, exist_ok=True)

            W_perturbed = W_true.copy()

            n_groups_perturbed = 0
            for (dst_n, src_t), edge_indices in tg.items():
                edge_indices_arr = np.array(edge_indices)
                delta_unit = generate_sum_preserving_perturbation(len(edge_indices_arr), rng)
                mean_abs_W = np.mean(np.abs(W_true[edge_indices_arr]))
                amplitude = scale * mean_abs_W
                W_perturbed[edge_indices_arr] += amplitude * delta_unit
                n_groups_perturbed += 1

            # Stats vs ground truth
            diff = W_perturbed - W_true
            rmse = np.sqrt(np.mean(diff ** 2))
            ss_res = np.sum(diff ** 2)
            r2 = 1.0 - ss_res / ss_tot

            # Save
            variant_state = dict(state)
            variant_state["W"] = torch.tensor(W_perturbed, dtype=torch.float32)
            torch.save(variant_state, os.path.join(variant_dir, "ode_params.pt"))

            meta = {
                "scale_factor": float(scale),
                "perturbed_type": int(t),
                "null_dim_type": int(null_dim_t),
                "n_groups_perturbed": n_groups_perturbed,
                "connectivity_R2_vs_gt": float(r2),
                "rmse_vs_gt": float(rmse),
                "W_mean": float(W_perturbed.mean()),
                "W_std": float(W_perturbed.std()),
                "seed": SEED,
            }
            torch.save(meta, os.path.join(variant_dir, "metadata.pt"))
            n_saved_single += 1

        # Print one summary line per type (last scale)
        r2_colored = color_r2(r2)
        type_name = type_id_to_name.get(t, f"type_{t:2d}")
        print(f"  {type_name:8s}: {n_groups_perturbed:5d} groups, null_dim={null_dim_t:6d}, "
              f"R2@scale={SCALE_FACTORS[-1]:.1f}: {r2_colored}")

    print(f"\n  ✓ Saved {n_saved_single} single-type variants")

    # =========================================================================
    # SECTION 2: MIXED-TYPE VARIANTS (top 10 types, 1000 variants)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"SECTION 2: MIXED-TYPE VARIANTS (top {N_TOP_TYPES_MIXED} types)")
    print(f"{'='*70}")

    # Select top types by null space dimension
    top_types = [t for t, _ in ranked_types[:N_TOP_TYPES_MIXED]]
    print(f"  Top {N_TOP_TYPES_MIXED} types by null space:")
    print(f"  {'#':>3} {'Type':>10} {'null_dim':>10} {'groups':>8}")
    for i, t in enumerate(top_types, 1):
        tg = type_groups[t] if t in type_groups else {}
        null_dim = sum(len(v) - 1 for v in tg.values()) if tg else 0
        n_groups = len(tg) if tg else 0
        type_name = type_id_to_name.get(t, f"type_{t:2d}") if type_id_to_name else f"type_{t:2d}"
        print(f"  {i:3d} {type_name:>10s} {null_dim:10d} {n_groups:8d}")

    # Compute total null space for top types
    total_null_dim_mixed = sum(
        sum(len(v) - 1 for v in type_groups[t].values())
        for t in top_types if t in type_groups
    )
    print(f"  Total null space (all top types): {total_null_dim_mixed}")
    print(f"  Variants to generate: {N_MIXED_VARIANTS}")
    print(f"  Amplitude range (log-uniform): [{AMPLITUDE_RANGE[0]:.2f}, {AMPLITUDE_RANGE[1]:.2f}]")

    n_saved_mixed = 0
    conn_r2_values = []  # Track connectivity R2

    for var_id in range(1, N_MIXED_VARIANTS + 1):
        # Each mixed variant gets a fresh random seed
        variant_rng = np.random.RandomState(SEED + 1000000 + var_id)

        # Build group deltas for all top types
        mixed_group_deltas = {}
        for t in top_types:
            if t not in type_groups:
                continue
            tg = type_groups[t]
            for (dst_n, src_t), edge_indices in tg.items():
                edge_indices_arr = np.array(edge_indices)
                delta_unit = generate_sum_preserving_perturbation(len(edge_indices_arr), variant_rng)
                mean_abs_W = np.mean(np.abs(W_true[edge_indices_arr]))
                mixed_group_deltas[(dst_n, src_t)] = (edge_indices_arr, delta_unit, mean_abs_W)

        # Apply perturbations with random amplitudes per type
        W_perturbed = W_true.copy()
        n_groups_perturbed = 0
        type_amplitudes = {}

        for t in top_types:
            if t not in type_groups:
                continue
            # Random amplitude for this type (log-uniform)
            log_amp = variant_rng.uniform(np.log(AMPLITUDE_RANGE[0]), np.log(AMPLITUDE_RANGE[1]))
            amplitude_scale = np.exp(log_amp)
            type_amplitudes[t] = amplitude_scale

            tg = type_groups[t]
            for (dst_n, src_t), edge_indices in tg.items():
                edge_indices_arr, delta_unit, mean_abs_W = mixed_group_deltas[(dst_n, src_t)]
                amplitude = amplitude_scale * mean_abs_W
                W_perturbed[edge_indices_arr] += amplitude * delta_unit
                n_groups_perturbed += 1

        # Compute R2
        diff = W_perturbed - W_true
        rmse = np.sqrt(np.mean(diff ** 2))
        ss_res = np.sum(diff ** 2)
        conn_r2 = 1.0 - ss_res / ss_tot
        conn_r2_values.append(conn_r2)

        # Save variant
        variant_name = f"mixed_types_var_{var_id:04d}"
        variant_dir = os.path.join(output_root, variant_name)
        os.makedirs(variant_dir, exist_ok=True)

        variant_state = dict(state)
        variant_state["W"] = torch.tensor(W_perturbed, dtype=torch.float32)
        torch.save(variant_state, os.path.join(variant_dir, "ode_params.pt"))

        meta = {
            "variant_id": int(var_id),
            "perturbed_types": [int(t) for t in top_types],
            "n_types": len(top_types),
            "n_groups_perturbed": n_groups_perturbed,
            "type_amplitudes": {int(t): float(a) for t, a in type_amplitudes.items()},
            "connectivity_R2_vs_gt": float(conn_r2),
            "rmse_vs_gt": float(rmse),
            "W_mean": float(W_perturbed.mean()),
            "W_std": float(W_perturbed.std()),
            "seed": int(SEED + 1000000 + var_id),
        }
        torch.save(meta, os.path.join(variant_dir, "metadata.pt"))
        n_saved_mixed += 1

        # Print progress every 100 variants with stats
        if var_id % 100 == 0:
            conn_mean = np.mean(conn_r2_values)
            conn_std = np.std(conn_r2_values)
            conn_min = np.min(conn_r2_values)
            conn_max = np.max(conn_r2_values)
            print(f"  Generated variant {var_id:4d}/{N_MIXED_VARIANTS}:")
            print(f"    connectivity R2: mean={conn_mean:.4f}±{conn_std:.4f}  min={conn_min:.4f}  max={conn_max:.4f}")
            print(f"    rollout R2:      [computed via separate ODE rollout]")

    print(f"\n  ✓ Saved {n_saved_mixed} mixed-type variants")

    # Print final statistics for mixed-type variants
    if conn_r2_values:
        conn_mean = np.mean(conn_r2_values)
        conn_std = np.std(conn_r2_values)
        conn_min = np.min(conn_r2_values)
        conn_max = np.max(conn_r2_values)
        print(f"\n  final statistics (mixed-type variants, 1000 variants):")
        print(f"    connectivity R2: mean={conn_mean:.4f}±{conn_std:.4f}  min={conn_min:.4f}  max={conn_max:.4f}")
        print(f"    rollout R2:      [to be computed via ODE rollout script]")

    # =========================================================================
    # Summary
    # =========================================================================
    # Save type mapping for reference
    print(f"\n{'='*70}")
    print(f"CELL TYPES WITH DEGENERATE GROUPS")
    print(f"{'='*70}")
    if len(type_id_to_name) > 0:
        print(f"\n{'Cell Type':>15} {'n_neurons':>10}")
        for t in sorted(type_groups.keys()):
            n_neurons = int((neuron_to_type_id == t).sum())
            cell_type_name = type_id_to_name.get(t, "Unknown")
            print(f"{cell_type_name:>15} {n_neurons:10d}")
    else:
        print(f"\n{'Type ID':>8} {'n_neurons':>10}")
        for t in sorted(type_groups.keys()):
            n_neurons = int((neuron_to_type_id == t).sum())
            print(f"{t:8d} {n_neurons:10d}")

    print(f"\n{'='*70}")
    print(f"NEXT STEP")
    print(f"{'='*70}")
    print(f"To compute rollout R2 (dynamics preservation) for all variants:")
    print(f"  python scripts/rollout_degenerate_W.py")
    print(f"This will run ODE simulations and generate rollout statistics.\n")

    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Source:               {SOURCE_ODE_PARAMS}")
    print(f"  Output:               {OUTPUT_ROOT}/")
    print(f"\n  SECTION 1 (Single-Type):")
    print(f"    Types:              {len(type_groups)}")
    print(f"    Scales per type:    {N_SCALES}")
    print(f"    Variants:           {n_saved_single}")
    print(f"    Scale factors:      {SCALE_FACTORS}")
    print(f"\n  SECTION 2 (Mixed-Type):")
    print(f"    Top types:          {N_TOP_TYPES_MIXED}")
    print(f"    Variants:           {n_saved_mixed}")
    print(f"    Amplitude range:    log-uniform [{AMPLITUDE_RANGE[0]}, {AMPLITUDE_RANGE[1]}]")
    print(f"\n  TOTAL VARIANTS:       {n_saved_single + n_saved_mixed}")
    print(f"  Seed:                 {SEED}")
    print(f"  Perturbations:        sum-preserving within (dst, src_type) groups")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
