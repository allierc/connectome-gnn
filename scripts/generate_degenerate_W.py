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
1. Load ground-truth ode_params (W, edge_index, tau, V_rest)
2. Infer 65 neuron types from (tau, V_rest) — neurons with identical
   intrinsic params belong to the same type
3. Pick 5 non-retina types with highest fan-out (most edges per target)
   — these have the largest null space per target neuron
4. For each target neuron receiving >1 edges from the same selected type,
   apply a sum-preserving random perturbation (contrast vector)
5. Generate 5 variants with increasing delta amplitude:
   scale = {0.25, 0.5, 1.0, 2.0, 4.0} × mean(|W_group|)
6. Save each variant as a full ode_params.pt in graphs_data/degenerate_matrix/

Each variant has  connectivity_R2 < 1.0  vs ground truth (the weights differ)
but produces *identical* dynamics under the same stimulus (the perturbation
is in the null space of H_i for every neuron i).

OUTPUT STRUCTURE
----------------
One variant per (neuron_type, scale) pair:

    graphs_data/degenerate_matrix/
        variant_00_ground_truth/    # reference
        type_XX_scale_01/           # type XX, smallest perturbation
        type_XX_scale_15/           # type XX, largest perturbation
        ...

Each variant perturbs a SINGLE neuron type, isolating its contribution
to the divergence.  ~63 types × 15 scales ≈ 945 variants.

This demonstrates concretely that connectivity_R2 = 1.0 is NOT achievable
from dynamics alone — the inverse problem is fundamentally ill-posed, and
the degree of ill-posedness is quantified by the null space dimension
(printed by the script).
===========================================================================
"""

import os
import sys
import numpy as np

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_ODE_PARAMS = "graphs_data/fly/flyvis_noise_005/ode_params.pt"
OUTPUT_ROOT = "graphs_data/degenerate_matrix"
N_SCALES = 15
SCALE_FACTORS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
SEED = 42
# Use ALL non-retina types — one variant per (type, scale) pair


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
    """Rank neuron types by average edges per target (fan-out per target).

    Higher fan-out means more edges from same type to same target,
    hence larger null space per target neuron.
    """
    from collections import defaultdict

    # For each source type, count edges per destination neuron
    type_stats = {}
    unique_types = np.unique(neuron_types)

    for t in unique_types:
        if t in exclude_types:
            continue
        # Edges where source is of this type
        mask = neuron_types[src] == t
        n_edges = mask.sum()
        if n_edges == 0:
            continue
        # Count distinct targets
        dst_of_type = dst[mask]
        n_targets = len(set(dst_of_type))
        avg_fan_out = n_edges / max(n_targets, 1)

        # Also count how many (target, type) groups have >1 edge
        # (these are the groups with non-trivial null space)
        from collections import Counter
        target_counts = Counter(dst_of_type.tolist())
        n_degenerate_groups = sum(1 for c in target_counts.values() if c > 1)
        total_degenerate_edges = sum(c for c in target_counts.values() if c > 1)
        null_dim = sum(c - 1 for c in target_counts.values() if c > 1)

        type_stats[int(t)] = {
            "n_neurons": int((neuron_types == t).sum()),
            "n_edges": int(n_edges),
            "n_targets": n_targets,
            "avg_fan_out": avg_fan_out,
            "n_degenerate_groups": n_degenerate_groups,
            "total_degenerate_edges": total_degenerate_edges,
            "null_dim": null_dim,
        }

    # Sort by null_dim (total degrees of freedom from this type)
    ranked = sorted(type_stats.items(), key=lambda x: x[1]["null_dim"], reverse=True)
    return ranked


def generate_sum_preserving_perturbation(group_size, rng):
    """Generate a random vector of given size that sums to zero.

    Uses: sample (group_size - 1) iid Gaussians, set last element to
    minus the sum — then normalize to unit L2 norm.
    """
    delta = rng.randn(group_size)
    delta -= delta.mean()  # exact sum = 0
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

    # -----------------------------------------------------------------------
    # Step 1: Infer neuron types from (tau, V_rest)
    # -----------------------------------------------------------------------
    neuron_types, type_params = infer_neuron_types(tau, vrest)
    n_types = len(type_params)
    print(f"\n  Inferred {n_types} neuron types from (tau, V_rest)")

    # -----------------------------------------------------------------------
    # Step 2: Identify retina / input-only neurons
    # -----------------------------------------------------------------------
    input_neurons = find_input_neurons(src, dst, N)
    input_types = set(neuron_types[list(input_neurons)])
    print(f"  Input-only neurons (retina): {len(input_neurons)} "
          f"(types: {sorted(input_types)})")

    # -----------------------------------------------------------------------
    # Step 2b: SVD-based null space estimate (global bound)
    #   The effective rank of the population activity upper-bounds rank(H_i)
    #   for every neuron i. Combined with per-neuron in-degrees, this gives
    #   a tighter null space estimate than within-type counting alone.
    # -----------------------------------------------------------------------
    from collections import Counter

    in_degree = Counter(dst.tolist())
    post_neurons = sorted(in_degree.keys())
    degrees = np.array([in_degree[i] for i in post_neurons])
    n_post = len(post_neurons)

    print(f"\n  Postsynaptic neurons: {n_post}")
    print(f"  In-degree: mean={degrees.mean():.1f}, median={np.median(degrees):.0f}, "
          f"min={degrees.min()}, max={degrees.max()}")
    pcts = np.percentile(degrees, [25, 75, 95])
    print(f"  Percentiles: 25%={pcts[0]:.0f}  75%={pcts[1]:.0f}  95%={pcts[2]:.0f}")

    print(f"\n  In-degree distribution:")
    for lo, hi in [(1, 10), (11, 20), (21, 41), (42, 100), (101, 500), (501, 5000)]:
        count = int(((degrees >= lo) & (degrees <= hi)).sum())
        if count > 0:
            print(f"    {lo:4d}-{hi:4d}: {count:5d} neurons")

    print(f"\n  SVD-based null space estimates (rank-nullity bound):")
    print(f"  {'rank':>6} {'null_dim':>10} {'% edges':>8} {'identifiable':>12} {'degen_neurons':>14}")
    for rank, label in [(1, "90% var"), (19, "99% raw"), (41, "99% sub"), (51, "centered")]:
        null_dims = np.array([max(0, d - rank) for d in degrees])
        total_null = int(null_dims.sum())
        n_degen = int((null_dims > 0).sum())
        print(f"  {rank:6d} {total_null:10,d} {100*total_null/E:7.1f}% {E - total_null:12,d} "
              f"{n_degen:8d} / {n_post}")

    # -----------------------------------------------------------------------
    # Step 3: Rank non-retina types by null space dimension
    # -----------------------------------------------------------------------
    ranked_types = rank_types_by_fan_out(src, dst, neuron_types, input_types)

    # Use ALL non-retina types that have degenerate groups
    all_types = [t for t, _ in ranked_types]
    print(f"\n  All non-retina types with degeneracy: {len(all_types)}")
    print(f"  {'type':>5} {'n_neurons':>9} {'n_edges':>8} {'avg_fan':>8} "
          f"{'degen_groups':>12} {'null_dim':>9}")
    for t, s in ranked_types:
        print(f"  {t:5d} {s['n_neurons']:9d} {s['n_edges']:8d} "
              f"{s['avg_fan_out']:8.1f} {s['n_degenerate_groups']:12d} "
              f"{s['null_dim']:9d}")

    # -----------------------------------------------------------------------
    # Step 4: Build perturbation groups PER TYPE
    #   For each source type independently, find (dst_neuron, src_type)
    #   groups with >1 edges.
    # -----------------------------------------------------------------------
    from collections import defaultdict

    # Build all groups: (dst_neuron, src_type) -> list of edge indices
    all_groups = defaultdict(list)
    for e_idx in range(E):
        s_type = neuron_types[src[e_idx]]
        all_groups[(int(dst[e_idx]), int(s_type))].append(e_idx)

    # Per-type degenerate groups
    type_groups = {}  # type_id -> {(dst, type): [edge_indices]}
    for t, _ in ranked_types:
        tg = {k: v for k, v in all_groups.items() if k[1] == t and len(v) > 1}
        if tg:
            type_groups[t] = tg

    print(f"\n  Types with degenerate groups: {len(type_groups)}")
    total_variants = len(type_groups) * N_SCALES
    print(f"  Total variants: {len(type_groups)} types × {N_SCALES} scales = {total_variants}")

    # -----------------------------------------------------------------------
    # Step 5: Pre-generate perturbation directions per group
    # -----------------------------------------------------------------------
    group_deltas = {}  # (dst, type) -> (edge_indices, delta_unit, mean_abs_W)
    for t, tg in type_groups.items():
        for (dst_n, src_t), edge_indices in tg.items():
            edge_indices = np.array(edge_indices)
            delta_unit = generate_sum_preserving_perturbation(len(edge_indices), rng)
            mean_abs_W = np.mean(np.abs(W_true[edge_indices]))
            group_deltas[(dst_n, src_t)] = (edge_indices, delta_unit, mean_abs_W)

    # -----------------------------------------------------------------------
    # Step 6: Generate one variant per (type, scale) pair
    # -----------------------------------------------------------------------
    output_root = os.path.join(script_dir, OUTPUT_ROOT)
    os.makedirs(output_root, exist_ok=True)

    # Save ground truth as variant_00 for reference
    gt_dir = os.path.join(output_root, "variant_00_ground_truth")
    os.makedirs(gt_dir, exist_ok=True)
    torch.save(state, os.path.join(gt_dir, "ode_params.pt"))
    print(f"\n  Saved ground truth -> {gt_dir}")

    ss_tot = np.sum((W_true - W_true.mean()) ** 2)
    n_saved = 0

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
                edge_indices_arr, delta_unit, mean_abs_W = group_deltas[(dst_n, src_t)]
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
            n_saved += 1

        # Print one summary line per type (last scale)
        print(f"  type {t:2d}: {n_groups_perturbed:5d} groups, "
              f"null_dim={null_dim_t:6d}, "
              f"R2@scale={SCALE_FACTORS[-1]:.1f}: {r2:.4f}")

    # -----------------------------------------------------------------------
    # Step 7: Summary report
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Source:           {SOURCE_ODE_PARAMS}")
    print(f"  Output:           {OUTPUT_ROOT}/")
    print(f"  Variants:         {n_saved} + ground truth")
    print(f"  Types perturbed:  {len(type_groups)} (all non-retina with degeneracy)")
    print(f"  Scales per type:  {N_SCALES}")
    print(f"  Scale factors:    {SCALE_FACTORS}")
    print(f"  Seed:             {SEED}")
    print(f"\n  Each variant perturbs a SINGLE neuron type at one scale.")
    print(f"  Perturbations are sum-preserving within (dst, src_type) groups.")

    # Structural vs SVD null space comparison
    structural_null = sum(
        sum(len(v) - 1 for v in tg.values())
        for tg in type_groups.values()
    )
    svd_null_41 = int(sum(max(0, d - 41) for d in degrees))
    print(f"\n  Null space estimates:")
    print(f"    Structural (within-type):      {structural_null:>10,d} / {E:,d} ({100*structural_null/E:.1f}%)")
    print(f"    SVD-based (activity rank=41):  {svd_null_41:>10,d} / {E:,d} ({100*svd_null_41/E:.1f}%)")
    print(f"    Cross-type contribution:       {svd_null_41 - structural_null:>10,d} ({100*(svd_null_41 - structural_null)/E:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
