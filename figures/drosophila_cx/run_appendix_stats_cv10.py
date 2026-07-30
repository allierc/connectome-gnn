"""Appendix A statistics across 10 Known-ODE CV models.

Replaces the single-model numbers in drosophila.tex Appendix A with
mean ± std across 10 cv folds. Reuses the perturbation/rollout
primitives from drosophila_nullspace.py.

Outputs (written under figures/drosophila_cx/):
  tab_variants_cx_cv10.tex   — per-cell-type sum-zero variants table,
                                 mean ± std across 10 models
  appendix_stats_cv10.json   — sparsify rollout r values (per-(i,α)
                                 + per-(i,α,instance), sum-preserving
                                 + calibrated) as mean ± std

Run with:
  python figures/drosophila_cx/run_appendix_stats_cv10.py --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import zarr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "connectome_gnn", "models"))

from connectome_gnn.utils import (
    log_path, load_data_root_from_json, set_data_root,
)
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model

from drosophila_nullspace import (  # noqa: E402
    build_degenerate_groups, build_unit_groups, load_neuron_instances,
    sum_zero_vector, make_single_type_variant,
    sparse_sum_preserving, sparse_calibrated,
    cx_rollout, rollout_and_metrics,
    compute_null_dim,
    PERTURBATION_SCALES, DT, N_ROLLOUT,
)


NOISE_FREE_DIR = (
    "/groups/saalfeld/home/allierc/GraphData/graphs_data/"
    "drosophila_cx/drosophila_cx_pi_epg_voltage_noise_free"
)


def _load_stim_and_v0(n_steps, device):
    """Reuse the noise-free dataset's input (constant-omega + initial
    impulse) and initial voltage. The same input drives every model;
    each model's W_rec gives a different voltage trajectory under it.
    """
    stim = np.asarray(zarr.open(
        os.path.join(NOISE_FREE_DIR, "x_list_test", "stimulus.zarr"),
        mode="r"))[:n_steps].astype(np.float32)
    v0 = np.asarray(zarr.open(
        os.path.join(NOISE_FREE_DIR, "x_list_test", "voltage.zarr"),
        mode="r"))[0].astype(np.float32)
    return (torch.from_numpy(stim).to(device),
            torch.from_numpy(v0).to(device))


def _load_model_W_rec(config_name, device):
    """Return (W_edge, edge_index, b, tau, neuron_types, type_names)
    from a trained checkpoint."""
    config, _ = load_run_config(config_name, explicit_output_root=False,
                                 task="train")
    net = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    ckpt_dir = os.path.join(log_path(config.config_file), "models")
    cands = sorted(
        glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
        key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")),
    )
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    sd = torch.load(cands[-1], map_location=device,
                    weights_only=False)["model_state_dict"]
    net.load_state_dict(sd)

    W_dense = net.W_rec.detach().cpu().numpy().astype(np.float32)
    W_con   = net.W_con.detach().cpu().numpy().astype(np.float32)
    # edge_index = nonzero entries of W_con, order = (pre, post). Match
    # drosophila_nullspace.load_ground_truth (already pre, post).
    post, pre = np.nonzero(W_con)
    edge_index = np.stack([pre, post], axis=0).astype(np.int64)
    W_edge = W_dense[post, pre].astype(np.float32)

    b = net.b.detach().cpu().numpy().astype(np.float32)
    tau = float(net.tau if hasattr(net, "tau") else 0.1)
    nt = np.asarray(net.neuron_types).astype(int)
    names = list(net.type_names)
    return W_edge, edge_index, b, tau, nt, names, os.path.basename(cands[-1])


def _run_one_model(config_name, stim, v0, n_steps, device, rng_seed=42):
    """Run all perturbation + sparsify analyses for one trained model."""
    W_gt, edge_index, b, tau, ntype, type_names, ckpt = _load_model_W_rec(
        config_name, device)
    N = b.size; E = W_gt.size

    print(f"  [{config_name}] N={N}, E={E}, ckpt={ckpt}")

    # GT voltage under shared stim.
    v_gt = cx_rollout(W_gt, edge_index, b, tau, stim, v0,
                       n_steps=n_steps, device=device)
    v_gt_np = v_gt.numpy()

    # Per-(post, pre-type) groups → sum-zero variants and sparsify.
    groups, _, _, _ = build_degenerate_groups(edge_index, ntype, E)

    out = {"ckpt": ckpt}
    # ---- single-type variants at λ ∈ {1, 2, 4} -----------------------
    type_results = {}     # type_name -> {scale -> (R²_W, r)}
    rng = np.random.RandomState(rng_seed)
    type_ids = sorted(set(int(t) for (_, t) in groups.keys()))
    for tid in type_ids:
        tname = type_names[tid] if tid < len(type_names) else f"t{tid}"
        type_results[tname] = {}
        for scale in PERTURBATION_SCALES:
            W_var = make_single_type_variant(W_gt, groups, tid, scale, rng)
            r2, r = rollout_and_metrics(W_var, W_gt, edge_index, b, tau,
                                          stim, v0, v_gt, device)
            type_results[tname][float(scale)] = (float(r2), float(r))
    out["type_variants"] = type_results

    # ---- sparsify per-(i,α): sum-preserving + calibrated -------------
    W_sp, sp_stats = sparse_sum_preserving(W_gt, groups)
    r2_sp, r_sp = rollout_and_metrics(W_sp, W_gt, edge_index, b, tau,
                                        stim, v0, v_gt, device)
    out["sparse_iaplha_sum_preserving"] = float(r_sp)
    W_cal, _ = sparse_calibrated(W_gt, groups, edge_index, v_gt_np)
    r2_cal, r_cal = rollout_and_metrics(W_cal, W_gt, edge_index, b, tau,
                                          stim, v0, v_gt, device)
    out["sparse_iaplha_calibrated"] = float(r_cal)

    # ---- sparsify per-(i,α,instance): sum-preserving + calibrated ----
    instances = load_neuron_instances()
    unit_groups = build_unit_groups(edge_index, ntype, instances, E)
    W_sp_u, _ = sparse_sum_preserving(W_gt, unit_groups)
    _, r_sp_u = rollout_and_metrics(W_sp_u, W_gt, edge_index, b, tau,
                                      stim, v0, v_gt, device)
    out["sparse_unit_sum_preserving"] = float(r_sp_u)
    W_cal_u, _ = sparse_calibrated(W_gt, unit_groups, edge_index, v_gt_np)
    _, r_cal_u = rollout_and_metrics(W_cal_u, W_gt, edge_index, b, tau,
                                       stim, v0, v_gt, device)
    out["sparse_unit_calibrated"] = float(r_cal_u)

    # group sizes (structural — same across models; report once).
    out["n_groups_iaplha"] = len(groups)
    out["n_groups_unit"]   = len(unit_groups)
    out["edges_iaplha"]    = int(sum(len(v) for v in groups.values()))
    out["edges_unit"]      = int(sum(len(v) for v in unit_groups.values()))

    print(f"    sparsify: (i,α) sum-pres r={r_sp:.3f}, calib r={r_cal:.3f} | "
          f"(i,α,inst) sum-pres r={r_sp_u:.3f}, calib r={r_cal_u:.3f}")
    return out


def _aggregate(per_model_results, type_names):
    """mean ± std per cell-type per lambda + per sparsify metric."""
    n = len(per_model_results)
    # Per-type per-scale
    types = sorted({t for m in per_model_results for t in m["type_variants"]})
    scales = sorted({s for m in per_model_results
                       for s in m["type_variants"][next(iter(m["type_variants"]))]})
    agg_type = {t: {s: {"r2": [], "r": []} for s in scales} for t in types}
    for m in per_model_results:
        for t, sd in m["type_variants"].items():
            for s, (r2, r) in sd.items():
                agg_type[t][s]["r2"].append(r2)
                agg_type[t][s]["r"].append(r)
    summary = {}
    for t in types:
        summary[t] = {}
        for s in scales:
            arr_r2 = np.array(agg_type[t][s]["r2"], dtype=np.float64)
            arr_r  = np.array(agg_type[t][s]["r"],  dtype=np.float64)
            summary[t][float(s)] = {
                "r2_mean": float(arr_r2.mean()), "r2_std": float(arr_r2.std()),
                "r_mean":  float(arr_r .mean()), "r_std":  float(arr_r .std()),
            }
    # Sparsify metrics
    def _ms(key):
        a = np.array([m[key] for m in per_model_results], dtype=np.float64)
        return float(a.mean()), float(a.std())
    sparsify = {
        "iaplha_sum_preserving": _ms("sparse_iaplha_sum_preserving"),
        "iaplha_calibrated":     _ms("sparse_iaplha_calibrated"),
        "unit_sum_preserving":   _ms("sparse_unit_sum_preserving"),
        "unit_calibrated":       _ms("sparse_unit_calibrated"),
    }
    structural = {
        "n_models": n,
        "n_groups_iaplha": per_model_results[0]["n_groups_iaplha"],
        "n_groups_unit":   per_model_results[0]["n_groups_unit"],
        "edges_iaplha":    per_model_results[0]["edges_iaplha"],
        "edges_unit":      per_model_results[0]["edges_unit"],
    }
    return summary, sparsify, structural


def _emit_table_tex(summary, structural, out_path,
                     scales_to_show=(1.0, 2.0, 4.0)):
    """Per-cell-type sum-zero variants table, mean±std across models."""
    types = list(summary.keys())
    # Re-order to match the existing table layout (EPG, EPGt, PEN_a, PEN_b,
    # Delta7, PEG, ER6) when present.
    desired = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "Delta7",
                "PEG", "ER6"]
    types = [t for t in desired if t in summary] + \
             [t for t in types if t not in desired]
    available = sorted(next(iter(summary.values())).keys())
    scales = [s for s in scales_to_show if s in available]
    lines = []
    lines.append(r"\begin{tabular}{l" + "rr" * len(scales) + "}")
    lines.append(r"\toprule")
    header1 = ["Cell Type"]
    for s in scales:
        header1.append(rf"\multicolumn{{2}}{{c}}{{$\lambda = {int(s)}$}}")
    lines.append(" & ".join(header1) + r" \\")
    header2 = [""]
    for _ in scales:
        header2.extend([r"$R^2_{\mathbf{W}}$", r"$r$"])
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")
    for t in types:
        row = [t.replace("_", r"\_")]
        for s in scales:
            r2 = summary[t][s]["r2_mean"]; r2s = summary[t][s]["r2_std"]
            rr = summary[t][s]["r_mean"];  rrs = summary[t][s]["r_std"]
            row.append(rf"${r2:.3f}\pm{r2s:.3f}$")
            row.append(rf"${rr:.3f}\pm{rrs:.3f}$")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="drosophila_cx_pi_epg_tv")
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--n_steps", type=int, default=N_ROLLOUT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=None)
    p.add_argument("--out_dir", default=HERE)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)
    stim, v0 = _load_stim_and_v0(args.n_steps, device)
    print(f"loaded shared stim ({args.n_steps} steps) and initial voltage")

    per_model = []
    for k in range(args.n_folds):
        cfg = f"{args.base}_cv{k}"
        res = _run_one_model(cfg, stim, v0, args.n_steps, device,
                              rng_seed=args.seed + k)
        per_model.append(res)
    type_names_first = list(per_model[0]["type_variants"].keys())
    summary, sparsify, structural = _aggregate(per_model, type_names_first)

    print("\n=== aggregate sparsify (mean ± std across 10 models) ===")
    for k, (m, s) in sparsify.items():
        print(f"  {k:30s} {m:.3f} ± {s:.3f}")
    print(f"  structural: n_groups (i,α)={structural['n_groups_iaplha']} "
          f"({structural['edges_iaplha']} edges); "
          f"(i,α,inst)={structural['n_groups_unit']} "
          f"({structural['edges_unit']} edges)")

    out_tex = os.path.join(args.out_dir, "tab_variants_cx_cv10.tex")
    _emit_table_tex(summary, structural, out_tex)

    out_json = os.path.join(args.out_dir, "appendix_stats_cv10.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "sparsify": sparsify,
                    "structural": structural,
                    "per_model": per_model}, f, indent=2)
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
