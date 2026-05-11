#!/usr/bin/env python
"""Stratified R²(W) analysis for opto-trained models — distinguishes
the two mechanistic levers of opto-driven recovery (T0-2).

For each trained (condition × fold) we partition the E edges into three
strata and compute R²(W_pred, W_gt) per stratum:

    (a) target-source — edges whose presynaptic cell type is in the
        condition's target set. Lever 1 (kernel breaking on target type).
        A perturbation that breaks the within-type sum-zero kernel for
        a target type t lifts recovery on edges sourced from t.

    (b) non-target degenerate — edges in k≥2 groups whose source type is
        NOT in the target set. Lever 2 (column-decorrelation acts as
        general data augmentation).  Any column-distinct drive that
        merely adds activity diversity to the postsynaptic cell can lift
        these edges, regardless of which type was perturbed.

    (c) singleton — edges in k=1 groups (already structurally
        identifiable). Should not lift under any opto perturbation;
        if it does, the analysis pipeline has a confound.

Differential pattern across (a, b, c) tells us which lever is active
in each condition.  Lever-1-only conditions (e.g. Tm4/05) lift (a) but
not (b).  Lever-2-only conditions (e.g. retina/heaviside_var, target
type has no k≥2 groups) lift (b) uniformly without preferential (a)
lift.  Both-levers conditions (e.g. L4/h05) lift both.

This is the priority-1 falsification figure: distinguishes "kernel
breaking helps recovery" from "any column-distinct drive helps recovery."
Pure post-processing — no new generation or training needed.

Inputs (auto-discovered):
    <data_root>/log/fly/flyvis_noise_free_blank50_opto_<cond>_cv<XX>/
        models/best_model_with_*.pt
        gt_weights.pt

    <data_root>/log/fly/flyvis_noise_free_blank50_unified_cv<XX>/
        — noise-free baseline (no opto), used as the reference row.

Outputs:
    figures/stratified_r2.json — per-run + per-condition summary
    figures/stratified_r2.png  — bar chart, 3 strata × N conditions

Usage:
    python figures/analyze_stratified_r2.py
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import zarr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from connectome_gnn.metrics import INDEX_TO_NAME, NAME_TO_INDEX  # noqa: E402
from connectome_gnn.models.flyvis_nullspace import build_degenerate_groups  # noqa: E402
from generate_opto_configs import TARGET_ALIASES  # noqa: E402


DATA_ROOT = "/groups/saalfeld/home/allierc/GraphData"
LOG_ROOT = f"{DATA_ROOT}/log/fly"

OPTO_PREFIX = "flyvis_noise_free_blank50_opto_"
BASELINE_PREFIX = "flyvis_noise_free_blank50_unified_"
NEURON_TYPE_ZARR = (
    f"{DATA_ROOT}/graphs_data/fly/flyvis_noise_free_blank50_cv00/"
    "x_list_train/neuron_type.zarr"
)
OUTPUT_DIR = os.path.join(REPO_ROOT, "figures")

# Waveform suffixes — must mirror scripts/generate_opto_configs.py:_waveform_suffix.
# Listed longest-first so endswith() matching is unambiguous: e.g.
# "TmY15_heaviside_05" matches "heaviside_05" before reaching "05".
KNOWN_WAVEFORM_SUFFIXES = [
    "heaviside_var_005", "heaviside_var_05", "heaviside_var_01", "heaviside_var_02", "heaviside_var_1",
    "heaviside_005", "heaviside_05", "heaviside_01", "heaviside_02", "heaviside_1",
    "heaviside_var", "heaviside",
    "dc_005", "dc_05", "dc_01", "dc_02", "dc_1",
    "constant", "impulse", "video",
    "005", "05", "01", "02", "1",
]


def parse_cond(cond):
    """'TmY15_heaviside_05' → ('TmY15', 'heaviside_05')."""
    for wf in KNOWN_WAVEFORM_SUFFIXES:
        if cond.endswith("_" + wf):
            return cond[: -(len(wf) + 1)], wf
    raise ValueError(f"could not parse waveform suffix from cond={cond!r}")


def expand_target(tag):
    """Mirror of scripts/generate_opto_configs.py:_expand_target."""
    if tag in TARGET_ALIASES:
        return list(TARGET_ALIASES[tag])
    return tag.split("+")


def load_topology():
    """Load gt_weights, edge_index, neuron_type. Topology is identical across
    all opto datasets (same connectome), so we read once from a baseline run."""
    base_dir = f"{LOG_ROOT}/{BASELINE_PREFIX}cv00"
    gt_W = torch.load(f"{base_dir}/gt_weights.pt", map_location="cpu",
                      weights_only=False).numpy()
    edge_index = torch.load(f"{base_dir}/training_edges.pt", map_location="cpu",
                            weights_only=False).numpy()
    neuron_type = np.array(zarr.open_array(NEURON_TYPE_ZARR, mode="r"), dtype=np.int64)
    return gt_W, edge_index, neuron_type


def stratify(edge_index, neuron_type, target_type_ids, groups):
    """Return three boolean masks (a, b, c) over E edges.

    (a) target-source AND in a k≥2 group
    (b) non-target source AND in a k≥2 group
    (c) singleton (in NO k≥2 group)

    Note: a target-source edge that happens to be in a singleton group is
    classified as (c) — it has no kernel direction to break, so it carries
    no lever-1 information regardless of who's perturbed.
    """
    src_types = neuron_type[edge_index[0]]
    is_target_src = np.isin(src_types, np.array(target_type_ids, dtype=np.int64))
    is_degenerate = np.zeros(edge_index.shape[1], dtype=bool)
    for edge_arr in groups.values():
        is_degenerate[edge_arr] = True
    mask_a = is_target_src & is_degenerate
    mask_b = (~is_target_src) & is_degenerate
    mask_c = ~is_degenerate
    return mask_a, mask_b, mask_c


def r2_calibrated(true, pred_calibrated):
    """R² between calibrated predictions and ground truth, on the same scale.

    Caller must have already mapped pred → pred_calibrated via a global
    linear fit. NaN if the stratum is empty or has zero variance.
    """
    if len(true) == 0:
        return float("nan")
    ss_res = float(np.sum((pred_calibrated - true) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def calibrate(W_pred, gt_W):
    """Fit pred = a*gt + b globally and return the gt-scale calibrated prediction.

    The trained GNN's raw W differs from gt_W by a per-neuron slope×grad_msg
    factor (see metrics.compute_corrected_weights); a single global linear
    fit captures the population mean of that factor. The published
    weights_comparison_corrected uses a per-neuron correction (slightly
    tighter), but a global fit is sufficient for the *relative*
    stratum-by-stratum lift this analysis asks about — and avoids needing
    to re-instantiate the trained model + its training data.

    Returns:
        cal_pred:  (E,) prediction on gt_W's scale  (W_pred - b) / a
        a, b:      fitted linear coefficients
    """
    a, b = np.polyfit(gt_W, W_pred, 1)
    cal_pred = (W_pred - b) / a if abs(a) > 1e-12 else W_pred
    return cal_pred, float(a), float(b)


def extract_W(ckpt_path):
    """Return (E,) numpy weights from a trained checkpoint.

    Handles bare W tensors (current architecture, key '_orig_mod.W' under
    torch.compile or 'W' otherwise) and the low-rank WL/WR factorization.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ms = sd["model_state_dict"] if "model_state_dict" in sd else sd
    for key in ("_orig_mod.W", "W"):
        if key in ms:
            return ms[key].squeeze().cpu().numpy()
    if "_orig_mod.WL" in ms and "_orig_mod.WR" in ms:
        return (ms["_orig_mod.WL"] @ ms["_orig_mod.WR"]).squeeze().cpu().numpy()
    if "WL" in ms and "WR" in ms:
        return (ms["WL"] @ ms["WR"]).squeeze().cpu().numpy()
    raise KeyError(
        f"no W or WL/WR in {ckpt_path}; first few keys: {sorted(ms.keys())[:8]}"
    )


def discover_opto_runs():
    """Yield (cond, fold, log_dir) for every opto run with a checkpoint."""
    pat = re.compile(r"^" + re.escape(OPTO_PREFIX) + r"(.+)_cv(\d{2})$")
    for d in sorted(os.listdir(LOG_ROOT)):
        m = pat.match(d)
        if not m:
            continue
        cond, fold = m.group(1), int(m.group(2))
        log_dir = os.path.join(LOG_ROOT, d)
        if glob.glob(f"{log_dir}/models/best_model_with_*.pt"):
            yield cond, fold, log_dir


def discover_baseline_runs():
    """Yield (fold, log_dir) for every noise-free baseline run with a checkpoint."""
    pat = re.compile(r"^" + re.escape(BASELINE_PREFIX) + r"cv(\d{2})$")
    for d in sorted(os.listdir(LOG_ROOT)):
        m = pat.match(d)
        if not m:
            continue
        fold = int(m.group(1))
        log_dir = os.path.join(LOG_ROOT, d)
        if glob.glob(f"{log_dir}/models/best_model_with_*.pt"):
            yield fold, log_dir


def per_run_metrics(log_dir, masks, gt_W_baseline):
    """Compute calibrated R² per stratum for one run.

    1. Extract raw W_pred from checkpoint.
    2. Fit one global linear calibration pred = a*gt + b on ALL edges.
    3. Compute R²(gt, cal_pred) restricted to each stratum.
    """
    ckpts = glob.glob(f"{log_dir}/models/best_model_with_*.pt")
    ckpt = max(ckpts, key=os.path.getmtime)
    W_pred = extract_W(ckpt)
    # Use the run's own gt_weights.pt when present (CV folds share a fixed
    # connectome but reading per-fold guarantees we compare against what
    # this run was trained against).
    gt_path = os.path.join(log_dir, "gt_weights.pt")
    gt_W = (
        torch.load(gt_path, map_location="cpu", weights_only=False).numpy()
        if os.path.isfile(gt_path) else gt_W_baseline
    )

    cal_pred, slope, intercept = calibrate(W_pred, gt_W)
    mask_a, mask_b, mask_c = masks
    return {
        "R2_a": r2_calibrated(gt_W[mask_a], cal_pred[mask_a]),
        "R2_b": r2_calibrated(gt_W[mask_b], cal_pred[mask_b]),
        "R2_c": r2_calibrated(gt_W[mask_c], cal_pred[mask_c]),
        "R2_global": r2_calibrated(gt_W, cal_pred),
        "fit_slope": slope,
        "fit_intercept": intercept,
    }


def aggregate(per_run, target_types_named, masks):
    """mean ± SD across folds; record stratum sizes."""
    s = {}
    for k in ("R2_a", "R2_b", "R2_c", "R2_global"):
        vals = [r[k] for r in per_run if not np.isnan(r[k])]
        if vals:
            s[f"{k}_mean"] = float(np.mean(vals))
            s[f"{k}_std"] = float(np.std(vals, ddof=0))
            s[f"{k}_n"] = len(vals)
    s["n_a"] = int(masks[0].sum())
    s["n_b"] = int(masks[1].sum())
    s["n_c"] = int(masks[2].sum())
    s["target_types"] = target_types_named
    s["n_folds"] = len(per_run)
    return s


def plot_summary(summary, out_path):
    """Stratified-R² bar chart, sorted by Δ(a−b) so lever-1 conditions
    are on the left and lever-2 / null conditions on the right.

    Conditions whose global R² is unstable across folds (mean − 1·SD ≤ 0)
    are excluded from the main panel and noted in the title — they
    otherwise compress the y-axis or report misleading Δ that's smaller
    than the per-fold variance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_key = "baseline_noise_free" if "baseline_noise_free" in summary else None
    opto_keys = [c for c in summary if c != baseline_key]

    # A run is "stable" iff global_R² minus 1·SD is still positive — i.e.
    # the mean is meaningful relative to its fold-to-fold spread.
    def is_stable(c):
        g = summary[c].get("R2_global_mean")
        sd = summary[c].get("R2_global_std", 0.0)
        return g is not None and (g - sd) > 0.0

    stable = [c for c in opto_keys if is_stable(c)]
    collapsed = [c for c in opto_keys if not is_stable(c)]

    # Sort stable conditions by Δ(a−b) descending — left = strong lever-1
    # signature, right = uniform lift (lever-2 / null).
    def delta_ab(c):
        a = summary[c].get("R2_a_mean")
        b = summary[c].get("R2_b_mean")
        return (a - b) if (a is not None and b is not None) else -np.inf

    stable_sorted = sorted(stable, key=delta_ab, reverse=True)
    order = ([baseline_key] if baseline_key else []) + stable_sorted

    n = len(order)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.55 * n + 3.0), 5.2))

    width = 0.27
    xs = np.arange(n)
    colors = {"a": "#d62728", "b": "#1f77b4", "c": "#7f7f7f"}
    labels = {
        "a": "(a) target-source × k≥2 — lever 1",
        "b": "(b) non-target × k≥2 — lever 2",
        "c": "(c) singleton (k=1) — control",
    }
    for i, k in enumerate("abc"):
        vals = [summary[c].get(f"R2_{k}_mean", float("nan")) for c in order]
        errs = [summary[c].get(f"R2_{k}_std", 0.0) for c in order]
        ax.bar(xs + (i - 1) * width, vals, width, yerr=errs,
               color=colors[k], label=labels[k], alpha=0.85,
               edgecolor="white", linewidth=0.5,
               error_kw=dict(lw=0.6, capsize=2))

    # Baseline reference lines (dashed) per stratum.
    if baseline_key:
        for k in "abc":
            v = summary[baseline_key].get(f"R2_{k}_mean")
            if v is not None:
                ax.axhline(v, color=colors[k], lw=0.6, ls="--", alpha=0.5)

    # Highlight the baseline column with a subtle background tint so it
    # reads as the reference rather than an opto condition.
    if baseline_key:
        ax.axvspan(-0.5, 0.5, color="black", alpha=0.04, zorder=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$R^2(\hat W, W_{\rm gt})$ per stratum  (calibrated)")
    title = ("Stratified R² across opto conditions — sorted by Δ(a−b) "
             "[left = lever-1 dominant; right = uniform / lever-2]")
    if collapsed:
        title += f"\n{len(collapsed)} unstable run(s) excluded: {', '.join(collapsed)}"
    ax.set_title(title, fontsize=9.5)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color="black", lw=0.4)
    ax.axhline(1, color="black", lw=0.4)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=3)
    ax.grid(axis="y", lw=0.3, alpha=0.3)

    # Annotate Δ(a−b) above each opto bar group for at-a-glance ranking.
    for xi, c in enumerate(order):
        if c == baseline_key:
            continue
        d = delta_ab(c)
        if np.isfinite(d):
            ax.text(xi, 1.02, f"Δ={d:+.2f}", ha="center", va="bottom",
                    fontsize=6.5, color="black", alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_table(summary):
    order = (["baseline_noise_free"] if "baseline_noise_free" in summary else []) + \
            sorted(c for c in summary if c != "baseline_noise_free")
    print(f"\n{'condition':<28}  {'n_folds':>7}  "
          f"{'R²_a (target k≥2)':>22}  {'R²_b (other k≥2)':>22}  "
          f"{'R²_c (k=1)':>16}  {'R²_global':>11}")
    print("-" * 120)
    for c in order:
        s = summary[c]
        def cell(k):
            m = s.get(f"R2_{k}_mean")
            sd = s.get(f"R2_{k}_std")
            if m is None:
                return "—"
            return f"{m:+.3f}±{sd:.3f}" if sd is not None else f"{m:+.3f}"
        print(f"{c:<28}  {s['n_folds']:>7}  "
              f"{cell('a'):>22}  {cell('b'):>22}  "
              f"{cell('c'):>16}  {cell('global'):>11}")


def main():
    print("loading baseline topology and degenerate groups ...")
    gt_W, edge_index, neuron_type = load_topology()
    n_edges = len(gt_W)
    groups, _, _, _ = build_degenerate_groups(edge_index, neuron_type, n_edges)
    n_degen_edges = sum(len(v) for v in groups.values())
    print(f"  E={n_edges}, |k≥2 groups|={len(groups)}, "
          f"edges in k≥2 groups={n_degen_edges} ({100 * n_degen_edges / n_edges:.1f}%)")

    results = {}

    # Baseline: empty target set — stratum (a) is empty, all degenerate edges go to (b).
    print("\n=== baseline (no opto) ===")
    masks_baseline = stratify(edge_index, neuron_type, [], groups)
    per_run = []
    for fold, log_dir in discover_baseline_runs():
        m = per_run_metrics(log_dir, masks_baseline, gt_W)
        m["fold"] = fold
        per_run.append(m)
        print(f"  cv{fold:02d}: a=NaN  b={m['R2_b']:+.3f}  c={m['R2_c']:+.3f}  "
              f"global={m['R2_global']:+.3f}")
    if per_run:
        results["baseline_noise_free"] = {
            "per_run": per_run,
            **aggregate(per_run, [], masks_baseline),
        }

    # Opto runs: group by condition, stratify per-condition, then iterate folds.
    print("\n=== opto conditions ===")
    by_cond = defaultdict(list)
    for cond, fold, log_dir in discover_opto_runs():
        by_cond[cond].append((fold, log_dir))

    for cond in sorted(by_cond):
        try:
            target_tag, waveform = parse_cond(cond)
        except ValueError as e:
            print(f"  skip {cond}: {e}")
            continue
        target_types_named = expand_target(target_tag)
        target_type_ids = [NAME_TO_INDEX[n] for n in target_types_named if n in NAME_TO_INDEX]
        masks = stratify(edge_index, neuron_type, target_type_ids, groups)
        n_a, n_b, n_c = (int(m.sum()) for m in masks)

        print(f"\n  {cond}  target={target_tag} → "
              f"{','.join(target_types_named)}  wf={waveform}  "
              f"n_a={n_a} n_b={n_b} n_c={n_c}")

        per_run = []
        for fold, log_dir in sorted(by_cond[cond]):
            m = per_run_metrics(log_dir, masks, gt_W)
            m["fold"] = fold
            per_run.append(m)
            print(f"    cv{fold:02d}: a={m['R2_a']:+.3f}  b={m['R2_b']:+.3f}  "
                  f"c={m['R2_c']:+.3f}  global={m['R2_global']:+.3f}")
        results[cond] = {
            "per_run": per_run,
            **aggregate(per_run, target_types_named, masks),
        }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "stratified_r2.json")
    with open(json_path, "w") as f:
        json.dump({
            "n_edges_total": int(n_edges),
            "n_k_ge_2_groups": int(len(groups)),
            "n_edges_in_k_ge_2_groups": int(n_degen_edges),
            "summary": {c: {k: v for k, v in s.items() if k != "per_run"}
                        for c, s in results.items()},
            "per_run": {c: s["per_run"] for c, s in results.items()},
        }, f, indent=2)
    print(f"\nwrote {json_path}")

    fig_path = os.path.join(OUTPUT_DIR, "stratified_r2.png")
    plot_summary({c: s for c, s in results.items()}, fig_path)
    print(f"wrote {fig_path}")

    print_table({c: s for c, s in results.items()})


if __name__ == "__main__":
    main()
