#!/usr/bin/env python3
"""Cross-seed ensemble analysis of learned W matrices.

===========================================================================
SCIENTIFIC MOTIVATION
===========================================================================
Multiple CV training seeds converge to DIFFERENT solutions — all with similar
rollout accuracy, but varying connectivity R². This is a direct consequence of
the ill-posed inverse problem: the null-space manifold

    [w_i*] = {w_i* + δ : δ ∈ ker(H_i)}

is continuous (~121K-dimensional), and gradient descent from different random
initializations lands on different points within it.

This script exploits the multi-seed ensemble to:

  1. ESTIMATE the null-space directions empirically: the SVD of the centered
     W matrix (each row = one seed's solution) captures the principal axes of
     variation across seeds — these are directions in the null space.

  2. IMPROVE connectivity recovery via the CONSENSUS ESTIMATE: averaging across
     seeds cancels out the random null-space components while amplifying the
     identifiable subspace. The mean W is expected to have higher R² than any
     individual seed.

  3. MAP edge identifiability: per-edge standard deviation across seeds
     measures how constrained each weight is. Low-std edges are robustly
     recovered (in the identifiable subspace); high-std edges are degenerate
     (in the null space).

  4. PROVIDE INPUT for sparse W optimization: the mean W is the best linear
     unbiased estimator of the true W restricted to the identifiable subspace,
     and is the natural starting point for L1-constrained sparsification.

ANALOGY TO SAMPLING FROM THE POSTERIOR
---------------------------------------
In Bayesian terms, each seed provides one sample from the posterior
P(W | dynamics data). The prior is implicit (random initialization ~ Gaussian).
The posterior is concentrated on the null-space affine manifold, and the
empirical distribution of seeds approximates the posterior covariance.

===========================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

LOG_ROOT = os.environ.get("GNN_OUTPUT_ROOT", os.path.join(REPO_ROOT, ".."))
# fallback to canonical HPC path if workspace local path is not available
_hpc_root = "/groups/saalfeld/home/allierc/GraphData"
if not os.path.isdir(os.path.join(LOG_ROOT, "log")):
    LOG_ROOT = _hpc_root

CONFIG_NAME = "flyvis_noise_005"
PRE_FOLDER = "fly"
N_SEEDS = 5  # cv00 .. cv04

OUTPUT_DIR = os.path.join(REPO_ROOT, "scripts", "cv_W_ensemble_results")


def r_squared(y_true, y_pred):
    """Coefficient of determination R²."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def load_cv_weights(log_root, config_name, pre_folder, n_seeds):
    """Load corrected_W.pt from each CV seed.

    Returns:
        W_stack:  (n_seeds, E) numpy array of learned weights
        gt_w:     (E,) ground-truth weights (same for all seeds)
        edges:    (2, E) edge index
        seed_dirs: list of log dirs
    """
    W_list = []
    gt_w = None
    edges = None
    seed_dirs = []

    for i in range(n_seeds):
        run_name = f"{config_name}_cv{i:02d}"
        log_dir = os.path.join(log_root, "log", pre_folder, run_name)
        seed_dirs.append(log_dir)

        cw_path = os.path.join(log_dir, "results", "corrected_W.pt")
        if not os.path.isfile(cw_path):
            print(f"  WARNING: {cw_path} not found — skipping seed {i}")
            W_list.append(None)
            continue

        cw = torch.load(cw_path, map_location="cpu", weights_only=True)
        W_list.append(cw.detach().squeeze().numpy())  # (E,)
        print(f"  cv{i:02d}: loaded W  shape={cw.shape}  "
              f"mean={cw.mean().item():.4f}  std={cw.std().item():.4f}")

        # Load GT once (same for all seeds using same config)
        if gt_w is None:
            gt_path = os.path.join(log_dir, "gt_weights.pt")
            if os.path.isfile(gt_path):
                gt_w = torch.load(gt_path, map_location="cpu", weights_only=True).numpy()
                print(f"  GT weights loaded: shape={gt_w.shape}  "
                      f"range=[{gt_w.min():.4f}, {gt_w.max():.4f}]")
            edges_path = os.path.join(log_dir, "training_edges.pt")
            if os.path.isfile(edges_path):
                edges = torch.load(edges_path, map_location="cpu", weights_only=True).numpy()
                print(f"  Training edges: shape={edges.shape}")

    # Filter out missing seeds
    valid = [(i, w) for i, w in enumerate(W_list) if w is not None]
    if not valid:
        raise RuntimeError("No CV seeds found!")
    seed_ids, Ws = zip(*valid)
    W_stack = np.stack(Ws, axis=0)  # (n_valid, E)
    print(f"\n  W_stack shape: {W_stack.shape}  ({len(valid)}/{n_seeds} seeds)")

    return W_stack, gt_w, edges, seed_dirs, list(seed_ids)


def compute_per_edge_stats(W_stack, gt_w):
    """Compute per-edge statistics across CV seeds."""
    W_mean = W_stack.mean(axis=0)     # (E,)
    W_std  = W_stack.std(axis=0)      # (E,)
    W_med  = np.median(W_stack, axis=0)  # (E,)

    # Coefficient of variation (normalized std)
    W_cv = np.abs(W_std) / (np.abs(W_mean) + 1e-8)

    # Per-seed R² vs GT
    seed_r2 = [r_squared(gt_w, W_stack[i]) for i in range(W_stack.shape[0])]

    # Consensus R² vs GT
    mean_r2 = r_squared(gt_w, W_mean)
    med_r2  = r_squared(gt_w, W_med)

    return {
        "W_mean":   W_mean,
        "W_std":    W_std,
        "W_med":    W_med,
        "W_cv":     W_cv,
        "seed_r2":  seed_r2,
        "mean_r2":  mean_r2,
        "med_r2":   med_r2,
    }


def null_space_svd(W_stack):
    """SVD of centered W matrix → empirical null-space directions.

    The row space of (W_stack - mean) gives the principal variation directions
    across seeds.  These are directions in the null space that gradient descent
    explores from different initializations.

    Returns:
        U:      (n_seeds, n_seeds) left singular vectors (seed combinations)
        S:      (n_seeds,) singular values (variance captured per direction)
        Vt:     (n_seeds, E) right singular vectors (edge-space null directions)
        var_exp: (n_seeds,) fraction of variance explained per direction
    """
    W_centered = W_stack - W_stack.mean(axis=0)  # (n_seeds, E)
    # SVD of (n_seeds, E) — much cheaper than SVD of (E, E)
    U, S, Vt = np.linalg.svd(W_centered, full_matrices=False)
    S = S[:W_centered.shape[0]]  # truncate to min(n_seeds, E) non-zero

    var_exp = S ** 2 / (S ** 2).sum() if S.sum() > 0 else np.zeros_like(S)

    return U, S, Vt, var_exp


def classify_edges(W_std, gt_w, W_mean, percentile_low=25, percentile_high=75):
    """Classify edges by identifiability.

    Returns:
        mask_robust:  edges with std < p25 (well-identified)
        mask_degen:   edges with std > p75 (likely degenerate)
    """
    p25 = np.percentile(W_std, percentile_low)
    p75 = np.percentile(W_std, percentile_high)

    mask_robust = W_std < p25
    mask_degen  = W_std > p75

    r2_robust = r_squared(gt_w[mask_robust], W_mean[mask_robust])
    r2_degen  = r_squared(gt_w[mask_degen],  W_mean[mask_degen])
    r2_all    = r_squared(gt_w, W_mean)

    print(f"\n  Edge classification:")
    print(f"    Robust  (std < p{percentile_low}): {mask_robust.sum():6d} edges  "
          f"R²={r2_robust:.4f}")
    print(f"    Degen   (std > p{percentile_high}): {mask_degen.sum():6d} edges  "
          f"R²={r2_degen:.4f}")
    print(f"    All:                {len(gt_w):6d} edges  R²={r2_all:.4f}")

    return mask_robust, mask_degen, r2_robust, r2_degen


def plot_ensemble(W_stack, gt_w, stats, seed_ids, var_exp, output_dir):
    """Produce 4-panel diagnostic figure."""
    os.makedirs(output_dir, exist_ok=True)
    W_mean = stats["W_mean"]
    W_std  = stats["W_std"]
    seed_r2 = stats["seed_r2"]
    mean_r2 = stats["mean_r2"]
    n_seeds = W_stack.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- Panel (0,0): GT vs Mean W, colored by per-edge std ----
    ax = axes[0, 0]
    # subsample 20K points for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(gt_w), size=min(20000, len(gt_w)), replace=False)
    sc = ax.scatter(gt_w[idx], W_mean[idx], c=W_std[idx], s=1, alpha=0.4,
                    cmap="hot_r", vmin=0, vmax=np.percentile(W_std, 95))
    plt.colorbar(sc, ax=ax, label="per-edge std across seeds")
    ax.set_xlabel("Ground-truth $W_{ij}$", fontsize=12)
    ax.set_ylabel("Mean learned $W_{ij}$", fontsize=12)
    ax.set_title(f"Consensus W vs GT   R²={stats['mean_r2']:.4f}", fontsize=13)
    lo = min(gt_w.min(), W_mean.min())
    hi = max(gt_w.max(), W_mean.max())
    ax.plot([lo, hi], [lo, hi], "b--", linewidth=0.8, alpha=0.5, label="y=x")
    ax.legend(fontsize=9)

    # ---- Panel (0,1): Per-edge std distribution + R² decomposition ----
    ax = axes[0, 1]
    ax.hist(W_std, bins=100, color="steelblue", alpha=0.7, label="per-edge std")
    ax.axvline(np.percentile(W_std, 25), color="green", linestyle="--",
               label=f"p25 (robust threshold)")
    ax.axvline(np.percentile(W_std, 75), color="red", linestyle="--",
               label=f"p75 (degen threshold)")
    ax.set_xlabel("Std of $W_{ij}$ across seeds", fontsize=12)
    ax.set_ylabel("Number of edges", fontsize=12)
    ax.set_title("Per-edge variability distribution", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_yscale("log")

    # ---- Panel (1,0): R² per seed + consensus ----
    ax = axes[1, 0]
    x = list(range(n_seeds)) + [n_seeds]
    r2_vals = seed_r2 + [mean_r2]
    colors = ["steelblue"] * n_seeds + ["tomato"]
    labels = [f"cv{seed_ids[i]:02d}" for i in range(n_seeds)] + ["mean"]
    bars = ax.bar(x, r2_vals, color=colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Connectivity $R^2$", fontsize=12)
    ax.set_title("Per-seed and consensus R²", fontsize=13)
    ax.set_ylim(min(r2_vals) - 0.01, 1.01)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    # ---- Panel (1,1): SVD variance explained by null-space directions ----
    ax = axes[1, 1]
    k = min(len(var_exp), n_seeds)
    ax.bar(range(1, k + 1), var_exp[:k] * 100, color="purple", alpha=0.7)
    ax.set_xlabel("SVD component (null-space direction)", fontsize=12)
    ax.set_ylabel("% variance explained", fontsize=12)
    ax.set_title("Null-space structure across seeds\n"
                 "(SVD of centered W matrix)", fontsize=13)
    ax.set_xticks(range(1, k + 1))

    plt.suptitle(f"CV ensemble W analysis — {CONFIG_NAME}  ({n_seeds} seeds)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "cv_W_ensemble.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  Saved {fig_path}")


def plot_identifiability_map(W_std, gt_w, W_mean, edges, n_neurons, output_dir):
    """Plot per-neuron average std (in-strength and out-strength of std)."""
    os.makedirs(output_dir, exist_ok=True)
    src, dst = edges[0], edges[1]

    # Per-neuron incoming edge std (average over incoming edges)
    in_std = np.zeros(n_neurons)
    in_count = np.zeros(n_neurons, dtype=int)
    np.add.at(in_std, dst, W_std)
    np.add.at(in_count, dst, 1)
    mask = in_count > 0
    in_std[mask] /= in_count[mask]

    # Per-neuron outgoing edge std
    out_std = np.zeros(n_neurons)
    out_count = np.zeros(n_neurons, dtype=int)
    np.add.at(out_std, src, W_std)
    np.add.at(out_count, src, 1)
    mask2 = out_count > 0
    out_std[mask2] /= out_count[mask2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(range(n_neurons), np.sort(in_std)[::-1], s=1, alpha=0.5, color="tomato")
    ax.set_xlabel("Neuron rank (by incoming std)", fontsize=11)
    ax.set_ylabel("Mean incoming edge std", fontsize=11)
    ax.set_title("Incoming edge identifiability per neuron", fontsize=12)

    ax = axes[1]
    ax.scatter(range(n_neurons), np.sort(out_std)[::-1], s=1, alpha=0.5, color="steelblue")
    ax.set_xlabel("Neuron rank (by outgoing std)", fontsize=11)
    ax.set_ylabel("Mean outgoing edge std", fontsize=11)
    ax.set_title("Outgoing edge identifiability per neuron", fontsize=12)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "cv_W_neuron_identifiability.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved {fig_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CV Ensemble W Analysis — {CONFIG_NAME}  ({N_SEEDS} seeds)")
    print(f"{'='*70}")
    print(f"Log root:  {LOG_ROOT}")
    print(f"Output:    {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # 1. Load CV weights
    # ------------------------------------------------------------------
    print(f"\n[1/5] Loading CV weights ...")
    W_stack, gt_w, edges, seed_dirs, seed_ids = load_cv_weights(
        LOG_ROOT, CONFIG_NAME, PRE_FOLDER, N_SEEDS)

    E = W_stack.shape[1]
    n_seeds = W_stack.shape[0]

    if gt_w is None:
        print("ERROR: GT weights not found. Aborting.")
        return
    if edges is None:
        print("ERROR: Training edges not found. Aborting.")
        return

    n_neurons = int(max(edges[0].max(), edges[1].max())) + 1
    print(f"  E={E} edges,  N={n_neurons} neurons")

    # ------------------------------------------------------------------
    # 2. Per-edge statistics
    # ------------------------------------------------------------------
    print(f"\n[2/5] Computing per-edge statistics ...")
    stats = compute_per_edge_stats(W_stack, gt_w)

    print(f"\n  Per-seed R² vs GT:")
    for i, r2 in enumerate(stats["seed_r2"]):
        print(f"    cv{seed_ids[i]:02d}: {r2:.4f}")
    print(f"  Consensus mean  R²: {stats['mean_r2']:.4f}")
    print(f"  Consensus median R²: {stats['med_r2']:.4f}")
    print(f"\n  Mean W: mean={stats['W_mean'].mean():.4f}  std={stats['W_mean'].std():.4f}")
    print(f"  Per-edge std: mean={stats['W_std'].mean():.4f}  "
          f"median={np.median(stats['W_std']):.4f}  "
          f"p95={np.percentile(stats['W_std'],95):.4f}")

    # ------------------------------------------------------------------
    # 3. Null-space SVD
    # ------------------------------------------------------------------
    print(f"\n[3/5] SVD of centered W matrix (empirical null-space directions) ...")
    U, S, Vt, var_exp = null_space_svd(W_stack)

    print(f"  Singular values: {S[:n_seeds]}")
    print(f"  Variance explained:")
    for k, ve in enumerate(var_exp[:n_seeds]):
        print(f"    Component {k+1}: {ve*100:.1f}%")

    # ------------------------------------------------------------------
    # 4. Edge classification
    # ------------------------------------------------------------------
    print(f"\n[4/5] Edge identifiability classification ...")
    mask_robust, mask_degen, r2_robust, r2_degen = classify_edges(
        stats["W_std"], gt_w, stats["W_mean"])

    # Fraction of total weight accounted for by each class
    w_total_abs = np.abs(gt_w).sum()
    print(f"    Weight fraction — robust: {np.abs(gt_w[mask_robust]).sum()/w_total_abs:.1%}, "
          f"degen: {np.abs(gt_w[mask_degen]).sum()/w_total_abs:.1%}")

    # ------------------------------------------------------------------
    # 5. Save results and plots
    # ------------------------------------------------------------------
    print(f"\n[5/5] Saving results ...")

    # Save consensus W
    W_mean_t = torch.tensor(stats["W_mean"], dtype=torch.float32)
    torch.save(W_mean_t, os.path.join(OUTPUT_DIR, "W_ensemble_mean.pt"))
    print(f"  Saved W_ensemble_mean.pt  (shape {W_mean_t.shape})")

    W_stack_t = torch.tensor(W_stack, dtype=torch.float32)
    torch.save(W_stack_t, os.path.join(OUTPUT_DIR, "W_stack.pt"))
    print(f"  Saved W_stack.pt  (shape {W_stack_t.shape})")

    # Save std and identifiability mask
    torch.save(torch.tensor(stats["W_std"], dtype=torch.float32),
               os.path.join(OUTPUT_DIR, "W_std.pt"))
    torch.save(torch.tensor(mask_robust, dtype=torch.bool),
               os.path.join(OUTPUT_DIR, "mask_robust.pt"))
    torch.save(torch.tensor(mask_degen, dtype=torch.bool),
               os.path.join(OUTPUT_DIR, "mask_degen.pt"))

    # Save null-space directions (top components)
    np.save(os.path.join(OUTPUT_DIR, "null_directions_Vt.npy"), Vt)
    print(f"  Saved null_directions_Vt.npy  (shape {Vt.shape})")

    # JSON summary
    summary = {
        "config":          CONFIG_NAME,
        "n_seeds":         n_seeds,
        "n_edges":         int(E),
        "n_neurons":       n_neurons,
        "seed_ids":        seed_ids,
        "seed_r2":         [float(r) for r in stats["seed_r2"]],
        "mean_r2":         float(stats["mean_r2"]),
        "median_r2":       float(stats["med_r2"]),
        "improvement_vs_best": float(stats["mean_r2"] - max(stats["seed_r2"])),
        "per_edge_std_mean":   float(stats["W_std"].mean()),
        "per_edge_std_median": float(np.median(stats["W_std"])),
        "per_edge_std_p95":    float(np.percentile(stats["W_std"], 95)),
        "null_svd_singular_values": [float(s) for s in S[:n_seeds]],
        "null_svd_var_explained":   [float(v) for v in var_exp[:n_seeds]],
        "n_robust_edges":   int(mask_robust.sum()),
        "n_degen_edges":    int(mask_degen.sum()),
        "r2_robust_edges":  float(r2_robust),
        "r2_degen_edges":   float(r2_degen),
    }
    json_path = os.path.join(OUTPUT_DIR, "cv_ensemble_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved {json_path}")

    # Plots
    plot_ensemble(W_stack, gt_w, stats, seed_ids, var_exp, OUTPUT_DIR)
    plot_identifiability_map(stats["W_std"], gt_w, stats["W_mean"], edges,
                             n_neurons, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Seeds analysed:           {n_seeds}")
    print(f"  Per-seed R² range:        [{min(stats['seed_r2']):.4f}, {max(stats['seed_r2']):.4f}]")
    print(f"  Consensus mean R²:        {stats['mean_r2']:.4f}  "
          f"(+{stats['mean_r2'] - max(stats['seed_r2']):.4f} vs best seed)")
    print(f"  Robust edges (p25 std):   {mask_robust.sum():,} / {E:,}  "
          f"({mask_robust.mean()*100:.1f}%)  R²={r2_robust:.4f}")
    print(f"  Degenerate edges (p75):   {mask_degen.sum():,} / {E:,}  "
          f"({mask_degen.mean()*100:.1f}%)  R²={r2_degen:.4f}")
    print(f"  Null-space var (top-1):   {var_exp[0]*100:.1f}%")
    print(f"  Null-space var (top-2):   {var_exp[1]*100:.1f}%  (cumulative)")
    print(f"\n  Output saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
