"""Cross-validation runner for connectome-GNN.

Three-phase pipeline (run across all seeds before advancing):
  Phase 1 — Generate YouTube-VOS data for every fold.
  Phase 2 — Zero-shot generalisation: test the ORIGINAL DAVIS-trained model on
             each YouTube-VOS fold  (one_step_r, rollout_r).
  Phase 3 — Train a new model on each fold then run rollout + parameter
             extraction  (yt_one_step_r, yt_rollout_r, W R², tau R², …).

All results are appended to results_cv.txt (paper audit log).

Usage (run from repo root):
    python GNN_Main.py -o cv /path/to/config/flyvis_noise_005 --n_seeds 5
    python GNN_Main.py -o cv /path/to/config/flyvis_noise_005 --seeds 42,43,44
"""

import argparse
import datetime
import gc
import glob as _glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# repo root: src/connectome_gnn/models/ -> src/connectome_gnn/ -> src/ -> repo root
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, sys_path)

from GNN_PlotFigure import data_plot
import torch

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.graph_data_generator import data_generate
from connectome_gnn.models.graph_trainer import data_test, data_train
from connectome_gnn.utils import add_pre_folder, config_path, git_sha, graphs_data_path, log_path, set_device

def _free_gpu():
    """Release GPU memory and CUDA Graph pools between CV folds."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch._dynamo.reset()


# Video dataset used for CV data generation (never seen during training).
# Must contain JPEGImages/480p/<video>/*.jpg
CV_DATAVIS_ROOTS = ["/groups/saalfeld/home/kumarv4/web_datasets/YouTube-VOS"]
CV_SKIP_SHORT_VIDEOS = False  # YouTube-VOS has many short clips


# Metrics from phase 2: zero-shot generalisation (DAVIS model → YouTube-VOS data)
GENERALIZATION_METRICS = [
    ('one_step_r', 'One-step r (DAVIS→YT)'),
    ('rollout_r',  'Rollout r  (DAVIS→YT)'),
]

# Metrics from phase 3: rollout + parameter recovery (model re-trained on YouTube-VOS)
RECOVERY_METRICS = [
    ('yt_one_step_r',     'One-step r (YT model→YT)'),
    ('yt_rollout_r',      'Rollout r  (YT model→YT)'),
    ('W_corrected_R2',    '$R^2$ $W$ (re-train YT)'),
    ('tau_R2',            '$R^2$ $\\tau$ (re-train YT)'),
    ('V_rest_R2',         '$R^2$ $V^{\\mathrm{rest}}$ (re-train YT)'),
    ('clustering_accuracy', 'Clustering acc (re-train YT)'),
]

ALL_METRICS = GENERALIZATION_METRICS + RECOVERY_METRICS


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_metrics(path):
    """Parse a key: value text file (e.g. metrics.txt from data_plot)."""
    metrics = {}
    if not os.path.isfile(path):
        return metrics
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    pass
    return metrics


def _yt_data_exists(graphs_dir):
    """Return True if graphs_dir has YouTube-VOS data already generated."""
    if not os.path.isdir(os.path.join(graphs_dir, 'x_list_train')):
        return False
    log = os.path.join(graphs_dir, 'generation_log.txt')
    if not os.path.isfile(log):
        return False
    with open(log) as f:
        content = f.read()
    return any(root in content for root in CV_DATAVIS_ROOTS)


def parse_pearson_from_log(path):
    """Return the first 'Pearson r: <value>' found in a .log file."""
    if not os.path.isfile(path):
        return float('nan')
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('Pearson r:'):
                try:
                    return float(line.split(':')[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
    return float('nan')


def _mtime_str(path):
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
    except OSError:
        return 'not found'


# ---------------------------------------------------------------------------
# Bar plot
# ---------------------------------------------------------------------------

def _save_barplot(all_metrics, config_name, seeds, cv_out_dir, n_done):
    n_gen = len(GENERALIZATION_METRICS)
    n_rec = len(RECOVERY_METRICS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             gridspec_kw={'width_ratios': [n_gen, n_rec]})

    rng = np.random.default_rng(0)
    for ax, metric_group, title in [
        (axes[0], GENERALIZATION_METRICS, 'Generalisation (DAVIS→YouTube-VOS)'),
        (axes[1], RECOVERY_METRICS,       'Parameter recovery (re-trained on YouTube-VOS)'),
    ]:
        x = np.arange(len(metric_group))
        means, sds = [], []
        for key, _ in metric_group:
            vals = [v for v in all_metrics[key] if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0.0)
            sds.append(np.std(vals) if len(vals) > 1 else 0.0)

        ax.bar(x, means, yerr=sds, capsize=5, color='steelblue', alpha=0.7,
               error_kw=dict(elinewidth=1.5, ecolor='black'))
        for xi, (key, _) in enumerate(metric_group):
            vals = [v for v in all_metrics[key] if not np.isnan(v)]
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(xi + jitter, vals, color='black', s=30, zorder=5, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in metric_group], fontsize=9)
        ax.set_ylabel('value')
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)

    fig.suptitle(f'CV — {config_name}  ({n_done}/{len(seeds)} seeds)', fontsize=11)
    plt.tight_layout()

    plot_path = os.path.join(cv_out_dir, "cv_barplot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  bar plot updated: {plot_path}")


# ---------------------------------------------------------------------------
# Main CV runner
# ---------------------------------------------------------------------------

def run_cv(config_name, seeds, skip_phase2=False):
    """Run the 3-phase CV pipeline for all seeds and append to results_cv.txt.

    Args:
        config_name:  Config name or absolute YAML path.
        seeds:        List of simulation seeds.
        skip_phase2:  If True, skip phase 2 (zero-shot DAVIS→YouTube test).
                      When False and no pre-trained DAVIS model exists, phase 2
                      will first generate DAVIS data (if missing) and train the
                      base DAVIS model, then run the zero-shot rollout.
    """

    # Resolve config path and loader
    if os.path.isabs(config_name) or os.path.isfile(config_name) or os.path.isfile(config_name + '.yaml'):
        yaml_file = config_name if config_name.endswith('.yaml') else config_name + '.yaml'
        parent = os.path.basename(os.path.dirname(os.path.abspath(yaml_file)))
        pre_folder = parent + "/" if parent else ""
        base_name = os.path.splitext(os.path.basename(yaml_file))[0]
        config_file = base_name
        yaml_loader = lambda: NeuralGraphConfig.from_yaml(yaml_file)
    else:
        config_file, pre_folder = add_pre_folder(config_name)
        base_name = os.path.basename(config_name)
        yaml_loader = lambda: NeuralGraphConfig.from_yaml(config_path(f"{config_file}.yaml"))

    base_config = yaml_loader()
    device = set_device(base_config.training.device)

    cv_out_dir = os.path.join(log_path(pre_folder + base_name), "results")
    os.makedirs(cv_out_dir, exist_ok=True)

    base_log_dir = log_path(pre_folder + base_name)

    all_metrics = {key: [] for key, _ in ALL_METRICS}

    # Build per-fold configs once — reused across all phases
    fold_configs = []
    for i, seed in enumerate(seeds):
        run_name = f"{base_name}_cv{i:02d}"
        fc = yaml_loader()
        fc.simulation.seed           = seed
        fc.training.seed             = seed + 1000
        fc.dataset                   = pre_folder + run_name
        fc.config_file               = pre_folder + run_name
        fc.simulation.datavis_roots  = CV_DATAVIS_ROOTS
        fc.simulation.skip_short_videos = CV_SKIP_SHORT_VIDEOS
        fold_configs.append(fc)

    # ===================================================================
    # PHASE 1 — Generate YouTube-VOS data for all folds
    # ===================================================================
    print(f"\n\033[94m{'='*70}\033[0m")
    print(f"\033[94mPHASE 1/3 — Generating YouTube-VOS data for all {len(seeds)} folds\033[0m")
    print(f"\033[94m{'='*70}\033[0m")
    for i, (seed, fold_config) in enumerate(zip(seeds, fold_configs)):
        graphs_dir = graphs_data_path(fold_config.dataset)
        if _yt_data_exists(graphs_dir):
            print(f"\033[90m  fold {i+1}/{len(seeds)} (seed={seed}) — YouTube-VOS data already exists, skipping\033[0m")
        else:
            print(f"\033[96m  fold {i+1}/{len(seeds)} (seed={seed}) — generating YouTube-VOS data ...\033[0m")
            data_generate(fold_config, device=device, visualize=False, run_vizualized=0,
                          style="color", alpha=1, erase=True, save=True, step=100)
            print(f"\033[92m  fold {i+1}/{len(seeds)} — generation done\033[0m")

    # ===================================================================
    # PHASE 2 — Zero-shot generalisation: DAVIS model → YouTube-VOS folds
    # ===================================================================
    # Auto-detect whether a DAVIS model exists; if not, train one (unless skipped)
    davis_models_dir = os.path.join(base_log_dir, 'models')
    davis_model_exists = os.path.isdir(davis_models_dir) and any(
        f.endswith('.pt') for f in os.listdir(davis_models_dir)
    )
    if skip_phase2:
        print(f"\n\033[93m{'='*70}\033[0m")
        print(f"\033[93mPHASE 2/3 — SKIPPED (--skip_phase2)\033[0m")
        print(f"\033[93m{'='*70}\033[0m")
    else:
        if not davis_model_exists:
            print(f"\n\033[94m{'='*70}\033[0m")
            print(f"\033[94mPHASE 2/3 — No DAVIS base model found; training one first\033[0m")
            print(f"\033[94m{'='*70}\033[0m")
            base_train_config = yaml_loader()
            base_train_config.dataset     = pre_folder + base_config.dataset
            base_train_config.config_file = pre_folder + base_name
            davis_graphs_dir = graphs_data_path(base_train_config.dataset)
            if not os.path.isdir(os.path.join(davis_graphs_dir, 'x_list_train')):
                print(f"\033[96m  generating DAVIS training data ...\033[0m")
                data_generate(base_train_config, device=device, visualize=False, run_vizualized=0,
                              style="color", alpha=1, erase=True, save=True, step=100)
            print(f"\033[96m  training DAVIS base model ...\033[0m")
            data_train(base_train_config, device=device, erase=True)
            _free_gpu()
            davis_model_exists = True

        print(f"\n\033[94m{'='*70}\033[0m")
        print(f"\033[94mPHASE 2/3 — Zero-shot rollout (DAVIS model → YouTube-VOS) for all {len(seeds)} folds\033[0m")
        print(f"\033[94m{'='*70}\033[0m")
        for i, (seed, fold_config) in enumerate(zip(seeds, fold_configs)):
            print(f"\033[96m  fold {i+1}/{len(seeds)} (seed={seed}) — testing DAVIS model on YouTube-VOS fold ...\033[0m")
            davis_config = yaml_loader()
            davis_config.config_file = pre_folder + base_name  # load DAVIS trained model
            davis_config.dataset     = pre_folder + base_config.dataset  # ensure fly/ prefix
            data_test(config=davis_config, visualize=False, best_model='best', run=0,
                      step=10, n_rollout_frames=250, device=device,
                      test_config=fold_config)   # test data from YouTube-VOS fold

            # Determine test_suffix used by data_test_gnn
            test_ds_short = fold_config.dataset.replace('flyvis_', '').replace('fly/', '')
            test_suffix   = f'_on_{test_ds_short}'
            one_step_r = parse_pearson_from_log(
                os.path.join(base_log_dir, f'results_test{test_suffix}.log'))
            rollout_r  = parse_pearson_from_log(
                os.path.join(base_log_dir, f'results_rollout{test_suffix}.log'))
            all_metrics['one_step_r'].append(one_step_r)
            all_metrics['rollout_r'].append(rollout_r)
            # Pad recovery metrics so lengths stay consistent for bar plot
            for key, _ in RECOVERY_METRICS:
                if len(all_metrics[key]) < i + 1:
                    all_metrics[key].append(float('nan'))
            print(f"\033[92m    one_step_r={one_step_r:.4f}  rollout_r={rollout_r:.4f}\033[0m")

            _save_barplot(all_metrics, config_name, seeds, cv_out_dir, n_done=i + 1)

    # ===================================================================
    # PHASE 3 — Train new models + parameter extraction for all folds
    # ===================================================================
    print(f"\n\033[94m{'='*70}\033[0m")
    print(f"\033[94mPHASE 3/3 — Training + parameter extraction for all {len(seeds)} folds\033[0m")
    print(f"\033[94m{'='*70}\033[0m")
    # Reset recovery metrics (were padded with nan during phase 2)
    for key, _ in RECOVERY_METRICS:
        all_metrics[key] = []

    for i, (seed, fold_config) in enumerate(zip(seeds, fold_configs)):
        fold_log_dir = log_path(fold_config.config_file)

        print(f"\033[96m  fold {i+1}/{len(seeds)} (seed={seed}) — training on YouTube-VOS data ...\033[0m")
        data_train(fold_config, device=device, erase=True)
        _free_gpu()
        print(f"\033[92m  fold {i+1}/{len(seeds)} — training done\033[0m")

        print(f"\033[96m  fold {i+1}/{len(seeds)} (seed={seed}) — rollout test + parameter extraction ...\033[0m")
        data_test(config=fold_config, visualize=False, best_model='best', run=0,
                  step=10, n_rollout_frames=250, device=device)
        data_plot(config=fold_config, epoch_list=['best'], style='color', extended='plots',
                  device=device, apply_weight_correction=True, skip_svd=True)
        _free_gpu()

        # Parse rollout metrics (no test_suffix: fold model tested on its own data)
        yt_one_step_r = parse_pearson_from_log(os.path.join(fold_log_dir, 'results_test.log'))
        yt_rollout_r  = parse_pearson_from_log(os.path.join(fold_log_dir, 'results_rollout.log'))
        all_metrics['yt_one_step_r'].append(yt_one_step_r)
        all_metrics['yt_rollout_r'].append(yt_rollout_r)
        print(f"\033[92m    yt model — one_step_r={yt_one_step_r:.4f}  rollout_r={yt_rollout_r:.4f}\033[0m")

        m = parse_metrics(os.path.join(fold_log_dir, "results", "metrics.txt"))
        for key, _ in RECOVERY_METRICS:
            if key in ('yt_one_step_r', 'yt_rollout_r'):
                continue  # already collected above
            val = m.get(key, float('nan'))
            all_metrics[key].append(val)
            if not np.isnan(val):
                print(f"\033[92m    {key}: {val:.4f}\033[0m")
            else:
                print(f"\033[91m    {key}: —\033[0m")

        _save_barplot(all_metrics, config_name, seeds, cv_out_dir, n_done=i + 1)

        # Append per-fold row to cv_summary.txt
        summary_path = os.path.join(cv_out_dir, "cv_summary.txt")
        with open(summary_path, 'a') as f:
            if i == 0:
                f.write(f"\nCV log: {config_name}\n")
                f.write(f"seeds (sim / train): {seeds} / {[s+1000 for s in seeds]}\n")
                f.write("=" * 90 + "\n")
                header = f"{'fold':<6} {'sim':<6} {'train':<6}" + \
                         "".join(f" {k:<22}" for k, _ in ALL_METRICS)
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
            gen_r1 = all_metrics['one_step_r'][i] if i < len(all_metrics['one_step_r']) else float('nan')
            gen_r2 = all_metrics['rollout_r'][i]   if i < len(all_metrics['rollout_r'])   else float('nan')
            row_vals = [gen_r1, gen_r2] + [
                all_metrics[k][-1] for k, _ in RECOVERY_METRICS
            ]
            vals_str = "".join(
                f" {v:>22.4f}" if not np.isnan(v) else f" {'—':>22}"
                for v in row_vals
            )
            f.write(f"{i:<6} {seed:<6} {seed+1000:<6}{vals_str}\n")

    # -------------------------------------------------------------------
    # Final summary statistics → cv_summary.txt
    # -------------------------------------------------------------------
    summary_path = os.path.join(cv_out_dir, "cv_summary.txt")
    with open(summary_path, 'a') as f:
        f.write("=" * 90 + "\n")
        f.write(f"{'Metric':<30} {'Mean':>8} {'SD':>8} {'CV%':>7} {'Min':>8} {'Max':>8}\n")
        f.write("-" * 70 + "\n")
        for key, _ in ALL_METRICS:
            vals = [v for v in all_metrics[key] if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                sd   = np.std(vals)
                cv_pct = (sd / mean * 100) if mean != 0 else float('nan')
                f.write(f"{key:<30} {mean:>8.4f} {sd:>8.4f} {cv_pct:>6.1f}% "
                        f"{np.min(vals):>8.4f} {np.max(vals):>8.4f}\n")
            else:
                f.write(f"{key:<30} {'—':>8} {'—':>8} {'—':>7} {'—':>8} {'—':>8}\n")

    # -------------------------------------------------------------------
    # Append full audit block to results_cv.txt (paper traceability)
    # -------------------------------------------------------------------
    results_cv_path = os.path.join(log_path(pre_folder + base_name), "results_cv.txt")
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sha = git_sha()

    # Resolve config YAML path and mtime
    if os.path.isabs(config_name) or os.path.isfile(config_name) or os.path.isfile(config_name + '.yaml'):
        config_yaml_path = os.path.abspath(config_name if config_name.endswith('.yaml') else config_name + '.yaml')
    else:
        config_yaml_path = os.path.abspath(config_path(f"{config_file}.yaml"))

    # DAVIS base model (used in phase 2)
    davis_model_candidates = sorted(_glob.glob(
        os.path.join(base_log_dir, "models", "best_model_with_*.pt")))
    davis_model = davis_model_candidates[-1] if davis_model_candidates else f"{base_log_dir}/models/ [not found]"

    # Build audit content once — written to both per-config and master files
    audit_lines = []
    audit_lines.append(f"\n{'='*80}\n")
    audit_lines.append(f"date:             {now_str}\n")
    audit_lines.append(f"git commit:       {sha}\n")
    audit_lines.append(f"config:           {config_yaml_path}  [{_mtime_str(config_yaml_path)}]\n")
    audit_lines.append(f"cv_datavis:       {CV_DATAVIS_ROOTS[0]}\n")
    audit_lines.append(f"seeds (sim/train):{seeds} / {[s+1000 for s in seeds]}\n")
    audit_lines.append(f"\n-- DAVIS model (phase 2: zero-shot generalisation) --\n")
    audit_lines.append(f"davis_model:      {davis_model}  [{_mtime_str(davis_model)}]\n")
    audit_lines.append(f"\n-- Re-trained YouTube-VOS models (phase 3: parameter recovery) --\n")
    for i, seed in enumerate(seeds):
        run_name  = f"{base_name}_cv{i:02d}"
        model_dir = os.path.join(log_path(pre_folder + run_name), "models")
        candidates = sorted(_glob.glob(os.path.join(model_dir, "best_model_with_*.pt")))
        best = candidates[-1] if candidates else f"{model_dir} [not found]"
        audit_lines.append(f"model[cv{i:02d}]:      {best}  [{_mtime_str(best)}]\n")
    audit_lines.append(f"\n{'Metric':<35} {'Mean':>8} {'SD':>8}   group\n")
    audit_lines.append(f"{'-'*65}\n")
    for key, label in GENERALIZATION_METRICS:
        vals = [v for v in all_metrics[key] if not np.isnan(v)]
        if vals:
            audit_lines.append(f"{key:<35} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}   generalisation\n")
        else:
            audit_lines.append(f"{key:<35} {'—':>8} {'—':>8}   generalisation\n")
    for key, label in RECOVERY_METRICS:
        vals = [v for v in all_metrics[key] if not np.isnan(v)]
        if vals:
            audit_lines.append(f"{key:<35} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}   parameter recovery\n")
        else:
            audit_lines.append(f"{key:<35} {'—':>8} {'—':>8}   parameter recovery\n")
    audit_content = "".join(audit_lines)

    # Per-config results file
    with open(results_cv_path, 'a') as f:
        f.write(audit_content)

    # Master results file at {data_root}/log/results_cv.txt — collects all configs
    master_cv_path = log_path("results_cv.txt")
    with open(master_cv_path, 'a') as f:
        f.write(audit_content)

    print(f"\nCV results (audit): {results_cv_path}")
    print(f"CV results (master): {master_cv_path}")
    print(f"CV summary:         {summary_path}")
    print(f"Bar plot:           {os.path.join(cv_out_dir, 'cv_barplot.png')}")


# ---------------------------------------------------------------------------
# Comparison table across multiple conditions
# ---------------------------------------------------------------------------

# Metrics shown in the comparison table (Phase 3 only — no DAVIS model needed)
COMPARISON_METRICS = [
    ('yt_one_step_r',     'One-step r'),
    ('yt_rollout_r',      'Rollout r'),
    ('W_corrected_R2',    'W R²'),
    ('tau_R2',            'τ R²'),
    ('V_rest_R2',         'V_rest R²'),
    ('clustering_accuracy', 'Clust. acc'),
]


def _parse_cv_summary_stats(summary_path):
    """Parse mean and SD for each metric from the stats block of cv_summary.txt."""
    stats = {}
    if not os.path.isfile(summary_path):
        return stats
    in_stats = False
    with open(summary_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('Metric') and 'Mean' in line:
                in_stats = True
                continue
            if in_stats and line.startswith('-'):
                continue
            if in_stats and line.startswith('='):
                break
            if in_stats and line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    key = parts[0]
                    try:
                        mean = float(parts[1]) if parts[1] != '—' else float('nan')
                        sd   = float(parts[2]) if parts[2] != '—' else float('nan')
                        stats[key] = (mean, sd)
                    except ValueError:
                        pass
    return stats


def compare_cv_results(condition_labels, config_names, pre_folder='fly/', output_path=None):
    """Print a comparison table across multiple CV conditions.

    Args:
        condition_labels: List of short display names (table row labels).
        config_names:     List of base config names (matching the log dir names).
        pre_folder:       Pre-folder prefix (default 'fly/').
        output_path:      If given, also write the table to this file.
    """
    rows = []
    for label, cfg in zip(condition_labels, config_names):
        summary_path = os.path.join(log_path(pre_folder + cfg), "results", "cv_summary.txt")
        stats = _parse_cv_summary_stats(summary_path)
        rows.append((label, stats))

    # Build table
    col_w = 18
    header = f"{'Condition':<28}" + "".join(f"{lbl:>{col_w}}" for _, lbl in COMPARISON_METRICS)
    sep    = "-" * len(header)
    lines  = [sep, header, sep]
    for label, stats in rows:
        row = f"{label:<28}"
        for key, _ in COMPARISON_METRICS:
            if key in stats:
                mean, sd = stats[key]
                if not np.isnan(mean):
                    row += f"{'%.3f±%.3f' % (mean, sd):>{col_w}}"
                else:
                    row += f"{'—':>{col_w}}"
            else:
                row += f"{'—':>{col_w}}"
        lines.append(row)
    lines.append(sep)
    table = "\n".join(lines)

    print(f"\n{'='*80}")
    print("CV COMPARISON TABLE")
    print(f"{'='*80}")
    print(table)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'a') as f:
            f.write(f"\n{table}\n")
        print(f"Table appended to: {output_path}")

    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV benchmark for connectome-GNN")
    parser.add_argument("config_name", help="Config name or absolute YAML path")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of seeds (uses 42, 43, ..., 42+N-1)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list, e.g. 42,43,44 (overrides --n_seeds)")
    parser.add_argument("--skip_phase2", action="store_true", default=False,
                        help="Skip phase 2 (zero-shot DAVIS→YouTube test)")
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = list(range(42, 42 + args.n_seeds))

    run_cv(args.config_name, seeds, skip_phase2=args.skip_phase2)
