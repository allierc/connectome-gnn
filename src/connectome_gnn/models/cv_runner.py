"""Cross-validation runner for connectome-GNN.

Four-step pipeline per fold:
  1. Generate YouTube-VOS data (held-out, never seen during training)
  2. Train a new model on that YouTube-VOS data
  3. Test the ORIGINAL DAVIS-trained model on YouTube-VOS
     → measures zero-shot generalisation (rollout r, one-step r)
  4. Extract parameters from the re-trained YouTube-VOS model
     → measures parameter recovery when trained on unseen data
     (W R², tau R², V_rest R², clustering accuracy)

All results are appended to results_cv.txt (paper audit log).

Usage (run from repo root):
    python GNN_Main.py -o cv /path/to/config/flyvis_noise_005 --n_seeds 5
    python GNN_Main.py -o cv /path/to/config/flyvis_noise_005 --seeds 42,43,44
"""

import argparse
import datetime
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
from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.graph_data_generator import data_generate
from connectome_gnn.models.graph_trainer import data_test, data_train
from connectome_gnn.utils import add_pre_folder, config_path, git_sha, graphs_data_path, log_path, set_device

# Video dataset used for CV data generation (never seen during training).
# Must contain JPEGImages/480p/<video>/*.jpg
CV_DATAVIS_ROOTS = ["/groups/saalfeld/home/kumarv4/web_datasets/YouTube-VOS"]
CV_SKIP_SHORT_VIDEOS = False  # YouTube-VOS has many short clips


# Metrics from step 3: zero-shot generalisation (DAVIS model → YouTube-VOS data)
GENERALIZATION_METRICS = [
    ('one_step_r', 'One-step r (DAVIS→YT)'),
    ('rollout_r',  'Rollout r  (DAVIS→YT)'),
]

# Metrics from step 4: parameter recovery (model re-trained on YouTube-VOS)
RECOVERY_METRICS = [
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
    n_total = n_gen + n_rec

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

def run_cv(config_name, seeds):
    """Run the 4-step CV pipeline for all seeds and append to results_cv.txt."""

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

    all_metrics = {key: [] for key, _ in ALL_METRICS}

    for i, seed in enumerate(seeds):
        run_name = f"{base_name}_cv{i:02d}"
        sim_seed   = seed
        train_seed = seed + 1000
        print(f"\n\033[94m{'='*70}\033[0m")
        print(f"\033[94mCV fold {i+1}/{len(seeds)}  sim_seed={sim_seed}  train_seed={train_seed}  ({run_name})\033[0m")
        print(f"\033[94m  Steps: [1] generate YouTube-VOS  [2] train on YT  [3] DAVIS→YT rollout  [4] YT model params\033[0m")

        fold_config = yaml_loader()
        fold_config.simulation.seed  = sim_seed
        fold_config.training.seed    = train_seed
        fold_config.dataset          = pre_folder + run_name
        fold_config.config_file      = pre_folder + run_name
        fold_config.simulation.datavis_roots    = CV_DATAVIS_ROOTS
        fold_config.simulation.skip_short_videos = CV_SKIP_SHORT_VIDEOS

        fold_log_dir  = log_path(fold_config.config_file)
        base_log_dir  = log_path(pre_folder + base_name)

        # ---------------------------------------------------------------
        # Step 1: Generate YouTube-VOS data (skip if already from YouTube-VOS)
        # ---------------------------------------------------------------
        graphs_dir = graphs_data_path(fold_config.dataset)
        if _yt_data_exists(graphs_dir):
            print(f"\033[90m  [1/4] YouTube-VOS data already exists — skipping generation\033[0m")
        else:
            print(f"\033[96m  [1/4] generating YouTube-VOS data ...\033[0m")
            data_generate(fold_config, device=device, visualize=False, run_vizualized=0,
                          style="color", alpha=1, erase=True, save=True, step=100)

        # ---------------------------------------------------------------
        # Step 2: Train new model on YouTube-VOS data (skip if model exists)
        # ---------------------------------------------------------------
        model_dir = os.path.join(fold_log_dir, "models")
        model_exists = (os.path.isdir(model_dir) and
                        any(f.startswith("best_model") for f in os.listdir(model_dir))
                        ) if os.path.isdir(model_dir) else False
        if model_exists:
            print(f"\033[90m  [2/4] trained model already exists — skipping training\033[0m")
        else:
            print(f"\033[96m  [2/4] training on YouTube-VOS data ...\033[0m")
            data_train(fold_config, device=device, erase=True)

        # ---------------------------------------------------------------
        # Step 3: Generalisation — DAVIS model tested on YouTube-VOS data
        # ---------------------------------------------------------------
        print(f"\033[96m  [3/4] generalisation test (DAVIS model → YouTube-VOS) ...\033[0m")
        davis_config = yaml_loader()
        davis_config.config_file = pre_folder + base_name  # points to the trained DAVIS model
        data_test(config=davis_config, visualize=False, best_model='best', run=0,
                  step=10, n_rollout_frames=250, device=device,
                  test_config=fold_config)   # test data from YouTube-VOS fold

        # Determine the test_suffix that data_test_gnn used
        test_ds_short = fold_config.dataset.replace('flyvis_', '').replace('fly/', '')
        test_suffix   = f'_on_{test_ds_short}'
        one_step_r = parse_pearson_from_log(
            os.path.join(base_log_dir, f'results_test{test_suffix}.log'))
        rollout_r  = parse_pearson_from_log(
            os.path.join(base_log_dir, f'results_rollout{test_suffix}.log'))
        all_metrics['one_step_r'].append(one_step_r)
        all_metrics['rollout_r'].append(rollout_r)
        print(f"\033[92m    generalisation — one_step_r={one_step_r:.4f}  rollout_r={rollout_r:.4f}\033[0m")

        # ---------------------------------------------------------------
        # Step 4: Parameter recovery — analyse re-trained YouTube-VOS model
        # (data_plot only: W R², tau R², V_rest R², clustering from model params)
        # ---------------------------------------------------------------
        print(f"\033[96m  [4/4] extracting parameters from re-trained YouTube-VOS model ...\033[0m")
        data_plot(config=fold_config, epoch_list=['best'], style='color', extended='plots',
                  device=device, apply_weight_correction=True, skip_svd=True)

        m = parse_metrics(os.path.join(fold_log_dir, "results", "metrics.txt"))
        for key, _ in RECOVERY_METRICS:
            val = m.get(key, float('nan'))
            all_metrics[key].append(val)
            if not np.isnan(val):
                print(f"\033[92m    {key}: {val:.4f}\033[0m")
            else:
                print(f"\033[91m    {key}: —\033[0m")

        # ---------------------------------------------------------------
        # Update bar plot and per-run summary after each fold
        # ---------------------------------------------------------------
        _save_barplot(all_metrics, config_name, seeds, cv_out_dir, n_done=i + 1)

        summary_path = os.path.join(cv_out_dir, "cv_summary.txt")
        with open(summary_path, 'a') as f:
            if i == 0:
                f.write(f"CV log: {config_name}\n")
                f.write(f"seeds (sim / train): {seeds} / {[s+1000 for s in seeds]}\n")
                f.write("=" * 90 + "\n")
                header = f"{'fold':<6} {'sim':<6} {'train':<6}" + \
                         "".join(f" {k:<22}" for k, _ in ALL_METRICS)
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
            vals_str = "".join(
                f" {all_metrics[key][-1]:>22.4f}" if not np.isnan(all_metrics[key][-1])
                else f" {'—':>22}"
                for key, _ in ALL_METRICS
            )
            f.write(f"{i:<6} {sim_seed:<6} {train_seed:<6}{vals_str}\n")

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

    # DAVIS base model (used in step 3)
    davis_model_candidates = sorted(_glob.glob(
        os.path.join(base_log_dir, "models", "best_model_with_*.pt")))
    davis_model = davis_model_candidates[-1] if davis_model_candidates else f"{base_log_dir}/models/ [not found]"

    with open(results_cv_path, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"date:             {now_str}\n")
        f.write(f"git commit:       {sha}\n")
        f.write(f"config:           {config_yaml_path}  [{_mtime_str(config_yaml_path)}]\n")
        f.write(f"cv_datavis:       {CV_DATAVIS_ROOTS[0]}\n")
        f.write(f"seeds (sim/train):{seeds} / {[s+1000 for s in seeds]}\n")
        f.write(f"\n-- DAVIS model (step 3: zero-shot generalisation) --\n")
        f.write(f"davis_model:      {davis_model}  [{_mtime_str(davis_model)}]\n")
        f.write(f"\n-- Re-trained YouTube-VOS models (step 4: parameter recovery) --\n")
        for i, seed in enumerate(seeds):
            run_name  = f"{base_name}_cv{i:02d}"
            model_dir = os.path.join(log_path(pre_folder + run_name), "models")
            candidates = sorted(_glob.glob(os.path.join(model_dir, "best_model_with_*.pt")))
            best = candidates[-1] if candidates else f"{model_dir} [not found]"
            f.write(f"model[cv{i:02d}]:      {best}  [{_mtime_str(best)}]\n")
        f.write(f"\n{'Metric':<35} {'Mean':>8} {'SD':>8}   group\n")
        f.write(f"{'-'*65}\n")
        for key, label in GENERALIZATION_METRICS:
            vals = [v for v in all_metrics[key] if not np.isnan(v)]
            if vals:
                f.write(f"{key:<35} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}   generalisation\n")
            else:
                f.write(f"{key:<35} {'—':>8} {'—':>8}   generalisation\n")
        for key, label in RECOVERY_METRICS:
            vals = [v for v in all_metrics[key] if not np.isnan(v)]
            if vals:
                f.write(f"{key:<35} {np.mean(vals):>8.4f} {np.std(vals):>8.4f}   parameter recovery\n")
            else:
                f.write(f"{key:<35} {'—':>8} {'—':>8}   parameter recovery\n")

    print(f"\nCV results (audit): {results_cv_path}")
    print(f"CV summary:         {summary_path}")
    print(f"Bar plot:           {os.path.join(cv_out_dir, 'cv_barplot.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV benchmark for connectome-GNN")
    parser.add_argument("config_name", help="Config name or absolute YAML path")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of seeds (uses 42, 43, ..., 42+N-1)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list, e.g. 42,43,44 (overrides --n_seeds)")
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = list(range(42, 42 + args.n_seeds))

    run_cv(args.config_name, seeds)
