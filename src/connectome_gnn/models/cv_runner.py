"""Cross-validation runner for connectome-GNN.

Runs generate + train + test + plot for a given config across N seeds,
then saves a summary log and a bar plot (mean ± SD, dots per seed) to
the cv00 log folder (e.g. log/<pre_folder>/<config_name>_cv00/).

Each CV fold uses a different simulation seed (for data generation) and a
different training seed (sim_seed + 1000), so data and model randomness are
independent.

Usage (run from repo root):
    python src/connectome_gnn/models/cv_runner.py flyvis_noise_005 --n_seeds 5
    python src/connectome_gnn/models/cv_runner.py flyvis_noise_005 --seeds 42,43,44,45,46
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


METRICS = [
    ('W_corrected_R2',   '$R^2$ conn ($W$)'),
    ('tau_R2',           '$R^2$ $\\tau$'),
    ('V_rest_R2',        '$R^2$ $V^{\\mathrm{rest}}$'),
    ('clustering_accuracy', 'Clustering acc.'),
]


def parse_metrics(path):
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


def _save_barplot(all_metrics, config_name, seeds, cv_out_dir, n_done):
    x = np.arange(len(METRICS))
    labels = [lbl for _, lbl in METRICS]
    means, sds = [], []
    for key, _ in METRICS:
        vals = [v for v in all_metrics[key] if not np.isnan(v)]
        means.append(np.mean(vals) if vals else 0.0)
        sds.append(np.std(vals) if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, means, yerr=sds, capsize=5, color='steelblue', alpha=0.7,
           error_kw=dict(elinewidth=1.5, ecolor='black'))

    rng = np.random.default_rng(0)
    for xi, (key, _) in enumerate(METRICS):
        vals = [v for v in all_metrics[key] if not np.isnan(v)]
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(xi + jitter, vals, color='black', s=30, zorder=5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('$R^2$ / accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'CV results — {config_name} ({n_done}/{len(seeds)} seeds done)')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    plt.tight_layout()

    plot_path = os.path.join(cv_out_dir, "cv_barplot.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  bar plot updated: {plot_path}")


def run_cv(config_name, seeds, retrain: bool = False):
    # Support absolute paths (e.g. configs stored outside the repo)
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

    # Summary goes into the base config's results folder
    cv_out_dir = os.path.join(log_path(pre_folder + base_name), "results")
    os.makedirs(cv_out_dir, exist_ok=True)

    all_metrics = {key: [] for key, _ in METRICS}

    for i, seed in enumerate(seeds):
        run_name = f"{base_name}_cv{i:02d}"
        sim_seed = seed               # simulation / data-generation seed
        train_seed = seed + 1000      # training seed (different from sim)
        print(f"\n\033[94mCV run {i+1}/{len(seeds)}  sim_seed={sim_seed}  train_seed={train_seed}  ({run_name})\033[0m")

        # Per-run dataset and log dir
        config = yaml_loader()
        config.simulation.seed = sim_seed       # each run generates its own data
        config.training.seed = train_seed       # different seed for training
        config.dataset = pre_folder + run_name  # per-run graphs_data dir
        config.config_file = pre_folder + run_name  # per-run log dir

        graphs_dir = graphs_data_path(config.dataset)

        # --- Generate ---
        # Always regenerate using the held-out video dataset (YouTube-VOS) so CV
        # data is never seen during training (which uses DAVIS or the config default).
        # erase=True ensures stale DAVIS data is replaced with YouTube-VOS data.
        config.simulation.datavis_roots = CV_DATAVIS_ROOTS
        config.simulation.skip_short_videos = CV_SKIP_SHORT_VIDEOS

        print(f"\033[96m  generating data (YouTube-VOS, erase=True) ...\033[0m")
        data_generate(config, device=device, visualize=False, run_vizualized=0,
                      style="color", alpha=1, erase=True, save=True, step=100)

        log_dir = log_path(config.config_file)

        # --- Train ---
        if retrain:
            print(f"\033[96m  training ...\033[0m")
            data_train(config, device=device, erase=True)
        else:
            print(f"\033[90m  skipping training (use -o cv_train to retrain)\033[0m")

        # --- Test ---
        print(f"\033[96m  testing ...\033[0m")
        data_test(config=config, visualize=True, style="color name continuous_slice",
                  verbose=False, best_model='best', run=0, step=10,
                  n_rollout_frames=250, device=device)

        # --- Plot / analyse ---
        print(f"\033[96m  analysing ...\033[0m")
        data_plot(config=config,
                  epoch_list=['best'], style='color', extended='plots',
                  device=device)

        # --- Collect metrics ---
        m = parse_metrics(os.path.join(log_dir, "results", "metrics.txt"))
        for key, _ in METRICS:
            val = m.get(key, float('nan'))
            all_metrics[key].append(val)
            print(f"\033[92m    {key}: {val:.4f}\033[0m" if not np.isnan(val) else f"\033[91m    {key}: —\033[0m")

        # --- Update bar plot after every run ---
        _save_barplot(all_metrics, config_name, seeds, cv_out_dir, n_done=i + 1)

        # --- Append per-run line to log immediately ---
        summary_path = os.path.join(cv_out_dir, "cv_summary.txt")
        with open(summary_path, 'a') as f:
            if i == 0:
                f.write(f"CV log: {config_name}\n")
                f.write(f"seeds (sim / train): {seeds} / {[s+1000 for s in seeds]}\n")
                f.write("=" * 80 + "\n")
                header = f"{'run':<6} {'sim':<6} {'train':<6}" + "".join(f" {k:<22}" for k, _ in METRICS)
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
            vals_str = "".join(
                f" {all_metrics[key][-1]:>22.4f}" if not np.isnan(all_metrics[key][-1])
                else f" {'—':>22}"
                for key, _ in METRICS
            )
            f.write(f"{i:<6} {sim_seed:<6} {train_seed:<6}{vals_str}\n")

    # --- Append summary statistics to log ---
    summary_path = os.path.join(cv_out_dir, "cv_summary.txt")
    with open(summary_path, 'a') as f:
        f.write("=" * 80 + "\n")
        f.write(f"{'Metric':<30} {'Mean':>8} {'SD':>8} {'CV%':>7} {'Min':>8} {'Max':>8}\n")
        f.write("-" * 80 + "\n")
        for key, _ in METRICS:
            vals = [v for v in all_metrics[key] if not np.isnan(v)]
            if vals:
                mean = np.mean(vals)
                sd = np.std(vals)
                cv_pct = (sd / mean * 100) if mean != 0 else float('nan')
                mn, mx = np.min(vals), np.max(vals)
                f.write(f"{key:<30} {mean:>8.4f} {sd:>8.4f} {cv_pct:>6.1f}% {mn:>8.4f} {mx:>8.4f}\n")
            else:
                f.write(f"{key:<30} {'—':>8} {'—':>8} {'—':>7} {'—':>8} {'—':>8}\n")
    # --- Append to persistent results_cv.txt ---
    results_cv_path = os.path.join(log_path(pre_folder + base_name), "results_cv.txt")
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sha = git_sha()

    # Resolve config YAML path
    if os.path.isabs(config_name) or os.path.isfile(config_name) or os.path.isfile(config_name + '.yaml'):
        config_yaml_path = config_name if config_name.endswith('.yaml') else config_name + '.yaml'
    else:
        config_yaml_path = config_path(f"{config_file}.yaml")
    config_yaml_path = os.path.abspath(config_yaml_path)
    try:
        config_mtime = datetime.datetime.fromtimestamp(
            os.path.getmtime(config_yaml_path)).strftime('%Y-%m-%d %H:%M:%S')
    except OSError:
        config_mtime = 'unknown'

    with open(results_cv_path, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"date:          {now_str}\n")
        f.write(f"git commit:    {sha}\n")
        f.write(f"config:        {config_yaml_path}  [{config_mtime}]\n")
        f.write(f"seeds:         {seeds}\n")
        # Per-fold best model paths
        for i, seed in enumerate(seeds):
            run_name = f"{base_name}_cv{i:02d}"
            model_dir = os.path.join(log_path(pre_folder + run_name), "models")
            candidates = sorted(_glob.glob(os.path.join(model_dir, "best_model_with_*.pt")))
            if candidates:
                best = candidates[-1]
                try:
                    mtime = datetime.datetime.fromtimestamp(
                        os.path.getmtime(best)).strftime('%Y-%m-%d %H:%M:%S')
                except OSError:
                    mtime = 'unknown'
                f.write(f"model[cv{i:02d}]:  {best}  [{mtime}]\n")
            else:
                f.write(f"model[cv{i:02d}]:  {model_dir}  [not found]\n")
        f.write(f"\n{'Metric':<30} {'Mean':>8} {'SD':>8}\n")
        f.write(f"{'-'*50}\n")
        for key, _ in METRICS:
            vals = [v for v in all_metrics[key] if not np.isnan(v)]
            if vals:
                mean, sd = np.mean(vals), np.std(vals)
                f.write(f"{key:<30} {mean:>8.4f} {sd:>8.4f}\n")
            else:
                f.write(f"{key:<30} {'—':>8} {'—':>8}\n")
    print(f"CV results:  {results_cv_path}")

    print(f"\nCV summary: {summary_path}")
    print(f"Bar plot:   {os.path.join(cv_out_dir, 'cv_barplot.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CV benchmark for connectome-GNN")
    parser.add_argument("config_name", help="Config name, e.g. flyvis_noise_005")
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of seeds (uses 42, 43, ..., 42+N-1)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed list, e.g. 42,43,44 (overrides --n_seeds)")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain from scratch on each fold (default: test only)")
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = list(range(42, 42 + args.n_seeds))

    run_cv(args.config_name, seeds, retrain=args.retrain)
