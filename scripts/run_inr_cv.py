"""
Unified INR CV driver — clean naming + robust caching.

For each seed in 42..42+N-1, produces two independent training folds per seed:

    DAVIS       fold -> log/<pre>/<base>_davis_cv<i>/     (stimuli: DAVIS)
    YouTube-VOS fold -> log/<pre>/<base>_yt_cv<i>/        (stimuli: YouTube-VOS)

Each fold pipeline (generate -> train -> test -> plot) is skipped per-step
when its output already exists, so re-running only fills what is missing.

Caching rules (per fold):
  data_generate   skipped if <graphs_data>/x_list_train exists
  data_train      skipped if <fold>/models/best_model_with_*.pt exists
  data_test       skipped if <fold>/results_rollout.log exists
  data_plot       skipped if <fold>/results/metrics.txt exists

Row 1 of tab:cv_inr (DAVIS)       aggregates the *_davis_cv{0..N} folds.
Row 2 of tab:cv_inr (YouTube-VOS) aggregates the *_yt_cv{0..N}    folds.

Usage:
    python scripts/run_inr_cv.py \\
        --config flyvis_noise_005_INR \\
        --output_root /groups/saalfeld/home/allierc/GraphData \\
        --n_seeds 5 \\
        [--conditions davis yt]
"""

import argparse
import glob
import os
import sys

# Resolve the repo root from THIS script's location, not a hardcoded path.
# The script lives at <repo>/scripts/run_inr_cv.py; repo = parent of scripts/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
# Put local repo at sys.path[0] so `import connectome_gnn` binds here, not
# to any editable install that might be in the conda env (e.g. GraphDebug).
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
import connectome_gnn.utils as _cg_utils  # noqa: E402
from connectome_gnn.utils import (  # noqa: E402
    add_pre_folder, config_path, log_path, set_device, graphs_data_path,
)
try:
    from connectome_gnn.utils import set_data_root  # newer version
except ImportError:
    # older connectome_gnn (e.g. GraphDebug) lacks set_data_root —
    # poke the module-level variable directly.
    def set_data_root(path):
        _cg_utils._data_root = path
from connectome_gnn.generators.graph_data_generator import data_generate  # noqa: E402
from connectome_gnn.models.graph_trainer import data_train, data_test  # noqa: E402
try:
    from connectome_gnn.models.cv_runner import CV_DATAVIS_ROOTS
except ImportError:
    # Older connectome_gnn (e.g. cluster GraphDebug) lacks this constant.
    CV_DATAVIS_ROOTS = ["/groups/saalfeld/home/kumarv4/web_datasets/YouTube-VOS"]
from GNN_PlotFigure import data_plot  # noqa: E402


def _have_data(graphs_dir):
    return os.path.isdir(os.path.join(graphs_dir, 'x_list_train'))


def _have_model(fold_log_dir):
    return bool(glob.glob(os.path.join(
        fold_log_dir, 'models', 'best_model_with_*.pt')))


def _have_test(fold_log_dir):
    return os.path.exists(os.path.join(fold_log_dir, 'results_rollout.log'))


def _have_plot(fold_log_dir):
    return os.path.exists(os.path.join(fold_log_dir, 'results', 'metrics.txt'))


def _fold_name(base_name, condition, i):
    return f'{base_name}_{condition}_cv{i:02d}'


def _make_fold_config(base_yaml, pre_folder, run_name, seed, condition):
    fc = NeuralGraphConfig.from_yaml(base_yaml)
    fc.simulation.seed = seed
    fc.training.seed = seed + 1000
    fc.dataset = pre_folder + run_name
    fc.config_file = pre_folder + run_name
    if condition == 'yt':
        fc.simulation.datavis_roots = CV_DATAVIS_ROOTS
    # condition == 'davis' -> leave datavis_roots at YAML default (DAVIS).
    # Include short sequences; otherwise YT-VOS floods the log with
    # per-sequence "too short" warnings (~3/4 of its ~4500 videos).
    fc.simulation.skip_short_videos = False
    return fc


def run_fold(i, seed, condition, base_yaml, pre_folder, base_name, device):
    run_name = _fold_name(base_name, condition, i)
    print(f'\n--- fold {i} (seed={seed}) — condition={condition} — {run_name} ---')
    fc = _make_fold_config(base_yaml, pre_folder, run_name, seed, condition)
    fold_log_dir = log_path(fc.config_file)
    graphs_dir = graphs_data_path(fc.dataset)

    os.makedirs(fold_log_dir, exist_ok=True)

    # 1. generate
    if _have_data(graphs_dir):
        print(f'  [skip] data already exists at {graphs_dir}')
    else:
        print(f'  [run]  data_generate -> {graphs_dir}')
        # compute_ranks=False skips the post-gen SVD and kinograph.png
        # (cheaper; matches the run_generate_YT_data.py pre-gen path).
        data_generate(fc, device=device, visualize=False, run_vizualized=0,
                      style='color', alpha=1, erase=True, save=True, step=100,
                      compute_ranks=False)

    # 2. train
    if _have_model(fold_log_dir):
        print(f'  [skip] model already exists at {fold_log_dir}/models')
    else:
        print(f'  [run]  data_train -> {fold_log_dir}')
        data_train(fc, device=device, erase=True)

    # 3. test (rollout)
    if _have_test(fold_log_dir):
        print(f'  [skip] rollout log already exists at {fold_log_dir}/results_rollout.log')
    else:
        print(f'  [run]  data_test -> {fold_log_dir}/results_rollout.log')
        data_test(config=fc, visualize=False, best_model='best', run=0,
                  step=10, n_rollout_frames=250, device=device)

    # 4. plot / parameter extraction
    if _have_plot(fold_log_dir):
        print(f'  [skip] metrics.txt already exists at {fold_log_dir}/results/metrics.txt')
    else:
        print(f'  [run]  data_plot -> {fold_log_dir}/results/metrics.txt')
        data_plot(config=fc, epoch_list=['best'], style='color',
                  extended='plots', device=device,
                  apply_weight_correction=True, skip_svd=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='flyvis_noise_005_INR',
                   help='Base config (no .yaml, no pre-folder)')
    p.add_argument('--output_root',
                   default='/groups/saalfeld/home/allierc/GraphData')
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--conditions', nargs='+',
                   default=['davis', 'yt'], choices=['davis', 'yt'],
                   help='Which conditions to run (default: both)')
    args = p.parse_args()

    assert os.path.isdir(args.output_root), \
        f'output_root does not exist: {args.output_root}'
    set_data_root(args.output_root)

    config_file, pre_folder = add_pre_folder(args.config)
    base_yaml = config_path(f'{config_file}.yaml')
    base_config = NeuralGraphConfig.from_yaml(base_yaml)
    device = set_device(base_config.training.device)

    seeds = list(range(42, 42 + args.n_seeds))
    print(f'INR CV — config: {args.config}')
    print(f'           seeds: {seeds}')
    print(f'           conditions: {args.conditions}')
    print(f'           data root: {args.output_root}')

    for i, seed in enumerate(seeds):
        for cond in args.conditions:
            run_fold(i, seed, cond, base_yaml, pre_folder,
                     args.config, device)

    print('\nINR CV complete.')


if __name__ == '__main__':
    main()
