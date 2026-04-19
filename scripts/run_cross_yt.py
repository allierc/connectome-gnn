"""
Unified runner for the YT-trained / DAVIS-cross-tested workflow used by
  run_GNN_conditions.sh   (per-condition HPs)
  run_GNN_cross.sh        (uniform null-edges HPs)

Per condition, driven from each YT-training yaml `<base>_<suffix>.yaml`:
  1. data_generate on YT stimulus     (skipped if graphs_data/x_list_train exists)
  2. data_train on YT                  (skipped if models/best_model_with_*.pt)
  3. data_test CROSS: YT-trained model rolled out on DAVIS held-out test data
     (skipped if <yt_log_dir>/results_rollout_on_<base>.log exists, unless
     --force_test is passed — in which case the log file is deleted first)
  4. data_plot on the YT-trained model (skipped if results/metrics.txt exists)

The cross-test loads the model from the YT-trained log dir and uses the
condition's base DAVIS config (<base>.yaml) as test_config — so its
simulation data serves as the DAVIS held-out test set for rollout
evaluation. The resulting rollout Pearson r fills the prediction columns
of the 6-column TeX table.

Usage:
    python scripts/run_cross_yt.py \\
        --suffix yt_per_cond \\
        --output_root /groups/saalfeld/home/allierc/GraphData \\
        [--force_test]
"""

import argparse
import glob
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
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
    from connectome_gnn.utils import set_data_root
except ImportError:
    def set_data_root(path):
        _cg_utils._data_root = path
from connectome_gnn.generators.graph_data_generator import data_generate  # noqa: E402
from connectome_gnn.models.graph_trainer import data_train, data_test  # noqa: E402
from GNN_PlotFigure import data_plot  # noqa: E402


# Condition base names — same list as scripts/write_cross_yt_configs.py.
CONDITION_BASES = [
    'flyvis_noise_free',
    'flyvis_noise_005',
    'flyvis_noise_05',
    'flyvis_noise_005_010',
    'flyvis_noise_005_null_edges_pc_400',
    'flyvis_noise_005_removed_pc_20',
    'flyvis_noise_005_stride_5',
    'flyvis_noise_005_hidden_010_ngp',
]


def _load(cfg_name):
    cfg_file, pre = add_pre_folder(cfg_name)
    cfg = NeuralGraphConfig.from_yaml(config_path(f'{cfg_file}.yaml'))
    cfg.dataset = pre + cfg.dataset
    cfg.config_file = pre + cfg_name
    return cfg, pre


def _have_data(graphs_dir):
    return os.path.isdir(os.path.join(graphs_dir, 'x_list_train'))


def _have_model(log_dir):
    return bool(glob.glob(os.path.join(log_dir, 'models', 'best_model_with_*.pt')))


def _have_plot(log_dir):
    return os.path.exists(os.path.join(log_dir, 'results', 'metrics.txt'))


def _cross_test_log(log_dir, base_name):
    """Path where data_test writes the cross-rollout log."""
    short = base_name.replace('flyvis_', '').replace('fly/', '')
    return os.path.join(log_dir, f'results_rollout_on_{short}.log')


def run_one(base_name, suffix, device, force_test):
    """Run generate+train+cross-test+plot for one condition."""
    yt_cfg_name = f'{base_name}_{suffix}'
    print(f'\n=== {yt_cfg_name}  (DAVIS test from {base_name}) ===')

    # YT training config on disk.
    yt_yaml = config_path('fly', f'{yt_cfg_name}.yaml')
    if not os.path.isfile(yt_yaml):
        print(f'  [skip] YT yaml missing: {yt_yaml} — run '
              f'scripts/write_cross_yt_configs.py first')
        return

    yt_cfg, _   = _load(yt_cfg_name)
    base_cfg, _ = _load(base_name)

    yt_log_dir    = log_path(yt_cfg.config_file)
    yt_graphs_dir = graphs_data_path(yt_cfg.dataset)
    os.makedirs(yt_log_dir, exist_ok=True)

    # 1. generate YT data
    if _have_data(yt_graphs_dir):
        print(f'  [skip] YT data exists: {yt_graphs_dir}')
    else:
        print(f'  [run ] data_generate -> {yt_graphs_dir}')
        data_generate(yt_cfg, device=device, visualize=False, run_vizualized=0,
                      style='color', alpha=1, erase=True, save=True, step=100)

    # 2. train on YT
    if _have_model(yt_log_dir):
        print(f'  [skip] YT model exists: {yt_log_dir}/models')
    else:
        print(f'  [run ] data_train -> {yt_log_dir}')
        data_train(yt_cfg, device=device, erase=True)

    # 3. cross-test: YT-trained model on base (DAVIS) test data
    cross_log = _cross_test_log(yt_log_dir, base_name)
    if force_test and os.path.exists(cross_log):
        os.remove(cross_log)
        print(f'  [force] removed {cross_log}')
    if os.path.exists(cross_log):
        print(f'  [skip] cross-test log exists: {cross_log}')
    else:
        print(f'  [run ] data_test (YT model -> DAVIS held-out)')
        data_test(config=yt_cfg, visualize=False, best_model='best', run=0,
                  step=10, n_rollout_frames=250, device=device,
                  test_config=base_cfg)

    # 4. plot -> extract params of YT-trained model
    if force_test and os.path.exists(os.path.join(yt_log_dir, 'results', 'metrics.txt')):
        os.remove(os.path.join(yt_log_dir, 'results', 'metrics.txt'))
        print(f'  [force] removed metrics.txt')
    if _have_plot(yt_log_dir):
        print(f'  [skip] plot / metrics.txt exists')
    else:
        print(f'  [run ] data_plot -> {yt_log_dir}/results/metrics.txt')
        data_plot(config=yt_cfg, epoch_list=['best'], style='color',
                  extended='plots', device=device,
                  apply_weight_correction=True, skip_svd=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suffix', required=True,
                   help='Suffix that the YT yamls were written with '
                        '(e.g. yt_per_cond or yt_cross).')
    p.add_argument('--output_root',
                   default='/groups/saalfeld/home/allierc/GraphData')
    p.add_argument('--conditions', nargs='+', default=CONDITION_BASES,
                   help='Subset of conditions (base names) to run.')
    p.add_argument('--force_test', action='store_true',
                   help='Delete existing cross-test log + metrics.txt before '
                        're-running those two steps (does NOT force '
                        'regenerate/retrain).')
    p.add_argument('--emit_tex', default=None,
                   help='If set, call scripts/emit_cross_table_rows.py with '
                        'this basename after EACH condition so the TeX file '
                        'is updated continuously (e.g. '
                        'cv_yt_per_cond_rows.tex).')
    args = p.parse_args()

    assert os.path.isdir(args.output_root), f'missing {args.output_root}'
    set_data_root(args.output_root)

    base_config = NeuralGraphConfig.from_yaml(
        config_path('fly', f'{args.conditions[0]}.yaml'))
    device = set_device(base_config.training.device)

    print(f'Cross YT runner — suffix={args.suffix}  '
          f'output_root={args.output_root}  force_test={args.force_test}  '
          f'emit_tex={args.emit_tex or "<off>"}')
    for base_name in args.conditions:
        run_one(base_name, args.suffix, device, args.force_test)
        if args.emit_tex:
            # Re-emit the whole TeX file after each condition so the user can
            # watch the table grow in real time; missing rows stay as
            # "$\\cdot$" until their condition finishes.
            import subprocess
            subprocess.run(
                ['python',
                 os.path.join(REPO_ROOT, 'scripts', 'emit_cross_table_rows.py'),
                 '--suffix', args.suffix,
                 '--output_tex', args.emit_tex,
                 '--output_root', args.output_root],
                check=False,
            )

    print('\nCross YT runner complete.')


if __name__ == '__main__':
    main()
