#!/usr/bin/env python3
"""Append a cross-noise test result block to the master results_cv.txt.

Called by run_cross_noise_batch.sh after each test run:

    python scripts/append_test_to_results_cv.py \\
        --model_config flyvis_noise_005_null_edges_pc_400 \\
        --test_config  flyvis_noise_005_null_edges_pc_400_cross_noise_free \\
        --output_root  /groups/saalfeld/home/allierc/GraphData
"""
import argparse
import datetime
import glob as _glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from connectome_gnn.utils import config_path, git_sha, log_path, set_data_root


def _mtime_str(path):
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
    except OSError:
        return 'not found'


def _parse_pearson(path):
    """Return (mean, std) from 'Pearson r: X +/- Y' in a log file."""
    if not os.path.exists(path):
        return None, None
    m = re.search(r'Pearson r:\s*([\d.]+)\s*\+/-\s*([\d.]+)', open(path).read())
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def _parse_float(path, key):
    """Return float value of 'key: X' from a log file, or None."""
    if not os.path.exists(path):
        return None
    m = re.search(rf'{re.escape(key)}:\s*([\d.nan]+)', open(path).read())
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', required=True,
                        help='Base config name of the trained model '
                             '(e.g. flyvis_noise_005_null_edges_pc_400)')
    parser.add_argument('--test_config', required=True,
                        help='Cross-test config path or name '
                             '(e.g. flyvis_noise_005_null_edges_pc_400_cross_noise_free)')
    parser.add_argument('--output_root', default=None,
                        help='Root for log/. Defaults to GNN_OUTPUT_ROOT env var.')
    args = parser.parse_args()

    if args.output_root:
        set_data_root(args.output_root)

    # ------------------------------------------------------------------ paths
    # Resolve test config YAML
    tc_yaml = args.test_config if args.test_config.endswith('.yaml') else args.test_config + '.yaml'
    if not os.path.isabs(tc_yaml) and not os.path.exists(tc_yaml):
        tc_yaml = config_path(tc_yaml)

    # Load test dataset name to compute test_suffix
    try:
        from connectome_gnn.config import NeuralGraphConfig
        test_cfg = NeuralGraphConfig.from_yaml(tc_yaml)
        test_ds = test_cfg.dataset
    except Exception:
        test_ds = os.path.basename(args.test_config)

    base_name = os.path.basename(args.model_config.rstrip('/'))
    pre_folder = 'fly/'
    log_dir = log_path(pre_folder + base_name)

    # Compute test_suffix using same logic as graph_tester.py
    if test_ds != base_name and test_ds != pre_folder + base_name:
        test_ds_short = test_ds.replace('flyvis_', '').replace('fly/', '')
        test_suffix = f'_on_{test_ds_short}'
    else:
        test_suffix = ''

    test_log    = os.path.join(log_dir, f'results_test{test_suffix}.log')
    rollout_log = os.path.join(log_dir, f'results_rollout{test_suffix}.log')

    # ------------------------------------------------------------ parse metrics
    onestep_r, onestep_std = _parse_pearson(test_log)
    rollout_r,  rollout_std = _parse_pearson(rollout_log)
    conn_r2  = _parse_float(test_log, 'connectivity_R2')
    tau_r2   = _parse_float(test_log, 'tau_R2')
    vrest_r2 = _parse_float(test_log, 'V_rest_R2')

    # Best model checkpoint
    candidates = sorted(_glob.glob(os.path.join(log_dir, 'models', 'best_model_with_*.pt')))
    model_path = candidates[-1] if candidates else f'{log_dir}/models/ [not found]'

    # Config YAML for the trained model
    try:
        model_yaml = config_path(f'{pre_folder}{base_name}.yaml')
    except Exception:
        model_yaml = args.model_config + '.yaml'

    # --------------------------------------------------------------- format block
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sha = git_sha()

    def _line(name, val, std=None, group='cross-noise test'):
        if val is None:
            return f"{name:<35} {'—':>8} {'—':>8}   {group}\n"
        std_s = f'{std:.4f}' if std is not None else '—'
        return f"{name:<35} {val:>8.4f} {std_s:>8}   {group}\n"

    lines = [
        f"\n{'='*80}\n",
        f"date:             {now_str}\n",
        f"git commit:       {sha}\n",
        f"config:           {model_yaml}  [{_mtime_str(model_yaml)}]\n",
        f"test_config:      {tc_yaml}  [{_mtime_str(tc_yaml)}]\n",
        f"test_dataset:     {test_ds}{test_suffix}\n",
        f"\n-- Trained model --\n",
        f"model:            {model_path}  [{_mtime_str(model_path)}]\n",
        f"\n-- Cross-noise test results --\n",
        f"test_log:         {test_log}  [{_mtime_str(test_log)}]\n",
        f"rollout_log:      {rollout_log}  [{_mtime_str(rollout_log)}]\n",
        f"\n{'Metric':<35} {'Value':>8} {'SD':>8}   group\n",
        f"{'-'*65}\n",
        _line('one_step_r',     onestep_r, onestep_std),
        _line('rollout_r',      rollout_r,  rollout_std),
    ]
    if conn_r2 is not None:
        lines.append(_line('W_corrected_R2', conn_r2))
    if tau_r2 is not None:
        lines.append(_line('tau_R2',         tau_r2))
    if vrest_r2 is not None:
        lines.append(_line('V_rest_R2',      vrest_r2))

    content = ''.join(lines)

    # --------------------------------------------------------- write master file
    master_path = log_path('results_cv.txt')
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    with open(master_path, 'a') as f:
        f.write(content)

    print(content)
    print(f"Appended to: {master_path}")


if __name__ == '__main__':
    main()
