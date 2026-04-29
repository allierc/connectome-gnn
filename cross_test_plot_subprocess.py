"""Cluster subprocess for the cross-check workflow.

Runs ONE fold of:
    1. data_test  (YT-trained model cross-rolled out on DAVIS held-out data,
                   i.e. config=YT yaml, test_config=base DAVIS yaml)
       -> writes <log_dir>/results_rollout_on_<short>.log
    2. data_plot  (parameter extraction on the YT-trained model)
       -> writes <log_dir>/results/metrics.txt

Sibling of test_plot_subprocess.py (used by the LLM exploration), but wired for
the cross-test workflow that needs --test_config. Submitted by
submit_cluster_cross_test_plot_job() in connectome_gnn/LLM/cluster.py.
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import matplotlib
matplotlib.use('Agg')

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.graph_trainer import data_test
from connectome_gnn.utils import (
    add_pre_folder, log_path, set_data_root,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cross test+plot cluster subprocess')
    parser.add_argument('--config',       required=True, help='YT training YAML path')
    parser.add_argument('--test_configs', required=True, nargs='+',
                        help='One or more DAVIS CV YAML paths (rollout test sets); '
                             'the script runs data_test once per entry, then a single data_plot.')
    parser.add_argument('--test_config_files', nargs='+', default=None,
                        help='Matching config_file fields for --test_configs '
                             '(e.g. fly/flyvis_noise_free_cv00). Must be same '
                             'length as --test_configs if given.')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--log_file',    default=None)
    parser.add_argument('--config_file', default=None, help='YT config_file field (e.g. fly/flyvis_..._cv00)')
    parser.add_argument('--error_log',   default=None)
    parser.add_argument('--iteration',   type=int, default=None)
    parser.add_argument('--slot',        type=int, default=None)
    parser.add_argument('--output_root', default=None)
    parser.add_argument('--n_rollout_frames', type=int, default=250)
    parser.add_argument('--skip-test', dest='skip_test', action='store_true',
                        help='Skip the data_test rollout phase (use when only re-emitting plots).')
    parser.add_argument('--skip-plot', dest='skip_plot', action='store_true',
                        help='Skip the data_plot phase (use when only re-running rollout).')
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda'

    output_root = args.output_root or os.environ.get('GNN_OUTPUT_ROOT')
    if output_root:
        assert os.path.isdir(output_root), f'--output_root does not exist: {output_root}'
        assert os.access(output_root, os.W_OK), f'--output_root is not writable: {output_root}'
        set_data_root(output_root)

    if args.test_config_files and len(args.test_config_files) != len(args.test_configs):
        print(f'ERROR: --test_config_files ({len(args.test_config_files)}) '
              f'must match --test_configs ({len(args.test_configs)})',
              file=sys.stderr)
        sys.exit(2)

    try:
        # YT-trained model config (source of weights).
        config = NeuralGraphConfig.from_yaml(args.config)
        if args.config_file:
            config.config_file = args.config_file
        # Prepend pre_folder from config_file to dataset (guarded).
        if args.config_file and '/' in args.config_file:
            pre = args.config_file.split('/')[0] + '/'
            if not config.dataset.startswith(pre):
                config.dataset = pre + config.dataset

        log_file = open(args.log_file, 'a', buffering=1) if args.log_file else None
        try:
            # 1. Cross-test: loop over every DAVIS CV test set.
            if args.skip_test:
                print('[cross] --skip-test set: skipping data_test rollout phase', flush=True)
            else:
                for j, test_yaml in enumerate(args.test_configs):
                    test_config = NeuralGraphConfig.from_yaml(test_yaml)
                    if args.test_config_files:
                        test_config.config_file = args.test_config_files[j]
                        # Prepend pre_folder from config_file to dataset (guarded).
                        if '/' in test_config.config_file:
                            pre = test_config.config_file.split('/')[0] + '/'
                            if not test_config.dataset.startswith(pre):
                                test_config.dataset = pre + test_config.dataset
                    else:
                        base_name = os.path.basename(test_yaml).replace('.yaml', '')
                        cfg_file, pre = add_pre_folder(base_name)
                        if not test_config.dataset.startswith(pre):
                            test_config.dataset = pre + test_config.dataset
                        test_config.config_file = pre + base_name
                    print(f'[cross] {config.config_file}  ->  {test_config.config_file}',
                          flush=True)
                    data_test(config=config, visualize=False, best_model='best', run=0,
                              step=10, n_rollout_frames=args.n_rollout_frames,
                              device=args.device, test_config=test_config,
                              log_file=log_file)

            # 2. Parameter recovery plot (once per YT model).
            if args.skip_plot:
                print('[cross] --skip-plot set: skipping data_plot phase', flush=True)
            else:
                from GNN_PlotFigure import data_plot
                data_plot(config=config, epoch_list=['best'], style='color',
                          extended='plots', device=args.device, log_file=log_file,
                          apply_weight_correction=True, skip_svd=True)
        finally:
            if log_file is not None:
                try:
                    log_file.close()
                except OSError:
                    pass  # stale NFS handle

        run_log_dir = log_path(config.config_file)
        with open(os.path.join(run_log_dir, '_cross_test_plot_complete'), 'w') as f:
            f.write(f'argv={sys.argv}\n')

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if args.error_log:
            with open(args.error_log, 'a') as f:
                f.write(f'\n--- iteration {args.iteration} slot {args.slot} ---\n')
                f.write(tb)
        sys.exit(1)
