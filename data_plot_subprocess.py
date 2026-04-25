"""Cluster subprocess: re-run data_plot only (no rollout).

Used by aggregate_blank50_tables.py --data_plot to refresh metrics.txt
on already-trained models when GNN_PlotFigure parameter extraction
changes. Skips data_test entirely; reads the trained model from
<log_dir>/models/best_model_with_*.pt and writes
<log_dir>/results/metrics.txt.
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import matplotlib
matplotlib.use('Agg')

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.utils import log_path, set_data_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data_plot-only cluster subprocess')
    parser.add_argument('--config',      required=True, help='YAML config path')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--log_file',    default=None)
    parser.add_argument('--config_file', default=None,
                        help='config_file field (e.g. fly/flyvis_..._cv00)')
    parser.add_argument('--error_log',   default=None)
    parser.add_argument('--iteration',   type=int, default=None)
    parser.add_argument('--slot',        type=int, default=None)
    parser.add_argument('--output_root', default=None)
    args = parser.parse_args()

    if args.device == 'auto':
        args.device = 'cuda'

    output_root = args.output_root or os.environ.get('GNN_OUTPUT_ROOT')
    if output_root:
        assert os.path.isdir(output_root), f'--output_root does not exist: {output_root}'
        assert os.access(output_root, os.W_OK), f'--output_root is not writable: {output_root}'
        set_data_root(output_root)

    try:
        config = NeuralGraphConfig.from_yaml(args.config)
        if args.config_file:
            config.config_file = args.config_file
            if '/' in args.config_file:
                pre = args.config_file.split('/')[0] + '/'
                if not config.dataset.startswith(pre):
                    config.dataset = pre + config.dataset

        log_file = open(args.log_file, 'a', buffering=1) if args.log_file else None
        try:
            from GNN_PlotFigure import data_plot
            data_plot(config=config, epoch_list=['best'], style='color',
                      extended='plots', device=args.device, log_file=log_file,
                      apply_weight_correction=True, skip_svd=True)
        finally:
            if log_file is not None:
                try:
                    log_file.close()
                except OSError:
                    pass

        run_log_dir = log_path(config.config_file)
        with open(os.path.join(run_log_dir, '_data_plot_complete'), 'w') as f:
            f.write(f'argv={sys.argv}\n')

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if args.error_log:
            with open(args.error_log, 'a') as f:
                f.write(f'\n--- iteration {args.iteration} slot {args.slot} ---\n')
                f.write(tb)
        sys.exit(1)
