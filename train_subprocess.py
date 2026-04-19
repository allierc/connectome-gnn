"""Standalone training script for cluster jobs.

Called by the LLM exploration pipeline (cluster.py) to run training only
on a cluster node. Data generation and test/plot are handled locally.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')
import argparse
import traceback

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.graph_trainer import data_train
from connectome_gnn.utils import set_data_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='flyvis training subprocess')
    parser.add_argument('--config', required=True, help='path to YAML config')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log_file', default=None, help='path for analysis log')
    parser.add_argument('--config_file', default=None, help='config_file field (e.g. fly/flyvis_noise_05_Claude_00)')
    parser.add_argument('--error_log', default=None, help='path for error details')
    parser.add_argument('--erase', action='store_true')
    parser.add_argument('--exploration_dir', default=None)
    parser.add_argument('--iteration', type=int, default=None)
    parser.add_argument('--slot', type=int, default=None)
    parser.add_argument('--output_root', default=None, help='root directory for log/ and graphs_data/')
    args = parser.parse_args()

    output_root = args.output_root or os.environ.get('GNN_OUTPUT_ROOT')
    if output_root:
        assert os.path.isdir(output_root), f"--output_root does not exist: {output_root}"
        assert os.access(output_root, os.W_OK), f"--output_root is not writable: {output_root}"
        set_data_root(output_root)

    try:
        config = NeuralGraphConfig.from_yaml(args.config)
        if args.config_file:
            config.config_file = args.config_file
        # Prepend pre_folder (e.g. 'fly/') from config_file to dataset so
        # data_train finds graphs_data/<pre>/<dataset>/. Guarded — if the
        # YAML already bakes the prefix in (e.g. GNN_LLM Claude flow), leave
        # it alone to avoid double-prefixing. Matches what
        # GNN_LLM/pipeline.py does upstream.
        if args.config_file and '/' in args.config_file:
            pre = args.config_file.split('/')[0] + '/'
            if not config.dataset.startswith(pre):
                config.dataset = pre + config.dataset

        log_file = open(args.log_file, 'w', buffering=1) if args.log_file else None
        try:
            data_train(
                config=config,
                erase=args.erase,
                device=args.device,
                log_file=log_file,
            )
        finally:
            if log_file:
                try:
                    log_file.close()
                except OSError:
                    pass  # Stale NFS handle — training completed, ignore close error

        # Mark run as complete
        from connectome_gnn.utils import log_path
        run_log_dir = log_path(config.config_file)
        with open(os.path.join(run_log_dir, '_complete'), 'w') as f:
            f.write(f"argv={sys.argv}\n")
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if args.error_log:
            with open(args.error_log, 'a') as f:
                f.write(f"\n--- iteration {args.iteration} slot {args.slot} ---\n")
                f.write(tb)
        sys.exit(1)
