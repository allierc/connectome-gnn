"""Standalone test+plot script for cluster jobs.

Called by the LLM exploration pipeline to run test and plot
on a cluster node after training completes.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import matplotlib
matplotlib.use('Agg')
import argparse
import traceback

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.graph_trainer import data_test
from connectome_gnn.utils import log_path, set_data_root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='connectome test+plot subprocess')
    parser.add_argument('--config', required=True, help='path to YAML config')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--log_file', default=None, help='path for analysis log (append mode)')
    parser.add_argument('--config_file', default=None, help='config_file field')
    parser.add_argument('--error_log', default=None, help='path for error details')
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

        # Derive pre_folder from config_file (e.g., "fly/flyvis_..." -> "fly/")
        pre_folder = config.config_file.split('/')[0] + '/' if '/' in config.config_file else ''

        log_file = open(args.log_file, 'a', buffering=1) if args.log_file else None
        try:
            # Test
            config.simulation.noise_model_level = 0.0
            data_test(
                config=config,
                visualize=False,
                style="color name continuous_slice",
                verbose=False,
                best_model='best',
                run=0,
                test_mode="",
                sample_embedding=False,
                step=10,
                n_rollout_frames=1000,
                device=args.device,
                particle_of_interest=0,
                new_params=None,
                log_file=log_file,
            )

            # Plot
            from GNN_PlotFigure import data_plot
            folder_name = log_path(pre_folder, 'tmp_results') + '/'
            os.makedirs(folder_name, exist_ok=True)
            data_plot(
                config=config,
                epoch_list=['best'],
                style='color',
                extended='plots',
                device=args.device,
                log_file=log_file,
                skip_svd=True,
            )
        finally:
            if log_file:
                try:
                    log_file.close()
                except OSError:
                    pass  # Stale NFS handle — ignore close error

        # Mark as complete
        run_log_dir = log_path(config.config_file)
        with open(os.path.join(run_log_dir, '_test_plot_complete'), 'w') as f:
            f.write(f"argv={sys.argv}\n")

    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if args.error_log:
            with open(args.error_log, 'a') as f:
                f.write(f"\n--- iteration {args.iteration} slot {args.slot} ---\n")
                f.write(tb)
        sys.exit(1)
