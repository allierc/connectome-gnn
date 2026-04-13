"""Print a CV comparison table across multiple conditions.

Usage (called by run_cross_noise_batch.sh):
    python print_cv_comparison.py \
        --repo_dir /path/to/repo \
        --labels   noise_free noise_005 noise_05 null_edges_pc_400 removed_pc_20 \
        --configs  flyvis_cmp_noise_free flyvis_cmp_noise_005 flyvis_cmp_noise_05 \
                   flyvis_noise_005_null_edges_pc_400 flyvis_cmp_removed_pc_20

Or standalone (reads partial results if some CV runs are still pending):
    python print_cv_comparison.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.utils import set_data_root, log_path
from connectome_gnn.models.cv_runner import compare_cv_results

# Default values used when called without arguments
DEFAULT_LABELS = [
    'noise_free',
    'noise_005',
    'noise_05',
    'null_edges_pc_400',
    'removed_pc_20',
]
DEFAULT_CONFIGS = [
    'flyvis_cmp_noise_free',
    'flyvis_cmp_noise_005',
    'flyvis_cmp_noise_05',
    'flyvis_noise_005_null_edges_pc_400',
    'flyvis_cmp_removed_pc_20',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Print CV comparison table')
    parser.add_argument('--repo_dir', type=str, default=None,
                        help='Repo root (used to locate data_paths.json if needed)')
    parser.add_argument('--labels',  nargs='+', default=DEFAULT_LABELS,
                        help='Display labels for each condition (row names)')
    parser.add_argument('--configs', nargs='+', default=DEFAULT_CONFIGS,
                        help='Base config names (must match log/fly/<name> dirs)')
    parser.add_argument('--output_root', type=str, default=None,
                        help='Root for log/. Defaults to GNN_OUTPUT_ROOT env var or cwd')
    args = parser.parse_args()

    output_root = args.output_root or os.environ.get('GNN_OUTPUT_ROOT')
    if output_root:
        set_data_root(output_root)

    table_path = os.path.join(log_path(), 'cv_comparison_table.txt')

    compare_cv_results(
        condition_labels=args.labels,
        config_names=args.configs,
        output_path=table_path,
    )
