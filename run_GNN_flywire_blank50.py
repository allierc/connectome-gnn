"""GNN training on the 4 flywireRF v2 connectomes, blank50 + DAVIS2017-partial-test.

5-fold CV per condition. Reads the datasets pre-built by
run_generate_flywire_holdout_data_blank50.py and trains one GNN per fold
using the per-condition winner from CONDITIONS in
src/connectome_gnn/cross/yaml_io.py — which for these 4 entries points
back at the same flywireRF base yaml (each variant's yaml already
contains a complete graph_model + training block, so winner = base).

Datasets at: <output_root>/graphs_data/fly/<base>_blank50_cv{0..4}/
                       (shared with run_KnownODE_flywire_blank50.py)
CV yamls   : <output_root>/config/fly/<base>_blank50_flywire_cv{0..4}.yaml
Logs       : <output_root>/log/fly/<base>_blank50_flywire_cv{0..4}/
Summary    : <output_root>/log/cv_blank50_flywire_summary.md

Total training units: 4 conditions × 5 folds = 20 GNNs.

Per-condition queue routing (CONDITION_NODES):
  - 327k / 1.3M edge variants (e8 / full_eye)              -> gpu_a100
  - 2.4M / 9.6M edge variants (proximal_nulls of either)   -> gpu_h100

This script does NOT generate data — it fails fast if the datasets are
missing. Run run_generate_flywire_holdout_data_blank50.py first.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import run_all_conditions, emit_summary_md
from connectome_gnn.utils import load_data_root_from_json


_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--retrain', action='store_true',
                     help='Re-run TRAIN: wipe models/, results/, tmp_training/ '
                          'and cluster_train.log per fold, then retrain from '
                          'scratch. Without this flag the [skip] guard keeps '
                          'every already-trained fold and only fills in folds '
                          'whose models/ is missing.')
_parser.add_argument('--retest', action='store_true',
                     help='Re-run TEST (rollout): wipe results_rollout.log per '
                          'fold and rerun data_test. Independent of --retrain '
                          'and --replot.')
_parser.add_argument('--replot', action='store_true',
                     help='Re-run PLOT (parameter scatters): wipe '
                          'results/metrics.txt per fold and rerun data_plot. '
                          'Independent of --retrain and --retest.')
_parser.add_argument('--redo-all', dest='redo_all', action='store_true',
                     help='Shortcut for --retrain --retest --replot.')
_parser.add_argument('--skip-test-plot', dest='skip_test_plot', action='store_true',
                     help='Submit the training wave only — suppress the '
                          'test+plot wave entirely.')
_parser.add_argument('--cluster', choices=['a100', 'h100', 'l4'], default=None,
                     help='Override the LSF GPU queue for ALL conditions '
                          '(otherwise uses CONDITION_NODES per-condition).')
_parser.add_argument('--cv00-only', dest='cv00_only', action='store_true',
                     help='Limit the CV grid to fold 0 (cv00) only.')
_args = _parser.parse_args()

_force_train = bool(_args.retrain or _args.redo_all)
_force_test  = bool(_args.retest  or _args.redo_all)
_force_plot  = bool(_args.replot  or _args.redo_all)
_n_folds     = 1 if _args.cv00_only else 5


BLANK50_SIM_OVERRIDES = {
    'blank_prefix_fraction': 0.50,
    'skip_short_videos': True,
}


# condition name -> LSF queue (gpu_<node>). Comment out a row to drop it.
# proximal_nulls variants are large (2.4M / 9.6M edges) -> h100.
CONDITION_NODES = {
    'e8_flywireRF_noise_005':                       'a100',
    'e8_flywireRF_proximal_nulls_noise_005':        'h100',
    'full_eye_flywireRF_noise_005':                 'a100',
    'full_eye_flywireRF_proximal_nulls_noise_005':  'h100',
}

CONDITION_FILTER   = list(CONDITION_NODES.keys())
NODE_PER_CONDITION = CONDITION_NODES


run_all_conditions(
    hp_source='per_condition',
    suffix='blank50_flywire',
    hp_yaml=None,
    node_name=_args.cluster or 'a100',
    node_name_per_condition=NODE_PER_CONDITION,
    hard_runtime_limit_min=2880,
    sim_overrides=BLANK50_SIM_OVERRIDES,
    dataset_tag='blank50',
    condition_filter=CONDITION_FILTER,
    data_augmentation_loop=500,
    n_folds=_n_folds,
    conditions_per_wave=2,
    emit_tex=False,
    force_train=_force_train,
    force_test=_force_test,
    force_plot=_force_plot,
    skip_test_plot=_args.skip_test_plot,
)


# Per-condition CV summary markdown (rows per fold + mean±SD), written to
# <output_root>/log/cv_blank50_flywire_summary.md.
emit_summary_md('blank50_flywire',
                output_root=os.environ.get('GNN_OUTPUT_ROOT')
                            or load_data_root_from_json(),
                n_folds=_n_folds)


# ---------------------------------------------------------------------------
# Example invocations  (run from /workspace/connectome-gnn)
# ---------------------------------------------------------------------------
#
# # Smoke test: train fold 0 of every condition only (per-condition queues)
# python run_GNN_flywire_blank50.py --cv00-only
#
# # Default: resume — skip already-trained folds
# python run_GNN_flywire_blank50.py
#
# # Re-render plots only (rollouts already present)
# python run_GNN_flywire_blank50.py --replot
#
# # Force fresh test rollout for every fold
# python run_GNN_flywire_blank50.py --retest
#
# # Full clean re-run (wipe + retrain + retest + replot every fold)
# python run_GNN_flywire_blank50.py --redo-all
#
# # Force every condition onto a single queue (override CONDITION_NODES)
# python run_GNN_flywire_blank50.py --cluster h100
