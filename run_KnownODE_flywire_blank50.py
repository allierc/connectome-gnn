"""Known_ODE training on the 4 flywireRF v2 connectomes, blank50 + DAVIS2017-partial-test.

5-fold CV per condition. Reuses the datasets pre-built by
run_generate_flywire_holdout_data_blank50.py — same datasets the GNN
runner reads, since the connectome and stimulus are identical between a
GNN/KODE pair (only graph_model differs at training).

Each variant has a dedicated _known_ode_ yaml whose graph_model block
sets the right signal_model_name (`<variant>_known_ode`). Because that
name varies per condition, we route via hp_yaml_overrides instead of a
single uniform HP yaml.

Datasets at: <output_root>/graphs_data/fly/<base>_blank50_cv{0..4}/
                       (shared with run_GNN_flywire_blank50.py)
CV yamls   : <output_root>/config/fly/<base>_blank50_flywire_known_ode_cv{0..4}.yaml
Logs       : <output_root>/log/fly/<base>_blank50_flywire_known_ode_cv{0..4}/
Summary    : <output_root>/log/cv_blank50_flywire_known_ode_summary.md

Total training units: 4 conditions × 5 folds = 20 Known_ODE runs.

Submitted as two waves of 10 concurrent jobs each (conditions_per_wave=2):
  - Wave 1: 2 small variants (e8 / full_eye) -> gpu_a100, 10 jobs
  - Wave 2: 2 proximal_nulls variants        -> gpu_h100, 10 jobs

The h100 wave only starts after the a100 wave finishes.

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
                          'scratch.')
_parser.add_argument('--retest', action='store_true',
                     help='Re-run TEST (rollout): wipe results_rollout.log per '
                          'fold and rerun data_test.')
_parser.add_argument('--replot', action='store_true',
                     help='Re-run PLOT (parameter scatters): wipe '
                          'results/metrics.txt per fold and rerun data_plot.')
_parser.add_argument('--redo-all', dest='redo_all', action='store_true',
                     help='Shortcut for --retrain --retest --replot.')
_parser.add_argument('--skip-test-plot', dest='skip_test_plot', action='store_true',
                     help='Submit the training wave only — suppress the '
                          'test+plot wave entirely.')
_parser.add_argument('--cluster', choices=['a100', 'h100', 'l4'], default=None,
                     help='Override the LSF GPU queue for ALL conditions '
                          '(otherwise uses CONDITION_NODES per-condition). '
                          'Takes precedence over --group.')
_parser.add_argument('--group', choices=['a100', 'h100', 'all'], default='all',
                     help='Submit only one queue group, so two terminals can '
                          'launch the a100 wave and the h100 wave in parallel '
                          '(without waiting for each other in a single process). '
                          'Default: all (run both groups sequentially).')
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


# condition name -> LSF queue (gpu_<node>). KODE's memory footprint is much
# smaller than the GNN's (no trainable MLPs / activations), so a100 with the
# bumped n_cpus=8 (= ~320 GB LSF memlimit) is sufficient for full_eye even
# though the GNN runner has to send it to h100. Only proximal_nulls (≤9.6M
# edges) needs h100.
CONDITION_NODES = {
    'e8_flywireRF_noise_005':                       'a100',
    'e8_flywireRF_proximal_nulls_noise_005':        'h100',
    'full_eye_flywireRF_noise_005':                 'a100',
    'full_eye_flywireRF_proximal_nulls_noise_005':  'h100',
}

# Each base reuses its sibling _known_ode_ yaml as the HP source.
# signal_model_name differs per variant so a single uniform HP yaml will not
# work — hp_yaml_overrides is the right knob.
HP_YAML_OVERRIDES = {
    'e8_flywireRF_noise_005':                       'e8_flywireRF_known_ode_noise_005',
    'e8_flywireRF_proximal_nulls_noise_005':        'e8_flywireRF_proximal_nulls_known_ode_noise_005',
    'full_eye_flywireRF_noise_005':                 'full_eye_flywireRF_known_ode_noise_005',
    'full_eye_flywireRF_proximal_nulls_noise_005':  'full_eye_flywireRF_proximal_nulls_known_ode_noise_005',
}

# Group bases by target queue so each call to run_all_conditions can use a
# single node_name + conditions_per_wave=N (= N*5 concurrent cluster jobs per
# wave). Mixing per-condition routing with conditions_per_wave>1 isn't
# supported by the runner — node_name_per_condition forces wave size 1.
A100_BASES = [b for b, n in CONDITION_NODES.items() if n == 'a100']
H100_BASES = [b for b, n in CONDITION_NODES.items() if n == 'h100']


def _run_group(condition_filter, node_name):
    if not condition_filter:
        return
    print(f'\n=== submitting wave on gpu_{node_name}: {condition_filter} ===')
    run_all_conditions(
        hp_source='uniform',
        suffix='blank50_flywire_known_ode',
        # Default HP yaml is irrelevant here because every base has an entry
        # in HP_YAML_OVERRIDES; we still pass one (matches the e8 base) so
        # the uniform-mode path has something to fall back to if the dict
        # is edited.
        hp_yaml='e8_flywireRF_known_ode_noise_005',
        node_name=node_name,
        hard_runtime_limit_min=2880,
        sim_overrides=BLANK50_SIM_OVERRIDES,
        dataset_tag='blank50',
        condition_filter=condition_filter,
        data_augmentation_loop=100,
        hp_yaml_overrides=HP_YAML_OVERRIDES,
        n_folds=_n_folds,
        conditions_per_wave=len(condition_filter),  # all bases in one wave
        emit_tex=False,
        force_train=_force_train,
        force_test=_force_test,
        force_plot=_force_plot,
        skip_test_plot=_args.skip_test_plot,
    )


# A single --cluster override forces every base onto that queue (and, by
# design, ignores --group: --cluster is the "all on one queue" knob).
if _args.cluster:
    _run_group(A100_BASES + H100_BASES, _args.cluster)
elif _args.group == 'a100':
    _run_group(A100_BASES, 'a100')
elif _args.group == 'h100':
    _run_group(H100_BASES, 'h100')
else:
    # 'all' (default): submit both groups sequentially in one process.
    _run_group(A100_BASES, 'a100')
    _run_group(H100_BASES, 'h100')


# Per-condition CV summary markdown (rows per fold + mean±SD), written to
# <output_root>/log/cv_blank50_flywire_known_ode_summary.md.
emit_summary_md('blank50_flywire_known_ode',
                output_root=os.environ.get('GNN_OUTPUT_ROOT')
                            or load_data_root_from_json(),
                n_folds=_n_folds)


# ---------------------------------------------------------------------------
# Example invocations  (run from /workspace/connectome-gnn)
# ---------------------------------------------------------------------------
#
# # Smoke test: train fold 0 of every condition only (per-condition queues)
# python run_KnownODE_flywire_blank50.py --cv00-only
#
# # Default: resume — skip already-trained folds (a100 wave then h100 wave)
# python run_KnownODE_flywire_blank50.py
#
# # Run the two queues in PARALLEL — launch each command in its own terminal:
# #   terminal 1:  python run_KnownODE_flywire_blank50.py --group a100
# #   terminal 2:  python run_KnownODE_flywire_blank50.py --group h100
# # Each handles its own LSF queue independently — no inter-process waiting.
#
# # Re-render plots only (rollouts already present)
# python run_KnownODE_flywire_blank50.py --replot
#
# # Force fresh test rollout for every fold
# python run_KnownODE_flywire_blank50.py --retest
#
# # Full clean re-run (wipe + retrain + retest + replot every fold)
# python run_KnownODE_flywire_blank50.py --redo-all
