"""GNN training over the optogenetics sweep, 5-fold CV.

Mirrors run_GNN_unified_blank50.py: TARGETS / WAVEFORMS lists where commenting
a row drops conditions, per-fold CV YAMLs, argparse front-end. Reuses
connectome_gnn.cross.pipeline.submit_training_wave to dispatch jobs to LSF
(same path as the unified blank50 runner — no duplicated bsub logic here).

Pre-requisite: the corresponding dataset must already exist at
    graphs_data/fly/<flyvis_noise_free_blank50_opto_<cond>_cv<XX>>/
(Run run_generate_optogenetics.py first.)

Usage:
    python run_GNN_optogenetics.py                  # submit all to LSF (default)
    python run_GNN_optogenetics.py --cv00-only      # cv00 only
    python run_GNN_optogenetics.py --retrain        # wipe models/ and retrain
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.cross.pipeline import submit_training_wave  # noqa: E402
from connectome_gnn.utils import (  # noqa: E402
    graphs_data_path, get_data_root, load_data_root_from_json, set_data_root,
)
from _opto_cv_yaml import emit_fold_yaml, fold_dataset_name  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cv00-only', dest='cv00_only', action='store_true',
                    help='Train only fold 0 (cv00).')
parser.add_argument('--cluster', choices=['a100', 'l4'], default='a100',
                    help='LSF GPU queue (default a100).')
parser.add_argument('--runtime-min', type=int, default=2880,
                    help='Cluster runtime cap in minutes (default 2880).')
parser.add_argument('--retrain', action='store_true',
                    help='Wipe models/, results/, tmp_training/ per fold and retrain.')
parser.add_argument('--max-parallel', type=int, default=10,
                    help='Maximum (cond, fold) jobs to submit per wave; each '
                         'wave blocks until it finishes before the next submits. '
                         'Default 10 (matches typical gpu_a100 throughput).')
args = parser.parse_args()


try:
    set_data_root(load_data_root_from_json())
except Exception:
    pass

OUTPUT_ROOT = get_data_root()


FOLDS = [0] if args.cv00_only else [0, 1, 2, 3, 4]


# Top-9 positive controls by null_dim. Comment a target row to drop both
# its variants; comment a waveform to drop that waveform for every target.
TARGETS = [
    'TmY15',  # 43,299  (rank 1)
    'Mi1',    # 25,834
    # 'Tm3',    # 20,471
    # 'Tm4',    # 15,971
    # 'Tm1',    # 15,525
    # 'Mi4',    # 14,439
    # 'T4c',    # 12,564
    # 'Mi9',    # 11,889
    # 'Tm2',    # 11,068
]
WAVEFORMS = [
    '05',         # white_noise, σ = 0.5  (matches flyvis_noise_05 convention)
    'heaviside',  # 35 ON / 35 OFF square wave, amp 1.0
]

CONDITIONS = [f'{t}_{w}' for t in TARGETS for w in WAVEFORMS]


def _dataset_voltage(cond: str, fold: int) -> str:
    return os.path.join(
        graphs_data_path('fly', fold_dataset_name(cond, fold)),
        'x_list_train', 'voltage.zarr',
    )


# Step 1: emit per-fold CV YAMLs into <output_root>/config/fly/ (idempotent).
print(f'emitting CV YAMLs for {len(CONDITIONS)} condition(s) × {len(FOLDS)} fold(s) ...')
yt_cfgs = []
for cond in CONDITIONS:
    for fold in FOLDS:
        ds_voltage = _dataset_voltage(cond, fold)
        if not os.path.isdir(ds_voltage):
            print(f'  SKIP {fold_dataset_name(cond, fold)}: dataset missing on disk')
            continue
        yaml_path = emit_fold_yaml(cond, fold)
        cfg = NeuralGraphConfig.from_yaml(yaml_path)
        # Mirror GNN_Main.py:249-250 — prefix dataset/config_file with fly/.
        if not cfg.dataset.startswith('fly/'):
            cfg.dataset = 'fly/' + cfg.dataset
        if not cfg.config_file.startswith('fly/'):
            cfg.config_file = 'fly/' + cfg.config_file
        yt_cfgs.append(cfg)

if not yt_cfgs:
    sys.exit('No runnable jobs — every dataset is missing on disk. '
             'Run run_generate_optogenetics.py first.')


# Step 2: submit jobs in waves of up to --max-parallel via the same LSF
# dispatcher used by run_GNN_unified_blank50.py. Each wave blocks on
# wait_for_cluster_jobs_with_metrics before the next wave submits.
n = len(yt_cfgs)
wave_size = max(1, args.max_parallel)
n_waves = (n + wave_size - 1) // wave_size
print(f'\nsubmitting {n} jobs to LSF in {n_waves} wave(s) of up to '
      f'{wave_size} (queue=gpu_{args.cluster}, runtime≤{args.runtime_min}min) ...')

for wave_i in range(n_waves):
    chunk = yt_cfgs[wave_i * wave_size:(wave_i + 1) * wave_size]
    print(f'\n=== wave {wave_i + 1}/{n_waves}: {len(chunk)} job(s) ===')
    submit_training_wave(
        yt_cfgs=chunk,
        output_root=OUTPUT_ROOT,
        node_name=args.cluster,
        hard_runtime_limit_min=args.runtime_min,
        force_train=args.retrain,
    )
