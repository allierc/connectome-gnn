"""GNN training over the optogenetics sweep, 5-fold CV.

Style mirrors run_GNN_unified_blank50.py: a CONDITION_NODES dict (commented
lines skip), per-fold CV emission, and an argparse front-end. Each enabled
(condition × fold) pair produces:

    log/fly/<flyvis_noise_free_blank50_opto_<cond>_cv<XX>>/

Pre-requisite: the corresponding dataset must already exist at
    graphs_data/fly/<flyvis_noise_free_blank50_opto_<cond>_cv<XX>>/
(Run run_generate_optogenetics.py first.)

This script does NOT use connectome_gnn.cross.run_all_conditions — opto
conditions are not registered in cross/yaml_io.py. Instead it emits the
per-fold YAML into <data_root>/config/fly/ and dispatches training as a
subprocess to GNN_Main.py per (cond, fold).

Usage:
    python run_GNN_optogenetics.py                 # train every enabled cond × fold
    python run_GNN_optogenetics.py --cv00-only     # cv00 only
    python run_GNN_optogenetics.py --print-bsub    # emit cluster commands
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from connectome_gnn.utils import (  # noqa: E402
    graphs_data_path, load_data_root_from_json, set_data_root, get_data_root,
)
from _opto_cv_yaml import emit_fold_yaml, fold_dataset_name  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cv00-only', dest='cv00_only', action='store_true',
                    help='Train only fold 0 (cv00).')
parser.add_argument('--cluster', choices=['a100', 'l4'], default='a100',
                    help='LSF GPU queue (default a100). Used only with --print-bsub.')
parser.add_argument('--runtime-min', type=int, default=2880,
                    help='Cluster runtime cap in minutes (default 2880).')
parser.add_argument('--print-bsub', action='store_true',
                    help='Print bsub commands instead of running locally.')
parser.add_argument('--retrain', action='store_true',
                    help='Force retrain by adding --force to the GNN_Main invocation.')
args = parser.parse_args()


FOLDS = [0] if args.cv00_only else [0, 1, 2, 3, 4]


# Condition -> LSF queue (gpu_<node>). Comment a row to drop that condition
# from the run. Sorted by null_dim from highest to lowest.
CONDITION_NODES = {
    'TmY15_005':       'a100',  # 43,299
    'TmY15_heaviside': 'a100',
    # 'Mi1_005':         'a100',  # 25,834
    # 'Mi1_heaviside':   'a100',
    # 'Tm3_005':         'a100',  # 20,471
    # 'Tm3_heaviside':   'a100',
    # 'Tm4_005':         'a100',  # 15,971
    # 'Tm4_heaviside':   'a100',
    # 'Tm1_005':         'a100',  # 15,525
    # 'Tm1_heaviside':   'a100',
    # 'Mi4_005':         'a100',  # 14,439
    # 'Mi4_heaviside':   'a100',
    # 'T4c_005':         'a100',  # 12,564
    # 'T4c_heaviside':   'a100',
    # 'Mi9_005':         'a100',  # 11,889
    # 'Mi9_heaviside':   'a100',
    # 'Tm2_005':         'a100',  # 11,068
    # 'Tm2_heaviside':   'a100',
}

CONDITIONS = list(CONDITION_NODES.keys())


try:
    set_data_root(load_data_root_from_json())
except Exception:
    pass


def _dataset_voltage(cond: str, fold: int) -> str:
    return os.path.join(
        graphs_data_path('fly', fold_dataset_name(cond, fold)),
        'x_list_train', 'voltage.zarr',
    )


# Emit per-fold YAMLs first (idempotent — overwrites if already there).
print(f'emitting CV YAMLs for {len(CONDITIONS)} conditions × {len(FOLDS)} folds ...')
fold_yamls: dict[tuple[str, int], str] = {}
for cond in CONDITIONS:
    for fold in FOLDS:
        fold_yamls[(cond, fold)] = emit_fold_yaml(cond, fold)


# Verify all datasets exist on disk before launching training jobs.
missing = [
    fold_dataset_name(cond, fold)
    for cond in CONDITIONS for fold in FOLDS
    if not os.path.isdir(_dataset_voltage(cond, fold))
]
if missing:
    print('WARNING: the following datasets are missing on disk:')
    for m in missing:
        print(f'  {m}')
    print('         run run_generate_optogenetics.py first.')


py = sys.executable
gnn_main = os.path.join(REPO_ROOT, 'GNN_Main.py')


def _train_cmd(cond: str, fold: int) -> list[str]:
    name = fold_dataset_name(cond, fold)
    cmd = [py, gnn_main, '-o', 'train', name]
    if args.retrain:
        cmd.append('--force')
    return cmd


if args.print_bsub:
    log_root = os.path.join(get_data_root(), 'log', 'opto_train')
    print(f"# bsub commands — log root: {log_root}")
    print(f"mkdir -p {log_root}")
    for cond in CONDITIONS:
        queue = CONDITION_NODES[cond]
        for fold in FOLDS:
            name = fold_dataset_name(cond, fold)
            cmd = ' '.join(_train_cmd(cond, fold))
            print(
                f'bsub -J opto_{cond}_cv{fold:02d} -q gpu_{queue} -gpu "num=1" '
                f'-W {args.runtime_min} -o {log_root}/{name}.cluster.log '
                f'"{cmd}"'
            )
    sys.exit(0)


# Local sequential dispatch.
for cond in CONDITIONS:
    for fold in FOLDS:
        name = fold_dataset_name(cond, fold)
        if not os.path.isdir(_dataset_voltage(cond, fold)):
            print(f'\n=== SKIP {name}: dataset missing ===', flush=True)
            continue
        print(f'\n=== train {name} ===', flush=True)
        proc = subprocess.run(_train_cmd(cond, fold))
        if proc.returncode != 0:
            sys.exit(f'FAILED: {name} (returncode={proc.returncode})')
