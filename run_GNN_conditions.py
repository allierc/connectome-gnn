"""
Standalone Python orchestrator — GNN cross-check with PER-CONDITION winner
hyperparameters (replaces run_GNN_conditions.sh).

For each of 8 conditions:
    1. Emit YT-training YAML per CV fold, merging:
         - simulation block from <base>.yaml
         - graph_model / training / plotting / claude blocks from
           <base>_winner.yaml   (per-condition HPs)
         - stimulus swapped to YouTube-VOS, per-fold seeds.
       (Uses scripts/write_cross_yt_configs.emit_one; --force re-emits.)
    2. data_generate  (LOCAL, devcontainer GPU)       — skip if x_list_train/
    3. data_train     (CLUSTER bsub, 5 folds in parallel) — skip if best_model_*
    4. wait the wave
    5. data_test      (LOCAL, YT-trained -> DAVIS held-out)
    6. data_plot      (LOCAL, param extraction from YT model)
    7. re-emit TeX after every condition so the 8-row table grows live.

All heavy lifting is delegated to the existing modules:
    scripts/write_cross_yt_configs.py   (YAML emission)
    scripts/run_cross_yt_parallel.py    (generate / cluster-train / test / plot)
    scripts/emit_cross_table_rows.py    (TeX row aggregation)

Usage:
    python run_GNN_conditions.py                 # cache-respecting run
    python run_GNN_conditions.py --force_test    # redo test + plot only
    python run_GNN_conditions.py --force_yaml    # re-emit YT YAMLs
    python run_GNN_conditions.py --conditions flyvis_noise_005 flyvis_noise_free

Submit via LSF interactive (optional):
    bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is < \\
        "python run_GNN_conditions.py"
"""

import argparse
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_REPO_ROOT, 'src'), _REPO_ROOT,
           os.path.join(_REPO_ROOT, 'scripts')):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
import connectome_gnn.utils as _cg_utils  # noqa: E402
from connectome_gnn.utils import config_path, set_device  # noqa: E402

try:
    from connectome_gnn.utils import set_data_root
except ImportError:
    def set_data_root(path):
        _cg_utils._data_root = path

import write_cross_yt_configs as _wcfg      # noqa: E402
import run_cross_yt_parallel as _runner     # noqa: E402
import emit_cross_table_rows as _emit       # noqa: E402
from run_cross_yt_parallel import cv_config_dir  # noqa: E402


DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'
SUFFIX    = 'yt_per_cond'
HP_SOURCE = 'per_condition'


def emit_yt_yamls(hp_source, suffix, hp_yaml_basename, n_folds, force,
                   output_root):
    """Mirror scripts/write_cross_yt_configs.py main loop — writes YT CV
    YAMLs to the shared-FS CV config dir <output_root>/config/fly/ so the
    cluster can read them. HP-source YAMLs are still pulled from the repo's
    config/fly/ (static, version-controlled)."""
    out_dir = cv_config_dir(output_root)
    os.makedirs(out_dir, exist_ok=True)
    written, skipped = [], []
    for base_name, winner_name in _wcfg.CONDITIONS:
        if hp_source == 'per_condition':
            hp_yaml_path = os.path.join(
                _REPO_ROOT, 'config', 'fly', f'{winner_name}.yaml')
        else:
            hp_yaml_path = os.path.join(
                _REPO_ROOT, 'config', 'fly', f'{hp_yaml_basename}.yaml')

        folds = list(range(n_folds)) if n_folds >= 1 else [None]
        for fold_i in folds:
            if fold_i is None:
                out_yaml = os.path.join(out_dir,
                    f'{base_name}_{suffix}.yaml')
                sim_seed = train_seed = None
            else:
                out_yaml = os.path.join(out_dir,
                    f'{base_name}_{suffix}_cv{fold_i:02d}.yaml')
                sim_seed   = 42 + fold_i
                train_seed = 1042 + fold_i
            if os.path.exists(out_yaml) and not force:
                skipped.append(out_yaml)
                continue
            ok = _wcfg.emit_one(base_name, hp_yaml_path, out_yaml,
                                suffix, _wcfg.YT_VOS_ROOT,
                                fold_i=fold_i, sim_seed=sim_seed,
                                train_seed=train_seed)
            if ok:
                written.append(out_yaml)
    print(f'  wrote {len(written)} YT YAMLs -> {out_dir}  '
          f'(skipped {len(skipped)} existing)')


def emit_tex_inplace(suffix, n_folds, output_root, output_tex):
    """In-process twin of scripts/emit_cross_table_rows.main() — no subprocess."""
    rows = []
    for base, label, nsig, ngam, edges in _emit.CONDITIONS:
        rows.append(_emit.emit_row(base, label, nsig, ngam, edges,
                                   output_root, 'fly', suffix, n_folds))
    out_dir = os.path.join(output_root, 'log')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, output_tex)
    with open(out_path, 'w') as f:
        f.write(f'% --- rows for YT-trained, DAVIS-cross-tested; '
                f'suffix={suffix} ---\n')
        for r in rows:
            f.write(r + '\n')
        f.write('% ' + '-' * 60 + '\n')
    print(f'  [tex ] {out_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('--output_root', default=DATA_ROOT,
                   help=f'Data root (default: {DATA_ROOT})')
    p.add_argument('--suffix', default=SUFFIX,
                   help=f'YT YAML suffix (default: {SUFFIX})')
    p.add_argument('--n_folds', type=int, default=5,
                   help='CV folds per condition (default: 5)')
    p.add_argument('--conditions', nargs='+',
                   default=_runner.CONDITION_BASES,
                   help='Subset of conditions (base names) to run.')
    p.add_argument('--node_name', default='a100',
                   help='Cluster GPU queue node name (default: a100)')
    p.add_argument('--hard_runtime_limit_min', type=int, default=120,
                   help='bsub -W limit in minutes (default: 120)')
    p.add_argument('--metrics_interval', type=int, default=300,
                   help='Print conn/Vr/τ R² for each cluster slot every N '
                        'seconds during the wait (default: 300).')
    p.add_argument('--mid_rollout', action='store_true',
                   help='During the cluster-training wait, run a silent '
                        'local data_test rollout per slot every '
                        '--metrics_interval seconds (YT model -> DAVIS CV '
                        'fold 0) and print the Pearson r with r2_color.')
    p.add_argument('--mid_rollout_frames', type=int, default=100,
                   help='Frames for the mid-training rollout (default: 100).')
    p.add_argument('--local_test_plot', action='store_true',
                   help='Run test+plot LOCALLY instead of on the cluster.')
    p.add_argument('--force_test', action='store_true',
                   help='Delete + re-run cross-test log and metrics.txt '
                        '(does NOT re-train).')
    p.add_argument('--force_yaml', action='store_true',
                   help='Overwrite existing <base>_<suffix>_cv*.yaml files.')
    p.add_argument('--skip_yaml', action='store_true',
                   help='Skip the YT YAML emission step (assume YAMLs exist).')
    p.add_argument('--emit_tex', default=None,
                   help='TeX filename emitted after each condition; '
                        'default cv_<suffix>_rows.tex')
    args = p.parse_args()

    emit_tex_name = args.emit_tex or f'cv_{args.suffix}_rows.tex'

    assert os.path.isdir(args.output_root), f'missing {args.output_root}'
    set_data_root(args.output_root)

    print('=' * 60)
    print('GNN conditions (per-condition winner HPs) — cluster training')
    print(f'  repo root:  {_REPO_ROOT}')
    print(f'  data root:  {args.output_root}')
    print(f'  suffix:     {args.suffix}')
    print(f'  hp source:  {HP_SOURCE}')
    print(f'  n folds:    {args.n_folds}')
    print(f'  node:       {args.node_name}')
    print(f'  tex out:    log/{emit_tex_name}')
    print(f'  force_test: {args.force_test}')
    print(f'  force_yaml: {args.force_yaml}')
    print('=' * 60)

    # Step 1 — emit YT-training YAMLs (per-condition HPs).
    if args.skip_yaml:
        print('\n[1] emit YT YAMLs — SKIPPED (--skip_yaml)')
    else:
        print('\n[1] emit YT YAMLs  (hp_source=per_condition)')
        emit_yt_yamls(HP_SOURCE, args.suffix, hp_yaml_basename=None,
                      n_folds=args.n_folds, force=args.force_yaml,
                      output_root=args.output_root)

    # Step 2 — per-condition cluster pipeline.
    base_cfg = NeuralGraphConfig.from_yaml(
        config_path('fly', f'{args.conditions[0]}.yaml'))
    device = set_device(base_cfg.training.device)

    for base_name in args.conditions:
        _runner.run_condition(
            base_name=base_name,
            suffix=args.suffix,
            n_folds=args.n_folds,
            device=device,
            output_root=args.output_root,
            node_name=args.node_name,
            hard_runtime_limit_min=args.hard_runtime_limit_min,
            force_test=args.force_test,
            emit_tex=None,  # handled below in-process
            cluster_test_plot=(not args.local_test_plot),
            metrics_interval=args.metrics_interval,
            mid_rollout=args.mid_rollout,
            mid_rollout_frames=args.mid_rollout_frames,
        )
        # In-process TeX refresh after each condition.
        emit_tex_inplace(args.suffix, args.n_folds,
                         args.output_root, emit_tex_name)

    # Step 3 — final TeX emission (idempotent).
    print('\n[3] final TeX')
    emit_tex_inplace(args.suffix, args.n_folds,
                     args.output_root, emit_tex_name)

    print('\n' + '=' * 60)
    print('GNN conditions complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()
