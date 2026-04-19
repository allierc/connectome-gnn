"""
Emit per-condition YouTube-VOS training configs for the cross-check table.

For each of 8 conditions, emit <base>_<suffix>.yaml whose contents are:

  - `simulation` block from the condition's own BASE yaml (data spec:
    noise, edges, stride, hidden, etc. — condition-specific and must not
    change).
  - `graph_model`, `plotting`, `training`, `claude` blocks from a HP-source
    yaml: either
        (a) the condition's own `_winner.yaml`  -> `--hp_source per_condition`
            (rewrites `run_GNN_conditions.sh`'s notion of "winner HPs")
        (b) a single shared yaml (default: null_edges_pc_400_winner) ->
            `--hp_source uniform` (for `run_GNN_cross.sh`, testing whether
            one set of HPs transfers across conditions).
  - `simulation.datavis_roots` overridden to YouTube-VOS.
  - `dataset` renamed to <base>_<suffix>.

Does NOT overwrite existing files unless --force is given.

Usage:
    python scripts/write_cross_yt_configs.py \\
        --hp_source per_condition \\
        --suffix yt_per_cond \\
        [--force]

    python scripts/write_cross_yt_configs.py \\
        --hp_source uniform \\
        --hp_yaml flyvis_noise_005_null_edges_pc_400_winner \\
        --suffix yt_cross \\
        [--force]
"""

import argparse
import os
import sys

import yaml


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


YT_VOS_ROOT = "/groups/saalfeld/home/kumarv4/web_datasets/YouTube-VOS"


# (condition_basename_for_data, condition_basename_for_winner_hps)
# First name = base (non-winner) yaml — source of simulation block.
# Second name = winner yaml — source of HPs when --hp_source per_condition.
CONDITIONS = [
    ('flyvis_noise_free',                'flyvis_noise_free_winner'),
    ('flyvis_noise_005',                 'flyvis_noise_005_winner'),
    ('flyvis_noise_05',                  'flyvis_noise_05_winner'),
    ('flyvis_noise_005_010',             'flyvis_noise_005_010_winner'),
    ('flyvis_noise_005_null_edges_pc_400', 'flyvis_noise_005_null_edges_pc_400_winner'),
    ('flyvis_noise_005_removed_pc_20',   'flyvis_noise_005_removed_pc_20_winner'),
    ('flyvis_noise_005_stride_5',        'flyvis_noise_005_stride_5_winner'),
    ('flyvis_noise_005_hidden_010_ngp',  'flyvis_noise_005_hidden_010_ngp_winner'),
]


def emit_one(base_name, hp_yaml_path, out_yaml_path, suffix, yt_root,
             fold_i=None, sim_seed=None, train_seed=None):
    base_yaml = os.path.join(REPO_ROOT, 'config', 'fly', f'{base_name}.yaml')
    if not os.path.isfile(base_yaml):
        print(f'WARN: missing base yaml {base_yaml} — skipping', file=sys.stderr)
        return False
    if not os.path.isfile(hp_yaml_path):
        print(f'WARN: missing HP yaml {hp_yaml_path} — skipping', file=sys.stderr)
        return False

    with open(base_yaml) as f:
        base = yaml.safe_load(f)
    with open(hp_yaml_path) as f:
        hp = yaml.safe_load(f)

    # Merge: simulation from base, everything else from hp, then override.
    merged = dict(hp)
    merged['simulation'] = dict(base['simulation'])
    # Stimulus swap to YouTube-VOS.
    merged['simulation']['datavis_roots']     = [yt_root]
    merged['simulation']['skip_short_videos'] = False
    # Blank 50% of training frames (blank_freq=2 → every other frame is
    # zero stimulus). Lets neurons decay back to V_rest, which
    # dramatically improves V_rest parameter recovery during training.
    # Only applied to the YT *training* configs; DAVIS CV test configs
    # (see run_cross_yt_parallel.emit_davis_cv_yaml) keep plain stimuli.
    _vit = str(merged['simulation'].get('visual_input_type', 'DAVIS'))
    if 'blank' not in _vit:
        merged['simulation']['visual_input_type'] = _vit + '_blank'
    merged['simulation']['blank_freq'] = 2
    # Per-fold seeds (CV convention: sim_seed = 42+i, train_seed = 1042+i).
    if sim_seed is not None:
        merged['simulation']['seed'] = sim_seed
    if 'training' in merged and train_seed is not None:
        merged['training'] = dict(merged['training'])
        merged['training']['seed'] = train_seed
    # Identity / dataset naming. No `fly/` prefix — train_subprocess.py
    # derives the pre_folder from config_file and prepends it with a guard.
    if fold_i is not None:
        new_name = f'{base_name}_{suffix}_cv{fold_i:02d}'
    else:
        new_name = f'{base_name}_{suffix}'
    merged['dataset']     = new_name
    fold_tag = f' fold={fold_i}' if fold_i is not None else ''
    merged['description'] = (
        f'Cross-check YT-training variant of {base_name}{fold_tag}. '
        f'Data: simulation block from {base_name}.yaml, '
        f'stimulus swapped to YouTube-VOS, '
        f'sim_seed={sim_seed} train_seed={train_seed}. '
        f'HPs: {os.path.basename(hp_yaml_path)}.'
    )

    with open(out_yaml_path, 'w') as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hp_source', choices=['per_condition', 'uniform'],
                   default='uniform')
    p.add_argument('--hp_yaml', default='flyvis_noise_005_null_edges_pc_400_winner',
                   help='Only used when --hp_source=uniform')
    p.add_argument('--suffix', default='yt_cross',
                   help='Output filename suffix: <base>_<suffix>.yaml')
    p.add_argument('--yt_root', default=YT_VOS_ROOT)
    p.add_argument('--n_folds', type=int, default=5,
                   help='If >=1, emit <base>_<suffix>_cv<i>.yaml for each '
                        'fold i with CV seed convention '
                        '(sim_seed=42+i, train_seed=1042+i). '
                        'If 0, emit a single <base>_<suffix>.yaml (no CV).')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    written, skipped = [], []
    for base_name, winner_name in CONDITIONS:
        if args.hp_source == 'per_condition':
            hp_yaml_path = os.path.join(REPO_ROOT, 'config', 'fly', f'{winner_name}.yaml')
        else:
            hp_yaml_path = os.path.join(REPO_ROOT, 'config', 'fly', f'{args.hp_yaml}.yaml')

        folds = list(range(args.n_folds)) if args.n_folds >= 1 else [None]
        for fold_i in folds:
            if fold_i is None:
                out_yaml_path = os.path.join(
                    REPO_ROOT, 'config', 'fly', f'{base_name}_{args.suffix}.yaml')
                sim_seed = train_seed = None
            else:
                out_yaml_path = os.path.join(
                    REPO_ROOT, 'config', 'fly',
                    f'{base_name}_{args.suffix}_cv{fold_i:02d}.yaml')
                sim_seed   = 42 + fold_i
                train_seed = 1042 + fold_i
            if os.path.exists(out_yaml_path) and not args.force:
                skipped.append(out_yaml_path)
                continue
            ok = emit_one(base_name, hp_yaml_path, out_yaml_path,
                          args.suffix, args.yt_root,
                          fold_i=fold_i,
                          sim_seed=sim_seed, train_seed=train_seed)
            if ok:
                written.append(out_yaml_path)

    print(f'wrote {len(written)} YAMLs')
    for p_ in written:
        print(f'  {p_}')
    if skipped:
        print(f'skipped {len(skipped)} existing (use --force to overwrite)')


if __name__ == '__main__':
    main()
