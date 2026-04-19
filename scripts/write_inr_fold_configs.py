"""
Emit per-fold YAML config files for the GNN+INR CV, mirroring the
in-memory fold construction done by scripts/run_inr_cv.py.

For each seed i in 0..N-1 and each condition C in {davis, yt}, writes:
    config/fly/<base>_<C>_cv<i>.yaml

with these overrides versus the base YAML:
    description  (tagged with seed and condition)
    dataset      = <base>_<C>_cv<i>
    simulation.seed    = 42 + i
    training.seed      = 1042 + i
    simulation.datavis_roots    (only for yt condition — YouTube-VOS roots)
    simulation.skip_short_videos (only for yt condition)

Does NOT overwrite existing YAMLs — delete them manually if you need a refresh.

Usage:
    python scripts/write_inr_fold_configs.py \\
        --config flyvis_noise_005_INR \\
        --n_seeds 5 \\
        [--conditions davis yt] \\
        [--force]
"""

import argparse
import os
import sys

import yaml


# Resolve repo root from this script's location (robust local + cluster).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)


CV_DATAVIS_ROOTS = ["/groups/saalfeld/home/kumarv4/web_datasets/YouTube-VOS"]


def emit_fold_yaml(base_yaml_path, out_yaml_path, base_name, condition, fold_i,
                   cv_datavis_roots=CV_DATAVIS_ROOTS):
    with open(base_yaml_path) as f:
        cfg = yaml.safe_load(f)

    run_name = f'{base_name}_{condition}_cv{fold_i:02d}'
    seed_sim   = 42 + fold_i
    seed_train = 1042 + fold_i

    original_desc = cfg.get('description', '').strip()
    cfg['description'] = (
        f'{original_desc}\n\n'
        f'[run_inr_cv fold] condition={condition} fold={fold_i} '
        f'sim_seed={seed_sim} train_seed={seed_train}.\n'
        f'Dataset/log dir: {run_name}.'
    )
    cfg['dataset'] = run_name

    sim = cfg.setdefault('simulation', {})
    sim['seed'] = seed_sim
    if condition == 'yt':
        sim['datavis_roots']     = list(cv_datavis_roots)
        sim['skip_short_videos'] = False
    # condition == 'davis' -> leave datavis_roots at base YAML default (DAVIS)

    tr = cfg.setdefault('training', {})
    tr['seed'] = seed_train

    with open(out_yaml_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='flyvis_noise_005_INR',
                   help='Base config (no .yaml, no pre-folder)')
    p.add_argument('--pre_folder', default='fly')
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--conditions', nargs='+',
                   default=['davis', 'yt'], choices=['davis', 'yt'])
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing fold YAMLs')
    args = p.parse_args()

    base_yaml = os.path.join(REPO_ROOT, 'config', args.pre_folder,
                             f'{args.config}.yaml')
    if not os.path.isfile(base_yaml):
        sys.exit(f'base YAML not found: {base_yaml}')

    out_dir = os.path.join(REPO_ROOT, 'config', args.pre_folder)
    os.makedirs(out_dir, exist_ok=True)

    written, skipped = [], []
    for i in range(args.n_seeds):
        for cond in args.conditions:
            out = os.path.join(out_dir, f'{args.config}_{cond}_cv{i:02d}.yaml')
            if os.path.exists(out) and not args.force:
                skipped.append(out)
                continue
            emit_fold_yaml(base_yaml, out, args.config, cond, i)
            written.append(out)

    print(f'wrote {len(written)} fold YAML(s):')
    for p_ in written:
        print(f'  {p_}')
    if skipped:
        print(f'skipped {len(skipped)} existing YAML(s) (use --force to overwrite):')
        for p_ in skipped:
            print(f'  {p_}')


if __name__ == '__main__':
    main()
