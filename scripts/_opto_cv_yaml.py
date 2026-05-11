"""Shared helper: emit per-CV-fold opto YAMLs into <data_root>/config/fly/.

Used by run_generate_optogenetics.py and run_GNN_optogenetics.py so both
runners agree on:
  * the master template location (repo's config/fly/<prefix>_opto_<cond>.yaml)
  * the per-fold patches (source_dataset, dataset, seed, output_suffix)
  * the on-disk location for emitted per-fold YAMLs
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

import yaml  # noqa: E402

from connectome_gnn.utils import config_path, get_data_root  # noqa: E402


BASELINE_PREFIX = 'flyvis_noise_free_blank50'
OUTPUT_PREFIX = 'flyvis_noise_free_blank50'
SIM_SEED_BASE = 42  # cv00 → seed 42, cv01 → 43, ..., matches unified blank50


def master_yaml_for(cond: str) -> str:
    """Return path to the per-condition master YAML in the repo's config/fly/."""
    name = f'{OUTPUT_PREFIX}_opto_{cond}.yaml'
    candidates = (
        config_path('fly', name),
        os.path.join(get_data_root(), 'config', 'fly', name),
    )
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f'master opto config for {cond!r} not found')


def emit_gen_yaml(cond: str) -> str:
    """Emit the per-condition '_gen' template into <data_root>/config/fly/.

    Mirrors the convention from run_generate_blank50.py / generate_all_yt_data:
    a fold-less generation config that documents the master parameters used to
    produce all 5 cv splits. opto.enabled is FALSE here so this YAML is never
    accidentally used for re-simulation; the actual generation is dispatched
    per-fold via emit_fold_yaml.
    """
    with open(master_yaml_for(cond)) as f:
        cfg = yaml.safe_load(f)
    out_name = f'{OUTPUT_PREFIX}_opto_{cond}_gen'
    cfg['dataset'] = out_name
    cfg['config_file'] = f'fly/{out_name}'
    # Mark as generation template, not a runnable opto config.
    cfg['simulation']['optogenetics']['enabled'] = False
    cfg['simulation']['optogenetics']['source_dataset'] = ''
    cfg['simulation']['optogenetics']['output_suffix'] = ''
    base_desc = cfg.get('description', '')
    cfg['description'] = f"{base_desc}  (gen template — see _cv00..cv04 for runnable configs)"

    out_dir = os.path.join(get_data_root(), 'config', 'fly')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{out_name}.yaml')
    with open(out_path, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out_path


def emit_fold_yaml(cond: str, fold: int) -> str:
    """Patch master YAML for this fold, write into <data_root>/config/fly/, return path."""
    src_dataset = f'{BASELINE_PREFIX}_cv{fold:02d}'
    out_dataset = f'{OUTPUT_PREFIX}_opto_{cond}_cv{fold:02d}'
    with open(master_yaml_for(cond)) as f:
        cfg = yaml.safe_load(f)
    cfg['dataset'] = out_dataset
    cfg['config_file'] = f'fly/{out_dataset}'
    cfg['simulation']['seed'] = SIM_SEED_BASE + fold
    cfg['simulation']['optogenetics']['source_dataset'] = src_dataset
    cfg['simulation']['optogenetics']['output_suffix'] = f'_opto_{cond}_cv{fold:02d}'
    base_desc = cfg.get('description', '')
    cfg['description'] = (
        f"{base_desc}  (CV fold {fold:02d}, source={src_dataset})"
    )

    out_dir = os.path.join(get_data_root(), 'config', 'fly')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{out_dataset}.yaml')
    with open(out_path, 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out_path


def fold_dataset_name(cond: str, fold: int) -> str:
    return f'{OUTPUT_PREFIX}_opto_{cond}_cv{fold:02d}'
