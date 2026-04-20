"""
Top-level entry point for the cross-check pipeline.

`run_all_conditions(hp_source, suffix, ...)` runs the full 5-fold CV across
all 8 conditions and drops the TeX table at
<output_root>/log/cv_<suffix>_rows.tex.

Used by the two 5-line orchestrator scripts at the repo root:
    run_GNN_conditions.py  (hp_source=per_condition)
    run_GNN_cross.py       (hp_source=uniform)
"""

import os

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.utils import config_path, set_data_root, set_device

from connectome_gnn.cross.pipeline import CONDITION_BASES, run_condition
from connectome_gnn.cross.yaml_io import emit_yt_yamls
from connectome_gnn.cross.tex import emit_tex_file


DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'


def run_all_conditions(hp_source, suffix, hp_yaml=None,
                        output_root=DATA_ROOT, n_folds=5,
                        node_name='a100', hard_runtime_limit_min=120,
                        metrics_interval=300, cluster_test_plot=True,
                        force_test=False):
    """Run the 8-condition × n_folds cross-check and emit the TeX table.

    Args:
        hp_source: 'per_condition' or 'uniform'.
        suffix:    YT YAML suffix (e.g. 'yt_per_cond' or 'yt_cross').
        hp_yaml:   HP-source YAML basename (only used when
                   hp_source == 'uniform').
    """
    assert os.path.isdir(output_root), f'missing {output_root}'
    set_data_root(output_root)

    print('=' * 60)
    print(f'GNN cross-check ({hp_source} HPs) — cluster training')
    print(f'  data root:  {output_root}')
    print(f'  suffix:     {suffix}')
    print(f'  hp source:  {hp_source}')
    if hp_yaml:
        print(f'  hp yaml:    {hp_yaml}')
    print(f'  n folds:    {n_folds}')
    print(f'  node:       {node_name}')
    print(f'  tex out:    log/cv_{suffix}_rows.tex')
    print('=' * 60)

    # Step 1 — emit YT CV YAMLs.
    print(f'\n[1] emit YT YAMLs  (hp_source={hp_source})')
    emit_yt_yamls(hp_source, suffix, hp_yaml_basename=hp_yaml,
                  n_folds=n_folds, output_root=output_root)

    # Step 2 — per-condition cluster pipeline.
    base_cfg = NeuralGraphConfig.from_yaml(
        config_path('fly', f'{CONDITION_BASES[0]}.yaml'))
    device = set_device(base_cfg.training.device)

    for base_name in CONDITION_BASES:
        run_condition(
            base_name=base_name, suffix=suffix, n_folds=n_folds,
            device=device, output_root=output_root, node_name=node_name,
            hard_runtime_limit_min=hard_runtime_limit_min,
            force_test=force_test, cluster_test_plot=cluster_test_plot,
            metrics_interval=metrics_interval,
        )
        emit_tex_file(suffix, output_root, n_folds=n_folds)

    # Step 3 — final TeX emission (idempotent).
    print('\n[3] final TeX')
    emit_tex_file(suffix, output_root, n_folds=n_folds)

    print('\n' + '=' * 60)
    print(f'GNN cross-check complete ({hp_source}).')
    print('=' * 60)
