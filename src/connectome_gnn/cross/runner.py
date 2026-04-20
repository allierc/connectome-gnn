"""
Top-level entry points for the YT-only cross-check pipeline.

`run_all_conditions(hp_source, suffix, ...)` runs the full n_folds CV across
all 8 conditions (train + test on YouTube-VOS, using the held-out 20% of
each fold for testing) and drops the TeX table at
<output_root>/log/cv_<suffix>_rows.tex.

Used by the orchestrator scripts at the repo root:
    run_GNN_conditions.py       (hp_source=per_condition)
    run_GNN_unique.py           (hp_source=uniform, GNN winner HPs)
    run_KnownODE_conditions.py  (hp_source=uniform, Known_ODE winner HPs)

`generate_all_yt_data(n_folds, ...)` — generate-only entry point used by
run_generate_YT_data.py to pre-build the shared {base}_yt_cv{i:02d}
datasets before the three training runners are launched in parallel.
"""

import os

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.utils import (
    config_path, load_data_root_from_json, set_data_root, set_device,
)

from connectome_gnn.cross.pipeline import (
    CONDITION_BASES, run_condition, generate_yt_data_for_condition,
)
from connectome_gnn.cross.yaml_io import emit_yt_yamls
from connectome_gnn.cross.tex import emit_tex_file


def _resolve_output_root(output_root):
    if output_root is None:
        output_root = os.environ.get('GNN_OUTPUT_ROOT') or load_data_root_from_json()
    assert output_root and os.path.isdir(output_root), (
        f'output_root not set or missing: {output_root!r}. '
        f'Set $GNN_OUTPUT_ROOT or cluster_data_dir in data_paths.json.'
    )
    set_data_root(output_root)
    return output_root


def run_all_conditions(hp_source, suffix, hp_yaml=None,
                        output_root=None, n_folds=5,
                        node_name='a100', hard_runtime_limit_min=120,
                        metrics_interval=300, cluster_test_plot=True,
                        force_test=False):
    """Run the 8-condition × n_folds YT-only cross-check and emit the TeX table.

    `output_root` resolution (highest priority wins):
        1. explicit kwarg
        2. $GNN_OUTPUT_ROOT env var
        3. cluster_data_dir in data_paths.json (same as GNN_LLM)

    Args:
        hp_source: 'per_condition' or 'uniform'.
        suffix:    YT YAML suffix (e.g. 'yt_per_cond', 'yt_cross',
                   'yt_known_ode'). Output TeX goes to
                   log/cv_<suffix>_rows.tex.
        hp_yaml:   HP-source YAML basename (only used when
                   hp_source == 'uniform').
    """
    output_root = _resolve_output_root(output_root)

    print('=' * 60)
    print(f'GNN YT-only cross-check ({hp_source} HPs)')
    print(f'  data root:  {output_root}')
    print(f'  suffix:     {suffix}')
    print(f'  hp source:  {hp_source}')
    if hp_yaml:
        print(f'  hp yaml:    {hp_yaml}')
    print(f'  n folds:    {n_folds}')
    print(f'  node:       {node_name}')
    print(f'  tex out:    log/cv_{suffix}_rows.tex')
    print('=' * 60)

    # Step 1 — emit YT CV YAMLs (idempotent; suffix-free dataset name
    # is shared with the other two training runners and the pre-gen script).
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
    print(f'GNN YT-only cross-check complete ({hp_source}).')
    print('=' * 60)


def generate_all_yt_data(output_root=None, n_folds=5, suffix='yt_gen',
                          hp_source='per_condition', hp_yaml=None):
    """Pre-build all 8 × n_folds YT datasets at <output_root>/graphs_data/fly/
    {base}_yt_cv{i:02d}/. The three training runners share these datasets
    (their `ensure_yt_data` calls become noops) so they can run in parallel.

    The `suffix` here only drives the throwaway yaml filename used for
    data generation — the dataset name itself is suffix-free.
    """
    output_root = _resolve_output_root(output_root)

    print('=' * 60)
    print('YT data pre-generation  (shared across all 3 training runners)')
    print(f'  data root:  {output_root}')
    print(f'  n folds:    {n_folds}')
    print(f'  suffix:     {suffix}  (yaml-only; dataset name is shared)')
    print('=' * 60)

    # 1. Emit YT CV YAMLs (hp_source=per_condition by default so each
    #    condition's own winner yaml supplies the graph_model block; the
    #    simulation block — which is all we need for generation — comes
    #    from the base yaml regardless of hp_source).
    print(f'\n[1] emit YT YAMLs  (hp_source={hp_source})')
    emit_yt_yamls(hp_source, suffix, hp_yaml_basename=hp_yaml,
                  n_folds=n_folds, output_root=output_root)

    # 2. Generate data for each condition × fold.
    base_cfg = NeuralGraphConfig.from_yaml(
        config_path('fly', f'{CONDITION_BASES[0]}.yaml'))
    device = set_device(base_cfg.training.device)

    print('\n[2] generate YT data per condition')
    for base_name in CONDITION_BASES:
        generate_yt_data_for_condition(
            base_name=base_name, suffix=suffix, n_folds=n_folds,
            device=device, output_root=output_root,
        )

    print('\n' + '=' * 60)
    print('YT data pre-generation complete.')
    print('=' * 60)
