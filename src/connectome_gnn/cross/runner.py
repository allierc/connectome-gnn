"""
Top-level entry points for the hold-out-only cross-check pipeline.

`run_all_conditions(hp_source, suffix, ...)` runs the full n_folds CV across
all 8 conditions (train + test on the hold-out dataset, using the held-out
20% of each fold for testing) and drops the TeX table at
<output_root>/log/cv_<suffix>_rows.tex.

Used by the orchestrator scripts at the repo root:
    run_GNN_conditions.py       (hp_source=per_condition)
    run_GNN_unique.py           (hp_source=uniform, GNN winner HPs)
    run_KnownODE_conditions.py  (hp_source=uniform, Known_ODE winner HPs)

`generate_all_yt_data(n_folds, ...)` — generate-only entry point used by
run_generate_holdout_data.py to pre-build the shared {base}_<tag>_cv{i:02d}
datasets before the three training runners are launched in parallel.
"""

import os

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.utils import (
    config_path, load_data_root_from_json, set_data_root, set_device,
)

from connectome_gnn.cross.pipeline import (
    CONDITION_BASES, run_condition, run_condition_wave,
    generate_yt_data_for_condition,
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
                        force_test=False, force_train=False, force_plot=False,
                        sim_overrides=None, dataset_tag=None,
                        condition_filter=None,
                        data_augmentation_loop=100,
                        data_augmentation_loop_overrides=None,
                        hp_yaml_overrides=None,
                        conditions_per_wave=1,
                        node_name_per_condition=None,
                        emit_tex=True,
                        skip_test_plot=False):
    """Run the 8-condition × n_folds and emit the TeX table.

    `output_root` resolution (highest priority wins):
        1. explicit kwarg
        2. $GNN_OUTPUT_ROOT env var
        3. cluster_data_dir in data_paths.json (same as GNN_LLM)

    Args:
        hp_source: 'per_condition' or 'uniform'.
        suffix:    Hold-out YAML suffix (e.g. 'yt_per_cond', 'yt_cross',
                   'yt_known_ode'). Output TeX goes to
                   log/cv_<suffix>_rows.tex.
        hp_yaml:   HP-source YAML basename (only used when
                   hp_source == 'uniform').
    """
    output_root = _resolve_output_root(output_root)

    print('=' * 60)
    print(f'Hold-out-only cross-check ({hp_source} HPs)')
    print(f'  data root:  {output_root}')
    print(f'  suffix:     {suffix}')
    print(f'  hp source:  {hp_source}')
    if hp_yaml:
        print(f'  hp yaml:    {hp_yaml}')
    print(f'  n folds:    {n_folds}')
    print(f'  node:       {node_name}')
    if emit_tex:
        print(f'  tex out:    log/cv_{suffix}_rows.tex')
    else:
        print(f'  tex out:    [disabled — emit_tex=False]')
    print('=' * 60)

    # Step 1 — emit hold-out CV YAMLs (idempotent; suffix-free dataset name
    # is shared with the other two training runners and the pre-gen script).
    print(f'\n[1] emit hold-out YAMLs  (hp_source={hp_source}, dataset_tag={dataset_tag})')
    if sim_overrides:
        print(f'    sim_overrides: {sim_overrides}')
    if condition_filter is not None:
        print(f'    condition_filter: {condition_filter}')
    emit_yt_yamls(hp_source, suffix, hp_yaml_basename=hp_yaml,
                  n_folds=n_folds, output_root=output_root,
                  sim_overrides=sim_overrides, dataset_tag=dataset_tag,
                  condition_filter=condition_filter,
                  data_augmentation_loop=data_augmentation_loop,
                  data_augmentation_loop_overrides=data_augmentation_loop_overrides,
                  hp_yaml_overrides=hp_yaml_overrides)

    # Step 2 — per-condition cluster pipeline.
    base_cfg = NeuralGraphConfig.from_yaml(
        config_path('fly', f'{CONDITION_BASES[0]}.yaml'))
    device = set_device(base_cfg.training.device)

    _active_bases = ([b for b in CONDITION_BASES if b in condition_filter]
                     if condition_filter is not None else CONDITION_BASES)
    assert conditions_per_wave >= 1

    # Per-condition node override — run one base at a time so each wave uses
    # exactly one LSF queue. Falls back to `node_name` for bases not in the dict.
    if node_name_per_condition:
        print(f'    node_name_per_condition active: forcing '
              f'conditions_per_wave=1 (per-base LSF queue control)')
        for base in _active_bases:
            effective_node = node_name_per_condition.get(base, node_name)
            print(f'  -> condition {base!r}: node=gpu_{effective_node}')
            run_condition_wave(
                base_names=[base], suffix=suffix, n_folds=n_folds,
                device=device, output_root=output_root,
                node_name=effective_node,
                hard_runtime_limit_min=hard_runtime_limit_min,
                force_test=force_test, cluster_test_plot=cluster_test_plot,
                metrics_interval=metrics_interval,
                force_train=force_train, force_plot=force_plot,
                skip_test_plot=skip_test_plot,
            )
            if emit_tex:
                emit_tex_file(suffix, output_root, n_folds=n_folds)
    else:
        n_waves = (len(_active_bases) + conditions_per_wave - 1) // conditions_per_wave
        print(f'    conditions_per_wave={conditions_per_wave} -> {n_waves} wave(s), '
              f'up to {conditions_per_wave * n_folds} concurrent cluster jobs per wave')
        for wave_i in range(n_waves):
            chunk = _active_bases[wave_i * conditions_per_wave:
                                  (wave_i + 1) * conditions_per_wave]
            run_condition_wave(
                base_names=chunk, suffix=suffix, n_folds=n_folds,
                device=device, output_root=output_root, node_name=node_name,
                hard_runtime_limit_min=hard_runtime_limit_min,
                force_test=force_test, cluster_test_plot=cluster_test_plot,
                metrics_interval=metrics_interval,
                force_train=force_train, force_plot=force_plot,
                skip_test_plot=skip_test_plot,
            )
            if emit_tex:
                emit_tex_file(suffix, output_root, n_folds=n_folds)

    # Step 3 — final TeX emission (idempotent).
    if emit_tex:
        print('\n[3] final TeX')
        emit_tex_file(suffix, output_root, n_folds=n_folds)

    print('\n' + '=' * 60)
    print(f'GNN hold-out-only cross-check complete ({hp_source}).')
    print('=' * 60)


def generate_all_yt_data(output_root=None, n_folds=5, suffix='yt_gen',
                          hp_source='per_condition', hp_yaml=None,
                          sim_overrides=None, dataset_tag=None,
                          condition_filter=None,
                          data_augmentation_loop=100,
                          data_augmentation_loop_overrides=None,
                          hp_yaml_overrides=None):
    """Pre-build all 8 × n_folds YT datasets at <output_root>/graphs_data/fly/
    {base}_yt_cv{i:02d}/. The three training runners share these datasets
    (their `ensure_yt_data` calls become noops) so they can run in parallel.

    The `suffix` here only drives the throwaway yaml filename used for
    data generation — the dataset name itself is suffix-free.
    """
    output_root = _resolve_output_root(output_root)

    print('=' * 60)
    print('Hold-out data pre-generation  (shared across all 3 training runners)')
    print(f'  data root:  {output_root}')
    print(f'  n folds:    {n_folds}')
    print(f'  suffix:     {suffix}  (yaml-only; dataset name is shared)')
    print('=' * 60)

    # 1. Emit hold-out CV YAMLs (hp_source=per_condition by default so each
    #    condition's own winner yaml supplies the graph_model block; the
    #    simulation block — which is all we need for generation — comesgit
    #    from the base yaml regardless of hp_source).
    print(f'\n[1] emit hold-out YAMLs  (hp_source={hp_source}, dataset_tag={dataset_tag})')
    if sim_overrides:
        print(f'    sim_overrides: {sim_overrides}')
    if condition_filter is not None:
        print(f'    condition_filter: {condition_filter}')
    emit_yt_yamls(hp_source, suffix, hp_yaml_basename=hp_yaml,
                  n_folds=n_folds, output_root=output_root,
                  sim_overrides=sim_overrides, dataset_tag=dataset_tag,
                  condition_filter=condition_filter,
                  data_augmentation_loop=data_augmentation_loop,
                  data_augmentation_loop_overrides=data_augmentation_loop_overrides,
                  hp_yaml_overrides=hp_yaml_overrides)

    # 2. Generate data for each condition × fold.
    base_cfg = NeuralGraphConfig.from_yaml(
        config_path('fly', f'{CONDITION_BASES[0]}.yaml'))
    device = set_device(base_cfg.training.device)

    _active_bases = ([b for b in CONDITION_BASES if b in condition_filter]
                     if condition_filter is not None else CONDITION_BASES)
    print('\n[2] generate hold-out data per condition')
    for base_name in _active_bases:
        generate_yt_data_for_condition(
            base_name=base_name, suffix=suffix, n_folds=n_folds,
            device=device, output_root=output_root,
        )

    print('\n' + '=' * 60)
    print('Hold-out data pre-generation complete.')
    print('=' * 60)
