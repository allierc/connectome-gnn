"""
YAML I/O for the hold-out-only cross-check pipeline.

Emits hold-out-training CV YAMLs (`<base>_<suffix>_cv<i>.yaml`) into
the shared-FS CV config dir (`<output_root>/config/fly/`). The dataset
name inside each yaml is suffix-free (`<base>_<tag>_cv<i>`) so the three
training runners (run_GNN_conditions / run_GNN_unique /
run_KnownODE_conditions) and the pre-gen script all share the same
hold-out datasets at <output_root>/graphs_data/fly/<base>_<tag>_cv<i>/.

The short on-disk `<tag>` is looked up in ROOT_TAGS from HOLDOUT_DS_ROOT;
pass `dataset_tag=...` explicitly to override.
"""

import os
import sys

import yaml

from connectome_gnn.utils import get_repo_root


# Map of datavis root path -> short dataset tag used in on-disk dataset dirs.
# Add new roots here when you want to use them.
ROOT_TAGS = {
    "/groups/saalfeld/home/kumarv4/web_datasets/YouTube-VOS":            "yt",
    "/groups/saalfeld/home/kumarv4/web_datasets/DAVIS2017-partial-test": "davis2017_pt",
}


HOLDOUT_DS_ROOT = "/groups/saalfeld/home/kumarv4/web_datasets/DAVIS2017-partial-test"


# (condition_basename_for_data, condition_basename_for_winner_hps)
# First name = base (non-winner) yaml — source of simulation block.
# Second name = winner yaml — source of HPs when hp_source=per_condition.
CONDITIONS = [
    ('flyvis_noise_free',                'flyvis_noise_free_winner'),
    ('flyvis_noise_005',                 'flyvis_noise_005_winner'),
    ('flyvis_noise_05',                  'flyvis_noise_05_winner'),
    ('flyvis_noise_005_010',             'flyvis_noise_005_010_winner'),
    ('flyvis_noise_005_020',             'flyvis_unified_winner'),
    ('flyvis_noise_005_null_edges_pc_400', 'flyvis_noise_005_null_edges_pc_400_winner'),
    ('flyvis_noise_005_removed_pc_20',   'flyvis_noise_005_removed_pc_20_winner'),
    ('flyvis_noise_005_removed_pc_50',   'flyvis_unified_winner'),
    ('flyvis_noise_005_stride_5',        'flyvis_noise_005_stride_5_winner'),
    ('flyvis_noise_005_hidden_010_ngp',  'flyvis_noise_005_hidden_010_ngp_anchors_winner'),
    ('flyvis_noise_005_hidden_020_ngp',  'flyvis_noise_005_hidden_020_ngp_anchors_winner'),
    # AR(1) measurement-noise sweep (blank50 + gamma=0.10 + temporal noise correlation rho).
    # Base yamls carry the simulation.noise_ar1_rho field; sim block propagates through
    # emit_one. HP source = flyvis_noise_005_010_winner (same training recipe as the
    # i.i.d. noise_005_010 condition).
    ('flyvis_noise_005_010_blank50_ar1_rho00', 'flyvis_noise_005_010_winner'),
    ('flyvis_noise_005_010_blank50_ar1_rho05', 'flyvis_noise_005_010_winner'),
    ('flyvis_noise_005_010_blank50_ar1_rho10', 'flyvis_noise_005_010_winner'),
]


def cv_config_dir(output_root):
    """<output_root>/config/fly/ — shared-FS CV config directory."""
    return os.path.join(output_root, 'config', 'fly')


def shared_cv_yaml_path(config_file, output_root):
    """Absolute shared-FS path for a CV YAML."""
    basename = os.path.basename(config_file) + '.yaml'
    return os.path.join(cv_config_dir(output_root), basename)


def _load_yaml_either(cfg_name, output_root):
    """Prefer <output_root>/config/fly/<name>.yaml; fall back to the repo's
    config/fly/ for static base YAMLs."""
    shared = os.path.join(cv_config_dir(output_root), f'{cfg_name}.yaml')
    if os.path.isfile(shared):
        return shared
    return os.path.join(get_repo_root(), 'config', 'fly', f'{cfg_name}.yaml')


def emit_one(base_name, hp_yaml_path, out_yaml_path, suffix, yt_root,
             fold_i=None, sim_seed=None, train_seed=None,
             sim_overrides=None, dataset_tag=None,
             data_augmentation_loop=100,
             data_augmentation_loop_overrides=None):
    """Emit one hold-out training YAML by merging:
    - simulation block from <repo>/config/fly/<base_name>.yaml
    - graph_model / training / plotting / claude from hp_yaml_path
    - stimulus swap to hold-out dataset, blank_freq=2, n_epochs=1, DAL=100
    Writes to out_yaml_path. Returns True on success.

    dataset_tag defaults to ROOT_TAGS[yt_root] — add an entry to ROOT_TAGS
    when introducing a new datavis root.
    """
    if dataset_tag is None:
        if yt_root not in ROOT_TAGS:
            raise KeyError(
                f'No dataset_tag registered for root {yt_root!r}. '
                f'Add an entry to ROOT_TAGS in {__file__} or pass '
                f'dataset_tag=... explicitly.')
        dataset_tag = ROOT_TAGS[yt_root]
    base_yaml = os.path.join(get_repo_root(), 'config', 'fly', f'{base_name}.yaml')
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

    merged = dict(hp)
    merged['simulation'] = dict(base['simulation'])
    merged['simulation']['datavis_roots']     = [yt_root]
    merged['simulation']['skip_short_videos'] = False
    # Optional simulation-block overrides (e.g. all_columns=True for the
    # full-fly variant that uses all 721 retinotopic columns).
    if sim_overrides:
        merged['simulation'].update(sim_overrides)

    # Preserve condition-specific graph_model knobs from the base yaml when
    # the HP yaml doesn't set them. Without this, uniform-HP runs silently
    # drop keys like hidden_neuron_fraction, making the "hidden" condition
    # trivially easy (the model sees all neurons).
    _base_gm = base.get('graph_model') or {}
    _hp_gm   = dict(merged.get('graph_model') or {})
    for _k, _v in _base_gm.items():
        if _k in _hp_gm:
            continue
        if any(tok in _k for tok in ('hidden', 'ngp', 'nnr', 'inr', 'anchor')):
            _hp_gm[_k] = _v
    merged['graph_model'] = _hp_gm

    # Fixed training budget across all 8 conditions. Per-condition DAL
    # overrides let expensive conditions (e.g. null_edges_pc_400 with 5×
    # edges) run a smaller gradient budget so wall time stays in range.
    # Rebuild training to drop the HP yaml's own n_epochs/DAL (avoids the
    # duplicate-key cosmetic artifact when yaml.safe_dump is called).
    _dal = data_augmentation_loop
    if data_augmentation_loop_overrides and base_name in data_augmentation_loop_overrides:
        _dal = data_augmentation_loop_overrides[base_name]
    if 'training' in merged:
        merged['training'] = {
            k: v for k, v in merged['training'].items()
            if k not in ('n_epochs', 'data_augmentation_loop')
        }
        merged['training']['n_epochs'] = 1
        merged['training']['data_augmentation_loop'] = _dal

    # Condition-defining training knobs always come from the base yaml. These
    # describe the data regime (e.g. stride_5 BPTT) rather than tunable HPs,
    # so a uniform HP yaml must not be allowed to silently disable them.
    _base_tr = base.get('training') or {}
    if 'training' in merged:
        for _k in ('recurrent_training', 'time_step'):
            if _k in _base_tr:
                merged['training'][_k] = _base_tr[_k]
    if 'claude' in merged:
        merged['claude'] = {
            k: v for k, v in merged['claude'].items()
            if k not in ('n_epochs', 'data_augmentation_loop')
        }
        merged['claude']['n_epochs'] = 1
        merged['claude']['data_augmentation_loop'] = _dal

    if sim_seed is not None:
        merged['simulation']['seed'] = sim_seed
    if 'training' in merged and train_seed is not None:
        merged['training'] = dict(merged['training'])
        merged['training']['seed'] = train_seed

    # YAML filename keeps the suffix (drives config_file -> log dir, so
    # run_GNN_conditions and run_GNN_unique stay in distinct log dirs).
    # dataset is suffix-free so the underlying hold-out training data (which
    # only depends on the base + seed, not on the HP block) is shared
    # between the two scripts.
    if fold_i is not None:
        yaml_name    = f'{base_name}_{suffix}_cv{fold_i:02d}'
        dataset_name = f'{base_name}_{dataset_tag}_cv{fold_i:02d}'
    else:
        yaml_name    = f'{base_name}_{suffix}'
        dataset_name = f'{base_name}_{dataset_tag}'
    merged['dataset']     = dataset_name
    # Always point config_file at this emitted YAML. Some winner yamls carry
    # a stale config_file pointing at the base condition (e.g.
    # flyvis_noise_005_hidden_010_ngp_winner.yaml embeds
    # `config_file: fly/flyvis_noise_005_hidden_010_ngp`). Without this line,
    # that stale value survives the merge and submit_cluster_job can't find
    # the emitted YAML in the shared-FS config dir.
    merged['config_file'] = f'fly/{yaml_name}'
    fold_tag = f' fold={fold_i}' if fold_i is not None else ''
    merged['description'] = (
        f'Cross-check YT-training variant of {base_name}{fold_tag} '
        f'({suffix}). sim_seed={sim_seed} train_seed={train_seed}. '
        f'HPs: {os.path.basename(hp_yaml_path)}.'
    )

    with open(out_yaml_path, 'w') as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return True


def emit_davis_cv_yaml(base_name, fold_i, output_root, force=False):
    """Emit <output_root>/config/fly/<base>_cv<i:02d>.yaml — copy of
    <repo>/config/fly/<base>.yaml with simulation.seed = 42 + fold_i
    and dataset = <base>_cv<i:02d>. Returns True if written."""
    src = os.path.join(get_repo_root(), 'config', 'fly', f'{base_name}.yaml')
    out_dir = cv_config_dir(output_root)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f'{base_name}_cv{fold_i:02d}.yaml')
    if os.path.exists(dst) and not force:
        return False
    if not os.path.isfile(src):
        print(f'  [warn] missing DAVIS base yaml {src}')
        return False
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg['simulation'] = dict(cfg.get('simulation', {}))
    cfg['simulation']['seed'] = 42 + fold_i
    cfg['dataset'] = f'{base_name}_cv{fold_i:02d}'
    cfg['description'] = (
        f'DAVIS CV fold {fold_i} of {base_name} (sim_seed={42 + fold_i}).'
    )
    with open(dst, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return True


def emit_yt_yamls(hp_source, suffix, hp_yaml_basename, n_folds, output_root,
                   sim_overrides=None, dataset_tag=None,
                   condition_filter=None, data_augmentation_loop=100,
                   data_augmentation_loop_overrides=None,
                   hp_yaml_overrides=None):
    """Emit hold-out CV YAMLs for all 8 conditions × n_folds, into
    <output_root>/config/fly/. Always overwrites existing files so HP
    tweaks in the source yamls propagate on every run."""
    out_dir = cv_config_dir(output_root)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    _active = [(b, w) for (b, w) in CONDITIONS
               if condition_filter is None or b in condition_filter]
    for base_name, winner_name in _active:
        # Per-condition HP yaml override lets a uniform-mode pipeline opt
        # specific conditions into their dedicated winner (e.g. stride_5's
        # BPTT recipe or the NGP+anchors recipe for hidden_*_ngp, which the
        # uniform noise_005-style HP yaml can't represent).
        if hp_yaml_overrides and base_name in hp_yaml_overrides:
            hp_yaml_path = os.path.join(
                get_repo_root(), 'config', 'fly',
                f'{hp_yaml_overrides[base_name]}.yaml')
        elif hp_source == 'per_condition':
            hp_yaml_path = os.path.join(
                get_repo_root(), 'config', 'fly', f'{winner_name}.yaml')
        else:
            hp_yaml_path = os.path.join(
                get_repo_root(), 'config', 'fly', f'{hp_yaml_basename}.yaml')

        folds = list(range(n_folds)) if n_folds >= 1 else [None]
        for fold_i in folds:
            if fold_i is None:
                out_yaml = os.path.join(out_dir, f'{base_name}_{suffix}.yaml')
                sim_seed = train_seed = None
            else:
                out_yaml = os.path.join(
                    out_dir, f'{base_name}_{suffix}_cv{fold_i:02d}.yaml')
                sim_seed   = 42 + fold_i
                train_seed = 1042 + fold_i
            ok = emit_one(base_name, hp_yaml_path, out_yaml,
                          suffix, HOLDOUT_DS_ROOT,
                          fold_i=fold_i, sim_seed=sim_seed,
                          train_seed=train_seed,
                          sim_overrides=sim_overrides,
                          dataset_tag=dataset_tag,
                          data_augmentation_loop=data_augmentation_loop,
                          data_augmentation_loop_overrides=data_augmentation_loop_overrides)
            if ok:
                written.append(out_yaml)
    print(f'  wrote {len(written)} hold-out YAMLs -> {out_dir}  (always overwrites)')
