"""Pre-generate the optogenetics sweep datasets across all CV folds.

Mirror of run_generate_blank50.py for the opto experiment. For each enabled
(condition × fold) pair, emits a per-fold YAML into <data_root>/config/fly/
and calls add_optogenetics_stimulus(config) on it.

Final output: enabled_conditions × len(FOLDS) datasets at
    <output_root>/graphs_data/fly/<base>_opto_<target>_<wf>_cv<XX>/

Comment lines in CONDITIONS or shrink FOLDS to skip individual runs.
Downstream training runner: run_GNN_optogenetics.py.

No CLI flags — edit CONDITIONS / FOLDS or scripts/generate_opto_configs.py
to change the sweep.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.generators.optogenetics import add_optogenetics_stimulus  # noqa: E402
from connectome_gnn.utils import (  # noqa: E402
    graphs_data_path, load_data_root_from_json, set_data_root,
)
from _opto_cv_yaml import (  # noqa: E402
    BASELINE_PREFIX, emit_fold_yaml, emit_gen_yaml, fold_dataset_name,
)


FOLDS = [0, 1, 2, 3, 4]

# Top-9 positive controls by null_dim. Comment a target row to drop both its
# variants; comment a waveform string to drop that waveform across all targets.
TARGETS = [
    'TmY15',  # 43,299  (rank 1)
    'Mi1',    # 25,834
    'Tm3',    # 20,471
    'Tm4',    # 15,971
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


try:
    set_data_root(load_data_root_from_json())
except Exception:
    pass


def _baseline_exists(fold: int) -> bool:
    voltage = os.path.join(
        graphs_data_path('fly', f'{BASELINE_PREFIX}_cv{fold:02d}'),
        'x_list_train', 'voltage.zarr',
    )
    return os.path.isdir(voltage)


missing = [f for f in FOLDS if not _baseline_exists(f)]
if missing:
    sys.exit(
        f"baseline datasets missing for folds {missing} "
        f"(expected at {graphs_data_path('fly', BASELINE_PREFIX + '_cv{XX}')}). "
        f"Run the unified blank50 generator first."
    )


for cond in CONDITIONS:
    # Step 1: emit the per-condition '_gen' template into <data_root>/config/fly/
    # (mirrors generate_all_yt_data's behaviour from run_generate_blank50.py).
    gen_yaml = emit_gen_yaml(cond)
    print(f'\n>>> condition={cond}  gen config: {gen_yaml}', flush=True)

    for fold in FOLDS:
        # Step 2: per-fold safety — re-confirm the source data folder exists.
        src_dir = graphs_data_path('fly', f'{BASELINE_PREFIX}_cv{fold:02d}')
        src_voltage = os.path.join(src_dir, 'x_list_train', 'voltage.zarr')
        if not os.path.isdir(src_voltage):
            print(f'  SKIP cv{fold:02d}: source missing at {src_dir}', flush=True)
            continue

        # Step 3: emit the per-fold _cvXX YAML into <data_root>/config/fly/.
        out_ds = fold_dataset_name(cond, fold)
        fold_yaml = emit_fold_yaml(cond, fold)
        print(f'\n=== {cond}  cv{fold:02d}  ({out_ds}) ===', flush=True)
        print(f'config: {fold_yaml}', flush=True)
        print(f'source: {src_dir}', flush=True)

        # Step 4: re-simulate with the configured opto current.
        config = NeuralGraphConfig.from_yaml(fold_yaml)
        # Mirror GNN_Main.py:249-250 — prefix dataset with the YAML's parent
        # directory (here always 'fly/') so add_optogenetics_stimulus writes to
        # graphs_data/fly/<dataset>/ rather than graphs_data/<dataset>/.
        if not config.dataset.startswith('fly/'):
            config.dataset = 'fly/' + config.dataset
        add_optogenetics_stimulus(config)
