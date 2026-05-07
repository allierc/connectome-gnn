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
    BASELINE_PREFIX, emit_fold_yaml, fold_dataset_name,
)


FOLDS = [0, 1, 2, 3, 4]

# Top-9 positive controls by null_dim, descending.
CONDITIONS = [
    'TmY15_005', 'TmY15_heaviside',  # 43,299
    'Mi1_005',   'Mi1_heaviside',    # 25,834
    'Tm3_005',   'Tm3_heaviside',    # 20,471
    'Tm4_005',   'Tm4_heaviside',    # 15,971
    'Tm1_005',   'Tm1_heaviside',    # 15,525
    'Mi4_005',   'Mi4_heaviside',    # 14,439
    'T4c_005',   'T4c_heaviside',    # 12,564
    'Mi9_005',   'Mi9_heaviside',    # 11,889
    'Tm2_005',   'Tm2_heaviside',    # 11,068
]


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
        f"baseline datasets missing for folds {missing}. "
        f"Run the unified blank50 generator first."
    )


for cond in CONDITIONS:
    for fold in FOLDS:
        out_ds = fold_dataset_name(cond, fold)
        fold_yaml = emit_fold_yaml(cond, fold)
        print(f'\n=== {cond}  cv{fold:02d}  ({out_ds}) ===', flush=True)
        print(f'config: {fold_yaml}', flush=True)
        config = NeuralGraphConfig.from_yaml(fold_yaml)
        add_optogenetics_stimulus(config)
