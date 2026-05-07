"""Pre-generate the optogenetics sweep datasets sequentially.

Mirror of run_generate_blank50.py for the opto experiment: each entry in
CONDITIONS names a YAML produced by scripts/generate_opto_configs.py;
add_optogenetics_stimulus(config) is called on it in turn.

The baseline source dataset (flyvis_noise_free_blank50_cv00) must already
exist on disk. Datasets land at:

    <output_root>/graphs_data/fly/<base>_opto_<target>_<waveform>/

Comment lines in CONDITIONS to skip individual conditions. Downstream
training runner: run_GNN_optogenetics.py.

No CLI flags — edit CONDITIONS or scripts/generate_opto_configs.py to
change the sweep.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.optogenetics import add_optogenetics_stimulus
from connectome_gnn.utils import (
    config_path, get_data_root, graphs_data_path,
    load_data_root_from_json, set_data_root,
)


# Source dataset (must already exist on disk).
SOURCE_DATASET = 'flyvis_noise_free_blank50_cv00'
# Output config / dataset prefix — opto runs aren't per-fold so we drop _cv00.
OUTPUT_PREFIX = 'flyvis_noise_free_blank50'

# Must match the conditions in scripts/generate_opto_configs.py and
# run_GNN_optogenetics.py. Top-9 positive controls by null_dim, descending.
CONDITIONS = [
    'TmY15_white_noise', 'TmY15_heaviside',  # 43,299
    'Mi1_white_noise',   'Mi1_heaviside',    # 25,834
    'Tm3_white_noise',   'Tm3_heaviside',    # 20,471
    'Tm4_white_noise',   'Tm4_heaviside',    # 15,971
    'Tm1_white_noise',   'Tm1_heaviside',    # 15,525
    'Mi4_white_noise',   'Mi4_heaviside',    # 14,439
    'T4c_white_noise',   'T4c_heaviside',    # 12,564
    'Mi9_white_noise',   'Mi9_heaviside',    # 11,889
    'Tm2_white_noise',   'Tm2_heaviside',    # 11,068
]


try:
    set_data_root(load_data_root_from_json())
except Exception:
    pass


def _config_path_for(cond):
    name = f'{OUTPUT_PREFIX}_opto_{cond}.yaml'
    for c in (
        config_path('fly', name),
        os.path.join(get_data_root(), 'config', 'fly', name),
    ):
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f'opto config for {cond!r} not found')


_baseline_voltage = os.path.join(
    graphs_data_path('fly', SOURCE_DATASET), 'x_list_train', 'voltage.zarr'
)
if not os.path.isdir(_baseline_voltage):
    sys.exit(
        f"baseline {SOURCE_DATASET!r} not found at "
        f"{graphs_data_path('fly', SOURCE_DATASET)}; generate it via the "
        f"unified blank50 pipeline first."
    )


for cond in CONDITIONS:
    cfg_path = _config_path_for(cond)
    print(f'\n=== {cond} ===', flush=True)
    print(f'config: {cfg_path}', flush=True)
    config = NeuralGraphConfig.from_yaml(cfg_path)
    add_optogenetics_stimulus(config)
