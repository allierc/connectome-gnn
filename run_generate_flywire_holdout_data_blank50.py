"""Pre-generate hold-out CV datasets for the 4 flywireRF v2 connectomes
under blank50 + DAVIS2017-partial-test.

Each dataset is shared between the GNN trainer
(run_GNN_flywire_blank50.py) and the Known_ODE trainer
(run_KnownODE_flywire_blank50.py): the connectome and stimulus are
identical between a GNN/KODE pair, only graph_model differs at training.

Datasets land at: <output_root>/graphs_data/fly/<base>_blank50_cv{i:02d}/
Resume-safe: _have_data() in cross/pipeline.py skips folds whose
x_list_train + generation_log.txt + voltage.zarr already match.

Side effect: each freshly generated dataset gets a panel-A-style
traces.png (stacked voltage of 12 representative neuron types) at
graphs_data/fly/<base>_blank50_cv{i:02d}/traces.png — cheap visual sanity
check, no SVD, no mp4, no per-frame Fig/ saves.

Sequential and in-process — no cluster fan-out. Run in a tmux session.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import generate_all_yt_data


BLANK50_SIM_OVERRIDES = {
    'blank_prefix_fraction': 0.50,
    'skip_short_videos': True,
}


# Comment out a row to skip it. The 4 connectomes (each one shared between
# the GNN/KODE training scripts) — datasets land at
# <output_root>/graphs_data/fly/<base>_blank50_cv{0..4}/.
CONDITION_FILTER = [
    'e8_flywireRF_noise_005',                       # 327k edges
    'e8_flywireRF_proximal_nulls_noise_005',        # 2.4M edges
    'e8_flywireRF_typed_nulls_noise_005',           # same-type nulls control
    'full_eye_flywireRF_noise_005',                 # 1.3M edges
    'full_eye_flywireRF_proximal_nulls_noise_005',  # 9.6M edges
    'full_eye_flywireRF_typed_nulls_noise_005',     # same-type nulls control
]


generate_all_yt_data(
    suffix='blank50_gen',
    dataset_tag='blank50',
    sim_overrides=BLANK50_SIM_OVERRIDES,
    condition_filter=CONDITION_FILTER,
    data_augmentation_loop=500,
)
