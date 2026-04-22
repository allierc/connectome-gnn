"""Pre-generate YouTube-VOS CV datasets with 50% blank-prefix injection.

Mirror of run_generate_YT_data_all_columns.py but:
  - standard 217-column network (no all_columns),
  - simulation.blank_prefix_fraction = 0.50 injected via sim_overrides
    (zero-stimulus for the first 50% of each video sequence — supplies
    the V_rest training signal we saw missing in the YT CV table),
  - restricted to the first 3 conditions: noise_free / noise_005 / noise_05.

Datasets land under a distinct folder tag so they don't collide with
the existing yt_cv / yt_all_cv datasets:

    <output_root>/graphs_data/fly/<base>_yt_blank50_cv<i:02d>/   (NEW, 15 total)

Downstream training runner: run_GNN_unified_blank50.py.

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import generate_all_yt_data


BLANK50_SIM_OVERRIDES = {
    'blank_prefix_fraction': 0.50,
}

# First 3 rows of the YT CV table: noise-free, low-intrinsic, high-intrinsic.
CONDITION_FILTER = [
    'flyvis_noise_free',
    'flyvis_noise_005',
    'flyvis_noise_05',
]


generate_all_yt_data(
    suffix='yt_blank50_gen',
    dataset_tag='yt_blank50',
    sim_overrides=BLANK50_SIM_OVERRIDES,
    condition_filter=CONDITION_FILTER,
)
