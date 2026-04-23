"""Pre-generate YouTube-VOS CV datasets with 50% blank-prefix injection.

Mirror of run_generate_YT_data_all_columns.py but:
  - standard 217-column network (no all_columns),
  - simulation.blank_prefix_fraction = 0.50 injected via sim_overrides
    (zero-stimulus for the first 50% of each video sequence — supplies
    the V_rest training signal we saw missing in the YT CV table).

CONDITION_FILTER must mirror the one in run_GNN_unified_blank50.py so
the generator and trainer operate on the same set of datasets. Bases
absent from CONDITIONS in src/connectome_gnn/cross/yaml_io.py are
silently skipped by both scripts.

Datasets land under a distinct folder tag so they don't collide with
the existing yt_cv / yt_all_cv datasets:

    <output_root>/graphs_data/fly/<base>_yt_blank50_cv<i:02d>/

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

# Must match CONDITION_FILTER in run_GNN_unified_blank50.py.
CONDITION_FILTER = [
    'flyvis_noise_free',
    'flyvis_noise_005',
    'flyvis_noise_05',
    'flyvis_noise_005_010',
    'flyvis_noise_005_020',
    'flyvis_noise_005_null_edges_pc_400',
    'flyvis_noise_005_removed_pc_20',
    'flyvis_noise_005_removed_pc_50',
    'flyvis_noise_005_stride_5',
    'flyvis_noise_005_hidden_010_ngp',
    'flyvis_noise_005_hidden_020_ngp',
]


generate_all_yt_data(
    suffix='yt_blank50_gen',
    dataset_tag='yt_blank50',
    sim_overrides=BLANK50_SIM_OVERRIDES,
    condition_filter=CONDITION_FILTER,
)
