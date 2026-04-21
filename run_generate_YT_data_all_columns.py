"""Pre-generate YouTube-VOS CV datasets for the FULL-FLY (all_columns=True) variant.

Mirror of run_generate_YT_data.py but sets simulation.all_columns=True in
every emitted YAML and uses dataset_tag='yt_all' so the datasets live in
separate folders:

    <output_root>/graphs_data/fly/<base>_yt_all_cv<i:02d>/   (NEW, 40 total)

All 8 conditions × 5 folds = 40 datasets. The network has:
    n_neurons        = 45669  (vs 13741 with extent=8)
    n_input_neurons  = 5768   (vs 1736)
    n_edges          = 1513231 (vs 434112)

The trained TeX row file for the downstream training runner will be:
    <output_root>/log/cv_yt_all_unified_rows.tex

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import generate_all_yt_data


ALL_COLUMNS_SIM_OVERRIDES = {
    'all_columns': True,
    'n_neurons': 45669,
    'n_input_neurons': 5768,
    'n_edges': 1513231,
}


generate_all_yt_data(
    suffix='yt_all_gen',
    dataset_tag='yt_all',
    sim_overrides=ALL_COLUMNS_SIM_OVERRIDES,
)
