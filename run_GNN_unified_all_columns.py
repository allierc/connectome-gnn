"""GNN YT-only cross-check with UNIFIED-winner HPs, FULL-FLY variant (all_columns=True).

Mirror of run_GNN_unified.py but trains on the 45669-neuron full-fly
datasets produced by run_generate_YT_data_all_columns.py.

    datasets: <output_root>/graphs_data/fly/<base>_yt_all_cv<i:02d>/
    configs : <output_root>/config/fly/<base>_yt_all_unified_cv<i:02d>.yaml
    tex out : <output_root>/log/cv_yt_all_unified_rows.tex

Wall-clock per GNN: ~1–5 hours on a100 (full fly is ~3.3× more neurons
and ~3.5× more edges than the 217-column baseline). Wall-clock limit
bumped to 480 min per job.

Runs 8 conditions × 5 folds = 40 trainings. Stride_5 and hidden_010_ngp
are structurally incompatible with the uniform HPs and will degrade — treat
their rows as lower bounds.

This script does NOT generate data — it fails fast if the datasets are
missing. Run run_generate_YT_data_all_columns.py first (or use the bash
wrapper run_GNN_all_columns_pipeline.sh).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import run_all_conditions


ALL_COLUMNS_SIM_OVERRIDES = {
    'all_columns': True,
    'n_neurons': 45669,
    'n_input_neurons': 5768,
    'n_edges': 1513231,
}


run_all_conditions(
    hp_source='uniform',
    suffix='yt_all_unified',
    hp_yaml='flyvis_unified_winner',
    hard_runtime_limit_min=480,
    sim_overrides=ALL_COLUMNS_SIM_OVERRIDES,
    dataset_tag='yt_all',
)
