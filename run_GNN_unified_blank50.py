"""GNN YT-only cross-check with UNIFIED-winner HPs on 50%-blank-prefix data.

Mirror of run_GNN_unified_all_columns.py but:
  - standard 217-column network (no all_columns),
  - simulation.blank_prefix_fraction = 0.50 (matches the datasets produced
    by run_generate_YT_data_blank50.py),
  - restricted to the first 3 conditions: noise_free / noise_005 / noise_05.

Goal: test whether 50% blank-prefix per video sequence recovers V_rest_R²
from the ~0 ceiling we hit on the standard YT CV table without blanks.

    datasets: <output_root>/graphs_data/fly/<base>_yt_blank50_cv<i:02d>/
    configs : <output_root>/config/fly/<base>_yt_blank50_unified_cv<i:02d>.yaml
    tex out : <output_root>/log/cv_yt_blank50_unified_rows.tex

Wall-clock per GNN: ~1 h on a100 (same compute as the non-blank runs).
Total training units: 3 conditions × 5 folds = 15 GNNs.

This script does NOT generate data — it fails fast if the datasets are
missing. Run run_generate_YT_data_blank50.py first (or use the bash
wrapper run_GNN_blank50_pipeline.sh).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import run_all_conditions


BLANK50_SIM_OVERRIDES = {
    'blank_prefix_fraction': 0.50,
}

CONDITION_FILTER = [
    'flyvis_noise_free',
    'flyvis_noise_005',
    'flyvis_noise_05',
]


run_all_conditions(
    hp_source='uniform',
    suffix='yt_blank50_unified',
    hp_yaml='flyvis_unified_winner',
    hard_runtime_limit_min=240,
    sim_overrides=BLANK50_SIM_OVERRIDES,
    dataset_tag='yt_blank50',
    condition_filter=CONDITION_FILTER,
)
