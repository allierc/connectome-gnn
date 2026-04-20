"""GNN YT-only cross-check with PER-CONDITION winner hyperparameters.

Runs the full 8-condition × 5-fold YT-train / YT-held-out-test pipeline
and drops the TeX table at <data_root>/log/cv_yt_per_cond_rows.tex.

Shares the {base}_yt_cv{i:02d} datasets with run_GNN_unique.py and
run_KnownODE_conditions.py. This script does NOT generate data — it
fails fast if the datasets are missing. Run run_generate_YT_data.py
first, then launch the three training scripts in parallel.

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior (conditions, n_folds, node type, etc.).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import run_all_conditions

run_all_conditions(hp_source='per_condition', suffix='yt_per_cond')
