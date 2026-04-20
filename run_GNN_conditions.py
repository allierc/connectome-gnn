"""GNN cross-check with PER-CONDITION winner hyperparameters.

Runs the full 8-condition × 5-fold CV and drops the TeX table at
<data_root>/log/cv_yt_per_cond_rows.tex.

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior (conditions, n_folds, node type, etc.).
"""

from connectome_gnn.cross import run_all_conditions

run_all_conditions(hp_source='per_condition', suffix='yt_per_cond')
