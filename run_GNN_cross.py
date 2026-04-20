"""GNN YT-only cross-check with a UNIFORM hyperparameter set across 8 conditions.

HPs come from flyvis_noise_005_null_edges_pc_400_winner.yaml. Runs the full
8-condition × 5-fold YT-train / YT-held-out-test pipeline and drops the TeX
table at <data_root>/log/cv_yt_cross_rows.tex.

Shares the {base}_yt_cv{i:02d} datasets with run_GNN_conditions.py and
run_KnownODE_conditions.py — run run_generate_YT_data.py first, then
launch all three scripts in parallel.

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior.
"""

from connectome_gnn.cross import run_all_conditions

run_all_conditions(hp_source='uniform', suffix='yt_cross',
                    hp_yaml='flyvis_noise_005_null_edges_pc_400_winner')
