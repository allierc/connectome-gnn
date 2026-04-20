"""GNN cross-check with a UNIFORM hyperparameter set across all 8 conditions.

The HPs come from flyvis_noise_005_null_edges_pc_400_winner.yaml. Runs the
full 8-condition × 5-fold CV and drops the TeX table at
<data_root>/log/cv_yt_cross_rows.tex.

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior.
"""

from connectome_gnn.cross import run_all_conditions

run_all_conditions(hp_source='uniform', suffix='yt_cross',
                    hp_yaml='flyvis_noise_005_null_edges_pc_400_winner')
