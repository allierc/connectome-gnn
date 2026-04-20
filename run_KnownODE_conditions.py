"""Known_ODE YT-only cross-check with a UNIFORM HP set across all 8 conditions.

HPs come from flyvis_noise_free_known_ode_reg_winner.yaml (the exploration
winner produced by:
    python GNN_LLM.py -o generate_train_test_plot_Claude \\
        flyvis_noise_free_known_ode_reg iterations=96 --cluster --resume
).

Runs the full 8-condition × 5-fold YT-train / YT-held-out-test pipeline
and drops the TeX table at <data_root>/log/cv_yt_known_ode_rows.tex.

Shares the {base}_yt_cv{i:02d} datasets with run_GNN_conditions.py and
run_GNN_cross.py — run run_generate_YT_data.py first, then launch all
three scripts in parallel.

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import run_all_conditions

run_all_conditions(hp_source='uniform', suffix='yt_known_ode',
                    hp_yaml='flyvis_noise_free_known_ode_reg_winner')
