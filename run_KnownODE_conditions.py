"""Known_ODE YT-only cross-check with a UNIFORM HP set across all 8 conditions.

HPs come from flyvis_noise_free_known_ode_reg_winner.yaml (the exploration
winner produced by:
    python GNN_LLM.py -o generate_train_test_plot_Claude \\
        flyvis_noise_free_known_ode_reg iterations=96 --cluster --resume
).

Runs the full 8-condition × 5-fold YT-train / YT-held-out-test pipeline
and drops the TeX table at <data_root>/log/cv_yt_known_ode_rows.tex.

Shares the {base}_yt_cv{i:02d} datasets with run_GNN_conditions.py and
run_GNN_unique.py. This script does NOT generate data — it fails fast
if the datasets are missing. Run run_generate_YT_data.py first, then
launch the three training scripts in parallel.

CLI: --no-test-plot suppresses the post-training rollout/plot wave;
otherwise edit constants in src/connectome_gnn/cross/ to tune behavior.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import run_all_conditions

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--no-test-plot', dest='no_test_plot', action='store_true',
                     help='Submit the training wave only — suppress the test+plot '
                          'wave entirely.')
_args = _parser.parse_args()

run_all_conditions(hp_source='uniform', suffix='yt_known_ode',
                    hp_yaml='flyvis_noise_free_known_ode_reg_winner',
                    hard_runtime_limit_min=480,
                    skip_test_plot=_args.no_test_plot)
