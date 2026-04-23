"""Known_ODE baseline cross-check on 50%-blank-prefix DAVIS data.

Mirror of run_GNN_unified_blank50.py but swaps the unified-GNN winner HPs
for the Known_ODE winner HPs (flyvis_noise_free_known_ode_reg_winner,
same as run_KnownODE_conditions.py). All other settings — 11-condition
filter, blank_prefix_fraction=0.50 override, dataset_tag='blank50',
a100 @ 48h, data_augmentation_loop=500, 3-condition waves — match the
GNN blank50 pipeline so the two tables are directly comparable.

Config files used (relative to repo config/fly/):

  simulation-block sources (one per condition):
    flyvis_noise_free.yaml
    flyvis_noise_005.yaml
    flyvis_noise_05.yaml
    flyvis_noise_005_010.yaml
    flyvis_noise_005_020.yaml
    flyvis_noise_005_null_edges_pc_400.yaml
    flyvis_noise_005_removed_pc_20.yaml
    flyvis_noise_005_removed_pc_50.yaml
    flyvis_noise_005_stride_5.yaml
    flyvis_noise_005_hidden_010_ngp.yaml
    flyvis_noise_005_hidden_020_ngp.yaml

  HP yaml (graph_model + training blocks, applied to every condition):
    flyvis_noise_free_known_ode_reg_winner.yaml

  emitted CV yamls (55 total, written to <output_root>/config/fly/):
    {base}_blank50_known_ode_cv{00..04}.yaml

    datasets: <output_root>/graphs_data/fly/<base>_blank50_cv{00..04}/  (shared with GNN blank50)
    tex out : <output_root>/log/cv_blank50_known_ode_rows.tex

Wall-clock per Known_ODE run: typically <<1h on a100 (far smaller
parameter count than the GNN), but the 48h ceiling matches the GNN
pipeline for scheduling consistency.
Total training units: 11 conditions × 5 folds = 55 Known_ODE runs.

Hidden-neuron conditions (hidden_010_ngp, hidden_020_ngp) train on
zero-silenced hidden voltages — Known_ODE has no NGP-T INR. That's
intentional for the baseline comparison, not a bug.

This script does NOT generate data — it fails fast if the datasets are
missing. Run run_generate_holdout_data_blank50.py first; its datasets
are shared across both the GNN and the Known_ODE blank50 pipelines.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import run_all_conditions


BLANK50_SIM_OVERRIDES = {
    'blank_prefix_fraction': 0.50,
    # Match the standalone flyvis_noise_005_blank50 run (which uses the config
    # default True); emit_one() otherwise forces False for the YT-VOS pipelines.
    'skip_short_videos': True,
}

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


run_all_conditions(
    hp_source='uniform',
    suffix='blank50_known_ode',
    hp_yaml='flyvis_noise_free_known_ode_reg_winner',
    hard_runtime_limit_min=2880,
    sim_overrides=BLANK50_SIM_OVERRIDES,
    dataset_tag='blank50',
    condition_filter=CONDITION_FILTER,
    data_augmentation_loop=500,
    conditions_per_wave=3,
)
