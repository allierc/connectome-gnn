"""Known_ODE baseline cross-check on 50%-blank-prefix DAVIS data.

Mirror of run_GNN_unified_blank50.py (new layout: CONDITION_NODES dict as
single source of truth, node_name_per_condition wired through) but swaps
the unified-GNN winner HPs for the Known_ODE winner HPs
(flyvis_noise_free_known_ode_reg_winner, same as run_KnownODE_conditions.py).

Restricted to 8 conditions — drops stride_5 and both hidden_NGP variants
(the hidden-NGP ones have no Known_ODE analogue since Known_ODE has no
NGP-T INR, and stride_5 is out of scope for this baseline table).

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

  HP yaml (graph_model + training blocks, applied to every condition):
    flyvis_noise_free_known_ode_reg_winner.yaml

  emitted CV yamls (40 total, written to <output_root>/config/fly/):
    {base}_blank50_known_ode_cv{00..04}.yaml

    datasets: <output_root>/graphs_data/fly/<base>_blank50_cv{00..04}/  (shared with GNN blank50)
    tex out : <output_root>/log/cv_blank50_known_ode_rows.tex

Wall-clock per Known_ODE run: typically <<1h on l4 (far smaller
parameter count than the GNN), but the 48h ceiling matches the GNN
pipeline for scheduling consistency.
Total training units: 8 conditions × 5 folds = 40 Known_ODE runs.

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

#  condition name                        ->  LSF queue (gpu_<node>)
# Comment out a row to drop it from the run. The dict is the single source of
# truth — CONDITION_FILTER is derived from its keys, NODE_PER_CONDITION from
# the full mapping.
CONDITION_NODES = {
    'flyvis_noise_free':                    'l4',
    'flyvis_noise_005':                     'l4',
    'flyvis_noise_05':                      'l4',
    'flyvis_noise_005_010':                 'l4',
    'flyvis_noise_005_020':                 'l4',
    'flyvis_noise_005_null_edges_pc_400':   'l4',
    'flyvis_noise_005_removed_pc_20':       'l4',
    'flyvis_noise_005_removed_pc_50':       'l4',
}

CONDITION_FILTER     = list(CONDITION_NODES.keys())
NODE_PER_CONDITION   = CONDITION_NODES


run_all_conditions(
    hp_source='uniform',
    suffix='blank50_known_ode',
    hp_yaml='flyvis_noise_free_known_ode_reg_winner',
    node_name='l4',
    hard_runtime_limit_min=2880,
    sim_overrides=BLANK50_SIM_OVERRIDES,
    dataset_tag='blank50',
    condition_filter=CONDITION_FILTER,
    data_augmentation_loop=500,
    conditions_per_wave=3,
    emit_tex=False,
)
