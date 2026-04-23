"""GNN YT-only cross-check with UNIFIED-winner HPs on 50%-blank-prefix data.

Expanded from the 3-condition prototype to all 8 canonical conditions
plus 3 new variants (noise_005_020, removed_pc_50, hidden_020_ngp).

Goal: test whether 50% blank-prefix per video sequence recovers V_rest_R²
from the ~0 ceiling we hit on the standard YT CV table without blanks.

Config files used (relative to repo config/fly/):

  simulation-block sources (one per condition):
    flyvis_noise_free.yaml
    flyvis_noise_005.yaml
    flyvis_noise_05.yaml
    flyvis_noise_005_010.yaml
    flyvis_noise_005_020.yaml               # NEW — template: flyvis_noise_005_010.yaml
    flyvis_noise_005_null_edges_pc_400.yaml
    flyvis_noise_005_removed_pc_20.yaml
    flyvis_noise_005_removed_pc_50.yaml     # NEW — template: flyvis_noise_05_removed_pc_20.yaml
    flyvis_noise_005_stride_5.yaml
    flyvis_noise_005_hidden_010_ngp.yaml
    flyvis_noise_005_hidden_020_ngp.yaml    # NEW — template: flyvis_noise_005_hidden_010_ngp.yaml

  HP yaml (graph_model + training blocks, applied to every condition):
    flyvis_unified_winner.yaml

  emitted CV yamls (55 total, written to <output_root>/config/fly/):
    {base}_blank50_unified_cv{00..04}.yaml

    datasets: <output_root>/graphs_data/fly/<base>_blank50_cv{00..04}/
    tex out : <output_root>/log/cv_blank50_unified_rows.tex

Wall-clock per GNN: ~1 h on a100.
Total training units: 11 conditions × 5 folds = 55 GNNs.

The 3 NEW conditions (noise_005_020, removed_pc_50, hidden_020_ngp) also need
to be registered in CONDITIONS in src/connectome_gnn/cross/yaml_io.py so
the emitter picks them up. Until their base yamls + registry entries
exist, those rows will be skipped with a warning rather than crash.

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
    suffix='blank50_unified',
    hp_yaml='flyvis_unified_blank50_winner',
    hard_runtime_limit_min=2880,
    sim_overrides=BLANK50_SIM_OVERRIDES,
    dataset_tag='blank50',
    condition_filter=CONDITION_FILTER,
    data_augmentation_loop=500,
    conditions_per_wave=3,
)
