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
    # Match the standalone flyvis_noise_005_blank50 run (which uses the config
    # default True); emit_one() otherwise forces False for the YT-VOS pipelines.
    'skip_short_videos': True,
}

#  condition name                        ->  LSF queue (gpu_<node>)
# Comment out a row to drop it from the run. The dict is the single source of
# truth — CONDITION_FILTER is derived from its keys, NODE_PER_CONDITION from
# the full mapping.
CONDITION_NODES = {
    'flyvis_noise_free':                    'a100',
    'flyvis_noise_005':                     'a100',
    'flyvis_noise_05':                      'a100',
    'flyvis_noise_005_010':                 'a100',
    'flyvis_noise_005_020':                 'a100',
    'flyvis_noise_005_null_edges_pc_400':   'a100',
    'flyvis_noise_005_removed_pc_20':       'a100',
    'flyvis_noise_005_removed_pc_50':       'a100',
    'flyvis_noise_005_stride_5':            'a100',
    'flyvis_noise_005_hidden_010_ngp':      'a100',
    'flyvis_noise_005_hidden_020_ngp':      'a100',
}

CONDITION_FILTER     = list(CONDITION_NODES.keys())
NODE_PER_CONDITION   = CONDITION_NODES


# Per-condition DAL overrides. Expensive conditions (5x edges / NGP encoder)
# need a smaller gradient budget to keep wall time under ~6h per fold.
# Others use the default data_augmentation_loop=500.
DAL_OVERRIDES = {
    'flyvis_noise_005_null_edges_pc_400': 100,   # 2.17M edges -> ~5.8h instead of ~29h
    'flyvis_noise_005_hidden_010_ngp':    100,   # NGP encoder + anchors loss; matches winner DAL
    'flyvis_noise_005_hidden_020_ngp':    100,   # same as _010_ngp
}

# Per-condition HP yaml overrides. stride_5 (BPTT with bs=1, coeff_g_phi_diff=9000,
# coeff_g_phi_norm=0.1) and the hidden_*_ngp conditions (NGP-T + anchors training:
# lr_NNR_f, coeff_anchor_voltage, n_anchor, alternate_training) have structurally
# different training recipes that the uniform noise_005-style HP yaml cannot
# represent — route them to their own winner yamls instead. Makes
# `python run_GNN_unified_blank50.py` equivalent to running
# patch_blank50_pending_cv_yamls.py after the uniform emit.
HP_YAML_OVERRIDES = {
    'flyvis_noise_005_stride_5':       'flyvis_noise_005_stride_5_winner',
    'flyvis_noise_005_hidden_010_ngp': 'flyvis_noise_005_hidden_010_ngp_anchors_winner',
    'flyvis_noise_005_hidden_020_ngp': 'flyvis_noise_005_hidden_020_ngp_anchors_winner',
}


run_all_conditions(
    hp_source='uniform',
    suffix='blank50_unified',
    hp_yaml='flyvis_unified_blank50_winner',
    node_name='a100',
    hard_runtime_limit_min=2880,
    sim_overrides=BLANK50_SIM_OVERRIDES,
    dataset_tag='blank50',
    condition_filter=CONDITION_FILTER,
    data_augmentation_loop=500,
    data_augmentation_loop_overrides=DAL_OVERRIDES,
    hp_yaml_overrides=HP_YAML_OVERRIDES,
    conditions_per_wave=3,
    emit_tex=False,
)
