"""Re-emit the 15 blank50_unified CV yamls for stride_5, hidden_010_ngp,
and hidden_020_ngp using the per-condition winner HPs now wired into
src/connectome_gnn/cross/yaml_io.py::CONDITIONS.

Run this while run_GNN_unified_blank50.py is still training null_edges
so the runner picks up the per-condition HPs when it advances to these
three pending conditions.

See /home/node/.claude/plans/glimmering-pondering-lerdorf.md for the plan.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross.yaml_io import emit_yt_yamls


BLANK50_SIM_OVERRIDES = {
    'blank_prefix_fraction': 0.50,
    'skip_short_videos': True,
}

# Same override table as run_GNN_unified_blank50.py — null_edges (5x edges)
# and the NGP-hidden conditions run at DAL=100; everything else at DAL=500.
DAL_OVERRIDES = {
    'flyvis_noise_005_null_edges_pc_400': 100,
    'flyvis_noise_005_hidden_010_ngp':    100,
    'flyvis_noise_005_hidden_020_ngp':    100,
}

# Re-emit only the pending conditions whose yamls need updating:
#   stride_5: currently DAL=100 on disk, bump to 500 (fast per-step, safe to use full budget)
#   null_edges: currently DAL=500 on disk, drop to 100 (5x edges too slow at 500)
#   hidden_*_ngp: re-emit anyway so HP tweaks to anchors_winner propagate
CONDITION_FILTER = [
    'flyvis_noise_005_stride_5',
    'flyvis_noise_005_null_edges_pc_400',
    'flyvis_noise_005_hidden_010_ngp',
    'flyvis_noise_005_hidden_020_ngp',
]


emit_yt_yamls(
    hp_source='per_condition',
    suffix='blank50_unified',
    hp_yaml_basename=None,
    n_folds=5,
    output_root='/groups/saalfeld/home/allierc/GraphData',
    sim_overrides=BLANK50_SIM_OVERRIDES,
    dataset_tag='blank50',
    condition_filter=CONDITION_FILTER,
    data_augmentation_loop=500,
    data_augmentation_loop_overrides=DAL_OVERRIDES,
)
