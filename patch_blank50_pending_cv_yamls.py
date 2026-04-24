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

CONDITION_FILTER = [
    'flyvis_noise_005_stride_5',
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
    data_augmentation_loop=100,
)
