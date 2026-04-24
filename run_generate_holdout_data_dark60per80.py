"""Pre-generate hold-out CV datasets with windowed blank-frame injection
(l=60, m=80) — i.e. inject 60 consecutive zero-stimulus ("dark") frames after
every 80 real video frames, counted across video boundaries.

Mirror of run_generate_holdout_data_blank50.py but uses the new windowed
injection feature (blank_window_size_frames / blank_insertion_every_n_frames
on SimulationConfig, added in PR #32) instead of the legacy
blank_prefix_fraction. Selected from the bsweep V_rest analysis on DAVIS:
(l=60, m=80) sat near the top of the V_rest plateau (~0.83) at modest
compute and storage cost.

CONDITION_FILTER must mirror the one in the matching trainer
(run_GNN_unified_dark60per80.py once it exists). Bases absent from
CONDITIONS in src/connectome_gnn/cross/yaml_io.py are silently skipped.

Datasets land under a distinct folder tag so they don't collide with
the existing yt_cv / yt_all_cv / blank50 datasets:

    <output_root>/graphs_data/fly/<base>_dark60per80_cv<i:02d>/

Downstream training runner: run_GNN_unified_dark60per80.py.

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import generate_all_yt_data


# Windowed blank injection — see SimulationConfig validator for compatibility.
# n_frames continues to count real video frames; injected blanks add on top.
# skip_short_videos=False  → keep all DAVIS clips (don't drop the short ones).
# truncate_max_frames=None → don't crop long clips. Together these let every
# clip contribute its full native length, so the m=80 / l=60 cadence sees the
# actual clip-length distribution rather than the capped one.
DARK60PER80_SIM_OVERRIDES = {
    'blank_window_size_frames': 60,
    'blank_insertion_every_n_frames': 80,
    'skip_short_videos': False,
    'truncate_max_frames': None,
}

# Must match CONDITION_FILTER in run_GNN_unified_dark60per80.py.
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


generate_all_yt_data(
    suffix='dark60per80_gen',
    dataset_tag='dark60per80',
    sim_overrides=DARK60PER80_SIM_OVERRIDES,
    condition_filter=CONDITION_FILTER,
)
