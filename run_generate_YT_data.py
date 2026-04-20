"""Pre-generate all YouTube-VOS CV datasets (8 conditions × n_folds).

Run this ONCE before launching the three training runners in parallel:
    run_GNN_conditions.py       (per-condition GNN winner HPs)
    run_GNN_cross.py            (uniform GNN winner HPs)
    run_KnownODE_conditions.py  (uniform Known_ODE winner HPs)

All three share the same {base}_yt_cv{i:02d} datasets written here, so
their ensure_yt_data() calls become noops and they can run concurrently
without racing on data generation.

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import generate_all_yt_data

generate_all_yt_data()
