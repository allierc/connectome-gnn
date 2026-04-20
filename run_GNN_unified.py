"""GNN YT-only cross-check with the UNIFIED-winner HP set across 6 conditions.

HPs come from flyvis_unified_winner.yaml, synthesized from the 6
per-condition LLM explorations (noise_free, noise_005, noise_05,
noise_005_010, null_edges_pc_400, removed_pc_20). Stride_5 and
hidden_010_ngp are excluded from the unification (structurally
incompatible); their rows in the TeX table will degrade noticeably —
treat them as lower bounds, not a fair comparison.

Runs the full 8-condition × 5-fold YT-train / YT-held-out-test pipeline
and drops the TeX table at <data_root>/log/cv_yt_unified_rows.tex.

Shares the {base}_yt_cv{i:02d} datasets with run_GNN_conditions.py,
run_GNN_unique.py, and run_KnownODE_conditions.py. This script does
NOT generate data — it fails fast if the datasets are missing. Run
run_generate_YT_data.py first, then launch the training scripts in
parallel.

No CLI flags — edit constants in src/connectome_gnn/cross/ if you need
to tune behavior.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.cross import run_all_conditions

run_all_conditions(hp_source='uniform', suffix='yt_unified',
                    hp_yaml='flyvis_unified_winner')
