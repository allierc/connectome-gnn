"""Train the Hulse Model A CX teacher (path-integration RNN).

Ref: Hulse, Aneesh, Romani, Jayaraman, Hermundstad (Janelia 2026 draft),
     docs/Hidden_Symmetries.pdf Methods, Eqs. 1-11.

Examples
--------
# Smoke run (CPU, ~5 s) — verify the pipeline end-to-end.
python run_hulse_cx_teacher.py --smoke

# Full Hulse-spec training (200k trials x 10 epochs, ~1 h on A100).
python run_hulse_cx_teacher.py \
    --output papers/hulse_cx/trained/hulse_cx_seed0.pt \
    --device cuda

# Short test (~5 min on a recent GPU) to confirm convergence trajectory.
python run_hulse_cx_teacher.py \
    --n_trials 5000 --n_epochs 3 \
    --output papers/hulse_cx/trained/hulse_cx_short.pt \
    --device cuda

# Override the connectome path (defaults to Beiran's bundled hemibrain CSVs).
python run_hulse_cx_teacher.py \
    --datapath /path/to/exported-traced-adjacencies-v1.2

After training, render the diagnostic figures (compass / EB ring /
kinograph / 3-D anatomy / readout) from any saved checkpoint:

    python -m connectome_gnn.teachers.hulse_cx_diagnostic \
        --checkpoint papers/hulse_cx/trained/hulse_cx_seed0.pt \
        --output-dir papers/hulse_cx/diagnostics_seed0
"""

import os
import sys

# Repo-relative src/ import so the script works from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from connectome_gnn.teachers.hulse_cx_teacher import _main  # noqa: E402


if __name__ == "__main__":
    _main()
