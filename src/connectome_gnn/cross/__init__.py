"""Hold-out-only CV pipeline: train + test on hold-out dataset folds."""

from connectome_gnn.cross.runner import run_all_conditions, generate_all_yt_data
from connectome_gnn.cross.pipeline import (
    CONDITION_BASES, run_condition, generate_yt_data_for_condition,
)
from connectome_gnn.cross.yaml_io import CONDITIONS, emit_yt_yamls, emit_one
from connectome_gnn.cross.tex import emit_tex_file, emit_row

__all__ = [
    'run_all_conditions', 'generate_all_yt_data',
    'CONDITION_BASES', 'run_condition', 'generate_yt_data_for_condition',
    'CONDITIONS', 'emit_yt_yamls', 'emit_one',
    'emit_tex_file', 'emit_row',
]
