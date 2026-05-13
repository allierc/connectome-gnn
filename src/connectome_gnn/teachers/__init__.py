"""Teacher-network training pipelines.

This subpackage contains *training* code for biomodels that need a learned
recurrent weight matrix before they can be used as ground-truth data
generators in the inverse-problem pipeline (graph_data_generator.py).

Each teacher trainer is self-contained: it loads a connectome (or another
external constraint), trains a network on a task, and writes a checkpoint
that the matching ODE-params class then consumes via `from_pretrained`.
"""

from connectome_gnn.teachers.hulse_cx_teacher import (
    HulseCxRNN,
    PathIntegrationBatch,
    generate_path_integration_batch,
    train_hulse_cx_teacher,
)

__all__ = [
    "HulseCxRNN",
    "PathIntegrationBatch",
    "generate_path_integration_batch",
    "train_hulse_cx_teacher",
]
