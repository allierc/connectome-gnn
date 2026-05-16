"""Connectome-GNN — Task-trainer LLM Exploration Loop.

Sibling of GNN_LLM.py for the path-integration trainer (`data_train_task_gnn`).
Differences from GNN_LLM:
  - No data regeneration loop (the path-integration zarrs are written once;
    only training hyperparameters change).
  - No test+plot phase (the trainer's matrix+kinograph snapshots and
    `tmp_training/metrics.log` are the authoritative output).
  - Per-epoch trajectory + collapse detection surfaced into the per-slot
    analysis log so Claude can spot curriculum-time instabilities.

Pipeline structure:
  setup → batch_0 → loop { code_session? → load → train → artifacts → analysis → finalize }

Command-line examples:

  # Local mode (sequential, single GPU). Good for the first few sanity batches.
  python GNN_LLM_task.py -o train_task drosophila_cx_pi

  # Cluster mode on l4, 8 slots in parallel, 148 iterations total.
  python GNN_LLM_task.py -o train_task drosophila_cx_pi iterations=148 \\
      --cluster --node l4

  # Resume from where it stopped.
  python GNN_LLM_task.py -o train_task drosophila_cx_pi iterations=148 \\
      --cluster --node l4 --resume

Options:
  -o/--option: task option name + base config (e.g. train_task drosophila_cx_pi)
  --cluster:   submit training to LSF cluster (default: local sequential)
  --node:      cluster node override (l4 / a100 / h100)
  --fresh:     start from iteration 1 (default; ignores prior state)
  --resume:    auto-resume from last completed batch
"""

import matplotlib
matplotlib.use('Agg')
import argparse
import os
import warnings
import sys
import os
import shutil

# Ensure src/ is on the path so connectome_gnn is always importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.LLM import (
    finalize_batch,
    init_shared_files,
    init_slot_configs,
    load_configs_and_seeds,
    make_batch_info,
    run_batch_0,
    run_claude_analysis,
    run_code_session,
    save_artifacts,
    setup_exploration,
)
from connectome_gnn.LLM.task_pipeline import (
    run_task_cluster_training,
    run_task_local_pipeline,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Connectome-GNN — Task-trainer LLM Loop")
    parser.add_argument("-o", "--option", nargs="+",
                        help="option that takes multiple values: <task> <config> [k=v ...]")
    parser.add_argument("--fresh", action="store_true", default=True,
                        help="start from iteration 1 (ignore auto-resume)")
    parser.add_argument("--resume", action="store_true",
                        help="auto-resume from last completed batch")
    parser.add_argument("--cluster", action="store_true",
                        help="submit training to LSF cluster (default: run locally)")
    parser.add_argument("--node", type=str, default=None,
                        help="cluster node name override (e.g. l4, a100, h100). "
                             "Overrides claude.node_name in YAML.")
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Setup (shared with GNN_LLM.py) ---
    state = setup_exploration(args, root_dir)
    init_slot_configs(state, is_resume=args.resume)
    init_shared_files(state, is_resume=args.resume)

    # --- Batch 0: initialize config variations (fresh start only) ---
    if state.start_iteration == 1 and not args.resume:
        run_batch_0(state)

    # --- Main batch loop ---
    for batch_start in range(state.start_iteration, state.n_iterations + 1, state.n_parallel):
        batch = make_batch_info(state, batch_start)

        # Optional interactive code modification at block boundaries.
        if state.interaction_code and batch.is_block_start and batch.block_number > 1:
            run_code_session(state, batch)

        print(f"\n\033[94mBATCH: iterations {batch.batch_first}-{batch.batch_last} / "
              f"{state.n_iterations}  (block {batch.block_number})\033[0m")

        # Per-slot config + seed forcing.
        load_configs_and_seeds(state, batch)

        # Training (no test+plot, no data regen — task-specific runner).
        if state.cluster_enabled:
            run_task_cluster_training(state, batch)
        else:
            run_task_local_pipeline(state, batch)

        # Save exploration artifacts (configs, snapshots).
        save_artifacts(state, batch)

        # Claude analysis + next mutations (reads the per-slot analysis log
        # populated by run_task_*).
        run_claude_analysis(state, batch)

        # Finalize: tree viz, protocol/memory snapshots.
        finalize_batch(state, batch)

# --- Examples ---
#
# Local single-GPU sanity check (1 batch = 1 slot since n_parallel forced down):
#   python GNN_LLM_task.py -o train_task drosophila_cx_pi iterations=4
#
# Cluster, l4, 8 slots/batch, 148 iterations (≈19 batches × ~30 min ≈ 9 h):
#   python GNN_LLM_task.py -o train_task drosophila_cx_pi iterations=148 --cluster --node l4
#
# Resume after Ctrl-C / job failure:
#   python GNN_LLM_task.py -o train_task drosophila_cx_pi iterations=148 --cluster --node l4 --resume
