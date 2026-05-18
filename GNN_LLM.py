"""Connectome-GNN — Parallel LLM Exploration Loop.

Orchestrates Claude-driven hyperparameter exploration with optional
interactive code modification sessions at block boundaries.

Pipeline structure:
  setup → batch_0 → loop { code_session? → load → train → artifacts → analysis → finalize }

Command-line examples for known_ode LLM exploration:

  # FlyVis noise_005 (baseline known_ode exploration)
  python GNN_LLM.py -o llm_exploration flyvis_noise_005_known_ode --cluster

  # FlyVis noise_free (clean data test)
  python GNN_LLM.py -o llm_exploration flyvis_noise_free_known_ode --cluster

  # FlyVis noise_05 (high-noise robustness)
  python GNN_LLM.py -o llm_exploration flyvis_noise_05_known_ode --cluster

  # FlyVis noise_005 + measurement_noise_010 (combined noise)
  python GNN_LLM.py -o llm_exploration flyvis_noise_005_010_known_ode --cluster

  # FlyVis noise_005 + null_edges_pc_100 (edge learning challenge)
  python GNN_LLM.py -o llm_exploration flyvis_noise_005_null_edges_pc_100_known_ode --cluster

  # FlyVis noise_005 + removed_pc_10 (incomplete connectivity)
  python GNN_LLM.py -o llm_exploration flyvis_noise_005_removed_pc_10_known_ode --cluster

Options:
  -o/--option: task option names (e.g., llm_exploration, train_test_plot)
  --cluster: submit training to LSF cluster (default: local)
  --fresh: start from iteration 1 (ignore auto-resume)
  --resume: auto-resume from last completed batch
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import warnings
import sys
import os
import shutil

# Ensure src/ is on the path so connectome_gnn is always importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.LLM import (
    resume,
    setup_exploration,
    init_slot_configs,
    init_shared_files,
    make_batch_info,
    run_batch_0,
    run_code_session,
    load_configs_and_seeds,
    should_generate_data,
    generate_data_locally,
    run_cluster_training,
    run_cluster_test_plot,
    run_local_pipeline,
    save_artifacts,
    run_claude_analysis,
    finalize_batch,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


def parse_args():
    parser = argparse.ArgumentParser(description="Connectome-GNN — Parallel LLM Loop")
    parser.add_argument("-o", "--option", nargs="+", help="option that takes multiple values")
    parser.add_argument("--fresh", action="store_true", default=True,
                        help="start from iteration 1 (ignore auto-resume)")
    parser.add_argument("--resume", action="store_true",
                        help="auto-resume from last completed batch")
    parser.add_argument("--cluster", action="store_true",
                        help="submit training to LSF cluster (default: run locally)")
    parser.add_argument("--node", type=str, default=None,
                        help="cluster node name override (e.g. a100, h100, l4). Overrides claude.node_name in YAML.")
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Setup ---
    state = setup_exploration(args, root_dir)
    init_slot_configs(state, is_resume=args.resume)
    init_shared_files(state, is_resume=args.resume)

    # --- Batch 0: initialize config variations (fresh start only) ---
    if state.start_iteration == 1 and not args.resume:
        run_batch_0(state)

    # --- Main batch loop ---
    for batch_start in range(state.start_iteration, state.n_iterations + 1, state.n_parallel):
        batch = make_batch_info(state, batch_start)

        # Code session: interactive code modification at block boundaries
        if state.interaction_code and batch.is_block_start and batch.block_number > 1:
            run_code_session(state, batch)

        print(f"\n\033[94mBATCH: iterations {batch.batch_first}-{batch.batch_last} / {state.n_iterations}  (block {batch.block_number})\033[0m")

        # Load configs + force seeds
        load_configs_and_seeds(state, batch)

        # Training (cluster or local)
        if "train" in state.task:
            if state.cluster_enabled:
                if should_generate_data(state, batch):
                    generate_data_locally(state, batch)
                run_cluster_training(state, batch)
                run_cluster_test_plot(state, batch)
            else:
                run_local_pipeline(state, batch)
        else:
            # No training — mark all slots as successful
            for slot in range(batch.n_slots):
                batch.job_results[slot] = True

        # Save exploration artifacts
        save_artifacts(state, batch)

        # Claude analysis + next mutations
        run_claude_analysis(state, batch)

        # Finalize: tree viz, protocol/memory snapshots
        finalize_batch(state, batch)

# --- LLM explorations --- conda activate neural-graph-linux
#
# == done: GNN ==
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_free iterations=120 --cluster --resume
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005 iterations=120 --cluster --resume
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_05 iterations=120 --cluster --resume
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_INR iterations=120 --cluster --resume
#
# CUDA_VISIBLE_DEVICES=1 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_stride_5 iteration=96 --cluster
# bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o train /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_noise_005_stride_5"
# bsub -n 2 -gpu "num=1" -q gpu_h100 -W 6000 -Is "python GNN_Main.py -o train_test_plot /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_noise_005_hidden_005"
# bsub -n 2 -gpu "num=1" -q gpu_h100 -W 6000 -Is "python GNN_Main.py -o train_test_plot /groups/saalfeld/home/allierc/Graph/connectome-gnn/config/fly/flyvis_noise_005_hidden_010 --output_root /groups/saalfeld/home/allierc/GraphData"
# CUDA_VISIBLE_DEVICES=0 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_hidden_010 iteration=128 --cluster

# CUDA_VISIBLE_DEVICES=0 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_stride_5_yt iteration=128 --cluster --resume
# CUDA_VISIBLE_DEVICES=0 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_stride_5 iteration=128 --cluster --resume
# CUDA_VISIBLE_DEVICES=1 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_hidden_010_ngp iteration=128 --cluster --resume
# CUDA_VISIBLE_DEVICES=1 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_hidden_010_siren iteration=128 --cluster --resume

# CUDA_VISIBLE_DEVICES=0 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_010_rc iteration=128 --cluster --resume
# CUDA_VISIBLE_DEVICES=1 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_emb_given iteration=96 --cluster --resume
# CUDA_VISIBLE_DEVICES=1 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_known_ode_reg iteration=128 --cluster --resume
# CUDA_VISIBLE_DEVICES=0 python GNN_LLM_code.py -o generate_train_test_plot_Claude flyvis_noise_005_from_zero --cluster --resume

# CUDA_VISIBLE_DEVICES=1 python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_hidden_010_ngp_anchors --cluster --fresh iteration=264

# --- flywireRF + zero-edge (cross-type) explorations (a100, sigma=0.05) ---
# python GNN_LLM.py -o generate_train_test_plot_Claude e8_flywireRF_proximal_nulls_noise_005 iterations=80 --cluster --resume
# python GNN_LLM.py -o generate_train_test_plot_Claude e8_flywireRF_proximal_nulls_known_ode_noise_005 iterations=72 --cluster --resume

# python GNN_LLM.py -o train_test_plot_CLaude e8_flywireRF_proximal_nulls_noise_005 iterations=150 --cluster --resume
# python GNN_LLM.py -o train_test_plot_CLaude flyvis_noise_005_hidden_010_blank50_consensus_ngp iterations=150 --cluster