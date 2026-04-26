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
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_hybrid_flywireRF_zeroedge_cross_sl_noise_005 iterations=80 --cluster --resume
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_hybrid_flywireRF_zeroedge_cross_sl_known_ode_noise_005 iterations=72 --cluster --resume

# --- SPEND Noise2Noise explorations (l4, blank50 stimulus, sigma=0.05, gamma=0.10) ---
# Cite: https://github.com/buchenglab/SPEND  (Ding et al. 2025, Newton 1, 100195)
#
# Step 1: generate the blank50 + measurement-noise dataset ONCE before launching the four agentic loops.
# bsub -n 8 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python GNN_Main.py -o generate flyvis_noise_005_010_blank50"
#
# Step 2: launch the four SPEND agentic explorations (each reads its own instruction_*.md).
# All use generate_data: false (claude block in each YAML); they reuse the dataset from step 1.
# Block plans (each block = 8 iter = 4 slots * 2 batches):
#   replay/time: 6 planned blocks (B1 SPEND coeff, B2 LR, B3 mechanism-knob,
#                B4 smoother LR, B5 merged regul, B6 CV) + stretch B7+
#                -> 48 iter (min: planned blocks only) ... 72 iter (with stretch).
#   typed:       5 planned blocks (no smoother branch: B1 coeff x dist, B2 LR,
#                B3 dist refine, B4 merged regul, B5 CV) + stretch B6+
#                -> 40 iter (min) ... 72 iter (with stretch).
#   combined:    6 planned blocks (B1 solo re-validate, B2 LR, B3 replay+typed,
#                B4 add time-permute, B5 merged regul, B6 CV) + stretch B7+
#                -> 48 iter (min) ... 88 iter (with stretch).
# At ~60 min/iter on a100 (4-slot parallel batch): 72 iter ~18 h, 88 iter ~22 h.
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_010_spend_replay   iterations=72 --cluster --resume
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_010_spend_time     iterations=72 --cluster --resume
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_010_spend_typed    iterations=72 --cluster --resume
# python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005_010_spend_combined iterations=88 --cluster --resume
