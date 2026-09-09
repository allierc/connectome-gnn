"""Connectome-GNN — Parallel LLM Exploration Loop, persistent-session variant.

Same pipeline as GNN_LLM.py — setup, batch 0, then {load, train, artifacts,
analysis, finalize} per batch — with one difference: every Claude call of an
exploration happens inside ONE conversation instead of a fresh one per batch.

Why. The stock loop spawns `claude -p <prompt>` per batch, so the agent starts
each batch with an empty context and re-reads the instruction file (~360 lines),
the working memory (past 20k tokens by the last block) and the four per-slot
logs — a 40-60k-token rebuild of context it had ten minutes earlier, roughly
thirty times per exploration, out of a 20-minute wall-clock budget it also needs
for the analysis itself. Here the instruction file is read at batch 0 and stays
in the conversation; the agent remembers the memory entries because it wrote
them; and the per-batch prompt carries only the four slot paths, this batch's
seeds and the block header.

The conversation is re-read in full at every block start, after a compaction and
after a failed --resume — see src/connectome_gnn/LLM_plus/session.py for why
each of those three invalidates it.

The conversation id lives in <exploration_dir>/claude_session.json and the
per-call record in claude_session.log beside it, so --resume of this script
re-attaches to the same conversation across process restarts. --fresh always
starts a new one: a fresh exploration erases the analysis and memory files, and
a conversation full of the erased run's conclusions is worse than none.

Usage — identical to GNN_LLM.py:

  python GNN_LLM+.py -o generate_train_test_plot_Claude \
      flyvis_conductance_noise_005_conductance_knownode_cv00 \
      iterations=96 --cluster --resume

  python GNN_LLM+.py -o generate_train_test_plot_Claude \
      flyvis_conductance_noise_005_conductance_gnn_cv00 \
      iterations=120 --cluster --resume

Options:
  -o/--option: task option names (e.g., generate_train_test_plot_Claude)
  --cluster: submit training to LSF cluster (default: local)
  --fresh: start from iteration 1, with a new conversation
  --resume: auto-resume from the last completed batch, in the same conversation
"""

import matplotlib
matplotlib.use('Agg')  # set non-interactive backend before other imports
import argparse
import os
import warnings
import sys

# Ensure src/ is on the path so connectome_gnn is always importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import connectome_gnn.LLM.pipeline as pipeline
from connectome_gnn.LLM import (
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
from connectome_gnn.LLM_plus import ClaudeSession, install_session_hooks

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Connectome-GNN — Parallel LLM Loop (one persistent Claude session)")
    parser.add_argument("-o", "--option", nargs="+", help="option that takes multiple values")
    parser.add_argument("--fresh", action="store_true", default=True,
                        help="start from iteration 1 in a new conversation (ignore auto-resume)")
    parser.add_argument("--resume", action="store_true",
                        help="auto-resume from last completed batch, in the same conversation")
    parser.add_argument("--cluster", action="store_true",
                        help="submit training to LSF cluster (default: run locally)")
    parser.add_argument("--node", type=str, default=None,
                        help="cluster node name override (e.g. a100, h100, l4). Overrides claude.node_name in YAML.")
    parser.add_argument("--new-session", action="store_true",
                        help="resume the exploration but start a new conversation "
                             "(use after editing the instruction file mid-run)")
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Setup ---
    state = setup_exploration(args, root_dir)
    init_slot_configs(state, is_resume=args.resume)
    init_shared_files(state, is_resume=args.resume)

    # --- The pinned conversation ---
    # Its working directory must match every other call of this exploration:
    # the CLI files conversations per project directory, so a --resume issued
    # from elsewhere would not find it.
    session = ClaudeSession(
        exploration_dir=state.exploration_dir,
        root_dir=state.root_dir,
        base_config_name=state.base_config_name,
    ).load_or_create(fresh=(not args.resume) or args.new_session)
    install_session_hooks(pipeline, session, state)

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

        # Claude analysis + next mutations, inside the pinned conversation
        run_claude_analysis(state, batch)

        # Finalize: tree viz, protocol/memory snapshots
        finalize_batch(state, batch)

    print(f"\n\033[94mExploration finished. Conversation {session.session_id}, "
          f"{session.n_calls} calls — transcript via `claude --resume {session.session_id}` "
          f"from {state.root_dir}\033[0m")

# --- conductance explorations (conda activate neural-graph-linux) ---
#
# python GNN_LLM+.py -o generate_train_test_plot_Claude flyvis_conductance_noise_005_conductance_knownode_cv00 iterations=96 --cluster --resume
# python GNN_LLM+.py -o generate_train_test_plot_Claude flyvis_conductance_noise_005_conductance_gnn_cv00 iterations=120 --cluster --resume
# python "GNN_LLM+.py" -o train_test_plot_Claude flyvis_conductance_noise_005_conductance_knownode_cv00 iterations=144 --cluster --node l4
# export SSH_AUTH_SOCK=$(ls -t /tmp/vscode-ssh-auth-*.sock | head -1)
# ssh -o BatchMode=yes allierc@login1 echo ok