"""Pipeline phase functions for the LLM exploration loop.

Each function corresponds to a phase in the main batch loop of GNN_LLM.py.
"""

import glob as globmod
import os
import re
import shutil
import subprocess
import sys

import yaml

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.graph_trainer import data_test, data_train
from connectome_gnn.models.utils import save_exploration_artifacts_flyvis
from connectome_gnn.utils import (
    add_pre_folder, config_path, get_data_root, load_data_root_from_json,
    log_path, set_data_root, set_device,
)

from .claude_cli import run_claude_cli
from connectome_gnn.LLM_code.claude_cli_ext import run_claude_cli_with_timeout
from .cluster import (
    check_cluster_repo,
    submit_cluster_job,
    submit_cluster_test_plot_job,
    wait_for_cluster_jobs,
    wait_for_cluster_jobs_with_metrics,
)
from .interactive_code import generate_code_brief, interactive_code_session
from .prompts import analysis_prompt, batch_0_prompt
from .resume import detect_last_iteration, get_modified_code_files, is_git_repo
from .state import BatchInfo, ExplorationState

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_exploration(args, root_dir: str, skip_confirm: bool = False) -> ExplorationState:
    """Parse CLI args, load config, create ExplorationState.

    Args:
        args: Parsed argparse namespace.
        root_dir: Project root directory (where GNN_LLM.py lives).
    """
    print()

    if args.option:
        print(f"Options: {args.option}")
    if args.option is not None:
        task = args.option[0]
        config_list = [args.option[1]]
        best_model = None
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        best_model = ''
        task = 'generate_train_test_plot_Claude'
        config_list = ['flyvis_62_0']
        task_params = {'iterations': 128}

    n_iterations = task_params.get('iterations', 128)
    base_config_name = config_list[0] if config_list else 'flyvis_62_0'
    instruction_name = task_params.get('instruction', f'instruction_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')
    exploration_name = task_params.get('exploration_name', f'LLM_{base_config_name}')

    # In cluster mode, direct all output (log/, graphs_data/, slot configs) to shared FS.
    # In local mode, data_root defaults to '.' (repo root when run from there).
    if args.cluster:
        set_data_root(load_data_root_from_json())

    config_root = config_path()                                    # repo root — source config lookup only
    slot_config_root = os.path.join(get_data_root(), 'config')    # data root — slot configs written here
    llm_dir = f"{root_dir}/LLM"
    exploration_dir = os.path.abspath(log_path('Claude_exploration', exploration_name))

    # Load source config and claude settings
    for cfg in config_list:
        if os.path.isfile(cfg):
            source_config = cfg
            parent = os.path.basename(os.path.dirname(os.path.abspath(cfg)))
            pre = parent + "/" if parent else ""
        else:
            cfg_file, pre = add_pre_folder(cfg)
            source_config = f"{config_root}/{cfg_file}.yaml"

    with open(source_config, 'r') as f:
        source_data = yaml.safe_load(f)
    claude_cfg = source_data.get('claude', {})

    generate_data = claude_cfg.get('generate_data', "generate" in task)

    # Simulation parameter constraint
    if generate_data:
        sim_constraint = (
            "IMPORTANT: Data is RE-GENERATED each iteration. Do NOT change simulation "
            "dimensions (n_neurons, n_frames, n_edges, delta_t, noise levels). "
            "You MAY set simulation.derivative_smoothing_window (int, default 1) to apply "
            "temporal smoothing to noisy derivative targets."
        )
    else:
        sim_constraint = (
            "IMPORTANT: Data is PRE-GENERATED in graphs_data/ — do NOT change simulation parameters."
        )

    state = ExplorationState(
        root_dir=root_dir,
        slot_config_root=slot_config_root,
        llm_dir=llm_dir,
        exploration_dir=exploration_dir,
        source_config=source_config,
        base_config_name=base_config_name,
        pre_folder=pre,
        n_epochs=claude_cfg.get('n_epochs', 1),
        data_augmentation_loop=claude_cfg.get('data_augmentation_loop', 25),
        n_iter_block=claude_cfg.get('n_iter_block', 16),
        node_name=claude_cfg.get('node_name', 'h100'),
        conda_env=claude_cfg.get('conda_env', 'connectome-gnn'),
        n_cpus=claude_cfg.get('n_cpus', 2),
        n_parallel=claude_cfg.get('n_parallel', 4),
        generate_data=generate_data,
        training_time_target_min=claude_cfg.get('training_time_target_min', 60),
        hard_runtime_limit_min=claude_cfg.get('hard_runtime_limit_min', 6000),
        interaction_code=claude_cfg.get('interaction_code', False),
        case_study=claude_cfg.get('case_study', ''),
        case_study_brief=claude_cfg.get('case_study_brief', ''),
        claude_call_timeout_min=claude_cfg.get('claude_call_timeout_min', 20),
        cluster_enabled=args.cluster,
        n_iterations=n_iterations,
        task=task,
        sim_constraint=sim_constraint,
        llm_task_name=llm_task_name,
        instruction_name=instruction_name,
        best_model=best_model,
    )

    # L4 nodes give ~15 GB host RAM per slot vs ~40 GB on A100/H100; flyvis
    # training peaks ~30 GB, so n_cpus<4 hits TERM_MEMLIMIT on L4. Bump to 4.
    if state.node_name == 'l4' and state.n_cpus < 4:
        print(f"\033[93m  auto-bumping n_cpus {state.n_cpus} -> 4 for gpu_l4 (memory headroom)\033[0m")
        state.n_cpus = 4

    # Detect resume point
    if args.resume:
        analysis_path_probe = f"{exploration_dir}/{llm_task_name}_analysis.md"
        config_save_dir_probe = f"{exploration_dir}/config"
        state.start_iteration = detect_last_iteration(
            analysis_path_probe, config_save_dir_probe, state.n_parallel
        )
        assert state.start_iteration >= 1, (
            f"detect_last_iteration returned invalid value: {state.start_iteration} < 1\n"
            f"  Analysis path: {analysis_path_probe}\n"
            f"  Config dir: {config_save_dir_probe}"
        )
        if state.start_iteration > 1:
            print(f"\033[93mAuto-resume: resuming from batch starting at {state.start_iteration}\033[0m")
        else:
            print("\033[93mfresh start (no previous iterations found)\033[0m")
    else:
        state.start_iteration = 1
        _analysis_check = f"{exploration_dir}/{llm_task_name}_analysis.md"
        if os.path.exists(_analysis_check):
            print("\033[91mWARNING: fresh start will erase existing results in:\033[0m")
            print(f"\033[91m  {_analysis_check}\033[0m")
            print(f"\033[91m  {exploration_dir}/{llm_task_name}_memory.md\033[0m")
            if not skip_confirm:
                answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
                if answer != 'y':
                    print("Aborted.")
                    sys.exit(0)
            else:
                print("\033[91m(skipping confirmation, proceeding with fresh start)\033[0m")
        print("\033[93mfresh start\033[0m")

    mode = "cluster" if state.cluster_enabled else "local (sequential)"
    ic_str = f", interaction_code: {state.case_study}" if state.interaction_code else ""
    print(f"\033[94mMode: {mode}, node: gpu_{state.node_name}, n_cpus: {state.n_cpus}, n_parallel: {state.n_parallel}, "
          f"generate_data: {state.generate_data}{ic_str}\033[0m")

    return state


def init_slot_configs(state: ExplorationState, is_resume: bool):
    """Create or preserve per-slot YAML configs."""
    config_file, pre_folder = add_pre_folder(state.llm_task_name + '_00')
    state.config_file = config_file
    # pre_folder should match state.pre_folder already

    for slot in range(state.n_parallel):
        slot_name = f"{state.llm_task_name}_{slot:02d}"
        state.slot_names[slot] = slot_name
        slot_cfg_file, _ = add_pre_folder(slot_name)
        target = f"{state.slot_config_root}/{slot_cfg_file}.yaml"
        state.config_paths[slot] = target
        state.analysis_log_paths[slot] = f"{state.exploration_dir}/{slot_name}_analysis.log"

        if state.start_iteration == 1 and not is_resume:
            if os.path.exists(target):
                # Slot config already exists (pre-seeded) — preserve training/graph_model params
                with open(target, 'r') as f:
                    config_data = yaml.safe_load(f)
                config_data['training']['n_epochs'] = state.n_epochs
                config_data['training']['data_augmentation_loop'] = state.data_augmentation_loop
                if state.generate_data:
                    config_data['dataset'] = f"{state.base_config_name}_{slot:02d}"
                config_data['claude'] = {
                    'n_epochs': state.n_epochs,
                    'data_augmentation_loop': state.data_augmentation_loop,
                    'n_iter_block': state.n_iter_block,
                    'n_parallel': state.n_parallel,
                    'node_name': state.node_name,
                    'generate_data': state.generate_data,
                    'training_time_target_min': state.training_time_target_min,
                }
                with open(target, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                print(f"\033[93m  slot {slot}: preserved pre-seeded {target} (dataset='{config_data['dataset']}')\033[0m")
            else:
                # No pre-seeded config — create from source
                shutil.copy2(state.source_config, target)
                with open(target, 'r') as f:
                    config_data = yaml.safe_load(f)
                if state.generate_data:
                    config_data['dataset'] = f"{state.base_config_name}_{slot:02d}"
                config_data['training']['n_epochs'] = state.n_epochs
                config_data['training']['data_augmentation_loop'] = state.data_augmentation_loop
                config_data['description'] = 'designed by Claude (parallel flyvis)'
                config_data['claude'] = {
                    'n_epochs': state.n_epochs,
                    'data_augmentation_loop': state.data_augmentation_loop,
                    'n_iter_block': state.n_iter_block,
                    'n_parallel': state.n_parallel,
                    'node_name': state.node_name,
                    'generate_data': state.generate_data,
                    'training_time_target_min': state.training_time_target_min,
                }
                with open(target, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                print(f"\033[93m  slot {slot}: created {target} from source (dataset='{config_data['dataset']}')\033[0m")
        else:
            if not os.path.exists(target):
                # Resuming but slot config missing — create from source
                shutil.copy2(state.source_config, target)
                with open(target, 'r') as f:
                    config_data = yaml.safe_load(f)
                if state.generate_data:
                    config_data['dataset'] = f"{state.base_config_name}_{slot:02d}"
                config_data['training']['n_epochs'] = state.n_epochs
                config_data['training']['data_augmentation_loop'] = state.data_augmentation_loop
                config_data['description'] = 'designed by Claude (parallel flyvis)'
                config_data['claude'] = {
                    'n_epochs': state.n_epochs,
                    'data_augmentation_loop': state.data_augmentation_loop,
                    'n_iter_block': state.n_iter_block,
                    'n_parallel': state.n_parallel,
                    'node_name': state.node_name,
                    'generate_data': state.generate_data,
                    'training_time_target_min': state.training_time_target_min,
                }
                with open(target, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                print(f"\033[93m  slot {slot}: created {target} from source (resume+missing)\033[0m")
            else:
                print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")


def init_shared_files(state: ExplorationState, is_resume: bool):
    """Create analysis/memory files on fresh start, or preserve on resume."""
    state.analysis_path = f"{state.exploration_dir}/{state.llm_task_name}_analysis.md"
    state.memory_path = f"{state.exploration_dir}/{state.llm_task_name}_memory.md"
    instruction_name = state.instruction_name or f'instruction_{state.base_config_name}'
    state.instruction_path = f"{state.llm_dir}/{instruction_name}.md"
    state.reasoning_log_path = f"{state.exploration_dir}/{state.llm_task_name}_reasoning.log"
    state.user_input_path = f"{state.exploration_dir}/user_input.md"
    state.log_dir = state.exploration_dir

    os.makedirs(state.exploration_dir, exist_ok=True)

    # Check instruction file exists
    if not os.path.exists(state.instruction_path):
        print(f"\033[91merror: instruction file not found: {state.instruction_path}\033[0m")
        sys.exit(1)

    # Create user input file if missing
    if not os.path.exists(state.user_input_path):
        with open(state.user_input_path, 'w') as f:
            f.write("# User Input\n\n")
            f.write("_Write instructions or advice here. The LLM will read this file at each batch and acknowledge below._\n\n")
            f.write("## Pending Instructions\n\n")
            f.write("_(empty — add instructions here)_\n\n")
            f.write("## Acknowledged\n\n")

    # Initialize shared files on fresh start
    if state.start_iteration == 1 and not is_resume:
        with open(state.analysis_path, 'w') as f:
            f.write(f"# Experiment Log: {state.base_config_name} (parallel)\n\n")
        print(f"\033[93mcleared {state.analysis_path}\033[0m")
        open(state.reasoning_log_path, 'w').close()
        print(f"\033[93mcleared {state.reasoning_log_path}\033[0m")
        with open(state.memory_path, 'w') as f:
            f.write(f"# Working Memory: {state.base_config_name} (parallel)\n\n")
            f.write("## Paper Summary (update at every block boundary)\n\n")
            f.write("- **GNN optimization**: [pending first results]\n")
            f.write("- **LLM-driven exploration**: [pending first results]\n\n")
            f.write("## Knowledge Base (accumulated across all blocks)\n\n")
            if state.generate_data:
                f.write("### Robustness Comparison Table\n\n")
                f.write("| Iter | Config summary | conn_R2 (mean±std) | CV% | min | max | tau_R2 (mean) | V_rest_R2 (mean) | Robust? | Hypothesis tested |\n")
                f.write("| ---- | -------------- | ------------------ | --- | --- | --- | ------------- | ---------------- | ------- | ----------------- |\n\n")
            else:
                f.write("### Parameter Effects Table\n\n")
                f.write("| Block | Focus | Best conn_R2 | Best tau_R2 | Best V_rest_R2 | Best Cluster_Acc | Time_min | Key finding |\n")
                f.write("| ----- | ----- | ------------ | ----------- | -------------- | ---------------- | -------- | ----------- |\n\n")
            f.write("### Established Principles\n\n")
            f.write("### Falsified Hypotheses\n\n")
            f.write("### Open Questions\n\n")
            f.write("---\n\n")
            f.write("## Previous Block Summary\n\n")
            f.write("---\n\n")
            f.write("## Current Block (Block 1)\n\n")
            f.write("### Block Info\n\n")
            f.write("### Hypothesis\n\n")
            f.write("### Iterations This Block\n\n")
            f.write("### Emerging Observations\n\n")
        print(f"\033[93mcleared {state.memory_path}\033[0m")
    else:
        print(f"\033[93mpreserving shared files (resuming from iter {state.start_iteration})\033[0m")

    print(f"\033[93m{state.base_config_name} PARALLEL FLYVIS "
          f"(N={state.n_parallel}, {state.n_iterations} iterations, starting at {state.start_iteration})\033[0m")


# ---------------------------------------------------------------------------
# Batch info
# ---------------------------------------------------------------------------

def make_batch_info(state: ExplorationState, batch_start: int) -> BatchInfo:
    """Compute BatchInfo for a batch starting at batch_start."""
    assert batch_start >= 1, f"batch_start must be >= 1 (got {batch_start})"

    iterations = [batch_start + s for s in range(state.n_parallel)
                  if batch_start + s <= state.n_iterations]

    batch_first = iterations[0]
    batch_last = iterations[-1]
    n_slots = len(iterations)

    block_number = (batch_first - 1) // state.n_iter_block + 1
    iter_in_block_first = (batch_first - 1) % state.n_iter_block + 1
    iter_in_block_last = (batch_last - 1) % state.n_iter_block + 1
    is_block_end = any((it - 1) % state.n_iter_block + 1 == state.n_iter_block for it in iterations)
    is_block_start = (batch_first == 1) or ((batch_first - 1) % state.n_iter_block == 0)

    return BatchInfo(
        iterations=iterations,
        batch_first=batch_first,
        batch_last=batch_last,
        n_slots=n_slots,
        block_number=block_number,
        iter_in_block_first=iter_in_block_first,
        iter_in_block_last=iter_in_block_last,
        is_block_start=is_block_start,
        is_block_end=is_block_end,
    )


# ---------------------------------------------------------------------------
# Batch 0
# ---------------------------------------------------------------------------

def run_batch_0(state: ExplorationState):
    """BATCH 0: Claude start call to initialize N config variations."""
    print(f"\n\033[94mBATCH 0: Claude initializing {state.n_parallel} config variations\033[0m")

    slot_list = "\n".join(
        f"  Slot {s}: {state.config_paths[s]}"
        for s in range(state.n_parallel)
    )
    seed_info = "\n".join(
        f"  Slot {s}: simulation_seed={(state.start_iteration + s) * 1000 + s}, "
        f"training_seed={(state.start_iteration + s) * 1000 + s + 500}"
        for s in range(state.n_parallel)
    )

    prompt = batch_0_prompt(state, slot_list, seed_info)

    timeout_sec = max(60, state.claude_call_timeout_min * 60)
    print(
        f"\033[93mClaude start call (timeout={state.claude_call_timeout_min}min, "
        f"live feedback below)...\033[0m"
    )
    output_text, timed_out = run_claude_cli_with_timeout(
        prompt, state.root_dir,
        allowed_tools=['Read', 'Edit', 'Write'],
        timeout_sec=timeout_sec,
        max_turns=100,
        log_prefix='[batch0] ',
    )

    if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
        print("\n\033[91mOAuth token expired during start call\033[0m")
        print("\033[93m  1. Run: claude /login\033[0m")
        print("\033[93m  2. Then re-run this script\033[0m")
        sys.exit(1)

    if timed_out:
        print(
            f"\n\033[93mWARNING: Claude start call timed out after "
            f"{state.claude_call_timeout_min}min. Slot configs left at "
            f"pre-seed values; proceeding with batch 0 as a robustness test.\033[0m"
        )

    # Validate slot YAMLs are still parseable (Edit tool writes atomically, but
    # confirm explicitly so we never submit broken jobs).
    for slot, path in state.config_paths.items():
        try:
            with open(path) as f:
                yaml.safe_load(f)
        except Exception as e:
            print(
                f"\033[91mFATAL: slot {slot} YAML at {path} no longer parses "
                f"({type(e).__name__}: {e}). Restore from {state.source_config} "
                f"and re-run.\033[0m"
            )
            sys.exit(1)

    if output_text.strip():
        with open(state.reasoning_log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write("=== BATCH 0 (start call) ===\n")
            f.write(f"{'='*60}\n")
            if timed_out:
                f.write(f"[NOTE: timed out after {state.claude_call_timeout_min}min]\n")
            f.write(output_text.strip())
            f.write("\n\n")


# ---------------------------------------------------------------------------
# Code session: Interactive code modification
# ---------------------------------------------------------------------------

def run_code_session(state: ExplorationState, batch: BatchInfo):
    """Interactive code modification session at block start (if enabled).

    Skips block 1, skips if already completed (marker file exists).
    """
    if batch.block_number <= 1:
        return

    # Check for completion marker (new name + old name for backward compat)
    marker = os.path.join(
        state.exploration_dir, f'code_session_block_{batch.block_number:03d}.done'
    )
    old_marker = os.path.join(
        state.exploration_dir, f'phase_a_block_{batch.block_number:03d}.done'
    )

    if os.path.exists(marker) or os.path.exists(old_marker):
        print(f"\033[93mCode session already completed for block {batch.block_number} — skipping\033[0m")
        return

    brief_path = generate_code_brief(
        state.memory_path, batch.block_number, state.case_study,
        state.case_study_brief, state.root_dir, state.exploration_dir
    )
    code_changed = interactive_code_session(
        brief_path, state.memory_path, state.analysis_path, state.root_dir,
        state.case_study, state.cluster_enabled, state.exploration_dir,
        batch.block_number
    )

    # Mark code session as done so it won't re-trigger on resume
    with open(marker, 'w') as f:
        f.write(f"completed at iteration {batch.batch_first}\n")

    if code_changed and state.cluster_enabled:
        print("\n\033[93mCode changes applied. Please:\033[0m")
        print("\033[93m  1. git add + commit + push locally\033[0m")
        print("\033[93m  2. git pull on the cluster\033[0m")
        print("\033[93mThen press Enter to continue.\033[0m")
        input("> ")
        while not check_cluster_repo():
            print("\033[91mCluster repo not in sync — please fix and press Enter.\033[0m")
            input("> ")


# ---------------------------------------------------------------------------
# Phase 1: Load configs + force seeds
# ---------------------------------------------------------------------------

def _check_causality(state: ExplorationState, batch: BatchInfo):
    """Warn if LLM changed more than one training parameter per slot vs slot 0."""
    # Keys to compare (training params the LLM controls)
    COMPARE_KEYS = [
        'lr_W', 'lr', 'lr_embedding', 'n_epochs', 'batch_size',
        'data_augmentation_loop', 'coeff_g_phi_diff', 'coeff_f_theta_weight_L2',
        'coeff_f_theta_diff', 'coeff_f_theta_msg_diff',
        'coeff_W_L1', 'coeff_W_L2', 'coeff_W_sign',
        'coeff_tau_L1', 'coeff_tau_L2', 'coeff_V_rest_L1', 'coeff_V_rest_L2',
        'w_init_mode', 'w_init_scale', 'dale_law',
    ]
    COMPARE_GRAPH_KEYS = [
        'hidden_dim', 'hidden_dim_update', 'embedding_dim',
        'input_size', 'input_size_update', 'g_phi_positive',
    ]
    COMPARE_SIM_KEYS = ['noise_model_level']

    ref_yaml_path = state.config_paths[0]
    with open(ref_yaml_path, 'r') as f:
        ref = yaml.safe_load(f)

    violations = []
    for slot in range(1, batch.n_slots):
        with open(state.config_paths[slot], 'r') as f:
            cfg = yaml.safe_load(f)
        diffs = []
        for k in COMPARE_KEYS:
            v0 = ref.get('training', {}).get(k)
            v1 = cfg.get('training', {}).get(k)
            if v0 != v1:
                diffs.append(f"training.{k}: {v0} -> {v1}")
        for k in COMPARE_GRAPH_KEYS:
            v0 = ref.get('graph_model', {}).get(k)
            v1 = cfg.get('graph_model', {}).get(k)
            if v0 != v1:
                diffs.append(f"graph_model.{k}: {v0} -> {v1}")
        for k in COMPARE_SIM_KEYS:
            v0 = ref.get('simulation', {}).get(k)
            v1 = cfg.get('simulation', {}).get(k)
            if v0 != v1:
                diffs.append(f"simulation.{k}: {v0} -> {v1}")
        if len(diffs) > 1:
            violations.append((slot, diffs))

    if violations:
        print(f"\n\033[91mCAUSALITY WARNING: Slots with >1 parameter changed vs slot 0:\033[0m")
        for slot, diffs in violations:
            print(f"\033[91m  Slot {slot}: {len(diffs)} changes — {', '.join(diffs)}\033[0m")
        print(f"\033[91m  This violates the one-parameter-per-slot rule.\033[0m\n")


def load_configs_and_seeds(state: ExplorationState, batch: BatchInfo):
    """PHASE 1: Load configs, force seeds, write seeds back to YAML."""
    if state.generate_data:
        print(f"\n\033[93mPHASE 1: Loading configs for {batch.n_slots} slots (data will be re-generated per slot)\033[0m")
    else:
        print(f"\n\033[93mPHASE 1: Loading configs for {batch.n_slots} slots (data is pre-generated)\033[0m")

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx

        # Sanitize description field before parsing (Claude may write unquoted
        # descriptions containing ': ' which breaks yaml.safe_load).
        _sanitize_description(state.config_paths[slot])

        # CRITICAL: Dataset suffix must always be _XX where XX is the slot number (_00, _01, _02, _03)
        # This is set once in init_slot_configs() and should NEVER change, even if Claude modifies the config.
        expected_dataset = f"{state.base_config_name}_{slot:02d}"
        if not expected_dataset.startswith(state.pre_folder):
            expected_dataset = state.pre_folder + expected_dataset

        # Read YAML once (avoid duplicate reads)
        with open(state.config_paths[slot], 'r') as f:
            yaml_data = yaml.safe_load(f)

        # Validate: dataset suffix must be slot-based (_00, _01, _02, _03), never iteration-based
        actual_dataset = yaml_data.get('dataset', '')
        expected_suffix = f"_{slot:02d}"

        if actual_dataset and not actual_dataset.endswith(expected_suffix):
            # Claude changed the dataset suffix — warn and fix it
            found_suffix = actual_dataset.split('_')[-1] if '_' in actual_dataset else 'unknown'
            print(
                f"\033[91mWARNING: Claude changed dataset suffix in slot {slot}!\033[0m\n"
                f"\033[91m  Expected suffix: {expected_suffix} (slot-based, IMMUTABLE)\033[0m\n"
                f"\033[91m  Found dataset:   {actual_dataset}\033[0m\n"
                f"\033[91m  Found suffix:    {found_suffix}\033[0m\n"
                f"\033[93m  Restoring to correct slot-based suffix...\033[0m"
            )
            # Reconstruct dataset with correct slot suffix
            yaml_data['dataset'] = expected_dataset

        # Force seeds (pipeline-controlled — LLM cannot override)
        assert iteration >= 1, f"iteration must be >= 1 for valid seeds (got {iteration} from batch.iterations[{slot_idx}])"
        sim_seed = (iteration * 1000 + slot) % (2**32)
        train_seed = (iteration * 1000 + slot + 500) % (2**32)
        batch.slot_seeds[slot] = {'simulation': sim_seed, 'training': train_seed}

        # Update YAML with forced seeds and slot-based dataset
        yaml_data['simulation']['seed'] = sim_seed
        yaml_data['training']['seed'] = train_seed
        yaml_data['dataset'] = expected_dataset
        # Restore intended n_epochs from claude section (training section gets
        # overwritten by yaml.dump round-trips, claude section is authoritative)
        yaml_data['training']['n_epochs'] = yaml_data.get('claude', {}).get('n_epochs', 1)

        # Write updated YAML back to file (cluster reads from file)
        with open(state.config_paths[slot], 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        # Load config object for in-memory use and storage
        config = NeuralGraphConfig.from_yaml(state.config_paths[slot])
        config.config_file = state.pre_folder + state.slot_names[slot]
        batch.configs[slot] = config

        if state.device is None:
            state.device = set_device(config.training.device)

    seed_info = "\n".join(
        f"  Slot {s}: simulation_seed={batch.slot_seeds[s]['simulation']}, "
        f"training_seed={batch.slot_seeds[s]['training']}"
        for s in range(batch.n_slots)
    )
    print(f"\033[90mSeeds (forced by pipeline):\n{seed_info}\033[0m")

    # Validate causality: warn if LLM changed >1 training param per slot vs slot 0
    _check_causality(state, batch)


# ---------------------------------------------------------------------------
# Phase 1.5 + 2 + 3: Training
# ---------------------------------------------------------------------------

def should_generate_data(state: ExplorationState, batch: BatchInfo) -> bool:
    """Return True if data should be generated for this batch.

    Logic (generate_data: false — new default):
      - Always generate on the very first batch (batch_first == 1) so each slot
        gets a different-seed dataset at startup.
      - On subsequent batches, only generate if the agent set
        claude.test_robustness_seed: true in any slot config (then reset the flag).
    Legacy (generate_data: true): generate every batch as before.
    """
    if state.generate_data:
        return True  # legacy mode: re-generate every batch
    if batch.batch_first == 1:
        return True  # new mode: generate once at startup
    # Check if agent requested robustness re-seeding
    for slot in range(batch.n_slots):
        with open(state.config_paths[slot], 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg.get('claude', {}).get('test_robustness_seed', False):
            print(f"\033[93m  slot {slot}: test_robustness_seed=true — triggering data re-generation\033[0m")
            return True
    return False


def _reset_robustness_seed_flag(state: ExplorationState, batch: BatchInfo):
    """Clear test_robustness_seed in all slot configs after re-generation."""
    for slot in range(batch.n_slots):
        with open(state.config_paths[slot], 'r') as f:
            cfg = yaml.safe_load(f)
        if cfg.get('claude', {}).get('test_robustness_seed', False):
            cfg['claude']['test_robustness_seed'] = False
            with open(state.config_paths[slot], 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def generate_data_locally(state: ExplorationState, batch: BatchInfo):
    """PHASE 1.5: Generate data locally for all slots."""
    print(f"\n\033[93mPHASE 1.5: Generating data locally for {batch.n_slots} slots\033[0m")
    from connectome_gnn.generators.graph_data_generator import data_generate

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = batch.configs[slot]
        print(f"\033[90m  slot {slot} (iter {iteration}): generating data with seed={batch.slot_seeds[slot]['simulation']}\033[0m")
        data_generate(
            config=config,
            device=state.device,
            visualize=False,
            run_vizualized=0,
            style="color",
            alpha=1,
            erase=True,
            save=True,
            step=100,
            compute_ranks=False,
        )
    _reset_robustness_seed_flag(state, batch)


def run_cluster_training(state: ExplorationState, batch: BatchInfo):
    """PHASE 2-3: Submit cluster jobs, wait, auto-repair failed jobs."""
    print(f"\n\033[93mPHASE 2: Submitting {batch.n_slots} flyvis training jobs to cluster (gpu_{state.node_name})\033[0m")

    # Guardrail: verify cluster repo is clean before submitting (warning only)
    if not check_cluster_repo():
        print("\033[93mWARNING: cluster repo has uncommitted changes — proceeding anyway\033[0m")

    job_ids = {}
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = batch.configs[slot]
        jid = submit_cluster_job(
            slot=slot,
            config_path=state.config_paths[slot],
            analysis_log_path=state.analysis_log_paths[slot],
            config_file_field=config.config_file,
            log_dir=state.log_dir,
            erase=True,
            node_name=state.node_name,
            conda_env=state.conda_env,
            n_cpus=state.n_cpus,
            device=config.training.device,
            exploration_dir=state.exploration_dir,
            iteration=iteration,
            output_root=get_data_root(),
            hard_runtime_limit_min=state.hard_runtime_limit_min,
        )
        if jid:
            job_ids[slot] = jid
        else:
            batch.job_results[slot] = False

    if job_ids:
        print(f"\n\033[93mPHASE 3.1: Waiting for {len(job_ids)} training jobs to complete\033[0m")
        # Per-slot log_dirs so the metrics-aware waiter can read each slot's
        # tmp_training/metrics.log and print conn/Vr/τ R² with color coding.
        slot_log_dirs = {s: log_path(batch.configs[s].config_file) for s in job_ids}
        cluster_results = wait_for_cluster_jobs_with_metrics(
            job_ids, slot_log_dirs, poll_interval=300, metrics_interval=300,
            job_prefix='cluster_train',
        )
        batch.job_results.update(cluster_results)

    # Auto-repair for failed jobs
    _auto_repair_failed_jobs(state, batch)


def _auto_repair_failed_jobs(state: ExplorationState, batch: BatchInfo):
    """Attempt to auto-repair failed cluster training jobs."""
    for slot_idx in range(batch.n_slots):
        if batch.job_results.get(slot_idx) != False:
            continue

        err_content = None
        err_file = f"{state.log_dir}/training_error_{slot_idx:02d}.log"
        lsf_err_file = f"{state.log_dir}/cluster_train_{slot_idx:02d}.err"

        for ef_path in [err_file, lsf_err_file]:
            if os.path.exists(ef_path):
                try:
                    with open(ef_path, 'r') as ef:
                        content = ef.read()
                    if 'FLYVIS SUBPROCESS ERROR' in content or 'Traceback' in content:
                        err_content = content
                        break
                except Exception:
                    pass

        if not err_content:
            continue

        print(f"\033[91m  slot {slot_idx} (iter {batch.iterations[slot_idx]}): TRAINING ERROR detected — attempting auto-repair\033[0m")

        code_files = [
            'src/connectome_gnn/models/graph_trainer.py',
            'src/connectome_gnn/models/Signal_Propagation.py',
            'GNN_PlotFigure.py',
        ]
        modified_code = get_modified_code_files(state.root_dir, code_files) if is_git_repo(state.root_dir) else []

        if not modified_code:
            print(f"\033[93m  slot {slot_idx} (iter {batch.iterations[slot_idx]}): no modified code files to repair — skipping\033[0m")
            continue

        max_repair_attempts = 3
        repaired = False
        for attempt in range(max_repair_attempts):
            print(f"\033[93m  slot {slot_idx} (iter {batch.iterations[slot_idx]}): repair attempt {attempt + 1}/{max_repair_attempts}\033[0m")
            repair_prompt = f"""TRAINING CRASHED - Please fix the code error.

Error traceback (last 3KB):
```
{err_content[-3000:]}
```

Modified files (these are the suspects):
{chr(10).join(f'- {state.root_dir}/{f}' for f in modified_code)}

Repair rules:
1. Fix ONLY the bug shown in the traceback. Do not refactor or add features.
2. Do NOT re-introduce the same pattern that caused the crash. Read the
   traceback carefully and identify what specifically failed.
3. **torch.compile + pydantic incompatibility** — if the traceback contains
   `torch._dynamo.exc.Unsupported`, `__getattribute__`, or mentions Dynamo
   tracing failures: the cause is almost always `getattr(pydantic_obj, ...)`
   or `pydantic_obj.attr` reached from a function inside the compiled
   forward/loss path (typically Regularizer.compute() or compute_update_regul()).
   FIX: read the config value ONCE in __init__ or _update_coeffs (which run
   outside the compiled region), store as a plain float on `self._foo`, and
   reference `self._foo` inside compute(). Never touch the pydantic config
   from inside compute().
4. If you genuinely cannot identify or fix the bug, print exactly the token
   `CANNOT_FIX` and stop.

Attempt {attempt + 1} of {max_repair_attempts}."""

            repair_cmd = [
                'claude', '-p', repair_prompt,
                '--output-format', 'text', '--max-turns', '10',
                '--allowedTools', 'Read', 'Edit', 'Write'
            ]
            repair_result = subprocess.run(repair_cmd, cwd=state.root_dir, capture_output=True, text=True)
            if 'CANNOT_FIX' in repair_result.stdout:
                print(f"\033[91m  slot {slot_idx} (iter {batch.iterations[slot_idx]}): Claude cannot fix — stopping repair\033[0m")
                break

            print(f"\033[96m  slot {slot_idx} (iter {batch.iterations[slot_idx]}): resubmitting after repair\033[0m")
            check_cluster_repo()
            config = batch.configs[slot_idx]
            jid = submit_cluster_job(
                slot=slot_idx,
                config_path=state.config_paths[slot_idx],
                analysis_log_path=state.analysis_log_paths[slot_idx],
                config_file_field=config.config_file,
                log_dir=state.log_dir,
                erase=True,
                node_name=state.node_name,
                conda_env=state.conda_env,
                n_cpus=state.n_cpus,
                device=config.training.device,
                exploration_dir=state.exploration_dir,
                iteration=batch.iterations[slot_idx],
                output_root=get_data_root(),
                hard_runtime_limit_min=state.hard_runtime_limit_min,
            )
            if jid:
                retry_results = wait_for_cluster_jobs_with_metrics(
                    {slot_idx: jid},
                    {slot_idx: log_path(config.config_file)},
                    poll_interval=300, metrics_interval=300,
                    job_prefix='cluster_train',
                )
                if retry_results.get(slot_idx):
                    batch.job_results[slot_idx] = True
                    repaired = True
                    print(f"\033[92m  slot {slot_idx} (iter {batch.iterations[slot_idx]}): repair successful!\033[0m")
                    break
                for ef_path in [err_file, lsf_err_file]:
                    if os.path.exists(ef_path):
                        try:
                            with open(ef_path, 'r') as ef:
                                err_content = ef.read()
                            break
                        except Exception:
                            pass

        if not repaired:
            print(f"\033[91m  slot {slot_idx} (iter {batch.iterations[slot_idx]}): repair failed after {max_repair_attempts} attempts — skipping\033[0m")
            if is_git_repo(state.root_dir):
                for fp in code_files:
                    try:
                        subprocess.run(['git', 'checkout', 'HEAD', '--', fp],
                                      cwd=state.root_dir, capture_output=True, timeout=10)
                    except Exception:
                        pass


def run_local_test_plot(state: ExplorationState, batch: BatchInfo):
    """PHASE 3.5: Run test locally (cluster mode — cluster only did training)."""
    print(f"\n\033[93mPHASE 3.5: Running test locally for {batch.n_slots} slots\033[0m")
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        if not batch.job_results.get(slot, False):
            print(f"\033[90m  slot {slot} (iter {iteration}): skipping test+plot (training failed)\033[0m")
            continue
        config = batch.configs[slot]
        print(f"\033[90m  slot {slot} (iter {iteration}): testing and plotting locally...\033[0m")
        log_file = open(state.analysis_log_paths[slot], 'a')

        # Test
        data_test(
            config=config,
            visualize=False,
            style="color name continuous_slice",
            verbose=False,
            best_model='best',
            run=0,
            test_mode="",
            sample_embedding=False,
            step=10,
            n_rollout_frames=1000,
            device=state.device,
            particle_of_interest=0,
            new_params=None,
            rollout_without_noise=True,
            log_file=log_file,
        )

        # Plot with skip_svd=True to skip expensive SVD analysis and avoid OOM
        from GNN_PlotFigure import data_plot
        slot_config_file = state.pre_folder + state.slot_names[slot]
        folder_name = log_path(state.pre_folder, 'tmp_results') + '/'
        os.makedirs(folder_name, exist_ok=True)
        data_plot(
            config=config,
            epoch_list=['best'],
            style='color',
            extended='plots',
            device=state.device,
            log_file=log_file,
            skip_svd=True,
        )
        log_file.close()


def _color_metric(val_str, green_thresh, orange_thresh):
    """Return ANSI-colored string: green >= green_thresh, orange >= orange_thresh, else red."""
    try:
        val = float(val_str)
        if val >= green_thresh:
            return f"\033[92m{val:.3f}\033[0m"
        elif val >= orange_thresh:
            return f"\033[93m{val:.3f}\033[0m"
        else:
            return f"\033[91m{val:.3f}\033[0m"
    except (ValueError, TypeError):
        return f"\033[90m{val_str}\033[0m"


def _print_batch_results(state: ExplorationState, batch: BatchInfo):
    """Print per-slot metrics with red/orange/green color coding."""
    print(f"\n\033[94m{'='*60}\033[0m")
    print(f"\033[94mBATCH RESULTS: iterations {batch.batch_first}-{batch.batch_last}\033[0m")
    print(f"\033[94m{'='*60}\033[0m")

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        if not batch.job_results.get(slot, False):
            print(f"  Slot {slot} (iter {iteration}):  \033[91mFAILED\033[0m")
            continue

        slot_log = state.analysis_log_paths[slot]
        if not os.path.exists(slot_log):
            print(f"  Slot {slot} (iter {iteration}):  \033[90mno log\033[0m")
            continue

        with open(slot_log, 'r') as f:
            log_content = f.read()

        def _p(key):
            m = re.search(rf'{key}[=:]\s*([\d.eE+-]+|nan)', log_content)
            return m.group(1) if m else None

        conn_r2       = _p('connectivity_R2')
        tau_r2        = _p('tau_R2')
        vrest_r2      = _p('V_rest_R2')
        clust_acc     = _p('cluster_accuracy')
        rollout_r     = _p('rollout_pearson')
        hid_rollout_r = _p('hidden_rollout_pearson')
        vis_rollout_r = _p('visible_rollout_pearson')
        hid_nnr_pear  = _p('hidden_nnr_pearson')
        anc_nnr_pear  = _p('anchor_nnr_pearson')
        onestep_r     = _p('onestep_pearson')
        train_min     = _p('training_time_min')

        parts = []
        if conn_r2:
            parts.append(f"conn={_color_metric(conn_r2, 0.9, 0.5)}")
        if vrest_r2:
            parts.append(f"Vr={_color_metric(vrest_r2, 0.9, 0.5)}")
        if tau_r2:
            parts.append(f"τ={_color_metric(tau_r2, 0.9, 0.5)}")
        if clust_acc:
            parts.append(f"cl={_color_metric(clust_acc, 0.9, 0.5)}")
        # Rollout: prefer hidden/visible split when available (hidden-NGP runs),
        # else the aggregate rollout_pearson, else one-step r.
        if hid_rollout_r and vis_rollout_r:
            parts.append(f"rH={_color_metric(hid_rollout_r, 0.9, 0.5)}({_color_metric(vis_rollout_r, 0.9, 0.5)})")
        elif rollout_r:
            parts.append(f"rN={_color_metric(rollout_r, 0.9, 0.5)}")
        elif onestep_r:
            parts.append(f"r1={_color_metric(onestep_r, 0.9, 0.5)}")
        # Hidden-NGP diagnostics
        if hid_nnr_pear:
            nnr_str = f"nnr={_color_metric(hid_nnr_pear, 0.5, 0.2)}"
            if anc_nnr_pear:
                nnr_str += f"({_color_metric(anc_nnr_pear, 0.5, 0.2)})"
            parts.append(nnr_str)
        if train_min:
            parts.append(f"\033[90mtrain={float(train_min):.1f}min\033[0m")

        metrics_str = "  ".join(parts) if parts else "\033[90mno metrics yet\033[0m"
        print(f"  Slot {slot} (iter {iteration}):  {metrics_str}")


def run_cluster_test_plot(state: ExplorationState, batch: BatchInfo):
    """PHASE 3.2: Submit test+plot jobs to cluster for all successful slots, then print results."""
    successful_slots = [s for s in range(batch.n_slots) if batch.job_results.get(s, False)]
    print(f"\n\033[93mPHASE 3.2: Submitting {len(successful_slots)} test+plot jobs to cluster (gpu_{state.node_name})\033[0m")

    job_ids = {}
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        if not batch.job_results.get(slot, False):
            print(f"\033[90m  slot {slot} (iter {iteration}): skipping test+plot (training failed)\033[0m")
            continue
        config = batch.configs[slot]
        jid = submit_cluster_test_plot_job(
            slot=slot,
            config_path=state.config_paths[slot],
            analysis_log_path=state.analysis_log_paths[slot],
            config_file_field=config.config_file,
            log_dir=state.log_dir,
            node_name=state.node_name,
            conda_env=state.conda_env,
            n_cpus=state.n_cpus,
            device=config.training.device,
            iteration=iteration,
            output_root=get_data_root(),
            hard_runtime_limit_min=state.hard_runtime_limit_min,
        )
        if jid:
            job_ids[slot] = jid
        else:
            batch.job_results[slot] = False

    if job_ids:
        print(f"\n\033[93mPHASE 3.2: Waiting for {len(job_ids)} test+plot jobs to complete\033[0m")
        test_plot_results = wait_for_cluster_jobs(
            job_ids, log_dir=state.log_dir, poll_interval=60, job_prefix='cluster_test_plot'
        )

        # One retry pass for transient test+plot failures (cluster hiccups,
        # NFS races, etc.). Training already succeeded for these slots.
        retry_slots = [s for s, ok in test_plot_results.items() if not ok]
        if retry_slots:
            print(f"\033[93m  retrying {len(retry_slots)} failed test+plot job(s) once\033[0m")
            retry_job_ids = {}
            for slot in retry_slots:
                config = batch.configs[slot]
                jid = submit_cluster_test_plot_job(
                    slot=slot,
                    config_path=state.config_paths[slot],
                    analysis_log_path=state.analysis_log_paths[slot],
                    config_file_field=config.config_file,
                    log_dir=state.log_dir,
                    node_name=state.node_name,
                    conda_env=state.conda_env,
                    n_cpus=state.n_cpus,
                    device=config.training.device,
                    iteration=batch.iterations[slot],
                    output_root=get_data_root(),
                    hard_runtime_limit_min=state.hard_runtime_limit_min,
                )
                if jid:
                    retry_job_ids[slot] = jid
                else:
                    test_plot_results[slot] = False
            if retry_job_ids:
                retry_results = wait_for_cluster_jobs(
                    retry_job_ids, log_dir=state.log_dir, poll_interval=60,
                    job_prefix='cluster_test_plot',
                )
                test_plot_results.update(retry_results)

        for slot, success in test_plot_results.items():
            if not success:
                batch.job_results[slot] = False
                print(f"\033[91m  slot {slot}: test+plot FAILED (after retry)\033[0m")

    _print_batch_results(state, batch)


def run_local_pipeline(state: ExplorationState, batch: BatchInfo):
    """PHASE 2 local: Generate + train + test sequentially."""
    print(f"\n\033[93mPHASE 2: Training {batch.n_slots} flyvis models locally (sequential)\033[0m")

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = batch.configs[slot]
        print(f"\033[90m  slot {slot} (iter {iteration}): training locally...\033[0m")

        config.training.save_all_checkpoints = False

        log_file = open(state.analysis_log_paths[slot], 'w')

        # Generate data on first batch or when agent requests robustness re-seeding
        if should_generate_data(state, batch):
            from connectome_gnn.generators.graph_data_generator import data_generate
            print(f"\033[90m  slot {slot} (iter {iteration}): generating data with seed={batch.slot_seeds[slot]['simulation']}\033[0m")
            data_generate(
                config=config,
                device=state.device,
                visualize=True,
                run_vizualized=0,
                style="color",
                alpha=1,
                erase=True,
                save=True,
                step=100,
            )

        # Train
        data_train(
            config=config,
            erase=True,
            best_model=state.best_model,
            style='color',
            device=state.device,
            log_file=log_file
        )

        # Test
        data_test(
            config=config,
            visualize=False,
            style="color name continuous_slice",
            verbose=False,
            best_model='best',
            run=0,
            test_mode="",
            sample_embedding=False,
            step=10,
            n_rollout_frames=1000,
            device=state.device,
            particle_of_interest=0,
            new_params=None,
            rollout_without_noise=True,
            log_file=log_file,
        )

        # Plot with skip_svd=True to skip expensive SVD analysis and avoid OOM
        from GNN_PlotFigure import data_plot
        slot_config_file = state.pre_folder + state.slot_names[slot]
        folder_name = log_path(state.pre_folder, 'tmp_results') + '/'
        os.makedirs(folder_name, exist_ok=True)
        data_plot(
            config=config,
            epoch_list=['best'],
            style='color',
            extended='plots',
            device=state.device,
            log_file=log_file,
            skip_svd=True,
        )

        # Copy models to exploration dir
        slot_log_dir = os.path.join('log', config.config_file)
        src_models = globmod.glob(os.path.join(slot_log_dir, 'models', '*.pt'))
        if src_models:
            models_save_dir = os.path.join(state.exploration_dir, 'models')
            os.makedirs(models_save_dir, exist_ok=True)
            for src in src_models:
                fname = os.path.basename(src)
                dst = os.path.join(models_save_dir, f'iter_{iteration:03d}_slot_{slot:02d}_{fname}')
                shutil.copy2(src, dst)
                print(f"\033[92m  copied model: {dst}\033[0m")

        batch.job_results[slot] = True
        log_file.close()


# ---------------------------------------------------------------------------
# Phase 4: Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(state: ExplorationState, batch: BatchInfo):
    """PHASE 4: Save exploration artifacts + check training time."""
    print("\n\033[93mPHASE 4: Saving exploration artifacts\033[0m")

    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        if not batch.job_results.get(slot, False):
            print(f"\033[90m  slot {slot} (iter {iteration}): skipping (training failed)\033[0m")
            continue

        config = batch.configs[slot]

        # Save exploration artifacts (flyvis-specific panels)
        iter_in_block = (iteration - 1) % state.n_iter_block + 1
        artifact_paths = save_exploration_artifacts_flyvis(
            state.root_dir, state.exploration_dir, config, state.slot_names[slot],
            state.pre_folder, iteration,
            iter_in_block=iter_in_block, block_number=batch.block_number
        )
        batch.activity_paths[slot] = artifact_paths['activity_path']

        # Save config file for EVERY iteration
        config_save_dir = f"{state.exploration_dir}/config"
        os.makedirs(config_save_dir, exist_ok=True)
        dst_config = f"{config_save_dir}/iter_{iteration:03d}_slot_{slot:02d}.yaml"
        shutil.copy2(state.config_paths[slot], dst_config)

        # Check training time
        slot_log_path = state.analysis_log_paths[slot]
        if os.path.exists(slot_log_path):
            with open(slot_log_path, 'r') as f:
                log_content = f.read()
            time_m = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
            if time_m:
                training_time = float(time_m.group(1))
                if training_time > state.training_time_target_min:
                    print(f"\033[93m  WARNING: slot {slot} (iter {iteration}) training took {training_time:.1f} min (>{state.training_time_target_min} min target)\033[0m")
                else:
                    print(f"\033[92m  slot {slot} (iter {iteration}): training time {training_time:.1f} min\033[0m")


# ---------------------------------------------------------------------------
# Phase 5: Claude analysis
# ---------------------------------------------------------------------------

def build_code_brief_context(state: ExplorationState) -> str:
    """Find code session briefs with .done markers for inclusion in analysis prompt."""
    briefs_dir = os.path.join(state.exploration_dir, 'briefs')
    if not os.path.isdir(briefs_dir):
        return ""

    applied_briefs = []
    for bf in sorted(os.listdir(briefs_dir)):
        if bf.startswith('block_') and bf.endswith('_brief.md'):
            bnum = bf.replace('block_', '').replace('_brief.md', '')
            # Check both new and old marker names for backward compat
            new_marker = os.path.join(state.exploration_dir, f'code_session_block_{bnum}.done')
            old_marker = os.path.join(state.exploration_dir, f'phase_a_block_{bnum}.done')
            if os.path.exists(new_marker) or os.path.exists(old_marker):
                applied_briefs.append(os.path.join(briefs_dir, bf))

    if not applied_briefs:
        return ""

    return (
        "\nCode session changes (READ THIS — new explorable parameters): "
        + ", ".join(applied_briefs)
        + "\nThese briefs describe structural code changes and NEW config fields added to the codebase. "
        + "Read them to learn about new explorable training/simulation parameters you can set in YAML configs.\n"
    )


def run_claude_analysis(state: ExplorationState, batch: BatchInfo):
    """PHASE 6: Claude analyzes results + proposes next mutations."""
    print("\n\033[93mPHASE 6: Claude analysis + next mutations\033[0m")

    # Build slot info string
    slot_info_lines = []
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        status = "COMPLETED" if batch.job_results.get(slot, False) else "FAILED"
        act_path = batch.activity_paths.get(slot, "N/A")
        slot_log_dir = os.path.join('log', state.pre_folder + state.slot_names[slot])
        slot_info_lines.append(
            f"Slot {slot} (iteration {iteration}) [{status}]:\n"
            f"  Seeds: simulation={batch.slot_seeds[slot]['simulation']}, "
            f"training={batch.slot_seeds[slot]['training']}\n"
            f"  Metrics: {state.analysis_log_paths[slot]}\n"
            f"  Activity: {act_path}\n"
            f"  Config: {state.config_paths[slot]}\n"
            f"  Training plots: {slot_log_dir}/tmp_training/"
        )
    slot_info = "\n\n".join(slot_info_lines)

    code_brief_context = build_code_brief_context(state)

    prompt = analysis_prompt(state, batch, slot_info, code_brief_context)

    timeout_sec = max(60, state.claude_call_timeout_min * 60)
    print(
        f"\033[93mClaude analysis (timeout={state.claude_call_timeout_min}min, "
        f"live feedback below)...\033[0m"
    )
    output_text, timed_out = run_claude_cli_with_timeout(
        prompt, state.root_dir,
        allowed_tools=['Read', 'Edit', 'Write'],
        timeout_sec=timeout_sec,
        max_turns=500,
        log_prefix=f'[batch{batch.batch_first}] ',
    )

    if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
        print(f"\n\033[91mOAuth token expired at batch {batch.batch_first}-{batch.batch_last}\033[0m")
        print("\033[93mTo resume: 1. Run: claude /login  2. Then re-run with --resume\033[0m")
        sys.exit(1)

    if timed_out:
        print(
            f"\n\033[93mWARNING: Claude analysis timed out after "
            f"{state.claude_call_timeout_min}min. Slot configs left at their "
            f"previous-batch values; proceeding with re-test.\033[0m"
        )

    # Validate slot YAMLs are still parseable.
    for slot, path in state.config_paths.items():
        try:
            with open(path) as f:
                yaml.safe_load(f)
        except Exception as e:
            print(
                f"\033[91mFATAL: slot {slot} YAML at {path} no longer parses "
                f"({type(e).__name__}: {e}). Restore from {state.source_config} "
                f"and re-run.\033[0m"
            )
            sys.exit(1)

    if output_text.strip():
        with open(state.reasoning_log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"=== Batch {batch.batch_first}-{batch.batch_last} ===\n")
            f.write(f"{'='*60}\n")
            if timed_out:
                f.write(f"[NOTE: timed out after {state.claude_call_timeout_min}min]\n")
            f.write(output_text.strip())
            f.write("\n\n")

    # Sanitize config files written by Claude: quote description values that
    # contain ": " so yaml.safe_load doesn't mistake them for mapping keys.
    for slot in range(batch.n_slots):
        _sanitize_description(state.config_paths[slot])


def _sanitize_description(path: str):
    """Fix unquoted description lines written by Claude that contain ': '."""
    import re
    try:
        with open(path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return
    # Match multi-line description: value that spans a continuation line
    # (the folded form written by Claude when the value is long)
    # Pattern: description: <text that contains ': '> possibly spanning next indented line
    def _quote_description(m):
        raw = m.group(1)
        # Collapse folded continuation (indented next line) into single string
        raw = re.sub(r'\n[ \t]+', ' ', raw).rstrip()
        # Only requote if it contains ': ' (the problematic YAML pattern)
        if ': ' in raw:
            escaped = raw.replace("'", "''")
            return f"description: '{escaped}'"
        return m.group(0)
    new_content = re.sub(
        r'^description: (.*(?:\n[ \t]+.*)*)',
        _quote_description,
        content,
        flags=re.MULTILINE,
    )
    if new_content != content:
        with open(path, 'w') as f:
            f.write(new_content)


# ---------------------------------------------------------------------------
# Finalize batch
# ---------------------------------------------------------------------------

def finalize_batch(state: ExplorationState, batch: BatchInfo):
    """Tree visualization, protocol/memory snapshots."""
    # Save instruction file at first iteration of each block
    protocol_save_dir = f"{state.exploration_dir}/protocol"
    os.makedirs(protocol_save_dir, exist_ok=True)
    if batch.iter_in_block_first == 1:
        dst_instruction = f"{protocol_save_dir}/block_{batch.block_number:03d}.md"
        if os.path.exists(state.instruction_path):
            shutil.copy2(state.instruction_path, dst_instruction)

    # Save memory file at end of block
    if batch.is_block_end:
        memory_save_dir = f"{state.exploration_dir}/memory"
        os.makedirs(memory_save_dir, exist_ok=True)
        dst_memory = f"{memory_save_dir}/block_{batch.block_number:03d}_memory.md"
        if os.path.exists(state.memory_path):
            shutil.copy2(state.memory_path, dst_memory)
            print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

    # Print batch summary
    n_success = sum(1 for v in batch.job_results.values() if v)
    n_failed = sum(1 for v in batch.job_results.values() if not v)
    print(f"\n\033[92mBatch {batch.batch_first}-{batch.batch_last} complete: {n_success} succeeded, {n_failed} failed\033[0m")
