"""Code-change exploration main loop.

Adapter on top of the existing LLM/ HPO pipeline:

  block start:  Phase R → Phase S → Phase C  (via LLM_code.code_session)
  per iter:     load → train → test_plot → save → analysis → finalize
                (all imported from LLM/ — untouched)
  block end:    collect metrics → verdict → keep or revert

The per-iteration body is identical to GNN_LLM.py's body so that the existing
HPO optimisation still runs within a block, operating on top of whatever Phase
C just committed.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import yaml

from connectome_gnn.LLM import (
    finalize_batch,
    generate_data_locally,
    init_shared_files,
    init_slot_configs,
    load_configs_and_seeds,
    make_batch_info,
    run_batch_0,
    run_claude_analysis,
    run_cluster_test_plot,
    run_cluster_training,
    run_local_pipeline,
    save_artifacts,
    setup_exploration,
    should_generate_data,
)
from connectome_gnn.LLM.state import BatchInfo, ExplorationState
from connectome_gnn.utils import log_path

from connectome_gnn.LLM_code.code_session import run_code_session
from connectome_gnn.LLM_code.git_checkpoint import (
    GitCheckpointError,
    diff_since_start,
    keep,
    require_branch,
    revert,
    start_block,
)
from connectome_gnn.LLM_code.state import (
    DEFAULT_BLOCK_THEMES,
    DEFAULT_PHASE_TIME_LIMITS,
    CodeExplorationState,
)
from connectome_gnn.LLM_code.verdict import VerdictReport, collect_metrics_from_run_dirs, decide


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def build_code_state(
    base_state: ExplorationState,
    source_config_path: str,
    literature_allowlist_path: Optional[str] = None,
    falsified_path: Optional[str] = None,
) -> CodeExplorationState:
    """Read claude_code.* from the YAML and wrap base_state into a CodeExplorationState.

    Falls back to defaults for any missing keys. Does not raise if the
    `claude_code:` section is absent — the user can run a code-change
    exploration from a YAML that only has a `claude:` section, in which case
    the defaults apply.
    """
    with open(source_config_path) as f:
        cfg = yaml.safe_load(f) or {}
    cc = cfg.get("claude_code", {}) or {}

    themes = cc.get("block_themes", list(DEFAULT_BLOCK_THEMES))
    limits = dict(DEFAULT_PHASE_TIME_LIMITS)
    limits.update(cc.get("phase_time_limits", {}) or {})

    state = CodeExplorationState(
        base=base_state,
        block_themes=list(themes),
        phase_time_limits=limits,
    )

    # Literature allowlist: default to the package-bundled file.
    if literature_allowlist_path is None:
        literature_allowlist_path = os.path.join(
            base_state.root_dir,
            "src", "connectome_gnn", "LLM_code", "literature", "allowlist.json",
        )
    state.literature_allowlist_path = literature_allowlist_path

    # Falsified list: derived from exploration dir so it's per-exploration.
    if falsified_path is None:
        falsified_path = os.path.join(base_state.exploration_dir, "falsified.md")
    state.falsified_path = falsified_path

    # Make sure the code_session output dir exists so path helpers are stable.
    state._bdir()
    return state


def _block_number_for_iter(state: CodeExplorationState, iteration: int) -> int:
    """1-based block number from iteration number."""
    return (iteration - 1) // state.base.n_iter_block + 1


# ---------------------------------------------------------------------------
# Metrics collection for verdict
# ---------------------------------------------------------------------------

def _run_dirs_for_block(state: CodeExplorationState, block_number: int) -> List[str]:
    """List log directories for every slot × iteration in the given block.

    The pipeline state's config_paths maps slot → YAML path; we resolve to the
    matching `log/<config_file>/` directory via log_path(config.config_file).
    """
    import glob as globmod
    # Each slot's current YAML contains its .config_file; iterate slots.
    paths: List[str] = []
    for slot, config_path in state.base.config_paths.items():
        if not os.path.isfile(config_path):
            continue
        with open(config_path) as f:
            slot_cfg = yaml.safe_load(f) or {}
        cf = slot_cfg.get("config_file") or ""
        if not cf:
            continue
        paths.append(log_path(cf))

    # Also search for CV-style sibling dirs named base_name_cvNN if the runner used them.
    return sorted(set(paths))


def _append_to_memory(memory_path: str, section: str, body: str) -> None:
    if not memory_path:
        return
    with open(memory_path, "a") as f:
        f.write(f"\n## {section}\n\n{body}\n")


def _append_to_falsified(falsified_path: str, block_number: int, reason: str, diff: str) -> None:
    os.makedirs(os.path.dirname(falsified_path) or ".", exist_ok=True)
    with open(falsified_path, "a") as f:
        f.write(
            f"\n### Block {block_number:02d} — REVERTED\n"
            f"**Reason**: {reason}\n\n"
            f"<details><summary>diff</summary>\n\n```\n{diff[:4000]}\n```\n</details>\n"
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_exploration(args, root_dir: str, source_config_path: str) -> None:
    """Code-change exploration main loop.

    Shape of per-iteration body matches GNN_LLM.py's body line-for-line so the
    existing HPO analysis still runs inside each block.
    """
    # ---- base setup via existing LLM pipeline ----
    base_state = setup_exploration(args, root_dir)
    init_slot_configs(base_state, is_resume=args.resume)
    init_shared_files(base_state, is_resume=args.resume)

    state = build_code_state(base_state, source_config_path)

    # Guard: must be on the code-change branch.
    try:
        require_branch(root_dir)
    except GitCheckpointError as e:
        raise SystemExit(f"\n\033[91m[LLM_code] {e}\033[0m\n")

    # Load the instruction file once (applied to every R/S/C prompt).
    instruction_text = ""
    if base_state.instruction_path and os.path.isfile(base_state.instruction_path):
        with open(base_state.instruction_path) as f:
            instruction_text = f.read()

    # Batch 0 is still the HPO initialisation step (no training).
    if base_state.start_iteration == 1 and not args.resume:
        run_batch_0(base_state)

    # Main loop — same shape as GNN_LLM.py's loop, but wrapped with blocks.
    current_block: Optional[int] = None
    per_block_metrics: Dict[int, Dict[str, List[float]]] = {}
    current_checkpoint = None

    for batch_start in range(
        base_state.start_iteration,
        base_state.n_iterations + 1,
        base_state.n_parallel,
    ):
        batch: BatchInfo = make_batch_info(base_state, batch_start)

        # ---- BLOCK START ----
        if batch.is_block_start:
            # Close the previous block (if any) before starting a new one.
            if current_block is not None and current_checkpoint is not None:
                _finalize_block(state, current_block, current_checkpoint,
                                per_block_metrics.get(current_block, {}))

            current_block = batch.block_number
            current_checkpoint = start_block(root_dir, current_block)
            per_block_metrics[current_block] = {}

            # Set up block-scoped paths.
            state.staging_block_dir = state.staging_block_path(current_block)
            state.analysis_block_dir = os.path.join(state.staging_block_dir, "analysis")
            state.research_log_path = state.research_path_for(current_block)
            state.verdict_log_path = state.verdict_path_for(current_block)
            state.code_diffs_dir = os.path.dirname(state.code_diff_path_for(current_block))

            print(
                f"\n\033[96m[LLM_code] BLOCK {current_block:02d} START — "
                f"theme={state.theme_for_block(current_block)}  "
                f"start_sha={current_checkpoint.start_sha[:12]}\033[0m"
            )

            # Code session (R/S/C).
            cs_result = run_code_session(
                state=state,
                block_number=current_block,
                checkpoint=current_checkpoint,
                instruction_text=instruction_text,
            )
            # Append session summary to memory.
            _append_to_memory(
                base_state.memory_path,
                f"Block {current_block:02d} code-session",
                cs_result.as_markdown(),
            )

        # ---- per-iteration body (verbatim from GNN_LLM.py) ----
        print(
            f"\n\033[94mBATCH: iterations {batch.batch_first}-{batch.batch_last} / "
            f"{base_state.n_iterations}  (block {batch.block_number})\033[0m"
        )
        load_configs_and_seeds(base_state, batch)

        if "train" in base_state.task:
            if base_state.cluster_enabled:
                if should_generate_data(base_state, batch):
                    generate_data_locally(base_state, batch)
                run_cluster_training(base_state, batch)
                run_cluster_test_plot(base_state, batch)
            else:
                run_local_pipeline(base_state, batch)
        else:
            for slot in range(batch.n_slots):
                batch.job_results[slot] = True

        save_artifacts(base_state, batch)
        run_claude_analysis(base_state, batch)
        finalize_batch(base_state, batch)

        # Collect this iteration's metrics into the current block bucket.
        if current_block is not None:
            run_dirs = _run_dirs_for_block(state, current_block)
            m = collect_metrics_from_run_dirs(run_dirs)
            for key, vals in m.items():
                per_block_metrics[current_block].setdefault(key, []).extend(vals)

    # After the last iteration, close the final block.
    if current_block is not None and current_checkpoint is not None:
        _finalize_block(
            state, current_block, current_checkpoint,
            per_block_metrics.get(current_block, {}),
        )


def _finalize_block(
    state: CodeExplorationState,
    block_number: int,
    checkpoint,
    post_metrics: Dict[str, List[float]],
) -> None:
    """Run Phase V (verdict) and either keep or revert."""
    pre = state.pre_block_baseline
    report: VerdictReport = decide(pre=pre, post=post_metrics)

    # Persist the diff of the block regardless of decision.
    diff_text = diff_since_start(checkpoint)
    with open(state.code_diff_path_for(block_number), "w") as f:
        f.write(diff_text or "(no diff)\n")

    with open(state.verdict_path_for(block_number), "w") as f:
        f.write(report.as_markdown())
        f.write(f"\n## Post-block metrics (per seed)\n\n")
        for k, v in post_metrics.items():
            f.write(f"- {k}: {v}\n")
        f.write(f"\n## Pre-block baseline (per seed)\n\n")
        for k, v in pre.items():
            f.write(f"- {k}: {v}\n")

    if report.decision == "KEEP":
        keep(checkpoint)
        # Update baseline to this block's metrics for the next block's verdict.
        state.pre_block_baseline = {k: list(v) for k, v in post_metrics.items()}
        msg = f"KEEP — {report.reason}"
    else:
        new_head = revert(checkpoint, verdict_reason=report.reason)
        # Record in falsified registry.
        _append_to_falsified(state.falsified_path, block_number, report.reason, diff_text)
        msg = f"REVERT — {report.reason}  (new HEAD {new_head[:12] if new_head else 'n/a'})"

    print(
        f"\n\033[93m[LLM_code] BLOCK {block_number:02d} verdict: "
        f"{report.decision}\n  {msg}\033[0m"
    )
    _append_to_memory(
        state.base.memory_path,
        f"Block {block_number:02d} verdict",
        report.as_markdown(),
    )
