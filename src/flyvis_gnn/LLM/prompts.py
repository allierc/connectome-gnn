"""LLM prompt templates for the exploration pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import BatchInfo, ExplorationState


def batch_0_prompt(state: 'ExplorationState', slot_list: str, seed_info: str) -> str:
    """Build the BATCH 0 start prompt for initializing N config variations."""
    return f"""PARALLEL START: Initialize {state.n_parallel} config variations for the first batch.

Instructions (follow all instructions): {state.instruction_path}
Working memory: {state.memory_path}
Full log (append only): {state.analysis_path}
User input (read and acknowledge any pending instructions): {state.user_input_path}

Config files to edit (all {state.n_parallel}):
{slot_list}

Seeds (forced by pipeline — DO NOT modify simulation.seed or training.seed in configs):
{seed_info}
Log these seed values in your iteration entries.

Read the instructions and the base config, then set up {state.n_parallel} experiments.

CAUSALITY RULE (MANDATORY):
- Slot 0 = BASELINE (identical to base config, no changes).
- Slots 1-{state.n_parallel - 1}: each changes EXACTLY ONE parameter from baseline.
- If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.
- Do NOT change parameters not listed in the current block focus (e.g. do not change w_init_mode, W_L2, batch_size unless the block says so).
- Each config already has a unique dataset name — do NOT change the dataset field.

{state.sim_constraint}
IMPORTANT: Training time target is ~{state.training_time_target_min} min per iteration. Adjust data_augmentation_loop (DAL) to hit this target: if training_time_min < 40, increase DAL; if > 70, decrease DAL. Longer training = better W convergence.
IMPORTANT: Read user_input.md — if there are pending instructions, acknowledge them by appending to the "Acknowledged" section with timestamp and moving them out of "Pending Instructions".

Write the planned mutations to the working memory file."""


def analysis_prompt(state: 'ExplorationState', batch: 'BatchInfo',
                    slot_info: str, code_brief_context: str) -> str:
    """Build the PHASE 6 Claude analysis prompt."""
    block_end_marker = "\n>>> BLOCK END <<<" if batch.is_block_end else ""

    return f"""Batch iterations {batch.batch_first}-{batch.batch_last} / {state.n_iterations}
Block info: block {batch.block_number}, iterations {batch.iter_in_block_first}-{batch.iter_in_block_last}/{state.n_iter_block} within block{block_end_marker}

PARALLEL MODE: Analyze {batch.n_slots} results, then propose next {state.n_parallel} mutations.

Instructions (follow all instructions): {state.instruction_path}
Working memory: {state.memory_path}
Full log (append only): {state.analysis_path}
User input (read and acknowledge any pending instructions): {state.user_input_path}
{code_brief_context}
{slot_info}

Seeds are forced by pipeline (DO NOT modify simulation.seed or training.seed in configs).
The seed values for this batch are shown in each slot above. Log them in your iteration entries.

Analyze all {batch.n_slots} results. For each successful slot:
1. Read the metrics from the analysis log.
2. Look at the connectivity matrix heatmap in tmp_training/matrix/connectivity_*.png — compare GT vs learned W visually. Note in your log entry: is the learned W sparse enough? Are signs correct? Is the structure emerging?
3. Write a separate iteration entry (## Iter N: ...) to the full log and memory file.
Then edit all {state.n_parallel} config files to set up the next batch of {state.n_parallel} experiments.

CAUSALITY RULE (MANDATORY):
- Pick a PARENT config (best so far or baseline).
- Slot 0 = PARENT config unchanged (control).
- Slots 1-{state.n_parallel - 1}: each changes EXACTLY ONE parameter from the parent.
- If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.
- Do NOT change parameters outside the current block focus unless the block says so.
- Exception: ROBUSTNESS TEST — set ALL {state.n_parallel} slots to the SAME config (pipeline forces different seeds). Use this only to confirm a promising config.
- State your choice (exploration vs robustness test) in the log entry.

IMPORTANT: Do NOT change the 'dataset' field in any config — it must stay as-is for each slot.
{state.sim_constraint}
IMPORTANT: Training time target is ~{state.training_time_target_min} min per iteration. Check training_time_min and adjust DAL for next batch: if < 40 min increase DAL, if > 70 min decrease DAL. Use the full time budget — longer training improves W convergence.
IMPORTANT: Read user_input.md — if there are pending instructions, acknowledge them by appending to the "Acknowledged" section with a timestamp and moving them out of "Pending Instructions".
"""
