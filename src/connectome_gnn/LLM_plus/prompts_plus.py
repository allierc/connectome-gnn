"""Prompts for the pinned-conversation loop.

Two shapes, chosen per batch:

  FULL          names the instruction file, the working memory, the analysis
                log and user_input.md and tells the agent to read them. Costs
                a 40-60k-token rebuild of context. Sent at batch 0, at every
                block start, after a compaction, and after a failed --resume.

  INCREMENTAL   names only what is new since the previous batch — the four
                slot metric paths, this batch's seeds, the block header — and
                explicitly tells the agent NOT to re-read the instruction and
                memory files, because it is still in the conversation where it
                read the first and wrote the second. Sent for every other
                batch, which on a 9-block run is roughly two thirds of them.

The full text is always built, even when the incremental one is sent: it is
stashed on the session as the prompt to resend if --resume turns out to have
failed, so a dead conversation costs one retry rather than one lost batch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from connectome_gnn.LLM.prompts import analysis_prompt, batch_0_prompt

if TYPE_CHECKING:
    from connectome_gnn.LLM.state import BatchInfo, ExplorationState

    from .session import ClaudeSession


TOOLS = """YOU HAVE EXACTLY THREE TOOLS: Read, Edit, Write. The wrapper runs you
with `--allowedTools Read Edit Write`, so Bash, Glob, Grep and Task all fail — and
a failed call still costs wall-clock against a hard deadline. An observed run
spent 50 seconds retrying `diff` eight ways before giving up and Reading the files,
which is what this paragraph is here to save you.

So: compare two configs by Reading both and comparing them yourself; find a file
by using a path named in this prompt or in the instructions, never by searching
for it; make every config change with Edit."""


def _session_preamble(state: 'ExplorationState') -> str:
    """Tell the agent, once, that this conversation outlives the call."""
    n_batches = max(1, -(-state.n_iterations // max(1, state.n_parallel)))
    return f"""PERSISTENT SESSION. This is one conversation that spans the whole
exploration — about {n_batches} batches over {state.n_iterations} iterations. You will be
re-invoked in it after every batch of {state.n_parallel} training runs, with everything
you have already read and written still in context.

{TOOLS}

Two consequences, both of which save you most of your time budget:

  - READ THE INSTRUCTION FILE NOW, IN FULL. It does not change during the
    exploration, and you will not be asked to read it again except at block
    boundaries. Everything you skim now you will be missing for {n_batches} batches.
  - The working memory file is still the durable record — write to it every
    batch exactly as the instructions say, because a compaction of this
    conversation, or a crash of the loop, leaves it as the only trace. But you
    will not need to re-read what you yourself wrote.
"""


def session_batch_0_prompt(state: 'ExplorationState', slot_list: str,
                           seed_info: str, session: 'ClaudeSession') -> str:
    """The start call: the stock batch-0 prompt under the session preamble."""
    full = _session_preamble(state) + "\n" + batch_0_prompt(state, slot_list, seed_info)
    session.fallback_prompt = full
    session.sent_full = True
    return full


def _full_analysis_prompt(state: 'ExplorationState', batch: 'BatchInfo',
                          slot_info: str, code_brief_context: str,
                          reason: str) -> str:
    """The stock analysis prompt, prefixed with why the re-read is being asked."""
    return f"""RE-READ REQUESTED — {reason}

{TOOLS}

Read {state.instruction_path} and {state.memory_path} in full before analysing
anything below. After this batch you go back to incremental prompts, so what
you take from them now has to carry the next batches.

{analysis_prompt(state, batch, slot_info, code_brief_context)}"""


def _incremental_analysis_prompt(state: 'ExplorationState', batch: 'BatchInfo',
                                 slot_info: str, code_brief_context: str) -> str:
    """Only what is new since the previous batch."""
    block_end_marker = "\n>>> BLOCK END <<<" if batch.is_block_end else ""
    dal_low = int(state.training_time_target_min * 0.65)
    dal_high = int(state.training_time_target_min * 1.15)

    return f"""Batch iterations {batch.batch_first}-{batch.batch_last} / {state.n_iterations}
Block info: block {batch.block_number}, iterations {batch.iter_in_block_first}-{batch.iter_in_block_last}/{state.n_iter_block} within block{block_end_marker}

You are still in the conversation where you read {state.instruction_path} and
wrote every entry of this exploration. DO NOT re-read the instruction file, the
working memory or the full analysis log — you have them. Read only the {batch.n_slots}
per-slot files named below, plus {state.user_input_path} (which the user may
have edited since the last batch, so it is the one shared file worth re-reading).

Still Read/Edit/Write only — no Bash, no Glob, no Grep.

⏱ TIME BUDGET: {state.claude_call_timeout_min} minutes, hard. The wrapper SIGTERMs you at the
deadline and unfinished edits are lost. Because you are not rebuilding context,
spend it on the {batch.n_slots} log reads, the entries, and the {state.n_parallel} config edits — configs
first if you run short.

PARALLEL MODE: Analyze {batch.n_slots} results, then propose next {state.n_parallel} mutations.
{code_brief_context}
{slot_info}

Seeds are forced by the pipeline (do NOT modify simulation.seed or training.seed
in configs). The values for this batch are shown per slot above; log them.

For each successful slot: read the metrics from its analysis log, look at
tmp_training/Wij/connectivity_*.png against ground truth, and write one
`## Iter N: <title>` entry to both the full log and the working memory, with
every metric column the instructions list — not just connectivity_R2.

CAUSALITY RULE (unchanged, and still fatal to break): slot 0 = parent config
unchanged as the control; slots 1-{state.n_parallel - 1} each change EXACTLY ONE parameter from
that parent. State in the entry whether this is an exploration batch or a
robustness test (all {state.n_parallel} slots identical, pipeline forces different seeds).

Do NOT change the 'dataset' field in any config.
{state.sim_constraint}
Training time target is ~{state.training_time_target_min} min per iteration: if training_time_min came in
below {dal_low} raise data_augmentation_loop, above {dal_high} lower it.
If {state.user_input_path} has pending instructions, acknowledge them there with
a timestamp and move them out of "Pending Instructions".
"""


def session_analysis_prompt(state: 'ExplorationState', batch: 'BatchInfo',
                            slot_info: str, code_brief_context: str,
                            session: 'ClaudeSession') -> str:
    """Pick the full or the incremental prompt for this batch.

    Full whenever the conversation cannot be trusted to still hold the
    instruction file — the start of a block, the batch after a compaction, or
    the first batch of a re-entered process — and incremental otherwise. The
    full text is stashed on the session either way, as the retry prompt for a
    failed --resume.
    """
    if session.needs_full_context:
        reason = ("the conversation was compacted or restarted, so the "
                  "instruction file may no longer be in context")
    elif batch.is_block_start and batch.block_number > 1:
        # Block 1's start needs no re-read: batch 0 read everything moments
        # ago in this same conversation, and on a --resume that skipped batch
        # 0, needs_full_context above has already caught it.
        reason = f"block {batch.block_number} starts here"
    else:
        reason = ""

    full = _full_analysis_prompt(state, batch, slot_info, code_brief_context,
                                 reason or "conversation restarted")
    session.fallback_prompt = full
    session.sent_full = bool(reason)

    if reason:
        return full

    return _incremental_analysis_prompt(state, batch, slot_info, code_brief_context)
