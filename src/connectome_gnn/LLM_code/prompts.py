"""Prompt templates for the code-change exploration loop.

One template per phase (R, S, C). The Phase V verdict is automatic and has no
prompt; the next block's Phase R reads the verdict report in the memory file.

All templates are f-string functions to keep them simple and greppable. The
instruction file (see LLM_code/instructions/) is prepended to every prompt,
not templated here — the pipeline module does the concatenation.
"""

from __future__ import annotations

from typing import List, Optional


# ---------------------------------------------------------------------------
# Phase R — Research + optional analysis staging
# ---------------------------------------------------------------------------

def phase_r_prompt(
    block_number: int,
    theme: str,
    memory_text: str,
    falsified_text: str,
    allowlist_urls: List[str],
    research_log_path: str,
    analysis_block_dir: str,
    time_budget_sec: int,
) -> str:
    urls = "\n".join(f"  - {u}" for u in allowlist_urls) or "  (none)"
    return f"""> **FOCUS**: this is Phase R. Follow the `## [Phase R]` section of the
> instruction prepended above, together with `## Shared context`. The other
> phase sections are reference-only — do not act on them.

You are running Phase R (Research) of the code-change exploration loop.

BLOCK: {block_number:02d}
THEME: {theme}
TIME BUDGET: {time_budget_sec // 60} minutes HARD CAP. Pace yourself accordingly.

## Your goal in this phase
1. Read the memory file and the falsified-hypotheses registry (below).
2. OPTIONALLY stage ONE standalone *analysis* function under
      {analysis_block_dir}
   if prior plots don't answer a question you need to answer. Its
   purpose is to look at prior-block artifacts (checkpoints, metrics,
   cached activations) from a new angle. You may execute it with Bash.
3. Write EXACTLY ONE concrete, falsifiable hypothesis for Phase S to
   {research_log_path}
   in this format:

     # Block {block_number:02d} — Research
     ## Theme: {theme}
     ## Hypothesis
     <one sentence stating a testable mechanism>
     ## Phase S function signature
     ```python
     def <name>(...) -> ...:
         \"\"\"<one-line description>
         PASS CONDITION: <exact measurable criterion, written NOW>
         \"\"\"
     ```
     ## Rationale
     <≤5 bullets tying the hypothesis to memory or literature>

## What NOT to do
- Do NOT edit any production source file.
- Do NOT propose anything on the falsified list below (they are closed).
- Do NOT propose "try A or B" — choose one.
- Do NOT exceed the time budget. If you hit it, write your best current
  hypothesis and stop — the harness will kill your process.

## Memory (prior blocks)
{memory_text}

## Falsified-hypotheses registry (do not retry)
{falsified_text}

## Literature allowlist (WebFetch permitted only for these URLs)
{urls}

Begin. Your tools this phase: Read, Grep, Glob, WebFetch (allowlist only),
Bash (to run the new analysis you optionally stage), Write/Edit scoped to
{analysis_block_dir}.
"""


# ---------------------------------------------------------------------------
# Phase S — Staging: function + pytest
# ---------------------------------------------------------------------------

def phase_s_prompt(
    block_number: int,
    theme: str,
    research_text: str,
    staging_block_dir: str,
    time_budget_sec: int,
) -> str:
    return f"""> **FOCUS**: this is Phase S. Follow the `## [Phase S]` section of the
> instruction prepended above, together with `## Shared context`. The other
> phase sections are reference-only — do not act on them.

You are running Phase S (Staging) of the code-change exploration loop.

BLOCK: {block_number:02d}
THEME: {theme}
TIME BUDGET: {time_budget_sec // 60} minutes HARD CAP.

## Your goal in this phase
Create EXACTLY TWO files under
  {staging_block_dir}
following the hypothesis from Phase R (pasted below):

1. The staged function module
     {staging_block_dir}/<name>.py
   with the exact signature and PASS CONDITION from Phase R.

2. A standalone test script
     {staging_block_dir}/test_<name>.py
   that exercises the function on the full flyvis dataset cache and
   prints on its last line:
     PASS: <one-line summary>   — on success, exit code 0
     FAIL: <one-line reason>    — on failure, exit code nonzero

The test is run as a plain Python subprocess (no pytest required). Use:

    from connectome_gnn.LLM_code.scratchpad import load_full_voltage
    v_clean, v_noisy = load_full_voltage('fly/flyvis_noise_free', 0.10)

to pull cached (clean, γ=0.10 noisy) voltage tensors of shape (T, N).
The loader handles data access; do not open zarr files directly.

## What NOT to do
- Do NOT edit ANY production source file. You may only Write/Edit under
  {staging_block_dir}.
- Do NOT require GPUs in the test; the harness runs tests on CPU.
- Do NOT skip the PASS condition. If your mechanism doesn't actually
  exhibit the hypothesised effect, print FAIL and exit. Do not fake it.
- Do NOT exceed the time budget.

## Hypothesis (from Phase R)
{research_text}

Begin. Your tools this phase: Read, Write/Edit under the staging dir, Bash
(to run your test once before declaring done).
"""


# ---------------------------------------------------------------------------
# Phase C — Wire-up: import the staged function into production
# ---------------------------------------------------------------------------

PHASE_C_WIRE_UP_ALLOW_LIST = (
    "src/connectome_gnn/models/regularizer.py",
    "src/connectome_gnn/models/recurrent_step.py",
    "src/connectome_gnn/models/graph_trainer.py",
    "src/connectome_gnn/models/neural_gnn.py",
)


def phase_c_prompt(
    block_number: int,
    theme: str,
    research_text: str,
    staging_block_dir: str,
    scratchpad_report: str,
    time_budget_sec: int,
) -> str:
    allow = "\n".join(f"  - {p}" for p in PHASE_C_WIRE_UP_ALLOW_LIST)
    return f"""> **FOCUS**: this is Phase C. Follow the `## [Phase C]` section of the
> instruction prepended above, together with `## Shared context` and
> `## [Phase-C hand-off → HPO]`. The other phase sections are reference-only.

You are running Phase C (Wire-up) of the code-change exploration loop.

BLOCK: {block_number:02d}
THEME: {theme}
TIME BUDGET: {time_budget_sec // 60} minutes HARD CAP.

## Your goal in this phase
Wire the staged function (which just passed its unit test) into the production
training pipeline with a MINIMAL edit — typically 3–10 lines across AT MOST
these files:

{allow}

For most blocks the right move is to add a new entry to the regularizer's
COMPONENTS list and a one-line call site in graph_trainer or recurrent_step.
You may expose a coefficient in the YAML config keyspace (`coeff_*`) so the
strength is tunable by later HPO iterations within the block.

## What NOT to do
- Do NOT edit any file not on the allow-list above. The pre-commit hook on
  this branch will reject commits touching other files.
- Do NOT copy the function body into production — import it from the
  staged module. The staged module is the source of truth.
- Do NOT add more than one new COMPONENT or one new call site. Keep the
  surface as small as possible.
- Do NOT remove existing regularizers or edit unrelated logic.

## Staged function + test (reference)
Directory: {staging_block_dir}

## Phase R hypothesis (goal)
{research_text}

## Phase S test result (PASSed — that's why you're here)
{scratchpad_report}

When done, print a short one-paragraph summary to stdout describing:
  (i)  which production file(s) you edited and how many lines,
  (ii) the YAML config key (if any) added for the coefficient,
  (iii) the expected effect on training.

The harness will git-add and git-commit your changes after this phase ends.

Begin. Your tools this phase: Read, Edit (files on the allow-list only),
Bash (read-only checks are fine).
"""
