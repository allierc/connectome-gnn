"""Block-start code session: orchestrates Phase R, Phase S, Phase C.

Each phase runs Claude as a subprocess via claude_cli_ext with a hard timeout.
Phase S only fires if Phase R wrote a research file. Phase C only fires if
Phase S's test runner saw at least one PASS:.

State update: on success the harness writes research_block_NN.md,
scratchpad_block_NN.md, and optionally commits Phase C's wire-up (via
git_checkpoint.commit_phase_c) on the caller's behalf.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from connectome_gnn.LLM_code.claude_cli_ext import run_claude_cli_with_timeout
from connectome_gnn.LLM_code.git_checkpoint import BlockCheckpoint, commit_phase_c
from connectome_gnn.LLM_code.prompts import (
    PHASE_C_WIRE_UP_ALLOW_LIST,
    phase_c_prompt,
    phase_r_prompt,
    phase_s_prompt,
)
from connectome_gnn.LLM_code.scratchpad import ScratchpadReport, run_tests
from connectome_gnn.LLM_code.state import CodeExplorationState


# Tool allow-lists per phase — passed to the claude CLI via --allowedTools.
PHASE_R_TOOLS = ("Read", "Grep", "Glob", "WebFetch", "Bash", "Write", "Edit")
PHASE_S_TOOLS = ("Read", "Write", "Edit", "Bash", "Glob", "Grep")
PHASE_C_TOOLS = ("Read", "Edit", "Bash", "Glob", "Grep")


@dataclass
class CodeSessionResult:
    phase_r_timed_out: bool
    phase_s_timed_out: bool
    phase_c_timed_out: bool
    phase_r_output_path: str
    phase_s_report_path: str
    any_test_passed: bool
    phase_c_committed: bool
    phase_c_sha: Optional[str]
    skip_reason: Optional[str] = None

    def as_markdown(self) -> str:
        lines = [
            "## Code session summary",
            "",
            f"- Phase R: {'TIMEOUT' if self.phase_r_timed_out else 'ok'}  "
            f"→ {self.phase_r_output_path}",
            f"- Phase S: {'TIMEOUT' if self.phase_s_timed_out else 'ok'}  "
            f"→ {self.phase_s_report_path}  (any PASS: {self.any_test_passed})",
            f"- Phase C: "
            + (
                "SKIPPED (" + self.skip_reason + ")"
                if self.skip_reason
                else ("TIMEOUT" if self.phase_c_timed_out else "ok")
            )
            + f"  (committed={self.phase_c_committed}, "
              f"sha={self.phase_c_sha[:12] if self.phase_c_sha else 'none'})",
        ]
        return "\n".join(lines) + "\n"


def _load_allowlist_urls(path: str) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    urls = data.get("urls", []) if isinstance(data, dict) else data
    return [u for u in urls if isinstance(u, str)]


def _read_safely(path: str, fallback: str = "") -> str:
    if not path or not os.path.isfile(path):
        return fallback
    with open(path) as f:
        return f.read()


def run_code_session(
    state: CodeExplorationState,
    block_number: int,
    checkpoint: BlockCheckpoint,
    instruction_text: str,
) -> CodeSessionResult:
    """Run the R/S/C triptych for one block. Returns a structured result.

    The caller is responsible for:
      - creating state.research_log_path / state.staging_block_dir etc.
      - persisting any output files the agent wrote
      - invoking verdict() after training (Phase T) is done
    """
    # Ensure output directories exist.
    os.makedirs(os.path.dirname(state.research_log_path) or ".", exist_ok=True)
    os.makedirs(state.staging_block_dir, exist_ok=True)
    os.makedirs(state.analysis_block_dir, exist_ok=True)

    theme = state.theme_for_block(block_number)
    caps = state.phase_time_limits
    memory_text = _read_safely(state.base.memory_path, "(memory file empty)")
    falsified_text = _read_safely(state.falsified_path, "(no falsified list yet)")
    allowlist = _load_allowlist_urls(state.literature_allowlist_path)

    # ---------------- Phase R ----------------
    r_prompt = instruction_text + "\n\n" + phase_r_prompt(
        block_number=block_number,
        theme=theme,
        memory_text=memory_text,
        falsified_text=falsified_text,
        allowlist_urls=allowlist,
        research_log_path=state.research_log_path,
        analysis_block_dir=state.analysis_block_dir,
        time_budget_sec=caps["R"],
    )
    r_out, r_timeout = run_claude_cli_with_timeout(
        prompt=r_prompt,
        root_dir=state.base.root_dir,
        allowed_tools=PHASE_R_TOOLS,
        timeout_sec=caps["R"],
        max_turns=80,
        log_prefix=f"[R{block_number:02d}] ",
    )
    # The agent is expected to have written research_log_path itself. If not,
    # fall back to dumping the session output.
    if not os.path.isfile(state.research_log_path):
        with open(state.research_log_path, "w") as f:
            f.write(
                f"# Block {block_number:02d} — Research (fallback dump, agent did not "
                f"write the expected file)\n\n"
                f"## Raw Phase R session output\n\n```\n{r_out[-4000:]}\n```\n"
            )

    # ---------------- Phase S ----------------
    research_text = _read_safely(state.research_log_path, "(empty)")
    s_prompt = instruction_text + "\n\n" + phase_s_prompt(
        block_number=block_number,
        theme=theme,
        research_text=research_text,
        staging_block_dir=state.staging_block_dir,
        time_budget_sec=caps["S"],
    )
    s_out, s_timeout = run_claude_cli_with_timeout(
        prompt=s_prompt,
        root_dir=state.base.root_dir,
        allowed_tools=PHASE_S_TOOLS,
        timeout_sec=caps["S"],
        max_turns=120,
        log_prefix=f"[S{block_number:02d}] ",
    )

    # Run the staged tests to see if any passed.
    scratch_report: ScratchpadReport = run_tests(
        staging_block_dir=state.staging_block_dir,
        repo_root=state.base.root_dir,
        block_number=block_number,
        per_test_timeout_sec=max(300, caps["S"] // 2),
    )
    with open(state.phase_s_report_path(), "w") as f:
        f.write(scratch_report.as_markdown())

    if not scratch_report.any_passed:
        return CodeSessionResult(
            phase_r_timed_out=r_timeout,
            phase_s_timed_out=s_timeout,
            phase_c_timed_out=False,
            phase_r_output_path=state.research_log_path,
            phase_s_report_path=state.phase_s_report_path(),
            any_test_passed=False,
            phase_c_committed=False,
            phase_c_sha=None,
            skip_reason="no PASS: in Phase S — wire-up blocked",
        )

    # ---------------- Phase C ----------------
    scratch_md = scratch_report.as_markdown()
    c_prompt = instruction_text + "\n\n" + phase_c_prompt(
        block_number=block_number,
        theme=theme,
        research_text=research_text,
        staging_block_dir=state.staging_block_dir,
        scratchpad_report=scratch_md,
        time_budget_sec=caps["C"],
    )
    c_out, c_timeout = run_claude_cli_with_timeout(
        prompt=c_prompt,
        root_dir=state.base.root_dir,
        allowed_tools=PHASE_C_TOOLS,
        timeout_sec=caps["C"],
        max_turns=60,
        log_prefix=f"[C{block_number:02d}] ",
    )

    # Commit whatever the agent edited (including the staging files).
    commit_body = (
        f"Block theme: {theme}\n\n"
        f"Phase C output (tail):\n\n{c_out[-1500:]}\n"
    )
    committed = commit_phase_c(checkpoint, message_body=commit_body)
    sha = checkpoint.phase_c_sha if committed else None

    return CodeSessionResult(
        phase_r_timed_out=r_timeout,
        phase_s_timed_out=s_timeout,
        phase_c_timed_out=c_timeout,
        phase_r_output_path=state.research_log_path,
        phase_s_report_path=state.phase_s_report_path(),
        any_test_passed=True,
        phase_c_committed=committed,
        phase_c_sha=sha,
    )
