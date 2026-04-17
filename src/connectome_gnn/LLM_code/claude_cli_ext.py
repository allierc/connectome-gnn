"""Timeout-aware wrapper around the Claude CLI.

Same call shape as LLM/claude_cli.run_claude_cli, plus:
  - timeout_sec: hard wall-clock cap. On timeout we send SIGTERM and, if the
    process hasn't exited within TERM_GRACE seconds, SIGKILL.
  - returns a (output_text, timed_out) tuple so callers can mark the phase
    as "cap reached" without failing loudly.

Kept tiny (~60 LOC) and self-contained so the HPO pipeline is not affected.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Iterable, Optional, Tuple

TERM_GRACE = 10  # seconds between SIGTERM and SIGKILL


def run_claude_cli_with_timeout(
    prompt: str,
    root_dir: str,
    allowed_tools: Iterable[str],
    timeout_sec: int,
    max_turns: int = 200,
    log_prefix: str = "",
) -> Tuple[str, bool]:
    """Run `claude -p <prompt>` with a hard timeout. Returns (stdout, timed_out).

    stdout is accumulated both for return and printed line-by-line with the
    given prefix for live monitoring.
    """
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "text",
        "--max-turns", str(max_turns),
        "--allowedTools",
        *list(allowed_tools),
    ]

    # Own session so we can SIGTERM the whole subtree.
    process = subprocess.Popen(
        cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    out_lines = []
    deadline = time.time() + timeout_sec
    timed_out = False

    # Non-blocking-ish loop: poll the pipe with a small sleep so we can check
    # the deadline. stdout is line-buffered (bufsize=1).
    assert process.stdout is not None
    while True:
        if time.time() >= deadline:
            timed_out = True
            break
        line = process.stdout.readline()
        if not line:
            # EOF or process exited.
            if process.poll() is not None:
                break
            time.sleep(0.05)
            continue
        if log_prefix:
            print(f"{log_prefix}{line}", end="", flush=True)
        else:
            print(line, end="", flush=True)
        out_lines.append(line)

    if timed_out:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        t0 = time.time()
        while time.time() - t0 < TERM_GRACE and process.poll() is None:
            time.sleep(0.1)
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Drain whatever the process emitted before it died.
        try:
            remainder = process.stdout.read() or ""
            if remainder:
                print(remainder, end="", flush=True)
                out_lines.append(remainder)
        except Exception:
            pass

    process.wait()
    return "".join(out_lines), timed_out
