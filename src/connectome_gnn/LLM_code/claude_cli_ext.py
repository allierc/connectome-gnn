"""Timeout-aware wrapper around the Claude CLI.

Same call shape as LLM/claude_cli.run_claude_cli, plus:
  - timeout_sec: hard wall-clock cap. On timeout we send SIGTERM and, if the
    process hasn't exited within TERM_GRACE seconds, SIGKILL.
  - returns a (output_text, timed_out) tuple so callers can mark the phase
    as "cap reached" without failing loudly.

Kept tiny (~60 LOC) and self-contained so the HPO pipeline is not affected.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from typing import Iterable, Optional, Tuple

TERM_GRACE = 10  # seconds between SIGTERM and SIGKILL
HEARTBEAT_SEC = 60  # print a "still alive" line if no output for this long


def _format_stream_event(ev: dict) -> Optional[str]:
    """Render one stream-json event as a short human-readable line.

    Returns None for events that don't need a line (e.g. init). Returned
    strings never end with a newline — the caller adds one.
    """
    t = ev.get("type")
    if t == "system":
        if ev.get("subtype") == "init":
            model = ev.get("model") or ev.get("session_id", "")
            return f"· session init ({model})"
        return None
    if t == "assistant":
        msg = ev.get("message") or {}
        parts = []
        for block in msg.get("content") or []:
            bt = block.get("type")
            if bt == "text":
                text = (block.get("text") or "").strip()
                if text:
                    parts.append(text)
            elif bt == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input") or {}
                # Compact input preview: show first useful field.
                preview = ""
                for k in ("file_path", "path", "pattern", "command", "url", "prompt"):
                    if k in inp and isinstance(inp[k], str):
                        preview = f" {k}={inp[k][:80]}"
                        break
                parts.append(f"→ {name}{preview}")
        return "\n".join(parts) if parts else None
    if t == "user":
        msg = ev.get("message") or {}
        for block in msg.get("content") or []:
            if block.get("type") == "tool_result":
                is_err = block.get("is_error")
                return "↪ tool_result" + (" (error)" if is_err else "")
        return None
    if t == "result":
        sub = ev.get("subtype", "")
        dur_ms = ev.get("duration_ms") or 0
        return f"· result ({sub}, {dur_ms / 1000:.1f}s)"
    return None


def _assistant_text(ev: dict) -> str:
    """Return just the assistant text content of an event (for return value)."""
    if ev.get("type") != "assistant":
        return ""
    out = []
    for block in (ev.get("message") or {}).get("content") or []:
        if block.get("type") == "text":
            text = block.get("text") or ""
            if text:
                out.append(text)
    return "\n".join(out)


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
        "--output-format", "stream-json",
        "--verbose",
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

    text_chunks = []           # accumulated assistant text (return value)
    started = time.time()
    deadline = started + timeout_sec
    last_output = started
    timed_out = False

    def _emit(line: str) -> None:
        nonlocal last_output
        last_output = time.time()
        elapsed = int(last_output - started)
        if log_prefix:
            print(f"{log_prefix}[{elapsed:4d}s] {line}", flush=True)
        else:
            print(f"[{elapsed:4d}s] {line}", flush=True)

    assert process.stdout is not None
    while True:
        now = time.time()
        if now >= deadline:
            timed_out = True
            break
        if now - last_output >= HEARTBEAT_SEC:
            _emit(f"... still running ({int(now - started)}s elapsed, "
                  f"{int(deadline - now)}s left)")
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.05)
            continue
        raw = line.rstrip("\n")
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            _emit(raw)
            continue
        rendered = _format_stream_event(ev)
        if rendered:
            for l in rendered.splitlines():
                _emit(l)
        text_chunks.append(_assistant_text(ev))

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
            for raw in remainder.splitlines():
                if not raw.strip():
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    _emit(raw)
                    continue
                rendered = _format_stream_event(ev)
                if rendered:
                    for l in rendered.splitlines():
                        _emit(l)
                text_chunks.append(_assistant_text(ev))
        except Exception:
            pass

    process.wait()
    return "\n".join(c for c in text_chunks if c), timed_out
