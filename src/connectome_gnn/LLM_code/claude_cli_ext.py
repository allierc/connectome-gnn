"""Timeout-aware wrapper around the Claude CLI.

Same call shape as LLM/claude_cli.run_claude_cli, plus:
  - timeout_sec: hard wall-clock cap. On timeout we send SIGTERM and, if the
    process hasn't exited within TERM_GRACE seconds, SIGKILL.
  - returns a (output_text, timed_out) tuple so callers can mark the phase
    as "cap reached" without failing loudly.
  - extra_args: raw flags spliced into the command line before --allowedTools
    (which must stay last, being variadic). GNN_LLM+.py uses this to pass
    --session-id / --resume so one conversation spans the whole exploration.
  - session_sink: optional dict the wrapper fills in with what the stream
    reported about the conversation — 'session_id' from the init event and
    'compacted' True if the CLI compacted mid-call, which tells the caller
    the agent may have lost the instruction file it read at batch 0.

Kept tiny and self-contained so the HPO pipeline is not affected.
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
                if block.get("is_error"):
                    return "↪ tool_result (error)"
                return None  # suppress successful tool_results — too noisy
        return None
    if t == "result":
        sub = ev.get("subtype", "")
        dur_ms = ev.get("duration_ms") or 0
        return f"· result ({sub}, {dur_ms / 1000:.1f}s)"
    return None


def _record_session(ev: dict, sink: Optional[dict]) -> None:
    """Note conversation-level facts from a stream event into `sink`.

    Three facts matter to a caller that pins one conversation across many
    calls: the id the CLI actually used (it should equal the one we asked
    for); whether the CLI compacted this call, since a compaction drops the
    oldest turns and the instruction file read at batch 0 is the oldest turn
    there is; and how the run ended, because a --resume that names a deleted
    conversation fails with result subtype 'error_during_execution' and an
    empty assistant text rather than a raised exception.
    """
    if sink is None:
        return
    t = ev.get("type")
    subtype = ev.get("subtype")
    if t == "system":
        if subtype == "init" and ev.get("session_id"):
            sink["session_id"] = ev["session_id"]
        elif subtype in ("compact_boundary", "compaction", "compacted"):
            sink["compacted"] = True
    elif t == "result":
        sink["result_subtype"] = subtype or ""
        if ev.get("is_error"):
            sink["result_is_error"] = True


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
    extra_args: Iterable[str] = (),
    session_sink: Optional[dict] = None,
) -> Tuple[str, bool]:
    """Run `claude -p <prompt>` with a hard timeout. Returns (stdout, timed_out).

    stdout is accumulated both for return and printed line-by-line with the
    given prefix for live monitoring.

    extra_args go before --allowedTools, which is variadic and so must stay
    last or it would swallow them as further tool names.
    """
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(max_turns),
        *list(extra_args),
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
            # Not a stream event — the CLI's own diagnostics come out here,
            # including "No conversation found with session ID: ...". Keep the
            # last few so a caller can tell a failed --resume from a terse turn.
            if session_sink is not None:
                session_sink.setdefault("plain_lines", []).append(raw)
            _emit(raw)
            continue
        _record_session(ev, session_sink)
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
                _record_session(ev, session_sink)
                rendered = _format_stream_event(ev)
                if rendered:
                    for l in rendered.splitlines():
                        _emit(l)
                text_chunks.append(_assistant_text(ev))
        except Exception:
            pass

    process.wait()
    return "\n".join(c for c in text_chunks if c), timed_out
