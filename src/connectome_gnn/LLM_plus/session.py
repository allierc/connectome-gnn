"""One Claude conversation pinned for a whole exploration.

The Claude CLI keeps conversations on disk, keyed by a UUID, under the project
entry for the working directory. Two flags reach them:

    claude -p "<prompt>" --session-id <uuid>    allocate this exact id
    claude -p "<prompt>" --resume    <uuid>     re-attach to it

So the first call of an exploration allocates a UUID we chose, and every later
call re-attaches to it. The `claude` process still starts once per batch — what
persists is the conversation, which is the thing that costs tokens to rebuild.
The alternative, holding one `claude` process open for the days an exploration
takes, buys nothing extra and loses everything if the loop dies.

What this is worth, concretely. Per batch the stock loop pays for a cold read
of the instruction file (~360 lines, ~6k tokens), the working memory (which
grows past 20k tokens by the last block) and the four per-slot logs, i.e. a
40-60k-token rebuild of context the agent had ten minutes earlier. Here the
instruction file is read at batch 0 and stays in the conversation, the agent
remembers the memory entries because it wrote them, and the batch prompt is
the four slot paths plus the block header.

Three events invalidate that and force a full re-read; all three are handled:

  1. A block boundary. The block table in the instruction file and the
     condensed memory both change meaning here, so re-read on purpose.
  2. A compaction. The CLI compacts when the conversation fills the context
     window; the oldest turns go first, and batch 0 — where the instruction
     file was read — is the oldest turn there is.
  3. A failed --resume (the conversation was deleted, or the loop moved to a
     different working directory). We allocate a new id and resend the full
     prompt rather than talk to an agent that knows nothing.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

from connectome_gnn.LLM_code.claude_cli_ext import run_claude_cli_with_timeout

# Substrings the CLI prints when --resume names a conversation it cannot find.
# Matched case-insensitively against the merged stdout/stderr of the call.
_RESUME_FAILURE_MARKERS = (
    "no conversation found",
    "no session found",
    "session not found",
    "could not resume",
    "failed to resume",
    "unable to resume",
)

SESSION_FILE = "claude_session.json"
SESSION_LOG = "claude_session.log"


@dataclass
class ClaudeSession:
    """The pinned conversation for one exploration, and its on-disk record.

    Attributes that decide what the next call looks like:
      session_id          the UUID both --session-id and --resume use
      attached            False until a call has actually created the
                          conversation; the first call uses --session-id, all
                          later ones --resume
      sent_full           what the prompt builder actually chose for THIS call,
                          set by prompts_plus just before run(). Separate from
                          needs_full_context, which is about the NEXT call: the
                          builder used to clear that flag to record its choice,
                          which made run() print "(incremental)" on exactly the
                          full-context calls — batch 0 and every block start
      needs_full_context  True when the next prompt must re-state the
                          instruction file and the memory file (set at
                          startup, after a compaction, and after a failed
                          --resume)
      fallback_prompt     the full-context version of the prompt about to be
                          sent, kept so a failed --resume can be retried on a
                          fresh conversation without losing the batch
    """

    exploration_dir: str
    root_dir: str
    base_config_name: str = ""
    session_id: str = ""
    n_calls: int = 0
    attached: bool = False
    needs_full_context: bool = True
    sent_full: bool = True
    fallback_prompt: str = ""
    last_compacted: bool = False
    last_mode: str = ""

    # Filled by the caller so the log line can name the batch. Not used for
    # any decision.
    current_label: str = "batch0"

    _sink: dict = field(default_factory=dict)

    # -- on-disk record ---------------------------------------------------

    @property
    def state_path(self) -> str:
        return os.path.join(self.exploration_dir, SESSION_FILE)

    @property
    def log_path(self) -> str:
        return os.path.join(self.exploration_dir, SESSION_LOG)

    def load_or_create(self, fresh: bool) -> "ClaudeSession":
        """Adopt the exploration's stored conversation, or start a new one.

        `fresh` (the loop's --fresh) always starts a new conversation: a fresh
        exploration erases the analysis and memory files, so resuming a
        conversation full of the erased run's conclusions would be worse than
        no conversation at all.
        """
        os.makedirs(self.exploration_dir, exist_ok=True)
        stored = None
        if not fresh and os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    stored = json.load(f)
            except Exception as exc:
                print(f"\033[93m  session record at {self.state_path} unreadable "
                      f"({type(exc).__name__}: {exc}); starting a new conversation\033[0m")
                stored = None

        if stored and stored.get("session_id"):
            self.session_id = stored["session_id"]
            self.n_calls = stored.get("n_calls", 0)
            self.attached = stored.get("attached", True)
            # We are re-entering from a new process: the conversation exists,
            # but we cannot see how much of it survived compaction, so pay for
            # one full-context call rather than guess.
            self.needs_full_context = True
            print(f"\033[94m  Claude session: resuming {self.session_id} "
                  f"({self.n_calls} calls so far)\033[0m")
        else:
            self.session_id = str(uuid.uuid4())
            self.n_calls = 0
            self.attached = False
            self.needs_full_context = True
            print(f"\033[94m  Claude session: new conversation {self.session_id}\033[0m")

        self.save()
        return self

    def save(self) -> None:
        record = {
            "session_id": self.session_id,
            "n_calls": self.n_calls,
            "attached": self.attached,
            "base_config": self.base_config_name,
            "root_dir": self.root_dir,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, self.state_path)

    def _log(self, line: str) -> None:
        with open(self.log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {line}\n")

    # -- the call ---------------------------------------------------------

    def cli_args(self) -> list:
        """The two flags that put this call in the pinned conversation."""
        if self.attached:
            return ["--resume", self.session_id]
        return ["--session-id", self.session_id]

    def run(
        self,
        prompt: str,
        root_dir: str,
        allowed_tools: Iterable[str],
        timeout_sec: int,
        max_turns: int = 200,
        log_prefix: str = "",
    ) -> Tuple[str, bool]:
        """Send one prompt inside the pinned conversation.

        Same signature and return value as run_claude_cli_with_timeout, so it
        can stand in for it wherever the pipeline calls it.
        """
        mode = "resume" if self.attached else "new"
        self._sink = {}
        print(f"\033[94m  [session {mode}] {self.session_id}"
              f"{'  (full context)' if self.sent_full else '  (incremental)'}\033[0m")

        output_text, timed_out = run_claude_cli_with_timeout(
            prompt, root_dir,
            allowed_tools=allowed_tools,
            timeout_sec=timeout_sec,
            max_turns=max_turns,
            log_prefix=log_prefix,
            extra_args=self.cli_args(),
            session_sink=self._sink,
        )

        if self.attached and self._resume_failed(self._sink, timed_out):
            print(f"\033[93m  --resume {self.session_id} failed; the conversation is "
                  f"gone. Starting a new one and resending the full prompt.\033[0m")
            self._log(f"{self.current_label}  RESUME FAILED for {self.session_id}")
            self.session_id = str(uuid.uuid4())
            self.attached = False
            self.needs_full_context = True
            retry_prompt = self.fallback_prompt or prompt
            self._sink = {}
            output_text, timed_out = run_claude_cli_with_timeout(
                retry_prompt, root_dir,
                allowed_tools=allowed_tools,
                timeout_sec=timeout_sec,
                max_turns=max_turns,
                log_prefix=log_prefix,
                extra_args=self.cli_args(),
                session_sink=self._sink,
            )
            mode = "new-after-failed-resume"

        reported = self._sink.get("session_id")
        if reported and reported != self.session_id:
            # The CLI allocated a different id than we asked for; follow it,
            # otherwise the next --resume points at nothing.
            print(f"\033[93m  CLI reported session {reported}, not {self.session_id}; "
                  f"following the CLI\033[0m")
            self.session_id = reported

        self.last_compacted = bool(self._sink.get("compacted"))
        self.attached = True
        self.n_calls += 1
        self.last_mode = mode
        # A compaction drops the oldest turns, which is where batch 0's read of
        # the instruction file lives — so the next call pays for it again.
        self.needs_full_context = self.last_compacted
        self.save()
        self._log(
            f"{self.current_label}  mode={mode}  id={self.session_id}  "
            f"call={self.n_calls}  compacted={self.last_compacted}  "
            f"timed_out={timed_out}  chars={len(output_text)}"
        )
        if self.last_compacted:
            print("\033[93m  conversation was compacted; next batch re-reads the "
                  "instruction and memory files\033[0m")
        return output_text, timed_out

    @staticmethod
    def _resume_failed(sink: dict, timed_out: bool) -> bool:
        """True when the call died because --resume named a missing conversation.

        The CLI does not raise here and does not put the reason in the
        assistant text: it prints `No conversation found with session ID: ...`
        as a plain line and emits a result event with subtype
        `error_during_execution`. So look at the plain lines the wrapper
        collected, not at the returned text — which is empty in this case.

        A timeout is never a resume failure: the agent was talking and we cut
        it off, so the conversation is alive and re-attaching to it is right.
        """
        if timed_out:
            return False
        blob = "\n".join(sink.get("plain_lines", [])).lower()
        if any(marker in blob for marker in _RESUME_FAILURE_MARKERS):
            return True
        # Belt and braces: an execution error with nothing said and no id ever
        # reported means the CLI never opened the conversation.
        return (sink.get("result_subtype") == "error_during_execution"
                and not sink.get("session_id"))


def install_session_hooks(pipeline_module, session: ClaudeSession, state) -> None:
    """Route the pipeline's two Claude phases through the pinned conversation.

    `pipeline.py` binds `run_claude_cli_with_timeout`, `batch_0_prompt` and
    `analysis_prompt` as module-level names at import time, so rebinding them
    on the module object redirects every call without editing the file. That
    keeps GNN_LLM.py — which imports the same module — behaving exactly as
    before, since it never calls this function.
    """
    from .prompts_plus import session_analysis_prompt, session_batch_0_prompt

    # The CLI files conversations per working directory, so a --resume issued
    # from a different cwd than the one that created the conversation finds
    # nothing. The pipeline always passes state.root_dir as cwd; if the session
    # was recorded against another root, say so now rather than at batch 12.
    if session.root_dir != state.root_dir:
        print(f"\033[93m  session was recorded under {session.root_dir} but the loop "
              f"runs in {state.root_dir}; --resume will not find it and a new "
              f"conversation will be started\033[0m")
        session.root_dir = state.root_dir

    def _run(prompt, root_dir, allowed_tools, timeout_sec,
             max_turns=200, log_prefix=""):
        return session.run(
            prompt, root_dir,
            allowed_tools=allowed_tools,
            timeout_sec=timeout_sec,
            max_turns=max_turns,
            log_prefix=log_prefix,
        )

    def _batch_0_prompt(st, slot_list, seed_info):
        session.current_label = "batch0"
        return session_batch_0_prompt(st, slot_list, seed_info, session)

    def _analysis_prompt(st, batch, slot_info, code_brief_context):
        session.current_label = f"batch{batch.batch_first}"
        return session_analysis_prompt(st, batch, slot_info, code_brief_context, session)

    pipeline_module.run_claude_cli_with_timeout = _run
    pipeline_module.batch_0_prompt = _batch_0_prompt
    pipeline_module.analysis_prompt = _analysis_prompt
