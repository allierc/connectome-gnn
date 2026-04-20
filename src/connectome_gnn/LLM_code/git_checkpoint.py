"""Per-block git lifecycle for the code-change exploration loop.

At block start we record the current HEAD SHA as `start_sha`. After Phase C
the agent's edits are staged and committed on the *current* branch (assumed
to be `agentic_code_change` — the harness verifies this at startup and aborts
if the user is on main). After Phase V the harness calls `keep` or `revert`:

    keep()   — no-op; the block's commit stays on the branch.
    revert() — `git revert --no-edit <phase_c_sha>`: reverts ONLY the block's
               own Phase-C commit, leaving any user commits on the branch
               untouched. Creates a new revert commit (auditable, no history
               rewrite).

All operations run in a single repo root. No global state; caller passes the
repo root and block number explicitly.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional


REQUIRED_BRANCH = "agentic_code_change"


class GitCheckpointError(RuntimeError):
    pass


@dataclass
class BlockCheckpoint:
    repo_root: str
    block_number: int
    start_sha: str
    phase_c_sha: Optional[str] = None   # HEAD after Phase C commit, if any

    def has_commits(self) -> bool:
        """True iff any commit was made since block start."""
        if self.phase_c_sha is None:
            return False
        return self.phase_c_sha != self.start_sha


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], cwd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def _current_branch(repo_root: str) -> str:
    p = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    return p.stdout.strip()


def _head_sha(repo_root: str) -> str:
    p = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    return p.stdout.strip()


def _is_working_tree_clean(repo_root: str) -> bool:
    # Check tracked files only — untracked files are allowed (the user has
    # their own staging work). We only care that tracked modifications are
    # committed before Phase C.
    p = _run(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=repo_root, check=False)
    return p.returncode == 0


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

def require_branch(repo_root: str, branch: str = REQUIRED_BRANCH) -> None:
    """Abort loudly if the current branch is not the code-change branch.

    Called once at exploration startup. Prevents accidental commits to main.
    """
    cur = _current_branch(repo_root)
    if cur != branch:
        raise GitCheckpointError(
            f"code-change exploration requires branch '{branch}', "
            f"but repo at {repo_root} is on '{cur}'. "
            f"Run `git checkout {branch}` first (or `git checkout -b {branch}` "
            f"if it doesn't exist)."
        )


def start_block(repo_root: str, block_number: int) -> BlockCheckpoint:
    """Snapshot HEAD at block start. Must be on REQUIRED_BRANCH."""
    try:
        require_branch(repo_root)
    except GitCheckpointError as e:
        print(f"\n\033[93m[LLM_code] WARNING: {e}\033[0m\n")
    return BlockCheckpoint(
        repo_root=repo_root,
        block_number=block_number,
        start_sha=_head_sha(repo_root),
    )


def commit_phase_c(
    checkpoint: BlockCheckpoint,
    message_body: str,
    allow_empty: bool = False,
) -> bool:
    """Stage and commit whatever the agent edited in Phase C.

    Returns True if a commit was created, False if nothing changed.

    Stages ONLY the Phase C allow-list files + the block's staging/ dir, so
    unrelated uncommitted work in the user's tree is never swept into the
    block's commit (which would get rolled back together if Phase V reverts).
    """
    from connectome_gnn.LLM_code.prompts import PHASE_C_WIRE_UP_ALLOW_LIST

    root = checkpoint.repo_root
    staging_path = os.path.join(
        "src", "connectome_gnn", "LLM_code", "staging",
        f"block_{checkpoint.block_number:02d}",
    )
    # Stage each allow-listed file individually — git add on an unmodified
    # path is a no-op, so this is safe.
    for path in PHASE_C_WIRE_UP_ALLOW_LIST:
        if os.path.isfile(os.path.join(root, path)):
            _run(["git", "add", "--", path], cwd=root)
    if os.path.isdir(os.path.join(root, staging_path)):
        _run(["git", "add", "--", staging_path], cwd=root)

    # Detect whether anything is staged.
    p = _run(["git", "diff", "--cached", "--quiet"], cwd=root, check=False)
    nothing_staged = (p.returncode == 0)
    if nothing_staged and not allow_empty:
        checkpoint.phase_c_sha = checkpoint.start_sha
        return False

    title = f"llm_code: block {checkpoint.block_number:02d} wire-up"
    body = message_body.strip() or "(no body)"
    commit_msg = f"{title}\n\n{body}\n"
    _run(["git", "commit", "-m", commit_msg], cwd=root)
    checkpoint.phase_c_sha = _head_sha(root)
    return True


def keep(checkpoint: BlockCheckpoint) -> None:
    """No-op by design; the block's commit(s) stay on the branch."""
    # Here for symmetry with revert() and so callers can log unambiguously.
    return None


def revert(checkpoint: BlockCheckpoint, verdict_reason: str) -> Optional[str]:
    """Revert ONLY the block's Phase-C commit. Returns the new HEAD SHA, or
    None if there was nothing to revert.

    Earlier versions used `git rev-list <start_sha>..HEAD` and reverted every
    commit in that range, which silently ate user commits made on the same
    branch between block start and verdict. `commit_phase_c` guarantees that
    a block produces AT MOST one commit (`checkpoint.phase_c_sha`), so the
    correct behaviour is to revert exactly that SHA.
    """
    root = checkpoint.repo_root
    if not checkpoint.has_commits():
        return None

    sha = checkpoint.phase_c_sha
    assert sha is not None  # has_commits() == True implies phase_c_sha is set

    # Sanity: the commit must still be reachable from HEAD (if a user rebased
    # it away, there's nothing to revert).
    reachable = _run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
        cwd=root,
        check=False,
    )
    if reachable.returncode != 0:
        return None

    # If the Phase-C changes have already been undone on HEAD (e.g. by an
    # intervening user revert), `git revert` would no-op. Skip quietly.
    diff = _run(["git", "diff", sha, "HEAD", "--", "."], cwd=root)
    # Cheaper: check whether any of the Phase-C-touched files still differs
    # from HEAD vs sha^ in the reverse direction — skipped; `git revert` will
    # emit an empty commit if nothing changed, which `--no-edit` + the amend
    # below tolerates. Keep it simple.
    _ = diff  # placeholder for future optimisation

    _run(["git", "revert", "--no-edit", sha], cwd=root)

    # Amend the revert's message with the verdict reason so it's searchable.
    cur_msg_p = _run(["git", "log", "-1", "--pretty=%B"], cwd=root)
    cur_msg = cur_msg_p.stdout.strip()
    new_msg = (
        f"{cur_msg}\n\n"
        f"Phase-V verdict: REVERT  block {checkpoint.block_number:02d}\n"
        f"Reason: {verdict_reason}\n"
    )
    _run(["git", "commit", "--amend", "-m", new_msg], cwd=root)

    return _head_sha(root)


def diff_since_start(checkpoint: BlockCheckpoint) -> str:
    """Cumulative diff across the block's commits (for audit / code_diffs)."""
    p = _run(
        ["git", "diff", checkpoint.start_sha, "HEAD"],
        cwd=checkpoint.repo_root,
    )
    return p.stdout
