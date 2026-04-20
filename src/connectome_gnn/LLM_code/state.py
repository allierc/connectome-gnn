"""CodeExplorationState — shared state for the code-change pipeline.

Extends ExplorationState with the fields needed for Phase R/S/C/V:
  - research / staging / verdict directories (per block)
  - pre_block_baseline: per-metric seed lists from the previous block's runs,
    used by verdict.decide()
  - current block's git checkpoint (BlockCheckpoint from git_checkpoint)
  - phase time caps (R/S/C)
  - block_themes list (one entry per block)

Does NOT subclass ExplorationState because adding fields to a frozen @dataclass
is error-prone; instead we keep a .base attribute. Harness callers read
base.<field> for the shared fields and direct attrs for the new ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from connectome_gnn.LLM.state import ExplorationState
from connectome_gnn.LLM_code.git_checkpoint import BlockCheckpoint


# Default block themes — match the plan. Override via YAML claude_code.block_themes.
DEFAULT_BLOCK_THEMES = [
    "denoising",            # block 1
    "recurrent_schemes",    # block 2
    "identifiability",      # block 3
    "best_of_combination",  # block 4
    "robustness",           # block 5
]

# Default phase time caps in seconds (R / S / C), matching the plan.
DEFAULT_PHASE_TIME_LIMITS = {"R": 600, "S": 600, "C": 300}


@dataclass
class CodeExplorationState:
    """State for the code-change loop. Wraps the shared ExplorationState."""

    base: ExplorationState

    # Phase configuration
    block_themes: List[str] = field(default_factory=lambda: list(DEFAULT_BLOCK_THEMES))
    phase_time_limits: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_PHASE_TIME_LIMITS)
    )

    # Per-block directories (computed at block start)
    research_log_path: str = ""
    verdict_log_path: str = ""
    staging_block_dir: str = ""
    analysis_block_dir: str = ""
    code_diffs_dir: str = ""

    # Cross-block memory: last block's per-metric seed values
    #   pre_block_baseline["W_R2"] = [0.80, 0.81, ...]
    pre_block_baseline: Dict[str, List[float]] = field(default_factory=dict)

    # Current block's git state (None until start_block())
    current_checkpoint: Optional[BlockCheckpoint] = None

    # Literature allowlist file path (resolved at init)
    literature_allowlist_path: str = ""

    # Falsified-hypotheses registry (path to .md appended across blocks)
    falsified_path: str = ""

    def theme_for_block(self, block_number: int) -> str:
        """1-based block number → theme string. Falls back to 'open' past list."""
        idx = block_number - 1
        if 0 <= idx < len(self.block_themes):
            return self.block_themes[idx]
        return "open"

    def staging_block_path(self, block_number: int) -> str:
        """Absolute path to staging/block_NN/ under src/connectome_gnn/LLM_code/."""
        import os
        return os.path.join(
            self.base.root_dir,
            "src", "connectome_gnn", "LLM_code", "staging",
            f"block_{block_number:02d}",
        )

    # ---------------------------------------------------------------------
    # Block-scoped output-file helpers. All derive from the exploration_dir,
    # which the pipeline sets up once at exploration start.
    # ---------------------------------------------------------------------

    def _bdir(self) -> str:
        import os
        path = os.path.join(self.base.exploration_dir, "code_session")
        os.makedirs(path, exist_ok=True)
        return path

    def research_path_for(self, block_number: int) -> str:
        import os
        return os.path.join(self._bdir(), f"research_block_{block_number:02d}.md")

    def phase_s_report_path(self) -> str:
        """Convenience: derived from the current in-flight block via the
        currently-assigned staging_block_dir (ends with block_NN)."""
        import os
        name = os.path.basename(self.staging_block_dir.rstrip("/")) or "block_00"
        return os.path.join(self._bdir(), f"scratchpad_{name}.md")

    def verdict_path_for(self, block_number: int) -> str:
        import os
        return os.path.join(self._bdir(), f"verdict_block_{block_number:02d}.md")

    def code_diff_path_for(self, block_number: int) -> str:
        import os
        dd = os.path.join(self._bdir(), "code_diffs")
        os.makedirs(dd, exist_ok=True)
        return os.path.join(dd, f"block_{block_number:02d}.diff")
