"""Persistent-session variant of the LLM exploration loop.

The stock loop (GNN_LLM.py) starts a brand new Claude conversation for every
batch, so the agent re-reads the instruction file, the working memory and the
per-slot logs from an empty context roughly thirty times per exploration.
This package pins ONE conversation for the whole exploration and re-attaches
to it each batch, so the instruction file is read once and every batch prompt
carries only what is genuinely new.

Nothing here is imported by GNN_LLM.py — the stock loop is untouched.
"""

from .session import ClaudeSession, install_session_hooks
from .prompts_plus import session_analysis_prompt, session_batch_0_prompt

__all__ = [
    "ClaudeSession",
    "install_session_hooks",
    "session_analysis_prompt",
    "session_batch_0_prompt",
]
