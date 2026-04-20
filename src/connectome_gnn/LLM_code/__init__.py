"""LLM_code — agentic code-change exploration loop.

Sibling package to LLM/. The existing HPO-only loop (GNN_LLM.py + LLM/) is
unchanged. This package adds a block-scoped Research / Staging / Code-change /
Train / Verdict pipeline that lets the agent write new training mechanisms (not
just YAML mutations) under tight discipline:

  - Phase S mechanism functions are staged in LLM_code/staging/block_NN/ and
    must pass a pytest before being wired into production.
  - Phase C wire-ups live on the `agentic_code_change` branch; failed blocks
    are automatically git-reverted based on the Phase V multi-seed verdict.
  - Time-boxed per phase so training always happens.

See docs/plan_adaptive_agent.md (mirror of /home/node/.claude/plans/) and
src/connectome_gnn/LLM_code/instructions/ for the per-exploration briefs.
"""
