# Refactor plan — unify LLM_code instruction file into one phase-partitioned doc

## Context

The existing `GNN_LLM_code.py` + `src/connectome_gnn/LLM_code/` package
(implemented and committed on branch `agentic_code_change`, SHA `cd617fb`)
ended up with two overlapping instruction files:

1. `LLM/instruction_flyvis_noise_005_010_code_change.md` — **actually loaded**
   by both the HPO within-block analysis AND the R/S/C `code_session`, via
   `state.instruction_path` resolved at `LLM/pipeline.py:268-269`.
2. `src/connectome_gnn/LLM_code/instructions/instruction_flyvis_noise_005_010.md`
   — **orphan**, never read by any code path. Its existence is an artefact
   of the earlier plan; the `claude_code.instruction_file` YAML key that
   supposedly pointed at it is never consumed.

Both files redundantly restate the objective, baselines, ceilings and
falsified list — perfect setup for drift. This refactor collapses them into
a single source of truth, partitioned into explicit per-phase sections.

**Scope: refactor only.** No new infrastructure. The existing package, entry
point, YAML, verdict logic, git checkpoint, scratchpad, and per-phase prompt
templates all stay. The wiring already loads the right path; the file just
needs to contain the right content, and each phase prompt needs to tell the
agent which section to focus on.

## The unified file

Path: **`LLM/instruction_flyvis_noise_005_010_code_change.md`**  
(auto-resolved from base config name by `LLM/init_shared_files` —
no wiring change needed).

Sections in order:

```
# Exploration brief — flyvis_noise_005_010 code-change loop

## Shared context                               (every consumer reads this)
  - Objective: W R² > 0.82 with ≥3-seed stability
  - Baseline (RC winner 0.8023 ± 0.0014); ceilings (0.78 oracle, 0.965 physical)
  - Inverse-problem frame (v, e observed; W, τ, V_rest, types unknown;
    τ + V_rest via post-hoc linear fit on trained f_theta)
  - Falsified hypotheses (~10 items — do NOT retry)
  - Scientific method (one hypothesis per block; falsifiable; causal verdict)
  - Metric-key map:
      HPO analysis log   |  metrics.txt / verdict
      connectivity_R2    ↔  W_corrected_R2      (primary → W_R2)
      cluster_accuracy   ↔  clustering_accuracy
      tau_R2, V_rest_R2  — same in both
  - Block themes (fixed: denoising, recurrent, identifiability, best-of, robustness)
  - Staging conventions:
      mechanism + test  → src/connectome_gnn/LLM_code/staging/block_NN/
      analysis fn       → src/connectome_gnn/LLM_code/staging/block_NN/analysis/

## [Phase R] Research + optional analysis staging   (10 min cap)
## [Phase S] Staging — mechanism fn + pytest        (10 min cap)
## [Phase C] Wire-up                                (5 min cap)
    — includes: "MUST set a non-zero YAML default for the new coeff_<name>
       OR explicitly announce the key so the HPO section can seed it"
## [Phase V] Verdict                                (documentation only)
## [Phase-C hand-off → HPO]                         (protocol every consumer reads)
    - How HPO discovers this block's new mechanism:
        1. Read "Block NN code-session" in memory.md
        2. Read src/connectome_gnn/LLM_code/staging/block_NN/ (function body)
        3. Read Phase-C commit on agentic_code_change branch
        4. Diff slot YAML against flyvis_noise_005_010_rc_winner.yaml
    - First-batch directive:
        • Do NOT leave coeff_<name> at default 0 for a whole block — verdict
          will always REVERT because the mechanism never trained.
        • Batch 1 sweeps the new coefficient log-scale across 4 slots
          (e.g. 0.001, 0.01, 0.1, 1.0).
        • Batch 2+ refines around the best slot.
## [HPO within-block]                               (per-iteration analysis)
    - Scope: tune coefficients + existing levers; never touch architecture,
      batch_size, time_step, pretrained_model, or LLM_code/ itself.
    - CAUSALITY RULE: one parameter per slot; keep one control slot.
    - Safe-range table for existing levers.
    - Reiterated falsified list.
```

## Consumer map (no change to wiring)

| Consumer                   | Loads via                                     | Focus section             |
|----------------------------|-----------------------------------------------|---------------------------|
| R/S/C `code_session`       | `base_state.instruction_path` (existing)      | `[Phase R/S/C]` + Shared  |
| HPO `run_claude_analysis`  | same path, same loader (existing)             | `[HPO within-block]` + Shared + Hand-off |
| Human audit                | Read file directly                            | Whole doc                 |

## Changes to implement

### 1. Write the unified file
- Rewrite `LLM/instruction_flyvis_noise_005_010_code_change.md` with the
  structure above, merging in all relevant content from the orphan file.

### 2. Delete the orphan
- `rm src/connectome_gnn/LLM_code/instructions/instruction_flyvis_noise_005_010.md`
- `rmdir src/connectome_gnn/LLM_code/instructions/`

### 3. Add FOCUS headers to each phase prompt
- Edit `src/connectome_gnn/LLM_code/prompts.py`:
  - Prepend one line to each `phase_r_prompt` / `phase_s_prompt` /
    `phase_c_prompt` f-string output:
    ```
    > **FOCUS**: this is Phase {X}. Follow the `## [Phase {X}]` section of
    > the instruction prepended above, together with `## Shared context`.
    > The other phase sections are reference-only — do not act on them.
    ```
  - 3 tiny edits, ~3 lines each.

### 4. Clean up the YAML
- `config/fly/flyvis_noise_005_010_code_change.yaml`: remove the dead
  `claude_code.instruction_file` key (path is auto-discovered by LLM/ from
  the base config name).

### 5. Save this plan under the repo's docs/
- Copy this file to
  `docs/plan_llm_code_instruction_refactor.md`
  so future audits can see what changed and why. Do not overwrite the
  existing `docs/plan_adaptive_agent.md` (that's a different, earlier doc).

## Files that stay exactly as they are

- `GNN_LLM_code.py` — entry point unchanged.
- `src/connectome_gnn/LLM_code/{pipeline,code_session,verdict,scratchpad,git_checkpoint,claude_cli_ext,state}.py`
  — no wiring change needed; they already load from
  `base_state.instruction_path` which resolves to the LLM/ path.
- `src/connectome_gnn/LLM_code/literature/allowlist.json` — keep.
- `src/connectome_gnn/LLM_code/staging/` — keep (empty, ready for block_NN
  subdirs).

## Verification

```bash
cd /workspace/connectome-gnn

# 1. Orphan removed
test ! -d src/connectome_gnn/LLM_code/instructions

# 2. Unified file has every required section
for h in "## Shared context" \
         "## [Phase R]" "## [Phase S]" "## [Phase C]" "## [Phase V]" \
         "## [Phase-C hand-off" "## [HPO within-block]"; do
    grep -qF "$h" LLM/instruction_flyvis_noise_005_010_code_change.md \
        || { echo "MISSING: $h"; exit 1; }
done

# 3. Dead YAML key removed
! grep -q "instruction_file:" config/fly/flyvis_noise_005_010_code_change.yaml

# 4. FOCUS markers present in each phase prompt
python - <<'EOF'
import sys; sys.path.insert(0, 'src')
from connectome_gnn.LLM_code.prompts import phase_r_prompt, phase_s_prompt, phase_c_prompt
r = phase_r_prompt(1, 'denoising', 'm', 'f', [], 'rp', 'ab', 600)
s = phase_s_prompt(1, 'denoising', 'rt', 'sb', 600)
c = phase_c_prompt(1, 'denoising', 'rt', 'sb', 'sr', 300)
for name, out, tag in [('R', r, '[Phase R]'), ('S', s, '[Phase S]'), ('C', c, '[Phase C]')]:
    assert 'FOCUS' in out and tag in out, f"phase {name}: missing FOCUS or tag {tag}"
print("FOCUS markers OK")
EOF

# 5. Plan saved under docs/
test -f docs/plan_llm_code_instruction_refactor.md
```

All five checks must pass before committing the refactor.

## Critical files to modify (concise list)

- `LLM/instruction_flyvis_noise_005_010_code_change.md` — rewrite (unified).
- `src/connectome_gnn/LLM_code/prompts.py` — add 3× FOCUS headers.
- `config/fly/flyvis_noise_005_010_code_change.yaml` — remove 1 key.
- `docs/plan_llm_code_instruction_refactor.md` — new (copy of this plan).
- `src/connectome_gnn/LLM_code/instructions/*` — delete.

## Commit message

```
llm_code: unify instruction file with explicit phase partitions

- Merge the orphan LLM_code/instructions/*.md into the single file at
  LLM/instruction_flyvis_noise_005_010_code_change.md with named sections
  (Shared context, [Phase R/S/C/V], [Phase-C hand-off], [HPO within-block]).
- Delete LLM_code/instructions/ (never loaded by any code path).
- Add FOCUS markers to LLM_code/prompts.py so each phase prompt points the
  agent at its own section without re-loading a separate file.
- Remove the dead claude_code.instruction_file YAML key.
- Save refactor rationale to docs/plan_llm_code_instruction_refactor.md.
```
