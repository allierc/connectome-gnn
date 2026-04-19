# Plan: Adaptive-Analysis Agent-in-the-Loop for Noisy GNN Training

**Date**: 2026-04-16  
**Problem**: `flyvis_noise_005_010` — combined dynamics noise 0.05 + measurement noise 0.10  
**Baseline**: connectivity_R2 = 0.739 (1-step), 0.80 (recurrent training, winner)  
**Target**: > 0.85 connectivity_R2  
**Key insight**: Hyperparameter tuning alone has reached a ceiling. The agent must now **write code**.  
**New capability needed**: When the agent writes new code (loss functions, training procedures), it must also **write its own analysis code** to evaluate whether the new mechanism is working — fixed-metric parsing is blind to new features.

---

## 1. Current Pipeline Limitations

### What the current agent can do
- Read YAML configs → propose next hyperparameter mutations
- At block boundaries (human-supervised): propose code changes → human approves → apply
- Analyze results via fixed metric patterns (`connectivity_R2=`, `tau_R2=`, etc.)
- Read connectivity heatmaps and activity kinographs

### What the current agent CANNOT do
- Understand whether a newly added training feature (e.g., a new loss term) is actually firing
- Compute derived diagnostics not present in the standard log (spectral analysis, gradient norms, noise estimation accuracy)
- Adapt the analysis to the specific code that was just written
- Execute exploratory analysis autonomously between batches

### The noise_005_010 problem
With derivative noise std ≈ 7.07, the MSE loss on per-frame targets is dominated by noise.
Techniques tried and their status:
- Wiener filter on targets → hurts connectivity R²
- Wavelet denoising on targets → hurts connectivity R²
- Recurrent training (time_step=5, batch_size=32) → best: 0.80 ✓
- dale_law=true → small improvement
- Reduced g_phi_L1 → +0.027 improvement

The 0.80 ceiling suggests that **hyperparameter tuning of the existing code cannot go further**.
Code changes to the loss function, training procedure, or model architecture are needed.

---

## 2. New Feature: Adaptive Analysis Step (Phase 5.5)

### Concept
After saving artifacts (Phase 4) and before the main analysis call (Phase 6), add a new phase where the agent:

1. **Writes a Python analysis script** tailored to the current block's hypothesis and any new code
2. **Executes the script** in a sandboxed subprocess
3. **Reads the output** (metrics, text diagnostics, saved figures)
4. **Includes output in the Phase 6 analysis prompt**

This closes the loop: the agent can now verify whether the new code it wrote is having the expected mechanistic effect.

### Key design decisions

**Autonomy**: The adaptive analysis runs automatically at every batch (not just block boundaries). No human approval needed — the script is read-only on model artifacts (`.pt`, `.log`, `.yaml` files). It writes only to a dedicated output directory.

**Safety**: The script cannot modify configs or model weights. It can only:
- Load model checkpoints (`torch.load`)
- Read log files
- Compute metrics and write results to `{exploration_dir}/adaptive_analysis/`
- Save diagnostic figures

**Adaptiveness**: At each new block, the agent generates a fresh analysis script (or updates the existing one). When new code features are added in code sessions, the brief for those features (`block_{N}_brief.md`) is given to the analysis generation prompt, so the agent knows what to check.

**Fail-safe**: If the script crashes, the pipeline logs the error and continues. The Phase 6 prompt includes either the script output or the error traceback (so the agent can debug it next batch).

---

## 3. Implementation Plan

### 3.1 New file: `src/connectome_gnn/LLM/adaptive_analysis.py`

```python
"""Adaptive analysis generation and execution for the LLM exploration loop.

The agent writes a custom Python analysis script each block,
executes it, and the output feeds into the Phase 6 analysis prompt.
"""

def generate_analysis_script(state, batch, code_brief_context) -> str:
    """Ask Claude to write a Python analysis script for the current block.
    
    Returns path to the generated script.
    """
    ...

def run_analysis_script(script_path, timeout_sec=120) -> tuple[str, bool]:
    """Execute the script in a subprocess. Returns (output_text, success)."""
    ...

def run_adaptive_analysis(state, batch, code_brief_context) -> str:
    """Full phase: generate script, execute, return output file path.
    
    Called between Phase 4 (save_artifacts) and Phase 6 (run_claude_analysis).
    Returns path to output file, or '' if analysis was skipped/failed.
    """
    ...
```

**Prompt for script generation** (inside `generate_analysis_script`):

```
You are generating a Python analysis script for batch {batch_first}-{batch_last}, block {block_number}.

Context:
- Working memory (current block hypothesis): {memory_path}
- Analysis log (all previous results): {analysis_path}
- Code changes this block: {code_brief_context}
- Available files per slot:
  - Model checkpoint: log/fly/{slot_name}/models/best_model_*.pt
  - Training log: {analysis_log_path[slot]}
  - Config YAML: {config_paths[slot]}
  - Activity kinograph: {activity_paths[slot]}

Write a Python script to {output_script_path} that:
1. Loads model checkpoints for each completed slot
2. Computes metrics that are NOT in the standard log — specifically ones that
   diagnose whether the current block hypothesis is supported or refuted
3. For code-change blocks: checks that the new code feature is actually active
   (e.g., if a new loss term was added, verify it is non-zero and has the expected magnitude)
4. Saves results as text to {output_results_path}
5. Optionally saves diagnostic figures to {output_figures_dir}

The script must import only standard libraries + torch + numpy + matplotlib.
Do NOT modify any config files or model checkpoints.
The script must complete within 120 seconds.
```

### 3.2 Modify `src/connectome_gnn/LLM/pipeline.py`

Add at line ~1090 (after `save_artifacts`, before `run_claude_analysis`):

```python
from .adaptive_analysis import run_adaptive_analysis

def run_adaptive_analysis_phase(state: ExplorationState, batch: BatchInfo) -> str:
    """Phase 5.5: Agent writes and executes its own analysis code."""
    print("\n\033[93mPHASE 5.5: Adaptive analysis (agent-generated script)\033[0m")
    code_brief_context = build_code_brief_context(state)
    return run_adaptive_analysis(state, batch, code_brief_context)
```

Modify `run_claude_analysis()` signature to accept `adaptive_analysis_context: str`:
- Add to analysis prompt: include the adaptive analysis output file path if non-empty

### 3.3 Modify `src/connectome_gnn/LLM/prompts.py`

Update `analysis_prompt()` to accept and include `adaptive_analysis_context`:

```python
def analysis_prompt(state, batch, slot_info, code_brief_context, 
                    adaptive_analysis_context="") -> str:
    adaptive_section = ""
    if adaptive_analysis_context:
        adaptive_section = f"""
Adaptive analysis results (agent-generated, read this first for mechanistic insight):
{adaptive_analysis_context}
These results were computed by a Python script you wrote for this block.
Use them to supplement the standard metrics — they may reveal why a config 
is succeeding or failing beyond what the metrics alone show.
"""
    return f"""...(existing prompt)...
{adaptive_section}
..."""
```

### 3.4 Modify `src/connectome_gnn/LLM/state.py`

Add to `ExplorationState`:
```python
adaptive_analysis_dir: str = ""   # {exploration_dir}/adaptive_analysis/
adaptive_analysis_enabled: bool = True
```

Add to `BatchInfo`:
```python
adaptive_analysis_output: str = ""  # path to results file for this batch
```

### 3.5 Modify `GNN_LLM.py`

After `save_artifacts(state, batch)` and before `run_claude_analysis(state, batch)`:

```python
# Phase 5.5: Adaptive analysis (agent writes and runs its own analysis code)
adaptive_context = ""
if state.interaction_code:  # only when code changes are possible
    adaptive_context = run_adaptive_analysis_phase(state, batch)

# Claude analysis + next mutations (now with adaptive analysis context)
run_claude_analysis(state, batch, adaptive_analysis_context=adaptive_context)
```

### 3.6 Modify `src/connectome_gnn/LLM/__init__.py`

Export `run_adaptive_analysis_phase`.

---

## 4. Script Generation Strategy

### What should the adaptive analysis script check?

The agent is instructed to write scripts that are **specific to the current block hypothesis**. Here are the key diagnostics by noise scenario:

#### For noise-robust loss functions (new code blocks):
```python
# Check if the new loss is firing
# Load log and check for new loss term values
# Verify gradient norms are reasonable (not exploding)
# Compute loss decomposition: standard_loss vs new_term
```

#### For any block:
```python
# 1. Spectral analysis of learned W
W = load_W_from_checkpoint(ckpt_path)
eigenvalues = torch.linalg.eigvals(W)
# Check: are eigenvalues inside unit circle (stable dynamics)?

# 2. Sign agreement with GT (Dale's law check)
sign_match_rate = (sign(W_learned) == sign(W_gt)).float().mean()

# 3. Noise amplification diagnosis
# If measurement_noise is high, check if |W_learned - W_gt| correlates with
# the noise-to-signal ratio per neuron type

# 4. Embedding quality
# Check if same-type neurons cluster in embedding space
# (before/after code change)
```

#### For recurrent training blocks:
```python
# Multi-step rollout consistency check
# Does the model's rollout stay stable for time_step steps?
# What is the divergence rate?
```

#### For new code features (checked against brief):
```python
# If brief says "added Huber loss":
#   - Verify Huber loss term is non-zero in log
#   - Check delta parameter value
#   - Compare loss curves: MSE slots vs Huber slots

# If brief says "added spectral regularization":
#   - Compute spectral norm of W for all slots
#   - Is it being suppressed relative to control?
```

### Script template (written to `adaptive_analysis/block_{N}_script.py`):

```python
"""Adaptive analysis script — Block {N}, Batch {first}-{last}.
Generated by Claude Code. DO NOT EDIT MANUALLY.
Hypothesis: {current_block_hypothesis}
"""
import sys, json, traceback
import torch, numpy as np

RESULTS = {}  # dict[str, any] — will be written to output file

def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)

def analyze_slot(slot_idx, ckpt_path, log_path):
    results = {}
    # ... slot-specific analysis ...
    return results

# Main
for slot in COMPLETED_SLOTS:
    try:
        results = analyze_slot(slot, CHECKPOINT_PATHS[slot], LOG_PATHS[slot])
        RESULTS[f'slot_{slot}'] = results
    except Exception as e:
        RESULTS[f'slot_{slot}'] = {'error': traceback.format_exc()}

# Write results
with open(OUTPUT_PATH, 'w') as f:
    json.dump(RESULTS, f, indent=2, default=str)
print(f"Adaptive analysis complete. Results: {OUTPUT_PATH}")
```

---

## 5. New Code-Change Explorations for Noise_005_010

The following code changes should be organized into blocks in the new instruction file.

### Block A: Robust Loss Functions

**Hypothesis**: MSE loss is dominated by noise spikes. Robust losses (Huber, Cauchy, log-cosh) downweight large residuals and focus on the signal.

Code changes needed in `graph_trainer.py` / `data_train_rollout.py`:
- Add `loss_type` config field: `mse` (default), `huber`, `cauchy`, `log_cosh`
- Huber: `F.huber_loss(pred, target, delta=huber_delta)`
- Cauchy: `torch.log(1 + (pred - target)**2 / c**2).mean()`
- Config fields: `loss_type: huber`, `huber_delta: 1.0`

**Adaptive analysis**: Verify that for noisy frames (high residual in MSE slot), the Huber/Cauchy loss is indeed lower (truncated). Compare training loss curves.

### Block B: Correlation/Phase Loss

**Hypothesis**: Under additive measurement noise, Pearson correlation between predicted and observed traces is a more noise-robust target than MSE (noise shifts the mean, MSE penalizes this; correlation ignores mean shifts).

Code changes needed:
- Add `loss_type: pearson` — computes `1 - corr(pred, target)` over the batch window
- Add `loss_type: cosine` — cosine similarity loss
- Config field: `pearson_window: 10` (number of frames per correlation window)

**Adaptive analysis**: Compute actual Pearson r on training data (per batch) for MSE vs Pearson loss slots. Does Pearson loss achieve higher training Pearson r?

### Block C: Noise-Weighted Loss

**Hypothesis**: Frames with high measurement noise contribute disproportionately large gradients. Weighting by estimated inverse noise variance focuses learning on reliable frames.

Code changes needed:
- Online noise variance estimator (exponential moving average of squared residuals per neuron)
- Loss weight: `w_i(t) = 1 / (sigma_i_estimated^2 + epsilon)`
- Config field: `noise_weighted_loss: true`, `noise_weight_ema: 0.99`

**Adaptive analysis**: Check if estimated noise sigma per neuron type correlates with ground-truth noise level. Verify that high-noise frames receive lower weights.

### Block D: Multi-Scale Temporal Loss

**Hypothesis**: Multi-step loss at MULTIPLE time horizons simultaneously gives richer gradient signal. Current recurrent training uses a single time_step=5. Using losses at steps {1,3,5} simultaneously may improve both short-term accuracy AND long-term consistency.

Code changes needed:
- Add `multi_scale_steps: [1, 3, 5]` list parameter
- Compute rollout to max step, take loss at each specified step
- Weight losses: `[0.5, 0.3, 0.2]` (configurable)
- Config field: `multi_scale_steps: [1, 3, 5]`, `multi_scale_weights: [0.5, 0.3, 0.2]`

**Adaptive analysis**: Compare single-scale (step=5) vs multi-scale. Plot per-step loss evolution across training.

### Block E: Spectral Regularization on W

**Hypothesis**: High measurement noise causes W to overfit to noise patterns (high-frequency structure). Penalizing the spectral norm or top-k singular values of W encourages smoother connectivity.

Code changes needed:
- Add spectral norm regularizer on W: `coeff_W_spectral * torch.linalg.matrix_norm(W, ord=2)`
- Or nuclear norm: `coeff_W_nuclear * torch.linalg.matrix_norm(W, ord='nuc')`
- Config fields: `coeff_W_spectral: 0.0`, `coeff_W_nuclear: 0.0`

**Adaptive analysis**: Track spectral norm of W across iterations. Compare singular value spectrum shape between GT and learned W.

### Block F: Gradient Clipping + Noise-Aware Scheduling

**Hypothesis**: Large gradient spikes from noisy frames destabilize training. Aggressive gradient clipping + adaptive LR reduction when gradient variance is high.

Code changes needed:
- `gradient_clip_norm: float` config field (currently not clipped, or not configurable)
- Gradient variance tracking: compute running variance of gradient norm
- When variance exceeds threshold, reduce effective LR for that step

**Adaptive analysis**: Track per-batch gradient norm distribution. Does clipping reduce outlier gradients?

### Block G: Denoising Pre-Processing in Training Loop

**Hypothesis**: Rather than preprocessing ALL data (which hurt W recovery), apply light temporal smoothing ONLY to the gradient targets during training, not to the input features.

Code changes needed:
- Add `target_smoothing_sigma: 0.0` (Gaussian σ in frames) parameter
- Apply 1D Gaussian smoothing to derivative targets before computing loss
- Keep input features (observed voltages) unsmoothed
- Config field: `target_smoothing_sigma: 0.5`

**Adaptive analysis**: Compare derivative target before/after smoothing. Check if smoothing reduces the high-frequency noise in targets while preserving low-frequency signal.

---

## 6. New Instruction File Structure

Create: `/workspace/connectome-gnn/LLM/instruction_flyvis_noise_005_010_code.md`

```markdown
# FlyVis GNN — Code-Change Exploration for Noise Robustness

## Goal
Push connectivity_R2 beyond the 0.80 ceiling reached by hyperparameter tuning.
This exploration focuses on CODE CHANGES: new loss functions, training procedures,
and regularizers designed to be robust to measurement noise 0.10.

## Starting Point
- Best config from previous exploration: config/fly/flyvis_noise_005_010_rc_winner.yaml
  (connectivity_R2: 0.80, recurrent_training: true, time_step: 5, batch_size: 32)
- Pretrained checkpoint: same as rc_winner

## Adaptive Analysis
You have access to an adaptive analysis step. At each batch, you will:
1. Write a Python script to analyze the current batch results mechanistically
2. The script will be executed and results fed back to you
3. Use these results to go BEYOND standard metrics

The adaptive analysis should verify that code changes are mechanistically working:
- New loss terms are non-zero and have expected magnitude
- Gradient behavior is as expected
- W spectral properties are improving

## Block Structure
[see Section 5 above — one block per code-change category]

## Explorable YAML Fields (new, from code changes)
[populated as code is written]

## Analysis Protocol (UPDATED)
1. Read standard metrics from analysis log
2. Read adaptive analysis results (mechanistic verification)
3. Write iteration entry with BOTH types of insight
4. Propose next config mutations AND (if needed) script improvements
```

---

## 7. Config Changes

### `flyvis_noise_005_010.yaml` (or new `flyvis_noise_005_010_code.yaml`)

Add to the `claude` section:
```yaml
claude:
  interaction_code: true
  case_study: "noise_robust_loss"
  case_study_brief: >
    Explore code changes to loss functions and training procedures to overcome
    the 0.80 connectivity_R2 ceiling in the flyvis_noise_005_010 case.
    Current best: recurrent_training=true, time_step=5, batch_size=32.
  adaptive_analysis: true   # NEW FLAG — enables Phase 5.5
  n_iter_block: 16
  n_parallel: 4
```

### New config key in `ExplorationState` (state.py)
```python
adaptive_analysis_enabled: bool = False  # read from claude.adaptive_analysis
```

---

## 8. Pipeline Flow (Updated)

```
setup → batch_0 → loop:
  [block_start] code_session (human-supervised, writes new code features)
  ↓
  load_configs_and_seeds
  ↓
  cluster_training + test_plot  (or local_pipeline)
  ↓
  save_artifacts
  ↓
  [NEW] run_adaptive_analysis_phase  ← AGENT WRITES + RUNS ANALYSIS SCRIPT
         ├── generate_analysis_script (Claude CLI, Write tool only)
         ├── execute script (subprocess, timeout=120s)
         └── capture output → adaptive_analysis/batch_{N}_results.json
  ↓
  run_claude_analysis  (now includes adaptive analysis context in prompt)
  ↓
  finalize_batch
```

---

## 9. Established Solutions from Noisy Time-Series Literature

Organized by applicability to the GNN training problem:

### High applicability (recommended for exploration)
| Technique | Why applicable | Block |
|-----------|---------------|-------|
| Huber/Cauchy loss | Suppress noise spikes in gradient | A |
| Pearson correlation loss | Invariant to additive noise mean shift | B |
| Multi-scale temporal loss | Richer gradient signal across time horizons | D |
| Noise-weighted loss | Focus learning on reliable frames | C |
| Gradient clipping (tuned) | Prevent noise spike destabilization | F |

### Medium applicability (secondary exploration)
| Technique | Why applicable | Notes |
|-----------|---------------|-------|
| Spectral regularization | Control W overfitting to noise | Block E |
| Target smoothing (light) | Reduce derivative noise | Block G |
| Curriculum learning | Start with low-noise samples | Requires data sorting |
| Co-training / ensemble | Multiple weak learners agree → reliable W | Too expensive |

### Already tried (status)
| Technique | Status |
|-----------|--------|
| Wiener filter on targets | ✗ hurts W |
| Wavelet denoising on targets | ✗ hurts W |
| Recurrent training (time_step=5) | ✓ best so far |
| Batch size increase to 32 | ✓ +0.027 W R² |
| Dale's law | ✓ small improvement |
| Remove g_phi_L1 | ✓ +0.027 W R² |

### Not applicable / too complex
| Technique | Why excluded |
|-----------|-------------|
| Kalman filter | Requires linear system (ODE is nonlinear) |
| Diffusion model denoising | Separate pre-processing step, too slow |
| Variational inference | Major architecture change, out of scope |
| Teacher-student | Requires noise-free teacher (not available) |

---

## 10. Implementation Sequence

**Week 1** (Days 1-3):
1. Implement `adaptive_analysis.py` with script generation and execution
2. Update `pipeline.py` to add Phase 5.5
3. Update `prompts.py` to include adaptive analysis context
4. Update `state.py`, `__init__.py`, `GNN_LLM.py`
5. Test with a dry-run (no cluster, local, 2 iterations)

**Week 1** (Days 4-5):
6. Write `instruction_flyvis_noise_005_010_code.md`
7. Implement Block A code changes (robust loss functions: Huber, Cauchy, log-cosh)
8. Run first 16-iteration block on cluster with adaptive analysis enabled

**Week 2**:
9. Based on Block A results, implement Block B or C code changes
10. Continue with subsequent blocks as the exploration progresses

---

## 11. Success Criteria

- Adaptive analysis script executes without errors at every batch
- The agent's log entries reference specific adaptive analysis findings (not just standard metrics)
- The agent autonomously identifies when a new code feature is/isn't working
- connectivity_R2 > 0.83 within 64 iterations of code-change exploration
- connectivity_R2 > 0.85 within 128 iterations

---

## 12. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Analysis script crashes → blocks pipeline | Wrap in try/except; continue without adaptive context if failed |
| Script takes too long → pipeline blocked | Hard timeout (120s) in subprocess call |
| Script modifies model checkpoints | Restrict write permissions in sandboxed subprocess (write only to adaptive_analysis/) |
| Agent writes bad code → crashes analysis | Include full traceback in next prompt so agent can fix it |
| Analysis script becomes stale across blocks | Regenerate script at each block start (overwrite old script) |
