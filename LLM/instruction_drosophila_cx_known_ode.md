# Drosophila CX — Known ODE Exploration

## Goal

Maximize **connectivity_R2** (PRIMARY) for the **Drosophila central complex ring attractor** using the **known_ode model** — the exact ground-truth ODE structure with learned parameters.

The known_ode model uses the **exact activation function**: `g_phi = exp(g) * softplus(v + b, beta=5)` and **exact dynamics**: `dv/dt = alpha * (-v + msg + I) / tau`. All parameters are **learned from data**: W (synaptic weights), tau (time constants), g (gain), bias (per-neuron bias in activation). This is an **upper bound** on what is achievable with perfect structural knowledge — unlike the GNN, the activation function form is given (not learned via MLP).

**Starting hypothesis**: "Known ODE with perfect structural knowledge achieves higher W R2 than GNN (GNN best: 0.804)"

Data is **re-generated each iteration** with a different seed to verify seed independence.

### Metrics (ranked by importance)

1. **connectivity_R2** (PRIMARY) — R² between learned W and ground-truth W (W is directly learned, so this is the most informative metric)
2. **rollout_pearson** (SECONDARY) — autoregressive rollout Pearson r on noise-free data
3. **tau_R2** (TERTIARY) — R² between learned tau and ground-truth tau (tau IS extractable for known_ode)

Informational: onestep_pearson, spectral_radius_learned vs spectral_radius_true, training_time_min.

**NOTE**: V_rest_R2 is always 0.0 (no resting potential). cluster_accuracy is not applicable (no learned embeddings).

## Scientific Method

Strict **hypothesize -> test -> validate/falsify** cycle:

1. **Hypothesize**: Form a specific, testable prediction
2. **Design experiment**: Change **EXACTLY ONE** parameter at a time to understand causality
3. **Run training**: 4 seeds — you cannot predict the outcome
4. **Analyze results**: Use metrics AND cross-seed variance
5. **Update understanding**: Revise hypotheses based on evidence

**CRITICAL**: You can only hypothesize. Only training results validate or falsify.

### CAUSALITY RULE (MANDATORY — READ THIS)

**If you change more than one parameter per slot, you CANNOT attribute the effect. This is a fatal experimental design error.**

- In EXPLORATION mode: Slot 0 = parent/baseline (unchanged control). Slots 1-3 each change **exactly one** parameter from the parent.
- Do NOT change parameters outside the current block focus.
- Do NOT skip the baseline — always keep one slot as an unchanged control.
- In ROBUSTNESS mode: all 4 slots use the same config (different seeds test robustness).

## Data Generation

Each slot re-generates data with a **different random seed**.
Seeds are **forced by the pipeline** — DO NOT modify them in config files.

- `simulation.seed = iteration * 1000 + slot`
- `training.seed = iteration * 1000 + slot + 500`

**DO NOT change `simulation:` parameters** except seed (managed automatically).

## CX Ring Attractor Model

```
dh/dt = alpha * (-h + exp(g_i) * softplus(h_j + b_j, beta=5) @ J^T + input) / tau_i
```

- **152 neurons**, 6 cell types (EPG, EPGt, PEN_a, PEN_b, Delta7, PEG), **9,722 GT edges**, **22,952 FC edges**
- tau bounded [0.2, 5.0], alpha=0.2, beta=5 (softplus sharpness)
- Pretrained teacher weights from hemibrain connectivity
- 10,000 frames, delta_t=0.1, bump + velocity stimuli
- Delta7 = inhibitory (negative W), others = excitatory (positive W)

## Known ODE Architecture

The model is registered as `drosophila_cx_known_ode`. Unlike the GNN:

- **No learned MLP curves**: The activation function `g_phi = exp(g) * softplus(v + b, beta=5)` is hardcoded with the exact ground-truth form. Parameters g (gain) and bias are learned per neuron.
- **No embeddings**: No per-neuron type embedding vectors.
- **Direct W learning**: Synaptic weight matrix W is learned directly on the graph edges.
- **Direct tau learning**: Time constants tau are learned per neuron.
- **Graph structure**: Uses message passing on edges (FC or GT), with per-edge weights W.

Key difference from GNN: structural knowledge is perfect, so the only challenge is parameter estimation.
Key difference from MLP: uses graph structure and per-edge weights, not a flat black-box.

**Parameters NOT used by known_ode** (do not modify): coeff_g_phi_diff, coeff_f_theta_msg_diff, coeff_g_phi_norm, coeff_g_phi_weight_L1, coeff_g_phi_weight_L2, coeff_f_theta_weight_L1, coeff_f_theta_weight_L2, embedding_dim, lr_embedding.

## Training Parameters

| Parameter                 | Default | Description                                            |
| ------------------------- | ------- | ------------------------------------------------------ |
| `lr_W`                    | 1e-3    | Learning rate for W (synaptic weights)                 |
| `lr`                      | 1e-3    | Learning rate for other params (tau, g, b)             |
| `n_epochs`                | 2       | Number of training epochs                              |
| `batch_size`              | 2       | Batch size                                             |
| `data_augmentation_loop`  | 100     | Data augmentation multiplier                           |
| `coeff_W_L1`              | 0       | L1 sparsity on W                                       |
| `coeff_W_L2`              | 0       | L2 penalty on W                                        |
| `coeff_W_sign`            | 0       | Dale's law penalty on W                                |
| `use_gt_edges`            | false   | Fully connected graph (22,952 edges)                   |
| `noise_model_level`       | 0.0     | No observation noise                                   |

## Training Time Constraint

**Target ~60 min per iteration.** Use `data_augmentation_loop` (DAL) to control training time. After each batch, check `training_time_min` in the metrics and adjust DAL for the next batch:

- If training_time_min < 40 min: **increase** DAL (e.g. multiply by 1.5-2x)
- If training_time_min > 70 min: **decrease** DAL (e.g. divide by 1.5-2x)
- DAL scales training time linearly — doubling DAL ~ doubles training time

Longer training gives W more time to converge. Always use the full time budget.

## Parallel Mode — 4 Slots Per Batch

Each batch runs 4 slots with different seeds (forced by pipeline). You choose the strategy:

- **Exploration** (default): Slot 0 = parent/control (unchanged). Slots 1-3 each change **exactly one** parameter. This gives 3 causal tests per batch.
- **Robustness test**: ALL 4 slots use the SAME config. The pipeline forces different seeds, so this measures seed robustness. Use this when a config looks promising.

State your choice (exploration vs robustness test) in the log entry.

### Robustness Assessment (when running same config across 4 slots)

- **Robust**: all 4 slots connectivity_R2 > 0.7
- **Partially robust**: 2-3 slots > 0.7
- **Fragile**: 0-1 slots > 0.7

## Block Partition

| Block | Focus                          | Parameters to scan                          | Ranges                                                                                                           |
| ----- | ------------------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1     | **lr_W + lr sweep**            | `lr_W`, `lr`                                | lr_W: {1e-4, 5e-4, 1e-3, 3e-3}, lr: {1e-4, 5e-4, 1e-3, 3e-3}. Known ODE has fewer params than GNN — may tolerate higher lr. |
| 2     | **Training volume**            | `data_augmentation_loop`, `n_epochs`        | DAL: {50, 100, 200, 500}, n_epochs: {2, 4, 8}. Fewer params to fit may need less data, or more data to resolve W from dynamics. |
| 3     | **W regularization**           | `coeff_W_L1`, `coeff_W_L2`, `coeff_W_sign` | W_L1: {0, 1e-6, 1e-5, 1e-4}, W_L2: {0, 1e-6, 1e-5, 1e-4}, W_sign: {0, 0.01, 0.1}. Known ODE learns W directly — regularization may help or hurt. |
| 4     | **Batch size**                 | `batch_size`                                | batch_size: {1, 2, 4, 8}. Smaller batches = noisier gradients but more updates. Larger batches = smoother, may help parameter estimation. |
| 5     | **Free exploration I**         | Any parameter                               | Consolidate best from blocks 1-4, test novel combinations                                                        |
| 6     | **Free exploration II**        | Any parameter                               | Continue ceiling-breaking attempts                                                                               |
| 7     | **Free exploration III**       | Any parameter                               | Continue ceiling-breaking attempts                                                                               |
| 8     | **Final robustness**           | None (robustness test)                      | 4-seed robustness test of best config from blocks 1-7                                                            |

### Known ODE context

- **Fewer parameters**: Known ODE has O(N^2 + 3N) parameters (W + tau + g + b) vs GNN which also has MLP weights. This means the optimization landscape is simpler but W must be recovered purely from dynamics fitting.
- **No identifiability issues**: Since the activation function is known exactly, there is no g_phi/W ambiguity — the learned W should directly correspond to ground-truth W.
- **Upper bound interpretation**: If known_ode achieves R2=X, then GNN cannot be expected to exceed X without additional inductive bias beyond the ODE structure.

## Iteration Workflow

### Step 1: Read Working Memory + User Input

### Step 2: Analyze Results (4 slots)

From `analysis.log`: connectivity_R2, rollout_pearson, tau_R2, training_time_min.

### Step 3: Write Log Entries + Update Memory

```
## Iter N: [robust/partially robust/fragile]
Node: id=N, parent=P
Hypothesis tested: "[quoted hypothesis]"
Config: lr_W=X, lr=Y, DAL=D, n_epochs=E, W_L1=A, W_L2=B, W_sign=C, batch_size=B
Slot 0: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 1: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 2: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Slot 3: conn_R2=A, rollout_pearson=B, tau_R2=C, sim_seed=S, train_seed=T
Seed stats: mean_conn_R2=X, std=Y, CV=Z%
Mutation: [param]: [old] -> [new]
W matrix: [visual comment from connectivity heatmap — sparsity, sign structure, convergence]
Verdict: [supported/falsified/inconclusive]
Next: parent=P
```

## Winner Config (COMPULSORY)

**At every block boundary**, you MUST save the current best config as a winner file.
This is a COMPULSORY task — do not skip it.

1. Identify the **best iteration** (highest connectivity_R2, or primary metric)
2. Copy its saved config from `log/Claude_exploration/LLM_<task_name>/config/iter_XXX_slot_YY.yaml`
3. Save it to `config/drosophila_cx/drosophila_cx_known_ode_winner.yaml` with a YAML comment header:

```yaml
# Winner config: drosophila_cx_known_ode_winner.yaml
# Source: iter_XXX_slot_YY (connectivity_R2 = X.XXX)
# Exploration: N iterations, M blocks
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - [1-2 sentence narrative: what made this config the best]
#   - [key hyperparameter choices and why they matter]
#
# Metrics:
#   connectivity_R2: X.XXX (best single seed)
#   robust_mean:     X.XXX +/- X.XXX (N seeds, CV=X.X%)
#   rollout_pearson: X.XXX
#   tau_R2:          X.XXX
#   spectral_radius: X.XXX (true: X.XXX)
#
# Key config differences from baseline:
#   - [list the parameters that differ from the initial baseline]
```

Destination: `config/drosophila_cx/drosophila_cx_known_ode_winner.yaml`

### Step 4: Acknowledge User Input

### Step 5: Formulate Next Hypothesis + Edit 4 Config Files

## Block Boundaries

1. Update "Paper Summary"
2. Summarize block findings
3. Update "Established Principles"
4. Clear "Current Block"
5. Carry forward best config

## Start Call

When prompt says `PARALLEL START`:

- Read base config — this IS the baseline. Do NOT change any default values.
- Slot 0 = baseline (no changes at all).
- Slots 1-3: each changes EXACTLY ONE parameter from the block focus.
- Hypothesis: "Known ODE with perfect structural knowledge achieves connectivity_R2 > 0.5 with default parameters"

---

# Working Memory Structure

```markdown
# Working Memory: drosophila_cx_known_ode

## Paper Summary (update at every block boundary)

- **Known ODE optimization**: [pending]
- **LLM-driven exploration**: [pending]

## Knowledge Base

### Robustness Comparison Table

| Iter | Config summary | conn_R2 (mean+-std) | CV% | rollout_r | tau_R2 | Robust? | Hypothesis |
| ---- | -------------- | ------------------- | --- | --------- | ------ | ------- | ---------- |

### Established Principles

### Falsified Hypotheses

### Open Questions

---

## Previous Block Summary

---

## Current Block

### Block Info

### Current Hypothesis

**Hypothesis**: [specific, testable prediction]
**Rationale**: [why]
**Test**: [what config change]
**Expected outcome**: [support vs falsify]
**Status**: untested / supported / falsified

### Iterations This Block

### Emerging Observations

**CRITICAL: This section must ALWAYS be at the END of memory file.**
```
