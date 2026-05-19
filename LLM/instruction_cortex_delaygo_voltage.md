# Cortex Delaygo Voltage — GNN Inverse-Problem Exploration

## Goal

Find the **best training scheme** for a `NeuralGNN` to recover the
teacher `TaskRNN`'s ground-truth `W_rec`, `tau`, and `V_rest` from the
saved cortex_delaygo voltage dataset. The teacher is a 256-unit,
free-W, tanh TaskRNN trained on cortex/delaygo; its `W_rec` (~65 280
non-zero edges, fully connected minus diagonal), `tau` (= 0.1 shared
across units), and `V_rest` (= teacher's recurrent bias `b`) are saved
in `ode_params.pt` and serve as the recovery target.

**Primary metric**: `conn_R²` — coefficient of determination on per-edge
weights between the GNN's learned `W` and the teacher's ground-truth
`W` (FlyVis-style stimuli_R² formulation, see
`models/graph_tester.py`). A trained baseline should reach 0.7+ within
a few epochs; near-perfect recovery (≥0.95) is the target.

**Secondary metrics**:

- `tau_R²` — coefficient of determination on per-neuron tau (recovered
  from f_theta slope; see `derive_tau` in metrics.py).
- `V_rest_R²` — coefficient of determination on per-neuron V_rest
  (recovered from f_theta slope/offset).
- `rollout_pearson_r` (when computed) — Pearson correlation between the
  GNN's rolled-out voltage trajectory and ground truth.

The dataset is **fixed**: 64 000 train + 16 000 test frames at dt = 0.02 s
on the 256-unit cortex_delaygo voltage data. Only the **training
hyperparameters** change between iterations.

## What's known going in

A first attempt with `w_init_mode: zeros` produced `conn_R² ≈ -3` and
nan recovery for tau / V_rest. The diagnosis: zero W-init starves the
gradient signal through V_rest and tau, so they never escape
initialisation. Switching to `randn_scaled` (scale 0.5) and adopting
signal_fig_2-style learning rates (`lr=5e-4`, `lr_W=1e-3`,
`lr_embedding=7.5e-4`).

## Available hyperparameters (the search space)

The agent may set these fields per-slot in `training:` / `graph_model:`.

### Learning rates (PRIORITY)

| Field          | Default  | Sweep values                   | Why                                                                  |
| -------------- | -------- | ------------------------------ | -------------------------------------------------------------------- |
| `lr`           | `5e-4`   | {1e-4, 3e-4, 5e-4, 1e-3, 3e-3} | Main MLP optimizer step. Signal_fig_2 found 5e-4 works on 1000-unit. |
| `lr_W`         | `1e-3`   | {3e-4, 1e-3, 3e-3, 1e-2}       | W edge weights have their own LR — usually higher than `lr`.         |
| `lr_embedding` | `7.5e-4` | {1e-4, 3e-4, 7.5e-4, 1.5e-3}   | Per-neuron embedding LR; controls how fast latent types emerge.      |

### Regularization

#### W (edge weights)

| Field          | Default  | Sweep values          | What it does                                        |
| -------------- | -------- | --------------------- | --------------------------------------------------- |
| `coeff_W_L1`   | `0`      | {0, 1e-6, 1e-5, 1e-4} | Sparsity penalty on edge weights.                   |
| `coeff_W_L2`   | `1.5e-6` | {0, 1e-7, 1e-6, 1e-5} | Weight decay on edges.                              |
| `coeff_W_sign` | `0`      | {0, 1e-5, 1e-4, 1e-3} | Forces W signs to stay consistent (E/I separation). |

#### f_theta (per-neuron update MLP)

| Field                     | Default | Sweep values          | What it does                                                                                |
| ------------------------- | ------- | --------------------- | ------------------------------------------------------------------------------------------- |
| `coeff_f_theta_diff`      | `0`     | {0, 1e-4, 1e-3, 1e-2} | **Monotonicity (leak)**: penalises positive df/dv. Enforces leak term `df/dv < 0`.          |
| `coeff_f_theta_msg_diff`  | `0`     | {0, 1e-4, 1e-3, 1e-2} | **Monotonicity vs msg**: df/d(msg) monotonic — encodes that bigger input → bigger response. |
| `coeff_f_theta_weight_L1` | `0`     | {0, 1e-6, 1e-5}       | L1 on f_theta MLP weights.                                                                  |
| `coeff_f_theta_weight_L2` | `0`     | {0, 1e-6, 1e-5}       | L2 on f_theta MLP weights.                                                                  |

#### g_phi (per-edge message MLP)

| Field                   | Default | Sweep values          | What it does                                                      |
| ----------------------- | ------- | --------------------- | ----------------------------------------------------------------- |
| `coeff_g_phi_diff`      | `0`     | {0, 1e-4, 1e-3, 1e-2} | Variance penalty on g_phi outputs across edges.                   |
| `coeff_g_phi_norm`      | `0`     | {0, 1e-4, 1e-3, 1e-2} | Norm penalty on g_phi edge messages (controls message magnitude). |
| `coeff_g_phi_weight_L1` | `0`     | {0, 1e-6, 1e-5}       | L1 on g_phi MLP weights.                                          |
| `coeff_g_phi_weight_L2` | `0`     | {0, 1e-6, 1e-5}       | L2 on g_phi MLP weights.                                          |

#### Embeddings & biases

| Field                     | Default | Sweep values          | What it does                                                                                 |
| ------------------------- | ------- | --------------------- | -------------------------------------------------------------------------------------------- |
| `coeff_model_a`           | `0`     | {0, 1e-4, 1e-3, 1e-2} | L2 on per-neuron embedding `a`. Pulls embeddings toward 0 (helps when clustering struggles). |
| `coeff_model_b`           | `0`     | {0, 1e-4, 1e-3, 1e-2} | L2 on per-neuron bias `b`. Restrains drift of recovered V_rest.                              |
| `coeff_embedding_cluster` | `0`     | {0, 1e-3, 1e-2, 1e-1} | Pulls same-cluster embeddings toward their centroid (only meaningful when sparsity ≠ none).  |

### Batch / iteration budget

| Field                    | Default | Sweep values          | Notes                            |
| ------------------------ | ------- | --------------------- | -------------------------------- |
| `batch_size`             | `8`     | {1, 4, 8, 16, 32, 64} | Per-batch frame count.           |
| `data_augmentation_loop` | `20`    | {10, 20, 50, 100}     | Multiplies per-epoch iter count. |

### Architecture (mutate one knob at a time)

| Field               | Default | Sweep values   | Notes                                                             |
| ------------------- | ------- | -------------- | ----------------------------------------------------------------- |
| `hidden_dim`        | `128`   | {64, 128, 256} | Width of `f_theta` and `g_phi` MLP hidden layers.                 |
| `n_layers`          | `3`     | {2, 3, 4}      | Depth of `f_theta` / `g_phi` MLPs.                                |
| `hidden_dim_update` | `128`   | {64, 128, 256} | Width of the update MLP (the per-edge / per-node update).         |
| `n_layers_update`   | `3`     | {2, 3, 4}      | Depth of the update MLP.                                          |
| `embedding_dim`     | `2`     | {2, 3, 4, 8}   | Per-neuron embedding dim. Sparsity clustering acts in this space. |
| `input_size_update` | `5`     | {3, 5, 7}      | Per-edge feature count fed into the update MLP.                   |

### W init

| Field          | Default        | Sweep values                                 | Notes                                            |
| -------------- | -------------- | -------------------------------------------- | ------------------------------------------------ |
| `w_init_mode`  | `randn_scaled` | {zeros, randn, randn_scaled, uniform_scaled} | `zeros` killed Vrest/tau — keep at randn_scaled. |
| `w_init_scale` | `0.5`          | {0.1, 0.25, 0.5, 1.0}                        | Signal_fig_2 flyvis-winner uses ~0.25.           |

### Things you must NOT change

- `dataset` (locked to `cortex_delaygo_voltage`).
- `simulation.task_model_config_path` (the teacher reference — locked).
- `simulation.n_neurons`, `n_frames`, `delta_t` (data shape).
- `graph_model.signal_model_name` (must stay `cortex` so it routes to NeuralGNN).
- `graph_model.aggr_type` (must stay `add`).
- `graph_model.prediction` (must stay `first_derivative`).
- `task` (irrelevant for voltage training; not present in this config).

## Metrics (per slot, per iteration)

Read from `<exploration_dir>/<slot_name>_analysis.log` after each batch.
The trainer also writes a per-iter `tmp_training/metrics.log` you can
tail during a run.

| Metric              | What it measures                                                                        | Target    |
| ------------------- | --------------------------------------------------------------------------------------- | --------- |
| `conn_R²` (primary) | Per-edge W recovery: `1 − SS_res/SS_tot` between learned `W` and `ode_params.pt`'s `W`. | **≥ 0.9** |
| `tau_R²`            | Per-neuron tau recovery. `nan(0%)` means f_theta never learned a slope.                 | **≥ 0.7** |
| `V_rest_R²`         | Per-neuron V_rest recovery. `nan(0%)` for the same reason.                              | **≥ 0.7** |
| `rollout_pearson_r` | Voltage trajectory Pearson r vs GT (when test phase runs).                              | **≥ 0.8** |

`nan(0%)` on either tau_R² or V_rest_R² is the canonical signal that
f_theta's slope did not emerge — typically caused by zero W init,
too-low `lr_embedding`, or `sparsity_freq` too aggressive early.

## CAUSALITY RULE (MANDATORY)

You can change **one or two** parameters per slot. Slot 0 is always the
**parent / control** (no mutation vs the previous winner). Slots 1–7
each test a single-axis mutation so the effect is attributable.

In **robustness mode** (every slot identical, different seeds), the
pipeline forces 8 different seeds; this measures the seed sensitivity
of a candidate winner.

## Block plan

8 slots/batch. Iterations: 320 total = 8 blocks × 40 iter/block = 5 batches/block.

| Block | Focus                              | Knobs to scan                                                                                                                                                                                                                                                                               | Why                                                                                                                                                        |
| ----- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | **Baseline + robustness**          | 8 seeds, no mutations.                                                                                                                                                                                                                                                                      | Establish the conn_R² floor at the signal_fig_2-style defaults. Detect noisy slots.                                                                        |
| 2     | **Learning rates**                 | `lr` ∈ {1e-4, 3e-4, 1e-3}; `lr_W` ∈ {3e-4, 3e-3}; `lr_embedding` ∈ {1e-4, 1.5e-3}.                                                                                                                                                                                                          | Most impactful axis for recovery speed and ceiling.                                                                                                        |
| 3     | **W regularisation**               | `coeff_W_L1` ∈ {0, 1e-6, 1e-5, 1e-4}; `coeff_W_L2` ∈ {0, 1e-7, 1e-5}; `coeff_W_sign` ∈ {0, 1e-5, 1e-4}.                                                                                                                                                                                     | Push W toward sparse / consistent-sign solutions; tighten the tail of bad edges.                                                                           |
| 4     | **f_theta / g_phi regularisation** | f_theta: `coeff_f_theta_diff`, `coeff_f_theta_msg_diff`, `coeff_f_theta_zero`, `coeff_f_theta_linearity`, `coeff_f_theta_weight_L1/L2`; g_phi: `coeff_g_phi_diff`, `coeff_g_phi_norm`, `coeff_g_phi_weight_L1/L2`; embeddings: `coeff_model_a`, `coeff_model_b`, `coeff_embedding_cluster`. | Function-shape priors (leak, monotonicity, linearity, zero-input), MLP-weight decay, embedding/bias L2 — direct levers on V_rest / tau / cluster recovery. |
| 5     | **MLP capacity**                   | `hidden_dim` ∈ {64, 256}; `n_layers` ∈ {2, 4}; `hidden_dim_update` ∈ {64, 256}; `embedding_dim` ∈ {3, 4, 8}.                                                                                                                                                                                | Architecture × inverse-problem trade-off.                                                                                                                  |
| 6     | **Batch / DAL**                    | `batch_size` ∈ {1, 4, 16, 32, 64}; `data_augmentation_loop` ∈ {10, 50, 100}.                                                                                                                                                                                                                | Trade-off: small batch = noisy grad but more updates / epoch. DAL multiplies iters.                                                                        |
| 7     | **Sparsity / cluster recovery**    | `sparsity` ∈ {none, replace_embedding}; `sparsity_freq` ∈ {2, 8, 16}; `cluster_distance_threshold` ∈ {0.05, 0.2, 0.5}.                                                                                                                                                                      | The agentic loop's most distinctive lever; tunes when/how clustering kicks in.                                                                             |
| 8     | **Final robustness**               | 8 seeds of the combined winner config from blocks 1–7.                                                                                                                                                                                                                                      | Confirm winner is seed-robust at full budget.                                                                                                              |

## Mutation log format (per iteration)

After each batch, append to working memory:

```
## Iter N (block B): [exploration | robustness]
Parent: iter_M_slot_K  (conn_R²=X.XXX  tau_R²=Y.YYY  V_rest_R²=Z.ZZZ)
Hypothesis: "[testable claim about what the mutation should do]"
Slot 0: [parent/control]   conn_R²=X.XXX  tau_R²=Y.YYY  V_rest_R²=Z.ZZZ  loss=L.LLL
Slot 1: [knob -> value]    conn_R²=X.XXX  tau_R²=Y.YYY  V_rest_R²=Z.ZZZ  loss=L.LLL
...
Slot 7: [knob -> value]    conn_R²=X.XXX  tau_R²=Y.YYY  V_rest_R²=Z.ZZZ  loss=L.LLL
Best slot: K  ->  conn_R²=X.XXX
Verdict: [supported | falsified | inconclusive]
Next parent: iter_N_slot_K
```

If `tau_R²` or `V_rest_R²` shows `nan(0%)` on the winning slot, that's a
red flag even if `conn_R²` is high — favour configurations that recover
both W and the per-neuron parameters cleanly.

## Winner config

At every block boundary, copy the best slot's config to
`config/cortex/cortex_delaygo_voltage_winner.yaml` with header:

```yaml
# Winner: cortex_delaygo_voltage_winner.yaml
# Source: iter_NNN_slot_KK  (conn_R² = X.XXX, tau_R² = Y.YYY, V_rest_R² = Z.ZZZ)
# Block: B  (focus: <focus>)
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - <one-sentence reason>
#   - <key knob change vs prior winner>
#
# Per-epoch trajectory:
#   e1: conn_R²=X.XXX  tau_R²=Y.YYY  V_rest_R²=Z.ZZZ
#   e2: ...
#   e3: ...
```

## Notes / hints

- **Watch for `nan(0%)` on tau / V_rest** — the canonical sign that
  f_theta never learned a slope. Almost always traces to:
  1. `w_init_mode: zeros` (kills gradient through V_rest / tau).
  2. `lr_embedding` too low (embeddings can't move into a useful regime).
  3. `sparsity_freq` too aggressive (collapses embeddings before they form).
- **conn_R² < 0 means worse than constant prediction** — typically caused
  by an exploding W (lr too high) or by a wrong sign convention. The
  teacher's W now has rows = presynaptic, cols = postsynaptic; the GNN's
  edge_index uses (src = pre, dst = post). If you see a sudden sign flip
  in conn_R² across iterations, suspect an indexing issue, not a model
  capacity issue.
- **f_theta blank plot diagnostic**: if `tmp_training/function/f_theta/*.png`
  looks empty after iter 1, the `plotting.xlim` is too narrow; increase
  it. The voltage data has range ≈ [-2.5, 2.5] in normalised units.
- **Use `use_gt_edges: false`** (current default) — the teacher is fully
  connected, so the GNN should train on the full N²−N edge set. Setting
  `use_gt_edges: true` would only matter for the einstein-masked teacher
  where the GT graph is sparse.
