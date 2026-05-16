# Cortex delaygo — Yang 2019 Task Learning

## Goal

Find the **best training scheme** for `TaskRNN` (free-W mode) on the Yang
2019 **delaygo** task — a one-shot delay-then-respond task where the network
must (a) maintain a memory of a brief stimulus angle during a variable-length
delay, then (b) emit the stored angle on a ring-readout after fixation drops.

Model: `TaskRNN` in
[src/connectome_gnn/models/task_rnn.py](../src/connectome_gnn/models/task_rnn.py)
in **free-W mode** (`graph_model.W_param: free`) — fully-learnable 256×256
recurrent matrix, MLP encoder (85→N) and decoder (N→33), no biological
connectome constraint.

**Primary metric**: `perf` = Yang `get_perf` correctness rate on the test set.
A trial is correct iff (a) fixation channel stays low during the response
window AND (b) decoded angle (popvec on the 32 ring channels) is within ~π/5
of the target. Yang 2019 reports **perf > 0.95** on single-task delaygo.

The dataset is **fixed**: 1000 train + 200 test trials × T=200 frames @
dt=20ms (one of Yang's standard task lengths). Only the training
hyperparameters change.

## What's known (data shapes)

```
u (stimulus):  (B, T=200, 85)   — fixation + rule encoding + 2 × 32-ch ring
y (target):    (B, T=200, 33)   — fixation channel + 32-ch ring readout
c_mask:        (T=200, 33)      — Yang weighting:
                                    × 5 on the response window
                                    × 2 on the fixation channel
                                    × 0 in the 100ms grace-window after fix_off
```

Loss = `mean(c_mask · (y_hat − y)²)` (Yang `'lsq'`).

## Common failure modes for delaygo

The agent should look for these in the per-epoch trajectory + readout traces:

1. **"Fixate forever"** — net outputs fixation throughout the whole trial.
   `perf ≈ 0` but `mse` is small (fixation dominates trial length). Fix:
   stronger `c_mask` weighting is already baked in; usually solved by more
   epochs or better lr schedule.
2. **"Respond early"** — net activates the ring before `fix_off`. Visible as
   pre-response ring activity in the snapshot traces.
3. **"Wrong angle"** — net responds at the right time but decodes the wrong
   direction. Implies the bump didn't survive the delay.
4. **"Delocalised bump"** — hidden activity spreads across all units; no
   stable population code. Often caused by σ saturation or weak recurrence.
5. **"W blow-up"** — `‖W_rec‖` grows unbounded; loss spikes. Fix with
   `grad_clip_W` and/or `coeff_W_L2`.

## Available hyperparameters (the search space)

These are the fields the agent may set per-slot in `training:` /
`graph_model:`. Anything else should NOT be touched unless explicitly noted.

### Recurrent training scheme (PRIORITY)

| Field                   | Default                                          | What it controls                                                                                                  |
| ----------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `lr`                    | `1e-3`                                           | Initial learning rate. Yang uses 1e-3 → 1e-4 over training. Try {5e-4, 1e-3, 2e-3}.                               |
| `lr_schedule`           | `[1e-3, 5e-4, 2e-4, 1e-4, …]`                    | Per-epoch lr. Try slower decay {1e-3, 1e-3, 5e-4, …} or faster {1e-3, 1e-4, 1e-5, …}.                             |
| `n_epochs`              | `10`                                             | More epochs help on delaygo because the delay-period invariance is slow to learn. Try {5, 10, 20, 30}.            |
| `batch_size`            | `64`                                             | Try {32, 64, 128}. Smaller = noisier gradients but more updates per epoch.                                        |
| `grad_clip_W`           | `1.0`                                            | Max-norm gradient clip. Yang trains with clip ≈ 1.0. Try {0, 0.5, 1.0, 2.0, 5.0}.                                 |
| `noise_recurrent_level` | `0.0`                                            | Gaussian noise on `h` per Euler step (training only). Try {0, 1e-3, 1e-2, 5e-2}. Often helps delay-memory tasks.  |
| `w_init_scale`          | `1.0`                                            | Scale for `randn_scaled` mode → `S = randn · scale/√N`. Yang papers use ≈ 1.0 (edge of chaos). Try {0.5, 1, 2}.   |
| `w_init_mode`           | `randn_scaled`                                   | `randn_scaled` (default, Yang-style) | `zeros` | `randn` | `uniform_scaled`.                                       |
| `coeff_rate_L2`         | `0.0`                                            | L2 on σ(h) — discourages saturation. Yang-specific. Try {0, 1e-4, 1e-3, 1e-2}.                                    |
| `coeff_W_L2`            | `0.0`                                            | Weight decay on `W_rec`. Try {0, 1e-5, 1e-4, 1e-3}.                                                                |

### Architecture (free-W mode)

| Field                  | Default | Role                                                                                |
| ---------------------- | ------- | ----------------------------------------------------------------------------------- |
| `n_units`              | `256`   | Recurrent population size. Yang uses 256; try {128, 256, 512}.                       |
| `recurrent_activation` | `relu`  | σ in `r = σ(h)`. `relu` (Yang default), `softplus`, `tanh`, `sigmoid`.               |
| `hidden_dim`           | `128`   | Encoder/decoder MLP hidden size.                                                    |
| `n_layers`             | `2`     | Encoder/decoder MLP depth.                                                          |
| `MLP_activation`       | `relu`  | Activation inside the encoder/decoder MLPs.                                         |
| `input_proj`           | `mlp`   | `mlp` (Yang default for 85-channel input) or `matrix`.                              |
| `output_proj`          | `mlp`   | `mlp` (default) or `matrix`.                                                        |

### Things you must NOT change

- `dataset` and `task.cortex.*` (the data is on disk; changing these doesn't
  regenerate).
- `task.task_type` (must stay `cortex`).
- `graph_model.signal_model_name` (must stay `cortex_delaygo`).
- `graph_model.W_param` (must stay `free` — `sign_locked` requires a
  connectome that doesn't exist for cortex).
- `n_input` / `n_output` (fixed by the data: 85 / 33).
- `aggr_type` (irrelevant for this model).

## Metrics (per slot, per iteration)

Read from `<exploration_dir>/<slot_name>_analysis.log` after training. The
trainer also writes a per-iteration `tmp_training/metrics.log` you can tail
during the run.

| Metric              | What it measures                                                                | Target              |
| ------------------- | ------------------------------------------------------------------------------- | ------------------- |
| `perf` (final)      | Yang `get_perf` on the full test set: fixation-correct AND popvec-decoded angle within tolerance. | **≥ 0.95**.         |
| `perf` (per epoch)  | End-of-epoch `perf`.                                                            | Monotonically high. |
| `loss`              | Masked MSE `mean(c_mask · (y_hat − y)²)`.                                       | Smooth, decreasing. |
| `mse`               | Unmasked MSE (sanity baseline).                                                 | Tracks `1 − perf`.  |
| `fix_perf`          | Fraction of trials with fixation-channel correctness only.                      | Should reach 1.0 fast (easy). |
| `dir_perf`          | Fraction of trials with popvec angle within tolerance (conditional on response).| The hard part.      |
| `rate_mean`         | Mean firing rate `<σ(h)>` over the test set. Watch for saturation.              | Stable, ~0.1–0.4.   |
| `W_norm`            | `‖W_rec‖_F`. Watch for blow-up.                                                 | Stable.             |

**The per-epoch `perf` trajectory is the most diagnostic signal.** A run with
`e1=0.5 e2=0.8 e3=0.95 e4=0.95 e5=0.95` has converged; a run stuck at
`e1=0.2 e2=0.2 …` is not learning at all (lr too small or activation issue);
a run with `e1=0.9 e2=0.95 e3=0.2 e4=0.0` collapsed (W blow-up or unstable
delay-period dynamics).

## Causality rule

You can change one or two parameters per slot.

In **robustness mode** (every slot identical), the pipeline forces N
different seeds; this measures seed sensitivity of a candidate winner.

## Block plan

4 slots/batch. Iterations: 96 total = 8 blocks × 12 iter/block = 3 batches/block.

| Block | Focus                                  | Knobs to scan                                                                                     | Why                                                                                |
| ----- | -------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 1     | **Baseline + activation sweep**        | `recurrent_activation` ∈ {relu, softplus, tanh, sigmoid}; 2 seeds each (robustness)               | Which σ trains best on delaygo? ReLU is Yang default but softplus / tanh may differ. |
| 2     | **W init**                             | `w_init_scale` ∈ {0.3, 0.5, 1.0, 1.5, 2.0}; `w_init_mode` ∈ {randn_scaled, uniform_scaled}        | Edge-of-chaos init is critical for delay-period memory.                            |
| 3     | **lr + schedule**                      | `lr` ∈ {3e-4, 1e-3, 3e-3}; `lr_schedule` variants                                                  | Find the sweet spot for stable delay-period learning.                              |
| 4     | **Stability (clip + noise)**           | `grad_clip_W` ∈ {0, 0.5, 1, 5}; `noise_recurrent_level` ∈ {0, 1e-3, 1e-2, 5e-2}                    | Stabilise W and smooth long-T BPTT.                                                |
| 5     | **Regularisers**                       | `coeff_rate_L2` ∈ {0, 1e-4, 1e-3, 1e-2}; `coeff_W_L2` ∈ {0, 1e-5, 1e-4, 1e-3}                      | Discourage saturation / runaway W norm.                                            |
| 6     | **Architecture (recurrent capacity)**  | `n_units` ∈ {128, 256, 512}                                                                       | Does the recurrent population size matter for delaygo specifically?                |
| 7     | **Architecture (encoder/decoder)**     | `hidden_dim` ∈ {64, 128, 256}; `n_layers` ∈ {1, 2, 3}; `input_proj`/`output_proj` ∈ {matrix, mlp} | I/O MLP capacity — Yang uses MLPs for delaygo's 85-channel input.                  |
| 8     | **Combine + final robustness**         | Best knobs from blocks 1–7 combined; 10 seeds of the resulting winner config.                     | Confirm winner is seed-robust.                                                     |

## Mutation log format (per iteration)

After each batch, append to working memory:

```
## Iter N (block B): [exploration | robustness]
Parent: iter_M_slot_K  (perf=X.XXX)
Hypothesis: "[testable claim about what the mutation should do]"
Slot 0: [parent/control]   perf=X.XXX  fix=Y.YYY  dir=Z.ZZZ  traj=e1=A e2=B …
Slot 1: [knob -> value]    perf=X.XXX  …
…
Slot N: [knob -> value]    perf=X.XXX  …
Best slot: K  ->  perf=X.XXX
Verdict: [supported | falsified | inconclusive]
Next parent: iter_N_slot_K
```

When a slot collapses, note the epoch at which `perf` dropped and the
`W_norm` / `rate_mean` values at that epoch — this is the most informative
diagnostic.

## Winner config

At every block boundary, copy the best slot's config to
`config/cortex/cortex_delaygo_winner.yaml` with header:

```yaml
# Winner: cortex_delaygo_winner.yaml
# Source: iter_NNN_slot_KK  (final perf = X.XXX)
# Block: B  (focus: <focus>)
# Date: YYYY-MM-DD
#
# Why this is the winner:
#   - <one-sentence reason>
#   - <key knob change>
#
# Per-epoch trajectory: e1=A e2=B e3=C e4=D e5=E
# Robustness: tested across N seeds, mean=X.XXX ± Y.YYY
```

## Notes / hints

- **Yang 2019's reference** trains a single-task delaygo network to perf ≈
  1.0 in ~10 epochs on this dataset size (1k trials). If your loop is stuck
  at perf ≈ 0.5 after several epochs, the issue is almost certainly
  activation choice or lr — not regularisation.
- **The `fix_perf` should hit 1.0 very fast** (it's the easy half of the
  task). If `fix_perf` lags, something is broken with the loss masking, not
  the recurrent dynamics.
- **`dir_perf` is the actual signal** — that's the delay-period memory
  part. Track `dir_perf` separately from `perf`.
- **Population-code diagnostics**: the trainer's snapshot directory has the
  hidden-rate kinographs. A successful delaygo run shows a stable activity
  bump that survives the entire delay period; collapsed runs show the bump
  decay into noise before fix_off.
