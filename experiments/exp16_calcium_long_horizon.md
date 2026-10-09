---
number: 16
name: calcium_long_horizon
title: 'Latent-calcium training with Plexus exp17''s time scales: horizons of 1-2
  s instead of 40-400 ms, every 20 ms frame or every 10th observed (2 x 2, 40 dB,
  kernel given)'
purpose: get W to train on calcium through the latent voltage, which experiment 13
  never did (R2_W -0.01 in all 8 runs), by rolling out over several indicator time
  constants as Plexus exp17 does, and test whether a 5 Hz recording (one calcium frame
  in ten, ten voltage updates between them) trains as well as a 50 Hz one
baseline: experiments/specs/exp13/fly/flyvis_cal_40db_kg.yaml (same GNN, regularisers,
  data, kernel given, 20 epochs, data_augmentation_loop 50, batch 4)
specs_dir: experiments/specs/exp16/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
n_cpus: 12
axes:
  data:
  - 40db
  - inf
  frame:
  - f20ms
  - f200ms
  horizon:
  - h50
  - h100
arms:
- id: kg
  label: latent calcium, kernel given, long horizon
  spec_pattern: flyvis_cal_{data}_{frame}_{horizon}
  differs_by:
    training.calcium_checkpoint_steps: true
report:
  axis_labels:
    data:
      40db: calcium, 40 dB
      inf: calcium, no meas. noise
    frame:
      f20ms: every 20 ms frame observed
      f200ms: 1 frame in 10 observed (200 ms), 10 voltage steps between
    horizon:
      h50: horizon 20 -> 50 steps (0.4 -> 1 s)
      h100: horizon 20 -> 100 steps (0.4 -> 2 s)
  arm_labels:
    kg: latent calcium, kernel given, long horizon
job_ids:
  flyvis_cal_40db_f20ms_h50: '154583479'
  flyvis_cal_40db_f20ms_h100: '154583480'
  flyvis_cal_40db_f200ms_h50: '154583481'
  flyvis_cal_40db_f200ms_h100: '154583482'
---

# Experiment 16 — calcium_long_horizon

**Latent-calcium training with Plexus exp17's time scales (2 x 2, 40 dB, kernel given)**

## Why

Experiment 13 trained the current GNN on calcium through a latent voltage and a
GCaMP6f indicator, and W never moved: R2_W -0.01 and latent rollout r 0.003 in all 8
runs. The gradient check of 2026-10-06 (exp13 md) located it: the horizon curriculum
started at 2 frames of 20 ms, where c(k+1), c(k+2) barely depend on the network, so the
data gradient on W was 1/1,700 of the W lasso's (coeff_W_L1 1.5e-4 per weight) and
the lasso drove mean |W| from 1.2e-3 to 1.2e-4 in epoch 0, a point the later, longer
horizons never left. At a 20-frame horizon the same gradient is 0.19 of the lasso's on
average and beats it on 2.2% of edges, more than the 0.35% a working one-step voltage
run starts from.

Plexus exp17, whose latent-calcium trainer works on zebrafish data, has the same
structure as experiment 13 (linear taps read the latent voltage from recorded
calcium, the network steps it, a leaky indicator turns it into calcium, the loss
scores the calcium) but different time scales: frames of 0.914 s with 4 voltage
substeps each, horizons of 1 -> 50 frames (100 tried), i.e. many indicator time
constants per rollout. Experiment 13's longest horizon, 20 frames of 20 ms, spans one
400 ms decay. Its session recommended 60-100-frame horizons for our 20 ms frames.

## What changes (code, `models/calcium_observation.py`)

- **`training.rollout_loss_stride: m`** -- the knob the voltage rollout already uses
  for a 1-in-m recording -- now applies to calcium: the latent and the indicator still
  advance every 20 ms (m voltage updates per observed frame, exp17's substeps), the loss
  scores only c(k+m), c(k+2m), ... and the start taps read c(k), c(k-m), .... The loss
  is the mean over the scored frames.
- **`training.calcium_checkpoint_steps: true`** recomputes every GNN step after the first
  in the backward pass. Without it a step costs ~1.1 GB (4 samples of the 13,741-neuron,
  434,112-edge network) and a 100-step horizon ~120 GB, more than the 95 GB GPU. The
  step's voltage and stimulus are arguments of the recomputed function, set on the state
  objects inside it -- Plexus exp17's checkpoint bug was a recompute reading state that
  had moved on. It also sets `model.grad_checkpoint`, so graph_trainer compiles without
  CUDA graphs.
- `rollout_burn_in` / `rollout_step_weighting` with calcium now raise instead of being
  silently ignored.

**Verification (2026-10-08, real Flyvis model, 40 dB data, one fixed batch of 4):**

| check | result |
|---|---|
| new code, defaults, against the code before (horizon 6) | loss bit-identical, gradients bit-identical |
| checkpointed vs stored, horizon 6 / 12 / 20 with stride 10 | loss bit-identical; worst relative gradient error 2.4e-9 (horizon 12); run-to-run GPU noise is 1.5e-8 absolute |
| negative control: stimulus set OUTSIDE the recomputed function (exp17's bug) | relative gradient error 3.5e-2 (worst tensor), 2.4e-3 on W -- the comparison detects it |
| compiled as graph_trainer does: checkpointed (default mode) vs stored (CUDA graphs) | loss bit-identical; relative gradient error 3.1e-8 |
| checkpointed with CUDA graphs forced | PyTorch stops: "accessing tensor output of CUDAGraphs that has been overwritten" (why the flag is set) |
| peak GPU memory, horizon 12 | 23.5 GB stored, 13.2 GB checkpointed |
| peak GPU memory, horizon 100, checkpointed | 13.2 GB |

`tests/test_calcium_observation.py` adds the checkpoint comparison (on a stub whose
stimulus enters nonlinearly -- added outside the nonlinearity, a stale stimulus leaves
dL/dW unchanged and the test went blind to the bug) and a test that the loss ignores
unobserved frames under a stride.

## Design

| run | observed frames | horizon schedule (steps of 20 ms, 20 epochs) | scored frames per rollout |
|---|---|---|---|
| `flyvis_cal_40db_f20ms_h50` | every 20 ms | 20 x5, 30 x5, 40 x5, 50 x5 | 20 -> 50 |
| `flyvis_cal_40db_f20ms_h100` | every 20 ms | 20, 20, 30, 30, ..., 90, 90, 100 x4 | 20 -> 100 |
| `flyvis_cal_40db_f200ms_h50` | every 10th (200 ms) | as f20ms_h50 | 2 -> 5 |
| `flyvis_cal_40db_f200ms_h100` | every 10th (200 ms) | as f20ms_h100 | 2 -> 10 |

All four: experiment 13's 40 dB data (`flyvis_unified_blank50_cal_40db`), kernel given
(GCaMP6f, rise 75 ms, decay 400 ms), 6 start taps, batch 4, every GNN step checkpointed.
The trainer divides an epoch's iterations by its horizon (constant compute per epoch),
so h50 runs ~103k iterations and h100 ~65k, against experiment 13's 423k.

**Noise-free twins (added 2026-10-09):** the same four runs on
`flyvis_unified_blank50_cal_inf` (`flyvis_cal_inf_<frame>_<horizon>`), for the
comparison with experiment 12's deconvolve-then-train without measurement noise,
R2_W 0.978 (20-step recurrent).

## Decisions

- **40 dB, not noise-free.** Experiment 12's deconvolve-then-train reaches R2_W 0.42 at
  40 dB with 20-step recurrent training (0.978 without noise), so 40 dB is where a
  latent-calcium trainer can show something deconvolution cannot. The gradient check
  was also run at 40 dB.
- **Residual kept in V/s (divided by dt), not by the SD of calcium's one-step change as
  exp17 does.** Measured on the 40 dB recording: the SD of c(t+m) - c(t) is 0.015 for m
  = 1, 0.067 for m = 10, so exp17's scaling multiplies the residual by 67 and 15,
  against 1/dt = 50. At 200 ms frames it would weaken the data gradient 3x against the
  lasso, the balance that killed experiment 13.
- **Kernel given.** exp17 fixes its indicator from batch 16 on, a learned one collapsing
  there as in experiment 13's kn arm (15 / 57 ms against the true 75 / 400 ms).
- **No gradient clipping** (exp17 clips the norm at 1.0): it bounds large steps, and
  experiment 13's problem was a vanishing one. Kept out so the horizon is the change.
- **Horizons start at 20, in multiples of 10**, so the two frame rates unroll the same
  rollouts and differ only in which frames are scored; 20 is where the gradient check
  found the data gradient beating the lasso.

## Reading

W trains if mean |W| leaves ~1e-4 in epoch 0 (experiment 13 stayed at 1.2e-4) and
R2_W rises. The bar is experiment 12's deconvolution at 40 dB, R2_W 0.42 / rollout r
0.83, on the same measurements.

## Specs

4 runs in `experiments/specs/exp16/fly/`.

## First reading (2026-10-08, iteration 801 of epoch 0, horizon 20, train split)

W leaves zero in all four runs, which experiment 13 never did. Template-readout R2_W,
learned/true slope and Pearson r of the raw learned W against the true W:

| run | R2_W | slope | Pearson r of W |
|---|---|---|---|
| experiment 13, `flyvis_cal_40db_kg` (horizon 2) | -0.012 | 0.000 | -0.008 |
| `f20ms_h50` | 0.105 | 0.111 | 0.161 |
| `f20ms_h100` | 0.102 | 0.107 | 0.292 |
| `f200ms_h50` | 0.030 | 0.073 | 0.151 |
| `f200ms_h100` | 0.008 | 0.089 | 0.181 |

The two pairs share epoch 0 (both start at horizon 20), so their differences there are
GPU nondeterminism (same seed, same batches), not the schedule. Held-out numbers come at `-o test_plot`.

<!-- STATUS:BEGIN -->

## Status

**0/4 landed**, 0 trained (awaiting `-o test_plot`), 4 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | frame | horizon | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | frame | horizon | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kg | f20ms | h50 | 89,706 |  | 0.232 ± 0.000 |  |  | 0.158 ± 0.000 | -0.019 ± 0.000 | 0.570 ± 0.000 | 0.538 ± 0.000 | 0.725 ± 0.000 |  |  | 0.684 ± 0.000 |
| kg | f20ms | h100 | 57,978 |  | 0.248 ± 0.000 |  |  | 0.178 ± 0.000 | -0.013 ± 0.000 | 0.545 ± 0.000 | 0.538 ± 0.000 | 0.610 ± 0.000 |  |  | 0.708 ± 0.000 |
| kg | f200ms | h50 | 90,186 |  | 0.228 ± 0.000 |  |  | 0.326 ± 0.000 | 0.049 ± 0.000 | 0.527 ± 0.000 | 0.611 ± 0.000 | 0.559 ± 0.000 |  |  | 0.666 ± 0.000 |
| kg | f200ms | h100 | 58,330 |  | 0.254 ± 0.000 |  |  | 0.355 ± 0.000 | -0.144 ± 0.000 | 0.532 ± 0.000 | 0.602 ± 0.000 | 0.576 ± 0.000 |  |  | 0.688 ± 0.000 |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_cal_40db_f20ms_h50` | running | 89,706 | `` |  |
| `flyvis_cal_40db_f20ms_h100` | running | 58,066 | `` |  |
| `flyvis_cal_40db_f200ms_h50` | running | 90,186 | `` |  |
| `flyvis_cal_40db_f200ms_h100` | running | 58,330 | `` |  |

<!-- STATUS:END -->
