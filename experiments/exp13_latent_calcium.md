---
number: 13
name: latent_calcium
title: Training on calcium through a latent voltage and a GCaMP6f indicator, kernel
  given against kernel learned, from no measurement noise to 20 dB
purpose: train the current GNN on the calcium itself, the voltage being a latent the
  model integrates and the loss scoring the calcium a GCaMP6f indicator would report
  from it (the batch-6 design of Plexus exp17), and compare with experiment 12's deconvolve-then-train
  on the same noisy measurements, with the indicator's kernel given and learned
baseline: experiments/specs/exp12/fly/flyvis_ca_<data>_rc20.yaml (same GNN, regularisers
  and 20-epoch recurrent schedule) on flyvis_unified_blank50_cal_<data>
specs_dir: experiments/specs/exp13/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
n_cpus: 12
axes:
  data:
  - inf
  - 40db
  - 30db
  - 20db
arms:
- id: kg
  label: kernel given
  spec_pattern: flyvis_cal_{data}_kg
  differs_by:
    training.observable: calcium
    training.calcium_kernel_learned: false
- id: kn
  label: kernel not given
  spec_pattern: flyvis_cal_{data}_kn
  differs_by:
    training.observable: calcium
    training.calcium_kernel_learned: true
    training.calcium_tau_rise: 0.05
    training.calcium_tau_decay: 1.0
report:
  axis_labels:
    data:
      inf: calcium, no meas. noise
      40db: calcium, 40 dB
      30db: calcium, 30 dB
      20db: calcium, 20 dB
  arm_labels:
    kg: current, latent calcium, kernel given
    kn: current, latent calcium, kernel learned
job_ids:
  flyvis_cal_inf_kg: '154498632'
  flyvis_cal_40db_kg: '154498633'
  flyvis_cal_30db_kg: '154498634'
  flyvis_cal_20db_kg: '154498635'
  flyvis_cal_inf_kn: '154498636'
  flyvis_cal_40db_kn: '154498637'
  flyvis_cal_30db_kn: '154498638'
  flyvis_cal_20db_kn: '154498639'
analyse_job_ids:
  flyvis_cal_inf_kg: '154536735'
  flyvis_cal_40db_kg: '154536736'
  flyvis_cal_30db_kg: '154536737'
  flyvis_cal_20db_kg: '154536738'
  flyvis_cal_inf_kn: '154536739'
  flyvis_cal_40db_kn: '154536740'
  flyvis_cal_30db_kn: '154536741'
  flyvis_cal_20db_kn: '154536742'
---

# Experiment 13 — latent_calcium

**Training on calcium through a latent voltage and a GCaMP6f indicator, kernel given against kernel learned, from no measurement noise to 20 dB**

**Purpose.** train the current GNN on the calcium itself, the voltage being a latent
the model integrates and the loss scoring the calcium a GCaMP6f indicator would report
from it, and compare with experiment 12's deconvolve-then-train on the same noisy
measurements.

## Why

Experiment 12 deconvolves the calcium to a voltage first and trains on that estimate,
so the deconvolution's errors become the target: at 40 dB the one-step run fell to
R2_W 0.149, and 20-step recurrent training lifted it only to 0.420. Here nothing is
deconvolved. The forward model runs the other way, voltage -> calcium, and the loss
is on the calcium that was measured.

## The forward model (`training.observable: calcium`)

`models/calcium_observation.py`. The dataset's `voltage.zarr` is the calcium recording
c; the voltage v is a latent state the GNN integrates; the indicator is two leaky
stages of unit gain, a rise state r then the calcium:

    r(t+1) = r(t) + k_r (v(t) - r(t)),       k_r = 1 - exp(-dt / tau_rise)
    c(t+1) = c(t) + k_d (r(t+1) - c(t)),     k_d = 1 - exp(-dt / tau_decay)

with dt = 20 ms, the frame interval. This is the unit-area difference of exponentials
the datasets were built with (rise 75 ms, decay 400 ms): run on the simulated voltage
it reproduces the dataset's calcium to 0.075% of SD(c), against a measurement noise of
1% of SD(c) at 40 dB, so "kernel given" is the true kernel.

Each rollout of K frames starts from the 6 recorded frames c(k) .. c(k-5), read by two
sets of 6 taps shared by every neuron and learned: v(k) = sum_j a_j c(k-j),
r(k) = sum_j b_j c(k-j), starting at v = r = c(k). It scores c(k+1) .. c(k+K), each
residual divided by dt so it reads in V/s like the one-step loss's derivative. c(k+s+1)
depends on v(k+s), which took s GNN steps, so the horizon schedule is 2, 3, ..., 20, 20
over the 20 epochs (experiment 12's 1 .. 20 shifted up at the bottom: a horizon of 1
gives the GNN no gradient).

The test rollout starts from the latent read by the learned taps and is scored against
`voltage_true.zarr`, the simulated voltage: "rollout r" here is the latent voltage
against the truth. The one-step metric scores the model on frames of `voltage.zarr`,
which is calcium here, so it is not comparable with experiment 12's.

## What differs

| arm | indicator taus | learned |
|---|---|---|
| kg, kernel given | rise 75 ms, decay 400 ms (the true kernel) | taps only |
| kn, kernel not given | start rise 50 ms, decay 1 s (a GCaMP6s-like guess) | taps and both taus |

Data: `flyvis_unified_blank50_cal_<inf|40db|30db|20db>`, built by
`tools/build_calcium_dataset.py --observe calcium` from `flyvis_unified_blank50_kernel`
with the same seed and draw order as experiment 12's `_ca_` datasets, so both
experiments see the same noisy measurements. Measurement noise: one SD = frac x SD of
all calcium, frac = 10^(-SNR/20) (40 dB -> 0.01, 30 dB -> 0.032, 20 dB -> 0.1).

Everything else is experiment 12's recurrent spec: current GNN, the same regularisers,
20 epochs, data_augmentation_loop 50, fold cv00.

## Bit-identity of the existing trainer

The switch is off by default and adds no parameter: `attach_calcium_indicator` returns
before touching the model unless `observable` is `calcium`, and `recurrent_loss`
dispatches to `calcium_rollout_loss` only in that case. Checked on experiment 12's
`flyvis_ca_true_rc20` and `flyvis_ca_true_1s`, 3 iterations per epoch, deterministic
(2026-10-02): two recurrent runs before the change, on different GPUs, are bit-identical
to each other, and the recurrent and one-step runs after the change are bit-identical to
the runs before it -- `loss.pt`, `loss_components.pt` and every tensor of every
checkpoint (6 recurrent, 3 one-step). A calcium smoke run (40 dB, kernel learned)
trained, moved the taps and taus, and its test rollout ran from the latent and was
scored against `voltage_true.zarr`.

## Specs

8 runs in `experiments/specs/exp13/fly/`, named `flyvis_cal_<data>_<kg|kn>`. Queue
`gpu_rtx6000`, wall 48 h.

## First reading (2026-10-06): W never trained

All 8 runs finished training (423,233 iterations each) and landed on 2026-10-06. The
held-out analysis confirms the checkpoints: R2_W -0.012 to -0.015 in all 8 runs, and the
latent voltage's test rollout against the simulated voltage r = 0.003, against
experiment 12's 0.978 / 0.997 (no noise) and 0.420 / 0.831 (40 dB) for R2_W / rollout r
with deconvolution and the same recurrent training. From the checkpoints:

- **The connectivity did not move.** Mean |W| over the 434k edges is 1.22e-4 after
  epoch 0 and 1.20e-4 after epoch 19, in every run; the Pearson r of the raw learned W
  against the true W is 0.00 at every epoch. Experiment 12's `flyvis_ca_inf_rc20`
  reaches mean |W| 0.12 and r 0.96 after its first epoch. The training-time template
  readout agrees: Wij R2 -0.01 in all 8 runs, gain 0.
- **The model fitted the calcium without the circuit.** With the kernel learned, the
  indicator sped up instead of finding GCaMP6f: rise / decay went from the 50 ms /
  1 s start to 15 / 57 ms (no noise), 44 / 201 ms (40 dB), 31 / 53 ms (30 dB),
  13 / 31 ms (20 dB), against the true 75 / 400 ms -- a faster indicator lets the
  6 taps carry the recorded calcium forward with little help from the dynamics.
- **Likely cause (not yet checked): the loss reaches the GNN about 90x weaker than
  in one-step training.** c(k+s+1) depends on v(k+s) through both stages, a factor
  k_r k_d = 0.234 x 0.049 = 0.011 per frame, which the 1/dt on the residual only
  partly restores. The regularisers were left at experiment 12's coefficients, and
  W, which starts near 0, gets almost no gradient to leave there.

Next: measure the gradient on W in one calcium iteration against one one-step
iteration on the same frames; if the 90x is confirmed, scale the calcium residual
by 1/(k_r k_d) or start from experiment 12's trained GNN, and relaunch.

**Against Plexus exp17's calcium trainer** (`trainer._train_trace`, spec 0172
zap_c16_t2; asked its session devcontainer-25 on 2026-10-06), the differences seen so far:

- exp17 teacher-forces 10 recorded frames (`task.warmup: 10`) before the free steps,
  so the latent and the indicator settle on the recording; exp13 has none.
- exp17 divides the residual by the SD of the recording's one-step change (GraphCast
  normalisation); exp13 divides by dt.
- exp17's indicator closes 0.37-0.60 of its gap per 0.914 s frame and its horizons of
  1-10 frames span about 5 indicator time constants; exp13's closes 0.234 and 0.049
  per 20 ms frame through its two stages, and 20 frames span about one 400 ms decay.
- exp17 clips the gradient norm at 1.0 and integrates the latent with 4 exponential
  substeps per frame; exp13 does neither.
- exp17 has no ground-truth connectivity: "working" there is calcium forecast skill,
  and its no-network twin (zap_c16_t2_now) forecasts as well (brain-mean R2 0.454
  against 0.420), so it does not show that the latent trainer identifies W.

**exp17's answer (devcontainer-25, 2026-10-06):**

- Connectivity was never checked against a known W (real data). The network tests show
  the coupling carries forecast skill: per-neuron r with the brain mean removed is
  +0.444 for 15.4 against +0.201 for its no-W twin. The no-W calcium twin matching on
  brain-mean R2 is the indicator's level bias, which R2 counts as error.
- In exp17, W trains as much with the indicator as without. Mean |W| (W starts at 0)
  for calcium vs its dF/F twin: 0.204 vs 0.243 (batch 6), 0.039 vs 0.049 (16.3),
  0.039 vs 0.038 (16.7), 0.046 vs 0.042 (18.7).
- Batch 6 had no warm-up and still trained W, so warm-up is not what makes W learn.
- A learned indicator collapses there too: tau_ca fell to 0.06 s whatever its start
  (0.5 or 1 s), which makes the indicator transparent and lets the taps carry the
  trace. exp17 fixes tau_ca (1, 2 or 3 s) from batch 16 on. This is exp13's 15 / 57 ms.
- Its diagnosis for exp13. (1) Loss scale against L1: c barely changes over a 20 ms
  frame, so the data gradient on W is tiny next to the voltage-tuned L1 1.5e-4, and
  |W| = 1.2e-4 held for 20 epochs looks like an L1 equilibrium. Normalise the residual
  by calcium's own one-step-change SD. (2) Time scale: horizons of 2-20 frames are
  under one 400 ms decay, so carrying c forward through the taps already fits. Use
  ~100-200 ms frames or 60-100-frame horizons. (3) Keep the kernel fixed. (4) Score a
  no-W and a W = 0 twin.

The kernel-given arm failed as completely as the learned one, so (3) explains the
indicator's speed-up but not the dead W. (1) is the first thing to measure: the
gradient on W from the calcium term against the L1 term, at the start of training.

**Gradient check (2026-10-06, `/tmp/grad_probe.py`).** The model was built as the
trainer builds it, with the same seed and torch.compile off. Mean over 4 batches of 4 start
frames: the gradient of the data term on each of the 434,112 weights, against the W
lasso's 1.5e-4 per weight (coeff_W_L1, the same in both experiments).

| loss (40 dB) | horizon | mean data / lasso | edges where the data gradient beats the lasso | 99th pct data gradient |
|---|---|---|---|---|
| voltage (exp12), at init | 1 | 0.108 | 0.35% | 1.0e-4 |
| voltage (exp12), at init | 20 | 0.023 | 0.00% | 2.0e-5 |
| calcium (exp13), at init | 2 | 0.0006 | 0.00% | 6.8e-7 |
| calcium (exp13), at init | 20 | 0.189 | 2.22% | 2.2e-4 |
| calcium (exp13), trained, epoch 19 | 20 | 6e-8 | 0.00% | 5.5e-11 |

What this shows: the voltage run's first epoch (horizon 1) has about 1,500 edges whose
data gradient beats the lasso. Those grow, and once W is off zero the rest of the
network gets gradient. The calcium run's first epoch is at horizon 2, where c(k+1) and
c(k+2) barely depend on the network, and no edge beats the lasso. The lasso shrinks
every weight (mean |W| 1.2e-3 at init -> 1.2e-4 after epoch 0). With W near zero, the
data gradient on W at the trained checkpoint is 7 orders of magnitude below the lasso,
a dead point the longer horizons of later epochs never leave. At horizon 20 from the
start, 2.2% of edges would beat the lasso, more than the voltage run's 0.35%.

So the cause is the curriculum's start, not the warm-up: exp17's batch 6 trained W
without warm-up, and here the data gradient at horizon 20 is already larger than in a
working voltage run. Normalising the residual Plexus's way would not suffice alone: the
SD of calcium's one-step change at 40 dB is dominated by the noise, about 0.011, so it
scales the residual about 2x more than 1/dt, against the ~200x missing at horizon 2.

<!-- STATUS:BEGIN -->

## Status

**8/8 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | data | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kg | inf | 1 |  | 0.003 ± 0.000 |  |  | -0.012 ± 0.000 (0.4) | -4.306 ± 0.000 (47.2) |  |  | -0.023 ± 0.000 (11.1) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.169 ± 0.000 |
| kg | 40db | 1 |  | 0.003 ± 0.000 |  |  | -0.012 ± 0.000 (0.4) | -4.154 ± 0.000 (41.6) |  |  | -0.024 ± 0.000 (11.1) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.132 ± 0.000 |
| kg | 30db | 1 |  | 0.003 ± 0.000 |  |  | -0.012 ± 0.000 (0.4) | -4.092 ± 0.000 (44.5) |  |  | -0.024 ± 0.000 (11.1) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.189 ± 0.000 |
| kg | 20db | 1 |  | 0.003 ± 0.000 |  |  | -0.015 ± 0.000 (0.4) | -4.201 ± 0.000 (42.4) |  |  | -0.021 ± 0.000 (11.3) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.223 ± 0.000 |
| kn | inf | 1 |  | 0.003 ± 0.000 |  |  | -0.012 ± 0.000 (0.4) | -4.097 ± 0.000 (13.1) |  |  | -0.023 ± 0.000 (11.1) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.172 ± 0.000 |
| kn | 40db | 1 |  | 0.003 ± 0.000 |  |  | -0.012 ± 0.000 (0.4) | -4.094 ± 0.000 (42.4) |  |  | -0.024 ± 0.000 (11.1) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.226 ± 0.000 |
| kn | 30db | 1 |  | 0.003 ± 0.000 |  |  | -0.012 ± 0.000 (0.4) | -3.832 ± 0.000 (32.7) |  |  | -0.024 ± 0.000 (11.1) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.192 ± 0.000 |
| kn | 20db | 1 |  | 0.003 ± 0.000 |  |  | -0.015 ± 0.000 (0.4) | -2.654 ± 0.000 (99.2) |  |  | -0.021 ± 0.000 (11.3) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.187 ± 0.000 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | data | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_cal_inf_kg` | landed | 423,233 | `b089e21e6998` |  |
| `flyvis_cal_40db_kg` | landed | 423,233 | `b089e21e6998` |  |
| `flyvis_cal_30db_kg` | landed | 423,233 | `b089e21e6998` |  |
| `flyvis_cal_20db_kg` | landed | 423,233 | `b089e21e6998` |  |
| `flyvis_cal_inf_kn` | landed | 423,233 | `b089e21e6998` |  |
| `flyvis_cal_40db_kn` | landed | 423,233 | `b089e21e6998` |  |
| `flyvis_cal_30db_kn` | landed | 423,233 | `b089e21e6998` |  |
| `flyvis_cal_20db_kn` | landed | 423,233 | `b089e21e6998` |  |

<!-- STATUS:END -->
