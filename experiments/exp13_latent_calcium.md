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
