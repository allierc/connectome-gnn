---
number: 12
name: calcium_deconvolution
title: 'Training on calcium: deconvolve to voltage (blind lambda), then the current
  GNN, one-step against 20-step recurrent, from no measurement noise to 20 dB'
purpose: reproduce, in the current codebase, the two-step calcium result of July 2026
  (deconvolve GCaMP6f calcium back to voltage, train the current GNN on it; R2_W 0.969
  with no measurement noise against 0.984 on the true voltage), with lambda chosen
  blind instead of by the oracle, and test whether 20-step recurrent training rescues
  the noisy levels where one-step training collapsed
baseline: GraphData/config/fly/nr2_ca_snr_g000_unified.yaml (the July calcium runs'
  spec, current GNN, flyvis_unified_blank50)
specs_dir: experiments/specs/exp12/fly
task: train
queue: gpu_rtx6000
wall: '48:00'
n_cpus: 12
axes:
  data:
  - 'true'
  - inf
  - 40db
  - 30db
  - 20db
arms:
- id: 1s
  label: one-step
  spec_pattern: flyvis_ca_{data}_1s
  differs_by: {}
- id: rc20
  label: recurrent 20
  spec_pattern: flyvis_ca_{data}_rc20
  differs_by:
    training.recurrent_training: true
report:
  axis_labels:
    data:
      'true': true voltage (control)
      inf: calcium, no meas. noise
      40db: calcium, 40 dB
      30db: calcium, 30 dB
      20db: calcium, 20 dB
  arm_labels:
    1s: current, one-step
    rc20: current, recurrent 20
job_ids:
  flyvis_ca_true_1s: '154483750'
  flyvis_ca_true_rc20: '154483751'
  flyvis_ca_inf_1s: '154483757'
  flyvis_ca_40db_1s: '154483758'
  flyvis_ca_30db_1s: '154483759'
  flyvis_ca_20db_1s: '154483760'
  flyvis_ca_inf_rc20: '154483761'
  flyvis_ca_40db_rc20: '154483762'
  flyvis_ca_30db_rc20: '154483763'
  flyvis_ca_20db_rc20: '154483764'
analyse_job_ids:
  flyvis_ca_true_1s: '154485255'
  flyvis_ca_inf_1s: '154485256'
  flyvis_ca_40db_1s: '154485257'
  flyvis_ca_30db_1s: '154485258'
  flyvis_ca_20db_1s: '154485259'
  flyvis_ca_true_rc20: '154487882'
  flyvis_ca_inf_rc20: '154487883'
  flyvis_ca_40db_rc20: '154487884'
  flyvis_ca_30db_rc20: '154487885'
  flyvis_ca_20db_rc20: '154487886'
---

# Experiment 12 — calcium_deconvolution

**Training on calcium: deconvolve to voltage (blind lambda), then the current GNN, one-step against 20-step recurrent, from no measurement noise to 20 dB**

**Purpose.** reproduce, in the current codebase, the two-step calcium result of July 2026 (deconvolve GCaMP6f calcium back to voltage, train the current GNN on it; R2_W 0.969 with no measurement noise against 0.984 on the true voltage), with lambda chosen blind instead of by the oracle, and test whether 20-step recurrent training rescues the noisy levels where one-step training collapsed

## What was recovered, and what was lost

The July work (branch `feat/calcium`, the `-ca` worktree) did not survive the
2026-09-26 history rewrite: the deconvolution module and the calcium generator
are in no git ref and nowhere on disk. What survives: the method
(`connectome-gnn-cx/neurips_review/calcium_note.tex`), the builder
(`build_snr_grid.py`), the configs (`GraphData/config/fly/nr2_ca_*`), the jobs'
stdout (`GraphData/log/neurips_review_jobs/nr2_ca_snr_*.out`), and the source
dataset `graphs_data/fly/flyvis_unified_blank50_kernel`, whose `calcium.zarr` is
the simulated voltage convolved with a unit-area GCaMP6f kernel (checked here:
relative residual 1.3e-7).

The July numbers (process noise 0.05, current GNN, one-step, lambda by oracle,
rollout scored against the ESTIMATE):

| measurement noise | lambda | 1-step r | rollout r | R2_W | R2_tau | R2_Vrest | cluster |
|---|---|---|---|---|---|---|---|
| none | 1e-6 | 0.974 | 0.992 | **0.969** | 0.952 | 0.537 | 0.866 |
| 40 dB | 1e-3 | 0.656 | 0.667 | 0.183 | 0.084 | -1.35 | 0.759 |
| 30 dB | 1e-2 | 0.550 | 0.403 | -1.27 | -5.67 | -19.6 | 0.730 |
| 20 dB | 1e-1 | 0.414 | 0.115 | -0.79 | -35.3 | -16.2 | 0.678 |

True-voltage control: R2_W 0.984.

## What is new in the codebase

- `models/gcamp.py`, the kernel (ported from `feat/oculomotor`; main imported
  it but did not have it): unit-area difference of exponentials, used here as
  rise 75 ms, decay 400 ms, support 2.4 s, the source dataset's kernel.
- `generators/calcium_deconvolution.py`: Tikhonov deconvolution in Fourier
  space with a second-difference penalty, reflect padding and a warm-up hold
  (`wiener_deconvolve`); the measurement-noise SD from the autocovariance
  (`estimate_noise_sd`); lambda by the discrepancy principle (`choose_lambda`).
  Tests in `tests/test_calcium_deconvolution.py`.
- `tools/build_calcium_dataset.py`: adds measurement noise (SD = frac x SD(C),
  frac = 10^(-SNR/20)), deconvolves with the blind lambda (chosen once, on the
  train split), writes `flyvis_unified_blank50_ca_<inf|40db|30db|20db>` whose
  `voltage.zarr` is the estimate and `voltage_true.zarr` the simulated voltage,
  and records r(v) and r(vdot) against the truth in `calcium_deconvolution.txt`
  (for the record; nothing is chosen from it).
- The tester scores the rollout against `voltage_true.zarr` when a dataset has
  one, not against the estimate as the July runs did.

`graph_trainer.py` is untouched.

## The four datasets (built 2026-09-30)

Blind lambda on 500 traces x 16,000 train frames; measurement noise one SD for
all neurons (frac x SD of all calcium, the July convention). r is the median
over 500 neurons of the per-neuron Pearson r against the simulated voltage
(July's figures were pooled over neurons, which favours high-variance cells).

| dataset | noise SD added / estimated | lambda (blind) | train r(v) | train r(vdot) | test r(v) | test r(vdot) |
|---|---|---|---|---|---|---|
| no meas. noise | 0 / 0 | 1e-8 | 0.999 | 0.995 | 0.994 | 0.935 |
| 40 dB | 0.00759 / 0.00764 | 1e-3 | 0.805 | 0.279 | 0.829 | 0.397 |
| 30 dB | 0.0240 / 0.0241 | 1 | 0.727 | 0.164 | 0.701 | 0.224 |
| 20 dB | 0.0759 / 0.0762 | 10 (grid top) | 0.611 | 0.097 | 0.565 | 0.136 |

The noise estimate is within 1% at every level. The blind lambda equals July's
oracle choice at 40 dB (1e-3) and is heavier than it at 30 and 20 dB (July 1e-2,
1e-1), where the discrepancy principle asks for more smoothing than the r(vdot)
optimum. Two earlier builds of the same day were discarded: the port first
extrapolated the autocovariance linearly (noise SD read as 0, lambda at the grid
floor) and then quadratically (noise read where there was none); the cubic
through lags 1-3 is the July recipe.

## Results (10/10 landed, 2026-10-01)

Held-out, fold 0, rollout scored against the simulated voltage:

| data | training | R2_W | R2_tau | R2_Vrest | rollout r | cluster |
|---|---|---|---|---|---|---|
| true voltage | one-step | 0.993 | 0.982 | 0.913 | 0.999 | 0.895 |
| true voltage | recurrent 20 | 0.979 | 0.961 | 0.839 | 0.997 | 0.856 |
| calcium, no meas. noise | one-step | **0.991** | 0.954 | 0.894 | 0.998 | 0.870 |
| calcium, no meas. noise | recurrent 20 | 0.978 | 0.971 | 0.764 | 0.997 | 0.848 |
| calcium, 40 dB | one-step | 0.149 | 0.314 | 0.731 | 0.517 | 0.762 |
| calcium, 40 dB | recurrent 20 | **0.420** | 0.103 | 0.609 | **0.831** | 0.711 |
| calcium, 30 dB | one-step | -0.008 | 0.000 | 0.587 | 0.039 | 0.211 |
| calcium, 30 dB | recurrent 20 | -0.012 | 0.310 | 0.579 | 0.116 | 0.442 |
| calcium, 20 dB | one-step | -0.011 | -3.90 | 0.000 | -0.003 | 0.137 |
| calcium, 20 dB | recurrent 20 | -0.011 | -4.15 | 0.000 | 0.003 | 0.170 |

**The July result is reproduced and improved**: with no measurement noise,
voltage deconvolved from calcium with a blind lambda gives R2_W 0.991, against
0.993 on the true voltage and July's 0.969 (oracle lambda, older code).

**Recurrent training helps at 40 dB and only there**: R2_W 0.149 -> 0.420 and
rollout r 0.517 -> 0.831, at a cost in R2_tau (0.31 -> 0.10). At 30 dB it lifts
the rollout and the clustering a little but recovers no connectivity; at 20 dB
nothing is recovered either way. On clean data it costs a little (R2_W 0.993 ->
0.979). The bottleneck below 40 dB is the deconvolution itself (median r(vdot)
0.16 at 30 dB, 0.10 at 20 dB), not the training objective.

## Why recurrent training

Deconvolution keeps the voltage and loses its derivative: in July r(v) was 0.98
at 40 dB and 0.96 at 20 dB, while r(vdot) was 0.44 and 0.17. One-step training
fits exactly that derivative; a 20-step rollout loss scores the voltage, which
survives.

## What differs

Every spec is the July spec with the dataset swapped; the recurrent arm adds
`recurrent_training: true`, `n_epochs: 20`, `data_augmentation_loop: 50`,
`rollout_horizon_schedule: 1..20` (experiments 3, 9, 10). 10 runs = 5 datasets
x 2 trainings, one fold (the source dataset has one).

## Specs

`experiments/specs/exp12/fly/flyvis_ca_<data>_<1s|rc20>.yaml`. Queue
`gpu_rtx6000`, wall 48 h.

<!-- STATUS:BEGIN -->

## Status

**10/10 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | data | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1s | true | 1 | 0.999 ± 0.000 | 0.999 ± 0.000 |  |  | 0.993 ± 0.000 (0.0) | 0.982 ± 0.000 (0.0) | 0.913 ± 0.000 (3.0) | 0.716 ± 0.000 | 0.981 ± 0.000 (0.0) | 0.033 ± 0.000 | 0.321 ± 0.000 | 0.895 ± 0.000 |
| 1s | inf | 1 | 0.974 ± 0.000 | 0.998 ± 0.000 |  |  | 0.991 ± 0.000 (0.0) | 0.954 ± 0.000 (0.2) | 0.894 ± 0.000 (3.8) | 0.739 ± 0.000 | 0.979 ± 0.000 (0.0) | 0.037 ± 0.000 | 0.328 ± 0.000 | 0.870 ± 0.000 |
| 1s | 40db | 1 | 0.483 ± 0.000 | 0.517 ± 0.000 |  |  | 0.149 ± 0.000 (0.5) | 0.314 ± 0.000 (7.3) | 0.731 ± 0.000 (58.9) | 0.722 ± 0.000 | 0.820 ± 0.000 (6.1) | 0.092 ± 0.000 | -0.689 ± 0.000 | 0.762 ± 0.000 |
| 1s | 30db | 1 | 0.073 ± 0.000 | 0.039 ± 0.000 |  |  | -0.008 ± 0.000 (0.3) |  | 0.587 ± 0.000 (69.5) | 0.581 ± 0.000 | 0.026 ± 0.000 (12.9) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.211 ± 0.000 |
| 1s | 20db | 1 |  | -0.003 ± 0.000 |  |  | -0.011 ± 0.000 (0.4) | -3.902 ± 0.000 (20.6) |  |  | -0.031 ± 0.000 (10.4) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.137 ± 0.000 |
| rc20 | true | 1 | 0.997 ± 0.000 | 0.997 ± 0.000 |  |  | 0.979 ± 0.000 (0.0) | 0.961 ± 0.000 (0.1) | 0.839 ± 0.000 (7.2) | 0.764 ± 0.000 | 0.976 ± 0.000 (0.0) | 0.028 ± 0.000 | 0.493 ± 0.000 | 0.856 ± 0.000 |
| rc20 | inf | 1 | 0.969 ± 0.000 | 0.997 ± 0.000 |  |  | 0.978 ± 0.000 (0.0) | 0.971 ± 0.000 (0.1) | 0.764 ± 0.000 (8.1) | 0.629 ± 0.000 | 0.971 ± 0.000 (0.1) | 0.040 ± 0.000 | 0.473 ± 0.000 | 0.848 ± 0.000 |
| rc20 | 40db | 1 | 0.432 ± 0.000 | 0.831 ± 0.000 |  |  | 0.420 ± 0.000 (0.1) | 0.103 ± 0.000 (4.9) | 0.609 ± 0.000 (59.7) | 0.612 ± 0.000 | 0.858 ± 0.000 (4.6) | 0.091 ± 0.000 | -0.796 ± 0.000 | 0.711 ± 0.000 |
| rc20 | 30db | 1 | 0.060 ± 0.000 | 0.116 ± 0.000 |  |  | -0.012 ± 0.000 (0.3) | 0.310 ± 0.000 (95.8) | 0.579 ± 0.000 (73.8) | 0.560 ± 0.000 | 0.045 ± 0.000 (11.1) | 0.007 ± 0.000 | -0.000 ± 0.000 | 0.442 ± 0.000 |
| rc20 | 20db | 1 |  | 0.003 ± 0.000 |  |  | -0.011 ± 0.000 (0.4) | -4.145 ± 0.000 (11.3) |  |  | -0.031 ± 0.000 (10.4) | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.170 ± 0.000 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | data | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_ca_true_1s` | landed | 1,520,001 | `b089e21e6998` |  |
| `flyvis_ca_inf_1s` | landed | 1,520,001 | `b089e21e6998` |  |
| `flyvis_ca_40db_1s` | landed | 1,520,001 | `b089e21e6998` |  |
| `flyvis_ca_30db_1s` | landed | 1,520,001 | `b089e21e6998` |  |
| `flyvis_ca_20db_1s` | landed | 1,520,001 | `b089e21e6998` |  |
| `flyvis_ca_true_rc20` | landed | 575,233 | `b089e21e6998` |  |
| `flyvis_ca_inf_rc20` | landed | 575,233 | `b089e21e6998` |  |
| `flyvis_ca_40db_rc20` | landed | 575,233 | `b089e21e6998` |  |
| `flyvis_ca_30db_rc20` | landed | 575,233 | `b089e21e6998` |  |
| `flyvis_ca_20db_rc20` | landed | 575,233 | `b089e21e6998` |  |

<!-- STATUS:END -->
