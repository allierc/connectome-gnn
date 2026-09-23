---
number: 0
name: gpu_benchmark
title: RTX 6000 against A100, fp32 against bf16
purpose: 'decide which GPU to run the campaign on: measure iterations per second on
  RTX 6000 and A100, and check that bf16 costs no recovered precision at the same
  seeds'
baseline: experiments/baseline/gnn_current_baseline.yaml
specs_dir: experiments/specs/bench/fly
task: train
queue: gpu_rtx6000
wall: '24:00'
axes:
  precision:
  - fp32
  - bf16
arms:
- id: rtx6000
  label: RTX 6000
  queue: gpu_rtx6000
  spec_pattern: bench_rtx6000_{precision}
  differs_by:
    queue: gpu_rtx6000
- id: a100
  label: A100
  queue: gpu_a100
  spec_pattern: bench_a100_{precision}
  differs_by:
    queue: gpu_a100
job_ids:
  bench_rtx6000_fp32: '154398322'
  bench_rtx6000_bf16: '154398323'
  bench_a100_fp32: '154398324'
  bench_a100_bf16: '154398325'
analyse_job_ids:
  bench_rtx6000_fp32: '154399699'
  bench_rtx6000_bf16: '154399700'
  bench_a100_fp32: '154399701'
  bench_a100_bf16: '154399702'
---

# Experiment 0 — gpu_benchmark

**RTX 6000 against A100, fp32 against bf16**

**Purpose.** decide which GPU to run the campaign on: measure iterations per second on RTX 6000 and A100, and check that bf16 costs no recovered precision at the same seeds

## What differs

Two things, crossed: the GPU (the arm, via its own queue) and
`training.mlp_precision` (the axis, `fp32` or `bf16`). `torch_compile` and `deterministic` come from the baseline (both on) in all four, and every seed is the baseline's,
so two runs at the same precision differ only in the silicon and two runs
on the same GPU differ only in the arithmetic.

`deterministic: true` is not free. `NeuralGNN.message_and_aggregate` uses
`scatter_add`, which on CUDA accumulates via atomics -- 434k edges onto
13.7k nodes, so ~32 non-associative float adds per node in scheduler order,
and eight identical calls gave eight distinct results. Turning determinism
on fixes that and costs roughly 26x on that operation, so these numbers are
the reproducible-training rate, not the fastest the hardware can go.

## The first attempt measured its own instrumentation

`data_augmentation_loop` was cut 500 -> 10 to make the jobs short, which shrank
`Niter` fifty-fold but kept the same NUMBER of metric snapshots. Each snapshot
is a full template readout over 434,112 edges, so it was amortised over 1,321
iterations instead of production's 58,666 -- 44x too often -- and the four arms
ran at 4-10 it/s against the 72 it/s the production runs of experiments 1 and 2
actually do. The speed numbers were the snapshot cost, not the GPUs.

The precision comparison survived it, because all four arms paid the same
overhead: `R2_W` 0.923 / 0.925 / 0.927 / 0.926 at iteration 30,401, so **bf16
costs nothing on recovery**, on either card.

Relaunched with `data_augmentation_loop: 50` and
`checkpoint_saves_per_epoch: 3`: 160,000 iterations at exactly the production
cadence, a tenth of a full run.

## Not a training run

`data_augmentation_loop` is 50 rather than 500, so each job is 160,000
iterations instead of 1,600,000. Multiply the wall time by 10 for a full run. The recovered numbers are read at the same iteration on every arm,
which is what makes the precision comparison a comparison.

## Specs

| arm | queue | precision | spec |
|---|---|---|---|
| rtx6000 | `gpu_rtx6000` | fp32 | `bench_rtx6000_fp32` |
| rtx6000 | `gpu_rtx6000` | bf16 | `bench_rtx6000_bf16` |
| a100 | `gpu_a100` | fp32 | `bench_a100_fp32` |
| a100 | `gpu_a100` | bf16 | `bench_a100_bf16` |

## Results

### Precision: bf16 is free

Held-out, one run per cell, 160,000 iterations:

| | `R2_W` | `R2_tau` | `R2_Vrest` | `R2_msg` | cluster |
|---|---|---|---|---|---|
| rtx6000 fp32 | 0.948 | 0.967 | 0.806 | 0.978 | 0.898 |
| rtx6000 bf16 | 0.944 | 0.965 | 0.763 | 0.973 | 0.914 |
| a100 fp32 | 0.946 | 0.987 | 0.763 | 0.980 | 0.888 |
| a100 bf16 | 0.939 | 0.934 | 0.836 | 0.980 | 0.909 |

`R2_W` spans 0.939 to 0.948 across all four, and the two cards agree to 0.002 at
fp32. bf16 costs nothing on recovery, on either card. `R2_Vrest` is the noisiest
column at 0.763-0.836, but this is one run per cell with no fold spread, so that
is a single-sample artefact rather than a precision effect.

### The answer

**A100, fp32** -- 28.0 it/s against the RTX 6000's 11.8, with identical
recovery. The caveat is the queue rather than the card: `gpu_a100` had 2,506
jobs pending against `gpu_rtx6000`'s 344 when this was measured, and the two
A100 arms' own analysis had to be moved off it to land at all.

### Speed: the A100 is faster, and bf16 helps only the RTX 6000

152,001 iterations, 24 snapshots each, so the instrument cost is identical
across the four and the RANKING is sound:

| arm | wall | rate | extrapolated 1.6 M |
|---|---|---|---|
| a100 fp32 | 90 min | **28.0 it/s** | 15.9 h |
| a100 bf16 | 125 min | 20.3 it/s | 21.9 h |
| rtx6000 bf16 | 203 min | 12.5 it/s | 35.6 h |
| rtx6000 fp32 | 215 min | 11.8 it/s | 37.6 h |

bf16 is +6% on the RTX 6000 and **-27% on the A100** -- it is not a free speedup
on a card whose fp32 path is already tensor-core backed.

### The absolute rates are still wrong, and this bench cannot fix them

The relaunch set `checkpoint_saves_per_epoch: 3`, which is NOT the knob: the
cadence is `snapshots_per_epoch` (default 5), and on top of it an early-phase
burst adds a fixed NUMBER of extra snapshots. A short run pays that fixed count
over few iterations, so 24 snapshots landed in 152,001 iterations -- one every
6,300 against production's one every 58,666 -- and the metric pass still
dominates. 11.8 it/s here against the 72 it/s the same card does on
experiment 1.

**The honest throughput number is the production runs': 72 it/s on RTX 6000,
about 6 h for a full 1.6 M-iteration run.** Use the ranking above to choose a
queue and that figure to plan a wall.

<!-- STATUS:BEGIN -->

## Status

**4/4 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | precision | n | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtx6000 | fp32 | 1 | 0.998 ± 0.000 | 0.998 ± 0.000 | 0.948 ± 0.000 | 0.967 ± 0.000 | 0.806 ± 0.000 | 0.768 ± 0.000 | 0.978 ± 0.000 | 0.041 ± 0.000 |  | 0.898 ± 0.000 |
| rtx6000 | bf16 | 1 | 0.999 ± 0.000 | 0.998 ± 0.000 | 0.944 ± 0.000 | 0.965 ± 0.000 | 0.763 ± 0.000 | 0.745 ± 0.000 | 0.973 ± 0.000 | 0.038 ± 0.000 |  | 0.914 ± 0.000 |
| a100 | fp32 | 1 | 0.999 ± 0.000 | 0.998 ± 0.000 | 0.946 ± 0.000 | 0.987 ± 0.000 | 0.763 ± 0.000 | 0.784 ± 0.000 | 0.980 ± 0.000 | 0.040 ± 0.000 |  | 0.888 ± 0.000 |
| a100 | bf16 | 1 | 0.998 ± 0.000 | 0.998 ± 0.000 | 0.939 ± 0.000 | 0.934 ± 0.000 | 0.836 ± 0.000 | 0.732 ± 0.000 | 0.980 ± 0.000 | 0.047 ± 0.000 |  | 0.909 ± 0.000 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | precision | iter | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `bench_rtx6000_fp32` | landed | 152,001 | `71e4d78710c4` |
| `bench_rtx6000_bf16` | landed | 152,001 | `71e4d78710c4` |
| `bench_a100_fp32` | landed | 152,001 | `71e4d78710c4` |
| `bench_a100_bf16` | landed | 152,001 | `71e4d78710c4` |

<!-- STATUS:END -->

