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
  bench_rtx6000_fp32: '154395993'
  bench_rtx6000_bf16: '154395994'
  bench_a100_fp32: '154395995'
  bench_a100_bf16: '154395996'
---

# Experiment 0 — gpu_benchmark

**RTX 6000 against A100, fp32 against bf16**

**Purpose.** decide which GPU to run the campaign on: measure iterations per second on RTX 6000 and A100, and check that bf16 costs no recovered precision at the same seeds

## What differs

Two things, crossed: the GPU (the arm, via its own queue) and
`training.mlp_precision` (the axis, `fp32` or `bf16`). `torch_compile` and
`deterministic` are **on** in all four, and every seed is the baseline's,
so two runs at the same precision differ only in the silicon and two runs
on the same GPU differ only in the arithmetic.

`deterministic: true` is not free. `NeuralGNN.message_and_aggregate` uses
`scatter_add`, which on CUDA accumulates via atomics -- 434k edges onto
13.7k nodes, so ~32 non-associative float adds per node in scheduler order,
and eight identical calls gave eight distinct results. Turning determinism
on fixes that and costs roughly 26x on that operation, so these numbers are
the reproducible-training rate, not the fastest the hardware can go.

## Not a training run

`data_augmentation_loop` is 10 rather than 500, so each job is 32,000
iterations instead of 1,600,000. Multiply the wall time by 50 for a full
run. The recovered numbers are read at the same iteration on every arm,
which is what makes the precision comparison a comparison.

## Specs

| arm | queue | precision | spec |
|---|---|---|---|
| rtx6000 | `gpu_rtx6000` | fp32 | `bench_rtx6000_fp32` |
| rtx6000 | `gpu_rtx6000` | bf16 | `bench_rtx6000_bf16` |
| a100 | `gpu_a100` | fp32 | `bench_a100_fp32` |
| a100 | `gpu_a100` | bf16 | `bench_a100_bf16` |

<!-- STATUS:BEGIN -->

## Status

**0/4 landed**, 2 running, 2 pending

### Landed --- held-out, `results/metrics.txt`

| arm | precision | n | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | precision | iter | one-step r | rollout r | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtx6000 | bf16 | 14,401 |  | 0.995 ± 0.000 | 0.922 ± 0.000 | 0.936 ± 0.000 | 0.707 ± 0.000 | 0.625 ± 0.000 | 0.964 ± 0.000 |  |  | 0.875 ± 0.000 |
| a100 | bf16 | 1,281 |  | 0.936 ± 0.000 | 0.692 ± 0.000 | 0.140 ± 0.000 | 0.338 ± 0.000 | 0.381 ± 0.000 | 0.923 ± 0.000 |  |  |  |

### Per run

| run | status | iter | commit |
|---|---|---|---|
| `bench_rtx6000_fp32` | pending |  | `` |
| `bench_rtx6000_bf16` | running | 14,401 | `` |
| `bench_a100_fp32` | pending |  | `` |
| `bench_a100_bf16` | running | 1,281 | `` |

<!-- STATUS:END -->

