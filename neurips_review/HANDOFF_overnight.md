# Overnight handoff — 2026-07-26 13:47

## What is running

**Local, 2x A6000 (Vzfg Q1, drosophila CX ring attractor).** Six arms, 20 epochs
each, logs at `/tmp/cx_<config>.log`. ETA ~14:30–15:30.

| arm | pid | running R2_W (`conn`) |
|---|---|---|
| `nr2_cx_ring_s00` GNN | 1057417 | 0.587 |
| `nr2_cx_ring_s005` GNN | 1057419 | 0.916 |
| `nr2_cx_ring_s05` GNN | 1079611 | 1.000 |
| `nr2_cx_ring_s00_known_ode` | 1076845 | **-0.484, drifting** |
| `nr2_cx_ring_s005_known_ode` | 1076846 | 1.000 |
| `nr2_cx_ring_s05_known_ode` | 1079612 | 1.000 |

**Cluster, gpu_a100 (Vzfg Q2, calcium observation model).**

| job | config | stage |
|---|---|---|
| 153174323 | `nr2_ca_voltage_unified` | voltage control — **the gating run** |
| 153174324 | `nr2_ca_calcium_unified` | kernel only, GNN |
| 153174679/680 | `..._shot01_{unified,known_ode}` | + shot noise, gamma=0.1 matched |
| 153174681/682 | `..._shot02_{unified,known_ode}` | + shot noise, gamma=0.2 matched |
| 153174684 | `..._sat_unified` | + saturation |
| 153174692 | `..._sat_known_ode` | + saturation, oracle |
| 153174697 | `..._rate25_unified` | + 1/25 imaging rate |
| 153174698 | `..._full25_unified` | saturation + rate + shot |
| 153172779 | `nr2_cad_K10_unified_cv00` | R1 leftover, 29 h |

Already finished: `nr2_ca_calcium_known_ode` (kernel-only oracle) —
`W_corrected_R2 = 0.3305`, rollout r = 0.111.

## The collector

`overnight_collect.py`, pid 1089287, polls every 5 min until 05:47. Writes
`overnight_results.csv` and `overnight_status.md` every pass, so a dead session
loses nothing. Two deliberate properties:

- A run is terminal only on `_complete` **and** a parseable `metrics.txt`. An
  ssh failure is UNKNOWN, never done — this is the `finish_joint.py` bug that
  kept the joint rows out of `results_table.csv`.
- The three calcium oracle arms could not be submitted with their GNN twins:
  both would have generated the same dataset concurrently under `--force`. They
  go out on the twin's `_completed_generate` marker, guarded by
  `.submitted_<config>` files so a collector restart cannot double-submit.
  `rate25_known_ode` and `full25_known_ode` are still awaiting their datasets.

## Two things to look at first in the morning

**The sigma=0 CX oracle is not converging.** It went 0.788 -> 0.759 -> -0.484
over 45 min with both `freeze_known_ode_gain` and `freeze_known_ode_bias` set,
while sigma=0.05 and sigma=0.5 both sit at 1.000. Freezing fixed the other two
arms completely, so this is not the same degeneracy. The coherent reading is
that noise-free is genuinely ill-posed here — the same claim the paper makes for
Flyvis — but it should be reported as "did not converge" rather than quoted as
a number.

**The calcium arms may all be uninterpretable.** The kernel-only oracle sits at
rollout r = 0.111 and one-step r = 0.499: it barely predicts the observable it
was trained on. Until `nr2_ca_voltage_unified` comes back healthy we cannot say
whether `R2_W = 0.33` measures the observation model destroying information or
the pipeline failing to fit this observable. Do not write Q2 numbers into the
reply before reading the voltage control.

## Known gap in the observation model

`apply_exposure_step` implements the exposure integral, not decimation. On this
data the exposure barely moves the trace (the GCaMP kernel has already
low-passed it) but does attenuate the training target: dC/dt SD falls to 0.695
at D=5 and 0.503 at D=25. The sample-count reduction of a genuinely slower rate
is not modelled — that axis is the paper's existing 1/5-frames row, so the reply
should compose the two rather than claim the exposure arm covers both.
