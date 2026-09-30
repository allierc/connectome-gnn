---
number: 8
name: readout_sampling
title: 'The template readout four ways on the same trained models: before and after
  the second-pass fix, and one uniform random draw of 1,024 or 2,048 frames'
purpose: how much does the second-pass bug move experiment 2's conductance lasso-25
  numbers across the noise sweep, and does one plain random draw of frames (no active
  frame choice, no second pass) score the same models as well as the fixed readout
baseline: experiment 2's flyvis_<noise>_blank50_condl25_cv0N runs (trained models,
  not retrained)
specs_dir: experiments/specs/exp08/fly
task: plot
queue: gpu_rtx6000
wall: '4:00'
n_cpus: 12
axes:
  noise:
  - noise_free
  - noise_005
  - noise_05
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: rdold
  label: before the fix
  spec_pattern: flyvis_{noise}_blank50_condl25_rdold_{fold}
  model_from: flyvis_{noise}_blank50_condl25_{fold}
  code_dir: ../connectome-gnn-prefix
  differs_by:
    code: b8ca1372 (pre-fix)
- id: rdfixed
  label: after the fix
  spec_pattern: flyvis_{noise}_blank50_condl25_rdfixed_{fold}
  model_from: flyvis_{noise}_blank50_condl25_{fold}
  differs_by: {}
- id: rdu1024
  label: random 1,024
  spec_pattern: flyvis_{noise}_blank50_condl25_rdu1024_{fold}
  model_from: flyvis_{noise}_blank50_condl25_{fold}
  differs_by:
    recovery.template_frame_choice: uniform
    recovery.template_second_pass_frames: 0
    recovery.template_n_frames: 1024
- id: rdu2048
  label: random 2,048
  spec_pattern: flyvis_{noise}_blank50_condl25_rdu2048_{fold}
  model_from: flyvis_{noise}_blank50_condl25_{fold}
  differs_by:
    recovery.template_frame_choice: uniform
    recovery.template_second_pass_frames: 0
    recovery.template_n_frames: 2048
report:
  arm_labels:
    rdold: before fix
    rdfixed: after fix
    rdu1024: random 1,024
    rdu2048: random 2,048
  arm_order:
  - rdold
  - rdfixed
  - rdu1024
  - rdu2048
  metric_columns:
    tmpl_pct_fitted: edges fitted %
    tmpl_frames_used: frames
job_ids:
  flyvis_noise_005_blank50_condl25_rdold_cv00: '154461188'
  flyvis_noise_005_blank50_condl25_rdfixed_cv00: '154461189'
  flyvis_noise_005_blank50_condl25_rdu1024_cv00: '154461190'
  flyvis_noise_005_blank50_condl25_rdu2048_cv00: '154461191'
  flyvis_noise_free_blank50_condl25_rdold_cv00: '154461391'
  flyvis_noise_free_blank50_condl25_rdold_cv01: '154461392'
  flyvis_noise_free_blank50_condl25_rdold_cv02: '154461393'
  flyvis_noise_free_blank50_condl25_rdold_cv03: '154461394'
  flyvis_noise_free_blank50_condl25_rdold_cv04: '154461395'
  flyvis_noise_05_blank50_condl25_rdold_cv00: '154461396'
  flyvis_noise_05_blank50_condl25_rdold_cv01: '154461397'
  flyvis_noise_05_blank50_condl25_rdold_cv02: '154461398'
  flyvis_noise_05_blank50_condl25_rdold_cv03: '154461399'
  flyvis_noise_05_blank50_condl25_rdold_cv04: '154461401'
  flyvis_noise_free_blank50_condl25_rdfixed_cv00: '154461405'
  flyvis_noise_free_blank50_condl25_rdfixed_cv01: '154461406'
  flyvis_noise_free_blank50_condl25_rdfixed_cv02: '154461407'
  flyvis_noise_free_blank50_condl25_rdfixed_cv03: '154461408'
  flyvis_noise_free_blank50_condl25_rdfixed_cv04: '154461409'
  flyvis_noise_05_blank50_condl25_rdfixed_cv00: '154461410'
  flyvis_noise_05_blank50_condl25_rdfixed_cv01: '154461411'
  flyvis_noise_05_blank50_condl25_rdfixed_cv02: '154461412'
  flyvis_noise_05_blank50_condl25_rdfixed_cv03: '154461413'
  flyvis_noise_05_blank50_condl25_rdfixed_cv04: '154461414'
  flyvis_noise_free_blank50_condl25_rdu1024_cv00: '154461415'
  flyvis_noise_free_blank50_condl25_rdu1024_cv01: '154461416'
  flyvis_noise_free_blank50_condl25_rdu1024_cv02: '154461417'
  flyvis_noise_free_blank50_condl25_rdu1024_cv03: '154461418'
  flyvis_noise_free_blank50_condl25_rdu1024_cv04: '154461419'
  flyvis_noise_05_blank50_condl25_rdu1024_cv00: '154461420'
  flyvis_noise_05_blank50_condl25_rdu1024_cv01: '154461421'
  flyvis_noise_05_blank50_condl25_rdu1024_cv02: '154461422'
  flyvis_noise_05_blank50_condl25_rdu1024_cv03: '154461423'
  flyvis_noise_05_blank50_condl25_rdu1024_cv04: '154461424'
  flyvis_noise_free_blank50_condl25_rdu2048_cv00: '154461425'
  flyvis_noise_free_blank50_condl25_rdu2048_cv01: '154461426'
  flyvis_noise_free_blank50_condl25_rdu2048_cv02: '154461427'
  flyvis_noise_free_blank50_condl25_rdu2048_cv03: '154461428'
  flyvis_noise_free_blank50_condl25_rdu2048_cv04: '154461429'
  flyvis_noise_05_blank50_condl25_rdu2048_cv00: '154461430'
  flyvis_noise_05_blank50_condl25_rdu2048_cv01: '154461431'
  flyvis_noise_05_blank50_condl25_rdu2048_cv02: '154461432'
  flyvis_noise_05_blank50_condl25_rdu2048_cv03: '154461433'
  flyvis_noise_05_blank50_condl25_rdu2048_cv04: '154461434'
  flyvis_noise_005_blank50_condl25_rdold_cv01: '154461436'
  flyvis_noise_005_blank50_condl25_rdold_cv02: '154461437'
  flyvis_noise_005_blank50_condl25_rdold_cv03: '154461438'
  flyvis_noise_005_blank50_condl25_rdold_cv04: '154461439'
  flyvis_noise_005_blank50_condl25_rdfixed_cv01: '154461440'
  flyvis_noise_005_blank50_condl25_rdfixed_cv02: '154461441'
  flyvis_noise_005_blank50_condl25_rdfixed_cv03: '154461442'
  flyvis_noise_005_blank50_condl25_rdfixed_cv04: '154461443'
  flyvis_noise_005_blank50_condl25_rdu1024_cv01: '154461444'
  flyvis_noise_005_blank50_condl25_rdu1024_cv02: '154461445'
  flyvis_noise_005_blank50_condl25_rdu1024_cv03: '154461446'
  flyvis_noise_005_blank50_condl25_rdu1024_cv04: '154461447'
  flyvis_noise_005_blank50_condl25_rdu2048_cv01: '154461448'
  flyvis_noise_005_blank50_condl25_rdu2048_cv02: '154461449'
  flyvis_noise_005_blank50_condl25_rdu2048_cv03: '154461450'
  flyvis_noise_005_blank50_condl25_rdu2048_cv04: '154461451'
---

# Experiment 8 — readout_sampling

**The template readout four ways on the same trained models: before and after the second-pass fix, and one uniform random draw of 1,024 or 2,048 frames**

**Purpose.** how much does the second-pass bug move experiment 2's conductance lasso-25 numbers across the noise sweep, and does one plain random draw of frames (no active frame choice, no second pass) score the same models as well as the fixed readout

## Why

The template readout fits, per edge, msg = W·u·(E − v_i) + C by least squares
over sampled frames, keeping only frames where the sender is active
(u = relu(v_j) above the floor, the median of the positive activations), and
needs 8 such frames. Two things were found on 2026-09-28 (experiment 6):

- **The second pass had a bug.** It re-sampled the edges in a different
  shuffled order and added its per-edge sums by position, so each short edge
  (about 27% of edges) received another edge's sums. It inflated the fitted
  share to 99% and put a quarter of the edges into `R2_W` fitted on the wrong
  data. On `flyvis_noise_005_blank50_condl25_cv00` the fix moved `R2_W`
  0.971 -> 0.981 and the fitted share 99.3% -> 76.3%.
- **The frame choice is not agreed.** The default is "active" (a 256-frame
  uniform base plus frames drawn from the active windows of rarely-active
  senders, up to 1,024) with a 768-frame second pass. The proposal is one
  uniform random draw and one pass, which is simpler and has no second-pass
  bookkeeping to get wrong.

## What differs

Nothing is retrained. Each run links the checkpoint of the matching
experiment 2 run (`model_from`) into its own `models/` and runs `-o plot`, the
same pass that wrote experiment 2's `results/metrics.txt`. The four arms differ
only in the readout:

| arm | code | frame choice | first draw | second pass |
|---|---|---|---|---|
| before the fix | b8ca1372, from `../connectome-gnn-prefix` | active | 256 base, up to 1,024 | 768, misaligned |
| after the fix | working tree | active | 256 base, up to 1,024 | 768 |
| random 1,024 | working tree | uniform | 1,024 | none |
| random 2,048 | working tree | uniform | 2,048 | none |

`one-step r` and `rollout r` are blank: they come from `-o test`, which does
not involve the readout. Every other column does, except `R2_tau`, `R2_msg`
and `k_i`, which come from the per-neuron update fit and should agree across
arms to the GPU's run-to-run noise (about 2e-7).

`R2_E` is not reported: the data is current-form flyvis, which has no
reversal potential to recover.

## Canary (noise_005, cv00), 2026-09-28

The four arms were run on one model first. "Before the fix" reproduced that
run's experiment 2 `results/metrics.txt` exactly (`Wij_R2` 0.971725 on 431,193
edges, 99.33% fitted, 25.98% rescued by the second pass), which is the check
that it ran the old code. Its `_commit` reads `unknown`: the checkout's `.git`
pointer is a devcontainer path the cluster cannot resolve, so it is b8ca1372 by
construction, not by record. The other three record `b8ca1372-dirty` (the fix
is uncommitted).

| arm | `Wij_R2` | edges fitted | frames | rescued by 2nd pass |
|---|---|---|---|---|
| before the fix | 0.9717 | 99.33% | 1,024 | 25.98% |
| after the fix | 0.9824 | 76.28% | 1,024 | 2.93% |
| random 1,024 | 0.9823 | 76.39% | 1,024 | -- |
| random 2,048 | 0.9821 | 78.39% | 2,048 | -- |

`R2_tau` (0.9814) and `k_i` (0.338) are identical across the four, as they
should be. Each job took 13-14 min.

## Results (60/60 landed, 2026-09-28)

Mean over the 5 folds (SD in the status table below). `R2_tau`, `R2_msg` and
`k_i` are identical in all four arms at every noise level, which is the check
that only the per-edge readout changed.

| noise | before the fix | after the fix | random 1,024 | random 2,048 |
|---|---|---|---|---|
| **`R2_W`** | | | | |
| noise_free | 0.898 | 0.939 | 0.934 | 0.934 |
| noise_005 | 0.974 | 0.989 | 0.989 | 0.989 |
| noise_05 | 0.985 | 0.991 | 0.991 | 0.991 |
| **edges fitted, % of 434,112** | | | | |
| noise_free | 99.3 | 76.2 | 76.8 | 77.6 |
| noise_005 | 99.3 | 76.5 | 76.9 | 78.4 |
| noise_05 | 100.0 | 96.3 | 96.2 | 96.4 |
| **`R2_Vrest`** | | | | |
| noise_free | 0.873 | 0.826 | 0.825 | 0.828 |
| noise_005 | 0.897 | 0.905 | 0.903 | 0.903 |
| noise_05 | 0.887 | 0.887 | 0.887 | 0.886 |

**The bug cost `R2_W` 0.006-0.041**, most at noise_free, where the most edges
were short and so the most were fitted on another edge's data. Experiment 2's
page-14 row for the conductance lasso 25 reads 0.90 / 0.97 / 0.98; on the fixed
readout it is 0.94 / 0.99 / 0.99.

**One uniform random draw scores these models as well as the active choice
plus the fixed second pass**: `R2_W` within 0.005 of it at every noise level,
and the same coverage. Doubling the draw to 2,048 frames adds 0.8-1.9 points of
coverage and nothing to `R2_W`.

**Coverage is set by the data, not the readout.** A quarter of the edges have
a sender that is almost never above the floor in the noise-free and
noise_005 recordings; at noise_05 the process noise drives most senders over
it and 96% of edges are fitted by every readout.

**`R2_Vrest` at noise_free falls 0.873 -> 0.826 with the fix**, in all three
fixed arms alike, the one number the fix makes worse. Not yet explained. A
candidate, unverified: the V_rest correction adds back each neuron's total
per-edge offset C, and the 24% of edges now unfitted contribute nothing to that
total, where before they contributed another edge's C, which is about the
right size.

The fit roll of the conductance form stays diverged in every arm (r 0.11-0.13
with 70-79% of neuron-frames on the clamp), so that column's conclusion in
experiment 2 does not depend on the readout.

## Specs

60 = 4 arms x 3 noise levels x 5 folds, in `experiments/specs/exp08/fly/`,
named `flyvis_<noise>_blank50_condl25_<arm>_cv0N`. Each is experiment 2's
`flyvis_<noise>_blank50_condl25_cv0N` with its description and `config_file`
changed and, for the two random arms, a `recovery:` section added; verified by
parsing both.

<!-- STATUS:BEGIN -->

## Status

**60/60 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | noise | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster | edges fitted % | frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rdold | noise_free | 5 |  |  |  |  | 0.898 ± 0.007 (0.1) | 0.902 ± 0.044 (4.1) | 0.873 ± 0.022 (19.6) | 0.774 ± 0.027 | 0.940 ± 0.012 (0.2) | 0.039 ± 0.008 | 0.387 ± 0.010 | 0.778 ± 0.046 | 99.267 ± 0.040 | 1024.000 ± 0.000 |
| rdold | noise_005 | 5 |  |  |  |  | 0.974 ± 0.002 (0.0) | 0.980 ± 0.004 (0.0) | 0.897 ± 0.017 (3.5) | 0.903 ± 0.017 | 0.990 ± 0.002 (0.0) | 0.009 ± 0.003 | 0.340 ± 0.003 | 0.895 ± 0.013 | 99.339 ± 0.031 | 1024.000 ± 0.000 |
| rdold | noise_05 | 5 |  |  |  |  | 0.985 ± 0.001 (0.0) | 0.997 ± 0.001 (0.0) | 0.887 ± 0.021 (2.0) | 0.924 ± 0.016 | 0.995 ± 0.001 (0.0) | 0.012 ± 0.006 | 0.173 ± 0.226 | 0.886 ± 0.004 | 100.000 ± 0.000 | 1024.000 ± 0.000 |
| rdfixed | noise_free | 5 |  |  |  |  | 0.939 ± 0.008 (0.0) | 0.902 ± 0.044 (4.1) | 0.826 ± 0.016 (19.2) | 0.774 ± 0.027 | 0.940 ± 0.012 (0.2) | 0.029 ± 0.007 | 0.387 ± 0.010 | 0.778 ± 0.046 | 76.207 ± 0.162 | 1024.000 ± 0.000 |
| rdfixed | noise_005 | 5 |  |  |  |  | 0.989 ± 0.005 (0.0) | 0.980 ± 0.004 (0.0) | 0.905 ± 0.014 (3.3) | 0.903 ± 0.017 | 0.990 ± 0.002 (0.0) | 0.006 ± 0.001 | 0.340 ± 0.003 | 0.895 ± 0.013 | 76.541 ± 0.238 | 1024.000 ± 0.000 |
| rdfixed | noise_05 | 5 |  |  |  |  | 0.991 ± 0.001 (0.0) | 0.997 ± 0.001 (0.0) | 0.887 ± 0.021 (2.0) | 0.924 ± 0.016 | 0.995 ± 0.001 (0.0) | 0.011 ± 0.005 | 0.173 ± 0.226 | 0.886 ± 0.004 | 96.333 ± 0.010 | 1024.000 ± 0.000 |
| rdu1024 | noise_free | 5 |  |  |  |  | 0.934 ± 0.010 (0.0) | 0.902 ± 0.044 (4.1) | 0.825 ± 0.018 (19.3) | 0.774 ± 0.027 | 0.940 ± 0.012 (0.2) | 0.029 ± 0.006 | 0.387 ± 0.010 | 0.768 ± 0.044 | 76.777 ± 0.185 | 1024.000 ± 0.000 |
| rdu1024 | noise_005 | 5 |  |  |  |  | 0.989 ± 0.004 (0.0) | 0.980 ± 0.004 (0.0) | 0.903 ± 0.017 (3.1) | 0.903 ± 0.017 | 0.990 ± 0.002 (0.0) | 0.006 ± 0.001 | 0.340 ± 0.003 | 0.895 ± 0.013 | 76.930 ± 0.298 | 1024.000 ± 0.000 |
| rdu1024 | noise_05 | 5 |  |  |  |  | 0.991 ± 0.001 (0.0) | 0.997 ± 0.001 (0.0) | 0.887 ± 0.022 (2.1) | 0.924 ± 0.016 | 0.995 ± 0.001 (0.0) | 0.011 ± 0.006 | 0.173 ± 0.226 | 0.886 ± 0.004 | 96.168 ± 0.017 | 1024.000 ± 0.000 |
| rdu2048 | noise_free | 5 |  |  |  |  | 0.934 ± 0.010 (0.0) | 0.902 ± 0.044 (4.1) | 0.828 ± 0.020 (19.2) | 0.774 ± 0.027 | 0.940 ± 0.012 (0.2) | 0.029 ± 0.006 | 0.387 ± 0.010 | 0.768 ± 0.044 | 77.635 ± 0.067 | 2048.000 ± 0.000 |
| rdu2048 | noise_005 | 5 |  |  |  |  | 0.989 ± 0.004 (0.0) | 0.980 ± 0.004 (0.0) | 0.903 ± 0.017 (3.1) | 0.903 ± 0.017 | 0.990 ± 0.002 (0.0) | 0.006 ± 0.001 | 0.340 ± 0.003 | 0.895 ± 0.013 | 78.385 ± 0.095 | 2048.000 ± 0.000 |
| rdu2048 | noise_05 | 5 |  |  |  |  | 0.991 ± 0.001 (0.0) | 0.997 ± 0.001 (0.0) | 0.886 ± 0.022 (2.0) | 0.924 ± 0.016 | 0.995 ± 0.001 (0.0) | 0.011 ± 0.006 | 0.173 ± 0.226 | 0.886 ± 0.004 | 96.440 ± 0.008 | 2048.000 ± 0.000 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | noise | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster | edges fitted % | frames |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `flyvis_noise_free_blank50_condl25_rdold_cv00` | landed |  | `unknown` |  |
| `flyvis_noise_free_blank50_condl25_rdold_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_free_blank50_condl25_rdold_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_free_blank50_condl25_rdold_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_free_blank50_condl25_rdold_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_005_blank50_condl25_rdold_cv00` | landed |  | `unknown` |  |
| `flyvis_noise_005_blank50_condl25_rdold_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_005_blank50_condl25_rdold_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_005_blank50_condl25_rdold_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_005_blank50_condl25_rdold_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_05_blank50_condl25_rdold_cv00` | landed |  | `unknown` |  |
| `flyvis_noise_05_blank50_condl25_rdold_cv01` | landed |  | `unknown` |  |
| `flyvis_noise_05_blank50_condl25_rdold_cv02` | landed |  | `unknown` |  |
| `flyvis_noise_05_blank50_condl25_rdold_cv03` | landed |  | `unknown` |  |
| `flyvis_noise_05_blank50_condl25_rdold_cv04` | landed |  | `unknown` |  |
| `flyvis_noise_free_blank50_condl25_rdfixed_cv00` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdfixed_cv01` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdfixed_cv02` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdfixed_cv03` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdfixed_cv04` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdfixed_cv00` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdfixed_cv01` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdfixed_cv02` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdfixed_cv03` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdfixed_cv04` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdfixed_cv00` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdfixed_cv01` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdfixed_cv02` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdfixed_cv03` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdfixed_cv04` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu1024_cv00` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu1024_cv01` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu1024_cv02` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu1024_cv03` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu1024_cv04` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu1024_cv00` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu1024_cv01` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu1024_cv02` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu1024_cv03` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu1024_cv04` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu1024_cv00` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu1024_cv01` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu1024_cv02` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu1024_cv03` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu1024_cv04` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu2048_cv00` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu2048_cv01` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu2048_cv02` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu2048_cv03` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_free_blank50_condl25_rdu2048_cv04` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu2048_cv00` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu2048_cv01` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu2048_cv02` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu2048_cv03` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_005_blank50_condl25_rdu2048_cv04` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu2048_cv00` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu2048_cv01` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu2048_cv02` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu2048_cv03` | landed |  | `b8ca1372f4bb` |  |
| `flyvis_noise_05_blank50_condl25_rdu2048_cv04` | landed |  | `b8ca1372f4bb` |  |

<!-- STATUS:END -->
