---
number: 6
name: conductance_memory
title: Which cluster node trains the conductance lasso-25 model on the hybrid connectomes
  without running out of memory
purpose: assess which cluster node is necessary to train the conductance lasso-25
  model on the four hybrid connectome variants without reaching out-of-memory, like
  experiment 0 but for memory
baseline: GraphData/config/fly/<variant>_blank50_flywire_cv00.yaml (the published
  GNN specs)
specs_dir: experiments/specs/exp06/fly
task: train
queue: gpu_rtx6000
wall: '12:00'
n_cpus: 12
axes:
  variant:
  - e8_flywireRF_noise_005
  - e8_flywireRF_proximal_nulls_noise_005
  - full_eye_flywireRF_noise_005
  - full_eye_flywireRF_proximal_nulls_noise_005
arms:
- id: a100
  label: A100 SXM4 80 GB
  queue: gpu_a100
  spec_pattern: '{variant}_blank50_condl25mem_a100_cv00'
  differs_by:
    queue: gpu_a100
- id: h100
  label: H100 80 GB
  queue: gpu_h100
  spec_pattern: '{variant}_blank50_condl25mem_h100_cv00'
  differs_by:
    queue: gpu_h100
- id: rtx6000
  label: RTX PRO 6000 95.5 GB
  queue: gpu_rtx6000
  spec_pattern: '{variant}_blank50_condl25mem_rtx6000_cv00'
  differs_by:
    queue: gpu_rtx6000
report:
  memory: true
  axis_labels:
    variant:
      e8_flywireRF_noise_005: e8 hybrid
      e8_flywireRF_proximal_nulls_noise_005: e8 hybrid + n.e.
      full_eye_flywireRF_noise_005: FlyWire eye
      full_eye_flywireRF_proximal_nulls_noise_005: FlyWire eye + n.e.
job_ids:
  e8_flywireRF_noise_005_blank50_condl25mem_a100_cv00: '154460640'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_a100_cv00: '154460999'
  full_eye_flywireRF_noise_005_blank50_condl25mem_a100_cv00: '154461000'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_a100_cv00: '154461001'
  e8_flywireRF_noise_005_blank50_condl25mem_h100_cv00: '154460645'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_h100_cv00: '154461002'
  full_eye_flywireRF_noise_005_blank50_condl25mem_h100_cv00: '154461003'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_h100_cv00: '154461004'
  e8_flywireRF_noise_005_blank50_condl25mem_rtx6000_cv00: '154460650'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_rtx6000_cv00: '154461005'
  full_eye_flywireRF_noise_005_blank50_condl25mem_rtx6000_cv00: '154461006'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_rtx6000_cv00: '154461007'
analyse_job_ids:
  e8_flywireRF_noise_005_blank50_condl25mem_a100_cv00: '154461008'
  e8_flywireRF_noise_005_blank50_condl25mem_h100_cv00: '154461009'
  e8_flywireRF_noise_005_blank50_condl25mem_rtx6000_cv00: '154461010'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_a100_cv00: '154463209'
  full_eye_flywireRF_noise_005_blank50_condl25mem_a100_cv00: '154463210'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_h100_cv00: '154463211'
  full_eye_flywireRF_noise_005_blank50_condl25mem_h100_cv00: '154463212'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_rtx6000_cv00: '154463214'
  full_eye_flywireRF_noise_005_blank50_condl25mem_rtx6000_cv00: '154463215'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_a100_cv00: '154465663'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_h100_cv00: '154465664'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_rtx6000_cv00: '154465665'
---

# Experiment 6 — conductance_memory

**Which cluster node trains the conductance lasso-25 model on the hybrid connectomes without running out of memory**

**Purpose.** assess which cluster node is necessary to train the conductance lasso-25 model on the four hybrid connectome variants without reaching out-of-memory, like experiment 0 but for memory

## Why

Experiment 7 fills the GNN and Known ODE rows of the hybrid-connectome table
(the poster's `poster_flybrid_inliers.tex`, built from
`figures/aggregate_flywireRF_table.py`) with the conductance lasso-25 model. Its
largest variant, the FlyWire eye with null edges, is 50,412 neurons and 9,642,335
edges -- 22x experiment 2's connectome in edges and 3.7x in neurons -- and the
conductance form doubles `g_phi`'s per-edge input (6 columns against 3). Five
folds of that on the wrong node is five 48-hour jobs lost, so this measures
first.

## What the nodes offer

| queue | card | GPU memory | slots per host |
|---|---|---|---|
| `gpu_a100` | A100 SXM4 | 80 GB | 48 |
| `gpu_h100` | H100 | 79.6 GB | 96 |
| `gpu_rtx6000` | RTX PRO 6000 | 95.5 GB | 128 |

**The A100 and the H100 have the same memory**, so for out-of-memory purposes
they should behave alike; the H100 is faster, not bigger. The RTX PRO 6000 has
more than either, and is the card every other experiment in this campaign ran
on. H200 (140 GB) and B300 (268 GB) exist and are the fallback if all three
fail.

## Two ways to run out, measured separately

- **GPU.** Nothing recorded it until now. `GNN_Main.py` writes
  `<run>/gpu_memory.log` after training, after the test and after the plot:
  the card, its total memory, and torch's peak reserved memory for that phase.
  A CUDA out-of-memory shows in the error log and the outcome column says so.
- **Host.** LSF allocates 20 GB of host RAM per slot, and experiment 4 died of
  `TERM_MEMLIMIT` at 8 slots on a graph a quarter this size. Every run asks for **12 slots, 240 GB** (see below), and
  LSF's `Max Memory` reports its peak -- which sets experiment 7's slot count.

## The slot request, revised at launch

The design asked for 32 slots per run. `bsub` on `gpu_a100` answered: *"The
average ratio for this queue is 12 slots per gpu. Your job has requested more
slots than the average ratio ... Your job submission will be delayed."* It does
not refuse, it stalls the submission for several minutes per job, and a job
holding 32 of a 48-slot host starves the other GPUs on it. So:

- `e8_flywireRF_noise_005` on `gpu_a100` (job `154460640`) was already through
  at 32 slots and keeps them -- one uncapped host-memory reading;
- the other 11 run at **12 slots, 240 GB**, the queue's own ratio.

A run that needs more host RAM than that dies of `TERM_MEMLIMIT` with LSF's
`Max Memory` at the cap, which is itself the answer, and only that run is
relaunched with more slots.

## What differs

Each spec is the published `<variant>_blank50_flywire_cv00` GNN spec with four
keys changed, verified by parsing both:

| key | published | here |
|---|---|---|
| `graph_model.signal_model_name` | `<variant>` (an alias of the current-form model) | `flyvis_conductance` |
| `graph_model.input_size` | 3 | 6 |
| `training.coeff_g_phi_input_group_L1` | -- | 25 |
| `training.data_augmentation_loop` | 500 | **5** |

The last makes each run 1% of a full one. Memory does not depend on how long a
run trains, only on what it holds at once; a 1% run still reaches the
training-time readout snapshots and then the `-o test_plot` analysis, which runs
the template readout over every edge and is the likeliest place to peak. Fold
cv00 only: memory does not depend on the fold.

## Results so far: the readout runs out, not the training

**Eight of twelve died, all at the same line, and on all three cards.** Each
died of a CUDA out-of-memory in the template readout at the first training
checkpoint (`metrics.py:832`, `sample_g_phi_vi_vj_observed`), before its
first logged iteration:

| variant | edges | A100 80 GB | H100 80 GB | RTX PRO 6000 95 GB |
|---|---|---|---|---|
| e8 hybrid | 327,358 | running | running | running |
| e8 hybrid + n.e. | 2,418,403 | OOM | OOM | OOM (asked 9.23 GiB) |
| FlyWire eye | 1,266,378 | OOM (asked 4.83 GiB) | OOM (asked 4.83 GiB) | running |
| FlyWire eye + n.e. | 9,642,335 | OOM (asked 36.78 GiB) | OOM (asked 36.78 GiB) | OOM (asked 36.78 GiB) |

The failed request is one `(edges x frames)` float32 array: 36.78 GiB over
9,642,335 edges is 4,096 bytes per edge, i.e. 1,024 frames (the readout's 256
base frames, which the active-frame choice grows up to 4x). The same holds on
the other two variants, so the readout's memory is **linear in the edge count
and independent of the card**.

What the readout holds at once, read from the code rather than measured: on the
GPU `vi`, `vj` and the `g_phi` output (one `(E, F)` array each) and the two
embedding copies `ai_flat`, `aj_flat` (`(E, F, 2)` each), so at least 7 x E x F x
4 bytes; the chunking at `G_PHI_EVAL_CHUNK` covers only the MLP's hidden
activations. The caller (`extract_template_params`) then copies them to the host
as float64 and builds about ten more `(E, F)` arrays before reducing them to
per-edge sums:

| variant | readout on GPU | readout on host |
|---|---|---|
| e8 hybrid | ~9 GiB | ~25 GB |
| e8 hybrid + n.e. | ~65 GiB | ~180 GB |
| FlyWire eye | ~34 GiB | ~95 GB |
| FlyWire eye + n.e. | ~257 GiB | ~700 GB |

That is on top of what training keeps resident: 56-78 GiB was already allocated
when the request failed, 5-19 GiB of it in torch.compile's CUDA-graph pools.

**So the node question has no answer for the two null-edge variants as the code
stands.** FlyWire eye + n.e. would need more GPU than a B300 has (268 GB) and
roughly three times the host RAM a 12-slot job gets (240 GB). A larger card does
not fix it; the readout has to stream.

**The fix is to chunk the readout over edges.** The fit already reduces
everything to per-edge sums (`_sums`, the normal equations), so a block of edges
at a time gives the same sums. The global quantities (the activity floor, a
quantile over all (edge, frame) pairs, and the 99th percentile of |v_i|) turned
out not to need the pairs at all: see the next section. With the readout
streaming, the node is decided by training alone.

## The readout now streams (2026-09-28)

`extract_template_params` sums a block of edges at a time
(`TEMPLATE_BLOCK_PAIRS` = 2^26 (edge, frame) pairs per block, about 2 GB on the
GPU and 7 GB on the host), whatever the graph. The activity floor and the 99th
percentile of |v_i| are computed exactly without the (edges x frames) array: on
an edge the activation is its sender's, so the floor is the out-degree-weighted
quantile of the (frames x neurons) activations, and |v_i|'s the in-degree-weighted
one; `_pooled_quantile` reproduces `np.quantile` to the bit, numpy's dtype rules
included (`tests/test_template_readout_stream.py`).

Checked on `flyvis_noise_005_blank50_condl25_cv00` (434,112 edges, 1,024 frames),
old code against new on the same checkpoint, second pass off: **every per-edge
quantity bit-identical** (W before the gauge, E_ij, the fit R2s, the offsets).
The per-neuron gauge, tau and V_rest differ by up to 2e-7, and so do two runs of
the OLD code against each other: that is the GPU's scatter-add order in the
update fit, untouched here. The streamed readout took 40 s against 94 s, at
12.3 GiB peak GPU against 23.1 GiB.

**It also fixed a bug in the second pass.** Each second-pass chunk called the
sampler, which re-permutes the edges from its own seed, and its sums were added
slot by slot onto the first pass's: a short edge received the frames of
whichever edge the new permutation put in its slot (4.6e-6 of slots coincide).
Mostly a busy edge, which is why nearly every short edge crossed the 8-row
minimum. On the same run with the production second pass (768 frames):

| | old | fixed |
|---|---|---|
| edges rescued by the second pass | 25.98% | 2.93% |
| edges fitted | 99.33% (431,193) | 76.28% (331,147) |
| `Wij_R2` | 0.971 | 0.981 |
| `Wij_rel_err_median` | 0.237 | 0.170 |
| `V_rest_R2` | 0.879 | 0.881 |

So a quarter of the edges in every template readout since the second pass was
added (c7499d1f) were fitted on another edge's data. The test that catches it
recovers E_ij on a fixture whose message is the template: the old code got 8 of
26 fitted edges wrong, the fixed one 0 of 40.

## Relaunched on the streamed readout

The 8 runs that died and the one still running on the old code (FlyWire eye on
RTX PRO 6000, killed) were relaunched on the new code: 9 runs = 3 cards x the
3 large variants. The 3 e8 hybrid runs finished training and are in `-o
test_plot`, which now also uses the streamed readout.

## The answer (2026-09-29, all 12 landed)

With the readout streaming, every variant trains, tests and plots on all three
cards. Peak GPU in training: e8 hybrid 23 GB, e8 hybrid + n.e. 25 GB, FlyWire
eye 46 GB, FlyWire eye + n.e. 70-76 GB (the plot pass peaks at 32 GB, the test
at 11 GB). Host RAM peaks at 105 GB for both FlyWire eye variants. A100 and H100
fit FlyWire eye + n.e. with only 3-9 GB to spare; the RTX PRO 6000 has 19 GB.
**Experiment 7 runs on `gpu_rtx6000` with 12 slots.** The it/s column is
dominated by the checkpoint readouts at 1% of a run's length and is not a
production rate.

## Specs

12 = 3 GPU arms x 4 variants, in `experiments/specs/exp06/fly/`, named
`<variant>_blank50_condl25mem_<queue>_cv00`.

<!-- STATUS:BEGIN -->

## Status

**12/12 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 0 pending

### Landed --- held-out, `results/metrics.txt`

| arm | variant | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a100 | e8_flywireRF_noise_005 | 1 | 0.996 ± 0.000 | 0.997 ± 0.000 |  |  | 0.962 ± 0.000 (0.0) | 0.976 ± 0.000 (0.8) | 0.720 ± 0.000 (12.9) | 0.634 ± 0.000 | 0.942 ± 0.000 (0.0) | 0.073 ± 0.000 | 0.950 ± 0.000 | 0.731 ± 0.000 |
| a100 | e8_flywireRF_proximal_nulls_noise_005 | 1 | 0.983 ± 0.000 | 0.986 ± 0.000 |  |  | 0.888 ± 0.000 (0.0) | 0.925 ± 0.000 (1.6) | 0.563 ± 0.000 (38.4) | 0.573 ± 0.000 | 0.800 ± 0.000 (3.0) | 0.111 ± 0.000 | 0.533 ± 0.000 | 0.611 ± 0.000 |
| a100 | full_eye_flywireRF_noise_005 | 1 | 0.996 ± 0.000 | 0.997 ± 0.000 |  |  | 0.952 ± 0.000 (0.0) | 0.928 ± 0.000 (1.5) | 0.753 ± 0.000 (13.4) | 0.650 ± 0.000 | 0.940 ± 0.000 (0.0) | 0.080 ± 0.000 | 1.742 ± 0.000 | 0.700 ± 0.000 |
| a100 | full_eye_flywireRF_proximal_nulls_noise_005 | 1 | 0.974 ± 0.000 | 0.978 ± 0.000 |  |  | 0.868 ± 0.000 (0.0) | 0.930 ± 0.000 (2.9) | 0.600 ± 0.000 (37.7) | 0.507 ± 0.000 | 0.828 ± 0.000 (2.8) | 0.089 ± 0.000 | 0.003 ± 0.000 | 0.531 ± 0.000 |
| h100 | e8_flywireRF_noise_005 | 1 | 0.996 ± 0.000 | 0.997 ± 0.000 |  |  | 0.962 ± 0.000 (0.0) | 0.975 ± 0.000 (0.6) | 0.728 ± 0.000 (12.7) | 0.620 ± 0.000 | 0.942 ± 0.000 (0.0) | 0.076 ± 0.000 | 0.943 ± 0.000 | 0.707 ± 0.000 |
| h100 | e8_flywireRF_proximal_nulls_noise_005 | 1 | 0.984 ± 0.000 | 0.989 ± 0.000 |  |  | 0.901 ± 0.000 (0.0) | 0.938 ± 0.000 (1.4) | 0.523 ± 0.000 (37.7) | 0.516 ± 0.000 | 0.799 ± 0.000 (2.9) | 0.108 ± 0.000 | 0.517 ± 0.000 | 0.590 ± 0.000 |
| h100 | full_eye_flywireRF_noise_005 | 1 | 0.996 ± 0.000 | 0.997 ± 0.000 |  |  | 0.952 ± 0.000 (0.0) | 0.951 ± 0.000 (3.1) | 0.763 ± 0.000 (14.3) | 0.646 ± 0.000 | 0.938 ± 0.000 (0.0) | 0.079 ± 0.000 | 1.716 ± 0.000 | 0.727 ± 0.000 |
| h100 | full_eye_flywireRF_proximal_nulls_noise_005 | 1 | 0.972 ± 0.000 | 0.977 ± 0.000 |  |  | 0.862 ± 0.000 (0.0) | 0.928 ± 0.000 (4.0) | 0.589 ± 0.000 (36.6) | 0.552 ± 0.000 | 0.822 ± 0.000 (2.7) | 0.076 ± 0.000 | 0.037 ± 0.000 | 0.529 ± 0.000 |
| rtx6000 | e8_flywireRF_noise_005 | 1 | 0.996 ± 0.000 | 0.997 ± 0.000 |  |  | 0.959 ± 0.000 (0.0) | 0.969 ± 0.000 (0.7) | 0.719 ± 0.000 (11.9) | 0.632 ± 0.000 | 0.943 ± 0.000 (0.0) | 0.076 ± 0.000 | 1.007 ± 0.000 | 0.736 ± 0.000 |
| rtx6000 | e8_flywireRF_proximal_nulls_noise_005 | 1 | 0.982 ± 0.000 | 0.985 ± 0.000 |  |  | 0.888 ± 0.000 (0.0) | 0.921 ± 0.000 (1.1) | 0.568 ± 0.000 (37.8) | 0.546 ± 0.000 | 0.779 ± 0.000 (3.7) | 0.121 ± 0.000 | 0.502 ± 0.000 | 0.588 ± 0.000 |
| rtx6000 | full_eye_flywireRF_noise_005 | 1 | 0.996 ± 0.000 | 0.997 ± 0.000 |  |  | 0.952 ± 0.000 (0.0) | 0.929 ± 0.000 (1.6) | 0.756 ± 0.000 (13.8) | 0.652 ± 0.000 | 0.939 ± 0.000 (0.0) | 0.076 ± 0.000 | 1.810 ± 0.000 | 0.733 ± 0.000 |
| rtx6000 | full_eye_flywireRF_proximal_nulls_noise_005 | 1 | 0.973 ± 0.000 | 0.975 ± 0.000 |  |  | 0.859 ± 0.000 (0.0) | 0.917 ± 0.000 (2.2) | 0.564 ± 0.000 (39.1) | 0.508 ± 0.000 | 0.825 ± 0.000 (3.0) | 0.073 ± 0.000 | -0.129 ± 0.000 | 0.542 ± 0.000 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | variant | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Memory --- GPU peak reserved per phase, host peak from LSF

| arm | variant | GPU | card GB | train GB | test GB | plot GB | host GB | it/s | outcome |
|---|---|---|---|---|---|---|---|---|---|
| a100 | e8_flywireRF_noise_005 | A100-SXM4-80GB | 79.3 | 22.8 | 1.0 | 6.5 | 37.6 | 8.7 | ok |
| a100 | e8_flywireRF_proximal_nulls_noise_005 | A100-SXM4-80GB | 79.3 | 24.8 | 2.8 | 6.7 | 21.8 | 2.6 | ok |
| a100 | full_eye_flywireRF_noise_005 | A100-SXM4-80GB | 79.3 | 46.1 | 3.6 | 25.1 | 103.3 | 3.0 | ok |
| a100 | full_eye_flywireRF_proximal_nulls_noise_005 | A100-SXM4-80GB | 79.3 | 75.8 | 10.6 | 32.1 | 105.7 | 1.3 | ok |
| h100 | e8_flywireRF_noise_005 | H100 80GB HBM3 | 79.2 | 22.8 | 1.0 | 6.5 | 37.4 | 8.2 | ok |
| h100 | e8_flywireRF_proximal_nulls_noise_005 | H100 80GB HBM3 | 79.2 | 24.8 | 2.8 | 6.7 | 27.9 | 2.5 | ok |
| h100 | full_eye_flywireRF_noise_005 | H100 80GB HBM3 | 79.2 | 46.0 | 3.6 | 25.1 | 104.3 | 2.6 | ok |
| h100 | full_eye_flywireRF_proximal_nulls_noise_005 | H100 80GB HBM3 | 79.2 | 70.5 | 10.6 | 32.1 | 105.0 | 1.1 | ok |
| rtx6000 | e8_flywireRF_noise_005 | RTX PRO 6000 Blackwell Server Edition | 95.1 | 22.9 | 1.0 | 6.5 | 37.7 | 8.9 | ok |
| rtx6000 | e8_flywireRF_proximal_nulls_noise_005 | RTX PRO 6000 Blackwell Server Edition | 95.1 | 25.0 | 2.8 | 6.7 | 31.0 | 2.5 | ok |
| rtx6000 | full_eye_flywireRF_noise_005 | RTX PRO 6000 Blackwell Server Edition | 95.1 | 45.5 | 3.6 | 25.1 | 104.4 | 3.0 | ok |
| rtx6000 | full_eye_flywireRF_proximal_nulls_noise_005 | RTX PRO 6000 Blackwell Server Edition | 95.1 | 76.1 | 10.6 | 32.1 | 104.7 | 1.3 | ok |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `e8_flywireRF_noise_005_blank50_condl25mem_a100_cv00` | landed | 15,201 | `b8ca1372f4bb` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_a100_cv00` | landed | 15,201 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_noise_005_blank50_condl25mem_a100_cv00` | landed | 15,201 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_a100_cv00` | landed | 30,401 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_condl25mem_h100_cv00` | landed | 15,201 | `b8ca1372f4bb` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_h100_cv00` | landed | 15,201 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_noise_005_blank50_condl25mem_h100_cv00` | landed | 15,201 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_h100_cv00` | landed | 30,401 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_condl25mem_rtx6000_cv00` | landed | 15,201 | `b8ca1372f4bb` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_rtx6000_cv00` | landed | 15,201 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_noise_005_blank50_condl25mem_rtx6000_cv00` | landed | 15,201 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25mem_rtx6000_cv00` | landed | 30,401 | `b8ca1372f4bb` |  |

<!-- STATUS:END -->
