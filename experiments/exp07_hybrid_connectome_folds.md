---
number: 7
name: hybrid_connectome_folds
title: The hybrid-connectome table with the conductance lasso-25 model in the GNN
  rows
purpose: refill the GNN and Known ODE rows of the hybrid-connectome table (tab:zero_edge_inliers,
  flybrid_inliers.tex) over five folds on the current code, with the conductance lasso-25
  model in place of the current-form GNN; the ML and EED rows stay as published
baseline: GraphData/config/fly/<variant>_blank50_flywire_cv0N.yaml and <variant>_blank50_flywire_known_ode_cv0N.yaml
  (the published specs)
specs_dir: experiments/specs/exp07/fly
task: train
queue: gpu_rtx6000
wall: '120:00'
n_cpus: 12
axes:
  variant:
  - e8_flywireRF_noise_005
  - e8_flywireRF_proximal_nulls_noise_005
  - full_eye_flywireRF_noise_005
  - full_eye_flywireRF_proximal_nulls_noise_005
  fold:
  - cv00
  - cv01
  - cv02
  - cv03
  - cv04
arms:
- id: kode
  label: Known ODE
  spec_pattern: '{variant}_blank50_kode_{fold}'
  differs_by: {}
- id: condl25
  label: conductance, group lasso 25
  spec_pattern: '{variant}_blank50_condl25_{fold}'
  differs_by:
    graph_model.signal_model_name: flyvis_conductance
    graph_model.input_size: 6
    training.coeff_g_phi_input_group_L1: 25.0
report:
  axis_labels:
    variant:
      e8_flywireRF_noise_005: e8 hybrid
      e8_flywireRF_proximal_nulls_noise_005: e8 hybrid + n.e.
      full_eye_flywireRF_noise_005: FlyWire eye
      full_eye_flywireRF_proximal_nulls_noise_005: FlyWire eye + n.e.
job_ids:
  e8_flywireRF_noise_005_blank50_kode_cv00: '154460872'
  e8_flywireRF_noise_005_blank50_kode_cv01: '154460873'
  e8_flywireRF_noise_005_blank50_kode_cv02: '154460874'
  e8_flywireRF_noise_005_blank50_kode_cv03: '154460875'
  e8_flywireRF_noise_005_blank50_kode_cv04: '154460876'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv00: '154460877'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv01: '154460878'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv02: '154460879'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv03: '154460880'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv04: '154460881'
  full_eye_flywireRF_noise_005_blank50_kode_cv00: '154460882'
  full_eye_flywireRF_noise_005_blank50_kode_cv01: '154460883'
  full_eye_flywireRF_noise_005_blank50_kode_cv02: '154460884'
  full_eye_flywireRF_noise_005_blank50_kode_cv03: '154460885'
  full_eye_flywireRF_noise_005_blank50_kode_cv04: '154460886'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv00: '154460887'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv01: '154460889'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv02: '154460890'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv03: '154460892'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv04: '154460893'
  e8_flywireRF_noise_005_blank50_condl25_cv00: '154460907'
  e8_flywireRF_noise_005_blank50_condl25_cv01: '154460908'
  e8_flywireRF_noise_005_blank50_condl25_cv02: '154460909'
  e8_flywireRF_noise_005_blank50_condl25_cv03: '154460910'
  e8_flywireRF_noise_005_blank50_condl25_cv04: '154460911'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00: '154469750'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv01: '154469751'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv02: '154469752'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv03: '154469753'
  e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv04: '154469754'
  full_eye_flywireRF_noise_005_blank50_condl25_cv00: '154469755'
  full_eye_flywireRF_noise_005_blank50_condl25_cv01: '154469756'
  full_eye_flywireRF_noise_005_blank50_condl25_cv02: '154469757'
  full_eye_flywireRF_noise_005_blank50_condl25_cv03: '154469758'
  full_eye_flywireRF_noise_005_blank50_condl25_cv04: '154469759'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00: '154469760'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv01: '154469761'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv02: '154469763'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv03: '154469765'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv04: '154469767'
analyse_job_ids:
  e8_flywireRF_noise_005_blank50_kode_cv00: '154463268'
  e8_flywireRF_noise_005_blank50_kode_cv01: '154463269'
  e8_flywireRF_noise_005_blank50_kode_cv02: '154463270'
  e8_flywireRF_noise_005_blank50_kode_cv03: '154463271'
  e8_flywireRF_noise_005_blank50_kode_cv04: '154463272'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv00: '154463273'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv01: '154463274'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv02: '154463275'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv03: '154463276'
  e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv04: '154463277'
  full_eye_flywireRF_noise_005_blank50_kode_cv00: '154463278'
  full_eye_flywireRF_noise_005_blank50_kode_cv01: '154463279'
  full_eye_flywireRF_noise_005_blank50_kode_cv02: '154463280'
  full_eye_flywireRF_noise_005_blank50_kode_cv03: '154463281'
  full_eye_flywireRF_noise_005_blank50_kode_cv04: '154463282'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv00: '154463283'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv01: '154463284'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv02: '154463285'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv03: '154463286'
  full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv04: '154463287'
  e8_flywireRF_noise_005_blank50_condl25_cv00: '154465673'
  e8_flywireRF_noise_005_blank50_condl25_cv01: '154465674'
  e8_flywireRF_noise_005_blank50_condl25_cv02: '154465675'
  e8_flywireRF_noise_005_blank50_condl25_cv03: '154465676'
  e8_flywireRF_noise_005_blank50_condl25_cv04: '154465677'
---

# Experiment 7 — hybrid_connectome_folds

**The hybrid-connectome table with the conductance lasso-25 model in the GNN rows**

**Purpose.** refill the GNN and Known ODE rows of the hybrid-connectome table (tab:zero_edge_inliers, flybrid_inliers.tex) over five folds on the current code, with the conductance lasso-25 model in place of the current-form GNN; the ML and EED rows stay as published

## Why

The published table was aggregated by `figures/aggregate_flywireRF_table.py`
from the five-fold runs archived in `log/fly/archive_4`. Its GNN rows are the
current-form model (`<variant>`, an alias of `flyvis_A`). Experiments 2, 4 and
5 settled on the conductance form under a group lasso of 25 as the general
model, so the GNN rows are re-run with it. The Known ODE rows are re-run
unchanged, so both rows of the table come from the same code and the same
readout. The ML and EED rows are not trained models and stay as published.

## What differs

Each conductance spec is the published `<variant>_blank50_flywire_cv0N` GNN spec
with three keys changed, verified by parsing both:

| key | published | here |
|---|---|---|
| `graph_model.signal_model_name` | `<variant>` (an alias of the current-form model) | `flyvis_conductance` |
| `graph_model.input_size` | 3 | 6 |
| `training.coeff_g_phi_input_group_L1` | -- | 25 |

Each Known ODE spec is the published `<variant>_blank50_flywire_known_ode_cv0N`
spec with nothing changed but its description and `config_file`, verified the
same way.

## Two batches now, one held

Experiment 6 found that the conductance model's template readout at the
first training checkpoint runs out of GPU memory on the three large variants,
on every card up to the 95 GB RTX PRO 6000, because it holds arrays of size
edges x frames for every edge at once. So:

- **Launched: all 20 Known ODE runs.** The template readout runs only for a GNN
  with a message MLP `g_phi` (`template_readout_enabled`), and the Known ODE has
  none. Its published runs took 6-20 min of training and at most 102 GB of
  host RAM (FlyWire eye + n.e.), inside the 240 GB of 12 slots.
- **Launched: the 5 conductance runs on e8 hybrid** (327,358 edges), which
  trained past the readout on all three cards in experiment 6.
- **Held: the 15 conductance runs on e8 hybrid + n.e., FlyWire eye and FlyWire
  eye + n.e.**, until the readout streams over edges and experiment 6 has
  measured what training alone needs on those graphs.

Queue `gpu_rtx6000`: the fastest card in experiment 0 and the largest card in
experiment 6. Wall 24 h: the conductance lasso-25 model on flyvis (434,112
edges) took 6.3 h for 1.52 M iterations on this queue in experiment 2, and e8
hybrid is smaller.

## The readout these runs are scored with

The template readout was changed on 2026-09-28, after these jobs were
submitted (experiment 6): it streams over edges, and its second pass now adds
each short edge's own frames instead of another edge's. Training-time readouts
of the conductance runs already running use the old code, so their
`tmp_training/` logs are on the old readout; `results/metrics.txt` comes from
`-o test_plot`, submitted after the change, and is on the new one. The Known
ODE runs have no template readout (`readout: chain`, W read directly).

**Every row is analysed at commit 9ef188e6 or later**: one uniform random draw
of 1,024 frames, no second pass (experiment 8), and the neuron panels drawn from
the final checkpoint. The 20 Known ODE runs had been analysed at b8ca1372; they
were re-analysed (`analyse 7 --redo pre_readout_fix --arm kode`, old
`metrics.txt` kept in `<run>/superseded/pre_readout_fix/`) so the table comes
from one commit, although their metrics do not depend on the template readout.

## The held batch, launched 2026-09-29

Experiment 6 (streamed readout) put every variant through training, test and
plot on all three cards. The training peak for FlyWire eye + n.e. is 70-76 GB,
within 3-9 GB of an A100's or H100's 80 GB and 19 GB under the RTX PRO 6000's
95 GB; host RAM peaks at 105 GB. So the 15 conductance runs on the three large
variants run on `gpu_rtx6000` with 12 slots (240 GB). The wall is raised to
120 h: the published current-form FlyWire eye + n.e. run never finished inside
24 h on an H100 (segments of 23.5 h and 18.7 h, both killed), and the queue
allows up to 14 days.

## Specs

40 = 2 arms x 4 variants x 5 folds, in `experiments/specs/exp07/fly/`, named
`<variant>_blank50_kode_cv0N` and `<variant>_blank50_condl25_cv0N`.

<!-- STATUS:BEGIN -->

## Status

**25/40 landed**, 0 trained (awaiting `-o test_plot`), 0 running, 15 pending

### Landed --- held-out, `results/metrics.txt`

| arm | variant | n | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kode | e8_flywireRF_noise_005 | 5 | 0.999 ± 0.000 | 0.999 ± 0.000 |  |  | 0.914 ± 0.000 (0.1) | 0.969 ± 0.000 (0.6) | 0.960 ± 0.000 (9.4) |  | 0.958 ± 0.005 (0.4) |  |  | 0.771 ± 0.021 |
| kode | e8_flywireRF_proximal_nulls_noise_005 | 5 | 0.999 ± 0.000 | 0.998 ± 0.000 |  |  | 0.899 ± 0.001 (0.0) | 0.971 ± 0.000 (0.1) | 0.900 ± 0.001 (6.9) |  | 0.982 ± 0.002 (0.0) |  |  | 0.682 ± 0.014 |
| kode | full_eye_flywireRF_noise_005 | 5 | 0.999 ± 0.000 | 0.999 ± 0.000 |  |  | 0.917 ± 0.000 (0.1) | 0.972 ± 0.000 (0.4) | 0.962 ± 0.000 (9.1) |  | 0.963 ± 0.005 (0.3) |  |  | 0.768 ± 0.011 |
| kode | full_eye_flywireRF_proximal_nulls_noise_005 | 5 | 0.998 ± 0.000 | 0.996 ± 0.000 |  |  | 0.890 ± 0.001 (0.0) | 0.970 ± 0.000 (0.5) | 0.858 ± 0.001 (8.5) |  | 0.976 ± 0.002 (0.0) |  |  | 0.670 ± 0.012 |
| condl25 | e8_flywireRF_noise_005 | 5 | 0.998 ± 0.000 | 0.998 ± 0.001 |  |  | 0.984 ± 0.007 (0.0) | 0.976 ± 0.008 (0.3) | 0.870 ± 0.016 (6.4) | 0.862 ± 0.033 | 0.984 ± 0.003 (0.0) | 0.008 ± 0.004 | 0.235 ± 0.010 | 0.730 ± 0.043 |

### Running --- train split, `tmp_training/`, blank where not written per checkpoint

| arm | variant | iter | one-step r | rollout r | fit roll r current form | fit roll r conductance form | R2_W | R2_tau | R2_Vrest | R2_Vrest noC | R2_msg | C_i | k_i | cluster |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | | | | | | | | | | | | | | |

### Per run

| run | status | iter | commit | LSF |
|---|---|---|---|---|
| `e8_flywireRF_noise_005_blank50_kode_cv00` | landed | 304,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_kode_cv01` | landed | 304,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_kode_cv02` | landed | 304,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_kode_cv03` | landed | 304,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_kode_cv04` | landed | 304,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv00` | landed | 38,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv01` | landed | 38,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv02` | landed | 38,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv03` | landed | 38,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_kode_cv04` | landed | 38,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_noise_005_blank50_kode_cv00` | landed | 304,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_noise_005_blank50_kode_cv01` | landed | 304,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_noise_005_blank50_kode_cv02` | landed | 304,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_noise_005_blank50_kode_cv03` | landed | 304,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_noise_005_blank50_kode_cv04` | landed | 304,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv00` | landed | 38,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv01` | landed | 38,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv02` | landed | 38,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv03` | landed | 38,001 | `b8ca1372f4bb` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_kode_cv04` | landed | 38,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_condl25_cv00` | landed | 1,520,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_condl25_cv01` | landed | 1,520,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_condl25_cv02` | landed | 1,520,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_condl25_cv03` | landed | 1,520,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_noise_005_blank50_condl25_cv04` | landed | 1,520,001 | `b8ca1372f4bb` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00` | pending |  | `` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv01` | pending |  | `` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv02` | pending |  | `` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv03` | pending |  | `` |  |
| `e8_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv04` | pending |  | `` |  |
| `full_eye_flywireRF_noise_005_blank50_condl25_cv00` | pending |  | `` |  |
| `full_eye_flywireRF_noise_005_blank50_condl25_cv01` | pending |  | `` |  |
| `full_eye_flywireRF_noise_005_blank50_condl25_cv02` | pending |  | `` |  |
| `full_eye_flywireRF_noise_005_blank50_condl25_cv03` | pending |  | `` |  |
| `full_eye_flywireRF_noise_005_blank50_condl25_cv04` | pending |  | `` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv00` | pending |  | `` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv01` | pending |  | `` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv02` | pending |  | `` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv03` | pending |  | `` |  |
| `full_eye_flywireRF_proximal_nulls_noise_005_blank50_condl25_cv04` | pending |  | `` |  |

<!-- STATUS:END -->
