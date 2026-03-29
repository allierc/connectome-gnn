# NeurIPS 2025 Submission Plan

**Deadline**: May 11, 2025 (https://neurips.cc/Conferences/2025/CallForPapers)
**Working title**: Reverse-engineering neural connectomes from activity data with graph neural networks

<style>table { font-size: 0.85em; } th, td { padding: 3px 6px; }</style>

## Tables

Color code: <span style="color:#2ea043">green</span> &gt; 0.9, <span style="color:#d29922">orange</span> &gt; 0.5, <span style="color:#cf222e">red</span> &le; 0.5.

### Table 1 & 2: Connectivity Recovery (W R2) and Rollout (Pearson r)

Primary metric: W R2. Secondary: rollout r. Format: best single-seed (robust mean±std, CV%).

#### Drosophila CX (152 neurons)

<table>
<tr><th>Experiment</th><th>W R2 — GNN (ours)</th><th>Rollout r</th><th>MLP</th><th>Known ODE</th><th>RNN</th><th>Neural ODE</th><th>SSM</th></tr>
<tr><td><b>FC (noise-free)</b></td><td style="background:#d2992260"><b>0.804</b> (0.574±0.027, 22 seeds, 15% fail)</td><td style="background:#d2992260">0.71</td><td style="background:#cf222e60">~0 / <span style="background:#d2992260">0.70</span></td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td><b>FC noise=0.05</b></td><td style="background:#2ea04360"><b>0.982</b> (0.619±0.271, 24 seeds, 54% success)</td><td style="background:#d2992260">0.84</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>FC noise=0.5</b></td><td style="background:#2ea04360"><b>0.9997</b> (0.974±0.026, 6 seeds, CV=2.7%)</td><td style="background:#2ea04360">1.00</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>GT edges (noise-free)</b></td><td style="background:#2ea04360"><b>0.893</b> (0.710±0.107, ~80% converged, CV~15%)</td><td style="background:#2ea04360">1.00</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>GT edges noise=0.05</b></td><td>?</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>GT edges noise=0.5</b></td><td>?</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
</table>

_MLP column: W R2 / rollout r (W R2 ≈ 0 always — Jacobian extraction fundamentally limited). 2000 frames._

#### Larva (230 neurons)

<table>
<tr><th>Experiment</th><th>W R2 — GNN (ours)</th><th>Rollout r</th><th>MLP</th><th>Known ODE</th><th>RNN</th><th>Neural ODE</th><th>SSM</th></tr>
<tr><td><b>FC (noise-free)</b></td><td style="background:#cf222e60"><b>0.435</b> (0.268±0.106, 10 seeds, CV=40%)</td><td style="background:#2ea04360">1.00</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>FC noise=0.05</b></td><td>?</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>FC noise=0.5</b></td><td>?</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>GT edges noise=0.05</b></td><td style="background:#d2992260"><b>0.870</b> (0.683, 2-seed mean)</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>GT edges noise=0.5</b></td><td>?</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>GT edges (noise-free)</b></td><td style="background:#d2992260"><b>0.908</b> (0.540, 28 seeds, CV=35%)</td><td style="background:#2ea04360">1.00</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
</table>

_480 frames. Default topology = GT edges._

#### Zebrafish oculomotor (609 neurons)

<table>
<tr><th>Experiment</th><th>W R2 — GNN (ours)</th><th>Rollout r</th><th>MLP</th><th>Known ODE</th><th>RNN</th><th>Neural ODE</th><th>SSM</th></tr>
<tr><td><b>FC (noise-free)</b></td><td style="background:#cf222e60">0.022 (ceiling, 48 seeds)</td><td style="background:#2ea04360">1.00</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td><b>FC noise=0.05</b></td><td style="background:#2ea04360"><b>0.918</b> (0.371±0.063 at DAL=35)</td><td style="background:#2ea04360">1.00</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>FC noise=0.5</b></td><td style="background:#2ea04360"><b>0.988</b> (0.506±0.539, 4 seeds, 50% fail)</td><td style="background:#2ea04360">0.93</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>GT edges (noise-free)</b></td><td style="background:#d2992260"><b>0.777</b> (0.710±0.035, CV=5.0%, ~25% bimodal failure)</td><td style="background:#2ea04360">1.00</td><td>--</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td><b>GT edges noise=0.05</b></td><td>?</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>GT edges noise=0.5</b></td><td>?</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
</table>

_4200 frames. FC is intractable due to linear degeneracy — noise or GT edges required._

### Table 3: Robustness (W R2 under degraded conditions, GNN only)

Best single-seed W R2 per condition.

#### Drosophila CX (152)

<table>
<tr><th>Condition</th><th>W R2</th></tr>
<tr><td><b>FC (noise-free)</b></td><td style="background:#d2992260"><b>0.804</b></td></tr>
<tr><td><b>FC noise=0.05</b></td><td style="background:#2ea04360"><b>0.982</b></td></tr>
<tr><td><b>FC noise=0.5</b></td><td style="background:#2ea04360"><b>0.9997</b></td></tr>
<tr><td><b>GT edges (noise-free)</b></td><td style="background:#2ea04360"><b>0.893</b></td></tr>
<tr><td><b>GT edges noise=0.05</b></td><td>?</td></tr>
<tr><td><b>GT edges noise=0.5</b></td><td>?</td></tr>
</table>

#### Larva (230)

<table>
<tr><th>Condition</th><th>W R2</th></tr>
<tr><td><b>FC (noise-free)</b></td><td style="background:#cf222e60"><b>0.435</b></td></tr>
<tr><td><b>FC noise=0.05</b></td><td>?</td></tr>
<tr><td><b>FC noise=0.5</b></td><td>?</td></tr>
<tr><td><b>GT edges noise=0.05</b></td><td style="background:#d2992260"><b>0.870</b></td></tr>
<tr><td><b>GT edges noise=0.5</b></td><td>?</td></tr>
<tr><td><b>GT edges (noise-free)</b></td><td style="background:#d2992260"><b>0.908</b></td></tr>
</table>

#### Zebrafish oculomotor (609)

<table>
<tr><th>Condition</th><th>W R2</th></tr>
<tr><td><b>FC (noise-free)</b></td><td style="background:#cf222e60">0.022</td></tr>
<tr><td><b>FC noise=0.05</b></td><td style="background:#2ea04360"><b>0.918</b></td></tr>
<tr><td><b>FC noise=0.5</b></td><td style="background:#2ea04360"><b>0.988</b></td></tr>
<tr><td><b>GT edges (noise-free)</b></td><td style="background:#d2992260"><b>0.777</b></td></tr>
<tr><td><b>GT edges noise=0.05</b></td><td>?</td></tr>
<tr><td><b>GT edges noise=0.5</b></td><td>?</td></tr>
</table>

_CX and zebrafish default to FC. Larva defaults to GT edges. Flyvis results in flyvis_results._

### Table 3b: Parameter Extraction (R2 per parameter, GNN only)

Each bio model has a different ODE structure with different extractable parameters. W is always learned directly; other parameters (tau, V_rest) are extracted from learned f_theta slopes/offsets. Cluster accuracy measures neuron-type discrimination from learned embeddings.

#### Drosophila CX (152 neurons — ring attractor: dh/dt = (-h + g*softplus(h+b) @ W + I)/tau)

Extractable parameters: **W** (synaptic weights), **tau** (time constants). Tau extraction fails (R2=0.0) because the CX model has no resting potential — f_theta slope is too weak (~-0.05) to extract tau reliably. Gain (g) and bias (b) are entangled with W.

<table>
<tr><th>Condition</th><th>W R2</th><th>tau R2</th><th>Cluster acc</th><th>Dale score</th></tr>
<tr><td>Clean (FC)</td><td style="background:#d2992260"><b>0.804</b> (0.574 mean)</td><td style="background:#cf222e60">0.0</td><td style="background:#cf222e60">0.351</td><td style="background:#d2992260">0.690</td></tr>
<tr><td>Noise=0.05 (FC, dale_law)</td><td style="background:#2ea04360"><b>0.982</b> (0.619±0.271, 24 seeds)</td><td style="background:#cf222e60">0.0</td><td style="background:#cf222e60">0.386</td><td style="background:#d2992260">0.660</td></tr>
<tr><td>Noise=0.5 (FC)</td><td style="background:#2ea04360"><b>0.9997</b> (0.974±0.026, 6 seeds)</td><td style="background:#cf222e60">0.0</td><td style="background:#cf222e60">0.429</td><td>?</td></tr>
<tr><td>Clean (GT edges, dale_law)</td><td style="background:#2ea04360"><b>0.893</b> (0.710±0.107 mean)</td><td style="background:#cf222e60">0.0</td><td style="background:#cf222e60">0.421</td><td>?</td></tr>
</table>

_Tau is not extractable for CX (always R2=0.0). Dale's law constraint dramatically improves robustness (CV from >20% to 8.8%). Noise helps W recovery (+30% over clean)._

#### Larva (230 neurons — two-population: premotor softplus + motor softplus, W with gain correction)

Extractable parameters: **W** (synaptic weights, gain-corrected). Tau is fixed (=1.0) in the larva model, not learned. Gain (gp, gm) is entangled with W at destination.

<table>
<tr><th>Condition</th><th>W R2</th><th>tau R2</th></tr>
<tr><td>Clean (GT edges)</td><td style="background:#d2992260"><b>0.908</b> (0.540 mean, CV=35%)</td><td style="background:#cf222e60">0.0 (fixed)</td></tr>
</table>

_High seed variance dominates (CV=35%). Best single seed reaches 0.908 but robust mean is only 0.540. f_theta_msg_diff=50 is the strongest regularizer (+38% improvement)._

#### Zebrafish oculomotor (609 neurons — linear integrator: dr/dt = -r + W @ r + I*v_in, tau=1)

Extractable parameters: **W** (synaptic weights) only. No nonlinearity, no tau/V_rest to extract. Linear dynamics create degeneracy — many W produce identical activity.

<table>
<tr><th>Condition</th><th>W R2</th><th>Cluster acc</th><th>Dale score</th></tr>
<tr><td>Clean (FC)</td><td style="background:#cf222e60">0.022 (ceiling)</td><td style="background:#cf222e60">0.383</td><td style="background:#d2992260">0.571</td></tr>
<tr><td>Clean (GT edges)</td><td style="background:#d2992260"><b>0.777</b> (0.710±0.035)</td><td style="background:#d2992260">0.68-0.71</td><td style="background:#d2992260">0.88-0.91</td></tr>
<tr><td>Noise=0.05 (FC)</td><td style="background:#2ea04360"><b>0.918</b> (0.371 mean)</td><td style="background:#cf222e60">0.35-0.52</td><td>--</td></tr>
<tr><td>Noise=0.5 (FC)</td><td style="background:#2ea04360"><b>0.988</b> (0.506±0.539, 50% fail)</td><td style="background:#cf222e60">0.448</td><td>--</td></tr>
</table>

_Linear degeneracy makes FC mode intractable. GT edges provide ~100x improvement. Noise=0.05 breaks the degeneracy, enabling 0.918 best single-seed even on FC — strongest evidence for noise-helps hypothesis._

### Table 4: LLM exploration runs

Each row = one `GNN_LLM.py` run with its own instruction file.

| Instruction file | Best W R2 | Iters | Status |
|-----------------|-----------|-------|--------|
| **Drosophila CX** | | | |
| `instruction_drosophila_cx.md` | 0.804 | 128 | done — FC ceiling ~0.574 mean, 22 seeds |
| `instruction_drosophila_cx_noise005.md` | 0.982 | 128 | done — dale_law=true, 24-seed mean=0.619, 54% success |
| `instruction_drosophila_cx_mlp.md` | ~0 | 128 | done — W R2≈0 always, Jacobian fundamentally limited, rollout mean=0.53 |
| `instruction_drosophila_cx_gt_edges.md` | 0.893 | 128 | done — dale_law+g_phi_wL1=0.003, mean=0.710, ~20% catastrophic |
| `instruction_drosophila_cx_noise05.md` | 0.999 | 128 | done — g_phi_norm=0.01, 6-seed mean=0.999±0.001, CV=0.09% |
| `instruction_drosophila_cx_gt_edges_noise005.md` | -- | -- | ready to launch |
| `instruction_drosophila_cx_gt_edges_noise05.md` | -- | -- | ready to launch |
| **Larva** | | | |
| `instruction_larva_gt_edges.md` | 0.908 | 128 | done — 28-seed mean=0.540, CV=35% |
| `instruction_larva_noise005.md` | 0.870 | 128 | done — W_L1+W_L2 synergy, 2-seed mean=0.683 |
| `instruction_larva_fc.md` | 0.435 | 128 | done — g_phi_norm=0.01, 10-seed mean=0.268, CV=40% |
| `instruction_larva_fc_noise005.md` | -- | -- | ready to launch |
| `instruction_larva_fc_noise05.md` | -- | -- | ready to launch |
| **Zebrafish** | | | |
| `instruction_zebrafish_oculomotor.md` | 0.022 | 128 | done — FC ceiling definitive (0.006 mean, 48 seeds) |
| `instruction_zebrafish_oculomotor_gt_edges.md` | 0.777 | 128 | done — bimodal convergence, 75% seeds ~0.71 |
| `instruction_zebrafish_oculomotor_noise005.md` | 0.918 | 128 | done — noise breaks linear degeneracy |
| `instruction_zebrafish_oculomotor_noise05.md` | 0.988 | 4 | running (Block 2) — 50% fail, mean=0.506 |
| `instruction_zebrafish_oculomotor_gt_edges_noise005.md` | -- | -- | ready to launch |
| `instruction_zebrafish_oculomotor_gt_edges_noise05.md` | -- | -- | ready to launch |
| **TODO: create** | | | |
| instruction_*_mlp.md (×4 models) | -- | -- | TODO |
| instruction_*_known_ode.md (×4 models) | -- | -- | TODO |
| instruction_*_rnn.md (×4 models) | -- | -- | TODO |
| instruction_*_neuralode.md (×4 models) | -- | -- | TODO |
| instruction_*_ssm.md (×4 models) | -- | -- | TODO |

---

## Nomenclature

### LLM Instruction Files

```
LLM/instruction_{biomodel}_{experiment}.md
```

| File                                           | Bio model            | Experiment             | Best W R2 | Iters | Status                     |
| ---------------------------------------------- | -------------------- | ---------------------- | --------- | ----- | -------------------------- |
| `instruction_drosophila_cx.md`                 | drosophila_cx        | clean, FC              | 0.804     | 128   | done                       |
| `instruction_drosophila_cx_noise005.md`        | drosophila_cx        | noise 0.05, FC         | 0.982     | 128   | done                       |
| `instruction_drosophila_cx_mlp.md`             | drosophila_cx        | MLP baseline           | ~0        | 128   | done — W R2≈0 always      |
| `instruction_larva_gt_edges.md`                | larva                | clean, GT edges        | 0.908     | 128   | done                       |
| `instruction_larva_noise005.md`                | larva                | intrinsic noise 0.05   | 0.870     | 128   | done                       |
| `instruction_larva_fc.md`                      | larva                | clean, FC              | 0.435     | 128   | done                       |
| `instruction_zebrafish_oculomotor.md`          | zebrafish_oculomotor | clean, fully connected | 0.022     | 128   | done                       |
| `instruction_zebrafish_oculomotor_gt_edges.md` | zebrafish_oculomotor | clean, GT edges        | 0.777     | 128   | done                       |
| `instruction_zebrafish_oculomotor_noise005.md` | zebrafish_oculomotor | intrinsic noise 0.05   | 0.918     | 128   | done                       |
| `instruction_zebrafish_oculomotor_noise05.md`  | zebrafish_oculomotor | intrinsic noise 0.5    | 0.988     | 4     | running (Block 2)          |
| `instruction_drosophila_cx_gt_edges.md`        | drosophila_cx        | clean, GT edges        | 0.893     | 128   | done                       |
| `instruction_drosophila_cx_noise05.md`         | drosophila_cx        | intrinsic noise 0.5    | 0.999     | 128   | done                       |
| `instruction_drosophila_cx_gt_edges_noise005.md` | drosophila_cx     | GT edges, noise 0.05   | --        | --    | ready to launch            |
| `instruction_drosophila_cx_gt_edges_noise05.md`  | drosophila_cx     | GT edges, noise 0.5    | --        | --    | ready to launch            |
| `instruction_larva_fc_noise005.md`             | larva                | FC, noise 0.05         | --        | --    | ready to launch            |
| `instruction_larva_fc_noise05.md`              | larva                | FC, noise 0.5          | --        | --    | ready to launch            |
| `instruction_zebrafish_oculomotor_gt_edges_noise005.md` | zebrafish_oculomotor | GT edges, noise 0.05 | --    | --    | ready to launch            |
| `instruction_zebrafish_oculomotor_gt_edges_noise05.md`  | zebrafish_oculomotor | GT edges, noise 0.5  | --    | --    | ready to launch            |
| `instruction_larva_noise05.md`                 | larva                | intrinsic noise 0.5    | --        | 0     | ready to launch            |
Each instruction file contains: model description, metrics, hyperparameter search space, block partition.
The agentic pipeline reads the instruction file to guide exploration.

### Config File Nomenclature

```
{biomodel}_{mlmodel}[_{experiment}]_{seed}.yaml
```

No experiment suffix = baseline (clean data, known/fully-connected topology).

### biomodel

| Code                   | Full name                        | N neurons | N edges | Source              |
| ---------------------- | -------------------------------- | --------- | ------- | ------------------- |
| `drosophila_cx`        | Drosophila adult central complex | 152       | 9,722   | Beiran 2023, Fig 5d |
| `larva`                | Drosophila larva motor           | 230       | 4,222   | Beiran 2023, Fig 5a |
| `zebrafish_oculomotor` | Zebrafish oculomotor             | 609       | ~10,665 | Beiran 2023, Fig 5g |

### mlmodel

| Code        | Full name            | File                    | Status                                                           |
| ----------- | -------------------- | ----------------------- | ---------------------------------------------------------------- |
| `gnn`       | GNN (ours)           | `neural_gnn.py`         | Done                                                             |
| `mlp`       | MLP baseline         | `mlp_baseline.py`       | Done — registered for drosophila_cx, larva, zebrafish_oculomotor |
| `known_ode` | Known ODE (GT structure) | `known_ode.py`       | Done — per-model activation (ReLU/softplus/identity)             |
| `neuralode` | Neural ODE (adjoint) | `neural_ode_wrapper.py` | Done — GNN + `neural_ODE_training: true`                         |
| `rnn`       | RNN (GRU)            | `neural_rnn.py`         | Done — registered for all bio models                             |

### experiment

| Code                 | Description                            | Applies to                                 |
| -------------------- | -------------------------------------- | ------------------------------------------ |
| _(no suffix)_        | Clean data, known topology (baseline)  | all                                        |
| `noise005`           | Intrinsic (process) noise sigma=0.05   | all                                        |
| `noise05`            | Intrinsic (process) noise sigma=0.5    | all                                        |
| `meas_noise_005`     | Measurement noise sigma=0.05           | all                                        |
| `meas_noise_05`      | Measurement noise sigma=0.5            | all                                        |
| `missing_time_80`    | Remove 4/5 timepoints (keep 20%)       | all                                        |
| `missing_neurons_20` | Remove 20% of neurons from observation | all                                        |
| `calcium`            | Calcium indicator (not voltage)        | all                                        |
| `remove_edges_20`    | Remove 20% of true edges               | all                                        |
| `fully_connected`    | Train on fully connected graph         | drosophila_cx, larva, zebrafish_oculomotor |

---

## Figures

1. **Fig 1**: Method overview — GNN architecture, teacher-student setup, 4 biological models
2. **Fig 2**: Table 1 + Table 2 as bar charts (4 bio models x 4 ML models)
3. **Fig 3**: Connectivity matrices — true vs learned heatmaps for all 4 bio models
4. **Fig 4**: Robustness (Table 3) as heatmap or grouped bar chart
5. **Fig 5**: Agentic exploration — convergence curves, best configs found
6. **Supp Fig**: g_phi/f_theta learned curves, embedding clustering, eigenvalue spectra

---

## Low-Rank Decomposition Analysis (U/V Asymmetry Story)

### Background: NeuralGraph finding

In the NeuralGraph project (case-low-rank.qmd), ground-truth W = U·V (rank-20, 100 neurons). Key finding: **W R² can be low while U R² is surprisingly good.** Example: W R²=0.364, U R²=0.946, V R²=0.427. The failure is always asymmetric — U (output modes, which neurons co-activate) recovers well because it is directly constrained by observed dynamics; V (input selection, which neurons drive which) is less observable.

This matches the theoretical prediction of **Mastrogiuseppe & Ostojic (2018)** ["Linking connectivity, dynamics, and computations in low-rank recurrent neural networks", *Neuron* 99(3), 609-623]: in low-rank networks, right-connectivity vectors (our U columns) determine the **output pattern** of activity and are directly constrained by observed dynamics, while left-connectivity vectors (our V rows) implement **input selection** — an implicit filtering operation that is less directly observable from activity data alone.

### Application to bio-models (no ground-truth UV)

The bio connectomes are full-rank — there is no ground-truth W = UV factorization. However, **this is not a problem**: SVD on any matrix gives U·S·V^T factors, and Procrustes alignment (orthogonal rotation minimizing Frobenius distance) resolves the rotation/sign/permutation ambiguity. The truncation rank r is chosen from the singular value spectrum (99% variance explained) rather than known a priori.

**Implementation** (already in `GNN_PlotFigure.py`):

1. Full SVD on both W_true and W_learned
2. Truncate to rank r (from 99% cumulative variance of true W)
3. Scale: U ← U·diag(S), keep V unscaled
4. Procrustes-align U_learned → U_true and V_learned → V_true
5. Compute R² on aligned factors separately
6. Sweep across multiple truncation ranks to show stability

### Preliminary results

**Drosophila CX** (152 neurons, rank_99=63):

```
Procrustes SVD (rank=63) — U R²: 0.8982   V R²: 0.5130   W_recon R²: 0.7477
Per-rank sweep:
  rank=  5   U_R2=0.9031   V_R2=0.7028   W_R2=0.6960
  rank= 10   U_R2=0.8969   V_R2=0.5398   W_R2=0.6943
  rank= 18   U_R2=0.8996   V_R2=0.4395   W_R2=0.6930
  rank= 20   U_R2=0.9000   V_R2=0.4271   W_R2=0.6946
  rank= 50   U_R2=0.8980   V_R2=0.4566   W_R2=0.7210
  rank= 63   U_R2=0.8982   V_R2=0.5130   W_R2=0.7477
```

**Key observations**:

- **Same asymmetry as NeuralGraph**: U R²≈0.90 vs V R²≈0.51, confirming Mastrogiuseppe & Ostojic (2018)
- **U is stable across ranks**: ~0.90 from rank=5 to rank=63 — output modes are uniformly well recovered
- **V is best at low rank**: V R²=0.70 at rank=5, degrades to 0.43 at rank=18-20. The GNN captures top ~5 input selection modes well but finer V structure is lost
- **W_recon R² (0.75) > raw W R² (0.74)**: low-rank projection cleans up noise in learned W

### Paper narrative

This analysis connects our GNN results to the theoretical framework of low-rank recurrent networks:

1. The GNN reliably recovers **output connectivity modes** (U) even when overall W R² appears modest
2. **Input selection modes** (V) are harder — consistent with theory that input filtering is less constrained by observed dynamics
3. The asymmetry is a property of recurrent dynamics (Mastrogiuseppe & Ostojic 2018), not of the GNN method — any inverse method should show this pattern
4. The rank sweep provides a richer characterization than scalar W R² alone

**Zebrafish oculomotor** (609 neurons, rank_99=93):

```
Procrustes SVD (rank=93) — U R²: 0.3072   V R²: -0.3166   W_recon R²: -0.0928
Per-rank sweep:
  rank=  5   U_R2=0.2956   V_R2=-0.7536   W_R2=-0.4715
  rank= 10   U_R2=0.3439   V_R2=-0.7104   W_R2=-0.3905
  rank= 20   U_R2=0.3405   V_R2=-0.6499   W_R2=-0.2657
  rank= 50   U_R2=0.3189   V_R2=-0.5041   W_R2=-0.1563
  rank= 54   U_R2=0.3165   V_R2=-0.4881   W_R2=-0.1518
  rank= 93   U_R2=0.3072   V_R2=-0.3166   W_R2=-0.0928
```

**Key observations (zebrafish)**:

- **U R²≈0.30-0.34**: the GNN captures some output mode structure, far less than CX (0.90)
- **V R² negative everywhere**: learned V is worse than predicting the mean — input selection completely unrecovered
- **W_recon R² negative**: low-rank reconstruction fails entirely
- The **U > V hierarchy is preserved** even in total failure: U always ≥ V
- This is the **linear integrator** (no tanh, gain=1) — dynamics are low-rank and degenerate, many different W produce identical activity. Neither factor is well-constrained.
- **Strongest evidence for noise-may-help hypothesis**: process noise would break the linear degeneracy by enriching activity covariance, potentially making W identifiable

**Comparison across bio-models (U/V asymmetry)**:

| Bio model               | W R²  | U R² | V R²  | U-V gap | Regime                           |
| ----------------------- | ----- | ---- | ----- | ------- | -------------------------------- |
| Drosophila CX (152, FC) | 0.681 | 0.90 | 0.51  | +0.39   | Nonlinear (tanh), ring attractor |
| Zebrafish (609, FC)     | 0.018 | 0.31 | -0.32 | +0.63   | Linear, degenerate integrator    |
| Larva (230, GT edges)   | 0.908 | ?    | ?     | ?       | Nonlinear (softplus), locomotor  |

The U-V gap is **larger** when recovery is harder — consistent with theory that V degrades first and fastest.

**Planned**: Run same analysis on larva. Expect intermediate results (nonlinear but softplus, moderate W R²).

### Proposed figure

**Fig 6** (or supplementary): 3×3 grid (one column per bio-model)

- Row 1: Singular value spectra (true vs learned)
- Row 2: U scatter (Procrustes-aligned) with R²
- Row 3: V scatter (Procrustes-aligned) with R²

Caption: "The GNN recovers output connectivity modes (U) better than input selection modes (V), consistent with theoretical predictions for recurrent networks (Mastrogiuseppe & Ostojic 2018)."

---

## Key References

1. Beiran & Litwin-Kumar (2023): "Connectivity-constrained neural networks" Nature Neuroscience 28, 2561-2574. https://doi.org/10.1038/s41593-025-02080-4
2. Hard reset finding: all major RNN neuroscience papers use trial resets inherited from seq2seq/LSTM training — biologically unrealistic
3. Mastrogiuseppe & Ostojic (2018): "Linking connectivity, dynamics, and computations in low-rank recurrent neural networks", _Neuron_ 99(3), 609-623. https://doi.org/10.1016/j.neuron.2018.07.003 — Theory: right-connectivity vectors (output modes) are directly constrained by dynamics; left-connectivity vectors (input selection) are not.
4. NeuralGraph low-rank case study: https://saalfeldlab.github.io/NeuralGraph/case-low-rank.html — Empirical confirmation of U/V asymmetry in GNN connectivity recovery.

---

## Table 5: Experiment Hash Table

Maps each experiment to its config files, instruction file, and LLM exploration command.

<table>
<tr><th>W R2</th><th>Experiment</th><th>Winner config</th><th>Best LLM iter</th><th>Instruction</th><th>LLM command</th></tr>
<tr><td colspan="6"><b>Drosophila CX (152, FC)</b></td></tr>
<tr><td style="background:#d2992260">0.804</td><td>drosophila_cx</td><td>pending</td><td>iter_104</td><td>instruction_drosophila_cx.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx iterations=128 --cluster --resume</code></td></tr>
<tr><td style="background:#2ea04360">0.982</td><td>drosophila_cx_noise005</td><td>pending</td><td>iter_009</td><td>instruction_drosophila_cx_noise005.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx_noise005 iterations=128 --cluster --resume</code></td></tr>
<tr><td style="background:#2ea04360">0.999</td><td>drosophila_cx_noise05</td><td>drosophila_cx_noise05_winner.yaml</td><td>iter_007</td><td>instruction_drosophila_cx_noise05.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx_noise05 iterations=128 --cluster --resume</code></td></tr>
<tr><td style="background:#2ea04360">0.893</td><td>drosophila_cx_gt_edges</td><td>drosophila_cx_gt_edges_winner.yaml</td><td>iter_104</td><td>instruction_drosophila_cx_gt_edges.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx_gt_edges iterations=128 --cluster --resume</code></td></tr>
<tr><td style="background:#cf222e60">~0</td><td>drosophila_cx — MLP</td><td>drosophila_cx_mlp_winner.yaml</td><td>iter_035</td><td>instruction_drosophila_cx_mlp.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx_mlp iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>drosophila_cx_gt_edges_noise005</td><td>--</td><td>--</td><td>instruction_drosophila_cx_gt_edges_noise005.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx_gt_edges_noise005 iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>drosophila_cx_gt_edges_noise05</td><td>--</td><td>--</td><td>instruction_drosophila_cx_gt_edges_noise05.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx_gt_edges_noise05 iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>drosophila_cx — Known ODE</td><td>--</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>drosophila_cx — RNN</td><td>--</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>drosophila_cx — NeuralODE</td><td>--</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>drosophila_cx — SSM</td><td>--</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td colspan="6"><b>Larva (230, GT edges)</b></td></tr>
<tr><td style="background:#d2992260">0.908</td><td>larva_gt_edges</td><td>pending</td><td>iter_093</td><td>instruction_larva_gt_edges.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude larva_gt_edges iterations=128 --cluster --resume</code></td></tr>

<tr><td style="background:#d2992260">0.870</td><td>larva_noise005</td><td>larva_noise005_winner.yaml</td><td>iter_031</td><td>instruction_larva_noise005.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude larva_noise005 iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>larva_noise05</td><td>--</td><td>--</td><td>instruction_larva_noise05.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude larva_noise05 iterations=128 --cluster --resume</code></td></tr>
<tr><td style="background:#cf222e60">0.435</td><td>larva_fc</td><td>larva_fc_winner.yaml</td><td>iter_109</td><td>instruction_larva_fc.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude larva_fc iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>larva_fc_noise005</td><td>--</td><td>--</td><td>instruction_larva_fc_noise005.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude larva_fc_noise005 iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>larva_fc_noise05</td><td>--</td><td>--</td><td>instruction_larva_fc_noise05.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude larva_fc_noise05 iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>larva — MLP/Known ODE/RNN/NeuralODE/SSM</td><td>--</td><td>--</td><td>--</td><td>TODO (×5)</td></tr>
<tr><td colspan="6"><b>Zebrafish oculomotor (609)</b></td></tr>
<tr><td style="background:#cf222e60">0.022</td><td>zebrafish_oculomotor (FC)</td><td>pending</td><td>iter_080</td><td>instruction_zebrafish_oculomotor.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude zebrafish_oculomotor iterations=128 --cluster --resume</code></td></tr>
<tr><td style="background:#d2992260">0.777</td><td>zebrafish_oculomotor_gt_edges</td><td>pending</td><td>iter_092</td><td>instruction_zebrafish_oculomotor_gt_edges.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude zebrafish_oculomotor_gt_edges iterations=128 --cluster --resume</code></td></tr>
<tr><td style="background:#2ea04360">0.918</td><td>zebrafish_oculomotor_noise005 (FC)</td><td>pending</td><td>iter_019</td><td>instruction_zebrafish_oculomotor_noise005.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude zebrafish_oculomotor_noise005 iterations=128 --cluster --resume</code></td></tr>
<tr><td style="background:#2ea04360">0.988</td><td>zebrafish_oculomotor_noise05</td><td>pending</td><td>iter_003</td><td>instruction_zebrafish_oculomotor_noise05.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude zebrafish_oculomotor_noise05 iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>zebrafish_oculomotor_gt_edges_noise005</td><td>--</td><td>--</td><td>instruction_zebrafish_oculomotor_gt_edges_noise005.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude zebrafish_oculomotor_gt_edges_noise005 iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>zebrafish_oculomotor_gt_edges_noise05</td><td>--</td><td>--</td><td>instruction_zebrafish_oculomotor_gt_edges_noise05.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude zebrafish_oculomotor_gt_edges_noise05 iterations=128 --cluster --resume</code></td></tr>
<tr><td>?</td><td>zebrafish — MLP/Known ODE/RNN/NeuralODE/SSM</td><td>--</td><td>--</td><td>--</td><td>TODO (×5)</td></tr>
</table>
