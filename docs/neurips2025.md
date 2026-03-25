# NeurIPS 2025 Submission Plan

**Deadline**: May 11, 2025 (https://neurips.cc/Conferences/2025/CallForPapers)
**Working title**: Reverse-engineering neural connectomes from activity data with graph neural networks

<style>table { font-size: 0.85em; } th, td { padding: 3px 6px; }</style>

## Tables

Color code: <span style="color:#2ea043">green</span> &gt; 0.9, <span style="color:#d29922">orange</span> &gt; 0.5, <span style="color:#cf222e">red</span> &le; 0.5.

### Table 1: Connectivity Recovery (W R2)

Primary metric. Higher = better.

<table>
<tr><th>Bio model</th><th>GNN (ours)</th><th>MLP</th><th>Linear ODE</th><th>RNN</th><th>Neural ODE</th><th>SSM</th></tr>
<tr><td><b>Flyvis noise-free (13.7K, GT)</b></td><td style="background:#2ea04360"><b>0.926</b> (0.923±0.008, CV=0.82%)</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td><b>Flyvis noise=0.05 (13.7K, GT)</b></td><td style="background:#2ea04360"><b>0.985</b> (0.982±0.003, CV=0.30%)</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td><b>Flyvis noise=0.5 (13.7K, GT)</b></td><td style="background:#2ea04360"><b>0.990</b> (0.996±0.006, CV=0.64%)</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td><b>Drosophila CX (152, FC)</b></td><td style="background:#d2992260"><b>0.681</b> (0.574, 20 seeds, 15% fail)</td><td style="background:#cf222e60">0.003</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Drosophila CX noise=0.05 (152, FC)</td><td style="background:#2ea04360"><b>0.990</b> (0.777±0.164, CV=21.1%)</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td>Drosophila CX noise=0.5 (152, FC)</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>Larva (230, GT edges)</b></td><td style="background:#d2992260"><b>0.908</b> (0.600±0.186, CV=31%, 0% blow-ups)</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Larva noise=0.05 (230, GT edges)</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td>Larva noise=0.5 (230, GT edges)</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td><b>Zebrafish (609, FC)</b></td><td style="background:#cf222e60">0.018 (ceiling)</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td><b>Zebrafish (609, GT edges)</b></td><td style="background:#d2992260"><b>0.777</b> (0.710±0.035, CV=5.0%)</td><td>--</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Zebrafish noise=0.05 (609, FC)</td><td style="background:#2ea04360"><b>0.918</b> (0.826 mean, 3 seeds)</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td>Zebrafish noise=0.5 (609, FC)</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
</table>

_Format: best single-seed (robust mean±std, CV%). 4 seeds for robustness tests. Bold rows = baselines used in Table 2._

### Table 2: Rollout Prediction (Pearson r)

Secondary metric — autoregressive prediction quality. For noisy models, rollout is evaluated on noise-free test data.

<table>
<tr><th>Bio model</th><th>Frames</th><th>GNN (ours)</th><th>MLP</th><th>Linear ODE</th><th>RNN</th><th>Neural ODE</th><th>SSM</th></tr>
<tr><td>Flyvis noise-free (13.7K)</td><td>8527</td><td style="background:#2ea04360"><b>0.997</b> ± 0.015</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Flyvis noise=0.05 (13.7K)</td><td>8527</td><td style="background:#2ea04360"><b>0.991</b> ± 0.069</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Flyvis noise=0.5 (13.7K)</td><td>8527</td><td style="background:#2ea04360"><b>0.984</b> ± 0.162</td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Drosophila CX (152, FC)</td><td>2000</td><td style="background:#d2992260">0.71</td><td style="background:#d2992260">0.70</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Drosophila CX noise=0.05 (152, FC)</td><td>2000</td><td style="background:#d2992260">0.76</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td>Larva (230, GT edges)</td><td>480</td><td style="background:#2ea04360"><b>1.00</b></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Larva noise=0.05 (230, GT edges)</td><td>480</td><td>?</td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
<tr><td>Zebrafish (609, FC)</td><td>4200</td><td style="background:#2ea04360"><b>1.00</b></td><td>?</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Zebrafish (609, GT edges)</td><td>4200</td><td style="background:#2ea04360"><b>1.00</b></td><td>--</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Zebrafish noise=0.05 (609, FC)</td><td>4200</td><td style="background:#2ea04360"><b>1.00</b></td><td>--</td><td>--</td><td>--</td><td>--</td><td>--</td></tr>
</table>

_Flyvis rollout r = mean ± std over 13,741 neurons. Noisy models evaluated on noise-free test data (see Notebook_02). SSM = state-space model. Noise conditions: only GNN rollout needed (baselines run on clean data)._

### Table 3: Robustness (W R2 under degraded conditions, GNN only)

All 4 bio models. Report mean W R2 over 5 seeds.

<table>
<tr><th>Condition</th><th>Flyvis (13.7K)</th><th>Drosophila CX (152)</th><th>Larva (230)</th><th>Zebrafish (609)</th></tr>
<tr><td><b>Baseline (clean)</b></td><td style="background:#2ea04360"><b>0.926</b></td><td style="background:#d2992260">0.681 (FC)</td><td style="background:#d2992260">0.908 (GT)</td><td><span style="background:#cf222e60">0.018 (FC)</span>, <span style="background:#d2992260">0.777 (GT)</span></td></tr>
<tr><td><b>Intrinsic noise (σ=0.05)</b></td><td style="background:#2ea04360"><b>0.985</b></td><td style="background:#2ea04360">0.990 (FC)</td><td>?</td><td style="background:#2ea04360">0.918 (FC)</td></tr>
<tr><td><b>Intrinsic noise (σ=0.5)</b></td><td style="background:#2ea04360"><b>0.990</b></td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Measurement noise (σ=0.04)</td><td style="background:#2ea04360"><b>0.925</b></td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Measurement noise (σ=0.10)</td><td style="background:#d2992260"><b>0.756</b></td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Measurement noise (σ=0.05)</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Measurement noise (σ=0.5)</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Missing timepoints (keep 20%)</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Missing neurons (remove 20%)</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Calcium (not voltage)</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Remove 20% edges</td><td>?</td><td>?</td><td>?</td><td>?</td></tr>
<tr><td>Add 100% null edges</td><td style="background:#2ea04360"><b>0.982</b></td><td>N/A</td><td>N/A</td><td>N/A</td></tr>
<tr><td>Add 200% null edges</td><td style="background:#2ea04360"><b>0.982</b></td><td>N/A</td><td>N/A</td><td>N/A</td></tr>
<tr><td>Fully connected</td><td>N/A</td><td style="background:#d2992260">0.681</td><td>?</td><td style="background:#cf222e60">0.018</td></tr>
</table>

Note: flyvis trains with known topology; drosophila_cx trains FC by default; larva trains with GT edges; zebrafish_oculomotor trains FC (GT edges variant also explored). Bold rows = conditions with results on ≥2 bio models.

### Table 4: LLM exploration runs

Each row = one `GNN_LLM.py` run with its own instruction file.

| Instruction file | Best W R2 | Iters | Status |
|-----------------|-----------|-------|--------|
| **Flyvis** | | | |
| `instruction_flyvis_noise_free.md` | 0.926 | 156 | done |
| `instruction_flyvis_noise_005.md` | 0.985 | 253 | done |
| `instruction_flyvis_noise_05.md` | 0.990 | 204 | done |
| **Drosophila CX** | | | |
| `instruction_drosophila_cx.md` | 0.681 | 96 | running (Block 9) — FC ceiling ~0.68 |
| `instruction_drosophila_cx_noise005.md` | 0.990 | 92 | running (Block 7) — CV reduced to 21% |
| `instruction_drosophila_cx_mlp.md` | 0.003 | 4 | running (Block 1) |
| **Larva** | | | |
| `instruction_larva.md` | 0.908 | 100 | running (Block 10) — 0% blow-ups with bs=4 |
| `instruction_larva_noise005.md` | -- | 0 | exists, not started |
| **Zebrafish** | | | |
| `instruction_zebrafish_oculomotor.md` | 0.018 | 80 | running (Block 8) — FC ceiling definitive |
| `instruction_zebrafish_oculomotor_gt_edges.md` | 0.777 | 92 | running (Block 7) — lr=3e-4 MLP robust (CV=5%) |
| `instruction_zebrafish_oculomotor_noise005.md` | 0.918 | 40 | running (Block 4) — W_L1=1e-4 breakthrough |
| **TODO: create** | | | |
| instruction_drosophila_cx_noise05.md | -- | -- | TODO |
| instruction_drosophila_cx_gt_edges.md | -- | -- | TODO |
| instruction_larva_noise05.md | -- | -- | TODO |
| instruction_larva_fc.md | -- | -- | TODO |
| instruction_zebrafish_oculomotor_noise05.md | -- | -- | TODO |
| instruction_zebrafish_oculomotor_gt_noise005.md | -- | -- | TODO |
| instruction_flyvis_missing_time.md | -- | -- | TODO |
| instruction_flyvis_missing_neurons.md | -- | -- | TODO |
| instruction_flyvis_flywire_edges.md | -- | -- | TODO |
| instruction_*_mlp.md (×4 models) | -- | -- | TODO |
| instruction_*_linear.md (×4 models) | -- | -- | TODO |
| instruction_*_rnn.md (×4 models) | -- | -- | TODO |
| instruction_*_neuralode.md (×4 models) | -- | -- | TODO |
| instruction_*_ssm.md (×4 models) | -- | -- | TODO |

---

## Nomenclature

### LLM Instruction Files

```
LLM/instruction_{biomodel}_{experiment}.md
```

| File                                           | Bio model            | Experiment             | Status                     |
| ---------------------------------------------- | -------------------- | ---------------------- | -------------------------- |
| `instruction_flyvis_noise_free.md`             | flyvis               | clean, known topology  | Exists                     |
| `instruction_flyvis_noise_005.md`              | flyvis               | intrinsic noise 0.05   | Exists                     |
| `instruction_flyvis_noise_05.md`               | flyvis               | intrinsic noise 0.5    | Exists                     |
| `instruction_drosophila_cx.md`                 | drosophila_cx        | clean, GT edges        | Exists (running, 24 iters) |
| `instruction_larva.md`                         | larva                | clean, GT edges        | Exists (running, 36 iters) |
| `instruction_zebrafish_oculomotor.md`          | zebrafish_oculomotor | clean, fully connected | Exists (running, 24 iters) |
| `instruction_zebrafish_oculomotor_gt_edges.md` | zebrafish_oculomotor | clean, GT edges        | **Exists**                 |
| `instruction_drosophila_cx_noise005.md`        | drosophila_cx        | intrinsic noise 0.05   | **Exists**                 |
| `instruction_drosophila_cx_noise05.md`         | drosophila_cx        | intrinsic noise 0.5    | **To write**               |
| `instruction_larva_noise005.md`                | larva                | intrinsic noise 0.05   | **Exists**                 |
| `instruction_larva_noise05.md`                 | larva                | intrinsic noise 0.5    | **To write**               |
| `instruction_zebrafish_oculomotor_noise005.md` | zebrafish_oculomotor | intrinsic noise 0.05   | **Exists**                 |
| `instruction_zebrafish_oculomotor_noise05.md`  | zebrafish_oculomotor | intrinsic noise 0.5    | **To write**               |
| `instruction_flyvis_missing_time_80.md`        | flyvis               | keep 20% timepoints    | **To write**               |
| `instruction_flyvis_remove_edges_20.md`        | flyvis               | remove 20% edges       | **To write**               |
| `instruction_flyvis_calcium.md`                | flyvis               | calcium indicator      | **To write** (colleague)   |

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
| `flyvis`               | Drosophila optic lobe            | 13,741    | 434,112 | flyvis package      |
| `drosophila_cx`        | Drosophila adult central complex | 152       | 9,722   | Beiran 2023, Fig 5d |
| `larva`                | Drosophila larva motor           | 230       | 4,222   | Beiran 2023, Fig 5a |
| `zebrafish_oculomotor` | Zebrafish oculomotor             | 609       | ~10,665 | Beiran 2023, Fig 5g |

### mlmodel

| Code        | Full name            | File                    | Status                                                           |
| ----------- | -------------------- | ----------------------- | ---------------------------------------------------------------- |
| `gnn`       | GNN (ours)           | `neural_gnn.py`         | Done                                                             |
| `mlp`       | MLP baseline         | `mlp_baseline.py`       | Done — registered for drosophila_cx, larva, zebrafish_oculomotor |
| `linear`    | Linear ODE           | `flyvis_linear.py`      | Done — registered for all bio models                             |
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
| `null_edges_200`     | Add 200% null edges (unknown topology) | flyvis only                                |
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
| Drosophila CX (152, FC) | 0.659 | 0.90 | 0.51  | +0.39   | Nonlinear (tanh), ring attractor |
| Zebrafish (609, FC)     | 0.018 | 0.31 | -0.32 | +0.63   | Linear, degenerate integrator    |
| Larva (230, GT edges)   | 0.831 | ?    | ?     | ?       | Nonlinear (softplus), locomotor  |

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

1. Cosyne flyvis-gnn: https://saalfeldlab.github.io/flyvis-gnn/
2. Beiran & Litwin-Kumar (2023): "Connectivity-constrained neural networks" Nature Neuroscience 28, 2561-2574. https://doi.org/10.1038/s41593-025-02080-4
3. flyvis package: Lappalainen et al.
4. Hard reset finding: all major RNN neuroscience papers use trial resets inherited from seq2seq/LSTM training — biologically unrealistic
5. Mastrogiuseppe & Ostojic (2018): "Linking connectivity, dynamics, and computations in low-rank recurrent neural networks", _Neuron_ 99(3), 609-623. https://doi.org/10.1016/j.neuron.2018.07.003 — Theory: right-connectivity vectors (output modes) are directly constrained by dynamics; left-connectivity vectors (input selection) are not.
6. NeuralGraph low-rank case study: https://saalfeldlab.github.io/NeuralGraph/case-low-rank.html — Empirical confirmation of U/V asymmetry in GNN connectivity recovery.

---

## Table 5: Experiment Hash Table

Maps each experiment to its config files, instruction file, and LLM exploration command.

<table>
<tr><th>W R2</th><th>Experiment</th><th>Config</th><th>Best LLM config</th><th>Instruction</th><th>LLM command</th></tr>
<tr><td colspan="6"><b>Flyvis (13.7K, GT edges)</b></td></tr>
<tr><td style="background:#2ea04360">0.926</td><td>flyvis_noise_free</td><td>config/fly/flyvis_noise_free.yaml</td><td>LLM_flyvis_noise_free/.../iter_157</td><td>instruction_flyvis_noise_free.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_free iterations=128 --cluster</code></td></tr>
<tr><td style="background:#2ea04360">0.985</td><td>flyvis_noise_005</td><td>config/fly/flyvis_noise_005.yaml</td><td>LLM_flyvis_noise_005/.../iter_253</td><td>instruction_flyvis_noise_005.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_005 iterations=128 --cluster</code></td></tr>
<tr><td style="background:#2ea04360">0.990</td><td>flyvis_noise_05</td><td>config/fly/flyvis_noise_05.yaml</td><td>LLM_flyvis_noise_05/.../iter_201</td><td>instruction_flyvis_noise_05.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude flyvis_noise_05 iterations=128 --cluster</code></td></tr>
<tr><td style="background:#2ea04360">0.925</td><td>flyvis_noise_005+meas_004</td><td>config/fly/flyvis_noise_005_004.yaml</td><td>LLM_flyvis_noise_005_004/.../iter_033</td><td>--</td><td>--</td></tr>
<tr><td style="background:#d2992260">0.756</td><td>flyvis_noise_005+meas_010</td><td>config/fly/flyvis_noise_005_010.yaml</td><td>LLM_flyvis_noise_005_010/.../iter_009</td><td>--</td><td>--</td></tr>
<tr><td style="background:#2ea04360">0.982</td><td>flyvis_noise_005+null_100%</td><td>--</td><td>--</td><td>--</td><td>-- (single seed)</td></tr>
<tr><td style="background:#2ea04360">0.982</td><td>flyvis_noise_005+null_200%</td><td>--</td><td>--</td><td>--</td><td>-- (single seed)</td></tr>
<tr><td style="background:#2ea04360">0.942</td><td>flyvis_noise_005+INR</td><td>--</td><td>LLM_flyvis_noise_005_INR/.../iter_037</td><td>--</td><td>--</td></tr>
<tr><td>?</td><td>flyvis_missing_time_5x</td><td>--</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>flyvis_missing_time_10x</td><td>--</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>flyvis_missing_neurons_20</td><td>--</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>flyvis_flywire_edges</td><td>--</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>flyvis — MLP/Linear/RNN/NeuralODE/SSM</td><td>--</td><td>--</td><td>--</td><td>TODO (×5)</td></tr>
<tr><td colspan="6"><b>Drosophila CX (152, FC)</b></td></tr>
<tr><td style="background:#d2992260">0.681</td><td>drosophila_cx</td><td>config/drosophila_cx/drosophila_cx.yaml</td><td>LLM_drosophila_cx/.../iter_049</td><td>instruction_drosophila_cx.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx iterations=128 --cluster</code></td></tr>
<tr><td style="background:#2ea04360">0.990</td><td>drosophila_cx_noise005</td><td>config/drosophila_cx/drosophila_cx_noise005.yaml</td><td>LLM_drosophila_cx_noise005/.../iter_049</td><td>instruction_drosophila_cx_noise005.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx_noise005 iterations=128 --cluster</code></td></tr>
<tr><td>?</td><td>drosophila_cx_noise05</td><td>--</td><td>--</td><td>--</td><td>TODO: create</td></tr>
<tr><td>?</td><td>drosophila_cx_gt_edges</td><td>--</td><td>--</td><td>--</td><td>TODO: create</td></tr>
<tr><td style="background:#cf222e60">0.003</td><td>drosophila_cx — MLP</td><td>config/drosophila_cx/drosophila_cx_mlp.yaml</td><td>LLM_drosophila_cx_mlp/.../iter_001</td><td>instruction_drosophila_cx_mlp.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude drosophila_cx_mlp iterations=128 --cluster</code></td></tr>
<tr><td>?</td><td>drosophila_cx — Linear</td><td>config/drosophila_cx/drosophila_cx_linear_00.yaml</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>drosophila_cx — RNN</td><td>config/drosophila_cx/drosophila_cx_rnn.yaml</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>drosophila_cx — NeuralODE</td><td>config/drosophila_cx/drosophila_cx_neuralode.yaml</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td>?</td><td>drosophila_cx — SSM</td><td>--</td><td>--</td><td>--</td><td>TODO</td></tr>
<tr><td colspan="6"><b>Larva (230, GT edges)</b></td></tr>
<tr><td style="background:#d2992260">0.908</td><td>larva</td><td>config/larva/larva.yaml</td><td>LLM_larva/.../iter_057</td><td>instruction_larva.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude larva iterations=128 --cluster</code></td></tr>
<tr><td>?</td><td>larva_noise005</td><td>config/larva/larva_noise005.yaml</td><td>--</td><td>instruction_larva_noise005.md</td><td>TODO</td></tr>
<tr><td>?</td><td>larva_noise05</td><td>--</td><td>--</td><td>--</td><td>TODO: create</td></tr>
<tr><td>?</td><td>larva_fc</td><td>--</td><td>--</td><td>--</td><td>TODO: create</td></tr>
<tr><td>?</td><td>larva — MLP/Linear/RNN/NeuralODE/SSM</td><td>--</td><td>--</td><td>--</td><td>TODO (×5)</td></tr>
<tr><td colspan="6"><b>Zebrafish oculomotor (609)</b></td></tr>
<tr><td style="background:#cf222e60">0.018</td><td>zebrafish_oculomotor (FC)</td><td>config/zebrafish_oculomotor/zebrafish_oculomotor.yaml</td><td>LLM_zebrafish_oculomotor/.../iter_041</td><td>instruction_zebrafish_oculomotor.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude zebrafish_oculomotor iterations=128 --cluster</code></td></tr>
<tr><td style="background:#d2992260">0.777</td><td>zebrafish_oculomotor_gt_edges</td><td>config/zebrafish_oculomotor/zebrafish_oculomotor_gt_edges.yaml</td><td>LLM_zebrafish_oculomotor_gt_edges/.../iter_045</td><td>instruction_zebrafish_oculomotor_gt_edges.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude zebrafish_oculomotor_gt_edges iterations=128 --cluster</code></td></tr>
<tr><td style="background:#2ea04360">0.918</td><td>zebrafish_oculomotor_noise005 (FC)</td><td>config/zebrafish_oculomotor/zebrafish_oculomotor_noise005.yaml</td><td>LLM_zebrafish_oculomotor_noise005/.../iter_005</td><td>instruction_zebrafish_oculomotor_noise005.md</td><td><code>python GNN_LLM.py -o generate_train_test_plot_Claude zebrafish_oculomotor_noise005 iterations=128 --cluster</code></td></tr>
<tr><td>?</td><td>zebrafish_oculomotor_noise05</td><td>--</td><td>--</td><td>--</td><td>TODO: create</td></tr>
<tr><td>?</td><td>zebrafish_oculomotor_gt_noise005</td><td>--</td><td>--</td><td>--</td><td>TODO: create</td></tr>
<tr><td>?</td><td>zebrafish — MLP/Linear/RNN/NeuralODE/SSM</td><td>--</td><td>--</td><td>--</td><td>TODO (×5)</td></tr>
</table>
