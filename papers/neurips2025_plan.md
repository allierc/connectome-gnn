# NeurIPS 2025 Submission Plan

**Deadline**: May 11, 2025 (https://neurips.cc/Conferences/2025/CallForPapers)
**Working title**: Reverse-engineering neural connectomes from activity data with graph neural networks

## Story

Extend Cosyne flyvis work (https://saalfeldlab.github.io/flyvis-gnn/) in two axes:

1. **Generality**: 4 biological models (flyvis + 3 from Beiran & Litwin-Kumar 2023)
2. **Robustness**: intrinsic noise, measurement noise, missing timepoints, missing neurons, calcium imaging, edge removal, edge addition (unknown topology)

Compare against 3 baselines: Linear ODE, RNN, Neural ODE.

Cite Beiran & Litwin-Kumar (Nature Neuroscience 2025) but no direct comparison — they solve the forward problem (connectome -> dynamics), we solve the inverse (dynamics -> connectome).

Discussion includes the **hard reset flaw** found in all major RNN-based neuroscience papers.

---

## Nomenclature

### LLM Instruction Files

```
LLM/instruction_{biomodel}_{experiment}.md
```

| File | Bio model | Experiment | Status |
|------|-----------|------------|--------|
| `instruction_flyvis_noise_free.md` | flyvis | clean, known topology | Exists |
| `instruction_flyvis_noise_005.md` | flyvis | intrinsic noise 0.05 | Exists |
| `instruction_flyvis_noise_05.md` | flyvis | intrinsic noise 0.5 | Exists |
| `instruction_drosophila_cx.md` | drosophila_cx | clean, GT edges | Exists (running, 24 iters) |
| `instruction_larva.md` | larva | clean, GT edges | Exists (running, 36 iters) |
| `instruction_zebrafish_oculomotor.md` | zebrafish_oculomotor | clean, fully connected | Exists (running, 24 iters) |
| `instruction_zebrafish_oculomotor_gt_edges.md` | zebrafish_oculomotor | clean, GT edges | **Exists** |
| `instruction_drosophila_cx_noise005.md` | drosophila_cx | intrinsic noise 0.05 | **Exists** |
| `instruction_drosophila_cx_noise05.md` | drosophila_cx | intrinsic noise 0.5 | **To write** |
| `instruction_larva_noise005.md` | larva | intrinsic noise 0.05 | **Exists** |
| `instruction_larva_noise05.md` | larva | intrinsic noise 0.5 | **To write** |
| `instruction_zebrafish_oculomotor_noise005.md` | zebrafish_oculomotor | intrinsic noise 0.05 | **Exists** |
| `instruction_zebrafish_oculomotor_noise05.md` | zebrafish_oculomotor | intrinsic noise 0.5 | **To write** |
| `instruction_flyvis_missing_time_80.md` | flyvis | keep 20% timepoints | **To write** |
| `instruction_flyvis_remove_edges_20.md` | flyvis | remove 20% edges | **To write** |
| `instruction_flyvis_calcium.md` | flyvis | calcium indicator | **To write** (colleague) |

Each instruction file contains: model description, metrics, hyperparameter search space, block partition.
The agentic pipeline reads the instruction file to guide exploration.

### Config File Nomenclature

```
{biomodel}_{mlmodel}[_{experiment}]_{seed}.yaml
```

No experiment suffix = baseline (clean data, known/fully-connected topology).

### biomodel
| Code | Full name | N neurons | N edges | Source |
|------|-----------|-----------|---------|--------|
| `flyvis` | Drosophila optic lobe | 13,741 | 434,112 | flyvis package |
| `drosophila_cx` | Drosophila adult central complex | 152 | 9,722 | Beiran 2023, Fig 5d |
| `larva` | Drosophila larva motor | 230 | 4,222 | Beiran 2023, Fig 5a |
| `zebrafish_oculomotor` | Zebrafish oculomotor | 609 | ~10,665 | Beiran 2023, Fig 5g |

### mlmodel
| Code | Full name | File | Status |
|------|-----------|------|--------|
| `gnn` | GNN (ours) | `flyvis_gnn.py` | Done |
| `linear` | Linear ODE | `flyvis_linear.py` | Done |
| `neuralode` | Neural ODE (adjoint) | `neural_ode_wrapper.py` | Done |
| `rnn` | Vanilla RNN | TBD | **To implement** |

### experiment
| Code | Description | Applies to |
|------|-------------|------------|
| *(no suffix)* | Clean data, known topology (baseline) | all |
| `noise005` | Intrinsic (process) noise sigma=0.05 | all |
| `noise05` | Intrinsic (process) noise sigma=0.5 | all |
| `meas_noise_005` | Measurement noise sigma=0.05 | all |
| `meas_noise_05` | Measurement noise sigma=0.5 | all |
| `missing_time_80` | Remove 4/5 timepoints (keep 20%) | all |
| `missing_neurons_20` | Remove 20% of neurons from observation | all |
| `calcium` | Calcium indicator (not voltage) | all |
| `remove_edges_20` | Remove 20% of true edges | all |
| `null_edges_200` | Add 200% null edges (unknown topology) | flyvis only |
| `fully_connected` | Train on fully connected graph | drosophila_cx, larva, zebrafish_oculomotor |

### seed
`_00` through `_04` for 5-seed benchmark, `_Claude_00` through `_Claude_03` for agentic exploration.

### Examples
```
flyvis_gnn_00.yaml                          # baseline (no suffix)
flyvis_linear_00.yaml                       # baseline (no suffix)
drosophila_cx_gnn_00.yaml                   # baseline (no suffix)
drosophila_cx_rnn_noise005_00.yaml
zebrafish_oculomotor_gnn_missing_time_80_00.yaml
flyvis_gnn_null_edges_200_00.yaml
drosophila_cx_gnn_fully_connected_00.yaml
```

---

## Tables

### Table 1: Connectivity Recovery (R2_conn)

Primary metric. Higher = better.

| Bio model | GNN (ours) | Linear ODE | RNN | Neural ODE |
|-----------|------------|------------|-----|------------|
| Flyvis (13.7K neurons) | **0.93** | ? | ? | ? |
| Drosophila CX (152) | 0.74 (partial, FC, high CV) | ? | ? | ? |
| Larva (230) | 0.55 (partial, GT edges) | ? | ? | ? |
| Zebrafish (609) | 0.02 (partial, FC, near zero) | ? | ? | ? |

*5 seeds each. Report mean +/- std.*

### Table 2: Rollout Prediction (Pearson r)

Secondary metric — autoregressive prediction quality.

| Bio model | GNN (ours) | Linear ODE | RNN | Neural ODE |
|-----------|------------|------------|-----|------------|
| Flyvis | ? | ? | ? | ? |
| Drosophila CX | ? | ? | ? | ? |
| Larva | ? | ? | ? | ? |
| Zebrafish | ? | ? | ? | ? |

*5 seeds each. Report mean +/- std.*

### Table 3: Robustness (R2_conn under degraded conditions, GNN only)

All 4 bio models. Report mean R2_conn over 5 seeds.

| Condition | Flyvis | Drosophila CX | Larva | Zebrafish |
|-----------|--------|---------------|-------|-----------|
| Baseline (clean) | 0.93 | 0.74 (partial) | 0.55 (partial) | 0.02 (partial) |
| Intrinsic noise (sigma=0.05) | 0.96 | ? | ? | ? |
| Intrinsic noise (sigma=0.5) | ? | ? | ? | ? |
| Measurement noise (sigma=0.05) | ? | ? | ? | ? |
| Measurement noise (sigma=0.5) | ? | ? | ? | ? |
| Missing timepoints (keep 20%) | ? | ? | ? | ? |
| Missing neurons (remove 20%) | ? | ? | ? | ? |
| Calcium (not voltage) | ? | ? | ? | ? |
| Remove 20% edges | ? | ? | ? | ? |
| Add 200% null edges (unknown topology) | ? | N/A | N/A | N/A |

Note: flyvis trains with known topology; drosophila_cx trains with GT edges; larva trains with GT edges; zebrafish_oculomotor trains fully connected (GT edges variant in separate instruction file).

**Partial results status** (as of Mar 23):
- **Drosophila CX**: 24 iterations, best single-seed 0.742 (W_L1=3e-6). FC training only — GT edges untested. Extreme seed variance (CV>50%). n_epochs=2 breakthrough (0.71) but fragile.
- **Larva**: 36 iterations, best single-seed 0.552 (W_sign=0.1, GT edges). FC caps at 0.19. GT edges essential but fragile (CV~50%); W_sign=0.05-0.1 stabilizes.
- **Zebrafish**: 24 iterations, best 0.017. Near zero — linear degeneracy makes FC intractable. GT edges variant created (`instruction_zebrafish_oculomotor_gt_edges.md`) — most promising next step.

### Supplementary Table: Topology ablation (flyvis only)

| Null edges (%) | R2_conn (mean +/- std) |
|----------------|------------------------|
| 0% (known topology) | 0.93 |
| 100% | 0.95 |
| 200% | 0.94 |
| 400% | 0.94 |
| 800% | ? |

---

## Figures

1. **Fig 1**: Method overview — GNN architecture, teacher-student setup, 4 biological models
2. **Fig 2**: Table 1 + Table 2 as bar charts (4 bio models x 4 ML models)
3. **Fig 3**: Connectivity matrices — true vs learned heatmaps for all 4 bio models
4. **Fig 4**: Robustness (Table 3) as heatmap or grouped bar chart
5. **Fig 5**: Agentic exploration — convergence curves, best configs found
6. **Supp Fig**: g_phi/f_theta learned curves, embedding clustering, eigenvalue spectra

---

## Compute Plan

### Agentic Exploration (GNN only)

| Bio model | Status | Est. iterations | Est. GPU-hours |
|-----------|--------|-----------------|----------------|
| Flyvis | ~144 iters done | Done (maybe refine) | Done |
| Drosophila CX | Running (Block 4, 24 iters, best 0.74) | 48-72 more | ~50h (A100) |
| Larva | Running (Block 7, 36 iters, best 0.55) | 24-48 more | ~20h (A100) |
| Zebrafish oculomotor | Running (Block 3, 24 iters, best 0.02) | 48-72 more | ~30h (A100) |

### Baselines (no agentic — manual HP sweep)

Each baseline: 5 configs x 5 seeds x 4 bio models = 100 runs per ML model.
3 ML baselines x 100 = **300 baseline runs**.

| ML model | Est. time per run | Total GPU-hours |
|----------|-------------------|-----------------|
| Linear ODE | ~30 min (flyvis), ~5 min (others) | ~60h |
| RNN | ~30 min (flyvis), ~5 min (others) | ~60h |
| Neural ODE | ~60 min (flyvis), ~10 min (others) | ~120h |

### Robustness Ablations (GNN only)

9 conditions x 4 bio models x 5 seeds = 180 runs (minus N/A combos).
Use best GNN config from agentic exploration (no re-exploration needed).

---

## Implementation TODO

### Must Build

- [ ] **RNN baseline model** (`flyvis_rnn.py`) — vanilla RNN with same graph topology, learns W + hidden dynamics
- [ ] **Missing timepoints** — subsample training frames (keep every 5th frame)
- [ ] **Missing neurons** — remove 20% of neurons from observation during training
- [ ] **Measurement noise** — add Gaussian noise to observed trajectories (sigma=0.05, 0.5)
- [ ] **Edge removal** — randomly remove 20% of true edges before training
- [ ] **Edge addition (unknown topology)** — add null edges (200%) to graph before training
- [ ] **Benchmark script** — automated: generate data, train all models, collect metrics into CSV

### Must Run

- [x] **Larva GNN** — running (Block 7, 36 iters, best 0.55)
- [x] **Zebrafish oculomotor GNN** — running (Block 3, 24 iters, best 0.02)
- [ ] **Drosophila CX GNN** — continue exploration (GT edges next, Block 4)
- [ ] **Noise ablations (3 bio models x 2 noise levels)** — write instruction files, start agentic exploration (see plan below)
- [ ] **All baselines** on all 4 bio models (linear, rnn, neuralode)
- [ ] **Robustness ablations** on all 4 bio models (measurement noise, missing timepoints, missing neurons, calcium, edge removal, edge addition)
- [ ] **Calcium ablation** — colleague handles implementation

### Must Write

- [ ] Introduction + related work
- [ ] Methods (GNN architecture, training, agentic exploration)
- [ ] Results (Tables 1-3, figures)
- [ ] Discussion (hard reset flaw, comparison to Beiran 2023, limitations)
- [ ] Supplement (detailed ablations, per-type analysis)

---

## Timeline

| Week | Dates | Focus |
|------|-------|-------|
| W1 | Mar 22-28 | CX agentic (Dale's law). Write larva+zebrafish_oculomotor instruction files. Start larva+zebrafish_oculomotor GNN. Implement RNN baseline. |
| W2 | Mar 29-Apr 4 | Run all baselines on flyvis (5 configs x 5 seeds x 3 models). Continue drosophila_cx/larva/zebrafish_oculomotor agentic. |
| W3 | Apr 5-11 | Run baselines on drosophila_cx/larva/zebrafish_oculomotor. Implement missing timepoints, missing neurons, measurement noise, edge removal, edge addition. |
| W4 | Apr 12-18 | Robustness ablations (all 4 models x 9 conditions x 5 seeds). Calcium from colleague. |
| W5 | Apr 19-25 | Collect all results into tables. Start writing methods + results. Make figures. |
| W6 | Apr 26-May 2 | Write intro, related work, discussion. Polish figures. |
| W7 | May 3-11 | Revisions, supplement, format check, submit. |

---

## Config Migration Plan

Keep flat per-biomodel structure (matches `graphs_data/` and `log/` layout). Benchmark configs live alongside existing configs, distinguished by naming convention.

```
config/fly/
  flyvis_gnn_00.yaml               ->  graphs_data/flyvis_gnn_00/  log/flyvis_gnn_00/
  flyvis_linear_00.yaml            ->  graphs_data/flyvis_linear_00/  ...
  flyvis_rnn_00.yaml
  flyvis_neuralode_00.yaml
  flyvis_gnn_noise005_00.yaml
  flyvis_gnn_meas_noise_005_00.yaml
  flyvis_gnn_missing_time_80_00.yaml
  flyvis_gnn_missing_neurons_20_00.yaml
  flyvis_gnn_remove_edges_20_00.yaml
  flyvis_gnn_null_edges_200_00.yaml
  ... (existing organic configs remain)

config/drosophila_cx/
  drosophila_cx_gnn_00.yaml
  drosophila_cx_linear_00.yaml
  drosophila_cx_rnn_00.yaml
  drosophila_cx_neuralode_00.yaml
  drosophila_cx_gnn_noise005_00.yaml
  ...

config/larva/
  larva_gnn_00.yaml
  ...

config/zebrafish_oculomotor/
  zebrafish_oculomotor_gnn_00.yaml
  ...
```

Existing agentic configs (`*_Claude_*.yaml`) remain untouched. Benchmark configs are frozen snapshots of best agentic results.

---

## Noise Instruction Files Plan

Six new instruction files for intrinsic noise experiments on the three bio models.
These start from the **best config found** in the clean exploration and explore whether noise changes the optimal hyperparameters.

### General approach

Each noise instruction file:
1. **Inherits** the best config from the clean exploration as baseline
2. **Sets** `noise_model_level` to 0.05 or 0.5 in the simulation section
3. **Reduces block partition** — skip Blocks 1-3 (lr_W, W_L1, w_init already established) and focus on:
   - Block 1: Baseline validation (4 seeds, robustness test with best clean config + noise)
   - Block 2: Regularization re-tune (noise may require different W_L1, W_L2, W_sign)
   - Block 3: Training volume re-tune (noise may require more DAL or epochs)
   - Block 4: Architecture (hidden_dim may need increase for noisy data)
   - Block 5: Free exploration
4. **Same metrics** as clean: connectivity_R2 (primary), rollout_pearson, cluster_accuracy
5. **Same parallel mode**: 4 slots, exploration vs robustness

### Files to create

| File | Parent config (best clean) | noise_model_level | Notes |
|------|---------------------------|-------------------|-------|
| `instruction_drosophila_cx_noise005.md` | W_L1=3e-6, lr_W=3e-4, w_init=zeros, n_epochs=2, DAL=300 | 0.05 | From flyvis experience: noise 0.05 actually helps (0.96 vs 0.93 clean). May improve CX too. |
| `instruction_drosophila_cx_noise05.md` | same | 0.5 | High noise — may need stronger regularization or more training |
| `instruction_larva_noise005.md` | W_sign=0.05, use_gt_edges=true, lr_W=1e-4, W_L1=1e-6, DAL=2800 | 0.05 | Larva has softplus activation — noise robustness depends on nonlinearity |
| `instruction_larva_noise05.md` | same | 0.5 | May need W_sign increase to stabilize under high noise |
| `instruction_zebrafish_oculomotor_noise005.md` | lr_W=1e-4, W_L1=1e-5, DAL=160, w_init=randn_scaled | 0.05 | Linear system — noise may actually help break degeneracy by providing richer excitation |
| `instruction_zebrafish_oculomotor_noise05.md` | same | 0.5 | High noise on linear system — unclear if helpful or destructive |

### Key considerations

- **Noise may help zebrafish**: The linear integrator has degenerate W solutions because clean dynamics are low-rank. Process noise enriches the activity covariance, potentially making W more identifiable. This is the most scientifically interesting noise experiment.
- **Noise may hurt larva**: The softplus nonlinearity already provides some identifiability. Adding noise may just increase variance without helping W recovery.
- **CX with noise**: From flyvis experience, mild noise (0.05) improved connectivity recovery. The ring attractor dynamics may similarly benefit from richer exploration of state space.
- **Parent config may need GT edges first**: For CX and zebrafish, the clean exploration hasn't tested GT edges yet. The noise instruction files should use the best config available at time of creation — update parent config if GT edges prove beneficial before starting noise runs.

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

| Bio model | W R² | U R² | V R² | U-V gap | Regime |
|-----------|------|------|------|---------|--------|
| Drosophila CX (152) | 0.74 | 0.90 | 0.51 | +0.39 | Nonlinear (tanh), ring attractor |
| Zebrafish (609) | 0.02 | 0.31 | -0.32 | +0.63 | Linear, degenerate integrator |
| Larva (230) | 0.55 | ? | ? | ? | Nonlinear (softplus), locomotor |

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
5. Mastrogiuseppe & Ostojic (2018): "Linking connectivity, dynamics, and computations in low-rank recurrent neural networks", *Neuron* 99(3), 609-623. https://doi.org/10.1016/j.neuron.2018.07.003 — Theory: right-connectivity vectors (output modes) are directly constrained by dynamics; left-connectivity vectors (input selection) are not.
6. NeuralGraph low-rank case study: https://saalfeldlab.github.io/NeuralGraph/case-low-rank.html — Empirical confirmation of U/V asymmetry in GNN connectivity recovery.
