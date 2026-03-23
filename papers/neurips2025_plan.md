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
| `instruction_drosophila_cx.md` | drosophila_cx | clean, known topology | Exists (running, 24 iters) |
| `instruction_larva.md` | larva | clean, known topology | Exists (running, 36 iters) |
| `instruction_zebrafish_oculomotor.md` | zebrafish_oculomotor | clean, known topology | Exists (running, 24 iters) |
| `instruction_drosophila_cx_noise005.md` | drosophila_cx | intrinsic noise 0.05 | **To write** |
| `instruction_drosophila_cx_noise05.md` | drosophila_cx | intrinsic noise 0.5 | **To write** |
| `instruction_larva_noise005.md` | larva | intrinsic noise 0.05 | **To write** |
| `instruction_larva_noise05.md` | larva | intrinsic noise 0.5 | **To write** |
| `instruction_zebrafish_oculomotor_noise005.md` | zebrafish_oculomotor | intrinsic noise 0.05 | **To write** |
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

Note: flyvis trains with known topology; drosophila_cx and zebrafish_oculomotor train fully connected; larva trains with GT edges.

**Partial results status** (as of Mar 23):
- **Drosophila CX**: 24 iterations, best single-seed 0.742 (W_L1=3e-6). FC training only — GT edges untested. Extreme seed variance (CV>50%). n_epochs=2 breakthrough (0.71) but fragile.
- **Larva**: 36 iterations, best single-seed 0.552 (W_sign=0.1, GT edges). FC caps at 0.19. GT edges essential but fragile (CV~50%); W_sign=0.05-0.1 stabilizes.
- **Zebrafish**: 24 iterations, best 0.017. Near zero — linear degeneracy makes FC intractable. GT edges untested (most promising next step).

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

## Key References

1. Cosyne flyvis-gnn: https://saalfeldlab.github.io/flyvis-gnn/
2. Beiran & Litwin-Kumar (2023): "Connectivity-constrained neural networks" Nature Neuroscience 28, 2561-2574. https://doi.org/10.1038/s41593-025-02080-4
3. flyvis package: Lappalainen et al.
4. Hard reset finding: all major RNN neuroscience papers use trial resets inherited from seq2seq/LSTM training — biologically unrealistic
