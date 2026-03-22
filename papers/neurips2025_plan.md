# NeurIPS 2025 Submission Plan

**Deadline**: May 11, 2025 (https://neurips.cc/Conferences/2025/CallForPapers)
**Working title**: Reverse-engineering neural connectomes from activity data with graph neural networks

## Story

Extend Cosyne flyvis work (https://saalfeldlab.github.io/flyvis-gnn/) in two axes:

1. **Generality**: 4 biological models (flyvis + 3 from Beiran & Litwin-Kumar 2023)
2. **Robustness**: intrinsic noise, missing timepoints, calcium imaging, edge removal

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
| `instruction_drosophila_cx.md` | drosophila_cx | clean, fully connected | Exists |
| `instruction_larva.md` | larva | clean, fully connected | **To write** |
| `instruction_zebrafish.md` | zebrafish | clean, fully connected | **To write** |
| `instruction_flyvis_missing_time_80.md` | flyvis | keep 20% timepoints | **To write** |
| `instruction_flyvis_remove_edges_20.md` | flyvis | remove 20% edges | **To write** |
| `instruction_flyvis_calcium.md` | flyvis | calcium indicator | **To write** (colleague) |

Each instruction file contains: model description, metrics, hyperparameter search space, block partition.
The agentic pipeline reads the instruction file to guide exploration.

### Config File Nomenclature

```
{biomodel}_{mlmodel}_{experiment}[_{seed}].yaml
```

### biomodel
| Code | Full name | N neurons | N edges | Source |
|------|-----------|-----------|---------|--------|
| `flyvis` | Drosophila optic lobe | 13,741 | 434,112 | flyvis package |
| `cx` | Drosophila adult central complex | 152 | 9,722 | Beiran 2023, Fig 5d |
| `larva` | Drosophila larva motor | ~100 | ~? | Beiran 2023, Fig 5a |
| `zebrafish` | Zebrafish oculomotor | ~500 | ~? | Beiran 2023, Fig 5g |

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
| `baseline` | Clean data, known topology | all |
| `noise005` | Intrinsic noise sigma=0.05 | all |
| `noise05` | Intrinsic noise sigma=0.5 | all |
| `missing_time_80` | Remove 4/5 timepoints (keep 20%) | all |
| `calcium` | Calcium indicator (not voltage) | all |
| `remove_edges_20` | Remove 20% of true edges | all |
| `null_edges_200` | Add 200% null edges (unknown topology) | flyvis only |
| `fully_connected` | Train on fully connected graph | cx, larva, zebrafish |

### seed
`_00` through `_04` for 5-seed benchmark, `_Claude_00` through `_Claude_03` for agentic exploration.

### Examples
```
flyvis_gnn_baseline_00.yaml
flyvis_linear_baseline_00.yaml
cx_gnn_baseline_00.yaml
cx_rnn_noise005_00.yaml
zebrafish_gnn_missing_time_80_00.yaml
flyvis_gnn_null_edges_200_00.yaml
cx_gnn_fully_connected_00.yaml
```

---

## Tables

### Table 1: Connectivity Recovery (R2_conn)

Primary metric. Higher = better.

| Bio model | GNN (ours) | Linear ODE | RNN | Neural ODE |
|-----------|------------|------------|-----|------------|
| Flyvis (13.7K neurons) | **0.93** | ? | ? | ? |
| Drosophila CX (152) | 0.57 (WIP) | ? | ? | ? |
| Larva (~100) | TBD | ? | ? | ? |
| Zebrafish (~500) | TBD | ? | ? | ? |

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

| Condition | Flyvis | CX | Larva | Zebrafish |
|-----------|--------|-------|-------|-----------|
| Baseline (clean) | 0.93 | ? | ? | ? |
| Intrinsic noise (sigma=0.05) | 0.96 | ? | ? | ? |
| Missing timepoints (keep 20%) | ? | ? | ? | ? |
| Calcium (not voltage) | ? | ? | ? | ? |
| Remove 20% edges | ? | ? | ? | ? |

Note: flyvis trains with known topology; cx/larva/zebrafish train fully connected.

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
| CX | Running (Block 1) | 48-96 more | ~50h (A100) |
| Larva | Not started | 48-96 | ~20h (A100) |
| Zebrafish | Not started | 48-96 | ~30h (A100) |

### Baselines (no agentic — manual HP sweep)

Each baseline: 5 configs x 5 seeds x 4 bio models = 100 runs per ML model.
3 ML baselines x 100 = **300 baseline runs**.

| ML model | Est. time per run | Total GPU-hours |
|----------|-------------------|-----------------|
| Linear ODE | ~30 min (flyvis), ~5 min (others) | ~60h |
| RNN | ~30 min (flyvis), ~5 min (others) | ~60h |
| Neural ODE | ~60 min (flyvis), ~10 min (others) | ~120h |

### Robustness Ablations (GNN only)

4 conditions x 4 bio models x 5 seeds = 80 runs.
Use best GNN config from agentic exploration (no re-exploration needed).

---

## Implementation TODO

### Must Build

- [ ] **RNN baseline model** (`flyvis_rnn.py`) — vanilla RNN with same graph topology, learns W + hidden dynamics
- [ ] **Missing timepoints** — subsample training frames (keep every 5th frame)
- [ ] **Edge removal** — randomly remove 20% of true edges before training
- [ ] **Benchmark script** — automated: generate data, train all models, collect metrics into CSV

### Must Run

- [ ] **Larva GNN** — write instruction file, start agentic exploration
- [ ] **Zebrafish GNN** — write instruction file, start agentic exploration
- [ ] **CX GNN** — finish current exploration (Dale's law, n_types=6)
- [ ] **All baselines** on all 4 bio models (linear, rnn, neuralode)
- [ ] **Robustness ablations** on all 4 bio models (noise, missing time, edge removal)
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
| W1 | Mar 22-28 | CX agentic (Dale's law). Write larva+zebrafish instruction files. Start larva+zebrafish GNN. Implement RNN baseline. |
| W2 | Mar 29-Apr 4 | Run all baselines on flyvis (5 configs x 5 seeds x 3 models). Continue cx/larva/zebrafish agentic. |
| W3 | Apr 5-11 | Run baselines on cx/larva/zebrafish. Implement missing timepoints + edge removal. |
| W4 | Apr 12-18 | Robustness ablations (all 4 models x 4 conditions x 5 seeds). Calcium from colleague. |
| W5 | Apr 19-25 | Collect all results into tables. Start writing methods + results. Make figures. |
| W6 | Apr 26-May 2 | Write intro, related work, discussion. Polish figures. |
| W7 | May 3-11 | Revisions, supplement, format check, submit. |

---

## Config Migration Plan

Keep flat per-biomodel structure (matches `graphs_data/` and `log/` layout). Benchmark configs live alongside existing configs, distinguished by naming convention.

```
config/fly/
  flyvis_gnn_baseline_00.yaml      ->  graphs_data/flyvis_gnn_baseline_00/  log/flyvis_gnn_baseline_00/
  flyvis_linear_baseline_00.yaml   ->  graphs_data/flyvis_linear_baseline_00/  ...
  flyvis_rnn_baseline_00.yaml
  flyvis_neuralode_baseline_00.yaml
  flyvis_gnn_noise005_00.yaml
  flyvis_gnn_missing_time_80_00.yaml
  flyvis_gnn_remove_edges_20_00.yaml
  flyvis_gnn_null_edges_200_00.yaml
  ... (existing organic configs remain)

config/drosophila_cx/
  cx_gnn_baseline_00.yaml
  cx_linear_baseline_00.yaml
  cx_rnn_baseline_00.yaml
  cx_neuralode_baseline_00.yaml
  cx_gnn_noise005_00.yaml
  ...

config/larva/
  larva_gnn_baseline_00.yaml
  ...

config/zebrafish_oculomotor/
  zebrafish_gnn_baseline_00.yaml
  ...
```

Existing agentic configs (`*_Claude_*.yaml`) remain untouched. Benchmark configs are frozen snapshots of best agentic results.

---

## Key References

1. Cosyne flyvis-gnn: https://saalfeldlab.github.io/flyvis-gnn/
2. Beiran & Litwin-Kumar (2023): "Connectivity-constrained neural networks" Nature Neuroscience 28, 2561-2574. https://doi.org/10.1038/s41593-025-02080-4
3. flyvis package: Lappalainen et al.
4. Hard reset finding: all major RNN neuroscience papers use trial resets inherited from seq2seq/LSTM training — biologically unrealistic
