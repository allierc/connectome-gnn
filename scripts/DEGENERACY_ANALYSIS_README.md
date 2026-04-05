# Degeneracy Analysis - Three Approaches

This directory contains scripts to estimate the null space (unconstrained weights) in the flyvis connectome inverse problem using three independent methods.

## Overview

The null space quantifies how many edge weights are free to vary without changing the neural dynamics. We estimate it using:

1. **Global SVD** (`global_svd_null_space.py`) — Population-level rank upper bound, ~26.5% degeneracy
2. **Per-neuron SVD** (`per_neuron_null_space_svd.py`) — Individual neuron measurement, ~44.6% degeneracy
3. **Structural Per-type** (`structural_per_type_nullspace.py`) — Mechanistic edge constraint counting, ~0% (connectome has no repeated edges)

All three methods analyze the same data across three noise conditions: `noise-free`, `noise-0.05`, `noise-0.5`.

## Quick Start

### Run all three approaches at once:

```bash
cd connectome-gnn/scripts
bash analyze_degeneracy_all.sh
```

This will:
1. ✓ Run global SVD analysis → `results_global_svd.json`
2. ✓ Run per-neuron SVD analysis → `results_per_neuron_svd.json`
3. ✓ Run structural analysis → `results_structural_nullspace.json`
4. ✓ Auto-update `docs/degeneracy_analysis.tex` with results
5. ✓ Generate visualization PNG files in `svg_*_plots/` directories

### Run individual analyses:

```bash
# Global SVD approach (coarse population-level bound)
conda run -n neural-graph-linux python global_svd_null_space.py

# Per-neuron SVD approach (empirical measurement, with visualizations)
conda run -n neural-graph-linux python per_neuron_null_space_svd.py

# Structural per-type approach (mechanistic counting)
conda run -n neural-graph-linux python structural_per_type_nullspace.py

# Auto-update tex file with all results
python update_degeneracy_tex.py
```

## Output Files

### Results JSON files:
- `results_global_svd.json` — Global effective ranks and null space estimates
- `results_per_neuron_svd.json` — Per-neuron effective ranks and identifiability metrics
- `results_structural_nullspace.json` — Degenerate edge group counts

### Visualization directories:
- `svg_global_svd_plots/` — Scree plots, cumulative variance, null space vs in-degree
- `svg_per_neuron_plots/` — Per-type heatmaps, diverging bars, scatter plots, waterfall charts
- `svg_structural_plots/` — Degenerate group summaries, null space distributions

### Updated documentation:
- `../docs/degeneracy_analysis.tex` — Auto-updated with actual results

## Results Format

Each JSON file contains results for all three noise conditions at multiple variance thresholds:

```json
{
  "noise-free": {
    "99.0%": {
      "null_space_dim": 193824,
      "degree_of_degeneracy": "44.6",
      "mean_effective_rank": "17.5",
      "fully_identifiable_neurons": 1481
    }
  }
}
```

The **degree of degeneracy** is the key metric: fraction of edges that are unconstrained.

## Variance Thresholds

All analyses use three variance thresholds:
- **99.5%** — More conservative, higher rank, lower null space
- **99%** — Standard threshold, balance between strictness and practicality
- **99.9%** — Very strict, lower rank, higher null space

## Key Findings

At the 99% variance threshold across all three approaches:

| Approach | Mechanism | Null space dim | **Degree of degeneracy** |
|----------|-----------|---|---|
| Global SVD | Population-level bound | 115,223 | **26.5%** |
| Per-neuron SVD | Individual measurement | 193,824 | **44.6%** |
| Structural per-type | Mechanistic counting | 0 (connectome) | **0.0%** |

**Interpretation:**
- Global SVD (26.5%) is a conservative lower bound
- Per-neuron SVD (44.6%) empirically measures how many weights are actually unconstrained
- Structural analysis (0%) shows the connectome has no repeated edges; the 121,100 null space in the paper comes from same-*type* correlations (requires cell type labels)

The **~44.6% degeneracy** from per-neuron SVD is the most direct empirical estimate.

## Automation

The analysis workflow is fully automated:

1. **Scripts output JSON** — Each analysis script writes results to JSON
2. **Tex auto-update** — `update_degeneracy_tex.py` reads all JSON files and updates tex with actual numbers
3. **Single command** — `bash run.sh` executes everything sequentially

## Dependencies

- Python 3.8+
- PyTorch
- NumPy, SciPy
- Zarr (for loading voltage traces)
- Matplotlib, Seaborn (for visualizations)
- Conda environment: `neural-graph-linux`

## Workflow

```bash
# 1. Run all three approaches (sequential, ~10-30 minutes total)
bash analyze_degeneracy_all.sh

# 2. Check what changed
git diff docs/degeneracy_analysis.tex

# 3. Commit results
git add results_*.json docs/degeneracy_analysis.tex
git commit -m "Add degeneracy analysis results for all three approaches"

# 4. Push
git push
```

## Troubleshooting

**Error: Conda environment not found**
```bash
conda env list  # Check available environments
# Make sure neural-graph-linux is installed
```

**Error: Missing data files**
```bash
# Check if graph data exists
ls graphs_data/fly/flyvis_noise_*/ode_params.pt
ls graphs_data/fly/flyvis_noise_*/y_list_test.zarr
```

**Results are 0%**
- Global SVD might have issues loading the subsampled matrix
- Structural analysis shows 0% because connectome has no repeated edges (expected)
- Check error messages in the script output

## Advanced Usage

### Run analyses in parallel (faster):

```bash
conda run -n neural-graph-linux python global_svd_null_space.py &
conda run -n neural-graph-linux python per_neuron_null_space_svd.py &
conda run -n neural-graph-linux python structural_per_type_nullspace.py &
wait
python update_degeneracy_tex.py
```

### Modify variance thresholds:

Edit the `variance_thresholds` parameter in each script (e.g., `(0.99, 0.999, 0.9999)` for stricter bounds).

### Custom output directory:

Modify the `output_dir` parameter in visualization functions to change where PNG plots are saved.

## References

- **Step 0:** Linear system formulation per neuron
- **Step 1:** Global SVD coarse bound
- **Step 2:** Per-neuron SVD empirical measurement
- **Step 3:** Structural per-type mechanistic explanation
- **Step 4:** Empirical validation (perturbation experiments)

See `../docs/degeneracy_analysis.tex` for detailed methodology and interpretation.
