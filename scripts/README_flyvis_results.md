# How to regenerate `docs/flyvis_results.md`

## Overview

The flyvis results documentation is generated automatically from log files.
Do **not** edit `docs/flyvis_results.md` by hand — use the scripts below.

## Pipeline

### 1. Run experiments (if needed)

Train CV folds via the `cv` task:

```bash
python GNN_Main.py -o cv flyvis_noise_005 --n_seeds 10
python GNN_Main.py -o cv flyvis_noise_005_default --n_seeds 10
python GNN_Main.py -o cv flyvis_noise_005_known_ode --n_seeds 10
# repeat for noise_free and noise_05
```

### 2. Retest rollouts with noise-free data (if needed)

If rollout Pearson values look low (< 0.8), the test data may contain noise.
Regenerate noise-free test data and retest:

```bash
python GNN_Main.py -o generate_test retest_noisy_rollouts
```

This uses the `retest_noisy_rollouts` config list defined in `GNN_Main.py`,
which covers all GNN (LLM-optimized) and GNN (default) CV folds for
noise=0.05 and noise=0.5 conditions.

To retest a single fold manually:

```bash
python GNN_Main.py -o generate_test flyvis_noise_005_cv00
```

### 3. Collect text tables

```bash
python scripts/collect_flyvis_results.py
```

- Scans `log/fly/` for all CV fold directories
- Parses `results.log`, `results_test.log`, `results_rollout.log`
- Outputs text-format tables to `docs/flyvis_tables.txt`
- Also prints tables to stdout for quick inspection

### 4. Generate the HTML markdown file

```bash
python scripts/generate_flyvis_md.py
```

- Reuses collection logic from `collect_flyvis_results.py`
- Outputs color-coded HTML tables to `docs/flyvis_results.md`
- Includes summary tables, per-seed detail tables, and key observations

## Quick one-liner

```bash
python scripts/collect_flyvis_results.py && python scripts/generate_flyvis_md.py
```

## What the scripts parse

Each CV fold directory (e.g., `log/fly/flyvis_noise_005_cv00/`) contains:

| File | Metrics extracted |
|------|-------------------|
| `results.log` | W R2, tau R2, V_rest R2, cluster accuracy |
| `results_test.log` | One-step Pearson r |
| `results_rollout.log` | Rollout Pearson r |

## Model variants

| Variant | Directory pattern | Base YAML |
|---------|-------------------|-----------|
| GNN (LLM-optimized) | `flyvis_{noise}_cv{NN}` | `flyvis_{noise}.yaml` |
| GNN (default) | `flyvis_{noise}_default_cv{NN}` | `flyvis_{noise}_default.yaml` |
| Known-ODE | `flyvis_{noise}_known_ode_cv{NN}` | `flyvis_{noise}_known_ode.yaml` |

Noise conditions: `noise_free`, `noise_005` (sigma=0.05), `noise_05` (sigma=0.5).

## Color coding

- **Green** (> 0.9): strong recovery
- **Orange** (> 0.5): moderate
- **Red** (<= 0.5): poor
