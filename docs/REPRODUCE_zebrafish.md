# Reproducing the figures and results in `docs/zebrafish.tex`

This manifest pins down everything needed to regenerate the 5 figures and
all numerical statements in `docs/zebrafish.tex` from a clean checkout.
It is intentionally precise about commit ids, conda env, seeds, trained
checkpoint paths, and per-figure command lines, so a future refactor
branch can rerun this pipeline and diff its outputs against the
artefacts that ship with the document.

---

## 0. Reference state

| Item | Value |
|---|---|
| Repo HEAD when this doc was written | `b6fa108a9e0bb4eabae1e1b56adc2561975587cb` (`b6fa108`) on `feat/janelia-cx`, committed `2026-05-29T11:59:09-04:00` |
| Tag to add when the branch is in a stable state | `git tag -a zebrafish-tex-2026-05-29 -m "trained dIPN HD model + zebrafish.tex figures, frozen" b6fa108` (push with `git push origin zebrafish-tex-2026-05-29`) |
| Active conda env (CLI prompt: `(neural-graph-linux)`) | `/workspace/.conda_envs/neural-graph-linux` |
| Activate from a fresh shell | `source /opt/conda/etc/profile.d/conda.sh && conda activate /workspace/.conda_envs/neural-graph-linux` |
| Repo root for every command below | `cd /workspace/connectome-gnn-cx` |
| GraphData root (logs + trained models + task data) | `GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData` (already set in `devcontainer.json`) |

> NB: the trained checkpoints and the generated task data live OUTSIDE the
> git repo, under `${GNN_OUTPUT_ROOT}`. Branch state alone is not enough
> to reproduce — the artefacts at the paths below have to be present too.

---

## 1. Cached input data (independent of the trained model)

### 1a. Cached neuprint-fish2 anatomy + connectome

Fetched once via the scripts below; the resulting CSVs / SWCs / OBJs ship in
the repo and never change unless re-fetched.

```
figures/zebrafish/zebrafish_anatomy_HD/          # SWC skeletons + OBJ ROIs for 727 HD-circuit neurons
figures/zebrafish/zebrafish_connectome_HD/       # neurons.csv (731 rows) + connections.csv (14,391 edges)
figures/zebrafish/zebrafish_anatomy_IPN12/       # 106 IPN12_a / IPN12_b SWCs (companion only, not used by tex)
```

If those folders are present, **skip the fetch step**. To regenerate from
scratch (requires `NEUPRINT_APPLICATION_CREDENTIALS` env var and a host
that can reach `neuprint-fish2.janelia.org`, currently gated by
`.devcontainer/init-firewall.sh`):

```
python figures/zebrafish/fetch_zebrafish_anatomy_HD.py
python figures/zebrafish/fetch_zebrafish_connectivity_HD.py
```

### 1b. Generated task data (swim-integration)

For training and for analyses that need a deterministic rollout, the
dataset `drosophila_cx/zebrafish_hd_si_task` (or its `_tv` variant) must
exist under `${GNN_OUTPUT_ROOT}/graphs_data/`. The seed that produced the
existing data is captured inside the `_complete` marker; regenerating with
the same simulation seed yields identical zarrs.

```
python GNN_Main.py -o generate zebrafish/zebrafish_hd_si_dipn      # base recipe
python GNN_Main.py -o generate zebrafish/zebrafish_hd_si_dipn_tv   # TV-regularised recipe
```

---

## 2. Trained model

The figures load the trained dIPN-only RNN at:

| | |
|---|---|
| Config name | `zebrafish_hd_si_dipn` |
| Yaml source | `config/zebrafish/zebrafish_hd_si_dipn.yaml` |
| Log dir | `${GNN_OUTPUT_ROOT}/log/zebrafish/zebrafish_hd_si_dipn/` |
| Checkpoints | `models/best_model_with_0_graphs_{1..5}.pt` (five epochs) |
| Default checkpoint used by figure scripts | `best_model_with_0_graphs_5.pt` (highest-epoch glob, picked by `_load(...)` in `fig_zebrafish_anatomy_3d_voltage_anim.py`) |

The two analysis scripts that import `_load_with_override` from
`fig_zebrafish_four_classes` also default to the same model
(`--model zebrafish_hd_si_dipn`).

Train from scratch (≈ 30 min on l4) with:

```
python GNN_Main.py -o train_task zebrafish/zebrafish_hd_si_dipn
```

Seeds are read from the yaml (`training.seed`, `task.swim_integration.seed`, `simulation.seed`); do not change them
between the baseline and the comparison run.

---

## 3. Per-figure commands

Each command writes to `figures/zebrafish/<scriptname>.png`. Run from the
repo root with the env active. Order matters for figures 4 and 5 because
they read `fig_zebrafish_four_classes.csv` produced by figure 3.

### Figure 1 — `fig_zebrafish_anatomy_3d_HD.png` (3-D HD-circuit anatomy)

```
python figures/zebrafish/fig_zebrafish_anatomy_3d_HD.py --bg white
```

Reads `zebrafish_anatomy_HD/`. No checkpoint needed.

### Figure 2 — `fig_connectome_summary_HD.png` (signed W^con + binary support)

```
python figures/zebrafish/fig_connectome_summary_HD.py
```

Reads `zebrafish_connectome_HD/`. No checkpoint needed.

### Figure 3 — `fig_zebrafish_four_classes.png` (R / L / D / Z partition)

This script is the *prerequisite* for figures 4 and 5: it writes
`fig_zebrafish_four_classes.csv`, which the next two scripts consume to
restrict to `R ∪ L` cells.

```
python figures/zebrafish/fig_zebrafish_four_classes.py
```

Defaults: `--model zebrafish_hd_si_dipn`, decision-tree thresholds set by
`--mi_q / --w_q / --swim_q` quantiles inside the script. Loads the latest
checkpoint under `${GNN_OUTPUT_ROOT}/log/zebrafish/zebrafish_hd_si_dipn/models/`.

### Figure 4 — `fig_zebrafish_pref_angle.png` (preferred-heading 3-D map)

```
python figures/zebrafish/fig_zebrafish_pref_angle.py
```

Needs the csv from figure 3. Default `--sigma_thr 0.30`; rollout type
`periodic` swim impulses (see script docstring for the impulse params).

### Figure 5 — `fig_zebrafish_tuning_sharpness.png` (4-panel sharpness summary)

```
python figures/zebrafish/fig_zebrafish_tuning_sharpness.py
```

Needs the csv from figure 3. Defaults: `--rollout periodic --n_steps 10000
--sigma_thr 0.70 --n_theta 36`. The 3 example cells shown in panel a are
picked automatically (median-σ of kept; median-σ pair shifted by ~π;
σ closest to 0.20 for the non-specific example). The script prints the
selected indices and stats to stdout — keep that log if you ever need to
reproduce the *same* three cells across re-runs.

Numbers reported in the tex (regenerate by re-running):
- 350 / 366 cells in R∪L pass σ ≥ 0.70
- FWHM ≈ 100°, von Mises κ ≈ 2
- Preferred-angle χ² ≈ 154 (df = 35), min/max per-bin = 3 / 28, mean ≈ 9.7

---

## 4. Build the PDF

```
cd docs
pdflatex -interaction=nonstopmode zebrafish.tex
pdflatex -interaction=nonstopmode zebrafish.tex   # second pass for refs
```

Two passes are required because `cleveref` resolves cross-references on
the second run. The expected page count for this revision of the doc is
**16 pages** (≈ 7 MB PDF).

---

## 5. Minimal regression check on a refactor branch

Before merging anything that touches `generators/`, `models/`, or the
connectome loader, confirm the three pillars from the design discussion:

**A. Task data (Layer 1) — connectome hash:**

```
python - <<'PY'
import hashlib, numpy as np
from connectome_gnn.generators.connconstr_data import load_zebrafish_hd_connectome
cx = load_zebrafish_hd_connectome("figures/zebrafish/zebrafish_connectome_HD")
for k in ("J_effective", "neuron_types"):
    a = np.asarray(cx[k])
    print(k, a.shape, a.dtype, hashlib.sha256(a.tobytes()).hexdigest()[:16])
PY
```

**B. Inference (Layer 2) — checkpoint produces identical decoded HD:**

A deterministic constant-omega rollout from the trained checkpoint
(e.g. via `_run_const(net, 1000, dt, 90.0, 0.0, 'cpu')`) should produce
a `decoded_hd` array that is bit-identical before and after a refactor
on CPU. Stash the array as a `.npz` baseline once; re-run and assert
`np.allclose(..., atol=0, rtol=0)` after the refactor.

**C. Training trajectory (Layer 3) — first 100 iters of metrics.log:**

Run a short `train_task` with the existing seeds and small
`n_trials_train`; capture the first 100 rows of `metrics.log`; diff after
the refactor. CPU bit-equal, GPU within numerical noise. Catches any
unintended init-order or RNG-order shift.

These three checks are cheap (< 10 min total) and they jointly say "the
trained checkpoint still belongs to the same scientific point" — which
is what you actually want from a refactor.

---

## 6. What this manifest does NOT cover

- The figure scripts assume `matplotlib >= 3.7` and the `neural-graph-linux`
  env. Switching matplotlib versions can re-render the same data with
  different fonts / antialias; treat numerical arrays as the ground truth,
  not pixels.
- The animation scripts
  (`fig_zebrafish_anatomy_3d_voltage_anim.py`, `fig_zebrafish_pca_ring.py`,
  `fig_zebrafish_readout_mi.py`, `fig_zebrafish_leaky.py`, …) are NOT in
  the current `zebrafish.tex` revision. Their commands are still in
  earlier git history if you need to regenerate any of them.
- The IPN12 overlay figure (`fig_zebrafish_IPN12.png`) is companion-only
  and not referenced by the tex.

---

_Owner: this file should be updated whenever the trained model is
retrained, the figure-generating scripts change their default args, or
the tex includes a new figure. Treat it as the contract between the
analysis pipeline and the document._
