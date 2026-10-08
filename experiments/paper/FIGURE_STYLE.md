# Figure style for the manuscript (experiments/paper)

The conventions of the published figures, as re-implemented in `scripts/paper_style.py`
and `scripts/params_panels.py`. Every figure script in `scripts/` draws from the runs'
saved arrays (`results/*.npz`, `models/template_fit_alt.pt`, `results/metrics.txt`),
never from a PNG, so fonts, spines and points are identical across panels.

## Page geometry

- Full-width figures are drawn **18 cm wide** and included at `\textwidth` (13.97 cm,
  scale 0.78). Drawn sizes: labels 8 pt, ticks 6 pt, annotations 5.5 pt, panel letters
  9 pt bold, column titles 9 pt. A figure may run to 19-20 cm when its gaps need it; the
  scale then drops to ~0.7, still legible.
- Square parameter panels of **1.95-3.0 cm**; even gaps of **1.35 cm** laterally and
  **1.4 cm** between rows, so no tick label, axis label or letter reaches a neighbour
  (`fig_gnn_params_3col.py`: PANEL_CM, GAP_IN_CM, GAP_BLOCK_CM, GAP_ROW_CM).
- Column titles centred over their block, one full line above the panel letters
  (`column_titles(..., dy_pt=16, fontsize=9)`).

## Axes

- Left and bottom spines only, **never trimmed** to the tick range (`axes.spines.top/right`
  off, line width 0.5 pt, ticks 2 pt outward).
- **Fixed ranges with three ticks** (ends and middle): W `[-1, 2]`, V_rest `[0, 1]`,
  tau `[0, 0.5]` s, rollout voltage `[-10, 10]`, stimulus `[0, 1]`; the embedding axes
  span the 0.2-99.8 percentile of a_i, ends rounded to one decimal, middle tick rounded
  to one decimal (`ticks3(..., mid_decimals=1)`).
- Tick labels carry the decimals they need, at least one when any tick is fractional:
  `-1.0 0.5 2.0`, `0.0 0.5 1.0`, `0.0 0.25 0.5` (`ticks3`).
- Time axes in **ms**, ticks at 10000 / 20000 / 30000, labelled on the first column only.

## Points and lines

- Learned-vs-true scatters: **black** points; W `s=0.1, alpha=0.1` (up to 150k edges);
  per-neuron tau and V_rest `s=0.3, alpha=0.3`; the identity line light grey (0.8,
  0.5 pt); for tau and V_rest the outlier band as dotted lines at +-0.1 s / +-0.2.
- **Outliers red** (`s=0.4, alpha=0.5`), drawn underneath the inliers, clipped to the
  panel range; they are the neurons beyond the paper's thresholds and are excluded
  from the quoted R^2 (Appendix, evaluation metrics).
- Embeddings a_i coloured by ground-truth cell type (65 colours: tab20, tab20b, tab20c,
  Set1; `type_cmap`), `s=0.5, alpha=0.5`. Nothing else is coloured by type.
- Pooled (neuron, frame) scatters (rollout, stimulus) as **log-density images**
  (`hist2d`, 300 bins, `inferno_r`, white where empty).
- Traces: ground truth green `#2ca02c` 1.0 pt, prediction black 0.4 pt, stimulus red
  `#cf222e` 0.6 pt; 12 representative cell types stacked, R1 at the bottom with its
  input below; the Fisher-z pooled Pearson r in a header inside the panel.

## Annotations and letters

- Top-left inside the panel, 5.5 pt: W `R²: x.xx / slope: x.xx`; tau and V_rest
  `R²: x.xx (all-neuron R²) / slope: x.xx / Outliers: x.x%`. Every number is read from
  the run's `metrics.txt` and the arrays are checked to reproduce it before drawing
  (`params_panels.load_run`), so a figure and the tables quote one value.
- Panel letters bold, lower case, outside the panel: starting at the left edge of the
  y-axis label, 2-4 pt above the top of the axes frame (`panel_labels`, `ha="left"`,
  `y_from="axes"`); panels that carry a header inside use the tight bbox instead
  (`y_from="tight"`). Letters run row-major across the whole figure.
- Caption colour: blue `{\color{blue} ...}` for a regenerated figure or table, red for a
  passage or figure still resting on the published current-form results.

## Data sources (one script = one figure)

| figure | script | arrays |
|---|---|---|
| Fig. 1, Supp. FlyWire parameters | `fig_gnn_params_3col.py`, `fig_gnn_params_4col_flywire.py` | `gt_weights.pt`, `training_edges.pt`, `models/template_fit_alt.pt` (W, softplus(raw_tau), V_rest), `results/panels_*.npz` (a, true tau / V_rest, type_ids), `results/recovered_pairs.npz` (per-edge and per-neuron message pairs, written by `-o plot` through `recovery_figures.write_recovered_pairs`), `results/metrics.txt` |
| Fig. 2 (panels e, f; a-d cropped from the published PNG) | `fig_flywire_hybrid.py` | `results/rollout_bundle.npz` of the two FlyWire-eye cv00 runs of experiment 7 |
| Supp. rollout, Supp. ablation | `fig_rollout_3col.py [--ablation50]` | `results/rollout_bundle.npz`, `ablation50/rollout_bundle_on_noise_free_mask_50.npz` |
| Supp. SIREN stimulus | `fig_stim_rollout_inr.py` | `results/rollout_bundle.npz` (stimulus_input_true / _pred, the raw SIREN output shown with its mean and SD matched to the true stimulus), dataset `pos.zarr` |
| Supp. clustering | `fig_clustering_appendix.py` | as Fig. 1 plus the dataset's `ode_params.pt` |
| tables | `make_tables.py` | `results/metrics.txt` of every fold |

`bash scripts/build.sh` regenerates everything and the PDF.
