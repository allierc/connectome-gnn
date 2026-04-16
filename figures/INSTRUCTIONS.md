# figure generation instructions

## one script = one figure
each script in this folder generates exactly one figure end-to-end from raw data.
no intermediate PNG files as inputs — all panels are rendered by matplotlib in the
same figure so fonts, spines, and rendering are identical across panels.

---

## how to generate data for scatter panels (parameter recovery)

scatter plots of learned vs true parameters (V_rest, tau, W) require running the
learned f_theta / g_phi networks over a voltage domain and fitting linear slopes.
use the helper in `fig_vrest_blank.py` as a template:

1. load config and apply `add_pre_folder` to get the correct `dataset` path prefix:
   ```python
   _, pre = add_pre_folder(config_name)
   config.dataset = pre + config.dataset
   config.config_file = pre + config_name
   ```

2. load ground truth from `ode_params.pt`:
   ```python
   ode_params = OdeCls.load(graphs_data_path(config.dataset), device='cpu')
   gt_vrest = ode_params.gt_vrest(n_neurons)
   ```

3. load best model checkpoint:
   ```python
   ckpt = log_path(config.config_file) + '/models/best_model_with_0_graphs_0.pt'
   state = torch.load(ckpt, weights_only=False)
   migrate_state_dict(state)
   model = create_model(...); model.load_state_dict(state['model_state_dict'], strict=False)
   ```

4. compute per-neuron activity mu/sigma (load only first 2000 frames for speed):
   ```python
   x_ts = load_simulation_data(gdata_dir + '/x_list_train', fields=['voltage'])
   x_ts.voltage = x_ts.voltage[:2000]
   mu, sigma = compute_activity_stats(x_ts, device)
   ```

5. evaluate f_theta over per-neuron domain and fit linear slope/offset:
   ```python
   rr = _vectorized_linspace(mu - 2*sigma, mu + 2*sigma, n_pts=1000, device)
   f  = _batched_mlp_eval(model.f_theta, model.a[:n_neurons], rr,
                          lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f), device)
   slopes, offsets = _vectorized_linear_fit(rr, f)
   learned_vrest = ode_params.derive_vrest(slopes, offsets, n_neurons)
   ```

all imports come from `connectome_gnn.metrics` and `connectome_gnn.utils`.

---

## panel labels
- place labels at the top-left corner of the **outer panel box** (including tick/axis label margins)
  using `get_tightbbox` after rendering — do NOT use `ax.transAxes` offsets which only reference the inner data area
- **all labels must share the same y-coordinate** — use the maximum top across all panels, otherwise
  panels with a title sit higher than panels without one, producing misaligned labels:
  ```python
  fig.canvas.draw()
  renderer = fig.canvas.get_renderer()
  inv = fig.transFigure.inverted()
  all_axes = [ax_a, ax_b, ax_c, ...]
  bboxes   = [ax.get_tightbbox(renderer) for ax in all_axes]
  y1_max   = max(inv.transform((bb.x0, bb.y1))[1] for bb in bboxes)
  for bb, lbl in zip(bboxes, ['a)', 'b)', 'c)', ...]):
      x0 = inv.transform((bb.x0, bb.y1))[0]
      fig.text(x0, y1_max, lbl, fontsize=20, fontweight='bold',
               va='bottom', ha='left', color='black', transform=fig.transFigure)
  ```

## titles
- no uppercase: start lowercase (e.g. `"no blank stimulus"`, not `"No Blank Stimulus"`)
- avoid jargon: use `"blank stimulus"` not `"blank prefix"`
- `ax.set_title(title, fontsize=14, pad=4)`

## style — match GNN_PlotFigure.py scatter plots
**do NOT use `figure_style.py`** — it removes spines and uses 14pt fonts,
which is inconsistent with the scatter plot style used throughout GNN_PlotFigure.

set only the font family globally:
```python
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
})
```

keep spines visible (default matplotlib — do NOT call `style.clean_ax()`).

## font sizes
GNN_PlotFigure standalone scatter plots use `figsize=(10, 9)` with:
- axis labels: 48 pt
- tick labels: 24 pt
- in-plot annotation (R², slope, N): 32 pt

in a composite figure, scale by `subplot_col_width / 10`:
- for a 27 in wide 4-column figure, each col ≈ 6.75 in → scale `_S ≈ 0.52` (paper)
- axis labels: `48 * _S`
- tick labels: `24 * _S`
- annotation: `32 * _S`
- legend: `28 * _S`
- panel subtitle (`ax.set_title`): `17` pt (fixed, not scaled)
- panel labels a) b) c) d): `20` pt (fixed, never scaled)

use `ax.set_xlabel()` / `ax.set_ylabel()` with explicit `fontsize=` — not `style.xlabel()`.

poster figures use a larger scale (e.g. `_S ≈ 0.65`); reduce to `_S ≈ 0.52` for paper.

## layout
- use `constrained_layout=True` in `plt.subplots()` — do NOT use `subplots_adjust` alongside it
- `constrained_layout` ensures axes in the same row are vertically aligned even when some panels
  have titles and others do not
- for a 4-panel single-row figure: `figsize=(27, 6)` gives roughly square subpanels

## saving
```python
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
```
save both PNG (300 dpi) and PDF.
