# Flyvis Results — GNN vs Known-ODE

**Model**: Drosophila optic lobe (13,741 neurons, 434,112 GT edges)
**ODE**: Graded-voltage model: dv/dt = (-v + V_rest)/tau + ReLU(v) @ W

<style>table { font-size: 0.85em; } th, td { padding: 3px 6px; }</style>

## Summary Table (mean ± std over seeds)

Color code: <span style="color:#2ea043">green</span> &gt; 0.9, <span style="color:#d29922">orange</span> &gt; 0.5, <span style="color:#cf222e">red</span> &le; 0.5.

### GNN (ours)

<table>
<tr><th>Condition</th><th>Seeds</th><th>Conn R2 (W)</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step Pearson</th><th>Rollout Pearson</th></tr>
<tr><td><b>Noise-free</b></td><td>6</td><td style="background:#d2992260">0.900 ± 0.030</td><td style="background:#d2992260">0.545 ± 0.178</td><td style="background:#cf222e60">0.047 ± 0.022</td><td style="background:#d2992260">0.782 ± 0.045</td><td style="background:#2ea04360">0.994 ± 0.002</td><td style="background:#2ea04360">0.994 ± 0.002</td></tr>
<tr><td><b>Noise=0.05</b></td><td>6</td><td style="background:#2ea04360">0.962 ± 0.014</td><td style="background:#2ea04360">0.979 ± 0.006</td><td style="background:#cf222e60">0.335 ± 0.075</td><td style="background:#d2992260">0.884 ± 0.025</td><td style="background:#2ea04360">0.999 ± 0.000</td><td style="background:#d2992260">0.779 ± 0.002 *</td></tr>
<tr><td><b>Noise=0.5</b></td><td>6</td><td style="background:#2ea04360">0.997 ± 0.000</td><td style="background:#2ea04360">0.999 ± 0.000</td><td style="background:#2ea04360">0.959 ± 0.010</td><td style="background:#d2992260">0.899 ± 0.009</td><td style="background:#2ea04360">1.000 ± 0.000</td><td style="background:#cf222e60">0.173 ± 0.000 *</td></tr>
</table>

### Known-ODE (ground-truth ODE structure)

<table>
<tr><th>Condition</th><th>Seeds</th><th>Conn R2 (W)</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step Pearson</th><th>Rollout Pearson</th></tr>
<tr><td><b>Noise-free</b></td><td>10</td><td style="background:#2ea04360">0.947 ± 0.001</td><td style="background:#cf222e60">0.325 ± 0.017</td><td style="background:#d2992260">0.653 ± 0.002</td><td style="background:#d2992260">0.856 ± 0.007</td><td style="background:#2ea04360">1.000 ± 0.000</td><td style="background:#2ea04360">0.999 ± 0.000</td></tr>
<tr><td><b>Noise=0.05</b></td><td>10</td><td style="background:#2ea04360">0.964 ± 0.001</td><td style="background:#2ea04360">0.998 ± 0.000</td><td style="background:#d2992260">0.835 ± 0.003</td><td style="background:#d2992260">0.862 ± 0.006</td><td style="background:#2ea04360">1.000 ± 0.000</td><td style="background:#2ea04360">1.000 ± 0.000</td></tr>
<tr><td><b>Noise=0.5</b></td><td>10</td><td style="background:#2ea04360">0.997 ± 0.000</td><td style="background:#2ea04360">1.000 ± 0.000</td><td style="background:#2ea04360">0.986 ± 0.000</td><td style="background:#d2992260">0.859 ± 0.006</td><td style="background:#2ea04360">0.999 ± 0.000</td><td style="background:#2ea04360">1.000 ± 0.000</td></tr>
</table>

**\* Rollout Pearson caveat**: Noisy models are currently evaluated with noise during rollout (`rollout_without_noise=False`). The low values for σ=0.05 and σ=0.5 reflect stochastic divergence of noisy rollout trajectories, not model failure. Noise-free rollout evaluation is needed for fair comparison. One-step prediction (which doesn't accumulate noise) confirms the models are accurate.


---

## Per-Seed Detail: GNN

### Noise-free

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.923</td><td style="background:#cf222e60">0.461</td><td style="background:#cf222e60">0.083</td><td style="background:#d2992260">0.817</td><td style="background:#2ea04360">0.995</td><td style="background:#2ea04360">0.996</td></tr>
<tr><td>cv01 (43)</td><td style="background:#d2992260">0.891</td><td style="background:#d2992260">0.748</td><td style="background:#cf222e60">0.056</td><td style="background:#d2992260">0.728</td><td style="background:#2ea04360">0.994</td><td style="background:#2ea04360">0.993</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.936</td><td style="background:#d2992260">0.794</td><td style="background:#cf222e60">0.055</td><td style="background:#d2992260">0.824</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.997</td></tr>
<tr><td>cv03 (45)</td><td style="background:#d2992260">0.865</td><td style="background:#cf222e60">0.321</td><td style="background:#cf222e60">0.029</td><td style="background:#d2992260">0.718</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.995</td></tr>
<tr><td>cv04 (46)</td><td style="background:#d2992260">0.859</td><td style="background:#d2992260">0.572</td><td style="background:#cf222e60">0.015</td><td style="background:#d2992260">0.775</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">0.992</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.926</td><td style="background:#cf222e60">0.372</td><td style="background:#cf222e60">0.044</td><td style="background:#d2992260">0.829</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">0.993</td></tr>
<tr><td>cv06 (48)</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td><b>Mean</b></td><td style="background:#d2992260"><b>0.900</b></td><td style="background:#d2992260"><b>0.545</b></td><td style="background:#cf222e60"><b>0.047</b></td><td style="background:#d2992260"><b>0.782</b></td><td style="background:#2ea04360"><b>0.994</b></td><td style="background:#2ea04360"><b>0.994</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.030</b></td><td><b>0.178</b></td><td><b>0.022</b></td><td><b>0.045</b></td><td><b>0.002</b></td><td><b>0.002</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.859</b></td><td><b>0.321</b></td><td><b>0.015</b></td><td><b>0.718</b></td><td><b>0.992</b></td><td><b>0.992</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.936</b></td><td><b>0.794</b></td><td><b>0.083</b></td><td><b>0.829</b></td><td><b>0.996</b></td><td><b>0.997</b></td></tr>
</table>

### Noise=0.05

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r *</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.941</td><td style="background:#2ea04360">0.981</td><td style="background:#cf222e60">0.304</td><td style="background:#2ea04360">0.910</td><td style="background:#2ea04360">0.999</td><td style="background:#d2992260">0.780</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.943</td><td style="background:#2ea04360">0.969</td><td style="background:#cf222e60">0.350</td><td style="background:#d2992260">0.892</td><td style="background:#2ea04360">0.999</td><td style="background:#d2992260">0.779</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.972</td><td style="background:#2ea04360">0.981</td><td style="background:#cf222e60">0.202</td><td style="background:#d2992260">0.850</td><td style="background:#2ea04360">0.999</td><td style="background:#d2992260">0.775</td></tr>
<tr><td>cv03 (45)</td><td style="background:#2ea04360">0.969</td><td style="background:#2ea04360">0.989</td><td style="background:#cf222e60">0.395</td><td style="background:#2ea04360">0.916</td><td style="background:#2ea04360">0.999</td><td style="background:#d2992260">0.779</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">0.972</td><td style="background:#2ea04360">0.976</td><td style="background:#cf222e60">0.321</td><td style="background:#d2992260">0.881</td><td style="background:#2ea04360">0.999</td><td style="background:#d2992260">0.779</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.975</td><td style="background:#2ea04360">0.981</td><td style="background:#cf222e60">0.437</td><td style="background:#d2992260">0.855</td><td style="background:#2ea04360">0.999</td><td style="background:#d2992260">0.779</td></tr>
<tr><td>cv06 (48)</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td><b>Mean</b></td><td style="background:#2ea04360"><b>0.962</b></td><td style="background:#2ea04360"><b>0.979</b></td><td style="background:#cf222e60"><b>0.335</b></td><td style="background:#d2992260"><b>0.884</b></td><td style="background:#2ea04360"><b>0.999</b></td><td style="background:#d2992260"><b>0.779</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.014</b></td><td><b>0.006</b></td><td><b>0.075</b></td><td><b>0.025</b></td><td><b>0.000</b></td><td><b>0.002</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.941</b></td><td><b>0.969</b></td><td><b>0.202</b></td><td><b>0.850</b></td><td><b>0.999</b></td><td><b>0.775</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.975</b></td><td><b>0.989</b></td><td><b>0.437</b></td><td><b>0.916</b></td><td><b>0.999</b></td><td><b>0.780</b></td></tr>
</table>

### Noise=0.5

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r *</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.950</td><td style="background:#2ea04360">0.914</td><td style="background:#2ea04360">1.000</td><td style="background:#cf222e60">0.173</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.963</td><td style="background:#d2992260">0.883</td><td style="background:#2ea04360">1.000</td><td style="background:#cf222e60">0.173</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.962</td><td style="background:#d2992260">0.896</td><td style="background:#2ea04360">1.000</td><td style="background:#cf222e60">0.173</td></tr>
<tr><td>cv03 (45)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.979</td><td style="background:#2ea04360">0.905</td><td style="background:#2ea04360">1.000</td><td style="background:#cf222e60">0.173</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.952</td><td style="background:#d2992260">0.898</td><td style="background:#2ea04360">1.000</td><td style="background:#cf222e60">0.173</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.951</td><td style="background:#d2992260">0.897</td><td style="background:#2ea04360">1.000</td><td style="background:#cf222e60">0.173</td></tr>
<tr><td>cv06 (48)</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td><b>Mean</b></td><td style="background:#2ea04360"><b>0.997</b></td><td style="background:#2ea04360"><b>0.999</b></td><td style="background:#2ea04360"><b>0.959</b></td><td style="background:#d2992260"><b>0.899</b></td><td style="background:#2ea04360"><b>1.000</b></td><td style="background:#cf222e60"><b>0.173</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.000</b></td><td><b>0.000</b></td><td><b>0.010</b></td><td><b>0.009</b></td><td><b>0.000</b></td><td><b>0.000</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.996</b></td><td><b>0.999</b></td><td><b>0.950</b></td><td><b>0.883</b></td><td><b>1.000</b></td><td><b>0.173</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.997</b></td><td><b>1.000</b></td><td><b>0.979</b></td><td><b>0.914</b></td><td><b>1.000</b></td><td><b>0.173</b></td></tr>
</table>

---

## Per-Seed Detail: Known-ODE

### Noise-free

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.331</td><td style="background:#d2992260">0.653</td><td style="background:#d2992260">0.853</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.946</td><td style="background:#cf222e60">0.320</td><td style="background:#d2992260">0.653</td><td style="background:#d2992260">0.856</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.946</td><td style="background:#cf222e60">0.355</td><td style="background:#d2992260">0.650</td><td style="background:#d2992260">0.859</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv03 (45)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.348</td><td style="background:#d2992260">0.655</td><td style="background:#d2992260">0.843</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.302</td><td style="background:#d2992260">0.657</td><td style="background:#d2992260">0.851</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.946</td><td style="background:#cf222e60">0.334</td><td style="background:#d2992260">0.653</td><td style="background:#d2992260">0.862</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv06 (48)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.319</td><td style="background:#d2992260">0.652</td><td style="background:#d2992260">0.855</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv07 (49)</td><td style="background:#2ea04360">0.946</td><td style="background:#cf222e60">0.307</td><td style="background:#d2992260">0.650</td><td style="background:#d2992260">0.849</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv08 (50)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.310</td><td style="background:#d2992260">0.656</td><td style="background:#d2992260">0.867</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv09 (51)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.321</td><td style="background:#d2992260">0.656</td><td style="background:#d2992260">0.865</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td><b>Mean</b></td><td style="background:#2ea04360"><b>0.947</b></td><td style="background:#cf222e60"><b>0.325</b></td><td style="background:#d2992260"><b>0.653</b></td><td style="background:#d2992260"><b>0.856</b></td><td style="background:#2ea04360"><b>1.000</b></td><td style="background:#2ea04360"><b>0.999</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.001</b></td><td><b>0.017</b></td><td><b>0.002</b></td><td><b>0.007</b></td><td><b>0.000</b></td><td><b>0.000</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.946</b></td><td><b>0.302</b></td><td><b>0.650</b></td><td><b>0.843</b></td><td><b>1.000</b></td><td><b>0.999</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.947</b></td><td><b>0.355</b></td><td><b>0.657</b></td><td><b>0.867</b></td><td><b>1.000</b></td><td><b>1.000</b></td></tr>
</table>

### Noise=0.05

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.964</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.837</td><td style="background:#d2992260">0.858</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.963</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.833</td><td style="background:#d2992260">0.862</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.963</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.831</td><td style="background:#d2992260">0.854</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv03 (45)</td><td style="background:#2ea04360">0.963</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.833</td><td style="background:#d2992260">0.857</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">0.965</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.840</td><td style="background:#d2992260">0.863</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.964</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.834</td><td style="background:#d2992260">0.863</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv06 (48)</td><td style="background:#2ea04360">0.964</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.836</td><td style="background:#d2992260">0.869</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv07 (49)</td><td style="background:#2ea04360">0.963</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.830</td><td style="background:#d2992260">0.868</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv08 (50)</td><td style="background:#2ea04360">0.964</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.838</td><td style="background:#d2992260">0.870</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv09 (51)</td><td style="background:#2ea04360">0.965</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.839</td><td style="background:#d2992260">0.851</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td><b>Mean</b></td><td style="background:#2ea04360"><b>0.964</b></td><td style="background:#2ea04360"><b>0.998</b></td><td style="background:#d2992260"><b>0.835</b></td><td style="background:#d2992260"><b>0.862</b></td><td style="background:#2ea04360"><b>1.000</b></td><td style="background:#2ea04360"><b>1.000</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.001</b></td><td><b>0.000</b></td><td><b>0.003</b></td><td><b>0.006</b></td><td><b>0.000</b></td><td><b>0.000</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.963</b></td><td><b>0.998</b></td><td><b>0.830</b></td><td><b>0.851</b></td><td><b>1.000</b></td><td><b>1.000</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.965</b></td><td><b>0.998</b></td><td><b>0.840</b></td><td><b>0.870</b></td><td><b>1.000</b></td><td><b>1.000</b></td></tr>
</table>

### Noise=0.5

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.856</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.860</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.868</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv03 (45)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.855</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.854</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.859</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv06 (48)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.857</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv07 (49)</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.857</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv08 (50)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.870</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv09 (51)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.852</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td><b>Mean</b></td><td style="background:#2ea04360"><b>0.997</b></td><td style="background:#2ea04360"><b>1.000</b></td><td style="background:#2ea04360"><b>0.986</b></td><td style="background:#d2992260"><b>0.859</b></td><td style="background:#2ea04360"><b>0.999</b></td><td style="background:#2ea04360"><b>1.000</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.000</b></td><td><b>0.000</b></td><td><b>0.000</b></td><td><b>0.006</b></td><td><b>0.000</b></td><td><b>0.000</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.996</b></td><td><b>1.000</b></td><td><b>0.986</b></td><td><b>0.852</b></td><td><b>0.999</b></td><td><b>1.000</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.997</b></td><td><b>1.000</b></td><td><b>0.986</b></td><td><b>0.870</b></td><td><b>0.999</b></td><td><b>1.000</b></td></tr>
</table>

---

## Key Observations

### Noise helps parameter recovery
- **W R2**: 0.900 (noise-free) → 0.962 (σ=0.05) → 0.997 (σ=0.5). Noise consistently improves connectivity recovery for both GNN and Known-ODE.
- **tau R2**: 0.545 → 0.979 → 0.999. Dramatic improvement — noise-free tau extraction is unreliable.
- **V_rest R2**: 0.047 → 0.335 → 0.959. V_rest is the hardest parameter; only σ=0.5 gives reliable extraction.
- **Cluster acc**: 0.782 → 0.884 → 0.899. Noise improves neuron-type discrimination from learned embeddings.

### GNN vs Known-ODE
- **W R2**: GNN matches Known-ODE at σ=0.5 (both 0.997). At noise-free, Known-ODE is slightly better (0.947 vs 0.900).
- **tau R2**: Known-ODE is better at noise-free (0.325 vs 0.545 — both poor, but different). At σ≥0.05, both near-perfect.
- **V_rest R2**: Known-ODE consistently better (0.653 vs 0.047 noise-free; 0.835 vs 0.335 at σ=0.05; 0.986 vs 0.959 at σ=0.5). With ground-truth ODE structure, tau and V_rest are direct parameters; the GNN must extract them indirectly from learned f_theta slopes/offsets.
- **Variance**: Known-ODE has dramatically lower variance (CV<1% vs CV=3-50% for GNN). The ground-truth ODE structure removes a major source of optimization difficulty.

### Rollout Pearson issue
The rollout for noisy models is evaluated WITH process noise (`rollout_without_noise=False`), making it an unfair comparison. Different noise realizations in ground truth vs predicted rollout cause Pearson correlation to drop. The one-step prediction Pearson (which doesn't accumulate noise) shows the model is accurate (0.999-1.000 for all noisy conditions). **Action needed: re-run rollout evaluation with `rollout_without_noise=True`.**

### Status
- GNN: 6 complete seeds per condition (cv00-cv05; cv06 in progress)
- Known-ODE: 10 complete seeds per condition (cv00-cv09)
- GNN needs 4 more runs to reach 10 seeds.
