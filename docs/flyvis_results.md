# Flyvis Results — GNN vs Known-ODE

**Model**: Drosophila optic lobe (13,741 neurons, 434,112 GT edges)
**ODE**: Graded-voltage model: dv/dt = (-v + V_rest)/tau + ReLU(v) @ W

<style>table { font-size: 0.85em; } th, td { padding: 3px 6px; }</style>

## Summary Table (mean &pm; std over seeds)

Color code: <span style="color:#2ea043">green</span> &gt; 0.9, <span style="color:#d29922">orange</span> &gt; 0.5, <span style="color:#cf222e">red</span> &le; 0.5.

### GNN (LLM-optimized)

<table>
<tr><th>Condition</th><th>Seeds</th><th>Conn R2 (W)</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step Pearson</th><th>Rollout Pearson</th></tr>
<tr><td><b>Noise-free</b></td><td>10</td><td style="background:#d2992260">0.899 &pm; 0.030</td><td style="background:#d2992260">0.511 &pm; 0.147</td><td style="background:#cf222e60">0.048 &pm; 0.037</td><td style="background:#d2992260">0.785 &pm; 0.035</td><td style="background:#2ea04360">0.994 &pm; 0.001</td><td style="background:#2ea04360">0.994 &pm; 0.001</td></tr>
<tr><td><b>Noise=0.05</b></td><td>10</td><td style="background:#2ea04360">0.962 &pm; 0.012</td><td style="background:#2ea04360">0.980 &pm; 0.007</td><td style="background:#cf222e60">0.300 &pm; 0.076</td><td style="background:#d2992260">0.880 &pm; 0.027</td><td style="background:#2ea04360">0.996 &pm; 0.004</td><td style="background:#2ea04360">0.930 &pm; 0.099</td></tr>
<tr><td><b>Noise=0.5</b></td><td>10</td><td style="background:#2ea04360">0.997 &pm; 0.000</td><td style="background:#2ea04360">0.999 &pm; 0.000</td><td style="background:#2ea04360">0.957 &pm; 0.010</td><td style="background:#d2992260">0.898 &pm; 0.009</td><td style="background:#2ea04360">0.989 &pm; 0.011</td><td style="background:#d2992260">0.739 &pm; 0.371</td></tr>
</table>

### GNN (default)

<table>
<tr><th>Condition</th><th>Seeds</th><th>Conn R2 (W)</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step Pearson</th><th>Rollout Pearson</th></tr>
<tr><td><b>Noise-free</b></td><td>10</td><td style="background:#d2992260">0.740 &pm; 0.066</td><td style="background:#cf222e60">0.204 &pm; 0.228</td><td style="background:#cf222e60">0.063 &pm; 0.052</td><td style="background:#d2992260">0.629 &pm; 0.048</td><td style="background:#2ea04360">0.986 &pm; 0.008</td><td style="background:#d2992260">0.651 &pm; 0.422</td></tr>
<tr><td><b>Noise=0.05</b></td><td>10</td><td style="background:#d2992260">0.698 &pm; 0.155</td><td style="background:#2ea04360">0.973 &pm; 0.027</td><td style="background:#cf222e60">0.350 &pm; 0.135</td><td style="background:#d2992260">0.722 &pm; 0.034</td><td style="background:#2ea04360">0.993 &pm; 0.003</td><td style="background:#2ea04360">0.992 &pm; 0.004</td></tr>
<tr><td><b>Noise=0.5</b></td><td>10</td><td style="background:#2ea04360">0.997 &pm; 0.004</td><td style="background:#2ea04360">0.999 &pm; 0.001</td><td style="background:#d2992260">0.842 &pm; 0.066</td><td style="background:#d2992260">0.843 &pm; 0.016</td><td style="background:#2ea04360">0.990 &pm; 0.008</td><td style="background:#2ea04360">0.989 &pm; 0.010</td></tr>
</table>

### Known-ODE

<table>
<tr><th>Condition</th><th>Seeds</th><th>Conn R2 (W)</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step Pearson</th><th>Rollout Pearson</th></tr>
<tr><td><b>Noise-free</b></td><td>10</td><td style="background:#2ea04360">0.947 &pm; 0.000</td><td style="background:#cf222e60">0.325 &pm; 0.016</td><td style="background:#d2992260">0.653 &pm; 0.002</td><td style="background:#d2992260">0.856 &pm; 0.007</td><td style="background:#2ea04360">1.000 &pm; 0.000</td><td style="background:#2ea04360">0.999 &pm; 0.000</td></tr>
<tr><td><b>Noise=0.05</b></td><td>10</td><td style="background:#2ea04360">0.964 &pm; 0.001</td><td style="background:#2ea04360">0.998 &pm; 0.000</td><td style="background:#d2992260">0.835 &pm; 0.003</td><td style="background:#d2992260">0.862 &pm; 0.006</td><td style="background:#2ea04360">1.000 &pm; 0.000</td><td style="background:#2ea04360">1.000 &pm; 0.000</td></tr>
<tr><td><b>Noise=0.5</b></td><td>10</td><td style="background:#2ea04360">0.997 &pm; 0.000</td><td style="background:#2ea04360">1.000 &pm; 0.000</td><td style="background:#2ea04360">0.986 &pm; 0.000</td><td style="background:#d2992260">0.859 &pm; 0.006</td><td style="background:#2ea04360">0.999 &pm; 0.000</td><td style="background:#2ea04360">1.000 &pm; 0.000</td></tr>
</table>

---

## Per-Seed Detail

### GNN (LLM-optimized) — Noise-free

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.923</td><td style="background:#cf222e60">0.461</td><td style="background:#cf222e60">0.083</td><td style="background:#d2992260">0.817</td><td style="background:#2ea04360">0.995</td><td style="background:#2ea04360">0.996</td></tr>
<tr><td>cv01 (43)</td><td style="background:#d2992260">0.890</td><td style="background:#d2992260">0.748</td><td style="background:#cf222e60">0.056</td><td style="background:#d2992260">0.728</td><td style="background:#2ea04360">0.994</td><td style="background:#2ea04360">0.993</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.936</td><td style="background:#d2992260">0.794</td><td style="background:#cf222e60">0.055</td><td style="background:#d2992260">0.824</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.997</td></tr>
<tr><td>cv03 (45)</td><td style="background:#d2992260">0.865</td><td style="background:#cf222e60">0.321</td><td style="background:#cf222e60">0.029</td><td style="background:#d2992260">0.718</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.995</td></tr>
<tr><td>cv04 (46)</td><td style="background:#d2992260">0.859</td><td style="background:#d2992260">0.572</td><td style="background:#cf222e60">0.015</td><td style="background:#d2992260">0.775</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">0.992</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.926</td><td style="background:#cf222e60">0.372</td><td style="background:#cf222e60">0.044</td><td style="background:#d2992260">0.829</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">0.993</td></tr>
<tr><td>cv06 (48)</td><td style="background:#d2992260">0.855</td><td style="background:#d2992260">0.503</td><td style="background:#cf222e60">0.004</td><td style="background:#d2992260">0.781</td><td style="background:#2ea04360">0.995</td><td style="background:#2ea04360">0.994</td></tr>
<tr><td>cv07 (49)</td><td style="background:#d2992260">0.889</td><td style="background:#cf222e60">0.422</td><td style="background:#cf222e60">0.133</td><td style="background:#d2992260">0.786</td><td style="background:#2ea04360">0.995</td><td style="background:#2ea04360">0.995</td></tr>
<tr><td>cv08 (50)</td><td style="background:#2ea04360">0.920</td><td style="background:#d2992260">0.515</td><td style="background:#cf222e60">0.054</td><td style="background:#d2992260">0.794</td><td style="background:#2ea04360">0.994</td><td style="background:#2ea04360">0.994</td></tr>
<tr><td>cv09 (51)</td><td style="background:#2ea04360">0.930</td><td style="background:#cf222e60">0.405</td><td style="background:#cf222e60">0.007</td><td style="background:#d2992260">0.798</td><td style="background:#2ea04360">0.994</td><td style="background:#2ea04360">0.994</td></tr>
<tr><td><b>Mean</b></td><td style="background:#d2992260"><b>0.899</b></td><td style="background:#d2992260"><b>0.511</b></td><td style="background:#cf222e60"><b>0.048</b></td><td style="background:#d2992260"><b>0.785</b></td><td style="background:#2ea04360"><b>0.994</b></td><td style="background:#2ea04360"><b>0.994</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.030</b></td><td><b>0.147</b></td><td><b>0.037</b></td><td><b>0.035</b></td><td><b>0.001</b></td><td><b>0.001</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.855</b></td><td><b>0.321</b></td><td><b>0.004</b></td><td><b>0.718</b></td><td><b>0.992</b></td><td><b>0.992</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.936</b></td><td><b>0.794</b></td><td><b>0.133</b></td><td><b>0.829</b></td><td><b>0.996</b></td><td><b>0.997</b></td></tr>
</table>

### GNN (LLM-optimized) — Noise=0.05

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.941</td><td style="background:#2ea04360">0.981</td><td style="background:#cf222e60">0.304</td><td style="background:#2ea04360">0.910</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.997</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.943</td><td style="background:#2ea04360">0.969</td><td style="background:#cf222e60">0.350</td><td style="background:#d2992260">0.892</td><td style="background:#2ea04360">0.998</td><td style="background:#2ea04360">0.998</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.972</td><td style="background:#2ea04360">0.981</td><td style="background:#cf222e60">0.201</td><td style="background:#d2992260">0.850</td><td style="background:#2ea04360">0.988</td><td style="background:#2ea04360">0.985</td></tr>
<tr><td>cv03 (45)</td><td style="background:#2ea04360">0.969</td><td style="background:#2ea04360">0.989</td><td style="background:#cf222e60">0.395</td><td style="background:#2ea04360">0.916</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">0.972</td><td style="background:#2ea04360">0.976</td><td style="background:#cf222e60">0.321</td><td style="background:#d2992260">0.881</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.975</td><td style="background:#2ea04360">0.981</td><td style="background:#cf222e60">0.437</td><td style="background:#d2992260">0.855</td><td style="background:#2ea04360">0.990</td><td style="background:#2ea04360">0.991</td></tr>
<tr><td>cv06 (48)</td><td style="background:#2ea04360">0.976</td><td style="background:#2ea04360">0.988</td><td style="background:#cf222e60">0.314</td><td style="background:#d2992260">0.892</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv07 (49)</td><td style="background:#2ea04360">0.952</td><td style="background:#2ea04360">0.972</td><td style="background:#cf222e60">0.241</td><td style="background:#d2992260">0.826</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.777</td></tr>
<tr><td>cv08 (50)</td><td style="background:#2ea04360">0.962</td><td style="background:#2ea04360">0.990</td><td style="background:#cf222e60">0.228</td><td style="background:#d2992260">0.899</td><td style="background:#2ea04360">0.999</td><td style="background:#d2992260">0.780</td></tr>
<tr><td>cv09 (51)</td><td style="background:#2ea04360">0.960</td><td style="background:#2ea04360">0.975</td><td style="background:#cf222e60">0.207</td><td style="background:#d2992260">0.882</td><td style="background:#2ea04360">0.999</td><td style="background:#d2992260">0.779</td></tr>
<tr><td><b>Mean</b></td><td style="background:#2ea04360"><b>0.962</b></td><td style="background:#2ea04360"><b>0.980</b></td><td style="background:#cf222e60"><b>0.300</b></td><td style="background:#d2992260"><b>0.880</b></td><td style="background:#2ea04360"><b>0.996</b></td><td style="background:#2ea04360"><b>0.930</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.012</b></td><td><b>0.007</b></td><td><b>0.076</b></td><td><b>0.027</b></td><td><b>0.004</b></td><td><b>0.099</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.941</b></td><td><b>0.969</b></td><td><b>0.201</b></td><td><b>0.826</b></td><td><b>0.988</b></td><td><b>0.777</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.976</b></td><td><b>0.990</b></td><td><b>0.437</b></td><td><b>0.916</b></td><td><b>0.999</b></td><td><b>0.999</b></td></tr>
</table>

### GNN (LLM-optimized) — Noise=0.5

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.950</td><td style="background:#2ea04360">0.914</td><td style="background:#2ea04360">0.973</td><td style="background:#2ea04360">0.969</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.963</td><td style="background:#d2992260">0.883</td><td style="background:#2ea04360">0.982</td><td style="background:#2ea04360">0.979</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.962</td><td style="background:#d2992260">0.896</td><td style="background:#2ea04360">0.973</td><td style="background:#2ea04360">0.974</td></tr>
<tr><td>cv03 (45)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.979</td><td style="background:#2ea04360">0.905</td><td style="background:#2ea04360">0.998</td><td style="background:#2ea04360">0.993</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.952</td><td style="background:#d2992260">0.898</td><td style="background:#2ea04360">0.985</td><td style="background:#2ea04360">0.982</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.951</td><td style="background:#d2992260">0.897</td><td style="background:#2ea04360">0.986</td><td style="background:#2ea04360">0.983</td></tr>
<tr><td>cv06 (48)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.960</td><td style="background:#d2992260">0.887</td><td style="background:#2ea04360">0.998</td><td style="background:#2ea04360">0.996</td></tr>
<tr><td>cv07 (49)</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.944</td><td style="background:#d2992260">0.896</td><td style="background:#2ea04360">1.000</td><td style="background:#cf222e60">0.172</td></tr>
<tr><td>cv08 (50)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.945</td><td style="background:#d2992260">0.899</td><td style="background:#2ea04360">1.000</td><td style="background:#cf222e60">0.173</td></tr>
<tr><td>cv09 (51)</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.965</td><td style="background:#2ea04360">0.906</td><td style="background:#2ea04360">1.000</td><td style="background:#cf222e60">0.173</td></tr>
<tr><td><b>Mean</b></td><td style="background:#2ea04360"><b>0.997</b></td><td style="background:#2ea04360"><b>0.999</b></td><td style="background:#2ea04360"><b>0.957</b></td><td style="background:#d2992260"><b>0.898</b></td><td style="background:#2ea04360"><b>0.989</b></td><td style="background:#d2992260"><b>0.739</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.000</b></td><td><b>0.000</b></td><td><b>0.010</b></td><td><b>0.009</b></td><td><b>0.011</b></td><td><b>0.371</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.996</b></td><td><b>0.999</b></td><td><b>0.944</b></td><td><b>0.883</b></td><td><b>0.973</b></td><td><b>0.172</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.997</b></td><td><b>1.000</b></td><td><b>0.979</b></td><td><b>0.914</b></td><td><b>1.000</b></td><td><b>0.996</b></td></tr>
</table>

### GNN (default) — Noise-free

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#d2992260">0.697</td><td style="background:#d2992260">0.533</td><td style="background:#cf222e60">0.152</td><td style="background:#d2992260">0.643</td><td style="background:#2ea04360">0.985</td><td style="background:#cf222e60">0.108</td></tr>
<tr><td>cv01 (43)</td><td style="background:#d2992260">0.851</td><td style="background:#cf222e60">0.241</td><td style="background:#cf222e60">0.116</td><td style="background:#d2992260">0.663</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">0.991</td></tr>
<tr><td>cv02 (44)</td><td style="background:#d2992260">0.776</td><td style="background:#cf222e60">0.037</td><td style="background:#cf222e60">0.094</td><td style="background:#d2992260">0.681</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">0.992</td></tr>
<tr><td>cv03 (45)</td><td style="background:#d2992260">0.762</td><td style="background:#d2992260">0.515</td><td style="background:#cf222e60">0.008</td><td style="background:#d2992260">0.609</td><td style="background:#2ea04360">0.988</td><td style="background:#2ea04360">0.987</td></tr>
<tr><td>cv04 (46)</td><td style="background:#d2992260">0.778</td><td style="background:#cf222e60">0.034</td><td style="background:#cf222e60">0.121</td><td style="background:#d2992260">0.688</td><td style="background:#2ea04360">0.991</td><td style="background:#2ea04360">0.990</td></tr>
<tr><td>cv05 (47)</td><td style="background:#d2992260">0.788</td><td style="background:#cf222e60">0.054</td><td style="background:#cf222e60">0.005</td><td style="background:#d2992260">0.606</td><td style="background:#2ea04360">0.991</td><td style="background:#2ea04360">0.988</td></tr>
<tr><td>cv06 (48)</td><td style="background:#d2992260">0.736</td><td style="background:#cf222e60">0.025</td><td style="background:#cf222e60">0.044</td><td style="background:#d2992260">0.664</td><td style="background:#2ea04360">0.988</td><td style="background:#cf222e60">0.032</td></tr>
<tr><td>cv07 (49)</td><td style="background:#d2992260">0.644</td><td style="background:#cf222e60">0.034</td><td style="background:#cf222e60">0.012</td><td style="background:#d2992260">0.606</td><td style="background:#2ea04360">0.975</td><td style="background:#2ea04360">0.977</td></tr>
<tr><td>cv08 (50)</td><td style="background:#d2992260">0.619</td><td style="background:#cf222e60">0.002</td><td style="background:#cf222e60">0.058</td><td style="background:#d2992260">0.517</td><td style="background:#2ea04360">0.967</td><td style="background:#cf222e60">0.049</td></tr>
<tr><td>cv09 (51)</td><td style="background:#d2992260">0.749</td><td style="background:#d2992260">0.568</td><td style="background:#cf222e60">0.017</td><td style="background:#d2992260">0.612</td><td style="background:#2ea04360">0.987</td><td style="background:#cf222e60">0.399</td></tr>
<tr><td><b>Mean</b></td><td style="background:#d2992260"><b>0.740</b></td><td style="background:#cf222e60"><b>0.204</b></td><td style="background:#cf222e60"><b>0.063</b></td><td style="background:#d2992260"><b>0.629</b></td><td style="background:#2ea04360"><b>0.986</b></td><td style="background:#d2992260"><b>0.651</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.066</b></td><td><b>0.228</b></td><td><b>0.052</b></td><td><b>0.048</b></td><td><b>0.008</b></td><td><b>0.422</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.619</b></td><td><b>0.002</b></td><td><b>0.005</b></td><td><b>0.517</b></td><td><b>0.967</b></td><td><b>0.032</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.851</b></td><td><b>0.568</b></td><td><b>0.152</b></td><td><b>0.688</b></td><td><b>0.992</b></td><td><b>0.992</b></td></tr>
</table>

### GNN (default) — Noise=0.05

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#d2992260">0.750</td><td style="background:#2ea04360">0.987</td><td style="background:#cf222e60">0.395</td><td style="background:#d2992260">0.717</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">0.997</td></tr>
<tr><td>cv01 (43)</td><td style="background:#d2992260">0.798</td><td style="background:#2ea04360">0.980</td><td style="background:#cf222e60">0.347</td><td style="background:#d2992260">0.724</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.996</td></tr>
<tr><td>cv02 (44)</td><td style="background:#cf222e60">0.489</td><td style="background:#2ea04360">0.988</td><td style="background:#cf222e60">0.462</td><td style="background:#d2992260">0.707</td><td style="background:#2ea04360">0.993</td><td style="background:#2ea04360">0.992</td></tr>
<tr><td>cv03 (45)</td><td style="background:#d2992260">0.759</td><td style="background:#2ea04360">0.983</td><td style="background:#cf222e60">0.300</td><td style="background:#d2992260">0.755</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.993</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">0.930</td><td style="background:#2ea04360">0.994</td><td style="background:#cf222e60">0.486</td><td style="background:#d2992260">0.784</td><td style="background:#2ea04360">0.991</td><td style="background:#2ea04360">0.990</td></tr>
<tr><td>cv05 (47)</td><td style="background:#d2992260">0.619</td><td style="background:#2ea04360">0.956</td><td style="background:#cf222e60">0.367</td><td style="background:#d2992260">0.709</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.994</td></tr>
<tr><td>cv06 (48)</td><td style="background:#cf222e60">0.454</td><td style="background:#2ea04360">0.986</td><td style="background:#d2992260">0.512</td><td style="background:#d2992260">0.735</td><td style="background:#2ea04360">0.985</td><td style="background:#2ea04360">0.982</td></tr>
<tr><td>cv07 (49)</td><td style="background:#d2992260">0.888</td><td style="background:#2ea04360">0.983</td><td style="background:#cf222e60">0.366</td><td style="background:#d2992260">0.739</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">0.991</td></tr>
<tr><td>cv08 (50)</td><td style="background:#d2992260">0.548</td><td style="background:#2ea04360">0.973</td><td style="background:#cf222e60">0.248</td><td style="background:#d2992260">0.696</td><td style="background:#2ea04360">0.991</td><td style="background:#2ea04360">0.992</td></tr>
<tr><td>cv09 (51)</td><td style="background:#d2992260">0.748</td><td style="background:#d2992260">0.896</td><td style="background:#cf222e60">0.021</td><td style="background:#d2992260">0.649</td><td style="background:#2ea04360">0.993</td><td style="background:#2ea04360">0.992</td></tr>
<tr><td><b>Mean</b></td><td style="background:#d2992260"><b>0.698</b></td><td style="background:#2ea04360"><b>0.973</b></td><td style="background:#cf222e60"><b>0.350</b></td><td style="background:#d2992260"><b>0.722</b></td><td style="background:#2ea04360"><b>0.993</b></td><td style="background:#2ea04360"><b>0.992</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.155</b></td><td><b>0.027</b></td><td><b>0.135</b></td><td><b>0.034</b></td><td><b>0.003</b></td><td><b>0.004</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.454</b></td><td><b>0.896</b></td><td><b>0.021</b></td><td><b>0.649</b></td><td><b>0.985</b></td><td><b>0.982</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.930</b></td><td><b>0.994</b></td><td><b>0.512</b></td><td><b>0.784</b></td><td><b>0.997</b></td><td><b>0.997</b></td></tr>
</table>

### GNN (default) — Noise=0.5

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td><td style="background:#d2992260">0.874</td><td style="background:#d2992260">0.822</td><td style="background:#2ea04360">0.980</td><td style="background:#2ea04360">0.977</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td><td style="background:#d2992260">0.884</td><td style="background:#d2992260">0.837</td><td style="background:#2ea04360">0.995</td><td style="background:#2ea04360">0.995</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.957</td><td style="background:#d2992260">0.829</td><td style="background:#2ea04360">0.972</td><td style="background:#2ea04360">0.967</td></tr>
<tr><td>cv03 (45)</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.914</td><td style="background:#d2992260">0.834</td><td style="background:#2ea04360">0.993</td><td style="background:#2ea04360">0.991</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td><td style="background:#d2992260">0.814</td><td style="background:#d2992260">0.823</td><td style="background:#2ea04360">0.993</td><td style="background:#2ea04360">0.995</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">0.997</td><td style="background:#d2992260">0.838</td><td style="background:#d2992260">0.857</td><td style="background:#2ea04360">0.993</td><td style="background:#2ea04360">0.982</td></tr>
<tr><td>cv06 (48)</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td><td style="background:#d2992260">0.844</td><td style="background:#d2992260">0.846</td><td style="background:#2ea04360">0.993</td><td style="background:#2ea04360">0.992</td></tr>
<tr><td>cv07 (49)</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">1.000</td><td style="background:#d2992260">0.718</td><td style="background:#d2992260">0.844</td><td style="background:#2ea04360">0.995</td><td style="background:#2ea04360">0.995</td></tr>
<tr><td>cv08 (50)</td><td style="background:#2ea04360">0.999</td><td style="background:#2ea04360">0.999</td><td style="background:#d2992260">0.775</td><td style="background:#d2992260">0.869</td><td style="background:#2ea04360">0.995</td><td style="background:#2ea04360">0.996</td></tr>
<tr><td>cv09 (51)</td><td style="background:#2ea04360">0.987</td><td style="background:#2ea04360">0.997</td><td style="background:#d2992260">0.805</td><td style="background:#d2992260">0.868</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.997</td></tr>
<tr><td><b>Mean</b></td><td style="background:#2ea04360"><b>0.997</b></td><td style="background:#2ea04360"><b>0.999</b></td><td style="background:#d2992260"><b>0.842</b></td><td style="background:#d2992260"><b>0.843</b></td><td style="background:#2ea04360"><b>0.990</b></td><td style="background:#2ea04360"><b>0.989</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.004</b></td><td><b>0.001</b></td><td><b>0.066</b></td><td><b>0.016</b></td><td><b>0.008</b></td><td><b>0.010</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.987</b></td><td><b>0.997</b></td><td><b>0.718</b></td><td><b>0.822</b></td><td><b>0.972</b></td><td><b>0.967</b></td></tr>
<tr><td><b>Max</b></td><td><b>1.000</b></td><td><b>1.000</b></td><td><b>0.957</b></td><td><b>0.869</b></td><td><b>0.996</b></td><td><b>0.997</b></td></tr>
</table>

### Known-ODE — Noise-free

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.331</td><td style="background:#d2992260">0.653</td><td style="background:#d2992260">0.853</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.946</td><td style="background:#cf222e60">0.319</td><td style="background:#d2992260">0.653</td><td style="background:#d2992260">0.856</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv02 (44)</td><td style="background:#2ea04360">0.946</td><td style="background:#cf222e60">0.355</td><td style="background:#d2992260">0.650</td><td style="background:#d2992260">0.859</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv03 (45)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.348</td><td style="background:#d2992260">0.655</td><td style="background:#d2992260">0.843</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv04 (46)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.302</td><td style="background:#d2992260">0.657</td><td style="background:#d2992260">0.851</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv05 (47)</td><td style="background:#2ea04360">0.946</td><td style="background:#cf222e60">0.334</td><td style="background:#d2992260">0.653</td><td style="background:#d2992260">0.862</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv06 (48)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.319</td><td style="background:#d2992260">0.652</td><td style="background:#d2992260">0.855</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv07 (49)</td><td style="background:#2ea04360">0.946</td><td style="background:#cf222e60">0.307</td><td style="background:#d2992260">0.650</td><td style="background:#d2992260">0.849</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv08 (50)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.310</td><td style="background:#d2992260">0.656</td><td style="background:#d2992260">0.867</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>cv09 (51)</td><td style="background:#2ea04360">0.947</td><td style="background:#cf222e60">0.321</td><td style="background:#d2992260">0.656</td><td style="background:#d2992260">0.865</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td><b>Mean</b></td><td style="background:#2ea04360"><b>0.947</b></td><td style="background:#cf222e60"><b>0.325</b></td><td style="background:#d2992260"><b>0.653</b></td><td style="background:#d2992260"><b>0.856</b></td><td style="background:#2ea04360"><b>1.000</b></td><td style="background:#2ea04360"><b>0.999</b></td></tr>
<tr><td><b>Std</b></td><td><b>0.000</b></td><td><b>0.016</b></td><td><b>0.002</b></td><td><b>0.007</b></td><td><b>0.000</b></td><td><b>0.000</b></td></tr>
<tr><td><b>Min</b></td><td><b>0.946</b></td><td><b>0.302</b></td><td><b>0.650</b></td><td><b>0.843</b></td><td><b>1.000</b></td><td><b>0.999</b></td></tr>
<tr><td><b>Max</b></td><td><b>0.947</b></td><td><b>0.355</b></td><td><b>0.657</b></td><td><b>0.867</b></td><td><b>1.000</b></td><td><b>1.000</b></td></tr>
</table>

### Known-ODE — Noise=0.05

<table>
<tr><th>Seed</th><th>W R2</th><th>tau R2</th><th>V_rest R2</th><th>Cluster acc</th><th>One-step r</th><th>Rollout r</th></tr>
<tr><td>cv00 (42)</td><td style="background:#2ea04360">0.964</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.837</td><td style="background:#d2992260">0.858</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>cv01 (43)</td><td style="background:#2ea04360">0.963</td><td style="background:#2ea04360">0.998</td><td style="background:#d2992260">0.832</td><td style="background:#d2992260">0.862</td><td style="background:#2ea04360">1.000</td><td style="background:#2ea04360">1.000</td></tr>
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

### Known-ODE — Noise=0.5

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

### LLM exploration improves over default config
- **Noise-free**: **W R2**: 0.740 → 0.899 (+22%), **tau R2**: 0.204 → 0.511 (+150%), **V_rest R2**: 0.063 → 0.048 (-23%), **Cluster acc**: 0.629 → 0.785 (+25%)
- **Noise=0.05**: **W R2**: 0.698 → 0.962 (+38%), **tau R2**: 0.973 → 0.980 (+1%), **V_rest R2**: 0.350 → 0.300 (-14%), **Cluster acc**: 0.722 → 0.880 (+22%)
- **Noise=0.5**: **W R2**: 0.997 → 0.997 (-0%), **tau R2**: 0.999 → 0.999 (+0%), **V_rest R2**: 0.842 → 0.957 (+14%), **Cluster acc**: 0.843 → 0.898 (+7%)

### Noise helps parameter recovery
- **W R2**: 0.899 (noise-free) → 0.962 (σ=0.05) → 0.997 (σ=0.5)
- **tau R2**: 0.511 (noise-free) → 0.980 (σ=0.05) → 0.999 (σ=0.5)
- **V_rest R2**: 0.048 (noise-free) → 0.300 (σ=0.05) → 0.957 (σ=0.5)
- **Cluster acc**: 0.785 (noise-free) → 0.880 (σ=0.05) → 0.898 (σ=0.5)

### GNN vs Known-ODE
- Known-ODE has near-zero variance across seeds (ground-truth ODE structure removes optimization difficulty)
- GNN matches Known-ODE at high noise (σ=0.5) for W R2 (both 0.997)
- Known-ODE consistently better for V_rest (direct parameter vs indirect extraction from f_theta)

### Status
- GNN (LLM-optimized): 10 (Noise-free), 10 (Noise=0.05), 10 (Noise=0.5)
- GNN (default): 10 (Noise-free), 10 (Noise=0.05), 10 (Noise=0.5)
- Known-ODE: 10 (Noise-free), 10 (Noise=0.05), 10 (Noise=0.5)
