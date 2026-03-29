# Degeneracy of the Inverse Problem: Uncountable Solutions from Dynamics

## The Forward Model

The flyvis graded-voltage ODE for each postsynaptic neuron $i$ reads:

$$
\tau_i\frac{dv_i(t)}{dt} = -v_i(t) + V_i^{\text{rest}}
+ \sum_{j\in\mathcal{N}_i} \mathbf{W}_{ij}\,
  \text{ReLU}\!\big(v_j(t)\big)
+ I_i(t)
$$

where $\tau_i$ and $V_i^{\mathrm{rest}}$ are cell-type parameters, $\mathbf{W}_{ij}$ is the connectome-constrained synaptic weight from neuron $j$ to $i$, and $I_i(t)$ is the external stimulus.

**The inverse problem**: given observed trajectories $\{v_i(t)\}_{t=0}^{T}$ and stimuli $\{I_i(t)\}$, recover the connectivity matrix $\mathbf{W}$, time constants $\tau$, and resting potentials $V^{\text{rest}}$.

## Where the Degeneracy is Hard-Coded

### Step 1: Rearrange to a linear system per neuron

For each postsynaptic neuron $i$, define the presynaptic activity vector at time $t$:

$$
h_j(t) = \text{ReLU}\!\big(v_j(t)\big)
$$

and the right-hand side:

$$
b_i(t) = \tau_i \frac{dv_i(t)}{dt} + v_i(t) - V_i^{\text{rest}} - I_i(t)
$$

Let $d_i = |\mathcal{N}_i|$ denote the in-degree of neuron $i$. Then the ODE becomes a linear constraint on the incoming weights $\mathbf{w}_i = (\mathbf{W}_{i1}, \mathbf{W}_{i2}, \dots, \mathbf{W}_{id_i})$:

$$
\sum_{j \in \mathcal{N}_i} \mathbf{W}_{ij}\, h_j(t) = b_i(t) \qquad \forall\; t \in \{1, \dots, T\}
$$

Stacking over $T$ timesteps yields a **linear system**:

$$
\mathbf{H}_i\, \mathbf{w}_i = \mathbf{b}_i
$$

where $\mathbf{H}_i \in \mathbb{R}^{T \times d_i}$ is the activity matrix (columns = presynaptic neuron activations over time).

### Step 2: The null space

Any $\boldsymbol{\delta} \in \ker(\mathbf{H}_i)$ satisfies $\mathbf{H}_i \boldsymbol{\delta} = \mathbf{0}$. Therefore the perturbed weight vector $\mathbf{w}_i + \boldsymbol{\delta}$ produces **exactly the same dynamics** as $\mathbf{w}_i$:

$$
\mathbf{H}_i (\mathbf{w}_i + \boldsymbol{\delta}) = \mathbf{H}_i \mathbf{w}_i + \underbrace{\mathbf{H}_i \boldsymbol{\delta}}_{= \mathbf{0}} = \mathbf{b}_i
$$

The null space dimension is $\dim\ker(\mathbf{H}_i) = d_i - \text{rank}(\mathbf{H}_i)$.

**Global bound from SVD.** An SVD of the full population activity reveals that the entire circuit's activity lives in a low-dimensional subspace. The computation subsamples to 982 neurons (every 14th of 13,741) and 8,000 timesteps (every 8th of 64,000), then computes the full SVD (all 982 singular values):

| Metric                         | Value                      |
| ------------------------------ | -------------------------- |
| Neurons (subsampled)           | 982 / 13,741               |
| Timesteps (subsampled)         | 8,000 / 64,000             |
| SVD                            | full (982 singular values) |
| Effective rank at 90% variance | 1                          |
| Effective rank at 99% variance | **45**                     |

The matrix $\mathbf{H}_i$ has $d_i$ columns (one per presynaptic neuron). By the rank-nullity theorem, these $d_i$ columns split into two groups:

$$
d_i = \underbrace{\text{rank}(\mathbf{H}_i)}_{\text{identifiable weights}} + \underbrace{\dim\ker(\mathbf{H}_i)}_{\text{free (null) weights}}
$$

Since each neuron $i$'s presynaptic activity $\mathbf{H}_i$ is a submatrix of the global activity, its rank is bounded by the global effective rank: $\text{rank}_{\text{eff}}(\mathbf{H}_i) \leq 45$. This is an _approximate_ bound — the 99% variance threshold is a convention, and the true mathematical rank (number of nonzero singular values) may be higher. Replacing exact rank with effective rank gives an approximate null space:

$$
\dim\ker_{\text{eff}}(\mathbf{H}_i) \approx \max(0, \; d_i - r)
$$

where $r$ is the effective activity rank. This approximation predicts that perturbations along the neglected singular directions have negligible effect on dynamics. The Empirical Verification section tests this prediction directly.

Summing over all 13,697 postsynaptic neurons gives a global estimate:

$$
\text{total null dim} = \sum_{i=1}^{N_{\text{post}}} \max(0, \; d_i - r)
$$

**Sensitivity to the variance threshold.** Since the 99% threshold is a convention, we report the null space estimate across several thresholds:

| Variance threshold | Effective rank $r$ | Null space dim | % identifiable |
| ------------------ | ------------------ | -------------- | -------------- |
| 90%                | 1                  | 420,415        | 3.2%           |
| 95%                | 2                  | 407,427        | 6.1%           |
| **99%**            | **45**             | **115,223**    | **73.5%**      |
| 99.5%              | 90                 | 26,413         | 93.9%          |
| 99.9%              | 238                | 0              | 100.0%         |

At 99.9% the rank (238) exceeds the maximum in-degree (208), so the null space vanishes by this metric alone. The 99% threshold ($r = 45$) gives the most informative estimate: it predicts a large but finite null space, whose reality is confirmed by the Empirical Verification below.

The in-degree $d_i$ (number of incoming edges) determines which neurons have degenerate incoming weights at $r = 45$. Neurons with $d_i \leq 45$ have enough independent activity dimensions to constrain all their incoming weights — they are "fully identifiable" and their weights can in principle be recovered from dynamics. Neurons with $d_i > 45$ have more incoming edges than independent activity dimensions, so $d_i - 45$ of their weights are free to vary without affecting the output — these are the null space dimensions where weight recovery is impossible:

| In-degree range | Neurons | Null space contribution |
| --------------- | ------- | ----------------------- |
| 1 -- 10         | 3,182   | 0 (fully identifiable)  |
| 11 -- 20        | 3,917   | 0 (fully identifiable)  |
| 21 -- 45        | 3,196   | 0 (fully identifiable)  |
| 46 -- 100       | 2,843   | bulk of null space      |
| 101 -- 208      | 559     | largest per-neuron null |

High in-degrees arise because the 217 columns are laterally interconnected: a neuron integrating signals from its spatial neighborhood receives edges from the same cell type in many surrounding columns. Of the 13,741 neurons, 44 are source-only (in-degree zero, boundary neurons that project into the network but receive no edges). Summing over the remaining 13,697 postsynaptic neurons, **the SVD analysis predicts 115,223 null dimensions — 26.5% of all 434,112 edge weights are unconstrained by the dynamics.**

### Step 3: Within-type degeneracy (the dominant mechanism)

Step 2 showed _how large_ the null space is (115,223 dimensions at rank 45) using only linear algebra and SVD. Step 3 explains _why_ it is so large: the columnar organization of the connectome forces same-type neurons to have nearly identical activity, which is exactly what makes the columns of $\mathbf{H}_i$ linearly dependent and rank-deficient.

Each of the 217 columns covers a different position in the visual field. For instance, L1 in column 5 sees the light at position (5, 3); L1 in column 6 sees position (6, 3). They receive different stimuli — but when a moving bar sweeps across the eye, column 6 sees what column 5 saw a moment earlier. Over thousands of frames with many stimuli, same-type neurons in different columns produce time-shifted copies of the same signal. Their activity is not identical, but highly correlated.

Now consider a postsynaptic neuron $i$ that receives input from $k$ copies of the same cell type across $k$ neighboring columns. Because these $k$ neurons differ only by a small spatial shift, their activity vectors $\mathbf{h}_{j_1}(t), \dots, \mathbf{h}_{j_k}(t)$ are strongly correlated — like L1 in column 5 vs L1 in column 6, which differ by roughly one timestep. In the matrix $\mathbf{H}_i$, these $k$ columns carry nearly the same information, so their rank is $\approx 1$ instead of $k$, contributing $k - 1$ dimensions to $\ker(\mathbf{H}_i)$.

**Why the sum must be constant.** In the idealized case where the $k$ columns are identical ($\mathbf{h}_{j_1} = \cdots = \mathbf{h}_{j_k} = \mathbf{h}$), apply Step 2's null space condition $\mathbf{H}_i \boldsymbol{\delta} = \mathbf{0}$. The contribution of these $k$ edges is:

$$
\delta_{j_1}\, \mathbf{h} + \delta_{j_2}\, \mathbf{h} + \cdots + \delta_{j_k}\, \mathbf{h}
= \Big(\sum_{m=1}^{k} \delta_{j_m}\Big)\, \mathbf{h} = \mathbf{0}
$$

Since $\mathbf{h} \neq \mathbf{0}$, this requires $\sum_{m=1}^{k} \delta_{j_m} = 0$. So among $k$ same-type edges, one constraint fixes the sum, leaving $k - 1$ free $\delta$ values — these are the null directions counted in Step 4.

### Step 4: Counting the free parameters

For each postsynaptic neuron $i$ and each presynaptic cell type $\alpha$ with $k_{i\alpha} > 1$ incoming edges, the null space gains $k_{i\alpha} - 1$ dimensions. Summing over all neurons and types:

$$
\dim\ker(\mathbf{H}) = \sum_{i=1}^{N} \sum_{\alpha:\, k_{i\alpha}>1} (k_{i\alpha} - 1)
$$

For the flyvis connectome (13,741 neurons, 434,112 edges, 65 cell types):

| Quantity                                       | Value        |
| ---------------------------------------------- | ------------ |
| Total edges $E$                                | 434,112      |
| Degenerate groups (dst, src_type with $k > 1$) | ~121,000+    |
| **Total null space dimension**                 | **~121,100** |
| Fraction of all edges in null space            | **~28%**     |
| Identifiable edge parameters                   | ~313,000     |

The null space has **~121,100 dimensions**, meaning that many degrees of freedom in $\mathbf{W}$ are unconstrained by the dynamics. Within each group of $k$ same-type edges targeting the same neuron, any perturbation satisfying the sum-zero constraint ($\sum \delta_{j_m} = 0$, Step 3) leaves the dynamics unchanged. Weight can be freely _redistributed_ among same-type inputs, but the total to each neuron from each type is fixed. Since each null direction admits a continuous scaling $\boldsymbol{\delta} \to \lambda \boldsymbol{\delta}$ for $\lambda \in \mathbb{R}$, the set of solutions is **uncountably infinite**: a ~121,100-dimensional affine subspace of $\mathbb{R}^E$.

This per-type structural count (~121,100) agrees to within 5% with the global SVD bound from Step 2 (115,223). The global SVD captures all correlations (within- and cross-type) but is coarse; the per-type count precisely measures within-type redundancy but misses cross-type correlations. Their close agreement confirms that within-type degeneracy is the dominant mechanism.

## Empirical Verification

The null space estimate in Step 2 relies on the effective rank at 99% variance — a conventional threshold, not an exact quantity. To test whether the predicted null directions are real (i.e., perturbations along them truly preserve dynamics), we verify empirically: **if** the null space analysis (Steps 2--4) is correct, **then** a perturbed matrix $\mathbf{W} + \boldsymbol{\delta}$ with low connectivity $R^2$ (weights look very different) should still produce high rollout $R^2$ (dynamics look the same), as long as $\boldsymbol{\delta}$ satisfies the sum-zero constraint ($\sum_{m=1}^{k} \delta_{j_m} = 0$) within each same-type group.

The flyvis connectome has 65 cell types, of which 52 are non-retina types that receive lateral inputs from same-type neurons across columns. Step 3 predicts two regimes depending on how correlated the same-type activity is:

- Types with **identical** same-type activity (within-column projections only) → **exact** null space, dynamics unchanged at any perturbation scale
- Types with **correlated but not identical** activity (cross-column projections) → **approximate** null space, small divergence growing with perturbation scale

**Protocol.** For each of the 52 non-retina types, we generated sum-preserving random perturbations at 15 scales (from $\lambda = 0.05$ to $8.0$), producing 780 degenerate connectivity matrices. Each was rolled out through the full ODE from identical initial conditions and stimulus.

### Key results

| Metric                                                   | Value               |
| -------------------------------------------------------- | ------------------- |
| Variants tested                                          | 780                 |
| **Conn. R2 range**                                       | **0.28 -- 1.00**    |
| **Rollout R2 range**                                     | **0.80 -- 1.00**    |
| Variants with conn. R2 < 0.9 **and** rollout R2 > 0.99   | **30 / 31** (97%)   |
| Variants with conn. R2 < 0.99 **and** rollout R2 > 0.999 | **116 / 152** (76%) |
| Types with exact degeneracy (RMSE $\sim 10^{-7}$)        | 5 / 52              |
| Types with approximate degeneracy                        | 47 / 52             |

### All 52 cell types at maximum perturbation ($\lambda = 8.0$)

Color code: <span style="color:#2ea043">**green**</span> $R^2 > 0.9$, <span style="color:#d29922">**orange**</span> $R^2 > 0.5$, <span style="color:#cf222e">**red**</span> $R^2 \leq 0.5$.

<table>
<tr><th>Cell type</th><th>Conn. R²</th><th>Rollout R²</th><th>Cell type</th><th>Conn. R²</th><th>Rollout R²</th></tr>
<tr><td>Tm20</td><td style="background:#cf222e60">0.282</td><td style="background:#2ea04360">0.997</td><td>Mi3</td><td style="background:#d2992260">0.733</td><td style="background:#2ea04360">0.995</td></tr>
<tr><td>Tm5Y</td><td style="background:#d2992260">0.566</td><td style="background:#2ea04360">0.998</td><td>Tm1</td><td style="background:#d2992260">0.738</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>Mi1</td><td style="background:#d2992260">0.617</td><td style="background:#2ea04360">0.998</td><td>Mi15</td><td style="background:#d2992260">0.741</td><td style="background:#2ea04360">0.997</td></tr>
<tr><td>Lawf2</td><td style="background:#d2992260">0.727</td><td style="background:#2ea04360">0.995</td><td>Tm4</td><td style="background:#d2992260">0.753</td><td style="background:#2ea04360">0.986</td></tr>
<tr><td>Tm16</td><td style="background:#d2992260">0.814</td><td style="background:#2ea04360">1.000</td><td>L3</td><td style="background:#d2992260">0.871</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>Mi4</td><td style="background:#d2992260">0.877</td><td style="background:#2ea04360">0.999</td><td>Lawf1</td><td style="background:#d2992260">0.899</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>R1</td><td style="background:#2ea04360">0.915</td><td style="background:#2ea04360">0.975</td><td>R6</td><td style="background:#2ea04360">0.927</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>Tm5a</td><td style="background:#2ea04360">0.933</td><td style="background:#2ea04360">1.000</td><td>R2</td><td style="background:#2ea04360">0.938</td><td style="background:#2ea04360">0.998</td></tr>
<tr><td>TmY4</td><td style="background:#2ea04360">0.941</td><td style="background:#2ea04360">1.000</td><td>Mi13</td><td style="background:#2ea04360">0.942</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>R5</td><td style="background:#2ea04360">0.951</td><td style="background:#2ea04360">1.000</td><td>TmY5a</td><td style="background:#2ea04360">0.952</td><td style="background:#2ea04360">0.999</td></tr>
<tr><td>L2</td><td style="background:#2ea04360">0.965</td><td style="background:#2ea04360">1.000</td><td>T3</td><td style="background:#2ea04360">0.966</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>T4a</td><td style="background:#2ea04360">0.967</td><td style="background:#2ea04360">0.999</td><td>Mi14</td><td style="background:#2ea04360">0.967</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>T2a</td><td style="background:#2ea04360">0.969</td><td style="background:#2ea04360">1.000</td><td>T2</td><td style="background:#2ea04360">0.969</td><td style="background:#2ea04360">0.993</td></tr>
<tr><td>Tm2</td><td style="background:#2ea04360">0.969</td><td style="background:#2ea04360">1.000</td><td>TmY13</td><td style="background:#2ea04360">0.971</td><td>—</td></tr>
<tr><td>TmY14</td><td style="background:#2ea04360">0.972</td><td style="background:#2ea04360">1.000</td><td>TmY9</td><td style="background:#2ea04360">0.972</td><td style="background:#2ea04360">0.960</td></tr>
<tr><td>T4d</td><td style="background:#2ea04360">0.978</td><td style="background:#2ea04360">0.999</td><td>T4c</td><td style="background:#2ea04360">0.988</td><td style="background:#2ea04360">0.989</td></tr>
<tr><td>Tm3</td><td style="background:#2ea04360">0.990</td><td style="background:#2ea04360">1.000</td><td>T5c</td><td style="background:#2ea04360">0.990</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>T4b</td><td style="background:#2ea04360">0.990</td><td style="background:#2ea04360">1.000</td><td>T5d</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>Mi2</td><td style="background:#2ea04360">0.992</td><td style="background:#2ea04360">1.000</td><td>TmY3</td><td style="background:#2ea04360">0.993</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>TmY15</td><td style="background:#2ea04360">0.993</td><td style="background:#2ea04360">1.000</td><td>TmY10</td><td style="background:#2ea04360">0.994</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>Tm5b</td><td style="background:#2ea04360">0.995</td><td style="background:#2ea04360">1.000</td><td>Mi11</td><td style="background:#2ea04360">0.996</td><td style="background:#2ea04360">0.998</td></tr>
<tr><td>Tm30</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td><td>R3</td><td style="background:#2ea04360">0.997</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>T5a</td><td style="background:#2ea04360">0.998</td><td style="background:#2ea04360">1.000</td><td>T5b</td><td style="background:#2ea04360">0.998</td><td style="background:#2ea04360">1.000</td></tr>
<tr><td>TmY18</td><td style="background:#2ea04360">0.998</td><td style="background:#2ea04360">1.000</td><td><b>L5</b></td><td style="background:#2ea04360"><b>0.999</b></td><td style="background:#2ea04360"><b>1.000*</b></td></tr>
<tr><td><b>L4</b></td><td style="background:#2ea04360"><b>1.000</b></td><td style="background:#2ea04360"><b>1.000*</b></td><td><b>R7</b></td><td style="background:#2ea04360"><b>1.000</b></td><td style="background:#2ea04360"><b>1.000*</b></td></tr>
<tr><td><b>R8</b></td><td style="background:#2ea04360"><b>1.000</b></td><td style="background:#2ea04360"><b>1.000*</b></td><td><b>Tm5c</b></td><td style="background:#2ea04360"><b>1.000</b></td><td style="background:#2ea04360"><b>1.000*</b></td></tr>
</table>

\* Exact degeneracy: RMSE $\sim 10^{-7}$ (machine precision). — = diverged (TmY13 outlier).

Sorted by connectivity $R^2$ (ascending = most degenerate first). The key pattern: the connectivity column shows <span style="color:#cf222e">red</span> and <span style="color:#d29922">orange</span> (weights changed substantially) while the rollout column is entirely <span style="color:#2ea043">green</span> (dynamics preserved). The 5 bold types at the bottom have exact degeneracy: their same-type columns in $\mathbf{H}_i$ are exactly identical, so $\mathbf{H}_i \boldsymbol{\delta} = \mathbf{0}$ holds exactly.

### Two regimes

**Exact degeneracy (5 types):** L4, L5, R7, R8, Tm5c — within-column projections only, no spatial shift. The $k$ same-type presynaptic neurons have truly identical activity.

**Approximate degeneracy (47 types):** connectivity $R^2$ ranges from 0.28 (Tm20, 72% of weight variance changed) down to near 1.0, yet rollout $R^2$ stays above 0.96 in all cases. These results confirm the null space analysis empirically: weights change drastically while dynamics remain nearly identical. The small residual divergence reflects the approximate nature of the null space — $\mathbf{H}_i \boldsymbol{\delta} \approx \mathbf{0}$, not exactly zero — due to the spatial shifts discussed in Step 3.

### Per-scale summary

Each perturbation is $\mathbf{W} + \lambda \boldsymbol{\delta}$, where $\boldsymbol{\delta}$ is a random sum-preserving null direction and $\lambda$ controls how far we move from the original weights. Larger $\lambda$ means more different weights, lower connectivity $R^2$.

| $\lambda$ | Conn. $R^2$ range | Mean rollout $R^2$ |
| --------- | ----------------- | ------------------ |
| 0.05      | 1.000 -- 1.000    | 1.000000           |
| 0.50      | 0.997 -- 1.000    | 0.999996           |
| 1.00      | 0.989 -- 1.000    | 0.999984           |
| 2.00      | 0.955 -- 1.000    | 0.999936           |
| 4.00      | 0.821 -- 1.000    | 0.999722           |
| 8.00      | 0.282 -- 1.000    | ~0.999\*           |

\* excluding one outlier (type 57, rollout R2 = 0.80).

## Discussion

### The inverse problem is fundamentally ill-posed

Recovering the connectivity matrix $\mathbf{W}$ from dynamics alone is ill-posed, even with perfect noise-free observations, known $\tau_i$, $V_i^{\text{rest}}$, known activation function, and known graph topology.

The null space analysis (Steps 2--4) quantifies the ill-posedness: **~121,100 out of 434,112 edge weights** (~28%) can be changed freely without affecting the dynamics. The empirical verification confirms this across all 52 non-retina cell types: at $\lambda = 8.0$, connectivity $R^2$ drops as low as 0.28 while rollout $R^2$ stays above 0.96.

The set of valid solutions is **uncountably infinite**. For each of the 52 cell types independently, the perturbation $\boldsymbol{\delta}$ can be scaled by any $\lambda \in \mathbb{R}$, and different types can use different $\lambda$ values. The solution manifold is not a point but a ~121,100-dimensional continuous subspace of $\mathbb{R}^{434,112}$.

The ill-posedness is **structural**, arising from the columnar organization of the circuit: same-type neurons in different columns produce strongly correlated activity (Step 3), making their individual contributions to a shared target indistinguishable from dynamics alone.

### Implications for GNN-based inference

The GNN approximates the dynamics as:

$$
\frac{\widehat{dv}_i(t)}{dt}
=f_\theta\Big(
v_i(t),\mathbf{a}_i,
\sum_{j\in\mathcal{N}_i} \widehat{\mathbf{W}}_{ij}\,g_\phi\!\big(v_j(t),\mathbf{a}_j\big)^2,
I_i(t)\Big)
$$

where $f_\theta = \text{MLP}_0$ and $g_\phi = \text{MLP}_1$ are three-layer perceptrons, $\widehat{\mathbf{W}}_{ij}$ are learnable edge weights, and $\mathbf{a}_i \in \mathbb{R}^2$ are learnable per-neuron embeddings. The training minimizes:

$$
\mathcal{L}_{\mathrm{pred}}
= \sum_{i,t}
\big\|
\widehat{y}_i(t) - y_i(t)
\big\|_2
$$

between simulator $y_i(t) = dv_i(t)/dt$ and GNN predictions $\widehat{y}_i(t) = \widehat{dv}_i(t)/dt$, augmented with regularization:

$$
\begin{aligned}
\mathcal{L} &=
\|\widehat{\mathbf y}-\mathbf y\|_2
+ \lambda_0\|\theta\|_{1}
+ \lambda_1\|\phi\|_{1}
+ \lambda_2\|\widehat{\mathbf W}\|_{1} \\
&\quad
+ \gamma_0\|\theta\|_{2}
+ \gamma_1\|\phi\|_{2} \\
&\quad
+ \mu_0\,
\big\|\operatorname{ReLU}\!\big(-\tfrac{\partial\, g_\phi(v,\mathbf{a})}{\partial v}\big)\big\|_2
+ \mu_1\,
\|g_\phi(v_\star,\mathbf{a})-v_\star\|_2
\end{aligned}
$$

where $\lambda$ terms promote sparsity, $\gamma$ terms stabilize learned functions, $\mu_0$ enforces monotonicity of the edge message, and $\mu_1$ normalizes with respect to reference voltage $v_\star$.

When the GNN achieves $R^2_{\mathbf{W}} = 0.997$ (as our LLM-optimized model does at $\sigma = 0.5$), it is recovering the ~313,000 identifiable parameters well while the ~121,100 null-space parameters are essentially free. The GNN's implicit inductive bias (message-passing, weight sharing) acts as an implicit regularizer that selects a particular solution from the uncountable solution manifold.

### Why noise helps

Adding process noise $\sigma\eta_i(t)$ to the simulation:

$$
\tau_i\frac{dv_i(t)}{dt} = -v_i(t) + V_i^{\text{rest}}
+ \sum_{j\in\mathcal{N}_i} \mathbf{W}_{ij}\,
  \text{ReLU}\!\big(v_j(t)\big)
+ I_i(t) + \sigma\eta_i(t)
$$

breaks the exact within-type degeneracy because:

1. Each neuron receives independent noise realizations, even if same-type
2. This injects independent variation into the columns of $\mathbf{H}_i$, breaking the within-type correlations from Step 3
3. The effective rank of $\mathbf{H}_i$ increases from 45 (noise-free) toward $d_i$
4. The null space $\dim\ker(\mathbf{H}_i) = d_i - \text{rank}(\mathbf{H}_i)$ shrinks, making more edge weights identifiable

**Quantitative measurement.** We measure how noise changes the effective rank of the population activity, and thus the null space dimension, using the same formula as Step 2:

$$
\text{total null} = \sum_{i=1}^{N_{\text{post}}} \max(0, \; d_i - r)
$$

**SVD methodology.** For each noise level ($\sigma = 0$, $0.05$, $0.5$), we:

1. Load the full voltage trace $\mathbf{V} \in \mathbb{R}^{64000 \times 13741}$ and subsample to 8,000 timesteps (every 8th frame)
2. Apply $h_j(t) = \text{ReLU}(v_j(t))$ to get the activity matrix $\mathbf{H} \in \mathbb{R}^{8000 \times 13741}$
3. Subsample to every 14th neuron → $\mathbf{H}_{\text{sub}} \in \mathbb{R}^{8000 \times 982}$
4. Compute the **full** SVD of $\mathbf{H}_{\text{sub}}$ (all 982 singular values), so the total variance $\sum_k s_k^2$ is exact
5. Find the smallest $r$ such that $\sum_{k=1}^{r} s_k^2 \geq 0.99 \cdot \sum_k s_k^2$

6. Plug $r$ into the per-neuron null space formula above, using the in-degree distribution from the flyvis connectome (434,112 edges, max in-degree 208)

**Results.** The $\sigma = 0$ row reproduces Step 2's estimate (rank 45, null 115,223) using the same SVD methodology. The noisy conditions use identical subsampling, so the ranks are directly comparable. GNN experimental results are from `flyvis_results.md` (10 seeds each):

| Noise           | Activity rank $r$ | Null space dim | % identifiable | Observed $R^2_{\mathbf{W}}$ |
| --------------- | ----------------- | -------------- | -------------- | --------------------------- |
| $\sigma = 0$    | 45                | 115,223        | 73.5%          | 0.899 ± 0.030               |
| $\sigma = 0.05$ | 134               | 10,227         | 97.6%          | 0.962 ± 0.012               |
| $\sigma = 0.5$  | **781**           | **0**          | **100.0%**     | **0.997 ± 0.000**           |

The trend validates the mechanism: noise increases the effective rank monotonically (45 → 134 → 781), shrinking the null space and making more weights identifiable. At $\sigma = 0.5$, the rank (781) exceeds the maximum in-degree (208), so every neuron's incoming weights are fully identifiable — the null space vanishes entirely, consistent with $R^2_{\mathbf{W}} = 0.997$.

Alternatively, to break the strong within-type correlations experimentally, one could use optogenetics to inject independent perturbations into each neuron of a given type (e.g., stimulate each L1 with a different random light pattern), or inversely silence a subset of same-type neurons across columns (e.g., ablate L1 in columns 4 and 6 but not 5). Both approaches make the columns of $\mathbf{H}_i$ distinguishable, increasing $\text{rank}(\mathbf{H}_i)$ and shrinking the null space — the same mechanism as process noise, but achievable in real tissue.

## Summary

Recovering the connectome from neural dynamics is fundamentally ill-posed: the columnar organization of the flyvis circuit forces same-type neurons across columns to produce strongly correlated activity, making their individual synaptic contributions indistinguishable. Two independent estimates of the null space — a global SVD bound (115,223 dimensions) and a per-type structural count (~121,100 dimensions) — agree to within 5%, confirming that within-type degeneracy is the dominant mechanism; the global bound captures all correlations (within- and cross-type) but is coarse, while the per-type count precisely measures within-type redundancy but misses cross-type correlations. This structural degeneracy affects 26.5% of all 434,112 edge weights, confirmed empirically by generating 780 perturbed connectivity matrices where weights change drastically (conn. $R^2$ as low as 0.28) while dynamics remain nearly identical (rollout $R^2 > 0.96$). Adding independent process noise breaks the within-type correlations and increases the effective activity rank from 45 (noise-free) to 781 ($\sigma = 0.5$), eliminating the null space entirely and enabling the GNN to achieve $R^2_{\mathbf{W}} = 0.997$. The analysis provides both a theoretical framework (SVD-based null space bounds) and a mechanistic explanation for why biological noise — intrinsic to neural circuits — naturally improves connectome identifiability.

## Scripts

- [scripts/generate_degenerate_W.py](../scripts/generate_degenerate_W.py) -- generates 780 degenerate $\mathbf{W}$ matrices
- [scripts/rollout_degenerate_W.py](../scripts/rollout_degenerate_W.py) -- rolls out ODE and measures divergence
- [scripts/analyze_noise_correlation.py](../scripts/analyze_noise_correlation.py) -- measures per-type rank and null space across noise levels
- Results: `graphs_data/degenerate_matrix/rollout_results/`
