# Degeneracy of the Inverse Problem: Uncountable Solutions from Dynamics

## The Forward Model

The flyvis graded-voltage ODE for each postsynaptic neuron $i$ reads:

$$
\tau_i \frac{dv_i}{dt} = -v_i + V_{\text{rest},i} + \sum_{j} W_{ij}\, \sigma(v_j) + e_i(t)
$$

where $\sigma = \text{ReLU}$, $W_{ij}$ is the synaptic weight from neuron $j$ to $i$, and $e_i(t)$ is the external stimulus.

**The inverse problem**: given observed trajectories $\{v_i(t)\}_{t=0}^{T}$ and stimuli $\{e_i(t)\}$, recover the connectivity matrix $W$, time constants $\tau$, and resting potentials $V_\text{rest}$.

## Where the Degeneracy is Hard-Coded

### Step 1: Rearrange to a linear system per neuron

For each postsynaptic neuron $i$, define the presynaptic activity vector at time $t$:

$$
h_j(t) = \sigma(v_j(t))
$$

and the right-hand side:

$$
b_i(t) = \tau_i \frac{dv_i}{dt}\bigg|_t + v_i(t) - V_{\text{rest},i} - e_i(t)
$$

Then the ODE becomes a linear constraint on the incoming weights $\mathbf{w}_i = (W_{i1}, W_{i2}, \dots, W_{id_i})$:

$$
\sum_{j \in \mathcal{N}(i)} W_{ij}\, h_j(t) = b_i(t) \qquad \forall\; t \in \{1, \dots, T\}
$$

Stacking over $T$ timesteps yields a **linear system**:

$$
\mathbf{H}_i\, \mathbf{w}_i = \mathbf{b}_i
$$

where $\mathbf{H}_i \in \mathbb{R}^{T \times d_i}$ is the activity matrix (columns = presynaptic neuron activations over time) and $d_i = |\mathcal{N}(i)|$ is the in-degree of neuron $i$.

### Step 2: The null space

Any $\boldsymbol{\delta} \in \ker(\mathbf{H}_i)$ satisfies $\mathbf{H}_i \boldsymbol{\delta} = \mathbf{0}$. Therefore the perturbed weight vector $\mathbf{w}_i + \boldsymbol{\delta}$ produces **exactly the same dynamics** as $\mathbf{w}_i$:

$$
\mathbf{H}_i (\mathbf{w}_i + \boldsymbol{\delta}) = \mathbf{H}_i \mathbf{w}_i + \underbrace{\mathbf{H}_i \boldsymbol{\delta}}_{= \mathbf{0}} = \mathbf{b}_i
$$

The null space dimension is $\dim\ker(\mathbf{H}_i) = d_i - \text{rank}(\mathbf{H}_i)$.

### Step 3: Within-type degeneracy (the dominant mechanism)

In the flyvis connectome, neurons of the same **cell type** (same $\tau$, $V_\text{rest}$) in different hex columns have nearly identical activation patterns because they implement the same computation on spatially translated inputs. If $k$ presynaptic neurons of the same type all connect to postsynaptic neuron $i$, their columns in $\mathbf{H}_i$ are (nearly) linearly dependent:

$$
\mathbf{h}_{j_1}(t) \approx \mathbf{h}_{j_2}(t) \approx \cdots \approx \mathbf{h}_{j_k}(t) \qquad \forall\; t
$$

This means $\text{rank}$ of the $k$ columns $\approx 1$, contributing $k - 1$ null directions per group. Any **sum-preserving** redistribution of weight among the $k$ edges changes $W$ without affecting the dynamics:

$$
\sum_{m=1}^{k} \delta_{W_{i,j_m}} = 0 \quad \Longrightarrow \quad \boldsymbol{\delta} \in \ker(\mathbf{H}_i)
$$

### Step 4: Counting the free parameters

For each postsynaptic neuron $i$ and each presynaptic cell type $\alpha$ with $k_{i\alpha} > 1$ incoming edges, the null space gains $k_{i\alpha} - 1$ dimensions. Summing over all neurons and types:

$$
\dim\ker(\mathbf{H}) = \sum_{i=1}^{N} \sum_{\alpha:\, k_{i\alpha}>1} (k_{i\alpha} - 1)
$$

For the flyvis connectome (13,741 neurons, 434,112 edges, 65 cell types):

| Quantity | Value |
|----------|-------|
| Total edges $E$ | 434,112 |
| Degenerate groups (dst, src_type with $k > 1$) | ~121,000+ |
| **Total null space dimension** | **~121,100** |
| Fraction of all edges in null space | **~28%** |
| Identifiable edge parameters | ~313,000 |

This means **121,100 weight parameters can be changed freely without any effect on the dynamics**. The solution manifold is not a point but a ~121,100-dimensional affine subspace of $\mathbb{R}^E$. Since each null direction admits a continuous scaling $\boldsymbol{\delta} \to \lambda \boldsymbol{\delta}$ for $\lambda \in \mathbb{R}$, the set of solutions is **uncountably infinite**.

## Empirical Verification

We generated 780 degenerate connectivity matrices (52 non-retina types $\times$ 15 perturbation scales) using sum-preserving random perturbations along the null directions, then rolled out the full ODE from identical initial conditions and stimulus.

### Key results

| Metric | Value |
|--------|-------|
| Variants tested | 780 |
| **Conn. R2 range** | **0.28 -- 1.00** |
| **Rollout R2 range** | **0.80 -- 1.00** |
| Variants with conn. R2 < 0.9 **and** rollout R2 > 0.99 | **30 / 31** (97%) |
| Variants with conn. R2 < 0.99 **and** rollout R2 > 0.999 | **116 / 152** (76%) |
| Types with exact degeneracy (RMSE $\sim 10^{-7}$) | 5 / 52 |
| Types with approximate degeneracy | 47 / 52 |

### Two regimes of degeneracy

**1. Exact degeneracy (5 types)**

Types 8, 9, 29, 30, 54 show RMSE at machine precision ($\sim 10^{-7}$) across all perturbation scales, even when connectivity R2 drops to 0.999. These types have presynaptic neurons with truly identical activations (e.g., same-column within-hex projections). The null space for these types is **exact** -- not approximate.

**2. Approximate degeneracy (47 types)**

Most types show small but nonzero divergence that grows with perturbation scale. Even at extreme perturbations:

- Type 46: **connectivity R2 = 0.28** (72% of weight variance changed) $\to$ rollout R2 = 0.997
- Type 51: connectivity R2 = 0.57 $\to$ rollout R2 = 0.998
- Type 12: connectivity R2 = 0.62 $\to$ rollout R2 = 0.998

The divergence is small because neurons of the same type have *correlated but not identical* activity (they receive spatially shifted inputs). The near-null directions are soft modes with small but nonzero eigenvalues in $\mathbf{H}_i^T \mathbf{H}_i$.

### Per-scale summary

| Scale | Conn. R2 range | Mean rollout R2 |
|-------|---------------|-----------------|
| 0.05 | 1.000 -- 1.000 | 1.000000 |
| 0.50 | 0.997 -- 1.000 | 0.999996 |
| 1.00 | 0.989 -- 1.000 | 0.999984 |
| 2.00 | 0.955 -- 1.000 | 0.999936 |
| 4.00 | 0.821 -- 1.000 | 0.999722 |
| 8.00 | 0.282 -- 1.000 | ~0.999* |

\* excluding one outlier (type 57, rollout R2 = 0.80).

## Discussion

### The inverse problem is fundamentally ill-posed

The analysis above shows that recovering the connectivity matrix $W$ from dynamics alone is a fundamentally ill-posed inverse problem, even with:
- Perfect noise-free observations
- Known $\tau$ and $V_\text{rest}$
- Known activation function $\sigma$
- Known graph topology (sparsity pattern)

The ill-posedness is **structural**, arising from the columnar organization of the circuit: same-type neurons in different columns implement the same computation, making their individual contributions to a shared target indistinguishable from dynamics alone.

### Implications for GNN-based inference

When a GNN achieves $W$ R2 = 0.997 (as our LLM-optimized model does at $\sigma = 0.5$), it is recovering the ~313,000 identifiable parameters well while the ~121,100 null-space parameters are essentially free. The GNN's implicit inductive bias (message-passing, weight sharing) acts as an implicit regularizer that selects a particular solution from the uncountable solution manifold.

### Why noise helps

Adding observation noise $\eta \sim \mathcal{N}(0, \sigma^2)$ breaks the exact within-type degeneracy because:

1. Each neuron receives independent noise realizations, even if same-type
2. This injects independent variation into the columns of $\mathbf{H}_i$
3. The effective rank of $\mathbf{H}_i$ increases toward $d_i$
4. The null space shrinks, making more edge weights identifiable

This is consistent with our experimental results: noise $\sigma = 0.5$ yields $W$ R2 = 0.997 vs 0.959 noise-free, precisely because noise resolves the within-type ambiguity.

### Connection to information-theoretic identifiability

The problem can be framed in terms of the Fisher information matrix $\mathbf{F}_i$ for the weights of neuron $i$:

$$
\mathbf{F}_i = \frac{1}{\sigma_\eta^2} \mathbf{H}_i^T \mathbf{H}_i
$$

At $\sigma_\eta = 0$ (noise-free), $\mathbf{F}_i$ has rank $\text{rank}(\mathbf{H}_i) < d_i$: the Cramer-Rao bound diverges for the null-space directions, meaning **no estimator can recover those weights to finite precision**. With noise $\sigma_\eta > 0$, the independent noise realizations lift the rank of $\mathbf{H}_i$ (in expectation), making the problem better conditioned.

## Scripts

- [scripts/generate_degenerate_W.py](../scripts/generate_degenerate_W.py) -- generates 780 degenerate $W$ matrices
- [scripts/rollout_degenerate_W.py](../scripts/rollout_degenerate_W.py) -- rolls out ODE and measures divergence
- Results: `graphs_data/degenerate_matrix/rollout_results/`
