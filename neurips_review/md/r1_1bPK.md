We thank the reviewer for the precise review: the summary is accurate, and Q1 and Q2 named experiments rather than concerns, so both could be run and both produced new results.

**Q1.** The three idealizations are coupled and never separated. Two mismatch experiments, all single-seed on Flyvis-217 at $\sigma = 0.05$ with consensus hyperparameters, median $\tau = 19.8$ ms.

**Model-class match.** A slow adaptation current leaves the class twice over, neither first order in the observables nor decomposable into pairwise messages: Eq. 1 gains a term $-g_a c_i$ with $\dot c_i = (v_i - c_i)/\tau_a$, $\tau_a = 200$ ms $\gg \tau$, and $c_i$ never observed. The GNN is unchanged.

| $g_a$ | GNN $R^2_{\widehat W}$ | GNN $R^2_{\hat V^{\mathrm{rest}}}$ | Known-ODE $R^2_{\widehat W}$ |
|---|---|---|---|
| $0$ | $0.987$ | $0.97\,(0.63)\,[5.0]$ | $0.986$ |
| $0.1$ | $0.974$ | $0.80\,(0.62)\,[5.7]$ | $0.972$ |
| $0.3$ | $0.909$ | $0.54\,(-0.35)\,[27.7]$ | $0.923$ |

*$R^2_{\hat V^{\mathrm{rest}}}$: inlier (full-sample) [excl. %]. $R^2_{\hat\tau} \geq 0.98$ in every row.*

Degradation is ordered and the oracle matches: $\widehat V^{\mathrm{rest}}$ absorbs the slow offset first, $\widehat W$ inherits its bias, $\tau$ (from the slope of $f_\theta$) survives. The class-narrowing priors are not load-bearing ($\mu_1 = 0$ leaves $R^2_{\widehat W}$ at $0.988$ against $0.987$), no sign convention is imposed ($\widehat W_{ij}$ carries the sign, $g_\phi^2$ the magnitude), and conductance-like coupling needs a different loss and is deferred. Applied jointly at $g_a = 0.3$, $h = 2$ ms the two costs are near-additive but deviate oppositely: the oracle loses $0.202$ where its parts sum to $0.174$, the GNN $0.161$ where they sum to $0.184$. Neither amplifies the other by $>0.03$.

**Temporal match.** The model always predicts at $20$ ms; the generator runs below that step, or at $2$ ms with the state observed every $20$ to $100$ ms.

| $\Delta t$ gen. | $\Delta t$ obs. | GNN $R^2_{\widehat W}$ | Known ODE $R^2_{\widehat W}$ |
|---|---|---|---|
| $20$ ms | $20$ ms | $0.970$ | $0.955$ |
| $10$ ms | $20$ ms | $0.864$ | $0.873$ |
| $4$ ms | $20$ ms | $0.822$ | $0.844$ |
| $2$ ms | $20$ ms | $0.812$ | $0.835$ |
| $2$ ms | $40$ ms | $0.720$ | $0.711$ |
| $2$ ms | $100$ ms | $0.315$ | $0.449$ |

At $20/20$ ms the generator *is* the model's Euler map; a finer step adds the truncation bias $(\Delta t/2)\,\ddot v$, neuron-specific rather than a global rescaling, costing $0.16$. Coarse observation is the real limitation: the GNN falls to $0.315$ at $\Delta t_{\mathrm{obs}} = 100$ ms ($5\tau$), reproducing the published $1/5$-frames row ($0.30$), with the oracle at $0.449$, while rollout $r$ stays above $0.97$, so prediction cannot detect it. Known-ODE degrades in step with the GNN, so the cost lies in the inverse problem, not the hypothesis class.

**What we'll change.** Keep "on par with an oracle" and drop "without assuming the form of the dynamical equation": the nonlinearities are learned, first-order dynamics and pairwise aggregation are assumed. Limitation 2 names this idealization but gives no tolerance, so we will add the one this sweep measures, $\Delta t_{\mathrm{obs}} \lesssim \tau$. All activity is synthetic, so we will retitle to "... from simulated neural activity".

**Q2.** Computed for all four tables (Tabs. 1, 2, Supp. 4, 7); the twins replace the submitted ones. Cells: inlier (full-sample) [excluded %].

| model | $\sigma$ | $R^2_{\hat\tau}$ in (full) [out. %] | $R^2_{\hat V^{\mathrm{rest}}}$ in (full) [out. %] |
|---|---|---|---|
| Known ODE | $0$ | $1.00\,(0.09)\,[1.9]$ | $0.97\,(0.86)\,[10.2]$ |
|  | $0.05$ | $1.00\,(1.00)\,[0.0]$ | $0.99\,(0.95)\,[4.7]$ |
|  | $0.5$ | $1.00\,(1.00)\,[0.0]$ | $1.00\,(1.00)\,[0.0]$ |
| GNN | $0$ | $0.94\,(-11.80)\,[3.1]$ | $0.76\,(-0.39)\,[13.7]$ |
|  | $0.05$ | $0.99\,(0.99)\,[0.0]$ | $0.93\,(0.76)\,[3.3]$ |
|  | $0.5$ | $1.00\,(1.00)\,[0.0]$ | $0.98\,(0.97)\,[0.0]$ |

Where exclusion is small the two agree; where it is large the inlier value is not informative. $\tau$ survives, full-sample $0.99$ against the oracle's $1.00$ at $\sigma = 0.05$, though noise-free it does not ($-11.80$ against $0.09$); $V^{\mathrm{rest}}$ does not, $0.76$ against $0.95$ and negative in every degraded row, down to $-149.37$ at $10\%$ hidden with $21.1$ to $56.3\%$ excluded. The Known-ODE runs were filtered at $\delta_V = 0.1$, not the stated $0.2$; full-sample is unaffected.

**What we'll change.** Report those rows as not recovered, keep "on par with an oracle" for $W$ and $\tau$ at $\sigma > 0$, drop it for $V^{\mathrm{rest}}$, and recompute the Known-ODE filter.

**Q3.** We restrict the claim to $\sigma > 0$ and present the noise-free row as the ill-posed regime.

At $\sigma = 0$, $R^2_{\widehat W}$ is $0.89$ (GNN) against $0.96$ (oracle) and full-sample $R^2_{\hat V^{\mathrm{rest}}}$ is $-0.39$ against $0.86$; at $\sigma = 0.05$, $R^2_{\widehat W}$ is $0.99$ for both and $R^2_{\hat V^{\mathrm{rest}}}$ is $0.76$ against $0.95$.

On the premise, $\sigma$ is intrinsic stochasticity, not an experimenter knob: release is probabilistic, channels noisy, potentials fluctuate. We have not calibrated its level, so the question is where real circuits sit on the $\sigma$ axis.

**Q4.** We have no such baseline and accept the reframing: no matched-compute random or Bayesian search was run, so the claim that it yields better configurations is unsupported, and it optimises $R^2_{\widehat W}$ against ground truth, unavailable in application. We will demote it to a methods note, keeping only the outcome: one configuration used everywhere (Supp. 6).

**Q5.** The **Allier et al. [2026]** preprint appeared February 2026, three months before the deadline: prior work, not concurrent.

It is a different system and a different formulation. The system is random assemblies with no anatomy, whose random weights carry no columnar degeneracy, hence $R^2_{\widehat W} = 1.00$ there against $0.89$ here at $\sigma = 0$. The formulation is additively separable in self-state and drive, where ours nests them, $\widehat{\dot v}_i = f_\theta(v_i, \mathbf{a}_i, m_i, I_i)$ with $m_i = \sum_j \widehat W_{ij}\, g_\phi(v_j,\mathbf{a}_j)^2$, not additive in $(v_i, m_i)$. The delta is the columnar identifiability analysis, the degradation battery, the FlyWire-scale result and the edge ablation, not the decomposition. No head-to-head was run.

**Q6.** **Shiu et al. [2024]** is a forward model, not an activity fit: weights from the connectome, sign from neurotransmitter prediction, one global scalar calibrated once. Matching $91\%$ of $164$ tested predictions, it refutes our claim that large-scale simulations have struggled to be predictive, and we will correct that sentence: forward simulation from a connectome is predictive at brain scale, the inverse direction is open.

**BrainTrace [2026]** is a learning system for spiking networks; its whole-brain *Drosophila* fit poses the same inverse problem as our Known-ODE control, form fixed (the Shiu LIF), connectome supplying the support of $\mathbf{W}$, weights optimised to match recordings. That is the best case in our study, where Appx. C already shows $\mathbf{W}$ underdetermined. Their setting is harder still: the loss is on *neuropil-level* rates, so each residual constrains a sum over thousands of neurons, and supervision at $1.2$ Hz is $42\tau$ against $T_{\mathrm{mbr}} = 20$ ms. Our sweep reaches only $5\tau$, where the GNN already falls to $0.315$. Without ground-truth $\mathbf{W}$, fit quality is the only signal, and Appx. C shows it cannot separate a recovered connectome from a dynamically equivalent one. Sect. 2 will separate the two.

**W4.** Agreed, and it will not stay in Limitations. $\gamma$ is the SD of Gaussian noise added after simulation, $v_i^{\mathrm{obs}} = v_i + \gamma\,\epsilon_i$: unlike $\sigma$ it corrupts the recording, not the trajectory. At $\gamma = 0.1$, $R^2_{\widehat W} = 0.63$ (GNN) and $0.69$ (oracle); at $\gamma = 0.2$, $0.38$ and $0.42$; $R^2_{\hat\tau}$ falls to $0.28$ and $0.08$. The oracle degrades as much, so the failure is in the inverse problem, not our architecture, though the method is unusable here. The contributions will say demonstrated under process noise, not realistic measurement noise.

**W7 and W8.** On "expected almost by construction": decorrelation is the plausible mechanism, but the paper asserted its size, we have now measured it (meta-review reply): the degenerate-edge fraction falls from $6.35\%$ to $0$, and the effect is not a by-product of GNN training. The oracle has no learned functions to regularise yet improves identically ($R^2_{\widehat W}: 0.96 \to 0.99 \to 1.00$). It does not establish that the improvement is *specific* to the columnar directions, which is how the paper stated it; we will claim the conditioning and report the selectivity as untested. Both abstract claims are still overstated. W7: optogenetics is an untested analogy, so the abstract will claim only that noise and perturbation help in simulation. W8: the SIREN rollout is scored on training frames and the stimulus is largely present in the photoreceptors, so "recovers unknown visual stimuli" comes out of the abstract, the experiment staying in the appendix as partial recovery.

**W9.** *$N$.* $13{,}741$ neurons, $13{,}697$ with at least one incoming edge, $44$ with none: Appx. C counts the former, so the discrepancy is neurons with no *input*. $\sum_i \max(0, d_i - 45) = 115{,}223$ matches Appx. C.

*$\pm 0.00$.* Rounding, not under-dispersion. At four decimals the GNN folds at $\sigma = 0.05$ span $0.9829$ to $0.9886$ (SD $0.0021$) against $0.0002$ for the oracle, least squares in the true parameters; a byte-identical rerun differs by $0.0014$, so the spread sits near the non-deterministic GPU floor. We will report three decimals. The $0.90 \pm 0.21$ entry is one bad fold, rollout $r \geq 0.998$ on four and $0.482$ on the fifth. That is the degeneracy at $\sigma = 0$.
