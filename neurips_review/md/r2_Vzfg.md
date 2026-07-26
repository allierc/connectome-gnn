We thank the reviewer. Our claim that predicting activity is not the same as recovering mechanism currently rests on one quantitative test, the $50\%$-edge ablation that both sequence baselines fail, plus one qualitative panel, the Jacobian comparison for the recurrent MLP in Supp. Fig. 18. Q4 and Q3 make it quantitative and apply it to our own model; Q1 and Q2 ask whether the result survives outside the regime we tested. We answer all four.

**Q1. Strongly recurrent benchmarks.** Accepted. We chose Flyvis for anatomy and scale: $13{,}741$ neurons, $65$ cell types, $434{,}112$ measured synapses at $0.23\%$ density, the sparse repeat-column structure that makes the biological inverse problem hard. Flyvis is predominantly feedforward, and the paper measures it: a stimulus-only MLP that never sees the connectome predicts every voltage at $r = 0.97$ from $16$ past frames (Appx. H.2). Limitation 5 reports this, adding that "stronger recurrent regimes are untested, though [Allier et al., 2026] suggests they need not be harder".

Three systems, differing in wiring, recurrence and activity:

| system | connectivity | recurrence | activity | $R^2_{\widehat W}$ ($\sigma{=}0$) |
|---|---|---|---|---|
| random assemblies [Allier et al., 2026] | random, dense | strong | chaotic, rich | $1.00$ |
| Flyvis-217 (this paper) | measured, $0.23\%$ | weak | stimulus-driven | $0.89$ |
| *Drosophila* CX | measured, $42\%$ | ring attractor | manifold-confined | $0.60$ * |

** provisional, single seed, training still in progress; the final value follows during the discussion period.*

Strong recurrence with rich activity is not the hard case; the first row shows that. A ring attractor is. Its dynamics contract onto a one-dimensional manifold of bump positions, so weights that move a state along the ring are constrained by the recording, while weights that would push it off the ring are undone by the attractor and constrained by nothing. Fit quality stays high while those directions stay unknown.

**Provisional.** We ran the *Drosophila* central complex — $156$ neurons, $7$ cell types, $10{,}263$ hemibrain edges at $180\times$ the density of Flyvis, so it separates recurrence from sparsity — through the identical pipeline. Single seed, still training, final numbers to follow: $R^2_{\widehat W} = 0.60$ (GNN) at $\sigma = 0$, $0.88$ at $0.05$ and $1.00$ at $0.5$, the oracle $1.00$ at both $\sigma > 0$ and not converging at $\sigma = 0$. The prediction we recorded holds, below Flyvis's $0.89$ noise-free with a steeper $\sigma$ dependence, and the generator says why: the activity's effective rank$(90\%)$ rises from $7$ to $37$ between $\sigma = 0.05$ and $0.5$ while $\mathbf{W}$ is unchanged at $11$. What governs recovery is how much of the space the activity visits, not how recurrent the circuit is — which is also why Yoon et al.'s [2025] smooth ring profile over $19$ of $100$ neurons is not the comparison it appears to be against $4\times10^5$ free weights on a measured connectome.

*On [2].* Das and Fiete show that in strongly recurrent circuits activity-based inference invents connections between unconnected but correlated neurons, and that noise or perturbation reduces the bias by enriching the activity distribution. That is our mechanism, so we will cite our process-noise result as a measured instance of their prescription rather than an independent finding. The failures differ: theirs is a false edge where no synapse exists, ours a true edge whose weight is unidentifiable against its same-type neighbours.

*On [1].* Pruned Dale-constrained RNNs are a good generator here. Our method does not assume Dale's law, since $\widehat W_{ij}$ is unconstrained and $g_\phi^2$ carries only magnitude, so such a benchmark would test sign recovery rather than sign compliance.

**Q2. Calcium-like observation model.** Accepted, and it is the change that most improves transfer to real recordings. The submitted degradations decompose the problem but never compose it: $1/5$ frames ($R^2_{\widehat W} = 0.30$) is subsampling alone, measurement noise ($0.63$ at $\gamma = 0.1$, $0.38$ at $\gamma = 0.2$) an additive term alone. A calcium model applies both at once and adds a saturation that compresses the large excursions carrying the weight information.

**[pending]** A causal double-exponential indicator kernel (GCaMP6f/6s/7f/8f/8m) and anti-aliased resampling to a slower imaging rate are already implemented. We add the saturation and the shot/read term, apply it to the same trajectories so nothing else changes, and report recovery on raw $\Delta F/F$ and after deconvolution, the practical question being whether standard preprocessing recovers what the observation model destroyed. We expect it to be worse than any single row of Supp. Tab. 4, and intend to present it as a failure mode rather than a robustness result.

**Q3. Out-of-distribution test.** We accept that our evidence for "black boxes predict but do not recover" is indirect. This test makes it direct and needs no retraining: vary $I_i(t)$ only, roll out the frozen checkpoints for $1000$ steps ($20$ s), and score against ground truth integrated from Eq. 1 under the same drive.

| stimulus | GNN roll. $r$ | Known-ODE roll. $r$ | recurrent MLP roll. $r$ |
|---|---|---|---|
| naturalistic held-out (ID reference) | $1.0000$ | $1.0000$ | [pending] |
| full-field white noise | $0.9986$ | $0.9992$ | [pending] |
| current injection, $5\%$ of neurons, $5\sigma_I$ | $0.9895$ | $0.9998$ | [pending] |
| current injection, $5\%$ of neurons, $20\sigma_I$ | $0.9748$ | $0.9985$ | [pending] |

In distribution the two are indistinguishable. Replacing natural video with spatially uniform white noise barely moves either, which is informative: the drive is still delivered through the photoreceptors, so the circuit is not taken far from the states it was fitted on. Direct current injection does take it off the manifold, and there a gap opens and grows with amplitude, $0.0103$ at $5\sigma_I$ and $0.0237$ at $20\sigma_I$, where $\sigma_I$ is the SD of the visual drive. The oracle stays above $0.998$ throughout, being exact by construction; the GNN degrades gently. At $1\sigma_I$ the gap is $0.0008$, i.e. that perturbation is not yet out of distribution, which sets the scale at which this test starts to discriminate.

*What this does and does not settle.* It bounds our own model's off-manifold fidelity. It does not yet make the black-box comparison the reviewer asks for: the gap above is between our model and an oracle that knows the equation, not between a mechanistic model and one that only predicts. The recurrent-MLP and EED checkpoints are temporarily inaccessible, held by a co-author who is away, so that column stays blank and will be filled during the discussion period; it needs rollouts, not retraining.

**Q4. GNN and Known-ODE Jacobians.** Agreed. Supp. Fig. 18 uses a Jacobian mismatch to argue against the MLP and never applies the same test to our own model.

At each of $200$ held-out frames we compute $J_{ij} = \partial \dot v_i / \partial v_j$ for every model. Differentiating Eq. 1 gives $J_{ij} = W_{ij}\,\mathrm{ReLU}'(v_j)/\tau_i$ and $J_{ii} = -1/\tau_i$, where $\mathrm{ReLU}'$ is $1$ when $v_j$ is above the rectifier threshold at zero and $0$ otherwise. The Known-ODE has the same form in $\widehat W, \hat\tau$; the GNN is differentiated by autograd through $f_\theta$ and $g_\phi$, since the ReLU kinks make finite differences unusable. All models see the same states, on the noise-free blank50 run Supp. Fig. 18 uses.

| block | model | $R^2$ vs GT $J$ | Pearson $r$ | $|J|$ off graph | sign agr. |
|---|---|---|---|---|---|
| off-diagonal $J_{ij}$ | Known-ODE | $0.977$ | $0.988$ | $0$ by construction | $0.882$ |
|  | GNN | $0.970$ | $0.988$ | $0$ by construction | $0.878$ |
|  | recurrent MLP | [pending] | [pending] | [pending] | [pending] |
| diagonal $J_{ii}$ | Known-ODE | $0.944$ | $0.975$ | — | $1.000$ |
|  | GNN | $0.865$ | $0.963$ | — | $1.000$ |
|  | recurrent MLP | [pending] | [pending] | [pending] | [pending] |

The GNN matches the true off-diagonal Jacobian at $R^2 = 0.970$, within $0.007$ of the oracle, so it has the right local dynamics and not merely plausible parameters. This is not an artefact of silent edges: $92.9\%$ carry a nonzero ground-truth Jacobian, and restricting to those moves $R^2$ by under $10^{-3}$. The diagonal is weaker for the GNN, $0.865$ against $0.944$, because the oracle holds $-1/\tau_i$ as a parameter while the GNN expresses the leak through $\partial f_\theta / \partial v$.

A good Jacobian is not a recovered connectome. $J$ is $\mathbf{W}$ rescaled twice: by $1/\tau_i$, and, once averaged over frames, by the fraction $a_j$ of them in which neuron $j$ sits above zero. Those rescalings change which edges dominate the score, which is why $J$ scores higher than $\mathbf{W}$ itself. On this fold the GNN reaches $0.970$ against $R^2_{\widehat W} = 0.913$, the oracle $0.977$ against $0.964$; for the oracle the whole gap is the rescaling, $0.964$ to $0.969$ after dividing by $\tau_i$, to $0.977$ after the activity factor, with no change to the fit. Agreement with the vector field therefore cannot separate a recovered connectome from a dynamically equivalent one, which is the argument we make about BrainTrace to Reviewer 1bPK.

Two entries carry less than they appear to. The fraction of $\sum_{i \neq j} |J_{ij}|$ on unconnected pairs is zero by construction for both our models, so it reports the hypothesis class, not what was learned. Sign agreement of $0.88$ reduces, once averaged over frames, to sign recovery of $\mathbf{W}$; the oracle sits at the same $0.882$.

**What we'll change.** Add these panels and metrics beside Supp. Fig. 18, with the MLP row once its checkpoints return, and state what this measurement forces: recovering the vector field is weaker than recovering the connectome. That qualification belongs beside the Appx. C degeneracy argument, which so far we apply to other people's fits and not to our own.

**Minor comments.** Both agreed. The Appx. F.2 mapping from $f_\theta, g_\phi$ to $\hat\tau, \widehat V^{\mathrm{rest}}, \widehat W$ is what makes the GNN interpretable rather than another black box, and we will move it to the main text. We will also reset Figs. 1 and 2 at main-text size, splitting Fig. 1's extraction panel out into the section that will describe it.
