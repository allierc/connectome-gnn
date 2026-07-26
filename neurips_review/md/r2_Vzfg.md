We thank the reviewer. Our claim that predicting activity is not the same as recovering mechanism currently rests on one quantitative test, the $50\%$-edge ablation both sequence baselines fail, and one qualitative panel, the Jacobian comparison for the recurrent MLP in Supp. Fig. 18. Q4 and Q3 make it quantitative and apply it to our own model; Q1 and Q2 ask whether it survives outside the regime we tested. We answer all four.

**Q1. Strongly recurrent benchmarks.** Accepted. We chose Flyvis for anatomy and scale: $13{,}741$ neurons, $65$ cell types, $434{,}112$ measured synapses at $0.23\%$ density, the sparse repeat-column structure that makes the biological inverse problem hard. It is predominantly feedforward, and the paper measures that: a stimulus-only MLP never seeing the connectome predicts every voltage at $r = 0.97$ from $16$ past frames (Appx. H.2), which Limitation 5 reports, adding that "stronger recurrent regimes are untested, though [Allier et al., 2026] suggests they need not be harder".

Three systems, differing in wiring, recurrence and activity:

| system | recurrence | activity | GNN $0$ | GNN $.05$ | GNN $.5$ | ODE $0$ | ODE $.05$ | ODE $.5$ |
|---|---|---|---|---|---|---|---|---|
| random assemblies [Allier, 2026] | strong | chaotic, rich | $1.00$ | — | — | — | — | — |
| Flyvis-217, $0.23\%$ dense | weak | stimulus-driven | $0.89$ | $0.99$ | $1.00$ | $0.96$ | $0.99$ | $1.00$ |
| *Drosophila* CX, $42\%$ | ring attractor | manifold-confined | $0.60$* | $0.88$* | $1.00$* | n.c.* | $1.00$* | $1.00$* |

*Cells are $R^2_{\widehat W}$ at process noise $\sigma \in \{0, 0.05, 0.5\}$; ODE = Known-ODE oracle. * provisional, single seed, still training; n.c. = did not converge; — = not reported for that system.*

Strong recurrence with rich activity is not the hard case; the first row shows that. A ring attractor is: it contracts onto a low-dimensional manifold of bump positions, so weights moving a state along the ring are constrained by the recording while weights pushing it off are undone by the attractor and constrained by nothing. Fit quality stays high while those directions stay unknown.

**Provisional.** The CX row is $156$ neurons, $7$ types, $10{,}263$ hemibrain edges at $180\times$ Flyvis's density, so it separates recurrence from sparsity; single seed, still training. The prediction we recorded holds, and the reason is visible before any fitting. $\mathbf{W}$ carries $12$ significant directions ($90\%$ of its spectral energy) and is identical across $\sigma$, since the noise enters the dynamics and not the weights; what changes is how many directions the activity explores — $6$ at $\sigma = 0$, $7$ at $0.05$, $95$ at $0.5$. Recovery is partial exactly where the recording spans fewer directions than $\mathbf{W}$ needs, and exact once it spans more. What governs recovery is how much of the space the activity visits, not how recurrent the circuit is.

*On [2].* Das and Fiete show that in strongly recurrent circuits activity-based inference invents connections between unconnected but correlated neurons, and that noise or perturbation reduces the bias by enriching the activity distribution — our mechanism, so we will cite our process-noise result as a measured instance of their prescription rather than an independent finding. The failures differ: theirs is a false edge where no synapse exists, ours a true edge unidentifiable against its same-type neighbours.

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

In distribution the two are indistinguishable. Spatially uniform white noise barely moves either, which is informative: the drive still arrives through the photoreceptors, so the circuit is not taken far from the states it was fitted on. Current injection does take it off the manifold, and there a gap opens and grows with amplitude — $0.0008$ at $1\sigma_I$, $0.0103$ at $5\sigma_I$, $0.0237$ at $20\sigma_I$, $\sigma_I$ being the SD of the visual drive — so $1\sigma_I$ is not yet out of distribution, which sets the scale at which this test discriminates. The oracle stays above $0.998$, exact by construction; the GNN degrades gently.

*What this does and does not settle.* It bounds our own model's off-manifold fidelity, but does not yet make the comparison asked for: the gap above is between our model and an oracle that knows the equation, not between a mechanistic model and one that only predicts. The recurrent-MLP and EED checkpoints are held by a co-author who is away, so that column stays blank and will be filled during the discussion period; it needs rollouts, not retraining.

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

A good Jacobian is not a recovered connectome. $J$ is $\mathbf{W}$ rescaled twice: by $1/\tau_i$, and, averaged over frames, by the fraction of them in which neuron $j$ sits above zero. Those rescalings change which edges dominate the score, which is why $J$ scores higher than $\mathbf{W}$ itself: the GNN reaches $0.970$ against $R^2_{\widehat W} = 0.913$, the oracle $0.977$ against $0.964$, and for the oracle the whole gap is the rescaling ($0.964 \to 0.969$ after $1/\tau_i$, $\to 0.977$ after the activity factor) with no change to the fit. Agreement with the vector field therefore cannot separate a recovered connectome from a dynamically equivalent one, which is our argument about BrainTrace to Reviewer 1bPK.

Two entries carry less than they appear to. The fraction of $\sum_{i \neq j} |J_{ij}|$ on unconnected pairs is zero by construction for both our models, so it reports the hypothesis class, not what was learned. Sign agreement of $0.88$ reduces, once averaged over frames, to sign recovery of $\mathbf{W}$; the oracle sits at the same $0.882$.

**What we'll change.** Add these panels and metrics beside Supp. Fig. 18, with the MLP row once its checkpoints return, and state what this measurement forces: recovering the vector field is weaker than recovering the connectome. That qualification belongs beside the Appx. C degeneracy argument, which so far we apply to other people's fits and not to our own.

**Minor comments.** Both agreed. The Appx. F.2 mapping from $f_\theta, g_\phi$ to $\hat\tau, \widehat V^{\mathrm{rest}}, \widehat W$ is what makes the GNN interpretable rather than another black box, and we will move it to the main text. We will also reset Figs. 1 and 2 at main-text size, splitting Fig. 1's extraction panel out into the section that will describe it.
