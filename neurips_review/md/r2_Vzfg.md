We thank the reviewer. Our claim that predicting activity is not recovering mechanism rests on one quantitative test, the $50\%$-edge ablation both sequence baselines fail, and one qualitative panel, the Jacobian comparison in Supp. Fig. 18. Q3 and Q4 make it quantitative and apply it to our own model; Q1 and Q2 ask whether it survives outside the regime we tested.

**Q1. Strongly recurrent benchmarks.** Accepted. We chose Flyvis for anatomy, scale and the sparse repeat-column structure that makes the inverse problem hard. It is predominantly feedforward and the paper measures that: a stimulus-only MLP never seeing the connectome predicts every voltage at $r = 0.97$ from $16$ past frames (Appx. H.2). Limitation 5 says so, and that stronger recurrent regimes are untested.

Three systems, differing in wiring, recurrence and activity:

| system | recurrence | activity | GNN $0$ | GNN $.05$ | GNN $.5$ | ODE $0$ | ODE $.05$ | ODE $.5$ |
|---|---|---|---|---|---|---|---|---|
| random assemblies [Allier 2026] | strong | chaotic | $1.00$ | n.r. | n.r. | n.r. | n.r. | n.r. |
| Flyvis-217, $0.23\%$ | weak | stimulus-driven | $0.89$ | $0.99$ | $1.00$ | $0.96$ | $0.99$ | $1.00$ |
| *Drosophila* CX, $42\%$ | ring attractor | manifold-bound | $0.54$ | $0.84$ | $1.00$ | $-2.07$ | $1.00$ | $1.00$ |

*Cells are $R^2_{\widehat W}$ at process noise $\sigma$; ODE = Known-ODE oracle. CX single seed, $10$ epochs, run for this reply; n.r. = not reported.*

Strong recurrence with rich activity is not the hard case; row one shows that. A ring attractor is: it contracts onto a low-dimensional manifold, so weights moving a state along the ring are constrained by the recording while those pushing it off are undone by the attractor, and fit quality stays high while they stay unknown. The CX row is $156$ neurons, $7$ types, $10{,}263$ hemibrain edges at $180\times$ Flyvis's density, separating recurrence from sparsity, and the reason is visible before any fitting. $\mathbf{W}$ carries $12$ significant directions ($90\%$ of its spectral energy), identical across $\sigma$ since the noise enters the dynamics, not the weights; what changes is how many the activity explores, $6$ at $\sigma = 0$, $7$ at $0.05$, $95$ at $0.5$. Recovery is partial where the recording spans fewer directions than $\mathbf{W}$ needs and exact once it spans more: what governs it is how rich the activity is, not how recurrent the circuit is. At $\sigma = 0$ even the oracle fails ($-2.07$ against the GNN's $0.54$), so what fails there is the recording, not the hypothesis class.

*On [2].* Das and Fiete show that in strongly recurrent circuits activity-based inference invents connections between unconnected but correlated neurons, and that noise or perturbation reduces the bias. That is our mechanism; we cite our process-noise result as a measured instance of their prescription, not an independent finding. The failures differ: theirs is a false edge where no synapse exists, ours a true edge unidentifiable against its same-type neighbours.

*On [1].* Pruned Dale-constrained RNNs are a good generator here, though we do not assume Dale's law ($\widehat W_{ij}$ unconstrained, $g_\phi^2$ carrying only magnitude), so such a benchmark tests sign recovery, not compliance.

**Q2. Calcium-like observation model.** Accepted, and it is the change that most improves transfer to real recordings. In the submission the degradations never compose: the $1/5$-frames row ($R^2_{\widehat W} = 0.30$) is subsampling alone and the $\gamma = 0.1$ row ($0.63$) is additive noise alone. Neither is an indicator model, and the runs below add no measurement noise at all: process noise stays at $\sigma = 0.05$ throughout and every degradation is a property of the imaging.

We built it: four observables on the same trajectories. (i) GCaMP6f kernel alone; (ii) plus saturation past $K_d$, set at $1.57\times$ the trace SD; (iii) plus a $500$ ms exposure ($2$ Hz); (iv) all of these with shot noise matched in SNR to our $\gamma$ rows. Before fitting, the $\dot C$ signal identifying $\mathbf{W}$ retains $0.40$, $0.70$ and $0.26$ of (i) under (ii)–(iv); deconvolved back to voltage, as an experimenter would, each distortion alone is largely invertible ($r = 0.98$, $0.93$, $0.90$) and shot noise defeats it ($0.42$), since inverting a low-pass filter amplifies what noise fills in. Each observable was then handed to both arms as the recording, $13{,}741$ neurons, the paper's schedule. In the calcium rows the model integrates $C$ itself; in the deconvolved rows it integrates the Wiener estimate of $v$, which is what an experimenter actually fits.

| observable | GNN $R^2_{\widehat W}$ | Known-ODE $R^2_{\widehat W}$ |
|---|---|---|
| voltage (control) | $0.984$ | — |
| calcium, kernel only | $-0.010$ | $0.373$ |
| \quad + saturation | $-0.010$ | $0.077$ |
| \quad + $2$ Hz exposure | $-0.010$ | $0.278$ |
| \quad + shot noise, $\gamma = 0.1$ | $-0.010$ | $-0.009$ |
| \quad + shot noise, $\gamma = 0.2$ | $-0.010$ | $-0.009$ |
| \quad all composed | $-0.010$ | $-0.009$ |

*Best over training. On the deconvolved observable below the oracle peaks early and then drifts, so we report the maximum rather than the endpoint throughout; the endpoints are lower ($0.332$, $0.006$, $0.277$, $-0.379$, $-0.325$, $-0.099$).*

This is a negative result and we report it as one. On the indicator trace the GNN recovers no weight structure at any degradation, the kernel alone included; $-0.010$ is $\widehat{\mathbf{W}}$ never leaving its initialisation, so nothing is recovered at any point in training rather than recovered and then lost. The oracle, which need only fit constants, keeps a third of it and loses even that once shot noise is present. The mechanism is that $\mathbf{W}$ is identified from the derivative while the indicator is a low-pass filter on exactly that band, so the trace stays informative long after $\dot C$ has stopped being so. This is why a good rollout $r$ on a calcium trace is not evidence that the weights behind it were recovered.

*On deconvolution.* We should be careful about what deconvolution can settle here, because the noise-free arm cannot settle it. With a perfectly known kernel and no measurement noise, $C = K * v$ is an exactly invertible linear map and recovering $v$ is division in Fourier space, limited only by floating-point round-off: we measure $r = 0.9998$ on the voltage and $0.993$ on its derivative. That is a property of the arithmetic, not a claim about calcium imaging, and we report it only as a check that the observation model and its inverse agree.

What the inversion is actually limited by is photon noise, and it is limited violently. Adding i.i.d. noise of SD $\gamma$ to the trace before deconvolving, with the Tikhonov parameter retuned at each level so each row is the best inversion available rather than a fixed filter:

| $\gamma / \mathrm{SD}(C)$ | SNR (dB) | $r$ on $v$ | $r$ on $\dot v$ | GNN $R^2_{\widehat W}$ | Known-ODE $R^2_{\widehat W}$ |
|---|---|---|---|---|---|
| $0$ | $\infty$ | $0.9998$ | $0.993$ | $0.967$ | $0.915$ |
| $0.01$ | $40$ | $0.980$ | $0.435$ | $0.144$ | $0.405$ |
| $0.03$ | $30$ | $0.970$ | $0.280$ | $-0.011$ | $0.181$ |
| $0.10$ | $20$ | $0.956$ | $0.172$ | $-0.010$ | $-0.009$ |

*$\lambda$ retuned per row by maximising $r(\dot v)$, so each row is the best inversion available at that noise level rather than a fixed filter. $R^2_{\widehat W}$ is best over training. The oracle arms have finished; the GNN arms are still training and their column is preliminary.*

Deconvolution works, and that is worth stating plainly: given a clean trace it returns the weights in full, $R^2_{\widehat W} = 0.967$ against the $0.984$ voltage control. The obstacle is not the indicator kernel, which is invertible, but the photon noise that inversion amplifies. Two-photon GCaMP recordings sit at roughly $20$ to $40$ dB, that is $\gamma/\mathrm{SD} \approx 1$ to $10\%$, and across that whole range recovery is gone: the best either arm reaches at $40$ dB is $0.405$, and by $20$ dB neither leaves initialisation. Note what the columns do: the trace stays faithful ($r \geq 0.96$ everywhere) while its derivative collapses, and $R^2_{\widehat W}$ follows the derivative rather than the trace. This is the same fact as the calcium rows above, seen from the preprocessing side. Process noise, which is what makes $\mathbf{W}$ identifiable at all, is white and therefore puts the identifying signal at high frequency; the indicator attenuates exactly that band; and inverting the attenuation multiplies up whatever noise now fills it. A high rollout $r$ on a calcium trace is consistent with the weights being entirely unrecoverable.

So the answer to the question deconvolution was supposed to settle is negative, and it does not depend on which model is used: even given Eq. 1 and only constants to fit, a deconvolved recording at a realistic photon budget does not identify $\mathbf{W}$.

One observation we did not expect and report as unresolved. On the deconvolved observable neither arm converges at $\sigma$-realistic noise — both peak early and then drift away from the truth, the oracle from $0.915$ at $32$k to $0.489$ at $1.52$M. Oracle runs on true voltage do not do this; they rise monotonically and end at their maximum ($0.977$–$0.979$ across five folds). The deconvolved trace is near voltage but not identical ($r = 0.9998$, derivative amplitude $0.93$), and we suspect the residual mismatch is being absorbed into $\widehat{\mathbf{W}}$, but we have not measured that and do not claim it. It is why the tables above report the best value over training rather than the endpoint, which is the more favourable convention for both arms and still gives the negative result.

We also correct an error of our own here. An earlier version of this reply carried deconvolved fits at $R^2_{\widehat W} = 0.07$ and read them as evidence that deconvolution cannot rescue the weights. Those runs used a derivative-penalising Tikhonov prior carried over from a noisy-data setting; on a noise-free trace it suppressed about $70\%$ of the derivative amplitude while leaving trace correlation at $0.98$, so it destroyed the very quantity being measured. The number characterised our preprocessing, not the observation model, and we withdraw it.

**Q3. Out-of-distribution test.** Our evidence for "black boxes predict but do not recover" is indirect; this test makes it direct and needs no retraining. Only the drive $I_i(t)$ changes: from a shared initial state each frozen checkpoint is rolled out on its own output for $1000$ steps ($20$ s), scored over the whole trajectory against ground truth integrated from Eq. 1 under that drive. A model that recovered the mechanism should hold in states it never saw.

| stimulus | GNN roll. $r$ | Known-ODE roll. $r$ |
|---|---|---|
| naturalistic held-out (ID reference) | $1.0000$ | $1.0000$ |
| full-field white noise | $0.9986$ | $0.9992$ |
| current injection, $5\%$ of neurons, $5\sigma_I$ | $0.9895$ | $0.9998$ |
| current injection, $5\%$ of neurons, $20\sigma_I$ | $0.9748$ | $0.9985$ |

The two stimuli move progressively off the manifold. White noise is one time series broadcast to every photoreceptor, and it barely moves either model. That is informative: an unnatural stimulus is not an unfamiliar state, since the drive still enters through the photoreceptors and the circuit visits states it was fitted on. Injection adds a constant to a random $5\%$ of *all* neurons, most beyond the reach of any stimulus, forcing voltages the visual pathway cannot produce. There a gap opens and grows with amplitude, $0.0008$ at $1\sigma_I$, $0.0103$ at $5\sigma_I$, $0.0237$ at $20\sigma_I$ ($\sigma_I$ = SD of the visual drive), so $1\sigma_I$ is not yet out of distribution and sets the discriminating scale. The oracle holds above $0.998$, knowing Eq. 1 and fitting only constants, where the GNN's $f_\theta, g_\phi$ are evaluated outside their training range.

*What this settles.* It bounds our own off-manifold fidelity but not the comparison asked for: the gap is between our model and an oracle knowing the equation, not between a mechanistic model and one that only predicts. The recurrent-MLP and EED checkpoints are with an absent co-author; that column needs rollouts, not retraining, and will be filled during the discussion.

**Q4. GNN and Known-ODE Jacobians.** Agreed: Supp. Fig. 18 uses a Jacobian mismatch against the MLP and never applies the same test to our own model.

At each of $200$ held-out frames we compute $J_{ij} = \partial \dot v_i / \partial v_j$ for every model. Differentiating Eq. 1 gives $J_{ij} = W_{ij}\,\mathrm{ReLU}'(v_j)/\tau_i$, $J_{ii} = -1/\tau_i$; the Known-ODE has the same form in $\widehat W, \hat\tau$, and the GNN is differentiated by autograd through $f_\theta, g_\phi$, since the ReLU kinks make finite differences unusable. All models see the same states, on the noise-free blank50 run of Supp. Fig. 18.

| block | model | $R^2$ vs GT $J$ | Pearson $r$ | $|J|$ off graph | sign agr. |
|---|---|---|---|---|---|
| off-diagonal $J_{ij}$ | Known-ODE | $0.977$ | $0.988$ | $0$ by construction | $0.882$ |
|  | GNN | $0.970$ | $0.988$ | $0$ by construction | $0.878$ |
| diagonal $J_{ii}$ | Known-ODE | $0.944$ | $0.975$ | n/a | $1.000$ |
|  | GNN | $0.865$ | $0.963$ | n/a | $1.000$ |

*MLP row pending; checkpoints inaccessible (Q3).*

The GNN matches the true off-diagonal Jacobian at $R^2 = 0.970$, within $0.007$ of the oracle, so it has the right local dynamics, not merely plausible parameters. Not an artefact of silent edges: $92.9\%$ carry a nonzero ground-truth Jacobian and restricting to those moves $R^2$ by $<10^{-3}$. The diagonal is weaker, $0.865$ against $0.944$: the oracle holds $-1/\tau_i$ as a parameter, the GNN expresses the leak through $\partial f_\theta / \partial v$.

A good Jacobian is not a recovered connectome. $J$ is $\mathbf{W}$ rescaled twice: by $1/\tau_i$, and, over frames, by the fraction in which neuron $j$ sits above zero. Those rescalings change which edges dominate, which is why $J$ scores higher than $\mathbf{W}$: the GNN reaches $0.970$ against $R^2_{\widehat W} = 0.913$, the oracle $0.977$ against $0.964$, where the whole gap is the rescaling ($0.964 \to 0.969$ after $1/\tau_i$, $\to 0.977$ after the activity factor) with no change to the fit. Agreement with the vector field therefore cannot separate a recovered connectome from a dynamically equivalent one, our argument about BrainTrace to Reviewer 1bPK.

Two entries carry less than they appear to: the off-graph fraction is zero by construction for both our models, reporting the hypothesis class rather than what was learned, and sign agreement reduces over frames to sign recovery of $\mathbf{W}$, the oracle at the same $0.882$.

**What we'll change.** Add these panels and metrics beside Supp. Fig. 18, with the MLP row once its checkpoints return, and state what this measurement forces: recovering the vector field is weaker than recovering the connectome, beside the Appx. C degeneracy argument we so far apply only to other people's fits.

**Minor comments.** Both agreed. The Appx. F.2 mapping from $f_\theta, g_\phi$ to $\hat\tau, \widehat V^{\mathrm{rest}}, \widehat W$ is what makes the GNN interpretable rather than another black box, and we will move it to the main text. We will also reset Figs. 1 and 2 at main-text size, splitting Fig. 1's extraction panel out into the section that will describe it.
