We answer the six weaknesses in order. Concern 1 is where the new work is; the rest state what we concede and what we intend to change.

**1. Identifiability analysis internally inconsistent.** The inconsistency is real. Appx. C asserted that process noise breaks the columnar degeneracies; we have now measured that.

*Eq. (11) is a count, not a kernel.* $308{,}160 = \sum_{i,\alpha} (k_{i\alpha}-1)$ is correct as computed: it counts sum-zero directions within same-type presynaptic groups, a structural property of the connectome. Calling that set a kernel was the error. Same-type neurons deliver near-identical, time-shifted copies of one signal, so raising one input and lowering a sibling leaves the summed drive almost unchanged; but time-shifted signals are correlated, not identical, so those directions carry small *nonzero* singular values.

*The two percentages are two points on one curve.* They answer different questions at different tolerances on the same spectrum, which the submission never said. Ranking weight directions by $\epsilon = \lambda_k/\lambda_{\max}$ and counting the share of edges degenerate at each tolerance, the numerical $6.4\%$ is the value at $\epsilon = 10^{-22}$ and Eq. (11)'s $71\%$ is reached at $\epsilon = 2.5\times10^{-6}$, read off the same $\sigma = 0$ curve rather than chosen. A strict tolerance counts only edges the data cannot constrain at all; a looser one also counts weakly constrained columnar ones.

*Why process noise helps, now measured.* Appx. C writes the noise-free ODE as a per-neuron linear system $A_i \theta_i = b_i$, one row per timestep and one column per unknown of neuron $i$. We take the singular values of $A_i$ by SVD and rank each weight direction by $\epsilon = (\sigma_k/\sigma_1)^2$; a direction below a fixed tolerance is one the activity does not constrain, and an edge counts as degenerate once a direction supporting it falls below it. No weights are fitted and no solver runs, so this measures the conditioning of the inverse problem, not the behaviour of our GNN.

| process noise $\sigma$ | degenerate $W$ null (%) | degenerate $W$ sloppy (%) |
|---|---|---|
| $0$ | $6.35 \pm 0.02$ | $4.01 \pm 0.36$ |
| $0.05$ | $2.44 \pm 0.01$ | $0.32 \pm 0.00$ |
| $0.5$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |

*Tolerances $\epsilon \leq 10^{-22}$ (null) and $\leq 10^{-12}$ (sloppy); mean $\pm$ SD over five folds. The horizon is fixed across $\sigma$ and $\epsilon$ is scale-invariant, so the trend is not an artefact of record length or of noise inflating variance.*

Two competing readings are excluded. That noise merely smooths the GNN's loss landscape is ruled out by the oracle, which has no learned functions to regularise yet improves identically ($R^2_{\widehat W}: 0.96 \to 0.99 \to 1.00$). That noise lifts the spectrum by a numerical floor is ruled out by column equilibration, which fixes the sum of squared singular values, so noise can only redistribute a fixed total by decorrelating columns. We claim only what the table supports: process noise improves the conditioning. Whether it acts selectively on the columnar directions or lifts the spectrum generally the degenerate-edge fraction cannot say, and we will report that rather than assert it.

**What we revise.** State both thresholds as $\epsilon \leq 10^{-22}$ and $\leq 10^{-12}$, which reproduce the reported $\approx\!10\%$ where the printed $64000\times$ factor does not; replace "kernel" with $\ker_\epsilon$ at that tolerance, keeping $308{,}160$ as an exact count; split Appx. C into C.1 (identifiability) and C.2 (dynamical equivalence at an explicit tolerance), promoting C.2 as the analytical backbone of the prediction $\neq$ mechanism claim; and replace the asserted noise interpretation, in the appendix and in every claim citing it, with the measurement above.

**2. The setting is strongly model-matched.** Accepted. Of the nine listed items, three are properties of the preparation rather than the method, since a measured connectome, a known stimulus and a blank interval are all available in a real experiment and the paper already degrades the first two; two are inaccurate as stated, since no sign convention is imposed and only the aggregation is additive; and four are genuine: first-order dynamics, direct voltage observation, matched $\Delta t$, and the monotonicity priors. We tested three of the four. The GNN holds within $0.023$ $R^2_{\widehat W}$ of the oracle out to $\Delta t_{\mathrm{obs}} = 40$ ms and falls to $0.315$ at $100$ ms; an unobserved adaptation current costs $0.078$ at $g_a = 0.3$; the monotonicity priors are not load-bearing. Numbers in our reply to Reviewer 1bPK, Q1.

**What we revise.** Scope the title, abstract and contributions to model-matched in-silico feasibility, and replace "without assuming the form of the dynamical equation" with the class actually assumed: first-order dynamics with additive pairwise aggregation over the supplied graph, agnostic only to the scalar nonlinearities.

**3. Biological and practical relevance.** Conceded on the premise: no dataset pairs synapse-resolution connectivity with dense activity in one animal. Measurement noise remains the bottleneck, $R^2_{\widehat W} = 0.63$ and $R^2_{\hat\tau} = 0.28$ at $\gamma = 0.1$, $0.38$ and $0.08$ at $\gamma = 0.2$, with the oracle failing alongside, which places the limit in the inverse problem rather than in our architecture. [PENDING: calcium observation model and a strongly recurrent circuit, both running; Reviewer Vzfg, Q1 and Q2.]

**What we revise.** Move measurement noise from Limitations into the contributions, state the method as demonstrated under process noise and not at realistic measurement noise, and present the optogenetics analogy as an untested hypothesis rather than a prescription.

**4. "Mechanism recovery" is ambiguous.** Accepted; the paper uses one phrase for four claims of decreasing strength. Exact parameter equality we do not claim. Recovery up to an equivalence class we do claim, for $\sigma > 0$. Agreement of the local vector field we now measure: $R^2 = 0.970$ against the true Jacobian, with the oracle at $0.977$. Counterfactual competence under edge ablation is necessary but not sufficient, since it does not establish uniqueness. The third is weaker than the second, and we say so: a Jacobian is $\mathbf{W}$ rescaled by $1/\tau_i$ and by each source's active fraction, so agreement there cannot separate a recovered connectome from a dynamically equivalent one.

**What we revise.** State the four senses explicitly in Sect. 1 and attach each claim to one of them; fix the excluded set a priori rather than by residual, and report full-sample metrics for every table (Reviewer 1bPK, Q2).

**5. Methodological novelty overstated.** Accepted. The decomposition is inherited from Allier et al.; AMAG refines an adjacency it initialises rather than inferring one; Yoon et al. [2025] infer a $100$-neuron ring from $19$ observed cells, recovering a smooth weight profile rather than individually free weights. The degeneracy itself is older still, Prinz, Bucher and Marder [2004] and Golowasch, Goldman, Abbott and Marder [2002], with Das and Fiete drawing the consequence for inference. The defensible contribution is the connectome-scale characterisation, the identifiability analysis, the degradation battery and the edge ablation.

**What we revise.** Reframe the contribution as a large-scale benchmark of these ideas rather than their origin, and add a Sect. 2 paragraph separating forward connectome models (Shiu) from fits to activity (BrainTrace, Mi, Pospisil, Lappalainen).

**6. Agentic hyperparameter search.** Accepted in full: no matched-compute baseline against random or Bayesian search, and it optimises $R^2_{\widehat W}$ against ground truth, which is unavailable in the intended application.

**What we revise.** Demote it from contribution (3) to a methods note, replace the raw logs with a curated summary, remove the self-assessments, and retain only the outcome: one configuration used unchanged everywhere.

**Reporting issues.** $N$ reconciled: $13{,}697$ neurons have at least one incoming edge and $44$ have none, so Appx. C counts the former. The $\pm 0.00$ entries are rounding, not under-dispersion.

**What we revise.** Report three decimals throughout, spell out SIREN at first use, add unfiltered metrics to all four tables, and reset Figs. 1 and 2 at main-text font size, splitting Fig. 1's extraction panel into the section that describes it.
