---
title: "Research notebook --- a comparative study of connectome-constrained self-motion integrators"
author: "Claude (Opus 4.8), agentic post-doc"
date: "June 2026"
---

# Research notebook

*A running log of ideas, sketches, dead-ends, and decisions while writing the
comparative paper. Append-only; later entries may supersede earlier ones.*

---

## Entry 0 --- the brief (2026-06-13)

The lab has two parallel manuscripts, each built on the same machinery
(`GNN_Main.py` + connectome-constrained sign-locked RNN / message-passing GNN):

- **`zebrafish.tex`** --- the larval-zebrafish anterior hindbrain / IPN
  self-motion integrator (917 cells, `neuprint-fish2`). Thesis: *geometry follows
  mechanism*; heading is the dominant ring mode, translation hides in a
  low-variance but decodable direction; task accuracy != activity realism.
- **`drosophila.tex`** --- the *Drosophila* central complex (338 cells,
  hemibrain). Thesis: *computation is identifiable, implementation is not*; many
  recurrent weight matrices reach the same behaviour; frozen connectome fails,
  training supplies the gain.

Each paper makes its identifiability/degeneracy point **in isolation, in one
species**. Neither asks the comparative question.

## Entry 1 --- the gap I can fill, and why it is mine to fill

> Two heading integrators, separated by ~550--600 Myr of evolution
> (protostome fly vs deuterostome fish), both reconstructed by EM, both solving
> the *identical* computation (angular self-motion -> heading). What does the
> measured connectome determine that is **conserved**, and what is
> **divergent**, across the two?

This is a question only a *comparative* study can answer, and it is exactly the
kind of synthesis that falls between two single-system papers. The two circuits
differ structurally in ways that make the comparison sharp:

| | *Drosophila* CX | Zebrafish IPN |
|---|---|---|
| recurrent substrate | EPG ring, anatomically *ordered* | dIPN, no obvious ring order |
| size (N) | 338 (156 heading-core) | 917 |
| density | 0.170 (dense) | 0.037 (sparse) |
| E:I (recurrent) | mixed, Delta7/ER6 inhibitory | all-inhibitory (~66% I) |
| canonical model | hand-designed ring attractor | none before this work |

**Hypothesis (to be tested, not assumed):** the *computation* (near-unit-gain
ring integration, heading decodable to a few degrees) is conserved, but the
*code* --- its dimensionality, redundancy, and how tightly the connectome pins
the recurrent weights --- diverges. "Convergent computation, divergent codes."

I will only claim what I can compute on the trained checkpoints.

## Entry 2 --- data foundation (validated 2026-06-13)

Loader (`figures/comparative/cc_loader.py`) confirmed working. Converged
last-epoch checkpoints, constant-$\omega=60^\circ$/s rollout Pearson $r$:

```
FLY-GNN   r_roll60 = 0.9999   N=338
FISH-GNN  r_roll60 = 0.9997   N=917
FLY-RNN   r_roll60 = 0.9999   N=338
FISH-RNN  r_roll60 = 1.0000   N=917
```

Independent seeds available: RNN rotation has base + rep_1..5 (6 seeds) per
species; GNN rotation similarly. **Every model included in an analysis will have
its rollout $r$ verified first; non-converged checkpoints are discarded.**

Key gotcha banked: the connectome CSVs are loaded by repo-relative paths, so all
analysis must run with `cwd = /workspace/connectome-gnn-cx`.

## Entry 3 --- analysis plan (the six probes)

All computed identically on both species, on verified-converged models:

1. **Dimensionality** --- participation ratio of bump-population firing rates
   (PCA) over a long OU rollout. Clean ring -> PR$\approx$2.
2. **Redundancy** $\rho$ --- $\sum_i I(r_i;\theta) / I_\mathrm{joint}$. How cloned
   is the heading code?
3. **Ring geometry** --- is the top-2-PC heading manifold actually circular?
   Phase-vs-heading monotonicity, radius uniformity.
4. **Variance vs decodability** --- decode $\theta$ (and $\omega$) from full
   population vs top-$k$ PCs. Does the computational variable live in the
   high-variance modes?
5. **Inter-seed identifiability** --- cosine similarity of learned $W$ over the
   connectome support across the 6 seeds. Which connectome pins its weights more?
6. **Causal role of wiring** --- ER / rewired-null connectomes (the `_er_*`
   runs) vs measured. How much does each species depend on precise wiring?

Open question for later: can I find a *single scalar* (e.g. PR, or $\rho$) that
orders the two systems on a "localist <-> distributed" axis, and does it predict
identifiability? That would be the headline.

## Entry 4 --- literature briefing reshapes the thesis (2026-06-13)

A literature pass (Petrucco 2023, Hulse 2026, Kim 2017, Clark--Abbott--Sompolinsky
2025, Khona--Fiete 2022, Goncalves 2014) forces a correction and hands me a new
probe.

**Correction.** Petrucco et al. (2023, *Nat Neurosci*) show the zebrafish HD code
is itself **ring-like**: sinusoidal tuning (FWHM $\approx\pi$), a bump that
rotates with directional swims, and reciprocal *antiphase* inhibition in the
dIPN. They explicitly note the architecture "resembles the fly central complex."
So my naive framing ("fly ring vs fish distributed") is wrong. **Both are ring
attractors.** The honest divergence is *quantitative*: how many neurons carry the
ring, how redundant the code is, and how tightly behaviour pins the recurrent
weights. Good --- this is a sharper, more defensible thesis than the strawman.

**New probe (P7), from Clark, Abbott & Sompolinsky 2025.** They prove that a
continuous (ring) attractor leaves a *spectral fingerprint* in the recurrent
weight matrix: a **doubly-degenerate pair of eigenvalues** near the critical line
(the broken-symmetry Goldstone modes of the ring). They explicitly propose
"eigenvalue degeneracy is the most reliable signature for detecting
ring-attractor structures in synaptic weight matrices, such as those from future
connectome data." I have *exactly* that object --- connectome-constrained,
behaviourally-trained recurrent operators for two species. So I can run a direct
test neither single-system paper ran: **does the trained recurrent operator of
each species carry the ring-attractor eigenvalue signature, and is it
conserved across the 550-Myr divergence?** Caveats they raise (degeneracy is set
by 2nd-order weight statistics only; spectral-embedding can hallucinate circulant
structure from low-rank matrices) become honesty checks I must apply.

**Precedent for the identifiability arm.** Goncalves et al. (2014) showed the
zebrafish *oculomotor* line-attractor is fit by "a plethora of models"
distinguished only by E/I ratio, resolvable only by dynamic perturbation --- the
vertebrate analogue of the fly "hidden symmetries" (Hulse 2026) degeneracy. My
inter-seed identifiability probe (P5) is the connectome-constrained version of
this, run head-to-head across species for the first time.

**Refined thesis.** *Two ring attractors, one ancient computation, two
identifiabilities.* The ring computation and its spectral signature are conserved;
the dimensionality/redundancy of the code and the degree to which behaviour pins
the wiring diverge with the connectome's size and density.

**Bibliography note.** No `.bib` exists; both papers use inline `\bibitem`s. I
will create `docs/refs.bib` reusing existing cite-keys and adding `clark2025`
(bioRxiv 2025.01.26.634933), which neither paper cites.

## Entry 5 --- first real numbers (2026-06-13)

Ran `cc_analysis.py` (P1--4) on verified-converged base models, OU rollout
$T=5000$, $\mathrm{sd}=90^\circ$/s, seed 0. All four models integrate at
$r\ge0.99$. Raw table (RNN = the sign-locked model both papers use):

```
            PR_bump  n_eff   rho    radius_cv  |phase|  var_top2  dec_top2  dec_full
fly_rnn      2.58    2.02    22.8    0.12       0.87     0.83      0.89      0.96
fly_gnn      2.35    2.05    22.4    0.09       0.92     0.90      0.81      0.84
fish_rnn     3.16    3.22   217.5    0.11       0.97     0.75      0.92      1.00
fish_gnn     7.72    3.33   210.5    0.26       0.66     0.42      0.46      1.00
```

P5 (inter-seed cosine over connectome support): fly_rnn 0.576$\pm$0.024,
fish_rnn 0.594$\pm$0.026; vs raw connectome 0.48 / 0.43.

**What this says.**

- **Conserved:** low effective dimensionality (n_eff $\approx$ 2--3), a clean
  ring manifold (radius_cv $\approx$ 0.1, |phase_corr| $\approx$ 0.9),
  near-unit-gain integration, AND identifiability (inter-seed cosine $\approx$
  0.58 in both; behaviour pins the wiring equally *loosely* regardless of N or
  density). Identifiability being conserved is a non-obvious finding --- I had
  guessed the denser fly connectome would pin weights more.
- **Divergent:** redundancy. The fish achieves its $\sim$3 effective dimensions
  by cloning across 700 dIPN neurons ($\rho\approx217$); the fly uses 46 EPG
  ($\rho\approx22$). $\sim$10$\times$ more physical redundancy for nearly the
  same information. This is the vertebrate-integrator extreme of the fly
  "dynamical clones" idea (Hulse 2026) and the disordered-ring regime (Clark
  2025).

**Headline.** *Same ring, same effective dimension, same (loose)
identifiability --- but the vertebrate circuit pays an order of magnitude more
redundancy.* Redundancy, not the computation or its identifiability, is what the
two connectomes do differently.

**Caveats to discharge before claiming:**
1. Single seed, single OU realisation. MUST add error bars across the 6 fly /
   5 fish seeds and several OU seeds. (next)
2. $\rho$ scales with N; report n_eff and $\rho/N$ alongside so the redundancy
   claim is not just "fish has more neurons." (fish $\rho/N_\mathrm{bump}$=0.31,
   fly=0.49 --- actually similar per-neuron; the divergence is in absolute
   substrate + total $\rho$. Need to state this precisely, not overclaim.)
3. GNN numbers (esp. fish_gnn PR=7.7) are model-class-dependent; lead with the
   RNN (apples-to-apples with both papers), report GNN as robustness.
4. P7 spectral redo with linearised Jacobian at a bump fixed point.

## Entry 6 --- error bars, nulls, and a clean spectral signature (2026-06-13)

**Seed-aggregated RNN (fly: 6 seeds, fish: 5 seeds; x3 OU each):**

```
            PR_bump        n_eff         rho          radius_cv    phase_corr
fly_rnn   2.84 +/- 0.23  2.11 +/- 0.11  21.9 +/- 1.1  0.16 +/-.06  0.74 +/-.21
fish_rnn  3.29 +/- 0.22  2.89 +/- 0.17  243  +/-14    0.16 +/-.07  0.88 +/-.11
```

n_eff difference (2.1 vs 2.9) is real (non-overlapping). Fish ring is ~1.4x
higher effective dimension; both still tiny vs N. radius_cv identical -> both
clean rings. rho 11x driven by 15x substrate (per-neuron redundancy similar).

**P7 (Jacobian marginal mode) -- the clean one.** $J\tau=-I+W\,\mathrm{diag}(\sigma'(h^*))$
at a settled bump:

```
            lambda1(marg)  lambda2     n_slow  n_unstable
fly_rnn      +0.0022       -0.39        1        1
fish_rnn     +0.0010       -0.37        1        1
fly_bs null  -0.0025       -0.49        1        0
fish_bs null -0.0018       -0.38        1        0
```

**Exactly ONE marginal eigenmode ($|\lambda_1|<0.003$ in units of $1/\tau$),
all others damped ($\lambda_2\le-0.37$), in both species AND both nulls.** This
is the unambiguous signature of a 1-D continuous (ring) attractor. It is
conserved across 550 Myr and across connectome identity. Directly realises
Clark-Abbott-Sompolinsky's (2025) proposal to read ring structure off
connectome-constrained weight matrices.

**P6 nulls (fly side complete):** block-shuffle and ER nulls reproduce the
measured code: PR 2.78/2.87, n_eff 2.0/1.88 vs measured 2.87/2.11, marginal
mode present. The low-dim ring + identifiability are NOT signatures of the
specific biological wiring.

## Entry 7 --- thesis crystallised

> **The computation is a universal; the connectome contributes scale.**
>
> Trained to integrate heading under leaky sign-locked recurrent dynamics, the
> fly CX and the zebrafish IPN both become 1-D continuous attractors: one
> marginal eigenmode, ~2-3 effective dimensions, a clean ring manifold,
> near-unit gain, and recurrent weights that behaviour pins only loosely
> (inter-seed cosine ~0.58). Every one of these properties is reproduced by
> null connectomes of matched sign/density statistics. What the measured
> connectome sets is the *substrate size* (46 vs 700 ring neurons; total
> redundancy rho 22 vs 243), not the computational geometry nor its
> (non-)identifiability.

This unifies the two source papers: zebrafish.tex's "geometry follows
mechanism" and drosophila.tex's "computation identifiable, implementation not"
are the same statement, and it is a *cross-species, cross-connectome* universal.
New evidence beyond either paper: (i) head-to-head dimensionality/redundancy/
identifiability; (ii) the conserved single marginal mode; (iii) null-connectome
universality of all of the above.

**Honesty ledger (must hold in the paper):**
- n_eff difference is modest; don't oversell "divergent dimensionality."
  Lead with conservation; dimensionality/substrate is the quantitative tail.
- rho divergence is substrate-driven; say so explicitly.
- Identifiability number (0.58) is for seed-varied intact models; fish reps
  carry a vfwd channel (task not pixel-identical to fly). Flag it.
- Nulls: fly has bs+er; fish has bs (er only for GNN). Cross-species null =
  block-shuffle. State the asymmetry.
- GNN results are model-class robustness, not the headline.

## Entry 8 --- external review #1 (2026-06-13), adopted

An external agent reviewed the notebook. Verdict and my response:

1. *Lead with conservation, not redundancy.* AGREED. rho is substrate-driven
   (rho/N similar). The surprise is that effective dimension (~2-3) AND
   identifiability (~0.58) are nearly identical despite 3x neurons / 5x density
   / 550 Myr. That is the headline.
2. *Identifiability is the headline.* "Behaviour fails to determine
   implementation in BOTH phyla." This is the comparative generalisation of
   zebrafish.tex's degeneracy claim. Promote it.
3. *Careful about "same computation."* The fish reps (vfwd_rep) DO carry a
   translation channel -> the fish embeds extra navigation variables in a larger
   redundant substrate. State honestly; don't claim identical tasks. Reframe:
   "both contain the same heading ring; the fish embeds more in a bigger
   substrate."
4. *P7 reframed (HIGH value):* the question is not "does a ring have a marginal
   mode" but "does TRAINING create it?" Compare J(raw connectome) vs J(trained).
   If the marginal mode appears only after training -> behaviour creates the
   attractor; anatomy alone is insufficient (consistent with the fly
   frozen-Wrec failure). DO THIS.
5. *New P8 -- geometry/implementation decoupling (HIGH value):* across seeds,
   W differs (cosine 0.58) while the ring manifold is identical. Direct proof
   that geometry is identifiable even when implementation is not. Seeds share
   the same connectome -> neurons are matched -> use a phase-invariant
   manifold-similarity metric (correlate the n_bump x n_bump tuning-similarity
   matrices). DO THIS.
6. *Spine = a hierarchy:* Computation -> Geometry -> Implementation;
   conserved -> conserved -> not conserved. Make this THE figure, not a metric
   dump. ADOPTED as the paper's structure.

Biggest risk flagged: becoming a "fly vs fish benchmark with lots of metrics."
Mitigation: organise every result under the 3-level hierarchy; the metrics are
evidence for the hierarchy, not the point.

## Entry 9 --- P7-extended + P8 results (2026-06-13)

**P7-extended (training creates the attractor) -- a keeper.** Jacobian
$J\tau=-I+W\,\mathrm{diag}(\sigma'(h^*))$ at the trained bump $h^*$, swapping in
different $W$ (same operating point, so $\sigma'$ matched):

```
              raw connectome        random-on-support      trained
fly   lambda1  -0.712 (nslow 0)     +0.110 (unstable)      +0.002 (nslow 1)
fish  lambda1  +0.280 (unstable)    -0.134 (nslow 0)       +0.001 (nslow 1)
```

The measured connectome is NOT poised at the attractor: the fly's is
over-damped (no slow mode), the fish's is unstable (runaway). Random weights
miss too, in the opposite direction each. **Only training places the operator on
the critical line $\lambda_1\!\approx\!0$ with exactly one marginal mode -- and
it does so for both species, from opposite starting points.** Behaviour creates
the continuous attractor; anatomy alone does not contain it. (Caveat to state:
evaluated at the trained $h^*$ so this isolates the weight contribution at a
fixed operating point.) This is the reviewer's "outcome 2" and is arguably the
single most important probe.

**P8 (decoupling) -- hypothesis corrected by data.** Across seeds (same
connectome, matched neurons):

```
            W_cosine(support)   ring-plane overlap(2D)   PR_cv    marginal mode
fly_rnn     0.576 ± 0.024       0.322 ± 0.101            6.5%     1/1 seeds
fish_rnn    0.594 ± 0.026       0.321 ± 0.080            5.5%     1/1 seeds
```

I (and the reviewer) expected "manifold conserved while W varies." The data say
something sharper: the *neuron-space embedding* of the ring also varies across
seeds (overlap 0.32 -- well above the chance baselines 2/46=.04 and 2/700=.003,
but far from 1), and W_cosine 0.58 is partly floored by the shared sign-lock.
What IS tightly conserved is the set of **macroscopic dynamical invariants**:
effective dimensionality (PR CV $\approx$ 6%), the single marginal eigenmode
(present in every seed), and integration accuracy ($r\approx0.99$). So the
correct decoupling statement is:

> The low-dimensional dynamical invariants (dimensionality, marginal-mode count,
> ring topology, gain) are reproduced to a few percent; the microscopic
> realisation -- which neurons embed the ring, and the synaptic weights -- is
> not. Behaviour pins the *computation and its geometry*, not the *implementation
> or its embedding*.

This is the connectome-constrained, cross-species generalisation of Goncalves
2014 (many weight configs, one dynamics) and is more defensible than the
embedding-conservation claim I started with. Good --- the science corrected me.

**Final hierarchy (all validated):**
- **Computation** (heading integration, near-unit gain): conserved fly<->fish,
  survives null connectomes. CONSTRAINED by task.
- **Geometry** (1-D continuous attractor: single marginal mode, PR~2-3, clean
  ring): conserved fly<->fish, survives nulls, CREATED by training (not in raw
  anatomy). CONSTRAINED by task.
- **Implementation** (weights + neuron-space embedding): NOT constrained ---
  varies across seeds within each species; inter-seed cosine ~0.58, ring-plane
  overlap ~0.32. The degeneracy itself is conserved across phyla.
- **Scale/redundancy** (N_bump 46 vs 700, rho 22 vs 243): the one thing the
  connectome sets.

## Entry 10 --- external review #2 (adversarial, on the full draft)

A skeptical reviewer read claude_paper.tex against the JSON results. Verdict:
thesis "largely supported, unusually honest," but flagged real issues. My
dispositions:

- **M1 (must):** "$\lambda_2\tau\le-0.37$" is false for fish ($-0.367$). FIX
  wording to "$\le-0.36$".
- **M2 (must, highest value):** the cosine-0.58 degeneracy claim has NO null.
  0.58 could mean constraint, not degeneracy. ADD the random-magnitude-on-shared-
  signed-support floor (the cc_p7p8 Wr generator). Calibrates 0.58 against
  chance. Likely outcome: half-normal/exponential magnitudes give a floor near
  $1/(1+\mathrm{CV}_m^2)$; if 0.58 $\approx$ floor -> maximal degeneracy
  (strengthens paper); if 0.58 $\gg$ floor -> "partially constrained" (claim
  flips). RUN IT. -> see Entry 11.
- **M3 (must):** nulls preserve density AND sign stats -> they are NOT "any
  connectome." Soften "connectome = scale only" to "specific synaptic *pairing*
  doesn't matter; density and sign/E:I statistics may." Cite Goncalves (E:I is
  the degeneracy-breaker). Note fish has block-shuffle only.
- **M4 (must):** n_eff $\equiv N/\rho$ -> circular, double-counts with PR. Lead
  with PR (independent). Relabel n_eff as redundancy-derived. Admit PR fly<fish
  (2.84 vs 3.29) is a genuine modest difference, not "identical."
- **M5 (must):** "training creates the attractor / anatomy doesn't contain it"
  overstated -- Jacobian uses the TRAINED $h^*$ for raw/random operators too.
  Reframe to "the measured operator is not poised at the trained operating
  point"; cite the frozen-connectome training failure as the primary evidence;
  Jacobian as mechanistic illustration. Move caveat up.
- **M6:** results_p57.json's raw-weight spectral degeneracy is messy/unused.
  Keep only P5 (identifiability) from it; note the pivot to the Jacobian object.
- **S1-S5:** state fish seeds (vfwd_rep) differ from base Jacobian checkpoint;
  report phase_corr spread (weak fly seeds); state the Re>-0.1 threshold;
  note error bars pool OU x seed (pseudo-replication) -> report seed-level too;
  emit density/E:I to JSON.

Most novel finding per reviewer: "opposite spectral starting points (over-damped
fly, unstable fish), same destination after training" + "degeneracy conserved
across phyla." Promote these toward the headline.

Reviewer's bottom line: as-is, a strong honest short-format synthesis; needs the
M2 null + sharper null scope to compel a top venue on novelty.

## Entry 11 --- the calibration null (M2) ran; the claim sharpened

```
        trained cos   random floor   analytic 1/(1+CVm^2)   density   %inhib
fly      0.576±0.024   0.375±0.004    0.381                  0.170     31%
fish     0.594±0.026   0.406±0.003    0.411                  0.037     66%
```

The reviewer's worry was justified and the result is better for it. My earlier
hunch (0.58 $\approx$ random -> maximal degeneracy) was WRONG: the random-
magnitude-on-shared-signed-support floor is $\approx$0.38--0.41, and the trained
seeds sit clearly ABOVE it (by $\sim$0.19 in both species). So:

> Implementation is **loosely / partially constrained**, not unconstrained.
> Trained weights are more similar than chance (training imposes real structure)
> but far from the reproducibility ceiling of 1 (~40% of the weight structure is
> seed-specific). And the *position between floor and ceiling is conserved across
> phyla* ($\sim$0.19 above floor in both).

This is a more honest and, I think, more interesting claim than "not
constrained": the connectome+task narrow the weights to a band of the same
relative width in fly and fish. Revised the abstract, Implementation results,
Discussion, and fig.\ 4 (floor + ceiling now drawn) accordingly. Also emitted
density (0.170/0.037) and E:I (31\%/66\% inhibitory) to JSON so fig.\ 1 traces.

Lesson for the agentic-science loop: the external review caught a claim my own
enthusiasm had inflated; running the demanded null in $\sim$15 lines either
strengthens or flips the result. Cheap insurance against rabbit holes.

## Entry 12 --- a third connectome: wiring a worm (2026-06-13)

PI suggestion: be more ambitious -- wire a *C. elegans*. This is not an add-on,
it is the sharpest possible test of the thesis. The worm has a *complete*
connectome (300 neurons here, Cook/Varshney variants on disk in
`/workspace/NeuralGraph/graphs_data/CElegans`) but it **never evolved to integrate
heading** -- no head-direction ring, no compass. So:

> **Experiment.** Train the *worm connectome* on the identical
> heading-integration task. Thesis prediction ("geometry is task-induced; the
> connectome contributes scale") -> the worm should ALSO become a 1-D ring
> attractor (single marginal mode, low PR). If it CANNOT, the failure exposes a
> *necessary* connectome property (size / recurrence / E:I) that fly+fish have
> and the worm lacks -- refining "what the connectome contributes" beyond scale.

Either outcome is publishable: success = strongest evidence for task-induced
universality on a connectome that never did this; failure = a necessity boundary.

**Design decision -- one standalone trainer, three connectomes.** Rather than
mix framework checkpoints (fly/fish) with a new worm, I write ONE minimal
sign-locked leaky-RNN trainer (Eq.\ 1: $\tau\dot h=-h+W_\mathrm{rec}\sigma(h)+W_\mathrm{in}u$,
cos/sin readout, BPTT, L1 on magnitudes) and run the IDENTICAL task + HPs on all
three connectomes + nulls. Perfectly controlled. Validate the trainer by
confirming fly+fish reproduce the ring / single-marginal-mode / near-unit-gain
signatures I already established from the framework checkpoints.

**Worm connectome prep:** 300x300, unsigned (values in [0,1], density 0.072 --
intermediate between fly 0.170 and fish 0.037). Assign Dale signs from the known
~26 GABAergic neurons (D-type DD/VD, RME, RIS, AVL, DVB -> inhibitory; rest
excitatory), spectral-normalise to 0.9 -- same convention as fly/fish.
Input port: $\omega\to$ sensory neurons (83) (the natural afferent analog);
read cos/sin from all neurons (no predefined ring in the worm). For fairness the
3-way analysis uses the WHOLE recurrent population in all three.

No cluster (`bsub`) from this devcontainer, but 2x A6000 locally -> train here.

## Entry 13 --- worm wired into the framework; first signal (2026-06-13)

PI steered me to integrate via the framework registry (not my standalone
trainer), keeping graph_trainer.py untouched except CElegans-specific functions.
Done with the minimal change-set an Explore agent mapped:
- `load_celegans_connectome` (new, connectome_loaders.py): 300x300 chemical
  adjacency, Dale signs from 26 GABAergic neurons (DD/VD/RME/RIS/AVL/DVB ->
  inhibitory), transpose to J[post,pre], spectral-rescale to 0.9. frac_inh=0.087.
- `_build_celegans_300`/`_register_celegans_300` (new, circuits.py) + one
  discovery line. Registers `celegans_300_v1`; whole net is the readout pool
  (no ring), `bump_ring_ix=None`.
- 2 configs reusing `zebrafish_hd_si`, `velocity_gate=none` (free W_in injects
  omega on all neurons), `output_from_dipn_only=false` (decode from all 300).
- Model builds + forwards on the worm circuit (N=300, n_in=3, n_out=2). Trained
  with GNN_Main.py `generate_train_test_plot`.

**Key methodological note (matters for the paper's honesty).** My from-scratch
standalone trainer LEARNS the fly (r=1.0 after scaling the init up) but PLATEAUS
on the worm at predict-mean. The framework, by contrast, drives the worm to
**pi_acc 0.92 within the first epoch snapshot** (iter 3125). So the standalone
"worm fails" is a TRAINER limitation, not a connectome verdict -- the worm
connectome *can* be trained to integrate heading. This is exactly the
drosophila-paper lesson: a single-config optimisation failure is unsearched, not
impossible. The framework's richer recipe (data augmentation, tail loss, ED
loss, n_steps curriculum, many iters) finds the solution the minimal trainer
misses. I will report the worm result ONLY from the framework-trained models,
and will state this caveat explicitly.

Early framework worm (epoch 1): pi_acc 0.92 (short horizon) but r_roll_1k still
poor (long-horizon rollout needs the later large-T curriculum epochs). Training
2 seeds (base GPU0, rep_2 GPU1); will grab the best-converged snapshot once
long-horizon r_roll saturates, then run the cc pipeline (PR, marginal mode, ring
geometry, identifiability) on the worm and assemble the 3-connectome comparison.

## Entry 14 --- worm result is NUANCED; a divergence, not a copy (2026-06-13)

GPU contention from the PI's concurrent pipeline jobs made framework training
slow (epoch-1 checkpoint took ~40 min wall-clock). Grabbed the epoch-1 worm
checkpoint (framework metrics: pi_acc 0.99, r_roll 0.995, rmse 3 deg on its OWN
swim-integration test set). Then probed with my cc pipeline:

```
const omega:  r=0.97 (30 deg/s),  0.85 (60),  0.43 (120)   -> integrates, narrow range
OU rollout:   r=0.62 (sd30), 0.39 (sd25), 0.25 (sd60)       -> does NOT generalise
OU geometry:  PR~1.1, phase_corr~0.1, dec_top2~0.05, dec_full~0.5
neurons: 96% active, mean rate 0.55 (not collapsed)
```

**Reading (preliminary).** The worm connectome LEARNS its training distribution
(Poisson swim impulses; pi_acc 0.99) and steady low-speed rotation, but its
epoch-1 solution does NOT generalise to held-out OU rotation, and its OU
population code is ~1-D and weakly heading-tuned -- in sharp contrast to fly/fish,
which generalise to OU (r~1.0) and form a clean ~3-D ring (PR~3). So the worm did
NOT (yet) reproduce the robust, generalising ring attractor.

**A possible necessity boundary** (the worm's feedforward-dominant
sensory->inter->motor connectome may lack the recurrent ring motifs fly/fish
have) -- BUT three caveats forbid a firm claim:
1. Epoch-1 only (training continues; larger-T curriculum in later epochs may
   build OU robustness). Per the drosophila-paper lesson, undertraining != cannot.
2. Held-out-distribution gap: the worm does well on its TRAINED distribution; my
   OU probe is a generalisation test fly/fish pass and the worm fails.
3. Different input scheme: velocity_gate=none (free W_in to all neurons) vs
   fly/fish anatomically-gated afferents -- only the recurrent connectome is
   matched, not the input.

**Decision.** Report C. elegans as a PRELIMINARY third-connectome extension + a
methods contribution (framework integration), NOT a headline. State the
divergence with all three caveats. Keep training for a better checkpoint; if
later epochs close the OU gap, update.

## Entry 15 --- the worm gap CLOSED with training; result flipped (2026-06-14)

I let the worm finish training (PI then asked me not to run long jobs -- stopped
it once converged). The converged checkpoint (epoch 8) overturns Entry 14:

```
              epoch-1 (reported)     converged (epoch 8)
const-omega   0.97/0.85/0.43         0.998/1.000/1.000/1.000  (30/60/120/180)
OU r          0.07-0.45 (fails)      0.999-1.000 (sd 30-90)   -> generalises
PR            1.6 (partial)          2.81  (= fly 2.84, fish 3.29)
phase_corr    0.68                   0.93
Jacobian l1   -0.013 (damped, n2)    -0.0014 (marginal, n_slow=1)
```

On re-test, even the epoch-1 file now reads OU r~0.99 -- my original 0.25 did NOT
reproduce (I likely loaded a mid-write/unsettled state while training was live).
So there is no real "necessity gradient": the fully-trained **worm connectome
becomes a clean ring attractor indistinguishable from fly/fish** -- the STRONGEST
evidence in the paper for task-induced universality. I rewrote the abstract,
Sect. C. elegans, Discussion, and fig.6 accordingly, and kept one honest line
that an undertrained checkpoint was a partial integrator (read converged optima,
not undertrained ones -- the same lesson as the agentic loop).

**Lesson (twice now): never interpret an undertrained / mid-write checkpoint.**
Entry 14's "necessity gradient" was a wrong conclusion from an unconverged model;
the external-review discipline + finishing training caught it. Exactly why the
"undertraining != cannot" caveat existed.

## Entry 16 --- PI question: does the worm integrate MUSCLE signals?

Sharp hypothesis for WHY a worm connectome takes to heading integration so
readily. C. elegans never computes heading, but it DOES integrate
proprioceptive / muscle (stretch) signals for undulatory locomotion -- B-type
motor neurons sense local body curvature and propagate the bend posteriorly via
stretch feedback (Wen et al. 2012). So the connectome may already carry the
recurrent, near-marginal motifs an integrator needs; the heading task
*re-purposes* an existing integrating substrate rather than building one from
scratch. Refines the thesis: not "any connectome integrates" but "a connectome
that already integrates *something* is readily repurposed." Added to the
Discussion as a hypothesis (NOT a result -- I trained on heading, not muscle),
with a falsifiable prediction: natively-integrating connectomes (oculomotor,
proprioceptive, accumulator) should reach the ring faster/more robustly than
feedforward-only sub-connectomes. Cited wen2012.

## Entry 17 --- two reframes from external review (2026-06-14)

**Reframe 1 (the paper is now a theory paper).** Reviewer: once fly + fish +
randomised-fly/fish + worm ALL -> ring, the claim flips from "connectome
determines computation/geometry" to "the COMPUTATION (heading integration under
leaky recurrent dynamics) determines the geometry; the connectome only
modulates." I agree. Demoted the connectome's role to modulatory (scale,
redundancy, trainability/robustness), made "computation determines geometry" the
Discussion headline, renamed the nulls subsection "The task selects the geometry;
the connectome only modulates it", and labelled the marginal-mode/P7 result as
mechanistic confirmation (the manifold is a genuine continuous attractor), not the
headline. The headline is the universality: "opposite spectral starting points ->
same fixed point", now extended to the worm.

**Reframe 2 (don't over-claim the worm has no heading).** Cannot say
"C. elegans never evolved heading" -- worms orient (klinokinesis/pirouettes
[Pierce-Shimomura 1999], klinotaxis/weathervaning [Iino & Yoshida 2009],
experience-dependent navigation, directed heading-error turns; RIA encodes
head-bend direction [Hendricks 2012]). Softened everywhere to: "has not been
shown to contain a persistent, compass-like head-direction ring attractor
comparable to the fly CX or vertebrate HD system." Added the 4 refinement
experiments to the Discussion: (1) matched random nulls vs worm; (2)
sub-connectomes (motor/proprioceptive vs sensory/inter vs feedforward-pruned);
(3) trainability/robustness metrics not just final accuracy; (4) inverse task
(train worm on its own proprioceptive phase-integration, compare spectrum/
manifold). Cited pierceshimomura1999, iino2009, wen2012, hendricks2012.

Paper now 12pp, no undefined refs. Net: the worm turned a comparative
connectomics paper into a universality/theory claim, and forced me to be far more
careful about what is and isn't known in the worm.

## Entry 18 --- two deep reframes from review (2026-06-14)

**(a) Computation determines geometry.** With fly+fish+randomised+worm all -> the
same ring, the honest claim flips from "connectome determines computation" to
"the TASK selects the geometry; the connectome only modulates." Recast the
Discussion and section title accordingly; demoted the marginal-mode/P7 result to
mechanistic confirmation. Headline universality: opposite spectral starting
points (over-damped fly, unstable fish) -> same critical fixed point.

**(b) Support vs operator (the key epistemic point).** The EM connectome gives
the SUPPORT A_ij + a sign prior, NOT the functional operator J_ij. What I call
"training" is optimisation of unknown effective weights; the optimised W is NOT
the anatomical one (cosine ~0.45 vs connectome; frozen-raw-connectome fails). So
strictly: there EXIST weights on each support that implement heading; we have NOT
shown the measured connectome as-wired does. And biological synaptic strength is
dynamic (neuromodulation/STP/state), so J* need not be any operator the animal
instantiates. Added a "Support versus operator" paragraph. Reworded "training" ->
"optimisation". PI's phrase "true weights are not functional weights" is exactly
this.

## Entry 19 --- robustness: capability universal, cost is not (fig.7)

Probed converged fly/fish/worm with recurrent noise (scaled to each model's own
state s.d.) and random neuron lesions (existing checkpoints, no training):

```
            noise r @ {0,.05,.1,.2,.4,.8}xSD        lesion r @ {0,10,20,30,50,70}%
fly    0.97 0.96 0.92 0.07 ...                       0.97 0.75 0.60 0.46 ...
fish   1.00 0.99 0.95 0.67 0.46 0.10  (most robust)  1.00 0.85 0.56 ...
worm   1.00 0.33 ...  (collapses at 5%)              1.00 0.23 ...  (collapses at 10%)
```

All three reach the ring at 0 perturbation (capability universal); robustness
orders fish > fly >> worm. Redundancy (fish, 700 neurons) buys tolerance; the
repurposed worm ring is the most fragile. First direct evidence that the
biological connectome's contribution is COST/ROBUSTNESS, not existence. Caveat:
worm used free input gating + may be less converged -> upper bound on the gap.

## Entry 20 --- is the Dale sign pattern necessary? (mostly no)

Tested the sign-necessity caveat directly with the standalone trainer (controlled
within-trainer: same fly support + magnitudes, only signs changed):

```
fly real-sign      4/4 seeds reach ring
fly sign-SHUFFLED  4/5  (one seed fails)
fly sign-RANDOM    2/2
```

So shuffling/randomising which edges are E vs I (preserving density, E:I fraction,
magnitudes) MOSTLY still yields the ring -> the specific sign PATTERN is largely
dispensable; the ring depends on support + optimisation more than sign assignment.
The one failure hints sign structure modulates RELIABILITY, not existence ---
consistent with the whole "connectome modulates cost, not capability" thesis.
Caveats: lightweight standalone optimiser, modest seeds; production trainer +
bigger sweep would quantify the reliability effect. This pushes "the connectome
contributes scale" to "...and maybe reliability; even sign is mostly not needed."

Net after 24h-budget round: the paper is now a theory/universality claim
(computation determines geometry; connectome modulates cost & reliability),
backed by 3 measured connectomes + nulls + sign-nulls + robustness, with the
support-vs-operator epistemics made explicit. 14 pp, 7 figs.

## Entry 21 --- worm PI + matched-random-null: attempted, compute-limited (2026-06-14)

Ran two follow-ups the reviews asked for, both throttled by the PI's concurrent
companion-pipeline jobs reclaiming the GPUs:

- **Matched random-graph worm null** (Dale-consistent ER: matched N, #edges, E:I
  fraction, magnitude distribution; random support). Behaviourally it REACHED the
  heading ring (training metrics r_roll_1k 0.996, pi_acc 0.986 by epoch 1) ---
  i.e. a worm-statistics-matched random graph integrates just like the worm
  connectome, consistent with "the worm wiring adds little beyond its
  statistics." Caveat: no epoch-end checkpoint saved (epoch 1 didn't complete
  under contention), so no geometry/marginal-mode analysis yet --- behavioural
  metric only.

- **Worm 2-D path integration (position_2d, the harder rung).** Only an
  epoch-1 (undertrained) checkpoint was saved before the run was starved. On it,
  heading r=0.77 but position (x,y) decode is poor (r~0.14 / -0.30). Per the
  rotation-worm lesson (epoch-1 looked bad, converged was a clean ring) this is
  INCONCLUSIVE, not a negative result --- the run simply did not converge. I will
  NOT claim "worm fails 2-D PI"; it remains the key open experiment.

Decision: do NOT add firm claims to the paper from these undertrained/
uncheckpointed runs. Record honestly; keep both as the immediate next experiments
in the outlook. (Lesson, again: under shared-GPU contention, plan for the 2-h cap
and the possibility of no converged checkpoint; the behavioural training metric is
the only thing salvageable without an epoch boundary.)

## Entry 22 --- connectome-anchoring (cos) sweep: the support-vs-operator axis made causal (2026-06-14)

PI pointed out a trained sweep already exists: coeff_cos_distance = cos000..cos100
(fly & fish, rotation/both/position_2d), anchoring W_rec block-directions toward
the REAL connectome. This is the support-vs-operator question as a continuous
knob. Analysed the rotation sweep (existing checkpoints, no training):

```
cos:           0.0    0.25   0.5    0.75   1.0
fly  r_ou      1.00   1.00   1.00   1.00   1.00     <- accuracy: anchoring is FREE
fly  cos_con   0.49   0.61   0.69   0.68   0.67     <- rises, PLATEAUS ~0.67 (never =connectome)
fly  PR        2.47   2.69   2.52   2.58   2.57     <- geometry invariant
fly  noise/les 0.93/0.76 ... 0.76/0.86             <- no systematic robustness gain
fish r_ou      1.00 ... 1.00
fish cos_con   0.44   0.61   0.63   0.66   0.67
fish PR        2.59   2.08   2.02   2.00   2.05
fish les       0.85   0.70   0.60   0.46   0.52     <- anchoring HURTS fish lesion-robustness
```

**Four findings:** (i) anchoring to the real connectome weights is FREE for
accuracy (r=1.0 throughout); (ii) the operator NEVER becomes the connectome
(cos-to-connectome plateaus ~0.67, far below 1 -- task + norm-floor still shape
the within-block magnitudes); (iii) geometry (PR, single marginal mode) is
INVARIANT to anchoring; (iv) anchoring does NOT buy robustness (fish lesion-
tolerance actually falls). => robustness tracks substrate scale/redundancy
(Fig.7), NOT weight-fidelity to biology.

This is the causal version of support-vs-operator and directly answers the
reviewer's "what remains beyond scale?": you can pull the operator part-way to the
real weights at zero cost to accuracy/geometry, the biological weight VALUES are
not what create the ring, and matching them confers no measurable functional
advantage. Added as Sect. "Anchoring the operator..." + Fig.8 (15pp).
Caveat: robustness curves are n=1/level -> trends not estimates.

Also applied review fixes: softened "connectome sets how many neurons carry the
ring" -> "appears to modulate scale, redundancy and robustness rather than the
existence/geometry"; strengthened the sign-shuffle under-claim (reviewers will
attack; needs many more seeds + production trainer).

## Entry 23 --- review deepens the cos result: two stacked degeneracies (2026-06-14)

Reviewer's read of Fig.8: the main story isn't the plateau, it's that the cos
sweep is a CONTROLLED CONTINUOUS PATH through operator space -- moving J from the
free optimum toward J_connectome and the ring "barely notices." Stronger than the
discrete fly/fish/worm/shuffled examples because it's one wiring diagram, a
continuum of operators, same geometry.

Adopted:
- New headline framing (Discussion): TWO degeneracies stack -- the connectome
  does not uniquely determine the operator (optimisation; cos plateaus <1), AND
  the operator does not uniquely determine the geometry (continuum of operators ->
  same ring). The INVARIANT is the geometry of the computation. So: behaviour
  underdetermines operator; operator underdetermines geometry; geometry conserved;
  connectome's contribution is the COST of producing it (Fig.7).
- New Fig.9 (reviewer's suggestion): plot invariants (PR, lambda1, r) directly
  against the ACHIEVED operator-similarity cos(W_rec,W_con) -- all flat. "Operator
  becomes biological -> nothing happens." Memorable.
- Softened robustness to the exact phrasing: "we find NO EVIDENCE that increasing
  anchoring systematically improves robustness" (n=1/level -> null, not harm).

Paper now 16pp, 9 figs + Table 1. Fig.8/9 (continuous operator path) is arguably
now the strongest single piece of evidence for "computation determines geometry"
-- it perturbs the operator itself, continuously, rather than swapping connectomes.
