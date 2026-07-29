---
title: "Research notebook II --- is a connectome a learning prior?"
author: "Claude (Opus 4.8), agentic post-doc"
date: "June 2026"
---

# Research notebook II

*Second paper. The first paper ("what a connectome determines") ended at a
near-tautology: heading integration on S1 forces a ring attractor, so "connectome
-> ring" carries no information about biology. This notebook chases the
non-tautological residual.*

---

## Entry 0 --- the thesis and why it escapes the tautology (2026-06-14)

**Principle.** The asymptotic solution geometry is fixed by the task's SYMMETRY,
not by the connectome (heading lives on S1 -> any good integrator is a continuous
ring attractor). So a non-tautological paper must measure a quantity the symmetry
leaves FREE and ask whether the connectome carries information about it. Free
axes: (i) learnability, (ii) precision/drift, (iii) capacity, (iv) off-manifold
dynamics.

**Thesis (route 1, committed).** *A connectome is a learning prior, not a circuit
diagram of a solution.* With unlimited data + a strong optimiser every substrate
converges to the same ring (the tautology). Brains learn online, from little
data, with noise. In that regime the connectome is an INDUCTIVE BIAS that sets
sample-efficiency, reliability, generalisation, and precision (drift). The strong
claim: a biological connectome is a better prior for heading integration than its
own degree/sign/density-matched random control.

**Key instrument insight.** Use a deliberately LIGHTWEIGHT optimiser (the
standalone sign-locked RNN trainer), not the production trainer. A strong
optimiser washes out the prior (everything converges); a weak one reveals it.
My earlier "underpowered trainer" nuisance (it converged fly fast but plateaued
on the worm) is, reframed, exactly the measurement: the connectome modulates
learnability, visible only when the optimiser is not all-powerful.

**MVP --- controlled ablation of the connectome as prior.** Same trainer, same
task (heading), vary ONLY the wiring across a ladder that strips progressively
more real structure:
- `fly`         : real support + real signs
- `fly_blockshuf`: magnitudes+signs permuted across the real support (keeps
  degree + magnitude distribution; destroys the pairing)
- `fly_signshuf` : real support+magnitudes, signs permuted (destroys sign pattern)
- `fly_er`       : random support at matched density + sign fraction, magnitudes
  resampled (destroys topology)
Metrics, many seeds each: (a) data-to-criterion (iters to rollout r>0.9),
(b) seed reliability (fraction converging in budget), (c) drift of the converged
integrator (heading diffusion under zero input + noise -- the precision the
S1 topology leaves free). The ORDERING across the ladder says which connectome
property, if any, is the useful prior.

Prediction to test (could go either way -- both publishable):
- If real > blockshuf > signshuf > er in learnability/drift -> the connectome is
  a structured prior; we can say WHICH structure matters.
- If all equal -> the connectome is not a learning prior for this task either,
  and the "task fixes everything" story extends from the solution to the search.

## Entry 1 --- MVP result: route 1 dead, route 2 alive (2026-06-14)

Ablation (fly real / blockshuf / signshuf / er-matched-random), standalone
lightweight trainer, 8 seeds each, 2200 iters. All 8/8 converge in every
condition (r=1.0) and all form the ring.

```
condition       median d2c   median drift
fly (real)        750          0.0119
fly_blockshuf     600          0.0370
fly_signshuf      750          0.0277
fly_er            600          0.0354
```

**Route 1 (learning prior): NOT supported.** Data-to-criterion shows NO real-vs-
control advantage (Mann-Whitney p=0.15/0.37/0.11; controls trend slightly
FASTER). The biological connectome is not a better learning prior for heading --
every wiring learns the ring equally fast. Clean negative; report it as such.

**Route 2 (precision/drift): SUPPORTED, significant.** Among integrators that all
solve the task and all form the ring, the REAL connectome has ~2.5-3x LOWER drift
(heading diffusion under zero-drive + noise) than every control:
fly vs blockshuf p=0.0009, vs signshuf p=0.033, vs er p=0.0003. And it's NOT
because it learns faster (it doesn't). 

**Pivot.** Paper II = route 2: *the connectome is tuned for PRECISION, not
geometry, learnability, or capability.* Task symmetry fixes the ring; every
wiring learns it equally; the biological connectome's distinctive contribution is
the drift rate -- the behaviourally critical free parameter the S1 topology leaves
open (HD cells drift little; evolution tunes for low drift). This is the
non-tautological biological result.

Strengthen next: (a) measure drift on the FRAMEWORK (production-trainer)
checkpoints -- real fly/fish vs their trained er/bs nulls -- a second, independent
validation on converged models I already have; (b) replicate the ablation on fish
(and worm if standalone converges it); (c) relate drift to a connectome property.

## Entry 2 --- framework validation tempers the story (HONEST) (2026-06-14)

Measured drift on PRODUCTION-trained checkpoints (real vs framework-trained
nulls). It does NOT fully reproduce the clean standalone result -- good, this is
the honest version:

```
                median drift   real<null p
fly_real        0.0124
fly_er  (random topology) 0.0471   p=0.015  <- real MORE precise than ER [ok]
fly_bs  (degree+weights kept) 0.0098 p=0.79 <- real NOT better than block-shuffle
fish_real       0.0027
fish_bs         0.0026         p=0.58        <- no difference
fish << fly drift (5x): redundant substrate -> more precise (links to Paper I scale)
```

**Reconciliation.** Standalone (weak optimiser): real < blockshuf < er (pairing
helps). Framework (strong optimiser): real ~= blockshuf < er (only topology/degree
matters). So:
- ROBUST across both trainers: real connectome > fully-random ER on precision.
- TRAINER-DEPENDENT: real vs degree-preserving shuffle (standalone: real better;
  framework: equal).

**Honest thesis (revised).** Precision (drift) is a FREE axis the task does not
fix; among integrators that all solve the task and all form the ring, drift
varies with the wiring. It is carried primarily by the connectome's DEGREE/WEIGHT
STATISTICS (real ~= degree-preserving shuffle >> random ER), not the specific
synaptic pairing -- which adds precision only under limited optimisation.
Substrate SCALE sets the precision floor (fish 5x more precise than fly). This
dovetails with Paper I (statistics not pairing; scale matters) and is the
non-tautological residual: the task fixes the geometry, the connectome's
second-order statistics + scale fix the precision.

Overnight: fly-extra seeds (->20) + fish standalone ablation running to tighten
the numbers; then build figures + write claude_paper2.

## Entry 3 --- honest conclusion: route 2 also rejected; scale is what's left (2026-06-14)

Full framework-drift across the rich model bank (real reps + bs + er + er_ei +
cos sweep, 3 species):

```
fly_real 0.0124 | fly_er 0.047 (p=.015 real<er) | fly_bs 0.0098 (p=.79) | fly_er_ei 0.0066 (real WORSE)
fish_real 0.0033 | fish_bs 0.0024 (p=.75)
worm_real 0.065
cos-anchoring -> drift: FLAT/noisy both species (anchoring real weights doesn't lower drift)
cross-species: fish 0.003 << fly 0.012 << worm 0.065  (tracks redundancy/scale)
```

**The "connectome tunes precision" hypothesis is REJECTED on production models.**
Structure-preserving random graphs (block-shuffle, er_ei) match or BEAT the real
connectome; anchoring toward the real weights doesn't help. The clean standalone
"real < all nulls" was an artifact of my looser standalone null construction --
the more carefully a null's statistics (degree, E:I) are matched, the more
completely it reproduces the real integrator's precision. Only fully-random ER
(which destroys degree+structure) is less precise.

**Honest Paper II thesis.** The task fixes the geometry (Paper I tautology); the
remaining FREE axes -- learnability AND precision -- are ALSO not connectome-
specific. They are governed by coarse network statistics (degree, E:I) and
substrate SCALE/redundancy (fish<<fly<<worm), not by the biological wiring or
weights. This EXTENDS Paper I ("scale not structure") from geometry to precision:
the connectome's only robust contribution to a heading integrator is its size.

Less exciting than "connectome tunes precision", but it is what the data say.
Report it honestly. The one clean POSITIVE law: precision tracks substrate
redundancy. Paper II = "Integration precision is set by substrate scale, not
connectivity structure."

## Entry 4 -- MERGE: one paper, the H1-H5 falsification ladder (2026-06-14)

PI + reviewers converged: combine Paper I and II into ONE paper. Paper II is not
an independent discovery -- it is the next stage of the same argument. Frame as
progressively stronger null hypotheses, each removing a candidate role for the
connectome:

```
H1 implementation? NO  (inter-seed cosine 0.58, loose; conserved across phyla)
H2 geometry?       NO  (fly/fish/worm/nulls/cos-sweep all -> ring; created by training)
H3 learnability?   NO  (matched controls learn equally fast)
H4 precision?      mostly NO (real ~ matched nulls; anchoring no help; spurious-advantage caution)
H5 scale/redundancy? YES (drift ~ 1/n_tuned across 38 models, rho=-0.66)
```

Conceptual headline: THE CONNECTOME IS NOT A BLUEPRINT. Computation + geometry =
task symmetry + optimisation; connectome = substrate scale. A radical reading of
a wiring diagram, consistent across 2 phyla + worm + nulls + continuous operator
interpolation.

Mechanism confirmation (cc_drift_mech.py, 38 models): drift vs n_tuned rho=-0.66
p<1e-4 (noise-averaging redundancy law); drift uncorrelated with PR/gap/lambda1.
Honest caveats: within-species drift varies several-fold at fixed n (solution-
dependent), worm off-trend -> the law is coarse/cross-species.

Wrote claude_paper_merged.tex (9pp, H-ladder, "connectome is not a blueprint").
Outlook = the real sequel: harder tasks (metric path integration x,y,theta) where
task symmetry leaves more free, and the connectome might finally matter -- the
prediction is sharp (heading: every connectome; path-integration: only some?).

This is the right structure. Paper I alone left "so what DOES the connectome do?"
unanswered; Paper II alone lacked the degeneracy setup. Together: a systematic
falsification of connectome-as-blueprint. Standalone overnight jobs (fly 20-seed +
fish ablation) still running to firm the H3/H4 standalone numbers; not required for
the merged manuscript.

## Entry 5 -- external review pass on merged paper -> revisions (2026-06-14)

Adversarial review: GO as one paper, minor-moderate revision. Fixes applied:
- H4 p-values were hand-transcribed and WRONG. Regenerated from results_drift.json,
  switched to TWO-SIDED Mann-Whitney (honest "is there a difference"):
  real vs er p=0.03 (real lower), vs bs p=0.54 (no diff), vs er_ei p=0.05 (er_ei
  BEATS real), fish vs bs p=0.56. Rewrote H4: "no connectome-specific ADVANTAGE",
  not "wiring harms precision". n=5-6 noisy -> stated in result text.
- H1 relabelled "not UNIQUELY determined / only loosely constrained" (cosine 0.58
  is ABOVE the 0.39 floor -> partial constraint, not zero). Table + section.
- H5 softened: "redundancy LAW" -> "cross-substrate scaling TREND consistent with
  noise-averaging". Stated it's ~3 clusters, fly-vs-fish confounded with
  substrate/density/Dale/trainset; within-fly drift varies 28x at fixed n.
- Title scoped: "The connectome is not a blueprint OF A HEADING INTEGRATOR: ...
  the wiring fixes mainly scale" (keep provocative line; scope to the task).
- H3: noted the lightweight-optimiser caveat applies -> read as "no advantage".

Attempted the reviewer's within-substrate n-sweep to break the H5 species confound:
lesion a fraction of the fish ring, drift vs surviving n. RESULT: lesioning breaks
integration (integ_r 1.0 -> 0.6/0.3/0.0), so drift becomes meaningless. The clean
test needs RETRAINING a subsampled ring (training) -- the paper names this as the
natural next test. Confound honestly stands. (results_nsweep.json kept for record.)

Merged paper now 9pp, all numbers verified vs JSON, GO. The one positive claim
(H5) is the softest and properly hedged; H2 (geometry/degeneracy) is the strong,
replicated core.
