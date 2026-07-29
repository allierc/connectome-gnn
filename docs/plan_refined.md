# Refined plan — consolidating all external reviews

*Paper: "What a connectome determines …" ([claude_paper.tex](claude_paper.tex)).
This file folds every external review received during the project into one
prioritised roadmap: the thesis as it now stands, each review point with its
status, and the forward experimental program.*

---

## 1. Thesis (as reshaped by the reviews)

Original framing → final framing, after the reviews:

- v0: "connectome determines computation/geometry" (comparative connectomics).
- **final: the heading-integration TASK, under leaky sign-locked recurrent
  dynamics, DETERMINES the low-dimensional ring geometry; the CONNECTOME supplies
  only the support + a sign prior + scale, and modulates cost (substrate,
  redundancy, robustness, reliability) — not the existence of the attractor.**
  The object that generates the ring is the effective operator J\* found by
  *optimisation*, which is not the anatomical weight matrix and need not be any
  operator the animal instantiates (support-vs-operator).

Evidence chain (all in the paper): fly+fish conserved computation & geometry →
loosely-constrained implementation (cosine 0.58 vs chance floor 0.39) →
null connectomes reproduce the ring → C. elegans reproduces the ring →
noise/lesion robustness differs (worm fragile) → sign-shuffle mostly preserves
the ring.

---

## 2. Review-by-review ledger (point → status)

### Review #1 (structure / conservation)
| point | status |
|---|---|
| Lead with conservation, not redundancy | DONE |
| Identifiability is the headline ("behaviour fails to determine implementation in both phyla") | DONE |
| "Same computation" nuance (fish carries extra vfwd channel) | DONE (stated) |
| P7 reframed: does TRAINING create the marginal mode? (raw vs trained Jacobian) | DONE (fig3e) |
| P8 geometry/implementation decoupling | DONE (data corrected the hypothesis; reported honestly) |
| Spine = Computation → Geometry → Implementation hierarchy | DONE (fig4c) |

### Review #2 (adversarial, M1–M6)
| point | status |
|---|---|
| M1 λ₂τ ≤ −0.37 false for fish | FIXED (≤ −0.36) |
| M2 cosine-0.58 needs chance-floor + ceiling | DONE (results_calib; floor 0.38–0.41) |
| M3 null scope overstated (nulls preserve density+sign) | FIXED (scope softened; Goncalves E:I caveat) |
| M4 n_eff = N/ρ circular | FIXED (PR is the independent measure; n_eff demoted) |
| M5 "training creates attractor" overstated (J at trained h\*) | FIXED (reframed + frozen-connectome cited) |
| M6 messy raw-weight spectral file unused | OK (not relied on) |

### Review #3 ("P7 secondary; computation determines geometry")
| point | status |
|---|---|
| Demote P7/marginal-mode to mechanistic confirmation | DONE |
| Headline = universality: opposite spectral starts → same fixed point | DONE |
| "connectome contributes scale" may be too strong → demote to modulatory | DONE |
| Strengthen the null→worm transition (the conceptual crossing) | DONE |

### Review #4 (support vs operator)
| point | status |
|---|---|
| Connectome = support A_ij + sign prior, NOT operator J_ij | DONE ("Support versus operator" paragraph) |
| "training" → "optimisation of effective weights" | DONE |
| W_trained ≠ W_connectome; true weights are non-functional (dynamic) | DONE |
| Robustness/cost is where evolution may act ("brutal test") | DONE (fig7) + remaining rungs listed |

### Review #5 (heading easy / displacement harder; cost not capability)
| point | status |
|---|---|
| "A heading integrator is not hard to synthesise" | DONE (framed) |
| Cost ladder: substrate/redundancy/precision/robustness/basin/speed | DONE (Discussion) |
| Computational ladder: heading → +ω horizon → body-frame translation → full PI | DONE (stated) |
| Run the matched cost/robustness comparison | PARTIAL (noise+lesion done; convergence-speed/seed-rate partial via sign test) |

### Review #6 (C. elegans — don't over-claim "no heading")
| point | status |
|---|---|
| Soften to "no known compass-like HD ring attractor" | DONE (everywhere) |
| Note worm DOES orient (klinotaxis/pirouettes/weathervaning, RIA) | DONE (+ cites) |
| 4 refinement experiments | LISTED in Discussion; (1) worm path-integration IN PROGRESS |
| Literature (Pierce-Shimomura, Iino&Yoshida, Wen, Hendricks) | DONE (cited) |

### Final verification reviews
- All numbers PASS against JSON; abstract support-vs-operator qualifier added; GO.

---

## 3. Forward experimental program (prioritised)

The reviews converge on one decisive axis: **once every connectome reaches the
ring, the interesting variables are COST, ROBUSTNESS, RELIABILITY, and the
HARDER computations.** Climb this ladder:

**A. Cost/robustness comparison (the "brutal test")** — *partly done.*
1. Noise + lesion tolerance, fly/fish/worm. ✅ (fig7; fish>fly≫worm)
2. Convergence speed (epochs-to-ring) + seed success rate across connectomes.
   *Partial:* sign-shuffle gave a first seed-reliability number; needs a matched
   fly/fish/worm sweep with the production trainer.
3. Perturbation/marginal-mode stability under weight jitter. *TODO.*

**B. Sign / support necessity** — *partly done.*
4. Sign-shuffle & sign-random nulls (fly): mostly preserve ring (4/5, 2/2). ✅
5. Matched degree/sign/density random graphs vs the worm connectome (worm null):
   does the worm wiring add anything beyond statistics? *TODO (short training).*
6. Sub-connectomes: motor/proprioceptive vs sensory/inter vs feedforward-pruned —
   does the proprioceptive subgraph reach the ring faster/more robustly?
   *TODO* (tests the "re-purposed native integration" hypothesis).

**C. The computational ladder (heading easy → PI hard)** — *in progress.*
7. Heading only (1-D ring). ✅ all connectomes.
8. Heading + ω over long horizons (drift / marginal-mode precision). *TODO.*
9. **Worm metric path integration (x,y,θ), position_2d** — *RUNNING NOW.*
   The decisive "harder rung": does the worm support a 2-D vector integrator,
   or only the 1-D ring? Fish already does PI; comparing the worm tests whether
   architecture finally matters when the demand exceeds a circular bump.
10. Inverse task: train the worm on its OWN proprioceptive/locomotor
    phase-integration and compare the learned spectrum/manifold to the heading
    solution (Wen 2012 motivation). *TODO.*

**D. Identifiability / dynamics (longer term)**
11. Per-neuron activity or perturbation constraints to break the weight
    degeneracy (Goncalves 2014 logic), since behaviour alone underdetermines J.
12. Eigenvalue-degeneracy detection on the *raw* connectome vs trained operator
    (Clark 2025) — already partly addressed by the Jacobian-at-bump analysis.

---

## 4. Honesty ledger (must hold in all text)
- Connectome gives support + sign prior + scale; OPTIMISATION gives the operator.
  We show an integrator *exists on each support*, not that the wiring *as
  measured* implements one.
- Worm result: converged, positive (clean ring), but used free input gating and
  is one run; robustness gap is an upper bound.
- Sign test: lightweight standalone trainer, modest N; reliability effect is
  suggestive, not quantified.
- All metric claims trace to a `results_*.json` emitted by the analysis scripts.
