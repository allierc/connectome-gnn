# Workflow evidence behind `multilevel_one_sided_note.pdf`

Companion to [`multilevel_one_sided_note.tex`](multilevel_one_sided_note.tex). Two agent
workflows were run to stress-test the multi-level NGP+GNN dynamics design against theory,
prior art, and the code on disk.

**Status: both workflows COMPLETE** (23 and 10 agents, each including adversarial
verification).

| | run id | scope | status |
|---|---|---|---|
| Workflow 1 | `wf_e4f1377a-ca5` | identifiability, expressivity, LoD transfer, units/derivatives, prior art, GNN-vs-CNN | complete, 23 agents |
| Workflow 2 | `wf_a83b751f-55f` | temporal bandwidth: 4-D grid vs band-pass vs multirate vs time-constant | complete, 10 agents |

**The two workflows contradicted each other**, and resolving it produced the sharpest result
in this file. Workflow 1 called the band-pass "provably a no-op"; workflow 2 measured detail
dimensions of `n_ℓ − 1` and said it "kills the constant mode and nothing else"; my own check
measured `n_ℓ − n_{ℓ−1}` and said it works. All three are correct **about their own
restriction operator** — see §0.

## 0. The finding that changes the note: the band-pass needs a dyadic ladder

Workflow 1 claimed the Laplacian-pyramid detail operator is "provably a no-op". That is too
strong, and the note's original claim was too weak on a condition it had flagged and then
ignored. I re-derived and measured it independently (`/tmp/bandpass_check/chk3.py`): a 1-D
five-level ladder, exact left inverse `P_ℓ = pinv(I_ℓ)`, 4097 query points.

| ladder | nesting violation | nullity naive → band-pass | Σ dim `W_ℓ` vs rank of span | contribution unique given the output? |
|---|---|---|---|---|
| `b = 1.5` (NGP default) | 0.23 | 31 → 31 | 86 vs 79 | **no** |
| `b = 1.6` | 0.22 | 26 → 26 | 123 vs 105 | **no** |
| `b = 2.0` (dyadic) | 2.2e-15 | 64 → 64 | **65 vs 65** | **yes** |

Nesting violation is `max ‖(Id − I_ℓ P_ℓ) I_{ℓ−1}‖` — *the largest fraction of a
level-(ℓ−1) function's own amplitude that fails to be representable at level ℓ*. The
**nullity is unchanged in every case**, so no *parameter* ambiguity is removed; what the
band-pass buys is that the **per-level contributions become recoverable from the output** —
and only on an exactly nested ladder.

**And the restriction operator matters just as much as the ladder** (`chk5.py`, dyadic
5-level ladder):

| `P_ℓ` | `max‖P I − Id‖` | `‖I P‖₂` | `dim W_ℓ` | verdict |
|---|---|---|---|---|
| `pinv(I)` least-squares | 2.4e-15 | 1 | 5, 4, 8, 16, 32 | `= n_ℓ − n_{ℓ−1}`, **telescopes, unique** |
| `D⁻¹Iᵀ` normalised adjoint | 0.33 | 1 | 5, 8, 16, 32, 64 | `= n_ℓ − 1`, **DC mode only** |
| `Iᵀ` raw adjoint | 3.4e2 | up to 475 | 5, 9, 17, 33, 65 | `= n_ℓ`, **kills nothing, amplifies** |

So `P_ℓ` must solve the level-ℓ Gram system, not scatter. Three prerequisites, all measured
by workflow 2, all violated by the module defaults:

1. **Integer scale AND integer per-axis base.** Resolutions are
   `int(round(base*scale**l))` per axis, so a non-integer base breaks divisibility even at
   scale 2: base 1.301 gives `3→5`, not `3→6`. Bases `(2.000, 1.301, 0.699)` fail 5 of 18
   axis-pairs at scale 2 and 29 of 33 at 1.45. Bases `(8, 5, 3)` at scale 2 pass 15 of 15
   and stay isotropic in microns to within 11% at every level over 830×540×290 µm.
2. **Linear interpolation on every band-passed axis.** Smoothstep spaces are not nested even
   at integer scale — 11–20% of each coarse basis function escapes — because `3w²−2w³`
   forces zero slope at every fine node. This **overrides workflow 1's** "smoothstep on
   x,y,z" recommendation wherever you band-pass. Use a separate encoder if you need the
   Laplacian.
3. **`P_ℓ` a true left inverse**, per the table above.

**Cost, and why the textbook recipe fails here.** On clustered 3-D geometry (13,700 cells in
~30 anisotropic nuclei in a brain-like shell; 67% of the finest level's nodes touch no cell),
the fraction of each level's block Frobenius norm still inside the coarse space:

| lifting | level 1 | level 2 | level 3 |
|---|---|---|---|
| none | 0.966 | 0.784 | 0.700 |
| 8 damped-Jacobi sweeps | 0.820 | 0.385 | 0.339 |
| 128 damped-Jacobi sweeps | 0.331 | 0.096 | 0.191 |
| 64 mass-preconditioned CG on **occupied** nodes | 3.4e-6 | 4.1e-3 | 7.0e-3 |

The "8 sweeps ≈ 1% of a forward pass" figure is a *uniform-sampling* number. Prune nodes by
mass `d_ℓ = I_ℓᵀ1` with a meaningful threshold (at 1e-8 the pruned Gram still has condition
number 1.1e17 on clustered cells vs 63 uniform), add Tikhonov `ε·diag(d_ℓ)` with `ε = 1e-4`,
and use Krylov or sparse Cholesky — not Jacobi. Pass `d_ℓ` to the GNN as an input channel;
never leave it inside the projector.

**Allocation is safe, structurally.** `level_offsets` give each level a disjoint slice of the
flat table, so hash competition is strictly within-level, while the band-pass is a fixed
linear map on the *sample* index — it cannot mix table rows or nodes and never enters the
hash. It should if anything sharpen the signal, turning the fine table's importance from
*total* importance into *detail* importance.

**The gauge freedom is visible in a trained model.** The baseline split one true slow, coarse
component as **−0.11 / −1.95 / +3.07 of that component's own amplitude** across levels 0/1/2
— three mutually cancelling contributions summing to +1.00. The band-passed model attributed
+1.00 / +0.00 / +0.00.

## 0b. Temporal verdict: build the spatial band-pass, not the temporal one

All four candidate temporal mechanisms — coarsening t alongside x, band-passing in t, gating
each level's update rate, capping each level's output with a time constant — are **one-sided
low-pass ladders**, i.e. the very defect this whole note is about. None stops a spatially
fine level from carrying slow dynamics.

The one genuinely two-sided temporal band-pass tested does work: slow-band leak **0.030 of
the model's total slow-band output power** against the baseline's **0.963**. But it is
brittle where the spatial band-pass is not — on a single-timescale control, held-out R² falls
to **−0.318**, against **+0.986** for the spatial band-pass on the same control, because it
annihilates any component off the length–time diagonal.

The structural reason bounds the whole design: **a single level index cannot carry a
(length, time) pair.** Indexing by ℓ alone forces the ladder onto one diagonal of the
(length scale, time scale) plane; slow fine structure and fast coarse structure have no level
to live on. Representing both needs a two-index family `W_{ℓx,ℓt}` — a much larger
architecture.

**Do this first, it costs no training:** take the temporal power spectrum of the observations
band-passed at each spatial scale and ask whether the characteristic frequency shifts with
the spatial band. If it does not, the diagonal ladder has nothing to capture and the temporal
apparatus is unnecessary. Workflow 2 predicts it does not.

Journals: `~/.claude/projects/-workspace--devcontainer/13b9fa06-35af-478a-9e56-5cbb7da98d1c/subagents/workflows/<run id>/journal.jsonl`.
Scratch code from the agents: `/tmp/ngpgnn/` (exp1–exp8).

---

## 1. The result that changes the design: on flyvis, the spatial reading is dead

Measured on `graphs_data/fly/e8_flywireRF_noise_005` (13,741 neurons, 65 cell types, 217
distinct retinotopic columns, `pos.zarr` is 2-D).

**Type purity of a spatial cell — the largest-type fraction — is 0.016 at every resolution
from 2 to 128 cells per axis.** 0.016 = 1/65 = exactly chance. Mean distinct types per
occupied cell falls only from 65.0 to 63.3 across that whole range. ARI of KMeans on (x,y)
against cell type, at K = 4/8/16/32/65/128: −0.000, −0.001, −0.002, −0.003, −0.004, −0.006.

The reason is structural, not a data defect: **flyvis retinotopy is the axis along which
neurons are equivalent** (same type, different column), and cell type is the axis along
which they differ. A spatial NGP embedding assigns identical `a_i` to the 65 different types
sharing a column, and different `a_i` to the 217 same-type neurons across columns — exactly
inverted relative to what `clustering_evaluation(model.a, type_list)`
(`graph_trainer.py:1079`) scores.

**The connectome reading works.** Per-neuron presynaptic-type connectivity signature
(327,358 edges binned by presynaptic type), cosine similarity: within-type 0.920,
between-type 0.094, within-column 0.093 — spatial colocation is indistinguishable from
random. A Ward tree on that signature (nested by construction, which is the property the
lattice was supplying) gives ARI vs cell type of 0.026 / 0.106 / 0.321 / 0.495 / **0.702** /
0.655 at K = 4/8/16/32/65/128. Residual variance of the ground-truth per-neuron latent
(τ, V_rest, both verified exact type constants) under that tree: 0.961 / 0.904 / 0.701 /
0.473 / **0.136** at K = 4/8/16/32/64 — against **100.0% unexplained at every spatial
resolution**.

> Recommendation from the analyst: build the graph-coarsening regime first, and do not put a
> spatial hash grid on flyvis at all.

## 2. The correction: the sharing mechanism is quantisation, not collision

This contradicts the motivation as originally stated, and the note has been amended.

Neurons per occupied level-ℓ cell on the real ZAPBench geometry: 3118.3, 1120.6, 409.8,
128.1, 49.1, 16.9, 6.2, 2.6, 1.4 for levels 0–8. **Levels 0–8 are all dense** — the
`is_dense` test at `hashgrid.py:131` puts them below the table size, so they have *zero*
collisions. Hashing first appears at level 9, where sharing has already fallen to ~1.1
neurons per cell, and the number of distinct rows hit falls only from 205,300 to 171,662.

So every unit of parameter sharing the design counts on comes from **grid quantisation**
(many neurons in one coarse cell), while the **collision** mechanism cited in the motivation
is active only where sharing has already collapsed to about one neuron per cell. The stated
mechanism and the operative mechanism sit at opposite ends of the level ladder.

Independently, `ngp-demo/tests/level_specialisation.py` is a **standing asserting test in
our own repo** recording that the levels do *not* specialise by scale: the right/left
contribution ratio spans 0.58 to 1.14 across levels from 8 to 1,407 cells per axis, a spread
of 2.0×, under `assert spread < 5.0`. That is direct empirical support for the note's
central claim.

## 3. `GNN_ℓ` on a lattice is a CNN — verified, not argued

A 16×16 lattice graph with a 3×3 stencil, per-offset linear messages and sum aggregation was
compared against `nn.Conv2d(3,5,3,padding=1)` with the transposed per-offset kernel: **max
absolute difference 1.19e-6 against a mean activation magnitude of 1.20**, i.e. exact to fp32
rounding (`/tmp/ngpgnn/exp2.py`). K rounds of message passing at level ℓ = a (2K+1)^D stencil
in level-ℓ cell units.

Two consequences. Σ_ℓ I_ℓ(GNN_ℓ) is, term for term, a Laplacian-pyramid CNN — a decoder-only
U-Net whose skip connections are the I_ℓ upsamplers. And on real geometry the graph framing
*costs*: the occupied lattice vertex set over 14 levels on ZAPBench totals 1,274,214 vertices
against 71,721 neurons, a ratio of **17.8× more nodes**, with finest-level occupancy 22.7% —
not sparse enough to defeat a dense 3-D U-Net.

The fork stated sharply: for the (x,y,z,t) ladder at L=16, T=2²², the table gives 1809×
compression, of which sparsity contributes 4.41× and **collision contributes the remaining
410×**. Take the sparse-vertex reading and you still have 1.72e9 occupied 4-D vertices at the
finest level; take the hash reading and a message-passing graph on hash buckets connects
maximally distant points (13 lattice nodes hashing to one row, mean pairwise distance 0.746
in the unit 4-cube against 0.66 for uniform-random). There is no configuration that gets both
the hash's compression and a meaningful vertex adjacency.

## 4. Prior art: where the construction sits, and what is actually new

- **MGNO** (Li, Kovachki, Anandkumar et al., NeurIPS 2020, arXiv:2006.09535) is the closest
  ancestor. Eq. (7) is `K = K₁ + … + K_L`. Three deltas: its inter-level transitions are
  *learned* kernel integrals (2(L−1) networks) where we use the fixed, parameter-free `I_ℓ`;
  **it is not a parallel sum** despite eq. (7) — §3.3 is an explicit V-cycle, sequential and
  coupled; and §3.1 explicitly concedes the identifiability problem, noting the decomposition
  generalises "by allowing overlap", with `K_{ℓ,ℓ}` integrating over nested balls, not annuli.
  **The V-cycle ordering, not disjoint support, is what breaks the degeneracy in MGNO.**
- **Laplacian pyramid** (Burt & Adelson 1983): `L_ℓ = G_ℓ − Up(G_{ℓ+1})`. The subtraction is
  what makes each level band-pass and the decomposition unique. The single most quotable form
  of the objection: **Σ_ℓ I_ℓ(ẋ_ℓ) without the "− upsampled coarser" term is a *Gaussian*
  pyramid sum, which is precisely the degenerate object.**
- **MINER** (Saragadam et al., ECCV 2022, arXiv:2202.03532) is this architecture with
  per-patch MLPs instead of per-level GNNs, with the internal Laplacian pyramid and
  coarse-to-fine residual fitting already in place.
- **WNO** (Tripura & Chakraborty, CMAME 2023, arXiv:2205.02191) is the closest published
  reconstruction and gets the band-pass right: detail coefficients are band-limited on both
  sides by construction, and `u = Σ_ℓ W_ℓ^{-1}(d_ℓ) + approximation` is literally our
  `Σ_ℓ I_ℓ(·)` with `I_ℓ` = inverse wavelet synthesis.
- **GraphCast** (Lam et al., Science 2023) — my earlier claim confirmed with the mechanism
  precise: node set of the finest icosahedral refinement (40,962 nodes), edge set the *union*
  across all refinements (327,660 directed edges), and **one** 16-layer processor on that
  single merged graph — three graph nets total, not one per level. It keeps all levels' edges
  including redundant fine ones, i.e. it explicitly does *not* band-pass. Note this means the
  annulus edge set I proposed in the first exchange is **not** GraphCast.
- **Learned multigrid** (Katrutsa et al. 2020; Greenfeld et al. ICML 2019; Luz et al. ICML
  2020 — a message-passing GNN that outputs prolongation weights; MgNO ICLR 2024): all learn
  the transfer operators and all are iterative. The lesson: the correction scheme is well
  posed because level ℓ only ever sees the *residual* left by the finer smoother.
- **FNO** mode truncation is the same one-sided upper cutoff, with the same known pathology —
  the residual `W` path bypasses the spectral filter and nonlinear activations re-inject high
  modes.

**Novelty that survives:** using an Instant-NGP multiresolution hash encoding as the
*generator* of a message-passing GNN's per-node embedding `a_i` in a system-identification
setting. No prior work found. The multi-level operator itself should be framed as "MGNO's
`K = ΣK_ℓ` with fixed multilinear prolongation", with the contribution being the removal of
the 2(L−1) learned transition networks.

## 4b. The other blocking objections (full text in Appendix A.1 of the PDF)

Beyond the band-pass question, workflow 1 rated three more objections blocking. Two of them
are about the construction as a whole, not about how the levels are arranged.

1. **No state input — it is an interpolant, not an operator.** If
   `ẋ_ℓ* = GNN_ℓ(NGP_ℓ(x,y,z,t))`, every input is a function of the coordinate, and the only
   "state" is the encoder's reconstruction of the data it was fit to. Cannot be rolled out
   (compare `neural_gnn.py:657` taking a `NeuronState`, and `recurrent_step.py:435–452`
   writing the prediction back); cannot be evaluated on a held-out trial, a novel stimulus,
   or any `t` past the last training frame without refitting the table. *Fix:* the GNN's
   per-vertex input should be the restricted **observed or rolled-out state** `P_ℓ u(·,t)`;
   the NGP supplies only `a_ℓ`.
2. **The two loss terms share one noise realisation** — errors-in-variables by construction,
   since the model term's target is a finite difference of the same array the data term fit.
   Measured on the analogous leak: fitted self-coupling **−0.316 against a true +0.142** at
   the ZAPBench noise level, **−0.906 at 5× that noise** — wrong sign, up to 6× magnitude.
   *Fix:* leave-one-out local-linear slope (fit `t−3…t+3` excluding `t`) plus a standing sign
   test. Do **not** pre-smooth: raw one-frame difference gives held-out R² 0.4391 against an
   oracle-trained 0.4382, while a 12-frame smoothed target drops to 0.3108.
3. **Half the proposed 4-D ladder is exactly redundant.** At `L=16`, base `(16,16,16,4)`,
   scale `(1.5,1.5,1.5,1.3)`, cap `(256,256,64,32)`, resolutions saturate from level 8, so
   levels 9–15 are *identical operators* — 3,670,016 of 7,339,283 table rows, exactly 50%.

Smaller but sharp: the DC mode of `ẋ` is exactly (L−1)-fold degenerate because `I_ℓ 1 = 1`
to 2.2e-16 in D=1..4, dense and hashed, linear and smoothstep — subtract each level's batch
mean for ℓ ≥ 1 and let level 0 carry it. And use smoothstep on x,y,z but **linear on t**:
smoothstep's `6w(1−w)` vanishes at cell boundaries, so on a t axis capped at the frame
spacing `∂/∂t` is identically zero at every observed frame.

## 5. Standing recommendations from the completed agents

1. Build the **graph-coarsening regime first**: replace the free `nn.Parameter a`
   (`neural_gnn.py:224–231`) with `a_i = Σ_ℓ E_ℓ[c_ℓ(i)]` where `c_ℓ` is a nested
   Ward/METIS/Louvain coarsening of the flyvis connectome and `E_ℓ` has K_ℓ = 4, 8, 16, 32,
   64, N rows. Score with `clustering_evaluation(model.a, type_list)`, which already exists.
   ~60 lines, plus the enumerated hard breaks at `sparsify.py:786/793/799` and
   `graph_trainer.py:1090/1758/1764`.
2. **Drop the spatial hash grid on flyvis** outright. Zero cost — it is a deletion.
3. ~~Replace `I_ℓ` with the Laplacian-pyramid detail operator, or equivalently fit level ℓ to
   the residual using `set_level_window()`.~~ **Superseded — the second half is wrong.**
   `set_level_window()` gates the level's *feature slice* before the concatenation
   (`hashgrid.py:233–237`), i.e. the network's **input**, not its contribution to the output.
   Band-disjoint inputs cannot form a cross-scale product like `v·∂u/∂x`: measured relMSE
   0.835 on advection against 0.0002 with low-pass features. **Keep inputs low-pass; put the
   lower cutoff on the output**, either as the band-pass of §0 (needs `b = 2`) or as a soft
   penalty `λ_band Σ_{ℓ≥1} ‖Π_{ℓ−1} I_ℓ ẋ_ℓ‖²` with `λ_band = 0.1` on a fixed reference
   query set (measured relMSE 0.0020–0.0021 across two seeds, against 0.0002 unpenalised).
4. **Rewrite the mechanism claim** from "collisions allocate fine levels to fast/small
   regions" to "per-level capacity below the cluster count forces sharing, and the finest
   level absorbs the residual" — before running experiments, since it changes what they must
   show.
5. Keep GraphCast's union-of-edges as the documented fallback if the per-level decomposition
   proves non-identifiable in practice.

## 6. Useful facts pinned by the grounding readers

- **`I_ℓ` exactly**: `hashgrid.py:182–237`. Partition of unity verified numerically (min
  0.9999999, max 1.0000001). Smoothstep preserves it. Corner weights are an explicit chain of
  multiplies rather than `Tensor.prod`, because `prod`'s double backward leaks ~1e-2 spurious
  curvature in fp32.
- **`level_gain`** (`hashgrid.py:157`) is a non-persistent buffer, not a Parameter — but it
  can be assigned arbitrarily, and `scripts/gui_image.py:316–318` already sets a one-hot to
  isolate a single level. That is the ready-made hook for per-level isolation experiments.
- **Message passing in `NeuralGNN`** (`neural_gnn.py:603–655`): message MLP input is
  `(v_j, a_j)` only — presynaptic voltage and presynaptic embedding; `a_i` enters only the
  update, except in `flyvis_conductance` which uses `[v_j, a_j, v_i, a_i]`. Aggregation is
  plain unweighted `scatter_add_`; the `aggr_type` constructor arg is stored but never read.
  No `torch_geometric.MessagePassing` anywhere in the repo.
- **ZAPBench is the only real (x,y,z,t) neural-activity data on disk**: 7,870 frames ×
  71,721 neurons ΔF/F (2.26 GB, zero NaNs), **frame spacing 0.915 s**, 7,201 s total, with
  soma centroids giving a bounding box of **818 × 504 × 254 µm** and a median
  nearest-neighbour spacing of **4.83 µm**. No connectome for those neurons (only ~481 are
  EM-matched). Units of the centroid file are *inferred, not documented*.

---

*This file will be superseded when both workflows finish. To resume either:*
`Workflow({scriptPath: "...workflows/scripts/<name>-<run id>.js", resumeFromRunId: "<run id>"})`
