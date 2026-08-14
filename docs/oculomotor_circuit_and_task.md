# The larval zebrafish oculomotor integrator: circuit, task, and training data

Working document for the `feat/oculomotor` branch. It records what the circuit
is taken to be, what the network will be asked to compute, and how the
training data is generated. Everything here that is a *claim about biology*
rather than a fact read off the reconstruction is marked as such — those are
the parts that need the circuit owner's signature before any result means
anything.

Source reconstruction: `Oculomotor_sortedData_081126.pkl`, 2949 cells,
42 cell types, 44,584 edges (density 0.51 %). Selected sub-circuit:
`config/zebrafish/zebrafish_om_intg_285_v1.yaml`, 285 cells, 8 types.

![**The oculomotor reconstruction and the sub-circuit modelled here.** (a) All 2949 cells, ordered by the 16 coarse cell-type families, support mask of the synapse-area matrix. (b) The 285-cell sub-circuit, ordered afferent then recurrent then output, and within a type left hemisphere before right. Colour carries the biology: blue = afferent, green = excitatory recurrent, red/orange = inhibitory recurrent, purple/pink = motor output. (c) The same sub-circuit signed by the Dale assignment of section 1.4 — each synapse coloured by the sign of its *presynaptic* cell, blue excitatory, red inhibitory, shade log-scaled in synaptic area. This is what the model multiplies, $\hat W_{ij} = |\hat S_{ij}|\,\mathrm{sign}(W^{\mathrm{con}}_{ij})$, and the support mask of (b) cannot show it. Because the sign belongs to the column, the panel reads as vertical bands: the two `INTG_contra` types are a solid red block, and it is visibly the only inhibition the 285 cells contain. Rows are postsynaptic, columns presynaptic in all three. Regenerate with `python scripts/plot_oculomotor_connectome.py --config config/zebrafish/zebrafish_om_intg_285_v1.yaml --out figures/zebrafish/fig_oculomotor_circuit.png`.](../figures/zebrafish/fig_oculomotor_circuit.png)

![**Figure 1 — the oculomotor circuit.** Twin of Figure 1 of the zebrafish heading-direction paper, rendered through the same code with the content swapped. **(a)** All 285 skeletons from the neuprint-fish2 reconstruction, dorsal view, coloured by role: AF5 afferents blue, ipsilateral (excitatory) INTG green, contralateral (inhibitory) INTG red, AMN purple, AIN pink. **(b)** The cell bodies alone, same view — the AF5 somas sit well anterior of the integrator/motor cluster (radii scaled x0.3 for legibility, not a measurement). **(c, d)** The same skeletons and somas recoloured by Dale sign: blue excitatory (233 cells), red inhibitory (52 cells). The inhibitory population is exactly the contralateral integrator of section 1.4, and the two rows together are the claim being made — that the E/I split is by projection laterality, not by anatomical position. **(e)** The computation of section 4.6 end to end: the target velocity $(\dot x,\dot y)$ enters through $\hat W^{\mathrm{in}}$ on the AF5 afferents, the INTG pool integrates under sign-locked continuous-time rate dynamics with mutual inhibition across the midline, $\hat W^{\mathrm{out}}$ reads non-negative motor drives, and the push-pull differences $u_\theta,u_\varphi$ drive the eye plant to the gaze angles $(\theta,\varphi)$. LR and MR have a pool in these 285 cells; SR and IR would come from OMN, which section 1.5 leaves out. Regenerate with `python figures/zebrafish/fig_1_oculomotor_overview.py`.](../figures/zebrafish/fig_1_oculomotor_overview.png)

## 1. The circuit

### 1.1 Three pools

Anatomy for all 285 cells was fetched from neuprint-fish2 and is shown in
Figure 1. It had to be keyed on bodyId rather than cell type: the server does
not carry the reconstruction's type names — `INTG_ipsi_m` is `INTGip1` there,
`INTG_contra_m` is `INTGco1` — and the two AF5 populations carry no type at
all, so a type query returns 142 of 285 cells and drops the whole afferent
stage. All 285 bodyIds resolve.

The 285 cells divide into an input stage, a recurrent integrator and a motor
output stage.

| pool | types | cells | Dale sign |
|---|---|---|---|
| afferent | `AF5_ipsi` (29), `AF5_contra` (12) | 41 | excitatory |
| recurrent | `INTG_ipsi_m` (38), `INTG_ipsi_i` (27), `INTG_contra_m` (34), `INTG_contra_i` (18) | 117 | ipsi excitatory, contra inhibitory |
| output | `AMN` (92), `AIN` (35) | 127 | excitatory |

Within the sub-circuit there are 5013 edges, a density of 6.2 % — an
order of magnitude above the 0.51 % of the full reconstruction. That
concentration is what one expects of a recurrent integrator core plus its
dedicated input and output stages, and it is the first (weak) evidence that
the selection is a circuit rather than an arbitrary subset.

### 1.2 AF5 is an afferent, and the wiring says so

The two AF5 populations are the pretectal arborization field that carries
optokinetic drive. Their afferent status is not assumed. In the measured
matrix, AF5 sends 88.2 % of its outgoing synaptic area into the
INTG/AMN/AIN core and receives 1/178th as much back (6.89e5 out against
3.87e3 in). This is the same asymmetry test that confirmed the matrix
orientation using the retinal ganglion cells, which are 15.4x more
outgoing than incoming.

Both AF5 types are left/right balanced — `AF5_ipsi` 15 L / 14 R,
`AF5_contra` 6 L / 6 R — so a bilateral input gate is expressible. This is
worth stating because it is not generally true in this dataset: `RGC_AF7`
is reconstructed in the left hemisphere only (804 L / 0 R) and could not
support a symmetric gate.

### 1.3 What drives the afferents — CLAIM, and an open one

The direction selectivity below is the circuit owner's description, not a
measurement in this file. The combination rule is explicitly unresolved.

- **`AF5_ipsi`** — the left pool is driven by rightward-eye motion that is
  leftward and/or forward; the right pool by leftward-eye motion that is
  rightward and/or forward.
- **`AF5_contra`** — the left pool is driven by rightward-eye motion that is
  rightward and/or backward; the right pool by leftward-eye motion that is
  leftward and/or backward.

Whether the two conditions combine as AND, OR or XOR is **not known**. This
is not a detail to be settled by a default: it decides whether a single
afferent unit reports a conjunction (a narrow direction tuning) or a
disjunction (a broad one), and therefore how much of the direction
computation the recurrent circuit has to do itself. Until it is resolved,
the input model should carry it as an explicit switch and the three variants
should be trained and compared.

### 1.4 The integrator, and why the E/I split is by laterality

The four INTG types are the velocity-to-position integrator. The sign
assignment follows their projection laterality: **ipsilateral projections
excitatory, contralateral projections inhibitory**. That is the classical
integrator motif — two half-integrators, one per hemisphere, each
self-exciting locally and inhibiting its opposite number across the midline.
Mutual inhibition is what lets the pair hold a *signed* position variable on
a single positive-rate substrate. Panels (c) and (d) of Figure 1 plot the
assignment on the anatomy — blue excitatory, red inhibitory — and what they
show is that it is not readable from position: the 52 inhibitory cells are
interdigitated with the 233 excitatory ones in the same hindbrain column,
which is why the split has to be asserted per cell type rather than inferred
from where a soma sits.

This is a Dale assignment by cell type, exactly as in the heading-direction
circuit, where `_ZHD_INH_PREFIXES` in `connectome_loaders.py` encodes the
claim that the r1pi/dIPN cells are GABAergic. Neither is a measurement.
The difference is that here the claim is written in the config, per type,
where it can be reviewed and flipped, rather than in a tuple of string
prefixes inside a loader.

### 1.5 The output stage, and one simplification worth flagging

Only the **horizontal** pair of extraocular muscles is modelled:

The **plant** is the control-theory name for the thing being controlled —
here the globe, its six muscles, the orbital tissue and their mechanics,
everything downstream of the last neuron. It is worth using rather than "the
eye" because it names a boundary: the circuit's output is a muscle command,
the plant turns that command into an angle, and the plant is measured and
frozen while the circuit is learned. Section 4.7 identifies it.

- `AMN` drives the **lateral rectus** (`LR` in the plant), which abducts the
  eye;
- `AIN` drives the **medial rectus** (`MR`), which adducts it.

Muscle keys refer to the six-muscle soft-body eye in
`Plexus/prototype/eye`, where each muscle carries an innervation state
`muscle.act` and the globe's pose is recovered by a Kabsch fit
(`eye_pose` -> horizontal, vertical, torsion in degrees). Coupling the
readout to that plant, rather than to an abstract scalar, is what would make
"eye position" a mechanical quantity instead of a fitted one.

**The simplification.** `AIN` are abducens *internuclear* neurons. In vivo
they are not motor neurons: they project across the midline to the
oculomotor nucleus (`OMN`), whose motor neurons innervate the medial rectus.
`OMN` is 207 cells in this reconstruction and is **not** in the selected
pool. Treating `AIN` as driving the medial rectus therefore collapses a
two-synapse pathway into one, and drops the sign inversion and delay that
the missing synapse would contribute. This is a legitimate modelling choice,
but it should be stated in any write-up, and the obvious control is to add
`OMN` to `circuit.cell_types` and check whether the fitted dynamics change.

### 1.6 What is not in the pool

Two omissions are deliberate and one of them constrains the task:

- **`Burst_T1A/T1B/T8/T9/T10`** (74 cells) are the saccadic burst
  generators — the source of the brief velocity command that a saccadic
  integrator integrates. They are excluded, so in the current pool the only
  anatomically defined input is the optokinetic AF5 drive. A saccade-driven
  task would have to inject velocity directly into INTG, which is a
  modelling choice, not a connectome-derived pathway.
- **`OMN`** (207 cells), as described in §1.5.

## 2. The task

### 2.1 What the network computes

The oculomotor integrator converts an eye-**velocity** command into an eye-
**position** command: the motor neurons must hold a rate proportional to
position long after the velocity signal has gone. Written as the leaky
integral of the drive, per axis,

```math
\frac{d\theta}{dt} = -\frac{\theta}{\tau} + s_\theta\,\dot x(t),
\qquad
\frac{d\varphi}{dt} = -\frac{\varphi}{\tau} + s_\varphi\,\dot y(t)
```

where $(\theta,\varphi)$ are the horizontal and vertical eye angles and
$(\dot x,\dot y)$ the target velocity — the notation of section 4.6, in which
$v$ is reserved for membrane voltage. The quantity of interest is $\tau$, the
time constant over which eye position leaks back to centre. A perfect integrator is $\tau = \infty$; the biological
integrator is finite, and measuring what $\tau$ the connectome-constrained
circuit can support is the point of the exercise.

### 2.2 It needs no new task code

This is already a mode of the existing swim-integration generator. Setting

```
task.swim_integration.target_kind: scalar_xi
task.swim_integration.xi_tau_s:    <the leak>
training.task_targets:             [translation]
```

yields a **1-input / 1-output** problem: the input channel is the velocity
drive, the target is `xi = integral of v dt` with leak `xi_tau_s`. The
`_integrate_leaky` helper in `graph_data_generator.py` implements exactly the
equation above (forward Euler, `alpha = 1 - dt/tau`), and `tau <= 0` recovers
the perfect integrator.

What the existing machinery does **not** provide is the neuron binding:
which cells the input enters and which cells the readout reads. In the
heading circuit both are hardwired to the HD taxonomy — the encoder through
`graph_model.velocity_gate`, the decoder through `output_from_dipn_only` and
a positional `[:n_bump]` slice. Redirecting them to the AF5 afferents and the
AMN/AIN outputs is the real work, and it is Procedure B in
`HOWTO_add_zebrafish_circuit.md`.

### 2.3 Saccades versus optokinetic drive

The two natural stimulus regimes differ, and the choice interacts with §1.6:

- **Optokinetic**: sustained whole-field motion, the drive AF5 actually
  carries. Slow, continuous velocity — the regime the selected pool supports
  anatomically.
- **Saccadic**: brief high-velocity pulses separated by fixations, the
  classical integrator probe. This is the regime the burst generators serve,
  and they are not in the pool.

The swim generator's stimulus is a Poisson train of boxcar impulses
(`swim_rate_hz`, `swim_duration_s`) — structurally a saccade train already.
So the saccadic regime is reachable by reparameterising the existing
generator rather than writing a new one; what is missing is anatomy for the
input, not code.

### 2.4 The open-loop problem

Coarsely: the circuit is asked to keep a moving target centred on the fovea
while being told only how fast the target is moving, never where it is. That
is the situation the anatomy puts it in. AF5 is a motion-sensitive
arborization field — it reports optokinetic *velocity* — and nothing in the
selected 285-cell pool reports absolute target position. A controller with
position feedback can always correct, so its errors do not accumulate; a
controller given velocity alone must reconstruct position by integration and
has nothing to check the result against. Drift is therefore not a failure of
the circuit but a property of the problem, and the meaningful question is not
whether the target is held but for how long.

More specifically, take the horizontal axis and write the target
eccentricity $\theta^{\star}(t)$, the gaze $\theta(t)$, and the retinal error
$\varepsilon = \theta^{\star} - \theta$; the vertical axis is the same with
$\varphi$. Every trial starts foveated at the centre,
$\theta(0)=\theta^{\star}(0)=0$. The plant here is idealised as rate control:
the motor command sets gaze *velocity*, so gaze angle is its integral. The
ideal controller is then

```math
\frac{d\theta}{dt} = \dot\theta^{\star}(t) = s_\theta\,\dot x(t)
\qquad\Longrightarrow\qquad
\varepsilon(t) = 0 \ \ \forall\, t
```

and tracking is exact. The interesting cases are the three ways a biological
integrator departs from it, each of which produces a different growth law for
$|\varepsilon|$ — which is what makes them separable from a single measured
trace.

**Gain.** If the velocity-to-position conversion is miscalibrated by a factor
$k$,

```math
\frac{d\theta}{dt} = k\,\dot\theta^{\star}(t)
\qquad\Longrightarrow\qquad
\varepsilon(t) = (1-k)\,\big[\theta^{\star}(t)-\theta^{\star}(0)\big]
```

The error is proportional to the *displacement* from the start, not to the
distance travelled, because integration is linear. In a bounded workspace
displacement is capped by the arena, so this error is self-limiting: in our
geometry the target reaches a median 1.16 arena units from the centre, and
$|1-k|$ must exceed 0.52 before the target can leave a 0.6-unit fovea at
all. A
realistic miscalibration of a few percent is invisible. Gain is not what
loses the target.

**Leak.** A real integrator is imperfect and relaxes toward its rest state
with a time constant $\tau$:

```math
\frac{d\theta}{dt} = -\frac{\theta}{\tau} + \dot\theta^{\star}(t)
```

In the frequency domain this is a *high-pass* on target eccentricity, corner
frequency $1/\tau$. Sustained excursions are under-represented, so the error
saturates rather than growing without bound, and gaze is a shrunken,
centre-biased copy of the truth. The counterintuitive consequence is that
*slow* targets are lost first, because their motion is exactly the
low-frequency content the leak removes. Measured: at $\tau = 2$ s a fast
target is never lost within 20 s while a slow one goes at 5.9 s, and the slow
case needs $\tau \geq 8$ s to survive. $\tau$ is the quantity the INTG pool
exists to make long, and this is the curve that says how long is long enough.

**Noise.** If the integrated velocity carries a white perturbation of
intensity $\sigma$,

```math
\frac{d\theta}{dt} = \dot\theta^{\star}(t) + \sigma\,\xi(t),
\qquad \langle \xi(t)\,\xi(t')\rangle = \delta(t-t')
```

the error is a random walk: $\mathrm{std}[\varepsilon(T)] = \sigma\sqrt{T}$
per axis, unbounded
but with zero mean. It is the only one of the three that cannot be corrected
by better calibration and the only one that averages away across trials, so
it is invisible to any analysis that pools trials before measuring drift.

Three growth laws — bounded-linear, saturating, and square-root — over one
shared quantity. A drift trace measured from the real circuit can therefore
be *classified*, not merely reported, which is the point of setting the
problem up this way. `prototype/dot_tracking/openloop.py` implements the gain
and leak cases and measures the survival time `t_lose`, the first moment
$|\varepsilon|$ exceeds the fovea; the noise case is stated here for completeness but
is not in the prototype, since a zero-mean drift is better measured against
recorded data than against a simulation of itself.

## 3. The training dataset

### 3.1 Where it comes from

Task data is generated by `GNN_Main.py -o generate <config>`, which reaches
`data_generate` -> `data_generate_task` (dispatch on `task.task_type`) ->
`_generate_swim_integration_task`, all in
`src/connectome_gnn/generators/graph_data_generator.py`.

The critical property: **generation is connectome-agnostic**. The generator
writes `stimulus.zarr` of shape (trials, T, channels) and `target.zarr` of
shape (trials, T, columns) — pure time series, with no notion of a neuron.
The circuit appears only as a `circuit_provenance.json` written beside the
splits, recording the circuit name, cell count and the SHA-256 of
`J_effective`, so a dataset can later be checked against the circuit a
checkpoint was trained on.

The consequence for planning: the dataset can be generated and inspected
**before** the circuit registry entry, the connectome CSVs or the E/I
assignment exist. That is the cheapest next step on this branch.

### 3.2 Parameters that define it

| parameter | meaning |
|---|---|
| `n_trials_train` / `n_trials_test` | independent trials per split |
| `n_steps`, `dt` | samples per trial and the timestep, so trial duration is `n_steps * dt` |
| `swim_rate_hz` | mean Poisson rate of impulse onsets — the saccade rate |
| `swim_duration_s` | boxcar width of one impulse — the saccade duration |
| `forward_vel_mean` / `forward_vel_std` | amplitude distribution of the velocity drive |
| `target_kind` | which target is assembled; `scalar_xi` is the integrator |
| `xi_tau_s` | the leak; absent or <= 0 gives a perfect integrator |
| `seed` | stimulus RNG; fixed across baseline and comparison runs |

`training.task_targets` then projects the on-disk superset down to the
columns a given run trains on, so one dataset serves several sub-tasks.

### 3.3 A caution about the column layout

The mapping from `task_targets` to input/output dimensions is duplicated
across **13 files** — `_TT_DIMS` in both the RNN and GNN model classes, and
`_PROFILE_BY_TARGET` in `graph_trainer`, `graph_tester`, `plot_cx` and six
figure scripts. A new task mode added in one place does not error elsewhere;
it silently mis-slices columns. Any new mode has to be added to all of them,
or the analysis figures will quietly describe the wrong variable.

## 4. Next step: learning the controller

The task is **supervised, not reinforcement**, and it is worth being explicit
about why, because the instinct to reach for RL here is strong and wrong. In
open loop the correct output is known in closed form at every timestep —
given the velocity stream, the target eye position is its integral — so the
teacher is dense rather than sparse. And the environment does not react to
the agent: the dot moves the same way whatever the eye does. With no feedback
loop and no unknown to explore, this is sequence-to-sequence regression
trained by backpropagation through time. RL would recover the same gradient
information with far more variance and no compensating benefit.

Closed loop does introduce a loop, but not a reason to change method: the
plant is `gaze = integral of command`, which is differentiable, so one simply
backpropagates through it. RL earns its place only where the objective stops
being differentiable — driving the MPM soft-body eye of `Plexus/prototype/eye`
without backpropagating through the simulator, or choosing *when* to make a
catch-up saccade, which is a discrete decision rather than a continuous
command. Smooth pursuit is regression; saccade timing is a policy.

### 4.1 Two stages, in this order

**Stage 1, in the prototype, with no biology in the way.** Generate a corpus
of trajectories with `trajectory.py`, and train a small unconstrained
recurrent network — free $\hat W$, no Dale, no connectome — on exactly the
task `openloop.py` measures. The point is not the model; it is the
calibration. It answers how many trials the task needs, what training horizon
is required, and what integrator time constant is reachable at all when
nothing anatomical constrains the solution. That number is the ceiling.

**Stage 2, the synaptic solution.** Replace the free recurrent matrix with
the 285-cell sign-locked $\hat W$, keeping the same data, loss and curriculum,
and keep the encoder and decoder small: the encoder maps the velocity signal
onto the AF5 afferents, the decoder reads AMN/AIN. The gap between stage 1
and stage 2 is then interpretable as the cost of the anatomy, rather than as
an unexplained training failure — which is exactly the comparison that cannot
be made if the constrained model is trained first and alone.

### 4.2 Four things that will bite

1. **Credit assignment over the horizon.** Gradients through an integrator
   across thousands of steps. The heading-direction configs handle this with
   a step-count curriculum (`n_steps_schedule`, 100 -> 800) and a tail-weighted
   loss (`coeff_tail_loss`); expect to need both.
2. **`tau` is not identifiable from short trials.** On an 8 s trial a 20 s
   time constant and a perfect integrator are indistinguishable. If `tau` is
   the scientific quantity, the training horizon has to approach it, or the
   protocol needs an explicit hold-and-decay probe: drive, then stop, and
   watch the decay.
3. **Degeneracy.** Many recurrent matrices integrate. Sign-lock, Dale and
   spectral normalisation are what select among them, and the residual
   degeneracy is the same identifiability problem the zebrafish
   heading-direction work already documents.
4. **Signed position on non-negative rates.** AMN and AIN are motor pools
   with rates bounded below by zero, so eye position must be carried as a
   push-pull difference (lateral minus medial rectus) rather than by a free
   linear readout. That is what the horizontal pair physically is, and
   building it in is cheaper than hoping the decoder discovers it.

### 4.3 Stage 1, measured

The calibration has been run. A balanced corpus of 5160 centre-started
trajectories — every combination of the four switches, 24 conditions, 8 s at
60 Hz, seeds disjoint across splits — and five learners with the same
interface: linear encoder, swappable core, linear decoder. Scored on one
shared 960-trial test set, mean |drift| in grid units:

| controller | mean drift | never lost | tau_e |
|---|---|---|---|
| `perfect` (analytic ideal) | 0.004 | 100 % | — |
| **`ctrnn`** | **0.007** | 100 % | inf |
| `gru` | 0.009 | 100 % | inf |
| `intlin` (fitted leaky integrator) | 0.090 | 100 % | 9.5 s |
| `gain`, k = 0.5 | 0.225 | 98 % | — |
| `leaky`, tau = 2 s | 0.284 | 67 % | — |
| `mlp` (windowed, memoryless) | 0.392 | 25 % | 0.48 s |
| `rnn` (discrete tanh) | 0.424 | 27 % | 0.40 s |

Four things follow.

**The ceiling is 0.007, and it is the evaluation's floor rather than the
model's.** `ctrnn` is the continuous-time rate network
$\tau_i\,\dot v_i = -v_i + \sum_j \hat W_{ij} r_j + I_i$ with
$\mathbf{I}=\hat W^{\mathrm{in}}(\dot x,\dot y)^{\top}$ — the same equation
as `zebrafish_hd_si`,
with the sign-lock and the connectome removed. Trained to convergence it
reaches 0.0074, against 0.0041 for exact analytic integration. The residual
is not a modelling shortfall: the dataset defines velocity by central
difference while the target is an Euler sum, and that mismatch alone is
~0.007 on a fast sharp trajectory. The network has reached the numerical
floor of its own scoring.

Which lever moved it is worth recording, because three of the four did
nothing. Training length with a cosine schedule took 0.0144 to 0.0074, a
factor of two. Widening the core from 64 to 192 neurons — nine times the
recurrent parameters — gave 0.0072, i.e. nothing. More data would give
nothing either: train and validation MSE are equal to five decimal places
(0.00004 each), so the model was never overfitting and the corpus was never
the constraint. The binding constraint was optimisation, exactly as it was
for the discrete RNN, only there it was fatal and here it was a factor of
two.

That 0.007 is the number the 285-cell constrained circuit should be compared
against. And the comparison is not speculative: the heading-direction work
in `zebrafish.tex` already trains this same equation under sign-lock on a
917-cell measured connectome and reaches r_theta = 0.998 with a precision
horizon beyond 60 s. The constrained form is known to train. What stage 2
measures is not whether it can be done but what it costs.

**Memory is the bottleneck, and the failure is quantitative.** The windowed
MLP has no state and can only reconstruct displacement within its 0.5 s
window. Its measured decay constant is **0.48 s** — its window, recovered
from a hold-and-decay probe it was never trained on. A memoryless model does
not approximately fail here; it fails by exactly the amount its architecture
predicts.

**A discrete tanh RNN cannot learn this, and that is an optimisation result,
not a capacity one.** It plateaus at 0.42 whether trained for 40 or 250
epochs, and identity-initialising the recurrent matrix does not rescue it.
The gradient of an 8 s integral through a contracting discrete map vanishes.
The continuous-time form fixes it because with `dt/tau` small the update is
near-identity by construction. This matters for stage 2: had we used the
discrete form as the stand-in, we would have concluded that the task was
hard, when the difficulty was in the parameterisation.

**The analytic controllers are lesions, not competitors.** Their gain and
time constant are not free choices to be tuned for a fair fight — fitted,
they go to `k = 1` and `tau -> infinity`, which is just `perfect` again,
because the task's ideal solution *is* a perfect integrator. `gain = 0.5` and
`tau = 2 s` are deliberate defects, chosen so the drift is visible in the
interface. The properly fitted member of that family is `intlin`, which
learns its own decay and lands at `tau = 9.5 s`. Read the analytic rows as
"how bad is this specific defect", never as "how good is this method".

### 4.4 Where the integration actually lives

The trained `ctrnn` was audited, because a mean drift of 0.007 over 8 s from
velocity alone is the kind of number that is usually a bug. It is not: the
train and test seeds are disjoint by construction (0 overlap, verified), the
one-sample look-ahead in the central-difference velocity is worth only 13 %
(0.0071 against 0.0080 with a strictly causal backward difference), and the
model was never trained at the horizon its time constant is probed at.

What the audit did turn up is the interesting part. Every neuron in the
trained network leaks, and leaks fast:

```
learned tau_i per neuron :  min 0.57 s   median 0.74 s   max 0.96 s
learned |W_ij|           :  mean 0.035   max 0.267   (initialised at ZERO)
```

A hold-and-decay probe on the same network returns an effectively infinite
time constant over 20 s. So the integration is not in the cellular time
constants at all — a 0.74 s membrane cannot hold anything for 20 s. It is
built by the recurrent matrix, which started at exactly zero and was learned:
positive feedback through $\hat W$ cancels the per-neuron leak, giving a network
time constant more than 27 times the neuronal one.

This is the classical oculomotor-integrator result arrived at from the other
direction. Real neural integrators hold eye position for tens of seconds
using neurons whose membrane time constants are of order 100 ms, and the
integration is a **network** property produced by recurrent feedback rather
than a cellular one. An optimiser given free choice of both — per-neuron
$\tau_i$ and recurrent $\hat W$ — put the integration in the network and left the
neurons leaky. It was not obliged to; the per-neuron time constants were
learnable and could have grown instead.

Two consequences for stage 2. First, the quantity to measure on the
constrained circuit is not any neuron's time constant but the **feedback gain
the connectome can support** — whether the measured INTG wiring, once
sign-locked, can supply enough recurrent excitation to cancel a leak it does
not control. Second, this solution is known to be fragile: a line attractor
built from tuned positive feedback requires the gain finely balanced against
the leak, and a few percent of detuning collapses or destabilises the
integrator. Perturbing $\hat W$ and re-measuring the effective time constant is
therefore the natural robustness test, and it is the one place a sign-locked
connectome-constrained $\hat W$ might behave quite differently from a freely
learned one.

### 4.5 How the recurrent models are trained

Worth stating precisely, because two of the choices are what make the task
learnable at all and a third is a known omission.

The state starts at $v_i = 0$ for every neuron on every trial, never learned
and never carried between trials. That is why the corpus is centre-started:
$v_i = 0$ has to *mean* "eye centred", so that the initial condition and the
task agree. The loss is a plain mean squared error over
**every timestep and both output channels** — dense supervision from t=0, not
an endpoint loss.

The horizon grows during training, always as a prefix from t=0 rather than a
sliding window:

| epochs | horizon | |
|---|---|---|
| 0-62 | 120 steps | 2.0 s |
| 63-124 | 240 steps | 4.0 s |
| 125-186 | 360 steps | 6.0 s |
| 187-249 | 480 steps | 8.0 s |

Each stage re-runs from $v_i = 0$ and truncates; there is no truncated BPTT and
no carried state. Gradients flow through the whole unroll — 480 sequential
steps at the last stage, no detach — with Adam at 2e-3 decayed by cosine to
zero, gradient-norm clipping at 1.0, batch 128, so 29 updates per epoch and
7250 in total.

The curriculum is not a refinement, it is the reason this trains. At the full
8 s horizon from a cold start the gradient of the integral is exactly what
defeated the discrete RNN; beginning at 2 s makes the 8 s case reachable. It
is the same device as `n_steps_schedule` in the heading-direction configs.

**What is missing is tail weighting.** Early timesteps, where the integral is
small and nearly free, currently count as much as late ones, so the loss
under-rewards precisely the long-horizon accuracy the exercise is about. The
heading-direction configs carry `coeff_tail_loss` for this reason and this
prototype does not; adding it is the obvious next attempt at the residual gap
to exact integration.

One reassurance about the horizon. The models are tested at the 480 steps
they were trained on, but the hold-and-decay time-constant probe runs 1200
steps — 2.5 times anything seen in training — and the integration holds.
Whatever the network learned, it is not a fit to the trial length.

### 4.6 The whole model in one notation

![**The eye, and the symbols of the model.** **(a)** The globe seen down the corneal axis with the six extraocular muscles at their true insertions, drawn from `eye_anatomy.MUSCLES` — the same table the MPM uses to shape the straps, so this is the model's own geometry. Only the horizontal pair is innervated by this circuit: LR abducts (positive gaze), MR adducts. The four faint muscles exist in the plant but receive no command. The two axes the model uses are marked: horizontal gaze $\theta$, positive in abduction, and vertical gaze $\varphi$. **(b)** Every symbol of the nine steps below, in the order the signal meets them: the target velocity $(\dot x,\dot y)$, the input map $\hat W^{\mathrm{in}}$ onto the AF5 afferents, the sign-locked recurrent core, the output map $\hat W^{\mathrm{out}}$ onto four non-negative motor pools, the push-pull commands $u_\theta,u_\varphi$, the frozen per-axis plant, and the loss on the gaze angles $(\theta,\varphi)$ — after the plant, not on the command. Regenerate with `python fit_plant.py` and `python fig_eye_schematic.py` in `Plexus/prototype/eye`.](../figures/zebrafish/fig_eye_schematic.png)


Written in the formalism of `neurips.tex`, with the symbols laid out in the
figure above, so the oculomotor circuit and the flyvis work can be read side
by side. Every stage from the two numbers the circuit is given to the two
angles it is scored on is written out below, because the intermediate
quantities are where the modelling choices are, and they are invisible in any
summary that jumps from "velocity in" to "position out".

| symbol | dimension | meaning |
|---|---|---|
| $x(t),\,y(t)$ | arena units | target position in the world |
| $\dot x(t),\,\dot y(t)$ | units s$^{-1}$ | target velocity — **the only input** |
| $s_\theta,\,s_\varphi$ | deg / unit | world-to-angle scale, one per axis |
| $\theta^{\star},\,\varphi^{\star}$ | deg | target eccentricity, horizontal and vertical |
| $v_i(t)$ | — | membrane voltage of neuron $i$ ($N=285$) |
| $\tau_i$ | s | membrane time constant of neuron $i$ |
| $r_i(t)$ | — | firing rate, $r_i=\rho(v_i)$ |
| $\hat W \in \mathbb{R}^{N\times N}$ | — | recurrent weights, sign-locked to the connectome |
| $\hat W^{\mathrm{in}} \in \mathbb{R}^{N\times 2}$ | — | velocity to afferents; rows zero outside AF5 |
| $\hat W^{\mathrm{out}} \in \mathbb{R}^{4\times N}$ | — | rates to muscles; columns zero outside the motor pools |
| $m_{\mathrm{LR}},m_{\mathrm{MR}},m_{\mathrm{SR}},m_{\mathrm{IR}}$ | $\ge 0$ | motor-pool drives, one per muscle |
| $u_\theta,\,u_\varphi$ | $[-1,1]$ | push-pull commands, one per axis |
| $\Phi_\theta,\,\Phi_\varphi$ | deg | static nonlinearity of the plant, per axis |
| $\omega_n,\,\zeta$ | rad s$^{-1}$, — | plant mechanics, per axis |
| $\theta(t),\,\varphi(t)$ | deg | **eye angles** — horizontal and vertical gaze |

A hat marks a learned quantity. $v$ is the membrane voltage throughout and
never a velocity; the velocities are $\dot x$ and $\dot y$, and the eye's
orientation is the pair $(\theta,\varphi)$, horizontal first.

**Step 0 — the world sets the angles.** The target moves in a bounded arena
in units, and each axis is mapped to degrees by its own scale, because the
eye's reachable travel is not the same horizontally and vertically
(section 4.7):

```math
\theta^{\star}(t) = s_\theta\, x(t),
\qquad
\varphi^{\star}(t) = s_\varphi\, y(t),
\qquad
\theta^{\star}(0)=\varphi^{\star}(0)=0
```

**Step 1 — what the circuit is given.** Only the velocity, never the
position. This is the open-loop premise of section 2.4 written as a vector:

```math
\mathbf{u}(t) \;=\; \big(\dot x(t),\ \dot y(t)\big)^{\!\top} \in \mathbb{R}^{2}
```

**Step 2 — the input map $\hat W^{\mathrm{in}}$.** The drive enters only
through the AF5 afferents $\mathcal{A}$ ($|\mathcal{A}|=41$). That restriction
is the whole content of "connectome-constrained input": a free input layer
would let the optimiser inject velocity wherever it is most convenient, which
is exactly the degree of freedom the anatomy is supposed to remove:

```math
\mathbf{I}(t) \;=\; \hat W^{\mathrm{in}}\,\mathbf{u}(t),
\qquad
\hat W^{\mathrm{in}}_{i,:} = \mathbf{0}\ \ \ \forall\, i\notin\mathcal{A},
\qquad \mathcal{A}=\mathcal{A}_{\mathrm{AF5}}^{L}\cup\mathcal{A}_{\mathrm{AF5}}^{R}
```

so $\hat W^{\mathrm{in}}$ is $285\times 2$ with $41\times 2 = 82$ free entries
out of 570. Componentwise, neuron $i$ receives
$I_i = \hat W^{\mathrm{in}}_{i1}\dot x + \hat W^{\mathrm{in}}_{i2}\dot y$, and
the direction tuning of section 1.3 — still a CLAIM — is what says whether
that row should be free or fixed to a preferred direction.

**Step 3 — the circuit.** The $N=285$ neurons obey the same rate equation as
Eq. (1) of `neurips.tex`:

```math
\tau_i\frac{d v_i(t)}{dt} \;=\; -\,v_i(t) \;+\; V_i^{\mathrm{rest}}
\;+\!\!\sum_{j\in\mathcal{N}_i}\!\! \hat{W}_{ij}\,r_j(t)
\;+\; I_i(t) \;+\; \sigma\,\xi_i(t),
\qquad r_j=\rho(v_j)
```

with $\mathcal{N}_i$ the presynaptic partners of $i$ and $\rho$ the rate
nonlinearity — $\mathrm{ReLU}$ in the constrained model, $\tanh$ in the
prototype of section 4.3.

**Step 4 — the sign lock.** Only the magnitudes are learned; the signs are
the measured Dale assignment of section 1.4, plotted in panels (c) and (d) of
Figure 1, so $\hat W$ never leaves the connectome's sign pattern:

```math
\hat{W}_{ij} \;=\; \big|\hat{S}_{ij}\big|\;\mathrm{sign}\big(W^{\mathrm{con}}_{ij}\big),
\qquad
\mathrm{sign}\big(W^{\mathrm{con}}_{ij}\big)=
\begin{cases}
-1, & j\in\mathcal{I} \quad(\text{INTG contra, } 52 \text{ cells})\\[2pt]
+1, & \text{otherwise} \quad(233 \text{ cells})
\end{cases}
```

**Step 5 — the output map $\hat W^{\mathrm{out}}$.** The motor pools are read
as non-negative drives, one per muscle, ordered LR, MR, SR, IR:

```math
\mathbf{m}(t) \;=\; \big[\,\hat W^{\mathrm{out}}\mathbf{r}(t)\,\big]_{+}
\;=\;\big(m_{\mathrm{LR}},\,m_{\mathrm{MR}},\,m_{\mathrm{SR}},\,m_{\mathrm{IR}}\big)^{\!\top},
\qquad
\hat W^{\mathrm{out}}_{:,i} = \mathbf{0}\ \ \ \forall\, i\notin\mathcal{M}
```

with $[\cdot]_+$ a rectifier ($\mathrm{softplus}$ in the prototype) enforcing
$m_\bullet\ge 0$, and $\mathcal{M}$ the motor pool. In the selected 285 cells
$\mathcal{M}$ is AMN (92 cells, row LR) and AIN (35 cells, row MR) only, so
$\hat W^{\mathrm{out}}$ is $4\times285$ with two live rows and $127$ live
columns. **The vertical pair has no anatomical pool here**: SR and IR are
innervated by OMN, which section 1.5 leaves out of the selection. The
two-axis form is written because the plant and the prototype are two-axis;
the constrained circuit as currently scoped commands $\theta$ alone, and
reaching $\varphi$ means adding OMN to `circuit.cell_types`.

**Step 6 — push-pull.** Each axis is the difference of an antagonist pair, so
a signed angle is carried on strictly non-negative rates by construction
rather than by a free linear readout that might or might not discover it
(hazard 4 of section 4.2):

```math
u_\theta(t) = m_{\mathrm{LR}}(t) - m_{\mathrm{MR}}(t),
\qquad
u_\varphi(t) = m_{\mathrm{SR}}(t) - m_{\mathrm{IR}}(t)
```

**Step 7 — the plant, per axis.** Each command drives a Hammerstein eye: a
monotone static nonlinearity followed by second-order mechanics, with
$\Phi$, $\omega_n$ and $\zeta$ identified in section 4.7 and **frozen** —
steps 6 and 7 together are what this note calls the **lateral-horizontal
Hammerstein model**, and section 4.8 says what has to replace it —
registered as buffers, not parameters, because the body is measured and
letting the optimiser retune it would defeat the point:

```math
\Phi_\theta(u_\theta)=a_\theta\,u_\theta + b_\theta\,u_\theta^{2},
\qquad
\ddot\theta+2\zeta_\theta\omega_\theta\,\dot\theta+\omega_\theta^{2}\,\theta
\;=\;\omega_\theta^{2}\,\Phi_\theta\big(u_\theta(t)\big)
```

```math
\Phi_\varphi(u_\varphi)=a_\varphi\,u_\varphi + b_\varphi\,u_\varphi^{2},
\qquad
\ddot\varphi+2\zeta_\varphi\omega_\varphi\,\dot\varphi+\omega_\varphi^{2}\,\varphi
\;=\;\omega_\varphi^{2}\,\Phi_\varphi\big(u_\varphi(t)\big)
```

The two axes have their own $(\Phi,\omega_n,\zeta)$, fitted from the LR/MR and
the SR/IR probes separately; on eye C they differ by more than a factor of
one and a half in travel, which is why $s_\theta \neq s_\varphi$ in step 0.

**Step 8 — the objective.** Supervision is on the eye angles *after* the
plant, never on the command, in the same summed form as Eq. (3) of
`neurips.tex`:

```math
\mathcal{L}_{\mathrm{pred}}=\sum_{t}
\Big\|\big(\theta(t),\varphi(t)\big)-\big(\theta^{\star}(t),\varphi^{\star}(t)\big)\Big\|_2
```

Putting the loss after the plant is what makes the network learn the eye's
inverse rather than a velocity command: the gradient reaches
$\hat W,\hat W^{\mathrm{in}},\hat W^{\mathrm{out}}$ through $\Phi$ and through
the second-order mechanics, so a different eye trains a different controller.
That is why section 4.7 lists one trained network per eye rather than one
network tested on five.

**Step 9 — discretisation and initial condition.** Forward Euler at
$\Delta t = 1/60$ s, with the network state started at rest on every trial:

```math
v_i(t{+}\Delta t) = v_i(t) + \frac{\Delta t}{\tau_i}
\Big(-v_i(t) + \textstyle\sum_j \hat W_{ij} r_j(t) + I_i(t)\Big),
\qquad v_i(0)=0,\ \ \theta(0)=\varphi(0)=0
```

$v_i(0)=0$ has to *mean* "eye centred", which is why every trajectory in the
corpus starts at the centre (section 4.5). Nothing is carried between trials.

Two differences from the flyvis setting are worth naming. There the
supervision is on $dv_i/dt$ of *every* neuron, because the simulator provides
the full state; here it is on two scalars at the far end of a plant, so the
circuit is constrained only through its behavioural consequence. And there
$\hat{W}$ is what is being recovered from activity, whereas here $\hat{W}$ is
fixed in sign by the connectome and only its magnitudes,
$\hat W^{\mathrm{in}}$ and $\hat W^{\mathrm{out}}$ are free. The oculomotor
problem is thus the better-posed of the two: fewer unknowns, but a much
narrower observation.

**What the prototype of section 4.3 instantiates.** The same nine steps with
the connectome removed, which is the point of stage 1: $N=64$ free units in
place of the 285 cells, $\rho=\tanh$, $\hat W$ dense and initialised at zero
with no sign lock, $\hat W^{\mathrm{in}}$ a dense $64\times2$ layer instead of
82 entries on AF5 rows, $\hat W^{\mathrm{out}}$ a dense $4\times64$ layer with
all four muscle rows live. Steps 6 to 9 — push-pull, the frozen per-axis
plant, the loss on $(\theta,\varphi)$, the Euler step from $v=0$ — are
identical, so the gap between the two is attributable to the anatomy and to
nothing else.

### 4.7 The eye model

The circuit's output is muscle drive, not eye position, so at some point the
mechanics has to be in the loop. The MPM eye of `Plexus/prototype/eye` cannot
be: its grid scatter is in-place and a single trial costs far more than the
7250 gradient updates of section 4.5. It is replaced by a reduced plant,
identified from the probe runs in that prototype's `archive/` by
`fit_plant.py`, and it is the plant of step 7 above — a static nonlinearity
$\Phi$ followed by second-order mechanics, per axis. Steps 6 and 7 together
are the **lateral-horizontal Hammerstein model**: two antagonist pairs, two
scalar commands, two independent single-input plants. This section says where
its numbers come from and what they are worth; the equations are not
repeated.

#### Why Hammerstein, and where it comes from

Command-to-gaze is plainly nonlinear: activation saturates, and the two
horizontal muscles are not mirror images. That does not force a nonlinear
ODE. Putting the whole nonlinearity in a memoryless block and leaving the
dynamics linear is the **Hammerstein** structure, and it is not an
improvisation for this note — it is the standard first move of *block-oriented
nonlinear system identification*, a field with fifty years of theory behind
it. The name is Adolf Hammerstein's, from the nonlinear integral equations he
studied around 1930; the four canonical cascades are Hammerstein (static
nonlinearity then LTI), Wiener (LTI then static nonlinearity),
Wiener-Hammerstein (LTI, nonlinearity, LTI) and Hammerstein-Wiener. The
reason they dominate practice is identifiability rather than expressiveness:
the linear factor remains a transfer function, with poles, a frequency
response and every tool of linear systems theory intact, while the nonlinear
factor is a curve that can be fitted pointwise. The classical identification
algorithms are Narendra and Gallman's iterative scheme (IEEE TAC, 1966) and
the overparameterisation / two-stage SVD methods that followed (Bai,
*Automatica*, 1998); the standard collection is Giri and Bai (eds),
*Block-oriented Nonlinear System Identification*, Springer LNCIS 404, 2010,
and the structure ships in MATLAB's System Identification Toolbox as `nlhw`.

For an eye the split is not merely convenient, it is anatomical. The
nonlinearity lives in the muscle — force-activation, the Hill length-tension
and force-velocity relations, a moment arm that changes with eye position,
and the geometric saturation of a globe that can only rotate so far — and on
the timescale that matters it is memoryless. The dynamics live in the globe
and the orbital tissue: inertia, viscosity and an elastic restoring force,
approximately linear for rotations this small. This is also the classical
oculomotor model, not a new proposal: Robinson's mechanics of human saccades
(*J. Physiol.*, 1964) and his control-systems review (*Annu. Rev. Neurosci.*,
1981) set the linear-plant description and the pulse-step command that
inverts it, and later work argued the real plant carries a wider spread of
time scales than a second-order fit allows (Sklavos, Porrill, Kaneko and
Dean, *Vision Research*, 2005). Our own $\zeta \approx 0.3$ is the one place
we are clearly off the biology, as noted below.

The factorisation earns its keep here — across the
five eyes the mechanics barely move, $\omega_n = 9.5$ to $11.1$ rad s$^{-1}$
and $\zeta = 0.22$ to $0.32$, while the static gain ranges over 14 to 23
degrees of travel. The configurations change $\Phi$, not the ODE, so one set
of dynamics covers the sweep. At $\zeta \approx 0.3$ the modelled eye is
underdamped and rings at about 1.5 Hz; real oculomotor plants are overdamped,
so that is a property of the MPM configuration rather than of a fish.

$\Phi$ and the mechanics are fitted **jointly**, against whole trajectories.
The alternative — read the plateaus for $\Phi$, then fit the transient — needs
settled plateaus, and this archive has none, for the reason given below.

![**The identified eye plants.** All three panels are drawn from the stored fit in `plant.npz` / `plant_v.npz`, in the symbols of section 4.6. **(a)** The static nonlinearity $\Phi_\theta(u_\theta)$ of all five eyes, and $\Phi_\varphi$ of eye C, the one the controller is coupled to. Every curve is monotone by construction; the curvature is the genuine asymmetry between abduction and adduction. **(b)** Step response of eye C on both axes at commands 0.5 and 1: the overshoot and the ringing are what only the inertial term can produce, and they are the reason the second-order plant beats the first. **(c)** Reachable travel per axis, $\min|\Phi(\pm 1)|$ — the quantity that decides whether a tracking task is expressible at all, and the reason the world is scaled anisotropically. Regenerate with `python fig_plant_summary.py` in `Plexus/prototype/eye`.](../figures/zebrafish/fig_eye_plant.png)

**Second order wins on every variant, by a factor of 3.** Against the
first-order alternative $\tau\dot\theta + \theta = \Phi_\theta(u_\theta)$ —
viscosity and an elastic restoring force, no globe inertia — the RMS error in
degrees is:

| variant | order 1 | order 2 | ratio |
|---|---|---|---|
| **B** `baseline_fixmat` | 1.22 | **0.41** | 3.0x |
| **A** `c_a` | 1.20 | **0.42** | 2.9x |
| **C** `p3a_length` | 2.58 | **0.62** | 4.2x |
| **D** `p3b_pulley` | 2.24 | **0.77** | 2.9x |
| **E** `p3c_drive` | 6.77 | 6.34 | 1.1x |

This is not a matter of degrees of freedom. A first-order system is monotone
towards its target and *cannot overshoot*, while the MPM eye overshoots and
rings — panel (b). Only the inertial term can produce that. The one variant
where the two orders are indistinguishable, `p3c_drive`, is also the one
neither fits at 6 degrees of error, which is a signal about that run rather
than about the model order. And the fit has to be **per variant**: pooled
across all five it gives RMS 5.05 deg and a static curve with plateaus at
+16, +13 and +3 degrees for the same command; fitted separately, 0.41 to
0.77 deg.

#### The five eyes

The archive is a sweep over mechanical configurations, not repeats of one
globe. They are labelled A-E in the order they were made, and the folders
`Plexus/prototype/eye/archive/eye_{A..E}/` collect each one's probe runs,
movies and curves.

| eye | variant | what changed | horizontal travel |
|---|---|---|---|
| **A** | `eye_probe_c_a` | soft tissue: sclera Young's 300, tendon 9, bone cap 40 — the softest globe | -10.1 / +3.4 deg |
| **B** | `eye_probe_baseline_fixmat` | same geometry, materials stiffened to 420 / 45 / 130. The reference eye | -9.7 / +3.2 deg |
| **C** | `eye_p3a_length` | B with the muscle gap widened 0.020 to 0.042 and strap fraction 0.55 to 0.95 — more contractile length, the largest travel | -14.4 / +15.1 deg |
| **D** | `eye_p3b_pulley` | C plus a `muscle_sleeve` constraint (k 2500, c 30, free 0.70-0.88): a connective-tissue pulley holding the strap to the globe | -11.4 / +12.2 deg |
| **E** | `eye_p3c_drive` | D with the drive amplitude raised 60 to 67 | -4.5 / +13.5 deg |

A and B are strongly asymmetric — barely 3 degrees of abduction against 10 of
adduction — so neither can serve a tracking task whatever its fit quality. C
is the one to couple to: nearly symmetric and the widest workspace.

#### A correction: there is no steady state in this archive

The first version of this fit used a free cubic and produced a static curve
whose slope at the origin was **negative** — pulling the lateral rectus would
rotate the eye medially. That is not a finding about the eye, it is an
artefact, and it invalidated everything downstream of it.

The cause was mundane. `static_curve` reads plateaus as steady states, but at
$\omega_n = 10$ rad s$^{-1}$ and $\zeta = 0.31$ the settling time
$4/(\zeta\omega_n)$ is 1.28 s, while the holds in these runs last **0.19 s,
1.27 s and 0.24 s**. Not one hold in the archive has settled; the two short
ones are still moving at 70 deg/s when sampled. What was called a static curve
was measured entirely from transients, and the post-step return — the eye
sailing back through zero after a full step was released — was being read as
"small positive command gives negative gaze".

$\Phi$ is now **monotone by construction**: in the quadratic of step 7 the
linear coefficient is parameterised as $a = e^{p} > 0$ and the quadratic one
as $b = \tfrac{a}{2}\tanh q$, which forces
$\Phi_\theta'(u_\theta) = a + 2b\,u_\theta > 0$ across the whole command range
$u_\theta \in [-1,1]$, and likewise in $\varphi$. The fit therefore cannot return a physically
impossible curve whatever the data does, while the quadratic term still
carries the genuine asymmetry between abduction and adduction. Every variant
now has a positive slope at the origin and zero non-monotone fraction. The
residuals rise a little — 0.41 to 0.72 deg on B — and that increase is the
honest cost of refusing to fit transients.

The identification still needs re-running with **holds longer than 1.3 s at
intermediate levels** (0.2 / 0.4 / 0.6 / 0.8 in both directions). Each
protocol steps straight to full activation, so there are only two plateaus per
variant, at command $-1$ and $+1$, and the curve between them is extrapolation
constrained to be monotone rather than measurement. The dynamics are well
determined, because those come from the transients; the static gain is not.

**A caveat on reproducibility.** The `archive/t*_probe_*/curves.npz` behind
eyes A-E have since been deleted, so `fit_plant.py` can no longer be re-run on
them and the figure above is drawn from the stored coefficients instead. The
fit itself survives — `plant.npz` and `plant_v.npz` carry $\Phi$, $\omega_n$
and $\zeta$ per variant, which is what the trained controllers use — but the
staircase re-probe will have to regenerate the runs from the specs.

#### Coupling the circuit to the eye: what the geometry forces

Driving the eye is not a matter of appending a plant to the readout. Two
constraints come from the mechanics and neither is negotiable by training.

**Four pools, two antagonist pairs.** The readout emits four non-negative
motor drives and each axis is commanded by a difference, which is step 6
above. A single command cannot serve both axes: they have different static
curves and different mechanics, fitted separately from the LR/MR and the
SR/IR probes. On eye C they differ by more than a factor of one and a half in
travel.

**The workspace is per axis, and it is the binding constraint.** Reachable
travel, in degrees, plotted in panel (c):

| eye | horizontal | vertical | usable world |
|---|---|---|---|
| A | 3.4 | 4.1 | 3.6 deg/unit |
| B | 3.4 | 4.1 | 3.6 |
| C | **15.0** | **9.1** | 9.5 |
| D | 12.3 | 5.0 | 5.3 |
| E | 5.1 | 5.9 | 5.3 |

An isotropic task can only be scaled to the *smaller* of the two. Scaling it
to the horizontal instead — which an implementation does by default, because
the horizontal pair is the one everybody models — puts the vertical target
outside the eye's travel 40 % of the time on eye C and 60 % on eye D. The eye
then cannot look where the target is, and the resulting error is
indistinguishable, in any plot, from a controller that has failed to learn.
The prototype avoids it by scaling each axis to its own reach, which is the
$s_\theta \neq s_\varphi$ of step 0.

That is the transferable point for the connectome model. Before any training
result is interpreted, the reachable range of each output axis has to be
compared against the eccentricity the task demands of it. A circuit whose
motor pools cannot produce the required rotation will look exactly like a
circuit that cannot compute — and only the first of those is fixed by anatomy
rather than by optimisation.

#### Co-contraction is a second input, not a second value of the first

Step 6 writes each axis as one signed scalar, which assumes reciprocal
innervation: one muscle pulls, the other releases. If instead both pull — the
lateral rectus leading and the medial rectus resisting — the difference is
unchanged but the **sum** is not, and the sum is not a position command at
all. Co-contraction raises the stiffness and the damping of the plant, so it
moves $\omega_n$ and $\zeta$ rather than $\Phi$.

A single-input Hammerstein cannot express that. Capturing it needs
$\Phi(m_{\mathrm{LR}}, m_{\mathrm{MR}})$ together with
$\omega_n(m_{\mathrm{LR}} + m_{\mathrm{MR}})$, identified from a probe that
sweeps the sum as well as the difference. This is not academic for the present
model: the motor readout already emits two independent non-negative drives per
axis, so the network is free to co-contract, and the plant will ignore it —
the capacity is there and its effect is silently discarded.


### 4.8 Characterising the MPM eye, from now on

The lateral-horizontal Hammerstein model of steps 6 and 7 assumes the eye is
two independent axes driven by two signed scalars. The first properly settled
measurements say it is not. On eye F the lateral rectus produces 3.86 degrees
of torsion at full drive against 6.17 of horizontal — 63 % — and the inferior
oblique produces 11.08 degrees of torsion, larger than any horizontal action
in the plant. A model with no torsion coordinate cannot represent that, and a
model with two commands cannot be told about it. This section is the
replacement, and the protocol that identifies it.

The aim is a procedure rather than a model of one eye. Eye F will be replaced,
eye G is coming, and the point of writing the protocol down is that neither
should require re-deriving anything here.

#### The shape of the model, coarsest first

The eye takes six muscle drives and returns three angles:

```math
\mathbf{m}(t)\in[0,1]^{6}
\;\longrightarrow\;
\mathbf{x}(t)=\big(\theta(t),\ \varphi(t),\ \psi(t)\big)\in\mathbb{R}^{3}
```

horizontal, vertical and torsion, in degrees. The drives are one-sided
because muscles pull and do not push, which is why they are not the signed
$u_\theta, u_\varphi$ of step 6.

The single structural assumption is that the eye is a **static map followed by
linear mechanics**. Where the eye eventually comes to rest is a nonlinear
function of the drives; how it gets there is linear:

```math
\mathbf{x}_{\infty}=g(\mathbf{m}),
\qquad
\ddot{\mathbf{x}}+C\,\dot{\mathbf{x}}+K\,\mathbf{x}=K\,\mathbf{x}_{\infty}
```

In plain terms: hold the muscles at some fixed activation and the eye settles
somewhere — that is $g$. Change the activation and the globe swings to the
new resting place with an overshoot and a ring — that is $C$ and $K$. It is
the same factorisation as section 4.7, widened from one input and one angle to
six inputs and three angles, and it is what makes the whole thing measurable:
$g$ is memoryless, so **it can be measured entirely from holds**, with no
differential equation anywhere in the fit.

That is not our idea. It is the block-oriented structure whose provenance
section 4.7 gives — Hammerstein's cascade, the identification literature from
Narendra and Gallman (1966) through Bai (1998) to Giri and Bai (2010) — and
for an eye it is also the classical description, Robinson (1964, 1981). What
is new here is only that the blocks are multi-input and multi-output.

#### The static map, specifically

Six inputs is too many to write in closed form and too few for a blind neural
network to fit from an affordable number of simulations. The standard way out
is to decompose the function by interaction order — the **functional ANOVA**
decomposition, due to Hoeffding (1948) and made a practical tool by Sobol'
(1993), which is also the basis of the additive models of Hastie and
Tibshirani (1986):

```math
g(\mathbf{m})=
\underbrace{\sum_{i=1}^{6}\phi_i(m_i)}_{\text{one muscle at a time}}
\;+\;
\underbrace{\sum_{i<j}\phi_{ij}(m_i,m_j)}_{\text{pairs}}
\;+\;
\underbrace{\eta(\mathbf{m})}_{\text{everything else}},
\qquad \phi_i:[0,1]\to\mathbb{R}^{3}
```

Read left to right this says: most of what the eye does is each muscle acting
on its own, some of it is two muscles fighting or helping each other, and
whatever remains is a small correction. The first term is six curves, each
measured by driving one muscle alone. The second is fifteen surfaces. The
third is a small neural network regularised towards zero, present so that the
model can absorb what the first two miss rather than pretend it does not
exist.

The reason this matters practically is sample size. A neural network on the
six-dimensional cube would need thousands of simulated holds; the marginals
need thirty, and the pairs are decided one at a time. The decomposition is
what turns an unaffordable experiment into a two-hour one.

Nothing in $g$ is constrained to be monotone. The negative-slope disaster of
section 4.7 was caused by fitting transients as though they were plateaus, not
by the fitting method, and with genuinely settled holds the constraint is
unnecessary. It also turns out to have been actively harmful: the monotone
parameterisation caps the curvature at $|b|\le a/2$, and eye F's horizontal
recti need $b/a=0.65$, so enforcing it doubles the residual from 0.54 to 0.95
degrees. Monotonicity is now **checked and reported**, not imposed.

#### The mechanics, specifically

$C$ and $K$ are three-by-three, so the model can express one axis dragging on
another, which the two independent second-order plants of section 4.7 cannot.
They are parameterised through their Cholesky factors,

```math
K=L_K L_K^{\!\top}+\varepsilon I,
\qquad
C=L_C L_C^{\!\top}
```

which makes them positive definite and the plant therefore **stable by
construction**, for any values the optimiser reaches. This replaces the
per-eye supervision of $\omega_n$ and $\zeta$: instead of assuming two
numbers, the eigenvalues of the fitted pair are reported, and if the eye turns
out to need more time scales than two — which Sklavos, Porrill, Kaneko and
Dean (2005) argue real oculomotor plants do — that shows up as a poor fit
rather than as a silently wrong assumption.

One optional block covers co-contraction, the failure of section 4.7 that no
single-input model can express. Pulling two antagonists together leaves the
difference unchanged and raises the stiffness, so the mechanics are allowed to
depend on the total drive $s=\mathbf{1}^{\!\top}\mathbf{m}$:

```math
K(s)=K_0\,(1+\kappa_K s),
\qquad
C(s)=C_0\,(1+\kappa_C s)
```

This is a linear-parameter-varying plant, and it is included only if the
measurement below says it is needed.

#### The protocol

Five stages, in `Plexus/prototype/eye/PROTOCOL_eye_characterisation.md`. Two
choices in it are worth stating here because both were paid for in lost data.

**Stage 0, the gate — six runs.** Each muscle alone at full drive. This
returns the reachable span per axis and the settling time. The controller
needs 25 degrees horizontal and 10 vertical; an eye that cannot reach that
cannot do the task, and characterising it is wasted compute. Eye F fails —
7.9 degrees horizontal from single-muscle extremes, 10.8 even allowing every
muscle to co-activate helpfully, against 25 — so it is the worked example of
the gate doing its job. Stage 0 also sets the hold length for everything that
follows, $T_{\rm hold}=\max(2\ {\rm s},\,1.5\times{\rm settling})$, derived per
eye. A fixed constant is exactly how eyes A to E came to be fitted entirely
from transients.

**Stage 1, the marginals — thirty holds.** Each muscle alone at
$m_i\in\{0.10, 0.25, 0.50, 0.75, 1.00\}$. Five levels weighted to the low end,
because eye F's nonlinearity is strongly convex — the lateral rectus gains
6.9 times more per unit drive at $m=1$ than at $m=0$ — so the shape lives near
the origin, which is also where a tracking controller spends its time.

**Stage 2, which pairs matter — fifteen holds, then nine per pair that does.**
Stage 1 gives an additive prediction for any combination. Drive each of the
fifteen pairs at $m_i=m_j=0.5$ and compare; a pair whose residual exceeds
0.2 degrees, four times the settling tolerance, gets a three-by-three grid,
and a pair below it gets nothing more. This is interaction screening in the
sense of classical design of experiments (Box, Hunter and Hunter), and it is
where the saving is: a pair that does not interact becomes a recorded
measurement instead of an untested assumption, at a cost of one run.

**Stage 3, the mechanics — about twenty-five trajectories.** Steps from rest
in many directions, single-muscle frequency sweeps to pin the damping, and
three matched pairs that reach the same final angle with different total drive
— the last of these being the measurement that decides whether $K(s)$ and
$C(s)$ are needed at all.

**Stage 4, fit and select.** $g$ from the holds alone, by ordinary regression;
$C$ and $K$ from the trajectories with $g$ frozen; then a short joint
refinement of both against the trajectories, which is what rescued the eye C
fit. Then a nested comparison on held-out runs: marginals against
marginals-plus-pairs against plus-network, diagonal against full $C,K$,
constant against drive-dependent. **This selection step is what makes the
procedure general.** Every eye runs the same comparison and the data chooses
how much structure that eye needs, so eye G requires re-running the selection
and not rewriting the model.

#### What it costs

About 112 runs against the 200 a blind low-discrepancy sweep of the cube would
take, the saving coming entirely from stage 2. Measured on the target
hardware, the whole protocol is **two hours on eight L4 GPUs**, which puts a
full re-characterisation within a single working session and means an eye can
be changed and re-measured in the same day.

#### What it changes upstream

Two things in section 4.6, and they are simplifications rather than
complications.

**Step 6 disappears.** There is no push-pull and no $u_\theta, u_\varphi$; the
readout emits six non-negative drives straight to the plant,

```math
\mathbf{m}(t)=\big[\hat W^{\rm out}\mathbf{r}(t)\big]_{+},
\qquad
\hat W^{\rm out}\in\mathbb{R}^{6\times N}
```

**Step 8 gains one term.** Six drives against two supervised angles leaves a
four-dimensional set of muscle patterns producing the same gaze, and the
network will wander in it arbitrarily. Penalising torsion pins it down:

```math
\mathcal{L}=\sum_{t}\Big\|\big(\theta,\varphi\big)-\big(\theta^{\star},\varphi^{\star}\big)\Big\|_2
\;+\;\lambda_\psi\sum_{t}\psi(t)^{2}
```

This is not an arbitrary regulariser. Real eyes resolve the same redundancy by
holding torsion to a fixed function of gaze direction — Donders' law, and its
sharper form Listing's law — and the term above is its simplest version. The
three-dimensional treatment of eye rotations that makes this precise is
Tweed and Vilis (1990) and Haslwanter (1995); adopting the full form later
costs nothing extra, because the torsion coordinate is now in the model.

#### When the eye is good enough

The same numbers for every eye, reported by the fit: the fraction of holds
that settled, held-out RMS per axis in degrees, the fraction of the command
cube where the fitted map is monotone in each muscle's own dominant axis, the
reachable span per axis against the 25 and 10 degrees the task needs, the
eigenvalues of $(C,K)$, and the size of the interaction and residual terms
relative to the marginals. An eye is usable when the span passes, the settled
fraction is near one, and the held-out error is small against the precision
the tracking task is scored at. Those are the criteria eye F fails on the
first, and they are the criteria eye G will be judged by without anything in
this section changing.


## 5. Progress, 11 August 2026

Opened the branch `feat/oculomotor` and put the zebrafish configs back under
version control — 254 of them had been absent from every worktree since the
`config/zebrafish` bind-mount was commented out of `devcontainer.json`, which
left no figure script able to resolve a run config. Traced how the 917-cell
heading-direction circuit was actually built (the colleagues' pickle, not the
neuPrint query the filenames suggest) and wrote that up as
`HOWTO_add_oculomotor_circuit.md`, since the same five inputs are needed
again here. Read the new `Oculomotor_sortedData_081126.pkl`: 2949 cells, 42
types, 5 keys, no per-cell ordering variable; confirmed its matrix
orientation empirically rather than assuming it. Selected the sub-circuit in
two steps — first the 244-cell INTG/AMN/AIN core, then 285 cells once AF5 was
added, which gave the pool the afferent stage it had been missing.
Introduced `CellTypeSpec` so each type's pool, Dale sign, L/R split and role
live in the config where they can be reviewed, rather than in hardcoded
prefix tuples inside the loader. Rendered the two-panel connectivity figure
above, and wrote this note.

Nothing has been trained, and no connectome CSVs exist yet. The five
decisions below are what stand between this document and a first run.

## 6. What has to be decided next

1. **The AF5 combination rule** (§1.3) — AND, OR or XOR. Blocks the input model.
2. **Whether `OMN` joins the pool** (§1.5) — decides whether `AIN` -> medial
   rectus is one synapse or two.
3. **Whether `Burst_*` joins the pool** (§1.6) — decides whether the saccadic
   regime has an anatomical input or an injected one.
4. **The readout convention** — scalar eye position, or innervation of the
   `LR`/`MR` pair coupled to the soft-body plant.
5. **The target leak `xi_tau_s`** — a free parameter, or the quantity to be
   fitted against recorded eye position.
