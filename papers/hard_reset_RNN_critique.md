# Hard Reset in RNN-Based Neuroscience Models: A Critique

## The Problem: Hard State Reset Between Trials

In the Ashok et al. code (and virtually all computational neuroscience RNN work), neural activity is **hard-reset** to a fixed initial state `h0` at the start of every trial.

### Evidence from Ashok Code

#### h0 Initialization (`fun_lib2.py`, `nn_fig5_drosophilaCx_teacher.py`)

The default code path in `fun_lib2.py` (lines 165-167) has a bug:
```python
self.h0.zero_    # line 166 — NOT a function call! Missing (). This is a no-op.
self.h0.fill_(1) # line 167 — sets all values to 1
```
`self.h0.zero_` references the method without calling it (missing parentheses `()`), so it does nothing. Then `fill_(1)` sets all values to 1. However, this fallback path is never reached in practice because `h0_init` is always provided.

In the Drosophila experiment (`nn_fig5_drosophilaCx_teacher.py`, lines 569-573):
```python
x0 = 0.05 * np.random.randn(N) - 0.1
```
So `h0_init = x0` — small random values centered around -0.1 (mean ≈ -0.1, std ≈ 0.05). Since `r = softplus(h + b, beta=5)` and `b` starts at `0.1 * randn(N)`, the initial firing rates are `softplus(-0.1 + b_j, 5)` — close to zero for most neurons.

`h0` is trainable (`train_h0=True`), so it evolves during training, but it remains a single shared initial state for all trials.

#### The Reset Mechanism (`fun_lib2.py`, lines 172-193)

There is no separate "neural activity" vs "h0". `h` IS the neural activity (membrane potential), and `r = softplus(h + b)` is the derived firing rate. At each `forward()` call (= each trial):

1. `h = self.h0` — overwrite neural activity with learned initial state (line 174)
2. `r = softplus(h0 + b)` — compute initial firing rate (line 175)
3. Run dynamics for `seq_len` steps (lines 186-189)
4. **Discard final state** — `h` at trial end is never saved or carried forward

Every trial starts from the same learned initial state `h0`, regardless of where the previous trial ended. There is no mechanism to carry state between sequences.

### Quantitative Analysis of the Ashok Paper

- **Trial length**: T = 6 with dt = 0.1 → 60 time steps per trial
- **State reset**: h = h0 at the start of each trial (line 187: the loop iterates over `seq_len` within a trial, and `h` is re-initialized from `h0` each trial)
- **Decay time constant**: Tau = 1.0 (line 603), alpha = dt/Tau = 0.1/1.0 = 0.1. The effective tau per neuron is bounded [0.2, 5.0] via the tanh parametrization (line 183), initialized at tau = 2.0 for all neurons (line 573: `arctanh((2 - 2.6)/2.4) = arctanh(-0.25) ≈ -0.255`, giving `tau = 2.6 + 2.4*tanh(-0.255) ≈ 2.0`)
- **Ratio**: Trial duration is T = 6 time units, decay time constants are τ ∈ [0.2, 5.0] (init ~2.0). The trial is only **1.2–30x the decay time** (and ~3x at initialization)

This means the slowest neurons (τ ≈ 5.0) have only experienced 6/5 = 1.2 time constants by trial end. They are still ~30% influenced by `h0` when the reset hits. The network is essentially learning to exploit the reset-to-transient trajectory rather than learning continuous dynamical computations.

The reset is a BPTT convenience (truncated backprop through 60 steps), not a feature of the biological circuit.

## Why Hard Resets Are Used (Computational Convenience)

1. **Gradient flow**: BPTT with hard resets keeps sequences short and bounded. Without resets, gradients must flow through the entire continuous history, making vanishing/exploding gradients much worse.
2. **Batch parallelism**: Hard resets let you batch independent sequences together. With continuous state, sequences are causally linked and must be processed serially.
3. **Loss landscape**: A fixed `h0` gives a deterministic starting point. Without it, the loss depends on the full history of prior inputs, making optimization much harder — the loss surface becomes non-stationary.

## Biological Concern

Real neural circuits don't reset. A fly's visual system processes a continuous stream — the response to stimulus B depends on what stimulus A was. The hard-reset approach effectively assumes each trial is independent, which discards temporal context effects (adaptation, habituation, afterimages, etc.).

## Consequences for the Ashok Paper

### Internal Consistency

The teacher network (ground truth) also uses hard resets. So the student is learning to match a system that itself resets. In that closed world, the approach is internally consistent — the student correctly learns the teacher's input-output mapping.

### Where It Becomes a Real Flaw (Biological Interpretation)

1. **The teacher is not a real neural circuit** — it's an RNN with hard resets. Any conclusions about "how biological circuits work" are conclusions about how reset-RNNs work, not how continuous neural systems work.

2. **Transient dynamics matter enormously** — after a hard reset, the network goes through a stereotyped transient from `h0` to its attractor. A continuous system would already be near some attractor and respond differently to the same stimulus depending on recent history. The learned connectivity optimized for post-reset transients may be qualitatively different from connectivity needed for continuous operation.

3. **The Drosophila connectome experiments (Fig 5)** — if they constrain the student with real connectome structure but train with hard resets, the learned parameters (gains, time constants) are optimized for a regime that doesn't exist in the animal.

### Concrete Consequences for Learned Parameters

- **Learned time constants are biased** — the optimization pushes τ values to fit the trial structure (T=6), not to match biological time constants. A neuron might learn τ=5 not because the circuit needs slow integration, but because a long transient from `h0` happens to produce the right output trajectory within the 60-step window.
- **Learned gains (g) and connectivity (W) compensate for the reset** — the network needs to rapidly drive activity from `h0` to a useful regime. In a continuous system, you don't need that initial "kick" — the weights and gains would be optimized for a different operating point.
- **The readout conflates transient and computation** — since the output is evaluated across the full trial including the early transient, the readout weights (`wout`) are partially fitted to the reset artifact.
- **For Fig 5 (Drosophila connectome)**: if they conclude that certain connectome motifs are necessary for a computation, those motifs might actually be necessary for the reset-recovery dynamics, not the computation itself. A continuous model with the same connectome might need different gain/tau parameters or even different motifs.

## Hard Reset Is Universal in Computational Neuroscience RNNs

Nearly every major paper in the field does this:

- Sussillo & Barak (2013) — opening the black box of trained RNNs — hard reset per trial
- Mante et al. (2013) — context-dependent computation in prefrontal cortex — hard reset
- Rajan, Harvey, Tank (2016) — recurrent network models of sequence generation — hard reset
- Yang et al. (2019) — task representations in neural networks trained on many cognitive tasks — hard reset
- Cueva & Wei (2018) — grid cells from RNNs — hard reset
- Sussillo et al. (2015) — neural circuit inference from RNNs — hard reset
- The entire MotorNet / neurogym / PsychRNN ecosystem — hard reset

The standard recipe is always: fixed h0 → run trial (50-200 steps) → compute loss → BPTT → reset → next trial.

### Why It's So Entrenched

1. It directly mirrors experimental neuroscience — trials with inter-trial intervals, where the assumption is "the animal resets between trials"
2. PyTorch/TensorFlow RNN APIs are built around it (batched sequences, independent)
3. It works — you get publishable results with clean dynamics
4. The alternative (continuous state across trials) is technically harder and there's no established framework for it

The irony is that the neuroscience community adopted this from machine learning (LSTMs, seq2seq) where hard resets make sense (sentences are independent). But neural circuits aren't language models — they run continuously.

### Why Nobody Has Complained

- It's an invisible assumption. Everyone does it, so it doesn't get flagged in review
- The field frames it as "trial-based tasks" — matching experimental design. The reset is hidden in methods as "activity was initialized to zero at trial onset"
- There's no obvious failure — hard-reset RNNs do learn plausible dynamics, produce fixed points and limit cycles, and give interesting analyses. The results aren't wrong per se — they're just about a different dynamical system than the one in the animal's head
- Nobody has shown it matters — there's no paper directly comparing conclusions from hard-reset vs. continuous-state RNNs on the same task and showing they diverge
- The people who would notice (dynamical systems theorists) mostly aren't reviewing these papers

### References That Touch on Continuous Dynamics (Without Explicitly Critiquing Resets)

- Barak (2017) — trained RNN review, describes trial-based setup as standard without questioning it
- Yang & Wang (2020) — same
- Reservoir computing community (Jaeger, Maass) — work with continuous streams, separate community
- FORCE learning (Sussillo & Abbott 2009) — trains on continuous signals without resets, but different training method (not BPTT)

## Our Approach: Continuous State with Single-Step Prediction

In the GNN framework with `simulation_initial_state=False` (default):

- Activity **carries over** between stimulus sequences — no hard reset
- The network integrates continuously across hundreds of sequences (thousands of frames)
- Dynamics are dominated by steady-state responses to stimuli, not reset transients
- Single-step prediction loss sidesteps the BPTT gradient flow problem entirely
- This is biologically realistic: the fly's visual system processes a continuous stream

This is a methodological advantage over the standard RNN framework for studying real connectomes. The comparison (hard-reset vs. continuous, same task, same architecture) could be done by toggling `simulation_initial_state` in the config.
