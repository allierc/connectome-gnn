# Two solutions to the same task: line attractor, with or without a limit cycle

Five folds of `zebrafish_om_intg_285_both_XL` (seeds 0-4, identical spec,
identical corpus). Measured on the held-out `dot_corpus_1d_xl` test split.
Everything below is at **dt/10** unless it says otherwise -- see "the solver
was lying" for why that matters.

## The table

| seed | err (deg) | max Re λ | n unst | λ freq | rollout peak | pow>5Hz | hold (deg) | AC | τ max (s) | ρ(Ŵ) | regime |
|------|-----------|----------|--------|--------|--------------|---------|------------|-----|-----------|------|--------|
| 0 | **0.0372** | +10.30 | 2 | 17.1 | 11.50 | 0.992 | 42.2 | 0.063 | 5.16 | 3.09 | limit cycle |
| 1 | 0.0409 | +6.65 | 2 | 18.3 | 13.00 | 0.981 | 37.0 | 0.034 | 9.34 | 3.72 | limit cycle |
| 2 | 0.0514 | +7.24 | 3 | 11.0 | 8.25 | 0.969 | 36.3 | 0.046 | 4.02 | 2.23 | limit cycle |
| 3 | 0.0434 | **-0.05** | **0** | -- | **0.13** | **0.006** | 35.3 | **0.001** | 5.87 | 2.79 | clean line attractor |
| 4 | 0.0486 | +9.21 | 2 | 22.6 | 16.50 | 0.984 | 36.7 | 0.023 | 7.64 | 2.63 | limit cycle |

`max Re λ` is of the continuous-time Jacobian `A = diag(1/τ)(-I + Ŵ)`, which
has no dt in it. `hold` and `AC` are the gaze and the mean per-neuron rate
s.d. over the last second of the integrator probe (2 s velocity step, then 8 s
of no input).

## Every fold is a line attractor

After the input is cut, all five HOLD a graded position for 8 s, monotone in
how hard they were driven (-46 to +42 deg across drive amplitudes -1 to +1).
The instability does not destroy the integration -- it rides on top of it.
Driving from rest with zero input gives AC exactly 0.000, so the limit cycle
has to be excited; it is not spontaneous.

## Four of five also carry a limit cycle, and the task cannot tell

Continuous-time unstable complex pairs at 11-23 Hz, amplitude-limited by tanh.
Accuracy is **identical across the two regimes** -- 0.0445 (oscillating, n=4)
vs 0.0434 (clean, n=1), against a fold-to-fold s.d. of 0.005. The best fold
(cv0, 0.0372) oscillates; the clean fold ranks 3rd of 5.

The reason nothing selects between them: the rates reach the loss only through
`Wout` and then through the eye, a second-order plant with a ~1 Hz corner. A
12 Hz limit cycle is low-passed away before it can reach the gaze error, so
the optimiser has no gradient against it. Training also inflates ρ(Ŵ) from the
`spectral_target: 0.9` init to 2.89 ± 0.50 for the same reason.

## The solver was lying, and only about one fold

At the training step dt = 1/60, ~55 of the 285 fitted τ fall below dt, where
`alpha = clamp(dt/τ, 1e-4, 1)` saturates at 1 and forward Euler sits at its
stability limit. Driven by the corpus:

| | dt = 1/60 | dt = 1/600 |
|---|---|---|
| cv0 | 7.50 Hz, 0.995 | 11.50 Hz, 0.992 |
| cv1 | 8.25 Hz, 0.988 | 13.00 Hz, 0.981 |
| cv2 | 6.38 Hz, 0.990 | 8.25 Hz, 0.969 |
| **cv3** | **15.00 Hz, 0.995** | **0.13 Hz, 0.006** |
| cv4 | 10.00 Hz, 0.995 | 16.50 Hz, 0.984 |

cv3's oscillation is entirely an integration artefact. The other four survive
20x refinement, and the coarse step was *under*-reporting their frequency.
Hence `rollout_substep` in `test_zebra_eyeG.py`.

## τ does not explain it. Neither does Ŵ. Their alignment does.

The τ distributions are nearly identical and point the wrong way if anything:
cv3, the non-oscillating fold, has MORE neurons below dt (70 vs 53) and a
slightly slower median (0.154 vs 0.132 s). ρ(Ŵ) is 3.09 vs 2.79.

Swapping the two factors between folds:

| combination | max Re λ | n unstable | freq |
|-------------|----------|------------|------|
| Ŵ cv0 + τ cv0 (trained) | **+10.302** | 2 | 17.1 Hz |
| Ŵ cv0 + τ cv3 | -0.006 | 0 | -- |
| Ŵ cv3 + τ cv0 | -0.074 | 0 | -- |
| Ŵ cv3 + τ cv3 (trained) | -0.046 | 0 | -- |

Neither factor carries the instability alone. cv0's weights are stable with
cv3's time constants and vice versa; only the trained pairing is unstable. The
recurrent loops run through exactly the neurons that fold made fast, because
training co-adapts the two. So the instability cannot be predicted from τ, nor
prevented by capping ρ(Ŵ) -- the quantity to penalise, if one wanted to, is
`max Re λ` of `diag(1/τ)(-I + Ŵ)` itself.

Note the three swapped combinations land at -0.006, -0.074, -0.046: all
essentially AT zero. Breaking the co-adaptation does not give a comfortably
stable circuit, it drops the system back to the marginal line-attractor
boundary, which is where the task alone puts it.

## What this does not establish

n = 5. The 4:1 split is not distinguishable from 3:2 or 9:1. cv3's
`max Re λ = -0.05` is marginal, not robustly stable -- the right label is "no
unstable mode at this fit". And all five were TRAINED at dt = 1/60, so their
weights partly compensate for an Euler error the substepped rollout no longer
has; a run trained with substeps might land elsewhere. That is an experiment,
not a correction to these.

## Reproducing

    export GNN_OUTPUT_ROOT=/groups/saalfeld/home/allierc/GraphData
    cd prototype/dot_tracking
    for s in 0 1 2 3 4; do
      python train_zebra_eyeG.py --spec ../../config/zebrafish/zebrafish_om_intg_285_both_XL_cv$s.yaml
      python test_zebra_eyeG.py  --spec ../../config/zebrafish/zebrafish_om_intg_285_both_XL_cv$s.yaml \
             --phi-zero --fps 30 --head-az 60
    done

The movies show the probe as their seventh phase. cv3's mp4 is 18.5 MB against
48-69 MB for the others at the same resolution and frame count: h264
compresses a flat held state far better than a 12 Hz limit cycle, so the one
non-oscillating fold is visible before you open it.
