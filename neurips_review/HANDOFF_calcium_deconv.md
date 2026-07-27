# Handoff — Q2 calcium / deconvolution. Paused 2026-07-27.

Local GPUs are free. 8 cluster jobs are RUNNING on gpu_a100 and need no babysitting.

## Resume in one command

    python neurips_review/launch_snr_grid.py --table     # jobs + deconvolution quality
    ssh allierc@login1 "source /etc/profile.d/profile.lsf.sh && bjobs"

Collect R2_W per run from
`/groups/saalfeld/home/allierc/GraphData/log/fly/nr2_ca_snr_*/results/metrics.txt`,
then fill the Q2 table in `reply_all.tex` (the section already says the fits are
running).

## Jobs in flight — the SNR grid (Vzfg Q2)

8 jobs, ids 153175950-57, queue gpu_a100. Process noise fixed at sigma = 0.05
(it is baked into the trajectory by the generator, so it is not a post-hoc axis;
a sigma sweep needs a new generation pass, which we dropped).

| config                    | arm       | sigma | gamma/SD | SNR dB | lambda | r(v)   | r(vdot) |
|---------------------------|-----------|-------|----------|--------|--------|--------|---------|
| nr2_ca_snr_g000_unified   | GNN       | 0.05  | 0        | inf    | 1e-6   | 0.9998 | 0.9933  |
| nr2_ca_snr_g000_known_ode | Known-ODE | 0.05  | 0        | inf    | 1e-6   | 0.9998 | 0.9933  |
| nr2_ca_snr_g010_unified   | GNN       | 0.05  | 0.01     | 40.0   | 1e-3   | 0.9797 | 0.4353  |
| nr2_ca_snr_g010_known_ode | Known-ODE | 0.05  | 0.01     | 40.0   | 1e-3   | 0.9797 | 0.4353  |
| nr2_ca_snr_g030_unified   | GNN       | 0.05  | 0.03     | 30.5   | 1e-2   | 0.9695 | 0.2803  |
| nr2_ca_snr_g030_known_ode | Known-ODE | 0.05  | 0.03     | 30.5   | 1e-2   | 0.9695 | 0.2803  |
| nr2_ca_snr_g100_unified   | GNN       | 0.05  | 0.10     | 20.0   | 1e-1   | 0.9562 | 0.1719  |
| nr2_ca_snr_g100_known_ode | Known-ODE | 0.05  | 0.10     | 20.0   | 1e-1   | 0.9562 | 0.1719  |

lambda is retuned per row by maximising r(vdot); each row is the best inversion
available at that noise level, not a fixed filter. r values are the train split;
test agrees within ~0.07. Built by `neurips_review/build_snr_grid.py`, submitted
by `neurips_review/launch_snr_grid.py`.

**g000 is a control, not a result.** At zero measurement noise C = K*v is exactly
invertible and recovering v is division in Fourier space. A high score there
confirms the pipeline; it says nothing about calcium imaging. The rows that carry
the argument are 40 / 30 / 20 dB. Real two-photon GCaMP sits at 20-40 dB.

**Preliminary read:** a local run on the g000 dataset reached conn (R2_W) = 0.968
at 5% of epoch 1, against 0.07 from the buggy deconvolution. Early and not final,
but it confirms the fix. That local run was killed as redundant with 153175950.

## Why the original deconvolution was ill

`neurips_review_build_deconv_dataset.py` hard-coded `LAM = 3e-3` with
`regularizer='derivative'`. That prior penalises ||DV||^2 -- exactly the
derivative W is identified from -- and it was applied to a NOISE-FREE trace
(`measurement_noise_level: 0.0`, `calcium_noise_level: 0.0`), where the correct
lambda is ~0. Per frequency the filter is

    V_hat = K*(f) F(f) / ( |K(f)|^2 + lambda max|K|^2 |R(f)|^2 ),  R = |1 - e^-iw|^2

R is zero at DC and maximal at Nyquist, and the GCaMP6f kernel is already a
low-pass, so the result is a double low-pass: the indicator attenuated the high
band and the regulariser attenuated it again instead of restoring it. Measured
cost: derivative amplitude 0.28x truth, r(vdot) 0.43, while r(v) stayed 0.98.
Hence R2_W = 0.07. That number characterised our preprocessing, not calcium.

lambda = 3e-3 came from the figure path (`-o deconvolve`), which INJECTS
synthetic noise when none exists (`FALLBACK_NOISE_FRACTION = 0.20`, commented "so
the deconv stress test still has something to chew on") and then sets
lambda = 100 sigma^2. Heavy smoothing is right for a noisy trace. The dataset
builder copied the constant while feeding it a clean one.

`LAM` is now overridable via the `DECONV_LAM` env var.

## Second bug, mine, already fixed — do not reintroduce

The first version of `build_snr_grid.py` repaired the head warm-up but omitted
reflect padding. Both are needed and they fix different ends:

- **padding** fixes the TAIL: the FFT is circular, so the end of the record wraps
  onto its start and puts a ~100x spike on the final frame at small lambda.
- **warm-up repair** (`est[:4] = est[4]`) fixes the HEAD: frames 0-3 of the
  calcium trace are an incomplete convolution, so v there is unrecoverable.

Omitting padding dropped g000 test r(vdot) from 0.97 to 0.24. Caught only because
g000 must reproduce the noiseless result and did not.

Both bugs were invisible in r(v) and showed up only in r(vdot). **Score any future
deconvolution change on r(vdot), never on r(v).**

Padding length does not matter (invariant 1L to 16L) -- it is not a decay
transient, it is the wrap.

## Measured facts worth not re-deriving

- Deconvolution collapses with tiny measurement noise: 0.2% noise (54 dB) already
  halves r(vdot) to 0.71, while r(v) only moves 0.9998 -> 0.993.
- Stimulus/sequence boundaries do NOT need dropping: 800 abrupt jumps cost only
  1.7x median error over the following 5 frames, decaying to 1.1x by 20. No need
  for longer data.
- CX effective ranks (W = 12 directions; activity 6 / 7 / 95) verified by exact
  full SVD. The "7 -> 37" from `rank_info.txt` is WRONG -- it uses
  `torch.svd_lowrank` capped at 50 components and normalises energy over those 50.

## reply_all.tex state

Final and in the document: CX row (GNN 0.54 / 0.84 / 1.00; Known-ODE -2.07 / 1.00
/ 1.00), the six raw-calcium rows, AC concern 3, the gamma=0.1 clarification, and
a rewritten Q2 deconvolution section carrying the SNR table, the 20-40 dB framing,
and an explicit withdrawal of the old R2_W = 0.07.

Still to do when the jobs land: add the fitted R2_W per grid cell.

## Open, user's call

- **Character limits.** `md/*.md` is what gets pasted into OpenReview, 10,000 each.
  Regenerate with `python neurips_review/to_markdown.py`. At the pause AC and Vzfg
  are both over. User said DO NOT COMPRESS, so this was left alone.
- Phase 2 discussion runs to 2026-08-03, so corrected numbers can still land.

## Prompt recovery

reply_all.tex originates in session `7bb5a6c9-af79-4b86-9335-a6b097a23901`
(2026-07-26 11:27, "create a full answer tex document now with R1 and R2 since
they may evolve together"). Also `8c9ecf19-...` (17:14-19:50 polish) and
`1eb40363-...` (overnight launch). NOT yet appended to PROMPTS.md.
