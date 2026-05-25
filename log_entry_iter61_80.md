
## Iter 61-80 (Block 2, Batch 2.3): exploration - decoupled_low ED at canonical peak 2e-3
Mutation (uniform 20 slots, n_units fixed per-slot axis):
  - lr_W_rec_schedule = canonical slower-decay x 2.0 (peak 2e-3) - REVERTED from 2.2 steeper tail
  - lr_W_ED_schedule = decoupled_low [5e-4, 2e-4, 1e-4 x 8]   <- NEW
Hypothesis (H5+H6): Block-1 collapse driver = ED tied to W_rec at long plateau, not peak height. Decoupling ED to 10x lower rates should rescue the canonical schedule and stabilise 1024.

Per-size summary:
  n_units=256 (slots 0-6, n=7, 10/10 ep):
    final r2: mean=0.805 std=0.027 ceiling-of-finals=0.834 (slot 2)
    best-in-trajectory r2: mean=0.884 ceiling=0.945 (slot 6 ep 9)
  n_units=512 (slots 7-13, n=7, ~7/10 ep, TRUNCATED):
    late r2 (iter ~80k): 0.80-0.87 across slots; ceiling 0.897 (slot 12 @ iter 72385)
  n_units=1024 (slots 14-19, n=6, 3/10 ep, TRUNCATED, NO COLLAPSE):
    late r2 (iter ~35k): 0.71-0.87; ceiling 0.872 (slot 14 @ iter 32449)
    -> NEW record for epoch-3 1024 r2 (prior 2.2 record was 0.857)

Per-slot ceilings (max r2 in trajectory):
  256: s0=0.865 s1=0.875 s2=0.883 s3=0.899 s4=0.867 s5=0.857 s6=0.945  mean=0.884
  512: s7=0.866 s8=0.804 s9=0.857 s10=0.870 s11=0.859 s12=0.897 s13=0.892  mean=0.864
  1024: s14=0.872 s15=0.812 s16=0.858 s17=0.843 s18=0.840 s19=0.838  mean=0.844

Saturation diagnosis: at equal compute (ep 3), 256 (~0.88 ceiling) ~ 512 (~0.90 ceiling) ~ 1024 (~0.87 ceiling). All sizes overlap; 1024 needs more epochs to potentially separate. CURVE IS FLAT-OR-DESCENDING with current truncation -> matrix decoder is likely the bottleneck (H3 SUPPORTED with current data; need converged 1024 to be sure).

Verdict (H5/H6 jointly): SUPPORTED. Canonical-peak schedule no longer collapses 1024 when ED is decoupled to <=10x lower rates. Block-1 collapse was an ED+W_rec coupling artifact, not a peak-height issue.

Slot 6 outlier note: r2 peak 0.945 at iter ~95k (epoch 9) - highest single-checkpoint r2 in this loop. Single seed, hard to credit to mutation alone.

Next mutation: Block 3.1 (regularisation no-reg baseline; carry forward 2.3 decoupled ED + canonical W_rec)
  - coeff_rate_L2: 0.01 -> 0
  - grad_clip_W: 2.0 -> 5.0
  - coeff_W_L2: 0.0 (parent)
  - noise_recurrent_level: 0.0 (parent)
Hypothesis: With sigma=true already keeping activity bounded, rate_L2 is overkill. Removing it + loosening grad clip should let the model use more capacity, especially at 1024.
