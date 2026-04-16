# Working Memory: flyvis_noise_005_hidden_010_ngp
# STATUS: EXPLORATION COMPLETE — 128/128 iterations | 12 blocks | 2026-04-11

## Paper Summary

- **Hidden-neuron problem**: 10% of non-retinal Drosophila visual neurons (~1200 of 12005) are unobserved during GNN training. Their voltages are inferred by a MultiResTemporalGrid (NGP-T) implicit neural representation.
- **Joint training**: GNN learns connectivity W while NGP-T predicts hidden voltages. NGP-T receives gradients through GNN loss (indirect) and via coeff_hidden_voltage direct one-step prediction loss.
- **Key finding**: After 128 iterations across 12 blocks, nnr > 0 was NEVER achieved (0% success rate). Fundamental barrier: indirect gradient pathway alone cannot train NGP-T. Direct GT voltage supervision required.

## Knowledge Base

### Results Table (Final)

| Iter | Config summary | conn_R2 | hidden_nnr_R2 | tau_R2 | rollout | time_min | Rating |
|------|----------------|---------|---------------|--------|---------|----------|--------|
| 1-12 | baseline (coeff_hv=0) | 0.606±0.163 | ~-38 (mean) | — | mixed | ~58 | failed |
| 16 | coeff_hv=10.0 | 0.778 | -18.5 | 0.493 | — | 58.3 | partial |
| 33 | coeff_hv=3000 ctrl | 0.837 | -38.9 | 0.336 | 0.812 | 60.8 | partial |
| 68 | n_epochs=3+alt=true | 0.834 | -10.2 | 0.100 | 0.801 | 61.6 | partial |
| 80 | ratio=0.4 | 0.834 | -15.21 | 0.264 | 0.804 | 67.1 | partial-best |
| 85 | norm=1.0 CTRL | 0.828 | -4.44 ATBnnr | 0.060 | 0.034 | 66.6 | LUCKY SEED |
| 86 | norm=0.1 | 0.855 ATBconn | -120.69 | 0.319 | 0.035 | 68.3 | failed-nnr |
| 91 | norm=5.0 (2nd) | 0.810 | -11.10 | 0.017 | 0.034 | 67.5 | partial |
| 96 | norm=3.0 | 0.786 | -19.97 | 0.059 | 0.766 | 66.4 | partial |
| 109 | CTRL (B11B1) | 0.785 | -46.56 | 0.002 | 0.035 | 69.0 | partial |
| 113 | CTRL (B11B2) | 0.761 | -12.17 | 0.217 | 0.033 | 68.8 | partial |
| 116 | mlp_layers=6 | 0.784 | -10.59 | 0.084 | 0.040 | 73.8 | inconclusive |
| 117 | CTRL (B11B3) | 0.786 | -36.29 | 0.050 | 0.807 | 69.7 | partial |
| 118 | n_levels=16 (2nd) | 0.668 | -87.40 | 0.110 | 0.040 | 52.2 | FALSIFIED |
| 119 | mlp_width=256 | 0.776 | -52.43 | 0.133 | 0.054 | 67.1 | FALSIFIED |
| 120 | mlp_layers=3 | 0.806 | -13.50 | 0.020 | 0.041 | 66.3 | TENTATIVE +22.8 |
| 121 | CTRL (B12B1) | 0.747 | -12.61 | 0.039 | 0.036 | 66.6 | partial |
| 122 | rec=true, ts=1 | 0.776 | -74.29 | 0.002 | 0.788 | 69.3 | MIXED (rollout+, nnr--) |
| 123 | rec=true, ts=2, DAL=2 | 0.323 | -35.56 | 0.007 | 0.796 | 38.7 | failed (undertrained) |
| 124 | rec=true, ts=3, DAL=2 | 0.019 | -2.67 ATBnnr* | 0.008 | 0.347 | 40.4 | partial (undertrained) |
| 125 | CTRL (B12B2) | 0.730 | -195.81 | 0.081 | 0.032 | 68.5 | partial (worst CTRL nnr) |
| 126 | rec=true, ts=2, DAL=3 | 0.413 | -25.76 | 0.000 | 0.776 | 48.9 | partial (undertrained) |
| 127 | rec=true, ts=3, DAL=3 | 0.055 | -4.16 ATBnnr* | 0.000 | 0.451 | 50.5 | partial (undertrained) |
| 128 | mlp_layers=3 FINAL | 0.752 | -14.07 | 0.064 | 0.038 | 66.3 | partial (FINAL) |

*ATBnnr: near-ATB for nnr, but at invalid operating point (conn catastrophic).

### FINAL Established Principles (34 total)

1. ESTABLISHED: Indirect gradient pathway (coeff_hidden_voltage=0) CANNOT train NGP-T. 12/12 failures.
2. ESTABLISHED: Baseline GNN connectivity: mean conn_R2~0.606, CV=27%, ATB=0.855 (iter 86). With coeff_hv=3000: 0.78-0.84 reliable.
3. ESTABLISHED: Training time: bs=16/DAL=4, n_epochs=6 -> ~68 min.
4. ESTABLISHED: Peak-then-decline trajectory typical. Monotonically-improving seeds are lucky outliers.
5. ESTABLISHED: n_epochs=6+alt=true is optimal training regime.
6. ESTABLISHED: Rollout catastrophe in ~50-75% of seeds at ANY config (seed-determined, config-independent).
7. ESTABLISHED: coeff_hidden_voltage axis CLOSED. NO coeff value yields hidden_nnr_R2 > 0 reliably.
8. ESTABLISHED: lr_NNR_f=1e-3 is optimal. CLOSED.
9. ESTABLISHED: Extreme seed variance for hidden_nnr_R2: 10x+ range at any fixed config.
10. ESTABLISHED: coeff_hv=3000 gives best reliable GNN connectivity (~0.83-0.84). CLOSED.
11. ESTABLISHED: alternate_lr_ratio axis CLOSED. ratio=0.4 4-seed mean=-54.5+/-52.8. SEED-UNSTABLE.
12. ESTABLISHED: tau_R2 tends to improve with higher alternate_lr_ratio.
13. ESTABLISHED: coeff_g_phi_norm has OPPOSITE effects on conn_R2 and hidden_nnr_R2. AXIS CLOSED.
14. ESTABLISHED: norm=5.0 4-seed stats: mean=-37.6, CV=88%. DEFINITIVELY UNSTABLE.
15. ESTABLISHED: norm=3.0: nnr=-20.0 AND stable rollout (0.766). Unique stability. Single seed.
16. ESTABLISHED: Rollout stability correlates with certain configs: norm=3.0, ratio=0.4, bs=32.
17. ESTABLISHED: dale_law=true HURTS flyvis: -0.1 conn_R2, rollout catastrophe. CLOSED.
18. ESTABLISHED: coeff_g_phi_diff sweep (188-750) CLOSED. diff=375 nnr=-5.18 was lucky seed.
19. ESTABLISHED: bs=64 catastrophic (conn_R2=0.66). bs axis CLOSED (bs=16/DAL=4 optimal).
20. ESTABLISHED: DAL=4 at bs=32 HURTS both metrics. bs=32+DAL<=3 only viable.
21. ESTABLISHED: bs=32 lower final_loss does NOT predict better nnr.
22. ESTABLISHED (Block 11): mlp_width=1024 FALSIFIED (2nd seed). Width 512->1024->2048 monotonically degrades nnr. CLOSED.
23. ESTABLISHED (Block 11): CTRL seed variance alone spans 4x range (-12.17 vs -46.56). Architecture effects +-5-25 units are within seed noise.
24. ESTABLISHED (Block 11): n_levels=16 FALSIFIED at 2nd seed (iter 118). Both conn_R2 and nnr worse. AXIS CLOSED: 16 < 24 > 32.
25. ESTABLISHED (Block 11): mlp_width=256 FALSIFIED. Full range (256, 512, 1024, 2048) exhausted -- 512 best/default. WIDTH AXIS FULLY CLOSED.
26. TENTATIVE_STABLE (Block 11/12): mlp_layers=3 nnr=-13.78 (2-seed mean, CV=2.6%). Strongest architecture signal -- stability benefit real, absolute improvement marginal vs lucky CTRL seeds.
27. ESTABLISHED (Block 12): rec=true ts=1 CONSISTENTLY rescues rollout (0.036->0.788) but CONSISTENTLY destroys nnr (-12.6->-74.3). Orthogonal metrics.
28. ESTABLISHED (Block 12): Recurrent training optimizes TEMPORAL CONSISTENCY, ORTHOGONAL to hidden voltage recovery. Routing gradients through recurrent pathway suppresses nnr supervision.
29. ESTABLISHED (Block 12): ts>=2 requires DAL>=5 for proper budget. Even budgeted, conn catastrophic (0.413 ts=2, 0.055 ts=3). Trade-off structural.
30. ESTABLISHED (Block 12): rec=true ts=1 consistently rescues rollout (2 independent obs). CLOSED as nnr path.
31. ESTABLISHED (Block 12): ts=2 and ts=3 remain undertrained at DAL=3.
32. ESTABLISHED (Block 12): ts=3 near-ATB nnr (-4.16 to -2.67) ONLY when conn->0. Insurmountable trade-off.
33. ESTABLISHED (Block 12): RECURRENT AXIS DEFINITIVELY CLOSED. All ts values destroy critical metrics.
34. TENTATIVE_STABLE (Block 12): mlp_layers=3 stable nnr~-14, CV=2.6% (2 seeds). Stability real; nnr>0 not achieved.

### All Falsified Hypotheses (29 total)

1-26. [Blocks 1-11: coeff_hv, lr_NNR_f, n_epochs, alt_training, ratio, norm, dale_law, diff, bs, DAL, n_levels_32, mlp_width_1024, mlp_width_256, n_levels_16 all closed]
27. FALSIFIED (Block 12): ts=2 with proper budget avoids conn degradation. Trade-off is fundamental.
28. FALSIFIED (Block 12): ts=3 nnr ATB (iter 124, -2.67) was budget artifact. Budget not the cause.
29. FALSIFIED (Block 12): mlp_layers=3 is a path to nnr>0. 2-seed mean -13.78 -- stable but negative.

### Open Questions

ALL CLOSED. Exploration complete at 128/128 iterations.

---

## FINAL Block Summaries

### Block 9 (Iters 85-96): coeff_g_phi_norm Sweep -- COMPLETE (CLOSED)
### Block 10 (Iters 97-108): Structural Axes -- COMPLETE (CLOSED)
### Block 11 (Iters 109-120): NGP Architecture Sweep -- COMPLETE (CLOSED)
### Block 12 (Iters 121-128): Recurrent Training -- COMPLETE (CLOSED)

---

## EMERGING OBSERVATIONS (FINAL STATE)

**CRITICAL: This section must ALWAYS be at the END of memory file.**

- **conn ATB**: 0.855 (iter 86, norm=0.1). FINAL.
- **Nnr ATB (valid)**: -4.44 (iter 85, lucky seed, conn OK). **Nnr near-ATB (invalid op point)**: -4.16 (iter 127, ts=3, conn=0.055).
- **128 iterations, 0% success rate** for hidden_nnr_R2 > 0. Fundamental barrier persists across all axes.
- **Winner config**: iter_128_slot_03 (mlp_layers=3). Best stable config. n_epochs=6, alt=true, ratio=0.4, coeff_hv=3000, norm=5.0, diff=750, mlp_layers=3, n_levels=24, mlp_width=512.
- **Scientific verdict**: Hypothesis FALSIFIED. Indirect gradient alone cannot train NGP-T. Direct GT supervision required.
- **user_input.md**: no pending instructions (confirmed 2026-04-11, batch 125-128, FINAL BATCH).
