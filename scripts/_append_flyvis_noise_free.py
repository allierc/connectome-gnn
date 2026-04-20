#!/usr/bin/env python3
"""One-shot append script for flyvis_noise_free analysis log."""

content = """\

---

## Iter 85: stable_robust [Block 8 ROBUSTNESS Batch 1/3, Slot 0]
Node: id=85, parent=champion (g_phi_wL1=0.14, W_L1=1.5e-4)
Mode: ROBUSTNESS TEST (Block 8 \u2014 all 4 slots identical config)
Hypothesis tested: "Champion config achieves mean conn_R2 > 0.95, CV < 3% on 4 new seeds"
Config: lr_W=3e-4, lr=9e-4, lr_emb=1.55e-3, DAL=150, n_epochs=1, bs=4,
        g_diff=1500, g_norm=0, g_phi_wL1=0.14, f_L1=0, W_L1=1.5e-4, W_L2=0,
        w_init=randn_scaled, emb_dim=4, hidden=80
Slot 0: conn_R2=0.9491, raw_W_R2=0.8095, dale_law_score=0.8455, tau_R2=0.8810, V_rest_R2=0.1514,
        onestep_r=0.9987, rollout_r=0.9335 (std=0.2106 \u2014 some trajectories diverged), cluster_acc=0.8042,
        final_loss=1184.7, sim_seed=85000, train_seed=85500, training_time_min=51.8
Mutation: ROBUSTNESS TEST \u2014 same config as champion
W visual: raw_W_R2=0.8095 \u2014 good structure. dale_law_score=0.8455 (~84.6% sign-correct). Typical raw_W pattern for champion config: sign partial degeneracy corrected by effective-W ratio \u2192 conn_R2=0.9491. rollout_r std=0.2106 suggests a fraction of rollout trajectories diverge while most are clean (rollout_r itself high enough for good classification).
Training time: 51.8 min \u2014 within target \u2713. No DAL adjustment.
Verdict: STABLE-ROBUST (single seed). conn_R2=0.9491 > prior winner 0.923 \u2713.

---

## Iter 86: stable_robust [Block 8 ROBUSTNESS Batch 1/3, Slot 1]
Node: id=86, parent=champion (g_phi_wL1=0.14, W_L1=1.5e-4)
Mode: ROBUSTNESS TEST (Block 8 \u2014 all 4 slots identical config)
Hypothesis tested: "Champion config achieves mean conn_R2 > 0.95, CV < 3% on 4 new seeds"
Config: same as Iter 85
Slot 1: conn_R2=0.9553, raw_W_R2=0.8246, dale_law_score=0.8464, tau_R2=0.0425, V_rest_R2=0.1441,
        onestep_r=0.9986, rollout_r=0.9982, cluster_acc=0.7767,
        final_loss=1084.3, sim_seed=86001, train_seed=86501, training_time_min=51.8
Mutation: ROBUSTNESS TEST \u2014 same config as champion
W visual: raw_W_R2=0.8246 \u2014 highest raw_W in this batch. Clean seed with both good raw alignment and strong conn_R2. rollout_r=0.9982 (excellent). dale_law_score=0.8464 (typical).
Training time: 51.8 min \u2014 within target \u2713.
Verdict: STABLE-ROBUST (single seed). conn_R2=0.9553 > prior winner 0.923 \u2713. Best result in batch.

---

## Iter 87: stable_robust [Block 8 ROBUSTNESS Batch 1/3, Slot 2]
Node: id=87, parent=champion (g_phi_wL1=0.14, W_L1=1.5e-4)
Mode: ROBUSTNESS TEST (Block 8 \u2014 all 4 slots identical config)
Hypothesis tested: "Champion config achieves mean conn_R2 > 0.95, CV < 3% on 4 new seeds"
Config: same as Iter 85
Slot 2: conn_R2=0.9484, raw_W_R2=0.7341, dale_law_score=0.8518, tau_R2=0.8576, V_rest_R2=0.0877,
        onestep_r=0.9989, rollout_r=0.9989, cluster_acc=0.7403,
        final_loss=1157.1, sim_seed=87002, train_seed=87502, training_time_min=51.8
Mutation: ROBUSTNESS TEST \u2014 same config as champion
W visual: raw_W_R2=0.7341 \u2014 lowest raw_W in batch. Stronger sign degeneracy than typical, but effective-W correction recovers conn_R2=0.9484 (above 0.923 floor). dale_law_score=0.8518 (highest in batch \u2014 more sign-correct signs despite lower raw_W_R2 paradox: consistent with stronger sign-flip degeneracy where block correction is cleaner). rollout_r=0.9989 (excellent dynamics fit). cluster_acc=0.7403 (lowest in batch \u2014 mildly softer embedding for this seed).
Training time: 51.8 min \u2014 within target \u2713.
Verdict: STABLE-ROBUST (single seed). conn_R2=0.9484 > prior winner 0.923 \u2713.

---

## Iter 88: stable_robust [Block 8 ROBUSTNESS Batch 1/3, Slot 3]
Node: id=88, parent=champion (g_phi_wL1=0.14, W_L1=1.5e-4)
Mode: ROBUSTNESS TEST (Block 8 \u2014 all 4 slots identical config)
Hypothesis tested: "Champion config achieves mean conn_R2 > 0.95, CV < 3% on 4 new seeds"
Config: same as Iter 85
Slot 3: conn_R2=0.9495, raw_W_R2=0.7733, dale_law_score=0.8484, tau_R2=0.6461, V_rest_R2=0.2596,
        onestep_r=0.9966, rollout_r=0.9969, cluster_acc=0.8034,
        final_loss=1121.5, sim_seed=88003, train_seed=88503, training_time_min=52.0
Mutation: ROBUSTNESS TEST \u2014 same config as champion
W visual: raw_W_R2=0.7733 \u2014 moderate; typical partial sign degeneracy. dale_law_score=0.8484 (typical). Effective-W correction recovers full conn_R2=0.9495. cluster_acc=0.8034 (2nd highest in batch alongside slot 0 \u2014 embedding quality strong on this seed).
Training time: 52.0 min \u2014 within target \u2713.
Verdict: STABLE-ROBUST (single seed). conn_R2=0.9495 > prior winner 0.923 \u2713.

---

## Batch 85\u201388 Summary: Block 8 ROBUSTNESS Batch 1/3 \u2014 STABLE-ROBUST CONFIRMED

Mode: ROBUSTNESS TEST \u2014 4 new seeds of champion config (g_phi_wL1=0.14, W_L1=1.5e-4)
Training times: 51.8\u201352.0 min \u2014 all clean \u2713. No DAL adjustment.

Results:
- Iter 85 (seed 85000): conn_R2=0.9491, raw_W_R2=0.8095, rollout_r=0.9335 (partial divergence)
- Iter 86 (seed 86001): conn_R2=0.9553, raw_W_R2=0.8246, rollout_r=0.9982
- Iter 87 (seed 87002): conn_R2=0.9484, raw_W_R2=0.7341, rollout_r=0.9989
- Iter 88 (seed 88003): conn_R2=0.9495, raw_W_R2=0.7733, rollout_r=0.9969

Block 8 Batch 1 stats:
- 4-seed mean conn_R2 = 0.9506, std = 0.0028, CV = 0.29%
- Min = 0.9484, Max = 0.9553
- All \u2265 0.923 (prior winner): YES (all \u2265 0.948)
- Catastrophic (< 0.50): 0/4
- Classification: STABLE-ROBUST (all \u2265 0.90, CV = 0.29% << 3%)

Combined 8-seed (Block 7 champion {71,73,77,81} + Block 8 batch 1 {85,86,87,88}):
- Values: 0.9579, 0.9630, 0.9665, 0.9502, 0.9491, 0.9553, 0.9484, 0.9495
- 8-seed mean = 0.9550, std = 0.0065, CV = 0.68%

Key findings:
1. **Block 8 Batch 1 STABLE-ROBUST**: CV=0.29% far below 3% threshold. All 4 seeds above 0.948.
2. **NO catastrophic seeds**: Hard-seed risk (~25% historical) did not materialize \u2014 seeds 85000\u201388003 are all "soft" seeds.
3. **Confirmed > prior winner 0.923**: Minimum 0.9484 comfortably exceeds prior winner.
4. **8-seed combined mean = 0.9550, CV=0.68%**: Very consistent across 8 total seeds.
5. **Hypothesis SUPPORTED**: Champion config confirms robust convergence on fresh seeds.
6. **Block 8 continues**: 2 more batches (Iters 89-92, 93-96) remain in Block 8 for complete 12-seed robustness.

---

## Block 8 \u2014 Final Robustness Batch 2/3 (Iters 89\u201392)

Mode: ROBUSTNESS TEST \u2014 all 4 slots = champion config (pipeline forces different seeds)
Goal: Continue confirming robustness; complete Block 8 12-seed formal test
Champion: g_phi_wL1=0.14, W_L1=1.5e-4, lr_W=3e-4, lr=9e-4, DAL=150, n_epochs=1, bs=4,
          g_diff=1500, g_norm=0, randn_scaled, emb_dim=4, hidden=80

Slot 0 (Iter 89, seeds 89000/89500): ROBUSTNESS \u2014 champion config
Slot 1 (Iter 90, seeds 90001/90501): ROBUSTNESS \u2014 champion config
Slot 2 (Iter 91, seeds 91002/91502): ROBUSTNESS \u2014 champion config
Slot 3 (Iter 92, seeds 92003/92503): ROBUSTNESS \u2014 champion config

Hypothesis: "Champion config continues to achieve conn_R2 > 0.93 (prior winner) on all seeds; running mean stays near 0.955, CV < 3%."
Expected: No catastrophic seeds; continued convergence in the 0.93\u20130.97 range.

---
"""

filepath = "/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_free/flyvis_noise_free_Claude_analysis.md"
with open(filepath, 'a', encoding='utf-8') as f:
    f.write(content)
print(f"Successfully appended {len(content)} characters to {filepath}")
