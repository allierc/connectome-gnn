#!/usr/bin/env python3
"""Append Block 3 Batch 3 entries (Iters 25-28) and Block 3 summary to analysis log."""

content = """
<!-- ============================================================ -->
<!-- BLOCK 3: g_phi Parameters BATCH 3 (Iters 25-28)             -->
<!-- ============================================================ -->

## Iter 25: FRAGILE
Node: id=25, parent=24
Mode: Exploration -- CONTROL (parent config, new seed 25000)
Hypothesis tested: "Parent config (g_phi_diff=300, g_phi_norm=0.45) replicates conn_R2~0.85 at new seeds"
Config: n_epochs=1, DAL=165, lr_W=6e-4, lr=1.2e-3, lr_emb=1.55e-3,
        rate=0.0, W_L1=1.5e-4, W_L2=1.5e-6,
        g_diff=300, g_norm=0.45, g_L1=0.28, f_L1=0.05, f_L2=0.001,
        w_init=randn_scaled(scale=1.0), bs=4
Slot 0: conn_R2=0.8115, tau_R2=0.8893, V_rest_R2=0.1086, cluster_acc=0.5184, rollout_r=0.9221, sim_seed=25000, train_seed=25500
Seed stats: mean_conn_R2=0.8115, std=N/A, CV=N/A, min=0.8115, max=0.8115, catastrophic=0/1
Mutation: CONTROL (no change from parent)
W matrix: raw_W_R2=0.6562 (lower than Iter 24 raw_W=0.7809). Seed 25000 harder than 24003 -- conn_R2 substantially below ATB despite identical config.
Verdict: fragile -- seed variability confirmed. ATB unchanged at Iter 24 (0.8524).
Next: parent=24

---

## Iter 26: DISQUALIFIED (tau_R2 collapse)
Node: id=26, parent=24
Mode: Exploration -- g_phi_norm=0.2 (reduction from 0.45)
Hypothesis tested: "g_phi_norm=0.2 further improves conn_R2 beyond 0.45 by allowing more flexible normalization"
Config: n_epochs=1, DAL=165, lr_W=6e-4, lr=1.2e-3, lr_emb=1.55e-3,
        rate=0.0, W_L1=1.5e-4, W_L2=1.5e-6,
        g_diff=300, g_norm=0.2, g_L1=0.28, f_L1=0.05, f_L2=0.001,
        w_init=randn_scaled(scale=1.0), bs=4
Slot 0: conn_R2=0.8253, tau_R2=0.2124, V_rest_R2=0.0001, cluster_acc=0.4579, rollout_r=0.9231, sim_seed=26001, train_seed=26501
Seed stats: mean_conn_R2=0.8253, tau_R2=0.2124 (CATASTROPHIC), catastrophic=1/1 (tau collapse)
Mutation: g_phi_norm: 0.45 -> 0.2
W matrix: raw_W_R2=0.7885 (highest in batch), but tau_R2=0.2124 means tau recovery is unreliable. V_rest_R2~0.0001 also co-collapsed with tau.
Verdict: DISQUALIFIED -- tau_R2=0.2124 catastrophic. g_phi_norm=0.2 triggers tau+V_rest co-collapse (same signature as lr_emb=2.325e-3 failure in Block 2). Hypothesis falsified.
Next: parent=24. g_phi_norm=0.2 eliminated.

---

## Iter 27: DISQUALIFIED
Node: id=27, parent=24
Mode: Exploration -- g_phi_diff=150 + g_phi_norm=0.45 (re-test lower diff with updated parent norm)
Hypothesis tested: "g_phi_diff=150 + g_phi_norm=0.45 combine positively -- Iter 22 result with diff=150 extends to norm=0.45 parent"
Config: n_epochs=1, DAL=165, lr_W=6e-4, lr=1.2e-3, lr_emb=1.55e-3,
        rate=0.0, W_L1=1.5e-4, W_L2=1.5e-6,
        g_diff=150, g_norm=0.45, g_L1=0.28, f_L1=0.05, f_L2=0.001,
        w_init=randn_scaled(scale=1.0), bs=4
Slot 0: conn_R2=0.7778, tau_R2=0.8710, V_rest_R2=0.1668, cluster_acc=0.5077, rollout_r=0.9382, sim_seed=27002, train_seed=27502
Seed stats: mean_conn_R2=0.7778, catastrophic=0/1 (conn DQ < 0.80)
Mutation: g_phi_diff: 300 -> 150
W matrix: raw_W_R2=0.7087. Despite excellent rollout (0.9382) and tau (0.8710), W recovery failed. Interaction: diff=150 + norm=0.45 is destructive.
Verdict: DISQUALIFIED -- conn_R2=0.7778 < 0.80. Critical interaction: diff=150 + norm=0.45 is incompatible. Iter 22 had diff=150 + norm=0.9 (gave 0.850) -- lower norm requires higher diff for stability.
Next: parent=24. g_phi_diff=150 with norm=0.45 eliminated.

---

## Iter 28: FRAGILE
Node: id=28, parent=24
Mode: Exploration -- g_phi_norm=0.0 (disable norm entirely)
Hypothesis tested: "g_phi_norm=0.0 (disabled saturation constraint) further improves conn_R2 beyond 0.45"
Config: n_epochs=1, DAL=165, lr_W=6e-4, lr=1.2e-3, lr_emb=1.55e-3,
        rate=0.0, W_L1=1.5e-4, W_L2=1.5e-6,
        g_diff=300, g_norm=0.0, g_L1=0.28, f_L1=0.05, f_L2=0.001,
        w_init=randn_scaled(scale=1.0), bs=4
Slot 0: conn_R2=0.8152, tau_R2=0.9517, V_rest_R2=0.2977, cluster_acc=0.4620, rollout_r=0.9317, sim_seed=28003, train_seed=28503
Seed stats: mean_conn_R2=0.8152, catastrophic=0/1
Mutation: g_phi_norm: 0.45 -> 0.0
W matrix: raw_W_R2=0.7628 (second highest in batch). Excellent tau (0.9517) and strong V_rest (0.2977). Disabling norm preserves tau but does not improve conn_R2.
Verdict: falsified -- conn_R2=0.8152 ~ control (0.8115), not better than 0.45 (ATB=0.8524). g_phi_norm=0.45 remains optimal.
Next: parent=24. g_phi_norm CLOSED at 0.45.

---

<!-- ============================================================ -->
<!-- BLOCK 3 SUMMARY: g_phi Parameters COMPLETE (Iters 17-28)     -->
<!-- ============================================================ -->

## Block 3 Summary: g_phi Parameters (COMPLETE)

Status: COMPLETE -- 12/12 iterations done (Iters 17-28)
ATB: Iter 24, conn_R2=0.8524 (g_phi_diff=300, g_phi_norm=0.45, seed 24003)

g_phi_diff CLOSED (Iters 17-23, 27):
- 300 is optimal. 150 ties 300 with norm=0.9 but DQs with norm=0.45 (interaction).
- 500 DISQUALIFIED (0.800). 750+ degrade tau progressively.
- 10% winner diff=1500 does NOT transfer to 20% removal.

g_phi_norm CLOSED (Iters 21, 24-28):
- 0.45 is optimal (ATB=0.8524). Non-monotonic dose-response:
  0.2: catastrophic tau_R2 collapse (0.2124) -- DANGEROUS
  0.0: conn_R2 ~ 0.45 (no improvement), tau preserved (0.9517)
  0.9: marginally lower (~0.849 in control)
- Critical interaction: diff=150 + norm=0.45 -> DISQUALIFIED (0.7778). Lower diff needs higher norm.

g_phi_weight_L1 UNTESTED: Budget exhausted. Remains at 0.28. Revisit in Block 8 (free exploration).

Key new failure mode discovered: g_phi_norm=0.2 causes tau_R2+V_rest_R2 co-collapse.
Identical signature to lr_emb=2.325e-3 failure. General "tau catastrophe" pattern emerges.

Winner config saved: iter_024_slot_03 (g_phi_diff=300, g_phi_norm=0.45, conn_R2=0.8524).
Moving to Block 4 (f_theta regularization, Iters 29-40).
"""

log_path = '/groups/saalfeld/home/allierc/GraphData/log/Claude_exploration/LLM_flyvis_noise_005_removed_pc_20/flyvis_noise_005_removed_pc_20_Claude_analysis.md'
with open(log_path, 'a') as f:
    f.write(content)
print('Analysis log appended successfully.')
