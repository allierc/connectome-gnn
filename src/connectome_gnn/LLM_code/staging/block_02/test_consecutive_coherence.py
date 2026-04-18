#!/usr/bin/env python
"""Phase-S test for consecutive_coherence_loss (block 02).

Loads full flyvis data (T=64000, N=13741, γ=0.10), uses stride-1
finite differences as a proxy for model predictions, and checks:
  (1) Noisy temporal variation ≥ 3× clean temporal variation
  (2) Absolute noisy coherence loss in [1e-3, 1e3] per neuron
  (3) Gradient norm w.r.t. predictions > 0
"""

from __future__ import annotations

import sys


def main():
    import torch
    from connectome_gnn.LLM_code.scratchpad import load_full_voltage
    from connectome_gnn.LLM_code.staging.block_02.consecutive_coherence import (
        consecutive_coherence_loss,
    )

    print("Loading flyvis voltage data (T=64000, N=13741, γ=0.10)...")
    v_clean, v_noisy = load_full_voltage("fly/flyvis_noise_free", 0.10)
    T, N = v_clean.shape
    print(f"  v_clean shape: {v_clean.shape}, v_noisy shape: {v_noisy.shape}")

    dt = 0.02  # from config: simulation.delta_t

    # --- Compute stride-1 derivatives as proxy for model predictions ---
    # d[k] ≈ (v[k+1] - v[k]) / dt  → shape (T-1, N)
    d_clean = (v_clean[1:] - v_clean[:-1]) / dt
    d_noisy = (v_noisy[1:] - v_noisy[:-1]) / dt
    print(f"  Derivative shapes: clean={d_clean.shape}, noisy={d_noisy.shape}")

    # --- Sample consecutive windows and compute coherence loss ---
    # Use batch_size=6 (matches training config) with many random windows
    B = 6  # batch_size
    n_windows = 500
    torch.manual_seed(42)
    max_start = d_clean.shape[0] - B
    starts = torch.randint(0, max_start, (n_windows,))

    coh_clean_list = []
    coh_noisy_list = []

    for s in starts:
        k = s.item()

        # Clean window: (B, N) → flat (B*N, 1)
        window_clean = d_clean[k : k + B]  # (B, N)
        pred_clean = window_clean.reshape(B * N, 1)
        coh_clean_list.append(
            consecutive_coherence_loss(pred_clean, batch_size=B, n_neurons=N).item()
        )

        # Noisy window
        window_noisy = d_noisy[k : k + B]
        pred_noisy = window_noisy.reshape(B * N, 1)
        coh_noisy_list.append(
            consecutive_coherence_loss(pred_noisy, batch_size=B, n_neurons=N).item()
        )

    mean_clean = sum(coh_clean_list) / len(coh_clean_list)
    mean_noisy = sum(coh_noisy_list) / len(coh_noisy_list)
    ratio = mean_noisy / max(mean_clean, 1e-30)

    print(f"\nCoherence loss (mean over {n_windows} windows of B={B}):")
    print(f"  Clean: {mean_clean:.6f}")
    print(f"  Noisy: {mean_noisy:.6f}")
    print(f"  Ratio (noisy/clean): {ratio:.2f}")

    # --- Condition (1): noise ratio ≥ 3× ---
    cond1 = ratio >= 3.0
    print(f"\nCondition 1 — ratio ≥ 3.0: {ratio:.2f} → {'OK' if cond1 else 'FAIL'}")

    # --- Condition (2): absolute noisy value in [1e-3, 1e3] ---
    cond2 = 1e-3 <= mean_noisy <= 1e3
    print(
        f"Condition 2 — noisy coherence in [1e-3, 1e3]: {mean_noisy:.6f} "
        f"→ {'OK' if cond2 else 'FAIL'}"
    )

    # --- Condition (3): differentiability ---
    # Create a small differentiable tensor and check gradient flows
    B_test = 4
    N_test = 100
    pred_test = torch.randn(B_test * N_test, 1, requires_grad=True)
    loss_test = consecutive_coherence_loss(pred_test, batch_size=B_test, n_neurons=N_test)
    loss_test.backward()
    grad_norm = pred_test.grad.norm().item()
    cond3 = grad_norm > 0
    print(
        f"Condition 3 — gradient norm > 0: {grad_norm:.6f} → {'OK' if cond3 else 'FAIL'}"
    )

    # --- Also verify batch_size=1 edge case returns zero ---
    pred_single = torch.randn(N_test, 1)
    loss_single = consecutive_coherence_loss(pred_single, batch_size=1, n_neurons=N_test)
    assert loss_single.item() == 0.0, f"batch_size=1 should return 0, got {loss_single.item()}"
    print("Edge case — batch_size=1 returns 0: OK")

    # --- Verdict ---
    all_pass = cond1 and cond2 and cond3
    if all_pass:
        print(
            f"\nPASS: consecutive coherence loss — noise ratio={ratio:.1f}x, "
            f"noisy={mean_noisy:.4f}, grad_norm={grad_norm:.4f}"
        )
    else:
        failures = []
        if not cond1:
            failures.append(f"ratio {ratio:.2f} < 3.0")
        if not cond2:
            failures.append(f"noisy coherence {mean_noisy:.6f} outside [1e-3, 1e3]")
        if not cond3:
            failures.append(f"gradient norm {grad_norm:.6f} ≤ 0")
        print(f"\nFAIL: {'; '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
