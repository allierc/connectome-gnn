"""Early bifurcation detector for flyvis training runs.

Phase R finding: conn_R2 at the *first few training checkpoints* (~2-7% of
total steps) separates seeds that will converge (final conn_R2 >= 0.90) from
those that will fail.  The improvement rate is identical between CONV and FAIL
seeds — the *level* at the early checkpoint is what differs.  This means the
random initialisation determines the optimisation basin within the first ~2000
gradient steps.

This module provides:
  - Trajectory loading from the r2_trajectory log directory
  - A threshold classifier on early-checkpoint conn_R2
  - Threshold optimisation (grid search)
  - Voltage-based SNR analysis explaining *why* the landscape is multi-basin
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """One training run's conn_R2 trajectory."""
    iter_num: int
    steps: List[int] = field(default_factory=list)
    conn_r2: List[float] = field(default_factory=list)

    @property
    def final_r2(self) -> float:
        return self.conn_r2[-1] if self.conn_r2 else float("nan")

    def r2_at_checkpoint(self, idx: int) -> Optional[float]:
        """Return conn_R2 at the given checkpoint index (0-based)."""
        if 0 <= idx < len(self.conn_r2):
            return self.conn_r2[idx]
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trajectories(traj_dir: str) -> List[Trajectory]:
    """Load all iter_*.log CSV files from *traj_dir*."""
    files = sorted(glob.glob(os.path.join(traj_dir, "iter_*.log")))
    trajs: List[Trajectory] = []
    for f in files:
        iter_num = int(os.path.basename(f).replace("iter_", "").replace(".log", ""))
        with open(f) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        if len(rows) < 3:
            continue  # too short to classify
        t = Trajectory(iter_num=iter_num)
        for r in rows:
            t.steps.append(int(r["iteration"]))
            t.conn_r2.append(float(r["connectivity_r2"]))
        trajs.append(t)
    return trajs


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_early(early_r2: float, threshold: float = 0.862) -> bool:
    """Predict convergence from an early checkpoint's conn_R2.

    Returns True  → predicted to converge  (final conn_R2 >= 0.90)
    Returns False → predicted to fail       (final conn_R2 <  0.90)
    """
    return early_r2 >= threshold


@dataclass
class ClassifierMetrics:
    accuracy: float
    n_correct: int
    n_total: int
    n_true_pos: int   # predicted CONV, actually CONV
    n_false_pos: int  # predicted CONV, actually FAIL
    n_true_neg: int   # predicted FAIL, actually FAIL
    n_false_neg: int  # predicted FAIL, actually CONV
    threshold: float
    checkpoint_idx: int
    baseline_accuracy: float  # always-predict-majority baseline


def evaluate_classifier(
    trajs: List[Trajectory],
    checkpoint_idx: int = 1,
    threshold: float = 0.862,
    final_threshold: float = 0.90,
) -> ClassifierMetrics:
    """Evaluate the threshold classifier on loaded trajectories.

    checkpoint_idx: 0-based index into the trajectory checkpoints.
        idx=1 is the first real checkpoint after the trivial step-1 entry.
    """
    tp = fp = tn = fn = 0
    n_conv = 0
    for t in trajs:
        early = t.r2_at_checkpoint(checkpoint_idx)
        if early is None:
            continue
        pred_conv = classify_early(early, threshold)
        actual_conv = t.final_r2 >= final_threshold
        if actual_conv:
            n_conv += 1
        if pred_conv and actual_conv:
            tp += 1
        elif pred_conv and not actual_conv:
            fp += 1
        elif not pred_conv and not actual_conv:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    baseline = max(n_conv, total - n_conv) / total if total > 0 else 0.0
    return ClassifierMetrics(
        accuracy=(tp + tn) / total if total > 0 else 0.0,
        n_correct=tp + tn,
        n_total=total,
        n_true_pos=tp,
        n_false_pos=fp,
        n_true_neg=tn,
        n_false_neg=fn,
        threshold=threshold,
        checkpoint_idx=checkpoint_idx,
        baseline_accuracy=baseline,
    )


def find_optimal_threshold(
    trajs: List[Trajectory],
    checkpoint_idx: int = 1,
    final_threshold: float = 0.90,
    search_lo: float = 0.50,
    search_hi: float = 0.96,
    search_step: float = 0.001,
) -> Tuple[float, float]:
    """Grid-search for the threshold that maximises accuracy.

    Returns (best_threshold, best_accuracy).
    """
    best_thr = search_lo
    best_acc = 0.0
    thr = search_lo
    while thr <= search_hi:
        m = evaluate_classifier(trajs, checkpoint_idx, thr, final_threshold)
        if m.accuracy > best_acc:
            best_acc = m.accuracy
            best_thr = thr
        thr += search_step
    return best_thr, best_acc


# ---------------------------------------------------------------------------
# Voltage-based mechanistic analysis
# ---------------------------------------------------------------------------

def voltage_spectral_analysis(
    v_clean: "torch.Tensor",  # (T, N)
    v_noisy: "torch.Tensor",  # (T, N)
    max_components: int = 200,
) -> dict:
    """Analyse the spectral structure of the voltage data.

    Returns a dict with:
      - effective_rank: number of singular values capturing 99% of variance
      - condition_number: ratio of largest to effective_rank-th singular value
      - snr_db: empirical signal-to-noise ratio in dB
      - top_sv_ratio: fraction of variance in the top singular value

    A high condition number with moderate effective rank explains why the
    W-recovery landscape has multiple basins: the dominant modes are easy
    to fit (all seeds agree), but the lower modes are noisy and seed-
    dependent — creating the observed early bifurcation.
    """
    import torch

    # Work on temporal derivatives (what the ODE actually fits)
    dv_clean = (v_clean[1:] - v_clean[:-1]).numpy()  # (T-1, N)
    dv_noisy = (v_noisy[1:] - v_noisy[:-1]).numpy()

    # Subsample for speed — SVD on (T, N) with T=64k is fine for N=13741
    # but we only need spectral structure, so keep it all
    n_components = min(max_components, dv_clean.shape[1])

    # Singular values of clean derivatives
    # Use randomized SVD (via partial) for speed
    U, s_clean, Vt = np.linalg.svd(dv_clean, full_matrices=False)
    s_clean = s_clean[:n_components]

    # Effective rank: number of SVs capturing 99% of total variance
    var_explained = np.cumsum(s_clean ** 2) / np.sum(s_clean ** 2)
    effective_rank = int(np.searchsorted(var_explained, 0.99)) + 1

    # Condition number (1st SV / effective_rank-th SV)
    cond = float(s_clean[0] / s_clean[min(effective_rank - 1, len(s_clean) - 1)])

    # SNR: compare clean signal power to noise power in derivative space
    noise_dv = dv_noisy - dv_clean
    signal_power = np.mean(dv_clean ** 2)
    noise_power = np.mean(noise_dv ** 2)
    snr_db = float(10.0 * np.log10(signal_power / noise_power)) if noise_power > 0 else float("inf")

    # Top SV ratio
    top_sv_ratio = float(s_clean[0] ** 2 / np.sum(s_clean ** 2))

    return {
        "effective_rank": effective_rank,
        "condition_number": cond,
        "snr_db": snr_db,
        "top_sv_ratio": top_sv_ratio,
        "n_neurons": dv_clean.shape[1],
        "n_timesteps": dv_clean.shape[0],
    }
