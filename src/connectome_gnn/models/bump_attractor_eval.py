"""Evaluation + snapshot helpers for the bump-attractor sign-locked RNN
family — shared between the drosophila CX path-integration model and the
zebrafish dIPN swim-integration model.

These helpers are duck-typed against any module exposing:
    .dt          (float)            — Euler step
    .n_units     (int)              — recurrent unit count
    .n_input     (int)              — input dim (1 = translation-only,
                                       3 = rotation-only [legacy CX],
                                       4 = rotation+translation)
    .W_rec       (Tensor (N, N))    — effective recurrent weight (read-only)
    forward(u)  -> (y_hat, h_buf)  — (B, T, n_in) -> (B, T, n_out), (B, T, N)

so they work on `teachers.JaneliaCxRNN`, `models.DrosophilaCxTaskRNN`, and
`models.ZebrafishHdTaskRNN`. Rotation channels are in u[:, :, 0] (ω) +
heading cue [cosθ0, sinθ0]; translation channel is v_fwd in u[:, :, 1]
(both mode) or u[:, :, 0] (translation-only mode).

History: lifted out of `teachers/janelia_cx_teacher.py` to keep the new
`data_train_task` from importing the teacher module. Renamed from
`drosophila_cx_eval.py` to drop the species prefix once the zebrafish
HD trainer started using it. The teacher re-exports these names for
backwards compat.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from connectome_gnn.generators.utils import generate_path_integration_batch


# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------


def path_integration_accuracy(
    net,
    n_trials: int = 64,
    n_steps: int = 100,
    device: str = "cpu",
) -> float:
    """Mean cosine similarity between predicted and true head direction.

    1.0 means perfect path integration; well-converged runs reach ~0.95+
    on the test set after 10 epochs. Skips the first 10 steps
    (initial-condition lead-in) before scoring.
    """
    net.eval()
    with torch.no_grad():
        batch = generate_path_integration_batch(n_trials, n_steps, device=device)
        y_hat, _ = net(batch.stimulus)
        warmup = 10
        y_hat_n = y_hat[:, warmup:, :] / (
            y_hat[:, warmup:, :].norm(dim=-1, keepdim=True) + 1e-8
        )
        y_n = batch.target[:, warmup:, :]
        acc = (y_hat_n * y_n).sum(dim=-1).mean().item()
    net.train()
    return acc


def path_integration_accuracy_from_data(
    net,
    u: torch.Tensor,           # (B, T, 3)
    y: torch.Tensor,           # (B, T, 2)
    *,
    warmup: int = 10,
    batch_size: int = 256,
    show_progress: bool = True,
) -> float:
    """Same metric as `path_integration_accuracy`, but on a pre-built
    (u, y) test split (the trainer already has this in GPU memory).

    Heading-bin ablation (net.use_heading_bins): the model emits K-bin logits,
    not a 2-D (cos,sin) pair, so we first decode the logits to (cos,sin) via the
    softmax circular mean — the same convention as
    heading_bins.softmax_logits_to_cos_sin_np and the deterministic-sweep
    decode — and then apply the identical cosine-to-target metric.
    """
    from tqdm import tqdm

    use_bins = bool(getattr(net, "use_heading_bins", False))
    K_bins = int(getattr(net, "n_heading_bins", 0))
    bin_centers = None
    if use_bins:
        from connectome_gnn.models.heading_bins import bin_centers_torch

    net.eval()
    cosines = []
    n_trials = u.shape[0]
    iterator = range(0, n_trials, batch_size)
    if show_progress:
        n_batches = (n_trials + batch_size - 1) // batch_size
        iterator = tqdm(iterator, total=n_batches, ncols=100,
                        desc=f"pi_acc test (B={batch_size}, T={u.shape[1]})",
                        leave=False)
    with torch.no_grad():
        for i in iterator:
            yh, _ = net(u[i : i + batch_size])
            yh = yh[:, warmup:, :]
            if use_bins:
                # (B, T, K) logits -> circular-mean (cos, sin).
                if bin_centers is None:
                    bin_centers = bin_centers_torch(
                        K_bins, device=yh.device, dtype=yh.dtype)
                p = torch.softmax(yh, dim=-1)
                yh = torch.stack((( p * torch.cos(bin_centers)).sum(dim=-1),
                                  ( p * torch.sin(bin_centers)).sum(dim=-1)),
                                 dim=-1)
            yh_n = yh / (yh.norm(dim=-1, keepdim=True) + 1e-8)
            yt = y[i : i + batch_size, warmup:, :]
            c = (yh_n * yt).sum(dim=-1).mean().item()
            cosines.append(c)
            if show_progress:
                iterator.set_postfix(pi_acc=f"{float(np.mean(cosines)):.4f}")
    net.train()
    return float(np.mean(cosines))


def bump_fwhm(
    net,
    epg_indices: np.ndarray,
    epg_ix: np.ndarray,
    *,
    n_trials: int = 64,
    n_steps: int = 100,
    device: str = "cpu",
    n_glom: int = 16,
    z_thresh: float = 1.0,
) -> float:
    """Mean bump width (radians) at the last frame of a fresh batch.

    Bin EPG firing rates into `n_glom` glomerular wedges, z-score per trial,
    and count contiguous wedges around the peak with z > `z_thresh`.
    Returns nan if no trial has a peak above threshold.
    """
    net.eval()
    with torch.no_grad():
        batch = generate_path_integration_batch(n_trials, n_steps, device=device)
        _, h = net(batch.stimulus)
    net.train()

    r_epg = torch.sigmoid(h[:, -1, epg_indices]).cpu().numpy()
    epg_ix_arr = np.asarray(epg_ix, dtype=int)
    glom_act = np.zeros((r_epg.shape[0], n_glom), dtype=np.float32)
    for g in range(n_glom):
        mask = epg_ix_arr == g
        if mask.any():
            glom_act[:, g] = r_epg[:, mask].mean(axis=1)

    mu = glom_act.mean(axis=1, keepdims=True)
    sigma = glom_act.std(axis=1, keepdims=True) + 1e-12
    z = (glom_act - mu) / sigma

    wedge_rad = 2.0 * np.pi / n_glom
    fwhms = []
    c = n_glom // 2
    for b in range(z.shape[0]):
        v = z[b]
        peak = int(np.argmax(v))
        if v[peak] <= z_thresh:
            continue
        v_rolled = np.roll(v, c - peak)
        left = c
        while left - 1 >= 0 and v_rolled[left - 1] > z_thresh:
            left -= 1
        right = c
        while right + 1 < n_glom and v_rolled[right + 1] > z_thresh:
            right += 1
        fwhms.append((right - left + 1) * wedge_rad)

    if not fwhms:
        return float("nan")
    return float(np.mean(fwhms))


# ---------------------------------------------------------------------------
# Snapshot rollout + figures
# ---------------------------------------------------------------------------


def _deterministic_sweep_rollout(
    net,
    *,
    n_steps: int,
    omega_deg_per_s: float = 60.0,
    v_fwd_per_s: float = 1.0,
    theta0_rad: float = 0.0,
    device: str,
) -> dict:
    """One trial with **constant ω and/or constant v_fwd**, no OU noise.

    Input shape adapts to ``net.n_input`` so the same probe works across
    every swim_integration sub-task:

      n_input == 3 → u = [ω, cosθ0, sinθ0]                (rotation-only)
      n_input == 1 → u = [v_fwd]                          (translation-only)
      n_input == 4 → u = [ω, v_fwd, cosθ0, sinθ0]         (both / position_2d)

    Constant drives from t=0 (no trial-start zeroing) — breaks parity with
    OU training data for a single frame, but produces a clean flat trace
    in the panel. The 10-frame warmup in any metric computation absorbs
    that one-frame offset.

    The returned dict always carries the I/O buffers; the
    ground-truth + decoded *integrated* quantities are present only for
    the integrators the model implements (detected from ``net.n_output``):

      always:                u, y_pred, h, r, n_steps, dt_s,
                             omega_deg_per_s, v_fwd_per_s
      rotation in net:       true_theta, decoded_theta
      translation in net:    true_xi, decoded_xi   (n_output ∈ {1, 3} —
                             scalar forward-axis displacement)
      position_2d in net:    true_xy, decoded_xy   (n_output == 4, shape (T, 2)
                             true 2D path integration GT computed from the
                             same constant drives via cumsum of v_fwd ·
                             (cos θ(t), sin θ(t)) · Δt)
    """
    import math
    T = int(n_steps)
    dt = float(net.dt)
    n_in = int(getattr(net, "n_input", 3))
    n_out = int(getattr(net, "n_output", 0))

    # Heading-bin ablation: rotation-only model with K-bin one-hot cue and
    # K-bin logit readout (n_in = 1 + K, n_out = K). The standard cos/sin
    # rotation-only branch (n_in == 3, n_out == 2) below is the
    # use_heading_bins=False reference; bins mode follows the same code
    # path with a different cue and decode step. has_rot is forced True so
    # the heading GT/decode block is populated regardless of the K-bin
    # n_out value (which would otherwise look like a "both"-mode n_out=3
    # ξ-bearing model and break the GT/decoded slicing below).
    use_bins = bool(getattr(net, "use_heading_bins", False))
    K_bins = int(getattr(net, "n_heading_bins", 0))

    # n_input decides what u channels look like; n_output decides which
    # GT/decoded integrators we report (so the same u shape can serve both
    # the scalar_xi "both" model (n_out=3) and the position_2d model
    # (n_out=4) without confusing ξ with x).
    # ω occupies channel 0 in every rotation-bearing input layout: the
    # rotation-only (3) / both (4) gates and the proprioception-split (5)
    # gate [ω, v_extero, v_proprio, cosθ₀, sinθ₀]. n_in ∈ {1, 2} are the
    # translation-only layouts (2 = propriocep-split, two parallel v_fwd
    # ports) and carry no heading drive.
    has_rot = use_bins or (n_in in (3, 4, 5))
    has_trans_xi = (not use_bins) and (n_out in (1, 3))   # scalar ξ in y_pred
    has_xy       = (not use_bins) and (n_out == 4)         # 2D (x, y) in y_pred[:, 2:4]
    if not (has_rot or has_trans_xi or has_xy):
        raise ValueError(
            f"_deterministic_sweep_rollout: unsupported (n_in, n_out) = "
            f"({n_in}, {n_out}) — expected n_in ∈ {{1, 2, 3, 4, 5}} with "
            f"n_out ∈ {{1, 2, 3, 4}}."
        )

    omega = np.full((1, T), float(omega_deg_per_s), dtype=np.float32)
    v_fwd = np.full((1, T), float(v_fwd_per_s),     dtype=np.float32)

    if use_bins:
        # Heading-bin ablation: input layout is [ω, K-bin one-hot cue at t=0].
        # Same bin convention as the trainer (connectome_gnn.models.heading_bins).
        from connectome_gnn.models.heading_bins import theta_to_bin_np
        u = np.zeros((1, T, 1 + K_bins), dtype=np.float32)
        u[:, :, 0] = omega
        bin_idx = int(theta_to_bin_np(float(theta0_rad), K_bins).item())
        u[:, 0, 1 + bin_idx] = 1.0
    elif n_in == 3:
        u = np.zeros((1, T, 3), dtype=np.float32)
        u[:, :, 0] = omega
        u[:, 0, 1] = math.cos(float(theta0_rad))
        u[:, 0, 2] = math.sin(float(theta0_rad))
    elif n_in == 1:
        u = np.zeros((1, T, 1), dtype=np.float32)
        u[:, :, 0] = v_fwd
    elif n_in == 2:
        # propriocep-split translation-only: [v_extero, v_proprio], both
        # carrying the same clean v_fwd drive.
        u = np.zeros((1, T, 2), dtype=np.float32)
        u[:, :, 0] = v_fwd
        u[:, :, 1] = v_fwd
    elif n_in == 5:
        # propriocep-split rotation + translation:
        # [ω, v_fwd, ω_proprio, cosθ₀, sinθ₀]. Channel 2 is the angular
        # efference copy ω_proprio (= ω) routed to motor_efferent.
        u = np.zeros((1, T, 5), dtype=np.float32)
        u[:, :, 0] = omega
        u[:, :, 1] = v_fwd
        u[:, :, 2] = omega
        u[:, 0, 3] = math.cos(float(theta0_rad))
        u[:, 0, 4] = math.sin(float(theta0_rad))
    else:  # n_in == 4
        u = np.zeros((1, T, 4), dtype=np.float32)
        u[:, :, 0] = omega
        u[:, :, 1] = v_fwd
        u[:, 0, 2] = math.cos(float(theta0_rad))
        u[:, 0, 3] = math.sin(float(theta0_rad))

    u_t = torch.from_numpy(u).to(device)
    # eval()/train() toggle so the deterministic-sweep is truly deterministic
    # (TaskRNN's training-mode forward injects Gaussian noise when
    # noise_recurrent_level > 0).
    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            y_hat, h = net(u_t)
    finally:
        if was_training:
            net.train()
    r = torch.sigmoid(h[0]).cpu().numpy()
    y_pred = y_hat[0].cpu().numpy()

    out: dict = {
        "u": u[0],
        "y_pred": y_pred,
        "h": h[0].cpu().numpy(),
        "r": r,
        "n_steps": T,
        "omega_deg_per_s": float(omega_deg_per_s),
        "v_fwd_per_s":     float(v_fwd_per_s),
        "dt_s": dt,
    }
    if has_rot:
        # Heading occupies y_pred[:, 0:2] in every rotation-bearing mode
        # (rotation-only: n_output=2 [cos, sin]; both: n_output=3
        # [cos, sin, ξ]). cumsum(deg2rad(ω)) * dt is the unwrapped GT angle.
        omega_rad = np.deg2rad(omega)
        out["true_theta"] = float(theta0_rad) + np.cumsum(omega_rad, axis=1)[0] * dt
        if use_bins:
            # K-bin logits → circular-mean softmax decode. Also synthesise a
            # (T, 2) cos/sin view of y_pred so downstream plotters
            # (plot_evolution and friends) — which read y_pred[:, :2] and
            # atan2 it — keep working unchanged. The synthetic cos/sin pair
            # has magnitude in (0, 1] reflecting softmax sharpness; the
            # plotters only consume its angle, so the magnitude is harmless.
            from connectome_gnn.models.heading_bins import (
                softmax_logits_to_cos_sin_np,
                softmax_logits_to_decoded_theta_np,
            )
            out["decoded_theta"] = softmax_logits_to_decoded_theta_np(
                y_pred, K_bins)
            cs = softmax_logits_to_cos_sin_np(y_pred, K_bins)
            # Replace y_pred in the rollout with a 2-col cos/sin view so the
            # plotter (which reads `rollout["y_pred"]` in some panels and
            # `rollout["decoded_theta"]` in others) sees a consistent
            # 2-channel layout matching the cos/sin baseline.
            out["y_pred"] = cs
            # Keep the raw K-bin logits available for any downstream caller
            # that wants the full distribution.
            out["y_pred_bins"] = y_pred
        else:
            out["decoded_theta"] = np.arctan2(y_pred[:, 1], y_pred[:, 0])
    if has_trans_xi:
        # ξ = ∫ v_fwd dt (linear, unbounded). Column placement depends on
        # n_output: translation-only (n_out=1) puts ξ in y_pred[:, 0];
        # scalar_xi both mode (n_out=3) puts ξ in y_pred[:, 2] after
        # [cosθ, sinθ]. position_2d (n_out=4) does NOT have ξ —
        # y_pred[:, 2:4] is (x, y) there, gated separately below.
        out["true_xi"] = np.cumsum(v_fwd, axis=1)[0] * dt
        xi_col = 2 if n_out == 3 else 0
        out["decoded_xi"] = y_pred[:, xi_col]

    if has_xy:
        # 2D path integration — model emits [cosθ, sinθ, x, y]. GT path
        # is cumsum of v_fwd · (cos θ(t), sin θ(t)) · Δt; for constant
        # drives this traces a circle of radius v_fwd / |ω_rad| centred
        # perpendicular to the initial heading.
        theta_gt = float(theta0_rad) + np.cumsum(np.deg2rad(omega), axis=1)[0] * dt
        vx = v_fwd[0] * np.cos(theta_gt)
        vy = v_fwd[0] * np.sin(theta_gt)
        x_gt = np.cumsum(vx) * dt
        y_gt = np.cumsum(vy) * dt
        x_gt[0] = 0.0
        y_gt[0] = 0.0
        out["true_xy"] = np.stack([x_gt, y_gt], axis=-1).astype(np.float32)
        out["decoded_xy"] = y_pred[:, 2:4].astype(np.float32)

    return out


def _rollout_heading_metrics(
    net,
    *,
    n_steps: int,
    omega_deg_per_s: float,
    device: str,
    warmup: int = 10,
) -> tuple[float, float]:
    """RMSE (deg) and Pearson correlation on a deterministic sweep rollout.

    - RMSE is computed on the wrapped angular residual decoded − true.
    - Pearson is computed between the unwrapped decoded trajectory and the
      (already-monotone) ground-truth trajectory, after a short warmup.
    Returns (nan, nan) on failure, degenerate input, or when the model
    has no heading output (translation-only mode).
    """
    try:
        rollout = _deterministic_sweep_rollout(
            net, n_steps=n_steps,
            omega_deg_per_s=omega_deg_per_s, device=device,
        )
    except Exception:
        return float("nan"), float("nan")
    if "true_theta" not in rollout:
        # Translation-only model: no heading to score.
        return float("nan"), float("nan")
    true_theta = np.asarray(rollout["true_theta"])
    decoded = np.asarray(rollout["decoded_theta"])
    if true_theta.size <= warmup:
        return float("nan"), float("nan")
    err = np.angle(np.exp(1j * (decoded[warmup:] - true_theta[warmup:])))
    rmse_deg = float(np.degrees(np.sqrt(np.mean(err ** 2))))
    decoded_unwrapped = np.unwrap(decoded[warmup:])
    if (decoded_unwrapped.std() < 1e-8
            or true_theta[warmup:].std() < 1e-8):
        return rmse_deg, float("nan")
    pearson = float(np.corrcoef(decoded_unwrapped, true_theta[warmup:])[0, 1])
    return rmse_deg, pearson


def _rollout_translation_metrics(
    net,
    *,
    n_steps: int,
    v_fwd_per_s: float,
    device: str,
    warmup: int = 10,
) -> tuple[float, float]:
    """Translation analog of ``_rollout_heading_metrics``.

    - RMSE is computed on the linear residual decoded_xi − true_xi (units
      match v_fwd × time — there is no wrapping).
    - Pearson is computed between decoded_xi and true_xi after warmup.
    Returns (nan, nan) on failure, degenerate input, or when the model
    has no displacement output (rotation-only mode).
    """
    try:
        rollout = _deterministic_sweep_rollout(
            net, n_steps=n_steps,
            v_fwd_per_s=v_fwd_per_s, device=device,
        )
    except Exception:
        return float("nan"), float("nan")
    if "true_xi" not in rollout:
        return float("nan"), float("nan")
    true_xi = np.asarray(rollout["true_xi"])
    decoded = np.asarray(rollout["decoded_xi"])
    if true_xi.size <= warmup:
        return float("nan"), float("nan")
    err = decoded[warmup:] - true_xi[warmup:]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if decoded[warmup:].std() < 1e-8 or true_xi[warmup:].std() < 1e-8:
        return rmse, float("nan")
    pearson = float(np.corrcoef(decoded[warmup:], true_xi[warmup:])[0, 1])
    return rmse, pearson


def _rollout_position_metrics(
    net,
    *,
    n_steps: int,
    omega_deg_per_s: float,
    v_fwd_per_s: float,
    device: str,
    warmup: int = 10,
) -> tuple[float, float]:
    """2D position analog of the heading / translation metric helpers.

    - RMSE is the Euclidean per-frame distance between decoded (x̂, ŷ) and
      GT (x, y), averaged over time (after a warmup).
    - Pearson is the average of the per-axis correlations
      (corr(x̂, x) + corr(ŷ, y)) / 2 — a single scalar that captures how
      well both axes track.
    Returns (nan, nan) on failure, degenerate input, or when the model
    has no 2D-position output (i.e. n_output != 4).
    """
    try:
        rollout = _deterministic_sweep_rollout(
            net, n_steps=n_steps,
            omega_deg_per_s=omega_deg_per_s,
            v_fwd_per_s=v_fwd_per_s,
            device=device,
        )
    except Exception:
        return float("nan"), float("nan")
    if "true_xy" not in rollout:
        return float("nan"), float("nan")
    true_xy = np.asarray(rollout["true_xy"])     # (T, 2)
    decoded = np.asarray(rollout["decoded_xy"])  # (T, 2)
    if true_xy.shape[0] <= warmup:
        return float("nan"), float("nan")
    err = decoded[warmup:] - true_xy[warmup:]
    rmse = float(np.sqrt(np.mean(err ** 2)))  # mean of squared error over all (t, axis)
    rs = []
    for axis in range(2):
        if (decoded[warmup:, axis].std() > 1e-8
                and true_xy[warmup:, axis].std() > 1e-8):
            rs.append(float(np.corrcoef(decoded[warmup:, axis],
                                         true_xy[warmup:, axis])[0, 1]))
    if not rs:
        return rmse, float("nan")
    pearson = float(np.mean(rs))
    return rmse, pearson


def _trajectory_metrics_2d(true_xy: np.ndarray, decoded_xy: np.ndarray,
                            warmup: int = 10) -> dict:
    """Extended 2-D trajectory comparison.

    Returns a dict with:
        euclid_rmse_per_frame:   sqrt(mean ||(x̂,ŷ) - (x,y)||²)
        mean_euclid_error:       mean ||(x̂,ŷ) - (x,y)||
        max_euclid_error:        max  ||(x̂,ŷ) - (x,y)||
        endpoint_error:          ||(x̂,ŷ)_T - (x,y)_T||
        path_length_true:        sum ||(x,y)_{t+1} - (x,y)_t||
        path_length_decoded:     sum ||(x̂,ŷ)_{t+1} - (x̂,ŷ)_t||
        path_length_ratio:       decoded / true
        cosine_velocity:         mean cosine between dxy_dt true and decoded
        pearson_x, pearson_y:    per-axis Pearson r
        pearson_mean:            (pearson_x + pearson_y) / 2
    """
    true_xy = np.asarray(true_xy)
    decoded_xy = np.asarray(decoded_xy)
    if true_xy.shape[0] <= warmup + 2:
        return {}
    tx = true_xy[warmup:]
    dx = decoded_xy[warmup:]
    err = dx - tx
    euclid = np.sqrt((err ** 2).sum(axis=-1))
    out = {
        "euclid_rmse_per_frame": float(np.sqrt(np.mean(err ** 2))),
        "mean_euclid_error":     float(np.mean(euclid)),
        "max_euclid_error":      float(np.max(euclid)),
        "endpoint_error":        float(np.linalg.norm(dx[-1] - tx[-1])),
    }
    dtx = np.diff(tx, axis=0)
    ddx = np.diff(dx, axis=0)
    pl_true = float(np.sum(np.linalg.norm(dtx, axis=-1)))
    pl_dec  = float(np.sum(np.linalg.norm(ddx, axis=-1)))
    out["path_length_true"]    = pl_true
    out["path_length_decoded"] = pl_dec
    out["path_length_ratio"]   = (pl_dec / pl_true) if pl_true > 0 else float("nan")
    norm_t = np.linalg.norm(dtx, axis=-1) + 1e-12
    norm_d = np.linalg.norm(ddx, axis=-1) + 1e-12
    cos_v = (dtx * ddx).sum(axis=-1) / (norm_t * norm_d)
    out["cosine_velocity"] = float(np.mean(cos_v))
    rxs = []
    for axis, name in ((0, "pearson_x"), (1, "pearson_y")):
        if dx[:, axis].std() > 1e-8 and tx[:, axis].std() > 1e-8:
            r = float(np.corrcoef(dx[:, axis], tx[:, axis])[0, 1])
        else:
            r = float("nan")
        out[name] = r
        if np.isfinite(r):
            rxs.append(r)
    out["pearson_mean"] = float(np.mean(rxs)) if rxs else float("nan")
    return out


def _trajectory_metrics_1d(true_y: np.ndarray, decoded_y: np.ndarray,
                            warmup: int = 10) -> dict:
    """Scalar-trajectory comparison (forward distance d, or heading drift).

    Returns: rmse, pearson, endpoint_error, growth_rate_ratio (decoded /
    true mean slope), and mean / max absolute error.
    """
    ty = np.asarray(true_y).reshape(-1)
    dy = np.asarray(decoded_y).reshape(-1)
    if ty.size <= warmup + 2:
        return {}
    ty = ty[warmup:]; dy = dy[warmup:]
    err = dy - ty
    out = {
        "rmse":              float(np.sqrt(np.mean(err ** 2))),
        "mean_abs_error":    float(np.mean(np.abs(err))),
        "max_abs_error":     float(np.max(np.abs(err))),
        "endpoint_error":    float(abs(dy[-1] - ty[-1])),
    }
    t_axis = np.arange(ty.size)
    if t_axis.std() > 1e-8:
        slope_true = float(np.polyfit(t_axis, ty, 1)[0])
        slope_dec  = float(np.polyfit(t_axis, dy, 1)[0])
        out["slope_true"]        = slope_true
        out["slope_decoded"]     = slope_dec
        out["growth_rate_ratio"] = (slope_dec / slope_true
                                     if abs(slope_true) > 1e-12 else float("nan"))
    if dy.std() > 1e-8 and ty.std() > 1e-8:
        out["pearson"] = float(np.corrcoef(dy, ty)[0, 1])
    else:
        out["pearson"] = float("nan")
    return out


def load_pi_fwhm_history(metrics_log_path: str):
    """Read pi_acc, fwhm_deg, and RMSE histories from a trainer metrics.log.

    Returns (pi_acc_hist, fwhm_hist, rmse_hist) where each is an
    (iterations, values) tuple of 1-D arrays, or None if the corresponding
    column is missing. Returns (None, None, None) if the file is
    missing/empty. RMSE is computed as sqrt(mse) from the metrics row.
    Used by both training-time snapshots and the offline figure script.
    """
    if not os.path.isfile(metrics_log_path):
        return None, None, None
    try:
        rows = np.genfromtxt(metrics_log_path, delimiter=",", names=True,
                              dtype=None, encoding="utf-8")
    except Exception:
        return None, None, None
    if rows.size == 0 or "iteration" not in rows.dtype.names:
        return None, None, None
    it = np.atleast_1d(rows["iteration"]).astype(np.float32)
    pi = (np.atleast_1d(rows["pi_acc"]).astype(np.float32)
          if "pi_acc" in rows.dtype.names else None)
    fw = (np.atleast_1d(rows["fwhm_deg"]).astype(np.float32)
          if "fwhm_deg" in rows.dtype.names else None)
    rmse = None
    if "mse" in rows.dtype.names:
        mse = np.atleast_1d(rows["mse"]).astype(np.float32)
        rmse = np.sqrt(np.maximum(mse, 0.0))
    return ((it, pi) if pi is not None else None,
            (it, fw) if fw is not None else None,
            (it, rmse) if rmse is not None else None)


def _build_test_trial(net, u_test, y_test, device, config):
    """Build a test_trial dict from a single OU / swim test trial for panel g.

    The trial is picked by sampling K candidates and keeping the one with
    the most informative target — defined as the column-wise max of
    |y_true - mean(y_true)| over time. For rotation-only this almost
    always picks a non-trivial trial (heading varies whenever ω is
    non-zero); for translation-only it skips the ~14% of trials that have
    no F/B events (v_fwd ≡ 0 ⇒ ξ ≡ 0, nothing to integrate) and finds
    one where displacement actually accumulates. For both-mode it picks
    a trial whose ξ varies — the heading half is always present anyway,
    so prioritising ξ-variation gives the most informative joint panel.

    Returns None when u_test / y_test are not provided (backwards-compat).
    """
    if u_test is None or y_test is None:
        return None
    import torch
    seed = int(getattr(config.training, "seed", 0)) + 17 if config else 17
    rng = np.random.default_rng(seed)
    n_test = u_test.shape[0]

    # Sample K candidate indices, score each on the spread of its
    # target trajectory, and take the most informative one. K=32 keeps
    # the snapshot cost negligible (K target reads, no extra forwards).
    # Fully random sampling stays the rotation-only fallback for any
    # exotic mode where the scoring is degenerate.
    K = int(min(32, n_test))
    cand_idx = np.sort(rng.choice(n_test, size=K, replace=False))
    if hasattr(y_test, "cpu"):
        y_cand = y_test[cand_idx].cpu().numpy()
    else:
        y_cand = np.asarray(y_test)[cand_idx]
    # Score = max column-wise centred amplitude (so a constant trial scores
    # ~0). For translation mode (y has the ξ column), this directly tracks
    # how far the network *should* drift over the trial — exactly the
    # quantity that's interesting to see decoded.
    if y_cand.size:
        y_centred = y_cand - y_cand.mean(axis=1, keepdims=True)
        scores = np.abs(y_centred).max(axis=(1, 2))
        best_in_cand = int(np.argmax(scores))
        trial_idx = int(cand_idx[best_in_cand])
    else:
        trial_idx = int(rng.integers(0, n_test))
    u_one = u_test[trial_idx]                    # (T, N_in) tensor or ndarray
    y_true = y_test[trial_idx]
    if hasattr(u_one, "cpu"):
        u_one_np = u_one.cpu().numpy()
        y_true_np = y_true.cpu().numpy()
    else:
        u_one_np = np.asarray(u_one)
        y_true_np = np.asarray(y_test[trial_idx])
    # Heading-bin ablation: the on-disk u_one_np is (T, 3) with the (cos θ₀,
    # sin θ₀) cue impulse at t=0; the trained model expects (T, 1+K) with a
    # one-hot K-bin cue. Convert u BEFORE the forward, then decode the K-bin
    # logits y_pred (T, K) back to a (T, 2) cos/sin view so the downstream
    # plot_evolution panel — which atan2's y_pred[:, :2] — works
    # unchanged. y_true is also rewritten to its (T, 2) cos/sin form (it
    # already IS cos/sin on disk; no conversion needed).
    use_bins = bool(getattr(net, "use_heading_bins", False))
    K_bins = int(getattr(net, "n_heading_bins", 0))
    with torch.no_grad():
        u_t = torch.from_numpy(u_one_np[None]).to(device) if not hasattr(u_one, "to") \
              else u_one[None].to(device)
        if use_bins:
            from connectome_gnn.models.heading_bins import (
                convert_cos_sin_input_to_bin_cue_torch,
            )
            u_t = convert_cos_sin_input_to_bin_cue_torch(u_t, K_bins)
        y_pred_t, _ = net(u_t)
    y_pred_np = y_pred_t[0].cpu().numpy()
    if use_bins:
        from connectome_gnn.models.heading_bins import (
            softmax_logits_to_cos_sin_np,
        )
        y_pred_np = softmax_logits_to_cos_sin_np(y_pred_np, K_bins)
    # Species-agnostic: drosophila uses task.path_integration, zebrafish
    # uses task.swim_integration. Fall back to net.dt if neither block
    # carries a dt (which would only happen for non-task models anyway).
    if config is not None:
        task_block = getattr(config, "task", None)
        sub = None
        for sub_name in ("path_integration", "swim_integration"):
            sub = getattr(task_block, sub_name, None) if task_block else None
            if sub is not None and getattr(sub, "dt", None) is not None:
                break
        dt = float(sub.dt) if sub is not None else float(net.dt)
    else:
        dt = float(net.dt)
    # rotation_mismatch task → panel g renders the two-integral-path view
    # (ω vs ω_proprio, θ_obs vs θ_pro, true vs decoded ∫(ω−ω_proprio)).
    _is_mismatch = (config is not None
                    and str(getattr(getattr(getattr(config, "task", None),
                                            "swim_integration", None),
                                    "target_kind", "")) == "rotation_mismatch")
    return dict(
        idx=trial_idx,
        u=u_one_np,
        y_true=y_true_np,
        y_pred=y_pred_np,
        dt=dt,
        label="OU test trial",
        mismatch=_is_mismatch,
    )


def _save_training_snapshot(
    *,
    net,
    log_dir: str,
    kinograph_dir: str,
    global_step: int,
    epoch: int,
    neuron_types: np.ndarray,
    type_names: list,
    epg_indices: np.ndarray,
    epg_glom_ix: np.ndarray,
    device: str,
    snapshot_n_steps: int,
    snapshot_omega_deg: float,
    snapshot_v_fwd: float = 1.0,
    iter_in_epoch: int | None = None,
    matrix_dir: str | None = None,    # backwards-compat; ignored
    config=None,
    u_test=None,
    y_test=None,
    calcium_panel=None,
) -> None:
    """Render the combined kinograph+matrix snapshot.

    The matrix is the top-left panel of the kinograph figure, so we no
    longer write a separate matrix-only PNG.
    """
    from connectome_gnn.plot_cx import cx_epg_directions

    # Filename: epoch_<E>_<ITER_IN_EPOCH>.png — readable curriculum position.
    # Falls back to step_<global_step>.png when iter_in_epoch isn't provided
    # (e.g. legacy callers).
    if iter_in_epoch is not None:
        name = f"epoch_{epoch}_{iter_in_epoch:05d}.png"
    else:
        name = f"step_{global_step:07d}.png"

    try:
        # Constant-ω and/or constant-v_fwd rollout at T=1000 — the snapshot
        # panel's `r=` matches the `r_roll_1k` printed in the trainer postfix
        # (also evaluated at T=1000) for rotation-bearing models. The
        # generalized rollout helper builds the input matching net.n_input
        # so the same call works in rotation-only, translation-only, and
        # both modes. `snapshot_n_steps` is kept in the signature for
        # backwards compatibility but is no longer used for the rollout
        # length (1000 is the canonical comparison horizon).
        rollout = _deterministic_sweep_rollout(
            net, n_steps=1000,
            omega_deg_per_s=snapshot_omega_deg,
            v_fwd_per_s=snapshot_v_fwd,
            device=device,
        )
        rollout["r_epg"] = rollout["r"][:, epg_indices]
        # rotation_mismatch: the constant-ω probe runs with ω_proprio = ω
        # (zero mismatch) and the 3rd output column is NOT a translation
        # distance here, so drop the translation keys → panel f renders
        # rotation-only (ω + HD). The mismatch is shown in panel g via the
        # two-integral-path view (_panel_trial_rollout).
        if config is not None and str(getattr(getattr(getattr(
                config, "task", None), "swim_integration", None),
                "target_kind", "")) == "rotation_mismatch":
            rollout.pop("true_xi", None)
            rollout.pop("decoded_xi", None)
        # Afferent population = union of the PEN-gate sub-population indicator
        # buffers (`_pen_ind_pena_l/r/penb_l/r`) populated by the model from
        # the loader's ``pen_subpop_ix``. Works species-agnostically: fly PEN
        # (PEN_a/PEN_b L/R) or zebrafish RIPN + pt-IPN L/R, whichever the
        # loader emitted. Falls back to a type-name lookup ("PEN" in name)
        # for older models that don't carry the indicator buffers.
        pen_indices_arr = None
        ind_keys = ("_pen_ind_pena_l", "_pen_ind_pena_r",
                    "_pen_ind_penb_l", "_pen_ind_penb_r")
        if all(hasattr(net, k) for k in ind_keys):
            union = sum(getattr(net, k) for k in ind_keys)
            idx = (union > 0).nonzero(as_tuple=True)[0].cpu().numpy()
            if idx.size:
                pen_indices_arr = idx.astype(np.int64)
        else:
            pen_type_idx = [i for i, n in enumerate(type_names)
                            if "PEN" in n and "PEG" not in n]
            if pen_type_idx:
                pen_idx_list: list[int] = []
                nt = np.asarray(neuron_types)
                for t in pen_type_idx:
                    pen_idx_list.extend(np.where(nt == t)[0].tolist())
                pen_indices_arr = np.array(sorted(pen_idx_list),
                                            dtype=np.int64)
        if pen_indices_arr is not None:
            rollout["r_pen"] = rollout["r"][:, pen_indices_arr]
        epg_theta = cx_epg_directions(epg_glom_ix)
        W_con_np = (net.W_con.detach().cpu().numpy()
                    if hasattr(net, "W_con") else None)

        # Use plot_evolution (two-row mode) for the training snapshot so
        # the in-training plot matches the paper figure exactly. Function
        # lives in connectome_gnn.plot_cx alongside every other CX-specific
        # plotting helper; it used to live in figures/drosophila_cx/
        # fig_evolution.py and be loaded via importlib here, which broke
        # when the figures/ directory was reorganised.
        from connectome_gnn.plot_cx import plot_evolution

        # Species-specific axis labels for panels d, e (afferent and
        # bump-carrying populations). Defaults to fly EPG/PEN via the base
        # class; zebrafish subclass overrides to r1π/dIPN and RIPN/pt-IPN.
        bump_label = getattr(type(net), "bump_label", "EPG")
        afferent_label = getattr(type(net), "afferent_label", "PEN")

        data = dict(
            net=net,
            config=config,
            W_rec=net.W_rec.detach().cpu().numpy(),
            W_con=W_con_np,
            neuron_types=neuron_types,
            type_names=type_names,
            pen_indices=pen_indices_arr,
            rollout=rollout,
            epg_theta=epg_theta,
            gain_data=[],        # third-row only; unused in n_rows=2
            test_trial=_build_test_trial(
                net, u_test, y_test, device, config),
            dt_s=float(net.dt),
            bump_label=bump_label,
            afferent_label=afferent_label,
            calcium_panel=calcium_panel,
        )
        plot_evolution(
            data, os.path.join(kinograph_dir, name), n_rows=2,
        )
    except Exception as exc:
        print(f"[bump_attractor_eval] kinograph snapshot failed @ step {global_step}: {exc}")

    # TaskGNN-only: render embedding scatter + g_phi / f_theta function
    # plots into tmp_training/{embedding,function/{g_phi,f_theta}}/.
    # No-op for sign_locked TaskRNN (no `a` / `g_phi` / `f_theta`).
    if config is not None and all(
        hasattr(net, name) for name in ("a", "g_phi", "f_theta")
    ):
        try:
            _plot_gnn_functions(
                net=net, config=config, log_dir=log_dir,
                global_step=global_step, device=device,
                neuron_types=neuron_types, type_names=type_names,
            )
        except Exception as exc:
            print(f"[bump_attractor_eval] gnn function plots failed @ step {global_step}: {exc}")


def _save_place_snapshot(net, log_dir, global_step, epoch, u_test, y_test,
                         device, trial_idx: int = 0):
    """4×1 place-cell training snapshot into tmp_training/place_evolution/.

    Panels: (a) arena trajectory true vs population-vector decoded; (b) the
    predicted place-code map at the final frame with the true position marked;
    (c) decoded vs true x/y over time; (d) the Net1 compass heading true vs
    decoded. ``u_test``/``y_test`` are the trainer's channel-sliced tensors
    (y_test cols [cosθ, sinθ, ξ, x, y])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = os.path.join(log_dir, "tmp_training", "place_evolution")
    os.makedirs(out_dir, exist_ok=True)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        T = int(min(u_test.shape[1], 1000))
        u = u_test[trial_idx:trial_idx + 1, :T]
        y_hat, _ = net(u)                               # (1, T, 3+K)
        K = int(net.net2_n_place)
        p = torch.softmax(y_hat[0, :, 3:3 + K], dim=-1)  # (T, K)
        centers = net.place_centers                      # (K, 2)
        xy_dec = (p @ centers).cpu().numpy()             # (T, 2)
        th_dec = torch.atan2(y_hat[0, :, 1], y_hat[0, :, 0]).cpu().numpy()
        p_np = p.cpu().numpy()
    if was_training:
        net.train()
    yt = y_test[trial_idx, :T].detach().cpu().numpy()
    xy_true = yt[:, 3:5]
    th_true = np.arctan2(yt[:, 1], yt[:, 0])
    A = float(net.arena_half)
    grid = int(round(np.sqrt(K)))
    t = np.arange(T) * float(net.dt)
    GT, PR = "tab:green", "black"

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.1))
    # (a) trajectory
    ax = axes[0]
    ax.plot(xy_true[:, 0], xy_true[:, 1], color=GT, lw=1.2, label="true")
    ax.plot(xy_dec[:, 0], xy_dec[:, 1], color=PR, lw=0.9, label="decoded")
    ax.add_patch(plt.Rectangle((-A, -A), 2 * A, 2 * A, fill=False, ec="0.5"))
    ax.set_xlim(-A * 1.05, A * 1.05); ax.set_ylim(-A * 1.05, A * 1.05)
    ax.set_aspect("equal"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("(a) arena trajectory"); ax.set_xlabel("x"); ax.set_ylabel("y")
    # (b) predicted place code map at the final frame
    ax = axes[1]
    tf = T - 1
    im = ax.imshow(p_np[tf].reshape(grid, grid), origin="lower",
                   extent=[-A, A, -A, A], cmap="viridis", aspect="equal")
    ax.plot(xy_true[tf, 0], xy_true[tf, 1], "o", mec="r", mfc="none",
            ms=11, mew=1.6)
    ax.set_title(f"(b) place code @ t={t[tf]:.1f}s (○ true)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    # (c) decoded vs true position over time
    ax = axes[2]
    ax.plot(t, xy_true[:, 0], color=GT, lw=1.0)
    ax.plot(t, xy_dec[:, 0], color=PR, lw=0.7)
    ax.plot(t, xy_true[:, 1], color=GT, lw=1.0, ls="--")
    ax.plot(t, xy_dec[:, 1], color=PR, lw=0.7, ls="--")
    ax.set_title("(c) x (—) / y (– –): true=green, decoded=black")
    ax.set_xlabel("time (s)"); ax.set_ylabel("position")
    # (d) Net1 compass heading
    ax = axes[3]
    ax.plot(t, np.angle(np.exp(1j * th_true)), color=GT, lw=0, marker=".", ms=2)
    ax.plot(t, np.angle(np.exp(1j * th_dec)), color=PR, lw=0, marker=".", ms=1)
    ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_ylim(-np.pi - 0.15, np.pi + 0.15)
    ax.set_title("(d) heading (Net1): true=green, decoded=black")
    ax.set_xlabel("time (s)"); ax.set_ylabel("HD (rad)")

    pos_rmse = float(np.sqrt(((xy_dec[10:] - xy_true[10:]) ** 2).sum(-1).mean()))
    fig.suptitle(f"place snapshot — step {global_step} (epoch {epoch}) — "
                 f"position RMSE = {pos_rmse:.3f} (arena ±{A:g})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, f"place_step_{global_step:06d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_gnn_functions(
    *,
    net, config, log_dir: str, global_step: int, device: str,
    neuron_types: np.ndarray, type_names: list,
) -> None:
    """Render TaskGNN embedding + per-type g_phi / f_theta function curves.

    Mirrors `plot_training_flyvis` in data_train_gnn: same three sub-plots,
    same filenames (`tmp_training/embedding/step_*.png`,
    `tmp_training/function/g_phi/step_*.png`,
    `tmp_training/function/f_theta/step_*.png`).
    """
    import matplotlib.pyplot as plt
    import torch

    from connectome_gnn.metrics import _batched_mlp_eval, _build_g_phi_features
    from connectome_gnn.plot import plot_embedding, plot_g_phi
    from connectome_gnn.utils import CustomColorMap, qualitative_colors

    name = f"step_{global_step:07d}.png"
    n_neurons = int(net.n_units)
    nt_np = np.asarray(neuron_types)
    n_types = len(type_names)
    cmap = CustomColorMap(config=config)

    # 1) Embedding scatter (a_0 vs a_1, coloured by neuron type)
    emb_dir = os.path.join(log_dir, 'tmp_training', 'embedding')
    os.makedirs(emb_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_embedding(ax, net, nt_np, n_types, cmap)
    plt.tight_layout()
    plt.savefig(os.path.join(emb_dir, name), dpi=87)
    plt.close(fig)

    # 2) g_phi function: v ∈ [-3, 3] on x (the GNN MLPs consume the raw
    # subthreshold state v ≡ h, no sigmoid wrap). Override
    # config.plotting.xlim/ylim for this voltage range, then restore so
    # other callers aren't affected.
    gphi_dir = os.path.join(log_dir, 'tmp_training', 'function', 'g_phi')
    os.makedirs(gphi_dir, exist_ok=True)
    orig_xlim = list(config.plotting.xlim)
    orig_ylim = list(config.plotting.ylim)
    try:
        config.plotting.xlim = [-3.0, 3.0]
        config.plotting.ylim = [-1.0, 1.0]
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_g_phi(ax, net, config, n_neurons, nt_np, cmap, device,
                    type_names=list(type_names))
        plt.tight_layout()
        plt.savefig(os.path.join(gphi_dir, name), dpi=87)
        plt.close(fig)
    finally:
        config.plotting.xlim = orig_xlim
        config.plotting.ylim = orig_ylim

    # 3) f_theta function: same voltage x-axis v ∈ [-3, 3], msg pinned to
    # 0 to probe the per-node update at zero recurrent input. TaskGNN's
    # f_theta input is (v, a, msg) — 1 + emb_dim + 1 — which doesn't match
    # the generic `_build_f_theta_features` (1 + emb_dim + 1 + 1, with
    # excitation), so we use a local feature builder.
    ftheta_dir = os.path.join(log_dir, 'tmp_training', 'function', 'f_theta')
    os.makedirs(ftheta_dir, exist_ok=True)
    n_pts = 1000
    rr_1d = torch.linspace(-3.0, 3.0, n_pts, device=device)
    rr = rr_1d.unsqueeze(0).expand(n_neurons, -1)
    feat_fn = lambda rr_f, emb_f: torch.cat(
        [rr_f, emb_f, torch.zeros_like(rr_f)], dim=1
    )
    func = _batched_mlp_eval(net.f_theta, net.a, rr, feat_fn, device)

    fig, ax = plt.subplots(figsize=(8, 8))
    type_np = nt_np.astype(int).ravel()
    x_np = rr_1d.detach().cpu().numpy()
    func_np = func.detach().cpu().numpy()
    # Per-type qualitative LUT keyed by type id (good for >32 types); avoids the
    # CustomColorMap.color(t)=t/nmap bunching that maps all high types to ~cyan.
    _type_cols = qualitative_colors(int(type_np.max()) + 1)
    for t in np.unique(type_np):
        mask = type_np == int(t)
        curves = func_np[mask]
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        color = _type_cols[int(t)] if int(t) < len(_type_cols) else cmap.color(int(t))
        label = (type_names[int(t)]
                 if int(t) < len(type_names) else f"type {int(t)}")
        ax.plot(x_np, mean, linewidth=1.5, color=color, label=label)
        if std.max() > 1e-6:
            ax.fill_between(x_np, mean - std, mean + std,
                             color=color, alpha=0.15)
    ax.axhline(0, color='#aaa', linewidth=0.5, linestyle='--')
    ax.set_xlim([-3.0, 3.0])
    ax.set_xlabel(r'$v_i$', fontsize=24)
    ax.set_ylabel(r'$f_\theta(\mathbf{a}_i, v_i)$', fontsize=24)
    if len(np.unique(type_np)) <= 12:
        ax.legend(fontsize=12, frameon=False, loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(ftheta_dir, name), dpi=87)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Type-pair masks (lifted from the teacher; used by both JaneliaCxRNN and
# TaskRNN to define the cosine-distance / norm-floor regulariser blocks)
# ---------------------------------------------------------------------------


def build_type_pair_blocks(
    neuron_types: np.ndarray,
    type_names: list,
    W_con: np.ndarray,
) -> dict:
    """(post-type → pre-type) bool-mask blocks for the cos-distance reg.

    Only blocks with at least one non-zero W_con entry are returned
    (matches the definition of set B in the cos-distance regulariser).
    """
    blocks: dict = {}
    nt = np.asarray(neuron_types).astype(np.int64)
    unique = sorted(set(nt.tolist()))
    for q in unique:
        post_mask = nt == q
        for p in unique:
            pre_mask = nt == p
            block = np.outer(post_mask, pre_mask)
            if block.sum() == 0:
                continue
            sub = W_con[block]
            if np.abs(sub).sum() < 1e-12:
                continue
            tp_name = f"{type_names[int(p)]}->{type_names[int(q)]}"
            blocks[tp_name] = torch.from_numpy(block.astype(np.bool_))
    return blocks


def precision_horizon_metrics(
    net, *, device, thr_deg: float = 15.0, rel_d: float = 0.20,
    floor_d: float = 0.5,
    leaky: bool = False, n_seed: int = 4, n_steps: int = 6000,
    dt: float | None = None,
) -> dict:
    """Discriminative integrator metrics on a long *naturalistic* rollout.

    Rollout Pearson saturates (~0.99) over a short trial, so we instead drive
    the model with an Ornstein--Uhlenbeck angular velocity ω(t) and forward
    speed v_fwd(t) for ``n_steps`` and watch the error grow:

      tau_theta_s  -- time the decoded heading stays within ``thr_deg`` of the
                      integrated truth (precision horizon, seconds).
      heading_gain_err -- |slope(decoded vs true unwrapped heading) - 1|, the
                      ω-independent driver of heading drift.
      tau_d_s      -- displacement precision horizon (seconds): time the
                      decoded distance / 2-D position stays within ``rel_d`` of
                      the true displacement (a *relative* band, since the
                      cumulative targets are unbounded; evaluated once the true
                      displacement clears ``floor_d``). Ground truth is the
                      leaky or cumulative integral per ``leaky``; None for
                      heading-only models.

    Values are averaged over ``n_seed`` OU realisations; a horizon that never
    crosses the threshold is right-censored at ``n_steps * dt`` seconds.
    """
    import math
    dt = float(net.dt) if dt is None else float(dt)
    T = int(n_steps)
    n_in = int(getattr(net, "n_input", 3))
    n_out = int(getattr(net, "n_output", 0))
    is_2d = (n_out == 4)
    leak_alpha = 1.0 - dt / 2.0          # xi/position_tau_s = 2.0 s

    def _ou(rng, tau, sd, mu, lo, hi):
        a = math.exp(-dt / tau)
        x = np.empty(T, np.float32); x[0] = mu
        for t in range(1, T):
            x[t] = mu + a * (x[t - 1] - mu) + math.sqrt(1 - a * a) * rng.normal(0, sd)
        return np.clip(x, lo, hi)

    def _leaky(v):
        o = np.zeros(T, np.float32)
        for t in range(1, T):
            o[t] = leak_alpha * o[t - 1] + v[t] * dt
        return o

    def _first(err, thr):
        ix = np.where(err[10:] > thr)[0]
        return (ix[0] + 10) * dt if len(ix) else T * dt

    def _first_rel(err, true):
        # first time |err| exceeds rel_d * |true displacement|, once the true
        # displacement has cleared floor_d (skips the near-zero start).
        bad = (np.abs(true) > floor_d) & (err > rel_d * np.abs(true))
        ix = np.where(bad[10:])[0]
        return (ix[0] + 10) * dt if len(ix) else T * dt

    was_training = net.training
    net.eval()
    th, gain, td = [], [], []
    try:
        for s in range(n_seed):
            rng = np.random.default_rng(s)
            om = _ou(rng, 1.0, 55.0, 0.0, -150, 150)
            vf = _ou(rng, 1.5, 0.4, 1.0, 0, 3)
            u = np.zeros((1, T, n_in), np.float32)
            u[0, :, 0] = om
            if n_in >= 4:
                u[0, :, 1] = vf; u[0, 0, 2] = 1.0
            elif n_in == 3:
                u[0, 0, 1] = 1.0
            with torch.no_grad():
                yp, _ = net(torch.from_numpy(u).to(device))
            yp = yp[0].cpu().numpy()
            th_true = np.cumsum(np.deg2rad(om)) * dt
            th_pred = np.arctan2(yp[:, 1], yp[:, 0])
            err = np.abs(np.degrees(np.angle(np.exp(1j * (th_pred - th_true)))))
            th.append(_first(err, thr_deg))
            gain.append(np.polyfit(th_true[10:], np.unwrap(th_pred)[10:], 1)[0])
            # displacement horizon (relative tolerance). Ground-truth is the
            # leaky or cumulative integral per the task; first time the error
            # exceeds rel_d of the true displacement.
            if is_2d:
                if leaky:
                    xt = _leaky(vf * np.cos(th_true))
                    yt = _leaky(vf * np.sin(th_true))
                else:
                    xt = np.cumsum(vf * np.cos(th_true)) * dt
                    yt = np.cumsum(vf * np.sin(th_true)) * dt
                perr = np.sqrt((yp[:, 2] - xt) ** 2 + (yp[:, 3] - yt) ** 2)
                td.append(_first_rel(perr, np.sqrt(xt ** 2 + yt ** 2)))
            elif n_out == 3:
                xi = _leaky(vf) if leaky else np.cumsum(vf) * dt
                td.append(_first_rel(np.abs(yp[:, 2] - xi), xi))
    finally:
        if was_training:
            net.train()
    return dict(
        tau_theta_s=float(np.mean(th)),
        heading_gain_err=float(np.mean(np.abs(np.array(gain) - 1.0))),
        tau_d_s=(float(np.mean(td)) if td else None),
        thr_deg=thr_deg, rel_d=rel_d, n_seed=n_seed, horizon_s=T * dt,
    )
