"""Path-integration evaluation + snapshot helpers for CX recurrent models.

These helpers are duck-typed against any module exposing:
    .dt          (float)            — Euler step
    .n_units     (int)              — recurrent unit count
    .W_rec       (Tensor (N, N))    — effective recurrent weight (read-only)
    forward(u)  -> (y_hat, h_buf)  — (B, T, 3) -> (B, T, 2), (B, T, N)

so they work on both `teachers.JaneliaCxRNN` and `models.TaskRNN`.

History: lifted out of `teachers/janelia_cx_teacher.py` to keep the new
`data_train_task_gnn` from importing the teacher module.  The teacher
re-exports these names for backwards compat.
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
        y_hat, _ = net(batch.u)
        warmup = 10
        y_hat_n = y_hat[:, warmup:, :] / (
            y_hat[:, warmup:, :].norm(dim=-1, keepdim=True) + 1e-8
        )
        y_n = batch.y[:, warmup:, :]
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
) -> float:
    """Same metric as `path_integration_accuracy`, but on a pre-built
    (u, y) test split (the trainer already has this in GPU memory).
    """
    net.eval()
    cosines = []
    with torch.no_grad():
        for i in range(0, u.shape[0], batch_size):
            yh, _ = net(u[i : i + batch_size])
            yh_n = yh[:, warmup:, :] / (
                yh[:, warmup:, :].norm(dim=-1, keepdim=True) + 1e-8
            )
            yt = y[i : i + batch_size, warmup:, :]
            cosines.append((yh_n * yt).sum(dim=-1).mean().item())
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
        _, h = net(batch.u)
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
    omega_deg_per_s: float,
    device: str,
) -> dict:
    """One trial with **constant ω**, no OU noise, no standing pauses.

    Designed to span the full HD circle by the end of the rollout so the
    kinograph shows the bump migrating across the full orientation axis.
    """
    T = int(n_steps)
    omega = np.full((1, T), float(omega_deg_per_s), dtype=np.float32)
    # Constant ω from t=0 — no trial-start zeroing. This breaks parity with
    # the OU training data (where ω[0]=0 by OU initial condition) for a
    # single frame, but produces a clean flat ω trace in the sweep plot. The
    # 10-frame warmup in the metric computation absorbs any one-frame offset
    # in the resulting theta_hd ramp.
    omega_rad = np.deg2rad(omega)
    theta_hd = np.cumsum(omega_rad, axis=1) * float(net.dt)

    u = np.zeros((1, T, 3), dtype=np.float32)
    u[:, :, 0] = omega
    u[:, 0, 1] = 1.0
    u[:, 0, 2] = 0.0

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
    return {
        "u": u[0],
        "y_pred": y_pred,
        "true_theta": theta_hd[0],
        "decoded_theta": np.arctan2(y_pred[:, 1], y_pred[:, 0]),
        "h": h[0].cpu().numpy(),
        "r": r,
        "n_steps": T,
        "omega_deg_per_s": float(omega_deg_per_s),
        "dt_s": float(net.dt),
    }


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
    Returns (nan, nan) on failure or degenerate input.
    """
    try:
        rollout = _deterministic_sweep_rollout(
            net, n_steps=n_steps,
            omega_deg_per_s=omega_deg_per_s, device=device,
        )
    except Exception:
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
    matrix_dir: str | None = None,    # backwards-compat; ignored
    config=None,
) -> None:
    """Render the combined kinograph+matrix snapshot.

    The matrix is the top-left panel of the kinograph figure, so we no
    longer write a separate matrix-only PNG.
    """
    from connectome_gnn.plot_cx import (
        cx_epg_directions,
        plot_cx_training_snapshot,
    )

    name = f"step_{global_step:07d}.png"

    try:
        rollout = _deterministic_sweep_rollout(
            net, n_steps=snapshot_n_steps,
            omega_deg_per_s=snapshot_omega_deg, device=device,
        )
        rollout["r_epg"] = rollout["r"][:, epg_indices]
        pen_type_idx = [i for i, n in enumerate(type_names)
                        if "PEN" in n and "PEG" not in n]
        if pen_type_idx:
            pen_idx_list: list[int] = []
            nt = np.asarray(neuron_types)
            for t in pen_type_idx:
                pen_idx_list.extend(np.where(nt == t)[0].tolist())
            pen_indices = np.array(sorted(pen_idx_list), dtype=np.int64)
            rollout["r_pen"] = rollout["r"][:, pen_indices]
        epg_theta = cx_epg_directions(epg_glom_ix)
        # Pass GT W_con if the model exposes it (TaskRNN, JaneliaCxRNN
        # both register the buffer); helpers that wrap with torch.compile
        # proxy buffer access through __getattr__.
        W_con_np = (net.W_con.detach().cpu().numpy()
                    if hasattr(net, "W_con") else None)
        pi_hist, _fw_hist, rmse_hist = load_pi_fwhm_history(
            os.path.join(log_dir, 'tmp_training', 'metrics.log'))
        plot_cx_training_snapshot(
            W_rec=net.W_rec.detach().cpu().numpy(),
            rollout=rollout,
            epg_theta=epg_theta,
            output_path=os.path.join(kinograph_dir, name),
            W_con=W_con_np,
            neuron_types=neuron_types,
            type_names=type_names,
            step=global_step,
            dt_s=float(net.dt),
            pi_acc_history=pi_hist,
            rmse_history=rmse_hist,
        )
    except Exception as exc:
        print(f"[cx_eval] kinograph snapshot failed @ step {global_step}: {exc}")

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
            print(f"[cx_eval] gnn function plots failed @ step {global_step}: {exc}")


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
    from connectome_gnn.utils import CustomColorMap

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
    for t in np.unique(type_np):
        mask = type_np == int(t)
        curves = func_np[mask]
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        color = cmap.color(int(t))
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
