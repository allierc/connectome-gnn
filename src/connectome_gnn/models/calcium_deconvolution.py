#!/usr/bin/env python
"""
calcium_deconvolution.py
========================
Wiener-style deconvolution of GCaMP6f calcium traces back to voltage, with
a per-neuron rollout-Pearson comparison against the ground-truth voltage
recorded by ``graph_data_generator``.

Forward model (matches ``connectome_gnn/generators/gcamp_kernel.py``):

    F[t] = sum_{k=0}^{L-1} K[k] * V[t-k]                  (causal conv)

The kernel ``K`` is built from the YAML config via
``build_kernel_from_config``, identical to what was used during data
generation, so the only error sources are (i) FFT circular wrap-around
on the leading frames and (ii) numerical regularisation.

Inverse: pad to N+L-1 and apply Wiener regularisation,

    V_hat = ifft( fft(F) * conj(fft(K)) / (|fft(K)|^2 + lambda) )

then crop the first ``L`` frames (kernel warm-up / wrap-around region).

Usage
-----
    python -m connectome_gnn.models.calcium_deconvolution config/fly/foo.yaml
    # or a direct path
    python src/connectome_gnn/models/calcium_deconvolution.py \\
        /groups/saalfeld/.../config/fly/flyvis_noise_free_blank50_heaviside_var_kernel_cv00.yaml

Outputs (under ``log_path(config.config_file, 'results')``):
    deconv_traces.png             — reference-cell V vs deconvolved-V vs Ca
    deconv_rollout.log            — per-window + global Pearson / RMSE
    deconv_rollout_pearson.npy    — per-neuron Pearson over the 1000-frame window
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import zarr

# Make ``src/`` importable when run as a script.
_REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'src')
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.gcamp_kernel import (
    build_kernel_from_config,
    plot_voltage_calcium_traces,
    select_reference_neurons,
)
from connectome_gnn.utils import (
    add_pre_folder,
    compute_trace_metrics,
    config_path,
    fisher_pool,
    graphs_data_path,
    log_path,
    set_data_root,
    validate_pre_folder,
)


N_ROLLOUT_FRAMES = 1000
# Plot/comparison window in physical time. At dt=0.02s the default 10-30s
# range corresponds to frames [500, 1500], i.e. it skips the early
# transient and lines up with where the visual stimulus has settled.
PLOT_START_MS = 10_000.0
PLOT_END_MS   = 30_000.0
# Fallback when sim.calcium_noise_level == 0 and calcium_noise.zarr is
# absent — synthesise i.i.d. noise at this fraction of per-neuron std,
# so the deconv stress test still has something to chew on.
FALLBACK_NOISE_FRACTION = 0.20
NOISE_SEED = 42


def _resolve_config(yaml_path: str) -> Tuple[NeuralGraphConfig, str]:
    """Replicate ``GNN_Main.py``'s config-loading semantics for a single YAML.

    Accepts either a config_name ('flyvis_noise_free_kernel') or an absolute
    .yaml path. Returns the loaded config plus the ``pre_folder`` (e.g.
    'fly/') so ``config.dataset`` and ``config.config_file`` are normalised.
    """
    if os.path.isabs(yaml_path) or os.path.isfile(yaml_path):
        yaml_file = yaml_path if yaml_path.endswith('.yaml') else yaml_path + '.yaml'
        parent = os.path.basename(os.path.dirname(os.path.abspath(yaml_file)))
        pre_folder = parent + '/' if parent else ''
        validate_pre_folder(pre_folder)
        config = NeuralGraphConfig.from_yaml(yaml_file)
        if not config.dataset.startswith(pre_folder):
            config.dataset = pre_folder + config.dataset
        if config.config_file == 'none':
            stem = os.path.splitext(os.path.basename(yaml_file))[0]
            config.config_file = pre_folder + stem
        return config, pre_folder

    config_file, pre_folder = add_pre_folder(yaml_path)
    yaml_file = config_path(f'{config_file}.yaml')
    config = NeuralGraphConfig.from_yaml(yaml_file)
    if not config.dataset.startswith(pre_folder):
        config.dataset = pre_folder + config.dataset
    if config.config_file == 'none':
        config.config_file = config_file
    return config, pre_folder


def _load_zarr(path: str, n_frames: int | None) -> np.ndarray:
    z = zarr.open_array(path, mode='r')
    if n_frames is None or n_frames >= z.shape[0]:
        return np.asarray(z[:], dtype=np.float32)
    return np.asarray(z[:n_frames], dtype=np.float32)


def wiener_deconvolve(
    calcium: np.ndarray,
    kernel: np.ndarray,
    lam: float = 1e-3,
    regularizer: str = 'derivative',
    noise_sigma: float = 0.0,
) -> np.ndarray:
    """FFT-based deconvolution. Three modes:

        'flat'         classical Tikhonov, R = I (penalises ||V||²).
        'derivative'   Tikhonov, R = first-diff (penalises ||DV||² → smooth V).
        'optimal'      Wiener-Helstrom (MMSE-optimal under stationary,
                       jointly-Gaussian (V, noise)). Requires ``noise_sigma``.
                       No λ knob — uses the per-frequency signal PSD
                       estimated from the data and the known noise PSD.

    For the Tikhonov modes,

        V_hat(f) = K*(f) F(f) / (|K(f)|² + λ · max|K|² · |R(f)|²)

    For 'optimal',

        V_hat(f) = K*(f) S_V(f) F(f) / (|K(f)|² S_V(f) + N(f))

    where N(f) = σ² · n_fft (white-noise PSD on the rfft grid) and
    S_V(f) = max(0, S_F(f) - N(f)) / max(|K(f)|², ε) with
    S_F(f) = mean_n |F_n(f)|². Pooling across neurons gives ~N samples
    per frequency bin so the estimate is stable.
    """
    T, N = calcium.shape
    L = kernel.shape[0]
    n_fft = T + L - 1

    f_pad = np.zeros((n_fft, N), dtype=np.float32)
    f_pad[:T] = calcium
    k_pad = np.zeros(n_fft, dtype=np.float32)
    k_pad[:L] = kernel  # newest-first kernel == causal (K[0] is at t=0)

    F_hat = np.fft.rfft(f_pad, axis=0)
    K_hat = np.fft.rfft(k_pad)
    power_k = np.abs(K_hat) ** 2

    if regularizer == 'optimal':
        if noise_sigma <= 0:
            raise ValueError(
                "regularizer='optimal' requires noise_sigma > 0 "
                "(pass sim.calcium_noise_level)"
            )
        # White-noise PSD on the rfft grid: E[|N(f)|²] = σ²·n_fft for
        # i.i.d. Gaussian noise of variance σ². Parseval: sum_t n[t]² ≈ σ²·T,
        # and the rfft's E[|X(f)|²] integrates to that.
        n_psd = (noise_sigma ** 2) * n_fft
        # Per-frequency observed PSD averaged across neurons.
        s_f = np.mean(np.abs(F_hat) ** 2, axis=1)
        # Debias by the noise floor, clip negative bins. Result is an
        # estimate of E[|F_noiseless(f)|²].
        s_f_clean = np.maximum(s_f - n_psd, 0.0)
        # Solve S_F_clean = |K|² · S_V, with a small floor so quiet
        # frequencies don't blow up the division.
        eps_k = 1e-12 * float(power_k.max())
        s_v = s_f_clean / (power_k + eps_k)
        # Optimal Wiener-Helstrom filter (per-frequency, broadcast over neurons).
        denom = power_k * s_v + n_psd
        # Guard zero (rare; only if a frequency has no signal AND no noise).
        denom = np.maximum(denom, 1e-30)
        H = np.conj(K_hat) * s_v / denom
        V_hat = F_hat * H[:, None]
    else:
        if regularizer == 'flat':
            reg = np.ones_like(power_k)
        elif regularizer == 'derivative':
            # First-difference operator d[t] = δ[t] - δ[t-1]; D_hat = 1 - e^{-jω}.
            d_pad = np.zeros(n_fft, dtype=np.float32)
            d_pad[0] = 1.0
            d_pad[1] = -1.0
            D_hat = np.fft.rfft(d_pad)
            reg = np.abs(D_hat) ** 2
        else:
            raise ValueError(f'unknown regularizer {regularizer!r}')

        eps = lam * float(power_k.max())
        V_hat = F_hat * np.conj(K_hat)[:, None] / (power_k[:, None] + eps * reg[:, None])

    v_pad = np.fft.irfft(V_hat, n=n_fft, axis=0)
    return v_pad[:T].astype(np.float32)


def _pearson_log_line(pearson: np.ndarray) -> str:
    """Match graph_tester._pearson_log_line so downstream parsers still work."""
    fz = fisher_pool(pearson)
    return (
        f"Pearson r (Fisher-z mean): {fz['r_mean']:.4f} "
        f"(sd_sym={fz['r_sd_sym']:.4f}, "
        f"[{fz['r_lo']:.4f}, {fz['r_hi']:.4f}], "
        f"n={fz['n']})\n"
    )


def data_deconvolve(
    config: NeuralGraphConfig,
    device: str | None = None,  # unused — FFT deconv runs on CPU via numpy
) -> dict:
    """GNN_Main-style entry point.

    Loads (voltage.zarr, calcium.zarr, calcium_noise.zarr) from the train
    split of ``graphs_data/<config.dataset>``, deconvolves the noise-mixed
    calcium with the YAML-configured GCaMP kernel, and writes a results
    bundle under ``log_path(config.config_file, 'results')`` mirroring how
    ``data_test`` lays out its rollout artifacts (PNG figure +
    ``deconv_rollout.log`` + per-neuron .npy arrays).
    """
    del device  # CPU-only FFT path
    return deconvolve_and_compare(config)


def deconvolve_and_compare(config: NeuralGraphConfig) -> dict:
    sim = config.simulation
    dataset_dir = graphs_data_path(config.dataset)
    x_train_dir = os.path.join(dataset_dir, 'x_list_train')

    voltage_path = os.path.join(x_train_dir, 'voltage.zarr')
    calcium_path = os.path.join(x_train_dir, 'calcium.zarr')
    neuron_type_path = os.path.join(x_train_dir, 'neuron_type.zarr')
    calcium_noise_path = os.path.join(x_train_dir, 'calcium_noise.zarr')

    for p in (voltage_path, calcium_path, neuron_type_path):
        if not os.path.isdir(p):
            raise FileNotFoundError(f'required zarr missing: {p}')

    kernel_t = build_kernel_from_config(sim, device='cpu')
    kernel = kernel_t.detach().cpu().numpy().astype(np.float32)
    L = kernel.shape[0]
    dt_s = float(sim.calcium_kernel_dt_seconds)

    # Comparison + plot window in frames, derived from PLOT_*_MS.
    start = int(round(PLOT_START_MS * 1e-3 / dt_s))
    end = int(round(PLOT_END_MS * 1e-3 / dt_s))
    if start < L:
        raise ValueError(
            f'PLOT_START_MS ({PLOT_START_MS}) is inside the kernel warm-up '
            f'region (first {L * dt_s * 1000:.0f} ms).'
        )

    # Load enough frames so the FFT wrap-around lives outside the window:
    # L extra on each side.
    n_load = end + L
    voltage_gt = _load_zarr(voltage_path, n_load)
    calcium = _load_zarr(calcium_path, n_load)
    neuron_type = torch.tensor(
        np.asarray(zarr.open_array(neuron_type_path, mode='r')[:], dtype=np.int64)
    )

    print(f'  voltage shape:  {voltage_gt.shape}')
    print(f'  calcium shape:  {calcium.shape}')
    print(f'  kernel length:  {L} samples '
          f'({L * dt_s:.3f} s)')
    print(f'  window:         frames [{start}, {end}] '
          f'= [{start * dt_s * 1000:.0f}, {end * dt_s * 1000:.0f}] ms')

    # Add calcium measurement noise before deconvolution. Three sources, in
    # priority order:
    #   1. dataset's calcium_noise.zarr (written by the generator when
    #      sim.calcium_noise_level > 0) — physically the right thing.
    #   2. synthesise σ = sim.calcium_noise_level on the fly (i.i.d.).
    #   3. fallback i.i.d. σ = FALLBACK_NOISE_FRACTION × per-neuron std
    #      (used when both 1 and 2 are absent so the figure isn't trivial).
    cfg_noise = float(getattr(sim, 'calcium_noise_level', 0.0))
    if os.path.isdir(calcium_noise_path):
        noise = _load_zarr(calcium_noise_path, n_load)
        noise_descr = f'calcium_noise.zarr (σ from sim.calcium_noise_level={cfg_noise})'
    elif cfg_noise > 0:
        rng = np.random.default_rng(NOISE_SEED)
        noise = (rng.standard_normal(calcium.shape).astype(np.float32) * cfg_noise)
        noise_descr = f'synth σ = sim.calcium_noise_level = {cfg_noise}'
    else:
        rng = np.random.default_rng(NOISE_SEED)
        ca_std = calcium.std(axis=0, keepdims=True).clip(min=1e-6)
        noise = (rng.standard_normal(calcium.shape).astype(np.float32)
                 * FALLBACK_NOISE_FRACTION * ca_std)
        noise_descr = (
            f'fallback σ = {FALLBACK_NOISE_FRACTION:.3f} × std(calcium) '
            '(sim.calcium_noise_level is 0)'
        )
    calcium_noisy = (calcium + noise).astype(np.float32)
    print(f'  added noise:    {noise_descr}')

    # Adaptive λ for derivative-Tikhonov. From a per-σ sweep on this
    # dataset family (sigma ∈ {0.01, 0.02, 0.03}, voltage RMS ≈ 0.94),
    # the optimum follows λ ≈ 100·σ² closely:
    #   σ=0.01 → λ_opt=0.010 (formula: 0.010)
    #   σ=0.02 → λ_opt=0.030 (formula: 0.040)
    #   σ=0.03 → λ_opt=0.100 (formula: 0.090)
    # Below noise_sigma < 1e-4 fall back to the original noise-free default.
    if cfg_noise > 1e-4:
        sigma_used = cfg_noise
        lam_used = 100.0 * sigma_used ** 2
        deconv_method = (
            f'derivative-Tikhonov, λ = 100·σ² = {lam_used:.4g}  '
            f'(σ = {sigma_used})'
        )
    else:
        # Synthetic-noise fallback for datasets without a real
        # calcium_noise.zarr: estimate σ from the noise we just added.
        sigma_used = float(noise.std())
        if sigma_used > 1e-6:
            lam_used = 100.0 * sigma_used ** 2
            deconv_method = (
                f'derivative-Tikhonov, λ = 100·σ_hat² = {lam_used:.4g}  '
                f'(σ_hat = {sigma_used:.5f}, synth)'
            )
        else:
            lam_used = 3e-3
            deconv_method = 'derivative-Tikhonov λ=3e-3 (no noise model)'
    voltage_hat = wiener_deconvolve(
        calcium_noisy, kernel, lam=lam_used, regularizer='derivative',
    )
    print(f'  deconvolution:  {deconv_method}')

    gt_win = voltage_gt[start:end]        # (n_frames, N)
    pred_win = voltage_hat[start:end]
    ca_win = calcium_noisy[start:end]

    # Transpose to (N, n_frames) so compute_trace_metrics treats each row as a
    # per-neuron trace (matches graph_tester.data_test_gnn convention).
    rmse, pearson, _feve, _r2 = compute_trace_metrics(
        gt_win.T, pred_win.T, label='deconvolve-rollout'
    )
    fz = fisher_pool(pearson)

    # ------------------------------------------------------------------ output
    results_dir = log_path(config.config_file, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Reference-cell traces, in the same visual style as the generator.
    neuron_idx, labels = select_reference_neurons(neuron_type)
    if not neuron_idx:
        print('  no reference neuron types found — falling back to first 12 ids')
        neuron_idx = list(range(min(12, gt_win.shape[1])))
        labels = [f'n{i}' for i in neuron_idx]

    fig_path = os.path.join(results_dir, 'deconv_traces.png')
    _plot_three_panel(
        voltage_gt=gt_win[:, neuron_idx],
        voltage_hat=pred_win[:, neuron_idx],
        calcium=ca_win[:, neuron_idx],
        labels=labels,
        dt_seconds=dt_s,
        save_path=fig_path,
        start_frame=start,
        noise_title=noise_descr,
        title=(
            f'{config.config_file} — Wiener deconvolution\n'
            f'pearson(V_gt, V_hat) = {fz["r_mean"]:.3f} '
            f'[{fz["r_lo"]:.3f}, {fz["r_hi"]:.3f}]'
        ),
    )
    print(f'  saved traces:   {fig_path}')

    # Per-window rollout metrics CSV, 100-frame windows — same shape as
    # graph_tester's results_rollout_by_step.csv but in calcium frames.
    rollout_log = os.path.join(results_dir, 'deconv_rollout.log')
    np.save(os.path.join(results_dir, 'deconv_rollout_pearson.npy'),
            pearson.astype(np.float32))
    np.save(os.path.join(results_dir, 'deconv_rollout_rmse.npy'),
            rmse.astype(np.float32))

    n_frames_win = gt_win.shape[0]
    with open(rollout_log, 'w') as f:
        f.write('Wiener-deconvolution rollout metrics\n')
        f.write('=' * 60 + '\n')
        f.write(f'config:          {config.config_file}\n')
        f.write(f'dataset:         {config.dataset}\n')
        f.write(f'frames compared: {n_frames_win} '
                f'(window {start * dt_s * 1000:.0f}-{end * dt_s * 1000:.0f} ms)\n')
        f.write(f'neurons:         {gt_win.shape[1]}\n')
        f.write(f'kernel:          {sim.calcium_kernel_variant} '
                f'(tau_r={sim.calcium_kernel_tau_rise}, '
                f'tau_d={sim.calcium_kernel_tau_decay}, '
                f'dt={sim.calcium_kernel_dt_seconds})\n')
        f.write(f'calcium noise:   {noise_descr}  (seed={NOISE_SEED})\n')
        f.write(f'deconvolution:   {deconv_method}\n\n')
        f.write(_pearson_log_line(pearson))
        f.write(f'RMSE: {float(np.nanmean(rmse)):.4f} '
                f'+/- {float(np.nanstd(rmse)):.4f}\n')
        f.write(f'Pearson r (Fisher-z mean, sd): '
                f'{fz["z_mean"]:.4f} {fz["z_sd"]:.4f}\n\n')
        f.write('window,frame_start,frame_end,RMSE,pearson\n')
        win = 100
        for k in range(0, n_frames_win, win):
            t_lo, t_hi = k, min(k + win, n_frames_win)
            w_true = gt_win[t_lo:t_hi]
            w_pred = pred_win[t_lo:t_hi]
            rmse_w = float(np.sqrt(np.mean((w_true - w_pred) ** 2)))
            pear_w = []
            for i in range(w_true.shape[1]):
                if (np.std(w_true[:, i]) > 1e-8
                        and np.std(w_pred[:, i]) > 1e-8):
                    pear_w.append(
                        np.corrcoef(w_true[:, i], w_pred[:, i])[0, 1]
                    )
            pear_w = float(np.nanmean(pear_w)) if pear_w else float('nan')
            f.write(f'{k // win},{t_lo},{t_hi},{rmse_w:.4f},{pear_w:.4f}\n')
    print(f'  saved rollout:  {rollout_log}')

    return dict(
        rollout_pearson=float(fz['r_mean']),
        rollout_pearson_sd=float(fz['r_sd_sym']),
        rollout_rmse=float(np.nanmean(rmse)),
        n_neurons=int(gt_win.shape[1]),
        results_dir=results_dir,
    )


def _plot_three_panel(
    voltage_gt: np.ndarray,
    voltage_hat: np.ndarray,
    calcium: np.ndarray,
    labels: list,
    dt_seconds: float,
    save_path: str,
    start_frame: int = 0,
    noise_title: str = '',
    title: str | None = None,
) -> None:
    """V_gt | V_hat | calcium traces — mirrors plot_voltage_calcium_traces."""
    import matplotlib
    import matplotlib.pyplot as plt

    rc_path = '/workspace/connectome-gnn/figures/janne.matplotlibrc'
    if os.path.isfile(rc_path):
        matplotlib.rc_file(rc_path)

    if not (voltage_gt.shape == voltage_hat.shape == calcium.shape):
        raise ValueError(
            f'shape mismatch: gt={voltage_gt.shape} hat={voltage_hat.shape} '
            f'ca={calcium.shape}'
        )

    n_frames, n_neurons = voltage_gt.shape
    _TRACE_SHRINK = 0.65
    _FS_LABEL, _FS_TICK, _FS_TYPE = 8, 6, 6
    _LW_V = 1.2          # green V_gt
    _LW_VHAT = 0.6       # thin black V_deconv (overlaid on top of green)
    _C_V = '#2ca02c'      # green — ground truth voltage
    _C_VHAT = 'black'     # black — Wiener-deconvolved voltage
    _C_CA = '#d62728'     # red — calcium F = V * K

    time_ms = (np.arange(n_frames) + start_frame) * dt_seconds * 1000.0

    row_stds = [
        max(voltage_gt[:, i].std(),
            voltage_hat[:, i].std(),
            calcium[:, i].std())
        for i in range(n_neurons)
    ]
    step_v = max(0.5 * _TRACE_SHRINK,
                 3.0 * _TRACE_SHRINK * (max(row_stds) if row_stds else 1.0))
    v_mean = voltage_gt.mean(axis=0)
    vh_mean = voltage_hat.mean(axis=0)
    ca_mean = calcium.mean(axis=0)
    s = _TRACE_SHRINK

    def _draw(ax, panel: str, header=None):
        for i in range(n_neurons):
            y_base = (n_neurons - 1 - i) * step_v
            if panel == 'calcium':
                ax.plot(time_ms, s * (calcium[:, i] - ca_mean[i]) + y_base,
                        lw=0.9, color=_C_CA, alpha=0.95, zorder=3,
                        label='calcium' if i == 0 else None)
            elif panel == 'overlay':
                ax.plot(time_ms, s * (voltage_gt[:, i] - v_mean[i]) + y_base,
                        lw=_LW_V, color=_C_V, alpha=0.95, zorder=2,
                        label='V (gt)' if i == 0 else None)
                ax.plot(time_ms, s * (voltage_hat[:, i] - vh_mean[i]) + y_base,
                        lw=_LW_VHAT, color=_C_VHAT, alpha=0.9, zorder=3,
                        label='V (deconv)' if i == 0 else None)
            ax.text(
                time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.02, y_base,
                labels[i], fontsize=_FS_TYPE, va='center', ha='right',
                color='black',
            )
        if header:
            ax.text(0.015, 0.99, header, transform=ax.transAxes,
                    va='top', ha='left', fontsize=_FS_TICK,
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.85, pad=0.4))
        ax.set_ylabel('neurons', fontsize=_FS_LABEL, labelpad=32)
        ax.set_ylim([-step_v, (n_neurons - 1) * step_v + 2.2 * step_v])
        ax.set_yticks([])
        ax.set_xlim([time_ms[0], time_ms[-1]])
        ax.set_xlabel('time (ms)', fontsize=_FS_LABEL, labelpad=1)
        ax.tick_params(axis='x', labelsize=_FS_TICK, pad=1)
        ax.spines['left'].set_visible(False)
        ax.legend(loc='upper right', fontsize=_FS_TICK, frameon=False)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 6.0))
    _draw(axL, 'calcium', header=title)
    _draw(axR, 'overlay', header=None)
    ca_title = 'input calcium F = V * K'
    if noise_title:
        ca_title += f'  +  noise: {noise_title}'
    axL.set_title(ca_title, fontsize=_FS_LABEL)
    axR.set_title('voltage gt (green) vs deconv (black)',
                  fontsize=_FS_LABEL)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Wiener-deconvolve calcium and compare to ground-truth voltage.'
    )
    parser.add_argument(
        'config',
        help='Config name (e.g. flyvis_noise_free_kernel) OR absolute path '
             'to a .yaml file.',
    )
    parser.add_argument(
        '--output_root',
        default=None,
        help='Override the data root (defaults to GNN_OUTPUT_ROOT env or cwd).',
    )
    args = parser.parse_args()

    output_root = args.output_root or os.environ.get('GNN_OUTPUT_ROOT')
    if output_root:
        if not os.path.isdir(output_root):
            raise SystemExit(f'--output_root does not exist: {output_root}')
        set_data_root(output_root)

    config, _ = _resolve_config(args.config)
    print(f'config:  {config.config_file}')
    print(f'dataset: {config.dataset}')
    print(f'data:    {graphs_data_path(config.dataset)}')
    print(f'log:     {log_path(config.config_file)}')

    metrics = deconvolve_and_compare(config)
    print()
    print('=== rollout (1000-frame) ===')
    print(f'  pearson (fisher-z): {metrics["rollout_pearson"]:.4f} '
          f'(+/- {metrics["rollout_pearson_sd"]:.4f})')
    print(f'  rmse:               {metrics["rollout_rmse"]:.4f}')
    print(f'  n_neurons:          {metrics["n_neurons"]}')
    print(f'  results dir:        {metrics["results_dir"]}')


if __name__ == '__main__':
    main()
