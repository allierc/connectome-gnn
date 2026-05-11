"""GCaMP indicator kernels for calcium-imaging-style observables.

Convolves a voltage trace with a fixed double-exponential impulse response
modeled after literature GCaMP kinetics. Replaces the leaky-integrator
calcium model when `simulation.calcium_type == "kernel"`.

Supported variants and their double-exponential parameters:

  "gcamp6f"      — cytosolic GCaMP6f (fly imaging convention):
                   tau_rise = 75 ms, tau_decay = 400 ms.
                   Chen et al. 2013 Nature, single-AP response in V1
                   pyramidal cells (Fig. 3, Supp. Table 3): half-rise ~45 ms,
                   decay tau ~400 ms. The 75 ms rise is the
                   double-exponential fit commonly adopted for fly imaging
                   models, e.g. Turner, Mann & Clandinin 2021 Curr Biol.
                   https://doi.org/10.1038/nature12354
                   https://doi.org/10.1016/j.cub.2021.03.004

  "gcamp6s"      — H2B-GCaMP6s as used in Ahrens-lab zebrafish whole-brain
                   imaging predating ZAPBench (e.g. Migault 2018):
                   tau_rise = 300 ms, tau_decay = 3.5 s.
                   Migault et al. 2018 Curr Biol fit Tg(elavl3:H2B-GCaMP6s)
                   in whole-brain light-sheet imaging of larval zebrafish.
                   Knafo et al. 2024 bioRxiv re-fits give tau_d = 1.78 s
                   for the faster H2B-GCaMP6f. The 300 ms rise corresponds
                   to cytosolic GCaMP6s half-rise ~180 ms (Chen 2013) plus
                   nuclear-envelope smoothing.
                   https://doi.org/10.1016/j.cub.2018.10.017
                   https://doi.org/10.1101/2024.03.22.586054

  "gcamp7f"      — H2B-GCaMP7f, the actual ZAPBench acquisition variant
                   (Tg(elavl3:H2B-GCaMP7f), Lange et al. 2025 arXiv
                   2503.02618 §Methods):
                   tau_rise = 150 ms, tau_decay = 1.2 s.
                   Cytosolic jGCaMP7f has half-rise ~27 ms / half-decay
                   ~280 ms (Dana et al. 2019 Nat Methods); the H2B fusion
                   slows both. Zhang et al. 2023 Nature (jGCaMP8 paper)
                   directly compares H2B-6f / H2B-7f / H2B-8f kinetics in
                   6-8 dpf larval zebrafish optic tectum (Supp Fig 11F);
                   H2B-7f sits between H2B-6f (~1.78 s decay) and H2B-6s
                   (~3.5 s decay). The 1.2 s value used here is a
                   defensible midpoint pending a finer fit from ZAPBench
                   data directly.
                   https://arxiv.org/abs/2503.02618
                   https://doi.org/10.1038/s41592-019-0435-6
                   https://doi.org/10.1038/s41586-023-05828-9
"""

from __future__ import annotations

import math

import torch


def gcamp6f_kernel(
    dt_seconds: float,
    tau_rise: float = 0.075,
    tau_decay: float = 0.4,
    length_seconds: float = 2.4,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a unit-sum GCaMP6f impulse response on a regular grid.

    Returns a 1-D tensor `K` of length `ceil(length_seconds / dt_seconds)` with
    `K[0]` corresponding to t=0 (most recent voltage sample). Ordering is
    "newest-first" so it can be dot-producted directly against a rolling
    history buffer whose column 0 holds the latest voltage.

    K(t) ∝ exp(-t/tau_d) - exp(-t/tau_r), normalized to unit discrete sum so
    the kernel acts as a unity-DC-gain low-pass filter. A constant voltage
    input V gives a steady-state calcium of V (same numerical scale as
    voltage). Peak amplitude depends on dt_seconds and the tau ratio.
    """
    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be > 0, got {dt_seconds}")
    if tau_rise <= 0 or tau_decay <= 0:
        raise ValueError("tau_rise and tau_decay must be > 0")
    if tau_rise >= tau_decay:
        raise ValueError(
            f"tau_rise ({tau_rise}) must be < tau_decay ({tau_decay}) "
            "for a well-formed GCaMP response"
        )

    n = max(2, int(math.ceil(length_seconds / dt_seconds)))
    t = torch.arange(n, device=device, dtype=dtype) * dt_seconds
    k = torch.exp(-t / tau_decay) - torch.exp(-t / tau_rise)
    k = k / k.sum()
    return k


_VARIANT_PRESETS: dict[str, dict[str, float]] = {
    "gcamp6f": {"tau_rise": 0.075, "tau_decay": 0.400, "length_seconds":  2.4},
    "gcamp7f": {"tau_rise": 0.150, "tau_decay": 1.200, "length_seconds":  7.2},
    "gcamp6s": {"tau_rise": 0.300, "tau_decay": 3.500, "length_seconds": 15.0},
}


def build_kernel_from_config(sim, device: torch.device | str = "cpu") -> torch.Tensor:
    """Construct the calcium kernel from a SimulationConfig.

    Reads `calcium_kernel_variant` and `calcium_kernel_dt_seconds`. The
    `calcium_kernel_tau_rise`, `calcium_kernel_tau_decay`, and
    `calcium_kernel_length_seconds` YAML fields act as overrides — if they
    equal the gcamp6f baseline defaults (0.075/0.4/2.4) they are treated as
    unset and the variant's preset is used; otherwise the YAML values win.
    """
    variant = getattr(sim, "calcium_kernel_variant", "gcamp6f")
    if variant not in _VARIANT_PRESETS:
        raise ValueError(
            f"Unknown calcium_kernel_variant '{variant}'. "
            f"Supported: {sorted(_VARIANT_PRESETS)}."
        )
    preset = _VARIANT_PRESETS[variant]
    baseline = _VARIANT_PRESETS["gcamp6f"]
    yaml_tau_r = float(sim.calcium_kernel_tau_rise)
    yaml_tau_d = float(sim.calcium_kernel_tau_decay)
    yaml_len   = float(sim.calcium_kernel_length_seconds)
    tau_rise       = preset["tau_rise"]       if yaml_tau_r == baseline["tau_rise"]       else yaml_tau_r
    tau_decay      = preset["tau_decay"]      if yaml_tau_d == baseline["tau_decay"]      else yaml_tau_d
    length_seconds = preset["length_seconds"] if yaml_len   == baseline["length_seconds"] else yaml_len
    return gcamp6f_kernel(
        dt_seconds=float(sim.calcium_kernel_dt_seconds),
        tau_rise=tau_rise,
        tau_decay=tau_decay,
        length_seconds=length_seconds,
        device=device,
    )


def select_reference_neurons(neuron_type: torch.Tensor) -> tuple[list[int], list[str]]:
    """Pick the canonical 12 fly cell types used in opto_compare_*.png.

    Returns (neuron_idx, labels). Order matches
    figures/fig_rollout_3col_noise_comparison.py: Am, T1, T5a, T4a, Tm9,
    Tm1, Mi9, Mi1, L3, L2, L1, R1.
    Missing types are skipped.
    """
    from connectome_gnn.metrics import INDEX_TO_NAME

    reference_types = [0, 31, 39, 35, 55, 43, 22, 12, 7, 6, 5, 23]
    nt = neuron_type.detach().cpu().numpy()
    neuron_idx: list[int] = []
    labels: list[str] = []
    for t_int in reference_types:
        ids = (nt == t_int).nonzero()[0]
        if len(ids) == 0:
            continue
        neuron_idx.append(int(ids[0]))
        labels.append(INDEX_TO_NAME.get(t_int, f"type_{t_int}"))
    return neuron_idx, labels


def plot_voltage_calcium_traces(
    voltage: "np.ndarray",
    calcium: "np.ndarray",
    labels: list[str],
    dt_seconds: float,
    save_path: str,
    title: str | None = None,
    start_frame: int = 0,
) -> None:
    """Two-panel V vs Ca trace plot in the opto_compare visual style.

    Left panel: voltage (green) with calcium overlaid in light red.
    Right panel: calcium only (light red).
    Mirrors connectome_gnn.generators.optogenetics._draw_opto_traces:
    per-row mean subtraction, shared vertical step, labels in the left
    margin, header text in a white bbox on the left panel, x-axis in ms.

    Args:
        voltage: (n_frames, n_neurons) selected-neuron voltage trace.
        calcium: (n_frames, n_neurons) corresponding calcium trace.
        labels: per-row cell-type label.
        dt_seconds: physical seconds per frame, for the x-axis.
        save_path: target PNG path.
    """
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    if voltage.shape != calcium.shape:
        raise ValueError(
            f"voltage {voltage.shape} and calcium {calcium.shape} shape mismatch"
        )
    n_frames, n_neurons = voltage.shape

    # Match opto_compare constants and global rc.
    rc_path = "/workspace/connectome-gnn/figures/janne.matplotlibrc"
    if os.path.isfile(rc_path):
        matplotlib.rc_file(rc_path)
    _TRACE_SHRINK = 0.65
    _FS_LABEL, _FS_TICK, _FS_TYPE = 8, 6, 6
    _LW_V, _LW_CA = 1.2, 0.9
    _COLOR_V = "#2ca02c"     # green, same as opto baseline
    _COLOR_CA = "#ff7a7a"    # light red, distinct from opto stim's #cf222e

    time_ms = (np.arange(n_frames) + start_frame) * dt_seconds * 1000.0

    # Shared vertical step derived from the combined V/Ca per-row std, so the
    # two panels use the same y-spacing (visually comparable).
    row_stds = [max(voltage[:, i].std(), calcium[:, i].std()) for i in range(n_neurons)]
    step_v = max(0.5 * _TRACE_SHRINK,
                 3.0 * _TRACE_SHRINK * (max(row_stds) if row_stds else 1.0))

    v_mean = voltage.mean(axis=0)
    ca_mean = calcium.mean(axis=0)
    s = _TRACE_SHRINK

    def _pretty_ticks(lo, hi, n_target=3):
        raw_step = (hi - lo) / max(1, n_target - 1)
        mag = 10 ** np.floor(np.log10(max(raw_step, 1e-12)))
        step = mag
        for m in (1, 2, 5, 10):
            if m * mag >= raw_step:
                step = m * mag
                break
        tick_lo = np.ceil(lo / step - 1e-9) * step
        return list(np.arange(tick_lo, hi + step / 2, step))

    def _draw(ax, draw_voltage: bool, draw_calcium: bool, header_text=None):
        for i in range(n_neurons):
            # Bottom row = first label; mirror opto's bottom-up stack.
            y_base = (n_neurons - 1 - i) * step_v
            if draw_voltage:
                ax.plot(time_ms, s * (voltage[:, i] - v_mean[i]) + y_base,
                        lw=_LW_V, color=_COLOR_V, alpha=0.95, zorder=2,
                        label="voltage" if i == 0 else None)
            if draw_calcium:
                ax.plot(time_ms, s * (calcium[:, i] - ca_mean[i]) + y_base,
                        lw=_LW_CA, color=_COLOR_CA, alpha=0.95, zorder=3,
                        label="calcium" if i == 0 else None)
            ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.02, y_base,
                    labels[i], fontsize=_FS_TYPE, va="center", ha="right",
                    color="black")

        if header_text:
            ax.text(0.015, 0.99, header_text, transform=ax.transAxes,
                    va="top", ha="left", fontsize=_FS_TICK,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=0.4))

        ax.set_ylabel("neurons", fontsize=_FS_LABEL, labelpad=32)
        ax.set_ylim([-step_v, (n_neurons - 1) * step_v + 2.2 * step_v])
        ax.set_yticks([])
        ax.set_xlim([time_ms[0], time_ms[-1]])
        ticks = _pretty_ticks(time_ms[0], time_ms[-1])
        if ticks:
            ax.set_xticks(ticks)
        ax.set_xlabel("time (ms)", fontsize=_FS_LABEL, labelpad=1)
        ax.tick_params(axis="x", labelsize=_FS_TICK, pad=1)
        ax.spines["left"].set_visible(False)
        ax.legend(loc="upper right", fontsize=_FS_TICK, frameon=False)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 6.0))
    _draw(axL, draw_voltage=True, draw_calcium=True, header_text=title)
    _draw(axR, draw_voltage=False, draw_calcium=True, header_text=None)
    axL.set_title("voltage + calcium", fontsize=_FS_LABEL)
    axR.set_title("calcium F = V * K", fontsize=_FS_LABEL)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def apply_kernel_step(
    v_hist: torch.Tensor,
    voltage: torch.Tensor,
    kernel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One streaming convolution step.

    Args:
        v_hist: (n_neurons, kernel_len) — column 0 holds previous-step voltage.
        voltage: (n_neurons,) — current-step voltage.
        kernel: (kernel_len,) — newest-first impulse response.

    Returns:
        v_hist_new: shifted buffer with `voltage` written into column 0.
        calcium: (n_neurons,) — current calcium = sum_k K[k] * V[t-k].
    """
    v_hist_new = torch.roll(v_hist, shifts=1, dims=-1)
    v_hist_new[:, 0] = voltage
    calcium = (v_hist_new * kernel).sum(dim=-1)
    return v_hist_new, calcium
