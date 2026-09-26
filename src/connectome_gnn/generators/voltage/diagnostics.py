"""Diagnostics of a generated dataset: conductance bracket, effective ranks, SNR, generation_log.txt.

Each is computed as a value from the train split read back from disk
(TrainTrace), logged, and -- for the generation log -- rendered to text by a
pure function before it is written.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from connectome_gnn.log import get_logger
from connectome_gnn.plot import plot_activity_traces
from connectome_gnn.utils import to_numpy
from connectome_gnn.zarr_io import load_raw_array, load_simulation_data

logger = get_logger(__name__)

# SVD analysis (4-panel plot) -- DISABLED 2026-09-07 at the user's request.
# analyze_data_svd computed a full singular-value decomposition of the
# (n_frames x n_neurons) activity and of the visual stimulus and wrote
# <dataset>/svd_analysis.png; the existing svd_analysis.png files were
# deleted at the same time. The effective ranks that the kinograph and
# generation_log.txt actually use are still computed in compute_ranks, so
# nothing downstream loses a number. Leaving svd_results empty makes the two
# `svd_results.get(...)` writes of the generation log no-ops.
SVD_RESULTS: dict = {}


@dataclass(frozen=True)
class TrainTrace:
    """x_list_train read back after generation (and after tiling)."""

    x_ts: Any                 # NeuronTimeSeries: voltage, stimulus, noise, types
    activity: Any             # x_ts.voltage.numpy(), (T, N)


def load_train_split(store) -> TrainTrace:
    """Read x_list_train back; y_list_train is read too, and not used.

    THE y_list_train READ IS LEGACY'S: its result is never used, but a
    missing or unreadable store still raises here, as it always did.
    """
    x_ts = load_simulation_data(store.path("x_list_train"))
    load_raw_array(store.path("y_list_train"))
    activity_full = x_ts.voltage.numpy()  # (n_frames, n_neurons) — needed for noise plotting
    return TrainTrace(x_ts=x_ts, activity=activity_full)


def check_bracket(spec, ode_params, trace: TrainTrace, store) -> dict | None:
    """Conductance ground truth only: did any voltage cross its reversal bracket?

    Every edge must keep the SIGN of its own driving force (E_ij - v_i) for the
    whole run: excitatory edges need v_i < E_exc, inhibitory ones v_i > E_inh,
    i.e. E_inh < v_i < E_exc. It is NOT a positivity test -- the driving force
    is negative on inhibitory edges by design, and that is what makes them
    inhibit.

    A crossing does not break the generator: (E - v) simply changes sign, which
    is what a reversal potential physically means. It matters for two narrower
    reasons, and they are why this reports rather than merely asserts:
      1. the student was distilled on the teacher's voltage band and its
         reversals are unconstrained by data outside it, so past a crossing the
         generator is extrapolating a fit;
      2. a downstream GNN with g_phi_positive squares g_phi and so forces ONE
         sign per edge -- it cannot fit data containing crossings, which would
         silently handicap the models this dataset exists to train.

    Returns the bracket numbers (None unless ground_truth_model is
    "conductance"). Crossings write BRACKET_CROSSINGS.txt when
    conductance_bracket_strict is False, else BRACKET_VIOLATION.txt and raise
    ValueError (the zarr data stays on disk; generation_log.txt is not written).
    """
    if spec.network.ground_truth_model != "conductance":
        return None
    x_ts = trace.x_ts
    _n_exc, _n_inh, _worst = ode_params.bracket_violation(x_ts.voltage)
    _bracket = dict(n_exc=_n_exc, n_inh=_n_inh, worst_margin=_worst,
                    E_exc=float(ode_params.E_exc.min()),
                    E_inh=float(ode_params.E_inh.max()),
                    v_min=float(x_ts.voltage.min()), v_max=float(x_ts.voltage.max()))
    _tot = _n_exc + _n_inh
    _msg = (f"conductance bracket: {_n_exc} above E_exc, {_n_inh} below E_inh, "
            f"worst margin {_worst:+.4f} "
            f"(v {_bracket['v_min']:+.3f}..{_bracket['v_max']:+.3f} vs "
            f"[{_bracket['E_inh']:+.3f}, {_bracket['E_exc']:+.3f}])")
    _strict = spec.network.conductance_bracket_strict
    if _tot and not _strict:
        # A CROSSING IS EXPECTED, NOT A FAILURE, once the reversals are
        # physiological. A real chloride equilibrium potential sits INSIDE the
        # operating range -- inhibition is hyperpolarising above it and
        # depolarising-but-shunting below it -- so a twin built with a
        # physiological E_Cl crosses by construction and a dataset that never
        # crossed would be the suspicious one. Recorded as a diagnostic, with the
        # dataset still validated.
        logger.warning(_msg + "  [conductance_bracket_strict False: recorded, "
                              "not fatal]")
        with open(store.path("BRACKET_CROSSINGS.txt"), "w") as _bf:
            _bf.write(_msg + "\n")
            for _k, _v in _bracket.items():
                _bf.write(f"{_k}: {_v}\n")
    elif _tot:
        logger.error(_msg)
        # The zarr writers have already flushed by this point, so x_list_*
        # and y_list_* ARE on disk. What is withheld is generation_log.txt
        # and .generate_done, without which _have_data refuses the dataset.
        # Record the numbers next to the half-written data so the failure is
        # inspectable later rather than only in whatever captured stderr.
        with open(store.path("BRACKET_VIOLATION.txt"), "w") as _bf:
            _bf.write(_msg + "\n")
            for _k, _v in _bracket.items():
                _bf.write(f"{_k}: {_v}\n")
        raise ValueError(
            _msg + " -- the generated voltages leave the bracket, so some "
            "edges reverse their driving force mid-run. Below E_inh this is "
            "self-amplifying: the inhibitory driving force turns positive, so "
            "inhibitory synapses start exciting. The partial x_list_*/y_list_* "
            "are left on disk but no generation_log.txt is written, so the "
            "dataset will not validate. Lower noise_model_level, widen the "
            "reversals (student_delta_inh/exc), or use a student fitted "
            "over a wider voltage range.")
    logger.info(_msg)
    return _bracket


@dataclass(frozen=True)
class Ranks:
    """Effective ranks (components for 90 % / 99 % of the variance) of the train split."""

    rank_90_act: int
    rank_99_act: int
    rank_90_mc: int           # mean-centred activity; 0 when the centred activity is zero
    rank_99_mc: int
    rank_90_inp: int          # photoreceptor input
    rank_99_inp: int


def _svd_lowrank(matrix_np, n_components, dev):
    """Randomized SVD via torch.svd_lowrank (GPU when available). Draws torch RNG."""
    t = torch.as_tensor(matrix_np, dtype=torch.float32, device=dev)
    _, S, _ = torch.svd_lowrank(t, q=n_components + 10, niter=4)
    return S[:n_components].cpu().numpy()


def compute_ranks(spec, trace: TrainTrace) -> Ranks:
    """Effective ranks of the activity, the mean-centred activity and the input (torch.svd_lowrank)."""
    device = spec.device
    x_ts, activity_full = trace.x_ts, trace.activity
    logger.info("computing effective rank ...")

    _svd_device = torch.device(device) if (device and device != 'cpu' and torch.cuda.is_available()) else torch.device('cpu')
    logger.info(f"  SVD device: {_svd_device}")

    n_comp = min(50, min(activity_full.shape) - 1)
    S_act = _svd_lowrank(activity_full, n_comp, _svd_device)
    cumvar_act = np.cumsum(S_act**2) / np.sum(S_act**2)
    rank_90_act = int(np.searchsorted(cumvar_act, 0.90) + 1)
    rank_99_act = int(np.searchsorted(cumvar_act, 0.99) + 1)

    # Mean-centered rank: subtract per-neuron temporal mean to remove static bias pattern.
    activity_centered = activity_full - activity_full.mean(axis=0, keepdims=True)
    centered_var = np.sum(activity_centered**2)
    if centered_var > 1e-12:
        S_mc = _svd_lowrank(activity_centered, n_comp, _svd_device)
        cumvar_mc = np.cumsum(S_mc**2) / centered_var
        rank_90_mc = int(np.searchsorted(cumvar_mc, 0.90) + 1)
        rank_99_mc = int(np.searchsorted(cumvar_mc, 0.99) + 1)
    else:
        rank_90_mc = rank_99_mc = 0

    input_for_svd = x_ts.stimulus[:, : spec.network.n_input_neurons].numpy()
    n_comp_input = min(50, min(input_for_svd.shape) - 1)
    S_inp = _svd_lowrank(input_for_svd, n_comp_input, _svd_device)
    cumvar_inp = np.cumsum(S_inp**2) / np.sum(S_inp**2)
    rank_90_inp = int(np.searchsorted(cumvar_inp, 0.90) + 1)
    rank_99_inp = int(np.searchsorted(cumvar_inp, 0.99) + 1)

    logger.info(
        f"activity rank(90%)={rank_90_act}  rank(99%)={rank_99_act}  centered rank(90%)={rank_90_mc}  rank(99%)={rank_99_mc}"
    )
    logger.info(f"visual input rank(90%)={rank_90_inp}  rank(99%)={rank_99_inp}")
    return Ranks(rank_90_act=rank_90_act, rank_99_act=rank_99_act, rank_90_mc=rank_90_mc, rank_99_mc=rank_99_mc,
                 rank_90_inp=rank_90_inp, rank_99_inp=rank_99_inp)


def kinograph_labels(spec, ode_params, trace: TrainTrace) -> tuple:
    """(activity labels, stimulus labels) for the kinograph: (type name, first, last + 1) per type.

    Needs ``ode_params.neuron_types``, which neither ODE-parameter class this
    generator builds has, so both are always None (dead code kept for the
    follow-up that removes it).
    """
    # Build neuron-type labels for kinograph annotations
    act_labels = None
    stim_labels = None
    if hasattr(ode_params, "neuron_types") and ode_params.neuron_types is not None:  # golden-uncovered: B46
        nt = to_numpy(ode_params.neuron_types)
        tnames = getattr(ode_params, "type_names", None)
        if tnames is not None:
            act_labels = []
            for ti, name in enumerate(tnames):
                idx = np.where(nt == ti)[0]
                if len(idx) > 0:
                    act_labels.append((name, int(idx.min()), int(idx.max()) + 1))
            # Stimulus labels: find which neurons receive non-zero stimulus
            stim_np = trace.x_ts.stimulus[:, : spec.network.n_input_neurons].numpy()
            stim_power = np.sum(stim_np**2, axis=0)  # (N,)
            stim_labels = []
            for ti, name in enumerate(tnames):
                idx = np.where(nt == ti)[0]
                active_idx = idx[stim_power[idx] > 1e-6] if idx.max() < len(stim_power) else np.array([])
                if len(active_idx) > 0:
                    stim_labels.append((name, int(active_idx.min()), int(active_idx.max()) + 1))
            if not stim_labels:
                stim_labels = None

    if act_labels:  # golden-partial: B46
        logger.info(f"kinograph act_labels: {act_labels}")
    if stim_labels:  # golden-partial: B46
        logger.info(f"kinograph stim_labels: {stim_labels}")
    return act_labels, stim_labels


def log_trace_window(spec, trace: TrainTrace) -> None:
    """Log the warm-up skip and window of the trace plots (visualize only; no plot uses them any more).

    Legacy also sliced the stimulus the same way and never used the slice;
    that no-op is gone.
    """
    activity_full = trace.activity
    # Skip warmup frames (100ms / dt) and show 400ms window for all plots
    warmup_ms = 100.0
    window_ms = 800.0
    warmup_frames = int(warmup_ms / spec.delta_t)
    window_frames = int(window_ms / spec.delta_t)
    activity_plot = activity_full[warmup_frames:] if activity_full.shape[0] > warmup_frames + 10 else activity_full
    logger.info(
        f"plotting traces (warmup_skip={warmup_frames} frames={warmup_ms}ms, window={window_frames} frames={window_ms}ms, {activity_plot.shape[0]} frames available)"
    )


def measurement_snr(spec, trace: TrainTrace, store, node_types_int, fig_style) -> dict | None:
    """Voltage and derivative SNR of the train split under its measurement noise (None without it).

    With visualize, also activity_traces_noisy.png (one trace per type: the
    type list is always passed, so plot_activity_traces never samples neurons
    and draws no RNG).
    """
    x_ts, activity_full = trace.x_ts, trace.activity
    # Plot noisy activity traces using the same neurons + compute SNR
    logger.debug("plot noisy activity traces ...")
    noise_data = x_ts.noise.numpy() if x_ts.noise is not None else None
    if noise_data is None:  # golden-uncovered: B49 (noise.zarr is always written)
        return None
    noisy_activity = activity_full + noise_data  # (T, N)
    if spec.output.visualize:
        plot_activity_traces(
            activity=noisy_activity.T,
            output_path=store.path("activity_traces_noisy.png"),
            n_traces=100,
            max_frames=10000,
            n_input_neurons=spec.network.n_input_neurons,
            style=fig_style,
            type_list=node_types_int,
            dpi=300,
            title="noisy voltage traces (measurement noise)",
        )

    # --- SNR analysis (per neuron) ---
    # Voltage SNR: std(clean_voltage) / std(measurement_noise) per neuron
    signal_std = np.std(activity_full, axis=0)  # (N,)
    noise_std = np.std(noise_data, axis=0)  # (N,)
    voltage_snr = np.where(noise_std > 0, signal_std / noise_std, np.inf)
    voltage_snr_finite = voltage_snr[np.isfinite(voltage_snr)]

    # Derivative SNR: std(clean_derivative) / std(derivative_noise) per neuron
    # derivative noise = (noise[t+1] - noise[t]) / dt
    deriv_noise = np.diff(noise_data, axis=0) / spec.delta_t  # (T-1, N)
    deriv_noise_std = np.std(deriv_noise, axis=0)  # (N,)
    y_clean = load_raw_array(store.path("y_list_train"))  # (T, N, 1)
    deriv_signal_std = np.std(y_clean[:, :, 0], axis=0)  # (N,)
    deriv_snr = np.where(deriv_noise_std > 0, deriv_signal_std / deriv_noise_std, np.inf)
    deriv_snr_finite = deriv_snr[np.isfinite(deriv_snr)]

    deriv_noise_std_theoretical = spec.train_noise.measurement_std * np.sqrt(2) / spec.delta_t
    deriv_noise_std_empirical = np.mean(deriv_noise_std)

    snr_stats = {
        "voltage_snr_mean": np.mean(voltage_snr_finite),
        "voltage_snr_median": np.median(voltage_snr_finite),
        "voltage_snr_min": np.min(voltage_snr_finite),
        "voltage_snr_max": np.max(voltage_snr_finite),
        "derivative_snr_mean": np.mean(deriv_snr_finite),
        "derivative_snr_median": np.median(deriv_snr_finite),
        "derivative_snr_min": np.min(deriv_snr_finite),
        "derivative_snr_max": np.max(deriv_snr_finite),
        "derivative_noise_std_theoretical": deriv_noise_std_theoretical,
        "derivative_noise_std_empirical": deriv_noise_std_empirical,
    }

    logger.info("--- Measurement noise SNR analysis ---")
    logger.info("  voltage SNR (std_signal / std_noise) per neuron:")
    logger.info(
        f"    mean: {snr_stats['voltage_snr_mean']:.2f}  "
        f"median: {snr_stats['voltage_snr_median']:.2f}  "
        f"min: {snr_stats['voltage_snr_min']:.2f}  "
        f"max: {snr_stats['voltage_snr_max']:.2f}"
    )
    logger.info("  derivative SNR (std_dy/dt / std_noise_dy/dt) per neuron:")
    logger.info(
        f"    mean: {snr_stats['derivative_snr_mean']:.2f}  "
        f"median: {snr_stats['derivative_snr_median']:.2f}  "
        f"min: {snr_stats['derivative_snr_min']:.2f}  "
        f"max: {snr_stats['derivative_snr_max']:.2f}"
    )
    logger.info(f"  derivative noise std (theoretical): {snr_stats['derivative_noise_std_theoretical']:.2f}")
    logger.info(f"  derivative noise std (empirical mean): {snr_stats['derivative_noise_std_empirical']:.2f}")
    logger.info("--------------------------------------")
    return snr_stats


@dataclass(frozen=True)
class GenerationSummary:
    """Every value generation_log.txt reports."""

    n_neurons: int
    n_frames_train: int       # after tiling
    n_frames_test: int
    n_sequences_train: int
    n_sequences_test: int
    split: Any                # VideoSplit
    bracket: dict | None
    ranks: Ranks | None       # None without compute_ranks
    snr: dict | None


def render_generation_log(spec, s: GenerationSummary, svd_results: dict = SVD_RESULTS) -> str:
    """The text of generation_log.txt (pure)."""
    lines = [
        f'dataset: {spec.output.dataset}\n',
        f'n_neurons: {s.n_neurons}\n',
        f'n_input_neurons: {spec.network.n_input_neurons}\n',
        f'n_frames_train: {s.n_frames_train}\n',
        f'n_frames_test: {s.n_frames_test}\n',
        f'n_sequences_train: {s.n_sequences_train}\n',
        f'n_sequences_test: {s.n_sequences_test}\n',
        f'n_train_videos: {s.split.n_train_vids}\n',
        f'n_test_videos: {s.split.n_test_vids}\n',
        f'train_videos: {s.split.train_video_names}\n',
        f'test_videos: {s.split.test_video_names}\n',
        f'visual_input_type: {spec.stimulus.visual_input_type}\n',
    ]
    if spec.stimulus.datavis_roots:
        lines.append(f'datavis_roots: {list(spec.stimulus.datavis_roots)}\n')
    lines += [
        f'noise_model_level: {spec.train_noise.process_std}\n',
        f'measurement_noise_level: {spec.train_noise.measurement_std}\n',
        f'ground_truth_model: {spec.network.ground_truth_model}\n',
    ]
    b = s.bracket
    if b is not None:
        lines += [
            f'conductance_checkpoint: {spec.network.conductance_checkpoint}\n',
            f'E_inh: {b["E_inh"]:.6f}\n',
            f'E_exc: {b["E_exc"]:.6f}\n',
            f'voltage_range: {b["v_min"]:.6f} .. {b["v_max"]:.6f}\n',
            f'bracket_crossings_above_E_exc: {b["n_exc"]}\n',
            f'bracket_crossings_below_E_inh: {b["n_inh"]}\n',
            f'bracket_worst_margin: {b["worst_margin"]:.6f}\n',
        ]
    lines += [
        f'model_id: {spec.network.model_id}\n',
        f'ensemble_id: {spec.network.ensemble_id}\n',
        '\n',
    ]
    if s.ranks is not None:
        lines += [
            f"activity_rank_90: {s.ranks.rank_90_act}\n",
            f"activity_rank_99: {s.ranks.rank_99_act}\n",
            f"input_rank_90: {s.ranks.rank_90_inp}\n",
            f"input_rank_99: {s.ranks.rank_99_inp}\n",
        ]
    if svd_results.get("activity"):  # golden-uncovered: B51 (svd_results is always {})
        lines += [f"svd_activity_rank_90: {svd_results['activity']['rank_90']}\n",
                  f"svd_activity_rank_99: {svd_results['activity']['rank_99']}\n"]
    if svd_results.get("visual_stimuli"):  # golden-uncovered: B51
        lines += [f"svd_visual_rank_90: {svd_results['visual_stimuli']['rank_90']}\n",
                  f"svd_visual_rank_99: {svd_results['visual_stimuli']['rank_99']}\n"]
    if s.snr is not None:
        lines.append("\n")
        for key, val in s.snr.items():
            lines.append(f"{key}: {val:.2f}\n")
    return "".join(lines)


def write_generation_log(spec, store, summary: GenerationSummary) -> str:
    """Write generation_log.txt and, for a conductance dataset, the reversal report beside it."""
    gen_log_path = store.path('generation_log.txt')
    text = render_generation_log(spec, summary)
    with open(gen_log_path, 'w') as log_f:
        log_f.write(text)
    # The reversals this dataset was generated with, beside the data: one row
    # per postsynaptic type, and the sorted E_inh figure against the voltage
    # band. Only conductance datasets carry them; the function returns None on
    # a current dataset.
    if summary.bracket is not None:
        try:
            from connectome_gnn.plot import report_dataset_reversals
            report_dataset_reversals(os.path.dirname(gen_log_path))
        except Exception as _exc:
            print(f"\033[93mreversal report skipped: {type(_exc).__name__}: {_exc}\033[0m")
    logger.info(f"generation log saved to {gen_log_path}")
    return gen_log_path
