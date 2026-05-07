"""Optogenetic perturbation pipeline.

This module implements the experimental arm of the structural-null-space
result (models/structural_nullspace_table.py): naturalistic visual drive
leaves ~71% of the connectome's edge-weight space in ker(H), structured as
sum-zero redistributions within (postsynaptic neuron, presynaptic cell-type)
columnar groups. Per-column-distinct optogenetic perturbation is the only
intervention that breaks the kernel.

Public API:
    make_optogenetics_stimulus  — (T, N) float32 tensor, zero off-target
    build_target_mask           — (N,) bool mask of opto targets
    build_waveform              — (T, n_targets) float32 waveform
    add_optogenetics_stimulus   — re-simulation pass adding opto current
    compare_traces              — quantification of with-vs-without opto
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from connectome_gnn.config import (
    OptogeneticsConfig,
    OptoTargetMode,
    OptoTargetSpec,
    OptoWaveform,
    OptoWaveformKind,
)
from connectome_gnn.metrics import (
    NAME_TO_INDEX,
    IDENTIFIABLE_TYPES,
    NO_OUTGOING_TYPES,
    fingerprint_dataset,
    load_nullspace_ranking,
    name_to_neuron_ids,
    neuron_column_ids,
    neuron_type_names,
    summarize_targets,
)
from connectome_gnn.neuron_state import NeuronState


def _resolve_topk_targets(
    spec: OptoTargetSpec,
) -> list[str]:
    """Pick the top-k cell-type names from the structural nullspace JSON."""
    ranking = load_nullspace_ranking(
        json_path=spec.structural_table_json,
        metric=spec.ranking.value if hasattr(spec.ranking, 'value') else str(spec.ranking),
    )
    if spec.k <= 0:
        raise ValueError(f"OptoTargetSpec.k must be >=1 for topk_nullspace, got {spec.k}")
    return [name for (name, _score, _lam) in ranking[: spec.k]]


def build_target_mask(
    state: NeuronState, spec: OptoTargetSpec
) -> torch.Tensor:
    """(N,) bool mask of which neurons receive opto current.

    Resolves each OptoTargetSpec.mode to a per-neuron boolean over state.
    """
    nt = state.neuron_type
    if nt is None:
        raise ValueError("build_target_mask requires state.neuron_type to be loaded")
    N = state.n_neurons

    mode = spec.mode if isinstance(spec.mode, str) else spec.mode.value

    if mode == OptoTargetMode.CELL_TYPE.value:
        if not spec.cell_types:
            raise ValueError("OptoTargetSpec(mode='cell_type') requires non-empty cell_types")
        for name in spec.cell_types:
            if name in NO_OUTGOING_TYPES:
                warnings.warn(
                    f"opto target {name!r} is in NO_OUTGOING_TYPES — perturbing it has no "
                    f"downstream effect. Consider a degenerate type from "
                    f"load_nullspace_ranking(metric='null_dim').",
                    UserWarning,
                )
        return name_to_neuron_ids(nt, list(spec.cell_types))

    if mode == OptoTargetMode.COLUMN.value:
        if state.pos is None:
            raise ValueError("OptoTargetSpec(mode='column') requires state.pos to be loaded")
        col_ids = neuron_column_ids(state.pos)
        wanted = torch.tensor(spec.columns, device=col_ids.device, dtype=col_ids.dtype)
        return torch.isin(col_ids, wanted)

    if mode == OptoTargetMode.EXPLICIT_INDICES.value:
        if not spec.indices:
            raise ValueError("OptoTargetSpec(mode='explicit_indices') requires non-empty indices")
        if spec.dataset_fingerprint is not None:
            actual = fingerprint_dataset(state)
            if actual != spec.dataset_fingerprint:
                raise ValueError(
                    f"dataset fingerprint mismatch: spec wants {spec.dataset_fingerprint}, "
                    f"current state hashes to {actual}. Indices are unsafe to apply across "
                    f"different connectome variants (e.g. extent=8 vs extent=15)."
                )
        mask = torch.zeros(N, dtype=torch.bool, device=nt.device)
        idx = torch.tensor(spec.indices, device=nt.device, dtype=torch.long)
        mask[idx] = True
        return mask

    if mode == OptoTargetMode.TOPK_NULLSPACE.value:
        names = _resolve_topk_targets(spec)
        return name_to_neuron_ids(nt, names)

    raise ValueError(f"unsupported OptoTargetSpec.mode: {mode!r}")


def _per_target_amplitudes(
    target_indices: torch.Tensor,
    target_type_per_neuron: list[str],
    waveform: OptoWaveform,
    nullspace_json_path: str,
) -> torch.Tensor:
    """Resolve per-target base amplitude.

    If waveform.amplitude is set, every target gets that scalar.
    If waveform.amplitude is None, look up per-type lambda_max from the
    nullspace JSON and use 0.5 * lambda_max(type_of_target). Falls back to 1.0
    when lambda_max is missing or NaN (older JSONs predate the instrumentation).
    """
    n_targets = int(target_indices.numel())
    if waveform.amplitude is not None:
        return torch.full((n_targets,), float(waveform.amplitude))

    try:
        ranking = load_nullspace_ranking(
            json_path=nullspace_json_path, metric='null_dim'
        )
    except FileNotFoundError:
        return torch.full((n_targets,), 1.0)
    lam_by_name: dict[str, float] = {
        name: lam for (name, _s, lam) in ranking
        if not math.isnan(lam)
    }
    out = torch.empty(n_targets, dtype=torch.float32)
    for i, idx in enumerate(target_indices.tolist()):
        type_name = target_type_per_neuron[idx]
        lam = lam_by_name.get(type_name, math.nan)
        out[i] = 0.5 * lam if not math.isnan(lam) else 1.0
    return out


def build_waveform(
    n_frames: int,
    target_indices: torch.Tensor,
    waveform: OptoWaveform,
    target_type_per_neuron: list[str],
    column_distinct: bool,
    nullspace_json_path: str = "scripts/structural_nullspace_table.json",
) -> torch.Tensor:
    """(T, n_targets) float32 — base waveform plus universal additive noise.

    column_distinct=True → independent realizations per target.
    column_distinct=False → identical waveform replicated across targets
        (preserves the columnar sum-zero kernel; emits warning at caller).
    """
    n_targets = int(target_indices.numel())
    amps = _per_target_amplitudes(target_indices, target_type_per_neuron, waveform,
                                  nullspace_json_path)  # (n_targets,)

    g = torch.Generator(device='cpu')
    g.manual_seed(int(waveform.seed))

    onset = max(0, int(waveform.onset_frame))
    offset = n_frames if waveform.offset_frame < 0 else int(waveform.offset_frame)
    offset = min(offset, n_frames)

    kind = waveform.kind if isinstance(waveform.kind, str) else waveform.kind.value

    # ---- base waveform (T, n_targets) ----
    base = torch.zeros(n_frames, n_targets, dtype=torch.float32)
    if onset < offset:
        if kind == OptoWaveformKind.WHITE_NOISE.value:
            # white_noise has no deterministic component; noise_level drives signal
            pass
        elif kind in (OptoWaveformKind.HEAVISIDE.value, OptoWaveformKind.CONSTANT.value):
            if column_distinct:
                base[onset:offset, :] = amps[None, :]
            else:
                # identical signal across targets — use amps mean to avoid hidden per-target variation
                base[onset:offset, :] = amps.mean()
        elif kind == OptoWaveformKind.IMPULSE.value:
            pw = max(1, int(waveform.pulse_width_frames))
            pp = max(pw + 1, int(waveform.pulse_period_frames))
            for k in range(0, (offset - onset) // pp + 1):
                a = onset + k * pp
                b = min(a + pw, offset)
                if a >= offset:
                    break
                base[a:b, :] = amps[None, :] if column_distinct else amps.mean()
        elif kind == OptoWaveformKind.VIDEO.value:
            if not waveform.video_path:
                raise ValueError("waveform.kind='video' requires video_path")
            arr = np.load(waveform.video_path)
            arr = torch.from_numpy(arr).float()
            if arr.shape != (n_frames, n_targets):
                raise ValueError(
                    f"video file shape {tuple(arr.shape)} != (n_frames, n_targets)="
                    f"({n_frames}, {n_targets})"
                )
            base[onset:offset, :] = arr[onset:offset, :]
        else:
            raise ValueError(f"unknown waveform kind: {kind}")

    # ---- universal additive Gaussian noise ----
    if waveform.noise_level > 0.0:
        if column_distinct:
            xi = torch.randn(n_frames, n_targets, generator=g, dtype=torch.float32)
        else:
            shared = torch.randn(n_frames, 1, generator=g, dtype=torch.float32)
            xi = shared.expand(n_frames, n_targets).clone()
        # zero noise outside the active window
        zero_mask = torch.ones(n_frames, 1, dtype=torch.float32)
        zero_mask[:onset, 0] = 0.0
        zero_mask[offset:, 0] = 0.0
        base = base + (xi * float(waveform.noise_level)) * zero_mask

    return base


def make_optogenetics_stimulus(
    state: NeuronState,
    n_frames: int,
    opto_cfg: OptogeneticsConfig,
    nullspace_json_path: str = "scripts/structural_nullspace_table.json",
) -> torch.Tensor:
    """Build the full (T, N) opto current tensor.

    Zero outside the targeted neuron mask. Returns float32 on the same device
    as state.neuron_type.
    """
    if not opto_cfg.enabled:
        return torch.zeros(n_frames, state.n_neurons, dtype=torch.float32)

    spec = opto_cfg.target
    if not spec.column_distinct:
        warnings.warn(
            "OptoTargetSpec.column_distinct=False: optogenetic drive is uniform across "
            "columns within each targeted cell type. This PRESERVES the columnar "
            "sum-zero kernel (see scripts/structural_nullspace_table.py) — the "
            "kernel-recovery experiment will be invalidated. Set column_distinct=True.",
            UserWarning,
        )

    mask = build_target_mask(state, spec)
    target_indices = torch.nonzero(mask, as_tuple=False).flatten().cpu()

    if target_indices.numel() == 0:
        warnings.warn("OptoTargetSpec resolves to ZERO targets — opto tensor will be all zeros.",
                      UserWarning)
        return torch.zeros(n_frames, state.n_neurons, dtype=torch.float32)

    type_names_per_neuron = neuron_type_names(state.neuron_type)
    wf = build_waveform(
        n_frames=n_frames,
        target_indices=target_indices,
        waveform=opto_cfg.waveform,
        target_type_per_neuron=type_names_per_neuron,
        column_distinct=spec.column_distinct,
        nullspace_json_path=nullspace_json_path,
    )

    out = torch.zeros(n_frames, state.n_neurons, dtype=torch.float32)
    out[:, target_indices] = wf
    return out


def _resolve_device(config) -> torch.device:
    dev = getattr(getattr(config, 'training', None), 'device', None)
    if dev is None or dev == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(dev)


def _build_flyvis_ode(state_dict: dict, neuron_types: torch.Tensor,
                     model_type: str, device: torch.device):
    """Reconstruct (FlyVisODE, edge_index) from a saved ode_params.pt.

    Mirrors models/structural_nullspace_table.py so the opto pipeline integrates
    with bit-for-bit the same forward as the source's data_generate_voltage.
    """
    from connectome_gnn.generators.flyvis_ode import FlyVisODE
    from connectome_gnn.generators.ode_params import FlyVisODEParams

    s = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
         for k, v in state_dict.items()}
    params = FlyVisODEParams(**s).to(device)
    n_types = int(neuron_types.max().item()) + 1
    ode = FlyVisODE(
        ode_params=params,
        g_phi=torch.nn.functional.relu,
        params=[],
        model_type=model_type,
        n_neuron_types=n_types,
        device=device,
    )
    return ode, params.edge_index.to(device)


def add_optogenetics_stimulus(config) -> None:
    """Re-simulate a baseline dataset with optogenetic current added.

    Reads the source's ODE state + per-frame visual stimulus, runs a fresh
    forward pass that sums the configured optogenetic current into I_ext
    (Phase 7), and writes a fully self-contained dataset under config.dataset.

    Output layout per split ∈ {train, test}:
        graphs_data/<config.dataset>/x_list_{split}/
            voltage.zarr stimulus.zarr noise.zarr
            optogenetics_stimulus.zarr
            pos.zarr group_type.zarr neuron_type.zarr

    Noise handling
    --------------
    Uses config.simulation.seed (matching the source) and mirrors the per-frame
    RNG order from data_generate_voltage._run_ode_generation:
      1. compute dv from ODE
      2. sample measurement noise (state.noise)
      3. write state at time t
      4. sample dynamics noise and advance voltage
    Because the source's data_generate_voltage interleaves additional RNG
    consumers (stimulus rendering, network init), the noise stream produced
    here is NOT bit-identical to the source's — but with a matched seed the
    expected (with - without)-opto delta is unbiased. For a strictly
    deterministic comparison set noise_model_level=measurement_noise_level=0
    on both runs.
    """
    import logging
    import os
    import zarr
    from connectome_gnn.neuron_state import NeuronTimeSeries, NeuronState
    from connectome_gnn.utils import graphs_data_path
    from connectome_gnn.zarr_io import ZarrSimulationWriterV3

    log = logging.getLogger(__name__)

    sim = config.simulation
    opto_cfg = sim.optogenetics
    if not opto_cfg.enabled:
        raise ValueError("OptogeneticsConfig.enabled must be True")
    if not opto_cfg.source_dataset:
        raise ValueError("OptogeneticsConfig.source_dataset must be set")

    # Resolve source/twin paths.
    source_root = graphs_data_path(opto_cfg.source_dataset)
    if not os.path.isdir(source_root):
        alt = graphs_data_path("fly", opto_cfg.source_dataset)
        if os.path.isdir(alt):
            source_root = alt
        else:
            raise FileNotFoundError(
                f"source dataset not found: tried {source_root!r} and {alt!r}"
            )
    target_root = graphs_data_path(config.dataset)
    os.makedirs(target_root, exist_ok=True)
    log.info(f"opto: source={source_root}  target={target_root}")

    # Load the source's ground-truth ODE state.
    ode_params_path = os.path.join(source_root, "ode_params.pt")
    if not os.path.isfile(ode_params_path):
        raise FileNotFoundError(
            f"source dataset is missing ode_params.pt at {ode_params_path}"
        )
    state_dict = torch.load(ode_params_path, map_location='cpu', weights_only=True)

    device = _resolve_device(config)
    model_type = config.graph_model.signal_model_name

    # Source neuron-type assignment.
    nt_path = os.path.join(source_root, "x_list_train", "neuron_type.zarr")
    neuron_types = torch.from_numpy(
        np.array(zarr.open_array(nt_path, mode='r'), dtype=np.int64)
    ).long().to(device)

    ode, edge_index = _build_flyvis_ode(state_dict, neuron_types, model_type, device)

    n_neurons = int(neuron_types.numel())
    if sim.n_neurons not in (0, n_neurons):
        log.warning(
            f"config.simulation.n_neurons ({sim.n_neurons}) "
            f"!= source n_neurons ({n_neurons}); using source value"
        )
    dt = float(sim.delta_t)
    log.info(
        f"opto: n_neurons={n_neurons} dt={dt} model={model_type} "
        f"noise_model={sim.noise_model_level} noise_meas={sim.measurement_noise_level}"
    )

    # Match data_generate_voltage's seed handling (line ~819).
    torch.manual_seed(sim.seed)
    np.random.seed(sim.seed)

    ar1_rho = float(getattr(sim, 'noise_ar1_rho', 0.0) or 0.0)

    for split in ('train', 'test'):
        source_split = os.path.join(source_root, f"x_list_{split}")
        target_split = os.path.join(target_root, f"x_list_{split}")
        if not os.path.isdir(source_split):
            log.warning(f"source split missing: {source_split} — skipping")
            continue

        # Match data_generate_voltage's split-conditional noise levels
        # (graph_data_generator.py L1400-1401).
        if split == 'train':
            split_noise_model = float(sim.noise_model_level)
            split_noise_meas = float(sim.measurement_noise_level)
        else:  # test
            split_noise_model = (
                float(sim.noise_model_level) if sim.noisy_test_data else 0.0
            )
            split_noise_meas = (
                float(sim.measurement_noise_level) if sim.noisy_test_data else 0.0
            )

        ts = NeuronTimeSeries.from_zarr_v3(
            source_split,
            fields=['stimulus', 'voltage', 'noise', 'pos', 'group_type', 'neuron_type'],
        )
        T = ts.n_frames
        log.info(f"opto [{split}]: T={T} frames")

        # Initial state — clone source's frame 0; opto is not applied at t=0,
        # so the first written frame matches the source for clean comparison.
        state = NeuronState(
            index=torch.arange(n_neurons, dtype=torch.long, device=device),
            pos=ts.pos.to(device),
            group_type=ts.group_type.to(device),
            neuron_type=ts.neuron_type.to(device),
            voltage=ts.voltage[0].to(device).clone(),
            stimulus=torch.zeros(n_neurons, dtype=torch.float32, device=device),
            optogenetics_stimulus=torch.zeros(n_neurons, dtype=torch.float32, device=device),
            noise=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        )

        # Per-split opto tensor (T, N).
        opto_full = make_optogenetics_stimulus(
            state, n_frames=T, opto_cfg=opto_cfg,
            nullspace_json_path=opto_cfg.target.structural_table_json,
        ).to(device)

        # Log opto coverage. Print a green block so the user sees exactly which
        # neurons are perturbed and how many per cell type.
        mask = build_target_mask(state, opto_cfg.target)
        coverage = summarize_targets(state, mask)
        log.info(
            f"opto [{split}]: "
            f"{int(mask.sum().item())} target neurons across {len(coverage)} types: "
            f"{coverage}"
        )
        G, R = "\033[92m", "\033[0m"
        print(f"{G}[opto {split}] reusing source x_list_{split} "
              f"({T} frames, {n_neurons} neurons){R}")
        print(f"{G}[opto {split}] perturbing {int(mask.sum().item())} neurons "
              f"across {len(coverage)} cell types:{R}")
        for name, (n_target, n_total, frac) in sorted(coverage.items()):
            print(f"{G}    {name:>10s}: {n_target:>4d} / {n_total:<4d} "
                  f"neurons ({100*frac:5.1f}%){R}")

        writer = ZarrSimulationWriterV3(
            path=target_split,
            n_neurons=n_neurons,
            time_chunks=2000,
            extra_dynamic_fields=['optogenetics_stimulus'],
        )

        # AR(1) state for measurement noise (zero at start, matches source).
        ar1_prev_noise = torch.zeros(n_neurons, dtype=torch.float32, device=device)
        ar1_inject_std = (
            split_noise_meas * (1.0 - ar1_rho ** 2) ** 0.5 if ar1_rho > 0 else 0.0
        )

        # Forward integration. Per-frame RNG order MATCHES
        # graph_data_generator._run_ode_generation L2208-2253 so that with a
        # matched seed the noise streams are statistically equivalent.
        with torch.no_grad():
            for t in range(T):
                state.stimulus = ts.stimulus[t].to(device, non_blocking=True)
                state.optogenetics_stimulus = opto_full[t]

                # 1. ODE step (Phase 7 forward sums opto into I_ext).
                dv = ode(state, edge_index, has_field=False).squeeze(-1)

                # 2. Measurement noise (state.noise).
                if split_noise_meas > 0:
                    if ar1_rho > 0:
                        ar1_prev_noise = (
                            ar1_rho * ar1_prev_noise
                            + torch.randn(n_neurons, dtype=torch.float32, device=device)
                            * ar1_inject_std
                        )
                        state.noise = ar1_prev_noise.clone()
                    else:
                        state.noise = (
                            torch.randn(n_neurons, dtype=torch.float32, device=device)
                            * split_noise_meas
                        )
                else:
                    state.noise = torch.zeros(
                        n_neurons, dtype=torch.float32, device=device
                    )

                # 3. Snapshot CURRENT frame.
                writer.append_state(state)

                # 4. Advance voltage with dynamics noise.
                if split_noise_model > 0:
                    state.voltage = (
                        state.voltage
                        + dt * dv
                        + torch.randn(n_neurons, dtype=torch.float32, device=device)
                        * split_noise_model
                    )
                else:
                    state.voltage = state.voltage + dt * dv

        n_written = writer.finalize()
        log.info(f"opto [{split}]: wrote {n_written} frames to {target_split}")

    # Auto-comparison: produce per-split with-vs-without trace plots.
    for split in ('train', 'test'):
        if not os.path.isdir(os.path.join(target_root, f"x_list_{split}")):
            continue
        try:
            metrics = compare_traces(
                source_dataset=opto_cfg.source_dataset,
                opto_dataset=config.dataset,
                split=split,
            )
            log.info(f"opto [{split}]: comparison figure -> {metrics['fig_path']}")
        except Exception as e:
            log.warning(f"opto [{split}]: compare_traces failed: {e}")


# Style constants — match fig_rollout_4col_flywire_comparison.py
_COLOR_GT, _COLOR_OPTO, _COLOR_STIM = '#2ca02c', 'black', '#cf222e'
_LW_GT, _LW_OPTO, _LW_STIM = 1.2, 0.45, 0.6
_TRACE_SHRINK = 0.65
_FS_LABEL, _FS_TICK, _FS_TYPE = 8, 6, 6


def _draw_opto_traces(ax, baseline, opto, opto_current, labels, time_ms, step_v,
                     onset, offset, header_text):
    """Stacked baseline vs opto traces, with opto-current trace below each row.

    Args
    ----
    baseline:     (n_traces, T) baseline voltage per representative neuron.
    opto:         (n_traces, T) opto-perturbed voltage.
    opto_current: (n_traces, T) optogenetics_stimulus per neuron.
    labels:       cell-type names per row.
    time_ms:      (T,) x-axis in ms.
    step_v:       vertical offset between rows.
    onset/offset: opto-on window boundaries (frame indices).
    """
    n_traces = baseline.shape[0]
    bl = baseline.mean(axis=1)
    s = _TRACE_SHRINK
    for i in range(n_traces):
        ax.plot(time_ms, s * (baseline[i] - bl[i]) + i * step_v,
                lw=_LW_GT, color=_COLOR_GT, alpha=0.95, zorder=2,
                label='baseline' if i == 0 else None)
        ax.plot(time_ms, s * (opto[i] - bl[i]) + i * step_v,
                lw=_LW_OPTO, color=_COLOR_OPTO, alpha=0.95, zorder=3,
                label='opto' if i == 0 else None)
        if opto_current is not None and opto_current[i].std() > 1e-9:
            curr = opto_current[i]
            curr_y = i * step_v - 0.4 * step_v
            ax.plot(time_ms, s * (curr - curr.mean()) + curr_y,
                    lw=_LW_STIM, color=_COLOR_STIM, alpha=0.95, zorder=4,
                    label='opto current' if i == 0 else None)
        ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.02,
                i * step_v, labels[i], fontsize=_FS_TYPE,
                va='bottom', ha='right', color='black')

    if onset > 0 or offset < len(time_ms):
        ax.axvspan(time_ms[onset], time_ms[min(offset, len(time_ms) - 1)],
                   color='orange', alpha=0.08, zorder=1)

    if header_text:
        ax.text(0.015, 0.99, header_text, transform=ax.transAxes,
                va='top', ha='left', fontsize=_FS_TICK,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=0.4))

    ax.set_ylabel('neurons', fontsize=_FS_LABEL, labelpad=18)
    ax.set_ylim([-step_v, (n_traces - 1) * step_v + 2.2 * step_v])
    ax.set_yticks([])
    ax.set_xlim([time_ms[0], time_ms[-1]])
    ax.set_xlabel('time (ms)', fontsize=_FS_LABEL, labelpad=1)
    ax.tick_params(axis='x', labelsize=_FS_TICK, pad=1)
    ax.spines['left'].set_visible(False)
    ax.legend(loc='upper right', fontsize=_FS_TICK, frameon=False)


def compare_traces(
    source_dataset: str,
    opto_dataset: str,
    split: str = "test",
    trace_start: int = 500,
    trace_end: int = 1500,
    save_fig_to: Optional[str] = None,
) -> dict:
    """Compare voltage trajectories with vs. without optogenetic perturbation.

    Loads both datasets, picks representative neurons (one per perturbed
    cell type plus the most-affected non-perturbed types up to PANEL_CAP),
    and renders a stacked-trace plot in the same visual style as
    figures/fig_rollout_4col_flywire_comparison.py: green = baseline,
    black = opto, red below = optogenetics_stimulus per neuron.
    Perturbed-type labels are suffixed with " *".

    Returns metrics dict {mean_abs_dv_per_type, opto_on_frames, fig_path}.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from connectome_gnn.neuron_state import NeuronTimeSeries
    from connectome_gnn.metrics import INDEX_TO_NAME, NAME_TO_INDEX
    from connectome_gnn.utils import graphs_data_path

    def _resolve(name):
        cand = graphs_data_path(name)
        if not os.path.isdir(cand):
            alt = graphs_data_path("fly", name)
            if os.path.isdir(alt):
                return alt
        return cand

    src_root, opto_root = _resolve(source_dataset), _resolve(opto_dataset)
    src_split = os.path.join(src_root, f"x_list_{split}")
    opto_split = os.path.join(opto_root, f"x_list_{split}")

    src_ts = NeuronTimeSeries.from_zarr_v3(src_split, fields=["voltage", "neuron_type"])
    opto_ts = NeuronTimeSeries.from_zarr_v3(
        opto_split, fields=["voltage", "optogenetics_stimulus", "neuron_type"],
    )
    T = min(src_ts.n_frames, opto_ts.n_frames)
    nt = opto_ts.neuron_type.cpu().numpy()

    # Locate opto-on window from the opto field
    opto_full = opto_ts.optogenetics_stimulus[:T] if opto_ts.optogenetics_stimulus is not None \
        else None
    if opto_full is not None:
        active = (opto_full.abs().sum(dim=1) > 0).cpu().numpy()
        if active.any():
            onset = int(active.argmax())
            offset = int(T - active[::-1].argmax())
        else:
            onset, offset = 0, T
    else:
        onset, offset = 0, T

    # Per-type mean |Δv| over the active window — for the metrics dict
    delta = (opto_ts.voltage[:T] - src_ts.voltage[:T]).abs()
    abs_dv_window = delta[onset:offset].mean(dim=0).cpu().numpy()
    mean_abs_dv: dict[str, float] = {}
    for t_id in np.unique(nt):
        idxs = np.where(nt == t_id)[0]
        mean_abs_dv[INDEX_TO_NAME.get(int(t_id), f"type_{t_id}")] = \
            float(abs_dv_window[idxs].mean())

    # Neuron selection — show both perturbed and non-perturbed:
    #   * perturbed: one representative per cell type that receives opto current
    #     (label suffixed with " *")
    #   * non-perturbed: one representative per cell type ranked by mean |Δv|
    #     (downstream of opto via the connectome) up to the panel cap
    if opto_full is not None:
        is_perturbed_neuron = (opto_full.abs().sum(dim=0) > 0).cpu().numpy()
    else:
        is_perturbed_neuron = np.zeros(len(nt), dtype=bool)
    perturbed_types = sorted(set(int(t) for t in nt[is_perturbed_neuron]))

    neuron_idx, labels = [], []
    seen_types: set[int] = set()
    # Perturbed types first — order by null_dim ranking-style importance
    # (just keep numerical order for determinism here).
    for t in perturbed_types:
        ids = np.where((nt == t) & is_perturbed_neuron)[0]
        if len(ids) > 0:
            neuron_idx.append(int(ids[0]))
            labels.append(INDEX_TO_NAME.get(int(t), f"type_{t}") + " *")
            seen_types.add(t)

    # Non-perturbed types — pick the most-affected first (largest mean |Δv|).
    PANEL_CAP = 12
    non_perturbed_ranked: list[tuple[int, float]] = []
    for t_id in np.unique(nt):
        t_int = int(t_id)
        if t_int in seen_types:
            continue
        ids = np.where(nt == t_int)[0]
        if len(ids) == 0:
            continue
        score = float(abs_dv_window[ids].mean())
        non_perturbed_ranked.append((t_int, score))
    non_perturbed_ranked.sort(key=lambda kv: kv[1], reverse=True)
    for t_int, _ in non_perturbed_ranked:
        if len(neuron_idx) >= PANEL_CAP:
            break
        ids = np.where(nt == t_int)[0]
        neuron_idx.append(int(ids[0]))
        labels.append(INDEX_TO_NAME.get(t_int, f"type_{t_int}"))

    # Slice the trace window
    sl = slice(trace_start, min(trace_end, T))
    v_b = src_ts.voltage[sl][:, neuron_idx].T.cpu().numpy()
    v_o = opto_ts.voltage[sl][:, neuron_idx].T.cpu().numpy()
    if opto_full is not None:
        i_o = opto_full[sl][:, neuron_idx].T.cpu().numpy()
    else:
        i_o = None

    # Build figure
    matplotlib.rc_file(os.path.join('/workspace/connectome-gnn/figures', 'janne.matplotlibrc'))
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 7.5))
    DT_MS = 20.0
    time_ms = np.arange(v_b.shape[1]) * DT_MS
    step_v = 1.4 * np.median([v_b[i].std() for i in range(v_b.shape[0])] + [1.0])
    header = (f"source: {source_dataset}\n"
              f"opto:   {opto_dataset}\n"
              f"split={split}  opto-on=[{onset}, {offset})")
    _draw_opto_traces(
        ax, v_b, v_o, i_o, labels, time_ms, step_v,
        onset=max(0, onset - sl.start), offset=min(v_b.shape[1], offset - sl.start),
        header_text=header,
    )

    fig.tight_layout()
    if save_fig_to is None:
        fig_dir = os.path.join(opto_root, "Fig")
        os.makedirs(fig_dir, exist_ok=True)
        save_fig_to = os.path.join(fig_dir, f"opto_compare_{split}.png")
    fig.savefig(save_fig_to, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "mean_abs_dv_per_type": mean_abs_dv,
        "opto_on_frames": (onset, offset),
        "fig_path": save_fig_to,
    }
