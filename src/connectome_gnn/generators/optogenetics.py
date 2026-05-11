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
    device: torch.device | str = 'cpu',
) -> torch.Tensor:
    """Resolve per-target base amplitude on `device`.

    If waveform.amplitude is set, every target gets that scalar.
    If waveform.amplitude is None, look up per-type lambda_max from the
    nullspace JSON and use 0.5 * lambda_max(type_of_target). Falls back to 1.0
    when lambda_max is missing or NaN (older JSONs predate the instrumentation).
    """
    n_targets = int(target_indices.numel())
    if waveform.amplitude is not None:
        return torch.full((n_targets,), float(waveform.amplitude), device=device)

    try:
        ranking = load_nullspace_ranking(
            json_path=nullspace_json_path, metric='null_dim'
        )
    except FileNotFoundError:
        return torch.full((n_targets,), 1.0, device=device)
    lam_by_name: dict[str, float] = {
        name: lam for (name, _s, lam) in ranking
        if not math.isnan(lam)
    }
    out = torch.empty(n_targets, dtype=torch.float32, device=device)
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
    device: torch.device | str = 'cpu',
) -> torch.Tensor:
    """(T, n_targets) float32 on `device` — base waveform + universal noise.

    column_distinct=True → independent realizations per target.
    column_distinct=False → identical waveform replicated across targets
        (preserves the columnar sum-zero kernel; emits warning at caller).
    """
    n_targets = int(target_indices.numel())
    amps = _per_target_amplitudes(target_indices, target_type_per_neuron, waveform,
                                  nullspace_json_path, device=device)  # (n_targets,)

    # Generator must match the device of the tensors it seeds.
    gen_device = 'cuda' if torch.device(device).type == 'cuda' else 'cpu'
    g = torch.Generator(device=gen_device)
    g.manual_seed(int(waveform.seed))

    kind = waveform.kind if isinstance(waveform.kind, str) else waveform.kind.value

    # ---- base waveform (T, n_targets) — opto runs across the whole [0, T) ----
    base = torch.zeros(n_frames, n_targets, dtype=torch.float32, device=device)
    if kind == OptoWaveformKind.WHITE_NOISE.value:
        # white_noise has no deterministic component; noise_level drives signal
        pass
    elif kind == OptoWaveformKind.CONSTANT.value:
        if column_distinct:
            base[:, :] = amps[None, :]
        else:
            base[:, :] = amps.mean()
    elif kind == OptoWaveformKind.HEAVISIDE.value:
        # Square wave: frames_on ON, frames_on OFF, repeat. Set frames_on=0
        # for a one-shot DC step (always ON).
        frames_on = int(getattr(waveform, 'frames_on', 0) or 0)
        if frames_on <= 0:
            if column_distinct:
                base[:, :] = amps[None, :]
            else:
                base[:, :] = amps.mean()
        else:
            if column_distinct:
                # Independent per-column random telegraph signal: flip ON/OFF
                # with probability 1/frames_on per frame (mean dwell ≈
                # frames_on, matches the deterministic schedule's timescale
                # but desynchronises columns so the columnar sum-zero kernel
                # is broken). Per-column amplitude is U(0, 1) to add a second
                # axis of asymmetry. State starts at 0 (OFF).
                p_flip = 1.0 / float(frames_on)
                flips = (torch.rand(n_frames, n_targets, generator=g,
                                    dtype=torch.float32, device=device)
                         < p_flip)
                state01 = torch.cumsum(flips.to(torch.int32), dim=0) % 2
                if getattr(waveform, 'resample_amplitude_per_transition', False):
                    # Fresh U(0,1) gain per (column, segment), where a
                    # "segment" is the run between two consecutive flips.
                    # Removes the persistent column-identity label (fixed
                    # gain across the whole trajectory) while keeping the
                    # column-distinct telegraph timing — isolates temporal
                    # decorrelation from the persistent-gain mechanism.
                    seg_id = torch.cumsum(flips.to(torch.int64), dim=0)
                    max_seg = int(seg_id.max().item()) + 1
                    seg_amps = torch.rand(max_seg, n_targets, generator=g,
                                          dtype=torch.float32, device=device)
                    col_amps_t = torch.gather(seg_amps, 0, seg_id)
                    base = state01.float() * col_amps_t
                else:
                    col_amps = torch.rand(n_targets, generator=g,
                                          dtype=torch.float32, device=device)
                    base = state01.float() * col_amps[None, :]
            else:
                period = 2 * frames_on
                on_mask = (torch.arange(n_frames, device=device) % period) < frames_on
                base[on_mask, :] = amps.mean()
    elif kind == OptoWaveformKind.IMPULSE.value:
        pw = max(1, int(waveform.pulse_width_frames))
        pp = max(pw + 1, int(waveform.pulse_period_frames))
        for k in range(0, n_frames // pp + 1):
            a = k * pp
            b = min(a + pw, n_frames)
            if a >= n_frames:
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
        base[:, :] = arr
    else:
        raise ValueError(f"unknown waveform kind: {kind}")

    # ---- universal additive Gaussian noise (applied across all frames) ----
    if waveform.noise_level > 0.0:
        if column_distinct:
            xi = torch.randn(n_frames, n_targets, generator=g,
                             dtype=torch.float32, device=device)
        else:
            shared = torch.randn(n_frames, 1, generator=g,
                                 dtype=torch.float32, device=device)
            xi = shared.expand(n_frames, n_targets).clone()
        base = base + xi * float(waveform.noise_level)

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
    # Allocate everything on the same device as the simulation state.
    device = state.neuron_type.device

    if not opto_cfg.enabled:
        return torch.zeros(n_frames, state.n_neurons, dtype=torch.float32, device=device)

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
    target_indices = torch.nonzero(mask, as_tuple=False).flatten().to(device)

    if target_indices.numel() == 0:
        warnings.warn("OptoTargetSpec resolves to ZERO targets — opto tensor will be all zeros.",
                      UserWarning)
        return torch.zeros(n_frames, state.n_neurons, dtype=torch.float32, device=device)

    type_names_per_neuron = neuron_type_names(state.neuron_type)
    wf = build_waveform(
        n_frames=n_frames,
        target_indices=target_indices,
        waveform=opto_cfg.waveform,
        target_type_per_neuron=type_names_per_neuron,
        column_distinct=spec.column_distinct,
        nullspace_json_path=nullspace_json_path,
        device=device,
    )

    out = torch.zeros(n_frames, state.n_neurons, dtype=torch.float32, device=device)
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
    from connectome_gnn.utils import to_numpy
    from connectome_gnn.zarr_io import ZarrArrayWriter, ZarrSimulationWriterV3

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

    # Mirror the source's ode_params.pt into the target — the ODE parameters
    # (connectome weights, per-neuron biophysics) are unchanged under opto
    # perturbation, only the injected current changes. Trainer requires this
    # file at graphs_data_path(config.dataset)/ode_params.pt.
    target_ode_params = os.path.join(target_root, "ode_params.pt")
    if not os.path.isfile(target_ode_params):
        import shutil
        shutil.copy2(ode_params_path, target_ode_params)
        log.info(f"opto: copied ode_params.pt to {target_ode_params}")

    device = _resolve_device(config)
    model_type = config.graph_model.signal_model_name

    # Source neuron-type assignment.
    nt_path = os.path.join(source_root, "x_list_train", "neuron_type.zarr")
    neuron_types = torch.from_numpy(
        np.array(zarr.open_array(nt_path, mode='r'), dtype=np.int64)
    ).long().to(device)

    ode, edge_index = _build_flyvis_ode(state_dict, neuron_types, model_type, device)

    n_neurons = int(neuron_types.numel())

    # Calcium kernel setup (mirrors graph_data_generator._run_ode_generation).
    # When calcium_type == "kernel", maintain a per-neuron voltage-history buffer
    # and compute state.calcium = sum_k K[k] * V[t-k] each step. y = dC/dt
    # replaces dV/dt as the supervision target.
    calcium_kernel = None
    trace_neuron_idx: list[int] = []
    trace_labels: list[str] = []
    TRACE_MAX_FRAMES = 1500
    if sim.calcium_type == "kernel":
        from connectome_gnn.generators.gcamp_kernel import (
            build_kernel_from_config,
            select_reference_neurons,
        )
        calcium_kernel = build_kernel_from_config(sim, device=device)
        trace_neuron_idx, trace_labels = select_reference_neurons(neuron_types)

        # Diagnostic K(t) plot at the dataset root, matching the source layout.
        import matplotlib
        import matplotlib.pyplot as plt
        t_axis = np.arange(calcium_kernel.shape[0]) * sim.calcium_kernel_dt_seconds
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(t_axis, calcium_kernel.cpu().numpy(), color="tab:blue")
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("K(t)", fontsize=8)
        ax.set_title(
            f"{sim.calcium_kernel_variant} kernel "
            f"(tau_r={sim.calcium_kernel_tau_rise:.3f}s, "
            f"tau_d={sim.calcium_kernel_tau_decay:.3f}s)",
            fontsize=8,
        )
        ax.tick_params(axis="both", labelsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(target_root, "kernel.png"), dpi=120)
        plt.close(fig)
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

        # Per-split kernel state (voltage + stimulus history + trace buffers).
        v_hist = None
        stim_hist = None
        v_trace_buf: list = []
        ca_trace_buf: list = []
        if calcium_kernel is not None:
            v_hist = torch.zeros(
                (n_neurons, calcium_kernel.shape[0]),
                dtype=torch.float32,
                device=device,
            )
            stim_hist = torch.zeros(
                (n_neurons, calcium_kernel.shape[0]),
                dtype=torch.float32,
                device=device,
            )
            state.calcium = torch.zeros(n_neurons, dtype=torch.float32, device=device)
            state.fluorescence = torch.zeros(n_neurons, dtype=torch.float32, device=device)
            state.stimulus_calcium = torch.zeros(n_neurons, dtype=torch.float32, device=device)

        # Per-split opto tensor (T, N) — already allocated on `device` since
        # state.neuron_type lives there.
        opto_full = make_optogenetics_stimulus(
            state, n_frames=T, opto_cfg=opto_cfg,
            nullspace_json_path=opto_cfg.target.structural_table_json,
        )

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
            save_calcium=(calcium_kernel is not None),
        )
        y_writer = ZarrArrayWriter(
            path=os.path.join(target_root, f"y_list_{split}"),
            n_neurons=n_neurons,
            n_features=1,
            time_chunks=2000,
        )

        # AR(1) state for measurement noise (zero at start, matches source).
        ar1_prev_noise = torch.zeros(n_neurons, dtype=torch.float32, device=device)
        ar1_inject_std = (
            split_noise_meas * (1.0 - ar1_rho ** 2) ** 0.5 if ar1_rho > 0 else 0.0
        )

        # Forward integration. Per-frame RNG order MATCHES
        # graph_data_generator._run_ode_generation L2208-2253 so that with a
        # matched seed the noise streams are statistically equivalent.
        from tqdm import tqdm
        with torch.no_grad():
            for t in tqdm(range(T), desc=f"opto {split}", ncols=100):
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

                # Calcium kernel: update history with current voltage and
                # compute state.calcium for the current frame. Same kernel is
                # applied to the visual stimulus so the excitation channel
                # lives in the same temporal regime. y = dC/dt replaces dV/dt
                # as the supervision target in kernel mode.
                if calcium_kernel is not None:
                    prev_calcium = state.calcium.clone()
                    v_hist = torch.roll(v_hist, shifts=1, dims=-1)
                    v_hist[:, 0] = state.voltage
                    state.calcium = (v_hist * calcium_kernel).sum(dim=-1)
                    state.fluorescence = (
                        sim.calcium_alpha * state.calcium + sim.calcium_beta
                    )
                    stim_hist = torch.roll(stim_hist, shifts=1, dims=-1)
                    stim_hist[:, 0] = state.stimulus
                    state.stimulus_calcium = (stim_hist * calcium_kernel).sum(dim=-1)
                    y_record = ((state.calcium - prev_calcium) / dt).unsqueeze(-1)
                    if trace_neuron_idx and len(v_trace_buf) < TRACE_MAX_FRAMES:
                        v_trace_buf.append(
                            state.voltage[trace_neuron_idx]
                            .detach().cpu().numpy().copy()
                        )
                        ca_trace_buf.append(
                            state.calcium[trace_neuron_idx]
                            .detach().cpu().numpy().copy()
                        )
                else:
                    y_record = dv.unsqueeze(-1)

                # 3. Snapshot CURRENT frame.
                writer.append_state(state)
                y_writer.append(to_numpy(y_record.clone().detach()))

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
        y_writer.finalize()
        log.info(f"opto [{split}]: wrote {n_written} frames to {target_split}")

        # Diagnostic calcium-trace plot for the train split, matching the
        # opto_compare window (frames 500-1500 -> 10000-30000 ms at dt=20ms).
        if calcium_kernel is not None and v_trace_buf and split == 'train':
            try:
                from connectome_gnn.generators.gcamp_kernel import (
                    plot_voltage_calcium_traces,
                )
                v_arr = np.stack(v_trace_buf, axis=0)
                ca_arr = np.stack(ca_trace_buf, axis=0)
                TRACE_START, TRACE_END = 500, 1500
                n_buf = v_arr.shape[0]
                start = min(TRACE_START, max(0, n_buf - 2))
                end = min(TRACE_END, n_buf)
                v_arr = v_arr[start:end]
                ca_arr = ca_arr[start:end]
                save_path = os.path.join(target_root, "calcium_trace.png")
                plot_voltage_calcium_traces(
                    voltage=v_arr,
                    calcium=ca_arr,
                    labels=trace_labels,
                    dt_seconds=float(sim.calcium_kernel_dt_seconds),
                    save_path=save_path,
                    start_frame=start,
                    title=(
                        f"{sim.calcium_kernel_variant}: V (left) vs F=V*K "
                        f"(right) - frames [{start}, {end})"
                    ),
                )
                log.info(f"opto [{split}]: calcium_trace -> {save_path}")
            except Exception as e:
                log.warning(f"opto [{split}]: calcium_trace plot failed: {e}")

        # Post-generation summary for this split.
        # voltage stats + Δv vs source on the perturbed mask vs the rest.
        from connectome_gnn.zarr_io import load_simulation_data
        x_ts_new = load_simulation_data(
            target_split, fields=['voltage', 'optogenetics_stimulus'],
        )
        v_new = x_ts_new.voltage.numpy()                # (T, N)
        v_src_arr = ts.voltage[:n_written].cpu().numpy()
        opto_arr = opto_full[:n_written].cpu().numpy()
        mask_np = mask.cpu().numpy()
        delta = v_new - v_src_arr                       # (T, N)
        Y, R = "\033[93m", "\033[0m"
        print(f"{Y}[opto {split} summary]{R}")
        print(f"  saved        : {target_split}")
        print(f"  frames       : {n_written}")
        print(f"  voltage      : mean={v_new.mean():+.4f}  std={v_new.std():.4f}  "
              f"min={v_new.min():+.4f}  max={v_new.max():+.4f}")
        print(f"  opto current : mean={opto_arr[:, mask_np].mean():+.4f}  "
              f"std={opto_arr[:, mask_np].std():.4f}  "
              f"max|·|={abs(opto_arr).max():.4f}")
        print(f"  |Δv| target  : mean={abs(delta[:, mask_np]).mean():.4f}  "
              f"max={abs(delta[:, mask_np]).max():.4f}  "
              f"({int(mask_np.sum())} perturbed neurons)")
        if (~mask_np).any():
            d_off = delta[:, ~mask_np]
            print(f"  |Δv| other   : mean={abs(d_off).mean():.4f}  "
                  f"max={abs(d_off).max():.4f}  "
                  f"({int((~mask_np).sum())} downstream/unrelated)")

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

    ax.set_ylabel('neurons', fontsize=_FS_LABEL, labelpad=32)
    ax.set_ylim([-step_v, (n_traces - 1) * step_v + 2.2 * step_v])
    ax.set_yticks([])
    ax.set_xlim([time_ms[0], time_ms[-1]])
    # Tick layout matching fig_rollout_3col_noise_comparison.py _pretty_ticks
    # (n_target=3 → 10000/20000/30000 ms for a 500-1500 frame window).
    lo, hi = time_ms[0], time_ms[-1]
    raw_step = (hi - lo) / max(1, 3 - 1)
    mag = 10 ** np.floor(np.log10(max(raw_step, 1e-12)))
    step = mag
    for m in (1, 2, 5, 10):
        if m * mag >= raw_step:
            step = m * mag
            break
    tick_lo = np.ceil(lo / step - 1e-9) * step
    ticks = list(np.arange(tick_lo, hi + step / 2, step))
    if ticks:
        ax.set_xticks(ticks)
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

    # Neuron selection — match the canonical SELECTED_TYPES from
    # figures/fig_rollout_3col_noise_comparison.py (rendered top -> bottom):
    #   Am, T1, T5a, T4a, Tm9, Tm1, Mi9, Mi1, L3, L2, L1, R1
    # so opto_compare_*.png is directly comparable to the reference figure.
    # Any perturbed cell type missing from this list is appended at the BOTTOM
    # (above R1) with a " *" suffix; types already in the list get the suffix
    # in place.
    if opto_full is not None:
        is_perturbed_neuron = (opto_full.abs().sum(dim=0) > 0).cpu().numpy()
    else:
        is_perturbed_neuron = np.zeros(len(nt), dtype=bool)
    perturbed_types = set(int(t) for t in nt[is_perturbed_neuron])

    REFERENCE_TYPES = [0, 31, 39, 35, 55, 43, 22, 12, 7, 6, 5, 23]
    # Names: Am, T1, T5a, T4a, Tm9, Tm1, Mi9, Mi1, L3, L2, L1, R1

    neuron_idx, labels = [], []
    seen_types: set[int] = set()
    for t_int in REFERENCE_TYPES:
        ids = np.where(nt == t_int)[0]
        if len(ids) == 0:
            continue
        if t_int in perturbed_types:
            ids_pert = np.where((nt == t_int) & is_perturbed_neuron)[0]
            neuron_idx.append(int(ids_pert[0] if len(ids_pert) else ids[0]))
            labels.append(INDEX_TO_NAME.get(t_int, f"type_{t_int}") + " *")
        else:
            neuron_idx.append(int(ids[0]))
            labels.append(INDEX_TO_NAME.get(t_int, f"type_{t_int}"))
        seen_types.add(t_int)

    # Append any perturbed types that aren't in REFERENCE_TYPES
    # (e.g. TmY15, Tm3, Mi4 ...). Insert just before R1 (the last entry) so the
    # photoreceptor stays at the bottom edge of the panel.
    for t_int in sorted(perturbed_types - seen_types):
        ids = np.where((nt == t_int) & is_perturbed_neuron)[0]
        if len(ids) == 0:
            continue
        # insert above R1 (last entry) — keep R1 visually at the bottom
        insert_at = len(neuron_idx) - 1 if neuron_idx and labels[-1].startswith("R1") else len(neuron_idx)
        neuron_idx.insert(insert_at, int(ids[0]))
        labels.insert(insert_at, INDEX_TO_NAME.get(t_int, f"type_{t_int}") + " *")
        seen_types.add(t_int)

    # Right-panel selection: up to PANEL_CAP perturbed neurons sampled
    # uniformly across the targeted-cell-type columns, so the per-column
    # heterogeneity of the opto drive (and its downstream effect) is visible.
    pert_indices_all = np.where(is_perturbed_neuron)[0]
    PANEL_CAP = max(1, len(neuron_idx))
    if len(pert_indices_all) > 0:
        if len(pert_indices_all) > PANEL_CAP:
            sel = np.linspace(0, len(pert_indices_all) - 1, PANEL_CAP).round().astype(int)
            pert_neuron_idx = [int(pert_indices_all[i]) for i in sel]
        else:
            pert_neuron_idx = [int(i) for i in pert_indices_all]
        pert_labels = []
        for k, gi in enumerate(pert_neuron_idx):
            t_int = int(nt[gi])
            type_name = INDEX_TO_NAME.get(t_int, f"type_{t_int}")
            pert_labels.append(f"{type_name} #{k}")
    else:
        pert_neuron_idx, pert_labels = [], []

    # Slice the trace window. trace_start/trace_end are FRAME indices; the
    # x-axis is rendered in ms starting at trace_start * DT_MS to match
    # figures/fig_rollout_3col_noise_comparison.py (default window
    # 500-1500 frames -> 10000-30000 ms at DT=20 ms).
    sl = slice(trace_start, min(trace_end, T))
    v_b = src_ts.voltage[sl][:, neuron_idx].T.cpu().numpy()
    v_o = opto_ts.voltage[sl][:, neuron_idx].T.cpu().numpy()
    if opto_full is not None:
        i_o = opto_full[sl][:, neuron_idx].T.cpu().numpy()
    else:
        i_o = None
    if pert_neuron_idx:
        v_b_pert = src_ts.voltage[sl][:, pert_neuron_idx].T.cpu().numpy()
        v_o_pert = opto_ts.voltage[sl][:, pert_neuron_idx].T.cpu().numpy()
        i_o_pert = (opto_full[sl][:, pert_neuron_idx].T.cpu().numpy()
                    if opto_full is not None else None)

    # Build figure — left: per-cell-type reference traces; right: perturbed
    # neurons across columns of the targeted type. step_v shared so both
    # panels are visually comparable.
    matplotlib.rc_file(os.path.join('/workspace/connectome-gnn/figures', 'janne.matplotlibrc'))
    DT_MS = 20.0
    time_ms = np.arange(v_b.shape[1]) * DT_MS + trace_start * DT_MS
    row_stds = [float(v_b[i].std()) for i in range(v_b.shape[0])]
    if pert_neuron_idx:
        row_stds += [float(v_b_pert[i].std()) for i in range(v_b_pert.shape[0])]
    step_v = max(0.5 * _TRACE_SHRINK,
                 3.0 * _TRACE_SHRINK * (max(row_stds) if row_stds else 1.0))
    header = (f"source: {source_dataset}\n"
              f"opto:   {opto_dataset}\n"
              f"split={split}  opto-on=[{onset}, {offset})")

    if pert_neuron_idx:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
        ax_left, ax_right = axes
    else:
        fig, ax_left = plt.subplots(1, 1, figsize=(6.0, 6.0))
        ax_right = None

    _draw_opto_traces(
        ax_left, v_b, v_o, i_o, labels, time_ms, step_v,
        onset=max(0, onset - sl.start), offset=min(v_b.shape[1], offset - sl.start),
        header_text=header,
    )
    ax_left.set_title('reference cell types', fontsize=_FS_LABEL)
    if ax_right is not None:
        _draw_opto_traces(
            ax_right, v_b_pert, v_o_pert, i_o_pert, pert_labels, time_ms, step_v,
            onset=max(0, onset - sl.start),
            offset=min(v_b_pert.shape[1], offset - sl.start),
            header_text=None,
        )
        ax_right.set_title(f'perturbed neurons (n={len(pert_neuron_idx)})',
                           fontsize=_FS_LABEL)

    fig.tight_layout()
    if save_fig_to is None:
        save_fig_to = os.path.join(opto_root, f"opto_compare_{split}.png")
    fig.savefig(save_fig_to, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return {
        "mean_abs_dv_per_type": mean_abs_dv,
        "opto_on_frames": (onset, offset),
        "fig_path": save_fig_to,
    }
