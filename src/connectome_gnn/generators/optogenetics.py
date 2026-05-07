"""Optogenetics twin-dataset pipeline.

This module implements the experimental arm of the structural-null-space
result (scripts/structural_nullspace_table.py): naturalistic visual drive
leaves ~71% of the connectome's edge-weight space in ker(H), structured as
sum-zero redistributions within (postsynaptic neuron, presynaptic cell-type)
columnar groups. Per-column-distinct optogenetic perturbation is the only
intervention that breaks the kernel.

Public API:
    make_optogenetics_stimulus  — (T, N) float32 tensor, zero off-target
    build_target_mask           — (N,) bool mask of opto targets
    build_waveform              — (T, n_targets) float32 waveform
    add_optogenetics_stimulus   — twin-dataset second pass (Phase 5)
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


def add_optogenetics_stimulus(config) -> None:
    """Twin-dataset second pass — implemented in Phase 5 follow-up."""
    raise NotImplementedError(
        "add_optogenetics_stimulus is implemented in Phase 5 of the optogenetics "
        "branch — this stub is a placeholder so the module is importable while the "
        "earlier phases land first."
    )


def compare_traces(*args, **kwargs) -> dict:
    """Quantify with-vs-without-opto traces — implemented in Phase 10."""
    raise NotImplementedError("compare_traces is implemented in Phase 10.")
