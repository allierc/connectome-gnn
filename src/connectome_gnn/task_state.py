"""Task state dataclasses for the drosophila_cx / cortex training pipelines.

Parallels neuron_state.py:
    NeuronState        -> TaskState     (single trial,  T-major)
    NeuronTimeSeries   -> TaskTrials    (batch of trials, (B, T, ...))

Replaces the loose (stimulus, target, c_mask, length, rule_idx) tuple
threaded through generation -> training -> testing -> figures with a
single named-field dataclass.  Field shape annotations are the source
of truth — generators populate them, trainers/testers consume them.

All fields default to None so a trial only carries what its task needs:
    cortex (Yang) trial:    stimulus / target / c_mask / length / rule_*
    PI (Hulse) trial:       stimulus / target / theta_hd / is_stop / omega
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any

import torch


# Field classification — used by I/O helpers to pick dtype/shape and
# decide which fields are time-varying.  Scalar metadata fields
# (n_input, n_output, dt, ...) are not tensors and live outside these
# sets.
STATIC_TENSOR_FIELDS: set[str] = set()                       # none today, kept for symmetry
DYNAMIC_FIELDS = {'stimulus', 'target', 'length',
                  'c_mask', 'stimulus_canonical', 'delta_stimulus',
                  'theta_hd', 'is_stop', 'omega'}
CORTEX_FIELDS = {'c_mask', 'stimulus_canonical', 'delta_stimulus', 'rule_idx', 'rule_name', 'epochs'}
PI_FIELDS = {'theta_hd', 'is_stop', 'omega'}
ALL_TENSOR_FIELDS = STATIC_TENSOR_FIELDS | DYNAMIC_FIELDS


def _apply(tensor, fn):
    """Apply fn to tensor if not None, else return None."""
    return fn(tensor) if tensor is not None else None


@dataclass
class TaskState:
    """Single-trial task state.

    Static fields (set once per trial, do not vary across frames):
        n_input, n_output, dt, rule_idx, rule_name, task_family

    Dynamic fields (per simulation frame, T leading axis):
        stimulus, target, length

    Per-task fields are populated only for the relevant family and
    mirror the # spiking / # HH blocks in NeuronState.
    """

    # static — per-trial metadata (scalars / strings, not tensors)
    task_family: str | None = None     # 'cortex' | 'path_integration' | None
    n_input: int | None = None         # input channel count N_i
    n_output: int | None = None        # output channel count N_o
    dt: float | None = None            # frame timestep (seconds)
    rule_idx: int | None = None        # multi-task id (cortex), None for PI
    rule_name: str | None = None       # human-readable rule (cortex)

    # dynamic — per-frame signals shared by both task families
    stimulus: torch.Tensor | None = None   # (T, N_i) float32 — input drive
    target:   torch.Tensor | None = None   # (T, N_o) float32 — desired output
    length:   torch.Tensor | None = None   # (T,)     float32 — real-step / valid mask
                                           #         (1.0 inside the trial, 0.0 in padding)

    # cortex (Yang multitask) — per-output cost mask + perturbation
    # breakdown + epoch markers.  None for PI trials.
    c_mask:             torch.Tensor | None = None  # (T, N_o) float32 — Yang's cost weighting
    stimulus_canonical: torch.Tensor | None = None  # (T, N_i) float32 — pre-perturbation input
    delta_stimulus:     torch.Tensor | None = None  # (T, N_i) float32 — added perturbation
    epochs:             dict | None        = None   # Yang trial epoch boundaries (plotting only)

    # path-integration (Hulse) — ground-truth heading + standing-pause
    # mask + raw angular velocity.  None for cortex trials.
    theta_hd: torch.Tensor | None = None   # (T,) float32 — ground-truth heading (rad)
    is_stop:  torch.Tensor | None = None   # (T,) float32 — standing-pause mask
    omega:    torch.Tensor | None = None   # (T,) float32 — angular velocity (deg/s)

    @property
    def n_frames(self) -> int:
        """Infer T from the first non-None dynamic tensor."""
        for name in DYNAMIC_FIELDS:
            val = getattr(self, name, None)
            if isinstance(val, torch.Tensor):
                return val.shape[0]
        raise ValueError("TaskState has no populated dynamic fields")

    @property
    def device(self) -> torch.device:
        """Infer device from the first non-None tensor field."""
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                return val.device
        raise ValueError("TaskState has no populated tensor fields")

    def to(self, device: torch.device) -> TaskState:
        """Move all non-None tensors to device; scalars/strings/dicts pass through."""
        return TaskState(**{
            f.name: (_apply(getattr(self, f.name), lambda t: t.to(device))
                     if isinstance(getattr(self, f.name), torch.Tensor)
                     else getattr(self, f.name))
            for f in dc_fields(self)
        })

    def clone(self) -> TaskState:
        return TaskState(**{
            f.name: (_apply(getattr(self, f.name), lambda t: t.clone())
                     if isinstance(getattr(self, f.name), torch.Tensor)
                     else getattr(self, f.name))
            for f in dc_fields(self)
        })

    def detach(self) -> TaskState:
        return TaskState(**{
            f.name: (_apply(getattr(self, f.name), lambda t: t.detach())
                     if isinstance(getattr(self, f.name), torch.Tensor)
                     else getattr(self, f.name))
            for f in dc_fields(self)
        })


@dataclass
class TaskTrials:
    """Batch of task trials — same field set as TaskState with a
    leading batch dimension on every dynamic field.

    Static fields are stored once (same for all trials in the batch),
    except `rule_idx`, which is per-trial when the cortex pipeline
    samples multiple rules into one dataset.
    """

    # static — batch-wide metadata
    task_family: str | None = None
    n_input: int | None = None
    n_output: int | None = None
    dt: float | None = None
    rules: list[str] | None = None              # ordered rule names, for cortex multitask
    ruleset: str | None = None                  # 'all', 'mante', ... (cortex only)

    # static-but-per-trial — survives even when the trial is a single
    # rule.  Shape (B,) long for cortex, None for PI.
    rule_idx: torch.Tensor | None = None

    # dynamic — per-frame, leading batch dim
    stimulus: torch.Tensor | None = None   # (B, T, N_i) float32
    target:   torch.Tensor | None = None   # (B, T, N_o) float32
    length:   torch.Tensor | None = None   # (B, T)      float32 — real-step mask

    # cortex
    c_mask:             torch.Tensor | None = None  # (B, T, N_o) float32
    stimulus_canonical: torch.Tensor | None = None  # (B, T, N_i) float32
    delta_stimulus:     torch.Tensor | None = None  # (B, T, N_i) float32
    epochs:             list[Any]   | None = None   # per-trial epoch dicts (Yang)

    # path-integration
    theta_hd: torch.Tensor | None = None   # (B, T) float32
    is_stop:  torch.Tensor | None = None   # (B, T) float32
    omega:    torch.Tensor | None = None   # (B, T) float32

    @property
    def n_trials(self) -> int:
        for name in DYNAMIC_FIELDS:
            val = getattr(self, name, None)
            if isinstance(val, torch.Tensor):
                return val.shape[0]
        raise ValueError("TaskTrials has no populated dynamic fields")

    @property
    def n_frames(self) -> int:
        for name in DYNAMIC_FIELDS:
            val = getattr(self, name, None)
            if isinstance(val, torch.Tensor):
                return val.shape[1]
        raise ValueError("TaskTrials has no populated dynamic fields")

    def trial(self, b: int) -> TaskState:
        """Extract a single trial as a TaskState (T-major)."""
        kw: dict[str, Any] = {
            'task_family': self.task_family,
            'n_input':     self.n_input,
            'n_output':    self.n_output,
            'dt':          self.dt,
        }
        if self.rule_idx is not None:
            ri = int(self.rule_idx[b].item())
            kw['rule_idx'] = ri
            if self.rules is not None and 0 <= ri < len(self.rules):
                kw['rule_name'] = self.rules[ri]
        for name in DYNAMIC_FIELDS:
            val = getattr(self, name, None)
            kw[name] = val[b] if isinstance(val, torch.Tensor) else None
        if self.epochs is not None and b < len(self.epochs):
            kw['epochs'] = self.epochs[b]
        return TaskState(**kw)

    def to(self, device: torch.device) -> TaskTrials:
        return TaskTrials(**{
            f.name: (_apply(getattr(self, f.name), lambda t: t.to(device))
                     if isinstance(getattr(self, f.name), torch.Tensor)
                     else getattr(self, f.name))
            for f in dc_fields(self)
        })

    def subset_trials(self, ids) -> TaskTrials:
        """Select a subset of trials by index — used by the train loop
        to slice minibatches without copying static metadata."""
        kw: dict[str, Any] = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                # rule_idx is (B,); everything else dynamic is (B, T, ...)
                kw[f.name] = val[ids]
            elif f.name == 'epochs' and isinstance(val, list):
                ids_list = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
                kw[f.name] = [val[i] for i in ids_list]
            else:
                kw[f.name] = val
        return TaskTrials(**kw)

    # ------------------------------------------------------------------
    # Disk I/O — symmetric with NeuronTimeSeries.from_zarr_v3 / .load
    # ------------------------------------------------------------------
    @classmethod
    def from_disk(cls, path) -> "TaskTrials":
        """Read a TaskTrials from a folder written by
        :class:`connectome_gnn.zarr_io.ZarrTaskTrialsWriter` or by
        :func:`task_trials_to_disk`."""
        return task_trials_from_disk(path)

    @classmethod
    def load(cls, path) -> "TaskTrials":
        """Dispatcher alias for :meth:`from_disk` (matches
        ``NeuronTimeSeries.load`` naming)."""
        return task_trials_from_disk(path)


# ---------------------------------------------------------------------------
# Disk I/O — flat folder of <field>.zarr files plus a meta.json sidecar.
# Same on-disk layout the legacy generators wrote (stimulus.zarr / target.zarr
# etc.), so existing zarr readers in the trainer keep working unchanged.
# ---------------------------------------------------------------------------

def task_trials_to_disk(
    trials: TaskTrials,
    split_dir,
    *,
    chunk_trials: int = 1000,
) -> int:
    """Serialise a TaskTrials to ``split_dir`` via :class:`ZarrTaskTrialsWriter`.

    Thin convenience wrapper for callers who already have the whole
    ``TaskTrials`` materialised in RAM: iterates trial-by-trial through the
    streaming writer so the on-disk layout is identical to a per-trial
    streaming generator (e.g. the cortex 20-task generator we're about to
    refactor).
    """
    from connectome_gnn.zarr_io import ZarrTaskTrialsWriter

    writer = ZarrTaskTrialsWriter(split_dir, chunk_trials=chunk_trials)
    for b in range(int(trials.n_trials)):
        writer.append_trial(trials.trial(b))
    return writer.finalize()


def task_trials_from_disk(split_dir) -> TaskTrials:
    """Read a directory written by :class:`ZarrTaskTrialsWriter` /
    :func:`task_trials_to_disk` back into a TaskTrials. Missing fields stay
    ``None``."""
    import json
    import os

    import numpy as np
    import zarr

    meta_path = os.path.join(split_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    else:
        meta = {}

    kw: dict[str, Any] = {
        "task_family": meta.get("task_family"),
        "n_input":     meta.get("n_input"),
        "n_output":    meta.get("n_output"),
        "dt":          meta.get("dt"),
        "rules":       meta.get("rules"),
        "ruleset":     meta.get("ruleset"),
    }

    for f in dc_fields(TaskTrials):
        if f.name in kw:
            continue                            # static, already filled
        zpath = os.path.join(split_dir, f"{f.name}.zarr")
        if not os.path.isdir(zpath):
            continue
        arr = np.asarray(zarr.open(zpath, mode="r"))
        # Writer convention: time-series 1-D fields are stored as (B, T, 1);
        # squeeze back to (B, T) on load. rule_idx is (B, 1, 1) → (B,) long.
        if arr.ndim == 3 and arr.shape[-1] == 1 and f.name in {
                "length", "theta_hd", "is_stop", "omega"}:
            arr = arr[..., 0]
        if f.name == "rule_idx" and arr.ndim == 3:
            arr = arr[:, 0, 0].astype("int64")
            kw[f.name] = torch.from_numpy(arr)
            continue
        kw[f.name] = torch.from_numpy(arr.astype("float32"))

    epochs = meta.get("epochs")
    if epochs is not None:
        kw["epochs"] = epochs
    return TaskTrials(**kw)
