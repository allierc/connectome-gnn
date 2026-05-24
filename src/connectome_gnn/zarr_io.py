"""zarr/tensorstore I/O utilities for simulation data.

provides:
- ZarrSimulationWriterV3: per-field writer for NeuronState data (static + dynamic fields)
- ZarrArrayWriter: incremental writer for raw (T, N, F) arrays (e.g. derivative targets)
- detect_format: check if V3 zarr or .npy exists at path
- load_simulation_data: load as NeuronTimeSeries with optional field selection
- load_raw_array: load raw numpy array from zarr or npy (for derivative targets etc.)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tensorstore as ts

if TYPE_CHECKING:
    from connectome_gnn.neuron_state import NeuronState, NeuronTimeSeries


class ZarrArrayWriter:
    """Incremental writer for raw (T, N, F) zarr arrays.

    Used for derivative targets (y_list) and other non-NeuronState data.

    Usage:
        writer = ZarrArrayWriter(path, n_neurons=14011, n_features=1)
        for frame in simulation:
            writer.append(frame)  # frame is (N, F)
        writer.finalize()
    """

    def __init__(
        self,
        path: str | Path,
        n_neurons: int,
        n_features: int,
        time_chunks: int = 2000,
        dtype: np.dtype = np.float32,
    ):
        self.path = Path(path)
        if not str(self.path).endswith('.zarr'):
            self.path = Path(str(self.path) + '.zarr')

        self.n_neurons = n_neurons
        self.n_features = n_features
        self.time_chunks = time_chunks
        self.dtype = dtype

        self._buffer: list[np.ndarray] = []
        self._total_frames = 0
        self._store: ts.TensorStore | None = None
        self._initialized = False

    def _initialize_store(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if self.path.exists():
            import shutil
            shutil.rmtree(self.path, ignore_errors=True)

        initial_cap = max(self.time_chunks * 10, 1000)
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(self.path)},
            'metadata': {
                'dtype': '<f4' if self.dtype == np.float32 else '<f8',
                'shape': [initial_cap, self.n_neurons, self.n_features],
                'chunks': [self.time_chunks, self.n_neurons, self.n_features],
                'compressor': {
                    'id': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 2,
                },
            },
            'create': True,
            'delete_existing': True,
        }
        self._store = ts.open(spec).result()
        # Force the directory to be visible on distributed filesystems (GPFS/NFS)
        # before the first chunk write does its lock→rename sequence.
        self.path.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    def append(self, frame: np.ndarray):
        if frame.shape != (self.n_neurons, self.n_features):
            raise ValueError(
                f"frame shape {frame.shape} doesn't match expected "
                f"({self.n_neurons}, {self.n_features})"
            )
        self._buffer.append(frame.astype(self.dtype, copy=False))
        if len(self._buffer) >= self.time_chunks:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        if not self._initialized:
            self._initialize_store()

        data = np.stack(self._buffer, axis=0)
        n_frames = data.shape[0]

        needed = self._total_frames + n_frames
        if needed > self._store.shape[0]:
            new_size = max(needed, self._store.shape[0] * 2)
            self._store = self._store.resize(
                exclusive_max=[new_size, self.n_neurons, self.n_features]
            ).result()

        # Retry on NOT_FOUND: distributed filesystems (GPFS/NFS) occasionally
        # fail the lock→rename with ENOENT if the directory entry isn't yet
        # visible to all nodes. Re-stating the mkdir and retrying fixes it.
        import time
        for attempt in range(3):
            try:
                self._store[self._total_frames:self._total_frames + n_frames].write(data).result()
                break
            except (ValueError, Exception) as e:
                if attempt < 2 and ('NOT_FOUND' in str(e) or 'ENOENT' in str(e)):
                    self.path.mkdir(parents=True, exist_ok=True)
                    time.sleep(1.0 * (attempt + 1))
                else:
                    raise
        self._total_frames += n_frames
        self._buffer.clear()

    def finalize(self):
        self._flush()
        if self._store is not None and self._total_frames > 0:
            self._store = self._store.resize(
                exclusive_max=[self._total_frames, self.n_neurons, self.n_features]
            ).result()
        return self._total_frames


_DYNAMIC_FIELDS = ['voltage', 'stimulus', 'calcium', 'fluorescence', 'noise']
_STATIC_FIELDS = ['pos', 'group_type', 'neuron_type']


class ZarrSimulationWriterV3:
    """Per-field zarr writer — each NeuronState field gets its own zarr array.

    Storage structure:
        path/
            pos.zarr          # (N, 2) float32 — static
            group_type.zarr   # (N,) int32 — static
            neuron_type.zarr  # (N,) int32 — static
            voltage.zarr      # (T, N) float32 — dynamic
            stimulus.zarr     # (T, N) float32 — dynamic
            calcium.zarr      # (T, N) float32 — dynamic
            fluorescence.zarr # (T, N) float32 — dynamic

    Note: index is NOT saved — it is arange(n_neurons) and constructed at load time.

    Usage:
        writer = ZarrSimulationWriterV3(path, n_neurons=14011)
        for state in simulation:
            writer.append_state(state)
        writer.finalize()
    """

    def __init__(
        self,
        path: str | Path,
        n_neurons: int,
        time_chunks: int = 2000,
        save_calcium: bool = False,
        extra_dynamic_fields: list[str] | None = None,
    ):
        self.path = Path(path)
        self.n_neurons = n_neurons
        self.time_chunks = time_chunks
        self.save_calcium = save_calcium

        extras = list(extra_dynamic_fields or [])
        base = [f for f in _DYNAMIC_FIELDS
                if save_calcium or f not in ('calcium', 'fluorescence')]
        # extras append after the canonical fields, dedup-preserving
        self._fields = base + [e for e in extras if e not in base]
        self._static_saved = False
        self._buffers: dict[str, list[np.ndarray]] = {f: [] for f in self._fields}
        self._stores: dict[str, ts.TensorStore] = {}
        self._total_frames = 0
        self._dynamic_initialized = False

    def _save_static(self, state: NeuronState):
        """Save static fields from first NeuronState frame."""
        from connectome_gnn.utils import to_numpy

        self.path.mkdir(parents=True, exist_ok=True)

        static_data = {
            'pos': to_numpy(state.pos).astype(np.float32),
            'group_type': to_numpy(state.group_type).astype(np.int32),
            'neuron_type': to_numpy(state.neuron_type).astype(np.int32),
        }

        for name, data in static_data.items():
            zarr_path = self.path / f'{name}.zarr'
            if zarr_path.exists():
                import shutil
                shutil.rmtree(zarr_path, ignore_errors=True)

            dtype_str = '<i4' if data.dtype in (np.int32, np.int64) else '<f4'
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': str(zarr_path)},
                'metadata': {
                    'dtype': dtype_str,
                    'shape': list(data.shape),
                    'chunks': list(data.shape),
                    'compressor': {
                        'id': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 2,
                    },
                },
                'create': True,
                'delete_existing': True,
            }
            store = ts.open(spec).result()
            store.write(data).result()

        self._static_saved = True

    def _initialize_dynamic_stores(self):
        """Create zarr stores for dynamic fields."""
        initial_cap = max(self.time_chunks * 10, 1000)

        for name in self._fields:
            zarr_path = self.path / f'{name}.zarr'
            if zarr_path.exists():
                import shutil
                shutil.rmtree(zarr_path, ignore_errors=True)

            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': str(zarr_path)},
                'metadata': {
                    'dtype': '<f4',
                    'shape': [initial_cap, self.n_neurons],
                    'chunks': [self.time_chunks, self.n_neurons],
                    'compressor': {
                        'id': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 2,
                    },
                },
                'create': True,
                'delete_existing': True,
            }
            store = ts.open(spec).result()
            # Force directory visibility on distributed filesystems before first write.
            (self.path / f'{name}.zarr').mkdir(parents=True, exist_ok=True)
            self._stores[name] = store

        self._dynamic_initialized = True

    def append_state(self, state: NeuronState):
        """Append one frame from NeuronState."""
        from connectome_gnn.utils import to_numpy

        if not self._static_saved:
            self._save_static(state)

        self._buffers['voltage'].append(to_numpy(state.voltage).astype(np.float32))
        self._buffers['stimulus'].append(to_numpy(state.stimulus).astype(np.float32))
        if self.save_calcium:
            self._buffers['calcium'].append(to_numpy(state.calcium).astype(np.float32))
            self._buffers['fluorescence'].append(to_numpy(state.fluorescence).astype(np.float32))
        noise_val = getattr(state, 'noise', None)
        self._buffers['noise'].append(
            to_numpy(noise_val).astype(np.float32) if noise_val is not None
            else np.zeros(self.n_neurons, dtype=np.float32)
        )
        # Any field beyond the canonical {voltage,stimulus,calcium,fluorescence,noise}
        # was added via extra_dynamic_fields; pull it from the state by name.
        for name in self._fields:
            if name in ('voltage', 'stimulus', 'calcium', 'fluorescence', 'noise'):
                continue
            val = getattr(state, name, None)
            self._buffers[name].append(
                to_numpy(val).astype(np.float32) if val is not None
                else np.zeros(self.n_neurons, dtype=np.float32)
            )

        if len(self._buffers['voltage']) >= self.time_chunks:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered dynamic data to zarr stores."""
        if not self._buffers['voltage']:
            return

        if not self._dynamic_initialized:
            self._initialize_dynamic_stores()

        n_frames = len(self._buffers['voltage'])

        for name in self._fields:
            data = np.stack(self._buffers[name], axis=0)  # (chunk, N)

            # resize if needed
            current_shape = self._stores[name].shape
            needed = self._total_frames + n_frames
            if needed > current_shape[0]:
                new_size = max(needed, current_shape[0] * 2)
                self._stores[name] = self._stores[name].resize(
                    exclusive_max=[new_size, self.n_neurons]
                ).result()

            import time
            for attempt in range(3):
                try:
                    self._stores[name][self._total_frames:self._total_frames + n_frames].write(data).result()
                    break
                except (ValueError, Exception) as e:
                    if attempt < 2 and ('NOT_FOUND' in str(e) or 'ENOENT' in str(e)):
                        (self.path / f'{name}.zarr').mkdir(parents=True, exist_ok=True)
                        time.sleep(1.0 * (attempt + 1))
                    else:
                        raise
            self._buffers[name].clear()

        self._total_frames += n_frames

    def finalize(self):
        """Flush remaining buffer and resize stores to exact size."""
        self._flush_buffer()

        for name in self._fields:
            if name in self._stores and self._total_frames > 0:
                self._stores[name] = self._stores[name].resize(
                    exclusive_max=[self._total_frames, self.n_neurons]
                ).result()

        return self._total_frames


class ZarrTaskTrialsWriter:
    """Streaming per-trial writer for task data — TaskState analogue of
    :class:`ZarrSimulationWriterV3`.

    The atom is a ``TaskState`` (one trial, T-major); the aggregate is a
    ``TaskTrials`` (B-major). Each populated tensor field on the first
    appended ``TaskState`` becomes a separate zarr array; missing fields
    are not written.

    Storage structure (path/):
        meta.json               # task_family, n_input, n_output, dt, rules, ...
        stimulus.zarr           # (B, T, N_i)
        target.zarr             # (B, T, N_o)
        theta_hd.zarr           # (B, T, 1)
        is_stop.zarr            # (B, T, 1)
        omega.zarr              # (B, T, 1)
        c_mask.zarr             # (B, T, N_o)         — cortex only
        stimulus_canonical.zarr # (B, T, N_i)         — cortex only
        delta_stimulus.zarr     # (B, T, N_i)         — cortex only
        rule_idx.zarr           # (B, 1, 1)           — cortex only

    Usage:
        writer = ZarrTaskTrialsWriter(path, chunk_trials=1000)
        for b in range(B):
            writer.append_trial(state_b)   # state_b: TaskState
        writer.finalize()

    The on-disk layout matches what ``task_trials_to_disk`` produced before
    the streaming writer existed, so existing readers (legacy
    ``load_raw_array`` and ``TaskTrials.from_disk``) keep working unchanged.
    """

    # 1-D-along-T fields stored as (B, T, 1) on disk to keep the same chunking
    # heuristics as the (B, T, n_feat) fields. The reader squeezes them back
    # to (B, T).
    _PROMOTE_TO_3D = ('theta_hd', 'is_stop', 'omega', 'length')

    def __init__(
        self,
        path,
        chunk_trials: int = 1000,
    ):
        self.path = Path(path)
        self.chunk_trials = int(chunk_trials)

        # Schema discovered from the first appended TaskState.
        self._tensor_fields: list[str] | None = None
        self._field_per_trial_shape: dict[str, tuple[int, ...]] = {}
        self._field_is_promoted: dict[str, bool] = {}

        self._buffers: dict[str, list[np.ndarray]] = {}
        self._stores: dict[str, ts.TensorStore] = {}
        self._epochs_buffer: list = []                  # cortex: per-trial dicts
        self._meta: dict = {}                           # static fields
        self._n_trials_written = 0
        self._dynamic_initialized = False

    # ------------------------------------------------------------------
    # Schema discovery
    # ------------------------------------------------------------------
    def _discover_schema(self, state):
        """Capture static metadata + per-field shapes from the first trial."""
        import torch

        # Static (scalar / string) metadata.
        for k in ('task_family', 'n_input', 'n_output', 'dt'):
            v = getattr(state, k, None)
            if v is not None:
                self._meta[k] = v

        # Tensor fields: keep only those that are populated.
        tensor_fields: list[str] = []
        for fname in ('stimulus', 'target', 'length',
                      'c_mask', 'stimulus_canonical', 'delta_stimulus',
                      'theta_hd', 'is_stop', 'omega'):
            val = getattr(state, fname, None)
            if isinstance(val, torch.Tensor):
                arr = val.detach().cpu().numpy()
                shape = tuple(arr.shape)
                if fname in self._PROMOTE_TO_3D and len(shape) == 1:
                    self._field_is_promoted[fname] = True
                    shape = shape + (1,)
                else:
                    self._field_is_promoted[fname] = False
                self._field_per_trial_shape[fname] = shape
                tensor_fields.append(fname)
                self._buffers[fname] = []

        # rule_idx is a per-trial scalar (cortex multitask).
        if getattr(state, 'rule_idx', None) is not None:
            self._field_per_trial_shape['rule_idx'] = (1, 1)
            self._field_is_promoted['rule_idx'] = True
            self._buffers['rule_idx'] = []
            tensor_fields.append('rule_idx')

        self._tensor_fields = tensor_fields

    # ------------------------------------------------------------------
    # Per-field zarr store init
    # ------------------------------------------------------------------
    def _initialize_stores(self):
        initial_cap = max(self.chunk_trials * 4, 1000)
        self.path.mkdir(parents=True, exist_ok=True)
        for name in self._tensor_fields:
            per = self._field_per_trial_shape[name]
            shape = (initial_cap,) + per
            chunks = (min(self.chunk_trials, initial_cap),) + per
            zarr_path = self.path / f'{name}.zarr'
            if zarr_path.exists():
                import shutil
                shutil.rmtree(zarr_path, ignore_errors=True)
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': str(zarr_path)},
                'metadata': {
                    'dtype': '<f4',
                    'shape': list(shape),
                    'chunks': list(chunks),
                    'compressor': {
                        'id': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 2,
                    },
                },
                'create': True,
                'delete_existing': True,
            }
            self._stores[name] = ts.open(spec).result()
        self._dynamic_initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def append_trial(self, state):
        """Append one ``TaskState`` to the in-memory buffer; flush when
        the buffer hits ``chunk_trials``."""
        import torch

        if self._tensor_fields is None:
            self._discover_schema(state)
            if not self._tensor_fields:
                raise ValueError(
                    "TaskState has no populated tensor fields — nothing to write"
                )

        for name in self._tensor_fields:
            if name == 'rule_idx':
                v = int(getattr(state, 'rule_idx'))
                arr = np.array([[v]], dtype=np.float32)        # (1, 1)
            else:
                val = getattr(state, name)
                arr = val.detach().cpu().numpy().astype(np.float32)
                if self._field_is_promoted[name]:
                    arr = arr[..., None]                       # (T,) -> (T, 1)
                # shape sanity-check
                exp = self._field_per_trial_shape[name]
                if tuple(arr.shape) != exp:
                    raise ValueError(
                        f"field '{name}' shape mismatch: got {tuple(arr.shape)}, "
                        f"first trial had {exp}"
                    )
            self._buffers[name].append(arr)

        # Cortex: stash per-trial epoch dicts in a JSON-friendly buffer.
        ep = getattr(state, 'epochs', None)
        if ep is not None:
            self._epochs_buffer.append(ep)

        if len(self._buffers[self._tensor_fields[0]]) >= self.chunk_trials:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self._tensor_fields:
            return
        n = len(self._buffers[self._tensor_fields[0]])
        if n == 0:
            return
        if not self._dynamic_initialized:
            self._initialize_stores()

        for name in self._tensor_fields:
            data = np.stack(self._buffers[name], axis=0)       # (n, *per)
            current_shape = self._stores[name].shape
            needed = self._n_trials_written + n
            if needed > current_shape[0]:
                new_cap = max(needed, current_shape[0] * 2)
                self._stores[name] = self._stores[name].resize(
                    exclusive_max=[new_cap, *current_shape[1:]]
                ).result()
            self._stores[name][self._n_trials_written:self._n_trials_written + n].write(data).result()
            self._buffers[name].clear()

        self._n_trials_written += n

    def finalize(self) -> int:
        """Flush, trim stores to exact size, and write meta.json sidecar."""
        import json

        self._flush_buffer()

        # Trim every store to the exact (n_trials, *per_trial_shape).
        for name in self._tensor_fields or ():
            if name not in self._stores or self._n_trials_written == 0:
                continue
            per = self._field_per_trial_shape[name]
            self._stores[name] = self._stores[name].resize(
                exclusive_max=[self._n_trials_written, *per]
            ).result()

        # Persist static metadata + counts.
        meta = dict(self._meta)
        meta['n_trials'] = int(self._n_trials_written)
        if 'stimulus' in self._field_per_trial_shape:
            meta['n_frames'] = int(self._field_per_trial_shape['stimulus'][0])
        if self._epochs_buffer:
            meta['epochs'] = self._epochs_buffer
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        return self._n_trials_written


def detect_format(path: str | Path) -> Literal['npy', 'zarr_v3', 'none']:
    """check what format exists at path.

    args:
        path: base path without extension

    returns:
        'zarr_v3' if V3 zarr directory exists (per-field .zarr arrays)
        'npy' if .npy file exists
        'none' if nothing exists
    """
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    # check for V3 zarr format (directory with per-field .zarr arrays)
    if base_path.exists() and base_path.is_dir():
        if (base_path / 'voltage.zarr').exists():
            return 'zarr_v3'

    # check for npy
    npy_path = Path(str(base_path) + '.npy')
    if npy_path.exists():
        return 'npy'

    return 'none'


def load_simulation_data(path: str | Path, fields=None) -> NeuronTimeSeries:
    """load simulation data as NeuronTimeSeries (V3 zarr or npy).

    args:
        path: base path (with or without extension)
        fields: list of field names to load (V3 only, e.g. ['voltage', 'stimulus']).
                None = all fields.

    returns:
        NeuronTimeSeries with requested fields (others are None)

    raises:
        FileNotFoundError: if no data found at path
    """
    from connectome_gnn.neuron_state import NeuronTimeSeries
    return NeuronTimeSeries.load(path, fields=fields)


def load_raw_array(path: str | Path) -> np.ndarray:
    """load a raw numpy array from .zarr or .npy (for y_list derivative targets etc.).

    args:
        path: base path (with or without extension)

    returns:
        numpy array

    raises:
        FileNotFoundError: if no data found at path
    """
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    # try zarr (single array)
    zarr_path = Path(str(base_path) + '.zarr')
    if zarr_path.exists() and zarr_path.is_dir():
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(zarr_path)},
        }
        return ts.open(spec).result().read().result()

    # try npy
    npy_path = Path(str(base_path) + '.npy')
    if npy_path.exists():
        return np.load(npy_path)

    raise FileNotFoundError(f"no .zarr or .npy found at {base_path}")
