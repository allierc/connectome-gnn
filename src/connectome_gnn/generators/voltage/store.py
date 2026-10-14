"""DatasetStore: where a voltage dataset lives, what each array carries, and legacy erase.

Every path under ``graphs_data/<dataset>/`` that generation writes goes through
this object, so the layout is stated once (ARRAY_PROVENANCE) instead of being
spread over 40 ``graphs_data_path(config.dataset, ...)`` calls.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

from connectome_gnn.generators.utils import rmtree_robust
from connectome_gnn.log import get_logger
from connectome_gnn.utils import graphs_data_path
from connectome_gnn.zarr_io import ZarrArrayWriter, ZarrSimulationWriterV3

logger = get_logger(__name__)

# WHICH ARRAY CARRIES WHICH NOISE. xi = process noise (noise_model_level), added
# to v after frame t is written; eta = measurement noise (measurement_noise_level).
# Frame t of a split, as integrate.FrameRecord names it.
ARRAY_PROVENANCE = {
    "x_list_{split}/voltage.zarr": "v[t], written BEFORE the step of frame t: carries xi[0..t-1], never eta",
    "x_list_{split}/stimulus.zarr": "the input of frame t (on CPU the state aliases net.stimulus.buffer)",
    "x_list_{split}/noise.zarr": "eta[t]; not added to any stored voltage (zeros when off)",
    "x_list_{split}/calcium.zarr": "only with save_calcium",
    "x_list_{split}/fluorescence.zarr": "only with save_calcium",
    "x_list_{split}/pos.zarr, group_type.zarr, neuron_type.zarr": "static, from the first frame",
    "y_list_{split}.zarr": "f(v[t]), the ODE right-hand side at the written v[t]: neither xi nor eta",
    "noisy_y_list_{split}.zarr": "y + (eta[t+1] - eta[t]) / dt, optionally smoothed: eta, NOT xi/dt",
}

TIME_CHUNKS = 2000


@dataclass(frozen=True)
class DatasetStore:
    """Paths of one dataset under graphs_data/, and the writers of its splits."""

    dataset: str

    def path(self, *parts: str) -> str:
        return graphs_data_path(self.dataset, *parts)

    @property
    def folder(self) -> str:
        """The dataset directory, with the trailing slash legacy passes around."""
        return graphs_data_path(self.dataset) + "/"

    def erase_legacy(self) -> None:
        """Delete ``x_list_{train,test}`` and ``y_list_{train,test}`` if present.

        QUIRK (bug-for-bug): THE CHECK IS ON THE NAME WITHOUT ``.zarr``. The
        x_list_* directories match, the y_list_*.zarr stores never do, and
        nothing else is looked at, so y_list_*.zarr, noisy_y_list_*.zarr,
        weights_full.pt, kept_edge_indices.pt, ablation_mask.pt,
        noisy_test_data.ok, BRACKET_*.txt and top-level PNGs all survive an erase.
        The zarr writers replace the stores they write, so what is left stale is
        exactly what the new run does not write, and ``tile_train`` tiles a stale
        noisy_y_list_train.zarr (golden cells dirty_erase, dirty_noerase_tile).
        Fixing it is a v2 change.
        """
        for split in ['train', 'test']:
            for data_file in ['x_list', 'y_list']:
                old_path = self.path(f"{data_file}_{split}")
                if os.path.exists(old_path):
                    rmtree_robust(old_path)
                    logger.info(f"erased old {data_file}_{split}")

    def make_folders(self) -> None:
        """Create the dataset directory and an empty ``Fig/``.

        ``graphs_data/fly`` is created whatever the dataset is called (legacy).
        """
        os.makedirs(graphs_data_path("fly"), exist_ok=True)
        print(f"\033[93m[data folder] {self.folder}\033[0m", flush=True)
        os.makedirs(self.folder, exist_ok=True)
        os.makedirs(self.path("Fig"), exist_ok=True)
        self.clear_figs()

    def clear_figs(self) -> None:
        """Remove every file in ``Fig/`` (the per-frame figures)."""
        for f in glob.glob(self.path("Fig", "*")):
            os.remove(f)

    def split_writer(self, split: str, n_neurons: int, save_calcium, to_numpy_fn) -> "SplitWriter":
        """The writer of x_list_<split>/ and y_list_<split>.zarr; nothing touches disk before the first frame."""
        x_writer = ZarrSimulationWriterV3(
            path=self.path(f"x_list_{split}"),
            n_neurons=n_neurons,
            time_chunks=TIME_CHUNKS,
            # ce0d1d9 ("finish calcium strip") removed sim.save_calcium along with
            # the 9 other calcium fields but left these call sites, so flyvis
            # voltage generation has raised AttributeError since 2026-08-03.
            # Defaulting False keeps the writer plumbing intact: making calcium
            # generation actually WORK needs calcium_tau/alpha/beta back too, which
            # is separate work and not needed for the conductance-vs-current survey.
            save_calcium=save_calcium,
        )
        y_writer = ZarrArrayWriter(
            path=self.path(f"y_list_{split}"),
            n_neurons=n_neurons,
            n_features=1,
            time_chunks=TIME_CHUNKS,
        )
        return SplitWriter(x_writer, y_writer, to_numpy_fn)


class _FrameView:
    """What ZarrSimulationWriterV3.append_state reads, with the dynamic fields taken from a FrameRecord."""

    __slots__ = ("pos", "group_type", "neuron_type", "voltage", "stimulus", "noise", "calcium", "fluorescence")

    def __init__(self, x, record):
        self.pos, self.group_type, self.neuron_type = x.pos, x.group_type, x.neuron_type
        self.calcium, self.fluorescence = x.calcium, x.fluorescence
        self.voltage = record.voltage
        self.stimulus = record.stimulus
        self.noise = record.measurement_noise


class SplitWriter:
    """Writes one split frame by frame from integrate.FrameRecord values (see ARRAY_PROVENANCE).

    Two calls per frame, in legacy order: ``write_state`` BEFORE the state
    update (voltage, stimulus and measurement noise of frame t; static fields
    and, with save_calcium, calcium and fluorescence from x), ``write_drift``
    after it (the drift of frame t, copied once, as legacy did).
    """

    def __init__(self, x_writer, y_writer, to_numpy_fn):
        self._x = x_writer
        self._y = y_writer
        self._to_numpy = to_numpy_fn

    def write_state(self, x, record) -> None:
        self._x.append_state(_FrameView(x, record))

    def write_drift(self, record) -> None:
        self._y.append(self._to_numpy(record.drift.clone().detach()))

    def finalize(self) -> int:
        """Flush both stores; the number of frames written."""
        n_frames = self._x.finalize()
        self._y.finalize()
        return n_frames
