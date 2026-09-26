"""Stimulus datasets: which videos are rendered, and the only ways generation indexes them.

INDEXING A DATASET IS PART OF THE RNG CONTRACT. flyvis datasets may draw
from the global torch / numpy streams in ``__getitem__`` when augmentation is
on (in the configs this function builds it is off: DAVIS gets
``augment=False`` and Sintel's jitter / noise / gamma stds are None, so no
access draws today; the RNG ledger checks that). So StimulusSource exposes
exactly the accesses legacy makes, in legacy order:

    item(0)                     init_state: the frame the network state starts from
    items(train), items(test)   materialize_sequences: ALL train sequences, before
                                the max_train_sequences truncation
    item(0)                     materialize_sequences: the hexal count
    iter(davis)                 the frame loop, mixed mode only (see QUIRKS)
"""

from __future__ import annotations

import os
from typing import Any

from connectome_gnn.log import get_logger
from connectome_gnn.utils import get_datavis_root_dir

logger = get_logger(__name__)


class StimulusSource:
    """The dataset the splits are drawn from, plus the DAVIS dataset mixed mode walks."""

    def __init__(self, stimulus_dataset: Any, davis_dataset: Any):
        self._dataset = stimulus_dataset
        self._davis = davis_dataset

    @property
    def arg_df(self):
        """One row per sequence (name, original_index, flip_ax, n_rot, ...), aligned with the items."""
        return self._dataset.arg_df

    def item(self, i: int) -> dict:
        return self._dataset[i]

    def items(self, indices) -> list:
        return [self._dataset[i] for i in indices]

    @property
    def davis_dataset(self):
        """The DAVIS dataset (None without DAVIS/mixed input); the frame loop iterates it in mixed mode."""
        return self._davis


def build_sources(spec, boxfilter: dict) -> StimulusSource:
    """Build the DAVIS dataset (DAVIS or mixed input) and pick the dataset the splits use.

    RNG: a cold rendering cache builds a flyvis BoxEye, whose Conv2d init draws
    torch RNG (see QUIRKS), and AugmentedVideoDataset reseeds stdlib ``random``
    with ``spec.seed`` when it shuffles its sequences.
    """
    st = spec.stimulus
    # Initialize datasets
    print(f"[DBG] visual_input_type={st.visual_input_type!r}  datavis_roots={list(st.datavis_roots)}", flush=True)
    if "DAVIS" in st.visual_input_type or "mixed" in st.visual_input_type:
        from connectome_gnn.generators.davis import AugmentedVideoDataset, CombinedVideoDataset

        # determine dataset roots: use config list if provided, otherwise fall back to default
        if st.datavis_roots:
            # Cluster paths may use environment variables; expand at use.
            datavis_root_list = [os.path.join(os.path.expandvars(r), "JPEGImages/480p")
                                 for r in st.datavis_roots]
        else:
            datavis_root_list = [os.path.join(get_datavis_root_dir(), "JPEGImages/480p")]

        print(f"[DBG] datavis_root_list={datavis_root_list}", flush=True)
        for root in datavis_root_list:
            print(f"[DBG] checking root exists: {root}", flush=True)
            assert os.path.exists(root), f"video data not found at {root}"
            print("[DBG]   OK exists", flush=True)

        video_config = {
            "n_frames": 50,
            "max_frames": st.truncate_max_frames,  # None = no per-clip truncation (never reaches RenderedDavis)
            # Hex rotate/flip rely on a regular hex-disk lattice; the FlyWire
            # column lattice is irregular, so disable them when rendering on it.
            # (augment=False already neutralises them at runtime, but this also
            # keeps the construction path safe for any future augment toggle.)
            # HexFlip/HexRotate operate on the standard hex lattice the
            # frames are rendered on (BoxEye extent above). For FlyWire
            # mode we render at a standard disk and project later, so
            # the same 8x augmentation factor applies to both modes.
            "flip_axes": [0, 1],
            "n_rotations": [0, 90, 180, 270],
            "temporal_split": False,
            "dt": spec.delta_t,
            "boxfilter": boxfilter,
            "vertical_splits": 1,
            "center_crop_fraction": 0.6,
            "augment": False,
            "unittest": False,
            "skip_short_videos": st.skip_short_videos,
            "shuffle_sequences": True,
            "shuffle_seed": spec.seed,
        }
        print(f"[DBG] video_config built (skip_short={st.skip_short_videos} max_frames={st.truncate_max_frames} "
              f"seed={spec.seed})", flush=True)

        # create dataset(s)
        if len(datavis_root_list) == 1:
            print(f"[DBG] creating AugmentedVideoDataset(root_dir={datavis_root_list[0]}) ...", flush=True)
            davis_dataset = AugmentedVideoDataset(root_dir=datavis_root_list[0], **video_config)
            print(f"[DBG] AugmentedVideoDataset ready: {len(davis_dataset)} sequences", flush=True)
        else:
            print(f"[DBG] creating {len(datavis_root_list)} AugmentedVideoDatasets (combined) ...", flush=True)
            datasets = [AugmentedVideoDataset(root_dir=root, **video_config) for root in datavis_root_list]
            davis_dataset = CombinedVideoDataset(datasets)
            logger.info(f"combined {len(datasets)} video datasets: {len(davis_dataset)} total sequences")
    else:
        davis_dataset = None

    if "DAVIS" in st.visual_input_type:
        stimulus_dataset = davis_dataset
        print(f"[DBG] using DAVIS-branch dataset: {len(stimulus_dataset)} sequences", flush=True)
    else:
        from flyvis.datasets.sintel import AugmentedSintel

        sintel_config = {
            "n_frames": 19,
            "flip_axes": [0, 1],
            "n_rotations": [0, 1, 2, 3, 4, 5],
            "temporal_split": True,
            "dt": spec.delta_t,
            "interpolate": True,
            "boxfilter": boxfilter,
            "vertical_splits": 3,
            "center_crop_fraction": 0.7,
        }
        print("[DBG] creating AugmentedSintel(...) ...", flush=True)
        stimulus_dataset = AugmentedSintel(**sintel_config)
        print(f"[DBG] AugmentedSintel ready: {len(stimulus_dataset)} sequences", flush=True)
    return StimulusSource(stimulus_dataset, davis_dataset)
