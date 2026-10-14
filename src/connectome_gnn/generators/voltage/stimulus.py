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
from dataclasses import dataclass
from typing import Any

import numpy as np

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


# Frames per sequence as legacy counts them to size the train passes and the
# test target (the sequences themselves are longer or shorter).
FRAMES_PER_SEQUENCE = 35


@dataclass(frozen=True)
class VideoSplit:
    """Which sequences go to train and which to test, split by source video."""

    train_indices: list
    test_indices: list
    train_video_names: list        # sorted
    test_video_names: list
    n_train_vids: int
    n_test_vids: int               # len(test_video_set)


@dataclass(frozen=True)
class Sequences:
    """The sequences each split runs over, materialised from the dataset."""

    train: list                    # truncated to max_train_sequences when that is > 0
    test: list                     # truncated to max(1, max_train_sequences // 4)
    train_meta: list               # (name, flip_ax, n_rot) of EVERY train index, not only the kept ones
    test_meta: list
    n_hexals: int


def split_videos(stimuli: StimulusSource) -> VideoSplit:
    """80/20 split of the source videos, so all augmentations of one video stay together.

    Draws from the GLOBAL numpy stream (``np.random.shuffle`` of the unique
    video indices), seeded by ``seed``.
    """
    # --- Subdirectory-level train/test split ---
    # arg_df is aligned with cached_sequences (shuffle applied to both in _build).
    # Split by original_index so all augmentations of the same base video stay together.
    df = stimuli.arg_df
    original_indices = df["original_index"].values
    unique_videos = np.unique(original_indices)
    np.random.shuffle(unique_videos)
    n_train_vids = int(len(unique_videos) * 0.8)
    train_video_set = set(unique_videos[:n_train_vids])
    test_video_set = set(unique_videos[n_train_vids:])

    train_indices = [i for i, oi in enumerate(original_indices) if oi in train_video_set]
    test_indices = [i for i, oi in enumerate(original_indices) if oi in test_video_set]

    # Extract the actual video subdirectory names for logging
    train_video_names = sorted(set(df.iloc[train_indices]["name"].values))
    test_video_names = sorted(set(df.iloc[test_indices]["name"].values))

    # Verify exclusivity
    train_name_set = set(train_video_names)
    test_name_set = set(test_video_names)
    overlap = train_name_set & test_name_set
    assert len(overlap) == 0, f"TRAIN/TEST OVERLAP: {overlap}"
    logger.info(
        f"subdirectory split: {n_train_vids} train / {len(unique_videos) - n_train_vids} test videos"
        f"  ({len(train_indices)} train seqs, {len(test_indices)} test seqs)"
    )
    logger.info(f"overlap: {overlap} (must be empty)")
    return VideoSplit(train_indices=train_indices, test_indices=test_indices,
                      train_video_names=train_video_names, test_video_names=test_video_names,
                      n_train_vids=n_train_vids, n_test_vids=len(test_video_set))


def materialize_sequences(spec, stimuli: StimulusSource, split: VideoSplit) -> Sequences:
    """Index every train and test sequence (ALL of them, before truncation), then truncate.

    The number and order of dataset accesses is part of the RNG contract (see
    the module docstring): all train items, all test items, then item(0) again
    for the hexal count.
    """
    # Build sequences lists for ODE generation
    train_sequences = stimuli.items(split.train_indices)
    test_sequences = stimuli.items(split.test_indices)

    # Optionally limit number of sequences for faster debugging
    max_train = spec.stimulus.max_train_sequences
    if max_train > 0:
        train_sequences = train_sequences[: max_train]
        test_sequences = test_sequences[: max(1, max_train // 4)]
        logger.info(
            f"max_train_sequences={max_train}: using {len(train_sequences)} train, {len(test_sequences)} test sequences"
        )

    # Build metadata labels for preview plots (name, flip_ax, n_rot)
    df = stimuli.arg_df
    train_meta = [(df.iloc[idx]["name"], df.iloc[idx]["flip_ax"], df.iloc[idx]["n_rot"]) for idx in split.train_indices]
    test_meta = [(df.iloc[idx]["name"], df.iloc[idx]["flip_ax"], df.iloc[idx]["n_rot"]) for idx in split.test_indices]

    n_hexals = stimuli.item(0)["lum"].shape[-1]
    return Sequences(train=train_sequences, test=test_sequences, train_meta=train_meta, test_meta=test_meta,
                     n_hexals=n_hexals)
