"""Figures of a generated dataset: stimulus previews, traces, per-type panels, the input video."""

from __future__ import annotations

import os

from connectome_gnn.log import get_logger
from connectome_gnn.plot import plot_kinograph, plot_sequence_preview

logger = get_logger(__name__)


def plot_previews(sequences, split, geometry, folder: str, fig_style) -> None:
    """shuffle_first_frames_{train,test}.png: the first frames of every sequence of each split."""
    # Plot preview for train and test splits
    hex_x = geometry.x_coords[:sequences.n_hexals]
    hex_y = geometry.y_coords[:sequences.n_hexals]
    plot_sequence_preview(
        sequences.train,
        hex_x,
        hex_y,
        f"TRAIN: {len(sequences.train)} seqs from {split.n_train_vids} videos",
        os.path.join(folder, "shuffle_first_frames_train.png"),
        fig_style,
        metadata=sequences.train_meta,
        logger=logger,
    )
    plot_sequence_preview(
        sequences.test,
        hex_x,
        hex_y,
        f"TEST: {len(sequences.test)} seqs from {split.n_test_vids} videos",
        os.path.join(folder, "shuffle_first_frames_test.png"),
        fig_style,
        metadata=sequences.test_meta,
        logger=logger,
    )


def plot_kinograph_figure(spec, trace, ranks, act_labels, stim_labels, store, fig_style) -> None:
    """kinograph.png: activity and photoreceptor input over time, annotated with the effective ranks."""
    logger.info("plotting kinograph ...")
    plot_kinograph(
        activity=trace.activity.T,
        stimulus=trace.x_ts.stimulus[:, : spec.network.n_input_neurons].numpy().T,
        output_path=store.path("kinograph.png"),
        rank_90_act=ranks.rank_90_act,
        rank_99_act=ranks.rank_99_act,
        rank_90_inp=ranks.rank_90_inp,
        rank_99_inp=ranks.rank_99_inp,
        rank_90_mc=ranks.rank_90_mc,
        rank_99_mc=ranks.rank_99_mc,
        zoom_size=200,
        style=fig_style,
        act_labels=act_labels,
        stim_labels=stim_labels,
    )
