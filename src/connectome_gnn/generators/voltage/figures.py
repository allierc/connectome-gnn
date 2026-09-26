"""Figures of a generated dataset: stimulus previews, traces, per-type panels, the input video."""

from __future__ import annotations

import os

from connectome_gnn.generators.utils import generate_compressed_video_mp4
from connectome_gnn.log import get_logger
from connectome_gnn.plot import plot_kinograph, plot_sequence_preview
from connectome_gnn.utils import to_numpy

logger = get_logger(__name__)

# Length of the window shown in <dataset>/activity.png, in simulation FRAMES.
# 1,000 frames is 20 s of simulated time at the flyvis delta_t of 20 ms, and it
# is the same window the trainer's rollout figures use (teacher_eval's
# n_frames default), so a dataset's activity.png and that run's
# tmp_training/traces/rollout_*.png can be laid side by side.
ACTIVITY_TRACE_FRAMES = 1000

# Neuron type index to name mapping (CamelCase for legacy plot_neuron_activity_analysis)
INDEX_TO_NAME = {
    0: "Am",
    1: "C2",
    2: "C3",
    3: "CT1(Lo1)",
    4: "CT1(M10)",
    5: "L1",
    6: "L2",
    7: "L3",
    8: "L4",
    9: "L5",
    10: "Lawf1",
    11: "Lawf2",
    12: "Mi1",
    13: "Mi10",
    14: "Mi11",
    15: "Mi12",
    16: "Mi13",
    17: "Mi14",
    18: "Mi15",
    19: "Mi2",
    20: "Mi3",
    21: "Mi4",
    22: "Mi9",
    23: "R1",
    24: "R2",
    25: "R3",
    26: "R4",
    27: "R5",
    28: "R6",
    29: "R7",
    30: "R8",
    31: "T1",
    32: "T2",
    33: "T2a",
    34: "T3",
    35: "T4a",
    36: "T4b",
    37: "T4c",
    38: "T4d",
    39: "T5a",
    40: "T5b",
    41: "T5c",
    42: "T5d",
    43: "Tm1",
    44: "Tm16",
    45: "Tm2",
    46: "Tm20",
    47: "Tm28",
    48: "Tm3",
    49: "Tm30",
    50: "Tm4",
    51: "Tm5Y",
    52: "Tm5a",
    53: "Tm5b",
    54: "Tm5c",
    55: "Tm9",
    56: "TmY10",
    57: "TmY13",
    58: "TmY14",
    59: "TmY15",
    60: "TmY18",
    61: "TmY3",
    62: "TmY4",
    63: "TmY5a",
    64: "TmY9",
}

# The ten types activity_selected.png shows, by type index.
CURATED_TYPES = [55, 15, 43, 39, 35, 31, 23, 19, 12, 5]


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


def render_figures(spec, trace, x, n_neurons: int, store) -> None:
    """The visualize figures: neuron activity analysis (GNN_PlotFigure), activity.png, activity_all/selected.png.

    The two per-type figures are skipped with a warning if they fail; the
    others raise.
    """
    device = spec.device
    x_ts = trace.x_ts
    index_to_name = dict(INDEX_TO_NAME)

    activity = x_ts.voltage.to(device).t()  # (n_neurons, n_frames)
    type_list = x.neuron_type.unsqueeze(-1).to(device)

    target_type_name_list = ["R1", "R7", "C2", "Mi11", "Tm1", "Tm4", "Tm30"]
    from GNN_PlotFigure import plot_neuron_activity_analysis

    plot_neuron_activity_analysis(
        activity,
        target_type_name_list,
        type_list,
        index_to_name,
        n_neurons,
        spec.n_frames,
        spec.delta_t,
        store.folder,
    )

    logger.info("plot figure activity ...")
    # activity.png used to be plot_selected_neuron_traces over the WHOLE run
    # (start_frame=0, end_frame=n_frames): 64,000 frames squeezed into one axis,
    # which draws every trace as a solid band and shows nothing. It is now the
    # same nominal trace figure the trainer writes into
    # tmp_training/traces/rollout_*.png -- save_trace_figure from
    # models/teacher_eval.py -- over a 1,000-frame window, i.e. 20 s of
    # simulated time at delta_t = 20 ms. No prediction and no rollout
    # correlation exist at generation time, so pred and r are passed as None and
    # only the green ground truth plus the red stimulus are drawn.
    from connectome_gnn.models.teacher_eval import save_trace_figure

    activity_np = to_numpy(activity).T  # (n_frames, n_neurons), as save_trace_figure expects
    n_trace_frames = int(min(ACTIVITY_TRACE_FRAMES, activity_np.shape[0]))
    # Neuron 0's drive, one value per frame -- the same scalar the trainer's
    # rollout figure puts on its "stim" row.
    stim_np = (to_numpy(x_ts.stimulus[:n_trace_frames, 0])
               if x_ts.stimulus is not None else None)
    save_trace_figure(
        store.path('activity.png'),
        activity_np[:n_trace_frames],
        None,
        stim_np,
        spec.delta_t,
        None,
        type_names=index_to_name,
        type_list=to_numpy(type_list.squeeze()),
    )

    # THE SAME RENDERER, ONE ROW PER CELL TYPE. `activity.png` samples 12 neurons
    # by index, which is a thin slice of 65 types; these two answer "what does
    # every type look like" and "what do the types we always look at look like".
    # Black rather than green and a dashed stimulus, because at generation time
    # there is no prediction to contrast a green ground truth against -- the
    # figure is the data, not a comparison.
    try:
        _types = to_numpy(x_ts.neuron_type).astype(int)
        _first_of_type = {}
        for _i, _t in enumerate(_types):
            _first_of_type.setdefault(int(_t), _i)
        for _name, _ids in (
            ("activity_all.png", [_first_of_type[t] for t in sorted(_first_of_type)]),
            ("activity_selected.png",
             [_first_of_type[t] for t in CURATED_TYPES if t in _first_of_type]),
        ):
            if not _ids:  # golden-uncovered: B55 (every network has all 65 types)
                continue
            save_trace_figure(
                store.path(_name),
                activity_np[:n_trace_frames][:, _ids],
                None,
                stim_np,
                spec.delta_t,
                None,
                n_traces=len(_ids),
                type_names=index_to_name,
                type_list=_types[_ids],
                n_neurons=len(_ids),
                true_color="black",
                stim_linestyle="--",
                figsize=(9.0, 0.28 * len(_ids) + 1.6),
            )
            logger.info(f"wrote {_name} ({len(_ids)} cell types)")
    except Exception as _e:
        logger.warning(f"per-type trace figures skipped: {type(_e).__name__}: {_e}")


def render_video(spec, store, run: int = 0) -> None:
    """input_<id>.png (a copy of Fig_0_000000.png) and <id>.mp4 from Fig/, then empty Fig/.

    <id> is the dataset name after "flyvis_", or "no_id". Raises
    FileNotFoundError when no per-frame figure was drawn (fewer than 5 frames).
    """
    logger.info("generating lossless video ...")

    dataset = spec.output.dataset
    output_name = dataset.split("flyvis_")[1] if "flyvis_" in dataset else "no_id"
    src = store.path("Fig", "Fig_0_000000.png")
    dst = store.path(f"input_{output_name}.png")
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())

    generate_compressed_video_mp4(output_dir=store.path(), run=run,
                                  output_name=output_name, framerate=10)

    store.clear_figs()
