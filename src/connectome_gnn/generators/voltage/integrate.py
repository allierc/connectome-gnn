"""Integration of one split: its frame target, its writers, and the frame loop.

What ``x.stimulus`` holds at each frame is decided by the split's stimulus
programs (programs.py); everything else -- drift, measurement noise, the
write, the step, process noise -- is the per-frame step of ``run_frames``.

The state ``x`` is a LINEAR resource carried from the train split into the
test split: ``reset_for_test`` resets its voltage in place and redraws its
calcium, and everything else (stimulus, noise) carries over.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from tqdm import tqdm

from connectome_gnn.generators.voltage.programs import SplitStimulus
from connectome_gnn.log import get_logger
from connectome_gnn.plot import plot_spatial_activity_grid
from connectome_gnn.utils import to_numpy

logger = get_logger(__name__)


class FrameRecord(NamedTuple):
    """Frame t of a split as it is written: WHICH ARRAY CARRIES WHICH NOISE.

    xi is the process noise (noise_model_level), eta the measurement noise
    (measurement_noise_level). References into the running state, not copies:
    the writer copies when it buffers.
    """

    voltage: torch.Tensor            # v[t], before the step of frame t: xi[0..t-1], never eta -> voltage.zarr
    stimulus: torch.Tensor           # the input of frame t (on CPU a view of net.stimulus.buffer) -> stimulus.zarr
    measurement_noise: torch.Tensor  # eta[t]; added to no stored voltage                     -> noise.zarr
    drift: torch.Tensor              # f(v[t]) from pde(): neither xi nor eta                 -> y_list_<split>.zarr


@dataclass(frozen=True)
class SplitRun:
    """What integrating one split left behind (the data itself is on disk)."""

    split: str
    n_frames: int      # frames written by the frame loop (before train tiling)
    it: int            # frame counter after the last frame (train starts at start_frame, test at 0)
    id_fig: int        # number of the next per-frame figure


def split_target(spec, split: str, n_sequences: int) -> tuple:
    """(target frames, passes over the sequences) of a split, with legacy's log line.

    train: ``n_frames`` (or ``n_frames // repeat_factor`` when the train block is
    tiled afterwards), in as many passes as 35 frames per sequence need;
    ``n_frames == 0`` means one pass and no frame cap. test: one pass, capped
    at ``n_frames_test`` if set, else MAX_TEST_FRAMES.
    """
    from connectome_gnn.generators.voltage.stimulus import FRAMES_PER_SEQUENCE

    if split == "test":
        # Test: single pass through test sequences, capped at MAX_TEST_FRAMES (or sim.n_frames_test if set)
        test_target_frames = (spec.n_frames_test_cap if spec.n_frames_test_cap > 0 else spec.max_test_frames)
        logger.info(f"generating TEST data (capped at {test_target_frames} frames from {n_sequences} sequences)...")
        return test_target_frames, 1
    total_frames_per_pass = n_sequences * FRAMES_PER_SEQUENCE
    repeat_factor = spec.stimulus.repeat_factor
    if spec.n_frames == 0:
        num_passes_needed = 1
        target_frames = float("inf")
        logger.info(f"n_frames=0 mode: single pass through {n_sequences} train sequences")
    else:
        # When tiling a short unique block, only generate n_frames // factor
        # frames; the helper below replicates them across the full n_frames.
        target_frames = spec.n_frames // repeat_factor if repeat_factor > 1 else spec.n_frames
        num_passes_needed = (target_frames // total_frames_per_pass) + 1

    if repeat_factor > 1:
        logger.info(f"generating TRAIN data ({target_frames} unique frames, will tile ×{repeat_factor} → "
                    f"{target_frames * repeat_factor} total)...")
    else:
        logger.info(f"generating TRAIN data ({target_frames} frames from {n_sequences} sequences)...")
    return target_frames, num_passes_needed


def integrate_split(spec, split: str, *, sequences, net, pde, x, edge_index, initial_state, n_neurons,
                    davis_dataset, geometry, store, fig_style, id_fig_start: int, run: int = 0) -> SplitRun:
    """Integrate one split into x_list_<split>/ and y_list_<split>.zarr.

    The train split draws per-frame figures when visualize is set and starts
    its frame counter at start_frame; the test split never draws and starts
    at 0 (so ``it == 0`` conditions fire again on its first frame).
    """
    noise = spec.noise_for(split)
    target_frames, num_passes = split_target(spec, split, len(sequences))
    writer = store.split_writer(split, n_neurons, spec.save_calcium, to_numpy)
    train = split == "train"
    it, id_fig = run_frames(
        stimulus_sequences=sequences, net=net, pde=pde, x=x, edge_index=edge_index, initial_state=initial_state,
        spec=spec, store=store, writer=writer, target_frames=target_frames, num_passes=num_passes,
        n_neurons=n_neurons, device=spec.device, noise=noise,
        visualize=spec.output.visualize if train else False, run=run, run_vizualized=spec.output.run_vizualized,
        id_fig_start=id_fig_start, it_start=spec.start_frame if train else 0,
        fig_style=fig_style, davis_dataset=davis_dataset, geometry=geometry,
    )
    n_frames = writer.finalize()
    if train:
        logger.info(f"generated {n_frames} TRAIN frames (saved as .zarr)")
    else:
        _noise_tag = (
            f"noisy (noise_model={noise.process_std:g}, meas={noise.measurement_std:g})"
            if spec.noisy_test_data else "without noise"
        )
        logger.info(f"generated {n_frames} TEST frames {_noise_tag} (saved as .zarr)")
        if spec.noisy_test_data:
            # Marker for consumers that need to know the test split carries train-level noise
            open(store.path("noisy_test_data.ok"), "w").close()
    return SplitRun(split=split, n_frames=n_frames, it=it, id_fig=id_fig)


def reset_for_test(x, initial_state, n_neurons: int, device) -> None:
    """Reset x between the splits: voltage IN PLACE to the steady state, fresh calcium (torch.rand), no fluorescence.

    x.stimulus and x.noise carry over from the last train frame.
    """
    # Reset neural state to avoid train→test leakage
    x.voltage[:] = initial_state
    _init_calcium = torch.rand(n_neurons, dtype=torch.float32, device=device)
    x.calcium = _init_calcium
    x.fluorescence = torch.zeros(n_neurons, dtype=torch.float32, device=device)


def run_frames(
    stimulus_sequences,
    net,
    pde,
    x,
    edge_index,
    initial_state,
    spec,
    store,
    writer,
    target_frames,
    num_passes,
    n_neurons,
    device,
    noise,
    visualize=False,
    run=0,
    run_vizualized=0,
    id_fig_start=0,
    it_start=0,
    fig_style=None,
    davis_dataset=None,
    geometry=None,
):
    """The frame loop: pass over the sequences until ``target_frames``, writing every frame.

    Returns (it, id_fig) — the final frame counter and figure counter.
    """
    noise_model_level = noise.process_std
    measurement_noise_level = noise.measurement_std
    X1, u_coords, v_coords = geometry.X1, geometry.u_coords, geometry.v_coords
    st = spec.stimulus
    n_input_neurons = spec.network.n_input_neurons
    it = it_start
    id_fig = id_fig_start

    # The stimulus programs of this split (flash / mixed / tiles / video) and its
    # frame cursor; their state starts over with every call.
    programs = SplitStimulus(st, spec.seed, n_input_neurons, net, stimulus_sequences, davis_dataset,
                             u_coords, v_coords, device)
    cursor = programs.cursor

    # Track per-sequence lengths so we can report a post-hoc summary.
    # Critical for blank_prefix diagnostics: blank_prefix_frames = int(seq_len *
    # blank_prefix_fraction), so a dataset full of 2-frame sequences gets
    # ~1 blank frame per sequence — not enough time for neurons to decay to
    # V_rest.
    _seq_lens = []

    # AR(1) measurement-noise state: persists across all frames/sequences within
    # this generator call. Recursion eta(t+1) = rho*eta(t) + sqrt(1-rho**2)*gamma*xi(t)
    # preserves marginal Var(eta) = gamma**2. rho = 0 -> standard i.i.d. (current default).
    ar1_rho = noise.ar1_rho
    ar1_inject_std = (1.0 - ar1_rho ** 2) ** 0.5 * measurement_noise_level
    if measurement_noise_level > 0 and ar1_rho > 0:
        # Initialise in stationary distribution: Var(eta_0) = gamma**2
        ar1_prev_noise = (
            torch.randn(n_neurons, dtype=torch.float32, device=device)
            * measurement_noise_level
        )
    else:
        ar1_prev_noise = None

    target_reached = False
    with torch.no_grad():
        for pass_num in range(num_passes):
            for data_idx, data in enumerate(tqdm(stimulus_sequences, desc="processing stimulus data", ncols=100)):
                if st.simulation_initial_state:
                    x.voltage[:] = initial_state
                    if st.only_noise_visual_input > 0:
                        x.stimulus[: n_input_neurons] = torch.clamp(
                            torch.relu(
                                0.5
                                + torch.rand(n_input_neurons, dtype=torch.float32, device=device)
                                * st.only_noise_visual_input
                                / 2
                            ),
                            0,
                            1,
                        )

                sequences, sequence_length = programs.begin_sequence(data["lum"])

                blank_prefix_frames = int(sequence_length * st.blank_prefix_fraction)
                _seq_lens.append(int(sequence_length))

                frame_id = 0
                while frame_id < sequence_length:
                    programs.program.frame(x, frame_id, it, data_idx, sequences)

                    # Blank prefix: force zero stimulus for the first N frames of each sequence
                    if blank_prefix_frames > 0 and frame_id < blank_prefix_frames:
                        x.stimulus[:n_input_neurons] = 0

                    y = pde(x, edge_index, has_field=False)
                    dv_step = y.squeeze()
                    # Generate measurement noise for this timestep.
                    # AR(1) recursion when noise_ar1_rho > 0; falls back to i.i.d. otherwise.
                    if measurement_noise_level > 0:
                        if ar1_rho > 0:
                            ar1_prev_noise = (
                                ar1_rho * ar1_prev_noise
                                + torch.randn(n_neurons, dtype=torch.float32, device=device)
                                * ar1_inject_std
                            )
                            x.noise = ar1_prev_noise.clone()
                        else:
                            x.noise = (
                                torch.randn(n_neurons, dtype=torch.float32, device=device) * measurement_noise_level
                            )
                    else:
                        x.noise = torch.zeros(n_neurons, dtype=torch.float32, device=device)

                    record = FrameRecord(voltage=x.voltage, stimulus=x.stimulus, measurement_noise=x.noise, drift=y)
                    # Save x[t] BEFORE updating voltage to x[t+1]
                    writer.write_state(x, record)

                    # EXPONENTIAL EULER ON THE CONDUCTANCE BRANCH. Forward Euler
                    # contracts only while (dt/tau_i)(1 + G_i) < 2 and the generating
                    # network reaches 4.4 at dt = 20 ms with tau_i = 19 ms, which blows
                    # every trace up inside ten frames. `pde.step` solves the step
                    # exactly at frozen coefficients instead, a contraction for any
                    # G_i >= 0. `dv_step` is still the true derivative and is what gets
                    # STORED as the training target; only the state update changes.
                    if spec.network.conductance_exponential_euler:
                        x.voltage = pde.step(x, edge_index, spec.delta_t)
                    else:
                        x.voltage = x.voltage + spec.delta_t * dv_step
                    if noise_model_level > 0:
                        x.voltage = x.voltage + torch.randn(
                            n_neurons, dtype=torch.float32, device=device
                        ) * noise_model_level
                    if spec.calcium_type == "leaky":
                        raise NotImplementedError(
                            "calcium_type 'leaky' needs sim.calcium_tau / calcium_alpha / "
                            "calcium_beta, which commit ce0d1d9 ('finish calcium strip') "
                            "removed from SimulationConfig while leaving this branch. "
                            "Restore those fields before using it -- it has been dead "
                            "since 2026-08-03 and no tracked config selects it.")

                    writer.write_drift(record)

                    if (visualize & (run == run_vizualized) & (it > 0) & (it % 4 == 0) & (it <= 400)):
                        num = f"{id_fig:06}"
                        id_fig += 1
                        plot_spatial_activity_grid(
                            positions=to_numpy(X1),
                            voltages=to_numpy(x.voltage),
                            stimulus=to_numpy(x.stimulus[: n_input_neurons]),
                            neuron_types=to_numpy(x.neuron_type).astype(int),
                            output_path=store.path("Fig", f"Fig_{run}_{num}.png"),
                            calcium=to_numpy(x.calcium) if spec.calcium_type != "none" else None,
                            n_input_neurons=n_input_neurons,
                            style=fig_style,
                        )

                    # Advance the per-iteration cursors. With blank-window injection,
                    # frame_id (the video cursor) is held during blank iterations so
                    # blanks are inserted between real frames rather than replacing them.
                    frame_id = cursor.advance(frame_id)

                    it = it + 1
                    target_reached = cursor.progress(it) >= target_frames
                    if target_reached:
                        break

                if target_reached:
                    break
            if target_reached:
                break

    # Sequence-length summary (diagnostic for blank_prefix effectiveness).
    if _seq_lens:
        _arr = np.asarray(_seq_lens, dtype=np.int64)
        _bpf = float(st.blank_prefix_fraction)
        _bp_min = int(np.floor(_arr.min() * _bpf))
        _bp_med = int(np.floor(float(np.median(_arr)) * _bpf))
        _bp_max = int(np.floor(_arr.max() * _bpf))
        logger.info(
            "\033[93msequence-length summary: n_sequences=%d  frames=[min=%d median=%d mean=%.1f max=%d]  "
            "total_frames=%d  frames_consumed=%d\033[0m",
            int(_arr.size), int(_arr.min()), int(np.median(_arr)),
            float(_arr.mean()), int(_arr.max()), int(_arr.sum()), int(it - it_start),
        )
        logger.info(
            "\033[93mblank_prefix summary: blank_prefix_fraction=%.3f  blank_frames_per_seq=[min=%d median=%d max=%d]\033[0m",
            _bpf, _bp_min, _bp_med, _bp_max,
        )

    return it, id_fig
