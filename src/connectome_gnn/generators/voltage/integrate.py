"""The frame loop of voltage generation: one call integrates one split.

What ``x.stimulus`` holds at each frame is decided by the split's stimulus
programs (programs.py); everything else -- drift, measurement noise, the
write, the step, process noise -- is the per-frame step below, unchanged.
``graph_data_generator._run_ode_generation`` stays importable.
"""

import numpy as np
import torch
from tqdm import tqdm

from connectome_gnn.generators.voltage.programs import SplitStimulus
from connectome_gnn.log import get_logger
from connectome_gnn.plot import plot_spatial_activity_grid

logger = get_logger(__name__)


def _run_ode_generation(
    stimulus_sequences,
    net,
    pde,
    x,
    edge_index,
    initial_state,
    spec,
    store,
    x_writer,
    y_writer,
    target_frames,
    num_passes,
    n_neurons,
    device,
    to_numpy_fn,
    noise_model_level: float,
    measurement_noise_level: float,
    visualize=False,
    run=0,
    run_vizualized=0,
    step=5,
    id_fig_start=0,
    it_start=0,
    fig_style=None,
    davis_dataset=None,
    X1=None,
    u_coords=None,
    v_coords=None,
):
    """Run ODE simulation over stimulus sequences, writing frames to zarr.

    This is the inner loop extracted so it can be called for both train and test.
    Returns (it, id_fig) — the final frame counter and figure counter.
    """
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
    ar1_rho = spec.train_noise.ar1_rho
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

                    # Save x[t] BEFORE updating voltage to x[t+1]
                    x_writer.append_state(x)

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

                    y_writer.append(to_numpy_fn(y.clone().detach()))

                    if (visualize & (run == run_vizualized) & (it > 0) & (it % 4 == 0) & (it <= 400)):
                        num = f"{id_fig:06}"
                        id_fig += 1
                        plot_spatial_activity_grid(
                            positions=to_numpy_fn(X1),
                            voltages=to_numpy_fn(x.voltage),
                            stimulus=to_numpy_fn(x.stimulus[: n_input_neurons]),
                            neuron_types=to_numpy_fn(x.neuron_type).astype(int),
                            output_path=store.path("Fig", f"Fig_{run}_{num}.png"),
                            calcium=to_numpy_fn(x.calcium) if spec.calcium_type != "none" else None,
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
