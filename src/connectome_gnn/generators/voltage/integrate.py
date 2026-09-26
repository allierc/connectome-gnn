"""The frame loop of voltage generation: one call integrates one split.

Moved verbatim from ``graph_data_generator`` (refactor phase 2, step 5);
``graph_data_generator._run_ode_generation`` stays importable.
"""

import numpy as np
import torch
from tqdm import tqdm

from connectome_gnn.generators.utils import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    greedy_blue_mask,
    mseq_bits,
)
from connectome_gnn.log import get_logger
from connectome_gnn.plot import plot_spatial_activity_grid
from connectome_gnn.utils import graphs_data_path

logger = get_logger(__name__)


def _run_ode_generation(
    stimulus_sequences,
    net,
    pde,
    x,
    edge_index,
    initial_state,
    sim,
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
    config=None,
    davis_dataset=None,
    X1=None,
    u_coords=None,
    v_coords=None,
):
    """Run ODE simulation over stimulus sequences, writing frames to zarr.

    This is the inner loop extracted so it can be called for both train and test.
    Returns (it, id_fig) — the final frame counter and figure counter.
    """
    it = it_start
    id_fig = id_fig_start


    tile_labels = None
    tile_codes_torch = None
    tile_period = None
    tile_idx = 0
    n_columns = sim.n_input_neurons // 8

    # Mixed sequence setup
    mixed_types_list = None
    if "mixed" in sim.visual_input_type:
        mixed_types_list = ["sintel", "davis", "blank", "noise"]
        mixed_cycle_lengths = [60, 60, 30, 60]
        mixed_current_type = 0
        mixed_frame_count = 0
        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
        sintel_iter = iter(stimulus_sequences)
        davis_iter = iter(davis_dataset) if davis_dataset else iter(stimulus_sequences)
        current_sintel_seq = None
        current_davis_seq = None
        sintel_frame_idx = 0
        davis_frame_idx = 0


    # Track per-sequence lengths so we can report a post-hoc summary.
    # Critical for blank_prefix diagnostics: blank_prefix_frames = int(seq_len *
    # blank_prefix_fraction), so a dataset full of 2-frame sequences gets
    # ~1 blank frame per sequence — not enough time for neurons to decay to
    # V_rest.
    _seq_lens = []

    # AR(1) measurement-noise state: persists across all frames/sequences within
    # this generator call. Recursion eta(t+1) = rho*eta(t) + sqrt(1-rho**2)*gamma*xi(t)
    # preserves marginal Var(eta) = gamma**2. rho = 0 -> standard i.i.d. (current default).
    ar1_rho = float(getattr(sim, 'noise_ar1_rho', 0.0))
    ar1_inject_std = (1.0 - ar1_rho ** 2) ** 0.5 * measurement_noise_level
    if measurement_noise_level > 0 and ar1_rho > 0:
        # Initialise in stationary distribution: Var(eta_0) = gamma**2
        ar1_prev_noise = (
            torch.randn(n_neurons, dtype=torch.float32, device=device)
            * measurement_noise_level
        )
    else:
        ar1_prev_noise = None

    # DAVIS blank-window injection (see SimulationConfig validator for compatibility).
    # State persists across video boundaries and across passes so the m-real / l-blank
    # pattern is preserved continuously.
    bw_size = int(getattr(sim, "blank_window_size_frames", 0))
    bw_every = int(getattr(sim, "blank_insertion_every_n_frames", 0))
    use_blank_injection = bw_size > 0 and bw_every > 0
    real_frames_consumed = 0
    real_frames_in_chunk = 0
    in_blank_window = False
    blank_remaining = 0

    target_reached = False
    with torch.no_grad():
        for pass_num in range(num_passes):
            for data_idx, data in enumerate(tqdm(stimulus_sequences, desc="processing stimulus data", ncols=100)):
                if sim.simulation_initial_state:
                    x.voltage[:] = initial_state
                    if sim.only_noise_visual_input > 0:
                        x.stimulus[: sim.n_input_neurons] = torch.clamp(
                            torch.relu(
                                0.5
                                + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device)
                                * sim.only_noise_visual_input
                                / 2
                            ),
                            0,
                            1,
                        )

                sequences = data["lum"]

                if "flash" in sim.visual_input_type:
                    flash_duration_options = [1, 2, 5]
                    flash_cycle_frames = flash_duration_options[
                        torch.randint(0, len(flash_duration_options), (1,), device=device).item()
                    ]
                    flash_intensity = torch.abs(torch.rand(sim.n_input_neurons, device=device) * 0.5 + 0.5)

                if mixed_types_list is not None:
                    if mixed_frame_count >= current_cycle_length:
                        mixed_current_type = (mixed_current_type + 1) % 4
                        mixed_frame_count = 0
                        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
                    current_type = mixed_types_list[mixed_current_type]

                    if current_type == "sintel":
                        if current_sintel_seq is None or sintel_frame_idx >= current_sintel_seq["lum"].shape[0]:
                            try:
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                            except StopIteration:
                                sintel_iter = iter(stimulus_sequences)
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                        sequences = current_sintel_seq["lum"]
                        start_frame = sintel_frame_idx
                    elif current_type == "davis":
                        if current_davis_seq is None or davis_frame_idx >= current_davis_seq["lum"].shape[0]:
                            try:
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                            except StopIteration:
                                davis_iter = iter(davis_dataset) if davis_dataset else iter(stimulus_sequences)
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                        sequences = current_davis_seq["lum"]
                        start_frame = davis_frame_idx
                    else:
                        start_frame = 0

                if "flash" in sim.visual_input_type:
                    sequence_length = 60
                else:
                    sequence_length = sequences.shape[0]

                blank_prefix_frames = int(sequence_length * getattr(sim, 'blank_prefix_fraction', 0.0))
                _seq_lens.append(int(sequence_length))

                frame_id = 0
                while frame_id < sequence_length:
                    if "flash" in sim.visual_input_type:
                        current_flash_frame = frame_id % (flash_cycle_frames * 2)
                        x.stimulus[:] = 0
                        if current_flash_frame < flash_cycle_frames:
                            x.stimulus[: sim.n_input_neurons] = flash_intensity
                    elif mixed_types_list is not None:
                        current_type = mixed_types_list[mixed_current_type]
                        if current_type == "blank":
                            x.stimulus[:] = 0
                        elif current_type == "noise":
                            x.stimulus[: sim.n_input_neurons] = torch.relu(
                                0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device) * 0.5
                            )
                        else:
                            actual_frame_id = (start_frame + frame_id) % sequences.shape[0]
                            frame = sequences[actual_frame_id][None, None]
                            net.stimulus.add_input(frame)
                            x.stimulus[:] = net.stimulus().squeeze()
                            if current_type == "sintel":
                                sintel_frame_idx += 1
                            elif current_type == "davis":
                                davis_frame_idx += 1
                        mixed_frame_count += 1
                    elif "tile_mseq" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            tile_labels_np = assign_columns_from_uv(
                                u_coords, v_coords, n_columns, random_state=sim.seed
                            )
                            base = mseq_bits(p=8, seed=sim.seed).astype(np.float32)
                            rng = np.random.RandomState(sim.seed)
                            phases = rng.randint(0, base.shape[0], size=n_columns)
                            tile_codes_np = np.stack([np.roll(base, ph) for ph in phases], axis=0)
                            tile_codes_torch = torch.from_numpy(tile_codes_np).to(device, dtype=torch.float32)
                            tile_labels = torch.from_numpy(tile_labels_np).to(device, dtype=torch.long)
                            tile_period = tile_codes_torch.shape[1]
                            tile_idx = 0

                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx),
                        )
                        col_vals_01 = 0.5 + (sim.tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[: sim.n_input_neurons] = col_vals_01[tile_labels]
                        tile_idx += 1
                    elif "tile_blue_noise" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            tile_labels_np, col_centers = compute_column_labels(
                                u_coords, v_coords, n_columns, seed=sim.seed
                            )
                            try:
                                adj = build_neighbor_graph(col_centers, k=6)
                            except Exception:
                                from scipy.spatial.distance import pdist, squareform

                                D = squareform(pdist(col_centers))
                                nn = np.partition(D + np.eye(D.shape[0]) * 1e9, 1, axis=1)[:, 1]
                                radius = 1.3 * np.median(nn)
                                adj = [
                                    set(np.where((D[i] > 0) & (D[i] <= radius))[0].tolist())
                                    for i in range(len(col_centers))
                                ]

                            tile_labels = torch.from_numpy(tile_labels_np).to(device, dtype=torch.long)
                            tile_period = 257
                            tile_idx = 0

                            tile_codes_torch = torch.empty((n_columns, tile_period), dtype=torch.float32, device=device)
                            rng = np.random.RandomState(sim.seed)
                            for t in range(tile_period):
                                mask = greedy_blue_mask(adj, n_columns, target_density=0.5, rng=rng)
                                vals = np.where(mask, 1.0, -1.0).astype(np.float32)
                                tile_codes_torch[:, t] = torch.from_numpy(vals).to(device, dtype=torch.float32)

                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx),
                        )
                        col_vals_01 = 0.5 + (sim.tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[: sim.n_input_neurons] = col_vals_01[tile_labels]
                        tile_idx += 1
                    elif use_blank_injection and in_blank_window:
                        # DAVIS blank-window injection: zero stimulus and hold the video
                        # cursor (frame_id is not advanced this iteration).
                        x.stimulus[:] = 0
                    else:
                        frame = sequences[frame_id][None, None]
                        net.stimulus.add_input(frame)
                        if sim.only_noise_visual_input > 0:
                            if (sim.visual_input_type == "") | (it == 0) | ("50/50" in sim.visual_input_type):
                                x.stimulus[: sim.n_input_neurons] = torch.relu(
                                    0.5
                                    + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device)
                                    * sim.only_noise_visual_input
                                    / 2
                                )
                        else:
                            # legacy blank injection
                            if sim.blank_freq > 0:
                                if data_idx % sim.blank_freq > 0:
                                    x.stimulus[:] = net.stimulus().squeeze()
                                else:
                                    x.stimulus[:] = 0
                            else:
                                x.stimulus[:] = net.stimulus().squeeze()
                            if sim.noise_visual_input > 0:
                                x.stimulus[: sim.n_input_neurons] = (
                                    x.stimulus[: sim.n_input_neurons]
                                    + torch.randn(sim.n_input_neurons, dtype=torch.float32, device=device)
                                    * sim.noise_visual_input
                                )

                    # Blank prefix: force zero stimulus for the first N frames of each sequence
                    if blank_prefix_frames > 0 and frame_id < blank_prefix_frames:
                        x.stimulus[:sim.n_input_neurons] = 0

                    prev_calcium = x.calcium.clone() if x.calcium is not None else None

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
                    if getattr(sim, "conductance_exponential_euler", True):
                        x.voltage = pde.step(x, edge_index, sim.delta_t)
                    else:
                        x.voltage = x.voltage + sim.delta_t * dv_step
                    if noise_model_level > 0:
                        x.voltage = x.voltage + torch.randn(
                            n_neurons, dtype=torch.float32, device=device
                        ) * noise_model_level
                    if sim.calcium_type == "leaky":
                        if sim.calcium_activation == "softplus":
                            s = torch.nn.functional.softplus(x.voltage)
                        elif sim.calcium_activation == "relu":
                            s = torch.nn.functional.relu(x.voltage)
                        elif sim.calcium_activation == "tanh":
                            s = 1 + torch.tanh(x.voltage)
                        elif sim.calcium_activation == "identity":
                            s = x.voltage.clone()

                        raise NotImplementedError(
                            "calcium_type 'leaky' needs sim.calcium_tau / calcium_alpha / "
                            "calcium_beta, which commit ce0d1d9 ('finish calcium strip') "
                            "removed from SimulationConfig while leaving this branch. "
                            "Restore those fields before using it -- it has been dead "
                            "since 2026-08-03 and no tracked config selects it.")
                        y = ((x.calcium - prev_calcium) / sim.delta_t).unsqueeze(-1)

                    y_writer.append(to_numpy_fn(y.clone().detach()))

                    if (visualize & (run == run_vizualized) & (it > 0) & (it % 4 == 0) & (it <= 400)):
                        num = f"{id_fig:06}"
                        id_fig += 1
                        plot_spatial_activity_grid(
                            positions=to_numpy_fn(X1),
                            voltages=to_numpy_fn(x.voltage),
                            stimulus=to_numpy_fn(x.stimulus[: sim.n_input_neurons]),
                            neuron_types=to_numpy_fn(x.neuron_type).astype(int),
                            output_path=graphs_data_path(config.dataset, "Fig", f"Fig_{run}_{num}.png"),
                            calcium=to_numpy_fn(x.calcium) if sim.calcium_type != "none" else None,
                            n_input_neurons=sim.n_input_neurons,
                            style=fig_style,
                        )

                    # Advance the per-iteration cursors. With blank-window injection,
                    # frame_id (the video cursor) is held during blank iterations so
                    # blanks are inserted between real frames rather than replacing them.
                    if use_blank_injection:
                        if in_blank_window:
                            blank_remaining -= 1
                            if blank_remaining <= 0:
                                in_blank_window = False
                                real_frames_in_chunk = 0
                        else:
                            frame_id += 1
                            real_frames_consumed += 1
                            real_frames_in_chunk += 1
                            if real_frames_in_chunk >= bw_every:
                                in_blank_window = True
                                blank_remaining = bw_size
                                real_frames_in_chunk = 0
                    else:
                        frame_id += 1

                    it = it + 1
                    target_reached = (
                        real_frames_consumed if use_blank_injection else it
                    ) >= target_frames
                    if target_reached:
                        break

                if target_reached:
                    break
            if target_reached:
                break

    # Sequence-length summary (diagnostic for blank_prefix effectiveness).
    if _seq_lens:
        _arr = np.asarray(_seq_lens, dtype=np.int64)
        _bpf = float(getattr(sim, 'blank_prefix_fraction', 0.0))
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
