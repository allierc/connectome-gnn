"""Stimulus programs: what ``x.stimulus`` holds at each frame of a split.

``visual_input_type`` is matched by SUBSTRING, so composite strings such as
"DAVIS flash" or "DAVIS mixed" are legal and choose behaviour. The per-frame
program is the FIRST match in legacy precedence order:

    flash  >  mixed  >  tile_mseq  >  tile_blue_noise  >  (blank window)  >  video

Two programs also act once per SEQUENCE, independently of which one owns the
frames: Flash draws its cycle length and intensity (torch RNG), and Mixed
advances its cycle and may switch to the next sintel / davis clip. Both
hooks run whenever their substring is present, in that order, as legacy did
(so "flash mixed" draws flash parameters AND walks the mixed iterators,
although Flash owns every frame).

Every program instance lives for ONE split: tile index, mixed cycle, flash
timing and blank-window cursor start over for the test split, as legacy's
per-call locals did.

The blank window (``blank_window_size_frames`` > 0 and
``blank_insertion_every_n_frames`` > 0) is two things: a cursor that holds the
video frame index during blank frames and counts real frames toward the
target (BlankWindowCursor, applied whatever program owns the frames), and a
per-frame override that zeroes the stimulus inside the window, which only
the video program sees (BlankWindow), because legacy's chain tested it after
the four named programs.
"""

from __future__ import annotations

import numpy as np
import torch

from connectome_gnn.generators.utils import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    greedy_blue_mask,
    mseq_bits,
)


class Flash:
    """Full-field flashes: on for c frames, off for c frames, c drawn per sequence from {1, 2, 5}."""

    SEQUENCE_LENGTH = 60      # frames per sequence, whatever the clip's length

    def __init__(self, n_input_neurons: int, device):
        self.n_input_neurons = n_input_neurons
        self.device = device

    def begin_sequence(self) -> None:
        """Draw this sequence's cycle length (torch.randint) and intensities (torch.rand)."""
        flash_duration_options = [1, 2, 5]
        self.flash_cycle_frames = flash_duration_options[
            torch.randint(0, len(flash_duration_options), (1,), device=self.device).item()
        ]
        self.flash_intensity = torch.abs(torch.rand(self.n_input_neurons, device=self.device) * 0.5 + 0.5)

    def frame(self, x, frame_id, it, data_idx, sequences) -> None:
        current_flash_frame = frame_id % (self.flash_cycle_frames * 2)
        x.stimulus[:] = 0
        if current_flash_frame < self.flash_cycle_frames:
            x.stimulus[: self.n_input_neurons] = self.flash_intensity


class Mixed:
    """Cycles sintel (60 frames) -> davis (60) -> blank (30) -> noise (60), switching at sequence starts.

    "sintel" clips come from the split's own sequences; "davis" clips from
    ``davis_dataset`` when there is one (truthy), else also from the split's
    sequences. QUIRK: the davis iterator walks the WHOLE DAVIS dataset, test
    videos included, while the TRAIN split is generated; see
    ``VoltageGeneration.integrate``. An exhausted iterator restarts from the
    beginning.
    """

    TYPES = ["sintel", "davis", "blank", "noise"]
    CYCLE_LENGTHS = [60, 60, 30, 60]

    def __init__(self, stimulus_sequences, davis_dataset, net, n_input_neurons: int, device):
        self.stimulus_sequences = stimulus_sequences
        self.davis_dataset = davis_dataset
        self.net = net
        self.n_input_neurons = n_input_neurons
        self.device = device
        self.mixed_current_type = 0
        self.mixed_frame_count = 0
        self.current_cycle_length = self.CYCLE_LENGTHS[self.mixed_current_type]
        self.sintel_iter = iter(stimulus_sequences)
        self.davis_iter = iter(davis_dataset) if davis_dataset else iter(stimulus_sequences)
        self.current_sintel_seq = None
        self.current_davis_seq = None
        self.sintel_frame_idx = 0
        self.davis_frame_idx = 0
        self.start_frame = 0
        self.sequences = None

    def begin_sequence(self, sequences):
        """Advance the cycle if its phase is over; return the clip this sequence plays."""
        if self.mixed_frame_count >= self.current_cycle_length:
            self.mixed_current_type = (self.mixed_current_type + 1) % 4
            self.mixed_frame_count = 0
            self.current_cycle_length = self.CYCLE_LENGTHS[self.mixed_current_type]
        current_type = self.TYPES[self.mixed_current_type]

        if current_type == "sintel":
            if self.current_sintel_seq is None or self.sintel_frame_idx >= self.current_sintel_seq["lum"].shape[0]:
                try:
                    self.current_sintel_seq = next(self.sintel_iter)
                    self.sintel_frame_idx = 0
                except StopIteration:
                    self.sintel_iter = iter(self.stimulus_sequences)
                    self.current_sintel_seq = next(self.sintel_iter)
                    self.sintel_frame_idx = 0
            sequences = self.current_sintel_seq["lum"]
            self.start_frame = self.sintel_frame_idx
        elif current_type == "davis":
            if self.current_davis_seq is None or self.davis_frame_idx >= self.current_davis_seq["lum"].shape[0]:
                try:
                    self.current_davis_seq = next(self.davis_iter)
                    self.davis_frame_idx = 0
                except StopIteration:
                    self.davis_iter = (iter(self.davis_dataset) if self.davis_dataset
                                       else iter(self.stimulus_sequences))
                    self.current_davis_seq = next(self.davis_iter)
                    self.davis_frame_idx = 0
            sequences = self.current_davis_seq["lum"]
            self.start_frame = self.davis_frame_idx
        else:
            self.start_frame = 0
        self.sequences = sequences
        return sequences

    def frame(self, x, frame_id, it, data_idx, sequences) -> None:
        current_type = self.TYPES[self.mixed_current_type]
        if current_type == "blank":
            x.stimulus[:] = 0
        elif current_type == "noise":
            x.stimulus[: self.n_input_neurons] = torch.relu(
                0.5 + torch.rand(self.n_input_neurons, dtype=torch.float32, device=self.device) * 0.5
            )
        else:
            actual_frame_id = (self.start_frame + frame_id) % self.sequences.shape[0]
            frame = self.sequences[actual_frame_id][None, None]
            self.net.stimulus.add_input(frame)
            x.stimulus[:] = self.net.stimulus().squeeze()
            if current_type == "sintel":
                self.sintel_frame_idx += 1
            else:  # "davis": blank and noise never get here (legacy's elif could not be False)
                self.davis_frame_idx += 1
        self.mixed_frame_count += 1


class _Tiles:
    """Column tiles: each photoreceptor column shows +-contrast/2 around 0.5 from a per-column code."""

    def __init__(self, stim, seed: int, n_input_neurons: int, u_coords, v_coords, device):
        self.stim = stim
        self.seed = seed
        self.n_input_neurons = n_input_neurons
        self.n_columns = n_input_neurons // 8
        self.u_coords = u_coords
        self.v_coords = v_coords
        self.device = device
        self.tile_codes_torch = None
        self.tile_labels = None
        self.tile_period = None
        self.tile_idx = 0

    def frame(self, x, frame_id, it, data_idx, sequences) -> None:
        if self.tile_codes_torch is None:
            self._init_codes()
        x.stimulus[:] = 0.5
        col_vals_pm1 = self.tile_codes_torch[:, self.tile_idx % self.tile_period]
        col_vals_pm1 = apply_pairwise_knobs_torch(
            code_pm1=col_vals_pm1,
            corr_strength=float(self.stim.tile_corr_strength),
            flip_prob=float(self.stim.tile_flip_prob),
            seed=int(self.seed) + int(self.tile_idx),
        )
        col_vals_01 = 0.5 + (self.stim.tile_contrast * 0.5) * col_vals_pm1
        x.stimulus[: self.n_input_neurons] = col_vals_01[self.tile_labels]
        self.tile_idx += 1


class TileMseq(_Tiles):
    """Each column plays one m-sequence (p = 8) at its own random phase (RandomState(seed))."""

    def _init_codes(self) -> None:
        n_columns, device = self.n_columns, self.device
        tile_labels_np = assign_columns_from_uv(
            self.u_coords, self.v_coords, n_columns, random_state=self.seed
        )
        base = mseq_bits(p=8, seed=self.seed).astype(np.float32)
        rng = np.random.RandomState(self.seed)
        phases = rng.randint(0, base.shape[0], size=n_columns)
        tile_codes_np = np.stack([np.roll(base, ph) for ph in phases], axis=0)
        self.tile_codes_torch = torch.from_numpy(tile_codes_np).to(device, dtype=torch.float32)
        self.tile_labels = torch.from_numpy(tile_labels_np).to(device, dtype=torch.long)
        self.tile_period = self.tile_codes_torch.shape[1]
        self.tile_idx = 0


class TileBlueNoise(_Tiles):
    """257 frames of spatially blue noise over the columns (greedy masks, RandomState(seed))."""

    def _init_codes(self) -> None:
        n_columns, device = self.n_columns, self.device
        tile_labels_np, col_centers = compute_column_labels(
            self.u_coords, self.v_coords, n_columns, seed=self.seed
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

        self.tile_labels = torch.from_numpy(tile_labels_np).to(device, dtype=torch.long)
        self.tile_period = 257
        self.tile_idx = 0

        self.tile_codes_torch = torch.empty((n_columns, self.tile_period), dtype=torch.float32, device=device)
        rng = np.random.RandomState(self.seed)
        for t in range(self.tile_period):
            mask = greedy_blue_mask(adj, n_columns, target_density=0.5, rng=rng)
            vals = np.where(mask, 1.0, -1.0).astype(np.float32)
            self.tile_codes_torch[:, t] = torch.from_numpy(vals).to(device, dtype=torch.float32)


class Video:
    """The clip's own frames through ``net.stimulus``, with the only_noise / blank_freq / noise options.

    only_noise_visual_input > 0: the photoreceptors get uniform noise instead,
    but only while ``visual_input_type == ""`` or it == 0 or "50/50" is in the
    type. QUIRK: otherwise x.stimulus is not written at all, and what gets
    stored is whatever ``add_input`` left in ``net.stimulus.buffer``, which
    x.stimulus aliases on CPU -- the rendered frame, not noise (golden cell F9).
    blank_freq > 0 zeroes every sequence whose index is a multiple of it;
    noise_visual_input adds Gaussian noise to the photoreceptors.
    """

    def __init__(self, stim, net, n_input_neurons: int, device):
        self.stim = stim
        self.net = net
        self.n_input_neurons = n_input_neurons
        self.device = device

    def frame(self, x, frame_id, it, data_idx, sequences) -> None:
        st, n_in = self.stim, self.n_input_neurons
        frame = sequences[frame_id][None, None]
        self.net.stimulus.add_input(frame)
        if st.only_noise_visual_input > 0:
            if (st.visual_input_type == "") | (it == 0) | ("50/50" in st.visual_input_type):
                x.stimulus[: n_in] = torch.relu(
                    0.5
                    + torch.rand(n_in, dtype=torch.float32, device=self.device)
                    * st.only_noise_visual_input
                    / 2
                )
        else:
            # legacy blank injection
            if st.blank_freq > 0:
                if data_idx % st.blank_freq > 0:
                    x.stimulus[:] = self.net.stimulus().squeeze()
                else:
                    x.stimulus[:] = 0
            else:
                x.stimulus[:] = self.net.stimulus().squeeze()
            if st.noise_visual_input > 0:
                x.stimulus[: n_in] = (
                    x.stimulus[: n_in]
                    + torch.randn(n_in, dtype=torch.float32, device=self.device)
                    * st.noise_visual_input
                )


class BlankWindow:
    """Zero stimulus while the cursor is in a blank window; the video program otherwise."""

    def __init__(self, inner: Video, cursor):
        self.inner = inner
        self.cursor = cursor

    def frame(self, x, frame_id, it, data_idx, sequences) -> None:
        if self.cursor.in_blank_window:
            # DAVIS blank-window injection: zero stimulus and hold the video
            # cursor (frame_id is not advanced this iteration).
            x.stimulus[:] = 0
        else:
            self.inner.frame(x, frame_id, it, data_idx, sequences)


class PlainCursor:
    """Every frame is a real frame: advance the clip, count ``it`` toward the target."""

    in_blank_window = False

    def advance(self, frame_id: int) -> int:
        return frame_id + 1

    def progress(self, it: int) -> int:
        return it


class BlankWindowCursor:
    """Blank windows of ``size`` frames after every ``every`` real frames, carried across clips and passes.

    Inside a window the clip's frame index is held, so blanks are inserted
    between real frames instead of replacing them, and only real frames count
    toward the target.
    """

    def __init__(self, size: int, every: int):
        self.bw_size = size
        self.bw_every = every
        self.real_frames_consumed = 0
        self.real_frames_in_chunk = 0
        self.in_blank_window = False
        self.blank_remaining = 0

    def advance(self, frame_id: int) -> int:
        if self.in_blank_window:
            self.blank_remaining -= 1
            if self.blank_remaining <= 0:
                self.in_blank_window = False
                self.real_frames_in_chunk = 0
            return frame_id
        frame_id += 1
        self.real_frames_consumed += 1
        self.real_frames_in_chunk += 1
        if self.real_frames_in_chunk >= self.bw_every:
            self.in_blank_window = True
            self.blank_remaining = self.bw_size
            self.real_frames_in_chunk = 0
        return frame_id

    def progress(self, it: int) -> int:
        return self.real_frames_consumed


class SplitStimulus:
    """The programs of one split: the per-sequence hooks, the per-frame program, the frame cursor."""

    def __init__(self, stim, seed: int, n_input_neurons: int, net, stimulus_sequences, davis_dataset,
                 u_coords, v_coords, device):
        vit = stim.visual_input_type
        use_blank_injection = stim.blank_window_size_frames > 0 and stim.blank_insertion_every_n_frames > 0
        self.cursor = (BlankWindowCursor(stim.blank_window_size_frames, stim.blank_insertion_every_n_frames)
                       if use_blank_injection else PlainCursor())
        self.flash = Flash(n_input_neurons, device) if "flash" in vit else None
        self.mixed = (Mixed(stimulus_sequences, davis_dataset, net, n_input_neurons, device)
                      if "mixed" in vit else None)
        if self.flash is not None:
            self.program = self.flash
        elif self.mixed is not None:
            self.program = self.mixed
        elif "tile_mseq" in vit:
            self.program = TileMseq(stim, seed, n_input_neurons, u_coords, v_coords, device)
        elif "tile_blue_noise" in vit:
            self.program = TileBlueNoise(stim, seed, n_input_neurons, u_coords, v_coords, device)
        elif use_blank_injection:
            self.program = BlankWindow(Video(stim, net, n_input_neurons, device), self.cursor)
        else:
            self.program = Video(stim, net, n_input_neurons, device)

    def begin_sequence(self, sequences):
        """Run the per-sequence hooks; return (the clip to play, its length in frames)."""
        if self.flash is not None:
            self.flash.begin_sequence()
        if self.mixed is not None:
            sequences = self.mixed.begin_sequence(sequences)
        sequence_length = Flash.SEQUENCE_LENGTH if self.flash is not None else sequences.shape[0]
        return sequences, sequence_length
