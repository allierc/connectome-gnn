"""Voltage-data generation for flyvis networks: the stages of ``data_generate_voltage``.

``graph_data_generator.data_generate_voltage`` runs the VoltageGeneration
chain of pipeline.py; its output is byte-identical to the implementation at
``tests/golden_voltage/BASE_SHA`` (golden harness in tests/golden_voltage).

Modules
    spec          GenerationSpec: every config read, resolved once
    rng           RngSnapshot and the per-stage RNG ledger
    store         DatasetStore: paths, legacy erase, the split writer; ARRAY_PROVENANCE
    network       flyvis / conductance / FlyWire-hybrid network, FlyWire stimulus adapter
    stimulus      stimulus datasets, the train/test video split, materialised sequences
    edges         null edges, ablation, removal
    dynamics      ODE parameters and the FlyVisODE
    initial       geometry, steady state, the initial neuron state
    programs      the per-frame stimulus programs (flash, mixed, tiles, video, blank window)
    integrate     FrameRecord, the frame loop, one split's run
    postprocess   noisy derivative targets, train tiling
    diagnostics   bracket, ranks, SNR, generation_log.txt
    figures       previews, kinograph, trace figures, the input video
    pipeline      VoltageGeneration

QUIRKS (kept byte-identical in phase 2; each is documented in CAPS at the top
of the chain method named, and a ``<method>_v2`` is to fix it)

1.  fork_rng is a no-op: ``torch.random.fork_rng(devices=device)`` without
    ``with``; the caller sees generation's final RNG state.        -> seed
2.  stdlib ``random`` is never seeded; null edges are reproducible only
    because building the DAVIS dataset reseeds it (not on Sintel). -> add_null_edges
3.  ``x.stimulus`` aliases ``net.stimulus.buffer`` on CPU (not on CUDA/MPS,
    where ``.to`` copies); with only_noise_visual_input the stored stimulus
    is the rendered frame whenever the noise condition is false.
    The PI accepts the GPU semantics for v2.                        -> init_state
4.  the erase checks x_list_* / y_list_* without ``.zarr``: y_list_*.zarr,
    noisy_y_list_*.zarr and .pt/.ok/.txt side files survive.       -> prepare_output
    ... and a stale noisy_y_list_train.zarr is tiled.               -> tile_train
5.  noisy_y_list = y + delta(eta)/dt, never + xi/dt.               -> derive_noisy_targets
6.  exponential Euler (``pde.step``) is applied to the current-based model
    too, so (v[t+1] - v[t]) / dt != y_list[t] even at sigma = 0
    (tau-hat ~ tau + dt/2; tanh models fall back to Euler).         -> integrate
7.  a cold rendering cache draws torch RNG (the BoxEye Conv2d init) after
    ``seed``, shifting every later torch draw.                      -> load_stimuli
8.  ``truncate_max_frames`` never reaches RenderedDavis (DAVIS path). -> load_stimuli
9.  two or more ``datavis_roots`` fail: CombinedVideoDataset has no
    ``arg_df`` (AttributeError).                                    -> load_stimuli, split_videos
10. mixed mode draws its davis clips from the whole DAVIS dataset, test
    videos included, while generating the train split.             -> integrate
"""

from connectome_gnn.generators.voltage.pipeline import StaleStageError, VoltageGeneration
from connectome_gnn.generators.voltage.spec import GenerationSpec

__all__ = ["GenerationSpec", "StaleStageError", "VoltageGeneration"]
