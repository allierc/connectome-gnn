"""GenerationSpec: every configuration value voltage generation reads, resolved once.

``data_generate_voltage`` used to read ``config.simulation`` in about 150
places, a third of them through ``getattr(sim, name, default)``. The spec
reads each value once, applies the same default, and freezes it, so a stage
receives values rather than the config and the defaults live in one place.

WHAT IS RESOLVED HERE AND WHAT IS NOT. A field keeps the exact value legacy
read -- raw where legacy used it raw, converted (``float``/``bool``/``int``)
where legacy converted it at every use -- because the values end up in log
lines and in generation_log.txt, where ``0.05`` and ``0.050`` differ. Two
reads are unused by legacy and not resolved: ``config.training`` and the
``step`` argument (kept on OutputSpec only because it is part of the
signature).

PHANTOM FIELDS. ``save_calcium`` and ``n_frames_test`` are read with
``getattr`` but SimulationConfig has no such fields and ``extra="ignore"``
drops them from YAML, so both are always their defaults unless a caller
passes a config object that carries them (the golden harness does, through a
proxy). They are resolved here once instead of three times and once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Test split length when the config does not cap it (phantom n_frames_test).
MAX_TEST_FRAMES = 8000


@dataclass(frozen=True)
class NoiseSpec:
    """Noise levels of one split, and which array carries each noise.

    process_std          xi, added to v AFTER frame t is written, so voltage.zarr
                         carries it from frame t+1 on (noise_model_level)
    measurement_std      eta, written to noise.zarr and never added to a stored
                         voltage (measurement_noise_level)
    ar1_rho              AR(1) coefficient of eta; 0 means i.i.d.
    gate_measurement_std the TRAIN measurement level, whatever the split: legacy
                         gates the noisy_y_list and SNR blocks on it
    """

    process_std: float
    measurement_std: float
    ar1_rho: float
    gate_measurement_std: float


@dataclass(frozen=True)
class StimulusSpec:
    """Where the visual input comes from and how each frame is built."""

    visual_input_type: str
    datavis_roots: tuple                 # printed as a list: use list(...) in messages
    skip_short_videos: Any               # raw; the banner prints bool(...)
    truncate_max_frames: Any             # None = no truncation (never reaches RenderedDavis, see QUIRKS)
    max_train_sequences: int
    repeat_factor: int                   # max(1, int(repeat_short_sequence_factor))
    flywire_stimulus: bool
    simulation_initial_state: bool
    only_noise_visual_input: float
    noise_visual_input: float
    blank_freq: int
    blank_prefix_fraction: float         # raw getattr value
    blank_window_size_frames: int
    blank_insertion_every_n_frames: int
    tile_contrast: float
    tile_corr_strength: float
    tile_flip_prob: float


@dataclass(frozen=True)
class EdgeEditSpec:
    """Edits of the ground-truth connectivity: null edges, ablation, removal.

    n_neurons_config is ``sim.n_neurons``, the CONFIG value: legacy samples
    null edges over range(sim.n_neurons) before the neuron count is replaced by
    the network's, so a config whose n_neurons differs from the network's
    samples null edges over the wrong range (the golden cells use that on
    purpose to reach the per_column corner cases).
    """

    n_extra_null_edges: int
    null_edges_mode: str
    n_neurons_config: int
    ablation_ratio: float
    ablation_seed: int
    edge_removal_ratio: float
    edge_removal_mode: str
    edge_removal_seed: int
    edge_mask_path: str


@dataclass(frozen=True)
class NetworkSpec:
    """Which network generates the data, and with which dynamics."""

    signal_model_name: str
    ground_truth_model: str
    ensemble_id: str
    model_id: str
    extent: int                          # 15 if all_columns else 8
    edge_uncertainty: Any                # hybrid only
    conductance_checkpoint: str
    conductance_checkpoint_index: int
    conductance_exponential_euler: Any   # raw getattr value; truthiness decides
    conductance_bracket_strict: bool
    steady_state_value: Any
    params: tuple                        # sim.params as a tuple of tuples; see ode_params_list
    n_neuron_types: int
    n_input_neurons: int

    def ode_params_list(self) -> list:
        """``sim.params`` as the list of lists FlyVisODE was always given."""
        return [list(p) for p in self.params]


@dataclass(frozen=True)
class OutputSpec:
    """Arguments of data_generate_voltage, kept exactly as passed (no bool coercion:
    legacy combines ``visualize`` with ``&``)."""

    dataset: str
    visualize: Any
    run_vizualized: Any
    style: str
    erase: Any
    step: Any                            # unused, as in legacy
    save: Any
    compute_ranks: Any


@dataclass(frozen=True)
class GenerationSpec:
    """Frozen resolution of every config read of voltage generation."""

    output: OutputSpec
    network: NetworkSpec
    stimulus: StimulusSpec
    edges: EdgeEditSpec
    train_noise: NoiseSpec
    test_noise: NoiseSpec                # zeros unless noisy_test_data
    device: Any
    seed: int
    delta_t: float
    n_frames: int
    start_frame: int
    noisy_test_data: bool
    derivative_smoothing_window: int
    calcium_type: Any                    # CalciumType StrEnum
    calcium_activation: Any
    save_calcium: Any                    # phantom, default False
    n_frames_test_cap: Any               # phantom, default 0
    max_test_frames: int = MAX_TEST_FRAMES

    @classmethod
    def from_config(cls, config, *, visualize, run_vizualized, style, erase, step, device, save,
                    compute_ranks) -> "GenerationSpec":
        sim = config.simulation
        g = config.graph_model
        noisy_test = sim.noisy_test_data
        ar1_rho = float(getattr(sim, "noise_ar1_rho", 0.0))
        train_noise = NoiseSpec(
            process_std=sim.noise_model_level,
            measurement_std=sim.measurement_noise_level,
            ar1_rho=ar1_rho,
            gate_measurement_std=sim.measurement_noise_level,
        )
        test_noise = NoiseSpec(
            process_std=sim.noise_model_level if noisy_test else 0.0,
            measurement_std=sim.measurement_noise_level if noisy_test else 0.0,
            ar1_rho=ar1_rho,
            gate_measurement_std=sim.measurement_noise_level,
        )
        return cls(
            output=OutputSpec(
                dataset=config.dataset, visualize=visualize, run_vizualized=run_vizualized, style=style,
                erase=erase, step=step, save=save, compute_ranks=compute_ranks,
            ),
            network=NetworkSpec(
                signal_model_name=g.signal_model_name,
                ground_truth_model=sim.ground_truth_model,
                ensemble_id=sim.ensemble_id,
                model_id=sim.model_id,
                extent=15 if getattr(sim, "all_columns", False) else 8,
                edge_uncertainty=getattr(sim, "edge_uncertainty", 1),
                conductance_checkpoint=sim.conductance_checkpoint,
                conductance_checkpoint_index=int(getattr(sim, "conductance_checkpoint_index", 0)),
                conductance_exponential_euler=getattr(sim, "conductance_exponential_euler", True),
                conductance_bracket_strict=bool(getattr(sim, "conductance_bracket_strict", True)),
                steady_state_value=getattr(sim, "steady_state_value", 0.5),
                params=tuple(tuple(p) for p in sim.params),
                n_neuron_types=sim.n_neuron_types,
                n_input_neurons=sim.n_input_neurons,
            ),
            stimulus=StimulusSpec(
                visual_input_type=sim.visual_input_type,
                datavis_roots=tuple(sim.datavis_roots or ()),
                skip_short_videos=sim.skip_short_videos,
                truncate_max_frames=sim.truncate_max_frames,
                max_train_sequences=sim.max_train_sequences,
                repeat_factor=max(1, int(getattr(sim, "repeat_short_sequence_factor", 1))),
                flywire_stimulus=getattr(sim, "flywire_stimulus", False),
                simulation_initial_state=sim.simulation_initial_state,
                only_noise_visual_input=sim.only_noise_visual_input,
                noise_visual_input=sim.noise_visual_input,
                blank_freq=sim.blank_freq,
                blank_prefix_fraction=getattr(sim, "blank_prefix_fraction", 0.0),
                blank_window_size_frames=int(getattr(sim, "blank_window_size_frames", 0)),
                blank_insertion_every_n_frames=int(getattr(sim, "blank_insertion_every_n_frames", 0)),
                tile_contrast=sim.tile_contrast,
                tile_corr_strength=sim.tile_corr_strength,
                tile_flip_prob=sim.tile_flip_prob,
            ),
            edges=EdgeEditSpec(
                n_extra_null_edges=sim.n_extra_null_edges,
                null_edges_mode=sim.null_edges_mode,
                n_neurons_config=sim.n_neurons,
                ablation_ratio=sim.ablation_ratio,
                ablation_seed=sim.ablation_seed,
                edge_removal_ratio=sim.edge_removal_ratio,
                edge_removal_mode=getattr(sim, "edge_removal_mode", "random"),
                edge_removal_seed=sim.edge_removal_seed,
                edge_mask_path=getattr(sim, "edge_mask_path", ""),
            ),
            train_noise=train_noise,
            test_noise=test_noise,
            device=device,
            seed=sim.seed,
            delta_t=sim.delta_t,
            n_frames=sim.n_frames,
            start_frame=sim.start_frame,
            noisy_test_data=noisy_test,
            derivative_smoothing_window=sim.derivative_smoothing_window,
            calcium_type=sim.calcium_type,
            calcium_activation=sim.calcium_activation,
            save_calcium=getattr(sim, "save_calcium", False),
            n_frames_test_cap=getattr(sim, "n_frames_test", 0),
        )

    def noise_for(self, split: str) -> NoiseSpec:
        """The noise levels of ``split`` ("train" or "test")."""
        return {"train": self.train_noise, "test": self.test_noise}[split]
