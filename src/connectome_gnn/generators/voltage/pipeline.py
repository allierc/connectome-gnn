"""VoltageGeneration: the byte-identical v1 voltage-data generator.

``data_generate_voltage`` calls the fluent methods explicitly in five ordered
sections, matching the order of their implementations below::

    generator = VoltageGeneration.from_config(
        config, visualize=visualize, run_vizualized=run_vizualized, style=style,
        erase=erase, step=step, device=device, save=save, compute_ranks=compute_ranks,
    )

    # PREPROCESS
    generator = (generator.prepare_output().seed().log_banner().make_folders()
                 .build_network().load_stimuli().extract_ode_params().add_null_edges().ablate().build_ode()
                 .init_geometry().steady_state().init_state().split_videos().materialize_sequences()
                 .plot_previews())

    # PRODUCE DATASETS
    generator = (generator.integrate("train").derive_noisy_targets("train").tile_train()
                 .reset_for_test().integrate("test").derive_noisy_targets("test"))

    # POSTPROCESS
    generator = generator.restore_grad().remove_edges().save_ground_truth()

    # EVALUATE
    generator = (generator.load_train_split().check_bracket().compute_ranks()
                 .log_trace_window().measurement_snr())

    # REPORT
    generator.write_generation_log().render_figures().render_video().finish()

There is no separate phase dispatcher. Each stage mutates this generation and
returns ``self``. A reusable ``StageTracker`` enforces declared prerequisites,
single-use stages, failure state, and the stages required by ``finish()``.
The rules form a partial order: independent leaf stages are not needlessly
ordered, although ``data_generate_voltage`` keeps the legacy call order for
byte-identical side effects.

RNG DISCIPLINE. Every stage first restores the RNG snapshot its parent left
(a no-op in the linear chain) and captures the one it leaves; it runs inside
``RngLedger.stage(name, draws=...)``, so check mode (CGNN_RNG_LEDGER_CHECK=1)
proves every ``draws=False`` claim. No stage seeds or creates a generator
that legacy did not.
"""

from __future__ import annotations

import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import numpy as np
import torch

from connectome_gnn.figure_style import dark_style, default_style
from connectome_gnn.generators.voltage import diagnostics, dynamics, edges, figures, initial, postprocess
from connectome_gnn.generators.voltage import integrate as integrate_mod
from connectome_gnn.generators.voltage import network as network_mod
from connectome_gnn.generators.voltage import stimulus as stimulus_mod
from connectome_gnn.generators.voltage.rng import RngLedger, RngSnapshot
from connectome_gnn.generators.voltage.spec import GenerationSpec
from connectome_gnn.generators.voltage.store import DatasetStore
from connectome_gnn.log import get_logger
from connectome_gnn.stages import StageRule, StageStateError, StageTracker

logger = get_logger(__name__)

GREEN, RESET = "\033[92m", "\033[0m"

# Legacy's `run`: always 0, so run_vizualized != 0 switches the per-frame
# figures and the video off.
RUN = 0


StaleStageError = StageStateError


VOLTAGE_STAGE_RULES: Mapping[str, StageRule] = {
    "prepare_output": StageRule(),
    "seed": StageRule(),
    "log_banner": StageRule(),
    "make_folders": StageRule(after=frozenset(["prepare_output"])),
    "build_network": StageRule(after=frozenset(["seed"])),
    "load_stimuli": StageRule(after=frozenset(["build_network"])),
    "extract_ode_params": StageRule(after=frozenset(["build_network"])),
    "add_null_edges": StageRule(after=frozenset(["load_stimuli", "extract_ode_params"])),
    "ablate": StageRule(after=frozenset(["add_null_edges"])),
    "build_ode": StageRule(after=frozenset(["ablate"])),
    "init_geometry": StageRule(after=frozenset(["build_network"])),
    "steady_state": StageRule(after=frozenset(["build_network"])),
    "init_state": StageRule(after=frozenset(["load_stimuli", "init_geometry", "steady_state"])),
    "split_videos": StageRule(after=frozenset(["load_stimuli"])),
    "materialize_sequences": StageRule(after=frozenset(["split_videos"])),
    "plot_previews": StageRule(after=frozenset(["make_folders", "init_geometry", "materialize_sequences"])),
    "integrate_train": StageRule(
        after=frozenset(["make_folders", "build_ode", "init_state", "materialize_sequences"])
    ),
    "derive_noisy_targets_train": StageRule(after=frozenset(["integrate_train"])),
    "tile_train": StageRule(after=frozenset(["derive_noisy_targets_train"])),
    "reset_for_test": StageRule(after=frozenset(["tile_train"])),
    "integrate_test": StageRule(after=frozenset(["reset_for_test"])),
    "derive_noisy_targets_test": StageRule(after=frozenset(["integrate_test"])),
    "restore_grad": StageRule(after=frozenset(["integrate_test"])),
    "remove_edges": StageRule(after=frozenset(["integrate_train", "integrate_test"])),
    "save_ground_truth": StageRule(after=frozenset(["remove_edges"])),
    "load_train_split": StageRule(after=frozenset(["tile_train"])),
    "check_bracket": StageRule(after=frozenset(["load_train_split"])),
    "compute_ranks": StageRule(after=frozenset(["load_train_split"])),
    "log_trace_window": StageRule(after=frozenset(["load_train_split"])),
    "measurement_snr": StageRule(after=frozenset(["load_train_split"])),
    "write_generation_log": StageRule(
        after=frozenset(["save_ground_truth", "check_bracket", "compute_ranks", "measurement_snr"])
    ),
    "render_figures": StageRule(after=frozenset(["load_train_split"])),
    "render_video": StageRule(after=frozenset(["integrate_train", "integrate_test"])),
}


@dataclass
class VoltageGeneration:
    """Mutable voltage-generation workflow; see the module docstring."""

    STAGE_RULES: ClassVar[Mapping[str, StageRule]] = VOLTAGE_STAGE_RULES

    spec: GenerationSpec
    store: DatasetStore
    ledger: RngLedger
    rng: RngSnapshot
    stages: StageTracker
    fig_style: Any = None
    network: network_mod.NetworkArtifact | None = None
    stimuli: stimulus_mod.StimulusSource | None = None
    ode_params: Any = None  # mutated in place by the edge edits (see edges.py)
    edge_index: Any = None
    ablation_mask: Any = None
    ode: Any = None  # FlyVisODE over ode_params
    geometry: initial.Geometry | None = None
    initial_state: Any = None
    x: Any = None  # NeuronState; the frame loop updates it in place
    split: stimulus_mod.VideoSplit | None = None
    sequences: stimulus_mod.Sequences | None = None
    runs: dict = field(default_factory=dict)
    n_frames_train: int | None = None  # after tiling
    trace: diagnostics.TrainTrace | None = None
    bracket: dict | None = None
    ranks: diagnostics.Ranks | None = None
    snr: dict | None = None

    @property
    def n_neurons(self) -> int:
        """The network's neuron count (known from steady_state on)."""
        return len(self.initial_state)

    @classmethod
    def from_config(
        cls,
        config,
        *,
        visualize=True,
        run_vizualized=0,
        style="color",
        erase=False,
        step=5,
        device=None,
        save=True,
        compute_ranks=True,
    ) -> Self:
        """Resolve the config into a GenerationSpec; nothing runs yet."""
        spec = GenerationSpec.from_config(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            device=device,
            save=save,
            compute_ranks=compute_ranks,
        )
        return cls(
            spec=spec,
            store=DatasetStore(spec.output.dataset),
            ledger=RngLedger(device),
            rng=RngSnapshot.capture(device),
            stages=StageTracker(cls.STAGE_RULES),
        )

    @contextmanager
    def _record_stage(self, name: str, *, draws: bool) -> Generator[None, None, None]:
        """Record one generator stage while preserving its RNG contract."""
        device = self.spec.device
        with self.stages.stage(name):
            self.rng.restore(device)
            with self.ledger.stage(name, draws=draws):
                yield
            self.rng = RngSnapshot.capture(device)

    # ------------------------------------------------------------------ PREPROCESS

    def prepare_output(self) -> Self:
        """Apply the figure style globally; with ``erase``, the legacy erase.

        QUIRK: THE ERASE CHECKS x_list_* / y_list_* WITHOUT ``.zarr``, so
        y_list_*.zarr, noisy_y_list_*.zarr and the .pt / .ok / .txt side files of
        an earlier run survive it (DatasetStore.erase_legacy).
        """
        with self._record_stage("prepare_output", draws=False):
            self.fig_style = dark_style if "black" in self.spec.output.style else default_style
            self.fig_style.apply_globally()
            if self.spec.output.erase:
                self.store.erase_legacy()
        return self

    def seed(self) -> Self:
        """Seed the torch (all devices) and numpy global streams with ``seed``.

        QUIRK: ``torch.random.fork_rng(devices=device)`` IS CALLED WITHOUT
        ``with``: it builds a context manager that is never entered, so nothing
        is forked or restored and the caller sees whatever global RNG state
        generation leaves behind (finish() leaves exactly that state). Kept
        verbatim. stdlib ``random`` is NOT seeded here (see add_null_edges).
        """
        with self._record_stage("seed", draws=True):
            torch.random.fork_rng(devices=self.spec.device)
            torch.random.manual_seed(self.spec.seed)
            np.random.seed(self.spec.seed)
        return self

    def log_banner(self) -> Self:
        """Print the model, noise and stimulus parameters generation actually uses."""
        with self._record_stage("log_banner", draws=False):
            spec = self.spec
            logger.info(
                f"generating data ... {spec.network.signal_model_name}  dynamics_noise: {spec.train_noise.process_std}  "
                f"measurement_noise: {spec.train_noise.measurement_std}  seed: {spec.seed}  "
                f"steady_state_value: {spec.network.steady_state_value}"
            )
            # Stimulus / blank-prefix summary -- printed up-front so the user sees the
            # actual parameters used at generation time (vs whatever default the loader
            # might silently apply if the YAML is incomplete).
            _bpf = float(spec.stimulus.blank_prefix_fraction)
            _vis_type = spec.stimulus.visual_input_type
            _datavis_roots = list(spec.stimulus.datavis_roots) or ["<flyvis default Sintel>"]
            _skip_short = bool(spec.stimulus.skip_short_videos)
            # `visual_input_type` is the renderer class (DAVIS/mixed/flash/...), not
            # the dataset identity. For video-based renderers the actual data source
            # comes from `datavis_roots` — surface its basename so the log isn't
            # misleading when e.g. visual_input_type='DAVIS' but the root is YouTube-VOS.
            if "DAVIS" in _vis_type or "mixed" in _vis_type:
                _data_source = ", ".join(os.path.basename(r.rstrip("/")) for r in _datavis_roots)
                _renderer_str = f"renderer={_vis_type}  data_source={_data_source}"
            else:
                _renderer_str = f"renderer={_vis_type}"
            print(
                f"\033[93m[stimulus] {_renderer_str}  "
                f"blank_prefix_fraction={_bpf:.3f} "
                f"({'BLANK PREFIX ENABLED' if _bpf > 0 else 'no blank prefix'})  "
                f"skip_short_videos={_skip_short}\033[0m",
                flush=True,
            )
            print(f"\033[93m[stimulus] datavis_roots={_datavis_roots}\033[0m", flush=True)
            _ar1_rho = spec.train_noise.ar1_rho
            print(
                f"\033[93m[noise] noise_model_level={spec.train_noise.process_std}  "
                f"measurement_noise_level={spec.train_noise.measurement_std}  "
                f"noise_ar1_rho={_ar1_rho:.3f} "
                f"({'AR(1) ENABLED' if _ar1_rho > 0 else 'i.i.d.'})\033[0m",
                flush=True,
            )
        return self

    def make_folders(self) -> Self:
        """Create graphs_data/<dataset>/ and an empty Fig/."""
        with self._record_stage("make_folders", draws=False):
            self.store.make_folders()
        return self

    # ------------------------------------------------------------------ network and stimuli

    def build_network(self) -> Self:
        """Build the flyvis / conductance / hybrid network and switch autograd off (network.build_network).

        Draws torch RNG only for flywire_stimulus (the standard BoxEye's
        Conv2d init). ``torch.set_grad_enabled(False)`` stays in force until
        restore_grad, and forever if a later stage raises (legacy).
        """
        with self._record_stage("build_network", draws=True):
            self.network = network_mod.build_network(self.spec)
        return self

    def load_stimuli(self) -> Self:
        """Build the stimulus datasets (stimulus.build_sources).

        QUIRK: A COLD RENDERING CACHE SHIFTS THE TORCH STREAM. On a cache miss
        RenderedDavis / RenderedSintel build a flyvis BoxEye whose Conv2d init
        draws from the torch global RNG after ``seed``; a warm cache draws
        nothing. So the first generation on a new machine or FLYVIS_ROOT_DIR
        cannot be reproduced by rerunning it (DETERMINISM.md, "cold vs warm").
        QUIRK: ``truncate_max_frames`` IS IGNORED ON THE DAVIS PATH: it goes
        into the dataset kwargs but MultiTaskDavis never passes max_frames to
        RenderedDavis.
        QUIRK: TWO OR MORE ``datavis_roots`` CANNOT GENERATE: CombinedVideoDataset
        has no ``arg_df``, so split_videos raises AttributeError.
        The DAVIS dataset reseeds stdlib ``random`` with ``seed`` (see add_null_edges).
        """
        with self._record_stage("load_stimuli", draws=True):
            self.stimuli = stimulus_mod.build_sources(self.spec, self.network.boxfilter)
        return self

    # ------------------------------------------------------------------ ground-truth dynamics

    def extract_ode_params(self) -> Self:
        """The generating model's parameters and edge_index (dynamics.extract_ode_params)."""
        with self._record_stage("extract_ode_params", draws=False):
            self.ode_params, self.edge_index = dynamics.extract_ode_params(
                self.spec, self.network.net, self.spec.device
            )
        return self

    def add_null_edges(self) -> Self:
        """Append ``n_extra_null_edges`` zero-weight edges (edges.add_null_edges; ode_params in place).

        QUIRK: THE DRAWS COME FROM THE GLOBAL STDLIB ``random``, WHICH GENERATION
        NEVER SEEDS. It is reproducible on DAVIS input only because building the
        DAVIS dataset reseeds it as a side effect (davis.py, random.seed of the
        shuffle seed); on Sintel input the null edges differ between processes.
        Sources and targets range over the CONFIG ``n_neurons``.
        """
        with self._record_stage("add_null_edges", draws=True):
            if self.spec.edges.n_extra_null_edges > 0:
                self.edge_index = edges.add_null_edges(
                    self.ode_params, self.edge_index, self.spec.edges, self.spec.device
                )
        return self

    def ablate(self) -> Self:
        """Zero the weights of ``ablation_ratio`` of the edges (RandomState(ablation_seed); W in place)."""
        with self._record_stage("ablate", draws=False):
            e = self.spec.edges
            if e.ablation_ratio > 0:
                n_edges = self.edge_index.shape[1]
                self.ablation_mask, n_ablate = edges.ablation_mask(
                    n_edges, e.ablation_ratio, e.ablation_seed, self.spec.device
                )
                self.ode_params.W[~self.ablation_mask] = 0.0
                logger.info(f"ablated {n_ablate}/{n_edges} edges ({e.ablation_ratio * 100:.0f}%)")
        return self

    def build_ode(self) -> Self:
        """The FlyVisODE (dynamics.build_ode); draws torch RNG only for multiple_ReLU with params <= 0."""
        draws = "multiple_ReLU" in self.spec.network.signal_model_name
        with self._record_stage("build_ode", draws=draws):
            self.ode = dynamics.build_ode(self.spec, self.ode_params, self.edge_index, self.spec.device)
        return self

    # ------------------------------------------------------------------ initial state and sequences

    def init_geometry(self) -> Self:
        """Positions and types of every neuron (initial.init_geometry)."""
        with self._record_stage("init_geometry", draws=False):
            self.geometry = initial.init_geometry(self.network.net, self.spec.device)
        return self

    def steady_state(self) -> Self:
        """The network's steady state, which every split starts from (initial.steady_state)."""
        with self._record_stage("steady_state", draws=False):
            self.initial_state = initial.steady_state(self.spec, self.network.net, self.spec.device)
        return self

    def init_state(self) -> Self:
        """Build the neuron state x (initial.init_state): item(0) into net.stimulus, torch.rand calcium.

        QUIRK: x.stimulus ALIASES net.stimulus.buffer ON CPU. It is
        ``net.stimulus().squeeze().to(device)``, a view on CPU, and
        ``Stimulus.add_input`` zeroes and refills that buffer in place, so every
        later add_input rewrites x.stimulus. With only_noise_visual_input the
        stored stimulus is therefore the rendered frame whenever the noise
        condition is false (golden cell F9). On CUDA/MPS ``.to(device)`` copies
        and there is no alias, so CPU and GPU runs differ in meaning, not only in
        rounding. Per the PI, a future fix may use the GPU semantics; the
        alias is not resolved here.
        """
        with self._record_stage("init_state", draws=True):
            self.x = initial.init_state(
                self.network.net, self.stimuli, self.geometry, self.initial_state, self.spec.device
            )
        return self

    def split_videos(self) -> Self:
        """80/20 train/test split by source video (stimulus.split_videos; np.random.shuffle).

        QUIRK: with more than one ``datavis_root`` this raises AttributeError
        (CombinedVideoDataset has no arg_df); see load_stimuli.
        """
        with self._record_stage("split_videos", draws=True):
            self.split = stimulus_mod.split_videos(self.stimuli)
        return self

    def materialize_sequences(self) -> Self:
        """Index all train and test sequences, then truncate (stimulus.materialize_sequences)."""
        with self._record_stage("materialize_sequences", draws=False):
            self.sequences = stimulus_mod.materialize_sequences(self.spec, self.stimuli, self.split)
        return self

    def plot_previews(self) -> Self:
        """shuffle_first_frames_{train,test}.png."""
        with self._record_stage("plot_previews", draws=False):
            figures.plot_previews(self.sequences, self.split, self.geometry, self.store.folder, self.fig_style)
        return self

    # ------------------------------------------------------------------ PRODUCE DATASETS

    def integrate(self, split: str) -> Self:
        """Integrate one split to x_list_<split>/ and y_list_<split>.zarr (integrate.integrate_split).

        QUIRK: EXPONENTIAL EULER ON THE CURRENT-BASED MODEL TOO. With
        ``conductance_exponential_euler`` True (the default) the state update is
        ``pde.step``, the exact step at frozen coefficients, for every model
        except tanh (which falls back to Euler), while y_list stores the
        instantaneous drift f(v[t]). So even at sigma = 0, (v[t+1] - v[t]) / dt
        != y_list[t]: the finite difference is y * tau (1 - exp(-dt/tau)) / dt,
        i.e. an effective tau-hat of about tau + dt/2.
        QUIRK: MIXED MODE WALKS THE WHOLE DAVIS DATASET, TEST VIDEOS INCLUDED,
        WHILE GENERATING THE TRAIN SPLIT: its davis clips come from
        ``iter(davis_dataset)`` in dataset order, not from the train sequences.
        The frame, the measurement noise eta and the process noise xi are drawn
        in that order every frame (FrameRecord says which array carries which).
        """
        label = f"integrate_{split}"
        with self._record_stage(label, draws=True):
            sequences = self.sequences.train if split == "train" else self.sequences.test
            id_fig_start = 0 if split == "train" else self.runs["train"].id_fig
            run = integrate_mod.integrate_split(
                self.spec,
                split,
                sequences=sequences,
                net=self.network.net,
                pde=self.ode,
                x=self.x,
                edge_index=self.edge_index,
                initial_state=self.initial_state,
                n_neurons=self.n_neurons,
                davis_dataset=self.stimuli.davis_dataset,
                geometry=self.geometry,
                store=self.store,
                fig_style=self.fig_style,
                id_fig_start=id_fig_start,
                run=RUN,
            )
            self.runs = {**self.runs, split: run}
            if split == "train":
                self.n_frames_train = run.n_frames
        return self

    def derive_noisy_targets(self, split: str) -> Self:
        """noisy_y_list_<split>.zarr when the split carries measurement noise (postprocess.noisy_derivatives).

        QUIRK: noisy_y = y + (eta[t+1] - eta[t]) / dt, optionally smoothed; IT
        NEVER INCLUDES THE PROCESS NOISE xi/dt, the defect the trainer side fixed
        in f88290e / 2a31625. The gate is the TRAIN measurement level, for the
        test split together with noisy_test_data.
        """
        with self._record_stage(f"derive_noisy_targets_{split}", draws=False):
            spec = self.spec
            gate = spec.noise_for(split).gate_measurement_std > 0
            if split == "test":
                gate = spec.noisy_test_data and gate
            if gate:
                postprocess.noisy_derivatives(spec, self.store, self.n_neurons, split=split)
        return self

    def tile_train(self) -> Self:
        """With repeat_short_sequence_factor > 1, tile the train block that many times (postprocess.tile_train).

        QUIRK: A STALE noisy_y_list_train.zarr IS TILED TOO: it is tiled whenever
        it exists, and the legacy erase leaves one from an earlier noisy run in
        place (golden cell dirty_noerase_tile).
        """
        with self._record_stage("tile_train", draws=False):
            factor = self.spec.stimulus.repeat_factor
            if factor > 1:
                postprocess.tile_train(self.store, factor, save_calcium=self.spec.save_calcium)
                self.n_frames_train = self.n_frames_train * factor
        return self

    def reset_for_test(self) -> Self:
        """Carry x from train to test: voltage reset in place, calcium redrawn, fluorescence zeroed."""
        with self._record_stage("reset_for_test", draws=True):
            integrate_mod.reset_for_test(self.x, self.initial_state, self.n_neurons, self.spec.device)
        return self

    # ------------------------------------------------------------------ POSTPROCESS

    def restore_grad(self) -> Self:
        """Switch autograd back on (build_network switched it off)."""
        with self._record_stage("restore_grad", draws=False):
            torch.set_grad_enabled(True)
        return self

    # ------------------------------------------------------------------ ground truth on disk

    def remove_edges(self) -> Self:
        """Prune the GNN's connectivity after generation (edges.remove_edges; ode_params in place)."""
        with self._record_stage("remove_edges", draws=False):
            self.edge_index = edges.remove_edges(
                self.ode_params, self.edge_index, self.spec.edges, self.store, self.spec.output.save, self.spec.device
            )
        return self

    def save_ground_truth(self) -> Self:
        """With ``save``: ode_params (edge_index, W, ...) and ablation_mask.pt."""
        with self._record_stage("save_ground_truth", draws=False):
            if self.spec.output.save:
                folder = self.store.folder
                self.ode_params.save(folder)
                print(
                    f"{GREEN}[GENERATE] saved ode_params: edge_index={self.ode_params.edge_index.shape}  "
                    f"W={self.ode_params.W.shape}  → {folder}{RESET}"
                )
                if self.ablation_mask is not None:
                    torch.save(self.ablation_mask, self.store.path("ablation_mask.pt"))
        return self

    # ------------------------------------------------------------------ EVALUATE

    def load_train_split(self) -> Self:
        """Read the (tiled) train split back for the diagnostics and figures."""
        with self._record_stage("load_train_split", draws=False):
            self.trace = diagnostics.load_train_split(self.store)
        return self

    def check_bracket(self) -> Self:
        """Conductance ground truth: reversal-bracket crossings (may raise, after BRACKET_*.txt)."""
        with self._record_stage("check_bracket", draws=False):
            self.bracket = diagnostics.check_bracket(self.spec, self.ode_params, self.trace, self.store)
        return self

    def compute_ranks(self) -> Self:
        """With ``compute_ranks``: effective ranks (torch.svd_lowrank draws) and kinograph.png."""
        with self._record_stage("compute_ranks", draws=True):
            if self.spec.output.compute_ranks:
                self.ranks = diagnostics.compute_ranks(self.spec, self.trace)
                act_labels, stim_labels = diagnostics.kinograph_labels(self.spec, self.ode_params, self.trace)
                figures.plot_kinograph_figure(
                    self.spec, self.trace, self.ranks, act_labels, stim_labels, self.store, self.fig_style
                )
        return self

    def log_trace_window(self) -> Self:
        """With ``visualize``: log the warm-up skip and window of the trace plots."""
        with self._record_stage("log_trace_window", draws=False):
            if self.spec.output.visualize:
                diagnostics.log_trace_window(self.spec, self.trace)
        return self

    def measurement_snr(self) -> Self:
        """With train measurement noise: SNR values, and activity_traces_noisy.png with ``visualize``."""
        with self._record_stage("measurement_snr", draws=False):
            if self.spec.train_noise.gate_measurement_std > 0:
                self.snr = diagnostics.measurement_snr(
                    self.spec, self.trace, self.store, self.geometry.node_types_int, self.fig_style
                )
        return self

    # ------------------------------------------------------------------ REPORT

    def write_generation_log(self) -> Self:
        """generation_log.txt, and the reversal report of a conductance dataset."""
        with self._record_stage("write_generation_log", draws=False):
            diagnostics.write_generation_log(
                self.spec,
                self.store,
                diagnostics.GenerationSummary(
                    n_neurons=self.n_neurons,
                    n_frames_train=self.n_frames_train,
                    n_frames_test=self.runs["test"].n_frames,
                    n_sequences_train=len(self.sequences.train),
                    n_sequences_test=len(self.sequences.test),
                    split=self.split,
                    bracket=self.bracket,
                    ranks=self.ranks,
                    snr=self.snr,
                ),
            )
        return self

    # ------------------------------------------------------------------ figures

    def render_figures(self) -> Self:
        """With ``visualize``: activity analysis, activity.png, activity_all.png, activity_selected.png."""
        with self._record_stage("render_figures", draws=False):
            if self.spec.output.visualize:
                figures.render_figures(self.spec, self.trace, self.x, self.n_neurons, self.store)
        return self

    def render_video(self) -> Self:
        """With ``visualize`` and run_vizualized == 0: input_<id>.png and <id>.mp4 from Fig/."""
        with self._record_stage("render_video", draws=False):
            out = self.spec.output
            if out.visualize & (RUN == out.run_vizualized):
                figures.render_video(self.spec, self.store, RUN)
        return self

    def finish(self) -> None:
        """End of the chain: leave the global RNG where the last stage left it; return None, as legacy."""
        self.stages.finish()
        self.rng.restore(self.spec.device)
