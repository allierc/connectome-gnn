"""VoltageGeneration: voltage-data generation as a chain of frozen stages.

    (VoltageGeneration.from_config(config, visualize=..., ...)
        .prepare_output().seed().log_banner().make_folders()
        .build_network().load_stimuli().extract_ode_params().add_null_edges().ablate().build_ode()
        .init_geometry().steady_state().init_state().split_videos().materialize_sequences()
        .plot_previews()
        .integrate("train").derive_noisy_targets("train").tile_train()
        .reset_for_test().integrate("test").derive_noisy_targets("test")
        .restore_grad().remove_edges().save_ground_truth()
        .load_train_split().check_bracket().compute_ranks().log_trace_window().measurement_snr()
        .write_generation_log().render_figures().render_video().finish())

is ``data_generate_voltage``, in legacy order. Each stage returns a NEW
VoltageGeneration and marks its parent consumed: calling a stage on a
consumed parent raises StaleStageError. That is not ceremony. The network,
the neuron state ``x`` and the ODE parameters are mutated in place by later
stages (the frame loop updates x every frame, ``x.stimulus`` aliases the
network's stimulus buffer on CPU, the edge edits write into
``ode_params``), so two children of one parent would share and corrupt
them. A stage whose legacy guard is false is a no-op that still returns a
new object.

RNG DISCIPLINE. Every stage first restores the RNG snapshot its parent left
(a no-op in the linear chain) and captures the one it leaves; it runs inside
``RngLedger.stage(name, draws=...)``, so check mode (CGNN_RNG_LEDGER_CHECK=1)
proves every ``draws=False`` claim. No stage seeds or creates a generator
that legacy did not.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field, replace
from typing import Any

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

logger = get_logger(__name__)

GREEN, RESET = '\033[92m', '\033[0m'

# Legacy's `run`: always 0, so run_vizualized != 0 switches the per-frame
# figures and the video off.
RUN = 0


class StaleStageError(RuntimeError):
    """A stage was called on a VoltageGeneration that an earlier stage already consumed."""


class _Lineage:
    """Mutable token of one chain link: set once a stage has produced a child from it."""

    __slots__ = ("consumed_by",)

    def __init__(self):
        self.consumed_by = None


def _stage(name, *, draws):
    """Stage decorator. ``name`` / ``draws`` may be callables of (self, *args) for split stages.

    The decorated method returns the fields that change; the wrapper checks
    the parent is live, restores its RNG snapshot, runs the body inside the
    ledger, captures the new snapshot, and consumes the parent.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args):
            label = name(self, *args) if callable(name) else name
            if self._lineage.consumed_by is not None:  # golden-uncovered: stale-guard
                raise StaleStageError(
                    f"{label}: this VoltageGeneration was already consumed by {self._lineage.consumed_by}; "
                    "stages mutate the network, the state and the ODE parameters in place, so a parent "
                    "cannot be reused")
            device = self.spec.device
            self.rng.restore(device)
            with self.ledger.stage(label, draws=draws(self, *args) if callable(draws) else draws):
                changes = fn(self, *args) or {}
            self._lineage.consumed_by = label
            return replace(self, **changes, rng=RngSnapshot.capture(device), _lineage=_Lineage())
        return wrapper
    return deco


@dataclass(frozen=True)
class VoltageGeneration:
    """One link of the generation chain; see the module docstring."""

    spec: GenerationSpec
    store: DatasetStore
    ledger: RngLedger                  # shared record of the whole chain
    rng: RngSnapshot                   # global RNG state this link left behind
    fig_style: Any = None
    network: network_mod.NetworkArtifact | None = None
    stimuli: stimulus_mod.StimulusSource | None = None
    ode_params: Any = None             # mutated in place by the edge edits (see edges.py)
    edge_index: Any = None
    ablation_mask: Any = None
    ode: Any = None                    # FlyVisODE over ode_params
    geometry: initial.Geometry | None = None
    initial_state: Any = None
    x: Any = None                      # NeuronState; the frame loop updates it in place
    split: stimulus_mod.VideoSplit | None = None
    sequences: stimulus_mod.Sequences | None = None
    runs: dict = field(default_factory=dict)
    n_frames_train: int | None = None  # after tiling
    trace: diagnostics.TrainTrace | None = None
    bracket: dict | None = None
    ranks: diagnostics.Ranks | None = None
    snr: dict | None = None
    _lineage: _Lineage = field(default_factory=_Lineage, repr=False, compare=False)

    @property
    def n_neurons(self) -> int:
        """The network's neuron count (known from steady_state on)."""
        return len(self.initial_state)

    # ------------------------------------------------------------------ set-up

    @classmethod
    def from_config(cls, config, *, visualize=True, run_vizualized=0, style="color", erase=False, step=5,
                    device=None, save=True, compute_ranks=True) -> "VoltageGeneration":
        """Resolve the config into a GenerationSpec; nothing runs yet."""
        spec = GenerationSpec.from_config(config, visualize=visualize, run_vizualized=run_vizualized, style=style,
                                          erase=erase, step=step, device=device, save=save,
                                          compute_ranks=compute_ranks)
        return cls(spec=spec, store=DatasetStore(spec.output.dataset), ledger=RngLedger(device),
                   rng=RngSnapshot.capture(device))

    @_stage("prepare_output", draws=False)
    def prepare_output(self):
        """Apply the figure style globally; with ``erase``, the legacy erase.

        QUIRK: THE ERASE CHECKS x_list_* / y_list_* WITHOUT ``.zarr``, so
        y_list_*.zarr, noisy_y_list_*.zarr and the .pt / .ok / .txt side files of
        an earlier run survive it (DatasetStore.erase_legacy).
        """
        fig_style = dark_style if "black" in self.spec.output.style else default_style
        fig_style.apply_globally()
        # Erase old data if requested (prevents appending to old runs)
        if self.spec.output.erase:
            self.store.erase_legacy()
        return dict(fig_style=fig_style)

    @_stage("seed", draws=True)
    def seed(self):
        """Seed the torch (all devices) and numpy global streams with ``seed``.

        QUIRK: ``torch.random.fork_rng(devices=device)`` IS CALLED WITHOUT
        ``with``: it builds a context manager that is never entered, so nothing
        is forked or restored and the caller sees whatever global RNG state
        generation leaves behind (finish() leaves exactly that state). Kept
        verbatim. stdlib ``random`` is NOT seeded here (see add_null_edges).
        """
        torch.random.fork_rng(devices=self.spec.device)
        torch.random.manual_seed(self.spec.seed)
        np.random.seed(self.spec.seed)

    @_stage("log_banner", draws=False)
    def log_banner(self):
        """Print the model, noise and stimulus parameters generation actually uses."""
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
        _datavis_roots = list(spec.stimulus.datavis_roots) or ['<flyvis default Sintel>']
        _skip_short = bool(spec.stimulus.skip_short_videos)
        # `visual_input_type` is the renderer class (DAVIS/mixed/flash/...), not
        # the dataset identity. For video-based renderers the actual data source
        # comes from `datavis_roots` — surface its basename so the log isn't
        # misleading when e.g. visual_input_type='DAVIS' but the root is YouTube-VOS.
        if 'DAVIS' in _vis_type or 'mixed' in _vis_type:
            _data_source = ', '.join(os.path.basename(r.rstrip('/')) for r in _datavis_roots)
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

    @_stage("make_folders", draws=False)
    def make_folders(self):
        """Create graphs_data/<dataset>/ and an empty Fig/."""
        self.store.make_folders()

    # ------------------------------------------------------------------ network and stimuli

    @_stage("build_network", draws=True)
    def build_network(self):
        """Build the flyvis / conductance / hybrid network and switch autograd off (network.build_network).

        Draws torch RNG only for flywire_stimulus (the standard BoxEye's
        Conv2d init). ``torch.set_grad_enabled(False)`` stays in force until
        restore_grad, and forever if a later stage raises (legacy).
        """
        return dict(network=network_mod.build_network(self.spec))

    @_stage("load_stimuli", draws=True)
    def load_stimuli(self):
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
        return dict(stimuli=stimulus_mod.build_sources(self.spec, self.network.boxfilter))

    # ------------------------------------------------------------------ ground-truth dynamics

    @_stage("extract_ode_params", draws=False)
    def extract_ode_params(self):
        """The generating model's parameters and edge_index (dynamics.extract_ode_params)."""
        ode_params, edge_index = dynamics.extract_ode_params(self.spec, self.network.net, self.spec.device)
        return dict(ode_params=ode_params, edge_index=edge_index)

    @_stage("add_null_edges", draws=True)
    def add_null_edges(self):
        """Append ``n_extra_null_edges`` zero-weight edges (edges.add_null_edges; ode_params in place).

        QUIRK: THE DRAWS COME FROM THE GLOBAL STDLIB ``random``, WHICH GENERATION
        NEVER SEEDS. It is reproducible on DAVIS input only because building the
        DAVIS dataset reseeds it as a side effect (davis.py, random.seed of the
        shuffle seed); on Sintel input the null edges differ between processes.
        Sources and targets range over the CONFIG ``n_neurons``.
        """
        if not self.spec.edges.n_extra_null_edges > 0:
            return None
        return dict(edge_index=edges.add_null_edges(self.ode_params, self.edge_index, self.spec.edges,
                                                    self.spec.device))

    @_stage("ablate", draws=False)
    def ablate(self):
        """Zero the weights of ``ablation_ratio`` of the edges (RandomState(ablation_seed); W in place)."""
        e = self.spec.edges
        if not e.ablation_ratio > 0:
            return None
        n_edges = self.edge_index.shape[1]
        mask, n_ablate = edges.ablation_mask(n_edges, e.ablation_ratio, e.ablation_seed, self.spec.device)
        self.ode_params.W[~mask] = 0.0
        logger.info(f"ablated {n_ablate}/{n_edges} edges ({e.ablation_ratio * 100:.0f}%)")
        return dict(ablation_mask=mask)

    @_stage("build_ode", draws=lambda self: "multiple_ReLU" in self.spec.network.signal_model_name)
    def build_ode(self):
        """The FlyVisODE (dynamics.build_ode); draws torch RNG only for multiple_ReLU with params <= 0."""
        return dict(ode=dynamics.build_ode(self.spec, self.ode_params, self.edge_index, self.spec.device))

    # ------------------------------------------------------------------ initial state and sequences

    @_stage("init_geometry", draws=False)
    def init_geometry(self):
        """Positions and types of every neuron (initial.init_geometry)."""
        return dict(geometry=initial.init_geometry(self.network.net, self.spec.device))

    @_stage("steady_state", draws=False)
    def steady_state(self):
        """The network's steady state, which every split starts from (initial.steady_state)."""
        return dict(initial_state=initial.steady_state(self.spec, self.network.net, self.spec.device))

    @_stage("init_state", draws=True)
    def init_state(self):
        """Build the neuron state x (initial.init_state): item(0) into net.stimulus, torch.rand calcium.

        QUIRK: x.stimulus ALIASES net.stimulus.buffer ON CPU. It is
        ``net.stimulus().squeeze().to(device)``, a view on CPU, and
        ``Stimulus.add_input`` zeroes and refills that buffer in place, so every
        later add_input rewrites x.stimulus. With only_noise_visual_input the
        stored stimulus is therefore the rendered frame whenever the noise
        condition is false (golden cell F9). On CUDA/MPS ``.to(device)`` copies
        and there is no alias, so CPU and GPU runs differ in meaning, not only in
        rounding. Per the PI, v2 accepts the GPU semantics; the alias is not
        resolved here.
        """
        return dict(x=initial.init_state(self.network.net, self.stimuli, self.geometry, self.initial_state,
                                         self.spec.device))

    @_stage("split_videos", draws=True)
    def split_videos(self):
        """80/20 train/test split by source video (stimulus.split_videos; np.random.shuffle).

        QUIRK: with more than one ``datavis_root`` this raises AttributeError
        (CombinedVideoDataset has no arg_df); see load_stimuli.
        """
        return dict(split=stimulus_mod.split_videos(self.stimuli))

    @_stage("materialize_sequences", draws=False)
    def materialize_sequences(self):
        """Index all train and test sequences, then truncate (stimulus.materialize_sequences)."""
        return dict(sequences=stimulus_mod.materialize_sequences(self.spec, self.stimuli, self.split))

    @_stage("plot_previews", draws=False)
    def plot_previews(self):
        """shuffle_first_frames_{train,test}.png."""
        figures.plot_previews(self.sequences, self.split, self.geometry, self.store.folder, self.fig_style)

    # ------------------------------------------------------------------ the two splits

    @_stage(lambda self, split: f"integrate_{split}", draws=True)
    def integrate(self, split: str):
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
        g = self
        sequences = g.sequences.train if split == "train" else g.sequences.test
        id_fig_start = 0 if split == "train" else g.runs["train"].id_fig
        run = integrate_mod.integrate_split(
            g.spec, split, sequences=sequences, net=g.network.net, pde=g.ode, x=g.x, edge_index=g.edge_index,
            initial_state=g.initial_state, n_neurons=g.n_neurons, davis_dataset=g.stimuli.davis_dataset,
            geometry=g.geometry, store=g.store, fig_style=g.fig_style, id_fig_start=id_fig_start, run=RUN)
        changes = dict(runs={**g.runs, split: run})
        if split == "train":
            changes["n_frames_train"] = run.n_frames
        return changes

    @_stage(lambda self, split: f"derive_noisy_targets_{split}", draws=False)
    def derive_noisy_targets(self, split: str):
        """noisy_y_list_<split>.zarr when the split carries measurement noise (postprocess.noisy_derivatives).

        QUIRK: noisy_y = y + (eta[t+1] - eta[t]) / dt, optionally smoothed; IT
        NEVER INCLUDES THE PROCESS NOISE xi/dt, the defect the trainer side fixed
        in f88290e / 2a31625. The gate is the TRAIN measurement level, for the
        test split together with noisy_test_data.
        """
        spec = self.spec
        gate = spec.noise_for(split).gate_measurement_std > 0
        if split == "test":
            gate = spec.noisy_test_data and gate
        if gate:
            postprocess.noisy_derivatives(spec, self.store, self.n_neurons, split=split)

    @_stage("tile_train", draws=False)
    def tile_train(self):
        """With repeat_short_sequence_factor > 1, tile the train block that many times (postprocess.tile_train).

        QUIRK: A STALE noisy_y_list_train.zarr IS TILED TOO: it is tiled whenever
        it exists, and the legacy erase leaves one from an earlier noisy run in
        place (golden cell dirty_noerase_tile).
        """
        factor = self.spec.stimulus.repeat_factor
        if not factor > 1:
            return None
        postprocess.tile_train(self.store, factor, save_calcium=self.spec.save_calcium)
        # Reflect the post-tile length in the generation log so _have_data
        # validates the on-disk zarr without flagging it as incomplete.
        return dict(n_frames_train=self.n_frames_train * factor)

    @_stage("reset_for_test", draws=True)
    def reset_for_test(self):
        """Carry x from train to test: voltage reset in place, calcium redrawn, fluorescence zeroed."""
        integrate_mod.reset_for_test(self.x, self.initial_state, self.n_neurons, self.spec.device)

    @_stage("restore_grad", draws=False)
    def restore_grad(self):
        """Switch autograd back on (build_network switched it off)."""
        # restore gradient computation now (before any early-return paths)
        torch.set_grad_enabled(True)

    # ------------------------------------------------------------------ ground truth on disk

    @_stage("remove_edges", draws=False)
    def remove_edges(self):
        """Prune the GNN's connectivity after generation (edges.remove_edges; ode_params in place)."""
        return dict(edge_index=edges.remove_edges(self.ode_params, self.edge_index, self.spec.edges, self.store,
                                                  self.spec.output.save, self.spec.device))

    @_stage("save_ground_truth", draws=False)
    def save_ground_truth(self):
        """With ``save``: ode_params (edge_index, W, ...) and ablation_mask.pt."""
        if not self.spec.output.save:
            return None
        folder = self.store.folder
        self.ode_params.save(folder)
        print(f"{GREEN}[GENERATE] saved ode_params: edge_index={self.ode_params.edge_index.shape}  "
              f"W={self.ode_params.W.shape}  → {folder}{RESET}")
        if self.ablation_mask is not None:
            torch.save(self.ablation_mask, self.store.path("ablation_mask.pt"))

    # ------------------------------------------------------------------ diagnostics

    @_stage("load_train_split", draws=False)
    def load_train_split(self):
        """Read the (tiled) train split back for the diagnostics and figures."""
        return dict(trace=diagnostics.load_train_split(self.store))

    @_stage("check_bracket", draws=False)
    def check_bracket(self):
        """Conductance ground truth: reversal-bracket crossings (may raise, after BRACKET_*.txt)."""
        return dict(bracket=diagnostics.check_bracket(self.spec, self.ode_params, self.trace, self.store))

    @_stage("compute_ranks", draws=True)
    def compute_ranks(self):
        """With ``compute_ranks``: effective ranks (torch.svd_lowrank draws) and kinograph.png."""
        if not self.spec.output.compute_ranks:
            return None
        ranks = diagnostics.compute_ranks(self.spec, self.trace)
        act_labels, stim_labels = diagnostics.kinograph_labels(self.spec, self.ode_params, self.trace)
        figures.plot_kinograph_figure(self.spec, self.trace, ranks, act_labels, stim_labels, self.store,
                                      self.fig_style)
        return dict(ranks=ranks)

    @_stage("log_trace_window", draws=False)
    def log_trace_window(self):
        """With ``visualize``: log the warm-up skip and window of the trace plots."""
        if self.spec.output.visualize:
            diagnostics.log_trace_window(self.spec, self.trace)

    @_stage("measurement_snr", draws=False)
    def measurement_snr(self):
        """With train measurement noise: SNR values, and activity_traces_noisy.png with ``visualize``."""
        if not self.spec.train_noise.gate_measurement_std > 0:
            return None
        return dict(snr=diagnostics.measurement_snr(self.spec, self.trace, self.store,
                                                    self.geometry.node_types_int, self.fig_style))

    @_stage("write_generation_log", draws=False)
    def write_generation_log(self):
        """generation_log.txt, and the reversal report of a conductance dataset."""
        g = self
        diagnostics.write_generation_log(g.spec, g.store, diagnostics.GenerationSummary(
            n_neurons=g.n_neurons, n_frames_train=g.n_frames_train, n_frames_test=g.runs["test"].n_frames,
            n_sequences_train=len(g.sequences.train), n_sequences_test=len(g.sequences.test), split=g.split,
            bracket=g.bracket, ranks=g.ranks, snr=g.snr))

    # ------------------------------------------------------------------ figures

    @_stage("render_figures", draws=False)
    def render_figures(self):
        """With ``visualize``: activity analysis, activity.png, activity_all.png, activity_selected.png."""
        if not self.spec.output.visualize:
            return None
        figures.render_figures(self.spec, self.trace, self.x, self.n_neurons, self.store)

    @_stage("render_video", draws=False)
    def render_video(self):
        """With ``visualize`` and run_vizualized == 0: input_<id>.png and <id>.mp4 from Fig/."""
        out = self.spec.output
        if not (out.visualize & (RUN == out.run_vizualized)):
            return None
        figures.render_video(self.spec, self.store, RUN)

    def finish(self) -> None:
        """End of the chain: leave the global RNG where the last stage left it; return None, as legacy."""
        if self._lineage.consumed_by is not None:  # golden-uncovered: stale-guard
            raise StaleStageError(f"finish: already consumed by {self._lineage.consumed_by}")
        self._lineage.consumed_by = "finish"
        self.rng.restore(self.spec.device)
