"""Declarative cell matrix for the data_generate_voltage golden harness.

A cell is one call of ``data_generate_voltage`` (or of ``data_generate``, for
the dispatch cell) with a fixed config, fixed arguments and fixed fixtures. The
harness runs a cell with the BASE implementation and with the HEAD
implementation, each in a fresh subprocess, and compares everything the call
leaves behind (manifest.py).

Branch IDs in ``covers`` refer to the inventory in the refactor plan (B.. =
data_generate_voltage, R.. = _run_ode_generation, T/N = helpers, F.. =
FlyVisODE). ``coverage_report.py`` measures what the cells actually reach; the
``covers`` field is documentation, not the source of truth.

Tiers
-----
fast   every commit; target < 10 min for base + head together
full   before merge / nightly; includes the fast tier
Cells with ``requires`` run only where that resource exists (see
``requirements_met``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Base cell: synthetic DAVIS root A, flyvis_current, current ground truth,
# extent 8, 120 train frames from 3 sequences, seed 7, no visualisation, CPU.
BASE_ARGS = dict(visualize=False, run_vizualized=0, style="color", erase=False, step=5,
                 save=True, compute_ranks=True)


@dataclass(frozen=True)
class Cell:
    name: str
    tier: str
    sim: dict = field(default_factory=dict)        # overrides of config.simulation
    model: str = "flyvis_current"                  # graph_model.signal_model_name
    dataset: str = "flyvis_golden"
    args: dict = field(default_factory=dict)       # overrides of BASE_ARGS
    roots: tuple = ("davis_a",)                    # fixture video roots -> datavis_roots
    phantom: dict = field(default_factory=dict)    # getattr-only sim fields, via a proxy
    patches: tuple = ()                            # names in cell_runner.PATCHES
    env: dict = field(default_factory=dict)        # extra env; "@fixture/<rel>" is resolved
    dirty: bool = False                            # pre-populate from fixtures/dirty
    entry: str = "voltage"                         # "voltage" | "dispatch"
    requires: tuple = ()                           # subset of REQUIREMENTS
    ledger: bool = True                            # RNG-ledger check mode (see driver._child_env)
    expect: object = "ok"                          # "ok" | (exception type name, message regex)
    covers: tuple = ()
    notes: str = ""

    def call_args(self) -> dict:
        a = dict(BASE_ARGS)
        a.update(self.args)
        return a

    def extent(self) -> int:
        return 15 if self.sim.get("all_columns") else 8

    def rendering_specs(self, work) -> list[dict]:
        """RenderedDavis configurations this cell will look up (for pre-rendering)."""
        from .fixtures import davis_root
        vit = self.sim.get("visual_input_type", "DAVIS")
        if not ("DAVIS" in vit or "mixed" in vit) or self.sim.get("flywire_stimulus"):
            return []
        roots = [davis_root(work, r) for r in self.roots] if self.roots else []
        if not roots and "DATAVIS_ROOT" in self.env:
            roots = [resolve_env_value(self.env["DATAVIS_ROOT"], work)]
        roots = [r for r in roots if os.path.isdir(os.path.join(r, "JPEGImages", "480p"))]
        return [dict(root=r, extent=self.extent(),
                     max_frames=self.sim.get("truncate_max_frames", 80),
                     skip_short=self.sim.get("skip_short_videos", True)) for r in roots]

    def needs_sintel_rendering(self) -> bool:
        vit = self.sim.get("visual_input_type", "DAVIS")
        return "DAVIS" not in vit


def resolve_env_value(value: str, work) -> str:
    from .fixtures import fixtures_dir
    if isinstance(value, str) and value.startswith("@fixture/"):
        return str(fixtures_dir(Path(work)) / value[len("@fixture/"):])
    return value


def _c(name, tier="full", **kw) -> Cell:
    return Cell(name=name, tier=tier, **kw)


_MASK = "@fixture/net/kept_mask.pt"
_TWIN_SAFE = "@fixture/net/twin_safe.pt"
_TWIN_CROSS = "@fixture/net/twin_cross.pt"

CELLS: list[Cell] = [
    # ------------------------------------------------------------------ fast
    _c("F1_base", "fast", covers=("B12", "B26", "B38", "B43", "R28"), ledger=False,
       notes="the one cell that runs the RNG ledger as production does (no check, no report)"),
    _c("F2_visualize", "fast", args=dict(visualize=True),
       covers=("B47", "B53", "B55", "B56", "R26")),
    _c("F3_process_and_meas_noise", "fast", sim=dict(noise_model_level=0.05, measurement_noise_level=0.1),
       covers=("R24", "R22", "B29"),
       notes="both xi and eta on: the only fast cell whose output depends on their draw order"),
    _c("F4_meas_iid_vis", "fast", sim=dict(measurement_noise_level=0.1), args=dict(visualize=True),
       covers=("B29", "B48", "B50", "R22")),
    _c("F5_ar1_noisytest_smooth", "fast",
       sim=dict(measurement_noise_level=0.1, noise_ar1_rho=0.8, noisy_test_data=True,
                derivative_smoothing_window=3),
       covers=("B05", "B30", "B31", "R03", "R22", "N01")),
    _c("F6_edges_random", "fast",
       sim=dict(n_extra_null_edges=1000, null_edges_mode="random", ablation_ratio=0.3,
                edge_removal_ratio=0.2, edge_removal_mode="random"),
       covers=("B18", "B19", "B22", "B23", "B25", "B32", "B33", "B35", "B37", "B39")),
    _c("F7_mixed", "fast", sim=dict(visual_input_type="DAVIS mixed", n_frames=800),
       covers=("R01", "R02", "R08", "R09", "R10", "R13", "R28"),
       notes="64-frame clips, one sequence per 60-frame phase: the 4th sintel phase (13th "
             "sequence) exhausts sintel_iter over the 3 train sequences (StopIteration refetch)."),
    _c("F8_tile_mseq", "fast", sim=dict(visual_input_type="DAVIS tile_mseq"), covers=("R14",)),
    _c("F9_only_noise_init_state", "fast", sim=dict(only_noise_visual_input=0.3, simulation_initial_state=True),
       covers=("R05", "R06", "R17"),
       notes="pins the stimulus-buffer alias: for it > 0 the stored stimulus is the rendered frame"),
    _c("F10_repeat_meas", "fast", sim=dict(repeat_short_sequence_factor=3, measurement_noise_level=0.1),
       covers=("B28", "T02")),

    # ------------------------------------------------------------------ full: seeds, args
    _c("seed_0", sim=dict(seed=0)),
    _c("seed_max", sim=dict(seed=2**32 - 1)),
    _c("style_black_vis", args=dict(visualize=True, style="black"), covers=("B01",)),
    _c("runviz1_vis", args=dict(visualize=True, run_vizualized=1), covers=("B56", "R26")),
    _c("vis_no_fig0_raise", sim=dict(n_frames=3), args=dict(visualize=True),
       expect=("FileNotFoundError", r"Fig_0_000000\.png"), covers=("B56",)),
    _c("vis_noid_dataset", dataset="golden_noid", args=dict(visualize=True), covers=("B56",)),
    _c("no_ranks", args=dict(compute_ranks=False), covers=("B43", "B51")),
    _c("all_train_sequences", sim=dict(max_train_sequences=0), covers=("B26",),
       notes="32 train / 16 test sequences; the test split runs 16 x 64 = 1024 frames"),
    _c("per_type_traces_raise_vis", args=dict(visualize=True, run_vizualized=1),
       patches=("per_type_traces_raise",), covers=("B55",)),
    _c("nosave_removal_ablation", sim=dict(ablation_ratio=0.3, edge_removal_ratio=0.2),
       args=dict(save=False), covers=("B33", "B38", "B39")),
    # ------------------------------------------------------------------ erase / dirty directory
    _c("dirty_erase", dirty=True, args=dict(erase=True), covers=("B02", "B06")),
    _c("dirty_noerase_tile", dirty=True, sim=dict(repeat_short_sequence_factor=2),
       covers=("B02", "T02"), notes="stale noisy_y_list_train.zarr is tiled although meas=0"),
    # ------------------------------------------------------------------ frame counts
    _c("nframes_0", sim=dict(n_frames=0), covers=("B27",)),
    _c("nframes_1", sim=dict(n_frames=1), covers=("B45",),
       notes="one train frame: the centred activity is exactly zero"),
    _c("start_frame_10", sim=dict(start_frame=10)),
    _c("nframes_30", sim=dict(n_frames=30)),
    _c("long_vis_5100", sim=dict(n_frames=5100), args=dict(visualize=True, run_vizualized=1),
       covers=("B47",), notes="> warmup_frames + 10 = 5010 train frames"),
    # ------------------------------------------------------------------ noise
    _c("noisytest_process_only", sim=dict(noisy_test_data=True, noise_model_level=0.05), covers=("B30",)),
    _c("xi_eta_noisytest", sim=dict(noisy_test_data=True, noise_model_level=0.05, measurement_noise_level=0.1),
       covers=("B30", "B31")),
    # ------------------------------------------------------------------ null edges
    _c("null_percol_zero_deg", sim=dict(n_extra_null_edges=2000, null_edges_mode="per_column", n_neurons=3000),
       covers=("B19", "B20"), notes="first zero-out-degree source is 2842"),
    _c("null_percol_saturate", sim=dict(n_extra_null_edges=5 * 10**6, null_edges_mode="per_column", n_neurons=20),
       covers=("B21",), notes="out-degree 5 x ratio 11.5 = 58 false targets wanted, at most 19 candidates"),
    _c("null_random_attempts", sim=dict(n_extra_null_edges=5, null_edges_mode="random", n_neurons=2),
       covers=("B22", "B23")),
    _c("null_random_empty", sim=dict(n_extra_null_edges=3, null_edges_mode="random", n_neurons=1),
       covers=("B24",)),
    # ------------------------------------------------------------------ edge removal
    _c("removal_percol_clamp", sim=dict(edge_removal_ratio=0.9, edge_removal_mode="per_column"),
       covers=("B35", "B36", "B37")),
    _c("mask_exists", sim=dict(edge_removal_ratio=0.2, edge_mask_path=_MASK), covers=("B34", "B37"),
       notes="the mask keeps 70%, so 30% removed vs 20% expected: the red arm"),
    _c("mask_missing", sim=dict(edge_removal_ratio=0.2, edge_mask_path="/nonexistent/golden/kept.pt"),
       covers=("B34",)),
    # ------------------------------------------------------------------ stimulus types
    _c("flash", sim=dict(visual_input_type="DAVIS flash"), covers=("R07", "R11", "R12")),
    _c("tile_blue_noise", sim=dict(visual_input_type="DAVIS tile_blue_noise"), covers=("R15",)),
    _c("tile_blue_noise_fallback", sim=dict(visual_input_type="DAVIS tile_blue_noise"),
       patches=("build_neighbor_graph_raises",), covers=("R15",)),
    _c("blank_window", sim=dict(blank_window_size_frames=5, blank_insertion_every_n_frames=20),
       covers=("R04", "R16", "R27", "R28")),
    _c("blank_freq_noise_input", sim=dict(blank_freq=2, noise_visual_input=0.1), covers=("R18", "R19")),
    _c("blank_prefix", sim=dict(blank_prefix_fraction=0.2), covers=("B04", "R20")),
    _c("only_noise_5050", sim=dict(visual_input_type="DAVIS 50/50", only_noise_visual_input=0.3), covers=("R17",)),
    _c("init_state_no_noise", sim=dict(simulation_initial_state=True), covers=("R05", "R06")),
    _c("combined_roots", roots=("davis_a", "davis_b"), expect=("AttributeError", "arg_df"),
       covers=("B15",), notes="CombinedVideoDataset has no arg_df: combined roots cannot generate"),
    _c("env_default_root", roots=(), env=dict(DATAVIS_ROOT="@fixture/davis_a"), covers=("B13", "B51")),
    _c("missing_root", roots=(), sim=dict(datavis_roots=["/nonexistent/golden/davis"]),
       expect=("AssertionError", "video data not found"), covers=("B14",)),
    _c("no_skip_short_untruncated", sim=dict(skip_short_videos=False, truncate_max_frames=None),
       notes="truncate_max_frames never reaches RenderedDavis (MultiTaskDavis drops max_frames): "
             "it is a no-op on the DAVIS path, so this cell differs from base only by the short videos"),
    _c("mixed_davis_stopiter", roots=("davis_short",),
       sim=dict(visual_input_type="DAVIS mixed", n_frames=2205, skip_short_videos=False),
       covers=("R10",), notes="2 videos x 8 augmentations = 16 davis sequences of 20 frames, 3 per "
                              "60-frame phase: the 6th davis phase exhausts davis_iter"),
    _c("single_video_empty_train", roots=("davis_single",), sim=dict(n_frames=0), covers=("R29",),
       expect=("FileNotFoundError", "no V3 zarr"),
       notes="int(1 * 0.8) = 0 train videos: nothing is written and the diagnostics find no x_list_train"),
    # ------------------------------------------------------------------ calcium
    _c("calcium_leaky_softplus", sim=dict(calcium_type="leaky", calcium_activation="softplus"),
       expect=("NotImplementedError", "calcium_type 'leaky'"), covers=("R25",)),
    _c("calcium_leaky_relu", sim=dict(calcium_type="leaky", calcium_activation="relu"),
       expect=("NotImplementedError", "calcium_type 'leaky'"), covers=("R25",)),
    _c("calcium_leaky_tanh", sim=dict(calcium_type="leaky", calcium_activation="tanh"),
       expect=("NotImplementedError", "calcium_type 'leaky'"), covers=("R25",)),
    _c("calcium_leaky_identity", sim=dict(calcium_type="leaky", calcium_activation="identity"),
       expect=("NotImplementedError", "calcium_type 'leaky'"), covers=("R25",)),
    _c("calcium_mc_vis", sim=dict(calcium_type="multi-compartment"), args=dict(visualize=True), covers=("R26",)),
    _c("phantom_save_calcium", phantom=dict(save_calcium=True), sim=dict(repeat_short_sequence_factor=2),
       covers=("T01", "T02")),
    _c("phantom_n_frames_test", phantom=dict(n_frames_test=20)),
    # ------------------------------------------------------------------ models (FlyVisODE substrings)
    _c("model_multiple_relu_pos", model="flyvis_multiple_ReLU", sim=dict(params=[[1.0]]), covers=("F01",)),
    _c("model_multiple_relu_rand", model="flyvis_multiple_ReLU", sim=dict(params=[[-1.0]]), covers=("F01",)),
    _c("model_null", model="flyvis_NULL", covers=("F02",)),
    _c("model_tanh", model="flyvis_tanh", sim=dict(params=[[0.5]]), covers=("F03",)),
    # ------------------------------------------------------------------ conductance twin
    _c("twin_safe", sim=dict(ground_truth_model="conductance", conductance_checkpoint=_TWIN_SAFE),
       covers=("B17", "B40", "B51", "B52", "F04")),
    _c("twin_cross_strict", sim=dict(ground_truth_model="conductance", conductance_checkpoint=_TWIN_CROSS),
       expect=("ValueError", "conductance bracket"), covers=("B42",)),
    _c("twin_cross_nonstrict", sim=dict(ground_truth_model="conductance", conductance_checkpoint=_TWIN_CROSS,
                                        conductance_bracket_strict=False), covers=("B41", "B52")),
    _c("twin_forward_euler", sim=dict(ground_truth_model="conductance", conductance_checkpoint=_TWIN_SAFE,
                                      conductance_exponential_euler=False),
       expect=("ValueError", "conductance bracket"), covers=("R23",),
       notes="forward Euler on the conductance twin diverges to NaN, as the generator's comment predicts"),
    _c("twin_report_raises", sim=dict(ground_truth_model="conductance", conductance_checkpoint=_TWIN_SAFE),
       patches=("report_dataset_reversals_raises",), covers=("B52",)),
    _c("flyvis_conductance_stub", sim=dict(ground_truth_model="flyvis_conductance"),
       env=dict(PYTHONPATH_EXTRA="@fixture/stub_conductance"),
       expect=("RuntimeError", "Missing key.*nodes_E_exc_raw"), covers=("B09",),
       notes="stub package: reaches the flyvis_conductance network build; load_state_dict of the "
             "current-trained checkpoint then fails on the two reversal parameters, before B10"),
    # ------------------------------------------------------------------ dispatch
    _c("dispatch_data_generate", entry="dispatch", covers=("dispatch",)),
    # ------------------------------------------------------------------ env-gated
    _c("hybrid_e8_flywire_stimulus", model="e8_flywireRF", sim=dict(flywire_stimulus=True),
       requires=("hybrid",), covers=("B08", "B11")),
    _c("hybrid_proximal_nulls_u2", model="e8_flywireRF_proximal_nulls", sim=dict(edge_uncertainty=2),
       requires=("hybrid",), covers=("B08",)),
    _c("sintel_only_noise", roots=(), sim=dict(visual_input_type="", only_noise_visual_input=0.3),
       requires=("sintel",), covers=("B03", "B16", "R17")),
    _c("sintel_mixed", sim=dict(visual_input_type="mixed", n_frames=300), requires=("sintel",),
       covers=("B16", "R09")),
    _c("sintel_null_edges", roots=(), sim=dict(visual_input_type="", n_extra_null_edges=500,
                                               null_edges_mode="random"),
       requires=("sintel",), notes="stdlib random is unseeded here; the runner pre-seeds it"),
    _c("all_columns_extent15", sim=dict(all_columns=True, n_input_neurons=5768, n_neurons=45669),
       requires=("extent15",), covers=("B07",)),
]

CELLS_BY_NAME = {c.name: c for c in CELLS}
assert len(CELLS_BY_NAME) == len(CELLS), "duplicate cell names"

REQUIREMENTS = ("hybrid", "sintel", "extent15", "cuda")


def hybrid_dir() -> Path:
    return Path(os.environ.get("HYBRID_CONNECTOME_DIR",
                               str(Path.home() / "Projects" / "connectome-gnn" / "data" / "hybrid_connectomes")))


def sintel_dir() -> Path:
    return Path(os.environ.get("CGNN_GOLDEN_SINTEL",
                               str(Path.home() / "Projects" / "flyvis" / "data" / "SintelDataSet")))


def requirements_met(cell: Cell) -> tuple[bool, str]:
    for r in cell.requires:
        if r == "hybrid" and not hybrid_dir().is_dir():
            return False, f"hybrid tables not at {hybrid_dir()}"
        if r == "sintel" and not sintel_dir().is_dir():
            return False, f"Sintel not at {sintel_dir()}"
        if r == "cuda":
            return False, "cuda cells run on the cluster only"
        if r == "extent15" and os.environ.get("CGNN_GOLDEN_EXTENT15", "1") == "0":
            return False, "extent-15 cells disabled (CGNN_GOLDEN_EXTENT15=0)"
    return True, ""


def select(tier: str = "fast", names=None) -> list[Cell]:
    if names:
        missing = [n for n in names if n not in CELLS_BY_NAME]
        if missing:
            raise KeyError(f"unknown cells: {missing}")
        return [CELLS_BY_NAME[n] for n in names]
    if tier == "fast":
        return [c for c in CELLS if c.tier == "fast"]
    if tier == "full":
        return list(CELLS)
    raise ValueError(tier)
