from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional


# Python 3.10 compatibility (StrEnum added in 3.11)
class StrEnum(str, Enum):
    pass
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# StrEnum types for config fields

class Boundary(StrEnum):
    PERIODIC = "periodic"
    NO = "no"
    PERIODIC_SPECIAL = "periodic_special"
    WALL = "wall"

class ExternalInputType(StrEnum):
    NONE = "none"
    SIGNAL = "signal"
    VISUAL = "visual"
    MODULATION = "modulation"

class ExternalInputMode(StrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    NONE = "none"

class SignalInputType(StrEnum):
    OSCILLATORY = "oscillatory"
    TRIGGERED = "triggered"

class CalciumType(StrEnum):
    NONE = "none"
    LEAKY = "leaky"
    MULTI_COMPARTMENT = "multi-compartment"
    SATURATION = "saturation"

class CalciumActivation(StrEnum):
    SOFTPLUS = "softplus"
    RELU = "relu"
    IDENTITY = "identity"
    TANH = "tanh"

class Prediction(StrEnum):
    FIRST_DERIVATIVE = "first_derivative"
    SECOND_DERIVATIVE = "2nd_derivative"
    NEXT_ACTIVITY = "next_activity"

class Integration(StrEnum):
    EULER = "Euler"
    RUNGE_KUTTA = "Runge-Kutta"

class UpdateType(StrEnum):
    LINEAR = "linear"
    MLP = "mlp"
    PRE_MLP = "pre_mlp"
    TWO_STEPS = "2steps"
    NONE = "none"
    NO_POS = "no_pos"
    GENERIC = "generic"
    EXCITATION = "excitation"
    GENERIC_EXCITATION = "generic_excitation"
    EMBEDDING_MLP = "embedding_MLP"
    TEST_FIELD = "test_field"

class MLPActivation(StrEnum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    SOFT_RELU = "soft_relu"
    NONE = "none"

class INRType(StrEnum):
    SIREN_T = "siren_t"
    SIREN_TXY = "siren_txy"
    SIREN_ID = "siren_id"
    SIREN_X = "siren_x"
    NGP = "ngp"
    LOWRANK = "lowrank"

class DenoiserType(StrEnum):
    NONE = "none"
    WINDOW = "window"
    LSTM = "LSTM"
    GAUSSIAN_FILTER = "Gaussian_filter"
    WAVELET = "wavelet"

class GhostMethod(StrEnum):
    NONE = "none"
    TENSOR = "tensor"
    MLP = "MLP"

class Sparsity(StrEnum):
    NONE = "none"
    REPLACE_EMBEDDING = "replace_embedding"
    REPLACE_EMBEDDING_FUNCTION = "replace_embedding_function"
    REPLACE_STATE = "replace_state"
    REPLACE_TRACK = "replace_track"

class ClusterMethod(StrEnum):
    KMEANS = "kmeans"
    KMEANS_AUTO_PLOT = "kmeans_auto_plot"
    KMEANS_AUTO_EMBEDDING = "kmeans_auto_embedding"
    DISTANCE_PLOT = "distance_plot"
    DISTANCE_EMBEDDING = "distance_embedding"
    DISTANCE_BOTH = "distance_both"
    INCONSISTENT_PLOT = "inconsistent_plot"
    INCONSISTENT_EMBEDDING = "inconsistent_embedding"
    NONE = "none"

class ClusterConnectivity(StrEnum):
    SINGLE = "single"
    AVERAGE = "average"

class OdeMethod(StrEnum):
    DOPRI5 = "dopri5"
    RK4 = "rk4"
    EULER = "euler"
    MIDPOINT = "midpoint"
    HEUN3 = "heun3"

class WInitMode(StrEnum):
    RANDN = "randn"
    RANDN_SCALED = "randn_scaled"
    UNIFORM_SCALED = "uniform_scaled"
    ZEROS = "zeros"
    W_CON = "w_con"  # CX task models: init recurrent param so W_rec == W_con

class GPhiMode(StrEnum):
    MLP = "mlp"
    TANH = "tanh"
    IDENTITY = "identity"

class WOptimizerType(StrEnum):
    ADAM = "adam"
    SGD = "sgd"

class UmapClusterMethod(StrEnum):
    NONE = "none"
    DBSCAN = "dbscan"
    GMM = "gmm"

class LabelStyle(StrEnum):
    MLP = "MLP"
    GREEK = "greek"


# Sub-config schemas for NeuralGraph


class OptoTargetMode(StrEnum):
    CELL_TYPE = "cell_type"
    COLUMN = "column"
    EXPLICIT_INDICES = "explicit_indices"
    TOPK_NULLSPACE = "topk_nullspace"


class OptoRanking(StrEnum):
    NULL_DIM = "null_dim"
    LEVERAGE = "leverage"


class OptoWaveformKind(StrEnum):
    WHITE_NOISE = "white_noise"
    HEAVISIDE = "heaviside"
    IMPULSE = "impulse"
    VIDEO = "video"
    CONSTANT = "constant"


class OptoTargetSpec(BaseModel):
    """Spatial pattern of opto targets.

    Modes:
        cell_type        — every neuron of listed cell types, all columns (Gal4 analogue)
        column           — every neuron in listed retinotopic columns (single-column holography)
        explicit_indices — exactly these neuron indices (advanced; needs dataset_fingerprint)
        topk_nullspace   — auto top-k from scripts/structural_nullspace_table.json
    """
    model_config = ConfigDict(extra="ignore")

    mode: OptoTargetMode = OptoTargetMode.TOPK_NULLSPACE
    cell_types: List[str] = []
    columns: List[int] = []
    indices: List[int] = []
    k: int = 1
    ranking: OptoRanking = OptoRanking.NULL_DIM
    structural_table_json: str = "figures/structural_nullspace_table.json"

    # Per-column independence is required to break the columnar sum-zero kernel.
    # False emits a UserWarning at opto-generation time.
    column_distinct: bool = True

    # Footgun guard for explicit_indices: sha256 of the source dataset's
    # (n_neurons, neuron_type) — set at write time by add_optogenetics_stimulus.
    dataset_fingerprint: Optional[str] = None


class OptoWaveform(BaseModel):
    """Temporal waveform applied to each (independent if column_distinct) target.

    Composition (per target):
        u_target(t) = base_waveform(t) + noise_level * xi(t)
    where xi ~ N(0,1) is i.i.d. per (target, t). noise_level applies for every
    kind. For kind='white_noise' the base contribution is zero and noise_level
    drives the signal.

    Units: amplitude and noise_level are on the same scale as state.stimulus,
    directly comparable to SimulationConfig.noise_model_level.
    """
    model_config = ConfigDict(extra="ignore")

    kind: OptoWaveformKind = OptoWaveformKind.WHITE_NOISE

    # Base amplitude. None = per-target auto-calibration to 0.5 * lambda_max(type)
    # read from the structural nullspace JSON.
    amplitude: Optional[float] = None

    # Universal additive Gaussian noise on top of base waveform.
    noise_level: float = 0.0

    seed: int = 0

    # heaviside-only: ON for frames_on frames, OFF for frames_on frames,
    # repeat. Full period = 2 * frames_on. Default 35 frames means
    # 35 ON / 35 OFF / ... (period 70 frames ~= 1.4 s at dt=0.02 s).
    # Set to 0 for a one-shot DC step (always ON).
    frames_on: int = 35

    # heaviside-only (column_distinct=True): if False (default), each column
    # draws a single per-column gain ~U(0,1) at simulation start and that
    # gain persists for the entire trajectory (column-identity label). If
    # True, a fresh per-column amplitude ~U(0,1) is drawn at every flip,
    # so the per-column gain varies across ON intervals — removes the
    # column-fingerprint and isolates the temporal-decorrelation effect
    # from the persistent-gain effect.
    resample_amplitude_per_transition: bool = False

    # impulse-only
    pulse_width_frames: int = 5
    pulse_period_frames: int = 50

    # video-only (experimental — off-manifold replay)
    video_path: Optional[str] = None


class OptogeneticsConfig(BaseModel):
    """Master config block for the optogenetic-perturbation pipeline.

    enabled=False (default) keeps existing pipelines untouched. When enabled,
    add_optogenetics_stimulus re-simulates the source dataset's forward model
    with this opto current added, using the same seed for matched comparison.
    """
    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    target: OptoTargetSpec = OptoTargetSpec()
    waveform: OptoWaveform = OptoWaveform()

    # Source dataset (must already exist on disk; the opto pass re-simulates from it).
    source_dataset: Optional[str] = None
    output_suffix: str = "_opto"


class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dimension: int = 2
    n_frames: int = 1000  # number of simulation time steps; 0 = use each source frame exactly once (no reuse)
    start_frame: int = 0
    seed: Annotated[int, Field(ge=0, lt=2**32)] = 42

    # Teacher-model voltage generation: when set, `data_generate` rolls out
    # the TaskRNN described by this YAML over fresh task stimuli and saves
    # the hidden-state trajectory as voltage.zarr (compatible with the
    # standard data_generate_voltage output). Empty = use the conventional
    # ODE-based / flyvis generators.
    task_model_config_path: str = ""
    # CX teacher-voltage variant only: list of cell-type tokens defining the
    # opto-style input target. Accepted tokens: "PEN_a" / "PEN_b" (= L ∪ R
    # subpops), specific subpop names ("PENa_L", "PENa_R", "PENb_L", "PENb_R"),
    # or any entry of the hemibrain `type_names` list (e.g. "EPG", "Delta7").
    # Only the resolved input rows of `stimulus.zarr` carry the drive; the
    # rest are zeroed. None defaults to ["PEN_a", "PEN_b"] inside the CX
    # variant of _generate_voltage_from_task_model.
    input_cell_types: Optional[List[str]] = None

    # GCaMP indicator used for the voltage->calcium observation model (registry
    # name in connectome_gnn.models.gcamp, e.g. gcamp6f/6s/7f/8f/8m). Selects the
    # rise/decay kinetics when rendering calcium from rolled-out voltage.
    gcamp_kernel: str = "gcamp7f"

    model_id: str = "000"
    ensemble_id: str = "0000"

    sub_sampling: int = 1
    delta_t: float = 1

    boundary: Boundary = Boundary.PERIODIC
    min_radius: float = 0.0
    max_radius: float = 0.1

    n_neurons: int = 1000
    n_neuron_types: int = 5
    n_input_neurons: int = 0
    n_edges: int = 0
    max_edges: float = 1.0e6
    n_extra_null_edges: int = 0
    null_edges_mode: str = "per_column"  # "random" or "per_column" (per pre-synaptic neuron)
    edge_removal_ratio: float = 0.0  # fraction of edges to remove before saving (0.0-1.0)
    edge_removal_mode: str = "random"  # "random" or "per_column"
    edge_removal_seed: Annotated[int, Field(ge=0, lt=2**32)] = 42      # RNG seed for reproducible removal
    edge_mask_path: str = ""         # path to precomputed kept_edge_indices.pt; if set and exists, reused instead of recomputing
    ablation_ratio: float = 0.0   # fraction of edges to ablate (0.0-1.0)
    ablation_seed: Annotated[int, Field(ge=0, lt=2**32)] = 42       # RNG seed for reproducible ablation

    baseline_value: float = -999.0
    shuffle_neuron_types: bool = False

    noise_visual_input: float = 0.0
    only_noise_visual_input: float = 0.0
    visual_input_type: str = ""  # for flyvis experiments
    datavis_roots: list[str] = []  # list of dataset roots (each contains JPEGImages/480p/); empty list uses default get_datavis_root_dir()
    skip_short_videos: bool = True  # skip videos with fewer frames than chunk size (n_frames in video_config)
    truncate_max_frames: Optional[int] = Field(default=80, gt=0)  # crop video clips to this length (frames); None = no truncation. Default preserves prior hardcoded 80.
    max_train_sequences: int = 0  # limit train sequences (0 = use all); reduces generation time proportionally
    blank_freq: int = 0  # Periodic-blank period: 0=off; N>=2 zeros stimulus on every Nth frame (data_idx % N == 0)
    blank_prefix_fraction: float = 0.0  # fraction of each sequence to blank at the start (e.g. 0.1 = first 10% frames zero stimulus)
    # DAVIS blank-window injection: after every `blank_insertion_every_n_frames` real video frames
    # (counted across video boundaries), inject `blank_window_size_frames` consecutive zero-stimulus
    # frames. When active, sim.n_frames counts only real video frames; injected blanks add on top.
    # Both must be specified together (both > 0) or neither (both == 0); enforced by validator.
    blank_window_size_frames: int = Field(default=0, ge=0)
    blank_insertion_every_n_frames: int = Field(default=0, ge=0)
    simulation_initial_state: bool = False
    # flyvis net.steady_state(value=…) passed during pre-warmup. Default 0.5 reproduces the
    # flyvis implicit default used in all prior experiments (constant 0.5 luminance during the
    # 2-second pre-stimulus window, giving an illuminated steady state as initial condition).
    # Set to 0.0 to pre-warm under zero luminance instead — equivalent to starting from the
    # network's true resting state with no silent-stimulus period.
    steady_state_value: float = 0.5
    all_columns: bool = False  # if True, use all 721 retinotopic columns (extent=15); default uses 217 (extent=8)
    edge_uncertainty: int = 1  # zero-edge radius multiplier (1–3); only used by hybrid zeroedge variants
    # If True, render visual stimuli onto the FlyWire column lattice carried
    # by the hybrid connectome (instead of flyvis's regular hex disk).
    # Required for hybrid networks with all_columns=True because their input
    # column count differs from BoxEye(extent=15)'s 721 hexals.
    flywire_stimulus: bool = False


    # external input configuration
    external_input_type: ExternalInputType = ExternalInputType.NONE
    external_input_mode: ExternalInputMode = ExternalInputMode.NONE
    permutation: bool = False  # whether to apply random permutation to external input

    # signal input parameters (external_input_type == "signal")
    signal_input_type: SignalInputType = SignalInputType.OSCILLATORY
    oscillation_max_amplitude: float = 1.0
    oscillation_frequency: float = 5.0

    # triggered oscillation parameters (signal_input_type == "triggered")
    triggered_n_impulses: int = 5  # number of impulse events
    triggered_n_input_neurons: int = 10  # number of neurons receiving impulse input per event
    triggered_impulse_strength: float = 5.0  # base strength of impulse (will vary randomly)
    triggered_min_start_frame: int = 50  # minimum frame for first trigger
    triggered_max_start_frame: int = 150  # maximum frame for first trigger (ignored if n_impulses > 1)
    triggered_duration_frames: int = 200  # duration of oscillation response per impulse
    triggered_amplitude_range: list[float] = [0.5, 2.0]  # min/max amplitude multiplier
    triggered_frequency_range: list[float] = [0.5, 2.0]  # min/max frequency multiplier

    tile_contrast: float = 0.2
    tile_corr_strength: float = 0.0   # correlation knob for tile_mseq / tile_blue_noise
    tile_flip_prob: float = 0.05      # per-frame random flip probability
    tile_seed: Annotated[int, Field(ge=0, lt=2**32)] = 42

    n_nodes: Optional[int] = None
    node_value_map: Optional[str] = "input_data/pattern_Null.tif"

    adjacency_matrix: str = ""
    short_term_plasticity_mode: str = "depression"

    # AdEx spiking model parameters
    adex_dt: float = 0.2              # ms — integration timestep for AdEx (0.2ms default from Zerlaut)
    adex_stim_scale: float = 1.0      # pA per unit stimulus — converts visual input to current
    adex_I_bias: float = 0.0          # pA — constant bias current injected into all neurons

    # Hodgkin-Huxley model parameters
    hh_substeps: int = 50             # number of Euler substeps per stimulus frame
    hh_stim_scale: float = 50.0       # uA/cm^2 per unit stimulus
    hh_I_bias: float = 3.0            # uA/cm^2 — tonic drive (subthreshold)
    hh_w_scale: float = 2.0           # global W multiplier (connectome weights calibrated for graded model)

    # Connconstr model parameters (Beiran & Litwin-Kumar 2023, Fig 5)
    connconstr_datapath: str = ""      # path to external data files (hemibrain CSVs, goldman_data/, etc.)
    connconstr_model: str = ""         # which model: drosophila_cx, larva, zebrafish
    connconstr_n_trials: int = 50      # number of stimulus trials (CX model)
    # Drosophila CX input streams. "full" = Hulse Model A (EPG landmark cues +
    # PEN_a angular velocity). "velocity_only" drops the landmark cues after
    # cx_seed_frames so EPG activity comes from recurrence rather than injection;
    # the seed window exists because the velocity drive is rotationally symmetric
    # and cannot break symmetry to form a bump on its own.
    cx_drive: str = "full"             # "full" | "velocity_only"
    cx_seed_frames: int = 100          # landmark-cue seed window for velocity_only
    connconstr_use_pretrained: bool = True  # use pre-trained teacher params if available

    connectivity_file: str = ""
    connectivity_init: list[float] = [-1]
    connectivity_filling_factor: float = 1
    connectivity_type: str = "none"  # none, Lorentz, Gaussian, uniform, chaotic, ring attractor, low_rank, successor, null, Lorentz_structured_X_Y
    connectivity_rank: int = 1
    connectivity_parameter: float = 1.0

    Dale_law: bool = False
    Dale_law_factor: float = 0.5  # fraction of excitatory (positive) columns, rest are inhibitory

    excitation_value_map: Optional[str] = None
    excitation: str = "none"

    params: list[list[float]]
    func_params: list[tuple] = None

    phi: str = "tanh"
    tau: float = 1.0
    sigma: float = 0.005

    calcium_type: CalciumType = CalciumType.NONE
    calcium_activation: CalciumActivation = CalciumActivation.SOFTPLUS
    noise_model_level: float = 0.0  # process noise added during dynamics simulation
    measurement_noise_level: float = 0.0  # observation noise saved separately in noise.zarr
    # Stationary AR(1) coefficient on measurement noise: 0 = i.i.d. (default),
    # 0.5 ~ GCaMP6f kinetics at dt=20ms, 0.99 = highly temporally correlated.
    # Recursion: eta(t+1) = rho*eta(t) + sqrt(1 - rho**2) * gamma * xi(t),
    # ξ ~ N(0,1) i.i.d. -- preserves marginal Var(eta) = gamma**2 across t.
    noise_ar1_rho: float = 0.0
    # Generate only n_frames // factor unique simulated frames, then tile that
    # short trajectory `factor` times across the train zarr (voltage / stimulus
    # / noise / y). Test data generation is unaffected. Use to study the role
    # of stimulus diversity vs noise averaging at fixed total dataset length.
    repeat_short_sequence_factor: int = 1
    noisy_test_data: bool = False  # if True, test split uses the same noise levels as train; default keeps test deterministic
    derivative_smoothing_window: int = 1  # temporal smoothing window for noisy derivatives (1 = no smoothing)
    # --- NeurIPS-2026 rebuttal: model-misspecification knobs (flyvis/graded ODE) ---
    # All default to base behaviour; the flyvis Euler path is byte-identical when
    # n_generation_substeps==1, finite_difference_target==False, adapt_g==0.
    # Test 1 (Δt mismatch): integrate the flyvis ODE with M Euler substeps of
    # h = delta_t / M per OBSERVED frame (finer than the fixed delta_t inference
    # step), with process noise scaled by 1/sqrt(M) so the per-observation noise
    # variance is unchanged. The observed cadence stays at delta_t.
    n_generation_substeps: int = 1
    # Test 1: store the target y as the OBSERVED one-step finite difference
    # (v[t+delta_t] - v[t]) / delta_t instead of the analytic derivative pde(x_t),
    # so the GNN/oracle is trained on what a delta_t observer can actually measure
    # (curvature-biased when the true trajectory is integrated more finely).
    finite_difference_target: bool = False
    # Test 3 (unobserved adaptation current): add a slow per-neuron adaptation
    # current -adapt_g * c_i to the flyvis ODE, with c_i integrated as
    # dc_i/dt = (v_i - c_i) / adapt_tau (tau in ms). c_i is NEVER observed or
    # written, so it is a latent variable outside the graph that violates the
    # first-order-in-observables assumption. adapt_g==0 disables it entirely.
    adapt_g: float = 0.0
    adapt_tau_ms: float = 200.0

    pos_init: str = "uniform"
    dpos_init: float = 0

    # Optogenetic perturbation pipeline. Disabled by default — enabling
    # triggers a separate code path (see generators/optogenetics.py) that
    # re-simulates the source dataset with an additive optogenetics_stimulus
    # current and writes a new dataset under config.dataset.
    optogenetics: OptogeneticsConfig = OptogeneticsConfig()

    @model_validator(mode="after")
    def _validate_blank_window_injection(self) -> "SimulationConfig":
        l = self.blank_window_size_frames
        m = self.blank_insertion_every_n_frames
        if (l > 0) != (m > 0):
            raise ValueError(
                "blank_window_size_frames and blank_insertion_every_n_frames must be "
                f"specified together (both > 0) or neither (both == 0); got "
                f"blank_window_size_frames={l}, blank_insertion_every_n_frames={m}"
            )
        if l > 0 and (self.blank_freq > 0 or self.blank_prefix_fraction > 0.0):
            raise ValueError(
                "blank-window injection (blank_window_size_frames / "
                "blank_insertion_every_n_frames) is mutually exclusive with "
                f"blank_freq (got {self.blank_freq}) and blank_prefix_fraction "
                f"(got {self.blank_prefix_fraction}); disable those to use it"
            )
        if l > 0:
            vit = self.visual_input_type
            if "DAVIS" not in vit:
                raise ValueError(
                    "blank-window injection requires visual_input_type to contain "
                    f"'DAVIS'; got visual_input_type={vit!r}"
                )
            for incompat in ("flash", "mixed", "tile_mseq", "tile_blue_noise"):
                if incompat in vit:
                    raise ValueError(
                        f"blank-window injection is not supported with visual_input_type "
                        f"containing {incompat!r}; got visual_input_type={vit!r}"
                    )
        return self


class ClaudeConfig(BaseModel):
    """Configuration for Claude-driven exploration experiments."""
    model_config = ConfigDict(extra="ignore")

    n_epochs: int = 1  # number of epochs per iteration
    data_augmentation_loop: int = 100  # data augmentation loop count
    n_iter_block: int = 24  # number of iterations per simulation block
    n_parallel: int = 4  # number of parallel config slots per batch (GNN_LLM_parallel.py)
    node_name: str = "a100"  # cluster GPU node: h100, a100, or l4
    generate_data: bool = False  # generate new simulation data before each training iteration
    test_robustness_seed: bool = False  # agent-triggered: re-generate data with new seeds for this batch only (pipeline resets after use)
    training_time_target_min: int = 60  # target training time per iteration in minutes (for LLM guidance)
    total_steps: int = 20000  # INR training iterations (used by INR_LLM.py)
    interaction_code: bool = False  # enable Phase A interactive code sessions at block boundaries
    case_study: str = ""  # case study identifier (e.g. "measurement_noise")
    case_study_brief: str = ""  # description of the case study for LLM code briefs
    claude_call_timeout_min: int = 4  # hard wall-clock cap per Claude CLI call (BATCH 0 + analysis)


class ClaudeCodeConfig(BaseModel):
    """Block-level code-change exploration config (GNN_LLM_code.py)."""
    model_config = ConfigDict(extra="ignore")

    block_themes: Optional[List[str]] = None
    phase_time_limits: Optional[Dict[str, int]] = None
    primary_metric: Optional[str] = None


class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    particle_model_name: str = ""
    cell_model_name: str = ""
    mesh_model_name: str = ""
    signal_model_name: str = ""
    prediction: Prediction = Prediction.SECOND_DERIVATIVE
    integration: Integration = Integration.EULER

    aggr_type: str
    embedding_dim: int = 2

    field_type: str = ""
    field_grid: Optional[str] = ""

    input_size: int = 1
    output_size: int = 1
    hidden_dim: int = 1
    n_layers: int = 1

    input_size_2: int = 1
    output_size_2: int = 1
    hidden_dim_2: int = 1
    n_layers_2: int = 1


    input_size_decoder: int = 1
    output_size_decoder: int = 1
    hidden_dim_decoder: int = 1
    n_layers_decoder: int = 1

    input_size_encoder: int = 1
    output_size_encoder: int = 1
    hidden_dim_encoder: int = 1
    n_layers_encoder: int = 1

    # Positive control for the "can g_phi discard useless inputs?" experiment.
    # Appends this many PURE NOISE columns to g_phi's per-edge input. They carry no
    # information about the target by construction, so a credit-assignment mechanism
    # that works must drive them to zero. Distinguishes two failure modes that the
    # regularisation sweep alone cannot separate:
    #   discards noise AND v_i/a_i -> credit assignment works
    #   discards noise, KEEPS v_i  -> v_i is kept for a reason (it is redundant with
    #                                 f_theta's own v_i input, not uninformative)
    #   keeps even the noise       -> the regulariser is not biting at all
    # input_size must be widened accordingly (6 + n_g_phi_noise_inputs for
    # flyvis_conductance). 0 = off, and nothing is appended.
    n_g_phi_noise_inputs: int = 0
    g_phi_positive: bool = False

    update_type: UpdateType = UpdateType.NONE

    # TaskRNN: shape of W_in / W_out (Hulse path-integration model).
    # "matrix" → learnable Linear (Hulse default). "mlp" → small MLP reusing
    # `hidden_dim` and `n_layers` above.
    input_proj: Literal["matrix", "mlp"] = "matrix"
    output_proj: Literal["matrix", "mlp"] = "matrix"
    # DrosophilaCxTaskRNN / DrosophilaCxTaskGNN: when True, the decoder W_out
    # reads only from the 46 EPG neurons (rows 0..45 in the model's neuron
    # ordering) instead of all 156. Matches Hulse 2021 nn_fig5_drosophilaCx_*.py
    # (lines 574-577: wout[0:46, ...] non-zero, rest zero, train_wout=False).
    # Here W_out stays learnable but its input is restricted to the 46 EPG
    # firing rates, so the optimiser is forced to put the heading code in
    # EPG cells (the biological prior). Default False keeps the legacy
    # behaviour (all 156 neurons connected to the decoder).
    output_from_epg_only: bool = False
    # Zebrafish-specific counterpart of ``output_from_epg_only``: restricts
    # the readout to the first ``n_epg = 443`` dIPN cells (IPNd* + IPNds*
    # = the r1π HD ring per Petrucco 2023). Read by ``ZebrafishHdTaskRNN``;
    # ignored by the drosophila models so fly yamls keep using the
    # ``output_from_epg_only`` flag unchanged.
    output_from_dipn_only: bool = False
    # CortexTaskRNN: whether the readout sees the firing rate r = σ(h) (default,
    # historical) or the raw subthreshold h (Yang 2019 convention). False matches
    # the paper's VanillaLeakyRNN exactly.
    readout_uses_sigma: bool = True
    # TaskGNN-only: when True, the effective per-edge weight is `|w| · sign_GT`
    # (Dale-conformant; only magnitudes are learned). When False, the per-edge
    # weight is learned with free sign — the GT connectome topology is still
    # enforced via the mask, but Dale's law is relaxed. DrosophilaCxTaskRNN uses
    # `wrec_param` below instead.
    lock_edge_signs: bool = True
    # NeuralGNN voltage-recovery path (Branch 0): hard Eq-10 sign-lock against
    # the GT connectome. When True, the per-edge effective weight in the
    # message is `|W| · sign_GT` (sign from ode_params.W, set by data_train_gnn
    # after model build) — only magnitudes are learned. Distinct from the
    # emergent `coeff_W_sign`/`dale_law` (which pick each neuron's own sign,
    # not the connectome's). Default False keeps every existing neural_gnn
    # config byte-equivalent.
    lock_edge_signs_from_connectome: bool = False
    # DrosophilaCxTaskRNN: recurrent-matrix parameterisation.
    #   "edge_magnitude" — W_rec = |S| ⊙ sign(W_con); sparsity locked to W_con,
    #                      per-edge sign locked to connectome (Dale).
    #   "edge_free"      — W_rec = S ⊙ mask(W_con); sparsity locked to W_con,
    #                      per-edge sign free.
    #   "column_dale"    — W_rec = |S| ⊙ col_sign[None,:]; dense N×N, the only
    #                      constraint is that every entry in pre-column j shares
    #                      sign(W_con[:, j].sum()). Diagonal still zero.
    wrec_param: Literal["edge_magnitude", "edge_free", "column_dale"] = "edge_magnitude"
    # TaskRNN: anatomical gate on the velocity column of W_in.
    # "pen_only"    — zero W_in[:, 0] outside PENa/PENb rows; per-unit
    #                 weights stay free.
    # "pen_4scalar" — strict Hulse 2025: 4 learnable scalars (L/R × PENa/PENb)
    #                 broadcast onto their subpopulations; sign initialised
    #                 opposite for L vs R.
    # "none"        — W_in fully free (default).
    velocity_gate: Literal["none", "pen_only", "pen_4scalar",
                           "pen_artr_ptipn1",
                           "pen_artr_ptipn1_propriocep",
                           "pen_pfn",
                           "pen_propriocep",
                           "pen_propriocep_swap"] = "none"
    # Sign-lock the bilateral velocity-gate scalars so the left port is ≤0 and
    # the right port is ≥0 (effective = ∓softplus(raw), magnitude free). This
    # forces the L/R afferents to be driven in antiphase — matching the
    # recorded ARTR L/R anti-correlation — instead of letting a task-only
    # model collapse to a degenerate same-sign (symmetric) gate. True by
    # default; set false for the unconstrained legacy gate.
    sign_constrain_gate: bool = True

    # TaskRNN (cortex/free-W mode): explicit recurrent population size + I/O
    # dimensions. For sign_locked (CX) mode these are derived from the
    # connectome and these fields are ignored.
    n_units: int = 0
    n_input: int = 0
    n_output: int = 0
    # W parameterisation. "sign_locked" → W_rec = |S| ⊙ W_con (CX, Hulse).
    # "free" → W_rec is a plain (N, N) Parameter (cortex/Yang, no biological
    # prior, no Dale).
    W_param: Literal["sign_locked", "free"] = "sign_locked"
    # Recurrent activation σ in r = σ(h). "sigmoid" is the Hulse paper
    # default; "relu" / "softplus" are Yang's defaults for cortex tasks.
    # "tanh"/"leaky_relu" allow a signed rate code (σ(h) not constrained ≥0).
    recurrent_activation: Literal["sigmoid", "relu", "tanh", "softplus", "leaky_relu"] = "sigmoid"
    # Activation checkpointing over the GNN rollout: recompute the per-step
    # (B, E, hidden) edge messages in backward instead of storing them for
    # every one of the up-to-800 rollout steps. Bounds the GNN's rollout
    # activation memory to ~O(1) in T so it fits a single GPU (without it the
    # message-passing model OOMs by T≈100 on a 22 GB L4). Used only by the
    # GNN task models (DrosophilaCxTaskGNN); the dense RNN ignores it.
    gnn_grad_checkpoint: bool = True
    # Optional image-derived binary mask on W_rec — a fun structural prior
    # to test capacity / sparsity trade-offs. The image is resized to N×N
    # and thresholded at its median to produce a 0/1 mask; W_rec is
    # multiplied by this mask, so dark pixels become forbidden connections.
    # Path is absolute or resolved via connectome_gnn.utils.config_path().
    w_mask_image_path: str = ""
    # Dynamics constants for TaskRNN's Euler integration. CX (sign_locked
    # mode) overrides these from task.path_integration; cortex (free mode)
    # reads them from here directly. Defaults match the Hulse paper.
    tau: float = 0.1
    dt: float = 0.01

    MLP_activation: MLPActivation = MLPActivation.RELU
    zero_init_output: bool = False  # zero-init final layer so model starts predicting dvdt=0
    add_skip_layers: bool = False  # linear skip connection at each hidden layer
    add_diagonal: bool = False  # learnable per-neuron diagonal term: dv_i/dt += alpha_i * v_i
    add_residual: bool = False  # ResNet-style residual connections across hidden layers


    input_size_update: int = 3
    n_layers_update: int = 3
    hidden_dim_update: int = 64
    output_size_update: int = 1

    kernel_type: str = "mlp"

    input_size_nnr: int = 3
    n_layers_nnr: int = 5
    hidden_dim_nnr: int = 128
    output_size_nnr: int = 1
    outermost_linear_nnr: bool = True
    omega: float = 80.0

    input_size_nnr_f: int = 3
    n_layers_nnr_f: int = 5
    hidden_dim_nnr_f: int = 128
    output_size_nnr_f: int = 1
    outermost_linear_nnr_f: bool = True
    omega_f: float = 80.0
    omega_f_learning: bool = False  # make omega learnable during training

    nnr_f_xy_period: float = 1.0
    nnr_f_T_period: float = 1.0

    # Hidden neuron INR — learns voltages of silenced neurons jointly with GNN.
    # "none"      : zero-silencing (original behaviour, no INR)
    # "siren_t"   : SIREN(t) -> (n_hidden,)  — independent signal per neuron
    # "siren_txy" : SIREN(x,y,t) -> scalar   — spatially-correlated field
    # "ngp_t"     : MultiResTemporalGrid(t) -> (n_hidden,)  — no waterbed, faster
    inr_type_hidden: str = "none"
    hidden_neuron_fraction: float = 0.0  # fraction of non-retina neurons to hide; 0 = disabled
    # SIREN hidden params
    hidden_dim_nnr_hidden: int = 2048
    n_layers_nnr_hidden: int = 4
    omega_hidden: float = 4096.0
    outermost_linear_nnr_hidden: bool = True
    nnr_hidden_T_period: float = 64000.0  # time normalisation for SIREN (raw frame index)
    # NGP hidden params (MultiResTemporalGrid) — used when inr_type_hidden = "ngp_t"
    ngp_hidden_n_levels: int = 24
    ngp_hidden_n_features_per_level: int = 4
    ngp_hidden_base_resolution: int = 16
    ngp_hidden_per_level_scale: float = 1.4
    ngp_hidden_mlp_width: int = 512
    ngp_hidden_mlp_layers: int = 4

    # NGP hidden spatial branch — when ngp_hidden_spatial=True the temporal grid
    # above is wrapped in a MultiResSpatioTemporalGrid that also queries a 2-D
    # MultiResHexGrid2D at every neuron position pos[i]. The two feature streams
    # are concatenated before the decoder MLP, so neighbouring columns share
    # spatial grid cells (retinotopic smoothness prior). Only used with
    # inr_type_hidden = "ngp_t".
    ngp_hidden_spatial: bool = False
    ngp_hidden_spatial_n_levels: int = 6
    ngp_hidden_spatial_n_features_per_level: int = 4
    ngp_hidden_spatial_base_resolution: int = 4
    ngp_hidden_spatial_per_level_scale: float = 1.5
    ngp_hidden_xy_period: float = 1.0  # divides pos before mapping to [0, 1]^2

    # Factorized output head for NGP-T / SIREN-T: parallel low-rank path that
    # mixes a per-neuron identity factor with time features, added to the
    # shared decoder output. rank=0 disables (current behavior).
    # from_a=True projects the GNN's self.a (shape (n_neurons, embedding_dim))
    # so the NGP shares neuron identity with the GNN; =False uses a dedicated
    # nn.Parameter (n_neurons, rank) with independent capacity.
    ngp_factorized_rank: int = 0
    ngp_factorized_from_a: bool = True

    # INR type for external input learning
    # siren_t: input=t, output=n_neurons (current implementation, works for n_neurons < 100)
    # siren_id: input=(t, id), output=1 (scales better for large n_neurons)
    # siren_x: input=(t, x, y), output=1 (uses neuron positions)
    # ngp: instantNGP hash encoding
    # lowrank: low-rank matrix factorization U @ V (not a neural network)
    inr_type: INRType = INRType.SIREN_T

    # LowRank factorization parameters
    lowrank_rank: int = 64  # rank of the factorization (params = rank * (n_frames + n_neurons))
    lowrank_svd_init: bool = True  # initialize with SVD of the data

    # InstantNGP (hash encoding) parameters
    ngp_n_levels: int = 24
    ngp_n_features_per_level: int = 2
    ngp_log2_hashmap_size: int = 22
    ngp_base_resolution: int = 16
    ngp_per_level_scale: float = 1.4
    ngp_n_neurons: int = 128
    ngp_n_hidden_layers: int = 4

    input_size_modulation: int = 2
    n_layers_modulation: int = 3
    hidden_dim_modulation: int = 64
    output_size_modulation: int = 1

    input_size_excitation: int = 3
    n_layers_excitation: int = 5
    hidden_dim_excitation: int = 128

    excitation_dim: int = 1

    latent_dim: int = 64
    latent_update_steps: int = 50
    stochastic_latent: bool = True
    latent_init_std: float = 1.0  # only used if you later add 'init from noise' modes

    # encoder sizes (x -> [mu, logvar])
    input_size_encoder: int = 1      # set to n_neurons in your YAML
    n_layers_encoder: int = 3
    hidden_dim_encoder: int = 256
    latent_n_layers_update: int = 2
    latent_hidden_dim_update: int = 64
    # EED (Encode-Evolve-Decode) sub-network mapping:
    #   encoder:          MLPWithSkips(n_neurons -> latent_dim, hidden=latent_dim, layers=n_layers_encoder)
    #   decoder:          MLPWithSkips(latent_dim -> n_neurons, hidden=latent_dim, layers=n_layers_encoder)  [symmetric]
    #   evolver:          MLPWithSkips(latent_dim+stim_latent_dims -> latent_dim, hidden=latent_dim, layers=n_layers_evolver)
    #   stimulus_encoder: MLPWithSkips(n_input_neurons -> stim_latent_dims, hidden=hidden_dim_stim_encoder, layers=n_layers_stim_encoder)
    n_layers_evolver: int = 1
    hidden_dim_stim_encoder: int = 64
    n_layers_stim_encoder: int = 3
    stim_latent_dims: int = 64
    output_size_decoder: int = 1      # set to n_neurons in your YAML
    n_layers_decoder: int = 3
    hidden_dim_decoder:  int = 256


class ZarrConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    store_fluo: str = ""
    store_seg: str = ""

    axis: int = 0
    frame: int = 0
    contrast: str = "1,99.9"
    rendering: str = "1,99.9"
    dz_um: float = 4
    dy_um: float = 0.406
    dx_um: float = 0.406
    labels_opacity: float = 0.7
    show_boundaries: bool = False


class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    colormap: str = "tab10"
    arrow_length: int = 10
    marker_size: int = 100
    xlim: list[float] = [-0.1, 0.1]
    ylim: list[float] = [-0.1, 0.1]
    embedding_lim: list[float] = [-40, 40]
    speedlim: list[float] = [0, 1]
    pic_folder: str = "none"
    pic_format: str = "jpg"
    pic_size: list[int] = [1000, 1100]
    data_embedding: int = 1
    plot_batch_size: int = 1000
    label_style: LabelStyle = LabelStyle.MLP  # MLP for MLP_0/MLP_1 labels; greek for phi/f labels

    # MLP plot axis limits
    mlp0_xlim: list[float] = [-5, 5]
    mlp0_ylim: list[float] = [-8, 8]
    mlp1_xlim: list[float] = [-5, 5]
    mlp1_ylim: list[float] = [-1.1, 1.1]

    # MLP normalization settings
    norm_method: str = "median"
    norm_x_start: float | None = None  # None = auto (0.85 * xnorm * 4 for training, 0.8 * xnorm for best)
    norm_x_stop: float | None = None   # None = auto (xnorm * 4 for training, xnorm for best)

    # --- 3D anatomy-voltage snapshot --------------------------------------
    # Consumed by connectome_gnn.plot_anatomy_voltage.render_*_anatomy_voltage,
    # invoked from data_test when GNN_Main.py is launched with
    # `--anatomy_voltage`. The render lives in `<log_dir>/tmp_recons/`
    # alongside the per-trial trace plots. Defaults match the user's
    # known-good settings for both drosophila CX (`--stride 5 --z_lo 0
    # --z_hi 15 --alpha 1.0`) and zebrafish HD (`--stride 5 --z_lo 0
    # --z_hi 15`). Per-yaml overrides are the expected customisation
    # path.
    anatomy_voltage_enabled: bool = False
    """yaml-side toggle; when True the snapshot is rendered even without
    the --anatomy_voltage CLI flag. Defaults to False."""
    anatomy_voltage_pattern: str = "const"
    """Probe rollout pattern. One of:
        ``"const"``     constant-omega sweep (default)
        ``"swim"``      stochastic swim-integration stimulus (zebrafish)
        ``"swim_left"`` periodic left-impulse train (zebrafish)
        ``"swim_right"`` periodic right-impulse train (zebrafish)
        ``"ou"``        OU velocity stream (drosophila Hulse defaults)
    Helpers live in :mod:`connectome_gnn.plot_anatomy_voltage`."""
    anatomy_voltage_n_steps: int = 2000
    """Length of the probe rollout (frames)."""
    anatomy_voltage_stride: int = 0
    """When 0, render a single PNG at ``anatomy_voltage_frame_idx``.
    When > 0, render every Nth frame as an animation strip into a
    per-type sub-folder ``tmp_recons/<types>/frame_NNNN.png`` (``<types>``
    = the ``anatomy_voltage_types`` whitelist joined by ``_``, or ``all``
    when empty), and assemble ``tmp_recons/<types>.mp4`` from the cropped
    frames."""
    anatomy_voltage_frame_idx: int = -1
    """Single-snapshot mode: which timestep of h_traj to visualise (-1 = last)."""
    anatomy_voltage_trial_idx: int = 0
    """Unused when ``anatomy_voltage_pattern`` is set (the probe rollout
    replaces the random test trial). Kept for back-compat; ignored by the
    v2 render entry point."""
    anatomy_voltage_omega_deg: float = 60.0
    """Constant-omega angular velocity (deg/s) for the ``const`` pattern."""
    anatomy_voltage_theta0_rad: float = 0.0
    """Initial bump heading (rad). Used by the const + swim_*  patterns."""
    anatomy_voltage_swim_interval_s: float = 1.0
    """Inter-impulse interval (s) for the swim_left/swim_right patterns,
    and the mean Poisson period (1/swim_rate_hz) for ``swim``. Pass 0
    for a single-shot impulse (swim_left/right)."""
    anatomy_voltage_swim_magnitude_rad: float = 0.393
    """Per-impulse magnitude (rad) for the swim_left/swim_right
    patterns. Default π/8 ≈ 22.5° per turn (half the Petrucco median)."""
    anatomy_voltage_swim_t_event_s: float = 0.0
    """Time of the first deterministic impulse (s) for swim_left/right."""
    anatomy_voltage_seed: int = 0
    """RNG seed for stochastic patterns (``swim`` and ``ou``)."""
    anatomy_voltage_warmup_s: float = 10.0
    """Zero-ω warmup (s) prepended to the ``zapbench_rotation`` stimulus so the
    bump settles; discarded after the rollout. 0 = no warmup."""
    anatomy_voltage_zapbench_connectome: str = ""
    """``zapbench_rotation`` only: connectome dir holding functional/
    rotation_heading.npz (the cached 45°/s heading). Empty -> the packaged
    figures/zebrafish/zebrafish_connectome_HD_IPN12 default."""
    anatomy_voltage_zapbench_fishfuncem_data: str = ""
    """``zapbench_rotation`` only: fishFuncEM data dir (onsets/stim) used to
    (re)build the heading cache. Empty -> packaged papers/fishFuncEM/data."""
    anatomy_voltage_elev: float = 90.0
    """Camera elevation for the 3D->2D projection. Default 90.0 = dorsal
    view (matches the zebrafish `--elev_top` default in
    fig_zebrafish_anatomy_3d_voltage_anim.py). Drosophila CX yamls
    override with -7.6 (the Hulse 2025 paper view used by
    fig_cx_anatomy_3d_voltage_anim.py --elev)."""
    anatomy_voltage_azim: float = -85.5
    """Camera azimuth. Default -85.5 = dorsal view. Drosophila CX yamls
    override with 86.6."""
    anatomy_voltage_z_lo: float = 0.0
    """z-score lower threshold (only z > z_lo lights up)."""
    anatomy_voltage_z_hi: float = 15.0
    """z-score saturation point (alpha = 1 at z >= z_hi)."""
    anatomy_voltage_alpha: float = 1.0
    """Global multiplier on per-segment green alpha."""
    anatomy_voltage_downsample: int = 10
    """SWC downsample factor for the skeleton lines (larger = sparser)."""
    anatomy_voltage_bg: str = "black"
    """Figure background: 'black' or 'white'."""
    anatomy_voltage_show_base: bool = True
    """When True, paint the dark-grey base skeleton beneath the green
    lit-segment overlay (matches the new -o test --anatomy_voltage
    default). When False, the base is dropped (matches the standalone
    ``fig_zebrafish_anatomy_3d_voltage_anim.py`` dorsal panel, which
    uses ``show_base=False`` so ROI pixel-sampling isn't biased by the
    static ink)."""
    anatomy_voltage_show_icon: bool = False
    """When True, draw a small fish (zebrafish) or fly (drosophila)
    silhouette in the top-right corner of every frame, oriented at the
    current heading ``theta_hd[t]``. Mirrors the standalone scripts'
    icon overlay. Default False."""
    anatomy_voltage_types: list[str] = []
    """Cell-type whitelist for the anatomy-voltage snapshot. Empty list
    = plot every neuron (the default). When non-empty, only neurons
    whose ``circuit.type_names[circuit.neuron_types[i]]`` is in this
    list contribute skeletons. Useful for showing just the bump pool
    (e.g. ``["EPG"]`` for fly or
    ``["IPNd13B","IPNd13A","IPNds13A","IPNds13B","IPN12_a","IPN12_b"]``
    for fish) without the surrounding afferents."""
    anatomy_voltage_fps: int = 20
    """Playback frame-rate for the mp4 assembled from the frame sequence
    (``stride > 0``). The render writes frames to
    ``tmp_recons/<types>/frame_NNNN.png`` and a movie
    ``tmp_recons/<types>.mp4`` at this fps."""
    anatomy_voltage_kinograph: bool = False
    """When True, also render a companion *sliding kinograph* movie next to
    the anatomy movie: a (neuron x time) z-scored heatmap of the same
    probe rollout, rows sorted by peak time, with a vertical time-cursor
    that slides across in lock-step with the anatomy frames (same
    ``stride`` / ``fps``). Written to
    ``tmp_recons/<types>_kino/frame_NNNN.png`` + ``tmp_recons/<types>_kino.mp4``.
    Toggle on the CLI with ``--anatomy_voltage_kinograph``."""


class TrainingConfig(BaseModel):
    # allow: LLM_code agents introduce new coeff_<name> keys per block; they

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_cond_keys(cls, v):
        """`cond_*` -> `conductance_*`.

        The 9 conductance knobs were named `cond_*` until this rename. The 18
        twin specs' archived config.yaml under log/fly still use the old keys
        and are NOT in git, so with extra="forbid" they would stop loading --
        which would make every finished twin run unreadable. Mapped here rather
        than aliased per field so the whole legacy set is handled in one place
        and the new name is the only one the class declares.
        """
        if isinstance(v, dict):
            for k in [k for k in v if isinstance(k, str) and k.startswith("cond_")]:
                v.setdefault("conductance_" + k[len("cond_"):], v.pop(k))
        return v
    # must survive the YAML→pydantic round-trip so getattr(tc, coeff_name)
    # works from the staged production hook.
    model_config = ConfigDict(extra="allow")
    device: Annotated[str, Field(pattern=r"^(auto|cpu|cuda:\d+)$")] = "auto"
    node_name: str = "a100"  # cluster GPU node: h100, a100, or l4

    n_epochs: int = 20
    n_epochs_init: int = 99999  # DEPRECATED: no longer used by regularizer
    epoch_reset: int = -1
    epoch_reset_freq: int = 99999
    batch_size: int = 1
    inr_batch_size: int = 8
    n_training_frames: int = 0  # 0 = use all frames; >0 = crop centered window

    # Data split (frame indices). 0 = use defaults (all frames for train, no validation).
    train_start: int = 0       # first usable frame (e.g. skip burn-in)
    train_end: int = 0         # exclusive upper bound; 0 = n_frames
    batch_ratio: float = 1
    small_init_batch_size: bool = True
    embedding_step: int = 1000
    shared_embedding: bool = False
    embedding_trial: bool = False
    remove_self: bool = True

    pretrained_model: str = ""
    pre_trained_W: str = ""

    multi_connectivity: bool = False
    with_connectivity_mask: bool = False
    has_missing_activity: bool = False

    epoch_distance_replace: int = 20
    warm_up_length: int = 10
    sequence_length: int = 32

    denoiser: bool = False
    denoiser_type: DenoiserType = DenoiserType.NONE
    denoiser_param: float = 1.0

    training_selected_neurons: bool = False
    selected_neuron_ids: list[int] = [1]

    time_window: int = 0

    n_runs: int = 2
    seed: Annotated[int, Field(ge=0, lt=2**32)] = 42
    # Bitwise-reproducible training. Off by default: it changes the arithmetic,
    # so an existing run's numbers are NOT recoverable by switching it on --
    # it only makes future runs repeatable. See utils.set_deterministic.
    deterministic: bool = False
    clamp: float = 0
    pred_limit: float = 1.0e10

    particle_dropout: float = 0
    n_ghosts: int = 0
    ghost_method: GhostMethod = GhostMethod.NONE
    ghost_logvar: float = -12

    sparsity_freq: int = 5
    sparsity: Sparsity = Sparsity.NONE
    fix_cluster_embedding: bool = False
    embedding_cell_type_init: bool = False  # init model.a with equidistant points per cell type
    embedding_cell_type_scale: float = 1.0  # scale factor applied to equidistant points radius
    fix_embedding: bool = False  # freeze model.a throughout training (requires_grad=False)
    cluster_method: ClusterMethod = ClusterMethod.DISTANCE_PLOT
    cluster_distance_threshold: float = 0.1
    cluster_connectivity: ClusterConnectivity = ClusterConnectivity.SINGLE

    umap_cluster_method: UmapClusterMethod = UmapClusterMethod.NONE
    umap_cluster_freq: int = 1
    umap_cluster_n_neighbors: int = 50
    umap_cluster_min_dist: float = 0.1
    umap_cluster_eps: float = 0.1
    umap_cluster_gmm_n: int = 50
    umap_cluster_fix_embedding: bool = False
    umap_cluster_fix_embedding_ratio: float = 0.0
    umap_cluster_reinit_mlps: bool = False
    umap_cluster_relearn_epochs: int = 0

    Ising_filter: str = "none"

    init_training_single_type: bool = False
    training_single_type: bool = False

    low_rank_factorization: bool = False
    low_rank: int = 20

    lr: float = 0.001
    lr_embedding: float = 0.001
    # Swim/CX-task GNN trainer only: when True, the per-neuron embedding ``a``
    # is pulled out of the schedule-driven ``w_rec`` param group into its own
    # constant-lr ``embedding`` group at ``lr_embedding`` (so the embedding can
    # be driven faster/slower than the recurrent weights — useful when the
    # cluster is meant to move freely, e.g. coeff_embedding_cluster=0). When
    # False (default) ``a`` stays in ``w_rec`` exactly as before, so existing
    # configs are byte-for-byte unchanged.
    embedding_separate_lr: bool = False
    lr_update: float = 0.0
    lr_W: float = 0.0001
    # CX task trainer (_data_train_drosophila_cx_task): separate LR for the recurrent-
    # core params (S in DrosophilaCxTaskRNN; W + a + g_phi + f_theta in DrosophilaCxTaskGNN).
    # When set, `lr_schedule` drives THIS group only (per-epoch trajectory of
    # the recurrent core); when None, the recurrent core starts at `lr` and is
    # still the group `lr_schedule` drives. Schedule never touches the other
    # groups (lr_W_ED and `lr`-biases stay constant).
    lr_W_rec: Optional[float] = None
    # Constant LR for encoder/decoder params: W_in, W_out, MLP variants
    # (_W_in_mlp.*, _W_out_mlp.*), and velocity-gate scalars (v_pena_l/r,
    # v_penb_l/r). Falls back to `lr` when None.
    lr_W_ED: Optional[float] = None

    lr_missing_activity: float = 0.0001
    lr_NNR_f_start: float = 0.0
    lr_NNR_f: float = 0.0001
    lr_omega_f: float = 0.0001
    coeff_omega_f_L2: float = 0.0
    training_NNR_start_epoch: int = 0

    coeff_W_L1: float = 0.0
    coeff_W_L2: float = 0.0
    coeff_W_sign: float = 0
    W_sign_temperature: float = 10.0

    # Stimulus-pathway sparsity penalty (predict_dvdt models, e.g. MLP baseline).
    # Adds  lambda * mean over batch of  sum_i ||delta[:, i]||_2
    # where delta = predict_dvdt(v, stim) - predict_dvdt(v, 0).
    # Encourages the stimulus to drive only a small subset of neurons,
    # without assuming which neurons receive stimulus. 0 = disabled.
    coeff_stim_sparsity: float = 0.0

    # Shared annealing rate for all weight regularization (L1 and L2)
    # Formula: coeff * (1 - exp(-rate * epoch)). With rate=0.5, ramps from 0 at
    # epoch 0 to ~0.39x at epoch 1 to ~0.92x at epoch 5. Set to 0 to disable.
    regul_annealing_rate: float = 0.5

    # Regularization coefficients
    # -- f_theta (neuron update) regularizers --
    coeff_f_theta_zero: float = 0  # Penalize f_theta(0) != 0 (enforce zero-input zero-output)
    coeff_f_theta_diff: float = 0  # Negative monotonicity of f_theta w.r.t. state v_i (enforces leak: df/dv < 0)
    coeff_f_theta_msg_diff: float = 0  # Monotonicity of f_theta w.r.t. aggregated message input
    coeff_f_theta_msg_sign: float = 0  # Sign consistency: f_theta output should match message sign
    coeff_func_f_theta: float = 0.0  # Penalize f_theta output at zero input
    coeff_f_theta_weight_L1: float = 0  # L1 penalty on f_theta MLP weights
    coeff_f_theta_weight_L2: float = 0  # L2 penalty on f_theta MLP weights

    # -- g_phi (edge message) regularizers --
    coeff_g_phi_diff: float = 0  # Positive-monotonicity prior on ∂g_phi/∂v (forces g_phi non-decreasing in presynaptic state — Dale-conformant when g_phi_positive=False)
    coeff_g_phi_norm: float = 0  # Norm penalty on g_phi edge messages
    # g_phi_norm anchoring target — pins g_phi(2*xnorm)^2 to resolve the W<->g_phi
    # scale degeneracy (the gain can float between W and g_phi, leaving W under-
    # scaled). "auto": legacy trainer_type behaviour (1 for signal, 2*xnorm for
    # flyvis). "unit": anchor to 1 (forces g_phi small so W carries the scale —
    # use this when recovered W is under-scaled, e.g. drosophila_cx_voltage).
    # "xnorm": anchor to 2*xnorm.
    g_phi_norm_target: str = "auto"
    coeff_func_g_phi: float = 0.0  # Penalize g_phi output at zero input
    # Reduction of the prediction residual. "norm2" = sqrt(sum r^2), the historical
    # behaviour; "mean" = mean(r^2), i.e. plain MSE as the task trainer uses.
    #
    # These are NOT interchangeable at fixed coeff_*. norm2 scales as
    # sqrt(n_visible * batch_size) while every coeff_* is fixed, so the
    # fit/regulariser balance depends on batch size and on how many neurons are
    # hidden; and d||r||/dr has magnitude 1 however small the residual is, so the
    # data term never yields to the penalties. mean removes both effects (it is
    # the proper penalised-least-squares form) but is ~sqrt(n*B)/(2*sqrt(mse))
    # times weaker, so switching REQUIRES rescaling every coeff_* or the
    # regularisers dominate and R^2_W collapses.
    # "huber" and "relative_l2" exist for the rollout curriculum, where the late
    # steps carry much larger residuals than the early ones: huber caps their
    # gradient share, relative_l2 normalizes each step by its own target norm.
    fit_reduction: Literal["norm2", "mean", "huber", "relative_l2"] = "norm2"
    fit_huber_delta: float = 1.0  # residual magnitude (in ynorm units) where huber turns linear
    coeff_g_phi_weight_L1: float = 0  # L1 penalty on g_phi MLP weights
    coeff_g_phi_weight_L2: float = 0  # L2 penalty on g_phi MLP weights
    # Group lasso (L2,1) on g_phi's first-layer input columns, grouped as [vi],[vj],[ai],[aj].
    # Unlike weight_L1/L2 (elementwise, diffuse shrinkage), this penalizes each input's whole
    # column jointly so the optimizer can drop an entire input pathway to ~0 instead of shrinking
    # all weights a little. No-op unless g_phi's first layer has the 2+2*embedding_dim input
    # width of flyvis_conductance (checked at compute time, not by model name).
    coeff_g_phi_input_group_L1: float = 0  # Group-lasso penalty on g_phi input columns (vi/vj/ai/aj)

    # -- W (connectivity) regularizers --
    # coeff_W_L1, coeff_W_L2, coeff_W_sign defined above

    # -- known_ode biophysical parameter regularizers (apply to model.raw_tau / model.V_rest) --
    coeff_tau_L1: float = 0.0     # L1 penalty on raw_tau (pulls tau toward identity-element of its transform)
    coeff_tau_L2: float = 0.0     # L2 penalty on raw_tau
    coeff_V_rest_L1: float = 0.0  # L1 penalty on V_rest (pulls V_rest toward 0)
    coeff_V_rest_L2: float = 0.0  # L2 penalty on V_rest

    # -- Other regularizers --
    coeff_entropy_loss: float = 0  # Entropy penalty on predictions
    coeff_permutation: float = 100  # Permutation invariance penalty
    coeff_TV_norm: float = 0  # Total variation norm on predictions
    coeff_missing_activity: float = 0  # Penalty for missing activity patterns
    coeff_model_a: float = 0  # Regularizer on embedding a

    # -- TaskRNN (path-integration) regularizers (Hulse Eqs. 10, 11 + circular TV) --
    coeff_cos_distance: float = 0.0    # cos-distance per (post, pre) type-pair block
    coeff_norm_floor:   float = 0.0    # soft floor on mean |W| per type-pair block
    kappa_norm_floor:   float = 0.05   # floor target for norm-floor reg
    coeff_tv_circular:  float = 0.0    # circular TV on EPG ring firing rates
    snapshots_per_epoch: int = 5       # cadence of matrix+kinograph + pi_acc/fwhm eval
    # Per-epoch trial-length curriculum (Hulse Methods). Each epoch slices the
    # first n_steps_schedule[epoch] frames from the on-disk T=1000 trials.
    # Padded with the last value if n_epochs > len(schedule). Empty list → use
    # the full T from the dataset throughout (no curriculum).
    n_steps_schedule: List[int] = Field(default_factory=lambda: [100, 250, 500, 1000, 1000])
    # Soft-curriculum tail weight. When > 0, the PI trainer rolls forward to
    # the full T (not n_steps_schedule[epoch]) and weights the per-frame MSE
    # by 1.0 for t < n_steps_schedule[epoch] and `coeff_tail_loss` for
    # t >= n_steps_schedule[epoch]. Default 0.0 keeps the hard-truncation
    # behaviour. Typical value 0.1 prevents late-time activity collapse by
    # supplying a small gradient on the post-horizon segment.
    coeff_tail_loss: float = 0.0
    # --- place-cell task (target_kind='place_cells', model drosophila_cx_pi_place) ---
    # Weight on the place-cell distribution loss KL(q‖softmax(place_logits))
    # and on the auxiliary population-vector position decode
    # ‖Σ_k p̂_k c_k − (x,y)‖². Heading+distance keep their unit-weight MSE.
    coeff_place: float = 1.0
    coeff_pos: float = 1.0
    # Net1↔Net2 integrator-consistency weight: penalises the per-step mismatch
    # between Net2's decoded position speed ‖Δ(x̂,ŷ)‖ and Net1's decoded
    # forward-distance speed |Δd̂| (both = |v_fwd|·dt). 0 = off. Couples the two
    # integrators so Net1's distance can't run away from Net2's position.
    coeff_consistency: float = 0.0
    # Net1 warm-up: for the first ``place_warmup_epochs`` epochs the place loss
    # (KL + position decode) is switched off (coeffs forced to 0), so only the
    # heading+distance MSE trains — Net1's compass converges before Net2 has to
    # read it. 0 = no warm-up (place loss on from step 1). Net2 still runs but
    # receives no gradient during warm-up.
    place_warmup_epochs: int = 0
    # Path-integration anchor: at t=0 seed Net2's place-cell population with the
    # Gaussian place-field code of the true start position (σ = place_sigma),
    # then let Net2 integrate velocity from there. Trajectories start at random
    # points in the arena, so absolute position is unobservable from the
    # velocity drive alone — this anchor supplies the one unrecoverable piece
    # (where the trial started), exactly as standard PI models do (Banino 2018,
    # Cueva & Wei 2018). False = no anchor (legacy behaviour). Eval/test/MP4
    # honour the same flag via the model so train/eval stay consistent.
    place_anchor: bool = False
    # Task mode for the swim-integration task — selects which sub-task the
    # network is trained on and projects the 4-ch / 3-col on-disk superset
    # onto the matching input / target sub-channels:
    #   ['rotation']                — angular only.    in=[ω,cosθ0,sinθ0]  (3)
    #                                                  out=[cosθ,sinθ]    (2)
    #   ['translation']             — displacement.   in=[v_fwd]          (1)
    #                                                  out=[ξ]            (1)
    #   ['rotation','translation']  — both.            in=[ω,v_fwd,cos,sin] (4)
    #                                                  out=[cos,sin,ξ]    (3)
    # graph_model.n_input / n_output auto-derive from this when not explicitly
    # set in the yaml. Legacy 3-ch on-disk datasets pass through unchanged
    # (the slicing is gated on the dataset's 4-ch input width).
    task_targets: List[str] = Field(default_factory=lambda: ['rotation'])
    # Heading-bin ablation. When False (default): standard cos/sin readout
    # and (cos θ₀, sin θ₀) input cue. When True: replace BOTH with a
    # K-bin one-hot representation — input cue becomes a one-hot bump
    # over `n_heading_bins` channels at t=0, output is K-bin logits trained
    # with cross-entropy. Purpose: remove the circular geometry of the
    # cos/sin target/cue from the supervision so the recurrent code is not
    # pushed toward a sinusoidal embedding of θ. The on-disk dataset is
    # untouched (cos/sin format); conversion happens at training time and
    # in the rollout helpers. Affects rotation-bearing task_targets only;
    # downstream targets (ξ, x, y) pass through unchanged.
    use_heading_bins: bool = False
    n_heading_bins: int = 64
    # Per-epoch learning-rate schedule (Hulse). Empty list → constant `lr`.
    # Padded with the last value if n_epochs > len(schedule).
    lr_schedule: List[float] = Field(default_factory=lambda: [5e-3, 1e-3, 5e-4, 2e-4, 1e-4])
    # PI trainer only: per-epoch lr trajectory for the w_rec group. Cortex
    # has a single param group and uses `lr_schedule` above. This field is
    # the unambiguous name for the three-group PI setup.
    lr_W_rec_schedule: List[float] = Field(default_factory=list)
    # TaskRNN: stddev of Gaussian noise added to the hidden state at
    # every Euler step during training (flyvis-style). 0 → off (Hulse default).
    # Try {0, 1e-3, 1e-2, 5e-2} — small noise smooths the long-T BPTT
    # landscape and is the key trick used by flyvis for stable recurrent
    # training. Noise is gated by `self.training`; eval / snapshot rollouts
    # are deterministic.
    noise_recurrent_level: float = 0.0
    coeff_model_b: float = 0  # Regularizer on bias b
    coeff_embedding_cluster: float = 0.0  # pull same-cell-type embeddings toward their per-type centroid (L2)

    # -- f_theta linearity regularizer (unsupervised V_rest recovery) --
    coeff_f_theta_linearity: float = 0.0           # Penalize f_theta nonlinearity (0 = disabled)
    f_theta_linearity_warmup_fraction: float = 0.3  # Fraction of iterations before activation
    f_theta_linearity_rampup_iters: int = 200       # Linear ramp-up after warmup ends

    # -- f_theta centering loss (unsupervised V_rest proxy) --
    coeff_f_theta_centering: float = 0.0   # Weight of centering loss (0 = disabled)
    f_theta_centering_warmup_fraction: float = 0.3   # Fraction of iters before activation
    f_theta_centering_rampup_iters: int = 200        # Linear ramp-up after warmup

    g_phi_mode: GPhiMode = GPhiMode.MLP  # mlp=learned MLP, tanh=fixed tanh(u_j), identity=fixed u_j
    w_optimizer_type: WOptimizerType = WOptimizerType.ADAM  # adam (default) or sgd (SGD with momentum)

    # Simple training parameters (matching ParticleGraph conceptually)
    first_coeff_L1: float = 0.0  # Phase 1 weak L1 regularization
    coeff_L1: float = 0.0  # Phase 2 target L1 regularization
    coeff_diff: float = 0.0  # Monotonicity constraint on edge function

    loss_noise_level: float = 0.0

    # Resample stored measurement noise (x_ts.noise) at the start of every epoch
    # using the per-epoch RNG seeded from simulation.seed + epoch. Lets the model
    # average noise across epochs instead of memorising the fixed realisation
    # baked into noise.zarr at data-generation time.
    resample_noise_per_epoch: bool = False

    # Compilation flag for torch.compile optimization
    torch_compile: bool = True

    # Arithmetic precision of the two MLPs, g_phi (E*B = 1.7M rows at flyvis
    # scale) and f_theta (N*B rows). Parameters, optimiser state, the residual and
    # every reduction stay fp32 in all three settings -- only the matmuls change.
    #   'fp32'  IEEE fp32 throughout. The published behaviour, and the only
    #           setting that reproduces an existing run to the last digit.
    #   'tf32'  TF32 tensor cores for the matmuls (10-bit mantissa). Measured on
    #           an A6000: g_phi fwd+bwd 20.75 -> 18.91 ms, max|d| 2.2e-4.
    #   'bf16'  autocast the MLPs to bfloat16 (8-bit mantissa), which also halves
    #           the [E*B, hidden] activation traffic -- the reason it wins most on
    #           a bandwidth-starved card. A6000: 20.75 -> 10.21 ms, max|d| 3.4e-3.
    # fp16 is deliberately NOT offered. Measured against bf16 on the same step it
    # is both slower (2.08x vs 2.71x) and flushes 3x as many gradients to exactly
    # zero -- 5 exponent bits against bf16's 8 -- and there is no GradScaler in
    # this path to catch that. Its one advantage, 11 bits of significand against
    # 8, is not what this workload is short of.
    # NOT free: tf32 and bf16 change the trajectory, so a run under them cannot be
    # compared digit-for-digit with an fp32 one. Judge them on the five-fold R2
    # distribution instead.
    mlp_precision: Literal["fp32", "tf32", "bf16"] = "fp32"

    # Per-neuron reweighting of the fit residual, after the inverse-variance s_j (supplement
    # sec 4.2), the "per-variable-level inverse variance of time differences".
    #   'none'                every neuron weighted equally -- the historical behaviour.
    #   'inv_increment_std'   weight_j = 1 / std_t(dv_j/dt), renormalised to mean 1.
    # Measured on flyvis_noise_free_blank50_cv00: the per-neuron std of the target
    # spans 4250x (min 5.9e-3, median 4.8, max 25.2), only 49% of neurons are within
    # 2x of the median, and the fastest 10% carry 45% of the squared loss. Unweighted,
    # the slow neurons are almost invisible to the objective.
    #
    # Applied to the RESIDUAL, not the target, so the model keeps predicting physical
    # dv/dt. Weighting the target instead -- which is the reference's literal form -- would
    # change the output units and silently break all 12 sites that integrate
    # `voltage + delta_t * pred`, and would need the weights reloaded at inference.
    # This way nothing downstream changes and no artifact has to be read back.
    #
    # Renormalised to RMS 1, not mean 1: 1/std is heavy-tailed, and norm2's magnitude
    # goes as sum(w^2), so mean-1 inflated the fit term 6.3x and would have
    # decalibrated every coeff_*. RMS 1 leaves it where it was.
    target_weighting: Literal["none", "inv_increment_std"] = "none"
    # Floor the per-neuron std at this percentile before inverting, so a near-silent
    # neuron (min is 800x below the median here) is not amplified 4000-fold into
    # dominating the loss with its own noise. 0 disables the floor. p5 keeps the
    # weight spread at 61x; p1 leaves it at 2971x and p0 at 4250x, which is the raw
    # 1/std range and almost certainly amplifying measurement noise.
    target_weight_floor_pct: float = 5.0

    # TEACHER-STUDENT DISTILLATION. False (default) is ordinary training against a
    # dataset, and every existing spec is unaffected. True marks the run as fitting a
    # STUDENT to a teacher's recorded activity, which changes what the run is judged
    # on: R2_W is meaningless here -- the teacher is current-based and has no
    # conductance ground truth to recover -- so the acceptance test is the ROLLOUT,
    # whether the student runs free and stays on the teacher's trajectory.
    train_on_teacher: bool = False
    # Frames in the in-training rollout. The full test rollout is ~7,200 frames and
    # takes ~40 s; 1,000 is a second and enough to see divergence, which is what a
    # per-checkpoint diagnostic is for. The reported number is Pearson r over all
    # (neuron, frame) pairs, the same statistic results_rollout.log quotes.
    teacher_rollout_frames: int = 1000

    # ---- flyvis_conductance_known_ode: which parameter groups are learnable -------------
    # The student has three groups and they differ by orders of magnitude in size,
    # so which are free is the experiment rather than a detail:
    #   reversals      2            E_exc, E_inh
    #   edges          434,112      W, entering squared so the conductance is >= 0
    #   neuron params  see below    tau, V_rest
    conductance_learn_reversal: bool = True
    conductance_learn_edges: bool = True
    # tau and V_rest. The teacher's own values have exactly 65 DISTINCT entries over
    # 13,741 neurons -- one per cell type -- so per-neuron spends 27,482 parameters
    # to represent 130, and 'per_type' is both smaller and the structure the data
    # actually has. 'frozen' pins them at the teacher's values, which turns the fit
    # into "can a conductance synapse reproduce this activity given the right
    # neurons" rather than "can it reproduce it at all".
    conductance_neuron_params: Literal["per_neuron", "per_type", "frozen"] = "per_type"
    # HOW THE REVERSAL POTENTIALS ARE SET. Not a regulariser -- a reparametrisation,
    # which is why it can GUARANTEE what a penalty could only encourage.
    #   'learned'  E_exc, E_inh are free parameters. Nothing stops them crossing the
    #              teacher's voltage range, and if V_i crosses E the driving force
    #              flips sign and excitation silently becomes inhibition.
    #   'margin'   E_exc = V_max + delta_exc * span, E_inh = V_min - delta_inh * span,
    #              with V_min/V_max/span measured from the teacher. Bracketing then
    #              holds by construction for any delta > 0, so the sign cannot flip.
    # delta is also the CONTINUITY KNOB between the two models: as delta grows,
    # (E - V_i) -> delta and loses its V_i dependence, so with a conductance scale
    # going as 1/delta the message tends to s_ij alpha' N f(V_j) -- the current-based
    # teacher. Small delta is strongly conductance-like; large delta degenerates to
    # the teacher, continuously. The asymmetric default mirrors the inhibitory
    # driving force being roughly half the excitatory one in real neurons.
    conductance_reversal_mode: Literal["learned", "margin"] = "margin"
    # GRANULARITY of E. The driving force is (E - V_i), so E belongs to the
    # POSTSYNAPTIC cell -- these are per postsynaptic neuron/type, not per edge.
    #   'global'      two scalars, E_exc and E_inh.
    #   'per_type'    two per cell type. Closest to PR #46, which carries one
    #                 reversal per (presynaptic type -> postsynaptic type) group.
    #   'per_neuron'  two per neuron: the overparameterised control. Physically a
    #                 reversal is a property of the receptor, shared by synapse type,
    #                 so a per-neuron gain is capacity absorbing model mismatch.
    # Crosses with conductance_reversal_mode: 'learned' fits them, 'margin' sets them from
    # the teacher's voltage range measured AT THE SAME GRANULARITY.
    conductance_reversal_dim: Literal["global", "per_type", "per_neuron"] = "global"
    # WHAT delta IS MEASURED IN. The reversals bracket from the teacher's min/max in
    # every case -- that is what guarantees the sign and the convex-hull bound -- but
    # delta needs a UNIT, and PR #46 uses (v_max - v_min), the raw extremes.
    #
    # On this teacher that unit is set by outliers: 98% of the voltage lies in a band
    # of width 3.86 while the global range is 16.27, a 3.4x inflation. The nominal
    # margin (0.4, 1.0) therefore acts ~3.4x larger than intended and E lands far
    # outside anything the network does -- E_exc = 24.6, against which (E_exc - V_i)
    # varies by only 16% across the entire bulk of the data. That is the delta ->
    # infinity limit the methods describe as degeneracy TO the current-based teacher,
    # reached by accident: the twin is then only weakly conductance-like.
    #
    # A percentile span keeps the bracket and restores the state dependence:
    #     extremes  span 16.271  E_exc 24.611  driving force varies 16.0%
    #     p99       span  3.862  E_exc 12.203                        32.8%
    #     p95       span  2.255  E_exc 10.595                        38.0%
    #     p90       span  1.405  E_exc  9.745                        41.4%
    # Default 'extremes' reproduces PR #46 exactly.
    conductance_span_mode: Literal["extremes", "p99", "p95", "p90"] = "extremes"
    # STAGE-1 CLOSED-FORM INITIALISATION, from the conductance-twin methods.
    # The two models differ only in what multiplies the synaptic activation
    # N f(V_j): a constant s_ij alpha_curr for the teacher, a state-dependent
    # alpha_cond (E - V_i) for the twin. Expanding about the per-cell-type mean
    # postsynaptic voltage Vbar_ti and equating the zeroth-order terms,
    #
    #     alpha_cond = alpha_curr / (E - Vbar_ti)          > 0
    #
    # positive by construction, because E - Vbar carries the same sign s_ij that
    # alpha_curr does -- E_exc lies above and E_inh below every voltage the teacher
    # visits, which is exactly what conductance_reversal_mode 'margin' guarantees. Exact
    # wherever the postsynaptic cell sits at its mean, and exact everywhere as
    # delta -> infinity. One number per (presynaptic type, postsynaptic type) group.
    #
    # This is an INITIALISER, not a fitting stage: it replaces w_init_mode's random
    # start with a physically motivated one and then hands over to the usual
    # derivative loss. Stage 2 of the methods -- the per-cell-type NNLS on synaptic
    # currents -- is deliberately NOT implemented: it solves for a SHARED alpha per
    # group, a few thousand unknowns, where this model learns 434,112 per-edge
    # weights, so its answer could only ever be a prior, and that is what stage 1
    # already provides more cheaply. It would also need the teacher's synaptic
    # current I_i(t) as a target, which the generator does not store.
    conductance_init: Literal["default", "teacher_closed_form"] = "teacher_closed_form"
    # (0.4, 1.0) is PR #46's own default -- derive_conductance_twin's
    # `reversal_margin: Union[float, Tuple[float, float]] = (0.4, 1.0)`, ordered
    # (inh, exc) -- reused deliberately so the twin derived there and the student
    # fitted here sit at the same operating point and their results are comparable.
    conductance_delta_inh: float = 0.4
    conductance_delta_exc: float = 1.0

    # Adam's second-moment decay. The reference (supplement sec 4.4) uses 0.95 rather
    # than torch's 0.999: a shorter second-moment window tracks a non-stationary
    # gradient scale faster, which is what a curriculum that changes the objective
    # partway through produces. Default keeps torch's value.
    adam_beta2: float = 0.999

    # external input learning
    learn_external_input: bool = False

    save_all_checkpoints: bool = False  # True = save iteration-level checkpoints too
    checkpoint_saves_per_epoch: int = 1  # >1 also saves within-epoch snapshots at a fixed cadence (see graph_trainer.py)

    test_dataset: str = ""  # dataset for testing; empty = same as training dataset

    data_augmentation_loop: int = 40

    rollout_train_steps: int = 1  # multi-step rollout training: unroll K steps and backprop

    recurrent_training: bool = False
    recurrent_training_start_epoch: int = 0
    recurrent_loop: int = 0
    noise_recurrent_level: float = 0.0

    hidden_neuron_fraction: float = 0.0  # fraction of non-input neurons to silence (0 = disabled); seed = simulation.seed

    neural_ODE_training: bool = False
    ode_method: OdeMethod = OdeMethod.DOPRI5
    ode_rtol: float = 1e-4
    ode_atol: float = 1e-5
    ode_adjoint: bool = True
    ode_state_clamp: float = 10.0
    ode_stab_lambda: float = 0.0
    grad_clip_W: float = 0.0
    # Global gradient-norm clip over ALL parameters, as the reference does at 32
    # (supplement sec 4.4). grad_clip_W above clips only model.W and defaults
    # off, so today nothing bounds f_theta / g_phi / a gradients at all -- a
    # plausible cause of the one-checkpoint R^2_W collapses that force the
    # trailing-median reading rule.
    grad_clip_norm: float = 0.0
    use_gt_edges: bool = False  # True = use ground truth edge_index; False = fully connected graph
    w_init_mode: WInitMode = WInitMode.RANDN  # randn=std=1, randn_scaled=std=scale/sqrt(N), zeros
    w_init_scale: float = 1.0  # scaling factor for 'randn_scaled' mode
    coeff_W_L1_proximal: float = 0.0  # proximal L1 soft-thresholding on W after optimizer step, 0 = disabled
    dale_law: bool = False  # enforce Dale's law: force each column of W to a consistent sign, 3 times per epoch
    freeze_known_ode_gain: bool = False  # drosophila_cx known-ODE: hold the per-neuron gain g fixed, so W is not free up to a per-source scale
    freeze_known_ode_bias: bool = False  # drosophila_cx known-ODE: hold the per-neuron bias b fixed; in the softplus tail b is a second per-source gain

    alternate_training: bool = False  # two-stage training: joint warmup then V_rest focus
    alternate_joint_ratio: float = 0.4  # fraction of total iterations for joint phase (all components at full LR)
    alternate_lr_ratio: float = 0.1  # LR multiplier for W/g_phi during V_rest focus phase

    # Learning rate scheduler
    # 'warmup_cosine_tail': linear warmup -> a SINGLE half-cosine decay -> a
    # constant tail floor. Note 'linear_warmup_cosine' does NOT decay to zero: it
    # chains warmup with CosineAnnealingWarmRestarts (T_mult=2), so the LR
    # sawtooths back up and bottoms out at eta_min_ratio * lr. Use
    # 'warmup_cosine_tail' for a monotone decay.
    lr_scheduler: str = "none"  # 'none'|'cosine_warm_restarts'|'linear_warmup_cosine'|'warmup_cosine_tail'
    # Fraction of the planned total updates over which the half-cosine runs. The
    # remainder is the constant tail, which is where the rollout phase lives.
    # 300000/311000 = 0.965 in the source this was taken from.
    lr_scheduler_decay_frac: float = 0.965
    # Tail LR as a fraction of peak: 3e-7 / 1e-3 = 3e-4 in the source.
    lr_scheduler_tail_ratio: float = 3e-4
    lr_scheduler_T0: int = 1000  # restart period in iterations
    lr_scheduler_T_mult: int = 2  # period multiplier after each restart
    lr_scheduler_eta_min_ratio: float = 0.01  # min LR as fraction of base LR
    lr_scheduler_warmup_iters: int = 100  # linear warmup iterations

    time_step: int = 1
    # Per-epoch rollout-horizon curriculum for recurrent GNN training: epoch e unrolls
    # rollout_horizon_schedule[e] steps and supervises EVERY intermediate step against the
    # observed voltage (dense supervision), on an UNSTRIDED dataset. Empty (default) keeps
    # the legacy endpoint-only, stride-subsampled `time_step` behaviour untouched.
    #
    # This is deliberately a separate knob from `time_step`, which is triple-duty: BPTT
    # depth, dataset decimation stride (training_utils.py init_training_data), and the
    # frame-sampling target offset. Ramping `time_step` would coarsen the observation grid
    # rather than lengthen the horizon; this knob lengthens the horizon at fixed dt.
    # Requires time_step == 1 (enforced in data_train_gnn) so intermediate frames exist.
    rollout_horizon_schedule: List[int] = Field(default_factory=list)
    # Cap on iterations for epochs with K > 1, so the rollout phase can be a SHORT
    # tail fine-tune rather than the bulk of training. The reference spends 96.5% of its
    # updates at K=1 and only 3.5% ramping K upward, at an LR ~3300x below peak;
    # without this knob a K>1 epoch costs Niter//K updates, which is far too many.
    # 0 = no cap (the pre-existing behaviour).
    rollout_tail_iters_per_epoch: int = 0
    # Pin the start-frame sampling range across arms of one comparison.
    #
    # get_training_frame_sampling derives last_frame = n_frames - 4 - target_offset,
    # and the rollout curriculum passes target_offset = max(K) while the one-step
    # path derives it from time_step. So a K=1..5 curriculum samples from 63,991
    # while its own t+1 control samples from 63,995: the two are NOT RNG-paired,
    # and any difference smaller than the run-to-run floor is unattributable.
    # A K=1 recurrent run SHOULD be identical to one-step training and was not.
    #
    # Set this to the largest horizon used anywhere in the comparison (including
    # on the one-step arms) so every arm draws from the same range. 0 = derive as
    # before.
    frame_target_offset: int = 0

    # Decouple regularisation strength from batch_size.
    #
    # The fit term is one norm2 over the whole batched graph, ||r||_2 over
    # n_visible * batch_size elements, so it grows as sqrt(batch_size) -- measured
    # 1.99x at B=4, 3.98x at B=16. The regularisers are parameter norms, computed
    # once per iteration and independent of B. So at fixed coeff_* the
    # regulariser/fit ratio falls as 1/sqrt(batch_size): batch_size is silently a
    # regularisation hyperparameter, and coeff_* cannot be copied between configs
    # with different batch sizes.
    #
    #   'none'  multiplier 1.0        -- the historical behaviour, bit-identical.
    #   'sqrt'  multiplier sqrt(B)    -- ratio constant in B, so coeff_* transfers.
    #
    # Default 'none' so every published config reproduces exactly. Do NOT combine
    # with fit_reduction 'mean', which removes the same coupling a different way
    # (mean(r^2) is already flat in B).
    regul_batch_scaling: Literal["none", "sqrt"] = "none"

    # -- How the K rollout steps are combined, how far the gradient is carried
    #    back through them, and how often the state is re-anchored on data.
    #    All three are no-ops when rollout_horizon_schedule is empty or K == 1.
    #
    # Weighting across the K supervised steps. The loss is
    #     sum_s w_s * fit(pred_s - y_{k+s})  /  sum_s w_s
    # so the objective scale stays horizon-independent under every choice.
    #   'uniform'      w_s = 1                — every step counts the same.
    #   'discount'     w_s = gamma^s          — the model-based-RL convention;
    #                  down-weights the far, drifted steps whose targets the
    #                  model can only reach through its own accumulated error.
    #   'linear_decay' w_s = (K - s) / K      — gentler version of the above.
    #   'last'         w_s = 1 iff s == K-1   — endpoint-only, i.e. the legacy
    #                  _standard_recurrent_loss objective but on the dense
    #                  unstrided grid with a live stimulus. Kept as the control
    #                  that isolates "dense supervision" from "long horizon".
    rollout_step_weighting: Literal["uniform", "discount", "linear_decay", "last"] = "uniform"
    rollout_discount: float = 0.9  # gamma, used only by rollout_step_weighting='discount'

    # How the K weighted per-step terms are combined. This is a DIFFERENT reduction
    # from fit_reduction: fit_reduction collapses the residual of ONE step over
    # n_visible * batch_size elements, this one collapses the K steps.
    #   'mean'  loss = sum_s w_s * fit_s / sum_s w_s   — the objective scale is
    #           independent of the horizon, so coeff_* set at K=1 stay calibrated
    #           as the curriculum grows K, and K=1 is term-for-term one-step
    #           training under every weighting.
    #   'sum'   loss = sum_s w_s * fit_s               — the fit term grows ~K, so
    #           the regulariser/fit ratio falls as 1/K and coeff_* would have to be
    #           re-tuned per horizon. Offered because with 'uniform' weights the
    #           two differ by exactly the factor K, which makes 'sum' the honest
    #           name for "no normalisation" rather than something to emulate by
    #           hand-scaling the weights.
    # NB neither touches the batch-size coupling: that is fit_reduction's job,
    # compensated by regul_batch_scaling.
    rollout_step_reduction: Literal["mean", "sum"] = "mean"

    # Backprop-through-time depth, in rollout steps. 0 = full BPTT through all K
    # (current behaviour): step s reaches the parameters by s+1 paths, so the
    # earliest model call is differentiated K times over and the share of the
    # gradient that is clean teacher-forced one-step supervision falls as 1/K.
    # m > 0 detaches the rolled state every m steps, so no gradient path is
    # longer than m. m = 1 is the "pushforward" trick of Brandstetter et al.
    # (Message Passing Neural PDE Solvers, ICLR 2022): the model still SEES its
    # own drifted states — which is the point, it is what makes one-step training
    # robust to rollout distribution shift — but each step contributes only a
    # one-step gradient, so nothing about the conditioning degrades with K.
    rollout_bptt_window: int = 0

    # Multiple shooting. 0 = off. m > 0 resets the rolled state to the OBSERVED
    # voltage every m steps, so the trajectory is a chain of length-m shooting
    # segments each launched from data rather than one length-K free run. This is
    # the standard remedy in ODE parameter estimation (and the direct fix for the
    # 1/K teacher-forcing dilution above): every segment gets an exact initial
    # condition, so the number of teacher-forced anchors grows with K instead of
    # staying at one. m = 1 degenerates to K independent one-step fits, i.e. the
    # nominal objective evaluated at K consecutive frames — a useful anchor.
    # NB: unlike textbook multiple shooting there is no continuity penalty
    # between segments; the observations ARE the continuity constraint, since
    # every segment start is pinned to data.
    rollout_shooting_stride: int = 0
    multi_start_recurrent: bool = False
    consecutive_batch: bool = False
    coeff_hidden_voltage: float = 0.0  # loss weight on GNN-predicted hidden voltages in recurrent training (NB: the self-consistency variant in graph_trainer was removed because it was a zero-attractor; only the GT-supervised variant in recurrent_step.py still reads this knob)
    # Differential LR damping around the NGP injection switch. The schedule is
    # a V centered at warmup_inject_nnr_iter: GNN param groups (W, f_theta,
    # g_phi) drop their LR to base_lr / lr_damping_factor over the first
    # warmup_inject_nnr_ramp_iter window, then recover back to base_lr over an
    # equal-length recovery window. Embedding (model.a), NNR_hidden, NNR_f are
    # left at full LR throughout. Default 100.0 mirrors the symmetric "divide
    # by 100, multiply by 100" pattern (factor>=1 used for both legs).
    # Applied only when warmup_inject_nnr_iter and warmup_inject_nnr_ramp_iter
    # are both > 0 (otherwise lr_mult stays at 1.0).
    lr_damping_factor: float = 100.0
    # Anchor neurons: observed neurons whose GT voltages directly supervise NGP-T backbone.
    # Only active when hidden_neuron_fraction > 0 and NNR_hidden is built.
    # n_anchor defaults to len(hidden_ids) when <=0; sampled from visible non-retina, saved to log_dir/anchor_neuron_ids.pt.
    train_with_anchor_neurons: bool = False  # True = add anchor-supervised outputs to NGP-T
    coeff_anchor_voltage: float = 0.0  # loss weight on NGP-T anchor outputs vs GT voltages
    n_anchor: int = 0  # number of anchor neurons; 0 = match |hidden_ids|
    recurrent_sequence: str = ""
    recurrent_parameters: list[float] = [0, 0]

    regul_matrix: bool = False
    sub_batches: int = 1
    sequence: list[str] = ["to track", "to cell"]

    max_iterations_per_epoch: int = 0  # 0 = use default (n_frames * aug / batch * 0.2); >0 = cap Niter
    profiling: bool = False  # print per-iteration timing + write Chrome trace

    MPM_trainer : str = "F"

    @model_validator(mode="after")
    def _regul_batch_scaling_not_with_mean(self):
        """regul_batch_scaling and fit_reduction 'mean' fix the SAME coupling.

        The batch-size dependence comes from the fit term: norm2 is ||r||_2 over
        n_visible * batch_size elements, so it grows as sqrt(batch_size) while the
        regularisers do not. 'sqrt' compensates by scaling the coefficients;
        fit_reduction 'mean' removes the dependence at the source, since mean(r^2)
        is already flat in batch_size. Enabling both over-corrects by sqrt(B) --
        a 2x over-regularisation at the usual batch_size 4, silently.

        Rejected rather than silently forced to 'none': a config that asks for two
        conflicting corrections is a mistake the author should see, and silent
        overrides are exactly the failure mode this codebase keeps hitting.
        """
        if self.regul_batch_scaling != "none" and self.fit_reduction == "mean":
            raise ValueError(
                f"regul_batch_scaling={self.regul_batch_scaling!r} with "
                f"fit_reduction='mean' double-corrects the batch-size coupling "
                f"(over-regularises by sqrt(batch_size)={float(self.batch_size) ** 0.5:.2f}). "
                "mean(r^2) is already batch-size independent -- set "
                "regul_batch_scaling='none', or keep 'sqrt' with fit_reduction='norm2'."
            )
        return self


# ---------------------------------------------------------------------------
# Task-data generation (input stimulus + target output). Three task families:
#   - path_integration: Hulse heading-direction estimation
#   - optical_flow: video-driven flow targets
#   - cortex: Yang et al. 2019 multitask cognitive battery, ported directly
#             from gyyang/multitask (see generators/cortex_task.py)
# ---------------------------------------------------------------------------


class InputPerturbation(BaseModel):
    """Stochastic decorrelation signal added to task-input channels.

    Wraps the existing OptoWaveform schema (kind/amplitude/frames_on/
    noise_level/...) and adds a channel mask. Channels not in the mask are
    untouched — critical for PI where channels 1,2 carry the initial heading
    only and perturbing them destroys the IC semantics.
    """
    model_config = ConfigDict(extra="ignore")

    waveform: OptoWaveform
    channel_mask: Optional[List[int]] = None  # None = all channels


class PathIntegrationTaskConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_trials_train: int
    n_trials_test: int
    n_steps: int = 100              # T per trial (Hulse default)
    dt: float = 0.01                # seconds (Hulse default)
    seed: int = 42

    tau_corr: float = 0.12
    sigma_omega_deg: float = 40.0
    stop_fraction: float = 0.20
    stop_mean_s: float = 2.0
    stop_max_s: float = 8.0

    device: Literal["cpu", "cuda", "auto"] = "cpu"
    # Additive Gaussian noise σ on the observed omega channel of the
    # stimulus. True heading (theta_hd / target_y) is computed from the
    # clean omega — only the network's input is corrupted. 0 = no noise.
    omega_noise_level: float = 0.0


class SwimIntegrationTaskConfig(BaseModel):
    """Larval-zebrafish swim-impulse heading-integration task.

    Companion of ``PathIntegrationTaskConfig`` for the dIPN HD-ring port
    (see ``docs/zebrafish.tex``). Where the drosophila PI generator drives
    the heading with a continuous OU angular-velocity stream, this generator
    drives it with a sparse Poisson sequence of typed swim events. Each
    event applies a finite-duration boxcar to a discrete angular-velocity
    channel and integrates to heading exactly as in PI; consumers see the
    same TaskTrials/zarr layout, so the trainer and the readout are
    unchanged.

    Four swim categories are sampled per onset, mirroring the larval
    zebrafish behavioural taxonomy (Petrucco et al.\\ 2023 Fig.\\ 3a,c):

      left      - CCW turn,  signed Δθ = +phase_impulse_mean_rad
      right     - CW turn,   signed Δθ = -phase_impulse_mean_rad
      forward   - propulsion, Δθ ≈ 0 (no net heading rotation)
      backward  - escape / large turn-around,
                  |Δθ| ≈ backward_phase_mean_rad (~π by default)

    Category proportions are configurable via the four ``*_fraction`` fields
    and must sum to 1. The boxcar duration ``swim_duration_s`` discretises
    each impulse over L = swim_duration_s / dt frames so heading integrates
    smoothly rather than via a Dirac.
    """
    model_config = ConfigDict(extra="ignore")

    n_trials_train: int
    n_trials_test:  int
    n_steps: int = 1000             # T per trial; ~10s at dt=0.01
    dt: float = 0.01                # seconds
    seed: int = 42

    # Swim event statistics
    swim_rate_hz: float = 0.5       # mean Poisson rate of swim onsets
    swim_duration_s: float = 0.3    # boxcar width per swim event

    # Phase-impulse magnitude per swim event (rad). Petrucco Fig 3c reports
    # median 0.83 rad, Q1=0.49, Q3=1.28 over n=31 fish, so the default
    # mean/std target that envelope on a lognormal in Δθ.
    phase_impulse_mean_rad: float = 0.785       # ~π/4
    phase_impulse_std_rad:  float = 0.40
    # Backward-swim mean phase (rad). Large turn-around / escape; ~π.
    backward_phase_mean_rad: float = 3.14
    backward_phase_std_rad:  float = 0.30

    # Category proportions; must sum to 1.
    left_fraction:     float = 0.40
    right_fraction:    float = 0.40
    forward_fraction:  float = 0.15
    backward_fraction: float = 0.05

    # Additive Gaussian σ on the observed ω channel of the stimulus (deg/s).
    # True heading is computed from the clean ω — only the network input is
    # corrupted. 0 = no noise. Mirrors PathIntegrationTaskConfig.
    omega_noise_level: float = 0.0

    # --- Translational (forward/backward) drive ---------------------------
    # The dataset ALWAYS writes the complete superset: forward/backward swims
    # drive a translational velocity channel v_fwd that integrates to a
    # displacement target ξ = ∫v_fwd·dt, in parallel to the rotational ω→heading
    # (forward = +v_fwd, backward = −v_fwd). The input is 4-channel
    # [ω, v_fwd, cosθ0·δ, sinθ0·δ] and the target is 3-column [cosθ, sinθ, ξ].
    # v_fwd is a drive, ξ its readout — never both. WHICH target(s) are actually
    # supervised (translation, rotation, or both) is a TRAINER choice, not a
    # data choice — the data carries everything.
    forward_vel_mean: float = 1.0   # mean |v_fwd| per forward/backward event (units/s)
    forward_vel_std:  float = 0.40

    # Leaky-integrator time constant for the displacement target ξ. None (or
    # ≤ 0) → perfect integrator ξ = ∫ v_fwd dt (the default, byte-identical
    # to the original generator). A finite value τ > 0 replaces cumsum with
    # the leaky recurrence
    #     ξ(t+Δt) = (1 − Δt/τ) ξ(t) + v_fwd(t) Δt,   ξ(0)=0,
    # i.e. dξ/dt = −ξ/τ + v_fwd. Steady-state for constant v_fwd is τ·v_fwd
    # so the target is bounded and the loss no longer grows with curriculum
    # T. Biologically defensible — real neural integrators leak; the
    # quantity then tracks "recent forward swim vigour" rather than
    # absolute displacement. Use a new dataset name when generating with
    # a non-None τ so the on-disk recipe stays self-describing
    # (CLAUDE.md: a new variant = a new dataset name).
    xi_tau_s: Optional[float] = None

    # Target representation written by the generator:
    #   "scalar_xi"  (default) — 3-col target [cosθ, sinθ, ξ]. ξ is the
    #                            scalar forward-axis displacement
    #                            ξ = ∫ v_fwd dt (or its leaky variant via
    #                            xi_tau_s). Heading is supervised separately.
    #   "position_2d"         — 4-col target [cosθ, sinθ, x, y]. The fish
    #                            actually moves in 2D: forward swim
    #                            distance is projected through the
    #                            current heading,
    #                                dx/dt = v_fwd · cosθ
    #                                dy/dt = v_fwd · sinθ
    #                            so position integration is COUPLED to
    #                            heading — the network must internally
    #                            maintain θ to predict (x, y) at all.
    #                            This is "true" path integration vs the
    #                            scalar_xi sub-task which is the
    #                            forward-axis projection.
    # The two recipes write different on-disk target shapes, so they must
    # live under separate dataset names — the trainer dispatches on
    # u_train.shape / y_train.shape in concert with training.task_targets.
    #   "place_cells"         — head-direction + distance + PLACE-CELL task.
    #                            The agent forages a bounded square arena
    #                            [-arena_half, +arena_half]² with REFLECTING
    #                            walls (modelled as a stop-and-turn: forward
    #                            motion freezes and the heading rotates to the
    #                            specular-reflected angle over one swim-boxcar
    #                            so θ=∫ω·dt and (x,y)=∫v·dir·dt stay exact and
    #                            the path stays inside the arena). On-disk
    #                            target is 5-col [cosθ, sinθ, ξ, x, y]; the
    #                            K=place_grid² Gaussian place-cell activations
    #                            are NOT stored (a dense (B,T,K) array is
    #                            prohibitively large) but computed on the fly
    #                            from (x,y) and the saved place_centers/σ. A
    #                            second learnable sign-locked E/I network
    #                            (Net2) reads Net1's state and emits the place
    #                            code; its synthetic connectome is generated
    #                            and saved next to the dataset (net2_Wcon.*).
    #   "grid_cells"          — head-direction + distance + GRID-CELL task: the
    #                            place task on a TORUS. The agent forages
    #                            freely (unbounded 2-D path integration, NO
    #                            walls); the K=grid_grid² cells tile a torus
    #                            [0,λ)² (λ=grid_period) and fire
    #                            exp(-d_torus(pos mod λ, c_k)²/2σ²) with WRAPPED
    #                            distance, so each fires on a periodic lattice
    #                            in real space (grid cells). On-disk target is
    #                            5-col [cosθ, sinθ, ξ, x, y] (x,y unbounded);
    #                            the grid code + circular position decode are
    #                            derived on the fly from (x,y) mod λ and the
    #                            saved grid geometry. Canonical toroidal
    #                            continuous-attractor target (cf. Burak & Fiete
    #                            2009; Gardner et al. 2022).
    #   "rotation_torus"      — head-direction + TORUS-POSITION task, read
    #                            directly off Net1 (NO Net2). The agent forages
    #                            freely (unbounded 2-D PI); position is encoded
    #                            as two toroidal phases φ=2π·(x,y)/λ, each a
    #                            (cos,sin) pair like the heading ring. On-disk
    #                            target is 6-col [cosθ, sinθ, cosφx, sinφx,
    #                            cosφy, sinφy]; trained with plain MSE (the
    #                            cos/sin encoding is circular). Net1 must read
    #                            from the full state (output_from_epg_only:
    #                            false), since position lives in PFN/hΔ.
    target_kind: Literal[
        "scalar_xi", "position_2d", "rotation_mismatch", "place_cells",
        "grid_cells", "rotation_torus"
    ] = "scalar_xi"

    # --- Place-cell task (target_kind="place_cells") -----------------------
    # Square arena half-width: the agent is confined to
    # [-arena_half, +arena_half]² by reflecting walls.
    arena_half: float = 1.0
    # K = place_grid² place cells tile the arena on a regular grid; their
    # centres span [-arena_half, +arena_half] on each axis.
    place_grid: int = 20
    # Gaussian place-field width σ (arena units): the target activation of
    # cell k at time t is exp(-‖(x,y)-c_k‖² / (2σ²)). Tunable.
    place_sigma: float = 0.2
    # Net2 — the synthetic sign-locked E/I network that reads Net1's state and
    # produces the place code. Its recurrent connectome is generated at
    # data-generation time and saved as net2_Wcon.* next to the dataset so
    # train/test/plot share one fixed matrix. n_place is derived (place_grid²);
    # total Net2 size = net2_n_interneurons + n_place.
    net2_n_interneurons: int = 200
    net2_sparsity: float = 0.10     # recurrent connection density
    net2_ei_ratio: float = 0.60     # fraction excitatory (Dale's law sign-lock)
    net2_seed: int = 700000
    net2_tau_s: float = 0.1         # Net2 membrane time constant (s)

    # --- Grid-cell task (target_kind="grid_cells") -------------------------
    # Spatial period λ of the torus the grid cells tile (real position units).
    # Each of grid_grid² cells centres on a regular [0,λ)² torus grid; a cell
    # fires whenever (x,y) mod λ is near its centre → a periodic lattice in
    # real space. grid_sigma is the toroidal Gaussian width (real units).
    grid_period: float = 0.5
    grid_grid: int = 20
    grid_sigma: float = 0.1

    # Diagnostic: append the velocity×heading conjunction vx=v·cosθ, vy=v·sinθ
    # as two extra input channels, so the network only has to INTEGRATE (not
    # learn the multiplication) to recover 2-D position. Adds 2 channels to the
    # stimulus (n_input += 2); use with velocity_gate: none (free W_in). Tests
    # whether the position-integration failure is the conjunction or the
    # integrator/attractor.
    conjunction_input: bool = False

    # Leaky-integrator time constant for the 2D position target (x, y).
    # Same semantics as xi_tau_s but applied to the position recurrence:
    #     x(t+Δt) = (1 − Δt/τ) x(t) + v_fwd cosθ · Δt
    #     y(t+Δt) = (1 − Δt/τ) y(t) + v_fwd sinθ · Δt
    # None → perfect 2D integrator (the default; trajectories are
    # unbounded random walks). τ > 0 → bounded "recent egocentric
    # displacement" — the fish-relevant short-range readout. Only
    # applies when target_kind == "position_2d".
    position_tau_s: Optional[float] = None

    # Proprioception-split input layout. When True the on-disk stimulus
    # gains a 3rd column carrying v_proprio so the
    # ``pen_artr_ptipn1_propriocep`` gate's two parallel pathways
    # (v_extero → pt-IPN1, v_proprio → motor_efferent) are anatomically
    # and parameterically independent rather than sharing a single
    # input column. For the first version v_proprio = v_extero =
    # v_fwd (same Poisson swim drive routed twice); per-channel delay
    # or noise can be added later. Stimulus layout becomes
    # ``[ω, v_extero, v_proprio, cos θ₀·δ, sin θ₀·δ]`` (5 channels).
    propriocep_split: bool = False

    # --- Proprioceptive-gain mismatch task (target_kind="rotation_mismatch") ---
    # Models a time-varying discrepancy between the OBSERVED angular velocity
    # ω (the sensory ARTR drive) and the PROPRIOCEPTIVE / effective angular
    # velocity ω_proprio routed to motor_efferent: ω_proprio(t) = g(t) · ω(t),
    # where the gain g(t) is a PIECEWISE-CONSTANT process in
    # [proprio_gain_min, proprio_gain_max] that steps to a new random value
    # every ~proprio_gain_segment_s seconds. The 3rd target column is the
    # integral of the mismatch, ∫(ω − ω_proprio) dt (radians), which the
    # recurrent circuit must recover from the two afferent streams. Forces the
    # 5-channel propriocep_split stimulus layout [ω, v_fwd, ω_proprio, cos0,
    # sin0]. Only used when target_kind == "rotation_mismatch".
    proprio_gain_min: float = 0.0
    proprio_gain_max: float = 1.5
    proprio_gain_segment_s: float = 2.0

    device: Literal["cpu", "cuda", "auto"] = "cpu"

    @model_validator(mode="after")
    def _fractions_sum_to_one(self):
        s = (self.left_fraction + self.right_fraction
             + self.forward_fraction + self.backward_fraction)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(
                f"swim_integration: left/right/forward/backward fractions must "
                f"sum to 1; got {s:.6f}"
            )
        return self


class OpticalFlowTaskConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_trials_train: int
    n_trials_test: int
    n_steps: int = 80
    dt: float = 1.0 / 24            # seconds; video framerate
    seed: int = 42

    flow_target: Literal["sintel_gt", "raft_pseudo", "photometric_selfsup"] = "sintel_gt"
    train_test_split: Literal["random", "video_held_out"] = "video_held_out"
    raft_model: Literal["raft_large", "raft_small"] = "raft_large"

    # Reused flyvis video-source fields (see SimulationConfig.visual_input_type etc.).
    # Used by raft_pseudo / photometric_selfsup; ignored for sintel_gt.
    datavis_roots: List[str] = []
    truncate_max_frames: Optional[int] = Field(default=80, gt=0)
    flywire_stimulus: bool = False
    all_columns: bool = False
    steady_state_value: float = 0.5

    device: Literal["cpu", "cuda", "auto"] = "auto"
    input_perturbation: Optional[InputPerturbation] = None


class CortexTaskConfig(BaseModel):
    """Yang et al. 2019 multitask cognitive battery (gyyang/multitask port).

    The Yang generator (`generators/cortex_task.py`) defines task dimensions
    from `ruleset`: for ruleset='all', N_i = 1 + 2*32 + 20 = 85 (fixation +
    two stimulus rings + 20-rule one-hot), N_o = 1 + 32 = 33 (fixation +
    motor ring). Per-trial tdim varies; trials get padded to `n_steps_max`.
    """
    model_config = ConfigDict(extra="ignore")

    # Task selection
    rules: List[str]                                      # Yang task names (subset of ruleset)
    rule_weights: List[float] = []                        # empty = uniform sampling
    ruleset: Literal["all", "mante", "oicdmc"] = "all"

    # Trial counts
    n_trials_train: int
    n_trials_test: int
    n_steps_max: int = 200                                # padding length; raises if any trial exceeds

    # Yang hp overrides — passed through `get_default_hp(ruleset)` then
    # mutated. Use to tweak dt, tau, sigma_x, sigma_rec, etc. Empty = Yang defaults.
    hp_overrides: Dict[str, Any] = {}
    seed: int = 0

    device: Literal["cpu"] = "cpu"
    input_perturbation: Optional[InputPerturbation] = None


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_type: Literal["path_integration", "swim_integration",
                        "optical_flow", "cortex"]

    path_integration: Optional[PathIntegrationTaskConfig] = None
    swim_integration: Optional[SwimIntegrationTaskConfig] = None
    optical_flow: Optional[OpticalFlowTaskConfig] = None
    cortex: Optional[CortexTaskConfig] = None

    # If True, the data_generate dispatcher returns immediately after writing
    # task data — skipping the (still-required-by-schema) simulation pipeline.
    # Default False so configs that legitimately want both task+sim still work.
    task_only: bool = False

    @model_validator(mode="after")
    def _exactly_matching_subblock(self):
        sub = {
            "path_integration": self.path_integration,
            "swim_integration": self.swim_integration,
            "optical_flow": self.optical_flow,
            "cortex": self.cortex,
        }
        present = [k for k, v in sub.items() if v is not None]
        if present != [self.task_type]:
            raise ValueError(
                f"task_type={self.task_type!r} requires exactly the matching "
                f"subblock to be populated; got populated={present}"
            )
        return self


class CircuitConfig(BaseModel):
    """Optional named-circuit selector — sister of TaskConfig.

    When ``name`` is set, the model class resolves the connectome via
    ``connectome_gnn.generators.circuits.get_circuit(name)`` (the named
    registry of pre-cached, sign-locked, spectrally-rescaled adjacency
    templates). When unset (default), the model falls through to the
    legacy ``load_<organism>_*_connectome(sim.connconstr_datapath)``
    path — so existing yamls keep loading byte-equivalently.

    See ``docs/REFACTOR_zebrafish_circuit_registry.md`` §4 for the
    motivation.
    """
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None


class NeuralGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = "connectome_gnn"
    dataset: str
    data_folder_name: str = "none"
    connectome_folder_name: str = "none"
    data_folder_mesh_name: str = "none"
    config_file: str = "none"


    simulation: SimulationConfig
    graph_model: GraphModelConfig
    claude: Optional[ClaudeConfig] = None
    claude_code: Optional[ClaudeCodeConfig] = None
    plotting: PlottingConfig
    training: TrainingConfig
    zarr: Optional[ZarrConfig] = None
    task: Optional[TaskConfig] = None
    circuit: Optional[CircuitConfig] = None

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, "r") as file:
            raw_config = yaml.safe_load(file)
        return NeuralGraphConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == "__main__":
    config_file = "../../config/arbitrary_3.yaml"  # Insert path to config file
    config = NeuralGraphConfig.from_yaml(config_file)
    print(config.pretty())

    print("Successfully loaded config file. Model description:", config.description)
