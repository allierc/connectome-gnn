from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional


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
    calcium_tau: float = 0.5  # decay time constant (same units as delta_t)
    calcium_alpha: float = 1.0  # scale factor to convert [Ca] to fluorescence
    calcium_beta: float = 0.0  # baseline offset for fluorescence
    calcium_initial: float = 0.0  # initial calcium concentration
    calcium_noise_level: float = 0.0  # optional Gaussian noise added to [Ca] updates
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
    calcium_saturation_kd: float = 1.0  # for nonlinear saturation models
    calcium_num_compartments: int = 1
    calcium_down_sample: int = 1  # down-sample [Ca] time series by this factor
    save_calcium: bool = False  # whether to save calcium/fluorescence in zarr output

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

    g_phi_positive: bool = False

    update_type: UpdateType = UpdateType.NONE

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


class TrainingConfig(BaseModel):
    # allow: LLM_code agents introduce new coeff_<name> keys per block; they
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
    lr_update: float = 0.0
    lr_W: float = 0.0001

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
    coeff_g_phi_diff: float = 0  # Variance penalty on g_phi output across edges
    coeff_g_phi_norm: float = 0  # Norm penalty on g_phi edge messages
    coeff_func_g_phi: float = 0.0  # Penalize g_phi output at zero input
    coeff_g_phi_weight_L1: float = 0  # L1 penalty on g_phi MLP weights
    coeff_g_phi_weight_L2: float = 0  # L2 penalty on g_phi MLP weights

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

    # -- SPEND-style Noise2Noise add-ons (graph_trainer_spend.py) --
    # Three N2N variants for measurement-noise data; consumed by data_train_spend.
    # Cite: https://github.com/buchenglab/SPEND  (Ding et al. 2025, Newton 1, 100195)
    coeff_spend_replay: float = 0.0          # Add-on #3 — stimulus-replay N2N weight (synth two noise seeds)
    coeff_spend_time: float = 0.0            # Add-on #1 — time-permutation N2N weight (even/odd frames)
    coeff_spend_typed: float = 0.0           # Add-on #2 — typed-equivariance loss (same-type neuron pairs)
    spend_load_clean: bool = False           # if True, load with measurement_noise_level=0 then synth noise inline
    spend_replay_noise_seed_a: int = 0       # RNG seed for first noise realisation
    spend_replay_noise_seed_b: int = 1       # RNG seed for second noise realisation (must differ from a)
    spend_time_window: int = 16              # frames per time-permutation N2N window
    spend_smoother_hidden: int = 32          # 1D-conv smoother hidden channels
    spend_smoother_lr: float = 1.0e-3        # separate LR for smoother param group
    spend_typed_max_pos_dist: float = 5.0    # max retinotopic position distance for typed-pair construction

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

    # external input learning
    learn_external_input: bool = False

    save_all_checkpoints: bool = False  # True = save iteration-level checkpoints too

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
    use_gt_edges: bool = False  # True = use ground truth edge_index; False = fully connected graph
    w_init_mode: WInitMode = WInitMode.RANDN  # randn=std=1, randn_scaled=std=scale/sqrt(N), zeros
    w_init_scale: float = 1.0  # scaling factor for 'randn_scaled' mode
    coeff_W_L1_proximal: float = 0.0  # proximal L1 soft-thresholding on W after optimizer step, 0 = disabled
    dale_law: bool = False  # enforce Dale's law: force each column of W to a consistent sign, 3 times per epoch

    alternate_training: bool = False  # two-stage training: joint warmup then V_rest focus
    alternate_joint_ratio: float = 0.4  # fraction of total iterations for joint phase (all components at full LR)
    alternate_lr_ratio: float = 0.1  # LR multiplier for W/g_phi during V_rest focus phase

    # Learning rate scheduler
    lr_scheduler: str = "none"  # 'none' | 'cosine_warm_restarts' | 'linear_warmup_cosine'
    lr_scheduler_T0: int = 1000  # restart period in iterations
    lr_scheduler_T_mult: int = 2  # period multiplier after each restart
    lr_scheduler_eta_min_ratio: float = 0.01  # min LR as fraction of base LR
    lr_scheduler_warmup_iters: int = 100  # linear warmup iterations

    time_step: int = 1
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



# ---------------------------------------------------------------------------
# Task-data generation (input stimulus + target output) — see plan
# /home/node/.claude/plans/structured-swimming-pearl.md. PR1 lands the schema
# for all three task families (path_integration, optical_flow, twenty_tasks)
# and the PI generator; OF and twenty-tasks generators are PR2/PR3.
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
    input_perturbation: Optional[InputPerturbation] = None


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


class TwentyTasksConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subtasks: List[str]
    n_trials_train_per_subtask: int
    n_trials_test_per_subtask: int
    dt: float = 0.020               # seconds (Yang default 20 ms)
    n_steps_max: int = 80
    seed: int = 42

    sigma_x: float = 0.01
    dataset_balance: Literal["uniform", "weighted"] = "uniform"

    device: Literal["cpu"] = "cpu"
    input_perturbation: Optional[InputPerturbation] = None


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_type: Literal["path_integration", "optical_flow", "twenty_tasks"]

    path_integration: Optional[PathIntegrationTaskConfig] = None
    optical_flow: Optional[OpticalFlowTaskConfig] = None
    twenty_tasks: Optional[TwentyTasksConfig] = None

    output_subdir: str = "task"

    # If True, the data_generate dispatcher returns immediately after writing
    # task data — skipping the (still-required-by-schema) simulation pipeline.
    # Default False so configs that legitimately want both task+sim still work.
    task_only: bool = False

    @model_validator(mode="after")
    def _exactly_matching_subblock(self):
        sub = {
            "path_integration": self.path_integration,
            "optical_flow": self.optical_flow,
            "twenty_tasks": self.twenty_tasks,
        }
        present = [k for k, v in sub.items() if v is not None]
        if present != [self.task_type]:
            raise ValueError(
                f"task_type={self.task_type!r} requires exactly the matching "
                f"subblock to be populated; got populated={present}"
            )
        return self


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
