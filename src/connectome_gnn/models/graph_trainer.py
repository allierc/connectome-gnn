import logging
import os
import shutil
import time
import warnings

# Suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# Suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from connectome_gnn.figure_style import default_style
from connectome_gnn.log import get_logger
from connectome_gnn.metrics import compute_dynamics_r2
from connectome_gnn.models.neural_ode_wrapper import (
    debug_check_gradients,
    neural_ode_loss,
)
from connectome_gnn.models.recurrent_step import recurrent_loss
from connectome_gnn.models.registry import create_model
from connectome_gnn.models.training_utils import build_lr_scheduler, build_model, dale_law_score, determine_load_fields, enforce_dale_law, load_flyvis_data
from connectome_gnn.models.utils import (
    ANSI_GREEN,
    ANSI_ORANGE,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
    LossRegularizer,
    _NGP_QUICK_FREQ,
    _batch_frames,
    _quick_ngp_pearson,
    analyze_data_svd,
    r2_color,
    set_trainable_parameters,
)
from connectome_gnn.plot import (
    plot_jacobian_w_scatter,
    plot_metrics,
    plot_signal_loss,
    plot_training_flyvis,
    plot_training_linear,
    plot_training_summary_panels,
    render_visual_field_video,
)
from connectome_gnn.sparsify import clustering_evaluation, umap_cluster_reassign
from connectome_gnn.utils import (
    CustomColorMap,
    check_and_clear_memory,
    create_log_dir,
    graphs_data_path,
    to_numpy,
)

_logger = get_logger(__name__)


def data_train(config=None, erase=False, best_model=None, style=None, device=None, log_file=None):
    # plt.rcParams['text.usetex'] = False  # LaTeX disabled - use mathtext instead
    # rc('font', **{'family': 'serif', 'serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif', 'serif']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    # Limit CPU threads to match cluster allocation (LSB_DJOB_NUMPROC set by bsub -n)
    num_proc = os.environ.get("LSB_DJOB_NUMPROC")
    # Limit torch.compile's Triton compilation workers to cluster allocation
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", num_proc or "12")

    if num_proc is not None and (device is None or 'cpu' in str(device)):
        torch.set_num_threads(int(num_proc))
        print(f"CPU threads: {num_proc} (from LSB_DJOB_NUMPROC)")

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    _logger.info(f"dataset: {config.dataset}")
    _logger.info(f"{config.description}")

    # Task-data trainer (path_integration etc.). Detected via the presence
    # of a populated task block — keeps train_subprocess.py / GNN_Main.py
    # routing transparent for both the LLM agentic loop and direct CLI use.
    if getattr(config, 'task', None) is not None:
        data_train_task(config, erase, best_model, device, log_file=log_file)
        _logger.info("training completed.")
        return

    _connconstr = any(x in config.dataset for x in ('drosophila_cx', 'zebrafish_oculomotor', 'larva'))
    _cortex_voltage = 'cortex' in config.dataset
    if 'fly' in config.dataset or _connconstr or _cortex_voltage:
        model_name = config.graph_model.signal_model_name.lower()
        if 'stimulus' in model_name:
            from connectome_gnn.models.data_train_stimulus import data_train_stimulus
            data_train_stimulus(config, erase, best_model, device, log_file=log_file)
        elif 'eed' in model_name and 'rnn' not in model_name:
            from connectome_gnn.models.data_train_eed import data_train_eed
            data_train_eed(config, erase, best_model, device, log_file=log_file)
        elif ('mlp' in model_name) and 'rnn' not in model_name:
            from connectome_gnn.models.data_train_rollout import data_train_rollout
            data_train_rollout(config, erase, best_model, device, log_file=log_file)
        elif 'rnn' in model_name or 'lstm' in model_name:
            data_train_gnn_RNN(config, erase, best_model, device)
        else:
            data_train_gnn(config, erase, best_model, device, log_file=log_file)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset}")

    _logger.info("training completed.")


def data_train_gnn(config, erase, best_model, device, log_file=None):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    replace_with_cluster = 'replace' in tc.sparsity
    umap_cluster_active = tc.umap_cluster_method != 'none'

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(config.training.seed)

    default_style.apply_globally()

    if 'visual' in model_config.field_type:
        has_visual_field = True
        if 'instantNGP' in model_config.field_type:
            _logger.info('train with visual field instantNGP')
        else:
            _logger.info('train with visual field NNR')
    else:
        has_visual_field = False
    if 'test' in model_config.field_type:
        test_neural_field = True
        _logger.info('train with test field NNR')
    else:
        test_neural_field = False

    log_dir, logger = create_log_dir(config, erase)

    load_fields = determine_load_fields(config)
    x_ts, y_ts, type_list = load_flyvis_data(
        config.dataset, split='train', fields=load_fields,
        training_selected_neurons=tc.training_selected_neurons,
        selected_neuron_ids=tc.selected_neuron_ids if tc.training_selected_neurons else None,
        measurement_noise_level=sim.measurement_noise_level,
    )

    # get n_neurons and n_frames from data, not config file
    n_neurons = x_ts.n_neurons
    config.simulation.n_neurons = n_neurons
    n_frames_raw = x_ts.n_frames
    sim.n_frames = n_frames_raw
    _logger.info(f'dataset: {n_frames_raw} frames,  n neurons: {n_neurons}')
    logger.info(f'n neurons: {n_neurons}')

    # Subsample every time_step frames for recurrent training to reduce GPU memory.
    # After subsampling, consecutive frames in x_ts are time_step original steps apart,
    # so the BPTT unroll of time_step steps spans exactly the same physical duration
    # as before, but GPU memory scales with n_frames/time_step instead of n_frames.
    stride = tc.time_step if (tc.recurrent_training and tc.time_step > 1) else 1
    if stride > 1:
        from tqdm import tqdm as _tqdm
        _fields_to_stride = ['voltage', 'stimulus', 'calcium', 'fluorescence', 'noise']
        print(f"\033[93msubsampling dataset: {n_frames_raw} frames → {n_frames_raw // stride} frames "
              f"(1 every {stride} steps for recurrent training with time_step={stride})\033[0m")
        for _field in _tqdm(_fields_to_stride, desc='subsampling x_ts', ncols=150):
            _val = getattr(x_ts, _field)
            if _val is not None:
                setattr(x_ts, _field, _val[::stride])
        y_ts = y_ts[::stride]
        sim.n_frames = x_ts.n_frames  # update after subsampling

    # Compute xnorm on CPU before moving to GPU (avoids OOM from temporary
    # boolean mask + filtered copy needing ~2x voltage memory)
    xnorm = x_ts.xnorm
    assert not torch.isnan(x_ts.voltage).any(), "voltage contains NaN — cannot train"
    assert not np.isnan(y_ts).any(), "derivative targets contain NaN — cannot train"
    x_ts = x_ts.to(device)
    # Block 01: temporal voltage denoising (reduces measurement noise in GNN input)
    _denoise_alpha = float(getattr(tc, 'coeff_voltage_denoise_alpha', 0.0))
    if _denoise_alpha > 0:
        from connectome_gnn.LLM_code.staging.block_01.temporal_voltage_denoise import temporal_voltage_denoise
        x_ts.voltage = (1 - _denoise_alpha) * x_ts.voltage + _denoise_alpha * temporal_voltage_denoise(x_ts.voltage)
        _logger.info(f'voltage denoising applied: alpha={_denoise_alpha}')
    y_ts_gpu = torch.from_numpy(y_ts).float().to(device)  # pre-convert once; avoids per-iter cudaStreamSynchronize
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    _logger.info(f'xnorm: {to_numpy(xnorm):0.3f}')
    logger.info(f'xnorm: {to_numpy(xnorm)}')
    xnorm = float(xnorm)  # Python float so compiled functions avoid .item()
    ynorm = torch.tensor(1.0, device=device)
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    _logger.info(f'ynorm: {to_numpy(ynorm):0.3f}')
    logger.info(f'ynorm: {to_numpy(ynorm)}')
    ynorm = float(ynorm)

    # SVD analysis of activity and visual stimuli (skip if already exists)
    svd_plot_path = os.path.join(log_dir, 'results', 'svd_analysis.png')
    if not os.path.exists(svd_plot_path):
        analyze_data_svd(x_ts, log_dir, config=config, logger=logger, is_flyvis=True)
    else:
        _logger.info(f'svd analysis already exists: {svd_plot_path}')

    # Load edges early so n_edges is correct before model creation
    from connectome_gnn.generators.ode_params import FlyVisODEParams, get_ode_params_class
    signal_model = config.graph_model.signal_model_name
    try:
        OdeParamsCls = get_ode_params_class(signal_model)
    except KeyError:
        OdeParamsCls = FlyVisODEParams
    try:
        ode_params = OdeParamsCls.load(graphs_data_path(config.dataset), device=device)
    except TypeError:
        # Schema mismatch — on-disk ode_params.pt was saved with a different
        # dataclass (typical when the same signal_model_name maps to two
        # different ODE param schemas, e.g. `drosophila_cx` is registered to
        # both DrosophilaCxODEParams (legacy Hulse-Beiran teacher) and to
        # the simpler FlyVisODEParams (voltage-recovery flow). Fall back to
        # FlyVisODEParams which only requires edge_index / W / tau_i / V_i_rest.
        _logger.info(
            f'ode_params schema mismatch for {OdeParamsCls.__name__}; '
            f'falling back to FlyVisODEParams'
        )
        ode_params = FlyVisODEParams.load(graphs_data_path(config.dataset), device=device)
    gt_weights = ode_params.W
    gt_edges = ode_params.edge_index

    _G = '\033[92m'; _R = '\033[91m'; _X = '\033[0m'
    _match = gt_edges.shape[1] == sim.n_edges
    _c = _G if _match else _R
    print(f"{_c}[TRAIN] loaded ode_params: edge_index={gt_edges.shape}  W={gt_weights.shape}  "
          f"config.n_edges={sim.n_edges}  {'OK' if _match else 'MISMATCH'}{_X}")

    # Optionally replace GT edges with fully connected graph
    if not tc.use_gt_edges:
        src = torch.arange(n_neurons, device=device).repeat_interleave(n_neurons)
        dst = torch.arange(n_neurons, device=device).repeat(n_neurons)
        # Remove self-loops
        mask = src != dst
        edges = torch.stack([src[mask], dst[mask]], dim=0)
        _logger.info(f'fully connected edges: {edges.shape[1]} (GT had {gt_edges.shape[1]})')
        config.simulation.n_edges = edges.shape[1]
        # Remap GT weights to fully connected edge ordering for R² evaluation
        gt_weight_map = torch.zeros(edges.shape[1], device=device)
        gt_edge_set = {}
        for k in range(gt_edges.shape[1]):
            gt_edge_set[(gt_edges[0, k].item(), gt_edges[1, k].item())] = gt_weights[k]
        for k in range(edges.shape[1]):
            key = (edges[0, k].item(), edges[1, k].item())
            if key in gt_edge_set:
                gt_weight_map[k] = gt_edge_set[key]
        gt_weights = gt_weight_map
    else:
        edges = gt_edges
        actual_n_edges = edges.shape[1]
        expected_total = sim.n_edges + sim.n_extra_null_edges
        if actual_n_edges == expected_total and sim.n_extra_null_edges > 0:
            _logger.info(f'null edges in data: {sim.n_edges} base + {sim.n_extra_null_edges} null = {actual_n_edges}')
            # Model W must cover all edges (real + null); update n_edges so build_model
            # allocates W of size actual_n_edges, not just the base n_edges.
            config.simulation.n_edges = actual_n_edges
            config.simulation.n_extra_null_edges = 0
        elif actual_n_edges != sim.n_edges:
            _logger.info(f'n_edges mismatch: config={sim.n_edges}, actual={actual_n_edges} — using actual')
            print(f"{_R}[TRAIN] n_edges override: config={sim.n_edges} → actual={actual_n_edges}{_X}")
            config.simulation.n_edges = actual_n_edges

    print(f"{_G}[TRAIN] edges for model: {edges.shape}  config.n_edges now={config.simulation.n_edges}{_X}")

    # Save training edges so tester uses the same graph
    torch.save(edges, os.path.join(log_dir, 'training_edges.pt'))
    torch.save(gt_weights, os.path.join(log_dir, 'gt_weights.pt'))
    print(f"{_G}[TRAIN] saved training_edges.pt: {edges.shape}  gt_weights.pt: {gt_weights.shape}{_X}")

    # Resolve checkpoint path from best_model argument
    checkpoint_path = None
    if best_model and best_model != '' and best_model != 'None':
        checkpoint_path = f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{best_model}.pt"
    elif tc.pretrained_model != '':
        checkpoint_path = tc.pretrained_model

    reset_epoch = (tc.pretrained_model != '' and not best_model)
    model, start_epoch = build_model(config, device, checkpoint_path=checkpoint_path, reset_epoch=reset_epoch)
    _w = model.W if hasattr(model, 'W') else None
    _w_match = _w is not None and _w.shape[0] == config.simulation.n_edges
    _c = _G if _w_match else _R
    print(f"{_c}[TRAIN] model.W={_w.shape if _w is not None else 'N/A'}  "
          f"config.n_edges={config.simulation.n_edges}  "
          f"{'OK' if _w_match else 'MISMATCH'}{_X}")
    list_loss = []

    # Initialize embedding with equidistant points per cell type
    if tc.embedding_cell_type_init:
        from connectome_gnn.utils import get_equidistant_points
        n_types = sim.n_neuron_types
        emb_dim = config.graph_model.embedding_dim
        if emb_dim != 2:
            _logger.warning(f'embedding_cell_type_init requires embedding_dim=2, got {emb_dim} — skipping')
        else:
            scale = tc.embedding_cell_type_scale
            ex, ey = get_equidistant_points(n_types)
            equidist_pts = np.stack([ex, ey], axis=1) * scale  # (n_types, 2)
            type_ids = type_list.squeeze(-1).long().cpu().numpy()  # (n_neurons,)
            with torch.no_grad():
                model.a.copy_(torch.tensor(equidist_pts[type_ids], dtype=torch.float32, device=device))
            _logger.info(f'embedding initialized with equidistant points for {n_types} cell types')

    # Freeze embedding if requested (must be done before optimizer build)
    if tc.fix_embedding:
        model.a.requires_grad_(False)
        _logger.info('embedding is fixed (requires_grad=False, excluded from optimizer)')

    # W init mode info
    w_init_mode = getattr(tc, 'w_init_mode', 'randn')
    if w_init_mode != 'randn':
        w_init_scale = getattr(tc, 'w_init_scale', 1.0)
        _logger.info(f'W init mode: {w_init_mode}' + (f' (scale={w_init_scale})' if w_init_mode == 'randn_scaled' else ''))

    # === LLM-MODIFIABLE: OPTIMIZER SETUP START ===
    # Change optimizer type, learning rate schedule, parameter groups

    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'total parameters: {n_total_params:,}')
    lr = tc.lr
    if tc.lr_update == 0:
        lr_update = tc.lr
    else:
        lr_update = tc.lr_update
    lr_embedding = tc.lr_embedding
    lr_W = tc.lr_W
    lr_NNR_f = tc.lr_NNR_f

    # Two-phase NNR schedule (SIREN-style).
    #   tc.training_NNR_start_epoch > 0  -> NNR (NNR_f and NNR_hidden) is
    #     frozen at lr_NNR_f_start during the warmup epochs [0, start_epoch),
    #     then catches up at full lr_NNR_f from epoch=start_epoch onward.
    #   The catch-up switch is fired by the existing alternate_training block
    #     below (which also scales the GNN lr's by alternate_lr_ratio so the
    #     graph backbone stops drifting while the NNR converges).
    nnr_warmup_epochs = int(getattr(tc, 'training_NNR_start_epoch', 0))
    lr_NNR_f_start = float(getattr(tc, 'lr_NNR_f_start', 0.0))
    lr_NNR_f_init = lr_NNR_f_start if nnr_warmup_epochs > 0 else lr_NNR_f

    _logger.info(
        f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, '
        f'lr_embedding {lr_embedding}, lr_NNR_f {lr_NNR_f_init} '
        f'(NNR warmup: {nnr_warmup_epochs} epoch(s) at {lr_NNR_f_start}, '
        f'then {lr_NNR_f})'
    )

    optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr,
                                                         lr_update=lr_update, lr_W=lr_W, lr_NNR_f=lr_NNR_f_init)

    lr_scheduler = build_lr_scheduler(optimizer, config)
    scheduler_type = getattr(tc, 'lr_scheduler', 'none')
    if scheduler_type != 'none':
        _logger.info(f'LR scheduler: {scheduler_type}')
    # === LLM-MODIFIABLE: OPTIMIZER SETUP END ===
    model.train()

    net = f"{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs.pt"
    _logger.info(f'network: {net}')
    _logger.info(f'initial tc.batch_size: {tc.batch_size}')

    ids = torch.arange(n_neurons, device=device)

    # --- Hidden neuron setup ---
    hidden_ids = None
    visible_ids = ids  # default: all neurons visible
    _hidden_frac = getattr(model_config, 'hidden_neuron_fraction', 0.0)
    has_hidden_neurons = _hidden_frac > 0.0
    if has_hidden_neurons:
        _hidden_path = os.path.join(log_dir, 'hidden_neuron_ids.pt')
        if os.path.exists(_hidden_path):
            hidden_ids = torch.load(_hidden_path, map_location=device, weights_only=True)
            logger.info(f'loaded {len(hidden_ids)} hidden neurons from checkpoint')
        else:
            _rng = np.random.RandomState(sim.seed)
            _candidates = np.arange(sim.n_input_neurons, n_neurons)
            _n_hidden = int(len(_candidates) * _hidden_frac)
            _hidden_np = np.sort(_rng.choice(_candidates, size=_n_hidden, replace=False))
            hidden_ids = torch.from_numpy(_hidden_np).long().to(device)
            torch.save(hidden_ids, _hidden_path)
            logger.info(f'sampled {len(hidden_ids)} hidden neurons ({_hidden_frac*100:.1f}%), saved')
        _hidden_mask = torch.zeros(n_neurons, dtype=torch.bool, device=device)
        _hidden_mask[hidden_ids] = True
        visible_ids = ids[~_hidden_mask]
        _logger.info(f'hidden neurons: {len(hidden_ids)}/{n_neurons}, visible for loss: {len(visible_ids)}')

    # --- Anchor neuron setup ---
    # Anchors are OBSERVED neurons whose GT voltages directly supervise the
    # NGP-T / SIREN-T backbone. Not cheating: anchors are sampled from the
    # visible (non-hidden) set.
    anchor_ids = None
    _inr_type_hidden = getattr(model_config, 'inr_type_hidden', 'none')
    _anchor_inner_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    has_anchor_neurons = (
        has_hidden_neurons
        and bool(getattr(tc, 'train_with_anchor_neurons', False))
        and _inr_type_hidden in ('siren_t', 'ngp_t')
        and getattr(_anchor_inner_model, 'n_anchor', 0) > 0
    )
    if has_anchor_neurons:
        _anchor_path = os.path.join(log_dir, 'anchor_neuron_ids.pt')
        _n_anchor = int(_anchor_inner_model.n_anchor)
        if os.path.exists(_anchor_path):
            anchor_ids = torch.load(_anchor_path, map_location=device, weights_only=True)
            if len(anchor_ids) != _n_anchor:
                _logger.warning(f'anchor_ids size {len(anchor_ids)} != model.n_anchor {_n_anchor}; re-sampling')
                anchor_ids = None
        if anchor_ids is None:
            _rng = np.random.RandomState(sim.seed + 1)
            _candidates = np.setdiff1d(
                np.arange(sim.n_input_neurons, n_neurons),
                hidden_ids.cpu().numpy(),
            )
            _n_anchor_eff = min(_n_anchor, len(_candidates))
            _anchor_np = np.sort(_rng.choice(_candidates, size=_n_anchor_eff, replace=False))
            anchor_ids = torch.from_numpy(_anchor_np).long().to(device)
            torch.save(anchor_ids, _anchor_path)
            _logger.info(f'sampled {len(anchor_ids)} anchor neurons, saved')
        else:
            _logger.info(f'loaded {len(anchor_ids)} anchor neurons from checkpoint')

    if tc.coeff_W_sign > 0:
        index_weight = []
        for i in range(n_neurons):
            # get source neurons that connect to neuron i
            mask = edges[1] == i
            index_weight.append(edges[0][mask])

    _logger.info(f'coeff_W_L1: {tc.coeff_W_L1} coeff_g_phi_diff: {tc.coeff_g_phi_diff} coeff_f_theta_diff: {tc.coeff_f_theta_diff}')
     # proximal L1 info
    coeff_proximal = getattr(tc, 'coeff_W_L1_proximal', 0.0)
    if coeff_proximal > 0:
        _logger.info(f'proximal L1 soft-thresholding on W: coeff={coeff_proximal}')

    _logger.info("start training ...")

    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)
    # torch.autograd.set_detect_anomaly(True)

    list_loss_regul = []

    regularizer = LossRegularizer(
        train_config=tc,
        model_config=model_config,
        activity_column=3,  # flyvis uses column 3 for activity
        plot_frequency=1,   # will be updated per epoch
        n_neurons=n_neurons,
        trainer_type='flyvis',
        dataset=config.dataset,
        type_list=type_list,
        n_neuron_types=sim.n_neuron_types,
    )
    regularizer.set_activity_stats(x_ts, device)
    regularizer.move_type_list_to_device(device)

    # Try to compile with torch.compile if enabled, but fall back to non-compiled if Triton fails
    if tc.torch_compile:
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
        regularizer.compute = torch.compile(regularizer.compute, mode='reduce-overhead', fullgraph=True)
        regularizer.compute_update_regul = torch.compile(regularizer.compute_update_regul, mode='reduce-overhead', fullgraph=True)
        logger.info("torch.compile enabled")
    else:
        logger.info("torch.compile disabled via config (torch_compile: false)")

    loss_components = {'loss': []}

    training_start_time = time.time()

    # Metrics log: tracks R2 evolution over training iterations
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    with open(metrics_log_path, 'w') as f:
        f.write('iteration,connectivity_r2,vrest_r2,tau_r2,hidden_nnr_pearson,anchor_nnr_pearson,'
                'vrest_r2_clean,n_out_vrest,n_total_vrest,'
                'tau_r2_clean,n_out_tau,n_total_tau\n')

    # Total iter count across all epochs — read by the LLM poller to display
    # iter=I/total in the periodic [metrics] line. Mirrors the Niter formula
    # used inside the epoch loop (deterministic per epoch).
    _Niter_per_epoch = int(sim.n_frames * tc.data_augmentation_loop // tc.batch_size * 0.2)
    if tc.max_iterations_per_epoch > 0:
        _Niter_per_epoch = min(_Niter_per_epoch, tc.max_iterations_per_epoch)
    _total_iter = _Niter_per_epoch * tc.n_epochs
    with open(os.path.join(log_dir, 'tmp_training', 'total_iter.txt'), 'w') as f:
        f.write(str(_total_iter))

    # NNR pearson log: tracks per-neuron Pearson r mean ± SD across
    # training iterations (populated only when an NGP/SIREN hidden head
    # is active). Consumed by plot_signal_loss to draw the mean+SD panel.
    nnr_pearson_log_path = os.path.join(log_dir, 'tmp_training', 'nnr_pearson.log')
    with open(nnr_pearson_log_path, 'w') as f:
        f.write('iteration,hidden_pearson_mean,hidden_pearson_std,'
                'anchor_pearson_mean,anchor_pearson_std\n')

    def _fmt_metric(x):
        """Format optional float for metrics.log CSV ('nan' when None)."""
        return 'nan' if x is None else f'{x:.6f}'

    # Valid frame range for sampling (matches np.random.randint logic it replaces)
    _frame_min_k = tc.time_window
    _stride_subsample = tc.recurrent_training and tc.time_step > 1
    _target_offset = 1 if _stride_subsample else tc.time_step
    _frame_max_k = sim.n_frames - 4 - _target_offset  # exclusive upper bound
    _frame_range = max(_frame_max_k - _frame_min_k, 1)

    embedding_frozen = False
    unfreeze_at_iteration = -1

    _profiling = tc.profiling
    _profiler_trace_dir = os.path.join(log_dir, 'profiler_traces')
    if _profiling:
        os.makedirs(_profiler_trace_dir, exist_ok=True)

    for epoch in range(start_epoch, tc.n_epochs):

        Niter = int(sim.n_frames * tc.data_augmentation_loop // tc.batch_size * 0.2)
        plot_frequency = max(1, int(Niter // 20))
        # Heavy R² checkpoint cadence — doubled (Niter//10 → Niter//20) so
        # nnr_plot.png gets twice the V_rest/τ-clean steps. Heavy plots
        # (embedding/W) still fire at the lower plot_frequency cadence.
        connectivity_plot_frequency = max(1, int(Niter // 20))
        # Early-phase R2: 4 extra checkpoints in [1, connectivity_plot_frequency)
        early_r2_frequency = connectivity_plot_frequency // 5
        n_plots_per_epoch = 4
        plot_iterations = set(int(x) for x in np.linspace(Niter // n_plots_per_epoch, Niter - 1, n_plots_per_epoch)) if n_plots_per_epoch > 0 else set()
        print(f'every {connectivity_plot_frequency} iterations: {Niter} iterations per epoch, plot '
              f'(early-phase every {early_r2_frequency} iterations)')

        # TRUNCATE ITERATIONS but only if config parameter says so.
        if tc.max_iterations_per_epoch > 0:
            Niter = min(Niter, tc.max_iterations_per_epoch)
        # Compute unfreeze point for this epoch if embedding was frozen by UMAP clustering
        if embedding_frozen and tc.umap_cluster_fix_embedding_ratio > 0:
            unfreeze_at_iteration = int(Niter * tc.umap_cluster_fix_embedding_ratio)
        else:
            unfreeze_at_iteration = -1

        total_loss = 0
        total_loss_regul = 0
        _total_loss_gpu = torch.zeros((), device=device)      # GPU accumulators — avoids per-iter .item() sync
        _total_regul_gpu = torch.zeros((), device=device)
        k = 0

        loss_noise_level = tc.loss_noise_level * (0.95 ** epoch)
        regularizer.set_epoch(epoch, plot_frequency, Niter=Niter)

        # Per-epoch resampling of measurement noise: overwrite x_ts.noise with a
        # fresh Gaussian realisation seeded by sim.seed + epoch so the GNN sees
        # an independent noise draw on every pass. Replaces the fixed noise.zarr
        # realisation; only active when measurement noise is enabled.
        if tc.resample_noise_per_epoch and sim.measurement_noise_level > 0 and x_ts.noise is not None:
            _noise_gen = torch.Generator(device=x_ts.noise.device).manual_seed(int(sim.seed) + int(epoch))
            x_ts.noise = (torch.randn(x_ts.noise.shape, generator=_noise_gen,
                                       dtype=x_ts.noise.dtype, device=x_ts.noise.device)
                          * sim.measurement_noise_level)
            _logger.info(f'epoch {epoch}: resampled measurement noise (seed={int(sim.seed) + int(epoch)})')

        # Two-phase training: epoch 0 = full LRs, epoch 1+ = reduce W/MLP, keep SIREN
        if tc.alternate_training and epoch >= 1:
            phase_mult = tc.alternate_lr_ratio
            optimizer, n_total_params = set_trainable_parameters(
                model=model,
                lr_embedding=lr_embedding * phase_mult,
                lr=lr * phase_mult,
                lr_update=lr_update * phase_mult,
                lr_W=lr_W * phase_mult,
                lr_NNR_f=lr_NNR_f,
            )
            lr_scheduler = build_lr_scheduler(optimizer, config)
            _logger.info(f'Phase 1 (SIREN focus): W/MLP LRs *= {phase_mult}, NNR_f LR = {lr_NNR_f}')

        # Reproducible per-epoch frame sampling (replaces bare np.random.randint)
        epoch_rng = np.random.RandomState((tc.seed + epoch) % (2**32))
        frame_indices = epoch_rng.randint(0, _frame_range, size=Niter * tc.batch_size) + _frame_min_k

        last_connectivity_r2 = None
        last_connectivity_r2_visible = None
        last_vrest_r2 = 0.0
        last_tau_r2 = 0.0
        last_vrest_r2_clean = float('nan')
        last_tau_r2_clean = float('nan')
        last_n_out_vrest = 0
        last_n_total_vrest = 0
        last_n_out_tau = 0
        last_n_total_tau = 0
        last_hidden_r2 = None
        last_anchor_r2 = None
        field_R2 = None
        field_slope = None
        pbar = trange(Niter, ncols=150)
        # Dale's law enforcement: 3 evenly spaced interventions per epoch
        dale_enabled = getattr(tc, 'dale_law', False)
        if dale_enabled:
            dale_checkpoints = {int(Niter * f) for f in (0.25, 0.5, 0.75)}
            dale_checkpoints.discard(0)
        # === LLM-MODIFIABLE: TRAINING LOOP START ===
        # Main training loop. Suggested changes: loss function, gradient clipping,
        # data sampling strategy, LR scheduler steps, early stopping.
        # Do NOT change: function signature, model construction, data loading, return values.
        _prof_wait, _prof_warmup, _prof_active = 3, 2, 3
        if _profiling:
            _prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=_prof_wait, warmup=_prof_warmup, active=_prof_active, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(_profiler_trace_dir, use_gzip=True),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )
            _prof.start()
        # NGP-injection schedule (two-phase, binary on/off).
        #
        # Phase 1 (N < warmup_inject_nnr_iter): hidden voltages are zero-silenced
        # (NGP forward is not called for the GNN input — see assignment site
        # below). NGP still trains via the anchor loss (which is gated by
        # coeff_anchor_voltage, NOT by injection state), so by phase 2 the NGP
        # already produces sensible per-(t,u,v) outputs at anchor positions
        # via shared backbone weights.
        #
        # Phase 2 (N >= warmup_inject_nnr_iter): NGP injection is hard-on (no
        # scalar α). The smooth absorption of the new input distribution is
        # done by the LR-damping V (see below), not by ramping the injected
        # signal magnitude.
        #
        # Defaults (warmup=0, ramp=0) preserve the legacy "always inject"
        # behavior — non-NGP runs and uncondiguered NGP runs follow the
        # exact same code path they used before this rewrite.
        _warmup_inject_iter_frac = float(getattr(tc, 'warmup_inject_nnr_iter_frac', 0.0))
        _warmup_inject_ramp_frac = float(getattr(tc, 'warmup_inject_nnr_ramp_iter_frac', 0.0))
        if _warmup_inject_iter_frac > 0.0:
            _warmup_inject_iter = int(Niter * _warmup_inject_iter_frac)
        else:
            _warmup_inject_iter = int(getattr(tc, 'warmup_inject_nnr_iter', 0))
        if _warmup_inject_ramp_frac > 0.0:
            _warmup_inject_ramp = int(Niter * _warmup_inject_ramp_frac)
        else:
            _warmup_inject_ramp = int(getattr(tc, 'warmup_inject_nnr_ramp_iter', 0))

        # LR-damping V-schedule. Fired only when an NGP injection switch is
        # configured (_warmup_inject_iter > 0) AND a damping window length is
        # set (_warmup_inject_ramp > 0). lr_damping_factor controls the depth
        # of the V (default 100.0 → /100 at the trough; one knob covers both
        # legs, recovery just multiplies back by the same factor).
        _lr_damping_factor = float(getattr(tc, 'lr_damping_factor', 100.0))
        _lr_damping_active = (_warmup_inject_iter > 0
                              and _warmup_inject_ramp > 0
                              and _lr_damping_factor > 1.0)
        _damp_groups = ('W', 'f_theta', 'g_phi')   # only GNN groups are damped
        # Cache base LRs so each iter we set pg['lr'] = base * lr_mult
        # without compounding rounding errors over the loop.
        _base_lrs = {id(pg): pg['base_lr'] for pg in optimizer.param_groups}

        # Stage boundaries surfaced as labeled verticals on metrics.png
        # (right panel only). Empty when no NGP injection is configured.
        _ngp_stages = []
        if _warmup_inject_iter > 0:
            _ramp_mid = _warmup_inject_iter + _warmup_inject_ramp
            _ramp_end = _warmup_inject_iter + 2 * _warmup_inject_ramp
            _ngp_stages.append((_warmup_inject_iter, 'inject'))
            if _lr_damping_active:
                _ngp_stages.append((_ramp_mid, 'trough'))
                _ngp_stages.append((_ramp_end, 'recover'))
            print(f'NGP binary-inject schedule: '
                  f'warmup [0, {_warmup_inject_iter}) NGP off + nominal LR, '
                  f'inject ON at {_warmup_inject_iter}.')
            if _lr_damping_active:
                print(f'  LR-damping V on {{{",".join(_damp_groups)}}}: '
                      f'damp [{_warmup_inject_iter}, {_ramp_mid}) base→base/{_lr_damping_factor:g}, '
                      f'recover [{_ramp_mid}, {_ramp_end}) base/{_lr_damping_factor:g}→base. '
                      f'Niter={Niter}.')
            else:
                print(f'  LR-damping V disabled (ramp window or factor not set). Niter={Niter}.')

        _prev_injection_active = None
        _prev_lr_mult = 1.0

        for N in pbar:

            # Binary injection gate. Cheap — branchless via int compare —
            # evaluated once per iteration in Python (not inside any
            # torch.compile region).
            injection_active = (_warmup_inject_iter <= 0) or (N >= _warmup_inject_iter)

            # LR-damping V-schedule multiplier (applied to W/f_theta/g_phi
            # param groups only). Outside the V window: 1.0 → no change vs
            # legacy behavior. embedding/NNR_hidden/NNR_f always at 1.0.
            if _lr_damping_active:
                if N < _warmup_inject_iter or N >= _ramp_end:
                    _lr_mult = 1.0
                elif N < _ramp_mid:
                    # Damping leg: linearly 1.0 → 1/factor.
                    _t = float(N - _warmup_inject_iter) / float(_warmup_inject_ramp)
                    _lr_mult = 1.0 + (1.0 / _lr_damping_factor - 1.0) * _t
                else:
                    # Recovery leg: linearly 1/factor → 1.0.
                    _t = float(N - _ramp_mid) / float(_warmup_inject_ramp)
                    _lr_mult = (1.0 / _lr_damping_factor
                                + (1.0 - 1.0 / _lr_damping_factor) * _t)
            else:
                _lr_mult = 1.0

            # Apply the multiplier to the GNN param groups whenever it
            # changes. Skipping when (_lr_mult == _prev_lr_mult) avoids the
            # per-iter Python loop overhead 99% of training (mult is 1.0
            # outside the V).
            if _lr_damping_active and _lr_mult != _prev_lr_mult:
                for pg in optimizer.param_groups:
                    if pg.get('name') in _damp_groups:
                        pg['lr'] = _base_lrs[id(pg)] * _lr_mult
                _prev_lr_mult = _lr_mult

            # Announce injection-on transition once.
            if (_warmup_inject_iter > 0
                    and _prev_injection_active is False
                    and injection_active):
                if _lr_damping_active:
                    print(f'\n[NGP inject] iter {N}: phase 1 → phase 2 '
                          f'(NGP hard-on; GNN-LR V-schedule starts: '
                          f'damp over {_warmup_inject_ramp} iters, '
                          f'recover over {_warmup_inject_ramp} iters).')
                else:
                    print(f'\n[NGP inject] iter {N}: phase 1 → phase 2 '
                          f'(NGP hard-on, no LR damping configured).')
            elif (_lr_damping_active
                    and N == _ramp_mid
                    and _warmup_inject_iter > 0):
                print(f'\n[NGP inject] iter {N}: GNN-LR damping leg → recovery leg '
                      f'(at trough, base/{_lr_damping_factor:g}).')
            elif (_lr_damping_active
                    and N == _ramp_end
                    and _warmup_inject_iter > 0):
                print(f'\n[NGP inject] iter {N}: GNN-LR back to nominal.')
            _prev_injection_active = injection_active

            # Unfreeze embedding at the midpoint after UMAP clustering froze it
            if embedding_frozen and N == unfreeze_at_iteration:
                embedding_frozen = False
                lr_embedding = tc.lr_embedding
                optimizer, n_total_params = set_trainable_parameters(
                    model=model, lr_embedding=lr_embedding, lr=lr,
                    lr_update=lr_update, lr_W=lr_W)
                _logger.debug(f'unfreezing embedding at iteration {N}/{Niter}')

            optimizer.zero_grad()

            # Recurrent training (standard or multi-start) — delegated to recurrent_step
            if tc.recurrent_training and not tc.neural_ODE_training:
                loss, regul_val = recurrent_loss(
                    model=model, x_ts=x_ts, y_ts=y_ts, edges=edges, ids=visible_ids,
                    frame_indices=frame_indices, iter_idx=N, config=config,
                    device=device, xnorm=xnorm, ynorm=ynorm,
                    regularizer=regularizer, has_visual_field=has_visual_field,
                    hidden_ids=hidden_ids,
                )
                loss.backward()
                if hasattr(tc, 'grad_clip_W') and tc.grad_clip_W > 0 and hasattr(model, 'W'):
                    if model.W.grad is not None:
                        torch.nn.utils.clip_grad_norm_([model.W], max_norm=tc.grad_clip_W)
                optimizer.step()
                if dale_enabled and N in dale_checkpoints:
                    enforce_dale_law(model, edges)
                lr_scheduler.step()
                _total_loss_gpu = _total_loss_gpu + loss.detach()
                total_loss_regul += regul_val
                regularizer.finalize_iteration()

                if regularizer.should_record():
                    _current_loss = loss.item()  # single sync per plot_frequency iters
                    loss_components['loss'].append((_current_loss - regul_val) / n_neurons)
                    plot_dict = {**regularizer.get_history(), 'loss': loss_components['loss']}
                    plot_signal_loss(plot_dict, log_dir, epoch=epoch, Niter=Niter,
                                    epoch_boundaries=regularizer.epoch_boundaries)

                # R2 checkpoint
                is_regular_r2 = (N % connectivity_plot_frequency == 0)
                is_early_r2 = (N < connectivity_plot_frequency) and (N % early_r2_frequency == 0)
                model_name = model_config.signal_model_name

                # Intermediate model checkpoint at every 1/10 of the epoch.
                # Overwrites the same path used by the end-of-epoch save (line
                # ~1019), so graph_tester (which picks the newest file in
                # models/ by mtime) keeps working unchanged. Lets a recovery
                # from a mid-training wedge cost at most one Niter/10 cycle.
                if is_regular_r2 and N > 0:
                    _intermediate_path = os.path.join(
                        log_dir, 'models',
                        f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt')
                    os.makedirs(os.path.dirname(_intermediate_path), exist_ok=True)
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               _intermediate_path)
                if (is_regular_r2 or is_early_r2) and ('linear' in model_name or 'known_ode' in model_name):
                    last_connectivity_r2, last_tau_r2, last_vrest_r2, _dyn = plot_training_linear(
                        model, config, epoch, N, log_dir, device, gt_weights, n_neurons=n_neurons)
                    last_vrest_r2_clean = _dyn['vrest_r2_clean']
                    last_tau_r2_clean   = _dyn['tau_r2_clean']
                    last_n_out_vrest    = _dyn['n_out_vrest']
                    last_n_total_vrest  = _dyn['n_total_vrest']
                    last_n_out_tau      = _dyn['n_out_tau']
                    last_n_total_tau    = _dyn['n_total_tau']
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f},{_fmt_metric(last_hidden_r2)},{_fmt_metric(last_anchor_r2)},'
                                f'{_fmt_metric(last_vrest_r2_clean)},{last_n_out_vrest},{last_n_total_vrest},'
                                f'{_fmt_metric(last_tau_r2_clean)},{last_n_out_tau},{last_n_total_tau}\n')
                    _metrics_changed = True
                elif (is_regular_r2 or is_early_r2) and 'mlp' not in model_name.lower():
                    last_connectivity_r2, _r2_visible, _h_r2, _a_r2 = plot_training_flyvis(x_ts, model, config, epoch, N, log_dir, device, type_list, gt_weights, edges, n_neurons=n_neurons, n_neuron_types=sim.n_neuron_types, ode_params=ode_params, hidden_ids=hidden_ids, anchor_ids=anchor_ids)
                    last_connectivity_r2_visible = _r2_visible
                    if _h_r2 is not None:
                        last_hidden_r2 = _h_r2
                    if _a_r2 is not None:
                        last_anchor_r2 = _a_r2
                    _dyn = compute_dynamics_r2(model, x_ts, config, device, n_neurons)
                    last_vrest_r2       = _dyn['vrest_r2']
                    last_tau_r2         = _dyn['tau_r2']
                    last_vrest_r2_clean = _dyn['vrest_r2_clean']
                    last_tau_r2_clean   = _dyn['tau_r2_clean']
                    last_n_out_vrest    = _dyn['n_out_vrest']
                    last_n_total_vrest  = _dyn['n_total_vrest']
                    last_n_out_tau      = _dyn['n_out_tau']
                    last_n_total_tau    = _dyn['n_total_tau']
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f},{_fmt_metric(last_hidden_r2)},{_fmt_metric(last_anchor_r2)},'
                                f'{_fmt_metric(last_vrest_r2_clean)},{last_n_out_vrest},{last_n_total_vrest},'
                                f'{_fmt_metric(last_tau_r2_clean)},{last_n_out_tau},{last_n_total_tau}\n')
                    _metrics_changed = True
                else:
                    _metrics_changed = False

                # Fast NGP Pearson refresh — independent of the heavy R²
                # checkpoint above. Subsamples (64 neurons × 256 frames),
                # one batched forward, no grad. Only fires when the model
                # has a hidden-neuron INR. Also appends a metrics.log row
                # so the runner's 300s collector picks the updated values
                # up between heavy checkpoints.
                _ngp_quick_updated = False
                _h_quick_std = None
                _a_quick_std = None
                if (has_hidden_neurons
                        and getattr(model, 'NNR_hidden', None) is not None
                        and N > 0 and N % _NGP_QUICK_FREQ == 0):
                    _h_quick, _h_quick_std = _quick_ngp_pearson(
                        model, x_ts, hidden_ids,
                        use_anchor=False, device=device, return_stats=True)
                    if _h_quick is not None:
                        last_hidden_r2 = _h_quick
                        _ngp_quick_updated = True
                    if has_anchor_neurons:
                        _a_quick, _a_quick_std = _quick_ngp_pearson(
                            model, x_ts, anchor_ids,
                            use_anchor=True, device=device, return_stats=True)
                        if _a_quick is not None:
                            last_anchor_r2 = _a_quick
                            _ngp_quick_updated = True
                if _ngp_quick_updated:
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},'
                                f'{_fmt_metric(last_connectivity_r2)},'
                                f'{_fmt_metric(last_vrest_r2)},'
                                f'{_fmt_metric(last_tau_r2)},'
                                f'{_fmt_metric(last_hidden_r2)},'
                                f'{_fmt_metric(last_anchor_r2)},'
                                f'{_fmt_metric(last_vrest_r2_clean)},'
                                f'{last_n_out_vrest},{last_n_total_vrest},'
                                f'{_fmt_metric(last_tau_r2_clean)},'
                                f'{last_n_out_tau},{last_n_total_tau}\n')
                    with open(nnr_pearson_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},'
                                f'{_fmt_metric(last_hidden_r2)},'
                                f'{_fmt_metric(_h_quick_std)},'
                                f'{_fmt_metric(last_anchor_r2)},'
                                f'{_fmt_metric(_a_quick_std)}\n')
                    _metrics_changed = True

                # Refresh metrics.png whenever metrics.log / nnr_pearson.log
                # gained a new row (heavy R² or quick NGP path).
                if _metrics_changed:
                    plot_metrics(log_dir,
                                 epoch_boundaries=regularizer.epoch_boundaries,
                                 ngp_stages=_ngp_stages)

                if last_connectivity_r2 is not None or last_hidden_r2 is not None:
                    bar_parts = []
                    if last_connectivity_r2 is not None:
                        c_conn = r2_color(last_connectivity_r2)
                        if last_connectivity_r2_visible is not None and abs(last_connectivity_r2_visible - last_connectivity_r2) > 1e-4:
                            conn_str = f'conn={last_connectivity_r2:.3f}({last_connectivity_r2_visible:.3f})'
                        else:
                            conn_str = f'conn={last_connectivity_r2:.3f}'
                        bar_parts.append(f'{c_conn}{conn_str}{ANSI_RESET}')
                        if ode_params.has_vrest():
                            _vr_pct = (100.0 * last_n_out_vrest / last_n_total_vrest) if last_n_total_vrest > 0 else 0.0
                            bar_parts.append(f'{r2_color(last_vrest_r2_clean)}Vr={last_vrest_r2_clean:.3f}({_vr_pct:.0f}%){ANSI_RESET}')
                        if ode_params.has_tau():
                            _tau_pct = (100.0 * last_n_out_tau / last_n_total_tau) if last_n_total_tau > 0 else 0.0
                            bar_parts.append(f'{r2_color(last_tau_r2_clean)}τ={last_tau_r2_clean:.3f}({_tau_pct:.0f}%){ANSI_RESET}')
                    if last_hidden_r2 is not None or last_anchor_r2 is not None:
                        # During warmup (injection_active=False), hidden
                        # voltages are zero-silenced so hidden_nnr_pearson
                        # against GT-injection rollout is meaningless; show
                        # n/a. Anchor is trained throughout, show it.
                        if not injection_active:
                            if last_anchor_r2 is not None:
                                nnr_str = f'nnr=n/a({last_anchor_r2:.3f})'
                            else:
                                nnr_str = 'nnr=n/a'
                            bar_parts.append(nnr_str)
                        elif last_hidden_r2 is not None:
                            if last_anchor_r2 is not None:
                                nnr_str = f'nnr={last_hidden_r2:.3f}({last_anchor_r2:.3f})'
                            else:
                                nnr_str = f'nnr={last_hidden_r2:.3f}'
                            bar_parts.append(f'{r2_color(last_hidden_r2)}{nnr_str}{ANSI_RESET}')
                    if bar_parts:
                        pbar.set_postfix_str(' '.join(bar_parts))
                continue

            state_batch = []
            y_list = []
            ids_list = []
            k_list = []
            visual_input_list = []
            ids_index = 0

            loss = torch.zeros((), device=device)
            regularizer.reset_iteration(device=device)

            # Consecutive batch: pick one random start, use batch_size consecutive frames
            if tc.consecutive_batch:
                k_start = int(frame_indices[N * tc.batch_size])

            for batch in range(tc.batch_size):

                if tc.consecutive_batch:
                    k = k_start + batch
                else:
                    k = int(frame_indices[N * tc.batch_size + batch])

                x = x_ts.frame(k)

                # Add measurement noise to observed voltage
                if x.noise is not None and sim.measurement_noise_level > 0:
                    x.voltage = x.voltage + x.noise

                # Hidden neurons: predict via SIREN/NGP or zero-silence.
                # injection_active is binary: phase 1 → False (v_h=0, identical
                # to the no-NGP baseline; NGP still trains via the anchor loss
                # elsewhere in the step), phase 2 → True (NGP fully injected).
                # The smooth absorption of the new input distribution at the
                # phase 1→2 transition is handled by the LR-damping V-schedule
                # on the GNN param groups, not by ramping injection magnitude.
                if has_hidden_neurons:
                    if model.NNR_hidden is not None and injection_active:
                        x.voltage[hidden_ids] = model.forward_hidden(x, k, hidden_ids)
                    else:
                        x.voltage[hidden_ids] = 0.0
                        # Phase 1: forward_hidden is skipped, so the spatial NGP
                        # position cache (normally primed there) stays empty —
                        # but the anchor-voltage loss below still routes through
                        # _ngp_query_spatial. Prime it here; it is idempotent.
                        if model.NNR_hidden is not None and getattr(model, '_ngp_spatial_enabled', False):
                            model._ngp_cache_pos(x)

                if tc.time_window > 0:
                    x_temporal = x_ts.voltage[k - tc.time_window + 1: k + 1].T
                    # x stays as NeuronState; x_temporal passed separately to temporal model

                if has_visual_field:
                    visual_input = model.forward_visual(x, k)
                    x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                    x.stimulus[model.n_input_neurons:] = 0

                if batch==0:  # apply regularization only once
                    regul_loss = regularizer.compute(
                        model=model,
                        x=x,
                        in_features=None,
                        ids=ids,
                        ids_batch=None,
                        edges=edges,
                        device=device,
                        xnorm=xnorm
                    )
                    loss = loss + regul_loss

                if tc.recurrent_training or tc.neural_ODE_training:
                    y = x_ts.voltage[k + 1 if _stride_subsample else k + tc.time_step].unsqueeze(-1)
                elif test_neural_field:
                    y = x_ts.stimulus[k, :sim.n_input_neurons].unsqueeze(-1)
                else:
                    y = y_ts_gpu[k] / ynorm

                if loss_noise_level>0:
                    y = y + torch.randn(y.shape, device=device) * loss_noise_level

                state_batch.append(x)
                n = x.n_neurons
                y_list.append(y)
                ids_list.append(visible_ids + ids_index)
                k_list.append(torch.ones((n, 1), dtype=torch.int, device=device) * k)
                if test_neural_field:
                    visual_input_list.append(visual_input)
                ids_index += n


            data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)
            y_batch = torch.cat(y_list, dim=0)
            ids_batch = torch.cat(ids_list, dim=0)
            k_batch = torch.cat(k_list, dim=0)

            _total_regul_gpu = _total_regul_gpu + loss.detach()  # regul-only at this point

            if test_neural_field:
                visual_input_batch = torch.cat(visual_input_list, dim=0)
                loss = loss + (visual_input_batch - y_batch).norm(2)


            elif 'mlp_ode' in model_config.signal_model_name.lower():
                batched_state, _ = _batch_frames(state_batch, edges)
                batched_x = batched_state.to_packed()
                pred = model(batched_x, data_id=data_id, return_all=False)

                loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

            elif 'mlp' in model_config.signal_model_name.lower():
                batched_state, _ = _batch_frames(state_batch, edges)
                pred = model(batched_state, data_id=data_id, return_all=False)

                loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

            else: # 'GNN' branch

                batched_state, batched_edges = _batch_frames(state_batch, edges)
                pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

                update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
                loss = loss + update_regul


                if tc.neural_ODE_training:

                    ode_state_clamp = getattr(tc, 'ode_state_clamp', 10.0)
                    ode_stab_lambda = getattr(tc, 'ode_stab_lambda', 0.0)
                    ode_loss, pred_x = neural_ode_loss(
                        model=model,
                        dataset_batch=state_batch,
                        edge_index=edges,
                        x_ts=x_ts,
                        k_batch=k_batch,
                        time_step=tc.time_step,
                        batch_size=tc.batch_size,
                        n_neurons=n_neurons,
                        ids_batch=ids_batch,
                        delta_t=sim.delta_t,
                        device=device,
                        data_id=data_id,
                        has_visual_field=has_visual_field,
                        y_batch=y_batch,
                        noise_level=tc.noise_recurrent_level,
                        ode_method=tc.ode_method,
                        rtol=tc.ode_rtol,
                        atol=tc.ode_atol,
                        adjoint=tc.ode_adjoint,
                        iteration=N,
                        state_clamp=ode_state_clamp,
                        stab_lambda=ode_stab_lambda
                    )
                    loss = loss + ode_loss


                elif tc.recurrent_training:

                    pred_x = batched_state.voltage.unsqueeze(-1) + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

                    if tc.time_step > 1:
                        for step in range(tc.time_step - 1):
                            neurons_per_sample = state_batch[0].n_neurons

                            for b in range(tc.batch_size):
                                start_idx = b * neurons_per_sample
                                end_idx = (b + 1) * neurons_per_sample

                                state_batch[b].voltage = pred_x[start_idx:end_idx].squeeze()
                                if has_hidden_neurons:
                                    state_batch[b].voltage[hidden_ids] = 0.0

                                k_current = k_batch[start_idx, 0].item() + step + 1

                                if has_visual_field:
                                    visual_input_next = model.forward_visual(state_batch[b], k_current)
                                    state_batch[b].stimulus[:model.n_input_neurons] = visual_input_next.squeeze(-1)
                                    state_batch[b].stimulus[model.n_input_neurons:] = 0
                                else:
                                    x_next = x_ts.frame(k_current)
                                    state_batch[b].stimulus = x_next.stimulus
                                    if x_next.optogenetics_stimulus is not None:
                                        state_batch[b].optogenetics_stimulus = x_next.optogenetics_stimulus

                            batched_state, batched_edges = _batch_frames(state_batch, edges)
                            pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

                            pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

                    loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (sim.delta_t * tc.time_step)).norm(2)

                else:

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)
                    # NB: the previous "hidden self-consistency" loss
                    # ‖pred_h − target_h‖ where target_h = NGP's own prediction
                    # has been removed. In phase 1 (v_h=0) pred_h is
                    # delta_t · O(small) ≈ 0, so the gradient on NGP reduced to
                    # an L2 ridge ‖NGP‖ that pinned NGP-hidden at the trivial
                    # zero fixed point throughout training (verified
                    # empirically: hidden_pearson stayed ≈ 0.02 across 320k
                    # iters). NGP-hidden is now supervised only via (a) the
                    # anchor loss (shared NNR backbone) and (b) backprop
                    # through injection during phase 2 — both of which point
                    # away from zero.
                    # Anchor voltage loss: NGP-T/SIREN-T anchor outputs vs observed GT voltages
                    if has_anchor_neurons and getattr(tc, 'coeff_anchor_voltage', 0.0) > 0:
                        n_per = state_batch[0].n_neurons
                        k_starts = k_batch[::n_per, 0].to(torch.long)                 # (B,)
                        pred_a = model.forward_anchor_batched(k_starts, anchor_ids=anchor_ids)  # (B, n_anchor)
                        gt_a = x_ts.voltage[k_starts[:, None], anchor_ids[None, :]]    # (B, n_anchor)
                        loss = loss + tc.coeff_anchor_voltage * (pred_a - gt_a).norm(2)


                # === LLM-MODIFIABLE: BACKWARD AND STEP START ===
                # Allowed changes: gradient clipping, LR scheduler step, loss scaling
                loss.backward()

                # debug gradient check for neural ODE training
                if tc.neural_ODE_training and (N % 500 == 0):
                    debug_check_gradients(model, loss, N)

                # W-specific gradient clipping: clip W gradients to force optimizer
                # to adjust lin_update (which contains V_rest/tau) instead of W
                if hasattr(tc, 'grad_clip_W') and tc.grad_clip_W > 0 and hasattr(model, 'W'):
                    if model.W.grad is not None:
                        torch.nn.utils.clip_grad_norm_([model.W], max_norm=tc.grad_clip_W)

                optimizer.step()
                if dale_enabled and N in dale_checkpoints:
                    enforce_dale_law(model, edges)
                lr_scheduler.step()
                # === LLM-MODIFIABLE: BACKWARD AND STEP END ===

                _total_loss_gpu = _total_loss_gpu + loss.detach()
                _total_regul_gpu = _total_regul_gpu + regularizer.get_iteration_total_tensor().detach()

                # finalize iteration to record history
                regularizer.finalize_iteration()


                if regularizer.should_record():
                    # get history from regularizer and add loss component
                    current_loss = loss.item()  # single sync per plot_frequency iters
                    regul_total_this_iter = regularizer.get_iteration_total()  # free after sync
                    loss_components['loss'].append((current_loss - regul_total_this_iter) / n_neurons)

                    # merge loss_components with regularizer history for plotting
                    plot_dict = {**regularizer.get_history(), 'loss': loss_components['loss']}

                    # pass per-neuron normalized values to debug (to match dictionary values)
                    plot_signal_loss(plot_dict, log_dir, epoch=epoch, Niter=Niter,
                                   epoch_boundaries=regularizer.epoch_boundaries, debug=False,
                                   current_loss=current_loss / n_neurons, current_regul=regul_total_this_iter / n_neurons,
                                   total_loss=total_loss, total_loss_regul=total_loss_regul)

                    # persist full loss decomposition so the plot can be regenerated
                    torch.save({
                        **plot_dict,
                        'epoch_boundaries': list(regularizer.epoch_boundaries),
                    }, os.path.join(log_dir, 'loss_components.pt'))

                    if tc.save_all_checkpoints:
                        torch.save(
                            {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                            os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}_{N}.pt'))

                # R2 checkpoint: regular interval + early-phase extra points
                is_regular_r2 = (N > 0) and (N % connectivity_plot_frequency == 0)
                is_early_r2 = (N < connectivity_plot_frequency) and (N % early_r2_frequency == 0)
                model_name = model_config.signal_model_name

                # Intermediate model checkpoint at every 1/10 of the epoch.
                # Overwrites the same path used by the end-of-epoch save (line
                # ~1019), so graph_tester (which picks the newest file in
                # models/ by mtime) keeps working unchanged. Lets a recovery
                # from a mid-training wedge cost at most one Niter/10 cycle.
                if is_regular_r2:
                    _intermediate_path = os.path.join(
                        log_dir, 'models',
                        f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt')
                    os.makedirs(os.path.dirname(_intermediate_path), exist_ok=True)
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               _intermediate_path)

                if (is_regular_r2 or is_early_r2) and not test_neural_field and '_mlp' in model_name:
                    from connectome_gnn.metrics import compute_jacobian_connectivity_r2
                    last_connectivity_r2 = compute_jacobian_connectivity_r2(
                        model, x_ts, ode_params, n_neurons=n_neurons, device=device)
                    last_tau_r2 = 0.0
                    last_vrest_r2 = 0.0
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f},{_fmt_metric(last_hidden_r2)},{_fmt_metric(last_anchor_r2)},'
                                f'nan,0,0,nan,0,0\n')
                    # W scatter plot using Jacobian
                    plot_jacobian_w_scatter(model, x_ts, ode_params, gt_weights, n_neurons,
                                            log_dir, epoch, N, device)
                    _metrics_changed = True
                elif (is_regular_r2 or is_early_r2) and not test_neural_field and ('linear' in model_name or 'known_ode' in model_name):
                    last_connectivity_r2, last_tau_r2, last_vrest_r2, _dyn = plot_training_linear(
                        model, config, epoch, N, log_dir, device, gt_weights, n_neurons=n_neurons)
                    last_vrest_r2_clean = _dyn['vrest_r2_clean']
                    last_tau_r2_clean   = _dyn['tau_r2_clean']
                    last_n_out_vrest    = _dyn['n_out_vrest']
                    last_n_total_vrest  = _dyn['n_total_vrest']
                    last_n_out_tau      = _dyn['n_out_tau']
                    last_n_total_tau    = _dyn['n_total_tau']
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f},{_fmt_metric(last_hidden_r2)},{_fmt_metric(last_anchor_r2)},'
                                f'{_fmt_metric(last_vrest_r2_clean)},{last_n_out_vrest},{last_n_total_vrest},'
                                f'{_fmt_metric(last_tau_r2_clean)},{last_n_out_tau},{last_n_total_tau}\n')
                    _metrics_changed = True
                elif (is_regular_r2 or is_early_r2) and not test_neural_field and 'mlp' not in model_name.lower():
                    last_connectivity_r2, _r2_visible, _h_r2, _a_r2 = plot_training_flyvis(x_ts, model, config, epoch, N, log_dir, device, type_list, gt_weights, edges, n_neurons=n_neurons, n_neuron_types=sim.n_neuron_types, ode_params=ode_params, hidden_ids=hidden_ids, anchor_ids=anchor_ids)
                    last_connectivity_r2_visible = _r2_visible
                    if _h_r2 is not None:
                        last_hidden_r2 = _h_r2
                    if _a_r2 is not None:
                        last_anchor_r2 = _a_r2
                    _dyn = compute_dynamics_r2(model, x_ts, config, device, n_neurons)
                    last_vrest_r2       = _dyn['vrest_r2']
                    last_tau_r2         = _dyn['tau_r2']
                    last_vrest_r2_clean = _dyn['vrest_r2_clean']
                    last_tau_r2_clean   = _dyn['tau_r2_clean']
                    last_n_out_vrest    = _dyn['n_out_vrest']
                    last_n_total_vrest  = _dyn['n_total_vrest']
                    last_n_out_tau      = _dyn['n_out_tau']
                    last_n_total_tau    = _dyn['n_total_tau']
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},{last_connectivity_r2:.6f},{last_vrest_r2:.6f},{last_tau_r2:.6f},{_fmt_metric(last_hidden_r2)},{_fmt_metric(last_anchor_r2)},'
                                f'{_fmt_metric(last_vrest_r2_clean)},{last_n_out_vrest},{last_n_total_vrest},'
                                f'{_fmt_metric(last_tau_r2_clean)},{last_n_out_tau},{last_n_total_tau}\n')
                    _metrics_changed = True
                else:
                    _metrics_changed = False

                # Fast NGP Pearson refresh — independent of the heavy R²
                # checkpoint above. Subsamples (64 neurons × 256 frames),
                # one batched forward, no grad. Also appends a metrics.log
                # row so the runner's 300s collector picks the updated
                # values up between heavy checkpoints.
                _ngp_quick_updated = False
                _h_quick_std = None
                _a_quick_std = None
                if (has_hidden_neurons
                        and getattr(model, 'NNR_hidden', None) is not None
                        and N > 0 and N % _NGP_QUICK_FREQ == 0):
                    _h_quick, _h_quick_std = _quick_ngp_pearson(
                        model, x_ts, hidden_ids,
                        use_anchor=False, device=device, return_stats=True)
                    if _h_quick is not None:
                        last_hidden_r2 = _h_quick
                        _ngp_quick_updated = True
                    if has_anchor_neurons:
                        _a_quick, _a_quick_std = _quick_ngp_pearson(
                            model, x_ts, anchor_ids,
                            use_anchor=True, device=device, return_stats=True)
                        if _a_quick is not None:
                            last_anchor_r2 = _a_quick
                            _ngp_quick_updated = True
                if _ngp_quick_updated:
                    with open(metrics_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},'
                                f'{_fmt_metric(last_connectivity_r2)},'
                                f'{_fmt_metric(last_vrest_r2)},'
                                f'{_fmt_metric(last_tau_r2)},'
                                f'{_fmt_metric(last_hidden_r2)},'
                                f'{_fmt_metric(last_anchor_r2)},'
                                f'{_fmt_metric(last_vrest_r2_clean)},'
                                f'{last_n_out_vrest},{last_n_total_vrest},'
                                f'{_fmt_metric(last_tau_r2_clean)},'
                                f'{last_n_out_tau},{last_n_total_tau}\n')
                    with open(nnr_pearson_log_path, 'a') as f:
                        f.write(f'{regularizer.iter_count},'
                                f'{_fmt_metric(last_hidden_r2)},'
                                f'{_fmt_metric(_h_quick_std)},'
                                f'{_fmt_metric(last_anchor_r2)},'
                                f'{_fmt_metric(_a_quick_std)}\n')
                    _metrics_changed = True

                # Refresh metrics.png whenever metrics.log / nnr_pearson.log
                # gained a new row (heavy R² or quick NGP path).
                if _metrics_changed:
                    plot_metrics(log_dir,
                                 epoch_boundaries=regularizer.epoch_boundaries,
                                 ngp_stages=_ngp_stages)

                if last_connectivity_r2 is not None or last_hidden_r2 is not None:
                    bar_parts = []
                    if last_connectivity_r2 is not None:
                        c_conn = r2_color(last_connectivity_r2)
                        if last_connectivity_r2_visible is not None and abs(last_connectivity_r2_visible - last_connectivity_r2) > 1e-4:
                            conn_str = f'conn={last_connectivity_r2:.3f}({last_connectivity_r2_visible:.3f})'
                        else:
                            conn_str = f'conn={last_connectivity_r2:.3f}'
                        bar_parts.append(f'{c_conn}{conn_str}{ANSI_RESET}')
                        if ode_params.has_vrest():
                            _vr_pct = (100.0 * last_n_out_vrest / last_n_total_vrest) if last_n_total_vrest > 0 else 0.0
                            bar_parts.append(f'{r2_color(last_vrest_r2_clean)}Vr={last_vrest_r2_clean:.3f}({_vr_pct:.0f}%){ANSI_RESET}')
                        if ode_params.has_tau():
                            _tau_pct = (100.0 * last_n_out_tau / last_n_total_tau) if last_n_total_tau > 0 else 0.0
                            bar_parts.append(f'{r2_color(last_tau_r2_clean)}τ={last_tau_r2_clean:.3f}({_tau_pct:.0f}%){ANSI_RESET}')
                    if last_hidden_r2 is not None or last_anchor_r2 is not None:
                        # During warmup (injection_active=False), hidden
                        # voltages are zero-silenced so hidden_nnr_pearson
                        # against GT-injection rollout is meaningless; show
                        # n/a. Anchor is trained throughout, show it.
                        if not injection_active:
                            if last_anchor_r2 is not None:
                                nnr_str = f'nnr=n/a({last_anchor_r2:.3f})'
                            else:
                                nnr_str = 'nnr=n/a'
                            bar_parts.append(nnr_str)
                        elif last_hidden_r2 is not None:
                            if last_anchor_r2 is not None:
                                nnr_str = f'nnr={last_hidden_r2:.3f}({last_anchor_r2:.3f})'
                            else:
                                nnr_str = f'nnr={last_hidden_r2:.3f}'
                            bar_parts.append(f'{r2_color(last_hidden_r2)}{nnr_str}{ANSI_RESET}')
                    if bar_parts:
                        pbar.set_postfix_str(' '.join(bar_parts))

                if (has_visual_field) & (N in plot_iterations):
                    field_R2, field_slope = render_visual_field_video(
                        model, x_ts, sim, log_dir, epoch, N, logger)


                    if last_connectivity_r2 is not None:
                        pbar.set_postfix_str(f'{r2_color(last_connectivity_r2)}R²={last_connectivity_r2:.3f}{ANSI_RESET}')
                    if tc.save_all_checkpoints:
                        torch.save(
                            {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                            os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}_{N}.pt'))

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)
            if _profiling:
                _prof.step()

        if _profiling:
            _prof.stop()
            print(f'[Profiler] Trace saved to {_profiler_trace_dir}/')
            print(f'  View with: tensorboard --logdir {_profiler_trace_dir}')

        # === LLM-MODIFIABLE: TRAINING LOOP END ===

        # Calculate epoch-level losses — two syncs per epoch, not per iteration
        total_loss = _total_loss_gpu.item()
        total_loss_regul = _total_regul_gpu.item()
        epoch_total_loss = total_loss / n_neurons
        epoch_regul_loss = total_loss_regul / n_neurons
        epoch_pred_loss = (total_loss - total_loss_regul) / n_neurons

        _logger.info("epoch {}. loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        logger.info("Epoch {}. Loss: {:.6f} (pred: {:.6f}, regul: {:.6f})".format(
            epoch, epoch_total_loss, epoch_pred_loss, epoch_regul_loss))
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'))

        if has_visual_field and hasattr(model, 'NNR_f'):
            torch.save(model.NNR_f.state_dict(),
                       os.path.join(log_dir, 'models', f'inr_stimulus_{epoch}.pt'))

        list_loss.append(epoch_pred_loss)
        list_loss_regul.append(epoch_regul_loss)

        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(3 * default_style.figure_height * default_style.default_aspect,
                                    2 * default_style.figure_height))

        # Plot 1: Loss
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(list_loss, color=default_style.foreground, linewidth=default_style.line_width)
        ax1.set_xlim([0, tc.n_epochs])
        default_style.ylabel(ax1, 'loss')
        default_style.xlabel(ax1, 'epochs')

        plot_training_summary_panels(fig, log_dir, Niter=Niter)

        if replace_with_cluster:

            if (epoch % tc.sparsity_freq == tc.sparsity_freq - 1) & (epoch < tc.n_epochs - tc.sparsity_freq):
                _logger.info('replace embedding with clusters ...')
                eps = tc.cluster_distance_threshold
                results = clustering_evaluation(to_numpy(model.a), type_list, eps=eps)
                _logger.info(f"eps={eps}: {results['n_clusters_found']} clusters, "
                      f"accuracy={results['accuracy']:.3f}")

                labels = results['cluster_labels']

                for n in np.unique(labels):
                    # if n == -1:
                    #     continue
                    indices = np.where(labels == n)[0]
                    if len(indices) > 1:
                        with torch.no_grad():
                            model.a[indices, :] = torch.mean(model.a[indices, :], dim=0, keepdim=True)

                fig.add_subplot(2, 3, 6)
                type_cmap = CustomColorMap(config=config)
                for n in range(sim.n_neuron_types):
                    pos = torch.argwhere(type_list == n)
                    plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=20, color=type_cmap.color(n),
                                edgecolors='none')
                plt.xlabel('embedding 0', fontsize=18)
                plt.ylabel('embedding 1', fontsize=18)
                plt.xticks([])
                plt.yticks([])
                plt.text(0.5, 0.9, f"eps={eps}: {results['n_clusters_found']} clusters, accuracy={results['accuracy']:.3f}")

                if tc.fix_cluster_embedding:
                    lr_embedding = 1.0E-10
                    # the embedding is fixed for 1 epoch

            else:
                lr = tc.lr
                lr_embedding = tc.lr_embedding
                lr_W = tc.lr_W

            logger.info(f'learning rates: lr_W {lr_W}, lr {lr}, lr_update {lr_update}, lr_embedding {lr_embedding}')
            optimizer, n_total_params = set_trainable_parameters(model=model, lr_embedding=lr_embedding, lr=lr, lr_update=lr_update, lr_W=lr_W)

        if umap_cluster_active:
            if (epoch % tc.umap_cluster_freq == tc.umap_cluster_freq - 1) & (epoch < tc.n_epochs - 1):
                _logger.info('UMAP cluster reassign ...')
                umap_results = umap_cluster_reassign(
                    model, config, x_ts, edges, n_neurons, type_list, device, logger=logger,
                    reinit_mlps=tc.umap_cluster_reinit_mlps,
                    relearn_epochs=tc.umap_cluster_relearn_epochs)

                if umap_results is not None:
                    fig.add_subplot(2, 3, 6)
                    type_cmap = CustomColorMap(config=config)
                    a_umap = umap_results['a_umap']
                    for n_type in range(sim.n_neuron_types):
                        pos = torch.argwhere(type_list == n_type)
                        pos_np = to_numpy(pos).flatten()
                        plt.scatter(a_umap[pos_np, 0], a_umap[pos_np, 1], s=20,
                                    color=type_cmap.color(n_type), edgecolors='none')
                    plt.xlabel(r'UMAP$_1$', fontsize=12)
                    plt.ylabel(r'UMAP$_2$', fontsize=12)
                    plt.xticks([])
                    plt.yticks([])
                    plt.title(f"{umap_results['n_clusters']} cl, acc={umap_results['accuracy']:.3f}", fontsize=10)

                if tc.umap_cluster_fix_embedding or tc.umap_cluster_fix_embedding_ratio > 0:
                    lr_embedding = 1.0E-10
                    embedding_frozen = True

                # rebuild optimizer to reset momentum and relearn f_theta/g_phi
                optimizer, n_total_params = set_trainable_parameters(
                    model=model, lr_embedding=lr_embedding, lr=lr,
                    lr_update=lr_update, lr_W=lr_W)

        plt.tight_layout()
        plt.savefig(f"{log_dir}/tmp_training/epoch_{epoch}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # Calculate and log training time
    training_time = time.time() - training_start_time
    training_time_min = training_time / 60.0
    _logger.info(f"training completed in {training_time_min:.1f} minutes")
    logger.info(f"training completed in {training_time_min:.1f} minutes")

    if log_file is not None:
        log_file.write(f"training_time_min: {training_time_min:.1f}\n")
        log_file.write(f"n_epochs: {tc.n_epochs}\n")
        log_file.write(f"data_augmentation_loop: {tc.data_augmentation_loop}\n")
        log_file.write(f"recurrent_training: {tc.recurrent_training}\n")
        log_file.write(f"batch_size: {tc.batch_size}\n")
        log_file.write(f"lr_W: {tc.lr_W}\n")
        log_file.write(f"lr: {tc.lr}\n")
        log_file.write(f"lr_embedding: {tc.lr_embedding}\n")
        log_file.write(f"coeff_g_phi_diff: {tc.coeff_g_phi_diff}\n")
        log_file.write(f"coeff_g_phi_norm: {tc.coeff_g_phi_norm}\n")
        log_file.write(f"coeff_g_phi_weight_L1: {tc.coeff_g_phi_weight_L1}\n")
        log_file.write(f"coeff_f_theta_weight_L1: {tc.coeff_f_theta_weight_L1}\n")
        log_file.write(f"coeff_f_theta_weight_L2: {tc.coeff_f_theta_weight_L2}\n")
        log_file.write(f"coeff_W_L1: {tc.coeff_W_L1}\n")
        log_file.write(f"dale_law: {getattr(tc, 'dale_law', False)}\n")
        dale_score = dale_law_score(model, edges)
        log_file.write(f"dale_law_score: {dale_score:.4f}\n")
        if field_R2 is not None:
            log_file.write(f"field_R2: {field_R2:.4f}\n")
            log_file.write(f"field_slope: {field_slope:.4f}\n")


# data_train_flyvis_alternate removed — use data_train_flyvis instead
def data_train_gnn_RNN(config, erase, best_model, device):
    """RNN training with sequential processing through time"""

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model


    warm_up_length = tc.warm_up_length  # e.g., 10
    sequence_length = tc.sequence_length  # e.g., 32
    total_length = warm_up_length + sequence_length

    seed = config.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    log_dir, logger = create_log_dir(config, erase)

    _logger.info(f"Loading data from {config.dataset}...")
    x_list = []
    y_list = []
    for run in trange(0, tc.n_runs, ncols=100):
        x = np.load(graphs_data_path(config.dataset, f'x_list_{run}.npy'))
        y = np.load(graphs_data_path(config.dataset, f'y_list_{run}.npy'))

        if tc.training_selected_neurons:
            selected_neuron_ids = np.array(tc.selected_neuron_ids).astype(int)
            x = x[:, selected_neuron_ids, :]
            y = y[:, selected_neuron_ids, :]

        x_list.append(x)
        y_list.append(y)

    _logger.info(f'dataset: {len(x_list)} runs, {len(x_list[0])} frames')

    # Normalization
    activity = torch.tensor(x_list[0][:, :, 3:4], device=device)
    activity = activity.squeeze()
    distrib = activity.flatten()
    valid_distrib = distrib[~torch.isnan(distrib)]

    if len(valid_distrib) > 0:
        xnorm = 1.5 * torch.std(valid_distrib)
    else:
        xnorm = torch.tensor(1.0, device=device)

    ynorm = torch.tensor(1.0, device=device)
    torch.save(xnorm, os.path.join(log_dir, 'xnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))

    _logger.info(f'xnorm: {xnorm.item():.3f}')
    _logger.info(f'ynorm: {ynorm.item():.3f}')
    logger.info(f'xnorm: {xnorm.item():.3f}')
    logger.info(f'ynorm: {ynorm.item():.3f}')

    # Create model
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type, config=config, device=device)
    use_lstm = 'lstm' in model_config.signal_model_name.lower()

    # Count parameters
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'total parameters: {n_total_params:,}')
    logger.info(f'Total parameters: {n_total_params:,}')

    # Optimizer
    lr = tc.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    _logger.info(f'learning rate: {lr}')
    logger.info(f'learning rate: {lr}')

    _logger.info("starting RNN training...")
    logger.info("Starting RNN training...")

    list_loss = []

    for epoch in range(tc.n_epochs):

        # Number of sequences per epoch
        n_sequences = (sim.n_frames - total_length) // 10 * tc.data_augmentation_loop
        plot_frequency = int(n_sequences // 10) # Sample ~10% of possible sequences
        if epoch == 0:
            _logger.debug(f'{n_sequences} sequences per epoch, plot every {plot_frequency} sequences')
            logger.info(f'{n_sequences} sequences per epoch, plot every {plot_frequency} sequences')

        total_loss = 0
        model.train()

        for seq_idx in trange(n_sequences, ncols=100, desc=f"Epoch {epoch}"):

            optimizer.zero_grad()

            # Sample random sequence
            run = np.random.randint(tc.n_runs)
            k_start = np.random.randint(0, sim.n_frames - total_length)

            # Initialize hidden state to None (GRU will initialize to zeros)
            h = None
            c = None if use_lstm else None

            # Warm-up phase
            with torch.no_grad():
                for t in range(k_start, k_start + warm_up_length):
                    x = torch.tensor(x_list[run][t], dtype=torch.float32, device=device)
                    if use_lstm:
                        _, h, c = model(x, h=h, c=c, return_all=True)
                    else:
                        _, h = model(x, h=h, return_all=True)

            # Prediction phase (compute loss)
            loss = 0
            for t in range(k_start + warm_up_length, k_start + total_length):
                x = torch.tensor(x_list[run][t], dtype=torch.float32, device=device)
                y_true = torch.tensor(y_list[run][t], dtype=torch.float32, device=device)

                # Forward pass
                if use_lstm:
                    y_pred, h, c = model(x, h=h, c=c, return_all=True)
                else:
                    y_pred, h = model(x, h=h, return_all=True)

                # Accumulate loss
                loss += (y_pred - y_true).norm(2)

                # # Truncated BPTT: detach hidden state
                # h = h.detach()

            # Normalize by sequence length
            loss = loss / sequence_length

            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            if tc.save_all_checkpoints and (seq_idx % plot_frequency == 0) and (seq_idx > 0):
                # Save intermediate model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs-1}_graphs_{epoch}_{seq_idx}.pt'))

        # Epoch statistics
        avg_loss = total_loss / n_sequences
        _logger.info(f"Epoch {epoch}. Loss: {avg_loss:.6f}")
        logger.info(f"Epoch {epoch}. Loss: {avg_loss:.6f}")

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(log_dir, 'models', f'best_model_with_{tc.n_runs-1}_graphs_{epoch}.pt'))

        list_loss.append(avg_loss)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        # Learning rate decay
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            _logger.info(f"Learning rate decreased to {param_group['lr']}")
            logger.info(f"Learning rate decreased to {param_group['lr']}")


# INR training moved to graph_trainer_inr.py — re-export for backwards compatibility
from connectome_gnn.models.graph_trainer_inr import _generate_inr_video, data_train_INR  # noqa: F401


def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, n_rollout_frames=600,
              ratio=1, run=0, test_mode='', sample_embedding=False, particle_of_interest=1, new_params = None, device=[],
              rollout_without_noise: bool = False, log_file=None, test_config=None):

    dataset_name = config.dataset
    _logger.info(f"dataset_name: {dataset_name}")
    _logger.info(f"{config.description}")

    # Task-trainer test dispatch.
    if getattr(config, 'task', None) is not None:
        task_type = str(getattr(config.task, 'task_type', '')).lower()
        if task_type == 'cortex':
            data_test_cortex_task_gnn(
                config, best_model=best_model, device=device, log_file=log_file,
            )
            return
        if task_type == 'path_integration':
            data_test_path_integration_task(
                config, best_model=best_model, device=device, log_file=log_file,
            )
            return

    _connconstr = any(x in config.dataset for x in ('drosophila_cx', 'zebrafish_oculomotor', 'larva'))
    _cortex_voltage = 'cortex' in config.dataset
    if 'fly' in config.dataset or _connconstr or _cortex_voltage:
        # Ablation modes (test_ablation_NN) zero out a fraction of model.W
        # before the full rollout, so the saved rollout_bundle reflects the
        # ablated dynamics. They go through the standard data_test_gnn path
        # (which writes the bundle). Other special modes (modified, inactivity,
        # ...) still need the regeneration / visualization path.
        special_modes = ('modified', 'inactivity', 'special')
        if 'ablation' in test_mode:
            data_test_gnn(
                config,
                best_model=best_model,
                device=device,
                log_file=log_file,
                test_config=test_config,
                test_mode=test_mode,
            )
        elif any(m in test_mode for m in special_modes):
            data_test_gnn_special(
                config,
                visualize,
                style,
                verbose,
                best_model,
                step,
                n_rollout_frames,
                test_mode,
                new_params,
                device,
                rollout_without_noise=rollout_without_noise,
                log_file=log_file,
            )
        else:
            data_test_gnn(
                config,
                best_model=best_model,
                device=device,
                log_file=log_file,
                test_config=test_config,
            )
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset}")



# Test functions moved to graph_tester.py
from connectome_gnn.models.graph_tester import (
    data_test_cortex_task_gnn,
    data_test_gnn,
    data_test_gnn_special,
    data_test_path_integration_task,
)


# ============================================================================
# Path-integration task trainer (TaskRNN)
# ============================================================================

def data_train_task(config, erase, best_model, device, log_file=None):
    """Dispatch to the task-specific trainer based on `config.task.task_type`.

    - `path_integration` → CX trainer (TaskRNN sign_locked mode, Hulse aux
      losses, pi_acc eval, EPG kinograph snapshots).
    - `cortex`           → Yang multitask trainer (TaskRNN free mode,
      masked-MSE loss, direction_acc eval, 8-panel snapshot).
    """
    task_type = str(getattr(config.task, "task_type", "path_integration")).lower()
    if task_type == "cortex":
        return _data_train_cortex_task(config, erase, best_model, device, log_file)
    elif task_type == "path_integration":
        return _data_train_drosophila_cx_task(config, erase, best_model, device, log_file)


def _data_train_drosophila_cx_task(config, erase, best_model, device, log_file=None):
    """Train a TaskRNN on the path-integration task data.

    Mirrors the skeleton of `data_train_gnn`:
    config → log_dir → data load → model (registry) → optimizer → epoch loop
    with regulariser coeffs → snapshot/eval cadence → per-epoch checkpoint.

    The task data is a flat per-trial layout under
    `<dataset>/{train,test}/{stimulus,target,...}.zarr` (produced by
    `_generate_path_integration_task`). Stimulus is (B, T, 3); target is
    (B, T, 2) = (cos θ_hd, sin θ_hd).

    Loss = MSE(y_hat, y):
        tc.coeff_cos_distance · L_cos  (Eq. 10)
        tc.coeff_norm_floor   · L_norm (Eq. 11, kappa=tc.kappa_norm_floor)
        tc.coeff_tv_circular  · L_tv   (circular TV on EPG/PEN rings)
        tc.coeff_W_L1         · |S|.sum()
    """
    import torch.nn.functional as F

    from connectome_gnn.models.drosophila_cx_eval import (
        _rollout_heading_metrics,
        _save_training_snapshot,
        bump_fwhm,
        path_integration_accuracy_from_data,
    )
    from connectome_gnn.zarr_io import load_raw_array

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    random.seed(tc.seed)

    default_style.apply_globally()

    log_dir, logger = create_log_dir(config, erase)
    # Wipe tmp_training so snapshots, metrics, etc. don't mix across runs.
    shutil.rmtree(os.path.join(log_dir, 'tmp_training'), ignore_errors=True)
    kinograph_dir = os.path.join(log_dir, 'tmp_training', 'evolution')
    os.makedirs(kinograph_dir, exist_ok=True)

    # --- load: trials stay on GPU between iterations ---------------
    root = graphs_data_path(config.dataset)
    _logger.info(f'loading task data from {root}/(train|test)/...')
    u_train = torch.from_numpy(load_raw_array(f"{root}/train/stimulus.zarr")).to(device)
    y_train = torch.from_numpy(load_raw_array(f"{root}/train/target.zarr")).to(device)
    u_test = torch.from_numpy(load_raw_array(f"{root}/test/stimulus.zarr")).to(device)
    y_test = torch.from_numpy(load_raw_array(f"{root}/test/target.zarr")).to(device)
    _logger.info(f'task data: train u={tuple(u_train.shape)} y={tuple(y_train.shape)}  '
                 f'test u={tuple(u_test.shape)} y={tuple(y_test.shape)}')
    logger.info(f'train trials: {u_train.shape[0]}  test trials: {u_test.shape[0]}  '
                f'T: {u_train.shape[1]}  in: {u_train.shape[2]}  out: {y_train.shape[2]}')

    # --- model build via registry ----------------------------------------
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type,
                         config=config, device=device)
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'model {model_config.signal_model_name}: {n_total_params:,} trainable params')
    logger.info(f'model: {model_config.signal_model_name}  params: {n_total_params}')

    # --- optimizer + scheduler -------------------------------------------
    # three named param groups (always built; missing field → tc.lr fallback):
    #   - "w_rec": recurrent core. S (DrosophilaCxTaskRNN) or W + a + g_phi + f_theta
    #              (DrosophilaCxTaskGNN). lr starts at tc.lr_W_rec or tc.lr.
    #              lr_W_rec_schedule drives THIS group exclusively
    #              (per-epoch trajectory).
    #   - "w_ED":  encoder/decoder. W_in, W_out, MLP variants, velocity-gate
    #              scalars (v_pena_l/r, v_penb_l/r). lr = tc.lr_W_ED or tc.lr.
    #              Constant — schedule does not touch.
    #   - "other": biases (b, b_out) and anything not in the above. lr = tc.lr.
    #              Constant — schedule does not touch.
    lr_W_rec = getattr(tc, 'lr_W_rec', None)
    lr_W_ED = getattr(tc, 'lr_W_ED', None)

    def _name_to_group(name: str) -> str:
        if name in ("S", "W", "a") or name.startswith(("g_phi.", "f_theta.")):
            return "w_rec"
        if (name in ("W_in", "W_out")
                or name.startswith(("_W_in_mlp.", "_W_out_mlp."))
                or name in ("v_pena_l", "v_pena_r", "v_penb_l", "v_penb_r")):
            return "w_ED"
        return "other"

    grouped: dict[str, list] = {"w_rec": [], "w_ED": [], "other": []}
    for _name, _p in model.named_parameters():
        grouped[_name_to_group(_name)].append(_p)

    optimizer = torch.optim.Adam(
        [
            {"params": grouped["w_rec"],
             "lr": float(lr_W_rec) if lr_W_rec is not None else tc.lr,
             "name": "w_rec"},
            {"params": grouped["w_ED"],
             "lr": float(lr_W_ED) if lr_W_ED is not None else tc.lr,
             "name": "w_ED"},
            {"params": grouped["other"], "lr": tc.lr, "name": "other"},
        ]
    )
    _logger.info(
        f'three-group optimizer: '
        f'w_rec lr={float(lr_W_rec) if lr_W_rec is not None else tc.lr} '
        f'({len(grouped["w_rec"])} params, schedule-driven) | '
        f'w_ED lr={float(lr_W_ED) if lr_W_ED is not None else tc.lr} '
        f'({len(grouped["w_ED"])} params, constant) | '
        f'other lr={tc.lr} ({len(grouped["other"])} params, constant)'
    )
    lr_scheduler = build_lr_scheduler(optimizer, config)
    _logger.info(f'lr={tc.lr}  lr_scheduler={getattr(tc, "lr_scheduler", "none")}')

    # --- regulariser coefficients (cached as Python scalars) -------------
    coeff_cos = float(tc.coeff_cos_distance)
    coeff_norm = float(tc.coeff_norm_floor)
    kappa_norm = float(tc.kappa_norm_floor)
    coeff_tv = float(tc.coeff_tv_circular)
    coeff_l1S = float(tc.coeff_W_L1)
    coeff_f_diff = float(getattr(tc, 'coeff_f_theta_diff', 0.0))
    coeff_g_diff = float(getattr(tc, 'coeff_g_phi_diff', 0.0))
    grad_clip = float(getattr(tc, 'grad_clip_W', 0.0))
    snapshots_per_epoch = int(getattr(tc, 'snapshots_per_epoch', 5))
    snapshot_n_steps = int(getattr(tc, 'snapshot_n_steps', 1500))
    snapshot_omega_deg = float(getattr(tc, 'snapshot_omega_deg', 60.0))
    _coeff_tail_log = float(getattr(tc, 'coeff_tail_loss', 0.0))
    _logger.info(f'losses: cos_distance={coeff_cos}  norm_floor={coeff_norm} (κ={kappa_norm})  '
                 f'tv_circular={coeff_tv}  W_L1={coeff_l1S}  f_theta_diff={coeff_f_diff}  '
                 f'g_phi_diff={coeff_g_diff}  tail_loss={_coeff_tail_log}')

    # --- training loop ---------------------------------------------------
    n_trials, T_full = u_train.shape[0], u_train.shape[1]
    # data_augmentation_loop multiplies iters/epoch by cycling through
    # additional independent shuffles of the trial pool (DAL=1 is a single
    # one-pass shuffle, matching the previous behaviour).
    dal = int(getattr(tc, 'data_augmentation_loop', 1))
    Niter = max(1, (n_trials // tc.batch_size) * dal)
    snap_every = max(1, Niter // max(1, snapshots_per_epoch))
    total_iters = tc.n_epochs * Niter
    best_loss = float('inf')
    global_step = 0
    n_nan_skips = 0       # cumulative count of skipped optimizer steps

    # per-epoch trial-length curriculum. Slice the first T_epoch frames
    # from the on-disk T=T_full trials. Empty schedule = use T_full.
    raw_schedule = list(getattr(tc, 'n_steps_schedule', []) or [])
    if raw_schedule:
        if len(raw_schedule) < tc.n_epochs:
            raw_schedule = raw_schedule + [raw_schedule[-1]] * (tc.n_epochs - len(raw_schedule))
        n_steps_schedule = [min(int(s), T_full) for s in raw_schedule[:tc.n_epochs]]
    else:
        n_steps_schedule = [T_full] * tc.n_epochs
    _logger.info(f'curriculum n_steps schedule (epochs 1..{tc.n_epochs}): {n_steps_schedule}')

    # per-epoch schedules for the three groups. Each is optional; an empty /
    # missing field leaves the corresponding group at its initial lr (constant).
    def _build_lr_schedule(field_name: str):
        raw = list(getattr(tc, field_name, []) or [])
        if not raw:
            return None
        if len(raw) < tc.n_epochs:
            raw = raw + [raw[-1]] * (tc.n_epochs - len(raw))
        sched = [float(x) for x in raw[:tc.n_epochs]]
        _logger.info(f'{field_name} (epochs 1..{tc.n_epochs}): {sched}')
        return sched

    lr_W_rec_schedule = _build_lr_schedule('lr_W_rec_schedule')
    lr_W_ED_schedule = _build_lr_schedule('lr_W_ED_schedule')

    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
    with open(metrics_log_path, 'w') as f:
        f.write('iteration,epoch,loss,mse,cosd,norm,tv,l1S,pi_acc,fwhm_deg,'
                'r_roll,rmse_roll_deg,r_roll_1k\n')

    last_pi_acc = float('nan')
    last_fwhm = float('nan')
    last_rmse_roll = float('nan')   # deg, rollout at T_epoch
    last_pearson_roll = float('nan')  # corr at T_epoch
    last_pearson_roll_1k = float('nan')  # corr at fixed T=1000 (matches plot title)
    model.train()

    # torch.compile (mirrors data_train_gnn line 451). The recurrent forward
    # has a Python `for t in range(T)` loop, so each T_epoch in the curriculum
    # triggers one recompile; iterations within an epoch reuse the cached
    # graph. fullgraph=True + reduce-overhead matches the flyvis trainer.
    #
    # We keep an `eval_model` handle to the un-compiled module: snapshot
    # rollouts use B=1, T=snapshot_n_steps and bump_fwhm uses fixed
    # n_trials=64; mode='reduce-overhead' (CUDA Graphs) doesn't tolerate
    # those varying shapes well and triggers tracer errors. Eval through
    # the un-compiled forward — small batches, negligible perf cost.
    eval_model = model
    if getattr(tc, 'torch_compile', True):
        try:
            model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
            logger.info('torch.compile enabled (mode=reduce-overhead, fullgraph=True); '
                        'eval/snapshot forward stays eager via _orig_mod')
            _logger.info('torch.compile enabled (eval via _orig_mod)')
        except Exception as exc:
            _logger.warning(f'torch.compile failed, falling back to eager: {exc}')
            logger.info(f'torch.compile failed: {exc}')
    else:
        logger.info('torch.compile disabled via config (torch_compile: false)')

    _logger.info(f'start training: {tc.n_epochs} epochs × {Niter} iters/epoch '
                 f'(n_trials={n_trials}, DAL={dal}); '
                 f'metrics+snapshot every {snap_every} iters '
                 f'(~{total_iters // snap_every} snapshots total)')

    # Rolling backup for param-finiteness rollback. Refreshed after every
    # successful step. If optimizer.step() pushes a param to NaN/Inf (Adam can
    # do this even with clip_grad_norm, since the clip bounds L2 not the
    # per-element update), subsequent forwards return NaN and the trainer
    # loops forever in the NaN-loss skip branch. We restore both model and
    # optimizer state because Adam's m/v are typically NaN too.
    last_good_model_state = {
        k: v.detach().clone() for k, v in eval_model.state_dict().items()
    }
    last_good_opt_state = optimizer.state_dict()

    for epoch in range(tc.n_epochs):
        T_epoch = n_steps_schedule[epoch]
        # Per-epoch lr replacement for the named groups. Each schedule is
        # optional and drives only its own group; "other" always stays at lr.
        for _gname, _gsched in (("w_rec", lr_W_rec_schedule),
                                ("w_ED", lr_W_ED_schedule)):
            if _gsched is not None:
                _lr = _gsched[epoch]
                for g in optimizer.param_groups:
                    if g.get("name") == _gname:
                        g['lr'] = _lr
                _logger.info(f'epoch {epoch+1}: {_gname} lr -> {_lr}')
        gen = torch.Generator(device=device).manual_seed(tc.seed + epoch)
        # Stack `dal` independent shuffles so Niter * batch_size indices
        # are always covered. DAL=1 reduces to a single randperm pass
        # (preserves the prior reproducibility contract).
        perm = torch.cat(
            [torch.randperm(n_trials, device=device, generator=gen)
             for _ in range(max(1, dal))],
            dim=0,
        )
        pbar = trange(Niter, ncols=150,
                      desc=f'epoch {epoch+1} (T={T_epoch})', leave=True)
        coeff_tail = float(getattr(tc, 'coeff_tail_loss', 0.0))
        for N in pbar:
            global_step += 1
            idx = perm[N * tc.batch_size:(N + 1) * tc.batch_size]
            if coeff_tail > 0:
                # Soft-curriculum: roll forward to min(2*T_epoch, T_max), then
                # weight the per-frame MSE = 1 for t < T_epoch and
                # `coeff_tail_loss` for t >= T_epoch. Gives a non-zero gradient
                # on the post-horizon segment (so late-time activity doesn't
                # collapse) while keeping the rollout cost bounded at ~2x the
                # truncated baseline.
                T_max = u_train.shape[1]
                T_eff = min(2 * T_epoch, T_max)
                u = u_train[idx, :T_eff]
                y = y_train[idx, :T_eff]
                y_hat, h_buf = model(u)
                w = torch.ones(T_eff, device=u.device)
                if T_epoch < T_eff:
                    w[T_epoch:] = coeff_tail
                sq_err = (y_hat - y).pow(2).mean(dim=-1)        # (B, T_eff)
                mse = ((sq_err * w[None, :]).sum(dim=-1) / w.sum()).mean()
            else:
                # Hard-truncation curriculum (original behaviour).
                u = u_train[idx, :T_epoch]
                y = y_train[idx, :T_epoch]
                y_hat, h_buf = model(u)
                mse = F.mse_loss(y_hat, y)
            cosd = (model.loss_cos_distance(coeff_cos)
                    if coeff_cos > 0 else u.new_zeros(()))
            norm = (model.loss_norm_floor(coeff_norm, kappa_norm)
                    if coeff_norm > 0 else u.new_zeros(()))
            tv = (model.loss_tv_circular(h_buf, coeff_tv)
                  if coeff_tv > 0 else u.new_zeros(()))
            l1S = (coeff_l1S * model.S.abs().sum()
                   if coeff_l1S > 0 else u.new_zeros(()))
            # f_θ-diff: only the GNN model exposes loss_f_theta_diff; the
            # sign-locked RNN has no f_θ and the coefficient is a no-op there.
            f_diff = (model.loss_f_theta_diff(h_buf, coeff_f_diff)
                      if coeff_f_diff > 0 and hasattr(model, 'loss_f_theta_diff')
                      else u.new_zeros(()))
            # g_φ-diff: positive-monotonicity prior on ∂g_φ/∂v. Only GNN
            # exposes loss_g_phi_diff (the sign-locked RNN has no g_φ).
            # Most useful with g_phi_positive=false to preserve Dale's law.
            g_diff = (model.loss_g_phi_diff(h_buf, coeff_g_diff)
                      if coeff_g_diff > 0 and hasattr(model, 'loss_g_phi_diff')
                      else u.new_zeros(()))
            loss = mse + cosd + norm + tv + l1S + f_diff + g_diff

            optimizer.zero_grad(set_to_none=True)
            # NaN guardrail: if the loss itself is non-finite we know the
            # gradients will be too, so skip backward+step entirely.
            if not torch.isfinite(loss):
                n_nan_skips += 1
                lr_scheduler.step()
                if n_nan_skips == 1:
                    # One-shot diagnostic dump on the first NaN to locate
                    # the source: params (corruption?), input (data issue?),
                    # y_hat / h_buf (forward divergence — and which T).
                    _logger.warning(
                        f'iter {global_step}: FIRST non-finite loss '
                        f'({loss.item()}); diagnostic dump:'
                    )
                    for _name, _p in eval_model.named_parameters():
                        _logger.warning(
                            f'  param {_name}: shape={tuple(_p.shape)} '
                            f'nan={int(torch.isnan(_p).any())} '
                            f'inf={int(torch.isinf(_p).any())} '
                            f'max_abs={_p.detach().abs().max().item():.3e}'
                        )
                    _logger.warning(
                        f'  input u: shape={tuple(u.shape)} '
                        f'nan={int(torch.isnan(u).any())} '
                        f'inf={int(torch.isinf(u).any())} '
                        f'max_abs={u.detach().abs().max().item():.3e}'
                    )
                    _yh_nan = int(torch.isnan(y_hat).any())
                    _yh_inf = int(torch.isinf(y_hat).any())
                    _logger.warning(
                        f'  y_hat: nan={_yh_nan} inf={_yh_inf} '
                        f'max_abs={y_hat.detach().abs().max().item():.3e}'
                    )
                    _hb_bad = torch.isnan(h_buf) | torch.isinf(h_buf)
                    if _hb_bad.any():
                        # Reduce sequentially: (B, T, N) -> (T, N) -> (T,).
                        _bad_per_t = _hb_bad.any(dim=0).any(dim=1)
                        _first_t = int(_bad_per_t.nonzero()[0].item())
                        _h_prev_max = (
                            h_buf[:, _first_t - 1].abs().max().item()
                            if _first_t > 0 else float('nan')
                        )
                        _logger.warning(
                            f'  h_buf: first non-finite at t={_first_t} '
                            f'(of T={h_buf.shape[1]}); '
                            f'h_buf[t-1].max_abs={_h_prev_max:.3e}'
                        )
                    else:
                        _logger.warning(
                            f'  h_buf: all finite, '
                            f'max_abs={h_buf.detach().abs().max().item():.3e}'
                        )
                elif n_nan_skips % 50 == 0:
                    _logger.warning(
                        f'iter {global_step}: non-finite loss '
                        f'({loss.item()}); skipping step '
                        f'(total skips={n_nan_skips})'
                    )
                continue

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Post-clip guardrail: skip the step if any parameter gradient
            # is non-finite (NaN/Inf). Clears the bad grads so the next
            # backward starts clean, and counts the skip for diagnostics.
            grads_finite = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if not grads_finite:
                optimizer.zero_grad(set_to_none=True)
                n_nan_skips += 1
                lr_scheduler.step()
                if n_nan_skips == 1 or n_nan_skips % 50 == 0:
                    _logger.warning(
                        f'iter {global_step}: non-finite gradient; '
                        f'skipping step (total skips={n_nan_skips})'
                    )
                continue
            optimizer.step()
            lr_scheduler.step()

            # Param-finiteness rollback. clip_grad_norm bounds the L2 norm of
            # the gradients, but Adam can still amplify a single component
            # past finite range; if any param goes NaN/Inf, every subsequent
            # forward returns NaN and the NaN-loss guard above traps forever.
            # Restore from the rolling backup and reset optimizer state.
            params_finite = all(
                torch.isfinite(p).all() for p in eval_model.parameters()
            )
            if params_finite:
                last_good_model_state = {
                    k: v.detach().clone()
                    for k, v in eval_model.state_dict().items()
                }
                last_good_opt_state = optimizer.state_dict()
            else:
                eval_model.load_state_dict(last_good_model_state)
                optimizer.load_state_dict(last_good_opt_state)
                n_nan_skips += 1
                if n_nan_skips == 1 or n_nan_skips % 50 == 0:
                    _logger.warning(
                        f'iter {global_step}: param NaN after step; '
                        f'restored model+optimizer from rolling backup '
                        f'(total skips={n_nan_skips})'
                    )
                continue

            # Uniform-in-global-step cadence: fires at gs = 1, snap_every+1,
            # 2*snap_every+1, ... plus a final one at end-of-training. Avoids
            # the end-of-epoch / start-of-next-epoch burst the per-epoch
            # `N % snap_every == 0 or N == Niter - 1` rule used to produce.
            if (global_step - 1) % snap_every == 0 or global_step == total_iters:
                with torch.no_grad():
                    # Eval/snapshot use varying shapes (B=512 for pi_acc,
                    # B=64/T=T_epoch for fwhm, B=1/T=snapshot_n_steps for the
                    # rollout) — pass the un-compiled module so we don't
                    # thrash the CUDA-Graph cache or trip dynamo tracer bugs.
                    last_pi_acc = path_integration_accuracy_from_data(
                        eval_model, u_test[:512, :T_epoch], y_test[:512, :T_epoch],
                        warmup=10, batch_size=tc.batch_size,
                    )
                    last_fwhm = bump_fwhm(
                        eval_model, eval_model.epg_indices, eval_model.epg_glom_ix,
                        n_trials=64, n_steps=T_epoch, device=device,
                    )
                    # Primary rollout at the current curriculum horizon —
                    # tracks training progress at the length actually trained.
                    last_rmse_roll, last_pearson_roll = _rollout_heading_metrics(
                        eval_model,
                        n_steps=T_epoch,
                        omega_deg_per_s=snapshot_omega_deg,
                        device=device,
                    )
                    # Reference rollout at fixed T=1000 — the evolution plot
                    # also uses T=1000 for its `heading tracking on snapshot
                    # rollout` panel, so the second value in r_roll=A (B)
                    # equals the `r=` printed in that panel's title.
                    _, last_pearson_roll_1k = _rollout_heading_metrics(
                        eval_model,
                        n_steps=1000,
                        omega_deg_per_s=snapshot_omega_deg,
                        device=device,
                    )
                _save_training_snapshot(
                    net=eval_model, log_dir=log_dir,
                    kinograph_dir=kinograph_dir,
                    global_step=global_step, epoch=epoch + 1,
                    iter_in_epoch=N + 1,
                    neuron_types=eval_model.neuron_types,
                    type_names=eval_model.type_names,
                    epg_indices=eval_model.epg_indices,
                    epg_glom_ix=eval_model.epg_glom_ix,
                    device=device,
                    snapshot_n_steps=snapshot_n_steps,
                    snapshot_omega_deg=snapshot_omega_deg,
                    config=config,
                )
                with open(metrics_log_path, 'a') as f:
                    fwhm_deg = (np.degrees(last_fwhm)
                                if not np.isnan(last_fwhm) else float('nan'))
                    f.write(f'{global_step},{epoch+1},{loss.item():.6f},'
                            f'{mse.item():.6f},{float(cosd):.6f},{float(norm):.6f},'
                            f'{float(tv):.6f},{float(l1S):.6f},'
                            f'{last_pi_acc:.6f},{fwhm_deg:.3f},'
                            f'{last_pearson_roll:.6f},{last_rmse_roll:.3f},'
                            f'{last_pearson_roll_1k:.6f}\n')

                # --- Memory debug (CPU RSS + GPU alloc/reserved) -----------
                # try:
                #     with open('/proc/self/status', 'r') as _sf:
                #         _rss_kb = next(
                #             (int(line.split()[1]) for line in _sf
                #              if line.startswith('VmRSS:')), 0)
                #     cpu_mb = _rss_kb / 1024.0
                # except Exception:
                #     cpu_mb = float('nan')
                # if torch.cuda.is_available():
                #     gpu_alloc_mb = torch.cuda.memory_allocated(device) / 1024**2
                #     gpu_reserved_mb = torch.cuda.memory_reserved(device) / 1024**2
                #     gpu_peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                #     _logger.info(
                #         f'[mem] iter={global_step}  '
                #         f'CPU_RSS={cpu_mb:.0f}MB  '
                #         f'GPU_alloc={gpu_alloc_mb:.0f}MB  '
                #         f'GPU_reserved={gpu_reserved_mb:.0f}MB  '
                #         f'GPU_peak={gpu_peak_mb:.0f}MB'
                #     )
                #     torch.cuda.reset_peak_memory_stats(device)
                # else:
                #     _logger.info(
                #         f'[mem] iter={global_step}  CPU_RSS={cpu_mb:.0f}MB'
                #     )

            if loss.item() < best_loss:
                best_loss = loss.item()

            # Progress bar: replaced fwhm with deterministic-sweep rollout
            # metrics. Pearson is colour-coded (red < 0.5, orange < 0.9, green).
            # Format: r_roll=<T_epoch> (<snapshot_n_steps>). The second value
            # matches the `r=` printed in the evolution-plot panel title.
            if np.isnan(last_rmse_roll):
                rmse_roll_str = 'n/a'
            else:
                rmse_roll_str = f'{last_rmse_roll:.1f}°'

            def _fmt_r(r):
                if np.isnan(r):
                    return 'n/a'
                if r >= 0.9:
                    c = '\033[32m'
                elif r >= 0.5:
                    c = '\033[33m'
                else:
                    c = '\033[31m'
                return f'{c}{r:.3f}\033[0m'

            pearson_str = f'{_fmt_r(last_pearson_roll)} ({_fmt_r(last_pearson_roll_1k)})'
            skips_str = f'  skips={n_nan_skips}' if n_nan_skips > 0 else ''
            pbar.set_postfix_str(
                f'loss={loss.item():.4f} '
                f'rmse_roll={rmse_roll_str} '
                f'r_roll={pearson_str} '
                f'best={best_loss:.4f}{skips_str}'
            )

        # Per-epoch checkpoint (matches data_train_gnn's naming). Save the
        # un-compiled module's state_dict so the file isn't tied to dynamo.
        ckpt_path = os.path.join(
            log_dir, 'models',
            f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt')
        torch.save({'model_state_dict': eval_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ckpt_path)
        _logger.info(
            f'epoch {epoch+1}/{tc.n_epochs} done — last_loss={loss.item():.4f}  '
            f'best={best_loss:.4f}  pi_acc={last_pi_acc:.4f}  saved {ckpt_path}'
        )

    # --- Final eval on full test split (full T) -------------------------
    final_pi = path_integration_accuracy_from_data(
        eval_model, u_test, y_test, warmup=10, batch_size=tc.batch_size,
    )
    _logger.info(f'final test pi_acc: {final_pi:.4f}  '
                 f'(n_test={u_test.shape[0]}, T={u_test.shape[1]})')
    logger.info(f'final test pi_acc: {final_pi:.4f}')


# ============================================================================
# Cortex (Yang 2019) task trainer (TaskRNN, free-W mode)
# ============================================================================

def _data_train_cortex_task(config, erase, best_model, device, log_file=None):
    """Train a TaskRNN (free-W mode) on a Yang cortex task (delaygo etc.).

    Data layout under <dataset>/{train,test}/{stimulus,target,c_mask}.zarr
    (produced by `_generate_cortex_task`):
        stimulus.zarr  (N, T, N_i)   padded Yang trial.x
        target.zarr    (N, T, N_o)   padded Yang trial.y    [fixation + ring]
        c_mask.zarr    (N, T, N_o)   padded Yang c_mask      (Yang lsq loss)

    Loss = mean(c_mask · (y_hat − y)²)
        + tc.coeff_W_L2    · ‖W_rec‖²
        + tc.coeff_rate_L2 · mean(σ(h)²)

    Eval (via `cortex_eval.compute_cortex_task_metrics` over N_EVAL test
    trials): {loss, motor_max, motor_peak_mean, direction_acc}.

    Snapshots at `snapshots_per_epoch` cadence via
    `cortex_eval.save_cortex_training_snapshot` (8-panel figure mirroring
    papers/multi-tasks/notebooks/analyze_gnn.ipynb cell 7).

    metrics.log schema (cortex):
        iteration,epoch,loss,mse,motor_max,motor_peak_mean,direction_acc
    """
    import torch.nn.functional as F

    from connectome_gnn.models.cortex_eval import (
        compute_cortex_task_metrics,
        save_cortex_matrix_snapshot,
        save_cortex_training_snapshot,
    )
    from connectome_gnn.zarr_io import load_raw_array

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    tc = config.training
    model_config = config.graph_model
    ct = config.task.cortex

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    random.seed(tc.seed)

    default_style.apply_globally()

    log_dir, logger = create_log_dir(config, erase)
    # Wipe tmp_training so snapshots, metrics, etc. don't mix across runs.
    shutil.rmtree(os.path.join(log_dir, 'tmp_training'), ignore_errors=True)
    snapshot_dir = os.path.join(log_dir, 'tmp_training', 'cortex_snapshot')
    matrix_dir = os.path.join(log_dir, 'tmp_training', 'matrix')
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)

    # --- Eager load: trials stay on GPU between iterations ---------------
    root = graphs_data_path(config.dataset)
    _logger.info(f'loading task data from {root}/(train|test)/...')
    u_train  = torch.from_numpy(load_raw_array(f"{root}/train/stimulus.zarr")).to(device)
    y_train  = torch.from_numpy(load_raw_array(f"{root}/train/target.zarr")).to(device)
    cm_train = torch.from_numpy(load_raw_array(f"{root}/train/c_mask.zarr")).to(device)
    u_test   = torch.from_numpy(load_raw_array(f"{root}/test/stimulus.zarr")).to(device)
    y_test   = torch.from_numpy(load_raw_array(f"{root}/test/target.zarr")).to(device)
    cm_test  = torch.from_numpy(load_raw_array(f"{root}/test/c_mask.zarr")).to(device)
    _logger.info(f'task data: train u={tuple(u_train.shape)} y={tuple(y_train.shape)} '
                 f'cm={tuple(cm_train.shape)}  '
                 f'test u={tuple(u_test.shape)} y={tuple(y_test.shape)} '
                 f'cm={tuple(cm_test.shape)}')
    logger.info(f'train trials: {u_train.shape[0]}  test trials: {u_test.shape[0]}  '
                f'T: {u_train.shape[1]}  in: {u_train.shape[2]}  out: {y_train.shape[2]}')

    # --- Model build via registry ----------------------------------------
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type,
                         config=config, device=device)
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'model {model_config.signal_model_name} '
                 f'(W_param={model.W_param}, sigma={model.recurrent_activation_name}): '
                 f'{n_total_params:,} trainable params')
    logger.info(f'model: {model_config.signal_model_name}  params: {n_total_params}')

    # --- Optimizer + scheduler -------------------------------------------
    # Three named param groups (mirrors _data_train_drosophila_cx_task). Missing field
    # → tc.lr fallback so old single-LR configs still work:
    #   - "w_rec": recurrent core. _W_rec_free (cortex) — and for forward-
    #              compat with GNN cortex variants, also W/a/g_phi.*/f_theta.*.
    #              lr starts at tc.lr_W_rec or tc.lr. lr_W_rec_schedule drives
    #              THIS group exclusively (per-epoch trajectory).
    #   - "w_ED":  encoder/decoder. W_in, W_out, _W_in_mlp.*, _W_out_mlp.*.
    #              lr = tc.lr_W_ED or tc.lr. Constant — schedule does not touch.
    #   - "other": biases (b, b_out) and anything not in the above. lr = tc.lr.
    #              Constant.
    lr_W_rec = getattr(tc, 'lr_W_rec', None)
    lr_W_ED = getattr(tc, 'lr_W_ED', None)

    def _name_to_group(name: str) -> str:
        if (name == "_W_rec_free"
                or name in ("S", "W", "a")
                or name.startswith(("g_phi.", "f_theta."))):
            return "w_rec"
        if (name in ("W_in", "W_out")
                or name.startswith(("_W_in_mlp.", "_W_out_mlp."))):
            return "w_ED"
        return "other"

    grouped: dict[str, list] = {"w_rec": [], "w_ED": [], "other": []}
    for _name, _p in model.named_parameters():
        grouped[_name_to_group(_name)].append(_p)

    optimizer = torch.optim.Adam(
        [
            {"params": grouped["w_rec"],
             "lr": float(lr_W_rec) if lr_W_rec is not None else tc.lr,
             "name": "w_rec"},
            {"params": grouped["w_ED"],
             "lr": float(lr_W_ED) if lr_W_ED is not None else tc.lr,
             "name": "w_ED"},
            {"params": grouped["other"], "lr": tc.lr, "name": "other"},
        ]
    )
    _logger.info(
        f'three-group optimizer: '
        f'w_rec lr={float(lr_W_rec) if lr_W_rec is not None else tc.lr} '
        f'({len(grouped["w_rec"])} params, schedule-driven) | '
        f'w_ED lr={float(lr_W_ED) if lr_W_ED is not None else tc.lr} '
        f'({len(grouped["w_ED"])} params, constant) | '
        f'other lr={tc.lr} ({len(grouped["other"])} params, constant)'
    )
    lr_scheduler = build_lr_scheduler(optimizer, config)
    _logger.info(f'lr={tc.lr}  lr_scheduler={getattr(tc, "lr_scheduler", "none")}')

    # --- Regulariser coefficients (cached as Python scalars) -------------
    coeff_W_L2 = float(getattr(tc, 'coeff_W_L2', 0.0))
    coeff_rate_L2 = float(getattr(tc, 'coeff_rate_L2', 0.0))
    grad_clip = float(getattr(tc, 'grad_clip_W', 0.0))
    # Snapshot cadence: prefer absolute `snap_every_iters` (decoupled from
    # epoch length so DAL doesn't change the snapshot rate). Falls back to
    # `snapshots_per_epoch` if `snap_every_iters` is 0 (default).
    snapshots_per_epoch = int(getattr(tc, 'snapshots_per_epoch', 1))
    snap_every_iters = int(getattr(tc, 'snap_every_iters', 0))
    _logger.info(f'losses: masked_mse + W_L2={coeff_W_L2}  rate_L2={coeff_rate_L2}  '
                 f'grad_clip={grad_clip}')

    # --- Training loop ---------------------------------------------------
    n_trials = u_train.shape[0]
    # data_augmentation_loop multiplies iters/epoch by sampling batches with
    # replacement (Yang's reference trainer generates trials on-the-fly each
    # iter; we approximate that by reusing the pre-generated trial pool).
    dal = int(getattr(tc, 'data_augmentation_loop', 1))
    Niter = max(1, (n_trials // tc.batch_size) * dal)
    if snap_every_iters > 0:
        snap_every = snap_every_iters
    else:
        snap_every = max(1, Niter // max(1, snapshots_per_epoch))
    rule_name = (ct.rules[0] if getattr(ct, "rules", None) else "cortex")
    global_step = 0

    # Per-epoch schedules for the named groups (mirrors PI). Each is optional
    # and drives only its own group; "other" always stays constant at lr.
    def _build_lr_schedule(field_name: str):
        raw = list(getattr(tc, field_name, []) or [])
        if not raw:
            return None
        if len(raw) < tc.n_epochs:
            raw = raw + [raw[-1]] * (tc.n_epochs - len(raw))
        sched = [float(x) for x in raw[:tc.n_epochs]]
        _logger.info(f'{field_name} (epochs 1..{tc.n_epochs}): {sched}')
        return sched

    lr_W_rec_schedule = _build_lr_schedule('lr_W_rec_schedule')
    lr_W_ED_schedule = _build_lr_schedule('lr_W_ED_schedule')

    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
    with open(metrics_log_path, 'w') as f:
        f.write('iteration,epoch,loss,mse,motor_max,motor_peak_mean,'
                'r2,direction_acc,r2_filtered,direction_acc_filtered,pct_outliers\n')

    last_metrics = {'loss': float('nan'), 'motor_max': float('nan'),
                    'motor_peak_mean': float('nan'),
                    'r2': float('nan'), 'r2_filtered': float('nan'),
                    'direction_acc': float('nan'),
                    'direction_acc_filtered': float('nan'),
                    'pct_outliers': float('nan')}
    model.train()

    # torch.compile (same pattern as PI trainer; eager fallback for eval).
    eval_model = model
    if getattr(tc, 'torch_compile', True):
        try:
            model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
            logger.info('torch.compile enabled (mode=reduce-overhead, fullgraph=True); '
                        'eval/snapshot forward stays eager via _orig_mod')
            _logger.info('torch.compile enabled (eval via _orig_mod)')
        except Exception as exc:
            _logger.warning(f'torch.compile failed, falling back to eager: {exc}')
            logger.info(f'torch.compile failed: {exc}')
    else:
        logger.info('torch.compile disabled via config (torch_compile: false)')

    n_eval = min(64, u_test.shape[0])
    total_iters = tc.n_epochs * Niter
    _logger.info(f'start training: {tc.n_epochs} epochs × {Niter} iters/epoch '
                 f'= {total_iters} iters  (n_trials={n_trials}, DAL={dal}, '
                 f'n_eval={n_eval} test trials, snap_every={snap_every} iters '
                 f'= {total_iters // snap_every} snapshots)')


    for epoch in range(tc.n_epochs):
        # Per-epoch lr replacement for the named groups. Each schedule is
        # optional and drives only its own group; "other" always stays at lr.
        for _gname, _gsched in (("w_rec", lr_W_rec_schedule),
                                ("w_ED", lr_W_ED_schedule)):
            if _gsched is not None:
                _lr = _gsched[epoch]
                for g in optimizer.param_groups:
                    if g.get("name") == _gname:
                        g['lr'] = _lr
                _logger.info(f'epoch {epoch+1}: {_gname} lr -> {_lr}')
        pbar = trange(
            Niter, ncols=150,
            desc=f'cortex/{rule_name} epoch {epoch+1}/{tc.n_epochs}',
            leave=True,
        )
        for N in pbar:
            global_step += 1
            # Sample with replacement (DAL > 1 makes one-pass coverage
            # impossible from a fixed trial pool). For DAL=1 this is
            # functionally equivalent to the bootstrap of a single pass.
            idx = torch.randint(0, n_trials, (tc.batch_size,), device=device)
            u = u_train[idx]
            y = y_train[idx]
            cm = cm_train[idx]

            y_hat, h_buf = model(u)
            sq_err = (y_hat - y) ** 2
            mse = (sq_err * cm).mean()
            W_L2 = (coeff_W_L2 * eval_model.W_rec.pow(2).sum()
                    if coeff_W_L2 > 0 else u.new_zeros(()))
            rate_L2 = (coeff_rate_L2 * eval_model._sigma(h_buf).pow(2).mean()
                       if coeff_rate_L2 > 0 else u.new_zeros(()))
            loss = mse + W_L2 + rate_L2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()

            if N % snap_every == 0 or N == Niter - 1:
                with torch.no_grad():
                    # Eval on the first n_eval test trials via the un-compiled
                    # module (varying B between train and eval otherwise
                    # thrashes the CUDA-Graph cache).
                    y_eval, _ = eval_model(u_test[:n_eval])
                    stimuli = [u_test[i] for i in range(n_eval)]
                    preds = [y_eval[i] for i in range(n_eval)]
                    targets = [y_test[i] for i in range(n_eval)]
                    cmasks = [cm_test[i] for i in range(n_eval)]
                    last_metrics = compute_cortex_task_metrics(preds, targets, cmasks)
                    snap_path = os.path.join(
                        snapshot_dir, f'step_{global_step:06d}.png')
                    try:
                        save_cortex_training_snapshot(
                            stimuli, preds, targets, cmasks,
                            output_path=snap_path, step=global_step,
                            rule_name=rule_name,
                        )
                    except Exception as exc:
                        _logger.warning(
                            f'[cortex_eval] snapshot failed @ step {global_step}: {exc}')
                    # W_rec matrix view — saved at the same cadence.
                    matrix_path = os.path.join(
                        matrix_dir, f'step_{global_step:06d}.png')
                    try:
                        save_cortex_matrix_snapshot(
                            eval_model.W_rec,
                            output_path=matrix_path, step=global_step,
                            title_suffix=f'epoch {epoch + 1}/{tc.n_epochs}',
                        )
                    except Exception as exc:
                        _logger.warning(
                            f'[cortex_eval] matrix snapshot failed @ step '
                            f'{global_step}: {exc}')
                with open(metrics_log_path, 'a') as f:
                    f.write(f'{global_step},{epoch+1},{loss.item():.6f},'
                            f'{mse.item():.6f},'
                            f'{last_metrics["motor_max"]:.6f},'
                            f'{last_metrics["motor_peak_mean"]:.6f},'
                            f'{last_metrics.get("r2", float("nan")):.6f},'
                            f'{last_metrics["direction_acc"]:.6f},'
                            f'{last_metrics.get("r2_filtered", float("nan")):.6f},'
                            f'{last_metrics.get("direction_acc_filtered", float("nan")):.6f},'
                            f'{last_metrics.get("pct_outliers", float("nan")):.4f}\n')

            da = last_metrics["direction_acc"]
            da_f = last_metrics.get("direction_acc_filtered", float("nan"))
            r2 = last_metrics.get("r2", float("nan"))
            r2_f = last_metrics.get("r2_filtered", float("nan"))
            pct = last_metrics.get("pct_outliers", float("nan"))
            col_r2 = r2_color(r2_f) if r2_f == r2_f else ""
            col_da = r2_color(da_f) if da_f == da_f else ""
            col_pct = ANSI_ORANGE if (pct == pct and pct > 15) else ""
            pbar.set_postfix_str(
                f'loss={loss.item():.2e}  '
                f'{col_r2}R2={r2_f:.3f}{ANSI_RESET} ({r2:.3f})  '
                f'{col_da}dir_acc={da_f:.2f}{ANSI_RESET} ({da:.2f})  '
                f'{col_pct}outlier={pct:.0f}%{ANSI_RESET if col_pct else ""}'
            )

        pbar.close()

        # Per-epoch checkpoint (matches PI trainer's naming).
        ckpt_path = os.path.join(
            log_dir, 'models',
            f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt')
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({'model_state_dict': eval_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ckpt_path)

    # --- Final eval on full test split ----------------------------------
    with torch.no_grad():
        y_eval, _ = eval_model(u_test)
        final_preds   = [y_eval[i]  for i in range(u_test.shape[0])]
        final_targets = [y_test[i]  for i in range(u_test.shape[0])]
        final_cmasks  = [cm_test[i] for i in range(u_test.shape[0])]
        final_metrics = compute_cortex_task_metrics(final_preds, final_targets, final_cmasks)
    _r2_f = final_metrics["r2_filtered"]
    _da_f = final_metrics["direction_acc_filtered"]
    _pct = final_metrics["pct_outliers"]
    _c_r2 = r2_color(_r2_f) if _r2_f == _r2_f else ""
    _c_da = r2_color(_da_f) if _da_f == _da_f else ""
    _c_pct = ANSI_ORANGE if (_pct == _pct and _pct > 15) else ""
    _logger.info(
        f'final test  '
        f'{_c_r2}R²={_r2_f:.4f}{ANSI_RESET} ({final_metrics["r2"]:.4f})  '
        f'{_c_da}dir_acc={_da_f:.4f}{ANSI_RESET} '
        f'({final_metrics["direction_acc"]:.4f})  '
        f'{_c_pct}outlier={_pct:.1f}%{ANSI_RESET if _c_pct else ""}  '
        f'(n_test={u_test.shape[0]}, T={u_test.shape[1]})'
    )
    logger.info(f'final test direction_acc: {final_metrics["direction_acc"]:.4f}')
