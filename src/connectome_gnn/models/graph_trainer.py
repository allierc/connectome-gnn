import logging
import os
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
    LossRegularizer,
    _batch_frames,
    analyze_data_svd,
    set_trainable_parameters,
)
from connectome_gnn.plot import (
    plot_jacobian_w_scatter,
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

ANSI_RESET = '\033[0m'
ANSI_GREEN = '\033[92m'
ANSI_YELLOW = '\033[93m'
ANSI_ORANGE = '\033[38;5;208m'
ANSI_RED = '\033[91m'

def _quick_ngp_pearson(model, x_ts, ids, *, use_anchor, device,
                        n_neurons_sample=64, n_frames_sample=256, rng=None):
    """Lightweight per-neuron Pearson r between batched NGP output and GT.

    Used by the tqdm bar to surface a real-time fit metric for the
    hidden-neuron InstantNGP between the heavyweight
    plot_training_flyvis checkpoints. Samples a small random subset of
    (neurons, frames), runs a single ``forward_{hidden,anchor}_batched``
    call (no grad), and returns the mean per-neuron Pearson r.

    Args:
        model:        NeuralGNN with NNR_hidden initialised.
        x_ts:         TimeSeries used as ground truth.
        ids:          (N,) torch.long ids — hidden_ids or anchor_ids.
        use_anchor:   pick forward_anchor_batched if True, else
                      forward_hidden_batched.
        device:       trainer device.
        n_neurons_sample / n_frames_sample: subset sizes (defaults are
                      cheap enough to call every ~200 iterations on a
                      single A100 without measurable wall-time cost).
        rng:          optional numpy Generator. Defaults to a fresh
                      default_rng() so the subset varies between calls.

    Returns:
        float | None — mean per-neuron Pearson r, or None if the call
        could not run (e.g. spatial NGP pos cache not yet populated).
    """
    if model.NNR_hidden is None or ids is None or len(ids) == 0:
        return None
    if use_anchor and getattr(model, 'n_anchor', 0) == 0:
        return None

    n_total = int(len(ids))
    n_neurons_sample = min(n_neurons_sample, n_total)
    n_frames_total = int(x_ts.n_frames)
    n_frames_sample = min(n_frames_sample, n_frames_total)
    if rng is None:
        rng = np.random.default_rng()
    sel_n = np.sort(rng.choice(n_total, n_neurons_sample, replace=False))
    sel_f = np.sort(rng.choice(n_frames_total, n_frames_sample, replace=False))

    if isinstance(ids, torch.Tensor):
        sel_ids = ids[sel_n].to(device=device, dtype=torch.long)
    else:
        sel_ids = torch.as_tensor(np.asarray(ids)[sel_n],
                                  device=device, dtype=torch.long)
    k_t = torch.as_tensor(sel_f, device=device, dtype=torch.long)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if use_anchor:
                pred = model.forward_anchor_batched(k_t, anchor_ids=sel_ids)
            else:
                pred = model.forward_hidden_batched(k_t, hidden_ids=sel_ids)
    except RuntimeError:
        # spatial NGP pos cache not yet populated — caller skips this iter
        if was_training:
            model.train()
        return None
    if was_training:
        model.train()

    # Time-only ngp_t (no spatial, no factorized head) ignores the
    # passed-in ids and returns (B, n_total) — every output slot maps 1:1
    # to the trainer's hidden_ids / anchor_ids order. Slice down to the
    # subsampled positions so pred matches gt's shape (B, n_sample).
    # The spatial path already returns (B, n_sample) directly via
    # _ngp_query_spatial, so no slicing needed in that case.
    if pred.shape[1] == n_total:
        pred = pred[:, sel_n]
    elif pred.shape[1] != n_neurons_sample:
        if was_training:
            model.train()
        return None  # shape mismatch we don't know how to recover from

    gt = x_ts.voltage[sel_f][:, sel_ids]                     # (F, N_sample)
    gt_np = gt.detach().to('cpu').numpy().astype(np.float32)
    pred_np = pred.detach().to('cpu').numpy().astype(np.float32)
    g = gt_np - gt_np.mean(axis=0, keepdims=True)
    p = pred_np - pred_np.mean(axis=0, keepdims=True)
    num = (g * p).sum(axis=0)
    denom = np.sqrt((g * g).sum(axis=0)) * np.sqrt((p * p).sum(axis=0))
    corrs = np.where(denom > 1e-12, num / (denom + 1e-12), 0.0)
    return float(corrs.mean())


# How often the tqdm bar gets a refreshed NGP Pearson r between the heavy
# plot_training_flyvis checkpoints. ~100 iters keeps the cost negligible
# (one batched forward over 64 neurons × 256 frames per refresh) while
# updating the bar — and the metrics.log row written from this path —
# often enough to give nnr_plot.png ~2× the resolution of the prior 200.
_NGP_QUICK_FREQ = 100


def r2_color(val, thresholds=(0.9, 0.7, 0.3)):
    """ANSI color for an R² value: green > t0, yellow > t1, orange > t2, red otherwise."""
    t0, t1, t2 = thresholds
    return ANSI_GREEN if val > t0 else ANSI_YELLOW if val > t1 else ANSI_ORANGE if val > t2 else ANSI_RED


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

    _connconstr = any(x in config.dataset for x in ('drosophila_cx', 'zebrafish_oculomotor', 'larva'))
    if 'fly' in config.dataset or _connconstr:
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
    ode_params = OdeParamsCls.load(graphs_data_path(config.dataset), device=device)
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
    # Block 06: differential warmup — ramp MLP/embedding LRs while W stays at full strength
    _diff_warmup_steps = int(getattr(tc, 'differential_warmup_steps', 0))
    if _diff_warmup_steps > 0:
        from connectome_gnn.LLM_code.staging.block_06.differential_warmup import apply_differential_warmup
        _diff_warmup_start = float(getattr(tc, 'differential_warmup_start_fraction', 0.01))
        lr_scheduler = apply_differential_warmup(optimizer, config, warmup_steps=_diff_warmup_steps, warmup_start_fraction=_diff_warmup_start)
        _logger.info(f'differential warmup: {_diff_warmup_steps} steps, start_fraction={_diff_warmup_start}')
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
        # NGP-injection warmup config (two-step training).
        # Phase 1 (N < warmup_inject_nnr_iter): hidden voltages are zero-silenced
        # (alpha=0). The NGP still trains via the anchor loss (which is gated by
        # coeff_anchor_voltage, NOT by alpha), so by phase 2 the NGP already
        # produces sensible per-(t,u,v) outputs at anchor positions.
        # Phase 2 (N >= warmup_inject_nnr_iter): alpha ramps 0 -> 1 over
        # warmup_inject_nnr_ramp_iter steps, then stays at 1. The ramp gives W a
        # window to absorb the new non-zero hidden contribution gradually
        # instead of stepwise. ramp=0 -> hard switch.
        # Defaults (warmup=0, ramp=0) preserve the legacy "always inject" behavior.
        # The *_frac variants override the absolute counts when > 0, expressed
        # as a fraction of Niter — useful when DAL/bs change since the warmup
        # tracks training length automatically.
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
        # alpha_inject_target caps the post-warmup alpha (default 1.0). Set to
        # 0.0 to leave NGP as a passive monitor (no injection ever) — useful
        # to isolate whether the conn_R² drop seen in phase 2 is caused by the
        # injection itself or by the hidden self-consistency loss.
        _alpha_inject_target = float(getattr(tc, 'alpha_inject_target', 1.0))

        # Three-phase schedule (when warmup_hidden_loss_iter_frac > 0):
        #   phase 1 [0, hidden_loss_iter)               : alpha=0, hidden loss OFF
        #   phase 2 [hidden_loss_iter, inject_iter)     : alpha=0, hidden loss ON
        #   phase 3 [inject_iter, ...)                  : alpha ramps to target, hidden loss ON
        # Default (frac=0) = legacy two-phase behavior where the hidden loss
        # gate is tied to alpha_inject (so it activates at the inject ramp).
        _hidden_loss_iter_frac = float(getattr(tc, 'warmup_hidden_loss_iter_frac', 0.0))
        if _hidden_loss_iter_frac > 0.0:
            _hidden_loss_iter = int(Niter * _hidden_loss_iter_frac)
        else:
            _hidden_loss_iter = int(getattr(tc, 'warmup_hidden_loss_iter', 0))
        _three_phase = _hidden_loss_iter > 0

        if _warmup_inject_iter > 0:
            if _three_phase:
                print(f'NGP three-phase schedule: '
                      f'phase 1 [0, {_hidden_loss_iter}) GNN+anchor only, '
                      f'phase 2 [{_hidden_loss_iter}, {_warmup_inject_iter}) +hidden loss (alpha=0), '
                      f'phase 3 ramp [{_warmup_inject_iter}, {_warmup_inject_iter + _warmup_inject_ramp}) → '
                      f'inject (alpha={_alpha_inject_target}). Niter={Niter}.')
            else:
                print(f'NGP warmup-inject: phase 1 = iters [0, {_warmup_inject_iter}) '
                      f'(alpha=0), ramp [{_warmup_inject_iter}, {_warmup_inject_iter + _warmup_inject_ramp}), '
                      f'phase 2 from iter {_warmup_inject_iter + _warmup_inject_ramp} '
                      f'(alpha={_alpha_inject_target}). Total Niter={Niter}.')

        # Track previous-iter alpha so we can announce the two phase transitions
        # (warmup-end / ramp-start, ramp-end / phase-2-start).
        _prev_alpha_inject = -1.0

        for N in pbar:

            # Compute per-iter alpha for NGP-into-hidden injection.
            # Branchless inside the compiled forward (just a scalar multiply);
            # the if-tree below runs once in the Python loop, not in any
            # compiled region. Final value is scaled by alpha_inject_target so
            # `alpha_inject_target=0` leaves alpha at 0 throughout (NGP never
            # injected — passive-monitor mode).
            if _warmup_inject_iter <= 0:
                alpha_inject = 1.0
            elif N < _warmup_inject_iter:
                alpha_inject = 0.0
            elif _warmup_inject_ramp > 0 and N < _warmup_inject_iter + _warmup_inject_ramp:
                alpha_inject = float(N - _warmup_inject_iter) / float(_warmup_inject_ramp)
            else:
                alpha_inject = 1.0
            alpha_inject = alpha_inject * _alpha_inject_target

            # Hidden-voltage self-consistency gate. Three-phase: turns on at
            # _hidden_loss_iter (independent of alpha_inject), so phase 2 can
            # train NGP-hidden against GNN(v_h=0) targets BEFORE injection
            # starts in phase 3 — pre-shaping NGP-hidden away from the
            # trivial-zero fixed point that emerges once alpha>0.
            # Two-phase fallback (when _three_phase is False): the gate
            # follows alpha_inject (legacy behavior).
            if _three_phase:
                hidden_loss_gate = 1.0 if N >= _hidden_loss_iter else 0.0
            else:
                hidden_loss_gate = alpha_inject

            # Announce phase transitions when crossed.
            if _warmup_inject_iter > 0 and _prev_alpha_inject != alpha_inject:
                if _prev_alpha_inject == 0.0 and alpha_inject > 0.0:
                    if _warmup_inject_ramp > 0:
                        print(f'\n[NGP warmup] iter {N}: phase 1 → ramp '
                              f'(alpha 0 -> 1 over {_warmup_inject_ramp} iters). '
                              f'Hidden NGP injection now ramping in.')
                    else:
                        print(f'\n[NGP warmup] iter {N}: phase 1 → phase 2 '
                              f'(alpha 0 -> 1, hard switch). '
                              f'Hidden NGP injection now fully active.')
                elif _prev_alpha_inject < 1.0 and alpha_inject >= 1.0:
                    print(f'\n[NGP warmup] iter {N}: ramp → phase 2 '
                          f'(alpha = 1). Hidden NGP injection now fully active.')
            _prev_alpha_inject = alpha_inject

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

                # Fast NGP Pearson refresh — independent of the heavy R²
                # checkpoint above. Subsamples (64 neurons × 256 frames),
                # one batched forward, no grad. Only fires when the model
                # has a hidden-neuron INR. Also appends a metrics.log row
                # so the runner's 300s collector picks the updated values
                # up between heavy checkpoints.
                _ngp_quick_updated = False
                if (has_hidden_neurons
                        and getattr(model, 'NNR_hidden', None) is not None
                        and N > 0 and N % _NGP_QUICK_FREQ == 0):
                    _h_quick = _quick_ngp_pearson(
                        model, x_ts, hidden_ids,
                        use_anchor=False, device=device)
                    if _h_quick is not None:
                        last_hidden_r2 = _h_quick
                        _ngp_quick_updated = True
                    if has_anchor_neurons:
                        _a_quick = _quick_ngp_pearson(
                            model, x_ts, anchor_ids,
                            use_anchor=True, device=device)
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
                        # During warmup (alpha_inject=0), hidden voltages are
                        # zero-silenced so hidden_nnr_pearson is meaningless;
                        # show NA. Anchor is still trained throughout, show it.
                        if alpha_inject <= 0.0:
                            if last_anchor_r2 is not None:
                                nnr_str = f'nnr=NA({last_anchor_r2:.3f})'
                            else:
                                nnr_str = 'nnr=NA'
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
                # alpha_inject ramps 0->1 across the warmup window: in phase 1
                # this is a multiply-by-zero so the GNN sees voltage[hidden]=0
                # exactly (same as the no-NGP baseline), while the NGP still
                # gets gradient from the anchor loss elsewhere in the step.
                if has_hidden_neurons:
                    if model.NNR_hidden is not None:
                        x.voltage[hidden_ids] = alpha_inject * model.forward_hidden(x, k, hidden_ids)
                    else:
                        x.voltage[hidden_ids] = 0.0

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

                            batched_state, batched_edges = _batch_frames(state_batch, edges)
                            pred, in_features, msg = model(batched_state, batched_edges, data_id=data_id, return_all=True)

                            pred_x = pred_x + sim.delta_t * pred + tc.noise_recurrent_level * torch.randn_like(pred)

                    loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (sim.delta_t * tc.time_step)).norm(2)

                else:

                    loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)
                    # Hidden voltage consistency loss: GNN-predicted v(k+1) vs NGP(k+1).
                    # No oracle GT leak at hidden neurons — the target is the INR's own
                    # prediction at t+1, so agreement requires GNN dynamics and INR trace
                    # to be self-consistent. Degeneracy (both constant) is prevented by
                    # the anchor loss and the visible-neuron rollout loss.
                    # Hidden self-consistency loss: gated by alpha_inject so it
                    # is OFF in phase 1 (alpha=0) and ON in phase 2 (alpha=1).
                    # Phase 1 the GNN sees voltage[hidden]=0, so a self-consistency
                    # term vs NGP(t+1) would push the GNN toward 0 instead of the
                    # true dynamics. Multiplying by alpha cleanly disables it.
                    if has_hidden_neurons and getattr(tc, 'coeff_hidden_voltage', 0.0) > 0:
                        n_per = state_batch[0].n_neurons
                        h_ids_b = torch.cat([hidden_ids + b * n_per for b in range(len(state_batch))]).to(device)
                        pred_h = batched_state.voltage[h_ids_b].unsqueeze(-1) + sim.delta_t * pred[h_ids_b]
                        k_starts = k_batch[::n_per, 0].to(torch.long)                 # (B,)
                        target_h = model.forward_hidden_batched(k_starts + 1, hidden_ids=hidden_ids).reshape(-1, 1)  # (B*n_hidden, 1)
                        # hidden_loss_gate: in three-phase mode this turns on
                        # at _hidden_loss_iter even while alpha_inject=0, so
                        # NGP-hidden gets shaped against GNN(v_h=0) targets
                        # before injection starts. In legacy two-phase mode it
                        # equals alpha_inject (gate tied to injection ramp).
                        loss = loss + hidden_loss_gate * tc.coeff_hidden_voltage * (pred_h - target_h).norm(2)
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

                # Fast NGP Pearson refresh — independent of the heavy R²
                # checkpoint above. Subsamples (64 neurons × 256 frames),
                # one batched forward, no grad. Also appends a metrics.log
                # row so the runner's 300s collector picks the updated
                # values up between heavy checkpoints.
                _ngp_quick_updated = False
                if (has_hidden_neurons
                        and getattr(model, 'NNR_hidden', None) is not None
                        and N > 0 and N % _NGP_QUICK_FREQ == 0):
                    _h_quick = _quick_ngp_pearson(
                        model, x_ts, hidden_ids,
                        use_anchor=False, device=device)
                    if _h_quick is not None:
                        last_hidden_r2 = _h_quick
                        _ngp_quick_updated = True
                    if has_anchor_neurons:
                        _a_quick = _quick_ngp_pearson(
                            model, x_ts, anchor_ids,
                            use_anchor=True, device=device)
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
                        # During warmup (alpha_inject=0), hidden voltages are
                        # zero-silenced so hidden_nnr_pearson is meaningless;
                        # show NA. Anchor is still trained throughout, show it.
                        if alpha_inject <= 0.0:
                            if last_anchor_r2 is not None:
                                nnr_str = f'nnr=NA({last_anchor_r2:.3f})'
                            else:
                                nnr_str = 'nnr=NA'
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

    _connconstr = any(x in config.dataset for x in ('drosophila_cx', 'zebrafish_oculomotor', 'larva'))
    if 'fly' in config.dataset or _connconstr:
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
from connectome_gnn.models.graph_tester import data_test_gnn, data_test_gnn_special
