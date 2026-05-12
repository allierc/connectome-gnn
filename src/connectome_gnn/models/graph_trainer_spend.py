"""SPEND-style Noise2Noise training for FlyVis GNN under measurement noise.

Ports three Noise2Noise variants from SPEND onto the existing flyvis-A
neural-ODE GNN to attack the measurement-noise bottleneck (gamma=0.10) without
modifying the production trainer.

Cite: https://github.com/buchenglab/SPEND  (Ding et al. 2025, Newton 1, 100195;
      "Self-supervised elimination of non-independent noise in hyperspectral
      imaging").

Three add-ons, each gated by its own coefficient:

  - coeff_spend_replay  (Add-on #3): stimulus-replay N2N. Load CLEAN voltage
                        (measurement_noise_level=0) and synthesize two
                        independent Gaussian noise tensors n_a, n_b with
                        explicit RNG seeds. Train an inline 1D-conv smoother
                        with ||smoother(v + n_a) - (v + n_b)||^2. The
                        smoother's output replaces v + n_a as the GNN input.
                        Closest direct port of SPEND (datasplit_with_aug_choose
                        Img_Split_Conc, but with two noise seeds rather than
                        even/odd permutation).
  - coeff_spend_time    (Add-on #1): time-permutation N2N. Even-frame trace as
                        input, odd-frame trace as N2N target (linear half-frame
                        interpolation, valid because dt=20ms is small relative
                        to dynamics). Direct analog of SPEND's Img_Split_Conc.
  - coeff_spend_typed   (Add-on #2): typed-equivariance. Pairs of same-type
                        neurons within position-distance threshold share
                        (approximately) the same clean signal; their voltage
                        difference is pure noise. Loss = ||v_i1 - v_i2||^2
                        - 2*gamma^2 (noise-cancelled estimator).

Smoother is co-trained with the GNN under a single Adam optimizer (second
param group). It is discarded at inference time; the deliverable is the
trained GNN, evaluated by `data_test` with strict=False checkpoint loading.

Invocation: import directly, do not modify graph_trainer.data_train dispatch.

    from connectome_gnn.models.graph_trainer_spend import data_train_spend
    data_train_spend(config, device=device)

Following the precedent set by graph_trainer_inr.data_train_INR.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from connectome_gnn.figure_style import default_style
from connectome_gnn.log import get_logger
from connectome_gnn.metrics import compute_dynamics_r2
from connectome_gnn.models.training_utils import (
    build_lr_scheduler,
    build_model,
    determine_load_fields,
    load_flyvis_data,
)
from connectome_gnn.models.utils import (
    LossRegularizer,
    _batch_frames,
    analyze_data_svd,
    set_trainable_parameters,
)
from connectome_gnn.plot import plot_signal_loss, plot_training_flyvis
from connectome_gnn.utils import (
    check_and_clear_memory,
    create_log_dir,
    graphs_data_path,
)

_logger = get_logger(__name__)

# ANSI colors — matches graph_trainer.py
ANSI_RESET = '\033[0m'
ANSI_GREEN = '\033[92m'
ANSI_YELLOW = '\033[93m'
ANSI_ORANGE = '\033[38;5;208m'
ANSI_RED = '\033[91m'


def _r2c(v):
    return ANSI_GREEN if v > 0.9 else ANSI_YELLOW if v > 0.7 else ANSI_ORANGE if v > 0.3 else ANSI_RED


# ============================================================================
# SPEND helpers (only three; per plan).
# Cite: https://github.com/buchenglab/SPEND
# ============================================================================

def _synth_noise_pair(shape, gamma, seed_a, seed_b, device, rho=0.0):
    """Two independent (T, N) Gaussian noise tensors, std=gamma.

    Uses local torch.Generator instances so the global RNG (used by dropout,
    sampling, etc.) is left untouched.

    If rho > 0, each tensor is a stationary AR(1) chain with autocorrelation
    rho preserving marginal variance gamma**2:
        eta(t+1) = rho * eta(t) + sqrt(1 - rho**2) * gamma * xi(t),  xi ~ N(0,1).
    The two chains share NO innovations (different seeds -> independent
    processes), which is the only requirement of the Noise2Noise theorem.

    Cite: https://github.com/buchenglab/SPEND -- N2N requires independence
    between the two noise realisations; that holds regardless of rho.
    """
    g_a = torch.Generator(device=device).manual_seed(int(seed_a))
    g_b = torch.Generator(device=device).manual_seed(int(seed_b))
    if rho <= 0.0:
        n_a = torch.randn(*shape, generator=g_a, device=device) * gamma
        n_b = torch.randn(*shape, generator=g_b, device=device) * gamma
        return n_a, n_b
    T, N = shape
    inject = (1.0 - rho ** 2) ** 0.5 * gamma
    n_a = torch.zeros(T, N, device=device)
    n_b = torch.zeros(T, N, device=device)
    # init in stationary distribution: Var(eta_0) = gamma**2
    n_a[0] = torch.randn(N, generator=g_a, device=device) * gamma
    n_b[0] = torch.randn(N, generator=g_b, device=device) * gamma
    for t in range(1, T):
        n_a[t] = rho * n_a[t-1] + torch.randn(N, generator=g_a, device=device) * inject
        n_b[t] = rho * n_b[t-1] + torch.randn(N, generator=g_b, device=device) * inject
    return n_a, n_b


def _build_smoother(hidden, device):
    """Inline 1D-conv N2N smoother: 1 -> H -> H -> 1, kernel 5, reflect padding.

    ~10K params at hidden=32. The architecture mirrors SPEND's small-U-Net
    philosophy (CSBDeep U-Net depth=4, base=32) at 1D trace scale rather than
    full 3D images. Operates on (B, 1, T) tensors and returns (B, 1, T).
    Cite: github.com/buchenglab/SPEND -- internals/nets.py custom_unet.
    """
    return nn.Sequential(
        nn.Conv1d(1, hidden, kernel_size=5, padding=2, padding_mode='reflect'),
        nn.ReLU(inplace=True),
        nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, padding_mode='reflect'),
        nn.ReLU(inplace=True),
        nn.Conv1d(hidden, 1, kernel_size=5, padding=2, padding_mode='reflect'),
    ).to(device)


def _build_typed_pairs(type_list, pos, max_dist):
    """Precompute (i1, i2) tensor of same-type neuron pairs within position
    distance < max_dist. Runs once at setup; O(sum_c |type_c|^2).

    Returns long tensor of shape (P, 2). Each row is a (i1, i2) index pair
    where type_list[i1] == type_list[i2] and ||pos[i1] - pos[i2]|| < max_dist.
    Used to construct an N2N-style equivariance loss across columns.
    """
    type_flat = type_list.squeeze(-1).long().cpu()
    pos_cpu = pos.cpu() if pos.is_cuda else pos
    pairs = []
    for c in torch.unique(type_flat):
        ids_c = torch.where(type_flat == c)[0]
        if len(ids_c) < 2:
            continue
        p = pos_cpu[ids_c].float()
        d = torch.cdist(p, p)
        d.fill_diagonal_(float('inf'))
        nn_dist, nn_idx = d.min(dim=1)
        mask = nn_dist < max_dist
        if mask.any():
            i1 = ids_c[mask]
            i2 = ids_c[nn_idx[mask]]
            pairs.append(torch.stack([i1, i2], dim=1))
    if not pairs:
        return torch.empty(0, 2, dtype=torch.long)
    return torch.cat(pairs, dim=0).long()


# ============================================================================
# Main trainer
# ============================================================================

def data_train_spend(config, erase=False, best_model=None, device=None, log_file=None):
    """SPEND-style Noise2Noise GNN training for FlyVis under measurement noise.

    Mirrors graph_trainer.data_train_gnn but adds three N2N losses gated by
    their respective coefficients. Smoother is co-trained with the GNN.

    Cite: https://github.com/buchenglab/SPEND
    """
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    # --- Reproducibility ---
    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    default_style.apply_globally()

    log_dir, logger = create_log_dir(config, erase)

    # --- SPEND coefficients (extract once to plain Python scalars) ---
    coeff_spend_replay = float(getattr(tc, 'coeff_spend_replay', 0.0))
    coeff_spend_time = float(getattr(tc, 'coeff_spend_time', 0.0))
    coeff_spend_typed = float(getattr(tc, 'coeff_spend_typed', 0.0))
    spend_load_clean = bool(getattr(tc, 'spend_load_clean', False))
    spend_seed_a = int(getattr(tc, 'spend_replay_noise_seed_a', 0))
    spend_seed_b = int(getattr(tc, 'spend_replay_noise_seed_b', 1))
    spend_time_window = int(getattr(tc, 'spend_time_window', 16))
    spend_smoother_hidden = int(getattr(tc, 'spend_smoother_hidden', 32))
    spend_smoother_lr = float(getattr(tc, 'spend_smoother_lr', 1e-3))
    spend_typed_max_pos_dist = float(getattr(tc, 'spend_typed_max_pos_dist', 5.0))

    spend_active = (coeff_spend_replay > 0) or (coeff_spend_time > 0) or (coeff_spend_typed > 0)
    if not spend_active:
        _logger.warning('SPEND trainer invoked but all coeff_spend_* are 0 -- behaves like baseline GNN')
    _logger.info(
        f'SPEND coefficients: replay={coeff_spend_replay} time={coeff_spend_time} '
        f'typed={coeff_spend_typed} | load_clean={spend_load_clean}'
    )

    # --- Data load (with optional clean override for replay) ---
    load_fields = determine_load_fields(config)
    # When spend_load_clean is on, we want clean voltage and the *clean* y_list
    # (not noisy_y_list). Pass measurement_noise_level=0 to the loader so it
    # picks the clean y_list. Drop 'noise' from fields since we synth it.
    if spend_load_clean:
        if 'noise' in load_fields:
            load_fields = [f for f in load_fields if f != 'noise']
        loader_meas_noise = 0.0
        _logger.info('SPEND: loading CLEAN voltage and clean y_list; noise will be synthesised inline')
    else:
        loader_meas_noise = sim.measurement_noise_level
    # Typed-equiv needs retinotopic positions for pair construction; force-add
    # 'pos' to the load list since determine_load_fields only requests it for
    # visual/test field-types.
    if coeff_spend_typed > 0 and 'pos' not in load_fields:
        load_fields = list(load_fields) + ['pos']

    x_ts, y_ts, type_list = load_flyvis_data(
        config.dataset, split='train', fields=load_fields,
        training_selected_neurons=tc.training_selected_neurons,
        selected_neuron_ids=tc.selected_neuron_ids if tc.training_selected_neurons else None,
        measurement_noise_level=loader_meas_noise,
        observable=tc.observable,
    )

    n_neurons = x_ts.n_neurons
    config.simulation.n_neurons = n_neurons
    sim.n_frames = x_ts.n_frames
    n_frames = sim.n_frames
    _logger.info(f'dataset: {n_frames} frames, n neurons: {n_neurons}')
    logger.info(f'n neurons: {n_neurons}')

    assert not torch.isnan(x_ts.voltage).any(), 'voltage contains NaN'
    assert not np.isnan(y_ts).any(), 'derivative targets contain NaN'

    xnorm = float(x_ts.xnorm)
    torch.save(torch.tensor(xnorm), os.path.join(log_dir, 'xnorm.pt'))
    _logger.info(f'xnorm: {xnorm:0.3f}')

    x_ts = x_ts.to(device)
    y_ts_gpu = torch.from_numpy(y_ts).float().to(device)
    ynorm = 1.0
    torch.save(torch.tensor(ynorm), os.path.join(log_dir, 'ynorm.pt'))

    # --- SVD analysis (skip if exists) ---
    svd_plot_path = os.path.join(log_dir, 'results', 'svd_analysis.png')
    if not os.path.exists(svd_plot_path):
        analyze_data_svd(x_ts, log_dir, config=config, logger=logger, is_flyvis=True)

    # --- Load edges + GT weights (mirrors graph_trainer) ---
    from connectome_gnn.generators.ode_params import FlyVisODEParams, get_ode_params_class
    try:
        OdeParamsCls = get_ode_params_class(model_config.signal_model_name)
    except KeyError:
        OdeParamsCls = FlyVisODEParams
    ode_params = OdeParamsCls.load(graphs_data_path(config.dataset), device=device)
    gt_weights = ode_params.W
    edges = ode_params.edge_index
    if edges.shape[1] != sim.n_edges:
        _logger.info(f'n_edges override: config={sim.n_edges} -> actual={edges.shape[1]}')
        config.simulation.n_edges = edges.shape[1]
    torch.save(edges, os.path.join(log_dir, 'training_edges.pt'))
    torch.save(gt_weights, os.path.join(log_dir, 'gt_weights.pt'))

    # --- Build GNN ---
    checkpoint_path = None
    if best_model and best_model not in ('', 'None'):
        checkpoint_path = f'{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{best_model}.pt'
    elif tc.pretrained_model:
        checkpoint_path = tc.pretrained_model
    reset_epoch = (tc.pretrained_model != '' and not best_model)
    model, start_epoch = build_model(config, device, checkpoint_path=checkpoint_path, reset_epoch=reset_epoch)

    # --- SPEND setup: smoother, typed pairs, synthesised noise ---
    smoother = None
    if coeff_spend_replay > 0 or coeff_spend_time > 0:
        smoother = _build_smoother(spend_smoother_hidden, device)
        n_smoother_params = sum(p.numel() for p in smoother.parameters())
        _logger.info(f'SPEND smoother: 1D-conv, hidden={spend_smoother_hidden}, params={n_smoother_params:,}')

    typed_pairs = None
    if coeff_spend_typed > 0:
        if x_ts.pos is None:
            raise RuntimeError('coeff_spend_typed > 0 requires pos field; not found in x_ts')
        typed_pairs = _build_typed_pairs(type_list, x_ts.pos, spend_typed_max_pos_dist).to(device)
        _logger.info(
            f'SPEND typed-equiv pairs: {typed_pairs.shape[0]} pairs across {len(torch.unique(type_list))} types '
            f'(max_dist={spend_typed_max_pos_dist})'
        )

    # Synthesised replay noise: two (T, N) tensors. ~7 GB float32 at 64k x 13.7k.
    # If RAM is tight switch to per-window synthesis.
    noise_a = None
    noise_b = None
    if coeff_spend_replay > 0:
        gamma = float(sim.measurement_noise_level)
        if gamma <= 0:
            raise RuntimeError(
                'coeff_spend_replay > 0 requires sim.measurement_noise_level > 0 '
                '(replay synthesises noise of std=gamma)'
            )
        # Match the synth-noise AR(1) statistics to the dataset's
        # (sim.noise_ar1_rho) so the smoother is trained on the same
        # distribution it would see in a real experiment with this rho.
        ar1_rho = float(getattr(sim, 'noise_ar1_rho', 0.0))
        noise_a, noise_b = _synth_noise_pair(
            (n_frames, n_neurons), gamma,
            spend_seed_a, spend_seed_b, device,
            rho=ar1_rho,
        )
        _logger.info(
            f'SPEND replay: synthesised two noise tensors, std={gamma}, rho={ar1_rho}, '
            f'seeds=({spend_seed_a}, {spend_seed_b}), shape={tuple(noise_a.shape)}'
        )

    # --- Optimizer (single Adam, smoother as second param group) ---
    if tc.lr_update == 0:
        lr_update = tc.lr
    else:
        lr_update = tc.lr_update
    optimizer, n_total_params = set_trainable_parameters(
        model=model, lr_embedding=tc.lr_embedding, lr=tc.lr,
        lr_update=lr_update, lr_W=tc.lr_W, lr_NNR_f=tc.lr_NNR_f,
    )
    if smoother is not None:
        optimizer.add_param_group({'params': list(smoother.parameters()),
                                   'lr': spend_smoother_lr})
        _logger.info(f'SPEND smoother param group added: lr={spend_smoother_lr}')

    lr_scheduler = build_lr_scheduler(optimizer, config)

    # --- Regularizer (reused as-is; no torch.compile in SPEND trainer) ---
    regularizer = LossRegularizer(
        train_config=tc, model_config=model_config,
        activity_column=3, plot_frequency=1, n_neurons=n_neurons,
        trainer_type='flyvis', dataset=config.dataset,
        type_list=type_list, n_neuron_types=sim.n_neuron_types,
    )
    regularizer.set_activity_stats(x_ts, device)
    regularizer.move_type_list_to_device(device)
    _logger.info('SPEND trainer: torch.compile DISABLED (research path; pydantic-attr safety).')
    model.train()
    if smoother is not None:
        smoother.train()

    # --- Logging files ---
    # metrics.log uses the SAME schema as graph_trainer.py so the agentic-loop
    # HPO parser works unchanged. SPEND-specific component losses are written
    # to a sibling file `spend_components.log` and read explicitly by the
    # SPEND instruction files.
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    os.makedirs(os.path.dirname(metrics_log_path), exist_ok=True)
    with open(metrics_log_path, 'w') as f:
        f.write('iteration,connectivity_r2,vrest_r2,tau_r2,'
                'hidden_nnr_pearson,anchor_nnr_pearson,'
                'vrest_r2_clean,n_out_vrest,n_total_vrest,'
                'tau_r2_clean,n_out_tau,n_total_tau,loss\n')
    spend_log_path = os.path.join(log_dir, 'tmp_training', 'spend_components.log')
    with open(spend_log_path, 'w') as f:
        f.write('iteration,loss_main,loss_replay,loss_time,loss_typed\n')

    def _fmt(x):
        return 'nan' if x is None else f'{float(x):.6f}'

    # --- Frame range ---
    visible_ids = torch.arange(n_neurons, device=device)
    ids = visible_ids
    _frame_min_k = max(tc.time_window, spend_time_window)  # ensure window fits
    _frame_max_k = sim.n_frames - 4 - tc.time_step
    _frame_range = max(_frame_max_k - _frame_min_k, 1)

    loss_components = {'loss': []}
    # Most-recent (regul-subtracted, per-neuron) loss for metrics.log.
    last_loss = None
    training_start_time = time.time()

    last_connectivity_r2 = None
    last_tau_r2 = 0.0
    last_vrest_r2 = 0.0

    # EMA of SPEND per-iter component losses for the progress bar.
    # The raw per-batch values fluctuate ~±100% iter-to-iter (stochastic
    # minibatch); the EMA (alpha=0.05, ~20-iter window) is what humans
    # actually want to read. The full per-iter values are still written to
    # spend_components.log.
    ema_alpha = 0.05
    ema_main = None
    ema_replay = None
    ema_time = None
    ema_typed = None

    def _ema_update(prev, val):
        if val == 0.0 or val is None:
            return prev
        return val if prev is None else (1.0 - ema_alpha) * prev + ema_alpha * val

    # ======================================================================
    # Training loop. Mirrors graph_trainer inner GNN branch with SPEND adds.
    # ======================================================================
    for epoch in range(start_epoch, tc.n_epochs):

        Niter = int(sim.n_frames * tc.data_augmentation_loop // tc.batch_size * 0.2)
        plot_frequency = max(1, int(Niter // 20))
        connectivity_plot_frequency = max(1, int(Niter // 10))
        early_r2_frequency = max(1, connectivity_plot_frequency // 5)
        print(f'every {connectivity_plot_frequency} iterations: {Niter} iterations per epoch')

        if tc.max_iterations_per_epoch > 0:
            Niter = min(Niter, tc.max_iterations_per_epoch)

        regularizer.set_epoch(epoch, plot_frequency, Niter=Niter)

        epoch_rng = np.random.RandomState((tc.seed + epoch) % (2**32))
        frame_indices = epoch_rng.randint(0, _frame_range, size=Niter * tc.batch_size) + _frame_min_k

        pbar = trange(Niter, ncols=150)
        for N in pbar:
            optimizer.zero_grad()

            state_batch = []
            y_list = []
            ids_list = []
            ids_index = 0
            loss = torch.zeros((), device=device)
            regularizer.reset_iteration(device=device)

            # Per-iteration component scalars (CPU floats for logging)
            li_main = 0.0
            li_replay = 0.0
            li_time = 0.0
            li_typed = 0.0

            for batch in range(tc.batch_size):
                k = int(frame_indices[N * tc.batch_size + batch])
                x = x_ts.frame(k)

                # SPEND: choose voltage view fed to the GNN.
                # - replay on:  use clean + noise_a (or smoother(clean + noise_a))
                # - replay off: use observed (already includes noise) -- standard path
                if coeff_spend_replay > 0:
                    v_a = x.voltage + noise_a[k]            # (N,) view a
                    v_b = x.voltage + noise_b[k]            # (N,) view b -- N2N target
                    # Smoother input is a window centred on k: (1, 1, T_w)
                    win = spend_time_window
                    half = win // 2
                    k0 = max(0, k - half)
                    k1 = min(n_frames, k0 + win)
                    k0 = max(0, k1 - win)
                    v_a_win = (x_ts.voltage[k0:k1] + noise_a[k0:k1])  # (T_w, N)
                    v_b_win = (x_ts.voltage[k0:k1] + noise_b[k0:k1])  # (T_w, N) target
                    # Treat each neuron as an independent 1D channel: (N, 1, T_w)
                    inp = v_a_win.T.unsqueeze(1)
                    tgt = v_b_win.T.unsqueeze(1)
                    smoothed = smoother(inp)               # (N, 1, T_w)
                    li_replay_t = F.mse_loss(smoothed, tgt)
                    li_replay = float(li_replay_t.detach())
                    loss = loss + coeff_spend_replay * li_replay_t
                    # Replace GNN input with smoothed centre-frame
                    centre = k - k0
                    x.voltage = smoothed[:, 0, centre]
                elif x.noise is not None and sim.measurement_noise_level > 0:
                    # Standard noisy path (matches graph_trainer.py:663-664)
                    x.voltage = x.voltage + x.noise

                # Add-on #1 -- time-permutation N2N (analog of SPEND
                # Img_Split_Conc with stride-2 along the time axis). Even-frame
                # trace -> smoothed, target = linear interp of adjacent odd
                # frames at even-frame timestamps. Linear interp is justified
                # by sim.delta_t = 0.02 (20 ms) << dynamics correlation time.
                # Cite: github.com/buchenglab/SPEND -- datasplit_with_aug_choose
                # Img_Split_Conc.
                if coeff_spend_time > 0:
                    win = spend_time_window
                    half = win // 2
                    k0 = max(0, k - half)
                    k1 = min(n_frames, k0 + win)
                    k0 = max(0, k1 - win)
                    if (k1 - k0) >= 4 and (k1 - k0) % 2 == 0:
                        v_obs_win = x_ts.voltage[k0:k1].clone()      # (T_w, N)
                        if coeff_spend_replay > 0:
                            v_obs_win = v_obs_win + noise_a[k0:k1]
                        elif x_ts.noise is not None and sim.measurement_noise_level > 0:
                            v_obs_win = v_obs_win + x_ts.noise[k0:k1]
                        even = v_obs_win[::2]                        # (T_w/2, N)
                        odd = v_obs_win[1::2]                        # (T_w/2, N)
                        if odd.shape[0] >= 2:
                            inp_t = even.T.unsqueeze(1)              # (N, 1, T_w/2)
                            smoothed_t = smoother(inp_t)             # (N, 1, T_w/2)
                            odd_interp = 0.5 * (odd[:-1] + odd[1:])  # (T_w/2 - 1, N)
                            # Compare smoother output at the first T_w/2 - 1
                            # even-frame positions to the interpolated odd targets.
                            pred_time = smoothed_t[:, 0, : odd_interp.shape[0]].T  # (T_w/2 - 1, N)
                            li_time_t = F.mse_loss(pred_time, odd_interp)
                            li_time = float(li_time_t.detach())
                            loss = loss + coeff_spend_time * li_time_t

                # Add-on #2 -- typed-equivariance (noise-cancelled estimator).
                if coeff_spend_typed > 0 and typed_pairs is not None and typed_pairs.shape[0] > 0:
                    v_pair = x.voltage[typed_pairs]                  # (P, 2)
                    diff = v_pair[:, 0] - v_pair[:, 1]
                    gamma2 = float(sim.measurement_noise_level) ** 2
                    li_typed_t = (diff.pow(2).mean() - 2.0 * gamma2).clamp(min=0.0)
                    li_typed = float(li_typed_t.detach())
                    loss = loss + coeff_spend_typed * li_typed_t

                # Standard regularization (only on first batch element)
                if batch == 0:
                    regul_loss = regularizer.compute(
                        model=model, x=x, in_features=None, ids=ids,
                        ids_batch=None, edges=edges, device=device, xnorm=xnorm,
                    )
                    loss = loss + regul_loss

                # Target derivative
                y = y_ts_gpu[k] / ynorm

                state_batch.append(x)
                y_list.append(y)
                ids_list.append(visible_ids + ids_index)
                ids_index += x.n_neurons

            # GNN forward + main MSE on dv/dt
            data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)
            y_batch = torch.cat(y_list, dim=0)
            ids_batch = torch.cat(ids_list, dim=0)
            batched_state, batched_edges = _batch_frames(state_batch, edges)
            pred, in_features, msg = model(batched_state, batched_edges,
                                            data_id=data_id, return_all=True)
            update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)
            loss = loss + update_regul

            main_term = (pred[ids_batch] - y_batch[ids_batch]).norm(2)
            li_main = float(main_term.detach())
            loss = loss + main_term

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            regularizer.finalize_iteration()

            # Update EMAs of SPEND component losses for the progress bar.
            ema_main = _ema_update(ema_main, li_main)
            ema_replay = _ema_update(ema_replay, li_replay)
            ema_time = _ema_update(ema_time, li_time)
            ema_typed = _ema_update(ema_typed, li_typed)

            # --- per-iteration logging ---
            if regularizer.should_record():
                current_loss = float(loss.detach())
                regul_total = regularizer.get_iteration_total()
                loss_components['loss'].append((current_loss - regul_total) / n_neurons)
                last_loss = loss_components['loss'][-1]
                plot_dict = {**regularizer.get_history(), 'loss': loss_components['loss']}
                plot_signal_loss(
                    plot_dict, log_dir, epoch=epoch, Niter=Niter,
                    epoch_boundaries=regularizer.epoch_boundaries, debug=False,
                    current_loss=current_loss / n_neurons,
                    current_regul=regul_total / n_neurons,
                    total_loss=current_loss, total_loss_regul=regul_total,
                )
                # SPEND-only scalars (read by the SPEND HPO instruction files).
                with open(spend_log_path, 'a') as f:
                    f.write(f'{regularizer.iter_count},{li_main:.6f},'
                            f'{li_replay:.6f},{li_time:.6f},{li_typed:.6f}\n')

            # R2 checkpoint (matches graph_trainer cadence + schema)
            is_regular_r2 = (N > 0) and (N % connectivity_plot_frequency == 0)
            is_early_r2 = (N < connectivity_plot_frequency) and (N % early_r2_frequency == 0)
            if is_regular_r2 or is_early_r2:
                last_connectivity_r2, _r2_visible, _h_r2, _a_r2 = plot_training_flyvis(
                    x_ts, model, config, epoch, N, log_dir, device, type_list,
                    gt_weights, edges, n_neurons=n_neurons,
                    n_neuron_types=sim.n_neuron_types, ode_params=ode_params,
                    hidden_ids=None, anchor_ids=None,
                )
                _dyn = compute_dynamics_r2(model, x_ts, config, device, n_neurons)
                last_vrest_r2 = _dyn['vrest_r2']
                last_tau_r2   = _dyn['tau_r2']
                # Schema matches graph_trainer.py extended layout; SPEND has no
                # hidden/anchor neurons so those slots are 'nan'.
                with open(metrics_log_path, 'a') as f:
                    f.write(f'{regularizer.iter_count},{_fmt(last_connectivity_r2)},'
                            f'{_fmt(last_vrest_r2)},{_fmt(last_tau_r2)},'
                            f'nan,nan,'
                            f'{_fmt(_dyn["vrest_r2_clean"])},{_dyn["n_out_vrest"]},{_dyn["n_total_vrest"]},'
                            f'{_fmt(_dyn["tau_r2_clean"])},{_dyn["n_out_tau"]},{_dyn["n_total_tau"]},{_fmt(last_loss)}\n')

            # progress bar -- conn, Vr, tau in colored R2 style; SPEND component
            # losses shown as EMAs (raw per-batch values fluctuate ~+/-100%).
            if last_connectivity_r2 is not None:
                bar = [
                    f'{_r2c(last_connectivity_r2)}conn={last_connectivity_r2:.3f}{ANSI_RESET}',
                    f'{_r2c(last_vrest_r2)}Vr={last_vrest_r2:.3f}{ANSI_RESET}',
                    f'{_r2c(last_tau_r2)}tau={last_tau_r2:.3f}{ANSI_RESET}',
                ]
                if coeff_spend_replay > 0 and ema_replay is not None:
                    bar.append(f'rep={ema_replay:.4f}')
                if coeff_spend_time > 0 and ema_time is not None:
                    bar.append(f'tim={ema_time:.4f}')
                if coeff_spend_typed > 0 and ema_typed is not None:
                    bar.append(f'typ={ema_typed:.4f}')
                pbar.set_postfix_str(' '.join(bar))

        # End of epoch -- save checkpoint with smoother (data_test_gnn loads
        # with strict=False so the extra 'smoother' key is ignored at eval).
        ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if smoother is not None:
            ckpt['smoother_state_dict'] = smoother.state_dict()
        ckpt_path = os.path.join(log_dir, 'models',
                                 f'best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt')
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(ckpt, ckpt_path)
        _logger.info(f'epoch {epoch}: saved {ckpt_path}')
        check_and_clear_memory(device=device, iteration_number=N,
                                every_n_iterations=1, memory_percentage_threshold=0.6)

    elapsed = (time.time() - training_start_time) / 60.0
    _logger.info(f'SPEND training complete: {elapsed:.1f} min')
    return model
