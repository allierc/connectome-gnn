"""Shared utilities for GNN training and testing.

Extracted from graph_trainer.py to eliminate duplication of data loading,
model construction, checkpoint management, and optimizer setup across
data_train_gnn, data_test_gnn, and data_test_gnn_special.
"""

import glob
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from connectome_gnn.models.registry import create_model
from connectome_gnn.models.neural_ode_wrapper import neural_ode_loss
from connectome_gnn.models.utils import _batch_frames, _quick_ngp_pearson, fit_residual_loss, set_trainable_parameters
from connectome_gnn.utils import graphs_data_path, migrate_state_dict, sort_key
from connectome_gnn.zarr_io import load_raw_array, load_simulation_data
from dataclasses import dataclass, field
from connectome_gnn.models.regularizer import LossRegularizer

@dataclass
class TrainingMetrics:
    """
    Latest diagnostic values.

    These are updated at R2/HH/NGP evaluation checkpoints.
    They do not affect the training loss.
    """

    connectivity_r2: float | None = None
    connectivity_r2_visible: float | None = None

    vrest_r2: float = 0.0
    tau_r2: float = 0.0

    vrest_r2_clean: float = float("nan")
    tau_r2_clean: float = float("nan")

    n_out_vrest: int = 0
    n_total_vrest: int = 0

    n_out_tau: int = 0
    n_total_tau: int = 0

    hidden_r2: float | None = None
    anchor_r2: float | None = None

    field_r2: float | None = None
    field_slope: float | None = None


@dataclass
class HiddenInjectionSchedule:
    """
    NGP hidden-voltage injection and GNN LR-damping schedule.

    warmup_iter:
        Iteration at which hidden-voltage injection switches on.

    ramp_iter:
        Duration of each side of the LR-damping V.

    damping_factor:
        Depth of the V. For example 100 means the trough is base_lr / 100.

    damping_active:
        Whether the V-shaped LR schedule is actually active.

    damp_groups:
        Optimizer groups affected by the damping schedule.

    stages:
        Boundaries used by metrics.png.
    """

    warmup_iter: int
    ramp_iter: int
    damping_factor: float

    damping_active: bool
    damp_groups: tuple[str, ...]

    ramp_mid: int = 0
    ramp_end: int = 0

    stages: list = field(default_factory=list)


@dataclass
class FrameSampling:
    """
    Valid frame range for training-frame sampling.
    """

    first_frame: int
    last_frame: int
    frame_range: int


@dataclass
class EpochState:
    """
    State initialized once at the beginning of each training epoch.

    This contains values that:
      - depend on the current epoch, or
      - reset at every epoch.

    It does not contain persistent training state such as the model,
    optimizer, regularizer, or hidden-injection schedule.
    """

    # Iteration / plotting schedule
    n_iter: int
    plot_frequency: int
    connectivity_plot_frequency: int
    early_r2_frequency: int
    plot_iterations: set[int]

    # Data sampling
    frame_indices: np.ndarray

    # Loss
    loss_noise_level: float
    total_loss_gpu: torch.Tensor
    total_regul_gpu: torch.Tensor

    # Diagnostics
    metrics: TrainingMetrics

    # Dale's law
    dale_enabled: bool
    dale_checkpoints: set[int]

    # Embedding / UMAP state
    unfreeze_at_iteration: int


def init_epoch_state(
    epoch,
    n_iter,
    training,
    frame_sampling,
    embedding_frozen,
    regularizer,
    device,
):
    """
    Initialize all state local to one training epoch.
    """

    # ---------------------------------------------------------------------
    # Plot / diagnostic frequencies
    # ---------------------------------------------------------------------

    plot_frequency = max(
        1,
        n_iter // 20,
    )

    connectivity_plot_frequency = max(
        1,
        n_iter // 20,
    )

    # Four extra R2 evaluations during the early part of the epoch.
    early_r2_frequency = max(
        1,
        connectivity_plot_frequency // 5,
    )

    # Visual-field / heavy plot locations.
    n_plots_per_epoch = 4

    if n_plots_per_epoch > 0:
        plot_iterations = {
            int(iteration)
            for iteration in np.linspace(
                n_iter // n_plots_per_epoch,
                n_iter - 1,
                n_plots_per_epoch,
            )
        }
    else:
        plot_iterations = set()

    # ---------------------------------------------------------------------
    # Embedding unfreeze
    # ---------------------------------------------------------------------

    if (
        embedding_frozen
        and training.umap_cluster_fix_embedding_ratio > 0
    ):
        unfreeze_at_iteration = int(
            n_iter
            * training.umap_cluster_fix_embedding_ratio
        )
    else:
        unfreeze_at_iteration = -1

    # ---------------------------------------------------------------------
    # Reproducible frame sampling
    # ---------------------------------------------------------------------

    epoch_rng = np.random.RandomState(
        (training.seed + epoch) % (2**32)
    )

    frame_indices = (
        epoch_rng.randint(
            0,
            frame_sampling.frame_range,
            size=n_iter * training.batch_size,
        )
        + frame_sampling.first_frame
    )

    # ---------------------------------------------------------------------
    # Loss noise
    # ---------------------------------------------------------------------

    loss_noise_level = (
        training.loss_noise_level
        * (0.95 ** epoch)
    )

    # ---------------------------------------------------------------------
    # Dale's law
    # ---------------------------------------------------------------------

    dale_enabled = getattr(
        training,
        "dale_law",
        False,
    )

    if dale_enabled:
        dale_checkpoints = {
            int(n_iter * fraction)
            for fraction in (
                0.25,
                0.50,
                0.75,
            )
        }

        dale_checkpoints.discard(0)

    else:
        dale_checkpoints = set()

    # ---------------------------------------------------------------------
    # Regularizer
    # ---------------------------------------------------------------------

    regularizer.set_epoch(
        epoch,
        plot_frequency,
        Niter=n_iter,
    )

    # ---------------------------------------------------------------------
    # Epoch-local GPU accumulators
    #
    # Keep these on the GPU during the entire epoch to avoid .item()
    # synchronization at every iteration.
    # ---------------------------------------------------------------------

    total_loss_gpu = torch.zeros(
        (),
        device=device,
    )

    total_regul_gpu = torch.zeros(
        (),
        device=device,
    )

    # ---------------------------------------------------------------------
    # Diagnostic state
    # ---------------------------------------------------------------------

    metrics = TrainingMetrics()

    # ---------------------------------------------------------------------
    # Return complete epoch state
    # ---------------------------------------------------------------------

    return EpochState(
        n_iter=n_iter,

        plot_frequency=plot_frequency,
        connectivity_plot_frequency=connectivity_plot_frequency,
        early_r2_frequency=early_r2_frequency,
        plot_iterations=plot_iterations,

        frame_indices=frame_indices,

        loss_noise_level=loss_noise_level,
        total_loss_gpu=total_loss_gpu,
        total_regul_gpu=total_regul_gpu,

        metrics=metrics,

        dale_enabled=dale_enabled,
        dale_checkpoints=dale_checkpoints,

        unfreeze_at_iteration=unfreeze_at_iteration,
    )

def get_training_frame_sampling(sim, training, target_offset=None):
    """
    Determine the valid starting-frame range.

    The training loop samples k such that:
      - enough history exists for time_window
      - enough future data exists for time_step / recurrent training
      - the same bounds as the legacy np.random.randint logic are preserved.

    target_offset: how many frames past k the loss needs. None (default) derives
    it from time_step as before. The rollout-horizon curriculum passes the MAXIMUM
    horizon explicitly, so the sampled-k distribution stays identical across
    epochs and the curriculum is the only thing that varies.
    """

    first_frame = training.time_window

    stride_subsample = (
        training.recurrent_training
        and training.time_step > 1
    )

    # An explicit config pin wins over both the caller's argument and the derived
    # value, so every arm of a comparison can be made to sample identically.
    _pin = getattr(training, 'frame_target_offset', 0)
    if _pin and _pin > 0:
        target_offset = _pin
    elif target_offset is None:
        target_offset = (
            1
            if stride_subsample
            else training.time_step
        )

    last_frame = (
        sim.n_frames
        - 4
        - target_offset
    )

    frame_range = max(
        last_frame - first_frame,
        1,
    )

    return FrameSampling(
        first_frame=first_frame,
        last_frame=last_frame,
        frame_range=frame_range,
    )


def init_metrics_files(log_dir):
    """
    Initialize the metric log files used by the training monitor.

    All of them are truncated here, including g_phi_discard.log, which the
    training loop opens in APPEND mode. Without truncation, re-running into an
    existing log dir (after a crash, or after LSF kills a job on its RUNLIMIT)
    silently interleaves the dead run's rows with the new run's at overlapping
    iteration numbers -- and unlike metrics.log there is no header row to make
    the seam visible.
    """

    g_phi_discard_log_path = os.path.join(
        log_dir,
        "tmp_training",
        "g_phi_discard.log",
    )
    if os.path.exists(g_phi_discard_log_path):
        os.remove(g_phi_discard_log_path)

    metrics_log_path = os.path.join(
        log_dir,
        "tmp_training",
        "metrics.log",
    )

    with open(
        metrics_log_path,
        "w",
    ) as f:
        f.write(
            "iteration,"
            "connectivity_r2,"
            # RAW, i.e. over every neuron including the outliers. A handful of
            # neurons with a near-zero fitted slope send these to -30 or worse, so
            # they are NOT the numbers metrics.png shows and NOT the paper
            # convention. The comparable values are vrest_r2_clean / tau_r2_clean
            # below, with their outlier counts. Named _raw so that reading the
            # obvious column no longer looks like a broken run.
            "vrest_r2_raw,"
            "tau_r2_raw,"
            "hidden_nnr_pearson,"
            "anchor_nnr_pearson,"
            "vrest_r2_clean,"
            "n_out_vrest,"
            "n_total_vrest,"
            "tau_r2_clean,"
            "n_out_tau,"
            "n_total_tau\n"
        )

    nnr_pearson_log_path = os.path.join(
        log_dir,
        "tmp_training",
        "nnr_pearson.log",
    )

    with open(
        nnr_pearson_log_path,
        "w",
    ) as f:
        f.write(
            "iteration,"
            "hidden_pearson_mean,"
            "hidden_pearson_std,"
            "anchor_pearson_mean,"
            "anchor_pearson_std\n"
        )

    return (
        metrics_log_path,
        nnr_pearson_log_path,
    )


def init_training_runtime(
    log_dir,
    sim,
    training,
):
    """
    Initialize runtime state that is independent of a particular epoch.

    This replaces the loose block of:
        metrics logs
        total_iter.txt
        frame range
        embedding freeze state
        profiler directory
    """

    metrics_log_path, nnr_pearson_log_path = (
        init_metrics_files(log_dir)
    )

    # This is the same formula used at epoch start.
    n_iter_per_epoch = int(
        sim.n_frames
        * training.data_augmentation_loop
        // training.batch_size
        * 0.2
    )

    if training.max_iterations_per_epoch > 0:
        n_iter_per_epoch = min(
            n_iter_per_epoch,
            training.max_iterations_per_epoch,
        )

    total_iterations = (
        n_iter_per_epoch
        * training.n_epochs
    )

    with open(
        os.path.join(
            log_dir,
            "tmp_training",
            "total_iter.txt",
        ),
        "w",
    ) as f:
        f.write(
            str(total_iterations)
        )

    frame_sampling = (
        get_training_frame_sampling(
            sim,
            training,
        )
    )

    profiler_trace_dir = os.path.join(
        log_dir,
        "profiler_traces",
    )

    if training.profiling:
        os.makedirs(
            profiler_trace_dir,
            exist_ok=True,
        )

    return (
        metrics_log_path,
        nnr_pearson_log_path,
        frame_sampling,
        profiler_trace_dir,
    )


def init_hidden_injection_schedule(training, Niter):
    """
    Build the NGP injection + GNN LR-damping schedule.

    This preserves the existing warmup/ramp semantics.
    """

    warmup_fraction = float(
        getattr(
            training,
            "warmup_inject_nnr_iter_frac",
            0.0,
        )
    )

    ramp_fraction = float(
        getattr(
            training,
            "warmup_inject_nnr_ramp_iter_frac",
            0.0,
        )
    )

    if warmup_fraction > 0.0:
        warmup_iter = int(
            Niter * warmup_fraction
        )
    else:
        warmup_iter = int(
            getattr(
                training,
                "warmup_inject_nnr_iter",
                0,
            )
        )

    if ramp_fraction > 0.0:
        ramp_iter = int(
            Niter * ramp_fraction
        )
    else:
        ramp_iter = int(
            getattr(
                training,
                "warmup_inject_nnr_ramp_iter",
                0,
            )
        )

    damping_factor = float(
        getattr(
            training,
            "lr_damping_factor",
            100.0,
        )
    )

    damping_active = (
        warmup_iter > 0
        and ramp_iter > 0
        and damping_factor > 1.0
    )

    damp_groups = (
        "W",
        "f_theta",
        "g_phi",
    )

    ramp_mid = (
        warmup_iter + ramp_iter
    )

    ramp_end = (
        warmup_iter
        + 2 * ramp_iter
    )

    stages = []

    if warmup_iter > 0:
        stages.append(
            (
                warmup_iter,
                "inject",
            )
        )

        if damping_active:
            stages.append(
                (
                    ramp_mid,
                    "trough",
                )
            )

            stages.append(
                (
                    ramp_end,
                    "recover",
                )
            )

    return HiddenInjectionSchedule(
        warmup_iter=warmup_iter,
        ramp_iter=ramp_iter,
        damping_factor=damping_factor,
        damping_active=damping_active,
        damp_groups=damp_groups,
        ramp_mid=ramp_mid,
        ramp_end=ramp_end,
        stages=stages,
    )


def apply_lr_damping(hs, optimizer, base_lrs, N, previous_lr_multiplier, previous_injection_active):
    """Advance the hidden-injection schedule by one iteration.

    Computes whether hidden-voltage injection is active at iteration N,
    updates the damped optimizer param-group LRs when the multiplier
    changes, and prints the warmup->inject / damp->recover / recover->nominal
    stage transitions.

    Returns:
        injection_active, lr_multiplier
    """
    injection_active = hs.warmup_iter <= 0 or N >= hs.warmup_iter

    if not hs.damping_active:
        lr_multiplier = 1.0
    elif N < hs.warmup_iter or N >= hs.ramp_end:
        lr_multiplier = 1.0
    elif N < hs.ramp_mid:
        progress = float(N - hs.warmup_iter) / float(hs.ramp_iter)
        lr_multiplier = 1.0 + (1.0 / hs.damping_factor - 1.0) * progress
    else:
        progress = float(N - hs.ramp_mid) / float(hs.ramp_iter)
        lr_multiplier = 1.0 / hs.damping_factor + (1.0 - 1.0 / hs.damping_factor) * progress

    if hs.damping_active and lr_multiplier != previous_lr_multiplier:
        for param_group in optimizer.param_groups:
            if param_group.get("name") in hs.damp_groups:
                param_group["lr"] = base_lrs[id(param_group)] * lr_multiplier

    if hs.warmup_iter > 0 and previous_injection_active is False and injection_active:
        if hs.damping_active:
            print(f"\n[NGP inject] iter {N}: phase 1 -> phase 2 (NGP hard-on; GNN-LR V-schedule starts).")
        else:
            print(f"\n[NGP inject] iter {N}: phase 1 -> phase 2 (NGP hard-on).")
    elif hs.damping_active and N == hs.ramp_mid and hs.warmup_iter > 0:
        print(f"\n[NGP inject] iter {N}: LR damping -> recovery.")
    elif hs.damping_active and N == hs.ramp_end and hs.warmup_iter > 0:
        print(f"\n[NGP inject] iter {N}: GNN LR back to nominal.")

    return injection_active, lr_multiplier


def format_metric(value):
    """
    Format an optional metric for CSV logging.
    """
    return (
        "nan"
        if value is None
        else f"{value:.6f}"
    )


def determine_load_fields(config):
    """Determine which NeuronTimeSeries fields to load based on config.

    Returns:
        list of field name strings for load_simulation_data().
    """
    model_config = config.graph_model
    sim = config.simulation
    fields = ['voltage', 'stimulus', 'neuron_type']
    if 'visual' in model_config.field_type or 'test' in model_config.field_type:
        fields.append('pos')
    # Hidden-neuron INR variants that read per-neuron positions:
    #   - SIREN(x, y, t)
    #   - NGP-T with the spatial branch (ngp_hidden_spatial=True)
    inr_hidden = getattr(model_config, 'inr_type_hidden', 'none')
    needs_pos_for_inr = (
        inr_hidden == 'siren_txy'
        or (inr_hidden == 'ngp_t'
            and bool(getattr(model_config, 'ngp_hidden_spatial', False)))
    )
    if needs_pos_for_inr and 'pos' not in fields:
        fields.append('pos')
    if sim.calcium_type != 'none':
        fields.append('calcium')
    if sim.measurement_noise_level > 0:
        fields.append('noise')
    # Datasets generated under optogenetic perturbation carry an additional
    # optogenetics_stimulus.zarr that the trainer / tester must load so the
    # forward pass can sum it into the excitation channel.
    opto_enabled = bool(getattr(getattr(sim, 'optogenetics', None), 'enabled', False))
    if opto_enabled:
        fields.append('optogenetics_stimulus')
    return fields


def load_flyvis_data(dataset_name, split='train', fields=None,
                     training_selected_neurons=False, selected_neuron_ids=None,
                     measurement_noise_level=0.0):
    """Load NeuronTimeSeries + derivative targets for a given split.

    Data is returned on CPU. Callers are responsible for moving to GPU
    (this avoids OOM when computing derived quantities like xnorm that
    need temporary memory proportional to the voltage tensor).

    Args:
        dataset_name: dataset identifier (e.g. 'fly/flyvis_noise_005')
        split: 'train' or 'test'
        fields: list of field names to load (from determine_load_fields)
        training_selected_neurons: if True, subset neurons
        selected_neuron_ids: list of neuron indices to keep
        measurement_noise_level: if > 0, load noisy_y_list instead of y_list

    Returns:
        x_ts: NeuronTimeSeries on CPU
        y_ts: numpy array of derivative targets, shape (T, N, 1)
        type_list: (N, 1) float tensor of neuron type labels (CPU)
    """
    split_name = f'x_list_{split}'
    path = graphs_data_path(dataset_name, split_name)

    # Choose derivative target: noisy or clean
    y_prefix = 'noisy_y_list' if measurement_noise_level > 0 else 'y_list'

    if os.path.exists(path):
        x_ts = load_simulation_data(path, fields=fields)
        y_ts = load_raw_array(graphs_data_path(dataset_name, f'{y_prefix}_{split}'))
    else:
        print(f"warning: {split_name} not found, falling back to x_list_0")
        x_ts = load_simulation_data(
            graphs_data_path(dataset_name, 'x_list_0'), fields=fields
        )
        y_ts = load_raw_array(graphs_data_path(dataset_name, 'y_list_0'))

    # Extract type_list, then construct index (not loaded from disk)
    type_list = x_ts.neuron_type.float().unsqueeze(-1)
    x_ts.neuron_type = None
    x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long)

    if training_selected_neurons and selected_neuron_ids is not None:
        selected = np.array(selected_neuron_ids).astype(int)
        x_ts = x_ts.subset_neurons(selected)
        y_ts = y_ts[:, selected, :]
        type_list = type_list[selected]

    return x_ts, y_ts, type_list


def build_model(config, device, checkpoint_path=None, reset_epoch=False):
    """Create a NeuralGNN model and optionally load a checkpoint.

    Args:
        config: NeuralGraphConfig
        device: torch device
        checkpoint_path: path to .pt checkpoint file (or None)

    Returns:
        model: NeuralGNN on device
        start_epoch: int, 0 unless resumed from a checkpoint with epoch in filename
    """
    model_config = config.graph_model
    model = create_model(
        model_config.signal_model_name,
        aggr_type=model_config.aggr_type,
        config=config, device=device,
    ).to(device)

    # Resolve relative ./log/... paths against data_root
    if checkpoint_path and not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        from connectome_gnn.utils import get_data_root
        resolved = os.path.join(get_data_root(), checkpoint_path.lstrip('./'))
        if os.path.exists(resolved):
            checkpoint_path = resolved

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f'loading state_dict from {checkpoint_path} ...')
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        migrate_state_dict(state_dict)
        model.load_state_dict(state_dict['model_state_dict'], strict=False)

        # Try to extract epoch from filename (e.g. best_model_with_1_graphs_5.pt → epoch 5)
        basename = os.path.basename(checkpoint_path)
        name_no_ext = basename.replace('.pt', '')
        parts = name_no_ext.split('_')
        try:
            start_epoch = int(parts[-1])
        except (ValueError, IndexError):
            pass
        if reset_epoch:
            start_epoch = 0
        print(f'state_dict loaded, start_epoch={start_epoch}')
    else:
        if checkpoint_path:
            print(f'checkpoint not found: {checkpoint_path} — using freshly initialized model')
        else:
            print('no state_dict loaded — using freshly initialized model')

    return model, start_epoch


def find_latest_epoch_checkpoint(log_dir, n_runs):
    """Locate the highest-numbered completed per-epoch checkpoint.

    Per-epoch checkpoints are written as
    ``best_model_with_{n_runs-1}_graphs_{E}.pt`` (E = 0-based epoch index) at
    the end of each epoch. Used by ``--resume`` to continue from the next epoch.

    Returns:
        (path, epoch_index) for the largest E, or (None, -1) if none exist.
        The plain ``best_model_with_{n_runs-1}_graphs.pt`` (no epoch suffix) is
        skipped — only epoch-numbered checkpoints count as completed epochs.
    """
    prefix = f'best_model_with_{n_runs - 1}_graphs_'
    pattern = os.path.join(log_dir, 'models', f'{prefix}*.pt')
    best_path, best_epoch = None, -1
    for path in glob.glob(pattern):
        stem = os.path.basename(path)[:-len('.pt')]
        suffix = stem[len(prefix):]
        # Require a pure-digit suffix. NB int('2_3400') == 23400 in Python
        # (underscore digit-grouping), so a mid-epoch "..._{E}_{N}.pt" name
        # would otherwise be misread as a giant epoch — isdigit() rejects it.
        if not suffix.isdigit():
            continue
        epoch = int(suffix)
        if epoch > best_epoch:
            best_path, best_epoch = path, epoch
    return best_path, best_epoch


def planned_total_updates(config):
    """Total optimizer steps the run will take, from config alone.

    Mirrors the trainer's own arithmetic (graph_trainer.py: Niter, then Niter//K
    per epoch under the rollout curriculum), so the LR schedule can be laid out
    before training starts without threading a counter through.
    """
    tc, sim = config.training, config.simulation
    niter = int(sim.n_frames * tc.data_augmentation_loop // tc.batch_size * 0.2)
    if getattr(tc, "max_iterations_per_epoch", 0) > 0:
        niter = min(niter, tc.max_iterations_per_epoch)
    K = list(getattr(tc, "rollout_horizon_schedule", []) or []) or [1] * tc.n_epochs
    K = (K + [K[-1]] * tc.n_epochs)[: tc.n_epochs]
    tail = getattr(tc, "rollout_tail_iters_per_epoch", 0)
    def _n(k):
        n = max(1, niter // k)
        return min(n, tail) if (tail and tail > 0 and k > 1) else n
    return sum(_n(k) for k in K)


def build_lr_scheduler(optimizer, config):
    """Build LR scheduler from config.

    Supports:
        'none': constant LR (no-op scheduler)
        'cosine_warm_restarts': CosineAnnealingWarmRestarts per iteration
        'linear_warmup_cosine': linear warmup then cosine decay

    New config fields (all with backwards-compatible defaults):
        training.lr_scheduler: str = 'none'
        training.lr_scheduler_T0: int = 1000
        training.lr_scheduler_T_mult: int = 2
        training.lr_scheduler_eta_min_ratio: float = 0.01
        training.lr_scheduler_warmup_iters: int = 100

    Returns:
        LR scheduler instance
    """
    tc = config.training
    scheduler_type = getattr(tc, 'lr_scheduler', 'none')

    if scheduler_type == 'none':
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    elif scheduler_type == 'cosine_warm_restarts':
        T_0 = getattr(tc, 'lr_scheduler_T0', 1000)
        T_mult = getattr(tc, 'lr_scheduler_T_mult', 2)
        eta_min_ratio = getattr(tc, 'lr_scheduler_eta_min_ratio', 0.01)

        # Compute per-group eta_min from initial lr
        eta_min = min(pg['lr'] for pg in optimizer.param_groups) * eta_min_ratio

        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    elif scheduler_type == 'linear_warmup_cosine':
        warmup_iters = getattr(tc, 'lr_scheduler_warmup_iters', 100)
        T_0 = getattr(tc, 'lr_scheduler_T0', 1000)
        T_mult = getattr(tc, 'lr_scheduler_T_mult', 2)
        eta_min_ratio = getattr(tc, 'lr_scheduler_eta_min_ratio', 0.01)
        eta_min = min(pg['lr'] for pg in optimizer.param_groups) * eta_min_ratio

        def lr_lambda(step):
            if step < warmup_iters:
                return max(step / max(warmup_iters, 1), 1e-6)
            return 1.0

        warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
        cosine = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
        return torch.optim.lr_scheduler.ChainedScheduler([warmup, cosine])

    elif scheduler_type == 'graphcast':
        # GraphCast's schedule (supplement sec 4.3-4.5): linear warmup, then ONE
        # half-cosine decay, then a constant floor for the rollout tail. Deliberately
        # a single LambdaLR rather than a ChainedScheduler of warm RESTARTS -- the
        # point is that the LR is monotone non-increasing after warmup and ends near
        # zero, which is what makes the final checkpoint usable instead of forcing a
        # trailing-median read.
        import math as _math
        total = int(getattr(tc, 'lr_scheduler_total_iters', 0)) or planned_total_updates(config)
        warmup = max(int(getattr(tc, 'lr_scheduler_warmup_iters', 100)), 0)
        decay_frac = float(getattr(tc, 'lr_scheduler_decay_frac', 0.965))
        tail = float(getattr(tc, 'lr_scheduler_tail_ratio', 3e-4))
        decay_end = max(int(round(total * decay_frac)), warmup + 1)

        def lr_lambda(step):
            if step < warmup:
                return max(step / max(warmup, 1), 1e-8)
            if step >= decay_end:
                return tail
            p = (step - warmup) / max(decay_end - warmup, 1)
            return tail + (1.0 - tail) * 0.5 * (1.0 + _math.cos(_math.pi * p))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)

    else:
        raise ValueError(f"Unknown lr_scheduler: {scheduler_type}")


def enforce_dale_law(model, edge_index):
    """Enforce Dale's law: for each source neuron, force all outgoing W to the dominant sign.

    For each presynaptic neuron j, compute the sum of W over all its outgoing edges.
    Then set W_ij = |W_ij| * sign(sum_j) for all postsynaptic i.
    Edges where the sum is exactly zero are left unchanged.

    Call this with torch.no_grad() — it modifies W in-place.
    """
    with torch.no_grad():
        W = model.W  # (n_edges, 1)
        src = edge_index[0]  # presynaptic neuron index per edge
        n_neurons = int(src.max().item()) + 1

        # sum of W per source neuron
        w_sum = torch.zeros(n_neurons, device=W.device, dtype=W.dtype)
        w_sum.scatter_add_(0, src, W[:, 0])

        # dominant sign per source neuron (+1 or -1), 0 stays 0
        dominant_sign = w_sum.sign()

        # apply: |W| * dominant_sign[src]
        per_edge_sign = dominant_sign[src].unsqueeze(1)
        mask = per_edge_sign != 0
        W.data[mask] = W.data[mask].abs() * per_edge_sign[mask]


def dale_law_score(model, edge_index):
    """Compute Dale's law compliance score in [0, 1].

    For each source neuron j with at least 2 outgoing edges, compute the fraction
    of edges that match the dominant sign. Average across all such neurons.
    Returns 1.0 if every neuron has consistent sign, 0.5 for random signs.
    """
    with torch.no_grad():
        W = model.W[:, 0]  # (n_edges,)
        src = edge_index[0]
        n_neurons = int(src.max().item()) + 1

        # dominant sign per source
        w_sum = torch.zeros(n_neurons, device=W.device, dtype=W.dtype)
        w_sum.scatter_add_(0, src, W)
        dominant_sign = w_sum.sign()

        # per-edge: does this edge match its source's dominant sign?
        edge_sign = W.sign()
        edge_dominant = dominant_sign[src]
        matches = (edge_sign == edge_dominant).float()

        # exclude zero-weight edges from scoring
        nonzero = W.abs() > 1e-12
        if nonzero.sum() == 0:
            return 0.0

        # count per source neuron
        match_count = torch.zeros(n_neurons, device=W.device)
        total_count = torch.zeros(n_neurons, device=W.device)
        match_count.scatter_add_(0, src[nonzero], matches[nonzero])
        total_count.scatter_add_(0, src[nonzero], torch.ones_like(matches[nonzero]))

        # only neurons with >= 2 nonzero edges
        valid = total_count >= 2
        if valid.sum() == 0:
            return 1.0

        per_neuron_score = match_count[valid] / total_count[valid]
        return per_neuron_score.mean().item()


from dataclasses import dataclass


@dataclass
class TrainingParams:
    replace_with_cluster: bool
    umap_cluster_active: bool

    has_visual_field: bool
    test_neural_field: bool

    lr: float
    lr_update: float
    lr_embedding: float
    lr_W: float
    lr_NNR_f: float

    nnr_warmup_epochs: int
    lr_NNR_f_start: float
    lr_NNR_f_init: float


@dataclass
class TrainingData:
    x_ts: object
    y_ts: object
    y_ts_gpu: object

    type_list: object
    ode_params: object

    n_neurons: int
    n_frames: int

    xnorm: float
    ynorm: float

    edges: object
    gt_weights: object

def init_training_params(config):
    training = config.training
    model_config = config.graph_model

    lr = training.lr

    lr_update = (
        training.lr
        if training.lr_update == 0
        else training.lr_update
    )

    lr_embedding = training.lr_embedding
    lr_W = training.lr_W
    lr_NNR_f = training.lr_NNR_f

    nnr_warmup_epochs = int(
        getattr(
            training,
            'training_NNR_start_epoch',
            0,
        )
    )

    lr_NNR_f_start = float(
        getattr(
            training,
            'lr_NNR_f_start',
            0.0,
        )
    )

    if nnr_warmup_epochs > 0:
        lr_NNR_f_init = lr_NNR_f_start
    else:
        lr_NNR_f_init = lr_NNR_f

    return TrainingParams(
        replace_with_cluster=(
            'replace' in training.sparsity
        ),

        umap_cluster_active=(
            training.umap_cluster_method != 'none'
        ),

        has_visual_field=(
            'visual' in model_config.field_type
        ),

        test_neural_field=(
            'test' in model_config.field_type
        ),

        lr=lr,
        lr_update=lr_update,
        lr_embedding=lr_embedding,
        lr_W=lr_W,
        lr_NNR_f=lr_NNR_f,

        nnr_warmup_epochs=nnr_warmup_epochs,
        lr_NNR_f_start=lr_NNR_f_start,
        lr_NNR_f_init=lr_NNR_f_init,
    )

# =============================================================================
# 2. TRAINING DATA
# =============================================================================

def init_training_data(
    config,
    device,
    log_dir,
    logger,
):
    """
    Load FlyVis data, construct derivative targets, normalization, ground
    truth connectome information, and the graph used for training.
    """

    # Imported here to avoid unnecessary module-level coupling and to preserve
    # the existing training_utils dependency structure.
    from connectome_gnn.models.training_utils import (
        determine_load_fields,
        load_flyvis_data,
    )

    from connectome_gnn.generators.ode_params import (
        FlyVisODEParams,
        get_ode_params_class,
    )

    simulation = config.simulation
    training = config.training

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    load_fields = determine_load_fields(
        config
    )

    x_ts, _, type_list = load_flyvis_data(
        config.dataset,
        split='train',
        fields=load_fields,
        training_selected_neurons=(
            training.training_selected_neurons
        ),
        selected_neuron_ids=(
            training.selected_neuron_ids
            if training.training_selected_neurons
            else None
        ),
        measurement_noise_level=(
            simulation.measurement_noise_level
        ),
    )

    # -------------------------------------------------------------------------
    # Derivative target from observed voltage
    # -------------------------------------------------------------------------

    voltage = x_ts.voltage.numpy()

    y_ts = np.zeros_like(voltage)

    y_ts[:-1] = (
        voltage[1:]
        - voltage[:-1]
    ) / simulation.delta_t

    y_ts[-1] = y_ts[-2]

    y_ts = y_ts[..., None]

    # -------------------------------------------------------------------------
    # Dimensions
    # -------------------------------------------------------------------------

    n_neurons = x_ts.n_neurons
    n_frames_raw = x_ts.n_frames

    config.simulation.n_neurons = n_neurons
    simulation.n_frames = n_frames_raw

    logger.info(
        f'dataset: {n_frames_raw} frames, '
        f'n neurons: {n_neurons}'
    )

    # -------------------------------------------------------------------------
    # Recurrent subsampling
    # -------------------------------------------------------------------------

    full_stimulus = getattr(
        training,
        'recurrent_full_stimulus',
        False,
    )

    stride = (
        training.time_step
        if (
            training.recurrent_training
            and training.time_step > 1
            and not full_stimulus
        )
        else 1
    )

    if stride > 1:

        from tqdm import tqdm

        fields_to_stride = [
            'voltage',
            'stimulus',
            'calcium',
            'fluorescence',
            'noise',
        ]

        print(
            f"\033[93msubsampling dataset: "
            f"{n_frames_raw} frames -> "
            f"{n_frames_raw // stride} frames "
            f"(stride={stride})\033[0m"
        )

        for field in tqdm(
            fields_to_stride,
            desc='subsampling x_ts',
            ncols=150,
        ):
            value = getattr(
                x_ts,
                field,
            )

            if value is not None:
                setattr(
                    x_ts,
                    field,
                    value[::stride],
                )

        y_ts = y_ts[::stride]

        simulation.n_frames = x_ts.n_frames

    # -------------------------------------------------------------------------
    # Validation / normalization
    # -------------------------------------------------------------------------

    xnorm = x_ts.xnorm

    assert not torch.isnan(
        x_ts.voltage
    ).any(), (
        "voltage contains NaN — cannot train"
    )

    assert not np.isnan(
        y_ts
    ).any(), (
        "derivative targets contain NaN — cannot train"
    )

    x_ts = x_ts.to(device)

    # -------------------------------------------------------------------------
    # Optional temporal voltage denoising
    # -------------------------------------------------------------------------

    denoise_alpha = float(
        getattr(
            training,
            'coeff_voltage_denoise_alpha',
            0.0,
        )
    )

    if denoise_alpha > 0:

        from connectome_gnn.LLM_code.staging.block_01.temporal_voltage_denoise import (
            temporal_voltage_denoise,
        )

        x_ts.voltage = (
            (1.0 - denoise_alpha)
            * x_ts.voltage
            + denoise_alpha
            * temporal_voltage_denoise(
                x_ts.voltage
            )
        )

        logger.info(
            f'voltage denoising applied: '
            f'alpha={denoise_alpha}'
        )

    y_ts_gpu = (
        torch.from_numpy(y_ts)
        .float()
        .to(device)
    )

    torch.save(
        xnorm,
        os.path.join(
            log_dir,
            'xnorm.pt',
        ),
    )

    xnorm = float(xnorm)

    # Current trainer uses ynorm=1.
    ynorm = 1.0

    ynorm_tensor = torch.tensor(
        ynorm,
        device=device,
    )

    torch.save(
        ynorm_tensor,
        os.path.join(
            log_dir,
            'ynorm.pt',
        ),
    )

    # -------------------------------------------------------------------------
    # Load ground-truth ODE / connectome information
    # -------------------------------------------------------------------------

    signal_model = (
        config.graph_model.signal_model_name
    )

    try:
        OdeParamsCls = get_ode_params_class(
            signal_model
        )
    except KeyError:
        OdeParamsCls = FlyVisODEParams

    try:
        ode_params = OdeParamsCls.load(
            graphs_data_path(config.dataset),
            device=device,
        )
    except TypeError:

        logger.info(
            f'ode_params schema mismatch for '
            f'{OdeParamsCls.__name__}; '
            f'falling back to FlyVisODEParams'
        )

        ode_params = FlyVisODEParams.load(
            graphs_data_path(config.dataset),
            device=device,
        )

    gt_weights = ode_params.W
    gt_edges = ode_params.edge_index

    # -------------------------------------------------------------------------
    # Construct training graph
    # -------------------------------------------------------------------------

    if not training.use_gt_edges:

        src = torch.arange(
            n_neurons,
            device=device,
        ).repeat_interleave(
            n_neurons
        )

        dst = torch.arange(
            n_neurons,
            device=device,
        ).repeat(n_neurons)

        mask = src != dst

        edges = torch.stack(
            [
                src[mask],
                dst[mask],
            ],
            dim=0,
        )

        config.simulation.n_edges = (
            edges.shape[1]
        )

        # Remap GT weights to the training-edge order.
        gt_weight_map = torch.zeros(
            edges.shape[1],
            device=device,
        )

        gt_edge_set = {
            (
                gt_edges[0, k].item(),
                gt_edges[1, k].item(),
            ): gt_weights[k]
            for k in range(
                gt_edges.shape[1]
            )
        }

        for k in range(
            edges.shape[1]
        ):
            key = (
                edges[0, k].item(),
                edges[1, k].item(),
            )

            if key in gt_edge_set:
                gt_weight_map[k] = (
                    gt_edge_set[key]
                )

        gt_weights = gt_weight_map

    else:

        edges = gt_edges

        actual_n_edges = (
            edges.shape[1]
        )

        expected_total = (
            simulation.n_edges
            + simulation.n_extra_null_edges
        )

        if (
            actual_n_edges == expected_total
            and simulation.n_extra_null_edges > 0
        ):
            config.simulation.n_edges = (
                actual_n_edges
            )

            config.simulation.n_extra_null_edges = 0

        elif actual_n_edges != simulation.n_edges:
            config.simulation.n_edges = (
                actual_n_edges
            )

    # -------------------------------------------------------------------------
    # Save the exact graph used during training
    # -------------------------------------------------------------------------

    torch.save(
        edges,
        os.path.join(
            log_dir,
            'training_edges.pt',
        ),
    )

    torch.save(
        gt_weights,
        os.path.join(
            log_dir,
            'gt_weights.pt',
        ),
    )

    return TrainingData(
        x_ts=x_ts,
        y_ts=y_ts,
        y_ts_gpu=y_ts_gpu,

        type_list=type_list,
        ode_params=ode_params,

        n_neurons=n_neurons,
        n_frames=simulation.n_frames,

        xnorm=xnorm,
        ynorm=ynorm,

        edges=edges,
        gt_weights=gt_weights,
    )


# =============================================================================
# 3. MODEL
# =============================================================================

def init_training_model(
    config,
    data,
    device,
    log_dir,
    best_model=None,
    resume=False,
):
    """
    Build/load the GNN and perform model initialization that must happen
    before optimizer construction.
    """

    from connectome_gnn.models.training_utils import (
        build_model,
        find_latest_epoch_checkpoint,
    )

    training = config.training

    checkpoint_path = None
    resumed_epoch = -1

    if resume:

        checkpoint_path, resumed_epoch = (
            find_latest_epoch_checkpoint(
                log_dir,
                training.n_runs,
            )
        )

    elif training.pretrained_model != '':

        checkpoint_path = (
            training.pretrained_model
        )

    reset_epoch = (
        training.pretrained_model != ''
        and not resume
    )

    model, start_epoch = build_model(
        config,
        device,
        checkpoint_path=checkpoint_path,
        reset_epoch=reset_epoch,
    )

    if (
        resume
        and checkpoint_path is not None
    ):
        start_epoch = resumed_epoch + 1

    # -------------------------------------------------------------------------
    # Hard connectome sign lock
    # -------------------------------------------------------------------------

    if getattr(
        model,
        'lock_edge_signs_from_connectome',
        False,
    ):
        model.set_edge_sign_from_weights(
            data.gt_weights
        )

    # -------------------------------------------------------------------------
    # Cell-type embedding initialization
    # -------------------------------------------------------------------------

    if training.embedding_cell_type_init:

        from connectome_gnn.utils import (
            get_equidistant_points,
        )

        n_types = (
            config.simulation.n_neuron_types
        )

        embedding_dim = (
            config.graph_model.embedding_dim
        )

        if embedding_dim == 2:

            ex, ey = (
                get_equidistant_points(
                    n_types
                )
            )

            points = (
                np.stack(
                    [ex, ey],
                    axis=1,
                )
                * training.embedding_cell_type_scale
            )

            type_ids = (
                data.type_list
                .squeeze(-1)
                .long()
                .cpu()
                .numpy()
            )

            with torch.no_grad():
                model.a.copy_(
                    torch.tensor(
                        points[type_ids],
                        dtype=torch.float32,
                        device=device,
                    )
                )

    # -------------------------------------------------------------------------
    # Freeze embedding before optimizer creation
    # -------------------------------------------------------------------------

    if training.fix_embedding:
        model.a.requires_grad_(False)

    # flyvis_cond_known_ode needs three things the config cannot carry: the per-edge
    # polarity, the cell-type map, and (when frozen) the teacher's own neuron
    # constants. All three are GT STRUCTURE rather than fitted quantities -- only the
    # SIGN of ode_params.W is read, never its magnitude. Dale's law does NOT hold in
    # this connectome (4,573 of 13,279 presynaptic neurons have mixed-sign outgoing
    # edges), so the polarity has to be per EDGE; a per-neuron or per-type reduction
    # would silently pick one sign for a third of them.
    if hasattr(model, "set_presynaptic_sign"):
        op = getattr(data, "ode_params", None)
        _get = (lambda k: op.get(k)) if isinstance(op, dict) else (lambda k: getattr(op, k, None))
        w = _get("W") if op is not None else None
        if w is None:
            raise RuntimeError(
                "flyvis_cond_known_ode needs ode_params.W for the per-edge polarity; "
                "this dataset carries none")
        model.set_presynaptic_sign(w)
        tl = getattr(data, "type_list", None)
        if tl is not None:
            model.set_neuron_types(tl)
        # BOTH modes: 'margin' fixes the reversals here, 'learned' merely starts
        # from them. Leaving 'learned' at +-1 made the closed-form init divide by an
        # (E - Vbar) whose sign did not match the connectome's on 44,735 of 434,112
        # edges, which the guard caught as a negative conductance at second zero.
        if True:
            xt = getattr(data, "x_ts", None)
            v = getattr(xt, "voltage", None) if xt is not None else None
            if v is None:
                raise RuntimeError(
                    "cond_reversal_mode 'margin' needs the teacher's voltage range and "
                    "this dataset carries no x_ts.voltage")
            # Per-neuron extremes for the BRACKET, and per-neuron percentiles for
            # delta's UNIT when cond_span_mode asks for them. The model reduces both
            # to whatever granularity cond_reversal_dim wants. Frames are subsampled
            # for the quantile: (64000, 13741) exact quantiles cost more than the
            # answer is worth and the tails are what we are deliberately trimming.
            _vmin, _vmax = v.float().amin(dim=0), v.float().amax(dim=0)
            _sm = getattr(training, "cond_span_mode", "extremes")
            if _sm == "extremes":
                _lo = _hi = None
            else:
                # torch.quantile REFUSES INPUTS OVER ~16M ELEMENTS ("input tensor is
                # too large") and allocates a full sort besides, so the obvious
                # (20000, 13741) call both raises and OOMs at 3 GB. Subsample frames,
                # then chunk over neurons.
                _q = {"p99": 0.01, "p95": 0.05, "p90": 0.10}[_sm]
                _nf = min(4000, v.shape[0])
                _idx = torch.linspace(0, v.shape[0] - 1, _nf, device=v.device).long()
                _sub = v[_idx].float()
                if getattr(training, "cond_reversal_dim", "global") == "global":
                    # ONE reversal pair -> POOL over every (neuron, frame) pair. Taking
                    # per-neuron quantiles and then the widest across neurons is a
                    # different and much larger statistic (span 9.2 against 3.9 here),
                    # because it keeps the most extreme neuron's tail.
                    _flat = _sub.reshape(-1)
                    _step = max(1, _flat.numel() // 4_000_000)
                    _flat = _flat[::_step]
                    _lo = torch.quantile(_flat, _q).expand(v.shape[1]).contiguous()
                    _hi = torch.quantile(_flat, 1.0 - _q).expand(v.shape[1]).contiguous()
                else:
                    _chunk = max(1, 4_000_000 // max(_nf, 1))
                    _lo = torch.empty(v.shape[1], device=v.device)
                    _hi = torch.empty(v.shape[1], device=v.device)
                    for _c0 in range(0, v.shape[1], _chunk):
                        _c = _sub[:, _c0:_c0 + _chunk]
                        _lo[_c0:_c0 + _chunk] = torch.quantile(_c, _q, dim=0)
                        _hi[_c0:_c0 + _chunk] = torch.quantile(_c, 1.0 - _q, dim=0)
                del _sub
            model.set_teacher_voltage_range(_vmin, _vmax, v_lo=_lo, v_hi=_hi)
        if getattr(training, "cond_init", "teacher_closed_form") == "teacher_closed_form":
            # Vbar per CELL TYPE, not per neuron: the expansion in the methods is
            # about the type's mean postsynaptic voltage, and the teacher's own
            # tau/V_rest are type-constant too. Falls back to a per-neuron mean when
            # no type map is available.
            xt = getattr(data, "x_ts", None)
            v = getattr(xt, "voltage", None) if xt is not None else None
            ei = _get("edge_index")
            if v is None or ei is None:
                raise RuntimeError(
                    "cond_init 'teacher_closed_form' needs x_ts.voltage and "
                    "ode_params.edge_index")
            # RAW PER-NEURON mean. The model reduces it onto E's own rows, and it
            # must: reducing here by cell type while cond_reversal_dim is per_neuron
            # left 512 of 434,112 edges with a Vbar outside the range their own
            # reversal was built to bracket, hence a negative conductance.
            model.init_from_teacher(w, ei, v.float().mean(dim=0))
        if getattr(training, "cond_neuron_params", "per_type") == "frozen":
            tau, vr = _get("tau_i"), _get("V_i_rest")
            if tau is None or vr is None:
                raise RuntimeError("cond_neuron_params 'frozen' needs ode_params tau_i/V_i_rest")
            model.set_teacher_neuron_params(tau, vr)

    model.train()

    return model, start_epoch


# =============================================================================
# 4. OPTIMIZER
# =============================================================================

def init_training_optimizer(
    config,
    model,
):
    """
    Build optimizer and scheduler.

    Returns:
        optimizer
        lr_scheduler
        n_total_params
    """

    from connectome_gnn.models.training_utils import (
        build_lr_scheduler,
    )

    training = config.training

    lr = training.lr

    lr_update = (
        training.lr
        if training.lr_update == 0
        else training.lr_update
    )

    lr_embedding = (
        training.lr_embedding
    )

    lr_W = training.lr_W
    lr_NNR_f = training.lr_NNR_f

    nnr_warmup_epochs = int(
        getattr(
            training,
            'training_NNR_start_epoch',
            0,
        )
    )

    lr_NNR_f_start = float(
        getattr(
            training,
            'lr_NNR_f_start',
            0.0,
        )
    )

    lr_NNR_f_init = (
        lr_NNR_f_start
        if nnr_warmup_epochs > 0
        else lr_NNR_f
    )

    optimizer, n_total_params = (
        set_trainable_parameters(
            model=model,
            lr_embedding=lr_embedding,
            lr=lr,
            lr_update=lr_update,
            lr_W=lr_W,
            lr_NNR_f=lr_NNR_f_init,
        )
    )

    lr_scheduler = build_lr_scheduler(
        optimizer,
        config,
    )

    return (
        optimizer,
        lr_scheduler,
        n_total_params,
    )


# =============================================================================
# 5. REGULARIZER
# =============================================================================

def init_training_regularizer(
    config,
    data,
    device,
):
    """
    Construct and initialize LossRegularizer.

    LossRegularizer stays in models/regularizer.py.
    This function simply owns its training-time initialization.
    """

    training = config.training
    simulation = config.simulation
    model_config = config.graph_model

    regularizer = LossRegularizer(
        train_config=training,
        model_config=model_config,
        activity_column=3,
        plot_frequency=1,
        n_neurons=data.n_neurons,
        trainer_type='flyvis',
        dataset=config.dataset,
        type_list=data.type_list,
        n_neuron_types=simulation.n_neuron_types,
    )

    regularizer.set_activity_stats(
        data.x_ts,
        device,
    )

    regularizer.move_type_list_to_device(
        device
    )

    return regularizer


# =============================================================================
# 6. HIDDEN + ANCHOR NEURONS
# =============================================================================

def init_hidden_neurons(
    config,
    model,
    n_neurons,
    log_dir,
    device,
):
    """
    Initialize hidden-neuron and anchor-neuron selections.

    Returns:
        hidden_ids
        visible_ids
        anchor_ids

    visible_ids is derived here because it is a direct consequence of the
    hidden-neuron selection.
    """

    simulation = config.simulation
    training = config.training
    model_config = config.graph_model

    hidden_ids = None
    anchor_ids = None

    # -------------------------------------------------------------------------
    # Hidden neurons
    # -------------------------------------------------------------------------

    hidden_fraction = float(
        getattr(
            model_config,
            'hidden_neuron_fraction',
            0.0,
        )
    )

    if hidden_fraction > 0:

        hidden_path = os.path.join(
            log_dir,
            'hidden_neuron_ids.pt',
        )

        if os.path.exists(
            hidden_path
        ):

            hidden_ids = torch.load(
                hidden_path,
                map_location=device,
                weights_only=True,
            )

        else:

            rng = np.random.RandomState(
                simulation.seed
            )

            candidates = np.arange(
                simulation.n_input_neurons,
                n_neurons,
            )

            n_hidden = int(
                len(candidates)
                * hidden_fraction
            )

            hidden_np = np.sort(
                rng.choice(
                    candidates,
                    size=n_hidden,
                    replace=False,
                )
            )

            hidden_ids = (
                torch.from_numpy(
                    hidden_np
                )
                .long()
                .to(device)
            )

            torch.save(
                hidden_ids,
                hidden_path,
            )

    # -------------------------------------------------------------------------
    # Anchor neurons
    # -------------------------------------------------------------------------

    if hidden_ids is not None:

        inr_type_hidden = getattr(
            model_config,
            'inr_type_hidden',
            'none',
        )

        inner_model = (
            model._orig_mod
            if hasattr(model, '_orig_mod')
            else model
        )

        has_anchor_neurons = (
            bool(
                getattr(
                    training,
                    'train_with_anchor_neurons',
                    False,
                )
            )
            and inr_type_hidden in (
                'siren_t',
                'ngp_t',
            )
            and getattr(
                inner_model,
                'n_anchor',
                0,
            ) > 0
        )

        if has_anchor_neurons:

            anchor_path = os.path.join(
                log_dir,
                'anchor_neuron_ids.pt',
            )

            n_anchor = int(
                inner_model.n_anchor
            )

            if os.path.exists(
                anchor_path
            ):

                anchor_ids = torch.load(
                    anchor_path,
                    map_location=device,
                    weights_only=True,
                )

                if len(anchor_ids) != n_anchor:
                    anchor_ids = None

            if anchor_ids is None:

                rng = np.random.RandomState(
                    simulation.seed + 1
                )

                candidates = np.setdiff1d(
                    np.arange(
                        simulation.n_input_neurons,
                        n_neurons,
                    ),
                    hidden_ids.cpu().numpy(),
                )

                n_anchor_eff = min(
                    n_anchor,
                    len(candidates),
                )

                anchor_np = np.sort(
                    rng.choice(
                        candidates,
                        size=n_anchor_eff,
                        replace=False,
                    )
                )

                anchor_ids = (
                    torch.from_numpy(
                        anchor_np
                    )
                    .long()
                    .to(device)
                )

                torch.save(
                    anchor_ids,
                    anchor_path,
                )

    # -------------------------------------------------------------------------
    # Visible neurons
    # -------------------------------------------------------------------------

    ids = torch.arange(
        n_neurons,
        device=device,
    )

    if hidden_ids is None:

        visible_ids = ids

    else:

        hidden_mask = torch.zeros(
            n_neurons,
            dtype=torch.bool,
            device=device,
        )

        hidden_mask[hidden_ids] = True

        visible_ids = ids[
            ~hidden_mask
        ]

    return (
        hidden_ids,
        visible_ids,
        anchor_ids,
    )

def inject_hidden_voltage(model, x, k, hidden_ids, injection_active):
    """Hidden-neuron voltage estimator: NGP/SIREN forward or zero-silence.

    Mutates x.voltage[hidden_ids] in place. injection_active is binary:
    phase 1 → False, phase 2 → True. The smooth absorption of the new
    input distribution at the phase 1→2 transition is handled by the
    LR-damping V-schedule on the GNN param groups, not by ramping the
    injection magnitude here.

    Phase 1 (injection_active=False): hidden voltages are zero-silenced
    (identical to the no-NGP baseline). NGP/SIREN still trains via the
    anchor loss elsewhere in the step, which routes through the spatial
    NGP position cache — normally primed inside forward_hidden, so it is
    primed here instead (idempotent) to keep that path populated.

    Phase 2 (injection_active=True): NGP/SIREN forward_hidden predicts the
    hidden voltages directly, and gradients flow back through injection.
    """
    if model.NNR_hidden is not None and injection_active:
        x.voltage[hidden_ids] = model.forward_hidden(x, k, hidden_ids)
    else:
        x.voltage[hidden_ids] = 0.0
        if model.NNR_hidden is not None and getattr(model, '_ngp_spatial_enabled', False):
            model._ngp_cache_pos(x)


class HiddenNeuronHandler:
    """Owns hidden/anchor neuron selection and mediates their voltage
    through the model's NGP/SIREN hidden-neuron generator (model.NNR_hidden).

    `model` is passed to each call rather than stored on the handler,
    since torch.compile rebinds the trainer's `model` variable to a
    wrapper after this handler is constructed.
    """

    def __init__(self, config, model, n_neurons, log_dir, device):
        self.hidden_ids, self.visible_ids, self.anchor_ids = init_hidden_neurons(
            config, model, n_neurons, log_dir, device
        )

    @classmethod
    def from_ids(cls, hidden_ids, anchor_ids=None, visible_ids=None):
        """Build a handler directly from already-loaded ids, bypassing the
        config-driven sampling in `init_hidden_neurons`. Used at eval/test
        time, where hidden/anchor ids are loaded from a trained run's
        log_dir rather than sampled fresh."""
        self = cls.__new__(cls)
        self.hidden_ids = hidden_ids
        self.anchor_ids = anchor_ids
        self.visible_ids = visible_ids
        return self

    @property
    def has_hidden(self):
        return self.hidden_ids is not None

    @property
    def has_anchor(self):
        return self.anchor_ids is not None

    def inject_hidden(self, model, x, k, injection_active):
        """No-op when there are no hidden neurons. See `inject_hidden_voltage`
        for the phase-transition rationale."""
        if self.has_hidden:
            inject_hidden_voltage(model, x, k, self.hidden_ids, injection_active)

    def zero_hidden(self, state):
        """Silence hidden-neuron voltage on a rolled-forward recurrent state."""
        if self.has_hidden:
            state.voltage[self.hidden_ids] = 0.0

    def anchor_residual(self, model, k_starts, x_ts):
        """(pred - gt) anchor voltage residual, or None if anchors are disabled."""
        if not self.has_anchor:
            return None
        pred_a = model.forward_anchor_batched(k_starts, anchor_ids=self.anchor_ids)
        gt_a = x_ts.voltage[k_starts[:, None], self.anchor_ids[None, :]]
        return pred_a - gt_a

    def quick_pearson(self, model, x_ts, device):
        """Lightweight Pearson r for the hidden/anchor NGP fit. Returns
        (hidden_r2, hidden_std, anchor_r2, anchor_std), each None where not
        computable (no hidden neurons / NNR_hidden not yet initialised)."""
        if not self.has_hidden or getattr(model, 'NNR_hidden', None) is None:
            return None, None, None, None

        hidden_r2, hidden_std = _quick_ngp_pearson(
            model, x_ts, self.hidden_ids, use_anchor=False, device=device, return_stats=True
        )

        anchor_r2 = anchor_std = None
        if self.has_anchor:
            anchor_r2, anchor_std = _quick_ngp_pearson(
                model, x_ts, self.anchor_ids, use_anchor=True, device=device, return_stats=True
            )

        return hidden_r2, hidden_std, anchor_r2, anchor_std


def run_nominal_train_step(
    model,
    x_ts,
    y_ts_gpu,
    edges,
    ids,
    hn,
    regularizer,
    epoch_state,
    training,
    sim,
    model_config,
    train,
    device,
    N,
    n_neurons,
    xnorm,
    ynorm,
    injection_active,
    target_weight=None,
):
    """One iteration of standard (non-recurrent) training: compute the loss and
    update the regularizer.

    Returns the loss still attached to the AUTOGRAD graph (it carries grad_fn), so
    the caller owns backward/step. Not the connectome graph.
    The regularisation value is not returned: it is accumulated on the regularizer object.
    """
    state_batch = []
    y_list = []
    ids_list = []
    k_list = []
    visual_input_list = []

    ids_index = 0

    loss = torch.zeros((), device=device)

    regularizer.reset_iteration(device=device)

    # -------------------------------------------------------------
    # Consecutive batch
    # -------------------------------------------------------------

    if training.consecutive_batch:
        k_start = int(epoch_state.frame_indices[N * training.batch_size])

    for batch in range(training.batch_size):
        if training.consecutive_batch:
            k = k_start + batch

        else:
            k = int(epoch_state.frame_indices[N * training.batch_size + batch])

        x = x_ts.frame(k)

        # ---------------------------------------------------------
        # Measurement noise
        # ---------------------------------------------------------

        if x.noise is not None and sim.measurement_noise_level > 0:
            x.voltage = x.voltage + x.noise

        # ---------------------------------------------------------
        # Hidden neurons
        # ---------------------------------------------------------

        hn.inject_hidden(model, x, k, injection_active)

        # ---------------------------------------------------------
        # Temporal window
        # ---------------------------------------------------------

        if training.time_window > 0:
            x_temporal = x_ts.voltage[k - training.time_window + 1 : k + 1].T

            # x stays as NeuronState;
            # x_temporal is passed separately if needed.

        # ---------------------------------------------------------
        # Visual field
        # ---------------------------------------------------------

        if train.has_visual_field:
            visual_input = model.forward_visual(x, k)

            x.stimulus[: model.n_input_neurons] = visual_input.squeeze(-1)

            x.stimulus[model.n_input_neurons :] = 0

        # ---------------------------------------------------------
        # Regularization
        # ---------------------------------------------------------

        if batch == 0:
            regul_loss = regularizer.compute(
                model=model,
                x=x,
                in_features=None,
                ids=ids,
                ids_batch=None,
                edges=edges,
                device=device,
                xnorm=xnorm,
                perm_indices=regularizer.sample_g_phi_perm(device),
            ) 

            loss = loss + regul_loss

        # ---------------------------------------------------------
        # Target
        # ---------------------------------------------------------

        if training.recurrent_training or training.neural_ODE_training:
            # Same predicate as get_training_frame_sampling's stride_subsample
            # (training_utils.py) — when the dataset was decimated by time_step at
            # load, one stored frame already IS time_step raw steps, so the target
            # is k+1; otherwise it is k+time_step.
            target_frame = (
                k + 1
                if (training.recurrent_training and training.time_step > 1)
                else (k + training.time_step)
            )

            y = x_ts.voltage[target_frame].unsqueeze(-1)

        elif train.test_neural_field:
            y = x_ts.stimulus[k, : sim.n_input_neurons].unsqueeze(-1)

        else:
            y = y_ts_gpu[k] / ynorm

        if epoch_state.loss_noise_level > 0:
            y = y + torch.randn(y.shape, device=device) * epoch_state.loss_noise_level

        # ---------------------------------------------------------
        # Accumulate batch
        # ---------------------------------------------------------

        state_batch.append(x)

        n = x.n_neurons

        y_list.append(y)

        ids_list.append(hn.visible_ids + ids_index)

        k_list.append(torch.ones((n, 1), dtype=torch.int, device=device) * k)

        if train.test_neural_field:
            visual_input_list.append(visual_input)

        ids_index += n

    # -----------------------------------------------------------------
    # Batch assembly
    # -----------------------------------------------------------------

    data_id = torch.zeros((ids_index, 1), dtype=torch.int, device=device)

    y_batch = torch.cat(y_list, dim=0)

    ids_batch = torch.cat(ids_list, dim=0)

    k_batch = torch.cat(k_list, dim=0)

    # -----------------------------------------------------------------
    # Visual-field testing
    # -----------------------------------------------------------------

    if train.test_neural_field:
        visual_input_batch = torch.cat(visual_input_list, dim=0)

        loss = loss + (visual_input_batch - y_batch).norm(2)

    # -----------------------------------------------------------------
    # MLP ODE
    # -----------------------------------------------------------------

    elif "mlp_ode" in model_config.signal_model_name.lower():
        batched_state, _ = _batch_frames(state_batch, edges)

        batched_x = batched_state.to_packed()

        pred = model(batched_x, data_id=data_id, return_all=False)

        loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

    # -----------------------------------------------------------------
    # MLP
    # -----------------------------------------------------------------

    elif "mlp" in model_config.signal_model_name.lower():
        batched_state, _ = _batch_frames(state_batch, edges)

        pred = model(batched_state, data_id=data_id, return_all=False)

        loss = loss + (pred[ids_batch] - y_batch[ids_batch]).norm(2)

    # -----------------------------------------------------------------
    # GNN
    # -----------------------------------------------------------------

    else:
        batched_state, batched_edges = _batch_frames(state_batch, edges)

        (pred, in_features, msg) = model(batched_state, batched_edges, data_id=data_id, return_all=True)

        update_regul = regularizer.compute_update_regul(model, in_features, ids_batch, device)

        loss = loss + update_regul

        # -------------------------------------------------------------
        # Neural ODE
        # -------------------------------------------------------------

        if training.neural_ODE_training:
            ode_state_clamp = getattr(training, "ode_state_clamp", 10.0)

            ode_stab_lambda = getattr(training, "ode_stab_lambda", 0.0)

            ode_loss, pred_x = neural_ode_loss(
                model=model,
                dataset_batch=state_batch,
                edge_index=edges,
                x_ts=x_ts,
                k_batch=k_batch,
                time_step=training.time_step,
                batch_size=training.batch_size,
                n_neurons=n_neurons,
                ids_batch=ids_batch,
                delta_t=sim.delta_t,
                device=device,
                data_id=data_id,
                has_visual_field=(train.has_visual_field),
                y_batch=y_batch,
                noise_level=(training.noise_recurrent_level),
                ode_method=training.ode_method,
                rtol=training.ode_rtol,
                atol=training.ode_atol,
                adjoint=training.ode_adjoint,
                iteration=N,
                state_clamp=(ode_state_clamp),
                stab_lambda=(ode_stab_lambda),
            )

            loss = loss + ode_loss

        # -------------------------------------------------------------
        # Recurrent GNN
        # -------------------------------------------------------------

        elif training.recurrent_training:
            pred_x = (
                batched_state.voltage.unsqueeze(-1)
                + sim.delta_t * pred
                + training.noise_recurrent_level * torch.randn_like(pred)
            )

            if training.time_step > 1:
                for step in range(training.time_step - 1):
                    neurons_per_sample = state_batch[0].n_neurons

                    for b in range(training.batch_size):
                        start_idx = b * neurons_per_sample

                        end_idx = (b + 1) * neurons_per_sample

                        state_batch[b].voltage = pred_x[start_idx:end_idx].squeeze()

                        hn.zero_hidden(state_batch[b])

                        k_current = k_batch[start_idx, 0].item() + step + 1

                        if train.has_visual_field:
                            visual_input_next = model.forward_visual(state_batch[b], k_current)

                            state_batch[b].stimulus[: model.n_input_neurons] = visual_input_next.squeeze(-1)

                            state_batch[b].stimulus[model.n_input_neurons :] = 0

                        else:
                            x_next = x_ts.frame(k_current)

                            state_batch[b].stimulus = x_next.stimulus

                            if x_next.optogenetics_stimulus is not None:
                                state_batch[b].optogenetics_stimulus = x_next.optogenetics_stimulus

                    (batched_state, batched_edges) = _batch_frames(state_batch, edges)

                    (pred, in_features, msg) = model(
                        batched_state, batched_edges, data_id=data_id, return_all=True
                    )

                    pred_x = (
                        pred_x + sim.delta_t * pred + training.noise_recurrent_level * torch.randn_like(pred)
                    )

            loss = loss + ((pred_x[ids_batch] - y_batch[ids_batch]) / (sim.delta_t * training.time_step)).norm(
                2
            )

        # -------------------------------------------------------------
        # Standard one-step GNN loss
        # -------------------------------------------------------------

        else:
            loss = loss + fit_residual_loss(
                pred[ids_batch] - y_batch[ids_batch],
                getattr(training, "fit_reduction", "norm2"),
                weight=None if target_weight is None else target_weight[ids_batch],
            )

            # Hidden self-consistency loss intentionally removed.
            #
            # NGP-hidden is supervised through:
            #   1. anchor voltage loss
            #   2. backpropagation through hidden-voltage injection
            #
            # This preserves the behavior of the current implementation.

            if hn.has_anchor and getattr(training, "coeff_anchor_voltage", 0.0) > 0:
                n_per = state_batch[0].n_neurons

                k_starts = k_batch[::n_per, 0].to(torch.long)

                anchor_residual = hn.anchor_residual(model, k_starts, x_ts)

                loss = loss + training.coeff_anchor_voltage * anchor_residual.norm(2)

    return loss


def run_recurrent_train_step(
    model,
    x_ts,
    y_ts,
    edges,
    ids,
    hn,
    regularizer,
    epoch_state,
    config,
    training,
    train,
    device,
    N,
    xnorm,
    ynorm,
    rollout_horizon,
    target_weight=None,
):
    """One iteration of recurrent training. Returns the loss, same contract as
    run_nominal_train_step — the caller owns backward/step and the shared
    metrics tail.

    Thin by design: recurrent_loss (recurrent_step.py) owns the rollout and
    dispatches on mode — dense-supervision curriculum when rollout_horizon is
    given, else multi-start or the legacy endpoint-only scheme.

    y_ts must be the DEVICE tensor (data.y_ts_gpu), not data.y_ts, which is a
    numpy array — the dense curriculum indexes it per step exactly as the nominal
    path indexes y_ts_gpu.

    Batching: modes 1 (standard) and 4 (dense curriculum) concatenate
    training.batch_size sampled start frames into ONE graph via _batch_frames,
    exactly as the nominal path does. Mode 2 (multi_start_recurrent) does NOT —
    it ignores batch_size and runs time_step sequential single-frame rollouts.
    """
    from connectome_gnn.models.recurrent_step import recurrent_loss

    loss, _regul_val = recurrent_loss(
        model=model,
        x_ts=x_ts,
        y_ts=y_ts,
        edges=edges,
        ids=ids,
        frame_indices=epoch_state.frame_indices,
        iter_idx=N,
        config=config,
        device=device,
        xnorm=xnorm,
        ynorm=ynorm,
        regularizer=regularizer,
        has_visual_field=train.has_visual_field,
        hn=hn,
        n_steps=rollout_horizon,
        target_weight=target_weight,
    )

    return loss
