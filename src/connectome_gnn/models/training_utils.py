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
from connectome_gnn.models.utils import set_trainable_parameters
from connectome_gnn.utils import graphs_data_path, migrate_state_dict, sort_key
from connectome_gnn.zarr_io import load_raw_array, load_simulation_data
from dataclasses import dataclass 
from connectome_gnn.models.regularizer import LossRegularizer

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
    # SVD analysis
    # -------------------------------------------------------------------------

    svd_plot_path = os.path.join(
        log_dir,
        'results',
        'svd_analysis.png',
    )

    if not os.path.exists(
        svd_plot_path
    ):
        analyze_data_svd(
            x_ts,
            log_dir,
            config=config,
            logger=logger,
            is_flyvis=True,
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
