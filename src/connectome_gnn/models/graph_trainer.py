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
from connectome_gnn.metrics import compute_dynamics_r2, fmt_r2_bar
from connectome_gnn.models.neural_ode_wrapper import (
    debug_check_gradients,
    neural_ode_loss,
)
from connectome_gnn.models.recurrent_step import recurrent_loss
from connectome_gnn.models.registry import create_model
from connectome_gnn.plot import (
    plot_jacobian_w_scatter,
    plot_metrics,
    plot_signal_loss,
    plot_training_gnn,
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
from connectome_gnn.models.utils import (
    ANSI_GREEN,
    ANSI_ORANGE,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
    _NGP_QUICK_FREQ,
    _batch_frames,
    _quick_ngp_pearson,
    analyze_data_svd,
    model_family,
    r2_color,
    set_trainable_parameters,
)

from connectome_gnn.models.training_utils import (
    build_lr_scheduler,
    build_model,
    dale_law_score,
    determine_load_fields,
    enforce_dale_law,
    find_latest_epoch_checkpoint,
    format_metric,
    init_epoch_state,
    init_hidden_neurons,
    init_ngp_schedule,
    init_training_data,
    init_training_model,
    init_training_optimizer,
    init_training_params,
    init_training_regularizer,
    init_training_runtime,
    load_flyvis_data,
)

_logger = get_logger(__name__)


def data_train(config=None, erase=False, best_model=None, style=None, device=None, log_file=None, resume=False):
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
    if config.training.deterministic:
        from connectome_gnn.utils import set_deterministic
        set_deterministic(seed)

    # torch.autograd.set_detect_anomaly(True)

    _logger.info(f"dataset: {config.dataset}")
    _logger.info(f"{config.description}")

    # Task-data trainer (path_integration etc.). Detected via the presence
    # of a populated task block — keeps train_subprocess.py / GNN_Main.py
    # routing transparent for both the LLM agentic loop and direct CLI use.
    if getattr(config, 'task', None) is not None:
        data_train_task(config, erase, best_model, device, log_file=log_file, resume=resume)
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
            data_train_gnn(config, erase, best_model, device, log_file=log_file, resume=resume)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset}")

    _logger.info("training completed.")


def _inject_hidden_voltage(model, x, k, hidden_ids, injection_active):
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


def data_train_gnn(
    config,
    erase,
    best_model,
    device,
    log_file=None,
    resume=False,
):

    # =====================================================================
    # INITIALIZATION
    # =====================================================================

    train = init_training_params(config)

    sim = config.simulation
    training = config.training
    model_config = config.graph_model

    # Derived training values used repeatedly in the loop.
    lr = train.lr
    lr_update = train.lr_update
    lr_embedding = train.lr_embedding
    lr_W = train.lr_W
    lr_NNR_f = train.lr_NNR_f

    log_dir, logger = create_log_dir(
        config,
        erase,
    )

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------

    data = init_training_data(
        config,
        device,
        log_dir,
        logger,
    )

    x_ts = data.x_ts
    y_ts = data.y_ts
    y_ts_gpu = data.y_ts_gpu
    type_list = data.type_list
    ode_params = data.ode_params

    n_neurons = data.n_neurons
    n_frames = data.n_frames

    xnorm = data.xnorm
    ynorm = data.ynorm

    edges = data.edges
    gt_weights = data.gt_weights

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------

    model, start_epoch = init_training_model(
        config,
        data,
        device,
        log_dir,
        best_model=best_model,
        resume=resume,
    )

    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------

    optimizer, lr_scheduler, n_total_params = (
        init_training_optimizer(
            config,
            model,
        )
    )

    # ---------------------------------------------------------------------
    # Neuron IDs
    # ---------------------------------------------------------------------

    ids = torch.arange(
        n_neurons,
        device=device,
    )

    hidden_ids, visible_ids, anchor_ids = (
        init_hidden_neurons(
            config,
            model,
            n_neurons,
            log_dir,
            device,
        )
    )

    has_hidden_neurons = (
        hidden_ids is not None
    )

    has_anchor_neurons = (
        anchor_ids is not None
    )

    # ---------------------------------------------------------------------
    # Regularizer
    # ---------------------------------------------------------------------

    regularizer = init_training_regularizer(
        config,
        data,
        device,
    )

    # =====================================================================
    # RUNTIME SETUP
    # =====================================================================

    loss_components = {
        "loss": []
    }

    list_loss_regul = []

    training_start_time = time.time()

    (
        metrics_log_path,
        nnr_pearson_log_path,
        frame_sampling,
        profiler_trace_dir,
    ) = init_training_runtime(
        log_dir=log_dir,
        sim=sim,
        training=training,
    )

    _profiling = training.profiling
    _profiler_trace_dir = profiler_trace_dir

    embedding_frozen = False
    unfreeze_at_iteration = -1

    # ---------------------------------------------------------------------
    # torch.compile
    #
    # Deterministic scatter_add prevents the CUDA-graph portion of
    # reduce-overhead. Use normal "default" compilation in that case.
    # ---------------------------------------------------------------------

    if training.torch_compile:

        _ckpt = bool(
            getattr(
                model,
                "grad_checkpoint",
                False,
            )
        )

        _use_cudagraphs = (
            not _ckpt
            and not training.deterministic
        )

        _mode = (
            "reduce-overhead"
            if _use_cudagraphs
            else "default"
        )

        model = torch.compile(
            model,
            mode=_mode,
            fullgraph=not _ckpt,
        )

        _reg_mode = (
            "reduce-overhead"
            if _use_cudagraphs
            else "default"
        )

        regularizer.compute = torch.compile(
            regularizer.compute,
            mode=_reg_mode,
            fullgraph=True,
        )

        regularizer.compute_update_regul = (
            torch.compile(
                regularizer.compute_update_regul,
                mode=_reg_mode,
                fullgraph=True,
            )
        )

        logger.info(
            "torch.compile enabled "
            f"(mode={_mode}, "
            f"regularizer_mode={_reg_mode}, "
            f"grad_checkpoint={_ckpt}, "
            f"deterministic={training.deterministic})"
        )

    else:

        logger.info(
            "torch.compile disabled via config "
            "(torch_compile: false)"
        )

    # =====================================================================
    # EPOCH LOOP
    # =====================================================================

    for epoch in range(
        start_epoch,
        training.n_epochs,
    ):

        # -----------------------------------------------------------------
        # Number of iterations
        # -----------------------------------------------------------------

        Niter = int(
            sim.n_frames
            * training.data_augmentation_loop
            // training.batch_size
            * 0.2
        )

        if training.max_iterations_per_epoch > 0:
            Niter = min(
                Niter,
                training.max_iterations_per_epoch,
            )

        plot_frequency = max(
            1,
            Niter // 20,
        )

        print(
            f"every {max(1, Niter // 20)} iterations: "
            f"{Niter} iterations per epoch, "
            f"plot "
            f"(early-phase every "
            f"{max(1, (Niter // 20) // 5)} iterations)"
        )

        # -----------------------------------------------------------------
        # Epoch state
        #
        # IMPORTANT:
        #   total_loss_gpu and total_regul_gpu are created here.
        #   They reset every epoch.
        # -----------------------------------------------------------------

        epoch_state = init_epoch_state(
            epoch=epoch,
            n_iter=Niter,
            training=training,
            frame_sampling=frame_sampling,
            embedding_frozen=embedding_frozen,
            regularizer=regularizer,
            device=device,
        )

        plot_frequency = (
            epoch_state.plot_frequency
        )

        connectivity_plot_frequency = (
            epoch_state.connectivity_plot_frequency
        )

        early_r2_frequency = (
            epoch_state.early_r2_frequency
        )

        plot_iterations = (
            epoch_state.plot_iterations
        )

        frame_indices = (
            epoch_state.frame_indices
        )

        loss_noise_level = (
            epoch_state.loss_noise_level
        )

        dale_enabled = (
            epoch_state.dale_enabled
        )

        dale_checkpoints = (
            epoch_state.dale_checkpoints
        )

        metrics = (
            epoch_state.metrics
        )

        # -----------------------------------------------------------------
        # Measurement-noise resampling
        # -----------------------------------------------------------------

        if (
            training.resample_noise_per_epoch
            and sim.measurement_noise_level > 0
            and x_ts.noise is not None
        ):
            noise_generator = (
                torch.Generator(
                    device=x_ts.noise.device
                ).manual_seed(
                    int(sim.seed) + int(epoch)
                )
            )

            x_ts.noise = (
                torch.randn(
                    x_ts.noise.shape,
                    generator=noise_generator,
                    dtype=x_ts.noise.dtype,
                    device=x_ts.noise.device,
                )
                * sim.measurement_noise_level
            )

            _logger.info(
                f"epoch {epoch}: "
                f"resampled measurement noise "
                f"(seed={int(sim.seed) + int(epoch)})"
            )

        # -----------------------------------------------------------------
        # Alternate training
        # -----------------------------------------------------------------

        if (
            training.alternate_training
            and epoch >= 1
        ):
            phase_mult = (
                training.alternate_lr_ratio
            )

            optimizer, n_total_params = (
                set_trainable_parameters(
                    model=model,
                    lr_embedding=(
                        lr_embedding
                        * phase_mult
                    ),
                    lr=(
                        lr
                        * phase_mult
                    ),
                    lr_update=(
                        lr_update
                        * phase_mult
                    ),
                    lr_W=(
                        lr_W
                        * phase_mult
                    ),
                    lr_NNR_f=lr_NNR_f,
                )
            )

            lr_scheduler = (
                build_lr_scheduler(
                    optimizer,
                    config,
                )
            )

            _logger.info(
                "Phase 1 (SIREN focus): "
                f"W/MLP LRs *= {phase_mult}, "
                f"NNR_f LR = {lr_NNR_f}"
            )

        # -----------------------------------------------------------------
        # NGP injection + LR-damping schedule
        # -----------------------------------------------------------------

        ngp_schedule = (
            init_ngp_schedule(
                training,
                Niter,
            )
        )

        if ngp_schedule.warmup_iter > 0:

            print(
                "NGP binary-inject schedule: "
                f"warmup [0, "
                f"{ngp_schedule.warmup_iter}) "
                "NGP off + nominal LR, "
                f"inject ON at "
                f"{ngp_schedule.warmup_iter}."
            )

            if ngp_schedule.damping_active:

                print(
                    "  LR-damping V on "
                    f"{ngp_schedule.damp_groups}: "
                    f"damp ["
                    f"{ngp_schedule.warmup_iter}, "
                    f"{ngp_schedule.ramp_mid}) "
                    f"base -> "
                    f"base/{ngp_schedule.damping_factor:g}, "
                    f"recover ["
                    f"{ngp_schedule.ramp_mid}, "
                    f"{ngp_schedule.ramp_end}) "
                    f"base/"
                    f"{ngp_schedule.damping_factor:g}"
                    " -> base."
                )

            else:

                print(
                    "  LR-damping V disabled."
                )

        # Base optimizer LRs.
        base_lrs = {
            id(param_group): param_group["base_lr"]
            for param_group in optimizer.param_groups
        }

        previous_lr_multiplier = 1.0
        previous_injection_active = None

        # -----------------------------------------------------------------
        # Profiling
        # -----------------------------------------------------------------

        pbar = trange(
            Niter,
            ncols=150,
        )

        if _profiling:

            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=3,
                    warmup=2,
                    active=3,
                    repeat=1,
                ),
                on_trace_ready=(
                    torch.profiler.tensorboard_trace_handler(
                        _profiler_trace_dir,
                        use_gzip=True,
                    )
                ),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            )

            profiler.start()

        # =================================================================
        # ITERATION LOOP
        # =================================================================

        for N in pbar:

            # -------------------------------------------------------------
            # NGP injection state
            # -------------------------------------------------------------

            injection_active = (
                ngp_schedule.warmup_iter <= 0
                or N >= ngp_schedule.warmup_iter
            )

            # -------------------------------------------------------------
            # NGP LR damping
            # -------------------------------------------------------------

            if ngp_schedule.damping_active:

                if (
                    N < ngp_schedule.warmup_iter
                    or N >= ngp_schedule.ramp_end
                ):
                    lr_multiplier = 1.0

                elif N < ngp_schedule.ramp_mid:

                    progress = (
                        float(
                            N
                            - ngp_schedule.warmup_iter
                        )
                        / float(
                            ngp_schedule.ramp_iter
                        )
                    )

                    lr_multiplier = (
                        1.0
                        + (
                            1.0
                            / ngp_schedule.damping_factor
                            - 1.0
                        )
                        * progress
                    )

                else:

                    progress = (
                        float(
                            N
                            - ngp_schedule.ramp_mid
                        )
                        / float(
                            ngp_schedule.ramp_iter
                        )
                    )

                    lr_multiplier = (
                        1.0
                        / ngp_schedule.damping_factor
                        + (
                            1.0
                            - 1.0
                            / ngp_schedule.damping_factor
                        )
                        * progress
                    )

            else:

                lr_multiplier = 1.0

            if (
                ngp_schedule.damping_active
                and lr_multiplier
                != previous_lr_multiplier
            ):

                for param_group in optimizer.param_groups:

                    if (
                        param_group.get("name")
                        in ngp_schedule.damp_groups
                    ):
                        param_group["lr"] = (
                            base_lrs[
                                id(param_group)
                            ]
                            * lr_multiplier
                        )

                previous_lr_multiplier = (
                    lr_multiplier
                )

            # -------------------------------------------------------------
            # NGP stage transitions
            # -------------------------------------------------------------

            if (
                ngp_schedule.warmup_iter > 0
                and previous_injection_active is False
                and injection_active
            ):

                if ngp_schedule.damping_active:

                    print(
                        f"\n[NGP inject] "
                        f"iter {N}: "
                        "phase 1 -> phase 2 "
                        "(NGP hard-on; "
                        "GNN-LR V-schedule starts)."
                    )

                else:

                    print(
                        f"\n[NGP inject] "
                        f"iter {N}: "
                        "phase 1 -> phase 2 "
                        "(NGP hard-on)."
                    )

            elif (
                ngp_schedule.damping_active
                and N == ngp_schedule.ramp_mid
                and ngp_schedule.warmup_iter > 0
            ):

                print(
                    f"\n[NGP inject] "
                    f"iter {N}: "
                    "LR damping -> recovery."
                )

            elif (
                ngp_schedule.damping_active
                and N == ngp_schedule.ramp_end
                and ngp_schedule.warmup_iter > 0
            ):

                print(
                    f"\n[NGP inject] "
                    f"iter {N}: "
                    "GNN LR back to nominal."
                )

            previous_injection_active = (
                injection_active
            )

            # -------------------------------------------------------------
            # Embedding unfreeze
            # -------------------------------------------------------------

            if (
                embedding_frozen
                and N
                == epoch_state.unfreeze_at_iteration
            ):

                embedding_frozen = False

                lr_embedding = (
                    training.lr_embedding
                )

                optimizer, n_total_params = (
                    set_trainable_parameters(
                        model=model,
                        lr_embedding=lr_embedding,
                        lr=lr,
                        lr_update=lr_update,
                        lr_W=lr_W,
                    )
                )

                _logger.debug(
                    f"unfreezing embedding at "
                    f"iteration {N}/{Niter}"
                )

            optimizer.zero_grad()

            # =============================================================
            # RECURRENT TRAINING
            # =============================================================

            if (
                training.recurrent_training
                and not training.neural_ODE_training
            ):

                loss, regul_val = (
                    recurrent_loss(
                        model=model,
                        x_ts=x_ts,
                        y_ts=y_ts,
                        edges=edges,
                        ids=visible_ids,
                        frame_indices=frame_indices,
                        iter_idx=N,
                        config=config,
                        device=device,
                        xnorm=xnorm,
                        ynorm=ynorm,
                        regularizer=regularizer,
                        has_visual_field=(
                            train.has_visual_field
                        ),
                        hidden_ids=hidden_ids,
                    )
                )

                loss.backward()

                if (
                    hasattr(
                        training,
                        "grad_clip_W",
                    )
                    and training.grad_clip_W > 0
                    and hasattr(model, "W")
                    and model.W.grad is not None
                ):
                    torch.nn.utils.clip_grad_norm_(
                        [model.W],
                        max_norm=(
                            training.grad_clip_W
                        ),
                    )

                optimizer.step()

                if (
                    dale_enabled
                    and N in dale_checkpoints
                ):
                    enforce_dale_law(
                        model,
                        edges,
                    )

                lr_scheduler.step()

                _total_loss_gpu = (
                    _total_loss_gpu
                    + loss.detach()
                )

                total_loss_regul += (
                    regul_val
                )

                regularizer.finalize_iteration()

                if regularizer.should_record():

                    current_loss = (
                        loss.item()
                    )

                    loss_components[
                        "loss"
                    ].append(
                        (
                            current_loss
                            - regul_val
                        )
                        / n_neurons
                    )

                    plot_dict = {
                        **regularizer.get_history(),
                        "loss": (
                            loss_components[
                                "loss"
                            ]
                        ),
                    }

                    plot_signal_loss(
                        plot_dict,
                        log_dir,
                        epoch=epoch,
                        Niter=Niter,
                        epoch_boundaries=(
                            regularizer
                            .epoch_boundaries
                        ),
                    )

                is_regular_r2 = (
                    N % connectivity_plot_frequency
                    == 0
                )

                is_early_r2 = (
                    N < connectivity_plot_frequency
                    and N % early_r2_frequency == 0
                )

                if is_regular_r2 and N > 0:

                    intermediate_path = os.path.join(
                        log_dir,
                        "models",
                        f"best_model_with_"
                        f"{training.n_runs - 1}"
                        f"_graphs_{epoch}.pt",
                    )

                    os.makedirs(
                        os.path.dirname(
                            intermediate_path
                        ),
                        exist_ok=True,
                    )

                    torch.save(
                        {
                            "model_state_dict":
                                model.state_dict(),
                            "optimizer_state_dict":
                                optimizer.state_dict(),
                        },
                        intermediate_path,
                    )

                if (
                    is_regular_r2
                    or is_early_r2
                ) and model_family(model) == "linear":

                    (
                        metrics.connectivity_r2,
                        metrics.tau_r2,
                        metrics.vrest_r2,
                        dynamics,
                    ) = plot_training_linear(
                        model,
                        config,
                        epoch,
                        N,
                        log_dir,
                        device,
                        gt_weights,
                        n_neurons=n_neurons,
                    )

                    metrics.vrest_r2_clean = (
                        dynamics["vrest_r2_clean"]
                    )
                    metrics.tau_r2_clean = (
                        dynamics["tau_r2_clean"]
                    )
                    metrics.n_out_vrest = (
                        dynamics["n_out_vrest"]
                    )
                    metrics.n_total_vrest = (
                        dynamics["n_total_vrest"]
                    )
                    metrics.n_out_tau = (
                        dynamics["n_out_tau"]
                    )
                    metrics.n_total_tau = (
                        dynamics["n_total_tau"]
                    )

                    with open(
                        metrics_log_path,
                        "a",
                    ) as f:
                        f.write(
                            f"{regularizer.iter_count},"
                            f"{metrics.connectivity_r2:.6f},"
                            f"{metrics.vrest_r2:.6f},"
                            f"{metrics.tau_r2:.6f},"
                            f"{format_metric(metrics.hidden_r2)},"
                            f"{format_metric(metrics.anchor_r2)},"
                            f"{format_metric(metrics.vrest_r2_clean)},"
                            f"{metrics.n_out_vrest},"
                            f"{metrics.n_total_vrest},"
                            f"{format_metric(metrics.tau_r2_clean)},"
                            f"{metrics.n_out_tau},"
                            f"{metrics.n_total_tau}\n"
                        )

                    metrics_changed = True

                elif (
                    is_regular_r2
                    or is_early_r2
                ) and model_family(model) == "gnn":

                    (
                        metrics.connectivity_r2,
                        metrics.connectivity_r2_visible,
                        hidden_r2,
                        anchor_r2,
                    ) = plot_training_gnn(
                        x_ts,
                        model,
                        config,
                        epoch,
                        N,
                        log_dir,
                        device,
                        type_list,
                        gt_weights,
                        edges,
                        n_neurons=n_neurons,
                        n_neuron_types=(
                            sim.n_neuron_types
                        ),
                        ode_params=ode_params,
                        hidden_ids=hidden_ids,
                        anchor_ids=anchor_ids,
                    )

                    if hidden_r2 is not None:
                        metrics.hidden_r2 = (
                            hidden_r2
                        )

                    if anchor_r2 is not None:
                        metrics.anchor_r2 = (
                            anchor_r2
                        )

                    dynamics = (
                        compute_dynamics_r2(
                            model,
                            x_ts,
                            config,
                            device,
                            n_neurons,
                        )
                    )

                    metrics.vrest_r2 = (
                        dynamics["vrest_r2"]
                    )
                    metrics.tau_r2 = (
                        dynamics["tau_r2"]
                    )
                    metrics.vrest_r2_clean = (
                        dynamics["vrest_r2_clean"]
                    )
                    metrics.tau_r2_clean = (
                        dynamics["tau_r2_clean"]
                    )
                    metrics.n_out_vrest = (
                        dynamics["n_out_vrest"]
                    )
                    metrics.n_total_vrest = (
                        dynamics["n_total_vrest"]
                    )
                    metrics.n_out_tau = (
                        dynamics["n_out_tau"]
                    )
                    metrics.n_total_tau = (
                        dynamics["n_total_tau"]
                    )

                    with open(
                        metrics_log_path,
                        "a",
                    ) as f:
                        f.write(
                            f"{regularizer.iter_count},"
                            f"{format_metric(metrics.connectivity_r2)},"
                            f"{format_metric(metrics.vrest_r2)},"
                            f"{format_metric(metrics.tau_r2)},"
                            f"{format_metric(metrics.hidden_r2)},"
                            f"{format_metric(metrics.anchor_r2)},"
                            f"{format_metric(metrics.vrest_r2_clean)},"
                            f"{metrics.n_out_vrest},"
                            f"{metrics.n_total_vrest},"
                            f"{format_metric(metrics.tau_r2_clean)},"
                            f"{metrics.n_out_tau},"
                            f"{metrics.n_total_tau}\n"
                        )

                    metrics_changed = True

                else:

                    metrics_changed = False

                # ---------------------------------------------------------
                # Fast NGP Pearson refresh
                # ---------------------------------------------------------

                ngp_quick_updated = False
                hidden_quick_std = None
                anchor_quick_std = None

                if (
                    has_hidden_neurons
                    and getattr(
                        model,
                        "NNR_hidden",
                        None,
                    ) is not None
                    and N > 0
                    and N % _NGP_QUICK_FREQ == 0
                ):

                    hidden_quick, hidden_quick_std = (
                        _quick_ngp_pearson(
                            model,
                            x_ts,
                            hidden_ids,
                            use_anchor=False,
                            device=device,
                            return_stats=True,
                        )
                    )

                    if hidden_quick is not None:

                        metrics.hidden_r2 = (
                            hidden_quick
                        )

                        ngp_quick_updated = True

                    if has_anchor_neurons:

                        anchor_quick, anchor_quick_std = (
                            _quick_ngp_pearson(
                                model,
                                x_ts,
                                anchor_ids,
                                use_anchor=True,
                                device=device,
                                return_stats=True,
                            )
                        )

                        if anchor_quick is not None:

                            metrics.anchor_r2 = (
                                anchor_quick
                            )

                            ngp_quick_updated = True

                if ngp_quick_updated:

                    with open(
                        metrics_log_path,
                        "a",
                    ) as f:

                        f.write(
                            f"{regularizer.iter_count},"
                            f"{format_metric(metrics.connectivity_r2)},"
                            f"{format_metric(metrics.vrest_r2)},"
                            f"{format_metric(metrics.tau_r2)},"
                            f"{format_metric(metrics.hidden_r2)},"
                            f"{format_metric(metrics.anchor_r2)},"
                            f"{format_metric(metrics.vrest_r2_clean)},"
                            f"{metrics.n_out_vrest},"
                            f"{metrics.n_total_vrest},"
                            f"{format_metric(metrics.tau_r2_clean)},"
                            f"{metrics.n_out_tau},"
                            f"{metrics.n_total_tau}\n"
                        )

                    with open(
                        nnr_pearson_log_path,
                        "a",
                    ) as f:

                        f.write(
                            f"{regularizer.iter_count},"
                            f"{format_metric(metrics.hidden_r2)},"
                            f"{format_metric(hidden_quick_std)},"
                            f"{format_metric(metrics.anchor_r2)},"
                            f"{format_metric(anchor_quick_std)}\n"
                        )

                    metrics_changed = True

                if metrics_changed:

                    plot_metrics(
                        log_dir,
                        epoch_boundaries=(
                            regularizer
                            .epoch_boundaries
                        ),
                        ngp_stages=(
                            ngp_schedule.stages
                        ),
                    )

                # ---------------------------------------------------------
                # Progress bar
                # ---------------------------------------------------------

                if (
                    metrics.connectivity_r2
                    is not None
                    or metrics.hidden_r2
                    is not None
                ):

                    bar_parts = []

                    if (
                        metrics.connectivity_r2
                        is not None
                    ):

                        conn_color = r2_color(
                            metrics.connectivity_r2
                        )

                        if (
                            metrics.connectivity_r2_visible
                            is not None
                            and abs(
                                metrics.connectivity_r2_visible
                                - metrics.connectivity_r2
                            ) > 1e-4
                        ):

                            conn_string = (
                                f"conn="
                                f"{metrics.connectivity_r2:.3f}"
                                f"("
                                f"{metrics.connectivity_r2_visible:.3f}"
                                f")"
                            )

                        else:

                            conn_string = (
                                f"conn="
                                f"{metrics.connectivity_r2:.3f}"
                            )

                        bar_parts.append(
                            f"{conn_color}"
                            f"{conn_string}"
                            f"{ANSI_RESET}"
                        )

                        if ode_params.has_vrest():

                            vr_pct = (
                                100.0
                                * metrics.n_out_vrest
                                / metrics.n_total_vrest
                                if metrics.n_total_vrest > 0
                                else 0.0
                            )

                            bar_parts.append(
                                f"{r2_color(metrics.vrest_r2_clean)}"
                                f"Vr="
                                f"{fmt_r2_bar(metrics.vrest_r2_clean)}"
                                f"({vr_pct:.0f}%)"
                                f"{ANSI_RESET}"
                            )

                        if ode_params.has_tau():

                            tau_pct = (
                                100.0
                                * metrics.n_out_tau
                                / metrics.n_total_tau
                                if metrics.n_total_tau > 0
                                else 0.0
                            )

                            bar_parts.append(
                                f"{r2_color(metrics.tau_r2_clean)}"
                                f"τ="
                                f"{fmt_r2_bar(metrics.tau_r2_clean)}"
                                f"({tau_pct:.0f}%)"
                                f"{ANSI_RESET}"
                            )

                    if (
                        metrics.hidden_r2
                        is not None
                        or metrics.anchor_r2
                        is not None
                    ):

                        if not injection_active:

                            if (
                                metrics.anchor_r2
                                is not None
                            ):
                                nnr_string = (
                                    f"nnr=n/a"
                                    f"({metrics.anchor_r2:.3f})"
                                )
                            else:
                                nnr_string = "nnr=n/a"

                            bar_parts.append(
                                nnr_string
                            )

                        elif metrics.hidden_r2 is not None:

                            if (
                                metrics.anchor_r2
                                is not None
                            ):

                                nnr_string = (
                                    f"nnr="
                                    f"{metrics.hidden_r2:.3f}"
                                    f"("
                                    f"{metrics.anchor_r2:.3f}"
                                    f")"
                                )

                            else:

                                nnr_string = (
                                    f"nnr="
                                    f"{metrics.hidden_r2:.3f}"
                                )

                            bar_parts.append(
                                f"{r2_color(metrics.hidden_r2)}"
                                f"{nnr_string}"
                                f"{ANSI_RESET}"
                            )

                    if bar_parts:
                        pbar.set_postfix_str(
                            " ".join(bar_parts)
                        )

                continue

            # =============================================================
            # STANDARD / GNN TRAINING
            # =============================================================

            state_batch = []
            y_list = []
            ids_list = []
            k_list = []
            visual_input_list = []

            ids_index = 0

            loss = torch.zeros(
                (),
                device=device,
            )

            regularizer.reset_iteration(
                device=device
            )

            # -------------------------------------------------------------
            # Consecutive batch
            # -------------------------------------------------------------

            if training.consecutive_batch:

                k_start = int(
                    frame_indices[
                        N * training.batch_size
                    ]
                )

            for batch in range(
                training.batch_size
            ):

                if training.consecutive_batch:

                    k = (
                        k_start
                        + batch
                    )

                else:

                    k = int(
                        frame_indices[
                            N * training.batch_size
                            + batch
                        ]
                    )

                x = x_ts.frame(k)

                # ---------------------------------------------------------
                # Measurement noise
                # ---------------------------------------------------------

                if (
                    x.noise is not None
                    and sim.measurement_noise_level > 0
                ):
                    x.voltage = (
                        x.voltage
                        + x.noise
                    )

                # ---------------------------------------------------------
                # Hidden neurons
                # ---------------------------------------------------------

                if has_hidden_neurons:

                    _inject_hidden_voltage(
                        model,
                        x,
                        k,
                        hidden_ids,
                        injection_active,
                    )

                # ---------------------------------------------------------
                # Temporal window
                # ---------------------------------------------------------

                if training.time_window > 0:

                    x_temporal = (
                        x_ts.voltage[
                            k
                            - training.time_window
                            + 1:
                            k + 1
                        ].T
                    )

                    # x stays as NeuronState;
                    # x_temporal is passed separately if needed.

                # ---------------------------------------------------------
                # Visual field
                # ---------------------------------------------------------

                if train.has_visual_field:

                    visual_input = (
                        model.forward_visual(
                            x,
                            k,
                        )
                    )

                    x.stimulus[
                        :model.n_input_neurons
                    ] = (
                        visual_input.squeeze(-1)
                    )

                    x.stimulus[
                        model.n_input_neurons:
                    ] = 0

                # ---------------------------------------------------------
                # Regularization
                # ---------------------------------------------------------

                if batch == 0:

                    regul_loss = (
                        regularizer.compute(
                            model=model,
                            x=x,
                            in_features=None,
                            ids=ids,
                            ids_batch=None,
                            edges=edges,
                            device=device,
                            xnorm=xnorm,
                        )
                    )

                    loss = (
                        loss
                        + regul_loss
                    )

                # ---------------------------------------------------------
                # Target
                # ---------------------------------------------------------

                if (
                    training.recurrent_training
                    or training.neural_ODE_training
                ):

                    target_frame = (
                        k + 1
                        if (
                            _stride_subsample
                        )
                        else (
                            k
                            + training.time_step
                        )
                    )

                    y = (
                        x_ts.voltage[
                            target_frame
                        ].unsqueeze(-1)
                    )

                elif train.test_neural_field:

                    y = (
                        x_ts.stimulus[
                            k,
                            :sim.n_input_neurons
                        ].unsqueeze(-1)
                    )

                else:

                    y = (
                        y_ts_gpu[k]
                        / ynorm
                    )

                if loss_noise_level > 0:

                    y = (
                        y
                        + torch.randn(
                            y.shape,
                            device=device,
                        )
                        * loss_noise_level
                    )

                # ---------------------------------------------------------
                # Accumulate batch
                # ---------------------------------------------------------

                state_batch.append(x)

                n = x.n_neurons

                y_list.append(y)

                ids_list.append(
                    visible_ids
                    + ids_index
                )

                k_list.append(
                    torch.ones(
                        (n, 1),
                        dtype=torch.int,
                        device=device,
                    )
                    * k
                )

                if train.test_neural_field:

                    visual_input_list.append(
                        visual_input
                    )

                ids_index += n

            # -----------------------------------------------------------------
            # Batch assembly
            # -----------------------------------------------------------------

            data_id = torch.zeros(
                (ids_index, 1),
                dtype=torch.int,
                device=device,
            )

            y_batch = torch.cat(
                y_list,
                dim=0,
            )

            ids_batch = torch.cat(
                ids_list,
                dim=0,
            )

            k_batch = torch.cat(
                k_list,
                dim=0,
            )

            epoch_state.total_regul_gpu = (
                epoch_state.total_regul_gpu
                + loss.detach()
            )

            # -----------------------------------------------------------------
            # Visual-field testing
            # -----------------------------------------------------------------

            if train.test_neural_field:

                visual_input_batch = (
                    torch.cat(
                        visual_input_list,
                        dim=0,
                    )
                )

                loss = (
                    loss
                    + (
                        visual_input_batch
                        - y_batch
                    ).norm(2)
                )

            # -----------------------------------------------------------------
            # MLP ODE
            # -----------------------------------------------------------------

            elif (
                "mlp_ode"
                in model_config.signal_model_name.lower()
            ):

                batched_state, _ = (
                    _batch_frames(
                        state_batch,
                        edges,
                    )
                )

                batched_x = (
                    batched_state.to_packed()
                )

                pred = model(
                    batched_x,
                    data_id=data_id,
                    return_all=False,
                )

                loss = (
                    loss
                    + (
                        pred[ids_batch]
                        - y_batch[ids_batch]
                    ).norm(2)
                )

            # -----------------------------------------------------------------
            # MLP
            # -----------------------------------------------------------------

            elif (
                "mlp"
                in model_config.signal_model_name.lower()
            ):

                batched_state, _ = (
                    _batch_frames(
                        state_batch,
                        edges,
                    )
                )

                pred = model(
                    batched_state,
                    data_id=data_id,
                    return_all=False,
                )

                loss = (
                    loss
                    + (
                        pred[ids_batch]
                        - y_batch[ids_batch]
                    ).norm(2)
                )

            # -----------------------------------------------------------------
            # GNN
            # -----------------------------------------------------------------

            else:

                batched_state, batched_edges = (
                    _batch_frames(
                        state_batch,
                        edges,
                    )
                )

                (
                    pred,
                    in_features,
                    msg,
                ) = model(
                    batched_state,
                    batched_edges,
                    data_id=data_id,
                    return_all=True,
                )

                update_regul = (
                    regularizer.compute_update_regul(
                        model,
                        in_features,
                        ids_batch,
                        device,
                    )
                )

                loss = (
                    loss
                    + update_regul
                )

                # -------------------------------------------------------------
                # Neural ODE
                # -------------------------------------------------------------

                if training.neural_ODE_training:

                    ode_state_clamp = (
                        getattr(
                            training,
                            "ode_state_clamp",
                            10.0,
                        )
                    )

                    ode_stab_lambda = (
                        getattr(
                            training,
                            "ode_stab_lambda",
                            0.0,
                        )
                    )

                    ode_loss, pred_x = (
                        neural_ode_loss(
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
                            has_visual_field=(
                                train.has_visual_field
                            ),
                            y_batch=y_batch,
                            noise_level=(
                                training.noise_recurrent_level
                            ),
                            ode_method=training.ode_method,
                            rtol=training.ode_rtol,
                            atol=training.ode_atol,
                            adjoint=training.ode_adjoint,
                            iteration=N,
                            state_clamp=(
                                ode_state_clamp
                            ),
                            stab_lambda=(
                                ode_stab_lambda
                            ),
                        )
                    )

                    loss = (
                        loss
                        + ode_loss
                    )

                # -------------------------------------------------------------
                # Recurrent GNN
                # -------------------------------------------------------------

                elif training.recurrent_training:

                    pred_x = (
                        batched_state.voltage.unsqueeze(-1)
                        + sim.delta_t
                        * pred
                        + training.noise_recurrent_level
                        * torch.randn_like(pred)
                    )

                    if training.time_step > 1:

                        for step in range(
                            training.time_step - 1
                        ):

                            neurons_per_sample = (
                                state_batch[0].n_neurons
                            )

                            for b in range(
                                training.batch_size
                            ):

                                start_idx = (
                                    b
                                    * neurons_per_sample
                                )

                                end_idx = (
                                    (b + 1)
                                    * neurons_per_sample
                                )

                                state_batch[b].voltage = (
                                    pred_x[
                                        start_idx:end_idx
                                    ].squeeze()
                                )

                                if has_hidden_neurons:

                                    state_batch[
                                        b
                                    ].voltage[
                                        hidden_ids
                                    ] = 0.0

                                k_current = (
                                    k_batch[
                                        start_idx,
                                        0,
                                    ].item()
                                    + step
                                    + 1
                                )

                                if (
                                    train.has_visual_field
                                ):

                                    visual_input_next = (
                                        model.forward_visual(
                                            state_batch[b],
                                            k_current,
                                        )
                                    )

                                    state_batch[
                                        b
                                    ].stimulus[
                                        :model.n_input_neurons
                                    ] = (
                                        visual_input_next
                                        .squeeze(-1)
                                    )

                                    state_batch[
                                        b
                                    ].stimulus[
                                        model.n_input_neurons:
                                    ] = 0

                                else:

                                    x_next = (
                                        x_ts.frame(
                                            k_current
                                        )
                                    )

                                    state_batch[
                                        b
                                    ].stimulus = (
                                        x_next.stimulus
                                    )

                                    if (
                                        x_next.optogenetics_stimulus
                                        is not None
                                    ):

                                        state_batch[
                                            b
                                        ].optogenetics_stimulus = (
                                            x_next
                                            .optogenetics_stimulus
                                        )

                            (
                                batched_state,
                                batched_edges,
                            ) = _batch_frames(
                                state_batch,
                                edges,
                            )

                            (
                                pred,
                                in_features,
                                msg,
                            ) = model(
                                batched_state,
                                batched_edges,
                                data_id=data_id,
                                return_all=True,
                            )

                            pred_x = (
                                pred_x
                                + sim.delta_t
                                * pred
                                + training.noise_recurrent_level
                                * torch.randn_like(
                                    pred
                                )
                            )

                    loss = (
                        loss
                        + (
                            (
                                pred_x[
                                    ids_batch
                                ]
                                - y_batch[
                                    ids_batch
                                ]
                            )
                            / (
                                sim.delta_t
                                * training.time_step
                            )
                        ).norm(2)
                    )

                # -------------------------------------------------------------
                # Standard one-step GNN loss
                # -------------------------------------------------------------

                else:

                    loss = (
                        loss
                        + (
                            pred[ids_batch]
                            - y_batch[ids_batch]
                        ).norm(2)
                    )

                    # Hidden self-consistency loss intentionally removed.
                    #
                    # NGP-hidden is supervised through:
                    #   1. anchor voltage loss
                    #   2. backpropagation through hidden-voltage injection
                    #
                    # This preserves the behavior of the current implementation.

                    if (
                        has_anchor_neurons
                        and getattr(
                            training,
                            "coeff_anchor_voltage",
                            0.0,
                        ) > 0
                    ):

                        n_per = (
                            state_batch[
                                0
                            ].n_neurons
                        )

                        k_starts = (
                            k_batch[
                                ::n_per,
                                0,
                            ].to(torch.long)
                        )

                        pred_a = (
                            model.forward_anchor_batched(
                                k_starts,
                                anchor_ids=anchor_ids,
                            )
                        )

                        gt_a = (
                            x_ts.voltage[
                                k_starts[:, None],
                                anchor_ids[None, :],
                            ]
                        )

                        loss = (
                            loss
                            + training.coeff_anchor_voltage
                            * (
                                pred_a
                                - gt_a
                            ).norm(2)
                        )

            # =============================================================
            # BACKWARD / STEP
            # =============================================================

            loss.backward()

            if (
                training.neural_ODE_training
                and N % 500 == 0
            ):
                debug_check_gradients(
                    model,
                    loss,
                    N,
                )

            if (
                hasattr(
                    training,
                    "grad_clip_W",
                )
                and training.grad_clip_W > 0
                and hasattr(model, "W")
                and model.W.grad is not None
            ):

                torch.nn.utils.clip_grad_norm_(
                    [model.W],
                    max_norm=training.grad_clip_W,
                )

            optimizer.step()

            if (
                dale_enabled
                and N in dale_checkpoints
            ):
                enforce_dale_law(
                    model,
                    edges,
                )

            lr_scheduler.step()

            epoch_state.total_loss_gpu += loss.detach()

            epoch_state.total_regul_gpu += (
                regularizer
                .get_iteration_total_tensor()
                .detach()
            )

            regularizer.finalize_iteration()

            # -------------------------------------------------------------
            # Loss recording
            # -------------------------------------------------------------

            if regularizer.should_record():

                current_loss = (
                    loss.item()
                )

                regul_total_this_iter = (
                    regularizer
                    .get_iteration_total()
                )

                loss_components[
                    "loss"
                ].append(
                    (
                        current_loss
                        - regul_total_this_iter
                    )
                    / n_neurons
                )

                plot_dict = {
                    **regularizer.get_history(),
                    "loss": (
                        loss_components[
                            "loss"
                        ]
                    ),
                }

                plot_signal_loss(
                    plot_dict,
                    log_dir,
                    epoch=epoch,
                    Niter=Niter,
                    epoch_boundaries=(
                        regularizer
                        .epoch_boundaries
                    ),
                    debug=False,
                    current_loss=(
                        current_loss
                        / n_neurons
                    ),
                    current_regul=(
                        regul_total_this_iter
                        / n_neurons
                    ),
                    total_loss=(
                        epoch_state.total_loss_gpu
                    ),
                    total_loss_regul=(
                        epoch_state.total_regul_gpu
                    ),
                )

                torch.save(
                    {
                        **plot_dict,
                        "epoch_boundaries": list(
                            regularizer
                            .epoch_boundaries
                        ),
                    },
                    os.path.join(
                        log_dir,
                        "loss_components.pt",
                    ),
                )

                if training.save_all_checkpoints:

                    torch.save(
                        {
                            "model_state_dict":
                                model.state_dict(),
                            "optimizer_state_dict":
                                optimizer.state_dict(),
                        },
                        os.path.join(
                            log_dir,
                            "models",
                            f"best_model_with_"
                            f"{training.n_runs - 1}"
                            f"_graphs_{epoch}_{N}.pt",
                        ),
                    )

            # =============================================================
            # R2 / DYNAMICS CHECKPOINT
            # =============================================================

            is_regular_r2 = (
                N > 0
                and
                N % connectivity_plot_frequency == 0
            )

            is_early_r2 = (
                N < connectivity_plot_frequency
                and
                N % early_r2_frequency == 0
            )

            if (
                is_regular_r2
                and model_family(model) == "mlp"
                and not train.test_neural_field
            ):

                from connectome_gnn.metrics import (
                    compute_jacobian_connectivity_r2,
                )

                metrics.connectivity_r2 = (
                    compute_jacobian_connectivity_r2(
                        model,
                        x_ts,
                        ode_params,
                        n_neurons=n_neurons,
                        device=device,
                    )
                )

                metrics.tau_r2 = 0.0
                metrics.vrest_r2 = 0.0

                with open(
                    metrics_log_path,
                    "a",
                ) as f:

                    f.write(
                        f"{regularizer.iter_count},"
                        f"{format_metric(metrics.connectivity_r2)},"
                        f"{format_metric(metrics.vrest_r2)},"
                        f"{format_metric(metrics.tau_r2)},"
                        f"{format_metric(metrics.hidden_r2)},"
                        f"{format_metric(metrics.anchor_r2)},"
                        f"nan,0,0,nan,0,0\n"
                    )

                plot_jacobian_w_scatter(
                    model,
                    x_ts,
                    ode_params,
                    gt_weights,
                    n_neurons,
                    log_dir,
                    epoch,
                    N,
                    device,
                )

                metrics_changed = True

            elif (
                (is_regular_r2 or is_early_r2)
                and not train.test_neural_field
                and model_family(model) == "linear"
            ):

                (
                    metrics.connectivity_r2,
                    metrics.tau_r2,
                    metrics.vrest_r2,
                    dynamics,
                ) = plot_training_linear(
                    model,
                    config,
                    epoch,
                    N,
                    log_dir,
                    device,
                    gt_weights,
                    n_neurons=n_neurons,
                )

                metrics.vrest_r2_clean = (
                    dynamics["vrest_r2_clean"]
                )

                metrics.tau_r2_clean = (
                    dynamics["tau_r2_clean"]
                )

                metrics.n_out_vrest = (
                    dynamics["n_out_vrest"]
                )

                metrics.n_total_vrest = (
                    dynamics["n_total_vrest"]
                )

                metrics.n_out_tau = (
                    dynamics["n_out_tau"]
                )

                metrics.n_total_tau = (
                    dynamics["n_total_tau"]
                )

                with open(
                    metrics_log_path,
                    "a",
                ) as f:

                    f.write(
                        f"{regularizer.iter_count},"
                        f"{format_metric(metrics.connectivity_r2)},"
                        f"{format_metric(metrics.vrest_r2)},"
                        f"{format_metric(metrics.tau_r2)},"
                        f"{format_metric(metrics.hidden_r2)},"
                        f"{format_metric(metrics.anchor_r2)},"
                        f"{format_metric(metrics.vrest_r2_clean)},"
                        f"{metrics.n_out_vrest},"
                        f"{metrics.n_total_vrest},"
                        f"{format_metric(metrics.tau_r2_clean)},"
                        f"{metrics.n_out_tau},"
                        f"{metrics.n_total_tau}\n"
                    )

                metrics_changed = True

            elif (
                (is_regular_r2 or is_early_r2)
                and not train.test_neural_field
                and model_family(model) == "gnn"
            ):

                (
                    metrics.connectivity_r2,
                    metrics.connectivity_r2_visible,
                    hidden_r2,
                    anchor_r2,
                ) = plot_training_gnn(
                    x_ts,
                    model,
                    config,
                    epoch,
                    N,
                    log_dir,
                    device,
                    type_list,
                    gt_weights,
                    edges,
                    n_neurons=n_neurons,
                    n_neuron_types=(
                        sim.n_neuron_types
                    ),
                    ode_params=ode_params,
                    hidden_ids=hidden_ids,
                    anchor_ids=anchor_ids,
                )

                if hidden_r2 is not None:
                    metrics.hidden_r2 = hidden_r2

                if anchor_r2 is not None:
                    metrics.anchor_r2 = anchor_r2

                # ---------------------------------------------------------
                # KEEP HH/V_rest/tau diagnostics in Step 1.
                # ---------------------------------------------------------

                dynamics = (
                    compute_dynamics_r2(
                        model,
                        x_ts,
                        config,
                        device,
                        n_neurons,
                    )
                )

                metrics.vrest_r2 = (
                    dynamics["vrest_r2"]
                )

                metrics.tau_r2 = (
                    dynamics["tau_r2"]
                )

                metrics.vrest_r2_clean = (
                    dynamics["vrest_r2_clean"]
                )

                metrics.tau_r2_clean = (
                    dynamics["tau_r2_clean"]
                )

                metrics.n_out_vrest = (
                    dynamics["n_out_vrest"]
                )

                metrics.n_total_vrest = (
                    dynamics["n_total_vrest"]
                )

                metrics.n_out_tau = (
                    dynamics["n_out_tau"]
                )

                metrics.n_total_tau = (
                    dynamics["n_total_tau"]
                )

                with open(
                    metrics_log_path,
                    "a",
                ) as f:

                    f.write(
                        f"{regularizer.iter_count},"
                        f"{format_metric(metrics.connectivity_r2)},"
                        f"{format_metric(metrics.vrest_r2)},"
                        f"{format_metric(metrics.tau_r2)},"
                        f"{format_metric(metrics.hidden_r2)},"
                        f"{format_metric(metrics.anchor_r2)},"
                        f"{format_metric(metrics.vrest_r2_clean)},"
                        f"{metrics.n_out_vrest},"
                        f"{metrics.n_total_vrest},"
                        f"{format_metric(metrics.tau_r2_clean)},"
                        f"{metrics.n_out_tau},"
                        f"{metrics.n_total_tau}\n"
                    )

                metrics_changed = True

            else:

                metrics_changed = False

            # -------------------------------------------------------------
            # Fast NGP Pearson refresh
            # -------------------------------------------------------------

            ngp_quick_updated = False
            hidden_quick_std = None
            anchor_quick_std = None

            if (
                has_hidden_neurons
                and getattr(
                    model,
                    "NNR_hidden",
                    None,
                ) is not None
                and N > 0
                and N % _NGP_QUICK_FREQ == 0
            ):

                hidden_quick, hidden_quick_std = (
                    _quick_ngp_pearson(
                        model,
                        x_ts,
                        hidden_ids,
                        use_anchor=False,
                        device=device,
                        return_stats=True,
                    )
                )

                if hidden_quick is not None:

                    metrics.hidden_r2 = (
                        hidden_quick
                    )

                    ngp_quick_updated = True

                if has_anchor_neurons:

                    anchor_quick, anchor_quick_std = (
                        _quick_ngp_pearson(
                            model,
                            x_ts,
                            anchor_ids,
                            use_anchor=True,
                            device=device,
                            return_stats=True,
                        )
                    )

                    if anchor_quick is not None:

                        metrics.anchor_r2 = (
                            anchor_quick
                        )

                        ngp_quick_updated = True

            if ngp_quick_updated:

                with open(
                    metrics_log_path,
                    "a",
                ) as f:

                    f.write(
                        f"{regularizer.iter_count},"
                        f"{format_metric(metrics.connectivity_r2)},"
                        f"{format_metric(metrics.vrest_r2)},"
                        f"{format_metric(metrics.tau_r2)},"
                        f"{format_metric(metrics.hidden_r2)},"
                        f"{format_metric(metrics.anchor_r2)},"
                        f"{format_metric(metrics.vrest_r2_clean)},"
                        f"{metrics.n_out_vrest},"
                        f"{metrics.n_total_vrest},"
                        f"{format_metric(metrics.tau_r2_clean)},"
                        f"{metrics.n_out_tau},"
                        f"{metrics.n_total_tau}\n"
                    )

                with open(
                    nnr_pearson_log_path,
                    "a",
                ) as f:

                    f.write(
                        f"{regularizer.iter_count},"
                        f"{format_metric(metrics.hidden_r2)},"
                        f"{format_metric(hidden_quick_std)},"
                        f"{format_metric(metrics.anchor_r2)},"
                        f"{format_metric(anchor_quick_std)}\n"
                    )

                metrics_changed = True

            if metrics_changed:

                plot_metrics(
                    log_dir,
                    epoch_boundaries=(
                        regularizer
                        .epoch_boundaries
                    ),
                    ngp_stages=(
                        ngp_schedule.stages
                    ),
                )

            # -------------------------------------------------------------
            # Progress bar
            # -------------------------------------------------------------

            if (
                metrics.connectivity_r2 is not None
                or metrics.hidden_r2 is not None
            ):

                bar_parts = []

                if (
                    metrics.connectivity_r2
                    is not None
                ):

                    conn_color = r2_color(
                        metrics.connectivity_r2
                    )

                    if (
                        metrics.connectivity_r2_visible
                        is not None
                        and abs(
                            metrics.connectivity_r2_visible
                            - metrics.connectivity_r2
                        ) > 1e-4
                    ):

                        conn_string = (
                            f"conn="
                            f"{metrics.connectivity_r2:.3f}"
                            f"("
                            f"{metrics.connectivity_r2_visible:.3f}"
                            f")"
                        )

                    else:

                        conn_string = (
                            f"conn="
                            f"{metrics.connectivity_r2:.3f}"
                        )

                    bar_parts.append(
                        f"{conn_color}"
                        f"{conn_string}"
                        f"{ANSI_RESET}"
                    )

                    if ode_params.has_vrest():

                        vr_pct = (
                            100.0
                            * metrics.n_out_vrest
                            / metrics.n_total_vrest
                            if metrics.n_total_vrest > 0
                            else 0.0
                        )

                        bar_parts.append(
                            f"{r2_color(metrics.vrest_r2_clean)}"
                            f"Vr="
                            f"{fmt_r2_bar(metrics.vrest_r2_clean)}"
                            f"({vr_pct:.0f}%)"
                            f"{ANSI_RESET}"
                        )

                    if ode_params.has_tau():

                        tau_pct = (
                            100.0
                            * metrics.n_out_tau
                            / metrics.n_total_tau
                            if metrics.n_total_tau > 0
                            else 0.0
                        )

                        bar_parts.append(
                            f"{r2_color(metrics.tau_r2_clean)}"
                            f"τ="
                            f"{fmt_r2_bar(metrics.tau_r2_clean)}"
                            f"({tau_pct:.0f}%)"
                            f"{ANSI_RESET}"
                        )

                if (
                    metrics.hidden_r2 is not None
                    or metrics.anchor_r2 is not None
                ):

                    if not injection_active:

                        if metrics.anchor_r2 is not None:

                            nnr_string = (
                                f"nnr=n/a"
                                f"({metrics.anchor_r2:.3f})"
                            )

                        else:

                            nnr_string = (
                                "nnr=n/a"
                            )

                        bar_parts.append(
                            nnr_string
                        )

                    elif metrics.hidden_r2 is not None:

                        if metrics.anchor_r2 is not None:

                            nnr_string = (
                                f"nnr="
                                f"{metrics.hidden_r2:.3f}"
                                f"("
                                f"{metrics.anchor_r2:.3f}"
                                f")"
                            )

                        else:

                            nnr_string = (
                                f"nnr="
                                f"{metrics.hidden_r2:.3f}"
                            )

                        bar_parts.append(
                            f"{r2_color(metrics.hidden_r2)}"
                            f"{nnr_string}"
                            f"{ANSI_RESET}"
                        )

                if bar_parts:

                    pbar.set_postfix_str(
                        " ".join(bar_parts)
                    )

            # -------------------------------------------------------------
            # Visual field rendering
            # -------------------------------------------------------------

            if (
                train.has_visual_field
                and N in plot_iterations
            ):

                field_R2, field_slope = (
                    render_visual_field_video(
                        model,
                        x_ts,
                        sim,
                        log_dir,
                        epoch,
                        N,
                        logger,
                    )
                )

                metrics.field_r2 = field_R2
                metrics.field_slope = field_slope

                if metrics.connectivity_r2 is not None:

                    pbar.set_postfix_str(
                        f"{r2_color(metrics.connectivity_r2)}"
                        f"R²="
                        f"{metrics.connectivity_r2:.3f}"
                        f"{ANSI_RESET}"
                    )

                if training.save_all_checkpoints:

                    torch.save(
                        {
                            "model_state_dict":
                                model.state_dict(),
                            "optimizer_state_dict":
                                optimizer.state_dict(),
                        },
                        os.path.join(
                            log_dir,
                            "models",
                            f"best_model_with_"
                            f"{training.n_runs - 1}"
                            f"_graphs_{epoch}_{N}.pt",
                        ),
                    )

            # -------------------------------------------------------------
            # Profiler
            # -------------------------------------------------------------

            if _profiling:
                profiler.step()

        # =================================================================
        # END INNER LOOP
        # =================================================================

        if _profiling:

            profiler.stop()

            print(
                f"[Profiler] Trace saved to "
                f"{_profiler_trace_dir}/"
            )

            print(
                "  View with: "
                f"tensorboard --logdir "
                f"{_profiler_trace_dir}"
            )

        # =================================================================
        # EPOCH LOSS
        # =================================================================


        total_loss = epoch_state.total_loss_gpu.item()
        total_loss_regul = epoch_state.total_regul_gpu.item()

        epoch_total_loss = total_loss / n_neurons
        epoch_regul_loss = total_loss_regul / n_neurons
        epoch_pred_loss = (
            total_loss - total_loss_regul
        ) / n_neurons





        _logger.info(
            "epoch {}. loss: {:.6f} "
            "(pred: {:.6f}, regul: {:.6f})".format(
                epoch,
                epoch_total_loss,
                epoch_pred_loss,
                epoch_regul_loss,
            )
        )

        logger.info(
            "Epoch {}. Loss: {:.6f} "
            "(pred: {:.6f}, regul: {:.6f})".format(
                epoch,
                epoch_total_loss,
                epoch_pred_loss,
                epoch_regul_loss,
            )
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
            },
            os.path.join(
                log_dir,
                "models",
                f"best_model_with_"
                f"{training.n_runs - 1}"
                f"_graphs_{epoch}.pt",
            ),
        )

        if (
            train.has_visual_field
            and hasattr(model, "NNR_f")
        ):

            torch.save(
                model.NNR_f.state_dict(),
                os.path.join(
                    log_dir,
                    "models",
                    f"inr_stimulus_{epoch}.pt",
                ),
            )

        # -----------------------------------------------------------------
        # Epoch loss history
        # -----------------------------------------------------------------

        if "list_loss" not in locals():
            list_loss = []

        if "list_loss_regul" not in locals():
            list_loss_regul = []

        list_loss.append(
            epoch_pred_loss
        )

        list_loss_regul.append(
            epoch_regul_loss
        )

        torch.save(
            list_loss,
            os.path.join(
                log_dir,
                "loss.pt",
            ),
        )

        # -----------------------------------------------------------------
        # Epoch summary figure
        # -----------------------------------------------------------------

        fig = plt.figure(
            figsize=(
                3
                * default_style.figure_height
                * default_style.default_aspect,
                2
                * default_style.figure_height,
            )
        )

        ax1 = fig.add_subplot(
            2,
            3,
            1,
        )

        ax1.plot(
            list_loss,
            color=default_style.foreground,
            linewidth=default_style.line_width,
        )

        ax1.set_xlim(
            [
                0,
                training.n_epochs,
            ]
        )

        default_style.ylabel(
            ax1,
            "loss",
        )

        default_style.xlabel(
            ax1,
            "epochs",
        )

        plot_training_summary_panels(
            fig,
            log_dir,
            Niter=Niter,
        )

        # -----------------------------------------------------------------
        # Replace embedding with clusters
        # -----------------------------------------------------------------

        if train.replace_with_cluster:

            if (
                epoch % training.sparsity_freq
                == training.sparsity_freq - 1
                and epoch
                < training.n_epochs
                - training.sparsity_freq
            ):

                _logger.info(
                    "replace embedding "
                    "with clusters ..."
                )

                eps = (
                    training.cluster_distance_threshold
                )

                results = (
                    clustering_evaluation(
                        to_numpy(model.a),
                        type_list,
                        eps=eps,
                    )
                )

                _logger.info(
                    f"eps={eps}: "
                    f"{results['n_clusters_found']} "
                    "clusters, "
                    f"accuracy="
                    f"{results['accuracy']:.3f}"
                )

                labels = (
                    results["cluster_labels"]
                )

                for cluster_id in np.unique(
                    labels
                ):

                    indices = np.where(
                        labels == cluster_id
                    )[0]

                    if len(indices) > 1:

                        with torch.no_grad():

                            model.a[
                                indices,
                                :,
                            ] = torch.mean(
                                model.a[
                                    indices,
                                    :,
                                ],
                                dim=0,
                                keepdim=True,
                            )

                fig.add_subplot(
                    2,
                    3,
                    6,
                )

                type_cmap = (
                    CustomColorMap(
                        config=config
                    )
                )

                for neuron_type in range(
                    sim.n_neuron_types
                ):

                    pos = torch.argwhere(
                        type_list
                        == neuron_type
                    )

                    plt.scatter(
                        to_numpy(
                            model.a[
                                pos,
                                0
                            ]
                        ),
                        to_numpy(
                            model.a[
                                pos,
                                1
                            ]
                        ),
                        s=20,
                        color=(
                            type_cmap.color(
                                neuron_type
                            )
                        ),
                        edgecolors="none",
                    )

                plt.xlabel(
                    "embedding 0",
                    fontsize=18,
                )

                plt.ylabel(
                    "embedding 1",
                    fontsize=18,
                )

                plt.xticks([])
                plt.yticks([])

                plt.text(
                    0.5,
                    0.9,
                    f"eps={eps}: "
                    f"{results['n_clusters_found']} "
                    f"clusters, "
                    f"accuracy="
                    f"{results['accuracy']:.3f}",
                )

                if training.fix_cluster_embedding:

                    lr_embedding = (
                        1.0E-10
                    )

            else:

                lr = training.lr
                lr_embedding = (
                    training.lr_embedding
                )
                lr_W = training.lr_W

            logger.info(
                f"learning rates: "
                f"lr_W {lr_W}, "
                f"lr {lr}, "
                f"lr_update {lr_update}, "
                f"lr_embedding {lr_embedding}"
            )

            optimizer, n_total_params = (
                set_trainable_parameters(
                    model=model,
                    lr_embedding=lr_embedding,
                    lr=lr,
                    lr_update=lr_update,
                    lr_W=lr_W,
                )
            )

        # -----------------------------------------------------------------
        # UMAP clustering
        # -----------------------------------------------------------------

        if train.umap_cluster_active:

            if (
                epoch % training.umap_cluster_freq
                == training.umap_cluster_freq - 1
                and epoch
                < training.n_epochs - 1
            ):

                _logger.info(
                    "UMAP cluster reassign ..."
                )

                umap_results = (
                    umap_cluster_reassign(
                        model,
                        config,
                        x_ts,
                        edges,
                        n_neurons,
                        type_list,
                        device,
                        logger=logger,
                        reinit_mlps=(
                            training
                            .umap_cluster_reinit_mlps
                        ),
                        relearn_epochs=(
                            training
                            .umap_cluster_relearn_epochs
                        ),
                    )
                )

                if umap_results is not None:

                    fig.add_subplot(
                        2,
                        3,
                        6,
                    )

                    type_cmap = (
                        CustomColorMap(
                            config=config
                        )
                    )

                    a_umap = (
                        umap_results["a_umap"]
                    )

                    for neuron_type in range(
                        sim.n_neuron_types
                    ):

                        pos = torch.argwhere(
                            type_list
                            == neuron_type
                        )

                        pos_np = (
                            to_numpy(
                                pos
                            ).flatten()
                        )

                        plt.scatter(
                            a_umap[
                                pos_np,
                                0
                            ],
                            a_umap[
                                pos_np,
                                1
                            ],
                            s=20,
                            color=(
                                type_cmap.color(
                                    neuron_type
                                )
                            ),
                            edgecolors="none",
                        )

                    plt.xlabel(
                        r"UMAP$_1$",
                        fontsize=12,
                    )

                    plt.ylabel(
                        r"UMAP$_2$",
                        fontsize=12,
                    )

                    plt.xticks([])
                    plt.yticks([])

                    plt.title(
                        f"{umap_results['n_clusters']} cl, "
                        f"acc="
                        f"{umap_results['accuracy']:.3f}",
                        fontsize=10,
                    )

                if (
                    training.umap_cluster_fix_embedding
                    or training.umap_cluster_fix_embedding_ratio > 0
                ):

                    lr_embedding = 1.0E-10
                    embedding_frozen = True

                # Rebuild optimizer to reset momentum and relearn
                # f_theta / g_phi.
                optimizer, n_total_params = (
                    set_trainable_parameters(
                        model=model,
                        lr_embedding=lr_embedding,
                        lr=lr,
                        lr_update=lr_update,
                        lr_W=lr_W,
                    )
                )

        plt.tight_layout()

        plt.savefig(
            f"{log_dir}/tmp_training/"
            f"epoch_{epoch}.png",
            bbox_inches="tight",
            pad_inches=0.1,
        )

        plt.close()

    # =====================================================================
    # TRAINING COMPLETE
    # =====================================================================

    training_time = (
        time.time()
        - training_start_time
    )

    training_time_min = (
        training_time
        / 60.0
    )

    _logger.info(
        f"training completed in "
        f"{training_time_min:.1f} minutes"
    )

    logger.info(
        f"training completed in "
        f"{training_time_min:.1f} minutes"
    )

    if log_file is not None:

        log_file.write(
            f"training_time_min: "
            f"{training_time_min:.1f}\n"
        )

        log_file.write(
            f"n_epochs: "
            f"{training.n_epochs}\n"
        )

        log_file.write(
            f"data_augmentation_loop: "
            f"{training.data_augmentation_loop}\n"
        )

        log_file.write(
            f"recurrent_training: "
            f"{training.recurrent_training}\n"
        )

        log_file.write(
            f"batch_size: "
            f"{training.batch_size}\n"
        )

        log_file.write(
            f"lr_W: "
            f"{training.lr_W}\n"
        )

        log_file.write(
            f"lr: "
            f"{training.lr}\n"
        )

        log_file.write(
            f"lr_embedding: "
            f"{training.lr_embedding}\n"
        )

        log_file.write(
            f"coeff_g_phi_diff: "
            f"{training.coeff_g_phi_diff}\n"
        )

        log_file.write(
            f"coeff_g_phi_norm: "
            f"{training.coeff_g_phi_norm}\n"
        )

        log_file.write(
            f"coeff_g_phi_weight_L1: "
            f"{training.coeff_g_phi_weight_L1}\n"
        )

        log_file.write(
            f"coeff_f_theta_weight_L1: "
            f"{training.coeff_f_theta_weight_L1}\n"
        )

        log_file.write(
            f"coeff_f_theta_weight_L2: "
            f"{training.coeff_f_theta_weight_L2}\n"
        )

        log_file.write(
            f"coeff_W_L1: "
            f"{training.coeff_W_L1}\n"
        )

        log_file.write(
            f"dale_law: "
            f"{getattr(training, 'dale_law', False)}\n"
        )

        dale_score = dale_law_score(
            model,
            edges,
        )

        log_file.write(
            f"dale_law_score: "
            f"{dale_score:.4f}\n"
        )

        if metrics.field_r2 is not None:

            log_file.write(
                f"field_R2: "
                f"{metrics.field_r2:.4f}\n"
            )

            log_file.write(
                f"field_slope: "
                f"{metrics.field_slope:.4f}\n"
            )

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
              rollout_without_noise: bool = False, log_file=None, test_config=None,
              anatomy_voltage: bool = False, anatomy_voltage_type_groups=None):

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
        # Path-integration and swim-integration share the same dIPN-/EPG-style
        # heading-rollout test pipeline; data_test_path_integration_task reads
        # the matching task sub-block (path_integration or swim_integration)
        # for dt/T at the call sites that need them.
        if task_type in ('path_integration', 'swim_integration'):
            # Place-cell model has a 3+K output (heading+distance + place
            # logits) the heading test can't parse — route to the dedicated
            # place test (metrics + MP4 animations).
            if str(getattr(config.graph_model, 'signal_model_name', '')) \
                    == 'drosophila_cx_pi_place':
                data_test_place_task(
                    config, best_model=best_model, device=device,
                    log_file=log_file)
                return
            data_test_path_integration_task(
                config, best_model=best_model, device=device, log_file=log_file,
                anatomy_voltage=anatomy_voltage,
                anatomy_voltage_type_groups=anatomy_voltage_type_groups,
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
    data_test_place_task,
)


# ============================================================================
# Path-integration task trainer (TaskRNN)
# ============================================================================

def data_train_task(config, erase, best_model, device, log_file=None, resume=False):
    """Dispatch to the task-specific trainer based on `config.task.task_type`.

    - `path_integration` → CX trainer (TaskRNN sign_locked mode, Hulse aux
      losses, pi_acc eval, EPG kinograph snapshots).
    - `swim_integration` → zebrafish dIPN HD trainer (same TaskTrials
      shape, same eval helpers, dispatched through a dedicated entry
      point so future swim-specific behaviour stays out of the fly path).
    - `cortex`           → Yang multitask trainer (TaskRNN free mode,
      masked-MSE loss, direction_acc eval, 8-panel snapshot).
    """
    task_type = str(getattr(config.task, "task_type", "path_integration")).lower()
    if task_type == "cortex":
        return _data_train_cortex_task(config, erase, best_model, device, log_file)
    elif task_type in ("path_integration", "swim_integration"):
        # One shared trainer for both the Drosophila CX (path_integration)
        # and the larval-zebrafish dIPN (swim_integration) self-motion tasks:
        # the TaskTrials on-disk layout and the eval helpers are identical;
        # the species differences live entirely in the model class + circuit.
        return _data_train_task_pi(config, erase, best_model, device, log_file, resume=resume)
    else:
        raise ValueError(
            f"data_train_task: unknown task_type {task_type!r}; "
            f"expected one of {{path_integration, swim_integration, cortex}}"
        )


def _data_train_task_pi(config, erase, best_model, device, log_file=None, resume=False):
    """Train a TaskRNN/GNN on the self-motion-integration task data — the
    SINGLE shared trainer for both the Drosophila CX path-integration task
    (``task_type='path_integration'``) and the larval-zebrafish dIPN
    swim-integration task (``task_type='swim_integration'``).

    The two species share one trainer because the TaskTrials on-disk layout
    is byte-identical (stimulus ``[ω, (v_fwd,) cosθ₀·δ, sinθ₀·δ]``, target
    ``[cosθ, sinθ, ...]``) and the eval helpers
    (``path_integration_accuracy_from_data``, ``bump_fwhm``,
    ``_rollout_heading_metrics``, ``_save_training_snapshot``) read only the
    canonical model attributes (``epg_indices``, ``epg_glom_ix``,
    ``neuron_types``, ``type_names``) that both ``DrosophilaCxTask*`` and
    ``ZebrafishHdTask*`` expose. All species differences live in the model
    class + circuit, not here.

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
        tc.coeff_tv_circular  · L_tv   (circular TV on EPG ring)
        tc.coeff_W_L1         · |S|.sum()
    """
    import torch.nn.functional as F

    from connectome_gnn.models.bump_attractor_eval import (
        _rollout_heading_metrics,
        _save_training_snapshot,
        _save_place_snapshot,
        _save_torus_snapshot,
        bump_fwhm,
        path_integration_accuracy_from_data,
    )
    from connectome_gnn.models.heading_bins import (
        convert_cos_sin_input_to_bin_cue_torch,
        convert_cos_sin_target_to_bin_labels_torch,
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
    # On --resume keep them: snapshots/metrics.log carry over from the
    # interrupted run and metrics.log is appended below.
    if not resume:
        shutil.rmtree(os.path.join(log_dir, 'tmp_training'), ignore_errors=True)
    kinograph_dir = os.path.join(log_dir, 'tmp_training', 'evolution')
    os.makedirs(kinograph_dir, exist_ok=True)

    # --- load: trials stay on GPU between iterations ---------------
    # Refactor: route through TaskTrials.load (one call per split) instead of
    # four separate load_raw_array calls. The split dir was written by
    # ZarrTaskTrialsWriter (or legacy _write_trial_zarr — same on-disk
    # layout), so the loader works on both new and legacy datasets.
    from connectome_gnn.task_state import TaskTrials
    root = graphs_data_path(config.dataset)
    _logger.info(f'loading task data from {root}/(train|test)/...')
    trials_train = TaskTrials.load(f"{root}/train").to(device)
    trials_test  = TaskTrials.load(f"{root}/test").to(device)
    u_train, y_train = trials_train.stimulus, trials_train.target
    u_test,  y_test  = trials_test.stimulus,  trials_test.target
    _logger.info(f'task data: train u={tuple(u_train.shape)} y={tuple(y_train.shape)}  '
                 f'test u={tuple(u_test.shape)} y={tuple(y_test.shape)}')

    # ---- task-mode channel selection (4-ch superset → sub-task) ----------
    # Generator A always writes the 4-ch input [ω, v_fwd, cosθ0, sinθ0]. The
    # on-disk target shape depends on task.swim_integration.target_kind:
    #     scalar_xi   → 3-col [cosθ, sinθ, ξ]                 (legacy default)
    #     position_2d → 4-col [cosθ, sinθ, x, y]              (true 2D PI)
    # task_targets selects the sub-task and slices both u and y onto the
    # active channels. Profiles depend on the on-disk target shape:
    #
    #   scalar_xi dataset (target_kind='scalar_xi', y has 3 cols):
    #     ['rotation']                 → in [0,2,3]    out [0,1]   (3, 2)
    #     ['translation']              → in [1]         out [2]     (1, 1)
    #     ['rotation','translation']   → in [0,1,2,3]   out [0,1,2] (4, 3)
    #
    #   position_2d dataset (target_kind='position_2d', y has 4 cols):
    #     ['rotation']                 → in [0,2,3]    out [0,1]    (3, 2)
    #     ['position_2d']              → in [0,1,2,3]  out [0,1,2,3] (4, 4)
    #
    # Legacy 3-ch on-disk datasets (u_train.shape[-1] == 3) pre-date the
    # 4-ch layout and pass through unchanged (n_in/n_out default to 3/2).
    # Keyed by (n_in_disk, n_out_disk, sorted_task_targets_tuple). The
    # propriocep-split layout (5-col disk stim) adds a 3rd input
    # carrying v_proprio and gets its own (5, ·, ·) entries; the rest
    # are unchanged.
    _PROFILE_BY_TARGET = {
        # scalar_xi (4-col stim, 3-col target):
        (4, 3, ("rotation",)):                ([0, 2, 3],    [0, 1]),
        (4, 3, ("translation",)):             ([1],          [2]),
        (4, 3, ("rotation", "translation")):  ([0, 1, 2, 3], [0, 1, 2]),
        # position_2d (4-col stim, 4-col target):
        (4, 4, ("rotation",)):                ([0, 2, 3],    [0, 1]),
        (4, 4, ("position_2d",)):             ([0, 1, 2, 3], [0, 1, 2, 3]),
        # heading-only supervision but KEEP v_fwd in the input (latent
        # path-integration probe): full 4-ch input, 2-col heading target.
        # Works on either the scalar_xi (3-col) or position_2d (4-col) disk
        # target; only heading [0,1] is supervised, the rest is unused GT
        # available for post-hoc decoding.
        (4, 3, ("rotation_vfwd",)):           ([0, 1, 2, 3], [0, 1]),
        (4, 4, ("rotation_vfwd",)):           ([0, 1, 2, 3], [0, 1]),
        # propriocep-split (5-col stim — col 2 carries v_proprio):
        (5, 3, ("rotation",)):                ([0, 3, 4],       [0, 1]),
        (5, 3, ("translation",)):             ([1, 2],          [2]),
        (5, 3, ("rotation", "translation")):  ([0, 1, 2, 3, 4], [0, 1, 2]),
        (5, 4, ("rotation",)):                ([0, 3, 4],       [0, 1]),
        (5, 4, ("position_2d",)):             ([0, 1, 2, 3, 4], [0, 1, 2, 3]),
        # place_cells (4-col stim, 5-col target [cosθ, sinθ, ξ, x, y]): the
        # model (drosophila_cx_pi_place) emits 3 heading+distance cols + K
        # place logits; the trainer keeps all 5 target cols (heading+distance
        # supervised directly, (x,y) used to build the place code on the fly).
        (4, 5, ("place_cells",)):             ([0, 1, 2, 3], [0, 1, 2, 3, 4]),
        # grid_cells (torus): identical on-disk layout to place_cells; the
        # model switches to toroidal targets / circular decode via grid_mode.
        (4, 5, ("grid_cells",)):              ([0, 1, 2, 3], [0, 1, 2, 3, 4]),
        # rotation_torus: Net1-only, 6-col target [cosθ,sinθ,cosφx,sinφx,
        # cosφy,sinφy], plain MSE (circular via the cos/sin encoding).
        (4, 6, ("rotation_torus",)):          ([0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),
        # conjunction_input: 6-ch stimulus (base 4 + vx, vy)
        (6, 6, ("rotation_torus",)):          ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
    }
    _task_raw = list(getattr(tc, 'task_targets', None) or [])
    # Canonical key: rotation always before translation; position_2d listed
    # separately. Sorting is by a fixed enumeration so the key is stable.
    _RECOGNISED = ("rotation", "translation", "position_2d", "rotation_vfwd",
                   "place_cells", "grid_cells", "rotation_torus")
    _task_key = tuple(t for t in _RECOGNISED if t in _task_raw)
    task_targets_canonical = list(_task_key)
    if u_train.shape[-1] >= 4 and _task_key:
        n_in_disk = int(u_train.shape[-1])
        n_out_disk = int(y_train.shape[-1])
        _profile_key = (n_in_disk, n_out_disk, _task_key)
        if _profile_key not in _PROFILE_BY_TARGET:
            raise ValueError(
                f"training.task_targets={_task_raw!r} is not a valid "
                f"projection for an on-disk stimulus with {n_in_disk} "
                f"cols / target with {n_out_disk} cols. Recognised "
                f"(n_in, n_out, targets) keys: "
                f"{sorted(_PROFILE_BY_TARGET.keys())}."
            )
        in_cols, out_cols = _PROFILE_BY_TARGET[_profile_key]
        u_train = u_train[..., in_cols].contiguous()
        y_train = y_train[..., out_cols].contiguous()
        u_test  = u_test[...,  in_cols].contiguous()
        y_test  = y_test[...,  out_cols].contiguous()
        _logger.info(
            f"task_targets={task_targets_canonical} (on-disk y has "
            f"{n_out_disk} cols) → sliced to in_cols={in_cols} "
            f"out_cols={out_cols}; train u={tuple(u_train.shape)} "
            f"y={tuple(y_train.shape)}"
        )

    logger.info(f'train trials: {u_train.shape[0]}  test trials: {u_test.shape[0]}  '
                f'T: {u_train.shape[1]}  in: {u_train.shape[2]}  out: {y_train.shape[2]}')

    # --- model build via registry ----------------------------------------
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type,
                         config=config, device=device)
    n_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'model {model_config.signal_model_name}: {n_total_params:,} trainable params')
    logger.info(f'model: {model_config.signal_model_name}  params: {n_total_params}')

    # --- optional: cluster the per-neuron embedding by cell type + freeze ---
    # embedding_cell_type_init: set each neuron's a_i to its cell type's
    # equidistant 2-D cluster point (so the embedding is a clean per-type
    # cluster map); fix_embedding: freeze a (requires_grad off → excluded
    # from the optimizer). Ported from data_train_gnn so the zebrafish / CX
    # swim-integration GNN can train on a frozen type-clustered embedding.
    # Must be set before the optimizer is built (below). Requires
    # embedding_dim == 2.
    if getattr(tc, 'embedding_cell_type_init', False) and hasattr(model, 'a'):
        from connectome_gnn.utils import get_equidistant_points
        emb_dim = int(getattr(model_config, 'embedding_dim', 0))
        type_ids = np.asarray(model.neuron_types).astype(int)
        n_types = int(len(model.type_names))
        if emb_dim != 2:
            _logger.warning(
                f'embedding_cell_type_init requires embedding_dim=2, got '
                f'{emb_dim} — skipping')
        else:
            scale = float(getattr(tc, 'embedding_cell_type_scale', 1.0))
            ex, ey = get_equidistant_points(n_types)
            pts = (np.stack([ex, ey], axis=1) * scale).astype(np.float32)
            with torch.no_grad():
                model.a.copy_(torch.tensor(
                    pts[type_ids], dtype=torch.float32, device=device))
            _logger.info(
                f'embedding initialised with equidistant points for '
                f'{n_types} cell types (scale={scale})')
    if getattr(tc, 'fix_embedding', False) and hasattr(model, 'a'):
        model.a.requires_grad_(False)
        _logger.info(
            'embedding is fixed (requires_grad=False, excluded from optimizer)')

    # --- resume: load the latest completed per-epoch checkpoint ----------
    # Continue at the next epoch (epoch-boundary granularity). The model
    # state is loaded here (before torch.compile); the matching optimizer
    # state is loaded right after the optimizer is built below.
    start_epoch = 0
    resume_ckpt = None
    if resume:
        ckpt_path, _resumed_epoch = find_latest_epoch_checkpoint(log_dir, tc.n_runs)
        if ckpt_path is None:
            _logger.warning('--resume: no per-epoch checkpoint found; starting from epoch 0')
        else:
            resume_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(resume_ckpt['model_state_dict'])
            start_epoch = _resumed_epoch + 1
            _logger.info(f'--resume: loaded {ckpt_path}; resuming at epoch {start_epoch + 1}')
            logger.info(f'resume: loaded {os.path.basename(ckpt_path)}, start_epoch={start_epoch}')

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
    # Opt-in: split the per-neuron embedding ``a`` into its own constant-lr
    # group (``lr_embedding``) instead of riding the schedule-driven w_rec
    # group. Default False → ``a`` stays in w_rec, so untouched configs are
    # unchanged. Lets the embedding move at its own rate (e.g. when the
    # cluster is meant to drift freely with coeff_embedding_cluster=0).
    _emb_separate = bool(getattr(tc, 'embedding_separate_lr', False))

    def _name_to_group(name: str) -> str:
        if _emb_separate and name == "a":
            return "embedding"
        if name in ("S", "W", "a") or name.startswith(("g_phi.", "f_theta.")):
            return "w_rec"
        if (name in ("W_in", "W_out")
                or name.startswith(("_W_in_mlp.", "_W_out_mlp."))
                or name in ("v_pena_l", "v_pena_r", "v_penb_l", "v_penb_r")):
            return "w_ED"
        return "other"

    grouped: dict[str, list] = {"w_rec": [], "w_ED": [], "other": [], "embedding": []}
    for _name, _p in model.named_parameters():
        grouped[_name_to_group(_name)].append(_p)

    _opt_groups = [
        {"params": grouped["w_rec"],
         "lr": float(lr_W_rec) if lr_W_rec is not None else tc.lr,
         "name": "w_rec"},
        {"params": grouped["w_ED"],
         "lr": float(lr_W_ED) if lr_W_ED is not None else tc.lr,
         "name": "w_ED"},
        {"params": grouped["other"], "lr": tc.lr, "name": "other"},
    ]
    if grouped["embedding"]:
        _opt_groups.append(
            {"params": grouped["embedding"], "lr": float(tc.lr_embedding),
             "name": "embedding"})
    optimizer = torch.optim.Adam(_opt_groups)
    _emb_log = (f' | embedding lr={float(tc.lr_embedding)} '
                f'({len(grouped["embedding"])} params, constant)'
                if grouped["embedding"] else '')
    _logger.info(
        f'optimizer groups: '
        f'w_rec lr={float(lr_W_rec) if lr_W_rec is not None else tc.lr} '
        f'({len(grouped["w_rec"])} params, schedule-driven) | '
        f'w_ED lr={float(lr_W_ED) if lr_W_ED is not None else tc.lr} '
        f'({len(grouped["w_ED"])} params, constant) | '
        f'other lr={tc.lr} ({len(grouped["other"])} params, constant)'
        + _emb_log
    )
    if resume_ckpt is not None and 'optimizer_state_dict' in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        _logger.info('--resume: restored optimizer state (Adam m/v)')
    lr_scheduler = build_lr_scheduler(optimizer, config)
    _logger.info(f'lr={tc.lr}  lr_scheduler={getattr(tc, "lr_scheduler", "none")}')

    # --- regulariser coefficients (cached as Python scalars) -------------
    coeff_cos = float(tc.coeff_cos_distance)
    coeff_norm = float(tc.coeff_norm_floor)
    kappa_norm = float(tc.kappa_norm_floor)
    coeff_tv = float(tc.coeff_tv_circular)
    coeff_l1S = float(tc.coeff_W_L1)
    # place-cell task (model drosophila_cx_pi_place): KL-distribution + aux
    # position-decode weights, and a flag selecting the place loss branch.
    is_place = (("place_cells" in task_targets_canonical
                 or "grid_cells" in task_targets_canonical)
                and hasattr(model, 'place_per_frame_loss'))
    # rotation_torus: Net1-only task (no place model); gets the 3-D torus
    # snapshot instead of the heading kinograph.
    is_torus = ("rotation_torus" in task_targets_canonical)
    coeff_place = float(getattr(tc, 'coeff_place', 1.0))
    coeff_pos = float(getattr(tc, 'coeff_pos', 1.0))
    coeff_consistency = float(getattr(tc, 'coeff_consistency', 0.0))
    place_warmup_epochs = int(getattr(tc, 'place_warmup_epochs', 0))
    last_place_score = float('nan')
    if is_place and place_warmup_epochs > 0:
        _logger.info(f'place: Net1 warm-up for {place_warmup_epochs} epoch(s) '
                     f'(place loss off; heading+distance only), then '
                     f'coeff_place={coeff_place} coeff_pos={coeff_pos}')
    coeff_f_diff = float(getattr(tc, 'coeff_f_theta_diff', 0.0))
    coeff_g_diff = float(getattr(tc, 'coeff_g_phi_diff', 0.0))
    # Soft embedding-cluster pull: each neuron's a_i toward the centroid of
    # its cell type (centroid computed on-the-fly from model.a, so the whole
    # cluster is free to drift during training). Ported from the LossRegular-
    # iser used by data_train_gnn. Only active when a is learnable.
    coeff_emb_cluster = float(getattr(tc, 'coeff_embedding_cluster', 0.0))
    _emb_type_ids = _emb_type_count = None
    if coeff_emb_cluster > 0 and hasattr(model, 'a'):
        _emb_type_ids = torch.as_tensor(
            np.asarray(model.neuron_types).astype(np.int64), device=device)
        _emb_n_types = int(len(model.type_names))
        _emb_type_count = torch.bincount(
            _emb_type_ids, minlength=_emb_n_types).clamp(min=1).float().unsqueeze(-1)
    grad_clip = float(getattr(tc, 'grad_clip_W', 0.0))
    snapshots_per_epoch = int(getattr(tc, 'snapshots_per_epoch', 5))
    snapshot_n_steps = int(getattr(tc, 'snapshot_n_steps', 1500))
    snapshot_omega_deg = float(getattr(tc, 'snapshot_omega_deg', 60.0))
    # Constant v_fwd for the deterministic translation rollout (snapshot
    # panel f when 'translation' is in task_targets). Default matches the
    # generator's forward_vel_mean.
    snapshot_v_fwd = float(getattr(tc, 'snapshot_v_fwd', 1.0))
    _coeff_tail_log = float(getattr(tc, 'coeff_tail_loss', 0.0))
    _logger.info(f'losses: cos_distance={coeff_cos}  norm_floor={coeff_norm} (κ={kappa_norm})  '
                 f'tv_circular={coeff_tv}  W_L1={coeff_l1S}  f_theta_diff={coeff_f_diff}  '
                 f'g_phi_diff={coeff_g_diff}  tail_loss={_coeff_tail_log}')

    # --- calcium observation supervision (dataset B), optional -----------
    # `use_calcium` is the single on/off flag (calcium_batch_size > 0). When on,
    # each task batch is augmented by `calcium_batch_size` real-calcium trials
    # (dataset B), the heading MSE runs over the combined batch, and the extra
    # trials' rolled-out voltage is turned into calcium via the GCaMP class
    # (sim.gcamp_kernel) and matched to the recorded ΔF/F with a scale-invariant
    # loss weighted by coeff_observation. calcium_batch_size == 0 → byte-equal
    # to task-only training (nothing below runs).
    calcium_batch_size = int(getattr(tc, 'calcium_batch_size', 0))
    coeff_obs = float(getattr(tc, 'coeff_observation', 0.0))
    use_calcium = calcium_batch_size > 0
    ca_u = ca_y = ca_target = ca_obs_ix = gcamp = None
    ca_test_u = ca_test_target = ca_kino_model_ix = ca_kino_obs_ix = None
    dt_model = float(getattr(model, 'dt', getattr(sim, 'delta_t', 0.01)))
    n_calc = 0
    if use_calcium:
        import zarr as _zarr
        # Resolve dataset B under the same species folder as dataset A.
        # `config.dataset` was already prefixed by load_run_config (e.g.
        # 'zebrafish/...'); `calcium_dataset` is the raw yaml value, so give it
        # the same prefix when it's a bare name — the yaml points both datasets
        # by their bare names, exactly like `dataset`.
        calc_name = getattr(tc, 'calcium_dataset', '') or config.dataset
        if '/' not in calc_name and '/' in config.dataset:
            calc_name = config.dataset.rsplit('/', 1)[0] + '/' + calc_name
        calc_root = graphs_data_path(calc_name)
        _logger.info(f'loading calcium dataset B from {calc_root}/(train|test)/...')
        _ct = TaskTrials.load(f"{calc_root}/train").to(device)
        ca_u, ca_y = _ct.stimulus, _ct.target

        # The calcium dataset's `stimulus.zarr` is rotation-shaped 3-col
        # [ω, cos θ₀·δ, sin θ₀·δ] (no v_fwd channel) and `target.zarr`
        # is 2-col [cos θ, sin θ] (no d head). For task_targets that
        # involve translation, fetch `swim_forward.zarr` (the tail-EMG
        # forward envelope shown in Fig. 17) and assemble a synthetic-
        # superset-shaped ca_u / ca_y on the fly so the batch-cat below
        # matches the synthetic shapes.
        def _maybe_load_zarr(path):
            try:
                return torch.from_numpy(
                    np.asarray(_zarr.open(path, mode="r"))
                ).to(device).float()
            except Exception:
                return None

        # Build the synthetic superset shape used by the calcium batch.
        # When the synthetic training data is propriocep-split (5-col
        # stimulus [ω, v_extero, v_proprio, cosθ₀, sinθ₀]) we need to
        # match that shape on the calcium side too; the swim_forward
        # signal is plumbed through BOTH v_extero and v_proprio columns
        # (same driver, two anatomically distinct ports). Otherwise
        # the legacy 4-col superset [ω, v_fwd, cosθ₀, sinθ₀] is built.
        _propriocep_split = bool(getattr(
            getattr(config.task, "swim_integration", None),
            "propriocep_split", False))
        _needs_vfwd = bool(_task_key) and ("translation" in _task_key
                                            or "position_2d" in _task_key)
        if _needs_vfwd and ca_u.shape[-1] == 3:
            swim_fwd = _maybe_load_zarr(f"{calc_root}/train/swim_forward.zarr")
            if swim_fwd is None:
                raise RuntimeError(
                    f"calcium dataset {calc_name} is rotation-shaped (3 stim "
                    f"channels, no v_fwd) and swim_forward.zarr is missing; "
                    f"cannot drive a translation/position_2d task_targets "
                    f"{task_targets_canonical} from it.")
            # swim_fwd may be (Bc, T) or (Bc, T, 1) — collapse to (Bc, T).
            if swim_fwd.dim() == 3:
                swim_fwd = swim_fwd[..., 0]
            n_super = 5 if _propriocep_split else 4
            ca_u_super = torch.zeros(
                (ca_u.shape[0], ca_u.shape[1], n_super),
                dtype=ca_u.dtype, device=device,
            )
            if _propriocep_split:
                # [ω, v_extero, v_proprio, cos θ₀·δ, sin θ₀·δ]
                ca_u_super[..., 0] = ca_u[..., 0]          # ω
                ca_u_super[..., 1] = swim_fwd              # v_extero
                ca_u_super[..., 2] = swim_fwd              # v_proprio
                ca_u_super[..., 3] = ca_u[..., 1]          # cos θ₀·δ
                ca_u_super[..., 4] = ca_u[..., 2]          # sin θ₀·δ
            else:
                # [ω, v_fwd, cos θ₀·δ, sin θ₀·δ]
                ca_u_super[..., 0] = ca_u[..., 0]
                ca_u_super[..., 1] = swim_fwd
                ca_u_super[..., 2] = ca_u[..., 1]
                ca_u_super[..., 3] = ca_u[..., 2]
            ca_u = ca_u_super
            # Build the synthetic target matching the synthetic-data
            # target_kind:
            #   scalar_xi / both:  3 cols [cos θ, sin θ, d_integrated]
            #   position_2d:       4 cols [cos θ, sin θ, x, y] where
            #                      x = cumsum(v_fwd · cos θ)·dt and
            #                      y = cumsum(v_fwd · sin θ)·dt — the
            #                      same 2D PI integration the synthetic
            #                      generator does.
            dt_ds = float(getattr(config.task.swim_integration, "dt", 0.01))
            _tgt_kind = str(getattr(getattr(config.task, "swim_integration", None),
                                     "target_kind", "scalar_xi")).lower()
            if _tgt_kind == "position_2d":
                cos_t = ca_y[..., 0]                          # (Bc, T)
                sin_t = ca_y[..., 1]
                # Leaky vs cumulative — same recipe as the synthetic
                # generator (graph_data_generator._integrate_leaky).
                pos_tau = getattr(
                    getattr(config.task, "swim_integration", None),
                    "position_tau_s", None)
                vx = swim_fwd * cos_t
                vy = swim_fwd * sin_t
                if pos_tau is None or float(pos_tau) <= 0.0:
                    x_pos = torch.cumsum(vx, dim=1) * dt_ds
                    y_pos = torch.cumsum(vy, dim=1) * dt_ds
                else:
                    alpha = max(0.0, min(1.0 - dt_ds / float(pos_tau), 1.0))
                    x_pos = torch.zeros_like(vx)
                    y_pos = torch.zeros_like(vy)
                    for _t in range(1, vx.shape[1]):
                        x_pos[:, _t] = alpha * x_pos[:, _t - 1] + vx[:, _t] * dt_ds
                        y_pos[:, _t] = alpha * y_pos[:, _t - 1] + vy[:, _t] * dt_ds
                ca_y_super = torch.zeros(
                    (ca_y.shape[0], ca_y.shape[1], 4),
                    dtype=ca_y.dtype, device=device,
                )
                ca_y_super[..., 0] = cos_t
                ca_y_super[..., 1] = sin_t
                ca_y_super[..., 2] = x_pos
                ca_y_super[..., 3] = y_pos
            else:
                d_int = torch.cumsum(swim_fwd, dim=1) * dt_ds   # (Bc, T)
                ca_y_super = torch.zeros(
                    (ca_y.shape[0], ca_y.shape[1], 3),
                    dtype=ca_y.dtype, device=device,
                )
                ca_y_super[..., 0] = ca_y[..., 0]               # cos θ
                ca_y_super[..., 1] = ca_y[..., 1]               # sin θ
                ca_y_super[..., 2] = d_int                       # cumulative d
            ca_y = ca_y_super
            _logger.info(
                f'calcium dataset augmented with swim_forward.zarr '
                f'(propriocep_split={_propriocep_split}, '
                f'target_kind={_tgt_kind}) → ca_u '
                f'{tuple(ca_u.shape)}, ca_y {tuple(ca_y.shape)}')

        # Apply the same task_targets projection to ca_u / ca_y so the
        # batch-cat shapes match the synthetic batch. Profile is keyed
        # by (n_in_disk, n_out_disk, task_targets).
        _prof_key_cal = (ca_u.shape[-1], ca_y.shape[-1], _task_key)
        if _task_key and _prof_key_cal in _PROFILE_BY_TARGET:
            _ic, _oc = _PROFILE_BY_TARGET[_prof_key_cal]
            ca_u = ca_u[..., _ic].contiguous()
            ca_y = ca_y[..., _oc].contiguous()
            _logger.info(f'calcium dataset sliced to in_cols={_ic} out_cols={_oc}'
                          f' (task_targets={task_targets_canonical})')
        elif _task_key:
            raise RuntimeError(
                f"calcium dataset {calc_name} has stimulus.shape[-1]="
                f"{ca_u.shape[-1]} / target.shape[-1]={ca_y.shape[-1]}; "
                f"no task_targets projection matches "
                f"{task_targets_canonical}")
        ca_target = torch.from_numpy(
            np.asarray(_zarr.open(f"{calc_root}/train/calcium.zarr", mode="r"))
        ).to(device)                                       # (Bc_all, T, R)
        n_calc = ca_u.shape[0]
        # Per-trial timestamp (start_frame, t0_seconds) — where in the ~600 s
        # recording each trial was sliced from. ca_t0 [s] lets the loss /
        # model condition on absolute time so the non-stationary real ΔF/F
        # (drift across the block, despite periodic ω) can be tracked.
        # Optional: datasets generated before trial_time.zarr existed fall
        # back to t0=0 (time-conditioning then a no-op), staying loadable.
        _tt_path = f"{calc_root}/train/trial_time.zarr"
        if os.path.isdir(_tt_path):
            _tt = np.asarray(_zarr.open(_tt_path, mode="r"))
            ca_t0 = torch.from_numpy(_tt[:, 1].astype(np.float32)).to(device)  # (Bc_all,)
        else:
            _logger.warning(f'no trial_time.zarr at {_tt_path}; '
                            'regenerate dataset B to get per-trial timestamps. '
                            'Falling back to t0=0 (time-conditioning is a no-op).')
            ca_t0 = torch.zeros(n_calc, device=device)
        _cte = TaskTrials.load(f"{calc_root}/test").to(device)
        ca_test_u = _cte.stimulus
        # Mirror the same superset-augmentation logic on the test split:
        # for translation / position_2d tasks the 3-col rotation-shaped
        # ca_test_u is widened to 4 cols by inserting swim_forward as
        # col 1, then the standard slicing keeps only the licensed cols.
        if _needs_vfwd and ca_test_u.shape[-1] == 3:
            _swim_fwd_t = _maybe_load_zarr(f"{calc_root}/test/swim_forward.zarr")
            if _swim_fwd_t is not None:
                if _swim_fwd_t.dim() == 3:
                    _swim_fwd_t = _swim_fwd_t[..., 0]
                _ca_test_super = torch.zeros(
                    (ca_test_u.shape[0], ca_test_u.shape[1], 4),
                    dtype=ca_test_u.dtype, device=device,
                )
                _ca_test_super[..., 0] = ca_test_u[..., 0]
                _ca_test_super[..., 1] = _swim_fwd_t
                _ca_test_super[..., 2] = ca_test_u[..., 1]
                _ca_test_super[..., 3] = ca_test_u[..., 2]
                ca_test_u = _ca_test_super
        # Profile keyed by target columns; when we augmented ca_test_u
        # to the 4-col superset above, the matching target shape is 3
        # (cos θ, sin θ, d) — same as the train branch.
        _y_cols_for_profile = (3 if _needs_vfwd and ca_test_u.shape[-1] == 4
                                else int(_cte.target.shape[-1]))
        if _task_key and (_y_cols_for_profile, _task_key) in _PROFILE_BY_TARGET:
            _ic, _oc = _PROFILE_BY_TARGET[(_y_cols_for_profile, _task_key)]
            ca_test_u = ca_test_u[..., _ic].contiguous()
        ca_test_target = torch.from_numpy(
            np.asarray(_zarr.open(f"{calc_root}/test/calcium.zarr", mode="r"))
        ).to(device)
        _map = torch.load(f"{calc_root}/calcium_mapping.pt", weights_only=False)
        ca_obs_ix = _map["model_index"].to(device).long()  # (R,) obs col -> model neuron
        # Which observed neurons the obs-loss supervises (training.observation_neurons):
        #   'all'              -> every shared observed neuron (default).
        #   'exclude_afferent' -> drop the input/afferent neurons (RIPN / pt-IPN),
        #      supervising only the recurrent bump-pool, via the circuit's
        #      afferent_subpop_ix model-neuron sets (verified == type RIPN/pt-IPN).
        # ca_keep_col indexes the R calcium.zarr columns that survive the filter;
        # ca_obs_ix is reduced to the matching model-neuron indices.
        obs_neurons_mode = str(getattr(tc, 'observation_neurons', 'all')).lower()
        ca_keep_col = torch.arange(ca_obs_ix.numel(), device=device)
        if obs_neurons_mode == 'exclude_afferent':
            _cname = getattr(getattr(config, 'circuit', None), 'name', None)
            if _cname is None:
                _logger.warning('observation_neurons=exclude_afferent but config '
                                'has no circuit.name; supervising ALL observed neurons.')
            else:
                from connectome_gnn.generators.circuits import get_circuit
                _circ = get_circuit(_cname)
                _aff = np.unique(np.concatenate(
                    [np.asarray(_circ.subpops.get(k, []), np.int64) for k in
                     ('afferent_RIPN_L', 'afferent_RIPN_R',
                      'afferent_ptIPN_L', 'afferent_ptIPN_R')]
                    + [np.zeros(0, np.int64)]))
                _keep = ~np.isin(_map["model_index"].numpy(), _aff)
                ca_keep_col = torch.from_numpy(np.where(_keep)[0]).to(device).long()
                ca_obs_ix = _map["model_index"][torch.from_numpy(_keep)].to(device).long()
        # rastermap-ordered bump-pool rows that are observed (panel h compare):
        _kmask = _map["kino_obs_index"] >= 0
        ca_kino_model_ix = _map["kino_model_index"][_kmask].to(device).long()
        ca_kino_obs_ix = _map["kino_obs_index"][_kmask].to(device).long()
        from connectome_gnn.models.gcamp import create_gcamp
        gcamp = create_gcamp(sim.gcamp_kernel)
        _logger.info(
            f'calcium obs: dataset={calc_name} train={n_calc} test={ca_test_u.shape[0]} '
            f'obs_neurons={ca_obs_ix.numel()} ({obs_neurons_mode}) '
            f'kino_obs={ca_kino_obs_ix.numel()} '
            f'kernel={sim.gcamp_kernel} dt_in={dt_model} '
            f'batch={calcium_batch_size} coeff_observation={coeff_obs}')
        logger.info(f'calcium obs ON: B={calcium_batch_size} coeff={coeff_obs} '
                    f'kernel={sim.gcamp_kernel} '
                    f'obs_neurons={ca_obs_ix.numel()} ({obs_neurons_mode})')

    def _zscore_time(x, eps=1e-3):
        """Per-(trial,neuron) z-score over time → scale+offset invariant."""
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True)
        return (x - mu) / sd.clamp_min(eps)

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
    # On --resume offset the global step so metrics.log iteration numbers stay
    # continuous with the interrupted run (start_epoch completed epochs done).
    global_step = start_epoch * Niter
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
    # On --resume append to the existing metrics so the interrupted run's
    # history is preserved; otherwise truncate and write a fresh header.
    if resume and os.path.exists(metrics_log_path):
        _logger.info('--resume: appending to existing metrics.log')
    else:
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
    # Heading-bin ablation (training.use_heading_bins). The model was built
    # with n_input=1+K and n_output=K when the flag is on; here we hoist the
    # flag + K so the inner training loop can swap (cos/sin) layout for
    # K-bin layout on the fly. eval_model is the un-compiled handle, so the
    # attributes are always reachable (torch.compile wraps it below).
    use_bins = bool(getattr(eval_model, 'use_heading_bins', False))
    K_bins = int(getattr(eval_model, 'n_heading_bins', 64))
    if use_bins:
        _logger.info(
            f'heading-bin ablation ON: K={K_bins} bins, '
            f'CE loss replaces cos/sin MSE on the heading head'
        )
    if getattr(tc, 'torch_compile', True):
        try:
            # 'reduce-overhead' (CUDA Graphs) pins a static buffer pool per
            # input shape and conflicts with activation checkpointing's
            # recompute-in-backward. The task GNN checkpoints its T-loop
            # (grad_checkpoint, default True) precisely to bound rollout memory
            # to O(1) in T, and the n_steps curriculum recompiles every epoch as
            # T grows — so CUDA Graphs both defeat the checkpoint savings and
            # accumulate one un-freed pool per T, OOMing at the long-T epochs
            # (T=500 on an 80 GB A100). Use plain 'default' for checkpointed
            # models — same Triton kernels, no CUDA-graph capture. Mirrors the
            # connectivity-recovery compile site above.
            _ckpt = bool(getattr(eval_model, 'grad_checkpoint', False))
            _mode = 'default' if _ckpt else 'reduce-overhead'
            model = torch.compile(model, mode=_mode, fullgraph=not _ckpt)
            logger.info(f'torch.compile enabled (mode={_mode}, '
                        f'grad_checkpoint={_ckpt}); '
                        'eval/snapshot forward stays eager via _orig_mod')
            _logger.info(f'torch.compile enabled (mode={_mode}, eval via _orig_mod)')
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

    for epoch in range(start_epoch, tc.n_epochs):
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
            # Curriculum slice length: 2*T_epoch (soft tail) or T_epoch (hard).
            if coeff_tail > 0:
                T_use = min(2 * T_epoch, u_train.shape[1])
            else:
                T_use = T_epoch
            u = u_train[idx, :T_use]
            y = y_train[idx, :T_use]
            n_task = u.shape[0]

            # Append the calcium (dataset B) batch to the SAME forward, so the
            # heading MSE (first term) is computed over the combined task+calcium
            # batch and the calcium trials' voltage is available for the
            # observation loss. ca_c is the recorded ΔF/F target (model grid).
            ca_c = None
            if use_calcium:
                gen_c = torch.Generator(device=device).manual_seed(
                    tc.seed + 1_000_000 * (epoch + 1) + N)
                cidx = torch.randint(0, n_calc, (calcium_batch_size,),
                                     device=device, generator=gen_c)
                u_in = torch.cat([u, ca_u[cidx, :T_use]], dim=0)
                y_in = torch.cat([y, ca_y[cidx, :T_use]], dim=0)
                ca_c = ca_target[cidx, :T_use]                  # (Bc, T_use, R)
                # Absolute time of every frame in the calcium batch, in seconds
                # into the recording: t0(trial) + frame_index * dt. Shape
                # (Bc, T_use). Available for time-conditioning of the obs loss.
                ca_t_abs = (ca_t0[cidx][:, None]
                            + torch.arange(T_use, device=device)[None, :] * dt_model)
            else:
                u_in, y_in = u, y

            # Heading-bin ablation. The on-disk u_in is (B, T, 3) with the
            # last two channels carrying the (cos θ₀, sin θ₀) cue impulse at
            # t=0; convert to (B, T, 1+K) with a K-bin one-hot bump at t=0
            # before the forward. y_in stays (B, T, 2) on the wire — we map
            # it to (B, T) bin labels inside the loss.
            if use_bins:
                u_in = convert_cos_sin_input_to_bin_cue_torch(u_in, K_bins)

            if is_place:
                # PI anchor: hand the model the start position (y cols [3,4] at
                # t=0) so it can seed the place cells. Ignored unless the model
                # was built with place_anchor=True.
                y_hat, h_buf = model(u_in, pos0=y_in[:, 0, 3:5])
            else:
                y_hat, h_buf = model(u_in)

            # First loss term — task supervision over the WHOLE (task+calcium)
            # batch. u_train / y_train were already sliced to the active task
            # channels at load time (see _TASK_PROFILES above), so y_hat and
            # y_in are in the same column basis here — no per-iter slicing.
            #   bins off: MSE on cos/sin (Hulse-style heading head).
            #   bins on : CrossEntropy on K-bin logits, with the cos/sin
            #             target converted to bin labels on the fly.
            if is_place:
                # Place-cell task: per-frame loss = heading+distance MSE +
                # coeff_place·MSE(tanh place field, Gaussian g_k), with the same
                # soft-curriculum tail weighting as the MSE branch. The [0,1]
                # place score (field vs g cosine) feeds the progress bar.
                # Net1 warm-up: zero the place coeffs for the first
                # place_warmup_epochs epochs so only heading+distance trains.
                _warm = epoch < place_warmup_epochs
                _cp = 0.0 if _warm else coeff_place
                _cq = 0.0 if _warm else coeff_pos
                _cc = 0.0 if _warm else coeff_consistency
                place_pf, _pscore, _ppos = eval_model.place_per_frame_loss(
                    y_hat, y_in, _cp, _cq, _cc)                 # pf: (B, T_use)
                last_place_score = float(_pscore)
                if coeff_tail > 0:
                    w = torch.ones(T_use, device=u.device)
                    if T_epoch < T_use:
                        w[T_epoch:] = coeff_tail
                    mse = ((place_pf * w[None, :]).sum(dim=-1) / w.sum()).mean()
                else:
                    mse = place_pf.mean()
            elif use_bins:
                # y_in: (B, T, 2) → (B, T) long. Per-frame CE, then optional
                # soft-curriculum tail weighting (same shape as the MSE
                # branch so the rest of the loss assembly is unchanged).
                y_lbl = convert_cos_sin_target_to_bin_labels_torch(
                    y_in, K_bins)                                 # (B, T_use)
                ce_pf = F.cross_entropy(
                    y_hat.reshape(-1, K_bins),
                    y_lbl.reshape(-1),
                    reduction='none',
                ).view_as(y_lbl)                                  # (B, T_use)
                if coeff_tail > 0:
                    w = torch.ones(T_use, device=u.device)
                    if T_epoch < T_use:
                        w[T_epoch:] = coeff_tail
                    mse = ((ce_pf * w[None, :]).sum(dim=-1) / w.sum()).mean()
                else:
                    mse = ce_pf.mean()
            elif coeff_tail > 0:
                # Soft-curriculum: weight the per-frame MSE = 1 for t < T_epoch
                # and `coeff_tail_loss` for t >= T_epoch (non-zero gradient on
                # the post-horizon segment so late activity doesn't collapse).
                w = torch.ones(T_use, device=u.device)
                if T_epoch < T_use:
                    w[T_epoch:] = coeff_tail
                sq_err = (y_hat - y_in).pow(2).mean(dim=-1)        # (B, T_use)
                mse = ((sq_err * w[None, :]).sum(dim=-1) / w.sum()).mean()
            else:
                mse = F.mse_loss(y_hat, y_in)

            # Observation loss — convert the calcium-batch voltage to calcium via
            # the GCaMP class (sim.gcamp_kernel), gather the observed neurons, and
            # match the recorded ΔF/F with a scale+offset-invariant (per-trial,
            # per-neuron z-scored over time) MSE. Scaled by coeff_observation.
            if use_calcium and coeff_obs > 0:
                h_calc = h_buf[n_task:]                          # (Bc, T_use, N)
                ca_model = gcamp(h_calc, dt_in=dt_model)         # (Bc, T_use, N)
                # gather the supervised model neurons; ca_keep_col selects the
                # matching real columns (== arange(R) unless exclude_afferent).
                ca_model = ca_model.index_select(-1, ca_obs_ix)  # (Bc, T_use, n_sup)
                ca_c_sup = ca_c.index_select(-1, ca_keep_col)    # (Bc, T_use, n_sup)
                obs = coeff_obs * F.mse_loss(
                    _zscore_time(ca_model), _zscore_time(ca_c_sup))
            else:
                obs = u.new_zeros(())
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
            # embedding-cluster pull (only when a is learnable; centroid
            # computed on-the-fly so the cluster drifts with training).
            if (coeff_emb_cluster > 0 and hasattr(model, 'a')
                    and model.a.requires_grad and _emb_type_ids is not None):
                _a = model.a
                _sum = _a.new_zeros((_emb_type_count.shape[0], _a.shape[1]))
                _sum.scatter_add_(
                    0, _emb_type_ids.unsqueeze(-1).expand(-1, _a.shape[1]), _a)
                _neuron_means = (_sum / _emb_type_count)[_emb_type_ids]
                emb_cluster = (_a - _neuron_means).norm(2) * coeff_emb_cluster
            else:
                emb_cluster = u.new_zeros(())
            loss = mse + cosd + norm + tv + l1S + f_diff + g_diff + obs + emb_cluster

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
                    # All four metrics below are heading-only — pi_acc needs a
                    # cos/sin target column, bump_fwhm and the rollouts build a
                    # 3-channel [ω, cos, sin] probe input. They're meaningful
                    # only when the network is trained on rotation. In a pure
                    # translation run there is no ω in the input and no
                    # cos/sin in the target, so skip them and write nan.
                    # Heading is in the target whenever cos/sin lead it —
                    # rotation, both (rotation+translation), and
                    # position_2d all have it. Translation-only is the
                    # one mode where heading isn't supervised.
                    _has_rotation = ("rotation" in task_targets_canonical
                                     or "position_2d" in task_targets_canonical
                                     or "rotation_vfwd" in task_targets_canonical
                                     or not task_targets_canonical)
                    if _has_rotation:
                        # pi_acc and bump_fwhm both assume a 3-channel
                        # [ω, cos θ₀, sin θ₀] input and a (cos, sin) target,
                        # which the heading-bin ablation replaces with a
                        # K-bin one-hot cue / K-bin logit head. Skip them
                        # in bins mode and let the rollout-based heading
                        # metrics below (which go through the bins-aware
                        # _deterministic_sweep_rollout) carry the heading
                        # accuracy signal.
                        if use_bins:
                            last_pi_acc = float('nan')
                            last_fwhm = float('nan')
                        else:
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
                    else:
                        last_pi_acc = float('nan')
                        last_fwhm = float('nan')
                        last_rmse_roll = float('nan')
                        last_pearson_roll = float('nan')
                        last_pearson_roll_1k = float('nan')
                    # Calcium panel (panel h): roll one held-out real-calcium
                    # test trial, convert voltage→calcium via the GCaMP class,
                    # and pair it with the recorded ΔF/F over the bump-pool rows
                    # (rastermap order) for a real-vs-learned kinograph compare.
                    # This is the cheap single-10 s peek during training; the
                    # full ~600 s block reconstruction (60 tiled trials stitched
                    # + the continuous drift rollout) is produced at eval time by
                    # graph_tester.data_test_path_integration_task section (e).
                    calcium_panel = None
                    if use_calcium:
                        T_show = min(ca_test_u.shape[1], 1000)
                        ca_u_in = ca_test_u[:1, :T_show]
                        if use_bins:
                            ca_u_in = convert_cos_sin_input_to_bin_cue_torch(
                                ca_u_in, K_bins)
                        _, h_one = eval_model(ca_u_in)
                        ca_learn = gcamp(h_one[0], dt_in=dt_model)   # (T, N)
                        calcium_panel = dict(
                            real=ca_test_target[0, :T_show]
                                .index_select(-1, ca_kino_obs_ix).cpu().numpy(),
                            learned=ca_learn
                                .index_select(-1, ca_kino_model_ix).cpu().numpy(),
                            dt=dt_model)
                if is_place:
                    # Place task: render the 4×1 place dashboard instead of the
                    # heading compass kinograph (which assumes a 3-ch heading
                    # probe). Net1's heading is still visible in panel (d).
                    try:
                        _save_place_snapshot(
                            eval_model, log_dir, global_step, epoch + 1,
                            u_test, y_test, device)
                    except Exception as _e:
                        _logger.warning(f'place snapshot failed @ step '
                                        f'{global_step}: {_e}')
                    # grid runs also get the 3-D torus viz (same folder).
                    if getattr(eval_model, 'grid_mode', False):
                        try:
                            _save_torus_snapshot(
                                eval_model, log_dir, global_step, epoch + 1,
                                u_test, y_test, device)
                        except Exception as _e:
                            _logger.warning(f'grid torus snapshot failed @ step '
                                            f'{global_step}: {_e}')
                elif is_torus:
                    try:
                        _save_torus_snapshot(
                            eval_model, log_dir, global_step, epoch + 1,
                            u_test, y_test, device)
                    except Exception as _e:
                        _logger.warning(f'torus snapshot failed @ step '
                                        f'{global_step}: {_e}')
                else:
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
                        snapshot_v_fwd=snapshot_v_fwd,
                        config=config,
                        u_test=u_test,
                        y_test=y_test,
                        calcium_panel=calcium_panel,
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
            obs_str = f'obs={float(obs):.4f} ' if use_calcium else ''
            place_str = (f'place={last_place_score:.3f} '
                         if is_place and not np.isnan(last_place_score) else '')
            pbar.set_postfix_str(
                f'loss={loss.item():.4f} '
                f'{place_str}'
                f'{obs_str}'
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
    # pi_acc is heading-only — skip in a pure translation run. The heading-bin
    # ablation (use_bins) is handled inside path_integration_accuracy_from_data,
    # which decodes the K-bin logits to (cos,sin) before the cosine metric, so
    # the bins model reports a real pi_acc here just like the cos/sin models.
    if ("rotation" in task_targets_canonical
            or "position_2d" in task_targets_canonical
            or "rotation_vfwd" in task_targets_canonical
            or not task_targets_canonical):
        final_pi = path_integration_accuracy_from_data(
            eval_model, u_test, y_test, warmup=10, batch_size=tc.batch_size,
        )
        _logger.info(f'final test pi_acc: {final_pi:.4f}  '
                     f'(n_test={u_test.shape[0]}, T={u_test.shape[1]})')
        logger.info(f'final test pi_acc: {final_pi:.4f}')
    else:
        _logger.info(f'final test pi_acc: n/a (task_targets={task_targets_canonical}, '
                     f'translation-only — heading metric not defined)')


# ============================================================================
# Zebrafish swim-integration task trainer (HD-ring dIPN port)
# ============================================================================

# Back-compat aliases. The Drosophila CX (path_integration) and larval-
# zebrafish dIPN (swim_integration) self-motion tasks now share ONE trainer,
# ``_data_train_task_pi`` (the ``data_train_task`` dispatcher routes both task
# types to it). These names are retained so any external reference to the
# old species-specific entry points still resolves.
_data_train_drosophila_cx_task = _data_train_task_pi
_data_train_zebrafish_hd_task = _data_train_task_pi


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
            # 'reduce-overhead' (CUDA Graphs) pins a static buffer pool per
            # input shape and conflicts with activation checkpointing's
            # recompute-in-backward. The task GNN checkpoints its T-loop
            # (grad_checkpoint, default True) precisely to bound rollout memory
            # to O(1) in T, and the n_steps curriculum recompiles every epoch as
            # T grows — so CUDA Graphs both defeat the checkpoint savings and
            # accumulate one un-freed pool per T, OOMing at the long-T epochs
            # (T=500 on an 80 GB A100). Use plain 'default' for checkpointed
            # models — same Triton kernels, no CUDA-graph capture. Mirrors the
            # connectivity-recovery compile site above.
            _ckpt = bool(getattr(eval_model, 'grad_checkpoint', False))
            _mode = 'default' if _ckpt else 'reduce-overhead'
            model = torch.compile(model, mode=_mode, fullgraph=not _ckpt)
            logger.info(f'torch.compile enabled (mode={_mode}, '
                        f'grad_checkpoint={_ckpt}); '
                        'eval/snapshot forward stays eager via _orig_mod')
            _logger.info(f'torch.compile enabled (mode={_mode}, eval via _orig_mod)')
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
