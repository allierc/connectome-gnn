"""Test functions for GNN models.

Extracted from graph_trainer.py to reduce file size.
Contains:
- data_test_gnn: standard test with 1-step + rollout evaluation
- data_test_gnn_special: ablation/modification test via ODE regeneration
"""

import glob
import os
import re
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm, trange

from connectome_gnn.figure_style import dark_style
from connectome_gnn.generators.graph_data_generator import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    greedy_blue_mask,
    mseq_bits,
)
from connectome_gnn.generators.ode_params import FlyVisCurrentODEParams, load_edge_index
from connectome_gnn.generators.utils import generate_compressed_video_mp4
from connectome_gnn.log import get_logger
from connectome_gnn.models.utils import (
    ANSI_ORANGE,
    ANSI_RESET,
    forward_kind,
    r2_color,
    restore_edge_sign_lock,
)
from connectome_gnn.metrics import INDEX_TO_NAME
from connectome_gnn.models.neural_ode_wrapper import integrate_neural_ode
from connectome_gnn.models.registry import create_model
from connectome_gnn.models.training_utils import HiddenNeuronHandler
from connectome_gnn.models.utils import _batch_frames
from connectome_gnn.neuron_state import NeuronState
from connectome_gnn.plot import plot_spatial_activity_grid, plot_weight_comparison
from connectome_gnn.utils import (
    compute_trace_metrics,
    fisher_pool,
    get_datavis_root_dir,
    get_equidistant_points,
    graphs_data_path,
    log_path,
    migrate_state_dict,
    to_numpy,
)


def _save_per_neuron_arrays(log_path_: str, pearson: np.ndarray,
                             rmse: np.ndarray) -> None:
    """Save per-neuron pearson/RMSE arrays next to the matching ``results_*.log``.

    Pass the log path; two sibling files are written using the log's stem:
    ``{stem}_pearson.npy`` and ``{stem}_rmse.npy``. Lets aggregators
    (``cv_runner``, ``emit_inr_table_rows``) pool across (neurons × folds) in
    Fisher-$z$ space instead of averaging already-collapsed scalars.
    """
    stem = os.path.splitext(log_path_)[0]
    try:
        np.save(f'{stem}_pearson.npy', np.asarray(pearson, dtype=np.float32))
        np.save(f'{stem}_rmse.npy',    np.asarray(rmse,    dtype=np.float32))
    except OSError as exc:
        logger.warning(f'could not save per-neuron arrays ({stem}): {exc}')


def _pearson_log_line(pearson: np.ndarray) -> str:
    """'Pearson r: {mean} +/- {sd}' with Fisher-z-pooled mean and symmetric SD.

    The numeric format is preserved so existing parsers (`parse_pearson_from_log`
    in cv_runner, `parse_pearson` in emit_inr_table_rows) keep working.
    """
    fz = fisher_pool(pearson)
    return f'Pearson r: {fz["r_mean"]:.3f} +/- {fz["r_sd_sym"]:.3f}\n'
from connectome_gnn.zarr_io import load_raw_array, load_simulation_data

try:
    from connectome_gnn.generators.davis import AugmentedVideoDataset, CombinedVideoDataset
except ImportError:
    AugmentedVideoDataset = None
    CombinedVideoDataset = None

logger = get_logger(__name__)


def _compute_inr_traces(model, x_ts, hidden_ids, device, n_traces=None, n_frames=None):
    """Evaluate INR hidden-neuron predictions without rollout state.

    Calls model.forward_hidden(x, k, hidden_ids) for each frame independently
    (pure INR, no GNN dynamics), then applies a global linear correction so
    that the saved traces are in the same scale as the ground truth.

    All ``len(hidden_ids)`` traces are returned by default (set
    ``n_traces`` to subsample). Per-neuron positions ``pos[hidden_ids]`` are
    also returned so downstream figures can render a column-resolved map of
    the hidden-neuron error.

    Args:
        model:      trained NeuralGNN with model.NNR_hidden initialised
        x_ts:       TimeSeries used for ground truth and forward_hidden state
        hidden_ids: (n_hidden,) tensor of global neuron indices
        device:     torch device
        n_traces:   how many hidden neurons to store. None → all hidden_ids.
                    If less than n_hidden, evenly-spaced ids are kept.
        n_frames:   number of frames to evaluate (None → all frames in x_ts)

    Returns dict with keys:
        gt_arr        (n_traces, n_frames)  ground-truth voltages
        pred_arr      (n_traces, n_frames)  raw INR predictions
        pred_corr_arr (n_traces, n_frames)  linearly-corrected INR predictions
        global_ids    (n_traces,)           global neuron indices of stored neurons
        global_pos    (n_traces, 2) | None  (x, y) positions of those neurons
                                            (None if x_ts has no pos field)
        inr_type      str                   value of model._inr_hidden_type
        r2            float                 mean R² of corrected predictions
        r2_per        (n_traces,) float32   per-neuron R² (corrected)
    """
    n_hidden = len(hidden_ids)
    if n_traces is None:
        n_traces = n_hidden
    n_traces = min(n_traces, n_hidden)
    n_frames = min(n_frames, x_ts.n_frames) if n_frames is not None else x_ts.n_frames

    if n_traces == n_hidden:
        sel = np.arange(n_hidden, dtype=int)
    else:
        sel = np.linspace(0, n_hidden - 1, n_traces, dtype=int)
    local_ids = hidden_ids[sel]                          # (n_traces,) global indices

    gt_arr   = np.zeros((n_traces, n_frames), dtype=np.float32)
    pred_arr = np.zeros((n_traces, n_frames), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for k in range(n_frames):
            x = x_ts.frame(k)
            pred_h = model.forward_hidden(x, k, hidden_ids)   # (n_hidden,)
            gt_h   = x_ts.voltage[k, hidden_ids]              # (n_hidden,)
            gt_arr[:, k]   = to_numpy(gt_h[sel])
            pred_arr[:, k] = to_numpy(pred_h[sel])
    model.train()

    # Global linear correction: pred_corr = a * pred + b  ≈  gt
    gt_T, pred_T = gt_arr.T, pred_arr.T
    gt_f, pred_f = gt_T.ravel(), pred_T.ravel()
    cov = ((pred_f - pred_f.mean()) * (gt_f - gt_f.mean())).mean()
    var = ((pred_f - pred_f.mean()) ** 2).mean()
    a_coeff = float(cov / (var + 1e-12))
    b_coeff = float(gt_f.mean() - a_coeff * pred_f.mean())
    pred_corr_arr = (a_coeff * pred_T + b_coeff).T.astype(np.float32)

    # Per-neuron R²
    gt_mean_n = gt_T.mean(axis=0)
    ss_res = ((gt_T - (a_coeff * pred_T + b_coeff)) ** 2).sum(axis=0)
    ss_tot = ((gt_T - gt_mean_n) ** 2).sum(axis=0)
    r2_per = (1.0 - ss_res / (ss_tot + 1e-12)).astype(np.float32)
    r2 = float(r2_per.mean())

    global_pos = None
    if getattr(x_ts, 'pos', None) is not None:
        try:
            global_pos = to_numpy(x_ts.pos[local_ids, :2]).astype(np.float32)
        except Exception:
            global_pos = None

    return dict(
        gt_arr        = gt_arr,
        pred_arr      = pred_arr,
        pred_corr_arr = pred_corr_arr,
        global_ids    = to_numpy(local_ids),
        global_pos    = global_pos,
        inr_type      = getattr(model, '_inr_hidden_type', 'siren_t'),
        r2            = r2,
        r2_per        = r2_per,
    )


def data_test_gnn(config, best_model=None, device=None, log_file=None, test_config=None, test_mode=''):
    """Test using pre-generated test data (x_list_test / y_list_test).

    Loads the held-out test split, runs the trained model on every frame,
    and reports per-neuron RMSE, Pearson r, R², and FEVE.

    Args:
        config: model config (model + log dir come from here)
        test_config: optional second config for cross-dataset evaluation
                     (test data loaded from test_config.dataset)
    """

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    log_dir = log_path(config.config_file)

    # Determine test dataset: test_config > tc.test_dataset > config.dataset
    if test_config is not None:
        test_ds = test_config.dataset
        logger.info(f'cross-dataset test: model from {config.dataset}, test data from {test_ds}')
    elif tc.test_dataset:
        test_ds = tc.test_dataset
    else:
        test_ds = config.dataset

    # Suffix for output files when testing on a different dataset
    if test_ds != config.dataset:
        test_ds_short = test_ds.replace('flyvis_', '').replace('fly/', '')
        test_suffix = f'_on_{test_ds_short}'
    else:
        test_suffix = ''
    # Append the test_mode (e.g. "test_ablation_50") so ablation bundles don't
    # overwrite the corresponding non-ablation rollout output. Defensive
    # `locals().get(...)` so this line stays compatible with any older copy of
    # the function whose signature predates the test_mode parameter (e.g. a
    # checkout where the suffix edit landed but the signature edit didn't).
    _tm = locals().get('test_mode', '')
    if _tm:
        test_suffix = f"{test_suffix}_{_tm}"

    # Determine which fields to load
    load_fields = ['voltage', 'stimulus', 'neuron_type']
    has_visual_field = 'visual' in model_config.field_type
    _inr_hidden = getattr(model_config, 'inr_type_hidden', 'none')
    has_hidden_neurons = getattr(model_config, 'hidden_neuron_fraction', 0.0) > 0.0
    if has_visual_field or 'test' in model_config.field_type or _inr_hidden == 'siren_txy':
        load_fields.append('pos')
    # When the hidden-neuron INR is active, the rollout bundle saves
    # pos[hidden_ids] alongside the traces so figures can render
    # column-resolved error maps without re-loading the dataset.
    if has_hidden_neurons and 'pos' not in load_fields:
        load_fields.append('pos')
    if sim.calcium_type != 'none':
        load_fields.append('calcium')

    # Load test data (fall back to x_list_0 for backwards compatibility)
    test_path = graphs_data_path(test_ds, 'x_list_test')
    if os.path.exists(test_path):
        x_ts = load_simulation_data(test_path, fields=load_fields).to(device)
        y_ts = load_raw_array(graphs_data_path(test_ds, 'y_list_test'))
    else:
        logger.warning("x_list_test not found, falling back to x_list_0")
        x_ts = load_simulation_data(
            graphs_data_path(test_ds, 'x_list_0'), fields=load_fields
        ).to(device)
        y_ts = load_raw_array(graphs_data_path(test_ds, 'y_list_0'))

    # Extract type_list and set up index
    type_list = x_ts.neuron_type.float().unsqueeze(-1)
    x_ts.neuron_type = None
    x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long, device=device)

    if tc.training_selected_neurons:
        selected_neuron_ids = np.array(tc.selected_neuron_ids).astype(int)
        x_ts = x_ts.subset_neurons(selected_neuron_ids)
        y_ts = y_ts[:, selected_neuron_ids, :]
        type_list = type_list[selected_neuron_ids]

    # Cap test frames to avoid runaway evaluation on large datasets (e.g. hold-out)
    MAX_TEST_FRAMES = 8000
    if x_ts.n_frames > MAX_TEST_FRAMES:
        logger.info(f'capping test frames: {x_ts.n_frames} → {MAX_TEST_FRAMES}')
        x_ts = x_ts.truncate_frames(MAX_TEST_FRAMES)
        y_ts = y_ts[:MAX_TEST_FRAMES]

    n_neurons = x_ts.n_neurons
    n_frames = x_ts.n_frames
    config.simulation.n_neurons = n_neurons
    logger.info(f'\033[94mtest dataset: {test_ds}\033[0m, {n_frames} frames, {n_neurons} neurons')

    # Adjust n_edges to match training edges
    training_edges_path = os.path.join(log_dir, 'training_edges.pt')
    if os.path.exists(training_edges_path):
        edges_for_size = torch.load(training_edges_path, map_location='cpu', weights_only=False)
    else:
        edges_for_size = load_edge_index(graphs_data_path(config.dataset), device='cpu')
    actual_n_edges = edges_for_size.shape[1]
    expected_total = sim.n_edges + sim.n_extra_null_edges
    if actual_n_edges == expected_total and sim.n_extra_null_edges > 0:
        logger.info(f'null edges in data: {sim.n_edges} base + {sim.n_extra_null_edges} null = {actual_n_edges}')
        config.simulation.n_edges = actual_n_edges
        config.simulation.n_extra_null_edges = 0
    elif actual_n_edges != sim.n_edges:
        logger.info(f'n_edges mismatch: config={sim.n_edges}, actual={actual_n_edges} — using actual')
        config.simulation.n_edges = actual_n_edges

    # Create and load model
    logger.info('creating model ...')
    model = create_model(
        model_config.signal_model_name,
        aggr_type=model_config.aggr_type, config=config, device=device,
    )
    model = model.to(device)

    # EVALUATION IS ALWAYS fp32, whatever training.mlp_precision says.
    #
    # The rollout is one continuous free run over ~7200 frames -- x.voltage is
    # never reset from ground truth -- and it is the reported result. Measured on
    # flyvis_current_noise_005_current_cv00, the SAME fp32-trained checkpoint scored under
    # bf16 instead of fp32:
    #
    #     fp32   RMSE 0.0165   r 0.999 +/- 0.003   Fisher-z 3.7991
    #     bf16   RMSE 0.0211   r 0.998 +/- 0.007   Fisher-z 3.3619
    #
    # +24% RMSE, and the z mean falls 11.5%. The error does NOT compound -- the
    # per-window ratio is 1.22-1.26 from frame 0 to 7207, flat, because the
    # stimulus re-anchors the state every step -- but a constant 24% is still 24%.
    # And there is nothing to buy with it: the rollout is no_grad and already runs
    # at ~172 it/s, so bf16 would trade a fifth of the reported accuracy for
    # throughput nobody is waiting on. Without this, `-o train,test` under bf16
    # would silently evaluate in bf16 too.
    if getattr(model, "_amp_dtype", None) is not None:
        logger.info(f"mlp_precision {config.training.mlp_precision} ignored at "
                    "test time — evaluation runs in fp32")
        model._amp_dtype = None
    torch.set_float32_matmul_precision("highest")

    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/best_model_with_*.pt")
        if not files:
            files = glob.glob(f"{log_dir}/models/*.pt")
        assert len(files), 'no model checkpoints found in models/ directory — using untrained model'
        best_model = max(files, key=os.path.getmtime)
        logger.info(f'best model: {best_model}')

    # best_model is already a full path from glob, or a filename to prepend log_dir/models/ to
    if os.path.isabs(best_model) or '/' in best_model:
        netname = best_model
    else:
        netname = f"{log_dir}/models/{best_model}"
    logger.info(f'loading {netname} ...')
    state_dict = torch.load(netname, map_location=device, weights_only=False)
    migrate_state_dict(state_dict)
    model.load_state_dict(state_dict['model_state_dict'], strict=False)
    logger.info(f'loaded checkpoint successfully')

    # Confirm hidden SIREN was loaded from checkpoint (weights are in main state_dict)
    if getattr(model, 'NNR_hidden', None) is not None:
        _nnr_keys = [k for k in state_dict['model_state_dict'] if k.startswith('NNR_hidden')]
        if _nnr_keys:
            logger.info(f'NNR_hidden loaded from checkpoint ({len(_nnr_keys)} tensors)')
        else:
            logger.warning('NNR_hidden not found in checkpoint — using random initialisation')

    # Load INR model if visual field is learned.
    # best_model may be a full path like
    #   <log_dir>/models/best_model_with_0_graphs_2.pt
    # The matching INR checkpoint is inr_stimulus_<graphs_N>.pt (same N).
    if has_visual_field and hasattr(model, 'NNR_f'):
        import re as _re
        _basename = os.path.basename(best_model) if best_model else ''
        _m = _re.search(r'_graphs_(\d+)\.pt$', _basename)
        epoch_str = _m.group(1) if _m else '0'
        inr_path = os.path.join(log_dir, 'models', f'inr_stimulus_{epoch_str}.pt')
        if os.path.exists(inr_path):
            model.NNR_f.load_state_dict(torch.load(inr_path, map_location=device, weights_only=False))
            logger.info(f'loaded INR from {inr_path}')
        else:
            logger.warning(f'INR checkpoint not found at {inr_path}')

    model.eval()

    # Re-establish the Eq-10 hard sign-lock. ``_edge_sign`` is non-persistent so
    # load_state_dict drops it; without this the rollout would run on the raw
    # (free-sign) W -- which under sign-lock training carries the WRONG signs,
    # collapsing the rollout Pearson. Re-derive from the connectome (ode_params.W)
    # exactly as the trainer does, using the model's own ODE registry entry.
    if getattr(model, 'lock_edge_signs_from_connectome', False):
        from connectome_gnn.generators.ode_params import get_ode_params_class
        try:
            _SignOdeCls = get_ode_params_class(model_config.signal_model_name)
        except KeyError:
            _SignOdeCls = FlyVisCurrentODEParams
        try:
            _sign_odep = _SignOdeCls.load(graphs_data_path(config.dataset), device=device)
            if restore_edge_sign_lock(model, getattr(_sign_odep, 'W', None)):
                logger.info('restored Eq-10 sign-lock from ode_params.W for rollout')
        except Exception as _e:
            logger.warning(f'could not restore sign-lock (rollout sign may be wrong): {_e}')

    # Apply ablation mask if test dataset has one
    mask_path = graphs_data_path(test_ds, 'ablation_mask.pt')
    if os.path.exists(mask_path):
        ablation_mask = torch.load(mask_path, map_location=device, weights_only=False)
        with torch.no_grad():
            model.W[~ablation_mask] = 0
        logger.info(f'applied ablation mask: {(~ablation_mask).sum().item()} edges zeroed in model.W')

    # Random test-time ablation (test_mode="test_ablation_50" → zero 50% of
    # edges). Deterministic seed so the same edges are removed on every run.
    # Same defensive locals() lookup as above so an older copy of this file
    # that lacks the test_mode parameter just skips the block.
    _tm = locals().get('test_mode', '')
    if 'test_ablation' in _tm:
        try:
            ablation_ratio = int(_tm.split('_')[-1]) / 100
        except ValueError:
            ablation_ratio = 0.0
        if ablation_ratio > 0:
            n_total = model.W.shape[0]
            n_ablate = int(n_total * ablation_ratio)
            rng = np.random.default_rng(0)
            idx = rng.choice(n_total, n_ablate, replace=False)
            with torch.no_grad():
                model.W[idx] = 0
            logger.info(
                f'test_mode ablation: zeroed {n_ablate}/{n_total} edges '
                f'(ratio {ablation_ratio})'
            )

    # When a field INR is learned (visual SIREN, hidden NGP-T) rollout must
    # happen on training frames — the INR was fit to those time indices only
    # and cannot extrapolate to held-out test frames. For hidden NGP-T this
    # matches the noisy training distribution the grid was fit on.
    _use_train_data = has_visual_field or has_hidden_neurons
    if _use_train_data:
        train_path = graphs_data_path(config.dataset, 'x_list_train')
        if os.path.exists(train_path):
            x_ts_train = load_simulation_data(train_path, fields=load_fields).to(device)
            y_ts_train = load_raw_array(graphs_data_path(config.dataset, 'y_list_train'))
            x_ts_train.neuron_type = None
            x_ts_train.index = torch.arange(x_ts_train.n_neurons, dtype=torch.long, device=device)
            if tc.training_selected_neurons:
                x_ts_train = x_ts_train.subset_neurons(selected_neuron_ids)
                y_ts_train = y_ts_train[:, selected_neuron_ids, :]
            n_eval_frames = min(n_frames, x_ts_train.n_frames)
            _reason = ('visual field learned' if has_visual_field
                       else 'hidden NGP-T learned')
            logger.info(f'{_reason}: evaluating on training data '
                        f'({x_ts_train.n_frames} frames available, using {n_eval_frames})')
            x_ts_eval = x_ts_train
            y_ts_eval = y_ts_train
        else:
            logger.warning('x_list_train not found, falling back to test data')
            x_ts_eval = x_ts
            y_ts_eval = y_ts
            n_eval_frames = n_frames
    else:
        x_ts_eval = x_ts
        y_ts_eval = y_ts
        n_eval_frames = n_frames

    # Load edges: prefer training_edges.pt (handles fully connected mode),
    # fall back to data folder edge_index.pt / ode_params.pt
    training_edges_path = os.path.join(log_dir, 'training_edges.pt')
    if os.path.exists(training_edges_path):
        edges = torch.load(training_edges_path, map_location=device, weights_only=False)
        logger.info(f'loaded training edges from {training_edges_path} ({edges.shape[1]} edges)')
    else:
        edges = load_edge_index(graphs_data_path(config.dataset), device=device)
    ids = np.arange(n_neurons)
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    # Load hidden/anchor neuron ids for rollout (saved by the trainer that
    # produced this checkpoint; never sampled fresh at eval time).
    # (has_hidden_neurons is defined once up top alongside has_visual_field)
    hidden_ids = None
    anchor_ids = None
    if has_hidden_neurons:
        _hidden_path = os.path.join(log_dir, 'hidden_neuron_ids.pt')
        if os.path.exists(_hidden_path):
            hidden_ids = torch.load(_hidden_path, map_location=device, weights_only=True)
            logger.info(f'hidden neurons: {len(hidden_ids)} — using during rollout')

        _anchor_path = os.path.join(log_dir, 'anchor_neuron_ids.pt')
        if getattr(model, 'n_anchor', 0) > 0 and os.path.exists(_anchor_path):
            anchor_ids = torch.load(_anchor_path, map_location=device, weights_only=True)

    hn = HiddenNeuronHandler.from_ids(hidden_ids, anchor_ids)

    # Run model on all frames (one-step prediction)
    logger.info(f'one-step prediction on {n_eval_frames} frames ...')
    all_pred = []
    all_true = []

    with torch.no_grad():
        for k in trange(n_eval_frames - 1, ncols=100, desc="one-step"):
            x = x_ts_eval.frame(k)
            y = torch.tensor(y_ts_eval[k], device=device)

            if torch.isnan(x.voltage).any() or torch.isnan(y).any():
                continue

            if has_visual_field:
                visual_input = model.forward_visual(x, k)
                x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                x.stimulus[model.n_input_neurons:] = 0

            if forward_kind(model) == 'stimulus':
                tw = tc.time_window
                if k < tw - 1:
                    continue
                stim_ctx = x_ts_eval.stimulus[k-tw+1:k+1, :sim.n_input_neurons].unsqueeze(0)
                pred = model.predict_voltage(stim_ctx).squeeze(0)
                all_pred.append(to_numpy(pred))
                all_true.append(to_numpy(x_ts_eval.voltage[k]))
                continue
            elif forward_kind(model) == 'rnn':
                pred = model(x.to_packed(), return_all=False)
            elif forward_kind(model) in ('mlp', 'eed'):
                batched_state, _ = _batch_frames([x], edges)
                pred = model(batched_state, data_id=data_id, return_all=False)
            else:
                batched_state, batched_edges = _batch_frames([x], edges)
                pred, _, _ = model(
                    batched_state, batched_edges,
                    data_id=data_id, return_all=True,
                )

            all_pred.append(to_numpy(pred.squeeze()))
            all_true.append(to_numpy(y.squeeze()))

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)

    # Compute per-neuron metrics: transpose to (n_neurons, n_frames)
    rmse, pearson, feve, r2 = compute_trace_metrics(
        all_true.T, all_pred.T, label="test"
    )

    # Save results
    results_path = os.path.join(log_dir, f'results_test{test_suffix}.log')
    _onestep_fz = fisher_pool(pearson)
    _save_per_neuron_arrays(results_path, pearson, rmse)
    with open(results_path, 'w') as f:
        f.write(f'test_dataset: {test_ds}\n')
        f.write(f'n_frames: {len(all_pred)}\n')
        f.write(f'n_neurons: {n_neurons}\n')
        f.write(f'model: {netname}\n')
        f.write(_pearson_log_line(pearson))
        f.write(f'Pearson r (Fisher-z mean, sd): {_onestep_fz["z_mean"]:.4f} {_onestep_fz["z_sd"]:.4f}\n')
        f.write(f'RMSE: {np.mean(rmse):.4f} +/- {np.std(rmse):.4f}\n')
    logger.debug(f'results saved to {results_path}')

    if log_file:
        log_file.write('\n--- One-step test results ---\n')
        log_file.write(f'test_dataset: {test_ds}\n')
        log_file.write(f'onestep_pearson: {_onestep_fz["r_mean"]:.4f}\n')
        log_file.write(f'onestep_pearson_std: {_onestep_fz["r_sd_sym"]:.4f}\n')
        log_file.write(f'onestep_RMSE: {np.mean(rmse):.4f}\n')
        log_file.write(f'onestep_RMSE_std: {np.std(rmse):.4f}\n')

    # Stimulus baseline: each prediction is independent (no recurrence),
    # so rollout is meaningless — return after one-step metrics.
    if forward_kind(model) == 'stimulus':
        logger.info('stimulus model — skipping rollout (no recurrence)')
        return

    # --- Rollout evaluation ---
    # Start from initial voltages at t=0, predict autoregressively
    logger.info('running rollout evaluation ...')
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    x = x_ts_eval.frame(0)
    with torch.no_grad():
        hn.inject_hidden(model, x, 0, True)

    h_state = None
    c_state = None

    # EED rollout runs in pure latent space: encode the initial voltage
    # once, chain the evolver in z, decode each step. z_latent persists
    # across iterations so the activity-space re-encoding loop is bypassed.
    is_eed = forward_kind(model) == 'eed'
    z_latent = None
    if is_eed:
        z_latent = model.encoder(x.voltage.unsqueeze(0))
        logger.info('EED detected — running rollout in pure latent space')

    rollout_pred_list = []
    rollout_true_list = []
    rollout_stim_list = []
    stimuli_true_list = []   # true stimulus (input neurons only)
    stimuli_pred_list = []   # SIREN predicted stimulus (input neurons only)

    with torch.no_grad():
        for k in trange(n_eval_frames - 1, ncols=100, desc="rollout"):
            # Collect state before integration
            rollout_pred_list.append(to_numpy(x.voltage))
            rollout_true_list.append(to_numpy(x_ts_eval.frame(k).voltage))

            # Set stimulus from rollout data
            frame_k = x_ts_eval.frame(k)
            x.stimulus = frame_k.stimulus.clone()
            if frame_k.optogenetics_stimulus is not None:
                x.optogenetics_stimulus = frame_k.optogenetics_stimulus.clone()
            rollout_stim_list.append(to_numpy(x.stimulus))

            if has_visual_field:
                stimuli_true_list.append(to_numpy(x.stimulus[:model.n_input_neurons]))
                visual_input = model.forward_visual(x, k)
                stimuli_pred_list.append(to_numpy(visual_input.squeeze(-1)))
                x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                x.stimulus[model.n_input_neurons:] = 0

            # Model prediction (forward/rollout dispatch via FORWARD_KIND;
            # 'lstm'/'mlp_ode' are vestigial — no registered class sets them).
            _fk = forward_kind(model)
            if _fk == 'rnn':
                y, h_state = model(x.to_packed(), h=h_state, return_all=True)
            elif _fk == 'lstm':
                y, h_state, c_state = model(x.to_packed(), h=h_state, c=c_state, return_all=True)
            elif _fk == 'mlp_ode':
                v = x.voltage.unsqueeze(-1)
                if tc.training_selected_neurons:
                    I = x.stimulus.unsqueeze(-1)
                else:
                    I = x.stimulus[:sim.n_input_neurons].unsqueeze(-1)
                y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v
            elif is_eed:
                # Pure latent rollout: chain evolver in z, never re-encode x.
                stim_in = x.stimulus[:model.n_input_neurons].unsqueeze(0)
                z_stim = model.stimulus_encoder(stim_in)
                z_latent = z_latent + model.evolver(torch.cat([z_latent, z_stim], dim=1))
                v_next = model.decoder(z_latent).squeeze(0)
                # Emit dvdt so the shared Euler step lands on v_next exactly.
                y = ((v_next - x.voltage) / sim.delta_t).unsqueeze(-1)
            elif _fk == 'mlp':
                y = model(x, data_id=data_id, return_all=False)
            elif hasattr(tc, 'neural_ODE_training') and tc.neural_ODE_training:
                v0 = x.voltage.flatten()
                v_final, _ = integrate_neural_ode(
                    model=model, v0=v0, x_template=x,
                    edge_index=edges, data_id=data_id,
                    time_steps=1, delta_t=sim.delta_t,
                    neurons_per_sample=n_neurons, batch_size=1,
                    has_visual_field=has_visual_field,
                    x_ts=None, device=device,
                    k_batch=torch.tensor([k], device=device),
                    ode_method=tc.ode_method,
                    rtol=tc.ode_rtol, atol=tc.ode_atol,
                    adjoint=False, noise_level=0.0
                )
                y = (v_final.view(-1, 1) - x.voltage.unsqueeze(-1)) / sim.delta_t
            else:
                y = model(x, edges, data_id=data_id, return_all=False)

            # Integration step
            if _fk == 'mlp_ode':
                x.voltage = x.voltage + y.squeeze(-1)
            else:
                x.voltage = x.voltage + sim.delta_t * y.squeeze(-1)

            # Update hidden neuron voltages via SIREN or keep silent
            hn.inject_hidden(model, x, k + 1, True)

            # Guard against NaN / divergence from a poorly trained model
            if torch.isnan(x.voltage).any() or torch.isinf(x.voltage).any():
                logger.error(f"rollout diverged at frame {k} (NaN/Inf in voltage) — aborting")
                break
            x.voltage = torch.clamp(x.voltage, min=-100.0, max=100.0)

            # Calcium dynamics
            if sim.calcium_type == "leaky":
                if sim.calcium_activation == "softplus":
                    u = torch.nn.functional.softplus(x.voltage)
                elif sim.calcium_activation == "relu":
                    u = torch.nn.functional.relu(x.voltage)
                elif sim.calcium_activation == "tanh":
                    u = torch.tanh(x.voltage)
                elif sim.calcium_activation == "identity":
                    u = x.voltage.clone()
                x.calcium = x.calcium + (sim.delta_t / sim.calcium_tau) * (-x.calcium + u)
                x.calcium = torch.clamp(x.calcium, min=0.0)
                x.fluorescence = sim.calcium_alpha * x.calcium + sim.calcium_beta

    rollout_pred_arr = np.array(rollout_pred_list)   # (n_frames-1, n_neurons)
    rollout_true_arr = np.array(rollout_true_list)   # (n_frames-1, n_neurons)
    rollout_stim_arr = np.array(rollout_stim_list)   # (n_frames-1, n_neurons)

    activity_pred = rollout_pred_arr.T   # (n_neurons, n_frames-1)
    activity_true = rollout_true_arr.T   # (n_neurons, n_frames-1)
    stimulus_arr = rollout_stim_arr.T    # (n_neurons, n_frames-1)

    # Compute stimuli_R2: SIREN output vs true stimulus (with linear correction ax+b)
    stimuli_R2 = None
    stim_true_2d = None
    stim_pred_2d = None
    stim_pred_corrected_2d = None
    if has_visual_field and stimuli_true_list:
        stim_true_2d = np.array(stimuli_true_list)   # (n_frames, n_input_neurons)
        stim_pred_2d = np.array(stimuli_pred_list)   # (n_frames, n_input_neurons)
        # Global linear fit: true = a * pred + b
        pred_flat = stim_pred_2d.ravel()
        true_flat = stim_true_2d.ravel()
        A_fit = np.vstack([pred_flat, np.ones(len(pred_flat))]).T
        a_coeff, b_coeff = np.linalg.lstsq(A_fit, true_flat, rcond=None)[0]
        pred_corrected = a_coeff * stim_pred_2d + b_coeff
        ss_res = np.sum((stim_true_2d - pred_corrected) ** 2)
        ss_tot = np.sum((stim_true_2d - np.mean(stim_true_2d)) ** 2)
        stimuli_R2 = float(1 - ss_res / (ss_tot + 1e-16))
        stimuli_r  = float(stimuli_R2 ** 0.5) if stimuli_R2 >= 0 else 0.0
        stim_pred_corrected_2d = pred_corrected
        logger.info(f'stimuli_R2 (corrected a={a_coeff:.4f} b={b_coeff:.4f}): {stimuli_R2:.4f}  stimuli_r={stimuli_r:.4f}')

        # Generate stimuli GT vs Pred video
        if hasattr(x_ts_eval.frame(0), 'pos') and x_ts_eval.frame(0).pos is not None:
            from connectome_gnn.models.graph_trainer_inr import _generate_inr_video
            pos_input = to_numpy(x_ts_eval.frame(0).pos[:model.n_input_neurons])
            results_dir = os.path.join(log_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            _generate_inr_video(
                gt_np=stim_true_2d,
                predict_frame_fn=lambda k: stim_pred_2d[k],
                pos_np=pos_input,
                field_name='stimulus',
                output_folder=results_dir,
                n_frames=stim_true_2d.shape[0],
            )

    # Compute rollout metrics
    rmse_ro, pearson_ro, feve_ro, r2_ro = compute_trace_metrics(
        activity_true, activity_pred, label="rollout"
    )

    # Split rollout Pearson into hidden vs visible when a hidden-NGP model is in use.
    hidden_rollout_pearson = None
    visible_rollout_pearson = None
    if hn.has_hidden:
        _hidden_np = hn.hidden_ids.detach().cpu().numpy().astype(int)
        _mask = np.zeros(n_neurons, dtype=bool)
        _mask[_hidden_np] = True
        _hidden_pear = pearson_ro[_mask]
        _visible_pear = pearson_ro[~_mask]
        if _hidden_pear.size:
            hidden_rollout_pearson = float(np.nanmean(_hidden_pear))
        if _visible_pear.size:
            visible_rollout_pearson = float(np.nanmean(_visible_pear))

    # Save rollout metrics
    rollout_log_path = os.path.join(log_dir, f'results_rollout{test_suffix}.log')
    _rollout_fz = fisher_pool(pearson_ro)
    _save_per_neuron_arrays(rollout_log_path, pearson_ro, rmse_ro)
    with open(rollout_log_path, 'w') as f:
        f.write("Rollout Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"RMSE: {np.mean(rmse_ro):.4f} +/- {np.std(rmse_ro):.4f}\n")
        f.write(_pearson_log_line(pearson_ro))
        f.write(f'Pearson r (Fisher-z mean, sd): {_rollout_fz["z_mean"]:.4f} {_rollout_fz["z_sd"]:.4f}\n')
        if hidden_rollout_pearson is not None:
            f.write(f"hidden_rollout_pearson: {hidden_rollout_pearson:.3f} "
                    f"(n={int(_mask.sum())})\n")
            f.write(f"visible_rollout_pearson: {visible_rollout_pearson:.3f} "
                    f"(n={int((~_mask).sum())})\n")
        f.write(f"\nNumber of neurons evaluated: {n_neurons}\n")
        f.write(f"Frames evaluated: 0 to {n_eval_frames - 1}\n")
        if _use_train_data:
            f.write("Rollout data source: training (INR/NGP-T learned on training data)\n")
        if stimuli_R2 is not None:
            f.write(f"stimuli_R2: {stimuli_R2:.4f}\n")
            f.write(f"stimuli_r: {stimuli_r:.4f}\n")
    logger.debug(f'rollout metrics saved to {rollout_log_path}')

    # RMSE and Pearson r as a function of rollout step (every 500 frames), saved as CSV
    checkpoint_interval = 500
    n_total = rollout_pred_arr.shape[0]
    checkpoints = list(range(checkpoint_interval, n_total, checkpoint_interval)) + [n_total]
    rollout_csv_path = os.path.join(log_dir, f'results_rollout_by_step{test_suffix}.csv')
    with open(rollout_csv_path, 'w') as f:
        f.write("frame_start,frame_end,RMSE,pearson\n")
        prev = 0
        for cp in checkpoints:
            w_true = rollout_true_arr[prev:cp]   # (window, n_neurons)
            w_pred = rollout_pred_arr[prev:cp]
            rmse_w = float(np.sqrt(np.mean((w_true - w_pred) ** 2)))
            with np.errstate(invalid='ignore'):
                pearson_w = float(np.nanmean([
                    pearsonr(w_true[:, i], w_pred[:, i])[0]
                    for i in range(w_true.shape[1])
                    if np.std(w_true[:, i]) > 1e-8 and np.std(w_pred[:, i]) > 1e-8
                ]))
            f.write(f"{prev},{cp},{rmse_w:.4f},{pearson_w:.4f}\n")
            prev = cp
    logger.debug(f'rollout-by-step metrics saved to {rollout_csv_path}')

    if log_file:
        log_file.write('\n--- Rollout results ---\n')
        log_file.write(f'rollout_pearson: {_rollout_fz["r_mean"]:.4f}\n')
        log_file.write(f'rollout_pearson_std: {_rollout_fz["r_sd_sym"]:.4f}\n')
        if hidden_rollout_pearson is not None:
            log_file.write(f'hidden_rollout_pearson: {hidden_rollout_pearson:.4f}\n')
            log_file.write(f'visible_rollout_pearson: {visible_rollout_pearson:.4f}\n')
        log_file.write(f'rollout_RMSE: {np.mean(rmse_ro):.4f}\n')
        log_file.write(f'rollout_RMSE_std: {np.std(rmse_ro):.4f}\n')
        if stimuli_R2 is not None:
            log_file.write(f'stimuli_R2: {stimuli_R2:.4f}\n')
            log_file.write(f'stimuli_r: {stimuli_r:.4f}\n')

    # --- Rollout trace plots ---
    neuron_types = to_numpy(type_list).astype(int).squeeze()
    n_neuron_types = sim.n_neuron_types
    n_neurons = len(neuron_types)

    # Model-specific type names from the model's own ODE registry entry. The
    # KeyError/TypeError fallbacks remain only as defence for unregistered or
    # schema-mismatched checkpoints; every shipped model (incl.
    # drosophila_cx_voltage) is now registered, so they normally don't fire.
    from connectome_gnn.generators.ode_params import FlyVisCurrentODEParams, get_ode_params_class
    try:
        try:
            _OdeCls = get_ode_params_class(config.graph_model.signal_model_name)
        except KeyError:
            _OdeCls = FlyVisCurrentODEParams
        try:
            _ode_p = _OdeCls.load(graphs_data_path(config.dataset), device='cpu')
        except TypeError:
            # On-disk schema mismatch (e.g. registered class expects fields
            # we didn't save). Retry with the simpler FlyVisCurrentODEParams.
            _ode_p = FlyVisCurrentODEParams.load(graphs_data_path(config.dataset), device='cpu')
        if hasattr(_ode_p, 'type_names') and _ode_p.type_names:
            index_to_name = {i: name for i, name in enumerate(_ode_p.type_names)}
        else:
            index_to_name = INDEX_TO_NAME if n_neuron_types >= 65 else {i: f'Type{i}' for i in range(n_neuron_types)}
    except Exception:
        index_to_name = INDEX_TO_NAME if n_neuron_types >= 65 else {i: f'Type{i}' for i in range(n_neuron_types)}

    start_frame = 0
    end_frame = activity_true.shape[1]

    _dataset_base = os.path.basename(config.dataset)  # strip pre_folder (e.g. 'drosophila_cx/')
    filename_ = _dataset_base.split('flyvis_')[1] if 'flyvis_' in _dataset_base else re.sub(r'_\d{2}$', '', _dataset_base)

    # Neurons per type for "all" plot: more for small models
    if n_neuron_types <= 10:
        neurons_per_type = max(1, min(5, n_neurons // (n_neuron_types * 2)))
    else:
        neurons_per_type = 1

    # Build selected types: for flyvis use curated list, for small models use all types
    if n_neuron_types > 10:
        _selected_types = [55, 15, 43, 39, 35, 31, 23, 19, 12, 5]
        _selected_types = [t for t in _selected_types if t < n_neuron_types]
    else:
        _selected_types = list(range(n_neuron_types))

    for fig_name, selected_types in [
        ("selected", _selected_types),
        ("all", np.arange(0, n_neuron_types)),
    ]:
        neuron_indices = []
        neuron_labels = []
        _n_per_type = neurons_per_type if fig_name == "all" else 1
        for stype in selected_types:
            indices = np.where(neuron_types == stype)[0]
            if len(indices) > 0:
                for j in range(min(_n_per_type, len(indices))):
                    neuron_indices.append(indices[j])
                    type_name = index_to_name.get(int(stype), f'Type{stype}')
                    neuron_labels.append(type_name if j == 0 else '')

        if not neuron_indices:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(15, max(6, len(neuron_indices) * 0.4 + 2)))

        true_slice = activity_true[neuron_indices, start_frame:end_frame]
        stim_slice = stimulus_arr[neuron_indices, start_frame:end_frame]
        pred_slice = activity_pred[neuron_indices, start_frame:end_frame]

        # Auto-adjust step_v based on activity amplitude
        activity_std = np.std(true_slice)
        step_v = max(0.5, 3.0 * activity_std) if activity_std > 0 else 2.5
        lw = 2

        name_fontsize = 10 if len(neuron_indices) > 50 else 18

        # ground truth (green, thick)
        baselines = {}
        for i in range(len(neuron_indices)):
            baseline = np.mean(true_slice[i])
            baselines[i] = baseline
            ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw + 2, c='#66cc66', alpha=0.9,
                    label='ground truth' if i == 0 else None)
            if ((neuron_indices[i] == 0) or (len(neuron_indices) < 50)) and stim_slice[i].mean() > 0:
                ax.plot(stim_slice[i] - baseline + i * step_v, linewidth=0.7, c='red', alpha=0.9,
                        linestyle='--', label='stimuli' if i == 0 else None)

        # predictions (black, thin)
        for i in range(len(neuron_indices)):
            baseline = baselines[i]
            ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=0.7,
                    label='prediction' if i == 0 else None, c='black')

        for i in range(len(neuron_indices)):
            if neuron_labels[i]:
                ax.text(-end_frame * 0.025, i * step_v, neuron_labels[i],
                        fontsize=name_fontsize, va='bottom', ha='right', color='black')

        ax.set_ylim([-step_v, (len(neuron_indices) - 1) * step_v + step_v])
        ax.set_yticks([])
        ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
        ax.set_xticklabels([start_frame, end_frame // 2, end_frame], fontsize=16)
        ax.set_xlabel('frame', fontsize=20)
        ax.set_xlim([-end_frame * 0.03, end_frame + end_frame * 0.05])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
                  bbox_transform=fig.transFigure, fontsize=14, frameon=False)

        plt.tight_layout()
        _vis_tag = f"_{sim.visual_input_type}" if sim.visual_input_type else ""
        plt.savefig(f"{results_dir}/rollout_{filename_}{_vis_tag}_{fig_name}{test_suffix}.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    # ── Save rollout bundle ───────────────────────────────────────────────────
    bundle = dict(
        activity_true = activity_true,          # (n_neurons, n_frames)
        activity_pred = activity_pred,          # (n_neurons, n_frames)
        stimulus      = stimulus_arr,           # (n_neurons, n_frames)
        type_ids      = neuron_types,           # (n_neurons,) int
        type_names    = np.array(
            [index_to_name.get(i, f'Type{i}') for i in range(n_neuron_types)],
            dtype=object),
        config_name   = np.array(config.config_file),
    )

    # ── Add INR stimulus arrays (input-neuron resolution) when available ─────
    # These are the time x n_input_neurons arrays produced during rollout:
    # GT photoreceptor stimulus and the INR's predicted stimulus (raw +
    # linear-corrected). Used by figures/fig_stim_rollout_inr.py.
    if stim_true_2d is not None:
        bundle['stimulus_input_true'] = stim_true_2d.astype(np.float32)
        bundle['stimulus_input_pred'] = stim_pred_2d.astype(np.float32)
        if stim_pred_corrected_2d is not None:
            bundle['stimulus_input_pred_corrected'] = stim_pred_corrected_2d.astype(np.float32)

    # ── Add INR traces when the model has a hidden-neuron INR ─────────────────
    # All hidden neurons are stored (n_traces=None) so downstream figures can
    # render column-resolved error maps; per-neuron R² and 2-D positions are
    # included in the bundle.
    siren_r2 = None
    if hn.has_hidden and getattr(model, 'NNR_hidden', None) is not None:
        inr = _compute_inr_traces(model, x_ts_eval, hn.hidden_ids, device,
                                   n_traces=None, n_frames=activity_true.shape[1])
        bundle['inr_true']       = inr['gt_arr']         # (n_hidden, n_frames)
        bundle['inr_pred_raw']   = inr['pred_arr']       # (n_hidden, n_frames)
        bundle['inr_pred_corr']  = inr['pred_corr_arr']  # (n_hidden, n_frames)
        bundle['inr_global_ids'] = inr['global_ids']     # (n_hidden,)
        bundle['inr_r2_per']     = inr['r2_per']         # (n_hidden,) per-neuron R²
        if inr['global_pos'] is not None:
            bundle['inr_global_pos'] = inr['global_pos'] # (n_hidden, 2) (x, y)
        bundle['inr_type']       = np.array(inr['inr_type'])
        siren_r2 = inr['r2']
        logger.info(f'hidden INR R²: {siren_r2:.4f} (over {len(inr["r2_per"])} neurons)')
        if log_file:
            log_file.write(f'hidden_nnr_R2: {siren_r2:.4f}\n')

    np.savez(f"{results_dir}/rollout_bundle{test_suffix}.npz", **bundle)

    # ── Hidden-neuron trace plot (uses its own n_traces/n_frames for the PNG) ─
    if hn.has_hidden and getattr(model, 'NNR_hidden', None) is not None:
        from connectome_gnn.plot import plot_hidden_siren_traces
        _hp, _ap = plot_hidden_siren_traces(
            model, x_ts_eval, hn.hidden_ids, log_dir,
            epoch=0, N=0, device=device,
            n_traces=40, n_frames=min(2000, n_eval_frames),
            anchor_ids=hn.anchor_ids,
        )
        if siren_r2 is None:
            logger.info(f'hidden INR pearson: {_hp:.4f}')
            if log_file:
                log_file.write(f'hidden_nnr_R2: {_hp:.4f}\n')
        if _ap is not None:
            logger.info(f'anchor INR pearson: {_ap:.4f}')
            if log_file:
                log_file.write(f'anchor_nnr_pearson: {_ap:.4f}\n')

    logger.debug(f'rollout plots saved to {results_dir}/')


def data_test_gnn_special(
        config,
        visualize=True,
        style="color",
        verbose=False,
        best_model=None,
        step=5,
        n_rollout_frames=600,
        test_mode='',
        new_params=None,
        device=None,
        rollout_without_noise: bool = False,
        log_file=None,
):

    if "black" in style:
        plt.style.use("dark_background")
        mc = 'white'
    else:
        plt.style.use("default")
        mc = 'black'

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    log_dir = log_path(config.config_file)

    torch.random.fork_rng(devices=device)
    if sim.seed is not None:
        torch.random.manual_seed(sim.seed)
        np.random.seed(sim.seed)

    logger.info(
        f"testing... {model_config.particle_model_name} {model_config.mesh_model_name} seed: {sim.seed}")


    if tc.training_selected_neurons:
        n_neurons = 13741
        n_neuron_types = 1736
    else:
        n_neurons = sim.n_neurons
        n_neuron_types = sim.n_neuron_types

    logger.info(f"noise_model_level: {sim.noise_model_level}")
    warm_up_length = 100

    run = 0

    extent = 8
    # Import only what's needed for mixed functionality
    import flyvis
    from flyvis import Network, NetworkView
    from flyvis.datasets.sintel import AugmentedSintel
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config

    from connectome_gnn.generators.flyvis_ode import (
        FlyVisODE,
        get_photoreceptor_positions_from_net,
        group_by_direction_and_function,
    )
    from connectome_gnn.utils import setup_flyvis_model_path

    setup_flyvis_model_path()
    # Initialize datasets
    if "DAVIS" in sim.visual_input_type or "mixed" in sim.visual_input_type:
        # determine dataset roots: use config list if provided, otherwise fall back to default
        if sim.datavis_roots:
            datavis_root_list = [os.path.join(r, "JPEGImages/480p") for r in sim.datavis_roots]
        else:
            datavis_root_list = [os.path.join(get_datavis_root_dir(), "JPEGImages/480p")]

        for root in datavis_root_list:
            assert os.path.exists(root), f"video data not found at {root}"

        video_config = {
            "n_frames": 50,
            "max_frames": 80,
            "flip_axes": [0, 1],
            "n_rotations": [0, 90, 180, 270],
            "temporal_split": True,
            "dt": sim.delta_t,
            "interpolate": True,
            "boxfilter": dict(extent=extent, kernel_size=13),
            "vertical_splits": 1,
            "center_crop_fraction": 0.6,
            "augment": False,
            "unittest": False,
            "shuffle_sequences": True,
            "shuffle_seed": sim.seed,
        }

        # create dataset(s)
        if len(datavis_root_list) == 1:
            davis_dataset = AugmentedVideoDataset(root_dir=datavis_root_list[0], **video_config)
        else:
            datasets = [AugmentedVideoDataset(root_dir=root, **video_config) for root in datavis_root_list]
            davis_dataset = CombinedVideoDataset(datasets)
            logger.info(f"combined {len(datasets)} video datasets: {len(davis_dataset)} total sequences")
    else:
        davis_dataset = None

    if "DAVIS" in sim.visual_input_type:
        stimulus_dataset = davis_dataset
    else:
        sintel_config = {
            "sintel_path": flyvis.sintel_dir,
            "n_frames": 19,
            "flip_axes": [0, 1],
            "n_rotations": [0, 1, 2, 3, 4, 5],
            "temporal_split": True,
            "dt": sim.delta_t,
            "interpolate": True,
            "boxfilter": dict(extent=extent, kernel_size=13),
            "vertical_splits": 3,
            "center_crop_fraction": 0.7
        }
        stimulus_dataset = AugmentedSintel(**sintel_config)

    # Initialize network
    config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    config_net.connectome.extent = extent
    net = Network(**config_net)
    nnv = NetworkView(f"flow/{sim.ensemble_id}/{sim.model_id}")
    trained_net = nnv.init_network(checkpoint=0)
    net.load_state_dict(trained_net.state_dict())
    torch.set_grad_enabled(False)

    ode_params = FlyVisCurrentODEParams.from_flyvis_network(net, device=device)
    edge_index = ode_params.edge_index

    if sim.n_extra_null_edges > 0:
        logger.info(f"adding {sim.n_extra_null_edges} extra null edges (mode={sim.null_edges_mode})...")
        import random
        src_np = edge_index[0].cpu().numpy()
        dst_np = edge_index[1].cpu().numpy()
        existing_edges = set(zip(src_np, dst_np))
        extra_edges = []

        if sim.null_edges_mode == 'per_column':
            from collections import Counter
            out_degree = Counter(src_np.tolist())
            total_real = edge_index.shape[1]
            ratio = sim.n_extra_null_edges / total_real
            targets_by_source = {}
            for s, d in zip(src_np, dst_np):
                targets_by_source.setdefault(int(s), set()).add(int(d))
            all_neurons = list(range(n_neurons))
            for source in range(n_neurons):
                deg = out_degree.get(source, 0)
                if deg == 0:
                    continue
                n_false = max(1, int(round(deg * ratio)))
                existing_targets = targets_by_source.get(source, set())
                candidates = [t for t in all_neurons if t != source and t not in existing_targets]
                if len(candidates) <= n_false:
                    chosen = candidates
                else:
                    chosen = random.sample(candidates, n_false)
                for t in chosen:
                    extra_edges.append([source, t])
                    existing_targets.add(t)
            logger.info(f"per_column: added {len(extra_edges)} false edges "
                        f"(requested ratio {ratio:.2f}, effective {len(extra_edges)/total_real:.2f})")
        else:
            max_attempts = sim.n_extra_null_edges * 10
            attempts = 0
            while len(extra_edges) < sim.n_extra_null_edges and attempts < max_attempts:
                source = random.randint(0, n_neurons - 1)
                target = random.randint(0, n_neurons - 1)
                if (source, target) not in existing_edges and source != target:
                    extra_edges.append([source, target])
                    existing_edges.add((source, target))
                attempts += 1

        if extra_edges:
            extra_edge_index = torch.tensor(extra_edges, dtype=torch.long, device=device).t()
            edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
            ode_params.edge_index = edge_index
            ode_params.W = torch.cat([ode_params.W, torch.zeros(len(extra_edges), device=device)])

    pde = FlyVisODE(ode_params=ode_params, g_phi=torch.nn.functional.relu, params=sim.params, model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)
    pde_modified = FlyVisODE(ode_params=ode_params.clone(), g_phi=torch.nn.functional.relu, params=sim.params, model_type=model_config.signal_model_name, n_neuron_types=n_neuron_types, device=device)


    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type, config=config, device=device)


    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/best_model_with_*.pt")
        if not files:
            files = glob.glob(f"{log_dir}/models/*.pt")
        assert len(files), 'no model checkpoints found in models/ directory'
        best_model = max(files, key=os.path.getmtime)
        logger.info(f'best model: {best_model}')

    # If it's a relative path (no slashes), assume it's in models/ directory
    if '/' not in best_model:
        netname = f"{log_dir}/models/{best_model}"
    else:
        netname = best_model
    logger.info(f'load {netname} ...')
    state_dict = torch.load(netname, map_location=device, weights_only=False)
    migrate_state_dict(state_dict)
    model.load_state_dict(state_dict['model_state_dict'], strict=False)

    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)

    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]
    grouped_types = np.array([group_by_direction_and_function(t) for t in node_types_str])
    unique_types, node_types_int = np.unique(node_types, return_inverse=True)

    X1 = torch.tensor(np.stack((x_coords, y_coords), axis=1), dtype=torch.float32, device=device)

    xc, yc = get_equidistant_points(n_points=n_neurons - x_coords.shape[0])
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    X1 = torch.cat((X1, pos[torch.randperm(pos.size(0), device=device)]), dim=0)

    _ss_value = getattr(sim, 'steady_state_value', 0.5)
    state = net.steady_state(t_pre=2.0, dt=sim.delta_t, batch_size=1, value=_ss_value)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)

    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)

    calcium_init = torch.rand(n_neurons, dtype=torch.float32, device=device)
    x = NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=X1,
        group_type=torch.tensor(grouped_types, dtype=torch.long, device=device),
        neuron_type=torch.tensor(node_types_int, dtype=torch.long, device=device),
        voltage=initial_state,
        stimulus=net.stimulus().squeeze(),
        calcium=calcium_init,
        fluorescence=sim.calcium_alpha * calcium_init + sim.calcium_beta,
    )

    if tc.training_selected_neurons:
        selected_neuron_ids = tc.selected_neuron_ids
        selected_neuron_ids = np.array(selected_neuron_ids).astype(int)
        logger.info(f'testing single neuron id {selected_neuron_ids} ...')
        x_selected = x.subset(selected_neuron_ids)

    # Mixed sequence setup
    if "mixed" in sim.visual_input_type:
        mixed_types = ["sintel", "davis", "blank", "noise"]
        mixed_cycle_lengths = [60, 60, 30, 60]  # Different lengths for each type
        mixed_current_type = 0
        mixed_frame_count = 0
        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
        if not davis_dataset:
            sintel_config_mixed = {
                "n_frames": 19,
                "flip_axes": [0, 1],
                "n_rotations": [0, 1, 2, 3, 4, 5],
                "temporal_split": True,
                "dt": sim.delta_t,
                "interpolate": True,
                "boxfilter": dict(extent=extent, kernel_size=13),
                "vertical_splits": 3,
                "center_crop_fraction": 0.7
            }
            davis_dataset = AugmentedSintel(**sintel_config_mixed)
        sintel_iter = iter(stimulus_dataset)
        davis_iter = iter(davis_dataset)
        current_sintel_seq = None
        current_davis_seq = None
        sintel_frame_idx = 0
        davis_frame_idx = 0

    target_frames = n_rollout_frames

    if 'full' in test_mode:
        target_frames = sim.n_frames
        step = 25000
    else:
        step = 10
    logger.info(f'plot activity frames 0-{target_frames}...')

    dataset_length = len(stimulus_dataset)
    frames_per_sequence = 35
    total_frames_per_pass = dataset_length * frames_per_sequence
    num_passes_needed = (target_frames // total_frames_per_pass) + 1

    y_list = []
    x_list = []
    x_generated_list = []
    x_generated_modified_list = []

    x_generated = x.clone()
    x_generated_modified = x.clone()

    # Initialize RNN hidden state (forward/rollout dispatch via FORWARD_KIND)
    _fk = forward_kind(model)
    if _fk == 'rnn':
        h_state = None
    if _fk == 'lstm':
        h_state = None
        c_state = None

    it = sim.start_frame
    id_fig = 0

    tile_labels = None
    tile_codes_torch = None
    tile_period = None
    tile_idx = 0
    tile_contrast = sim.tile_contrast
    n_columns = sim.n_input_neurons // 8
    tile_seed = sim.seed

    edges = ode_params.edge_index

    if ('test_ablation' in test_mode) & ('MLP' not in model_config.signal_model_name) & (_fk != 'rnn') & (_fk != 'lstm'):
        #  test_mode="test_ablation_100"
        ablation_ratio = int(test_mode.split('_')[-1]) / 100
        if ablation_ratio > 0:
            logger.info(f'test ablation ratio {ablation_ratio}')
        n_ablation = int(edges.shape[1] * ablation_ratio)
        index_ablation = np.random.choice(np.arange(edges.shape[1]), n_ablation, replace=False)

        with torch.no_grad():
            pde.ode_params.W[index_ablation] = 0
            pde_modified.ode_params.W[index_ablation] = 0
            model.W[index_ablation] = 0

    if 'test_modified' in test_mode:
        noise_W = float(test_mode.split('_')[-1])
        if noise_W > 0:
            logger.info(f'test modified W with noise level {noise_W}')
            noise_p_W = torch.randn_like(pde.ode_params.W) * noise_W
            pde_modified.ode_params.W = pde.ode_params.W.clone() + noise_p_W

        plot_weight_comparison(pde.ode_params.W, pde_modified.ode_params.W, f"{log_dir}/results/weight_comparison_{noise_W}.png")


    fig_style = dark_style
    index_to_name = INDEX_TO_NAME


    # Main loop #####################################

    with torch.no_grad():
        for pass_num in range(num_passes_needed):
            for data_idx, data in enumerate(tqdm(stimulus_dataset, desc="processing stimulus data", ncols=100)):

                sequences = data["lum"]
                # Sample flash parameters for each subsequence if flash stimulus is requested
                if "flash" in sim.visual_input_type:
                    # Sample flash duration from specific values: 1, 2, 5, 10, 20 frames
                    flash_duration_options = [1, 2, 5] #, 10, 20]
                    flash_cycle_frames = flash_duration_options[
                        torch.randint(0, len(flash_duration_options), (1,), device=device).item()
                    ]

                    flash_intensity = torch.abs(torch.rand(sim.n_input_neurons, device=device) * 0.5 + 0.5)
                if "mixed" in sim.visual_input_type:
                    if mixed_frame_count >= current_cycle_length:
                        mixed_current_type = (mixed_current_type + 1) % 4
                        mixed_frame_count = 0
                        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
                    current_type = mixed_types[mixed_current_type]

                    if current_type == "sintel":
                        if current_sintel_seq is None or sintel_frame_idx >= current_sintel_seq["lum"].shape[0]:
                            try:
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                            except StopIteration:
                                sintel_iter = iter(stimulus_dataset)
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                        sequences = current_sintel_seq["lum"]
                        start_frame = sintel_frame_idx
                    elif current_type == "davis":
                        if current_davis_seq is None or davis_frame_idx >= current_davis_seq["lum"].shape[0]:
                            try:
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                            except StopIteration:
                                davis_iter = iter(davis_dataset)
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                        sequences = current_davis_seq["lum"]
                        start_frame = davis_frame_idx
                    else:
                        start_frame = 0
                # Determine sequence length based on stimulus type
                if "flash" in sim.visual_input_type:
                    sequence_length = 60  # Fixed 60 frames for flash sequences
                else:
                    sequence_length = sequences.shape[0]

                for frame_id in range(sequence_length):

                    if "flash" in sim.visual_input_type:
                        # Generate repeating flash stimulus
                        current_flash_frame = frame_id % (flash_cycle_frames * 2)  # Create on/off cycle
                        x.stimulus[:] = 0
                        if current_flash_frame < flash_cycle_frames:
                            x.stimulus[:sim.n_input_neurons] = flash_intensity
                    elif "mixed" in sim.visual_input_type:
                        current_type = mixed_types[mixed_current_type]

                        if current_type == "blank":
                            x.stimulus[:] = 0
                        elif current_type == "noise":
                            x.stimulus[:sim.n_input_neurons] = torch.relu(
                                0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device) * 0.5)
                        else:
                            actual_frame_id = (start_frame + frame_id) % sequences.shape[0]
                            frame = sequences[actual_frame_id][None, None]
                            net.stimulus.add_input(frame)
                            x.stimulus = net.stimulus().squeeze()
                            if current_type == "sintel":
                                sintel_frame_idx += 1
                            elif current_type == "davis":
                                davis_frame_idx += 1
                        mixed_frame_count += 1
                    elif "tile_mseq" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            # 1) Cluster photoreceptors into columns based on (u,v)
                            tile_labels_np = assign_columns_from_uv(
                                u_coords, v_coords, n_columns, random_state=tile_seed
                            )  # shape: (sim.n_input_neurons,)

                            # 2) Build per-column m-sequences (±1) with random phase per column
                            base = mseq_bits(p=8, seed=tile_seed).astype(np.float32)  # ±1, shape (255,)
                            rng = np.random.RandomState(tile_seed)
                            phases = rng.randint(0, base.shape[0], size=n_columns)
                            tile_codes_np = np.stack([np.roll(base, ph) for ph in phases], axis=0)  # (n_columns, 255), ±1

                            # 3) Convert to torch on the right device/dtype; keep as ±1 (no [0,1] mapping here)
                            tile_codes_torch = torch.from_numpy(tile_codes_np).to(x.device,
                                                                                  dtype=torch.float32)  # (n_columns, 255), ±1
                            tile_labels = torch.from_numpy(tile_labels_np).to(x.device,
                                                                              dtype=torch.long)  # (sim.n_input_neurons,)
                            tile_period = tile_codes_torch.shape[1]
                            tile_idx = 0

                        # 4) Baseline for all neurons (mean luminance), then write per-column values to PRs
                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]  # (n_columns,), ±1 before knobs
                        # Apply the two simple knobs per frame on ±1 codes
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx)
                        )
                        # Map to [0,1] with your contrast convention and broadcast via labels
                        col_vals_01 = 0.5 + (tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[:sim.n_input_neurons] = col_vals_01[tile_labels]

                        tile_idx += 1
                    elif "tile_blue_noise" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            # Label columns and build neighborhood graph
                            tile_labels_np, col_centers = compute_column_labels(u_coords, v_coords, n_columns, seed=tile_seed)
                            try:
                                adj = build_neighbor_graph(col_centers, k=6)
                            except Exception:
                                from scipy.spatial.distance import pdist, squareform
                                D = squareform(pdist(col_centers))
                                nn = np.partition(D + np.eye(D.shape[0]) * 1e9, 1, axis=1)[:, 1]
                                radius = 1.3 * np.median(nn)
                                adj = [set(np.where((D[i] > 0) & (D[i] <= radius))[0].tolist()) for i in
                                       range(len(col_centers))]

                            tile_labels = torch.from_numpy(tile_labels_np).to(x.device, dtype=torch.long)
                            tile_period = 257
                            tile_idx = 0

                            # Pre-generate ±1 codes (keep ±1; no [0,1] mapping here)
                            tile_codes_torch = torch.empty((n_columns, tile_period), dtype=torch.float32, device=x.device)
                            rng = np.random.RandomState(tile_seed)
                            for t in range(tile_period):
                                mask = greedy_blue_mask(adj, n_columns, target_density=0.5, rng=rng)  # boolean mask
                                vals = np.where(mask, 1.0, -1.0).astype(np.float32)  # ±1
                                # NOTE: do not apply flip prob here; we do it uniformly via the helper per frame below
                                tile_codes_torch[:, t] = torch.from_numpy(vals).to(x.device, dtype=torch.float32)

                        # Baseline luminance
                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]  # (n_columns,), ±1 before knobs

                        # Apply the two simple knobs per frame on ±1 codes
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx)
                        )

                        # Map to [0,1] with contrast and broadcast via labels
                        col_vals_01 = 0.5 + (tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[:sim.n_input_neurons] = col_vals_01[tile_labels]

                        tile_idx += 1
                    else:
                        frame = sequences[frame_id][None, None]
                        net.stimulus.add_input(frame)
                        if (sim.only_noise_visual_input > 0):
                            if (sim.visual_input_type == "") | (it == 0) | ("50/50" in sim.visual_input_type):
                                x.stimulus[:sim.n_input_neurons] = torch.relu(
                                    0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32,
                                                     device=device) * sim.only_noise_visual_input / 2)
                        else:
                            if sim.blank_freq > 0:
                                if (data_idx % sim.blank_freq > 0):
                                    x.stimulus = net.stimulus().squeeze()
                                else:
                                    x.stimulus[:] = 0
                            else:
                                x.stimulus = net.stimulus().squeeze()
                            if sim.noise_visual_input > 0:
                                x.stimulus[:sim.n_input_neurons] = x.stimulus[:sim.n_input_neurons] + torch.randn(sim.n_input_neurons,
                                                                                                  dtype=torch.float32,
                                                                                                  device=device) * sim.noise_visual_input

                    x_generated.stimulus = x.stimulus.clone()
                    y_generated = pde(x_generated, edge_index, has_field=False)

                    x_generated_modified.stimulus = x.stimulus.clone()
                    y_generated_modified = pde_modified(x_generated_modified, edge_index, has_field=False)

                    if 'visual' in model_config.field_type:
                        visual_input = model.forward_visual(x, it)
                        x.stimulus[:model.n_input_neurons] = visual_input.squeeze(-1)
                        x.stimulus[model.n_input_neurons:] = 0

                    # Prediction step
                    if tc.training_selected_neurons:
                        x_selected.stimulus = x.stimulus[selected_neuron_ids].clone().detach()
                        if _fk == 'rnn':
                            y, h_state = model(x_selected.to_packed(), h=h_state, return_all=True)
                        elif _fk == 'lstm':
                            y, h_state, c_state = model(x_selected.to_packed(), h=h_state, c=c_state, return_all=True)
                        elif _fk == 'mlp_ode':
                            v = x_selected.voltage.unsqueeze(-1)
                            I = x_selected.stimulus.unsqueeze(-1)
                            y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v  # Return as delta
                        elif _fk in ('mlp', 'eed'):
                            y = model(x_selected.to_packed(), data_id=None, return_all=False)

                    else:
                        if _fk == 'rnn':
                            y, h_state = model(x.to_packed(), h=h_state, return_all=True)
                        elif _fk == 'lstm':
                            y, h_state, c_state = model(x.to_packed(), h=h_state, c=c_state, return_all=True)
                        elif _fk == 'mlp_ode':
                            v = x.voltage.unsqueeze(-1)
                            I = x.stimulus[:sim.n_input_neurons].unsqueeze(-1)
                            y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v  # Return as delta
                        elif _fk in ('mlp', 'eed'):
                            y = model(x.to_packed(), data_id=None, return_all=False)
                        elif tc.neural_ODE_training:
                            data_id = torch.zeros((x.n_neurons, 1), dtype=torch.int, device=device)
                            v0 = x.voltage.flatten()
                            v_final, _ = integrate_neural_ode(
                                model=model,
                                v0=v0,
                                x_template=x,
                                edge_index=edge_index,
                                data_id=data_id,
                                time_steps=1,
                                delta_t=sim.delta_t,
                                neurons_per_sample=n_neurons,
                                batch_size=1,
                                has_visual_field='visual' in model_config.field_type,
                                x_ts=None,
                                device=device,
                                k_batch=torch.tensor([it], device=device),
                                ode_method=tc.ode_method,
                                rtol=tc.ode_rtol,
                                atol=tc.ode_atol,
                                adjoint=False,
                                noise_level=0.0
                            )
                            y = (v_final.view(-1, 1) - x.voltage.unsqueeze(-1)) / sim.delta_t
                        else:
                            data_id = torch.zeros((x.n_neurons, 1), dtype=torch.int, device=device)
                            y = model(x, edge_index, data_id=data_id, return_all=False)

                    # Save states (pack to legacy (N, 9) numpy for downstream analysis)
                    x_generated_list.append(to_numpy(x_generated.to_packed().clone().detach()))
                    x_generated_modified_list.append(to_numpy(x_generated_modified.to_packed().clone().detach()))

                    if tc.training_selected_neurons:
                        x_list.append(to_numpy(x_selected.to_packed().clone().detach()))
                    else:
                        x_list.append(to_numpy(x.to_packed().clone().detach()))

                    # Integration step
                    # Optionally disable process noise at test time, even if model was trained with noise
                    effective_noise_level = 0.0 if rollout_without_noise else sim.noise_model_level
                    if effective_noise_level > 0:
                        x_generated.voltage = x_generated.voltage + sim.delta_t * y_generated.squeeze(-1) + torch.randn(
                            n_neurons, dtype=torch.float32, device=device
                        ) * effective_noise_level
                        x_generated_modified.voltage = x_generated_modified.voltage + sim.delta_t * y_generated_modified.squeeze(-1) + torch.randn(
                            n_neurons, dtype=torch.float32, device=device
                        ) * effective_noise_level
                    else:
                        x_generated.voltage = x_generated.voltage + sim.delta_t * y_generated.squeeze(-1)
                        x_generated_modified.voltage = x_generated_modified.voltage + sim.delta_t * y_generated_modified.squeeze(-1)

                    if tc.training_selected_neurons:
                        if _fk == 'mlp_ode':
                            x_selected.voltage = x_selected.voltage + y.squeeze(-1)  # y already contains full update
                        else:
                            x_selected.voltage = x_selected.voltage + sim.delta_t * y.squeeze(-1)
                        if (it <= warm_up_length) and _fk in ('rnn', 'lstm'):
                            x_selected.voltage = x_generated.voltage[selected_neuron_ids].clone()
                    else:
                        if _fk == 'mlp_ode':
                            x.voltage = x.voltage + y.squeeze(-1)  # y already contains full update
                        else:
                            x.voltage = x.voltage + sim.delta_t * y.squeeze(-1)
                        if (it <= warm_up_length) and _fk in ('rnn', 'lstm'):
                            x.voltage = x_generated.voltage.clone()

                    # Guard against NaN / divergence from a poorly trained model
                    v_model = x_selected.voltage if tc.training_selected_neurons else x.voltage
                    if torch.isnan(v_model).any() or torch.isinf(v_model).any():
                        logger.error(f"rollout diverged at iteration {it} (NaN/Inf in voltage) — aborting")
                        break
                    if tc.training_selected_neurons:
                        x_selected.voltage = torch.clamp(x_selected.voltage, min=-100.0, max=100.0)
                    else:
                        x.voltage = torch.clamp(x.voltage, min=-100.0, max=100.0)

                    if sim.calcium_type == "leaky":
                        # Voltage-driven activation
                        if sim.calcium_activation == "softplus":
                            u = torch.nn.functional.softplus(x.voltage)
                        elif sim.calcium_activation == "relu":
                            u = torch.nn.functional.relu(x.voltage)
                        elif sim.calcium_activation == "tanh":
                            u = torch.tanh(x.voltage)
                        elif sim.calcium_activation == "identity":
                            u = x.voltage.clone()

                        x.calcium = x.calcium + (sim.delta_t / sim.calcium_tau) * (-x.calcium + u)
                        x.calcium = torch.clamp(x.calcium, min=0.0)
                        x.fluorescence = sim.calcium_alpha * x.calcium + sim.calcium_beta

                        y = (x.calcium - torch.tensor(x_list[-1][:, 7], dtype=torch.float32, device=device)).unsqueeze(-1) / sim.delta_t

                    y_list.append(to_numpy(y.clone().detach()))

                    if (it > 0) & (it < 100) & (it % step == 0) & visualize & (not tc.training_selected_neurons):
                        num = f"{id_fig:06}"
                        id_fig += 1
                        plot_spatial_activity_grid(
                            positions=to_numpy(x.pos),
                            voltages=to_numpy(x.voltage),
                            stimulus=to_numpy(x.stimulus[:sim.n_input_neurons]),
                            neuron_types=to_numpy(x.neuron_type).astype(int),
                            output_path=f"{log_dir}/tmp_recons/Fig_{run}_{num}.png",
                            calcium=to_numpy(x.calcium) if sim.calcium_type != "none" else None,
                            n_input_neurons=sim.n_input_neurons,
                            style=fig_style,
                        )

                    it = it + 1
                    if it >= target_frames:
                        break
                if it >= target_frames:
                    break

            if it >= target_frames:
                break
    logger.info(f"generated {len(x_list)} frames total")


    if visualize:
        logger.info('generating lossless video ...')

        output_name = os.path.basename(config.dataset).split('flyvis_')[1] if 'flyvis_' in config.dataset else re.sub(r'_\d{2}$', '', os.path.basename(config.dataset))
        src = f"{log_dir}/tmp_recons/Fig_0_000000.png"
        dst = f"{log_dir}/results/input_{output_name}.png"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

        generate_compressed_video_mp4(output_dir=f"{log_dir}/results", run=run,
                                        output_name=output_name,framerate=20)

        # files = glob.glob(f'./{log_dir}/tmp_recons/*')
        # for f in files:
        #     os.remove(f)


    x_list = np.array(x_list)
    x_generated_list = np.array(x_generated_list)
    x_generated_modified_list = np.array(x_generated_modified_list)
    y_list = np.array(y_list)

    neuron_types = node_types_int

    if sim.calcium_type != "none":
        # Use calcium (index 7)
        activity_true = x_generated_list[:, :, 7].squeeze().T  # (n_neurons, n_frames)
        activity_pred = x_list[:, :, 7].squeeze().T
    else:
        # Use voltage (index 3)
        activity_true = x_generated_list[:, :, 3].squeeze().T
        visual_input_true = x_generated_list[:, :, 4].squeeze().T
        activity_true_modified = x_generated_modified_list[:, :, 3].squeeze().T
        activity_pred = x_list[:, :, 3].squeeze().T


    start_frame = 0
    end_frame = target_frames


    if tc.training_selected_neurons:           # MLP, RNN and ODE are trained on limted number of neurons

        logger.info(f"evaluating on selected neurons only: {selected_neuron_ids}")
        x_generated_list = x_generated_list[:, selected_neuron_ids, :]
        x_generated_modified_list = x_generated_modified_list[:, selected_neuron_ids, :]
        neuron_types = neuron_types[selected_neuron_ids]

        true_slice = activity_true[selected_neuron_ids, start_frame:end_frame]
        visual_input_slice = visual_input_true[selected_neuron_ids, start_frame:end_frame]
        pred_slice = activity_pred[start_frame:end_frame]

        rmse_all, pearson_all, feve_all, r2_all = compute_trace_metrics(true_slice, pred_slice, "selected neurons")
        _sel_fz = fisher_pool(pearson_all)

        # Log rollout metrics to file
        rollout_log_path = f"{log_dir}/results_rollout.log"
        _save_per_neuron_arrays(rollout_log_path, pearson_all, rmse_all)
        with open(rollout_log_path, 'w') as f:
            f.write("Rollout Metrics for Selected Neurons\n")
            f.write("="*60 + "\n")
            f.write(f"RMSE: {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f} [{np.min(rmse_all):.4f}, {np.max(rmse_all):.4f}]\n")
            f.write(f"Pearson r: {_sel_fz['r_mean']:.3f} ± {_sel_fz['r_sd_sym']:.3f} [{_sel_fz['r_lo']:.3f}, {_sel_fz['r_hi']:.3f}]\n")
            f.write(f"Pearson r (Fisher-z mean, sd): {_sel_fz['z_mean']:.4f} {_sel_fz['z_sd']:.4f}\n")
            # f.write(f"R²: {np.nanmean(r2_all):.3f} ± {np.nanstd(r2_all):.3f} [{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]\n")
            # f.write(f"FEVE: {np.mean(feve_all):.3f} ± {np.std(feve_all):.3f} [{np.min(feve_all):.3f}, {np.max(feve_all):.3f}]\n")
            f.write(f"\nNumber of neurons evaluated: {len(selected_neuron_ids)}\n")

        if len(selected_neuron_ids)==1:
            pred_slice = pred_slice[None,:]

        _dataset_base = os.path.basename(config.dataset)  # strip pre_folder (e.g. 'drosophila_cx/')
        filename_ = _dataset_base.split('flyvis_')[1] if 'flyvis_' in _dataset_base else re.sub(r'_\d{2}$', '', _dataset_base)

        # Determine which figures to create
        if len(selected_neuron_ids) > 50:
            # Create sample: take the last 10 neurons from selected_neuron_ids
            sample_indices = list(range(len(selected_neuron_ids) - 10, len(selected_neuron_ids)))

            figure_configs = [
                ("all", list(range(len(selected_neuron_ids)))),
                ("sample", sample_indices)
            ]
        else:
            figure_configs = [("", list(range(len(selected_neuron_ids))))]

        for fig_suffix, neuron_plot_indices in figure_configs:
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            step_v = 2.5
            lw = 6

            # Adjust fontsize based on number of neurons being plotted
            name_fontsize = 10 if len(neuron_plot_indices) > 50 else 18

            # Plot ground truth (green, thick) — all traces first
            baselines = {}
            for plot_idx, i in enumerate(trange(len(neuron_plot_indices), ncols=100, desc=f"plotting {fig_suffix}")):
                neuron_idx = neuron_plot_indices[i]
                baseline = np.mean(true_slice[neuron_idx])
                baselines[plot_idx] = baseline
                ax.plot(true_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=lw+2, c='#66cc66', alpha=0.9,
                        label='ground truth' if plot_idx == 0 else None)
                # Plot visual input only for neuron_id = 0
                if ((selected_neuron_ids[neuron_idx] == 0) | (len(neuron_plot_indices) < 50)) and visual_input_slice[neuron_idx].mean() > 0:
                    ax.plot(visual_input_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=1, c='yellow', alpha=0.9,
                            linestyle='--', label='stimuli')

            # Plot predictions (black, thin) — on top
            for plot_idx, i in enumerate(range(len(neuron_plot_indices))):
                neuron_idx = neuron_plot_indices[i]
                baseline = baselines[plot_idx]
                ax.plot(pred_slice[neuron_idx] - baseline + plot_idx * step_v, linewidth=1, c=mc,
                        label='prediction' if plot_idx == 0 else None)

            for plot_idx, i in enumerate(neuron_plot_indices):
                type_idx = int(to_numpy(x.neuron_type[selected_neuron_ids[i]]).item())
                ax.text(-50, plot_idx * step_v, f'{index_to_name[type_idx]}', fontsize=name_fontsize, va='bottom', ha='right', color='black')

            ax.set_ylim([-step_v, len(neuron_plot_indices) * (step_v + 0.25 + 0.15 * (len(neuron_plot_indices)//50))])
            ax.set_yticks([])
            ax.set_xlabel('time (frames)', fontsize=20)
            ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
            ax.set_xticklabels([start_frame, end_frame//2, end_frame], fontsize=16)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.legend(loc='upper right', fontsize=14, frameon=False)
            ax.set_xlim([0, end_frame - start_frame + 100])

            plt.tight_layout()
            save_suffix = f"_{fig_suffix}" if fig_suffix else ""
            _vis_tag = f"_{sim.visual_input_type}" if sim.visual_input_type else ""
            plt.savefig(f"{log_dir}/results/rollout_{filename_}{_vis_tag}{save_suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()

    else:

        rmse_all, pearson_all, feve_all, r2_all = compute_trace_metrics(activity_true, activity_pred, "all neurons")
        _all_fz = fisher_pool(pearson_all)

        # Log rollout metrics to file
        rollout_log_path = f"{log_dir}/results_rollout.log"
        _save_per_neuron_arrays(rollout_log_path, pearson_all, rmse_all)
        with open(rollout_log_path, 'w') as f:
            f.write("Rollout Metrics for All Neurons\n")
            f.write("="*60 + "\n")
            f.write(f"RMSE: {np.mean(rmse_all):.4f} ± {np.std(rmse_all):.4f} [{np.min(rmse_all):.4f}, {np.max(rmse_all):.4f}]\n")
            f.write(f"Pearson r: {_all_fz['r_mean']:.3f} ± {_all_fz['r_sd_sym']:.3f} [{_all_fz['r_lo']:.3f}, {_all_fz['r_hi']:.3f}]\n")
            f.write(f"Pearson r (Fisher-z mean, sd): {_all_fz['z_mean']:.4f} {_all_fz['z_sd']:.4f}\n")
            # f.write(f"R²: {np.nanmean(r2_all):.3f} ± {np.nanstd(r2_all):.3f} [{np.nanmin(r2_all):.3f}, {np.nanmax(r2_all):.3f}]\n")
            # f.write(f"FEVE: {np.mean(feve_all):.3f} ± {np.std(feve_all):.3f} [{np.min(feve_all):.3f}, {np.max(feve_all):.3f}]\n")
            f.write(f"\nNumber of neurons evaluated: {len(activity_true)}\n")
            f.write(f"Frames evaluated: {start_frame} to {end_frame}\n")

        # Write to analysis log file for Claude
        if log_file:
            # log_file.write(f"test_R2: {np.nanmean(r2_all):.4f}\n")
            log_file.write(f"test_pearson: {_all_fz['r_mean']:.4f}\n")

        _dataset_base = os.path.basename(config.dataset)  # strip pre_folder (e.g. 'drosophila_cx/')
        filename_ = _dataset_base.split('flyvis_')[1] if 'flyvis_' in _dataset_base else re.sub(r'_\d{2}$', '', _dataset_base)

        # Create two figures with different neuron type selections
        for fig_name, selected_types in [
            ("selected", [55, 15, 43, 39, 35, 31, 23, 19, 12, 5]),  # L1, Mi12, Mi2, R1, T1, T4a, T5a, Tm1, Tm4, Tm9
            ("all", np.arange(0, n_neuron_types))
        ]:
            neuron_indices = []
            neuron_labels = []
            for stype in selected_types:
                indices = np.where(neuron_types == stype)[0]
                if len(indices) > 0:
                    neuron_indices.append(indices[0])
                    type_name = index_to_name.get(int(stype), f'Type{stype}')
                    neuron_labels.append(type_name)

            if not neuron_indices:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(15, max(6, len(neuron_indices) * 0.4 + 2)))

            true_slice = activity_true[neuron_indices, start_frame:end_frame]
            visual_input_slice = visual_input_true[neuron_indices, start_frame:end_frame]
            pred_slice = activity_pred[neuron_indices, start_frame:end_frame]

            # Auto-adjust step_v based on activity amplitude
            activity_std = np.std(true_slice)
            step_v = max(0.5, 3.0 * activity_std) if activity_std > 0 else 2.5
            lw = 2

            # Adjust fontsize based on number of neurons plotted
            name_fontsize = 10 if len(neuron_indices) > 50 else 18

            # Plot ground truth (green, thick) — all traces first
            baselines = {}
            for i in range(len(neuron_indices)):
                baseline = np.mean(true_slice[i])
                baselines[i] = baseline
                ax.plot(true_slice[i] - baseline + i * step_v, linewidth=lw+2, c='#66cc66', alpha=0.9,
                        label='ground truth' if i == 0 else None)
                # Plot visual input for neuron 0 OR when fewer than 50 neurons
                if ((neuron_indices[i] == 0) | (len(neuron_indices) < 50)) and visual_input_slice[i].mean() > 0:
                    ax.plot(visual_input_slice[i] - baseline + i * step_v, linewidth=0.7, c='red', alpha=0.9,
                            linestyle='--', label='stimuli')

            # Plot predictions (black, thin) — on top
            for i in range(len(neuron_indices)):
                baseline = baselines[i]
                ax.plot(pred_slice[i] - baseline + i * step_v, linewidth=0.7, label='prediction' if i == 0 else None, c=mc)

            # Add neuron type labels
            for i in range(len(neuron_indices)):
                ax.text(-end_frame * 0.025, i * step_v, neuron_labels[i], fontsize=name_fontsize, va='bottom', ha='right', color='black')

            ax.set_ylim([-step_v, len(neuron_indices) * (step_v + 0.25 + 0.15 * (len(neuron_indices)//50))])
            ax.set_yticks([])
            ax.set_xticks([0, (end_frame - start_frame) // 2, end_frame - start_frame])
            ax.set_xticklabels([start_frame, end_frame//2, end_frame], fontsize=16)
            ax.set_xlabel('frame', fontsize=20)
            ax.set_xlim([-end_frame * 0.03, end_frame + end_frame * 0.05])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            ax.legend(loc='upper right', fontsize=14, frameon=False)

            plt.tight_layout()
            _vis_tag = f"_{sim.visual_input_type}" if sim.visual_input_type else ""
            plt.savefig(f"{log_dir}/results/rollout_{filename_}{_vis_tag}_{fig_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

        if ('test_ablation' in test_mode) or ('test_inactivity' in test_mode):
            np.save(f"{log_dir}/results/activity_modified.npy", activity_true_modified)
            np.save(f"{log_dir}/results/activity_modified_pred.npy", activity_pred)
        else:
            np.save(f"{log_dir}/results/activity_true.npy", activity_true)
            np.save(f"{log_dir}/results/activity_pred.npy", activity_pred)




# ============================================================================
# Cortex (Yang 2019) task tester
# ============================================================================

def data_test_cortex_task_gnn(config, best_model=None, device=None, log_file=None):
    """Test a TaskRNN (free-W) on a cortex task: load test zarrs, rollout the
    trained model on the first 10 consecutive test trials, and save a 2x10
    kinograph (row 0 = GT motor, row 1 = predicted motor) to log_dir.

    Also reports per-trial direction_acc + aggregate metrics across the full
    test split via compute_cortex_task_metrics.
    """
    from connectome_gnn.models.cortex_eval import (
        compute_cortex_task_metrics,
        save_cortex_test_kinograph,
    )

    tc = config.training
    model_config = config.graph_model
    ct = config.task.cortex

    log_dir = log_path(config.config_file)
    os.makedirs(log_dir, exist_ok=True)

    # --- Load test data ---
    root = graphs_data_path(config.dataset)
    logger.info(f'[cortex test] loading from {root}/test/...')
    u_test = torch.from_numpy(load_raw_array(f"{root}/test/stimulus.zarr")).to(device)
    y_test = torch.from_numpy(load_raw_array(f"{root}/test/target.zarr")).to(device)
    cm_test = torch.from_numpy(load_raw_array(f"{root}/test/c_mask.zarr")).to(device)
    logger.info(f'  test shapes: u={tuple(u_test.shape)}  y={tuple(y_test.shape)}  '
                f'cm={tuple(cm_test.shape)}')

    # --- Rebuild model from registry; load best checkpoint ---
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type,
                         config=config, device=device)
    ckpt_dir = os.path.join(log_dir, 'models')
    # Find latest checkpoint (best_model arg is the epoch index if int)
    if isinstance(best_model, int):
        ckpt_path = os.path.join(
            ckpt_dir,
            f'best_model_with_{tc.n_runs - 1}_graphs_{best_model}.pt')
    else:
        # Pick the highest-epoch checkpoint in ckpt_dir
        cand = sorted(glob.glob(os.path.join(
            ckpt_dir, f'best_model_with_{tc.n_runs - 1}_graphs_*.pt')))
        if not cand:
            raise FileNotFoundError(
                f'no cortex checkpoint found in {ckpt_dir}; train first.')
        ckpt_path = cand[-1]
    logger.info(f'  loading checkpoint: {ckpt_path}')
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # --- Rollout on first 10 consecutive test trials ---
    n_kino = min(10, u_test.shape[0])
    with torch.no_grad():
        y_hat, _ = model(u_test[:n_kino])
    stim_kino = [u_test[i] for i in range(n_kino)]
    preds_kino = [y_hat[i] for i in range(n_kino)]
    tgts_kino = [y_test[i] for i in range(n_kino)]
    cms_kino = [cm_test[i] for i in range(n_kino)]
    per_trial = compute_cortex_task_metrics(preds_kino, tgts_kino, cms_kino)
    _r2_f = per_trial["r2_filtered"]; _da_f = per_trial["direction_acc_filtered"]
    _pct = per_trial["pct_outliers"]
    _c_r2 = r2_color(_r2_f) if _r2_f == _r2_f else ""
    _c_da = r2_color(_da_f) if _da_f == _da_f else ""
    _c_pct = ANSI_ORANGE if (_pct == _pct and _pct > 15) else ""
    logger.info(
        f'  10-trial  '
        f'{_c_r2}R²={_r2_f:.3f}{ANSI_RESET} ({per_trial["r2"]:.3f})  '
        f'{_c_da}dir_acc={_da_f:.3f}{ANSI_RESET} '
        f'({per_trial["direction_acc"]:.3f})  '
        f'{_c_pct}outlier={_pct:.0f}%{ANSI_RESET if _c_pct else ""}  '
        f'loss={per_trial["loss"]:.2e}'
    )

    rule_name = (ct.rules[0] if getattr(ct, "rules", None) else "cortex")
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    kino_path = os.path.join(results_dir, f'test_kinograph_{rule_name}.png')
    save_cortex_test_kinograph(
        stim_kino, preds_kino, tgts_kino, cms_kino,
        output_path=kino_path, rule_name=rule_name, n_trials=n_kino,
    )
    logger.info(f'  saved kinograph: {kino_path}')

    # --- Aggregate metrics over the full test split ---
    with torch.no_grad():
        y_hat_full, _ = model(u_test)
    preds = [y_hat_full[i] for i in range(u_test.shape[0])]
    tgts = [y_test[i] for i in range(u_test.shape[0])]
    cms = [cm_test[i] for i in range(u_test.shape[0])]
    full = compute_cortex_task_metrics(preds, tgts, cms)
    _r2_f = full["r2_filtered"]; _da_f = full["direction_acc_filtered"]
    _pct = full["pct_outliers"]
    _c_r2 = r2_color(_r2_f) if _r2_f == _r2_f else ""
    _c_da = r2_color(_da_f) if _da_f == _da_f else ""
    _c_pct = ANSI_ORANGE if (_pct == _pct and _pct > 15) else ""
    logger.info(
        f'  full test (n={u_test.shape[0]}):  '
        f'{_c_r2}R²={_r2_f:.4f}{ANSI_RESET} ({full["r2"]:.4f})  '
        f'{_c_da}dir_acc={_da_f:.4f}{ANSI_RESET} '
        f'({full["direction_acc"]:.4f})  '
        f'{_c_pct}outlier={_pct:.1f}%{ANSI_RESET if _c_pct else ""}  '
        f'loss={full["loss"]:.2e}'
    )


# ============================================================================
# Path-integration task test (DrosophilaCxTaskRNN / DrosophilaCxTaskGNN)
# ============================================================================

def data_test_place_task(config, best_model=None, device=None, log_file=None):
    """Test the heading+distance+place model (drosophila_cx_pi_place).

    Computes held-out metrics (heading RMSE, place-code score, population-
    vector position RMSE, distance correlation) over a test sample, writes
    them to results/test_metrics.npz + results_path_integration.log, and
    renders MP4 animations of a few high-motion trials (the animated 5-panel
    place diagnostic) into results/."""
    from connectome_gnn.models.bump_attractor_eval import (
        animate_place_trial, animate_torus_trial, _save_place_snapshot,
    )
    from connectome_gnn.task_state import TaskTrials

    tc = config.training
    mc = config.graph_model
    log_dir = log_path(config.config_file)
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f'[place test] results dir: {results_dir}')

    root = graphs_data_path(config.dataset)
    te = TaskTrials.load(f"{root}/test")
    u_test = te.stimulus.to(device)            # (N, T, 4)
    y_test = te.target.to(device)              # (N, T, 5) [cos,sin,d,x,y]
    dt = float(config.task.swim_integration.dt)
    logger.info(f'[place test] u={tuple(u_test.shape)} y={tuple(y_test.shape)}')

    model = create_model(mc.signal_model_name, aggr_type=mc.aggr_type,
                         config=config, device=device)
    ckpt_dir = os.path.join(log_dir, 'models')

    def _epoch_of(p):
        try:
            return int(os.path.basename(p).rsplit('_', 1)[1].split('.')[0])
        except Exception:
            return -1
    bm = int(best_model) if isinstance(best_model, str) and best_model.isdigit() \
        else best_model
    if isinstance(bm, int):
        ckpt_path = os.path.join(
            ckpt_dir, f'best_model_with_{tc.n_runs - 1}_graphs_{bm}.pt')
    else:
        cand = sorted(glob.glob(os.path.join(
            ckpt_dir, f'best_model_with_{tc.n_runs - 1}_graphs_*.pt')),
            key=_epoch_of)
        if not cand:
            raise FileNotFoundError(f'no checkpoint in {ckpt_dir}; train first.')
        ckpt_path = cand[-1]
    logger.info(f'[place test] loading {ckpt_path}')
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device,
                   weights_only=False)['model_state_dict'])
    model.eval()
    K = int(model.net2_n_place)
    centers = model.place_centers
    sig2 = 2.0 * model.place_sigma * model.place_sigma

    # --- metrics over a test sample ---------------------------------------
    nseg = int(min(256, u_test.shape[0]))
    rmse_deg, place_score, pos_rmse, dist_r = [], [], [], []
    with torch.no_grad():
        for i in range(0, nseg, 64):
            u = u_test[i:i + 64]; y = y_test[i:i + 64]
            yh, _ = model(u, pos0=y[:, 0, 3:5])      # PI anchor (start pos)
            tht = torch.atan2(y[..., 1], y[..., 0])
            thp = torch.atan2(yh[..., 1], yh[..., 0])
            d = thp[:, 10:] - tht[:, 10:]
            e = torch.atan2(torch.sin(d), torch.cos(d))
            rmse_deg += (e.pow(2).mean(1).sqrt() * 180.0 / np.pi).cpu().tolist()
            place = yh[..., 3:3 + K]                       # tanh field ∈(-1,1)
            xy = y[..., 3:5]
            g = model.place_field(xy)                      # raw Gaussian target
            pr, gg = torch.relu(place[:, 10:]), g[:, 10:]
            cs = ((pr * gg).sum(-1)
                  / (pr.norm(dim=-1) * gg.norm(dim=-1) + 1e-9))
            place_score += cs.mean(1).cpu().tolist()
            pp = model.place_prob(place)                   # rectified pop code
            xyd = (model.decode_position_anchored(pp, y[:, 0, 3:5])
                   if getattr(model, "place_anchor", False)
                   else model.decode_position(pp))
            pos_rmse += model._pos_sqerr(xyd[:, 10:], xy[:, 10:]).mean(1).sqrt().cpu().tolist()
            dt_, dd = y[..., 2].cpu().numpy(), yh[..., 2].cpu().numpy()
            for b in range(u.shape[0]):
                a1, a2 = dt_[b, 10:], dd[b, 10:]
                if a1.std() > 1e-6 and a2.std() > 1e-6:
                    dist_r.append(float(np.corrcoef(a1, a2)[0, 1]))
    summary = (f"[place test] n={nseg} ckpt={os.path.basename(ckpt_path)}  "
               f"heading_rmse={np.mean(rmse_deg):.2f}±{np.std(rmse_deg):.2f}deg  "
               f"place_score={np.mean(place_score):.3f}±{np.std(place_score):.3f}  "
               f"pos_rmse={np.mean(pos_rmse):.3f}±{np.std(pos_rmse):.3f}  "
               f"dist_r={np.mean(dist_r):.3f}±{np.std(dist_r):.3f}")
    logger.info(summary)
    np.savez(os.path.join(results_dir, 'test_metrics.npz'),
             heading_rmse_deg=np.asarray(rmse_deg),
             place_score=np.asarray(place_score),
             pos_rmse=np.asarray(pos_rmse), dist_r=np.asarray(dist_r))
    with open(os.path.join(log_dir, 'results_path_integration.log'), 'a') as f:
        f.write(summary + "\n")

    # --- MP4 animations of the most dynamic trials ------------------------
    # Rank by total path length AND total turning (not just bounding-box
    # span), so the chosen trials have lots of movement and lots of turns.
    xy = y_test[..., 3:5]
    path_len = (xy[:, 1:] - xy[:, :-1]).norm(dim=-1).sum(1)            # total path
    th = torch.atan2(y_test[..., 1], y_test[..., 0])
    dth = th[:, 1:] - th[:, :-1]
    turning = torch.atan2(torch.sin(dth), torch.cos(dth)).abs().sum(1)  # Σ|Δθ|
    score = (path_len / (path_len.std() + 1e-9)
             + turning / (turning.std() + 1e-9))
    idx = torch.argsort(score, descending=True)[:3].cpu().tolist()
    _grid = bool(getattr(model, "grid_mode", False))
    for j, ti in enumerate(idx):
        # grid task lives on a torus → animate on the 3-D torus donut; the
        # bounded place task → the arena animation.
        pfx = 'torus_trial' if _grid else 'place_trial'
        out = os.path.join(results_dir, f'{pfx}_{j}_idx{ti}.mp4')
        if _grid:
            animate_torus_trial(model, u_test[ti], y_test[ti], out,
                                dt=dt, device=device)
        else:
            animate_place_trial(model, u_test[ti], y_test[ti], out,
                                dt=dt, device=device)
        logger.info(f'[place test] wrote {out}')
    # one static 5-panel snapshot for quick reference
    _save_place_snapshot(model, log_dir, _epoch_of(ckpt_path), 0,
                         u_test[idx[:1]], y_test[idx[:1]], device)
    logger.info('[place test] done')


def data_test_path_integration_task(
    config, best_model=None, device=None, log_file=None,
    anatomy_voltage: bool = False, anatomy_voltage_type_groups=None,
):
    """Test the trained CX path-integration model.

    Runs two evaluations and saves figures + metrics to
    `<log_dir>/results/path_integration/`:

    (a) 5 random test trials (held-out 10k split, T=1000 frames each):
        forward the model, plot input/wrapped-HD/output traces vs ground
        truth, report per-trial RMSE_deg and Pearson r on the unwrapped
        decoded angle vs ground-truth heading.

    (b) 5 deterministic constant-ω sweeps at ω ∈ {-120, -60, 0, 60, 120}
        deg/s, T=2000 frames (= 20s, 2x the training horizon). Same per-
        trial plotting and metrics; characterises long-horizon stability
        and ω-asymmetry.

    Aggregate mean ± std across both rollout sets is written to
    `<log_dir>/results_path_integration.log`.
    """
    from connectome_gnn.models.bump_attractor_eval import (
        _deterministic_sweep_rollout,
        path_integration_accuracy_from_data,
    )
    from connectome_gnn.plot import (
        plot_function_dynamics,
        plot_integration_gain,
        plot_task_pi_traces,
    )

    tc = config.training
    model_config = config.graph_model

    log_dir = log_path(config.config_file)
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f'[pi test] results dir: {results_dir}')

    # Machine-loadable export. Every raw decoded-vs-true trace + per-trial
    # metric produced below is stashed here and written once to
    # results/test_metrics.npz at the end, so downstream analysis loads the
    # heading and displacement trajectories with a single np.load(). Keys are
    # grouped by drive: ``*_random_*`` are the held-out naturalistic (OU)
    # swim trials, ``*_sweep_*`` / ``gain_*`` are the deterministic
    # constant-velocity probes.
    npz_bundle: dict = {}

    # --- Load test data ----------------------------------------------------
    # Refactor: single TaskTrials.load instead of four zarr reads.
    # On legacy datasets without theta_hd.zarr / is_stop.zarr these fields
    # come back as None and we fall back to reconstructing theta_hd from
    # y = (cos θ, sin θ); is_stop defaults to zeros so the rollout-corr
    # masking is a no-op.
    from connectome_gnn.task_state import TaskTrials
    root = graphs_data_path(config.dataset)
    logger.info(f'[pi test] loading from {root}/test/...')
    trials_test = TaskTrials.load(f"{root}/test")
    u_test_np = trials_test.stimulus.numpy()
    y_test_np = trials_test.target.numpy()
    if trials_test.theta_hd is not None:
        # New writer persists the integrated (unwrapped) heading directly.
        theta_test_np = trials_test.theta_hd.numpy().astype(np.float32)
    else:
        theta_wrap = np.arctan2(y_test_np[:, :, 1], y_test_np[:, :, 0])
        theta_test_np = np.unwrap(theta_wrap, axis=-1).astype(np.float32)
    if trials_test.is_stop is not None and \
            tuple(trials_test.is_stop.shape) == theta_test_np.shape:
        is_stop_test_np = trials_test.is_stop.numpy().astype(np.float32)
    else:
        is_stop_test_np = np.zeros(theta_test_np.shape, dtype=np.float32)
    u_test = torch.from_numpy(u_test_np).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)
    logger.info(f'  shapes: u={tuple(u_test.shape)}  y={tuple(y_test.shape)}')

    # --- task-mode channel selection (mirrors the trainer's slicing) -------
    # The on-disk target shape (3 or 4 cols) plus task_targets picks the
    # profile. Mirrors the load-time slicing in
    # graph_trainer._data_train_drosophila_cx_task — see that block for the
    # detailed table.
    # Keyed by (n_in_disk, n_out_disk, sorted_task_targets) — mirrors
    # the trainer table so propriocep-split (5-col disk stim) routes
    # cleanly at test time.
    _PROFILE_BY_TARGET = {
        # scalar_xi (4-col stim, 3-col target):
        (4, 3, ("rotation",)):                ([0, 2, 3],    [0, 1]),
        (4, 3, ("translation",)):             ([1],          [2]),
        (4, 3, ("rotation", "translation")):  ([0, 1, 2, 3], [0, 1, 2]),
        # position_2d (4-col stim, 4-col target):
        (4, 4, ("rotation",)):                ([0, 2, 3],    [0, 1]),
        (4, 4, ("position_2d",)):             ([0, 1, 2, 3], [0, 1, 2, 3]),
        # propriocep-split (5-col stim — col 2 carries v_proprio):
        (5, 3, ("rotation",)):                ([0, 3, 4],       [0, 1]),
        (5, 3, ("translation",)):             ([1, 2],          [2]),
        (5, 3, ("rotation", "translation")):  ([0, 1, 2, 3, 4], [0, 1, 2]),
        (5, 4, ("rotation",)):                ([0, 3, 4],       [0, 1]),
        (5, 4, ("position_2d",)):             ([0, 1, 2, 3, 4], [0, 1, 2, 3]),
        # heading-only supervision but KEEP v_fwd in the input (latent
        # path-integration probe): full 4-ch input, 2-col heading target.
        (4, 3, ("rotation_vfwd",)):           ([0, 1, 2, 3], [0, 1]),
        (4, 4, ("rotation_vfwd",)):           ([0, 1, 2, 3], [0, 1]),
        # rotation_torus: 6-col target, Net1-only (heading metrics use cols 0,1).
        (4, 6, ("rotation_torus",)):          ([0, 1, 2, 3], [0, 1, 2, 3, 4, 5]),
        # conjunction_input: 6-ch stimulus (base 4 + vx, vy)
        (6, 6, ("rotation_torus",)):          ([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
    }
    _RECOGNISED = ("rotation", "translation", "position_2d", "rotation_vfwd",
                   "rotation_torus")
    _task_raw = list(getattr(tc, 'task_targets', None) or [])
    task_targets_canonical = [t for t in _RECOGNISED if t in _task_raw]
    _task_key = tuple(task_targets_canonical)
    has_position_2d  = "position_2d" in task_targets_canonical
    has_rotation     = ("rotation" in task_targets_canonical
                        or "rotation_vfwd" in task_targets_canonical
                        or has_position_2d
                        or not task_targets_canonical)
    has_translation  = "translation" in task_targets_canonical
    if u_test.shape[-1] >= 4 and _task_key:
        n_in_disk = int(u_test.shape[-1])
        n_out_disk = int(y_test.shape[-1])
        _profile_key = (n_in_disk, n_out_disk, _task_key)
        if _profile_key not in _PROFILE_BY_TARGET:
            raise ValueError(
                f"task_targets={_task_raw!r} not a valid projection of "
                f"an on-disk stimulus with {n_in_disk} cols / target "
                f"with {n_out_disk} cols. Recognised "
                f"(n_in, n_out, targets) keys: "
                f"{sorted(_PROFILE_BY_TARGET.keys())}."
            )
        in_cols, out_cols = _PROFILE_BY_TARGET[_profile_key]
        u_test = u_test[..., in_cols].contiguous()
        y_test = y_test[..., out_cols].contiguous()
        u_test_np = u_test_np[..., in_cols]
        y_test_np = y_test_np[..., out_cols]
        logger.info(f'  task_targets={task_targets_canonical} (on-disk y has '
                    f'{n_out_disk} cols) → sliced to in_cols={in_cols} '
                    f'out_cols={out_cols}; u={tuple(u_test.shape)} '
                    f'y={tuple(y_test.shape)}')

    # --- Rebuild model from registry; load best checkpoint -----------------
    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type,
                         config=config, device=device)
    ckpt_dir = os.path.join(log_dir, 'models')
    if isinstance(best_model, int):
        ckpt_path = os.path.join(
            ckpt_dir,
            f'best_model_with_{tc.n_runs - 1}_graphs_{best_model}.pt')
    else:
        cand = sorted(glob.glob(os.path.join(
            ckpt_dir, f'best_model_with_{tc.n_runs - 1}_graphs_*.pt')))
        if not cand:
            raise FileNotFoundError(
                f'no path-integration checkpoint found in {ckpt_dir}; '
                f'train first.')
        ckpt_path = cand[-1]
    logger.info(f'  loading checkpoint: {ckpt_path}')
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # Heading-bin ablation: the model was built with n_input = 1 + K (ω drive +
    # K-bin one-hot heading cue) and emits K-bin logits. The on-disk stimulus is
    # the cos/sin layout, so convert u_test's trailing (cos θ₀, sin θ₀) cue to
    # the K-bin one-hot the model expects — mirrors the trainer's load-time
    # conversion (graph_trainer use_bins path). Model outputs are decoded back
    # to (cos,sin) downstream: path_integration_accuracy_from_data does it
    # internally, block (a) via softmax_logits_to_cos_sin_np, and the
    # deterministic sweep builds its own bin cue + decode.
    use_bins = bool(getattr(model, 'use_heading_bins', False))
    K_bins = int(getattr(model, 'n_heading_bins', 64))
    if use_bins:
        from connectome_gnn.models.heading_bins import (
            convert_cos_sin_input_to_bin_cue_torch,
        )
        u_test = convert_cos_sin_input_to_bin_cue_torch(u_test, K_bins)
        logger.info(f'  heading-bin ablation: K={K_bins}; converted u_test cue '
                    f'to K-bin one-hot → u={tuple(u_test.shape)}')

    # --- Aggregate test pi_acc on full split (T=u_test.shape[1]) -----------
    # Test does no backward pass, so we use a much larger batch than training.
    # The training-side `tc.batch_size` is tuned to fit BPTT memory; here we
    # only need forward passes so 256 fits comfortably even for the GNN.
    # pi_acc is cos-similarity on a (cos, sin) target — meaningful only when
    # heading is supervised. Skip in translation-only mode.
    test_bs = max(int(tc.batch_size), 256)
    if has_rotation:
        full_pi = path_integration_accuracy_from_data(
            model, u_test, y_test, warmup=10, batch_size=test_bs,
        )
        logger.info(f'  full test pi_acc (n={u_test.shape[0]}, '
                    f'T={u_test.shape[1]}): {full_pi:.4f}')
    else:
        full_pi = float('nan')
        logger.info(f'  full test pi_acc: n/a (translation-only mode, '
                    f'no heading target)')

    # Task-block resolver: this function supports both path_integration
    # and swim_integration test runs. ``dt`` lives on whichever sub-block
    # this run uses, so we look it up via the task's task_type once.
    _ttype = str(getattr(config.task, 'task_type', '')).lower()
    if _ttype == 'swim_integration':
        _task_block = config.task.swim_integration
    else:
        _task_block = config.task.path_integration
    _task_dt = float(_task_block.dt)

    # --- Torus position metric (rotation_torus: 6-col output) --------------
    # Heading (cols 0,1) is scored above; here we score whether the 2-D TORUS
    # POSITION is recovered. φx=(cols 2,3 as cos,sin), φy=(cols 4,5). We report
    # the circular RMSE of the decoded vs true phases and a drift time constant
    # tau_pos (first time the mean circular error crosses 30°). Chance (decoded
    # phase independent of truth) ≈ 104° circ-RMSE.
    if int(y_test.shape[-1]) >= 6:
        n_t = int(min(512, u_test.shape[0]))
        idx_t = np.arange(n_t)
        with torch.no_grad():
            yh_t = model(u_test[idx_t])[0].cpu().numpy()
        yt_t = y_test[idx_t].cpu().numpy()
        def _cerr(a, b):
            d = a - b
            return np.abs(np.arctan2(np.sin(d), np.cos(d)))
        phx_t = np.arctan2(yt_t[..., 3], yt_t[..., 2]); phx_d = np.arctan2(yh_t[..., 3], yh_t[..., 2])
        phy_t = np.arctan2(yt_t[..., 5], yt_t[..., 4]); phy_d = np.arctan2(yh_t[..., 5], yh_t[..., 4])
        _ex = _cerr(phx_d[:, 10:], phx_t[:, 10:]); _ey = _cerr(phy_d[:, 10:], phy_t[:, 10:])
        torus_phi_rmse_deg = float(np.sqrt(((_ex ** 2 + _ey ** 2) / 2).mean()) * 180.0 / np.pi)
        _mean_err_t = np.sqrt((_cerr(phx_d, phx_t) ** 2 + _cerr(phy_d, phy_t) ** 2) / 2).mean(0)
        _over = np.where(_mean_err_t > np.deg2rad(30.0))[0]
        torus_tau_pos_s = float(_over[0] * _task_dt) if len(_over) else float('nan')
        logger.info(f'  TORUS POSITION: phi circ-RMSE={torus_phi_rmse_deg:.1f}deg  '
                    f'tau_pos(30deg)={torus_tau_pos_s:.1f}s  (chance ~104deg)')
        npz_bundle.update(torus_phi_rmse_deg=np.float32(torus_phi_rmse_deg),
                          torus_tau_pos_s=np.float32(torus_tau_pos_s))
        with open(os.path.join(log_dir, 'results_path_integration.log'), 'a') as _f:
            _f.write(f'torus_phi_circ_rmse_deg={torus_phi_rmse_deg:.2f}  '
                     f'tau_pos_30deg_s={torus_tau_pos_s:.2f}\n')

    # --- (a) random test trials: per-trial RMSE/Pearson over a 512-trial
    # sample (robust mean for cross-run comparison); only a few are plotted.
    # plot_task_pi_traces / _per_trial_heading_metrics consume (cos θ, sin θ)
    # targets — meaningful only when heading is supervised, so we gate them
    # on has_rotation. In translation-only mode we skip the panel entirely
    # (the deterministic v_fwd sweep below is the analogous diagnostic).
    rng = np.random.default_rng(config.training.seed)
    if has_rotation:
        n_metric = int(min(512, u_test.shape[0]))
        idx_sample = np.sort(rng.choice(u_test.shape[0], size=n_metric, replace=False))
        with torch.no_grad():
            y_pred_sample, _ = model(u_test[idx_sample])
        y_pred_sample_np = y_pred_sample.cpu().numpy()
        if use_bins:
            # K-bin logits → (cos,sin) circular-mean decode so the heading
            # metrics + trace plot consume the same 2-D layout as the cos/sin
            # models. Same convention as the deterministic-sweep decode.
            from connectome_gnn.models.heading_bins import (
                softmax_logits_to_cos_sin_np,
            )
            y_pred_sample_np = softmax_logits_to_cos_sin_np(
                y_pred_sample_np, K_bins)

        metrics_random = _per_trial_heading_metrics(
            y_pred_sample_np, theta_test_np[idx_sample],
        )
        _rm = np.array([m['rmse_deg'] for m in metrics_random], dtype=float)
        _pr = np.array([m['pearson'] for m in metrics_random], dtype=float)
        # OU (naturalistic) held-out heading trials: raw decoded (cos,sin),
        # true heading, and per-trial metrics over the full 512-trial sample.
        npz_bundle.update(
            heading_full_pi_acc=np.float32(full_pi),
            heading_random_idx=idx_sample.astype(np.int64),
            heading_random_theta_true=theta_test_np[idx_sample].astype(np.float32),
            heading_random_pred=y_pred_sample_np.astype(np.float32),
            # Full target for these OU trials: cols [0,1]=(cosθ,sinθ); on
            # joint tasks col 2 is true d and cols [2,3] are true (x,y), so
            # displacement on the OU trials is recoverable for "both" /
            # position_2d runs without a separate sample.
            heading_random_target_true=y_test_np[idx_sample].astype(np.float32),
            heading_random_u=u_test_np[idx_sample].astype(np.float32),
            heading_random_rmse_deg=_rm.astype(np.float32),
            heading_random_pearson=_pr.astype(np.float32),
        )
        logger.info(
            f'  {n_metric} random test trials: '
            f'rmse={np.nanmean(_rm):.2f}±{np.nanstd(_rm):.2f}°  '
            f'r={np.nanmean(_pr):.4f}±{np.nanstd(_pr):.4f}'
        )
        n_show = int(min(5, n_metric))
        idx_show = idx_sample[:n_show]
        random_plot_path = os.path.join(results_dir, 'test_random_trials.png')
        plot_task_pi_traces(
            u=u_test_np[idx_show],
            y=y_test_np[idx_show],
            theta_hd=theta_test_np[idx_show],
            is_stop=is_stop_test_np[idx_show],
            dt=_task_dt,
            out_path=random_plot_path,
            n_show=n_show,
            y_pred=y_pred_sample_np[:n_show],
            metrics=metrics_random[:n_show],
        )
        logger.info(f'  saved: {random_plot_path}')
    else:
        logger.info('  (a) random test trials: skipped (translation-only mode)')

    # --- (a') Translation analog: random test trials on the d head -----
    # When the model has a translation head (and no rotation), there's
    # no cos/sin to score; we instead report per-trial d-RMSE and Pearson
    # between decoded d and true d on a sample of trials, and plot
    # v_fwd (top) + d true vs decoded (bottom). Matches the rotation
    # plot's colour scheme (green = GT, black = decoded) and font sizes.
    if has_translation and not has_rotation:
        n_metric = int(min(512, u_test.shape[0]))
        idx_sample_t = np.sort(rng.choice(u_test.shape[0],
                                            size=n_metric, replace=False))
        with torch.no_grad():
            y_pred_t, _ = model(u_test[idx_sample_t])
        y_pred_t_np = y_pred_t.cpu().numpy()
        # In translation-only the model output is the scalar d (col 0).
        d_pred = y_pred_t_np[..., 0]
        d_true = y_test_np[idx_sample_t][..., 0]
        rms = np.sqrt(np.mean((d_pred[:, 10:] - d_true[:, 10:]) ** 2, axis=1))
        prs = []
        for k in range(d_pred.shape[0]):
            a = d_pred[k, 10:]; b = d_true[k, 10:]
            if a.std() > 1e-8 and b.std() > 1e-8:
                prs.append(float(np.corrcoef(a, b)[0, 1]))
            else:
                prs.append(float('nan'))
        prs = np.asarray(prs)
        # OU (naturalistic) held-out translation trials: decoded vs true d.
        npz_bundle.update(
            d_random_idx=idx_sample_t.astype(np.int64),
            d_random_d_true=d_true.astype(np.float32),
            d_random_d_pred=d_pred.astype(np.float32),
            d_random_rmse=rms.astype(np.float32),
            d_random_pearson=prs.astype(np.float32),
        )
        logger.info(
            f'  {n_metric} random translation test trials: '
            f'rmse={np.nanmean(rms):.3f}±{np.nanstd(rms):.3f}  '
            f'r={np.nanmean(prs):.4f}±{np.nanstd(prs):.4f}'
        )
        n_show = int(min(5, n_metric))
        idx_show_t = idx_sample_t[:n_show]
        try:
            import matplotlib.pyplot as plt
            GT_COLOR = "#4daf4a"; PRED_COLOR = "black"
            INP_BG, OUT_BG = "0.92", "0.97"
            TICK_FS = 9; LABEL_FS = 11; TITLE_FS = 10
            fig, axes = plt.subplots(2, n_show,
                                       figsize=(2.6 * n_show, 4.0),
                                       sharex='col', sharey='row')
            if n_show == 1:
                axes = axes.reshape(2, 1)
            for col in range(n_show):
                idx = int(idx_show_t[col])
                T_loc = u_test_np.shape[1]
                t = np.arange(T_loc) * _task_dt
                u_one = u_test_np[idx, :, 0]  # v_fwd channel
                ax_top = axes[0, col]
                ax_bot = axes[1, col]
                ax_top.set_facecolor(INP_BG)
                ax_top.plot(t, u_one, color=GT_COLOR, lw=2.0)
                ax_top.set_title(
                    f"trial #{idx}\nrmse={rms[col]:.2f}  r={prs[col]:+.3f}",
                    fontsize=TITLE_FS,
                )
                if col == 0:
                    ax_top.set_ylabel(r"$v_{\rm fwd}$", fontsize=LABEL_FS)
                ax_top.tick_params(labelbottom=False, labelsize=TICK_FS)
                ax_bot.set_facecolor(OUT_BG)
                ax_bot.plot(t, d_true[col], color=GT_COLOR, lw=2.8, label='GT')
                ax_bot.plot(t, d_pred[col], color=PRED_COLOR, lw=0.5,
                            label='decoded')
                ax_bot.set_xlabel('time (s)', fontsize=LABEL_FS)
                if col == 0:
                    ax_bot.set_ylabel(r"$d$", fontsize=LABEL_FS)
                    ax_bot.legend(loc='best', fontsize=TICK_FS,
                                   frameon=False)
                ax_bot.tick_params(labelsize=TICK_FS)
            fig.tight_layout()
            trans_random_path = os.path.join(results_dir,
                                              'test_random_trials.png')
            fig.savefig(trans_random_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f'  saved: {trans_random_path}')
        except Exception as exc:
            logger.warning(f'  translation random-trials plot failed: {exc}')

    # --- (a'') 2D-position analog: random test trials in the (x, y) plane --
    # Companion of test_deterministic_2d_sweep.png — sample held-out
    # trials, run the model, plot the GT path (green) + decoded path
    # (black) in spatial coordinates, with per-trial 2-D RMSE and
    # mean per-axis Pearson in the title. Saves to
    # results/test_random_trials.png.
    if has_position_2d:
        from connectome_gnn.models.bump_attractor_eval import (
            _trajectory_metrics_2d as _tm_2d,
        )
        n_metric = int(min(512, u_test.shape[0]))
        idx_sample_p = np.sort(rng.choice(u_test.shape[0],
                                            size=n_metric, replace=False))
        with torch.no_grad():
            y_pred_p, _ = model(u_test[idx_sample_p])
        y_pred_p_np = y_pred_p.cpu().numpy()
        # (x, y) are the last two output columns.
        true_xy = y_test_np[idx_sample_p][..., -2:]
        dec_xy  = y_pred_p_np[..., -2:]
        rms = np.sqrt(np.mean((dec_xy[:, 10:] - true_xy[:, 10:]) ** 2,
                                axis=(1, 2)))
        prs = []
        for k in range(dec_xy.shape[0]):
            rs = []
            for axis in range(2):
                a = dec_xy[k, 10:, axis]; b = true_xy[k, 10:, axis]
                if a.std() > 1e-8 and b.std() > 1e-8:
                    rs.append(float(np.corrcoef(a, b)[0, 1]))
            prs.append(float(np.mean(rs)) if rs else float('nan'))
        prs = np.asarray(prs)
        # OU (naturalistic) held-out 2D-position trials: decoded vs true (x,y).
        npz_bundle.update(
            xy_random_idx=idx_sample_p.astype(np.int64),
            xy_random_true=true_xy.astype(np.float32),
            xy_random_pred=dec_xy.astype(np.float32),
            xy_random_rmse=rms.astype(np.float32),
            xy_random_pearson=prs.astype(np.float32),
        )
        logger.info(
            f'  {n_metric} random 2D test trials: '
            f'euclid_rmse={np.nanmean(rms):.3f}±{np.nanstd(rms):.3f}  '
            f'r̄={np.nanmean(prs):.4f}±{np.nanstd(prs):.4f}'
        )
        n_show = int(min(5, n_metric))
        idx_show_p = idx_sample_p[:n_show]
        try:
            import matplotlib.pyplot as plt
            GT_COLOR = "#4daf4a"; PRED_COLOR = "black"
            TICK_FS = 9; LABEL_FS = 11; TITLE_FS = 10
            fig, axes = plt.subplots(
                1, n_show, figsize=(2.8 * n_show, 3.0),
            )
            if n_show == 1:
                axes = np.asarray([axes])
            for col in range(n_show):
                idx = int(idx_show_p[col])
                ax = axes[col]
                t_xy = true_xy[col]; d_xy = dec_xy[col]
                ax.plot(t_xy[:, 0], t_xy[:, 1], color=GT_COLOR, lw=2.4,
                        label='GT')
                ax.plot(d_xy[:, 0], d_xy[:, 1], color=PRED_COLOR, lw=0.8,
                        label='decoded')
                ax.plot([0.0], [0.0], 'o', color='0.4', ms=4, zorder=5)
                ax.set_aspect('equal', adjustable='datalim')
                ax.set_title(
                    f"trial #{idx}\nrmse={rms[col]:.2f}  r̄={prs[col]:+.3f}",
                    fontsize=TITLE_FS,
                )
                ax.set_xlabel(r"$x$", fontsize=LABEL_FS)
                if col == 0:
                    ax.set_ylabel(r"$y$", fontsize=LABEL_FS)
                    ax.legend(loc='best', fontsize=TICK_FS, frameon=False)
                ax.tick_params(labelsize=TICK_FS)
            fig.tight_layout()
            pos2d_random_path = os.path.join(results_dir,
                                              'test_random_trials.png')
            fig.savefig(pos2d_random_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f'  saved: {pos2d_random_path}')
        except Exception as exc:
            logger.warning(f'  2D random-trials plot failed: {exc}')

    # --- (a.5) Anatomy-voltage snapshot ------------------------------------
    # The probe rollout (pattern / n_steps / stride / per-pattern params)
    # is driven entirely by plotting.anatomy_voltage_* in the yaml. The
    # rendering work (rollout + projection + frame writing) lives in
    # connectome_gnn.plot_anatomy_voltage so this dispatcher stays small.
    plot_cfg = config.plotting
    yaml_toggle = bool(getattr(plot_cfg, 'anatomy_voltage_enabled', False))
    circuit_cfg = getattr(config, 'circuit', None)
    if (anatomy_voltage or yaml_toggle) and circuit_cfg is not None \
            and getattr(circuit_cfg, 'name', None):
        from connectome_gnn.generators.circuits import get_circuit
        from connectome_gnn.plot_anatomy_voltage import run_anatomy_voltage_test
        try:
            c = get_circuit(circuit_cfg.name)
            out = run_anatomy_voltage_test(
                model, c, plot_cfg, log_dir, device=device,
                type_groups=anatomy_voltage_type_groups,
            )
            if out is not None:
                logger.info(f'  [anatomy_voltage] wrote: {out}')
        except Exception as _e:
            logger.warning(
                f'  [anatomy_voltage] failed: {type(_e).__name__}: {_e}'
            )
    elif anatomy_voltage and (circuit_cfg is None
                              or not getattr(circuit_cfg, 'name', None)):
        logger.warning(
            '  [anatomy_voltage] --anatomy_voltage set but config has no '
            'circuit.name; skipping. Add `circuit: {name: <registered>}` '
            'to the yaml.'
        )

    T_sweep = 2000
    # --- (b) 5 deterministic sweeps at ω ∈ {-120,-60,30,60,120}, T=2000 -----
    # Heading-only — gated on has_rotation. In translation-only mode the
    # rollout would build a 1-ch v_fwd input and the heading-frame plotting
    # would be meaningless. See section (b') below for the translation analog.
    if has_rotation:
        omega_set = [-120.0, -60.0, 30.0, 60.0, 120.0]
        u_sweep, y_sweep, theta_sweep, y_pred_sweep = [], [], [], []
        for omega in omega_set:
            rollout = _deterministic_sweep_rollout(
                model, n_steps=T_sweep, omega_deg_per_s=omega, device=device,
            )
            u_sweep.append(rollout['u'])
            theta_t = rollout['true_theta']
            theta_sweep.append(theta_t)
            # Ground-truth (cos, sin) target from theta_t.
            y_sweep.append(np.stack(
                [np.cos(theta_t), np.sin(theta_t)], axis=-1
            ).astype(np.float32))
            y_pred_sweep.append(rollout['y_pred'])
        u_sweep_arr = np.stack(u_sweep, axis=0)
        y_sweep_arr = np.stack(y_sweep, axis=0)
        theta_sweep_arr = np.stack(theta_sweep, axis=0)
        y_pred_sweep_arr = np.stack(y_pred_sweep, axis=0)

        metrics_sweep = _per_trial_heading_metrics(
            y_pred_sweep_arr, theta_sweep_arr,
        )
        # Inject ω into metrics so the plot title shows it.
        for m, omega in zip(metrics_sweep, omega_set):
            m['omega_deg'] = float(omega)
        # Constant-ω heading sweeps: raw decoded (cos,sin) + true heading.
        npz_bundle.update(
            heading_sweep_omega_deg=np.asarray(omega_set, np.float32),
            heading_sweep_theta_true=theta_sweep_arr.astype(np.float32),
            heading_sweep_pred_cossin=y_pred_sweep_arr.astype(np.float32),
            heading_sweep_rmse_deg=np.asarray(
                [m['rmse_deg'] for m in metrics_sweep], np.float32),
            heading_sweep_pearson=np.asarray(
                [m['pearson'] for m in metrics_sweep], np.float32),
        )
        logger.info(
            '  5 deterministic ω sweeps (T=2000): '
            + '  '.join(
                f"ω={o:+.0f}: r={_color_r(m['pearson'])}"
                for o, m in zip(omega_set, metrics_sweep)
            )
        )
        sweep_plot_path = os.path.join(results_dir, 'test_deterministic_sweep.png')
        plot_task_pi_traces(
            u=u_sweep_arr,
            y=y_sweep_arr,
            theta_hd=theta_sweep_arr,
            is_stop=None,
            dt=_task_dt,
            out_path=sweep_plot_path,
            n_show=5,
            y_pred=y_pred_sweep_arr,
            metrics=metrics_sweep,
        )
        logger.info(f'  saved: {sweep_plot_path}')

        # --- (c) Integration-gain analysis (Hulse-style slope test) --------
        # Denser ω scan than the 5-panel deterministic_sweep so the gain
        # curve has enough points to resolve where integration breaks down.
        gain_omega_set = [-180.0, -150.0, -120.0, -90.0, -60.0, -30.0,
                           30.0,  60.0,  90.0, 120.0, 150.0, 180.0]
        gain_theta, gain_y_pred = [], []
        for omega in gain_omega_set:
            ro = _deterministic_sweep_rollout(
                model, n_steps=T_sweep, omega_deg_per_s=omega, device=device,
            )
            gain_theta.append(ro['true_theta'])
            gain_y_pred.append(ro['y_pred'])
        gain_plot_path = os.path.join(results_dir, 'test_integration_gain.png')
        gain_metrics = plot_integration_gain(
            theta_hd=np.stack(gain_theta, axis=0),
            y_pred=np.stack(gain_y_pred, axis=0),
            omega_deg_per_s=gain_omega_set,
            dt=_task_dt,
            out_path=gain_plot_path,
        )
        # Constant-ω integration-gain curve + the raw rollouts behind it.
        npz_bundle.update(
            gain_omega_deg=np.asarray(
                [m['omega_deg'] for m in gain_metrics], np.float32),
            gain_slope_deg_per_s=np.asarray(
                [m.get('slope_deg_per_s', np.nan) for m in gain_metrics],
                np.float32),
            gain=np.asarray([m['gain'] for m in gain_metrics], np.float32),
            gain_fit_r2=np.asarray(
                [m.get('fit_r2', np.nan) for m in gain_metrics], np.float32),
            gain_theta_true=np.stack(gain_theta, axis=0).astype(np.float32),
            gain_pred_cossin=np.stack(gain_y_pred, axis=0).astype(np.float32),
        )
        logger.info(
            f'  {len(gain_omega_set)} integration gains (slope ÷ ω): '
            + '  '.join(
                f"ω={m['omega_deg']:+.0f}: g={m['gain']:+.3f}"
                for m in gain_metrics
            )
        )
        logger.info(f'  saved: {gain_plot_path}')
    else:
        logger.info('  (b)/(c) deterministic ω sweeps + gain: skipped '
                    '(translation-only mode)')

    # --- (c') Precision horizons on a long naturalistic (OU) drive ----------
    # Rollout Pearson saturates over a short trial, so we drive the model with
    # an Ornstein--Uhlenbeck ω(t)/v_fwd(t) for 60 s and report the time the
    # decoded heading stays within 15° (tau_theta), the heading integration-gain
    # error, and — for the LEAKY (bounded) integrators only — the displacement
    # precision horizon. Guarded so it can never break the rest of the test.
    try:
        from connectome_gnn.models.bump_attractor_eval import (
            precision_horizon_metrics,
        )
        _leaky_run = bool(getattr(tc, 'xi_decay_tau_s', None)) or (
            'leaky' in str(getattr(config, 'dataset', '')).lower())
        ph = precision_horizon_metrics(
            model, device=device, leaky=_leaky_run)
        _td = (f"{ph['tau_d_s']:.1f}s" if ph['tau_d_s'] is not None
               else '-- (heading-only)')
        logger.info(
            f"  precision horizon (60 s naturalistic drive, "
            f"{ph['n_seed']} seeds): tau_theta(15deg)={ph['tau_theta_s']:.1f}s  "
            f"|g_theta-1|={ph['heading_gain_err']:.3f}  tau_d={_td}")
    except Exception as exc:                       # never break the test
        logger.warning(f'  precision-horizon metrics failed: {exc}')

    # --- (b') Translation analog: 5 deterministic v_fwd sweeps, T=2000 ------
    # Constant-v_fwd rollouts. Reports per-rollout RMSE on ξ and Pearson r
    # between decoded ξ and GT ξ = v_fwd × t. Saves a simple 5-row figure
    # (v_fwd top, ξ true vs decoded bottom). Only runs when the model
    # carries a translation output.
    if has_translation:
        v_fwd_set = [-2.0, -1.0, 0.5, 1.0, 2.0]
        rollouts_trans = []
        for v in v_fwd_set:
            ro = _deterministic_sweep_rollout(
                model, n_steps=T_sweep, v_fwd_per_s=v, device=device,
            )
            rollouts_trans.append((v, ro))
        # Constant-v_fwd displacement sweeps: raw decoded vs true ξ (= d).
        npz_bundle.update(
            vfwd_sweep_v=np.asarray([v for v, _ in rollouts_trans], np.float32),
            vfwd_sweep_xi_true=np.stack(
                [np.asarray(ro['true_xi']) for _, ro in rollouts_trans],
                axis=0).astype(np.float32),
            vfwd_sweep_xi_pred=np.stack(
                [np.asarray(ro['decoded_xi']) for _, ro in rollouts_trans],
                axis=0).astype(np.float32),
        )
        # Per-rollout summary.
        log_bits = []
        for v, ro in rollouts_trans:
            xi_true = np.asarray(ro['true_xi'])
            xi_pred = np.asarray(ro['decoded_xi'])
            rmse = float(np.sqrt(np.mean((xi_pred - xi_true) ** 2)))
            if (xi_pred[10:].std() > 1e-8 and xi_true[10:].std() > 1e-8):
                r = float(np.corrcoef(xi_pred[10:], xi_true[10:])[0, 1])
            else:
                r = float('nan')
            log_bits.append(f"v={v:+.1f}: r={_color_r(r)} rmse={rmse:.2f}")
        logger.info('  5 deterministic v_fwd sweeps (T=2000): '
                    + '  '.join(log_bits))
        # Save the figure: 2 rows × n_sweeps cols. Companion of the
        # rotation test_deterministic_sweep.png — same colour scheme
        # (green = GT, black = decoded), same line widths, same font
        # sizes (tick 9, label 11, title 10).
        try:
            import matplotlib.pyplot as plt
            GT_COLOR = "#4daf4a"
            PRED_COLOR = "black"
            INP_BG, OUT_BG = "0.92", "0.97"
            TICK_FS = 9
            LABEL_FS = 11
            TITLE_FS = 10
            n_cols = len(rollouts_trans)
            fig, axes = plt.subplots(
                2, n_cols,
                figsize=(2.6 * n_cols, 1.5 * 2 + 1.0),
                sharex='col', sharey='row',
            )
            if n_cols == 1:
                axes = axes.reshape(2, 1)
            for col, (v, ro) in enumerate(rollouts_trans):
                T_loc = ro['n_steps']
                t = np.arange(T_loc) * _task_dt
                ax_top = axes[0, col]
                ax_bot = axes[1, col]
                u_col = ro['u'][:, 0]  # translation-only mode: v_fwd in col 0
                if u_col.std() < 1e-8 and ro['u'].shape[-1] >= 2:
                    u_col = ro['u'][:, 1]   # "both" mode: v_fwd in col 1
                ax_top.set_facecolor(INP_BG)
                ax_top.plot(t, u_col, color=GT_COLOR, lw=2.0)
                # Per-column title: v_fwd + r + rmse on the rollout.
                xi_true = np.asarray(ro['true_xi'])
                xi_pred = np.asarray(ro['decoded_xi'])
                rmse = float(np.sqrt(np.mean((xi_pred[10:] - xi_true[10:]) ** 2)))
                if xi_pred[10:].std() > 1e-8 and xi_true[10:].std() > 1e-8:
                    r = float(np.corrcoef(xi_pred[10:], xi_true[10:])[0, 1])
                else:
                    r = float('nan')
                ax_top.set_title(
                    fr"$v_{{\rm fwd}}={v:+.1f}$"
                    f"\nrmse={rmse:.2f}  r={r:+.3f}",
                    fontsize=TITLE_FS,
                )
                if col == 0:
                    ax_top.set_ylabel(r"$v_{\rm fwd}$", fontsize=LABEL_FS)
                ax_top.tick_params(labelbottom=False, labelsize=TICK_FS)

                ax_bot.set_facecolor(OUT_BG)
                ax_bot.plot(t, xi_true, color=GT_COLOR, lw=2.8, label='GT')
                ax_bot.plot(t, xi_pred, color=PRED_COLOR, lw=0.5,
                            label='decoded')
                ax_bot.set_xlabel('time (s)', fontsize=LABEL_FS)
                if col == 0:
                    ax_bot.set_ylabel(r"$d$", fontsize=LABEL_FS)
                    ax_bot.legend(loc='best', fontsize=TICK_FS,
                                   frameon=False)
                ax_bot.tick_params(labelsize=TICK_FS)
            fig.tight_layout()
            trans_sweep_path = os.path.join(
                results_dir, 'test_deterministic_v_fwd_sweep.png')
            fig.savefig(trans_sweep_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f'  saved: {trans_sweep_path}')
        except Exception as exc:
            logger.warning(f'  v_fwd-sweep plot failed: {exc}')

    # ---- write an aggregated trajectory_metrics.txt for the run -------
    # Every numeric scalar produced by the sweep blocks above (and below)
    # is collected into a single per-run text file so cross-model
    # comparison tables can be assembled without re-running the sweeps.
    trajectory_metrics_path = os.path.join(results_dir,
                                            "trajectory_metrics.txt")
    _traj_rows: list[str] = []

    def _emit(prefix: str, mdict: dict):
        if not mdict:
            return
        for k, v in mdict.items():
            try:
                _traj_rows.append(f"{prefix}.{k}={float(v):.6g}")
            except (TypeError, ValueError):
                _traj_rows.append(f"{prefix}.{k}={v}")

    # Heading sweep extended metrics (rotation-bearing models).
    if has_rotation:
        from connectome_gnn.models.bump_attractor_eval import (
            _trajectory_metrics_1d as _tm_1d,
        )
        for om in [-120.0, -60.0, 60.0, 120.0]:
            try:
                ro = _deterministic_sweep_rollout(
                    model, n_steps=T_sweep, omega_deg_per_s=om, device=device,
                )
                if "true_theta" in ro and "decoded_theta" in ro:
                    th_true = np.unwrap(np.asarray(ro["true_theta"]))
                    th_dec  = np.unwrap(np.asarray(ro["decoded_theta"]))
                    _emit(f"heading_w{int(om):+d}",
                          _tm_1d(np.degrees(th_true), np.degrees(th_dec)))
            except Exception:
                pass

    # Translation sweep extended metrics.
    if has_translation:
        from connectome_gnn.models.bump_attractor_eval import (
            _trajectory_metrics_1d as _tm_1d,
        )
        for vf in [-2.0, -1.0, 0.5, 1.0, 2.0]:
            try:
                ro = _deterministic_sweep_rollout(
                    model, n_steps=T_sweep, v_fwd_per_s=vf, device=device,
                )
                if "true_xi" in ro and "decoded_xi" in ro:
                    _emit(f"distance_v{vf:+.1f}",
                          _tm_1d(np.asarray(ro["true_xi"]),
                                  np.asarray(ro["decoded_xi"])))
            except Exception:
                pass

    # --- (b'') 2D PI analog: 5 deterministic (ω, v_fwd) sweeps --------------
    # For position_2d models (n_out=4), probe with constant ω AND constant
    # v_fwd: GT trajectory is a circle of radius v_fwd / |ω_rad| centred
    # perpendicular to the initial heading. Reports per-rollout 2D RMSE and
    # axis-averaged Pearson between decoded (x̂, ŷ) and GT (x, y). Saves a
    # 5-panel figure with each panel showing GT path (green) + decoded path
    # (black) in the (x, y) plane.
    if has_position_2d:
        from connectome_gnn.models.bump_attractor_eval import (
            _rollout_position_metrics,
            _trajectory_metrics_2d as _tm_2d,
        )
        omega_v_set = [(-120.0, 1.0), (-60.0, 1.0), (30.0, 1.0),
                       (60.0, 1.0),   (120.0, 1.0)]
        rollouts_2d = []
        for om, vf in omega_v_set:
            ro = _deterministic_sweep_rollout(
                model, n_steps=T_sweep, omega_deg_per_s=om,
                v_fwd_per_s=vf, device=device,
            )
            rollouts_2d.append((om, vf, ro))
        # Constant-(ω, v_fwd) 2D-path sweeps: raw decoded vs true (x, y).
        npz_bundle.update(
            xy_sweep_omega_deg=np.asarray(
                [om for om, _, _ in rollouts_2d], np.float32),
            xy_sweep_v_fwd=np.asarray(
                [vf for _, vf, _ in rollouts_2d], np.float32),
            xy_sweep_true=np.stack(
                [np.asarray(ro['true_xy']) for _, _, ro in rollouts_2d],
                axis=0).astype(np.float32),
            xy_sweep_pred=np.stack(
                [np.asarray(ro['decoded_xy']) for _, _, ro in rollouts_2d],
                axis=0).astype(np.float32),
        )
        log_bits = []
        for om, vf, ro in rollouts_2d:
            rmse, r = _rollout_position_metrics(
                model, n_steps=T_sweep, omega_deg_per_s=om,
                v_fwd_per_s=vf, device=device,
            )
            log_bits.append(f"ω={om:+.0f},v={vf:+.1f}: r̄={_color_r(r)} "
                            f"rmse={rmse:.2f}")
            # Extended trajectory metrics on the same rollout.
            try:
                if "true_xy" in ro and "decoded_xy" in ro:
                    _emit(f"path2d_w{int(om):+d}_v{vf:+.1f}",
                          _tm_2d(np.asarray(ro["true_xy"]),
                                  np.asarray(ro["decoded_xy"])))
            except Exception:
                pass
        logger.info('  5 deterministic 2D-PI sweeps (T=' + str(T_sweep) + '): '
                    + '  '.join(log_bits))
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(
                1, len(rollouts_2d),
                figsize=(2.8 * len(rollouts_2d), 3.0),
            )
            for col, (om, vf, ro) in enumerate(rollouts_2d):
                ax = axes[col]
                true_xy = np.asarray(ro['true_xy'])
                dec_xy = np.asarray(ro['decoded_xy'])
                ax.plot(true_xy[:, 0], true_xy[:, 1], color='green', lw=1.0,
                        label='GT')
                ax.plot(dec_xy[:, 0],  dec_xy[:, 1],  color='k', lw=0.8,
                        label='decoded')
                ax.plot([0.0], [0.0], 'o', color='0.4', ms=4, zorder=5)
                ax.set_aspect('equal', adjustable='datalim')
                ax.set_title(f"ω={om:+.0f}°/s  v={vf:+.1f}", fontsize=9)
                ax.set_xlabel('x', fontsize=8)
                if col == 0:
                    ax.set_ylabel('y', fontsize=8)
                    ax.legend(loc='best', fontsize=7, frameon=False)
                ax.tick_params(labelsize=7)
            fig.tight_layout()
            pos_sweep_path = os.path.join(
                results_dir, 'test_deterministic_2d_sweep.png')
            fig.savefig(pos_sweep_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f'  saved: {pos_sweep_path}')
        except Exception as exc:
            logger.warning(f'  2D-PI sweep plot failed: {exc}')

    # --- (d) Function dynamics along ω=60°/s rollout (GNN teachers only) ---
    # Hexbin of (h_i(t), f_theta(h_i(t))) and (h_j(t), g_phi(h_j(t))^2)
    # over the ω = +60°/s rollout, with the static curves of fig 4 (k)/(l)
    # overlaid. Skipped for non-GNN teachers (TaskRNN has no f_theta /
    # g_phi). Also skipped in translation-only mode — no ω drive to probe.
    if has_rotation and all(hasattr(model, name)
                            for name in ("a", "g_phi", "f_theta")):
        # Re-run the +60°/s sweep with T_sweep frames; cheap (~ms on l4)
        # and lets us cleanly extract the per-neuron h trajectory.
        ro_60 = _deterministic_sweep_rollout(
            model, n_steps=T_sweep, omega_deg_per_s=60.0, device=device,
        )
        h_traj = np.asarray(ro_60['h'])                      # (T, N)
        fdyn_plot_path = os.path.join(
            results_dir, 'test_function_dynamics.png')
        try:
            plot_function_dynamics(
                net=model, h_traj=h_traj, out_path=fdyn_plot_path,
                device=device,
            )
            logger.info(f'  saved: {fdyn_plot_path}')
        except Exception as exc:
            logger.warning(f'  function-dynamics plot failed: {exc}')

    # --- (e) calcium 600 s reconstruction (dataset B / gcamp obs runs) ------
    # When a calcium dataset B is wired to this run, REASSEMBLE the recording:
    # its first `n_block_tiles` train trials are consecutive 10 s windows paving
    # the whole ~600 s block. Roll the model on each window (re-anchored from its
    # own cue — the regime the model is trained/evaluated in), stitch the per-
    # trial voltage into one 600 s series, convert to calcium with the GCaMP
    # class ONCE over the full series (no 60 kernel-edge artifacts), and compare
    # to the recorded ΔF/F over the bump-pool rastermap rows. A continuous single
    # rollout (zapbench_rotation) is overlaid as the integration-drift view. This
    # is the full-block successor to the single-10 s training "panel h". Gated on
    # the dataset existing; silently skipped otherwise (e.g. drosophila_cx).
    calcium_metrics = None
    try:
        _calc_name = getattr(tc, 'calcium_dataset', '') or config.dataset
        if '/' not in _calc_name and '/' in config.dataset:
            _calc_name = config.dataset.rsplit('/', 1)[0] + '/' + _calc_name
        _calc_root = graphs_data_path(_calc_name)
        _has_ca = os.path.isdir(os.path.join(_calc_root, 'train', 'calcium.zarr'))
    except Exception:
        _has_ca = False
    if _has_ca:
        import zarr as _zarr
        from connectome_gnn.models.gcamp import create_gcamp
        from connectome_gnn.plot_anatomy_voltage import run_task_rollout
        from connectome_gnn.plot_cx import plot_calcium_reconstruction
        from connectome_gnn.task_state import TaskTrials as _TT
        _repo = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..'))
        _meta = np.load(os.path.join(_calc_root, 'calcium_meta.npz'),
                        allow_pickle=True)
        n_tile = int(_meta['n_block_tiles']) if 'n_block_tiles' in _meta.files else 0
        _map = torch.load(os.path.join(_calc_root, 'calcium_mapping.pt'),
                          weights_only=False)
        _kmask = _map['kino_obs_index'] >= 0
        kino_model_ix = _map['kino_model_index'][_kmask].to(device).long()
        kino_obs_ix = _map['kino_obs_index'][_kmask].cpu().numpy()
        # Afferent neurons = observed columns NOT in the bump-pool kinograph
        # (RIPN / pt-IPN inputs). The observation loss supervises these too, so
        # plot them as their own real/learned group. Ordered by cell type for a
        # readable kinograph. model_index maps every observed column (481/481).
        _model_index_all = _map['model_index'].cpu().numpy()
        _types_all = _meta['type'].astype(str)
        _bump_cols = set(int(c) for c in kino_obs_ix)
        aff_obs_ix = np.asarray(
            sorted((c for c in range(len(_types_all))
                    if c not in _bump_cols and _model_index_all[c] >= 0),
                   key=lambda c: (str(_types_all[c]), c)), dtype=np.int64)
        aff_model_ix = torch.as_tensor(
            _model_index_all[aff_obs_ix], device=device).long()
        gcamp_name = getattr(config.simulation, 'gcamp_kernel', 'gcamp7f')
        gcamp = create_gcamp(gcamp_name)
        dt = float(model.dt)
        if n_tile <= 0:
            logger.warning('  [calcium] n_block_tiles=0 in meta; regenerate '
                           'dataset B for the block tiles — skipping reconstruction.')
        else:
            _ct = _TT.load(f"{_calc_root}/train")
            ca_u = _ct.stimulus[:n_tile].to(device)            # (n_tile, T, 3)
            ca_real = np.asarray(
                _zarr.open(f"{_calc_root}/train/calcium.zarr", 'r'))[:n_tile]
            n_steps_trial = ca_u.shape[1]
            # ω is column 0 of the 3-col rotation-shape stimulus; stash
            # it now because the task_targets projection below may slice
            # ca_u down to a single non-ω channel for translation-only
            # variants.
            omega_blk = ca_u[:, :, 0].reshape(-1).cpu().numpy()

            # Match the trainer's calcium-dataset reshape (see
            # graph_trainer.py: build 4-col superset from
            # swim_forward.zarr, then apply the task_targets in_cols
            # projection) so the model sees its training-time n_in
            # at test time too. Without this the calcium dataset's
            # rotation-shaped 3-col stimulus is fed straight into a
            # 1-col translation-only model and crashes _project_in.
            def _maybe_load_zarr(path):
                try:
                    return torch.from_numpy(
                        np.asarray(_zarr.open(path, mode="r"))
                    ).to(device).float()
                except Exception:
                    return None

            _propriocep_split = bool(getattr(
                getattr(config.task, "swim_integration", None),
                "propriocep_split", False))
            _needs_vfwd = bool(_task_key) and (
                "translation" in _task_key or "position_2d" in _task_key)
            _has_super = False
            if _needs_vfwd and ca_u.shape[-1] == 3:
                swim_fwd = _maybe_load_zarr(
                    f"{_calc_root}/train/swim_forward.zarr")
                if swim_fwd is None:
                    raise RuntimeError(
                        f"calcium dataset {_calc_name} is rotation-shaped "
                        f"(3 stim channels) and swim_forward.zarr is "
                        f"missing; cannot drive task_targets "
                        f"{task_targets_canonical} at test time.")
                if swim_fwd.dim() == 3:
                    swim_fwd = swim_fwd[..., 0]
                swim_fwd = swim_fwd[:n_tile, :n_steps_trial]
                n_super = 5 if _propriocep_split else 4
                ca_u_super = torch.zeros(
                    (ca_u.shape[0], ca_u.shape[1], n_super),
                    dtype=ca_u.dtype, device=device)
                if _propriocep_split:
                    # [ω, v_fwd, ω_proprio, cos θ₀·δ, sin θ₀·δ] — channel 2
                    # is the angular efference copy ω_proprio (= ω), routed
                    # to motor_efferent (head-direction cells, not forward).
                    ca_u_super[..., 0] = ca_u[..., 0]
                    ca_u_super[..., 1] = swim_fwd          # v_fwd → pt-IPN1
                    ca_u_super[..., 2] = ca_u[..., 0]      # ω_proprio → motor_efferent
                    ca_u_super[..., 3] = ca_u[..., 1]
                    ca_u_super[..., 4] = ca_u[..., 2]
                else:
                    # [ω, v_fwd, cos θ₀·δ, sin θ₀·δ]
                    ca_u_super[..., 0] = ca_u[..., 0]
                    ca_u_super[..., 1] = swim_fwd
                    ca_u_super[..., 2] = ca_u[..., 1]
                    ca_u_super[..., 3] = ca_u[..., 2]
                ca_u = ca_u_super
                _has_super = True

            # Synthetic target columns: 2 for rotation-only datasets, 3
            # once the v_fwd column was added (target gains a d head).
            _y_cols_for_profile = 3 if _has_super else 2
            _prof_key = (ca_u.shape[-1], _y_cols_for_profile, _task_key)
            if _task_key and _prof_key in _PROFILE_BY_TARGET:
                _ic, _oc = _PROFILE_BY_TARGET[_prof_key]
                ca_u = ca_u[..., _ic].contiguous()
                logger.info(
                    f'  [calcium] ca_u sliced to in_cols={_ic} '
                    f'(task_targets={task_targets_canonical})')

            # --- stitched per-trial learned calcium + HD decode -----------
            with torch.no_grad():
                y_stitch, h_stitch = model(ca_u)               # (n_tile,T,?),(.,N)
                h_full = h_stitch.reshape(n_tile * n_steps_trial, -1)  # (T_blk, N)
                ca_full = gcamp(h_full, dt_in=dt)               # (T_blk, N)
            learned_stitch = ca_full.index_select(
                -1, kino_model_ix).cpu().numpy()                # (T_blk, K)
            aff_stitch = ca_full.index_select(
                -1, aff_model_ix).cpu().numpy()                 # (T_blk, K_aff)
            # HD decode requires a 2-col (cos, sin) readout; skip on
            # translation-only models whose readout is the 1-col d.
            hd_info = {}
            if has_rotation and y_stitch.shape[-1] >= 2:
                dec_stitch = np.rad2deg(torch.atan2(
                    y_stitch[..., 1], y_stitch[..., 0]).reshape(-1).cpu().numpy())
                if _ct.theta_hd is not None:
                    true_blk = np.rad2deg(
                        _ct.theta_hd[:n_tile].reshape(-1).cpu().numpy())
                else:
                    true_blk = np.rad2deg(
                        np.cumsum(np.deg2rad(omega_blk)) * dt)
                hd_info = {'stitch': {'true': true_blk, 'pred': dec_stitch}}
            # --- real block: concat the tiled windows ---------------------
            real_blk = ca_real.reshape(n_tile * n_steps_trial, -1)  # (T_blk, R)
            real_kino = real_blk[:, kino_obs_ix]                # (T_blk, K)
            real_aff = real_blk[:, aff_obs_ix]                  # (T_blk, K_aff)
            # --- continuous single rollout (drift view), best-effort ------
            # Builds an n_in-aware continuous stimulus over the n_tile-trial
            # block (NOT the rolled-out rotation-only zapbench probe), so a
            # 4-input "both" model gets the recorded swim_forward in
            # column 1 instead of zeros. Channel 0 is the recorded ω
            # (already in ca_u_super[..., 0]), channel 1 (when n_in == 4)
            # is swim_forward concatenated trial-by-trial, channels 2/3
            # carry (cos θ₀, sin θ₀) on the first frame only --- exactly
            # the contract the model was trained on. The 1-col
            # translation variant gets v_fwd in column 0.
            learned_cont = None
            aff_cont = None
            v_fwd_blk = None
            d_info = {}
            traj_info = {}
            try:
                n_in = int(getattr(model, 'n_input', None)
                            or getattr(model, 'n_inputs', None)
                            or ca_u.shape[-1])
                # Recorded ω across the block (rotation-shape stim col 0)
                # before any task_targets slicing. We already computed
                # omega_blk above from the raw 3-col stimulus.
                T_blk = n_tile * n_steps_trial
                # swim_forward concatenated across the same n_tile trials
                # — needed whenever the model expects v_fwd
                # (n_in ∈ {1, 2, 4, 5}; the propriocep-split layouts
                # carry it on two parallel columns).
                if n_in in (1, 2, 4, 5):
                    sw = _maybe_load_zarr(
                        f"{_calc_root}/train/swim_forward.zarr")
                    if sw is None:
                        raise RuntimeError(
                            "swim_forward.zarr missing — cannot build "
                            f"continuous rollout for n_in={n_in} model.")
                    if sw.dim() == 3:
                        sw = sw[..., 0]
                    sw = sw[:n_tile, :n_steps_trial].reshape(-1).cpu().numpy()
                    v_fwd_blk = sw.astype(np.float32)
                # Build the n_in-shaped continuous stimulus
                u_cont = torch.zeros((1, T_blk, n_in), device=device,
                                      dtype=torch.float32)
                if n_in == 3:
                    # [ω, cosθ₀, sinθ₀]
                    u_cont[0, :, 0] = torch.as_tensor(
                        omega_blk, device=device, dtype=torch.float32)
                    u_cont[0, 0, 1] = 1.0     # cos θ₀ = 1, sin θ₀ = 0
                elif n_in == 1:
                    # [v_fwd] (translation-only, non-propriocep)
                    u_cont[0, :, 0] = torch.as_tensor(
                        v_fwd_blk, device=device, dtype=torch.float32)
                elif n_in == 2:
                    # [v_extero, v_proprio] (translation-only, propriocep-split)
                    vf = torch.as_tensor(
                        v_fwd_blk, device=device, dtype=torch.float32)
                    u_cont[0, :, 0] = vf       # v_extero
                    u_cont[0, :, 1] = vf       # v_proprio
                elif n_in == 4:
                    # [ω, v_fwd, cosθ₀, sinθ₀]
                    u_cont[0, :, 0] = torch.as_tensor(
                        omega_blk, device=device, dtype=torch.float32)
                    u_cont[0, :, 1] = torch.as_tensor(
                        v_fwd_blk, device=device, dtype=torch.float32)
                    u_cont[0, 0, 2] = 1.0     # cos θ₀ = 1, sin θ₀ = 0
                elif n_in == 5:
                    # [ω, v_fwd, ω_proprio, cosθ₀, sinθ₀] — channel 2 is the
                    # angular efference copy ω_proprio (= ω) → motor_efferent.
                    vf = torch.as_tensor(
                        v_fwd_blk, device=device, dtype=torch.float32)
                    om = torch.as_tensor(
                        omega_blk, device=device, dtype=torch.float32)
                    u_cont[0, :, 0] = om
                    u_cont[0, :, 1] = vf       # v_fwd → pt-IPN1
                    u_cont[0, :, 2] = om       # ω_proprio → motor_efferent
                    u_cont[0, 0, 3] = 1.0     # cos θ₀ = 1, sin θ₀ = 0
                else:
                    raise RuntimeError(f"unsupported n_in={n_in}")
                with torch.no_grad():
                    y_c, h_c = model(u_cont)
                h_c = h_c[0].cpu().numpy()
                y_c = y_c[0].cpu().numpy()
                with torch.no_grad():
                    ca_c = gcamp(torch.from_numpy(h_c).to(device), dt_in=dt)
                learned_cont = ca_c.index_select(
                    -1, kino_model_ix).cpu().numpy()
                aff_cont = ca_c.index_select(
                    -1, aff_model_ix).cpu().numpy()
                # Heading decode (uses readout cols 0/1 = cosθ, sinθ).
                if has_rotation and y_c.shape[-1] >= 2:
                    th_true = np.cumsum(np.deg2rad(omega_blk)) * dt
                    hd_info['continuous'] = {
                        'true': np.rad2deg(th_true),
                        'pred': np.rad2deg(
                            np.arctan2(y_c[:, 1], y_c[:, 0]))}
                # Translation decode (d head). The cumulative GT is
                # cumsum(v_fwd)·dt for the cumulative variant; the leaky
                # variant decays to v_fwd·τ in steady state but the GT
                # the model was trained on is recorded in
                # ``ca_y[..., 2]``. We display the cumulative reference
                # so the reader can see what unleaked PI would look like.
                if has_translation and v_fwd_blk is not None:
                    d_true_cum = np.cumsum(v_fwd_blk) * dt
                    d_col = 2 if y_c.shape[-1] >= 3 else 0
                    d_pred = y_c[:, d_col]
                    d_info['continuous'] = {
                        'true': d_true_cum, 'pred': d_pred}
                # 2-D path-integration trajectory (position_2d models,
                # readout cols 2/3 = x, y). True path is the cumulative
                # PI reference x=∫v_fwd cosθ, y=∫v_fwd sinθ (θ from the
                # recorded ω), matching the cumulative-d reference above;
                # for the leaky variant the decoded path stays bounded
                # while the reference grows.
                if has_position_2d and y_c.shape[-1] >= 4 \
                        and v_fwd_blk is not None:
                    th_blk = np.cumsum(np.deg2rad(omega_blk)) * dt
                    x_true = np.cumsum(v_fwd_blk * np.cos(th_blk)) * dt
                    y_true = np.cumsum(v_fwd_blk * np.sin(th_blk)) * dt
                    traj_info['continuous'] = {
                        'true_xy': np.stack([x_true, y_true], axis=1),
                        'pred_xy': y_c[:, 2:4]}
            except Exception as _e:
                logger.warning(f'  [calcium] continuous rollout skipped: '
                               f'{type(_e).__name__}: {_e}')
            recon_path = os.path.join(results_dir,
                                      'test_calcium_reconstruction.png')
            # Split the single afferent kinograph into the three velocity-gate
            # afferent classes (ARTR / pt-IPN1 / motor_efferent), each sorted
            # by per-cell preferred-heading angle φ_i = arg(Σ ΔF/F_i·e^{iθ})
            # computed from the recorded ΔF/F (same sort as Fig. 15), so the
            # real and model rows share the per-class ordering.
            from connectome_gnn.plot_cx import (
                _preferred_phase, _HD_ARTR_TYPES, _HD_MOTOR_EFFERENT_TYPES,
            )
            _AFF_CLASSES = [
                ("ARTR",           _HD_ARTR_TYPES),
                ("pt-IPN1",        {"pt-IPN1"}),
                ("motor_efferent", _HD_MOTOR_EFFERENT_TYPES),
            ]
            _aff_types = np.asarray(_types_all)[aff_obs_ix]      # type per col
            _theta_rad = np.cumsum(np.deg2rad(omega_blk)) * dt   # recorded HD
            ca_groups = [
                {'name': 'bump-pool', 'real': real_kino,
                 'stitch': learned_stitch, 'continuous': learned_cont},
            ]
            for _cname, _ctypes in _AFF_CLASSES:
                _m = np.array([str(t) in _ctypes for t in _aff_types])
                if not _m.any():
                    continue
                _r = real_aff[:, _m]
                _st = aff_stitch[:, _m] if aff_stitch is not None else None
                _co = aff_cont[:, _m] if aff_cont is not None else None
                # preferred-heading sort from the recorded ΔF/F of this class
                _phi = _preferred_phase(_r, _theta_rad[:_r.shape[0]])
                _order = np.argsort(_phi, kind="stable")
                ca_groups.append({
                    'name': _cname,
                    'real': _r[:, _order],
                    'stitch': _st[:, _order] if _st is not None else None,
                    'continuous': _co[:, _order] if _co is not None else None,
                })
            calcium_metrics = plot_calcium_reconstruction(
                ca_groups, dt, recon_path,
                title=(f'{config.dataset} → {gcamp_name} calcium '
                       f'reconstruction ({n_tile}×{n_steps_trial * dt:.0f}s '
                       f'block)'),
                omega=omega_blk, v_fwd=v_fwd_blk,
                hd=hd_info, d=d_info, traj=traj_info,
                trial_s=n_steps_trial * dt, show_stitch=False)
            logger.info(
                f'  [calcium] {n_tile}-tile block reconstruction '
                f'(n_bump={real_kino.shape[1]}, n_aff={real_aff.shape[1]}):')
            for _gname, _gm in calcium_metrics.items():
                logger.info('    ' + _gname + ': ' + '  '.join(
                    f'{k} z-MSE={v["z_mse"]:.3f} SSIM={v["ssim"]:.3f}'
                    for k, v in _gm.items()))
            logger.info(f'  saved: {recon_path}')

    # --- Aggregate metrics log --------------------------------------------
    log_path_ = os.path.join(log_dir, 'results_path_integration.log')
    with open(log_path_, 'w') as f:
        f.write(f'full_test_pi_acc (n={u_test.shape[0]}, T={u_test.shape[1]}): {full_pi:.6f}\n')
        if has_rotation:
            f.write(f'mean_trial_rmse_deg (n={len(idx_sample)}): '
                    f'{np.nanmean(_rm):.4f} +- {np.nanstd(_rm):.4f}\n\n')
            f.write('# Random test trials\n')
            f.write('trial_idx,rmse_deg,pearson\n')
            for i, m in zip(idx_sample, metrics_random):
                f.write(f'{int(i)},{m["rmse_deg"]:.4f},{m["pearson"]:.6f}\n')
            f.write('\n# Deterministic sweeps (T=2000)\n')
            f.write('omega_deg,rmse_deg,pearson\n')
            for o, m in zip(omega_set, metrics_sweep):
                f.write(f'{o:.1f},{m["rmse_deg"]:.4f},{m["pearson"]:.6f}\n')
            f.write('\n# Integration gain (decoded HD slope / true ω)\n')
            f.write('omega_deg,slope_deg_per_s,gain,fit_r2\n')
            for m in gain_metrics:
                f.write(
                    f'{m["omega_deg"]:.1f},{m["slope_deg_per_s"]:.4f},'
                    f'{m["gain"]:.6f},{m["r2"]:.6f}\n'
                )
        if calcium_metrics is not None:
            f.write('\n# Calcium 600 s block reconstruction (per-neuron '
                    'z-scored)\n')
            f.write('neuron_group,rollout,z_mse,ssim,T_frames\n')
            for gname, gm in calcium_metrics.items():
                for key in ('stitch', 'continuous'):
                    m = gm.get(key)
                    if m is not None:
                        f.write(f'{gname},{key},{m["z_mse"]:.6f},'
                                f'{m["ssim"]:.6f},{int(m["T"])}\n')
    logger.info(f'  saved metrics log: {log_path_}')
    if log_file is not None:
        log_file.write('\n--- Path-integration test results ---\n')
        log_file.write(f'full_test_pi_acc: {full_pi:.4f}\n')
        log_file.write(
            'sweep mean rmse_deg: '
            f'{np.nanmean([m["rmse_deg"] for m in metrics_sweep]):.2f}°  '
            'sweep mean pearson: '
            f'{np.nanmean([m["pearson"] for m in metrics_sweep]):.3f}\n'
        )

    # Persist all extended trajectory metrics from the sweep blocks.
    if _traj_rows:
        with open(trajectory_metrics_path, 'w') as f:
            f.write(f'# trajectory metrics for {os.path.basename(log_dir)}\n')
            for line in _traj_rows:
                f.write(line + '\n')
        logger.info(f'  saved trajectory metrics: {trajectory_metrics_path}')

    # Single machine-loadable bundle of every raw decoded-vs-true trace and
    # per-trial metric (held-out OU trials + constant-velocity sweeps), so
    # downstream analysis loads heading/displacement with one np.load().
    if npz_bundle:
        try:
            npz_path = os.path.join(results_dir, 'test_metrics.npz')
            np.savez(npz_path, **npz_bundle)
            logger.info(f'  saved metrics bundle: {npz_path} '
                        f'({len(npz_bundle)} arrays)')
        except Exception as exc:
            logger.warning(f'  metrics-bundle save failed: {exc}')


def _color_r(r: float) -> str:
    """ANSI-colour-coded Pearson r for terminal output.

    Matches the progress-bar thresholds in graph_trainer.py: green ≥ 0.9,
    orange ≥ 0.5, red otherwise (including negative correlations, which
    indicate the integrator runs with the wrong sign).
    """
    if np.isnan(r):
        return 'n/a'
    if r >= 0.9:
        col = '\033[32m'  # green
    elif r >= 0.5:
        col = '\033[33m'  # orange/yellow
    else:
        col = '\033[31m'  # red (negative r included — anti-correlated)
    return f'{col}{r:+.3f}\033[0m'


def _per_trial_heading_metrics(
    y_pred: np.ndarray, theta_hd: np.ndarray, warmup: int = 10,
) -> list:
    """Per-trial (RMSE in deg, Pearson) on heading.

    y_pred: (N, T, 2) predicted (cos, sin)
    theta_hd: (N, T) ground-truth heading (cumsum / monotone or wrapped)
    """
    out = []
    for b in range(y_pred.shape[0]):
        decoded = np.arctan2(y_pred[b, :, 1], y_pred[b, :, 0])
        true = np.asarray(theta_hd[b])
        if true.size <= warmup:
            out.append({'rmse_deg': float('nan'), 'pearson': float('nan')})
            continue
        err = np.angle(np.exp(1j * (decoded[warmup:] - true[warmup:])))
        rmse_deg = float(np.degrees(np.sqrt(np.mean(err ** 2))))
        decoded_unwrap = np.unwrap(decoded[warmup:])
        if (decoded_unwrap.std() < 1e-8 or true[warmup:].std() < 1e-8):
            pearson = float('nan')
        else:
            pearson = float(np.corrcoef(decoded_unwrap, true[warmup:])[0, 1])
        out.append({'rmse_deg': rmse_deg, 'pearson': pearson})
    return out
