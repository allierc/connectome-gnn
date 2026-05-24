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
from connectome_gnn.generators.ode_params import FlyVisODEParams, load_edge_index
from connectome_gnn.generators.utils import generate_compressed_video_mp4
from connectome_gnn.log import get_logger
from connectome_gnn.models.utils import (
    ANSI_ORANGE,
    ANSI_RESET,
    r2_color,
)
from connectome_gnn.metrics import INDEX_TO_NAME
from connectome_gnn.models.neural_ode_wrapper import integrate_neural_ode
from connectome_gnn.models.registry import create_model
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

    # Load hidden neuron list for rollout
    # (has_hidden_neurons is defined once up top alongside has_visual_field)
    hidden_ids = None
    if has_hidden_neurons:
        _hidden_path = os.path.join(log_dir, 'hidden_neuron_ids.pt')
        if os.path.exists(_hidden_path):
            hidden_ids = torch.load(_hidden_path, map_location=device, weights_only=True)
            logger.info(f'hidden neurons: {len(hidden_ids)} — using during rollout')

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

            if 'stimulus' in model_config.signal_model_name.lower():
                tw = tc.time_window
                if k < tw - 1:
                    continue
                stim_ctx = x_ts_eval.stimulus[k-tw+1:k+1, :sim.n_input_neurons].unsqueeze(0)
                pred = model.predict_voltage(stim_ctx).squeeze(0)
                all_pred.append(to_numpy(pred))
                all_true.append(to_numpy(x_ts_eval.voltage[k]))
                continue
            elif 'rnn' in model_config.signal_model_name.lower():
                pred = model(x.to_packed(), return_all=False)
            elif 'mlp' in model_config.signal_model_name.lower() or 'eed' in model_config.signal_model_name.lower():
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
    if 'stimulus' in model_config.signal_model_name.lower():
        logger.info('stimulus model — skipping rollout (no recurrence)')
        return

    # --- Rollout evaluation ---
    # Start from initial voltages at t=0, predict autoregressively
    logger.info('running rollout evaluation ...')
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    x = x_ts_eval.frame(0)
    if has_hidden_neurons:
        if model.NNR_hidden is not None:
            x.voltage[hidden_ids] = model.forward_hidden(x, 0, hidden_ids).detach()
        else:
            x.voltage[hidden_ids] = 0.0

    h_state = None
    c_state = None

    # EED rollout runs in pure latent space: encode the initial voltage
    # once, chain the evolver in z, decode each step. z_latent persists
    # across iterations so the activity-space re-encoding loop is bypassed.
    is_eed = 'eed' in model_config.signal_model_name.lower()
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

            # Model prediction
            if 'rnn' in model_config.signal_model_name.lower():
                y, h_state = model(x.to_packed(), h=h_state, return_all=True)
            elif 'lstm' in model_config.signal_model_name.lower():
                y, h_state, c_state = model(x.to_packed(), h=h_state, c=c_state, return_all=True)
            elif 'mlp_ode' in model_config.signal_model_name.lower():
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
            elif 'mlp' in model_config.signal_model_name.lower():
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
            if 'mlp_ode' in model_config.signal_model_name.lower():
                x.voltage = x.voltage + y.squeeze(-1)
            else:
                x.voltage = x.voltage + sim.delta_t * y.squeeze(-1)

            # Update hidden neuron voltages via SIREN or keep silent
            if has_hidden_neurons:
                if model.NNR_hidden is not None:
                    x.voltage[hidden_ids] = model.forward_hidden(x, k + 1, hidden_ids).detach()
                else:
                    x.voltage[hidden_ids] = 0.0

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
    if has_hidden_neurons and hidden_ids is not None:
        _hidden_np = hidden_ids.detach().cpu().numpy().astype(int)
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

    # Model-specific type names. Fall through to FlyVisODEParams when the
    # signal_model_name isn't in the ODE registry (e.g. drosophila_cx_voltage),
    # so the saved type_names list (if any) is still picked up.
    from connectome_gnn.generators.ode_params import FlyVisODEParams, get_ode_params_class
    try:
        try:
            _OdeCls = get_ode_params_class(config.graph_model.signal_model_name)
        except KeyError:
            _OdeCls = FlyVisODEParams
        try:
            _ode_p = _OdeCls.load(graphs_data_path(config.dataset), device='cpu')
        except TypeError:
            # On-disk schema mismatch (e.g. registered class expects fields
            # we didn't save). Retry with the simpler FlyVisODEParams.
            _ode_p = FlyVisODEParams.load(graphs_data_path(config.dataset), device='cpu')
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
    if has_hidden_neurons and hidden_ids is not None and \
            getattr(model, 'NNR_hidden', None) is not None:
        inr = _compute_inr_traces(model, x_ts_eval, hidden_ids, device,
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
    if has_hidden_neurons and getattr(model, 'NNR_hidden', None) is not None:
        from connectome_gnn.plot import plot_hidden_siren_traces
        # Load anchor_ids if the run produced any (so the plot gets the 2-panel layout).
        _anchor_ids = None
        _anchor_path = os.path.join(log_dir, 'anchor_neuron_ids.pt')
        if getattr(model, 'n_anchor', 0) > 0 and os.path.exists(_anchor_path):
            _anchor_ids = torch.load(_anchor_path, map_location=device, weights_only=True)
        _hp, _ap = plot_hidden_siren_traces(
            model, x_ts_eval, hidden_ids, log_dir,
            epoch=0, N=0, device=device,
            n_traces=40, n_frames=min(2000, n_eval_frames),
            anchor_ids=_anchor_ids,
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

    ode_params = FlyVisODEParams.from_flyvis_network(net, device=device)
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

    # Initialize RNN hidden state
    _smn_lower = model_config.signal_model_name.lower()
    if 'rnn' in _smn_lower:
        h_state = None
    if 'lstm' in _smn_lower:
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

    if ('test_ablation' in test_mode) & ('MLP' not in model_config.signal_model_name) & ('rnn' not in _smn_lower) & ('lstm' not in _smn_lower):
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
                        if 'rnn' in _smn_lower:
                            y, h_state = model(x_selected.to_packed(), h=h_state, return_all=True)
                        elif 'lstm' in _smn_lower:
                            y, h_state, c_state = model(x_selected.to_packed(), h=h_state, c=c_state, return_all=True)
                        elif 'mlp_ode' in model_config.signal_model_name.lower():
                            v = x_selected.voltage.unsqueeze(-1)
                            I = x_selected.stimulus.unsqueeze(-1)
                            y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v  # Return as delta
                        elif 'mlp' in model_config.signal_model_name.lower() or 'eed' in model_config.signal_model_name.lower():
                            y = model(x_selected.to_packed(), data_id=None, return_all=False)

                    else:
                        if 'rnn' in _smn_lower:
                            y, h_state = model(x.to_packed(), h=h_state, return_all=True)
                        elif 'lstm' in _smn_lower:
                            y, h_state, c_state = model(x.to_packed(), h=h_state, c=c_state, return_all=True)
                        elif 'mlp_ode' in model_config.signal_model_name.lower():
                            v = x.voltage.unsqueeze(-1)
                            I = x.stimulus[:sim.n_input_neurons].unsqueeze(-1)
                            y = model.rollout_step(v, I, dt=sim.delta_t, method='rk4') - v  # Return as delta
                        elif 'mlp' in model_config.signal_model_name.lower() or 'eed' in model_config.signal_model_name.lower():
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
                        if 'mlp_ode' in model_config.signal_model_name.lower():
                            x_selected.voltage = x_selected.voltage + y.squeeze(-1)  # y already contains full update
                        else:
                            x_selected.voltage = x_selected.voltage + sim.delta_t * y.squeeze(-1)
                        if (it <= warm_up_length) and ('rnn' in _smn_lower or 'lstm' in _smn_lower):
                            x_selected.voltage = x_generated.voltage[selected_neuron_ids].clone()
                    else:
                        if 'mlp_ode' in model_config.signal_model_name.lower():
                            x.voltage = x.voltage + y.squeeze(-1)  # y already contains full update
                        else:
                            x.voltage = x.voltage + sim.delta_t * y.squeeze(-1)
                        if (it <= warm_up_length) and ('rnn' in _smn_lower):
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

def data_test_path_integration_task(
    config, best_model=None, device=None, log_file=None,
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
    from connectome_gnn.models.drosophila_cx_eval import (
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

    # --- Load test data ----------------------------------------------------
    # theta_hd is reconstructed from y = (cos θ, sin θ) rather than loaded
    # from theta_hd.zarr (which uses the 1-D-per-trial writer and reads back
    # with a different shape). y_test is (N, T, 2), so arctan2 → (N, T)
    # wrapped HD; np.unwrap restores the monotone cumulative-omega ramp the
    # Pearson metric needs.
    root = graphs_data_path(config.dataset)
    logger.info(f'[pi test] loading from {root}/test/...')
    u_test_np = load_raw_array(f"{root}/test/stimulus.zarr")
    y_test_np = load_raw_array(f"{root}/test/target.zarr")
    theta_wrap = np.arctan2(y_test_np[:, :, 1], y_test_np[:, :, 0])
    theta_test_np = np.unwrap(theta_wrap, axis=-1).astype(np.float32)
    try:
        is_stop_test_np = load_raw_array(f"{root}/test/is_stop.zarr")
        if is_stop_test_np.shape != theta_test_np.shape:
            is_stop_test_np = np.zeros(theta_test_np.shape, dtype=np.float32)
    except Exception:
        is_stop_test_np = np.zeros(theta_test_np.shape, dtype=np.float32)
    u_test = torch.from_numpy(u_test_np).to(device)
    y_test = torch.from_numpy(y_test_np).to(device)
    logger.info(f'  shapes: u={tuple(u_test.shape)}  y={tuple(y_test.shape)}')

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

    # --- Aggregate test pi_acc on full split (T=u_test.shape[1]) -----------
    # Test does no backward pass, so we use a much larger batch than training.
    # The training-side `tc.batch_size` is tuned to fit BPTT memory; here we
    # only need forward passes so 256 fits comfortably even for the GNN.
    test_bs = max(int(tc.batch_size), 256)
    full_pi = path_integration_accuracy_from_data(
        model, u_test, y_test, warmup=10, batch_size=test_bs,
    )
    logger.info(f'  full test pi_acc (n={u_test.shape[0]}, '
                f'T={u_test.shape[1]}): {full_pi:.4f}')

    # --- (a) 5 random test trials ------------------------------------------
    rng = np.random.default_rng(config.training.seed)
    idx_sample = rng.choice(u_test.shape[0], size=5, replace=False)
    idx_sample = np.sort(idx_sample)
    with torch.no_grad():
        y_pred_sample, _ = model(u_test[idx_sample])
    y_pred_sample_np = y_pred_sample.cpu().numpy()

    metrics_random = _per_trial_heading_metrics(
        y_pred_sample_np, theta_test_np[idx_sample],
    )
    logger.info(
        f'  5 random test trials (idx={idx_sample.tolist()}): '
        + '  '.join(
            f"#{i}: r={_color_r(m['pearson'])}"
            for i, m in zip(idx_sample, metrics_random)
        )
    )
    random_plot_path = os.path.join(results_dir, 'test_random_trials.png')
    plot_task_pi_traces(
        u=u_test_np[idx_sample],
        y=y_test_np[idx_sample],
        theta_hd=theta_test_np[idx_sample],
        is_stop=is_stop_test_np[idx_sample],
        dt=float(config.task.path_integration.dt),
        out_path=random_plot_path,
        n_show=5,
        y_pred=y_pred_sample_np,
        metrics=metrics_random,
    )
    logger.info(f'  saved: {random_plot_path}')

    # --- (b) 5 deterministic sweeps at ω ∈ {-120,-60,30,60,120}, T=2000 -----
    omega_set = [-120.0, -60.0, 30.0, 60.0, 120.0]
    T_sweep = 2000
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
    logger.info(
        '  5 deterministic sweeps (T=2000): '
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
        dt=float(config.task.path_integration.dt),
        out_path=sweep_plot_path,
        n_show=5,
        y_pred=y_pred_sweep_arr,
        metrics=metrics_sweep,
    )
    logger.info(f'  saved: {sweep_plot_path}')

    # --- (c) Integration-gain analysis (Hulse-style slope test) ------------
    # Denser ω scan than the 5-panel deterministic_sweep so the gain curve
    # has enough points to resolve where integration breaks down.
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
        dt=float(config.task.path_integration.dt),
        out_path=gain_plot_path,
    )
    logger.info(
        f'  {len(gain_omega_set)} integration gains (slope ÷ ω): '
        + '  '.join(
            f"ω={m['omega_deg']:+.0f}: g={m['gain']:+.3f}"
            for m in gain_metrics
        )
    )
    logger.info(f'  saved: {gain_plot_path}')

    # --- (d) Function dynamics along ω=60°/s rollout (GNN teachers only) ---
    # Hexbin of (h_i(t), f_theta(h_i(t))) and (h_j(t), g_phi(h_j(t))^2)
    # over the ω = +60°/s rollout already computed in (b), with the
    # static curves of fig 4 (k)/(l) overlaid. Skipped for non-GNN
    # teachers (TaskRNN has no f_theta / g_phi).
    if all(hasattr(model, name) for name in ("a", "g_phi", "f_theta")):
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

    # --- Aggregate metrics log --------------------------------------------
    log_path_ = os.path.join(log_dir, 'results_path_integration.log')
    with open(log_path_, 'w') as f:
        f.write(f'full_test_pi_acc (n={u_test.shape[0]}, T={u_test.shape[1]}): {full_pi:.6f}\n\n')
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
