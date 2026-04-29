import os
import glob
import re
import time
import logging
import warnings

import umap
# Fix umap / scikit-learn >=1.6 incompatibility (force_all_finite renamed to ensure_all_finite)
try:
    import sklearn.utils.validation as _skval
    _orig_check_array = _skval.check_array
    def _check_array_compat(*args, **kwargs):
        kwargs.pop('force_all_finite', None)
        return _orig_check_array(*args, **kwargs)
    _skval.check_array = _check_array_compat
    # Also patch the reference cached in umap's module namespace
    import umap.umap_ as _umap_mod
    if hasattr(_umap_mod, 'check_array'):
        _umap_mod.check_array = _check_array_compat
except Exception:
    pass
import torch
import numpy as np
import seaborn as sns
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD

_ANSI_RED = '\033[91m'
_ANSI_ORANGE = '\033[38;5;208m'
_ANSI_GREEN = '\033[92m'
_ANSI_BLUE = '\033[94m'
_ANSI_WHITE = '\033[97m'
_ANSI_RESET = '\033[0m'

def _r2_color(val):
    """Red < 0.3, orange < 0.7, green >= 0.7."""
    if val < 0.3: return _ANSI_RED
    if val < 0.7: return _ANSI_ORANGE
    return _ANSI_GREEN

def _rmse_color(val, good=0.05, bad=0.2):
    """Blue for all RMSE values."""
    return _ANSI_BLUE

from connectome_gnn.figure_style import default_style as fig_style
from connectome_gnn.zarr_io import load_simulation_data, load_raw_array
from connectome_gnn.sparsify import clustering_gmm
from connectome_gnn.models.neural_gnn import NeuralGNN  # noqa: F401 — kept for backwards compat
from connectome_gnn.models.registry import create_model
from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.metrics import (
    get_model_W,
    compute_r_squared,
    compute_r_squared_filtered,
    compute_all_corrected_weights,
    compute_activity_stats,
    extract_g_phi_slopes,
    extract_f_theta_slopes,
    derive_tau,
    derive_vrest,
    INDEX_TO_NAME,
    _vectorized_linspace,
    _batched_mlp_eval,
    _vectorized_linear_fit,
    _build_g_phi_features,
    _build_f_theta_features,
)
from connectome_gnn.plot import _plot_curves_fast
from connectome_gnn.generators.ode_params import load_edge_index, load_weights
from connectome_gnn.utils import (
    to_numpy,
    CustomColorMap,
    sort_key,
    create_log_dir,
    add_pre_folder,
    graphs_data_path,
    log_path,
    config_path,
    migrate_state_dict,
)

# Optional imports
try:
    from connectome_gnn.models.Ising_analysis import analyze_ising_model
except ImportError:
    analyze_ising_model = None

# Suppress matplotlib/PDF warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*Missing.*')

# Suppress fontTools logging (PDF font subsetting messages)
logging.getLogger('fontTools').setLevel(logging.ERROR)
logging.getLogger('fontTools.subset').setLevel(logging.ERROR)

# Configure matplotlib for Helvetica-style fonts (no LaTeX) and a clean
# rollout-style frame: only left + bottom spines visible on every panel,
# matching figures/fig_rollout_3col_noise_comparison_ablation50.py. The
# scatter PNGs consumed by figures/fig_*_params_3col_noise_comparison.py
# inherit this style so the composite figures look uniform.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'text.usetex': False,
    'mathtext.fontset': 'dejavusans',  # sans-serif math text
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def query_cell_types(config_or_dataset, name=None, device='cpu'):
    """Cell-type lookup for a flyvis (or biomodel) dataset.

    The per-neuron type-id vector lives inside x_list_train (`x.neuron_type`);
    the index→name mapping comes from `ode_params.type_names` when available
    (drosophila_cx, larva, zebrafish biomodels), or falls back to the FlyVis
    constant `INDEX_TO_NAME` (Am, C2, …, L4, …, TmY9).

    Args:
        config_or_dataset: a NeuralGraphConfig instance, or the dataset
            string (e.g. 'fly/flyvis_noise_005_010_blank50_ar1_rho25_blank50_cv00').
        name: optional shortcut. If a single str (e.g. 'L4'), returns the
            np.ndarray of neuron indices of that type. If a list of strs,
            returns one np.ndarray per name in the same order. If None,
            returns the full info dict described below.
        device: where to place the returned `type_list` tensor.

    Returns (when name is None):
        dict with
            type_list:     (N,) torch.LongTensor — per-neuron type id
            n_neurons:     int
            n_types:       int (= len(unique(type_list)))
            index_to_name: {int: str}
            name_to_index: {str: int}
            neurons_of:    callable name(s) -> np.ndarray (or list of)
            counts:        {str: int} — neurons per type name (only types present)
    """
    from connectome_gnn.generators.ode_params import (
        get_ode_params_class, FlyVisODEParams,
    )

    dataset = (config_or_dataset.dataset
               if isinstance(config_or_dataset, NeuralGraphConfig)
               else config_or_dataset)

    # 1. Per-neuron type ids — pulled from the saved x_list_train tensor.
    x_ts = load_simulation_data(graphs_data_path(dataset, 'x_list_train'),
                                fields=['neuron_type'])
    type_list = x_ts.neuron_type.to(device).long()
    n_neurons = int(type_list.shape[0])
    n_types_present = int(torch.unique(type_list).numel())

    # 2. Index → name mapping. Prefer ode_params.type_names if it exists
    # (set by the connectome biomodels); otherwise use the FlyVis hardcoded
    # mapping. Connectome biomodels with no explicit names get 'Type{i}'.
    index_to_name = None
    ode_params_path = graphs_data_path(dataset, 'ode_params.pt')
    if os.path.exists(ode_params_path):
        signal_model = (config_or_dataset.graph_model.signal_model_name
                        if isinstance(config_or_dataset, NeuralGraphConfig)
                        else None)
        try:
            cls = (get_ode_params_class(signal_model) if signal_model
                   else FlyVisODEParams)
        except KeyError:
            cls = FlyVisODEParams
        try:
            ode_params = cls.load(graphs_data_path(dataset), device='cpu')
            if hasattr(ode_params, 'type_names') and ode_params.type_names:
                index_to_name = {i: n for i, n in enumerate(ode_params.type_names)}
        except Exception:
            pass

    if index_to_name is None:
        is_connconstr = any(s in dataset for s in
                            ('drosophila_cx', 'zebrafish_oculomotor', 'larva'))
        if is_connconstr:
            index_to_name = {i: f'Type{i}' for i in range(n_types_present)}
        else:
            index_to_name = dict(INDEX_TO_NAME)

    name_to_index = {v: k for k, v in index_to_name.items()}
    type_list_np = to_numpy(type_list).astype(int).ravel()

    def _neurons_of(query):
        if isinstance(query, str):
            if query not in name_to_index:
                raise KeyError(
                    f"unknown cell type {query!r}. "
                    f"Known types ({len(name_to_index)}): "
                    f"{sorted(name_to_index)[:8]}…")
            return np.where(type_list_np == name_to_index[query])[0]
        return [_neurons_of(q) for q in query]

    if name is not None:
        return _neurons_of(name)

    counts = {nm: int(np.sum(type_list_np == idx))
              for idx, nm in index_to_name.items()
              if int(np.sum(type_list_np == idx)) > 0}

    return {
        'type_list':     type_list,
        'n_neurons':     n_neurons,
        'n_types':       n_types_present,
        'index_to_name': index_to_name,
        'name_to_index': name_to_index,
        'neurons_of':    _neurons_of,
        'counts':        counts,
    }


def get_training_files(log_dir, n_runs):
    """Return the list of per-epoch training checkpoints (sorted) used to
    plot weight evolution.

    Handles both filename conventions in the repo:
      - modern: best_model_with_<run>_graphs_<epoch>.pt   (one per epoch)
      - legacy: best_model_with_<run>_<epoch>_<sub>.pt    (multiple per epoch)
    """
    files = glob.glob(f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_*.pt")
    if len(files) == 0:
        print(f"  [get_training_files] no checkpoints in {log_dir}/models/ "
              f"matching best_model_with_{n_runs - 1}_graphs_*.pt")
        return [], np.array([])
    try:
        files.sort(key=sort_key)
    except (ValueError, TypeError) as e:
        # sort_key raises on filenames it doesn't recognise; surface what we
        # actually found so the user can see whether the directory is empty,
        # populated with a different naming convention, or has a stray file.
        print(f"  [get_training_files] failed to sort checkpoints in "
              f"{log_dir}/models/ — {type(e).__name__}: {e}")
        print(f"  files found ({len(files)}): "
              f"{[os.path.basename(f) for f in files][:10]}"
              + (" ..." if len(files) > 10 else ""))
        raise

    # Modern format ("..._graphs_<epoch>.pt"): one ckpt per epoch, return them all.
    if all(os.path.basename(f).split('_')[-2] == 'graphs' for f in files):
        return files, np.arange(len(files), dtype=int)

    # Legacy format ("..._<epoch>_<sub>.pt"): subsample sub-iters within each
    # epoch (50 points across the post-epoch-0 sub-iters, plus all epoch-0 ones).
    files = [f for f in files if os.path.basename(f).split('_')[-2].isdigit()]
    if len(files) == 0:
        return [], np.array([])

    files_with_0    = [f for f in files if int(os.path.basename(f).split('_')[-2]) == 0]
    files_without_0 = [f for f in files if int(os.path.basename(f).split('_')[-2]) != 0]

    indices_with_0    = np.arange(0, max(len(files_with_0) - 1, 0), dtype=int)
    selected = [files_with_0[i] for i in indices_with_0]
    if files_without_0:
        indices_without_0 = np.linspace(0, len(files_without_0) - 1, 50, dtype=int)
        selected += [files_without_0[i] for i in indices_without_0]

    return selected, np.arange(len(selected), dtype=int)


def _plot_tau_outlier_traces(activity_true, neuron_types, outlier_neuron_indices,
                              n_frames_actual, start_frame, index_to_name,
                              log_dir, config_indices, mc):
    """GT activity traces — one row per τ-outlier neuron, plus an R1 reference.
    Single long-window panel (≤64k frames).

    outlier_neuron_indices is the explicit list of neuron IDs flagged by the τ
    scatter outlier mask, so every outlier gets its own trace (no per-type
    subsampling).
    """
    outlier_neuron_indices = [int(i) for i in outlier_neuron_indices]
    if not len(outlier_neuron_indices):
        return
    name_to_idx = {v: k for k, v in index_to_name.items()}
    _to_indices, _to_labels = [], []
    for nid in outlier_neuron_indices:
        stype = int(neuron_types[nid])
        name = index_to_name.get(stype, f'Type{stype}')
        _to_indices.append(nid)
        _to_labels.append(f'{name}/{nid}')

    r1_id = name_to_idx.get('R1')
    r1_neuron_idx = None
    if r1_id is not None:
        r1_cands = np.where(neuron_types == r1_id)[0]
        if len(r1_cands) > 0:
            r1_neuron_idx = int(r1_cands[0])

    all_indices = list(_to_indices)
    all_labels  = list(_to_labels)
    if r1_neuron_idx is not None:
        all_indices.append(r1_neuron_idx)
        all_labels.append(f'R1/{r1_neuron_idx}')
    n_rows = len(all_indices)

    _n_show  = min(n_frames_actual, 64000)

    _baselines = np.array([
        float(np.mean(activity_true[nid, start_frame:start_frame + _n_show]))
        for nid in all_indices
    ])
    _act_std = np.std(activity_true[_to_indices, start_frame:start_frame + _n_show])
    _step_v = max(0.5, 3.0 * _act_std) if _act_std > 0 else 2.5

    # Per-row height shrinks for tall figures so 260 traces still render in a
    # tractable canvas. Label fontsize tracks the row spacing.
    _per_row_in = 0.15 if n_rows > 60 else 0.4
    _label_fs   = 7 if n_rows > 60 else 11
    _row_h = max(6, n_rows * _per_row_in + 2)
    fig, _ax = plt.subplots(1, 1, figsize=(15, _row_h))
    for i, nid in enumerate(all_indices):
        _is_r1 = (r1_neuron_idx is not None) and (i == n_rows - 1)
        _trace = activity_true[nid, start_frame:start_frame + _n_show] - _baselines[i]
        _color = 'red' if _is_r1 else 'green'
        _lw = 1.2 if _is_r1 else 0.8
        _label = ('R1 (reference)' if _is_r1 else
                  ('activity (GT)' if i == 0 else None))
        _ax.plot(_trace + i * _step_v, linewidth=_lw, c=_color, alpha=0.9, label=_label)
    for i, lbl in enumerate(all_labels):
        _is_r1 = (r1_neuron_idx is not None) and (i == n_rows - 1)
        _ax.text(-_n_show * 0.025, i * _step_v, lbl,
                 fontsize=_label_fs, va='center', ha='right',
                 color=('red' if _is_r1 else mc))
    _ax.set_ylim([-_step_v, n_rows * (_step_v + 0.25)])
    _ax.set_yticks([])
    _mid = _n_show // 2
    _ax.set_xticks([0, _mid, _n_show])
    _ax.set_xticklabels([0, _mid, _n_show], fontsize=14)
    _ax.set_xlabel(f'frame ({_n_show} frames)', fontsize=18)
    _ax.set_xlim([0, _n_show])
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.spines['left'].set_visible(False)
    _ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/activity_{config_indices}_tau_outliers.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_synaptic_linear(model, config, config_indices, log_dir, logger, mc,
                          edges, gt_weights, gt_taus, gt_V_Rest,
                          type_list, n_types, n_neurons, cmap, device,
                          extended, log_file, mu_activity, sigma_activity,
                          ode_params=None,
                          activity_true=None, n_frames_actual=None,
                          start_frame=0, index_to_name=None):
    """Analysis plots for LinearODE: W, tau, V_rest, gain, bias R² + clustering."""
    import torch.nn.functional as F
    sim = config.simulation

    # Cell-type index → name (used by the *_cell_type plots to label per-type
    # outlier clusters). Mirrors the lookup in plot_synaptic.
    if ode_params is not None and getattr(ode_params, 'type_names', None):
        _idx2name = {i: n for i, n in enumerate(ode_params.type_names)}
    else:
        _idx2name = INDEX_TO_NAME
    _type_list_np = to_numpy(type_list).astype(int).ravel()

    # --- Parameter table ---
    w_params = get_model_W(model).numel()
    tau_params = model.raw_tau.numel() if hasattr(model, 'raw_tau') else 0
    has_V_rest = hasattr(model, 'V_rest')
    has_bias = hasattr(model, 'bias') and not has_V_rest
    vrest_params = model.V_rest.numel() if has_V_rest else (model.bias.numel() if has_bias else 0)
    total_params = w_params + tau_params + vrest_params
    if hasattr(model, 'g'):
        total_params += model.g.numel()
    if hasattr(model, 'gain'):
        total_params += model.gain.numel()
    if hasattr(model, 's'):
        total_params += model.s.numel()
    print('learnable parameters:')
    print(f'  W (connectivity): {w_params:,}')
    if tau_params > 0:
        print(f'  tau (time constant): {tau_params:,}')
    if has_V_rest:
        print(f'  V_rest (resting potential): {vrest_params:,}')
    elif has_bias:
        print(f'  bias: {vrest_params:,}')
    print(f'  total: {total_params:,}')

    gt_taus_np = to_numpy(gt_taus[:n_neurons])
    gt_V_rest_np = to_numpy(gt_V_Rest[:n_neurons])
    gt_w_np = to_numpy(gt_weights)
    learned_weights = to_numpy(get_model_W(model).squeeze())

    if hasattr(model, 'get_learned_tau') and model.get_learned_tau() is not None:
        learned_tau = to_numpy(model.get_learned_tau()[:n_neurons])
    elif hasattr(model, 'raw_tau'):
        learned_tau = to_numpy(F.softplus(model.raw_tau[:n_neurons]).detach())
    else:
        learned_tau = np.zeros(n_neurons)
    if hasattr(model, 'get_learned_vrest') and model.get_learned_vrest() is not None:
        learned_V_rest = to_numpy(model.get_learned_vrest()[:n_neurons])
    elif has_V_rest:
        learned_V_rest = to_numpy(model.V_rest[:n_neurons].detach())
    elif has_bias:
        learned_V_rest = to_numpy(model.bias[:n_neurons].detach())
    else:
        learned_V_rest = np.zeros(n_neurons)

    # Gain and bias extraction (known_ode models)
    gt_gain_np = ode_params.gt_gain(n_neurons) if ode_params is not None else None
    gt_bias_np = ode_params.gt_bias(n_neurons) if ode_params is not None else None
    learned_gain = to_numpy(model.get_learned_gain()[:n_neurons]) if hasattr(model, 'get_learned_gain') and model.get_learned_gain() is not None else None
    learned_bias = to_numpy(model.get_learned_bias()[:n_neurons]) if hasattr(model, 'get_learned_bias') and model.get_learned_bias() is not None else None

    # --- Save learned parameters (mirrors ode_params.pt schema, no edge_index) ---
    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)
    learned_state = {
        'tau_i': torch.from_numpy(learned_tau),
        'V_i_rest': torch.from_numpy(learned_V_rest),
        'W': torch.from_numpy(learned_weights),
    }
    if learned_gain is not None:
        learned_state['gain'] = torch.from_numpy(learned_gain)
    if learned_bias is not None:
        learned_state['bias'] = torch.from_numpy(learned_bias)
    torch.save(learned_state, os.path.join(log_dir, 'results', 'learned_ode_params.pt'))

    # --- Plot 2: Raw W comparison ---
    fig = plt.figure(figsize=(10, 9))
    plt.scatter(gt_w_np, learned_weights, c=mc, s=0.1, alpha=0.1)
    r_squared_W, slope_W = compute_r_squared(gt_w_np, learned_weights)
    plt.text(0.05, 0.95, f'R²: {r_squared_W:.3f}\nslope: {slope_W:.2f}',
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $W_{ij}$', fontsize=48)
    plt.ylabel(r'learned $W_{ij}$', fontsize=48)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/weights_comparison_raw.png', dpi=300)
    plt.close()
    print(f"weights R²: {_r2_color(r_squared_W)}{r_squared_W:.4f}{_ANSI_RESET}  slope: {slope_W:.4f}")
    logger.info(f"weights R²: {r_squared_W:.4f}  slope: {slope_W:.4f}")
    # Relative error |learned - true| / max(|true|, eps), full sample.
    # Mean ± SD intentionally omitted: with the heavy-tailed weight
    # distribution a few near-zero true edges dominate the mean. Median + IQR
    # are the only honest summary numbers here.
    _gt_w_arr  = np.asarray(gt_w_np).ravel()
    _lrn_w_arr = np.asarray(learned_weights).ravel()
    _rel_err_w = np.abs(_lrn_w_arr - _gt_w_arr) / np.maximum(np.abs(_gt_w_arr), 1e-6)
    _rel_err_w_med  = float(np.median(_rel_err_w))
    _q1_w_re, _q3_w_re = np.percentile(_rel_err_w, [25.0, 75.0])
    _rel_err_w_iqr  = float(_q3_w_re - _q1_w_re)
    print(f"W rel.err: {_ANSI_GREEN}median {100*_rel_err_w_med:.1f}%  IQR {100*_rel_err_w_iqr:.1f}%{_ANSI_RESET}")
    logger.info(f"W rel.err: median {100*_rel_err_w_med:.2f}%  IQR {100*_rel_err_w_iqr:.2f}%")
    with open(os.path.join(log_dir, 'results', 'metrics.txt'), 'a') as _mf:
        _mf.write(f"W_rel_err_median: {_rel_err_w_med:.4f}\n")
        _mf.write(f"W_rel_err_iqr: {_rel_err_w_iqr:.4f}\n")

    # Fixed *x* range (true τ never changes) for cross-run comparability. y
    # range stays data-driven on learned_tau because that's what varies per run.
    _tau_xlim = (-0.025, 0.5)
    _tau_ticks = [0.0, 0.25, 0.5]
    _tau_tick_labels = ['0.0', '0.25', '0.5']
    _tau_lo = float(np.min(learned_tau)); _tau_hi = float(np.max(learned_tau))
    _tau_pad = 0.02 * (_tau_hi - _tau_lo) if _tau_hi > _tau_lo else 0.01
    _tau_ylim = (_tau_lo - _tau_pad, _tau_hi + _tau_pad)

    # --- Plot 3: tau comparison ---
    fig = plt.figure(figsize=(10, 9))
    plt.scatter(gt_taus_np, learned_tau, c=mc, s=1, alpha=0.3)
    r_squared_tau, slope_tau = compute_r_squared(gt_taus_np, learned_tau)
    plt.text(0.05, 0.95, f'R²: {r_squared_tau:.2f}\nslope: {slope_tau:.2f}',
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $\tau$', fontsize=48)
    plt.ylabel(r'learned $\tau$', fontsize=48)
    plt.xlim(_tau_xlim); plt.ylim(_tau_ylim)
    plt.xticks(_tau_ticks, _tau_tick_labels, fontsize=24)
    plt.yticks(_tau_ticks, _tau_tick_labels, fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/tau_comparison_{config_indices}.png', dpi=300)
    plt.close()
    print(f"tau R²: {_r2_color(r_squared_tau)}{r_squared_tau:.3f}{_ANSI_RESET}  slope: {slope_tau:.2f}")
    logger.info(f"tau R²: {r_squared_tau:.3f}  slope: {slope_tau:.2f}")
    # Relative error |learned - true| / max(|true|, eps), full sample. Mean ± SD
    # intentionally omitted (heavy tails inflate them); median + IQR only.
    _gt_t_arr   = np.asarray(gt_taus_np).ravel()
    _lrn_t_arr  = np.asarray(learned_tau).ravel()
    _rel_err_tau = np.abs(_lrn_t_arr - _gt_t_arr) / np.maximum(np.abs(_gt_t_arr), 1e-6)
    _rel_err_tau_med = float(np.median(_rel_err_tau))
    _q1_t_re, _q3_t_re = np.percentile(_rel_err_tau, [25.0, 75.0])
    _rel_err_tau_iqr = float(_q3_t_re - _q1_t_re)
    print(f"tau rel.err: {_ANSI_GREEN}median {100*_rel_err_tau_med:.1f}%  IQR {100*_rel_err_tau_iqr:.1f}%{_ANSI_RESET}")
    logger.info(f"tau rel.err: median {100*_rel_err_tau_med:.2f}%  IQR {100*_rel_err_tau_iqr:.2f}%")
    with open(os.path.join(log_dir, 'results', 'metrics.txt'), 'a') as _mf:
        _mf.write(f"tau_rel_err_median: {_rel_err_tau_med:.4f}\n")
        _mf.write(f"tau_rel_err_iqr: {_rel_err_tau_iqr:.4f}\n")

    # Outlier rule for the tau plots: absolute distance from identity line
    # > 0.1 (i.e. outside the y = x ± 0.1 band). Computed once here so Plot 3b
    # (cell-type labels) and Plot 3c (outlier viz) share the same mask.
    _gt_t   = np.asarray(gt_taus_np).ravel()
    _lrn_t  = np.asarray(learned_tau).ravel()
    _tau_outlier_thresh = 0.1
    _outlier_mask_t = np.abs(_lrn_t - _gt_t) > _tau_outlier_thresh
    _inlier_mask_t  = ~_outlier_mask_t
    n_outliers_tau = int(_outlier_mask_t.sum())

    # GT activity traces — one row per τ-outlier neuron (so the figure shows
    # every outlier flagged by the |Δτ|>thresh mask).
    if (activity_true is not None and n_frames_actual is not None
            and index_to_name is not None and n_outliers_tau > 0):
        _outlier_neuron_ids_t = np.where(_outlier_mask_t)[0].tolist()
        _plot_tau_outlier_traces(activity_true, _type_list_np, _outlier_neuron_ids_t,
                                 n_frames_actual, start_frame, index_to_name,
                                 log_dir, config_indices, mc)

    # --- Plot 3b: tau comparison colored by cell type, outlier clusters labelled ---
    fig = plt.figure(figsize=(10, 9))
    ax_3b = plt.gca()
    plt.scatter(gt_taus_np, learned_tau,
                c=cmap.color(to_numpy(type_list).astype(int)),
                s=4, alpha=0.6)
    # One label per cell type that has any outliers, placed at the centroid
    # of that type's outlier subset.
    for _t in np.unique(_type_list_np):
        _t_mask = (_type_list_np == _t) & _outlier_mask_t
        if not _t_mask.any():
            continue
        _x_lbl = float(_gt_t[_t_mask].mean())
        _y_lbl = float(_lrn_t[_t_mask].mean())
        ax_3b.text(_x_lbl, _y_lbl, _idx2name.get(int(_t), f'Type{int(_t)}'),
                   fontsize=11, ha='center', va='center',
                   color='black', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                             edgecolor='gray', alpha=0.75, linewidth=0.5))
    plt.text(0.05, 0.95, f'R²: {r_squared_tau:.2f}\nslope: {slope_tau:.2f}',
             transform=ax_3b.transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $\tau$', fontsize=48)
    plt.ylabel(r'learned $\tau$', fontsize=48)
    plt.xlim(_tau_xlim); plt.ylim(_tau_ylim)
    plt.xticks(_tau_ticks, _tau_tick_labels, fontsize=24)
    plt.yticks(_tau_ticks, _tau_tick_labels, fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/tau_comparison_cell_type_{config_indices}.png', dpi=300)
    plt.close()

    # --- Plot 3c: tau comparison with outliers in red, R²/slope on inliers only ---
    if int(_inlier_mask_t.sum()) >= 2:
        r2_tau_clean, slope_tau_clean = compute_r_squared(_gt_t[_inlier_mask_t], _lrn_t[_inlier_mask_t])
    else:
        r2_tau_clean, slope_tau_clean = float('nan'), float('nan')
    fig = plt.figure(figsize=(10, 9))
    plt.scatter(_gt_t[_inlier_mask_t], _lrn_t[_inlier_mask_t], c=mc, s=1, alpha=0.3)
    if n_outliers_tau > 0:
        plt.scatter(_gt_t[_outlier_mask_t], _lrn_t[_outlier_mask_t],
                    c='red', s=6, alpha=0.7)
    # Identity line ± threshold band (visual cue for the rejection criterion).
    _tlin = np.linspace(_tau_xlim[0], _tau_xlim[1], 2)
    plt.plot(_tlin, _tlin, '--', color='gray', linewidth=1, alpha=0.6)
    plt.plot(_tlin, _tlin + _tau_outlier_thresh, ':', color='gray', linewidth=1, alpha=0.5)
    plt.plot(_tlin, _tlin - _tau_outlier_thresh, ':', color='gray', linewidth=1, alpha=0.5)
    _pct_outliers_tau = (100.0 * n_outliers_tau / _gt_t.size) if _gt_t.size else 0.0
    _ax_3c = plt.gca()
    _ax_3c.text(0.05, 0.95,
                f'R²: {r2_tau_clean:.2f} ({r_squared_tau:.2f})\n'
                f'slope: {slope_tau_clean:.2f}',
                transform=_ax_3c.transAxes, verticalalignment='top', fontsize=32)
    _ax_3c.text(0.05, 0.78,
                f'outliers: {_pct_outliers_tau:.1f}%',
                transform=_ax_3c.transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $\tau$', fontsize=48)
    plt.ylabel(r'learned $\tau$', fontsize=48)
    plt.xlim(_tau_xlim); plt.ylim(_tau_ylim)
    plt.xticks(_tau_ticks, _tau_tick_labels, fontsize=24)
    plt.yticks(_tau_ticks, _tau_tick_labels, fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/tau_comparison_wo_outliers_{config_indices}.png', dpi=300)
    plt.close()
    print(f"tau (wo outliers) R²: {_r2_color(r2_tau_clean)}{r2_tau_clean:.3f}{_ANSI_RESET}  "
          f"slope: {slope_tau_clean:.2f}  "
          f"outliers: {n_outliers_tau}/{_gt_t.size} ({_pct_outliers_tau:.1f}%)")
    logger.info(f"tau_wo_outliers R²: {r2_tau_clean:.4f}  slope: {slope_tau_clean:.4f}  "
                f"outliers: {n_outliers_tau}/{_gt_t.size} ({_pct_outliers_tau:.1f}%)")
    _metrics_path_tau = os.path.join(log_dir, 'results', 'metrics.txt')
    with open(_metrics_path_tau, 'a') as _mf:
        _mf.write(f"tau_no_outliers_R2: {r2_tau_clean:.4f}\n")
        _mf.write(f"tau_no_outliers_slope: {slope_tau_clean:.4f}\n")
        _mf.write(f"tau_n_outliers: {n_outliers_tau}\n")

    # Fixed axis range for ALL V_rest comparison plots (4, 4b, 4c) so the
    # window is comparable across configs/runs.
    _v_xlim = (-0.025, 1.0)
    _v_ticks = [0.0, 0.5, 1.0]
    _v_tick_labels = ['0.0', '0.5', '1.0']
    _v_lo = float(np.min(learned_V_rest)); _v_hi = float(np.max(learned_V_rest))
    _v_pad = 0.02 * (_v_hi - _v_lo) if _v_hi > _v_lo else 0.01
    _v_ylim = (_v_lo - _v_pad, _v_hi + _v_pad)

    # --- Plot 4: V_rest comparison ---
    fig = plt.figure(figsize=(10, 9))
    plt.scatter(gt_V_rest_np, learned_V_rest, c=mc, s=1, alpha=0.3)
    r_squared_V_rest, slope_V_rest = compute_r_squared(gt_V_rest_np, learned_V_rest)
    plt.text(0.05, 0.95, f'R²: {r_squared_V_rest:.2f}\nslope: {slope_V_rest:.2f}',
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $V_{rest}$', fontsize=48)
    plt.ylabel(r'learned $V_{rest}$', fontsize=48)
    plt.xlim(_v_xlim); plt.ylim(_v_ylim)
    plt.xticks(_v_ticks, _v_tick_labels, fontsize=24)
    plt.yticks(_v_ticks, _v_tick_labels, fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/V_rest_comparison_{config_indices}.png', dpi=300)
    plt.close()
    print(f"V_rest R²: {_r2_color(r_squared_V_rest)}{r_squared_V_rest:.3f}{_ANSI_RESET}  slope: {slope_V_rest:.2f}")
    logger.info(f"V_rest R²: {r_squared_V_rest:.3f}  slope: {slope_V_rest:.2f}")
    # Relative error |learned - true| / max(|true|, eps), full sample. Mean ± SD
    # intentionally omitted (heavy tails inflate them); median + IQR only.
    _gt_v_arr   = np.asarray(gt_V_rest_np).ravel()
    _lrn_v_arr  = np.asarray(learned_V_rest).ravel()
    _rel_err_v = np.abs(_lrn_v_arr - _gt_v_arr) / np.maximum(np.abs(_gt_v_arr), 1e-6)
    _rel_err_v_med = float(np.median(_rel_err_v))
    _q1_v_re, _q3_v_re = np.percentile(_rel_err_v, [25.0, 75.0])
    _rel_err_v_iqr = float(_q3_v_re - _q1_v_re)
    print(f"V_rest rel.err: {_ANSI_GREEN}median {100*_rel_err_v_med:.1f}%  IQR {100*_rel_err_v_iqr:.1f}%{_ANSI_RESET}")
    logger.info(f"V_rest rel.err: median {100*_rel_err_v_med:.2f}%  IQR {100*_rel_err_v_iqr:.2f}%")
    with open(os.path.join(log_dir, 'results', 'metrics.txt'), 'a') as _mf:
        _mf.write(f"V_rest_rel_err_median: {_rel_err_v_med:.4f}\n")
        _mf.write(f"V_rest_rel_err_iqr: {_rel_err_v_iqr:.4f}\n")

    # Outlier rule for V_rest plots: absolute distance from identity line
    # > 0.2 (i.e. outside the y = x ± 0.2 band). Computed once so Plot 4b
    # (cell-type labels) and Plot 4c (outlier viz) share the same mask.
    _gt_v   = np.asarray(gt_V_rest_np).ravel()
    _lrn_v  = np.asarray(learned_V_rest).ravel()
    _vrest_outlier_thresh = 0.1
    _outlier_mask = np.abs(_lrn_v - _gt_v) > _vrest_outlier_thresh
    _inlier_mask  = ~_outlier_mask
    n_outliers = int(_outlier_mask.sum())

    # --- Plot 4b: V_rest comparison colored by cell type, outlier clusters labelled ---
    fig = plt.figure(figsize=(10, 9))
    ax_4b = plt.gca()
    plt.scatter(gt_V_rest_np, learned_V_rest,
                c=cmap.color(to_numpy(type_list).astype(int)),
                s=4, alpha=0.6)
    # Label a type only when >50% of its neurons are outliers, so labels
    # mark types that are systematically off rather than types with a few
    # stragglers (whose centroid would land inside the ±thresh band).
    for _t in np.unique(_type_list_np):
        _type_mask = (_type_list_np == _t)
        _n_type = int(_type_mask.sum())
        if _n_type == 0:
            continue
        _t_mask = _type_mask & _outlier_mask
        if _t_mask.sum() / _n_type <= 0.5:
            continue
        _x_lbl = float(_gt_v[_t_mask].mean())
        _y_lbl = float(_lrn_v[_t_mask].mean())
        ax_4b.text(_x_lbl, _y_lbl, _idx2name.get(int(_t), f'Type{int(_t)}'),
                   fontsize=11, ha='center', va='center',
                   color='black', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                             edgecolor='gray', alpha=0.75, linewidth=0.5))
    plt.text(0.05, 0.95, f'R²: {r_squared_V_rest:.2f}\nslope: {slope_V_rest:.2f}',
             transform=ax_4b.transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $V_{rest}$', fontsize=48)
    plt.ylabel(r'learned $V_{rest}$', fontsize=48)
    plt.xlim(_v_xlim); plt.ylim(_v_ylim)
    plt.xticks(_v_ticks, _v_tick_labels, fontsize=24)
    plt.yticks(_v_ticks, _v_tick_labels, fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/V_rest_comparison_cell_type_{config_indices}.png', dpi=300)
    plt.close()

    # --- Plot 4c: V_rest comparison with outliers in red, R²/slope on inliers only ---
    if int(_inlier_mask.sum()) >= 2:
        r2_v_clean, slope_v_clean = compute_r_squared(_gt_v[_inlier_mask], _lrn_v[_inlier_mask])
    else:
        r2_v_clean, slope_v_clean = float('nan'), float('nan')
    fig = plt.figure(figsize=(10, 9))
    plt.scatter(_gt_v[_inlier_mask], _lrn_v[_inlier_mask], c=mc, s=1, alpha=0.3)
    if n_outliers > 0:
        plt.scatter(_gt_v[_outlier_mask], _lrn_v[_outlier_mask],
                    c='red', s=6, alpha=0.7)
    # Identity line ± threshold band (visual cue for the rejection criterion).
    _vlin = np.linspace(_v_xlim[0], _v_xlim[1], 2)
    plt.plot(_vlin, _vlin, '--', color='gray', linewidth=1, alpha=0.6)
    plt.plot(_vlin, _vlin + _vrest_outlier_thresh, ':', color='gray', linewidth=1, alpha=0.5)
    plt.plot(_vlin, _vlin - _vrest_outlier_thresh, ':', color='gray', linewidth=1, alpha=0.5)
    _pct_outliers_v = (100.0 * n_outliers / _gt_v.size) if _gt_v.size else 0.0
    _ax_4c = plt.gca()
    _ax_4c.text(0.05, 0.95,
                f'R²: {r2_v_clean:.2f} ({r_squared_V_rest:.2f})\n'
                f'slope: {slope_v_clean:.2f}',
                transform=_ax_4c.transAxes, verticalalignment='top', fontsize=32)
    _ax_4c.text(0.05, 0.78,
                f'outliers: {_pct_outliers_v:.1f}%',
                transform=_ax_4c.transAxes, verticalalignment='top', fontsize=32)
    plt.xlabel(r'true $V_{rest}$', fontsize=48)
    plt.ylabel(r'learned $V_{rest}$', fontsize=48)
    plt.xlim(_v_xlim); plt.ylim(_v_ylim)
    plt.xticks(_v_ticks, _v_tick_labels, fontsize=24)
    plt.yticks(_v_ticks, _v_tick_labels, fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/V_rest_comparison_wo_outliers_{config_indices}.png', dpi=300)
    plt.close()
    print(f"V_rest (wo outliers) R²: {_r2_color(r2_v_clean)}{r2_v_clean:.3f}{_ANSI_RESET}  "
          f"slope: {slope_v_clean:.2f}  "
          f"outliers: {n_outliers}/{_gt_v.size} ({_pct_outliers_v:.1f}%)")
    logger.info(f"V_rest_wo_outliers R²: {r2_v_clean:.4f}  slope: {slope_v_clean:.4f}  "
                f"outliers: {n_outliers}/{_gt_v.size} ({_pct_outliers_v:.1f}%)")
    _metrics_path = os.path.join(log_dir, 'results', 'metrics.txt')
    with open(_metrics_path, 'a') as _mf:
        _mf.write(f"V_rest_no_outliers_R2: {r2_v_clean:.4f}\n")
        _mf.write(f"V_rest_no_outliers_slope: {slope_v_clean:.4f}\n")
        _mf.write(f"V_rest_n_outliers: {n_outliers}\n")

    # --- Plot 5: tau and V_rest per neuron ---
    fig = plt.figure(figsize=(10, 9))
    ax = plt.subplot(2, 1, 1)
    plt.scatter(np.arange(n_neurons), learned_tau,
                c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
    plt.ylabel(r'$\tau_i$', fontsize=48)
    plt.xticks([])
    plt.yticks(fontsize=24)
    ax = plt.subplot(2, 1, 2)
    plt.scatter(np.arange(n_neurons), learned_V_rest,
                c=cmap.color(to_numpy(type_list).astype(int)), s=2, alpha=0.5)
    plt.xlabel('neuron index', fontsize=48)
    plt.ylabel(r'$V^{\mathrm{rest}}_i$', fontsize=48)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/results/dynamics_params_{config_indices}.png", dpi=300)
    plt.close()

    # --- Plot 6: Gain comparison (if available) ---
    r_squared_gain = 0.0
    if gt_gain_np is not None and learned_gain is not None:
        fig = plt.figure(figsize=(10, 9))
        plt.scatter(gt_gain_np, learned_gain, c=mc, s=1, alpha=0.3)
        r_squared_gain, slope_gain = compute_r_squared(gt_gain_np, learned_gain)
        plt.text(0.05, 0.95, f'R²: {r_squared_gain:.2f}\nslope: {slope_gain:.2f}',
                 transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
        plt.xlabel(r'true gain', fontsize=48)
        plt.ylabel(r'learned gain', fontsize=48)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.savefig(f'{log_dir}/results/gain_comparison_{config_indices}.png', dpi=300)
        plt.close()
        print(f"gain R²: {_r2_color(r_squared_gain)}{r_squared_gain:.3f}{_ANSI_RESET}  slope: {slope_gain:.2f}")
        logger.info(f"gain R²: {r_squared_gain:.3f}  slope: {slope_gain:.2f}")

    # --- Plot 7: Bias comparison (if available) ---
    r_squared_bias = 0.0
    if gt_bias_np is not None and learned_bias is not None:
        fig = plt.figure(figsize=(10, 9))
        plt.scatter(gt_bias_np, learned_bias, c=mc, s=1, alpha=0.3)
        r_squared_bias, slope_bias = compute_r_squared(gt_bias_np, learned_bias)
        plt.text(0.05, 0.95, f'R²: {r_squared_bias:.2f}\nslope: {slope_bias:.2f}',
                 transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
        plt.xlabel(r'true bias', fontsize=48)
        plt.ylabel(r'learned bias', fontsize=48)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.savefig(f'{log_dir}/results/bias_comparison_{config_indices}.png', dpi=300)
        plt.close()
        print(f"bias R²: {_r2_color(r_squared_bias)}{r_squared_bias:.3f}{_ANSI_RESET}  slope: {slope_bias:.2f}")
        logger.info(f"bias R²: {r_squared_bias:.3f}  slope: {slope_bias:.2f}")

    # --- Write R² to log file ---
    if log_file:
        log_file.write(f"connectivity_R2: {r_squared_W:.4f}\n")
        log_file.write(f"tau_R2: {r_squared_tau:.4f}\n")
        log_file.write(f"tau_no_outliers_R2: {r2_tau_clean:.4f}\n")
        log_file.write(f"tau_n_outliers: {n_outliers_tau}\n")
        log_file.write(f"V_rest_R2: {r_squared_V_rest:.4f}\n")
        log_file.write(f"V_rest_no_outliers_R2: {r2_v_clean:.4f}\n")
        log_file.write(f"V_rest_n_outliers: {n_outliers}\n")
        if gt_gain_np is not None and learned_gain is not None:
            log_file.write(f"gain_R2: {r_squared_gain:.4f}\n")
        if gt_bias_np is not None and learned_bias is not None:
            log_file.write(f"bias_R2: {r_squared_bias:.4f}\n")

    # --- Eigenvalue / SVD analysis ---
    print('plot eigenvalue spectrum and eigenvector comparison ...')
    edges_np = to_numpy(edges)
    true_sparse = scipy.sparse.csr_matrix(
        (gt_w_np.flatten(), (edges_np[1], edges_np[0])),
        shape=(n_neurons, n_neurons))
    learned_sparse = scipy.sparse.csr_matrix(
        (learned_weights.flatten(), (edges_np[1], edges_np[0])),
        shape=(n_neurons, n_neurons))

    n_components = min(100, n_neurons - 1)
    svd_true = TruncatedSVD(n_components=n_components, random_state=42)
    svd_learned = TruncatedSVD(n_components=n_components, random_state=42)
    svd_true.fit(true_sparse)
    svd_learned.fit(learned_sparse)
    sv_true = svd_true.singular_values_
    sv_learned = svd_learned.singular_values_

    n_eigs = min(200, n_neurons - 2)
    eig_true = eig_learned = None
    try:
        eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
        eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')
    except Exception:
        try:
            n_eigs = min(50, n_neurons - 2)
            if eig_true is None:
                eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
            if eig_learned is None:
                eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')
        except Exception as e:
            logger.warning(f"eigenvalue computation failed: {e}")

    V_true = svd_true.components_
    V_learned = svd_learned.components_
    alignment = np.abs(V_true @ V_learned.T)

    n_show = min(100, n_components)
    U_true = svd_true.transform(true_sparse)[:, :n_show]
    U_learned = svd_learned.transform(learned_sparse)[:, :n_show]
    U_true = U_true / (np.linalg.norm(U_true, axis=0, keepdims=True) + 1e-10)
    U_learned = U_learned / (np.linalg.norm(U_learned, axis=0, keepdims=True) + 1e-10)
    alignment_L = np.abs(U_true.T @ U_learned)

    cumvar = np.cumsum(sv_true**2) / np.sum(sv_true**2)
    rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    rank_r = min(rank_99, n_components)

    # Full dense SVD -> Procrustes alignment. Memory: ~N^2 * 8 bytes for the
    # dense W; for flyvis (N≈13.7k) this is ~1.5 GB and SVD takes a few minutes.
    W_true_dense = true_sparse.toarray()
    W_learned_dense = learned_sparse.toarray()

    try:
        U_t, S_t, Vt_t = np.linalg.svd(W_true_dense, full_matrices=False)
        U_l, S_l, Vt_l = np.linalg.svd(W_learned_dense, full_matrices=False)

        U_t_r = U_t[:, :rank_r] * S_t[:rank_r]
        V_t_r = Vt_t[:rank_r, :]
        U_l_r = U_l[:, :rank_r] * S_l[:rank_r]
        V_l_r = Vt_l[:rank_r, :]

        from scipy.linalg import orthogonal_procrustes
        R_U, _ = orthogonal_procrustes(U_l_r, U_t_r)
        U_l_aligned = U_l_r @ R_U
        R_V, _ = orthogonal_procrustes(V_l_r.T, V_t_r.T)
        V_l_aligned = (V_l_r.T @ R_V).T

        def _r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - y_true.mean())**2)
            return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        U_r2 = _r2(U_t_r.ravel(), U_l_aligned.ravel())
        V_r2 = _r2(V_t_r.ravel(), V_l_aligned.ravel())

        W_recon_true = U_t_r @ V_t_r
        W_recon_learned = U_l_aligned @ V_l_aligned
        W_recon_r2 = _r2(W_recon_true.ravel(), W_recon_learned.ravel())

        rank_list = sorted(set([rank_90, rank_99, min(5, n_components),
                                min(10, n_components), min(20, n_components),
                                min(50, n_components)]))
        rank_list = [r for r in rank_list if r <= n_components]
        U_r2_per_rank = []
        V_r2_per_rank = []
        W_r2_per_rank = []
        for rr in rank_list:
            Ut_rr = U_t[:, :rr] * S_t[:rr]
            Vt_rr = Vt_t[:rr, :]
            Ul_rr = U_l[:, :rr] * S_l[:rr]
            Vl_rr = Vt_l[:rr, :]
            Ru, _ = orthogonal_procrustes(Ul_rr, Ut_rr)
            Rv, _ = orthogonal_procrustes(Vl_rr.T, Vt_rr.T)
            Ul_a = Ul_rr @ Ru
            Vl_a = (Vl_rr.T @ Rv).T
            U_r2_per_rank.append(_r2(Ut_rr.ravel(), Ul_a.ravel()))
            V_r2_per_rank.append(_r2(Vt_rr.ravel(), Vl_a.ravel()))
            W_r2_per_rank.append(_r2((Ut_rr @ Vt_rr).ravel(), (Ul_a @ Vl_a).ravel()))

        _rank20 = min(20, n_components)
        if _rank20 in rank_list:
            _idx20 = rank_list.index(_rank20)
            U_r2_rank20 = U_r2_per_rank[_idx20]
            V_r2_rank20 = V_r2_per_rank[_idx20]
            W_r2_rank20 = W_r2_per_rank[_idx20]
        else:
            Ut_20 = U_t[:, :_rank20] * S_t[:_rank20]
            Vt_20 = Vt_t[:_rank20, :]
            Ul_20 = U_l[:, :_rank20] * S_l[:_rank20]
            Vl_20 = Vt_l[:_rank20, :]
            Ru20, _ = orthogonal_procrustes(Ul_20, Ut_20)
            Rv20, _ = orthogonal_procrustes(Vl_20.T, Vt_20.T)
            Ul_20a = Ul_20 @ Ru20
            Vl_20a = (Vl_20.T @ Rv20).T
            U_r2_rank20 = _r2(Ut_20.ravel(), Ul_20a.ravel())
            V_r2_rank20 = _r2(Vt_20.ravel(), Vl_20a.ravel())
            W_r2_rank20 = _r2((Ut_20 @ Vt_20).ravel(), (Ul_20a @ Vl_20a).ravel())

        procrustes_ok = True
    except Exception as e:
        logger.warning(f"Procrustes SVD analysis failed: {e}")
        procrustes_ok = False

    fig, axes = plt.subplots(3, 3, figsize=(30, 30))

    # --- Row 1: spectral overview ---
    if eig_true is not None and eig_learned is not None:
        axes[0, 0].scatter(eig_true.real, eig_true.imag, s=100, c='green', alpha=0.7, label='true')
        axes[0, 0].scatter(eig_learned.real, eig_learned.imag, s=100, c='black', alpha=0.7, label='learned')
    elif eig_true is not None:
        axes[0, 0].scatter(eig_true.real, eig_true.imag, s=100, c='green', alpha=0.7, label='true')
        axes[0, 0].text(0.5, 0.5, 'learned W ≈ 0\n(no eigenvalues)', transform=axes[0, 0].transAxes,
                       ha='center', va='center', fontsize=20, color='red')
    else:
        axes[0, 0].text(0.5, 0.5, 'eigenvalue computation failed', transform=axes[0, 0].transAxes,
                       ha='center', va='center', fontsize=20, color='red')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0, 0].set_xlabel('real', fontsize=32)
    axes[0, 0].set_ylabel('imag', fontsize=32)
    axes[0, 0].legend(fontsize=20)
    axes[0, 0].tick_params(labelsize=20)
    axes[0, 0].set_title('eigenvalues in complex plane', fontsize=28)

    n_compare = min(len(sv_true), len(sv_learned))
    sv_r2 = _r2(sv_true[:n_compare], sv_learned[:n_compare]) if procrustes_ok else 0.0
    axes[0, 1].scatter(sv_true[:n_compare], sv_learned[:n_compare], s=100, c='black', edgecolors='black', alpha=0.7)
    max_val = max(sv_true.max(), sv_learned.max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'g--', linewidth=2)
    axes[0, 1].set_xlabel('true singular value', fontsize=32)
    axes[0, 1].set_ylabel('learned singular value', fontsize=32)
    axes[0, 1].tick_params(labelsize=20)
    axes[0, 1].set_title(f'singular value comparison  R²={sv_r2:.3f}', fontsize=28)

    axes[0, 2].plot(sv_true, color='green', linewidth=2, label='true')
    axes[0, 2].plot(sv_learned, color='black', linewidth=2, label='learned')
    axes[0, 2].axvline(x=rank_90, color='orange', linestyle='--', linewidth=1.5, label=f'rank@90%={rank_90}')
    axes[0, 2].axvline(x=rank_99, color='red', linestyle='--', linewidth=1.5, label=f'rank@99%={rank_99}')
    axes[0, 2].set_xlabel('index', fontsize=32)
    axes[0, 2].set_ylabel('singular value', fontsize=32)
    axes[0, 2].set_yscale('log')
    axes[0, 2].legend(fontsize=18)
    axes[0, 2].tick_params(labelsize=20)
    axes[0, 2].set_title('singular value spectrum (log scale)', fontsize=28)

    # --- Row 2: Procrustes-aligned U, V, W_recon ---
    if procrustes_ok:
        u_flat_t = U_t_r.ravel()
        u_flat_l = U_l_aligned.ravel()
        axes[1, 0].scatter(u_flat_t, u_flat_l, s=1, c='black', alpha=0.15, rasterized=True)
        umax = max(np.abs(u_flat_t).max(), np.abs(u_flat_l).max())
        axes[1, 0].plot([-umax, umax], [-umax, umax], 'g--', linewidth=2)
        axes[1, 0].set_xlabel('true U (output modes)', fontsize=28)
        axes[1, 0].set_ylabel('learned U (Procrustes-aligned)', fontsize=28)
        axes[1, 0].tick_params(labelsize=20)
        axes[1, 0].set_title(f'U (rank={rank_r})  R²={U_r2:.3f}', fontsize=28)
        axes[1, 0].set_aspect('equal')

        v_flat_t = V_t_r.ravel()
        v_flat_l = V_l_aligned.ravel()
        axes[1, 1].scatter(v_flat_t, v_flat_l, s=1, c='black', alpha=0.15, rasterized=True)
        vmax = max(np.abs(v_flat_t).max(), np.abs(v_flat_l).max())
        axes[1, 1].plot([-vmax, vmax], [-vmax, vmax], 'g--', linewidth=2)
        axes[1, 1].set_xlabel('true V (input selection)', fontsize=28)
        axes[1, 1].set_ylabel('learned V (Procrustes-aligned)', fontsize=28)
        axes[1, 1].tick_params(labelsize=20)
        axes[1, 1].set_title(f'V (rank={rank_r})  R²={V_r2:.3f}', fontsize=28)
        axes[1, 1].set_aspect('equal')

        w_flat_t = W_recon_true.ravel()
        w_flat_l = W_recon_learned.ravel()
        axes[1, 2].scatter(w_flat_t, w_flat_l, s=1, c='black', alpha=0.05, rasterized=True)
        wmax = max(np.abs(w_flat_t).max(), np.abs(w_flat_l).max())
        axes[1, 2].plot([-wmax, wmax], [-wmax, wmax], 'g--', linewidth=2)
        axes[1, 2].set_xlabel('true W (rank-r approx)', fontsize=28)
        axes[1, 2].set_ylabel('learned W (rank-r approx)', fontsize=28)
        axes[1, 2].tick_params(labelsize=20)
        axes[1, 2].set_title(f'W_recon (rank={rank_r})  R²={W_recon_r2:.3f}', fontsize=28)
        axes[1, 2].set_aspect('equal')
    else:
        for j in range(3):
            axes[1, j].text(0.5, 0.5, 'Procrustes failed\n(N too large or singular)',
                           transform=axes[1, j].transAxes, ha='center', va='center', fontsize=20, color='red')

    # --- Row 3: alignment matrices + R² per rank ---
    im = axes[2, 0].imshow(alignment[:n_show, :n_show], cmap='hot', vmin=0, vmax=1)
    axes[2, 0].set_xlabel('learned SV index', fontsize=28)
    axes[2, 0].set_ylabel('true SV index', fontsize=28)
    axes[2, 0].set_title('right SV alignment (V)', fontsize=28)
    axes[2, 0].tick_params(labelsize=16)
    plt.colorbar(im, ax=axes[2, 0], fraction=0.046)

    im_L = axes[2, 1].imshow(alignment_L, cmap='hot', vmin=0, vmax=1)
    axes[2, 1].set_xlabel('learned SV index', fontsize=28)
    axes[2, 1].set_ylabel('true SV index', fontsize=28)
    axes[2, 1].set_title('left SV alignment (U)', fontsize=28)
    axes[2, 1].tick_params(labelsize=16)
    plt.colorbar(im_L, ax=axes[2, 1], fraction=0.046)

    if procrustes_ok:
        axes[2, 2].plot(rank_list, U_r2_per_rank, 'o-', color='green', linewidth=2, markersize=8, label='U R²')
        axes[2, 2].plot(rank_list, V_r2_per_rank, 's-', color='black', linewidth=2, markersize=8, label='V R²')
        axes[2, 2].plot(rank_list, W_r2_per_rank, '^-', color='blue', linewidth=2, markersize=8, label='W_recon R²')
        axes[2, 2].axvline(x=rank_90, color='orange', linestyle='--', linewidth=1.5, label=f'rank@90%={rank_90}')
        axes[2, 2].axvline(x=rank_99, color='red', linestyle='--', linewidth=1.5, label=f'rank@99%={rank_99}')
        axes[2, 2].set_xlabel('truncation rank', fontsize=28)
        axes[2, 2].set_ylabel('R²', fontsize=28)
        axes[2, 2].set_title('Procrustes R² vs truncation rank', fontsize=28)
        axes[2, 2].set_ylim([-0.05, 1.05])
        axes[2, 2].legend(fontsize=18)
        axes[2, 2].tick_params(labelsize=20)
    else:
        best_alignment_R = np.max(alignment[:n_show, :n_show], axis=1)
        best_alignment_L = np.max(alignment_L, axis=1)
        axes[2, 2].scatter(range(len(best_alignment_R)), best_alignment_R, s=50, c='green', alpha=0.7, label=f'right (mean={np.mean(best_alignment_R):.2f})')
        axes[2, 2].scatter(range(len(best_alignment_L)), best_alignment_L, s=50, c='black', alpha=0.7, label=f'left (mean={np.mean(best_alignment_L):.2f})')
        axes[2, 2].axhline(y=1/np.sqrt(n_show), color='gray', linestyle='--', linewidth=2, label=f'random ({1/np.sqrt(n_show):.2f})')
        axes[2, 2].set_xlabel('SV index (sorted by singular value)', fontsize=28)
        axes[2, 2].set_ylabel('best alignment score', fontsize=28)
        axes[2, 2].set_title('best alignment per SV', fontsize=28)
        axes[2, 2].set_ylim([0, 1.05])
        axes[2, 2].legend(fontsize=20)
        axes[2, 2].tick_params(labelsize=16)

    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/eigen_comparison.png', dpi=87)
    plt.close()

    # --- Print and log all spectral/SVD metrics ---
    best_alignment_R = np.max(alignment[:n_show, :n_show], axis=1)
    best_alignment_L = np.max(alignment_L, axis=1)

    if eig_true is not None and eig_learned is not None:
        true_spectral_radius = np.max(np.abs(eig_true))
        learned_spectral_radius = np.max(np.abs(eig_learned))
        print(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
        logger.info(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
    else:
        print('spectral radius - skipped (eigenvalue computation failed)')
        logger.warning('spectral radius computation skipped')

    print(f'SV alignment - right (V): {np.mean(best_alignment_R):.3f}  left (U): {np.mean(best_alignment_L):.3f}')
    logger.info(f'SV alignment - right (V): {np.mean(best_alignment_R):.3f}  left (U): {np.mean(best_alignment_L):.3f}')
    if procrustes_ok:
        print(f'SV R² (singular values): {sv_r2:.4f}')
        logger.info(f'SV R² (singular values): {sv_r2:.4f}')
    print(f'effective rank - 90%: {rank_90}  99%: {rank_99}')
    logger.info(f'effective rank - 90%: {rank_90}  99%: {rank_99}')

    if procrustes_ok:
        print(f'Procrustes SVD (rank={rank_r}) - U R²: {U_r2:.4f}  V R²: {V_r2:.4f}  W_recon R²: {W_recon_r2:.4f}')
        logger.info(f'Procrustes SVD (rank={rank_r}) - U R²: {U_r2:.4f}  V R²: {V_r2:.4f}  W_recon R²: {W_recon_r2:.4f}')
        print(f'Procrustes rank-20: U R²: {U_r2_rank20:.4f}  V R²: {V_r2_rank20:.4f}  W R²: {W_r2_rank20:.4f}')
        logger.info(f'Procrustes rank-20: U R²: {U_r2_rank20:.4f}  V R²: {V_r2_rank20:.4f}  W R²: {W_r2_rank20:.4f}')
        print(f'Procrustes R² per rank:')
        logger.info(f'Procrustes R² per rank:')
        for i_rk, rr in enumerate(rank_list):
            line = f'  rank={rr:3d}  U_R2={U_r2_per_rank[i_rk]:.4f}  V_R2={V_r2_per_rank[i_rk]:.4f}  W_R2={W_r2_per_rank[i_rk]:.4f}'
            print(line)
            logger.info(line)

    if log_file:
        if eig_true is not None and eig_learned is not None:
            log_file.write(f"spectral_radius_true: {true_spectral_radius:.4f}\n")
            log_file.write(f"spectral_radius_learned: {learned_spectral_radius:.4f}\n")
        log_file.write(f"sv_alignment_R: {np.mean(best_alignment_R):.4f}\n")
        log_file.write(f"sv_alignment_L: {np.mean(best_alignment_L):.4f}\n")
        if procrustes_ok:
            log_file.write(f"sv_r2: {sv_r2:.4f}\n")
        log_file.write(f"effective_rank_90: {rank_90}\n")
        log_file.write(f"effective_rank_99: {rank_99}\n")
        if procrustes_ok:
            log_file.write(f"procrustes_rank: {rank_r}\n")
            log_file.write(f"procrustes_U_r2: {U_r2:.4f}\n")
            log_file.write(f"procrustes_V_r2: {V_r2:.4f}\n")
            log_file.write(f"procrustes_W_recon_r2: {W_recon_r2:.4f}\n")
            for i_rk, rr in enumerate(rank_list):
                log_file.write(f"procrustes_rank_{rr}_U_r2: {U_r2_per_rank[i_rk]:.4f}\n")
                log_file.write(f"procrustes_rank_{rr}_V_r2: {V_r2_per_rank[i_rk]:.4f}\n")
                log_file.write(f"procrustes_rank_{rr}_W_r2: {W_r2_per_rank[i_rk]:.4f}\n")
            log_file.write(f"U_r2_rank20: {U_r2_rank20:.4f}\n")
            log_file.write(f"V_r2_rank20: {V_r2_rank20:.4f}\n")
            log_file.write(f"W_r2_rank20: {W_r2_rank20:.4f}\n")

    # --- Clustering (no embeddings — use tau, V_rest, W stats) ---
    print('clustering learned features...')
    src, dst = edges_np[0], edges_np[1]

    def _connectivity_stats(w, src, dst, n):
        in_count = np.bincount(dst, minlength=n).astype(np.float64)
        out_count = np.bincount(src, minlength=n).astype(np.float64)
        in_sum = np.bincount(dst, weights=w, minlength=n)
        out_sum = np.bincount(src, weights=w, minlength=n)
        in_sq = np.bincount(dst, weights=w ** 2, minlength=n)
        out_sq = np.bincount(src, weights=w ** 2, minlength=n)
        safe_in = np.where(in_count > 0, in_count, 1)
        safe_out = np.where(out_count > 0, out_count, 1)
        in_mean = in_sum / safe_in
        out_mean = out_sum / safe_out
        in_std = np.sqrt(np.maximum(in_sq / safe_in - in_mean ** 2, 0))
        out_std = np.sqrt(np.maximum(out_sq / safe_out - out_mean ** 2, 0))
        # Per-neuron min / max of incoming and outgoing edge weights.
        in_max = np.full(n, -np.inf); np.maximum.at(in_max, dst, w)
        in_min = np.full(n,  np.inf); np.minimum.at(in_min, dst, w)
        out_max = np.full(n, -np.inf); np.maximum.at(out_max, src, w)
        out_min = np.full(n,  np.inf); np.minimum.at(out_min, src, w)
        for arr, c in [(in_mean, in_count), (in_std, in_count),
                       (in_min, in_count), (in_max, in_count),
                       (out_mean, out_count), (out_std, out_count),
                       (out_min, out_count), (out_max, out_count)]:
            arr[c == 0] = 0
        return in_mean, in_std, out_mean, out_std, in_min, in_max, out_min, out_max

    (w_in_mean, w_in_std, w_out_mean, w_out_std,
     w_in_min, w_in_max, w_out_min, w_out_max) = _connectivity_stats(
        learned_weights.flatten(), src, dst, n_neurons)
    W_learned = np.column_stack([w_in_mean, w_in_std, w_out_mean, w_out_std,
                                 w_in_min, w_in_max, w_out_min, w_out_max])

    (w_in_mean_t, w_in_std_t, w_out_mean_t, w_out_std_t,
     w_in_min_t, w_in_max_t, w_out_min_t, w_out_max_t) = _connectivity_stats(
        gt_w_np.flatten(), src, dst, n_neurons)
    W_true = np.column_stack([w_in_mean_t, w_in_std_t, w_out_mean_t, w_out_std_t,
                              w_in_min_t, w_in_max_t, w_out_min_t, w_out_max_t])

    n_gmm = min(100, n_neurons - 1)

    # Augmented clustering: (tau, V_rest, W_stats) since no embeddings
    a_aug = np.column_stack([learned_tau, learned_V_rest,
                             w_in_mean, w_in_std, w_out_mean, w_out_std,
                             w_in_min, w_in_max, w_out_min, w_out_max])
    results = clustering_gmm(a_aug, type_list, n_components=n_gmm)
    cluster_acc = results['accuracy']
    print(f"GMM (n_components={n_gmm}): accuracy={_r2_color(cluster_acc)}{cluster_acc:.3f}{_ANSI_RESET}, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")
    logger.info(f"GMM n_components={n_gmm}, accuracy={cluster_acc:.3f}, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")

    metrics_path = os.path.join(log_dir, 'results', 'metrics.txt')
    try:
        with open(metrics_path, 'a') as mf:
            mf.write(f"clustering_accuracy: {cluster_acc:.4f}\n")
    except OSError:
        pass
    if log_file:
        log_file.write(f"cluster_accuracy: {cluster_acc:.4f}\n")

    # UMAP scatter
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    a_umap = reducer.fit_transform(a_aug)
    cluster_labels = GaussianMixture(n_components=n_gmm, random_state=42).fit_predict(a_aug)

    colors_65 = sns.color_palette("Set3", 12) * 6
    colors_65 = colors_65[:65]
    from matplotlib.colors import ListedColormap
    cmap_65 = ListedColormap(colors_65)

    plt.figure(figsize=(10, 9))
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_alpha(0.75)
    plt.scatter(a_umap[:, 0], a_umap[:, 1], c=cluster_labels, s=24, cmap=cmap_65, alpha=0.8, edgecolors='none')
    plt.xlabel(r'UMAP$_1$', fontsize=48)
    plt.ylabel(r'UMAP$_2$', fontsize=48)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.text(0.05, 0.95, f"accuracy: {cluster_acc:.2f}",
             transform=plt.gca().transAxes, fontsize=32, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f'{log_dir}/results/embedding_augmented_{config_indices}.png', dpi=300)
    plt.close()

    # Per-neuron type analysis
    analyze_neuron_type_reconstruction(
        config=config, model=model, edges=to_numpy(edges),
        true_weights=gt_w_np, gt_taus=gt_taus_np, gt_V_Rest=gt_V_rest_np,
        learned_weights=learned_weights, learned_tau=learned_tau,
        learned_V_rest=learned_V_rest, type_list=to_numpy(type_list),
        n_frames=sim.n_frames, dimension=sim.dimension,
        n_neuron_types=sim.n_neuron_types, device=device,
        log_dir=log_dir, dataset_name=config.dataset, logger=logger,
        index_to_name=INDEX_TO_NAME,
        r_squared=r_squared_W, slope_corrected=slope_W,
        r_squared_tau=r_squared_tau, r_squared_V_rest=r_squared_V_rest)


def plot_synaptic(config, epoch_list, log_dir, logger, cc, style, extended, device, log_file=None, skip_svd=False):
    sim = config.simulation
    model_config = config.graph_model
    tc = config.training
    config_indices = os.path.basename(config.dataset).split('flyvis_')[1] if 'flyvis_' in config.dataset else re.sub(r'_\d{2}$', '', os.path.basename(config.dataset))


    colors_65 = sns.color_palette("Set3", 12) * 6  # pastel, repeat until 65
    colors_65 = colors_65[:65]

    config.simulation.max_radius if hasattr(config.simulation, 'max_radius') else 2.5

    results_log = os.path.join(log_dir, 'results.log')
    if os.path.exists(results_log):
        os.remove(results_log)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler only, no console output
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear any existing handlers

    file_handler = logging.FileHandler(results_log, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (which might have console handlers)
    logger.propagate = False

    # Clear metrics.txt at the start so all subsequent append-writes start fresh
    _metrics_path = os.path.join(log_dir, 'results', 'metrics.txt')
    if os.path.exists(_metrics_path):
        os.remove(_metrics_path)

    # Mirror rollout metrics from results_rollout.log (written by graph_tester)
    # into metrics.txt so downstream consumers (CV runners, LLM exploration,
    # verdict parsers) can read them from a single key-value file.
    _rollout_log = os.path.join(log_dir, 'results_rollout.log')
    if os.path.exists(_rollout_log):
        import re as _re
        with open(_rollout_log, 'r') as _rf:
            _rtext = _rf.read()
        _keys = (
            'Pearson r', 'RMSE', 'hidden_rollout_pearson',
            'visible_rollout_pearson', 'stimuli_R2', 'stimuli_r',
        )
        _mirrored = []
        for _k in _keys:
            _m = _re.search(rf'{_re.escape(_k)}:\s*([-\d.eE+]+)', _rtext)
            if _m:
                # Normalize 'Pearson r' → rollout_pearson, 'RMSE' → rollout_RMSE
                _canonical = {
                    'Pearson r': 'rollout_pearson',
                    'RMSE': 'rollout_RMSE',
                }.get(_k, _k)
                _mirrored.append(f'{_canonical}: {_m.group(1)}\n')
        if _mirrored:
            with open(_metrics_path, 'a') as _mf:
                _mf.writelines(_mirrored)

    print(f'experiment description: {config.description}')
    logger.info(f'experiment description: {config.description}')

    # Load neuron group mapping for flyvis

    cmap = CustomColorMap(config=config)

    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    time.sleep(0.5)
    print('\033[93mextracting parameters...\033[0m')
    x_path = graphs_data_path(config.dataset, 'x_list_train')
    if not os.path.exists(x_path):
        x_path = graphs_data_path(config.dataset, 'x_list_0')
    x_ts = load_simulation_data(x_path,
                                fields=['index', 'voltage', 'stimulus', 'neuron_type', 'group_type'])

    # Apply same stride as training to reduce memory for recurrent models with time_step > 1
    _stride = tc.time_step if (tc.recurrent_training and tc.time_step > 1) else 1
    if _stride > 1:
        print(f"\033[93msubsampling plot data: {x_ts.n_frames} → {x_ts.n_frames // _stride} frames (stride={_stride})\033[0m")
        for _field in ['voltage', 'stimulus']:
            _val = getattr(x_ts, _field)
            if _val is not None:
                setattr(x_ts, _field, _val[::_stride])

    # Cap plot data after stride (e.g. non-recurrent 320K hold-out → 64K)
    MAX_PLOT_FRAMES = 64000
    if x_ts.n_frames > MAX_PLOT_FRAMES:
        print(f"\033[93mcapping plot data: {x_ts.n_frames} → {MAX_PLOT_FRAMES} frames\033[0m")
        x_ts = x_ts.truncate_frames(MAX_PLOT_FRAMES)

    xnorm_path = os.path.join(log_dir, 'xnorm.pt')
    if os.path.exists(xnorm_path):
        xnorm = torch.load(xnorm_path, map_location=device, weights_only=False)
    else:
        xnorm = x_ts.xnorm.to(device)

    print(f'xnorm: {xnorm.item():0.3f}')
    logger.info(f'xnorm: {xnorm.item():0.3f}')

    type_list = x_ts.neuron_type.to(device)
    n_types = len(torch.unique(type_list))
    region_list = x_ts.group_type.to(device)
    n_region_types = len(torch.unique(region_list))
    n_neurons = x_ts.n_neurons

    # Load ODE params for model-specific analysis
    from connectome_gnn.generators.ode_params import get_ode_params_class, FlyVisODEParams
    signal_model = model_config.signal_model_name
    try:
        OdeParamsCls = get_ode_params_class(signal_model)
    except KeyError:
        OdeParamsCls = FlyVisODEParams
    _ode_params_path = graphs_data_path(config.dataset, 'ode_params.pt')
    if os.path.exists(_ode_params_path):
        ode_params = OdeParamsCls.load(graphs_data_path(config.dataset), device='cpu')
    else:
        ode_params = OdeParamsCls()  # empty, analysis methods return defaults

    gt_taus_np = ode_params.gt_tau(n_neurons)
    gt_taus = torch.tensor(gt_taus_np, device=device) if gt_taus_np is not None else torch.zeros(n_neurons, device=device)
    gt_vrest_np = ode_params.gt_vrest(n_neurons)
    gt_V_Rest = torch.tensor(gt_vrest_np, device=device) if gt_vrest_np is not None else torch.zeros(n_neurons, device=device)
    # Prefer training edges and gt_weights (handles fully connected mode)
    training_edges_path = os.path.join(log_dir, 'training_edges.pt')
    gt_weights_path = os.path.join(log_dir, 'gt_weights.pt')
    if os.path.exists(training_edges_path):
        edges = torch.load(training_edges_path, map_location=device, weights_only=False)
        gt_weights = torch.load(gt_weights_path, map_location=device, weights_only=False)
    else:
        data_folder = graphs_data_path(config.dataset)
        edges = load_edge_index(data_folder, device=device)
        gt_weights = load_weights(data_folder, device=device)
    true_weights = torch.zeros((n_neurons, n_neurons), dtype=torch.float32, device=edges.device)
    true_weights[edges[1], edges[0]] = gt_weights

    # Hidden neuron edge mask: exclude edges where source OR target was silenced during training
    _hidden_ids_path = os.path.join(log_dir, 'hidden_neuron_ids.pt')
    visible_edge_mask = None  # None = all edges visible
    if os.path.exists(_hidden_ids_path):
        _hidden_ids = torch.load(_hidden_ids_path, map_location='cpu', weights_only=True).numpy()
        _hidden_set = set(_hidden_ids.tolist())
        _e = edges.cpu().numpy()
        visible_edge_mask = np.array([
            _e[0, i] not in _hidden_set and _e[1, i] not in _hidden_set
            for i in range(_e.shape[1])
        ])
        n_hidden_edges = (~visible_edge_mask).sum()
        print(f"hidden neuron mask: {len(_hidden_ids)} hidden neurons, "
              f"{n_hidden_edges}/{len(visible_edge_mask)} edges excluded from R²")

    _connconstr = any(x in config.dataset for x in ('drosophila_cx', 'zebrafish_oculomotor', 'larva'))

    # Neuron type index to name mapping — load from ode_params if available
    if hasattr(ode_params, 'type_names') and ode_params.type_names:
        index_to_name = {i: name for i, name in enumerate(ode_params.type_names)}
    elif _connconstr:
        index_to_name = {i: f'Type{i}' for i in range(n_types)}
    else:
        index_to_name = {
            0: 'Am', 1: 'C2', 2: 'C3', 3: 'CT1(Lo1)', 4: 'CT1(M10)', 5: 'L1', 6: 'L2', 7: 'L3', 8: 'L4', 9: 'L5',
            10: 'Lawf1', 11: 'Lawf2', 12: 'Mi1', 13: 'Mi10', 14: 'Mi11', 15: 'Mi12', 16: 'Mi13', 17: 'Mi14',
            18: 'Mi15', 19: 'Mi2', 20: 'Mi3', 21: 'Mi4', 22: 'Mi9', 23: 'R1', 24: 'R2', 25: 'R3', 26: 'R4',
            27: 'R5', 28: 'R6', 29: 'R7', 30: 'R8', 31: 'T1', 32: 'T2', 33: 'T2a', 34: 'T3', 35: 'T4a',
            36: 'T4b', 37: 'T4c', 38: 'T4d', 39: 'T5a', 40: 'T5b', 41: 'T5c', 42: 'T5d', 43: 'Tm1',
            44: 'Tm16', 45: 'Tm2', 46: 'Tm20', 47: 'Tm28', 48: 'Tm3', 49: 'Tm30', 50: 'Tm4', 51: 'Tm5Y',
            52: 'Tm5a', 53: 'Tm5b', 54: 'Tm5c', 55: 'Tm9', 56: 'TmY10', 57: 'TmY13', 58: 'TmY14',
            59: 'TmY15', 60: 'TmY18', 61: 'TmY3', 62: 'TmY4', 63: 'TmY5a', 64: 'TmY9'
        }

    activity = x_ts.voltage.to(device).t()  # (N, T)
    mu_activity, sigma_activity = compute_activity_stats(x_ts, device)

    print(f'neurons: {n_neurons}  edges: {edges.shape[1]}  neuron types: {n_types}  region types: {n_region_types}')
    logger.info(f'neurons: {n_neurons}  edges: {edges.shape[1]}  neuron types: {n_types}  region types: {n_region_types}')
    os.makedirs(f'{log_dir}/results/', exist_ok=True)

    sorted_neuron_type_names = [index_to_name.get(i, f'Type{i}') for i in range(sim.n_neuron_types)]

    target_type_name_list = ['R1', 'R7', 'C2', 'Mi11', 'Tm1', 'Tm4', 'Tm30']
    activity_results = plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, sim.n_frames, sim.delta_t, f'{log_dir}/results/')
    plot_ground_truth_distributions(to_numpy(edges), to_numpy(gt_weights), to_numpy(gt_taus), to_numpy(gt_V_Rest), to_numpy(type_list), n_types, sorted_neuron_type_names, f'{log_dir}/results/')

    if ('Ising' in extended) | ('ising' in extended):
        analyze_ising_model(x_ts, sim.delta_t, log_dir, logger, to_numpy(edges))

    # Activity plots
    config_indices = os.path.basename(config.dataset).split('flyvis_')[1] if 'flyvis_' in config.dataset else re.sub(r'_\d{2}$', '', os.path.basename(config.dataset))
    neuron_types = to_numpy(type_list).astype(int).squeeze()

    # Activity traces are kept in memory so the per-epoch τ-outlier traces
    # plot can render them; voltage is (T, N), transpose to (N, T).
    activity_true = to_numpy(x_ts.voltage).T     # (n_neurons, n_frames_actual)
    n_frames_actual = activity_true.shape[1]

    start_frame = 0

    # τ-outlier traces are rendered later, inside the per-epoch loop,
    # using the actual outlier-type set detected from the learned τ — so the
    # figure always tracks the cell-type labels on tau_comparison_cell_type.

    if epoch_list[0] != 'all':
        config_indices = os.path.basename(config.dataset).split('flyvis_')[1] if 'flyvis_' in config.dataset else re.sub(r'_\d{2}$', '', os.path.basename(config.dataset))
        files, file_id_list = get_training_files(log_dir, tc.n_runs)

        for epoch in epoch_list:

            net = f'{log_dir}/models/best_model_with_{tc.n_runs - 1}_graphs_{epoch}.pt'
            # Load checkpoint first so we can use its W shape as the authoritative n_edges
            # (config yaml n_edges is an estimate; actual removal may differ by a few edges)
            state_dict = torch.load(net, map_location=device, weights_only=False)
            migrate_state_dict(state_dict)
            if 'W' in state_dict.get('model_state_dict', {}):
                ckpt_n_edges = state_dict['model_state_dict']['W'].shape[0]
                if ckpt_n_edges != config.simulation.n_edges:
                    logger.info(f'n_edges from checkpoint: {ckpt_n_edges} '
                                f'(config: {config.simulation.n_edges}) — using checkpoint')
                    config.simulation.n_edges = ckpt_n_edges
            model = create_model(model_config.signal_model_name,
                                 aggr_type=model_config.aggr_type, config=config, device=device)
            model.load_state_dict(state_dict['model_state_dict'], strict=False)
            model.edges = edges

            logger.info(f'net: {net}')

            # --- Linear model branch ---
            if 'linear' in model_config.signal_model_name or 'known_ode' in model_config.signal_model_name:
                _plot_synaptic_linear(
                    model, config, config_indices, log_dir, logger, mc,
                    edges, gt_weights, gt_taus, gt_V_Rest,
                    type_list, n_types, n_neurons, cmap, device,
                    extended, log_file, mu_activity, sigma_activity,
                    ode_params=ode_params,
                    activity_true=activity_true, n_frames_actual=n_frames_actual,
                    start_frame=start_frame, index_to_name=index_to_name)
                continue

            # print learnable parameters table
            if hasattr(model, 'f_theta') and hasattr(model, 'g_phi'):
                mlp0_params = sum(p.numel() for p in model.f_theta.parameters())
                mlp1_params = sum(p.numel() for p in model.g_phi.parameters())
                a_params = model.a.numel()
                w_params = get_model_W(model).numel()
                print('learnable parameters:')
                print(f'  f_theta: {mlp0_params:,}')
                print(f'  g_phi: {mlp1_params:,}')
                print(f'  a (embeddings): {a_params:,}')
                print(f'  W (connectivity): {w_params:,}')
                total_params = mlp0_params + mlp1_params + a_params + w_params
            else:
                total_params = sum(p.numel() for p in model.parameters())
                print(f'learnable parameters: {total_params:,} (flat model)')
            if hasattr(model, 'NNR_f') and model.NNR_f is not None:
                nnr_f_params = sum(p.numel() for p in model.NNR_f.parameters())
                print(f'  INR (NNR_f): {nnr_f_params:,}')
                total_params += nnr_f_params
            print(f'  total: {total_params:,}')

            # Adaptive dot size and alpha for different neuron counts
            _dot_s = max(10, min(48, 2000 / max(n_neurons, 1))) if n_neurons > 500 else max(30, min(80, 5000 / max(n_neurons, 1)))
            _dot_alpha = max(0.3, min(0.9, 100 / max(n_neurons, 1))) if n_neurons > 500 else 1.0
            _curve_alpha = max(0.1, min(0.8, 50 / max(n_neurons, 1))) if n_neurons > 500 else max(0.3, min(0.9, 100 / max(n_neurons, 1)))

            skip_gnn_plots = ('mlp' in model_config.signal_model_name.lower() or 'eed' in model_config.signal_model_name.lower() or 'stimulus' in model_config.signal_model_name.lower()) and not hasattr(model, 'f_theta')

            # --- Skip GNN-specific plots for MLP/EED baselines ---
            if skip_gnn_plots:
                print('skipping GNN-specific plots (embedding, g_phi, f_theta) for MLP/EED baseline')
                if log_file:
                    log_file.write(f"\n--- MLP baseline (no GNN parameter extraction) ---\n")
                    log_file.write(f"model_type: MLP\n")
                    log_file.write(f"total_params: {total_params}\n")
                continue  # skip to next epoch in epoch_list

            # Plot 2: Embedding using model.a
            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            for n in range(n_types):
                pos = torch.argwhere(type_list == n)
                plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]),
                            s=80, color=colors_65[n],
                            alpha=0.6, edgecolors='none')
            plt.xlabel(r'$a_{i0}$', fontsize=56)
            plt.ylabel(r'$a_{i1}$', fontsize=56)
            # Per-panel bbox from the 5-95 percentile of each axis to ignore
            # any outlier embeddings; small relative margin so points aren't
            # right on the spines.
            _a0 = to_numpy(model.a[:, 0])
            _a1 = to_numpy(model.a[:, 1])
            _x_lo, _x_hi = float(np.percentile(_a0, 5)), float(np.percentile(_a0, 95))
            _y_lo, _y_hi = float(np.percentile(_a1, 5)), float(np.percentile(_a1, 95))
            _x_pad = max(0.05 * (_x_hi - _x_lo), 1e-3)
            _y_pad = max(0.05 * (_y_hi - _y_lo), 1e-3)
            ax.set_xlim(_x_lo - _x_pad, _x_hi + _x_pad)
            ax.set_ylim(_y_lo - _y_pad, _y_hi + _y_pad)
            # Exactly 3 ticks per axis, rounded to 1 decimal for clean labels.
            ax.set_xticks([round(_x_lo, 1),
                           round(0.5 * (_x_lo + _x_hi), 1),
                           round(_x_hi, 1)])
            ax.set_yticks([round(_y_lo, 1),
                           round(0.5 * (_y_lo + _y_hi), 1),
                           round(_y_hi, 1)])
            ax.tick_params(axis='both', labelsize=51)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/embedding_{config_indices}.png', dpi=300)
            plt.close()

            n_pts = 1000
            post_fn = (lambda x: x ** 2) if model_config.g_phi_positive else None
            build_fn = lambda rr_f, emb_f: _build_g_phi_features(rr_f, emb_f, model_config.signal_model_name)
            type_np = to_numpy(type_list).astype(int).ravel()

            # g_phi domain range: evaluate + slope extraction (vectorized)
            mu = to_numpy(mu_activity).astype(np.float32)
            sigma = to_numpy(sigma_activity).astype(np.float32)

            # Slope extraction uses positive domain (clamped to 0)
            valid_edge = (mu + sigma) > 0
            starts_edge_slope = np.maximum(mu - 2 * sigma, 0.0)
            ends_edge = mu + 2 * sigma
            starts_edge_slope[~valid_edge] = 0.0
            ends_edge[~valid_edge] = 1.0
            rr_domain_edge_slope = _vectorized_linspace(starts_edge_slope, ends_edge, n_pts, device)
            func_domain_edge_slope = _batched_mlp_eval(model.g_phi, model.a[:n_neurons], rr_domain_edge_slope,
                                                 build_fn, device, post_fn=post_fn)
            slopes_edge, _ = _vectorized_linear_fit(rr_domain_edge_slope, func_domain_edge_slope)
            slopes_edge[~valid_edge] = 1.0
            slopes_g_phi_list = slopes_edge  # (N,) numpy array

            # Domain plot includes negative values to show g_phi → 0 for v < 0
            starts_edge_plot = mu - 2 * sigma
            starts_edge_plot[~valid_edge] = -0.5
            ends_edge_plot = ends_edge.copy()
            rr_domain_edge = _vectorized_linspace(starts_edge_plot, ends_edge_plot, n_pts, device)
            func_domain_edge = _batched_mlp_eval(model.g_phi, model.a[:n_neurons], rr_domain_edge,
                                                 build_fn, device, post_fn=post_fn)

            rr_np = to_numpy(rr_domain_edge)
            func_np = to_numpy(func_domain_edge)

            # Ground truth g_phi via ODE params registry
            func_true_g_phi = ode_params.gt_g_phi_func(rr_np)

            # Side-by-side: true (left) vs learned (right)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
            _plot_curves_fast(ax1, rr_np, func_true_g_phi,
                              type_np, cmap, linewidth=1, alpha=_curve_alpha)
            ax1.set_xlabel('$v_j$', fontsize=48)
            ax1.set_ylabel(f'true {ode_params.g_phi_label()}', fontsize=48)
            ax1.tick_params(axis='both', which='major', labelsize=24)
            ax1.set_xlim([-1, 5])
            ax1.set_ylim([-config.plotting.xlim[1]/10, config.plotting.xlim[1]*2])

            _plot_curves_fast(ax2, rr_np, func_np,
                              type_np, cmap, linewidth=1, alpha=_curve_alpha)
            ax2.set_xlabel('$v_j$', fontsize=48)
            ax2.set_ylabel(r'learned $g_\phi(a_j, v_j)$', fontsize=48)
            ax2.tick_params(axis='both', which='major', labelsize=24)
            ax2.set_xlim([-1, 5])
            ax2.set_ylim([-config.plotting.xlim[1]/10, config.plotting.xlim[1]*2])

            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/g_phi_{config_indices}_domain.png", dpi=300)
            plt.close()

            # NEW: scatter plot of learned vs true g_phi outputs.
            # Each point = one (neuron, v_j domain point) pair, coloured by
            # neuron type. Complements the side-by-side curve view above.
            if func_true_g_phi is not None and func_true_g_phi.shape == func_np.shape:
                _x_flat = func_true_g_phi.ravel()
                _y_flat = func_np.ravel()
                _type_flat = np.repeat(type_np, func_np.shape[1])
                _r2_g_scatter, _slope_g_scatter = compute_r_squared(_x_flat, _y_flat)
                fig = plt.figure(figsize=(10, 9))
                plt.scatter(_x_flat[::10], _y_flat[::10],
                            c=cmap.color(_type_flat[::10].astype(int)),
                            s=max(4.0, _dot_s * 0.8),
                            alpha=0.04,
                            edgecolors=None)
                plt.text(0.05, 0.95,
                         f'R²: {_r2_g_scatter:.2f}\nslope: {_slope_g_scatter:.2f}',
                         transform=plt.gca().transAxes,
                         verticalalignment='top', fontsize=42)
                plt.xlabel(r'true $g^*_\phi(a_j, v_j)$', fontsize=56)
                plt.ylabel(r'learned $g^*_\phi(a_j, v_j)$', fontsize=56)
                # g_phi^* is non-negative; ticks placed inside the [0, 6]
                # limits so the bottom-left corner labels don't overlap.
                plt.xlim(0, 6); plt.ylim(0, 6)
                plt.xticks([0.5, 3, 5.5], ['0.5', '3', '5.5'], fontsize=51)
                plt.yticks([0.5, 3, 5.5], ['0.5', '3', '5.5'], fontsize=51)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/g_phi_scatter_{config_indices}.png',
                            dpi=300)
                plt.close()

            # Functional Pearson r² for g_phi: per-neuron correlation between true and learned curves
            if func_true_g_phi is not None and func_true_g_phi.shape == func_np.shape:
                _true_g = func_true_g_phi
                _learned_g = func_np
                # Per-neuron Pearson r² (shape match, invariant to scale/offset)
                _corr_g = np.array([np.corrcoef(t, l)[0, 1] ** 2
                                    if np.std(t) > 1e-10 and np.std(l) > 1e-10 else 0.0
                                    for t, l in zip(_true_g, _learned_g)])
                r2_g_phi_mean = float(np.mean(_corr_g))
                r2_g_phi_median = float(np.median(_corr_g))
            else:
                r2_g_phi_mean = 0.0
                r2_g_phi_median = 0.0

            fig = plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            slopes_g_phi_array = np.array(slopes_g_phi_list)
            plt.scatter(np.arange(n_neurons), slopes_g_phi_array,
                        c=cmap.color(to_numpy(type_list).astype(int)), s=_dot_s, alpha=_dot_alpha)
            plt.xlabel('neuron index', fontsize=48)
            plt.ylabel(r'$r_j$', fontsize=48)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/g_phi_slope_{config_indices}.png", dpi=300)
            plt.close()

            # f_theta domain range: evaluate + slope extraction (vectorized)
            starts_phi = mu - 2 * sigma
            ends_phi = mu + 2 * sigma
            rr_domain_phi = _vectorized_linspace(starts_phi, ends_phi, n_pts, device)
            func_domain_phi = _batched_mlp_eval(model.f_theta, model.a[:n_neurons], rr_domain_phi,
                                                lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f), device)
            slopes_phi, offsets_phi = _vectorized_linear_fit(rr_domain_phi, func_domain_phi)
            slopes_f_theta_list = slopes_phi  # (N,) numpy array
            offsets_list = offsets_phi

            # Ground truth f_theta via ODE params registry
            gt_taus_np = to_numpy(gt_taus[:n_neurons])
            gt_V_rest_np = to_numpy(gt_V_Rest[:n_neurons])
            rr_domain_phi_np = to_numpy(rr_domain_phi)
            func_true_f_theta = ode_params.gt_f_theta_func(rr_domain_phi_np, n_neurons) if hasattr(ode_params, 'gt_f_theta_func') else (-rr_domain_phi_np + gt_V_rest_np[:, None]) / np.maximum(gt_taus_np[:, None], 1e-8)

            # Side-by-side: true (left) vs learned (right)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
            if func_true_f_theta is not None:
                _plot_curves_fast(ax1, rr_domain_phi_np, func_true_f_theta,
                                  type_np, cmap, linewidth=1, alpha=_curve_alpha)
            ax1.set_xlim(config.plotting.xlim)
            ax1.set_ylim(config.plotting.ylim)
            ax1.set_xlabel('$v_i$', fontsize=48)
            ax1.set_ylabel(f'true {ode_params.f_theta_label()}', fontsize=48)
            ax1.tick_params(axis='both', which='major', labelsize=24)

            _plot_curves_fast(ax2, to_numpy(rr_domain_phi), to_numpy(func_domain_phi),
                              type_np, cmap, linewidth=1, alpha=_curve_alpha)
            ax2.set_xlim(config.plotting.xlim)
            ax2.set_ylim(config.plotting.ylim)
            ax2.set_xlabel('$v_i$', fontsize=48)
            ax2.set_ylabel(r'learned $f_\theta(a_i, v_i)$', fontsize=48)
            ax2.tick_params(axis='both', which='major', labelsize=24)

            plt.tight_layout()
            plt.savefig(f"{log_dir}/results/f_theta_{config_indices}_domain.png", dpi=300)
            plt.close()

            # NEW: scatter plot of learned vs true f_theta outputs.
            # Each point = one (neuron, v_i domain point) pair, coloured by
            # neuron type. Complements the side-by-side curve view above.
            _func_learned_f = to_numpy(func_domain_phi)
            if func_true_f_theta is not None and func_true_f_theta.shape == _func_learned_f.shape:
                _x_flat = func_true_f_theta.ravel()
                _y_flat = _func_learned_f.ravel()
                _type_flat = np.repeat(type_np, _func_learned_f.shape[1])
                _r2_f_scatter, _slope_f_scatter = compute_r_squared(_x_flat, _y_flat)
                fig = plt.figure(figsize=(10, 9))
                plt.scatter(_x_flat[::10], _y_flat[::10],
                            c=cmap.color(_type_flat[::10].astype(int)),
                            s=max(4.0, _dot_s * 0.8),
                            alpha=0.04,
                            edgecolors=None)
                plt.text(0.05, 0.95,
                         f'R²: {_r2_f_scatter:.2f}\nslope: {_slope_f_scatter:.2f}',
                         transform=plt.gca().transAxes,
                         verticalalignment='top', fontsize=42)
                plt.xlabel(r'true $f_\theta(a_i, v_i)$', fontsize=56)
                plt.ylabel(r'learned $f_\theta(a_i, v_i)$', fontsize=56)
                # f_theta values widen with noise level: σ=0.5 column needs
                # ±275; nf and low-noise stay on ±210. Tick values are placed
                # slightly INSIDE the axis limit so the leftmost / bottommost
                # labels don't overlap each other at the corner.
                if sim.noise_model_level >= 0.5:
                    _ft_lim, _ft_tick = 275, 250
                else:
                    _ft_lim, _ft_tick = 210, 200
                plt.xlim(-_ft_lim, _ft_lim); plt.ylim(-_ft_lim, _ft_lim)
                plt.xticks([-_ft_tick, 0, _ft_tick], fontsize=51)
                plt.yticks([-_ft_tick, 0, _ft_tick], fontsize=51)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/f_theta_scatter_{config_indices}.png',
                            dpi=300)
                plt.close()

            # Functional Pearson r² for f_theta: per-neuron correlation between true and learned curves
            func_domain_phi_np = to_numpy(func_domain_phi)
            if func_true_f_theta is not None and func_true_f_theta.shape == func_domain_phi_np.shape:
                _corr_f = np.array([np.corrcoef(t, l)[0, 1] ** 2
                                    if np.std(t) > 1e-10 and np.std(l) > 1e-10 else 0.0
                                    for t, l in zip(func_true_f_theta, func_domain_phi_np)])
                r2_f_theta_mean = float(np.mean(_corr_f))
                r2_f_theta_median = float(np.median(_corr_f))
            else:
                r2_f_theta_mean = 0.0
                r2_f_theta_median = 0.0

            slopes_f_theta_array = np.array(slopes_f_theta_list)
            offsets_array = np.array(offsets_list)
            gt_taus_np = ode_params.gt_tau(n_neurons)
            learned_tau = ode_params.derive_tau(slopes_f_theta_array, n_neurons)
            r_squared_tau = 0.0
            slope_tau = 0.0

            if ode_params.has_tau() and gt_taus_np is not None:
                # Square fixed window matching the x-axis so the identity line
                # is the diagonal across the whole panel.
                _tau_ylim_p = (-0.025, 0.5)

                fig = plt.figure(figsize=(10, 9))
                plt.scatter(gt_taus_np, learned_tau, c=mc, s=_dot_s, alpha=_dot_alpha)
                r_squared_tau, slope_tau = compute_r_squared(gt_taus_np, learned_tau)
                plt.text(0.05, 0.95, f'R²: {r_squared_tau:.2f}\nslope: {slope_tau:.2f}',
                         transform=plt.gca().transAxes, verticalalignment='top', fontsize=42)
                plt.xlabel(r'true $\tau$', fontsize=56)
                plt.ylabel(r'learned $\tau$', fontsize=56)
                plt.xlim(-0.025, 0.5); plt.ylim(_tau_ylim_p)
                plt.xticks([0.0, 0.25, 0.5], ['0.0', '0.25', '0.5'], fontsize=51)
                plt.yticks([0.0, 0.25, 0.5], ['0.0', '0.25', '0.5'], fontsize=51)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/tau_comparison_{config_indices}.png', dpi=300)
                plt.close()

                # Outlier mask on |learned - true| > 0.1, shared by both extra plots.
                _gt_t   = np.asarray(gt_taus_np).ravel()
                _lrn_t  = np.asarray(learned_tau).ravel()
                _tau_outlier_thresh = 0.1
                _outlier_mask_t = np.abs(_lrn_t - _gt_t) > _tau_outlier_thresh
                _inlier_mask_t  = ~_outlier_mask_t
                n_outliers_tau = int(_outlier_mask_t.sum())
                _type_list_np_t = to_numpy(type_list).astype(int).ravel()

                # GT activity traces for the τ-outlier types in this run
                # (matches tau_comparison_cell_type's labeled cell types).
                if n_outliers_tau > 0:
                    _outlier_types_t = sorted({int(t) for t in _type_list_np_t[_outlier_mask_t]})
                    _plot_tau_outlier_traces(activity_true, _type_list_np_t,
                                             _outlier_types_t,
                                             n_frames_actual, start_frame,
                                             index_to_name, log_dir,
                                             config_indices, mc)

                # tau_comparison_cell_type — colored by cell type, outlier clusters labelled.
                fig = plt.figure(figsize=(10, 9))
                ax_tc = plt.gca()
                plt.scatter(gt_taus_np, learned_tau,
                            c=cmap.color(_type_list_np_t),
                            s=4, alpha=0.6)
                for _t in np.unique(_type_list_np_t):
                    _t_mask = (_type_list_np_t == _t) & _outlier_mask_t
                    if not _t_mask.any():
                        continue
                    _x_lbl = float(_gt_t[_t_mask].mean())
                    _y_lbl = float(_lrn_t[_t_mask].mean())
                    ax_tc.text(_x_lbl, _y_lbl,
                               index_to_name.get(int(_t), f'Type{int(_t)}'),
                               fontsize=11, ha='center', va='center',
                               color='black', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                         edgecolor='gray', alpha=0.75, linewidth=0.5))
                plt.text(0.05, 0.95, f'R²: {r_squared_tau:.2f}\nslope: {slope_tau:.2f}',
                         transform=ax_tc.transAxes, verticalalignment='top', fontsize=42)
                plt.xlabel(r'true $\tau$', fontsize=56)
                plt.ylabel(r'learned $\tau$', fontsize=56)
                plt.xlim(-0.025, 0.5); plt.ylim(_tau_ylim_p)
                plt.xticks([0.0, 0.25, 0.5], ['0.0', '0.25', '0.5'], fontsize=51)
                plt.yticks([0.0, 0.25, 0.5], ['0.0', '0.25', '0.5'], fontsize=51)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/tau_comparison_cell_type_{config_indices}.png', dpi=300)
                plt.close()

                # tau_comparison_wo_outliers — outliers in red, R²/slope on inliers.
                if int(_inlier_mask_t.sum()) >= 2:
                    r2_tau_clean, slope_tau_clean = compute_r_squared(
                        _gt_t[_inlier_mask_t], _lrn_t[_inlier_mask_t])
                else:
                    r2_tau_clean, slope_tau_clean = float('nan'), float('nan')
                fig = plt.figure(figsize=(10, 9))
                plt.scatter(_gt_t[_inlier_mask_t], _lrn_t[_inlier_mask_t],
                            c=mc, s=_dot_s, alpha=_dot_alpha)
                if n_outliers_tau > 0:
                    plt.scatter(_gt_t[_outlier_mask_t], _lrn_t[_outlier_mask_t],
                                c='red', s=6, alpha=0.7)
                _tlin = np.linspace(-0.025, 0.5, 2)
                plt.plot(_tlin, _tlin, '--', color='gray', linewidth=1, alpha=0.6)
                plt.plot(_tlin, _tlin + _tau_outlier_thresh, ':', color='gray', linewidth=1, alpha=0.5)
                plt.plot(_tlin, _tlin - _tau_outlier_thresh, ':', color='gray', linewidth=1, alpha=0.5)
                _pct_outliers_tau = (100.0 * n_outliers_tau / _gt_t.size) if _gt_t.size else 0.0
                _ax_tc3 = plt.gca()
                _ax_tc3.text(0.05, 0.95,
                             f'R²: {r2_tau_clean:.2f} ({r_squared_tau:.2f})\n'
                             f'slope: {slope_tau_clean:.2f}',
                             transform=_ax_tc3.transAxes, verticalalignment='top', fontsize=32)
                _ax_tc3.text(0.05, 0.78,
                             f'outliers: {_pct_outliers_tau:.1f}%',
                             transform=_ax_tc3.transAxes, verticalalignment='top', fontsize=32)
                plt.xlabel(r'true $\tau$', fontsize=56)
                plt.ylabel(r'learned $\tau$', fontsize=56)
                plt.xlim(-0.025, 0.5); plt.ylim(_tau_ylim_p)
                plt.xticks([0.0, 0.25, 0.5], ['0.0', '0.25', '0.5'], fontsize=51)
                plt.yticks([0.0, 0.25, 0.5], ['0.0', '0.25', '0.5'], fontsize=51)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/tau_comparison_wo_outliers_{config_indices}.png', dpi=300)
                plt.close()

            gt_vrest_np = ode_params.gt_vrest(n_neurons)
            learned_V_rest = ode_params.derive_vrest(slopes_f_theta_array, offsets_array, n_neurons)
            r_squared_V_rest = 0.0
            slope_V_rest = 0.0

            if ode_params.has_vrest() and gt_vrest_np is not None:
                # Square fixed window matching the x-axis so the identity line
                # is the diagonal across the whole panel.
                _v_ylim_p = (-0.025, 1.0)

                fig = plt.figure(figsize=(10, 9))
                plt.scatter(gt_vrest_np, learned_V_rest, c=mc, s=_dot_s, alpha=_dot_alpha)
                r_squared_V_rest, slope_V_rest = compute_r_squared(gt_vrest_np, learned_V_rest)
                plt.text(0.05, 0.95, f'R²: {r_squared_V_rest:.2f}\nslope: {slope_V_rest:.2f}',
                         transform=plt.gca().transAxes, verticalalignment='top', fontsize=42)
                plt.xlabel(r'true $V_{rest}$', fontsize=56)
                plt.ylabel(r'learned $V_{rest}$', fontsize=56)
                plt.xlim(-0.025, 1.0); plt.ylim(_v_ylim_p)
                plt.xticks([0.0, 0.5, 1.0], ['0.0', '0.5', '1.0'], fontsize=51)
                plt.yticks([0.0, 0.5, 1.0], ['0.0', '0.5', '1.0'], fontsize=51)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/V_rest_comparison_{config_indices}.png', dpi=300)
                plt.close()

                # Outlier mask on |learned - true| > 0.2, shared by both extra plots.
                _gt_v   = np.asarray(gt_vrest_np).ravel()
                _lrn_v  = np.asarray(learned_V_rest).ravel()
                _vrest_outlier_thresh = 0.1
                _outlier_mask_v = np.abs(_lrn_v - _gt_v) > _vrest_outlier_thresh
                _inlier_mask_v  = ~_outlier_mask_v
                n_outliers_v = int(_outlier_mask_v.sum())
                _type_list_np_v = to_numpy(type_list).astype(int).ravel()

                # V_rest_comparison_cell_type — colored by cell type, outlier clusters labelled.
                fig = plt.figure(figsize=(10, 9))
                ax_vc = plt.gca()
                plt.scatter(gt_vrest_np, learned_V_rest,
                            c=cmap.color(_type_list_np_v),
                            s=4, alpha=0.6)
                for _t in np.unique(_type_list_np_v):
                    _t_mask = (_type_list_np_v == _t) & _outlier_mask_v
                    if not _t_mask.any():
                        continue
                    _x_lbl = float(_gt_v[_t_mask].mean())
                    _y_lbl = float(_lrn_v[_t_mask].mean())
                    ax_vc.text(_x_lbl, _y_lbl,
                               index_to_name.get(int(_t), f'Type{int(_t)}'),
                               fontsize=11, ha='center', va='center',
                               color='black', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                         edgecolor='gray', alpha=0.75, linewidth=0.5))
                plt.text(0.05, 0.95, f'R²: {r_squared_V_rest:.2f}\nslope: {slope_V_rest:.2f}',
                         transform=ax_vc.transAxes, verticalalignment='top', fontsize=42)
                plt.xlabel(r'true $V_{rest}$', fontsize=56)
                plt.ylabel(r'learned $V_{rest}$', fontsize=56)
                plt.xlim(-0.025, 1.0); plt.ylim(_v_ylim_p)
                plt.xticks([0.0, 0.5, 1.0], ['0.0', '0.5', '1.0'], fontsize=51)
                plt.yticks([0.0, 0.5, 1.0], ['0.0', '0.5', '1.0'], fontsize=51)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/V_rest_comparison_cell_type_{config_indices}.png', dpi=300)
                plt.close()

                # V_rest_comparison_wo_outliers — outliers in red, R²/slope on inliers.
                if int(_inlier_mask_v.sum()) >= 2:
                    r2_v_clean, slope_v_clean = compute_r_squared(
                        _gt_v[_inlier_mask_v], _lrn_v[_inlier_mask_v])
                else:
                    r2_v_clean, slope_v_clean = float('nan'), float('nan')
                fig = plt.figure(figsize=(10, 9))
                plt.scatter(_gt_v[_inlier_mask_v], _lrn_v[_inlier_mask_v],
                            c=mc, s=_dot_s, alpha=_dot_alpha)
                if n_outliers_v > 0:
                    plt.scatter(_gt_v[_outlier_mask_v], _lrn_v[_outlier_mask_v],
                                c='red', s=6, alpha=0.7)
                _vlin = np.linspace(-0.025, 1.0, 2)
                plt.plot(_vlin, _vlin, '--', color='gray', linewidth=1, alpha=0.6)
                plt.plot(_vlin, _vlin + _vrest_outlier_thresh, ':', color='gray', linewidth=1, alpha=0.5)
                plt.plot(_vlin, _vlin - _vrest_outlier_thresh, ':', color='gray', linewidth=1, alpha=0.5)
                _pct_outliers_v = (100.0 * n_outliers_v / _gt_v.size) if _gt_v.size else 0.0
                _ax_vc3 = plt.gca()
                _ax_vc3.text(0.05, 0.95,
                             f'R²: {r2_v_clean:.2f} ({r_squared_V_rest:.2f})\n'
                             f'slope: {slope_v_clean:.2f}',
                             transform=_ax_vc3.transAxes, verticalalignment='top', fontsize=32)
                _ax_vc3.text(0.05, 0.78,
                             f'outliers: {_pct_outliers_v:.1f}%',
                             transform=_ax_vc3.transAxes, verticalalignment='top', fontsize=32)
                plt.xlabel(r'true $V_{rest}$', fontsize=56)
                plt.ylabel(r'learned $V_{rest}$', fontsize=56)
                plt.xlim(-0.025, 1.0); plt.ylim(_v_ylim_p)
                plt.xticks([0.0, 0.5, 1.0], ['0.0', '0.5', '1.0'], fontsize=51)
                plt.yticks([0.0, 0.5, 1.0], ['0.0', '0.5', '1.0'], fontsize=51)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/V_rest_comparison_wo_outliers_{config_indices}.png', dpi=300)
                plt.close()

            # f_theta derived params plot — panels depend on model
            param_names = ode_params.f_theta_param_names()
            n_panels = max(1, len(param_names))
            _param_arrays = []
            if ode_params.has_tau():
                _param_arrays.append((r'$\tau_i$', learned_tau))
            if ode_params.has_vrest():
                _param_arrays.append((r'$V^{\mathrm{rest}}_i$', learned_V_rest))

            if _param_arrays:
                # Build GT arrays for overlay
                _gt_arrays = []
                if ode_params.has_tau() and gt_taus_np is not None:
                    _gt_arrays.append(gt_taus_np)
                else:
                    _gt_arrays.append(None)
                if ode_params.has_vrest() and gt_vrest_np is not None:
                    _gt_arrays.append(gt_vrest_np)
                else:
                    _gt_arrays.append(None)
                _gt_arrays = _gt_arrays[:len(_param_arrays)]

                fig = plt.figure(figsize=(10, 4.5 * len(_param_arrays)))
                for pi, (plabel, pdata) in enumerate(_param_arrays):
                    ax = plt.subplot(len(_param_arrays), 1, pi + 1)
                    # GT as grey crosses if available
                    if pi < len(_gt_arrays) and _gt_arrays[pi] is not None:
                        plt.scatter(np.arange(n_neurons), _gt_arrays[pi],
                                    c='grey', s=_dot_s * 1.5, alpha=0.4, marker='x', label='ground truth')
                    plt.scatter(np.arange(n_neurons), pdata,
                                c=cmap.color(to_numpy(type_list).astype(int)), s=_dot_s, alpha=_dot_alpha, label='learned')
                    plt.ylabel(plabel, fontsize=48)
                    if pi == 0:
                        plt.legend(fontsize=14, loc='upper right')
                    if pi < len(_param_arrays) - 1:
                        plt.xticks([])
                    else:
                        plt.xlabel('neuron index', fontsize=48)
                        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
                        plt.xticks(fontsize=24)
                    plt.yticks(fontsize=24)
                plt.tight_layout()
                plt.savefig(f"{log_dir}/results/f_theta_{config_indices}_params.png", dpi=300)
                plt.close()


            # Plot 4: Weight comparison using model.W and gt_weights
            # Check Dale's Law for learned weights
            # dale_results = check_dales_law(
            #     edges=edges,
            #     weights=model.W,
            #     type_list=type_list,
            #     n_neurons=n_neurons,
            #     verbose=False,
            #     logger=None
            # )

            fig = plt.figure(figsize=(10, 9))
            learned_weights = to_numpy(get_model_W(model).squeeze())
            true_weights = to_numpy(gt_weights)
            # R² over ALL edges (including those touching hidden neurons).
            _tw = true_weights
            _lw = learned_weights
            _edge_s = max(0.1, min(10, 2000 / max(len(_tw), 1)))
            _edge_alpha = max(0.05, min(0.8, 500 / max(len(_tw), 1)))
            plt.scatter(_tw, _lw, c=mc, s=_edge_s, alpha=_edge_alpha)
            r_squared, slope_raw = compute_r_squared(_tw, _lw)
            plt.text(0.05, 0.95, f'R²: {r_squared:.3f}\nslope: {slope_raw:.2f}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=24)

            # Add Dale's Law statistics
            # dale_text = (f"excitatory neurons (all W>0): {dale_results['n_excitatory']} "
            #              f"({100*dale_results['n_excitatory']/n_neurons:.1f}%)\n"
            #              f"inhibitory neurons (all W<0): {dale_results['n_inhibitory']} "
            #              f"({100*dale_results['n_inhibitory']/n_neurons:.1f}%)\n"
            #              f"mixed/zero neurons (violates Dale's Law): {dale_results['n_mixed']} "
            #              f"({100*dale_results['n_mixed']/n_neurons:.1f}%)")
            # plt.text(0.05, 0.05, dale_text, transform=plt.gca().transAxes,
            #          verticalalignment='bottom', fontsize=10)

            plt.xlabel(r'true $W_{ij}$', fontsize=48)
            plt.ylabel(r'learned $W_{ij}$', fontsize=48)
            plt.xticks(fontsize = 24)
            plt.yticks(fontsize = 24)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/weights_comparison_raw.png', dpi=300)
            plt.close()
            raw_W_r2 = r_squared
            logger.info(f"raw W R²: {r_squared:.2f}  slope: {np.round(slope_raw, 4)}")

            # Corrected weights via metrics pipeline (replaces inline DataLoader +
            # gradient computation + correction formula — see metrics.py)
            corrected_W, ret_slopes_f, ret_slopes_g, ret_offsets, g_phi_fitted = compute_all_corrected_weights(
                model, config, edges, x_ts, device, n_grad_frames=8, ode_params=ode_params)
            torch.save(corrected_W, f'{log_dir}/results/corrected_W.pt')

            learned_weights = to_numpy(corrected_W.squeeze())

            # Mirror _plot_synaptic_linear's learned_ode_params.pt schema so
            # downstream tooling (e.g. fig_clustering_appendix.py) can load
            # (tau, V_rest, W, a) without re-running data_plot. `a` is the
            # GNN-only 2-D learned embedding; absent in the linear/known_ode
            # save schema.
            torch.save({
                'tau_i':    torch.from_numpy(np.asarray(learned_tau).astype(np.float32)),
                'V_i_rest': torch.from_numpy(np.asarray(learned_V_rest).astype(np.float32)),
                'W':        torch.from_numpy(learned_weights.astype(np.float32)),
                'a':        torch.from_numpy(to_numpy(model.a).astype(np.float32)),
            }, f'{log_dir}/results/learned_ode_params.pt')
            # Use effective true weights (includes g_phi gain for models like CX)
            true_weights = ode_params.effective_true_weights(
                to_numpy(gt_weights), to_numpy(edges), n_neurons)

            # Corrected R² over ALL edges (including those touching hidden neurons).
            _tw_c = true_weights
            _lw_c = learned_weights

            # Outlier removal + R² via metrics
            r_squared, slope_corrected, mask = compute_r_squared_filtered(
                _tw_c, _lw_c, outlier_threshold=5.0)
            residuals = _lw_c - _tw_c
            true_in = _tw_c[mask]
            learned_in = _lw_c[mask]

            if extended:
                # Partial correction (without g_phi factor) for diagnostic plot
                n_w = model.n_edges + model.n_extra_null_edges
                prior_ids = edges[0, :] % n_w
                slopes_g_t = torch.tensor(ret_slopes_g, dtype=torch.float32, device=device)
                corrected_W_ = corrected_W / slopes_g_t[prior_ids].unsqueeze(1)
                corrected_W_ = torch.nan_to_num(corrected_W_, nan=0.0, posinf=0.0, neginf=0.0)

                learned_in_ = to_numpy(corrected_W_.squeeze())
                learned_in_ = learned_in_[mask]

                fig = plt.figure(figsize=(10, 9))
                plt.scatter(true_in, learned_in_, c=mc, s=_edge_s, alpha=_edge_alpha)
                r_squared_rj, slope_rj = compute_r_squared(true_in, learned_in_)
                plt.text(0.05, 0.95,
                        f'R²: {r_squared_rj:.3f}\nslope: {slope_rj:.2f}',
                        transform=plt.gca().transAxes, verticalalignment='top', fontsize=24)
                plt.xlabel(r'true $W_{ij}$', fontsize=48)
                plt.ylabel(r'learned $W_{ij}r_j$', fontsize=48)
                plt.xticks(fontsize = 24)
                plt.yticks(fontsize = 24)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/weights_comparison_rj.png', dpi=300)
                plt.close()

            fig = plt.figure(figsize=(10, 9))
            plt.scatter(true_in, learned_in, c=mc, s=2.0, alpha=0.3,
                        edgecolors='none')
            plt.text(0.05, 0.95,
                     f'R²: {r_squared:.2f}\nslope: {slope_corrected:.2f}',
                     transform=plt.gca().transAxes, verticalalignment='top', fontsize=42)

            plt.xlabel(r'true $W_{ij}$', fontsize=56)
            plt.ylabel(r'learned $W_{ij}^*$', fontsize=56)
            plt.xlim([-1, 2])
            plt.ylim([-1, 2])
            plt.xticks([-1, 0.5, 2], ['-1', '0.5', '2'], fontsize=51)
            plt.yticks([-1, 0.5, 2], ['-1', '0.5', '2'], fontsize=51)
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/weights_comparison_corrected.png', dpi=300)
            plt.close()

            # weights_comparison_corrected_true_tau — recompute corrected W
            # using ground-truth τ instead of the f_theta-derived slope.
            # Derivation: slopes_f_theta = -1/τ, and
            #   corrected_W = -W / slopes_f_theta * grad_msg * slopes_g_phi
            # So replacing learned slopes with -1/τ_true is equivalent to
            # rescaling the existing corrected_W by (s_learned / s_true)
            # = (τ_true / τ_learned_raw) per post-synaptic neuron.
            if gt_taus_np is not None:
                n_w_tt = model.n_edges + model.n_extra_null_edges
                target_ids_tt = edges[1, :] % n_w_tt

                gt_tau_arr = np.asarray(gt_taus_np, dtype=np.float32).ravel()
                with np.errstate(divide='ignore', invalid='ignore'):
                    slopes_f_true = np.where(gt_tau_arr > 0, -1.0 / gt_tau_arr, 0.0)
                slopes_f_learned = np.asarray(ret_slopes_f, dtype=np.float32).ravel()

                slopes_f_true_t = torch.tensor(slopes_f_true, dtype=torch.float32, device=device)
                slopes_f_learned_t = torch.tensor(slopes_f_learned, dtype=torch.float32, device=device)

                s_learned_edge = slopes_f_learned_t[target_ids_tt]
                s_true_edge = slopes_f_true_t[target_ids_tt]
                # Avoid div-by-zero when gt_tau is 0/missing (factor → 0 there).
                scale_edge = torch.where(
                    s_true_edge != 0,
                    s_learned_edge / s_true_edge,
                    torch.zeros_like(s_true_edge))
                corrected_W_true_tau = corrected_W * scale_edge.unsqueeze(1)
                corrected_W_true_tau = torch.nan_to_num(
                    corrected_W_true_tau, nan=0.0, posinf=0.0, neginf=0.0)

                learned_weights_tt = to_numpy(corrected_W_true_tau.squeeze())
                r_squared_tt, slope_tt, mask_tt = compute_r_squared_filtered(
                    _tw_c, learned_weights_tt, outlier_threshold=5.0)
                true_in_tt = _tw_c[mask_tt]
                learned_in_tt = learned_weights_tt[mask_tt]

                fig = plt.figure(figsize=(10, 9))
                plt.scatter(true_in_tt, learned_in_tt, c=mc, s=2.0, alpha=0.3,
                            edgecolors='none')
                plt.text(0.05, 0.95,
                         f'R²: {r_squared_tt:.2f}\nslope: {slope_tt:.2f}',
                         transform=plt.gca().transAxes, verticalalignment='top', fontsize=42)
                plt.xlabel(r'true $W_{ij}$', fontsize=56)
                plt.ylabel(r'learned $W_{ij}^*$ (true $\tau$)', fontsize=56)
                plt.xlim([-1, 2])
                plt.ylim([-1, 2])
                plt.xticks([-1, 0.5, 2], ['-1', '0.5', '2'], fontsize=51)
                plt.yticks([-1, 0.5, 2], ['-1', '0.5', '2'], fontsize=51)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/weights_comparison_corrected_true_tau.png', dpi=300)
                plt.close()
                logger.info(f"corrected W (true τ) R²: {r_squared_tt:.2f}  slope: {np.round(slope_tt, 4)}")

            # ── Consolidated panel-data dump (npz) ────────────────────────
            # Saved arrays let figure scripts re-render the six parameter
            # panels without re-running the (slow) model evaluation.
            # `_maybe` defensively skips any entry whose source variable
            # isn't defined yet at this point of plot_synaptic (some are
            # set later in the function), so the dump degrades gracefully
            # rather than crashing on a NameError or None.attribute.
            _panel_npz = f'{log_dir}/results/panels_{config_indices}.npz'
            _panel_data = {}

            def _maybe(key, fn):
                try:
                    _panel_data[key] = fn()
                except (NameError, AttributeError, TypeError):
                    pass

            _maybe('config_indices',   lambda: np.asarray(config_indices))
            _maybe('n_neurons',        lambda: np.int64(n_neurons))
            _maybe('type_ids',         lambda: type_np.astype(np.int32))
            # W panel (corrected — what the figure script displays)
            _maybe('W_true',           lambda: np.asarray(true_in, dtype=np.float32))
            _maybe('W_learned',        lambda: np.asarray(learned_in, dtype=np.float32))
            _maybe('W_r2',             lambda: np.float32(r_squared))
            _maybe('W_slope',          lambda: np.float32(slope_corrected))
            # Embedding
            _maybe('a',                lambda: to_numpy(model.a).astype(np.float32))
            _maybe('cluster_accuracy', lambda: np.float32(cluster_acc))
            # g_phi (curves + scatter share these arrays)
            _maybe('g_phi_domain',     lambda: rr_np.astype(np.float32))
            _maybe('g_phi_learned',    lambda: func_np.astype(np.float32))
            _maybe('g_phi_true',       lambda: func_true_g_phi.astype(np.float32))
            # f_theta (curves + scatter share these arrays)
            _maybe('f_theta_domain',   lambda: rr_domain_phi_np.astype(np.float32))
            _maybe('f_theta_learned',  lambda: to_numpy(func_domain_phi).astype(np.float32))
            _maybe('f_theta_true',     lambda: func_true_f_theta.astype(np.float32))
            # tau / V_rest
            _maybe('tau_true',         lambda: gt_taus_np.astype(np.float32))
            _maybe('tau_learned',      lambda: learned_tau.astype(np.float32))
            _maybe('tau_r2',           lambda: np.float32(r_squared_tau))
            _maybe('tau_slope',        lambda: np.float32(slope_tau))
            _maybe('V_rest_true',      lambda: gt_vrest_np.astype(np.float32))
            _maybe('V_rest_learned',   lambda: learned_V_rest.astype(np.float32))
            _maybe('V_rest_r2',        lambda: np.float32(r_squared_V_rest))
            _maybe('V_rest_slope',     lambda: np.float32(slope_V_rest))

            if _panel_data:
                np.savez_compressed(_panel_npz, **_panel_data)
                logger.info(f'saved panel data → {_panel_npz} ({len(_panel_data)} arrays)')

            print(f"weights R²: {_r2_color(r_squared)}{r_squared:.4f}{_ANSI_RESET}  slope: {np.round(slope_corrected, 4)}")
            logger.info(f"weights R²: {r_squared:.4f}  slope: {np.round(slope_corrected, 4)}")
            # Relative error |learned - true| / max(|true|, eps), full sample.
            # Mean ± SD intentionally omitted (heavy tails dominate); median + IQR only.
            _gt_w_arr  = np.asarray(_tw_c).ravel()
            _lrn_w_arr = np.asarray(_lw_c).ravel()
            _rel_err_w = np.abs(_lrn_w_arr - _gt_w_arr) / np.maximum(np.abs(_gt_w_arr), 1e-6)
            _rel_err_w_med  = float(np.median(_rel_err_w))
            _q1_w_re, _q3_w_re = np.percentile(_rel_err_w, [25.0, 75.0])
            _rel_err_w_iqr  = float(_q3_w_re - _q1_w_re)
            print(f"W rel.err: {_ANSI_WHITE}median {100*_rel_err_w_med:.1f}%  IQR {100*_rel_err_w_iqr:.1f}%{_ANSI_RESET}")
            logger.info(f"W rel.err: median {100*_rel_err_w_med:.2f}%  IQR {100*_rel_err_w_iqr:.2f}%")
            with open(os.path.join(log_dir, 'results', 'metrics.txt'), 'a') as _mf:
                _mf.write(f"W_rel_err_median: {_rel_err_w_med:.4f}\n")
                _mf.write(f"W_rel_err_iqr: {_rel_err_w_iqr:.4f}\n")
            _rel_err_tau_med = _rel_err_tau_iqr = None
            _rel_err_v_med = _rel_err_v_iqr = None

            # R² on only real (non-null) edges
            connectivity_r2_real = None
            if hasattr(model, 'n_extra_null_edges') and model.n_extra_null_edges > 0:
                n_real = model.n_edges
                try:
                    r2_real, _ = compute_r_squared(true_weights[:n_real], learned_weights[:n_real])
                    connectivity_r2_real = r2_real
                    print(f"connectivity R² (real edges only): {_r2_color(r2_real)}{r2_real:.4f}{_ANSI_RESET}")
                    logger.info(f"connectivity R² (real edges only): {r2_real:.4f}")
                except Exception:
                    pass
            logger.info(f'median residuals: {np.median(residuals):.4f}')
            inlier_residuals = residuals[mask]
            logger.info(f'inliers: {len(inlier_residuals)}  mean residual: {np.mean(inlier_residuals):.4f}  std: {np.std(inlier_residuals):.4f}  min,max: {np.min(inlier_residuals):.4f}, {np.max(inlier_residuals):.4f}')
            outlier_residuals = residuals[~mask]
            if len(outlier_residuals) > 0:
                logger.info(
                    f'outliers: {len(outlier_residuals)}  mean residual: {np.mean(outlier_residuals):.4f}  std: {np.std(outlier_residuals):.4f}  min,max: {np.min(outlier_residuals):.4f}, {np.max(outlier_residuals):.4f}')
            else:
                logger.info('outliers: 0  (no outliers detected)')
            if ode_params.has_tau():
                print(f"tau R²: {_r2_color(r_squared_tau)}{r_squared_tau:.3f}{_ANSI_RESET}  slope: {slope_tau:.2f}")
                logger.info(f"tau R²: {r_squared_tau:.3f}  slope: {slope_tau:.2f}")
                # Relative error |learned - true| / max(|true|, eps), full sample.
                _gt_t_arr  = np.asarray(gt_taus_np).ravel()
                _lrn_t_arr = np.asarray(learned_tau).ravel()
                _rel_err_tau = np.abs(_lrn_t_arr - _gt_t_arr) / np.maximum(np.abs(_gt_t_arr), 1e-6)
                _rel_err_tau_med  = float(np.median(_rel_err_tau))
                _q1_t_re, _q3_t_re = np.percentile(_rel_err_tau, [25.0, 75.0])
                _rel_err_tau_iqr  = float(_q3_t_re - _q1_t_re)
                print(f"tau rel.err: {_ANSI_WHITE}median {100*_rel_err_tau_med:.1f}%  IQR {100*_rel_err_tau_iqr:.1f}%{_ANSI_RESET}")
                logger.info(f"tau rel.err: median {100*_rel_err_tau_med:.2f}%  IQR {100*_rel_err_tau_iqr:.2f}%")
                with open(os.path.join(log_dir, 'results', 'metrics.txt'), 'a') as _mf:
                    _mf.write(f"tau_rel_err_median: {_rel_err_tau_med:.4f}\n")
                    _mf.write(f"tau_rel_err_iqr: {_rel_err_tau_iqr:.4f}\n")
                # Outlier count + trimmed R²/slope (|learned - true| > 0.1).
                _tau_outlier_thresh_g = 0.1
                _outlier_mask_t_g = np.abs(_lrn_t_arr - _gt_t_arr) > _tau_outlier_thresh_g
                _inlier_mask_t_g  = ~_outlier_mask_t_g
                n_outliers_tau_g = int(_outlier_mask_t_g.sum())
                _pct_outliers_tau_g = (100.0 * n_outliers_tau_g / _gt_t_arr.size) if _gt_t_arr.size else 0.0
                if int(_inlier_mask_t_g.sum()) >= 2:
                    r2_tau_clean_g, slope_tau_clean_g = compute_r_squared(
                        _gt_t_arr[_inlier_mask_t_g], _lrn_t_arr[_inlier_mask_t_g])
                else:
                    r2_tau_clean_g, slope_tau_clean_g = float('nan'), float('nan')
                print(f"tau (wo outliers) R²: {_r2_color(r2_tau_clean_g)}{r2_tau_clean_g:.3f}{_ANSI_RESET}  "
                      f"slope: {slope_tau_clean_g:.2f}  "
                      f"outliers: {n_outliers_tau_g}/{_gt_t_arr.size} ({_pct_outliers_tau_g:.1f}%)")
                logger.info(f"tau_wo_outliers R²: {r2_tau_clean_g:.4f}  slope: {slope_tau_clean_g:.4f}  "
                            f"outliers: {n_outliers_tau_g}/{_gt_t_arr.size} ({_pct_outliers_tau_g:.1f}%)")
                with open(os.path.join(log_dir, 'results', 'metrics.txt'), 'a') as _mf:
                    _mf.write(f"tau_no_outliers_R2: {r2_tau_clean_g:.4f}\n")
                    _mf.write(f"tau_no_outliers_slope: {slope_tau_clean_g:.4f}\n")
                    _mf.write(f"tau_n_outliers: {n_outliers_tau_g}\n")
            if ode_params.has_vrest():
                print(f"V_rest R²: {_r2_color(r_squared_V_rest)}{r_squared_V_rest:.3f}{_ANSI_RESET}  slope: {slope_V_rest:.2f}")
                logger.info(f"V_rest R²: {r_squared_V_rest:.3f}  slope: {slope_V_rest:.2f}")
                # Relative error |learned - true| / max(|true|, eps), full sample.
                # Mean ± SD intentionally omitted (heavy tails dominate); median + IQR only.
                _gt_v_arr  = np.asarray(gt_vrest_np).ravel()
                _lrn_v_arr = np.asarray(learned_V_rest).ravel()
                _rel_err_v = np.abs(_lrn_v_arr - _gt_v_arr) / np.maximum(np.abs(_gt_v_arr), 1e-6)
                _rel_err_v_med  = float(np.median(_rel_err_v))
                _q1_v_re, _q3_v_re = np.percentile(_rel_err_v, [25.0, 75.0])
                _rel_err_v_iqr  = float(_q3_v_re - _q1_v_re)
                print(f"V_rest rel.err: {_ANSI_WHITE}median {100*_rel_err_v_med:.1f}%  IQR {100*_rel_err_v_iqr:.1f}%{_ANSI_RESET}")
                logger.info(f"V_rest rel.err: median {100*_rel_err_v_med:.2f}%  IQR {100*_rel_err_v_iqr:.2f}%")
                with open(os.path.join(log_dir, 'results', 'metrics.txt'), 'a') as _mf:
                    _mf.write(f"V_rest_rel_err_median: {_rel_err_v_med:.4f}\n")
                    _mf.write(f"V_rest_rel_err_iqr: {_rel_err_v_iqr:.4f}\n")
                # Outlier count + trimmed R²/slope (|learned - true| > 0.2).
                _vrest_outlier_thresh_g = 0.2
                _outlier_mask_v_g = np.abs(_lrn_v_arr - _gt_v_arr) > _vrest_outlier_thresh_g
                _inlier_mask_v_g  = ~_outlier_mask_v_g
                n_outliers_v_g = int(_outlier_mask_v_g.sum())
                _pct_outliers_v_g = (100.0 * n_outliers_v_g / _gt_v_arr.size) if _gt_v_arr.size else 0.0
                if int(_inlier_mask_v_g.sum()) >= 2:
                    r2_v_clean_g, slope_v_clean_g = compute_r_squared(
                        _gt_v_arr[_inlier_mask_v_g], _lrn_v_arr[_inlier_mask_v_g])
                else:
                    r2_v_clean_g, slope_v_clean_g = float('nan'), float('nan')
                print(f"V_rest (wo outliers) R²: {_r2_color(r2_v_clean_g)}{r2_v_clean_g:.3f}{_ANSI_RESET}  "
                      f"slope: {slope_v_clean_g:.2f}  "
                      f"outliers: {n_outliers_v_g}/{_gt_v_arr.size} ({_pct_outliers_v_g:.1f}%)")
                logger.info(f"V_rest_wo_outliers R²: {r2_v_clean_g:.4f}  slope: {slope_v_clean_g:.4f}  "
                            f"outliers: {n_outliers_v_g}/{_gt_v_arr.size} ({_pct_outliers_v_g:.1f}%)")
                with open(os.path.join(log_dir, 'results', 'metrics.txt'), 'a') as _mf:
                    _mf.write(f"V_rest_no_outliers_R2: {r2_v_clean_g:.4f}\n")
                    _mf.write(f"V_rest_no_outliers_slope: {slope_v_clean_g:.4f}\n")
                    _mf.write(f"V_rest_n_outliers: {n_outliers_v_g}\n")
            _summary_parts = [f"W rel err {100*_rel_err_w_med:.1f}±{100*_rel_err_w_iqr:.1f}%"]
            if _rel_err_tau_med is not None:
                _summary_parts.append(f"tau rel err {100*_rel_err_tau_med:.1f}±{100*_rel_err_tau_iqr:.1f}%")
            if _rel_err_v_med is not None:
                _summary_parts.append(f"Vrest rel err {100*_rel_err_v_med:.1f}±{100*_rel_err_v_iqr:.1f}%")
            logger.info(f"SUMMARY: {', '.join(_summary_parts)}")
            print(f"f_theta Pearson r²: {_r2_color(r2_f_theta_mean)}{r2_f_theta_mean:.3f}{_ANSI_RESET}  median={r2_f_theta_median:.3f}")
            print(f"g_phi Pearson r²: {_r2_color(r2_g_phi_mean)}{r2_g_phi_mean:.3f}{_ANSI_RESET}  median={r2_g_phi_median:.3f}")
            logger.info(f"f_theta Pearson r²: mean={r2_f_theta_mean:.3f}  median={r2_f_theta_median:.3f}")
            logger.info(f"g_phi Pearson r²: mean={r2_g_phi_mean:.3f}  median={r2_g_phi_median:.3f}")

            # g_phi parameter R² (model-specific: gain/bias for CX, slope for flyvis)
            gt_g_params = ode_params.gt_g_phi_params(n_neurons)
            if gt_g_params is not None and g_phi_fitted is not None:
                for pname in ode_params.g_phi_param_names():
                    if pname in g_phi_fitted and pname in gt_g_params:
                        gt_vals = gt_g_params[pname]
                        learned_vals = g_phi_fitted[pname][:n_neurons]
                        r2_p, slope_p = compute_r_squared(gt_vals, learned_vals)
                        print(f"g_phi {pname} R²: {_r2_color(r2_p)}{r2_p:.3f}{_ANSI_RESET}  slope: {slope_p:.2f}")
                        logger.info(f"g_phi {pname} R²: {r2_p:.3f}  slope: {slope_p:.2f}")

                        # Scatter plot for each g_phi parameter
                        fig = plt.figure(figsize=(10, 9))
                        plt.scatter(gt_vals, learned_vals, c=mc, s=_dot_s, alpha=_dot_alpha)
                        plt.text(0.05, 0.95, f'R²: {r2_p:.2f}\nslope: {slope_p:.2f}',
                                 transform=plt.gca().transAxes, verticalalignment='top', fontsize=32)
                        plt.xlabel(f'true {pname}', fontsize=48)
                        plt.ylabel(f'learned {pname}', fontsize=48)
                        plt.xticks(fontsize=24)
                        plt.yticks(fontsize=24)
                        plt.tight_layout()
                        plt.savefig(f'{log_dir}/results/g_phi_{pname}_comparison_{config_indices}.png', dpi=300)
                        plt.close()

            # Write to analysis log file for Claude
            if log_file:
                log_file.write(f"\n--- Parameter extraction results ---\n")
                log_file.write(f"raw_W_R2: {raw_W_r2:.4f}\n")
                log_file.write(f"connectivity_R2: {r_squared:.4f}\n")
                if connectivity_r2_real is not None:
                    log_file.write(f"connectivity_R2_real: {connectivity_r2_real:.4f}\n")
                log_file.write(f"f_theta_functional_R2: {r2_f_theta_mean:.4f}\n")
                log_file.write(f"g_phi_functional_R2: {r2_g_phi_mean:.4f}\n")
                if ode_params.has_tau():
                    log_file.write(f"tau_R2: {r_squared_tau:.4f}\n")
                if ode_params.has_vrest():
                    log_file.write(f"V_rest_R2: {r_squared_V_rest:.4f}\n")
                if gt_g_params is not None and g_phi_fitted is not None:
                    for pname in ode_params.g_phi_param_names():
                        if pname in g_phi_fitted and pname in gt_g_params:
                            gt_v = gt_g_params[pname]
                            lr_v = g_phi_fitted[pname][:n_neurons]
                            r2_p, _ = compute_r_squared(gt_v, lr_v)
                            log_file.write(f"g_phi_{pname}_R2: {r2_p:.4f}\n")


            # Plot connectivity matrix comparison (only for small networks)
            if n_neurons < 1000:
                print('plot true vs learned connectivity matrix ...')
                edges_np = to_numpy(edges)
                J_true = np.zeros((n_neurons, n_neurons), dtype=np.float32)
                J_true[edges_np[0], edges_np[1]] = true_weights.flatten()
                J_learned = np.zeros((n_neurons, n_neurons), dtype=np.float32)
                J_learned[edges_np[0], edges_np[1]] = to_numpy(corrected_W.squeeze()).flatten()
                nonzero = np.abs(true_weights.flatten())
                vmax = np.percentile(nonzero[nonzero > 0], 98) if np.any(nonzero > 0) else 1.0
                vmax = max(vmax, 1e-6)
                fig_mat, (ax_t, ax_l) = plt.subplots(1, 2, figsize=(14, 6))
                im_t = ax_t.imshow(J_true.T, cmap='bwr_r', vmin=-vmax, vmax=vmax,
                                   aspect='auto', interpolation='nearest', origin='upper')
                ax_t.set_title('True connectivity')
                fig_mat.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)
                im_l = ax_l.imshow(J_learned.T, cmap='bwr_r', vmin=-vmax, vmax=vmax,
                                   aspect='auto', interpolation='nearest', origin='upper')
                ax_l.set_title('Learned connectivity')
                fig_mat.colorbar(im_l, ax=ax_l, fraction=0.046, pad=0.04)
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/connectivity_matrix.png', dpi=200)
                plt.close(fig_mat)
                logger.info("saved connectivity_matrix.png")

                # Edge mask: binary adjacency (red=edge, white=no edge)
                edge_mask = np.zeros((n_neurons, n_neurons), dtype=np.float32)
                edge_mask[edges_np[0], edges_np[1]] = 1.0
                fig_mask, ax_mask = plt.subplots(1, 1, figsize=(7, 6))
                ax_mask.imshow(edge_mask.T, cmap='Reds', vmin=0, vmax=1,
                               aspect='auto', interpolation='nearest', origin='upper')
                n_edges_actual = int(edge_mask.sum())
                n_possible = n_neurons * (n_neurons - 1)
                density = n_edges_actual / n_possible * 100 if n_possible > 0 else 0
                ax_mask.set_title(f'Edge mask ({n_edges_actual} edges, {density:.1f}% density)')
                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/edge_mask.png', dpi=200)
                plt.close(fig_mask)
                logger.info("saved edge_mask.png")

                # Zebrafish: extra two-panel figure with full learned matrix + cropped/sorted
                if 'zebrafish_oculomotor' in config.dataset:
                    # J convention: J[post, pre] — transpose of W_dense[src, dst]
                    J_true_T = J_true.T
                    J_learned_T = J_learned.T
                    # Remove disconnected neurons (zeroed by final_adjustments)
                    has_conn = (np.abs(J_true_T).sum(axis=0) + np.abs(J_true_T).sum(axis=1)) > 0
                    J_true_active = J_true_T[has_conn, :][:, has_conn]
                    J_learned_active = J_learned_T[has_conn, :][:, has_conn]
                    # Sort by total outgoing weight of ground truth (column sum, strongest first)
                    col_sum = np.sum(J_true_active, axis=0)
                    sort_idx = np.argsort(col_sum)[::-1]
                    J_true_sorted = J_true_active[sort_idx, :][:, sort_idx]
                    J_learned_sorted = J_learned_active[sort_idx, :][:, sort_idx]

                    n_active = J_true_sorted.shape[0]
                    fig_zf, (ax_gt, ax_learned) = plt.subplots(1, 2, figsize=(14, 6))
                    im_gt = ax_gt.imshow(J_true_sorted, cmap='bwr_r', vmin=-vmax, vmax=vmax,
                                         aspect='auto', interpolation='nearest', origin='upper')
                    ax_gt.set_title(f'True sorted ({n_active} neurons)')
                    ax_gt.set_xlabel('presynaptic neuron')
                    ax_gt.set_ylabel('postsynaptic neuron')
                    fig_zf.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

                    im_learned = ax_learned.imshow(J_learned_sorted, cmap='bwr_r', vmin=-vmax, vmax=vmax,
                                                   aspect='auto', interpolation='nearest', origin='upper')
                    ax_learned.set_title(f'Learned sorted ({n_active} neurons)')
                    ax_learned.set_xlabel('presynaptic neuron')
                    ax_learned.set_ylabel('postsynaptic neuron')
                    fig_zf.colorbar(im_learned, ax=ax_learned, fraction=0.046, pad=0.04)

                    plt.tight_layout()
                    plt.savefig(f'{log_dir}/results/connectivity_matrix_zebrafish_sorted.png', dpi=200)
                    plt.close(fig_zf)
                    logger.info("saved connectivity_matrix_zebrafish_sorted.png")

            # eigenvalue and singular value analysis using sparse matrices
            if not skip_svd:
                print('plot eigenvalue spectrum and eigenvector comparison ...')

                # build sparse matrices for true and learned weights
                edges_np = to_numpy(edges)
                true_sparse = scipy.sparse.csr_matrix(
                    (true_weights.flatten(), (edges_np[1], edges_np[0])),
                    shape=(n_neurons, n_neurons)
                )
                learned_sparse = scipy.sparse.csr_matrix(
                    (to_numpy(corrected_W.squeeze().flatten()), (edges_np[1], edges_np[0])),
                    shape=(n_neurons, n_neurons)
                )

                # compute SVD using TruncatedSVD (for large sparse matrices)
                # 100 components captures dominant structure; 1000 was very slow for N>10000
                n_components = min(100, n_neurons - 1)
                svd_true = TruncatedSVD(n_components=n_components, random_state=42)
                svd_learned = TruncatedSVD(n_components=n_components, random_state=42)

                svd_true.fit(true_sparse)
                svd_learned.fit(learned_sparse)

                sv_true = svd_true.singular_values_
                sv_learned = svd_learned.singular_values_

                # get right singular vectors (V^T rows)
                V_true = svd_true.components_
                V_learned = svd_learned.components_

                # compute alignment matrix (right singular vectors V)
                alignment = np.abs(V_true @ V_learned.T)

                # compute left singular vectors U and their alignment
                n_show = min(100, n_components)
                U_true = svd_true.transform(true_sparse)[:, :n_show]
                U_learned = svd_learned.transform(learned_sparse)[:, :n_show]
                U_true = U_true / (np.linalg.norm(U_true, axis=0, keepdims=True) + 1e-10)
                U_learned = U_learned / (np.linalg.norm(U_learned, axis=0, keepdims=True) + 1e-10)
                alignment_L = np.abs(U_true.T @ U_learned)

                # --- Procrustes-aligned SVD decomposition (NeuralGraph-style) ---
                # Pick truncation rank from 90% and 99% variance thresholds
                cumvar = np.cumsum(sv_true**2) / np.sum(sv_true**2)
                rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
                rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
                rank_r = min(rank_99, n_components)  # use 99% rank for Procrustes

                # Full SVD on dense matrices for Procrustes (feasible for N < ~2000)
                W_true_dense = true_sparse.toarray()
                W_learned_dense = learned_sparse.toarray()

                try:
                    U_t, S_t, Vt_t = np.linalg.svd(W_true_dense, full_matrices=False)
                    U_l, S_l, Vt_l = np.linalg.svd(W_learned_dense, full_matrices=False)

                    # Truncate to rank_r
                    U_t_r = U_t[:, :rank_r] * S_t[:rank_r]     # N x r, scaled
                    V_t_r = Vt_t[:rank_r, :]                     # r x N
                    U_l_r = U_l[:, :rank_r] * S_l[:rank_r]
                    V_l_r = Vt_l[:rank_r, :]

                    # Procrustes alignment: find orthogonal R that minimizes ||A - B @ R||
                    # For U: align U_learned to U_true
                    from scipy.linalg import orthogonal_procrustes
                    R_U, _ = orthogonal_procrustes(U_l_r, U_t_r)
                    U_l_aligned = U_l_r @ R_U

                    # For V: align V_learned to V_true (transpose: work with r x N -> N x r)
                    R_V, _ = orthogonal_procrustes(V_l_r.T, V_t_r.T)
                    V_l_aligned = (V_l_r.T @ R_V).T  # back to r x N

                    # Compute R² for Procrustes-aligned factors
                    def _r2(y_true, y_pred):
                        ss_res = np.sum((y_true - y_pred)**2)
                        ss_tot = np.sum((y_true - y_true.mean())**2)
                        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

                    U_r2 = _r2(U_t_r.ravel(), U_l_aligned.ravel())
                    V_r2 = _r2(V_t_r.ravel(), V_l_aligned.ravel())

                    # Reconstruct W from aligned low-rank factors
                    W_recon_true = U_t_r @ V_t_r      # rank-r approx of true W
                    W_recon_learned = U_l_aligned @ V_l_aligned
                    W_recon_r2 = _r2(W_recon_true.ravel(), W_recon_learned.ravel())

                    # Also compute subspace R² at multiple ranks
                    rank_list = sorted(set([rank_90, rank_99, min(5, n_components),
                                            min(10, n_components), min(20, n_components),
                                            min(50, n_components)]))
                    rank_list = [r for r in rank_list if r <= n_components]
                    U_r2_per_rank = []
                    V_r2_per_rank = []
                    W_r2_per_rank = []
                    for rr in rank_list:
                        Ut_rr = U_t[:, :rr] * S_t[:rr]
                        Vt_rr = Vt_t[:rr, :]
                        Ul_rr = U_l[:, :rr] * S_l[:rr]
                        Vl_rr = Vt_l[:rr, :]
                        Ru, _ = orthogonal_procrustes(Ul_rr, Ut_rr)
                        Rv, _ = orthogonal_procrustes(Vl_rr.T, Vt_rr.T)
                        Ul_a = Ul_rr @ Ru
                        Vl_a = (Vl_rr.T @ Rv).T
                        U_r2_per_rank.append(_r2(Ut_rr.ravel(), Ul_a.ravel()))
                        V_r2_per_rank.append(_r2(Vt_rr.ravel(), Vl_a.ravel()))
                        W_r2_per_rank.append(_r2((Ut_rr @ Vt_rr).ravel(), (Ul_a @ Vl_a).ravel()))

                    # Extract rank-20 values for tracking across exploration iterations
                    # rank 20 chosen to match NeuralGraph ground truth, enabling cross-project comparison
                    _rank20 = min(20, n_components)
                    if _rank20 in rank_list:
                        _idx20 = rank_list.index(_rank20)
                        U_r2_rank20 = U_r2_per_rank[_idx20]
                        V_r2_rank20 = V_r2_per_rank[_idx20]
                        W_r2_rank20 = W_r2_per_rank[_idx20]
                    else:
                        # compute directly if not in sweep list
                        Ut_20 = U_t[:, :_rank20] * S_t[:_rank20]
                        Vt_20 = Vt_t[:_rank20, :]
                        Ul_20 = U_l[:, :_rank20] * S_l[:_rank20]
                        Vl_20 = Vt_l[:_rank20, :]
                        Ru20, _ = orthogonal_procrustes(Ul_20, Ut_20)
                        Rv20, _ = orthogonal_procrustes(Vl_20.T, Vt_20.T)
                        Ul_20a = Ul_20 @ Ru20
                        Vl_20a = (Vl_20.T @ Rv20).T
                        U_r2_rank20 = _r2(Ut_20.ravel(), Ul_20a.ravel())
                        V_r2_rank20 = _r2(Vt_20.ravel(), Vl_20a.ravel())
                        W_r2_rank20 = _r2((Ut_20 @ Vt_20).ravel(), (Ul_20a @ Vl_20a).ravel())

                    procrustes_ok = True
                except Exception as e:
                    logger.warning(f"Procrustes SVD analysis failed: {e}")
                    procrustes_ok = False

                # compute eigenvalues using sparse eigensolver for complex plane plot
                n_eigs = min(200, n_neurons - 2)
                eig_true = eig_learned = None
                try:
                    eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
                    eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')
                except Exception:
                    try:
                        n_eigs = min(50, n_neurons - 2)
                        if eig_true is None:
                            eig_true, _ = scipy.sparse.linalg.eigs(true_sparse.astype(np.float64), k=n_eigs, which='LM')
                        if eig_learned is None:
                            eig_learned, _ = scipy.sparse.linalg.eigs(learned_sparse.astype(np.float64), k=n_eigs, which='LM')
                    except Exception as e:
                        logger.warning(f"eigenvalue computation failed (learned W may be all zeros): {e}")
                        eig_true = eig_learned = None

                # create 3x3 figure
                fig, axes = plt.subplots(3, 3, figsize=(30, 30))

                # === Row 1: Spectral overview ===
                # (0,0) eigenvalues in complex plane
                if eig_true is not None and eig_learned is not None:
                    axes[0, 0].scatter(eig_true.real, eig_true.imag, s=100, c='green', alpha=0.7, label='true')
                    axes[0, 0].scatter(eig_learned.real, eig_learned.imag, s=100, c='black', alpha=0.7, label='learned')
                elif eig_true is not None:
                    axes[0, 0].scatter(eig_true.real, eig_true.imag, s=100, c='green', alpha=0.7, label='true')
                    axes[0, 0].text(0.5, 0.5, 'learned W ≈ 0\n(no eigenvalues)', transform=axes[0, 0].transAxes,
                                   ha='center', va='center', fontsize=20, color='red')
                else:
                    axes[0, 0].text(0.5, 0.5, 'eigenvalue computation failed', transform=axes[0, 0].transAxes,
                                   ha='center', va='center', fontsize=20, color='red')
                axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
                axes[0, 0].axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
                axes[0, 0].set_xlabel('real', fontsize=32)
                axes[0, 0].set_ylabel('imag', fontsize=32)
                axes[0, 0].legend(fontsize=20)
                axes[0, 0].tick_params(labelsize=20)
                axes[0, 0].set_title('eigenvalues in complex plane', fontsize=28)

                # (0,1) singular value scatter
                n_compare = min(len(sv_true), len(sv_learned))
                sv_r2 = _r2(sv_true[:n_compare], sv_learned[:n_compare]) if procrustes_ok else 0.0
                axes[0, 1].scatter(sv_true[:n_compare], sv_learned[:n_compare], s=100, c='black', edgecolors='black', alpha=0.7)
                max_val = max(sv_true.max(), sv_learned.max())
                axes[0, 1].plot([0, max_val], [0, max_val], 'g--', linewidth=2)
                axes[0, 1].set_xlabel('true singular value', fontsize=32)
                axes[0, 1].set_ylabel('learned singular value', fontsize=32)
                axes[0, 1].tick_params(labelsize=20)
                axes[0, 1].set_title(f'singular value comparison  R²={sv_r2:.3f}', fontsize=28)

                # (0,2) singular value spectrum (log scale) with rank markers
                axes[0, 2].plot(sv_true, color='green', linewidth=2, label='true')
                axes[0, 2].plot(sv_learned, color='black', linewidth=2, label='learned')
                axes[0, 2].axvline(x=rank_90, color='orange', linestyle='--', linewidth=1.5, label=f'rank@90%={rank_90}')
                axes[0, 2].axvline(x=rank_99, color='red', linestyle='--', linewidth=1.5, label=f'rank@99%={rank_99}')
                axes[0, 2].set_xlabel('index', fontsize=32)
                axes[0, 2].set_ylabel('singular value', fontsize=32)
                axes[0, 2].set_yscale('log')
                axes[0, 2].legend(fontsize=18)
                axes[0, 2].tick_params(labelsize=20)
                axes[0, 2].set_title('singular value spectrum (log scale)', fontsize=28)

                # === Row 2: Procrustes-aligned SVD (U, V, W_recon) ===
                if procrustes_ok:
                    # (1,0) U scatter: true vs learned (Procrustes-aligned)
                    u_flat_t = U_t_r.ravel()
                    u_flat_l = U_l_aligned.ravel()
                    axes[1, 0].scatter(u_flat_t, u_flat_l, s=1, c='black', alpha=0.15, rasterized=True)
                    umax = max(np.abs(u_flat_t).max(), np.abs(u_flat_l).max())
                    axes[1, 0].plot([-umax, umax], [-umax, umax], 'g--', linewidth=2)
                    axes[1, 0].set_xlabel('true U (output modes)', fontsize=28)
                    axes[1, 0].set_ylabel('learned U (Procrustes-aligned)', fontsize=28)
                    axes[1, 0].tick_params(labelsize=20)
                    axes[1, 0].set_title(f'U (rank={rank_r})  R²={U_r2:.3f}', fontsize=28)
                    axes[1, 0].set_aspect('equal')

                    # (1,1) V scatter: true vs learned (Procrustes-aligned)
                    v_flat_t = V_t_r.ravel()
                    v_flat_l = V_l_aligned.ravel()
                    axes[1, 1].scatter(v_flat_t, v_flat_l, s=1, c='black', alpha=0.15, rasterized=True)
                    vmax = max(np.abs(v_flat_t).max(), np.abs(v_flat_l).max())
                    axes[1, 1].plot([-vmax, vmax], [-vmax, vmax], 'g--', linewidth=2)
                    axes[1, 1].set_xlabel('true V (input selection)', fontsize=28)
                    axes[1, 1].set_ylabel('learned V (Procrustes-aligned)', fontsize=28)
                    axes[1, 1].tick_params(labelsize=20)
                    axes[1, 1].set_title(f'V (rank={rank_r})  R²={V_r2:.3f}', fontsize=28)
                    axes[1, 1].set_aspect('equal')

                    # (1,2) W_recon scatter: low-rank reconstruction
                    w_flat_t = W_recon_true.ravel()
                    w_flat_l = W_recon_learned.ravel()
                    axes[1, 2].scatter(w_flat_t, w_flat_l, s=1, c='black', alpha=0.05, rasterized=True)
                    wmax = max(np.abs(w_flat_t).max(), np.abs(w_flat_l).max())
                    axes[1, 2].plot([-wmax, wmax], [-wmax, wmax], 'g--', linewidth=2)
                    axes[1, 2].set_xlabel('true W (rank-r approx)', fontsize=28)
                    axes[1, 2].set_ylabel('learned W (rank-r approx)', fontsize=28)
                    axes[1, 2].tick_params(labelsize=20)
                    axes[1, 2].set_title(f'W_recon (rank={rank_r})  R²={W_recon_r2:.3f}', fontsize=28)
                    axes[1, 2].set_aspect('equal')
                else:
                    for j in range(3):
                        axes[1, j].text(0.5, 0.5, 'Procrustes failed\n(N too large or singular)',
                                       transform=axes[1, j].transAxes, ha='center', va='center', fontsize=20, color='red')

                # === Row 3: Alignment matrices + subspace R² per rank ===
                # (2,0) right singular vector alignment matrix
                im = axes[2, 0].imshow(alignment[:n_show, :n_show], cmap='hot', vmin=0, vmax=1)
                axes[2, 0].set_xlabel('learned SV index', fontsize=28)
                axes[2, 0].set_ylabel('true SV index', fontsize=28)
                axes[2, 0].set_title('right SV alignment (V)', fontsize=28)
                axes[2, 0].tick_params(labelsize=16)
                plt.colorbar(im, ax=axes[2, 0], fraction=0.046)

                # (2,1) left singular vector alignment matrix
                im_L = axes[2, 1].imshow(alignment_L, cmap='hot', vmin=0, vmax=1)
                axes[2, 1].set_xlabel('learned SV index', fontsize=28)
                axes[2, 1].set_ylabel('true SV index', fontsize=28)
                axes[2, 1].set_title('left SV alignment (U)', fontsize=28)
                axes[2, 1].tick_params(labelsize=16)
                plt.colorbar(im_L, ax=axes[2, 1], fraction=0.046)

                # (2,2) subspace R² as function of truncation rank
                if procrustes_ok:
                    axes[2, 2].plot(rank_list, U_r2_per_rank, 'o-', color='green', linewidth=2, markersize=8, label='U R²')
                    axes[2, 2].plot(rank_list, V_r2_per_rank, 's-', color='black', linewidth=2, markersize=8, label='V R²')
                    axes[2, 2].plot(rank_list, W_r2_per_rank, '^-', color='blue', linewidth=2, markersize=8, label='W_recon R²')
                    axes[2, 2].axvline(x=rank_90, color='orange', linestyle='--', linewidth=1.5, label=f'rank@90%={rank_90}')
                    axes[2, 2].axvline(x=rank_99, color='red', linestyle='--', linewidth=1.5, label=f'rank@99%={rank_99}')
                    axes[2, 2].set_xlabel('truncation rank', fontsize=28)
                    axes[2, 2].set_ylabel('R²', fontsize=28)
                    axes[2, 2].set_title('Procrustes R² vs truncation rank', fontsize=28)
                    axes[2, 2].set_ylim([-0.05, 1.05])
                    axes[2, 2].legend(fontsize=18)
                    axes[2, 2].tick_params(labelsize=20)
                else:
                    best_alignment_R = np.max(alignment[:n_show, :n_show], axis=1)
                    best_alignment_L = np.max(alignment_L, axis=1)
                    axes[2, 2].scatter(range(len(best_alignment_R)), best_alignment_R, s=50, c='green', alpha=0.7, label=f'right (mean={np.mean(best_alignment_R):.2f})')
                    axes[2, 2].scatter(range(len(best_alignment_L)), best_alignment_L, s=50, c='black', alpha=0.7, label=f'left (mean={np.mean(best_alignment_L):.2f})')
                    axes[2, 2].axhline(y=1/np.sqrt(n_show), color='gray', linestyle='--', linewidth=2, label=f'random ({1/np.sqrt(n_show):.2f})')
                    axes[2, 2].set_xlabel('SV index (sorted by singular value)', fontsize=28)
                    axes[2, 2].set_ylabel('best alignment score', fontsize=28)
                    axes[2, 2].set_title('best alignment per SV', fontsize=28)
                    axes[2, 2].set_ylim([0, 1.05])
                    axes[2, 2].legend(fontsize=20)
                    axes[2, 2].tick_params(labelsize=16)

                plt.tight_layout()
                plt.savefig(f'{log_dir}/results/eigen_comparison.png', dpi=87)
                plt.close()

                # --- Print and log all spectral/SVD metrics ---
                best_alignment_R = np.max(alignment[:n_show, :n_show], axis=1)
                best_alignment_L = np.max(alignment_L, axis=1)

                if eig_true is not None and eig_learned is not None:
                    true_spectral_radius = np.max(np.abs(eig_true))
                    learned_spectral_radius = np.max(np.abs(eig_learned))
                    print(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
                    logger.info(f'spectral radius - true: {true_spectral_radius:.3f}  learned: {learned_spectral_radius:.3f}')
                else:
                    print('spectral radius - skipped (eigenvalue computation failed)')
                    logger.warning('spectral radius computation skipped')

                print(f'SV alignment - right (V): {np.mean(best_alignment_R):.3f}  left (U): {np.mean(best_alignment_L):.3f}')
                logger.info(f'SV alignment - right (V): {np.mean(best_alignment_R):.3f}  left (U): {np.mean(best_alignment_L):.3f}')
                print(f'SV R² (singular values): {sv_r2:.4f}')
                logger.info(f'SV R² (singular values): {sv_r2:.4f}')
                print(f'effective rank - 90%: {rank_90}  99%: {rank_99}')
                logger.info(f'effective rank - 90%: {rank_90}  99%: {rank_99}')

                if procrustes_ok:
                    print(f'Procrustes SVD (rank={rank_r}) - U R²: {U_r2:.4f}  V R²: {V_r2:.4f}  W_recon R²: {W_recon_r2:.4f}')
                    logger.info(f'Procrustes SVD (rank={rank_r}) - U R²: {U_r2:.4f}  V R²: {V_r2:.4f}  W_recon R²: {W_recon_r2:.4f}')
                    print(f'Procrustes rank-20: U R²: {U_r2_rank20:.4f}  V R²: {V_r2_rank20:.4f}  W R²: {W_r2_rank20:.4f}')
                    logger.info(f'Procrustes rank-20: U R²: {U_r2_rank20:.4f}  V R²: {V_r2_rank20:.4f}  W R²: {W_r2_rank20:.4f}')
                    print(f'Procrustes R² per rank:')
                    logger.info(f'Procrustes R² per rank:')
                    for i_rk, rr in enumerate(rank_list):
                        line = f'  rank={rr:3d}  U_R2={U_r2_per_rank[i_rk]:.4f}  V_R2={V_r2_per_rank[i_rk]:.4f}  W_R2={W_r2_per_rank[i_rk]:.4f}'
                        print(line)
                        logger.info(line)

                if log_file:
                    if eig_true is not None and eig_learned is not None:
                        log_file.write(f"spectral_radius_true: {true_spectral_radius:.4f}\n")
                        log_file.write(f"spectral_radius_learned: {learned_spectral_radius:.4f}\n")
                    log_file.write(f"sv_alignment_R: {np.mean(best_alignment_R):.4f}\n")
                    log_file.write(f"sv_alignment_L: {np.mean(best_alignment_L):.4f}\n")
                    log_file.write(f"sv_r2: {sv_r2:.4f}\n")
                    log_file.write(f"effective_rank_90: {rank_90}\n")
                    log_file.write(f"effective_rank_99: {rank_99}\n")
                    if procrustes_ok:
                        log_file.write(f"procrustes_rank: {rank_r}\n")
                        log_file.write(f"procrustes_U_r2: {U_r2:.4f}\n")
                        log_file.write(f"procrustes_V_r2: {V_r2:.4f}\n")
                        log_file.write(f"procrustes_W_recon_r2: {W_recon_r2:.4f}\n")
                        for i_rk, rr in enumerate(rank_list):
                            log_file.write(f"procrustes_rank_{rr}_U_r2: {U_r2_per_rank[i_rk]:.4f}\n")
                            log_file.write(f"procrustes_rank_{rr}_V_r2: {V_r2_per_rank[i_rk]:.4f}\n")
                            log_file.write(f"procrustes_rank_{rr}_W_r2: {W_r2_per_rank[i_rk]:.4f}\n")
                        # Dedicated rank-20 keys for exploration tracking
                        log_file.write(f"U_r2_rank20: {U_r2_rank20:.4f}\n")
                        log_file.write(f"V_r2_rank20: {V_r2_rank20:.4f}\n")
                        log_file.write(f"W_r2_rank20: {W_r2_rank20:.4f}\n")

            # plot analyze_neuron_type_reconstruction
            results_per_neuron = analyze_neuron_type_reconstruction(
                config=config,
                model=model,
                edges=to_numpy(edges),
                true_weights=true_weights,
                gt_taus=to_numpy(gt_taus[:n_neurons]),
                gt_V_Rest=to_numpy(gt_V_Rest[:n_neurons]),
                learned_weights=learned_weights,
                learned_tau=learned_tau,
                learned_V_rest=learned_V_rest,
                type_list=to_numpy(type_list),
                n_frames=sim.n_frames,
                dimension=sim.dimension,
                n_neuron_types=sim.n_neuron_types,
                device=device,
                log_dir=log_dir,
                dataset_name=config.dataset,
                logger=logger,
                index_to_name=index_to_name,
                r_squared=r_squared,
                slope_corrected=slope_corrected,
                r_squared_tau=r_squared_tau,
                r_squared_V_rest=r_squared_V_rest,
                ode_params=ode_params,
            )

            print('alternative clustering methods...')


            # compute connectivity statistics (vectorized via bincount)
            print('computing connectivity statistics...')
            edges_np = to_numpy(edges)
            src, dst = edges_np[0], edges_np[1]

            def _connectivity_stats(w, src, dst, n):
                """Per-neuron mean/std/min/max of in-weights and out-weights."""
                in_count = np.bincount(dst, minlength=n).astype(np.float64)
                out_count = np.bincount(src, minlength=n).astype(np.float64)
                in_sum = np.bincount(dst, weights=w, minlength=n)
                out_sum = np.bincount(src, weights=w, minlength=n)
                in_sq = np.bincount(dst, weights=w ** 2, minlength=n)
                out_sq = np.bincount(src, weights=w ** 2, minlength=n)
                safe_in = np.where(in_count > 0, in_count, 1)
                safe_out = np.where(out_count > 0, out_count, 1)
                in_mean = in_sum / safe_in
                out_mean = out_sum / safe_out
                in_std = np.sqrt(np.maximum(in_sq / safe_in - in_mean ** 2, 0))
                out_std = np.sqrt(np.maximum(out_sq / safe_out - out_mean ** 2, 0))
                in_max = np.full(n, -np.inf); np.maximum.at(in_max, dst, w)
                in_min = np.full(n,  np.inf); np.minimum.at(in_min, dst, w)
                out_max = np.full(n, -np.inf); np.maximum.at(out_max, src, w)
                out_min = np.full(n,  np.inf); np.minimum.at(out_min, src, w)
                for arr, c in [(in_mean, in_count), (in_std, in_count),
                               (in_min, in_count), (in_max, in_count),
                               (out_mean, out_count), (out_std, out_count),
                               (out_min, out_count), (out_max, out_count)]:
                    arr[c == 0] = 0
                return in_mean, in_std, out_mean, out_std, in_min, in_max, out_min, out_max

            (w_in_mean_true, w_in_std_true, w_out_mean_true, w_out_std_true,
             w_in_min_true, w_in_max_true, w_out_min_true, w_out_max_true) = \
                _connectivity_stats(true_weights.flatten(), src, dst, n_neurons)
            (w_in_mean_learned, w_in_std_learned, w_out_mean_learned, w_out_std_learned,
             w_in_min_learned, w_in_max_learned, w_out_min_learned, w_out_max_learned) = \
                _connectivity_stats(learned_weights.flatten(), src, dst, n_neurons)

            # all connectivity stats combined (mean, std, min, max for in and out)
            W_learned = np.column_stack([w_in_mean_learned, w_in_std_learned,
                                        w_out_mean_learned, w_out_std_learned,
                                        w_in_min_learned, w_in_max_learned,
                                        w_out_min_learned, w_out_max_learned])
            W_true = np.column_stack([w_in_mean_true, w_in_std_true,
                                    w_out_mean_true, w_out_std_true,
                                    w_in_min_true, w_in_max_true,
                                    w_out_min_true, w_out_max_true])

            n_gmm = min(100, n_neurons - 1)

            # Build augmented embedding for GMM + UMAP
            _aug_parts = [to_numpy(model.a), learned_tau.reshape(-1, 1)]
            if ode_params.has_vrest():
                _aug_parts.append(learned_V_rest.reshape(-1, 1))
            _aug_parts.extend([w_in_mean_learned.reshape(-1, 1), w_in_std_learned.reshape(-1, 1),
                               w_out_mean_learned.reshape(-1, 1), w_out_std_learned.reshape(-1, 1),
                               w_in_min_learned.reshape(-1, 1), w_in_max_learned.reshape(-1, 1),
                               w_out_min_learned.reshape(-1, 1), w_out_max_learned.reshape(-1, 1)])
            a_aug = np.column_stack(_aug_parts)

            results = clustering_gmm(a_aug, type_list, n_components=n_gmm)
            cluster_acc = results['accuracy']
            print(f"GMM (n_components={n_gmm}): accuracy={_r2_color(cluster_acc)}{cluster_acc:.3f}{_ANSI_RESET}, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")
            logger.info(f"GMM n_components={n_gmm}, accuracy={cluster_acc:.3f}, ARI={results['ari']:.3f}, NMI={results['nmi']:.3f}")

            # Write cluster accuracy to metrics.txt and analysis log
            metrics_path = os.path.join(log_dir, 'results', 'metrics.txt')
            try:
                with open(metrics_path, 'a') as mf:
                    mf.write(f"clustering_accuracy: {cluster_acc:.4f}\n")
            except OSError:
                pass
            if log_file:
                log_file.write(f"cluster_accuracy: {cluster_acc:.4f}\n")

            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            a_umap = reducer.fit_transform(a_aug)

            # Get cluster labels from GMM
            results = clustering_gmm(a_aug, type_list, n_components=n_gmm)
            cluster_labels = None
            for cov_type in ('full', 'diag', 'spherical'):
                try:
                    cluster_labels = GaussianMixture(
                        n_components=n_gmm, random_state=42,
                        reg_covar=1e-3, covariance_type=cov_type,
                    ).fit_predict(a_aug)
                    break
                except Exception:
                    continue
            if cluster_labels is None:
                cluster_labels = np.zeros(len(a_aug), dtype=int)

            plt.figure(figsize=(10, 9))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_alpha(0.75)
            from matplotlib.colors import ListedColormap
            cmap_65 = ListedColormap(colors_65)
            plt.scatter(a_umap[:, 0], a_umap[:, 1], c=cluster_labels, s=24, cmap=cmap_65, alpha=0.8, edgecolors='none')


            plt.xlabel(r'UMAP$_1$', fontsize=48)
            plt.ylabel(r'UMAP$_2$', fontsize=48)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.text(0.05, 0.95, f"accuracy: {cluster_acc:.2f}",
                    transform=plt.gca().transAxes, fontsize=32, verticalalignment='top')
            plt.tight_layout()
            plt.savefig(f'{log_dir}/results/embedding_augmented_{config_indices}.png', dpi=300)
            plt.close()

    # ---- Activity traces: clean vs noisy (measurement noise) ----
    if getattr(sim, 'measurement_noise_level', 0) > 0:
        try:
            import zarr as _zarr
            data_dir = graphs_data_path(config.dataset, 'x_list_train')
            voltage_path = os.path.join(data_dir, 'voltage.zarr')
            noise_path = os.path.join(data_dir, 'noise.zarr')
            if os.path.isdir(voltage_path) and os.path.isdir(noise_path):
                _n_traces = 20
                _frame_start, _frame_end = 5000, 5500
                _rng = np.random.RandomState(42)
                _voltage = _zarr.open(voltage_path, 'r')[_frame_start:_frame_end, :]
                _noise = _zarr.open(noise_path, 'r')[_frame_start:_frame_end, :]
                _n_neurons_total = _voltage.shape[1]
                _indices = np.sort(_rng.choice(_n_neurons_total, _n_traces, replace=False))
                _clean = _voltage[:, _indices].T
                _noisy = (_voltage[:, _indices] + _noise[:, _indices]).T
                _trace_range = np.median(np.ptp(_clean, axis=1))
                _spacing = _trace_range * 1.8
                _offsets = _spacing * np.arange(_n_traces)[:, None]
                _clean_off = _clean + _offsets
                _noisy_off = _noisy + _offsets
                _sigma = sim.measurement_noise_level
                _xvals = np.arange(_frame_start, _frame_end)

                # SNR for voltage: var(signal) / var(noise)
                _signal_var = np.var(_clean)
                _noise_var = np.var(_noise[:, _indices].T)
                _snr_v = _signal_var / _noise_var if _noise_var > 0 else float('inf')
                _snr_v_db = 10 * np.log10(_snr_v) if np.isfinite(_snr_v) else float('inf')

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor='white', sharey=True)
                for ax, data, subtitle in [(ax1, _clean_off, 'without measurement noise'),
                                            (ax2, _noisy_off, 'with measurement noise')]:
                    ax.set_facecolor('white')
                    ax.plot(_xvals, data.T, linewidth=0.6, alpha=0.85, color='#333333')
                    ax.set_xlim([_frame_start, _frame_end])
                    ax.set_ylim([data[0].min() - _spacing, data[-1].max() + _spacing])
                    ax.set_yticks([])
                    ax.tick_params(axis='x', labelsize=9)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlabel('time (frames)', fontsize=14)
                    ax.set_title(subtitle, fontsize=18, pad=12)
                ax1.set_ylabel(f'{_n_traces} / {_n_neurons_total} neurons', fontsize=14)
                ax2.text(0.97, 0.90, f'SNR(V) = {_snr_v_db:.1f} dB',
                         transform=ax2.transAxes, fontsize=18,
                         verticalalignment='top', horizontalalignment='right')
                fig.subplots_adjust(wspace=0.05)
                _out = graphs_data_path(config.dataset, 'activity_traces_noisy.png')
                plt.savefig(_out, dpi=300, facecolor='white', bbox_inches='tight')
                plt.close()
                logger.info(f'saved activity_traces_noisy.png')
        except Exception as _e:
            logger.warning(f'could not generate activity_traces_noisy: {_e}')


def analyze_neuron_type_reconstruction(config, model, edges, true_weights, gt_taus, gt_V_Rest,
                                       learned_weights, learned_tau, learned_V_rest, type_list, n_frames, dimension,
                                       n_neuron_types, device, log_dir, dataset_name, logger, index_to_name,
                                       r_squared=None, slope_corrected=None, r_squared_tau=None, r_squared_V_rest=None,
                                       ode_params=None):

    print('stratified analysis by neuron type...')

    # Determine which RMSE panels to show based on model
    panels = ode_params.neuron_type_rmse_panels() if ode_params else ["weights", "tau", "vrest"]

    rmse_weights = []
    rmse_taus = []
    rmse_vrests = []
    n_connections = []

    for neuron_type in range(n_neuron_types):
        type_indices_edge = np.where(type_list[edges[1,:]] == neuron_type)[0]
        gt_w_type = true_weights[type_indices_edge]
        learned_w_type = learned_weights[type_indices_edge]
        n_conn = len(type_indices_edge)

        type_indices = np.where(type_list == neuron_type)[0]

        rmse_w = np.sqrt(np.mean((gt_w_type - learned_w_type)** 2))
        rmse_weights.append(rmse_w)
        n_connections.append(n_conn)

        if "tau" in panels:
            rmse_taus.append(np.sqrt(np.mean((gt_taus[type_indices] - learned_tau[type_indices])** 2)))
        if "vrest" in panels:
            rmse_vrests.append(np.sqrt(np.mean((gt_V_Rest[type_indices] - learned_V_rest[type_indices])** 2)))

    n_neurons = len(type_list)

    # Per-neuron RMSE
    rmse_weights_per_neuron = np.zeros(n_neurons)
    for neuron_idx in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == neuron_idx)[0]
        if len(incoming_edges) > 0:
            rmse_weights_per_neuron[neuron_idx] = np.sqrt(np.mean((learned_weights[incoming_edges] - true_weights[incoming_edges])**2))

    rmse_tau_per_neuron = np.abs(learned_tau - gt_taus) if "tau" in panels else np.zeros(n_neurons)
    rmse_vrest_per_neuron = np.abs(learned_V_rest - gt_V_Rest) if "vrest" in panels else np.zeros(n_neurons)

    rmse_weights = np.array(rmse_weights)
    rmse_taus = np.array(rmse_taus) if rmse_taus else np.zeros(n_neuron_types)
    rmse_vrests = np.array(rmse_vrests) if rmse_vrests else np.zeros(n_neuron_types)

    sorted_neuron_type_names = [index_to_name.get(t, f'Type{t}') for t in range(n_neuron_types)]
    x_pos = np.arange(n_neuron_types)

    # Build panel config: (data, ylabel, color, ylim, threshold)
    panel_defs = []
    if "weights" in panels:
        panel_defs.append((rmse_weights, 'RMSE weights', 'skyblue', [0, 2.5], 0.5))
    if "tau" in panels:
        panel_defs.append((rmse_taus, r'RMSE $\tau$', 'lightcoral', [0, 0.3], 0.03))
    if "vrest" in panels:
        panel_defs.append((rmse_vrests, r'RMSE $V_{rest}$', 'lightgreen', [0, 0.8], 0.08))

    n_panels = len(panel_defs)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, (data, ylabel, color, ylim, thresh) in zip(axes, panel_defs):
        ax.bar(x_pos, data, color=color, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_ylim(ylim)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_neuron_type_names, rotation=90, ha='right', fontsize=6)
        ax.grid(False)
        ax.tick_params(axis='y', labelsize=12)
        for i, (tick, val) in enumerate(zip(ax.get_xticklabels(), data)):
            if val > thresh:
                tick.set_color('red')
                tick.set_fontsize(8)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'results', 'neuron_type_reconstruction.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Log summary statistics
    logger.info(f"mean weights RMSE: {np.mean(rmse_weights):.3f} ± {np.std(rmse_weights):.3f}")
    if "tau" in panels:
        logger.info(f"mean tau RMSE: {np.mean(rmse_taus):.3f} ± {np.std(rmse_taus):.3f}")
    if "vrest" in panels:
        logger.info(f"mean V_rest RMSE: {np.mean(rmse_vrests):.3f} ± {np.std(rmse_vrests):.3f}")

    # Write key-value metrics — use append so clustering_accuracy written
    # earlier by _plot_synaptic_linear / plot_synaptic is not overwritten.
    metrics_path = os.path.join(log_dir, 'results', 'metrics.txt')
    if r_squared is not None:
        with open(metrics_path, 'a') as mf:
            mf.write(f"W_corrected_R2: {r_squared:.4f}\n")
            mf.write(f"W_corrected_slope: {slope_corrected:.4f}\n")
            if "tau" in panels:
                mf.write(f"tau_R2: {r_squared_tau:.4f}\n")
            if "vrest" in panels:
                mf.write(f"V_rest_R2: {r_squared_V_rest:.4f}\n")

    return {
        'rmse_weights_per_neuron': rmse_weights_per_neuron,
        'rmse_tau_per_neuron': rmse_tau_per_neuron,
        'rmse_vrest_per_neuron': rmse_vrest_per_neuron,
        'rmse_weights_per_type': rmse_weights,
        'rmse_tau_per_type': rmse_taus,
        'rmse_vrest_per_type': rmse_vrests
    }


def plot_neuron_activity_analysis(activity, target_type_name_list, type_list, index_to_name, n_neurons, n_frames, delta_t, output_path):

   # Calculate mean and std for each neuron
   mu_activity = torch.mean(activity, dim=1)
   sigma_activity = torch.std(activity, dim=1)

   # Create the plot (keeping original visualization)
   plt.figure(figsize=(16, 8))
   plt.errorbar(np.arange(n_neurons), to_numpy(mu_activity), yerr=to_numpy(sigma_activity),
                fmt='o', ecolor='lightgray', alpha=0.6, elinewidth=1, capsize=0,
                markersize=3, color='red')

   # Group neurons by type and add labels at type boundaries (similar to plot_ground_truth_distributions)
   type_boundaries = {}
   current_type = None
   for i in range(n_neurons):
       neuron_type_id = to_numpy(type_list[i]).item()
       if neuron_type_id != current_type:
           if current_type is not None:
               type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
           type_boundaries[neuron_type_id] = (i, i)
           current_type = neuron_type_id

   # Close the last type boundary
   if current_type is not None:
       type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

   # Add vertical lines and x-tick labels for each neuron type
   tick_positions = []
   tick_labels = []

   for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
       center_pos = (start_idx + end_idx) / 2
       neuron_type_name = index_to_name.get(neuron_type_id, f'Type{neuron_type_id}')

       tick_positions.append(center_pos)
       tick_labels.append(neuron_type_name)

       # Add vertical line at type boundary
       if start_idx > 0:
           plt.axvline(x=start_idx, color='gray', linestyle='--', alpha=0.3)

   # Set x-ticks with neuron type names rotated 90 degrees
   plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=10)
   plt.ylabel(r'neuron voltage $v_i(t)\quad\mu_i \pm \sigma_i$', fontsize=16)
   plt.yticks(fontsize=18)

   plt.tight_layout()
   plt.savefig(os.path.join(output_path, 'activity_mu_sigma.png'), dpi=300, bbox_inches='tight')
   plt.close()

   # Return per-neuron statistics (NEW)
   return {
       'mu_activity': to_numpy(mu_activity),
       'sigma_activity': to_numpy(sigma_activity)
   }


def plot_ground_truth_distributions(edges, true_weights, gt_taus, gt_V_Rest, type_list, n_neuron_types,
                                    sorted_neuron_type_names, output_path):
    """
    Create a 4-panel vertical figure showing ground truth parameter distributions per neuron type
    with neuron type names as x-axis labels
    """

    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # Get type boundaries for labels
    type_boundaries = {}
    current_type = None
    n_neurons = len(type_list)

    for i in range(n_neurons):
        neuron_type_id = int(type_list[i])
        if neuron_type_id != current_type:
            if current_type is not None:
                type_boundaries[current_type] = (type_boundaries[current_type][0], i - 1)
            type_boundaries[neuron_type_id] = (i, i)
            current_type = neuron_type_id

    # Close the last type boundary
    if current_type is not None:
        type_boundaries[current_type] = (type_boundaries[current_type][0], n_neurons - 1)

    def add_type_labels_and_setup_axes(ax, y_values, title):
        # Add mean line for each type and collect type positions
        type_positions = []
        type_names = []

        for neuron_type_id, (start_idx, end_idx) in type_boundaries.items():
            center_pos = (start_idx + end_idx) / 2
            type_positions.append(center_pos)
            neuron_type_name = sorted_neuron_type_names[int(neuron_type_id)] if int(neuron_type_id) < len(
                sorted_neuron_type_names) else f'Type{neuron_type_id}'
            type_names.append(neuron_type_name)

            # Add mean line for this type
            type_mean = np.mean(y_values[start_idx:end_idx + 1])
            ax.hlines(type_mean, start_idx, end_idx, colors='red', linewidth=3)

        # Set x-ticks to neuron type names
        ax.set_xticks(type_positions)
        ax.set_xticklabels(type_names, rotation=90, fontsize=8)
        ax.tick_params(axis='y', labelsize=16)

    # Panel 1: Scatter plot of true weights per connection with neuron index
    ax1 = axes[0]
    connection_targets = edges[1, :]
    connection_weights = true_weights

    ax1.scatter(connection_targets, connection_weights, c='white', s=0.1)
    ax1.set_ylabel('true weights', fontsize=16)

    # For weights, compute means per target neuron
    weight_means_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        incoming_edges = np.where(edges[1, :] == i)[0]
        if len(incoming_edges) > 0:
            weight_means_per_neuron[i] = np.mean(true_weights[incoming_edges])

    add_type_labels_and_setup_axes(ax1, weight_means_per_neuron, 'distribution of true weights by neuron type')

    # Panel 2: Number of connections per neuron
    ax2 = axes[1]
    n_connections_per_neuron = np.zeros(n_neurons)
    for i in range(n_neurons):
        n_connections_per_neuron[i] = np.sum(edges[1, :] == i)

    ax2.scatter(np.arange(n_neurons), n_connections_per_neuron, c='white', s=0.1)
    ax2.set_ylabel('number of connections', fontsize=16)
    add_type_labels_and_setup_axes(ax2, n_connections_per_neuron, 'number of incoming connections by neuron type')

    # Panel 3: Scatter plot of true tau values per neuron
    ax3 = axes[2]
    ax3.scatter(np.arange(n_neurons), gt_taus * 1000, c='white', s=0.1)
    ax3.set_ylabel(r'true $\tau$ values [ms]', fontsize=16)
    add_type_labels_and_setup_axes(ax3, gt_taus * 1000, r'distribution of true $\tau$ by neuron type')

    # Panel 4: Scatter plot of true V_rest values per neuron
    ax4 = axes[3]
    ax4.scatter(np.arange(n_neurons), gt_V_Rest, c='white', s=0.1)
    ax4.set_ylabel(r'true $v_{rest}$ values [a.u.]', fontsize=16)
    add_type_labels_and_setup_axes(ax4, gt_V_Rest, r'distribution of true $v_{rest}$ by neuron type')

    plt.tight_layout()
    plt.savefig(f'{output_path}/ground_truth_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    return fig
    plt.close()


def data_plot(config, epoch_list, style, extended, device, apply_weight_correction=False, log_file=None, skip_svd=False):
    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    fig_style.apply_globally()

    log_dir, logger = create_log_dir(config=config, erase=False, erase_results=False)

    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)

    if epoch_list==['best']:
        _models_dir = f"{log_dir}/models"
        _pattern = f"{_models_dir}/best_model_with_*"
        files = glob.glob(_pattern)
        files.sort(key=sort_key)
        if not files:
            # Diagnose: log_dir? models/? what's in there?
            _log_exists = os.path.isdir(log_dir)
            _models_exists = os.path.isdir(_models_dir)
            _all_in_models = sorted(os.listdir(_models_dir)) if _models_exists else []
            _best_any = [f for f in _all_in_models if f.startswith('best_model')]
            print('\033[91m' + '=' * 70)
            print(f"ERROR: no 'best_model_with_*' checkpoint matched.")
            print(f"  log_dir:       {log_dir}  (exists={_log_exists})")
            print(f"  models/:       {_models_dir}  (exists={_models_exists})")
            print(f"  glob pattern:  {_pattern}")
            print(f"  matches:       {len(files)}")
            if _models_exists:
                print(f"  files in models/: {len(_all_in_models)} total")
                if _best_any:
                    print(f"  best_model* present (but didn't match 'with_'): {_best_any[:5]}")
                else:
                    print(f"  no best_model* files at all. First 10 entries: {_all_in_models[:10]}")
                    print(f"  → training may not have produced a best checkpoint yet,")
                    print(f"    or this run wrote a different naming scheme.")
            print('=' * 70 + '\033[0m')
            raise FileNotFoundError(
                f"No best_model_with_* checkpoint in {_models_dir}. "
                f"Pass epoch_list explicitly or train the model first."
            )
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]

        epoch_list=[filename]
        print(f'best model: {epoch_list}')
        logger.info(f'best model: {epoch_list}')

    # Mirror final loss to analysis.log without producing loss.png
    # (the figure is no longer generated by data_plot).
    if log_file and os.path.exists(f'{log_dir}/loss.pt'):
        loss = torch.load(f'{log_dir}/loss.pt', weights_only=False)
        if len(loss) > 0:
            log_file.write(f"final_loss: {loss[-1]:.4e}\n")


    _connconstr = any(x in config.dataset for x in ('drosophila_cx', 'zebrafish_oculomotor', 'larva'))
    if 'fly' in config.dataset or _connconstr:
        if config.simulation.calcium_type != 'none':
            plot_synaptic_calcium(config, epoch_list, log_dir, logger, 'viridis', style, extended, device, skip_svd=skip_svd) # noqa: F821
        else:
            plot_synaptic(config, epoch_list, log_dir, logger, 'viridis', style, extended, device, log_file=log_file, skip_svd=skip_svd)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=FutureWarning)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    print(' ')
    print(f'device {device}')

    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass


    config_list = ['signal_Claude']


    for config_file_ in config_list:
        print(' ')
        config_file, pre_folder = add_pre_folder(config_file_)
        config = NeuralGraphConfig.from_yaml(config_path(f'{config_file}.yaml'))
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        print(f'\033[94mconfig_file  {config.config_file}\033[0m')
        folder_name = log_path(pre_folder, 'tmp_results') + '/'
        os.makedirs(folder_name, exist_ok=True)
        data_plot(config=config, epoch_list=['best'], style='black color', extended='plots', device=device, apply_weight_correction=True)
        # data_plot(config=config, epoch_list=['all'], style='black color', extended='plots', device=device, apply_weight_correction=False)
        # data_plot(config=config, epoch_list=['all'], style='black color', extended='plots', device=device, apply_weight_correction=True)


    print("analysis completed")


