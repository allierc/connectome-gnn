"""Standalone: cv03 known_ode rollout with learned τ vs ground-truth τ.

Tests whether the τ outliers identified in
``tau_comparison_wo_outliers_noise_free_blank50_cv03.png`` (Tm4, C3, CT1(M10),
Mi12, T4d, Tm30) measurably hurt the rollout — by replacing the model's
learned τ with the dataset's ground-truth τ and re-running the autoregressive
rollout end-to-end.

Why known_ode (not unified):
    The unified GNN's f_theta is a free MLP returning dv/dt — there is no
    explicit τ parameter to swap. The known_ode model uses
        dv/dt = (-v + msg + I + V_rest) / softplus(raw_tau)
    so we can literally set raw_tau ← inv_softplus(gt_tau) and re-run the
    integrator.

What the script outputs:
    1. Re-runs the rollout with the model as-loaded (learned τ) — sanity check.
    2. Overrides raw_tau with inv_softplus(gt_tau), re-runs the rollout.
    3. Saves both bundles next to the existing rollout_bundle.npz:
         results/rollout_bundle_relearned_tau.npz   (sanity)
         results/rollout_bundle_gt_tau.npz          (the test)
    4. Prints a side-by-side metrics table (Pearson r Fisher-pooled, RMSE,
       max|pred|), both globally and restricted to the 6 outlier cell types.

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/rollout_known_ode_cv03_gt_tau.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free_blank50_known_ode_cv03.yaml
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_blank50_cv03/{edge_index.pt, ode_params.pt}
# Test data      : <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_blank50_cv03/x_list_test/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_blank50_cv03/y_list_test/
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_known_ode_cv03/models/best_model_with_0_graphs_0.pt
#                  <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_known_ode_cv03/training_edges.pt
# Eval logs      : <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_known_ode_cv03/results/rollout_bundle.npz
# Output         : <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_known_ode_cv03/results/rollout_bundle_relearned_tau.npz
#                  <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_known_ode_cv03/results/rollout_bundle_gt_tau.npz
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange


REPO_ROOT = '/workspace/connectome-gnn'
DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'

# Make `connectome_gnn` importable.
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from connectome_gnn.config import NeuralGraphConfig                 # noqa: E402
from connectome_gnn.utils import set_data_root, migrate_state_dict  # noqa: E402
from connectome_gnn.utils import compute_trace_metrics, fisher_pool # noqa: E402
import connectome_gnn.models.registry as _reg                       # noqa: E402
from connectome_gnn.models.registry import create_model             # noqa: E402
_reg._discover_models()  # populate model registry
from connectome_gnn.models.utils import _batch_frames                # noqa: E402
from connectome_gnn.zarr_io import (                                 # noqa: E402
    load_simulation_data, load_raw_array,
)
from connectome_gnn.generators.ode_params import load_edge_index    # noqa: E402


# ---------------------------------------------------------------------------
MODEL_NAME  = 'flyvis_noise_free_blank50_known_ode_cv03'
MODEL_YAML  = f'{DATA_ROOT}/config/fly/{MODEL_NAME}.yaml'
LOG_DIR     = f'{DATA_ROOT}/log/fly/{MODEL_NAME}'
RESULTS_DIR = f'{LOG_DIR}/results'
DATASET     = 'flyvis_noise_free_blank50_cv03'
DATA_DIR    = f'{DATA_ROOT}/graphs_data/fly/{DATASET}'

# 6 outlier cell types from tau_comparison_wo_outliers_noise_free_blank50_cv03.png
OUTLIER_TYPES = ['Tm4', 'C3', 'CT1(M10)', 'Mi12', 'T4d', 'Tm30']


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse softplus: returns x s.t. softplus(x) = y.

    softplus(x) = log(1+exp(x))  ⇒  x = log(exp(y) - 1) = y + log(1 - exp(-y))
    Implemented via expm1 to stay accurate near y=0 and large-y asymptote.
    """
    # log(expm1(y)) is well-defined for y > 0.
    return torch.where(
        y > 20.0,
        y,                                # softplus(y) ≈ y for y > 20
        torch.log(torch.expm1(torch.clamp(y, min=1e-12))),
    )


def load_known_ode_model(config, device):
    """Build FlyvisKnownODE, load checkpoint."""
    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    # Match the n_edges adjustment data_test_gnn does so the model dim agrees
    # with training_edges.pt (handles null-edges or fully connected modes).
    training_edges_path = f'{LOG_DIR}/training_edges.pt'
    if os.path.exists(training_edges_path):
        edges_for_size = torch.load(training_edges_path, map_location='cpu',
                                    weights_only=False)
    else:
        edges_for_size = load_edge_index(DATA_DIR, device='cpu')
    actual_n_edges = edges_for_size.shape[1]
    expected_total = sim.n_edges + sim.n_extra_null_edges
    if actual_n_edges == expected_total and sim.n_extra_null_edges > 0:
        config.simulation.n_edges = actual_n_edges
        config.simulation.n_extra_null_edges = 0
    elif actual_n_edges != sim.n_edges:
        config.simulation.n_edges = actual_n_edges

    model = create_model(model_config.signal_model_name,
                         aggr_type=model_config.aggr_type,
                         config=config, device=device).to(device)

    ckpt_path = f'{LOG_DIR}/models/best_model_with_0_graphs_0.pt'
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    migrate_state_dict(state_dict)
    sd = state_dict['model_state_dict']
    # Strip torch.compile prefix if present.
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, ckpt_path


def load_test_data(config, device):
    sim = config.simulation
    load_fields = ['voltage', 'stimulus', 'neuron_type']
    test_path = f'{DATA_DIR}/x_list_test'
    if not os.path.exists(test_path):
        sys.exit(f'missing test data: {test_path}')
    x_ts = load_simulation_data(test_path, fields=load_fields).to(device)
    y_ts = load_raw_array(f'{DATA_DIR}/y_list_test')
    type_ids = x_ts.neuron_type.detach().cpu().numpy().astype(int).copy()
    x_ts.neuron_type = None
    x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long, device=device)

    MAX_TEST_FRAMES = 8000
    if x_ts.n_frames > MAX_TEST_FRAMES:
        x_ts = x_ts.truncate_frames(MAX_TEST_FRAMES)
        y_ts = y_ts[:MAX_TEST_FRAMES]

    edges_path = f'{LOG_DIR}/training_edges.pt'
    if os.path.exists(edges_path):
        edges = torch.load(edges_path, map_location=device, weights_only=False)
    else:
        edges = load_edge_index(DATA_DIR, device=device)
    return x_ts, y_ts, edges, type_ids


@torch.no_grad()
def run_rollout(model, x_ts, edges, sim, device):
    """Replicate the autoregressive rollout from graph_tester.data_test_gnn."""
    n_eval = x_ts.n_frames
    n_neurons = x_ts.n_neurons
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)

    x = x_ts.frame(0)
    pred_list, true_list, stim_list = [], [], []

    for k in trange(n_eval - 1, ncols=100, desc='rollout'):
        pred_list.append(x.voltage.detach().cpu().numpy().astype(np.float32))
        true_list.append(x_ts.frame(k).voltage.detach().cpu().numpy().astype(np.float32))
        x.stimulus = x_ts.frame(k).stimulus.clone()
        stim_list.append(x.stimulus.detach().cpu().numpy().astype(np.float32))

        batched_state, batched_edges = _batch_frames([x], edges)
        pred, _, _ = model(batched_state, batched_edges,
                           data_id=data_id, return_all=True)
        # Forward Euler: v_{k+1} = v_k + dt * dvdt. Same clamp as the
        # canonical rollout in graph_tester (predictions are stored raw;
        # explosion shows up as ±100 only after the dataset's clip).
        x.voltage = (x.voltage.unsqueeze(-1) + sim.delta_t * pred).squeeze(-1)
        x.voltage = torch.clamp(x.voltage, -100.0, 100.0)

    pred_arr = np.stack(pred_list, axis=0).T   # (n_neurons, n_frames-1)
    true_arr = np.stack(true_list, axis=0).T
    stim_arr = np.stack(stim_list, axis=0).T
    return pred_arr, true_arr, stim_arr


def metrics(pred, true):
    rmse, pearson, _, _ = compute_trace_metrics(true, pred)
    fp = fisher_pool(pearson)
    return {
        'pearson_mean': float(fp['r_mean']),
        'pearson_sd':   float(fp['r_sd_sym']),
        'rmse_mean':    float(np.mean(rmse)),
        'max_abs_pred': float(np.abs(pred).max()),
        'frac_clamped': float((np.abs(pred) >= 99.99).any(axis=1).mean()),
    }


def metrics_subset(pred, true, type_ids, type_names, picked):
    out = {}
    for name in picked:
        if name not in type_names:
            continue
        tid = type_names.index(name)
        mask = type_ids == tid
        if mask.sum() == 0:
            continue
        out[name] = metrics(pred[mask], true[mask])
    return out


def fmt_row(label, m):
    return (f"  {label:<20s}  "
            f"r={m['pearson_mean']:+.4f}±{m['pearson_sd']:.4f}  "
            f"RMSE={m['rmse_mean']:.3f}  "
            f"|max|={m['max_abs_pred']:8.2f}  "
            f"clamped={m['frac_clamped']*100:5.1f}%")


def main():
    set_data_root(DATA_ROOT)
    if not os.path.isfile(MODEL_YAML):
        sys.exit(f'missing yaml: {MODEL_YAML}')
    cfg = NeuralGraphConfig.from_yaml(MODEL_YAML)
    parent = os.path.basename(os.path.dirname(os.path.abspath(MODEL_YAML)))
    if not cfg.dataset.startswith(parent + '/'):
        cfg.dataset = parent + '/' + cfg.dataset
    if cfg.config_file == 'none':
        cfg.config_file = parent + '/' + os.path.splitext(os.path.basename(MODEL_YAML))[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # --- model + data ---
    model, ckpt_path = load_known_ode_model(cfg, device)
    print(f'loaded {ckpt_path}')
    x_ts, y_ts, edges, type_ids = load_test_data(cfg, device)
    type_names = list(np.load(f'{RESULTS_DIR}/rollout_bundle.npz',
                              allow_pickle=True)['type_names'])
    print(f'test data: {x_ts.n_neurons} neurons × {x_ts.n_frames} frames; '
          f'{edges.shape[1]} edges')

    # --- gt τ from the dataset ---
    op = torch.load(f'{DATA_DIR}/ode_params.pt', map_location=device,
                    weights_only=False)
    gt_tau = op['tau_i'].to(device).float()
    n_neurons = x_ts.n_neurons
    if gt_tau.numel() < n_neurons:
        sys.exit(f'gt_tau has {gt_tau.numel()} entries, need {n_neurons}')
    gt_tau = gt_tau[:n_neurons]
    learned_tau = F.softplus(model.raw_tau).detach()

    print(f'\nτ stats:')
    print(f'  gt_tau:      min={gt_tau.min().item():.4f}  '
          f'max={gt_tau.max().item():.4f}  mean={gt_tau.mean().item():.4f}')
    print(f'  learned τ:   min={learned_tau.min().item():.4f}  '
          f'max={learned_tau.max().item():.4f}  mean={learned_tau.mean().item():.4f}')
    print(f'  |Δτ| max:    {(learned_tau - gt_tau).abs().max().item():.4f}')
    print(f'  |Δτ| median: {(learned_tau - gt_tau).abs().median().item():.4f}')

    # --- rollout #1: learned τ (sanity vs existing rollout_bundle.npz) ---
    print('\n=== rollout #1: as-loaded (learned τ) ===')
    pred1, true1, stim1 = run_rollout(model, x_ts, edges, cfg.simulation, device)
    out1 = f'{RESULTS_DIR}/rollout_bundle_relearned_tau.npz'
    np.savez(out1, activity_true=true1, activity_pred=pred1, stimulus=stim1,
             type_ids=type_ids, type_names=np.array(type_names))
    print(f'  saved {out1}')

    # --- override τ ← inv_softplus(gt_tau), rollout #2 ---
    print('\n=== rollout #2: gt τ injected ===')
    raw_gt = inv_softplus(gt_tau)
    with torch.no_grad():
        model.raw_tau.copy_(raw_gt)
    # Verify
    new_tau = F.softplus(model.raw_tau).detach()
    err = (new_tau - gt_tau).abs().max().item()
    print(f'  inv_softplus round-trip max error: {err:.2e}')
    pred2, true2, stim2 = run_rollout(model, x_ts, edges, cfg.simulation, device)
    out2 = f'{RESULTS_DIR}/rollout_bundle_gt_tau.npz'
    np.savez(out2, activity_true=true2, activity_pred=pred2, stimulus=stim2,
             type_ids=type_ids, type_names=np.array(type_names))
    print(f'  saved {out2}')

    # --- comparison ---
    print('\n=== global metrics ===')
    m1 = metrics(pred1, true1)
    m2 = metrics(pred2, true2)
    print(fmt_row('learned τ',     m1))
    print(fmt_row('gt τ',          m2))

    print('\n=== per-cell-type (the 6 τ-outlier types) ===')
    s1 = metrics_subset(pred1, true1, type_ids, type_names, OUTLIER_TYPES)
    s2 = metrics_subset(pred2, true2, type_ids, type_names, OUTLIER_TYPES)
    for name in OUTLIER_TYPES:
        if name in s1 and name in s2:
            print(f'  {name}:')
            print(fmt_row('    learned τ', s1[name]))
            print(fmt_row('    gt τ     ', s2[name]))


if __name__ == '__main__':
    main()
