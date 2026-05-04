"""Standalone test: do unified_cv03's *extracted* (τ, V_rest, W) explode when
plugged into a clean known_ode integrator? And does swapping τ for gt fix it?

Motivation
----------
The unified GNN explodes on cv03 (predictions clamp at ±100 starting at frame
~160). We can't directly inject gt τ into unified — its f_theta is a free MLP
with no τ parameter to swap. But we can:

  1. Take unified_cv03's extracted (W, V_rest, τ) — saved in
     ``results/learned_ode_params.pt`` by the parameter-recovery analysis.
  2. Plug those values into a FlyvisKnownODE model:
        dv/dt = (-v + msg + I + V_rest) / softplus(raw_tau)
     with msg_j = W_j * ReLU(v_src). The known_ode integrator is bounded /
     well-conditioned by construction — any explosion comes purely from the
     parameter values, not from a non-Lipschitz update MLP.
  3. Run the rollout with the unified-extracted params: condition **A**.
  4. Override τ ← gt τ (keeping unified-extracted W and V_rest): condition
     **B**.

Logic of the test
-----------------
Inspecting learned_ode_params.pt for unified_cv03 reveals **τ_max = 10.000
(saturated at the extraction upper bound) vs gt τ_max = 0.316** — i.e. many
neurons got τ extracted as ~30× too large. Hypothesis: the bad τ values
make the leak too slow relative to the learned synaptic gain in W, so the
recurrent loop gain ends up > 1 and a tiny perturbation runs away.

Reading the result
------------------
* If A explodes (clamps at ±100) and B is healthy → τ extraction is the
  trigger; gt τ rescues the dynamics; the explosion is a τ-identifiability
  bug, not a fundamental W problem.
* If A explodes and B *also* explodes → W (or V_rest) is bad too; τ alone
  isn't the cause.
* If A is fine and the unified rollout still explodes → the explosion is
  intrinsic to the unified MLP (escaping its training regime), not the
  parameter values per se.

Caveats
-------
Known_ode uses g_phi=ReLU on source voltages; unified's actual g_phi is a
learned MLP. So this is *not* a faithful re-simulation of unified — it's a
test of whether the parameter *values* alone constitute a stable system in
the simplest plausible integrator. That's still informative: if the values
are themselves unstable, no amount of MLP expressiveness can save them.

Usage
-----
    conda run -n neural-graph-linux \\
        python figures/rollout_unified_extracted_in_known_ode_cv03.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/allierc/GraphData
# Configs        : <DATA_ROOT>/config/fly/flyvis_noise_free_blank50_known_ode_cv03.yaml
#                  (skeleton model only — checkpoint is NOT loaded; params come from the unified run)
# Training data  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_blank50_cv03/{edge_index.pt, ode_params.pt}
# Test data      : <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_blank50_cv03/x_list_test/
#                  <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_blank50_cv03/y_list_test/
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_known_ode_cv03/training_edges.pt
#                  <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_unified_cv03/results/learned_ode_params.pt
# Eval logs      : <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_unified_cv03/results/rollout_bundle.npz
# Output         : <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_unified_cv03/results/rollout_bundle_extracted_in_KO.npz
#                  <DATA_ROOT>/log/fly/flyvis_noise_free_blank50_unified_cv03/results/rollout_bundle_extracted_in_KO_gt_tau.npz
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange


REPO_ROOT = '/workspace/connectome-gnn'
DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'

for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from connectome_gnn.config import NeuralGraphConfig                 # noqa: E402
from connectome_gnn.utils import set_data_root                      # noqa: E402
from connectome_gnn.utils import compute_trace_metrics, fisher_pool # noqa: E402
import connectome_gnn.models.registry as _reg                       # noqa: E402
from connectome_gnn.models.registry import create_model             # noqa: E402
_reg._discover_models()
from connectome_gnn.models.utils import _batch_frames                # noqa: E402
from connectome_gnn.zarr_io import (                                 # noqa: E402
    load_simulation_data, load_raw_array,
)
from connectome_gnn.generators.ode_params import load_edge_index    # noqa: E402


UNIFIED_NAME   = 'flyvis_noise_free_blank50_unified_cv03'
KO_NAME        = 'flyvis_noise_free_blank50_known_ode_cv03'
KO_YAML        = f'{DATA_ROOT}/config/fly/{KO_NAME}.yaml'
UNIFIED_PARAMS = f'{DATA_ROOT}/log/fly/{UNIFIED_NAME}/results/learned_ode_params.pt'
DATASET        = 'flyvis_noise_free_blank50_cv03'
DATA_DIR       = f'{DATA_ROOT}/graphs_data/fly/{DATASET}'
KO_LOG_DIR     = f'{DATA_ROOT}/log/fly/{KO_NAME}'
OUT_DIR        = f'{DATA_ROOT}/log/fly/{UNIFIED_NAME}/results'   # save next to unified results

OUTLIER_TYPES = ['Tm4', 'C3', 'CT1(M10)', 'Mi12', 'T4d', 'Tm30']


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse softplus."""
    return torch.where(
        y > 20.0, y,
        torch.log(torch.expm1(torch.clamp(y, min=1e-12))),
    )


def build_known_ode_skeleton(device):
    """Build an *empty* FlyvisKnownODE with the right shapes for cv03.

    We don't load the trained known_ode checkpoint — we want random init that
    we'll immediately overwrite with the unified-extracted params.
    """
    cfg = NeuralGraphConfig.from_yaml(KO_YAML)
    parent = os.path.basename(os.path.dirname(os.path.abspath(KO_YAML)))
    if not cfg.dataset.startswith(parent + '/'):
        cfg.dataset = parent + '/' + cfg.dataset
    if cfg.config_file == 'none':
        cfg.config_file = parent + '/' + os.path.splitext(os.path.basename(KO_YAML))[0]

    sim = cfg.simulation
    training_edges_path = f'{KO_LOG_DIR}/training_edges.pt'
    if os.path.exists(training_edges_path):
        edges_for_size = torch.load(training_edges_path, map_location='cpu',
                                    weights_only=False)
    else:
        edges_for_size = load_edge_index(DATA_DIR, device='cpu')
    actual_n_edges = edges_for_size.shape[1]
    expected_total = sim.n_edges + sim.n_extra_null_edges
    if actual_n_edges == expected_total and sim.n_extra_null_edges > 0:
        cfg.simulation.n_edges = actual_n_edges
        cfg.simulation.n_extra_null_edges = 0
    elif actual_n_edges != sim.n_edges:
        cfg.simulation.n_edges = actual_n_edges

    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type,
                         config=cfg, device=device).to(device)
    model.eval()
    return cfg, model


def load_test_data(cfg, device):
    sim = cfg.simulation
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
    edges_path = f'{KO_LOG_DIR}/training_edges.pt'
    if os.path.exists(edges_path):
        edges = torch.load(edges_path, map_location=device, weights_only=False)
    else:
        edges = load_edge_index(DATA_DIR, device=device)
    return x_ts, y_ts, edges, type_ids


@torch.no_grad()
def run_rollout(model, x_ts, edges, sim, device, label):
    n_eval = x_ts.n_frames
    n_neurons = x_ts.n_neurons
    data_id = torch.zeros((n_neurons, 1), dtype=torch.int, device=device)
    x = x_ts.frame(0)
    pred_list, true_list, stim_list = [], [], []
    for k in trange(n_eval - 1, ncols=100, desc=f'rollout {label}'):
        pred_list.append(x.voltage.detach().cpu().numpy().astype(np.float32))
        true_list.append(x_ts.frame(k).voltage.detach().cpu().numpy().astype(np.float32))
        x.stimulus = x_ts.frame(k).stimulus.clone()
        stim_list.append(x.stimulus.detach().cpu().numpy().astype(np.float32))
        bs, be = _batch_frames([x], edges)
        pred, _, _ = model(bs, be, data_id=data_id, return_all=True)
        x.voltage = (x.voltage.unsqueeze(-1) + sim.delta_t * pred).squeeze(-1)
        x.voltage = torch.clamp(x.voltage, -100.0, 100.0)
    return (np.stack(pred_list, axis=0).T,
            np.stack(true_list, axis=0).T,
            np.stack(stim_list, axis=0).T)


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


def fmt_row(label, m):
    return (f"  {label:<32s}  "
            f"r={m['pearson_mean']:+.4f}±{m['pearson_sd']:.4f}  "
            f"RMSE={m['rmse_mean']:8.3f}  "
            f"|max|={m['max_abs_pred']:8.2f}  "
            f"clamped={m['frac_clamped']*100:5.1f}%")


def first_explosion_frame(pred, thresh=10.0):
    abs_pred = np.abs(pred)
    hits = abs_pred > thresh
    any_hit = hits.any(axis=0)
    return int(np.argmax(any_hit)) if any_hit.any() else -1


def main():
    set_data_root(DATA_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # --- skeleton known_ode model ---
    cfg, model = build_known_ode_skeleton(device)
    x_ts, _y_ts, edges, type_ids = load_test_data(cfg, device)
    n_neurons = x_ts.n_neurons
    type_names = list(np.load(f'{DATA_ROOT}/log/fly/{UNIFIED_NAME}/results/'
                              f'rollout_bundle.npz', allow_pickle=True)['type_names'])
    print(f'test data: {n_neurons} neurons × {x_ts.n_frames} frames; '
          f'{edges.shape[1]} edges')

    # --- unified-extracted params ---
    p = torch.load(UNIFIED_PARAMS, map_location=device, weights_only=False)
    ext_tau   = p['tau_i'].to(device).float()[:n_neurons]
    ext_vrest = p['V_i_rest'].to(device).float()[:n_neurons]
    ext_W     = p['W'].to(device).float()
    if ext_W.dim() == 1:
        ext_W = ext_W.unsqueeze(-1)
    print(f'\nunified-extracted params:')
    print(f'  τ     : min={ext_tau.min().item():.4f}  '
          f'max={ext_tau.max().item():.4f}  mean={ext_tau.mean().item():.4f}')
    print(f'  V_rest: min={ext_vrest.min().item():.4f}  '
          f'max={ext_vrest.max().item():.4f}  mean={ext_vrest.mean().item():.4f}')
    print(f'  W     : min={ext_W.min().item():.4f}  '
          f'max={ext_W.max().item():.4f}  mean={ext_W.mean().item():.4f}')

    # --- gt params ---
    gt = torch.load(f'{DATA_DIR}/ode_params.pt', map_location=device,
                    weights_only=False)
    gt_tau = gt['tau_i'].to(device).float()[:n_neurons]
    print(f'\ngt params:')
    print(f'  τ     : min={gt_tau.min().item():.4f}  '
          f'max={gt_tau.max().item():.4f}  mean={gt_tau.mean().item():.4f}')
    print(f'  |Δτ| max:    {(ext_tau - gt_tau).abs().max().item():.4f}')
    print(f'  |Δτ| median: {(ext_tau - gt_tau).abs().median().item():.4f}')
    n_over = ((ext_tau / gt_tau) > 2.0).sum().item()
    n_huge = (ext_tau >= 9.99).sum().item()
    print(f'  # neurons with ext_τ > 2× gt_τ : {n_over}/{n_neurons}')
    print(f'  # neurons with ext_τ ≥ 9.99 (saturated): {n_huge}/{n_neurons}')

    # ====================================================================
    # Condition A: known_ode with all unified-extracted params
    # ====================================================================
    print('\n' + '=' * 70)
    print('Condition A: known_ode integrator with unified-extracted (τ, V_rest, W)')
    print('=' * 70)
    with torch.no_grad():
        model.W.copy_(ext_W)
        model.V_rest.copy_(ext_vrest)
        model.raw_tau.copy_(inv_softplus(ext_tau))
    pred_A, true_A, stim_A = run_rollout(model, x_ts, edges, cfg.simulation,
                                          device, label='A')
    fA = first_explosion_frame(pred_A)
    print(f'  first frame |pred|>10: {fA}')

    # ====================================================================
    # Condition B: same W, V_rest; replace τ with gt
    # ====================================================================
    print('\n' + '=' * 70)
    print('Condition B: extracted W, V_rest + GT τ')
    print('=' * 70)
    with torch.no_grad():
        model.raw_tau.copy_(inv_softplus(gt_tau))
    pred_B, true_B, stim_B = run_rollout(model, x_ts, edges, cfg.simulation,
                                          device, label='B')
    fB = first_explosion_frame(pred_B)
    print(f'  first frame |pred|>10: {fB}')

    # ====================================================================
    # Save bundles
    # ====================================================================
    out_A = f'{OUT_DIR}/rollout_bundle_extracted_in_KO.npz'
    out_B = f'{OUT_DIR}/rollout_bundle_extracted_in_KO_gt_tau.npz'
    np.savez(out_A, activity_true=true_A, activity_pred=pred_A, stimulus=stim_A,
             type_ids=type_ids, type_names=np.array(type_names))
    np.savez(out_B, activity_true=true_B, activity_pred=pred_B, stimulus=stim_B,
             type_ids=type_ids, type_names=np.array(type_names))
    print(f'\nsaved:\n  {out_A}\n  {out_B}')

    # ====================================================================
    # Compare
    # ====================================================================
    print('\n=== Global metrics ===')
    mA = metrics(pred_A, true_A)
    mB = metrics(pred_B, true_B)
    print(fmt_row('A: ext (W, V_rest, τ)', mA))
    print(fmt_row('B: ext (W, V_rest) + gt τ', mB))

    # Reference: load unified rollout bundle so we can put it side-by-side
    ub = np.load(f'{DATA_ROOT}/log/fly/{UNIFIED_NAME}/results/rollout_bundle.npz',
                 allow_pickle=True)
    pred_U = np.asarray(ub['activity_pred']); true_U = np.asarray(ub['activity_true'])
    mU = metrics(pred_U, true_U)
    print(fmt_row('reference: unified rollout', mU))

    # Per-cell-type for the τ outliers
    print('\n=== Per cell-type (τ-outlier types only) ===')
    for name in OUTLIER_TYPES:
        if name not in type_names:
            continue
        tid = type_names.index(name)
        mask = type_ids == tid
        if mask.sum() == 0:
            continue
        mAt = metrics(pred_A[mask], true_A[mask])
        mBt = metrics(pred_B[mask], true_B[mask])
        # extracted τ vs gt τ for this type
        tau_e_t = ext_tau[mask].mean().item()
        tau_g_t = gt_tau[mask].mean().item()
        print(f'  {name}  (mean ext_τ={tau_e_t:.3f}, gt_τ={tau_g_t:.3f}):')
        print(fmt_row('    A (ext all)         ', mAt))
        print(fmt_row('    B (ext W,Vr + gt τ) ', mBt))


if __name__ == '__main__':
    main()
