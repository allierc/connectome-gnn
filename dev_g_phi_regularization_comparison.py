"""Does regularization help g_phi discard vi/ai? Flyvis_conductance's true
generative model is ReLU(vj) only (no vi/ai dependence) -- the extra
[vi, ai] capacity in g_phi's input is a hypothesis-test, not a known
necessity. Compares 3 regularization settings on the same "grok" config
(cv00's blank50 fold, extended to n_epochs=8 to watch for late/slow
convergence), each with checkpoints saved at epochs 0-4 so far:

    grok             coeff_g_phi_weight_L1=0.28 only (flat elementwise L1)
    grouplasso_grok  + coeff_g_phi_input_group_L1=2.0 (structured group lasso)
    nosparsity_grok  no g_phi sparsity regularization at all

Two independent signals per (series, epoch), so a real "learned to ignore
vi/ai" trend has to show up in both, not just one:

  1. First-layer weight discard score: L1 mass on the [vi, ai] input columns
     of g_phi.layers[0].weight, as a fraction of the total L1 mass across
     all 4 groups [vi, vj, ai, aj]. Inspects the learned filter directly --
     no forward pass, no data. (L1 here, not the L2 the group-lasso PENALTY
     itself needs for the group-sparsity property -- a post-hoc score has no
     such constraint.)
  2. Gradient-based dvi/dvj, dai/dvj ratios on real (edge, frame) pairs via
     autograd (compute_all_grads, same as dev_g_phi_all_grad_by_fold.py) --
     confirms the weight-level signal actually shows up in g_phi's behavior.

Usage:
    /workspace/.conda_envs/neural-graph-linux/bin/python dev_g_phi_regularization_comparison.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from connectome_gnn.generators.ode_params import FlyVisODEParams, get_ode_params_class
from connectome_gnn.metrics import W_OUTLIER_THRESH, compute_all_corrected_weights, recovery_param_metrics
from connectome_gnn.models.registry import create_model
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.utils import graphs_data_path, migrate_state_dict, set_data_root, to_numpy
from connectome_gnn.zarr_io import load_simulation_data
from dev_g_phi_all_grad_by_fold import compute_all_grads

_OUTPUT_ROOT = '/groups/saalfeld/home/allierc/GraphData'
_FIGURES_DIR = os.path.join(_REPO_ROOT, 'figures')

SERIES = [
    ('flyvis_noise_005_conductance_grok_cv00',            'L1 weights only (0.28)'),
    ('flyvis_noise_005_conductance_grouplasso_grok_cv00',  'L1 + group lasso (2.0)'),
    ('flyvis_noise_005_conductance_nosparsity_grok_cv00',  'no regularization'),
]
EPOCHS = [0, 1, 2, 3, 4]


def load_run_at_epoch(log_dir, config_name, epoch, device):
    """Same resolution as dev_g_phi_vi_vj.load_run, but for a specific
    mid-training checkpoint instead of hardcoding epoch 0."""
    config, _ = load_run_config(config_name, explicit_output_root=True, task='train')
    model_config = config.graph_model

    edges = torch.load(os.path.join(log_dir, 'training_edges.pt'), map_location=device, weights_only=False)

    net_path = os.path.join(log_dir, 'models', f'best_model_with_{config.training.n_runs - 1}_graphs_{epoch}.pt')
    state_dict = torch.load(net_path, map_location=device, weights_only=False)
    migrate_state_dict(state_dict)
    if 'W' in state_dict.get('model_state_dict', {}):
        config.simulation.n_edges = state_dict['model_state_dict']['W'].shape[0]
        config.simulation.n_extra_null_edges = 0

    model = create_model(model_config.signal_model_name, aggr_type=model_config.aggr_type,
                          config=config, device=device)
    model.load_state_dict(state_dict['model_state_dict'], strict=False)
    model.eval()

    x_path = graphs_data_path(config.dataset, 'x_list_train')
    if not os.path.exists(x_path):
        x_path = graphs_data_path(config.dataset, 'x_list_0')
    x_ts = load_simulation_data(x_path, fields=['index', 'voltage', 'stimulus']).to(device)

    return config, model, edges, x_ts


def first_layer_discard_score(model, emb_dim):
    """L1 mass on [vi, ai] (should be discarded) as a fraction of the total
    L1 mass across all 4 input groups of g_phi's first layer. 0 = filter has
    fully dropped vi/ai; 0.5 = no preference; matches a dot product of the
    per-group L1 norms against the discard mask [1, 0, 1, 0], normalized."""
    W0 = model.g_phi.layers[0].weight.detach()
    n_vi = W0[:, 0:1].abs().sum().item()
    n_vj = W0[:, 1:2].abs().sum().item()
    n_ai = W0[:, 2:2 + emb_dim].abs().sum().item()
    n_aj = W0[:, 2 + emb_dim:2 + 2 * emb_dim].abs().sum().item()
    total = n_vi + n_vj + n_ai + n_aj
    return (n_vi + n_ai) / total, (n_vi, n_vj, n_ai, n_aj)


def compute_r2_W(model, config, edges, x_ts, device, gt_weights):
    """R^2 of the corrected weights against ground truth -- same pipeline
    compute_all_corrected_weights/plot_synaptic use in production, just
    without the scatter plot (recovery_param_metrics is the R^2 itself)."""
    n_neurons = model.a.shape[0]
    try:
        OdeParamsCls = get_ode_params_class(config.graph_model.signal_model_name)
    except KeyError:
        OdeParamsCls = FlyVisODEParams
    ode_params_path = graphs_data_path(config.dataset, 'ode_params.pt')
    ode_params = (OdeParamsCls.load(graphs_data_path(config.dataset), device='cpu')
                  if os.path.exists(ode_params_path) else OdeParamsCls())
    gt_w_full = ode_params.effective_true_weights(to_numpy(gt_weights), to_numpy(edges), n_neurons)
    corrected_W, _, _, _, _ = compute_all_corrected_weights(model, config, edges, x_ts, device, ode_params=ode_params)
    learned_w = to_numpy(corrected_W.squeeze())
    m = recovery_param_metrics(gt_w_full, learned_w, W_OUTLIER_THRESH)
    return m['r2_clean']


def main():
    set_data_root(_OUTPUT_ROOT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(_FIGURES_DIR, exist_ok=True)

    weight_scores = {cfg: [] for cfg, _ in SERIES}
    grad_ratios_vi = {cfg: [] for cfg, _ in SERIES}
    grad_ratios_ai = {cfg: [] for cfg, _ in SERIES}

    for cfg, label in SERIES:
        log_dir = os.path.join(_OUTPUT_ROOT, 'log', 'fly', cfg)
        print(f'=== {label} ({cfg}) ===')
        for ep in EPOCHS:
            ckpt = os.path.join(log_dir, 'models', f'best_model_with_0_graphs_{ep}.pt')
            if not os.path.isfile(ckpt):
                print(f'  epoch {ep}: no checkpoint yet, skipping')
                weight_scores[cfg].append(np.nan)
                grad_ratios_vi[cfg].append(np.nan)
                grad_ratios_ai[cfg].append(np.nan)
                continue

            config, model, edges, x_ts = load_run_at_epoch(log_dir, cfg, ep, device)
            emb_dim = model.a.shape[1]

            score, (n_vi, n_vj, n_ai, n_aj) = first_layer_discard_score(model, emb_dim)
            weight_scores[cfg].append(score)

            grad_vi, grad_vj, grad_ai, grad_aj = compute_all_grads(model, config, edges, x_ts, n_frames=32, seed=0)
            m_vi, m_vj = np.abs(grad_vi).mean(), np.abs(grad_vj).mean()
            m_ai, m_aj = grad_ai.mean(), grad_aj.mean()
            ratio_vi = m_vi / m_vj if m_vj > 0 else np.nan
            ratio_ai = m_ai / m_vj if m_vj > 0 else np.nan
            grad_ratios_vi[cfg].append(ratio_vi)
            grad_ratios_ai[cfg].append(ratio_ai)

            print(f'  epoch {ep}: weight discard score {score:.3f}  '
                  f'(|vi|={n_vi:.3f} |vj|={n_vj:.3f} |ai|={n_ai:.3f} |aj|={n_aj:.3f})   '
                  f'grad dvi/dvj={ratio_vi:.3f}  dai/dvj={ratio_ai:.3f}')

            del model
            if device == 'cuda':
                torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='white')
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for (cfg, label), c in zip(SERIES, colors):
        axes[0].plot(EPOCHS, weight_scores[cfg], 'o-', color=c, label=label)
        axes[1].plot(EPOCHS, grad_ratios_vi[cfg], 'o-', color=c, label=label)
        axes[2].plot(EPOCHS, grad_ratios_ai[cfg], 'o-', color=c, label=label)

    axes[0].set_title('first-layer weight discard score\n(L1 mass on [vi,ai] / total, lower = better)')
    axes[1].set_title('gradient ratio |d g_phi/d vi| / |d g_phi/d vj|\n(real data, lower = better)')
    axes[2].set_title('gradient ratio |d g_phi/d ai| / |d g_phi/d vj|\n(real data, lower = better)')
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(_FIGURES_DIR, 'g_phi_regularization_comparison_grok.png')
    plt.savefig(out_path, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'\nsaved: {out_path}')


if __name__ == '__main__':
    main()
