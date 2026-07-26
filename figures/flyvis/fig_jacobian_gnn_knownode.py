"""Vzfg Q4: Jacobian of the GNN and the Known-ODE against ground truth.

Supp. Fig. 18 computes a Jacobian mismatch for the recurrent MLP only. This
script applies the same test to our own two models and reports it quantitatively.

Quantities (Eq. 1 ground truth)
    J_ij = W_ij * 1[v_j > 0] / tau_i        (j != i)
    J_ii = -1 / tau_i
    S_ik = delta_ik / tau_i                 (retinal neurons)

Known-ODE has the same closed form in (W_hat, tau_hat) -- analytic, no autograd.

GNN (Eq. 2):  pred_i = f_theta([v_i, a_i, m_i, I_i]),
              m_i    = sum_j W_ij * g_phi([v_j, a_j])^2
so by the chain rule
    J_ij = (df/dm)_i * W_ij * 2 g_phi_ij * (dg/dv)_ij     (j != i)
    J_ii = (df/dv)_i
    S_ik = (df/dI)_i * delta_ik
Every term is a per-neuron or per-edge scalar, so no dense 13,741^2 Jacobian is
built: J is evaluated on the 434,112 connectome edges. Autograd throughout --
finite differences are unusable because ReLU kinks make them unstable.

All models are evaluated on the SAME frames sampled from the held-out set.

Usage
-----
    GNN_OUTPUT_ROOT=/groups/.../GraphData PYTHONPATH=src \
      python figures/flyvis/fig_jacobian_gnn_knownode.py [--n-frames 200]

Output
------
    figures/flyvis/results_jacobian_gnn_knownode.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'src'))

from connectome_gnn.config import NeuralGraphConfig                    # noqa: E402
from connectome_gnn.models.utils import restore_edge_sign_lock         # noqa: E402
from connectome_gnn.models.registry import create_model                 # noqa: E402
from connectome_gnn.generators.ode_params import get_ode_params_class   # noqa: E402
from connectome_gnn.utils import (set_data_root, graphs_data_path, log_path,  # noqa: E402
                                  to_numpy, add_pre_folder, migrate_state_dict,
                                  sort_key)
from connectome_gnn.zarr_io import load_simulation_data                 # noqa: E402
from connectome_gnn.neuron_state import NeuronState                     # noqa: E402

GNN_CFG = 'flyvis_noise_free_blank50_unified_cv00'
ODE_CFG = 'flyvis_noise_free_blank50_known_ode_cv00'
SEED = 0

# Recurrent-MLP arm (Vzfg Q4, MLP row). Off by default: the MLP is unconstrained,
# so its Jacobian is dense (N x N) rather than supported on the connectome, which
# costs a chunked backward pass per frame. Supply --mlp-config <name> once the
# checkpoint is reachable; see neurips_review/HANDOFF_mlp_jacobian.md.
MLP_CHUNK = 128


# ---------------------------------------------------------------------------
def load_run(config_name, device):
    """Config + best checkpoint + ode_params, mirroring GNN_PlotFigure 'best'."""
    # The run's own config.yaml is the canonical record of what it trained on.
    root = os.environ.get('GNN_OUTPUT_ROOT', '/groups/saalfeld/home/allierc/GraphData')
    cfg_path = os.path.join(root, 'log', 'fly', config_name, 'config.yaml')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(root, 'config', 'fly', f'{config_name}.yaml')
    config = NeuralGraphConfig.from_yaml(cfg_path)
    _, pre = add_pre_folder(config_name)
    if not config.dataset.startswith(pre):
        config.dataset = pre + config.dataset
    config.config_file = pre + config_name

    gdata = graphs_data_path(config.dataset)
    OdeCls = get_ode_params_class(config.graph_model.signal_model_name)
    ode_params = OdeCls.load(gdata, device=device)

    ckpts = sorted(glob.glob(os.path.join(log_path(config.config_file), 'models',
                                          'best_model_with_*.pt')), key=sort_key)
    if not ckpts:
        raise FileNotFoundError(f'no checkpoint for {config_name}')
    state = torch.load(ckpts[-1], map_location=device, weights_only=False)
    migrate_state_dict(state)
    if 'W' in state.get('model_state_dict', {}):
        config.simulation.n_edges = state['model_state_dict']['W'].shape[0]
    model = create_model(config.graph_model.signal_model_name,
                         aggr_type=config.graph_model.aggr_type,
                         config=config, device=device)
    model.load_state_dict(state['model_state_dict'], strict=False)
    restore_edge_sign_lock(model, ode_params.W)
    model.eval()
    print(f'  {config_name}: {os.path.basename(ckpts[-1])}')
    return config, model, ode_params, gdata


def load_mlp(config_name, device):
    """Load a recurrent-MLP baseline run (same layout as the GNN runs)."""
    root = os.environ.get('GNN_OUTPUT_ROOT', '/groups/saalfeld/home/allierc/GraphData')
    log_dir = os.path.join(root, 'log', 'fly', config_name)
    config = NeuralGraphConfig.from_yaml(os.path.join(log_dir, 'config.yaml'))
    model = create_model(config.graph_model.signal_model_name,
                         aggr_type=config.graph_model.aggr_type,
                         config=config, device=device)
    ckpts = sorted(glob.glob(os.path.join(log_dir, 'models', 'best_model_with_*.pt')),
                   key=sort_key)
    if not ckpts:
        raise FileNotFoundError(f'no checkpoint for {config_name} under {log_dir}')
    state = torch.load(ckpts[-1], map_location=device, weights_only=False)
    migrate_state_dict(state)
    model.load_state_dict(state['model_state_dict'], strict=False)
    model.eval()
    print(f'  {config_name}: {os.path.basename(ckpts[-1])}')
    return config, model


def mlp_jacobian_dense(model, v, stim, chunk=MLP_CHUNK):
    """Dense (N, N) Jacobian of the MLP baseline on one frame.

    The MLP is not graph-constrained, so J has support everywhere and the whole
    matrix is needed to measure how much of it falls off the connectome.
    """
    N = v.shape[0]
    J = torch.empty((N, N), dtype=torch.float32, device='cpu')
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        vv = v.clone().detach().requires_grad_(True)
        dvdt = model.predict_dvdt(vv, stim)
        go = torch.zeros((i1 - i0, N), device=vv.device)
        for k, i in enumerate(range(i0, i1)):
            go[k, i] = 1.0
        g = torch.autograd.grad(dvdt, vv, grad_outputs=go, is_grads_batched=True)[0]
        J[i0:i1] = g.detach().cpu()
    return J


def sample_frames(gdata, n_frames, device):
    """n_frames voltages + stimuli from the held-out split, fixed seed."""
    x = load_simulation_data(os.path.join(gdata, 'x_list_test'))
    T = x.voltage.shape[0]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(T, size=min(n_frames, T), replace=False))
    return (x.voltage[idx].to(device), x.stimulus[idx].to(device), idx)


def _t(x, device):
    """Coerce numpy arrays / tensors to a float32 torch tensor on `device`."""
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
def gt_jacobian(ode_params, v, edge_index):
    """J on connectome edges and the diagonal, from Eq. 1."""
    src, dst = edge_index
    dev = v.device
    tau = _t(ode_params.gt_tau(v.shape[0]), dev).squeeze()
    W = _t(ode_params.W, dev).squeeze()
    J_edge = W * (v.squeeze()[src] > 0).float() / tau[dst]
    return J_edge, -1.0 / tau


def known_ode_jacobian(model, v, edge_index):
    """Same closed form, in the learned parameters."""
    src, dst = edge_index
    dev = v.device
    tau = _t(model.get_learned_tau(), dev).squeeze()
    W = _t(model.W, dev).squeeze().detach()   # KnownODE msg = W[e] * relu(v_src)
    J_edge = W * (v.squeeze()[src] > 0).float() / tau[dst]
    return J_edge, -1.0 / tau


def gnn_jacobian(model, state, edge_index):
    """Chain rule through f_theta and g_phi (autograd, not finite differences)."""
    src, dst = edge_index
    v = state.observable(model.calcium_type)
    emb = model.a[state.index.long()]
    if emb.dim() == 1:
        emb = emb.unsqueeze(-1)
    excitation = state.stimulus.unsqueeze(-1)

    # --- g_phi and dg/dv per edge -----------------------------------------
    v_src = v[src].detach().clone().requires_grad_(True)
    g_in = torch.cat([v_src, emb[src]], dim=1)
    g_out = model.g_phi(g_in)
    dg_dv = torch.autograd.grad(g_out.sum(), v_src)[0].squeeze()
    g_out = g_out.squeeze().detach()

    W_e = model._effective_edge_weights(
        torch.arange(edge_index.shape[1], device=v.device)
        % (model.n_edges + model.n_extra_null_edges)).squeeze().detach()

    # message, rebuilt exactly as the model does
    g_msg = g_out**2 if model.g_phi_positive else g_out
    edge_msg = (W_e * g_msg).unsqueeze(1)
    msg = torch.zeros_like(v)
    msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

    # --- partials of f_theta per neuron -----------------------------------
    emb_dim = emb.shape[1]
    f_in = torch.cat([v, emb, msg, excitation], dim=1).detach().clone().requires_grad_(True)
    pred = model.f_theta(f_in)
    grad = torch.autograd.grad(pred.sum(), f_in)[0]          # (N, 2 + emb_dim)
    df_dv = grad[:, 0]
    df_dm = grad[:, 1 + emb_dim]
    df_dI = grad[:, 2 + emb_dim]

    # d msg_i / d v_j on edge (j -> i); chain through g_phi^2 when squared
    dmsg_dv = W_e * (2.0 * g_out * dg_dv if model.g_phi_positive else dg_dv)
    J_edge = df_dm[dst] * dmsg_dv
    return J_edge.detach(), df_dv.detach(), df_dI.detach()


# ---------------------------------------------------------------------------
def metrics(J_hat, J_gt):
    """R^2, Pearson r and sign agreement of learned vs ground-truth J."""
    a, b = to_numpy(J_hat).ravel(), to_numpy(J_gt).ravel()
    ss_res = float(np.sum((b - a) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
    nz = b != 0
    sign = float(np.mean(np.sign(a[nz]) == np.sign(b[nz]))) if nz.any() else float('nan')
    return dict(r2=r2, pearson_r=r, sign_agreement=sign)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-frames', type=int, default=200)
    ap.add_argument('--mlp-config', default=None,
                    help='recurrent-MLP run name; adds the MLP row (slow: dense Jacobian)')
    ap.add_argument('--mlp-frames', type=int, default=20,
                    help='frames for the MLP arm (dense J per frame is expensive)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    set_data_root(os.environ.get('GNN_OUTPUT_ROOT',
                                 '/groups/saalfeld/home/allierc/GraphData'))
    dev = torch.device(args.device)
    print(f'device={dev}  n_frames={args.n_frames}')

    print('loading runs ...')
    cfg_g, gnn, ode_g, gdata = load_run(GNN_CFG, dev)
    cfg_o, ode_model, ode_o, _ = load_run(ODE_CFG, dev)

    edge_index = ode_g.edge_index.to(dev)
    volt, stim, idx = sample_frames(gdata, args.n_frames, dev)
    n_neurons = volt.shape[1]
    print(f'  N={n_neurons}  E={edge_index.shape[1]}  frames={len(idx)}')

    acc = {k: [] for k in ('gt_e', 'gnn_e', 'ode_e', 'gt_d', 'gnn_d', 'ode_d')}
    for t in range(volt.shape[0]):
        v = volt[t].unsqueeze(-1).to(dev)
        state = NeuronState(voltage=v.squeeze(-1), stimulus=stim[t].to(dev),
                            index=torch.arange(n_neurons, device=dev))

        Je_gt, Jd_gt = gt_jacobian(ode_g, v, edge_index)
        Je_od, Jd_od = known_ode_jacobian(ode_model, v, edge_index)
        Je_gn, Jd_gn, _ = gnn_jacobian(gnn, state, edge_index)

        acc['gt_e'].append(Je_gt); acc['ode_e'].append(Je_od); acc['gnn_e'].append(Je_gn)
        acc['gt_d'].append(Jd_gt); acc['ode_d'].append(Jd_od); acc['gnn_d'].append(Jd_gn)
        if (t + 1) % 25 == 0:
            print(f'  frame {t+1}/{volt.shape[0]}')

    mean = {k: torch.stack(v).mean(0) for k, v in acc.items()}

    out = {
        'n_frames': int(len(idx)), 'n_neurons': int(n_neurons),
        'n_edges': int(edge_index.shape[1]),
        'gnn_config': GNN_CFG, 'known_ode_config': ODE_CFG,
        'edges': {'known_ode': metrics(mean['ode_e'], mean['gt_e']),
                  'gnn': metrics(mean['gnn_e'], mean['gt_e'])},
        'diagonal': {'known_ode': metrics(mean['ode_d'], mean['gt_d']),
                     'gnn': metrics(mean['gnn_d'], mean['gt_d'])},
        # J is evaluated on connectome edges only, so off-support mass is 0 by
        # construction for both models. Recorded explicitly rather than implied.
        'off_support_mass': {'known_ode': 0.0, 'gnn': 0.0},
    }

    # ---- optional recurrent-MLP arm -------------------------------------
    if args.mlp_config:
        print(f'\nMLP arm: {args.mlp_config}  ({args.mlp_frames} frames, dense J)')
        _, mlp = load_mlp(args.mlp_config, dev)
        ei_np = to_numpy(edge_index)
        nf = min(args.mlp_frames, volt.shape[0])
        Jsum = torch.zeros((n_neurons, n_neurons), dtype=torch.float32)
        for t in range(nf):
            Jsum += mlp_jacobian_dense(mlp, volt[t].to(dev), stim[t].to(dev))
            if (t + 1) % 5 == 0:
                print(f'  MLP frame {t+1}/{nf}')
        Jm = (Jsum / nf).numpy()
        on_edge = Jm[ei_np[1], ei_np[0]]                    # J[dst, src]
        diag = np.diag(Jm).copy()
        off_diag_total = np.abs(Jm).sum() - np.abs(diag).sum()
        on_graph = np.abs(on_edge).sum()
        out['edges']['recurrent_mlp'] = metrics(
            torch.as_tensor(on_edge), mean['gt_e'].cpu())
        out['diagonal']['recurrent_mlp'] = metrics(
            torch.as_tensor(diag), mean['gt_d'].cpu())
        out['off_graph_fraction'] = {
            'known_ode': 0.0, 'gnn': 0.0,
            'recurrent_mlp': float(1.0 - on_graph / off_diag_total)}
        out['mlp_config'] = args.mlp_config
        out['mlp_n_frames'] = int(nf)

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'results_jacobian_gnn_knownode.json')
    with open(dst, 'w') as f:
        json.dump(out, f, indent=2)

    print(f'\n{"":<12}{"R2 vs GT J":>12}{"Pearson r":>12}{"sign agr":>10}')
    for blk in ('edges', 'diagonal'):
        print(f'-- {blk}')
        for m in ('known_ode', 'gnn'):
            d = out[blk][m]
            print(f'{m:<12}{d["r2"]:>12.4f}{d["pearson_r"]:>12.4f}{d["sign_agreement"]:>10.3f}')
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()
