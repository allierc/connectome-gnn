"""Per-fold summary: how much does g_phi actually depend on vi (not just vj),
and does that correlate with W_corrected R^2 degradation?

For each conductance fold, samples real (edge, frame) points and computes
BOTH d(g_phi)/d(vi) and d(g_phi)/d(vj) via autograd at the same points
(extends compute_g_phi_edge_grad, which only computes the vj one). Reports
mean(|d/dvi|) / mean(|d/dvj|) as a normalized vi-sensitivity ratio: ~0 means
g_phi collapsed to a vj-only function (matches true ReLU(vj)); larger means
real, non-collapsed vi-dependence.
"""
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from connectome_gnn.utils import set_data_root, to_numpy
from dev_g_phi_vi_vj import load_run


def compute_both_grads(model, config, edges, x_ts, n_frames=32, seed=0):
    signal_model_name = config.graph_model.signal_model_name
    g_phi_positive = config.graph_model.g_phi_positive
    device = model.a.device

    src = edges[0].to(device)
    dst = edges[1].to(device)
    n_edges = edges.shape[1]

    rng = np.random.default_rng(seed)
    n_frames = min(n_frames, x_ts.n_frames)
    frame_idx = rng.choice(x_ts.n_frames, size=n_frames, replace=False)

    voltage = x_ts.voltage.to(device)
    ai = model.a[dst]
    aj = model.a[src]

    grad_vi_all = torch.zeros(n_edges, n_frames, device=device)
    grad_vj_all = torch.zeros(n_edges, n_frames, device=device)

    for f_idx, k in enumerate(frame_idx):
        vj_k = voltage[k, src].clone().detach().requires_grad_(True)
        vi_k = voltage[k, dst].clone().detach().requires_grad_(True)

        in_features = torch.cat([vi_k.unsqueeze(1), vj_k.unsqueeze(1), ai, aj], dim=1)
        out = model.g_phi(in_features.float())
        if g_phi_positive:
            out = out ** 2

        grad_vi, grad_vj = torch.autograd.grad(out.sum(), [vi_k, vj_k], retain_graph=False, create_graph=False)
        grad_vi_all[:, f_idx] = grad_vi.detach()
        grad_vj_all[:, f_idx] = grad_vj.detach()

    return to_numpy(grad_vi_all), to_numpy(grad_vj_all)


def main():
    set_data_root('/groups/saalfeld/home/allierc/GraphData')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    R2 = {'00': 0.5580, '01': 0.9340, '02': 0.7855, '03': 0.8058, '04': 0.9290}

    print(f'{"fold":<6}{"R2_W":>8}{"mean|dvi|":>12}{"mean|dvj|":>12}{"ratio(vi/vj)":>15}{"frac vi>0.1*vj":>16}')
    rows = []
    for cv in ['00', '01', '02', '03', '04']:
        cfg = f'flyvis_noise_005_conductance_cv{cv}'
        log_dir = f'/groups/saalfeld/home/allierc/GraphData/log/fly/{cfg}'
        config, model, edges, x_ts = load_run(log_dir, cfg, device)

        grad_vi, grad_vj = compute_both_grads(model, config, edges, x_ts, n_frames=32, seed=0)

        mean_abs_vi = np.abs(grad_vi).mean()
        mean_abs_vj = np.abs(grad_vj).mean()
        ratio = mean_abs_vi / mean_abs_vj if mean_abs_vj > 0 else float('nan')
        frac_vi_matters = (np.abs(grad_vi) > 0.1 * np.abs(grad_vj)).mean()

        print(f'{cv:<6}{R2[cv]:>8.3f}{mean_abs_vi:>12.4f}{mean_abs_vj:>12.4f}{ratio:>15.4f}{frac_vi_matters:>16.3f}')
        rows.append((cv, R2[cv], ratio, frac_vi_matters))

        del model
        torch.cuda.empty_cache()

    print()
    print('correlation(R2_W, vi/vj ratio):',
         np.corrcoef([r[1] for r in rows], [r[2] for r in rows])[0, 1])
    print('correlation(R2_W, frac vi matters):',
         np.corrcoef([r[1] for r in rows], [r[3] for r in rows])[0, 1])


if __name__ == '__main__':
    main()
