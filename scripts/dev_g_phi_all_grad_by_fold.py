"""Per-fold summary: sensitivity of g_phi to ALL FOUR inputs (vi, vj, ai, aj),
not just vi/vj. ai/aj are embedding_dim-vectors (dim=2 here), so their
"gradient magnitude" is the L2 norm of d(g_phi)/d(a) per point.

Extends dev_g_phi_vi_sensitivity_by_fold.py — same real (edge, frame)
sampling, one autograd.grad call for all four inputs at once.
"""
import os
import sys

import numpy as np
import torch

# Lives in scripts/ but the package and its sibling dev_ scripts are at the
# repo root, so both go on the path: `src` for connectome_gnn, the root itself
# for `dev_g_phi_vi_vj`, which is still there.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))
sys.path.insert(0, _REPO_ROOT)

from connectome_gnn.utils import set_data_root, to_numpy
from dev_g_phi_vi_vj import load_run


def compute_all_grads(model, config, edges, x_ts, n_frames=32, seed=0):
    g_phi_positive = config.graph_model.g_phi_positive
    device = model.a.device

    src = edges[0].to(device)
    dst = edges[1].to(device)
    n_edges = edges.shape[1]
    emb_dim = model.a.shape[1]

    rng = np.random.default_rng(seed)
    n_frames = min(n_frames, x_ts.n_frames)
    frame_idx = rng.choice(x_ts.n_frames, size=n_frames, replace=False)

    voltage = x_ts.voltage.to(device)
    ai_fixed = model.a[dst]   # (E, emb_dim) — same real embeddings every frame
    aj_fixed = model.a[src]

    grad_vi_all = torch.zeros(n_edges, n_frames, device=device)
    grad_vj_all = torch.zeros(n_edges, n_frames, device=device)
    grad_ai_norm_all = torch.zeros(n_edges, n_frames, device=device)
    grad_aj_norm_all = torch.zeros(n_edges, n_frames, device=device)

    for f_idx, k in enumerate(frame_idx):
        vj_k = voltage[k, src].clone().detach().requires_grad_(True)
        vi_k = voltage[k, dst].clone().detach().requires_grad_(True)
        ai_k = ai_fixed.clone().detach().requires_grad_(True)
        aj_k = aj_fixed.clone().detach().requires_grad_(True)

        in_features = torch.cat([vi_k.unsqueeze(1), vj_k.unsqueeze(1), ai_k, aj_k], dim=1)
        out = model.g_phi(in_features.float())
        if g_phi_positive:
            out = out ** 2

        grad_vi, grad_vj, grad_ai, grad_aj = torch.autograd.grad(
            out.sum(), [vi_k, vj_k, ai_k, aj_k], retain_graph=False, create_graph=False)

        grad_vi_all[:, f_idx] = grad_vi.detach()
        grad_vj_all[:, f_idx] = grad_vj.detach()
        grad_ai_norm_all[:, f_idx] = grad_ai.detach().norm(dim=1)
        grad_aj_norm_all[:, f_idx] = grad_aj.detach().norm(dim=1)

    return (to_numpy(grad_vi_all), to_numpy(grad_vj_all),
            to_numpy(grad_ai_norm_all), to_numpy(grad_aj_norm_all))


def main():
    set_data_root('/groups/saalfeld/home/allierc/GraphData')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    R2 = {'00': 0.5580, '01': 0.9340, '02': 0.7855, '03': 0.8058, '04': 0.9290}

    print(f'{"fold":<6}{"R2_W":>8}{"|dvi|":>10}{"|dvj|":>10}{"|dai|":>10}{"|daj|":>10}'
         f'{"dvi/dvj":>10}{"dai/dvj":>10}{"daj/dvj":>10}')
    rows = []
    for cv in ['00', '01', '02', '03', '04']:
        cfg = f'flyvis_current_noise_005_conductance_nominal_cv{cv}'
        log_dir = f'/groups/saalfeld/home/allierc/GraphData/log/fly/{cfg}'
        config, model, edges, x_ts = load_run(log_dir, cfg, device)

        grad_vi, grad_vj, grad_ai, grad_aj = compute_all_grads(model, config, edges, x_ts, n_frames=32, seed=0)

        m_vi, m_vj = np.abs(grad_vi).mean(), np.abs(grad_vj).mean()
        m_ai, m_aj = grad_ai.mean(), grad_aj.mean()   # already norms, non-negative

        print(f'{cv:<6}{R2[cv]:>8.3f}{m_vi:>10.4f}{m_vj:>10.4f}{m_ai:>10.4f}{m_aj:>10.4f}'
             f'{m_vi/m_vj:>10.4f}{m_ai/m_vj:>10.4f}{m_aj/m_vj:>10.4f}')
        rows.append((cv, R2[cv], m_vi/m_vj, m_ai/m_vj, m_aj/m_vj))

        del model
        torch.cuda.empty_cache()

    r2s = [r[1] for r in rows]
    print()
    print('correlation(R2_W, dvi/dvj):', np.corrcoef(r2s, [r[2] for r in rows])[0, 1])
    print('correlation(R2_W, dai/dvj):', np.corrcoef(r2s, [r[3] for r in rows])[0, 1])
    print('correlation(R2_W, daj/dvj):', np.corrcoef(r2s, [r[4] for r in rows])[0, 1])


if __name__ == '__main__':
    main()
