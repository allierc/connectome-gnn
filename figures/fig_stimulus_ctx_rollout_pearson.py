"""Stimulus-baseline context sweep: per-neuron Pearson r over an 8k-step rollout.

For each context length in CTXS:
    - load the trained flyvis_noise_free_stimulus_ctx{ctx} model
    - load each of the 5 noise-free YT test splits (cv00..cv04)
    - compute per-neuron Pearson r between predicted and true voltage trace
    - collect distribution across (neuron x cv)
    - violin plot with Fisher z-transform mean and 95% CI

Output: figures/fig_stimulus_ctx_rollout_pearson.{pdf,png}
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib.pyplot as plt
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.registry import create_model


CTXS = [1, 2, 4, 8, 16, 32]
CV_IDS = ['00', '01', '02', '03', '04']
LOG_ROOT = Path('/groups/saalfeld/home/kumarv4/repos/connectome-gnn/log/fly')
CV_ROOT = Path('/groups/saalfeld/home/allierc/GraphData/graphs_data/fly')
OUT_BASE = REPO / 'figures' / 'fig_stimulus_ctx_rollout_pearson'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Font family (global); keep default spines, no figure_style.py.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Nimbus Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
})

FS_AXIS = 9
FS_TICK = 8
FS_TITLE = 10


def load_test_splits():
    voltage_list, stimulus_list = [], []
    for cv in CV_IDS:
        d = CV_ROOT / f'flyvis_noise_free_yt_cv{cv}' / 'x_list_test'
        v = np.asarray(zarr.open(str(d / 'voltage.zarr'), mode='r')[:], dtype=np.float32)
        s = np.asarray(zarr.open(str(d / 'stimulus.zarr'), mode='r')[:], dtype=np.float32)
        voltage_list.append(v)
        stimulus_list.append(s)
    return voltage_list, stimulus_list


def load_ctx_model(ctx):
    cfg_path = REPO / 'config' / 'fly' / f'flyvis_noise_free_stimulus_ctx{ctx}.yaml'
    cfg = NeuralGraphConfig.from_yaml(str(cfg_path))
    cfg.config_file = f'fly/flyvis_noise_free_stimulus_ctx{ctx}'

    log_dir = LOG_ROOT / f'flyvis_noise_free_stimulus_ctx{ctx}'
    ckpts = sorted(glob.glob(f'{log_dir}/models/best_model_with_*.pt'),
                   key=os.path.getmtime)
    if not ckpts:
        ckpts = sorted(glob.glob(f'{log_dir}/models/*.pt'), key=os.path.getmtime)
    assert ckpts, f'no checkpoint for ctx={ctx} at {log_dir}'

    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type,
                         config=cfg, device=DEVICE).to(DEVICE)
    sd = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd['model_state_dict'], strict=False)
    model.eval()
    return cfg, model


@torch.no_grad()
def per_neuron_pearson(cfg, model, voltage, stimulus):
    """Per-neuron Pearson r across the full rollout. One shot on GPU."""
    tw = cfg.training.time_window
    n_in = cfg.simulation.n_input_neurons
    T = voltage.shape[0]
    stim = torch.from_numpy(stimulus[:, :n_in]).to(DEVICE)
    volt = torch.from_numpy(voltage).to(DEVICE)

    valid = torch.arange(tw - 1, T, device=DEVICE)
    idx = valid[:, None] + torch.arange(-tw + 1, 1, device=DEVICE)[None, :]
    ctx = stim[idx]
    pred = model.predict_voltage(ctx).double()
    tgt = volt[valid].double()

    p = pred - pred.mean(dim=0, keepdim=True)
    g = tgt - tgt.mean(dim=0, keepdim=True)
    num = (p * g).sum(dim=0)
    den = torch.sqrt((p ** 2).sum(dim=0) * (g ** 2).sum(dim=0)).clamp_min(1e-24)
    return (num / den).cpu().numpy()


def fisher_mean_ci(rs, alpha=0.05):
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    z_bar = z.mean()
    se = z.std(ddof=1) / np.sqrt(len(z))
    zcrit = norm.ppf(1 - alpha / 2)
    return np.tanh(z_bar), np.tanh(z_bar - zcrit * se), np.tanh(z_bar + zcrit * se)


def compute_all():
    voltage_list, stimulus_list = load_test_splits()
    print(f'[data] {len(CV_IDS)} splits, N={voltage_list[0].shape[1]}')

    pearson_r = {}
    for ctx in CTXS:
        cfg, model = load_ctx_model(ctx)
        rs = np.stack([per_neuron_pearson(cfg, model, v, s)
                       for v, s in zip(voltage_list, stimulus_list)], axis=0)
        pearson_r[ctx] = rs
        del model
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        flat = rs[np.isfinite(rs)]
        m, lo, hi = fisher_mean_ci(flat)
        print(f'ctx={ctx:>2d}: fisher r = {m:.4f} [{lo:.4f}, {hi:.4f}]')
    return pearson_r


def plot(pearson_r):
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)

    data = [pearson_r[c][np.isfinite(pearson_r[c])].ravel() for c in CTXS]
    parts = ax.violinplot(data, positions=CTXS, widths=[c * 0.6 for c in CTXS],
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.4)
        pc.set_facecolor('steelblue')

    for c in CTXS:
        flat = pearson_r[c][np.isfinite(pearson_r[c])].ravel()
        m, lo, hi = fisher_mean_ci(flat)
        ax.errorbar([c], [m], yerr=[[m - lo], [hi - m]],
                    fmt='o', color='black', capsize=3, markersize=5, lw=1.2)

    ax.set_xscale('log', base=2)
    ax.set_xticks(CTXS)
    ax.set_xticklabels([str(c) for c in CTXS], fontsize=FS_TICK)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.set_xlabel('context length (log scale)', fontsize=FS_AXIS)
    ax.set_ylabel('per-neuron Pearson r (8k-step rollout)', fontsize=FS_AXIS)
    ax.set_title('stimulus baseline: distribution across neurons',
                 fontsize=FS_TITLE, pad=4)
    ax.grid(True)

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_BASE.with_suffix('.pdf')
    out_png = OUT_BASE.with_suffix('.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'[wrote] {out_pdf}')
    print(f'[wrote] {out_png}')


def main():
    pearson_r = compute_all()
    plot(pearson_r)


if __name__ == '__main__':
    main()
