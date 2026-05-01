"""Stimulus-baseline context sweep: per-neuron Pearson r over an 8k-step rollout.

Single-panel scatter: per-neuron Pearson r vs context length, sampled across
neurons x CV splits, colored by coarse functional group (R1-R8, L1-L5, Lawf,
Am, C2-C3, CT1, Mi, T, Tm), with smooth per-group mean lines overlaid.

Output: figures/fig_stimulus_ctx_pearson.{pdf,png}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Inputs / paths
# ─────────────────────────────────────────────────────────────────────────────
# Data root      : /groups/saalfeld/home/kumarv4/repos/connectome-gnn
# Configs        : <REPO>/config/fly/flyvis_noise_free_stimulus_ctx1.yaml
#                  (time_window overridden in-memory for ctx=2..14)
# Test data      : <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_eed_cv{00..04}/x_list_test/
#                  {voltage.zarr, stimulus.zarr, neuron_type.zarr}
# Trained models : <DATA_ROOT>/log/fly/flyvis_noise_free_stimulus_ctx{1..14}/models/best_model_with_*.pt
# Output         : figures/fig_stimulus_ctx_pearson.{pdf,png}
# ─────────────────────────────────────────────────────────────────────────────

import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.metrics import INDEX_TO_NAME
from connectome_gnn.models.registry import create_model

try:
    from flyvis.analysis.visualization.plt_utils import trim_axis
except ImportError:
    def trim_axis(ax):
        return None


CTXS = list(range(1, 15))
CV_IDS = ['00', '01', '02', '03', '04']
LOG_ROOT = Path('/groups/saalfeld/home/kumarv4/repos/connectome-gnn/log/fly')
CV_ROOT = Path('/groups/saalfeld/home/kumarv4/repos/connectome-gnn/graphs_data/fly')
OUT_BASE = REPO / 'figures' / 'fig_stimulus_ctx_pearson'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Janne style per figures/INSTRUCTIONS.md
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.default': 'it',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 12,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'legend.fontsize': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.0,
    'legend.frameon': False,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})

GROUP_NAMES = {
    0: 'R1-R8', 1: 'L1-L5', 2: 'Lawf', 3: 'Am',
    4: 'C2-C3', 5: 'CT1', 6: 'Mi', 7: 'T', 8: 'Tm',
}


def name_to_group(name: str) -> int:
    if name.startswith('R') and len(name) >= 2 and name[1].isdigit():
        return 0
    if name.startswith('L') and len(name) >= 2 and name[1].isdigit():
        return 1
    if name.startswith('Lawf'):
        return 2
    if name == 'Am':
        return 3
    if name in ('C2', 'C3'):
        return 4
    if name.startswith('CT1'):
        return 5
    if name.startswith('Mi'):
        return 6
    if name.startswith('Tm'):  # check Tm before T
        return 8
    if name.startswith('T'):
        return 7
    return -1


def load_test_splits():
    voltage_list, stimulus_list = [], []
    for cv in CV_IDS:
        d = CV_ROOT / f'flyvis_noise_free_eed_cv{cv}' / 'x_list_test'
        v = np.asarray(zarr.open(str(d / 'voltage.zarr'), mode='r')[:], dtype=np.float32)
        s = np.asarray(zarr.open(str(d / 'stimulus.zarr'), mode='r')[:], dtype=np.float32)
        voltage_list.append(v)
        stimulus_list.append(s)
    type_list = np.asarray(
        zarr.open(str(CV_ROOT / f'flyvis_noise_free_eed_cv{CV_IDS[0]}'
                      / 'x_list_test' / 'neuron_type.zarr'), mode='r')[:]
    ).astype(int).ravel()
    return voltage_list, stimulus_list, type_list


def load_ctx_model(ctx):
    # only ctx1 yaml exists on disk; override time_window for other contexts
    cfg_path = REPO / 'config' / 'fly' / 'flyvis_noise_free_stimulus_ctx1.yaml'
    cfg = NeuralGraphConfig.from_yaml(str(cfg_path))
    cfg.config_file = f'fly/flyvis_noise_free_stimulus_ctx{ctx}'
    cfg.training.time_window = ctx

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
    voltage_list, stimulus_list, type_list = load_test_splits()
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
    return pearson_r, type_list


def plot(pearson_r, type_list, n_sample=500, seed=0):
    rng = np.random.default_rng(seed)

    neuron_group = np.array([
        name_to_group(INDEX_TO_NAME.get(int(t), '')) for t in type_list
    ])
    present_groups = sorted({int(g) for g in neuron_group if g >= 0})
    group_cmap = plt.get_cmap('tab10')
    group_color = {g: group_cmap(i % 10) for i, g in enumerate(present_groups)}

    n_cv, n_neurons = pearson_r[CTXS[0]].shape
    sample_cv = rng.integers(0, n_cv, size=n_sample)
    sample_n = rng.choice(n_neurons, size=n_sample, replace=False)
    sample_groups = neuron_group[sample_n]

    data = {c: pearson_r[c][sample_cv, sample_n] for c in CTXS}

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    for c in CTXS:
        jitter = (rng.random(n_sample) - 0.5) / 2
        ax.scatter(
            c + jitter,
            np.clip(data[c], 0, 1),
            c=[group_color.get(int(g), (0.7, 0.7, 0.7, 1.0)) for g in sample_groups],
            s=3,
            linewidths=0,
        )

    # smooth per-group mean lines (over the sampled subset, to match scatter)
    ctx_arr = np.asarray(CTXS, dtype=float)
    x_smooth = np.linspace(ctx_arr.min(), ctx_arr.max(), 200)
    k = min(3, len(ctx_arr) - 1)
    for g, color in group_color.items():
        mask = sample_groups == g
        if not mask.any():
            continue
        means = np.array([np.clip(data[c], 0, 1)[mask].mean() for c in CTXS])
        spline = make_interp_spline(ctx_arr, means, k=k)
        ax.plot(x_smooth, spline(x_smooth), color=color, linewidth=1.2)

    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel('context length')
    ax.set_ylabel('Pearson correlation $r$')

    handles = [
        Line2D([], [], marker='o', linestyle='-', color=group_color[g],
               markeredgecolor='none', label=GROUP_NAMES.get(g, f'group {g}'))
        for g in present_groups
    ]
    ax.legend(handles=handles, loc='lower right')
    trim_axis(ax)

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_BASE.with_suffix('.pdf')
    out_png = OUT_BASE.with_suffix('.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'[wrote] {out_pdf}')
    print(f'[wrote] {out_png}')


def main():
    pearson_r, type_list = compute_all()
    plot(pearson_r, type_list)


if __name__ == '__main__':
    main()
