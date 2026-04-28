"""Stimulus-baseline context sweep: per-neuron Pearson r over an 8k-step rollout.

Two-panel figure:
    A) violin distribution of per-neuron Pearson r across all neurons x 5 CVs,
       at each context length, with Fisher-mean +/- 95% CI overlaid.
    B) Fisher-mean Pearson r vs context length, one line per coarse functional
       cell-type group: R1-R8, L1-L5, Lawf, Am, C2-C3, CT1, Mi, T, Tm.

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
OUT_BASE = REPO / 'figures' / 'fig_stimulus_ctx_rollout_pearson'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Janne style per figures/INSTRUCTIONS.md
plt.rcParams.update({
    'text.usetex': False,
    'mathtext.default': 'it',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'figure.titlesize': 8,
    'figure.dpi': 300,
    'legend.fontsize': 6,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
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


def plot(pearson_r, type_list):
    # neuron -> group id
    neuron_group = np.array([
        name_to_group(INDEX_TO_NAME.get(int(t), '')) for t in type_list
    ])
    present_groups = sorted({int(g) for g in neuron_group if g >= 0})

    # per-group Fisher-mean curves (pool all CVs, all neurons in group)
    group_curves = {}
    for g in present_groups:
        mask = (neuron_group == g)
        means = []
        for c in CTXS:
            rs = pearson_r[c][:, mask]
            flat = rs[np.isfinite(rs)].ravel()
            z = np.arctanh(np.clip(flat, -0.999999, 0.999999))
            means.append(np.tanh(z.mean()) if z.size else np.nan)
        group_curves[g] = np.array(means)

    group_cmap = plt.get_cmap('tab10')
    group_color = {g: group_cmap(i % 10) for i, g in enumerate(present_groups)}

    # 18 cm ~ 7.1 in usable width: two ~3.5 in panels
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 2.6),
                                     constrained_layout=True)

    # --- panel A: per-neuron r distribution as violins + Fisher mean +/- CI ---
    data = [pearson_r[c][np.isfinite(pearson_r[c])].ravel() for c in CTXS]
    parts = ax_a.violinplot(data, positions=CTXS, widths=0.8,
                            showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.4)
        pc.set_facecolor('steelblue')
        pc.set_edgecolor('none')
    for c in CTXS:
        flat = pearson_r[c][np.isfinite(pearson_r[c])].ravel()
        m, lo, hi = fisher_mean_ci(flat)
        ax_a.errorbar([c], [m], yerr=[[m - lo], [hi - m]],
                      fmt='o', color='black', capsize=2, markersize=2, lw=0.6)
    ax_a.set_xlabel('context length')
    ax_a.set_ylabel('per-neuron Pearson r')
    ax_a.set_title('distribution across neurons', pad=4)
    ax_a.set_xticks([1, 5, 10, 14])
    trim_axis(ax_a)

    # --- panel B: per-group Fisher-mean r vs ctx ---
    for g in present_groups:
        ax_b.plot(CTXS, group_curves[g], '-', color=group_color[g], lw=1.2,
                  label=GROUP_NAMES.get(g, f'group {g}'))
    ax_b.set_xlabel('context length')
    ax_b.set_ylabel('Fisher-mean Pearson r')
    ax_b.set_title('per-group vs context length', pad=4)
    ax_b.set_xticks([1, 5, 10, 14])
    ax_b.legend(loc='lower right', fontsize=6,
                handlelength=1.4, ncol=2, columnspacing=0.8, labelspacing=0.25)
    trim_axis(ax_b)

    # --- panel labels A, B at top-left of outer panel boxes (shared y) ---
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bboxes = [ax_a.get_tightbbox(renderer), ax_b.get_tightbbox(renderer)]
    y1_max = max(inv.transform((bb.x0, bb.y1))[1] for bb in bboxes)
    for bb, lbl in zip(bboxes, ['A', 'B']):
        x0 = inv.transform((bb.x0, bb.y1))[0]
        fig.text(x0, y1_max, lbl, fontsize=10, fontweight='bold',
                 va='bottom', ha='left', color='black',
                 transform=fig.transFigure)

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_BASE.with_suffix('.pdf')
    out_png = OUT_BASE.with_suffix('.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'[wrote] {out_pdf}')
    print(f'[wrote] {out_png}')

    # group-level summary
    print(f'\n{"group":<10} {"r@ctx1":>8} {"r@ctxN":>8} {"delta":>8}')
    for g in present_groups:
        r0, rN = group_curves[g][0], group_curves[g][-1]
        print(f'{GROUP_NAMES.get(g, str(g)):<10} '
              f'{r0:>8.3f} {rN:>8.3f} {rN - r0:>+8.3f}')


def main():
    pearson_r, type_list = compute_all()
    plot(pearson_r, type_list)


if __name__ == '__main__':
    main()
