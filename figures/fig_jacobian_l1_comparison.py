"""Stimulus and voltage Jacobians: GT vs no-reg vs L1-regularized MLP.

2x3 figure comparing two trained MLP baselines (no L1 vs L1 1e-6) against
the ground-truth flyvis ODE structure.

  Top row    : stimulus Jacobian J^s  (input rows x stim cols)
               cols: GT identity (red diagonal on blank) / no-reg / L1
  Bottom row : voltage  Jacobian J^v  zoomed to a mixed-sign ROI
               cols: GT W (j -> i) / no-reg / L1

Output: figures/fig_jacobian_l1_comparison.{pdf,png}
"""

# -----------------------------------------------------------------------------
# Inputs / paths
# -----------------------------------------------------------------------------
# Data root      : /groups/saalfeld/home/kumarv4/repos/connectome-gnn
# Configs / log  : <DATA_ROOT>/log/fly/{flyvis_noise_free_mlp_blank50_l1_0,
#                                       flyvis_noise_free_mlp_blank50_l1_1em6}
# GT ODE params  : <DATA_ROOT>/graphs_data/fly/flyvis_noise_free_blank50/ode_params.pt
# Cache (npz)    : <REPO>/figures/_baseline_cache/jacobian_l1_comparison_*.npz
# Output         : <REPO>/figures/fig_jacobian_l1_comparison.{pdf,png}
# -----------------------------------------------------------------------------

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from torch.func import jacfwd
from scipy.ndimage import uniform_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.ode_params import get_ode_params_class
from connectome_gnn.models.registry import create_model
from connectome_gnn.models.training_utils import load_flyvis_data
from connectome_gnn.utils import (
    graphs_data_path, migrate_state_dict, set_data_root,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DATA_ROOT       = '/groups/saalfeld/home/kumarv4/repos/connectome-gnn'
CONFIG_NO_REG   = 'flyvis_noise_free_mlp_blank50_l1_0'
CONFIG_L1       = 'flyvis_noise_free_mlp_blank50_l1_1em6'
LAMBDA_L1_LABEL = r'10^{-6}'

T_EVAL          = 40       # frame at which Jacobians are evaluated
JV_CHUNK        = 128      # row-chunk for the (N x N) voltage Jacobian backward pass

W_ROI           = 400      # bottom-row ROI size (square)
L1_THRESH       = 0.05     # |J^v_l1| > this counts as "signal" when scoring ROI
ROI_DI          = -150     # nudge auto-picked ROI (negative = up)
ROI_DJ          = -150     # nudge auto-picked ROI (negative = left)

TOP_LINTHRESH   = 0.1      # SymLogNorm linear-threshold for top-row J^s
BOT_LINTHRESH   = 0.1      # SymLogNorm linear-threshold for bottom-row J^v / W

CACHE_DIR       = REPO / 'figures' / '_baseline_cache'
OUT_BASE        = REPO / 'figures' / 'fig_jacobian_l1_comparison'


# -----------------------------------------------------------------------------
# Style: Janne + small bumps so the figure is readable at one-column print width
# -----------------------------------------------------------------------------
matplotlib.rc_file(str(REPO / 'figures' / 'janne.matplotlibrc'))
plt.rcParams.update({
    # Janne defaults are 6-8 pt; bump for a 6-panel composite figure.
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'axes.titlesize':   12,
    'axes.labelsize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'figure.titlesize': 13,
    'legend.fontsize':  10,
})


# -----------------------------------------------------------------------------
# Jacobian computation (cached to disk so the figure regenerates fast)
# -----------------------------------------------------------------------------
def _load_mlp(config_name, device):
    log_dir = Path(DATA_ROOT) / 'log' / 'fly' / config_name
    cfg = NeuralGraphConfig.from_yaml(str(log_dir / 'config.yaml'))
    model = create_model(
        cfg.graph_model.signal_model_name,
        aggr_type=cfg.graph_model.aggr_type,
        config=cfg, device=device,
    ).to(device)
    epoch = cfg.training.n_epochs - 1
    ckpt = log_dir / 'models' / f'best_model_with_0_graphs_{epoch}.pt'
    sd = torch.load(str(ckpt), map_location=device, weights_only=False)
    migrate_state_dict(sd)
    model.load_state_dict(sd['model_state_dict'], strict=False)
    model.eval()
    return cfg, model


def _full_voltage_jacobian(model, v_flat, stim_flat, chunk):
    """N x N Jacobian J^v[i, j] = d(dv_i/dt)/dv_j on a single frame."""
    N = v_flat.shape[0]
    J = torch.empty((N, N), dtype=torch.float32, device='cpu')
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        v = v_flat.clone().detach().requires_grad_(True)
        dvdt = model.predict_dvdt(v, stim_flat)
        grad_out = torch.zeros((i1 - i0, N), device=v.device)
        for k, i in enumerate(range(i0, i1)):
            grad_out[k, i] = 1.0
        grads = torch.autograd.grad(
            outputs=dvdt, inputs=v, grad_outputs=grad_out,
            is_grads_batched=True, retain_graph=False, create_graph=False,
        )[0]
        J[i0:i1] = grads.detach().cpu()
        del v, dvdt, grad_out, grads
    return J.numpy()


def _compute_jacobians(config_name, t_eval, device):
    cfg, model = _load_mlp(config_name, device)
    n_input = model.n_input_neurons
    x_ts, _, _ = load_flyvis_data(
        dataset_name=f'fly/{cfg.dataset}', split='train',
        fields=['voltage', 'stimulus', 'neuron_type'],
    )
    v_eval = x_ts.voltage[t_eval].to(device)
    s_eval = x_ts.stimulus[t_eval, :n_input].to(device)

    f_stim = lambda s: model.predict_dvdt(v_eval, s)
    Je = jacfwd(f_stim)(s_eval).detach().cpu().numpy()
    Jv = _full_voltage_jacobian(model, v_eval, s_eval, chunk=JV_CHUNK)
    return cfg, Je, Jv, n_input


def _cached_jacobians(config_name, t_eval, device):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f'jacobian_l1_comparison_{config_name}_t{t_eval}.npz'
    if path.exists():
        z = np.load(path)
        return z['Je'], z['Jv'], int(z['n_input'])
    print(f'  computing Jacobians for {config_name} (t={t_eval}) ...')
    t0 = time.time()
    _, Je, Jv, n_input = _compute_jacobians(config_name, t_eval, device)
    print(f'  done in {time.time() - t0:.1f}s; caching to {path.name}')
    np.savez_compressed(path, Je=Je, Jv=Jv, n_input=np.int64(n_input))
    return Je, Jv, n_input


def _build_W_dense_gt(dataset_name, n_neurons, device):
    OdeCls = get_ode_params_class('flyvis_known_ode')
    ode = OdeCls.load(graphs_data_path(f'fly/{dataset_name}'), device=device)
    ei = ode.edge_index.cpu()
    W = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    W[ei[1].numpy(), ei[0].numpy()] = ode.W.cpu().float().numpy()
    return W


# -----------------------------------------------------------------------------
# ROI selection
# -----------------------------------------------------------------------------
def _find_mixed_sign_window(M_gt, M_signal, win, signal_thresh):
    """Window with substantial positive AND negative MASS in M_gt + signal in M_signal.

    Score = sum(W_+) * sum(|W_-|) * density(|signal| > thresh).
    """
    pos_mass = uniform_filter(np.where(M_gt > 0, M_gt,  0.0).astype(np.float32),
                              size=win, mode='constant', cval=0.0)
    neg_mass = uniform_filter(np.where(M_gt < 0, -M_gt, 0.0).astype(np.float32),
                              size=win, mode='constant', cval=0.0)
    sig_dens = uniform_filter((np.abs(M_signal) > signal_thresh).astype(np.float32),
                              size=win, mode='constant', cval=0.0)
    score = pos_mass * neg_mass * sig_dens
    H, W = M_gt.shape
    valid = score[: H - win + 1, : W - win + 1]
    cy, cx = np.unravel_index(np.argmax(valid), valid.shape)
    return int(cy), int(cx)


def _vmax_pct(M, pct=99.0):
    return float(np.percentile(np.abs(M), pct))


# -----------------------------------------------------------------------------
# Panel labels: top-left of outer panel bbox, shared y across panels
# -----------------------------------------------------------------------------
def _add_panel_labels(fig, axes_flat, labels, fontsize=14):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bboxes = [ax.get_tightbbox(renderer) for ax in axes_flat]
    y1_max = max(inv.transform((bb.x0, bb.y1))[1] for bb in bboxes)
    for bb, lbl in zip(bboxes, labels):
        x0 = inv.transform((bb.x0, bb.y1))[0]
        fig.text(x0, y1_max, lbl, fontsize=fontsize, fontweight='bold',
                 va='bottom', ha='left', color='black', transform=fig.transFigure)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    set_data_root(DATA_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # ---- compute / load Jacobians for both configs ----
    Je_no_reg, Jv_no_reg, n_input = _cached_jacobians(CONFIG_NO_REG, T_EVAL, device)
    Je_l1,     Jv_l1,    _        = _cached_jacobians(CONFIG_L1,     T_EVAL, device)

    # ---- ground truth ----
    cfg = NeuralGraphConfig.from_yaml(
        str(Path(DATA_ROOT) / 'log' / 'fly' / CONFIG_NO_REG / 'config.yaml')
    )
    N = cfg.simulation.n_neurons
    W_dense_gt = _build_W_dense_gt(cfg.dataset, N, device)

    Je_gt = np.zeros_like(Je_no_reg)
    np.fill_diagonal(Je_gt[:n_input, :n_input], 1.0)

    # ---- top row crop: input-neuron rows x all stim columns ----
    top_slice = (slice(0, n_input), slice(None))
    Je_gt_p     = Je_gt[top_slice]
    Je_no_reg_p = Je_no_reg[top_slice]
    Je_l1_p     = Je_l1[top_slice]

    # ---- bottom row crop: ROI maximizing pos-mass * neg-mass * L1-signal ----
    i_auto, j_auto = _find_mixed_sign_window(W_dense_gt, Jv_l1, W_ROI, L1_THRESH)
    H, W = W_dense_gt.shape
    i0 = int(np.clip(i_auto + ROI_DI, 0, H - W_ROI))
    j0 = int(np.clip(j_auto + ROI_DJ, 0, W - W_ROI))
    print(f'auto ROI: ({i_auto}, {j_auto})  ->  shifted by ({ROI_DI:+d}, {ROI_DJ:+d})  '
          f'->  ({i0}, {j0})')

    bot_slice = (slice(i0, i0 + W_ROI), slice(j0, j0 + W_ROI))
    W_gt_p      = W_dense_gt[bot_slice]
    Jv_no_reg_p = Jv_no_reg[bot_slice]
    Jv_l1_p     = Jv_l1[bot_slice]

    sum_pos = float(W_gt_p[W_gt_p > 0].sum())
    sum_neg = float(-W_gt_p[W_gt_p < 0].sum())
    n_sig   = int((np.abs(Jv_l1_p) > L1_THRESH).sum())
    print(f'mixed-sign ROI: rows [{i0}, {i0+W_ROI}), cols [{j0}, {j0+W_ROI}) - '
          f'+sum {sum_pos:.1f}, -sum {sum_neg:.1f}, {n_sig} L1 signal pixels')

    # ---- per-row shared color norms (SymLog so small magnitudes are visible) ----
    vmax_top = max(1.0, _vmax_pct(Je_no_reg_p, 99.0), _vmax_pct(Je_l1_p, 99.0))
    vmax_bot = max(_vmax_pct(W_gt_p, 99.5) if W_gt_p.any() else 1.0,
                   _vmax_pct(Jv_no_reg_p, 99.0),
                   _vmax_pct(Jv_l1_p, 99.0))
    top_norm = SymLogNorm(linthresh=TOP_LINTHRESH, vmin=-vmax_top, vmax=vmax_top, base=10)
    bot_norm = SymLogNorm(linthresh=BOT_LINTHRESH, vmin=-vmax_bot, vmax=vmax_bot, base=10)

    # ---- figure ----
    n_diag = min(n_input, Je_gt_p.shape[1])
    top_extent = [0, Je_gt_p.shape[1], 0, n_input]
    bot_extent = [j0, j0 + W_ROI, i0, i0 + W_ROI]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True,
                             sharey='row')

    top_xlabel = r'stimulus index'
    top_ylabel = r'retinal neurons'
    bot_xlabel = r'pre-synaptic neuron'
    bot_ylabel = r'post-synaptic neuron'

    # -- top row: stimulus Jacobian J^s --
    axes[0, 0].imshow(np.zeros_like(Je_gt_p), aspect='auto', cmap='Greys', origin='lower',
                      vmin=0, vmax=1, extent=top_extent)
    axes[0, 0].plot([0, n_diag - 1], [0, n_diag - 1],
                    color='red', lw=1.5, solid_capstyle='butt')
    axes[0, 0].set_title(r'gt $J^s$ (identity for $i<n_\mathrm{in}$)', pad=4)
    axes[0, 0].set_xlabel(top_xlabel); axes[0, 0].set_ylabel(top_ylabel)

    im_top = axes[0, 1].imshow(Je_no_reg_p, aspect='auto', cmap='RdBu_r', origin='lower',
                               norm=top_norm, extent=top_extent)
    axes[0, 1].set_title(r'$J^s$ — no reg ($\lambda_{L1}=0$)', pad=4)
    axes[0, 1].set_xlabel(top_xlabel)

    axes[0, 2].imshow(Je_l1_p, aspect='auto', cmap='RdBu_r', origin='lower',
                      norm=top_norm, extent=top_extent)
    axes[0, 2].set_title(rf'$J^s$ — L1 ($\lambda_{{L1}}={LAMBDA_L1_LABEL}$)', pad=4)
    axes[0, 2].set_xlabel(top_xlabel)

    cbar_top = fig.colorbar(im_top, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)
    cbar_top.set_label(rf'$J^s$  (symlog, linthresh={TOP_LINTHRESH:g}, vmax={vmax_top:.2f})')

    # -- bottom row: voltage Jacobian J^v vs GT W --
    im_bot = axes[1, 0].imshow(W_gt_p, aspect='auto', cmap='RdBu_r', origin='lower',
                               norm=bot_norm, extent=bot_extent)
    axes[1, 0].set_title(r'gt $W$ ($j \to i$)', pad=4)
    axes[1, 0].set_xlabel(bot_xlabel); axes[1, 0].set_ylabel(bot_ylabel)

    axes[1, 1].imshow(Jv_no_reg_p, aspect='auto', cmap='RdBu_r', origin='lower',
                      norm=bot_norm, extent=bot_extent)
    axes[1, 1].set_title(r'$J^v$ — no reg ($\lambda_{L1}=0$)', pad=4)
    axes[1, 1].set_xlabel(bot_xlabel)

    axes[1, 2].imshow(Jv_l1_p, aspect='auto', cmap='RdBu_r', origin='lower',
                      norm=bot_norm, extent=bot_extent)
    axes[1, 2].set_title(rf'$J^v$ — L1 ($\lambda_{{L1}}={LAMBDA_L1_LABEL}$)', pad=4)
    axes[1, 2].set_xlabel(bot_xlabel)

    cbar_bot = fig.colorbar(im_bot, ax=axes[1, :].tolist(), fraction=0.025, pad=0.02)
    cbar_bot.set_label(rf'$J^v$ / $W$  (symlog, linthresh={BOT_LINTHRESH:g}, vmax={vmax_bot:.2f})')

    # -- panel labels (a..f), aligned to outer panel bbox top --
    _add_panel_labels(fig, [axes[0, 0], axes[0, 1], axes[0, 2]],
                      ['a', 'b', 'c'], fontsize=14)
    _add_panel_labels(fig, [axes[1, 0], axes[1, 1], axes[1, 2]],
                      ['d', 'e', 'f'], fontsize=14)

    out_png = OUT_BASE.with_suffix('.png')
    out_pdf = OUT_BASE.with_suffix('.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'wrote {out_png.name}, {out_pdf.name}')


if __name__ == '__main__':
    main()
