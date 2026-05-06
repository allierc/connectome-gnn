"""Stimulus and voltage Jacobians: GT vs trained EED model.

2x2 figure comparing one trained EED baseline against the ground-truth
flyvis ODE structure.

  Top row    : stimulus Jacobian J^s   (input rows x stim cols)
               cols: GT identity / EED
  Bottom row : voltage  Jacobian J^v   tiled across top-3 post x top-3 pre
               cell types
               cols: GT W (j -> i) / EED

Output: figures/fig_jacobian_eed.{pdf,png}
"""

# -----------------------------------------------------------------------------
# Inputs / paths
# -----------------------------------------------------------------------------
# Config / log   : <DATA_ROOT>/log/fly/flyvis_noise_free_eed_blank50_cv00
# GT ODE params  : <DATA_ROOT>/graphs_data/fly/<dataset>/ode_params.pt
# Cache (npz)    : <REPO>/figures/_baseline_cache/jacobian_eed_*.npz
# Output         : <REPO>/figures/fig_jacobian_eed.{pdf,png}
# -----------------------------------------------------------------------------

import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from torch.func import jacfwd

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
DATA_ROOT       = os.environ.get('TRAINED_MODEL_OUTPUT_ROOT', '.')
CONFIG_EED      = 'flyvis_noise_free_eed_blank50_cv00'

T_EVAL          = 40       # frame at which Jacobians are evaluated
JV_CHUNK        = 128      # row-chunk for the (N x N) voltage Jacobian backward pass

SIG_THRESH       = 0.05    # |J^v_eed| > this counts as "signal" when scoring a type pair
MIN_TYPE_NEURONS = 30      # require this many neurons of each type to qualify
N_POST_TYPES    = 3        # number of post-synaptic types tiled along Y
N_PRE_TYPES     = 3        # number of pre-synaptic  types tiled along X

TOP_LINTHRESH   = 0.1      # SymLogNorm linear-threshold for top-row J^s
BOT_LINTHRESH   = 0.1      # SymLogNorm linear-threshold for bottom-row J^v / W
BOT_VMAX_SCALE  = 0.3      # tighten bottom-row clim (smaller -> more saturated color)

CACHE_DIR       = REPO / 'figures' / '_baseline_cache'
OUT_BASE        = REPO / 'figures' / 'fig_jacobian_eed'


# -----------------------------------------------------------------------------
# Style: Janne + small bumps so the figure is readable at one-column print width
# -----------------------------------------------------------------------------
matplotlib.rc_file(str(REPO / 'figures' / 'janne.matplotlibrc'))
plt.rcParams.update({
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
def _load_model(config_name, device):
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
    cfg, model = _load_model(config_name, device)
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
    path = CACHE_DIR / f'jacobian_eed_{config_name}_t{t_eval}.npz'
    if path.exists():
        z = np.load(path)
        return z['Je'], z['Jv'], int(z['n_input'])
    print(f'  computing Jacobians for {config_name} (t={t_eval}) ...')
    t0 = time.time()
    _, Je, Jv, n_input = _compute_jacobians(config_name, t_eval, device)
    print(f'  done in {time.time() - t0:.1f}s; caching to {path.name}')
    np.savez_compressed(path, Je=Je, Jv=Jv, n_input=np.int64(n_input))
    return Je, Jv, n_input


# -----------------------------------------------------------------------------
# Per-neuron flyvis cell type names (cached: requires loading flyvis Network)
# -----------------------------------------------------------------------------
def _cached_neuron_type_names(cfg):
    """Return (N,) array of cell-type strings (e.g. 'T4a', 'Mi1') per neuron."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f'flyvis_neuron_types_{cfg.dataset}.npz'
    if path.exists():
        return np.load(path, allow_pickle=False)['types_str']
    print(f'  loading flyvis Network for cell-type names ({cfg.dataset}) ...')
    from flyvis import Network
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config
    sim = cfg.simulation
    extent = 15 if getattr(sim, 'all_columns', False) else 8
    config_net = get_default_config(
        overrides=[], path=f'{CONFIG_PATH}/network/network.yaml'
    )
    config_net.connectome.extent = extent
    net = Network(**config_net)
    raw = np.array(net.connectome.nodes['type'])
    types_str = np.array(
        [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in raw]
    )
    np.savez_compressed(path, types_str=types_str)
    return types_str


def _build_W_dense_gt(dataset_name, n_neurons, device):
    OdeCls = get_ode_params_class('flyvis_known_ode')
    ode = OdeCls.load(graphs_data_path(f'fly/{dataset_name}'), device=device)
    ei = ode.edge_index.cpu()
    W = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    W[ei[1].numpy(), ei[0].numpy()] = ode.W.cpu().float().numpy()
    return W


# -----------------------------------------------------------------------------
# Cell-type pair selection
# -----------------------------------------------------------------------------
def _score_type_pairs(W, types_str, signal, signal_thresh, min_neurons):
    """Score (post_type, pre_type) by |W| mass * GT density * learned signal density."""
    unique = np.unique(types_str)
    type_idx = {t: np.where(types_str == t)[0] for t in unique}
    candidates = []
    for post in unique:
        rows = type_idx[post]
        if len(rows) < min_neurons:
            continue
        for pre in unique:
            cols = type_idx[pre]
            if len(cols) < min_neurons:
                continue
            sub_W   = W[np.ix_(rows, cols)]
            sub_sig = signal[np.ix_(rows, cols)]
            w_mass     = float(np.abs(sub_W).sum())
            if w_mass == 0.0:
                continue
            gt_density  = float((sub_W != 0).mean())
            sig_density = float((np.abs(sub_sig) > signal_thresh).mean())
            score = w_mass * gt_density * sig_density
            candidates.append((post, pre, score, w_mass, gt_density, sig_density))
    if not candidates:
        raise RuntimeError(
            f'no type pair satisfies min_neurons={min_neurons} with non-zero W mass'
        )
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def _topk_unique(candidates, axis, k):
    """Top-k unique values on `axis` ('post' or 'pre') by max pair score."""
    col = 0 if axis == 'post' else 1
    seen, out = set(), []
    for c in candidates:
        v = c[col]
        if v in seen:
            continue
        seen.add(v); out.append(v)
        if len(out) == k:
            break
    return out


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

    # ---- compute / load EED Jacobians ----
    Je_eed, Jv_eed, n_input = _cached_jacobians(CONFIG_EED, T_EVAL, device)

    # ---- ground truth ----
    cfg = NeuralGraphConfig.from_yaml(
        str(Path(DATA_ROOT) / 'log' / 'fly' / CONFIG_EED / 'config.yaml')
    )
    N = cfg.simulation.n_neurons
    W_dense_gt = _build_W_dense_gt(cfg.dataset, N, device)

    Je_gt = np.zeros_like(Je_eed)
    np.fill_diagonal(Je_gt[:n_input, :n_input], 1.0)

    # ---- top row crop: input-neuron rows x all stim columns ----
    top_slice = (slice(0, n_input), slice(None))
    Je_gt_p  = Je_gt[top_slice]
    Je_eed_p = Je_eed[top_slice]

    # ---- bottom row: tile top-N post x top-N pre cell types into one block ----
    types_str = _cached_neuron_type_names(cfg)
    if len(types_str) != N:
        raise RuntimeError(
            f'flyvis cell-type vector length {len(types_str)} != n_neurons {N}'
        )
    candidates = _score_type_pairs(
        W_dense_gt, types_str, signal=Jv_eed,
        signal_thresh=SIG_THRESH, min_neurons=MIN_TYPE_NEURONS,
    )
    print('top type-pair candidates (post, pre, score, w_mass, gt_density, sig_density):')
    for c in candidates[:8]:
        print(f'  {c[0]:>8s} <- {c[1]:>8s}  score={c[2]:.2e}  '
              f'w={c[3]:.1f}  gt={c[4]:.3f}  sig={c[5]:.3f}')

    post_types = _topk_unique(candidates, 'post', N_POST_TYPES)
    pre_types  = _topk_unique(candidates, 'pre',  N_PRE_TYPES)
    rows_per_type = [np.where(types_str == t)[0] for t in post_types]
    cols_per_type = [np.where(types_str == t)[0] for t in pre_types]
    rows = np.concatenate(rows_per_type)
    cols = np.concatenate(cols_per_type)
    n_post, n_pre = len(rows), len(cols)
    row_edges = np.concatenate([[0], np.cumsum([len(r) for r in rows_per_type])])
    col_edges = np.concatenate([[0], np.cumsum([len(c) for c in cols_per_type])])
    row_centers = 0.5 * (row_edges[:-1] + row_edges[1:])
    col_centers = 0.5 * (col_edges[:-1] + col_edges[1:])
    print(f'selected post types: {post_types}  (n={[len(r) for r in rows_per_type]})')
    print(f'selected pre  types: {pre_types}   (n={[len(c) for c in cols_per_type]})')

    rc = np.ix_(rows, cols)
    W_gt_p   = W_dense_gt[rc]
    Jv_eed_p = Jv_eed[rc]

    # ---- per-row shared color norms (SymLog so small magnitudes are visible) ----
    vmax_top = max(1.0, _vmax_pct(Je_eed_p, 99.0))
    vmax_bot = max(_vmax_pct(W_gt_p, 99.5) if W_gt_p.any() else 1.0,
                   _vmax_pct(Jv_eed_p, 99.0)) * BOT_VMAX_SCALE
    top_norm = SymLogNorm(linthresh=TOP_LINTHRESH, vmin=-vmax_top, vmax=vmax_top, base=10)
    bot_norm = SymLogNorm(linthresh=BOT_LINTHRESH, vmin=-vmax_bot, vmax=vmax_bot, base=10)

    # ---- figure ----
    n_diag = min(n_input, Je_gt_p.shape[1])
    top_extent = [0, Je_gt_p.shape[1], 0, n_input]
    bot_extent = [0, n_pre, 0, n_post]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True,
                             sharey='row')

    top_xlabel = r'stimulus index'
    top_ylabel = r'retinal neurons'
    bot_xlabel = 'pre-synaptic cell type'
    bot_ylabel = 'post-synaptic cell type'

    # -- top row: stimulus Jacobian J^s --
    axes[0, 0].imshow(np.zeros_like(Je_gt_p), aspect='auto', cmap='Greys', origin='lower',
                      vmin=0, vmax=1, extent=top_extent, rasterized=True)
    axes[0, 0].plot([0, n_diag - 1], [0, n_diag - 1],
                    color='red', lw=1.5, solid_capstyle='butt')
    axes[0, 0].set_title(r'Jacobian stimulus (GT)', pad=4)
    axes[0, 0].set_xlabel(top_xlabel); axes[0, 0].set_ylabel(top_ylabel)

    im_top = axes[0, 1].imshow(Je_eed_p, aspect='auto', cmap='RdBu_r', origin='lower',
                               norm=top_norm, extent=top_extent, rasterized=True)
    axes[0, 1].set_title(r'Jacobian stimulus (EED)', pad=4)
    axes[0, 1].set_xlabel(top_xlabel)

    fig.colorbar(im_top, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)

    # -- bottom row: voltage Jacobian J^v vs GT W --
    im_bot = axes[1, 0].imshow(W_gt_p, aspect='auto', cmap='RdBu_r', origin='lower',
                               norm=bot_norm, extent=bot_extent, rasterized=True)
    axes[1, 0].set_title(r'Jacobian weight (GT)', pad=4)
    axes[1, 0].set_xlabel(bot_xlabel); axes[1, 0].set_ylabel(bot_ylabel)

    axes[1, 1].imshow(Jv_eed_p, aspect='auto', cmap='RdBu_r', origin='lower',
                      norm=bot_norm, extent=bot_extent, rasterized=True)
    axes[1, 1].set_title(r'Jacobian weight (EED)', pad=4)
    axes[1, 1].set_xlabel(bot_xlabel)

    fig.colorbar(im_bot, ax=axes[1, :].tolist(), fraction=0.025, pad=0.02)

    # -- bottom-row tick labels at type-block centers + thin boundary lines --
    for j, ax in enumerate(axes[1, :]):
        ax.set_xticks(col_centers)
        ax.set_xticklabels([f'{t}\n(n={len(c)})'
                            for t, c in zip(pre_types, cols_per_type)])
        if j == 0:
            ax.set_yticks(row_centers)
            ax.set_yticklabels([f'{t}\n(n={len(r)})'
                                for t, r in zip(post_types, rows_per_type)])
        for x in col_edges[1:-1]:
            ax.axvline(x, color='k', lw=0.5, alpha=0.6)
        for y in row_edges[1:-1]:
            ax.axhline(y, color='k', lw=0.5, alpha=0.6)
        ax.tick_params(axis='both', which='both', length=0)

    # -- panel labels (a..d), aligned to outer panel bbox top --
    _add_panel_labels(fig, [axes[0, 0], axes[0, 1]],
                      ['a', 'b'], fontsize=14)
    _add_panel_labels(fig, [axes[1, 0], axes[1, 1]],
                      ['c', 'd'], fontsize=14)

    out_png = OUT_BASE.with_suffix('.png')
    out_pdf = OUT_BASE.with_suffix('.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'wrote {out_png.name}, {out_pdf.name}')


if __name__ == '__main__':
    main()
