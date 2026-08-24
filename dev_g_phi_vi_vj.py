"""Standalone diagnostic: does the learned g_phi for flyvis_conductance
actually depend on vi (postsynaptic voltage) where real activity lives, or
did it collapse to a vj-only function matching the true flyvis synapse
ReLU(vj)?

Loads a trained checkpoint + its training edges/activity, samples real
co-occurring (vi(t), vj(t)) pairs per edge via
metrics.sample_g_phi_vi_vj_observed, and renders a 4x4 grid of
scatter(vj, g_phi, color=vi) panels — one per sampled real edge. If g_phi
ignores vi, all colors collapse onto one curve per panel; if it uses vi,
colors separate into distinct bands.

Usage:
    /workspace/.conda_envs/neural-graph-linux/bin/python dev_g_phi_vi_vj.py \
        --log_dir /groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_005_conductance_cv00
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from connectome_gnn.metrics import sample_g_phi_vi_vj_observed
from connectome_gnn.models.registry import create_model
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.utils import migrate_state_dict, graphs_data_path, set_data_root
from connectome_gnn.zarr_io import load_simulation_data


def load_run(log_dir, config_name, device):
    # load_run_config applies the same pre_folder / dataset-prefix resolution
    # as the trainer (config.yaml alone doesn't carry the "fly/" prefix).
    config, _ = load_run_config(config_name, explicit_output_root=True, task='train')
    model_config = config.graph_model

    edges = torch.load(os.path.join(log_dir, 'training_edges.pt'), map_location=device, weights_only=False)

    net_path = os.path.join(log_dir, 'models', f'best_model_with_{config.training.n_runs - 1}_graphs_0.pt')
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


def plot_vi_vj_grid(res, config, out_path, n_grid=4):
    n_edges = res['edge_ij'].shape[0]
    n_grid = min(n_grid, int(np.ceil(np.sqrt(n_edges))))

    plt.style.use('default')
    fig, axes = plt.subplots(n_grid, n_grid, figsize=(4 * n_grid, 4 * n_grid), facecolor='white')
    axes = np.atleast_1d(axes).ravel()

    vmin, vmax = res['vi'].min(), res['vi'].max()

    # Shared x/y limits (with padding) so panels are directly comparable —
    # otherwise each panel auto-scales to its own range and shapes/slopes
    # can't be visually compared across edges.
    x_lo, x_hi = res['vj'].min(), res['vj'].max()
    y_lo, y_hi = res['g_phi'].min(), res['g_phi'].max()
    x_pad = 0.05 * (x_hi - x_lo)
    y_pad = 0.05 * (y_hi - y_lo)
    xlim = (x_lo - x_pad, x_hi + x_pad)
    ylim = (y_lo - y_pad, y_hi + y_pad)

    sc = None
    for e in range(n_edges):
        ax = axes[e]
        i, j = res['edge_ij'][e]
        sc = ax.scatter(res['vj'][e], res['g_phi'][e], c=res['vi'][e],
                         cmap='coolwarm', vmin=vmin, vmax=vmax, s=6, alpha=0.7)
        ax.set_title(f'i={i}  j={j}', fontsize=11)
        ax.set_xlabel('$v_j$', fontsize=11)
        ax.set_ylabel(r'$g_\phi$', fontsize=11)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', labelsize=8)

    for e in range(n_edges, len(axes)):
        axes[e].axis('off')

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.6, pad=0.01)
        cbar.set_label('$v_i$', fontsize=11)

    fig.suptitle(
        f"{'flyvis_conductance' if 'flyvis_conductance' in config.graph_model.signal_model_name else config.graph_model.signal_model_name} "
        f"— learned $g_\\phi$ at real (co-occurring) $(v_i, v_j)$ pairs, colored by $v_i$",
        fontsize=13,
    )
    plt.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


def plot_vi_vj_3d(res, config, out_path, window_size=520):
    """3D scatter (vj, vi, g_phi) per edge, one panel each, shared axes —
    shows vi's influence as actual height/tilt instead of color.
    """
    import pyvista as pv

    n_edges = res['edge_ij'].shape[0]

    x_lo, x_hi = res['vj'].min(), res['vj'].max()
    y_lo, y_hi = res['vi'].min(), res['vi'].max()
    z_lo, z_hi = res['g_phi'].min(), res['g_phi'].max()
    bounds = (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi)

    pl = pv.Plotter(off_screen=True, shape=(1, n_edges),
                     window_size=(window_size * n_edges, window_size))

    for e in range(n_edges):
        i, j = res['edge_ij'][e]
        pl.subplot(0, e)
        points = np.stack([res['vj'][e], res['vi'][e], res['g_phi'][e]], axis=1)
        pl.add_points(points, scalars=res['g_phi'][e], cmap='viridis',
                      clim=(z_lo, z_hi), point_size=6, render_points_as_spheres=True,
                      scalar_bar_args={'title': 'g_phi', 'height': 0.5, 'width': 0.06,
                                        'vertical': True, 'position_x': 0.88})
        pl.show_grid(xtitle='vj', ytitle='vi', ztitle='g_phi', bounds=bounds,
                     font_size=10, minor_ticks=False)
        pl.add_title(f'i={i}  j={j}', font_size=10)
        pl.camera_position = 'iso'

    pl.screenshot(out_path)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str,
                        default='/groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_005_conductance_cv00')
    parser.add_argument('--config_name', type=str, default='flyvis_noise_005_conductance_cv00')
    parser.add_argument('--data_root', type=str, default='/groups/saalfeld/home/allierc/GraphData')
    parser.add_argument('--n_edges', type=int, default=16)
    parser.add_argument('--n_frames', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    set_data_root(args.data_root)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config, model, edges, x_ts = load_run(args.log_dir, args.config_name, device)

    res = sample_g_phi_vi_vj_observed(model, config, edges, x_ts,
                                       n_edges=args.n_edges, n_frames=args.n_frames, seed=args.seed)

    out_path = os.path.join(args.log_dir, 'results', 'g_phi_vi_vj_observed.png')
    plot_vi_vj_grid(res, config, out_path)

    # 3D view: the two vi-dependent edges found above, plus two flat
    # (vi-independent) edges from the same grid as a baseline for contrast.
    pairs_3d = [(5131, 4914), (12019, 12445), (10868, 9132), (10761, 11195)]
    edges_3d = torch.tensor([[j for i, j in pairs_3d], [i for i, j in pairs_3d]], device=device)
    res_3d = sample_g_phi_vi_vj_observed(model, config, edges_3d, x_ts,
                                          n_edges=len(pairs_3d), n_frames=args.n_frames, seed=args.seed)
    out_path_3d = os.path.join(args.log_dir, 'results', 'g_phi_vi_vj_3d.png')
    plot_vi_vj_3d(res_3d, config, out_path_3d)
