"""Plotting functions for connectome-gnn.

Used by the training loop (graph_trainer.py), data generation
(graph_data_generator.py), testing (graph_tester.py), and
post-training analysis (GNN_PlotFigure.py).

Metric computation lives in connectome_gnn.metrics — re-exported here
for backward compatibility.
"""
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit

from connectome_gnn.fitting_models import linear_model

# Re-export all metrics functions for backward compatibility.
# Callers can import from either connectome_gnn.metrics or connectome_gnn.plot.
from connectome_gnn.metrics import (  # noqa: F401
    ANATOMICAL_ORDER,
    INDEX_TO_NAME,
    _batched_mlp_eval,
    _build_f_theta_features,
    _build_g_phi_features,
    _vectorized_linear_fit,
    _vectorized_linspace,
    compute_activity_stats,
    compute_all_corrected_weights,
    compute_corrected_weights,
    compute_dynamics_r2,
    compute_grad_msg,
    compute_r_squared_NSE,
    compute_r_squared_filtered,
    derive_tau,
    derive_vrest,
    extract_f_theta_slopes,
    extract_g_phi_slopes,
    get_model_W,
)
from connectome_gnn.utils import to_numpy

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def plot_training_summary_panels(fig, log_dir, Niter=None):
    """Add embedding, weight comparison, g_phi, and f_theta function panels to a summary figure.

    Finds the last saved training snapshot and loads the PNG images into subplots 2-5
    of a 2x3 grid figure.

    Args:
        fig: matplotlib Figure (expected 2x3 subplot layout, panel 1 already used for loss)
        log_dir: path to the training log directory
        Niter: iterations per epoch (for global iteration x-axis in R² panel)
    """
    import glob
    import os

    import imageio

    from connectome_gnn.figure_style import default_style
    style = default_style

    embedding_files = glob.glob(f"{log_dir}/tmp_training/embedding/*.png")
    if not embedding_files:
        return

    last_file = max(embedding_files, key=os.path.getctime)
    filename = os.path.basename(last_file)
    last_epoch, last_N = filename.replace('.png', '').split('_')

    panels = [
        (2, f"{log_dir}/tmp_training/embedding/{last_epoch}_{last_N}.png", 'learned embedding'),
        (3, f"{log_dir}/tmp_training/matrix/comparison_{last_epoch}_{last_N}.png", 'weight comparison'),
        (4, f"{log_dir}/tmp_training/function/g_phi/func_{last_epoch}_{last_N}.png", r'$g_\phi$'),
        (5, f"{log_dir}/tmp_training/function/f_theta/func_{last_epoch}_{last_N}.png", r'$f_\theta$'),
    ]
    for pos, path, title in panels:
        fig.add_subplot(2, 3, pos)
        img = imageio.imread(path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title, fontsize=style.label_font_size)

    # Panel 6: R² metrics trajectory
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if os.path.exists(metrics_log_path):
        r2_iters, conn_vals, vrest_vals, tau_vals = [], [], [], []
        try:
            with open(metrics_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(('epoch', 'iteration')):
                        continue
                    parts = line.split(',')
                    r2_iters.append(int(parts[0]))
                    conn_vals.append(float(parts[1]))
                    vrest_vals.append(float(parts[2]) if len(parts) > 2 else 0.0)
                    tau_vals.append(float(parts[3]) if len(parts) > 3 else 0.0)
        except Exception:
            pass
        if conn_vals:
            ax6 = fig.add_subplot(2, 3, 6)
            ax6.plot(r2_iters, conn_vals, color='#d62728', linewidth=style.line_width, label='conn')
            ax6.plot(r2_iters, vrest_vals, color='#1f77b4', linewidth=style.line_width, label=r'$V_{rest}$')
            ax6.plot(r2_iters, tau_vals, color='#2ca02c', linewidth=style.line_width, label=r'$\tau$')
            ax6.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=1)
            ax6.set_ylim(-0.05, 1.05)
            style.xlabel(ax6, 'iteration')
            style.ylabel(ax6, r'$R^2$')
            ax6.set_title(r'$R^2$ metrics', fontsize=style.label_font_size)
            ax6.legend(fontsize=style.annotation_font_size, loc='lower right')
            ax6.grid(True, alpha=0.3)



def _plot_curves_fast(ax, rr, func, type_list, cmap, linewidth=1, alpha=0.1):
    """Plot per-neuron curves using LineCollection (single draw call).

    Instead of N individual ax.plot() calls (high matplotlib overhead),
    build an (N, n_pts, 2) segments array and add one LineCollection.

    Args:
        ax: matplotlib Axes.
        rr: (N, n_pts) or (n_pts,) numpy array of x-values.
        func: (N, n_pts) numpy array of y-values.
        type_list: (N,) int array of neuron type indices.
        cmap: CustomColorMap with .color(int) method.
        linewidth: line width.
        alpha: transparency.
    """
    N, n_pts = func.shape

    # If rr is 1D (shared range), broadcast to (N, n_pts)
    if rr.ndim == 1:
        rr = np.broadcast_to(rr[None, :], (N, n_pts))

    # Build (N, n_pts, 2) segments array: each row is [(x0,y0), (x1,y1), ...]
    segments = np.stack([rr, func], axis=-1)                  # (N, n_pts, 2)

    # Build per-neuron RGBA color array
    type_np = np.asarray(type_list).astype(int).ravel()
    colors = [(*cmap.color(type_np[n])[:3], alpha) for n in range(N)]

    lc = LineCollection(segments, colors=colors, linewidths=linewidth)
    ax.add_collection(lc)
    ax.autoscale_view()






# ------------------------------------------------------------------ #
#  Subplot functions — shared between training and GNN_PlotFigure
# ------------------------------------------------------------------ #

def plot_embedding(ax, model, type_list, n_types, cmap):
    """Plot embedding scatter colored by neuron type.

    Args:
        ax: matplotlib Axes.
        model: model with .a embedding tensor (N, emb_dim).
        type_list: (N,) tensor/array of integer type indices.
        n_types: number of neuron types.
        cmap: CustomColorMap with .color(int) method.
    """
    embedding = to_numpy(model.a)
    type_np = to_numpy(type_list).squeeze()
    n_neurons = len(type_np)
    _dot_s = max(20, min(120, 5000 / max(n_neurons, 1)))
    if n_neurons < 100:
        _dot_s = max(60, _dot_s)

    if embedding.shape[1] < 2:
        # 1D embedding: plot as histogram-like strip
        for n in range(n_types):
            mask = (type_np == n)
            if np.any(mask):
                ax.scatter(embedding[mask, 0], np.zeros(mask.sum()),
                           c=cmap.color(n), s=_dot_s, edgecolors='none')
        ax.set_xlabel('$a_0$', fontsize=32)
        ax.set_ylabel('')
    else:
        for n in range(n_types):
            mask = (type_np == n)
            if np.any(mask):
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           c=cmap.color(n), s=_dot_s, edgecolors='none')
        ax.set_xlabel('$a_0$', fontsize=32)
        ax.set_ylabel('$a_1$', fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


def plot_f_theta(ax, model, config, n_neurons, type_list, cmap, device, step=20,
                 gt_curves=None, gt_v_range=None, type_names=None):
    """Plot f_theta: learned mean±std per type, with optional GT overlay.

    Args:
        gt_curves: (N, n_pts) ground truth f_theta values (from ode_params.gt_f_theta_func).
        gt_v_range: (n_pts,) x values for gt_curves.
        type_names: list of type name strings for legend.
    """
    n_pts = 1000
    xlim = config.plotting.xlim

    neuron_ids = np.arange(0, n_neurons)
    n_sel = len(neuron_ids)

    rr_1d = torch.linspace(xlim[0], xlim[1], n_pts, device=device)
    rr = rr_1d.unsqueeze(0).expand(n_sel, -1)

    func = _batched_mlp_eval(
        model.f_theta, model.a[neuron_ids], rr,
        lambda rr_f, emb_f: _build_f_theta_features(rr_f, emb_f),
        device)

    type_np = to_numpy(type_list).astype(int).ravel()
    x_np = to_numpy(rr_1d)
    func_np = to_numpy(func)
    unique_types = np.unique(type_np)
    mpl_cmap = plt.cm.get_cmap('tab10', max(len(unique_types), 1))

    for idx, t in enumerate(unique_types):
        mask = type_np == t
        curves = func_np[mask]
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        color = cmap.color(t) if hasattr(cmap, 'color') else mpl_cmap(idx)
        label = type_names[idx] if type_names and idx < len(type_names) else f"type {t}"
        ax.plot(x_np, mean, linewidth=1.5, color=color, label=label)
        if std.max() > 1e-6:
            ax.fill_between(x_np, mean - std, mean + std, color=color, alpha=0.15)

    # GT overlay (dashed)
    if gt_curves is not None and gt_v_range is not None:
        gt_type_np = type_np[:gt_curves.shape[0]] if gt_curves.shape[0] <= len(type_np) else type_np
        for idx, t in enumerate(unique_types):
            mask = gt_type_np == t
            if not np.any(mask):
                continue
            gt_mean = gt_curves[mask].mean(axis=0)
            color = cmap.color(t) if hasattr(cmap, 'color') else mpl_cmap(idx)
            ax.plot(gt_v_range, gt_mean, linewidth=1.5, color=color, linestyle='--', alpha=0.7)

    ax.axhline(0, color='#aaa', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='#aaa', linewidth=0.5, linestyle='--')
    ax.set_xlim(xlim)
    ax.set_ylim(config.plotting.ylim)
    ax.set_xlabel('$v_i$', fontsize=24)
    ax.set_ylabel(r'$f_\theta(\mathbf{a}_i, v_i)$', fontsize=24)
    if len(unique_types) <= 10:
        ax.legend(fontsize=16, frameon=False, loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=18)


def plot_g_phi(ax, model, config, n_neurons, type_list, cmap, device, step=20,
               gt_curves=None, gt_v_range=None, type_names=None):
    """Plot g_phi: learned mean±std per type, with optional GT overlay.

    Args:
        gt_curves: (N, n_pts) or (n_pts,) ground truth g_phi values.
        gt_v_range: (n_pts,) x values for gt_curves.
        type_names: list of type name strings for legend.
    """
    model_config = config.graph_model
    n_pts = 1000

    neuron_ids = np.arange(0, n_neurons)
    n_sel = len(neuron_ids)

    rr_1d = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], n_pts, device=device)
    rr = rr_1d.unsqueeze(0).expand(n_sel, -1)

    post_fn = (lambda x: x ** 2) if model_config.g_phi_positive else None
    build_fn = lambda rr_f, emb_f: _build_g_phi_features(rr_f, emb_f, model_config.signal_model_name)

    func = _batched_mlp_eval(
        model.g_phi, model.a[neuron_ids], rr,
        build_fn, device, post_fn=post_fn)

    type_np = to_numpy(type_list).astype(int).ravel()
    x_np = to_numpy(rr_1d)
    func_np = to_numpy(func)
    unique_types = np.unique(type_np)
    mpl_cmap = plt.cm.get_cmap('tab10', max(len(unique_types), 1))

    for idx, t in enumerate(unique_types):
        mask = type_np == t
        curves = func_np[mask]
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        color = cmap.color(t) if hasattr(cmap, 'color') else mpl_cmap(idx)
        label = type_names[idx] if type_names and idx < len(type_names) else f"type {t}"
        ax.plot(x_np, mean, linewidth=1.5, color=color, label=label)
        if std.max() > 1e-6:
            ax.fill_between(x_np, mean - std, mean + std, color=color, alpha=0.15)

    # GT overlay (dashed)
    if gt_curves is not None and gt_v_range is not None:
        if gt_curves.ndim == 1:
            ax.plot(gt_v_range, gt_curves, linewidth=1.5, color='black',
                    linestyle='--', alpha=0.7, label='GT')
        else:
            gt_type_np = type_np[:gt_curves.shape[0]] if gt_curves.shape[0] <= len(type_np) else type_np
            for idx, t in enumerate(unique_types):
                mask = gt_type_np == t
                if not np.any(mask):
                    continue
                gt_mean = gt_curves[mask].mean(axis=0)
                color = cmap.color(t) if hasattr(cmap, 'color') else mpl_cmap(idx)
                ax.plot(gt_v_range, gt_mean, linewidth=1.5, color=color, linestyle='--', alpha=0.7)

    ax.axhline(0, color='#aaa', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='#aaa', linewidth=0.5, linestyle='--')
    ax.set_xlim(config.plotting.xlim)
    ax.set_ylim([-config.plotting.xlim[1] / 10, config.plotting.xlim[1] * 1.2])
    ax.set_xlabel('$v_j$', fontsize=24)
    ax.set_ylabel(r'$g_\phi(\mathbf{a}_j, v_j)$', fontsize=24)
    if len(unique_types) <= 10:
        ax.legend(fontsize=16, frameon=False, loc='upper left')
    ax.tick_params(axis='both', which='major', labelsize=18)


def plot_weight_scatter(ax, gt_weights, learned_weights, corrected=False,
                        xlim=None, ylim=None, mc=None, scatter_size=0.5,
                        outlier_threshold=None):
    """Plot true vs learned weight scatter with R² and slope.

    Args:
        ax: matplotlib Axes.
        gt_weights: (E,) numpy array of ground truth weights.
        learned_weights: (E,) numpy array of learned (or corrected) weights.
        corrected: if True, use W* label; if False, use W label.
        xlim: optional (lo, hi) for x-axis.
        ylim: optional (lo, hi) for y-axis.
        mc: per-edge color array; if None, uses black.
        scatter_size: scatter point size (default 0.5).
        outlier_threshold: if set, remove points with |residual| > threshold.
    """
    if outlier_threshold is not None:
        residuals = learned_weights - gt_weights
        mask = np.abs(residuals) <= outlier_threshold
        true_in = gt_weights[mask]
        learned_in = learned_weights[mask]
        mc_in = mc[mask] if mc is not None else None
    else:
        true_in = gt_weights
        learned_in = learned_weights
        mc_in = mc

    r_squared, slope = compute_r_squared_NSE(true_in, learned_in)

    scatter_color = mc_in if mc_in is not None else 'k'
    ax.scatter(true_in, learned_in, s=scatter_size, c=scatter_color, alpha=0.04)
    ax.text(0.05, 0.95,
            f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}\nN: {len(true_in)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)

    ylabel = r'learned $W_{ij}^*$' if corrected else r'learned $W_{ij}$'
    ax.set_xlabel(r'true $W_{ij}$', fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=24)

    return r_squared, slope


def plot_jacobian_w_scatter(model, x_ts, ode_params, gt_weights, n_neurons,
                            log_dir, epoch, N, device):
    """Plot W scatter using Jacobian-extracted effective connectivity."""
    model.eval()
    J_mean = model.compute_jacobian_batched(x_ts, n_samples=50, seed=0)
    model.train()

    ei = to_numpy(ode_params.edge_index)
    gt_W = to_numpy(ode_params.W)

    # Compare Jacobian entries at GT edge locations
    J_np = to_numpy(J_mean)
    learned_at_edges = J_np[ei[0], ei[1]]

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_weight_scatter(ax, gt_weights=gt_W, learned_weights=learned_at_edges,
                        corrected=False, outlier_threshold=5)
    ax.set_xlabel('true $W$', fontsize=24)
    ax.set_ylabel('Jacobian $\\partial F / \\partial v$', fontsize=24)
    plt.tight_layout()
    os.makedirs(f"{log_dir}/tmp_training/matrix", exist_ok=True)
    plt.savefig(f"{log_dir}/tmp_training/matrix/raw_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_tau(ax, slopes_f_theta, gt_taus, n_neurons, mc=None):
    """Plot learned tau vs ground truth tau.

    Args:
        ax: matplotlib Axes.
        slopes_f_theta: (N,) numpy array of f_theta slopes.
        gt_taus: (N,) tensor/array of ground truth taus.
        n_neurons: number of neurons.
        mc: color for scatter points.
    """
    learned_tau = np.where(slopes_f_theta != 0, 1.0 / -slopes_f_theta, 1.0)
    learned_tau = learned_tau[:n_neurons]
    learned_tau = np.clip(learned_tau, 0, 1)
    gt_taus_np = to_numpy(gt_taus[:n_neurons]) if torch.is_tensor(gt_taus) else np.asarray(gt_taus[:n_neurons])

    r_squared, slope = compute_r_squared_NSE(gt_taus_np, learned_tau)

    ax.scatter(gt_taus_np, learned_tau, c=mc, s=1, alpha=0.25)
    ax.text(0.05, 0.95,
            f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}\nN: {len(gt_taus_np)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)
    ax.set_xlabel(r'true $\tau$', fontsize=32)
    ax.set_ylabel(r'learned $\tau$', fontsize=32)
    ax.set_xlim([0, 0.35])
    ax.set_ylim([0, 0.35])
    ax.tick_params(axis='both', which='major', labelsize=24)

    return r_squared


def plot_vrest(ax, slopes_f_theta, offsets_f_theta, gt_V_rest, n_neurons, mc=None):
    """Plot learned V_rest vs ground truth V_rest.

    Args:
        ax: matplotlib Axes.
        slopes_f_theta: (N,) numpy array of f_theta slopes.
        offsets_f_theta: (N,) numpy array of f_theta offsets.
        gt_V_rest: (N,) tensor/array of ground truth V_rest.
        n_neurons: number of neurons.
        mc: color for scatter points.
    """
    learned_V_rest = np.where(slopes_f_theta != 0, -offsets_f_theta / slopes_f_theta, 1.0)
    gt_vr_np = to_numpy(gt_V_rest[:n_neurons]) if torch.is_tensor(gt_V_rest) else np.asarray(gt_V_rest[:n_neurons])

    r_squared, slope = compute_r_squared_NSE(gt_vr_np, learned_V_rest)

    ax.scatter(gt_vr_np, learned_V_rest, c=mc, s=1, alpha=0.25)
    ax.text(0.05, 0.95,
            f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}\nN: {len(gt_vr_np)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=24)
    ax.set_xlabel(r'true $V_{rest}$', fontsize=32)
    ax.set_ylabel(r'learned $V_{rest}$', fontsize=32)
    ax.set_xlim([-0.05, 0.9])
    ax.set_ylim([-0.05, 0.9])
    ax.tick_params(axis='both', which='major', labelsize=24)

    return r_squared


# ================================================================== #
#  CONSOLIDATED FROM generators/plots.py
# ================================================================== #

from typing import Optional

from connectome_gnn.figure_style import FigureStyle, default_style


def plot_spatial_activity_grid(
    positions: np.ndarray,
    voltages: np.ndarray,
    stimulus: np.ndarray,
    neuron_types: np.ndarray,
    output_path: str,
    calcium: Optional[np.ndarray] = None,
    n_input_neurons: Optional[int] = None,
    index_to_name: Optional[dict] = None,
    anatomical_order: Optional[list] = None,
    style: FigureStyle = default_style,
) -> None:
    """8x9 or 16x9 hex scatter grid of per-neuron-type spatial activity.

    Args:
        positions: (N, 2) spatial positions for hex scatter.
        voltages: (N,) voltage per neuron.
        stimulus: (n_input,) stimulus values for input neurons.
        neuron_types: (N,) integer neuron type per neuron.
        output_path: where to save the figure.
        calcium: (N,) calcium values (if not None, adds bottom 8 rows).
        n_input_neurons: number of input neurons (defaults to len(stimulus)).
        index_to_name: type index -> name mapping. Defaults to INDEX_TO_NAME.
        anatomical_order: panel ordering. Defaults to ANATOMICAL_ORDER.
        style: FigureStyle instance.
    """
    names = index_to_name or INDEX_TO_NAME
    order = anatomical_order or ANATOMICAL_ORDER
    n_inp = n_input_neurons or len(stimulus)
    include_calcium = calcium is not None

    n_cols = 9
    n_rows = 16 if include_calcium else 8
    panel_w, panel_h = 2.0, 1.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        facecolor=style.background,
    )
    plt.subplots_adjust(hspace=1.2)
    axes_flat = axes.flatten()

    # hide trailing panels in voltage section
    n_panels = len(order)
    for i in range(n_panels, n_cols * 8):
        if i < len(axes_flat):
            axes_flat[i].set_visible(False)
    if include_calcium:
        for i in range(n_panels + n_cols * 8, len(axes_flat)):
            axes_flat[i].set_visible(False)

    vmin_v, vmax_v = style.hex_voltage_range
    vmin_s, vmax_s = style.hex_stimulus_range
    vmin_ca, vmax_ca = style.hex_calcium_range

    for panel_idx, type_idx in enumerate(order):
        # --- voltage panel ---
        ax_v = axes_flat[panel_idx]
        _draw_hex_panel(
            ax_v, type_idx, positions, voltages, stimulus,
            neuron_types, n_inp, names,
            cmap=style.cmap, vmin=vmin_v, vmax=vmax_v,
            stim_cmap=style.cmap, stim_vmin=vmin_s, stim_vmax=vmax_s,
            style=style,
        )

        # --- calcium panel (if present) ---
        if include_calcium:
            ax_ca = axes_flat[panel_idx + n_cols * 8]
            if type_idx is None:
                # stimulus panel (same as voltage section)
                ax_ca.scatter(
                    positions[:n_inp, 0], positions[:n_inp, 1],
                    s=style.hex_stimulus_marker_size, c=stimulus,
                    cmap=style.cmap, vmin=vmin_s, vmax=vmax_s,
                    marker=style.hex_marker, alpha=1.0, linewidths=0,
                )
                ax_ca.set_title(style._label('stimuli'), fontsize=style.font_size)
            else:
                mask = neuron_types == type_idx
                count = int(np.sum(mask))
                name = names.get(type_idx, f'type_{type_idx}')
                if count > 0:
                    ax_ca.scatter(
                        positions[:count, 0], positions[:count, 1],
                        s=style.hex_marker_size, c=calcium[mask],
                        cmap=style.cmap_calcium, vmin=vmin_ca, vmax=vmax_ca,
                        marker=style.hex_marker, alpha=1, linewidths=0,
                    )
                ax_ca.set_title(style._label(name), fontsize=style.font_size)
            ax_ca.set_facecolor(style.background)
            ax_ca.set_xticks([])
            ax_ca.set_yticks([])
            ax_ca.set_aspect('equal')
            for spine in ax_ca.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95 if not include_calcium else 0.92, bottom=0.05)
    style.savefig(fig, output_path)


def plot_kinograph(
    activity: np.ndarray,
    stimulus: np.ndarray,
    output_path: str,
    rank_90_act: int = 0,
    rank_99_act: int = 0,
    rank_90_inp: int = 0,
    rank_99_inp: int = 0,
    rank_90_mc: int = 0,
    rank_99_mc: int = 0,
    zoom_size: int = 200,
    zoom_neuron_start: int = 4900,
    style: FigureStyle = default_style,
    act_labels: list | None = None,
    stim_labels: list | None = None,
) -> None:
    """2x2 kinograph: full activity + zoom, full stimulus + zoom.

    Args:
        activity: (n_neurons, n_frames) transposed voltage array.
        stimulus: (n_input_neurons, n_frames) transposed stimulus array.
        output_path: where to save the figure.
        rank_90_act: effective rank at 90% variance (activity).
        rank_99_act: effective rank at 99% variance (activity).
        rank_90_inp: effective rank at 90% variance (input).
        rank_99_inp: effective rank at 99% variance (input).
        zoom_size: size of zoom window in neurons and frames.
        zoom_neuron_start: first neuron index for the activity zoom panel.
        style: FigureStyle instance.
        act_labels: optional list of (label, y_start, y_end) tuples for
            annotating neuron type bands on the activity panel.
        stim_labels: optional list of (label, y_start, y_end) tuples for
            annotating stimulus bands on the stimulus panel.
    """
    n_neurons, n_frames = activity.shape
    n_input, _ = stimulus.shape
    vmax_act = np.abs(activity).max()
    vmax_inp = np.abs(stimulus).max() * 1.2
    zoom_f = min(zoom_size, n_frames)
    zoom_n_act = min(zoom_size, n_neurons - zoom_neuron_start)
    zoom_n_inp = min(zoom_size, n_input)

    # Downsample full-panel arrays to avoid OOM on large datasets
    # (zoom panels use small fixed-size slices and don't need this)
    MAX_DISPLAY_NEURONS = 2000
    MAX_DISPLAY_FRAMES = 4000
    step_n = max(1, n_neurons // MAX_DISPLAY_NEURONS)
    step_f = max(1, n_frames // MAX_DISPLAY_FRAMES)
    step_inp = max(1, n_input // MAX_DISPLAY_NEURONS)
    activity_ds = activity[::step_n, ::step_f]
    stimulus_ds = stimulus[::step_inp, ::step_f]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(style.figure_height * 3.5, style.figure_height * 2.5),
        gridspec_kw={'width_ratios': [2, 1]},
    )

    imshow_kw = dict(aspect='auto', cmap=style.cmap, origin='lower', interpolation='nearest')

    # top-left: full activity
    ax = axes[0, 0]

    im = ax.imshow(activity_ds, vmin=-vmax_act, vmax=vmax_act, **imshow_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=style.tick_font_size)
    ax.set_ylabel('neurons', fontsize=style.label_font_size)
    ax.set_xlabel('time (frames)', fontsize=style.label_font_size)
    ax.set_xticks([0, activity_ds.shape[1] - 1])
    ax.set_xticklabels([0, n_frames], fontsize=style.tick_font_size)
    ax.set_yticks([0, activity_ds.shape[0] - 1])
    ax.set_yticklabels([1, n_neurons], fontsize=style.tick_font_size)
    rank_label = f'rank(90%)={rank_90_act}  rank(99%)={rank_99_act}'
    if rank_90_mc > 0:
        rank_label += f'  |  centered(90%)={rank_90_mc}  (99%)={rank_99_mc}'
    ax.text(0.02, 0.97, rank_label,
            transform=ax.transAxes, fontsize=style.annotation_font_size,
            va='top', ha='left')

    # Annotate activity bands if type labels are provided
    if act_labels is not None:
        ann_fs = max(4, style.annotation_font_size - 1)
        for label, y_start, y_end in act_labels:
            y_mid = (y_start + y_end) / 2.0
            ax.text(0.99, y_mid / n_neurons, label,
                    transform=ax.transAxes, fontsize=ann_fs,
                    va='center', ha='right', color='white',
                    fontweight='bold', alpha=0.9)

    # top-right: zoom activity
    ax = axes[0, 1]

    zoom_neuron_end = zoom_neuron_start + zoom_n_act
    im = ax.imshow(activity[zoom_neuron_start:zoom_neuron_end, :zoom_f], vmin=-vmax_act, vmax=vmax_act, **imshow_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=style.tick_font_size)
    ax.set_ylabel('neurons', fontsize=style.label_font_size)
    ax.set_xlabel('time (frames)', fontsize=style.label_font_size)
    ax.set_xticks([0, zoom_f - 1])
    ax.set_xticklabels([0, zoom_f], fontsize=style.tick_font_size)
    ax.set_yticks([0, zoom_n_act - 1])
    ax.set_yticklabels([zoom_neuron_start, zoom_neuron_end], fontsize=style.tick_font_size)

    # bottom-left: full stimulus
    ax = axes[1, 0]

    im = ax.imshow(stimulus_ds, vmin=-vmax_inp, vmax=vmax_inp, **imshow_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=style.tick_font_size)
    ax.set_ylabel('stimulus', fontsize=style.label_font_size)
    ax.set_xlabel('time (frames)', fontsize=style.label_font_size)
    ax.set_xticks([0, stimulus_ds.shape[1] - 1])
    ax.set_xticklabels([0, n_frames], fontsize=style.tick_font_size)
    ax.set_yticks([0, stimulus_ds.shape[0] - 1])
    ax.set_yticklabels([1, n_input], fontsize=style.tick_font_size)
    ax.text(0.02, 0.97, f'rank(90%)={rank_90_inp}  rank(99%)={rank_99_inp}',
            transform=ax.transAxes, fontsize=style.annotation_font_size,
            va='top', ha='left')

    # Annotate stimulus bands if type labels are provided
    if stim_labels is not None:
        ann_fs = max(4, style.annotation_font_size - 1)
        for label, y_start, y_end in stim_labels:
            y_mid = (y_start + y_end) / 2.0
            ax.text(0.99, y_mid / n_input, label,
                    transform=ax.transAxes, fontsize=ann_fs,
                    va='center', ha='right', color='white',
                    fontweight='bold', alpha=0.9)

    # bottom-right: zoom stimulus
    ax = axes[1, 1]

    im = ax.imshow(stimulus[:zoom_n_inp, :zoom_f], vmin=-vmax_inp, vmax=vmax_inp, **imshow_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=style.tick_font_size)
    ax.set_ylabel('input neurons', fontsize=style.label_font_size)
    ax.set_xlabel('time (frames)', fontsize=style.label_font_size)
    ax.set_xticks([0, zoom_f - 1])
    ax.set_xticklabels([0, zoom_f], fontsize=style.tick_font_size)
    ax.set_yticks([0, zoom_n_inp - 1])
    ax.set_yticklabels([1, zoom_n_inp], fontsize=style.tick_font_size)

    plt.tight_layout()
    style.savefig(fig, output_path)


def plot_activity_traces(
    activity: np.ndarray,
    output_path: str,
    n_traces: int = 100,
    max_frames: int = 10000,
    n_input_neurons: int = 0,
    style: FigureStyle = default_style,
    neuron_indices: np.ndarray | None = None,
    type_list: np.ndarray | None = None,
    stimulus: np.ndarray | None = None,
    dt_ms: float = 0.5,
    dpi: int | None = None,
    title: str | None = None,
) -> np.ndarray:
    """Sampled neuron voltage traces stacked vertically.

    If type_list is provided, picks one neuron per type (65 types) and labels
    them by name.  Otherwise falls back to random sampling of n_traces neurons.

    Returns:
        neuron_indices used (for reuse in paired plots).
    """
    n_neurons, n_frames_raw = activity.shape
    if max_frames > 0:
        n_frames_raw = min(n_frames_raw, max_frames)
    activity = activity[:, :n_frames_raw]
    n_frames = n_frames_raw

    # Select one neuron per type when type_list is provided
    type_labels = None
    if type_list is not None and neuron_indices is None:
        names = INDEX_TO_NAME
        unique_types = np.unique(type_list)
        neuron_indices = []
        type_labels = []
        for t in unique_types:
            indices = np.where(type_list == t)[0]
            if len(indices) > 0:
                neuron_indices.append(indices[0])
                type_labels.append(names.get(int(t), f'type_{t}'))
        neuron_indices = np.array(neuron_indices)
    elif neuron_indices is None:
        n_traces = min(n_traces, n_neurons)
        neuron_indices = np.sort(np.random.choice(n_neurons, n_traces, replace=False))

    sampled = activity[neuron_indices] / 20.0  # scale down so spikes don't dominate
    step_v = 2.0
    offset = sampled + step_v * np.arange(len(neuron_indices))[:, None]

    fig, ax = style.figure(aspect=1.5)
    ax.plot(offset.T, linewidth=0.5, alpha=0.7, color=style.foreground)

    # Red stimulus trace at the bottom
    if stimulus is not None:
        stim_mean = stimulus.mean(axis=0)
        if max_frames > 0:
            stim_mean = stim_mean[:min(len(stim_mean), max_frames)]
        stim_y = offset[0].min() - step_v * 1.5 + stim_mean * step_v * 5
        ax.plot(stim_y, linewidth=0.8, alpha=0.9, color='red')

    style.xlabel(ax, 'frames', fontsize=10)

    if type_labels is not None:
        ax.set_yticks([i * step_v for i in range(len(neuron_indices))])
        ax.set_yticklabels(type_labels, fontsize=3)
        style.ylabel(ax, '')
    else:
        style.ylabel(ax, f'{len(neuron_indices)} / {n_neurons} neurons')
        ax.set_yticks([])

    ax.tick_params(axis='x', labelsize=6)
    ax.set_xlim([0, n_frames])
    y_bottom = (offset[0].min() - step_v * 3) if stimulus is not None else (offset[0].min() - 2)
    ax.set_ylim([y_bottom, offset[-1].max() + 2])

    # Secondary x-axis: time in ms
    ax2 = ax.twiny()
    ax2.set_xlim([0, n_frames * dt_ms])
    ax2.set_xlabel('time (ms)', fontsize=10)
    ax2.tick_params(axis='x', labelsize=6)
    if title:
        ax.set_title(title, fontsize=style.font_size)

    plt.tight_layout()
    save_kwargs = {}
    if dpi is not None:
        save_kwargs['dpi'] = dpi
    style.savefig(fig, output_path, **save_kwargs)
    return neuron_indices


def plot_selected_neuron_traces(
    activity: np.ndarray,
    type_list: np.ndarray,
    output_path: str,
    selected_types: Optional[list[int]] = None,
    start_frame: int = 63000,
    end_frame: int = 63500,
    index_to_name: Optional[dict] = None,
    step_v: float = 1.5,
    style: FigureStyle = default_style,
) -> None:
    """Traces for specific neuron types over a time window.

    Args:
        activity: (n_neurons, n_frames) full activity array.
        type_list: (n_neurons,) integer neuron type per neuron.
        output_path: where to save the figure.
        selected_types: list of type indices to plot. Defaults to
            [l1, mi1, mi2, r1, t1, t4a, t5a, tm1, tm4, tm9].
        start_frame: start of time window.
        end_frame: end of time window.
        index_to_name: type index -> name mapping. Defaults to INDEX_TO_NAME.
        step_v: vertical offset between traces.
        style: FigureStyle instance.
    """
    names = index_to_name or INDEX_TO_NAME
    if selected_types is None:
        selected_types = [5, 12, 19, 23, 31, 35, 39, 43, 50, 55]

    # find one neuron per selected type
    neuron_indices = []
    for stype in selected_types:
        indices = np.where(type_list == stype)[0]
        if len(indices) > 0:
            neuron_indices.append(indices[0])

    n_sel = len(neuron_indices)
    if n_sel == 0:
        return

    true_slice = activity[neuron_indices, start_frame:end_frame]

    fig, ax = style.figure(aspect=1.5)
    for i in range(n_sel):
        baseline = np.mean(true_slice[i])
        ax.plot(true_slice[i] - baseline + i * step_v,
                linewidth=style.line_width, c='green', alpha=0.75)

    # neuron ids as y-tick labels
    ytick_positions = [i * step_v for i in range(n_sel)]
    ytick_labels = [names.get(selected_types[i], f'type_{selected_types[i]}') for i in range(n_sel)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=style.tick_font_size)
    ax.set_ylim([-step_v, n_sel * step_v])
    style.ylabel(ax, 'neuron')

    n_frames_shown = end_frame - start_frame
    tick_step = max(1, round(n_frames_shown / 8 / 1000) * 1000) if n_frames_shown > 2000 else n_frames_shown
    tick_positions = list(range(0, n_frames_shown + 1, tick_step))
    if tick_positions[-1] != n_frames_shown:
        tick_positions.append(n_frames_shown)
    tick_labels = [start_frame + t for t in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=14)
    style.xlabel(ax, 'time (frames)', fontsize=16)

    plt.tight_layout()
    style.savefig(fig, output_path)


def plot_retina_traces(
    activity: np.ndarray,
    stimulus: np.ndarray,
    type_list: np.ndarray,
    output_path: str,
    max_frames: int = 0,
    dt_ms: float = 0.5,
    style: FigureStyle = default_style,
) -> None:
    """Plot R1-R8 + L1/L2 traces with stimulus overlay.

    One trace per type (R1..R8, L1, L2), picking the first neuron of each type.
    A stimulus trace (first photoreceptor input) is shown in red at the bottom.

    Args:
        activity: (n_neurons, n_frames) voltage array.
        stimulus: (n_input_neurons, n_frames) stimulus array.
        type_list: (n_neurons,) integer neuron type per neuron.
        output_path: where to save the figure.
        max_frames: truncate at this many frames (0 = show all).
        dt_ms: timestep in ms (for x-axis label).
        style: FigureStyle instance.
    """
    # R1-R8 + L1/L2 type indices from INDEX_TO_NAME
    retina_types = [23, 24, 25, 26, 27, 28, 29, 30, 5, 6]
    retina_names = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'L1', 'L2']

    neuron_indices = []
    labels = []
    for t, name in zip(retina_types, retina_names):
        indices = np.where(type_list == t)[0]
        if len(indices) > 0:
            neuron_indices.append(indices[0])
            labels.append(name)

    n_sel = len(neuron_indices)
    if n_sel == 0:
        return

    n_frames = activity.shape[1]
    if max_frames > 0:
        n_frames = min(n_frames, max_frames)

    traces = activity[neuron_indices, :n_frames]

    # Vertical offset between traces
    v_range = np.max(np.ptp(traces, axis=1))
    step_v = max(v_range * 1.2, 1.0)
    offset = traces + step_v * np.arange(n_sel)[:, None]

    fig, ax = style.figure(aspect=2.0)
    colors = plt.cm.tab20(np.linspace(0, 0.95, n_sel))
    for i in range(n_sel):
        ax.plot(offset[i], linewidth=0.8, alpha=0.85, color=colors[i], label=labels[i])

    # Stimulus trace at the bottom
    if stimulus.shape[0] > 0:
        stim_trace = stimulus[0, :n_frames]
        stim_min = offset.min() - step_v * 1.5
        stim_range = max(stim_trace.max() - stim_trace.min(), 1e-6)
        stim_scaled = (stim_trace - stim_trace.min()) / stim_range * step_v + stim_min
        ax.plot(stim_scaled, linewidth=1.0, alpha=0.9, color='red', label='stimulus')

    ax.set_yticks([i * step_v for i in range(n_sel)])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim([0, n_frames])
    ax.set_ylim([offset.min() - step_v * 2, offset.max() + step_v * 0.5])
    style.xlabel(ax, f'time (dt={dt_ms:.1f}ms)', fontsize=14)
    ax.set_title('Retina (R1-R8) + L1/L2 voltage traces', fontsize=14)
    ax.legend(loc='upper right', fontsize=10, ncol=4, framealpha=0.7)

    plt.tight_layout()
    style.savefig(fig, output_path, dpi=300)


def plot_hh_debug(
    voltage_history: np.ndarray,
    stimulus_history: np.ndarray,
    gate_m_history: np.ndarray,
    gate_h_history: np.ndarray,
    gate_n_history: np.ndarray,
    type_list: np.ndarray,
    output_path: str,
    dt_ms: float = 0.5,
    hh_substeps: int = 50,
    hh_params: dict = None,
    style: FigureStyle = default_style,
    warmup_frames: int = 0,
    max_frames: int = 0,
) -> None:
    """Multi-panel HH debug plot for R1-R8 + L1/L2.

    5 panels: voltage, stimulus, gate variables, current decomposition, dv/dt.

    Args:
        voltage_history: (n_frames, n_neurons)
        stimulus_history: (n_frames, n_neurons)
        gate_m/h/n_history: (n_frames, n_neurons)
        type_list: (n_neurons,) int type indices
        hh_params: dict with per-neuron arrays: g_L, E_L, g_Na, E_Na, g_K, E_K, C,
                   I_bias, stim_scale. If None, current panel is skipped.
        warmup_frames: skip this many frames at the start.
        max_frames: show at most this many frames after warmup (0 = all).
    """
    # R1-R8 + L1/L2
    trace_types = [23, 24, 25, 26, 27, 28, 29, 30, 5, 6]
    trace_names = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'L1', 'L2']

    idx_map = {}
    for t, name in zip(trace_types, trace_names):
        indices = np.where(type_list == t)[0]
        if len(indices) > 0:
            idx_map[name] = indices[0]

    if not idx_map:
        return

    # Slice warmup and window
    total = voltage_history.shape[0]
    start = min(warmup_frames, total - 1)
    end = total if max_frames <= 0 else min(start + max_frames, total)
    voltage_history = voltage_history[start:end]
    stimulus_history = stimulus_history[start:end]
    gate_m_history = gate_m_history[start:end]
    gate_h_history = gate_h_history[start:end]
    gate_n_history = gate_n_history[start:end]

    n_frames = voltage_history.shape[0]
    t_axis = np.arange(n_frames) * dt_ms + start * dt_ms
    r_indices = [idx_map[n] for n in trace_names[:8] if n in idx_map]  # R1-R8 only for gates/currents

    n_panels = 5 if hh_params else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.0 * n_panels), sharex=True)
    colors = plt.cm.tab20(np.linspace(0, 0.95, len(idx_map)))

    # Panel 1: Voltage
    ax = axes[0]
    for i, (name, nidx) in enumerate(idx_map.items()):
        ax.plot(t_axis, voltage_history[:, nidx], linewidth=0.8, color=colors[i], label=name)
    ax.axhline(-55, color='gray', ls='--', lw=0.7, label='spike thresh ~-55mV')
    ax.set_ylabel('voltage (mV)')
    ax.set_title(f'HH Debug: R1-R8 + L1/L2  (dt={dt_ms}ms, substeps={hh_substeps})')
    ax.legend(fontsize=12, ncol=6, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: Stimulus (raw x.stimulus value) — only R1-R8 (L1/L2 are not input neurons)
    ax = axes[1]
    for i, (name, nidx) in enumerate(idx_map.items()):
        if name.startswith('R'):
            ax.plot(t_axis, stimulus_history[:, nidx], linewidth=0.8, color=colors[i], label=name)
    ax.set_ylabel('x.stimulus')
    ax.set_title('Stimulus injected into R1-R8')
    ax.set_ylim([0.0, 1.0])
    ax.legend(fontsize=12, ncol=4, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: Gate variables (mean of m, h, n across R1-R8)
    ax = axes[2]
    m_mean = gate_m_history[:, r_indices].mean(axis=1)
    h_mean = gate_h_history[:, r_indices].mean(axis=1)
    n_mean = gate_n_history[:, r_indices].mean(axis=1)
    ax.plot(t_axis, m_mean, linewidth=1.2, color='red', label='m (Na act)')
    ax.plot(t_axis, h_mean, linewidth=1.2, color='blue', label='h (Na inact)')
    ax.plot(t_axis, n_mean, linewidth=1.2, color='green', label='n (K act)')
    ax.set_ylabel('gate value')
    ax.set_title('HH gates (mean R1-R8)')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)

    # Panel 4: Current decomposition (mean over R1-R8 first neurons)
    panel_idx = 3
    if hh_params:
        ax = axes[panel_idx]
        panel_idx += 1
        v = voltage_history[:, r_indices]   # (T, n_retina)
        m = gate_m_history[:, r_indices]
        h = gate_h_history[:, r_indices]
        n = gate_n_history[:, r_indices]
        s = stimulus_history[:, r_indices]

        g_L = np.array([hh_params['g_L'][i] for i in r_indices])
        E_L = np.array([hh_params['E_L'][i] for i in r_indices])
        g_Na = np.array([hh_params['g_Na'][i] for i in r_indices])
        E_Na = np.array([hh_params['E_Na'][i] for i in r_indices])
        g_K = np.array([hh_params['g_K'][i] for i in r_indices])
        E_K = np.array([hh_params['E_K'][i] for i in r_indices])
        I_bias = np.array([hh_params['I_bias'][i] for i in r_indices])
        stim_scale = np.array([hh_params['stim_scale'][i] for i in r_indices])

        I_Na_t = (g_Na * (m**3) * h * (v - E_Na)).mean(axis=1)
        I_K_t  = (g_K * (n**4) * (v - E_K)).mean(axis=1)
        I_L_t  = (g_L * (v - E_L)).mean(axis=1)
        I_ext_t = (I_bias + stim_scale * s).mean(axis=1)

        ax.plot(t_axis, -I_L_t, linewidth=1.0, color='gray', label='-I_L (leak)')
        ax.plot(t_axis, -I_Na_t, linewidth=1.0, color='red', label='-I_Na')
        ax.plot(t_axis, -I_K_t, linewidth=1.0, color='blue', label='-I_K')
        ax.plot(t_axis, I_ext_t, linewidth=1.0, color='green', label='I_ext')
        ax.plot(t_axis, -I_Na_t - I_K_t - I_L_t + I_ext_t, linewidth=1.5, color='black', ls='--', label='net (dv*C)')
        ax.axhline(0, color='gray', ls=':', lw=0.5)
        ax.set_ylabel('current (uA/cm²)')
        ax.set_title(f'Current decomposition R1-R8 (g_L={g_L.mean():.2f}, standard=0.3)')
        ax.legend(fontsize=12, ncol=3, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Panel 5: dv/dt (finite difference)
    ax = axes[panel_idx]
    if n_frames > 1:
        for i, (name, nidx) in enumerate(idx_map.items()):
            dv = np.diff(voltage_history[:, nidx]) / dt_ms
            ax.plot(t_axis[1:], dv, linewidth=0.6, color=colors[i], alpha=0.7, label=name)
    ax.set_ylabel('dv/dt (mV/ms)')
    ax.set_xlabel('time (ms)')
    ax.set_title('Voltage derivative (finite diff)')
    ax.legend(fontsize=12, ncol=6, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    style.savefig(fig, output_path, dpi=200)


def plot_spiking_traces(
    voltage: np.ndarray,
    spike_raster: np.ndarray,
    stimulus: np.ndarray,
    is_excitatory: np.ndarray,
    type_list: np.ndarray,
    output_path: str,
    n_traces: int = 100,
    n_input_neurons: int = 0,
    max_frames: int = 0,
    dt_ms: float = 0.2,
    style: FigureStyle = None,
) -> None:
    """Spiking voltage traces — same layout as plot_activity_traces.

    Produces two separate figures saved to output_path:
      1. ``spiking_traces.png`` — sampled voltage traces stacked vertically
         with one red stimulus trace at the bottom (matching activity_traces.png).
      2. ``spiking_raster.png`` — spike raster (E=black/light, I=gray).

    Args:
        voltage: (n_neurons, n_frames) voltage array at substep resolution.
        spike_raster: (n_neurons, n_frames) bool spike array.
        stimulus: (n_input_neurons, n_frames) stimulus array.
        is_excitatory: (n_neurons,) bool array.
        type_list: (n_neurons,) integer neuron type per neuron.
        output_path: base path for figures (directory).
        n_traces: number of sampled voltage traces to show.
        n_input_neurons: number of input (photoreceptor) neurons.
        max_frames: truncate at this many frames (0 = show all).
        dt_ms: substep timestep in ms (for x-axis).
        style: FigureStyle instance.
    """
    from connectome_gnn.figure_style import default_style
    if style is None:
        style = default_style

    n_neurons, n_frames = voltage.shape
    if max_frames > 0:
        n_frames = min(n_frames, max_frames)
    voltage = voltage[:, :n_frames]
    spike_raster = spike_raster[:, :n_frames]
    if stimulus.shape[1] > n_frames:
        stimulus = stimulus[:, :n_frames]

    # --- Figure 1: voltage traces — one neuron per type (like data_test) ---
    names = INDEX_TO_NAME
    unique_types = np.unique(type_list)
    neuron_indices = []
    type_labels = []
    for t in unique_types:
        indices = np.where(type_list == t)[0]
        if len(indices) > 0:
            neuron_indices.append(indices[0])
            type_labels.append(names.get(int(t), f'type_{t}'))
    neuron_indices = np.array(neuron_indices)
    n_sel = len(neuron_indices)

    sampled = voltage[neuron_indices]
    step_v = 40.0  # mV offset between traces
    offset = sampled + step_v * np.arange(n_sel)[:, None]

    fig, ax = style.figure(aspect=1.5)
    ax.plot(offset.T, linewidth=0.5, alpha=0.7, color=style.foreground)

    # One red stimulus trace at the bottom — larger amplitude, closer to traces
    if stimulus.shape[0] > 0:
        stim_trace = stimulus[0]
        stim_min = offset.min() - 30.0
        stim_range = max(stim_trace.max() - stim_trace.min(), 1e-6)
        stim_scaled = (stim_trace - stim_trace.min()) / stim_range * 50.0 + stim_min
        ax.plot(stim_scaled, linewidth=0.8, alpha=0.9, color='red')

    style.xlabel(ax, 'time (substeps, dt={:.1f}ms)'.format(dt_ms), fontsize=12)
    ax.set_yticks([i * step_v for i in range(n_sel)])
    ax.set_yticklabels(type_labels, fontsize=4)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlim([0, n_frames])
    ax.set_ylim([offset.min() - 50, offset.max() + 20])

    plt.tight_layout()
    traces_path = os.path.join(output_path, 'spiking_traces.png') if os.path.isdir(output_path) else output_path
    style.savefig(fig, traces_path)

    # --- Figure 2: spike raster — one neuron per type (same 65 as traces) ---
    raster_data = spike_raster[neuron_indices]
    is_exc_raster = is_excitatory[neuron_indices]

    fig2, ax2 = style.figure(aspect=1.5)
    exc_plotted = inh_plotted = False
    for i in range(n_sel):
        spike_frames = np.where(raster_data[i])[0]
        if len(spike_frames) == 0:
            continue
        is_exc = is_exc_raster[i]
        color = style.foreground if is_exc else 'gray'
        label = None
        if is_exc and not exc_plotted:
            label = 'excitatory'
            exc_plotted = True
        elif not is_exc and not inh_plotted:
            label = 'inhibitory'
            inh_plotted = True
        ax2.plot(spike_frames, np.full_like(spike_frames, i), '|',
                 color=color, ms=2.0, mew=0.6, alpha=0.9, label=label)

    # Red stimulus trace at bottom of raster — smaller amplitude
    if stimulus.shape[0] > 0:
        stim_trace = stimulus[0]
        stim_min = -3
        stim_range = max(stim_trace.max() - stim_trace.min(), 1e-6)
        stim_scaled = (stim_trace - stim_trace.min()) / stim_range * 4.0 + stim_min
        ax2.plot(stim_scaled, linewidth=0.8, alpha=0.9, color='red', label='stimulus')

    # Legend with bigger spike markers
    leg = ax2.legend(loc='upper right', fontsize=8, framealpha=0.8, markerscale=5.0)
    style.xlabel(ax2, 'time (substeps, dt={:.1f}ms)'.format(dt_ms), fontsize=12)
    ax2.set_yticks(list(range(n_sel)))
    ax2.set_yticklabels(type_labels, fontsize=4)
    ax2.tick_params(axis='x', labelsize=8)
    ax2.set_xlim([0, n_frames])
    ax2.set_ylim([-5, n_sel + 1])

    plt.tight_layout()
    raster_path = os.path.join(output_path, 'spiking_raster.png') if os.path.isdir(output_path) else output_path.replace('traces', 'raster')
    style.savefig(fig2, raster_path)


# --------------------------------------------------------------------------- #
#  Private helpers
# --------------------------------------------------------------------------- #

def _draw_hex_panel(
    ax, type_idx, positions, voltages, stimulus, neuron_types,
    n_input_neurons, names, cmap, vmin, vmax,
    stim_cmap, stim_vmin, stim_vmax, style,
        
):
    
    if n_input_neurons > 2000:
        s = style.hex_stimulus_marker_size // 3 
    else:
        s = style.hex_stimulus_marker_size


    """Draw a single hex scatter panel (voltage or stimulus)."""
    if type_idx is None:
        ax.scatter(
            positions[:n_input_neurons, 0], positions[:n_input_neurons, 1],
            s=s, c=stimulus,
            cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax,
            marker=style.hex_marker, alpha=1.0, linewidths=0,
        )
        ax.set_title(style._label('stimuli'), fontsize=style.font_size)
    else:
        mask = neuron_types == type_idx
        count = int(np.sum(mask))
        name = names.get(type_idx, f'type_{type_idx}')
        if count > 0:
            ax.scatter(
                positions[:count, 0], positions[:count, 1],
                s=s, c=voltages[mask],
                cmap=cmap, vmin=vmin, vmax=vmax,
                marker=style.hex_marker, alpha=1, linewidths=0,
            )
        ax.set_title(style._label(name), fontsize=style.font_size)

    ax.set_facecolor(style.background)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)


# ================================================================== #
#  CONSOLIDATED FROM generators/utils.py
# ================================================================== #



def plot_signal_loss(loss_dict, log_dir, epoch=None, Niter=None, epoch_boundaries=None,
                     debug=False, current_loss=None, current_regul=None, total_loss=None,
                     total_loss_regul=None):
    """
    Plot stratified loss components over training iterations.

    Creates a three-panel figure showing loss and regularization terms in both
    linear and log scale, plus connectivity R2 trajectory. Saves to {log_dir}/tmp_training/loss.png.

    Parameters:
    -----------
    loss_dict : dict
        Dictionary containing loss component lists with keys:
        - 'loss': Loss without regularization
        - 'regul_total': Total regularization loss
        - 'iteration': Global iteration numbers (for x-axis)
        - 'W_L1': W L1 sparsity penalty
        - 'W_L2': W L2 regularization penalty
        - 'g_phi_diff': g_phi monotonicity penalty
        - 'g_phi_norm': g_phi normalization
        - 'g_phi_weight': g_phi MLP weight regularization
        - 'f_theta_weight': f_theta MLP weight regularization
        - 'W_sign': W sign consistency penalty
    log_dir : str
        Directory to save the figure
    epoch : int, optional
        Current epoch number
    Niter : int, optional
        Number of iterations per epoch
    debug : bool, optional
        If True, print debug information about loss components
    current_loss : float, optional
        Current iteration total loss (for debug)
    current_regul : float, optional
        Current iteration regularization (for debug)
    total_loss : float, optional
        Accumulated total loss (for debug)
    total_loss_regul : float, optional
        Accumulated regularization loss (for debug)
    """
    if len(loss_dict['loss']) == 0:
        return

    # Debug output if requested
    if debug and current_loss is not None and current_regul is not None:
        current_pred_loss = current_loss - current_regul

        # Get current iteration component values (last element in each list)
        comp_sum = (loss_dict['W_L1'][-1] + loss_dict['W_L2'][-1] +
                   loss_dict['g_phi_diff'][-1] + loss_dict['g_phi_norm'][-1] +
                   loss_dict['g_phi_weight'][-1] + loss_dict['f_theta_weight'][-1] +
                   loss_dict['W_sign'][-1])

        print(f"\n=== DEBUG Loss Components (Epoch {epoch}, Iter {Niter}) ===")
        print("Current iteration:")
        print(f"  loss.item() (total): {current_loss:.6f}")
        print(f"  regul_this_iter: {current_regul:.6f}")
        print(f"  prediction_loss (loss - regul): {current_pred_loss:.6f}")
        print("\nRegularization breakdown:")
        print(f"  W_L1: {loss_dict['W_L1'][-1]:.6f}")
        print(f"  W_L2: {loss_dict['W_L2'][-1]:.6f}")
        print(f"  W_sign: {loss_dict['W_sign'][-1]:.6f}")
        print(f"  g_phi_diff: {loss_dict['g_phi_diff'][-1]:.6f}")
        print(f"  g_phi_norm: {loss_dict['g_phi_norm'][-1]:.6f}")
        print(f"  g_phi_weight: {loss_dict['g_phi_weight'][-1]:.6f}")
        print(f"  f_theta_weight: {loss_dict['f_theta_weight'][-1]:.6f}")
        print(f"  Sum of components: {comp_sum:.6f}")
        if total_loss is not None and total_loss_regul is not None:
            print("\nAccumulated (for reference):")
            print(f"  total_loss (accumulated): {total_loss:.6f}")
            print(f"  total_loss_regul (accumulated): {total_loss_regul:.6f}")
        if current_loss > 0:
            print(f"\nRatio: regul / loss (current iter) = {current_regul / current_loss:.4f}")
        if current_pred_loss < 0:
            print("\n⚠️  WARNING: Negative prediction loss! regul > total loss")
        print("="*60)

    style = default_style
    lw = style.line_width
    fig_loss, (ax1, ax2, ax3) = style.figure(ncols=3, width=3 * style.figure_height * style.default_aspect)

    # x-axis: use global iteration if available, otherwise list index
    x_iter = loss_dict.get('iteration') or list(range(len(loss_dict['loss'])))

    # Linear scale
    legend_fs = 7
    for a in (ax1, ax2, ax3):
        a.tick_params(axis='x', labelsize=9)
        a.tick_params(axis='y', labelsize=9)
    ax1.plot(x_iter, loss_dict['loss'], color='b', linewidth=1, label='loss (no regul)', alpha=0.8)
    ax1.plot(x_iter, loss_dict['regul_total'], color='b', linewidth=1, label='total regularization', alpha=0.8)
    ax1.plot(x_iter, loss_dict['W_L1'], color='r', linewidth=1, label='W l1 sparsity', alpha=0.7)
    ax1.plot(x_iter, loss_dict['W_L2'], color='darkred', linewidth=1, label='W l2 regul', alpha=0.7)
    ax1.plot(x_iter, loss_dict['W_sign'], color='navy', linewidth=1, label='W sign (dale)', alpha=0.7)
    ax1.plot(x_iter, loss_dict['f_theta_weight'], color='lime', linewidth=1, label=r'$f_\theta$ weight regul', alpha=0.7)
    ax1.plot(x_iter, loss_dict['g_phi_diff'], color='orange', linewidth=1, label=r'$g_\phi$ monotonicity', alpha=0.7)
    ax1.plot(x_iter, loss_dict['g_phi_norm'], color='brown', linewidth=1, label=r'$g_\phi$ norm', alpha=0.7)
    ax1.plot(x_iter, loss_dict['g_phi_weight'], color='pink', linewidth=1, label=r'$g_\phi$ weight regul', alpha=0.7)
    style.xlabel(ax1, 'iteration')
    style.ylabel(ax1, 'loss')
    ax1.legend(fontsize=legend_fs, loc='best', ncol=2)

    # Log scale
    ax2.plot(x_iter, loss_dict['loss'], color='b', linewidth=1, label='loss (no regul)', alpha=0.8)
    ax2.plot(x_iter, loss_dict['regul_total'], color='b', linewidth=1, label='total regularization', alpha=0.8)
    ax2.plot(x_iter, loss_dict['W_L1'], color='r', linewidth=1, label='W l1 sparsity', alpha=0.7)
    ax2.plot(x_iter, loss_dict['W_L2'], color='darkred', linewidth=1, label='W l2 regul', alpha=0.7)
    ax2.plot(x_iter, loss_dict['W_sign'], color='navy', linewidth=1, label='W sign (dale)', alpha=0.7)
    ax2.plot(x_iter, loss_dict['f_theta_weight'], color='lime', linewidth=1, label=r'$f_\theta$ weight regul', alpha=0.7)
    ax2.plot(x_iter, loss_dict['g_phi_diff'], color='orange', linewidth=1, label=r'$g_\phi$ monotonicity', alpha=0.7)
    ax2.plot(x_iter, loss_dict['g_phi_norm'], color='brown', linewidth=1, label=r'$g_\phi$ norm', alpha=0.7)
    ax2.plot(x_iter, loss_dict['g_phi_weight'], color='pink', linewidth=1, label=r'$g_\phi$ weight regul', alpha=0.7)
    style.xlabel(ax2, 'iteration')
    style.ylabel(ax2, 'loss')
    ax2.set_yscale('log')
    ax2.legend(fontsize=legend_fs, loc='best', ncol=2)

    # Epoch boundary lines on all three panels
    if epoch_boundaries:
        for xb in epoch_boundaries:
            for ax in (ax1, ax2, ax3):
                ax.axvline(x=xb, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # R2 metrics panel (conn, V_rest, tau)
    metrics_log_path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if os.path.exists(metrics_log_path):
        r2_iters, conn_vals, vrest_vals, tau_vals = [], [], [], []
        try:
            with open(metrics_log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(('epoch', 'iteration')):
                        continue
                    parts = line.split(',')
                    r2_iters.append(int(parts[0]))
                    conn_vals.append(float(parts[1]))
                    vrest_vals.append(float(parts[2]) if len(parts) > 2 else 0.0)
                    tau_vals.append(float(parts[3]) if len(parts) > 3 else 0.0)
        except Exception:
            pass
        if conn_vals:
            ax3.plot(r2_iters, conn_vals, color='#d62728', linewidth=1,
                     label=r'connectivity $R^2$')
            ax3.plot(r2_iters, vrest_vals, color='#1f77b4', linewidth=1,
                     label=r'$V_{rest}$ $R^2$')
            ax3.plot(r2_iters, tau_vals, color='#2ca02c', linewidth=1,
                     label=r'$\tau$ $R^2$')
            ax3.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=1)
            ax3.set_ylim(-0.05, 1.05)
            style.xlabel(ax3, 'iteration')
            style.ylabel(ax3, r'$R^2$')
            ax3.legend(fontsize=legend_fs, loc='lower right')
            # most recent R2 values
            latest_text = (f"conn={conn_vals[-1]:.3f}\n"
                           f"vrest={vrest_vals[-1]:.3f}\n"
                           f"tau={tau_vals[-1]:.3f}")
            ax3.text(0.98, 0.97, latest_text, transform=ax3.transAxes,
                     fontsize=8, verticalalignment='top', horizontalalignment='right')
        else:
            ax3.text(0.5, 0.5, 'no r\u00b2 data yet', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=style.label_font_size, color='gray')
    else:
        ax3.text(0.5, 0.5, 'no r\u00b2 data yet', ha='center', va='center',
                 transform=ax3.transAxes, fontsize=style.label_font_size, color='gray')

    style.savefig(fig_loss, f'{log_dir}/tmp_training/loss.png')
    plt.close()


def plot_loss_from_file(log_dir):
    """Load loss_components.pt and plot loss decomposition (log scale).

    Parameters
    ----------
    log_dir : str
        Log directory containing ``loss_components.pt``.

    Returns
    -------
    str
        Path to the saved ``loss.png``, or *None* if the file was not found.
    """
    import torch
    pt_path = os.path.join(log_dir, 'loss_components.pt')
    if not os.path.isfile(pt_path):
        return None
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    epoch_boundaries = data.pop('epoch_boundaries', None)

    style = default_style
    fig, ax = style.figure(ncols=1)
    x_iter = data.get('iteration') or list(range(len(data['loss'])))
    legend_fs = 7

    ax.plot(x_iter, data['loss'], color='b', linewidth=1, label='loss (no regul)', alpha=0.8)
    ax.plot(x_iter, data['regul_total'], color='b', linewidth=1, label='total regularization', alpha=0.8)
    ax.plot(x_iter, data['W_L1'], color='r', linewidth=1, label='W l1 sparsity', alpha=0.7)
    ax.plot(x_iter, data['W_L2'], color='darkred', linewidth=1, label='W l2 regul', alpha=0.7)
    ax.plot(x_iter, data['W_sign'], color='navy', linewidth=1, label='W sign (dale)', alpha=0.7)
    ax.plot(x_iter, data['f_theta_weight'], color='lime', linewidth=1, label=r'$f_\theta$ weight regul', alpha=0.7)
    ax.plot(x_iter, data['g_phi_diff'], color='orange', linewidth=1, label=r'$g_\phi$ monotonicity', alpha=0.7)
    ax.plot(x_iter, data['g_phi_norm'], color='brown', linewidth=1, label=r'$g_\phi$ norm', alpha=0.7)
    ax.plot(x_iter, data['g_phi_weight'], color='pink', linewidth=1, label=r'$g_\phi$ weight regul', alpha=0.7)
    ax.set_yscale('log')
    style.xlabel(ax, 'iteration')
    style.ylabel(ax, 'loss')
    ax.legend(fontsize=legend_fs, loc='best', ncol=2)

    if epoch_boundaries:
        for xb in epoch_boundaries:
            ax.axvline(x=xb, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    out_path = os.path.join(log_dir, 'tmp_training', 'loss_log.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    style.savefig(fig, out_path)
    plt.close()
    return out_path


# ================================================================== #
#  CONSOLIDATED FROM models/utils.py
# ================================================================== #

def plot_training_flyvis(x_ts, model, config, epoch, N, log_dir, device, type_list,
                         gt_weights, edges, n_neurons=None, n_neuron_types=None,
                         ode_params=None, hidden_ids=None, anchor_ids=None):
    from connectome_gnn.plot import (
        plot_embedding,
        plot_f_theta,
        plot_g_phi,
        plot_weight_scatter,
    )
    from connectome_gnn.utils import CustomColorMap

    if n_neurons is None:
        n_neurons = len(type_list)

    cmap = CustomColorMap(config=config)

    # Plot 1: Embedding scatter plot
    os.makedirs(f"{log_dir}/tmp_training/embedding", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_embedding(ax, model, type_list, n_neuron_types, cmap)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/embedding/{epoch}_{N}.png", dpi=87)
    plt.close()

    # Compute visible-edge mask (exclude edges that touch any hidden neuron).
    # The MAIN R² always uses every edge (both Known_ODE zero-silencing and
    # NGP-T fill-in paths) so the headline metric is comparable across
    # conditions. The `_visible_mask` is only used to compute a parallel
    # r_squared_visible below, for diagnostics.
    _visible_mask = None
    _nnr_active = getattr(model, 'NNR_hidden', None) is not None
    if hidden_ids is not None:
        _hidden_set = set(hidden_ids.cpu().numpy().tolist())
        _e = edges.cpu().numpy()
        _visible_mask = np.array([
            _e[0, i] not in _hidden_set and _e[1, i] not in _hidden_set
            for i in range(_e.shape[1])
        ])

    # Plot 2: Raw W scatter (no correction) — all edges.
    fig, ax = plt.subplots(figsize=(8, 8))
    _gt_w = to_numpy(gt_weights)
    raw_W = to_numpy(get_model_W(model).squeeze())
    r_squared_raw, _ = plot_weight_scatter(
        ax,
        gt_weights=_gt_w,
        learned_weights=raw_W,
        corrected=False,
        outlier_threshold=5,
    )
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/matrix/raw_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    os.makedirs(f"{log_dir}/results", exist_ok=True)
    plt.savefig(f"{log_dir}/results/weights_comparison_raw.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Compute corrected weights
    corrected_W, _, _, _, _ = compute_all_corrected_weights(
        model, config, edges, x_ts, device)

    # Plot 3: Corrected weight comparison scatter plot — all edges.
    fig, ax = plt.subplots(figsize=(8, 8))
    _gt_w_full = to_numpy(gt_weights)
    _corr_w_full = to_numpy(corrected_W.squeeze())
    r_squared, _ = plot_weight_scatter(
        ax,
        gt_weights=_gt_w_full,
        learned_weights=_corr_w_full,
        corrected=True,
        xlim=[-1, 2],
        ylim=[-1, 2],
        outlier_threshold=5,
    )
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"{log_dir}/results/weights_comparison_corrected.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Visible-only R² — parallel diagnostic, computed without plotting.
    # R² over edges that don't touch any hidden neuron; same as r_squared when
    # no hidden_ids.
    if _visible_mask is not None:
        _gt_w_vis = _gt_w_full[_visible_mask]
        _corr_w_vis = _corr_w_full[_visible_mask]
        _fig, _ax_tmp = plt.subplots(figsize=(4, 4))
        r_squared_visible, _ = plot_weight_scatter(
            _ax_tmp,
            gt_weights=_gt_w_vis,
            learned_weights=_corr_w_vis,
            corrected=True,
            xlim=[-1, 2],
            ylim=[-1, 2],
            outlier_threshold=5,
        )
        plt.close(_fig)
    else:
        r_squared_visible = r_squared

    # Hidden-neuron INR trace comparison (only when NNR_hidden is active).
    # Returns (hidden_pearson, anchor_pearson_or_None).
    hidden_pearson = None
    anchor_pearson = None
    if _nnr_active and hidden_ids is not None:
        hidden_pearson, anchor_pearson = plot_hidden_siren_traces(
            model, x_ts, hidden_ids, log_dir, epoch, N, device, anchor_ids=anchor_ids,
        )

    # Compute GT curves and type names from ode_params if available
    gt_g_phi = gt_f_theta = gt_v_range = _type_names = None
    if ode_params is not None:
        _v = np.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 500)
        gt_v_range = _v
        try:
            gt_g_phi = ode_params.gt_g_phi_func(_v)
        except Exception:
            pass
        try:
            gt_f_theta = ode_params.gt_f_theta_func(_v, n_neurons)
        except Exception:
            pass
        _type_names = getattr(ode_params, 'type_names', None)

    # Plot 3b: Connectivity matrix heatmap (small networks only)
    if n_neurons < 1000 and ode_params is not None:
        ei = to_numpy(edges)
        gt_W = to_numpy(gt_weights)
        learned_W = to_numpy(corrected_W.squeeze())

        # GT connectivity matrix
        J_gt = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        J_gt[ei[1], ei[0]] = gt_W  # J[post, pre]
        vmax_gt = np.percentile(np.abs(gt_W[np.abs(gt_W) > 0]), 98) if np.any(gt_W != 0) else 1.0

        # Learned connectivity matrix
        J_learned = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        J_learned[ei[1], ei[0]] = learned_W
        vmax_lr = max(vmax_gt, 1e-6)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        im0 = axes[0].imshow(J_gt, cmap='bwr_r', vmin=-vmax_gt, vmax=vmax_gt,
                             aspect='auto', interpolation='nearest', origin='upper')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_xlabel('presynaptic', fontsize=18)
        axes[0].set_ylabel('postsynaptic', fontsize=18)
        axes[0].set_title('GT $W$', fontsize=20)
        axes[0].tick_params(labelsize=14)

        im1 = axes[1].imshow(J_learned, cmap='bwr_r', vmin=-vmax_lr, vmax=vmax_lr,
                             aspect='auto', interpolation='nearest', origin='upper')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_xlabel('presynaptic', fontsize=18)
        axes[1].set_ylabel('postsynaptic', fontsize=18)
        axes[1].set_title('learned $W^*$', fontsize=20)
        axes[1].tick_params(labelsize=14)

        plt.tight_layout()
        plt.savefig(f"{log_dir}/tmp_training/matrix/connectivity_{epoch}_{N}.png", dpi=87)
        plt.savefig(f"{log_dir}/results/connectivity_matrix.png", dpi=87)
        plt.close()

    # Plot 4: Edge function visualization (g_phi)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_g_phi(ax, model, config, n_neurons, type_list, cmap, device,
               gt_curves=gt_g_phi, gt_v_range=gt_v_range, type_names=_type_names)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/function/g_phi/func_{epoch}_{N}.png", dpi=87)
    plt.savefig(f"{log_dir}/results/g_phi_func.png", dpi=87)
    plt.close()

    # Plot 5: Phi function visualization (f_theta)
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_f_theta(ax, model, config, n_neurons, type_list, cmap, device,
                 gt_curves=gt_f_theta, gt_v_range=gt_v_range, type_names=_type_names)
    plt.tight_layout()
    plt.savefig(f"{log_dir}/tmp_training/function/f_theta/func_{epoch}_{N}.png", dpi=87)
    plt.savefig(f"{log_dir}/results/f_theta_func.png", dpi=87)
    plt.close()

    return r_squared, r_squared_visible, hidden_pearson, anchor_pearson


def plot_training_linear(model, config, epoch, N, log_dir, device,
                         gt_weights, n_neurons=None):
    """Training diagnostics for LinearODE — raw W scatter + tau/Vrest vs GT.

    Uses compute_dynamics_r2_linear from metrics for R² computation,
    and generates scatter plots for W, tau, V_rest.

    Returns:
        (connectivity_r2, tau_r2, vrest_r2)
    """
    import torch.nn.functional as F

    from connectome_gnn.metrics import compute_dynamics_r2_linear
    from connectome_gnn.plot import plot_weight_scatter

    if n_neurons is None:
        n_neurons = model.n_neurons

    # Compute all R² values via shared metrics function. Returns (dict, conn_r2)
    # where dict has vrest_r2 / tau_r2 plus cleaned + outlier-count fields.
    dyn_r2, conn_r2 = compute_dynamics_r2_linear(model, config, device, n_neurons)
    vrest_r2 = dyn_r2['vrest_r2']
    tau_r2   = dyn_r2['tau_r2']

    # Load ground-truth ODE params (use correct class for connconstr models)
    from connectome_gnn.generators.ode_params import FlyVisODEParams, get_ode_params_class
    from connectome_gnn.utils import graphs_data_path
    signal_model = config.graph_model.signal_model_name
    try:
        OdeParamsCls = get_ode_params_class(signal_model)
    except KeyError:
        OdeParamsCls = FlyVisODEParams
    ode_params = OdeParamsCls.load(graphs_data_path(config.dataset), device=device)

    # Plot 1: Raw W scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_weight_scatter(
        ax,
        gt_weights=to_numpy(gt_weights),
        learned_weights=to_numpy(get_model_W(model).squeeze()),
        corrected=False,
        outlier_threshold=5,
    )
    plt.tight_layout()
    os.makedirs(f"{log_dir}/tmp_training/matrix", exist_ok=True)
    plt.savefig(f"{log_dir}/tmp_training/matrix/raw_{epoch}_{N}.png",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot 2: tau scatter (only for models with tau_i)
    if hasattr(ode_params, 'tau_i') and ode_params.tau_i is not None:
        learned_tau = to_numpy(F.softplus(model.raw_tau[:n_neurons]).detach())
        gt_tau_np = to_numpy(ode_params.tau_i[:n_neurons])
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_weight_scatter(ax, gt_weights=gt_tau_np, learned_weights=learned_tau, corrected=False)
        ax.set_xlabel(r'true $\tau$', fontsize=24)
        ax.set_ylabel(r'learned $\tau$', fontsize=24)
        plt.tight_layout()
        os.makedirs(f"{log_dir}/tmp_training/dynamics", exist_ok=True)
        plt.savefig(f"{log_dir}/tmp_training/dynamics/tau_{epoch}_{N}.png",
                    dpi=87, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Plot 3: V_rest scatter (only for models with V_i_rest)
    if hasattr(ode_params, 'V_i_rest') and ode_params.V_i_rest is not None:
        learned_vrest = to_numpy(model.V_rest[:n_neurons].detach())
        gt_vrest_np = to_numpy(ode_params.V_i_rest[:n_neurons])
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_weight_scatter(ax, gt_weights=gt_vrest_np, learned_weights=learned_vrest, corrected=False)
        ax.set_xlabel(r'true $V_{rest}$', fontsize=24)
        ax.set_ylabel(r'learned $V_{rest}$', fontsize=24)
        plt.tight_layout()
        os.makedirs(f"{log_dir}/tmp_training/dynamics", exist_ok=True)
        plt.savefig(f"{log_dir}/tmp_training/dynamics/vrest_{epoch}_{N}.png",
                    dpi=87, bbox_inches='tight', pad_inches=0)
        plt.close()

    return conn_r2, tau_r2, vrest_r2, dyn_r2


def plot_weight_comparison(w_true, w_modified, output_path, xlabel='true $W$', ylabel='modified $W$', color='white'):
    w_true_np = w_true.detach().cpu().numpy().flatten()
    w_modified_np = w_modified.detach().cpu().numpy().flatten()
    plt.figure(figsize=(8, 8))
    plt.scatter(w_true_np, w_modified_np, s=8, alpha=0.5, color=color, edgecolors='none')
    # Fit linear model
    lin_fit, _ = curve_fit(linear_model, w_true_np, w_modified_np)
    slope = lin_fit[0]
    lin_fit[1]
    # R2 calculation
    residuals = w_modified_np - linear_model(w_true_np, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((w_modified_np - np.mean(w_modified_np)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    # Plot identity line
    plt.plot([w_true_np.min(), w_true_np.max()], [w_true_np.min(), w_true_np.max()], 'r--', linewidth=2, label='identity')
    # Add text
    plt.text(w_true_np.min(), w_true_np.max(), f'$R^2$: {r_squared:.3f}\nslope: {slope:.2f}', fontsize=18, va='top', ha='left')
    plt.xlabel(xlabel, fontsize=24)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return slope, r_squared


# ================================================================== #
#  CONSOLIDATED FROM models/plot_utils.py
# ================================================================== #

import warnings

from tqdm import trange

warnings.filterwarnings('ignore')


def _sample_ngp_traces(model, x_ts, ids, n_traces, n_frames, use_anchor):
    """Collect (gt, pred) arrays of shape (n_traces, n_frames) for hidden or anchor ids."""
    import torch as _torch

    n_total = len(ids)
    n_traces = min(n_traces, n_total)
    sel = np.linspace(0, n_total - 1, n_traces, dtype=int)
    local_ids = ids[sel]

    gt_arr = np.zeros((n_traces, n_frames), dtype=np.float32)
    pred_arr = np.zeros((n_traces, n_frames), dtype=np.float32)

    model.eval()
    with _torch.no_grad():
        for k in range(n_frames):
            if use_anchor:
                pred = model.forward_anchor(k, anchor_ids=ids) # (n_anchor,)
            else:
                x = x_ts.frame(k)
                pred = model.forward_hidden(x, k, ids)         # (n_hidden,)
            gt = x_ts.voltage[k, ids]
            gt_arr[:, k] = to_numpy(gt[sel])
            pred_arr[:, k] = to_numpy(pred[sel])
    model.train()
    return gt_arr, pred_arr, local_ids


def _per_neuron_pearson(gt_arr, pred_arr):
    """Per-neuron Pearson correlations (shape (n,)) between rows of gt_arr and pred_arr."""
    n = gt_arr.shape[0]
    corrs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        g = gt_arr[i] - gt_arr[i].mean()
        p = pred_arr[i] - pred_arr[i].mean()
        denom = float(np.sqrt((g * g).sum()) * np.sqrt((p * p).sum()))
        corrs[i] = float((g * p).sum() / (denom + 1e-12)) if denom > 0 else 0.0
    return corrs


def _mean_pearson(gt_arr, pred_arr):
    """Mean per-neuron Pearson correlation between rows of gt_arr and pred_arr."""
    return float(_per_neuron_pearson(gt_arr, pred_arr).mean())


def _plot_pearson_violin(ax, hidden_corrs, anchor_corrs):
    """Side-by-side violins of per-neuron Pearson correlations for hidden and anchor traces."""
    data = [hidden_corrs]
    labels = ['hidden']
    colors = ['#4477cc']
    if anchor_corrs is not None:
        data.append(anchor_corrs)
        labels.append('anchor')
        colors.append('#cc6644')
    parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.7)
    for body, c in zip(parts['bodies'], colors):
        body.set_facecolor(c)
        body.set_edgecolor('black')
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=14)
    ax.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel('per-neuron Pearson', fontsize=14)
    # Overlay mean values as text
    for i, c in enumerate(data):
        ax.text(i + 1, -0.15, f'μ={c.mean():.3f}\nn={len(c)}',
                ha='center', va='top', fontsize=10)
    ax.set_aspect('auto')  # axes box controls squareness via figure layout


def _plot_trace_panel(ax, gt_arr, pred_arr, local_ids, pearson, title,
                       n_frames, type_names=None):
    """Render one stacked-trace panel (GT green + prediction black, per-neuron pearson).

    type_names: optional list of cell-type strings (one per trace) used as
                left-margin row labels in place of the raw neuron index.
    """
    from connectome_gnn.metrics import INDEX_TO_NAME

    n_traces = gt_arr.shape[0]
    activity_std = float(np.std(gt_arr))
    # 2.5x activity std (was 1.2) so neighbouring traces don't collide when
    # GT and NGP predictions both wiggle by ~+/-2 std around the baseline.
    step_v = max(0.5, 2.5 * activity_std) if activity_std > 0 else 1.0

    # Per-neuron linear rescale so prediction can be drawn on the same stacked axis
    for i in range(n_traces):
        bl_gt = float(np.mean(gt_arr[i]))
        ax.plot(gt_arr[i] - bl_gt + i * step_v, lw=3, c='#66cc66', alpha=0.9,
                label='GT' if i == 0 else None)
        g = gt_arr[i] - bl_gt
        p = pred_arr[i] - float(np.mean(pred_arr[i]))
        denom = float((p * p).sum())
        a_i = float((g * p).sum() / (denom + 1e-12)) if denom > 0 else 0.0
        ax.plot(a_i * p + i * step_v, lw=0.9, c='black', alpha=0.9,
                label='NGP' if i == 0 else None)
        if type_names is not None and i < len(type_names) and type_names[i]:
            label = type_names[i]
        else:
            label = f'n{local_ids[i].item()}'
        ax.text(-n_frames * 0.025, i * step_v, label,
                fontsize=12, va='bottom', ha='right', color='black')

    ax.set_ylim([-step_v, n_traces * step_v + step_v])
    ax.set_yticks([])
    ax.set_xticks([0, n_frames // 2, n_frames])
    ax.set_xticklabels([0, n_frames // 2, n_frames], fontsize=15)
    ax.set_xlabel('frame', fontsize=17)
    ax.set_xlim([-n_frames * 0.06, n_frames * 1.05])
    ax.set_title(f'{title}   pearson={pearson:.3f}', fontsize=15)
    ax.legend(loc='upper right', fontsize=14, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_hidden_siren_traces(model, x_ts, hidden_ids, log_dir, epoch, N, device,
                             n_traces=13, n_frames=1000, anchor_ids=None):
    """Plot GT voltage vs NGP-predicted voltage for a sample of hidden neurons.

    When anchor_ids is provided AND the model has anchor outputs, adds a right panel
    showing GT voltage vs NGP anchor prediction for a sample of anchor neurons.

    Saves:
        log_dir/tmp_training/hidden_{inr_type}/{epoch}_{N}.png  (checkpoint copy)
        log_dir/results/hidden_inr_traces.png                   (latest copy)

    Returns:
        (hidden_pearson, anchor_pearson)
        anchor_pearson is None when anchor_ids is not provided.
    """
    from connectome_gnn.metrics import INDEX_TO_NAME

    n_frames = min(n_frames, x_ts.n_frames)
    inr_type = getattr(model, '_inr_hidden_type', 'siren_t')
    inr_label = inr_type.upper().replace('_', '-')

    # Helper: cell-type names for the row labels (one per sampled neuron).
    def _names_for(local_ids):
        ntype = getattr(x_ts, 'neuron_type', None)
        if ntype is None:
            return None
        names = []
        for nid in local_ids:
            try:
                t = int(ntype[int(nid)].item() if hasattr(ntype[int(nid)], 'item') else ntype[int(nid)])
                names.append(INDEX_TO_NAME.get(t, f'T{t}'))
            except Exception:
                names.append('')
        return names

    # Hidden traces (always)
    gt_h, pred_h, local_h = _sample_ngp_traces(model, x_ts, hidden_ids, n_traces, n_frames, use_anchor=False)
    corrs_h = _per_neuron_pearson(gt_h, pred_h)
    pearson_h = float(corrs_h.mean())
    names_h = _names_for(local_h)

    anchor_active = (anchor_ids is not None) and (getattr(model, 'n_anchor', 0) > 0)

    if anchor_active:
        gt_a, pred_a, local_a = _sample_ngp_traces(model, x_ts, anchor_ids, n_traces, n_frames, use_anchor=True)
        corrs_a = _per_neuron_pearson(gt_a, pred_a)
        pearson_a = float(corrs_a.mean())
        names_a = _names_for(local_a)

        # 3-panel layout: two trace panels (width 15 each) + square violin panel.
        # Height scales with n_traces — at n_traces=13 this gives ~9 inches.
        panel_h = max(8, n_traces * 0.7 + 2)
        fig = plt.figure(figsize=(30 + panel_h, panel_h))
        gs = fig.add_gridspec(1, 3, width_ratios=[15, 15, panel_h])
        ax_h = fig.add_subplot(gs[0, 0])
        ax_a = fig.add_subplot(gs[0, 1])
        ax_v = fig.add_subplot(gs[0, 2])
        _plot_trace_panel(ax_h, gt_h, pred_h, local_h, pearson_h,
                          f'Hidden {inr_label}  (epoch {epoch}  iter {N})',
                          n_frames, type_names=names_h)
        _plot_trace_panel(ax_a, gt_a, pred_a, local_a, pearson_a,
                          f'Anchor {inr_label}  (epoch {epoch}  iter {N})',
                          n_frames, type_names=names_a)
        _plot_pearson_violin(ax_v, corrs_h, corrs_a)
        ax_v.set_box_aspect(1.0)  # force the axes box to be square
    else:
        pearson_a = None
        fig, ax = plt.subplots(figsize=(15, max(8, n_traces * 0.7 + 2)))
        _plot_trace_panel(ax, gt_h, pred_h, local_h, pearson_h,
                          f'Hidden {inr_label}  (epoch {epoch}  iter {N})',
                          n_frames, type_names=names_h)

    out_dir = os.path.join(log_dir, 'tmp_training', f'hidden_{inr_type}')
    os.makedirs(out_dir, exist_ok=True)
    results_dir = os.path.join(log_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{epoch}_{N}.png'), dpi=87, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, 'hidden_inr_traces.png'), dpi=87, bbox_inches='tight')
    plt.close()

    return pearson_h, pearson_a


def render_visual_field_video(model, x_ts, sim, log_dir, epoch, N, logger):
    """Render a 3-panel visual field video (GT hex, predicted hex, rolling traces).

    Computes a linear correction gt = a*pred + b over frames 0..800, then
    renders an MP4 with ground-truth vs corrected-prediction hex scatter
    plots and rolling traces for 10 representative neurons.

    Args:
        model: NeuralGNN model with forward_visual method
        x_ts: NeuronTimeSeries on GPU
        sim: SimulationConfig
        log_dir: output directory path
        epoch: current epoch number
        N: current iteration number
        logger: logging.Logger instance

    Returns:
        field_R2: R² of corrected predictions vs ground truth
        field_slope: slope coefficient 'a' of the linear fit
    """
    with torch.no_grad():

        # Static XY locations
        X1 = to_numpy(x_ts.pos[:sim.n_input_neurons])

        # group-based selection of 10 traces
        groups = 217
        group_size = sim.n_input_neurons // groups  # expect 8
        assert groups * group_size == sim.n_input_neurons, "Unexpected packing of input neurons"
        picked_groups = np.linspace(0, groups - 1, 10, dtype=int)
        member_in_group = group_size // 2
        trace_ids = (picked_groups * group_size + member_in_group).astype(int)

        # MP4 writer setup
        fps = 10
        metadata = dict(title='Field Evolution', artist='Matplotlib', comment='NN Reconstruction over time')
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        fig = plt.figure(figsize=(12, 4))

        out_dir = f"{log_dir}/tmp_training/external_input"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/field_movie_{epoch}_{N}.mp4"
        if os.path.exists(out_path):
            os.remove(out_path)

        # rolling buffers
        win = 200
        offset = 1.25
        hist_t = deque(maxlen=win)
        hist_gt = {i: deque(maxlen=win) for i in trace_ids}
        hist_pred = {i: deque(maxlen=win) for i in trace_ids}

        step_video = 2

        # First pass: collect all gt and pred, fit linear transform gt = a*pred + b
        all_gt = []
        all_pred = []
        for k_fit in range(0, 800, step_video):
            x_fit = x_ts.frame(k_fit)
            pred_fit = to_numpy(model.forward_visual(x_fit, k_fit)).squeeze()
            gt_fit = to_numpy(x_ts.stimulus[k_fit, :sim.n_input_neurons]).squeeze()
            all_gt.append(gt_fit)
            all_pred.append(pred_fit)
        all_gt = np.concatenate(all_gt)
        all_pred = np.concatenate(all_pred)

        # Least-squares fit: gt = a * pred + b
        A_fit = np.vstack([all_pred, np.ones(len(all_pred))]).T
        a_coeff, b_coeff = np.linalg.lstsq(A_fit, all_gt, rcond=None)[0]
        logger.info(f"field linear fit: gt = {a_coeff:.4f} * pred + {b_coeff:.4f}")

        # Compute field_R2 on corrected predictions
        pred_corrected_all = a_coeff * all_pred + b_coeff
        ss_res = np.sum((all_gt - pred_corrected_all) ** 2)
        ss_tot = np.sum((all_gt - np.mean(all_gt)) ** 2)
        field_R2 = 1 - ss_res / (ss_tot + 1e-16)
        field_slope = a_coeff
        logger.info(f"external input R² (corrected): {field_R2:.4f}")

        # GT value range for consistent color scaling
        gt_vmin = float(all_gt.min())
        gt_vmax = float(all_gt.max())

        with writer.saving(fig, out_path, dpi=200):
            error_list = []

            for k in trange(0, 800, step_video, ncols=100):
                # inputs and predictions
                x = x_ts.frame(k)
                pred = to_numpy(model.forward_visual(x, k))
                pred_vec = np.asarray(pred).squeeze()  # (sim.n_input_neurons,)
                pred_corrected = a_coeff * pred_vec + b_coeff  # corrected to GT scale

                gt_vec = to_numpy(x_ts.stimulus[k, :sim.n_input_neurons]).squeeze()

                # update rolling traces (store corrected predictions)
                hist_t.append(k)
                for i in trace_ids:
                    hist_gt[i].append(gt_vec[i])
                    hist_pred[i].append(pred_corrected[i])

                # draw three panels
                fig.clf()

                # RMSE on corrected predictions
                rmse_frame = float(np.sqrt(((pred_corrected - gt_vec) ** 2).mean()))
                running_rmse = float(np.mean(error_list + [rmse_frame])) if len(error_list) else rmse_frame

                # Traces (both on GT scale)
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_axis_off()
                ax3.set_facecolor("black")

                t = np.arange(len(hist_t))
                for j, i in enumerate(trace_ids):
                    y0 = j * offset
                    ax3.plot(t, np.array(hist_gt[i])   + y0, color='lime',  lw=1.6, alpha=0.95)
                    ax3.plot(t, np.array(hist_pred[i]) + y0, color='k', lw=1.2, alpha=0.95)

                ax3.set_xlim(max(0, len(t) - win), len(t))
                ax3.set_ylim(-offset * 0.5, offset * (len(trace_ids) + 0.5))
                ax3.text(
                    0.02, 0.98,
                    f"frame: {k}   RMSE: {rmse_frame:.3f}   avg RMSE: {running_rmse:.3f}   a={a_coeff:.3f} b={b_coeff:.3f}",
                    transform=ax3.transAxes,
                    va='top', ha='left',
                    fontsize=6, color='k')

                # GT field
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.scatter(X1[:, 0], X1[:, 1], s=256, c=gt_vec, cmap=default_style.cmap, marker='h', vmin=gt_vmin, vmax=gt_vmax)
                ax1.set_axis_off()
                ax1.set_title('ground truth', fontsize=12)

                # Predicted field (corrected, same scale as GT)
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.scatter(X1[:, 0], X1[:, 1], s=256, c=pred_corrected, cmap=default_style.cmap, marker='h')
                ax2.set_axis_off()
                ax2.set_title('prediction (corrected)', fontsize=12)

                plt.tight_layout()
                writer.grab_frame()

                error_list.append(rmse_frame)

    return field_R2, field_slope


def plot_connconstr_diagnostics(
    voltage_history, stimulus_history, ode_params, edge_index,
    model_name, n_neurons, dt, config, device, frame_indices=None,
    rank_info=None,
):
    """Generate traces, connectivity, and g_phi plots for connconstr models.

    Uses the same FigureStyle as the connectome-gnn pipeline:
    - flat design (no spines), 14pt labels, 12pt ticks, 200dpi
    - activity_traces: all neurons stacked, auto-scaled amplitude
    - connectivity: weight matrix heatmap with optimal contrast (percentile clamp)
    - g_phi: teacher activation function
    """
    from connectome_gnn.figure_style import default_style as style
    from connectome_gnn.utils import graphs_data_path

    style.apply_globally()
    folder = graphs_data_path(config.dataset)
    os.makedirs(folder, exist_ok=True)

    voltage_arr = np.array(voltage_history)   # (T_sampled, N)
    stimulus_arr = np.array(stimulus_history)  # (T_sampled, N)

    # --- 1. Activity traces (all neurons, auto-scaled) ---
    # Follows plot_activity_traces pattern: stacked traces, black on white
    activity = voltage_arr.T  # (N, T_sampled)
    n_frames = activity.shape[1]

    # Auto-scale: subtract per-neuron mean, normalize by global amplitude
    mu = activity.mean(axis=1, keepdims=True)
    activity_centered = activity - mu
    amp = np.percentile(np.abs(activity_centered), 99)
    if amp < 1e-12:
        amp = 1.0
    activity_scaled = activity_centered / amp

    step_v = 2.0
    offset = activity_scaled + step_v * np.arange(n_neurons)[:, None]

    fig, ax = style.figure(aspect=2.5)
    ax.plot(offset.T, linewidth=0.3, alpha=0.6, color=style.foreground)

    # Red stimulus trace at bottom — scale proportional to neuron count
    stim_mean = stimulus_arr.mean(axis=1)  # mean across neurons per timestep
    if np.abs(stim_mean).max() > 1e-12:
        stim_scaled = stim_mean / np.abs(stim_mean).max()
        stim_height = max(step_v * 8, n_neurons * step_v * 0.08)
        stim_y = offset[0].min() - stim_height * 0.6 + stim_scaled * stim_height * 0.4
        ax.plot(stim_y, linewidth=1.5, alpha=0.9, color='red')

    style.xlabel(ax, 'time (frames)')
    style.ylabel(ax, f'{n_neurons} neurons')
    ax.set_yticks([])
    if frame_indices is not None:
        # Map subsampled index to true frame numbers on x-axis
        n_samples = len(frame_indices)
        n_ticks = 5
        tick_step = max(1, n_samples // n_ticks)
        tick_pos = list(range(0, n_samples, tick_step))
        tick_labels = [str(frame_indices[i]) for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=style.tick_font_size)
    ax.set_xlim([0, n_frames])
    y_bottom = offset[0].min() - step_v * 4
    ax.set_ylim([y_bottom, offset[-1].max() + 2])

    style.savefig(fig, os.path.join(folder, "activity_traces.png"))

    # --- 2. Connectivity heatmap (optimal contrast) ---
    # W_dense[pre, post] from edge_index convention; transpose to J[post, pre]
    # to match neuroscience convention: rows=postsynaptic, cols=presynaptic
    ei = to_numpy(edge_index)
    W = to_numpy(ode_params.W)
    W_dense = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    W_dense[ei[0], ei[1]] = W
    J = W_dense.T  # J[post, pre] — paper convention

    # Zebrafish: remove disconnected neurons, sort by total outgoing weight
    # CX/larva: keep natural cell-type ordering (EPG/PEN/Δ7/PEG or PMN/MN)
    if model_name in ("zebrafish", "zebrafish_oculomotor"):
        # Remove neurons with no connections (zeroed by final_adjustments)
        has_conn = (np.abs(J).sum(axis=0) + np.abs(J).sum(axis=1)) > 0
        J_active = J[has_conn, :][:, has_conn]
        # Sort by total outgoing weight (column sum, strongest first)
        col_sum = np.sum(J_active, axis=0)
        sort_idx = np.argsort(col_sum)[::-1]
        W_plot = J_active[sort_idx, :][:, sort_idx]
    else:
        W_plot = J

    # Optimal contrast: use percentile-based clamp instead of global min/max
    nonzero_W = W[np.abs(W) > 0]
    if len(nonzero_W) > 0:
        vmax = np.percentile(np.abs(nonzero_W), 98)
    else:
        vmax = 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = style.figure(aspect=1.0)
    im = ax.imshow(
        W_plot, cmap='bwr_r', vmin=-vmax, vmax=vmax,
        aspect='auto', interpolation='nearest', origin='upper',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=style.tick_font_size)
    style.xlabel(ax, 'presynaptic neuron')
    style.ylabel(ax, 'postsynaptic neuron')

    style.savefig(fig, os.path.join(folder, "connectivity.png"))

    # --- 3. g_phi plot (per-neuron-type teacher activation function) ---
    v_range = np.linspace(-2, 5, 500)
    g_phi_vals = ode_params.gt_g_phi_func(v_range)  # (N, n_pts) or (n_pts,)

    neuron_types_np = ode_params.neuron_types.cpu().numpy() if ode_params.neuron_types is not None else np.zeros(n_neurons, dtype=int)
    unique_types = np.unique(neuron_types_np)

    # Type name labels
    type_names = getattr(ode_params, 'type_names', None)
    if type_names is None:
        type_names = [f"type {t}" for t in unique_types]

    cmap = plt.cm.get_cmap('tab10', max(len(unique_types), 1))

    fig, ax = style.figure(aspect=1.2)
    if g_phi_vals.ndim == 1:
        # Neuron-independent (e.g. zebrafish identity)
        ax.plot(v_range, g_phi_vals, linewidth=style.line_width, color=style.foreground,
                label=ode_params.g_phi_label())
    else:
        # Per-neuron curves — plot mean per type with shaded std
        for idx, t in enumerate(unique_types):
            mask = neuron_types_np == t
            curves = g_phi_vals[mask]  # (n_type, n_pts)
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            color = cmap(idx)
            label = type_names[idx] if idx < len(type_names) else f"type {t}"
            ax.plot(v_range, mean, linewidth=style.line_width, color=color, label=label)
            if std.max() > 1e-6:
                ax.fill_between(v_range, mean - std, mean + std, color=color, alpha=0.15)

    ax.axhline(0, color='#aaa', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='#aaa', linewidth=0.5, linestyle='--')
    style.xlabel(ax, '$v$ (presynaptic)')
    style.ylabel(ax, r'$g_\phi(v)$')
    ax.legend(fontsize=style.tick_font_size - 1, frameon=False, loc='upper left')

    style.savefig(fig, os.path.join(folder, "g_phi.png"))

    # --- 3b. f_theta plot (per-neuron-type update function) ---
    v_range = np.linspace(-2, 5, 500)
    f_theta_vals = ode_params.gt_f_theta_func(v_range, n_neurons)  # (N, n_pts) or None
    if f_theta_vals is not None:
        fig, ax = style.figure(aspect=1.2)
        for idx, t in enumerate(unique_types):
            mask = neuron_types_np == t
            curves = f_theta_vals[mask]
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            color = cmap(idx)
            label = type_names[idx] if idx < len(type_names) else f"type {t}"
            ax.plot(v_range, mean, linewidth=style.line_width, color=color, label=label)
            if std.max() > 1e-6:
                ax.fill_between(v_range, mean - std, mean + std, color=color, alpha=0.15)

        ax.axhline(0, color='#aaa', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='#aaa', linewidth=0.5, linestyle='--')
        style.xlabel(ax, '$v_i$ (postsynaptic)')
        style.ylabel(ax, r'$f_\theta(v_i)$')
        ax.legend(fontsize=style.tick_font_size - 1, frameon=False, loc='upper right')

        style.savefig(fig, os.path.join(folder, "f_theta.png"))

    # --- 4. Kinograph (neurons x time heatmap, viridis LUT) ---
    fig, axes = plt.subplots(
        2, 1,
        figsize=(style.figure_height * 3.0, style.figure_height * 2.0),
        gridspec_kw={'height_ratios': [3, 1]},
    )
    imshow_kw = dict(aspect='auto', cmap='viridis', origin='lower', interpolation='nearest')

    # Compute true-frame x-axis ticks for kinograph
    n_samples = voltage_arr.shape[0]
    if frame_indices is not None and len(frame_indices) == n_samples:
        n_ticks = 6
        tick_step = max(1, n_samples // n_ticks)
        tick_pos = list(range(0, n_samples, tick_step))
        tick_labels = [str(frame_indices[i]) for i in tick_pos]
    else:
        tick_pos = None
        tick_labels = None

    # Build neuron-type labels from ode_params
    type_labels = None
    if hasattr(ode_params, 'neuron_types') and ode_params.neuron_types is not None:
        tnames = getattr(ode_params, 'type_names', None)
        if tnames is not None:
            nt = to_numpy(ode_params.neuron_types)
            type_labels = []
            for ti, name in enumerate(tnames):
                idx = np.where(nt == ti)[0]
                if len(idx) > 0:
                    type_labels.append((name, int(idx.min()), int(idx.max()) + 1))

    ann_fs = max(4, style.tick_font_size - 2)

    # Top: activity kinograph
    ax = axes[0]
    vmax_act = np.percentile(np.abs(voltage_arr), 99)
    if vmax_act < 1e-12:
        vmax_act = 1.0
    im = ax.imshow(voltage_arr.T, vmin=-vmax_act, vmax=vmax_act, **imshow_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=style.tick_font_size)
    ax.set_ylabel('neurons', fontsize=style.label_font_size)
    if rank_info is not None:
        ax.set_title(
            f"activity  rank(90%)={rank_info['rank_90_act']}  rank(99%)={rank_info['rank_99_act']}"
            f"  |  centered rank(90%)={rank_info['rank_90_mc']}  rank(99%)={rank_info['rank_99_mc']}",
            fontsize=style.tick_font_size, pad=4,
        )
    if tick_pos is not None:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([])  # labels on bottom panel only
    else:
        ax.set_xticks([])
    ax.set_yticks([0, n_neurons - 1])
    ax.set_yticklabels([1, n_neurons], fontsize=style.tick_font_size)

    if type_labels is not None:
        for label, y_start, y_end in type_labels:
            y_mid = (y_start + y_end) / 2.0
            ax.text(0.99, y_mid / n_neurons, label,
                    transform=ax.transAxes, fontsize=ann_fs,
                    va='center', ha='right', color='white',
                    fontweight='bold', alpha=0.9)

    # Bottom: stimulus kinograph
    ax = axes[1]
    vmax_stim = np.percentile(np.abs(stimulus_arr), 99)
    if vmax_stim < 1e-12:
        vmax_stim = 1.0
    im = ax.imshow(stimulus_arr.T, vmin=-vmax_stim, vmax=vmax_stim, **imshow_kw)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=style.tick_font_size)
    ax.set_ylabel('stimulus', fontsize=style.label_font_size)
    ax.set_xlabel('time (frames)', fontsize=style.label_font_size)
    if rank_info is not None:
        ax.set_title(
            f"stimulus  rank(90%)={rank_info['rank_90_stim']}  rank(99%)={rank_info['rank_99_stim']}",
            fontsize=style.tick_font_size, pad=4,
        )
    if tick_pos is not None:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=style.tick_font_size)
    ax.set_yticks([0, n_neurons - 1])
    ax.set_yticklabels([1, n_neurons], fontsize=style.tick_font_size)

    if type_labels is not None:
        # Only label types that receive non-zero stimulus
        stim_power = np.sum(stimulus_arr ** 2, axis=0)
        for label, y_start, y_end in type_labels:
            band_idx = np.arange(y_start, min(y_end, len(stim_power)))
            if len(band_idx) > 0 and np.sum(stim_power[band_idx]) > 1e-6:
                y_mid = (y_start + y_end) / 2.0
                ax.text(0.99, y_mid / n_neurons, label,
                        transform=ax.transAxes, fontsize=ann_fs,
                        va='center', ha='right', color='white',
                        fontweight='bold', alpha=0.9)

    plt.tight_layout()
    style.savefig(fig, os.path.join(folder, "kinograph.png"))


def plot_sequence_preview(sequences, hex_x, hex_y, title, save_path, fig_style,
                          metadata=None, logger=None):
    """Plot first frame of first N sequences as hex maps.

    Args:
        metadata: optional list of (name, flip_ax, n_rot) tuples per sequence.
        logger: optional logger for info/warning messages.
    """
    try:
        # Compute cumulative frame offsets from actual sequence lengths
        cum_offsets = []
        offset = 0
        for seq in sequences:
            n_fr = seq["lum"].shape[0]
            cum_offsets.append((offset, offset + n_fr))
            offset += n_fr

        n_cols = 8
        n_preview = min(n_cols * 8, len(sequences))
        n_rows = (n_preview + n_cols - 1) // n_cols
        fig_preview, axes_preview = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8))
        axes_preview = np.atleast_2d(axes_preview)
        for i in range(n_preview):
            row, col = divmod(i, n_cols)
            lum = sequences[i]["lum"]
            vals = lum[0].squeeze().cpu().numpy() if isinstance(lum, torch.Tensor) else lum[0].squeeze()
            start, stop = cum_offsets[i]
            ax = axes_preview[row, col]
            ax.scatter(hex_x, hex_y, c=vals,
                       s=fig_style.hex_stimulus_marker_size,
                       marker=fig_style.hex_marker,
                       cmap=fig_style.cmap,
                       vmin=fig_style.hex_stimulus_range[0],
                       vmax=fig_style.hex_stimulus_range[1],
                       alpha=1.0, linewidths=0)
            ax.set_facecolor(fig_style.background)
            if metadata is not None and i < len(metadata):
                name, flip, rot = metadata[i][:3]
                short = str(name).split('_split_')[0].split('sequence_')[-1] if 'sequence_' in str(name) else str(name)
                ax.set_title(f"{short}\nf{flip} r{rot} [{start}:{stop}]", fontsize=4)
            else:
                ax.set_title(f"seq {i} [{start}:{stop}]", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax in axes_preview.flat:
            if not ax.has_data():
                ax.set_visible(False)
        fig_preview.suptitle(title, fontsize=9)
        fig_preview.tight_layout()
        fig_preview.savefig(save_path, dpi=200)
        plt.close(fig_preview)
        if logger is not None:
            logger.info(f"saved: {save_path}")
    except Exception as e:
        if logger is not None:
            logger.warning(f"could not save sequence preview: {e}")
        import traceback
        traceback.print_exc()
        plt.close("all")

