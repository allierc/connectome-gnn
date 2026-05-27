"""Paper figure: drosophila_cx_pi training evolution (4 x 2 panels).

Standalone — does not call the training-time snapshot renderer. All
plotting code lives in this file. Only the model construction, checkpoint
loading, and deterministic-sweep rollout are reused from connectome_gnn.

Layout (panel labels a-h):
  Top row     a) GT W_con
              b) Learned W_rec
              c) per-neuron PEN  (no title, z-score colorbar)
              d) per-neuron EPG  (no title, z-score colorbar)

  Bottom row  e) population EPG kinograph  (no traces overlay)
              f) HD tracking on constant-ω rollout
              g) Trial test rollout — top: ω(t)  (green)
                                       bottom: HD true (green) + decoded (black)
              h) Integration gain — measured slope vs true ω

Usage:
    python docs/figure/fig_evolution.py \
        --run_dir /groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi \
        --out docs/figure/fig_evolution.png
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))


# --- styling -------------------------------------------------------------

PANEL_LABEL_FS = 16
TITLE_FS = 12
LABEL_FS = 11
TICK_FS = 9
GT_COLOR = "#4daf4a"
PRED_COLOR = "black"


def _panel_label(ax, letter: str):
    ax.text(-0.12, 1.02, letter, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FS, fontweight="bold",
            va="bottom", ha="right")


# --- data loading --------------------------------------------------------


def _load_model_and_rollouts(
    run_dir: str,
    snapshot_n_steps: int = 1500,
    snapshot_omega_deg: float = 60.0,
    gain_omegas=tuple(float(v) for v in np.concatenate([
        np.arange(-180.0, -9.9, 15.0),
        np.arange(15.0, 180.1, 15.0),
    ])),
    trial_seed: int | None = None,
    trial_idx: int | None = None,
):
    """Load model + run two rollouts + pick one OU test trial."""
    import torch
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.models.drosophila_cx_eval import _deterministic_sweep_rollout
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.plot_cx import cx_epg_directions
    from connectome_gnn.utils import set_data_root
    from connectome_gnn.zarr_io import load_raw_array

    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.yaml missing in {run_dir}")
    config = NeuralGraphConfig.from_yaml(cfg_path)

    # Replicate load_run_config's dataset-prefixing + data-root setup. The
    # run_dir is `<data_root>/log/<group>/<config_name>/`, so data_root is
    # two parents up from run_dir and the dataset prefix is the group name.
    run_dir_abs = os.path.abspath(run_dir)
    group = os.path.basename(os.path.dirname(run_dir_abs))     # e.g. drosophila_cx
    data_root = os.path.dirname(os.path.dirname(os.path.dirname(run_dir_abs)))
    set_data_root(data_root)
    if group and not config.dataset.startswith(group + "/"):
        config.dataset = f"{group}/{config.dataset}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = create_model(config.graph_model.signal_model_name,
                       aggr_type=config.graph_model.aggr_type,
                       config=config, device=device)
    # Pick the highest-epoch checkpoint by default. Sort numerically by
    # the trailing _<epoch>.pt — lexicographic sort would mis-order
    # _9.pt vs _10.pt. Per-config override: the GNN tail-loss runs are
    # best at epoch 5 (past that they overfit and degrade on the
    # constant-ω extrapolation rollout — keeps Fig 4 and Fig 6 of the
    # paper consistent).
    ckpts = glob.glob(os.path.join(run_dir, "models",
                                     "best_model_with_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/models/")

    def _epoch_of(p):
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    ckpts.sort(key=_epoch_of)
    chosen = ckpts[-1]
    run_basename = os.path.basename(os.path.abspath(run_dir))
    if "gnn_tailloss" in run_basename:
        prefer = [p for p in ckpts if _epoch_of(p) == 5]
        if prefer:
            chosen = prefer[0]
    sd = torch.load(chosen, map_location=device,
                    weights_only=False)["model_state_dict"]
    net.load_state_dict(sd, strict=False)
    net.eval()

    rollout = _deterministic_sweep_rollout(
        net, n_steps=snapshot_n_steps,
        omega_deg_per_s=snapshot_omega_deg, device=device,
    )
    rollout["r_epg"] = rollout["r"][:, net.epg_indices]
    pen_type_idx = [i for i, n in enumerate(net.type_names)
                    if "PEN" in n and "PEG" not in n]
    nt = np.asarray(net.neuron_types)
    pen_indices = None
    if pen_type_idx:
        pen_idx_list: list[int] = []
        for t in pen_type_idx:
            pen_idx_list.extend(np.where(nt == t)[0].tolist())
        pen_indices = np.array(sorted(pen_idx_list), dtype=np.int64)
        rollout["r_pen"] = rollout["r"][:, pen_indices]

    epg_theta = cx_epg_directions(net.epg_glom_ix)

    # Integration-gain sweeps
    gain_data = []
    for omega in gain_omegas:
        ro = _deterministic_sweep_rollout(
            net, n_steps=snapshot_n_steps,
            omega_deg_per_s=float(omega), device=device,
        )
        gain_data.append((float(omega), ro))

    # One OU test trial (random seeded)
    from connectome_gnn.utils import graphs_data_path
    root = graphs_data_path(config.dataset)
    u_test = load_raw_array(f"{root}/test/stimulus.zarr")
    y_test = load_raw_array(f"{root}/test/target.zarr")
    if trial_idx is None:
        # Different default seed than config.training.seed so we don't
        # always reproduce the same trial picked by data_test_path_integration_task.
        if trial_seed is None:
            trial_seed = int(getattr(config.training, "seed", 0)) + 17
        rng = np.random.default_rng(trial_seed)
        trial_idx = int(rng.integers(0, u_test.shape[0]))
    trial_idx = int(trial_idx) % u_test.shape[0]
    u_one = u_test[trial_idx]
    y_true = y_test[trial_idx]
    with torch.no_grad():
        u_t = torch.from_numpy(u_one[None]).to(device)
        y_pred, _ = net(u_t)
    y_pred = y_pred[0].cpu().numpy()
    test_trial = dict(
        idx=trial_idx,
        u=u_one,
        y_true=y_true,
        y_pred=y_pred,
        dt=float(config.task.path_integration.dt),
    )

    return dict(
        net=net,
        config=config,
        W_rec=net.W_rec.detach().cpu().numpy(),
        W_con=net.W_con.detach().cpu().numpy(),
        neuron_types=net.neuron_types,
        type_names=net.type_names,
        pen_indices=pen_indices,
        rollout=rollout,
        epg_theta=epg_theta,
        gain_data=gain_data,
        test_trial=test_trial,
        dt_s=float(net.dt),
        checkpoint=chosen,
    )


# --- panels --------------------------------------------------------------


def _panel_matrix(ax, M: np.ndarray, neuron_types, type_names, title: str):
    """Type-pair grouped W matrix, z-scored over non-zero entries (±3 clipped)."""
    if M is None:
        ax.text(0.5, 0.5, "no matrix", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off"); return
    nz = M[M != 0]
    if nz.size:
        mu, sigma = float(nz.mean()), float(nz.std())
        sigma = max(sigma, 1e-8)
    else:
        mu, sigma = 0.0, 1.0
    Z = np.where(M != 0, (M - mu) / sigma, 0.0).clip(-3.0, 3.0)
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-3.0, vmax=3.0,
                    interpolation="nearest", aspect="equal")
    nt = np.asarray(neuron_types)
    if nt.size:
        order = np.argsort(nt, kind="stable")
        b = np.where(np.diff(nt[order]) != 0)[0] + 0.5
        for x in b:
            ax.axvline(x, color="k", lw=0.3, alpha=0.5)
            ax.axhline(x, color="k", lw=0.3, alpha=0.5)
        boundaries = np.concatenate([[0], b + 0.5, [nt.size]])
        centres = (boundaries[:-1] + boundaries[1:]) / 2 - 0.5
        labels = [type_names[int(nt[order[int(c)]])] for c in centres]
        ax.set_xticks(centres); ax.set_xticklabels(labels, fontsize=TICK_FS,
                                                     rotation=45, ha="right")
        ax.set_yticks(centres); ax.set_yticklabels(labels, fontsize=TICK_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel("presynaptic", fontsize=LABEL_FS)
    ax.set_ylabel("postsynaptic", fontsize=LABEL_FS)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=TICK_FS)


def _panel_neuron_kinograph(ax, r_pop, neuron_types_sub, type_names,
                             dt_s: float, ylabel: str):
    """Per-neuron z-scored firing-rate kinograph, no title."""
    if r_pop is None or r_pop.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off"); return
    T = r_pop.shape[0]
    z = (r_pop - r_pop.mean(axis=0, keepdims=True))
    sd = r_pop.std(axis=0, keepdims=True); sd[sd < 1e-8] = 1.0
    z = (z / sd).clip(-3.0, 3.0)
    im = ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-3.0, vmax=3.0,
                    extent=[0, T * dt_s, 0, z.shape[1]],
                    interpolation="nearest")
    # Type-block lines, if multiple types
    if neuron_types_sub is not None and neuron_types_sub.size:
        nt = np.asarray(neuron_types_sub)
        order = np.argsort(nt, kind="stable")
        boundaries = np.where(np.diff(nt[order]) != 0)[0] + 0.5
        for b in boundaries:
            ax.axhline(b, color="k", lw=0.3, alpha=0.6)
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=TICK_FS)


def _panel_all_neurons_kinograph(ax, r_full: np.ndarray, neuron_types,
                                    type_names, dt_s: float):
    """Per-neuron firing-rate kinograph for ALL neurons.

    Neurons are reordered by neuron type so cell-type blocks are visible.
    Z-scored per neuron (column-wise), clipped to ±3.
    """
    if r_full is None or r_full.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes); ax.axis("off"); return
    nt = np.asarray(neuron_types)
    order = np.argsort(nt, kind="stable")
    r_sorted = r_full[:, order]
    T = r_sorted.shape[0]
    mu = r_sorted.mean(axis=0, keepdims=True)
    sd = r_sorted.std(axis=0, keepdims=True); sd[sd < 1e-8] = 1.0
    z = ((r_sorted - mu) / sd).clip(-3.0, 3.0)
    im = ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-3.0, vmax=3.0,
                    extent=[0, T * dt_s, 0, z.shape[1]],
                    interpolation="nearest")
    # Type-block lines + tick labels at block centres
    nt_sorted = nt[order]
    boundaries = np.where(np.diff(nt_sorted) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="k", lw=0.3, alpha=0.6)
    bounds_full = np.concatenate([[0], boundaries + 0.5, [nt_sorted.size]])
    centres = (bounds_full[:-1] + bounds_full[1:]) / 2 - 0.5
    labels = [type_names[int(nt_sorted[int(c)])] for c in centres]
    ax.set_yticks(centres)
    ax.set_yticklabels(labels, fontsize=TICK_FS)
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel("neuron type", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=TICK_FS)


def _panel_population_kinograph(ax, rollout: dict, epg_theta: np.ndarray,
                                 dt_s: float, n_bins: int = 32):
    """Population EPG kinograph (orientation × time), no overlay."""
    r_epg = np.asarray(rollout["r_epg"])
    T = r_epg.shape[0]
    # Wrap to (-π, π]
    theta = np.angle(np.exp(1j * np.asarray(epg_theta)))
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(theta, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    grid = np.zeros((T, n_bins), dtype=np.float32)
    cnt = np.zeros(n_bins, dtype=np.float32)
    for k, b in enumerate(bin_idx):
        grid[:, b] += r_epg[:, k]
        cnt[b] += 1.0
    cnt[cnt < 1.0] = 1.0
    grid /= cnt[None, :]
    z = (grid - grid.mean(axis=1, keepdims=True))
    sd = grid.std(axis=1, keepdims=True); sd[sd < 1e-8] = 1.0
    z = (z / sd).clip(-3.0, 3.0)
    im = ax.imshow(z.T, aspect="auto", origin="lower", cmap="RdBu_r",
                    vmin=-3.0, vmax=3.0,
                    extent=[0, T * dt_s, -np.pi, np.pi],
                    interpolation="nearest")
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=TICK_FS)
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel("orientation (rad)", fontsize=LABEL_FS)
    ax.set_title("EPG bump", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
    cb.ax.tick_params(labelsize=TICK_FS)


def _panel_hd_tracking_stacked(fig, subplotspec, rollout: dict, dt_s: float,
                                  warmup: int = 10):
    """Constant-ω rollout in the same stacked layout as the OU panel:
    top ω(t) (green), bottom HD true (green) + decoded (black).

    Returns the top axis (to attach the panel label).
    """
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=subplotspec,
                                   height_ratios=[1.0, 1.8], hspace=0.18)
    ax_top = fig.add_subplot(sub[0])
    ax_bot = fig.add_subplot(sub[1], sharex=ax_top)

    true_t = np.asarray(rollout["true_theta"])
    dec_t = np.asarray(rollout["decoded_theta"])
    u = np.asarray(rollout["u"])              # (T, 3)
    T = true_t.size
    t_axis = np.arange(T) * dt_s

    # Top: ω(t) (flat at the constant rollout ω)
    ax_top.plot(t_axis, u[:, 0], color=GT_COLOR, lw=1.2)
    ax_top.axhline(0, color="0.7", lw=0.3)
    ax_top.set_ylabel("ω (°/s)", fontsize=LABEL_FS)
    ax_top.tick_params(labelsize=TICK_FS, labelbottom=False)
    if T > warmup:
        d_uw = np.unwrap(dec_t[warmup:])
        if d_uw.std() > 1e-8 and true_t[warmup:].std() > 1e-8:
            r = float(np.corrcoef(d_uw, true_t[warmup:])[0, 1])
            r_str = f"r = {r:.3f}"
        else:
            r_str = "r = n/a"
    else:
        r_str = "r = n/a"
    err = np.angle(np.exp(1j * (dec_t - true_t)))
    rmse_deg = float(np.degrees(np.sqrt(np.mean(err ** 2))))
    ax_top.set_title(f"constant-ω rollout  ({r_str},  RMSE = {rmse_deg:.1f}°)",
                       fontsize=TITLE_FS)

    # Bottom: HD true (green) + decoded (black), wrapped
    true_wrap = np.angle(np.exp(1j * true_t))
    dec_wrap = np.angle(np.exp(1j * dec_t))
    ax_bot.plot(t_axis, true_wrap, color=GT_COLOR, lw=0.0, marker=".", ms=2.5)
    ax_bot.plot(t_axis, dec_wrap, color=PRED_COLOR, lw=0.0, marker=".", ms=0.8)
    ax_bot.set_yticks([-np.pi, 0, np.pi])
    ax_bot.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=TICK_FS)
    ax_bot.set_ylim(-np.pi - 0.15, np.pi + 0.15)
    ax_bot.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax_bot.set_ylabel("HD (rad)", fontsize=LABEL_FS)
    ax_bot.tick_params(labelsize=TICK_FS)
    return ax_top


def _panel_trial_rollout(fig, subplotspec, test_trial: dict):
    """Stacked sub-panels: top ω(t) in green, bottom HD true+decoded.

    Returns the *top* axis (used to attach the panel label).
    """
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=subplotspec,
                                   height_ratios=[1.0, 1.8], hspace=0.18)
    ax_top = fig.add_subplot(sub[0])
    ax_bot = fig.add_subplot(sub[1], sharex=ax_top)

    u = np.asarray(test_trial["u"])           # (T, 3)
    y_true = np.asarray(test_trial["y_true"])  # (T, 2)
    y_pred = np.asarray(test_trial["y_pred"])  # (T, 2)
    dt = float(test_trial["dt"])
    T = u.shape[0]
    t_axis = np.arange(T) * dt

    # Top: ω(t)
    ax_top.plot(t_axis, u[:, 0], color=GT_COLOR, lw=0.8)
    ax_top.axhline(0, color="0.7", lw=0.3)
    ax_top.set_ylabel("ω (°/s)", fontsize=LABEL_FS)
    ax_top.tick_params(labelsize=TICK_FS, labelbottom=False)
    ax_top.set_title(f"OU test trial #{int(test_trial['idx'])}",
                      fontsize=TITLE_FS)

    # Bottom: HD true (green) + decoded (black), wrapped
    theta_true = np.arctan2(y_true[:, 1], y_true[:, 0])
    theta_pred = np.arctan2(y_pred[:, 1], y_pred[:, 0])
    ax_bot.plot(t_axis, theta_true, color=GT_COLOR, lw=0.0,
                marker=".", ms=2.0)
    ax_bot.plot(t_axis, theta_pred, color=PRED_COLOR, lw=0.0,
                marker=".", ms=0.6)
    ax_bot.set_yticks([-np.pi, 0, np.pi])
    ax_bot.set_yticklabels([r"$-\pi$", "0", r"$\pi$"], fontsize=TICK_FS)
    ax_bot.set_ylim(-np.pi - 0.15, np.pi + 0.15)
    ax_bot.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax_bot.set_ylabel("HD (rad)", fontsize=LABEL_FS)
    ax_bot.tick_params(labelsize=TICK_FS)
    return ax_top


def _is_gnn(net) -> bool:
    return all(hasattr(net, n) for n in ("a", "f_theta", "g_phi"))


def _compute_tuning_data(gain_data, n_neurons, n_bins=16, warmup=10):
    """Concatenate constant-omega rollouts and build per-neuron HD curves.

    Returns:
        curves      (N, n_bins) per-neuron HD-tuning curve
        preferred   (N,) preferred HD (rad) = bin centre of the argmax
        omega_all   (T_total,) per-timepoint omega
        r_all       (T_total, N) per-timepoint firing rates
    """
    all_r, all_dec, all_om = [], [], []
    for omega, ro in gain_data:
        T = ro["r"].shape[0]
        if T <= warmup:
            continue
        all_r.append(ro["r"][warmup:])
        dec = np.angle(np.exp(1j * ro["decoded_theta"][warmup:]))
        all_dec.append(dec)
        all_om.append(np.full(T - warmup, omega))
    r_all = np.concatenate(all_r, axis=0)
    dec_all = np.concatenate(all_dec)
    om_all = np.concatenate(all_om)

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(dec_all, bins) - 1, 0, n_bins - 1)
    curves = np.zeros((n_neurons, n_bins))
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        m = bin_idx == b
        counts[b] = int(m.sum())
        if m.any():
            curves[:, b] = r_all[m].mean(axis=0)
    centres = (bins[:-1] + bins[1:]) / 2
    preferred = centres[np.argmax(curves, axis=1)]
    return curves, preferred, om_all, r_all


def _panel_preferred_direction_polar(ax, curves, preferred,
                                       neuron_types, type_names):
    """Polar scatter of preferred head direction per neuron.

    Angular position: preferred HD (peak of the per-neuron HD-tuning curve).
    Radial position:  HD-tuning strength = (max - min)/max of the curve.
    Coloured by cell type. Hulse 2024 Fig 2g analogue: shows whether
    same-type units cluster around discrete glomerular orientations
    (the canonical CX signature) or tile the ring uniformly.
    """
    nt = np.asarray(neuron_types).astype(int)
    hd_max = curves.max(axis=1)
    hd_min = curves.min(axis=1)
    strength = (hd_max - hd_min) / np.maximum(hd_max, 1e-8)
    palette = plt.get_cmap("tab10").colors
    for t in sorted(set(nt.tolist())):
        m = nt == t
        if not m.any():
            continue
        col = palette[t % len(palette)]
        ax.scatter(preferred[m], strength[m],
                    c=[col], s=24, alpha=0.85,
                    edgecolors="none", label=type_names[t])
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetagrids([0, 90, 180, 270],
                       [r"$0$", r"$\pi/2$", r"$\pi$", r"$-\pi/2$"],
                       fontsize=TICK_FS)
    ax.set_rlim(0, 1.05)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=TICK_FS - 1)
    ax.set_title("preferred HD vs tuning strength",
                  fontsize=TITLE_FS, pad=15)
    ax.legend(fontsize=TICK_FS - 1, loc="upper right",
              bbox_to_anchor=(1.30, 1.10),
              framealpha=0.85, ncol=1, handletextpad=0.3)


def _panel_tuning_scatter(ax, curves, om_all, r_all,
                           neuron_types, type_names):
    """HD vs velocity tuning scatter (Hulse Fig 2g analogue)."""
    N = curves.shape[0]
    hd_max = curves.max(axis=1)
    hd_min = curves.min(axis=1)
    hd_strength = (hd_max - hd_min) / np.maximum(hd_max, 1e-8)

    x = om_all - om_all.mean()
    x_var = (x ** 2).sum()
    vel_slope = np.zeros(N)
    if x_var > 1e-8:
        for i in range(N):
            y = r_all[:, i] - r_all[:, i].mean()
            vel_slope[i] = (x * y).sum() / x_var
    # Scale to per-1000x to put it in a readable range (Hulse convention)
    vel_slope_scaled = vel_slope * 1000.0

    nt = np.asarray(neuron_types).astype(int)
    palette = plt.get_cmap("tab10").colors
    for t in sorted(set(nt.tolist())):
        m = nt == t
        col = palette[t % len(palette)]
        ax.scatter(hd_strength[m], vel_slope_scaled[m],
                    c=[col], s=18, alpha=0.85, edgecolors="none",
                    label=type_names[t])
    ax.axhline(0, color="0.7", lw=0.4)
    ax.set_xlabel("HD-tuning strength", fontsize=LABEL_FS)
    ax.set_ylabel(r"velocity tuning ($\times 10^3$)", fontsize=LABEL_FS)
    ax.set_title("HD vs velocity tuning",
                  fontsize=TITLE_FS)
    ax.legend(fontsize=TICK_FS - 1, loc="best", framealpha=0.85, ncol=2,
              handletextpad=0.3, columnspacing=0.4)
    ax.tick_params(labelsize=TICK_FS)


def _panel_phase_shift_histogram(ax, preferred, edge_index,
                                    neuron_types, type_names):
    """Per-edge phase shift histogram (post preferred HD - pre preferred HD)
    coloured by presynaptic cell type. Hulse Fig 2i analogue."""
    src, dst = edge_index[0], edge_index[1]
    delta = preferred[dst] - preferred[src]
    delta = np.angle(np.exp(1j * delta))
    pre_types = np.asarray(neuron_types)[src]

    palette = plt.get_cmap("tab10").colors
    bins = np.linspace(-np.pi, np.pi, 36)
    for t in sorted(set(pre_types.tolist())):
        m = pre_types == t
        col = palette[t % len(palette)]
        ax.hist(np.asarray(delta)[m], bins=bins, alpha=0.55, color=col,
                 edgecolor="0.3", linewidth=0.3,
                 label=type_names[int(t)])
    ax.axvline(0, color="0.7", lw=0.4)
    ax.set_xlim(-np.pi - 0.1, np.pi + 0.1)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0",
                          r"$\pi/2$", r"$\pi$"], fontsize=TICK_FS)
    ax.set_xlabel(r"phase shift $\delta$ (rad)", fontsize=LABEL_FS)
    ax.set_ylabel("edge count", fontsize=LABEL_FS)
    ax.set_title("per-edge phase shift (pre $\\to$ post)",
                  fontsize=TITLE_FS)
    ax.legend(fontsize=TICK_FS - 1, loc="best", framealpha=0.85, ncol=2,
              handletextpad=0.3, columnspacing=0.4)
    ax.tick_params(labelsize=TICK_FS)


def _panel_bump_fwhm(ax, rollout, epg_theta, dt_s,
                       n_bins=32, fwhm_z_thresh=1.0):
    """EPG bump FWHM (degrees) over time on the constant-omega rollout."""
    r_epg = np.asarray(rollout["r_epg"])
    T = r_epg.shape[0]
    theta = np.angle(np.exp(1j * np.asarray(epg_theta)))
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    bin_rad = 2 * np.pi / n_bins
    fwhms = np.full(T, np.nan)
    for t in range(T):
        grid = np.zeros(n_bins)
        cnt = np.zeros(n_bins)
        for k, b in enumerate(bin_idx):
            grid[b] += r_epg[t, k]
            cnt[b] += 1
        cnt[cnt < 1] = 1
        grid /= cnt
        if grid.std() < 1e-8:
            continue
        z = (grid - grid.mean()) / grid.std()
        peak = int(np.argmax(z))
        z_rolled = np.roll(z, n_bins // 2 - peak)
        c = n_bins // 2
        left, right = c, c
        while left - 1 >= 0 and z_rolled[left - 1] > fwhm_z_thresh:
            left -= 1
        while right + 1 < n_bins and z_rolled[right + 1] > fwhm_z_thresh:
            right += 1
        fwhms[t] = (right - left + 1) * bin_rad
    t_axis = np.arange(T) * dt_s
    ax.plot(t_axis, np.degrees(fwhms), color="black", lw=0.8)
    ax.axhline(80, color=GT_COLOR, lw=0.7, ls="--", alpha=0.7,
                label=r"~80$^\circ$ (Hulse target)")
    ax.set_xlabel("time (s)", fontsize=LABEL_FS)
    ax.set_ylabel("EPG bump FWHM (deg)", fontsize=LABEL_FS)
    ax.set_title("bump width on constant-$\\omega$ rollout",
                  fontsize=TITLE_FS)
    ax.legend(fontsize=TICK_FS - 1, loc="upper right", framealpha=0.85)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_ylim(0, 360)


def _panel_voltage_distribution(ax, h_rollout, neuron_types, type_names):
    """Per-cell-type distribution of subthreshold $\\hat h_i(t)$.

    Violin per cell type plus mean (black bar) and ±1 SD (whiskers).
    Reveals where each population's activity actually lives -- which
    types sit near the sigmoid's linear regime, which saturate at the
    rails.
    """
    nt = np.asarray(neuron_types).astype(int)
    type_ids = sorted(set(nt.tolist()))
    data = [h_rollout[:, nt == t].ravel() for t in type_ids]
    parts = ax.violinplot(data, positions=range(len(type_ids)),
                           widths=0.7, showmeans=False, showextrema=False)
    palette = plt.get_cmap("tab10").colors
    for i, p in enumerate(parts["bodies"]):
        p.set_facecolor(palette[i % len(palette)])
        p.set_edgecolor("0.3")
        p.set_alpha(0.7)
    means = [float(np.mean(d)) for d in data]
    stds  = [float(np.std(d))  for d in data]
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.errorbar(i, m, yerr=s, fmt="o", color="black",
                     markersize=3, capsize=3, lw=1.0)
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xticks(range(len(type_ids)))
    ax.set_xticklabels([type_names[t] for t in type_ids],
                        rotation=45, ha="right", fontsize=TICK_FS)
    ax.set_ylabel(r"$\hat h_i(t)$", fontsize=LABEL_FS)
    ax.set_title("subthreshold $h$ distribution by cell type",
                  fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)


def _panel_image_from_png(ax, png_path):
    """Embed a PNG file as a borderless axis."""
    if not os.path.isfile(png_path):
        ax.text(0.5, 0.5, "snapshot missing", ha="center", va="center",
                 transform=ax.transAxes, fontsize=10, color="0.5")
        ax.axis("off")
        return
    img = plt.imread(png_path)
    ax.imshow(img, interpolation="bilinear")
    ax.axis("off")


def _latest_training_snapshot(run_dir, subdir):
    """Return the highest-step training-snapshot PNG under
    `run_dir/tmp_training/<subdir>/step_*.png`, or None if missing.
    """
    import re as _re
    pat = os.path.join(run_dir, "tmp_training", subdir, "step_*.png")
    files = glob.glob(pat)
    if not files:
        return None

    def _step_of(p):
        m = _re.search(r"step_(\d+)\.png$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    files.sort(key=_step_of)
    return files[-1]


def _panel_embedding(ax, net, neuron_types, type_names):
    """Scatter of the per-neuron latent embedding $\\mathbf{a}_i$,
    coloured by neuron type."""
    emb = net.a.detach().cpu().numpy()
    nt = np.asarray(neuron_types).astype(int)
    n_types = len(type_names)
    palette = plt.get_cmap("tab10").colors
    for t in range(n_types):
        mask = (nt == t)
        if not mask.any():
            continue
        col = palette[t % len(palette)]
        ax.scatter(emb[mask, 0], emb[mask, 1],
                    c=[col], s=14, edgecolors="none",
                    alpha=0.9, label=type_names[t])
    ax.set_xlabel(r"$a_0$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$a_1$", fontsize=LABEL_FS)
    ax.set_title(r"embedding $\mathbf{a}_i$", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=TICK_FS - 1, loc="best", framealpha=0.85,
              ncol=2, handletextpad=0.3, columnspacing=0.6)


def _panel_function_curves(ax, net, mlp_name: str, h_rollout: np.ndarray,
                             neuron_types, type_names, *,
                             square_output: bool, xlabel: str, ylabel: str,
                             title: str):
    """Mean +- SD per cell type of an MLP (f_theta or g_phi) over a fixed
    voltage domain v in [-3, 3]. Matches the training-snapshot style
    used by cx_eval._plot_gnn_functions, but renders with this figure's
    font sizes so it sits cleanly next to the other panels.
    """
    import torch

    device = next(getattr(net, mlp_name).parameters()).device
    n_pts = 400
    v_grid = torch.linspace(-3.0, 3.0, n_pts, device=device)
    a = net.a.to(device)
    N, emb_dim = a.shape
    rr = v_grid.unsqueeze(0).expand(N, -1)                 # (N, n_pts)
    rr_flat = rr.reshape(-1, 1)
    a_flat = a.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, emb_dim)
    if mlp_name == "g_phi":
        feat = torch.cat([rr_flat, a_flat], dim=1)
    else:
        feat = torch.cat([rr_flat, a_flat, torch.zeros_like(rr_flat)], dim=1)
    mlp = getattr(net, mlp_name)
    with torch.no_grad():
        out = mlp(feat).reshape(N, n_pts, -1).squeeze(-1)
    if square_output and bool(getattr(net, "_g_phi_positive", True)):
        out = out.pow(2)
    v_np = v_grid.cpu().numpy()
    out_np = out.cpu().numpy()                            # (N, n_pts)
    nt = np.asarray(neuron_types).astype(int)
    n_types = len(type_names)
    palette = plt.get_cmap("tab10").colors
    for t in range(n_types):
        mask = (nt == t)
        if not mask.any():
            continue
        col = palette[t % len(palette)]
        curves = out_np[mask]                              # (n_t, n_pts)
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        ax.plot(v_np, mean, color=col, lw=1.4, label=type_names[t])
        if std.max() > 1e-6:
            ax.fill_between(v_np, mean - std, mean + std,
                             color=col, alpha=0.15)
    ax.axhline(0, color="0.6", lw=0.4)
    ax.set_xlim(-3.0, 3.0)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=TICK_FS - 1, loc="best", framealpha=0.85,
              ncol=2, handletextpad=0.3, columnspacing=0.4)


def _panel_integration_gain(ax, gain_data, dt: float, warmup: int = 10):
    """Hulse-style scatter: measured slope (deg/s) vs true ω (deg/s)."""
    omegas, slopes = [], []
    for omega, ro in gain_data:
        dec = np.asarray(ro["decoded_theta"])
        T = dec.size
        t = np.arange(T) * dt
        if T <= warmup:
            continue
        d_uw = np.unwrap(dec[warmup:])
        t_post = t[warmup:]
        if d_uw.std() < 1e-8 or t_post.size < 2:
            slope = 0.0
        else:
            slope, _ = np.polyfit(t_post, d_uw, 1)
        omegas.append(float(omega))
        slopes.append(float(np.degrees(slope)))
    omegas = np.array(omegas); slopes = np.array(slopes)
    lim = max(float(np.abs(omegas).max()),
              float(np.abs(slopes).max()) if slopes.size else 1.0,
              1.0) * 1.10
    ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=0.8, ls="--")
    ax.axhline(0, color="0.8", lw=0.4)
    ax.axvline(0, color="0.8", lw=0.4)

    # --- Linearity-domain overlay -------------------------------------------
    # Per-point gain g = slope / ω. Flag points within `linearity_tol` of
    # g = 1 (where g is well-defined). Domain is taken as the range
    # [min ω_ok, max ω_ok] over all linear-OK points (not requiring
    # contiguity, so a few outliers in the middle don't truncate the band).
    linearity_tol = 0.25
    valid = np.abs(omegas) > 1e-8
    gains = np.full_like(omegas, np.nan)
    gains[valid] = slopes[valid] / omegas[valid]
    linear_mask = np.isfinite(gains) & (np.abs(gains - 1.0) <= linearity_tol)
    if linear_mask.sum() >= 2:
        om_ok = omegas[linear_mask]
        om_lo, om_hi = float(om_ok.min()), float(om_ok.max())
        ax.axvspan(om_lo, om_hi, color="0.6", alpha=0.18, zorder=0)
        domain_str = (f"linear: $[{om_lo:+.0f}, {om_hi:+.0f}]$"
                       r"$^\circ\!/\mathrm{s}$")
    else:
        domain_str = "linear: none"

    ax.scatter(omegas, slopes, s=10, c=PRED_COLOR, zorder=3)
    ax.text(0.03, 0.97, domain_str, transform=ax.transAxes,
             va="top", ha="left", fontsize=TICK_FS,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                        boxstyle="round,pad=0.2"))

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("true ω (°/s)", fontsize=LABEL_FS)
    ax.set_ylabel("measured slope (°/s)", fontsize=LABEL_FS)
    ax.set_title("integration gain  (target: y = x)", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)


# --- assembly ------------------------------------------------------------


def build_figure(data: dict, out_path: str, run_dir: str | None = None,
                  n_rows: int = 3):
    """Render the evolution figure.

    n_rows=3 (default): full paper figure with panels a-l.
    n_rows=2: training-time snapshot — panels a-h only, third row dropped.

    `data["test_trial"]` may be None when n_rows=2: panel g is then hidden.
    """
    plt.style.use("default")
    is_gnn = _is_gnn(data["net"])
    if n_rows == 2:
        figsize = (20, 9.5)
    else:
        figsize = (20, 14)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_rows, 4, hspace=0.55, wspace=0.42,
                          left=0.05, right=0.97, top=0.96, bottom=0.05)

    # Top row
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    # Second row
    ax_e = fig.add_subplot(gs[1, 0])
    # Panel f and g use nested gridspecs (stacked input + HD)
    ax_f_top = _panel_hd_tracking_stacked(
        fig, gs[1, 1], data["rollout"], data["dt_s"])
    if data.get("test_trial") is not None:
        ax_g_top = _panel_trial_rollout(fig, gs[1, 2], data["test_trial"])
    else:
        ax_g_top = fig.add_subplot(gs[1, 2])
        ax_g_top.axis("off")
    ax_h = fig.add_subplot(gs[1, 3])

    # (a) GT W_con
    _panel_matrix(ax_a, data["W_con"],
                   data["neuron_types"], data["type_names"],
                   "GT $W_{\\mathrm{con}}$")
    _panel_label(ax_a, "a")

    # (b) Learned W_rec
    _panel_matrix(ax_b, data["W_rec"],
                   data["neuron_types"], data["type_names"],
                   "learned $\\hat W_{\\mathrm{rec}}$")
    _panel_label(ax_b, "b")

    nt = np.asarray(data["neuron_types"])

    # (c) all neurons over time (per-neuron firing rate, sorted by type)
    _panel_all_neurons_kinograph(
        ax_c, np.asarray(data["rollout"]["r"]),
        neuron_types=data["neuron_types"], type_names=data["type_names"],
        dt_s=data["dt_s"],
    )
    _panel_label(ax_c, "c")

    # (d) per-neuron PEN, no title
    pen_idx = data["pen_indices"]
    if pen_idx is not None and pen_idx.size:
        _panel_neuron_kinograph(
            ax_d, np.asarray(data["rollout"]["r_pen"]),
            neuron_types_sub=nt[pen_idx], type_names=data["type_names"],
            dt_s=data["dt_s"], ylabel="PEN neuron",
        )
    else:
        ax_d.axis("off")
    _panel_label(ax_d, "d")

    # (e) per-neuron EPG, no title
    epg_indices = data["net"].epg_indices
    _panel_neuron_kinograph(
        ax_e, np.asarray(data["rollout"]["r_epg"]),
        neuron_types_sub=nt[epg_indices], type_names=data["type_names"],
        dt_s=data["dt_s"], ylabel="EPG neuron",
    )
    _panel_label(ax_e, "e")

    # (f) HD tracking on constant-ω rollout (stacked layout like g)
    _panel_label(ax_f_top, "f")

    # (g) trial test rollout (label on the top sub-axis)
    _panel_label(ax_g_top, "g")

    # (h) per-cell-type subthreshold h(t) distribution (NEW)
    h_rollout = np.asarray(data["rollout"]["h"])
    _panel_voltage_distribution(
        ax_h, h_rollout,
        neuron_types=data["neuron_types"],
        type_names=data["type_names"],
    )
    _panel_label(ax_h, "h")

    if n_rows < 3:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    # --- third row -----------------------------------------------------
    ax_i = fig.add_subplot(gs[2, 0])
    ax_j = fig.add_subplot(gs[2, 1])
    ax_k = fig.add_subplot(gs[2, 2])
    ax_l = fig.add_subplot(gs[2, 3])

    skip_extras = bool(run_dir and "frozen" in os.path.basename(
        os.path.abspath(run_dir)).lower())

    if skip_extras:
        # frozen_Wrec: the model fails to form a bump, so the integration-
        # gain panel and the comparative panels (j-l) are uninformative.
        for ax in (ax_i, ax_j, ax_k, ax_l):
            ax.axis("off")
    elif is_gnn:
        # (i) integration gain  (moved from former panel h)
        _panel_integration_gain(
            ax_i, data["gain_data"], data["test_trial"]["dt"],
        )
        _panel_label(ax_i, "i")
        # (j) embedding scatter
        _panel_embedding(
            ax_j, data["net"], data["neuron_types"], data["type_names"],
        )
        _panel_label(ax_j, "j")
        # (k, l) f_theta and g_phi: mean+-SD per cell type over fixed
        # v in [-3, 3] -- the same content as the training snapshots
        # but with this figure's font / box conventions.
        _panel_function_curves(
            ax_k, data["net"], "f_theta", h_rollout,
            neuron_types=data["neuron_types"],
            type_names=data["type_names"],
            square_output=False,
            xlabel=r"$\hat{h}_i$",
            ylabel=r"$f_\theta(\hat{h}_i, \mathbf{a}_i, m{=}0)$",
            title=r"$f_\theta$ (mean $\pm$ SD per type)",
        )
        _panel_label(ax_k, "k")
        _panel_function_curves(
            ax_l, data["net"], "g_phi", h_rollout,
            neuron_types=data["neuron_types"],
            type_names=data["type_names"],
            square_output=True,
            xlabel=r"$\hat{h}_j$",
            ylabel=r"$g_\phi(\hat{h}_j, \mathbf{a}_j)^2$",
            title=r"$g_\phi^2$ (mean $\pm$ SD per type)",
        )
        _panel_label(ax_l, "l")
    else:
        # Non-GNN, non-frozen path: only (i) integration gain. j, k, l hidden.
        _panel_integration_gain(
            ax_i, data["gain_data"], data["test_trial"]["dt"],
        )
        _panel_label(ax_i, "i")
        ax_j.axis("off")
        ax_k.axis("off")
        ax_l.axis("off")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_evolution] wrote {out_path}")


# --- CLI -----------------------------------------------------------------


DEFAULT_RUN_DIRS = [
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi",
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi_frozen_Wrec",
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi_fc",
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi_gnn",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", action="append", default=None,
        help="training-run directory (with config.yaml, models/, ...). "
             "May be passed multiple times to generate one figure per run.")
    p.add_argument(
        "--out_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="output directory. Each figure is written as "
             "fig_evolution_<run_basename>.png.")
    p.add_argument("--snapshot_n_steps", type=int, default=1500)
    p.add_argument("--snapshot_omega_deg", type=float, default=60.0)
    p.add_argument("--trial_seed", type=int, default=None,
                    help="seed picking the OU test trial (default: config.training.seed + 17)")
    p.add_argument("--trial_idx", type=int, default=None,
                    help="explicit test-trial index (overrides --trial_seed).")
    args = p.parse_args()

    run_dirs = args.run_dir or DEFAULT_RUN_DIRS
    os.makedirs(args.out_dir, exist_ok=True)
    for run_dir in run_dirs:
        try:
            data = _load_model_and_rollouts(
                run_dir,
                snapshot_n_steps=args.snapshot_n_steps,
                snapshot_omega_deg=args.snapshot_omega_deg,
                trial_seed=args.trial_seed,
                trial_idx=args.trial_idx,
            )
        except Exception as exc:
            print(f"[fig_evolution] SKIP {run_dir}: {exc}")
            continue
        print(f"[fig_evolution] loaded {data['checkpoint']}")
        out_path = os.path.join(
            args.out_dir,
            f"fig_evolution_{os.path.basename(os.path.abspath(run_dir))}.png",
        )
        build_figure(data, out_path, run_dir=run_dir)


if __name__ == "__main__":
    main()
