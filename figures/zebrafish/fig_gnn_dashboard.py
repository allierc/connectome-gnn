"""Paper figure: GNN dashboard --- companion of the RNN evolution figure
(Fig. 4, ``fig_evolution_zebrafish_hd_si_ipn_917_v1_selfmotion_rotation.png``)
for the message-passing GNN variant.

One ``plt.figure`` + one ``gridspec``, three row-blocks (no PNG montage):

  Row 1 (a-d): GT connectome W_con, learned per-edge gain W_rec, per-edge
               weight scatter, all-neuron rate kinograph on a constant-omega
               rollout.
  Row 2 (e-h): dIPN phase-sorted kinograph, decoded-vs-true HD tracking on
               the constant-omega rollout, one held-out swim test trial,
               rotation integration-gain scatter.  (mirrors Fig. 4 a-h)
  Row 3 (i-k): GNN-SPECIFIC --- the per-neuron learnable embedding a_i, the
               learned per-edge signalling function g_phi(v), and the learned
               node update f_theta(a_i, v).  These three panels are what the
               GNN adds over the sign-locked RNN: the recurrent operator is no
               longer a fixed sigmoid + leak but a pair of MLPs over a
               per-neuron embedding, and the figure shows what they converged
               to.

Data loading (checkpoint discovery, deterministic rollouts, test-trial pick,
task-target channel projection, heading-bin handling) is shared with the
RNN figure via ``fig_evolution._load_model_and_rollouts`` so the GNN
dashboard is driven by exactly the same probe as its RNN companion. The
panel renderers are the same ``connectome_gnn.plot_cx._panel_*`` helpers
used by Fig. 4; the GNN row reuses ``connectome_gnn.plot.plot_embedding`` /
``plot_g_phi`` and the f_theta evaluator from ``connectome_gnn.metrics``.

Usage:
    python figures/zebrafish/fig_gnn_dashboard.py \\
        --run_dir /groups/.../log/zebrafish/zebrafish_hd_si_gnn_ipn_917_v1_selfmotion_rotation \\
        --out_dir figures/zebrafish/
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
# Sibling shim: reuse the exact data-loading path of the RNN evolution figure.
sys.path.insert(0, _HERE)


# Shared typography (one place, applied to every panel for consistency).
LABEL_FS = 15
TICK_FS = 13
PANEL_LABEL_FS = 18


def _panel_label(ax, letter, y=1.04, x=-0.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LABEL_FS,
            fontweight="bold", va="bottom", ha="right")


def _gnn_embedding_panel(ax, data, device):
    """Per-neuron embedding scatter a_0 vs a_1, coloured by cell type."""
    from connectome_gnn.plot import plot_embedding
    from connectome_gnn.utils import CustomColorMap

    net = data["net"]
    nt = np.asarray(data["neuron_types"])
    n_types = len(data["type_names"])
    cmap = CustomColorMap(config=data["config"])
    plot_embedding(ax, net, nt, n_types, cmap)
    ax.set_title("")
    ax.set_xlabel(r"$a_{i,0}$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$a_{i,1}$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)


def _gnn_gphi_panel(ax, data, device):
    """Learned per-edge signalling function g_phi(v), mean +/- SD per type."""
    from connectome_gnn.plot import plot_g_phi
    from connectome_gnn.utils import CustomColorMap

    net = data["net"]
    config = data["config"]
    nt = np.asarray(data["neuron_types"])
    cmap = CustomColorMap(config=config)
    # g_phi/f_theta consume the raw subthreshold state v (no sigmoid wrap);
    # probe over v in [-3, 3]. Override xlim/ylim for the call, then restore.
    _xlim, _ylim = list(config.plotting.xlim), list(config.plotting.ylim)
    try:
        config.plotting.xlim = [-3.0, 3.0]
        config.plotting.ylim = [-1.0, 1.0]
        plot_g_phi(ax, net, config, int(net.n_units), nt, cmap, device,
                   type_names=list(data["type_names"]))
    finally:
        config.plotting.xlim = _xlim
        config.plotting.ylim = _ylim
    ax.set_title("")
    ax.set_xlabel(r"$v_j$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$g_\phi(\mathbf{a}_j, v_j)$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def _gnn_ftheta_panel(ax, data, device):
    """Learned node update f_theta(a_i, v) at zero recurrent input, per type.

    Replicates ``bump_attractor_eval._plot_gnn_functions`` (the training-time
    f_theta plot) so the dashboard panel matches what the trainer logs.
    """
    import torch

    from connectome_gnn.metrics import _batched_mlp_eval
    from connectome_gnn.utils import qualitative_colors

    net = data["net"]
    nt = np.asarray(data["neuron_types"])
    type_names = list(data["type_names"])
    n_neurons = int(net.n_units)
    n_pts = 1000
    rr_1d = torch.linspace(-3.0, 3.0, n_pts, device=device)
    rr = rr_1d.unsqueeze(0).expand(n_neurons, -1)
    # TaskGNN f_theta input is (v, a, msg); pin msg=0 to probe the bare update.
    feat_fn = lambda rr_f, emb_f: torch.cat(
        [rr_f, emb_f, torch.zeros_like(rr_f)], dim=1)
    func = _batched_mlp_eval(net.f_theta, net.a, rr, feat_fn, device)

    type_np = nt.astype(int).ravel()
    x_np = rr_1d.detach().cpu().numpy()
    func_np = func.detach().cpu().numpy()
    _type_cols = qualitative_colors(int(type_np.max()) + 1)
    for t in np.unique(type_np):
        mask = type_np == int(t)
        mean = func_np[mask].mean(axis=0)
        std = func_np[mask].std(axis=0)
        color = (_type_cols[int(t)] if int(t) < len(_type_cols) else "0.4")
        ax.plot(x_np, mean, linewidth=1.5, color=color)
        if std.max() > 1e-6:
            ax.fill_between(x_np, mean - std, mean + std, color=color,
                            alpha=0.15)
    ax.axhline(0, color="#aaaaaa", linewidth=0.5, linestyle="--")
    ax.set_xlim([-3.0, 3.0])
    ax.set_xlabel(r"$v_i$", fontsize=LABEL_FS)
    ax.set_ylabel(r"$f_\theta(\mathbf{a}_i, v_i)$", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)


def build_dashboard(data, out_path):
    """Render the whole dashboard into a single figure."""
    import torch

    from connectome_gnn.plot_cx import (
        _hd_partition_ids,
        _panel_all_neurons_kinograph,
        _panel_hd_tracking_stacked,
        _panel_integration_gain,
        _panel_matrix,
        _panel_phase_sorted_kinograph,
        _panel_trial_rollout,
        _panel_weight_scatter,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hd_part = _hd_partition_ids(data["neuron_types"], data["type_names"])
    rollout = data["rollout"]
    r_traj = np.asarray(rollout["r"])
    dt_s = data["dt_s"]

    fig = plt.figure(figsize=(20, 15))
    # Three stacked row-blocks. Rows 1-2 are 4-column (the Fig. 4 a-h
    # panels); row 3 is 3-column (the GNN-specific embedding + 2 MLPs).
    outer = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.42,
        left=0.05, right=0.97, top=0.96, bottom=0.06,
    )
    gs1 = outer[0].subgridspec(1, 4, wspace=0.42)
    gs2 = outer[1].subgridspec(1, 4, wspace=0.42)
    gs3 = outer[2].subgridspec(1, 3, wspace=0.34)

    # ---- Row 1: connectome / learned weights / activity ------------------
    ax_a = fig.add_subplot(gs1[0, 0])
    _panel_matrix(ax_a, data["W_con"], data["neuron_types"],
                  data["type_names"], "", partition=hd_part,
                  show_cbar=False, show_title=False)
    _panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs1[0, 1])
    _panel_matrix(ax_b, data["W_rec"], data["neuron_types"],
                  data["type_names"], "", partition=hd_part,
                  show_cbar=False, show_title=False, show_axis_labels=False)
    _panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs1[0, 2])
    _panel_weight_scatter(ax_c, data["W_con"], data["W_rec"])
    _panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs1[0, 3])
    _panel_all_neurons_kinograph(
        ax_d, r_traj, neuron_types=data["neuron_types"],
        type_names=data["type_names"], dt_s=dt_s,
        partition=hd_part, show_cbar=False)
    _panel_label(ax_d, "d")

    # ---- Row 2: dIPN kinograph / HD tracking / test trial / gain ---------
    ax_e = fig.add_subplot(gs2[0, 0])
    _theta = None
    for _k in ("true_theta", "theta"):
        if _k in rollout:
            _theta = np.asarray(rollout[_k])
            break
    if hd_part is not None and _theta is not None:
        _panel_phase_sorted_kinograph(
            ax_e, r_traj, _theta, dt_s=dt_s, partition=hd_part,
            show_cbar=False)
    else:
        ax_e.axis("off")
    _panel_label(ax_e, "e")

    ax_f = _panel_hd_tracking_stacked(fig, gs2[0, 1], rollout, dt_s)
    _panel_label(ax_f, "f", y=1.22)

    if data.get("test_trial") is not None:
        ax_g = _panel_trial_rollout(fig, gs2[0, 2], data["test_trial"])
    else:
        ax_g = fig.add_subplot(gs2[0, 2])
        ax_g.axis("off")
    _panel_label(ax_g, "g", y=1.22)

    ax_h = fig.add_subplot(gs2[0, 3])
    _gain = data.get("gain_data")
    _dt_h = (data["test_trial"]["dt"]
             if data.get("test_trial") else dt_s)
    if _gain:
        _panel_integration_gain(ax_h, _gain, _dt_h)
    else:
        ax_h.axis("off")
    _panel_label(ax_h, "h")

    # ---- Row 3: GNN-specific (embedding + two learned MLPs) --------------
    ax_i = fig.add_subplot(gs3[0, 0])
    _gnn_embedding_panel(ax_i, data, device)
    _panel_label(ax_i, "i")

    ax_j = fig.add_subplot(gs3[0, 1])
    _gnn_gphi_panel(ax_j, data, device)
    _panel_label(ax_j, "j")

    ax_k = fig.add_subplot(gs3[0, 2])
    _gnn_ftheta_panel(ax_k, data, device)
    _panel_label(ax_k, "k")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    from _despine import open_axes
    open_axes(fig)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", required=True,
        help="GNN training-run directory (with config.yaml, models/).")
    p.add_argument(
        "--out_dir", default=_HERE,
        help="output directory (default: figures/zebrafish/).")
    p.add_argument("--snapshot_n_steps", type=int, default=1500)
    p.add_argument("--snapshot_omega_deg", type=float, default=60.0)
    p.add_argument("--trial_seed", type=int, default=None)
    p.add_argument("--trial_idx", type=int, default=None)
    args = p.parse_args()

    from fig_evolution import _load_model_and_rollouts

    data = _load_model_and_rollouts(
        args.run_dir,
        snapshot_n_steps=args.snapshot_n_steps,
        snapshot_omega_deg=args.snapshot_omega_deg,
        trial_seed=args.trial_seed,
        trial_idx=args.trial_idx,
    )
    net = data["net"]
    if not all(hasattr(net, n) for n in ("a", "g_phi", "f_theta")):
        raise SystemExit(
            f"[fig_gnn_dashboard] {args.run_dir} is not a GNN run "
            f"(model has no a/g_phi/f_theta); use fig_evolution.py instead.")
    print(f"[fig_gnn_dashboard] loaded {data['checkpoint']}")
    out_path = os.path.join(
        args.out_dir,
        f"fig_gnn_dashboard_{os.path.basename(os.path.abspath(args.run_dir))}.png",
    )
    build_dashboard(data, out_path)
    print(f"[fig_gnn_dashboard] wrote {out_path}")


if __name__ == "__main__":
    main()
