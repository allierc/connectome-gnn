"""3 × 2 operating-range hexbin for three CX teachers, all on the same
natural OU velocity stream:

    Row 1 — Known-ODE RNN (drosophila_cx_pi)             -h/tau, sigma
    Row 2 — fully connected RNN (drosophila_cx_pi_fc)    -h/tau, sigma
    Row 3 — GNN (drosophila_cx_pi_gnn_tailloss_unsquared) f_theta, g_phi

Output: docs/figure/fig_function_dynamics_combined.png

CLI:
    python docs/figure/fig_function_dynamics_combined.py
    python docs/figure/fig_function_dynamics_combined.py --n_steps_ou 10000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connectome_gnn.utils import log_path, load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model
from connectome_gnn.generators.utils import generate_path_integration_batch


def _load_model(config_name: str, device: torch.device, prefer_epoch=None):
    config, _ = load_run_config(config_name, explicit_output_root=False, task="train")
    ckpt_dir = os.path.join(log_path(config.config_file), "models")
    cands = sorted(
        glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
        key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")),
    )
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    if prefer_epoch is None and "gnn_tailloss" in config_name:
        prefer_epoch = 5
    elif prefer_epoch is None and "gnn_epg" in config_name:
        prefer_epoch = 3
    ckpt_path = cands[-1]
    if prefer_epoch is not None:
        match = [p_ for p_ in cands
                 if int(p_.rsplit("_", 1)[1].rstrip(".pt")) == prefer_epoch]
        if match:
            ckpt_path = match[0]
    model = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"loaded {config_name}: {ckpt_path}")
    return model


def _run_ou(net, n_steps: int, device: torch.device, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    batch = generate_path_integration_batch(
        batch_size=1, n_steps=n_steps,
        dt=float(net.dt), device=device, rng=rng,
    )
    u_t = batch.stimulus
    net.eval()
    with torch.no_grad():
        _, h = net(u_t)
    return h[0].cpu().numpy()


def _gnn_eval(net, h_traj: np.ndarray, device: torch.device):
    """Evaluate (f_theta, g_phi[^2]) for a GNN teacher along h_traj."""
    g_squared = bool(getattr(net, "_g_phi_positive", True))
    h_t = torch.from_numpy(h_traj.astype(np.float32)).to(device)
    T, N = h_t.shape
    a_full = net.a.unsqueeze(0).expand(T, -1, -1)
    zero_msg = torch.zeros(T, N, 1, device=device)
    with torch.no_grad():
        f_in = torch.cat([h_t.unsqueeze(-1), a_full, zero_msg], dim=-1)
        f_dyn = net.f_theta(f_in).squeeze(-1).cpu().numpy()
        g_in = torch.cat([h_t.unsqueeze(-1), a_full], dim=-1)
        g_raw = net.g_phi(g_in).squeeze(-1).cpu().numpy()
        g_dyn = g_raw ** 2 if g_squared else g_raw
    return f_dyn, g_dyn, g_squared


def _gnn_static(net, v_range: float, n_pts: int, device: torch.device):
    g_squared = bool(getattr(net, "_g_phi_positive", True))
    N = int(net.n_units)
    v_grid = torch.linspace(-v_range, v_range, n_pts, device=device)
    rr = v_grid.unsqueeze(0).expand(N, -1).unsqueeze(-1)
    a_exp = net.a.unsqueeze(1).expand(-1, n_pts, -1)
    z_msg = torch.zeros_like(rr)
    with torch.no_grad():
        f_static = net.f_theta(
            torch.cat([rr, a_exp, z_msg], dim=-1)).squeeze(-1).cpu().numpy()
        g_raw = net.g_phi(
            torch.cat([rr, a_exp], dim=-1)).squeeze(-1).cpu().numpy()
        g_static = g_raw ** 2 if g_squared else g_raw
    return v_grid.cpu().numpy(), f_static, g_static


def _panel(ax, v_samples, fn_samples, v_static, fn_static_mean, fn_static_std,
           v_range, ylabel, title, ylim, xlim=None):
    """v_range sets a symmetric ±v_range x-axis; xlim overrides it explicitly."""
    if xlim is None:
        xlo, xhi = -v_range, v_range
    else:
        xlo, xhi = xlim
    ylo, yhi = ylim
    hb = ax.hexbin(v_samples, fn_samples,
                   gridsize=60, cmap="Blues", mincnt=1, bins="log",
                   extent=(xlo, xhi, ylo, yhi))
    ax.plot(v_static, fn_static_mean, color="#cc4444", lw=1.6)
    if fn_static_std is not None:
        ax.fill_between(v_static,
                        fn_static_mean - fn_static_std,
                        fn_static_mean + fn_static_std,
                        color="#cc4444", alpha=0.15)
    ax.axhline(0, color="0.6", lw=0.5, ls="--")
    ax.axvline(0, color="0.6", lw=0.5, ls="--")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$\hat h(t)$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=8)
    return hb


def _quantile_range(samples: np.ndarray, lo_q: float = 0.001,
                    hi_q: float = 0.999, pad_frac: float = 0.05) -> tuple:
    lo = float(np.quantile(samples, lo_q))
    hi = float(np.quantile(samples, hi_q))
    pad = pad_frac * (hi - lo + 1e-6)
    return lo - pad, hi + pad


MODELS = [
    ("drosophila_cx_pi_epg_no_tv_cv0",        "Known-ODE no-TV"),
    ("drosophila_cx_pi_epg_tv_cv0",           "Known-ODE $+$TV"),
    ("drosophila_cx_pi_gnn_epg_no_tv_cv0",    "GNN no-TV"),
    ("drosophila_cx_pi_gnn_epg_tv_cv0",       "GNN $+$TV"),
    ("drosophila_cx_pi_fc_epg_cv0",           "fully connected"),
    ("drosophila_cx_pi_frozen_Wrec_epg_cv0",  "frozen $W^{\\mathrm{rec}}$"),
]


def _is_gnn_type(net) -> bool:
    """GNN-type if it carries learned drift/message MLPs."""
    return hasattr(net, "f_theta") and hasattr(net, "g_phi")


def _drift_and_rate(net, h_traj, device, args):
    """Return (v_samp, f_samp, g_samp, v_static, f_static_m, f_static_s,
    g_static_m, g_static_s, xlim, f_ylim, g_ylim, f_label, g_label) for
    one model. RNN-type uses hardcoded -h/tau + sigma; GNN-type uses
    learned f_theta + g_phi (mean ± SD across nodes for the static curve).
    """
    if _is_gnn_type(net):
        f_dyn, g_dyn, _ = _gnn_eval(net, h_traj, device)
        v_static, f_static, g_static = _gnn_static(
            net, args.v_range, args.n_static, device)
        f_static_m = f_static.mean(0); f_static_s = f_static.std(0)
        g_static_m = g_static.mean(0); g_static_s = g_static.std(0)
        xlim   = _quantile_range(h_traj.reshape(-1))
        f_ylim = _quantile_range(f_dyn.reshape(-1))
        g_ylim = _quantile_range(g_dyn.reshape(-1))
        return (h_traj.reshape(-1), f_dyn.reshape(-1), g_dyn.reshape(-1),
                v_static, f_static_m, f_static_s, g_static_m, g_static_s,
                xlim, f_ylim, g_ylim,
                r"$f_\theta(\hat h_i, \mathbf{a}_i, m{=}0)$",
                r"$g_\phi$")
    # RNN-type teacher
    V_RANGE_RNN = 10.0
    v_static = np.linspace(-V_RANGE_RNN, V_RANGE_RNN, args.n_static)
    tau = float(net.tau)
    f_dyn = -h_traj / tau
    sigma_fn = net._sigma
    sigma_name = getattr(net, "recurrent_activation_name", "sigma")
    with torch.no_grad():
        g_dyn = sigma_fn(torch.from_numpy(h_traj).to(device)).cpu().numpy()
        f_static = -v_static / tau
        g_static = sigma_fn(torch.from_numpy(
            v_static.astype(np.float32)).to(device)).cpu().numpy()
    xlim   = (-V_RANGE_RNN, V_RANGE_RNN)
    f_ylim = (-100.0, 100.0)
    g_ylim = (-0.1, 1.1)
    return (h_traj.reshape(-1), f_dyn.reshape(-1), g_dyn.reshape(-1),
            v_static, f_static, None, g_static, None,
            xlim, f_ylim, g_ylim,
            rf"$-\hat h / \tau$  ($\tau = {tau:.3g}$ s)",
            rf"$\sigma(\hat h)$  ($\sigma$ = {sigma_name})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rnn_config",      default=None,
                   help="legacy single-model override (overrides MODELS).")
    p.add_argument("--coldale_config",  default=None,
                   help="legacy: fully-connected config (overrides MODELS).")
    p.add_argument("--gnn_config",      default=None,
                   help="legacy: GNN config (overrides MODELS).")
    p.add_argument("--n_steps_ou", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--v_range", type=float, default=3.0)
    p.add_argument("--n_static", type=int, default=400)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)

    # Build the model list — legacy 3-config args override MODELS for
    # backward compatibility with older invocations.
    if args.rnn_config or args.coldale_config or args.gnn_config:
        model_list = []
        if args.rnn_config:
            model_list.append((args.rnn_config, "Known-ODE RNN"))
        if args.coldale_config:
            model_list.append((args.coldale_config, "fully connected"))
        if args.gnn_config:
            model_list.append((args.gnn_config, "GNN"))
    else:
        model_list = list(MODELS)

    # Pair conditions two-per-row so the resulting grid is
    # ceil(len(MODELS)/2) rows x 4 columns. Each row contains two
    # conditions placed side by side, each occupying a (drift,
    # firing-rate non-linearity) column pair.
    n_per_row = 2
    n_rows = (len(model_list) + n_per_row - 1) // n_per_row
    n_cols = 2 * n_per_row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.0, 2.17 * n_rows),
                              squeeze=False)
    letters = "abcdefghijklmnop"
    handles = []
    for r, (cfg, label) in enumerate(model_list):
        net = _load_model(cfg, device)
        h = _run_ou(net, args.n_steps_ou, device, args.seed)
        (v_s, f_s, g_s, v_static, fsm, fss, gsm, gss,
         xlim, f_ylim, g_ylim, f_label, g_label) = _drift_and_rate(
            net, h, device, args)
        row = r // n_per_row
        col = (r % n_per_row) * 2
        ax_f = axes[row, col]
        ax_g = axes[row, col + 1]
        hb_f = _panel(ax_f, v_s, f_s, v_static, fsm, fss,
                      None, ylabel=f_label,
                      title=f"{label}: drift", ylim=f_ylim, xlim=xlim)
        hb_g = _panel(ax_g, v_s, g_s, v_static, gsm, gss,
                      None, ylabel=g_label,
                      title=f"{label}: $\\sigma(\\hat h)$",
                      ylim=g_ylim, xlim=xlim)
        handles.extend([(ax_f, hb_f), (ax_g, hb_g)])
        ax_f.text(-0.12, 1.02, letters[2 * r], transform=ax_f.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="right")
        ax_g.text(-0.12, 1.02, letters[2 * r + 1], transform=ax_g.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="right")

    for ax, hb in handles:
        cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("log10(count)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.10,
                         hspace=0.55, wspace=0.55)
    for r_, row in enumerate(axes):
        for c_, ax_ in enumerate(row):
            ax_.set_box_aspect(1.0)
    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig_function_dynamics_combined.png",
    )
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
