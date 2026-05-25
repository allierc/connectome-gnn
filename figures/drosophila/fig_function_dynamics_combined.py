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
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=8)
    return hb


def _quantile_range(samples: np.ndarray, lo_q: float = 0.001,
                    hi_q: float = 0.999, pad_frac: float = 0.05) -> tuple:
    lo = float(np.quantile(samples, lo_q))
    hi = float(np.quantile(samples, hi_q))
    pad = pad_frac * (hi - lo + 1e-6)
    return lo - pad, hi + pad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rnn_config",      default="drosophila_cx_pi")
    p.add_argument("--coldale_config",  default="drosophila_cx_pi_fc")
    p.add_argument("--gnn_config",
                   default="drosophila_cx_pi_gnn_tailloss_unsquared")
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

    # --- Load all three models ---
    rnn     = _load_model(args.rnn_config,     device)
    coldale = _load_model(args.coldale_config, device)
    gnn     = _load_model(args.gnn_config,     device)

    # --- OU rollouts (same seed → same stimulus, different model dynamics) ---
    h_rnn     = _run_ou(rnn,     args.n_steps_ou, device, args.seed)
    h_coldale = _run_ou(coldale, args.n_steps_ou, device, args.seed)
    h_gnn     = _run_ou(gnn,     args.n_steps_ou, device, args.seed)
    print(f"OU rollouts: RNN {h_rnn.shape} | fully connected {h_coldale.shape} | "
          f"GNN {h_gnn.shape}")

    # --- RNN-type teachers: f(h) = -h/τ, g(h) = σ(h) ---
    V_RANGE_RNN = 10.0
    v_rnn_static = np.linspace(-V_RANGE_RNN, V_RANGE_RNN, args.n_static)

    # Known-ODE RNN
    tau_rnn = float(rnn.tau)
    f_rnn = -h_rnn / tau_rnn
    sigma_rnn_fn = rnn._sigma
    sigma_rnn_name = getattr(rnn, "recurrent_activation_name", "sigma")
    with torch.no_grad():
        g_rnn = sigma_rnn_fn(torch.from_numpy(h_rnn).to(device)).cpu().numpy()
        f_rnn_static = -v_rnn_static / tau_rnn
        g_rnn_static = sigma_rnn_fn(torch.from_numpy(
            v_rnn_static.astype(np.float32)).to(device)).cpu().numpy()

    # fully connected RNN
    tau_col = float(coldale.tau)
    f_col = -h_coldale / tau_col
    sigma_col_fn = coldale._sigma
    sigma_col_name = getattr(coldale, "recurrent_activation_name", "sigma")
    with torch.no_grad():
        g_col = sigma_col_fn(torch.from_numpy(h_coldale).to(device)).cpu().numpy()
        f_col_static = -v_rnn_static / tau_col
        g_col_static = sigma_col_fn(torch.from_numpy(
            v_rnn_static.astype(np.float32)).to(device)).cpu().numpy()

    # --- GNN teacher ---
    f_gnn, g_gnn, _ = _gnn_eval(gnn, h_gnn, device)
    v_gnn_static, f_gnn_static, g_gnn_static = _gnn_static(
        gnn, args.v_range, args.n_static, device)
    f_gnn_static_mean, f_gnn_static_std = f_gnn_static.mean(0), f_gnn_static.std(0)
    g_gnn_static_mean, g_gnn_static_std = g_gnn_static.mean(0), g_gnn_static.std(0)

    # --- 3 × 2 figure: 3 teachers × (drift, firing-rate-NL). ---
    # Row 1 (a, b): Known-ODE RNN     — hardcoded -h/τ and σ.
    # Row 2 (c, d): fully connected RNN — same hardcoded form, different W_rec.
    # Row 3 (e, f): GNN              — learned f_θ and g_φ.
    fig, axes = plt.subplots(3, 2, figsize=(9, 10.5))
    (ax_a, ax_b) = axes[0]
    (ax_c, ax_d) = axes[1]
    (ax_e, ax_f) = axes[2]

    # Y-limits.
    F_YLIM_RNN = (-100.0, 100.0)
    G_YLIM_RNN = (-0.1,   1.1)

    # Auto-tuned x and y limits for the GNN panels so the operating
    # distribution + static curve fit cleanly.
    GNN_XLIM   = _quantile_range(h_gnn.reshape(-1))
    F_YLIM_GNN = _quantile_range(f_gnn.reshape(-1))
    G_YLIM_GNN = _quantile_range(g_gnn.reshape(-1))

    # (a, b) Known-ODE RNN
    hb_a = _panel(ax_a,
        h_rnn.reshape(-1), f_rnn.reshape(-1),
        v_rnn_static, f_rnn_static, None, V_RANGE_RNN,
        ylabel=rf"$-\hat h / \tau$  ($\tau = {tau_rnn:.3g}$ s)",
        title="Known-ODE RNN — leak (OU rollout)",
        ylim=F_YLIM_RNN)
    hb_b = _panel(ax_b,
        h_rnn.reshape(-1), g_rnn.reshape(-1),
        v_rnn_static, g_rnn_static, None, V_RANGE_RNN,
        ylabel=rf"$\sigma(\hat h)$  ($\sigma$ = {sigma_rnn_name})",
        title=r"Known-ODE RNN — $\sigma$ (OU rollout)",
        ylim=G_YLIM_RNN)

    # (c, d) fully connected RNN
    hb_c = _panel(ax_c,
        h_coldale.reshape(-1), f_col.reshape(-1),
        v_rnn_static, f_col_static, None, V_RANGE_RNN,
        ylabel=rf"$-\hat h / \tau$  ($\tau = {tau_col:.3g}$ s)",
        title="fully connected RNN — leak (OU rollout)",
        ylim=F_YLIM_RNN)
    hb_d = _panel(ax_d,
        h_coldale.reshape(-1), g_col.reshape(-1),
        v_rnn_static, g_col_static, None, V_RANGE_RNN,
        ylabel=rf"$\sigma(\hat h)$  ($\sigma$ = {sigma_col_name})",
        title=r"fully connected RNN — $\sigma$ (OU rollout)",
        ylim=G_YLIM_RNN)

    # (e, f) GNN
    hb_e = _panel(ax_e,
        h_gnn.reshape(-1), f_gnn.reshape(-1),
        v_gnn_static, f_gnn_static_mean, f_gnn_static_std,
        args.v_range,
        ylabel=r"$f_\theta(\hat h_i, \mathbf{a}_i, m{=}0)$",
        title=r"GNN — $f_\theta$ (OU rollout)",
        ylim=F_YLIM_GNN, xlim=GNN_XLIM)
    hb_f = _panel(ax_f,
        h_gnn.reshape(-1), g_gnn.reshape(-1),
        v_gnn_static, g_gnn_static_mean, g_gnn_static_std,
        args.v_range, ylabel=r"$g_\phi$",
        title=r"GNN — $g_\phi$ (OU rollout)",
        ylim=G_YLIM_GNN, xlim=GNN_XLIM)

    for ax, hb in [
        (ax_a, hb_a), (ax_b, hb_b),
        (ax_c, hb_c), (ax_d, hb_d),
        (ax_e, hb_e), (ax_f, hb_f),
    ]:
        cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("log10(count)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # Panel labels a–f at top-left of each subplot.
    for ax, letter in zip(
        [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f],
        list("abcdef"),
    ):
        ax.text(-0.12, 1.02, letter, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()

    out = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fig_function_dynamics_combined.png",
    )
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
