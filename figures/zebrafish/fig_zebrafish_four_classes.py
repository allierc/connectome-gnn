"""Four functional classes of zebrafish HD-circuit neurons on the 3D anatomy.

Classifies each of the 731 neurons in the trained ``zebrafish_hd_si_dipn``
model by combining three per-neuron scores:

    mi[i]       = I(h[:, i] ; theta) — plug-in histogram MI in bits
    wmag[i]    = ||W_out[:, i]||_2 — magnitude of readout weight column
                  (zero for i >= 443, since the decoder slices the dIPN block)
    swim[i]    = swim-triggered modulation: rms of the swim-onset-locked STA
                  of h[:, i] over a ±0.5 s window, divided by the burn-in std

Four classes (decision-tree on quantile thresholds, see ``--mi_q`` /
``--w_q`` / ``--swim_q``):

    R  bump representation     mi high  &  wmag high   (read-out HD code)
    L  latent / redundant      mi high  &  wmag low    (encodes HD, ignored
                                                        by the linear readout)
    D  driver / updater        mi low   &  swim high   (swim-locked, no HD)
    Z  leak / uninvolved       mi low   &  swim low

Outputs:
    fig_zebrafish_four_classes.png  — 2×2 dorsal 3D anatomy, one panel per class
    fig_zebrafish_four_classes.csv  — per-neuron table (scores + class + type)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig_zebrafish_anatomy_3d_voltage_anim import (
    _load, _run_swim, _run_single_impulse,
    _model_index_to_bodyid, _load_skeletons_in_model_order,
)
from fig_zebrafish_readout_mi import _mi_neuron, _category_of
from connectome_gnn.utils import (
    load_data_root_from_json, set_data_root, log_path,
)


def _load_with_override(config_name, device, ckpt_dir=None):
    """``_load`` but with an optional checkpoint-directory override so a
    config can be paired with a renamed run dir (e.g. ``zebrafish_hd_si_dipn``
    yaml + checkpoints from ``log/zebrafish/zebrafish_hd_si_dipn_L/``)."""
    if ckpt_dir is None:
        return _load(config_name, device)
    # Re-create the model from the yaml, then load state_dict from `ckpt_dir`.
    import glob as _glob
    import torch as _torch
    from connectome_gnn.models.utils import load_run_config
    from connectome_gnn.models.registry import create_model
    config, _ = load_run_config(config_name, explicit_output_root=False,
                                 task="train")
    cands = sorted(
        _glob.glob(os.path.join(ckpt_dir, "best_model_with_0_graphs_*.pt")),
        key=lambda p_: int(p_.rsplit("_", 1)[1].rstrip(".pt")),
    )
    if not cands:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    ckpt_path = cands[-1]
    model = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type,
        config=config, device=device,
    )
    state = _torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"loaded {config_name} (override): {ckpt_path}")
    return model, config


# ── colour scheme: one accent colour per class, all somas on dark-grey ──
CLASS_COLOR = {
    "R": "#e41a1c",   # red:    representation (read out)
    "L": "#ff7f00",   # orange: latent / redundant (encodes, not read out)
    "D": "#377eb8",   # blue:   driver (swim updater)
    "Z": "#7f7f7f",   # grey:   leak / uninvolved
}
CLASS_LABEL = {
    "R": "R — bump representation",
    "L": "L — latent / redundant",
    "D": "D — driver / updater",
    "Z": "Z — leak / uninvolved",
}
CLASS_ORDER = ["R", "L", "D", "Z"]
PANEL_LETTER = {"R": "a", "L": "b", "D": "c", "Z": "d"}


def _project_2d(xyz, elev, azim):
    """matplotlib mplot3d-equivalent projection of (N, 3) world -> (N, 2)."""
    e = np.deg2rad(elev)
    a = np.deg2rad(azim)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    R = np.array([[-sa,        ca,        0.0],
                  [-ca * se, -sa * se,   ce ]])
    return xyz @ R.T


def _swim_modulation(h, swim_onset_mask, dt, win_s=0.5):
    """Per-neuron rms of the swim-triggered STA over ±win_s, divided by the
    baseline std of h. Output shape (N,)."""
    T, N = h.shape
    L = int(round(win_s / dt))
    onsets = np.where(swim_onset_mask)[0]
    # drop onsets too close to the boundary
    onsets = onsets[(onsets >= L) & (onsets + L < T)]
    if len(onsets) == 0:
        return np.zeros(N, dtype=np.float32)

    # baseline std away from swim windows
    busy = np.zeros(T, dtype=bool)
    for o in onsets:
        busy[max(0, o - L): min(T, o + L)] = True
    quiet = h[~busy] if (~busy).any() else h
    sd = quiet.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)

    # swim-triggered average (subtract per-neuron pre-event mean per onset)
    sta = np.zeros((2 * L, N), dtype=np.float32)
    for o in onsets:
        seg = h[o - L: o + L]
        sta += seg - seg[:L].mean(axis=0, keepdims=True)
    sta /= len(onsets)
    return np.sqrt((sta ** 2).mean(axis=0)) / sd


def _classify(mi, wmag, swim, mi_q=0.5, w_q=0.5, swim_q=0.5):
    """Return (class_arr, thresholds) — element-wise class in {R,L,D,Z}."""
    mi_thr = float(np.quantile(mi, mi_q))
    # Only score wmag relative to its non-zero (dIPN) entries; otherwise the
    # split would put every non-dIPN cell into the "low w_out" bucket which
    # is trivially true.
    wmag_dipn = wmag[wmag > 0]
    w_thr = float(np.quantile(wmag_dipn, w_q)) if len(wmag_dipn) else 0.0
    swim_thr = float(np.quantile(swim, swim_q))

    cls = np.empty(len(mi), dtype="<U1")
    for i in range(len(mi)):
        if mi[i] >= mi_thr:
            cls[i] = "R" if wmag[i] >= w_thr else "L"
        else:
            cls[i] = "D" if swim[i] >= swim_thr else "Z"
    return cls, {"mi": mi_thr, "wmag": w_thr, "swim": swim_thr}


def _extract_per_neuron_segments(neurons):
    """List of (E_i, 2, 3) arrays per model neuron (empty if missing)."""
    out = []
    for n in neurons:
        if n is None:
            out.append(np.zeros((0, 2, 3), dtype=np.float32))
            continue
        nodes = n.nodes
        child = nodes[nodes.parent_id != -1]
        if len(child) == 0:
            out.append(np.zeros((0, 2, 3), dtype=np.float32))
            continue
        parent_xyz = nodes.set_index("node_id").loc[
            child.parent_id.values, ["x", "y", "z"]
        ].values
        child_xyz = child[["x", "y", "z"]].values
        out.append(np.stack([parent_xyz, child_xyz], axis=1).astype(np.float32))
    return out


def _per_neuron_soma(neurons):
    """One (x, y, z) per model neuron (NaN if no skeleton)."""
    out = np.full((len(neurons), 3), np.nan, dtype=np.float32)
    for i, n in enumerate(neurons):
        if n is None:
            continue
        nodes = n.nodes
        i_max = int(nodes.radius.values.argmax())
        out[i] = (float(nodes.iloc[i_max].x),
                  float(nodes.iloc[i_max].y),
                  float(nodes.iloc[i_max].z))
    return out


def _render_panels(seg_per_neuron, somas, cls, mi, wmag, swim,
                   type_names_per_neuron, thresholds, out_path,
                   elev=90.0, azim=-90.0, background="black",
                   grey_color=(0.30, 0.30, 0.30),
                   grey_lw=0.2, grey_alpha=0.30,
                   soma_size=14):
    from matplotlib.collections import LineCollection

    text_color = "white" if background == "black" else "black"

    # Pre-project everything once — same view for all four panels.
    all_segs2d = []
    for segs3d in seg_per_neuron:
        if len(segs3d) == 0:
            all_segs2d.append(np.zeros((0, 2, 2), dtype=np.float32))
        else:
            s = _project_2d(segs3d.reshape(-1, 3), elev, azim).reshape(-1, 2, 2)
            all_segs2d.append(s.astype(np.float32))
    grey_backdrop = np.concatenate(all_segs2d, axis=0) \
        if all_segs2d else np.zeros((0, 2, 2))
    soma_xy = _project_2d(somas, elev, azim)
    valid_soma = np.isfinite(soma_xy[:, 0])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0),
                              facecolor=background)
    for ax in axes.flat:
        ax.set_facecolor(background)
        ax.set_aspect("equal")
        ax.set_axis_off()

    # Compute a common axis limit from the backdrop so panels share scale.
    if len(grey_backdrop):
        all_pts = grey_backdrop.reshape(-1, 2)
        x_lo, y_lo = np.percentile(all_pts, 1, axis=0)
        x_hi, y_hi = np.percentile(all_pts, 99, axis=0)
        pad_x = 0.04 * (x_hi - x_lo)
        pad_y = 0.04 * (y_hi - y_lo)
    else:
        x_lo, x_hi, y_lo, y_hi, pad_x, pad_y = -1, 1, -1, 1, 0, 0

    for ax, klass in zip(axes.flat, CLASS_ORDER):
        in_cls = (cls == klass) & valid_soma
        other = (~in_cls) & valid_soma
        # Other-class somas as faint backdrop dots.
        other_dot = (0.55, 0.55, 0.55) if background == "white" else (0.4, 0.4, 0.4)
        if other.any():
            ax.scatter(soma_xy[other, 0], soma_xy[other, 1],
                       c=[other_dot], s=3, alpha=0.30,
                       edgecolors="none", zorder=2)
        # In-class skeletons overlaid on the grey backdrop in the class colour,
        # so the panel shows *where the projections of this class go*, not
        # just where the cell bodies sit.
        in_class_segs = [all_segs2d[i] for i in np.where(in_cls)[0]
                          if len(all_segs2d[i])]
        if in_class_segs:
            ax.add_collection(LineCollection(
                np.concatenate(in_class_segs, axis=0),
                colors=[CLASS_COLOR[klass]],
                linewidths=0.35, alpha=0.55, zorder=3,
            ))
        # This-class somas in the accent colour, on top of the skeletons.
        if in_cls.any():
            ax.scatter(soma_xy[in_cls, 0], soma_xy[in_cls, 1],
                       c=CLASS_COLOR[klass], s=soma_size,
                       edgecolors="none", zorder=4)

        n_in = int(in_cls.sum())
        ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
        ax.set_ylim(y_lo - pad_y, y_hi + pad_y)

        # Bold panel letter top-left + compact class summary just under it.
        ax.text(0.02, 0.97, PANEL_LETTER[klass], transform=ax.transAxes,
                ha="left", va="top", fontsize=16, fontweight="bold",
                color=text_color)
        ax.text(0.08, 0.965, f"{CLASS_LABEL[klass]}   (n={n_in})",
                transform=ax.transAxes, ha="left", va="top", fontsize=10,
                color=text_color)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        hspace=0.04, wspace=0.02)
    fig.savefig(out_path, dpi=210, facecolor=background, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="zebrafish_hd_si_dipn")
    p.add_argument("--n_steps", type=int, default=12000,
                   help="swim-rollout length (in dt units)")
    p.add_argument("--rollout", default="periodic",
                   choices=["swim", "periodic"],
                   help="probe stimulus to use. 'periodic' (default) "
                        "fires one --swim_direction swim impulse every "
                        "--swim_interval_s seconds for the entire "
                        "rollout, matching the kinograph rollout; "
                        "'swim' uses the Poisson training distribution.")
    p.add_argument("--swim_interval_s", type=float, default=0.3)
    p.add_argument("--swim_magnitude_rad", type=float, default=0.393)
    p.add_argument("--swim_direction", default="L", choices=["L", "R"],
                   help="direction of every impulse in the periodic train")
    p.add_argument("--burn_in_s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_theta_bins", type=int, default=32)
    p.add_argument("--n_r_bins", type=int, default=20)
    p.add_argument("--win_s", type=float, default=0.5,
                   help="swim-triggered average window (s)")
    p.add_argument("--mi_q", type=float, default=0.5,
                   help="quantile threshold on MI (above = high)")
    p.add_argument("--w_q", type=float, default=0.5,
                   help="quantile threshold on |W_out| (over dIPN cells only)")
    p.add_argument("--swim_q", type=float, default=0.5,
                   help="quantile threshold on swim-triggered modulation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--anatomy_dir",
                   default=os.path.join(here, "zebrafish_anatomy_HD"))
    p.add_argument("--connectome_dir",
                   default=os.path.join(here, "zebrafish_connectome_HD"))
    p.add_argument("--downsample", type=int, default=10)
    p.add_argument("--elev", type=float, default=90.0)
    p.add_argument("--azim", type=float, default=-90.0)
    p.add_argument("--bg", default="white", choices=["black", "white"])
    p.add_argument("--run_name", default=None,
                   help="run directory name under log/zebrafish/ to load the "
                        "checkpoint from; when omitted, _load uses the "
                        "default run for the given config")
    p.add_argument("--ckpt_dir", default=None,
                   help="explicit override; takes precedence over --run_name")
    p.add_argument("--out", default=os.path.join(here, "fig_zebrafish_four_classes.png"))
    p.add_argument("--csv_out", default=os.path.join(here, "fig_zebrafish_four_classes.csv"))
    p.add_argument("--output_root", default=None)
    args = p.parse_args()

    if args.output_root:
        set_data_root(args.output_root)
    else:
        try:
            set_data_root(load_data_root_from_json())
        except FileNotFoundError:
            pass

    device = torch.device(args.device)

    # 1) model + W_out (config <- args.model yaml, weights <- run dir)
    ckpt_dir = args.ckpt_dir or (
        log_path("zebrafish", args.run_name, "models") if args.run_name else None
    )
    print(f"[1/5] loading model config={args.model}  ckpt_dir={ckpt_dir}")
    net, _ = _load_with_override(args.model, device, ckpt_dir=ckpt_dir)
    dt = float(net.dt)
    type_names = list(net.type_names)
    neuron_types = np.asarray(net.neuron_types).astype(int)
    type_per_neuron = np.array([type_names[t] for t in neuron_types])
    W_out = net.W_out.detach().cpu().numpy()             # (2, 443)
    N = len(neuron_types)
    wmag = np.zeros(N, dtype=np.float32)
    wmag[: W_out.shape[1]] = np.linalg.norm(W_out, axis=0)
    print(f"      N={N}  n_readout={W_out.shape[1]}  n_types={len(type_names)}")

    # 2) rollout — periodic single-direction swim impulses (controlled
    # probe, matches the kinograph rollout): every interval_s seconds we
    # deliver one swim of magnitude `magnitude_rad`. Compared with the
    # Poisson swim distribution used at training time this gives more
    # swim onsets per second and a regular heading-sweep schedule, so
    # the MI / W_out / swim-modulation scores are estimated on the same
    # dynamical regime that drives the kinograph figure
    # (\cref{fig:zhd_pref_angle_kinograph}).
    if args.rollout == "swim":
        print(f"[2/5] Poisson swim rollout n_steps={args.n_steps} "
              f"({args.n_steps * dt:.0f} s, seed={args.seed})")
        h, theta, omega, _decoded, _turn_lr, _swim_fb = _run_swim(
            net, args.n_steps, dt, device, seed=args.seed)
    else:
        print(f"[2/5] periodic-{args.swim_direction} rollout n_steps={args.n_steps} "
              f"({args.n_steps * dt:.0f} s; Δt={args.swim_interval_s}s, "
              f"mag={args.swim_magnitude_rad:.3f} rad)")
        h, theta, omega, _decoded, _turn_lr, _swim_fb = _run_single_impulse(
            net, args.n_steps, dt, device,
            direction=args.swim_direction,
            magnitude_rad=args.swim_magnitude_rad,
            t_event_s=0.0,
            interval_s=args.swim_interval_s,
        )
    burn = int(args.burn_in_s / dt)
    h = h[burn:]
    theta = theta[burn:]
    omega = omega[burn:]

    # 3) per-neuron scores
    print(f"[3/5] scoring per neuron")
    mi = np.array([_mi_neuron(h[:, i], theta,
                              n_t=args.n_theta_bins, n_r=args.n_r_bins)
                   for i in range(N)], dtype=np.float32)
    # Swim onsets = frames where |omega| jumps above a small threshold
    # AND the previous frame is below it. Robust w.r.t. boxcar widening of
    # the swim impulse: we only mark the leading edge.
    omg_abs = np.abs(omega)
    above = omg_abs > (0.05 * omg_abs.max() if omg_abs.max() > 0 else 0.0)
    onset_mask = above & np.r_[True, ~above[:-1]]
    print(f"      detected {int(onset_mask.sum())} swim onsets in {h.shape[0]} frames")
    swim = _swim_modulation(h, onset_mask, dt, win_s=args.win_s)

    # 4) classify
    cls, thr = _classify(mi, wmag, swim,
                         mi_q=args.mi_q, w_q=args.w_q, swim_q=args.swim_q)
    print(f"[4/5] classify  thresholds: {thr}")
    for k in CLASS_ORDER:
        n = int((cls == k).sum())
        print(f"      class {k}: {n:4d} cells")

    # Per-cell-type breakdown
    df = pd.DataFrame({
        "model_ix": np.arange(N),
        "type": type_per_neuron,
        "category": [_category_of(t) for t in type_per_neuron],
        "mi_bits": mi,
        "w_out_mag": wmag,
        "swim_mod": swim,
        "klass": cls,
    })
    print("\n[per-cell-type breakdown]")
    by_type = (df.groupby(["category", "type", "klass"]).size()
                  .unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0))
    by_type["total"] = by_type.sum(axis=1)
    print(by_type.sort_values("total", ascending=False).to_string())

    df.to_csv(args.csv_out, index=False)
    print(f"\nwrote {args.csv_out}")

    # 5) render 4 panels
    print(f"[5/5] loading skeletons in model order (downsample={args.downsample})")
    model_bodyids, model_categories = _model_index_to_bodyid(args.connectome_dir)
    if len(model_bodyids) != N:
        print(f"warning: connectome N={len(model_bodyids)} != model N={N}")
    neurons, _cats, has_skel = _load_skeletons_in_model_order(
        args.anatomy_dir, model_bodyids, model_categories,
        downsample=args.downsample,
    )
    print(f"      {int(has_skel.sum())}/{len(neurons)} model neurons have a skeleton")

    seg_per_neuron = _extract_per_neuron_segments(neurons)
    somas = _per_neuron_soma(neurons)

    # Truncate cls/type to skeleton count if the model had extra rows
    # without a skeleton (shouldn't happen but be defensive).
    if len(seg_per_neuron) != N:
        n_min = min(len(seg_per_neuron), N)
        seg_per_neuron = seg_per_neuron[:n_min]
        somas = somas[:n_min]
        cls = cls[:n_min]
        type_per_neuron = type_per_neuron[:n_min]

    _render_panels(seg_per_neuron, somas, cls, mi, wmag, swim,
                   type_per_neuron, thr, args.out,
                   elev=args.elev, azim=args.azim, background=args.bg)


if __name__ == "__main__":
    main()
