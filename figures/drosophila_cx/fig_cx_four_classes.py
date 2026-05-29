"""Four functional classes of CX neurons under the EPG-only readout, in
the spirit of figures/zebrafish/fig_zebrafish_four_classes.py.

Per-neuron scores on a controlled rollout:
    mi[i]    = I(sigma(h_i); theta) in bits (plug-in histogram MI)
    wmag[i]  = ||W_out[:, i]||_2   (zero for i >= n_epg under
                                    output_from_epg_only=True)
    omod[i]  = constant-omega modulation amplitude
               = max(|corr(sigma(h_i), omega_+)|,
                     |corr(sigma(h_i), omega_-)|)
              where omega_+, omega_- are the signed-half angular
              velocities (analog of the L/R swim drive in the zebrafish
              analysis). For OU rollouts this captures PEN-like
              angular-velocity sensitivity.

Decision tree (sample-median quantile thresholds on the *non-zero*
W_out support and on MI / omod across all neurons):
    R  bump representation   mi high  &  wmag high   (read-out HD code)
    L  latent / redundant    mi high  &  wmag low    (HD inside the
                                                      ring but silent at
                                                      the readout layer)
    D  driver / updater      mi low   &  omod high   (omega-locked, no HD)
    Z  leak / uninvolved     mi low   &  omod low

Per condition (one cv0 model per cell), the script emits:
    figure/<basename>__cv0.csv          per-neuron table
    figure/tab_cx_four_classes.tex      condensed per-cell-type LaTeX
                                        table aggregated across the 4
                                        trainable parameterisations
    figure/fig_cx_four_classes.png      3 x 4 diagnostic figure:
                                        rows = (MI/wmag), (MI/omod),
                                        per-cell-type bar; cols = model

CLI:
    python figures/drosophila_cx/fig_cx_four_classes.py --device cpu
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connectome_gnn.utils import log_path, load_data_root_from_json, set_data_root
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.registry import create_model
from connectome_gnn.generators.utils import generate_path_integration_batch
from connectome_gnn.task_state import TaskTrials


MODELS = [
    ("drosophila_cx_pi_epg_no_tv_cv0",     "Known-ODE no-TV"),
    ("drosophila_cx_pi_epg_tv_cv0",        "Known-ODE +TV"),
    ("drosophila_cx_pi_gnn_epg_no_tv_cv0", "GNN no-TV"),
    ("drosophila_cx_pi_gnn_epg_tv_cv0",    "GNN +TV"),
]

CLASS_COLOR = {
    "R": "#e41a1c",
    "L": "#ff7f00",
    "D": "#377eb8",
    "Z": "#7f7f7f",
}
CLASS_ORDER = ["R", "L", "D", "Z"]
CLASS_LABEL = {
    "R": "R  bump representation",
    "L": "L  latent / redundant",
    "D": "D  driver / updater",
    "Z": "Z  leak / uninvolved",
}


def _load(config_name, device, prefer_epoch=None):
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


def _run_ou(net, n_steps, device, seed):
    rng = np.random.default_rng(seed)
    batch = generate_path_integration_batch(
        batch_size=1, n_steps=n_steps,
        dt=float(net.dt), device=device, rng=rng,
    )
    theta = batch.theta_hd[0].cpu().numpy()
    omega = batch.omega[0].cpu().numpy()
    with torch.no_grad():
        _, h = net(batch.stimulus)
    return h[0].cpu().numpy(), theta, omega


def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def _mi_neuron(h_i, theta, n_t=32, n_r=20):
    """Plug-in histogram MI between sigma(h_i) and theta (radians).
    Theta is wrapped to (-pi, pi] so the unwrapped OU heading covers
    the full circle uniformly across the n_t angular bins."""
    r = _sigmoid(h_i)
    if r.std() < 1e-6:
        return 0.0
    theta_wrap = ((theta + math.pi) % (2 * math.pi)) - math.pi
    t_edges = np.linspace(-math.pi, math.pi, n_t + 1)
    r_edges = np.quantile(r, np.linspace(0.0, 1.0, n_r + 1))
    r_edges[0] -= 1e-6; r_edges[-1] += 1e-6
    P, _, _ = np.histogram2d(theta_wrap, r, bins=[t_edges, r_edges])
    P = P / max(1.0, P.sum())
    Pt = P.sum(axis=1, keepdims=True)
    Pr = P.sum(axis=0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (Pt @ Pr + 1e-12)[nz])).sum())


def _wmag_from_net(net):
    """Per-neuron column-norm of W_out, zero-padded outside the readout
    slice. CX models use ``output_from_epg_only=True`` so only the first
    n_epg columns are non-zero."""
    N = int(len(net.neuron_types))
    wmag = np.zeros(N, dtype=np.float32)
    W = getattr(net, "W_out", None)
    if W is None:
        return wmag
    W = W.detach().cpu().numpy()
    n_cols = int(W.shape[1])
    wmag[:n_cols] = np.linalg.norm(W, axis=0)
    return wmag


def _omega_modulation(h_traj, omega):
    """Per-neuron correlation of sigma(h) with omega_+ / omega_- (signed
    half angular velocities), reported as the max of |corr+|, |corr-|.
    Output in [0, 1]; high = strong L/R angular-velocity tuning."""
    r = _sigmoid(h_traj)
    om_pos = np.clip(omega, 0.0, None)
    om_neg = np.clip(-omega, 0.0, None)
    r = r - r.mean(axis=0, keepdims=True)
    om_pos = om_pos - om_pos.mean()
    om_neg = om_neg - om_neg.mean()
    rs = r.std(axis=0) + 1e-12
    out = np.zeros(r.shape[1], dtype=np.float32)
    for s_om, label in [(om_pos, "+"), (om_neg, "-")]:
        sd = s_om.std() + 1e-12
        cor = (r * s_om[:, None]).sum(axis=0) / (rs * sd * r.shape[0])
        out = np.maximum(out, np.abs(cor))
    return out


def _classify(mi, wmag, omod, mi_q=0.5, w_q=0.5, omod_q=0.5):
    mi_thr = float(np.quantile(mi, mi_q))
    nz = wmag[wmag > 0]
    w_thr = float(np.quantile(nz, w_q)) if len(nz) else 0.0
    omod_thr = float(np.quantile(omod, omod_q))
    cls = np.empty(len(mi), dtype="<U1")
    for i in range(len(mi)):
        if mi[i] >= mi_thr:
            cls[i] = "R" if wmag[i] >= w_thr else "L"
        else:
            cls[i] = "D" if omod[i] >= omod_thr else "Z"
    return cls, {"mi": mi_thr, "wmag": w_thr, "omod": omod_thr}


def _score_one(net, n_steps, seed, device):
    h, theta, omega = _run_ou(net, n_steps, device, seed)
    N = h.shape[1]
    mi = np.array([_mi_neuron(h[:, i], theta) for i in range(N)],
                  dtype=np.float32)
    wmag = _wmag_from_net(net)
    omod = _omega_modulation(h, omega)
    return mi, wmag, omod


def _type_strings(net):
    tn = list(net.type_names)
    nt = np.asarray(net.neuron_types).astype(int)
    return np.array([tn[t] for t in nt])


def _short_type(t):
    s = str(t)
    return {"PEN_a(PEN1)": "PEN$_a$", "PEN_b(PEN2)": "PEN$_b$",
            "Delta7": r"$\Delta 7$", "EPGt": "EPGt", "EPG": "EPG",
            "PEG": "PEG", "ER6": "ER6"}.get(s, s)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_steps", type=int, default=10000)
    p.add_argument("--seed",    type=int, default=0)
    p.add_argument("--device",  default="cpu")
    p.add_argument("--out_dir", default=here)
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

    per_model_rows = []
    per_model_dfs = []

    fig, axes = plt.subplots(3, len(MODELS), figsize=(4.0 * len(MODELS), 11.0),
                             gridspec_kw=dict(hspace=0.42, wspace=0.32))

    for k, (cfg, label) in enumerate(MODELS):
        net = _load(cfg, device)
        mi, wmag, omod = _score_one(net, args.n_steps, args.seed, device)
        cls, thr = _classify(mi, wmag, omod)
        tps = _type_strings(net)
        N = len(mi)
        df = pd.DataFrame({
            "model_ix": np.arange(N),
            "type":     tps,
            "mi_bits":  mi,
            "w_out":    wmag,
            "omega_mod": omod,
            "klass":    cls,
        })
        df["model"] = cfg
        df["label"] = label
        per_model_dfs.append(df)
        df.to_csv(os.path.join(args.out_dir,
                                f"fig_cx_four_classes__{cfg}.csv"),
                  index=False)

        # row 0: MI vs |W_out|
        ax = axes[0, k]
        for klass in CLASS_ORDER:
            m = cls == klass
            ax.scatter(wmag[m], mi[m], s=14,
                       c=CLASS_COLOR[klass], alpha=0.7,
                       label=klass, edgecolors="none")
        ax.axhline(thr["mi"], ls="--", lw=0.5, c="black", alpha=0.4)
        ax.axvline(thr["wmag"], ls="--", lw=0.5, c="black", alpha=0.4)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(r"$\|W^{\mathrm{out}}_{:,i}\|_2$", fontsize=9)
        if k == 0:
            ax.set_ylabel("MI (bits)", fontsize=9)
            ax.legend(fontsize=8, loc="upper right", frameon=False)

        # row 1: MI vs omega-mod
        ax = axes[1, k]
        for klass in CLASS_ORDER:
            m = cls == klass
            ax.scatter(omod[m], mi[m], s=14,
                       c=CLASS_COLOR[klass], alpha=0.7,
                       edgecolors="none")
        ax.axhline(thr["mi"], ls="--", lw=0.5, c="black", alpha=0.4)
        ax.axvline(thr["omod"], ls="--", lw=0.5, c="black", alpha=0.4)
        ax.set_xlabel(r"$|\mathrm{corr}(\sigma(h_i),\omega_\pm)|_{\max}$",
                      fontsize=9)
        if k == 0:
            ax.set_ylabel("MI (bits)", fontsize=9)

        # row 2: per-cell-type stacked bars
        ax = axes[2, k]
        ord_types = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)",
                     "PEG", "Delta7", "ER6"]
        ord_types = [t for t in ord_types if t in set(tps)]
        x = np.arange(len(ord_types))
        bottom = np.zeros(len(ord_types))
        for klass in CLASS_ORDER:
            heights = np.array([int(((tps == t) & (cls == klass)).sum())
                                for t in ord_types])
            ax.bar(x, heights, bottom=bottom,
                   color=CLASS_COLOR[klass], label=klass, edgecolor="white",
                   linewidth=0.3)
            bottom += heights
        ax.set_xticks(x)
        ax.set_xticklabels([_short_type(t) for t in ord_types],
                            rotation=30, ha="right", fontsize=8)
        if k == 0:
            ax.set_ylabel("# neurons", fontsize=9)

        # Stash the per-type counts for the LaTeX table.
        type_counts = {}
        for t in ord_types:
            counts = {"R": 0, "L": 0, "D": 0, "Z": 0}
            mt = tps == t
            for klass in CLASS_ORDER:
                counts[klass] = int(((cls == klass) & mt).sum())
            counts["total"] = int(mt.sum())
            type_counts[t] = counts
        per_model_rows.append((cfg, label, type_counts, thr,
                               int((cls == "R").sum()),
                               int((cls == "L").sum()),
                               int((cls == "D").sum()),
                               int((cls == "Z").sum())))

    out_png = os.path.join(args.out_dir, "fig_cx_four_classes.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    # Combined LaTeX table:
    #   one row per cell type, four blocks (R/L/D/Z) per model column.
    # Compact: show only Known-ODE +TV and GNN +TV side-by-side.
    PRESENT_MODELS = [
        ("drosophila_cx_pi_epg_no_tv_cv0",     "Known-ODE no-TV"),
        ("drosophila_cx_pi_gnn_epg_no_tv_cv0", "GNN no-TV"),
    ]
    type_order_present = []
    for cfg, label in PRESENT_MODELS:
        for cfg2, _, tc, _, _, _, _, _ in per_model_rows:
            if cfg2 == cfg:
                for t in tc:
                    if t not in type_order_present:
                        type_order_present.append(t)
                break

    lines = []
    lines.append(r"\begin{tabular}{l " +
                 " ".join(["rrrrr"] * len(PRESENT_MODELS)) + "}")
    lines.append(r"\toprule")
    hdr = "type "
    for _, label in PRESENT_MODELS:
        hdr += rf"& \multicolumn{{5}}{{c}}{{{label}}} "
    lines.append(hdr + r"\\")
    lines.append(r"\midrule")
    sub = " "
    for _, _ in PRESENT_MODELS:
        sub += "& R & L & D & Z & total "
    lines.append(sub + r"\\")
    lines.append(r"\midrule")
    for t in type_order_present:
        row = _short_type(t)
        for cfg, _ in PRESENT_MODELS:
            tc = next(tc for c2, _, tc, _, _, _, _, _ in per_model_rows
                       if c2 == cfg)
            c = tc.get(t, {"R": 0, "L": 0, "D": 0, "Z": 0, "total": 0})
            row += f" & {c['R']} & {c['L']} & {c['D']} & {c['Z']} & {c['total']}"
        lines.append(row + r" \\")
    lines.append(r"\midrule")
    foot = "total"
    for cfg, _ in PRESENT_MODELS:
        _, _, tc, _, R, L, D, Z = next(r for r in per_model_rows
                                         if r[0] == cfg)
        foot += f" & {R} & {L} & {D} & {Z} & {R+L+D+Z}"
    lines.append(foot + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    tex_out = os.path.join(args.out_dir, "tab_cx_four_classes.tex")
    with open(tex_out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {tex_out}")

    # Print thresholds + class totals for the four conditions
    print("\n=== per-model summary ===")
    for cfg, label, _, thr, R, L, D, Z in per_model_rows:
        print(f"  {label:18s}  mi*={thr['mi']:.3f}  w*={thr['wmag']:.3f} "
              f" omod*={thr['omod']:.3f}   "
              f"R/L/D/Z = {R}/{L}/{D}/{Z}")


if __name__ == "__main__":
    main()
