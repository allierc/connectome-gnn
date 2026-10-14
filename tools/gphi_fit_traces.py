"""The per-edge fit of the message at work, on the strongest synapses onto one neuron: the model's
message W_model * g_phi on each edge against the fitted W * relu(v_j) + C, over consecutive
training frames.

    PYTHONPATH=src python tools/gphi_fit_traces.py [run] [--neuron 2895] [--n-edges 4] [--frames 300] [--device cuda:0]

The parameter fit (extract_template_params, current data) fits each edge's message
W_model_ij * g_phi(v_j, a_j, v_i, a_i) against W_ij * relu(v_j) + C_ij by least squares, on the
1,024 frames `_observed_plan` draws (seed 0), keeping only the frames where relu(v_j) is above
the filter threshold (`tmpl_vj_floor` in the run's metrics.txt). This redoes that fit for the
`--n-edges` incoming edges of `--neuron` with the largest true |W|, on the same frames and
threshold, and draws both sides on `--frames` consecutive training frames from frame 0: green
the model's message, black dashed the fit (before the per-neuron gauge k_i). Writes
presentation/Fig/meeting_gphi_fit_<fold>_n<neuron>.{pdf,png}.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "experiments", "paper", "scripts"))
from clustering_template_vs_chain import ROOT, load                # noqa: E402
from connectome_gnn.metrics import (INDEX_TO_NAME, _frame_voltages, _g_phi_on_edges,  # noqa: E402
                                    _observed_plan, get_model_W)
from connectome_gnn.utils import set_data_root, to_numpy           # noqa: E402
from paper_style import CM, COLOR_GT, COLOR_PRED                   # noqa: E402

RUN = "flyvis_noise_005_blank50_condl25_cv00"
FS = 9


def main(run, neuron, n_edges, n_win, device):
    set_data_root(ROOT)
    dev = torch.device(device)
    log_dir, config, x_ts, edges, model = load(run, dev)
    core = getattr(model, "_orig_mod", model)
    from connectome_gnn.generators.ode_params import load_ode_params_for_run
    op = load_ode_params_for_run(config, device="cpu")
    floor = next(float(line.split(":")[1]) for line in open(f"{log_dir}/results/metrics.txt")
                 if line.startswith("tmpl_vj_floor:"))
    e = to_numpy(edges).reshape(2, -1)
    E, T = e.shape[1], int(x_ts.n_frames)
    W_true = np.asarray(to_numpy(op.W)).ravel()
    W_model = to_numpy(get_model_W(core)).ravel().astype(np.float64)
    inc_all = np.flatnonzero(e[1] == neuron)
    _, fit_frames = _observed_plan(E, E, T, 1024, 0)
    types = np.asarray(x_ts.neuron_type).astype(int).ravel()
    name = lambda k: INDEX_TO_NAME.get(int(types[k]), "")

    def msg(frames):
        vf = _frame_voltages(x_ts, frames, dev)
        _, vj, g = _g_phi_on_edges(core, config, torch.as_tensor(e[0, inc], device=dev),
                                   torch.as_tensor(e[1, inc], device=dev), vf)
        vj = to_numpy(vj).astype(np.float64)
        return np.maximum(vj, 0.0), W_model[inc][:, None] * to_numpy(g).astype(np.float64)

    # fitted edges (>= 8 frames above the filter) and excluded ones, each strongest first
    inc = inc_all
    u_all, _ = msg(fit_frames)
    n_kept = (u_all > max(floor, 1e-6)).sum(1)
    order = np.argsort(-np.abs(W_true[inc_all]))
    fitted = [k for k in order if n_kept[k] >= 8][:n_edges]
    excluded = [k for k in order if n_kept[k] < 8]
    sel = fitted + excluded
    inc = inc_all[sel]
    u_f, y_f = msg(fit_frames)
    # the gauge k_i = tau_i * df/dmsg_i = a2 / -a1 from the per-neuron update fit, as the readout does it
    from ftheta_fit_trace import columns
    Xf, yf = columns(core, edges, x_ts, np.random.default_rng(0).choice(T, size=min(64, T), replace=False),
                     int(x_ts.n_neurons), neuron, dev)
    a = np.linalg.lstsq(Xf, yf, rcond=None)[0]
    k_i = a[2] / -a[1]
    u_w, y_w = msg(np.arange(n_win))
    dt = float(config.simulation.delta_t)
    t = np.arange(n_win) * dt

    fig, grid = plt.subplots(len(inc), 2, figsize=(22 * CM, 2.6 * CM * len(inc)),
                             gridspec_kw={"width_ratios": [4.2, 1]}, sharex="col")
    axes, taxes = grid[:, 0], grid[:, 1]
    for row, (ax, k) in enumerate(zip(np.atleast_1d(axes), range(len(inc)))):
        is_excl = k >= len(fitted)
        keep = u_f[k] > (1e-6 if is_excl else max(floor, 1e-6))
        A = np.stack([u_f[k][keep], np.ones(keep.sum())], 1)
        (w, c), *_ = np.linalg.lstsq(A, y_f[k][keep], rcond=None)
        fit = w * u_w[k] + c
        r2 = 1 - np.sum((y_w[k] - fit) ** 2) / max(np.sum((y_w[k] - y_w[k].mean()) ** 2), 1e-30)
        ax.plot(t, y_w[k], color=COLOR_GT, lw=2.0, label=r"$W^{model}_{ij}\,g_\phi$")
        ax.plot(t, fit, color=COLOR_PRED, lw=0.8, ls="--", label=r"fit $W_{ij}\,\mathrm{relu}(v_j) + C_{ij}$")
        j = int(e[0, inc[k]])
        how = (f"excluded by the filter; fit without it on {int(keep.sum())} frames" if is_excl
               else f"{int(keep.sum())} of 1,024 frames kept")
        ax.text(0.0, 1.02, f"from {j} ({name(j)})   {how}",
                transform=ax.transAxes, va="bottom", fontsize=FS - 2, color="red" if is_excl else "black")
        ta = taxes[row]; ta.axis("off")
        tab = [("", "true", "fitted"), ("W", f"{W_true[inc[k]]:+.3f}", f"{k_i * w:+.3f}"),
               ("C", "0", f"{k_i * c:+.4f}"), ("R²", "", f"{r2:.3f}")]
        for r_, (lab, tv, fv) in enumerate(tab):
            y = 0.9 - 0.27 * r_
            wt = "bold" if r_ == 0 else "normal"
            ta.text(0.0, y, lab, fontsize=FS - 2, transform=ta.transAxes, fontweight=wt)
            ta.text(0.75, y, tv, fontsize=FS - 2, transform=ta.transAxes, ha="right", color=COLOR_GT, fontweight=wt)
            ta.text(1.25, y, fv, fontsize=FS - 2, transform=ta.transAxes, ha="right", fontweight=wt)
        ax.set_ylabel("message", fontsize=FS)
        ax.tick_params(labelsize=FS - 1)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if row == 0:
            ax.legend(fontsize=FS - 1, frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.12))
    np.atleast_1d(axes)[-1].set_xlabel("time (s)", fontsize=FS)
    fig.tight_layout()
    base = os.path.join(REPO, "presentation", "Fig", f"meeting_gphi_fit_{run.split('_')[-1]}_n{neuron}")
    fig.savefig(base + ".pdf", dpi=300); fig.savefig(base + ".png", dpi=300)
    print("wrote", base + ".{pdf,png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=RUN)
    ap.add_argument("--neuron", type=int, default=2895)
    ap.add_argument("--n-edges", type=int, default=4)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    main(a.run, a.neuron, a.n_edges, a.frames, a.device)
