"""The per-neuron fit of the update f_theta at work, on one neuron: the model's f_theta output
against the fitted affine form, over consecutive training frames.

    PYTHONPATH=src python tools/ftheta_fit_trace.py [run] [--neuron 2895] [--frames 1000] [--device cuda:0]

The parameter fit (metrics._update_template_fit) fits, per neuron, f_theta(v_i, a_i, msg_i, I_i)
-- the model's dv_i/dt -- against a0 + a1 v_i + a2 msg_i + a3 I_i by least squares on 64 frames
drawn with the readout's own seed (np.random.default_rng(0)); tau = -1/a1, V_rest = -a0/a1 and
the gauge use a1, a0, a2. This redoes that 4-column fit for one neuron on the same 64 frames,
then evaluates both sides on `--frames` consecutive training frames from frame 0: green the
model's f_theta output (the fit's target), black a0 + a1 v + a2 msg + a3 I. Writes
presentation/Fig/meeting_ftheta_fit_<fold>_n<neuron>.{pdf,png}.
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
from connectome_gnn.metrics import INDEX_TO_NAME                   # noqa: E402
from connectome_gnn.utils import set_data_root                     # noqa: E402
from paper_style import CM, COLOR_GT, COLOR_PRED  # noqa: E402
FS_LABEL, FS_TICK, FS_ANNOT = 13, 11, 11        # slide-size fonts

RUN = "flyvis_noise_005_blank50_condl25_cv00"


def columns(core, edges, x_ts, frames, n, i, dev):
    emb = int(core.a.shape[1])
    did = torch.zeros((n, 1), dtype=torch.int, device=dev)
    X, y = [], []
    with torch.no_grad():
        for k in frames:
            pred, feats, _ = core(x_ts.frame(int(k)).to(dev), edges, data_id=did, return_all=True)
            f = feats[i].double().cpu().numpy()
            X.append([1.0, f[0], f[1 + emb], f[2 + emb]])          # 1, v_i, msg_i, I_i
            y.append(float(pred.reshape(-1)[i]))
    return np.array(X), np.array(y)


def main(run, neuron, n_win, device):
    set_data_root(ROOT)
    dev = torch.device(device)
    log_dir, config, x_ts, edges, model = load(run, dev)
    core = getattr(model, "_orig_mod", model)
    n, T = int(x_ts.n_neurons), int(x_ts.n_frames)
    fit_frames = np.random.default_rng(0).choice(T, size=min(64, T), replace=False)
    Xf, yf = columns(core, edges, x_ts, fit_frames, n, neuron, dev)
    a = np.linalg.lstsq(Xf, yf, rcond=None)[0]
    Xw, yw = columns(core, edges, x_ts, np.arange(n_win), n, neuron, dev)
    fit = Xw @ a
    r2 = 1 - np.sum((yw - fit) ** 2) / np.sum((yw - yw.mean()) ** 2)
    dt = float(config.simulation.delta_t)
    t = np.arange(n_win) * dt
    typ = INDEX_TO_NAME.get(int(np.asarray(x_ts.neuron_type).ravel()[neuron]), "")

    # true values from the ground truth; the reported (offset-corrected) V_rest from the plot pass
    from connectome_gnn.generators.ode_params import load_ode_params_for_run
    op = load_ode_params_for_run(config, device="cpu")
    tau_true = float(np.asarray(op.gt_tau(n))[neuron]); V_true = float(np.asarray(op.gt_vrest(n))[neuron])
    rp = np.load(os.path.join(log_dir, "results", "extras", "recovered_pairs.npz"))
    V_rep = float(rp["V_rest_learned"][neuron]) if rp["V_rest_learned"].size == n else float("nan")
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(26 * CM, 7.5 * CM), gridspec_kw={"width_ratios": [4.2, 1]})
    ax.plot(t, yw, color=COLOR_GT, lw=2.0, label=r"$f_\theta$ (model)")
    ax.plot(t, fit, color=COLOR_PRED, lw=0.7, ls="--", label=r"fit $a_0 + a_1 v_i + a_2\,\mathrm{msg}_i + a_3 I_i$")
    ax.set_xlabel("time (s)", fontsize=FS_LABEL); ax.set_ylabel(r"$dv_i/dt$", fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=FS_ANNOT, frameon=False, loc="lower right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    axt.axis("off")
    rows = [("", "true", "fitted"),
            (r"$\tau$ (s)", f"{tau_true:.4f}", f"{-1 / a[1]:.4f}"),
            (r"$V^{rest}$, $-a_0/a_1$", f"{V_true:.3f}", f"{-a[0] / a[1]:.3f}"),
            (r"$V^{rest}$ + offset", f"{V_true:.3f}", f"{V_rep:.3f}"),
            (r"$R^2$ of the fit", "", f"{r2:.3f}")]
    for r_, (lab, tv, fv) in enumerate(rows):
        y = 0.85 - 0.16 * r_
        w = "bold" if r_ == 0 else "normal"
        axt.text(0.0, y, lab, fontsize=FS_ANNOT, transform=axt.transAxes, fontweight=w)
        axt.text(1.05, y, tv, fontsize=FS_ANNOT, transform=axt.transAxes, ha="right", color=COLOR_GT, fontweight=w)
        axt.text(1.45, y, fv, fontsize=FS_ANNOT, transform=axt.transAxes, ha="right", fontweight=w)
    fig.tight_layout()
    base = os.path.join(REPO, "presentation", "Fig", f"meeting_ftheta_fit_{run.split('_')[-1]}_n{neuron}")
    fig.savefig(base + ".pdf", dpi=300); fig.savefig(base + ".png", dpi=300)
    print("wrote", base + ".{pdf,png}", f" R2 {r2:.4f}  a {np.round(a, 4)}  tau {-1 / a[1]:.4f}  V {-a[0] / a[1]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=RUN)
    ap.add_argument("--neuron", type=int, default=2895)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    main(a.run, a.neuron, a.frames, a.device)
