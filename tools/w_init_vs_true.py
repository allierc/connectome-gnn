"""The weight a GNN starts from, against the weight it has to reach.

WHY THIS FIGURE. `w_init_mode: randn_scaled` draws W ~ N(0,1) * scale/sqrt(n_edges),
so `w_init_scale: 88` is not an amplitude of 88: over 434,112 edges it is a standard
deviation of 0.133. Whether that is large or small is only answerable against the
generator's own weights, and on the conductance twin those span eight orders of
magnitude with half the edges below 1e-6 -- so an initialisation that is a single
lump in the top decade starts most edges a hundred thousand times above their
target, and gradient descent has to walk them down six decades.

Writes figures/w_init_vs_true.png and prints the quantiles behind it.
"""
import logging, math, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from connectome_gnn.models.utils import load_run_config
from connectome_gnn.models.training_utils import init_training_data
from connectome_gnn.utils import log_path, to_numpy

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
RUNS = ["flyvis_conductance_noise_005_conductance_gnnsil_cv00",
        "flyvis_current_noise_005_conductance_cv00"]
out = {}
for name in RUNS:
    cfg, _ = load_run_config(name, False, "plot")
    t = cfg.training
    data = init_training_data(cfg, DEV, log_path(cfg.config_file),
                              logging.getLogger("x"))
    edges = data.edges
    n_e = int(edges.shape[1]) + int(cfg.simulation.n_extra_null_edges or 0)
    scale = float(getattr(t, "w_init_scale", 1.0))
    torch.manual_seed(int(getattr(t, "seed", 42)))
    w0 = (torch.randn(n_e) * (scale / math.sqrt(n_e))).numpy()
    w_true = np.asarray(data.ode_params.effective_true_weights(
        to_numpy(data.ode_params.W), to_numpy(edges), int(data.n_neurons)),
        dtype=float).ravel()
    sq = bool(getattr(cfg.graph_model, "w_squared", False))
    out[name] = dict(w0=w0, w_true=w_true, scale=scale, sq=sq,
                     dataset=cfg.dataset, n_e=n_e)
    print(f"{name}\n   dataset {cfg.dataset}  edges {n_e}  scale {scale}  w_squared {sq}")
    print(f"   init  W : std {w0.std():.5f}  |W| median {np.median(abs(w0)):.5f}")
    if sq:
        print(f"   init  W^2 (what the message uses): median {np.median(w0**2):.3e}  "
              f"mean {np.mean(w0**2):.3e}")
    f = w_true[np.isfinite(w_true)]
    nz = np.abs(f[f != 0])
    q = np.percentile(nz, [10, 50, 90, 99])
    print(f"   true |W| quantiles 10/50/90/99: {q[0]:.2e} / {q[1]:.2e} / "
          f"{q[2]:.2e} / {q[3]:.2e}   max {nz.max():.3f}")
    e0 = np.abs(w0 ** 2 if sq else w0)
    qe = np.percentile(e0, [10, 50, 90])
    print(f"   init  {'W^2' if sq else 'W'} quantiles 10/50/90: {qe[0]:.2e} / "
          f"{qe[1]:.2e} / {qe[2]:.2e}")
    print(f"   -> the median edge starts {qe[1] / q[1]:.1f}x its target")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, name in zip(axes, RUNS):
    d = out[name]
    eff0 = d["w0"] ** 2 if d["sq"] else d["w0"]
    lab0 = ("initial $W^2$ (what the message uses)" if d["sq"] else "initial $W$")
    tr = d["w_true"][np.isfinite(d["w_true"])]
    tr = tr[tr != 0]
    lo = min(np.percentile(np.abs(eff0[eff0 != 0]), 0.1), np.percentile(np.abs(tr), 0.1))
    bins = np.logspace(np.log10(max(lo, 1e-9)), np.log10(max(np.abs(tr).max(),
                        np.abs(eff0).max())), 90)
    ax.hist(np.abs(eff0[eff0 != 0]), bins=bins, color="tab:red", alpha=0.6, label=lab0)
    ax.hist(np.abs(tr), bins=bins, color="tab:blue", alpha=0.6,
            label="generator's $W$ (target)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$|W|$", fontsize=13); ax.set_ylabel("edges", fontsize=13)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.text(0.0, 1.02, f"{'a' if name is RUNS[0] else 'b'}   {name.replace('flyvis_','')}"
            f"   scale {d['scale']:g}, w_squared {d['sq']}",
            transform=ax.transAxes, va="bottom", fontsize=11)
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
p = "figures/w_init_vs_true.png"
fig.savefig(p, dpi=150); print("\nwrote", p)
