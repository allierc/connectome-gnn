"""The two panels of `conductance_euler_stability.tex`.

    python docs/conductance_euler_stability_fig.py

Panel a is arithmetic and needs nothing. Panel b reads the last checkpoint of the
run that exploded, so it is skipped with a message when that directory is not
mounted -- the note then keeps the panel it can honestly draw.

    FLYVIS results  <FLYVIS_ROOT_DIR>/results/flow/2000/000/chkpts/chkpt_00005
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = "flow/2000/000"
CHKPT = "chkpt_00005"
DT = 0.02  # seconds, flyvis's integration step for the optic-flow task

# Red and blue for two traces of DIFFERENT ORIGIN -- one integrator against the
# other -- following the repository's plotting convention, which reserves green
# and black for ground truth against prediction.
C_EULER = "#cf222e"
C_EXACT = "#1f6feb"


def stability_factors():
    """(dt/tau_i)(1 + G_i) per neuron at unit presynaptic activity, or None.

    G_i = sum_j g_ij relu(v_j) is the total synaptic conductance onto neuron i in
    units of its own leak conductance; evaluated here at relu(v_j) = 1, which is
    inside the range the run actually visited (its recorded activity maximum was
    2.67), so the panel is a conservative reading of the checkpoint rather than a
    worst case.
    """
    root = os.environ.get(
        "FLYVIS_ROOT_DIR", "/groups/saalfeld/home/allierc/GraphData/flyvis"
    )
    path = os.path.join(root, "results", RUN, "chkpts", CHKPT)
    if not os.path.exists(path):
        print(f"no checkpoint at {path}; drawing panel a only")
        return None

    import torch
    import yaml
    from datamate import namespacify

    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
    import flyvis_conductance_optical_flow  # noqa: F401  -- registers the classes
    from flyvis.network import Network

    meta = yaml.safe_load(
        open(os.path.join(root, "results", RUN, "_meta.yaml"))
    )
    net = Network(**namespacify(meta["config"]).network)
    net.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=False)["network"]
    )
    p = net._param_api()
    g = (p.edges.syn_count * p.edges.syn_strength).detach()
    dev = g.device
    dst = torch.as_tensor(
        np.asarray(net.connectome.edges.target_index[:]), device=dev
    ).long()
    n = len(net.connectome.nodes.index)
    G = torch.zeros(n, device=dev).index_add_(0, dst, g)
    tau = p.nodes.time_const.detach()
    return ((DT / torch.clamp(tau, min=DT)) * (1.0 + G)).cpu().numpy()


def main():
    factors = stability_factors()
    ncols = 1 if factors is None else 2
    fig, axes = plt.subplots(
        1, ncols, figsize=(5.2 * ncols, 3.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    ax = axes[0]
    z = np.linspace(0, 6, 601)
    ax.plot(z, np.abs(1 - z), color=C_EULER, lw=1.8,
            label="forward Euler,  $|1-z|$")
    ax.plot(z, np.exp(-z), color=C_EXACT, lw=1.8,
            label="exponential Euler,  $e^{-z}$")
    ax.axhline(1.0, color="0.6", lw=0.9, ls="--")
    ax.axvline(2.0, color="0.6", lw=0.9, ls=":")
    ax.text(2.06, 2.3, "$z=2$", color="0.4", fontsize=9)
    ax.text(0.1, 1.06, "no decay", color="0.5", fontsize=9)
    ax.set_xlabel("$z = (1+G_i)\\,\\Delta t/\\tau_i$, relaxation times per step")
    ax.set_ylabel("factor multiplying the distance to equilibrium")
    ax.set_ylim(0, 3.2)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.text(0, 1.03, "a   one step multiplies the distance to equilibrium by",
            transform=ax.transAxes, fontsize=10)

    if factors is not None:
        ax = axes[1]
        ax.hist(factors, bins=80, color="0.45", edgecolor="none")
        ax.axvline(2.0, color=C_EULER, lw=1.4)
        n_bad = int((factors > 2).sum())
        ax.text(2.06, ax.get_ylim()[1] * 0.92,
                f"{n_bad} of {factors.size} neurons\nbeyond forward Euler's bound",
                color=C_EULER, fontsize=9, va="top")
        ax.set_xlabel("$(\\Delta t/\\tau_i)(1+G_i)$ at unit presynaptic activity")
        ax.set_ylabel("neurons")
        ax.text(0, 1.03,
                "b   the same factor, measured on flow/2000/000 at its last checkpoint",
                transform=ax.transAxes, fontsize=10)

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    out = os.path.join(HERE, "conductance_euler_stability.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
