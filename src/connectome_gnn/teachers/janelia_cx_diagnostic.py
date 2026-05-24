"""Diagnostic visualisations for a trained Hulse Model A checkpoint.

Loads a `.pt` checkpoint produced by janelia_cx_teacher.train_janelia_cx_teacher
and renders the four canonical CX-imaging panels (compass / EB ring /
kinograph / 3-D anatomy) on a fresh path-integration batch.

Usage:
    python -m connectome_gnn.teachers.janelia_cx_diagnostic \
        --checkpoint papers/janelia_cx/trained/janelia_cx_seed0.pt \
        --output-dir papers/janelia_cx/diagnostics_seed0 \
        --n-steps 400
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from connectome_gnn.generators.connconstr_data import load_drosophila_cx_connectome
from connectome_gnn.plot_cx import (
    cx_epg_directions,
    plot_cx_anatomy_3d,
    plot_cx_compass,
    plot_cx_eb_ring,
    plot_cx_kinograph_pva,
)
from connectome_gnn.teachers.janelia_cx_teacher import (
    JaneliaCxRNN,
    generate_path_integration_batch,
)


def _load_checkpoint(path: str, device: str = "cpu") -> tuple[JaneliaCxRNN, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    if "W_con" not in state:
        raise ValueError(
            f"checkpoint {path} has no 'W_con' — cannot reconstruct the "
            "Hulse Eq. 9 sign mask. Please re-train and re-save."
        )
    W_con = state["W_con"].to(device)
    net = JaneliaCxRNN(
        n_units=int(state["n_units"]),
        n_input=int(state["n_input"]),
        n_output=int(state["n_output"]),
        tau=float(state["tau"]),
        dt=float(state["dt"]),
        W_con=W_con,
    ).to(device)
    if "S" in state:
        # New format (Hulse Eq. 9 parameterisation).
        net.S.data.copy_(state["S"].to(device))
    else:
        # Legacy format: free W_rec parameter. Recover S = |W_rec|; the
        # sign of W_rec from the legacy run is *not* guaranteed to match
        # sign(W_con), so we warn loudly.
        W_rec_old = state["W_rec"].to(device)
        sign_match = (torch.sign(W_rec_old) * net.W_con_sign >= 0).float().mean().item()
        if sign_match < 0.999:
            print(f"[janelia_cx_diagnostic] WARNING: legacy checkpoint has "
                  f"{(1 - sign_match) * 100:.1f}% of recurrent entries with "
                  f"signs that disagree with W_con. Reloading as |W_rec|; "
                  f"those entries will silently flip to the connectome sign.")
        net.S.data.copy_(W_rec_old.abs())
    net.W_in.data.copy_(state["W_in"].to(device))
    net.b.data.copy_(state["b"].to(device))
    net.W_out.data.copy_(state["W_out"].to(device))
    net.b_out.data.copy_(state["b_out"].to(device))
    return net, state


def render_diagnostics(
    *,
    checkpoint: str,
    output_dir: str,
    datapath: str = "papers/Code_NN/Code_NN/Data/Figure5/exported-traced-adjacencies-v1.2",
    n_steps: int = 400,
    seed: int = 0,
    device: str = "cpu",
    activation: str = "sigmoid",
) -> None:
    """Render compass / EB-ring / kinograph / 3-D anatomy panels."""
    os.makedirs(output_dir, exist_ok=True)
    net, _ = _load_checkpoint(checkpoint, device=device)
    net.eval()

    cx = load_drosophila_cx_connectome(datapath)
    epg_indices = np.arange(int(cx["n_epg"]), dtype=int)
    epg_theta = cx_epg_directions(cx["epg_ix"], n_glom=16)
    type_names = list(cx["type_names"])

    rng = np.random.default_rng(seed)
    batch = generate_path_integration_batch(
        batch_size=1, n_steps=n_steps, device=device, rng=rng,
    )
    with torch.no_grad():
        y_hat, h_buf = net(batch.stimulus)

    voltage_history = h_buf[0].cpu().numpy()       # (T, N)
    true_theta = batch.theta_hd[0].cpu().numpy()   # (T,)

    plot_cx_compass(
        voltage_history,
        epg_indices=epg_indices,
        epg_theta=epg_theta,
        output_path=os.path.join(output_dir, "compass.png"),
        n_panels=9,
        activation=activation,
        title=f"EPG compass — {os.path.basename(checkpoint)}",
    )

    plot_cx_eb_ring(
        voltage_history,
        epg_indices=epg_indices,
        epg_theta=epg_theta,
        output_path=os.path.join(output_dir, "eb_ring.png"),
        n_panels=9,
        activation=activation,
    )

    plot_cx_kinograph_pva(
        voltage_history,
        epg_indices=epg_indices,
        epg_theta=epg_theta,
        output_path=os.path.join(output_dir, "kinograph_pva.png"),
        activation=activation,
        dt_s=float(net.dt),
        true_theta_hd=true_theta,
    )

    # Edge overlay: draw the strongest |W| connections coloured by sign.
    W_con = net.W_con.detach().cpu().numpy()
    src, dst = np.nonzero(W_con)
    edge_index = np.stack([src, dst], axis=0)
    edge_weights = W_con[src, dst]

    plot_cx_anatomy_3d(
        output_path=os.path.join(output_dir, "anatomy_3d.png"),
        neuron_types=np.asarray(cx["neuron_types"]).astype(int),
        type_names=type_names,
        epg_ix=cx["epg_ix"],
        anatomy_dir="papers/janelia_cx/anatomy",
        edge_index=edge_index,
        edge_weights=edge_weights,
        n_edge_draw=300,
    )

    # Decoded-HD diagnostic plot: predicted (cos, sin) vs ground truth.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
    t_axis = np.arange(n_steps) * float(net.dt)
    pred = y_hat[0].cpu().numpy()
    true_y = batch.target[0].cpu().numpy()
    axes[0].plot(t_axis, true_y[:, 0], color="black", label="true cos")
    axes[0].plot(t_axis, pred[:, 0], color="red", label="pred cos", alpha=0.8)
    axes[0].set_ylabel("cos(HD)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(t_axis, true_y[:, 1], color="black", label="true sin")
    axes[1].plot(t_axis, pred[:, 1], color="red", label="pred sin", alpha=0.8)
    axes[1].set_ylabel("sin(HD)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "readout.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[janelia_cx_diagnostic] wrote diagnostics to {output_dir}")


def _main():
    p = argparse.ArgumentParser(description="Hulse CX trained-checkpoint diagnostics")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--datapath",
                   default="papers/Code_NN/Code_NN/Data/Figure5/exported-traced-adjacencies-v1.2")
    p.add_argument("--n-steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--activation", default="sigmoid",
                   choices=["sigmoid", "relu", "none"])
    args = p.parse_args()
    render_diagnostics(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        datapath=args.datapath,
        n_steps=args.n_steps,
        seed=args.seed,
        device=args.device,
        activation=args.activation,
    )


if __name__ == "__main__":
    _main()
