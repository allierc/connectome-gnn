"""Voltage animation on the 3-D CX anatomy.

Runs the Known-ODE RNN under a constant-omega rollout, computes per-neuron
firing rate r_i(t) = sigmoid(h_i(t)), and renders one PNG every K frames
showing every CX skeleton in dark grey overlaid with a green tint whose
alpha is the current rate. Output: figures/drosophila/3D/frame_NNNN.png.

The geometry is the same hemibrain SWC pull used by fig_cx_anatomy_3d.py
(papers/janelia_cx/anatomy/cx_anatomy_test/). The model -> bodyId mapping
replays load_drosophila_cx_connectome's selection so model index i lines
up with the correct skeleton.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import navis

from fig_cx_anatomy_3d import (
    TYPE_COLOR, TYPE_ORDER, _load_rois, _project_2d,
)
from fig_kinographs_const_omega import _load, _run_const

from connectome_gnn.utils import load_data_root_from_json, set_data_root


# Permutation reordering EPG indices in the connectome loader (glomerular
# ring order). Verbatim from connconstr_data.py.
EPG_PERM = np.array([
    23, 24, 0, 1, 42, 43, 44, 45, 2, 3, 39, 40, 41, 4, 5, 6,
    36, 37, 38, 7, 8, 9, 33, 34, 35, 10, 11, 12,
    30, 31, 32, 13, 14, 15, 27, 28, 29, 16, 17, 18,
    25, 26, 19, 20, 21, 22,
])


def _model_index_to_bodyid(datapath: str) -> np.ndarray:
    """Replay load_drosophila_cx_connectome's neuron selection so that
    model index i corresponds to a specific hemibrain bodyId."""
    neuronsall = pd.read_csv(os.path.join(datapath, "traced-neurons.csv"))
    neuronsall.sort_values(by=["instance"], ignore_index=True, inplace=True)
    types = np.array(neuronsall.type).astype(str)

    def sub(t):
        return np.nonzero([t in x for x in types])[0]

    epg, pen = sub("EPG"), sub("PEN")
    peg, delta7 = sub("PEG"), sub("Delta7")
    allcx = np.concatenate((epg, pen, delta7, peg))
    allcx[0:46] = allcx[EPG_PERM]
    er6 = np.array([i for i, t in enumerate(types) if t == "ER6"], dtype=int)
    if er6.size:
        allcx = np.concatenate((allcx, er6))
    return neuronsall.bodyId.values[allcx]


def _load_skeletons_in_model_order(anatomy_dir: str, body_ids: np.ndarray,
                                    downsample: int = 10):
    """Return a list of 156 navis TreeNeurons indexed by model order, and
    a parallel list of cell-type strings for colour lookup."""
    swcs = {}
    for fname in os.listdir(os.path.join(anatomy_dir, "skeletons")):
        if not fname.endswith(".swc"):
            continue
        stem = fname[:-4]
        safe_t, _, bid_str = stem.rpartition("__")
        swcs[int(bid_str)] = (
            os.path.join(anatomy_dir, "skeletons", fname),
            safe_t,
        )

    neurons = []
    types = []
    for bid in body_ids:
        if int(bid) not in swcs:
            raise SystemExit(f"missing skeleton for bodyId {bid}")
        path, safe_t = swcs[int(bid)]
        n = navis.read_swc(path)
        if downsample and downsample > 1:
            n = navis.downsample_neuron(n, downsampling_factor=downsample,
                                         preserve_nodes=None)
        neurons.append(n)
        for canon in TYPE_ORDER:
            safe = canon.replace("(", "_").replace(")", "")
            if safe_t == safe:
                types.append(canon); break
        else:
            types.append(safe_t)
    return neurons, types


def _extract_per_neuron_segments(neurons):
    """Return:
      seg_arrays: list of (E_i, 2, 3) arrays per neuron
      seg_owner:  flat (E_total,) int array, neuron index per segment
      all_segs:   stacked (E_total, 2, 3) array
    """
    seg_arrays = []
    for n in neurons:
        nodes = n.nodes
        child = nodes[nodes.parent_id != -1]
        if len(child) == 0:
            seg_arrays.append(np.zeros((0, 2, 3), dtype=np.float32)); continue
        parent_xyz = nodes.set_index("node_id").loc[
            child.parent_id.values, ["x", "y", "z"]
        ].values
        child_xyz = child[["x", "y", "z"]].values
        seg_arrays.append(np.stack([parent_xyz, child_xyz], axis=1)
                          .astype(np.float32))
    counts = np.array([len(s) for s in seg_arrays])
    seg_owner = np.repeat(np.arange(len(neurons)), counts)
    all_segs = (np.concatenate(seg_arrays, axis=0) if seg_arrays
                else np.zeros((0, 2, 3), dtype=np.float32))
    return seg_arrays, seg_owner, all_segs


def _render_frame(out_path, segs2d, seg_owner, rates_t, mesh_segs2d,
                  bg="black", lw_base=0.18, lw_top=0.45,
                  base_color=(0.25, 0.25, 0.25), green=(0.0, 1.0, 0.3),
                  xlim=None, ylim=None, frame_idx=None, total_frames=None,
                  hd_deg=None, ax_ref=None, fig_ref=None):
    """Render a single animation frame -- two LineCollections (dark base
    + green overlay with per-segment alpha) on top of the neuropil
    silhouette."""
    if fig_ref is None:
        fig, ax = plt.subplots(figsize=(7.5, 8.5), facecolor=bg)
    else:
        fig, ax = fig_ref, ax_ref
        ax.clear()
    ax.set_facecolor(bg)

    # Neuropil silhouette
    if mesh_segs2d is not None and len(mesh_segs2d):
        ax.add_collection(LineCollection(
            mesh_segs2d, colors=("0.85" if bg == "black" else "0.45",),
            linewidths=0.25, alpha=0.12,
        ))

    # Base layer: every neuron in dark grey
    ax.add_collection(LineCollection(
        segs2d, colors=[base_color], linewidths=lw_base, alpha=0.5,
    ))

    # Green overlay: per-segment alpha driven by the owning neuron's rate.
    alpha = rates_t[seg_owner]
    # Drop segments whose rate is essentially zero to skip useless overdraw.
    keep = alpha > 0.02
    if keep.any():
        rgba = np.tile(np.array([*green, 1.0], dtype=np.float32),
                       (int(keep.sum()), 1))
        rgba[:, 3] = alpha[keep]
        ax.add_collection(LineCollection(
            segs2d[keep], colors=rgba, linewidths=lw_top,
        ))

    if xlim is not None:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    else:
        ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_axis_off()

    if frame_idx is not None:
        txt_color = "white" if bg == "black" else "black"
        label = f"t = {frame_idx:04d}"
        if hd_deg is not None:
            label += f"  HD = {hd_deg:+.0f}°"
        ax.text(0.02, 0.97, label, color=txt_color, fontsize=10,
                family="monospace", ha="left", va="top",
                transform=ax.transAxes)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_path, dpi=150, facecolor=bg, bbox_inches="tight")
    return fig, ax


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anatomy_dir",
                   default="papers/janelia_cx/anatomy/cx_anatomy_test")
    p.add_argument("--datapath",
                   default="papers/Code_NN/Code_NN/Data/Figure5/"
                           "exported-traced-adjacencies-v1.2")
    p.add_argument("--model", default="drosophila_cx_pi",
                   help="config name for the Known-ODE checkpoint")
    p.add_argument("--n_steps", type=int, default=2000,
                   help="number of rollout frames")
    p.add_argument("--stride", type=int, default=2,
                   help="render every Nth frame")
    p.add_argument("--omega_deg", type=float, default=60.0)
    p.add_argument("--theta0", type=float, default=0.0)
    p.add_argument("--elev", type=float, default=-7.6)
    p.add_argument("--azim", type=float, default=86.6)
    p.add_argument("--downsample", type=int, default=10)
    p.add_argument("--out_dir",
                   default="figures/drosophila/3D")
    p.add_argument("--max_frames", type=int, default=None,
                   help="stop after N rendered frames (smoke-test)")
    p.add_argument("--device", default="cpu")
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
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/4] loading model {args.model} ...")
    t0 = time.time()
    net = _load(args.model, device)
    print(f"      done ({time.time() - t0:.1f}s)")

    print(f"[2/4] running constant-omega rollout, "
          f"n_steps={args.n_steps} omega={args.omega_deg}")
    t0 = time.time()
    h_traj, theta = _run_const(net, args.n_steps, float(net.dt),
                                args.omega_deg, args.theta0, device)
    # Match Fig 9 (fig_kinographs_const_omega.py): per-neuron z-score of
    # the subthreshold state h over the rollout, displayed in [-3, 3].
    # Here we map z > 0 to green alpha (z=3 -> saturated).
    mu = h_traj.mean(axis=0, keepdims=True)
    sd = h_traj.std (axis=0, keepdims=True) + 1e-6
    z  = (h_traj - mu) / sd                       # (T, N)
    rates_lit = np.clip(z / 3.0, 0.0, 1.0)        # green only above baseline
    print(f"      done ({time.time() - t0:.1f}s); "
          f"z range = [{z.min():.2f}, {z.max():.2f}]; "
          f"lit median {np.median(rates_lit):.3f}, "
          f"frac > 0.5: {float((rates_lit > 0.5).mean()):.3f}")

    print(f"[3/4] loading skeletons + meshes (downsample={args.downsample}) ...")
    t0 = time.time()
    body_ids = _model_index_to_bodyid(args.datapath)
    assert len(body_ids) == rates_lit.shape[1], (len(body_ids), rates_lit.shape)
    neurons, types_str = _load_skeletons_in_model_order(
        args.anatomy_dir, body_ids, downsample=args.downsample,
    )
    rois = _load_rois(args.anatomy_dir)
    seg_arrays, seg_owner, all_segs = _extract_per_neuron_segments(neurons)
    print(f"      done ({time.time() - t0:.1f}s); "
          f"{all_segs.shape[0]:,} skeleton segments")

    # Project segments once (camera doesn't move).
    segs2d = _project_2d(all_segs.reshape(-1, 3),
                          args.elev, args.azim).reshape(-1, 2, 2)

    # Mesh outline silhouette
    mesh_segs = []
    for mesh in rois.values():
        try:
            outline = mesh.outline().entities
            for ent in outline:
                pts = mesh.vertices[ent.points]
                mesh_segs.extend([(pts[i], pts[i + 1])
                                  for i in range(len(pts) - 1)])
        except Exception:
            pass
    if mesh_segs:
        mesh_segs3d = np.array(mesh_segs)
        mesh_segs2d = _project_2d(mesh_segs3d.reshape(-1, 3),
                                   args.elev, args.azim).reshape(-1, 2, 2)
    else:
        mesh_segs2d = None

    # Frame-invariant view limits: union of every projected point + margin
    pts = np.concatenate(
        [segs2d.reshape(-1, 2)] +
        ([mesh_segs2d.reshape(-1, 2)] if mesh_segs2d is not None else []),
        axis=0,
    )
    pad = 0.04 * (pts.max(0) - pts.min(0))
    xlim = (pts[:, 0].min() - pad[0], pts[:, 0].max() + pad[0])
    ylim = (pts[:, 1].min() - pad[1], pts[:, 1].max() + pad[1])

    # Render loop
    print(f"[4/4] rendering frames into {args.out_dir}/")
    frame_ids = list(range(0, args.n_steps, args.stride))
    if args.max_frames is not None:
        frame_ids = frame_ids[:args.max_frames]

    fig, ax = None, None
    render_times = []
    for k, t in enumerate(frame_ids):
        tic = time.time()
        out = os.path.join(args.out_dir, f"frame_{t:04d}.png")
        fig, ax = _render_frame(
            out, segs2d, seg_owner, rates_lit[t],
            mesh_segs2d=mesh_segs2d,
            xlim=xlim, ylim=ylim,
            frame_idx=t, total_frames=args.n_steps,
            hd_deg=float(np.rad2deg(theta[t])),
            fig_ref=fig, ax_ref=ax,
        )
        render_times.append(time.time() - tic)
        if k < 3 or k % 50 == 0:
            print(f"  frame {t:04d} -> {out}  "
                  f"({render_times[-1]:.2f}s, "
                  f"mean {np.mean(render_times):.2f}s)")

    plt.close(fig)
    print(f"done: {len(frame_ids)} frames, "
          f"mean {np.mean(render_times):.2f}s/frame, "
          f"total {sum(render_times):.1f}s")


if __name__ == "__main__":
    main()
