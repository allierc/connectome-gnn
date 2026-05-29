"""3-D view of the zebrafish dorsal interpeduncular nucleus (dIPN) HD ring.

Renders just the IPNd*/IPNds* skeletons (the HD ring per Petrucco et al. 2023,
"Neural dynamics and architecture of the heading direction circuit in
zebrafish"), each neuron coloured by its angular position around the ring
centroid (-180 to +180, cyclic colormap). L and R rings are handled as two
independent rings whose angles are computed about their own per-hemisphere
soma centroid, matching the L/R-split anatomy on the fish2 server.

Inputs (produced by fetch_zebrafish_anatomy_HD.py):
    <anatomy_dir>/
        skeletons/<type>__<bodyId>.swc      (coords in nm)

Default anatomy_dir: figures/zebrafish/zebrafish_anatomy_HD

Output:
    figures/zebrafish/fig_zebrafish_dIPN_ring.png

Usage:
    python fig_zebrafish_dIPN_ring.py [--elev 90 --azim -90]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _is_dipn(safe_type: str) -> bool:
    """True for the HD ring cell types only (IPNd*, IPNds*); excludes RIPN
    and pt-IPN afferents."""
    return safe_type.startswith("IPNd") or safe_type.startswith("IPNds")


def _load_all_skeletons(anatomy_dir: str, downsample: int = 5):
    """Load every SWC as a navis.NeuronList. Returns (nl, somas, types,
    is_dipn_mask) — the mask flags IPNd*/IPNds* (the HD ring); the
    remainder are the RIPN* / pt-IPN* afferents drawn as grey context."""
    import navis
    swc_paths = sorted(glob.glob(os.path.join(anatomy_dir, "skeletons", "*.swc")))
    if not swc_paths:
        sys.exit(f"no SWCs under {anatomy_dir}/skeletons/ -- "
                 "run fetch_zebrafish_anatomy_HD.py first")

    neurons, somas, types, is_dipn = [], [], [], []
    for path in swc_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        safe_type, _, _ = stem.rpartition("__")
        n = navis.read_swc(path)
        if downsample and downsample > 1:
            n = navis.downsample_neuron(n, downsampling_factor=downsample,
                                         preserve_nodes=None)
        nodes = n.nodes
        i_max = int(nodes.radius.values.argmax())
        row = nodes.iloc[i_max]
        somas.append([float(row.x), float(row.y), float(row.z)])
        neurons.append(n)
        types.append(safe_type)
        is_dipn.append(_is_dipn(safe_type))

    if not neurons:
        sys.exit("no skeletons found")

    return (navis.NeuronList(neurons), np.asarray(somas),
            np.asarray(types), np.asarray(is_dipn, dtype=bool))


def _angle_around_centroid(somas: np.ndarray) -> np.ndarray:
    """Angle (in degrees, -180..180) of each soma about the per-hemisphere
    centroid in the dorsal (x, y) plane. Splits L/R by the sign of x relative
    to the global x-median of the input somas."""
    if len(somas) == 0:
        return np.zeros(0)
    x_split = np.median(somas[:, 0])
    angles = np.zeros(len(somas))
    for side in (0, 1):
        mask = (somas[:, 0] < x_split) if side == 0 else (somas[:, 0] >= x_split)
        if not mask.any():
            continue
        cx = somas[mask, 0].mean()
        cy = somas[mask, 1].mean()
        dx = somas[mask, 0] - cx
        dy = somas[mask, 1] - cy
        angles[mask] = np.degrees(np.arctan2(dy, dx))
    return angles


def _extract_segments_per_neuron(nl):
    """One (parent, child) numpy array per neuron, so we can colour each
    skeleton individually by its ring angle."""
    out = []
    for n in nl:
        nodes = n.nodes
        child = nodes[nodes.parent_id != -1]
        if len(child) == 0:
            out.append(np.empty((0, 2, 3)))
            continue
        parent_xyz = nodes.set_index("node_id").loc[
            child.parent_id.values, ["x", "y", "z"]
        ].values
        child_xyz = child[["x", "y", "z"]].values
        out.append(np.stack([parent_xyz, child_xyz], axis=1))
    return out


def _project_2d(xyz: np.ndarray, elev: float, azim: float) -> np.ndarray:
    """Project (N, 3) world coords to (N, 2) screen coords matching the
    matplotlib mplot3d convention for (elev, azim) in degrees."""
    e = np.deg2rad(elev)
    a = np.deg2rad(azim)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    R = np.array([[-sa,           ca,         0.0],
                  [-ca * se,    -sa * se,     ce ]])
    return xyz @ R.T


def _render(nl, somas, is_dipn, angles_dipn, output_path,
            elev=90.0, azim=-90.0,
            cmap_name="hsv",
            figsize=(8.0, 8.0), dpi=240,
            background="black",
            lw=0.5,
            grey_color=(0.30, 0.30, 0.30),
            grey_alpha=0.30,
            grey_lw=0.25):
    """Render every neuron — dIPN HD-ring cells coloured by ring angle,
    everything else (RIPN*, pt-IPN*) drawn underneath in dark grey as
    anatomical context. `angles_dipn` has one entry per True position in
    `is_dipn`, in the same order."""
    from matplotlib.collections import LineCollection

    text_color = "white" if background == "black" else "black"
    cmap = plt.get_cmap(cmap_name)
    # Map angle -180..180 -> 0..1 for the cyclic colormap.
    dipn_colors = cmap((angles_dipn + 180.0) / 360.0)

    segs_per_neuron = _extract_segments_per_neuron(nl)

    fig, ax = plt.subplots(figsize=figsize, facecolor=background)
    ax.set_facecolor(background)

    # --- backdrop: non-dIPN neurons (RIPN*, pt-IPN*) in dark grey ----
    grey_segs = []
    for segs3d, is_d in zip(segs_per_neuron, is_dipn):
        if is_d or len(segs3d) == 0:
            continue
        segs2d = _project_2d(segs3d.reshape(-1, 3), elev, azim).reshape(-1, 2, 2)
        grey_segs.append(segs2d)
    if grey_segs:
        ax.add_collection(LineCollection(
            np.concatenate(grey_segs, axis=0),
            colors=[grey_color], linewidths=grey_lw, alpha=grey_alpha,
            zorder=1,
        ))

    # --- dIPN: somas only (rainbow dots) on top of the grey backdrop ---
    soma_xy = _project_2d(somas, elev, azim)
    ax.scatter(soma_xy[is_dipn, 0], soma_xy[is_dipn, 1],
               c=dipn_colors, s=18, edgecolors="none", zorder=4)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()

    # Colour bar / angle legend matching the -180..180 labels in the
    # reference figure (Petrucco et al. 2023).
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=-180, vmax=180))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.04, pad=0.04, aspect=30)
    cbar.set_label("ring angle (deg)", color=text_color)
    cbar.set_ticks([-180, -90, 0, 90, 180])
    cbar.ax.tick_params(colors=text_color)
    cbar.outline.set_edgecolor(text_color)

    n_dipn = int(is_dipn.sum())
    n_other = int((~is_dipn).sum())
    dipn_somas = somas[is_dipn]
    n_dipn_L = int((dipn_somas[:, 0] < np.median(dipn_somas[:, 0])).sum())
    title = (f"dIPN HD ring  |  ring n={n_dipn}  "
             f"(L={n_dipn_L}, R={n_dipn - n_dipn_L})  |  "
             f"+{n_other} RIPN/pt-IPN context  |  "
             "colour = soma angle about per-hemisphere centroid")
    ax.set_title(title, color=text_color, fontsize=9, pad=8)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.10)
    fig.savefig(output_path, dpi=dpi, facecolor=background,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}  (n={len(nl)}, bg={background})")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy_dir",
                   default=os.path.join(here, "zebrafish_anatomy_HD"))
    p.add_argument("--out", default=os.path.join(here, "fig_zebrafish_dIPN_ring.png"))
    p.add_argument("--downsample", type=int, default=5,
                   help="navis.downsample_neuron factor; 1 = no downsample")
    p.add_argument("--elev", type=float, default=90.0,
                   help="default 90 = dorsal view (look down at brain)")
    p.add_argument("--azim", type=float, default=-90.0)
    p.add_argument("--cmap", default="hsv",
                   help="cyclic colormap (hsv, twilight, twilight_shifted)")
    p.add_argument("--bg", default="black", choices=["black", "white"])
    p.add_argument("--lw", type=float, default=0.5)
    args = p.parse_args()

    if not os.path.isdir(args.anatomy_dir):
        sys.exit(f"{args.anatomy_dir} does not exist -- "
                 "run fetch_zebrafish_anatomy_HD.py first")

    nl, somas, types, is_dipn = _load_all_skeletons(args.anatomy_dir,
                                                     downsample=args.downsample)
    angles_dipn = _angle_around_centroid(somas[is_dipn])
    print(f"loaded {len(nl)} neurons: "
          f"{int(is_dipn.sum())} dIPN ring + {int((~is_dipn).sum())} "
          f"RIPN/pt-IPN context  ({len(np.unique(types))} subtypes)")

    _render(nl, somas, is_dipn, angles_dipn, args.out,
            elev=args.elev, azim=args.azim,
            cmap_name=args.cmap, background=args.bg, lw=args.lw)


if __name__ == "__main__":
    main()
