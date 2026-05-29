"""3-D view of the zebrafish IPN12_a / IPN12_b neurons over the HD-circuit backdrop.

These two cell types are present on neuprint-fish2 in the IPN region but are
NOT part of the 731-neuron HD-circuit fetch (`fetch_zebrafish_anatomy_HD.py`
covers IPNd*, IPNds*, RIPN*, pt-IPN* only). This script

  1. caches IPN12_a / IPN12_b skeletons from the fish2 server on first run
     (writes `zebrafish_anatomy_IPN12/skeletons/<type>__<bodyId>.swc`),
  2. loads the existing HD-circuit skeletons in `zebrafish_anatomy_HD/` as
     a dark-grey backdrop (same look as `fig_zebrafish_dIPN_ring.py`),
  3. overlays the IPN12 cells in a distinct colour on top.

Auth / network: needs a working `NEUPRINT_APPLICATION_CREDENTIALS` env var
(or `--token`) and reachability of `neuprint-fish2.janelia.org`. Once the
SWCs are cached under `--ipn12_dir`, the script runs offline.

Output: figures/zebrafish/fig_zebrafish_IPN12.png

Usage examples:
    python fig_zebrafish_IPN12.py                       # default: dorsal view
    python fig_zebrafish_IPN12.py --elev 5.7 --azim -92.4  # lateral
    python fig_zebrafish_IPN12.py --types IPN12_a       # only one subtype
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
import pandas as pd


# Match the convention of fetch_zebrafish_anatomy_HD.py for SWC scaling.
_SCALE_X = 16
_SCALE_Y = 16
_SCALE_Z = 15
_OFFSET_X = 21120 * 8
_OFFSET_Y = 19200 * 8

IPN12_TYPES_DEFAULT = ["IPN12_a", "IPN12_b"]


def _safe(s: str) -> str:
    return (s.replace("/", "_").replace("(", "_")
             .replace(")", "").replace(" ", "_"))


def _voxel_to_nm_swc(swc_df: pd.DataFrame) -> pd.DataFrame:
    swc_df["x"] = swc_df["x"] * _SCALE_X - _OFFSET_X
    swc_df["y"] = swc_df["y"] * _SCALE_Y - _OFFSET_Y
    swc_df["z"] = swc_df["z"] * _SCALE_Z
    if "radius" in swc_df.columns:
        swc_df["radius"] = swc_df["radius"] * _SCALE_X
    return swc_df


def _write_swc(swc_df: pd.DataFrame, path: str) -> None:
    out = pd.DataFrame({
        "n": swc_df["rowId"].astype(int),
        "type": 0,
        "x": swc_df["x"], "y": swc_df["y"], "z": swc_df["z"],
        "radius": swc_df["radius"],
        "parent": swc_df["link"].astype(int),
    })
    with open(path, "w") as f:
        f.write("# fish2 skeleton, coordinates in nm\n")
        out.to_csv(f, sep=" ", header=False, index=False, float_format="%.3f")


def _fetch_ipn12(ipn12_dir: str, types, token: str, server: str,
                 dataset: str) -> None:
    """Cache IPN12 skeletons under `ipn12_dir/skeletons/`. No-op if a
    type is already cached (any SWC matching `<type>__*.swc` present)."""
    os.makedirs(os.path.join(ipn12_dir, "skeletons"), exist_ok=True)
    needs_fetch = []
    for t in types:
        present = glob.glob(
            os.path.join(ipn12_dir, "skeletons", f"{_safe(t)}__*.swc")
        )
        if present:
            print(f"  cached {_safe(t)}: {len(present)} SWCs")
        else:
            needs_fetch.append(t)
    if not needs_fetch:
        return

    if not token:
        sys.exit(
            "need NEUPRINT_APPLICATION_CREDENTIALS / --token to fetch "
            f"{needs_fetch} from {server}"
        )

    from neuprint import (Client, NeuronCriteria as NC,
                          fetch_neurons, set_default_client)
    client = Client(server, dataset=dataset, token=token)
    set_default_client(client)
    print(f"connected: {server} dataset={dataset}")

    index_rows = []
    for t in needs_fetch:
        nrns, _ = fetch_neurons(NC(type=t))
        print(f"  {t}: {len(nrns)} neurons")
        for _, row in nrns.iterrows():
            bid = int(row.bodyId)
            try:
                swc_df = client.fetch_skeleton(bid, format="pandas")
            except Exception as e:
                print(f"    skip {bid}: {e}")
                continue
            _voxel_to_nm_swc(swc_df)
            fname = f"{_safe(t)}__{bid}.swc"
            path = os.path.join(ipn12_dir, "skeletons", fname)
            _write_swc(swc_df, path)
            index_rows.append({
                "bodyId": bid, "type": t,
                "instance": row.get("instance", "") or "",
                "swc": f"skeletons/{fname}",
            })

    if index_rows:
        idx_path = os.path.join(ipn12_dir, "index.csv")
        prev = (pd.read_csv(idx_path) if os.path.isfile(idx_path)
                else pd.DataFrame())
        pd.concat([prev, pd.DataFrame(index_rows)], ignore_index=True
                  ).to_csv(idx_path, index=False)
        print(f"  wrote {len(index_rows)} new SWCs to {idx_path}")


def _load_swcs(skeleton_dir: str, downsample: int = 5):
    """Return a navis NeuronList of every SWC in `skeleton_dir`."""
    import navis
    paths = sorted(glob.glob(os.path.join(skeleton_dir, "*.swc")))
    if not paths:
        return navis.NeuronList([]), []
    neurons, names = [], []
    for path in paths:
        n = navis.read_swc(path)
        if downsample and downsample > 1:
            n = navis.downsample_neuron(n, downsampling_factor=downsample,
                                         preserve_nodes=None)
        neurons.append(n)
        names.append(os.path.splitext(os.path.basename(path))[0])
    return navis.NeuronList(neurons), names


def _extract_segments(nl):
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
    e = np.deg2rad(elev)
    a = np.deg2rad(azim)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    R = np.array([[-sa,        ca,         0.0],
                  [-ca * se, -sa * se,     ce ]])
    return xyz @ R.T


def _render(nl_bg, nl_fg, fg_names, output_path,
            elev=90.0, azim=-90.0,
            figsize=(15.0, 8.0), dpi=240,
            background="black",
            bg_color=(0.30, 0.30, 0.30), bg_alpha=0.25, bg_lw=0.20,
            fg_lw=0.30, fg_alpha=0.95,
            soma_size=8.0):
    """Two-panel render — one panel per IPN12 subtype. Both panels share
    the same dark-grey HD-circuit backdrop and view, so the spatial
    organisation of the two cell types is directly comparable."""
    from matplotlib.collections import LineCollection

    text_color = "white" if background == "black" else "black"

    # Pre-compute the backdrop once (same for both panels).
    bg_segs = _extract_segments(nl_bg)
    flat_bg = (np.concatenate([s for s in bg_segs if len(s)], axis=0)
               if bg_segs else np.zeros((0, 2, 3)))
    bg_seg2d = (_project_2d(flat_bg.reshape(-1, 3), elev, azim)
                .reshape(-1, 2, 2) if len(flat_bg) else None)

    # Pre-compute foreground per subtype.
    palette = plt.get_cmap("tab10").colors
    subtype_of = [name.split("__")[0] for name in fg_names]
    uniq = sorted(set(subtype_of))
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(uniq)}

    fg_segs = _extract_segments(nl_fg)
    by_subtype = {t: {"seg2d": [], "soma": []} for t in uniq}
    for n, segs3d, name in zip(nl_fg, fg_segs, fg_names):
        if not len(segs3d):
            continue
        t = name.split("__")[0]
        seg2d = _project_2d(segs3d.reshape(-1, 3),
                             elev, azim).reshape(-1, 2, 2)
        by_subtype[t]["seg2d"].append(seg2d)
        nodes = n.nodes
        i_max = int(nodes.radius.values.argmax())
        row = nodes.iloc[i_max]
        by_subtype[t]["soma"].append(
            [float(row.x), float(row.y), float(row.z)])

    # Shared view limits across panels for direct comparability.
    bbox_pts = []
    if bg_seg2d is not None:
        bbox_pts.append(bg_seg2d.reshape(-1, 2))
    for t in uniq:
        for s in by_subtype[t]["seg2d"]:
            bbox_pts.append(s.reshape(-1, 2))
    if bbox_pts:
        all_pts = np.concatenate(bbox_pts, axis=0)
        pad = 0.03 * (all_pts.max(0) - all_pts.min(0))
        xlim = (all_pts[:, 0].min() - pad[0],
                all_pts[:, 0].max() + pad[0])
        ylim = (all_pts[:, 1].min() - pad[1],
                all_pts[:, 1].max() + pad[1])
    else:
        xlim = ylim = None

    fig, axes = plt.subplots(1, len(uniq), figsize=figsize,
                              facecolor=background, squeeze=False)
    axes = axes[0]
    n_bg = len(nl_bg)
    for ax, t in zip(axes, uniq):
        ax.set_facecolor(background)

        # backdrop
        if bg_seg2d is not None:
            ax.add_collection(LineCollection(
                bg_seg2d, colors=[bg_color], linewidths=bg_lw,
                alpha=bg_alpha, zorder=1,
            ))

        # this subtype's skeletons
        color = color_map[t]
        sub_seg = by_subtype[t]["seg2d"]
        sub_soma = by_subtype[t]["soma"]
        if sub_seg:
            ax.add_collection(LineCollection(
                np.concatenate(sub_seg, axis=0),
                colors=[color], linewidths=fg_lw,
                alpha=fg_alpha, zorder=3,
            ))
        if sub_soma:
            soma_xy = _project_2d(np.asarray(sub_soma), elev, azim)
            ax.scatter(soma_xy[:, 0], soma_xy[:, 1],
                       c=[color], s=soma_size, edgecolors="none",
                       zorder=4)

        if xlim is not None:
            ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_axis_off()
        n_t = len(sub_soma)
        ax.set_title(f"{t}  (n={n_t})  over HD-circuit backdrop "
                     f"(n_bg={n_bg})",
                     color=text_color, fontsize=9, pad=6)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.02,
                        wspace=0.02)
    fig.savefig(output_path, dpi=dpi, facecolor=background,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}  (n_fg={len(nl_fg)}, "
          f"n_bg={n_bg}, subtypes={uniq}, bg={background})")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anatomy_dir",
                   default=os.path.join(here, "zebrafish_anatomy_HD"),
                   help="HD-circuit skeletons (drawn as dark-grey backdrop)")
    p.add_argument("--ipn12_dir",
                   default=os.path.join(here, "zebrafish_anatomy_IPN12"),
                   help="cache dir for IPN12_a/b skeletons (auto-fetched "
                        "from neuprint-fish2 on first run)")
    p.add_argument("--types", nargs="+", default=IPN12_TYPES_DEFAULT,
                   help="IPN12 subtypes to overlay")
    p.add_argument("--out",
                   default=os.path.join(here, "fig_zebrafish_IPN12.png"))
    p.add_argument("--downsample", type=int, default=5)
    p.add_argument("--elev", type=float, default=90.0,
                   help="default 90 = dorsal view")
    p.add_argument("--azim", type=float, default=-90.0)
    p.add_argument("--bg", default="black", choices=["black", "white"])
    p.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("NEUPRINT_TOKEN"),
    )
    p.add_argument("--server", default="https://neuprint-fish2.janelia.org")
    p.add_argument("--dataset", default="fish2")
    args = p.parse_args()

    if not os.path.isdir(args.anatomy_dir):
        sys.exit(f"{args.anatomy_dir} does not exist -- "
                 "run fetch_zebrafish_anatomy_HD.py first")

    _fetch_ipn12(args.ipn12_dir, args.types, args.token,
                 args.server, args.dataset)

    nl_bg, _ = _load_swcs(os.path.join(args.anatomy_dir, "skeletons"),
                          downsample=args.downsample)
    nl_fg, fg_names = _load_swcs(
        os.path.join(args.ipn12_dir, "skeletons"),
        downsample=args.downsample,
    )
    if not len(nl_fg):
        sys.exit(f"no IPN12 SWCs in {args.ipn12_dir}/skeletons/ -- "
                 "fetch failed or wrong types?")

    print(f"loaded {len(nl_bg)} backdrop neurons + "
          f"{len(nl_fg)} IPN12 neurons")
    _render(nl_bg, nl_fg, fg_names, args.out,
            elev=args.elev, azim=args.azim, background=args.bg)


if __name__ == "__main__":
    main()
