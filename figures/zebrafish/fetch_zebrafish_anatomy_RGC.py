"""One-off fetch of zebrafish fish2 RGC-target visual-pathway skeletons + ROI meshes.

Run this ONCE on a machine with internet access and a neuprint-fish2 token;
it writes `zebrafish_anatomy_RGC/` (and an optional tarball) which the renderer
(fig_zebrafish_anatomy_3d_RGC.py) reads with no network needed.

Server: neuprint-fish2.janelia.org, dataset 'fish2' (the Ahrens-lab larval-zebrafish
connectome). Skeletons and ROI meshes come in voxel coordinates; this script
converts both to nanometres using the fish2 transform
    x_nm = x_vox * 16 - 21120 * 8
    y_nm = y_vox * 16 - 19200 * 8
    z_nm = z_vox * 15
(see papers/fishFuncEM/fishfuncem/utils/coords.py).

Prereqs:
    pip install navis neuprint-python
    export NEUPRINT_APPLICATION_CREDENTIALS=<your-token>   # or pass --token

Usage:
    python fetch_zebrafish_anatomy_RGC.py [--out zebrafish_anatomy_RGC] [--token TOKEN]

Output tree:
    zebrafish_anatomy_RGC/
        skeletons/<type>__<bodyId>.swc      (one per neuron, x/y/z in nm)
        rois/<ROI>.obj                      (vertices in nm)
        index.csv                           bodyId, type, instance
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile

import pandas as pd


# Default RGC subtypes for the AF5-AF8 inputs plus the pretectal-projecting
# RGCpt population that feeds AF9. RGC (generic) and GC also project broadly
# but are not labelled per AF, so they are excluded from the default set.
RGC_TYPES = ["RGC_AF5", "RGC_AF6", "RGC_AF7", "RGC_AF8", "RGCpt"]

# Pretectal output neurons sampled by Ahrens lab — small population, makes the
# downstream targets of AF6/AF7 visible without bloating the file. Optional.
PRETECTUM_TYPES: list[str] = []  # leave empty by default; add "PyN" / "TRP" to extend

# Neuropil ROIs containing the visual-input arborisation fields plus the
# main downstream visual targets. The fish2 server only serves meshes for
# the short L/R-split AF names; the long "Retinal_Arborization_Field_*"
# variants exist as ROI labels but have no associated mesh.
AF_ROIS = [
    "AF5(L)", "AF5(R)",
    "AF6(L)", "AF6(R)",
    "AF7(L)", "AF7(R)",
    "AF8(L)", "AF8(R)",
    "AF9(L)", "AF9(R)",
    "Pretectum",
    "Tectum",
    "Retina(L)", "Retina(R)",
]

# fish2 voxel -> nm transform (matches fishfuncem.utils.coords.voxel_to_nm)
_SCALE_X = 16
_SCALE_Y = 16
_SCALE_Z = 15
_OFFSET_X = 21120 * 8
_OFFSET_Y = 19200 * 8


def _safe(s: str) -> str:
    return (
        s.replace("/", "_")
        .replace("(", "_")
        .replace(")", "")
        .replace(" ", "_")
    )


def _voxel_to_nm_swc(swc_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the fish2 voxel->nm transform to an SWC DataFrame in place."""
    swc_df["x"] = swc_df["x"] * _SCALE_X - _OFFSET_X
    swc_df["y"] = swc_df["y"] * _SCALE_Y - _OFFSET_Y
    swc_df["z"] = swc_df["z"] * _SCALE_Z
    # radius column in fish2 SWCs comes in voxels; scale to nm using x pitch
    if "radius" in swc_df.columns:
        swc_df["radius"] = swc_df["radius"] * _SCALE_X
    return swc_df


def _write_swc(swc_df: pd.DataFrame, path: str) -> None:
    """Write an SWC dataframe to disk in standard 7-column SWC format."""
    # rowId  type  x  y  z  radius  parent  (type = 0 for unset)
    out = pd.DataFrame({
        "n": swc_df["rowId"].astype(int),
        "type": 0,
        "x": swc_df["x"],
        "y": swc_df["y"],
        "z": swc_df["z"],
        "radius": swc_df["radius"],
        "parent": swc_df["link"].astype(int),
    })
    with open(path, "w") as f:
        f.write("# fish2 skeleton, coordinates in nm\n")
        out.to_csv(f, sep=" ", header=False, index=False, float_format="%.3f")


def _transform_obj_inplace(path: str) -> None:
    """Apply the fish2 voxel->nm transform to OBJ vertex lines."""
    out_lines = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                _, xs, ys, zs = line.split()[:4]
                x = float(xs) * _SCALE_X - _OFFSET_X
                y = float(ys) * _SCALE_Y - _OFFSET_Y
                z = float(zs) * _SCALE_Z
                out_lines.append(f"v {x:.3f} {y:.3f} {z:.3f}\n")
            else:
                out_lines.append(line)
    with open(path, "w") as f:
        f.writelines(out_lines)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="zebrafish_anatomy_RGC",
                   help="output directory")
    p.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("NEUPRINT_TOKEN"),
        help="neuprint-fish2 token (or set NEUPRINT_APPLICATION_CREDENTIALS / "
             "NEUPRINT_TOKEN env var)",
    )
    p.add_argument("--server", default="https://neuprint-fish2.janelia.org")
    p.add_argument("--dataset", default="fish2")
    p.add_argument("--types", nargs="+", default=RGC_TYPES,
                   help="neuron types to fetch (default: RGC_AF5/6/7/8 + RGCpt)")
    p.add_argument("--rois", nargs="+", default=AF_ROIS,
                   help="ROI mesh names to fetch")
    p.add_argument("--no_tar", action="store_true",
                   help="leave the output directory unpacked (skip tarball)")
    args = p.parse_args()

    if not args.token:
        sys.exit("need a neuprint token via --token or "
                 "NEUPRINT_APPLICATION_CREDENTIALS / NEUPRINT_TOKEN env var")

    from neuprint import (
        Client, NeuronCriteria as NC, fetch_neurons, set_default_client,
    )

    client = Client(args.server, dataset=args.dataset, token=args.token)
    set_default_client(client)
    print(f"connected: {args.server} dataset={args.dataset}")

    os.makedirs(os.path.join(args.out, "skeletons"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "rois"), exist_ok=True)

    # --- Per-neuron skeletons --------------------------------------------
    rows = []
    for t in args.types:
        nrns, _ = fetch_neurons(NC(type=t))
        print(f"{t:18s} -> {len(nrns)} neurons")
        for _, row in nrns.iterrows():
            bid = int(row.bodyId)
            try:
                swc_df = client.fetch_skeleton(bid, format="pandas")
            except Exception as e:
                print(f"  skip {bid}: {e}")
                continue
            _voxel_to_nm_swc(swc_df)
            fname = f"{_safe(t)}__{bid}.swc"
            _write_swc(swc_df, os.path.join(args.out, "skeletons", fname))
            rows.append({
                "bodyId": bid,
                "type": t,
                "instance": row.get("instance", "") or "",
                "swc": f"skeletons/{fname}",
            })
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "index.csv"), index=False)
    print(f"wrote {len(rows)} skeletons + index.csv")

    # --- ROI meshes -------------------------------------------------------
    for roi in args.rois:
        fname = f"{_safe(roi)}.obj"
        path = os.path.join(args.out, "rois", fname)
        try:
            client.fetch_roi_mesh(roi, export_path=path)
        except TypeError:
            # older neuprint-python: returns the OBJ payload
            try:
                obj = client.fetch_roi_mesh(roi)
            except Exception as e:
                print(f"  skip ROI {roi}: {e}")
                continue
            mode = "wb" if isinstance(obj, (bytes, bytearray)) else "w"
            with open(path, mode) as f:
                f.write(obj)
        except Exception as e:
            print(f"  skip ROI {roi}: {e}")
            continue
        _transform_obj_inplace(path)
        print(f"  wrote {path}")

    if not args.no_tar:
        tar_path = args.out.rstrip("/") + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(args.out, arcname=os.path.basename(args.out))
        print(f"wrote {tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
