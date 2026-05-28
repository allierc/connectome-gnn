"""One-off fetch of hemibrain CX neuron skeletons + ROI meshes (+ optional
soma meshes).

Run this ONCE on a machine with internet access and a neuprint token; it
produces `cx_anatomy.tar.gz` (~tens of MB without `--with_somas`,
~few-hundred MB with) which you then drop under
`papers/janelia_cx/anatomy/`. The renderer (fig_cx_anatomy_3d.py) reads
the unpacked tarball; no network needed there.

Prereqs:
    pip install navis neuprint-python                      # always
    pip install dvidtools                                  # for --with_somas
    export NEUPRINT_APPLICATION_CREDENTIALS=<your-token>     # or pass --token

Network requirements:
    - https://neuprint.janelia.org           (skeletons, ROI meshes)
    - https://hemibrain-dvid.janelia.org     (full per-neuron meshes,
                                              only when --with_somas)

Usage:
    python fetch_cx_anatomy.py [--out cx_anatomy] [--token TOKEN] \\
        [--dataset hemibrain:v1.2.1] [--with_somas]

Output tree:
    cx_anatomy/
        skeletons/<type>__<bodyId>.swc       # 156 files (one per neuron)
        rois/<ROI>.obj                       # EB, PB, FB, NO, BU_L, BU_R
        somas/<type>__<bodyId>.obj           # only with --with_somas; full
                                             #   neuron mesh cropped to a
                                             #   sphere around the SWC soma
                                             #   node (radius = swc_radius *
                                             #   soma_pad)
        index.csv                            # bodyId, type, instance, hemisphere
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile

import pandas as pd

# These cell types match the 156-neuron CX set used in
# load_drosophila_cx_connectome() (EPG, EPGt, PEN_a, PEN_b, Delta7, PEG, ER6).
CX_TYPES = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "Delta7", "PEG", "ER6"]

# Neuropil ROIs containing the CX columns (hemibrain naming).
CX_ROIS = ["EB", "PB", "FB", "NO", "BU(L)", "BU(R)"]


def _safe(s: str) -> str:
    return s.replace("/", "_").replace("(", "_").replace(")", "").replace(" ", "_")


HEMIBRAIN_VOXEL_NM = 8.0  # neuPrint SWC + ROI meshes are in 8 nm voxels;
                          # DVID per-neuron meshes are in nm -> scale the
                          # latter down so all geometry shares one frame.


def _soma_center_radius_from_swc(swc_path: str):
    """Read SWC and return (xyz_voxel, radius_voxel) of the
    largest-radius node. Hemibrain SWCs don't tag the soma with
    type=1; the cell body is reliably the only node with radius far
    above the segmental baseline (skeleton radii ~5-30 voxels, soma
    radii >100 voxels). Coordinates are kept in the neuPrint native
    voxel frame so the cropped soma mesh matches the SWC and ROI
    meshes the renderer also reads."""
    import navis
    n = navis.read_swc(swc_path)
    i = int(n.nodes.radius.values.argmax())
    row = n.nodes.iloc[i]
    import numpy as np
    xyz = np.array([float(row.x), float(row.y), float(row.z)])
    return xyz, float(row.radius)


def _crop_mesh_to_sphere(mesh, center, radius):
    """Return a submesh of `mesh` whose faces all sit inside the sphere of
    `radius` around `center`. None if no faces survive."""
    import numpy as np
    d = np.linalg.norm(mesh.vertices - center, axis=1)
    keep_v = d < radius
    if not keep_v.any():
        return None
    face_mask = keep_v[mesh.faces].all(axis=1)
    if not face_mask.any():
        return None
    return mesh.submesh([face_mask], append=True)


DVID_MESH_SERVER = "https://hemibrain-dvid.janelia.org"
DVID_MESH_NODE = "31597d95bd844060b0ccc928a1a8a0a4"   # leaf node advertised
                                                       # by neuPrint meta as
                                                       # `uuid` for v1.2.1;
                                                       # lives in repo
                                                       # `hemibrain-flattened`
                                                       # which holds the
                                                       # `segmentation_meshes`
                                                       # keyvalue instance.


def _fetch_somas(out_dir: str, rows: list, pad: float = 1.4,
                  max_threads: int = 4,
                  dvid_server: str = DVID_MESH_SERVER,
                  dvid_node: str = DVID_MESH_NODE) -> None:
    """Fetch the full hemibrain mesh per neuron, crop to a sphere around
    the SWC soma node, and save the cropped mesh as
    `somas/<safe-type>__<bodyId>.obj`. Calls `dvid.get_meshes` directly
    against `hemibrain-dvid` at the v1.2.1 leaf UUID (the navis wrapper
    auto-resolves to the wrong repo for hemibrain and silently returns
    empty meshes)."""
    import dvid
    import trimesh
    os.makedirs(os.path.join(out_dir, "somas"), exist_ok=True)
    n_ok, n_skip = 0, 0
    for r in rows:
        bid = int(r["bodyId"])
        t = r["type"]
        safe_t = _safe(t)
        swc_path = os.path.join(out_dir, r["swc"])
        if not os.path.exists(swc_path):
            print(f"  skip soma {bid}: SWC missing at {swc_path}")
            n_skip += 1
            continue
        center, r_swc = _soma_center_radius_from_swc(swc_path)
        try:
            res = dvid.get_meshes(
                bid, server=dvid_server, node=dvid_node,
                output="trimesh", on_error="raise", progress=False,
                max_threads=max_threads,
            )
        except Exception as e:
            print(f"  skip soma {bid}: fetch failed: {e}")
            n_skip += 1
            continue
        # singleton input -> list-of-1
        mesh = res[0] if isinstance(res, (list, tuple)) else res
        if mesh is None or len(mesh.vertices) <= 1:
            print(f"  skip soma {bid}: empty mesh")
            n_skip += 1
            continue
        # DVID returns mesh vertices in nm; SWC + ROI meshes are in 8 nm
        # voxels, so rescale to keep one frame across all loaded geometry.
        mesh = trimesh.Trimesh(
            vertices=mesh.vertices / HEMIBRAIN_VOXEL_NM,
            faces=mesh.faces, process=False,
        )
        cropped = _crop_mesh_to_sphere(mesh, center, r_swc * pad)
        if cropped is None or len(cropped.vertices) < 8:
            print(f"  skip soma {bid}: crop empty (r_swc={r_swc:.0f}, "
                  f"pad={pad})")
            n_skip += 1
            continue
        out_path = os.path.join(out_dir, "somas", f"{safe_t}__{bid}.obj")
        cropped.export(out_path)
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"  somas: {n_ok} written")
    print(f"somas done: {n_ok} written, {n_skip} skipped")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="cx_anatomy", help="output directory")
    p.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFsbGllcmNlZEBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tLV2x3cWNkb1hKVzhTdDYyRERhOVhQMHdNX0xHNUpvekRoTmlEQ0pqRDN5SHBMdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTk1OTY2MjE2NH0.JyR51iYA78A1j74LUPEy-GIyT6AjjDgwq75PjyBt0JM",
    )
    p.add_argument("--server", default="https://neuprint.janelia.org")
    p.add_argument("--dataset", default="hemibrain:v1.2.1")
    p.add_argument("--no_tar", action="store_true",
                   help="leave the output directory unpacked (skip tarball)")
    p.add_argument("--with_somas", action="store_true",
                   help="also fetch the full per-neuron 3D mesh from "
                        "hemibrain-dvid, crop to a sphere around the SWC "
                        "soma node, and save the cropped soma mesh as "
                        "somas/<type>__<bodyId>.obj")
    p.add_argument("--soma_pad", type=float, default=1.4,
                   help="multiplicative pad on the SWC soma radius when "
                        "cropping the full mesh (default 1.4 captures the "
                        "whole cell body without grabbing arbor)")
    p.add_argument("--soma_max_threads", type=int, default=4,
                   help="max parallel DVID mesh fetches (default 4)")
    args = p.parse_args()

    if not args.token:
        sys.exit("need a neuprint token via --token or "
                 "NEUPRINT_APPLICATION_CREDENTIALS env var")

    from neuprint import (
        Client, NeuronCriteria as NC, fetch_neurons, fetch_skeleton,
        set_default_client,
    )
    import navis

    # Hold a reference and explicitly register as the default; some
    # neuprint-python versions don't auto-register and the Client gets GC'd.
    client = Client(args.server, dataset=args.dataset, token=args.token)
    set_default_client(client)
    print(f"connected: {args.server} dataset={args.dataset}")

    os.makedirs(os.path.join(args.out, "skeletons"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "rois"), exist_ok=True)

    # --- Per-neuron skeletons --------------------------------------------
    rows = []
    for t in CX_TYPES:
        nrns, _ = fetch_neurons(NC(type=t))
        print(f"{t:20s} -> {len(nrns)} neurons")
        for _, row in nrns.iterrows():
            bid = int(row.bodyId)
            try:
                swc = fetch_skeleton(bid, format="swc")
            except Exception as e:
                print(f"  skip {bid}: {e}")
                continue
            fname = f"{_safe(t)}__{bid}.swc"
            with open(os.path.join(args.out, "skeletons", fname), "w") as f:
                f.write(swc)
            rows.append({
                "bodyId": bid, "type": t,
                "instance": row.get("instance", ""),
                "swc": f"skeletons/{fname}",
            })
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "index.csv"), index=False)
    print(f"wrote {len(rows)} skeletons + index.csv")

    # --- ROI meshes -------------------------------------------------------
    # fetch_roi_mesh is a Client method (not a top-level function). The
    # return type depends on the neuprint-python version: older returns the
    # OBJ payload (str or bytes), newer accepts export_path= and writes
    # directly. Try export_path first, then fall back to capturing the
    # return value.
    for roi in CX_ROIS:
        fname = f"{_safe(roi)}.obj"
        path = os.path.join(args.out, "rois", fname)
        try:
            client.fetch_roi_mesh(roi, export_path=path)
        except TypeError:
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
        print(f"  wrote {path}")

    # --- per-neuron soma meshes (optional) -------------------------------
    if args.with_somas:
        _fetch_somas(args.out, rows, pad=args.soma_pad,
                     max_threads=args.soma_max_threads)

    if not args.no_tar:
        tar_path = args.out.rstrip("/") + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(args.out, arcname=os.path.basename(args.out))
        print(f"wrote {tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
