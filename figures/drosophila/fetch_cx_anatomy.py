"""One-off fetch of hemibrain CX neuron skeletons + ROI meshes.

Run this ONCE on a machine with internet access and a neuprint token; it
produces `cx_anatomy.tar.gz` (~tens of MB) which you then drop under
`papers/janelia_cx/anatomy/`. The renderer (fig_cx_anatomy_3d.py) reads
the unpacked tarball; no network needed there.

Prereqs:
    pip install navis neuprint-python
    export NEUPRINT_APPLICATION_CREDENTIALS=<your-token>     # or pass --token

Usage:
    python fetch_cx_anatomy.py [--out cx_anatomy] [--token TOKEN] \\
        [--dataset hemibrain:v1.2.1]

Output tree:
    cx_anatomy/
        skeletons/<type>__<bodyId>.swc       # 156 files (one per neuron)
        rois/<ROI>.obj                       # EB, PB, FB, NO, BU_L, BU_R
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

    if not args.no_tar:
        tar_path = args.out.rstrip("/") + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(args.out, arcname=os.path.basename(args.out))
        print(f"wrote {tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
