"""One-off fetch of the oculomotor-circuit skeletons + ROI meshes from fish2.

Companion of ``fetch_zebrafish_anatomy_HD.py``, same server, same voxel->nm
transform, same output tree — but it selects cells by **bodyId** rather than
by type, because the oculomotor pool cannot be selected by type on this
server.

Why bodyId. The circuit is defined by ``Oculomotor_sortedData_081126.pkl``,
whose type strings are the colleagues' names. Those do not all exist as
``type`` on neuprint-fish2:

    pkl type        fish2 type      n
    INTG_ipsi_m     INTGip1        38
    INTG_ipsi_i     INTGip2        27
    INTG_contra_m   INTGco1        34
    INTG_contra_i   INTGco2        18
    AMN             AMN            92
    AIN             AIN            35
    AF5_ipsi        (untyped)      29
    AF5_contra      (untyped)      12

The two AF5 populations carry no ``type`` on the server at all, so a
type-based query silently returns 41 fewer cells than the circuit has — the
afferent stage, precisely the part that matters. All 285 bodyIds do resolve,
so the fetch keys on those and writes the PICKLE's type into the filename and
the index, keeping the anatomy consistent with the config and the connectome.

Run ONCE on a machine that can reach neuprint-fish2 and holds a token; it
writes ``zebrafish_anatomy_OCULO/`` which the renderer reads offline.

Prereqs::

    export NEUPRINT_APPLICATION_CREDENTIALS=<your-token>   # or --token

Usage::

    python fetch_zebrafish_anatomy_OCULO.py [--out zebrafish_anatomy_OCULO]

Output tree::

    zebrafish_anatomy_OCULO/
        skeletons/<pkl_type>__<bodyId>.swc   (one per neuron, x/y/z in nm)
        rois/<ROI>.obj                       (vertices in nm)
        index.csv    bodyId, type, fish2_type, instance, side, swc
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch_zebrafish_anatomy_HD import (            # noqa: E402
    _safe, _voxel_to_nm_swc, _write_swc, _transform_obj_inplace,
)

DEFAULT_PKL = os.path.join(os.path.dirname(HERE), "..", "config", "zebrafish",
                           "Oculomotor_sortedData_081126.pkl")
# The 8 types of zebrafish_om_intg_285_v1. Kept here rather than read from the
# yaml so the fetch has no dependency on the model package.
OCULO_TYPES = ["AF5_ipsi", "AF5_contra",
               "INTG_ipsi_m", "INTG_ipsi_i", "INTG_contra_m", "INTG_contra_i",
               "AMN", "AIN"]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, "zebrafish_anatomy_OCULO"))
    p.add_argument("--pkl", default=os.path.abspath(DEFAULT_PKL))
    p.add_argument("--types", nargs="+", default=OCULO_TYPES)
    p.add_argument("--rois", nargs="+", default=None,
                   help="ROI meshes to fetch; default = every ROI the server "
                        "reports for these neurons")
    p.add_argument("--max-rois", type=int, default=12)
    p.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("NEUPRINT_TOKEN"))
    p.add_argument("--server", default="https://neuprint-fish2.janelia.org")
    p.add_argument("--dataset", default="fish2")
    a = p.parse_args()
    if not a.token:
        sys.exit("need a neuprint token via --token or "
                 "NEUPRINT_APPLICATION_CREDENTIALS")

    from neuprint import (Client, NeuronCriteria as NC, fetch_neurons,
                          set_default_client)
    client = Client(a.server, dataset=a.dataset, token=a.token)
    set_default_client(client)
    print(f"connected: {a.server} dataset={a.dataset}")

    with open(a.pkl, "rb") as fh:
        d = pickle.load(fh)
    T = np.asarray(d["Type"]).astype(str)
    H = np.asarray(d["Hemi"]).astype(str)
    NID = np.asarray(d["NeuronIDs"]).astype(np.int64)
    sel = np.isin(T, a.types)
    want = pd.DataFrame({"bodyId": NID[sel], "type": T[sel],
                         "side": [h[0].upper() for h in H[sel]]})
    print(f"circuit: {len(want)} cells, {want['type'].nunique()} types")

    srv, _ = fetch_neurons(NC(bodyId=want.bodyId.tolist()))
    srv = srv.rename(columns={"type": "fish2_type"})
    meta = want.merge(srv[["bodyId", "fish2_type", "instance"]],
                      on="bodyId", how="left")
    missing = int(meta.fish2_type.isna().sum())
    print(f"resolved on the server: {len(srv)}/{len(want)}  "
          f"({missing} carry no fish2 type — expected for AF5)")

    os.makedirs(os.path.join(a.out, "skeletons"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "rois"), exist_ok=True)

    rows, failed = [], []
    for i, r in enumerate(meta.itertuples(), 1):
        try:
            swc = client.fetch_skeleton(int(r.bodyId), format="pandas")
        except Exception as e:
            failed.append((int(r.bodyId), f"{type(e).__name__}"))
            continue
        _voxel_to_nm_swc(swc)
        fname = f"{_safe(r.type)}__{int(r.bodyId)}.swc"
        _write_swc(swc, os.path.join(a.out, "skeletons", fname))
        rows.append({"bodyId": int(r.bodyId), "type": r.type,
                     "fish2_type": r.fish2_type if pd.notna(r.fish2_type) else "",
                     "instance": r.instance if pd.notna(r.instance) else "",
                     "side": r.side, "swc": f"skeletons/{fname}",
                     "n_nodes": len(swc)})
        if i % 40 == 0:
            print(f"  {i}/{len(meta)} skeletons")
    idx = pd.DataFrame(rows)
    idx.to_csv(os.path.join(a.out, "index.csv"), index=False)
    print(f"skeletons: {len(idx)} written, {len(failed)} failed")
    if failed:
        print("  failures:", failed[:8])

    # --- ROI meshes -------------------------------------------------------
    rois = a.rois
    if rois is None:
        # Whatever the server reports as innervated by these cells, most
        # innervated first — better than a hardcoded list that goes stale.
        counts = {}
        for blob in srv.get("roiInfo", []):
            try:
                info = json.loads(blob) if isinstance(blob, str) else (blob or {})
            except Exception:
                continue
            for k, v in info.items():
                counts[k] = counts.get(k, 0) + int(v.get("pre", 0)
                                                   + v.get("post", 0))
        rois = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])
                ][:a.max_rois]
    print(f"ROI meshes: {rois}")
    got = 0
    for roi in rois:
        try:
            mesh = client.fetch_roi_mesh(roi, export_path=None)
        except Exception as e:
            print(f"  skip {roi}: {type(e).__name__}")
            continue
        path = os.path.join(a.out, "rois", f"{_safe(roi)}.obj")
        with open(path, "wb") as f:
            f.write(mesh)
        _transform_obj_inplace(path)
        got += 1
    print(f"ROI meshes: {got}/{len(rois)} written")

    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"server": a.server, "dataset": a.dataset,
                   "source_pickle": os.path.basename(a.pkl),
                   "selected_by": "bodyId (fish2 lacks the pkl type names)",
                   "n_cells": len(idx), "types": a.types,
                   "type_map": (meta.groupby("type")["fish2_type"]
                                .agg(lambda s: sorted(set(s.dropna())))
                                .to_dict()),
                   "rois": rois, "coords": "nm (fish2 voxel->nm transform)"},
                  f, indent=2)
    print(f"[done] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
