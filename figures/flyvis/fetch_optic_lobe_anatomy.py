"""One-off fetch of Janelia optic-lobe skeletons from neuprint.

Mirrors fetch_cx_anatomy.py: same auth, same neuprint client, just a
different dataset (`optic-lobe:v1.0` instead of `hemibrain:v1.2.1`).
This is the lighter-weight alternative to fetch_flywire_anatomy.py
(which requires CAVE / Codex auth). Trade-off: covers a single Janelia
optic lobe, and the (u, v) column scheme may not match flyvis's
FlyWire-derived parquet exactly -- we report match diagnostics so you
can decide whether per-neuron alignment is needed.

Run this ONCE on a machine with internet + neuprint token; produces
`optic_lobe_anatomy.tar.gz`. Drop under `papers/optic_lobe_anatomy/`
in the devcontainer.

Prereqs:
    pip install neuprint-python navis pandas

Usage (same shape as fetch_cx_anatomy.py):
    python figures/flyvis/fetch_optic_lobe_anatomy.py \
        --token <NEUPRINT_JWT> \
        --out optic_lobe_anatomy

Output tree:
    optic_lobe_anatomy/
        skeletons/<type>__<bodyId>.swc
        rois/<ROI>.obj
        neuron_table.csv             # bodyId, type, instance, u, v if present
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import warnings

import pandas as pd

# Latest optic-lobe dataset on neuprint as of 2024. If the user needs a
# different version, pass --dataset.
DEFAULT_DATASET = "optic-lobe:v1.1"

# ROIs known to exist in the optic-lobe dataset.
DEFAULT_ROIS = ("ME(R)", "LO(R)", "LOP(R)", "LA(R)", "AME(R)")

# Janelia's optic-lobe naming differs from flyvis / FlyWire. This dict
# maps every flyvis type that wasn't a direct match to the equivalent
# Janelia type(s). Values are lists because some flyvis types correspond
# to multiple Janelia subtypes (e.g. R7 subdivides into R7d/R7p/R7y).
# Discovered via figures/flyvis/discover_optic_lobe_types.py.
FLYVIS_TO_JANELIA = {
    # R1-R6: Janelia groups them as a single 2265-neuron class.
    "R1": ["R1-R6"], "R2": ["R1-R6"], "R3": ["R1-R6"],
    "R4": ["R1-R6"], "R5": ["R1-R6"], "R6": ["R1-R6"],
    # R7, R8: Janelia splits into yellow (y), pale (p), DRA (d), unclear.
    "R7": ["R7d", "R7p", "R7y", "R7_unclear", "R7R8_unclear"],
    "R8": ["R8d", "R8p", "R8y", "R8_unclear"],
    # CT1: flyvis splits into Lo1/M10 compartments; Janelia has one CT1.
    "CT1(Lo1)": ["CT1"],
    "CT1(M10)": ["CT1"],
    # Amacrine cells: Janelia has only one named Am cell ("Am1").
    "Am": ["Am1"],
    # TmY9: Janelia splits into a/b subtypes.
    "TmY9": ["TmY9a", "TmY9b"],
    # Mi3, Mi11, Mi12, Tm28: no Janelia equivalent in optic-lobe:v1.1.
    # Empty list = skip with warning.
    "Mi3": [],
    "Mi11": [],
    "Mi12": [],
    "Tm28": [],
}


def _safe(s: str) -> str:
    return (s.replace("/", "_").replace("(", "_").replace(")", "")
              .replace(" ", "_"))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="optic_lobe_anatomy")
    p.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImFsbGllcmNlZEBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0tLV2x3cWNkb1hKVzhTdDYyRERhOVhQMHdNX0xHNUpvekRoTmlEQ0pqRDN5SHBMdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTk1OTY2MjE2NH0.JyR51iYA78A1j74LUPEy-GIyT6AjjDgwq75PjyBt0JM",
    )
    p.add_argument("--server", default="https://neuprint.janelia.org")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--types",
                   help="comma-separated cell-type names to fetch. If unset, "
                        "fetch every type present in the flyvis parquet "
                        "(read from --nodes_parquet).")
    p.add_argument("--nodes_parquet", default=None,
                   help="optional flyvis nodes.parquet path. If passed, the "
                        "fetch is restricted to its 65 types and a "
                        "neuron-table mapping is written.")
    p.add_argument("--max_per_type", type=int, default=None,
                   help="cap N skeletons per cell type for a smoke run.")
    p.add_argument("--rois", nargs="+", default=list(DEFAULT_ROIS))
    p.add_argument("--no_tar", action="store_true")
    args = p.parse_args()

    if not args.token:
        sys.exit("need --token or NEUPRINT_APPLICATION_CREDENTIALS env var")

    from neuprint import (
        Client, NeuronCriteria as NC,
        fetch_neurons, fetch_skeleton, set_default_client,
    )
    client = Client(args.server, dataset=args.dataset, token=args.token)
    set_default_client(client)
    print(f"connected: {args.server} dataset={args.dataset}")

    # Decide which types to fetch
    if args.types:
        target_types = [t.strip() for t in args.types.split(",")]
    elif args.nodes_parquet:
        df = pd.read_parquet(args.nodes_parquet)
        target_types = sorted(df.type.unique())
        print(f"loaded flyvis parquet: {len(df)} rows, "
              f"{len(target_types)} types")
    else:
        sys.exit("pass either --types or --nodes_parquet so we know "
                 "which cell types to fetch.")

    os.makedirs(os.path.join(args.out, "skeletons"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "rois"), exist_ok=True)

    # Per-type fetch with dedup: a Janelia bodyId is fetched at most once
    # even when several flyvis types map to it (e.g. R1-R6 is shared by
    # all six flyvis photoreceptor types).
    rows = []                                     # mapping rows for CSV
    fetched_bids = set()                          # dedup across flyvis types
    missing_types = []                            # flyvis types with no map
    t0_global = time.time()
    for t_idx, ft in enumerate(target_types):
        janelia_types = FLYVIS_TO_JANELIA.get(ft, [ft])
        if not janelia_types:
            print(f"[{t_idx+1}/{len(target_types)}] {ft:14s} "
                  f"-> no Janelia equivalent, skipping")
            missing_types.append(ft)
            continue
        per_type_total = 0
        for jt in janelia_types:
            try:
                nrns, _ = fetch_neurons(NC(type=jt))
            except Exception as e:
                print(f"  {ft} -> {jt}: ERR {e}")
                continue
            if len(nrns) == 0:
                print(f"  {ft} -> {jt}: 0 neurons")
                continue
            if args.max_per_type:
                nrns = nrns.iloc[: args.max_per_type]
            t_jt = time.time()
            print(f"  [{t_idx+1}/{len(target_types)}] {ft} -> {jt}: "
                  f"{len(nrns)} neurons (fetching...)", flush=True)
            local_count = 0
            local_fetched = 0
            for _, row in nrns.iterrows():
                bid = int(row.bodyId)
                fname = f"{_safe(jt)}__{bid}.swc"
                path = os.path.join(args.out, "skeletons", fname)
                if bid in fetched_bids or (
                    os.path.exists(path) and os.path.getsize(path) > 100
                ):
                    status = "cached"
                else:
                    try:
                        swc = fetch_skeleton(bid, format="swc")
                    except Exception as e:
                        print(f"    skip {bid}: {e}")
                        continue
                    with open(path, "w") as f:
                        f.write(swc)
                    fetched_bids.add(bid)
                    status = "fetched"
                    local_fetched += 1
                rows.append({
                    "bodyId": bid,
                    "flyvis_type": ft,
                    "janelia_type": jt,
                    "instance": row.get("instance", ""),
                    "swc": f"skeletons/{fname}",
                    "status": status,
                })
                per_type_total += 1
                local_count += 1
                if local_count % 50 == 0:
                    dt = time.time() - t_jt
                    rate = local_count / dt if dt > 0 else 0
                    remain = (len(nrns) - local_count) / rate if rate else 0
                    print(f"      {local_count}/{len(nrns)}  "
                          f"({rate:.1f}/s, eta {remain:.0f}s)", flush=True)
            print(f"    done {jt}: {local_count}/{len(nrns)} "
                  f"({local_fetched} new) in {time.time()-t_jt:.0f}s",
                  flush=True)
        elapsed_total = time.time() - t0_global
        print(f"[{t_idx+1}/{len(target_types)}] {ft:14s} -> "
              f"{','.join(janelia_types):30s} {per_type_total} files "
              f"(cum {elapsed_total:.0f}s, {len(fetched_bids)} unique)",
              flush=True)

    elapsed = time.time() - t0_global
    n_unique = len(set(r["bodyId"] for r in rows))
    print(f"skeleton fetch done in {elapsed:.0f}s; "
          f"{len(rows)} flyvis-mapping rows, {n_unique} unique skeletons")
    if missing_types:
        print(f"flyvis types with no Janelia equivalent: {missing_types}")

    pd.DataFrame(rows).to_csv(
        os.path.join(args.out, "neuron_table.csv"), index=False
    )

    # ROI meshes (Client.fetch_roi_mesh; export_path-or-return version
    # detection from the CX fetch script).
    for roi in args.rois:
        fname = f"{_safe(roi)}.obj"
        path = os.path.join(args.out, "rois", fname)
        if os.path.exists(path):
            continue
        try:
            client.fetch_roi_mesh(roi, export_path=path)
            print(f"  wrote {path}")
        except TypeError:
            try:
                obj = client.fetch_roi_mesh(roi)
            except Exception as e:
                print(f"  skip ROI {roi}: {e}")
                continue
            mode = "wb" if isinstance(obj, (bytes, bytearray)) else "w"
            with open(path, mode) as f:
                f.write(obj)
            print(f"  wrote {path}")
        except Exception as e:
            print(f"  skip ROI {roi}: {e}")

    if not args.no_tar:
        tar_path = args.out.rstrip("/") + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(args.out, arcname=os.path.basename(args.out))
        print(f"wrote {tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
