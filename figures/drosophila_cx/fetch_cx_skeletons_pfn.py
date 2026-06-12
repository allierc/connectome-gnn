"""Fetch hemibrain SWC skeletons for the path-integration extension cell
types (PFNd, PFNv, PFNa, hDeltaB, PFR_a, PFR_b) and add them to an existing
CX anatomy cache so the Figure-1 anatomy render can show the new
populations alongside the original 156-cell heading set.

Lean companion to ``fetch_cx_anatomy.py``: skeletons only (no navis / no
soma meshes), so it runs with just ``neuprint-python``. The neuropil ROIs
the new cells occupy (FB, NO, EB, PB, BU) are already in the cache.

Writes ``skeletons/<safe-type>__<bodyId>.swc`` and appends the new rows to
``index.csv`` (existing rows preserved, de-duplicated by bodyId).

Server: neuprint.janelia.org, dataset 'hemibrain:v1.2.1'.

Usage::

    export NEUPRINT_TOKEN=<token>
    python fetch_cx_skeletons_pfn.py \\
        --out ../../papers/janelia_cx/anatomy/cx_anatomy_test
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Path-integration extension types (exact hemibrain v1.2.1 type names).
PI_EXTENSION_TYPES = ["PFNd", "PFNv", "PFNa", "hDeltaB", "PFR_a", "PFR_b"]


def _safe(s: str) -> str:
    return s.replace("/", "_").replace("(", "_").replace(")", "").replace(" ", "_")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", required=True,
                   help="existing CX anatomy cache dir (skeletons/ + index.csv)")
    p.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("NEUPRINT_TOKEN"),
    )
    p.add_argument("--server", default="https://neuprint.janelia.org")
    p.add_argument("--dataset", default="hemibrain:v1.2.1")
    p.add_argument("--types", nargs="+", default=PI_EXTENSION_TYPES)
    args = p.parse_args()

    if not args.token:
        sys.exit("need a neuprint token via --token or NEUPRINT_TOKEN env var")

    from neuprint import (
        Client, NeuronCriteria as NC, fetch_neurons, fetch_skeleton,
        set_default_client,
    )

    client = Client(args.server, dataset=args.dataset, token=args.token)
    set_default_client(client)
    print(f"connected: {args.server} dataset={args.dataset}")

    skel_dir = os.path.join(args.out, "skeletons")
    os.makedirs(skel_dir, exist_ok=True)

    rows = []
    for t in args.types:
        nrns, _ = fetch_neurons(NC(type=t))
        print(f"{t:18s} -> {len(nrns)} neurons")
        for _, row in nrns.iterrows():
            bid = int(row.bodyId)
            try:
                swc = fetch_skeleton(bid, format="swc")
            except Exception as e:
                print(f"  skip {bid}: {e}")
                continue
            fname = f"{_safe(t)}__{bid}.swc"
            with open(os.path.join(skel_dir, fname), "w") as f:
                f.write(swc)
            rows.append({
                "bodyId": bid, "type": t,
                "instance": row.get("instance", ""),
                "swc": f"skeletons/{fname}",
            })
    new_df = pd.DataFrame(rows)
    print(f"fetched {len(new_df)} new skeletons")

    # --- merge into index.csv (preserve existing, de-dup by bodyId) ------
    index_path = os.path.join(args.out, "index.csv")
    if os.path.isfile(index_path):
        old = pd.read_csv(index_path)
        merged = pd.concat([old, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset="bodyId", keep="last")
    else:
        merged = new_df
    merged.to_csv(index_path, index=False)
    print(f"index.csv now {len(merged)} rows (was "
          f"{len(merged) - len(new_df)} + {len(new_df)} new)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
