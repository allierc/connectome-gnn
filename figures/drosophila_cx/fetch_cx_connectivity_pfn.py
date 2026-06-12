"""One-off fetch of hemibrain CX connectivity for the **path-integration
extended** circuit: the 156-cell heading set (EPG/EPGt/PEN_a/PEN_b/Delta7/
PEG/ER6) PLUS the FB columnar / vector populations needed for the
translation companion to the zebrafish self-motion integrator:

    afferent (forward-velocity gate):  PFNd, PFNv
    recurrent + displacement readout:  PFNa, hDeltaB, PFR (PFR_a + PFR_b)

This is the *exact mirror* of
``figures/zebrafish/fetch_zebrafish_connectivity_HD_IPN12.py`` — same
protocol, same output schema — but pointed at the hemibrain server. Run
ONCE on a machine with internet access and a neuprint token; it writes
``<out>/neurons.csv`` and ``<out>/connections.csv`` in the partition-
flexible zebrafish schema, consumed by the drosophila CX loader and
registered as ``drosophila_cx_338_v1`` in
``connectome_gnn.generators.circuits``.

Server: neuprint.janelia.org, dataset 'hemibrain:v1.2.1'.

Prereqs::

    pip install neuprint-python pandas
    export NEUPRINT_TOKEN=<your-token>            # or pass --token

Usage::

    python fetch_cx_connectivity_pfn.py \\
        [--out drosophila_cx_connectome_338] [--token TOKEN] \\
        [--weight_thresh 1]

Output tree::

    <out>/
        neurons.csv      bodyId, type, instance, side, somaLocation{X,Y,Z}
        connections.csv  bodyId_pre, bodyId_post, weight  (directed edges)
        roiInfo.json     per-neuron ROI synapse breakdown
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd


# Existing 156-cell heading circuit (hemibrain type names — note the
# parenthetical PEN instance-type tags used by hemibrain v1.2.1).
HEADING_TYPES = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)",
                 "Delta7", "PEG", "ER6"]

# Path-integration extension. Afferent forward-velocity gate:
PFN_AFFERENT_TYPES = ["PFNd", "PFNv"]
# Recurrent vector / displacement-readout populations (PFR = PFR_a + PFR_b):
VECTOR_TYPES = ["PFNa", "hDeltaB", "PFR_a", "PFR_b"]

CX_PI_TYPES = HEADING_TYPES + PFN_AFFERENT_TYPES + VECTOR_TYPES


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="drosophila_cx_connectome_338",
                   help="output directory (created under cwd)")
    p.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("NEUPRINT_TOKEN"),
        help="neuprint token (or set NEUPRINT_APPLICATION_CREDENTIALS / "
             "NEUPRINT_TOKEN env var)",
    )
    p.add_argument("--server", default="https://neuprint.janelia.org")
    p.add_argument("--dataset", default="hemibrain:v1.2.1")
    p.add_argument("--types", nargs="+", default=CX_PI_TYPES,
                   help="neuron types to include (default: heading set + "
                        "PFNd/PFNv + PFNa/hDeltaB/PFR_a/PFR_b)")
    p.add_argument("--weight_thresh", type=int, default=1,
                   help="drop edges with synapse weight strictly below this")
    args = p.parse_args()

    if not args.token:
        sys.exit("need a neuprint token via --token or "
                 "NEUPRINT_APPLICATION_CREDENTIALS / NEUPRINT_TOKEN env var")

    from neuprint import (
        Client, NeuronCriteria as NC, fetch_neurons, fetch_adjacencies,
        set_default_client,
    )

    client = Client(args.server, dataset=args.dataset, token=args.token)
    set_default_client(client)
    print(f"connected: {args.server} dataset={args.dataset}")

    os.makedirs(args.out, exist_ok=True)

    # --- Fetch neurons one type at a time --------------------------------
    all_neurons = []
    for t in args.types:
        nrns, _ = fetch_neurons(NC(type=t))
        print(f"{t:18s} -> {len(nrns)} neurons")
        if len(nrns) > 0:
            all_neurons.append(nrns)
    if not all_neurons:
        sys.exit("no neurons found for any of the requested types")
    nrn_df = pd.concat(all_neurons, ignore_index=True)
    print(f"total: {len(nrn_df)} CX path-integration neurons")

    body_ids = nrn_df["bodyId"].astype(int).tolist()

    # --- Fetch adjacencies restricted to the CX-PI set -------------------
    _, conn_df = fetch_adjacencies(
        sources=body_ids, targets=body_ids, client=client,
    )
    if args.weight_thresh > 0:
        before = len(conn_df)
        conn_df = conn_df[conn_df["weight"] >= args.weight_thresh].reset_index(drop=True)
        print(f"edges: {before} -> {len(conn_df)} after weight>={args.weight_thresh}")
    else:
        print(f"edges: {len(conn_df)}")

    # --- Write neurons.csv ----------------------------------------------
    soma_xyz = nrn_df["somaLocation"].apply(
        lambda v: v if isinstance(v, (list, tuple)) else (None, None, None)
    )
    nrn_out = pd.DataFrame({
        "bodyId": nrn_df["bodyId"].astype(int),
        "type": nrn_df["type"].fillna("").astype(str),
        "instance": nrn_df["instance"].fillna("").astype(str),
        "side": nrn_df["side"].fillna("").astype(str) if "side" in nrn_df.columns else "",
        "somaLocationX": [v[0] if v is not None else None for v in soma_xyz],
        "somaLocationY": [v[1] if v is not None else None for v in soma_xyz],
        "somaLocationZ": [v[2] if v is not None else None for v in soma_xyz],
    })
    nrn_out.to_csv(os.path.join(args.out, "neurons.csv"), index=False)
    print(f"wrote {os.path.join(args.out, 'neurons.csv')}")

    # --- Write connections.csv ------------------------------------------
    # fetch_adjacencies returns one row per (pre, post, ROI); sum over ROIs
    # to the total edge weight so connections.csv is one row per directed
    # edge (matches the hemibrain traced-total-connections schema).
    edge_out = (
        conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"]
        .sum()
    )
    edge_out["bodyId_pre"] = edge_out["bodyId_pre"].astype(int)
    edge_out["bodyId_post"] = edge_out["bodyId_post"].astype(int)
    edge_out["weight"] = edge_out["weight"].astype(int)
    edge_out.to_csv(os.path.join(args.out, "connections.csv"), index=False)
    print(f"wrote {os.path.join(args.out, 'connections.csv')} "
          f"({len(edge_out)} unique directed edges)")

    # --- Raw roiInfo for posterity --------------------------------------
    if "roiInfo" in nrn_df.columns:
        roi_payload = {
            int(bid): (json.loads(blob) if isinstance(blob, str) else blob)
            for bid, blob in zip(nrn_df["bodyId"], nrn_df["roiInfo"])
            if isinstance(blob, str) and blob.strip()
        }
        with open(os.path.join(args.out, "roiInfo.json"), "w") as f:
            json.dump(roi_payload, f)
        print(f"wrote {os.path.join(args.out, 'roiInfo.json')}")

    # --- Quick connectivity summary -------------------------------------
    counts = nrn_out["type"].value_counts()
    print()
    print("neurons per type:")
    for t, n in counts.items():
        print(f"  {t:18s} {n}")
    print()
    print(f"edge density: {len(edge_out)} / {len(body_ids)**2} = "
          f"{len(edge_out)/(len(body_ids)**2):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
