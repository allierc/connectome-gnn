"""One-off fetch of zebrafish fish2 HD-circuit connectivity (cells + edges).

Companion to fetch_zebrafish_anatomy_HD.py: that script grabs SWC skeletons +
ROI meshes for the 3-D figure; this script grabs the adjacency matrix between
the same HD-circuit cells (IPNd*/IPNds* = r1pi HD ring, RIPN* = habenula
afferents, pt-IPN* = pretectal afferents) so the connectome-constrained RNN
loader has something to read.

Run ONCE on a machine with internet access and a neuprint-fish2 token; it
writes `<out>/neurons.csv` and `<out>/connections.csv`, both consumed by
`load_zebrafish_hd_connectome()` in
`src/connectome_gnn/generators/connconstr_data.py`.

Server: neuprint-fish2.janelia.org, dataset 'fish2'.

Prereqs:
    pip install neuprint-python pandas
    export NEUPRINT_APPLICATION_CREDENTIALS=<your-token>   # or pass --token

Usage:
    python fetch_zebrafish_connectivity_HD.py [--out zebrafish_connectome_HD] \
        [--token TOKEN] [--weight_thresh 1]

Output tree:
    <out>/
        neurons.csv      bodyId, type, instance, side, somaLocation{X,Y,Z}
        connections.csv  bodyId_pre, bodyId_post, weight  (one row per directed edge)
        roiInfo.json     per-neuron ROI synapse breakdown (raw neuprint payload)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd


# Same HD-circuit cell types as fetch_zebrafish_anatomy_HD.py.
IPND_TYPES = ["IPNd", "IPNd01",
              "IPNd13A", "IPNd13B", "IPNd13C", "IPNd13D", "IPNd13E", "IPNd13S",
              "IPNd14", "IPNd15", "IPNd16",
              "IPNd17A", "IPNd17B",
              "IPNdp01"]
IPNDS_TYPES = ["IPNds", "IPNds13A", "IPNds13B", "IPNds17"]
RIPN_TYPES = ["RIPN01", "RIPN02", "RIPN03_a", "RIPN03_b",
              "RIPN05", "RIPN11",
              "RIPN12_a", "RIPN12_b", "RIPN12_c",
              "RIPN16", "RIPN17"]
PTIPN_TYPES = ["pt-IPN1", "pt-IPN2"]

HD_TYPES = IPND_TYPES + IPNDS_TYPES + RIPN_TYPES + PTIPN_TYPES


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", default="zebrafish_connectome_HD",
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
    p.add_argument("--types", nargs="+", default=HD_TYPES,
                   help="neuron types to include (default: IPNd*/IPNds*/RIPN*/pt-IPN*)")
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
    # We chunk per type rather than passing the whole list, so the
    # per-type count is logged and a single bad name doesn't kill the
    # whole fetch.
    all_neurons = []
    for t in args.types:
        nrns, _ = fetch_neurons(NC(type=t))
        print(f"{t:18s} -> {len(nrns)} neurons")
        if len(nrns) > 0:
            all_neurons.append(nrns)
    if not all_neurons:
        sys.exit("no neurons found for any of the requested types")
    nrn_df = pd.concat(all_neurons, ignore_index=True)
    print(f"total: {len(nrn_df)} HD-circuit neurons")

    body_ids = nrn_df["bodyId"].astype(int).tolist()

    # --- Fetch adjacencies restricted to the HD set ----------------------
    # `sources=body_ids, targets=body_ids` returns only edges where both
    # endpoints are in the HD circuit; the second return value is the
    # per-edge connection table we want.
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
    # Keep the columns the loader actually needs (bodyId, type, instance,
    # side, soma coords); everything else (raw roiInfo blob, synweight,
    # etc.) goes to a separate JSON dump for posterity.
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
    edge_out = conn_df[["bodyId_pre", "bodyId_post", "weight"]].copy()
    edge_out["bodyId_pre"] = edge_out["bodyId_pre"].astype(int)
    edge_out["bodyId_post"] = edge_out["bodyId_post"].astype(int)
    edge_out["weight"] = edge_out["weight"].astype(int)
    edge_out.to_csv(os.path.join(args.out, "connections.csv"), index=False)
    print(f"wrote {os.path.join(args.out, 'connections.csv')}")

    # --- Raw roiInfo for posterity --------------------------------------
    # roiInfo is a JSON string of per-ROI synapse counts. Keep it as a
    # separate dump so the loader stays trivial but downstream analyses
    # (e.g. per-ROI input current splits) can read it back if needed.
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
    # Spot-check that the matrix isn't trivially empty.
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
