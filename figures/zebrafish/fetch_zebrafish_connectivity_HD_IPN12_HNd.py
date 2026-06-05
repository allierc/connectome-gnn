"""One-off fetch of the zebrafish fish2 HD circuit **extended with the HNd
(dorsal-habenula) afferent population**.

Companion to ``fetch_zebrafish_connectivity_HD_IPN12.py``: identical protocol and
output schema, but the type list adds ``HNd`` — the single largest unmodelled
input to the dIPN bump ring (the unrestricted partner census
``census_zebrafish_partners_HD_IPN12.py`` shows HNd alone delivers more synaptic
weight to the bump pool than all of RIPN + pt-IPN combined). The resulting CSV
pair carries the 839-cell IPN12 pool + the HNd cells, and registers as
``zebrafish_HD_IPN12_HNd_<N>_v1`` in ``connectome_gnn.generators.circuits``.

Run ONCE on a machine with internet access and a neuprint-fish2 token::

    pip install neuprint-python pandas
    export NEUPRINT_APPLICATION_CREDENTIALS=<your-token>   # or pass --token
    python fetch_zebrafish_connectivity_HD_IPN12_HNd.py \\
        [--out zebrafish_connectome_HD_IPN12_HNd] [--token TOKEN]

Server: neuprint-fish2.janelia.org, dataset 'fish2'.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the exact 33-type HD+IPN12 list so the base pool can't drift.
from fetch_zebrafish_connectivity_HD_IPN12 import HD_IPN12_TYPES  # noqa: E402

# The new afferent family. HNd appears as a single fish2 type (~214 cells in the
# census); add subtypes here if a finer split shows up on fetch day.
HND_TYPES = ["HNd"]
HD_IPN12_HND_TYPES = HD_IPN12_TYPES + HND_TYPES


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="zebrafish_connectome_HD_IPN12_HNd",
                   help="output directory")
    p.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("NEUPRINT_TOKEN"),
        help="neuprint-fish2 token (or NEUPRINT_APPLICATION_CREDENTIALS / "
             "NEUPRINT_TOKEN env var)")
    p.add_argument("--server", default="https://neuprint-fish2.janelia.org")
    p.add_argument("--dataset", default="fish2")
    p.add_argument("--types", nargs="+", default=HD_IPN12_HND_TYPES)
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
    print(f"total: {len(nrn_df)} HD+IPN12+HNd neurons")

    body_ids = nrn_df["bodyId"].astype(int).tolist()

    # --- Adjacencies restricted to the extended (closed) set -------------
    _, conn_df = fetch_adjacencies(sources=body_ids, targets=body_ids, client=client)
    if args.weight_thresh > 0:
        before = len(conn_df)
        conn_df = conn_df[conn_df["weight"] >= args.weight_thresh].reset_index(drop=True)
        print(f"edges: {before} -> {len(conn_df)} after weight>={args.weight_thresh}")
    else:
        print(f"edges: {len(conn_df)}")

    # --- Write neurons.csv ----------------------------------------------
    soma_xyz = nrn_df["somaLocation"].apply(
        lambda v: v if isinstance(v, (list, tuple)) else (None, None, None))
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
    if "roiInfo" in nrn_df.columns:
        roi_payload = {
            int(bid): (json.loads(blob) if isinstance(blob, str) else blob)
            for bid, blob in zip(nrn_df["bodyId"], nrn_df["roiInfo"])
            if isinstance(blob, str) and blob.strip()}
        with open(os.path.join(args.out, "roiInfo.json"), "w") as f:
            json.dump(roi_payload, f)

    counts = nrn_out["type"].value_counts()
    print("\nneurons per type:")
    for t, n in counts.items():
        print(f"  {t:18s} {n}")
    print(f"\nedge density: {len(edge_out)} / {len(body_ids)**2} = "
          f"{len(edge_out)/(len(body_ids)**2):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
