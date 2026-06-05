"""Unrestricted upstream/downstream partner census of the zebrafish fish2
HD-circuit (839-cell IPN12 pool).

Companion to ``fetch_zebrafish_connectivity_HD_IPN12.py``. That script builds a
**closed** subgraph: it restricts ``fetch_adjacencies`` to ``sources=targets=``
the 839 in-circuit bodyIds, so every synapse to/from a cell outside the
33-type list is discarded. This script removes that restriction on the partner
side to answer the question the closed graph cannot:

  * UPSTREAM   ``fetch_adjacencies(sources=None, targets=circuit)``
                -> every cell in fish2 that synapses ONTO our circuit.
  * DOWNSTREAM  ``fetch_adjacencies(sources=circuit, targets=None)``
                -> every cell our circuit projects TO.

It then reports:
  (1) the ring's input budget -- what fraction of synapses ONTO the bump pool
      (IPNd*/IPNds*/IPN12) and onto the whole circuit comes from INSIDE the
      modelled 839 cells vs from OUTSIDE (unmodelled types);
  (2) a leaderboard of the EXTERNAL upstream partner types, ranked by total
      synapse weight onto the circuit (and onto the bump pool specifically) --
      the concrete candidate input populations we currently ignore;
  (3) the analogous EXTERNAL downstream leaderboard -- where the heading signal
      goes outside the dIPN.

This is connectivity only, not function: it names the cell types and weights,
not what they encode (that needs the ZAPBench functional match).

Run ONCE on a machine with internet access and a neuprint-fish2 token::

    pip install neuprint-python pandas
    export NEUPRINT_APPLICATION_CREDENTIALS=<your-token>   # or pass --token
    python census_zebrafish_partners_HD_IPN12.py [--out census_HD_IPN12] \\
        [--token TOKEN] [--min_weight 1]

Server: neuprint-fish2.janelia.org, dataset 'fish2' (same as the fetch script).

Output tree::

    <out>/
        census_upstream_by_type.csv     external+internal presynaptic types
        census_downstream_by_type.csv   external+internal postsynaptic types
        census_summary.json             the inside/outside synapse fractions
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

# Reuse the EXACT in-circuit type list from the fetch companion so membership
# is defined identically (no drift between the two scripts).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_zebrafish_connectivity_HD_IPN12 import HD_IPN12_TYPES  # noqa: E402

# Bump-pool (HD ring) type prefixes -- the cells whose input budget we care
# about most. Matches connconstr_data._ZHD_BUMP_PREFIXES.
BUMP_PREFIXES = ("IPNd", "IPNds", "IPN12_a", "IPN12_b")


def _is_bump(type_name: str) -> bool:
    s = str(type_name or "")
    return s.startswith(BUMP_PREFIXES)


def _edge_totals(roi_conn: pd.DataFrame) -> pd.DataFrame:
    """Collapse a (pre, post, roi, weight) roi table to total weight per edge."""
    return (roi_conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)
            ["weight"].sum())


def _summarise_side(edges: pd.DataFrame, partner_col: str, type_of: dict,
                    in_circuit: set, bump_ids: set) -> pd.DataFrame:
    """Per-partner-type weight table for one direction.

    ``partner_col`` is the column holding the *partner* bodyId (the non-circuit
    end of each edge); the other end is the circuit cell. Returns one row per
    partner type with cell counts, total weight onto the whole circuit, weight
    onto the bump pool, and an inside/outside flag.
    """
    circ_col = "bodyId_post" if partner_col == "bodyId_pre" else "bodyId_pre"
    e = edges.copy()
    e["ptype"] = e[partner_col].map(lambda b: type_of.get(int(b)) or "<untyped>")
    e["inside"] = e[partner_col].map(lambda b: int(b) in in_circuit)
    e["to_bump"] = e[circ_col].map(lambda b: int(b) in bump_ids)

    rows = []
    for (ptype, inside), g in e.groupby(["ptype", "inside"]):
        rows.append(dict(
            type=ptype,
            location="inside" if inside else "outside",
            n_partner_cells=g[partner_col].nunique(),
            total_weight=int(g["weight"].sum()),
            weight_to_bump=int(g.loc[g["to_bump"], "weight"].sum()),
            n_edges=len(g),
        ))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["location", "total_weight"],
                              ascending=[True, False]).reset_index(drop=True)
    return out


def _budget(edges: pd.DataFrame, circ_col: str, partner_col: str,
            in_circuit: set, restrict_ids: set | None = None) -> dict:
    """Inside-vs-outside synapse budget for edges touching ``restrict_ids``
    (or the whole circuit if None) on the ``circ_col`` end."""
    e = edges
    if restrict_ids is not None:
        e = e[e[circ_col].map(lambda b: int(b) in restrict_ids)]
    inside = int(e.loc[e[partner_col].map(lambda b: int(b) in in_circuit),
                       "weight"].sum())
    outside = int(e.loc[e[partner_col].map(lambda b: int(b) not in in_circuit),
                        "weight"].sum())
    total = inside + outside
    return dict(inside=inside, outside=outside, total=total,
                outside_frac=(outside / total if total else 0.0))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="census_HD_IPN12", help="output directory")
    p.add_argument(
        "--token",
        default=os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
        or os.environ.get("NEUPRINT_TOKEN"),
        help="neuprint-fish2 token (or NEUPRINT_APPLICATION_CREDENTIALS / "
             "NEUPRINT_TOKEN env var)")
    p.add_argument("--server", default="https://neuprint-fish2.janelia.org")
    p.add_argument("--dataset", default="fish2")
    p.add_argument("--min_weight", type=int, default=1,
                   help="min total synapse weight per edge to keep")
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

    # --- in-circuit membership (same protocol as the fetch companion) -----
    all_neurons = []
    for t in HD_IPN12_TYPES:
        nrns, _ = fetch_neurons(NC(type=t))
        if len(nrns) > 0:
            all_neurons.append(nrns)
    nrn_df = pd.concat(all_neurons, ignore_index=True)
    circuit_ids = set(nrn_df["bodyId"].astype(int))
    bump_ids = set(nrn_df.loc[nrn_df["type"].map(_is_bump), "bodyId"].astype(int))
    body_ids = sorted(circuit_ids)
    print(f"in-circuit: {len(circuit_ids)} cells ({len(bump_ids)} bump pool)")

    # type lookup for the circuit cells (extended below with partner types)
    type_of: dict[int, str] = {int(b): str(t) for b, t in
                               zip(nrn_df["bodyId"], nrn_df["type"])}

    # --- UPSTREAM: everything presynaptic to the circuit ------------------
    print("querying UPSTREAM (sources=None, targets=circuit) ...")
    up_neur, up_roi = fetch_adjacencies(
        sources=None, targets=body_ids, min_total_weight=args.min_weight,
        client=client)
    for b, t in zip(up_neur["bodyId"], up_neur["type"]):
        type_of.setdefault(int(b), str(t) if pd.notna(t) else None)
    up_edges = _edge_totals(up_roi)
    up_edges = up_edges[up_edges["weight"] >= args.min_weight]

    # --- DOWNSTREAM: everything postsynaptic to the circuit ---------------
    print("querying DOWNSTREAM (sources=circuit, targets=None) ...")
    dn_neur, dn_roi = fetch_adjacencies(
        sources=body_ids, targets=None, min_total_weight=args.min_weight,
        client=client)
    for b, t in zip(dn_neur["bodyId"], dn_neur["type"]):
        type_of.setdefault(int(b), str(t) if pd.notna(t) else None)
    dn_edges = _edge_totals(dn_roi)
    dn_edges = dn_edges[dn_edges["weight"] >= args.min_weight]

    # --- per-type leaderboards -------------------------------------------
    up_tbl = _summarise_side(up_edges, "bodyId_pre", type_of, circuit_ids, bump_ids)
    dn_tbl = _summarise_side(dn_edges, "bodyId_post", type_of, circuit_ids, bump_ids)
    up_tbl.to_csv(os.path.join(args.out, "census_upstream_by_type.csv"), index=False)
    dn_tbl.to_csv(os.path.join(args.out, "census_downstream_by_type.csv"), index=False)

    # --- input/output budgets --------------------------------------------
    summary = {
        "n_circuit": len(circuit_ids),
        "n_bump": len(bump_ids),
        "min_weight": args.min_weight,
        "upstream_budget_circuit": _budget(up_edges, "bodyId_post", "bodyId_pre",
                                           circuit_ids),
        "upstream_budget_bump": _budget(up_edges, "bodyId_post", "bodyId_pre",
                                        circuit_ids, restrict_ids=bump_ids),
        "downstream_budget_circuit": _budget(dn_edges, "bodyId_pre", "bodyId_post",
                                             circuit_ids),
        "downstream_budget_bump": _budget(dn_edges, "bodyId_pre", "bodyId_post",
                                          circuit_ids, restrict_ids=bump_ids),
    }
    with open(os.path.join(args.out, "census_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # --- print ------------------------------------------------------------
    def _pct(d):
        return (f"inside {d['inside']:>7d}  outside {d['outside']:>7d}  "
                f"({100*d['outside_frac']:.1f}% external of {d['total']} syn)")

    print("\n===== INPUT BUDGET (incoming synapses) =====")
    print(f"  bump pool   : {_pct(summary['upstream_budget_bump'])}")
    print(f"  whole circ. : {_pct(summary['upstream_budget_circuit'])}")
    print("\n===== OUTPUT BUDGET (outgoing synapses) =====")
    print(f"  bump pool   : {_pct(summary['downstream_budget_bump'])}")
    print(f"  whole circ. : {_pct(summary['downstream_budget_circuit'])}")

    def _top_external(tbl, weight_col, k=15):
        ext = tbl[tbl["location"] == "outside"].nlargest(k, weight_col)
        for _, r in ext.iterrows():
            print(f"  {str(r['type'])[:22]:22s}  cells={r['n_partner_cells']:4d}  "
                  f"weight={r['total_weight']:7d}  to_bump={r['weight_to_bump']:7d}")

    print("\n===== TOP EXTERNAL UPSTREAM TYPES (unmodelled inputs) =====")
    _top_external(up_tbl, "total_weight")
    print("\n===== TOP EXTERNAL DOWNSTREAM TYPES (readout targets) =====")
    _top_external(dn_tbl, "total_weight")

    print(f"\nwrote {args.out}/census_upstream_by_type.csv, "
          f"census_downstream_by_type.csv, census_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
