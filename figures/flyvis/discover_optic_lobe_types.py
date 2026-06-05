"""List every cell-type in the Janelia optic-lobe neuprint dataset,
grouped by substring patterns matching the missing flyvis types. Lets us
build a flyvis-name -> Janelia-name mapping for fetch_optic_lobe_anatomy.py.

Usage:
    python figures/flyvis/discover_optic_lobe_types.py
"""
from __future__ import annotations

import os
import sys

# Flyvis types that returned 0 neurons in the smoke test
MISSING_FLYVIS_TYPES = [
    "Am", "CT1(Lo1)", "CT1(M10)",
    "Mi11", "Mi12", "Mi3",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
    "Tm28", "TmY9",
]


def main():
    from neuprint import Client, set_default_client

    token = (os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
             or os.environ.get("NEUPRINT_TOKEN"))
    if not token:
        sys.exit("need a neuprint token via NEUPRINT_APPLICATION_CREDENTIALS "
                 "/ NEUPRINT_TOKEN env var")
    client = Client("https://neuprint.janelia.org",
                     dataset="optic-lobe:v1.1", token=token)
    set_default_client(client)

    # Pull every cell type with at least one neuron via a single Cypher query
    types_df = client.fetch_custom(
        "MATCH (n:Neuron) "
        "WHERE n.type IS NOT NULL "
        "RETURN n.type AS type, count(*) AS n "
        "ORDER BY n.type"
    )
    print(f"{len(types_df)} distinct types in optic-lobe:v1.1\n")

    # Build candidates per missing flyvis type via substring match.
    print("=== suggested mappings for missing flyvis types ===")
    for ft in MISSING_FLYVIS_TYPES:
        # Strip parenthetical for matching: 'CT1(Lo1)' -> 'CT1'
        bare = ft.split("(")[0]
        candidates = types_df[
            types_df.type.str.contains(bare, regex=False, case=False, na=False)
        ]
        if len(candidates) == 0:
            print(f"  {ft:12s} -> [none found]")
            continue
        # Print top 6 candidates
        cands_str = ", ".join(
            f"{r.type}({r.n})" for r in candidates.head(6).itertuples()
        )
        print(f"  {ft:12s} -> {cands_str}")

    print("\n=== all R-cell-like types (Janelia photoreceptor naming) ===")
    rs = types_df[types_df.type.str.startswith("R", na=False)]
    for r in rs.itertuples():
        print(f"  {r.type:25s}  n={r.n}")


if __name__ == "__main__":
    main()
