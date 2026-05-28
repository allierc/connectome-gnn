"""One-off fetch of FlyWire optic-lobe skeletons for the e8_flywireRF
flyvis connectome.

The flyvis parquet nodes table (`nodes.parquet`) only carries
`(index, type, u, v, role)` -- the FlyWire `pt_root_id` was stripped
during the upstream `flyrewire` export. This script joins the parquet
back against the public FlyWire optic-lobe annotation table on CAVE,
recovers a `pt_root_id` per flyvis neuron, fetches the skeleton, and
writes a tarball that the in-devcontainer renderer can read.

Run this ONCE on a machine with internet access and a CAVE token; it
produces `flywire_anatomy.tar.gz` (~hundreds of MB for 13,741 neurons).
Then drop the tarball under `papers/flywire_anatomy/` and unpack.

Prereqs:
    pip install caveclient fafbseg navis pandas pyarrow

Usage (mirrors fetch_cx_anatomy.py):
    python fetch_flywire_anatomy.py \
        --token <CAVE_JWT> \
        --nodes_parquet /path/to/e8_flywireRF/nodes.parquet \
        --out flywire_anatomy

Or, to discover the right annotation table name first:
    python fetch_flywire_anatomy.py --token <T> --list_tables

Output tree:
    flywire_anatomy/
        mapping_e8.csv                   # flyvis_index, type, u, v, pt_root_id, match
        skeletons/<type>__<idx>__<rid>.swc
        rois/<roi_name>.obj
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import time
import warnings

import pandas as pd

# Datastack to query. flywire_fafb_production is the canonical materialised
# stack as of 2024-2025; if access is denied, the public read-only stack
# 'flywire_fafb_public' should be tried.
DEFAULT_DATASTACK = "flywire_fafb_production"

# Candidate optic-lobe column-annotation tables. Codex uses different names
# across releases; we try in order and use the first one that exists.
CANDIDATE_OPTIC_TABLES = (
    "column_assignments_optic_lobe_v2",
    "column_assignments_optic_lobe",
    "optic_lobe_columns_v2",
    "optic_lobe_columns_v1",
    "matsliah_et_al_2024_columns",
    "schlegel_2024_optic_lobe",
)

# Optic-lobe ROI names (FlyWire / FAFB naming).
OPTIC_ROIS = ("ME(R)", "ME(L)", "LO(R)", "LO(L)",
              "LOP(R)", "LOP(L)", "LA(R)", "LA(L)",
              "AME(R)", "AME(L)")


def _safe(s: str) -> str:
    return (s.replace("/", "_").replace("(", "_").replace(")", "")
              .replace(" ", "_"))


def _connect(token: str, datastack: str):
    from caveclient import CAVEclient
    client = CAVEclient(datastack, auth_token=token)
    print(f"connected to CAVE datastack={datastack}, "
          f"materialization={client.materialize.version}")
    return client


def _discover_optic_table(client):
    available = client.materialize.get_tables()
    print(f"  {len(available)} annotation tables available")
    for cand in CANDIDATE_OPTIC_TABLES:
        if cand in available:
            return cand
    # Fallback: print tables that look optic-lobe-related so the user can
    # rerun with --table <name>.
    print("  no canonical optic-lobe table found. candidates:")
    for t in available:
        low = t.lower()
        if any(k in low for k in ("optic", "column", "schlegel", "matsliah",
                                    "olc", "olr", "medulla", "lobula")):
            print(f"    {t}")
    sys.exit("pass --table explicitly after picking one from the list above")


def _fetch_annotation(client, table_name: str) -> pd.DataFrame:
    print(f"  fetching annotation table '{table_name}' ...")
    df = client.materialize.query_table(table_name)
    print(f"  got {len(df)} rows; columns = {list(df.columns)}")
    return df


def _join(flyvis_df: pd.DataFrame, annot_df: pd.DataFrame) -> pd.DataFrame:
    """Match flyvis (type, u, v) -> annot pt_root_id.

    We tolerate column-name variation in the annotation table. The flyvis
    parquet uses int (u, v); the annotation table may use float / string.
    """
    # Heuristics for the type column in the annotation table
    type_cols = [c for c in annot_df.columns if c.lower()
                 in ("cell_type", "type", "neuron_type", "celltype")]
    u_cols = [c for c in annot_df.columns if c.lower()
              in ("column_u", "u", "col_u", "hex_u", "p", "p_coordinate")]
    v_cols = [c for c in annot_df.columns if c.lower()
              in ("column_v", "v", "col_v", "hex_v", "q", "q_coordinate")]
    root_cols = [c for c in annot_df.columns if c.lower()
                 in ("pt_root_id", "root_id", "rootid")]
    if not (type_cols and u_cols and v_cols and root_cols):
        sys.exit(
            f"annotation table missing required columns. found "
            f"type={type_cols} u={u_cols} v={v_cols} root={root_cols}; "
            f"available columns: {list(annot_df.columns)}"
        )
    t_col, u_col, v_col, r_col = (type_cols[0], u_cols[0],
                                    v_cols[0], root_cols[0])
    print(f"  using join keys: type={t_col} u={u_col} v={v_col} "
          f"root={r_col}")

    annot = annot_df[[t_col, u_col, v_col, r_col]].copy()
    annot.columns = ["type", "u", "v", "pt_root_id"]
    annot["u"] = annot["u"].astype(int)
    annot["v"] = annot["v"].astype(int)
    annot["pt_root_id"] = annot["pt_root_id"].astype("int64")

    merged = flyvis_df.merge(annot, on=["type", "u", "v"], how="left")
    matched = merged.pt_root_id.notna().sum()
    print(f"  matched {matched}/{len(merged)} "
          f"({100.0*matched/len(merged):.1f}%)")
    if matched < 0.95 * len(merged):
        warnings.warn(
            "low match rate (<95%) -- check annotation-table version "
            "matches the parquet's flyrewire export.")
    return merged


def _fetch_skeletons(matched_df: pd.DataFrame, out_dir: str,
                     datastack: str, token: str,
                     parallel: int = 8) -> int:
    """Bulk skeleton fetch via fafbseg. Skips already-saved files so the
    fetch can be resumed."""
    from fafbseg import flywire

    sk_dir = os.path.join(out_dir, "skeletons")
    os.makedirs(sk_dir, exist_ok=True)

    flywire.set_default_dataset(datastack)
    # CAVE token can be passed via the env var; fafbseg picks it up.
    os.environ["CAVE_TOKEN"] = token

    matched = matched_df[matched_df.pt_root_id.notna()].copy()
    print(f"  fetching {len(matched)} skeletons (parallel={parallel}) ...")
    written = 0
    t0 = time.time()
    for chunk_start in range(0, len(matched), 64):
        chunk = matched.iloc[chunk_start: chunk_start + 64]
        ids_needed = []
        meta_for_ids = []
        for _, row in chunk.iterrows():
            rid = int(row.pt_root_id)
            fname = f"{_safe(row.type)}__{int(row['index'])}__{rid}.swc"
            path = os.path.join(sk_dir, fname)
            if os.path.exists(path) and os.path.getsize(path) > 100:
                continue
            ids_needed.append(rid)
            meta_for_ids.append((path, fname))
        if not ids_needed:
            continue
        try:
            skels = flywire.get_skeletons(ids_needed, parallel=parallel,
                                            datastack=datastack)
        except Exception as e:
            print(f"    chunk {chunk_start}: {e}")
            continue
        # `skels` is a navis.NeuronList; match by id back to file path
        sk_by_id = {int(s.id): s for s in skels} if skels else {}
        for rid, (path, fname) in zip(ids_needed, meta_for_ids):
            sk = sk_by_id.get(rid)
            if sk is None:
                continue
            try:
                sk.to_swc(path)
                written += 1
            except Exception as e:
                print(f"    write fail {fname}: {e}")
        if chunk_start % 512 == 0:
            elapsed = time.time() - t0
            print(f"    progress {chunk_start + len(chunk)}/{len(matched)} "
                  f"({written} written, {elapsed:.0f}s)")
    print(f"  done: {written} skeletons in {time.time() - t0:.0f}s")
    return written


def _fetch_rois(out_dir: str, datastack: str):
    """ROI meshes for the optic lobe via fafbseg / flybrains."""
    from fafbseg import flywire
    roi_dir = os.path.join(out_dir, "rois")
    os.makedirs(roi_dir, exist_ok=True)
    for roi in OPTIC_ROIS:
        path = os.path.join(roi_dir, f"{_safe(roi)}.obj")
        if os.path.exists(path):
            continue
        try:
            vol = flywire.get_neuropil_volumes(roi)
            if vol is None:
                continue
            mesh = vol[0] if isinstance(vol, (list, tuple)) else vol
            mesh.to_obj(path) if hasattr(mesh, "to_obj") \
                else mesh.export(path)
            print(f"  wrote {path}")
        except Exception as e:
            print(f"  skip ROI {roi}: {e}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--token", default=os.environ.get("CAVE_TOKEN")
                                       or os.environ.get("CAVE_SECRET"),
                   help="CAVE / FlyWire auth token (JWT)")
    p.add_argument("--datastack", default=DEFAULT_DATASTACK)
    p.add_argument("--nodes_parquet", default=None,
                   help="path to flyvis e8_flywireRF/nodes.parquet "
                        "(default: probe the standard data tree)")
    p.add_argument("--table", default=None,
                   help="annotation table name; if unset, search the "
                        "candidate list and pick the first that exists")
    p.add_argument("--list_tables", action="store_true",
                   help="list all annotation tables and exit (no fetch)")
    p.add_argument("--out", default="flywire_anatomy")
    p.add_argument("--no_tar", action="store_true")
    p.add_argument("--parallel", type=int, default=8)
    args = p.parse_args()

    if not args.token:
        sys.exit("need a CAVE token via --token or CAVE_TOKEN env var")

    client = _connect(args.token, args.datastack)

    if args.list_tables:
        for t in sorted(client.materialize.get_tables()):
            print(t)
        return 0

    # Locate the flyvis parquet
    candidate_paths = [args.nodes_parquet] if args.nodes_parquet else [
        "data/hybrid_connectomes/e8_flywireRF/nodes.parquet",
        "../connectome-gnn-ca/data/hybrid_connectomes/e8_flywireRF/nodes.parquet",
        os.path.expanduser("~/Graph/connectome-gnn-ca/data/"
                           "hybrid_connectomes/e8_flywireRF/nodes.parquet"),
    ]
    parquet_path = next((p for p in candidate_paths
                         if p and os.path.exists(p)), None)
    if parquet_path is None:
        sys.exit("nodes.parquet not found; pass --nodes_parquet explicitly")
    flyvis_df = pd.read_parquet(parquet_path)
    print(f"loaded flyvis nodes: {len(flyvis_df)} rows from {parquet_path}")

    # Resolve annotation table
    table = args.table or _discover_optic_table(client)
    annot_df = _fetch_annotation(client, table)

    merged = _join(flyvis_df, annot_df)
    os.makedirs(args.out, exist_ok=True)
    mapping_path = os.path.join(args.out, "mapping_e8.csv")
    merged.to_csv(mapping_path, index=False)
    print(f"wrote {mapping_path}")

    # Match-rate diagnostic per type
    by_type = (merged.assign(matched=merged.pt_root_id.notna())
               .groupby("type")
               .agg(n=("index", "size"),
                    matched=("matched", "sum")))
    by_type["rate"] = (100.0 * by_type.matched / by_type.n).round(1)
    print(by_type.to_string())

    # Bulk skeleton fetch
    _fetch_skeletons(merged, args.out, args.datastack, args.token,
                      parallel=args.parallel)

    # ROI meshes
    _fetch_rois(args.out, args.datastack)

    if not args.no_tar:
        tar_path = args.out.rstrip("/") + ".tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(args.out, arcname=os.path.basename(args.out))
        print(f"wrote {tar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
