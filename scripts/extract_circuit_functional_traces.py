"""Pull functional (calcium ΔF/F) activity traces for the neurons of a
registered zebrafish EM circuit, via the fishFuncEM bodyId↔zapbenchId link.

Pipeline (see papers/fishFuncEM/notebooks/04_functional_analysis.py):

  circuit bodyId --(neuprint-fish2)--> zapbenchId --(column)--> data_z (ΔF/F)

  1. Read the circuit's neurons.csv (bodyId / type / side / soma).
  2. Query neuprint-fish2 (get_custom_neuron_list) for each bodyId's
     zapbenchId — only functionally *matched* neurons have one.
  3. Report match coverage per cell type and save a mapping CSV.
  4. (--traces) Open the public zapbench ΔF/F zarr on GCS, slice out the
     matched neurons' columns, z-score per neuron, and save an .npz with the
     traces + per-neuron metadata + the stimulus/onset frames needed for
     task slicing (FishFunctional.get_task_data downstream).
  5. (--plot) Quick ΔF/F trace panel grouped by type/side.

Requirements (already satisfied in the neural-graph-linux env):
  * NEURON_TOKEN in papers/fishFuncEM/.env (auto-loaded by NeuprintServer).
  * internet to neuprint-fish2.janelia.org and storage.googleapis.com.

Run with the env that has fishfuncem + tensorstore installed, e.g.
  /workspace/.conda_envs/neural-graph-linux/bin/python \
      scripts/extract_circuit_functional_traces.py --traces --plot
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
_FISHFUNCEM = os.path.join(_REPO_ROOT, "papers", "fishFuncEM")
# fishfuncem is pip-installed in the env, but add the checkout too so the
# packaged data/ dir (onsets, stim_info) resolves and .env is found.
if _FISHFUNCEM not in sys.path:
    sys.path.insert(0, _FISHFUNCEM)

# Ensure the token is loaded even if CWD differs (NeuprintServer also calls
# load_dotenv on its own _PROJECT_ROOT/.env at import, but be explicit).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_FISHFUNCEM, ".env"))
except Exception:
    pass

# Public zapbench ΔF/F store (segmentation-filtered, clipped) — from
# notebooks/04_functional_analysis.py. Shape (T_full, N); columns are
# 0-based zapbenchId. The trailing 9 frames are trimmed to match behaviour.
ZARR_URL = (
    "gs://ahrens-connectomics-eda3e36871e30f4c0b73531c3a0f6a90/"
    "alex_data/traces/20220404_emf3/xm-97797086.ckpt-332.240317/"
    "segmentation_filtered/df_over_f_clipped/"
)
TRACE_TRIM_TAIL = 9


def _read_circuit_neurons(connectome_dir: str) -> pd.DataFrame:
    path = os.path.join(connectome_dir, "neurons.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"neurons.csv not found: {path}")
    df = pd.read_csv(path)
    if "bodyId" not in df.columns:
        raise ValueError(f"{path} has no bodyId column (cols={list(df.columns)})")
    keep = [c for c in ("bodyId", "type", "instance", "side") if c in df.columns]
    return df[keep].copy()


def map_bodyids_to_zapbench(circuit_df: pd.DataFrame) -> pd.DataFrame:
    """Query neuprint-fish2 and attach a 0-based zapbenchId (NaN if the
    neuron has no functional match) to each circuit neuron."""
    from fishfuncem import NeuprintServer  # noqa: WPS433

    body_ids = [int(b) for b in circuit_df["bodyId"].tolist()]
    server = NeuprintServer()
    meta = server.get_custom_neuron_list(body_ids, dir_postfix="_circuit")
    # corrections() already converted zapbenchId to 0-based and populated the
    # L/R `side` from soma_LR_classification. The connectome neurons.csv leaves
    # `side` empty, so pull zapbenchId AND side from the neuprint result.
    take = ["bodyId", "zapbenchId"] + (["side"] if "side" in meta.columns else [])
    zb = meta[take].copy()
    circuit_df = circuit_df.drop(
        columns=[c for c in ("side",) if c in circuit_df.columns])
    out = circuit_df.merge(zb, on="bodyId", how="left")
    out["matched"] = ~out["zapbenchId"].isna()
    return out


def report_coverage(mapped: pd.DataFrame) -> None:
    n_total = len(mapped)
    n_match = int(mapped["matched"].sum())
    print(f"\n=== functional coverage: {n_match}/{n_total} circuit neurons "
          f"have a zapbenchId ({100.0 * n_match / max(1, n_total):.1f}%) ===")
    if "type" in mapped.columns:
        grp = (mapped.groupby("type")["matched"]
               .agg(["sum", "count"]).astype(int)
               .sort_values("sum", ascending=False))
        print(f"{'type':<12} {'matched':>8} {'total':>6}")
        for t, row in grp.iterrows():
            print(f"{str(t):<12} {row['sum']:>8} {row['count']:>6}")


def load_traces_for(mapped: pd.DataFrame):
    """Return (traces, meta) for the matched neurons.

    traces : (T, n_matched) z-scored ΔF/F, columns aligned with meta rows.
    meta   : DataFrame subset (bodyId/type/side/zapbenchId) of matched rows.
    """
    import tensorstore as ts
    from scipy.stats import zscore

    matched = mapped[mapped["matched"]].copy()
    matched["zapbenchId"] = matched["zapbenchId"].astype(int)
    cols = matched["zapbenchId"].to_numpy()

    print(f"[traces] opening zapbench store ({len(cols)} columns) ...")
    ds = ts.open({"driver": "zarr3", "kvstore": ZARR_URL}).result()
    n_cols_total = ds.shape[1]
    in_range = cols < n_cols_total
    if not in_range.all():
        bad = int((~in_range).sum())
        print(f"[traces] WARNING: {bad} zapbenchId >= N={n_cols_total} "
              f"(store/mapping mismatch); dropping them")
        matched = matched[in_range]
        cols = cols[in_range]

    # Outer-index just the matched columns (cheap) instead of reading the
    # full (T x N) matrix (~2 GB). Order is preserved to align with `matched`.
    try:
        raw = ds.oindex[:, cols].read().result()
    except Exception as exc:  # pragma: no cover - API fallback
        print(f"[traces] oindex failed ({exc}); reading full matrix then slicing")
        raw = ds.read().result()[:, cols]

    raw = raw[:-TRACE_TRIM_TAIL]            # trim tail to match behaviour length
    traces = zscore(raw, axis=0)           # z-score each neuron over time
    return traces, matched.reset_index(drop=True)


def plot_traces(traces, meta, out_png: str, max_neurons: int = 24) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(max_neurons, traces.shape[1])
    order = np.argsort(meta["type"].astype(str).values)[:n]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * n)))
    T = traces.shape[0]
    for k, j in enumerate(order):
        tr = traces[:, j]
        ax.plot(np.arange(T), tr / (np.abs(tr).max() + 1e-9) * 0.45 + k,
                lw=0.5, color="black")
        ax.text(-0.01 * T, k,
                f"{meta['type'].iloc[j]} {meta.get('side', pd.Series(['']*len(meta))).iloc[j]} "
                f"zb{int(meta['zapbenchId'].iloc[j])}",
                ha="right", va="center", fontsize=6, family="monospace")
    ax.set_xlim(-0.12 * T, T)
    ax.set_yticks([])
    ax.set_xlabel("frame")
    ax.set_title(f"zebrafish circuit ΔF/F (z-scored) — {n} matched neurons")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[plot] wrote {out_png}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--connectome",
        default=os.path.join(_REPO_ROOT, "figures", "zebrafish",
                             "zebrafish_connectome_HD_IPN12"),
        help="directory containing neurons.csv (default: HD_IPN12 connectome)")
    p.add_argument("--out", default=None,
                   help="output dir (default: <connectome>/functional)")
    p.add_argument("--traces", action="store_true",
                   help="also pull the ΔF/F traces for matched neurons")
    p.add_argument("--plot", action="store_true",
                   help="also write an example trace panel (implies --traces)")
    args = p.parse_args()

    out_dir = args.out or os.path.join(args.connectome, "functional")
    os.makedirs(out_dir, exist_ok=True)

    circuit_df = _read_circuit_neurons(args.connectome)
    print(f"[circuit] {len(circuit_df)} neurons from "
          f"{os.path.basename(args.connectome)}")

    mapped = map_bodyids_to_zapbench(circuit_df)
    report_coverage(mapped)

    map_csv = os.path.join(out_dir, "bodyid_zapbench_map.csv")
    mapped.to_csv(map_csv, index=False)
    print(f"[map] wrote {map_csv}")

    if args.traces or args.plot:
        if int(mapped["matched"].sum()) == 0:
            print("[traces] no matched neurons — nothing to pull.")
            return
        traces, meta = load_traces_for(mapped)
        npz = os.path.join(out_dir, "circuit_functional_traces.npz")
        np.savez_compressed(
            npz,
            traces=traces.astype(np.float32),          # (T, n_matched), z-scored
            bodyId=meta["bodyId"].to_numpy(),
            zapbenchId=meta["zapbenchId"].to_numpy().astype(int),
            type=meta["type"].astype(str).to_numpy() if "type" in meta else np.array([]),
            side=meta["side"].astype(str).to_numpy() if "side" in meta else np.array([]),
        )
        print(f"[traces] saved {traces.shape[0]}x{traces.shape[1]} (T x neurons) "
              f"-> {npz}")
        if args.plot:
            plot_traces(traces, meta,
                        os.path.join(out_dir, "circuit_functional_traces.png"))


if __name__ == "__main__":
    main()


# /workspace/.conda_envs/neural-graph-linux/bin/python \
#     scripts/extract_circuit_functional_traces.py            # coverage only
# /workspace/.conda_envs/neural-graph-linux/bin/python \
#     scripts/extract_circuit_functional_traces.py --traces --plot
