"""Build the 943-cell HD/IPN connectome CSVs from the colleagues' refreshed
``IPN_sortedData_060826.pkl`` reconstruction.

This is the on-disk cache step (step 1 of "HOW TO ADD A NEW CIRCUIT" in
``generators/circuits.py``) for the ``zebrafish_HD_IPN12_943_*`` circuit
family. It restricts the full 1854-cell IPN reconstruction to the 36
``cellTypes2Cons`` cell types (943 cells) and writes the loader's CSV pair
plus a per-cell ``angle`` column (the cell's functional preferred-heading,
used for the ring ordering instead of the soma-x proxy).

Key conventions, fixed with the colleagues / the circuit owner:

  * Edge weight  = ``adjacency_matrix_size`` (synapse contact area). The
    magnitudes are preserved verbatim — the loader for this circuit applies
    sign only (no 5x inhibitory amplification, no spectral rescale), so the
    area information survives into ``J_effective``.
  * Orientation  = ``A[i, j]`` is the weight from pre-synaptic cell ``i`` to
    post-synaptic cell ``j`` (confirmed: afferents send 64 % of their output
    to the bump ring). connections.csv therefore writes
    ``bodyId_pre = NeuronIDs[i]``, ``bodyId_post = NeuronIDs[j]``.
  * Bump/readout = IPNd* / IPNds* / IPN12_a/b  PLUS the new IPN-core families
    (IPN20/26/28/29/31-36) — all join the readout ring (circuit-owner
    decision) and are Dale-flipped inhibitory.
  * Soma XYZ    = back-filled by bodyId from the existing 839-cell
    neurons.csv where available (733/943); NaN otherwise. Not needed for the
    ring order (angle-based), only for 3-D anatomy renders.

Run once::

    python figures/zebrafish/build_connectome_HD_IPN12_943_from_pickle.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

# The nominal 917-cell circuit types. cellTypes2Cons minus IPN20 and IPN26
# (dropped per the colleague's request). IPN12 and IPN-core stay separate
# partitions in Figs 1/2. Bump-ring vs afferent vs IPN-core split is by
# prefix/membership in the loader; this list only decides which cells are
# kept at all.
CELLTYPES = [
    "pt-IPN1",
    "RIPN01", "RIPN02", "RIPN03_a", "RIPN03_b", "RIPN05",
    "RIPN11", "RIPN12_a", "RIPN12_b", "RIPN12_c",
    "IPN28", "IPN29", "IPN31", "IPN32",
    "IPN33", "IPN34", "IPN35", "IPN36",
    "IPN12_a", "IPN12_b",
    "IPNd01", "IPNd13A", "IPNd13B", "IPNd13C", "IPNd13D", "IPNd13E",
    "IPNd13S", "IPNd14", "IPNd15", "IPNd16", "IPNd17A", "IPNd17B",
    "IPNds13A", "IPNds13B",
]

_HEMI2SIDE = {"Left": "L", "Right": "R", "left": "L", "right": "R"}


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    # The pickle now ships next to the CSVs it generates, so the default no
    # longer depends on the read-only GraphData bind-mount (which exists only
    # under /workspace/connectome-gnn, not in the cx worktree). The mount path
    # stays as a fallback for checkouts predating the in-repo copy.
    _pkl_local = os.path.join(here, "zebrafish_connectome_HD_IPN_917",
                              "IPN_sortedData_060826.pkl")
    _pkl_mount = ("/workspace/connectome-gnn/graphs_data/remote/zebrafish/"
                  "IPN_sortedData_060826.pkl")
    p.add_argument(
        "--pkl",
        default=_pkl_local if os.path.isfile(_pkl_local) else _pkl_mount)
    p.add_argument(
        "--soma_ref",
        default=os.path.join(here, "zebrafish_connectome_HD_IPN12",
                             "neurons.csv"),
        help="existing 839-cell neurons.csv to back-fill soma XYZ from")
    p.add_argument(
        "--out_dir",
        default=os.path.join(here, "zebrafish_connectome_HD_IPN_917"))
    args = p.parse_args()

    import pickle
    with open(args.pkl, "rb") as fh:
        d = pickle.load(fh)

    Type = np.asarray(d["Type"]).astype(str)
    Hemi = np.asarray(d["Hemi"]).astype(str)
    NID = np.asarray(d["NeuronIDs"]).astype(np.int64)
    A = np.asarray(d["adjacency_matrix_size"], dtype=np.float64)

    # Per-cell preferred-heading angle: IPN_angles is a per-type list in
    # first-occurrence type order; the global cell order is the concatenation
    # of those per-type blocks (verified), so a flat concat aligns per-cell.
    uniq = list(dict.fromkeys(Type.tolist()))
    angle = np.concatenate([np.asarray(d["IPN_angles"][uniq.index(t)],
                                       dtype=np.float64)
                            for t in uniq])
    assert angle.shape[0] == Type.shape[0], "angle/cell misalignment"

    keep = set(CELLTYPES)
    sel = np.array([t in keep for t in Type])
    idx = np.where(sel)[0]
    if idx.size != 917:
        print(f"[warn] selected {idx.size} cells (expected 917)")

    bodyId = NID[idx]
    typ = Type[idx]
    side = np.array([_HEMI2SIDE.get(h, "") for h in Hemi[idx]])
    instance = np.array([f"{t}_{s}" if s else t for t, s in zip(typ, side)])
    ang = angle[idx]

    # --- soma XYZ ---------------------------------------------------------
    # Prefer the fresh fish2 live fetch (soma_fish2.csv, full 943 coverage,
    # written by the one-off neuprint pull); fall back to back-filling by
    # bodyId from the existing 839-cell export. The pickle's own ``zb`` field
    # is unusable (misaligned, 879 entries) so it is never used for soma.
    soma = {c: np.full(idx.size, np.nan) for c in
            ("somaLocationX", "somaLocationY", "somaLocationZ")}
    soma_fetch = os.path.join(args.out_dir, "soma_fish2.csv")
    soma_src = soma_fetch if os.path.isfile(soma_fetch) else args.soma_ref
    if os.path.isfile(soma_src):
        ref = pd.read_csv(soma_src).set_index("bodyId")
        for k, b in enumerate(bodyId):
            if b in ref.index:
                r = ref.loc[b]
                for c in soma:
                    if c in ref.columns:
                        soma[c][k] = float(r[c])
        n_soma = int(np.isfinite(soma["somaLocationX"]).sum())
        print(f"[soma] filled {n_soma}/{idx.size} from {soma_src}")

    neurons = pd.DataFrame({
        "bodyId": bodyId,
        "type": typ,
        "instance": instance,
        "side": side,
        "somaLocationX": soma["somaLocationX"],
        "somaLocationY": soma["somaLocationY"],
        "somaLocationZ": soma["somaLocationZ"],
        "angle": ang,
    })

    # --- edges: restrict to the selected subgraph, A[i,j] = pre i -> post j -
    subA = A[np.ix_(idx, idx)]
    pre_i, post_j = np.nonzero(subA)
    conns = pd.DataFrame({
        "bodyId_pre": bodyId[pre_i],
        "bodyId_post": bodyId[post_j],
        "weight": subA[pre_i, post_j],
    })

    os.makedirs(args.out_dir, exist_ok=True)
    neurons.to_csv(os.path.join(args.out_dir, "neurons.csv"), index=False)
    conns.to_csv(os.path.join(args.out_dir, "connections.csv"), index=False)

    meta = {
        "source_pickle": args.pkl,
        "n_neurons": int(idx.size),
        "n_edges": int(len(conns)),
        "edge_weight": "adjacency_matrix_size (synapse contact area)",
        "orientation": "A[i,j] = pre i -> post j; csv pre=i, post=j",
        "n_types": int(len(set(typ.tolist()))),
        "cellTypes2Cons": CELLTYPES,
        "weight_note": ("magnitudes preserved verbatim; the circuit loader "
                        "applies sign only (no 5x inh amplify, no spectral "
                        "rescale)"),
        "ring_order": "per-cell IPN_angles (preferred heading), not soma-x",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[done] wrote {idx.size} neurons, {len(conns)} edges -> "
          f"{args.out_dir}")
    print(f"  weight range: min={conns['weight'].min():.0f} "
          f"median={conns['weight'].median():.0f} "
          f"max={conns['weight'].max():.0f}")


if __name__ == "__main__":
    main()
