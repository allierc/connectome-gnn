"""Null connectomes for the Drosophila CX companion (drosophila_cx_338_v1) —
the 'is the measured wiring necessary?' control, mirroring the zebrafish
``build_er_connectomes.py`` + ``build_block_shuffle_connectomes.py``.

Three null families, all keeping the node set + cell identities (types, PEN/PFN
gate targets, EPG ring order) fixed and only randomising the recurrent wiring,
so each is a drop-in swap on the existing task datasets:

  er_<i>      (i=1..4)   full-network Erdos-Renyi, size+density+|W|-matched,
                         recurrent inhibitory-fraction sweep p_inh in {.4,.6,.8,1}.
  er_ei_<i>   (i=1..10)  full-network ER matched to the MEASURED overall
                         inhibitory fraction (the necessity control proper;
                         spread across seeds = null variance).
  bs_<i>      (i=1..10)  block-preserving shuffle: per (post-type, pre-type)
                         block the edge count / |W| / Dale sign are preserved,
                         only the fine within-block wiring is randomised.

Sign is Dale-locked per PRE cell (column), as in the measured circuit
(Delta7 + ER6 inhibitory). Output:
``graphs_data/drosophila_cx/er_connectomes/{er,er_ei,bs}_<i>.npz`` carrying
``J_effective`` (338x338, row=post, col=pre) + provenance, consumed by the
``drosophila_cx_338_v1_{er,er_ei,bs}_<i>`` circuits in
``connectome_gnn.generators.circuits``.

Usage:  python figures/drosophila_cx/build_null_connectomes.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

BASE_CIRCUIT = "drosophila_cx_338_v1"
P_INH_SWEEP = (0.4, 0.6, 0.8, 1.0)
SEEDS_ER = (40260, 40261, 40262, 40263)
SEEDS_EI = tuple(range(40270, 40280))    # 10 nominal-E/I replicates
SEEDS_BS = tuple(range(40280, 40290))    # 10 block-shuffle replicates


def _cell_polarity(J, N):
    """Per pre-cell (col) Dale polarity: +1 exc, -1 inh, 0 if no out-edges."""
    pol = np.zeros(N)
    for j in range(N):
        nz = J[:, j][J[:, j] != 0]
        if nz.size:
            pol[j] = np.sign(nz.mean())
    return pol


def _build_er(J_base, N, n_edges, emp_mag, is_afferent, p_inh, seed,
              target_inh_frac=None):
    rng = np.random.default_rng(seed)
    n_off = N * N - N
    flat = rng.choice(n_off, size=n_edges, replace=False)
    rows = flat // (N - 1)
    cols = flat % (N - 1)
    cols = cols + (cols >= rows)                       # skip the diagonal
    rows, cols = rows.astype(np.int64), cols.astype(np.int64)
    mag = rng.choice(emp_mag, size=n_edges, replace=True)
    out_deg = np.bincount(cols, minlength=N)
    if target_inh_frac is not None:
        # E/I-matched: ANY cell may be inhibitory so the null can actually
        # reach the measured fraction. The fly's inhibitory pool (Delta7+ER6,
        # 46 cells) is far too small to carry ~31% of edges on its own, so the
        # zebrafish 'recurrent-only' rule would cap the null at ~13%.
        candidates = np.arange(N)
        target_inh_out = float(target_inh_frac) * n_edges
    else:
        candidates = np.nonzero(~is_afferent)[0]
        target_inh_out = p_inh * int(out_deg[candidates].sum())
    polarity = np.ones(N)
    cum = 0
    for c in rng.permutation(candidates):
        if cum >= target_inh_out:
            break
        polarity[c] = -1.0
        cum += int(out_deg[c])
    J = np.zeros((N, N), dtype=np.float32)
    J[rows, cols] = (mag * polarity[cols]).astype(np.float32)
    e = int((J > 0).sum()); i = int((J < 0).sum()); tot = e + i
    prov = {
        "base_circuit": BASE_CIRCUIT, "kind": "er", "seed": int(seed),
        "p_inh_recurrent_target": (None if target_inh_frac is not None else float(p_inh)),
        "target_inh_frac_overall": (None if target_inh_frac is None else float(target_inh_frac)),
        "n_edges": int(n_edges), "density": float(n_edges / n_off),
        "exc_edges": e, "inh_edges": i,
        "exc_frac": float(e / tot), "inh_frac": float(i / tot),
        "note": "full-338 ER; node identities inherited from base; only "
                "support/sign/magnitude randomised",
    }
    return J, prov


def _build_block_shuffle(J, part, base_pol, seed):
    rng = np.random.default_rng(int(seed))
    N = J.shape[0]
    mag = np.zeros((N, N), dtype=np.float32)
    blocks = list(dict.fromkeys(part.tolist()))
    idx = {b: np.where(part == b)[0] for b in blocks}
    for pb in blocks:
        rows = idx[pb]
        for qb in blocks:
            cols = idx[qb]
            sub = J[np.ix_(rows, cols)]
            nz = sub != 0
            n_e = int(nz.sum())
            if n_e == 0:
                continue
            mags = np.abs(sub[nz]).astype(np.float32)
            rr = np.repeat(rows, cols.size)
            cc = np.tile(cols, rows.size)
            offdiag = np.where(rr != cc)[0]
            chosen = rng.choice(offdiag, size=n_e, replace=False)
            flat = np.zeros(rows.size * cols.size, dtype=np.float32)
            flat[chosen] = rng.permutation(mags)
            mag[np.ix_(rows, cols)] = flat.reshape(rows.size, cols.size)
    J_new = (mag * base_pol[None, :].astype(np.float32)).astype(np.float32)
    e = int((J_new > 0).sum()); i = int((J_new < 0).sum())
    prov = {"base_circuit": BASE_CIRCUIT, "kind": "block_shuffle",
            "seed": int(seed), "N": N, "n_edges": int((J_new != 0).sum()),
            "exc_edges": e, "inh_edges": i, "inh_frac": float(i / (e + i)),
            "note": "per (post-type,pre-type) block: edge count/|W|/Dale "
                    "preserved, fine within-block wiring randomised"}
    return J_new, prov


def main():
    from connectome_gnn.generators.circuits import get_circuit
    from connectome_gnn.utils import (
        graphs_data_path, load_data_fallback_roots, set_data_root,
    )
    roots = load_data_fallback_roots()
    if roots:
        set_data_root(roots[0]); print(f"[data root] {roots[0]}")

    cx = get_circuit(BASE_CIRCUIT).as_loader_dict()
    J_base = np.asarray(cx["J_effective"], dtype=np.float32)
    N = int(cx["N"])
    n_edges = int((J_base != 0).sum())
    n_off = N * N - N
    emp_mag = np.abs(J_base[J_base != 0]).astype(np.float64)
    base_pol = _cell_polarity(J_base, N)
    is_afferent = base_pol > 0                         # excitatory cells
    part = np.asarray(cx["neuron_types"]).astype(int)  # cell type = block
    e0 = int((J_base > 0).sum()); i0 = int((J_base < 0).sum())
    measured_inh_frac = float(i0 / (e0 + i0))
    print(f"[base] N={N} edges={n_edges} density={n_edges/n_off:.5f} "
          f"E/I={e0}/{i0} ({100*i0/(e0+i0):.1f}% inh) "
          f"exc(afferent-like) cells={int(is_afferent.sum())} types={len(set(part.tolist()))}")

    out_dir = graphs_data_path("drosophila_cx", "er_connectomes")
    os.makedirs(out_dir, exist_ok=True)
    index = []

    for k, (p_inh, seed) in enumerate(zip(P_INH_SWEEP, SEEDS_ER), start=1):
        J, prov = _build_er(J_base, N, n_edges, emp_mag, is_afferent, p_inh, seed)
        np.savez_compressed(os.path.join(out_dir, f"er_{k}.npz"),
                            J_effective=J, provenance=json.dumps(prov))
        index.append({"file": f"er_{k}.npz", **prov})
        print(f"[er_{k}] p_inh={p_inh:.1f} inh={100*prov['inh_frac']:.1f}% "
              f"density={prov['density']:.5f}")

    for k, seed in enumerate(SEEDS_EI, start=1):
        J, prov = _build_er(J_base, N, n_edges, emp_mag, is_afferent,
                            p_inh=1.0, seed=seed, target_inh_frac=measured_inh_frac)
        np.savez_compressed(os.path.join(out_dir, f"er_ei_{k}.npz"),
                            J_effective=J, provenance=json.dumps(prov))
        index.append({"file": f"er_ei_{k}.npz", **prov})
        print(f"[er_ei_{k}] inh={100*prov['inh_frac']:.1f}% (measured "
              f"{100*measured_inh_frac:.1f}%)")

    for k, seed in enumerate(SEEDS_BS, start=1):
        J, prov = _build_block_shuffle(J_base, part, base_pol, seed)
        np.savez_compressed(os.path.join(out_dir, f"bs_{k}.npz"),
                            J_effective=J, provenance=json.dumps(prov))
        index.append({"file": f"bs_{k}.npz", **prov})
        print(f"[bs_{k}] inh={100*prov['inh_frac']:.1f}%")

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"base_circuit": BASE_CIRCUIT, "n_nodes": N,
                   "n_edges": n_edges, "measured_inh_frac": measured_inh_frac,
                   "matrices": index}, f, indent=2)
    print(f"[done] wrote {len(index)} null matrices + index.json to {out_dir}")


if __name__ == "__main__":
    main()
