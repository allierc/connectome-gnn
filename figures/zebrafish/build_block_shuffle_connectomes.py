"""Block-preserving null connectomes (third null family).

Where the full-network Erdos-Renyi nulls (build_er_connectomes.py: er_1..er_4
and er_ei_*) draw a brand-new random support over the whole 917x917 matrix and
so destroy the functional-block structure, the BLOCK-SHUFFLE null keeps the
measured block (functional-partition) architecture and only scrambles the
microscopic wiring INSIDE each block-pair.

For every (post-block, pre-block) submatrix it preserves, exactly:
  * the edge count (so the coarse block-to-block connectivity = density map is
    identical to the measured circuit),
  * the magnitude distribution (the block-pair's |W| values, reshuffled),
  * the Dale sign (per pre-synaptic cell),
and randomizes only WHICH cells within the two blocks are wired. It is the
intermediate null: same block architecture, random fine wiring. 10 independent
seeds give the null variance.

Node identities (cell types, ARTR/pt-IPN1 gate targets, dIPN ring) are inherited
from the base circuit, so these are drop-in swaps like the ER nulls.

Output: graphs_data/zebrafish/er_connectomes/bs_<i>.npz, each carrying
``J_effective`` (917x917, row=post, col=pre, sign-locked) + provenance.

Usage:
    python figures/zebrafish/build_block_shuffle_connectomes.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)  # sibling fig_2_connectome._partition_ids

BASE_CIRCUIT = "zebrafish_HD_IPN_917_artr_pt1"
SEEDS = tuple(range(20280, 20290))   # 10 independent block-shuffle replicates


def _cell_polarity_base(J: np.ndarray, N: int) -> np.ndarray:
    pol = np.zeros(N)
    for j in range(N):
        nz = J[:, j][J[:, j] != 0]
        if nz.size:
            pol[j] = np.sign(nz.mean())
    return pol


def _build_block_shuffle(J: np.ndarray, part: np.ndarray, base_pol: np.ndarray,
                         seed: int) -> np.ndarray:
    """One block-preserving shuffle: per block-pair, reseat the same number of
    edges at random off-diagonal positions, with magnitudes drawn from that
    block-pair's |W| and signs set by the (new) pre-cell polarity."""
    rng = np.random.default_rng(int(seed))
    N = J.shape[0]
    mag = np.zeros((N, N), dtype=np.float32)
    blocks = list(dict.fromkeys(part.tolist()))
    idx = {b: np.where(part == b)[0] for b in blocks}
    for pb in blocks:                                  # post-synaptic block (rows)
        rows = idx[pb]
        for qb in blocks:                              # pre-synaptic block (cols)
            cols = idx[qb]
            sub = J[np.ix_(rows, cols)]
            nz = sub != 0
            n_e = int(nz.sum())
            if n_e == 0:
                continue
            mags = np.abs(sub[nz]).astype(np.float32)
            rr = np.repeat(rows, cols.size)            # global row per sub cell
            cc = np.tile(cols, rows.size)              # global col per sub cell
            offdiag = np.where(rr != cc)[0]            # exclude the global diagonal
            chosen = rng.choice(offdiag, size=n_e, replace=False)
            flat = np.zeros(rows.size * cols.size, dtype=np.float32)
            flat[chosen] = rng.permutation(mags)
            mag[np.ix_(rows, cols)] = flat.reshape(rows.size, cols.size)
    J_new = mag * base_pol[None, :].astype(np.float32)  # Dale: sign by pre (col)
    return J_new.astype(np.float32)


def _block_density_map(J, part, blocks):
    """Per block-pair edge density (fraction of possible off-diagonal cells)."""
    idx = {b: np.where(part == b)[0] for b in blocks}
    D = np.zeros((len(blocks), len(blocks)))
    for a, pb in enumerate(blocks):
        for b, qb in enumerate(blocks):
            sub = J[np.ix_(idx[pb], idx[qb])]
            poss = sub.size - (pb == qb) * min(sub.shape)
            D[a, b] = (sub != 0).sum() / max(poss, 1)
    return D


def main():
    from connectome_gnn.generators.circuits import get_circuit
    from connectome_gnn.utils import (
        graphs_data_path, load_data_fallback_roots, set_data_root,
    )
    from fig_2_connectome import _partition_ids

    roots = load_data_fallback_roots()
    if roots:
        set_data_root(roots[0])
        print(f"[data root] {roots[0]}")

    cx = get_circuit(BASE_CIRCUIT).as_loader_dict()
    J_base = np.asarray(cx["J_effective"], dtype=np.float32)
    N = int(cx["N"])
    nt = np.asarray(cx["neuron_types"]).astype(int)
    names = list(cx["type_names"])
    part = _partition_ids(nt, names)
    base_pol = _cell_polarity_base(J_base, N)
    blocks = list(dict.fromkeys(part.tolist()))
    D_base = _block_density_map(J_base, part, blocks)

    out_dir = graphs_data_path("zebrafish", "er_connectomes")
    os.makedirs(out_dir, exist_ok=True)
    n_edges = int((J_base != 0).sum())
    print(f"[base] N={N} edges={n_edges} blocks={blocks}")

    index = []
    for k, seed in enumerate(SEEDS, start=1):
        J_new = _build_block_shuffle(J_base, part, base_pol, seed)
        D_new = _block_density_map(J_new, part, blocks)
        r_block = float(np.corrcoef(D_base.ravel(), D_new.ravel())[0, 1])
        e = int((J_new > 0).sum()); i = int((J_new < 0).sum())
        prov = dict(kind="block_shuffle", seed=int(seed), N=N,
                    n_edges=int((J_new != 0).sum()),
                    exc_edges=e, inh_edges=i, inh_frac=float(i / (e + i)),
                    block_density_corr_to_measured=r_block,
                    note="block-pair edge count/magnitude/Dale preserved; "
                         "fine within-block wiring randomised")
        out_path = os.path.join(out_dir, f"bs_{k}.npz")
        np.savez_compressed(out_path, J_effective=J_new,
                            provenance=json.dumps(prov))
        index.append({"file": f"bs_{k}.npz", **prov})
        print(f"[bs_{k}] inh={100*prov['inh_frac']:.1f}%  "
              f"block-density r-to-measured={r_block:.3f}  -> {out_path}")

    with open(os.path.join(out_dir, "index_block_shuffle.json"), "w") as f:
        json.dump({"base_circuit": BASE_CIRCUIT, "n_nodes": N,
                   "n_edges": n_edges, "blocks": blocks,
                   "matrices": index}, f, indent=2)
    print(f"[done] wrote {len(index)} block-shuffle matrices to {out_dir}")


if __name__ == "__main__":
    main()
