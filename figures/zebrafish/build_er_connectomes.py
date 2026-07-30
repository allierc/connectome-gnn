"""Generate Erdos-Renyi (ER) null connectomes for the 'is the connectome
necessary?' control on the 917-cell zebrafish HD/IPN circuit.

Each ER matrix is a full-network random rewiring matched to the measured
connectome on:
  * size                N = 917 nodes (identical node set + cell identities)
  * density             same edge count (30,851 off-diagonal edges)
  * weight scale        magnitudes drawn i.i.d. from the empirical |W| of the
                        measured connectome
  * Dale's law          sign is set per PRE-synaptic cell (one polarity per
                        cell), as in the measured circuit

and SWEPT on one knob: the recurrent-pool inhibitory fraction ``p_inh``.
The afferent cells (RIPN / pt-IPN, glutamatergic) are kept excitatory; among
the recurrent cells a fraction ``p_inh`` (by out-degree) is made inhibitory.
``p_inh = 1.0`` reproduces the measured all-inhibitory recurrent pool, so that
matrix is simultaneously the pure size/density/E-I-matched necessity control;
lower values open the 'what inhibitory fraction supports the task?' sweep.

What is NOT changed: the node identities (cell types, the ARTR / pt-IPN1
afferent-gate targets, the dIPN ring order, the soma coordinates). Only the
adjacency support, its sign assignment, and the magnitudes are randomised, so
the trained model's afferent gate and dIPN decoder stay well defined and the
ER circuits are drop-in swaps on the existing rotation task dataset.

Output: graphs_data/zebrafish/er_connectomes/er_<i>.npz, each carrying
``J_effective`` (917x917, row=post, col=pre, sign-locked) plus provenance
(seed, p_inh, achieved E/I, density). Consumed by the ER circuit builders
registered in connectome_gnn.generators.circuits.

Usage:
    python figures/zebrafish/build_er_connectomes.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

BASE_CIRCUIT = "zebrafish_HD_IPN_917_artr_pt1"
# Sweep of the recurrent-pool inhibitory fraction (by out-degree). 1.0 = the
# measured all-inhibitory recurrent pool.
P_INH_SWEEP = (0.4, 0.6, 0.8, 1.0)
# One fixed seed per matrix so the sweep is reproducible and the variants are
# independent draws (different support AND different polarity assignment).
SEEDS = (20260, 20261, 20262, 20263)
# Nominal-E/I replicates: N independent ER draws whose OVERALL inhibitory edge
# fraction matches the measured connectome (size + density + E/I matched). The
# 'is the connectome necessary?' control proper; the spread across the 5 seeds
# is the null variance. Polarity targets the measured inhibitory fraction
# directly (afferents excitatory, recurrent cells marked inhibitory by
# out-degree until the overall inhibitory edge count is reached).
EI_SEEDS = tuple(range(20270, 20280))  # 10 independent nominal-E/I replicates


def _empirical_magnitudes(J: np.ndarray) -> np.ndarray:
    return np.abs(J[J != 0]).astype(np.float64)


def _cell_polarity_base(J: np.ndarray, N: int) -> np.ndarray:
    """Per pre-synaptic cell polarity in the measured connectome:
    +1 excitatory, -1 inhibitory, 0 = no outgoing edges. col = pre."""
    pol = np.zeros(N)
    for j in range(N):
        col = J[:, j]
        nz = col[col != 0]
        if nz.size:
            pol[j] = np.sign(nz.mean())
    return pol


def _build_er(J_base: np.ndarray, N: int, n_edges: int, emp_mag: np.ndarray,
              is_afferent: np.ndarray, p_inh: float, seed: int,
              target_inh_frac: float | None = None) -> dict:
    """One ER matrix: random support + magnitudes + Dale-signed columns.

    Polarity: afferents excitatory; recurrent cells marked inhibitory (random
    order, by out-degree) until either the recurrent inhibitory fraction
    reaches ``p_inh`` (default) OR, when ``target_inh_frac`` is given, the
    OVERALL inhibitory edge fraction reaches that value (used for the
    nominal-E/I replicates)."""
    rng = np.random.default_rng(seed)

    # --- random support over the off-diagonal (row=post, col=pre) ----------
    # Sample n_edges distinct positions from the N*N - N off-diagonal cells.
    n_off = N * N - N
    flat = rng.choice(n_off, size=n_edges, replace=False)
    # Map flat off-diagonal index -> (row, col), skipping the diagonal.
    rows_full = flat // (N - 1)
    cols_full = flat % (N - 1)
    cols_full = cols_full + (cols_full >= rows_full)  # shift past the diagonal
    rows, cols = rows_full.astype(np.int64), cols_full.astype(np.int64)

    # --- magnitudes drawn i.i.d. from the empirical |W| --------------------
    mag = rng.choice(emp_mag, size=n_edges, replace=True)

    # --- per-cell polarity: afferents excitatory; recurrent ~p_inh inhib ---
    out_deg = np.bincount(cols, minlength=N)            # out-degree per pre cell
    recurrent = (~is_afferent)
    rec_cells = np.nonzero(recurrent)[0]
    rec_total_out = int(out_deg[rec_cells].sum())
    # Target cumulative inhibitory out-degree. Only recurrent cells can be
    # inhibitory, so the cumulative inhibitory out-degree == the overall
    # inhibitory edge count; targeting target_inh_frac*n_edges therefore hits
    # the overall inhibitory fraction directly.
    if target_inh_frac is not None:
        target_inh_out = float(target_inh_frac) * n_edges
    else:
        target_inh_out = p_inh * rec_total_out
    # Greedily mark recurrent cells inhibitory (random order) until their
    # cumulative out-degree reaches the target inhibitory out-degree.
    order = rng.permutation(rec_cells)
    polarity = np.ones(N)                               # default excitatory
    polarity[is_afferent] = 1.0
    cum = 0
    for c in order:
        if cum >= target_inh_out:
            break
        polarity[c] = -1.0
        cum += int(out_deg[c])
    # cells with no outgoing edges keep +1 (irrelevant; no edges carry it).

    # --- assemble signed J (sign by pre cell = column) ---------------------
    J = np.zeros((N, N), dtype=np.float32)
    J[rows, cols] = (mag * polarity[cols]).astype(np.float32)

    e = int((J > 0).sum()); i = int((J < 0).sum()); tot = e + i
    prov = {
        "base_circuit": BASE_CIRCUIT,
        "seed": int(seed),
        "p_inh_recurrent_target": (None if target_inh_frac is not None
                                   else float(p_inh)),
        "target_inh_frac_overall": (None if target_inh_frac is None
                                    else float(target_inh_frac)),
        "n_edges": int(n_edges),
        "density": float(n_edges / n_off),
        "exc_edges": e, "inh_edges": i,
        "exc_frac": float(e / tot), "inh_frac": float(i / tot),
        "recurrent_inh_out_achieved": float(cum / max(rec_total_out, 1)),
        "note": ("full-917 ER; node identities (types, gate targets, ring) "
                 "inherited from base; only support/sign/magnitude randomised"),
    }
    return {"J_effective": J, "provenance": prov}


def main():
    from connectome_gnn.generators.circuits import get_circuit
    from connectome_gnn.utils import (
        graphs_data_path,
        load_data_fallback_roots,
        set_data_root,
    )

    # Write under the cluster data root (where graphs_data/ actually lives),
    # not the repo cwd. data_paths.json -> cluster_data_dir; fall back to '.'.
    roots = load_data_fallback_roots()
    if roots:
        set_data_root(roots[0])
        print(f"[data root] {roots[0]}")

    cx = get_circuit(BASE_CIRCUIT).as_loader_dict()
    J_base = np.asarray(cx["J_effective"], dtype=np.float32)
    N = int(cx["N"])
    n_edges = int((J_base != 0).sum())
    emp_mag = _empirical_magnitudes(J_base)
    base_pol = _cell_polarity_base(J_base, N)
    is_afferent = base_pol > 0   # excitatory cells = afferents (RIPN / pt-IPN)

    e0 = int((J_base > 0).sum()); i0 = int((J_base < 0).sum())
    print(f"[base] N={N} edges={n_edges} density={n_edges/(N*N-N):.5f} "
          f"E/I = {e0}/{i0} ({100*e0/(e0+i0):.1f}% / {100*i0/(e0+i0):.1f}%) "
          f"| afferent(exc) cells = {int(is_afferent.sum())}")

    out_dir = graphs_data_path("zebrafish", "er_connectomes")
    os.makedirs(out_dir, exist_ok=True)

    measured_inh_frac = float(i0 / (e0 + i0))

    index = []
    # (1) recurrent inhibitory-fraction sweep: er_1 .. er_4
    for k, (p_inh, seed) in enumerate(zip(P_INH_SWEEP, SEEDS), start=1):
        res = _build_er(J_base, N, n_edges, emp_mag, is_afferent, p_inh, seed)
        prov = res["provenance"]
        out_path = os.path.join(out_dir, f"er_{k}.npz")
        np.savez_compressed(out_path, J_effective=res["J_effective"],
                            provenance=json.dumps(prov))
        index.append({"file": f"er_{k}.npz", **prov})
        print(f"[er_{k}] p_inh={p_inh:.1f}  E/I = "
              f"{prov['exc_edges']}/{prov['inh_edges']} "
              f"({100*prov['exc_frac']:.1f}% / {100*prov['inh_frac']:.1f}%)  "
              f"density={prov['density']:.5f}  -> {out_path}")

    # (2) nominal-E/I replicates: er_ei_1 .. er_ei_5 (matched overall
    # inhibitory fraction; independent seeds give the null variance).
    for k, seed in enumerate(EI_SEEDS, start=1):
        res = _build_er(J_base, N, n_edges, emp_mag, is_afferent,
                        p_inh=1.0, seed=seed,
                        target_inh_frac=measured_inh_frac)
        prov = res["provenance"]
        out_path = os.path.join(out_dir, f"er_ei_{k}.npz")
        np.savez_compressed(out_path, J_effective=res["J_effective"],
                            provenance=json.dumps(prov))
        index.append({"file": f"er_ei_{k}.npz", **prov})
        print(f"[er_ei_{k}] nominal-E/I  E/I = "
              f"{prov['exc_edges']}/{prov['inh_edges']} "
              f"({100*prov['exc_frac']:.1f}% / {100*prov['inh_frac']:.1f}%)  "
              f"density={prov['density']:.5f}  -> {out_path}")

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"base_circuit": BASE_CIRCUIT, "n_nodes": N,
                   "n_edges": n_edges,
                   "measured_inh_frac": measured_inh_frac,
                   "matrices": index}, f, indent=2)
    print(f"[done] wrote {len(index)} ER matrices + index.json to {out_dir}")


if __name__ == "__main__":
    main()
