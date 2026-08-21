#!/usr/bin/env python
"""zebrafish_circuit -- load the 285-cell oculomotor pool as a trainable circuit.

    python zebrafish_circuit.py                 # sanity-print the loaded circuit

This is NOT a new circuit definition. `scripts/plot_oculomotor_connectome.py`
is: it reads the raw pickle, applies `config/zebrafish/zebrafish_om_intg_285_v1.yaml`'s
`circuit.cell_types` (which 8 of the 42 types are in the pool, their role,
their Dale sign, their L/R hemisphere), and produces the exact 285-cell,
row=post/col=pre matrix that Figure 1 of the oculomotor note plots
(`build_pair_figure`'s panel (c)). This module runs the same selection --
same pickle, same yaml, same sort order, same sign convention -- so what a
training run sees is provably the circuit the note's figure shows, not a
re-derivation of it. See that script's own docstring for why it is otherwise
"a LOOK-AT-THE-DATA script, not a circuit definition": nothing here decides
roles or signs either. Those are `zebrafish_om_intg_285_v1.yaml`'s.

WHAT WAS MISSING, AND WHAT THIS DOES NOT SOLVE. Training a circuit through the
full connectome-gnn stack (`GNN_Main.py`, `ZebrafishHdTaskRNN`, the circuit
registry) needs three things the oculomotor pool doesn't have yet: cached
connectome CSVs, a registry entry, and an AF5-in/AMN+AIN-out task binding for
that model's HD-specific input gate and readout (see
`docs/HOWTO_add_oculomotor_circuit.md`). This module is the lightweight
alternative train_eyeG.py already uses for the eye: read the pickle directly,
build (W_hat, masks) once, and train a small standalone nn.Module through the
frozen eye -- the same shape of solution CTRNNEyeG was, now fed the real
connectome instead of a free (hidden, hidden) matrix.

ONLY TWO OF SIX MUSCLES ARE REACHABLE. AMN -> LR, AIN -> MR are the pool's
only output cell types (eq:output-map has no SR/IR/SO/IO row here: those
muscles are driven by OMN, which section 1.6 of the note leaves out of the
285-cell pool). This loader returns that as fact, not as something to work
around -- the four missing muscle channels are constant zero drive into
`EyeG`, and the resulting torsion/vertical behaviour is the actual scientific
content of training this circuit, not a bug to paper over.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))               # connectome-gnn/
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from connectome_gnn.config import NeuralGraphConfig          # noqa: E402
import plot_oculomotor_connectome as POC                     # noqa: E402

DEFAULT_PKL = os.path.join(REPO, "config", "zebrafish",
                           "Oculomotor_sortedData_081126.pkl")
DEFAULT_CONFIG = os.path.join(REPO, "config", "zebrafish",
                              "zebrafish_om_intg_285_v1.yaml")
MUSCLES = ["LR", "SR", "MR", "IR", "SO", "IO"]               # eye_anatomy order


def load_oculomotor_circuit(config_path=DEFAULT_CONFIG, pkl_path=DEFAULT_PKL,
                            weights="size"):
    """Everything a training script needs, in `eq:circuit`/`eq:sign-lock`'s own
    layout: row = post, column = pre.

    Returns a dict:
      names        (285,) str        cell type of each row/column, sorted
                                      afferent -> recurrent -> output, and
                                      within a type left before right --
                                      the exact order Figure 1 panel (b)/(c) use
      W_mag        (285, 285) f32    the reconstruction's raw weight, |.| already
                                      (synapse contact area by default); this is
                                      what a trainable S is initialised from,
                                      NOT what the model multiplies -- see W_hat
      sign_col     (285,) f32        +1/-1, the Dale sign of the COLUMN's
                                      (presynaptic) cell type, from the yaml
      support      (285, 285) bool   W_mag > 0 -- the measured synapses; a
                                      trainable S must stay zero off this mask,
                                      or "connectome-constrained" is a fiction
      afferent     (285,) bool       AF5_ipsi / AF5_contra rows
      output_idx   dict[str,int]     {"AMN": row indices, "AIN": row indices}
      effector_col dict[str,int]     {"AMN": MUSCLES.index("LR"), "AIN": ...}
      cfg          CircuitConfig     cfg.circuit, for role/sign/effector lookup
    """
    d, types, hemi, body = POC.load_pickle(pkl_path)
    A = np.asarray(d[f"adjacency_matrix_{weights}"], dtype=np.float64)

    cfg = NeuralGraphConfig.from_yaml(config_path)
    specs = cfg.circuit.cell_types or []
    if not specs:
        raise ValueError(f"{config_path} declares no circuit.cell_types")
    missing = [s.name for s in specs if s.name not in set(types)]
    if missing:
        raise ValueError(f"cell_types absent from the reconstruction: {missing}")

    # Exactly build_pair_figure's selection and sort -- see that function's
    # docstring for why (afferent -> recurrent -> output, L before R).
    names_decl = [s.name for s in specs]
    role_rank = {s.name: POC._ROLE_ORDER.index(s.role or "recurrent")
                for s in specs}
    type_rank = {n: i for i, n in enumerate(names_decl)}
    sel = np.where(np.isin(types, names_decl))[0]
    key = [(role_rank[types[i]], type_rank[types[i]],
            0 if str(hemi[i]).lower().startswith("l") else 1) for i in sel]
    sel = sel[np.lexsort((
        [k[2] for k in key], [k[1] for k in key], [k[0] for k in key]))]
    W_mag = A[np.ix_(sel, sel)].T                             # (n, n), post x pre
    names = types[sel]

    sign_of = {s.name: (1.0 if (s.sign or "E") == "E" else -1.0) for s in specs}
    sign_col = np.array([sign_of[str(t)] for t in names], np.float32)

    afferent_names = set(cfg.circuit.types_by_role("afferent"))
    output_names = set(cfg.circuit.types_by_role("output"))
    afferent = np.isin(names, list(afferent_names))
    output_idx = {n: np.where(names == n)[0] for n in output_names}
    effector_col = {s.name: MUSCLES.index(s.effector) for s in specs
                    if s.effector}

    return dict(names=names, W_mag=W_mag.astype(np.float32),
                sign_col=sign_col, support=W_mag > 0, afferent=afferent,
                output_idx=output_idx, effector_col=effector_col, cfg=cfg,
                hemi=hemi[sel])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pkl", default=DEFAULT_PKL)
    a = p.parse_args()
    c = load_oculomotor_circuit(a.config, a.pkl)
    n = len(c["names"])
    n_e = int((c["sign_col"] > 0).sum())
    print(f"[circuit] {c['cfg'].circuit.name}: {n} cells, "
          f"{int(c['support'].sum())} edges, {n_e} excitatory / {n - n_e} inhibitory")
    print(f"[circuit] afferent (AF5): {int(c['afferent'].sum())} cells")
    for muscle_name, idx in c["output_idx"].items():
        eff = c["effector_col"].get(muscle_name)
        eff = MUSCLES[eff] if eff is not None else "(none)"
        print(f"[circuit] output {muscle_name}: {len(idx)} cells -> muscle {eff}")
    reachable = sorted(c["effector_col"].values())
    unreachable = [m for i, m in enumerate(MUSCLES) if i not in reachable]
    print(f"[circuit] muscles reachable from this pool: "
          f"{[MUSCLES[i] for i in reachable]}; NOT reachable: {unreachable}")


if __name__ == "__main__":
    main()
