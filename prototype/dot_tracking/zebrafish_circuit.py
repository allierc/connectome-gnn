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


def _side_of(hemi):
    """'l'/'r' per cell, from the pickle's Hemi column ('left'/'right')."""
    return np.array([str(h).lower()[:1] for h in hemi])


def load_oculomotor_circuit(config_path=DEFAULT_CONFIG, pkl_path=DEFAULT_PKL,
                            weights="size", eye_side=None):
    """Everything a training script needs, in `eq:circuit`/`eq:sign-lock`'s own
    layout: row = post, column = pre.

    Returns a dict:
      names        (285,) str        cell type of each row/column, sorted
                                      afferent -> recurrent -> output, and
                                      within a type left before right --
                                      the exact order Figure 1 panel (b)/(c) use
      W_mag        (285, 285) f32    the reconstruction's weight, |.| already
                                      (synapse contact area by default), scaled
                                      by `spectral_scale`; this is what a
                                      trainable S is initialised from, NOT what
                                      the model multiplies -- see W_hat
      spectral_scale float           the single global factor applied to W_mag
                                      so the SIGNED matrix has spectral radius
                                      circuit.spectral_target (1.0 if unset)
      sign_col     (285,) f32        +1/-1, the Dale sign of the COLUMN's
                                      (presynaptic) cell type, from the yaml
      support      (285, 285) bool   W_mag > 0 -- the measured synapses; a
                                      trainable S must stay zero off this mask,
                                      or "connectome-constrained" is a fiction
      afferent     (285,) bool       AF5_ipsi / AF5_contra rows
      output_idx   dict[str,int]     {"AMN": row indices, "AIN": row indices},
                                      restricted to the ONE hemisphere that
                                      drives `eye_side`'s eye -- see below
      effector_col dict[str,int]     {"AMN": MUSCLES.index("LR"), "AIN": ...}
      eye_side     str | None        which eye the readout drives
      cfg          CircuitConfig     cfg.circuit, for role/sign/effector lookup

    ONE EYE, TWO HEMISPHERES. The pool is bilateral and stays that way -- the
    contralateral INTG projections are the integrator, so the recurrence must
    keep both sides. But a single eye is not driven by both sides' motor
    neurons: its lateral rectus takes the IPSILATERAL abducens motor neurons,
    and its medial rectus takes the CONTRALATERAL abducens internuclear
    neurons (which cross to the oculomotor nucleus; the yaml collapses that
    two-synapse path into one arrow, so the crossing is carried by
    `CellTypeSpec.projection` instead). `eye_side` (default:
    `circuit.eye_side` in the yaml) therefore filters `output_idx` only.
    Leaving it None pools both hemispheres into each muscle channel, which is
    the pre-2026-09 behaviour and is anatomically wrong for a one-eye plant.
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

    # Spectral rescale. The reconstruction's weights are synapse contact areas
    # (median 897, max 24368 here) -- physical units, not network units. Used
    # raw as a tanh recurrent init they put ~91% of cells past |v| > 3 after
    # ONE step, so the recurrence saturates flat and no gradient reaches S.
    # One global scalar fixes it and preserves every relative magnitude:
    # `Jf = rho * J / max(Re lambda)`, the same form as
    # connectome_loaders.py's Dale rescale. Declared as
    # circuit.spectral_target, not hardcoded.
    spectral_scale = 1.0
    rho = cfg.circuit.spectral_target
    if rho is not None:
        max_re = float(np.linalg.eigvals(W_mag * sign_col[None, :]).real.max())
        if not np.isfinite(max_re) or max_re <= 0:
            raise ValueError(
                f"spectral_target={rho} but the signed connectome's largest "
                f"real eigenvalue is {max_re}; nothing to normalise against")
        spectral_scale = rho / max_re
        W_mag = W_mag * spectral_scale

    afferent_names = set(cfg.circuit.types_by_role("afferent"))
    output_names = set(cfg.circuit.types_by_role("output"))
    afferent = np.isin(names, list(afferent_names))
    effector_col = {s.name: MUSCLES.index(s.effector) for s in specs
                    if s.effector}

    # Output readout: one eye, so one hemisphere per output type.
    hemi_sel = hemi[sel]
    side = _side_of(hemi_sel)
    if eye_side is None:
        eye_side = cfg.circuit.eye_side
    proj_of = {s.name: s.projection for s in specs}
    output_idx = {}
    for n in output_names:
        rows = np.where(names == n)[0]
        proj = proj_of.get(n)
        if eye_side is not None and proj is not None:
            want = eye_side[:1] if proj == "ipsilateral" else \
                ("r" if eye_side == "left" else "l")
            rows = rows[side[rows] == want]
            if rows.size == 0:
                raise ValueError(
                    f"output type {n} ({proj} to the {eye_side} eye) has no "
                    f"cell in the required hemisphere -- the readout would be "
                    f"empty")
        elif eye_side is not None:
            print(f"[circuit] WARNING: output type {n} declares no "
                  f"`projection:`, so BOTH hemispheres drive its muscle")
        output_idx[n] = rows

    return dict(names=names, W_mag=W_mag.astype(np.float32),
                spectral_scale=spectral_scale, spectral_target=rho,
                sign_col=sign_col, support=W_mag > 0, afferent=afferent,
                output_idx=output_idx, effector_col=effector_col, cfg=cfg,
                hemi=hemi_sel, eye_side=eye_side)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pkl", default=DEFAULT_PKL)
    p.add_argument("--eye-side", default=None, choices=["left", "right"],
                   help="override circuit.eye_side from the yaml")
    a = p.parse_args()
    c = load_oculomotor_circuit(a.config, a.pkl, eye_side=a.eye_side)
    n = len(c["names"])
    n_e = int((c["sign_col"] > 0).sum())
    print(f"[circuit] {c['cfg'].circuit.name}: {n} cells, "
          f"{int(c['support'].sum())} edges, {n_e} excitatory / {n - n_e} inhibitory")
    _nz = c["W_mag"][c["W_mag"] > 0]
    print(f"[circuit] spectral_target={c['spectral_target']}  scale={c['spectral_scale']:.4g}"
          f"  -> |W| median {np.median(_nz):.4g} max {_nz.max():.4g}")
    print(f"[circuit] afferent (AF5): {int(c['afferent'].sum())} cells")
    print(f"[circuit] readout drives the {c['eye_side'] or 'BOTH (unfiltered)'} eye")
    side = _side_of(c["hemi"])
    for muscle_name, idx in c["output_idx"].items():
        eff = c["effector_col"].get(muscle_name)
        eff = MUSCLES[eff] if eff is not None else "(none)"
        n_all = int((c["names"] == muscle_name).sum())
        sides = "/".join(sorted(set(side[idx]))) or "-"
        print(f"[circuit] output {muscle_name}: {len(idx)} of {n_all} cells "
              f"(hemisphere {sides}) -> muscle {eff}")
    reachable = sorted(c["effector_col"].values())
    unreachable = [m for i, m in enumerate(MUSCLES) if i not in reachable]
    print(f"[circuit] muscles reachable from this pool: "
          f"{[MUSCLES[i] for i in reachable]}; NOT reachable: {unreachable}")


if __name__ == "__main__":
    main()
