"""Named connectome registry — the circuit side of the
(circuit / task / IO mapping) refactor.

See ``docs/REFACTOR_zebrafish_circuit_registry.md`` for the motivation.
A ``Circuit`` is "which neurons + how they connect" — a named, sign-locked,
spectrally-rescaled adjacency template plus the cell-type metadata and
named sub-populations a downstream model needs to wire its
encoder/decoder/IO gate. It carries no task info and no IO-mapping info.

Public API::

    from connectome_gnn.generators.circuits import (
        Circuit, register_circuit, get_circuit, list_circuits,
    )

    cx = get_circuit("zebrafish_HD_731_v1")
    cx.N, cx.J_effective.shape, list(cx.subpops)


HOW TO ADD A NEW CIRCUIT
========================

A circuit is a 3-step contribution: connectome data on disk, a build
function in this file, and (optionally) a yaml that selects it by name.
The trained-checkpoint identity is pinned by ``J_effective_sha256``, so
new circuits never overwrite old ones as long as the registry name is
unique.

1. **Cache the connectome tables on disk.** Pick a stable directory
   under ``figures/<organism>/<dataset_name>/`` and produce two CSVs::

       <dataset_name>/neurons.csv      bodyId, type, instance, side,
                                       somaLocationX/Y/Z
       <dataset_name>/connections.csv  bodyId_pre, bodyId_post, weight

   Use the existing fetchers as templates::

       figures/zebrafish/fetch_zebrafish_connectivity_HD.py        # 731 cells
       figures/zebrafish/fetch_zebrafish_connectivity_HD_IPN12.py  # 837 cells

   These run once on a machine with a neuprint token and write the
   tables locally. The fetch output is intentionally untouched by the
   refactor: a circuit is a thin wrapper that points the existing
   loader at a different CSV directory.

2. **Write a build function in this file** that returns a fully
   populated ``Circuit`` and call ``register_circuit("<name>_vN",
   build)`` at the bottom (see ``_register_zebrafish_hd_731`` for the
   shortest example). Steps inside the build function::

       cx = load_zebrafish_hd_connectome("figures/<organism>/<dataset_name>")
       return Circuit(
           name="<organism>_<region>_<count>_v<N>",  # MUST match register key
           N=int(cx["N"]),
           neuron_types=np.asarray(cx["neuron_types"], dtype=np.int64),
           type_names=list(cx["type_names"]),
           J_effective=np.asarray(cx["J_effective"], dtype=np.float32),
           soma_xyz=np.asarray(cx.get("somaLocation"), dtype=np.float64),
           subpops={                            # any named index sets you need
               "bump": np.arange(cx["n_dipn"], dtype=np.int64),
               "afferent_RIPN_L": ..., ...
           },
           bump_ring_ix=np.asarray(cx["dipn_ix"], dtype=np.int64),
           dale_signs=np.asarray(cx["dale_signs"], dtype=np.float32),
           provenance={                         # free-form, human-readable
               "server": ..., "dataset": ..., "design_notes": ...,
           },
       )

   Then register::

       def _register_<organism>_<region>_<count>() -> None:
           def build() -> Circuit:
               ...
           register_circuit("<organism>_<region>_<count>_v1", build)

   Add the new ``_register_*()`` call to ``_discover_circuits`` so the
   registry is populated on first lookup.

   **Versioning rule** (§8 of the plan): if you re-derive the same
   logical pool with different Dale config / spectral target /
   filtering rule, that is a ``_v2``. Never reuse a ``_vN`` name with
   different semantics.

3. **Wire a yaml** (optional, but the usual entry point). Copy an
   existing zebrafish yaml and add::

       circuit:
         name: <organism>_<region>_<count>_v1

   When the field is set, the model class resolves the connectome via
   ``get_circuit(name)``. When it's omitted, the model falls through to
   the legacy ``load_<organism>_*_connectome(sim.connconstr_datapath)``
   path — so existing yamls remain byte-equivalent.

4. **Add a section to the relevant docs/<organism>.tex.** Document
   what the new pool contains, where its data lives, and any design
   choices that aren't already encoded in the build function (which
   types are in the bump pool, Dale sign overrides, IO-gate wiring).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# =============================================================================
# Dataclass
# =============================================================================

@dataclass
class Circuit:
    """A named connectome subset.

    Fields mirror the canonical dict returned by the per-organism loaders
    in :mod:`connectome_gnn.generators.connconstr_data`, but renamed to
    species-neutral vocabulary so the same dataclass shape can describe
    drosophila CX, zebrafish HD, larva, etc.

    The dataclass is the in-memory builder. The on-disk record of "how
    the training data was generated against this circuit" is the
    ``circuit_provenance.json`` written next to the TaskTrials zarrs.
    """

    name: str
    """Stable registry name, e.g. ``zebrafish_HD_731_v1``."""

    N: int
    """Number of neurons."""

    neuron_types: np.ndarray
    """(N,) int64 — per-neuron type id into ``type_names``."""

    type_names: list
    """Length-K list of unique cell-type name strings."""

    J_effective: np.ndarray
    """(N, N) float32 — signed, sign-locked, spectrally-rescaled
    adjacency. Layout: row=post, col=pre (matches ``r @ W_rec.T`` in the
    sign-locked RNN/GNN forward pass)."""

    soma_xyz: Optional[np.ndarray] = None
    """(N, 3) float64 — per-neuron soma position in dataset coordinates,
    or None when not available."""

    subpops: dict = field(default_factory=dict)
    """Named index sets into [0, N). Convention for HD circuits:
        ``bump``                — the dIPN / EPG ring cells (indices 0..n_bump-1)
        ``afferent_RIPN_L/R``   — left/right habenula → IPN afferents
        ``afferent_ptIPN_L/R``  — left/right pretectal afferents
    """

    bump_ring_ix: Optional[np.ndarray] = None
    """(n_bump,) int64 — for each bump cell, its ring-bin index along the
    circuit's circular axis. Used by the circular-TV regulariser and
    ring-anchored kinograph plots."""

    dale_signs: Optional[np.ndarray] = None
    """(N,) float32 — per-neuron Dale prior sign in {-1, 0, +1}. Used as
    a fallback by the ``column_dale`` wrec_param mode for orphan cells
    whose outgoing column-sum is zero in ``J_effective``."""

    body_ids: Optional[np.ndarray] = None
    """(N,) int64 — per-neuron source-DB body id (neuprint bodyId for the
    drosophila / zebrafish fetches). Used by the anatomy-voltage render
    helpers in :mod:`connectome_gnn.plot_anatomy_voltage` to find the
    matching SWC skeleton under
    ``provenance['anatomy_dir']/skeletons/<type>__<bodyId>.swc``."""

    provenance: dict = field(default_factory=dict)
    """Free-form: source server URL, dataset name, fetch date, raw type
    list, Dale-flip + spectral-rescale params, etc. ``J_effective_sha256``
    is set by ``register_circuit`` so a checkpoint can later pin the
    exact connectome content it was trained against."""

    def compute_J_sha256(self) -> str:
        """sha256 of ``J_effective.tobytes()`` — small, fast, identifies
        connectome content modulo dtype/shape."""
        arr = np.ascontiguousarray(self.J_effective, dtype=np.float32)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def as_loader_dict(self) -> dict:
        """Return this circuit in the canonical loader-output dict shape
        (``N``, ``J_effective``, ``neuron_types``, ``type_names``,
        ``n_dipn``/``n_epg``, ``dipn_ix``/``epg_ix``,
        ``afferent_subpop_ix``/``pen_subpop_ix``, ``dale_signs``, …).

        Bridge between the named-registry path and the model class's
        legacy ``cx[...]`` access pattern — lets a model consume a
        Circuit without changing its constructor body. Carries both
        fish-native keys (``n_dipn``, ``afferent_subpop_ix``) AND
        fly-vocab aliases (``n_epg``, ``pen_subpop_ix``) so the same
        dict serves both standalone fish and any future drosophila CX
        consumer.
        """
        n_bump = int(len(self.subpops.get("bump", [])))
        bump_ring = (np.asarray(self.bump_ring_ix, dtype=np.int64)
                     if self.bump_ring_ix is not None
                     else np.array([], dtype=np.int64))

        afferent = {
            "RIPN_L":  np.asarray(self.subpops.get("afferent_RIPN_L",  []), dtype=np.int64),
            "RIPN_R":  np.asarray(self.subpops.get("afferent_RIPN_R",  []), dtype=np.int64),
            "ptIPN_L": np.asarray(self.subpops.get("afferent_ptIPN_L", []), dtype=np.int64),
            "ptIPN_R": np.asarray(self.subpops.get("afferent_ptIPN_R", []), dtype=np.int64),
        }
        pen = {  # fly-vocab back-compat
            "PENa_L": afferent["RIPN_L"],
            "PENa_R": afferent["RIPN_R"],
            "PENb_L": afferent["ptIPN_L"],
            "PENb_R": afferent["ptIPN_R"],
        }

        out: dict = {
            "N": int(self.N),
            "J_effective": np.asarray(self.J_effective, dtype=np.float32),
            "neuron_types": np.asarray(self.neuron_types, dtype=np.int64),
            "type_names": list(self.type_names),
            "n_dipn": n_bump,           # fish-native primary
            "dipn_ix": bump_ring,
            "afferent_subpop_ix": afferent,
            "n_epg": n_bump,            # fly-vocab alias
            "epg_ix": bump_ring,
            "pen_subpop_ix": pen,
            # Provenance fields — used by the data generator to write
            # circuit_provenance.json next to the TaskTrials zarrs.
            "_circuit_name": self.name,
            "_circuit_sha256": self.provenance.get("J_effective_sha256", ""),
        }
        if self.dale_signs is not None:
            out["dale_signs"] = np.asarray(self.dale_signs, dtype=np.float32)
        if self.soma_xyz is not None:
            out["somaLocation"] = np.asarray(self.soma_xyz, dtype=np.float64)
        return out

    def __repr__(self) -> str:
        return (
            f"Circuit(name={self.name!r}, N={self.N}, "
            f"types={len(self.type_names)}, subpops={list(self.subpops)})"
        )


# =============================================================================
# Registry
# =============================================================================

_BUILD_FNS: "dict[str, Callable[[], Circuit]]" = {}
_CACHE: "dict[str, Circuit]" = {}


def register_circuit(name: str, build_fn: "Callable[[], Circuit]") -> None:
    """Register a circuit build function under ``name``. The build function
    is called lazily on first ``get_circuit(name)`` and the result is
    cached for subsequent calls. Re-registering the same name raises.
    """
    if name in _BUILD_FNS:
        raise ValueError(
            f"Circuit name {name!r} is already registered to "
            f"{_BUILD_FNS[name].__module__}.{_BUILD_FNS[name].__qualname__}"
        )
    _BUILD_FNS[name] = build_fn


def get_circuit(name: str) -> Circuit:
    """Look up and (on first call) build a registered circuit. Subsequent
    calls return the cached ``Circuit`` instance — the build is therefore
    safe to be expensive (e.g. reads CSV tables, eigendecomposes the
    raw adjacency for spectral rescale)."""
    if name not in _CACHE:
        _discover_circuits()
        if name not in _BUILD_FNS:
            available = sorted(_BUILD_FNS)
            raise KeyError(
                f"Unknown circuit {name!r}. Available: {available}"
            )
        circuit = _BUILD_FNS[name]()
        if circuit.name != name:
            raise ValueError(
                f"Circuit build for {name!r} returned a Circuit with "
                f"name={circuit.name!r}; the build function must set the "
                f"name attribute to match its registration key."
            )
        circuit.provenance.setdefault(
            "J_effective_sha256", circuit.compute_J_sha256(),
        )
        _CACHE[name] = circuit
    return _CACHE[name]


def list_circuits() -> "list[str]":
    """Sorted list of all registered circuit names. Triggers discovery so
    the listing is complete after a single import of this module."""
    _discover_circuits()
    return sorted(_BUILD_FNS)


def _discover_circuits() -> None:
    """Trigger registration of all built-in circuits. Idempotent: each
    builder's ``register_circuit`` raises on duplicate, so the discovery
    pass uses module-import side effects + an internal flag."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    # Each circuit registers a build function. Add new circuits here.
    _register_zebrafish_hd_731()
    _register_zebrafish_hd_ipn12_839()
    _register_drosophila_cx_156()


_DISCOVERED: bool = False


# =============================================================================
# Built-in circuits
# =============================================================================

def _register_zebrafish_hd_731() -> None:
    """Register the current 731-cell zebrafish HD pool as
    ``zebrafish_HD_731_v1``. The build function reads the cached
    neuprint-fish2 tables under ``figures/zebrafish/zebrafish_connectome_HD/``
    and wraps :func:`load_zebrafish_hd_connectome`."""

    def build() -> Circuit:
        # Imported lazily so this module stays cheap when only
        # Circuit/registry types are needed (e.g. type hints in config).
        from connectome_gnn.generators.connconstr_data import (
            load_zebrafish_hd_connectome,
        )
        # Resolve the connectome path relative to the repo root via the
        # same fallback chain the model uses (get_data_root / repo cwd).
        cx = load_zebrafish_hd_connectome(
            "figures/zebrafish/zebrafish_connectome_HD"
        )
        N = int(cx["N"])
        n_dipn = int(cx.get("n_dipn", cx["n_epg"]))
        soma = cx.get("somaLocation", None)
        soma_xyz = np.asarray(soma, dtype=np.float64) if soma is not None else None

        # Build the named sub-populations from the loader output. The
        # fish-native ``afferent_subpop_ix`` keys are preferred; the
        # legacy ``pen_subpop_ix`` mapping is a last-resort fallback.
        aff = cx.get("afferent_subpop_ix", None) or {}
        pen = cx.get("pen_subpop_ix", {}) or {}
        def _aff(k_fish: str, k_fly: str) -> np.ndarray:
            arr = aff.get(k_fish, None)
            if arr is None:
                arr = pen.get(k_fly, np.array([], dtype=np.int64))
            return np.asarray(arr, dtype=np.int64)

        subpops = {
            "bump":              np.arange(n_dipn, dtype=np.int64),
            "afferent_RIPN_L":   _aff("RIPN_L",  "PENa_L"),
            "afferent_RIPN_R":   _aff("RIPN_R",  "PENa_R"),
            "afferent_ptIPN_L":  _aff("ptIPN_L", "PENb_L"),
            "afferent_ptIPN_R":  _aff("ptIPN_R", "PENb_R"),
        }

        dipn_glom_ix = np.asarray(
            cx.get("dipn_ix", cx["epg_ix"]), dtype=np.int64,
        )

        provenance = {
            "server": "neuprint-fish2.janelia.org",
            "dataset": "fish2",
            "source_tables": "figures/zebrafish/zebrafish_connectome_HD/{neurons,connections}.csv",
            "anatomy_dir": "figures/zebrafish/zebrafish_anatomy_HD",
            "dale_inh_amplify": 5.0,
            "dale_spectral_target": 0.9,
            "type_count": len(cx["type_names"]),
            "n_bump_cells": n_dipn,
        }

        body_ids = (np.asarray(cx["bodyId"], dtype=np.int64)
                    if "bodyId" in cx else None)

        return Circuit(
            name="zebrafish_HD_731_v1",
            N=N,
            neuron_types=np.asarray(cx["neuron_types"], dtype=np.int64),
            type_names=list(cx["type_names"]),
            J_effective=np.asarray(cx["J_effective"], dtype=np.float32),
            soma_xyz=soma_xyz,
            subpops=subpops,
            bump_ring_ix=dipn_glom_ix,
            dale_signs=(np.asarray(cx["dale_signs"], dtype=np.float32)
                        if "dale_signs" in cx else None),
            body_ids=body_ids,
            provenance=provenance,
        )

    register_circuit("zebrafish_HD_731_v1", build)


def _register_zebrafish_hd_ipn12_839() -> None:
    """Register the extended 837-cell HD pool as ``zebrafish_HD_IPN12_839_v1``.

    Adds IPN12_a + IPN12_b (51 + 55 cells, exact counts depend on the
    live fish2 fetch) to the IPNd*/IPNds*/RIPN*/pt-IPN* set and feeds
    the joined neuron list through the same Dale-flip + spectral-rescale
    pipeline as :func:`load_zebrafish_hd_connectome`. IPN12 cells join
    the bump pool (per the Step-2 design choice — see
    ``docs/zebrafish.tex`` §Circuit variants), so ``n_bump`` grows from
    443 to ~549 and the bump-only decoder sees them.

    Requires the IPN12-extended CSV pair at
    ``figures/zebrafish/zebrafish_connectome_HD_IPN12/{neurons,connections}.csv``,
    produced once by ``figures/zebrafish/fetch_zebrafish_connectivity_HD_IPN12.py``.
    """

    def build() -> Circuit:
        from connectome_gnn.generators.connconstr_data import (
            load_zebrafish_hd_connectome,
        )
        datapath = "figures/zebrafish/zebrafish_connectome_HD_IPN12"
        cx = load_zebrafish_hd_connectome(datapath)

        N = int(cx["N"])
        n_bump = int(cx.get("n_dipn", cx["n_epg"]))
        soma = cx.get("somaLocation", None)
        soma_xyz = np.asarray(soma, dtype=np.float64) if soma is not None else None

        aff = cx.get("afferent_subpop_ix", None) or {}
        pen = cx.get("pen_subpop_ix", {}) or {}

        def _aff(k_fish: str, k_fly: str) -> np.ndarray:
            arr = aff.get(k_fish, None)
            if arr is None:
                arr = pen.get(k_fly, np.array([], dtype=np.int64))
            return np.asarray(arr, dtype=np.int64)

        subpops = {
            "bump":              np.arange(n_bump, dtype=np.int64),
            "afferent_RIPN_L":   _aff("RIPN_L",  "PENa_L"),
            "afferent_RIPN_R":   _aff("RIPN_R",  "PENa_R"),
            "afferent_ptIPN_L":  _aff("ptIPN_L", "PENb_L"),
            "afferent_ptIPN_R":  _aff("ptIPN_R", "PENb_R"),
        }
        bump_ring_ix = np.asarray(
            cx.get("dipn_ix", cx["epg_ix"]), dtype=np.int64,
        )

        provenance = {
            "server": "neuprint-fish2.janelia.org",
            "dataset": "fish2",
            "source_tables":
                "figures/zebrafish/zebrafish_connectome_HD_IPN12/{neurons,connections}.csv",
            # Primary anatomy_dir holds the 731-cell HD SWCs (IPNd*/IPNds*/
            # RIPN*/pt-IPN*). IPN12_a/b SWCs live alongside under a sibling
            # cache; the render helper joins both at lookup time.
            "anatomy_dir": "figures/zebrafish/zebrafish_anatomy_HD",
            "anatomy_extra_dirs": ["figures/zebrafish/zebrafish_anatomy_IPN12"],
            "dale_inh_amplify": 5.0,
            "dale_spectral_target": 0.9,
            "type_count": len(cx["type_names"]),
            "n_bump_cells": n_bump,
            "ipn12_design_note": (
                "IPN12_a + IPN12_b joined the bump ring; outgoing weights "
                "Dale-flipped to inhibitory. See docs/zebrafish.tex "
                "§Circuit variants."
            ),
        }

        body_ids = (np.asarray(cx["bodyId"], dtype=np.int64)
                    if "bodyId" in cx else None)

        return Circuit(
            name="zebrafish_HD_IPN12_839_v1",
            N=N,
            neuron_types=np.asarray(cx["neuron_types"], dtype=np.int64),
            type_names=list(cx["type_names"]),
            J_effective=np.asarray(cx["J_effective"], dtype=np.float32),
            soma_xyz=soma_xyz,
            subpops=subpops,
            bump_ring_ix=bump_ring_ix,
            dale_signs=(np.asarray(cx["dale_signs"], dtype=np.float32)
                        if "dale_signs" in cx else None),
            body_ids=body_ids,
            provenance=provenance,
        )

    register_circuit("zebrafish_HD_IPN12_839_v1", build)


def _register_drosophila_cx_156() -> None:
    """Register the 156-cell hemibrain CX as ``drosophila_cx_156_v1``.

    The drosophila CX loader (``load_drosophila_cx_connectome``) returns
    the canonical adjacency + cell-type fields but does NOT expose
    bodyIds — those live in ``<datapath>/traced-neurons.csv`` and need
    to be replayed in the same order the loader uses
    (instance-sorted, then EPG glomerular permutation applied to the
    first 46 rows). This logic is verbatim from
    ``figures/drosophila_cx/fig_cx_anatomy_3d_voltage_anim._model_index_to_bodyid``.

    Anatomy SWCs + ROI meshes live under
    ``papers/janelia_cx/anatomy/cx_anatomy_test/``.
    """
    # Glomerular permutation reordering EPG indices 0..45 into the ring
    # ordering used by the connectome loader. Same array as in
    # connconstr_data.load_drosophila_cx_connectome.
    _EPG_PERM = np.array([
        23, 24, 0, 1, 42, 43, 44, 45, 2, 3, 39, 40, 41, 4, 5, 6,
        36, 37, 38, 7, 8, 9, 33, 34, 35, 10, 11, 12,
        30, 31, 32, 13, 14, 15, 27, 28, 29, 16, 17, 18,
        25, 26, 19, 20, 21, 22,
    ], dtype=np.int64)

    def _cx_body_ids(datapath: str) -> np.ndarray:
        import os
        import pandas as pd
        neuronsall = pd.read_csv(os.path.join(datapath, "traced-neurons.csv"))
        neuronsall.sort_values(by=["instance"], ignore_index=True, inplace=True)
        types = np.array(neuronsall.type).astype(str)
        def _sub(t: str) -> np.ndarray:
            return np.nonzero([t in x for x in types])[0]
        epg, pen = _sub("EPG"), _sub("PEN")
        peg, delta7 = _sub("PEG"), _sub("Delta7")
        allcx = np.concatenate((epg, pen, delta7, peg))
        allcx[0:46] = allcx[_EPG_PERM]
        er6 = np.array(
            [i for i, t in enumerate(types) if t == "ER6"], dtype=int,
        )
        if er6.size:
            allcx = np.concatenate((allcx, er6))
        return neuronsall.bodyId.values[allcx].astype(np.int64)

    def build() -> Circuit:
        from connectome_gnn.generators.connconstr_data import (
            load_drosophila_cx_connectome,
        )
        datapath = "papers/Code_NN/Code_NN/Data/Figure5/exported-traced-adjacencies-v1.2"
        cx = load_drosophila_cx_connectome(datapath)
        N = int(cx["N"])
        n_epg = int(cx["n_epg"])
        body_ids = _cx_body_ids(datapath)
        if body_ids.shape[0] != N:
            raise RuntimeError(
                f"drosophila_cx body-id resolver returned {body_ids.shape[0]} "
                f"ids but loader expects N={N}; the EPG-permutation / cell-"
                f"type indexing in _register_drosophila_cx_156 is out of sync "
                f"with load_drosophila_cx_connectome."
            )

        pen = cx.get("pen_subpop_ix", {}) or {}
        subpops = {
            "bump":              np.arange(n_epg, dtype=np.int64),
            "afferent_PENa_L":   np.asarray(pen.get("PENa_L", []), dtype=np.int64),
            "afferent_PENa_R":   np.asarray(pen.get("PENa_R", []), dtype=np.int64),
            "afferent_PENb_L":   np.asarray(pen.get("PENb_L", []), dtype=np.int64),
            "afferent_PENb_R":   np.asarray(pen.get("PENb_R", []), dtype=np.int64),
        }
        bump_ring_ix = np.asarray(cx["epg_ix"], dtype=np.int64)

        provenance = {
            "server": "hemibrain v1.2.1",
            "dataset": "hemibrain:v1.2.1",
            "source_tables":
                "papers/Code_NN/Code_NN/Data/Figure5/exported-traced-adjacencies-v1.2",
            "anatomy_dir": "papers/janelia_cx/anatomy/cx_anatomy_test",
            "type_count": len(cx["type_names"]),
            "n_bump_cells": n_epg,
            "design_note": (
                "Hulse 2025 Model A (156 neurons: EPG + PEN + Delta7 + PEG + ER6); "
                "Delta7+ER6 columns Dale-flipped to inhibitory and "
                "spectrally rescaled to ρ=0.9."
            ),
        }

        return Circuit(
            name="drosophila_cx_156_v1",
            N=N,
            neuron_types=np.asarray(cx["neuron_types"], dtype=np.int64),
            type_names=list(cx["type_names"]),
            J_effective=np.asarray(cx["J_effective"], dtype=np.float32),
            soma_xyz=None,
            subpops=subpops,
            bump_ring_ix=bump_ring_ix,
            dale_signs=None,
            body_ids=body_ids,
            provenance=provenance,
        )

    register_circuit("drosophila_cx_156_v1", build)
