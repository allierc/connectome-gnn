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

    cx = get_circuit("zebrafish_HD_IPN12_839_v1")
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

       figures/zebrafish/fetch_zebrafish_connectivity_HD_IPN12.py      # 839-cell HD pool
       figures/zebrafish/fetch_zebrafish_connectivity_HD_IPN12_HNd.py  # + HNd afferents

   These run once on a machine with a neuprint token and write the
   tables locally. The fetch output is intentionally untouched by the
   refactor: a circuit is a thin wrapper that points the existing
   loader at a different CSV directory.

2. **Write a build function in this file** that returns a fully
   populated ``Circuit`` and call ``register_circuit("<name>_vN",
   build)`` at the bottom (see ``_register_zebrafish_hd_ipn12_839`` for
   the reference example). Steps inside the build function::

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
               "afferent_RIPN_L": ..., ...      # encoder side: W_in injects here
               "readout": ..., ...              # decoder side: W_out reads here
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

5. **Regenerate the calcium-observation mapping** *(only if the new
   circuit changes the neuron set / ordering and you train with the
   ZAPBench obs loss).* Re-run::

       python -m connectome_gnn.generators.make_calcium_dataset \
           --circuit <new_name> \
           --connectome figures/<organism>/<dataset_name> \
           --out graphs_data/.../<new_calcium_dataset>

   so ``calcium_mapping.pt`` re-derives ``model_index`` (the EM
   bodyId -> model-neuron-index hop) into the NEW index space. The map
   is bodyId-keyed so it re-derives automatically, but it MUST be
   re-run: a mapping built for a different circuit will silently gather
   the wrong model neurons. Point ``training.calcium_dataset`` at the
   new dataset.


═══════════════════════════════════════════════════════════════════════
THE CRITICAL FORK: same afferent taxonomy vs a NEW afferent type
═══════════════════════════════════════════════════════════════════════

Steps 1–5 are COMPLETE only if the new circuit keeps the existing
afferent taxonomy — the four L/R subpopulations RIPN_L/R, ptIPN_L/R
(habenula + pretectum). That covers the common cases:

    PROCEDURE A (same taxonomy) — just steps 1–5.
      • a re-fetch of the same cell families,
      • IPN12 Dale-sign variants (v1 inhibitory / v2 excitatory),
      • per-type ablation/lesion circuits,
      • spectral-target or Dale-amplify retunes.
    These only change WHICH cells / WHAT signs are in the pool; the
    input ports are unchanged, so nothing downstream needs touching.

If the new circuit introduces a NEW afferent / input population — e.g.
adding HNd (dorsal habenula), the single largest unmodelled input to
the bump ring, or the rhombomere-2/3 commissural velocity candidate —
then steps 1–5 are NOT sufficient. The afferent input path is
hard-bound to exactly those four subpops at three coupled seams, and a
new type is dropped or rejected at each:

    PROCEDURE B (new afferent type) = Procedure A + un-hardcode the
    afferent taxonomy at all of:

      (i)   LOADER — connectome_loaders._ZHD_AFFERENT_PREFIXES is the
            fixed pair ("RIPN", "pt-IPN"); _zhd_category() RETURNS ""
            for any unknown prefix, so a CSV row of type HNd is
            rejected, not loaded. Add the new prefix to the category
            map and to the afferent-subpop construction (the RIPN->PENa
            / pt-IPN->PENb mapping that fills afferent_subpop_ix).

      (ii)  BRIDGE — Circuit.to_dict() hardwires exactly the four keys
            RIPN_L/R, ptIPN_L/R into afferent_subpop_ix and DROPS any
            other declared subpop. Make it pass through every afferent
            subpop the circuit declares.

      (iii) MODEL GATE — models/zebrafish_hd_task_{rnn,gnn}.py:
            velocity_gate='pen_4scalar' builds exactly four scalars and
            RAISES if RIPN_L/R, ptIPN_L/R are not all present. Replace
            with a gate over the N afferent subpops the circuit
            declares (the per-(channel × subpop) gate matrix of the
            extended multi-modal input model in docs/zebrafish.tex).

      (iv)  n_input — the model pins self.n_input = 3 and IGNORES the
            config `n_input`. A multi-channel input vector (visual +
            self-motion + cue) needs this read from config / dataset
            meta, not hardcoded.

      (v)   READOUT (decoder-side mirror of the afferent gate) — the
            decoder reads the first n_dipn cells by ORDERING convention:
            output_from_dipn_only (config) + the positional slice
            r[..., :n_dipn] in models/zebrafish_hd_task_{rnn,gnn}.py.
            That is a contiguous prefix, NOT a declared index set, so a
            circuit whose readout cells are a different / non-contiguous
            set cannot be expressed. The clean parallel to the afferent
            list: declare a ``readout`` (decoder) subpop and have W_out
            gather it (index_select) instead of slicing [:n_dipn]. Same
            widen-the-consumer refactor, encoder mirror image.

    The seam to widen is the same in all of these: make the consumers
    read the circuit's DECLARED subpops (Circuit.subpops is already a
    free-form dict) — afferent_* on the encoder side, ``readout`` on the
    decoder side — instead of the hardcoded RIPN/ptIPN four and the
    [:n_dipn] readout slice. That generalization is exactly what the
    extended-input model needs anyway, so Procedure B and the
    extended-model refactor are one job.

    Worked example (HNd): the unrestricted partner census
    (figures/zebrafish/census_zebrafish_partners_HD_IPN12.py) shows HNd
    alone delivers MORE synaptic weight to the bump pool than all of
    RIPN+pt-IPN combined — i.e. the present circuit captures under half
    the ring's real input. Adding it is the motivating case for B.
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
    in :mod:`connectome_gnn.generators.connectome_loaders`, but renamed to
    species-neutral vocabulary so the same dataclass shape can describe
    drosophila CX, zebrafish HD, larva, etc.

    The dataclass is the in-memory builder. The on-disk record of "how
    the training data was generated against this circuit" is the
    ``circuit_provenance.json`` written next to the TaskTrials zarrs.
    """

    name: str
    """Stable registry name, e.g. ``zebrafish_HD_IPN12_839_v1``."""

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
            # Coarse aggregates — v1 circuit + `pen_4scalar` gate read these.
            "RIPN_L":  np.asarray(self.subpops.get("afferent_RIPN_L",  []), dtype=np.int64),
            "RIPN_R":  np.asarray(self.subpops.get("afferent_RIPN_R",  []), dtype=np.int64),
            "ptIPN_L": np.asarray(self.subpops.get("afferent_ptIPN_L", []), dtype=np.int64),
            "ptIPN_R": np.asarray(self.subpops.get("afferent_ptIPN_R", []), dtype=np.int64),
            # Refined gate targets — v2 circuit + `pen_artr_ptipn1` gate read
            # these (ARTR drives ω, pt-IPN1 drives v_fwd). Empty arrays when
            # the registered circuit didn't populate the subpop (the v1
            # circuit registers only the coarse keys above), keeping the
            # dict's shape uniform across registrations.
            "ARTR_L":    np.asarray(self.subpops.get("afferent_ARTR_L",    []), dtype=np.int64),
            "ARTR_R":    np.asarray(self.subpops.get("afferent_ARTR_R",    []), dtype=np.int64),
            "pt_IPN1_L": np.asarray(self.subpops.get("afferent_pt_IPN1_L", []), dtype=np.int64),
            "pt_IPN1_R": np.asarray(self.subpops.get("afferent_pt_IPN1_R", []), dtype=np.int64),
            # Proprioceptive / efference-copy afferent — motor-efferent
            # RIPN cells (RIPN11 + RIPN12_a + RIPN12_c). Used by the
            # proprioception circuit + its accompanying 6-scalar gate.
            "motor_efferent_L": np.asarray(
                self.subpops.get("afferent_motor_efferent_L", []), dtype=np.int64),
            "motor_efferent_R": np.asarray(
                self.subpops.get("afferent_motor_efferent_R", []), dtype=np.int64),
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
    _register_zebrafish_hd_ipn12_839()
    _register_zebrafish_hd_ipn12_839_artr_pt1()
    _register_zebrafish_hd_ipn12_839_artr_pt1_proprioception()
    _register_zebrafish_hd_ipn_917()
    _register_zebrafish_hd_ipn12_hnd()
    _register_zebrafish_hd_ipn12_exc_839()
    _register_zebrafish_hd_ipn12_ablations()
    _register_drosophila_cx_156()
    _register_drosophila_cx_338()


_DISCOVERED: bool = False


# =============================================================================
# Built-in circuits
# =============================================================================

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
        from connectome_gnn.generators.connectome_loaders import (
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


def _register_zebrafish_hd_ipn12_839_artr_pt1() -> None:
    """Register the 839-cell HD pool with the **refined afferent partition**
    as ``zebrafish_HD_IPN12_839_artr_pt1``.

    Same connectome as v1 (839 cells, same Dale-flip + spectral rescale,
    same neurons.csv / connections.csv files at
    ``figures/zebrafish/zebrafish_connectome_HD_IPN12``). The only
    difference is the afferent taxonomy:

      v1 (`pen_4scalar` gate)             v2 (`pen_artr_ptipn1` gate)
      ──────────────────────────────────  ──────────────────────────────────
      RIPN_L/R  = ALL RIPN cells by side  ARTR_L/R    = RIPN01+02+03_a+03_b
                  (lumps ARTR, motor                   only — the cells
                  efferents, lateral-line                annotated as the
                  receivers, ...)                       angular-velocity
                                                        source (Petrucco
                                                        et al. 2023, Dunn
                                                        et al. 2016)
      ptIPN_L/R = pt-IPN1 + pt-IPN2       pt_IPN1_L/R = pt-IPN1 only — the
                  (lumps pretectal +                   "processed optic /
                  thalamic)                            water flow input to
                                                        IPN" cells. Drops
                                                        pt-IPN2 (thalamic,
                                                        unrelated to optic
                                                        flow).

    With the v2 partition, ω is routed exclusively to ARTR and v_fwd is
    routed exclusively to pt-IPN1 — separating the rotation and
    translation drive paths, which the v1 `pen_4scalar` gate conflated by
    pushing ω through both RIPN and pt-IPN aggregates. See
    ``docs/zebrafish.tex`` §Circuit variants for the rationale.

    Cell-type annotations are from Liangyu/Xiao/Shin-ya's literature-
    referenced inventory (RIPN01-03 = ARTR; pt-IPN1 = optic/water flow).
    """

    def build() -> Circuit:
        from connectome_gnn.generators.connectome_loaders import (
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
            "bump":                np.arange(n_bump, dtype=np.int64),
            # v2 publishes ONLY the refined subpops — the coarse RIPN/ptIPN
            # aggregates are not advertised here, so a config that wires
            # this circuit to `velocity_gate=pen_4scalar` fails loudly
            # rather than silently mixing taxonomies.
            "afferent_ARTR_L":     _aff("ARTR_L",    ""),
            "afferent_ARTR_R":     _aff("ARTR_R",    ""),
            "afferent_pt_IPN1_L":  _aff("pt_IPN1_L", ""),
            "afferent_pt_IPN1_R":  _aff("pt_IPN1_R", ""),
        }
        bump_ring_ix = np.asarray(
            cx.get("dipn_ix", cx["epg_ix"]), dtype=np.int64,
        )

        # Loud guard — v2 makes no sense if the refined subpops are empty
        # (would mean the connectome lacks the curated cell-type tags).
        for _k in ("afferent_ARTR_L", "afferent_ARTR_R",
                   "afferent_pt_IPN1_L", "afferent_pt_IPN1_R"):
            if subpops[_k].size == 0:
                raise ValueError(
                    f"zebrafish_HD_IPN12_839_artr_pt1: subpop {_k!r} is empty; "
                    f"the connectome at {datapath!r} appears to lack the "
                    f"required type tags (RIPN01/02/03_a/03_b for ARTR, "
                    f"pt-IPN1 for translation)."
                )

        provenance = {
            "server": "neuprint-fish2.janelia.org",
            "dataset": "fish2",
            "source_tables":
                "figures/zebrafish/zebrafish_connectome_HD_IPN12/{neurons,connections}.csv",
            "anatomy_dir": "figures/zebrafish/zebrafish_anatomy_HD",
            "anatomy_extra_dirs": ["figures/zebrafish/zebrafish_anatomy_IPN12"],
            "dale_inh_amplify": 5.0,
            "dale_spectral_target": 0.9,
            "type_count": len(cx["type_names"]),
            "n_bump_cells": n_bump,
            "afferent_partition_note": (
                "v2 splits the afferent gate by functional role: ARTR "
                "(RIPN01+02+03_a+03_b) drives ω; pt-IPN1 drives v_fwd. "
                "Cell-type tags from the literature-referenced inventory "
                "by Liangyu/Xiao/Shin-ya. Same connectome as v1, refined "
                "afferent taxonomy only. Use with velocity_gate="
                "'pen_artr_ptipn1'."
            ),
        }

        body_ids = (np.asarray(cx["bodyId"], dtype=np.int64)
                    if "bodyId" in cx else None)

        return Circuit(
            name="zebrafish_HD_IPN12_839_artr_pt1",
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

    register_circuit("zebrafish_HD_IPN12_839_artr_pt1", build)


def _register_zebrafish_hd_ipn12_839_artr_pt1_proprioception() -> None:
    """Register the 839-cell HD pool with the **proprioception-extended
    afferent partition** as
    ``zebrafish_HD_IPN12_839_artr_pt1_proprioception``.

    Identical connectome to v1 / artr_pt1 (same 839 cells, same Dale flip,
    same spectral rescale). What changes is the advertised afferent
    taxonomy — on top of the artr_pt1 split (ARTR for ω, pt-IPN1 for
    exteroceptive v_fwd), this variant exposes a THIRD afferent subpop:

      motor_efferent  =  RIPN11 + RIPN12_a + RIPN12_c   (37 L / 31 R cells)

    These are the cells the fish2 inventory tags as ascending / descending
    motor efferent — they carry the swim motor command (or its efference
    copy) back into IPN and so are the candidate proprioceptive afferent
    for forward-translation velocity.

    The companion 6-scalar gate routes:
      ω      → ARTR_L/R                 (v_artr_l/r — same as artr_pt1)
      v_fwd  → pt_IPN1_L/R              (v_pt1_l/r  — exteroceptive copy)
      v_fwd  → motor_efferent_L/R       (v_me_l/r   — proprioceptive copy)

    The same v_fwd signal feeds both translation branches in this first
    version — exteroceptive and proprioceptive afferents share a clean
    drive. A follow-up can add per-channel delay / gain noise to model
    the latency difference between optic-flow (delayed) and efference-copy
    (prompt). Cell-type annotations from the literature-referenced
    inventory by Liangyu/Xiao/Shin-ya.
    """

    def build() -> Circuit:
        from connectome_gnn.generators.connectome_loaders import (
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
            "bump":                       np.arange(n_bump, dtype=np.int64),
            "afferent_ARTR_L":            _aff("ARTR_L",            ""),
            "afferent_ARTR_R":            _aff("ARTR_R",            ""),
            "afferent_pt_IPN1_L":         _aff("pt_IPN1_L",         ""),
            "afferent_pt_IPN1_R":         _aff("pt_IPN1_R",         ""),
            "afferent_motor_efferent_L":  _aff("motor_efferent_L",  ""),
            "afferent_motor_efferent_R":  _aff("motor_efferent_R",  ""),
        }
        bump_ring_ix = np.asarray(
            cx.get("dipn_ix", cx["epg_ix"]), dtype=np.int64,
        )

        # Loud guard — proprioception variant makes no sense if any of
        # the six refined subpops are empty.
        for _k in ("afferent_ARTR_L", "afferent_ARTR_R",
                   "afferent_pt_IPN1_L", "afferent_pt_IPN1_R",
                   "afferent_motor_efferent_L", "afferent_motor_efferent_R"):
            if subpops[_k].size == 0:
                raise ValueError(
                    f"zebrafish_HD_IPN12_839_artr_pt1_proprioception: subpop "
                    f"{_k!r} is empty; the connectome at {datapath!r} appears "
                    f"to lack the required type tags (RIPN01/02/03_a/03_b for "
                    f"ARTR, pt-IPN1 for exteroceptive translation, "
                    f"RIPN11/RIPN12_a/RIPN12_c for motor efferent)."
                )

        provenance = {
            "server": "neuprint-fish2.janelia.org",
            "dataset": "fish2",
            "source_tables":
                "figures/zebrafish/zebrafish_connectome_HD_IPN12/{neurons,connections}.csv",
            "anatomy_dir": "figures/zebrafish/zebrafish_anatomy_HD",
            "anatomy_extra_dirs": ["figures/zebrafish/zebrafish_anatomy_IPN12"],
            "dale_inh_amplify": 5.0,
            "dale_spectral_target": 0.9,
            "type_count": len(cx["type_names"]),
            "n_bump_cells": n_bump,
            "afferent_partition_note": (
                "Proprioception-extended afferent partition: ω → ARTR "
                "(RIPN01+02+03_a+03_b), exteroceptive v_fwd → pt-IPN1, "
                "proprioceptive v_fwd → motor_efferent (RIPN11+12_a+12_c). "
                "Same connectome as v1 / artr_pt1; the three-way split "
                "exposes the canonical sensory + efference-copy redundancy "
                "real fish use for forward-translation velocity. Use with "
                "velocity_gate='pen_artr_ptipn1_propriocep'."
            ),
        }

        body_ids = (np.asarray(cx["bodyId"], dtype=np.int64)
                    if "bodyId" in cx else None)

        return Circuit(
            name="zebrafish_HD_IPN12_839_artr_pt1_proprioception",
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

    register_circuit(
        "zebrafish_HD_IPN12_839_artr_pt1_proprioception", build)


def _register_zebrafish_hd_ipn12_hnd() -> None:
    """Register the 839-cell IPN12 pool **extended with the HNd (dorsal
    habenula) afferent** as ``zebrafish_HD_IPN12_HNd_1062_v1``.

    HNd is the single largest unmodelled input to the dIPN bump ring (the
    partner census ``figures/zebrafish/census_zebrafish_partners_HD_IPN12.py``
    shows it delivers more synaptic weight to the bump pool than all of
    RIPN + pt-IPN combined). It is added as a third afferent family, kept
    excitatory (not Dale-flipped), and declared as the ``afferent_HNd_L/R``
    subpops alongside RIPN / pt-IPN. NOTE: the fish2 reconstruction labels
    all 223 HNd cells left-sided (``HNd_L``), reflecting the zebrafish
    habenula's L/R asymmetry — so ``afferent_HNd_R`` is empty.

    Requires the HNd-extended CSV pair at
    ``figures/zebrafish/zebrafish_connectome_HD_IPN12_HNd/{neurons,connections}.csv``,
    produced once by
    ``figures/zebrafish/fetch_zebrafish_connectivity_HD_IPN12_HNd.py``.

    Using it to actually DRIVE the model through HNd needs the afferent-gate
    generalisation (Procedure B in docs/HOWTO_add_zebrafish_circuit.md); this
    registration only makes the circuit loadable.
    """

    def build() -> Circuit:
        from connectome_gnn.generators.connectome_loaders import (
            load_zebrafish_hd_connectome,
        )
        datapath = "figures/zebrafish/zebrafish_connectome_HD_IPN12_HNd"
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
            "afferent_HNd_L":    _aff("HNd_L",   "HNd_L"),
            "afferent_HNd_R":    _aff("HNd_R",   "HNd_R"),
        }
        bump_ring_ix = np.asarray(
            cx.get("dipn_ix", cx["epg_ix"]), dtype=np.int64,
        )

        provenance = {
            "server": "neuprint-fish2.janelia.org",
            "dataset": "fish2",
            "source_tables":
                "figures/zebrafish/zebrafish_connectome_HD_IPN12_HNd/{neurons,connections}.csv",
            "anatomy_dir": "figures/zebrafish/zebrafish_anatomy_HD",
            "anatomy_extra_dirs": ["figures/zebrafish/zebrafish_anatomy_IPN12"],
            "dale_inh_amplify": 5.0,
            "dale_spectral_target": 0.9,
            "type_count": len(cx["type_names"]),
            "n_bump_cells": n_bump,
            "hnd_design_note": (
                "839-cell IPN12 pool + 223 HNd (dorsal habenula) afferents "
                "(all left-sided in fish2). HNd kept excitatory. Largest "
                "unmodelled bump input per the partner census. See "
                "docs/HOWTO_add_zebrafish_circuit.md (Procedure B)."
            ),
        }

        body_ids = (np.asarray(cx["bodyId"], dtype=np.int64)
                    if "bodyId" in cx else None)

        return Circuit(
            name="zebrafish_HD_IPN12_HNd_1062_v1",
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

    register_circuit("zebrafish_HD_IPN12_HNd_1062_v1", build)


def _register_zebrafish_hd_ipn12_exc_839() -> None:
    """Register the 839-cell HD pool with IPN12 treated as **excitatory**
    (alternative biological hypothesis) as ``zebrafish_HD_IPN12_839_v2``.

    Identical to v1 except the Dale-flip is restricted to the bump
    cells of the IPNd / IPNds block; IPN12_a and IPN12_b outgoing
    weights stay positive. Loader is the same; only the
    ``inh_prefixes`` argument differs. Per §8 of the refactor plan,
    re-deriving the same neuron pool with different Dale config gets a
    new version tag (``_v2``) so v1 runs stay reproducible from the
    original tables.
    """

    def build() -> Circuit:
        from connectome_gnn.generators.connectome_loaders import (
            load_zebrafish_hd_connectome,
        )
        datapath = "figures/zebrafish/zebrafish_connectome_HD_IPN12"
        cx = load_zebrafish_hd_connectome(
            datapath,
            inh_prefixes=("IPNd", "IPNds"),   # ← IPN12_a / _b EXCLUDED
        )

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
            "anatomy_dir": "figures/zebrafish/zebrafish_anatomy_HD",
            "anatomy_extra_dirs": ["figures/zebrafish/zebrafish_anatomy_IPN12"],
            "dale_inh_amplify": 5.0,
            "dale_spectral_target": 0.9,
            "dale_inh_prefixes": ["IPNd", "IPNds"],   # IPN12 omitted
            "type_count": len(cx["type_names"]),
            "n_bump_cells": n_bump,
            "ipn12_design_note": (
                "IPN12_a + IPN12_b joined the bump ring; outgoing "
                "weights are LEFT POSITIVE (excitatory) — alternative "
                "to v1's Dale-flipped inhibitory treatment. Used to "
                "ablate the Dale-sign dependency of the IPN12 "
                "alternation. See docs/zebrafish.tex §Functional "
                "comparison."
            ),
        }

        body_ids = (np.asarray(cx["bodyId"], dtype=np.int64)
                    if "bodyId" in cx else None)

        return Circuit(
            name="zebrafish_HD_IPN12_839_v2",
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

    register_circuit("zebrafish_HD_IPN12_839_v2", build)


# Refreshed 917-cell reconstruction (colleagues' IPN_sortedData_060826.pkl,
# converted by figures/zebrafish/build_connectome_HD_IPN12_943_from_pickle.py).
_ZHD_917_DATAPATH = "figures/zebrafish/zebrafish_connectome_HD_IPN_917"


def _build_zebrafish_hd_ipn_917(name: str, gate: str) -> "Circuit":
    """Shared builder for the 917-cell ``zebrafish_HD_IPN_917_*`` family.

    Same neuron pool / connectome for all three; only the advertised afferent
    taxonomy (``gate``) differs, mirroring the 839 family:

      ``pen_4scalar``    → coarse RIPN_L/R + ptIPN_L/R
      ``artr_pt1``       → ARTR_L/R + pt_IPN1_L/R
      ``proprioception`` → ARTR + pt_IPN1 + motor_efferent (L/R)

    Differences from the 839 circuit (all per the circuit owner):
      * 917 cells (34 fish2 types): the 839 HD pool + the IPN-core families
        IPN28/29/31-36 (184 cells; IPN20/26 dropped per the colleague's
        request), which JOIN the readout bump ring (n_bump ≈ 700) and are
        Dale-flipped inhibitory.
      * Edge weight is synapse contact area (adjacency_matrix_size). No 5×
        inhibitory amplification (``inh_amplify=1.0``) — so the relative E/I
        area magnitudes are kept — but the matrix IS spectrally normalised
        (``spectral_target=0.9``, a single global scalar that preserves all
        relative magnitudes while bringing J to a trainable scale).
      * Ring order is the per-cell functional preferred-heading angle
        (``angle`` column), not the soma-x mediolateral proxy.
    """
    from connectome_gnn.generators.connectome_loaders import (
        load_zebrafish_hd_connectome,
    )
    cx = load_zebrafish_hd_connectome(
        _ZHD_917_DATAPATH, inh_amplify=1.0, spectral_target=0.9)

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

    subpops = {"bump": np.arange(n_bump, dtype=np.int64)}
    if gate == "pen_4scalar":
        subpops.update({
            "afferent_RIPN_L":  _aff("RIPN_L",  "PENa_L"),
            "afferent_RIPN_R":  _aff("RIPN_R",  "PENa_R"),
            "afferent_ptIPN_L": _aff("ptIPN_L", "PENb_L"),
            "afferent_ptIPN_R": _aff("ptIPN_R", "PENb_R"),
        })
        required: tuple = ()
        gate_note = "velocity_gate='pen_4scalar' (coarse RIPN / pt-IPN)."
    elif gate == "artr_pt1":
        subpops.update({
            "afferent_ARTR_L":    _aff("ARTR_L",    ""),
            "afferent_ARTR_R":    _aff("ARTR_R",    ""),
            "afferent_pt_IPN1_L": _aff("pt_IPN1_L", ""),
            "afferent_pt_IPN1_R": _aff("pt_IPN1_R", ""),
        })
        required = ("afferent_ARTR_L", "afferent_ARTR_R",
                    "afferent_pt_IPN1_L", "afferent_pt_IPN1_R")
        gate_note = ("velocity_gate='pen_artr_ptipn1' (ω→ARTR, "
                     "v_fwd→pt-IPN1).")
    elif gate == "proprioception":
        subpops.update({
            "afferent_ARTR_L":           _aff("ARTR_L",           ""),
            "afferent_ARTR_R":           _aff("ARTR_R",           ""),
            "afferent_pt_IPN1_L":        _aff("pt_IPN1_L",        ""),
            "afferent_pt_IPN1_R":        _aff("pt_IPN1_R",        ""),
            "afferent_motor_efferent_L": _aff("motor_efferent_L", ""),
            "afferent_motor_efferent_R": _aff("motor_efferent_R", ""),
        })
        required = ("afferent_ARTR_L", "afferent_ARTR_R",
                    "afferent_pt_IPN1_L", "afferent_pt_IPN1_R",
                    "afferent_motor_efferent_L", "afferent_motor_efferent_R")
        gate_note = ("velocity_gate='pen_artr_ptipn1_propriocep' "
                     "(ω→ARTR, v_fwd→pt-IPN1 + motor_efferent).")
    else:
        raise ValueError(f"unknown gate {gate!r} for {name!r}")

    for _k in required:
        if subpops[_k].size == 0:
            raise ValueError(
                f"{name}: subpop {_k!r} is empty; the connectome at "
                f"{_ZHD_917_DATAPATH!r} appears to lack the required type tags."
            )

    bump_ring_ix = np.asarray(cx.get("dipn_ix", cx["epg_ix"]), dtype=np.int64)
    body_ids = (np.asarray(cx["bodyId"], dtype=np.int64)
                if "bodyId" in cx else None)

    provenance = {
        "server": "neuprint-fish2.janelia.org",
        "dataset": "fish2",
        "source_pickle": "IPN_sortedData_060826.pkl",
        "source_tables":
            f"{_ZHD_917_DATAPATH}/{{neurons,connections}}.csv",
        "converter":
            "figures/zebrafish/build_connectome_HD_IPN12_943_from_pickle.py",
        "anatomy_dir": "figures/zebrafish/zebrafish_anatomy_HD",
        "anatomy_extra_dirs": ["figures/zebrafish/zebrafish_anatomy_IPN12"],
        "edge_weight": "synapse contact area (adjacency_matrix_size)",
        "dale_inh_amplify": 1.0,          # relative E/I area magnitudes kept
        "dale_spectral_target": 0.9,      # global spectral rescale to ρ=0.9
        "ring_order": "per-cell IPN_angles (functional preferred heading)",
        "type_count": len(cx["type_names"]),
        "n_bump_cells": n_bump,
        "design_note": (
            "Refreshed 917-cell IPN reconstruction (34 fish2 types). The "
            "IPN-core families IPN28/29/31-36 (184 cells; IPN20/26 dropped) "
            "join the readout bump ring and are Dale-flipped inhibitory. "
            "Synapse-area "
            "edge weights with no 5× inhibitory amplify (relative E/I "
            "magnitudes kept), spectrally normalised to ρ=0.9. " + gate_note
        ),
    }

    return Circuit(
        name=name,
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


def _register_zebrafish_hd_ipn_917() -> None:
    """Register the 917-cell refreshed-reconstruction circuit family."""
    for _nm, _gate in (
        ("zebrafish_HD_IPN_917_v1",                      "pen_4scalar"),
        ("zebrafish_HD_IPN_917_artr_pt1",                "artr_pt1"),
        ("zebrafish_HD_IPN_917_artr_pt1_proprioception", "proprioception"),
    ):
        register_circuit(
            _nm,
            (lambda nm=_nm, gate=_gate:
                _build_zebrafish_hd_ipn_917(nm, gate)),
        )


# The 33 cell types present in the HD_IPN12 connectome (20 bump-pool types —
# IPNd*/IPNds* + IPN12_a/b — plus 13 afferent types — RIPN*/pt-IPN*). Kept as
# an explicit literal so the ablation variants can be registered (which needs
# the names up front, at discovery time) without eagerly building the base
# circuit; ``_register_zebrafish_hd_ipn12_ablations`` validates each name
# against the live ``type_names`` at build time so a connectome change can't
# silently register a stale type.
_IPN12_ABLATION_TYPES = (
    "IPNd", "IPNd01", "IPNd13A", "IPNd13B", "IPNd13C", "IPNd13D", "IPNd13E",
    "IPNd13S", "IPNd14", "IPNd15", "IPNd16", "IPNd17A", "IPNd17B", "IPNdp01",
    "IPNds", "IPNds13A", "IPNds13B", "IPNds17", "IPN12_a", "IPN12_b",
    "RIPN01", "RIPN02", "RIPN03_a", "RIPN03_b", "RIPN05", "RIPN11",
    "RIPN12_a", "RIPN12_b", "RIPN12_c", "RIPN16", "RIPN17",
    "pt-IPN1", "pt-IPN2",
)


def ablation_type_token(type_name: str) -> str:
    """Filesystem / registry-name-safe token for a cell type. Only ``-`` ->
    ``_`` is needed (``pt-IPN1`` -> ``pt_IPN1``); every other type name is
    already alphanumeric/underscore. Shared with the ablation runner so the
    circuit name, config filename and log dir all agree on the token."""
    return type_name.replace("-", "_")


def _register_zebrafish_hd_ipn12_ablations() -> None:
    """Register one connectivity-lesion variant per cell type of
    ``zebrafish_HD_IPN12_839_v1`` as
    ``zebrafish_HD_IPN12_839_v1_ablate_<token>``.

    Each variant is identical to v1 except the ablated type's rows AND
    columns in ``J_effective`` are zeroed — i.e. all incoming (row=post) and
    outgoing (col=pre) recurrent edges of that type are removed. The neurons
    keep their node identity: ``N=839`` is unchanged, as are the neuron
    ordering, ``subpops``, bump ring and decoder dims, so every ablation is a
    drop-in swap on the shared task dataset (no regeneration). Because the
    RNN gates the trainable recurrent weight by ``W_con_mask = (W_con != 0)``
    (see ``zebrafish_hd_task_rnn.py``), the zeroed edges stay off through
    training — a true functional knockout rather than a soft prior.

    Built by lesioning a copy of the cached v1 ``Circuit`` so the
    non-ablated content is byte-identical to v1 (no duplicated assembly
    logic). Consumed by
    ``scripts/run_GNN_zebrafish_hd_si_ipn12_ablation.py``.
    """
    base_name = "zebrafish_HD_IPN12_839_v1"
    for _type_name in _IPN12_ABLATION_TYPES:
        cname = f"{base_name}_ablate_{ablation_type_token(_type_name)}"

        def build(_type_name=_type_name, cname=cname) -> Circuit:
            base = get_circuit(base_name)
            type_names = list(base.type_names)
            if _type_name not in type_names:
                raise KeyError(
                    f"ablation type {_type_name!r} not in {base_name} "
                    f"type_names {type_names}"
                )
            tix = type_names.index(_type_name)
            neuron_types = np.asarray(base.neuron_types, dtype=np.int64)
            mask = neuron_types == tix
            n_ablated = int(mask.sum())

            J = np.asarray(base.J_effective, dtype=np.float32).copy()
            J[mask, :] = 0.0   # remove incoming edges (row = post)
            J[:, mask] = 0.0   # remove outgoing edges (col = pre)

            prov = dict(base.provenance)
            prov.pop("J_effective_sha256", None)  # recomputed for the lesioned J
            prov.update({
                "ablation_mode": "connectivity_lesion_keepN",
                "ablated_type": _type_name,
                "n_ablated_cells": n_ablated,
                "base_circuit": base_name,
            })

            def _cp(a):
                return None if a is None else np.array(a, copy=True)

            return Circuit(
                name=cname,
                N=int(base.N),
                neuron_types=neuron_types.copy(),
                type_names=type_names,
                J_effective=J,
                soma_xyz=_cp(base.soma_xyz),
                subpops={k: np.array(v, copy=True)
                         for k, v in base.subpops.items()},
                bump_ring_ix=_cp(base.bump_ring_ix),
                dale_signs=_cp(base.dale_signs),
                body_ids=_cp(base.body_ids),
                provenance=prov,
            )

        register_circuit(cname, build)


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
    # connectome_loaders.load_drosophila_cx_connectome.
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
        from connectome_gnn.generators.connectome_loaders import (
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


def _register_drosophila_cx_338() -> None:
    """Register the path-integration-extended hemibrain CX as
    ``drosophila_cx_338_v1`` — the drosophila companion to the zebrafish
    self-motion integrator.

    Strict superset of ``drosophila_cx_156_v1``: the 156-cell heading core
    (EPG/EPGt/PEN/Delta7/PEG/ER6) keeps its exact identity and ordering, and
    the FB columnar / vector populations are appended (182 cells, N=338):

        afferent (forward-velocity gate):  PFNd (40), PFNv (20)
        recurrent + displacement readout:  PFNa (58), hDeltaB (19),
                                           PFR_a (29), PFR_b (16)

    Data fetched once from neuprint (hemibrain:v1.2.1) into the partition-
    flexible zebrafish CSV schema by
    ``figures/drosophila_cx/fetch_cx_connectivity_pfn.py``; SWC skeletons for
    the new families by ``fetch_cx_skeletons_pfn.py``.

    Declares the afferent gate targets (PEN + PFN, L/R) and the decoder
    readout sets (heading<-EPG, distance<-PFNa, position<-hDeltaB/PFR) as
    free-form subpops. Like the zebrafish ``..._HNd_1062_v1`` registration,
    this only makes the circuit LOADABLE; actually DRIVING the model through
    the new forward-velocity gate and the displacement readouts needs the
    model-side generalisation (the drosophila analogue of Procedure B in
    docs/HOWTO_add_zebrafish_circuit.md: n_input>3, a PFN velocity gate, and
    declared multi-readout decoding beyond the [:46] EPG slice).
    """

    def build() -> Circuit:
        from connectome_gnn.generators.connectome_loaders import (
            load_drosophila_cx_pi_connectome,
        )
        datapath = "figures/drosophila_cx/drosophila_cx_connectome_338"
        cx = load_drosophila_cx_pi_connectome(datapath)
        N = int(cx["N"])
        n_epg = int(cx["n_epg"])

        pen = cx.get("pen_subpop_ix", {}) or {}
        pfn = cx.get("pfn_subpop_ix", {}) or {}
        readout = cx.get("readout_ix", {}) or {}

        def _ix(d, k):
            return np.asarray(d.get(k, []), dtype=np.int64)

        subpops = {
            "bump":               np.arange(n_epg, dtype=np.int64),
            # encoder side — angular-velocity gate (existing) ...
            "afferent_PENa_L":    _ix(pen, "PENa_L"),
            "afferent_PENa_R":    _ix(pen, "PENa_R"),
            "afferent_PENb_L":    _ix(pen, "PENb_L"),
            "afferent_PENb_R":    _ix(pen, "PENb_R"),
            # ... and forward-velocity gate (new) ...
            "afferent_PFNd_L":    _ix(pfn, "PFNd_L"),
            "afferent_PFNd_R":    _ix(pfn, "PFNd_R"),
            "afferent_PFNv_L":    _ix(pfn, "PFNv_L"),
            "afferent_PFNv_R":    _ix(pfn, "PFNv_R"),
            # decoder side — declared readout sets (mirror of the gate) ...
            "readout_heading":    _ix(readout, "heading"),
            "readout_distance":   _ix(readout, "distance"),
            "readout_position":   _ix(readout, "position"),
        }
        bump_ring_ix = np.asarray(cx["epg_ix"], dtype=np.int64)
        body_ids = np.asarray(cx["bodyId"], dtype=np.int64)

        provenance = {
            "server": "neuprint.janelia.org",
            "dataset": "hemibrain:v1.2.1",
            "source_tables":
                "figures/drosophila_cx/drosophila_cx_connectome_338/"
                "{neurons,connections}.csv",
            "fetcher":
                "figures/drosophila_cx/fetch_cx_connectivity_pfn.py",
            "anatomy_dir": "papers/janelia_cx/anatomy/cx_anatomy_test",
            "anatomy_skeletons_fetcher":
                "figures/drosophila_cx/fetch_cx_skeletons_pfn.py",
            "type_count": len(cx["type_names"]),
            "n_bump_cells": n_epg,
            "dale_inh_types": ["Delta7", "ER6"],
            "dale_spectral_target": 0.9,
            "design_note": (
                "Path-integration companion to the zebrafish integrator. "
                "156-cell heading core (identity-equal to drosophila_cx_156) "
                "+ PFNd/PFNv afferents + PFNa/hDeltaB/PFR_a/PFR_b vector "
                "cells. Delta7+ER6 Dale-flipped inhibitory (5x), spectrally "
                "rescaled to rho=0.9; new families kept excitatory "
                "(PROVISIONAL). Readouts: heading<-EPG, distance<-PFNa, "
                "position<-hDeltaB/PFR. Loadable-only until the model-side "
                "forward-velocity gate + multi-readout decoder land."
            ),
        }

        return Circuit(
            name="drosophila_cx_338_v1",
            N=N,
            neuron_types=np.asarray(cx["neuron_types"], dtype=np.int64),
            type_names=list(cx["type_names"]),
            J_effective=np.asarray(cx["J_effective"], dtype=np.float32),
            soma_xyz=None,
            subpops=subpops,
            bump_ring_ix=bump_ring_ix,
            dale_signs=(np.asarray(cx["dale_signs"], dtype=np.float32)
                        if "dale_signs" in cx else None),
            body_ids=body_ids,
            provenance=provenance,
        )

    register_circuit("drosophila_cx_338_v1", build)
