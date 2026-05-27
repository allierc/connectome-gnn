"""ZebrafishHdTaskGNN — companion of DrosophilaCxTaskGNN for the larval-
zebrafish dorsal-IPN heading-direction ring.

Same shape as the RNN companion (``zebrafish_hd_task_rnn.py``): inherits
the entire dynamics / message-passing / sign-locking / encoder-decoder
machinery from ``DrosophilaCxTaskGNN``, overriding only the loader hook
and the bump-only-readout config-flag hook.

Fly-anatomy names retained in the inherited code carry zebrafish
semantics in this subclass per docs/zebrafish.tex §2:

    fly name           zebrafish semantics
    ---------          --------------------
    EPG (n_epg, epg_ix) IPNd* + IPNds* (dIPN HD ring, n_epg = 443)
    PEN_a sub-pops      RIPN* (habenula → IPN afferents)
    PEN_b sub-pops      pt-IPN* (pretectum → IPN afferents)
    Δ7 / ER6           (no analogue — bump cells are themselves inhibitory)

Registered name: ``zebrafish_hd_si_gnn``.
"""
from __future__ import annotations

from connectome_gnn.models.drosophila_cx_task_gnn import DrosophilaCxTaskGNN
from connectome_gnn.models.registry import register_model


@register_model("zebrafish_hd_si_gnn")
class ZebrafishHdTaskGNN(DrosophilaCxTaskGNN):
    """Sign-locked larval-zebrafish dIPN heading-direction GNN."""

    # Species-specific display labels, replacing the fly defaults inherited
    # from DrosophilaCxTaskGNN.
    bump_label: str = "r1π / dIPN"
    afferent_label: str = "RIPN / pt-IPN"

    @staticmethod
    def _load_connectome(datapath):
        from connectome_gnn.generators.connconstr_data import (
            load_zebrafish_hd_connectome,
        )
        return load_zebrafish_hd_connectome(datapath)

    @staticmethod
    def _resolve_bump_only_readout(gm) -> bool:
        """Read the zebrafish-specific yaml flag ``output_from_dipn_only``
        (counterpart of the fly's ``output_from_epg_only``). When True the
        decoder reads only from the first n_epg = 443 neurons of the
        loader's index ordering — the IPNd* + IPNds* block, i.e. the
        r1π HD ring per Petrucco 2023."""
        return bool(getattr(gm, "output_from_dipn_only", False))
