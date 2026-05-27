"""ZebrafishHdTaskRNN — companion of DrosophilaCxTaskRNN for the larval-
zebrafish dorsal-IPN heading-direction ring.

The dynamics, the sign-locked recurrent operator, the 4-scalar afferent
gate and the HD-ring-only readout are identical to the drosophila model
(see ``drosophila_cx_task_rnn.py``). The only thing that changes is the
connectome loader, which returns the zebrafish HD-circuit topology
(731 neurons across IPNd*/IPNds*/RIPN*/pt-IPN*) instead of the
fly hemibrain CX (156 across EPG/PEN/Δ7/PEG/ER6).

Fly-anatomy names retained in the inherited code carry zebrafish
semantics in this subclass per docs/zebrafish.tex §2:

    fly name           zebrafish semantics
    ---------          --------------------
    EPG (n_epg, epg_ix) IPNd* + IPNds* (dIPN HD ring, n_epg = 443)
    PEN_a sub-pops      RIPN* (habenula → IPN afferents)
    PEN_b sub-pops      pt-IPN* (pretectum → IPN afferents)
    Δ7 / ER6           (no analogue — bump cells provide the inhibition)

The task input is the same 3-channel stream as the drosophila model
(``[ω(t), cos θ₀·δ, sin θ₀·δ]`` in deg/s and radians); only the
distribution of ω(t) differs (swim-impulse boxcars instead of an OU
stream). See ``_generate_swim_integration_task`` for the data side.

Registered name: ``zebrafish_hd_si`` (HD = heading direction,
SI = swim integration).
"""
from __future__ import annotations

from connectome_gnn.models.drosophila_cx_task_rnn import DrosophilaCxTaskRNN
from connectome_gnn.models.registry import register_model


@register_model("zebrafish_hd_si")
class ZebrafishHdTaskRNN(DrosophilaCxTaskRNN):
    """Sign-locked larval-zebrafish dIPN heading-direction ring."""

    # Species-specific display labels, replacing the fly defaults inherited
    # from DrosophilaCxTaskRNN. The bump-carrying cells in this circuit are
    # the dIPN r1π neurons (IPNd* + IPNds*) and the afferents are the
    # habenula → IPN (RIPN*) + pretectum → IPN (pt-IPN*) streams.
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
        """Read the species-specific yaml flag for this preparation:
        ``output_from_dipn_only`` (the larval-zebrafish equivalent of the
        fly's ``output_from_epg_only``). When True the decoder reads only
        from the first ``n_epg = 443`` neurons of the loader's index
        ordering — the IPNd* + IPNds* block, \ie{} the r1π HD ring per
        Petrucco 2023."""
        return bool(getattr(gm, "output_from_dipn_only", False))
