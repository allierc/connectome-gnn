"""The reversal potential as a flyvis node parameter.

REGISTERED BY CLASS NAME, NOT IMPORTED BY FLYVIS. `Network.__init__` builds every
node parameter with `forward_subclass(Parameter, {"type": <name>, ...})`, and
`find_subclass` walks the live `Parameter.__subclasses__()` tree. Defining the
class here and importing this module before a Network is constructed is the whole
of the registration; flyvis needs no patch.

A NODE PARAMETER, NOT AN EDGE PARAMETER. The driving force is (E - v_i), so the
reversal belongs to the POSTSYNAPTIC cell -- it is a property of the receiving
neuron's ion gradients, not of the synapse. flyvis gives every node parameter
three readers (`nodes`, `sources`, `targets`, see network.py:180), so registering
it here means `params.targets.E_inh_raw` is the per-edge view of the receiving
cell's value with no gather written by hand.

WHAT IS STORED IS THE RAW VARIABLE, NOT THE REVERSAL. The reversal is anchored on
the cell's resting potential, which is a *different* parameter (`bias`), so the
composition cannot happen inside this class -- it happens in
`ConductanceSynapses.write_derived_params`, where both are reachable. See
dynamics.py for the transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datamate import Namespace
from flyvis.connectome import ConnectomeFromAvgFilters
from flyvis.network.initialization import (
    InitialDistribution,
    Parameter,
    deepcopy_config,
    get_scatter_indices,
    symmetry_masks,
)
from flyvis.utils.class_utils import forward_subclass
from flyvis.utils.type_utils import byte_to_str

__all__ = ["ReversalPotential"]

# The column name used to put every neuron in one group. Any constant column
# works -- `get_scatter_indices` only ever zips the groupby columns and maps each
# row to its group's position -- but naming it explicitly keeps the intent
# visible in a traceback.
_ALL_NEURONS = "__all_neurons__"


class ReversalPotential(Parameter):
    """Free variable behind one reversal potential, shared globally or by cell type.

    `reversal_dim` in the parameter's config chooses the granularity:

        'global'    one row for the whole network. The biologically motivated
                    choice for the CATION reversal: the nicotinic reversal is set
                    by the sodium and potassium gradients, which every cell holds
                    near the same values.
        'per_type'  one row per cell type, 65 of them. The choice for the
                    CHLORIDE reversal, whose value is set by the KCC/NKCC
                    transporter balance and genuinely differs cell type to cell
                    type.

    The stored values are pre-squash, so they are unbounded and initialising them
    at 0.0 puts each row at the MIDDLE of its band (sigmoid(0) = 0.5).
    """

    @deepcopy_config
    def __init__(
        self, param_config: Namespace, connectome: ConnectomeFromAvgFilters
    ) -> None:
        reversal_dim = param_config.get("reversal_dim", "per_type")
        if reversal_dim not in ("global", "per_type"):
            raise ValueError(
                f"reversal_dim {reversal_dim!r} is neither 'global' nor 'per_type'. "
                "flyvis shares node parameters by cell type, so the per-neuron "
                "granularity of the teacher-student trainer has no counterpart here."
            )

        nodes_dir = connectome.nodes
        types = np.asarray(byte_to_str(nodes_dir["type"][:]))

        if reversal_dim == "global":
            # One group for every neuron in the network. The constant column is
            # what collapses the groupby; `type` is carried alongside it only so
            # the grouped frame still has a name to report.
            nodes = pd.DataFrame({
                _ALL_NEURONS: np.zeros(len(types), dtype=np.int64),
            })
            groupby = [_ALL_NEURONS]
        else:
            nodes = pd.DataFrame({"type": types})
            groupby = ["type"]

        grouped_nodes = nodes.groupby(groupby, as_index=False, sort=False).first()
        n_rows = len(grouped_nodes)

        # `Value` reads one entry per row; np.repeat turns the single scalar in
        # the yaml into the per-row vector it expects, exactly as TimeConstant
        # does for its own scalar `value`.
        param_config["value"] = np.repeat(param_config["value"], n_rows)

        self.parameter = forward_subclass(
            InitialDistribution, param_config, subclass_key="initial_dist"
        )
        self.indices = get_scatter_indices(nodes, grouped_nodes, groupby)
        self.keys = (
            ["global"] if reversal_dim == "global"
            else grouped_nodes["type"].tolist()
        )
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )
