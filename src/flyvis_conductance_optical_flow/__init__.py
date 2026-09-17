"""Conductance-based flyvis model trained on the optic-flow task.

The model is a `flyvis.network.dynamics.NetworkDynamics` subclass and its
reversal potentials are `flyvis.network.initialization.Parameter` subclasses.
flyvis selects both by class NAME from config, resolved against the live Python
subclass tree, so IMPORTING THIS PACKAGE BEFORE A NETWORK IS CONSTRUCTED is what
makes them selectable. Nothing here patches flyvis.

`forward_subclass` does not raise on a name it cannot find -- it warns and
instantiates the base class, whose methods are all `pass`. `assert_registered`
below is the check worth running after building a network, because the failure
mode is otherwise a network that trains quietly with no dynamics at all.
"""

# THE IMPORT IS THE EFFECT. Neither name is used in this file: importing them
# executes their `class` statements, which is what puts them into the subclass
# trees flyvis searches by name. `__all__` below is what keeps them -- it marks
# both as intentional re-exports, so ruff's F401 does not fire and `ruff --fix`
# cannot delete the two lines that perform the registration -- verified by running
# ruff over this file with the unused-import rule selected.
from flyvis_conductance_optical_flow.dynamics import ConductanceSynapses
from flyvis_conductance_optical_flow.parameters import ReversalPotential

__all__ = ["ConductanceSynapses", "ReversalPotential", "assert_registered"]


def assert_registered(network, expected: str) -> None:
    """Raise unless `network.dynamics` really is the class the config asked for.

    Args:
        network: a constructed `flyvis.network.Network`.
        expected: the class name that was put in `network.dynamics.type`.
    """
    got = type(network.dynamics).__name__
    if got != expected:
        raise RuntimeError(
            f"network.dynamics is {got}, not {expected}. flyvis resolves dynamics "
            "by name against the live subclass tree and falls back to the base "
            "NetworkDynamics with a warning when the name is unknown -- whose "
            "write_state_velocity is `pass`, so the network would train with no "
            "dynamics. Import flyvis_conductance_optical_flow before building the "
            "Network."
        )
