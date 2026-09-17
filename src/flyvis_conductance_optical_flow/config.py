"""Composing the flyvis solver config, and the rig knobs that shape the reversals.

WHY NOT `flyvis.utils.config_utils.get_default_config`. That helper resolves its
config path with `inspect.stack()[1]` and then hands hydra a path RELATIVE TO THE
CALLER, so it only composes correctly for callers sitting inside the flyvis tree.
Called from here it asks hydra for `<site-packages>/config`, which does not exist.
`initialize_config_dir` takes an absolute directory and has no such dependency on
who is calling, so that is what this module uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

from datamate import namespacify
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# The published training regime, as specified in the fork's
# scripts/dvs_sim_parity/REPRODUCING.md. `task_original` restores the original
# temporal resampling of input and targets and the original train/validation
# split; `scheduler_original` spreads the ten learning-rate steps over the first
# 200,000 iterations of 250,000 and then holds the last rate, which is the
# trajectory the published models actually followed. Plain `scheduler` spreads
# the same ten steps over all 250,000 and is a DIFFERENT trajectory.
PUBLISHED_N_ITERS = 250000
PUBLISHED_SCHED_STOP_ITER = 200000

PUBLISHED_REGIME = [
    "task=task_original",
    "scheduler=scheduler_original",
    f"task.n_iters={PUBLISHED_N_ITERS}",
    f"scheduler.sched_stop_iter={PUBLISHED_SCHED_STOP_ITER}",
]

# The learning rate finishes its ten steps four fifths of the way through the run
# and is then held flat. Shortening a run has to carry this ratio with it: the
# schedule array is built with length `sched_stop_iter` and then padded out to
# `n_iters` (MultiTaskSolver.stepwise, solver.py:1142), so a run shorter than
# `sched_stop_iter` asks numpy for a NEGATIVE pad width and dies with
# "index can't contain negative values" before the first iteration.
PUBLISHED_STOP_FRACTION = PUBLISHED_SCHED_STOP_ITER / PUBLISHED_N_ITERS


def scaled_schedule_overrides(n_iters: int, chkpt_every_epoch: int | None = None):
    """Overrides that keep the published schedule SHAPE at a different length.

    Args:
        n_iters: the shortened iteration count.
        chkpt_every_epoch: lower this for a short run, whose total epoch count
            would otherwise fall below the published 334 and leave the run with
            only its first and last checkpoint.
    """
    out = [
        f"task.n_iters={n_iters}",
        f"scheduler.sched_stop_iter={max(1, round(n_iters * PUBLISHED_STOP_FRACTION))}",
    ]
    if chkpt_every_epoch is not None:
        out.append(f"scheduler.chkpt_every_epoch={chkpt_every_epoch}")
    return out

CURRENT_DYNAMICS = "PPNeuronIGRSynapses"
CONDUCTANCE_DYNAMICS = "ConductanceSynapses"


@dataclass
class ReversalRig:
    """How the two reversal potentials are parameterised.

    Mirrors the `student_*` knobs of the teacher-student trainer
    (connectome_gnn.models.known_ode.FlyvisConductanceKnownODE), reduced to what
    flyvis can express: flyvis node parameters are shared by cell type, so the
    per-neuron granularity of the twin has no counterpart here.

    Each reversal is a free variable squashed into a band of excursions from the
    postsynaptic cell's own resting potential:

        E(row) = bias(row) + (lo + (hi - lo) * sigmoid(raw)) * excursion

    with `bias` flyvis's own learned resting potential (one value per cell type,
    initialised Normal(0.5, 0.05)) and `excursion` a fixed nominal voltage scale.

    A BAND OF ZERO WIDTH PINS THE ROW: lo == hi makes the sigmoid term constant,
    so the raw variable receives no gradient and is registered with
    requires_grad False rather than left as a parameter the transform ignores.
    That is how the twin's `ion_sub` rig held the cation reversal fixed at 2.5
    excursions above rest while fitting only the chloride row.

    WHY THE BANDS DEFAULT WIDER THAN THE TWIN'S. In the fitted `ion_sub` twin, 40
    of the 65 chloride rows ended up sitting exactly on a band edge (38 on the
    lower, 2 on the upper) and only 25 in the interior: the band was placing the
    reversals, not the data. Training on a task rather than on a teacher's
    recorded activity, the whole point is to let the task place them, so the
    default bands span several excursions and `band_saturation` is reported as a
    diagnostic.

    Attributes:
        exc_dim: granularity of the cation reversal. 'global' is one value for
            the whole network, which is the biologically motivated choice: the
            nicotinic reversal is set by the sodium and potassium gradients,
            which every cell holds near the same values.
        inh_dim: granularity of the chloride reversal. 'per_type' gives one value
            per postsynaptic cell type, because the chloride equilibrium
            potential is set by the transporter balance, which genuinely differs
            between cell types.
        exc_band: (lo, hi) excursions from rest for the cation reversal.
        inh_band: (lo, hi) excursions from rest for the chloride reversal.
        excursion: the voltage scale one excursion means. Fixed rather than
            measured, because training from scratch on optic flow there is no
            recorded trajectory to measure a range from -- the voltage range is
            an outcome of training, not an input. 1.0 is defensible because
            flyvis activity is O(1) by construction of its initialisation.
    """

    exc_dim: Literal["global", "per_type"] = "global"
    inh_dim: Literal["global", "per_type"] = "per_type"
    exc_band: tuple[float, float] = (1.5, 6.0)
    inh_band: tuple[float, float] = (-6.0, -0.5)
    excursion: float = 1.0

    @property
    def n_learnable_reversals(self) -> str:
        """Human-readable count, for the run description and the log banner."""

        def rows(dim, band):
            if band[0] >= band[1]:
                return 0  # pinned: a zero-width band carries no free parameter
            return 1 if dim == "global" else 65

        return f"{rows(self.exc_dim, self.exc_band)} E_exc + {rows(self.inh_dim, self.inh_band)} E_inh"

    def node_config_overrides(self) -> list[str]:
        """Hydra overrides adding the two raw reversal variables to node_config.

        They are registered as NODE parameters, not edge parameters, because the
        driving force is (E - v_i): the reversal belongs to the POSTSYNAPTIC
        cell. flyvis then gives every node parameter a `targets` reader for free,
        so `params.targets.E_exc_raw` is the per-edge view of the receiving
        cell's value with no gather written by hand.
        """
        out = [f"+network.node_config.{name}={{"
               f"type:ReversalPotential,"
               f"groupby:[type],"
               f"initial_dist:Value,"
               f"value:0.0,"
               f"reversal_dim:{dim},"
               f"requires_grad:{str(band[0] < band[1]).lower()}"
               f"}}"
               for name, dim, band in (
                   ("E_exc_raw", self.exc_dim, self.exc_band),
                   ("E_inh_raw", self.inh_dim, self.inh_band),
               )]
        # The bands and the excursion belong to the DYNAMICS, not to the
        # parameters: the transform from raw variable to reversal potential is
        # applied in write_derived_params, where the resting potential it is
        # anchored on is reachable as params.nodes.bias.
        out += [
            f"+network.dynamics.exc_band=[{self.exc_band[0]},{self.exc_band[1]}]",
            f"+network.dynamics.inh_band=[{self.inh_band[0]},{self.inh_band[1]}]",
            f"+network.dynamics.excursion={self.excursion}",
        ]
        return out


def compose_config(
    task_name: str = "flow",
    ensemble_and_network_id: str = "9999/000",
    description: str = "",
    dynamics: str = CURRENT_DYNAMICS,
    published_regime: bool = True,
    rig: ReversalRig | None = None,
    extra_overrides: list[str] | None = None,
):
    """The full solver config, composed from flyvis's own yaml tree.

    Args:
        task_name: flyvis task directory, 'flow' for the optic-flow task.
        ensemble_and_network_id: '<ensemble>/<member>', e.g. '1000/000'. Results
            land at `<results_dir>/<task_name>/<ensemble>/<member>`, which is
            flyvis's own layout, so NetworkView and EnsembleView read our runs
            without any adaptation.
        description: free text stored with the run.
        dynamics: the NetworkDynamics subclass NAME. Resolved by flyvis against
            the live subclass tree, so the class must be imported first.
        published_regime: apply PUBLISHED_REGIME above.
        rig: reversal parameterisation; None for the current-based model, which
            has no reversals at all.
        extra_overrides: raw hydra overrides applied last, so they win.
    """
    overrides = [
        f"task_name={task_name}",
        f"ensemble_and_network_id={ensemble_and_network_id}",
        f"network.dynamics.type={dynamics}",
    ]
    if published_regime:
        overrides += PUBLISHED_REGIME
    if rig is not None:
        overrides += rig.node_config_overrides()
    if description:
        # QUOTED, because hydra parses the right-hand side with its own grammar:
        # an unquoted description containing '[' (as "... [smoke]" does) is read
        # as the start of a list literal and raises OverrideParseException.
        overrides.append("description='{}'".format(description.replace("'", "")))
    overrides += list(extra_overrides or [])

    config_dir = str(files("flyvis") / "config")
    GlobalHydra.instance().clear()
    try:
        initialize_config_dir(config_dir=config_dir, version_base=None)
        cfg = compose(config_name="solver", overrides=overrides)
        container = OmegaConf.to_container(cfg, resolve=True)
    finally:
        GlobalHydra.instance().clear()
    return namespacify(container)
