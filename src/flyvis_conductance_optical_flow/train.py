"""Train a flyvis network on the optic-flow task, current-based or conductance.

WHY NOT `flyvis train-single`. That entry point is a hydra `@main` in flyvis's own
package, so the conductance dynamics and reversal parameters would have to be
imported into its process before it builds the Network -- which the CLI gives no
hook for. Driving `MultiTaskSolver` directly costs nothing (the solver takes a
plain config) and makes the import order explicit.

THE SAME RUNNER TRAINS BOTH FAMILIES, differing only in `--dynamics`, so the
current-based reference ensemble and the conductance one cannot drift apart
through their harness. REPRODUCING.md is explicit that a conductance ensemble
must be compared against a reference trained with THIS pipeline rather than
against the published `flow/0000` directly.

Usage:
    # nominal run, current model, short: does the task pipeline work at all
    python -m flyvis_conductance_optical_flow.train --smoke \
        --dynamics PPNeuronIGRSynapses --ensemble 1000 --member 000

    # conductance, full published regime
    python -m flyvis_conductance_optical_flow.train \
        --dynamics ConductanceSynapses --ensemble 2000 --member 000
"""

from __future__ import annotations

import argparse
import logging

import flyvis_conductance_optical_flow as fcof
from flyvis_conductance_optical_flow.config import (
    CONDUCTANCE_DYNAMICS,
    CURRENT_DYNAMICS,
    ReversalRig,
    compose_config,
    scaled_schedule_overrides,
)
from flyvis_conductance_optical_flow.progress import TrainingProgress

logger = logging.getLogger(__name__)

# Iterations for --smoke. Enough to leave the initial transient and see the
# validation loss move, short enough to finish in minutes on one A6000; the full
# regime is 250,000.
SMOKE_ITERS = 1000
# Checkpoint interval in EPOCHS for --smoke. The published 334 exceeds a short
# run's total epoch count, which would leave it with only a first and a last
# checkpoint and nothing for the analysis pass to select over.
SMOKE_CHKPT_EVERY_EPOCH = 10


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--dynamics",
        default=CONDUCTANCE_DYNAMICS,
        choices=[CURRENT_DYNAMICS, CONDUCTANCE_DYNAMICS],
        help="NetworkDynamics subclass name; selects the synapse family",
    )
    p.add_argument("--task-name", default="flow")
    p.add_argument(
        "--ensemble",
        required=True,
        help="ensemble id, e.g. 1000 for the current reference, 2000 for conductance",
    )
    p.add_argument("--member", default="000", help="member id within the ensemble")
    p.add_argument("--description", default="")
    p.add_argument(
        "--smoke",
        action="store_true",
        help=f"short run ({SMOKE_ITERS} iterations) to prove the pipeline, not a result",
    )
    p.add_argument(
        "--n-iters",
        type=int,
        default=None,
        help="override the iteration count of the regime",
    )
    p.add_argument(
        "--ncols", type=int, default=150, help="width of the progress bar, in characters"
    )
    p.add_argument(
        "--delete-if-exists",
        action="store_true",
        help="overwrite an existing run directory of the same name",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="compose the config, build the solver, verify registration, then stop "
        "without training",
    )

    rig = p.add_argument_group(
        "reversal rig (conductance only)",
        "A band of zero width (lo == hi) PINS that reversal and leaves it with no "
        "free parameter, which is how the teacher-student 'ion_sub' rig held the "
        "cation reversal fixed.",
    )
    rig.add_argument("--exc-dim", default="global", choices=["global", "per_type"])
    rig.add_argument("--inh-dim", default="per_type", choices=["global", "per_type"])
    rig.add_argument("--exc-band", type=float, nargs=2, default=(1.5, 6.0),
                     metavar=("LO", "HI"))
    rig.add_argument("--inh-band", type=float, nargs=2, default=(-6.0, -0.5),
                     metavar=("LO", "HI"))
    rig.add_argument("--excursion", type=float, default=1.0,
                     help="voltage scale one excursion of rest means")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    conductance = args.dynamics == CONDUCTANCE_DYNAMICS
    rig = (
        ReversalRig(
            exc_dim=args.exc_dim,
            inh_dim=args.inh_dim,
            exc_band=tuple(args.exc_band),
            inh_band=tuple(args.inh_band),
            excursion=args.excursion,
        )
        if conductance
        else None
    )

    extra = []
    n_iters = args.n_iters if args.n_iters is not None else (
        SMOKE_ITERS if args.smoke else None
    )
    if n_iters is not None:
        # Not just task.n_iters: the learning-rate schedule has to be rescaled
        # with it, or the run dies before iteration 0. See
        # config.scaled_schedule_overrides.
        extra += scaled_schedule_overrides(
            n_iters, chkpt_every_epoch=SMOKE_CHKPT_EVERY_EPOCH if args.smoke else None
        )

    config = compose_config(
        task_name=args.task_name,
        ensemble_and_network_id=f"{args.ensemble}/{args.member}",
        description=args.description or (
            f"{args.dynamics}"
            + (f", {rig.n_learnable_reversals} learnable" if rig else "")
            + (" [smoke]" if args.smoke else "")
        ),
        dynamics=args.dynamics,
        rig=rig,
        extra_overrides=extra,
    )

    print(f"\033[96m{args.dynamics}  ->  {config.network_name}\033[0m")
    print(f"  iterations        {config.task.n_iters}")
    print(f"  original split    {config.task.original_split}")
    print(f"  original sampling {config.task.dataset.original_sampling}")
    print(f"  sched_stop_iter   {config.scheduler.sched_stop_iter}")
    if rig is not None:
        print(f"  reversals         {rig.n_learnable_reversals} "
              f"(exc band {rig.exc_band}, inh band {rig.inh_band}, "
              f"excursion {rig.excursion})")

    # Imported here so the banner above prints before flyvis spends time on the
    # connectome; `fcof` is already imported at module scope, which is what
    # registers the dynamics and the reversal parameters.
    from flyvis.solver import MultiTaskSolver

    solver = MultiTaskSolver(
        name=config.network_name,
        config=config,
        delete_if_exists=args.delete_if_exists,
    )

    # THE CHECK THAT MATTERS. flyvis resolves dynamics by name and falls back to
    # the base class with a warning, whose write_state_velocity is `pass` -- a
    # network that trains happily and computes nothing.
    fcof.assert_registered(solver.network, args.dynamics)
    n_free = sum(p.numel() for p in solver.network.parameters() if p.requires_grad)
    print(f"\033[92mdynamics {type(solver.network.dynamics).__name__} registered; "
          f"{n_free} free network parameters\033[0m")

    if args.dry_run:
        print("--dry-run: stopping before training")
        return 0

    with TrainingProgress(solver, ncols=args.ncols):
        solver.train()
    print(f"\033[92mtrained; results at {solver.path}\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
