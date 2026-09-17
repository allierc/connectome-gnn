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
        "--extent",
        type=int,
        default=None,
        help="retinotopic extent of the connectome. The published protocol is 15 "
        "(45,669 neurons, 1,513,231 edges); 8 gives 13,741 and 434,112, the network "
        "the teacher-student twin was fitted on. PARAMETER COUNT IS UNCHANGED -- "
        "flyvis shares parameters by cell type and filter tap, so a smaller extent "
        "is the same model on a smaller retina, roughly 3x faster. Departs from the "
        "published regime, so give such runs their own ensemble id.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed of the resting-potential draw, which is the ONLY sampled "
        "initialisation flyvis has -- Normal(0.5, 0.05) per cell type. Every other "
        "parameter starts at a fixed value, so this is what separates one ensemble "
        "member from another. Default leaves flyvis's own 0. Give each member both "
        "its own --seed and its own member id, or the two runs collide on disk.",
    )
    p.add_argument(
        "--ncols", type=int, default=150, help="width of the progress bar, in characters"
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="keep flyvis's INFO logging. It prints the whole scheduler state once "
        "per epoch, and an epoch here is 12 iterations, so it overruns the progress "
        "bar several times a second; off by default",
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


def _integrator_note(network) -> str:
    """One line naming the time step the dynamics actually implements.

    Read off the class ATTRIBUTE, not off the class name, and that distinction is
    the whole point of the line. The name `ConductanceSynapses` is the same in
    every checkout; `INTEGRATION` exists only where the exponential-Euler step
    does. So a job launched from a checkout that predates the fix prints `forward
    Euler` and can be killed in the first seconds, rather than looking identical
    for the two hours it takes to explode the way flow/2000/000 did at iteration
    16,368.

    Every other NetworkDynamics -- flyvis's own current model included -- carries
    no such attribute and keeps flyvis's forward Euler at the fixed dt, which is
    the truth for them and is what they say here.
    """
    got = type(network.dynamics).__name__
    step = getattr(type(network.dynamics), "INTEGRATION", None)
    if step is not None:
        return f"{step} (exact at frozen coefficients)"
    if got == CONDUCTANCE_DYNAMICS:
        return ("forward Euler -- STALE CHECKOUT. This conductance model has no "
                "INTEGRATION attribute, so it is the pre-fix step that explodes "
                "once (dt/tau_i)(1 + G_i) passes 2. Kill the job and pull.")
    return f"flyvis forward Euler at the fixed dt (unchanged; {got} is flyvis's own)"


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.verbose:
        # flyvis calls logging.basicConfig(level=INFO) at import, and its solver
        # logs the full scheduler state at every epoch. An epoch is 12 iterations
        # on this split, so that is several screens a second overwriting the
        # progress bar. Raising the level on the `flyvis` parent logger silences
        # its children (flyvis.solver, flyvis.network, ...) while leaving
        # warnings and errors -- the things worth interrupting a bar for.
        logging.getLogger("flyvis").setLevel(logging.WARNING)

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
    if args.extent is not None:
        # BOTH EXTENTS, ALWAYS. The connectome extent sets how many photoreceptors
        # the network has; the boxfilter extent sets how many hexals the movie is
        # rendered onto. Change one alone and `Stimulus.add_input` refuses the
        # batch -- "input has shape (1, 40, 1, 721) but buffer has shape
        # (1, 40, 13741)" -- because the rendering still carries extent 15's 721
        # hexals. Changing the boxfilter also means a NEW RenderedSintel cache,
        # built once on first use.
        extra.append(f"network.connectome.extent={args.extent}")
        extra.append(f"task.dataset.boxfilter.extent={args.extent}")
    if args.seed is not None:
        # THE ONLY SEEDED PARAMETER IN THE WHOLE NETWORK. Every other initial
        # distribution is a `Value`: syn_strength is scale/<N> per edge type, the
        # time constant is 0.05 s, both reversal variables start at 0.0 and the
        # Dale sign comes from the connectome. The resting potential alone is
        # SAMPLED, Normal(0.5, 0.05) per cell type, through a generator seeded by
        # this key (initialization.py:166), so it is what makes one ensemble member
        # differ from another at iteration 0.
        extra.append(f"network.node_config.bias.seed={args.seed}")

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
    print(f"  connectome        extent {config.network.connectome.extent}")
    print(f"  iterations        {config.task.n_iters}")
    print(f"  original split    {config.task.original_split}")
    print(f"  original sampling {config.task.dataset.original_sampling}")
    print(f"  sched_stop_iter   {config.scheduler.sched_stop_iter}")
    print(f"  bias seed         {config.network.node_config.bias.seed} "
          f"(resting potential, the only sampled initialisation)")
    if rig is not None:
        print(f"  reversals         {rig.n_learnable_reversals} "
              f"(exc band {rig.exc_band}, inh band {rig.inh_band}, "
              f"excursion {rig.excursion})")

    # Imported here so the banner above prints before flyvis spends time on the
    # connectome; `fcof` is already imported at module scope, which is what
    # registers the dynamics and the reversal parameters.
    from flyvis.network import Network
    from flyvis.solver import MultiTaskSolver

    if args.dry_run:
        # BUILD THE NETWORK ALONE, not the solver. Constructing MultiTaskSolver
        # creates the run directory and stores its config, so a dry run left a
        # half-made directory behind that then refused the real run with
        # datamate's "incompatible config". A dry run should touch no results.
        network = Network(**config.network)
        fcof.assert_registered(network, args.dynamics)
        n_free = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print(f"\033[92mdynamics {type(network.dynamics).__name__} registered; "
              f"{n_free} free network parameters\033[0m")
        # The same line the real run prints, so `--dry-run` is enough to confirm a
        # checkout has the exponential-Euler step without queueing for a GPU.
        print(f"\033[92mintegrator        {_integrator_note(network)}\033[0m")
        print("--dry-run: nothing written")
        return 0

    # AN EXISTING RUN DIRECTORY IS REFUSED BY DATAMATE, NOT BY FLYVIS, and the
    # refusal reads as a config diff on `delete_if_exists` -- a key datamate
    # strips from the passed config before comparing but keeps in the stored
    # `_meta.yaml`, so the two can never agree. Any second run of a name hits it,
    # including a re-launch after a job died. Caught here so the message names
    # the directory and the way out rather than ending in a datamate traceback.
    import flyvis

    run_dir = flyvis.results_dir / config.network_name
    if run_dir.exists() and not args.delete_if_exists:
        n_chkpts = len(list((run_dir / "chkpts").glob("*"))) if (run_dir / "chkpts").exists() else 0
        raise SystemExit(
            f"{run_dir} already exists ({n_chkpts} checkpoints).\n"
            "  --delete-if-exists   start over, discarding it\n"
            "  or choose another member/ensemble id in the run name\n"
            "Datamate refuses to reopen it: it strips `delete_if_exists` from the "
            "passed config but stores it, so the configs never compare equal."
        )

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
    # WHICH INTEGRATOR IS ABOUT TO RUN, said out loud. The step is a property of
    # the dynamics class, not of the config, so nothing else in the run's output
    # distinguishes the exponential-Euler conductance model from the forward-Euler
    # one that died at iteration 16,368 -- and a run that starts with the wrong
    # one looks identical for two hours. flyvis's own current model is untouched
    # by that change and says so here.
    print(f"\033[92mintegrator        {_integrator_note(solver.network)}\033[0m")
    print(f"\033[96mwriting to        {solver.path}\033[0m")

    with TrainingProgress(solver, ncols=args.ncols):
        solver.train()
    print(f"\033[92mtrained; results at {solver.path}\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
