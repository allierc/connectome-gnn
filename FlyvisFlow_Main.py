"""Entry point for the conductance-flyvis optic-flow work, in GNN_Main.py's shape.

    python FlyvisFlow_Main.py -o train conductance_2000_000
    python FlyvisFlow_Main.py -o train current_1000_000
    python FlyvisFlow_Main.py -o train_smoke conductance_9991_000

A RUN NAME IS `{family}_{ensemble}_{member}`, family being `current` or
`conductance`. It resolves to flyvis's own results layout,
`<FLYVIS_ROOT_DIR>/results/flow/<ensemble>/<member>`, so NetworkView and
EnsembleView read these runs with no adaptation.

Nothing needs to be exported to run this: `src/` goes on the path below, and
FLYVIS_ROOT_DIR is exported by the env's own activate.d, so an interactive
`bsub -Is` that inherits an activated shell has everything it needs.
"""

import argparse
import os
import sys

# Ensure src/ is on the path so flyvis_conductance_optical_flow is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import matplotlib

matplotlib.use('Agg')  # non-interactive backend before other imports

from flyvis_conductance_optical_flow.config import (  # noqa: E402
    CONDUCTANCE_DYNAMICS,
    CURRENT_DYNAMICS,
)
from flyvis_conductance_optical_flow.train import main as train_main  # noqa: E402

FAMILY_DYNAMICS = {
    "current": CURRENT_DYNAMICS,
    "conductance": CONDUCTANCE_DYNAMICS,
}


def parse_run_name(name: str):
    """`{family}_{ensemble}_{member}` -> (dynamics class name, ensemble, member).

    The member is kept as a STRING rather than parsed to an int: flyvis names
    member directories by the literal text, so '000' and '0' would be two
    different runs on disk and only one of them is the one the ensemble tools
    expect.
    """
    parts = name.split("_")
    if len(parts) != 3 or parts[0] not in FAMILY_DYNAMICS:
        raise SystemExit(
            f"run name {name!r} is not '{{family}}_{{ensemble}}_{{member}}' with "
            f"family one of {sorted(FAMILY_DYNAMICS)}; e.g. conductance_2000_000"
        )
    family, ensemble, member = parts
    return FAMILY_DYNAMICS[family], ensemble, member


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="flyvis conductance on optic flow")
    # `-o` takes the task FOLLOWED BY the run name, exactly as GNN_Main.py takes
    # the task followed by the config name: `-o train conductance_2000_000`.
    parser.add_argument(
        "-o", "--option", nargs="+", required=True,
        help="task then run name, e.g. `-o train conductance_2000_000`. Task is "
        "train, or train_smoke for a short pipeline check that is not a result.",
    )
    args, passthrough = parser.parse_known_args()

    if len(args.option) < 2:
        raise SystemExit(
            "-o takes the task and the run name, e.g. -o train conductance_2000_000"
        )
    option, run_name = args.option[0], args.option[-1]
    dynamics, ensemble, member = parse_run_name(run_name)

    argv = ["--dynamics", dynamics, "--ensemble", ensemble, "--member", member]
    if "smoke" in option:
        argv.append("--smoke")
    if "train" not in option:
        raise SystemExit(f"option {option!r} is not a train option")
    # Anything else on the command line (rig bands, --ncols, --delete-if-exists)
    # is handed through to the trainer unchanged.
    argv += passthrough

    raise SystemExit(train_main(argv))
