"""Check the conductance twin against the model it was derived from.

Two independent checks:

1. **Integrator agreement.** flyvis-gnn's ``FlyVisODE`` and flyvis' own
   ``Network`` are driven with the same stimulus from the same initial state and
   the discrepancy is reported. They implement the same equation with the same
   forward-Euler step, so any difference is float32 rounding. This is what
   licenses fitting the twin entirely inside flyvis-gnn.

2. **Twin fidelity.** The conductance twin and the current-based model are rolled
   out freely on held-out clips and compared by voltage R^2, overall, per cell
   type, and resolved into time windows.

Example:
    python -m connectome_gnn.generators.verify_flyvis_conductance \\
        --params graphs_data/fly/conductance_twin
"""

import argparse
import json
import logging

import numpy as np
import torch

from connectome_gnn.generators.flyvis_conductance_fit import (
    build_index,
    concatenated_movies,
    current_based_ode,
    drift_profile,
    evaluate_twin,
    load_flyvis_network,
    naturalistic_movies,
    rollout,
    shunting_statistics,
    steady_state,
    stimulus_from_movie,
)
from connectome_gnn.generators.ode_params import FlyVisConductanceODEParams

logger = logging.getLogger(__name__)


def compare_integrators(net, movie, delta_t: float, device: str = "cpu") -> dict:
    """Roll out flyvis and flyvis-gnn on the same current-based model.

    Args:
        net: trained flyvis Network with the stock dynamics.
        movie: (frames, 1, hexals) clip.
        delta_t: integration step.
        device: device for the flyvis-gnn rollout.

    Returns:
        Error statistics between the two trajectories.
    """
    batched = movie[None].to(next(net.parameters()).device)
    with torch.no_grad():
        state = net.steady_state(1.0, delta_t, 1)
        net.stimulus.zero(1, batched.shape[1])
        net.stimulus.add_input(batched)
        reference = net(net.stimulus(), delta_t, state=state)[0].to(device)

    ode, params = current_based_ode(net, device=device)
    input_index = torch.as_tensor(
        np.asarray(net.stimulus.input_index), dtype=torch.long, device=device
    )
    n_nodes = params.V_i_rest.shape[0]
    stimulus = stimulus_from_movie(movie.to(device), input_index, n_nodes)
    initial = steady_state(ode, params, input_index, n_nodes, delta_t)
    predicted = rollout(ode, params, stimulus, initial, delta_t)

    error = (predicted - reference).abs()
    return {
        "max_abs_error": float(error.max()),
        "rms_error": float(error.pow(2).mean().sqrt()),
        "relative_rms_error": float(error.pow(2).mean().sqrt() / reference.std()),
        "reference_std": float(reference.std()),
    }


def main(args: argparse.Namespace) -> None:
    net = load_flyvis_network(args.model, args.checkpoint, args.extent)
    twin = FlyVisConductanceODEParams.load(args.params, device=args.device)
    if twin.G.shape[0] != net.n_edges:
        raise ValueError(
            f"the twin has {twin.G.shape[0]} edges but this connectome has "
            f"{net.n_edges}; pass the --extent the twin was derived at"
        )
    index = build_index(net, twin.reversal_exc, twin.reversal_inh)

    dataset_kwargs, n_frames = {}, args.n_frames
    if args.extent is not None:
        dataset_kwargs = dict(
            tasks=["flow"], interpolate=True,
            boxfilter=dict(extent=args.extent, kernel_size=13),
            vertical_splits=3, center_crop_fraction=0.7,
        )
        n_frames = 19
    movies, dataset = naturalistic_movies(
        dt=args.dt, n_sequences=args.n_sequences, n_frames=n_frames, **dataset_kwargs
    )
    if args.n_concat > 1:
        movies = concatenated_movies(
            dataset, n_concat=args.n_concat, n_sequences=args.n_sequences
        )

    print("\n1. flyvis vs flyvis-gnn, same current-based model")
    integrator = [
        compare_integrators(net, movie, args.dt, device=args.device)
        for movie in movies[: args.n_integrator_checks]
    ]
    print(
        f"   max |flyvis - flyvis-gnn|   "
        f"{max(r['max_abs_error'] for r in integrator):.3e}\n"
        f"   rms                          "
        f"{np.mean([r['rms_error'] for r in integrator]):.3e}\n"
        f"   rms / std(reference)         "
        f"{np.mean([r['relative_rms_error'] for r in integrator]):.3e}"
    )

    ode, params = current_based_ode(net, device=args.device)
    initial = steady_state(
        ode, params, index.input_index, index.n_nodes, args.dt
    )
    stimuli, teacher_voltages = [], []
    for movie in movies:
        stimulus = stimulus_from_movie(
            movie.to(args.device), index.input_index, index.n_nodes
        )
        stimuli.append(stimulus)
        teacher_voltages.append(
            rollout(ode, params, stimulus, initial.clone(), args.dt)
        )

    print("\n2. conductance twin vs current-based model")
    result = evaluate_twin(twin, index, teacher_voltages, stimuli, args.dt)
    duration = teacher_voltages[0].shape[0] * args.dt
    print(
        f"   {len(movies)} clips of {duration:.2f} s   voltage r2 {result['r2']:.4f}"
    )
    worst = sorted(result["r2_per_type"].items(), key=lambda kv: kv[1])[:8]
    print(f"   worst cell types: {[(k, round(v, 3)) for k, v in worst]}")

    profile = drift_profile(
        twin, index, teacher_voltages, stimuli, args.dt, n_windows=args.n_windows
    )
    print("   r2 per time window:")
    for time, value in zip(profile["time"], profile["r2"]):
        print(f"     t={time:5.2f}s  r2={value:.4f}")

    shunting = shunting_statistics(twin, index, stimuli[:2], args.dt)
    strongest = sorted(shunting.items(), key=lambda kv: -kv[1]["G_q95"])[:8]
    print(f"   most shunted: {[(k, round(v['G_q95'], 2)) for k, v in strongest]}")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(
                {
                    "integrator": integrator,
                    "r2": result["r2"],
                    "r2_per_type": result["r2_per_type"],
                    "drift": profile,
                    "shunting": shunting,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="flow/0000/000")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument(
        "--extent", type=int, default=None,
        help="extent the twin was derived at; 8 for the graph_data_generator twin",
    )
    parser.add_argument("--params", default="graphs_data/fly/conductance_twin")
    parser.add_argument("--dt", type=float, default=1 / 100)
    parser.add_argument("--n-sequences", type=int, default=4)
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--n-concat", type=int, default=1)
    parser.add_argument("--n-windows", type=int, default=8)
    parser.add_argument("--n-integrator-checks", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    main(parser.parse_args())
