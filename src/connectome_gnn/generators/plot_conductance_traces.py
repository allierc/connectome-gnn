"""Plot central-neuron traces of the current-based model and its conductance twin.

One panel per cell type, showing the central column's neuron, over a long
naturalistic rollout. Both models are driven by the same stimulus from their own
steady states, so the panels are directly comparable.

The per-panel R² is for one neuron, and runs higher than the population figure
reported by ``verify_flyvis_conductance`` (median ~0.99 here against ~0.93 over
all 45,669 neurons). That is not a contradiction: the central column is the
best-driven part of the retinotopic array, while the population number is
variance-weighted over every column including the poorly driven edges. Read this
figure for trace shape, and the population number for overall fidelity.

The stimulus is several short clips joined end to end, so the traces contain
scene cuts; both models see the identical input through them.

Example:
    python -m connectome_gnn.generators.plot_conductance_traces \\
        --params graphs_data/fly/conductance_twin --out assets/conductance_twin_traces.png
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from connectome_gnn.generators.flyvis_conductance_fit import (  # noqa: E402
    concatenated_movies,
    current_based_ode,
    load_flyvis_network,
    naturalistic_movies,
    rollout,
    steady_state,
    stimulus_from_movie,
)
from connectome_gnn.generators.flyvis_conductance_ode import FlyVisConductanceODE  # noqa: E402
from connectome_gnn.generators.ode_params import FlyVisConductanceODEParams  # noqa: E402

logger = logging.getLogger(__name__)

CURRENT_COLOR = "#111111"
TWIN_COLOR = "#e8590c"


def central_cell_traces(net, twin, movie, delta_t, device="cpu"):
    """Roll out both models and pull the central neuron of every cell type.

    Args:
        net: trained flyvis Network with the stock current-based dynamics.
        twin: the conductance twin's parameters.
        movie: (frames, 1, hexals) stimulus.
        delta_t: integration step.
        device: device to simulate on.

    Returns:
        (cell_types, current_based, conductance_based) with the two trace arrays
        shaped (frames, n_cell_types).
    """
    cell_types = [
        t.decode("utf-8") if isinstance(t, bytes) else str(t)
        for t in net.connectome.unique_cell_types[:]
    ]
    central = torch.as_tensor(
        np.asarray(net.connectome.central_cells_index[:]), dtype=torch.long
    )

    ode, params = current_based_ode(net, device=device)
    input_index = torch.as_tensor(
        np.asarray(net.stimulus.input_index), dtype=torch.long, device=device
    )
    n_nodes = params.V_i_rest.shape[0]
    stimulus = stimulus_from_movie(movie.to(device), input_index, n_nodes)

    initial = steady_state(ode, params, input_index, n_nodes, delta_t)
    current_based = rollout(ode, params, stimulus, initial, delta_t)

    twin_ode = FlyVisConductanceODE(ode_params=twin, device=device, delta_t=delta_t)
    twin_initial = steady_state(twin_ode, twin, input_index, n_nodes, delta_t)
    conductance_based = rollout(twin_ode, twin, stimulus, twin_initial, delta_t)

    return (
        cell_types,
        current_based[:, central].cpu().numpy(),
        conductance_based[:, central].cpu().numpy(),
    )


def plot_traces(cell_types, current_based, conductance_based, delta_t, out,
                n_columns=3, panel_height=0.62, dpi=200):
    """Stack one panel per cell type, both models overlaid.

    Args:
        cell_types: cell type names.
        current_based: (frames, n_cell_types) traces of the pretrained model.
        conductance_based: (frames, n_cell_types) traces of the twin.
        delta_t: integration step, for the time axis.
        out: output image path.
        n_columns: number of columns of panels.
        panel_height: height of one panel, in inches.
        dpi: output resolution.

    Returns:
        Per-cell-type R² of the central neuron's trace.
    """
    n_types = len(cell_types)
    n_rows = int(np.ceil(n_types / n_columns))
    time = np.arange(current_based.shape[0]) * delta_t

    figure, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(5.2 * n_columns, panel_height * n_rows),
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    r2 = {}
    for index, name in enumerate(cell_types):
        row, column = index % n_rows, index // n_rows
        ax = axes[row, column]
        reference, predicted = current_based[:, index], conductance_based[:, index]
        variance = reference.var()
        r2[name] = float(
            1 - ((predicted - reference) ** 2).mean() / max(variance, 1e-30)
        )

        ax.plot(time, reference, color=CURRENT_COLOR, linewidth=0.9)
        ax.plot(time, predicted, color=TWIN_COLOR, linewidth=0.9, alpha=0.85)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=7)
        # headroom so the annotation never lands on a trace
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.35 * (high - low))
        ax.text(
            0.995, 0.97, f"$R^2$ {r2[name]:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.5, color="#555555",
        )
        ax.tick_params(labelsize=6, length=2, pad=1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if row != n_rows - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)

    # blank any unused panels
    for index in range(n_types, n_rows * n_columns):
        axes[index % n_rows, index // n_rows].set_visible(False)

    for column in range(n_columns):
        axes[-1, column].set_xlabel("time (s)", fontsize=7)

    handles = [
        plt.Line2D([], [], color=CURRENT_COLOR, linewidth=1.4,
                   label="current-based (pretrained flyvis)"),
        plt.Line2D([], [], color=TWIN_COLOR, linewidth=1.4,
                   label="conductance-based twin"),
    ]
    figure.legend(
        handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
    )
    figure.suptitle(
        f"central neuron per cell type, {time[-1] + delta_t:.1f} s naturalistic "
        f"rollout   (median $R^2$ {np.median(list(r2.values())):.3f})",
        fontsize=10, y=1.006,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    figure.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return r2


def main(args: argparse.Namespace) -> None:
    net = load_flyvis_network(args.model, args.checkpoint, args.extent)
    twin = FlyVisConductanceODEParams.load(args.params, device=args.device)
    if twin.G.shape[0] != net.n_edges:
        raise ValueError(
            f"the twin has {twin.G.shape[0]} edges but this connectome has "
            f"{net.n_edges}; pass the --extent the twin was derived at"
        )

    dataset_kwargs, n_frames = {}, args.n_frames
    if args.extent is not None:
        dataset_kwargs = dict(
            tasks=["flow"], interpolate=True,
            boxfilter=dict(extent=args.extent, kernel_size=13),
            vertical_splits=3, center_crop_fraction=0.7,
        )
        n_frames = 19
    _, dataset = naturalistic_movies(
        dt=args.dt, n_sequences=1, n_frames=n_frames, **dataset_kwargs
    )
    movie = concatenated_movies(
        dataset, n_concat=args.n_concat, n_sequences=1, seed=args.seed
    )[0]
    logger.info(
        "stimulus: %d frames (%.2f s)", movie.shape[0], movie.shape[0] * args.dt
    )

    cell_types, current_based, conductance_based = central_cell_traces(
        net, twin, movie, args.dt, device=args.device
    )
    r2 = plot_traces(
        cell_types, current_based, conductance_based, args.dt, args.out,
        n_columns=args.n_columns,
    )

    worst = sorted(r2.items(), key=lambda kv: kv[1])[:8]
    print(f"wrote {args.out}")
    print(f"central-neuron R2: median {np.median(list(r2.values())):.4f}, "
          f"{sum(1 for v in r2.values() if v > 0.9)}/{len(r2)} above 0.9")
    print(f"worst: {[(k, round(v, 3)) for k, v in worst]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("flyvis").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="flow/0000/000")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--extent", type=int, default=None)
    parser.add_argument("--params", default="graphs_data/fly/conductance_twin")
    parser.add_argument("--dt", type=float, default=1 / 100)
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument("--n-concat", type=int, default=16)
    parser.add_argument("--n-columns", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="assets/conductance_twin_traces.png")
    main(parser.parse_args())
