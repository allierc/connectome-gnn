"""Derive a conductance-based twin of the pretrained current-based flyvis model.

The pretrained flyvis networks integrate a passive point-neuron equation in
which synaptic input is a *current*,

    tau_i dv_i/dt = -(v_i - v_rest_i) + sum_j s_ij alpha_ij N_ij relu(v_j) + e_i

with the sign ``s`` fixed by the connectome. Its conductance-based twin instead
drives the voltage towards a reversal potential,

    tau_i dv_i/dt = -(v_i - v_rest_i) + sum_j G_ij relu(v_j) (E_ij - v_i) + e_i

which shunts: gain and membrane time constant are both divided by
``1 + sum_j G_ij relu(v_j)``, and the voltage cannot leave the convex hull of
the reversal potentials.

This module fits the peak conductances ``G`` so that the twin reproduces the
pretrained model's voltage trajectories under naturalistic stimuli, without
retraining on the optic-flow task. Three stages of increasing cost:

1. :func:`initial_conductances` — closed form. Linearizing ``(E - v_i)`` about
   the mean postsynaptic voltage gives ``G = alpha N / |E - mean_v|``, exact to
   first order.
2. :func:`fit_synaptic_currents` — *convex*. Along the teacher's trajectories
   the synaptic current is linear in ``G``, and the shared conductances decouple
   across postsynaptic cell types, so the optimum follows from one non-negative
   least-squares solve per cell type. No gradient descent, no local optima.
   This does nearly all of the work.
3. :func:`fit_rollouts` — truncated backpropagation through time on free
   rollouts, which corrects the drift that the teacher-forced stage 2 cannot
   observe.

flyvis is used only to read the pretrained parameters and to render Sintel; both
are public API, so this runs against a stock ``pip install flyvis``. Every
rollout — teacher and twin, fitting and evaluation — uses flyvis-gnn's own
integrators, so the resulting :class:`FlyVisConductanceODEParams` needs no
flyvis at all.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from connectome_gnn.generators.flyvis_conductance_ode import FlyVisConductanceODE
from connectome_gnn.generators.flyvis_ode import FlyVisODE
from connectome_gnn.generators.ode_params import (
    FlyVisConductanceODEParams,
    FlyVisODEParams,
)
from connectome_gnn.neuron_state import NeuronState

logger = logging.getLogger(__name__)

GREY = 0.5  # the value flyvis' steady_state drives the photoreceptors with


# ---------------------------------------------------------------------------
# connectome bookkeeping
# ---------------------------------------------------------------------------


def _decode(values) -> List[str]:
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


@dataclass
class ConductanceIndex:
    """Index structures the conductance fit operates on.

    Attributes:
        source_index / target_index: (E,) node indices per edge.
        edge_group: (E,) index of the shared conductance parameter, one per
            (presynaptic type, postsynaptic type) pair.
        edge_syn_count: (E,) mean synapse count.
        edge_weight: (E,) the teacher's signed current-based weight.
        group_reversal: (n_groups,) reversal potential per group.
        n_groups: number of shared conductance parameters.
        node_type: (N,) cell type index per node.
        cell_types: cell type names, indexed by type index.
        type_nodes / type_groups: per cell type, its node and group indices.
        type_pairs: per cell type, the (n_nodes, n_groups) matrix of indices
            into the compact (node, presynaptic group) layout.
        pair_index: (E,) index of the (node, group) pair each edge scatters to.
        n_pairs: size of that compact layout.
        input_index: (n_input_cell_types, n_hexals) stimulus injection sites.
    """

    source_index: torch.Tensor
    target_index: torch.Tensor
    edge_group: torch.Tensor
    edge_syn_count: torch.Tensor
    edge_weight: torch.Tensor
    group_reversal: torch.Tensor
    n_groups: int
    node_type: torch.Tensor
    n_nodes: int
    cell_types: List[str] = field(repr=False, default_factory=list)
    type_nodes: List[torch.Tensor] = field(repr=False, default_factory=list)
    type_groups: List[torch.Tensor] = field(repr=False, default_factory=list)
    type_pairs: List[torch.Tensor] = field(repr=False, default_factory=list)
    pair_index: torch.Tensor = None
    n_pairs: int = 0
    input_index: torch.Tensor = None


def build_index(net, reversal_exc: float, reversal_inh: float) -> ConductanceIndex:
    """Precompute the index structures from a trained current-based network.

    Args:
        net: trained flyvis Network with the stock current-based dynamics.
        reversal_exc: excitatory reversal potential.
        reversal_inh: inhibitory reversal potential.

    Returns:
        A :class:`ConductanceIndex`.
    """
    params = net._param_api()
    device = params.nodes.bias.device
    edges = net.connectome.edges

    source_index = torch.tensor(edges.source_index[:], device=device)
    target_index = torch.tensor(edges.target_index[:], device=device)

    # shared conductances are grouped by (presynaptic type, postsynaptic type),
    # matching how the pretrained model shares its synaptic strengths
    source_type = np.asarray(_decode(edges.source_type[:]))
    target_type = np.asarray(_decode(edges.target_type[:]))
    pairs = np.char.add(np.char.add(source_type, "->"), target_type)
    _, edge_group_np = np.unique(pairs, return_inverse=True)
    edge_group = torch.as_tensor(edge_group_np, dtype=torch.long, device=device)
    n_groups = int(edge_group.max()) + 1

    edge_sign = params.edges.sign.detach()
    edge_syn_count = params.edges.syn_count.detach()
    edge_weight = (edge_sign * edge_syn_count * params.edges.syn_strength).detach()

    group_sign = torch.zeros(n_groups, device=device)
    group_sign[edge_group] = edge_sign
    group_reversal = torch.where(
        group_sign > 0,
        torch.full_like(group_sign, float(reversal_exc)),
        torch.full_like(group_sign, float(reversal_inh)),
    )

    cell_types = _decode(net.connectome.nodes.type[:])
    unique_types = list(dict.fromkeys(cell_types))
    type_to_index = {name: i for i, name in enumerate(unique_types)}
    node_type = torch.tensor(
        [type_to_index[name] for name in cell_types], dtype=torch.long, device=device
    )
    n_nodes = len(cell_types)

    group_target_type = torch.zeros(n_groups, dtype=torch.long, device=device)
    group_target_type[edge_group] = node_type[target_index]

    # compact (node, presynaptic group) layout: each node gets a slot only for
    # the groups that actually target its cell type
    type_nodes, type_groups, type_pairs = [], [], []
    group_rank = torch.full((n_groups,), -1, dtype=torch.long, device=device)
    node_offset = torch.zeros(n_nodes, dtype=torch.long, device=device)
    offset = 0
    for type_idx in range(len(unique_types)):
        groups = torch.nonzero(group_target_type == type_idx, as_tuple=False).flatten()
        nodes = torch.nonzero(node_type == type_idx, as_tuple=False).flatten()
        group_rank[groups] = torch.arange(len(groups), device=device)
        node_offset[nodes] = offset + torch.arange(
            len(nodes), device=device
        ) * len(groups)
        type_pairs.append(
            (offset + torch.arange(len(nodes) * len(groups), device=device)).reshape(
                len(nodes), len(groups)
            )
        )
        offset += len(nodes) * len(groups)
        type_nodes.append(nodes)
        type_groups.append(groups)

    return ConductanceIndex(
        source_index=source_index,
        target_index=target_index,
        edge_group=edge_group,
        edge_syn_count=edge_syn_count,
        edge_weight=edge_weight,
        group_reversal=group_reversal,
        n_groups=n_groups,
        node_type=node_type,
        n_nodes=n_nodes,
        cell_types=unique_types,
        type_nodes=type_nodes,
        type_groups=type_groups,
        type_pairs=type_pairs,
        pair_index=node_offset[target_index] + group_rank[edge_group],
        n_pairs=offset,
        input_index=torch.as_tensor(
            np.asarray(net.stimulus.input_index), dtype=torch.long, device=device
        ),
    )


# ---------------------------------------------------------------------------
# stimuli and rollouts, on flyvis-gnn's own integrators
# ---------------------------------------------------------------------------


def naturalistic_movies(
    dt: float = 1 / 100,
    n_sequences: Optional[int] = 16,
    n_frames: int = 40,
    seed: int = 0,
    dataset=None,
    **dataset_kwargs,
) -> Tuple[List[torch.Tensor], object]:
    """Naturalistic (Sintel) clips to fit and evaluate on.

    Args:
        dt: sampling/integration step.
        n_sequences: number of clips to draw; None uses all.
        n_frames: frames per clip.
        seed: seed of the subsample.
        dataset: optional preconstructed AugmentedSintel.
        **dataset_kwargs: overrides for AugmentedSintel.

    Returns:
        (movies, dataset), movies being (frames, 1, hexals) tensors.
    """
    from flyvis.datasets.sintel import AugmentedSintel

    if dataset is None:
        config = dict(
            tasks=["lum"],
            interpolate=False,
            boxfilter={"extent": 15, "kernel_size": 13},
            temporal_split=True,
            n_frames=n_frames,
            dt=dt,
            augment=False,
        )
        config.update(dataset_kwargs)
        dataset = AugmentedSintel(**config)

    indices = np.arange(len(dataset))
    if n_sequences is not None and n_sequences < len(indices):
        indices = np.random.default_rng(seed).choice(
            indices, size=n_sequences, replace=False
        )
    return [dataset[int(i)]["lum"] for i in indices], dataset


def concatenated_movies(
    dataset, n_concat: int = 8, n_sequences: Optional[int] = 4, seed: int = 0
) -> List[torch.Tensor]:
    """Join short clips end to end into long stimuli, without re-rendering.

    Where the dataset bins one rendered sequence into several consecutive
    temporal splits those are joined in order and the result is continuous;
    otherwise clips are joined across scenes, so the stimulus contains cuts.
    Both networks see the identical input either way.

    Args:
        dataset: an AugmentedSintel built with temporal_split=True.
        n_concat: clips per long stimulus.
        n_sequences: number of long stimuli; None returns all.
        seed: seed of the subsample.

    Returns:
        List of (frames, 1, hexals) tensors.

    Raises:
        ValueError: if the dataset was not built with temporal_split=True.
    """
    if not getattr(dataset, "temporal_split", False):
        raise ValueError("concatenated_movies requires temporal_split=True")

    frame = dataset.arg_df.reset_index(drop=True)
    keys = [c for c in frame.columns if c not in ("temporal_split_index", "index")]
    ordered: List[int] = []
    for _, rows in frame.groupby(keys, sort=False):
        ordered.extend(rows.sort_values("temporal_split_index").index.tolist())

    groups = [
        ordered[start : start + n_concat]
        for start in range(0, len(ordered) - n_concat + 1, n_concat)
    ]
    if n_sequences is not None and n_sequences < len(groups):
        chosen = np.random.default_rng(seed).choice(
            len(groups), size=n_sequences, replace=False
        )
        groups = [groups[int(i)] for i in chosen]

    return [
        torch.cat([dataset[int(i)]["lum"] for i in group], dim=0) for group in groups
    ]


def stimulus_from_movie(
    movie: torch.Tensor, input_index: torch.Tensor, n_nodes: int
) -> torch.Tensor:
    """Map a hexagonal movie onto the whole network's drive.

    Args:
        movie: (frames, 1, hexals) or (frames, hexals) luminance.
        input_index: (n_input_cell_types, hexals) node indices to write to.
        n_nodes: number of neurons.

    Returns:
        (frames, n_nodes) drive, zero everywhere but the photoreceptors.
    """
    if movie.dim() == 3:
        movie = movie[:, 0]
    stimulus = torch.zeros(
        movie.shape[0], n_nodes, device=movie.device, dtype=movie.dtype
    )
    stimulus[:, input_index.reshape(-1)] = movie.repeat(1, input_index.shape[0])
    return stimulus


def grey_stimulus(input_index: torch.Tensor, n_nodes: int, device, value=GREY):
    """Constant mid-grey drive, as used for the pre-stimulus steady state."""
    stimulus = torch.zeros(n_nodes, device=device)
    stimulus[input_index.reshape(-1)] = value
    return stimulus


def rollout(
    ode, params, stimulus: torch.Tensor, voltage: torch.Tensor, delta_t: float,
    grad: bool = False, neuron_type: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward-Euler rollout, reporting the state after each frame's update.

    Args:
        ode: FlyVisODE or FlyVisConductanceODE.
        params: matching ODE params (for edge_index).
        stimulus: (T, N) drive per frame.
        voltage: (N,) initial voltage.
        delta_t: integration step.
        grad: keep the graph (for backpropagation through time).
        neuron_type: (N,) cell type index. FlyVisODE reads it but only uses it
            for its per-type-gain variants, so zeros are fine for the plain
            model and the conductance ODE ignores it entirely.

    Returns:
        (T, N) voltages.
    """
    if neuron_type is None:
        neuron_type = torch.zeros_like(voltage, dtype=torch.long)
    state = NeuronState(
        voltage=voltage, stimulus=stimulus[0], neuron_type=neuron_type
    )
    out = []
    with torch.set_grad_enabled(grad):
        for frame in range(stimulus.shape[0]):
            state.stimulus = stimulus[frame]
            dv = ode(state, params.edge_index)
            state.voltage = state.voltage + delta_t * dv.squeeze(-1)
            out.append(state.voltage)
    return torch.stack(out)


def steady_state(ode, params, input_index, n_nodes, delta_t, t_pre=1.0, value=GREY):
    """Settle the network under constant grey input, as flyvis does before a clip.

    Args:
        ode: FlyVisODE or FlyVisConductanceODE.
        params: matching ODE params.
        input_index: (n_input_cell_types, hexals) stimulus injection sites.
        n_nodes: number of neurons.
        delta_t: integration step.
        t_pre: duration of the grey pre-stimulus.
        value: grey level.

    Returns:
        (N,) settled voltage.
    """
    frames = max(int(t_pre / delta_t), 1)
    stimulus = grey_stimulus(input_index, n_nodes, params.V_i_rest.device, value)
    stimulus = stimulus.expand(frames, n_nodes)
    return rollout(ode, params, stimulus, params.V_i_rest.clone(), delta_t)[-1]


def current_based_ode(net, device="cpu") -> Tuple[FlyVisODE, FlyVisODEParams]:
    """The pretrained current-based model, as a flyvis-gnn integrator.

    Args:
        net: trained flyvis Network with the stock dynamics.
        device: device to build on.

    Returns:
        (ode, params).
    """
    params = FlyVisODEParams.from_flyvis_network(net, device=device)
    ode = FlyVisODE(ode_params=params, model_type="flyvis_A", device=device)
    return ode, params


# ---------------------------------------------------------------------------
# stage 1: closed form
# ---------------------------------------------------------------------------


def initial_conductances(
    index: ConductanceIndex, mean_voltage: torch.Tensor
) -> torch.Tensor:
    """Closed-form peak conductances matching the teacher to first order.

    Equating ``G (E - mean_v_i)`` with the teacher's ``sign * alpha * N`` at the
    postsynaptic mean voltage gives ``G = |w| / |E - mean_v_i|``. The signs
    cancel because ``E - mean_v_i`` carries the connectome sign by construction.

    Args:
        index: connectome bookkeeping.
        mean_voltage: (N,) mean teacher voltage per neuron.

    Returns:
        (E,) per-edge peak conductances, strictly positive.
    """
    reversal = index.group_reversal[index.edge_group]
    driving_force = (reversal - mean_voltage[index.target_index]).abs().clamp(min=1e-3)
    return (index.edge_weight.abs() / driving_force).clamp(min=1e-12)


def group_conductances(index: ConductanceIndex, edge_conductance: torch.Tensor):
    """Per-group conductance scale, factoring the synapse count back out.

    Args:
        index: connectome bookkeeping.
        edge_conductance: (E,) per-edge conductances.

    Returns:
        (n_groups,) mean of ``G / N`` within each group.
    """
    scale = edge_conductance / index.edge_syn_count.clamp(min=1e-12)
    total = torch.zeros(index.n_groups, device=scale.device)
    count = torch.zeros(index.n_groups, device=scale.device)
    total.scatter_add_(0, index.edge_group, scale)
    count.scatter_add_(0, index.edge_group, torch.ones_like(scale))
    return total / count.clamp(min=1)


def conductance_pruning_stats(index: ConductanceIndex, conductance: torch.Tensor,
                             threshold: float = 1e-10) -> Dict[str, float]:
    """How much of the connectome the non-negative fit drove to zero.

    Stage 2 constrains G >= 0 while the sign of the driving force is fixed by the
    connectome. Where a shared group's contribution is better explained with the
    opposite polarity — or is collinear with another group onto the same cell
    type — the NNLS clamps it to zero. The twin is then not merely "the same
    network with conductances": it is also effectively sparser, and a GNN trained
    on its rollouts is being asked to recover a connectivity in which those edges
    really do carry no weight.

    That is self-consistent (connectivity_r2 compares against the twin's own W),
    but it makes numbers from twin rollouts not directly comparable with numbers
    from current-based rollouts, so it is worth recording per derivation rather
    than left to be discovered.

    Args:
        index: connectome bookkeeping.
        conductance: (E,) per-edge peak conductances.
        threshold: per-group scale at or below which a group counts as pruned.

    Returns:
        Dict with the pruned group and edge counts, and the share of the
        teacher's total |W| those edges carried.
    """
    scale = group_conductances(index, conductance)
    pruned_groups = scale <= threshold
    pruned_edges = pruned_groups[index.edge_group]
    weight = index.edge_weight.abs()
    total = float(weight.sum())
    return {
        "n_groups": int(index.n_groups),
        "n_pruned_groups": int(pruned_groups.sum()),
        "pruned_group_fraction": float(pruned_groups.float().mean()),
        "pruned_edge_fraction": float(pruned_edges.float().mean()),
        "pruned_share_of_teacher_weight": float(weight[pruned_edges].sum() / total)
        if total > 0
        else 0.0,
    }


# ---------------------------------------------------------------------------
# stage 2: convex synaptic-current matching
# ---------------------------------------------------------------------------


def fit_synaptic_currents(
    index: ConductanceIndex,
    voltages: Sequence[torch.Tensor],
    prior: torch.Tensor,
    time_chunk: int = 8,
    ridge: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Fit shared conductances by matching the teacher's synaptic currents.

    Along the teacher's trajectories the conductance-based synaptic current into
    cell *i*,

        I_i = sum_k a_k [ sum_{e in k, e -> i} N_e relu(v_source) ] (E_k - v_i),

    is *linear* in the shared conductance scales ``a_k``, one per (presynaptic
    type, postsynaptic type) pair. Matching it to the teacher's synaptic current
    matches the two velocity fields exactly, because both models share tau,
    v_rest and the drive. Since a group targets exactly one postsynaptic cell
    type the problem decouples into one small non-negative least-squares problem
    per cell type, solved here in closed form from accumulated normal equations.

    Args:
        index: connectome bookkeeping.
        voltages: teacher trajectories, each (T, N).
        prior: (E,) per-edge conductances the ridge pulls towards, typically the
            closed-form initialization.
        time_chunk: frames per accumulation step; trades memory for speed and
            does not change the result.
        ridge: Tikhonov weight relative to the mean diagonal of the normal
            equations.

    Returns:
        (per-edge conductances, diagnostics) where diagnostics reports the
        fraction of the teacher's synaptic-current variance left unexplained.
    """
    from scipy.optimize import nnls

    device = index.edge_weight.device
    n_types = len(index.cell_types)
    prior_group = group_conductances(index, prior.to(device)).cpu().double()

    gram = [torch.zeros(len(g), len(g), dtype=torch.float64) for g in index.type_groups]
    rhs = [torch.zeros(len(g), dtype=torch.float64) for g in index.type_groups]
    energy = torch.zeros(n_types, dtype=torch.float64)

    for trajectory in voltages:
        for start in range(0, trajectory.shape[0], time_chunk):
            v = trajectory[start : start + time_chunk].to(device)
            with torch.no_grad():
                release = torch.relu(v)[:, index.source_index]
                # the teacher's synaptic current into every node
                target = torch.zeros(v.shape[0], index.n_nodes, device=device)
                target.scatter_add_(
                    -1,
                    index.target_index.expand(*release.shape),
                    release * index.edge_weight,
                )
                # synaptic activation per (node, presynaptic group)
                activation = torch.zeros(v.shape[0], index.n_pairs, device=device)
                activation.scatter_add_(
                    -1,
                    index.pair_index.expand(*release.shape),
                    release * index.edge_syn_count,
                )
                del release

            for type_idx in range(n_types):
                pairs = index.type_pairs[type_idx]
                if pairs.numel() == 0:
                    continue
                nodes = index.type_nodes[type_idx]
                groups = index.type_groups[type_idx]
                design = activation[:, pairs.reshape(-1)].reshape(
                    v.shape[0], len(nodes), len(groups)
                )
                driving_force = (
                    index.group_reversal[groups] - v[:, nodes].unsqueeze(-1)
                )
                # normal equations accumulate on the CPU in double precision:
                # the accelerator may not support float64 and these are small
                design = (
                    (design * driving_force).reshape(-1, len(groups)).cpu().double()
                )
                observed = target[:, nodes].reshape(-1).cpu().double()
                gram[type_idx] += design.T @ design
                rhs[type_idx] += design.T @ observed
                energy[type_idx] += observed.pow(2).sum()
            del activation, target

    solution = prior_group.clone()
    per_type: Dict[str, float] = {}
    total_residual, total_energy = 0.0, 0.0
    for type_idx in range(n_types):
        groups = index.type_groups[type_idx].cpu()
        if groups.numel() == 0:
            continue
        g, c = gram[type_idx], rhs[type_idx]
        lam = ridge * torch.diagonal(g).mean().clamp(min=1e-30)
        regularized = g + lam * torch.eye(len(groups), dtype=torch.float64)
        centered = c + lam * prior_group[groups]
        # min a'Ga - 2a'c  s.t. a >= 0  <=>  min ||L'a - L^-1 c||^2 s.t. a >= 0
        chol = torch.linalg.cholesky(regularized)
        z = torch.linalg.solve_triangular(
            chol, centered.unsqueeze(-1), upper=False
        ).squeeze(-1)
        fitted, _ = nnls(chol.T.numpy(), z.numpy())
        fitted = torch.as_tensor(fitted, dtype=torch.float64).clamp(min=1e-12)
        solution[groups] = fitted

        residual = (fitted @ g @ fitted - 2 * fitted @ c + energy[type_idx]).clamp(min=0)
        total_residual += float(residual)
        total_energy += float(energy[type_idx])
        per_type[index.cell_types[type_idx]] = float(
            residual / energy[type_idx].clamp(min=1e-30)
        )

    conductance = (
        solution.to(device).float()[index.edge_group] * index.edge_syn_count
    ).clamp(min=1e-12)
    diagnostics = {
        "unexplained_current_fraction": total_residual / max(total_energy, 1e-30),
        "per_type": per_type,
    }
    return conductance, diagnostics


# ---------------------------------------------------------------------------
# stage 3: rollout matching
# ---------------------------------------------------------------------------


def _bptt_windows(n_frames: int, window: int) -> Iterator[slice]:
    for start in range(0, n_frames, window):
        yield slice(start, min(start + window, n_frames))


def fit_rollouts(
    twin_params: FlyVisConductanceODEParams,
    index: ConductanceIndex,
    teacher_voltages: Sequence[torch.Tensor],
    stimuli: Sequence[torch.Tensor],
    delta_t: float,
    epochs: int = 8,
    bptt_steps: int = 20,
    lr: float = 5e-4,
    lr_rest: float = 1e-4,
    grad_clip: Optional[float] = 1.0,
    share: bool = True,
    log_every: int = 0,
) -> Dict[str, List[float]]:
    """Match free twin rollouts to the teacher's, by truncated BPTT.

    Stage 2 is teacher-forced: it scores the twin's velocity field only on states
    the teacher visits. At run time the twin is driven by its own state, so the
    residual displaces the trajectory into states the fit never scored and the
    errors compound. This stage optimizes the quantity that actually matters —
    the free rollout — with the state detached across windows so that memory and
    gradient path length stay bounded.

    Conductances and time constants are stepped in log space, so ``lr`` is a
    multiplicative step; the additive resting potentials get ``lr_rest``.
    Modifies ``twin_params`` in place.

    Args:
        twin_params: the twin, updated in place.
        index: connectome bookkeeping.
        teacher_voltages: per sequence, the teacher's (T, N) trajectory.
        stimuli: per sequence, the matching (T, N) drive.
        delta_t: integration step.
        epochs: passes over the sequences.
        bptt_steps: truncation length.
        lr: learning rate for the log-parametrized conductances and taus.
        lr_rest: learning rate for the resting potentials.
        grad_clip: global gradient-norm clip, or None.
        share: keep the pretrained model's parameter sharing — one conductance
            scale per (presynaptic type, postsynaptic type) pair and one time
            constant and resting potential per cell type, so the twin has as
            many free parameters as the model it reparametrizes (hundreds). Set
            False to free every synapse and every neuron individually, which
            fits better but is a different, much larger model class.
        log_every: log the running loss every this many sequences; 0 to silence.

    Returns:
        Dict with the per-sequence loss history.
    """
    device = twin_params.G.device

    if share:
        # one scale per (presynaptic type, postsynaptic type) pair; the synapse
        # count keeps modulating individual synapses, exactly as in the teacher
        raw_conductance = (
            group_conductances(index, twin_params.G).clamp(min=1e-12).log()
        )
        raw_tau = torch.stack([
            twin_params.tau_i[nodes].mean() for nodes in index.type_nodes
        ]).clamp(min=1e-12).log()
        raw_rest = torch.stack([
            twin_params.V_i_rest[nodes].mean() for nodes in index.type_nodes
        ])

        def conductance_of(raw):
            return raw.exp()[index.edge_group] * index.edge_syn_count

        def node_of(raw):
            return raw[index.node_type]

        def tau_of(raw):
            return raw.exp()[index.node_type]
    else:
        raw_conductance = twin_params.G.clamp(min=1e-12).log()
        raw_tau = twin_params.tau_i.clamp(min=1e-12).log()
        raw_rest = twin_params.V_i_rest

        def conductance_of(raw):
            return raw.exp()

        def node_of(raw):
            return raw

        def tau_of(raw):
            return raw.exp()

    raw_conductance = raw_conductance.clone().requires_grad_(True)
    raw_tau = raw_tau.clone().requires_grad_(True)
    raw_rest = raw_rest.clone().requires_grad_(True)
    logger.info(
        "rollout stage over %d free parameters (%s)",
        raw_conductance.numel() + raw_tau.numel() + raw_rest.numel(),
        "shared as in the teacher" if share else "one per synapse and neuron",
    )
    optimizer = torch.optim.Adam([
        {"params": [raw_conductance, raw_tau], "lr": lr},
        {"params": [raw_rest], "lr": lr_rest},
    ])

    # normalize the error per cell type so all of them contribute comparably
    stacked = torch.cat([v.to(device) for v in teacher_voltages])
    scale = torch.ones(index.n_nodes, device=device)
    for type_idx, nodes in enumerate(index.type_nodes):
        if nodes.numel():
            scale[nodes] = stacked[:, nodes].std().clamp(min=1e-3)
    del stacked

    working = FlyVisConductanceODEParams(
        tau_i=twin_params.tau_i,
        V_i_rest=twin_params.V_i_rest,
        edge_index=twin_params.edge_index,
        G=twin_params.G,
        E_rev=twin_params.E_rev,
        input_index=twin_params.input_index,
        reversal_exc=twin_params.reversal_exc,
        reversal_inh=twin_params.reversal_inh,
    )
    ode = FlyVisConductanceODE(ode_params=working, device=device, delta_t=delta_t)

    history: List[float] = []

    def materialize():
        working.G = conductance_of(raw_conductance)
        working.tau_i = tau_of(raw_tau)
        working.V_i_rest = node_of(raw_rest)

    def fit_one(teacher, stimulus):
        """One truncated-BPTT pass over a sequence; returns its mean loss."""
        optimizer.zero_grad(set_to_none=True)
        materialize()
        with torch.no_grad():
            state = steady_state(ode, working, index.input_index, index.n_nodes, delta_t)

        loss_value = 0.0
        for window in _bptt_windows(teacher.shape[0], bptt_steps):
            # re-derive the parameters so each window gets its own graph
            materialize()
            voltages = rollout(ode, working, stimulus[window], state, delta_t, grad=True)
            loss = ((voltages - teacher[window]) / scale).pow(2).mean()
            weight = (window.stop - window.start) / teacher.shape[0]
            (loss * weight).backward()
            loss_value += float(loss.detach()) * weight
            state = voltages[-1].detach()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_([raw_conductance, raw_tau, raw_rest], grad_clip)
        optimizer.step()
        return loss_value

    # Enable grad explicitly rather than relying on the ambient mode: the data
    # generator disables it globally before it ever gets here, which would leave
    # the materialized parameters without a grad_fn and break BPTT.
    with torch.enable_grad():
        for epoch in range(epochs):
            for sequence, (teacher, stimulus) in enumerate(zip(teacher_voltages, stimuli)):
                loss_value = fit_one(teacher.to(device), stimulus.to(device))
                history.append(loss_value)
                if log_every and sequence % log_every == 0:
                    logger.info(
                        "epoch %d sequence %d normalized voltage MSE %.5f",
                        epoch, sequence, loss_value,
                    )

    with torch.no_grad():
        twin_params.G = conductance_of(raw_conductance).detach()
        twin_params.tau_i = tau_of(raw_tau).detach()
        twin_params.V_i_rest = node_of(raw_rest).detach()
        # W is the small-signal weight derived from G, E and V_rest, all of
        # which just moved
        twin_params.refresh_effective_weights()
    return {"loss": history}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def evaluate_twin(
    twin_params: FlyVisConductanceODEParams,
    index: ConductanceIndex,
    teacher_voltages: Sequence[torch.Tensor],
    stimuli: Sequence[torch.Tensor],
    delta_t: float,
) -> Dict[str, object]:
    """Voltage R^2 of free twin rollouts against the teacher's.

    Args:
        twin_params: the twin.
        index: connectome bookkeeping.
        teacher_voltages: per sequence, the teacher's (T, N) trajectory.
        stimuli: per sequence, the matching (T, N) drive.
        delta_t: integration step.

    Returns:
        Overall and per-cell-type R^2, and the voltage range each model visits.
    """
    device = twin_params.G.device
    ode = FlyVisConductanceODE(ode_params=twin_params, device=device, delta_t=delta_t)
    n_types = len(index.cell_types)
    residual = torch.zeros(n_types, dtype=torch.float64)
    variance = torch.zeros(n_types, dtype=torch.float64)
    twin_range = [float("inf"), -float("inf")]
    teacher_range = [float("inf"), -float("inf")]

    for teacher, stimulus in zip(teacher_voltages, stimuli):
        teacher = teacher.to(device)
        stimulus = stimulus.to(device)
        with torch.no_grad():
            state = steady_state(
                ode, twin_params, index.input_index, index.n_nodes, delta_t
            )
            predicted = rollout(ode, twin_params, stimulus, state, delta_t)
        twin_range = [
            min(twin_range[0], float(predicted.min())),
            max(twin_range[1], float(predicted.max())),
        ]
        teacher_range = [
            min(teacher_range[0], float(teacher.min())),
            max(teacher_range[1], float(teacher.max())),
        ]
        error = (predicted - teacher).cpu().double()
        centered = (teacher - teacher.mean(0)).cpu().double()
        for type_idx, nodes in enumerate(index.type_nodes):
            nodes = nodes.cpu()
            residual[type_idx] += error[:, nodes].pow(2).sum()
            variance[type_idx] += centered[:, nodes].pow(2).sum()

    r2_per_type = (1 - residual / variance.clamp(min=1e-30)).numpy()
    return {
        "r2": float(1 - residual.sum() / variance.sum().clamp(min=1e-30)),
        "r2_per_type": dict(zip(index.cell_types, map(float, r2_per_type))),
        "teacher_voltage_range": tuple(teacher_range),
        "twin_voltage_range": tuple(twin_range),
    }


def drift_profile(
    twin_params: FlyVisConductanceODEParams,
    index: ConductanceIndex,
    teacher_voltages: Sequence[torch.Tensor],
    stimuli: Sequence[torch.Tensor],
    delta_t: float,
    n_windows: int = 8,
) -> Dict[str, object]:
    """Resolve the twin's R^2 against elapsed rollout time.

    Separates a constant error floor from genuine divergence. Sequences are
    truncated to the shortest so the windows line up.

    Args:
        twin_params: the twin.
        index: connectome bookkeeping.
        teacher_voltages: per sequence, the teacher's (T, N) trajectory.
        stimuli: per sequence, the matching (T, N) drive.
        delta_t: integration step.
        n_windows: number of equal time windows.

    Returns:
        Window mid-times, the R^2 within each, and the rollout duration.
    """
    device = twin_params.G.device
    ode = FlyVisConductanceODE(ode_params=twin_params, device=device, delta_t=delta_t)
    n_frames = min(int(v.shape[0]) for v in teacher_voltages)
    edges = np.linspace(0, n_frames, n_windows + 1).astype(int)
    residual = torch.zeros(n_windows, dtype=torch.float64)
    variance = torch.zeros(n_windows, dtype=torch.float64)

    for teacher, stimulus in zip(teacher_voltages, stimuli):
        teacher = teacher[:n_frames].to(device)
        stimulus = stimulus[:n_frames].to(device)
        with torch.no_grad():
            state = steady_state(
                ode, twin_params, index.input_index, index.n_nodes, delta_t
            )
            predicted = rollout(ode, twin_params, stimulus, state, delta_t)
        error = (predicted - teacher).cpu().double()
        centered = (teacher - teacher.mean(0)).cpu().double()
        for window in range(n_windows):
            span = slice(edges[window], edges[window + 1])
            residual[window] += error[span].pow(2).sum()
            variance[window] += centered[span].pow(2).sum()

    r2 = (1 - residual / variance.clamp(min=1e-30)).numpy()
    return {
        "time": [float(t) for t in (edges[:-1] + edges[1:]) / 2 * delta_t],
        "r2": [float(v) for v in r2],
        "duration": n_frames * delta_t,
    }


def shunting_statistics(
    twin_params: FlyVisConductanceODEParams,
    index: ConductanceIndex,
    stimuli: Sequence[torch.Tensor],
    delta_t: float,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> Dict[str, Dict[str, float]]:
    """How strongly each cell type is shunted by its synaptic input.

    The total synaptic conductance G divides both the gain and the membrane time
    constant, so its distribution says how far the twin has moved from the
    linear, current-based regime.

    Args:
        twin_params: the twin.
        index: connectome bookkeeping.
        stimuli: per sequence, the (T, N) drive.
        delta_t: integration step.
        quantiles: quantiles of G to report per cell type.

    Returns:
        Per cell type, quantiles of G and the implied effective time constant.
    """
    device = twin_params.G.device
    ode = FlyVisConductanceODE(ode_params=twin_params, device=device, delta_t=delta_t)
    samples = []
    for stimulus in stimuli:
        stimulus = stimulus.to(device)
        with torch.no_grad():
            state = steady_state(
                ode, twin_params, index.input_index, index.n_nodes, delta_t
            )
            voltages = rollout(ode, twin_params, stimulus, state, delta_t)
            release = torch.relu(voltages)[:, index.source_index]
            total = torch.zeros_like(voltages)
            total.scatter_add_(
                -1,
                index.target_index.expand(*release.shape),
                release * twin_params.G,
            )
        samples.append(total.cpu())
    conductance = torch.cat(samples)

    quantile_tensor = torch.tensor(list(quantiles))
    result: Dict[str, Dict[str, float]] = {}
    for type_idx, name in enumerate(index.cell_types):
        nodes = index.type_nodes[type_idx].cpu()
        values = conductance[:, nodes].reshape(-1)
        levels = torch.quantile(values, quantile_tensor)
        entry = {f"G_q{int(p * 100)}": float(v) for p, v in zip(quantiles, levels)}
        entry["tau"] = float(twin_params.tau_i[nodes[0]])
        entry["tau_eff_median"] = entry["tau"] / (1 + entry["G_q50"])
        result[name] = entry
    return result


# ---------------------------------------------------------------------------
# top-level derivation
# ---------------------------------------------------------------------------


def derive_conductance_twin(
    net,
    movies: Sequence[torch.Tensor],
    delta_t: float = 1 / 100,
    reversal_margin: Union[float, Tuple[float, float]] = (0.4, 1.0),
    reversal_exc: Optional[float] = None,
    reversal_inh: Optional[float] = None,
    epochs: int = 8,
    bptt_steps: int = 20,
    time_chunk: int = 8,
    ridge: float = 1e-6,
    share: bool = True,
    device: str = "cpu",
) -> Tuple[FlyVisConductanceODEParams, ConductanceIndex, Dict[str, object]]:
    """Derive the conductance twin of a trained current-based flyvis network.

    Runs all three stages and returns parameters that need no flyvis to
    simulate.

    Args:
        net: trained flyvis Network with the stock current-based dynamics.
        movies: naturalistic clips, each (frames, 1, hexals), to fit on.
        delta_t: integration step.
        reversal_margin: (inhibitory, excitatory) headroom between the extreme
            voltages the teacher visits and the reversal potentials, in units of
            the teacher's voltage span. Large margins recover the current-based
            model; small margins give strong shunting. The asymmetric default
            mirrors the inhibitory driving force being about half the excitatory
            one in real neurons.
        reversal_exc / reversal_inh: set these to override the margins.
        epochs: rollout-matching epochs; 0 to stop after the convex stage.
        bptt_steps: truncation length of the rollout stage.
        time_chunk: frames per accumulation step of the convex stage.
        ridge: Tikhonov weight of the convex stage.
        share: keep the pretrained model's parameter sharing in the rollout
            stage; see :func:`fit_rollouts`.
        device: device to fit on.

    Returns:
        (twin params, index, diagnostics).
    """
    teacher_ode, teacher_params = current_based_ode(net, device=device)
    n_nodes = teacher_params.V_i_rest.shape[0]
    input_index = torch.as_tensor(
        np.asarray(net.stimulus.input_index), dtype=torch.long, device=device
    )

    if float(teacher_params.tau_i.min()) < delta_t:
        raise ValueError(
            f"the shortest time constant ({float(teacher_params.tau_i.min()):.4f}) is "
            f"below delta_t ({delta_t}); flyvis would clamp it and this integrator "
            "would not, so the twin would be fitted to different dynamics"
        )

    logger.info("simulating the current-based teacher on %d clips", len(movies))
    stimuli, teacher_voltages = [], []
    initial = steady_state(
        teacher_ode, teacher_params, input_index, n_nodes, delta_t
    )
    for movie in movies:
        stimulus = stimulus_from_movie(movie.to(device), input_index, n_nodes)
        stimuli.append(stimulus)
        teacher_voltages.append(
            rollout(teacher_ode, teacher_params, stimulus, initial.clone(), delta_t)
        )

    stacked = torch.cat(teacher_voltages)
    v_min, v_max = float(stacked.min()), float(stacked.max())
    span = max(v_max - v_min, 1e-3)
    if np.isscalar(reversal_margin):
        margin_inh = margin_exc = float(reversal_margin)
    else:
        margin_inh, margin_exc = map(float, reversal_margin)
    if reversal_exc is None:
        reversal_exc = v_max + margin_exc * span
    if reversal_inh is None:
        reversal_inh = v_min - margin_inh * span
    logger.info(
        "teacher voltages in [%.3f, %.3f]; E_inh=%.3f E_exc=%.3f",
        v_min, v_max, reversal_inh, reversal_exc,
    )

    index = build_index(net, reversal_exc, reversal_inh)
    mean_voltage = stacked.mean(0)
    del stacked

    conductance = initial_conductances(index, mean_voltage)
    twin = FlyVisConductanceODEParams(
        tau_i=teacher_params.tau_i.clone(),
        V_i_rest=teacher_params.V_i_rest.clone(),
        edge_index=teacher_params.edge_index,
        G=conductance,
        E_rev=index.group_reversal[index.edge_group],
        input_index=input_index,
        reversal_exc=float(reversal_exc),
        reversal_inh=float(reversal_inh),
    ).refresh_effective_weights()
    diagnostics: Dict[str, object] = {
        "reversal_exc": float(reversal_exc),
        "reversal_inh": float(reversal_inh),
        "teacher_voltage_range": (v_min, v_max),
        "shared_parameters": bool(share),
    }

    logger.info("stage 2: convex synaptic-current matching")
    twin.G, stage2 = fit_synaptic_currents(
        index, teacher_voltages, prior=conductance,
        time_chunk=time_chunk, ridge=ridge,
    )
    twin.refresh_effective_weights()
    diagnostics["unexplained_current_fraction"] = stage2[
        "unexplained_current_fraction"
    ]
    diagnostics["pruning"] = conductance_pruning_stats(index, twin.G)
    logger.info(
        "non-negativity pruned %d/%d shared groups (%.1f%% of edges, carrying %.2f%% "
        "of the teacher's total |W|)",
        diagnostics["pruning"]["n_pruned_groups"],
        diagnostics["pruning"]["n_groups"],
        100 * diagnostics["pruning"]["pruned_edge_fraction"],
        100 * diagnostics["pruning"]["pruned_share_of_teacher_weight"],
    )
    logger.info(
        "unexplained synaptic-current variance %.5f",
        stage2["unexplained_current_fraction"],
    )

    if epochs:
        logger.info("stage 3: rollout matching (%d epochs)", epochs)
        history = fit_rollouts(
            twin, index, teacher_voltages, stimuli, delta_t,
            epochs=epochs, bptt_steps=bptt_steps, share=share,
        )
        diagnostics["rollout_loss"] = [history["loss"][0], history["loss"][-1]]
        logger.info(
            "rollout loss %.5f -> %.5f", history["loss"][0], history["loss"][-1]
        )

    return twin, index, diagnostics


def ensure_conductance_twin(
    net,
    folder: str,
    dataset=None,
    extent: Optional[int] = None,
    delta_t: float = 1 / 100,
    n_train: int = 12,
    n_test: int = 4,
    n_concat: int = 4,
    epochs: int = 6,
    write_tables_too: bool = True,
    **derive_kwargs,
) -> FlyVisConductanceODEParams:
    """Load the twin at `folder`, deriving and saving it first if it is absent.

    Lets a config point at a twin that has not been built yet: the submission
    branch stays small and whoever runs the pipeline gets the twin derived on
    first use instead of a missing-file error.

    Deriving from the caller's own ``net`` and ``dataset`` is what makes this
    safe. The alternative — reloading flyvis here — would have to guess the
    extent and the Sintel settings, and a twin derived at the wrong extent has
    the wrong number of edges. Passing the network the data will actually be
    generated with removes that class of mismatch.

    Args:
        net: the flyvis network the rollouts will be generated with.
        folder: where the twin lives, or should be written.
        dataset: an AugmentedSintel to draw the fitting clips from. Pass None
            when the run's own stimuli are not Sintel (DAVIS, say) and one will
            be built — in which case ``extent`` must be given, or the rendering
            will not match the connectome.
        extent: hexagonal extent of ``net``, used only when building a dataset.
        delta_t: integration step; the twin should be simulated at the step it
            was fitted at.
        n_train: clips to fit on.
        n_test: clips held out for the reported R².
        n_concat: clips joined per fitting sequence, so sequences outlast the
            BPTT window.
        epochs: rollout-matching epochs; 0 stops after the convex stage.
        write_tables_too: also write nodes.csv / edges.parquet / meta.json.
        **derive_kwargs: forwarded to :func:`derive_conductance_twin`.

    Returns:
        The twin, loaded from disk if it was already there.

    Raises:
        ValueError: if an existing twin at `folder` does not match ``net``.
    """
    existing = os.path.join(folder, "ode_params.pt")
    if os.path.exists(existing):
        twin = FlyVisConductanceODEParams.load(folder)
        if twin.G is None:
            raise ValueError(f"{existing} holds no conductances; it is not a twin")
        if twin.G.shape[0] != net.n_edges:
            raise ValueError(
                f"the twin at {folder} has {twin.G.shape[0]} edges but this connectome has "
                f"{net.n_edges}; delete it to re-derive, or point at the twin for this extent"
            )
        logger.info("using the conductance twin at %s", folder)
        return twin

    logger.info(
        "no conductance twin at %s — deriving one now (%d neurons, %d edges); "
        "this takes a few minutes and only happens once",
        folder, net.n_nodes, net.n_edges,
    )
    if dataset is None:
        if extent is None:
            raise ValueError(
                "ensure_conductance_twin needs either a dataset to draw clips from or the "
                "extent to build one at; a rendering that does not match the connectome "
                "would silently produce a twin with the wrong number of input hexals"
            )
        _, dataset = naturalistic_movies(
            dt=delta_t, n_sequences=1, n_frames=19, tasks=["flow"], interpolate=True,
            boxfilter=dict(extent=extent, kernel_size=13), vertical_splits=3,
            center_crop_fraction=0.7,
        )
    movies = concatenated_movies(dataset, n_concat=n_concat, n_sequences=n_train + n_test)
    if len(movies) < n_train + n_test:
        raise ValueError(
            f"the stimulus dataset yields only {len(movies)} sequences of {n_concat} joined "
            f"clips, fewer than the {n_train + n_test} the fit asks for; lower n_concat"
        )

    twin, index, diagnostics = derive_conductance_twin(
        net, movies[:n_train], delta_t=delta_t, epochs=epochs, **derive_kwargs
    )

    teacher_ode, teacher_params = current_based_ode(net)
    initial = steady_state(
        teacher_ode, teacher_params, index.input_index, index.n_nodes, delta_t
    )
    stimuli, voltages = [], []
    for movie in movies[n_train:]:
        stimulus = stimulus_from_movie(movie, index.input_index, index.n_nodes)
        stimuli.append(stimulus)
        voltages.append(
            rollout(teacher_ode, teacher_params, stimulus, initial.clone(), delta_t)
        )
    if voltages:
        diagnostics["held_out_r2"] = evaluate_twin(
            twin, index, voltages, stimuli, delta_t
        )["r2"]
        logger.info("derived twin: held-out voltage r2 %.4f", diagnostics["held_out_r2"])

    os.makedirs(folder, exist_ok=True)
    twin.save(folder)
    if write_tables_too:
        meta = write_tables(net, twin, folder)
        meta["diagnostics"] = diagnostics
        with open(os.path.join(folder, "meta.json"), "w") as handle:
            json.dump(meta, handle, indent=2)
    logger.info("saved the derived twin to %s", folder)
    return twin


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def write_tables(net, twin: FlyVisConductanceODEParams, out: str) -> Dict[str, object]:
    """Write the node and edge tables that define the simulator.

    Args:
        net: the flyvis network the twin was derived from.
        twin: the derived twin.
        out: destination folder.

    Returns:
        The metadata that was written alongside them.
    """
    import pandas as pd

    os.makedirs(out, exist_ok=True)
    nodes_dir, edges_dir = net.connectome.nodes, net.connectome.edges

    nodes = pd.DataFrame({
        "index": np.arange(twin.V_i_rest.shape[0]),
        "type": _decode(nodes_dir["type"][:]),
        "role": _decode(nodes_dir["role"][:]),
        "u": np.asarray(nodes_dir["u"][:]),
        "v": np.asarray(nodes_dir["v"][:]),
        "tau_i": twin.tau_i.cpu().numpy(),
        "V_i_rest": twin.V_i_rest.cpu().numpy(),
    })
    nodes.to_csv(os.path.join(out, "nodes.csv"), index=False)

    edges = pd.DataFrame({
        "source_index": twin.edge_index[0].cpu().numpy(),
        "target_index": twin.edge_index[1].cpu().numpy(),
        "source_type": _decode(edges_dir["source_type"][:]),
        "target_type": _decode(edges_dir["target_type"][:]),
        "du": np.asarray(edges_dir["du"][:]),
        "dv": np.asarray(edges_dir["dv"][:]),
        "n_syn": np.asarray(edges_dir["n_syn"][:]),
        "sign": np.asarray(edges_dir["sign"][:]),
        "G": twin.G.cpu().numpy(),
        "E_rev": twin.E_rev.cpu().numpy(),
    })
    try:
        edges.to_parquet(os.path.join(out, "edges.parquet"), index=False)
        edge_file = "edges.parquet"
    except ImportError:  # pyarrow/fastparquet missing
        edges.to_csv(os.path.join(out, "edges.csv.gz"), index=False)
        edge_file = "edges.csv.gz"

    meta = {
        "n_neurons": int(len(nodes)),
        "n_edges": int(len(edges)),
        "n_neuron_types": int(nodes["type"].nunique()),
        "n_input_neurons": int((nodes["role"] == "input").sum()),
        "reversal_exc": twin.reversal_exc,
        "reversal_inh": twin.reversal_inh,
        "edge_table": edge_file,
        "ode": "connectome_gnn.generators.flyvis_conductance_ode.FlyVisConductanceODE",
        "ode_params": "connectome_gnn.generators.ode_params.FlyVisConductanceODEParams",
    }
    with open(os.path.join(out, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    with open(os.path.join(out, "README.md"), "w") as handle:
        handle.write(README_TEMPLATE.format(**meta))
    return meta


README_TEMPLATE = """\
# Conductance-based flyvis twin

A {n_neurons}-neuron, {n_edges}-synapse simulator of the fly visual system in
which synaptic input enters as a conductance towards a reversal potential:

    tau_i dv_i/dt = -(v_i - v_rest_i) + sum_j G_ij relu(v_j) (E_ij - v_i) + e_i

Derived from the pretrained current-based flyvis model to reproduce its voltage
trajectories under naturalistic stimuli. **flyvis is not needed to run it** —
only torch, numpy and connectome_gnn.

## Run it

```python
import torch
from connectome_gnn.generators.flyvis_conductance_ode import FlyVisConductanceODE
from connectome_gnn.generators.ode_params import FlyVisConductanceODEParams
from connectome_gnn.neuron_state import NeuronState

dt = 0.01
params = FlyVisConductanceODEParams.load("<this folder>", device="cpu")
ode = FlyVisConductanceODE(ode_params=params, device="cpu", delta_t=dt)

# stimulus: (T, N), written onto the photoreceptors listed in params.input_index
stimulus = torch.zeros(n_frames, params.V_i_rest.shape[0])
stimulus[:, params.input_index.reshape(-1)] = movie.repeat(1, params.input_index.shape[0])

state = NeuronState(voltage=params.V_i_rest.clone(), stimulus=stimulus[0])
for frame in range(n_frames):
    state.stimulus = stimulus[frame]
    state.voltage = state.voltage + dt * ode(state, params.edge_index).squeeze(-1)
```

`delta_t` floors the effective time constant `tau/(1 + sum_j G_ij relu(v_j))`,
which is what keeps forward Euler from overshooting the fixed point. Pass the
same value you integrate with.

## Files

| file | contents |
|---|---|
| `ode_params.pt` | tensors the simulator consumes: `tau_i`, `V_i_rest` (N,), `edge_index` (2, E), `G`, `E_rev` (E,), `input_index` |
| `nodes.csv` | one row per neuron: index, cell type, role, hex column (u, v), tau, resting potential |
| `{edge_table}` | one row per synapse: source/target index and cell type, columnar offsets (du, dv), synapse count, connectome sign, peak conductance G, reversal potential E |
| `meta.json` | shapes, reversal potentials, and the fit diagnostics |

Reversal potentials: E_exc = {reversal_exc:.4f}, E_inh = {reversal_inh:.4f}.
`G >= 0` always; the sign of a synapse lives in `E_rev`, not in `G`.

## Reproducing and checking it

```bash
python -m connectome_gnn.generators.flyvis_conductance_fit --out <this folder>
python -m connectome_gnn.generators.verify_flyvis_conductance --params <this folder>
```

Both need flyvis, only to read the pretrained parameters and render Sintel.
"""


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def load_flyvis_network(model: str = "flow/0000/000", checkpoint="best", extent=None):
    """Load the pretrained current-based model, optionally at a smaller extent.

    ``graph_data_generator`` builds its connectome at ``extent=8`` rather than the
    extent the checkpoint was trained at, which is sound because every flyvis
    parameter is shared per cell type or edge type: the same state dict applies
    at any extent. A twin meant to generate data for that pipeline has to be
    derived at the same extent, or its edges will not line up.

    Args:
        model: flyvis NetworkView name.
        checkpoint: checkpoint to recover.
        extent: hexagonal extent, or None for whatever the checkpoint stores.

    Returns:
        The network, in eval mode with gradients off.
    """
    import flyvis

    from connectome_gnn.utils import setup_flyvis_model_path

    setup_flyvis_model_path()
    trained = flyvis.NetworkView(model).init_network(checkpoint=checkpoint)
    if extent is None:
        net = trained
    else:
        from flyvis import Network
        from flyvis.utils.config_utils import CONFIG_PATH, get_default_config

        config = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
        config.connectome.extent = extent
        net = Network(**config)
        net.load_state_dict(trained.state_dict())
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def main(args) -> None:
    net = load_flyvis_network(args.model, args.checkpoint, args.extent)
    logger.info(
        "connectome: %d neurons, %d edges (extent %s)",
        net.n_nodes, net.n_edges, args.extent if args.extent else "from checkpoint",
    )

    dataset_kwargs = {}
    n_frames = args.n_frames
    if args.extent is not None:
        # Match graph_data_generator's stimulus_dataset exactly, so the render
        # cache is shared and the twin is fitted on the very clips the GNN
        # experiments will be trained on.
        dataset_kwargs = dict(
            tasks=["flow"], interpolate=True,
            boxfilter=dict(extent=args.extent, kernel_size=13),
            vertical_splits=3, center_crop_fraction=0.7,
        )
        n_frames = 19
        logger.info("matching graph_data_generator's Sintel settings (n_frames=19)")

    n_clips = args.n_train + args.n_test
    _, dataset = naturalistic_movies(
        dt=args.dt, n_sequences=1, n_frames=n_frames, **dataset_kwargs
    )
    if args.n_concat > 1:
        movies = concatenated_movies(
            dataset, n_concat=args.n_concat, n_sequences=n_clips
        )
    else:
        movies, _ = naturalistic_movies(
            dt=args.dt, n_sequences=n_clips, n_frames=n_frames, dataset=dataset
        )
    train, test = movies[: args.n_train], movies[args.n_train :]
    logger.info(
        "%d train / %d test clips of %d frames (%.2f s)",
        len(train), len(test), train[0].shape[0], train[0].shape[0] * args.dt,
    )

    twin, index, diagnostics = derive_conductance_twin(
        net, train, delta_t=args.dt,
        reversal_margin=(args.margin_inh, args.margin_exc),
        epochs=args.epochs, bptt_steps=args.bptt_steps,
        time_chunk=args.time_chunk, ridge=args.ridge, share=not args.per_edge,
        device=args.device,
    )

    teacher_ode, teacher_params = current_based_ode(net, device=args.device)
    initial = steady_state(
        teacher_ode, teacher_params, index.input_index, index.n_nodes, args.dt
    )
    test_stimuli, test_voltages = [], []
    for movie in test:
        stimulus = stimulus_from_movie(
            movie.to(args.device), index.input_index, index.n_nodes
        )
        test_stimuli.append(stimulus)
        test_voltages.append(
            rollout(teacher_ode, teacher_params, stimulus, initial.clone(), args.dt)
        )

    result = evaluate_twin(twin, index, test_voltages, test_stimuli, args.dt)
    diagnostics["held_out_r2"] = result["r2"]
    logger.info("held-out voltage r2 %.4f", result["r2"])
    worst = sorted(result["r2_per_type"].items(), key=lambda kv: kv[1])[:8]
    logger.info("worst cell types %s", [(k, round(v, 3)) for k, v in worst])

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        twin.save(args.out)
        meta = write_tables(net, twin, args.out)
        meta["source_model"] = args.model
        meta["diagnostics"] = diagnostics
        with open(os.path.join(args.out, "meta.json"), "w") as handle:
            json.dump(meta, handle, indent=2)
        logger.info("saved the twin to %s", args.out)

    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("flyvis").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="flow/0000/000",
                        help="pretrained current-based flyvis model")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument(
        "--extent", type=int, default=None,
        help="hexagonal extent; use 8 to match graph_data_generator's connectome",
    )
    parser.add_argument("--dt", type=float, default=1 / 100)
    parser.add_argument("--n-train", type=int, default=16)
    parser.add_argument("--n-test", type=int, default=4)
    parser.add_argument("--n-frames", type=int, default=40)
    parser.add_argument(
        "--n-concat", type=int, default=1,
        help="join this many consecutive clips per fitting sequence; use it when "
             "the rendered clips are shorter than the BPTT window",
    )
    parser.add_argument("--margin-exc", type=float, default=1.0)
    parser.add_argument("--margin-inh", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bptt-steps", type=int, default=20)
    parser.add_argument("--time-chunk", type=int, default=8)
    parser.add_argument(
        "--ridge", type=float, default=1e-6,
        help="Tikhonov weight pulling the convex stage towards the closed-form "
             "prior, relative to the mean diagonal of the normal equations. "
             "Raise it to keep more shared groups off zero at some cost in fit.",
    )
    parser.add_argument(
        "--per-edge", action="store_true",
        help="free every synapse and neuron in the rollout stage instead of "
             "keeping the pretrained model's parameter sharing",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="graphs_data/fly/conductance_twin")
    main(parser.parse_args())
