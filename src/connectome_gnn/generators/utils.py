import os
import subprocess
from time import sleep

import numpy as np
import scipy
import seaborn as sns
import torch
import xarray as xr
from scipy import stats
from scipy.spatial import Delaunay
from tifffile import imread
from tqdm import trange

from connectome_gnn.figure_style import default_style
from connectome_gnn.utils import get_equidistant_points, graphs_data_path, large_tensor_nonzero, to_numpy

# Optional imports
try:
    from fa2_modified import ForceAtlas2
except ImportError:
    ForceAtlas2 = None


def mseq_bits(p=8, taps=(8, 6, 5, 4), seed=1, length=None):
    """
    Simple LFSR-based m-sequence generator that returns a numpy array of ±1.
    Default p=8 -> period 2**8 - 1 = 255.
    """
    if length is None:
        length = 2**p - 1
    state = (1 << p) - 1 if seed is None else (seed % (1 << p)) or 1
    bits = []
    for _ in range(length):
        bits.append(1 if (state & 1) else -1)
        fb = 0
        for t in taps:
            fb ^= (state >> (t - 1)) & 1
        state = (state >> 1) | (fb << (p - 1))
    return np.array(bits, dtype=np.int8)


def assign_columns_from_uv(u_coords, v_coords, n_cols, random_state=0):
    """Cluster photoreceptors into n_cols tiles via k-means on (u,v)."""
    try:
        from sklearn.cluster import KMeans
    except Exception as e:
        raise RuntimeError("scikit-learn is required for 'tile_mseq' visual_input_type") from e
    X = np.stack([u_coords, v_coords], axis=1)
    km = KMeans(n_clusters=n_cols, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels


def compute_column_labels(u_coords, v_coords, n_columns, seed=0):
    labels = assign_columns_from_uv(u_coords, v_coords, n_columns, random_state=seed)
    centers = np.zeros((n_columns, 2), dtype=np.float32)
    counts = np.zeros(n_columns, dtype=np.int32)
    for i, lab in enumerate(labels):
        centers[lab, 0] += u_coords[i]
        centers[lab, 1] += v_coords[i]
        counts[lab] += 1
    counts[counts == 0] = 1
    centers /= counts[:, None]
    return labels, centers


def build_neighbor_graph(centers, k=6):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(centers)), algorithm="auto").fit(centers)
    dists, idxs = nbrs.kneighbors(centers)
    adj = [set() for _ in range(len(centers))]
    for i in range(len(centers)):
        for j in idxs[i, 1:]:
            adj[i].add(int(j))
            adj[int(j)].add(i)
    return adj


def greedy_blue_mask(adj, n_cols, target_density=0.5, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    order = rng.permutation(n_cols)
    chosen = np.zeros(n_cols, dtype=bool)
    blocked = np.zeros(n_cols, dtype=bool)
    target = int(target_density * n_cols)
    for i in order:
        if not blocked[i]:
            chosen[i] = True
            for j in adj[i]:
                blocked[j] = True
        if chosen.sum() >= target:
            break
    if chosen.sum() < target:
        remain = np.where(~chosen)[0]
        rng.shuffle(remain)
        for i in remain:
            conflict = any(chosen[j] for j in adj[i])
            if not conflict:
                chosen[i] = True
            if chosen.sum() >= target:
                break
    return chosen


def apply_pairwise_knobs_torch(
    code_pm1: torch.Tensor, corr_strength: float, flip_prob: float, seed: int
) -> torch.Tensor:
    """
    code_pm1: shape [n_tiles], values in approximately {-1, +1}
    corr_strength: 0..1; blends in a global shared ±1 component (↑ pairwise corr)
    flip_prob: 0..1; per-tile random sign flips (decorrelates)
    seed: for reproducibility (we also add tile_idx later to vary per frame)
    """
    out = code_pm1.clone()

    # Torch RNG on correct device
    gen = torch.Generator(device=out.device)
    gen.manual_seed(int(seed) & 0x7FFFFFFF)

    # (1) Optional global shared component
    if corr_strength > 0.0:
        g = torch.randint(0, 2, (1,), generator=gen, device=out.device, dtype=torch.int64)
        g = g.float().mul_(2.0).add_(-1.0)  # {0,1} -> {-1,+1}
        out.mul_(1.0 - float(corr_strength)).add_(float(corr_strength) * g)

    # (2) Optional per-tile random flips
    if flip_prob > 0.0:
        flips = torch.rand(out.shape, generator=gen, device=out.device) < float(flip_prob)
        out[flips] = -out[flips]

    return out


def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)


def init_neurons(config=[], scenario="none", ratio=1, device=[]):
    sim = config.simulation
    n_neurons = sim.n_neurons * ratio

    xc, yc = get_equidistant_points(n_points=n_neurons)
    pos = torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    perm = torch.randperm(pos.size(0), device=device)
    pos = pos[perm]

    dpos = sim.dpos_init * torch.randn((n_neurons, sim.dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))

    type = torch.zeros(int(n_neurons / sim.n_neuron_types), device=device)

    for n in range(1, sim.n_neuron_types):
        type = torch.cat((type, n * torch.ones(int(n_neurons / sim.n_neuron_types), device=device)), 0)
    if type.shape[0] < n_neurons:
        type = torch.cat((type, n * torch.ones(n_neurons - type.shape[0], device=device)), 0)

    if (config.graph_model.signal_model_name == "PDE_N6") | (config.graph_model.signal_model_name == "PDE_N7"):
        features = torch.cat(
            (
                torch.rand((n_neurons, 1), device=device),
                0.1 * torch.randn((n_neurons, 1), device=device),
                torch.ones((n_neurons, 1), device=device),
                torch.zeros((n_neurons, 1), device=device),
            ),
            1,
        )
    elif "excitation_single" in config.graph_model.field_type:
        features = torch.zeros((n_neurons, 2), device=device)
    else:
        features = torch.cat(
            (torch.randn((n_neurons, 1), device=device) * 5, 0.1 * torch.randn((n_neurons, 1), device=device)), 1
        )

    type = type[:, None]
    particle_id = torch.arange(n_neurons, device=device)
    particle_id = particle_id[:, None]
    age = torch.zeros((n_neurons, 1), device=device)

    return pos, dpos, type, features, age, particle_id


def random_rotation_matrix(device="cpu"):
    # Random Euler angles
    roll = torch.rand(1, device=device) * 2 * torch.pi
    pitch = torch.rand(1, device=device) * 2 * torch.pi
    yaw = torch.rand(1, device=device) * 2 * torch.pi

    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Rotation matrices around each axis
    R_x = torch.tensor([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]], device=device).squeeze()

    R_y = torch.tensor([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]], device=device).squeeze()

    R_z = torch.tensor([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]], device=device).squeeze()

    # Combined rotation matrix: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x
    return R


def get_index(n_neurons, n_neuron_types):
    index_particles = []
    for n in range(n_neuron_types):
        index_particles.append(np.arange((n_neurons // n_neuron_types) * n, (n_neurons // n_neuron_types) * (n + 1)))
    return index_particles


def get_time_series(x_list, cell_id, feature):

    match feature:
        case "velocity_x":
            feature = 3
        case "velocity_y":
            feature = 4
        case "type" | "state":
            feature = 5
        case "age":
            feature = 8
        case "mass":
            feature = 10

        case _:  # default
            feature = 0

    time_series = []
    for it in range(len(x_list)):
        x = x_list[it].clone().detach()
        pos_cell = torch.argwhere(x[:, 0] == cell_id)
        if len(pos_cell) > 0:
            time_series.append(x[pos_cell, feature].squeeze())
        else:
            time_series.append(torch.tensor([0.0]))

    return to_numpy(torch.stack(time_series))


def init_mesh(config, device):

    sim = config.simulation
    model_config = config.graph_model

    n_input_neurons_per_axis = int(np.sqrt(sim.n_input_neurons))
    xs = torch.linspace(
        1 / (2 * n_input_neurons_per_axis), 1 - 1 / (2 * n_input_neurons_per_axis), steps=n_input_neurons_per_axis
    )
    ys = torch.linspace(
        1 / (2 * n_input_neurons_per_axis), 1 - 1 / (2 * n_input_neurons_per_axis), steps=n_input_neurons_per_axis
    )
    x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing="xy")
    x_mesh = torch.reshape(x_mesh, (n_input_neurons_per_axis**2, 1))
    y_mesh = torch.reshape(y_mesh, (n_input_neurons_per_axis**2, 1))
    mesh_size = 1 / n_input_neurons_per_axis
    pos_mesh = torch.zeros((sim.n_input_neurons, 2), device=device)
    pos_mesh[0 : sim.n_input_neurons, 0:1] = x_mesh[0 : sim.n_input_neurons]
    pos_mesh[0 : sim.n_input_neurons, 1:2] = y_mesh[0 : sim.n_input_neurons]

    i0 = imread(graphs_data_path(sim.node_value_map))
    if len(i0.shape) == 2:
        # i0 = i0[0,:, :]
        i0 = np.flipud(i0)
        values = i0[(to_numpy(pos_mesh[:, 1]) * 255).astype(int), (to_numpy(pos_mesh[:, 0]) * 255).astype(int)]

    mask_mesh = (
        (x_mesh > torch.min(x_mesh) + 0.02)
        & (x_mesh < torch.max(x_mesh) - 0.02)
        & (y_mesh > torch.min(y_mesh) + 0.02)
        & (y_mesh < torch.max(y_mesh) - 0.02)
    )

    if "grid" in model_config.field_grid:
        pos_mesh = pos_mesh
    else:
        if "pattern_Null.tif" in sim.node_value_map:
            pos_mesh = pos_mesh + torch.randn(sim.n_input_neurons, 2, device=device) * mesh_size / 24
        else:
            pos_mesh = pos_mesh + torch.randn(sim.n_input_neurons, 2, device=device) * mesh_size / 8

    match model_config.mesh_model_name:
        case "RD_Gray_Scott_Mesh":
            node_value = torch.zeros((sim.n_input_neurons, 2), device=device)
            node_value[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
            node_value[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
        case "RD_FitzHugh_Nagumo_Mesh":
            node_value = (
                torch.zeros((sim.n_input_neurons, 2), device=device)
                + torch.rand((sim.n_input_neurons, 2), device=device) * 0.1
            )
        case "RD_Mesh" | "RD_Mesh2" | "RD_Mesh3":
            node_value = torch.rand((sim.n_input_neurons, 3), device=device)
            s = torch.sum(node_value, dim=1)
            for k in range(3):
                node_value[:, k] = node_value[:, k] / s
        case "DiffMesh" | "WaveMesh" | "Particle_Mesh_A" | "Particle_Mesh_B" | "WaveSmoothParticle":
            node_value = torch.zeros((sim.n_input_neurons, 2), device=device)
            node_value[:, 0] = torch.tensor(values / 255 * 5000, device=device)
        case "PDE_O_Mesh":
            node_value = torch.zeros((sim.n_neurons, 5), device=device)
            node_value[0 : sim.n_neurons, 0:1] = x_mesh[0 : sim.n_neurons]
            node_value[0 : sim.n_neurons, 1:2] = y_mesh[0 : sim.n_neurons]
            node_value[0 : sim.n_neurons, 2:3] = torch.randn(sim.n_neurons, 1, device=device) * 2 * np.pi  # theta
            node_value[0 : sim.n_neurons, 3:4] = torch.ones(sim.n_neurons, 1, device=device) * np.pi / 200  # d_theta
            node_value[0 : sim.n_neurons, 4:5] = node_value[0 : sim.n_neurons, 3:4]  # d_theta0
            pos_mesh[:, 0] = node_value[:, 0] + (3 / 8) * mesh_size * torch.cos(node_value[:, 2])
            pos_mesh[:, 1] = node_value[:, 1] + (3 / 8) * mesh_size * torch.sin(node_value[:, 2])
        case "":
            node_value = torch.zeros((sim.n_input_neurons, 2), device=device)

    type_mesh = torch.zeros((sim.n_input_neurons, 1), device=device)

    node_id_mesh = torch.arange(sim.n_input_neurons, device=device)
    node_id_mesh = node_id_mesh[:, None]
    dpos_mesh = torch.zeros((sim.n_input_neurons, 2), device=device)

    x_mesh = torch.concatenate(
        (
            node_id_mesh.clone().detach(),
            pos_mesh.clone().detach(),
            dpos_mesh.clone().detach(),
            type_mesh.clone().detach(),
            node_value.clone().detach(),
        ),
        1,
    )

    pos = to_numpy(x_mesh[:, 1:3])
    tri = Delaunay(pos, qhull_options="QJ")
    face = torch.from_numpy(tri.simplices)
    face_longest_edge = np.zeros((face.shape[0], 1))

    sleep(0.5)
    for k in trange(face.shape[0], ncols=100):
        # compute edge distances
        x1 = pos[face[k, 0], :]
        x2 = pos[face[k, 1], :]
        x3 = pos[face[k, 2], :]
        a = np.sqrt(np.sum((x1 - x2) ** 2))
        b = np.sqrt(np.sum((x2 - x3) ** 2))
        c = np.sqrt(np.sum((x3 - x1) ** 2))
        A = np.max([a, b]) / np.min([a, b])
        B = np.max([a, c]) / np.min([a, c])
        C = np.max([c, b]) / np.min([c, b])
        face_longest_edge[k] = np.max([A, B, C])

    face_kept = np.argwhere(face_longest_edge < 5)
    face_kept = face_kept[:, 0]
    face = face[face_kept, :]
    face = face.t().contiguous()
    face = face.to(device, torch.long)

    pos_3d = torch.cat((x_mesh[:, 1:3], torch.ones((x_mesh.shape[0], 1), device=device)), dim=1)
    from torch_geometric.utils import get_mesh_laplacian

    edge_index_mesh, edge_weight_mesh = get_mesh_laplacian(pos=pos_3d, face=face, normalization="None")
    edge_weight_mesh = edge_weight_mesh.to(dtype=torch.float32)
    mesh_data = {
        "mesh_pos": pos_3d,
        "face": face,
        "edge_index": edge_index_mesh,
        "edge_weight": edge_weight_mesh,
        "mask": mask_mesh,
        "size": mesh_size,
    }

    if (model_config.particle_model_name == "PDE_ParticleField_A") | (
        model_config.particle_model_name == "PDE_ParticleField_B"
    ):
        type_mesh = 0 * type_mesh

    a_mesh = torch.zeros_like(type_mesh)
    type_mesh = type_mesh.to(dtype=torch.float32)

    if "Smooth" in model_config.mesh_model_name:
        distance = torch.sum((pos_mesh[:, None, :] - pos_mesh[None, :, :]) ** 2, dim=2)
        adj_t = ((distance < sim.max_radius**2) & (distance >= 0)).float() * 1
        mesh_data["edge_index"] = adj_t.nonzero().t().contiguous()

    return pos_mesh, dpos_mesh, type_mesh, node_value, a_mesh, node_id_mesh, mesh_data


def init_connectivity(
    connectivity_file,
    connectivity_type,
    connectivity_filling_factor,
    T1,
    n_neurons,
    n_neuron_types,
    dataset_name,
    device,
    connectivity_rank=1,
    Dale_law=False,
    Dale_law_factor=0.5,
):

    low_rank_factors = None

    if "adjacency.pt" in connectivity_file:
        connectivity = torch.load(connectivity_file, map_location=device, weights_only=False)
    elif "mat" in connectivity_file:
        mat = scipy.io.loadmat(connectivity_file)
        connectivity = torch.tensor(mat["A"], device=device)
    elif "zarr" in connectivity_file:
        print("loading zarr ...")
        dataset = xr.open_zarr(connectivity_file)
        trained_weights = dataset["trained"]  # alpha * sign * N
        print(f"weights {trained_weights.shape}")
        dataset["untrained"]  # sign * N
        values = trained_weights[0:n_neurons, 0:n_neurons]
        values = np.array(values)
        values = values / np.max(values)
        connectivity = torch.tensor(values, dtype=torch.float32, device=device)
        values = []
    elif "tif" in connectivity_file:
        # TODO: constructRandomMatrices function not implemented
        raise NotImplementedError("constructRandomMatrices function not implemented for tif files")
        # connectivity = constructRandomMatrices(n_neurons=n_neurons, density=1.0, connectivity_mask=f"./graphs_data/{connectivity_file}" ,device=device)
        # n_neurons = connectivity.shape[0]
        # TODO: config parameter not passed to this function
        # config.simulation.n_neurons = n_neurons
    elif connectivity_type != "none":
        if "chaotic" in connectivity_type:
            # Chaotic network
            connectivity = np.random.randn(n_neurons, n_neurons) * np.sqrt(1 / n_neurons)
        elif "ring attractor" in connectivity_type:
            # Ring attractor network
            th = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)  # Preferred firing location (angle)
            J1 = 1.0
            J0 = 0.5
            connectivity = (J1 * np.cos(th[:, None] - th[None, :]) + J0) / n_neurons  # Synaptic weight matrix
        elif "low_rank" in connectivity_type:
            # Low rank network: W = U @ V where U is (N x rank) and V is (rank x N)
            U = np.random.randn(n_neurons, connectivity_rank)
            V = np.random.randn(connectivity_rank, n_neurons)
            connectivity = U @ V / np.sqrt(connectivity_rank * n_neurons)
            low_rank_factors = (U, V)

        elif "successor" in connectivity_type:
            # Successor Representation
            T = np.eye(n_neurons, k=1)
            gamma = 0.98
            connectivity = np.linalg.inv(np.eye(n_neurons) - gamma * T)
        elif "null" in connectivity_type:
            connectivity = np.zeros((n_neurons, n_neurons))
        elif "Gaussian" in connectivity_type:
            connectivity = torch.randn((n_neurons, n_neurons), dtype=torch.float32, device=device)
            connectivity = connectivity / np.sqrt(n_neurons)
            print(f"Gaussian   1/sqrt(N)  {1 / np.sqrt(n_neurons)}    std {torch.std(connectivity.flatten())}")
        elif "Lorentz" in connectivity_type:
            s = np.random.standard_cauchy(n_neurons**2)
            s[(s < -25) | (s > 25)] = 0
            if n_neurons < 2000:
                s = s / n_neurons**0.7
            elif n_neurons < 4000:
                s = s / n_neurons**0.675
            elif n_neurons < 8000:
                s = s / n_neurons**0.67
            elif n_neurons == 8000:
                s = s / n_neurons**0.66
            elif n_neurons > 8000:
                s = s / n_neurons**0.5
            print(f"Lorentz   1/sqrt(N)  {1 / np.sqrt(n_neurons):0.3f}    std {np.std(s):0.3f}")
            connectivity = torch.tensor(s, dtype=torch.float32, device=device)
            connectivity = torch.reshape(connectivity, (n_neurons, n_neurons))
        elif "uniform" in connectivity_type:
            connectivity = torch.rand((n_neurons, n_neurons), dtype=torch.float32, device=device)
            connectivity = connectivity - 0.5

        connectivity = torch.tensor(connectivity, dtype=torch.float32, device=device)
        connectivity.fill_diagonal_(0)

    # Apply Dale's law: each neuron (column) is either excitatory or inhibitory
    if Dale_law:
        n_excitatory = int(n_neurons * Dale_law_factor)
        n_inhibitory = n_neurons - n_excitatory

        # Take absolute values
        connectivity = torch.abs(connectivity)

        # First n_excitatory columns are positive (excitatory), rest are negative (inhibitory)
        # Columns represent presynaptic neurons in W[post, pre] convention
        connectivity[:, n_excitatory:] = -connectivity[:, n_excitatory:]

        print(f"Dale's law applied: {n_excitatory} excitatory columns, {n_inhibitory} inhibitory columns")

    if connectivity_filling_factor != 1:
        mask = torch.rand(connectivity.shape) > connectivity_filling_factor
        connectivity[mask] = 0
        mask = (connectivity != 0).float()

        # Calculate effective filling factor
        total_possible = connectivity.shape[0] * connectivity.shape[1]
        actual_connections = mask.sum().item()
        effective_filling_factor = actual_connections / total_possible

        print(f"target filling factor: {connectivity_filling_factor}")
        print(f"effective filling factor: {effective_filling_factor:.6f}")
        print(f"actual connections: {int(actual_connections)}/{total_possible}")

        if n_neurons > 10000:
            edge_index = large_tensor_nonzero(mask)
            print(f"edge_index {edge_index.shape}")
        else:
            edge_index = mask.nonzero().t().contiguous()

    else:
        adj_matrix = torch.ones((n_neurons)) - torch.eye(n_neurons)
        from torch_geometric.utils import dense_to_sparse

        edge_index, edge_attr = dense_to_sparse(adj_matrix)
        mask = (adj_matrix != 0).float()

    if "structured" in connectivity_type:
        parts = connectivity_type.split("_")
        float_value1 = float(parts[-2])  # repartition pos/neg
        float_value2 = float(parts[-1])  # filling factor

        matrix_sign = torch.tensor(
            stats.bernoulli(float_value1).rvs(n_neuron_types**2) * 2 - 1, dtype=torch.float32, device=device
        )
        matrix_sign = matrix_sign.reshape(n_neuron_types, n_neuron_types)

        fig, ax = default_style.figure(width=10, height=10)
        sns.heatmap(
            to_numpy(connectivity),
            center=0,
            square=True,
            cmap="bwr",
            cbar_kws={"fraction": 0.046},
            vmin=-0.1,
            vmax=0.1,
            ax=ax,
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=default_style.tick_font_size)
        ax.set_xticks([0, n_neurons - 1])
        ax.set_xticklabels([1, n_neurons])
        ax.set_yticks([0, n_neurons - 1])
        ax.set_yticklabels([1, n_neurons])
        default_style.savefig(fig, graphs_data_path(dataset_name, "adjacency_0.png"))

        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        connectivity = torch.abs(connectivity)
        T1_ = to_numpy(T1.squeeze())
        xy_grid = np.stack(np.meshgrid(T1_, T1_), -1)
        sign_matrix = matrix_sign[xy_grid[..., 0], xy_grid[..., 1]]
        connectivity *= sign_matrix

        fig_sign, ax_sign = default_style.figure(width=10, height=10)
        ax_sign.imshow(to_numpy(sign_matrix))
        default_style.savefig(fig_sign, graphs_data_path(dataset_name, "large_connectivity_sign.tif"))

        fig, ax = default_style.figure(width=10, height=10)
        sns.heatmap(
            to_numpy(connectivity),
            center=0,
            square=True,
            cmap="bwr",
            cbar_kws={"fraction": 0.046},
            vmin=-0.1,
            vmax=0.1,
            ax=ax,
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=default_style.tick_font_size)
        ax.set_xticks([0, n_neurons - 1])
        ax.set_xticklabels([1, n_neurons])
        ax.set_yticks([0, n_neurons - 1])
        ax.set_yticklabels([1, n_neurons])
        default_style.savefig(fig, graphs_data_path(dataset_name, "adjacency_1.png"))

        flat_sign_matrix = sign_matrix.flatten()
        num_elements = len(flat_sign_matrix)
        num_ones = int(num_elements * float_value2)
        indices = np.random.choice(num_elements, num_ones, replace=False)
        flat_sign_matrix[:] = 0
        flat_sign_matrix[indices] = 1
        sign_matrix = flat_sign_matrix.reshape(sign_matrix.shape)

        connectivity *= sign_matrix

        fig, ax = default_style.figure(width=10, height=10)
        sns.heatmap(
            to_numpy(connectivity),
            center=0,
            square=True,
            cmap="bwr",
            cbar_kws={"fraction": 0.046},
            vmin=-0.1,
            vmax=0.1,
            ax=ax,
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=default_style.tick_font_size)
        ax.set_xticks([0, n_neurons - 1])
        ax.set_xticklabels([1, n_neurons])
        ax.set_yticks([0, n_neurons - 1])
        ax.set_yticklabels([1, n_neurons])
        default_style.savefig(fig, graphs_data_path(dataset_name, "adjacency_2.png"))

        total_possible = connectivity.shape[0] * connectivity.shape[1]
        actual_connections = (connectivity != 0).sum().item()
        effective_filling_factor = actual_connections / total_possible

        print(f"target filling factor: {float_value2}")
        print(f"effective filling factor: {effective_filling_factor:.6f}")
        print(f"actual connections: {actual_connections}/{total_possible}")

    edge_index = edge_index.to(device=device)

    return edge_index, connectivity, mask, low_rank_factors


def generate_compressed_video_mp4(output_dir, run=0, framerate=10, output_name=None, crf=23, log_dir=None):
    """
    Generate a compressed video using ffmpeg's libx264 codec in MP4 format.
    Automatically handles odd dimensions by scaling to even dimensions.

    Parameters:
        output_dir (str): Path to directory containing Fig/Fig_*.png.
        run (int): Run index to use in filename pattern.
        framerate (int): Desired video framerate.
        output_name (str): Name of output .mp4 file.
        crf (int): Constant Rate Factor for quality (0-51, lower = better quality, 23 is default).
        log_dir (str): If provided, save mp4 to log_dir instead of output_dir.
    """

    fig_dir = os.path.join(output_dir, "Fig")
    input_pattern = os.path.join(fig_dir, f"Fig_{run}_%06d.png")

    # Save to log_dir if provided, otherwise to output_dir
    save_dir = log_dir if log_dir is not None else output_dir
    output_path = os.path.join(save_dir, f"{output_name}.mp4")

    # Video filter to ensure even dimensions (required for yuv420p)
    # This scales the video so both width and height are divisible by 2

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",  # Suppress verbose output
        "-framerate",
        str(framerate),
        "-i",
        input_pattern,
        "-vf",
        "scale='trunc(iw/2)*2:trunc(ih/2)*2'",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"compressed video (libx264) saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating video: {e}")
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to generate videos.")


def is_adex_model(signal_model_name: str) -> bool:
    """Check if signal_model_name maps to the AdEx spiking ODE params class."""
    from connectome_gnn.generators.ode_params import FlyVisAdExODEParams, get_ode_params_class

    try:
        cls = get_ode_params_class(signal_model_name)
        return cls is FlyVisAdExODEParams
    except KeyError:
        return False


def is_hodgkin_huxley_model(signal_model_name: str) -> bool:
    """Check if signal_model_name maps to the Hodgkin-Huxley ODE params class."""
    from connectome_gnn.generators.ode_params import FlyVisHodgkinHuxleyODEParams, get_ode_params_class

    try:
        cls = get_ode_params_class(signal_model_name)
        return cls is FlyVisHodgkinHuxleyODEParams
    except KeyError:
        return False


def is_connconstr_model(signal_model_name: str) -> bool:
    """Check if signal_model_name maps to a connconstr ODE params class."""
    from connectome_gnn.generators.ode_params import (
        DrosophilaCxODEParams,
        LarvaODEParams,
        ZebrafishODEParams,
        get_ode_params_class,
    )

    connconstr_classes = (ZebrafishODEParams, DrosophilaCxODEParams, LarvaODEParams)
    try:
        cls = get_ode_params_class(signal_model_name)
        return cls in connconstr_classes
    except KeyError:
        return False


_FLYVIS_HYBRID_MODELS = {
    # truncated flyvis hex disk (extent=8), FlyWire RFs
    "e8_flywireRF",
    "e8_flywireRF_known_ode",
    "e8_flywireRF_mlp",
    "e8_flywireRF_eed",
    "e8_flywireRF_stimulus",
    "e8_flywireRF_proximal_nulls",
    "e8_flywireRF_proximal_nulls_known_ode",
    "e8_flywireRF_proximal_nulls_mlp",
    "e8_flywireRF_proximal_nulls_eed",
    "e8_flywireRF_proximal_nulls_stimulus",
    "e8_flywireRF_random_nulls",
    "e8_flywireRF_random_nulls_known_ode",
    # full FlyWire eye map
    "full_eye_flywireRF",
    "full_eye_flywireRF_known_ode",
    "full_eye_flywireRF_mlp",
    "full_eye_flywireRF_eed",
    "full_eye_flywireRF_stimulus",
    "full_eye_flywireRF_proximal_nulls",
    "full_eye_flywireRF_proximal_nulls_known_ode",
    "full_eye_flywireRF_proximal_nulls_mlp",
    "full_eye_flywireRF_proximal_nulls_eed",
    "full_eye_flywireRF_proximal_nulls_stimulus",
    "full_eye_flywireRF_random_nulls",
    "full_eye_flywireRF_random_nulls_known_ode",
}


def is_flyvis_hybrid_model(signal_model_name: str) -> bool:
    """Check if signal_model_name is a flyrewire hybrid model."""
    return signal_model_name in _FLYVIS_HYBRID_MODELS


def _print_opto_banner(config, opto_cfg) -> None:
    """Green banner: confirms source-dataset reuse and dumps opto parameters."""
    G, R = "\033[92m", "\033[0m"
    src = opto_cfg.source_dataset
    src_dir = graphs_data_path(src)
    if not os.path.isdir(src_dir):
        alt = graphs_data_path("fly", src)
        if os.path.isdir(alt):
            src_dir = alt
    voltage_zarr = os.path.join(src_dir, "x_list_train", "voltage.zarr")
    src_ok = os.path.isdir(voltage_zarr)
    tgt = opto_cfg.target
    wf = opto_cfg.waveform
    target_str = (
        f"mode={tgt.mode} k={tgt.k}" if str(tgt.mode) == "OptoTargetMode.TOPK_NULLSPACE"
        or tgt.mode == "topk_nullspace"
        else f"mode={tgt.mode} cell_types={list(tgt.cell_types)}"
    )
    print(f"{G}{'='*70}{R}")
    print(f"{G}[opto] OPTOGENETIC PERTURBATION — re-simulation from existing source{R}")
    print(f"{G}[opto] source dataset:  {src}{R}")
    print(f"{G}[opto] source on disk:  {src_dir}  ({'OK' if src_ok else 'MISSING'}){R}")
    print(f"{G}[opto] target output:   {config.dataset}{R}")
    print(f"{G}[opto] target spec:     {target_str}  column_distinct={tgt.column_distinct}{R}")
    wf_extra = f"  frames_on={wf.frames_on}" if wf.kind == "heaviside" else ""
    print(f"{G}[opto] waveform:        kind={wf.kind}  amplitude={wf.amplitude}  "
          f"noise_level={wf.noise_level}{wf_extra}{R}")
    print(f"{G}[opto] seed:            {wf.seed}  (paired with source for matched comparison){R}")
    print(f"{G}{'='*70}{R}", flush=True)


# --- Diagnostic traces for the visual_perturbation pipeline ----------------
# Sister to optogenetics._draw_opto_traces / plot_voltage_calcium_traces but
# for the inline visual_perturbation pass (heaviside-var added on x.stimulus).
# No baseline-vs-perturbed comparison — single dataset, V (+optional Ca)
# stacked per neuron with the per-neuron stimulus trace beneath.


def select_retina_trace_neurons(
    neuron_type: "torch.Tensor",
    per_type: int = 2,
) -> tuple[list[int], list[str]]:
    """Pick `per_type` indices for each R1..R8 photoreceptor type.

    Returns (neuron_idx, labels) ordered R1, R1, R2, R2, ... R8, R8 (drops
    types with no neurons). Indices are spaced across the available pool so
    different retinotopic columns are represented.
    """
    from connectome_gnn.metrics import NAME_TO_INDEX
    nt = neuron_type.detach().cpu().numpy()
    neuron_idx: list[int] = []
    labels: list[str] = []
    for r in (f"R{k}" for k in range(1, 9)):
        if r not in NAME_TO_INDEX:
            continue
        t_int = NAME_TO_INDEX[r]
        ids = (nt == t_int).nonzero()[0]
        if len(ids) == 0:
            continue
        k = min(per_type, len(ids))
        # uniform sample across the population (different columns)
        picks = ids[np.linspace(0, len(ids) - 1, k).round().astype(int)]
        for j, idx in enumerate(picks.tolist()):
            neuron_idx.append(int(idx))
            labels.append(f"{r} #{j}")
    return neuron_idx, labels


def select_downstream_trace_neurons(
    neuron_type: "torch.Tensor",
    per_type: int = 2,
) -> tuple[list[int], list[str]]:
    """Pick `per_type` indices for each of 8 representative non-retina types.

    Eight types × 2 columns = 16 traces, matching the retina panel's
    8 photoreceptors × 2 columns. Covers lamina (L1-L3), medulla
    (Mi1, Mi9, Tm1), and lobula direction-selective (T4a, T5a) cells.
    """
    from connectome_gnn.metrics import NAME_TO_INDEX
    nt = neuron_type.detach().cpu().numpy()
    downstream_types = ["L1", "L2", "L3", "Mi1", "Mi9", "Tm1", "T4a", "T5a"]
    neuron_idx: list[int] = []
    labels: list[str] = []
    for name in downstream_types:
        if name not in NAME_TO_INDEX:
            continue
        t_int = NAME_TO_INDEX[name]
        ids = (nt == t_int).nonzero()[0]
        if len(ids) == 0:
            continue
        k = min(per_type, len(ids))
        picks = ids[np.linspace(0, len(ids) - 1, k).round().astype(int)]
        for j, idx in enumerate(picks.tolist()):
            neuron_idx.append(int(idx))
            labels.append(f"{name} #{j}")
    return neuron_idx, labels


def _draw_stacked_traces(ax, voltage, calcium, labels, time_ms, ylabel):
    """Stacked V (+optional Ca) draw, no stimulus row.

    voltage: (T, n), calcium: (T, n) or None.
    """
    _TRACE_SHRINK = 0.65
    _LW_V, _LW_CA = 0.45, 0.7
    _COLOR_V  = "#000000"   # black voltage trace
    _COLOR_CA = "#cf222e"   # pure red calcium (drawn on top of V)
    _FS_LABEL, _FS_TICK, _FS_TYPE = 8, 6, 6

    n_neurons = voltage.shape[1]
    row_stds = [
        max(voltage[:, i].std(),
            (calcium[:, i].std() if calcium is not None else 0.0))
        for i in range(n_neurons)
    ]
    step_v = max(0.5 * _TRACE_SHRINK,
                 3.0 * _TRACE_SHRINK * (max(row_stds) if row_stds else 1.0))
    v_mean = voltage.mean(axis=0)
    ca_mean = calcium.mean(axis=0) if calcium is not None else None
    s = _TRACE_SHRINK

    for i in range(n_neurons):
        y_base = (n_neurons - 1 - i) * step_v
        ax.plot(time_ms, s * (voltage[:, i] - v_mean[i]) + y_base,
                lw=_LW_V, color=_COLOR_V, alpha=0.95, zorder=3,
                label="voltage" if i == 0 else None)
        if calcium is not None:
            ax.plot(time_ms, s * (calcium[:, i] - ca_mean[i]) + y_base,
                    lw=_LW_CA, color=_COLOR_CA, alpha=0.95, zorder=4,
                    label="calcium" if i == 0 else None)
        ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.02, y_base,
                labels[i], fontsize=_FS_TYPE, va="center", ha="right",
                color="black")

    ax.set_ylabel(ylabel, fontsize=_FS_LABEL, labelpad=32)
    ax.set_ylim([-step_v, (n_neurons - 1) * step_v + 2.2 * step_v])
    ax.set_yticks([])
    ax.set_xlim([time_ms[0], time_ms[-1]])

    lo, hi = time_ms[0], time_ms[-1]
    raw_step = (hi - lo) / max(1, 3 - 1)
    mag = 10 ** np.floor(np.log10(max(raw_step, 1e-12)))
    step = mag
    for m in (1, 2, 5, 10):
        if m * mag >= raw_step:
            step = m * mag
            break
    tick_lo = np.ceil(lo / step - 1e-9) * step
    ticks = list(np.arange(tick_lo, hi + step / 2, step))
    if ticks:
        ax.set_xticks(ticks)
    ax.set_xlabel("time (ms)", fontsize=_FS_LABEL, labelpad=1)
    ax.tick_params(axis="x", labelsize=_FS_TICK, pad=1)
    ax.spines["left"].set_visible(False)


def _draw_kernel_convolution_panel(ax, kernel, dt_seconds):
    """Stacked rows showing K * pulse for several pulse widths.

    Each row: red step input (off -> on -> off) and the kernel-convolved
    output (light pink) at the same vertical offset. Pulse widths sweep
    through values relevant to the GCaMP6f time-constants.
    """
    _TRACE_SHRINK = 0.65
    _LW_INPUT, _LW_OUT = 0.8, 1.2
    _COLOR_INPUT = "#cf222e"
    _COLOR_OUT   = "#ff7a7a"
    _FS_LABEL, _FS_TICK, _FS_TYPE = 8, 6, 6

    kernel_np = kernel.detach().cpu().numpy() if hasattr(kernel, 'detach') else np.asarray(kernel)
    kernel_np = kernel_np.astype(np.float32)
    K = kernel_np.shape[0]

    # Pulse widths in seconds: short / medium / long / very long
    pulse_seconds = [0.1, 0.5, 1.0, 2.0]
    n_pulses = len(pulse_seconds)
    # Total trace length: long enough that the longest pulse + kernel tail fit
    total_seconds = max(pulse_seconds) * 1.5 + K * dt_seconds + 0.5
    total_frames = int(round(total_seconds / dt_seconds))
    pre_frames = int(round(0.2 / dt_seconds))   # 200 ms baseline before pulse on

    time_ms = np.arange(total_frames) * dt_seconds * 1000.0

    step_v = 1.6 * _TRACE_SHRINK
    s = _TRACE_SHRINK

    for i, pw_s in enumerate(pulse_seconds):
        y_base = (n_pulses - 1 - i) * step_v
        pulse_frames = int(round(pw_s / dt_seconds))
        u = np.zeros(total_frames, dtype=np.float32)
        end = min(total_frames, pre_frames + pulse_frames)
        u[pre_frames:end] = 1.0
        # 'full' convolution then truncated to total_frames, matching the
        # causal forward sweep used in the simulator.
        y = np.convolve(u, kernel_np, mode='full')[:total_frames]

        ax.plot(time_ms, s * u + y_base, lw=_LW_INPUT, color=_COLOR_INPUT,
                alpha=0.9, zorder=2,
                label="heaviside" if i == 0 else None)
        ax.plot(time_ms, s * y + y_base, lw=_LW_OUT, color=_COLOR_OUT,
                alpha=0.95, zorder=3,
                label="K * heaviside" if i == 0 else None)
        ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.02, y_base,
                f"{int(pw_s * 1000)} ms", fontsize=_FS_TYPE,
                va="center", ha="right", color="black")

    ax.set_ylabel("pulse width", fontsize=_FS_LABEL, labelpad=32)
    ax.set_ylim([-0.4 * step_v, (n_pulses - 1) * step_v + 2.0 * step_v])
    ax.set_yticks([])
    ax.set_xlim([time_ms[0], time_ms[-1]])
    ax.set_xlabel("time (ms)", fontsize=_FS_LABEL, labelpad=1)
    ax.tick_params(axis="x", labelsize=_FS_TICK, pad=1)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", fontsize=_FS_TICK, frameon=False)


def plot_visual_perturbation_traces(
    voltage_retina: "np.ndarray",
    labels_retina: list[str],
    voltage_other: "np.ndarray",
    labels_other: list[str],
    dt_seconds: float,
    save_path: str,
    calcium_retina: "np.ndarray | None" = None,
    calcium_other: "np.ndarray | None" = None,
    start_frame: int = 0,
    title: str | None = None,
) -> None:
    """Two-panel V (+optional Ca) trace plot.

    Left:  retina neurons (R1..R8, 2 columns each).
    Right: downstream non-retina cell types (L1-L3, Mi1, Mi9, Tm1, T4a,
           T5a; 2 columns each), connectome-mediated response.

    Voltage is drawn in black. If `calcium_*` is provided it is overlaid
    in light-red on top of the voltage trace. Stimulus is intentionally
    not drawn — every retina neuron carries the same telegraph timing
    (different amplitudes), so a per-neuron stimulus trace would be
    redundant. The K_GCaMP impulse response and its convolution with
    heaviside pulses live in a separate kernel.png (plot_kernel_diagram).
    """
    import os
    import matplotlib
    import matplotlib.pyplot as plt

    if voltage_retina.ndim != 2 or voltage_other.ndim != 2:
        raise ValueError("voltage arrays must be (T, n_neurons)")
    if voltage_retina.shape[0] != voltage_other.shape[0]:
        raise ValueError(
            f"frame count mismatch retina={voltage_retina.shape[0]} "
            f"other={voltage_other.shape[0]}"
        )
    if calcium_retina is not None and calcium_retina.shape != voltage_retina.shape:
        raise ValueError(
            f"calcium_retina {calcium_retina.shape} != voltage_retina {voltage_retina.shape}"
        )
    if calcium_other is not None and calcium_other.shape != voltage_other.shape:
        raise ValueError(
            f"calcium_other {calcium_other.shape} != voltage_other {voltage_other.shape}"
        )

    rc_path = "/workspace/connectome-gnn/figures/janne.matplotlibrc"
    if os.path.isfile(rc_path):
        matplotlib.rc_file(rc_path)

    n_frames = voltage_retina.shape[0]
    time_ms = (np.arange(n_frames) + start_frame) * dt_seconds * 1000.0
    n_neurons_retina = voltage_retina.shape[1]

    fig_width = 9.0
    fig_height = 0.42 * n_neurons_retina + 1.2
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    _draw_stacked_traces(
        axes[0], voltage_retina, calcium_retina, labels_retina,
        time_ms, ylabel="retina neurons",
    )
    axes[0].legend(loc="lower right", fontsize=6, frameon=False)

    _draw_stacked_traces(
        axes[1], voltage_other, calcium_other, labels_other,
        time_ms, ylabel="downstream neurons",
    )

    if title:
        # Figure-level title above both panels so it never collides with traces.
        fig.suptitle(title, fontsize=8, y=0.995)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # Leave room at the top for the suptitle.
    fig.tight_layout(rect=(0, 0, 1, 0.97) if title else None)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_kernel_diagram(
    kernel: "np.ndarray | torch.Tensor",
    dt_seconds: float,
    save_path: str,
    title: str | None = None,
) -> None:
    """Two-panel diagnostic for the GCaMP impulse response.

    Left:  K(t) curve in physical seconds.
    Right: K * heaviside(off-on-off) for several pulse widths, showing
           the calcium response a single column would produce if it saw
           an isolated ON pulse.
    """
    import os
    import matplotlib
    import matplotlib.pyplot as plt

    rc_path = "/workspace/connectome-gnn/figures/janne.matplotlibrc"
    if os.path.isfile(rc_path):
        matplotlib.rc_file(rc_path)

    kernel_np = (kernel.detach().cpu().numpy()
                 if hasattr(kernel, 'detach') else np.asarray(kernel)).astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2),
                             gridspec_kw={"width_ratios": [1.0, 1.5]})
    t_axis_s = np.arange(kernel_np.shape[0]) * dt_seconds
    axes[0].plot(t_axis_s, kernel_np, color="#ff7a7a", lw=1.4)
    axes[0].set_xlabel("time (s)", fontsize=8)
    axes[0].set_ylabel("K(t)", fontsize=8)
    axes[0].tick_params(axis="both", labelsize=6)
    if title:
        axes[0].set_title(title, fontsize=8)

    _draw_kernel_convolution_panel(axes[1], kernel_np, dt_seconds)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
