#!/usr/bin/env python
"""
Pseudoinverse weight recovery + rollout verification for the flyvis connectome.

For each of three noise conditions (sigma = 0, 0.05, 0.5) and each recovery
method, this script:

  1.  Loads voltage traces, dv/dt, and stimulus from pre-generated zarr datasets.
  2.  Assembles, for each postsynaptic neuron i, the linear system
          H_i w_i = b_i
      where H_i (T x d_i) is the presynaptic activity matrix (ReLU applied to
      voltage) and b_i (T,) is the ODE right-hand side derived by rearranging
      the flyvis update equation.
  3.  Solves H_i w_i = b_i with several methods (truncated SVD at 90/95/99/100%
      variance, ridge with lambda=1, cross-validated ridge).
  4.  Replaces the ground-truth edge weights W with the recovered weights and
      runs a 1000-frame ODE rollout with the same stimulus and initial condition.
  5.  Reports connectivity R^2 (weight-space agreement) and rollout Pearson r
      (dynamics agreement).
  6.  Writes a LaTeX-ready table to pseudoinverse_table.tex and raw numbers to
      pseudoinverse_table.json.

Output table layout  (noise sigma as column groups, one row per method):

  Method          | sigma=0          | sigma=0.05       | sigma=0.5
                  | R^2_W   r        | R^2_W   r        | R^2_W   r
  Trunc SVD 90%  |  0.07  0.99      |  0.09  0.98      |  0.34  0.87
  ...
"""

import os
import sys
import json
import numpy as np
import torch
import zarr
from collections import defaultdict
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from connectome_gnn.generators.flyvis_ode  import FlyVisODE
from connectome_gnn.generators.ode_params  import FlyVisODEParams
from connectome_gnn.neuron_state           import NeuronState

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Three noise conditions; keys become the LaTeX column-group headers.
NOISE_CONDITIONS = {
    "noise-free": {
        "sigma_label": r"$0$",
        "ode_path":    "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_free/ode_params.pt",
        "data_dir":    "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_free",
    },
    "noise-005": {
        "sigma_label": r"$0.05$",
        "ode_path":    "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_005/ode_params.pt",
        "data_dir":    "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_005",
    },
    "noise-05": {
        "sigma_label": r"$0.5$",
        "ode_path":    "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_05/ode_params.pt",
        "data_dir":    "/workspace/flyvis-gnn/graphs_data/fly/flyvis_noise_05",
    },
}

# Recovery methods: (internal key, LaTeX label for the table).
# SVD methods are identified as "svd_XX" where XX is the integer variance
# threshold in percent (90 → keep singular values explaining 90% of variance).
# "svd_100" keeps all singular values above the numerical floor 1e-12.
METHODS = [
    ("svd_90",   r"Trunc.\ SVD 90\%"),
    ("svd_95",   r"Trunc.\ SVD 95\%"),
    ("svd_99",   r"Trunc.\ SVD 99\%"),
    ("svd_100",  r"Full SVD"),
    ("ridge",    r"Ridge ($\lambda=1$)"),
    ("cv_ridge", r"CV-ridge"),
]

# ODE integration parameters (must match the flyvis dataset generation).
N_FRAMES    = 1_000   # number of rollout timesteps
DT          = 0.02    # integration timestep in seconds
MODEL_TYPE  = "flyvis_A"  # graded-potential model (ReLU nonlinearity)

# Subsampling of the 64 000-frame training data along the time axis.
# 64000 / 8 = 8000 frames used for the linear system — enough to resolve
# per-neuron activity structure while remaining tractable.
SUBSAMPLE_T = 8

# Ridge / CV-ridge settings.
RIDGE_LAMBDA = 1.0
CV_LAMBDAS   = np.logspace(-4, 4, 30)  # 30 candidates on a log scale
CV_FRAC      = 0.2   # fraction of time-frames held out for cross-validation


# ===========================================================================
# Section 1: Data loading
# ===========================================================================

def load_voltage(data_dir):
    """Load subsampled training voltage traces.

    The zarr array at x_list_train/voltage.zarr has shape (T_full, N) where
    T_full=64000 and N=13741.  We subsample every SUBSAMPLE_T-th frame so
    that the resulting (T, N) matrix with T=8000 is manageable for per-neuron
    SVD while still spanning the full stimulus sequence.

    Returns:
        v: (T, N) float64 array of membrane potentials.
    """
    path = os.path.join(data_dir, "x_list_train", "voltage.zarr")
    z = zarr.open_array(path, mode="r")
    return np.array(z[::SUBSAMPLE_T, :], dtype=np.float64)


def load_derivatives(data_dir):
    """Load subsampled time-derivatives dv/dt.

    y_list_train.zarr has shape (T_full, N, 1); axis-2 indexes the single
    derivative channel.  After subsampling and squeezing we get (T, N).

    Returns:
        dv_dt: (T, N) float64 array of dv/dt values.
    """
    path = os.path.join(data_dir, "y_list_train.zarr")
    z = zarr.open_array(path, mode="r")
    return np.array(z[::SUBSAMPLE_T, :, 0], dtype=np.float64)


def load_stimulus(data_dir):
    """Load subsampled external stimulus.

    x_list_train/stimulus.zarr has shape (T_full, N).

    Returns:
        stim: (T, N) float64 array of stimulus values.
    """
    path = os.path.join(data_dir, "x_list_train", "stimulus.zarr")
    z = zarr.open_array(path, mode="r")
    return np.array(z[::SUBSAMPLE_T, :], dtype=np.float64)


def load_stimulus_rollout(data_dir):
    """Load the first N_FRAMES frames of stimulus for the ODE rollout.

    This is the *test* stimulus (not subsampled), used to drive the forward
    ODE integration.  Shape (N_FRAMES, N).
    """
    path = os.path.join(data_dir, "x_list_train", "stimulus.zarr")
    z = zarr.open_array(path, mode="r")
    T_avail = z.shape[0]
    T = min(N_FRAMES, T_avail)
    return torch.tensor(np.array(z[:T]), dtype=torch.float32)


def load_v0(data_dir):
    """Load the initial voltage v(t=0) for the ODE rollout.

    We use the first frame of the training voltage zarr as the shared
    starting point for ground-truth and recovered-W rollouts.

    Returns:
        v0: (N,) float32 tensor.
    """
    path = os.path.join(data_dir, "x_list_train", "voltage.zarr")
    z = zarr.open_array(path, mode="r")
    return torch.tensor(np.array(z[0]), dtype=torch.float32)


def load_neuron_types(data_dir):
    """Load the integer cell-type id for each neuron.

    neuron_type.zarr has shape (N,); values are in [0, 64].  Required by
    FlyVisODE to look up type-specific parameters.

    Returns:
        ntypes: (N,) int64 tensor.
    """
    path = os.path.join(data_dir, "x_list_train", "neuron_type.zarr")
    z = zarr.open_array(path, mode="r")
    return torch.tensor(np.array(z), dtype=torch.long)


# ===========================================================================
# Section 2: Linear system assembly
# ===========================================================================

def compute_rhs(v, dv_dt, stim, tau, v_rest):
    """Compute the right-hand side b_i(t) of the per-neuron linear system.

    The flyvis ODE is:
        tau_i * dv_i/dt = -v_i + V_rest_i + sum_j W_ij * ReLU(v_j) + stim_i(t)

    Rearranging to isolate the synaptic sum gives:
        sum_j W_ij * ReLU(v_j(t))  =  tau_i * dv_i/dt + v_i - V_rest_i - stim_i(t)
                                    =: b_i(t)

    Stacking over T time-points turns this into H_i w_i = b_i, where
    H_i[t, j] = ReLU(v_j(t)) for presynaptic j, and b_i[t] is computed here.

    Args:
        v:      (T, N) voltage traces
        dv_dt:  (T, N) time derivatives
        stim:   (T, N) external stimulus
        tau:    (N,)   time constants tau_i
        v_rest: (N,)   resting potentials V_rest_i

    Returns:
        b: (T, N) right-hand side matrix — b[:, i] is the RHS for neuron i.
    """
    return dv_dt * tau[None, :] + v - v_rest[None, :] - stim


# ===========================================================================
# Section 3: Per-neuron weight recovery
# ===========================================================================

def recover_weights_all_methods(h, b, edge_index, method_keys):
    """Solve H_i w_i = b_i for each postsynaptic neuron and each recovery method.

    A single SVD per neuron is shared by all SVD-based methods (truncated SVD
    at different thresholds).  Ridge and CV-ridge reuse the same SVD factors.

    Args:
        h:           (T, N) presynaptic activity h_j(t) = ReLU(v_j(t)).
                     Rows are time; columns are neurons.
        b:           (T, N) right-hand side matrix from compute_rhs.
        edge_index:  (2, E) tensor with rows [src_neuron; dst_neuron].
        method_keys: list of method identifiers (e.g. ["svd_90", "ridge"]).

    Returns:
        w_all: dict{method_key: (E,) numpy array of recovered edge weights}.
               Indexing matches edge_index.
    """
    T, N = h.shape
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    E   = len(src)

    # Build look-up tables: for each destination neuron, which presynaptic
    # neuron ids connect to it, and which positions in the edge list are theirs.
    in_nodes  = defaultdict(list)   # dst_id → [src_id, ...]
    in_eidxs  = defaultdict(list)   # dst_id → [edge_index position, ...]
    for e, (s, d) in enumerate(zip(src, dst)):
        in_nodes[int(d)].append(int(s))
        in_eidxs[int(d)].append(e)

    # CV-ridge: split time axis once (same split for all neurons).
    n_val   = max(1, int(CV_FRAC * T))
    n_train = T - n_val

    w_all = {m: np.zeros(E, dtype=np.float64) for m in method_keys}

    for i in tqdm(sorted(in_nodes), desc="per-neuron SVD recovery", ncols=80):
        pre    = np.array(in_nodes[i])    # presynaptic neuron ids, shape (d_i,)
        eidxs  = np.array(in_eidxs[i])   # corresponding edge positions
        d_i    = len(pre)

        # Activity matrix for neuron i.
        # H_i[t, k] = ReLU(v_{pre[k]}(t)), shape (T, d_i).
        H_i = h[:, pre]
        b_i = b[:, i]          # (T,) RHS for this neuron

        # Full SVD: H_i = U Σ V^T
        #   U:     (T, r)  — left singular vectors
        #   sigma: (r,)    — singular values in descending order
        #   Vt:    (r, d_i)— right singular vectors transposed
        # where r = min(T, d_i).
        U, sigma, Vt = np.linalg.svd(H_i, full_matrices=False)

        # Project the RHS onto the left singular basis once; reused by all methods.
        # UtB[k] = U[:, k]^T b_i  (k-th component of b in the singular basis)
        UtB       = U.T @ b_i              # (r,)
        total_var = float(np.sum(sigma**2))
        active    = total_var > 1e-16      # False if neuron has zero presynaptic input

        for method in method_keys:

            # ------------------------------------------------------------------
            # Truncated SVD  ("svd_XX" where XX is the variance threshold in %)
            # ------------------------------------------------------------------
            if method.startswith("svd_"):
                pct    = int(method.split("_")[1])   # e.g. 90, 95, 99, 100
                thresh = min(pct / 100.0, 1.0)

                if active:
                    # Cumulative variance explained by the top-k singular values.
                    # cumvar[k] = (sigma[0]^2 + ... + sigma[k]^2) / total_var
                    cumvar = np.cumsum(sigma**2) / total_var

                    # r_i = smallest rank such that cumvar[r_i - 1] >= thresh.
                    # searchsorted returns the first index where cumvar >= thresh;
                    # adding 1 converts from 0-based index to rank count.
                    r_i = int(np.searchsorted(cumvar, thresh)) + 1
                    r_i = min(r_i, len(sigma))   # never exceed the full rank
                else:
                    r_i = 1   # inactive neuron: rank-1 fallback avoids division by zero

                # Pseudoinverse using only the top r_i components:
                #   w_i = V_r Σ_r^{-1} U_r^T b_i
                # where V_r, Σ_r, U_r are the rank-r_i truncation.
                # s_inv[k] = 1/sigma[k] if sigma[k] > floor, else 0.
                s_inv = np.where(sigma[:r_i] > 1e-12, 1.0 / sigma[:r_i], 0.0)
                w_i   = Vt[:r_i].T @ (s_inv * UtB[:r_i])

            # ------------------------------------------------------------------
            # Ridge regression  (closed form via SVD)
            # ------------------------------------------------------------------
            elif method == "ridge":
                # Ridge solution: w = (H^T H + lambda I)^{-1} H^T b
                # In the SVD basis this becomes:
                #   w_i = V diag(sigma_k / (sigma_k^2 + lambda)) U^T b_i
                # All singular values are used (no truncation); small sigma_k
                # are damped towards zero by the lambda term.
                sf  = sigma / (sigma**2 + RIDGE_LAMBDA)
                w_i = Vt.T @ (sf * UtB)

            # ------------------------------------------------------------------
            # Cross-validated ridge  (lambda selected per neuron)
            # ------------------------------------------------------------------
            elif method == "cv_ridge":
                # Fit lambda on the training split using the training-split SVD,
                # then refit on all T frames with the best lambda.

                H_tr, b_tr = H_i[:n_train], b_i[:n_train]
                H_val, b_val = H_i[n_train:], b_i[n_train:]

                # SVD of the training split (separate from the full-data SVD above).
                U_tr, s_tr, Vt_tr = np.linalg.svd(H_tr, full_matrices=False)
                UtB_tr = U_tr.T @ b_tr

                best_lam, best_err = CV_LAMBDAS[0], np.inf
                for lam in CV_LAMBDAS:
                    # Candidate solution on train split
                    sf_tr   = s_tr / (s_tr**2 + lam)
                    w_try   = Vt_tr.T @ (sf_tr * UtB_tr)
                    # Validation mean-squared error
                    val_err = float(np.mean((H_val @ w_try - b_val)**2))
                    if val_err < best_err:
                        best_lam, best_err = lam, val_err

                # Refit on full T frames with the selected lambda
                sf  = sigma / (sigma**2 + best_lam)
                w_i = Vt.T @ (sf * UtB)

            else:
                raise ValueError(f"Unknown method: {method!r}")

            w_all[method][eidxs] = w_i

    return w_all


# ===========================================================================
# Section 4: Connectivity R²
# ===========================================================================

def connectivity_r2(w_gt, w_rec):
    """Coefficient of determination between recovered and ground-truth weights.

    R^2 = 1 - SS_res / SS_tot
    where SS_res = ||w_gt - w_rec||^2 and SS_tot = ||w_gt - mean(w_gt)||^2.

    A value of 1.0 means perfect recovery; 0.0 means the recovered weights
    are no better than predicting the mean; negative values indicate worse.
    This is a weight-space metric only, independent of ODE dynamics.

    Args:
        w_gt:  (E,) ground-truth edge weights.
        w_rec: (E,) recovered edge weights.
    Returns:
        r2: scalar float.
    """
    ss_res = np.sum((w_gt - w_rec)**2)
    ss_tot = np.sum((w_gt - w_gt.mean())**2)
    return float(1.0 - ss_res / (ss_tot + 1e-16))


# ===========================================================================
# Section 5: ODE rollout and Pearson r
# ===========================================================================

def make_ode(ode_params, neuron_types, device):
    """Instantiate the FlyVisODE for the graded-potential model."""
    n_types = int(neuron_types.max().item()) + 1
    return FlyVisODE(
        ode_params    = ode_params,
        g_phi         = torch.nn.functional.relu,
        params        = [],
        model_type    = MODEL_TYPE,
        n_neuron_types= n_types,
        device        = device,
    )


def make_state(n, neuron_types, v0, device):
    """Create a NeuronState initialised with voltage v0."""
    return NeuronState(
        index      = torch.arange(n, dtype=torch.long,    device=device),
        pos        = torch.zeros(n, 2, dtype=torch.float32, device=device),
        voltage    = v0.clone().to(device),
        stimulus   = torch.zeros(n, dtype=torch.float32,  device=device),
        group_type = torch.zeros(n, dtype=torch.long,     device=device),
        neuron_type= neuron_types.to(device),
        calcium    = torch.zeros(n, dtype=torch.float32,  device=device),
        fluorescence=torch.zeros(n, dtype=torch.float32,  device=device),
        noise      = torch.zeros(n, dtype=torch.float32,  device=device),
    )


def run_gt_rollout(ode_params, stim_all, neuron_types, v0, device):
    """Run the ground-truth ODE and return the full voltage trajectory.

    The trajectory v_gt[t] is stored on CPU to save GPU memory.

    Args:
        ode_params:   FlyVisODEParams with ground-truth W.
        stim_all:     (T, N) stimulus tensor.
        neuron_types: (N,) cell-type ids.
        v0:           (N,) initial voltage.
        device:       torch device.

    Returns:
        v_gt: (T, N) float32 CPU tensor.
    """
    n  = ode_params.tau_i.shape[0]
    T  = stim_all.shape[0]
    ode = make_ode(ode_params, neuron_types, device)
    x   = make_state(n, neuron_types, v0, device)
    ei  = ode_params.edge_index.to(device)
    stim_all = stim_all.to(device)

    v_gt = torch.zeros(T, n, dtype=torch.float32)   # stored on CPU

    with torch.no_grad():
        for t in tqdm(range(T), desc="GT rollout", ncols=80):
            x.stimulus[:] = stim_all[t]
            v_gt[t]       = x.voltage.cpu()
            dv = ode(x, ei)
            x.voltage = x.voltage + DT * dv.squeeze()

    return v_gt


def run_variant_rollout(ode_params, stim_all, neuron_types, v0, v_gt, device):
    """Run one variant ODE rollout and return mean Pearson r vs ground truth.

    Pearson r at timestep t is computed between the N-dimensional voltage
    vectors v_variant[t] and v_gt[t]; the mean across all T timesteps is
    returned.

    Args:
        ode_params:   FlyVisODEParams with recovered W.
        stim_all:     (T, N) stimulus (same as GT rollout).
        neuron_types: (N,) cell-type ids.
        v0:           (N,) initial voltage (same as GT rollout).
        v_gt:         (T, N) ground-truth trajectory from run_gt_rollout.
        device:       torch device.

    Returns:
        pearson_mean: mean over all timesteps of the per-timestep Pearson r
                      between v_variant[t] and v_gt[t].
    """
    n  = ode_params.tau_i.shape[0]
    T  = stim_all.shape[0]
    ode = make_ode(ode_params, neuron_types, device)
    x   = make_state(n, neuron_types, v0, device)
    ei  = ode_params.edge_index.to(device)
    stim_all = stim_all.to(device)

    pearson_t = np.zeros(T, dtype=np.float64)

    with torch.no_grad():
        for t in range(T):
            x.stimulus[:] = stim_all[t]

            # Per-timestep Pearson r between variant and ground-truth voltage.
            # Both vectors have shape (N,); we compute the standard correlation.
            v_var = x.voltage.cpu().numpy()       # (N,)
            v_ref = v_gt[t].numpy()               # (N,)

            v_var_c = v_var - v_var.mean()        # zero-centred
            v_ref_c = v_ref - v_ref.mean()
            denom   = (np.linalg.norm(v_var_c) * np.linalg.norm(v_ref_c))
            pearson_t[t] = np.dot(v_var_c, v_ref_c) / denom if denom > 0 else 0.0

            dv = ode(x, ei)
            x.voltage = x.voltage + DT * dv.squeeze()

    # Return the mean Pearson r across all timesteps as the scalar summary.
    return float(np.mean(pearson_t))


# ===========================================================================
# Section 6: Build ODE params with a different W
# ===========================================================================

def params_with_w(state, w_rec, device):
    """Return a FlyVisODEParams with all fields from state but W replaced.

    Args:
        state: dict loaded from ode_params.pt (contains W, tau_i, edge_index, …).
        w_rec: (E,) numpy array of recovered edge weights.
        device: torch device.
    Returns:
        FlyVisODEParams instance on device.
    """
    s = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
    s["W"] = torch.tensor(w_rec, dtype=torch.float32)
    return FlyVisODEParams(**s).to(device)


# ===========================================================================
# Section 7: LaTeX table output
# ===========================================================================

def fmt(x, decimals=2):
    """Format a float to a fixed number of decimal places."""
    return f"{x:.{decimals}f}"


def write_latex_table(results, out_path):
    """Write a LaTeX tabular environment to out_path.

    Layout: rows = methods, column groups = noise levels,
    each group has two columns (R^2_W, rollout Pearson r).

    Args:
        results: dict[(method_key, noise_key)] → {"r2_W": float, "pearson": float}
        out_path: path to the output .tex file.
    """
    noise_keys   = list(NOISE_CONDITIONS.keys())
    sigma_labels = [NOISE_CONDITIONS[k]["sigma_label"] for k in noise_keys]
    n_sigma      = len(noise_keys)

    lines = []

    # ---- table header ----
    # Column spec: method name column + two columns per noise level
    col_spec = "l" + "rr" * n_sigma
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # First header row: noise-level group labels spanning 2 columns each
    top = ""
    for label in sigma_labels:
        top += r" & \multicolumn{2}{c}{$\sigma=" + label.strip("$") + r"$}"
    lines.append(r"Method" + top + r" \\")

    # Cmidrules under each noise-level group
    cmidrules = ""
    for g in range(n_sigma):
        col_start = 2 + g * 2           # 1-indexed: method col=1, then pairs
        col_end   = col_start + 1
        cmidrules += rf"\cmidrule(lr){{{col_start}-{col_end}}}"
    lines.append(cmidrules)

    # Second header row: column sub-labels
    sub = r"& $R^2_{\mathbf{W}}$ & $r$" * n_sigma
    lines.append(r" " + sub + r" \\")
    lines.append(r"\midrule")

    # ---- data rows ----
    for method_key, method_label in METHODS:
        row = method_label
        for noise_key in noise_keys:
            key = (method_key, noise_key)
            if key in results:
                d    = results[key]
                row += " & " + fmt(d["r2_W"]) + " & " + fmt(d["pearson"])
            else:
                row += " & --- & ---"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  LaTeX table written to {out_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    method_keys = [m for m, _ in METHODS]

    # results[(method_key, noise_key)] = {"r2_W": float, "pearson": float}
    results = {}

    for noise_key, cfg in NOISE_CONDITIONS.items():
        ode_path = cfg["ode_path"]
        data_dir = cfg["data_dir"]
        print(f"\n{'='*60}")
        print(f"Noise condition: {noise_key}")
        print(f"{'='*60}")

        # ---- load ODE parameters (ground-truth W, tau, V_rest, edge_index) ----
        state  = torch.load(ode_path, map_location="cpu", weights_only=True)
        w_gt   = state["W"].numpy()           # (E,) ground-truth edge weights
        tau    = state["tau_i"].numpy()       # (N,) time constants
        v_rest = state["V_i_rest"].numpy()    # (N,) resting potentials
        ei     = state["edge_index"]          # (2, E) edge list

        # ---- load training data for the linear system ----
        print("[1/3] Loading training data for H_i w_i = b_i ...")
        v      = load_voltage(data_dir)       # (T, N) voltage
        dv_dt  = load_derivatives(data_dir)   # (T, N) dv/dt
        stim   = load_stimulus(data_dir)      # (T, N) stimulus

        # Presynaptic activity: h_j(t) = ReLU(v_j(t))
        # This is the signal that actually drives postsynaptic neurons.
        h = np.maximum(0.0, v)               # (T, N)

        # ODE right-hand side: b_i(t) = tau_i dv_i/dt + v_i - V_rest_i - stim_i
        b = compute_rhs(v, dv_dt, stim, tau, v_rest)   # (T, N)

        # ---- recover weights with all methods ----
        print("[2/3] Recovering weights ...")
        w_all = recover_weights_all_methods(h, b, ei, method_keys)

        # ---- load rollout data and run GT rollout once ----
        print("[3/3] Running ODE rollouts ...")
        stim_ro  = load_stimulus_rollout(data_dir)
        v0       = load_v0(data_dir)
        ntypes   = load_neuron_types(data_dir)

        gt_params = FlyVisODEParams(**state).to(device)
        v_gt_traj = run_gt_rollout(gt_params, stim_ro, ntypes, v0, device)

        # ---- rollout + metrics per method ----
        for method_key in method_keys:
            w_rec  = w_all[method_key]
            r2_W   = connectivity_r2(w_gt, w_rec)

            var_params = params_with_w(state, w_rec, device)
            pearson    = run_variant_rollout(var_params, stim_ro, ntypes, v0, v_gt_traj, device)

            results[(method_key, noise_key)] = {"r2_W": r2_W, "pearson": pearson}
            print(f"  [{method_key:12s}]  R²_W={r2_W:.4f}  Pearson r={pearson:.4f}")

    # ---- write outputs ----
    print(f"\n{'='*60}")
    print("Writing outputs ...")

    # JSON with full-precision numbers
    json_path = os.path.join(SCRIPT_DIR, "pseudoinverse_table.json")
    json_data = {f"{mk}|{nk}": v for (mk, nk), v in results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  JSON saved to {json_path}")

    # LaTeX table
    tex_path = os.path.join(SCRIPT_DIR, "pseudoinverse_table.tex")
    write_latex_table(results, tex_path)

    # Human-readable console table
    print(f"\n{'Method':<20}", end="")
    for nk in NOISE_CONDITIONS:
        sigma = NOISE_CONDITIONS[nk]["sigma_label"]
        print(f"  {sigma:>18}", end="")
    print()
    print(" " * 20, end="")
    for _ in NOISE_CONDITIONS:
        print(f"  {'R²_W':>8}  {'Pearson r':>9}", end="")
    print()
    print("-" * (20 + len(NOISE_CONDITIONS) * 22))
    for mk, mlabel in METHODS:
        print(f"{mlabel:<20}", end="")
        for nk in NOISE_CONDITIONS:
            d = results.get((mk, nk), {})
            if d:
                print(f"  {d['r2_W']:8.4f}  {d['pearson']:9.4f}", end="")
            else:
                print(f"  {'---':>8}  {'---':>9}", end="")
        print()


if __name__ == "__main__":
    main()
