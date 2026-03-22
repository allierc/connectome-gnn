"""Data loaders for Beiran & Litwin-Kumar (2023) connectome-constrained models.

Paper: "Connectivity-constrained neural networks" (Ashok.pdf)
Paper repo: papers/Code_NN/Code_NN/

This module loads connectivity matrices for three biological connectomes (Fig 5):
  - Drosophila adult central complex (ring attractor)
  - Drosophila larva (premotor + motor, two-population)
  - Zebrafish oculomotor integrator

Each loader returns a dict with the connectivity matrix, neuron metadata,
and model-specific parameters. Dense J matrices are converted to sparse
(edge_index, W) format for GNN compatibility via dense_to_sparse().
"""

import copy
import os

import h5py
import numpy as np
import pandas as pd
import scipy.io
import torch


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def dense_to_sparse(J, threshold=0.0):
    """Convert dense weight matrix to sparse (edge_index, W) format.

    Args:
        J: (N, N) numpy array or torch tensor — dense connectivity matrix.
            Convention: J[post, pre] = weight from pre to post.
        threshold: minimum absolute weight to include as an edge.

    Returns:
        edge_index: (2, E) long tensor — [src, dst] pairs
        W: (E,) float tensor — edge weights
    """
    if isinstance(J, torch.Tensor):
        J_np = J.detach().cpu().numpy()
    else:
        J_np = np.array(J)

    mask = np.abs(J_np) > threshold
    dst, src = np.nonzero(mask)  # J[post, pre] → dst=post, src=pre

    edge_index = torch.stack([
        torch.tensor(src, dtype=torch.long),
        torch.tensor(dst, dtype=torch.long),
    ], dim=0)
    W = torch.tensor(J_np[dst, src], dtype=torch.float32)

    return edge_index, W


# ---------------------------------------------------------------------------
# Drosophila adult central complex (ring attractor)
# ---------------------------------------------------------------------------

def load_drosophila_cx_connectome(datapath):
    """Load hemibrain connectivity for the Drosophila central complex ring attractor.

    Ref: papers/Code_NN/Code_NN/nn_fig5_drosophilaCx_teacher.py lines 431-598

    The CX circuit includes 4 cell types from the hemibrain v1.2 dataset:
      - EPG (46 neurons, mapped to 16 glomeruli)
      - PEN (velocity-to-heading neurons)
      - Delta7 (inhibitory interneurons)
      - PEG (heading-to-velocity neurons)

    The connectivity matrix J is:
      1. Subselected from the full hemibrain adjacency matrix (line 480)
      2. Delta7 columns made inhibitory: J2[:,delta7] = -5*|J[:,delta7]| (line 521)
      3. Normalized by spectral radius: Jf = 0.9 * J2 / max(Re(eig(J2))) (line 524)
      4. Decomposed into log-space: wrec_log = log(|Jf|), mwrec = sign(Jf) (lines 587-591)
         so effective J = exp(wrec) * mwrec (line 184 of RNN.forward)

    Args:
        datapath: path to 'exported-traced-adjacencies-v1.2/' directory

    Returns:
        dict with keys:
            J_effective: (N, N) effective connectivity = exp(wrec_log) * mwrec
            wrec_log: (N, N) log-space weights
            mwrec: (N, N) sign matrix
            neuron_types: (N,) int type labels
            type_names: list of unique type names
            epg_ix: list mapping 46 EPG neurons to 16 glomeruli
            N: number of neurons
            n_epg: 46 (EPG population size)
    """
    # Load hemibrain neuron and connection data
    # Ref: nn_fig5_drosophilaCx_teacher.py lines 437-448
    neuronsall = pd.read_csv(os.path.join(datapath, "traced-neurons.csv"))
    neuronsall.sort_values(by=['instance'], ignore_index=True, inplace=True)
    conns = pd.read_csv(os.path.join(datapath, "traced-total-connections.csv"))

    Nall = len(neuronsall)
    Jall = np.zeros([Nall, Nall], dtype=np.uint32)

    idhash = dict(zip(neuronsall.bodyId, np.arange(Nall)))
    preinds = [idhash[x] for x in conns.bodyId_pre]
    postinds = [idhash[x] for x in conns.bodyId_post]
    Jall[postinds, preinds] = conns.weight

    # Identify cell types
    # Ref: lines 452-460
    types = np.array(neuronsall.type).astype(str)

    def getsubtype(t, types, resort=False):
        inds = np.nonzero([t in x for x in types])[0]
        if resort:
            sortinds = np.argsort(types[inds])
            inds = inds[sortinds]
        return inds

    # Key ring attractor cell types
    # Ref: lines 463-466
    epg = getsubtype("EPG", types, resort=False)
    pen = getsubtype("PEN", types, resort=False)
    peg = getsubtype("PEG", types, resort=False)
    delta7 = getsubtype("Delta7", types, resort=False)

    # Combine and reorder EPG neurons to match ring topology
    # Ref: lines 469-474
    allcx = np.concatenate((epg, pen, delta7, peg))
    allcx[0:46] = allcx[[
        23, 24, 0, 1, 42, 43, 44, 45, 2, 3, 39, 40, 41, 4, 5, 6,
        36, 37, 38, 7, 8, 9, 33, 34, 35, 10, 11, 12,
        30, 31, 32, 13, 14, 15, 27, 28, 29, 16, 17, 18,
        25, 26, 19, 20, 21, 22
    ]]

    # EPG glomerulus mapping: 46 neurons → 16 functional groups
    # Ref: lines 476-477
    epg_ix = [
        0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5,
        6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9,
        10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13,
        14, 14, 15, 15, 15, 15
    ]

    # Subselect connectivity for the ring attractor circuit
    # Ref: line 480
    J = 1.0 * Jall[allcx, :][:, allcx]
    N = J.shape[0]

    # Build type labels for selected neurons
    neurons = neuronsall.iloc[allcx, :]
    neurons = neurons.reset_index(drop=True)
    uniqtypes = pd.unique(neurons.type)
    typehash = dict(zip(uniqtypes, np.arange(len(uniqtypes))))
    typeclasses = np.array([typehash[x] for x in neurons.type])

    # One-hot type encoding to identify Delta7 (inhibitory)
    # Ref: lines 506-515
    Ntype = len(uniqtypes)
    types_1hot = np.zeros([N, Ntype])
    types_1hot[np.arange(N), typeclasses] = 1.0

    # Apply inhibitory sign to Delta7 neurons and normalize by spectral radius
    # Ref: lines 520-524 — Delta7 is type index -2 in the concatenation order
    # J2[:,types_1hot[:,-2]==1.] = -5*np.abs(J[:,types_1hot[:,-2]==1.])
    J2 = np.copy(J)
    J2[:, types_1hot[:, -2] == 1.0] = -5 * np.abs(J[:, types_1hot[:, -2] == 1.0])
    u = np.linalg.eigvals(J2)

    # Ref: line 524 — Jf = 0.9*J2/np.max(np.real(u))
    Jf = 0.9 * J2 / np.max(np.real(u))

    # Log-space decomposition for the RNN
    # Ref: lines 586-591
    Wr_ini = Jf
    wrec_log = np.copy(Jf)
    nonzero = np.abs(wrec_log) > 0
    wrec_log[nonzero] = np.log(np.abs(wrec_log[nonzero]))
    wrec_log[~nonzero] = -20.0  # large negative → exp(-20) ≈ 0
    mwrec = np.sign(Wr_ini)

    # Build input/output weight matrices
    # Ref: lines 575-583
    input_size = 46 + 2
    output_size = 3 + 46

    wout = np.zeros((N, output_size))
    W_16to46 = np.zeros((46, 16))
    for i in range(len(np.unique(epg_ix))):
        ixx = np.where(np.array(epg_ix) == i)[0]
        W_16to46[ixx, i] = 1

    W_46to16 = (W_16to46 / np.sum(W_16to46, 0)).T
    W_16to3 = np.zeros((16, 3))
    for i in range(16):
        ori = (i / 16) * 2 * np.pi - np.pi
        W_16to3[i, 0] = np.cos(ori)
        W_16to3[i, 1] = np.sin(ori)
    W_46to3 = W_16to3.T.dot(W_46to16)
    W_46to3[2, :] = 1.0 / 46

    wout[0:46, 0:3] = W_46to3.T
    wout[0:46, 3:] = np.eye(46)

    winp = np.zeros((input_size, N))
    for ii in range(46):
        winp[ii, ii] = 2.0
    # Ref: lines 582-583 — velocity inputs mapped to PEN neurons
    winp[-1, 50:60] = 1.0  # left PEN neurons
    winp[-2, 60:70] = 1.0  # right PEN neurons

    return {
        "J_effective": np.exp(wrec_log) * mwrec,
        "wrec_log": wrec_log,
        "mwrec": mwrec,
        "neuron_types": typeclasses,
        "type_names": list(uniqtypes),
        "epg_ix": epg_ix,
        "N": N,
        "n_epg": 46,
        "winp": winp,
        "sinp": np.zeros((input_size, 1)),
        "wout": wout,
        "input_size": input_size,
        "output_size": output_size,
        "W_46to3": W_46to3,
        "W_16to46": W_16to46,
    }


# ---------------------------------------------------------------------------
# Drosophila larva (two-population: premotor + motor)
# ---------------------------------------------------------------------------

def load_larva_connectome(datapath):
    """Load larva connectome from h5 data file.

    Ref: papers/Code_NN/Code_NN/Data/Figure5/setup.py, loadconns() lines 68-81

    The larva model has two populations:
      - PMN (premotor neurons, N neurons): recurrent via Jpp
      - MN (motor neurons, M neurons): driven by PMN via Jpm

    Connectivity is loaded from data.h5, which contains:
      - Jpm: (M, N) premotor-to-motor connectivity
      - Jpp: (N, N) premotor recurrent connectivity
      - types/nt: neuron type labels (used to assign inhibitory sign)

    Inhibitory sign assignment (initJpp, setup.py lines 150-170):
      - Neurons with 'inh' or 'unknown' in type name get negative weights

    Args:
        datapath: path to directory containing data.h5

    Returns:
        dict with connectivity and metadata
    """
    h5_path = os.path.join(datapath, "data.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Larva data file not found: {h5_path}\n"
            "Download from the connconstr paper repo: "
            "https://github.com/mbeiran/connconstr Data/Figure5/data.h5"
        )

    # Ref: setup.py loadconns() lines 68-81
    # "after loading, rows are postsynaptic and columns are presynaptic"
    f = h5py.File(h5_path, "r")
    Jpm_raw = (f["Jpm"][:].T).astype(np.float32)
    Jpp_raw = (f["Jpp"][:].T).astype(np.float32)
    pnames = f["p"][:]
    mnames = f["m"][:]
    types = f["nt"][:]
    mnorder = f["mnorder"][:]
    f.close()

    M = len(mnames)
    N = len(pnames)

    # Apply inhibitory sign based on neuron type
    # Ref: setup.py initJpp() lines 150-170
    def initJpp(J0, types):
        J = np.copy(J0)
        for qi in range(J.shape[0]):
            if 'inh' in str(types[qi]):
                J[qi, :] = -J[qi, :]
            elif 'unknown' in str(types[qi]):
                J[qi, :] = -J[qi, :]
        return J

    Jpp = initJpp(Jpp_raw, types)
    Jpm = initJpp(Jpm_raw, types)

    return {
        "Jpp": Jpp,
        "Jpm": Jpm,
        "N": N,
        "M": M,
        "types": types,
        "pnames": pnames,
        "mnames": mnames,
        "mnorder": mnorder,
    }


def load_larva_pretrained(datapath):
    """Load pre-trained larva teacher model parameters.

    Ref: papers/Code_NN/Code_NN/nn_fig5_plots_abc.py lines 31-41

    The ashokF_softplus.npz file contains the trained teacher parameters:
      Jpm, Jpp, bm, bp, taum, taup, gm, gp, wsp, p0, m0

    Args:
        datapath: path to directory containing ashokF_softplus.npz

    Returns:
        dict with all teacher model parameters
    """
    npz_path = os.path.join(datapath, "ashokF_softplus.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Larva pretrained file not found: {npz_path}\n"
            "Download from the connconstr paper repo Data/Figure5/"
        )

    # Ref: nn_fig5_plots_abc.py lines 31-41
    AA = np.load(npz_path)
    result = {
        "Jpm": AA['arr_0'],
        "Jpp": AA['arr_1'],
        "bm": AA['arr_2'],
        "bp": AA['arr_3'],
        "taum": AA['arr_4'],
        "taup": AA['arr_5'],
        "gm": AA['arr_6'],
        "gp": AA['arr_7'],
        "wsp": AA['arr_8'],
        "p0": AA['arr_9'],
        "m0": AA['arr_10'],
    }

    stim_path = os.path.join(datapath, "ashok_s.npz")
    if os.path.exists(stim_path):
        BB = np.load(stim_path)
        result["s"] = BB['arr_0']

    return result


# ---------------------------------------------------------------------------
# Zebrafish oculomotor integrator
# ---------------------------------------------------------------------------

def load_zebrafish_connectome(datapath):
    """Load zebrafish oculomotor connectivity from Goldman lab MATLAB data.

    Ref: papers/Code_NN/Code_NN/nn_fig5_zebrafish_teacher.py lines 64-179

    Processing steps:
      1. Load ConnMatrix from MATLAB (lines 65-68)
      2. Normalize by total inputs: W[i,:] = connMat[i,:] / totalInputs[i] (lines 99-102)
      3. Apply final_adjustments(): negate DOs/MOs, zero ABD/vSPNs/IBN/axial (lines 35-55)
      4. Scale by spectral radius: W = 0.9 * W / max(Re(eig(W))) (line 179)

    The ODE is linear: dr/dt = (-r + W @ r + I * v_in) / tau (line 172)

    Args:
        datapath: path to 'goldman_data/' directory

    Returns:
        dict with W, v_in, N, cell type info
    """
    # Load connectivity matrix
    # Ref: lines 65-68
    connMatFile = os.path.join(
        datapath, "ConnMatrix_CO_top500_2blocks_gamma038_08062020.mat"
    )
    connMat = scipy.io.loadmat(connMatFile)
    connMatDict = list(connMat)
    connMat = np.float32(connMat[connMatDict[-1]])
    N = connMat.shape[0]

    # Load total inputs (for normalization)
    # Ref: lines 72-76
    totalInputFile = os.path.join(
        datapath, "totalInputs_CO_top500_2blocks_gamma038_08062020.mat"
    )
    totalInputs = scipy.io.loadmat(totalInputFile)
    totalInputsDict = list(totalInputs)
    totalInputs = np.int32(totalInputs[totalInputsDict[-1]])
    totalInputs = np.ravel(totalInputs)

    # Load cell IDs
    # Ref: lines 79-83
    cellIDFile = os.path.join(
        datapath, "cellIDType_CO_top500_2blocks_gamma038_08062020.mat"
    )
    cellIDs = scipy.io.loadmat(cellIDFile)
    cellIDFileDict = list(cellIDs)
    cellIDs = cellIDs[cellIDFileDict[-1]]

    # Get cell type locations
    # Ref: lines 90-92
    cellLocations = np.array([
        (cellIDs == '_Int_'), (cellIDs == 'Ibn_m'), (cellIDs == 'Ibn_i'),
        (cellIDs == '_MOs_'), (cellIDs == '_Axlm'), (cellIDs == '_Axl_'),
        (cellIDs == '_DOs_'), (cellIDs == 'ABD_m'), (cellIDs == 'ABD_i'),
        (cellIDs == 'vSPNs')
    ])
    cellNames = ('integ', 'Ibnm', 'Ibni', 'MO', 'axlm', 'axl',
                 'vest', 'abdm', 'abdi', 'vspns')
    lb_cdf = pd.DataFrame(cellLocations, cellNames)

    # Zero ABD outgoing connections
    # Ref: lines 95-96
    connMat[:, lb_cdf.loc['abdm']] = 0
    connMat[:, lb_cdf.loc['abdi']] = 0

    # Normalize by total inputs
    # Ref: lines 99-102
    lb_Wnorm = np.zeros(connMat.shape)
    for i in range(connMat.shape[0]):
        if totalInputs[i] > 0:
            lb_Wnorm[i, :] = connMat[i, :] / totalInputs[i, None]

    # Apply final adjustments: negate inhibitory, zero output-only populations
    # Ref: final_adjustments() lines 35-55
    W = copy.deepcopy(lb_Wnorm)
    W[:, lb_cdf.loc['vest']] = -W[:, lb_cdf.loc['vest']]
    W[:, lb_cdf.loc['MO']] = -W[:, lb_cdf.loc['MO']]
    W[:, lb_cdf.loc['abdm']] = 0
    W[:, lb_cdf.loc['abdi']] = 0
    W[:, lb_cdf.loc['vspns']] = 0
    W[:, lb_cdf.loc['Ibni']] = 0
    W[:, lb_cdf.loc['Ibnm']] = 0
    W[:, lb_cdf.loc['axl']] = 0
    W[lb_cdf.loc['axl'], :] = 0
    W[:, lb_cdf.loc['axlm']] = 0
    W[lb_cdf.loc['axlm'], :] = 0

    # Compute eigenvectors for input design and spectral scaling
    # Ref: lines 176-179
    y_eig, v1 = np.linalg.eig(W)
    sort_idx = np.flip(np.argsort(np.real(y_eig)))
    y_eig = y_eig[sort_idx]
    v1 = v1[:, sort_idx]

    # Input vector: combination of leading eigenvectors + noise
    # Ref: lines 177-178
    rng = np.random.RandomState(42)
    my_v_in = (0.1 * abs(rng.randn(N))
               + np.real(np.sum(v1[:, 0:1], axis=1))
               + 1 * np.real(np.sum(v1[:, 1:3], axis=1)))

    # Scale connectivity to spectral radius 0.9
    # Ref: line 179 — simulate_series uses ynew=0.9
    ymax = np.real(y_eig[0])
    W_scaled = 0.9 * W / ymax

    return {
        "W": W_scaled,
        "v_in": my_v_in,
        "N": N,
        "cell_types": cellIDs,
        "cell_type_names": cellNames,
        "cdf": lb_cdf,
    }


def load_zebrafish_pretrained(datapath):
    """Load pre-processed zebrafish data (output of teacher script).

    Ref: papers/Code_NN/Code_NN/nn_fig5_zebrafish_teacher.py line 394
    The teacher script saves: np.savez('zebrafish.npz', W, I, v_in, dt)

    Args:
        datapath: path to directory containing zebrafish.npz

    Returns:
        dict with W, I (stimulus), v_in (input vector), dt
    """
    npz_path = os.path.join(datapath, "zebrafish.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Zebrafish pretrained file not found: {npz_path}\n"
            "Run nn_fig5_zebrafish_teacher.py first or download from paper repo."
        )

    AA = np.load(npz_path)
    return {
        "W": AA['arr_0'],
        "I": AA['arr_1'],
        "v_in": AA['arr_2'],
        "dt": float(AA['arr_3']),
    }


# ---------------------------------------------------------------------------
# Stimulus generation
# ---------------------------------------------------------------------------

def generate_cx_stimulus(n_trials, T, dt, epg_ix, W_16to46, W_46to3, seed=None):
    """Generate ring attractor stimulus (bump + velocity integration).

    Ref: papers/Code_NN/Code_NN/nn_fig5_drosophilaCx_teacher.py
         generate_targets() lines 46-86

    Creates initial bump at random orientation and velocity inputs that
    rotate the bump around the ring.

    Args:
        n_trials: number of stimulus trials
        T: total time
        dt: time step
        epg_ix: EPG glomerulus mapping (46 → 16)
        W_16to46: (46, 16) expansion matrix
        W_46to3: (3, 46) readout matrix
        seed: random seed

    Returns:
        targets: (n_trials, n_frames, 3+46) — target EPG activity
        inputs: (n_trials, n_frames, 48) — 46 EPG init + 2 velocity
        masks: (n_trials, n_frames, 3+46) — loss mask
        ts: (n_frames,) — time array
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(-1, 1, 1000)
    bump = np.exp(-(x / (3 / 16)) ** 2)

    ts = np.arange(0, T, dt)
    n_frames = len(ts)
    startMask = 3

    iMask = np.argmin(np.abs(ts - startMask))
    y = np.zeros((n_trials, n_frames, 3 + 46))
    m = np.zeros((n_trials, n_frames, 3 + 46))
    x_new = np.linspace(0, 1, len(np.unique(epg_ix)))
    x_old = np.linspace(0, 1, len(bump))
    inps = np.zeros((n_trials, n_frames, 46 + 2))

    wbump = 0.1
    s_amp = 0.2
    mu_amp = 1.0

    for tr in range(n_trials):
        # Random initial orientation
        # Ref: line 65
        ori = np.random.rand() * 2 * np.pi - np.pi
        i_ori = int((len(x) / 2) * ori / np.pi)
        bump_shift = np.roll(bump, i_ori)
        subbump = np.interp(x_new, x_old, bump_shift)
        subbump = subbump / np.mean(subbump)

        subbump46 = W_16to46.dot(subbump)
        y[tr, :, 3:] = subbump46
        y[tr, :, 0] = W_46to3[0, :].dot(subbump46)
        y[tr, :, 1] = W_46to3[1, :].dot(subbump46)
        y[tr, :, 2] = W_46to3[2, :].dot(subbump46)

        m[tr, iMask:, :] = 1.0
        m[tr, iMask:, 2] = 1.0
        m[tr, iMask:, 3:] = wbump * (1 / 46.0) * m[tr, iMask:, 3:]
        inps[tr, 0:5, 0:46] = subbump46
        # Ref: line 85 — scale input with random gain
        inps[tr, 0:5, 0:46] = max(0.2, mu_amp + s_amp * np.random.randn()) * inps[tr, 0:5, 0:46]

    return y, m, inps, ts


def generate_zebrafish_stimulus(n_frames, seed=42):
    """Generate zebrafish velocity-command stimulus (continuous filtered noise).

    The zebrafish oculomotor integrator receives velocity commands that the
    network must integrate into persistent firing rate changes (eye position).

    Ref: papers/Code_NN/Code_NN/nn_fig5_zebrafish_teacher.py
         simulate_series() lines 157-163 — original uses 3 identical pulses.

    We extend this with a continuous, smoothly varying signal to provide
    richer dynamics for GNN training: Gaussian white noise filtered by the
    same exponential kernel as the paper, producing a smooth OU-process-like
    stimulus with amplitude modulated from 0 to max over the trajectory.

    Args:
        n_frames: total number of frames
        seed: random seed for reproducibility

    Returns:
        I: (n_frames,) stimulus signal
    """
    rng = np.random.RandomState(seed)

    # Slow signal: exponential filter with ~1500 frame correlation time
    # (paper used ~100 frames which is too fast for 21000 frame trajectories)
    signal_tau = 1500
    signal_filter = np.exp(-np.arange(signal_tau * 3) / signal_tau)
    signal_filter /= signal_filter.sum()

    noise = rng.randn(n_frames + len(signal_filter))
    I_smooth = np.convolve(noise, signal_filter, mode='full')[:n_frames]

    # Slow amplitude envelope (correlation ~3000 frames)
    env_tau = 3000
    env_filter = np.exp(-np.arange(env_tau * 3) / env_tau)
    env_filter /= env_filter.sum()
    env_noise = rng.randn(n_frames + len(env_filter))
    envelope = np.convolve(env_noise, env_filter, mode='full')[:n_frames]
    # Normalize to [0, 1]
    envelope = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-12)

    max_amplitude = 400
    I = I_smooth * max_amplitude * envelope

    return I


def generate_larva_stimulus(mnorder, B, S, dt):
    """Generate larva locomotion targets and stimulus.

    Ref: papers/Code_NN/Code_NN/Data/Figure5/setup.py
         gentargets() lines 97-148, genpulse() lines 83-95

    Args:
        mnorder: (B, M) motor neuron activation ordering
        B: number of batch conditions (2: forward + backward)
        S: number of stimulus channels (2: one per direction)
        dt: time step

    Returns:
        mtarg: (T, B, M) target motor neuron activity
        s: (T, B, S) stimulus input
    """
    M = mnorder.shape[1]
    Tstop = 6
    Tpulse = 2.0
    dtpulse = 0.25
    dtpulse_end = 0.125
    tstart = 1.0
    segdelay = 1.0
    T = int(Tstop / dt)

    # Ref: setup.py genpulse() lines 83-95
    def genpulse(Tstop, dt, pstart, pend, Trise):
        T_local = int(Tstop / dt)
        p = np.zeros(T_local)
        istart = int(pstart / dt)
        iend = int(pend / dt) + 1
        irise = int(Trise / dt)
        p[istart:iend] = 1.0
        p[istart:(istart + irise)] = np.sin(np.pi * np.arange(irise) / (2.0 * irise))
        p[(iend - irise):iend] = np.flipud(np.sin(np.pi * np.arange(irise) / (2.0 * irise)))
        return p

    # Ref: setup.py gentargets() lines 111-123
    pstarts = np.zeros([B, M])
    pends = np.zeros([B, M])

    for mi in range(M):
        for bi in range(B):
            segoffset = 0
            if (bi == 1) and (mi >= int(M / 2)):
                segoffset = segdelay
            elif (bi == 0) and (mi < int(M / 2)):
                segoffset = segdelay
            pstarts[bi, mi] = tstart + (mnorder[bi, mi] - 1) * dtpulse + segoffset
            pends[bi, mi] = tstart + Tpulse + (mnorder[bi, mi] - 1) * dtpulse_end + segoffset

    mtarg = np.zeros([T, B, M])
    s = np.zeros([T, B, S], dtype=np.float32)

    for mi in range(M):
        for bi in range(B):
            if mnorder[bi, mi] > 0:
                mtarg[:, bi, mi] = genpulse(
                    Tstop, dt, pstarts[bi, mi], pends[bi, mi],
                    (pends[bi, mi] - pstarts[bi, mi]) / 2.0
                )

    # Ref: setup.py gentargets() lines 134-147
    seg2inds = np.where(np.sum(mtarg[:, 0, 0:int(M / 2)], 1) > 0)[0]
    square_pulse = np.zeros(T, dtype=np.float32)
    istart = int(tstart / dt)
    iend = np.max(seg2inds) if len(seg2inds) > 0 else T
    square_pulse[istart:iend] = 1.0

    for bi in range(int(np.ceil(B / 2))):
        s[:, bi, 0] = square_pulse
    for bi in range(int(np.ceil(B / 2)), B):
        s[:, bi, 1] = square_pulse

    return mtarg, s
