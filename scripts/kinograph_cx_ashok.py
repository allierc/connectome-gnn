"""Generate kinograph of Drosophila CX using Ashok's original trial-based stimulus.

This reproduces the original Beiran & Litwin-Kumar (2023) stimulus generation
(generate_targets + give_velInp) with hard resets every trial, and runs the
teacher ODE to produce activity for kinograph visualization.

Ref: papers/Code_NN/Code_NN/nn_fig5_drosophilaCx_teacher.py

Usage:
    conda activate neural-graph-linux
    cd /workspace/connectome-gnn
    python scripts/kinograph_cx_ashok.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt

from connectome_gnn.generators.connconstr_data import load_drosophila_cx_connectome
from connectome_gnn.generators.ode_params import DrosophilaCxODEParams
from connectome_gnn.neuron_state import NeuronState


# ─── Ashok's original stimulus functions (verbatim from paper repo) ───

def give_velInp(ts, tau=1.0, p0=0.4, runt0=0.5, steps=2, ptot0=0.1):
    """Original velocity input: sparse random step-function pulses.

    Ref: nn_fig5_drosophilaCx_teacher.py lines 33-44
    """
    inp = np.zeros_like(ts)
    if np.random.rand() > ptot0:
        runt = runt0
        while runt < ts[-1]:
            runtp = runt + np.random.exponential(scale=tau)
            if np.random.rand() < p0:
                inp[(ts >= runt) * (ts < runtp)] = 0.
            else:
                inp[(ts >= runt) * (ts < runtp)] = 0.5 * np.sign(np.random.randn()) * (np.random.randint(steps) + 1)
            runt = runtp
    return inp


def generate_targets_ashok(trials, epg_ix, W_16to46, T=6, dt=0.1,
                           mu_amp=1., s_amp=0.2, AvgRate=1.):
    """Original trial-based stimulus generation.

    Ref: nn_fig5_drosophilaCx_teacher.py lines 46-86

    Each trial:
    - Random orientation bump injected in first 5 frames only
    - Sparse velocity pulses via give_velInp
    - Hard reset between trials (state reset to h0)
    """
    x = np.linspace(-1, 1, 1000)
    bump = np.exp(-(x / (3 / 16)) ** 2)

    ts = np.arange(0, T, dt)
    n_glom = len(np.unique(epg_ix))
    x_new = np.linspace(0, 1, n_glom)
    x_old = np.linspace(0, 1, len(bump))
    inps = np.zeros((trials, len(ts), 46 + 2))

    for tr in range(trials):
        ori = np.random.rand() * 2 * np.pi - np.pi
        i_ori = int((len(x) / 2) * ori / np.pi)
        bump_shift = np.roll(bump, i_ori)
        subbump = np.interp(x_new, x_old, bump_shift)
        subbump = AvgRate * subbump / np.mean(subbump)

        extInp = give_velInp(ts)
        subbump46 = W_16to46.dot(subbump)

        # Visual bump: only first 5 frames
        inps[tr, 0:5, 0:46] = subbump46
        gain = max(0.2, mu_amp + s_amp * np.random.randn())
        inps[tr, 0:5, 0:46] *= gain

        # Velocity: bilateral push-pull
        inps[tr, :, -2] = extInp
        inps[tr, :, -1] = -extInp

    return inps, ts


# ─── Main ───

def main():
    datapath = "papers/Code_NN/Code_NN/Data/Figure5"
    device = torch.device('cpu')
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load pretrained teacher
    print("Loading CX teacher...")
    ode_params = DrosophilaCxODEParams.from_pretrained(datapath, device=device)
    pde = ode_params.create_ode(device=device)
    edge_index = ode_params.edge_index
    n_neurons = ode_params.get_n_neurons()
    dt = ode_params.get_dt()

    # Load connectome for stimulus generation (need epg_ix, W_16to46)
    cx_data = load_drosophila_cx_connectome(
        os.path.join(datapath, "exported-traced-adjacencies-v1.2")
    )

    # Generate Ashok's trial-based stimulus
    n_trials = 50
    T_trial = 6.0  # seconds
    trial_len = int(T_trial / dt)  # 60 frames per trial

    print(f"Generating Ashok stimulus: {n_trials} trials x {trial_len} frames = {n_trials * trial_len} total frames")
    inps, ts = generate_targets_ashok(
        n_trials, cx_data['epg_ix'], cx_data['W_16to46'],
        T=T_trial, dt=dt,
    )

    # Project raw 48-channel stimulus to per-neuron stimulus (N,)
    # using the input weight matrix from the teacher
    # Ref: ode_params.generate_stimulus does cx_inps @ winp * 2.5
    winp_np = ode_params.winp.cpu().numpy()  # (48, N)

    # Simulate with hard resets
    print("Running teacher simulation with hard resets...")
    voltage_all = np.zeros((n_trials, trial_len, n_neurons))
    stimulus_all = np.zeros((n_trials, trial_len, n_neurons))

    x = NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=torch.zeros(n_neurons, 2, dtype=torch.float32, device=device),
        voltage=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        stimulus=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        group_type=torch.zeros(n_neurons, dtype=torch.long, device=device),
        neuron_type=ode_params.neuron_types if hasattr(ode_params, 'neuron_types') and ode_params.neuron_types is not None
                    else torch.zeros(n_neurons, dtype=torch.long, device=device),
        calcium=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        fluorescence=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        noise=torch.zeros(n_neurons, dtype=torch.float32, device=device),
    )

    with torch.no_grad():
        for trial in range(n_trials):
            # HARD RESET: reinit state at start of each trial
            ode_params.init_state(x.voltage, datapath=datapath, device=device)

            for t in range(trial_len):
                # Project 48-channel stimulus to per-neuron input
                stim_48 = inps[trial, t]  # (48,)
                per_neuron_stim = 2.5 * (stim_48 @ winp_np)  # (N,)
                per_neuron_stim = torch.tensor(per_neuron_stim, dtype=torch.float32, device=device)
                x.stimulus[:] = per_neuron_stim

                voltage_all[trial, t] = x.voltage.cpu().numpy()
                stimulus_all[trial, t] = x.stimulus.cpu().numpy()

                # Euler step
                dv = pde(x, edge_index)
                x.voltage = x.voltage + dt * dv.squeeze()

    # Reshape to continuous for kinograph: (n_trials * trial_len, N)
    voltage_flat = voltage_all.reshape(-1, n_neurons)
    stimulus_flat = stimulus_all.reshape(-1, n_neurons)
    n_total = voltage_flat.shape[0]

    print(f"Simulation complete: {n_total} frames, {n_neurons} neurons")

    # ─── Plot kinograph ───
    fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    # Panel 1: Activity kinograph (neurons x time)
    ax = axes[0]
    vmax_act = np.percentile(np.abs(voltage_flat), 99)
    im = ax.imshow(voltage_flat.T, aspect='auto', cmap='viridis',
                   interpolation='nearest', origin='lower',
                   vmin=-vmax_act, vmax=vmax_act)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    ax.set_ylabel(f'{n_neurons} neurons')
    ax.set_title(f"Ashok's trial-based stimulus ({n_trials} trials x {trial_len} frames, hard reset)")

    # Draw trial boundaries (red lines in all panels)
    for ax_i in axes:
        for i in range(1, n_trials):
            ax_i.axvline(i * trial_len, color='red', linewidth=0.8, alpha=0.7)

    # Panel 2: Stimulus (raw 48-channel, flattened across trials)
    ax = axes[1]
    inps_flat = inps.reshape(-1, 48)
    im2 = ax.imshow(inps_flat[:, :46].T, aspect='auto', cmap='hot',
                    interpolation='nearest', origin='lower')
    fig.colorbar(im2, ax=ax, fraction=0.02, pad=0.02)
    ax.set_ylabel('EPG channels')

    # Panel 3: Velocity input
    ax = axes[2]
    ax.plot(inps_flat[:, -2], label='PEN right', alpha=0.8, linewidth=0.5)
    ax.plot(-inps_flat[:, -1], label='PEN left', alpha=0.8, linewidth=0.5)
    ax.set_ylabel('velocity')
    ax.set_xlabel('frame')
    ax.legend(fontsize=8)

    plt.tight_layout()
    outpath = "graphs_data/drosophila_cx/kinograph_ashok_trials.png"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    print(f"Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
