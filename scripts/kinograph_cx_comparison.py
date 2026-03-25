"""Side-by-side comparison: Ashok's trial-based vs continuous CX simulation.

Generates a 2x3 figure comparing:
  Left column:  Ashok's original (hard resets, step velocity, single-shot EPG bump)
  Right column: Our continuous (no resets, OU velocity, periodic landmark cues)

Row 1: Activity kinograph (neurons x time)
Row 2: EPG stimulus channels (46 channels x time)
Row 3: Angular velocity input

Plus quantitative analysis: SVD rank, autocorrelation, velocity statistics.

Usage:
    conda activate neural-graph-linux
    cd /workspace/connectome-gnn
    python scripts/kinograph_cx_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from connectome_gnn.generators.connconstr_data import (
    load_drosophila_cx_connectome, generate_cx_stimulus,
)
from connectome_gnn.generators.ode_params import DrosophilaCxODEParams
from connectome_gnn.neuron_state import NeuronState


# ─── Ashok's original stimulus (from paper repo) ───

def give_velInp(ts, tau=1.0, p0=0.4, runt0=0.5, steps=2, ptot0=0.1):
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
        inps[tr, 0:5, 0:46] = subbump46
        gain = max(0.2, mu_amp + s_amp * np.random.randn())
        inps[tr, 0:5, 0:46] *= gain
        inps[tr, :, -2] = extInp
        inps[tr, :, -1] = -extInp

    return inps, ts


def run_teacher(ode_params, pde, edge_index, stim_per_neuron, n_neurons, dt,
                device, datapath, n_frames, trial_len=0):
    """Run teacher ODE, optionally with hard resets at trial boundaries."""
    x = NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=torch.zeros(n_neurons, 2, dtype=torch.float32, device=device),
        voltage=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        stimulus=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        group_type=torch.zeros(n_neurons, dtype=torch.long, device=device),
        neuron_type=ode_params.neuron_types if ode_params.neuron_types is not None
                    else torch.zeros(n_neurons, dtype=torch.long, device=device),
        calcium=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        fluorescence=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        noise=torch.zeros(n_neurons, dtype=torch.float32, device=device),
    )
    ode_params.init_state(x.voltage, datapath=datapath, device=device)

    voltage_history = np.zeros((n_frames, n_neurons), dtype=np.float32)
    with torch.no_grad():
        for t in range(n_frames):
            if trial_len > 0 and t > 0 and t % trial_len == 0:
                ode_params.init_state(x.voltage, datapath=datapath, device=device)
            x.stimulus[:] = stim_per_neuron[t]
            voltage_history[t] = x.voltage.cpu().numpy()
            dv = pde(x, edge_index)
            x.voltage = x.voltage + dt * dv.squeeze()

    return voltage_history


def svd_rank(X, threshold=0.99):
    """Effective rank at given cumulative variance threshold."""
    U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    cumvar = np.cumsum(S ** 2) / np.sum(S ** 2)
    return int(np.searchsorted(cumvar, threshold) + 1), S


def main():
    datapath = "papers/Code_NN/Code_NN/Data/Figure5"
    hemibrain_dir = os.path.join(datapath, "exported-traced-adjacencies-v1.2")
    device = torch.device('cpu')

    # Load teacher
    print("Loading CX teacher...")
    ode_params = DrosophilaCxODEParams.from_pretrained(datapath, device=device)
    pde = ode_params.create_ode(device=device)
    edge_index = ode_params.edge_index
    n_neurons = ode_params.get_n_neurons()
    dt = ode_params.get_dt()
    winp_np = ode_params.winp.cpu().numpy()
    cx_data = load_drosophila_cx_connectome(hemibrain_dir)

    # ─── A. Ashok's trial-based simulation ───
    n_trials = 50
    T_trial = 6.0
    trial_len = int(T_trial / dt)  # 60
    n_frames_ashok = n_trials * trial_len  # 3000

    np.random.seed(42)
    torch.manual_seed(42)
    print(f"Generating Ashok stimulus: {n_trials} trials x {trial_len} frames...")
    inps_ashok, _ = generate_targets_ashok(
        n_trials, cx_data['epg_ix'], cx_data['W_16to46'], T=T_trial, dt=dt,
    )
    inps_ashok_flat = inps_ashok.reshape(-1, 48)  # (3000, 48)
    stim_ashok = torch.tensor(
        2.5 * (inps_ashok_flat @ winp_np), dtype=torch.float32, device=device,
    )  # (3000, N)

    print("Running Ashok simulation (with hard resets)...")
    volt_ashok = run_teacher(
        ode_params, pde, edge_index, stim_ashok, n_neurons, dt, device,
        datapath, n_frames_ashok, trial_len=trial_len,
    )

    # ─── B. Continuous simulation ───
    n_frames_cont = 3000  # same length for fair comparison

    np.random.seed(42)
    torch.manual_seed(42)
    print(f"Generating continuous stimulus: {n_frames_cont} frames...")
    inps_cont = generate_cx_stimulus(
        n_frames_cont, cx_data['epg_ix'], cx_data['W_16to46'], seed=42,
    )  # (3000, 48)
    stim_cont = torch.tensor(
        2.5 * (inps_cont @ winp_np), dtype=torch.float32, device=device,
    )

    print("Running continuous simulation (no resets)...")
    volt_cont = run_teacher(
        ode_params, pde, edge_index, stim_cont, n_neurons, dt, device,
        datapath, n_frames_cont, trial_len=0,
    )

    # ─── Quantitative analysis ───
    rank99_ashok, S_ashok = svd_rank(volt_ashok, 0.99)
    rank99_cont, S_cont = svd_rank(volt_cont, 0.99)
    rank90_ashok, _ = svd_rank(volt_ashok, 0.90)
    rank90_cont, _ = svd_rank(volt_cont, 0.90)

    vel_ashok = inps_ashok_flat[:, -2]  # PEN right channel
    vel_cont_right = inps_cont[:, -2]
    frac_zero_ashok = np.mean(np.abs(vel_ashok) < 1e-6)
    frac_zero_cont = np.mean(np.abs(vel_cont_right) < 1e-6)
    n_unique_ashok = len(np.unique(np.round(vel_ashok, 3)))
    vel_std_ashok = np.std(vel_ashok)
    vel_std_cont = np.std(vel_cont_right)

    # EPG stimulus statistics
    epg_ashok = inps_ashok_flat[:, :46]
    epg_cont = inps_cont[:, :46]
    frac_epg_active_ashok = np.mean(np.abs(epg_ashok).sum(axis=1) > 0.01)
    frac_epg_active_cont = np.mean(np.abs(epg_cont).sum(axis=1) > 0.01)

    print("\n" + "=" * 70)
    print("QUANTITATIVE COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<40} {'Ashok':>12} {'Continuous':>12}")
    print("-" * 70)
    print(f"{'Activity rank (99% var)':<40} {rank99_ashok:>12d} {rank99_cont:>12d}")
    print(f"{'Activity rank (90% var)':<40} {rank90_ashok:>12d} {rank90_cont:>12d}")
    print(f"{'Velocity: fraction zero':<40} {frac_zero_ashok:>12.1%} {frac_zero_cont:>12.1%}")
    print(f"{'Velocity: std':<40} {vel_std_ashok:>12.3f} {vel_std_cont:>12.3f}")
    print(f"{'Velocity: unique values':<40} {n_unique_ashok:>12d} {'continuous':>12}")
    print(f"{'EPG: fraction of frames active':<40} {frac_epg_active_ashok:>12.1%} {frac_epg_active_cont:>12.1%}")
    print(f"{'Hard resets':<40} {'every 60 fr':>12} {'none':>12}")
    print(f"{'Trial structure':<40} {'50 x 6s':>12} {'continuous':>12}")
    print("=" * 70)

    # ─── Plot ───
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[3, 1, 1, 0.8], hspace=0.35, wspace=0.15)

    # Shared color scales
    vmax_act = max(np.percentile(np.abs(volt_ashok), 99),
                   np.percentile(np.abs(volt_cont), 99))
    vmax_epg = max(np.percentile(np.abs(epg_ashok[epg_ashok > 0]), 98) if np.any(epg_ashok > 0) else 1,
                   np.percentile(np.abs(epg_cont[epg_cont > 0]), 98) if np.any(epg_cont > 0) else 1)
    vel_ylim = max(np.abs(vel_ashok).max(), np.abs(vel_cont_right).max()) * 1.1

    titles = [
        f"Beiran & Litwin-Kumar (2023)\n"
        f"Trial-based, hard reset  |  rank(99%)={rank99_ashok}",
        f"This work (continuous)\n"
        f"No resets, OU velocity  |  rank(99%)={rank99_cont}",
    ]
    datasets = [
        (volt_ashok, epg_ashok, vel_ashok, -inps_ashok_flat[:, -1]),
        (volt_cont, epg_cont, vel_cont_right, -inps_cont[:, -1]),
    ]

    for col, (volt, epg, vel_r, vel_l) in enumerate(datasets):
        n_frames = volt.shape[0]

        # Row 0: Activity
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(volt.T, aspect='auto', cmap='viridis',
                       interpolation='nearest', origin='lower',
                       vmin=-vmax_act, vmax=vmax_act)
        ax.set_title(titles[col], fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{n_neurons} neurons' if col == 0 else '')
        if col == 0:
            for i in range(1, n_trials):
                ax.axvline(i * trial_len, color='red', linewidth=0.6, alpha=0.8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

        # Row 1: EPG channels
        ax = fig.add_subplot(gs[1, col])
        im2 = ax.imshow(epg.T, aspect='auto', cmap='hot',
                        interpolation='nearest', origin='lower',
                        vmin=0, vmax=vmax_epg)
        ax.set_ylabel('EPG ch.' if col == 0 else '')
        if col == 0:
            for i in range(1, n_trials):
                ax.axvline(i * trial_len, color='red', linewidth=0.6, alpha=0.8)
            ax.text(0.02, 0.85, 'bump in first\n5 frames only',
                    transform=ax.transAxes, fontsize=9, color='white',
                    fontweight='bold', va='top')
        else:
            ax.text(0.02, 0.85, 'periodic landmarks\n(random intervals)',
                    transform=ax.transAxes, fontsize=9, color='white',
                    fontweight='bold', va='top')
        fig.colorbar(im2, ax=ax, fraction=0.02, pad=0.02)

        # Row 2: Velocity
        ax = fig.add_subplot(gs[2, col])
        ax.plot(vel_r, label='PEN right', alpha=0.8, linewidth=0.6, color='C0')
        ax.plot(vel_l, label='PEN left', alpha=0.8, linewidth=0.6, color='C1')
        ax.set_ylim(-vel_ylim, vel_ylim)
        ax.set_ylabel('velocity' if col == 0 else '')
        ax.set_xlabel('frame')
        ax.legend(fontsize=8, loc='upper right')
        if col == 0:
            for i in range(1, n_trials):
                ax.axvline(i * trial_len, color='red', linewidth=0.6, alpha=0.5)
            ax.text(0.02, 0.92, f'step function\n{n_unique_ashok} unique values\n{frac_zero_ashok:.0%} silent',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax.text(0.02, 0.92, f'OU process\ncontinuous\n{frac_zero_cont:.0%} silent',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Row 3: Singular value spectra (both on same axes)
    ax = fig.add_subplot(gs[3, :])
    n_sv = min(50, len(S_ashok), len(S_cont))
    ax.semilogy(range(1, n_sv + 1), (S_ashok[:n_sv] ** 2) / (S_ashok ** 2).sum(),
                'o-', markersize=4, label=f'Ashok trial-based  (rank₉₉={rank99_ashok})', color='C3')
    ax.semilogy(range(1, n_sv + 1), (S_cont[:n_sv] ** 2) / (S_cont ** 2).sum(),
                's-', markersize=4, label=f'Continuous  (rank₉₉={rank99_cont})', color='C0')
    ax.axhline(0.01, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(n_sv - 2, 0.012, '1% variance', fontsize=8, color='gray', ha='right')
    ax.set_xlabel('singular value index')
    ax.set_ylabel('fraction of variance')
    ax.set_title('Activity singular value spectrum', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, n_sv + 0.5)

    outpath = "graphs_data/drosophila_cx/kinograph_comparison.png"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    plt.close()


if __name__ == "__main__":
    main()
