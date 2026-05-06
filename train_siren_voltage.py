"""Standalone SIREN voltage trainer.

Trains a SIREN(t) -> (n_select,) directly on ground-truth voltages of a
random subset of neurons. Used to validate that the SIREN architecture can
learn voltage dynamics before attempting the indirect-gradient hidden-neuron
setting.

Usage:
    python train_siren_voltage.py config/fly/flyvis_noise_005_hidden_010.yaml
    python train_siren_voltage.py config/fly/flyvis_noise_005_hidden_010.yaml --n_neurons 1000 --steps 50000

"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

# ── make sure the package is importable ────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.models.Siren_Network import Siren
from connectome_gnn.utils import graphs_data_path, set_data_root
from connectome_gnn.zarr_io import load_simulation_data


# ── ANSI color helpers ─────────────────────────────────────────────────────────
_G, _Y, _O, _R, _RST = '\033[92m', '\033[93m', '\033[38;5;208m', '\033[91m', '\033[0m'

def _col(r2):
    return _G if r2 > 0.7 else _Y if r2 > 0.4 else _O if r2 > 0.1 else _R


# ── linear fit helper ──────────────────────────────────────────────────────────
def _linear_fit(gt, pred):
    """Fit gt = a*pred + b globally, return (a, b, r2_temporal).

    gt, pred: (T, N) — frames x neurons
    a, b: global linear correction (fit across all neurons and frames)
    r2_temporal: mean per-neuron R² after linear correction — measures how
                 well temporal dynamics are captured, not just DC offsets.
    """
    gt_f, pred_f = gt.ravel(), pred.ravel()
    pred_mean = pred_f.mean()
    gt_mean   = gt_f.mean()
    cov  = ((pred_f - pred_mean) * (gt_f - gt_mean)).mean()
    var  = ((pred_f - pred_mean) ** 2).mean()
    a = float(cov / (var + 1e-12))
    b = float(gt_mean - a * pred_mean)

    # Per-neuron R² (temporal fit within each neuron after global a*pred+b)
    # gt, pred shape: (T, N)
    pred_corr = a * pred + b          # (T, N)
    gt_mean_n = gt.mean(axis=0)       # (N,) per-neuron mean
    ss_res = ((gt - pred_corr) ** 2).sum(axis=0)   # (N,)
    ss_tot = ((gt - gt_mean_n) ** 2).sum(axis=0)   # (N,)
    r2_per_neuron = 1.0 - ss_res / (ss_tot + 1e-12)
    r2 = float(r2_per_neuron.mean())
    return a, b, r2


# ── trace plot ─────────────────────────────────────────────────────────────────
def _plot_traces(gt_arr, pred_arr, sel_ids, step, r2, a, b, out_path, n_show=10):
    """Plot GT vs corrected SIREN traces for n_show sampled neurons."""
    n_neurons, n_frames = gt_arr.shape
    n_show = min(n_show, n_neurons)
    idx = np.linspace(0, n_neurons - 1, n_show, dtype=int)

    pred_corr = a * pred_arr + b

    activity_std = float(np.std(gt_arr))
    step_v = max(0.5, 3.0 * activity_std)

    fig, ax = plt.subplots(figsize=(15, max(4, n_show * 0.5 + 2)))
    ax.set_facecolor('white')

    for i, ni in enumerate(idx):
        gt_i   = gt_arr[ni]   - gt_arr[ni].mean()
        pred_i = pred_corr[ni] - pred_corr[ni].mean()
        ax.plot(gt_i   + i * step_v, lw=2.0, c='#66cc66', alpha=0.9,
                label='GT' if i == 0 else None)
        ax.plot(pred_i + i * step_v, lw=0.9, c='black', alpha=0.9,
                label='SIREN (corrected)' if i == 0 else None)
        ax.text(-n_frames * 0.025, i * step_v, f'n{sel_ids[ni]}',
                fontsize=8, va='bottom', ha='right')

    ax.set_xlim([-n_frames * 0.03, n_frames * 1.02])
    ax.set_ylim([-step_v, n_show * step_v + step_v])
    ax.set_yticks([])
    ax.set_xlabel('frame', fontsize=13)
    ax.set_title(
        f'SIREN voltage  step {step}   R²={r2:.3f}   a={a:.3f} b={b:.3f}',
        fontsize=12)
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Standalone SIREN voltage trainer')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--n_neurons', type=int, default=1000,
                        help='Number of neurons to select randomly (default: 1000)')
    parser.add_argument('--steps', type=int, default=50000,
                        help='Training steps (default: 50000)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Frames per step (default: 512) — SIREN needs large batches to avoid waterbed problem: fitting few frames breaks all others')
    parser.add_argument('--lr', type=float, default=1e-8,
                        help='Learning rate (default: 1e-8)')
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help='SIREN hidden dim (default: 2048)')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='SIREN hidden layers (default: 4)')
    parser.add_argument('--omega', type=float, default=4096.0,
                        help='SIREN omega (default: 4096)')
    parser.add_argument('--output_root', type=str, default=None,
                        help='Override data root (sets graphs_data/ location)')
    parser.add_argument('--plot_every', type=int, default=5000,
                        help='Plot traces every N steps (default: 5000)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ── config ────────────────────────────────────────────────────────────────
    config = NeuralGraphConfig.from_yaml(args.config)
    sim = config.simulation

    if args.output_root:
        set_data_root(args.output_root)

    # Prepend parent folder prefix (e.g. "fly/") to dataset — same logic as GNN_Main.py
    config_path_abs = os.path.abspath(args.config)
    parent = os.path.basename(os.path.dirname(config_path_abs))
    pre_folder = parent + '/' if parent else ''
    if pre_folder and not config.dataset.startswith(pre_folder):
        config.dataset = pre_folder + config.dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    print(f'dataset: {config.dataset}')

    # ── load data ─────────────────────────────────────────────────────────────
    train_path = graphs_data_path(config.dataset, 'x_list_train')
    if not os.path.exists(train_path):
        train_path = graphs_data_path(config.dataset, 'x_list_0')
    print(f'loading data from {train_path}')
    x_ts = load_simulation_data(train_path)

    voltage_np = x_ts.voltage.numpy()          # (T, N)
    n_frames, n_neurons_total = voltage_np.shape
    t_period = float(n_frames) / (2 * np.pi)

    print(f'data: {n_frames} frames, {n_neurons_total} neurons total')

    # ── random neuron selection (non-retina only) ─────────────────────────────
    rng = np.random.default_rng(args.seed)
    n_retina = sim.n_input_neurons
    non_retina = np.arange(n_retina, n_neurons_total)
    n_select = min(args.n_neurons, len(non_retina))
    sel_ids = rng.choice(non_retina, size=n_select, replace=False)
    sel_ids = np.sort(sel_ids)
    print(f'selected {n_select} non-retinal neurons (ids {sel_ids[0]}..{sel_ids[-1]})')

    # voltage subset: (T, n_select)
    voltage_sel = voltage_np[:, sel_ids]

    # SVD rank
    from sklearn.utils.extmath import randomized_svd
    n_comp = min(50, min(voltage_sel.shape) - 1)
    _, S, _ = randomized_svd(voltage_sel, n_components=n_comp, random_state=0)
    cumvar = np.cumsum(S**2) / np.sum(S**2)
    rank_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    rank_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    print(f'effective rank: 90%={rank_90}, 99%={rank_99}')

    # ── output dir ────────────────────────────────────────────────────────────
    from connectome_gnn.utils import create_log_dir
    log_dir, _ = create_log_dir(config, erase=False)
    out_dir = os.path.join(log_dir, 'tmp_training', 'siren_voltage')
    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    print(f'output: {out_dir}')

    # ── SIREN ─────────────────────────────────────────────────────────────────
    siren = Siren(
        in_features=1,
        hidden_features=args.hidden_dim,
        hidden_layers=args.n_layers,
        out_features=n_select,
        outermost_linear=True,
        first_omega_0=args.omega,
        hidden_omega_0=args.omega,
    ).to(device)

    total_params = sum(p.numel() for p in siren.parameters())
    compression = (n_frames * n_select) / total_params
    print(f'SIREN: in=1 hidden={args.hidden_dim}x{args.n_layers} out={n_select}')
    print(f'params: {total_params:,}  compression: {compression:.1f}x  lr={args.lr}')

    optim = torch.optim.Adam(siren.parameters(), lr=args.lr)
    ground_truth = torch.tensor(voltage_sel, dtype=torch.float32, device=device)  # (T, n_select)

    # ── training loop ─────────────────────────────────────────────────────────
    r2, a, b = 0.0, 0.0, 0.0
    eval_interval  = 500
    pbar = trange(args.steps + 1, ncols=110, desc='SIREN')

    for step in pbar:
        frame_ids = np.random.randint(0, n_frames, size=args.batch_size)
        t_in  = torch.tensor(frame_ids / t_period, dtype=torch.float32, device=device).unsqueeze(1)
        gt_batch = ground_truth[frame_ids]          # (bs, n_select)

        pred = siren(t_in)                          # (bs, n_select)
        loss = F.mse_loss(pred, gt_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(siren.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

        # ── R² eval ───────────────────────────────────────────────────────────
        if step % eval_interval == 0 and step > 0:
            sample_ids = np.linspace(0, n_frames - 1, 200, dtype=int)
            with torch.no_grad():
                t_s = torch.tensor(sample_ids / t_period, dtype=torch.float32, device=device).unsqueeze(1)
                pred_s = siren(t_s).cpu().numpy()   # (200, n_select)
            gt_s = voltage_sel[sample_ids]           # (200, n_select)
            a, b, r2 = _linear_fit(gt_s, pred_s)

        c = _col(r2)
        pbar.set_postfix_str(f'loss={loss.item():.4e}  {c}R²={r2:.3f}{_RST}  a={a:.3f} b={b:.3f}')

        # ── trace plot ────────────────────────────────────────────────────────
        if step % args.plot_every == 0 and step > 0:
            sample_ids = np.linspace(0, n_frames - 1, 800, dtype=int)
            with torch.no_grad():
                t_s = torch.tensor(sample_ids / t_period, dtype=torch.float32, device=device).unsqueeze(1)
                pred_plot = siren(t_s).cpu().numpy()   # (800, n_select)
            gt_plot = voltage_sel[sample_ids]
            # transpose to (n_select, 800) for plot
            _plot_traces(gt_plot.T, pred_plot.T, sel_ids, step, r2, a, b,
                         out_path=os.path.join(out_dir, f'traces_{step:06d}.png'))
            # also save as latest
            _plot_traces(gt_plot.T, pred_plot.T, sel_ids, step, r2, a, b,
                         out_path=os.path.join(out_dir, 'traces_latest.png'))

    # ── final eval ────────────────────────────────────────────────────────────
    sample_ids = np.linspace(0, n_frames - 1, 500, dtype=int)
    with torch.no_grad():
        t_s = torch.tensor(sample_ids / t_period, dtype=torch.float32, device=device).unsqueeze(1)
        pred_final = siren(t_s).cpu().numpy()
    gt_final = voltage_sel[sample_ids]
    a, b, r2 = _linear_fit(gt_final, pred_final)

    print(f'\nfinal R²={r2:.4f}  a={a:.4f}  b={b:.4f}')

    # final plot
    _plot_traces(gt_final.T, pred_final.T, sel_ids, args.steps, r2, a, b,
                 out_path=os.path.join(out_dir, 'traces_final.png'))

    # save model
    model_path = os.path.join(out_dir, 'siren_voltage.pt')
    torch.save(siren.state_dict(), model_path)
    print(f'model saved to {model_path}')

    # results log
    with open(os.path.join(out_dir, 'results.log'), 'w') as f:
        f.write(f'n_neurons_selected: {n_select}\n')
        f.write(f'n_frames: {n_frames}\n')
        f.write(f'steps: {args.steps}\n')
        f.write(f'lr: {args.lr}\n')
        f.write(f'hidden_dim: {args.hidden_dim}\n')
        f.write(f'n_layers: {args.n_layers}\n')
        f.write(f'omega: {args.omega}\n')
        f.write(f'total_params: {total_params}\n')
        f.write(f'compression: {compression:.2f}x\n')
        f.write(f'rank_90: {rank_90}\n')
        f.write(f'rank_99: {rank_99}\n')
        f.write(f'final_r2: {r2:.6f}\n')
        f.write(f'a: {a:.6f}\n')
        f.write(f'b: {b:.6f}\n')
    print(f'results written to {out_dir}/results.log')


if __name__ == '__main__':
    main()
