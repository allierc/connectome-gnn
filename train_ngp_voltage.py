"""Standalone multi-resolution temporal grid voltage trainer.

Trains a multi-resolution feature grid (InstantNGP-style, pure PyTorch) on
ground-truth voltages of a random neuron subset.  Unlike SIREN, the grid is
local in time — updating one time bucket leaves other frames untouched (no
waterbed problem).  Can use small batches (96 frames) with high LR (1e-3).

Architecture:
    t ∈ [0,1]  →  multi-resolution 1-D grid (24 levels × 2 features = 48-dim,
                   linear interp within each level)
               →  PyTorch MLP (256 × 4)  →  (n_neurons,)

100% pure PyTorch — no tinycudann required.

Usage:
    python train_ngp_voltage.py config/fly/flyvis_noise_005_hidden_010.yaml
    python train_ngp_voltage.py config/fly/flyvis_noise_005_hidden_010.yaml \\
        --n_neurons 1000 --steps 50000

Cluster:
    bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is \\
      "python train_ngp_voltage.py config/fly/flyvis_noise_005_hidden_010.yaml \\
       --n_neurons 1000 --steps 50000 --lr 1e-3 --batch_size 96
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.utils import graphs_data_path, set_data_root
from connectome_gnn.zarr_io import load_simulation_data


# ── multi-resolution 1-D feature grid (pure PyTorch, no tinycudann) ───────────
class MultiResTemporalGrid(nn.Module):
    """Multi-resolution 1-D temporal feature grid + MLP.

    Mirrors InstantNGP locality: each level is a 1-D grid of learnable feature
    vectors.  Forward pass does linear interpolation between the two nearest
    grid points, then concatenates features across all levels and runs an MLP.

    Because each level only reads two neighbouring grid points, updating a time
    sample only changes those two entries — no waterbed problem.

    Args:
        n_levels: number of grid levels (default 24, as in NGP)
        n_features_per_level: features per grid cell (default 2)
        base_resolution: coarsest grid resolution (default 16)
        per_level_scale: resolution multiplier per level (default 1.4)
        n_output: output neurons
        mlp_width: hidden width of MLP after encoding
        mlp_layers: number of hidden MLP layers
    """

    def __init__(
        self,
        n_levels: int = 24,
        n_features_per_level: int = 2,
        base_resolution: int = 16,
        per_level_scale: float = 1.4,
        n_output: int = 1000,
        mlp_width: int = 256,
        mlp_layers: int = 4,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level

        # Build one nn.Embedding per level
        self.grids = nn.ModuleList()
        self.resolutions: list[int] = []
        res = float(base_resolution)
        for _ in range(n_levels):
            r = max(2, int(res))
            # r+1 entries so that index r is valid (upper boundary)
            emb = nn.Embedding(r + 1, n_features_per_level)
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
            self.grids.append(emb)
            self.resolutions.append(r)
            res *= per_level_scale

        n_enc = n_levels * n_features_per_level

        # MLP
        layers: list[nn.Module] = [nn.Linear(n_enc, mlp_width), nn.ReLU()]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]
        layers.append(nn.Linear(mlp_width, n_output))
        self.mlp = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, 1) normalized in [0, 1]"""
        t = t.squeeze(1)   # (B,)
        features = []
        for emb, res in zip(self.grids, self.resolutions):
            pos = t * res                                   # (B,)
            i0  = pos.long().clamp(0, res - 1)             # lower index
            i1  = (i0 + 1).clamp(0, res)                   # upper index
            w1  = (pos - pos.floor()).unsqueeze(1)          # (B, 1)
            w0  = 1.0 - w1
            feat = w0 * emb(i0) + w1 * emb(i1)            # (B, n_feat)
            features.append(feat)
        enc = torch.cat(features, dim=1)                   # (B, n_enc)
        return self.mlp(enc)


# ── ANSI color helpers ─────────────────────────────────────────────────────────
_G, _Y, _O, _R, _RST = '\033[92m', '\033[93m', '\033[38;5;208m', '\033[91m', '\033[0m'

def _col(r2):
    return _G if r2 > 0.7 else _Y if r2 > 0.4 else _O if r2 > 0.1 else _R


# ── linear fit + per-neuron R² ─────────────────────────────────────────────────
def _linear_fit(gt, pred):
    """Fit gt = a*pred + b globally, return (a, b, r2_temporal).

    gt, pred: (T, N)
    r2: mean per-neuron R² after correction — measures temporal dynamics.
    """
    gt_f, pred_f = gt.ravel(), pred.ravel()
    cov = ((pred_f - pred_f.mean()) * (gt_f - gt_f.mean())).mean()
    var = ((pred_f - pred_f.mean()) ** 2).mean()
    a = float(cov / (var + 1e-12))
    b = float(gt_f.mean() - a * pred_f.mean())
    pred_corr = a * pred + b
    gt_mean_n = gt.mean(axis=0)
    ss_res = ((gt - pred_corr) ** 2).sum(axis=0)
    ss_tot = ((gt - gt_mean_n) ** 2).sum(axis=0)
    r2 = float((1.0 - ss_res / (ss_tot + 1e-12)).mean())
    return a, b, r2


# ── trace plot ─────────────────────────────────────────────────────────────────
def _plot_traces(gt_arr, pred_arr, sel_ids, step, r2, a, b, out_path, n_show=10):
    """gt_arr, pred_arr: (n_neurons, T)"""
    n_neurons, n_frames = gt_arr.shape
    n_show = min(n_show, n_neurons)
    idx = np.linspace(0, n_neurons - 1, n_show, dtype=int)
    pred_corr = a * pred_arr + b
    activity_std = float(np.std(gt_arr))
    step_v = max(0.5, 3.0 * activity_std)

    fig, ax = plt.subplots(figsize=(15, max(4, n_show * 0.5 + 2)))
    for i, ni in enumerate(idx):
        gt_i   = gt_arr[ni]   - gt_arr[ni].mean()
        pred_i = pred_corr[ni] - pred_corr[ni].mean()
        ax.plot(gt_i   + i * step_v, lw=2.0, c='#66cc66', alpha=0.9,
                label='GT' if i == 0 else None)
        ax.plot(pred_i + i * step_v, lw=0.9, c='black', alpha=0.9,
                label='Grid+MLP (corrected)' if i == 0 else None)
        ax.text(-n_frames * 0.025, i * step_v, f'n{sel_ids[ni]}',
                fontsize=8, va='bottom', ha='right')

    ax.set_xlim([-n_frames * 0.03, n_frames * 1.02])
    ax.set_ylim([-step_v, n_show * step_v + step_v])
    ax.set_yticks([])
    ax.set_xlabel('frame', fontsize=13)
    ax.set_title(f'MultiResGrid voltage  step {step}   R²={r2:.3f}   a={a:.3f} b={b:.3f}',
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
    parser = argparse.ArgumentParser(
        description='Standalone multi-resolution temporal grid voltage trainer')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--n_neurons', type=int, default=1000,
                        help='Number of neurons to select randomly (default: 1000)')
    parser.add_argument('--steps', type=int, default=50000,
                        help='Training steps (default: 50000)')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='Frames per step (default: 96 — grid is local, small batches ok)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    # Grid architecture — mirrors NGP signal_N4_5.yaml
    parser.add_argument('--n_levels', type=int, default=24)
    parser.add_argument('--n_features_per_level', type=int, default=2)
    parser.add_argument('--base_resolution', type=int, default=16)
    parser.add_argument('--per_level_scale', type=float, default=1.4)
    parser.add_argument('--mlp_width', type=int, default=256)
    parser.add_argument('--mlp_layers', type=int, default=4)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--plot_every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # ── config ────────────────────────────────────────────────────────────────
    config = NeuralGraphConfig.from_yaml(args.config)
    sim = config.simulation

    if args.output_root:
        set_data_root(args.output_root)

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
    t_period = float(n_frames)                  # normalize t -> [0, 1]

    print(f'data: {n_frames} frames, {n_neurons_total} neurons total')

    # ── random neuron selection (non-retina only) ─────────────────────────────
    rng = np.random.default_rng(args.seed)
    n_retina = sim.n_input_neurons
    non_retina = np.arange(n_retina, n_neurons_total)
    n_select = min(args.n_neurons, len(non_retina))
    sel_ids = np.sort(rng.choice(non_retina, size=n_select, replace=False))
    print(f'selected {n_select} non-retinal neurons (ids {sel_ids[0]}..{sel_ids[-1]})')

    voltage_sel = voltage_np[:, sel_ids]        # (T, n_select)

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
    out_dir = os.path.join(log_dir, 'tmp_training', 'ngp_voltage')
    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    print(f'output: {out_dir}')

    # ── model ─────────────────────────────────────────────────────────────────
    model = MultiResTemporalGrid(
        n_levels=args.n_levels,
        n_features_per_level=args.n_features_per_level,
        base_resolution=args.base_resolution,
        per_level_scale=args.per_level_scale,
        n_output=n_select,
        mlp_width=args.mlp_width,
        mlp_layers=args.mlp_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    n_enc = args.n_levels * args.n_features_per_level
    compression = (n_frames * n_select) / total_params
    print(f'MultiResGrid: {args.n_levels}L x {args.n_features_per_level}f = {n_enc}-dim enc')
    print(f'MLP: {n_enc} -> {args.mlp_width}x{args.mlp_layers} -> {n_select}')
    print(f'params: {total_params:,}  compression: {compression:.1f}x  lr={args.lr}')

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    ground_truth = torch.tensor(voltage_sel, dtype=torch.float32, device=device)  # (T, n_select)

    # ── training loop ─────────────────────────────────────────────────────────
    r2, a, b = 0.0, 0.0, 0.0
    eval_interval = 500
    pbar = trange(args.steps + 1, ncols=110, desc='NGP')

    for step in pbar:
        frame_ids = np.random.randint(0, n_frames, size=args.batch_size)
        t_in     = torch.tensor(frame_ids / t_period, dtype=torch.float32,
                                device=device).unsqueeze(1)
        gt_batch = ground_truth[frame_ids]          # (bs, n_select)

        pred = model(t_in)                          # (bs, n_select)
        loss = F.mse_loss(pred, gt_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

        # ── R² eval ───────────────────────────────────────────────────────────
        if step % eval_interval == 0 and step > 0:
            sample_ids = np.linspace(0, n_frames - 1, 200, dtype=int)
            with torch.no_grad():
                t_s = torch.tensor(sample_ids / t_period, dtype=torch.float32,
                                   device=device).unsqueeze(1)
                pred_s = model(t_s).cpu().numpy()
            gt_s = voltage_sel[sample_ids]
            a, b, r2 = _linear_fit(gt_s, pred_s)

        c = _col(r2)
        pbar.set_postfix_str(f'loss={loss.item():.4e}  {c}R²={r2:.3f}{_RST}  a={a:.3f} b={b:.3f}')

        # ── trace plot ────────────────────────────────────────────────────────
        if step % args.plot_every == 0 and step > 0:
            sample_ids = np.linspace(0, n_frames - 1, 800, dtype=int)
            with torch.no_grad():
                t_s = torch.tensor(sample_ids / t_period, dtype=torch.float32,
                                   device=device).unsqueeze(1)
                pred_plot = model(t_s).cpu().numpy()
            gt_plot = voltage_sel[sample_ids]
            _plot_traces(gt_plot.T, pred_plot.T, sel_ids, step, r2, a, b,
                         out_path=os.path.join(out_dir, f'traces_{step:06d}.png'))
            _plot_traces(gt_plot.T, pred_plot.T, sel_ids, step, r2, a, b,
                         out_path=os.path.join(out_dir, 'traces_latest.png'))

    # ── final eval ────────────────────────────────────────────────────────────
    sample_ids = np.linspace(0, n_frames - 1, 500, dtype=int)
    with torch.no_grad():
        t_s = torch.tensor(sample_ids / t_period, dtype=torch.float32,
                           device=device).unsqueeze(1)
        pred_final = model(t_s).cpu().numpy()
    gt_final = voltage_sel[sample_ids]
    a, b, r2 = _linear_fit(gt_final, pred_final)
    print(f'\nfinal R²={r2:.4f}  a={a:.4f}  b={b:.4f}')

    _plot_traces(gt_final.T, pred_final.T, sel_ids, args.steps, r2, a, b,
                 out_path=os.path.join(out_dir, 'traces_final.png'))

    model_path = os.path.join(out_dir, 'ngp_voltage.pt')
    torch.save(model.state_dict(), model_path)
    print(f'model saved to {model_path}')

    with open(os.path.join(out_dir, 'results.log'), 'w') as f:
        f.write(f'n_neurons_selected: {n_select}\n')
        f.write(f'n_frames: {n_frames}\n')
        f.write(f'steps: {args.steps}\n')
        f.write(f'lr: {args.lr}\n')
        f.write(f'batch_size: {args.batch_size}\n')
        f.write(f'n_levels: {args.n_levels}\n')
        f.write(f'n_features_per_level: {args.n_features_per_level}\n')
        f.write(f'base_resolution: {args.base_resolution}\n')
        f.write(f'per_level_scale: {args.per_level_scale}\n')
        f.write(f'mlp_width: {args.mlp_width}\n')
        f.write(f'mlp_layers: {args.mlp_layers}\n')
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
