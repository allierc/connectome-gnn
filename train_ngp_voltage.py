"""Standalone InstantNGP voltage trainer.

Trains HashEncodingMLP (InstantNGP-style hash grid + MLP) on ground-truth
voltages of a random neuron subset. Parallel alternative to train_siren_voltage.py.

Hash grid is local in time — no waterbed problem. Can use small batches (96 frames)
with high LR (1e-3). Based on working config from neural-gnn/config/signal/signal_N4_5.yaml.

Architecture: t (normalized [0,1]) -> HashGrid (24 levels x 2 features = 48-dim)
              -> float() -> PyTorch MLP (256 x 4 layers) -> (n_neurons,)
Note: tinycudann requires input in [0,1] and at least 2D — 1D time is padded [t]->[t,t].

Usage:
    python train_ngp_voltage.py config/fly/flyvis_noise_005_hidden_010.yaml
    python train_ngp_voltage.py config/fly/flyvis_noise_005_hidden_010.yaml --n_neurons 1000 --steps 50000 --output_root /groups/saalfeld/home/allierc/GraphData

Cluster:
    bsub -n 2 -gpu "num=1" -q gpu_a100 -W 6000 -Is "python train_ngp_voltage.py config/fly/flyvis_noise_005_hidden_010.yaml --n_neurons 1000 --steps 50000 --lr 1e-3 --batch_size 96 --output_root /groups/saalfeld/home/allierc/GraphData"
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Add sibling repos so HashEncodingMLP is importable
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _repo in ('cell-gnn', 'neural-gnn'):
    _p = os.path.join(_repo_root, _repo, 'src')
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.utils import graphs_data_path, set_data_root
from connectome_gnn.zarr_io import load_simulation_data

try:
    from cell_gnn.models.HashEncoding_Network import HashEncodingMLP
    NGP_AVAILABLE = True
except ImportError:
    try:
        from neural_gnn.models.HashEncoding_Network import HashEncodingMLP
        NGP_AVAILABLE = True
    except ImportError:
        NGP_AVAILABLE = False


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
                label='NGP (corrected)' if i == 0 else None)
        ax.text(-n_frames * 0.025, i * step_v, f'n{sel_ids[ni]}',
                fontsize=8, va='bottom', ha='right')

    ax.set_xlim([-n_frames * 0.03, n_frames * 1.02])
    ax.set_ylim([-step_v, n_show * step_v + step_v])
    ax.set_yticks([])
    ax.set_xlabel('frame', fontsize=13)
    ax.set_title(f'InstantNGP voltage  step {step}   R²={r2:.3f}   a={a:.3f} b={b:.3f}', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Standalone InstantNGP voltage trainer')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--n_neurons', type=int, default=1000,
                        help='Number of neurons to select randomly (default: 1000)')
    parser.add_argument('--steps', type=int, default=50000,
                        help='Training steps (default: 50000)')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='Frames per step (default: 96 — NGP is local, small batches work)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3 — NGP converges fast)')
    # NGP architecture — from neural-gnn/config/signal/signal_N4_5.yaml
    parser.add_argument('--n_levels', type=int, default=24)
    parser.add_argument('--n_features_per_level', type=int, default=2)
    parser.add_argument('--log2_hashmap_size', type=int, default=22)
    parser.add_argument('--base_resolution', type=int, default=16)
    parser.add_argument('--per_level_scale', type=float, default=1.4)
    parser.add_argument('--mlp_width', type=int, default=256)
    parser.add_argument('--mlp_layers', type=int, default=4)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--plot_every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not NGP_AVAILABLE:
        print('ERROR: tinycudann not available. Install from https://github.com/NVlabs/tiny-cuda-nn')
        sys.exit(1)

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
    t_period = float(n_frames)                  # normalize t to [0, 1]

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
    os.makedirs(out_dir, exist_ok=True)
    print(f'output: {out_dir}')

    # ── model ─────────────────────────────────────────────────────────────────
    encoding_dim = args.n_levels * args.n_features_per_level
    model = HashEncodingMLP(
        n_input_dims=1,
        n_output_dims=n_select,
        n_levels=args.n_levels,
        n_features_per_level=args.n_features_per_level,
        log2_hashmap_size=args.log2_hashmap_size,
        base_resolution=args.base_resolution,
        per_level_scale=args.per_level_scale,
        n_neurons=args.mlp_width,
        n_hidden_layers=args.mlp_layers,
        output_activation='none',
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    compression = (n_frames * n_select) / total_params
    print(f'NGP: {args.n_levels}L x {args.n_features_per_level}f = {encoding_dim}-dim encoding')
    print(f'MLP: {encoding_dim} -> {args.mlp_width}x{args.mlp_layers} -> {n_select}')
    print(f'params: {total_params:,}  compression: {compression:.1f}x  lr={args.lr}')

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    ground_truth = torch.tensor(voltage_sel, dtype=torch.float32, device=device)  # (T, n_select)

    # ── training loop ─────────────────────────────────────────────────────────
    r2, a, b = 0.0, 0.0, 0.0
    eval_interval = 500
    pbar = trange(args.steps + 1, ncols=110, desc='NGP')

    for step in pbar:
        frame_ids = np.random.randint(0, n_frames, size=args.batch_size)
        t_in  = torch.tensor(frame_ids / t_period, dtype=torch.float32, device=device).unsqueeze(1)
        gt_batch = ground_truth[frame_ids]          # (bs, n_select)

        pred = model(t_in)                          # (bs, n_select)

        # Relative L2 loss (from neural-gnn working config — more stable than MSE)
        rel_l2 = (pred - gt_batch) ** 2 / (pred.detach() ** 2 + 0.01)
        loss = rel_l2.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()

        # ── R² eval ───────────────────────────────────────────────────────────
        if step % eval_interval == 0 and step > 0:
            sample_ids = np.linspace(0, n_frames - 1, 200, dtype=int)
            with torch.no_grad():
                t_s = torch.tensor(sample_ids / t_period, dtype=torch.float32, device=device).unsqueeze(1)
                pred_s = model(t_s).cpu().numpy()
            gt_s = voltage_sel[sample_ids]
            a, b, r2 = _linear_fit(gt_s, pred_s)

        c = _col(r2)
        pbar.set_postfix_str(f'loss={loss.item():.4e}  {c}R²={r2:.3f}{_RST}  a={a:.3f} b={b:.3f}')

        # ── trace plot ────────────────────────────────────────────────────────
        if step % args.plot_every == 0 and step > 0:
            sample_ids = np.linspace(0, n_frames - 1, 800, dtype=int)
            with torch.no_grad():
                t_s = torch.tensor(sample_ids / t_period, dtype=torch.float32, device=device).unsqueeze(1)
                pred_plot = model(t_s).cpu().numpy()
            gt_plot = voltage_sel[sample_ids]
            _plot_traces(gt_plot.T, pred_plot.T, sel_ids, step, r2, a, b,
                         out_path=os.path.join(out_dir, f'traces_{step:06d}.png'))
            _plot_traces(gt_plot.T, pred_plot.T, sel_ids, step, r2, a, b,
                         out_path=os.path.join(out_dir, 'traces_latest.png'))

    # ── final eval ────────────────────────────────────────────────────────────
    sample_ids = np.linspace(0, n_frames - 1, 500, dtype=int)
    with torch.no_grad():
        t_s = torch.tensor(sample_ids / t_period, dtype=torch.float32, device=device).unsqueeze(1)
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
        f.write(f'log2_hashmap_size: {args.log2_hashmap_size}\n')
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
