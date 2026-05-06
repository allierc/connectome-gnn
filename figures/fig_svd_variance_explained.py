"""SVD variance explained vs K, stratified by coarse cell-type group.

Motivates the Encoder-Evolver-Decoder bottleneck: the (T x N) neural
activity matrix is well approximated by its top-K singular components
for K << N. Per-group R^2(K) is defined as

    R^2_g(K) = 1 - MSE_g(K) / mean_var_g

with
    var_n          = Var_t( A[:, n] )                       (per neuron)
    mean_var_g     = mean_{n in g} var_n
    MSE_g(K)       = mean_{n in g} mean_t (A[:,n] - A_K[:,n])^2
    A_K            = U_K diag(s_K) V_K^T  (best rank-K SVD reconstruction)

Group definitions follow figures/fig_stimulus_ctx_pearson.py
(name_to_group + GROUP_NAMES).

Output: figures/fig_svd_variance_explained.{pdf,png}
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.metrics import INDEX_TO_NAME
from connectome_gnn.models.training_utils import load_flyvis_data
from connectome_gnn.utils import set_data_root


DATA_ROOT  = os.environ.get('TRAINED_MODEL_OUTPUT_ROOT', '.')
CONFIG_EED = 'flyvis_noise_free_eed_blank50_cv00'
K_MAX      = 1024

CACHE_DIR = REPO / 'figures' / '_baseline_cache'
OUT_BASE  = REPO / 'figures' / 'fig_svd_variance_explained'


matplotlib.rc_file(str(REPO / 'figures' / 'janne.matplotlibrc'))
plt.rcParams.update({
    'figure.dpi':      150,
    'savefig.dpi':     300,
    'axes.titlesize':  12,
    'axes.labelsize':  12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
})


GROUP_NAMES = {
    0: 'R1-R8', 1: 'L1-L5', 2: 'Lawf', 3: 'Am',
    4: 'C2-C3', 5: 'CT1', 6: 'Mi', 7: 'T', 8: 'Tm',
}


def name_to_group(name: str) -> int:
    if name.startswith('R') and len(name) >= 2 and name[1].isdigit():
        return 0
    if name.startswith('L') and len(name) >= 2 and name[1].isdigit():
        return 1
    if name.startswith('Lawf'):
        return 2
    if name == 'Am':
        return 3
    if name in ('C2', 'C3'):
        return 4
    if name.startswith('CT1'):
        return 5
    if name.startswith('Mi'):
        return 6
    if name.startswith('Tm'):  # check Tm before T
        return 8
    if name.startswith('T'):
        return 7
    return -1


def _cached_svd(dataset_name, device):
    """Top-K truncated SVD plus per-column statistics for the activity matrix.

    Returns
    -------
    s          : (q,)       singular values (descending)
    V          : (N, q)     right singular vectors
    col_norm_sq: (N,)       per-neuron sum_t A[t,n]^2
    col_mean   : (N,)       per-neuron mean over time
    T, N       : ints
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f'svd_lowrank_{dataset_name.replace("/", "_")}.npz'
    if path.exists():
        z = np.load(path)
        return (z['s'], z['V'], z['col_norm_sq'], z['col_mean'],
                int(z['T']), int(z['N']))
    print(f'  loading voltage for {dataset_name} ...')
    x_ts, _, _ = load_flyvis_data(
        dataset_name=dataset_name, split='train',
        fields=['voltage', 'neuron_type'],
    )
    A = x_ts.voltage.to(device).float()  # (T, N)
    T, N = A.shape
    print(f'  voltage matrix: T={T}, N={N}; '
          f'computing top-{K_MAX} truncated SVD on {device} ...')
    A64 = A.double()
    col_norm_sq = (A64 ** 2).sum(dim=0).cpu().numpy()
    col_mean = A64.mean(dim=0).cpu().numpy()
    q = min(K_MAX, T, N)
    _, s, V = torch.svd_lowrank(A, q=q, niter=4)
    s = s.detach().cpu().numpy().astype(np.float64)
    V = V.detach().cpu().numpy().astype(np.float64)  # (N, q)
    np.savez_compressed(
        path, s=s, V=V, col_norm_sq=col_norm_sq, col_mean=col_mean,
        T=np.int64(T), N=np.int64(N),
    )
    return s, V, col_norm_sq, col_mean, T, N


def _neuron_groups(dataset_name):
    """Return (N,) array of group ids (>=0) and the type ints (for QC)."""
    x_ts, _, type_list = load_flyvis_data(
        dataset_name=dataset_name, split='train',
        fields=['voltage', 'neuron_type'],
    )
    types = type_list.long().squeeze(-1).cpu().numpy()
    groups = np.array(
        [name_to_group(INDEX_TO_NAME.get(int(t), '')) for t in types]
    )
    return groups, types


def main():
    set_data_root(DATA_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    cfg = NeuralGraphConfig.from_yaml(
        str(Path(DATA_ROOT) / 'log' / 'fly' / CONFIG_EED / 'config.yaml')
    )
    dataset_name = f'fly/{cfg.dataset}'

    s, V, col_norm_sq, col_mean, T, N = _cached_svd(dataset_name, device)
    groups, _ = _neuron_groups(dataset_name)
    assert groups.shape[0] == N, (groups.shape, N)

    # Per-neuron variance and per-neuron MSE(K).
    # ||A[:,n] - A_K[:,n]||^2 = ||A[:,n]||^2 - sum_{i<=K} s_i^2 V[n,i]^2
    var_n = col_norm_sq / T - col_mean ** 2  # (N,)

    s2 = s ** 2                                          # (q,)
    contrib = s2[None, :] * (V ** 2)                     # (N, q)
    cum_contrib = np.cumsum(contrib, axis=1)             # (N, q)

    Ks = 2 ** np.arange(1, int(np.log2(K_MAX)) + 1)
    Ks = Ks[Ks <= s.shape[0]]
    # MSE per neuron at each K
    sse_n_K = col_norm_sq[:, None] - cum_contrib[:, Ks - 1]  # (N, len(Ks))
    mse_n_K = sse_n_K / T

    present_groups = sorted({int(g) for g in groups if g >= 0})
    group_color = {g: plt.get_cmap('tab10')(i % 10)
                   for i, g in enumerate(present_groups)}

    print('group, n_neurons, mean_var, R^2(K=1..1024):')
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for g in present_groups:
        mask = groups == g
        n_g = int(mask.sum())
        mean_var_g = float(var_n[mask].mean())
        if mean_var_g <= 0:
            continue
        mse_g = mse_n_K[mask].mean(axis=0)         # (len(Ks),)
        r2_g = 1.0 - mse_g / mean_var_g
        ax.plot(Ks, r2_g, marker='o', ms=4, lw=1.3,
                color=group_color[g], label=GROUP_NAMES.get(g, f'g{g}'))
        rs = '  '.join(f'{r:.3f}' for r in r2_g)
        print(f'  {GROUP_NAMES.get(g, str(g)):>6s}  n={n_g:5d}  '
              f'var={mean_var_g:.3e}  R2=[{rs}]')

    ax.set_xscale('log', base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks], rotation=45)
    ax.set_xlabel('K (number of singular components)')
    ax.set_ylabel(r'$R^2$ (NSE)')
    ax.set_ylim(0.0, 1.0)
    ax.axhline(1.0, color='gray', lw=0.5, ls='--')
    ax.set_title(f'SVD reconstruction of voltage matrix per group\n'
                 f'(T={T}, N={N})', pad=4)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower right', ncol=2)

    out_png = OUT_BASE.with_suffix('.png')
    out_pdf = OUT_BASE.with_suffix('.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'wrote {out_png.name}, {out_pdf.name}')


if __name__ == '__main__':
    main()
