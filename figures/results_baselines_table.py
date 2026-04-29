"""Produce LaTeX rows for one-step and rollout Pearson $r$ from a CV run.

Usage
-----
    python scripts/tex_table_cv_r.py <config_name>

Takes the config name that was passed to `GNN_Main.py -o cv <config_name>`
(e.g. ``flyvis_noise_005_mlp``), reloads every checkpoint under
``log/fly/<config>_cv{00..04}/``, evaluates one-step and rollout Pearson
correlations on each fold's test split, pools the per-neuron values
across neurons and folds in Fisher-$z$ space, and emits two LaTeX table
fragments:

  * **Symmetric** table  (printed live) — ``$r\\pm s$`` with
    ``s = (rhi - rlo) / 2``.
  * **Asymmetric** table (commented out) — ``$r^{+up}_{-dn}$`` with
    ``up = rhi - rmean`` and ``dn = rmean - rlo``.

The asymmetric version is commented so the default paste is publication-
ready; uncomment to use it instead.

Assumptions
-----------
* Same test data is shared across all CV folds of a given config (the
  `` _cv{XX}`` subdirectories).
* ``noise_free`` / ``noise_005`` / ``noise_05`` in the config name encode
  intrinsic noise ($\\sigma = 0, 0.05, 0.5$).
* ``mlp`` / ``eed`` substrings drive the model family label; ``winner``
  suffix is reflected in the condition name.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
from tqdm import trange

# Allow running from the repo root without installing the package.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(REPO, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(REPO, 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.generators.ode_params import load_edge_index
from connectome_gnn.models.registry import create_model
from connectome_gnn.utils import (
    graphs_data_path, log_path, migrate_state_dict, set_data_root, set_device,
)
from connectome_gnn.zarr_io import load_raw_array, load_simulation_data


# ---------------------------------------------------------------------------
# Config-name parsing
# ---------------------------------------------------------------------------

NOISE_MAP = {
    'noise_free': '0',
    'noise_005':  '0.05',
    'noise_05':   '0.5',
}

# When a stimulus-baseline config is supplied we don't have its own CV test
# data (stimulus runs aren't CV'd). Reuse the MLP CV test data at the matching
# noise level — the underlying sim data is shared.
STIM_DATA_DONOR = {
    '0':    'flyvis_noise_free_mlp',
    '0.05': 'flyvis_noise_005_mlp',
    '0.5':  'flyvis_noise_05_mlp',
}


def infer_metadata(config_name: str) -> dict:
    """Pull σ, family, and display name out of a base config string."""
    sigma = next((v for k, v in NOISE_MAP.items() if k in config_name), None)
    if sigma is None:
        raise ValueError(
            f'could not infer noise σ from {config_name!r}; '
            f'expected one of {list(NOISE_MAP)} in the name')

    if 'stimulus' in config_name:
        family = 'Stimulus'
        m = re.search(r'_ctx(\d+)', config_name)
        ctx = int(m.group(1)) if m else None
        ctx_str = f', $t_w{{=}}{ctx}$' if ctx is not None else ''
        suffix = ', noise-free' if sigma == '0' else f', $\\sigma{{=}}{sigma}$'
        display = f'{family}{ctx_str}{suffix}'
        return dict(sigma=sigma, family=family, display=display, kind='stimulus')

    if '_mlp' in config_name:
        family = 'MLP'
    elif '_eed' in config_name:
        family = 'EED'
    else:
        family = '?'

    winner = ' winner' if '_winner' in config_name else ''
    if sigma == '0':
        display = f'{family}{winner}, noise-free'
    else:
        display = f'{family}{winner}, $\\sigma{{=}}{sigma}$'
    return dict(sigma=sigma, family=family, display=display, kind='mlp_eed')


# ---------------------------------------------------------------------------
# File existence checks
# ---------------------------------------------------------------------------

def preflight(config_name: str, folds: list[int]) -> None:
    """Verify the yaml, model dir(s), and test data are all present before
    running any expensive evaluation.

    For stimulus configs, the model is a single (non-CV) run; its test data
    is borrowed from the matching-noise MLP CV run (STIM_DATA_DONOR).
    """
    yaml_path = os.path.join(REPO, 'config', 'fly', f'{config_name}.yaml')
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f'config yaml not found: {yaml_path}')

    meta = infer_metadata(config_name)
    missing: list[str] = []

    if meta['kind'] == 'stimulus':
        stim_log = log_path('fly/' + config_name)
        if not glob.glob(os.path.join(stim_log, 'models', 'best_model_with_*.pt')):
            missing.append(f'  [stim] no checkpoints under {stim_log}/models/')
        donor = STIM_DATA_DONOR[meta['sigma']]
        for f in folds:
            dn = f'{donor}_cv{f:02d}'
            for split in ('x_list_test', 'x_list_0'):
                if os.path.exists(graphs_data_path('fly/' + dn, split)):
                    break
            else:
                missing.append(
                    f'  [cv{f:02d}] no x_list_test or x_list_0 under graphs_data/fly/{dn}/ '
                    f'(donor for {config_name})')
    else:
        for f in folds:
            run_name = f'{config_name}_cv{f:02d}'
            log_dir  = log_path('fly/' + run_name)
            if not glob.glob(os.path.join(log_dir, 'models', 'best_model_with_*.pt')):
                missing.append(f'  [cv{f:02d}] no checkpoints under {log_dir}/models/')
            for split in ('x_list_test', 'x_list_0'):
                if os.path.exists(graphs_data_path('fly/' + run_name, split)):
                    break
            else:
                missing.append(f'  [cv{f:02d}] no x_list_test or x_list_0 under graphs_data/fly/{run_name}/')

    if missing:
        raise FileNotFoundError(
            'Missing inputs — aborting before evaluation:\n' + '\n'.join(missing))


# ---------------------------------------------------------------------------
# Data + model loading (one fold)
# ---------------------------------------------------------------------------

def _load_test_data(dataset: str, device, max_frames: int, load_fields: list[str]):
    """Load (x_ts, y_ts) for a given `fly/<name>` dataset path."""
    test_path = graphs_data_path(dataset, 'x_list_test')
    if os.path.exists(test_path):
        x_ts = load_simulation_data(test_path, fields=load_fields).to(device)
        y_ts = load_raw_array(graphs_data_path(dataset, 'y_list_test'))
    else:
        x_ts = load_simulation_data(
            graphs_data_path(dataset, 'x_list_0'), fields=load_fields
        ).to(device)
        y_ts = load_raw_array(graphs_data_path(dataset, 'y_list_0'))
    x_ts.neuron_type = None
    x_ts.index = torch.arange(x_ts.n_neurons, dtype=torch.long, device=device)
    if x_ts.n_frames > max_frames:
        x_ts = x_ts.truncate_frames(max_frames)
        y_ts = y_ts[:max_frames]
    return x_ts, y_ts


def load_saved_fold(config_name: str, fold_idx: int) -> dict:
    """Load per-neuron pearson + rmse arrays already saved by data_test_gnn,
    plus the model param count (from the latest checkpoint state_dict).

    Returns dict with keys: r1, rR, rmseR (each per-neuron 1d), n_params, kind.
    Stimulus folds have rollout metrics filled with NaN.
    """
    meta = infer_metadata(config_name)
    is_stim = meta['kind'] == 'stimulus'
    if is_stim:
        log_dir = log_path('fly/' + config_name)
    else:
        log_dir = log_path('fly/' + f'{config_name}_cv{fold_idx:02d}')

    r1   = np.load(os.path.join(log_dir, 'results_test_pearson.npy'))
    if is_stim:
        rR    = np.full_like(r1, np.nan)
        rmseR = np.full_like(r1, np.nan)
    else:
        rR    = np.load(os.path.join(log_dir, 'results_rollout_pearson.npy'))
        rmseR = np.load(os.path.join(log_dir, 'results_rollout_rmse.npy'))

    # Param count from checkpoint state_dict (no model build needed).
    ckpts = glob.glob(os.path.join(log_dir, 'models', 'best_model_with_*.pt'))
    ckpt_path = max(ckpts, key=os.path.getmtime)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    n_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))

    return dict(r1=r1, rR=rR, rmseR=rmseR, n_params=n_params, kind=meta['kind'])


def load_fold(config_name: str, fold_idx: int, device, max_frames: int) -> dict:
    yaml_path = os.path.join(REPO, 'config', 'fly', f'{config_name}.yaml')
    config = NeuralGraphConfig.from_yaml(yaml_path)
    meta = infer_metadata(config_name)
    is_stim = meta['kind'] == 'stimulus'

    # For stimulus: single log dir, data borrowed from the matching-noise
    # MLP CV fold. For MLP/EED: log dir and data are per-fold for this config.
    if is_stim:
        log_dir      = log_path('fly/' + config_name)
        data_dataset = 'fly/' + f'{STIM_DATA_DONOR[meta["sigma"]]}_cv{fold_idx:02d}'
    else:
        run_name     = f'{config_name}_cv{fold_idx:02d}'
        log_dir      = log_path('fly/' + run_name)
        data_dataset = 'fly/' + run_name
    config.dataset     = data_dataset
    config.config_file = 'fly/' + (config_name if is_stim else f'{config_name}_cv{fold_idx:02d}')

    sim = config.simulation
    mc  = config.graph_model

    load_fields = ['voltage', 'stimulus', 'neuron_type']
    if sim.calcium_type != 'none':
        load_fields.append('calcium')
    x_ts, y_ts = _load_test_data(data_dataset, device, max_frames, load_fields)
    config.simulation.n_neurons = x_ts.n_neurons

    # Edges: only for MLP/EED; stimulus model ignores them.
    edges = None
    if not is_stim:
        training_edges_path = os.path.join(log_dir, 'training_edges.pt')
        if os.path.exists(training_edges_path):
            edges = torch.load(training_edges_path, map_location=device, weights_only=False)
        else:
            edges = load_edge_index(graphs_data_path(data_dataset), device=device)
        if edges.shape[1] != sim.n_edges:
            config.simulation.n_edges = edges.shape[1]
            config.simulation.n_extra_null_edges = 0

    model = create_model(
        mc.signal_model_name, aggr_type=mc.aggr_type, config=config, device=device,
    ).to(device)
    ckpts = glob.glob(os.path.join(log_dir, 'models', 'best_model_with_*.pt'))
    ckpt_path = max(ckpts, key=os.path.getmtime)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    migrate_state_dict(state_dict)
    model.load_state_dict(state_dict['model_state_dict'], strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(config=config, model=model, x_ts=x_ts, y_ts=y_ts, edges=edges,
                log_dir=log_dir, ckpt_path=ckpt_path, kind=meta['kind'],
                n_params=n_params)


# ---------------------------------------------------------------------------
# Eval loops (vectorised MLP/EED path — same as notebook)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_one_step(model, x_ts, y_ts, device, chunk: int = 1024):
    n_input = model.n_input_neurons
    T = x_ts.n_frames - 1
    voltage = x_ts.voltage[:T].contiguous()
    stim    = x_ts.stimulus[:T, :n_input].contiguous()
    y_all   = torch.as_tensor(np.asarray(y_ts[:T]).squeeze(-1), device=device)

    pred_all = torch.empty_like(voltage)
    for s in trange(0, T, chunk, ncols=100, desc='one-step'):
        e = min(s + chunk, T)
        pred_all[s:e] = model.predict_dvdt(voltage[s:e], stim[s:e])

    valid = ~(torch.isnan(voltage).any(1) | torch.isnan(y_all).any(1))
    return y_all[valid].T.contiguous(), pred_all[valid].T.contiguous()


@torch.no_grad()
def run_one_step_stimulus(model, x_ts, device, chunk: int = 256):
    """Stimulus baseline: predict voltage from a tw-frame stimulus context.
    No rollout (non-recurrent)."""
    tw = model.tw
    n_input = model.n_input_neurons
    stim = x_ts.stimulus[:, :n_input].contiguous()
    windows = stim.unfold(0, tw, 1).permute(0, 2, 1).contiguous()
    true_v  = x_ts.voltage[tw - 1 : x_ts.n_frames].contiguous()

    pred_v = torch.empty_like(true_v)
    for s in trange(0, windows.shape[0], chunk, ncols=100, desc='one-step stim'):
        e = min(s + chunk, windows.shape[0])
        pred_v[s:e] = model.predict_voltage(windows[s:e])

    valid = ~(torch.isnan(true_v).any(1) | torch.isnan(pred_v).any(1))
    return true_v[valid].T.contiguous(), pred_v[valid].T.contiguous()


@torch.no_grad()
def run_rollout(config, model, x_ts, device):
    sim = config.simulation
    n_input = model.n_input_neurons
    n_neurons = x_ts.n_neurons
    T = x_ts.n_frames - 1
    stim_all    = x_ts.stimulus[:T, :n_input].contiguous()
    voltage_all = x_ts.voltage[:T].contiguous()

    pred_all = torch.empty((T, n_neurons), device=device, dtype=voltage_all.dtype)
    v = x_ts.voltage[0].clone()
    diverged_at = None
    for k in trange(T, ncols=100, desc='rollout'):
        pred_all[k] = v
        y = model.predict_dvdt(v.unsqueeze(0), stim_all[k].unsqueeze(0)).squeeze(0)
        v = v + sim.delta_t * y
        if torch.isnan(v).any() or torch.isinf(v).any():
            diverged_at = k
            break
        v.clamp_(-100.0, 100.0)

    if diverged_at is not None:
        print(f'  rollout diverged at frame {diverged_at}', file=sys.stderr)
        pred_all = pred_all[:diverged_at]
        voltage_all = voltage_all[:diverged_at]

    return voltage_all.T.contiguous(), pred_all.T.contiguous()


def torch_trace_metrics(true, pred):
    """Per-row Pearson r and RMSE on-device. Returns (rmse, pearson) as
    1-D numpy arrays of length n_neurons."""
    t = true.to(torch.float64)
    p = pred.to(torch.float64)
    mask = torch.isnan(t) | torch.isnan(p)
    n = (~mask).sum(dim=1)
    denom = n.clamp(min=1).to(t.dtype)
    t0 = torch.where(mask, torch.zeros_like(t), t)
    p0 = torch.where(mask, torch.zeros_like(p), p)
    tm = t0.sum(dim=1) / denom
    pm = p0.sum(dim=1) / denom
    tc = torch.where(mask, torch.zeros_like(t), t - tm[:, None])
    pc = torch.where(mask, torch.zeros_like(p), p - pm[:, None])
    num = (tc * pc).sum(dim=1)
    den = torch.sqrt((tc * tc).sum(dim=1) * (pc * pc).sum(dim=1))
    pearson = torch.where(den > 1e-12, num / den, torch.full_like(num, float('nan')))
    pearson = torch.where(n > 1, pearson, torch.full_like(pearson, float('nan')))

    sq = torch.where(mask, torch.zeros_like(t), (t - p) ** 2)
    rmse = torch.sqrt(sq.sum(dim=1) / denom)
    rmse = torch.where(n > 0, rmse, torch.full_like(rmse, float('nan')))
    return rmse.cpu().numpy(), pearson.cpu().numpy()


# ---------------------------------------------------------------------------
# Pooling + formatting
# ---------------------------------------------------------------------------

def fisher_summary(r_stack: np.ndarray, clip: float = 0.9999):
    """Pool an (n_neurons, n_folds) r-matrix in Fisher-z space.
    Returns (rmean, rlo, rhi) back-transformed to r-space."""
    if r_stack.size == 0:
        return (np.nan, np.nan, np.nan)
    z = np.arctanh(np.clip(r_stack, -clip, clip))
    z = z[~np.isnan(z).any(axis=1), :]
    if z.size == 0:
        return (np.nan, np.nan, np.nan)
    zmean = z.mean()
    zstd  = z.std(ddof=0)
    return np.tanh(zmean), np.tanh(zmean - zstd), np.tanh(zmean + zstd)


def mean_sd_summary(arr_stack: np.ndarray):
    """Pool an (n_neurons, n_folds) positive-scalar matrix (e.g. RMSE) across
    all valid cells. Returns (mean, lo, hi) = (mean, mean-sd, mean+sd)."""
    if arr_stack.size == 0:
        return (np.nan, np.nan, np.nan)
    flat = arr_stack[~np.isnan(arr_stack)]
    if flat.size == 0:
        return (np.nan, np.nan, np.nan)
    m  = float(flat.mean())
    sd = float(flat.std(ddof=0))
    return m, m - sd, m + sd


def _fmt_sym(r, lo, hi, digits=2, good=0.9):
    if np.isnan(r):
        return '---'
    sd = (hi - lo) / 2
    body = f'${r:.{digits}f}{{\\pm}}{sd:.{digits}f}$'
    return f'\\good{{{body}}}' if r > good else body


def _fmt_asym(r, lo, hi, digits=2, good=0.9):
    if np.isnan(r):
        return '---'
    up, dn = hi - r, r - lo
    body = f'${r:.{digits}f}^{{+{up:.{digits}f}}}_{{-{dn:.{digits}f}}}$'
    return f'\\good{{{body}}}' if r > good else body


def _fmt_rmse_sym(m, lo, hi, digits=3):
    """RMSE: plain mean±sd, no colour threshold."""
    if np.isnan(m):
        return '---'
    sd = (hi - lo) / 2
    return f'${m:.{digits}f}{{\\pm}}{sd:.{digits}f}$'


def _fmt_rmse_asym(m, lo, hi, digits=3):
    if np.isnan(m):
        return '---'
    up, dn = hi - m, m - lo
    return f'${m:.{digits}f}^{{+{up:.{digits}f}}}_{{-{dn:.{digits}f}}}$'


def emit_table(label, caption, rows, fmt_r, fmt_rmse, *,
               commented: bool = False, out=sys.stdout):
    """Print a LaTeX table; if `commented`, prefix every line with '% '.

    Each row: (display, sigma, (r1,lo1,hi1), (rR,loR,hiR), (mR,loR_rmse,hiR_rmse))
    """
    def p(line: str):
        print(('% ' + line) if commented else line, file=out)

    p(r'\begin{table}[h!]')
    p(r'\centering')
    p(f'\\caption{{{caption}}}')
    p(f'\\label{{tab:{label}}}')
    p(r'\tiny')
    p(r'\setlength{\tabcolsep}{4pt}')
    p(r'\begin{tabular}{lcrrrrr}')
    p(r'\toprule')
    p(r'condition & noise $\sigma$ & params (M) & rollout steps & one-step $r$ & rollout $r$ & rollout RMSE \\')
    p(r'\midrule')
    prev_family = None
    for display, sigma, n_params, n_rollout, r1t, rRt, rmseRt in rows:
        family = display.split(',', 1)[0]   # "Stimulus" / "MLP" / "EED"
        if prev_family is not None and family != prev_family:
            p(r'\midrule')
        prev_family = family
        rollout_str = '---' if n_rollout is None else f'{n_rollout}'
        p(f'{display:<34} & ${sigma}$  '
          f'& {n_params/1e6:.2f} '
          f'& {rollout_str} '
          f'& {fmt_r(*r1t)} & {fmt_r(*rRt)} & {fmt_rmse(*rmseRt)} \\\\')
    p(r'\bottomrule')
    p(r'\end{tabular}')
    p(r'\end{table}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CV_FOLDS   = [0, 1, 2, 3, 4]         # matches `GNN_Main.py -o cv` default
MAX_FRAMES = 8000                    # matches `data_test_gnn` cap

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('data_root',
                        help='repo / data root containing log/ and graphs_data/ '
                             '(e.g. /groups/saalfeld/home/kumarv4/repos/connectome-gnn)')
    parser.add_argument('config_names', nargs='+',
                        help='one or more base configs (e.g. flyvis_noise_005_mlp '
                             'flyvis_noise_05_eed); one row per config appears in '
                             'the table, in the order given.')
    args = parser.parse_args(argv)

    set_data_root(args.data_root)

    # Provenance: re-emit the exact invocation as a LaTeX comment so the
    # checked-in .tex records how it was generated.
    import shlex
    print('% Generated by:')
    print('%   python ' + ' '.join(shlex.quote(a) for a in sys.argv))

    # Build one row per config from per-fold per-neuron arrays already
    # saved by data_test_gnn (results_{test,rollout}_{pearson,rmse}.npy).
    rows = []
    for cfg in args.config_names:
        meta = infer_metadata(cfg)
        print(f'\n# ===== {meta["display"]}  (σ={meta["sigma"]}, '
              f'config={cfg}) =====', file=sys.stderr)

        # Pull rollout_train_steps straight from the yaml (per-config, fold-independent).
        # Not meaningful for the non-recurrent stimulus baseline.
        if meta['kind'] == 'stimulus':
            n_rollout = None
        else:
            cfg_yaml = NeuralGraphConfig.from_yaml(
                os.path.join(REPO, 'config', 'fly', f'{cfg}.yaml'))
            n_rollout = int(cfg_yaml.training.rollout_train_steps)

        per_fold_r_1s, per_fold_r_ro, per_fold_rmse_ro = [], [], []
        n_params = None
        for f in CV_FOLDS:
            bundle = load_saved_fold(cfg, f)
            if n_params is None:
                n_params = bundle['n_params']
                print(f'#   n_params={n_params:,}', file=sys.stderr)
            per_fold_r_1s.append(bundle['r1'])
            per_fold_r_ro.append(bundle['rR'])
            per_fold_rmse_ro.append(bundle['rmseR'])
            print(f'#   cv{f:02d}: one-step r={np.nanmean(bundle["r1"]):.4f}  '
                  f'rollout r={np.nanmean(bundle["rR"]):.4f}  '
                  f'rollout RMSE={np.nanmean(bundle["rmseR"]):.4f}',
                  file=sys.stderr)

        r_onestep = np.stack(per_fold_r_1s, axis=1)
        r_rollout = np.stack(per_fold_r_ro, axis=1)
        rmse_rollout = np.stack(per_fold_rmse_ro, axis=1)
        rows.append((
            meta['display'], meta['sigma'], n_params, n_rollout,
            fisher_summary(r_onestep),
            fisher_summary(r_rollout),
            mean_sd_summary(rmse_rollout),
        ))

    # ── Symmetric table (default; printed as-is) ────────────────────────
    emit_table(
        'cv_baselines_sym',
        r'Five-fold CV Pearson $r$ and rollout RMSE. Pearson $r$ values are '
        r'Fisher-$z$ mean $\pm$ half-range of the 1-SD interval, '
        r'back-transformed to $r$-space. RMSE is plain mean $\pm$ 1~SD. Both '
        r'are pooled across neurons and folds. '
        r'\textcolor{green!50!black}{Green}: value $>0.9$.',
        rows, _fmt_sym, _fmt_rmse_sym, commented=False,
    )

    # ── Asymmetric table (commented out by default) ─────────────────────
    print()
    print('% ---------------------------------------------------------------')
    print('% Asymmetric super/subscript version of the table above. Each r')
    print('% cell reads $r^{+up}_{-dn}$, where (up, dn) are the back-')
    print('% transformed Fisher-z ±1 SD bounds — up = tanh(z+sd) - r,')
    print('% dn = r - tanh(z-sd). RMSE uses the same notation but without')
    print('% any transform (bounds are literal ±1 SD).')
    print('% ---------------------------------------------------------------')
    emit_table(
        'cv_baselines_asym',
        r'Five-fold CV Pearson $r$ and rollout RMSE. Pearson $r$ reported as '
        r'Fisher-$z$ mean with asymmetric $\pm1$~SD bounds (back-transformed); '
        r'RMSE as mean with $\pm1$~SD. Both pooled across neurons and folds. '
        r'\textcolor{green!50!black}{Green}: value $>0.9$.',
        rows, _fmt_asym, _fmt_rmse_asym, commented=True,
    )

    # ── Raw values for caption / sanity-checks, as LaTeX comments ───────
    print('%')
    print('% raw values (mean / lo / hi):')
    for display, _sigma, n_params, n_rollout, (r1, lo1, hi1), (rR, loR, hiR), (rmR, rmLo, rmHi) in rows:
        rollout_str = '---' if n_rollout is None else f'{n_rollout}'
        print(f'%   {display:<34}  params: {n_params/1e6:.2f}M   '
              f'rollout steps: {rollout_str}   '
              f'1s r: {r1:.4f} [{lo1:.4f}, {hi1:.4f}]   '
              f'ro r: {rR:.4f} [{loR:.4f}, {hiR:.4f}]   '
              f'ro RMSE: {rmR:.4f} [{rmLo:.4f}, {rmHi:.4f}]')


if __name__ == '__main__':
    main()
