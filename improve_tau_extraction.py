"""Standalone analysis of improved tau extraction methods.

Evaluates several strategies for extracting per-neuron time constants
from a trained flyvis-gnn model, and compares their R^2 against ground
truth. Target run:
    /groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_free_blank50_unified_cv00

Methods (applied at the final checkpoint of that run):
    baseline_mu2sigma     : chord slope over [mu - 2 sigma, mu + 2 sigma]   (the current one)
    quantiles_5_95        : chord slope over [p5, p95] of v_i(t)
    quantiles_floored     : same as quantiles_5_95, but window width >= floor
    density_weighted      : linear fit over full observed range, weighted by empirical histogram
    local_deriv_mean      : tau_i = -1 / (d f_theta / d v) at v = mu_i (autograd)
    local_deriv_median    : tau_i = -1 / (d f_theta / d v) at v = median(v_i(t)) (autograd)
    local_deriv_density   : density-weighted average of -1 / (d f_theta / d v) over v_i samples

Usage (neural-graph-linux conda env):
    conda run -n neural-graph-linux python improve_tau_extraction.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from connectome_gnn.config import NeuralGraphConfig
from connectome_gnn.zarr_io import load_simulation_data
from connectome_gnn.models.registry import create_model
from connectome_gnn.utils import (
    to_numpy, graphs_data_path, migrate_state_dict,
    set_data_root, load_data_root_from_json,
)
from connectome_gnn.metrics import (
    compute_activity_stats,
    compute_r_squared,
    _build_f_theta_features,
)
from connectome_gnn.generators.ode_params import get_ode_params_class, FlyVisODEParams


DEFAULT_LOG_DIR = '/groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_free_blank50_unified_cv00'

def _paths_for(log_dir):
    run_name = os.path.basename(log_dir.rstrip('/'))
    config_yaml = f'/groups/saalfeld/home/allierc/GraphData/config/fly/{run_name}.yaml'
    out_dir = os.path.join(log_dir, 'tau_extraction_analysis')
    return config_yaml, out_dir

N_PTS = 1000            # grid size for chord / density methods
MIN_WINDOW = 0.1        # voltage-unit floor on window width for quiet neurons
CHUNK = 2000            # chunk size for batched MLP eval


# ------------------------------------------------------------------ #
# Setup
# ------------------------------------------------------------------ #

def load_everything(device, log_dir, config_yaml):
    set_data_root(load_data_root_from_json())

    config = NeuralGraphConfig.from_yaml(config_yaml)
    if not config.dataset.startswith('fly/'):
        config.dataset = 'fly/' + config.dataset

    x_path = graphs_data_path(config.dataset, 'x_list_train')
    x_ts = load_simulation_data(
        x_path, fields=['index', 'voltage', 'neuron_type', 'group_type'])

    mu, sigma = compute_activity_stats(x_ts, device)
    voltage = x_ts.voltage.to(device)  # (T, N)
    n_neurons = x_ts.n_neurons

    # ODE params -> gt tau
    OdeParamsCls = get_ode_params_class(config.graph_model.signal_model_name)
    ode_params = OdeParamsCls.load(graphs_data_path(config.dataset), device='cpu')
    gt_tau = ode_params.gt_tau(n_neurons)

    # Model
    ckpt_path = os.path.join(log_dir, 'models', 'best_model_with_0_graphs_0.pt')
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    migrate_state_dict(state)
    if 'W' in state.get('model_state_dict', {}):
        config.simulation.n_edges = state['model_state_dict']['W'].shape[0]
    model = create_model(
        config.graph_model.signal_model_name,
        aggr_type=config.graph_model.aggr_type, config=config, device=device)
    model.load_state_dict(state['model_state_dict'], strict=False)
    model.eval()

    return config, model, voltage, mu, sigma, gt_tau, n_neurons, ode_params


# ------------------------------------------------------------------ #
# f_theta evaluation
# ------------------------------------------------------------------ #

def eval_f_theta_grid(model, rr, device, chunk=CHUNK):
    """Evaluate model.f_theta at (N, n_pts) voltage grid. Returns (N, n_pts)."""
    N, n_pts = rr.shape
    a = model.a[:N]
    emb_dim = a.shape[1]
    out = torch.empty(N, n_pts, device=device)
    for i in range(0, N, chunk):
        chunk_rr = rr[i:i + chunk]
        chunk_a = a[i:i + chunk]
        C = chunk_rr.shape[0]
        rr_flat = chunk_rr.reshape(-1, 1)
        emb_flat = chunk_a[:, None, :].expand(-1, n_pts, -1).reshape(-1, emb_dim)
        feats = _build_f_theta_features(rr_flat, emb_flat)
        with torch.no_grad():
            y = model.f_theta(feats.float()).squeeze(-1).reshape(C, n_pts)
        out[i:i + C] = y
    return out


def f_theta_at_points(model, v_points, device, chunk=CHUNK, create_graph=False):
    """Evaluate f_theta at a single v per neuron. v_points shape (N,).
    Returns (N,) output and enables gradient wrt v if requires_grad=True."""
    N = v_points.shape[0]
    a = model.a[:N]
    emb_dim = a.shape[1]
    outs = []
    for i in range(0, N, chunk):
        v_chunk = v_points[i:i + chunk].reshape(-1, 1)
        emb_chunk = a[i:i + chunk]
        feats = _build_f_theta_features(v_chunk, emb_chunk)
        y = model.f_theta(feats.float()).squeeze(-1)
        outs.append(y)
    return torch.cat(outs, dim=0)


def local_derivative(model, v_points, device, chunk=CHUNK):
    """d f_theta / d v at each v_points[i]. Uses autograd."""
    v = v_points.clone().detach().to(device).requires_grad_(True)
    N = v.shape[0]
    a = model.a[:N]
    emb_dim = a.shape[1]

    grads = torch.empty(N, device=device)
    for i in range(0, N, chunk):
        v_chunk = v[i:i + chunk].reshape(-1, 1)
        emb_chunk = a[i:i + chunk]
        feats = _build_f_theta_features(v_chunk, emb_chunk)
        y = model.f_theta(feats.float()).squeeze(-1)
        g, = torch.autograd.grad(y.sum(), v, retain_graph=False, create_graph=False)
        grads[i:i + chunk] = g[i:i + chunk]
    return to_numpy(grads)


# ------------------------------------------------------------------ #
# Tau extraction strategies
# ------------------------------------------------------------------ #

def vectorized_linear_fit(x, y, weights=None):
    """Weighted least squares slope/offset per row."""
    if weights is None:
        w = np.ones_like(x)
    else:
        w = weights
    sw  = w.sum(axis=1)
    swx = (w * x).sum(axis=1)
    swy = (w * y).sum(axis=1)
    swxy = (w * x * y).sum(axis=1)
    swxx = (w * x * x).sum(axis=1)
    denom = sw * swxx - swx * swx
    safe = np.abs(denom) > 1e-12
    slope = np.where(safe, (sw * swxy - swx * swy) / np.where(safe, denom, 1.0), 0.0)
    offset = np.where(safe, (swy - slope * swx) / np.where(sw > 0, sw, 1.0), 0.0)
    return slope, offset


def slopes_chord(model, starts, ends, device, weights=None):
    """Chord slopes via linspace(start, end) grid."""
    t = torch.linspace(0, 1, N_PTS, device=device)
    starts_t = torch.as_tensor(starts, dtype=torch.float32, device=device)
    ends_t = torch.as_tensor(ends, dtype=torch.float32, device=device)
    rr = starts_t[:, None] + t[None, :] * (ends_t - starts_t)[:, None]  # (N, n_pts)
    func = eval_f_theta_grid(model, rr, device)
    rr_np = to_numpy(rr)
    func_np = to_numpy(func)
    slope, offset = vectorized_linear_fit(rr_np, func_np, weights)
    return slope, offset


def per_neuron_histogram(voltage_np, starts, ends, n_bins=N_PTS):
    """Empirical density of v_i per neuron sampled on the linspace grid.

    Returns weights shape (N, n_bins) — normalized so rows sum to n_bins (unit mean)
    so the weighted-LS denominator is numerically comparable to unweighted.
    """
    T, N = voltage_np.shape
    weights = np.zeros((N, n_bins), dtype=np.float64)
    for i in range(N):
        lo, hi = starts[i], ends[i]
        if hi - lo < 1e-8:
            weights[i] = 1.0
            continue
        h, _ = np.histogram(voltage_np[:, i], bins=n_bins, range=(lo, hi))
        # soft floor to avoid zero-weight bins dominating denom sign
        h = h.astype(np.float64) + 1e-3
        h *= n_bins / h.sum()
        weights[i] = h
    return weights


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def derive_tau_from_slope(slopes, clip=(0, 10)):
    """Matches ODEParamsBase.derive_tau (clip [0,10])."""
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = np.where(np.abs(slopes) > 1e-8, 1.0 / -slopes, 1.0)
    return np.clip(tau, *clip)


def _read_prior_baselines(log_dir):
    """Pull previously reported tau R^2 from the two existing sources:
      - tmp_training/metrics.log : printed live during training (last iter)
      - results/metrics.txt      : written by GNN_PlotFigure post-hoc
    Returns (train_log_tau_r2, plotfig_tau_r2), each Optional[float]."""
    train_val = None
    plot_val = None
    mlog = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if os.path.exists(mlog):
        with open(mlog) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) >= 2:
            header = lines[0].split(',')
            last = lines[-1].split(',')
            if 'tau_r2' in header:
                idx = header.index('tau_r2')
                try:
                    train_val = float(last[idx])
                except (ValueError, IndexError):
                    pass
    mtxt = os.path.join(log_dir, 'results', 'metrics.txt')
    if os.path.exists(mtxt):
        with open(mtxt) as f:
            for line in f:
                if line.startswith('tau_R2'):
                    try:
                        plot_val = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                    break
    return train_val, plot_val


def analyze(log_dir, device):
    config_yaml, out_dir = _paths_for(log_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {os.path.basename(log_dir)} ===")

    train_log_tau, plotfig_tau = _read_prior_baselines(log_dir)
    print(f"[prior] tmp_training/metrics.log tau_r2 = {train_log_tau}")
    print(f"[prior] results/metrics.txt       tau_R2 = {plotfig_tau}")

    print("[load] config, voltage, model ...")
    config, model, voltage, mu, sigma, gt_tau, n_neurons, ode_params = load_everything(
        device, log_dir, config_yaml)
    print(f"[load] n_neurons={n_neurons}  T={voltage.shape[0]}  "
          f"signal_model={config.graph_model.signal_model_name}")

    voltage_np = to_numpy(voltage)  # (T, N)
    mu_np = to_numpy(mu)
    sigma_np = to_numpy(sigma)

    # Empirical quantiles and median per neuron
    p5 = np.percentile(voltage_np, 5, axis=0)
    p95 = np.percentile(voltage_np, 95, axis=0)
    median = np.median(voltage_np, axis=0)

    # Floored quantiles — ensure at least MIN_WINDOW wide around mu
    half = np.maximum((p95 - p5) / 2, MIN_WINDOW / 2)
    center = (p95 + p5) / 2
    p5_f = center - half
    p95_f = center + half

    results = {}

    # ---------- 1. Baseline: mu +/- 2 sigma, unweighted chord ----------
    print("[method] baseline_mu2sigma")
    s, _ = slopes_chord(model, mu_np - 2 * sigma_np, mu_np + 2 * sigma_np, device)
    results['baseline_mu2sigma'] = derive_tau_from_slope(s)

    # ---------- 2. Empirical quantiles [p5, p95], unweighted chord ----------
    print("[method] quantiles_5_95")
    s, _ = slopes_chord(model, p5, p95, device)
    results['quantiles_5_95'] = derive_tau_from_slope(s)

    # ---------- 3. Quantiles with floor on window width ----------
    print("[method] quantiles_floored")
    s, _ = slopes_chord(model, p5_f, p95_f, device)
    results['quantiles_floored'] = derive_tau_from_slope(s)

    # ---------- 4. Density-weighted fit over [p5, p95] ----------
    print("[method] density_weighted")
    weights = per_neuron_histogram(voltage_np, p5_f, p95_f)
    s, _ = slopes_chord(model, p5_f, p95_f, device, weights=weights)
    results['density_weighted'] = derive_tau_from_slope(s)

    # ---------- 5. Local derivative at v = mu (autograd) ----------
    print("[method] local_deriv_mean")
    dfdv = local_derivative(model, torch.as_tensor(mu_np, dtype=torch.float32), device)
    results['local_deriv_mean'] = derive_tau_from_slope(dfdv)

    # ---------- 6. Local derivative at v = median (autograd) ----------
    print("[method] local_deriv_median")
    dfdv = local_derivative(model, torch.as_tensor(median, dtype=torch.float32), device)
    results['local_deriv_median'] = derive_tau_from_slope(dfdv)

    # ---------- 7. Density-weighted local derivative (avg over sampled v) ----------
    # Sample a handful of voltages per neuron from the empirical distribution,
    # evaluate local derivative at each, and take mean weighted by density.
    print("[method] local_deriv_density")
    K = 32
    T = voltage_np.shape[0]
    # Random rows from the trajectory
    rng = np.random.default_rng(0)
    idx = rng.integers(0, T, size=K)
    v_samples = voltage_np[idx, :]  # (K, N)
    grads_accum = np.zeros(n_neurons)
    for k in range(K):
        g = local_derivative(model, torch.as_tensor(v_samples[k], dtype=torch.float32), device)
        grads_accum += g
    grads_mean = grads_accum / K
    results['local_deriv_density'] = derive_tau_from_slope(grads_mean)

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    report_lines = []
    report_lines.append(f"run: {os.path.basename(log_dir)}")
    report_lines.append(f"n_neurons: {n_neurons}")
    report_lines.append(f"gt_tau  min/mean/max: "
                        f"{gt_tau.min():.4f} / {gt_tau.mean():.4f} / {gt_tau.max():.4f}")
    report_lines.append("")
    report_lines.append(f"{'method':25s}  {'R^2':>7s}  {'slope':>7s}  {'tau-mean':>9s}")
    report_lines.append("-" * 55)

    summary = {}
    if train_log_tau is not None:
        summary['prior_training_log'] = {'R2': float(train_log_tau),
                                         'slope': float('nan'),
                                         'tau_mean': float('nan')}
        report_lines.append(
            f"{'prior_training_log':25s}  {train_log_tau:7.4f}  {'':>7s}  {'':>9s}")
    if plotfig_tau is not None:
        summary['prior_plotfig'] = {'R2': float(plotfig_tau),
                                    'slope': float('nan'),
                                    'tau_mean': float('nan')}
        report_lines.append(
            f"{'prior_plotfig':25s}  {plotfig_tau:7.4f}  {'':>7s}  {'':>9s}")

    for name, tau_hat in results.items():
        r2, slope = compute_r_squared(gt_tau, tau_hat)
        summary[name] = {'R2': float(r2), 'slope': float(slope),
                         'tau_mean': float(tau_hat.mean())}
        report_lines.append(
            f"{name:25s}  {r2:7.4f}  {slope:7.4f}  {tau_hat.mean():9.4f}")

    report = "\n".join(report_lines)
    print("\n" + report)

    with open(os.path.join(out_dir, 'tau_extraction_report.txt'), 'w') as f:
        f.write(report + "\n")
    with open(os.path.join(out_dir, 'tau_extraction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    n_methods = len(results)
    ncols = 4
    nrows = int(np.ceil(n_methods / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes).ravel()
    for ax, (name, tau_hat) in zip(axes, results.items()):
        r2 = summary[name]['R2']
        slope = summary[name]['slope']
        ax.scatter(gt_tau, tau_hat, s=6, alpha=0.3, c='tab:blue')
        lo = min(gt_tau.min(), tau_hat.min())
        hi = max(gt_tau.max(), tau_hat.max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('true tau')
        ax.set_ylabel('learned tau')
        ax.set_title(f'{name}\nR^2={r2:.3f}  slope={slope:.2f}', fontsize=10)
    for ax in axes[len(results):]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'tau_comparison_methods.png'), dpi=150)
    plt.close()

    # Activity-window diagnostic plot: how wide is each method's window?
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    w_mu = 4 * sigma_np
    w_q = p95 - p5
    w_qf = p95_f - p5_f
    order = np.argsort(mu_np)
    ax.plot(w_mu[order], label='mu +/- 2 sigma', alpha=0.7)
    ax.plot(w_q[order], label='p5..p95', alpha=0.7)
    ax.plot(w_qf[order], label='p5..p95 (floored)', alpha=0.7)
    ax.set_xlabel('neuron (sorted by mu_v)')
    ax.set_ylabel('window width (voltage units)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'window_widths.png'), dpi=150)
    plt.close()

    best = max(summary.items(), key=lambda kv: kv[1]['R2'])
    print(f"[best] {best[0]}: R^2={best[1]['R2']:.4f} "
          f"(baseline R^2={summary['baseline_mu2sigma']['R2']:.4f})")

    # free GPU memory between runs
    del model, voltage
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dirs', nargs='*', default=[DEFAULT_LOG_DIR],
                        help='Run folders to analyze (default: single cv00 run)')
    parser.add_argument('--summary-csv', type=str, default=None,
                        help='Optional CSV path to aggregate per-run R^2 results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    all_summaries = {}
    for d in args.log_dirs:
        try:
            all_summaries[os.path.basename(d.rstrip('/'))] = analyze(d, device)
        except Exception as e:
            print(f"[error] {d}: {e}")

    if args.summary_csv and all_summaries:
        method_order = [
            'prior_training_log', 'prior_plotfig',
            'baseline_mu2sigma', 'quantiles_5_95', 'quantiles_floored',
            'density_weighted', 'local_deriv_mean', 'local_deriv_median',
            'local_deriv_density',
        ]
        import csv
        with open(args.summary_csv, 'w') as f:
            w = csv.writer(f)
            w.writerow(['run'] + method_order)
            for run, s in all_summaries.items():
                row = [run]
                for m in method_order:
                    row.append(f"{s[m]['R2']:.4f}" if m in s else "")
                w.writerow(row)
        print(f"\n[csv] wrote {args.summary_csv}")

        # markdown table
        print("\n| run | " + " | ".join(method_order) + " |")
        print("|" + "---|" * (len(method_order) + 1))
        for run, s in all_summaries.items():
            vals = []
            for m in method_order:
                vals.append(f"{s[m]['R2']:.3f}" if m in s else "—")
            print(f"| {run} | " + " | ".join(vals) + " |")


if __name__ == '__main__':
    main()
