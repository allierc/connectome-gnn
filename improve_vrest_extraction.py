"""Standalone analysis of improved V_rest extraction methods.

Now that tau is better extracted via local derivative at median voltage, can
V_rest get a similar lift? V_rest is derived from the linearization
    f_theta(v) approx slope * v + offset
via V_rest = -offset / slope. The current pipeline fits a chord over
[mu - 2*sigma, mu + 2*sigma]; this script evaluates candidates that exploit
the local tangent of f_theta, analogous to the tau improvement.

Methods (applied at the final checkpoint of each run):
    baseline_mu2sigma        : chord slope/offset over [mu-2sigma, mu+2sigma] (current)
    quantiles_floored        : chord over [p5, p95] floored to MIN_WINDOW width
    tangent_at_mean          : V_rest = mu     - f_theta(mu)     / slope_local(mu)
    tangent_at_median        : V_rest = median - f_theta(median) / slope_local(median)
    root_find_nearest_median : find v in [mu-5sigma, mu+5sigma] where f_theta(v)=0;
                               pick crossing closest to median per neuron.

Usage (neural-graph-linux conda env):
    conda run -n neural-graph-linux python improve_vrest_extraction.py
    conda run -n neural-graph-linux python improve_vrest_extraction.py \\
        /groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_free_blank50_unified_cv0{0,1,2,3,4} \\
        --summary-csv /tmp/vrest_extraction_5cv.csv
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
from connectome_gnn.generators.ode_params import get_ode_params_class


DEFAULT_LOG_DIR = '/groups/saalfeld/home/allierc/GraphData/log/fly/flyvis_noise_free_blank50_unified_cv00'


def _paths_for(log_dir):
    run_name = os.path.basename(log_dir.rstrip('/'))
    config_yaml = f'/groups/saalfeld/home/allierc/GraphData/config/fly/{run_name}.yaml'
    out_dir = os.path.join(log_dir, 'vrest_extraction_analysis')
    return config_yaml, out_dir


N_PTS = 1000             # grid size for chord methods
MIN_WINDOW = 0.1         # voltage-unit floor on window width
CHUNK = 2000             # chunk size for batched MLP eval
ROOT_N_PTS = 2000        # grid size for root-finding method
ROOT_WIDEN_SIGMA = 5.0   # half-width of root-find domain in units of sigma


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

    OdeParamsCls = get_ode_params_class(config.graph_model.signal_model_name)
    ode_params = OdeParamsCls.load(graphs_data_path(config.dataset), device='cpu')
    gt_vrest = ode_params.gt_vrest(n_neurons)

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

    return config, model, voltage, mu, sigma, gt_vrest, n_neurons, ode_params


# ------------------------------------------------------------------ #
# f_theta evaluation helpers
# ------------------------------------------------------------------ #

def eval_f_theta_grid(model, rr, device, chunk=CHUNK):
    """Evaluate model.f_theta at (N, n_pts) voltage grid. Returns (N, n_pts) tensor."""
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


def f_theta_at_points(model, v_points_t, device, chunk=CHUNK):
    """f_theta at one v per neuron. Returns (N,) numpy array."""
    v_points_t = v_points_t.to(device)
    N = v_points_t.shape[0]
    a = model.a[:N]
    outs = []
    for i in range(0, N, chunk):
        v_chunk = v_points_t[i:i + chunk].reshape(-1, 1)
        emb_chunk = a[i:i + chunk]
        feats = _build_f_theta_features(v_chunk, emb_chunk)
        with torch.no_grad():
            y = model.f_theta(feats.float()).squeeze(-1)
        outs.append(y)
    return to_numpy(torch.cat(outs, dim=0))


def local_derivative(model, v_points_t, device, chunk=CHUNK):
    """d f_theta / d v at each v_points[i]. Returns (N,) numpy array."""
    v = v_points_t.clone().detach().to(device).requires_grad_(True)
    N = v.shape[0]
    a = model.a[:N]
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
# V_rest extraction strategies
# ------------------------------------------------------------------ #

def vectorized_linear_fit(x, y):
    """Per-row OLS fit y = slope*x + offset."""
    sw = x.shape[1]
    swx = x.sum(axis=1)
    swy = y.sum(axis=1)
    swxy = (x * y).sum(axis=1)
    swxx = (x * x).sum(axis=1)
    denom = sw * swxx - swx * swx
    safe = np.abs(denom) > 1e-12
    slope = np.where(safe, (sw * swxy - swx * swy) / np.where(safe, denom, 1.0), 0.0)
    offset = np.where(safe, (swy - slope * swx) / sw, 0.0)
    return slope, offset


def chord_fit_vrest(model, starts, ends, device):
    """Chord slope/offset over [start, end] -> V_rest = -offset/slope."""
    t = torch.linspace(0, 1, N_PTS, device=device)
    starts_t = torch.as_tensor(starts, dtype=torch.float32, device=device)
    ends_t = torch.as_tensor(ends, dtype=torch.float32, device=device)
    rr = starts_t[:, None] + t[None, :] * (ends_t - starts_t)[:, None]
    func = eval_f_theta_grid(model, rr, device)
    rr_np = to_numpy(rr)
    func_np = to_numpy(func)
    slope, offset = vectorized_linear_fit(rr_np, func_np)
    with np.errstate(divide='ignore', invalid='ignore'):
        vrest = np.where(np.abs(slope) > 1e-8, -offset / slope, 0.0)
    return vrest


def tangent_vrest(model, v_np, device):
    """V_rest = v - f_theta(v) / slope_local(v)."""
    v_t = torch.as_tensor(v_np, dtype=torch.float32, device=device)
    slope = local_derivative(model, v_t, device)
    f_val = f_theta_at_points(model, v_t, device)
    with np.errstate(divide='ignore', invalid='ignore'):
        vrest = np.where(np.abs(slope) > 1e-8, v_np - f_val / slope, v_np)
    return vrest


def root_find_vrest(model, mu_np, sigma_np, median_np, device):
    """Zero-crossing of f_theta on [mu-ROOT_WIDEN*sigma, mu+ROOT_WIDEN*sigma].
    For neurons with multiple crossings, pick the one nearest median voltage.
    Falls back to median_v if no crossing is found."""
    half_width = np.maximum(ROOT_WIDEN_SIGMA * sigma_np, MIN_WINDOW)
    starts = mu_np - half_width
    ends = mu_np + half_width
    t = torch.linspace(0, 1, ROOT_N_PTS, device=device)
    starts_t = torch.as_tensor(starts, dtype=torch.float32, device=device)
    ends_t = torch.as_tensor(ends, dtype=torch.float32, device=device)
    rr = starts_t[:, None] + t[None, :] * (ends_t - starts_t)[:, None]
    func = eval_f_theta_grid(model, rr, device)
    rr_np = to_numpy(rr)
    func_np = to_numpy(func)

    N = rr_np.shape[0]
    vrest = np.full(N, np.nan)
    for i in range(N):
        sign = np.sign(func_np[i])
        idx = np.where(np.diff(sign) != 0)[0]
        if len(idx) == 0:
            vrest[i] = median_np[i]
            continue
        zeros = []
        for j in idx:
            y0 = func_np[i, j]; y1 = func_np[i, j + 1]
            v0 = rr_np[i, j]; v1 = rr_np[i, j + 1]
            if y1 != y0:
                zeros.append(v0 - y0 * (v1 - v0) / (y1 - y0))
        if not zeros:
            vrest[i] = median_np[i]
            continue
        zeros = np.array(zeros)
        best = int(np.argmin(np.abs(zeros - median_np[i])))
        vrest[i] = zeros[best]
    return vrest


# ------------------------------------------------------------------ #
# Reporting
# ------------------------------------------------------------------ #

def _read_prior_baselines(log_dir):
    train_val = None
    plot_val = None
    mlog = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if os.path.exists(mlog):
        with open(mlog) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) >= 2:
            header = lines[0].split(',')
            last = lines[-1].split(',')
            if 'vrest_r2' in header:
                idx = header.index('vrest_r2')
                try:
                    train_val = float(last[idx])
                except (ValueError, IndexError):
                    pass
    mtxt = os.path.join(log_dir, 'results', 'metrics.txt')
    if os.path.exists(mtxt):
        with open(mtxt) as f:
            for line in f:
                if line.startswith('V_rest_R2'):
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

    train_log_v, plotfig_v = _read_prior_baselines(log_dir)
    print(f"[prior] tmp_training/metrics.log vrest_r2 = {train_log_v}")
    print(f"[prior] results/metrics.txt       V_rest_R2 = {plotfig_v}")

    print("[load] config, voltage, model ...")
    config, model, voltage, mu, sigma, gt_vrest, n_neurons, ode_params = load_everything(
        device, log_dir, config_yaml)
    print(f"[load] n_neurons={n_neurons}  T={voltage.shape[0]}  "
          f"signal_model={config.graph_model.signal_model_name}")

    voltage_np = to_numpy(voltage)
    mu_np = to_numpy(mu)
    sigma_np = to_numpy(sigma)

    p5 = np.percentile(voltage_np, 5, axis=0)
    p95 = np.percentile(voltage_np, 95, axis=0)
    median_np = np.median(voltage_np, axis=0)
    half = np.maximum((p95 - p5) / 2, MIN_WINDOW / 2)
    center = (p95 + p5) / 2
    p5_f = center - half
    p95_f = center + half

    results = {}

    print("[method] baseline_mu2sigma")
    results['baseline_mu2sigma'] = chord_fit_vrest(
        model, mu_np - 2 * sigma_np, mu_np + 2 * sigma_np, device)

    print("[method] quantiles_floored")
    results['quantiles_floored'] = chord_fit_vrest(model, p5_f, p95_f, device)

    print("[method] tangent_at_mean")
    results['tangent_at_mean'] = tangent_vrest(model, mu_np, device)

    print("[method] tangent_at_median")
    results['tangent_at_median'] = tangent_vrest(model, median_np, device)

    print("[method] root_find_nearest_median")
    results['root_find_nearest_median'] = root_find_vrest(
        model, mu_np, sigma_np, median_np, device)

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    report_lines = []
    report_lines.append(f"run: {os.path.basename(log_dir)}")
    report_lines.append(f"n_neurons: {n_neurons}")
    if gt_vrest is not None:
        report_lines.append(f"gt_vrest  min/mean/max: "
                            f"{gt_vrest.min():.4f} / {gt_vrest.mean():.4f} / {gt_vrest.max():.4f}")
    report_lines.append("")
    report_lines.append(f"{'method':28s}  {'R^2':>7s}  {'slope':>7s}  {'v-mean':>8s}")
    report_lines.append("-" * 58)

    summary = {}
    if train_log_v is not None:
        summary['prior_training_log'] = {'R2': float(train_log_v),
                                         'slope': float('nan'),
                                         'vrest_mean': float('nan')}
        report_lines.append(
            f"{'prior_training_log':28s}  {train_log_v:7.4f}  {'':>7s}  {'':>8s}")
    if plotfig_v is not None:
        summary['prior_plotfig'] = {'R2': float(plotfig_v),
                                    'slope': float('nan'),
                                    'vrest_mean': float('nan')}
        report_lines.append(
            f"{'prior_plotfig':28s}  {plotfig_v:7.4f}  {'':>7s}  {'':>8s}")

    for name, vrest_hat in results.items():
        if gt_vrest is not None:
            r2, slope = compute_r_squared(gt_vrest, vrest_hat)
        else:
            r2, slope = float('nan'), float('nan')
        summary[name] = {'R2': float(r2), 'slope': float(slope),
                         'vrest_mean': float(np.nanmean(vrest_hat))}
        report_lines.append(
            f"{name:28s}  {r2:7.4f}  {slope:7.4f}  {np.nanmean(vrest_hat):8.4f}")

    report = "\n".join(report_lines)
    print("\n" + report)

    with open(os.path.join(out_dir, 'vrest_extraction_report.txt'), 'w') as f:
        f.write(report + "\n")
    with open(os.path.join(out_dir, 'vrest_extraction_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------ #
    # Plots
    # ------------------------------------------------------------------ #
    if gt_vrest is not None:
        n_methods = len(results)
        ncols = 3
        nrows = int(np.ceil(n_methods / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = np.atleast_2d(axes).ravel()
        for ax, (name, vrest_hat) in zip(axes, results.items()):
            r2 = summary[name]['R2']
            slope = summary[name]['slope']
            ax.scatter(gt_vrest, vrest_hat, s=6, alpha=0.3, c='tab:blue')
            lo = float(gt_vrest.min())
            hi = float(gt_vrest.max())
            pad = 0.2 * (hi - lo + 1e-3)
            lo -= pad; hi += pad
            ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel('true V_rest')
            ax.set_ylabel('learned V_rest')
            ax.set_title(f'{name}\nR^2={r2:.3f}  slope={slope:.2f}', fontsize=10)
        for ax in axes[len(results):]:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'vrest_comparison_methods.png'), dpi=150)
        plt.close()

    non_prior = {k: v for k, v in summary.items() if not k.startswith('prior_')}
    if non_prior:
        best = max(non_prior.items(),
                   key=lambda kv: kv[1]['R2'] if not np.isnan(kv[1]['R2']) else -1)
        baseline_r2 = non_prior.get('baseline_mu2sigma', {}).get('R2', float('nan'))
        print(f"[best] {best[0]}: R^2={best[1]['R2']:.4f}  "
              f"(baseline R^2={baseline_r2:.4f})")

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
            import traceback; traceback.print_exc()
            print(f"[error] {d}: {e}")

    if args.summary_csv and all_summaries:
        method_order = [
            'prior_training_log', 'prior_plotfig',
            'baseline_mu2sigma', 'quantiles_floored',
            'tangent_at_mean', 'tangent_at_median',
            'root_find_nearest_median',
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

        print("\n| run | " + " | ".join(method_order) + " |")
        print("|" + "---|" * (len(method_order) + 1))
        for run, s in all_summaries.items():
            vals = []
            for m in method_order:
                vals.append(f"{s[m]['R2']:.3f}" if m in s else "—")
            print(f"| {run} | " + " | ".join(vals) + " |")


if __name__ == '__main__':
    main()
