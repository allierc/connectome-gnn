"""GNN+INR stimulus recovery and rollout — fig_stim_rollout_inr.py.

Layout (after the 2026-04 revisit, modelled on
``fig_rollout_3col_noise_comparison.py``):

  row a) 3 × 11 hex grid of GT photoreceptor stimuli across 11 evenly-
         spaced frames within the trace window
         (top = GT, middle = INR-predicted, bottom = residual).
  row b/c) side-by-side trace panels — same style as the 3-col rollout
         figure, no residual column:
         b) stimulus rollout — 12 representative photoreceptors
         c) voltage rollout  — 12 representative cell types
  row d/e/f) three scatter panels at the bottom:
         d) learned INR vs true INR (pooled over neuron × frame)
         e) rollout voltage vs noisy ground-truth voltage
         f) rollout voltage vs noise-free ground-truth voltage
            (loaded from NF_BUNDLE_PATH; shows a placeholder if absent).

Data sources (rollout_bundle.npz at
  <output_root>/log/fly/flyvis_noise_005_INR_davis_cv00/results/):
  - activity_true / activity_pred              (n_neurons, n_frames)
  - stimulus_input_true / stimulus_input_pred  (n_frames, n_input)
  - stimulus_input_pred_corrected (when present — matches the reported
    stimuli_r in results_rollout.log).
  - type_ids, type_names

Hex positions come from the simulation data (x_list_train/pos field).

Usage
-----
    /workspace/.conda_envs/neural-graph-linux/bin/python \\
        figures/fig_stim_rollout_inr.py

Output
------
    figures/fig_stim_rollout_inr.{pdf,png}
"""

import os
import shutil
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
matplotlib.rc_file(os.path.join(os.path.dirname(__file__), 'janne.matplotlibrc'))

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import matplotlib.cm as _mcm
import matplotlib.colors as _mcolors
import numpy as np
import yaml


# Try the flyvis trim_axis; fall back to a local equivalent if unavailable.
try:
    from flyvis.analysis.visualization.plt_utils import trim_axis as _trim_axis
except Exception:
    def _trim_axis(ax, xmargin=0.0, ymargin=0.0, yaxis=True, xaxis=True):
        if xaxis:
            xticks = ax.get_xticks()
            xlo, xhi = ax.get_xlim()
            xticks = [t for t in xticks if xlo <= t <= xhi]
            if xticks:
                ax.spines['bottom'].set_bounds(xticks[0], xticks[-1])
        if yaxis:
            yticks = ax.get_yticks()
            ylo, yhi = ax.get_ylim()
            yticks = [t for t in yticks if ylo <= t <= yhi]
            if yticks:
                ax.spines['left'].set_bounds(yticks[0], yticks[-1])


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
for _p in (os.path.join(REPO_ROOT, 'src'), REPO_ROOT):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import connectome_gnn.utils as _cg_utils  # noqa: E402
from connectome_gnn.utils import graphs_data_path  # noqa: E402
from connectome_gnn.zarr_io import load_simulation_data  # noqa: E402


# ── config ──────────────────────────────────────────────────────────────────
CONFIG_NAME = 'flyvis_noise_005_INR_davis_cv00'
DATA_ROOT = '/groups/saalfeld/home/allierc/GraphData'
REPO_ROOT_FOR_RUN = REPO_ROOT          # alias for clarity in subprocess calls
BASE_DIR = os.path.join(DATA_ROOT, 'log', 'fly', CONFIG_NAME)
BUNDLE_PATH = os.path.join(BASE_DIR, 'results', 'rollout_bundle.npz')

# Yaml paths needed by the ensure_* pipeline. The cv00 model yaml is the
# canonical parent; both `_noisy` and `_nf` datasets are cloned from it
# so they share seed=42, DAVIS root, blank fraction, and the per-frame
# stimulus-computation pipeline. ONLY noise_model_level differs between
# the two variants → the photoreceptor stimulus arrays stored on disk are
# identical, only the voltage trajectories diverge.
CFG_DIR        = os.path.join(DATA_ROOT, 'config', 'fly')
INR_BASE_YAML  = os.path.join(REPO_ROOT, 'config', 'fly',
                               'flyvis_noise_005_INR.yaml')
CV00_YAML      = os.path.join(CFG_DIR, f'{CONFIG_NAME}.yaml')

NOISY_DATASET    = f'{CONFIG_NAME}_noisy'
NOISY_YAML       = os.path.join(CFG_DIR, f'{NOISY_DATASET}.yaml')
NOISY_DATA_DIR   = os.path.join(DATA_ROOT, 'graphs_data', 'fly', NOISY_DATASET)
NOISY_BUNDLE_PATH = os.path.join(
    BASE_DIR, 'results',
    f'rollout_bundle_on_{NOISY_DATASET.replace("flyvis_", "")}.npz')

NF_DATASET     = f'{CONFIG_NAME}_nf'
NF_YAML        = os.path.join(CFG_DIR, f'{NF_DATASET}.yaml')
NF_DATA_DIR    = os.path.join(DATA_ROOT, 'graphs_data', 'fly', NF_DATASET)
NF_BUNDLE_PATH = os.environ.get(
    'INR_NF_BUNDLE_PATH',
    os.path.join(BASE_DIR, 'results',
                 f'rollout_bundle_on_{NF_DATASET.replace("flyvis_", "")}.npz'),
)

# DAVIS stimulus root — same fallback list as fig_rollout_3col_noise_comparison.
_DAVIS_CANDIDATES = [
    '/groups/saalfeld/home/kumarv4/web_datasets/DAVIS2017-partial-test/',
    '/groups/saalfeld/home/allierc/signaling/DATAVIS/',
    os.environ.get('DATAVIS_ROOT', ''),
]
DAVIS_ROOT = next(
    (p for p in _DAVIS_CANDIDATES
     if p and os.path.isdir(os.path.join(p, 'JPEGImages/480p'))),
    None,
)


# ── subprocess + yaml helpers ───────────────────────────────────────────────
def _run(*args, tag):
    print(f'{tag} python GNN_Main.py {" ".join(args)}')
    subprocess.check_call(
        ['python', os.path.join(REPO_ROOT_FOR_RUN, 'GNN_Main.py'), *args,
         '--output_root', DATA_ROOT],
        cwd=REPO_ROOT_FOR_RUN,
    )


def _clone_yaml(src_yaml, dst_yaml, dataset_name, description, sim_overrides,
                config_file=None):
    """Copy src_yaml → dst_yaml, override dataset/description/simulation."""
    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)
    cfg['description'] = description
    cfg['dataset']     = dataset_name
    if config_file is not None:
        cfg['config_file'] = config_file
    sim = cfg.get('simulation', {})
    sim.setdefault('seed', 42)
    sim.update(sim_overrides)
    cfg['simulation'] = sim
    os.makedirs(os.path.dirname(dst_yaml), exist_ok=True)
    with open(dst_yaml, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def ensure_cv00_yaml():
    """Materialise the cv00 model yaml from the repo's INR base if missing."""
    if os.path.isfile(CV00_YAML):
        return
    if not os.path.isfile(INR_BASE_YAML):
        sys.exit(f'ERROR: {INR_BASE_YAML} missing — cannot reconstruct '
                 f'{CV00_YAML}. Either restore the base yaml or place the '
                 f'cv00 yaml in DATA_ROOT/config/fly/ manually.')
    print(f'cloning INR base → {CV00_YAML}')
    sim_overrides = {
        # Match the existing cv00 bundle (8000 frames → noisy_test_data=True).
        'noise_model_level': 0.05,
        'noisy_test_data'  : True,
        'visual_input_type': 'DAVIS',
        'skip_short_videos' : True,
        'blank_prefix_fraction': 0.5,
    }
    if DAVIS_ROOT is not None:
        sim_overrides['datavis_roots'] = [DAVIS_ROOT]
    _clone_yaml(
        INR_BASE_YAML, CV00_YAML,
        dataset_name=CONFIG_NAME,
        description=(
            f'Reconstructed cv00 yaml for {CONFIG_NAME} '
            f'(model_yaml needed by figures/fig_stim_rollout_inr.py).'
        ),
        sim_overrides=sim_overrides,
        config_file=f'fly/{CONFIG_NAME}',
    )


def _ensure_variant(*, yaml_path, data_dir, bundle_path, dataset, noise_level,
                     marker_name='noisy_test_data.ok', tag=CONFIG_NAME):
    """Common pipeline: clone cv00 yaml → variant; generate dataset; rollout."""
    ensure_cv00_yaml()

    if not os.path.isfile(yaml_path):
        print(f'cloning cv00 → {yaml_path} (noise={noise_level})')
        sim_overrides = {
            'noise_model_level': noise_level,
            # noisy_test_data=True keeps the test split shape consistent
            # (~8000 frames) regardless of noise level.
            'noisy_test_data'  : True,
        }
        if DAVIS_ROOT is not None:
            sim_overrides['datavis_roots'] = [DAVIS_ROOT]
        _clone_yaml(
            CV00_YAML, yaml_path,
            dataset_name=dataset,
            description=(
                f'Noise={noise_level} twin of {CONFIG_NAME} (figure '
                f'fig_stim_rollout_inr.py): seed=42, same DAVIS videos.'
            ),
            sim_overrides=sim_overrides,
            config_file=f'fly/{dataset}',
        )
    else:
        print(f'yaml exists: {yaml_path}')

    marker = os.path.join(data_dir, marker_name)
    if not os.path.isfile(marker):
        if os.path.isdir(data_dir):
            print(f'removing stale {data_dir}')
            shutil.rmtree(data_dir)
        print(f'generating dataset {dataset} (tens of minutes)')
        _run('-o', 'generate', yaml_path, tag=f'[{tag}]')
        if not os.path.isfile(marker):
            sys.exit(f'expected marker missing after generation: {marker}')
    else:
        print(f'dataset exists: {data_dir}')

    if not os.path.isfile(bundle_path):
        print(f'running rollout on {dataset}')
        _run('-o', 'test', CV00_YAML, 'best', yaml_path, tag=f'[{tag}]')
    else:
        print(f'bundle exists: {bundle_path}')


def ensure_noisy_variant():
    """Generate flyvis_noise_005_INR_davis_cv00_noisy (σ=0.05, same DAVIS+seed)."""
    _ensure_variant(yaml_path=NOISY_YAML, data_dir=NOISY_DATA_DIR,
                    bundle_path=NOISY_BUNDLE_PATH,
                    dataset=NOISY_DATASET, noise_level=0.05)


def ensure_nf_variant():
    """Generate flyvis_noise_005_INR_davis_cv00_nf (σ=0, same DAVIS+seed)."""
    _ensure_variant(yaml_path=NF_YAML, data_dir=NF_DATA_DIR,
                    bundle_path=NF_BUNDLE_PATH,
                    dataset=NF_DATASET, noise_level=0.0)


# ── Synthetic noise-free ODE pass ──────────────────────────────────────────
# Path of the synthetic bundle produced by re-running the flyvis_A ground
# truth ODE forward with σ=0 over the EXISTING training-rollout window.
# This bundle keeps the original `activity_pred` (model's learned rollout)
# and replaces `activity_true` with the deterministic ε=0 trajectory of
# the same stimulus, starting from the same IC. Used by panels d / g so
# the comparison is "same stimulus + same IC, only per-step noise differs".
NF_SYNTH_BUNDLE_PATH = os.path.join(
    BASE_DIR, 'results', 'rollout_bundle_nf_synthetic.npz')


def ensure_nf_synthetic_bundle():
    """Re-simulate the original training-rollout window with σ=0 using the
    original bundle's stimulus and initial voltage. Output: a bundle whose
    `activity_true` is the deterministic version of the SAME stimulus the
    INR was trained on. Skipped if already cached.
    """
    if os.path.isfile(NF_SYNTH_BUNDLE_PATH):
        print(f'synthetic nf bundle exists: {NF_SYNTH_BUNDLE_PATH}')
        return

    print(f'building synthetic nf bundle (σ=0 ODE over original stim) → '
          f'{NF_SYNTH_BUNDLE_PATH}')
    import torch
    from tqdm import trange
    from flyvis import Network, NetworkView
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.generators.flyvis_ode import FlyVisODE
    from connectome_gnn.generators.ode_params import FlyVisODEParams
    from connectome_gnn.neuron_state import NeuronState
    from connectome_gnn.utils import setup_flyvis_model_path

    setup_flyvis_model_path()

    # 1. Load the cv00 yaml (already auto-cloned by ensure_cv00_yaml above).
    ensure_cv00_yaml()
    cfg = NeuralGraphConfig.from_yaml(CV00_YAML)
    sim = cfg.simulation

    # 2. Build the standard flyvis_A network + extract ode_params.
    config_net = get_default_config(
        overrides=[], path=f'{CONFIG_PATH}/network/network.yaml')
    config_net.connectome.extent = 8
    net = Network(**config_net)
    nnv = NetworkView(f'flow/{sim.ensemble_id}/{sim.model_id}')
    net.load_state_dict(nnv.init_network(checkpoint=0).state_dict())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ode_params = FlyVisODEParams.from_flyvis_network(net, device=device)
    edge_index = ode_params.edge_index
    pde = FlyVisODE(
        ode_params=ode_params,
        g_phi=torch.nn.functional.relu,
        params=sim.params,
        model_type=cfg.graph_model.signal_model_name,
        n_neuron_types=sim.n_neuron_types,
        device=device,
    )

    # 3. Load original bundle and pull stim + IC + activity_pred + metadata.
    orig = np.load(BUNDLE_PATH, allow_pickle=True)
    stim = torch.from_numpy(np.asarray(orig['stimulus'])).to(device)         # (N, T)
    n_neurons, n_frames = stim.shape
    ic   = torch.from_numpy(np.asarray(orig['activity_true'][:, 0])).to(device).float()
    print(f'  stim={tuple(stim.shape)}  ic={tuple(ic.shape)}  delta_t={sim.delta_t}')

    # 4. Build a NeuronState carrying the per-frame stimulus and integrate
    # forward with σ=0. Only the fields the ODE actually reads need to be
    # populated; calcium / hh_* / fluorescence / noise are zeroed.
    zeros = torch.zeros(n_neurons, dtype=torch.float32, device=device)
    x = NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=torch.zeros((n_neurons, 2), dtype=torch.float32, device=device),
        voltage=ic.clone(),
        stimulus=zeros.clone(),
        group_type=zeros.long().clone(),
        neuron_type=zeros.long().clone(),
        calcium=zeros.clone(),
        fluorescence=zeros.clone(),
        noise=zeros.clone(),
    )

    activity_nf = torch.empty((n_neurons, n_frames),
                              dtype=torch.float32, device=device)
    with torch.no_grad():
        for t in trange(n_frames, ncols=80, desc='σ=0 ODE'):
            x.stimulus = stim[:, t].clone()
            activity_nf[:, t] = x.voltage          # save BEFORE update
            dv = pde(x, edge_index, has_field=False).squeeze()
            x.voltage = x.voltage + sim.delta_t * dv

    activity_nf_np = activity_nf.cpu().numpy()
    print(f'  σ=0 trajectory: shape={activity_nf_np.shape} '
          f'std={activity_nf_np.std():.4f} (vs noisy std={float(np.asarray(orig["activity_true"]).std()):.4f})')

    # 5. Save synthetic bundle. Keep `activity_pred` from the original
    # bundle so panels d/g use the SAME model rollout as panels c/f.
    bundle_out = {
        'activity_true': activity_nf_np,
        'activity_pred': np.asarray(orig['activity_pred']),
        'stimulus':      np.asarray(orig['stimulus']),
        'type_ids':      np.asarray(orig['type_ids']),
        'type_names':    np.asarray(orig['type_names']),
    }
    if 'stimulus_input_true' in orig.files:
        bundle_out['stimulus_input_true'] = np.asarray(orig['stimulus_input_true'])
    if 'stimulus_input_pred' in orig.files:
        bundle_out['stimulus_input_pred'] = np.asarray(orig['stimulus_input_pred'])
    if 'stimulus_input_pred_corrected' in orig.files:
        bundle_out['stimulus_input_pred_corrected'] = np.asarray(
            orig['stimulus_input_pred_corrected'])
    np.savez_compressed(NF_SYNTH_BUNDLE_PATH, **bundle_out)
    print(f'wrote {NF_SYNTH_BUNDLE_PATH}')

# hexagon panel — 3 rows x 11 cols at evenly spaced frames within trace window
N_INPUT = 1736                # photoreceptor count for 217-column flyvis
SERIES_COLS = 11

# trace window (frame indices into rollout_bundle arrays)
TRACE_START = 500
TRACE_END   = 1500
DT_MS = 20.0

SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]
N_STIM_TRACES = 12

# Trace style — match fig_rollout_3col_noise_comparison.py.
COLOR_GT   = '#2ca02c'
COLOR_PRED = 'black'
LW_GT, LW_PRED = 1.2, 0.45

# Trace amplitude scale (also scales step_v, so the trace-to-gap ratio is
# preserved while the absolute size is reduced — same knob as fig_rollout).
TRACE_SHRINK = 0.65

# Scatter — pooled (neuron, frame), subsampled.
SCATTER_N_MAX = 2_000_000
SCATTER_RNG   = np.random.default_rng(0)
SCATTER_LO_V, SCATTER_HI_V = -10.0, 10.0   # voltage range
SCATTER_LO_S, SCATTER_HI_S = 0.0, 1.0      # stimulus range (raw [0, 1] DAVIS pixels)

# Fonts (janne.matplotlibrc sets defaults to 8/6 pt).
FS_LABEL  = 8
FS_TICK   = 6
FS_ANNOT  = 6
FS_LEGEND = 6
FS_TYPE   = 6
PANEL_LBL = 8

# ~18 cm wide; 3 rows of content (hex / traces / scatters).
FIG_W_IN  = 18.0 * 0.3937       # ≈ 7.09 in
FIG_H_IN  = 7.7

CMAP = 'RdBu_r'
HEX_VMIN, HEX_VMAX = -3.0, 3.0
HEX_MARKER_S = 6
HEX_EDGE_C = 'black'
HEX_EDGE_W = 0.1


# ── data loading ────────────────────────────────────────────────────────────
def _set_data_root(path):
    _cg_utils._data_root = path


def load_bundle(path):
    if not os.path.isfile(path):
        sys.exit(
            f'ERROR: bundle missing at {path}\n'
            '  re-run `-o test` to regenerate with the new stimulus fields:\n'
            f'    python GNN_Main.py -o test {CONFIG_NAME} best {CONFIG_NAME} '
            f'--output_root {DATA_ROOT}'
        )
    b = np.load(path, allow_pickle=True)
    keys = list(b.keys())
    if 'stimulus_input_true' not in keys or 'stimulus_input_pred' not in keys:
        sys.exit(
            'ERROR: rollout_bundle.npz does not contain stimulus_input_true /\n'
            '       stimulus_input_pred — re-run `-o test` with the patched\n'
            '       graph_tester.py.'
        )
    return b


def load_positions():
    """Hex positions are fixed by the connectome (photoreceptor omatidia
    centres), so any flyvis-noise dataset works as a source. Falls back
    through a list of likely candidates so the script keeps running even
    when the cv-specific train zarr isn't present at this data root."""
    _set_data_root(DATA_ROOT)
    candidates = [CONFIG_NAME, 'flyvis_noise_005', 'flyvis_noise_005_blank50']
    last_err = None
    for ds in candidates:
        path = os.path.join(graphs_data_path('fly', ds), 'x_list_train')
        try:
            x_ts = load_simulation_data(path, fields=['pos'])
            print(f'positions: loaded from {path}')
            return x_ts.pos.numpy().astype(np.float32)
        except FileNotFoundError as e:
            last_err = e
    raise FileNotFoundError(
        f'no x_list_train/pos found in any of: {candidates} '
        f'under {DATA_ROOT}/graphs_data/fly/'
    ) from last_err


# ── hex panel helpers ───────────────────────────────────────────────────────
def _zscore(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def _draw_hex(ax, xy, values, xlim, ylim, vmin=HEX_VMIN, vmax=HEX_VMAX):
    ax.scatter(xy[:, 0], xy[:, 1], c=values,
               s=HEX_MARKER_S, marker='h',
               cmap=CMAP, vmin=vmin, vmax=vmax,
               edgecolors=HEX_EDGE_C, linewidths=HEX_EDGE_W, alpha=1.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)


def _pretty_xticks(ax, lo, hi, n_target=4):
    span = hi - lo
    raw_step = span / max(1, n_target - 1)
    mag = 10 ** np.floor(np.log10(max(raw_step, 1e-12)))
    step = mag
    for m in (1, 2, 5, 10):
        if m * mag >= raw_step:
            step = m * mag
            break
    tick_lo = np.ceil(lo / step - 1e-9) * step
    ticks = np.arange(tick_lo, hi + step / 2, step)
    if len(ticks) == 0 or ticks[-1] < hi - step * 1e-6:
        ticks = np.append(ticks, hi)
    ax.set_xticks(ticks)
    ax.set_xlim([lo, hi])


# ── trace panel (no residuals, fig_rollout style) ───────────────────────────
def draw_trace_panel(ax, true_w, pred_w, labels, step_v, time_ms,
                     pearson_r, header_label, show_xlabel,
                     show_type_labels=True):
    n_traces, n_frames = true_w.shape
    baselines = true_w.mean(axis=1)
    s = TRACE_SHRINK
    for i in range(n_traces):
        bl = baselines[i]
        ax.plot(time_ms, s * (true_w[i] - bl) + i * step_v,
                lw=LW_GT, color=COLOR_GT, alpha=0.95, zorder=2)
        ax.plot(time_ms, s * (pred_w[i] - bl) + i * step_v,
                lw=LW_PRED, color=COLOR_PRED, alpha=0.95, zorder=3)
    if show_type_labels:
        for i, lbl in enumerate(labels):
            ax.text(time_ms[0] - (time_ms[-1] - time_ms[0]) * 0.025,
                    i * step_v, lbl, fontsize=FS_TYPE,
                    va='bottom', ha='right', color='black')

    # Header inside the axes top-left (same as fig_rollout).
    r_txt = f'{pearson_r:.2f}' if pearson_r is not None else 'n/a'
    ax.text(0.015, 0.99,
            f'{header_label}, $r$ = {r_txt}',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=FS_TICK, fontweight='normal',
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.85, pad=0.4))

    # Headroom above the topmost trace so the in-axes header doesn't
    # collide with its wiggle peak.
    ax.set_ylim([-step_v, (n_traces - 1) * step_v + 2.2 * step_v])
    ax.set_yticks([])
    ax.set_xlim([time_ms[0], time_ms[-1]])
    ax.spines['left'].set_visible(False)
    if show_xlabel:
        ticks = np.linspace(time_ms[0], time_ms[-1], 3)
        ax.set_xticks(ticks)
        ax.set_xlabel('time (ms)', fontsize=FS_LABEL, labelpad=1)
        ax.tick_params(axis='x', labelsize=FS_TICK, pad=1)
        _trim_axis(ax, yaxis=False)
    else:
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)


# ── scatter helper ──────────────────────────────────────────────────────────
def _subsample_pair(x_full, y_full, n_max=SCATTER_N_MAX):
    assert x_full.shape == y_full.shape
    x = x_full.reshape(-1).astype(np.float32)
    y = y_full.reshape(-1).astype(np.float32)
    n_tot = x.size
    if n_tot <= n_max:
        return x, y, n_tot
    stride = int(np.ceil(n_tot / n_max))
    offset = int(SCATTER_RNG.integers(0, stride))
    return x[offset::stride], y[offset::stride], n_tot


def draw_scatter(ax, x_all, y_all, lo, hi, xlabel, ylabel, title=None):
    """Hexbin density of (x, y) on a fixed range with R²/slope inside."""
    x, y, _ = _subsample_pair(x_all, y_all)
    corr = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x.astype(np.float64),
                                   y.astype(np.float64), 1)
    ax.hexbin(x, y, gridsize=140, bins='log', cmap='magma_r',
              mincnt=1, extent=(lo, hi, lo, hi), linewidths=0.0)
    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.set_xticks([lo, 0.0, hi])
    ax.set_yticks([lo, 0.0, hi])
    _trim_axis(ax)
    # Header above axes (subtitle if any) + R²/slope inside top-left.
    if title is not None:
        ax.text(0.5, 1.02, title, transform=ax.transAxes,
                va='bottom', ha='center', fontsize=FS_TICK,
                fontweight='normal')
    ax.text(0.05, 0.97,
            f"$R^2$ = {corr ** 2:.2f}\nslope = {slope:.2f}",
            transform=ax.transAxes, va='top', ha='left',
            fontsize=FS_TICK)


def draw_missing_panel(ax, message):
    """Render a fenced placeholder so the text stays inside the panel."""
    # Keep an empty axes box (don't set_axis_off) so the panel still has
    # the same footprint as a real scatter; clip text so it can't overflow.
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(0.5)
    ax.text(0.5, 0.5, message,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=FS_ANNOT, color='red',
            wrap=True, clip_on=True)


# ── pearson-from-log helper ─────────────────────────────────────────────────
def _parse_rollout_log(path):
    import re
    out = {'voltage': None, 'stimulus': None}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        txt = f.read()
    m = re.search(r'Pearson r:\s*([-\d.]+)', txt)
    if m:
        out['voltage'] = float(m.group(1))
    m = re.search(r'stimuli_r:\s*([-\d.]+)', txt)
    if m:
        out['stimulus'] = float(m.group(1))
    return out


# ── main ────────────────────────────────────────────────────────────────────
def main():
    # Panels b/c/e/f are sourced from the ORIGINAL training-time bundle
    # (this is the only bundle whose photoreceptor stimulus matches the
    # one the INR was actually fit on — the original 64k training frames
    # the SIREN learned to predict). We never overwrite this file.
    bundle = load_bundle(BUNDLE_PATH)
    activity_true = bundle['activity_true']                         # (N, T)
    activity_pred = bundle['activity_pred']
    stim_in_true  = bundle['stimulus_input_true']                   # (T, n_input)
    stim_in_pred = (bundle['stimulus_input_pred_corrected']
                    if 'stimulus_input_pred_corrected' in bundle.files
                    else bundle['stimulus_input_pred'])
    type_ids      = bundle['type_ids'].astype(int)
    type_names    = list(bundle['type_names'])
    index_to_name = {i: type_names[i] for i in range(len(type_names))}
    print(f'original bundle: activity_true={activity_true.shape}  '
          f'stim_in={stim_in_true.shape}')

    # Panels d/g (noise-free) — re-simulate σ=0 ODE over the SAME stimulus
    # the INR was trained on, starting from the same IC. Result: a
    # synthetic bundle whose `activity_true` is the deterministic ε=0
    # trajectory of the same SDE; `activity_pred` is copied from the
    # original bundle so the model rollout is shared between panels.
    ensure_nf_synthetic_bundle()
    _nb = np.load(NF_SYNTH_BUNDLE_PATH, allow_pickle=True)
    nf_bundle = {
        'true': np.asarray(_nb['activity_true']),
        'pred': np.asarray(_nb['activity_pred']),
    }
    print(f'synthetic nf bundle: true={nf_bundle["true"].shape}  '
          f'pred={nf_bundle["pred"].shape}')

    # Sanity check: synthetic nf bundle MUST share its stimulus with the
    # original (it was built from it) and its activity_pred must be
    # identical to the original bundle's.
    same_stim = bool(np.array_equal(np.asarray(_nb['stimulus']),
                                     np.asarray(bundle['stimulus'])))
    same_pred = bool(np.array_equal(np.asarray(_nb['activity_pred']),
                                     np.asarray(bundle['activity_pred'])))
    print(f'sanity: synth nf stimulus == original? {same_stim}   '
          f'pred == original? {same_pred}')

    pos = load_positions()
    n_input = stim_in_true.shape[1]
    pos_input = pos[:n_input]
    _pad_x = (pos_input[:, 0].max() - pos_input[:, 0].min()) * 0.03
    _pad_y = (pos_input[:, 1].max() - pos_input[:, 1].min()) * 0.03
    HEX_XLIM = (pos_input[:, 0].min() - _pad_x, pos_input[:, 0].max() + _pad_x)
    HEX_YLIM = (pos_input[:, 1].min() - _pad_y, pos_input[:, 1].max() + _pad_y)

    T = stim_in_true.shape[0]
    hex_step_frames = int(round(80.0 / DT_MS))
    t0 = min(TRACE_START, T - 1 - hex_step_frames * (SERIES_COLS - 1))
    series_frames = np.array(
        [t0 + k * hex_step_frames for k in range(SERIES_COLS)], dtype=int
    )

    # ── pick voltage traces ────────────────────────────────────────────────
    neuron_idx, labels_v = [], []
    for t in SELECTED_TYPES:
        ids = np.where(type_ids == t)[0]
        if len(ids) > 0:
            neuron_idx.append(int(ids[0]))
            labels_v.append(index_to_name.get(t, f'Type{t}'))

    true_v = activity_true[neuron_idx, TRACE_START:TRACE_END].astype(np.float32)
    pred_v = activity_pred[neuron_idx, TRACE_START:TRACE_END].astype(np.float32)

    stim_idx = np.linspace(0, n_input - 1, N_STIM_TRACES, dtype=int)
    labels_s = [f'R{(i % 8) + 1}' for i in range(N_STIM_TRACES)]
    true_s = stim_in_true[TRACE_START:TRACE_END, stim_idx].T.astype(np.float32)
    pred_s = stim_in_pred[TRACE_START:TRACE_END, stim_idx].T.astype(np.float32)

    n_frames = true_v.shape[1]
    time_ms = np.arange(n_frames) * DT_MS + TRACE_START * DT_MS

    step_v_volt = max(0.5 * TRACE_SHRINK,
                      3.0 * TRACE_SHRINK * float(np.std(true_v)))
    step_v_stim = max(0.5 * TRACE_SHRINK,
                      3.0 * TRACE_SHRINK * float(np.std(true_s)))

    # Compute pooled Pearson r directly from the loaded bundles so the
    # header text reflects the actual noisy variant we just generated.
    r_volt = float(np.corrcoef(activity_true.ravel(), activity_pred.ravel())[0, 1])
    r_stim = float(np.corrcoef(stim_in_true.ravel(), stim_in_pred.ravel())[0, 1])
    print(f'  voltage Pearson r  (noisy) = {r_volt:.3f}')
    print(f'  stimulus Pearson r (INR)   = {r_stim:.3f}')

    # ── figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
    outer = mgs.GridSpec(3, 1, figure=fig,
                         height_ratios=[2.4, 1.4, 1.4],
                         left=0.06, right=0.92, top=0.97, bottom=0.05,
                         hspace=0.10)        # tighter blank between rows

    # (a) 3 × 11 hexagons — minimal hspace between the GT / learned / residual rows
    gs_a = mgs.GridSpecFromSubplotSpec(3, SERIES_COLS, subplot_spec=outer[0],
                                        wspace=0.05, hspace=-0.10)
    axes_hex_top = []
    axes_hex_mid = []
    axes_hex_res = []
    for col, t in enumerate(series_frames):
        vals_gt = _zscore(stim_in_true[t, :])
        vals_pd = _zscore(stim_in_pred[t, :])
        vals_rs = vals_pd - vals_gt

        ax_gt = fig.add_subplot(gs_a[0, col])
        _draw_hex(ax_gt, pos_input, vals_gt, HEX_XLIM, HEX_YLIM)
        ax_gt.set_title(f't = {int(t * DT_MS)} ms', fontsize=4, pad=1)
        axes_hex_top.append(ax_gt)

        ax_pd = fig.add_subplot(gs_a[1, col])
        _draw_hex(ax_pd, pos_input, vals_pd, HEX_XLIM, HEX_YLIM)
        axes_hex_mid.append(ax_pd)

        ax_rs = fig.add_subplot(gs_a[2, col])
        _draw_hex(ax_rs, pos_input, vals_rs, HEX_XLIM, HEX_YLIM)
        axes_hex_res.append(ax_rs)

    axes_hex_top[0].text(0.35, 1.28, 'ground truth visual stimulus',
                         transform=axes_hex_top[0].transAxes,
                         va='bottom', ha='left', fontsize=FS_LABEL)
    axes_hex_mid[0].text(0.35, 1.10, 'learned visual stimulus',
                         transform=axes_hex_mid[0].transAxes,
                         va='bottom', ha='left', fontsize=FS_LABEL)
    axes_hex_res[0].text(0.35, 1.10, 'residual (learned $-$ ground truth)',
                         transform=axes_hex_res[0].transAxes,
                         va='bottom', ha='left', fontsize=FS_LABEL)

    fig.canvas.draw()
    _norm = _mcolors.Normalize(vmin=HEX_VMIN, vmax=HEX_VMAX)
    _sm = _mcm.ScalarMappable(norm=_norm, cmap=CMAP)
    _top_pos = axes_hex_top[-1].get_position()
    _res_pos = axes_hex_res[-1].get_position()
    _cbar_h = (_top_pos.y1 - _res_pos.y0) * 0.45
    _cbar_y0 = (_top_pos.y1 + _res_pos.y0) / 2.0 - _cbar_h / 2.0
    _cax = fig.add_axes([
        _top_pos.x1 + 0.010,
        _cbar_y0,
        0.008,
        _cbar_h,
    ])
    _cbar = fig.colorbar(_sm, cax=_cax)
    _cbar.set_label('voltage (z-score)', fontsize=FS_LABEL)
    _cbar.ax.tick_params(labelsize=FS_TICK)
    _cbar.outline.set_linewidth(0.5)

    # (b/c) Two trace panels — stimulus | voltage vs noisy.
    gs_bc = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                         wspace=0.20)
    ax_b = fig.add_subplot(gs_bc[0, 0])
    ax_c = fig.add_subplot(gs_bc[0, 1])
    draw_trace_panel(ax_b, true_s, pred_s, labels_s, step_v_stim, time_ms,
                     pearson_r=r_stim,
                     header_label='stimulus, INR vs gt',
                     show_xlabel=True, show_type_labels=True)
    draw_trace_panel(ax_c, true_v, pred_v, labels_v, step_v_volt, time_ms,
                     pearson_r=r_volt,
                     header_label='voltage, GNN vs noisy gt',
                     show_xlabel=True, show_type_labels=True)

    # (d/e/f) Three scatter panels at the bottom.
    gs_def = mgs.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2],
                                          wspace=0.40)
    ax_d = fig.add_subplot(gs_def[0, 0])
    ax_e = fig.add_subplot(gs_def[0, 1])
    ax_f = fig.add_subplot(gs_def[0, 2])

    draw_scatter(
        ax_d,
        stim_in_true.ravel(), stim_in_pred.ravel(),
        lo=SCATTER_LO_S, hi=SCATTER_HI_S,
        xlabel='true stimulus', ylabel='learned stimulus',
        title=None,
    )
    draw_scatter(
        ax_e,
        activity_true, activity_pred,
        lo=SCATTER_LO_V, hi=SCATTER_HI_V,
        xlabel='noisy gt voltage', ylabel='rollout voltage',
        title='vs noisy',
    )
    draw_scatter(
        ax_f,
        nf_bundle['true'], nf_bundle['pred'],
        lo=SCATTER_LO_V, hi=SCATTER_HI_V,
        xlabel='noise-free gt voltage', ylabel='rollout voltage',
        title='vs noise-free',
    )

    # Panel labels a..f
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    anchors = [(axes_hex_top[0], 'a'),
               (ax_b, 'b'), (ax_c, 'c'),
               (ax_d, 'd'), (ax_e, 'e'), (ax_f, 'f')]
    for ax_anchor, lbl in anchors:
        bb = ax_anchor.get_tightbbox(renderer)
        x0, y1 = inv.transform((bb.x0, bb.y1))
        fig.text(x0, y1, lbl, fontsize=PANEL_LBL, fontweight='bold',
                 va='bottom', ha='left', color='black',
                 transform=fig.transFigure)

    out_base = os.path.join(_SCRIPT_DIR, 'fig_stim_rollout_inr')
    fig.savefig(out_base + '.pdf', bbox_inches='tight')
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_base}.pdf')
    print(f'Saved: {out_base}.png')


if __name__ == '__main__':
    main()
