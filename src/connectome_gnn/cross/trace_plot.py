"""Cheap post-generation trace summary plot for hold-out CV datasets.

Produces a panel-A-style stacked-voltage figure (mirroring the trace row
of figures/fig_rollout_3col_noise_comparison.py) from the already-written
voltage.zarr / stimulus.zarr, without re-running the simulator and
without touching data_generate's visualize / mp4 / kinograph paths.

Output: <dataset_dir>/traces.png  (~50 KB, <1 s per dataset).
Style: green ground-truth voltage (#2ca02c) + red stimulus overlay
(#cf222e) on neurons with non-trivial visual input. Type labels come
from connectome_gnn.metrics.INDEX_TO_NAME.
"""

import os

import numpy as np


# Mirrors figures/fig_rollout_3col_noise_comparison.py:139.
# 12 representative flyvis cell types spanning lamina/medulla/lobula:
#   23=R1, 5=L1, 6=L2, 7=L3, 12=Mi1, 22=Mi9, 43=Tm1, 55=Tm9,
#   35=T4a, 39=T5a, 31=T1, 0=Am
SELECTED_TYPES = [23, 5, 6, 7, 12, 22, 43, 55, 35, 39, 31, 0]

# Time window (frames) used for the trace snapshot. Mirrors
# fig_rollout_3col_noise_comparison.py:137-138.
TRACE_START = 500
TRACE_END   = 1500
DT_MS       = 20.0

# Style mirrors fig_rollout_3col_noise_comparison.py:136-139.
COLOR_GT   = 'black'     # neural activity (voltage) — was '#2ca02c' green
COLOR_STIM = '#cf222e'   # red
LW_GT      = 1.0
LW_STIM    = 0.6


def _open_zarr_array(path):
    """Open a zarr array on disk; return None if missing."""
    if not os.path.isdir(path):
        return None
    try:
        import zarr
        return zarr.open(path, mode='r')
    except Exception:
        return None


def _pick_one_per_type(neuron_type, selected_types):
    """For each type in selected_types, pick the first matching neuron index.
    Skips types absent from this dataset (e.g. full_eye drops some types).

    Fallback for typeless datasets (e.g. cortex teacher voltage, where
    every unit has neuron_type=0): when there's only a single unique type
    value, the per-type loop would collapse to one pick — instead return
    up to 12 evenly-spaced neuron indices so the trace plot still
    produces a useful stacked-voltage figure.
    """
    N = int(neuron_type.shape[0])
    if N == 0:
        return [], []
    if int(np.unique(neuron_type).size) <= 1:
        n_show = min(12, N)
        picks = [int(round(i * (N - 1) / max(1, n_show - 1)))
                 for i in range(n_show)]
        type_ids = list(range(n_show))  # placeholder type ids
        return picks, type_ids

    picks, type_ids = [], []
    for t in selected_types:
        ids = np.where(neuron_type == t)[0]
        if ids.size:
            picks.append(int(ids[0]))
            type_ids.append(t)
    return picks, type_ids


def save_trace_plot(dataset_dir, force=False):
    """Render <dataset_dir>/traces.png from <dataset_dir>/x_list_train/voltage.zarr.

    Idempotent: returns early if traces.png already exists unless force=True.
    Silent on missing inputs — callers run this best-effort.
    """
    out_path = os.path.join(dataset_dir, 'traces.png')
    if os.path.isfile(out_path) and not force:
        return False

    x_list_train_dir = os.path.join(dataset_dir, 'x_list_train')
    if not os.path.isdir(x_list_train_dir):
        return False

    voltage = _open_zarr_array(os.path.join(x_list_train_dir, 'voltage.zarr'))
    if voltage is None:
        return False
    nt_arr = _open_zarr_array(os.path.join(x_list_train_dir, 'neuron_type.zarr'))
    if nt_arr is None:
        return False
    neuron_type = np.asarray(nt_arr)
    stimulus = _open_zarr_array(os.path.join(x_list_train_dir, 'stimulus.zarr'))

    n_frames = int(voltage.shape[0])
    end = min(TRACE_END, n_frames)
    start = min(TRACE_START, max(0, end - 100))
    if end - start < 10:
        return False

    picks, type_ids = _pick_one_per_type(neuron_type, SELECTED_TYPES)
    if not picks:
        return False

    # Resolve human-readable type names; fall back to type{N} when the
    # canonical lookup is missing (e.g. non-flyvis dataset). For typeless
    # datasets (cortex teacher voltage), label by unit index instead.
    try:
        from connectome_gnn.metrics import INDEX_TO_NAME
    except Exception:
        INDEX_TO_NAME = {}
    if int(np.unique(neuron_type).size) <= 1:
        labels = [f'unit {i}' for i in picks]
    else:
        labels = [INDEX_TO_NAME.get(t, f'type{t}') for t in type_ids]

    # Read (n_frames, n_neurons)[start:end, picks] -> (n_picks, window).
    volt_w = np.asarray(voltage[start:end, picks], dtype=np.float32).T
    stim_w = (np.asarray(stimulus[start:end, picks], dtype=np.float32).T
              if stimulus is not None else None)
    time_ms = np.arange(end - start) * DT_MS + start * DT_MS

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sigma = float(np.std(volt_w)) or 1.0
    step_v = 3.0 * sigma

    fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(picks) + 0.6))
    for i, lbl in enumerate(labels):
        bl = float(np.mean(volt_w[i]))
        # Green ground-truth voltage trace.
        ax.plot(time_ms, (volt_w[i] - bl) + i * step_v,
                color=COLOR_GT, lw=LW_GT, alpha=0.95, zorder=2)
        # Red stimulus trace — only for neurons with non-trivial visual input
        # (in our 12 selected types only R1 has stim). Plotted slightly BELOW
        # the voltage trace so it's visually distinct.
        if stim_w is not None and stim_w[i].std() > 1e-6:
            stim_bl = float(np.mean(stim_w[i]))
            ax.plot(time_ms, (stim_w[i] - stim_bl) + i * step_v - 0.4 * step_v,
                    color=COLOR_STIM, lw=LW_STIM, alpha=0.95, zorder=3)
    ax.set_yticks([i * step_v for i in range(len(labels))])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('time (ms)', fontsize=8)
    ax.set_ylabel('neurons', fontsize=8)
    ax.set_title(os.path.basename(dataset_dir.rstrip('/')), fontsize=8)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return True
