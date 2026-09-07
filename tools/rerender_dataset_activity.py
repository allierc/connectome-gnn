"""Re-render <dataset>/activity.png and drop <dataset>/svd_analysis.png in place.

WHY THIS EXISTS. Both changes are plot-only: activity.png moved from a
whole-run trace plot (64,000 frames on one axis, which draws every neuron as a
solid band) to the nominal trace figure the trainer writes into
tmp_training/traces/rollout_*.png, and svd_analysis.png was dropped entirely.
Regenerating the datasets to pick that up would re-run the simulator over 6 GB
of zarr per dataset for two pictures. This reads the already-written
x_list_train/ instead and rewrites just the figure, so the voltages on disk are
untouched.

It calls the SAME renderer as the trainer -- connectome_gnn.models.teacher_eval
.save_trace_figure -- with pred=None and r=None, because a generated dataset has
ground truth only: no student rollout to overlay and no rollout-vs-truth
correlation to annotate.

Usage:
    python tools/rerender_dataset_activity.py <dataset_dir> [<dataset_dir> ...]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from connectome_gnn.metrics import INDEX_TO_NAME
from connectome_gnn.models.teacher_eval import save_trace_figure
from connectome_gnn.zarr_io import load_simulation_data


# Same window as the generator's ACTIVITY_TRACE_FRAMES: 1,000 frames, i.e. 20 s
# of simulated time at the flyvis delta_t of 20 ms.
TRACE_FRAMES = 1000


def rerender(dataset_dir, delta_t=0.02, n_frames=TRACE_FRAMES):
    """Rewrite activity.png from x_list_train/ and delete svd_analysis.png."""
    x_train = os.path.join(dataset_dir, 'x_list_train')
    if not os.path.isdir(x_train):
        print(f'  SKIP {dataset_dir}: no x_list_train/')
        return False

    x_ts = load_simulation_data(x_train)
    voltage = x_ts.voltage.numpy()                       # (T, N)
    n = int(min(n_frames, voltage.shape[0]))
    stim = x_ts.stimulus[:n, 0].numpy() if x_ts.stimulus is not None else None
    type_list = x_ts.neuron_type.numpy() if x_ts.neuron_type is not None else None

    out = os.path.join(dataset_dir, 'activity.png')
    save_trace_figure(out, voltage[:n], None, stim, delta_t, None,
                      type_names=INDEX_TO_NAME, type_list=type_list)
    print(f'  wrote {out}  ({n} frames = {n * delta_t:.0f} s of simulated time, '
          f'{voltage.shape[1]} neurons available)')

    svd = os.path.join(dataset_dir, 'svd_analysis.png')
    if os.path.isfile(svd):
        os.remove(svd)
        print(f'  removed {svd}')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset_dirs', nargs='+', help='dataset directories under graphs_data/')
    ap.add_argument('--delta_t', type=float, default=0.02,
                    help='seconds per simulation frame; 0.02 s (20 ms) for flyvis')
    ap.add_argument('--n_frames', type=int, default=TRACE_FRAMES,
                    help='length of the plotted window, in frames')
    args = ap.parse_args()

    for d in args.dataset_dirs:
        print(d)
        rerender(d.rstrip('/'), delta_t=args.delta_t, n_frames=args.n_frames)


if __name__ == '__main__':
    main()
