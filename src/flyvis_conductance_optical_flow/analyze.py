"""Rollout, checkpoint selection, movie and reversal report for a trained run.

FOUR THINGS, ALL WRITTEN INTO `<run_dir>/analysis/`:

  epe.h5 + checkpoints.csv   the endpoint error of EVERY checkpoint, and which
                             one argmin picks. REPRODUCING.md is explicit that
                             the published models were selected on EPE and that
                             selecting on the training loss instead costs about
                             0.033 EPE on the ensemble mean -- and that the cost
                             is right-skewed, so it cannot be estimated from a
                             few members. Nothing in flyvis's training path
                             writes an EPE validation file; writing one here is
                             what makes NetworkView's own default
                             (`loss_file_name="epe"`) resolve natively instead of
                             silently falling back to the l2norm it stores.

  rollout.mp4                input | ground-truth flow | predicted flow, from
                             flyvis's own `SintelSample`, on held-out sequences.

  summary.png                EPE and l2norm against checkpoint.

  reversals.png + .csv       conductance runs only: the learned reversal of every
                             postsynaptic cell type against the voltages that
                             type actually visits. Drawn by connectome_gnn's
                             `_write_reversal_report`, the SAME function that
                             draws the teacher-student twin's figure, so a flow
                             run and a twin are read on one pair of axes rather
                             than two that merely look alike.

  BRACKET_CROSSINGS.txt      how far, and for which cell types, the driving force
                             (E_inh - v_i) changes sign. On a generated dataset
                             that check can abort generation; here there is no
                             generator to abort, so it is a diagnostic -- and the
                             band-saturation count beside it says how many rows
                             were placed by their bound rather than by the task.

Usage:
    python FlyvisFlow_Main.py -o analyze conductance_2000_000
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Held-out sequences rendered into the movie. The whole validation split is 16
# sequences of 49 frames; three is enough to see whether the prediction tracks
# the target and short enough that the mp4 stays a thing people actually open.
MOVIE_SEQUENCES = 3


def _run_dir(network_name: str):
    import flyvis

    return flyvis.results_dir / network_name


def evaluate_checkpoints(nv, task, device) -> dict:
    """EPE and l2norm of every checkpoint on the held-out split.

    Returns `{indices, epe, l2norm, best_index, best_epe}`. Both metrics come
    from flyvis's own objectives, and the loop is the one
    `MultiTaskSolver.test` uses -- steady state, then the network, then the
    decoder -- so these numbers are comparable to the run's own validation.
    """
    from flyvis.task.objectives import epe as epe_fn
    from flyvis.task.objectives import l2norm as l2_fn

    indices = list(nv.checkpoints.indices)
    rows = {"index": [], "epe": [], "l2norm": []}

    for idx in indices:
        network = nv.init_network(checkpoint=idx)
        decoder = nv.init_decoder(checkpoint=idx)["flow"]
        network.eval()
        decoder.eval()

        e_acc, l_acc = [], []
        with torch.no_grad():
            with task.dataset.augmentation(False):
                initial_state = network.steady_state(
                    t_pre=0.25, dt=task.dataset.dt,
                    batch_size=task.val_data.batch_size, value=0.5,
                )
                for data in task.val_data:
                    lum = data["lum"]
                    n_samples, n_frames = lum.shape[0], lum.shape[1]
                    network.stimulus.zero(n_samples, n_frames)
                    network.stimulus.add_input(lum)
                    activity = network(network.stimulus(), task.dataset.dt,
                                       state=initial_state)
                    y_est = decoder(activity)
                    y = data["flow"].to(y_est.device)
                    e_acc.append(float(epe_fn(y_est, y)))
                    l_acc.append(float(l2_fn(y_est, y)))

        rows["index"].append(int(idx))
        rows["epe"].append(float(np.mean(e_acc)))
        rows["l2norm"].append(float(np.mean(l_acc)))
        print(f"  chkpt {idx:>5}  epe {rows['epe'][-1]:8.4f}  "
              f"l2norm {rows['l2norm'][-1]:10.2f}")

    best = int(np.argmin(rows["epe"]))
    rows["best_index"] = rows["index"][best]
    rows["best_epe"] = rows["epe"][best]
    return rows


def write_epe_file(nv, rows) -> None:
    """`validation/epe.h5`, so NetworkView's default selection finds it.

    `best_checkpoint_default_fn` looks for `loss_file_name="epe"` and, finding
    nothing, falls back to the stored l2norm without saying so. Writing the file
    is the difference between "selected on EPE" and "believed to be".
    """
    import h5py

    out = nv.dir.path / "validation"
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(out / "epe.h5", "w") as f:
        f.create_dataset("data", data=np.asarray(rows["epe"], dtype=np.float32))


def rollout(nv, task, checkpoint, n_sequences=MOVIE_SEQUENCES):
    """Held-out sequences through the network and decoder.

    Returns (lum, target, prediction) as numpy, shaped for `SintelSample`:
    lum (n, frames, hexals), target/prediction (n, frames, 2, hexals).
    """
    network = nv.init_network(checkpoint=checkpoint)
    decoder = nv.init_decoder(checkpoint=checkpoint)["flow"]
    network.eval()
    decoder.eval()

    lums, targets, preds = [], [], []
    with torch.no_grad():
        with task.dataset.augmentation(False):
            for data in task.val_data:
                lum = data["lum"]
                n_samples, n_frames = lum.shape[0], lum.shape[1]
                state = network.steady_state(
                    t_pre=0.25, dt=task.dataset.dt, batch_size=n_samples, value=0.5
                )
                network.stimulus.zero(n_samples, n_frames)
                network.stimulus.add_input(lum)
                activity = network(network.stimulus(), task.dataset.dt, state=state)
                y_est = decoder(activity)
                lums.append(lum.detach().cpu().numpy()[:, :, 0])
                targets.append(data["flow"].detach().cpu().numpy())
                preds.append(y_est.detach().cpu().numpy())
                if sum(x.shape[0] for x in lums) >= n_sequences:
                    break

    cat = lambda xs: np.concatenate(xs, axis=0)[:n_sequences]  # noqa: E731
    return cat(lums), cat(targets), cat(preds)


def write_movie(lum, target, prediction, path) -> None:
    """input | target | prediction, by flyvis's own Sintel animation.

    `SintelSample` is exactly this three-panel figure and already knows the hex
    lattice, so there is no renderer to write here -- only the call.
    """
    from flyvis.analysis.animations.sintel import SintelSample

    anim = SintelSample(lum, target, prediction=prediction)
    anim.to_vid(
        os.path.basename(path).replace(".mp4", ""),
        dest_path=os.path.dirname(path),
        type="mp4",
        framerate=10,
        delete_if_exists=True,
    )


def reversal_report(nv, checkpoint, activity_lo, activity_hi, out_dir):
    """`reversals.png`/`.csv` and the crossing record, for a conductance run.

    Reuses connectome_gnn's `_write_reversal_report`, the function that draws the
    teacher-student twin's figure, so the two are read on one set of axes. It
    wants PER-NEURON arrays, which is what `params.nodes.E_exc/E_inh` already
    are -- the dynamics materialises them at node level once per forward pass.

    Returns the dict written to BRACKET_CROSSINGS.txt.
    """
    from connectome_gnn.plot import NAME_TO_INDEX, _write_reversal_report
    from flyvis.utils.type_utils import byte_to_str

    network = nv.init_network(checkpoint=checkpoint)
    dynamics = network.dynamics
    if not hasattr(dynamics, "reversal_per_edge"):
        return None  # current-based run: no reversals to report

    network.clamp()
    params = network._param_api()
    E_exc = params.nodes.E_exc.detach().cpu().numpy().astype(float)
    E_inh = params.nodes.E_inh.detach().cpu().numpy().astype(float)

    names = byte_to_str(network.connectome.nodes.type[:])
    types = np.array([NAME_TO_INDEX.get(str(t), -1) for t in names], dtype=int)

    targeted = np.zeros(E_exc.size, dtype=bool)
    targeted[np.asarray(network.connectome.edges.target_index[:], dtype=int)] = True

    csv_path = _write_reversal_report(
        E_exc, E_inh, types, targeted,
        v_lo=activity_lo, v_hi=activity_hi,
        out_dir=out_dir, stem="reversals",
        note=f"checkpoint {checkpoint}",
    )

    # THE CROSSINGS, in the generator's own vocabulary so the two files diff.
    n_inh = int((activity_lo <= E_inh).sum())
    n_exc = int((activity_hi >= E_exc).sum())
    worst = float(np.minimum(activity_lo - E_inh, E_exc - activity_hi).min())
    # HOW MANY ROWS THE BAND PLACED RATHER THAN THE TASK. A row on its bound is a
    # constraint speaking; in the ion_sub twin 40 of 65 were, which is why the
    # count belongs beside the figure rather than in a footnote.
    sat = {}
    for row, band in (("E_exc_raw", dynamics.exc_band), ("E_inh_raw", dynamics.inh_band)):
        raw = dict(network.named_parameters())[f"nodes_{row}"].detach()
        s = torch.sigmoid(raw)
        sat[row] = dict(
            n_rows=int(s.numel()),
            pinned_low=int((s < 0.02).sum()),
            pinned_high=int((s > 0.98).sum()),
            band=list(band),
        )

    report = dict(n_exc=n_exc, n_inh=n_inh, worst_margin=worst,
                  E_exc=float(E_exc.mean()), E_inh_min=float(E_inh.min()),
                  E_inh_max=float(E_inh.max()), band_saturation=sat)
    with open(os.path.join(out_dir, "BRACKET_CROSSINGS.txt"), "w") as f:
        f.write(f"conductance bracket: {n_exc} above E_exc, {n_inh} below E_inh, "
                f"worst margin {worst:.4f}\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")
    return report, csv_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--network-name", required=True, help="e.g. flow/2000/000")
    p.add_argument("--checkpoint", default=None,
                   help="which checkpoint to roll out; default is argmin EPE")
    p.add_argument("--no-movie", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.verbose:
        logging.getLogger("flyvis").setLevel(logging.WARNING)

    import flyvis  # noqa: F401
    import flyvis_conductance_optical_flow  # noqa: F401  -- registers the classes
    from flyvis.network.network_view import NetworkView
    from flyvis.task.tasks import Task

    nv = NetworkView(args.network_name)
    out_dir = nv.dir.path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = Task(**nv.dir.config.task)
    print(f"\033[96m{args.network_name}\033[0m")
    print(f"  dynamics        {nv.dir.config.network.dynamics.type}")
    print(f"  checkpoints     {len(nv.checkpoints.indices)}")
    print(f"  analysis ->     {out_dir}")

    rows = evaluate_checkpoints(nv, task, device)
    write_epe_file(nv, rows)
    chosen = args.checkpoint if args.checkpoint is not None else rows["best_index"]
    print(f"\033[92mbest by EPE: checkpoint {rows['best_index']} "
          f"at {rows['best_epe']:.4f}\033[0m")

    import csv as _csv

    with open(out_dir / "checkpoints.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["index", "epe", "l2norm"])
        for i, e, l in zip(rows["index"], rows["epe"], rows["l2norm"]):
            w.writerow([i, f"{e:.6f}", f"{l:.6f}"])

    lum, target, prediction = rollout(nv, task, chosen)
    print(f"  rollout         lum {lum.shape}, flow {target.shape}")
    if not args.no_movie:
        write_movie(lum, target, prediction, str(out_dir / "rollout.mp4"))
        print(f"  movie           {out_dir / 'rollout.mp4'}")

    # The voltage extremes each neuron actually visits, for the reversal figure's
    # activity segments. Taken from the same rollout the movie shows, so the
    # figure and the movie describe one pass of the network rather than two.
    network = nv.init_network(checkpoint=chosen)
    with torch.no_grad(), task.dataset.augmentation(False):
        data = next(iter(task.val_data))
        n_samples, n_frames = data["lum"].shape[0], data["lum"].shape[1]
        state = network.steady_state(t_pre=0.25, dt=task.dataset.dt,
                                     batch_size=n_samples, value=0.5)
        network.stimulus.zero(n_samples, n_frames)
        network.stimulus.add_input(data["lum"])
        act = network(network.stimulus(), task.dataset.dt, state=state)
    a = act.detach().cpu().numpy().reshape(-1, act.shape[-1])
    v_lo, v_hi = a.min(axis=0).astype(float), a.max(axis=0).astype(float)

    out = reversal_report(nv, chosen, v_lo, v_hi, str(out_dir))
    if out is None:
        print("  reversals       (current-based run: none)")
    else:
        report, csv_path = out
        print(f"  reversals       {csv_path}")
        print(f"  crossings       {report['n_inh']} below E_inh, "
              f"{report['n_exc']} above E_exc, worst margin "
              f"{report['worst_margin']:.4f}")
        for row, s in report["band_saturation"].items():
            print(f"  {row:<15} {s['pinned_low'] + s['pinned_high']} of {s['n_rows']} "
                  f"rows pinned on band {s['band']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
