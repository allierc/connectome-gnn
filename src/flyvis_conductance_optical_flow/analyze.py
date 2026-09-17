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

  STABILITY.txt              conductance runs only: the forward-Euler
                             amplification factor (dt/tau_i)(1 + G_i) of every
                             neuron on that same rollout, where G_i is its total
                             synaptic conductance in units of its own leak. Not a
                             survival test -- the model integrates with
                             exponential Euler and is stable at any value -- but a
                             measure of how much of the fitted behaviour is
                             shunting, and of how much of the run the forward
                             Euler that killed flow/2000/000 would have lost.

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


def evaluate_checkpoints(nv, task, device, every: int = 1) -> dict:
    """EPE and l2norm of every checkpoint on the held-out split.

    Returns `{indices, epe, l2norm, best_index, best_epe}`. Both metrics come
    from flyvis's own objectives, and the loop is the one
    `MultiTaskSolver.test` uses -- steady state, then the network, then the
    decoder -- so these numbers are comparable to the run's own validation.
    """
    from flyvis.task.objectives import epe as epe_fn
    from flyvis.task.objectives import l2norm as l2_fn

    indices = list(nv.checkpoints.indices)[::every]
    rows = {"index": [], "epe": [], "l2norm": []}

    # THE ZERO-PREDICTION BASELINE, and it is the only number that answers
    # "did this learn anything". The l2norm of this task barely moves -- a
    # trained ensemble sits near 1150 against an untrained 1300 -- because most
    # of it is the intrinsic spread of the flow field rather than anything the
    # network controls. Predicting a flow of zero everywhere gives the error of
    # having learned nothing at all, so a model is only doing something if its
    # EPE is below this.
    with torch.no_grad(), task.dataset.augmentation(False):
        z_e, z_l = [], []
        for data in task.val_data:
            y = data["flow"]
            zero = torch.zeros_like(y)
            z_e.append(float(epe_fn(zero, y)))
            z_l.append(float(l2_fn(zero, y)))
    rows["zero_epe"] = float(np.mean(z_e))
    rows["zero_l2norm"] = float(np.mean(z_l))
    print(f"  zero-prediction baseline: epe {rows['zero_epe']:.4f}  "
          f"l2norm {rows['zero_l2norm']:.2f}")

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


def write_summary_png(nv, rows, out_path) -> None:
    """Training, validation and EPE against iteration, in one figure.

    flyvis STORES all of this and plots none of it. Per checkpoint it writes four
    losses -- `validation/`, `validation_batch/`, `training/`, `training_batch/`
    -- and per iteration it writes `loss.h5`; `chkpt_iter.h5` maps a checkpoint
    index to the iteration it was taken at, which is what puts the two on one
    x-axis.

    Three panels, because the quantities do not share a scale: the per-iteration
    training loss is noisy and huge, the per-checkpoint losses are smooth, and
    EPE is the one that answers whether the model learned anything -- so it gets
    its own panel with the zero-prediction baseline drawn across it.
    """
    import h5py
    import matplotlib.pyplot as plt

    d = nv.dir.path

    def read(rel):
        """The array at `<run>/<rel>`, or None.

        A PUBLISHED run has none of the training history a locally trained one
        does -- one checkpoint, no per-iteration loss, and what it does store can
        be a scalar rather than a series. Everything downstream therefore has to
        treat every one of these as optional; returning None for an unsized
        object is what lets one code path plot both.
        """
        p = d / rel
        if not p.exists():
            return None
        try:
            with h5py.File(p, "r") as f:
                a = np.asarray(f["data"])
            return a if a.ndim >= 1 and a.size else None
        except Exception:
            return None

    iters = read("chkpt_iter.h5")
    train_iter = read("loss.h5")
    curves = {
        "validation": read("validation/loss.h5"),
        "training": read("training/loss.h5"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    ax = axes[0]
    if train_iter is not None:
        # SMOOTHED OVER A HUNDREDTH OF THE RUN, not over an epoch. The
        # per-iteration loss swings between about 750 and 1850 from batch to
        # batch -- the batch is 4 sequences and their intrinsic flow magnitudes
        # differ more than anything training changes -- so a 12-iteration mean is
        # still a solid band with no visible trend. The raw trace is kept faintly
        # behind it so the spread is not hidden.
        ax.plot(np.arange(train_iter.size), train_iter, lw=0.3, color="0.85")
        w = max(50, train_iter.size // 100)
        smooth = np.convolve(train_iter, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(smooth.size) + w // 2, smooth, lw=1.4, color="0.2",
                label=f"mean of {w} iterations")
        ax.legend(frameon=False)
    ax.set_xlabel("iteration")
    ax.set_ylabel("training loss (l2norm)")
    ax.text(0, 1.02, "training loss, per iteration", transform=ax.transAxes)

    ax = axes[1]
    for name, y in curves.items():
        if y is None:
            continue
        x = iters[: len(y)] if iters is not None else np.arange(len(y))
        ax.plot(x, y, marker="o", ms=3, lw=1.2, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (l2norm)")
    ax.legend(frameon=False)
    ax.text(0, 1.02, "loss per checkpoint", transform=ax.transAxes)

    ax = axes[2]
    idx = np.asarray(rows["index"])
    x = iters[idx] if (iters is not None and idx.max() < len(iters)) else idx
    ax.plot(x, rows["epe"], marker="o", ms=3, lw=1.2, color="k", label="held-out EPE")
    ax.axhline(rows["zero_epe"], ls="--", lw=1.0, color="0.5",
               label=f"zero prediction ({rows['zero_epe']:.3f})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("EPE")
    ax.legend(frameon=False)
    ax.text(0, 1.02, "endpoint error, the test that it learned", transform=ax.transAxes)

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


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


def write_movie(lum, target, prediction, path, panel_cm=9.0, dpi=200) -> None:
    """input | target | prediction, by flyvis's own Sintel animation.

    `SintelSample` is exactly this three-panel figure and already knows the hex
    lattice, so there is no renderer to write here -- only the call.
    """
    import shutil
    import sys

    from flyvis.analysis.animations.sintel import SintelSample

    # FFMPEG HAS TO BE ON PATH, because `Animation.convert` shells out to it.
    # Running the interpreter by absolute path -- which is what a bsub command
    # string does -- leaves the env's own bin/ off PATH, so the binary sitting
    # right beside python is invisible and the movie step dies after the frames
    # have already been rendered. Look there first, then at imageio-ffmpeg's
    # bundled binary.
    if shutil.which("ffmpeg") is None:
        env_bin = os.path.dirname(sys.executable)
        candidates = [env_bin]
        try:
            import imageio_ffmpeg

            candidates.append(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()))
        except Exception:
            pass
        for d in candidates:
            if os.path.exists(os.path.join(d, "ffmpeg")):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                break
        else:
            raise SystemExit(
                "no ffmpeg on PATH and none beside the interpreter; the frames "
                "render but cannot be assembled. Pass --no-movie, or put one on PATH."
            )

    # RESOLUTION COMES FROM TWO PLACES, and both of SintelSample's defaults are
    # small: panels 3.6 cm wide capped at an 18 cm figure, rendered at 100 dpi,
    # which is the 362x150 mp4 the defaults produce. The panel sizes set the
    # figure's shape and the dpi sets how many pixels that shape becomes, so
    # raising one alone either gives a big blurry frame or a small sharp one.
    anim = SintelSample(
        lum, target, prediction=prediction,
        panel_width_cm=panel_cm,
        panel_height_cm=panel_cm,
        max_figure_width_cm=3 * panel_cm + 2,
        max_figure_height_cm=panel_cm + 2,
        fontsize=9,
    )
    anim.to_vid(
        os.path.basename(path).replace(".mp4", ""),
        dest_path=os.path.dirname(path),
        type="mp4",
        framerate=10,
        dpi=dpi,
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
    network = nv.init_network(checkpoint=checkpoint)
    dynamics = network.dynamics
    if not hasattr(dynamics, "reversal_per_edge"):
        return None  # current-based run: no reversals to report

    # Imported AFTER that check: connectome_gnn is only needed for the reversal
    # figure, and a current-based run should not fail on it.
    from connectome_gnn.plot import INDEX_TO_NAME, _write_reversal_report
    from flyvis.utils.type_utils import byte_to_str

    # connectome_gnn publishes the index -> name direction only; the reversal
    # report wants a type id per neuron, so invert it here rather than keeping a
    # second table that could disagree with the first.
    name_to_index = {v: k for k, v in INDEX_TO_NAME.items()}

    network.clamp()
    params = network._param_api()
    E_exc = params.nodes.E_exc.detach().cpu().numpy().astype(float)
    E_inh = params.nodes.E_inh.detach().cpu().numpy().astype(float)

    names = byte_to_str(network.connectome.nodes.type[:])
    types = np.array([name_to_index.get(str(t), -1) for t in names], dtype=int)

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


def stability_report(network, activity, dt, out_dir):
    """`STABILITY.txt`: how stiff the fitted network is, on the rollout it just ran.

    THE NUMBER IS THE FORWARD-EULER AMPLIFICATION FACTOR

        f_i(t) = (dt / max(tau_i, dt)) * (1 + G_i(t)),
        G_i(t) = sum_j g_ij relu(v_j(t))

    with dt the integration step in seconds (20 ms here), tau_i the membrane time
    constant of neuron i in seconds and G_i its total synaptic conductance in units
    of its own leak conductance. A forward-Euler step contracts only while f_i < 2;
    the model does not use forward Euler any more (see dynamics.py), so this is no
    longer a survival test but a measure of HOW FAR INTO THE STIFF REGIME the fit
    has gone -- that is, how much of the network's behaviour is shunting rather than
    plain summation, and how much of the run the old integrator would have lost.

    `activity` is the (..., n_nodes) rollout the movie and the reversal figure are
    drawn from, so all three describe one pass of the network.

    Returns None for a current-based run, which has no conductance to sum.
    """
    # DISPATCH ON THE DYNAMICS, not on the parameters. `params.edges` is an
    # AutoDeref, whose __getattr__ forwards to __getitem__ and raises KeyError --
    # which `hasattr` does not catch, so asking it for a key a current-based run
    # has never written propagates the KeyError instead of answering False. This is
    # the same test `reversal_report` uses, for the same reason.
    if not hasattr(network.dynamics, "reversal_per_edge"):
        return None

    params = network._param_api()
    dev = params.nodes.time_const.device
    g = params.edges.conductance.detach().reshape(-1)
    src = network._source_indices.to(dev).long()
    dst = network._target_indices.to(dev).long()
    tau = torch.clamp(params.nodes.time_const.detach().reshape(-1), min=dt)
    ratio = dt / tau
    n_nodes = tau.numel()

    v = torch.as_tensor(activity).to(dev).reshape(-1, n_nodes)
    worst = torch.zeros(n_nodes, device=dev)
    # FRAME BY FRAME, keeping only the per-neuron MAXIMUM over the rollout: the
    # per-edge product is 1,513,231 numbers and there are a few hundred frames, so
    # materialising all of them at once is a gigabyte for a statistic that is one
    # max away.
    for k in range(v.shape[0]):
        r = torch.relu(v[k])[src]
        G = torch.zeros(n_nodes, device=dev).index_add_(0, dst, g * r)
        torch.maximum(worst, ratio * (1.0 + G), out=worst)

    w = worst.cpu().numpy()
    report = dict(
        dt=float(dt),
        n_frames=int(v.shape[0]),
        tau_min=float(tau.min()),
        factor_mean=float(w.mean()),
        factor_p99=float(np.percentile(w, 99)),
        factor_max=float(w.max()),
        n_unstable=int((w > 2.0).sum()),
        n_nodes=int(n_nodes),
    )
    with open(os.path.join(out_dir, "STABILITY.txt"), "w") as f:
        f.write("forward-Euler amplification factor (dt/tau_i)(1 + G_i), per neuron, "
                "maximised over the rollout.\n")
        f.write("A forward-Euler step contracts only while this is below 2; "
                "exponential Euler is stable at any value.\n")
        for k, val in report.items():
            f.write(f"{k}: {val}\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--network-name", required=True, help="e.g. flow/2000/000")
    p.add_argument("--checkpoint", type=int, default=None,
                   help="which checkpoint to roll out; default is argmin EPE. "
                   "Must be an int: NetworkView indexes its checkpoint list with "
                   "it, so a string fails deep inside get_checkpoint")
    p.add_argument("--no-movie", action="store_true")
    p.add_argument("--dpi", type=int, default=200,
                   help="movie resolution; SintelSample defaults to 100")
    p.add_argument("--panel-cm", type=float, default=9.0,
                   help="size of each of the three panels, in cm; "
                   "SintelSample defaults to 3.6 wide by 3 high")
    p.add_argument(
        "--every", type=int, default=1,
        help="score every Nth checkpoint; the whole set is the default",
    )
    p.add_argument(
        "--no-epe-file", action="store_true",
        help="skip writing validation/epe.h5. Use while the run is still "
        "training, so the analysis does not write into a directory the trainer "
        "is also writing",
    )
    p.add_argument(
        "--no-stability", action="store_true",
        help="skip STABILITY.txt. It reuses the rollout that is computed anyway "
        "and costs a scatter per frame, so it is on by default for a conductance "
        "run -- it is the quantity that killed flow/2000/000",
    )
    p.add_argument(
        "--original-split", action="store_true",
        help="force task.original_split=True when building the dataloaders. "
        "THE PUBLISHED MODELS NEED THIS: their stored config predates the key, "
        "so building a Task from it falls back to get_random_data_split, a "
        "different algorithm that shares only 6 of 16 validation sequences with "
        "the split they actually trained on -- ten of the original split's "
        "validation sequences are TRAINING data under it. Comparing a published "
        "model scored on that against ours scored on the real split is not a "
        "comparison.",
    )
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

    task_config = dict(nv.dir.config.task)
    # `type: Task` is stored by the PUBLISHED models' config and is not a
    # constructor argument; a run trained by this repo has no such key. Dropped
    # rather than special-cased on the run, so one code path reads both.
    task_config.pop("type", None)
    # `task_weight` is the old spelling of `task_weights`, renamed since the
    # published models were exported. Both are null on a single-task config, so
    # the rename is all there is to it.
    if "task_weight" in task_config:
        task_config["task_weights"] = task_config.pop("task_weight")
    if args.original_split:
        task_config["original_split"] = True
    task = Task(**task_config)
    print(f"\033[96m{args.network_name}\033[0m")
    print(f"  dynamics        {nv.dir.config.network.dynamics.type}")
    print(f"  checkpoints     {len(nv.checkpoints.indices)}")
    print(f"  analysis ->     {out_dir}")

    rows = evaluate_checkpoints(nv, task, device, every=args.every)
    if not args.no_epe_file:
        write_epe_file(nv, rows)
    chosen = args.checkpoint if args.checkpoint is not None else rows["best_index"]
    gain = 100.0 * (rows["zero_epe"] - rows["best_epe"]) / rows["zero_epe"]
    print(f"\033[92mbest by EPE: checkpoint {rows['best_index']} at "
          f"{rows['best_epe']:.4f}, {gain:.1f}% below the zero-prediction "
          f"baseline of {rows['zero_epe']:.4f}\033[0m")

    import csv as _csv

    with open(out_dir / "checkpoints.csv", "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["index", "epe", "l2norm"])
        for i, e, l in zip(rows["index"], rows["epe"], rows["l2norm"]):
            w.writerow([i, f"{e:.6f}", f"{l:.6f}"])

    write_summary_png(nv, rows, str(out_dir / "summary.png"))
    print(f"  summary         {out_dir / 'summary.png'}")

    lum, target, prediction = rollout(nv, task, chosen)
    print(f"  rollout         lum {lum.shape}, flow {target.shape}")
    if not args.no_movie:
        write_movie(lum, target, prediction, str(out_dir / "rollout.mp4"),
                    panel_cm=args.panel_cm, dpi=args.dpi)
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

    if not args.no_stability:
        stab = stability_report(network, act.detach(), float(task.dataset.dt),
                                str(out_dir))
        if stab is None:
            print("  stability       (current-based run: no conductance to sum)")
        else:
            print(f"  stability       forward-Euler factor (dt/tau)(1+G): max "
                  f"{stab['factor_max']:.2f}, p99 {stab['factor_p99']:.2f}, "
                  f"{stab['n_unstable']} of {stab['n_nodes']} neurons above 2 "
                  f"(where forward Euler would have diverged)")

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
