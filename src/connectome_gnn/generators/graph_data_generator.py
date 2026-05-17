import datetime
import fcntl
import glob
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch

# Optional imports (not available in flyvis-gnn spinoff)
try:
    from connectome_gnn.data_loaders import load_wormvae_data, load_zebrafish_data
except ImportError:
    load_wormvae_data = None
    load_zebrafish_data = None
from connectome_gnn.figure_style import dark_style, default_style
from connectome_gnn.generators.optogenetics import build_input_perturbation
from connectome_gnn.log import get_logger
from connectome_gnn.neuron_state import NeuronState
from connectome_gnn.plot import (
    plot_activity_traces,
    plot_connconstr_diagnostics,
    plot_hh_debug,
    plot_sequence_preview,
    plot_kinograph,
    plot_selected_neuron_traces,
    plot_spatial_activity_grid,
    plot_spiking_traces,
    plot_task_pi_traces,
)
# plot_task_cortex_* are imported lazily inside _generate_cortex_task so plot.py
# can be edited without affecting non-task code paths.
from connectome_gnn.zarr_io import ZarrArrayWriter, ZarrSimulationWriterV3


def _rmtree(path):
    """Remove a directory tree robustly on network filesystems (Lustre/GPFS).

    Python 3.12 changed shutil.rmtree on Linux to use _rmtree_safe_fd, which
    calls openat()/unlinkat() relative to a directory fd.  On Lustre/GPFS
    these syscalls can raise ENOTEMPTY on rmdir() even after all child files
    were successfully unlinked, because Lustre's metadata propagation lags
    behind the unlinkat() calls.  This is a known Python 3.12 + Lustre
    incompatibility, not a bug in our code.

    Workaround: use os.walk() bottom-up with plain path-based unlink()/rmdir(),
    which go through Lustre's regular code path and do not exhibit the race.
    """
    path = str(path)
    for root, dirs, files in os.walk(path, topdown=False):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            os.rmdir(os.path.join(root, d))
    os.rmdir(path)


try:
    from connectome_gnn.generators.davis import AugmentedVideoDataset, CombinedVideoDataset
except ImportError:
    AugmentedVideoDataset = None
    CombinedVideoDataset = None
import os

from tqdm import tqdm, trange

from connectome_gnn.generators.utils import (
    apply_pairwise_knobs_torch,
    assign_columns_from_uv,
    build_neighbor_graph,
    compute_column_labels,
    generate_compressed_video_mp4,
    get_equidistant_points,
    greedy_blue_mask,
    is_adex_model,
    is_connconstr_model,
    is_flyvis_hybrid_model,
    is_hodgkin_huxley_model,
    mseq_bits,
)
from connectome_gnn.utils import get_datavis_root_dir, git_sha, graphs_data_path, to_numpy

logger = get_logger(__name__)


def _resolve_opto_data_root(opto_cfg) -> None:
    """Switch the data root to whichever fallback contains the opto source.

    Opto re-simulation requires the source dataset to already exist on disk.
    GNN_Main.py's _maybe_fallback_data_root() skips fallback resolution for
    'generate' tasks (it expects fresh local data). For opto we override that:
    if the source can't be found at the current data root, scan the
    data_paths.json fallback roots and switch to the first one that has it.
    """
    from connectome_gnn.utils import (
        get_data_root, load_data_fallback_roots, set_data_root,
    )

    src = opto_cfg.source_dataset
    if not src:
        return

    def _has_source(root: str) -> bool:
        for sub in (src, os.path.join("fly", src)):
            voltage = os.path.join(root, "graphs_data", sub,
                                  "x_list_train", "voltage.zarr")
            if os.path.isdir(voltage):
                return True
        return False

    if _has_source(get_data_root()):
        return

    Y, R = "\033[93m", "\033[0m"
    for root in load_data_fallback_roots():
        if _has_source(root):
            print(f"{Y}[opto] source not found at current data root; "
                  f"switching to {root}{R}", flush=True)
            set_data_root(root)
            return
    # No fallback worked — let downstream fail with a clear error.


def _print_opto_banner(config, opto_cfg) -> None:
    """Green banner: confirms source-dataset reuse and dumps opto parameters."""
    G, R = "\033[92m", "\033[0m"
    src = opto_cfg.source_dataset
    src_dir = graphs_data_path(src)
    if not os.path.isdir(src_dir):
        alt = graphs_data_path("fly", src)
        if os.path.isdir(alt):
            src_dir = alt
    voltage_zarr = os.path.join(src_dir, "x_list_train", "voltage.zarr")
    src_ok = os.path.isdir(voltage_zarr)
    tgt = opto_cfg.target
    wf = opto_cfg.waveform
    target_str = (
        f"mode={tgt.mode} k={tgt.k}" if str(tgt.mode) == "OptoTargetMode.TOPK_NULLSPACE"
        or tgt.mode == "topk_nullspace"
        else f"mode={tgt.mode} cell_types={list(tgt.cell_types)}"
    )
    print(f"{G}{'='*70}{R}")
    print(f"{G}[opto] OPTOGENETIC PERTURBATION — re-simulation from existing source{R}")
    print(f"{G}[opto] source dataset:  {src}{R}")
    print(f"{G}[opto] source on disk:  {src_dir}  ({'OK' if src_ok else 'MISSING'}){R}")
    print(f"{G}[opto] target output:   {config.dataset}{R}")
    print(f"{G}[opto] target spec:     {target_str}  column_distinct={tgt.column_distinct}{R}")
    wf_extra = f"  frames_on={wf.frames_on}" if wf.kind == "heaviside" else ""
    print(f"{G}[opto] waveform:        kind={wf.kind}  amplitude={wf.amplitude}  "
          f"noise_level={wf.noise_level}{wf_extra}{R}")
    print(f"{G}[opto] seed:            {wf.seed}  (paired with source for matched comparison){R}")
    print(f"{G}{'='*70}{R}", flush=True)


def data_generate(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    best_model=None,
    device=None,
    save=True,
    log_file=None,
    compute_ranks=True,
):

    logger.info(f"dataset: {config.dataset}")

    # Optogenetics dispatch: if the config has opto enabled, route to the
    # re-simulation pass (generators/optogenetics.py) instead of generating
    # the baseline dataset from scratch. The opto pipeline reads the source's
    # existing voltage/stimulus zarrs and only writes the perturbed dataset.
    opto_cfg = getattr(config.simulation, 'optogenetics', None)
    if opto_cfg is not None and opto_cfg.enabled:
        from connectome_gnn.generators.optogenetics import add_optogenetics_stimulus
        _resolve_opto_data_root(opto_cfg)
        _print_opto_banner(config, opto_cfg)
        add_optogenetics_stimulus(config)
        return

    # Teacher-voltage generation: roll out a trained TaskRNN over fresh task
    # stimuli and write the hidden-state trajectory to x_list_train/voltage.zarr
    # (compatible with data_train_gnn). Owns the whole pipeline -> return.
    if getattr(config.simulation, 'task_model_config_path', ''):
        _generate_voltage_from_task_model(
            config, device=device, visualize=visualize,
        )
        return

    # Task-data generation (PR1: path_integration only). Runs independently
    # from the simulation pipeline below; merging the two (task-trained
    # circuit -> simulate activity -> GNN recovery) is a follow-up PR.
    if getattr(config, 'task', None) is not None:
        data_generate_task(config, device=device, visualize=visualize)
        if config.task.task_only:
            return

    dataset_dir = graphs_data_path(config.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    lock_path   = os.path.join(dataset_dir, ".generate.lock")
    done_marker = os.path.join(dataset_dir, ".generate_done")

    def _data_exists():
        # `.generate_done` is the authoritative marker; the legacy probes
        # keep datasets created before the marker existed from regenerating.
        return (
            os.path.isfile(done_marker)
            or os.path.isdir(graphs_data_path(config.dataset, "x_list_train"))
            or os.path.isfile(graphs_data_path(config.dataset, "x_list_0.npy"))
            or os.path.isfile(graphs_data_path(config.dataset, "x_list_0.pt"))
        )

    # Fast path: avoid touching the lock if the data is already there.
    if _data_exists() and not erase:
        logger.info("data already generated, skipping (use --force to regenerate)")
        return

    # Serialize check+generate against peers. `erase` only wipes
    # x_list_*/y_list_*, so the lock file itself survives regeneration.
    # Non-blocking: if a peer holds the lock, fail fast — this function is
    # typically called from GPU-reserving jobs, so we must not sit idle.
    with open(lock_path, "w") as lf:
        try:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"dataset '{config.dataset}' is being generated by another process "
                f"(lock held at {lock_path}). Pre-generate shared datasets before "
                f"launching parallel jobs, or wait for the peer to finish."
            )

        # Re-check under lock: a peer may have finished between our fast-path
        # check and acquiring the lock.
        if _data_exists() and not erase:
            logger.info("data generated by concurrent process, skipping")
            return

        if _data_exists() and erase:
            logger.warning("data already exists, erasing and regenerating (--force)")

        # Clear the done marker so a crash mid-generation leaves the dataset
        # correctly flagged as incomplete for the next caller.
        if os.path.isfile(done_marker):
            os.remove(done_marker)

        if config.data_folder_name != "none":
            generate_from_data(config=config, device=device, visualize=visualize, style=style, step=step)
        elif is_connconstr_model(config.graph_model.signal_model_name):
            data_generate_connconstr(
                config,
                visualize=visualize,
                device=device,
                save=save,
                erase=erase,
            )
        elif is_adex_model(config.graph_model.signal_model_name):
            data_generate_spiking(
                config,
                visualize=visualize,
                run_vizualized=run_vizualized,
                style=style,
                erase=erase,
                step=step,
                device=device,
                save=save,
                compute_ranks=compute_ranks,
            )
        elif is_hodgkin_huxley_model(config.graph_model.signal_model_name):
            data_generate_voltage(
                config,
                visualize=visualize,
                run_vizualized=run_vizualized,
                style=style,
                erase=erase,
                step=step,
                device=device,
                save=save,
                compute_ranks=compute_ranks,
            )
        else:
            data_generate_voltage(
                config,
                visualize=visualize,
                run_vizualized=run_vizualized,
                style=style,
                erase=erase,
                step=step,
                device=device,
                save=save,
                compute_ranks=compute_ranks,
            )

        # Mark generation complete — only reached on successful dispatch.
        # Record the git SHA so the dataset's provenance is self-describing.
        with open(done_marker, "w") as _dm:
            _dm.write(f"git_sha: {git_sha()}\n")
            _dm.write(f"date:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            _dm.write(f"dataset: {config.dataset}\n")

    default_style.apply_globally()


# ---------------------------------------------------------------------------
# Task-data generation (PR1: path_integration only; OF + twenty_tasks stubbed)
# Schema: see config.TaskConfig + InputPerturbation.
# Plan:   /home/node/.claude/plans/structured-swimming-pearl.md
# ---------------------------------------------------------------------------


def _write_trial_zarr(
    path: str,
    arr: np.ndarray,
    *,
    chunk_trials: int = 1000,
    desc: str | None = None,
) -> None:
    """Write a (N_trials, T, n) array as a zarr with per-trial chunks.

    `chunk_trials` is the number of trials per zstd block. 1000 trials × T=1000 ×
    3 channels ≈ 12 MB per chunk — the sweet spot for sequential reads. Reuses
    ZarrArrayWriter by remapping its (T, N, F) axes to (N_trials, T, n).
    """
    n_trials, T, n_feat = arr.shape
    chunks = max(1, min(int(chunk_trials), n_trials))
    writer = ZarrArrayWriter(
        path=path,
        n_neurons=T,
        n_features=n_feat,
        time_chunks=chunks,
    )
    n_flushes = math.ceil(n_trials / chunks)
    label = desc or os.path.basename(path)
    for i in tqdm(range(n_trials), desc=f"  zarr {label} ({n_flushes} chunks)",
                  leave=False, ncols=150):
        writer.append(arr[i])
    writer.finalize()


def _write_trial_zarr_1d(
    path: str,
    arr: np.ndarray,
    *,
    chunk_trials: int = 1000,
    desc: str | None = None,
) -> None:
    """Write a (N_trials, T) array as zarr by promoting to (N_trials, T, 1)."""
    _write_trial_zarr(path, arr[..., None].astype(np.float32),
                      chunk_trials=chunk_trials, desc=desc)


def _generate_path_integration_task(config, *, device, visualize: bool = True) -> None:
    """Generate the Hulse path-integration task data (input + heading target).

    Layout — flyvis-style flat dataset folder:
        <dataset>/
            task_traces_{train,test}.png
            train/
                stimulus.zarr   (N, T, 3)   [omega(t), cos(theta_0)*delta_t0, sin(theta_0)*delta_t0]
                target.zarr     (N, T, 2)   [cos(theta_hd(t)), sin(theta_hd(t))]
                theta_hd.zarr   (N, T)      ground-truth heading
                is_stop.zarr    (N, T)      standing-pause mask
                stimulus_canonical.zarr / delta_stimulus.zarr  (only when input_perturbation is set)
            test/
                same fields

    Inlines the OU-velocity / heading integration from Hulse Methods Eqs. 5-7;
    the trainer in teachers/janelia_cx_teacher.py keeps its own torch-tensor
    version for the BPTT loop.
    """
    task = config.task
    pi = task.path_integration
    seed_seq = np.random.SeedSequence(pi.seed)
    rng_train, rng_test = (np.random.default_rng(s) for s in seed_seq.spawn(2))

    out_root = graphs_data_path(config.dataset)
    os.makedirs(out_root, exist_ok=True)
    logger.info(f"[task] path_integration -> {out_root}")
    logger.info(
        f"[task] T={pi.n_steps} dt={pi.dt} sigma_omega={pi.sigma_omega_deg} "
        f"tau_corr={pi.tau_corr} train={pi.n_trials_train} test={pi.n_trials_test} "
        f"perturb={pi.input_perturbation is not None}"
    )

    T = int(pi.n_steps)
    dt = float(pi.dt)
    alpha = 1.0 / float(pi.tau_corr)
    sigma_step = float(pi.sigma_omega_deg) * math.sqrt(2.0 * alpha) * math.sqrt(dt)
    decay = 1.0 - alpha * dt
    mean_steps = pi.stop_mean_s / dt
    max_steps = int(pi.stop_max_s / dt)
    target_stop = int(pi.stop_fraction * T)

    for split, n_trials, rng in [
        ("train", pi.n_trials_train, rng_train),
        ("test", pi.n_trials_test, rng_test),
    ]:
        if n_trials <= 0:
            continue
        B = int(n_trials)

        logger.info(f"[task] {split}: generating {B} trials of T={T} ...")

        # OU-driven angular velocity (Hulse Eq. 5). Loop is over T (not B), so
        # the bar shows time-step progress; B-scaling is in the array width.
        omega = np.zeros((B, T), dtype=np.float32)
        eta = rng.standard_normal(size=(B, T)).astype(np.float32)
        for t in tqdm(range(1, T), desc=f"  {split} OU velocity (B={B})",
                      ncols=150, leave=False):
            omega[:, t] = decay * omega[:, t - 1] + sigma_step * eta[:, t]

        # Standing-pause mask: insert exponential-duration stops per trial.
        is_stop = np.zeros((B, T), dtype=np.float32)
        if pi.stop_fraction > 0.0:
            for b in tqdm(range(B), desc=f"  {split} stop-mask",
                          ncols=150, leave=False):
                covered = 0
                attempts = 0
                while covered < target_stop and attempts < 100:
                    attempts += 1
                    start = rng.integers(0, T)
                    length = min(max_steps, int(rng.exponential(mean_steps)), T - start)
                    if length <= 0:
                        continue
                    end = start + length
                    already = int(is_stop[b, start:end].sum())
                    is_stop[b, start:end] = 1.0
                    covered += length - already
            omega *= 1.0 - is_stop  # zero velocity during stops

        # Integrate to heading (Hulse Eq. 6).
        theta0 = rng.uniform(0.0, 2.0 * math.pi, size=B).astype(np.float32)
        theta_hd = theta0[:, None] + np.cumsum(np.deg2rad(omega), axis=1) * dt
        theta_hd[:, 0] = theta0
        target_y = np.stack([np.cos(theta_hd), np.sin(theta_hd)],
                            axis=-1).astype(np.float32)

        # Input vector (Hulse Eq. 7): [omega, cos(theta0)·δ_t0, sin(theta0)·δ_t0].
        stimulus_canonical = np.zeros((B, T, 3), dtype=np.float32)
        stimulus_canonical[:, :, 0] = omega
        stimulus_canonical[:, 0, 1] = np.cos(theta0)
        stimulus_canonical[:, 0, 2] = np.sin(theta0)

        delta_stimulus = None
        if pi.input_perturbation is not None:
            delta_stimulus = np.zeros_like(stimulus_canonical)
            # Per-trial deterministic perturbation seed: derive from the split RNG
            # so reordering trials doesn't shift all subsequent perturbations.
            trial_seeds = rng.integers(0, 2**31 - 1, size=B, dtype=np.int64)
            for i in tqdm(range(B), desc=f"  {split} perturbation",
                          ncols=150, leave=False):
                pert = build_input_perturbation(
                    n_frames=T,
                    n_channels=stimulus_canonical.shape[-1],
                    perturbation=pi.input_perturbation,
                    seed=int(trial_seeds[i]),
                    device=device,
                )
                delta_stimulus[i] = pert.detach().cpu().numpy()
            stimulus = (stimulus_canonical + delta_stimulus).astype(np.float32)
        else:
            stimulus = stimulus_canonical

        split_dir = os.path.join(out_root, split)
        os.makedirs(split_dir, exist_ok=True)

        _write_trial_zarr(os.path.join(split_dir, "stimulus.zarr"), stimulus)
        _write_trial_zarr(os.path.join(split_dir, "target.zarr"), target_y)
        _write_trial_zarr_1d(os.path.join(split_dir, "theta_hd.zarr"), theta_hd)
        _write_trial_zarr_1d(os.path.join(split_dir, "is_stop.zarr"), is_stop)
        if delta_stimulus is not None:
            _write_trial_zarr(os.path.join(split_dir, "stimulus_canonical.zarr"), stimulus_canonical)
            _write_trial_zarr(os.path.join(split_dir, "delta_stimulus.zarr"), delta_stimulus)
        logger.info(f"[task]   {split}: wrote {B} trials of T={T}")

        if visualize:
            plot_task_pi_traces(
                u=stimulus, y=target_y, theta_hd=theta_hd, is_stop=is_stop, dt=dt,
                out_path=os.path.join(out_root, f"task_traces_{split}.png"),
            )


def _generate_cortex_task(config, *, device, visualize: bool = True) -> None:
    """Generate Yang et al. 2019 multitask cognitive battery data.

    Drives `generators/cortex_task.py` (verbatim port of gyyang/multitask) and
    stores trials as zarr arrays under `<dataset>/<split>/`. Single flat
    layout: for any fixed `ruleset` Yang's N_i / N_o are constant across
    rules (the rule one-hot embedded in the input acts as the task ID).

    Layout — flyvis-style flat dataset folder:
        <dataset>/
            rules.json            ruleset + N_i + N_o + dt + rules list
            task_cortex_overview_<split>.png      multi-rule heatmap grid (only if >1 rule)
            task_cortex_example_<split>_<rule>.png  single-rule close-up
            task_cortex_traces_<split>_<rule>.png   line-overlay sanity check
            train/
                stimulus.zarr     (N, T_max, N_i)     padded Yang trial.x
                target.zarr       (N, T_max, N_o)     padded Yang trial.y
                c_mask.zarr       (N, T_max, N_o)     padded Yang c_mask
                length.zarr       (N, T_max)          real-step mask
                rule_idx.zarr     (N,)                index into ct.rules
                stimulus_canonical.zarr / delta_stimulus.zarr  (only when input_perturbation set)
            test/
                same fields
    """
    import json

    from connectome_gnn.generators.cortex_task import generate_trials, get_default_hp
    from connectome_gnn.generators.cortex_adapter import trial_to_numpy

    task = config.task
    ct = task.cortex

    if ct.rule_weights and len(ct.rule_weights) != len(ct.rules):
        raise ValueError(
            f"cortex.rule_weights length {len(ct.rule_weights)} != "
            f"rules length {len(ct.rules)}"
        )

    # Build Yang hp + apply overrides. `get_default_hp` returns a fresh dict.
    hp = get_default_hp(ct.ruleset)
    for k, v in (ct.hp_overrides or {}).items():
        hp[k] = v
    n_in = int(hp["n_input"])
    n_out = int(hp["n_output"])
    dt_s = float(hp["dt"]) / 1000.0   # Yang stores dt in ms; convert to seconds

    out_root = graphs_data_path(config.dataset)
    os.makedirs(out_root, exist_ok=True)
    logger.info(f"[task] cortex -> {out_root}")
    logger.info(
        f"[task] ruleset={ct.ruleset!r} rules={ct.rules} n_in={n_in} n_out={n_out} "
        f"dt={dt_s}s n_steps_max={ct.n_steps_max} "
        f"train={ct.n_trials_train} test={ct.n_trials_test} "
        f"perturb={ct.input_perturbation is not None}"
    )

    rules = list(ct.rules)
    weights = list(ct.rule_weights) if ct.rule_weights else None
    if weights:
        s = float(sum(weights))
        weights = [w / s for w in weights]

    # Persist ruleset metadata at dataset root.
    with open(os.path.join(out_root, "rules.json"), "w") as f:
        json.dump({
            "rules":   rules,
            "ruleset": ct.ruleset,
            "N_i":     n_in,
            "N_o":     n_out,
            "dt":      dt_s,
            "n_steps_max": int(ct.n_steps_max),
            "hp_overrides": dict(ct.hp_overrides or {}),
        }, f, indent=2)

    # Per-split deterministic seed streams: train and test use different RNGs
    # spawned from the cortex.seed so adding test trials doesn't shift train.
    seed_seq = np.random.SeedSequence(ct.seed)
    split_seeds = dict(zip(("train", "test"), seed_seq.spawn(2)))

    for split, n_total in [("train", ct.n_trials_train),
                           ("test",  ct.n_trials_test)]:
        if n_total <= 0:
            continue

        # One RNG drives both rule choice and Yang's per-trial randomness
        # (passed in via hp['rng']). One RNG drives the perturbation stream.
        rule_rng, yang_rng, pert_rng = (
            np.random.default_rng(s) for s in split_seeds[split].spawn(3)
        )
        # Yang uses RandomState (legacy); bridge it with the per-split seed.
        hp = dict(hp)
        hp['rng'] = np.random.RandomState(int(yang_rng.integers(0, 2**31 - 1)))

        T_max = int(ct.n_steps_max)
        stimulus_canonical = np.zeros((n_total, T_max, n_in), dtype=np.float32)
        target             = np.zeros((n_total, T_max, n_out), dtype=np.float32)
        c_mask             = np.zeros((n_total, T_max, n_out), dtype=np.float32)
        length             = np.zeros((n_total, T_max),        dtype=np.float32)
        rule_idx           = np.zeros((n_total,),              dtype=np.int64)

        # Stash the first 5 Trials per rule so plotters can render per-trial
        # epoch boundaries (each trial has its own random epoch timing).
        sampled_trials_per_rule: dict[str, list] = {}

        for i in tqdm(range(n_total), desc=f"  {split} trials",
                      ncols=150, leave=False):
            r_idx = int(rule_rng.choice(len(rules), p=weights))
            r = rules[r_idx]
            trial = generate_trials(r, hp, mode='random', batch_size=1)
            T_trial = int(trial.tdim)
            if T_trial > T_max:
                raise ValueError(
                    f"[cortex/{r}] trial length {T_trial} > "
                    f"n_steps_max={T_max}; raise n_steps_max in the YAML."
                )
            x_in, y_tgt, cm = trial_to_numpy(trial, 0)
            stimulus_canonical[i, :T_trial] = x_in
            target[i, :T_trial]             = y_tgt
            c_mask[i, :T_trial]             = cm
            length[i, :T_trial]             = 1.0
            rule_idx[i]                     = r_idx
            bucket = sampled_trials_per_rule.setdefault(r, [])
            if len(bucket) < 5:
                bucket.append(trial)

        # Optional decorrelation perturbation on top of the canonical input.
        delta_stimulus = None
        if ct.input_perturbation is not None:
            trial_seeds = pert_rng.integers(0, 2**31 - 1, size=n_total, dtype=np.int64)
            delta_stimulus = np.zeros_like(stimulus_canonical)
            for i in tqdm(range(n_total), desc=f"  {split} perturbation",
                          ncols=150, leave=False):
                pert = build_input_perturbation(
                    n_frames=T_max,
                    n_channels=n_in,
                    perturbation=ct.input_perturbation,
                    seed=int(trial_seeds[i]),
                    device=device,
                )
                # Mask perturbation to real timesteps so padding stays clean.
                delta_stimulus[i] = pert.detach().cpu().numpy() * length[i, :, None]
            stimulus = (stimulus_canonical + delta_stimulus).astype(np.float32)
        else:
            stimulus = stimulus_canonical

        # Write zarrs.
        split_dir = os.path.join(out_root, split)
        os.makedirs(split_dir, exist_ok=True)
        _write_trial_zarr(os.path.join(split_dir, "stimulus.zarr"), stimulus)
        _write_trial_zarr(os.path.join(split_dir, "target.zarr"),   target)
        _write_trial_zarr(os.path.join(split_dir, "c_mask.zarr"),   c_mask)
        _write_trial_zarr_1d(os.path.join(split_dir, "length.zarr"), length)
        # rule_idx is one-per-trial (1D); zarr chunked layout is overkill — store
        # as .npy at the split root. Plotting / loaders read it as a flat array.
        np.save(os.path.join(split_dir, "rule_idx.npy"), rule_idx)
        if delta_stimulus is not None:
            _write_trial_zarr(
                os.path.join(split_dir, "stimulus_canonical.zarr"), stimulus_canonical
            )
            _write_trial_zarr(
                os.path.join(split_dir, "delta_stimulus.zarr"), delta_stimulus
            )

        rule_counts = {r: int((rule_idx == ri).sum()) for ri, r in enumerate(rules)}
        logger.info(
            f"[task]   {split}: wrote {n_total} trials "
            f"(N_i={n_in}, N_o={n_out}, T_max={T_max}) — rule counts: {rule_counts}"
        )

        if visualize:
            from connectome_gnn.plot import (
                plot_task_cortex_example,
                plot_task_cortex_overview,
                plot_task_cortex_samples,
            )

            # Per-rule: 5-trial heatmap grid + 5-trial line-trace grid.
            for r, trial_list in sampled_trials_per_rule.items():
                idxs_for_r = np.where(rule_idx == rules.index(r))[0]
                sample_idx = idxs_for_r[:5]
                epochs_per_trial = [getattr(t, 'epochs', None) for t in trial_list]
                plot_task_cortex_example(
                    stimulus=stimulus[sample_idx],
                    target=target[sample_idx],
                    length=length[sample_idx],
                    dt=dt_s, rule=r, epochs=epochs_per_trial,
                    n_rule=int(hp.get('n_rule', 0)),
                    n_eachring=int(hp.get('n_eachring', 32)),
                    out_path=os.path.join(out_root, f"task_cortex_example_{split}_{r}.png"),
                )
                plot_task_cortex_samples(
                    stimulus=stimulus[sample_idx],
                    target=target[sample_idx],
                    length=length[sample_idx],
                    dt=dt_s, rule=r,
                    n_eachring=int(hp.get('n_eachring', 32)),
                    out_path=os.path.join(out_root, f"task_cortex_samples_{split}_{r}.png"),
                )

            # Multi-rule grid overview — only meaningful for multi-rule runs.
            if len(rules) > 1:
                # Pick one trial per rule (the first occurrence) for the grid.
                ridx_for_grid = [int(np.where(rule_idx == ri)[0][0])
                                 for ri in range(len(rules))
                                 if (rule_idx == ri).any()]
                plot_task_cortex_overview(
                    stimulus=stimulus[ridx_for_grid],
                    target=target[ridx_for_grid],
                    rules=[rules[int(rule_idx[i])] for i in ridx_for_grid],
                    n_rule=int(hp.get('n_rule', 0)),
                    n_eachring=int(hp.get('n_eachring', 32)),
                    out_path=os.path.join(out_root, f"task_cortex_overview_{split}.png"),
                )


def data_generate_task(config, *, device, visualize: bool = True) -> None:
    """Top-level dispatcher for task-data generation.

    Routes on config.task.task_type. path_integration: Hulse heading task;
    cortex: Yang 2019 multitask cognitive battery; optical_flow: not yet
    implemented (schema only).
    """
    task = config.task
    if task is None:
        return
    if task.task_type == "path_integration":
        _generate_path_integration_task(config, device=device, visualize=visualize)
    elif task.task_type == "optical_flow":
        raise NotImplementedError(
            "optical_flow task generation is not implemented yet; "
            "schema is declared so YAMLs validate."
        )
    elif task.task_type == "cortex":
        _generate_cortex_task(config, device=device, visualize=visualize)
    else:
        raise ValueError(f"unknown task_type: {task.task_type!r}")


def data_generate_connconstr(config, visualize=True, device=None, save=True, erase=False):
    """Generate simulation data from a connconstr biological connectome model.

    Ref: Beiran & Litwin-Kumar (2023), Fig 5

    Model-agnostic: uses registry methods on ODE params classes
    (create_ode, generate_stimulus, init_state, etc.)
    """
    # Erase old data if requested (prevents appending to old runs)
    if erase:
        for split in ['train', 'test', '0']:  # '0' for fallback compat
            for data_file in ['x_list', 'y_list']:
                old_path = graphs_data_path(config.dataset, f"{data_file}_{split}")
                if os.path.exists(old_path):
                    _rmtree(old_path)
                    logger.info(f"erased old {data_file}_{split}")

    from connectome_gnn.generators.ode_params import get_ode_params_class

    sim = config.simulation
    model_name = config.graph_model.signal_model_name

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(sim.seed)
    np.random.seed(sim.seed)

    logger.info(f"generating connconstr data ... model={model_name}  datapath={sim.connconstr_datapath}")

    folder = graphs_data_path(config.dataset) + "/"
    os.makedirs(folder, exist_ok=True)

    # Load ODE params via registry
    OdeParamsCls = get_ode_params_class(model_name)
    datapath = sim.connconstr_datapath

    if sim.connconstr_use_pretrained and hasattr(OdeParamsCls, "from_pretrained"):
        try:
            ode_params = OdeParamsCls.from_pretrained(datapath, device=device)
            logger.info(f"loaded pretrained params for {model_name}")
        except FileNotFoundError:
            logger.info(f"pretrained not found, using connectome for {model_name}")
            ode_params = OdeParamsCls.from_connectome(datapath, device=device)
    else:
        ode_params = OdeParamsCls.from_connectome(datapath, device=device)

    edge_index = ode_params.edge_index
    if save:
        ode_params.save(folder)
        torch.save(edge_index.clone(), os.path.join(folder, "edge_index.pt"))
        torch.save(ode_params.W.clone(), os.path.join(folder, "weights.pt"))

    # Create ODE, get integration params — all via registry methods
    pde = ode_params.create_ode(device=device)
    n_neurons = ode_params.get_n_neurons()
    dt = ode_params.get_dt()
    n_frames_total = ode_params.get_n_frames(sim)
    trial_len = ode_params.get_trial_length()

    logger.info(f"n_neurons={n_neurons}  n_edges={edge_index.shape[1]}  dt={dt}  n_frames={n_frames_total}")

    # Generate per-neuron stimulus (T, N) via registry method
    stim_all = ode_params.generate_stimulus(n_frames_total, sim, device=device)
    n_frames_total = stim_all.shape[0]  # may be adjusted by stimulus generator

    # Initialize neuron state
    x = NeuronState(
        index=torch.arange(n_neurons, dtype=torch.long, device=device),
        pos=torch.zeros(n_neurons, 2, dtype=torch.float32, device=device),
        voltage=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        stimulus=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        group_type=torch.zeros(n_neurons, dtype=torch.long, device=device),
        neuron_type=ode_params.neuron_types
        if hasattr(ode_params, "neuron_types") and ode_params.neuron_types is not None
        else torch.zeros(n_neurons, dtype=torch.long, device=device),
        calcium=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        fluorescence=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        noise=torch.zeros(n_neurons, dtype=torch.float32, device=device),
    )

    # Set initial state via registry method
    ode_params.init_state(x.voltage, datapath=datapath, device=device)

    # Split into train/test (80/20 by time)
    n_train = int(n_frames_total * 0.8)
    n_test = n_frames_total - n_train

    # Collect voltage history for visualization (train split only)
    voltage_history = [] if visualize else None
    stimulus_history = [] if visualize else None
    frame_index_history = [] if visualize else None

    for split, (frame_start, frame_end) in [("train", (0, n_train)), ("test", (n_train, n_frames_total))]:
        n_split = frame_end - frame_start
        logger.info(f"generating {split} split: frames [{frame_start}, {frame_end}) ({n_split} frames)")

        # Test data is noise-free: the model learns deterministic dynamics,
        # so rollout comparison against noisy ground truth is meaningless.
        noise_level = sim.noise_model_level if split == "train" else 0.0

        if split == "test":
            x.voltage[:] = 0

        x_writer = ZarrSimulationWriterV3(
            path=graphs_data_path(config.dataset, f"x_list_{split}"),
            n_neurons=n_neurons,
            time_chunks=2000,
        )
        y_writer = ZarrArrayWriter(
            path=graphs_data_path(config.dataset, f"y_list_{split}"),
            n_neurons=n_neurons,
            n_features=1,
            time_chunks=2000,
        )

        with torch.no_grad():
            for t in tqdm(range(frame_start, frame_end), desc=f"connconstr {split}", ncols=100):
                # Reset state at trial boundaries if model has trial structure
                if trial_len > 0 and t % trial_len == 0:
                    ode_params.init_state(x.voltage, datapath=datapath, device=device)

                # Set per-neuron stimulus from precomputed tensor
                x.stimulus[:] = stim_all[t]

                x_writer.append_state(x)

                if visualize and split == "train" and (t - frame_start) % max(1, n_split // 5000) == 0:
                    voltage_history.append(to_numpy(x.voltage.clone()))
                    stimulus_history.append(to_numpy(x.stimulus.clone()))
                    frame_index_history.append(t)

                # Euler step
                dv = pde(x, edge_index)
                dv_squeeze = dv.squeeze()

                if noise_level > 0:
                    x.voltage = (
                        x.voltage
                        + dt * dv_squeeze
                        + torch.randn(n_neurons, dtype=torch.float32, device=device) * noise_level
                    )
                else:
                    x.voltage = x.voltage + dt * dv_squeeze

                y_writer.append(to_numpy(dv.clone().detach()))

        n_written = x_writer.finalize()
        y_writer.finalize()
        logger.info(f"generated {n_written} {split} frames")

    # --- Compute effective ranks (W matrix, activity, stimulus) ---
    logger.info("computing effective rank ...")

    def _svd_lowrank(matrix_np, n_components, dev):
        """Randomized SVD via torch.svd_lowrank (GPU-accelerated when available)."""
        t = torch.as_tensor(matrix_np, dtype=torch.float32, device=dev)
        _, S, _ = torch.svd_lowrank(t, q=n_components + 10, niter=4)
        return S[:n_components].cpu().numpy()

    _svd_device = torch.device(device) if (device and device != 'cpu' and torch.cuda.is_available()) else torch.device('cpu')
    logger.info(f"  SVD device: {_svd_device}")

    # W matrix rank from dense reconstruction
    ei_np = to_numpy(edge_index)
    W_np = to_numpy(ode_params.W)
    W_dense = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    W_dense[ei_np[0], ei_np[1]] = W_np
    n_comp_w = min(50, min(W_dense.shape) - 1)
    S_w = _svd_lowrank(W_dense, n_comp_w, _svd_device)
    cumvar_w = np.cumsum(S_w**2) / np.sum(S_w**2)
    rank_90_w = int(np.searchsorted(cumvar_w, 0.90) + 1)
    rank_99_w = int(np.searchsorted(cumvar_w, 0.99) + 1)
    logger.info(f"W matrix rank(90%)={rank_90_w}  rank(99%)={rank_99_w}")

    # Activity rank from train zarr
    from connectome_gnn.zarr_io import load_simulation_data

    x_ts = load_simulation_data(graphs_data_path(config.dataset, "x_list_train"))
    activity_full = x_ts.voltage.numpy()
    n_comp_a = min(50, min(activity_full.shape) - 1)
    S_act = _svd_lowrank(activity_full, n_comp_a, _svd_device)
    cumvar_act = np.cumsum(S_act**2) / np.sum(S_act**2)
    rank_90_act = int(np.searchsorted(cumvar_act, 0.90) + 1)
    rank_99_act = int(np.searchsorted(cumvar_act, 0.99) + 1)

    # Mean-centered rank: subtract per-neuron temporal mean to remove static bias pattern.
    # This captures dynamic information content (what the GNN must learn beyond a constant offset).
    activity_centered = activity_full - activity_full.mean(axis=0, keepdims=True)
    centered_var = np.sum(activity_centered**2)
    if centered_var > 1e-12:
        S_mc = _svd_lowrank(activity_centered, n_comp_a, _svd_device)
        cumvar_mc = np.cumsum(S_mc**2) / centered_var
        rank_90_mc = int(np.searchsorted(cumvar_mc, 0.90) + 1)
        rank_99_mc = int(np.searchsorted(cumvar_mc, 0.99) + 1)
    else:
        rank_90_mc = rank_99_mc = 0
    logger.info(
        f"activity rank(90%)={rank_90_act}  rank(99%)={rank_99_act}  mean-centered rank(90%)={rank_90_mc}  rank(99%)={rank_99_mc}"
    )

    # Stimulus rank
    stim_full = x_ts.stimulus.numpy()
    n_comp_s = min(50, min(stim_full.shape) - 1)
    if n_comp_s > 0 and np.abs(stim_full).max() > 1e-12:
        S_stim = _svd_lowrank(stim_full, n_comp_s, _svd_device)
        cumvar_stim = np.cumsum(S_stim**2) / np.sum(S_stim**2)
        rank_90_stim = int(np.searchsorted(cumvar_stim, 0.90) + 1)
        rank_99_stim = int(np.searchsorted(cumvar_stim, 0.99) + 1)
    else:
        rank_90_stim = rank_99_stim = 0
    logger.info(f"stimulus rank(90%)={rank_90_stim}  rank(99%)={rank_99_stim}")

    # Write rank info to logfile in dataset folder
    rank_log_path = os.path.join(folder, "rank_info.txt")
    with open(rank_log_path, "w") as f:
        f.write(f"model: {model_name}\n")
        f.write(f"n_neurons: {n_neurons}\n")
        f.write(f"n_edges: {edge_index.shape[1]}\n")
        f.write(f"W matrix rank(90%): {rank_90_w}  rank(99%): {rank_99_w}\n")
        f.write(f"activity rank(90%): {rank_90_act}  rank(99%): {rank_99_act}\n")
        f.write(f"activity mean-centered rank(90%): {rank_90_mc}  rank(99%): {rank_99_mc}\n")
        f.write(f"stimulus rank(90%): {rank_90_stim}  rank(99%): {rank_99_stim}\n")

    rank_info = {
        "rank_90_w": rank_90_w,
        "rank_99_w": rank_99_w,
        "rank_90_act": rank_90_act,
        "rank_99_act": rank_99_act,
        "rank_90_mc": rank_90_mc,
        "rank_99_mc": rank_99_mc,
        "rank_90_stim": rank_90_stim,
        "rank_99_stim": rank_99_stim,
    }

    if visualize and voltage_history:
        plot_connconstr_diagnostics(
            voltage_history,
            stimulus_history,
            ode_params,
            edge_index,
            model_name,
            n_neurons,
            dt,
            config,
            device,
            frame_indices=frame_index_history,
            rank_info=rank_info,
        )


def generate_from_data(config, device, visualize=True, step=None, cmap=None, style=None):
    data_folder_name = config.data_folder_name

    if "wormvae" in data_folder_name:
        load_wormvae_data(config, device, visualize, step)
    elif "NeuroPAL" in data_folder_name:
        # load_neuropal_data(config, device, visualize, step)  # TODO: Function not yet implemented
        raise NotImplementedError("NeuroPAL data loading not yet implemented")
    elif "Zapbench" in data_folder_name:
        load_zebrafish_data(config, device, visualize, step, cmap, style)
    else:
        raise ValueError(f"Unknown data folder name {data_folder_name}")


def data_generate_spiking(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    device=None,
    save=True,
    compute_ranks=True,
):
    """Generate spiking (AdEx) simulation data using the flyvis connectome.

    Uses the same visual stimulus pipeline as data_generate_voltage,
    but integrates AdEx dynamics with event-triggered synaptic transmission.
    """
    from connectome_gnn.generators.flyvis_adex_ode import FlyVisAdExODE
    from connectome_gnn.generators.flyvis_ode import (
        get_photoreceptor_positions_from_net,
        group_by_direction_and_function,
    )
    from connectome_gnn.generators.ode_params import FlyVisAdExODEParams
    from connectome_gnn.utils import setup_flyvis_model_path

    fig_style = dark_style if "black" in style else default_style
    fig_style.apply_globally()

    sim = config.simulation
    model_config = config.graph_model

    # Erase old data if requested (prevents appending to old runs)
    if erase:
        for split in ['train', 'test', '0']:  # '0' for fallback compat
            for data_file in ['x_list', 'y_list']:
                old_path = graphs_data_path(config.dataset, f"{data_file}_{split}")
                if os.path.exists(old_path):
                    _rmtree(old_path)
                    logger.info(f"erased old {data_file}_{split}")

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(sim.seed)
    np.random.seed(sim.seed)

    n_frames = sim.n_frames

    synapse_model = "COBA" if "coba" in model_config.signal_model_name else "CUBA"
    logger.info(
        f"generating spiking data ... {model_config.signal_model_name}  synapse_model: {synapse_model}  seed: {sim.seed}"
    )

    os.makedirs(graphs_data_path("fly"), exist_ok=True)
    folder = graphs_data_path(config.dataset) + "/"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(graphs_data_path(config.dataset, "Fig"), exist_ok=True)
    files = glob.glob(graphs_data_path(config.dataset, "Fig", "*"))
    for f in files:
        os.remove(f)

    # extent=15 → 721 retinotopic columns (5768 photoreceptors); extent=8 → 217 columns (1736 photoreceptors)
    extent = 15 if getattr(sim, 'all_columns', False) else 8

    import logging

    from flyvis import Network, NetworkView
    from flyvis.datasets.sintel import AugmentedSintel
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config

    logging.getLogger().setLevel(logging.WARNING)
    setup_flyvis_model_path()

    # Initialize the flyvis network first (fast) so we can print actual network stats before
    # the slow stimulus rendering begins.
    import logging as _logging
    _logging.getLogger("flyvis.utils.logging_utils").setLevel(_logging.ERROR)
    config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
    config_net.connectome.extent = extent
    net = Network(**config_net)
    nnv = NetworkView(f"flow/{sim.ensemble_id}/{sim.model_id}")
    trained_net = nnv.init_network(checkpoint=0)
    net.load_state_dict(trained_net.state_dict())
    torch.set_grad_enabled(False)

    _node_types_str = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in net.connectome.nodes["type"][:]]
    _photoreceptor_types = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'}
    n_input_neurons_net = int(np.sum([t in _photoreceptor_types for t in _node_types_str]))
    print(f"  n_neurons:       {net.n_nodes}")
    print(f"  n_input_neurons: {n_input_neurons_net}")
    print(f"  n_edges:         {net.n_edges}")

    # Initialize stimulus dataset
    sintel_config = {
        "n_frames": 19,
        "flip_axes": [0, 1],
        "n_rotations": [0, 1, 2, 3, 4, 5],
        "temporal_split": True,
        "dt": sim.delta_t,
        "interpolate": True,
        "boxfilter": dict(extent=extent, kernel_size=13),
        "vertical_splits": 3,
        "center_crop_fraction": 0.7,
    }
    stimulus_dataset = AugmentedSintel(**sintel_config)

    # Build spiking ODE params from flyvis connectome
    adex_overrides = {}
    if hasattr(sim, "adex_stim_scale"):
        adex_overrides["stim_scale"] = sim.adex_stim_scale
    if hasattr(sim, "adex_I_bias"):
        adex_overrides["I_bias"] = sim.adex_I_bias

    ode_params = FlyVisAdExODEParams.from_flyvis_network(
        net,
        synapse_model=synapse_model,
        device=device,
        overrides=adex_overrides if adex_overrides else None,
    )

    if save:
        ode_params.save(folder)

    # Create AdEx ODE
    pde = FlyVisAdExODE(ode_params=ode_params, device=device)

    # Extract positions and neuron metadata. Same fix as in
    # data_generate_voltage: positions are derived from per-node (u, v) for
    # the entire population, not just photoreceptors.
    from connectome_gnn.generators.flyvis_ode import (
        get_all_neuron_positions_from_net,
    )
    x_coords_all, y_coords_all, _u_all, _v_all = get_all_neuron_positions_from_net(net)
    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)
    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]
    # available node types: {'T5d', 'R3', 'T2a', 'TmY14', 'R7', 'CT1(Lo1)', 'Tm4', 'TmY10', 'T4d', 'L1', 'R1', 'R6', 'Am', 'T2', 'Tm5Y', 'L5', 'Tm20', 'L2', 'Mi4', 'Mi12', 'T4c', 'TmY4', 'CT1(M10)', 'TmY15', 'Lawf1', 'T1', 'TmY13', 'Tm5b', 'Tm28', 'L3', 'R8', 'L4', 'C3', 'Mi14', 'Tm2', 'R5', 'R2', 'Mi15', 'Tm9', 'Tm16', 'T5a', 'Mi3', 'TmY9', 'T4b', 'Mi9', 'Mi1', 'T5b', 'Tm1', 'Lawf2', 'C2', 'T4a', 'TmY3', 'Mi2', 'T3', 'TmY5a', 'Mi11', 'Tm5a', 'TmY18', 'Tm30', 'R4', 'Mi13', 'Tm5c', 'T5c', 'Mi10', 'Tm3'}
    grouped_types = np.array([group_by_direction_and_function(t) for t in node_types_str])
    _, node_types_int = np.unique(node_types, return_inverse=True)

    n_neurons = sim.n_neurons
    X1 = torch.tensor(
        np.stack((x_coords_all, y_coords_all), axis=1),
        dtype=torch.float32, device=device,
    )

    # Initialize spiking neuron state
    x = pde.init_state(n_neurons)
    x.index = torch.arange(n_neurons, dtype=torch.long, device=device)
    x.pos = X1
    x.group_type = torch.tensor(grouped_types, dtype=torch.long, device=device)
    x.neuron_type = torch.tensor(node_types_int, dtype=torch.long, device=device)
    x.calcium = torch.zeros(n_neurons, dtype=torch.float32, device=device)
    x.fluorescence = torch.zeros(n_neurons, dtype=torch.float32, device=device)
    x.noise = torch.zeros(n_neurons, dtype=torch.float32, device=device)

    # AdEx integration timestep (ms) — much finer than graded model
    adex_dt = getattr(sim, "adex_dt", 0.2)  # default 0.2 ms
    # Number of AdEx substeps per stimulus frame
    substeps = max(1, int(sim.delta_t / adex_dt))
    logger.info(f"AdEx dt={adex_dt}ms, stimulus dt={sim.delta_t}ms, substeps={substeps}")

    # Train/test split (same logic as graded model)
    df = stimulus_dataset.arg_df
    original_indices = df["original_index"].values
    unique_videos = np.unique(original_indices)
    np.random.shuffle(unique_videos)
    n_train_vids = int(len(unique_videos) * 0.8)
    train_video_set = set(unique_videos[:n_train_vids])
    test_video_set = set(unique_videos[n_train_vids:])

    train_indices = [i for i, oi in enumerate(original_indices) if oi in train_video_set]
    test_indices = [i for i, oi in enumerate(original_indices) if oi in test_video_set]
    train_sequences = [stimulus_dataset[i] for i in train_indices]
    test_sequences = [stimulus_dataset[i] for i in test_indices]

    logger.info(
        f"subdirectory split: {n_train_vids} train / {len(unique_videos) - n_train_vids} test videos"
        f"  ({len(train_indices)} train seqs, {len(test_indices)} test seqs)"
    )

    frames_per_sequence = 35

    def _run_spiking_generation(sequences, x, split_name, target_frames, record_plot_frames=0):
        """Inner loop: run AdEx simulation over stimulus sequences.

        Args:
            record_plot_frames: number of stimulus frames for which to record
                substep-level voltage/spike/stimulus (for plotting). 0 = no recording.

        Returns:
            n_written: number of frames written to zarr.
            plot_data: dict with 'voltage', 'spike_raster', 'stimulus' arrays
                at substep resolution, or None if record_plot_frames == 0.
        """
        x_writer = ZarrSimulationWriterV3(
            path=graphs_data_path(config.dataset, f"x_list_{split_name}"),
            n_neurons=n_neurons,
            time_chunks=2000,
        )
        y_writer = ZarrArrayWriter(
            path=graphs_data_path(config.dataset, f"y_list_{split_name}"),
            n_neurons=n_neurons,
            n_features=1,
            time_chunks=2000,
        )

        # Substep-level recording for plotting
        v_record = [] if record_plot_frames > 0 else None
        spike_record = [] if record_plot_frames > 0 else None
        stim_record = [] if record_plot_frames > 0 else None
        plot_frames_left = record_plot_frames

        it = 0
        with torch.no_grad():
            for data_idx, data in enumerate(tqdm(sequences, desc=f"spiking {split_name}", ncols=100)):
                lum = data["lum"]
                sequence_length = lum.shape[0]

                for frame_id in range(sequence_length):
                    # Set stimulus from visual input (photoreceptors only)
                    frame = lum[frame_id][None, None]
                    net.stimulus.add_input(frame)
                    x.stimulus[:] = 0
                    x.stimulus[: sim.n_input_neurons] = net.stimulus().squeeze()[: sim.n_input_neurons]

                    # Record state BEFORE integration (same convention as graded model)
                    x_writer.append_state(x)

                    # Integrate AdEx for substeps within this stimulus frame
                    v_before = x.voltage.clone()
                    for sub in range(substeps):
                        pde.step(x, adex_dt)

                        # Record substep data for plotting
                        if plot_frames_left > 0:
                            v_record.append(to_numpy(x.voltage.clone()))
                            spike_record.append(to_numpy(x.spiked.clone()))
                            stim_record.append(to_numpy(x.stimulus[: sim.n_input_neurons].clone()))

                    if plot_frames_left > 0:
                        plot_frames_left -= 1

                    # Compute effective dv/dt for this frame (for GNN training target)
                    dv = ((x.voltage - v_before) / sim.delta_t).unsqueeze(-1)
                    y_writer.append(to_numpy(dv.clone().detach()))

                    it += 1
                    if it >= target_frames:
                        break
                if it >= target_frames:
                    break

        n_written = x_writer.finalize()
        y_writer.finalize()

        plot_data = None
        if v_record:
            plot_data = {
                "voltage": np.stack(v_record, axis=1),  # (N, T_substeps)
                "spike_raster": np.stack(spike_record, axis=1),  # (N, T_substeps)
                "stimulus": np.stack(stim_record, axis=1),  # (n_input, T_substeps)
            }
        return n_written, plot_data

    # --- Generate TRAIN split ---
    total_frames_per_pass = len(train_sequences) * frames_per_sequence
    if n_frames == 0:
        target_frames = float("inf")
    else:
        target_frames = n_frames

    # Record substep-level data for first 400 stimulus frames (for plotting ~20000 substeps)
    plot_record_frames = 400
    logger.info(f"generating spiking TRAIN data ({target_frames} frames from {len(train_sequences)} sequences)...")
    n_frames_train, train_plot_data = _run_spiking_generation(
        train_sequences,
        x,
        "train",
        target_frames,
        record_plot_frames=plot_record_frames,
    )
    logger.info(f"generated {n_frames_train} spiking TRAIN frames")

    # --- Plot spiking traces ---
    if train_plot_data is not None:
        logger.info("plotting spiking traces...")
        dataset_dir = graphs_data_path(config.dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        is_exc_np = to_numpy(ode_params.is_excitatory)
        plot_spiking_traces(
            voltage=train_plot_data["voltage"],
            spike_raster=train_plot_data["spike_raster"],
            stimulus=train_plot_data["stimulus"],
            is_excitatory=is_exc_np,
            type_list=node_types_int,
            output_path=dataset_dir,
            n_input_neurons=sim.n_input_neurons,
            dt_ms=adex_dt,
            style=fig_style,
        )
        logger.info(f"saved spiking plots to {dataset_dir}")

    # --- Generate TEST split ---
    # Reset state for test
    x_test = pde.init_state(n_neurons)
    x_test.index = x.index
    x_test.pos = x.pos
    x_test.group_type = x.group_type
    x_test.neuron_type = x.neuron_type
    x_test.calcium = torch.zeros(n_neurons, dtype=torch.float32, device=device)
    x_test.fluorescence = torch.zeros(n_neurons, dtype=torch.float32, device=device)
    x_test.noise = torch.zeros(n_neurons, dtype=torch.float32, device=device)

    MAX_TEST_FRAMES = 8000
    test_target = len(test_sequences) * frames_per_sequence
    logger.info(f"generating spiking TEST data (capped at {MAX_TEST_FRAMES} frames from {len(test_sequences)} sequences)...")
    n_frames_test, _ = _run_spiking_generation(test_sequences, x_test, "test", MAX_TEST_FRAMES)
    logger.info(f"generated {n_frames_test} spiking TEST frames")

    torch.set_grad_enabled(True)
    logger.info("spiking data generation complete")


def data_generate_voltage(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    device=None,
    save=True,
    compute_ranks=True,
):

    fig_style = dark_style if "black" in style else default_style
    fig_style.apply_globally()

    sim = config.simulation
    tc = config.training
    model_config = config.graph_model

    # Erase old data if requested (prevents appending to old runs)
    if erase:
        for split in ['train', 'test']:
            for data_file in ['x_list', 'y_list']:
                old_path = graphs_data_path(config.dataset, f"{data_file}_{split}")
                if os.path.exists(old_path):
                    _rmtree(old_path)
                    logger.info(f"erased old {data_file}_{split}")

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(sim.seed)
    np.random.seed(sim.seed)

    n_frames = sim.n_frames
    n_neurons = sim.n_neurons

    logger.info(
        f"generating data ... {model_config.signal_model_name}  dynamics_noise: {sim.noise_model_level}  measurement_noise: {sim.measurement_noise_level}  seed: {sim.seed}  steady_state_value: {getattr(sim, 'steady_state_value', 0.5)}"
    )

    # Stimulus / blank-prefix summary -- printed up-front so the user sees the
    # actual parameters used at generation time (vs whatever default the loader
    # might silently apply if the YAML is incomplete).
    _bpf = float(getattr(sim, 'blank_prefix_fraction', 0.0))
    _vis_type = getattr(sim, 'visual_input_type', 'DAVIS')
    _datavis_roots = getattr(sim, 'datavis_roots', None) or ['<flyvis default Sintel>']
    _skip_short = bool(getattr(sim, 'skip_short_videos', True))
    # `visual_input_type` is the renderer class (DAVIS/mixed/flash/...), not
    # the dataset identity. For video-based renderers the actual data source
    # comes from `datavis_roots` — surface its basename so the log isn't
    # misleading when e.g. visual_input_type='DAVIS' but the root is YouTube-VOS.
    if 'DAVIS' in _vis_type or 'mixed' in _vis_type:
        _data_source = ', '.join(os.path.basename(r.rstrip('/')) for r in _datavis_roots)
        _renderer_str = f"renderer={_vis_type}  data_source={_data_source}"
    else:
        _renderer_str = f"renderer={_vis_type}"
    print(
        f"\033[93m[stimulus] {_renderer_str}  "
        f"blank_prefix_fraction={_bpf:.3f} "
        f"({'BLANK PREFIX ENABLED' if _bpf > 0 else 'no blank prefix'})  "
        f"skip_short_videos={_skip_short}\033[0m",
        flush=True,
    )
    print(f"\033[93m[stimulus] datavis_roots={_datavis_roots}\033[0m", flush=True)
    _ar1_rho = float(getattr(sim, 'noise_ar1_rho', 0.0))
    print(
        f"\033[93m[noise] noise_model_level={sim.noise_model_level}  "
        f"measurement_noise_level={sim.measurement_noise_level}  "
        f"noise_ar1_rho={_ar1_rho:.3f} "
        f"({'AR(1) ENABLED' if _ar1_rho > 0 else 'i.i.d.'})\033[0m",
        flush=True,
    )

    run = 0

    os.makedirs(graphs_data_path("fly"), exist_ok=True)
    folder = graphs_data_path(config.dataset) + "/"
    print(f"\033[93m[data folder] {folder}\033[0m", flush=True)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(graphs_data_path(config.dataset, "Fig"), exist_ok=True)
    files = glob.glob(graphs_data_path(config.dataset, "Fig", "*"))
    for f in files:
        os.remove(f)

    # extent=15 → 721 retinotopic columns (5768 photoreceptors); extent=8 → 217 columns (1736 photoreceptors)
    extent = 15 if getattr(sim, 'all_columns', False) else 8

    # flyvis.__init__ sets root logger to INFO via basicConfig — restore to WARNING
    import logging

    from flyvis import Network, NetworkView
    from flyvis.datasets.sintel import AugmentedSintel
    from flyvis.utils.config_utils import CONFIG_PATH, get_default_config

    from connectome_gnn.generators.flyvis_ode import (
        FlyVisODE,
        get_photoreceptor_positions_from_net,
        group_by_direction_and_function,
    )
    from connectome_gnn.generators.ode_params import FlyVisHodgkinHuxleyODEParams, FlyVisODEParams, get_ode_params_class
    from connectome_gnn.utils import setup_flyvis_model_path

    is_hh = is_hodgkin_huxley_model(model_config.signal_model_name)

    logging.getLogger().setLevel(logging.WARNING)
    setup_flyvis_model_path()

    # Initialize the flyvis network first (fast) so we can print actual network stats before
    # the slow stimulus rendering begins.
    import logging as _logging
    _logging.getLogger("flyvis.utils.logging_utils").setLevel(_logging.ERROR)

    if is_flyvis_hybrid_model(model_config.signal_model_name):
        # --- Flywirevis hybrid: load pre-computed connectome tables ---
        from connectome_gnn.generators.hybrid_connectome import load_hybrid_network

        signal_name = model_config.signal_model_name
        edge_uncertainty = getattr(sim, "edge_uncertainty", 1)
        flyvis_model_id = f"flow/{sim.ensemble_id}/{sim.model_id}"
        logger.info(f"loading hybrid network ({signal_name}, extent={extent}, u={edge_uncertainty})...")
        net, _orig_net = load_hybrid_network(
            signal_name=signal_name,
            extent=extent,
            edge_uncertainty=edge_uncertainty,
            model=flyvis_model_id,
        )
        logger.info(
            f"hybrid network: {net.connectome.nodes.type[:].shape[0]} nodes, "
            f"{net.connectome.edges.source_index[:].shape[0]} edges"
        )
    else:
        assert "flywire" not in model_config.signal_model_name, "Should have taken the if-branch"
        # --- Standard flyvis network ---
        config_net = get_default_config(overrides=[], path=f"{CONFIG_PATH}/network/network.yaml")
        config_net.connectome.extent = extent
        net = Network(**config_net)
        nnv = NetworkView(f"flow/{sim.ensemble_id}/{sim.model_id}")
        trained_net = nnv.init_network(checkpoint=0)
        net.load_state_dict(trained_net.state_dict())
    torch.set_grad_enabled(False)

    _node_types_str = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in net.connectome.nodes["type"][:]]
    _photoreceptor_types = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8'}
    n_input_neurons_net = int(np.sum([t in _photoreceptor_types for t in _node_types_str]))
    print(f"  n_neurons:       {net.n_nodes}", flush=True)
    print(f"  n_input_neurons: {n_input_neurons_net}", flush=True)
    print(f"  n_edges:         {net.n_edges}", flush=True)

    # Stimulus filter: regular flyvis hex disk (default) or, for FlyWire
    # hybrids, a *standard* hex disk large enough to contain every
    # FlyWire input column, plus an index that projects the rendered
    # standard-hex frame to the FlyWire column subset.  Rendering at the
    # standard hex disk lets us apply flyvis's HexFlip/HexRotate
    # augmentations unchanged (they require a regular lattice); the
    # projection then maps each augmented frame onto the actual FlyWire
    # input columns at ``Stimulus.add_input`` time.
    if getattr(sim, 'flywire_stimulus', False):
        from connectome_gnn.generators.flywire_eye import (
            standard_boxeye_and_flywire_index,
        )
        _be, _flywire_proj_idx, _be_extent = standard_boxeye_and_flywire_index(
            net, kernel_size=13,
        )
        boxfilter_arg = dict(extent=_be_extent, kernel_size=13)
        print(
            f"[stimulus] flywire_stimulus=True: rendering at standard "
            f"BoxEye(extent={_be_extent}) with {_be.hexals} hexals; "
            f"projecting to {_flywire_proj_idx.numel()} FlyWire columns",
            flush=True,
        )
        # Monkey-patch ``net.stimulus.add_input`` to project standard-hex
        # frames down to FlyWire columns. This way every existing call
        # site (including those inside top-level helpers like
        # ``_run_ode_generation``) works unchanged.
        _orig_add_input = net.stimulus.add_input
        def _patched_add_input(frame, *args, **kwargs):
            idx = _flywire_proj_idx.to(frame.device)
            return _orig_add_input(frame.index_select(-1, idx), *args, **kwargs)
        net.stimulus.add_input = _patched_add_input
    else:
        boxfilter_arg = dict(extent=extent, kernel_size=13)

    # Initialize datasets
    print(f"[DBG] visual_input_type={sim.visual_input_type!r}  datavis_roots={sim.datavis_roots}", flush=True)
    if "DAVIS" in sim.visual_input_type or "mixed" in sim.visual_input_type:
        # determine dataset roots: use config list if provided, otherwise fall back to default
        if sim.datavis_roots:
            datavis_root_list = [os.path.join(r, "JPEGImages/480p") for r in sim.datavis_roots]
        else:
            datavis_root_list = [os.path.join(get_datavis_root_dir(), "JPEGImages/480p")]

        print(f"[DBG] datavis_root_list={datavis_root_list}", flush=True)
        for root in datavis_root_list:
            print(f"[DBG] checking root exists: {root}", flush=True)
            assert os.path.exists(root), f"video data not found at {root}"
            print(f"[DBG]   OK exists", flush=True)

        video_config = {
            "n_frames": 50,
            "max_frames": sim.truncate_max_frames,  # None = no per-clip truncation
            # Hex rotate/flip rely on a regular hex-disk lattice; the FlyWire
            # column lattice is irregular, so disable them when rendering on it.
            # (augment=False already neutralises them at runtime, but this also
            # keeps the construction path safe for any future augment toggle.)
            # HexFlip/HexRotate operate on the standard hex lattice the
            # frames are rendered on (BoxEye extent above). For FlyWire
            # mode we render at a standard disk and project later, so
            # the same 8x augmentation factor applies to both modes.
            "flip_axes": [0, 1],
            "n_rotations": [0, 90, 180, 270],
            "temporal_split": False,
            "dt": sim.delta_t,
            "boxfilter": boxfilter_arg,
            "vertical_splits": 1,
            "center_crop_fraction": 0.6,
            "augment": False,
            "unittest": False,
            "skip_short_videos": sim.skip_short_videos,
            "shuffle_sequences": True,
            "shuffle_seed": sim.seed,
        }
        print(f"[DBG] video_config built (skip_short={sim.skip_short_videos} max_frames={sim.truncate_max_frames} seed={sim.seed})", flush=True)

        # create dataset(s)
        if len(datavis_root_list) == 1:
            print(f"[DBG] creating AugmentedVideoDataset(root_dir={datavis_root_list[0]}) ...", flush=True)
            davis_dataset = AugmentedVideoDataset(root_dir=datavis_root_list[0], **video_config)
            print(f"[DBG] AugmentedVideoDataset ready: {len(davis_dataset)} sequences", flush=True)
        else:
            print(f"[DBG] creating {len(datavis_root_list)} AugmentedVideoDatasets (combined) ...", flush=True)
            datasets = [AugmentedVideoDataset(root_dir=root, **video_config) for root in datavis_root_list]
            davis_dataset = CombinedVideoDataset(datasets)
            logger.info(f"combined {len(datasets)} video datasets: {len(davis_dataset)} total sequences")
    else:
        davis_dataset = None

    if "DAVIS" in sim.visual_input_type:
        stimulus_dataset = davis_dataset
        print(f"[DBG] using DAVIS-branch dataset: {len(stimulus_dataset)} sequences", flush=True)
    else:
        sintel_config = {
            "n_frames": 19,
            "flip_axes": [0, 1],
            "n_rotations": [0, 1, 2, 3, 4, 5],
            "temporal_split": True,
            "dt": sim.delta_t,
            "interpolate": True,
            "boxfilter": boxfilter_arg,
            "vertical_splits": 3,
            "center_crop_fraction": 0.7,
        }
        print(f"[DBG] creating AugmentedSintel(...) ...", flush=True)
        stimulus_dataset = AugmentedSintel(**sintel_config)
        print(f"[DBG] AugmentedSintel ready: {len(stimulus_dataset)} sequences", flush=True)

    # Extract ground-truth parameters from flyvis connectome.
    print(f"[DBG] extracting ODE params from flyvis network (is_hh={is_hh}) ...", flush=True)
    if is_hh:
        hh_overrides = {}
        if getattr(sim, "hh_stim_scale", None) is not None:
            hh_overrides["stim_scale"] = sim.hh_stim_scale
        if getattr(sim, "hh_I_bias", None) is not None:
            hh_overrides["I_bias"] = sim.hh_I_bias
        if getattr(sim, "hh_w_scale", None) is not None:
            hh_overrides["w_scale"] = sim.hh_w_scale
        ode_params = FlyVisHodgkinHuxleyODEParams.from_flyvis_network(
            net, device=device, overrides=hh_overrides or None
        )
    else:
        ode_params = FlyVisODEParams.from_flyvis_network(net, device=device)
    edge_index = ode_params.edge_index.to(device)
    print(f"[DBG] ODE params ready (edges={edge_index.shape[1]})", flush=True)

    if sim.n_extra_null_edges > 0:
        logger.info(f"adding {sim.n_extra_null_edges} extra null edges (mode={sim.null_edges_mode})...")
        import random

        src_np = edge_index[0].cpu().numpy()
        dst_np = edge_index[1].cpu().numpy()
        existing_edges = set(zip(src_np, dst_np))
        extra_edges = []

        if sim.null_edges_mode == "per_column":
            # Per pre-synaptic neuron: add a proportional number of false targets
            # Compute out-degree per source neuron
            from collections import Counter

            out_degree = Counter(src_np.tolist())
            total_real = edge_index.shape[1]
            ratio = sim.n_extra_null_edges / total_real

            # Build per-neuron target sets for fast lookup
            targets_by_source = {}
            for s, d in zip(src_np, dst_np):
                targets_by_source.setdefault(int(s), set()).add(int(d))

            all_neurons = list(range(n_neurons))
            for source in range(n_neurons):
                deg = out_degree.get(source, 0)
                if deg == 0:
                    continue
                n_false = max(1, int(round(deg * ratio)))
                existing_targets = targets_by_source.get(source, set())
                # Sample false targets not already connected and not self
                candidates = [t for t in all_neurons if t != source and t not in existing_targets]
                if len(candidates) <= n_false:
                    chosen = candidates
                else:
                    chosen = random.sample(candidates, n_false)
                for t in chosen:
                    extra_edges.append([source, t])
                    existing_targets.add(t)

            logger.info(
                f"per_column: added {len(extra_edges)} false edges "
                f"(requested ratio {ratio:.2f}, effective {len(extra_edges) / total_real:.2f})"
            )
        else:
            # Random: sample uniformly across the full matrix
            max_attempts = sim.n_extra_null_edges * 10
            attempts = 0
            while len(extra_edges) < sim.n_extra_null_edges and attempts < max_attempts:
                source = random.randint(0, n_neurons - 1)
                target = random.randint(0, n_neurons - 1)
                if (source, target) not in existing_edges and source != target:
                    extra_edges.append([source, target])
                    existing_edges.add((source, target))
                attempts += 1

        if extra_edges:
            extra_edge_index = torch.tensor(extra_edges, dtype=torch.long, device=device).t()
            edge_index = torch.cat([edge_index, extra_edge_index], dim=1)
            ode_params.edge_index = edge_index
            ode_params.W = torch.cat([ode_params.W, torch.zeros(len(extra_edges), device=device)])
            logger.info(f"Total extra edges added: {len(extra_edges)}")

    # Edge ablation: zero out a fraction of edge weights before ODE simulation
    ablation_mask = None
    if sim.ablation_ratio > 0:
        rng = np.random.RandomState(sim.ablation_seed)
        n_edges = edge_index.shape[1]
        n_ablate = int(np.round(n_edges * sim.ablation_ratio))
        ablate_indices = rng.choice(n_edges, size=n_ablate, replace=False)
        ablation_mask = torch.ones(n_edges, dtype=torch.bool, device=device)
        ablation_mask[ablate_indices] = False
        ode_params.W[~ablation_mask] = 0.0
        logger.info(f"ablated {n_ablate}/{n_edges} edges ({sim.ablation_ratio * 100:.0f}%)")

    if is_hh:
        from connectome_gnn.generators.flyvis_hodgkin_huxley_ode import FlyVisHodgkinHuxleyODE

        pde = FlyVisHodgkinHuxleyODE(ode_params=ode_params, device=device)
        p = ode_params
        logger.info(
            f"[HH] params: g_L={p.g_L[0]:.2f} E_L={p.E_L[0]:.1f} g_Na={p.g_Na[0]:.0f} E_Na={p.E_Na[0]:.0f} "
            f"g_K={p.g_K[0]:.0f} E_K={p.E_K[0]:.0f} C={p.C[0]:.1f} (mS/cm2, mV, uF/cm2)"
        )
        logger.info(
            f"[HH] drive: I_bias={p.I_bias[0]:.1f} uA/cm2, stim_scale={p.stim_scale[0]:.1f}, "
            f"syn_v_half={p.syn_v_half[0]:.1f} mV, syn_slope={p.syn_slope[0]:.1f} mV"
        )
        logger.info(
            f"[HH] connectome: W range=[{p.W.min():.3f}, {p.W.max():.3f}] mean={p.W.mean():.4f} "
            f"nonzero={int((p.W != 0).sum())}/{len(p.W)} edges"
        )
    else:
        pde = FlyVisODE(
            ode_params=ode_params,
            g_phi=torch.nn.functional.relu,
            params=sim.params,
            model_type=model_config.signal_model_name,
            n_neuron_types=sim.n_neuron_types,
            device=device,
        )

    _G = '\033[92m'  # green
    _R = '\033[91m'  # red
    _X = '\033[0m'   # reset

    # Activity will be generated with the FULL connectivity below.
    # Edge removal (if any) is applied AFTER generation so that x_list/y_list
    # reflect the full-network dynamics. Only ode_params (the connectivity seen
    # by the GNN) is pruned, giving the GNN an incomplete adjacency matrix to
    # work with while the ground-truth activity it must predict is from the
    # full connectome. This tests whether the GNN can recover parameters
    # despite missing edges.
    print(f"{_G}[GENERATE] full connectivity: edge_index={edge_index.shape}  W={ode_params.W.shape}{_X}")

    # Per-neuron hexagonal Cartesian positions for the *full* network — every
    # node carries (u, v) in net.connectome.nodes, so we use the standard
    # x = u + 0.5*v, y = v*sqrt(3)/2 mapping for both photoreceptors and
    # non-retinal neurons. This replaces the previous behaviour where
    # non-retinal neurons received random equidistant positions; the spatial
    # NGP path (ngp_hidden_spatial=True) and any column-aware figure that
    # reads pos[hidden_ids] depend on this fix.
    from connectome_gnn.generators.flyvis_ode import (
        get_all_neuron_positions_from_net,
    )
    x_coords_all, y_coords_all, _u_all, _v_all = get_all_neuron_positions_from_net(net)
    # Keep the photoreceptor-only arrays available for downstream code that
    # filters on input neurons (visual SIREN initialisation, etc.).
    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)

    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types]
    # available node types: {'T5d', 'R3', 'T2a', 'TmY14', 'R7', 'CT1(Lo1)', 'Tm4', 'TmY10', 'T4d', 'L1', 'R1', 'R6', 'Am', 'T2', 'Tm5Y', 'L5', 'Tm20', 'L2', 'Mi4', 'Mi12', 'T4c', 'TmY4', 'CT1(M10)', 'TmY15', 'Lawf1', 'T1', 'TmY13', 'Tm5b', 'Tm28', 'L3', 'R8', 'L4', 'C3', 'Mi14', 'Tm2', 'R5', 'R2', 'Mi15', 'Tm9', 'Tm16', 'T5a', 'Mi3', 'TmY9', 'T4b', 'Mi9', 'Mi1', 'T5b', 'Tm1', 'Lawf2', 'C2', 'T4a', 'TmY3', 'Mi2', 'T3', 'TmY5a', 'Mi11', 'Tm5a', 'TmY18', 'Tm30', 'R4', 'Mi13', 'Tm5c', 'T5c', 'Mi10', 'Tm3'}
    grouped_types = np.array([group_by_direction_and_function(t) for t in node_types_str])
    _, node_types_int = np.unique(node_types, return_inverse=True)

    X1 = torch.tensor(
        np.stack((x_coords_all, y_coords_all), axis=1),
        dtype=torch.float32, device=device,
    )

    _ss_value = getattr(sim, 'steady_state_value', 0.5)
    state = net.steady_state(t_pre=2.0, dt=sim.delta_t, batch_size=1, value=_ss_value)
    initial_state = state.nodes.activity.squeeze().to(device)
    n_neurons = len(initial_state)

    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)

    # init neuron state x

    _init_calcium = torch.rand(n_neurons, dtype=torch.float32, device=device)

    if is_hh:
        # HH: initialize at resting potential with steady-state gates
        hh_state = pde.init_state(n_neurons)
        x = NeuronState(
            index=torch.arange(n_neurons, dtype=torch.long, device=device),
            pos=X1,
            voltage=hh_state.voltage,
            stimulus=net.stimulus().squeeze(),
            group_type=torch.tensor(grouped_types, dtype=torch.long, device=device),
            neuron_type=torch.tensor(node_types_int, dtype=torch.long, device=device),
            calcium=_init_calcium,
            fluorescence=sim.calcium_alpha * _init_calcium + sim.calcium_beta,
            noise=torch.zeros(n_neurons, dtype=torch.float32, device=device),
            hh_m=hh_state.hh_m,
            hh_h=hh_state.hh_h,
            hh_n=hh_state.hh_n,
        )
    else:
        x = NeuronState(
            index=torch.arange(n_neurons, dtype=torch.long, device=device),
            pos=X1,
            voltage=initial_state.to(device),
            stimulus=net.stimulus().squeeze().to(device),
            group_type=torch.tensor(grouped_types, dtype=torch.long, device=device),
            neuron_type=torch.tensor(node_types_int, dtype=torch.long, device=device),
            calcium=_init_calcium,
            fluorescence=sim.calcium_alpha * _init_calcium + sim.calcium_beta,
            noise=torch.zeros(n_neurons, dtype=torch.float32, device=device),
        )

    # --- Subdirectory-level train/test split ---
    # arg_df is aligned with cached_sequences (shuffle applied to both in _build).
    # Split by original_index so all augmentations of the same base video stay together.
    df = stimulus_dataset.arg_df
    original_indices = df["original_index"].values
    unique_videos = np.unique(original_indices)
    np.random.shuffle(unique_videos)
    n_train_vids = int(len(unique_videos) * 0.8)
    train_video_set = set(unique_videos[:n_train_vids])
    test_video_set = set(unique_videos[n_train_vids:])

    train_indices = [i for i, oi in enumerate(original_indices) if oi in train_video_set]
    test_indices = [i for i, oi in enumerate(original_indices) if oi in test_video_set]

    # Extract the actual video subdirectory names for logging
    train_video_names = sorted(set(df.iloc[train_indices]["name"].values))
    test_video_names = sorted(set(df.iloc[test_indices]["name"].values))

    # Verify exclusivity
    train_name_set = set(train_video_names)
    test_name_set = set(test_video_names)
    overlap = train_name_set & test_name_set
    assert len(overlap) == 0, f"TRAIN/TEST OVERLAP: {overlap}"
    logger.info(
        f"subdirectory split: {n_train_vids} train / {len(unique_videos) - n_train_vids} test videos"
        f"  ({len(train_indices)} train seqs, {len(test_indices)} test seqs)"
    )
    logger.info(f"overlap: {overlap} (must be empty)")

    # Build sequences lists for ODE generation
    train_sequences = [stimulus_dataset[i] for i in train_indices]
    test_sequences = [stimulus_dataset[i] for i in test_indices]

    # Optionally limit number of sequences for faster debugging
    if sim.max_train_sequences > 0:
        train_sequences = train_sequences[: sim.max_train_sequences]
        test_sequences = test_sequences[: max(1, sim.max_train_sequences // 4)]
        logger.info(
            f"max_train_sequences={sim.max_train_sequences}: using {len(train_sequences)} train, {len(test_sequences)} test sequences"
        )

    # Build metadata labels for preview plots (name, flip_ax, n_rot)
    train_meta = [(df.iloc[idx]["name"], df.iloc[idx]["flip_ax"], df.iloc[idx]["n_rot"]) for idx in train_indices]
    test_meta = [(df.iloc[idx]["name"], df.iloc[idx]["flip_ax"], df.iloc[idx]["n_rot"]) for idx in test_indices]

    # Plot preview for train and test splits
    frames_per_sequence = 35
    n_hexals = stimulus_dataset[0]["lum"].shape[-1]
    hex_x = x_coords[:n_hexals]
    hex_y = y_coords[:n_hexals]
    plot_sequence_preview(
        train_sequences,
        hex_x,
        hex_y,
        f"TRAIN: {len(train_sequences)} seqs from {n_train_vids} videos",
        os.path.join(folder, "shuffle_first_frames_train.png"),
        fig_style,
        metadata=train_meta,
        logger=logger,
    )
    plot_sequence_preview(
        test_sequences,
        hex_x,
        hex_y,
        f"TEST: {len(test_sequences)} seqs from {len(test_video_set)} videos",
        os.path.join(folder, "shuffle_first_frames_test.png"),
        fig_style,
        metadata=test_meta,
        logger=logger,
    )

    # --- Generate TRAIN split ---
    total_frames_per_pass = len(train_sequences) * frames_per_sequence

    repeat_factor = max(1, int(getattr(sim, 'repeat_short_sequence_factor', 1)))
    if n_frames == 0:
        num_passes_needed = 1
        target_frames = float("inf")
        logger.info(f"n_frames=0 mode: single pass through {len(train_sequences)} train sequences")
    else:
        # When tiling a short unique block, only generate n_frames // factor
        # frames; the helper below replicates them across the full n_frames.
        target_frames = n_frames // repeat_factor if repeat_factor > 1 else n_frames
        num_passes_needed = (target_frames // total_frames_per_pass) + 1

    if repeat_factor > 1:
        logger.info(f"generating TRAIN data ({target_frames} unique frames, will tile ×{repeat_factor} → {target_frames * repeat_factor} total)...")
    else:
        logger.info(f"generating TRAIN data ({target_frames} frames from {len(train_sequences)} sequences)...")

    x_writer = ZarrSimulationWriterV3(
        path=graphs_data_path(config.dataset, "x_list_train"),
        n_neurons=n_neurons,
        time_chunks=2000,
        save_calcium=sim.save_calcium,
    )
    y_writer = ZarrArrayWriter(
        path=graphs_data_path(config.dataset, "y_list_train"),
        n_neurons=n_neurons,
        n_features=1,
        time_chunks=2000,
    )

    it, id_fig = _run_ode_generation(
        stimulus_sequences=train_sequences,
        net=net,
        pde=pde,
        x=x,
        edge_index=edge_index,
        initial_state=initial_state,
        sim=sim,
        x_writer=x_writer,
        y_writer=y_writer,
        target_frames=target_frames,
        num_passes=num_passes_needed,
        n_neurons=n_neurons,
        device=device,
        to_numpy_fn=to_numpy,
        noise_model_level=sim.noise_model_level,
        measurement_noise_level=sim.measurement_noise_level,
        visualize=visualize,
        run=run,
        run_vizualized=run_vizualized,
        step=step,
        id_fig_start=0,
        it_start=sim.start_frame,
        fig_style=fig_style,
        config=config,
        davis_dataset=davis_dataset,
        X1=X1,
        u_coords=u_coords,
        v_coords=v_coords,
    )

    n_frames_train = x_writer.finalize()
    y_writer.finalize()
    logger.info(f"generated {n_frames_train} TRAIN frames (saved as .zarr)")

    # --- Compute noisy derivatives for TRAIN split ---
    if sim.measurement_noise_level > 0:
        _compute_noisy_derivatives(config, sim, n_neurons, split="train")

    # --- Tile unique block ×factor across all dynamic train fields ---
    if repeat_factor > 1:
        _tile_train_zarrs(config, repeat_factor, save_calcium=sim.save_calcium)
        # Reflect the post-tile length in the generation log so _have_data
        # validates the on-disk zarr without flagging it as incomplete.
        n_frames_train = n_frames_train * repeat_factor

    # --- Generate TEST split ---
    # Default: test data is deterministic (sim.noisy_test_data=False) so that rollout
    # comparison against ground truth reflects the model's dynamics, not observation
    # noise. Set sim.noisy_test_data=True to keep train-level noise on the test split
    # (e.g. for figures that show the noisy stimulus-response trace the model saw).
    test_noise_model = sim.noise_model_level if sim.noisy_test_data else 0.0
    test_noise_meas = sim.measurement_noise_level if sim.noisy_test_data else 0.0

    # Reset neural state to avoid train→test leakage
    if is_hh:
        hh_state = pde.init_state(n_neurons)
        x.voltage = hh_state.voltage
        x.hh_m = hh_state.hh_m
        x.hh_h = hh_state.hh_h
        x.hh_n = hh_state.hh_n
    else:
        x.voltage[:] = initial_state
    _init_calcium = torch.rand(n_neurons, dtype=torch.float32, device=device)
    x.calcium = _init_calcium
    x.fluorescence = sim.calcium_alpha * _init_calcium + sim.calcium_beta

    # Test: single pass through test sequences, capped at MAX_TEST_FRAMES (or sim.n_frames_test if set)
    MAX_TEST_FRAMES = 8000
    _n_frames_test_cap = getattr(sim, 'n_frames_test', 0)
    test_target_frames = (_n_frames_test_cap if _n_frames_test_cap > 0 else MAX_TEST_FRAMES)
    test_target = len(test_sequences) * frames_per_sequence
    logger.info(f"generating TEST data (capped at {test_target_frames} frames from {len(test_sequences)} sequences)...")

    x_writer = ZarrSimulationWriterV3(
        path=graphs_data_path(config.dataset, "x_list_test"),
        n_neurons=n_neurons,
        time_chunks=2000,
        save_calcium=sim.save_calcium,
    )
    y_writer = ZarrArrayWriter(
        path=graphs_data_path(config.dataset, "y_list_test"),
        n_neurons=n_neurons,
        n_features=1,
        time_chunks=2000,
    )

    _run_ode_generation(
        stimulus_sequences=test_sequences, net=net, pde=pde, x=x,
        edge_index=edge_index, initial_state=initial_state, sim=sim,
        x_writer=x_writer, y_writer=y_writer,
        target_frames=test_target_frames, num_passes=1,
        n_neurons=n_neurons, device=device, to_numpy_fn=to_numpy,
        noise_model_level=test_noise_model,
        measurement_noise_level=test_noise_meas,
        visualize=False, run=run, run_vizualized=run_vizualized,
        step=step, id_fig_start=id_fig, it_start=0,
        fig_style=fig_style, config=config, davis_dataset=davis_dataset,
        X1=X1, u_coords=u_coords, v_coords=v_coords,
    )

    n_frames_test = x_writer.finalize()
    y_writer.finalize()
    _noise_tag = (
        f"noisy (noise_model={test_noise_model:g}, meas={test_noise_meas:g})"
        if sim.noisy_test_data else "without noise"
    )
    logger.info(f"generated {n_frames_test} TEST frames {_noise_tag} (saved as .zarr)")
    if sim.noisy_test_data:
        # Marker for consumers that need to know the test split carries train-level noise
        open(graphs_data_path(config.dataset, "noisy_test_data.ok"), "w").close()

    # --- Compute noisy derivatives for TEST split (mirrors TRAIN) ---
    if sim.noisy_test_data and sim.measurement_noise_level > 0:
        _compute_noisy_derivatives(config, sim, n_neurons, split="test")

    # restore gradient computation now (before any early-return paths)
    torch.set_grad_enabled(True)

    # --- Edge removal: applied AFTER activity generation ---
    # Activity data (x_list, y_list) was generated with the full connectome above.
    # Now prune ode_params so the GNN only sees the incomplete adjacency matrix.
    print(f"{_G}[GENERATE] activity generated with full connectivity: "
          f"edge_index={edge_index.shape}  W={ode_params.W.shape}{_X}")
    if sim.edge_removal_ratio > 0:
        if save:
            torch.save(ode_params.W.clone(), graphs_data_path(config.dataset, "weights_full.pt"))
            torch.save(edge_index.clone(), graphs_data_path(config.dataset, "edge_index_full.pt"))

        n_total = edge_index.shape[1]
        edge_mask_path = getattr(sim, 'edge_mask_path', '')
        if edge_mask_path and os.path.exists(edge_mask_path):
            kept_indices = torch.load(edge_mask_path, weights_only=True)
            print(f"{_G}[GENERATE] mask loaded from {edge_mask_path}: "
                  f"{len(kept_indices)}/{n_total} edges kept{_X}")
        else:
            if edge_mask_path:
                print(f"{_R}[GENERATE] edge_mask_path set but NOT FOUND: {edge_mask_path} "
                      f"— computing new mask{_X}")
            else:
                print(f"{_G}[GENERATE] no edge_mask_path — computing new mask "
                      f"(mode={getattr(sim,'edge_removal_mode','random')}, "
                      f"ratio={sim.edge_removal_ratio}){_X}")
            rng_rm = np.random.RandomState(sim.edge_removal_seed)
            removal_mode = getattr(sim, 'edge_removal_mode', 'random')
            if removal_mode == 'per_column':
                src_np = edge_index[0].cpu().numpy()
                keep_mask = np.ones(n_total, dtype=bool)
                for source in np.unique(src_np):
                    source_edges = np.where(src_np == source)[0]
                    n_remove = max(1, int(round(len(source_edges) * sim.edge_removal_ratio)))
                    if n_remove >= len(source_edges):
                        n_remove = len(source_edges) - 1
                    remove_idx = rng_rm.choice(source_edges, n_remove, replace=False)
                    keep_mask[remove_idx] = False
                kept_indices = np.where(keep_mask)[0]
            else:
                n_keep = int(n_total * (1 - sim.edge_removal_ratio))
                kept_indices = np.sort(rng_rm.choice(n_total, n_keep, replace=False))

        edge_index = edge_index[:, kept_indices]
        ode_params.edge_index = edge_index
        ode_params.W = ode_params.W[kept_indices]
        pct_removed = (1 - len(kept_indices) / n_total) * 100
        expected_pct = sim.edge_removal_ratio * 100
        color = _G if abs(pct_removed - expected_pct) < 2 else _R
        print(f"{color}[GENERATE] ode_params pruned: edge_index={edge_index.shape}  "
              f"W={ode_params.W.shape}  removed={pct_removed:.1f}% "
              f"(expected {expected_pct:.0f}%){_X}")
        if save:
            torch.save(torch.tensor(kept_indices, device=device),
                       graphs_data_path(config.dataset, "kept_edge_indices.pt"))
    else:
        print(f"{_G}[GENERATE] no edge removal (ratio=0){_X}")

    if save:
        ode_params.save(folder)
        print(f"{_G}[GENERATE] saved ode_params: edge_index={ode_params.edge_index.shape}  "
              f"W={ode_params.W.shape}  → {folder}{_X}")
        if ablation_mask is not None:
            torch.save(ablation_mask, graphs_data_path(config.dataset, "ablation_mask.pt"))

    # --- Always run diagnostics after data generation ---
    from connectome_gnn.zarr_io import load_raw_array, load_simulation_data

    x_ts = load_simulation_data(graphs_data_path(config.dataset, "x_list_train"))
    y_list = load_raw_array(graphs_data_path(config.dataset, "y_list_train"))
    activity_full = x_ts.voltage.numpy()  # (n_frames, n_neurons) — needed for noise plotting

    # Compute ranks (used in kinographs and traces)
    if compute_ranks:
        logger.info("computing effective rank ...")

        def _svd_lowrank(matrix_np, n_components, dev):
            """Randomized SVD via torch.svd_lowrank (GPU when available)."""
            t = torch.as_tensor(matrix_np, dtype=torch.float32, device=dev)
            _, S, _ = torch.svd_lowrank(t, q=n_components + 10, niter=4)
            return S[:n_components].cpu().numpy()

        _svd_device = torch.device(device) if (device and device != 'cpu' and torch.cuda.is_available()) else torch.device('cpu')
        logger.info(f"  SVD device: {_svd_device}")

        n_comp = min(50, min(activity_full.shape) - 1)
        S_act = _svd_lowrank(activity_full, n_comp, _svd_device)
        cumvar_act = np.cumsum(S_act**2) / np.sum(S_act**2)
        rank_90_act = int(np.searchsorted(cumvar_act, 0.90) + 1)
        rank_99_act = int(np.searchsorted(cumvar_act, 0.99) + 1)

        # Mean-centered rank: subtract per-neuron temporal mean to remove static bias pattern.
        activity_centered = activity_full - activity_full.mean(axis=0, keepdims=True)
        centered_var = np.sum(activity_centered**2)
        if centered_var > 1e-12:
            S_mc = _svd_lowrank(activity_centered, n_comp, _svd_device)
            cumvar_mc = np.cumsum(S_mc**2) / centered_var
            rank_90_mc = int(np.searchsorted(cumvar_mc, 0.90) + 1)
            rank_99_mc = int(np.searchsorted(cumvar_mc, 0.99) + 1)
        else:
            rank_90_mc = rank_99_mc = 0

        input_for_svd = x_ts.stimulus[:, : sim.n_input_neurons].numpy()
        n_comp_input = min(50, min(input_for_svd.shape) - 1)
        S_inp = _svd_lowrank(input_for_svd, n_comp_input, _svd_device)
        cumvar_inp = np.cumsum(S_inp**2) / np.sum(S_inp**2)
        rank_90_inp = int(np.searchsorted(cumvar_inp, 0.90) + 1)
        rank_99_inp = int(np.searchsorted(cumvar_inp, 0.99) + 1)

        logger.info(
            f"activity rank(90%)={rank_90_act}  rank(99%)={rank_99_act}  centered rank(90%)={rank_90_mc}  rank(99%)={rank_99_mc}"
        )
        logger.info(f"visual input rank(90%)={rank_90_inp}  rank(99%)={rank_99_inp}")

        # Build neuron-type labels for kinograph annotations
        act_labels = None
        stim_labels = None
        if hasattr(ode_params, "neuron_types") and ode_params.neuron_types is not None:
            nt = to_numpy(ode_params.neuron_types)
            tnames = getattr(ode_params, "type_names", None)
            if tnames is not None:
                act_labels = []
                for ti, name in enumerate(tnames):
                    idx = np.where(nt == ti)[0]
                    if len(idx) > 0:
                        act_labels.append((name, int(idx.min()), int(idx.max()) + 1))
                # Stimulus labels: find which neurons receive non-zero stimulus
                stim_np = x_ts.stimulus[:, : sim.n_input_neurons].numpy()
                stim_power = np.sum(stim_np**2, axis=0)  # (N,)
                stim_labels = []
                for ti, name in enumerate(tnames):
                    idx = np.where(nt == ti)[0]
                    active_idx = idx[stim_power[idx] > 1e-6] if idx.max() < len(stim_power) else np.array([])
                    if len(active_idx) > 0:
                        stim_labels.append((name, int(active_idx.min()), int(active_idx.max()) + 1))
                if not stim_labels:
                    stim_labels = None

        if act_labels:
            logger.info(f"kinograph act_labels: {act_labels}")
        if stim_labels:
            logger.info(f"kinograph stim_labels: {stim_labels}")

        logger.info("plotting kinograph ...")
        plot_kinograph(
            activity=activity_full.T,
            stimulus=x_ts.stimulus[:, : sim.n_input_neurons].numpy().T,
            output_path=graphs_data_path(config.dataset, "kinograph.png"),
            rank_90_act=rank_90_act,
            rank_99_act=rank_99_act,
            rank_90_inp=rank_90_inp,
            rank_99_inp=rank_99_inp,
            rank_90_mc=rank_90_mc,
            rank_99_mc=rank_99_mc,
            zoom_size=200,
            style=fig_style,
            act_labels=act_labels,
            stim_labels=stim_labels,
        )

    # Skip warmup frames (100ms / dt) and show 400ms window for all plots
    if visualize:
        warmup_ms = 100.0
        window_ms = 800.0
        warmup_frames = int(warmup_ms / sim.delta_t)
        window_frames = int(window_ms / sim.delta_t)
        activity_plot = activity_full[warmup_frames:] if activity_full.shape[0] > warmup_frames + 10 else activity_full
        stim_plot = (
            x_ts.stimulus[warmup_frames:, : sim.n_input_neurons].numpy()
            if x_ts.stimulus.shape[0] > warmup_frames + 10
            else x_ts.stimulus[:, : sim.n_input_neurons].numpy()
        )
        logger.info(
            f"plotting traces (warmup_skip={warmup_frames} frames={warmup_ms}ms, window={window_frames} frames={window_ms}ms, {activity_plot.shape[0]} frames available)"
        )

    # HH-specific spiking plots (detect spikes from voltage threshold crossings)
    if visualize and is_hh:
        logger.info("plotting HH spiking traces ...")
        # Use warmup-skipped data: (T, N) -> (N, T)
        voltage_NT = activity_plot.T
        stimulus_NT = stim_plot.T
        # Detect spikes: voltage crosses 0mV from below
        spike_raster = np.zeros_like(voltage_NT, dtype=bool)
        spike_raster[:, 1:] = (voltage_NT[:, 1:] > 0) & (voltage_NT[:, :-1] <= 0)
        # Infer E/I from connectome weights
        W_np = to_numpy(ode_params.W)
        src_np = to_numpy(ode_params.edge_index[0])
        sum_w = np.zeros(voltage_NT.shape[0])
        np.add.at(sum_w, src_np.astype(int), W_np)
        is_exc_np = sum_w >= 0

        plot_spiking_traces(
            voltage=voltage_NT,
            spike_raster=spike_raster,
            stimulus=stimulus_NT,
            is_excitatory=is_exc_np,
            type_list=node_types_int,
            output_path=graphs_data_path(config.dataset),
            n_input_neurons=sim.n_input_neurons,
            max_frames=20000,
            dt_ms=sim.delta_t,
            style=fig_style,
        )
        logger.info(f"saved HH spiking plots to {graphs_data_path(config.dataset)}")

    # Plot noisy activity traces using the same neurons + compute SNR
    snr_stats = None
    if sim.measurement_noise_level > 0:
        logger.debug("plot noisy activity traces ...")
        noise_data = x_ts.noise.numpy() if x_ts.noise is not None else None
        if noise_data is not None:
            noisy_activity = activity_full + noise_data  # (T, N)
            if visualize:
                plot_activity_traces(
                    activity=noisy_activity.T,
                    output_path=graphs_data_path(config.dataset, "activity_traces_noisy.png"),
                    n_traces=100,
                    max_frames=10000,
                    n_input_neurons=sim.n_input_neurons,
                    style=fig_style,
                    type_list=node_types_int,
                    dpi=300,
                    title="noisy voltage traces (measurement noise)",
                )

            # --- SNR analysis (per neuron) ---
            # Voltage SNR: std(clean_voltage) / std(measurement_noise) per neuron
            signal_std = np.std(activity_full, axis=0)  # (N,)
            noise_std = np.std(noise_data, axis=0)  # (N,)
            voltage_snr = np.where(noise_std > 0, signal_std / noise_std, np.inf)
            voltage_snr_finite = voltage_snr[np.isfinite(voltage_snr)]

            # Derivative SNR: std(clean_derivative) / std(derivative_noise) per neuron
            # derivative noise = (noise[t+1] - noise[t]) / dt
            deriv_noise = np.diff(noise_data, axis=0) / sim.delta_t  # (T-1, N)
            deriv_noise_std = np.std(deriv_noise, axis=0)  # (N,)
            y_clean = load_raw_array(graphs_data_path(config.dataset, "y_list_train"))  # (T, N, 1)
            deriv_signal_std = np.std(y_clean[:, :, 0], axis=0)  # (N,)
            deriv_snr = np.where(deriv_noise_std > 0, deriv_signal_std / deriv_noise_std, np.inf)
            deriv_snr_finite = deriv_snr[np.isfinite(deriv_snr)]

            deriv_noise_std_theoretical = sim.measurement_noise_level * np.sqrt(2) / sim.delta_t
            deriv_noise_std_empirical = np.mean(deriv_noise_std)

            snr_stats = {
                "voltage_snr_mean": np.mean(voltage_snr_finite),
                "voltage_snr_median": np.median(voltage_snr_finite),
                "voltage_snr_min": np.min(voltage_snr_finite),
                "voltage_snr_max": np.max(voltage_snr_finite),
                "derivative_snr_mean": np.mean(deriv_snr_finite),
                "derivative_snr_median": np.median(deriv_snr_finite),
                "derivative_snr_min": np.min(deriv_snr_finite),
                "derivative_snr_max": np.max(deriv_snr_finite),
                "derivative_noise_std_theoretical": deriv_noise_std_theoretical,
                "derivative_noise_std_empirical": deriv_noise_std_empirical,
            }

            logger.info("--- Measurement noise SNR analysis ---")
            logger.info("  voltage SNR (std_signal / std_noise) per neuron:")
            logger.info(
                f"    mean: {snr_stats['voltage_snr_mean']:.2f}  "
                f"median: {snr_stats['voltage_snr_median']:.2f}  "
                f"min: {snr_stats['voltage_snr_min']:.2f}  "
                f"max: {snr_stats['voltage_snr_max']:.2f}"
            )
            logger.info("  derivative SNR (std_dy/dt / std_noise_dy/dt) per neuron:")
            logger.info(
                f"    mean: {snr_stats['derivative_snr_mean']:.2f}  "
                f"median: {snr_stats['derivative_snr_median']:.2f}  "
                f"min: {snr_stats['derivative_snr_min']:.2f}  "
                f"max: {snr_stats['derivative_snr_max']:.2f}"
            )
            logger.info(f"  derivative noise std (theoretical): {snr_stats['derivative_noise_std_theoretical']:.2f}")
            logger.info(f"  derivative noise std (empirical mean): {snr_stats['derivative_noise_std_empirical']:.2f}")
            logger.info("--------------------------------------")

    # SVD analysis (4-panel plot)
    svd_results = {}
    if visualize:
        logger.info("svd analysis ...")
        from connectome_gnn.models.utils import analyze_data_svd

        folder = graphs_data_path(config.dataset)
        svd_results = analyze_data_svd(
            x_ts, folder, config=config, is_flyvis=True, save_in_subfolder=False, logger=logger
        )

    # Save ranks to log file
    gen_log_path = graphs_data_path(config.dataset, 'generation_log.txt')
    with open(gen_log_path, 'w') as log_f:
        log_f.write(f'dataset: {config.dataset}\n')
        log_f.write(f'n_neurons: {n_neurons}\n')
        log_f.write(f'n_input_neurons: {sim.n_input_neurons}\n')
        log_f.write(f'n_frames_train: {n_frames_train}\n')
        log_f.write(f'n_frames_test: {n_frames_test}\n')
        log_f.write(f'n_sequences_train: {len(train_sequences)}\n')
        log_f.write(f'n_sequences_test: {len(test_sequences)}\n')
        log_f.write(f'n_train_videos: {n_train_vids}\n')
        log_f.write(f'n_test_videos: {len(test_video_set)}\n')
        log_f.write(f'train_videos: {train_video_names}\n')
        log_f.write(f'test_videos: {test_video_names}\n')
        log_f.write(f'visual_input_type: {sim.visual_input_type}\n')
        if sim.datavis_roots:
            log_f.write(f'datavis_roots: {sim.datavis_roots}\n')
        log_f.write(f'noise_model_level: {sim.noise_model_level}\n')
        log_f.write(f'measurement_noise_level: {sim.measurement_noise_level}\n')
        log_f.write(f'model_id: {sim.model_id}\n')
        log_f.write(f'ensemble_id: {sim.ensemble_id}\n')
        log_f.write('\n')
        if compute_ranks:
            log_f.write(f"activity_rank_90: {rank_90_act}\n")
            log_f.write(f"activity_rank_99: {rank_99_act}\n")
            log_f.write(f"input_rank_90: {rank_90_inp}\n")
            log_f.write(f"input_rank_99: {rank_99_inp}\n")
        if svd_results.get("activity"):
            log_f.write(f"svd_activity_rank_90: {svd_results['activity']['rank_90']}\n")
            log_f.write(f"svd_activity_rank_99: {svd_results['activity']['rank_99']}\n")
        if svd_results.get("visual_stimuli"):
            log_f.write(f"svd_visual_rank_90: {svd_results['visual_stimuli']['rank_90']}\n")
            log_f.write(f"svd_visual_rank_99: {svd_results['visual_stimuli']['rank_99']}\n")
        if snr_stats is not None:
            log_f.write("\n")
            for key, val in snr_stats.items():
                log_f.write(f"{key}: {val:.2f}\n")
    logger.info(f"generation log saved to {gen_log_path}")

    if not visualize:
        return

    # Neuron type index to name mapping (CamelCase for legacy plot_neuron_activity_analysis)
    index_to_name = {
        0: "Am",
        1: "C2",
        2: "C3",
        3: "CT1(Lo1)",
        4: "CT1(M10)",
        5: "L1",
        6: "L2",
        7: "L3",
        8: "L4",
        9: "L5",
        10: "Lawf1",
        11: "Lawf2",
        12: "Mi1",
        13: "Mi10",
        14: "Mi11",
        15: "Mi12",
        16: "Mi13",
        17: "Mi14",
        18: "Mi15",
        19: "Mi2",
        20: "Mi3",
        21: "Mi4",
        22: "Mi9",
        23: "R1",
        24: "R2",
        25: "R3",
        26: "R4",
        27: "R5",
        28: "R6",
        29: "R7",
        30: "R8",
        31: "T1",
        32: "T2",
        33: "T2a",
        34: "T3",
        35: "T4a",
        36: "T4b",
        37: "T4c",
        38: "T4d",
        39: "T5a",
        40: "T5b",
        41: "T5c",
        42: "T5d",
        43: "Tm1",
        44: "Tm16",
        45: "Tm2",
        46: "Tm20",
        47: "Tm28",
        48: "Tm3",
        49: "Tm30",
        50: "Tm4",
        51: "Tm5Y",
        52: "Tm5a",
        53: "Tm5b",
        54: "Tm5c",
        55: "Tm9",
        56: "TmY10",
        57: "TmY13",
        58: "TmY14",
        59: "TmY15",
        60: "TmY18",
        61: "TmY3",
        62: "TmY4",
        63: "TmY5a",
        64: "TmY9",
    }

    activity = x_ts.voltage.to(device).t()  # (n_neurons, n_frames)
    type_list = x.neuron_type.unsqueeze(-1).to(device)

    target_type_name_list = ["R1", "R7", "C2", "Mi11", "Tm1", "Tm4", "Tm30"]
    from GNN_PlotFigure import plot_neuron_activity_analysis

    plot_neuron_activity_analysis(
        activity,
        target_type_name_list,
        type_list,
        index_to_name,
        n_neurons,
        n_frames,
        sim.delta_t,
        graphs_data_path(config.dataset) + "/",
    )

    logger.info("plot figure activity ...")
    plot_selected_neuron_traces(
        activity=to_numpy(activity),
        type_list=to_numpy(type_list.squeeze()),
        output_path=graphs_data_path(config.dataset, 'activity.png'),
        start_frame=0,
        end_frame=activity.shape[1],
        style=fig_style,
    )

    if visualize & (run == run_vizualized):
        logger.info("generating lossless video ...")

        output_name = config.dataset.split("flyvis_")[1] if "flyvis_" in config.dataset else "no_id"
        src = graphs_data_path(config.dataset, "Fig", "Fig_0_000000.png")
        dst = graphs_data_path(config.dataset, f"input_{output_name}.png")
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())

        generate_compressed_video_mp4(output_dir=graphs_data_path(config.dataset), run=run,
                                      output_name=output_name, framerate=10)

        files = glob.glob(graphs_data_path(config.dataset, "Fig", "*"))
        for f in files:
            os.remove(f)


def _run_ode_generation(
    stimulus_sequences,
    net,
    pde,
    x,
    edge_index,
    initial_state,
    sim,
    x_writer,
    y_writer,
    target_frames,
    num_passes,
    n_neurons,
    device,
    to_numpy_fn,
    noise_model_level: float,
    measurement_noise_level: float,
    visualize=False,
    run=0,
    run_vizualized=0,
    step=5,
    id_fig_start=0,
    it_start=0,
    fig_style=None,
    config=None,
    davis_dataset=None,
    X1=None,
    u_coords=None,
    v_coords=None,
):
    """Run ODE simulation over stimulus sequences, writing frames to zarr.

    This is the inner loop extracted so it can be called for both train and test.
    Returns (it, id_fig) — the final frame counter and figure counter.
    """
    it = it_start
    id_fig = id_fig_start

    tile_labels = None
    tile_codes_torch = None
    tile_period = None
    tile_idx = 0
    n_columns = sim.n_input_neurons // 8

    # Mixed sequence setup
    mixed_types_list = None
    if "mixed" in sim.visual_input_type:
        mixed_types_list = ["sintel", "davis", "blank", "noise"]
        mixed_cycle_lengths = [60, 60, 30, 60]
        mixed_current_type = 0
        mixed_frame_count = 0
        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
        sintel_iter = iter(stimulus_sequences)
        davis_iter = iter(davis_dataset) if davis_dataset else iter(stimulus_sequences)
        current_sintel_seq = None
        current_davis_seq = None
        sintel_frame_idx = 0
        davis_frame_idx = 0

    # Collect HH traces for diagnostic plot (hh_debug_seq0.png)
    _hh_debug_buffers = None
    _hh_debug_n_seqs = 30  # capture enough sequences for 400ms window
    if hasattr(pde, "step_gates"):
        _hh_debug_buffers = {"volt": [], "stim": [], "m": [], "h": [], "n": []}

    # Track per-sequence lengths so we can report a post-hoc summary.
    # Critical for blank_prefix diagnostics: blank_prefix_frames = int(seq_len *
    # blank_prefix_fraction), so a dataset full of 2-frame sequences gets
    # ~1 blank frame per sequence — not enough time for neurons to decay to
    # V_rest.
    _seq_lens = []

    # AR(1) measurement-noise state: persists across all frames/sequences within
    # this generator call. Recursion eta(t+1) = rho*eta(t) + sqrt(1-rho**2)*gamma*xi(t)
    # preserves marginal Var(eta) = gamma**2. rho = 0 -> standard i.i.d. (current default).
    ar1_rho = float(getattr(sim, 'noise_ar1_rho', 0.0))
    ar1_inject_std = (1.0 - ar1_rho ** 2) ** 0.5 * measurement_noise_level
    if measurement_noise_level > 0 and ar1_rho > 0:
        # Initialise in stationary distribution: Var(eta_0) = gamma**2
        ar1_prev_noise = (
            torch.randn(n_neurons, dtype=torch.float32, device=device)
            * measurement_noise_level
        )
    else:
        ar1_prev_noise = None

    # DAVIS blank-window injection (see SimulationConfig validator for compatibility).
    # State persists across video boundaries and across passes so the m-real / l-blank
    # pattern is preserved continuously.
    bw_size = int(getattr(sim, "blank_window_size_frames", 0))
    bw_every = int(getattr(sim, "blank_insertion_every_n_frames", 0))
    use_blank_injection = bw_size > 0 and bw_every > 0
    real_frames_consumed = 0
    real_frames_in_chunk = 0
    in_blank_window = False
    blank_remaining = 0

    target_reached = False
    with torch.no_grad():
        for pass_num in range(num_passes):
            for data_idx, data in enumerate(tqdm(stimulus_sequences, desc="processing stimulus data", ncols=100)):
                if sim.simulation_initial_state:
                    x.voltage[:] = initial_state
                    if sim.only_noise_visual_input > 0:
                        x.stimulus[: sim.n_input_neurons] = torch.clamp(
                            torch.relu(
                                0.5
                                + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device)
                                * sim.only_noise_visual_input
                                / 2
                            ),
                            0,
                            1,
                        )

                sequences = data["lum"]

                if "flash" in sim.visual_input_type:
                    flash_duration_options = [1, 2, 5]
                    flash_cycle_frames = flash_duration_options[
                        torch.randint(0, len(flash_duration_options), (1,), device=device).item()
                    ]
                    flash_intensity = torch.abs(torch.rand(sim.n_input_neurons, device=device) * 0.5 + 0.5)

                if mixed_types_list is not None:
                    if mixed_frame_count >= current_cycle_length:
                        mixed_current_type = (mixed_current_type + 1) % 4
                        mixed_frame_count = 0
                        current_cycle_length = mixed_cycle_lengths[mixed_current_type]
                    current_type = mixed_types_list[mixed_current_type]

                    if current_type == "sintel":
                        if current_sintel_seq is None or sintel_frame_idx >= current_sintel_seq["lum"].shape[0]:
                            try:
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                            except StopIteration:
                                sintel_iter = iter(stimulus_sequences)
                                current_sintel_seq = next(sintel_iter)
                                sintel_frame_idx = 0
                        sequences = current_sintel_seq["lum"]
                        start_frame = sintel_frame_idx
                    elif current_type == "davis":
                        if current_davis_seq is None or davis_frame_idx >= current_davis_seq["lum"].shape[0]:
                            try:
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                            except StopIteration:
                                davis_iter = iter(davis_dataset) if davis_dataset else iter(stimulus_sequences)
                                current_davis_seq = next(davis_iter)
                                davis_frame_idx = 0
                        sequences = current_davis_seq["lum"]
                        start_frame = davis_frame_idx
                    else:
                        start_frame = 0

                if "flash" in sim.visual_input_type:
                    sequence_length = 60
                else:
                    sequence_length = sequences.shape[0]

                blank_prefix_frames = int(sequence_length * getattr(sim, 'blank_prefix_fraction', 0.0))
                _seq_lens.append(int(sequence_length))

                frame_id = 0
                while frame_id < sequence_length:
                    if "flash" in sim.visual_input_type:
                        current_flash_frame = frame_id % (flash_cycle_frames * 2)
                        x.stimulus[:] = 0
                        if current_flash_frame < flash_cycle_frames:
                            x.stimulus[: sim.n_input_neurons] = flash_intensity
                    elif mixed_types_list is not None:
                        current_type = mixed_types_list[mixed_current_type]
                        if current_type == "blank":
                            x.stimulus[:] = 0
                        elif current_type == "noise":
                            x.stimulus[: sim.n_input_neurons] = torch.relu(
                                0.5 + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device) * 0.5
                            )
                        else:
                            actual_frame_id = (start_frame + frame_id) % sequences.shape[0]
                            frame = sequences[actual_frame_id][None, None]
                            net.stimulus.add_input(frame)
                            x.stimulus[:] = net.stimulus().squeeze()
                            if current_type == "sintel":
                                sintel_frame_idx += 1
                            elif current_type == "davis":
                                davis_frame_idx += 1
                        mixed_frame_count += 1
                    elif "tile_mseq" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            tile_labels_np = assign_columns_from_uv(
                                u_coords, v_coords, n_columns, random_state=sim.seed
                            )
                            base = mseq_bits(p=8, seed=sim.seed).astype(np.float32)
                            rng = np.random.RandomState(sim.seed)
                            phases = rng.randint(0, base.shape[0], size=n_columns)
                            tile_codes_np = np.stack([np.roll(base, ph) for ph in phases], axis=0)
                            tile_codes_torch = torch.from_numpy(tile_codes_np).to(device, dtype=torch.float32)
                            tile_labels = torch.from_numpy(tile_labels_np).to(device, dtype=torch.long)
                            tile_period = tile_codes_torch.shape[1]
                            tile_idx = 0

                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx),
                        )
                        col_vals_01 = 0.5 + (sim.tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[: sim.n_input_neurons] = col_vals_01[tile_labels]
                        tile_idx += 1
                    elif "tile_blue_noise" in sim.visual_input_type:
                        if tile_codes_torch is None:
                            tile_labels_np, col_centers = compute_column_labels(
                                u_coords, v_coords, n_columns, seed=sim.seed
                            )
                            try:
                                adj = build_neighbor_graph(col_centers, k=6)
                            except Exception:
                                from scipy.spatial.distance import pdist, squareform

                                D = squareform(pdist(col_centers))
                                nn = np.partition(D + np.eye(D.shape[0]) * 1e9, 1, axis=1)[:, 1]
                                radius = 1.3 * np.median(nn)
                                adj = [
                                    set(np.where((D[i] > 0) & (D[i] <= radius))[0].tolist())
                                    for i in range(len(col_centers))
                                ]

                            tile_labels = torch.from_numpy(tile_labels_np).to(device, dtype=torch.long)
                            tile_period = 257
                            tile_idx = 0

                            tile_codes_torch = torch.empty((n_columns, tile_period), dtype=torch.float32, device=device)
                            rng = np.random.RandomState(sim.seed)
                            for t in range(tile_period):
                                mask = greedy_blue_mask(adj, n_columns, target_density=0.5, rng=rng)
                                vals = np.where(mask, 1.0, -1.0).astype(np.float32)
                                tile_codes_torch[:, t] = torch.from_numpy(vals).to(device, dtype=torch.float32)

                        x.stimulus[:] = 0.5
                        col_vals_pm1 = tile_codes_torch[:, tile_idx % tile_period]
                        col_vals_pm1 = apply_pairwise_knobs_torch(
                            code_pm1=col_vals_pm1,
                            corr_strength=float(sim.tile_corr_strength),
                            flip_prob=float(sim.tile_flip_prob),
                            seed=int(sim.seed) + int(tile_idx),
                        )
                        col_vals_01 = 0.5 + (sim.tile_contrast * 0.5) * col_vals_pm1
                        x.stimulus[: sim.n_input_neurons] = col_vals_01[tile_labels]
                        tile_idx += 1
                    elif use_blank_injection and in_blank_window:
                        # DAVIS blank-window injection: zero stimulus and hold the video
                        # cursor (frame_id is not advanced this iteration).
                        x.stimulus[:] = 0
                    else:
                        frame = sequences[frame_id][None, None]
                        net.stimulus.add_input(frame)
                        if sim.only_noise_visual_input > 0:
                            if (sim.visual_input_type == "") | (it == 0) | ("50/50" in sim.visual_input_type):
                                x.stimulus[: sim.n_input_neurons] = torch.relu(
                                    0.5
                                    + torch.rand(sim.n_input_neurons, dtype=torch.float32, device=device)
                                    * sim.only_noise_visual_input
                                    / 2
                                )
                        else:
                            # legacy blank injection
                            if sim.blank_freq > 0:
                                if data_idx % sim.blank_freq > 0:
                                    x.stimulus[:] = net.stimulus().squeeze()
                                else:
                                    x.stimulus[:] = 0
                            else:
                                x.stimulus[:] = net.stimulus().squeeze()
                            if sim.noise_visual_input > 0:
                                x.stimulus[: sim.n_input_neurons] = (
                                    x.stimulus[: sim.n_input_neurons]
                                    + torch.randn(sim.n_input_neurons, dtype=torch.float32, device=device)
                                    * sim.noise_visual_input
                                )

                    # Blank prefix: force zero stimulus for the first N frames of each sequence
                    if blank_prefix_frames > 0 and frame_id < blank_prefix_frames:
                        x.stimulus[:sim.n_input_neurons] = 0

                    prev_calcium = x.calcium.clone() if x.calcium is not None else None

                    # HH models use substeps for numerical stability
                    hh_substeps = getattr(sim, "hh_substeps", 1)
                    has_gates = hasattr(pde, "step_gates")

                    if has_gates and hh_substeps > 1:
                        # Multiple substeps per stimulus frame (HH)
                        sub_dt = sim.delta_t / hh_substeps
                        for _sub in range(hh_substeps):
                            y = pde(x, edge_index, has_field=False)
                            dv = y.squeeze()
                            if noise_model_level > 0:
                                x.voltage = (
                                    x.voltage
                                    + sub_dt * dv
                                    + torch.randn(n_neurons, dtype=torch.float32, device=device)
                                    * noise_model_level
                                    / (hh_substeps**0.5)
                                )
                            else:
                                x.voltage = x.voltage + sub_dt * dv
                            pde.step_gates(x, sub_dt)
                        # y for recording is the last substep's derivative
                        y = pde(x, edge_index, has_field=False)
                    else:
                        y = pde(x, edge_index, has_field=False)
                        dv_step = y.squeeze()

                    # Generate measurement noise for this timestep.
                    # AR(1) recursion when noise_ar1_rho > 0; falls back to i.i.d. otherwise.
                    if measurement_noise_level > 0:
                        if ar1_rho > 0:
                            ar1_prev_noise = (
                                ar1_rho * ar1_prev_noise
                                + torch.randn(n_neurons, dtype=torch.float32, device=device)
                                * ar1_inject_std
                            )
                            x.noise = ar1_prev_noise.clone()
                        else:
                            x.noise = (
                                torch.randn(n_neurons, dtype=torch.float32, device=device) * measurement_noise_level
                            )
                    else:
                        x.noise = torch.zeros(n_neurons, dtype=torch.float32, device=device)

                    # Save x[t] BEFORE updating voltage to x[t+1]
                    x_writer.append_state(x)

                    if not (has_gates and hh_substeps > 1):
                        if noise_model_level > 0:
                            x.voltage = (
                                x.voltage
                                + sim.delta_t * dv_step
                                + torch.randn(n_neurons, dtype=torch.float32, device=device) * noise_model_level
                            )
                        else:
                            x.voltage = x.voltage + sim.delta_t * dv_step
                        if has_gates:
                            pde.step_gates(x, sim.delta_t)

                    # Collect traces for first N sequences (for hh_debug plot)
                    if _hh_debug_buffers is not None and data_idx < _hh_debug_n_seqs and pass_num == 0:
                        _hh_debug_buffers["volt"].append(x.voltage.cpu().numpy().copy())
                        _hh_debug_buffers["stim"].append(x.stimulus.cpu().numpy().copy())
                        _hh_debug_buffers["m"].append(x.hh_m.cpu().numpy().copy())
                        _hh_debug_buffers["h"].append(x.hh_h.cpu().numpy().copy())
                        _hh_debug_buffers["n"].append(x.hh_n.cpu().numpy().copy())

                    if sim.calcium_type == "leaky":
                        if sim.calcium_activation == "softplus":
                            s = torch.nn.functional.softplus(x.voltage)
                        elif sim.calcium_activation == "relu":
                            s = torch.nn.functional.relu(x.voltage)
                        elif sim.calcium_activation == "tanh":
                            s = 1 + torch.tanh(x.voltage)
                        elif sim.calcium_activation == "identity":
                            s = x.voltage.clone()

                        x.calcium = x.calcium + (sim.delta_t / sim.calcium_tau) * (-x.calcium + s)
                        x.fluorescence = sim.calcium_alpha * x.calcium + sim.calcium_beta
                        y = ((x.calcium - prev_calcium) / sim.delta_t).unsqueeze(-1)

                    y_writer.append(to_numpy_fn(y.clone().detach()))

                    if (visualize & (run == run_vizualized) & (it > 0) & (it % 4 == 0) & (it <= 400)):
                        num = f"{id_fig:06}"
                        id_fig += 1
                        plot_spatial_activity_grid(
                            positions=to_numpy_fn(X1),
                            voltages=to_numpy_fn(x.voltage),
                            stimulus=to_numpy_fn(x.stimulus[: sim.n_input_neurons]),
                            neuron_types=to_numpy_fn(x.neuron_type).astype(int),
                            output_path=graphs_data_path(config.dataset, "Fig", f"Fig_{run}_{num}.png"),
                            calcium=to_numpy_fn(x.calcium) if sim.calcium_type != "none" else None,
                            n_input_neurons=sim.n_input_neurons,
                            style=fig_style,
                        )

                    # Advance the per-iteration cursors. With blank-window injection,
                    # frame_id (the video cursor) is held during blank iterations so
                    # blanks are inserted between real frames rather than replacing them.
                    if use_blank_injection:
                        if in_blank_window:
                            blank_remaining -= 1
                            if blank_remaining <= 0:
                                in_blank_window = False
                                real_frames_in_chunk = 0
                        else:
                            frame_id += 1
                            real_frames_consumed += 1
                            real_frames_in_chunk += 1
                            if real_frames_in_chunk >= bw_every:
                                in_blank_window = True
                                blank_remaining = bw_size
                                real_frames_in_chunk = 0
                    else:
                        frame_id += 1

                    it = it + 1
                    target_reached = (
                        real_frames_consumed if use_blank_injection else it
                    ) >= target_frames
                    if target_reached:
                        break
                # Save HH diagnostic plot after collecting enough sequences
                if (
                    _hh_debug_buffers is not None
                    and data_idx == _hh_debug_n_seqs - 1
                    and pass_num == 0
                    and _hh_debug_buffers["volt"]
                ):
                    logger.info(f"saving hh_debug_seq0.png ({len(_hh_debug_buffers['volt'])} frames)")
                    # Build HH params dict for current decomposition plot
                    _hh_plot_params = None
                    if hasattr(pde, "ode_params"):
                        _pp = pde.ode_params
                        _hh_plot_params = {
                            k: getattr(_pp, k).cpu().numpy()
                            for k in ("g_L", "E_L", "g_Na", "E_Na", "g_K", "E_K", "C", "I_bias", "stim_scale")
                            if hasattr(_pp, k) and getattr(_pp, k) is not None
                        }
                    _warmup_f = int(100.0 / sim.delta_t)  # 100ms warmup
                    _window_f = int(800.0 / sim.delta_t)  # 800ms window
                    plot_hh_debug(
                        voltage_history=np.stack(_hh_debug_buffers["volt"]),
                        stimulus_history=np.stack(_hh_debug_buffers["stim"]),
                        gate_m_history=np.stack(_hh_debug_buffers["m"]),
                        gate_h_history=np.stack(_hh_debug_buffers["h"]),
                        gate_n_history=np.stack(_hh_debug_buffers["n"]),
                        type_list=to_numpy_fn(x.neuron_type).astype(int),
                        output_path=graphs_data_path(config.dataset, "hh_debug_seq0.png"),
                        dt_ms=sim.delta_t,
                        hh_substeps=getattr(sim, "hh_substeps", 1),
                        hh_params=_hh_plot_params,
                        style=fig_style,
                        warmup_frames=_warmup_f,
                        max_frames=_window_f,
                    )
                    _hh_debug_buffers = None  # free memory

                if target_reached:
                    break
            if target_reached:
                break

    # Sequence-length summary (diagnostic for blank_prefix effectiveness).
    if _seq_lens:
        _arr = np.asarray(_seq_lens, dtype=np.int64)
        _bpf = float(getattr(sim, 'blank_prefix_fraction', 0.0))
        _bp_min = int(np.floor(_arr.min() * _bpf))
        _bp_med = int(np.floor(float(np.median(_arr)) * _bpf))
        _bp_max = int(np.floor(_arr.max() * _bpf))
        logger.info(
            "\033[93msequence-length summary: n_sequences=%d  frames=[min=%d median=%d mean=%.1f max=%d]  "
            "total_frames=%d  frames_consumed=%d\033[0m",
            int(_arr.size), int(_arr.min()), int(np.median(_arr)),
            float(_arr.mean()), int(_arr.max()), int(_arr.sum()), int(it - it_start),
        )
        logger.info(
            "\033[93mblank_prefix summary: blank_prefix_fraction=%.3f  blank_frames_per_seq=[min=%d median=%d max=%d]\033[0m",
            _bpf, _bp_min, _bp_med, _bp_max,
        )

    return it, id_fig


def _tile_train_zarrs(config, factor: int, save_calcium: bool):
    """Tile the unique train block `factor` times across every dynamic field.

    After data_generate_fly_voltage simulates n_frames // factor unique frames
    and finalises the writers, this helper re-opens each per-field store
    (voltage, stimulus, noise, calcium/fluorescence, y_clean, noisy_y) and
    writes (factor - 1) more copies, yielding a length-(unique_n * factor)
    dataset of the short trajectory repeated end-to-end.
    """
    import tensorstore as ts
    from pathlib import Path

    base = Path(graphs_data_path(config.dataset, "x_list_train"))
    fields = ['voltage', 'stimulus', 'noise']
    if save_calcium:
        fields += ['calcium', 'fluorescence']

    for name in fields:
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(base / f'{name}.zarr')},
        }
        store = ts.open(spec).result()
        unique = store.read().result()
        unique_n, N = unique.shape
        new_n = unique_n * factor
        store = store.resize(exclusive_max=[new_n, N]).result()
        for k in range(1, factor):
            store[unique_n * k: unique_n * (k + 1)].write(unique).result()

    # 3-D arrays: y_list_train and (if present) noisy_y_list_train
    for tag in ("y_list_train", "noisy_y_list_train"):
        path = Path(graphs_data_path(config.dataset, tag) + ".zarr")
        if not path.exists():
            continue
        spec = {'driver': 'zarr', 'kvstore': {'driver': 'file', 'path': str(path)}}
        store = ts.open(spec).result()
        unique = store.read().result()
        unique_n, N, F = unique.shape
        new_n = unique_n * factor
        store = store.resize(exclusive_max=[new_n, N, F]).result()
        for k in range(1, factor):
            store[unique_n * k: unique_n * (k + 1)].write(unique).result()

    logger.info(f"tiled train zarrs ×{factor}: unique_n={unique_n} → total={unique_n * factor}")


def _compute_noisy_derivatives(config, sim, n_neurons, split="train"):
    """Compute noisy derivatives from saved clean derivatives and noise.

    noisy_y[t] = y_clean[t] + (noise[t+1] - noise[t]) / dt
    Last frame uses clean derivative (no future noise available).
    """
    from connectome_gnn.utils import graphs_data_path
    from connectome_gnn.zarr_io import ZarrArrayWriter, load_raw_array, load_simulation_data

    y_clean = load_raw_array(graphs_data_path(config.dataset, f"y_list_{split}"))  # (T, N, 1)
    noise_ts = load_simulation_data(graphs_data_path(config.dataset, f"x_list_{split}"), fields=["noise"])
    noise = noise_ts.noise.numpy()  # (T, N)

    # Compute noise derivative: (noise[t+1] - noise[t]) / dt
    noise_diff = np.zeros_like(noise)
    noise_diff[:-1] = (noise[1:] - noise[:-1]) / sim.delta_t  # last frame: 0

    noisy_y = y_clean + noise_diff[:, :, np.newaxis]  # broadcast to (T, N, 1)

    # Temporal smoothing of noisy derivatives (reduces derivative noise by sqrt(window))
    window = sim.derivative_smoothing_window
    if window > 1:
        from scipy.ndimage import uniform_filter1d

        # Apply centered moving average along time axis (axis=0)
        # mode='nearest' pads boundaries with edge values
        noisy_y = uniform_filter1d(noisy_y, size=window, axis=0, mode="nearest")
        logger.debug(f"  applied derivative smoothing: window={window} (noise reduction ~{1 / np.sqrt(window):.2f}x)")

    noisy_y_writer = ZarrArrayWriter(
        path=graphs_data_path(config.dataset, f"noisy_y_list_{split}"),
        n_neurons=n_neurons,
        n_features=1,
        time_chunks=2000,
    )
    for t in range(noisy_y.shape[0]):
        noisy_y_writer.append(noisy_y[t])
    noisy_y_writer.finalize()
    logger.info(
        f"computed noisy derivatives for {split}: {noisy_y.shape[0]} frames "
        f"(measurement_noise_level={sim.measurement_noise_level})"
    )


# ============================================================================
# Voltage generation from a trained task-optimized TaskRNN
# ============================================================================

def _resolve_task_config_path(path_str: str) -> str:
    """Resolve a task-model yaml path.

    Resolution order:
      1. absolute path or relative-to-cwd file
      2. <repo>/config/<path_str>[.yaml]
      3. <repo>/config/<pre_folder>/<basename>[.yaml]  (matches GNN_Main's
         add_pre_folder routing so users can write `cortex_delaygo_winner`
         instead of `cortex/cortex_delaygo_winner`)
    """
    from connectome_gnn.utils import add_pre_folder, config_path
    if os.path.isabs(path_str) and os.path.isfile(path_str):
        return path_str
    if os.path.isfile(path_str):
        return os.path.abspath(path_str)
    candidates = []
    candidates.append(config_path(path_str))
    if not path_str.endswith(".yaml"):
        candidates.append(config_path(path_str + ".yaml"))
    # Apply add_pre_folder routing (e.g. 'cortex_delaygo_winner' -> 'cortex/cortex_delaygo_winner')
    try:
        cfg_file, _ = add_pre_folder(path_str)
        candidates.append(config_path(cfg_file))
        if not cfg_file.endswith(".yaml"):
            candidates.append(config_path(cfg_file + ".yaml"))
    except Exception:
        pass
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"task_model_config_path not found: {path_str}\n"
        f"  checked: {candidates}"
    )


def _resolve_task_checkpoint(task_cfg, task_cfg_path: str) -> str:
    """Find the latest best_model checkpoint for a task-trained TaskRNN."""
    from connectome_gnn.utils import add_pre_folder, log_path
    cf = task_cfg.config_file
    if cf in ("none", ""):
        stem = os.path.splitext(os.path.basename(task_cfg_path))[0]
        cf, _ = add_pre_folder(stem)
    task_log_dir = log_path(cf)
    ckpt_dir = os.path.join(task_log_dir, "models")
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "best_model_with_*_graphs_*.pt")))
    if not cands:
        raise FileNotFoundError(
            f"no task-model checkpoint found in {ckpt_dir}; train the task "
            f"model first via `GNN_Main.py -o train <task_config>`"
        )
    return cands[-1]


def _generate_voltage_from_task_model(
    config, *, device=None, visualize: bool = True
) -> None:
    """Generate voltage data by rolling out a trained TaskRNN over fresh task stimuli.

    Loads the TaskRNN described by `simulation.task_model_config_path`,
    runs it forward over freshly-sampled cortex trials, and stitches the
    hidden-state trajectory into a continuous (T, N) sequence. The
    on-disk format matches `data_generate_voltage` (ZarrSimulationWriterV3
    + ZarrArrayWriter) so a downstream `data_train_gnn` can train on the
    teacher's dynamics without any loader changes.

    Inputs:
      simulation.task_model_config_path : path to TaskRNN yaml (winner).
      simulation.n_frames               : target voltage frames for train split.
      simulation.seed                   : seed for trial sampling.

    Outputs (under graphs_data/<config.dataset>/):
      x_list_train/voltage.zarr      (T_train, N)
      x_list_train/stimulus.zarr     (T_train, N) — per-unit input drive
      x_list_train/pos.zarr          (N, 2)        — synthetic 2D grid
      x_list_train/group_type.zarr   (N,)          — zeros
      x_list_train/neuron_type.zarr  (N,)          — zeros
      y_list_train.zarr              (T_train, N, 1) — numerical dv/dt
      (and *_test under x_list_test / y_list_test at 25% of n_frames)
    """
    import torch

    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.generators.cortex_adapter import trial_to_numpy
    from connectome_gnn.generators.cortex_task import (
        generate_trials, get_default_hp,
    )
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.neuron_state import NeuronState
    from connectome_gnn.zarr_io import ZarrArrayWriter, ZarrSimulationWriterV3

    sim = config.simulation

    task_cfg_path = _resolve_task_config_path(sim.task_model_config_path)
    logger.info(f"[voltage_from_task] loading task config: {task_cfg_path}")
    task_cfg = NeuralGraphConfig.from_yaml(task_cfg_path)

    if device is None:
        from connectome_gnn.utils import set_device
        device = set_device(task_cfg.training.device)
    if isinstance(device, str):
        device = torch.device(device)

    model = create_model(
        task_cfg.graph_model.signal_model_name,
        aggr_type=task_cfg.graph_model.aggr_type,
        config=task_cfg, device=device,
    )
    ckpt_path = _resolve_task_checkpoint(task_cfg, task_cfg_path)
    logger.info(f"[voltage_from_task] loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    N = int(model.n_units)
    dt = float(getattr(model, "dt", 0.02))
    logger.info(
        f"[voltage_from_task] N={N}  dt={dt}s  task={task_cfg.task.cortex.rules}"
    )

    # Synthetic neuron metadata (TaskRNN has no biological positions or types):
    #   - pos: 2D grid (sqrt(N) × sqrt(N))
    #   - group_type, neuron_type: zeros
    n_side = int(np.ceil(np.sqrt(N)))
    grid = np.array(
        [[i // n_side, i % n_side] for i in range(N)],
        dtype=np.float32,
    ) / max(1, n_side - 1)
    pos = torch.from_numpy(grid).to(device)
    group_type_t = torch.zeros(N, dtype=torch.int32, device=device)
    neuron_type_t = torch.zeros(N, dtype=torch.int32, device=device)

    # Task-stimulus generator parameters from task config.
    ct = task_cfg.task.cortex
    hp = get_default_hp(ct.ruleset)
    if ct.hp_overrides:
        hp.update(ct.hp_overrides)
    rules = list(ct.rules)

    folder = graphs_data_path(config.dataset)
    os.makedirs(folder, exist_ok=True)
    print(f"\033[93m[voltage_from_task] writing to {folder}\033[0m", flush=True)

    # --- Save ground-truth ODE params for the downstream GNN ---
    # Map TaskRNN's W_rec / b / tau into the FlyVisODEParams schema so the
    # standard data_train_gnn loader picks them up unchanged. W_rec layout
    # is (rows=presynaptic j, cols=postsynaptic i) — np.nonzero returns
    # (row=src=pre, col=dst=post) which is exactly the edge_index
    # convention the GNN expects.
    from connectome_gnn.generators.ode_params import FlyVisODEParams
    W_rec_full = model.W_rec.detach().cpu().numpy().astype(np.float32)
    src, dst = np.nonzero(W_rec_full)
    edge_index_gt = np.stack([src, dst], axis=0).astype(np.int64)
    W_gt = W_rec_full[src, dst].astype(np.float32)
    tau_i_gt = np.full(N, float(model.tau), dtype=np.float32)
    V_i_rest_gt = model.b.detach().cpu().numpy().astype(np.float32)
    ode_params = FlyVisODEParams(
        tau_i=torch.from_numpy(tau_i_gt),
        V_i_rest=torch.from_numpy(V_i_rest_gt),
        edge_index=torch.from_numpy(edge_index_gt),
        W=torch.from_numpy(W_gt),
    )
    ode_params.save(folder)
    logger.info(
        f"[voltage_from_task] saved ode_params.pt: N={N}  E={W_gt.size}  "
        f"density={W_gt.size / (N * (N - 1)):.3f}  "
        f"tau={float(model.tau):.4f}  ||b||={float(np.linalg.norm(V_i_rest_gt)):.3f}"
    )

    # Dynamics-noise injection during rollout (mirrors data_generate_voltage's
    # sim.noise_model_level). The TaskRNN's forward already adds
    # `noise_recurrent_level · randn` per Euler step when the module is in
    # training mode; we route sim.noise_model_level through that field.
    # Test split is deterministic unless sim.noisy_test_data is True
    # (matches the standard generator's convention).
    base_noise = float(getattr(sim, "noise_model_level", 0.0))
    noisy_test = bool(getattr(sim, "noisy_test_data", False))
    print(
        f"\033[93m[noise] noise_model_level={base_noise}  "
        f"noisy_test_data={noisy_test}\033[0m",
        flush=True,
    )

    splits = [
        ("train", int(sim.n_frames), base_noise),
        ("test",  max(1, int(sim.n_frames) // 4),
         base_noise if noisy_test else 0.0),
    ]
    for split, n_frames_split, split_noise in splits:
        x_path = graphs_data_path(config.dataset, f"x_list_{split}")
        y_path = graphs_data_path(config.dataset, f"y_list_{split}")
        # Clean any prior data so we don't append.
        for p in (x_path, y_path):
            if os.path.isdir(p):
                _rmtree(p)

        x_writer = ZarrSimulationWriterV3(
            path=x_path, n_neurons=N, time_chunks=2000, save_calcium=False,
        )
        y_writer = ZarrArrayWriter(
            path=y_path, n_neurons=N, n_features=1, time_chunks=2000,
        )

        seed_offset = 0 if split == "train" else 1
        rng = np.random.default_rng(sim.seed + seed_offset)
        hp_split = dict(hp)
        hp_split["rng"] = np.random.RandomState(
            int(rng.integers(0, 2**31 - 1))
        )

        # Activate / deactivate dynamics noise for this split. TaskRNN's
        # forward only injects noise when `self.training and
        # noise_recurrent_level > 0`; we honour that gate explicitly.
        model.noise_recurrent_level = float(split_noise)
        if split_noise > 0:
            model.train()
        else:
            model.eval()
        logger.info(
            f"[voltage_from_task] {split}: dynamics noise σ={split_noise}  "
            f"(mode={'train' if split_noise > 0 else 'eval'})"
        )

        n_done = 0
        n_trials_done = 0
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=n_frames_split, ncols=150,
                     desc=f"  {split}: voltage frames", leave=True)
        with torch.no_grad():
            while n_done < n_frames_split:
                r = rules[int(rng.integers(len(rules)))]
                trial = generate_trials(r, hp_split, mode="random", batch_size=1)
                x_in_np, _y_tgt, _cm = trial_to_numpy(trial, 0)
                T_trial = int(trial.tdim)
                u = torch.from_numpy(x_in_np[None]).to(device)
                _y_hat, h_buf = model(u)

                voltage = h_buf[0, :T_trial].detach().cpu()
                u_t = u[0, :T_trial]
                if model.input_proj == "matrix":
                    drive_t = u_t @ model.W_in.t()
                else:
                    drive_t = model._W_in_mlp(u_t)
                drive = drive_t.detach().cpu()

                # Numerical dv/dt (forward diff; last frame copies previous).
                dv = torch.zeros_like(voltage)
                if T_trial >= 2:
                    dv[:-1] = (voltage[1:] - voltage[:-1]) / dt
                    dv[-1] = dv[-2]

                for t in range(T_trial):
                    if n_done >= n_frames_split:
                        break
                    st = NeuronState(
                        pos=pos, group_type=group_type_t, neuron_type=neuron_type_t,
                        voltage=voltage[t].to(device),
                        stimulus=drive[t].to(device),
                    )
                    x_writer.append_state(st)
                    y_writer.append(dv[t].unsqueeze(-1).numpy())
                    n_done += 1
                    pbar.update(1)
                n_trials_done += 1
        pbar.close()
        x_writer.finalize()
        y_writer.finalize()
        logger.info(
            f"[voltage_from_task] {split}: {n_done} frames from "
            f"{n_trials_done} trials -> {x_path}"
        )

    # --- Sanity plots (saved at dataset root, before any downstream GNN
    # training kicks off). Stimulus.zarr is the deterministic per-unit
    # drive (W_in @ u(t)) — no noise is ever added to it; only the
    # voltage receives `noise_recurrent_level · randn` during the rollout
    # when sim.noise_model_level > 0. ---

    # 1. Trace plot: reuse the canonical flyvis-style stacked-voltage
    #    figure (cross.trace_plot.save_trace_plot). Falls back to picking
    #    12 evenly-spaced units when neuron_type is uniform (cortex case).
    #    Output: <dataset>/traces.png
    from connectome_gnn.cross.trace_plot import save_trace_plot
    save_trace_plot(folder, force=True)
    logger.info(
        f"[voltage_from_task] saved traces: {os.path.join(folder, 'traces.png')}"
    )

    # 2. Decoder sanity plot: re-run the teacher end-to-end on 5 fresh
    #    trials and pass through `save_cortex_test_kinograph` (3 rows ×
    #    5 cols + 2 right panels). If the decoder reproduces the cortex
    #    target, the rollout is consistent.
    #    Output: <dataset>/sanity_decoder.png
    from connectome_gnn.models.cortex_eval import save_cortex_test_kinograph
    n_sanity = 5
    rng_s = np.random.default_rng(sim.seed + 9)
    hp_s = dict(hp)
    hp_s["rng"] = np.random.RandomState(int(rng_s.integers(0, 2**31 - 1)))
    sanity_stim, sanity_pred, sanity_tgt, sanity_cm = [], [], [], []
    # Decoder check should be deterministic — temporarily disable noise.
    saved_noise = model.noise_recurrent_level
    model.noise_recurrent_level = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(n_sanity):
            r = rules[int(rng_s.integers(len(rules)))]
            trial = generate_trials(r, hp_s, mode="random", batch_size=1)
            x_in_np, y_tgt_np, cm_np = trial_to_numpy(trial, 0)
            T_trial = int(trial.tdim)
            u = torch.from_numpy(x_in_np[None]).to(device)
            y_hat, _ = model(u)
            sanity_stim.append(torch.from_numpy(x_in_np[:T_trial]))
            sanity_pred.append(y_hat[0, :T_trial].detach().cpu())
            sanity_tgt.append(torch.from_numpy(y_tgt_np[:T_trial]))
            sanity_cm.append(torch.from_numpy(cm_np[:T_trial]))
    model.noise_recurrent_level = saved_noise
    sanity_path = os.path.join(folder, "sanity_decoder.png")
    save_cortex_test_kinograph(
        sanity_stim, sanity_pred, sanity_tgt, sanity_cm,
        output_path=sanity_path,
        rule_name=(rules[0] if rules else "cortex"),
        n_trials=n_sanity,
    )
    logger.info(f"[voltage_from_task] saved decoder sanity plot: {sanity_path}")
