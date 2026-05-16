"""Task-trainer (path-integration) variant of the LLM exploration pipeline.

Sibling of pipeline.py for the `data_train_task_gnn` route. The only
task-specific bits are:

- Schema-specific metric reading / printing / collapse detection
  (`tmp_training/metrics.log` columns differ from the flyvis schema).
- Skipping the test+plot phase (the trainer's own snapshots and metrics.log
  are the authoritative output).

Everything else is reused from existing code:
- `submit_cluster_job` + `wait_for_cluster_jobs` (cluster.py) — the
  `train_subprocess.py` cluster command works unchanged because the
  `data_train` dispatcher routes any config with a `task` block to
  `data_train_task_gnn`.
- `data_train` (graph_trainer.py) for the local path — same dispatcher.
- `setup_exploration`, `init_*`, `run_batch_0`, `load_configs_and_seeds`,
  `save_artifacts`, `run_claude_analysis`, `finalize_batch` (pipeline.py)
  for the orchestration.
"""
from __future__ import annotations

import os
import time

from connectome_gnn.LLM.cluster import (
    check_cluster_repo,
    submit_cluster_job,
    wait_for_cluster_jobs,
)
from connectome_gnn.LLM.state import BatchInfo, ExplorationState
from connectome_gnn.utils import get_data_root, log_path

_ANSI_RESET   = '\033[0m'
_ANSI_GREEN   = '\033[92m'
_ANSI_YELLOW  = '\033[93m'
_ANSI_ORANGE  = '\033[38;5;208m'
_ANSI_RED     = '\033[91m'
_ANSI_BLUE    = '\033[94m'
_ANSI_GREY    = '\033[90m'


# ---------------------------------------------------------------------------
# Metrics readers / printers (task schema)
# ---------------------------------------------------------------------------


_PI_COLS = {'iteration', 'epoch', 'loss', 'mse', 'cosd', 'norm', 'tv',
            'l1S', 'pi_acc', 'fwhm_deg'}
_CORTEX_COLS = {'iteration', 'epoch', 'loss', 'mse', 'motor_max',
                'motor_peak_mean', 'direction_acc'}


def _read_all_task_metrics(log_dir: str) -> list:
    """Read every data row of `<log_dir>/tmp_training/metrics.log`.

    Two schemas are supported (detected by header row):

      PI (data_train_pi_task_gnn):
        iteration,epoch,loss,mse,cosd,norm,tv,l1S,pi_acc,fwhm_deg
      Cortex (data_train_cortex_task_gnn):
        iteration,epoch,loss,mse,motor_max,motor_peak_mean,direction_acc

    Each returned dict carries the union of fields (missing fields = None).
    Two synthetic fields are added so downstream code stays uniform:
      `primary`      — the headline metric (pi_acc for PI, direction_acc
                       for cortex), in [0,1] for both.
      `primary_name` — string label for displays/logs.
    Returns [] if the file is missing or empty.
    """
    path = os.path.join(log_dir, 'tmp_training', 'metrics.log')
    if not os.path.isfile(path):
        return []
    rows: list = []
    try:
        with open(path) as f:
            header = f.readline().strip()
            if not header:
                return []
            cols = [c.strip() for c in header.split(',')]
            col_index = {c: i for i, c in enumerate(cols)}

            def _f(parts, name):
                idx = col_index.get(name)
                if idx is None or idx >= len(parts):
                    return None
                v = parts[idx].strip()
                if not v or v.lower() == 'nan':
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            for line in f:
                line = line.strip()
                if not line or line.startswith('iteration'):
                    continue
                parts = line.split(',')
                if len(parts) != len(cols):
                    continue
                row = {
                    'iter':   int(float(parts[col_index['iteration']])),
                    'epoch':  int(float(parts[col_index['epoch']])),
                    'loss':   _f(parts, 'loss'),
                    'mse':    _f(parts, 'mse'),
                    # PI fields (None when cortex)
                    'cosd':   _f(parts, 'cosd'),
                    'norm':   _f(parts, 'norm'),
                    'tv':     _f(parts, 'tv'),
                    'l1S':    _f(parts, 'l1S'),
                    'pi_acc': _f(parts, 'pi_acc'),
                    'fwhm':   _f(parts, 'fwhm_deg'),
                    # Cortex fields (None when PI)
                    'motor_max':        _f(parts, 'motor_max'),
                    'motor_peak_mean':  _f(parts, 'motor_peak_mean'),
                    'direction_acc':    _f(parts, 'direction_acc'),
                }
                # Pick a primary metric for collapse detection / display.
                if row['pi_acc'] is not None:
                    row['primary'] = row['pi_acc']
                    row['primary_name'] = 'pi_acc'
                elif row['direction_acc'] is not None:
                    row['primary'] = row['direction_acc']
                    row['primary_name'] = 'direction_acc'
                else:
                    row['primary'] = None
                    row['primary_name'] = 'metric'
                rows.append(row)
    except OSError:
        return []
    return rows


def _read_latest_task_metrics(log_dir: str) -> dict | None:
    rows = _read_all_task_metrics(log_dir)
    return rows[-1] if rows else None


def _per_epoch_task_summary(log_dir: str) -> list:
    """For each epoch present in metrics.log, return dict with end-of-epoch
    + best-of-epoch primary/loss/fwhm. Useful for diagnosing collapse mid-run.
    The `primary` field is `pi_acc` for path-integration and `direction_acc`
    for cortex; both are stored as `primary_end`/`primary_best` so display
    code stays uniform across tasks.
    """
    rows = _read_all_task_metrics(log_dir)
    if not rows:
        return []
    by_ep: dict = {}
    for r in rows:
        by_ep.setdefault(r['epoch'], []).append(r)
    summary = []
    for ep in sorted(by_ep.keys()):
        ep_rows = by_ep[ep]
        last = ep_rows[-1]
        pr_vals = [r['primary'] for r in ep_rows if r['primary'] is not None]
        loss_vals = [r['loss'] for r in ep_rows if r['loss'] is not None]
        summary.append({
            'epoch':        ep,
            'iter_last':    last['iter'],
            'primary_name': last['primary_name'],
            'primary_end':  last['primary'],
            'primary_best': max(pr_vals) if pr_vals else None,
            # PI-specific (None for cortex)
            'pi_acc_end':   last['pi_acc'],
            'pi_acc_best': (max((r['pi_acc'] for r in ep_rows
                                 if r['pi_acc'] is not None), default=None)),
            'fwhm_end':     last['fwhm'],
            'cosd_end':     last['cosd'],
            'norm_end':     last['norm'],
            'tv_end':       last['tv'],
            # Cortex-specific (None for PI)
            'direction_acc_end': last['direction_acc'],
            'motor_max_end':     last['motor_max'],
            'motor_peak_mean_end': last['motor_peak_mean'],
            # Shared
            'loss_end':    last['loss'],
            'loss_best':   min(loss_vals) if loss_vals else None,
            'mse_end':     last['mse'],
        })
    return summary


def _detect_collapse(per_epoch: list, drop_thresh: float = 0.4) -> str | None:
    """Return a one-line diagnostic if the primary metric collapses by
    >drop_thresh between consecutive epochs (curriculum instability)."""
    if len(per_epoch) < 2:
        return None
    for prev, cur in zip(per_epoch[:-1], per_epoch[1:]):
        a, b = prev['primary_end'], cur['primary_end']
        if a is None or b is None:
            continue
        if a - b >= drop_thresh:
            name = prev.get('primary_name', 'metric')
            return (f"collapse_detected: epoch {prev['epoch']} {name}={a:.3f}"
                    f" → epoch {cur['epoch']} {name}={b:.3f}"
                    f" (drop={a - b:.3f})")
    return None


def _pi_color(val):
    """Color the primary task metric (pi_acc or direction_acc, both in [0,1]):
    ≥0.9 green, ≥0.5 yellow, ≥0 orange, <0 red."""
    if val is None:
        return _ANSI_GREY
    if val >= 0.9:
        return _ANSI_GREEN
    if val >= 0.5:
        return _ANSI_YELLOW
    if val >= 0.0:
        return _ANSI_ORANGE
    return _ANSI_RED


def _print_task_metrics(log_dirs: dict, slots: list, prefix: str = '  [metrics]'):
    """Per-slot summary of latest snapshot + per-epoch trajectory so the
    curriculum collapse pattern (T=100 OK → T=500 explodes) is visible
    while training is still running.
    """
    for slot in sorted(slots):
        log_dir = log_dirs.get(slot)
        if not log_dir:
            print(f"{prefix} slot {slot}: (no log_dir)")
            continue
        m = _read_latest_task_metrics(log_dir)
        if m is None:
            print(f"{prefix} slot {slot}: (no metrics.log yet)")
            continue
        primary = m['primary']; loss = m['loss']; mse = m['mse']
        pname = m['primary_name']
        pi_col = _pi_color(primary)
        primary_str = f"{primary:.3f}" if primary is not None else "nan"
        loss_str = f"{loss:.4f}" if loss is not None else "nan"
        mse_str = f"{mse:.4f}" if mse is not None else "nan"
        # Schema-specific secondary diagnostic.
        if m['fwhm'] is not None:
            secondary = f"fwhm={m['fwhm']:.0f}°"
        elif m['motor_max'] is not None:
            secondary = f"motor_max={m['motor_max']:.3f}"
        else:
            secondary = ""

        per_epoch = _per_epoch_task_summary(log_dir)
        traj_parts = []
        for e in per_epoch:
            ep_pr = e['primary_end']
            if ep_pr is None:
                traj_parts.append(f"e{e['epoch']}=nan")
            else:
                traj_parts.append(f"{_pi_color(ep_pr)}e{e['epoch']}={ep_pr:.2f}{_ANSI_RESET}")
        traj = "  ".join(traj_parts)
        collapse = _detect_collapse(per_epoch)
        collapse_tag = f"  {_ANSI_RED}[COLLAPSE]{_ANSI_RESET}" if collapse else ""
        print(
            f"{prefix} slot {slot}  it={m['iter']:>5d} ep={m['epoch']}  "
            f"{pi_col}{pname}={primary_str}{_ANSI_RESET}  {secondary}  "
            f"loss={loss_str}  mse={mse_str}{collapse_tag}"
        )
        if traj:
            print(f"{prefix}   traj: {traj}")


# ---------------------------------------------------------------------------
# Per-batch metric extraction → analysis log (Claude reads this)
# ---------------------------------------------------------------------------


def _write_task_metrics_to_analysis_log(log_dir: str, analysis_log_path: str,
                                        slot: int, iteration: int) -> dict | None:
    """Append final + per-epoch summary to the slot's analysis log.

    Claude reads this file to make the next mutation decision. We surface:
      - The end-of-run row (final pi_acc / fwhm / losses).
      - A per-epoch table so collapse during the curriculum is obvious
        (e.g. pi_acc=0.999 at T=100, then 0.000 at T=500 = numerical
        instability, very different from "didn't converge yet").
      - A `collapse_detected:` flag the LLM can grep for.
    """
    rows = _read_all_task_metrics(log_dir)
    if not rows:
        with open(analysis_log_path, 'a') as f:
            f.write(f"\n--- slot {slot} iter {iteration} ---\n")
            f.write("ERROR: no metrics.log found\n")
        return None
    final = rows[-1]
    per_epoch = _per_epoch_task_summary(log_dir)
    collapse = _detect_collapse(per_epoch)

    pname = final.get('primary_name', 'metric')
    with open(analysis_log_path, 'a') as f:
        f.write(f"\n--- slot {slot} iter {iteration} ---\n")
        f.write(f"final {pname}: {final['primary']}\n")
        f.write(f"final loss: {final['loss']}\n")
        f.write(f"final mse: {final['mse']}\n")
        # PI-specific lines (omitted when cortex)
        if final.get('fwhm') is not None or pname == 'pi_acc':
            f.write(f"final fwhm_deg: {final['fwhm']}\n")
            f.write(f"final cosd: {final['cosd']}\n")
            f.write(f"final norm: {final['norm']}\n")
            f.write(f"final tv: {final['tv']}\n")
        # Cortex-specific lines (omitted when PI)
        if final.get('motor_max') is not None or pname == 'direction_acc':
            f.write(f"final motor_max: {final['motor_max']}\n")
            f.write(f"final motor_peak_mean: {final['motor_peak_mean']}\n")
        f.write(f"final iter: {final['iter']}  epoch: {final['epoch']}\n")
        f.write("\nper-epoch trajectory:\n")
        if pname == 'direction_acc':
            f.write("  ep | dir_acc_end dir_acc_best | motor_max_end | "
                    "loss_end loss_best | mse_end\n")
        else:
            f.write("  ep | pi_acc_end pi_acc_best | fwhm_end | "
                    "loss_end loss_best | mse_end cosd_end norm_end tv_end\n")
        for e in per_epoch:
            def _fmt(v, w=6):
                return ("nan".ljust(w) if v is None
                        else f"{v:.3f}".ljust(w))
            if pname == 'direction_acc':
                f.write(
                    f"  {e['epoch']:>2d} | {_fmt(e['direction_acc_end'])} "
                    f"{_fmt(e['primary_best'])} | {_fmt(e['motor_max_end'])} | "
                    f"{_fmt(e['loss_end'])} {_fmt(e['loss_best'])} | "
                    f"{_fmt(e['mse_end'])}\n"
                )
            else:
                f.write(
                    f"  {e['epoch']:>2d} | {_fmt(e['pi_acc_end'])} {_fmt(e['pi_acc_best'])}"
                    f" | {_fmt(e['fwhm_end'])} | "
                    f"{_fmt(e['loss_end'])} {_fmt(e['loss_best'])} | "
                    f"{_fmt(e['mse_end'])} {_fmt(e['cosd_end'])} "
                    f"{_fmt(e['norm_end'])} {_fmt(e['tv_end'])}\n"
                )
        if collapse:
            f.write(f"\n{collapse}\n")
        else:
            f.write("\ncollapse_detected: no\n")
    final['per_epoch'] = per_epoch
    final['collapse'] = collapse
    return final


def _print_task_batch_results(state: ExplorationState, batch: BatchInfo,
                              metrics_per_slot: dict):
    """Final coloured summary: per-epoch trajectory + collapse warning."""
    print(f"\n{_ANSI_BLUE}{'=' * 70}{_ANSI_RESET}")
    print(f"{_ANSI_BLUE}BATCH RESULTS: iterations "
          f"{batch.batch_first}-{batch.batch_last}{_ANSI_RESET}")
    print(f"{_ANSI_BLUE}{'=' * 70}{_ANSI_RESET}")
    for slot_idx, iteration in enumerate(batch.iterations):
        ok = batch.job_results.get(slot_idx, False)
        if not ok:
            print(f"  Slot {slot_idx} (iter {iteration}):  {_ANSI_RED}FAILED{_ANSI_RESET}")
            continue
        m = metrics_per_slot.get(slot_idx)
        if m is None:
            print(f"  Slot {slot_idx} (iter {iteration}):  {_ANSI_GREY}no metrics{_ANSI_RESET}")
            continue

        # Per-epoch line — shows the trajectory at a glance.
        per_epoch = m.get('per_epoch', [])
        ep_strs = []
        for e in per_epoch:
            pr = e['primary_end']
            if pr is None:
                ep_strs.append(f"e{e['epoch']}=nan")
            else:
                col = _pi_color(pr)
                ep_strs.append(f"{col}e{e['epoch']}={pr:.2f}{_ANSI_RESET}")
        traj = "  ".join(ep_strs) if ep_strs else "(no epochs)"

        pname = m.get('primary_name', 'metric')
        primary = m['primary']
        pi_col = _pi_color(primary)
        primary_str = f"{primary:.3f}" if primary is not None else "n/a"
        loss = f"{m['loss']:.4f}" if m['loss'] is not None else "n/a"
        if m.get('fwhm') is not None:
            secondary = f"fwhm={m['fwhm']:.0f}°"
        elif m.get('motor_max') is not None:
            secondary = f"motor_max={m['motor_max']:.3f}"
        else:
            secondary = ""
        collapse_tag = (f"  {_ANSI_RED}[COLLAPSE]{_ANSI_RESET}"
                        if m.get('collapse') else "")

        print(f"  Slot {slot_idx} (iter {iteration}):  "
              f"final {pi_col}{pname}={primary_str}{_ANSI_RESET}  "
              f"{secondary}  loss={loss}{collapse_tag}")
        print(f"     trajectory: {traj}")
        if m.get('collapse'):
            print(f"     {_ANSI_RED}{m['collapse']}{_ANSI_RESET}")


# ---------------------------------------------------------------------------
# Cluster + local runners
# ---------------------------------------------------------------------------


def run_task_cluster_training(state: ExplorationState, batch: BatchInfo):
    """Submit task-trainer cluster jobs, wait, parse final metrics → analysis log.

    Reuses `submit_cluster_job` (which calls train_subprocess.py, which routes
    to data_train_task_gnn via the data_train dispatcher). No test+plot phase
    — the trainer's own snapshots and metrics.log are the authoritative output.
    """
    print(f"\n{_ANSI_YELLOW}PHASE 2: Submitting {batch.n_slots} task-trainer "
          f"jobs to cluster (gpu_{state.node_name}){_ANSI_RESET}")

    if not check_cluster_repo():
        print(f"{_ANSI_YELLOW}WARNING: cluster repo has uncommitted changes — "
              f"proceeding anyway{_ANSI_RESET}")

    job_ids = {}
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = batch.configs[slot]
        jid = submit_cluster_job(
            slot=slot,
            config_path=state.config_paths[slot],
            analysis_log_path=state.analysis_log_paths[slot],
            config_file_field=config.config_file,
            log_dir=state.log_dir,
            erase=True,
            node_name=state.node_name,
            conda_env=state.conda_env,
            n_cpus=state.n_cpus,
            device=config.training.device,
            exploration_dir=state.exploration_dir,
            iteration=iteration,
            output_root=get_data_root(),
            hard_runtime_limit_min=state.hard_runtime_limit_min,
        )
        if jid:
            job_ids[slot] = jid
        else:
            batch.job_results[slot] = False

    metrics_per_slot: dict = {}
    if job_ids:
        print(f"\n{_ANSI_YELLOW}PHASE 3: Waiting for {len(job_ids)} jobs"
              f"{_ANSI_RESET}")
        slot_log_dirs = {s: log_path(batch.configs[s].config_file) for s in job_ids}

        # Reuse the existing cluster waiter (it polls bjobs and returns
        # {slot: bool}). Periodic in-flight metric printing happens in a
        # tiny side-thread that tails each slot's tmp_training/metrics.log.
        import threading
        stop_print = threading.Event()

        def _periodic_print():
            while not stop_print.wait(timeout=300):
                _print_task_metrics(slot_log_dirs, list(job_ids.keys()),
                                    prefix='  [metrics]')

        printer = threading.Thread(target=_periodic_print, daemon=True)
        printer.start()
        try:
            cluster_results = wait_for_cluster_jobs(
                job_ids, log_dir=state.log_dir, poll_interval=60,
                job_prefix='cluster_train',
            )
        finally:
            stop_print.set()
            printer.join(timeout=1.0)

        batch.job_results.update(cluster_results)
        # Final per-slot metric snapshot (Claude reads the analysis log).
        for slot, ok in cluster_results.items():
            if ok:
                m = _write_task_metrics_to_analysis_log(
                    slot_log_dirs[slot], state.analysis_log_paths[slot],
                    slot, batch.iterations[slot],
                )
                metrics_per_slot[slot] = m

    _print_task_batch_results(state, batch, metrics_per_slot)


def run_task_local_pipeline(state: ExplorationState, batch: BatchInfo):
    """Local-mode runner: train each slot sequentially via the data_train
    dispatcher (which routes task configs to data_train_task_gnn)."""
    from connectome_gnn.models.graph_trainer import data_train

    print(f"\n{_ANSI_YELLOW}PHASE 2: Training {batch.n_slots} task-models "
          f"locally (sequential){_ANSI_RESET}")
    metrics_per_slot: dict = {}
    for slot_idx, iteration in enumerate(batch.iterations):
        slot = slot_idx
        config = batch.configs[slot]
        print(f"{_ANSI_GREY}  slot {slot} (iter {iteration}): training locally..."
              f"{_ANSI_RESET}")
        log_file = open(state.analysis_log_paths[slot], 'w')
        try:
            data_train(
                config=config, erase=True,
                best_model=state.best_model, device=state.device,
                log_file=log_file,
            )
            batch.job_results[slot] = True
        except Exception as exc:
            batch.job_results[slot] = False
            log_file.write(f"\nERROR: training crashed: {exc}\n")
            print(f"{_ANSI_RED}  slot {slot}: training FAILED ({exc}){_ANSI_RESET}")
        finally:
            log_file.close()
        slot_log_dir = log_path(config.config_file)
        m = _write_task_metrics_to_analysis_log(
            slot_log_dir, state.analysis_log_paths[slot],
            slot, iteration,
        )
        metrics_per_slot[slot] = m

    _print_task_batch_results(state, batch, metrics_per_slot)
