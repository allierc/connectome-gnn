"""Paper figure: drosophila_cx_pi training evolution (4 x 2 panels).

Thin CLI shim. All panel rendering lives in
``connectome_gnn.plot_cx.plot_cx_evolution`` (the public entry point),
which is also imported by the training-time snapshot helper in
``connectome_gnn.models.drosophila_cx_eval._save_training_snapshot``.
That centralisation removed the importlib hack that used to load this
file via a hard-coded path; the bulk of the panel code (formerly here,
~700 lines of ``_panel_*`` helpers + ``build_figure``) now lives in
``plot_cx`` alongside ``plot_cx_matrix``, ``plot_cx_training_snapshot``
etc.

What this file still does:

    * ``_load_model_and_rollouts``: load a run directory's checkpoint,
      run the deterministic constant-ω rollout + the integration-gain
      sweep + pick an OU test trial. Produces the ``data`` dict that
      ``plot_cx_evolution`` consumes.
    * ``main``: argparse, iterate over run dirs, write one PNG each.

Usage:
    python figures/drosophila_cx/fig_evolution.py \
        --run_dir /groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi \
        --out_dir figures/drosophila_cx/
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))


def _load_model_and_rollouts(
    run_dir: str,
    snapshot_n_steps: int = 1500,
    snapshot_omega_deg: float = 60.0,
    gain_omegas=tuple(float(v) for v in np.concatenate([
        np.arange(-180.0, -9.9, 15.0),
        np.arange(15.0, 180.1, 15.0),
    ])),
    trial_seed: int | None = None,
    trial_idx: int | None = None,
):
    """Load model + run two rollouts + pick one OU test trial."""
    import torch
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.models.drosophila_cx_eval import _deterministic_sweep_rollout
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.plot_cx import cx_epg_directions
    from connectome_gnn.utils import set_data_root
    from connectome_gnn.zarr_io import load_raw_array

    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.yaml missing in {run_dir}")
    config = NeuralGraphConfig.from_yaml(cfg_path)

    # Replicate load_run_config's dataset-prefixing + data-root setup. The
    # run_dir is `<data_root>/log/<group>/<config_name>/`, so data_root is
    # two parents up from run_dir and the dataset prefix is the group name.
    run_dir_abs = os.path.abspath(run_dir)
    group = os.path.basename(os.path.dirname(run_dir_abs))
    data_root = os.path.dirname(os.path.dirname(os.path.dirname(run_dir_abs)))
    set_data_root(data_root)
    if group and not config.dataset.startswith(group + "/"):
        config.dataset = f"{group}/{config.dataset}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = create_model(config.graph_model.signal_model_name,
                       aggr_type=config.graph_model.aggr_type,
                       config=config, device=device)
    # Pick the highest-epoch checkpoint by default. Sort numerically by
    # the trailing _<epoch>.pt — lexicographic sort would mis-order
    # _9.pt vs _10.pt. Per-config override: the GNN tail-loss runs are
    # best at epoch 5 (past that they overfit and degrade on the
    # constant-ω extrapolation rollout).
    ckpts = glob.glob(os.path.join(run_dir, "models",
                                     "best_model_with_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/models/")

    def _epoch_of(p):
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    ckpts.sort(key=_epoch_of)
    chosen = ckpts[-1]
    run_basename = os.path.basename(os.path.abspath(run_dir))
    if "gnn_tailloss" in run_basename:
        prefer = [p for p in ckpts if _epoch_of(p) == 5]
        if prefer:
            chosen = prefer[0]
    sd = torch.load(chosen, map_location=device,
                    weights_only=False)["model_state_dict"]
    net.load_state_dict(sd, strict=False)
    net.eval()

    rollout = _deterministic_sweep_rollout(
        net, n_steps=snapshot_n_steps,
        omega_deg_per_s=snapshot_omega_deg, device=device,
    )
    rollout["r_epg"] = rollout["r"][:, net.epg_indices]
    pen_type_idx = [i for i, n in enumerate(net.type_names)
                    if "PEN" in n and "PEG" not in n]
    nt = np.asarray(net.neuron_types)
    pen_indices = None
    if pen_type_idx:
        pen_idx_list: list[int] = []
        for t in pen_type_idx:
            pen_idx_list.extend(np.where(nt == t)[0].tolist())
        pen_indices = np.array(sorted(pen_idx_list), dtype=np.int64)
        rollout["r_pen"] = rollout["r"][:, pen_indices]

    epg_theta = cx_epg_directions(net.epg_glom_ix)

    gain_data = []
    for omega in gain_omegas:
        ro = _deterministic_sweep_rollout(
            net, n_steps=snapshot_n_steps,
            omega_deg_per_s=float(omega), device=device,
        )
        gain_data.append((float(omega), ro))

    from connectome_gnn.utils import graphs_data_path
    root = graphs_data_path(config.dataset)
    u_test = load_raw_array(f"{root}/test/stimulus.zarr")
    y_test = load_raw_array(f"{root}/test/target.zarr")
    if trial_idx is None:
        if trial_seed is None:
            trial_seed = int(getattr(config.training, "seed", 0)) + 17
        rng = np.random.default_rng(trial_seed)
        trial_idx = int(rng.integers(0, u_test.shape[0]))
    trial_idx = int(trial_idx) % u_test.shape[0]
    u_one = u_test[trial_idx]
    y_true = y_test[trial_idx]
    with torch.no_grad():
        u_t = torch.from_numpy(u_one[None]).to(device)
        y_pred, _ = net(u_t)
    y_pred = y_pred[0].cpu().numpy()
    test_trial = dict(
        idx=trial_idx,
        u=u_one,
        y_true=y_true,
        y_pred=y_pred,
        dt=float(config.task.path_integration.dt),
    )

    return dict(
        net=net,
        config=config,
        W_rec=net.W_rec.detach().cpu().numpy(),
        W_con=net.W_con.detach().cpu().numpy(),
        neuron_types=net.neuron_types,
        type_names=net.type_names,
        pen_indices=pen_indices,
        rollout=rollout,
        epg_theta=epg_theta,
        gain_data=gain_data,
        test_trial=test_trial,
        dt_s=float(net.dt),
        checkpoint=chosen,
    )


# --- CLI -----------------------------------------------------------------


DEFAULT_RUN_DIRS = [
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi",
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi_frozen_Wrec",
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi_fc",
    "/groups/saalfeld/home/allierc/GraphData/log/drosophila_cx/drosophila_cx_pi_gnn",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", action="append", default=None,
        help="training-run directory (with config.yaml, models/, ...). "
             "May be passed multiple times to generate one figure per run.")
    p.add_argument(
        "--out_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="output directory. Each figure is written as "
             "fig_evolution_<run_basename>.png.")
    p.add_argument("--snapshot_n_steps", type=int, default=1500)
    p.add_argument("--snapshot_omega_deg", type=float, default=60.0)
    p.add_argument("--trial_seed", type=int, default=None,
                    help="seed picking the OU test trial "
                         "(default: config.training.seed + 17)")
    p.add_argument("--trial_idx", type=int, default=None,
                    help="explicit test-trial index (overrides --trial_seed).")
    args = p.parse_args()

    from connectome_gnn.plot_cx import plot_cx_evolution

    run_dirs = args.run_dir or DEFAULT_RUN_DIRS
    os.makedirs(args.out_dir, exist_ok=True)
    for run_dir in run_dirs:
        try:
            data = _load_model_and_rollouts(
                run_dir,
                snapshot_n_steps=args.snapshot_n_steps,
                snapshot_omega_deg=args.snapshot_omega_deg,
                trial_seed=args.trial_seed,
                trial_idx=args.trial_idx,
            )
        except Exception as exc:
            print(f"[fig_evolution] SKIP {run_dir}: {exc}")
            continue
        print(f"[fig_evolution] loaded {data['checkpoint']}")
        out_path = os.path.join(
            args.out_dir,
            f"fig_evolution_{os.path.basename(os.path.abspath(run_dir))}.png",
        )
        plot_cx_evolution(data, out_path, run_dir=run_dir)


if __name__ == "__main__":
    main()
