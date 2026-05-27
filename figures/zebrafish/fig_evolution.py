"""Paper figure: zebrafish_hd_si training evolution (4 x 2 / 4 x 3 panels).

Companion of figures/drosophila_cx/fig_evolution.py. Same panel layout
(a–h in two-row mode, a–l with extras in three-row mode). All rendering
is shared via ``connectome_gnn.plot_cx.plot_cx_evolution``; this file is
just the CLI / data-loading shim that:

  * loads a zebrafish run directory (``<data_root>/log/zebrafish/<run>/``),
  * runs the same deterministic constant-ω rollout used by the
    drosophila companion (the model integrates ω regardless of whether
    ω came from an OU stream or a swim-impulse boxcar, so a
    constant-ω probe is still meaningful for the bump trajectory),
  * picks one swim-integration test trial,
  * passes the species-specific axis labels (``r1π / dIPN`` for the
    bump cells, ``RIPN / pt-IPN`` for the afferents) through to the
    figure builder.

Usage:
    python figures/zebrafish/fig_evolution.py \\
        --run_dir /groups/saalfeld/home/allierc/GraphData/log/zebrafish/zebrafish_hd_si_dipn \\
        --out_dir figures/zebrafish/
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
    """Load model + run rollouts + pick one swim-integration test trial."""
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

    # Replicate load_run_config's dataset-prefixing + data-root setup.
    # run_dir = <data_root>/log/<group>/<config_name>/. data_root is two
    # parents up; the prefix is the group name.
    run_dir_abs = os.path.abspath(run_dir)
    group = os.path.basename(os.path.dirname(run_dir_abs))   # e.g. zebrafish
    data_root = os.path.dirname(os.path.dirname(os.path.dirname(run_dir_abs)))
    set_data_root(data_root)
    if group and not config.dataset.startswith(group + "/"):
        config.dataset = f"{group}/{config.dataset}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = create_model(config.graph_model.signal_model_name,
                       aggr_type=config.graph_model.aggr_type,
                       config=config, device=device)
    # Pick the highest-epoch checkpoint by default.
    ckpts = glob.glob(os.path.join(run_dir, "models",
                                    "best_model_with_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/models/")

    def _epoch_of(p):
        m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    ckpts.sort(key=_epoch_of)
    chosen = ckpts[-1]
    sd = torch.load(chosen, map_location=device,
                    weights_only=False)["model_state_dict"]
    net.load_state_dict(sd, strict=False)
    net.eval()

    # Deterministic constant-ω rollout (the network's "compass test"): the
    # bump should rotate at a constant angular velocity. Probes the
    # ring-attractor's gain regardless of whether the training stream was
    # OU or swim impulses.
    rollout = _deterministic_sweep_rollout(
        net, n_steps=snapshot_n_steps,
        omega_deg_per_s=snapshot_omega_deg, device=device,
    )
    rollout["r_epg"] = rollout["r"][:, net.epg_indices]

    # Afferent population (RIPN + pt-IPN here, PEN_a/b L/R for the fly):
    # union of the velocity-gate sub-population indicator buffers populated
    # by the model from the loader's pen_subpop_ix. Buffer-based so the
    # lookup is species-agnostic.
    ind_keys = ("_pen_ind_pena_l", "_pen_ind_pena_r",
                 "_pen_ind_penb_l", "_pen_ind_penb_r")
    pen_indices = None
    if all(hasattr(net, k) for k in ind_keys):
        union = sum(getattr(net, k) for k in ind_keys)
        idx = (union > 0).nonzero(as_tuple=True)[0].cpu().numpy()
        if idx.size:
            pen_indices = idx.astype(np.int64)
            rollout["r_pen"] = rollout["r"][:, pen_indices]

    epg_theta = cx_epg_directions(net.epg_glom_ix)

    # Integration-gain sweeps
    gain_data = []
    for omega in gain_omegas:
        ro = _deterministic_sweep_rollout(
            net, n_steps=snapshot_n_steps,
            omega_deg_per_s=float(omega), device=device,
        )
        gain_data.append((float(omega), ro))

    # One swim-integration test trial
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
    # dt comes from whichever task block this run uses (swim_integration
    # for zebrafish, path_integration for the fly companion).
    task_block = (config.task.path_integration
                  if config.task.task_type == "path_integration"
                  else config.task.swim_integration)
    test_trial = dict(
        idx=trial_idx,
        u=u_one,
        y_true=y_true,
        y_pred=y_pred,
        dt=float(task_block.dt),
        label="swim test trial",
    )

    # Species-specific display labels picked up from the model class
    # (DrosophilaCxTaskRNN → "EPG"/"PEN", ZebrafishHdTaskRNN →
    # "r1π / dIPN"/"RIPN / pt-IPN", same for the GNN subclasses).
    bump_label = getattr(type(net), "bump_label", "EPG")
    afferent_label = getattr(type(net), "afferent_label", "PEN")

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
        bump_label=bump_label,
        afferent_label=afferent_label,
    )


# --- CLI -----------------------------------------------------------------


DEFAULT_RUN_DIRS = [
    "/groups/saalfeld/home/allierc/GraphData/log/zebrafish/zebrafish_hd_si_dipn",
    "/groups/saalfeld/home/allierc/GraphData/log/zebrafish/zebrafish_hd_si_frozen_Wrec_dipn",
    "/groups/saalfeld/home/allierc/GraphData/log/zebrafish/zebrafish_hd_si_fc_dipn",
    "/groups/saalfeld/home/allierc/GraphData/log/zebrafish/zebrafish_hd_si_gnn_dipn",
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
                    help="seed picking the swim-integration test trial "
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
