#!/usr/bin/env python
"""Write the tmp_training Wij, Eij and neuron panels for a run's saved checkpoints.

The diagnostic panels are drawn on the `should_record` cadence and the models are
saved on `checkpoint_saves_per_epoch`; the two do not coincide, so a run ends with
figures at iterations it kept no model for and models it drew no figure for. This
fills the second gap: for every checkpoint under models/, it runs the readout and
writes the three panels under the checkpoint's own iteration, beside the ones
training produced.

Everything is the production path -- extract_recovered_params,
extract_template_params, score_recovery, plot_recovery_panels,
plot_reversal_scatter, analyse_neurons -- called the way graph_trainer calls it,
so a panel written here and one written during training are the same figure.

Usage:
    python tools/panels_for_checkpoints.py [RUN] [--neuron 2895] [--only 200000]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from connectome_gnn import metrics as M  # noqa: E402
from connectome_gnn.config import NeuralGraphConfig  # noqa: E402
from connectome_gnn.generators.ode_params import load_ode_params_for_run  # noqa: E402
from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.models.training_utils import init_training_data  # noqa: E402
from connectome_gnn.neuron_panels import analyse_neurons  # noqa: E402
from connectome_gnn.plot import (  # noqa: E402
    INDEX_TO_NAME,
    W_OUTLIER_THRESH,
    plot_recovery_panels,
    plot_reversal_scatter,
)
from connectome_gnn.utils import migrate_state_dict, set_data_root, to_numpy  # noqa: E402

LOG = f"{os.environ['GNN_OUTPUT_ROOT']}/log/fly"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", default="flyvis_flowcond_noise_005_gnn_nosq_cv00")
    ap.add_argument("--neuron", type=int, default=2895)
    ap.add_argument("--only", type=int, default=None,
                    help="one checkpoint iteration; default is all of them")
    a = ap.parse_args(argv)

    set_data_root(os.environ["GNN_OUTPUT_ROOT"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir = os.path.join(LOG, a.run)
    cfg = NeuralGraphConfig.from_yaml(f"config/fly/{a.run}.yaml")
    cks = sorted(glob.glob(f"{log_dir}/models/*graphs_0_*.pt"),
                 key=lambda f: int(re.findall(r"_(\d+)\.pt$", f)[0]))
    if a.only is not None:
        cks = [c for c in cks if int(re.findall(r"_(\d+)\.pt$", c)[0]) == a.only]
    if not cks:
        print(f"{a.run}: no checkpoint")
        return 1

    cfg.dataset = "fly/" + cfg.dataset
    op = load_ode_params_for_run(cfg, device=dev)
    data = init_training_data(cfg, dev, log_dir, logging.getLogger("panels"))
    x_ts = data.x_list[0] if hasattr(data, "x_list") else data.x_ts
    edges = data.edges
    n_neurons = cfg.simulation.n_neurons
    types = to_numpy(data.type_list).ravel() if getattr(data, "type_list", None) is not None else None

    for ck in cks:
        it = int(re.findall(r"_(\d+)\.pt$", ck)[0])
        sd = torch.load(ck, map_location=dev, weights_only=False)
        migrate_state_dict(sd)
        cfg.simulation.n_edges = sd["model_state_dict"]["W"].shape[0]
        cfg.simulation.n_extra_null_edges = 0
        model = create_model(cfg.graph_model.signal_model_name,
                             aggr_type=cfg.graph_model.aggr_type, config=cfg, device=dev)
        model.load_state_dict(sd["model_state_dict"], strict=False)
        model.eval()
        model.edges = edges

        rec = M.extract_recovered_params(
            model, op, cfg, edges=edges, x_ts=x_ts, device=dev, n_neurons=n_neurons,
            need=("W", "tau", "V_rest", "E_ij", "msg_i"))
        rec = M.extract_template_params(
            model, op, config=cfg, edges=edges, x_ts=x_ts, device=dev,
            n_neurons=n_neurons, base=rec)
        scored = M.score_recovery(rec, cfg)

        # Wij, the corrected pair, grouped by the PRESYNAPTIC type as the
        # trainer's panel groups it.
        w = rec.get("W")
        if w is not None:
            src = to_numpy(edges).reshape(2, -1)[0]
            grp = types[src] if types is not None else None
            plot_recovery_panels(
                w[0], w[1],
                os.path.join(log_dir, "tmp_training", "Wij",
                             f"comparison_0_{it}.png"),
                symbol="W_{ij}", corrected=True, groups=grp,
                group_names=INDEX_TO_NAME, outlier_threshold=W_OUTLIER_THRESH,
                violin_log_y=True)

        # Eij, through the trainer's own entry point so the gate, the threshold
        # and the +-20 axes are whatever that function says they are.
        e = rec.get("E_ij")
        gate = None
        if e is None and not rec.valid.get("E_ij", True):
            e = rec.pairs.get("E_ij")
            gate = rec.diagnostics.get("msg_form_r2_median")
        if e is not None:
            dst = to_numpy(edges).reshape(2, -1)[1]
            plot_reversal_scatter(
                {"true": e[0], "learned": e[1],
                 "edge_type": types[dst] if types is not None else None,
                 "gate": gate},
                log_dir, 0, it)

        try:
            analyse_neurons(cfg, model, data, log_dir, device=dev,
                            out_dir=os.path.join(log_dir, "tmp_training",
                                                 "neuron_panels"),
                            tag=f"{it:08d}", sr_enabled=False, use_rollout=False,
                            quiet=True)
        except Exception as exc:
            print(f"  panel skipped: {type(exc).__name__}: {exc}")

        g = scored.get
        nan = float("nan")
        print(f"{it:8d}  Wij_R2 {g('Wij_R2', nan):+7.3f}  gain {g('Wij_gain', nan):6.3f}"
              f"  Eij_R2 {g('Eij_R2', nan):+8.3f}  msg_i_R2 {g('msg_i_R2', nan):+6.3f}"
              f"  tau_R2 {g('tau_R2', nan):+6.3f}")
    print(f"-> {os.path.join(log_dir, 'tmp_training')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
