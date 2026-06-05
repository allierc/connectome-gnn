"""Functional panel figure: given a trained-model CONFIG, render the model
functional panel (voltage -> GCaMP calcium -> kinograph) for the zebrafish
HD/IPN12 circuit.

The model rollout is NOT duplicated here: the rotation stimulus is the
first-class `zapbench_rotation` anatomy_voltage pattern, and the forward pass is
connectome_gnn.plot_anatomy_voltage.run_task_rollout (which returns the voltage
trajectory + readout). This script only loads the checkpoint, convolves the
voltage with the GCaMP indicator from connectome_gnn.models.gcamp (defaulting to
config.simulation.gcamp_kernel), maps it onto the fixed 300-neuron bump row set,
and renders.

  log_dir is derived from the config path:  .../config/<sub>/<name>.yaml
                                       ->   .../log/<sub>/<name>
  circuit comes from the config's circuit.name (else --model-circuit).

NOTE — this panel is the *continuous-rollout* (integration-drift) view: ONE
600 s zapbench_rotation rollout, so HD error accumulates over the block. It is
NOT the per-trial reconstruction: `GNN_Main -o test` section (e)
(graph_tester.data_test_path_integration_task) instead stitches 60 independent
10 s rollouts, each re-anchored from its cue (the regime the model is trained
in), and reports both views side by side. Use this script for the paper drift
figure; use `-o test` for the reconstruction/drift comparison.

Run (env with fishfuncem + torch + the trained model):
  /workspace/.conda_envs/neural-graph-linux/bin/python \
      figures/zebrafish/fig_functional_panel.py \
      --config zebrafish/zebrafish_hd_si_gnn_dipn_v1_cv0 [--gcamp gcamp7f]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "papers", "fishFuncEM"))

# Row set + rendering helpers live in the functional-panel script (figure
# domain); the rollout/stimulus now come from the package (no duplication).
_PANEL_PY = os.path.join(_REPO, "figures", "zebrafish",
                         "zebrafish_functional_traces_panel.py")
_spec = importlib.util.spec_from_file_location("_ftp", _PANEL_PY)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)


def _resolve_config(arg: str) -> str:
    a = arg if arg.endswith(".yaml") else arg + ".yaml"
    if os.path.isabs(a):
        cands = [a]
    else:
        root = os.environ.get("GNN_OUTPUT_ROOT",
                              "/groups/saalfeld/home/allierc/GraphData")
        cands = [os.path.join(root, "config", a), os.path.join(_REPO, "config", a),
                 os.path.join(_REPO, a), a]
        # Fall back to the species subdirs (config/zebrafish/, config/drosophila_cx/,
        # ...) so a bare config name resolves the same way the training CLI does.
        if "/" not in a:
            for cfg_root in (os.path.join(root, "config"),
                             os.path.join(_REPO, "config")):
                cands += sorted(glob.glob(os.path.join(cfg_root, "*", a)))
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"config not found for '{arg}' (tried {cands})")


def _log_dir_for(cfg_path: str) -> str:
    stem = cfg_path[:-5] if cfg_path.endswith(".yaml") else cfg_path
    # Training writes under GNN_OUTPUT_ROOT/log/<species>/<name>, not the
    # repo-local log/. Take the part after .../config/ and join it there.
    root = os.environ.get("GNN_OUTPUT_ROOT",
                          "/groups/saalfeld/home/allierc/GraphData")
    marker = os.sep + "config" + os.sep
    if marker in stem:
        rel = stem.split(marker, 1)[1]      # e.g. "zebrafish/<name>"
        return os.path.join(root, "log", rel)
    return os.path.join(os.path.dirname(stem), "log", os.path.basename(stem))


# zapbench task blocks (figures/zebrafish/zapbench_full_kinograph.py partitioning)
# that carry a heading the HD circuit can be read against — the only blocks for
# which a "true HD" trace + decode is defined.
_SEQUENCES = ["rotation", "turning"]


def _render_real_sequences(args):
    """REAL zapbench ΔF/F functional panel, one per requested task block. Only
    the heading-bearing blocks are supported (`rotation`, `turning`); `all` does
    both. Reuses the row set / decode / render of zebrafish_functional_traces_
    panel.py — this is the data-only counterpart of the model panel (no
    checkpoint loaded). Rows are in the whole-brain rastermap order."""
    seqs = _SEQUENCES if args.sequence == "all" else [args.sequence]
    out_dir = args.out or os.path.join(_REPO, "figures", "zebrafish")
    os.makedirs(out_dir, exist_ok=True)

    rows, _ = panel.build_rows(args.connectome, args.circuit)
    panel_rows = panel.sort_rows_rastermap(
        rows[rows["matched"]].reset_index(drop=True))
    z = np.load(os.path.join(args.connectome, "functional",
                             "circuit_functional_traces.npz"), allow_pickle=True)
    trace_len = int(z["traces"].shape[0])
    print(f"[rows] {len(panel_rows)} mapped bump neurons; "
          f"sequences={seqs}")

    for seq in seqs:
        if seq == "rotation":
            _theta_hr, ts, theta_frame = panel.rotation_headings(
                args.fishfuncem_data, args.connectome, 0.01)
            keep = ts < trace_len
            ts = ts[keep]; theta_frame = theta_frame[:len(ts)]
            drive = panel.heading_to_drive(theta_frame, dt=0.915, src_dt=0.915)
            title = ("REAL zapbench ΔF/F — Rotations "
                     "(45°/s grating, imaging-sampled)")
        else:  # turning (omr_turning)
            stim, ts = panel.omr_block_stim(args.fishfuncem_data,
                                            trace_len=trace_len)
            drive = panel.stim_to_drive(stim, dt=0.915,
                                        nominal_omega=args.nominal_omega)
            title = "REAL zapbench ΔF/F — omr_turning (dIPN ring + IPN12)"
        if args.no_title:
            title = ""
        print(f"[stim] {seq}: {len(ts)} frames (~{len(ts) * 0.915 / 60:.1f} min)")

        d = panel.real_panel(panel_rows, args.connectome, ts, drive,
                             decoder="mlp")
        win = "" if args.t_window is None else f"_z{int(args.t_window)}s"
        out_png = os.path.join(out_dir, f"functional_panel_real_{seq}{win}.png")
        panel.render(d, panel_rows, out_png, title, bg="white",
                     show_swim_panels=False, cmap_name="viridis",
                     show_partition=False, t_window=args.t_window)
        print(f"[done] {out_png}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="config path or name (e.g. zebrafish/zebrafish_hd_si_ipn12_v1); "
                        "required for the model panel, ignored with --sequence")
    p.add_argument("--gcamp", default=None,
                   help="GCaMP indicator name from connectome_gnn.models.gcamp "
                        "(e.g. gcamp6f/6s/7f/8f/8m); default = "
                        "config.simulation.gcamp_kernel")
    p.add_argument("--log-dir", default=None, help="override derived log dir")
    p.add_argument("--model-circuit", default=None,
                   help="circuit name when the config has no circuit.name "
                        "(e.g. zebrafish_HD_IPN12_839_v1)")
    p.add_argument("--connectome",
                   default=os.path.join(_REPO, "figures", "zebrafish",
                                        "zebrafish_connectome_HD_IPN12"))
    p.add_argument("--circuit", default="zebrafish_HD_IPN12_839_v1",
                   help="reference circuit giving the fixed 300-neuron row set")
    p.add_argument("--fishfuncem-data",
                   default=os.path.join(_REPO, "papers", "fishFuncEM", "data"))
    p.add_argument("--warmup-s", type=float, default=10.0)
    p.add_argument("--sequence", default=None,
                   choices=["rotation", "turning", "all"],
                   help="render the REAL zapbench ΔF/F panel for a task block "
                        "instead of the model rollout. Only the blocks with a "
                        "real heading drive are supported: 'rotation' (45°/s "
                        "grating) and 'turning' (omr_turning directional "
                        "labels); 'all' does both. The model checkpoint is not "
                        "needed in this mode.")
    p.add_argument("--nominal-omega", type=float, default=30.0,
                   help="deg/s assigned to omr 'turning' left/right epochs "
                        "(the true-HD trace for the turning block)")
    p.add_argument("--out", default=None,
                   help="output dir (default: <log_dir>/results, or "
                        "figures/zebrafish for --sequence)")
    p.add_argument("--no-title", action="store_true",
                   help="suppress the kinograph title (for montage figures)")
    p.add_argument("--t-window", type=float, default=None,
                   help="display-crop every panel to the first T seconds "
                        "(a zoom-in); the rollout/decode are unchanged. The "
                        "output filename gains a _z<T>s suffix.")
    args = p.parse_args()

    # --sequence: REAL ΔF/F per zapbench block (no model). Handled up front so
    # the trained checkpoint is not required for this data-only mode.
    if args.sequence:
        _render_real_sequences(args)
        return
    if not args.config:
        p.error("--config is required for the model panel (omit only with --sequence)")

    import torch
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.models.gcamp import create_gcamp, list_gcamp
    from connectome_gnn.generators.circuits import get_circuit
    from connectome_gnn.generators.zapbench_stimulus import heading_to_drive
    from connectome_gnn.plot_anatomy_voltage import run_task_rollout
    from connectome_gnn.utils import migrate_state_dict, set_data_root
    set_data_root(os.environ.get("GNN_OUTPUT_ROOT",
                                 "/groups/saalfeld/home/allierc/GraphData"))

    cfg_path = _resolve_config(args.config)
    log_dir = args.log_dir or _log_dir_for(cfg_path)
    stem = os.path.splitext(os.path.basename(cfg_path))[0]
    cfg = NeuralGraphConfig.from_yaml(cfg_path)

    gcamp_name = args.gcamp or getattr(cfg.simulation, "gcamp_kernel", "gcamp7f")
    if gcamp_name not in list_gcamp():
        raise SystemExit(f"--gcamp '{gcamp_name}' unknown; available: {list_gcamp()}")
    gm = create_gcamp(gcamp_name)
    print(f"[config] {cfg_path}\n[log_dir] {log_dir}")
    print(f"[gcamp] {gcamp_name}: tau_rise={gm.tau_rise}s tau_decay={gm.tau_decay}s "
          f"support={gm.support_s():.1f}s")
    if not os.path.isdir(os.path.join(log_dir, "models")):
        raise SystemExit(f"no models/ under {log_dir} — is this a trained run?")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = args.out or os.path.join(log_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    # ---- load the trained model ------------------------------------------- #
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type,
                         config=cfg, device=device).to(device)
    ck = max(glob.glob(f"{log_dir}/models/best_model_with_*.pt"),
             key=os.path.getmtime)
    sd = torch.load(ck, map_location=device, weights_only=False)
    migrate_state_dict(sd)
    model.load_state_dict(sd["model_state_dict"], strict=False)
    model.eval()
    dt = float(model.dt)
    cc = getattr(getattr(cfg, "circuit", None), "name", None) or args.model_circuit
    if cc is None:
        raise SystemExit("no circuit.name in config; pass --model-circuit")
    idx_of = {int(b): i for i, b in
              enumerate(np.asarray(get_circuit(cc).body_ids, dtype=np.int64))}
    print(f"[model] {os.path.basename(ck)}  dt={dt}  N={model.n_units}  circuit={cc}")

    # ---- shared rollout: zapbench_rotation stimulus -> voltage + readout --- #
    pc = cfg.plotting.model_copy(update=dict(
        anatomy_voltage_pattern="zapbench_rotation",
        anatomy_voltage_warmup_s=args.warmup_s,
        anatomy_voltage_zapbench_connectome=args.connectome,
        anatomy_voltage_zapbench_fishfuncem_data=args.fishfuncem_data))
    h, y_hat, _theta_hd, label, extra = run_task_rollout(model, pc, device)
    warm, ss = extra["warm_steps"], extra["sample_steps"]
    theta_frame, n_frames = extra["theta_frame"], extra["n_frames"]
    print(f"[rollout] {label}; voltage {h.shape}, warmup {warm} steps")

    # ---- voltage -> calcium (registry), drop warmup, sample at imaging fps - #
    calcium = gm(h, dt_in=dt)[warm:][ss]                 # (n_frames, N)
    decoded = np.arctan2(y_hat[warm:, 1], y_hat[warm:, 0])[ss]

    # ---- fixed 300-neuron bump row set, fill by bodyId -------------------- #
    rows, _ = panel.build_rows(args.connectome, args.circuit)
    panel_rows = panel.sort_rows_rastermap(
        rows[rows["matched"]].reset_index(drop=True))
    kino = np.full((len(panel_rows), n_frames), np.nan, dtype=np.float32)
    n_fill = 0
    for ri, b in enumerate(panel_rows["bodyId"].to_numpy()):
        mi = idx_of.get(int(b))
        if mi is not None and mi < calcium.shape[1]:
            kino[ri] = calcium[:, mi]; n_fill += 1
    kino = panel._zscore_global(kino)
    print(f"[rows] {len(panel_rows)} mapped bump neurons; filled {n_fill}")

    dd = heading_to_drive(theta_frame, dt=0.915, src_dt=0.915)   # display grid
    d = dict(kino=kino, t_sec=dd["t_sec"], omega=dd["omega"], theta=dd["theta"],
             decoded=decoded, turn_lr=dd["turn_lr"], swim_fb=dd["swim_fb"],
             pred_label="readout decode", omega_label="ω (°/s)")

    win = "" if args.t_window is None else f"_z{int(args.t_window)}s"
    out_png = os.path.join(out_dir, f"functional_panel_{stem}_{gcamp_name}{win}.png")
    title = ("" if args.no_title
             else f"{stem} -> {gcamp_name} calcium — Rotations 45 deg/s")
    # white background; with bg="white" render uses black text, so the predicted
    # HD trace (drawn in the text colour) is black, not white.
    panel.render(d, panel_rows, out_png, title,
                 bg="white", cmap_name="viridis", show_partition=False,
                 show_swim_panels=False, t_window=args.t_window)
    print(f"[done] {out_png}")


if __name__ == "__main__":
    main()


# python figures/zebrafish/fig_functional_panel.py --config zebrafish/zebrafish_hd_si_ipn12_v1_cv0