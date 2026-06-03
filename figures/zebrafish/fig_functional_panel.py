"""Functional panel figure: given a trained-model CONFIG, render the model
functional panel (voltage -> GCaMP calcium -> kinograph) for the zebrafish
HD/IPN12 circuit, exercising the connectome_gnn.models.gcamp registry.

A quick "does this checkpoint produce a sensible bump kinograph?" view — the
same model panel that scripts/zebrafish_functional_traces_panel.py builds, but
driven from a single config path and with the GCaMP indicator selectable by
name from the registry.

  log_dir is derived from the config path:  .../config/<sub>/<name>.yaml
                                       ->   .../log/<sub>/<name>
  circuit comes from the config's circuit.name (else --model-circuit).

Run (env with fishfuncem + torch + the trained model):
  /workspace/.conda_envs/neural-graph-linux/bin/python \
      figures/zebrafish/fig_functional_panel.py \
      --config /groups/.../config/zebrafish/zebrafish_hd_si_gnn_dipn_v1_cv0 \
      --gcamp gcamp7f
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "papers", "fishFuncEM"))

# Reuse the heavy lifting (row set, stimulus, model_panel, render) from the
# main panel script — single source of truth so the sanity panel matches.
_PANEL_PY = os.path.join(_REPO, "scripts", "zebrafish_functional_traces_panel.py")
_spec = importlib.util.spec_from_file_location("_ftp", _PANEL_PY)
panel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel)


def _resolve_config(arg: str) -> str:
    """Accept a full path (with or without .yaml) or a bare name, return the
    .yaml path. Bare names are searched under $GNN_OUTPUT_ROOT/config and the
    repo's config/."""
    a = arg if arg.endswith(".yaml") else arg + ".yaml"
    if os.path.isabs(a):
        cands = [a]                              # absolute: don't build fallbacks
    else:
        root = os.environ.get("GNN_OUTPUT_ROOT",
                              "/groups/saalfeld/home/allierc/GraphData")
        cands = [os.path.join(root, "config", a), os.path.join(_REPO, "config", a),
                 os.path.join(_REPO, a), a]
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"config not found for '{arg}' (tried {cands})")


def _log_dir_for(cfg_path: str) -> str:
    """.../config/<sub>/<name>.yaml -> .../log/<sub>/<name> (the trainer's rule)."""
    stem = cfg_path[:-5] if cfg_path.endswith(".yaml") else cfg_path
    if os.sep + "config" + os.sep in stem:
        return stem.replace(os.sep + "config" + os.sep, os.sep + "log" + os.sep)
    return os.path.join(os.path.dirname(stem), "log", os.path.basename(stem))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="config path or name (e.g. zebrafish/zebrafish_hd_si_ipn12_v1)")
    p.add_argument("--gcamp", default="gcamp7f",
                   help="GCaMP indicator name from the connectome_gnn.models."
                        "gcamp registry (e.g. gcamp6f/6s/7f/8f/8m); default gcamp7f")
    p.add_argument("--log-dir", default=None, help="override derived log dir")
    p.add_argument("--model-circuit", default=None,
                   help="circuit name when the config has no circuit.name "
                        "(e.g. zebrafish_HD_731_v1 for the 731-cell dipn runs)")
    p.add_argument("--connectome",
                   default=os.path.join(_REPO, "figures", "zebrafish",
                                        "zebrafish_connectome_HD_IPN12"))
    p.add_argument("--circuit", default="zebrafish_HD_IPN12_839_v1",
                   help="reference circuit giving the fixed 300-neuron row set")
    p.add_argument("--fishfuncem-data",
                   default=os.path.join(_REPO, "papers", "fishFuncEM", "data"))
    p.add_argument("--model-dt", type=float, default=0.01)
    p.add_argument("--warmup-s", type=float, default=10.0)
    p.add_argument("--out", default=os.path.join(_REPO, "figures", "zebrafish"),
                   help="output dir (default: the figures/zebrafish root)")
    args = p.parse_args()

    from connectome_gnn.models.gcamp import create_gcamp, list_gcamp
    if args.gcamp not in list_gcamp():
        raise SystemExit(f"--gcamp '{args.gcamp}' unknown; available: {list_gcamp()}")
    gm = create_gcamp(args.gcamp)
    print(f"[gcamp] {args.gcamp}: tau_rise={gm.tau_rise}s tau_decay={gm.tau_decay}s "
          f"support={gm.support_s():.1f}s")

    cfg_path = _resolve_config(args.config)
    log_dir = args.log_dir or _log_dir_for(cfg_path)
    stem = os.path.splitext(os.path.basename(cfg_path))[0]
    print(f"[config] {cfg_path}\n[log_dir] {log_dir}")
    if not os.path.isdir(os.path.join(log_dir, "models")):
        raise SystemExit(f"no models/ under {log_dir} — is this a trained run?")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    rows, _ = panel.build_rows(args.connectome, args.circuit)
    panel_rows = panel.sort_rows_rastermap(
        rows[rows["matched"]].reset_index(drop=True))
    print(f"[rows] {len(panel_rows)} mapped bump neurons (rastermap order)")

    # Rotations block — the real 45 deg/s grating heading, same as the curated
    # panels; the model is run at model_dt then sampled at imaging frames.
    z = np.load(os.path.join(args.connectome, "functional",
                             "circuit_functional_traces.npz"), allow_pickle=True)
    theta_hr, ts, theta_frame = panel.rotation_headings(
        args.fishfuncem_data, args.connectome, args.model_dt)
    keep = ts < z["traces"].shape[0]
    ts = ts[keep]
    theta_frame = theta_frame[: len(ts)]
    drive_real = panel.heading_to_drive(theta_frame, dt=0.915, src_dt=0.915)
    drive_model = panel.heading_to_drive(theta_hr, dt=args.model_dt,
                                         src_dt=args.model_dt)
    sample_steps = np.round(np.arange(len(ts)) * 0.915 / args.model_dt).astype(np.int64)

    d = panel.model_panel(
        panel_rows, log_dir, drive_model, device,
        model_circuit=args.model_circuit, warmup_s=args.warmup_s,
        config_path=cfg_path, sample_steps=sample_steps, display=drive_real,
        gcamp=dict(tau_rise=gm.tau_rise, tau_decay=gm.tau_decay, length_s=None))

    out_png = os.path.join(args.out, f"functional_panel_{stem}_{args.gcamp}.png")
    panel.render(d, panel_rows, out_png,
                 f"{stem} -> {args.gcamp} calcium — Rotations 45 deg/s",
                 cmap_name="viridis", show_partition=False)
    print(f"[done] {out_png}")


if __name__ == "__main__":
    main()
