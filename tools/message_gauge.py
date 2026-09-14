"""Is the message gauge affine, and does its offset explain the V_rest error?

The model's message is only ever seen through the update, so it is free up to a
transform that the update can undo. The panels so far assumed that transform was a
pure scale, msg_model = msg_true / k with k = T * G * tau, and corrected panel c by
multiplying. But a CONSTANT is equally invisible: writing

    msg_true = a * msg_model + b

and matching the model's T * [(V - v_i) + G * msg_model] to the generator's
(V_rest - v_i + msg_true) / tau leaves a = k on the message coefficient and, on the
constants,

    V_rest = tau * T * V - b

so b is a level the model carries in its message and the update hands straight to
V_rest. The same identity holds for the slope-based readout, where f_theta is
evaluated at msg = 0 and fitted over each neuron's voltage range: if f_theta is
affine, f = c0 + c1*v + c2*msg, then matching gives c1 = -1/tau and

    V_rest_learned = -c0 / c1 = V_rest_true + b

This script measures a and b per neuron by least squares over real frames, then
tests exactly that: does V_rest_learned - b recover V_rest_true better than
V_rest_learned alone? If it does, the V_rest error is the message's offset and
nothing else, and coeff_g_phi_silent -- which drives b to zero by forcing the
message to vanish at silent presynaptic input -- is the lever on it.

Two residuals are reported alongside, because they say where the model departs from
the affine gauge:

  resid   how much of msg_true the affine fit leaves unexplained. Large means the
          gauge is not affine at all, e.g. because the conductance message carries
          its own -v_i term, msg = A(t) - B(t) * v_i, whose B can slide into the
          leak rate rather than into V_rest.
  tau      reported for the same reason: under an affine f_theta the leak rate is
          NOT touched by a or b, so a tau error is evidence for the B(t) route.

Usage:
    python tools/message_gauge.py <config> [<config> ...] [--frames 40] [--json out.json]
"""
import argparse
import json
import logging
import os
import sys

import numpy as np
import torch


def _fit_per_neuron(msg_true, msg_model):
    """Least squares msg_true ~ a * msg_model + b, one fit per neuron.

    Arrays are (n_frames, n_neurons). Returns a, b, r and the residual standard
    deviation as a fraction of msg_true's own, each (n_neurons,). Neurons whose
    model message never moves get a = nan: there is no slope to fit, and passing
    them through would put a divide-by-zero into the summary.
    """
    x, y = msg_model, msg_true
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    sxx = (xc * xc).sum(0)
    sxy = (xc * yc).sum(0)
    syy = (yc * yc).sum(0)
    ok = sxx > 1e-20
    a = np.where(ok, sxy / np.where(ok, sxx, 1.0), np.nan)
    b = np.where(ok, ym - a * xm, np.nan)
    r = np.where(ok & (syy > 1e-20), sxy / np.sqrt(np.where(ok, sxx, 1.0)
                                                   * np.where(syy > 1e-20, syy, 1.0)), np.nan)
    resid = y - (a * x + b)
    rel = resid.std(0) / np.maximum(y.std(0), 1e-12)
    return a, b, r, rel


def _collect(model, data, device, n_frames):
    """msg_true and the model's RAW msg, (n_frames, n_neurons) each.

    Deliberately the raw message off forward(return_all=True), NOT the one
    compute_msg_i_recovery normalises through f_theta's leak slope: that division
    already folds part of the gauge away, which is the very thing being measured.
    Frames are evenly spaced over the recording so two runs are compared on the
    same stretch of stimulus.
    """
    from connectome_gnn.utils import to_numpy

    op = data.ode_params
    x_ts = data.x_ts
    edges = data.edges.to(device)
    src, dst = edges[0], edges[1]
    n = int(data.n_neurons)
    idx = np.linspace(0, int(x_ts.n_frames) - 1, n_frames).astype(int)

    W = torch.as_tensor(to_numpy(op.W).ravel(), dtype=torch.float32, device=device)
    E_edge = None
    if getattr(op, "E_exc", None) is not None:
        E_edge = op.reversal_per_edge().to(device).float().ravel()
    did = torch.zeros((n, 1), dtype=torch.int, device=device)

    TRUE, MODEL = [], []
    model.eval()
    with torch.no_grad():
        for k in idx:
            st = x_ts.frame(int(k)).to(device)
            v = st.voltage.float().ravel()
            act = torch.as_tensor(np.asarray(op.gt_g_phi_func(to_numpy(v[src]))),
                                  dtype=torch.float32, device=device).ravel()
            edge_msg = W[:act.numel()] * act
            if E_edge is not None:
                edge_msg = edge_msg * (E_edge[:act.numel()] - v[dst][:act.numel()])
            mt = torch.zeros(n, device=device)
            mt.scatter_add_(0, dst[:edge_msg.numel()], edge_msg)
            _, _, mm = model(st, edges, data_id=did, return_all=True)
            TRUE.append(to_numpy(mt).ravel()[:n])
            MODEL.append(to_numpy(mm).ravel()[:n])
    return np.stack(TRUE).astype(float), np.stack(MODEL).astype(float)


def _r2(truth, est, keep):
    t, e = truth[keep], est[keep]
    if t.size < 3:
        return float("nan")
    ss = ((t - t.mean()) ** 2).sum()
    return float(1.0 - ((t - e) ** 2).sum() / ss) if ss > 0 else float("nan")


def run_one(config_name, frames, device):
    from connectome_gnn.config import NeuralGraphConfig
    from connectome_gnn.models.training_utils import init_training_data
    from connectome_gnn.models.registry import create_model
    from connectome_gnn.utils import migrate_state_dict
    from connectome_gnn.metrics import extract_recovered_params
    import glob

    from connectome_gnn.utils import log_path
    from connectome_gnn.models.utils import load_run_config
    cfg, _ = load_run_config(config_name, False, "plot")
    # The same derivation GNN_Main uses for -o plot (GNN_Main.py:223), so this
    # reads the directory the panels and metrics.txt were written into.
    log_dir = log_path(cfg.config_file)

    log = logging.getLogger(__name__)
    data = init_training_data(cfg, device, log_dir, log)
    model = create_model(cfg.graph_model.signal_model_name,
                         aggr_type=cfg.graph_model.aggr_type, config=cfg,
                         device=device).to(device)
    ck = sorted(glob.glob(os.path.join(log_dir, "models",
                                       "best_model_with_*_graphs_*.pt")))
    if not ck:
        raise FileNotFoundError(f"no checkpoint under {log_dir}/models")
    sd = torch.load(ck[-1], map_location=device, weights_only=False)
    migrate_state_dict(sd)
    _, unexpected = model.load_state_dict(sd["model_state_dict"], strict=False)
    loaded = len(sd["model_state_dict"]) - len(unexpected)
    if loaded == 0:
        raise RuntimeError("checkpoint loaded 0 tensors")
    model.eval()

    msg_true, msg_model = _collect(model, data, device, frames)
    a, b, r, rel = _fit_per_neuron(msg_true, msg_model)

    rec = extract_recovered_params(model, data.ode_params, cfg, edges=data.edges,
                                   device=device, n_neurons=int(data.n_neurons),
                                   need=("tau", "V_rest"))
    out = {"config": config_name, "checkpoint": os.path.basename(ck[-1]),
           "n_frames": frames, "n_neurons": int(data.n_neurons)}

    pair = rec.get("V_rest")
    keep = np.isfinite(a) & np.isfinite(b)
    if pair is not None:
        vt, vl = np.asarray(pair[0], float), np.asarray(pair[1], float)
        m = keep & np.isfinite(vt) & np.isfinite(vl)
        out["vrest_R2_raw"] = _r2(vt, vl, m)
        out["vrest_R2_minus_b"] = _r2(vt, vl - b, m)
        out["n_vrest"] = int(m.sum())
    pair = rec.get("tau")
    if pair is not None:
        tt, tl = np.asarray(pair[0], float), np.asarray(pair[1], float)
        m = np.isfinite(tt) & np.isfinite(tl)
        out["tau_R2"] = _r2(tt, tl, m)
        out["tau_ratio_median"] = float(np.median((tl / tt)[m])) if m.any() else float("nan")

    for name, arr in (("a", a), ("b", b), ("r", r), ("resid", rel)):
        v = arr[keep]
        out[f"{name}_median"] = float(np.median(v)) if v.size else float("nan")
        out[f"{name}_iqr"] = float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else float("nan")
    out["n_fitted"] = int(keep.sum())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    rows = []
    for name in args.configs:
        try:
            rows.append(run_one(name, args.frames, args.device))
            print(f"[ok] {name}")
        except Exception as exc:
            print(f"[fail] {name}: {type(exc).__name__}: {exc}")

    if not rows:
        return 1
    cols = ["config", "a_median", "b_median", "r_median", "resid_median",
            "vrest_R2_raw", "vrest_R2_minus_b", "tau_R2", "tau_ratio_median"]
    w = [max(len(c), *(len(f"{row.get(c, float('nan')):.4g}")
                       if isinstance(row.get(c), float) else len(str(row.get(c, "")))
                       for row in rows)) for c in cols]
    print()
    print("  ".join(c.ljust(wi) for c, wi in zip(cols, w)))
    for row in rows:
        cells = []
        for c, wi in zip(cols, w):
            v = row.get(c, float("nan"))
            cells.append((f"{v:.4g}" if isinstance(v, float) else str(v)).ljust(wi))
        print("  ".join(cells))
    print("\nvrest_R2_minus_b > vrest_R2_raw means the V_rest error IS the message's "
          "offset b, which coeff_g_phi_silent is the lever on.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
