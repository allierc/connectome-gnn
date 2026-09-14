"""One neuron, read out term by term, and compared with the equation it came from.

WHY ONE NEURON. Fitting g_phi over every edge at once asks the search to find the
synapse's shape AND how its constants vary across 65 cell types in a single
expression, and it answers in the aggregate: an R2 that mixes edges the model got
right with edges it never had information about. Fixing one postsynaptic neuron i
removes both problems. Its embedding a_i is then a constant, each of its incoming
edges has a constant a_j too, so every fit is a two-variable problem in (v_j, v_i)
whose answer can be read and checked by eye against the number the generator used.

WHAT IS FITTED, for one chosen i:

  the neuron      f_theta(a_i, v_i, msg, I) against
                  f_true = (V_rest_i - v_i + msg + I) / tau_i
                  -- linear, and its four coefficients are known: -1/tau_i on
                  v_i, +1/tau_i on msg and on I, V_rest_i/tau_i as the constant.

  each synapse    W_ij^2 * g_phi(v_j, a_j, v_i, a_i) against
                  m_true_ij = W_ij * relu(v_j) * (E_ij - v_i)
                  -- one fit per presynaptic partner j, each a surface over
                  (v_j, v_i) with two known constants, the conductance W_ij and
                  the reversal E_ij.

  the sum         the recovered per-edge expressions added over j, against
                  sum_j m_true_ij, which is the message the neuron actually
                  receives. A model can be right edge by edge and wrong in the
                  sum if the errors share a sign, so the sum is scored
                  separately rather than inferred.

THE COMPARISON IS TO THE TRUE FUNCTION ON THE SAME ROWS, never to a re-derived
trajectory: f_true is evaluated at the msg value the model itself was given, so
the two functions are compared as functions and a wrong message does not
contaminate the verdict on f_theta.

Usage:
    python tools/sr_single_neuron.py --log-dir <run> --pick          # candidates
    python tools/sr_single_neuron.py --log-dir <run> --neuron 1234
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.utils import set_data_root, to_numpy  # noqa: E402

_root = os.environ.get("GNN_OUTPUT_ROOT")
if _root:
    set_data_root(_root)


def _r2(y, pred):
    y, pred = np.asarray(y, float).ravel(), np.asarray(pred, float).ravel()
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else float("nan")


def load(log_dir, device, n_frames):
    from message_decomposition import _load
    config, data, model, ckpt = _load(log_dir, device)
    core = getattr(model, "_orig_mod", model)
    n = int(data.n_neurons)
    frames = np.linspace(0, int(data.x_ts.n_frames) - 1, n_frames).astype(int)
    return config, data, model, core, n, frames, ckpt


def gather(data, model, core, n, frames, device):
    """Per-frame voltage, stimulus, the model's own message and update."""
    V, S, MSG, PRED = [], [], [], []
    edges = data.edges.to(device)
    did = torch.zeros((n, 1), dtype=torch.int, device=device)
    with torch.no_grad():
        for k in frames:
            st = data.x_ts.frame(int(k)).to(device)
            pred, feats, msg = model(st, edges, data_id=did, return_all=True)
            V.append(to_numpy(feats[:, 0]).ravel()[:n])
            MSG.append(to_numpy(msg).ravel()[:n])
            S.append(to_numpy(feats[:, 2 + int(core.a.shape[1])]).ravel()[:n])
            PRED.append(to_numpy(pred).ravel()[:n])
    return (np.stack(V), np.stack(S), np.stack(MSG), np.stack(PRED))


def pick_neuron(data, op, V, n, min_deg=6, max_deg=18, top=12):
    """Candidates: enough presynaptic partners to make a sum worth checking, both
    signs present, the neuron itself active, and its partners active too -- an
    edge whose presynaptic cell never fires carries no information about its
    reversal, as the 13 unrecoverable constants of the earlier run showed."""
    e = to_numpy(data.edges).reshape(2, -1)
    src, dst = e[0], e[1]
    is_inh = to_numpy(op.edge_is_inh).ravel().astype(bool)
    deg = np.bincount(dst, minlength=n)
    act = np.maximum(V, 0).mean(axis=0)          # mean relu(v_j) per neuron
    rows = []
    for i in np.where((deg >= min_deg) & (deg <= max_deg))[0]:
        m = dst == i
        n_inh = int(is_inh[m].sum())
        n_exc = int(m.sum()) - n_inh
        if n_inh == 0 or n_exc == 0:
            continue
        rows.append((i, int(m.sum()), n_exc, n_inh, float(V[:, i].std()),
                     float(act[src[m]].mean()), float(act[src[m]].min())))
    if not rows:
        return []
    rows.sort(key=lambda r: -(r[4] * r[6]))      # active neuron, all partners active
    return rows[:top]


def fit(X, y, names, niterations=30, maxsize=15, guess=None):
    from pysr import PySRRegressor
    kw = dict(niterations=niterations,
              operators={2: ["+", "-", "*"], 1: ["relu"]},
              maxsize=maxsize, progress=False, temp_equation_file=True, verbosity=0)
    if guess:
        kw["guesses"] = guess
    m = PySRRegressor(**kw)
    m.fit(X, y, variable_names=names)
    best = m.get_best()
    return str(best["equation"]), np.asarray(m.predict(X)).ravel()


def run(log_dir, neuron, n_frames=1024, device="cpu", niterations=30, out=None):
    config, data, model, core, n, frames, ckpt = load(log_dir, device, n_frames)
    op = data.ode_params
    V, S, MSG, PRED = gather(data, model, core, n, frames, device)

    e = to_numpy(data.edges).reshape(2, -1)
    src, dst = e[0], e[1]
    W_true = to_numpy(op.W).ravel()
    E_true = to_numpy(op.reversal_per_edge()).ravel()
    is_inh = to_numpy(op.edge_is_inh).ravel().astype(bool)
    tau = np.asarray(op.gt_tau(n), float)[neuron]
    vrest = np.asarray(op.gt_vrest(n), float)[neuron]
    W_model = to_numpy(__import__("connectome_gnn.metrics", fromlist=["get_model_W"])
                       .get_model_W(core)).ravel()

    label = os.path.basename(log_dir.rstrip("/"))
    lines = [f"run {label}   checkpoint {ckpt}",
             f"neuron {neuron}   tau {tau:.4f}   V_rest {vrest:+.4f}   "
             f"frames {len(frames)}"]

    # ---------------- the neuron ----------------
    v_i, msg_i, I_i, pred_i = V[:, neuron], MSG[:, neuron], S[:, neuron], PRED[:, neuron]
    f_true = (vrest - v_i + msg_i + I_i) / tau
    Xf = np.column_stack([v_i, msg_i, I_i]).astype(np.float64)
    eq_f, pred_f = fit(Xf, pred_i.astype(np.float64), ["v_i", "msg", "I"], niterations,
                       guess=[f"({vrest:.4f} - v_i + msg + I) * {1.0/tau:.4f}"])
    lines += ["", "-- f_theta --",
              f"true      ({vrest:+.4f} - v_i + msg + I) * {1.0 / tau:.4f}",
              f"recovered {eq_f}",
              f"R2(recovered, model f_theta) {_r2(pred_i, pred_f):+.4f}",
              f"R2(model f_theta, true f)    {_r2(f_true, pred_i):+.4f}"]

    # ---------------- each synapse ----------------
    sel = np.where(dst == neuron)[0]
    lines += ["", f"-- {len(sel)} synapses onto neuron {neuron} --",
              f"{'j':>7} {'sign':>4} {'W_true':>8} {'E_true':>8} {'R2 rec':>8} "
              f"{'R2 model':>9}  recovered"]
    sum_rec = np.zeros_like(v_i)
    sum_true = np.zeros_like(v_i)
    sum_model = np.zeros_like(v_i)
    for idx in sel:
        j = int(src[idx])
        v_j = V[:, j]
        m_true = W_true[idx] * np.maximum(v_j, 0.0) * (E_true[idx] - v_i)
        with torch.no_grad():
            from connectome_gnn.models.utils import pad_g_phi_input
            emb = core.a.detach()
            feat = torch.cat([torch.as_tensor(v_j, dtype=torch.float32, device=device)[:, None],
                              emb[j].expand(len(v_j), -1),
                              torch.as_tensor(v_i, dtype=torch.float32, device=device)[:, None],
                              emb[neuron].expand(len(v_j), -1)], dim=1)
            g = core.g_phi(pad_g_phi_input(feat, core)).ravel()
            if getattr(core, "g_phi_positive", False):
                g = g ** 2
            w = W_model[idx] ** 2 if getattr(core, "w_squared", False) else W_model[idx]
            m_model = to_numpy(g) * float(w)
        X = np.column_stack([v_j, v_i]).astype(np.float64)
        eq, pred_e = fit(X, m_model.astype(np.float64), ["v_j", "v_i"], niterations)
        sum_rec += pred_e
        sum_true += m_true
        sum_model += m_model
        lines.append(f"{j:>7} {'inh' if is_inh[idx] else 'exc':>4} {W_true[idx]:>8.4f} "
                     f"{E_true[idx]:>+8.3f} {_r2(m_model, pred_e):>8.3f} "
                     f"{_r2(m_true, m_model):>9.3f}  {eq}")

    lines += ["", "-- summed over the presynaptic partners --",
              f"R2(sum of recovered, sum of model)  {_r2(sum_model, sum_rec):+.4f}",
              f"R2(sum of model,     true message)  {_r2(sum_true, sum_model):+.4f}",
              f"R2(sum of recovered, true message)  {_r2(sum_true, sum_rec):+.4f}",
              f"std: true {sum_true.std():.4f}  model {sum_model.std():.4f}  "
              f"recovered {sum_rec.std():.4f}"]

    text = "\n".join(lines)
    print(text)
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"neuron{neuron}_{label}.txt"), "w") as f:
            f.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--neuron", type=int, default=None)
    ap.add_argument("--pick", action="store_true")
    ap.add_argument("--frames", type=int, default=1024)
    ap.add_argument("--niterations", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.pick:
        config, data, model, core, n, frames, _ = load(args.log_dir, args.device, 256)
        V, *_ = gather(data, model, core, n, frames, args.device)
        rows = pick_neuron(data, data.ode_params, V, n)
        print(f"{'neuron':>7} {'deg':>4} {'exc':>4} {'inh':>4} {'v_i std':>8} "
              f"{'mean relu(v_j)':>15} {'min relu(v_j)':>14}")
        for r in rows:
            print(f"{r[0]:>7} {r[1]:>4} {r[2]:>4} {r[3]:>4} {r[4]:>8.3f} "
                  f"{r[5]:>15.4f} {r[6]:>14.4f}")
        return
    if args.neuron is None:
        ap.error("--neuron is required unless --pick")
    run(args.log_dir, args.neuron, args.frames, args.device, args.niterations, args.out)


if __name__ == "__main__":
    main()
