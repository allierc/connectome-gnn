"""Where a conductance GNN's error sits: in the message, or swapped with the leak.

THE QUESTION THIS ANSWERS. A trained GNN can reproduce dv/dt while getting the
message badly wrong, because only the SUM of the two terms is observed:

    dv_i/dt = f_theta(a_i, v_i, stim_i [, msg_i]) + (message contribution)

Anything that depends on the postsynaptic neuron's own state -- a constant c_i,
or a term -k_i v_i that reads as extra leak -- can be moved from one term to the
other with the sum, and therefore the rollout, unchanged. On the additive form
(`additive_message`) the swap is exact and explicit; on the standard form the
message passes through f_theta, and the same freedom exists with a per-neuron
gain on top of it.

WHAT IS COMPARED, in dv/dt units (the generator's ODE is
dv_i/dt = (-v_i + V_rest_i + I_i + sum_j msg_ij) / tau_i, so BOTH terms carry
the 1/tau_i):

    m_true_i  = (sum_j W_ij relu(v_j) (E_ij - v_i)) / tau_i    true message term
    f_true_i  = (-v_i + V_rest_i + stim_i) / tau_i             true own-state term
    m_learn_i = pred_i - f_theta(..., msg = 0, ...)            learned message term
    f_learn_i = f_theta(..., msg = 0, ...)                     learned own-state term

`m_learn` is defined by DIFFERENCE rather than as the raw aggregate, so the same
definition works for both forms and is the message's actual contribution to
dv/dt -- the quantity the trajectory sees.

WHAT IS REPORTED

  r2_dvdt        R2 of (f_learn + m_learn) against the true derivative. The
                 rollout in other words: near 1 means the SUM is right, which is
                 the premise of everything below.
  r2_msg         R2 of m_learn against m_true. The message itself.
  corr_swap      Pearson r between (f_learn - f_true) and (m_learn - m_true) over
                 all (frame, neuron) pairs. -1 means every error in one term is
                 cancelled by the other: a pure swap, invisible to the rollout.
                 ONLY MEANINGFUL WHEN r2_dvdt IS NEAR 1. A model that learned
                 nothing also scores about -1, because both residuals are then
                 just minus the true terms, which anticorrelate by themselves.
  swap_share     Share of the message error's variance that is explained, per
                 neuron, by a linear function of the neuron's OWN state,
                 r_i = a_i + b_i v_i + c_i stim_i. That part is exactly what the
                 sum cannot see. What is left is a real message error.
  r2_msg_ident   THE HEADLINE. R2 of the learned against the true message after
                 the same per-neuron projection onto (1, v_i, stim_i) is removed
                 from both -- the part of the message the trajectory can pin
                 down at all. Fitted on the even frames, scored on the odd ones.
  identifiable_share  How much of the true message's variance that residual is,
                 per neuron. The rest is degenerate with the leak by
                 construction, because the driving force carries -v_i times the
                 conductance sum.
  swap_share     Share of the message ERROR that is such an own-state function,
                 i.e. invisible to the sum.
  b_true_x_tau / b_learn_x_tau  Median per-neuron regression coefficient of the
                 true and of the learned message on v_i, times tau_i. NOT the
                 partial derivative: the driving force contributes
                 -sum_j W_ij relu(v_j), but the presynaptic drive rises with v_i
                 as well, so the total association can come out either sign. The
                 two are comparable to each other, and a learned value near zero
                 with a true value far from it means the model put none of that
                 dependence in the message.

Usage:
    python tools/message_decomposition.py <log_dir> [<log_dir> ...] [--frames 32]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.utils import set_data_root  # noqa: E402

# GNN_Main does this from --output_root / GNN_OUTPUT_ROOT before anything reads a
# path; a standalone tool has to do it too or every graphs_data path comes out
# relative to the working directory.
_root = os.environ.get("GNN_OUTPUT_ROOT")
if _root:
    set_data_root(_root)

from connectome_gnn.models.registry import create_model  # noqa: E402
from connectome_gnn.models.training_utils import init_training_data  # noqa: E402
from connectome_gnn.models.utils import load_run_config  # noqa: E402
from connectome_gnn.utils import migrate_state_dict, to_numpy  # noqa: E402


def _r2(true, pred):
    """Identity-line R2 (never a correlation), over finite pairs."""
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    ok = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[ok], pred[ok]
    if true.size < 2:
        return float("nan")
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _load(log_dir, device, untrained=False):
    """The run's config, data, and the model at its last checkpoint."""
    # The run's own config.yaml, through the normal loader. It sits in the log
    # dir, whose name is the run rather than the domain, so it is staged under a
    # directory named for its domain (from config_file: "<domain>/<name>") --
    # that is what load_run_config reads the dataset prefix from.
    import shutil
    import tempfile
    import yaml as _yaml
    cfg_path = os.path.join(log_dir, "config.yaml")
    domain = (_yaml.safe_load(open(cfg_path)).get("config_file", "fly/x").split("/")[0]) or "fly"
    tmp = tempfile.mkdtemp()
    staged = os.path.join(tmp, domain)
    os.makedirs(staged, exist_ok=True)
    staged_cfg = os.path.join(staged, os.path.basename(log_dir.rstrip("/")) + ".yaml")
    shutil.copy(cfg_path, staged_cfg)
    config, _ = load_run_config(staged_cfg, True, "test")
    import logging
    data = init_training_data(config, device, log_dir, logging.getLogger(__name__))
    import glob
    ckpts = sorted(glob.glob(os.path.join(log_dir, "models", "best_model_with_*_graphs_*.pt")))
    if not ckpts and not untrained:
        raise FileNotFoundError(f"no checkpoint in {log_dir}/models")
    model = create_model(config.graph_model.signal_model_name,
                         aggr_type=config.graph_model.aggr_type, config=config,
                         device=device).to(device)
    if ckpts:
        sd = torch.load(ckpts[-1], map_location=device, weights_only=False)
        # Checkpoints are written from a torch.compile-wrapped model, so every
        # key carries an `_orig_mod.` prefix; migrate_state_dict strips it. With
        # strict=False and no strip, load_state_dict silently loads NOTHING and
        # the measurement is of an untrained model -- which is exactly what
        # happened the first time this ran.
        migrate_state_dict(sd)
        missing, unexpected = model.load_state_dict(sd["model_state_dict"], strict=False)
        loaded = len(sd["model_state_dict"]) - len(unexpected)
        if loaded == 0:
            raise RuntimeError(f"checkpoint {ckpts[-1]} loaded 0 tensors "
                               f"(unexpected keys: {list(unexpected)[:4]})")
        print(f"  loaded {loaded}/{len(sd['model_state_dict'])} tensors from {os.path.basename(ckpts[-1])}"
              + (f", {len(missing)} missing" if missing else ""))
    model.eval()
    return config, data, model, os.path.basename(ckpts[-1]) if ckpts else "UNTRAINED"


def decompose(log_dir, n_frames=64, device="cuda:0", untrained=False):
    config, data, model, ckpt = _load(log_dir, device, untrained)
    op, x_ts, edges = data.ode_params, data.x_ts, data.edges.to(device)
    n = int(data.n_neurons)
    emb = int(config.graph_model.embedding_dim)
    msg_col = 1 + emb

    tau = torch.as_tensor(np.asarray(op.gt_tau(n), dtype=np.float32), device=device)
    vrest = torch.as_tensor(np.asarray(op.gt_vrest(n), dtype=np.float32), device=device)
    W = torch.as_tensor(to_numpy(op.W).ravel(), dtype=torch.float32, device=device)
    E_edge = op.reversal_per_edge().to(device).float().ravel()
    src, dst = edges[0], edges[1]
    data_id = torch.zeros((n, 1), dtype=torch.int, device=device)

    frames = np.linspace(0, int(x_ts.n_frames) - 1, n_frames).astype(int)
    F, M_T, F_T, M_L, F_L, V, S = [], [], [], [], [], [], []
    with torch.no_grad():
        for k in frames:
            state = x_ts.frame(int(k)).to(device)
            v = state.voltage.float().ravel()[:n]
            stim = state.stimulus.float().ravel()[:n]

            act = torch.as_tensor(np.asarray(op.gt_g_phi_func(to_numpy(v[src]))),
                                  dtype=torch.float32, device=device).ravel()
            edge_msg = W[:act.numel()] * act * (E_edge[:act.numel()] - v[dst][:act.numel()])
            msg_true = torch.zeros(n, device=device)
            msg_true.scatter_add_(0, dst[:edge_msg.numel()], edge_msg)

            pred, feats, _ = model(state, edges, data_id=data_id, return_all=True)
            x0 = feats.clone()
            x0[:, msg_col] = 0.0
            f_learn = model._orig_mod._run_mlp(model._orig_mod.f_theta, x0) \
                if hasattr(model, "_orig_mod") else model._run_mlp(model.f_theta, x0)

            M_T.append(to_numpy(msg_true / tau))
            F_T.append(to_numpy((-v + vrest + stim) / tau))
            M_L.append(to_numpy((pred.ravel()[:n] - f_learn.ravel()[:n])))
            F_L.append(to_numpy(f_learn.ravel()[:n]))
            V.append(to_numpy(v)); S.append(to_numpy(stim))
    m_t, f_t = np.stack(M_T), np.stack(F_T)          # (K, N)
    m_l, f_l = np.stack(M_L), np.stack(F_L)
    v_, s_ = np.stack(V), np.stack(S)

    # --- the swap: is the message error cancelled by the leak error? ---
    dm, df = m_l - m_t, f_l - f_t
    ok = np.isfinite(dm) & np.isfinite(df)
    corr_swap = float(np.corrcoef(df[ok], dm[ok])[0, 1]) if ok.sum() > 2 else float("nan")

    # --- WHAT IS EVEN IDENTIFIABLE ---
    #
    # Only the SUM of the two terms is observed, so any part of the message that
    # is a function of the postsynaptic neuron's own state can be moved into
    # f_theta and back. The true message HAS such a part, and a large one: the
    # driving force contributes -v_i * sum_j W_ij relu(v_j) / tau_i, and the
    # conductance sum varies slowly, so the true message is itself substantially
    # linear in v_i. Removing it from the learned message alone therefore
    # flatters the result -- an untrained model scored 0.92 that way.
    #
    # So the same projection is applied to BOTH sides: per neuron, the component
    # of the message along (1, v_i, stim_i) is removed from m_true and from
    # m_learn, and what remains is compared. That residual is the part of the
    # message the trajectory can pin down; `identifiable_share` says how much of
    # the true message's variance it is. Coefficients fitted on the even frames,
    # everything scored on the odd ones.
    K = m_t.shape[0]
    fit_k, test_k = np.arange(0, K, 2), np.arange(1, K, 2)
    X_all = np.transpose(np.stack([np.ones_like(v_), v_, s_], axis=-1), (1, 0, 2))   # (N, K, 3)
    Xf = X_all[:, fit_k, :]
    XtX = np.einsum("nki,nkj->nij", Xf, Xf) + 1e-8 * np.eye(3)

    def _project_out(a):
        """Per neuron, remove the part of `a` along (1, v_i, stim_i); coefficients
        from the even frames, residual returned on the odd ones."""
        beta = np.linalg.solve(XtX, np.einsum("nki,nkj->nij", Xf, a.T[:, fit_k, None]))
        fit_t = np.einsum("nki,nij->nkj", X_all[:, test_k, :], beta)[:, :, 0].T
        return a[test_k] - fit_t, beta[:, 1, 0]

    r_t, b_true = _project_out(m_t)
    r_l, b_learn = _project_out(m_l)
    r_dm, b_coef = _project_out(dm)

    with np.errstate(invalid="ignore", divide="ignore"):
        ident = np.nanvar(r_t, axis=0) / np.nanvar(m_t[test_k], axis=0)
        swap = 1.0 - np.nanvar(r_dm, axis=0) / np.nanvar(dm[test_k], axis=0)
    identifiable_share = float(np.nanmedian(ident))
    swap_share = float(np.nanmedian(swap))

    return {
        "run": os.path.basename(log_dir.rstrip("/")),
        "checkpoint": ckpt,
        "additive": bool(getattr(config.graph_model, "additive_message", False)),
        "n_frames": int(K),
        "r2_dvdt": _r2(m_t + f_t, m_l + f_l),
        "r2_f": _r2(f_t, f_l),
        "corr_swap": corr_swap,
        "swap_share": swap_share,
        "r2_msg": _r2(m_t[test_k], m_l[test_k]),
        "r2_msg_ident": _r2(r_t, r_l),
        "identifiable_share": identifiable_share,
        "b_median": float(np.nanmedian(b_coef)),
        "b_true_x_tau": float(np.nanmedian(b_true * to_numpy(tau))),
        "b_learn_x_tau": float(np.nanmedian(b_learn * to_numpy(tau))),
        "b_median_x_tau": float(np.nanmedian(b_coef * to_numpy(tau))),
        "msg_true_std": float(np.nanstd(m_t)),
        "msg_learn_std": float(np.nanstd(m_l)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dirs", nargs="+")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--untrained", action="store_true",
                    help="run on a freshly initialised model (control, and a way to "
                         "exercise the tool before the first checkpoint lands)")
    args = ap.parse_args()
    rows = []
    for d in args.log_dirs:
        try:
            rows.append(decompose(d, args.frames, args.device, args.untrained))
        except Exception as exc:
            print(f"{os.path.basename(d.rstrip('/')):55s} FAILED {type(exc).__name__}: {exc}")
    if not rows:
        return
    keys = ["r2_dvdt", "r2_msg", "r2_msg_ident", "identifiable_share", "r2_f",
            "corr_swap", "swap_share", "b_true_x_tau", "b_learn_x_tau",
            "msg_true_std", "msg_learn_std"]
    print(f"\n{'run':52s} {'add':>4s} " + " ".join(f"{k:>16s}" for k in keys))
    for r in rows:
        print(f"{r['run'][:52]:52s} {str(r['additive']):>4s} "
              + " ".join(f"{r[k]:16.3f}" for k in keys))


if __name__ == "__main__":
    main()
