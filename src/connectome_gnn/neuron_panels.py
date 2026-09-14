"""One postsynaptic neuron, opened up: its trajectory, its terms, its equations.

WHAT THIS ANSWERS. A trained model can reproduce the dynamics while getting the
synapses badly wrong, because only the SUM of the neuron's own dynamics and its
incoming message is observed. Measured on a conductance GNN whose free-running
rollout reached r = 0.9993 at one neuron: its total message was ANTI-correlated
with the truth at r = -0.967 and seven times too large, f_theta took that message
with a coefficient of -1.64 where the generator uses +1/tau = +11.1, and the two
errors multiplied back to +11.8. The trajectory was right because the mistakes
cancelled. Aggregate scores cannot see this; one neuron, term by term, can.

WHY ONE NEURON MAKES THE EQUATIONS READABLE. Fix the postsynaptic neuron i and
its embedding a_i is a constant; fix one of its incoming edges and a_j is a
constant too. Every per-synapse fit is then a two-variable problem in (v_j, v_i)
whose answer can be read against the conductance and reversal the generator used,
instead of one aggregate expression that has to cover 65 cell types at once.

THE PANELS, all on the same consecutive frames of the same split so they line up:

    a   voltage, the free-running rollout against the generator's trajectory
    b   dv/dt at the true voltages, the model's update against the generator's
    c   the total incoming message
    d   every incoming synapse separately, z-scored, so shape can be compared
        independently of amplitude -- which is usually where they differ
    e   the closed forms: the generator's, with its own constants substituted,
        against what symbolic regression reads out of the trained model

BOTH MESSAGE FAMILIES ARE HANDLED. A current generator's per-edge message is
W_ij * act(v_j), with W carrying the synapse's sign; a conductance generator's is
W_ij * act(v_j) * (E_ij - v_i), with W non-negative and the sign living in the
driving force. The true forms below are built from ode_params, which knows which
it is, so the same readout serves both.
"""

import os

import numpy as np
import torch

from connectome_gnn.utils import to_numpy

# Panels b, c and d are teacher forced: the model is given the true voltages and
# asked for one step. Panel a is the free run. Keeping them on the same frames
# is what makes the figure readable, and mixing splits silently is what made an
# earlier version of it wrong.
_DT_FALLBACK = 0.02


# ------------------------------------------------------------------ #
#  What the generator actually computes, for this neuron
# ------------------------------------------------------------------ #

def _is_conductance(ode_params) -> bool:
    return getattr(ode_params, "E_exc", None) is not None


def true_forms(ode_params, neuron, edge_ids, n_neurons):
    """The generator's own equations for one neuron, as strings and as arrays.

    Returns a dict with `update` (the dv/dt formula with tau and V_rest
    substituted), `edges` (one formula per incoming synapse, with that synapse's
    conductance and, on conductance data, its reversal), and the constants the
    caller needs to evaluate them.
    """
    tau = float(np.asarray(ode_params.gt_tau(n_neurons), dtype=float)[neuron])
    vrest = float(np.asarray(ode_params.gt_vrest(n_neurons), dtype=float)[neuron])
    W = to_numpy(ode_params.W).ravel()
    cond = _is_conductance(ode_params)
    E = to_numpy(ode_params.reversal_per_edge()).ravel() if cond else None
    is_inh = (to_numpy(ode_params.edge_is_inh).ravel().astype(bool)
              if getattr(ode_params, "edge_is_inh", None) is not None else None)

    update = (f"dv/dt = ({vrest:+.4f} - v_i + msg + I) * {1.0 / tau:.4f}"
              f"        [ (V_rest - v + msg + I) / tau,  tau = {tau:.4f} ]")
    edges = {}
    for idx in edge_ids:
        if cond:
            edges[int(idx)] = (f"{W[idx]:.4f} * relu(v_j) * ({E[idx]:+.3f} - v_i)")
        else:
            edges[int(idx)] = f"{W[idx]:+.4f} * relu(v_j)"
    return {"update": update, "edges": edges, "tau": tau, "vrest": vrest,
            "W": W, "E": E, "is_inh": is_inh, "conductance": cond}


def _edge_message_true(ode_params, idx, v_j, v_i, forms):
    """The generator's message on one edge over the sampled frames."""
    act = np.asarray(ode_params.gt_g_phi_func(v_j), dtype=float).ravel()
    m = forms["W"][idx] * act
    if forms["conductance"]:
        m = m * (forms["E"][idx] - v_i)
    return m


# ------------------------------------------------------------------ #
#  What the model computes
# ------------------------------------------------------------------ #

def gather(model, data, neuron, start, n_frames, device="cpu"):
    """Teacher-forced quantities for one neuron over consecutive frames.

    Returns the neuron's voltage, stimulus, the model's aggregated message and
    its update, plus the per-edge true and model messages for every incoming
    synapse. The model is fed the TRUE voltages here; panel a's free run is read
    from the rollout bundle instead.
    """
    from connectome_gnn.models.utils import pad_g_phi_input
    from connectome_gnn.metrics import get_model_W

    core = getattr(model, "_orig_mod", model)
    op = data.ode_params
    edges = data.edges.to(device)
    e = to_numpy(edges).reshape(2, -1)
    src, dst = e[0], e[1]
    n = int(data.n_neurons)
    n_total = int(data.x_ts.n_frames)
    start = max(0, min(start, max(0, n_total - n_frames)))
    frames = np.arange(start, min(start + n_frames, n_total))

    did = torch.zeros((n, 1), dtype=torch.int, device=device)
    V, S, MSG, PRED = [], [], [], []
    with torch.no_grad():
        for k in frames:
            st = data.x_ts.frame(int(k)).to(device)
            pred, feats, msg = model(st, edges, data_id=did, return_all=True)
            V.append(to_numpy(feats[:, 0]).ravel()[:n])
            S.append(float(to_numpy(feats[:, 2 + int(core.a.shape[1])]).ravel()[neuron]))
            MSG.append(float(to_numpy(msg).ravel()[neuron]))
            PRED.append(float(to_numpy(pred).ravel()[neuron]))
    V = np.stack(V)
    v_i = V[:, neuron].astype(float)

    inc = np.where(dst == neuron)[0]
    forms = true_forms(op, neuron, inc, n)
    # Strongest conductances first, and capped: the fits are one per synapse.
    inc = inc[np.argsort(-np.abs(forms["W"][inc]))]

    emb = core.a.detach()
    W_model = to_numpy(get_model_W(core)).ravel()
    squared = bool(getattr(core, "w_squared", False))
    # Matches NeuralGNN._compute_messages: only the conductance family appends
    # the postsynaptic pair to g_phi's input.
    wide_g_phi = (getattr(core, "model", "") == "flyvis_conductance")
    m_true, m_model, v_js = [], [], []
    with torch.no_grad():
        for idx in inc:
            j = int(src[idx])
            v_j = V[:, j].astype(float)
            v_js.append(v_j)
            m_true.append(_edge_message_true(op, idx, v_j, v_i, forms))
            # THE MODEL'S OWN LAYOUT, not a fixed one. A conductance g_phi reads
            # (v_j, a_j, v_i, a_i) because its message needs the postsynaptic
            # voltage for the driving force; a current g_phi reads (v_j, a_j)
            # only, since W * act(v_j) does not. Building the wide row for both
            # is how the current family failed with "6 columns but the MLP
            # takes 3".
            cols = [torch.as_tensor(v_j, dtype=torch.float32, device=device)[:, None],
                    emb[j].expand(len(v_j), -1)]
            if wide_g_phi:
                cols += [torch.as_tensor(v_i, dtype=torch.float32, device=device)[:, None],
                         emb[neuron].expand(len(v_j), -1)]
            feat = torch.cat(cols, dim=1)
            g = core.g_phi(pad_g_phi_input(feat, core)).ravel()
            if getattr(core, "g_phi_positive", False):
                g = g ** 2
            w = W_model[idx] ** 2 if squared else W_model[idx]
            m_model.append(to_numpy(g).astype(float) * float(w))

    return {"frames": frames, "v_i": v_i, "stim": np.array(S), "msg_model": np.array(MSG),
            "pred": np.array(PRED), "m_true": np.stack(m_true) if len(inc) else np.zeros((0, len(v_i))),
            "m_model": np.stack(m_model) if len(inc) else np.zeros((0, len(v_i))),
            "edge_ids": inc, "src": src[inc], "forms": forms, "W_model": W_model[inc],
            "v_j": np.stack(v_js) if len(inc) else np.zeros((0, len(v_i)))}


# ------------------------------------------------------------------ #
#  Symbolic regression
# ------------------------------------------------------------------ #

def _sr_fit(X, y, names, cfg, guess=None):
    """One PySR fit. Returns the expression string, or a reason it is missing."""
    try:
        from pysr import PySRRegressor
    except Exception as exc:           # no Julia runtime, no PySR install
        return None, f"PySR unavailable ({type(exc).__name__})"
    if not np.isfinite(y).all() or float(np.std(y)) < 1e-12:
        return None, "target is constant"
    kw = dict(niterations=int(cfg.sr_niterations),
              operators={2: list(cfg.sr_binary_operators), 1: list(cfg.sr_unary_operators)},
              maxsize=int(cfg.sr_maxsize), progress=False, temp_equation_file=True,
              verbosity=0)
    if guess:
        kw["guesses"] = guess
    try:
        m = PySRRegressor(**kw)
        m.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64),
              variable_names=list(names))
        return str(m.get_best()["equation"]), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def symbolic_forms(g, cfg):
    """Fit the update and every incoming synapse of one neuron.

    The update is fitted in (v_i, msg, stim) against the model's own output, and
    each synapse in (v_j, v_i) against that synapse's message. The generator's
    formula is offered as a starting guess for the update, which costs nothing if
    the model did not learn it and saves search if it did.
    """
    out = {"update": None, "update_note": "disabled", "edges": {}, "edge_notes": {}}
    if not cfg.sr_enabled:
        return out
    f = g["forms"]
    guess = [f"({f['vrest']:.4f} - v_i + msg + stim) * {1.0 / f['tau']:.4f}"]
    eq, note = _sr_fit(np.column_stack([g["v_i"], g["msg_model"], g["stim"]]),
                       g["pred"], ["v_i", "msg", "stim"], cfg, guess=guess)
    out["update"], out["update_note"] = eq, note

    for row, idx in enumerate(g["edge_ids"][: int(cfg.sr_max_edges)]):
        eq, note = _sr_fit(np.column_stack([g["v_j"][row], g["v_i"]]),
                           g["m_model"][row], ["v_j", "v_i"], cfg)
        out["edges"][int(idx)] = eq
        out["edge_notes"][int(idx)] = note
    return out


# ------------------------------------------------------------------ #
#  The figure
# ------------------------------------------------------------------ #

def plot_neuron_panels(g, sr, neuron, log_dir, rollout=None, dt=_DT_FALLBACK,
                       label=""):
    """Write results/neuron<id>_panels.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = g["forms"]
    v_i, stim = g["v_i"], g["stim"]
    msg_true = g["m_true"].sum(0) if g["m_true"].size else np.zeros_like(v_i)
    dvdt_true = (f["vrest"] - v_i + msg_true + stim) / f["tau"]
    t = np.arange(len(v_i)) * dt

    def r2(y, p):
        y, p = np.asarray(y, float), np.asarray(p, float)
        ss = ((y - y.mean()) ** 2).sum()
        return 1 - ((y - p) ** 2).sum() / ss if ss > 0 else np.nan

    def pear(y, p):
        y, p = np.asarray(y, float), np.asarray(p, float)
        a, b = y - y.mean(), p - p.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else np.nan

    n_edges = g["m_true"].shape[0]
    fig = plt.figure(figsize=(22, max(11, 3.0 + 1.1 * n_edges)))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.35, 1],
                          height_ratios=[1, 1, 1, max(2.5, 0.5 * n_edges)],
                          hspace=0.45, wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    if rollout is not None:
        ax.plot(t, rollout[0], color="black", lw=0.9, label="generator")
        ax.plot(t, rollout[1], color="tab:green", lw=0.9, label="model, free running")
        ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
        head = f"a   voltage, free-running rollout   r = {pear(*rollout):.4f}"
    else:
        ax.plot(t, v_i, color="black", lw=0.9)
        head = "a   voltage (no rollout bundle; -o test writes one)"
    ax.text(0.004, 1.03, head, transform=ax.transAxes, va="bottom", fontsize=11)
    ax.set_ylabel("voltage")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, dvdt_true, color="black", lw=0.9)
    ax.plot(t, g["pred"], color="tab:green", lw=0.9)
    ax.text(0.004, 1.03, f"b   dv/dt at the true voltages   R2 = {r2(dvdt_true, g['pred']):+.3f}",
            transform=ax.transAxes, va="bottom", fontsize=11)
    ax.set_ylabel("dv/dt")

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, msg_true, color="black", lw=0.9)
    ax.plot(t, g["msg_model"], color="tab:green", lw=0.9)
    ratio = g["msg_model"].std() / max(msg_true.std(), 1e-12)
    ax.text(0.004, 1.03, f"c   total incoming message   r = {pear(msg_true, g['msg_model']):+.3f}"
                         f"   model {ratio:.1f}x",
            transform=ax.transAxes, va="bottom", fontsize=11)
    ax.set_ylabel("message")

    ax = fig.add_subplot(gs[3, 0])
    step = 5.0
    for row in range(n_edges):
        off = -row * step
        a, b = g["m_true"][row], g["m_model"][row]
        ax.plot(t, (a - a.mean()) / (a.std() + 1e-12) + off, color="black", lw=0.8)
        rr = b.std() / max(a.std(), 1e-12)
        if rr < 1e-3:
            ax.plot(t, np.zeros_like(t) + off, color="tab:green", lw=0.8)
            note = "model ~ 0"
        else:
            ax.plot(t, (b - b.mean()) / (b.std() + 1e-12) + off, color="tab:green", lw=0.8)
            note = f"r={pear(a, b):+.2f}  x{rr:.3g}"
        sign = ("inh" if f["is_inh"] is not None and f["is_inh"][g["edge_ids"][row]]
                else "exc")
        ax.text(-0.05, off, f"j={int(g['src'][row])}\n{sign}",
                transform=ax.get_yaxis_transform(), va="center", ha="right", fontsize=8)
        ax.text(1.004, off, note, transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8)
    ax.set_ylim(-step * max(n_edges, 1), step)
    ax.set_yticks([])
    ax.text(0.004, 1.005, f"d   the {n_edges} synapses onto neuron {neuron}, z-scored, "
            "strongest first", transform=ax.transAxes, va="bottom", fontsize=11)

    for a_ in fig.axes:
        a_.set_xlim(t[0], t[-1])
        a_.set_xlabel("time (s)")
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)

    # ---- e: the equations ----
    axe = fig.add_subplot(gs[:, 1])
    axe.axis("off")
    lines = [f"e   closed forms, neuron {neuron}{('   ' + label) if label else ''}", ""]
    lines += ["THE UPDATE", "  generator", f"    {f['update']}", "  recovered"]
    lines += [f"    {sr['update']}" if sr.get("update")
              else f"    [{sr.get('update_note') or 'not fitted'}]", ""]
    lines += [f"THE {n_edges} SYNAPSES   "
              + ("W * relu(v_j) * (E - v_i)" if f["conductance"] else "W * relu(v_j)"), ""]
    for row in range(n_edges):
        idx = int(g["edge_ids"][row])
        j = int(g["src"][row])
        sign = "inh" if f["is_inh"] is not None and f["is_inh"][idx] else "exc"
        lines.append(f"  j = {j}  ({sign})")
        lines.append(f"    generator  {f['edges'][idx]}")
        eq = sr["edges"].get(idx)
        lines.append(f"    recovered  {eq}" if eq
                     else f"    recovered  [{sr['edge_notes'].get(idx) or 'not fitted'}]")
    axe.text(0.0, 1.0, "\n".join(lines), transform=axe.transAxes, va="top", ha="left",
             fontsize=7.5, family="monospace")

    out_dir = os.path.join(log_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"neuron{neuron}_panels.png")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.965, bottom=0.035)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #

def analyse_neurons(config, model, data, log_dir, device="cpu", logger=None):
    """Write one panel figure per neuron named in config.analysis.

    Returns the list of paths written. Never raises: a readout that fails must
    not take down the plotting pass that produced everything else.
    """
    cfg = getattr(config, "analysis", None)
    if cfg is None or not cfg.neurons:
        return []
    n = int(data.n_neurons)
    dt = float(getattr(config.simulation, "delta_t", _DT_FALLBACK))
    bundle = _load_rollout(log_dir)
    written = []
    for neuron in cfg.neurons:
        if not (0 <= int(neuron) < n):
            _say(logger, f"neuron {neuron} out of range (0..{n - 1}), skipped")
            continue
        try:
            g = gather(model, data, int(neuron), cfg.sr_start_frame, cfg.sr_frames, device)
            sr = symbolic_forms(g, cfg)
            roll = _rollout_slice(bundle, int(neuron), g["frames"])
            path = plot_neuron_panels(g, sr, int(neuron), log_dir, rollout=roll, dt=dt,
                                      label=os.path.basename(log_dir.rstrip("/")))
            written.append(path)
            _say(logger, f"neuron {neuron}: panels -> {path}")
        except Exception as exc:
            _say(logger, f"neuron {neuron}: readout failed: {type(exc).__name__}: {exc}")
    return written


def _load_rollout(log_dir):
    p = os.path.join(log_dir, "results", "rollout_bundle.npz")
    if not os.path.isfile(p):
        return None
    try:
        return np.load(p, allow_pickle=True)
    except Exception:
        return None


def _rollout_slice(bundle, neuron, frames):
    """The free run over the same frames, when -o test left a bundle behind.

    The bundle indexes the TEST split while the panels are computed on whatever
    split init_training_data loaded, so this is only used when the two line up in
    length; otherwise the voltage panel falls back to the observed trace.
    """
    if bundle is None:
        return None
    try:
        true, pred = bundle["activity_true"], bundle["activity_pred"]
    except Exception:
        return None
    k0, k1 = int(frames[0]), int(frames[-1]) + 1
    if k1 > true.shape[1]:
        return None
    return true[neuron, k0:k1].astype(float), pred[neuron, k0:k1].astype(float)


def _say(logger, msg):
    if logger is not None:
        logger.info(msg)
    print(msg)
