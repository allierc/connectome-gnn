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

def gather(model, data, neuron, start, n_frames, device="cpu", x_ts=None):
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
    # The caller may hand in the TEST split so that these panels and the free
    # run in panel a describe the same frames of the same trajectory. Showing
    # one split's rollout above another split's messages, on axes that both
    # start at zero, is the mistake this argument exists to prevent.
    x_ts = data.x_ts if x_ts is None else x_ts
    n_total = int(x_ts.n_frames)
    start = max(0, min(start, max(0, n_total - n_frames)))
    frames = np.arange(start, min(start + n_frames, n_total))

    did = torch.zeros((n, 1), dtype=torch.int, device=device)
    V, S, MSG, PRED = [], [], [], []
    with torch.no_grad():
        for k in frames:
            st = x_ts.frame(int(k)).to(device)
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

def _r2(y, p):
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - p) ** 2).sum()) / ss if ss > 0 else float("nan")


def _sr_fit(X, y, names, cfg, guess=None, spec=None):
    """One PySR fit. Returns (expression, R2, note, prediction).

    The R2 is the point: an expression that used the whole complexity budget and
    still explains little is a different statement from one that fits perfectly,
    and without it a long formula cannot be told from a good one. A constant
    target reports ITS VALUE rather than the bare word constant, because on a
    pruned synapse that value is the finding.
    """
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        return None, float("nan"), "target has non-finite values", None
    if float(np.std(y)) < 1e-12:
        return None, float("nan"), f"target constant at {float(np.mean(y)):.4g}", None
    try:
        from pysr import PySRRegressor
    except Exception as exc:           # no Julia runtime, no PySR install
        return None, float("nan"), f"PySR unavailable ({type(exc).__name__})", None
    kw = dict(niterations=int(cfg.sr_niterations),
              operators={2: list(cfg.sr_binary_operators), 1: list(cfg.sr_unary_operators)},
              maxsize=int(cfg.sr_maxsize), progress=False, temp_equation_file=True,
              verbosity=0)
    if guess:
        kw["guesses"] = guess
    if spec is not None:
        kw["expression_spec"] = spec
    try:
        m = PySRRegressor(**kw)
        X = np.asarray(X, dtype=np.float64)
        m.fit(X, y, variable_names=list(names))
        pred = np.asarray(m.predict(X)).ravel()
        return str(m.get_best()["equation"]), _r2(y, pred), None, pred
    except Exception as exc:
        return None, float("nan"), f"{type(exc).__name__}: {exc}", None


def _effective_W(pred, v_j, v_i, E, C=0.0):
    """The conductance implied by a fitted conductance-family message.

    The generator's synapse is W * relu(v_j) * (E - v_i); given the fitted E and
    the fitted message, W is the single least-squares scale that maps the driving
    term onto it, W = <m, d> / <d, d> with d = relu(v_j) * (E - v_i). Reported in
    the MODEL's gauge: it still carries the global message gain, which the panel
    divides out with the same T * G * tau it uses for the total message.

    THE OFFSET COMES OFF FIRST. The template now fits a pedestal C alongside, and
    a message sitting on one would otherwise be projected onto the driving term
    as though the pedestal were conductance -- inflating W by however much of C
    happens to correlate with relu(v_j) * (E - v_i) over these frames.
    """
    if pred is None or E is None:
        return None
    d = np.maximum(np.asarray(v_j, float), 0.0) * (float(E) - np.asarray(v_i, float))
    den = float(d @ d)
    m = np.asarray(pred, float) - float(C or 0.0)
    return float(m @ d) / den if den > 0 else None


def _conductance_template(cfg):
    """f(v_j) * (E[cat] - v_i) + C[cat]: the conductance family, with its reversal
    AND its silent-input offset as fitted constants.

    C IS NOT A NUISANCE PARAMETER, it is the measurement. The generator's
    per-edge message is W * relu(v_j) * (E - v_i), identically zero whenever the
    presynaptic cell is silent, for every edge and whatever the postsynaptic
    state. The model's g_phi is under no such obligation, and what it emits at
    silent input is exactly C -- the per-edge form of the offset that the total
    message carries (msg_model = msg_true / k + beta), which the update hands
    straight to V_rest, and which coeff_g_phi_silent exists to drive to zero.
    Fitting it explicitly also unbiases W and E: without a constant in the form,
    a message sitting on a pedestal is fitted by tilting the driving force
    instead, which moves the reversal.

    The free search answers "what did the model learn"; this answers "what is
    the closest conductance to what the model learned, and with which reversal".
    Both are kept, because only the free one can show that the model left the
    family altogether, and only this one gives a number to put beside the
    generator's. `cat` is a column of ones: PySR 2 indexes per-category
    constants by a data column, and here there is a single category per fit.
    """
    try:
        from pysr import TemplateExpressionSpec
    except Exception:
        return None
    return TemplateExpressionSpec(combine="f(v_j) * (E[cat] - v_i) + C[cat]",
                                  expressions=["f"], parameters={"E": 1, "C": 1},
                                  variable_names=["v_j", "v_i", "cat"])


def _update_template():
    """T * (V - v_i + G * msg + f(stim)): the generator's update with its three
    constants left free.

    The generator computes dv/dt = (V_rest - v_i + msg + I) / tau, so the truth is
    T = 1/tau, V = V_rest and G = 1 exactly. G is the number this readout exists
    for: it is the coefficient the model applies to its own message, and the whole
    degeneracy lives there. A free search reports it buried inside an expression
    like ((3.82 - msg) + (v_i * -11.46)) * 1.65, where the message coefficient is
    the product of two printed numbers; here it is one fitted constant with a
    known target. The stimulus is left as a sub-expression rather than a fourth
    constant because PySR requires at least one, and because f = stim is itself
    worth seeing: the stimulus is an observed input, so its coefficient is the one
    the data pin down.
    """
    try:
        from pysr import TemplateExpressionSpec
    except Exception:
        return None
    return TemplateExpressionSpec(
        combine="T[cat] * ((V[cat] - v_i) + (G[cat] * msg) + f(stim))",
        expressions=["f"], parameters={"T": 1, "V": 1, "G": 1},
        variable_names=["v_i", "msg", "stim", "cat"])


def _strip_parameter_lists(equation):
    """The sub-expression of a template equation, without its parameter lists.

    PySR prints `f = #1 * 1.4467; E = [10.940051]`, and the panels already report
    E as a named number beside the generator's own, so the bracketed list would
    be the same constant a second time.
    """
    import re
    if not equation:
        return equation
    txt = re.sub(r";?\s*\w+\s*=\s*\[[^\]]*\]", "", str(equation)).strip()
    return txt.strip(";").strip() or str(equation)


def _fitted_parameters(equation):
    """The per-category constants out of a template equation, e.g. `E = [-5.17]`."""
    import re
    out = {}
    for name, body in re.findall(r"(\w+)\s*=\s*\[([^\]]*)\]", str(equation)):
        try:
            out[name] = [float(x) for x in body.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            pass
    return out


def symbolic_forms(g, cfg):
    """Fit the update, and every incoming synapse twice: free, and templated.

    The update is fitted in (v_i, msg, stim) against the model's own output, with
    the generator's formula offered as a starting guess. Each synapse is fitted in
    (v_j, v_i) against that synapse's message, once with no constraint and once
    inside the conductance family, so the recovered reversal can be read as a
    number beside the one the generator used.
    """
    out = {"update": None, "update_r2": float("nan"), "update_note": "disabled",
           "update_tmpl": None, "update_tmpl_r2": float("nan"),
           "update_tmpl_note": "disabled", "update_tmpl_p": {},
           "edges": {}, "edge_r2": {}, "edge_notes": {},
           "tmpl": {}, "tmpl_r2": {}, "tmpl_E": {}, "tmpl_C": {}, "tmpl_W": {},
           "tmpl_notes": {}}
    if not cfg.sr_enabled:
        return out
    f = g["forms"]
    guess = [f"({f['vrest']:.4f} - v_i + msg + stim) * {1.0 / f['tau']:.4f}"]
    eq, r2v, note, _ = _sr_fit(np.column_stack([g["v_i"], g["msg_model"], g["stim"]]),
                               g["pred"], ["v_i", "msg", "stim"], cfg, guess=guess)
    out["update"], out["update_r2"], out["update_note"] = eq, r2v, note

    uspec = _update_template()
    if uspec is not None:
        ones = np.ones_like(g["v_i"])
        ueq, ur2, unote, _ = _sr_fit(
            np.column_stack([g["v_i"], g["msg_model"], g["stim"], ones]),
            g["pred"], ["v_i", "msg", "stim", "cat"], cfg, spec=uspec)
        out["update_tmpl"], out["update_tmpl_r2"], out["update_tmpl_note"] = ueq, ur2, unote
        p = _fitted_parameters(ueq) if ueq else {}
        out["update_tmpl_p"] = {k: (v[0] if v else None) for k, v in p.items()}

    spec = _conductance_template(cfg) if f["conductance"] else None
    ones = np.ones_like(g["v_i"])
    for row, idx in enumerate(g["edge_ids"][: int(cfg.sr_max_edges)]):
        idx = int(idx)
        eq, r2v, note, _ = _sr_fit(np.column_stack([g["v_j"][row], g["v_i"]]),
                                   g["m_model"][row], ["v_j", "v_i"], cfg)
        out["edges"][idx], out["edge_r2"][idx], out["edge_notes"][idx] = eq, r2v, note
        if spec is not None:
            teq, tr2, tnote, tpred = _sr_fit(
                np.column_stack([g["v_j"][row], g["v_i"], ones]),
                g["m_model"][row], ["v_j", "v_i", "cat"], cfg, spec=spec)
            out["tmpl"][idx], out["tmpl_r2"][idx], out["tmpl_notes"][idx] = teq, tr2, tnote
            _p = _fitted_parameters(teq) if teq else {}
            E = _p.get("E")
            E = E[0] if E else None
            C = _p.get("C")
            C = C[0] if C else None
            out["tmpl_E"][idx] = E
            out["tmpl_C"][idx] = C
            # W IS NOT PRINTED BY THE TEMPLATE: it lives inside the sub-expression
            # f, which PySR writes as `f = #1 * 1.4467`. Rather than parse a form
            # that is only sometimes linear, the conductance is measured from the
            # fit itself -- the one scale that carries relu(v_j) * (E - v_i) onto
            # the fitted message. That works whatever shape f took.
            out["tmpl_W"][idx] = _effective_W(tpred, g["v_j"][row], g["v_i"], E, C)
    return out


# ------------------------------------------------------------------ #
#  The figure
# ------------------------------------------------------------------ #

def plot_neuron_panels(g, sr, neuron, log_dir, rollout=None, dt=_DT_FALLBACK,
                       label=""):
    """Write results/neuron<id>_panels.png.

    Panels a to d share one time axis; e carries the update's forms and f carries
    the synapses', ROW BY ROW BESIDE PANEL d so each formula sits next to the
    trace it describes rather than in a list the reader has to re-index.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fm = g["forms"]
    v_i, stim = g["v_i"], g["stim"]
    msg_true = g["m_true"].sum(0) if g["m_true"].size else np.zeros_like(v_i)
    dvdt_true = (fm["vrest"] - v_i + msg_true + stim) / fm["tau"]
    t = np.arange(len(v_i)) * dt

    def pear(y, p):
        y, p = np.asarray(y, float), np.asarray(p, float)
        a, b = y - y.mean(), p - p.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else np.nan

    def fmt_r2(x):
        return "" if x is None or x != x else f"  (R2 {x:+.3f})"

    n_edges = g["m_true"].shape[0]
    step = 8.0                      # room for three lines of formula per synapse
    fig = plt.figure(figsize=(23, max(13, 5.0 + 1.35 * n_edges)))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.15, 1],
                          height_ratios=[1, 1, 1, max(3.0, 0.62 * n_edges)],
                          hspace=0.45, wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    if rollout is not None:
        ax.plot(t, rollout[0], color="tab:green", lw=0.9, label="generator")
        ax.plot(t, rollout[1], color="black", lw=0.9, label="model, free running")
        ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
        head = f"a   voltage, free-running rollout   r = {pear(*rollout):.4f}"
    else:
        ax.plot(t, v_i, color="tab:green", lw=0.9)
        head = "a   voltage (no aligned rollout available)"
    ax.text(0.004, 1.03, head, transform=ax.transAxes, va="bottom", fontsize=11)
    ax.set_ylabel("voltage")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t, dvdt_true, color="tab:green", lw=0.9)
    ax.plot(t, g["pred"], color="black", lw=0.9)
    ax.text(0.004, 1.03, f"b   dv/dt at the true voltages   R2 = {_r2(dvdt_true, g['pred']):+.3f}",
            transform=ax.transAxes, va="bottom", fontsize=11)
    ax.set_ylabel("dv/dt")

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(t, msg_true, color="tab:green", lw=0.9)
    ax.plot(t, g["msg_model"], color="black", lw=0.9)
    ratio = g["msg_model"].std() / max(msg_true.std(), 1e-12)
    head_c = (f"c   total incoming message   r = {pear(msg_true, g['msg_model']):+.3f}"
              f"   model {ratio:.1f}x")
    # THE GAUGE BETWEEN THE TWO MESSAGES IS AFFINE, NOT A PURE SCALE. Writing the
    # model's message as msg_model = msg_true / k + beta and matching the model's
    # T * [(V - v_i) + G * msg_model] to the generator's
    # (V_rest - v_i + msg_true) / tau leaves k = T * G * tau on the message
    # coefficient and, on the constants,
    #
    #     T * V + T * G * beta = V_rest / tau    i.e.   V_rest = tau * T * V + k * beta
    #
    # so beta is a level the model carries in its message and the update hands
    # straight to V_rest. The template fit in e CANNOT see it -- a constant is
    # absorbed into V by construction, which is why its R2 says nothing about it.
    # Scaling by k alone therefore matches the amplitude and leaves the level off
    # by k * beta, so the correction drawn here is the affine one: least squares
    # of msg_true on msg_model over these frames, msg_true ~ a * msg_model + b.
    # Inverting the gauge above, a = k and b = -k * beta, so the identity to check
    # is V_rest = tau * T * V - b. b is in volts, is the offset V_rest absorbs,
    # and is what coeff_g_phi_silent drives to zero.
    a_fit, b_fit = np.polyfit(g["msg_model"], msg_true, 1)
    corrected = a_fit * g["msg_model"] + b_fit
    ax.plot(t, corrected, color="black", lw=0.9, ls="--")
    head_c += (f"   |   dashed: model x {a_fit:.4f} {b_fit:+.3f} V, residual "
               f"{(msg_true - corrected).std() / max(msg_true.std(), 1e-12):.2f}x")
    p = sr.get("update_tmpl_p") or {}
    T, G, V = p.get("T"), p.get("G"), p.get("V")
    if T is not None and G is not None and V is not None:
        # The same b read back through the update: the template's own V accounts
        # for tau * T * V of the generator's V_rest, and -b is meant to be the rest.
        k = float(T) * float(G) * fm["tau"]
        tTV = fm["tau"] * float(T) * float(V)
        head_c += (f"   |   a vs T*G*tau = {k:.4f}   V_rest: tau*T*V - b = "
                   f"{tTV:+.3f} {-b_fit:+.3f} = {tTV - b_fit:+.3f} vs {fm['vrest']:+.3f}")
    # THE TOTAL OFFSET AGAINST THE SUM OF THE PER-SYNAPSE ONES. b is what the
    # whole message carries as a level; panel f fits one offset per synapse, and
    # the message is their sum, so sum(C_e) * k must come back as -b if the two
    # readouts are describing the same thing. It is the one cross-check the
    # figure can make between its own panels, and a gap means the per-edge fits
    # are putting the level somewhere the total does not see -- most likely into
    # the synapses that were capped out of the fit.
    _C = [v for v in (sr.get("tmpl_C") or {}).values() if v is not None]
    if _C and (sr.get("update_tmpl_p") or {}).get("T") is not None:
        _p = sr["update_tmpl_p"]
        _k = float(_p["T"]) * float(_p["G"]) * fm["tau"]
        head_c += (f"   |   offset: {-b_fit:+.3f} total vs {_k * float(np.sum(_C)):+.3f} "
                   f"from {len(_C)} synapse fits")
    ax.text(0.004, 1.03, head_c, transform=ax.transAxes, va="bottom", fontsize=11)
    ax.set_ylabel("message")

    axd = fig.add_subplot(gs[3, 0])
    offs = []
    for row in range(n_edges):
        off = -row * step
        offs.append(off)
        a, b = g["m_true"][row], g["m_model"][row]
        axd.plot(t, 1.6 * (a - a.mean()) / (a.std() + 1e-12) + off, color="tab:green", lw=0.8)
        rr = b.std() / max(a.std(), 1e-12)
        if rr < 1e-3:
            axd.plot(t, np.zeros_like(t) + off, color="black", lw=0.8)
            note = "model ~ 0"
        else:
            axd.plot(t, 1.6 * (b - b.mean()) / (b.std() + 1e-12) + off,
                     color="black", lw=0.8)
            note = f"r={pear(a, b):+.2f}  x{rr:.3g}"
        sign = ("inh" if fm["is_inh"] is not None and fm["is_inh"][g["edge_ids"][row]]
                else "exc")
        axd.text(-0.055, off, f"j={int(g['src'][row])}\n{sign}\n{note}",
                 transform=axd.get_yaxis_transform(), va="center", ha="right", fontsize=7.5)
    axd.set_ylim(-step * max(n_edges, 1) + step * 0.35, step * 0.65)
    axd.set_yticks([])
    axd.text(0.004, 1.005, f"d   the {n_edges} synapses onto neuron {neuron}, z-scored, "
             "strongest first", transform=axd.transAxes, va="bottom", fontsize=11)

    for a_ in (fig.axes[0], fig.axes[1], fig.axes[2], axd):
        a_.set_xlim(t[0], t[-1])
        a_.set_xlabel("time (s)")
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)

    # ---- e: the update ----
    axe = fig.add_subplot(gs[0:3, 1])
    axe.axis("off")
    lines = [f"e   the update, neuron {neuron}{('   ' + label) if label else ''}", ""]
    p = sr.get("update_tmpl_p") or {}
    lines += ["  generator", f"    {fm['update']}", ""]
    lines += [f"  template   T * ((V - v_i) + G * msg + f(stim))"
              f"{fmt_r2(sr.get('update_tmpl_r2'))}"]
    if sr.get("update_tmpl"):
        def _cmp(name, got, want):
            return (f"    {name} = {got:+.4f}   (generator {want:+.4f})"
                    if got is not None else f"    {name} not parsed")
        lines += [_cmp("T  (1/tau)   ", p.get("T"), 1.0 / fm["tau"]),
                  _cmp("V  (V_rest)  ", p.get("V"), fm["vrest"]),
                  _cmp("G  (msg gain)", p.get("G"), 1.0),
                  f"    {_strip_parameter_lists(sr['update_tmpl'])}"]
    else:
        lines += [f"    [{sr.get('update_tmpl_note') or 'not fitted'}]"]
    lines += ["", f"  free{fmt_r2(sr.get('update_r2'))}"]
    lines += [f"    {sr['update']}" if sr.get("update")
              else f"    [{sr.get('update_note') or 'not fitted'}]"]
    lines += ["", f"  the message enters the generator's update with coefficient "
                  f"{1.0 / fm['tau']:.4f} = 1/tau, so G = 1 is the target"]
    axe.text(0.0, 1.0, "\n".join(lines), transform=axe.transAxes, va="top", ha="left",
             fontsize=8, family="monospace")

    # ---- f: the synapses, aligned with d ----
    axf = fig.add_subplot(gs[3, 1], sharey=axd)
    axf.axis("off")
    fam = ("W * relu(v_j) * (E - v_i) + offset" if fm["conductance"]
           else "W * relu(v_j)")
    # The model's message carries the global gain that f_theta divides back out,
    # so its conductance is only comparable with the generator's after the same
    # T * G * tau correction the total message gets in panel c.
    kW = None
    if fm["conductance"]:
        pu = sr.get("update_tmpl_p") or {}
        if pu.get("T") is not None and pu.get("G") is not None:
            kW = float(pu["T"]) * float(pu["G"]) * fm["tau"]

    _gain_note = (f", W and offset scaled by T*G*tau = {kW:.4f}"
                  if fm["conductance"] and kW else ", W in the model's own gauge")
    axf.text(0.0, 1.005, f"f   the synapses: generator, the same fitted inside "
             f"{fam}{_gain_note}, and a free search", transform=axf.transAxes,
             va="bottom", fontsize=11)
    def fmt_W(w):
        """Four decimals, except where that would print a synapse as zero.

        The connectome's conductances run over eight orders of magnitude: an
        edge at 1.1e-08 and one at 3.6e-13 both print as 0.0000 on a fixed
        format, and beside panel d -- which z-scores every row, so a synapse
        carrying 1e-08 of a volt is drawn at the same height as one carrying 1.05
        -- that reads as "the generator says zero and its trace is not flat".
        Neither is zero; the format was.
        """
        return f"{w:8.4f}" if abs(w) >= 5e-5 or w == 0 else f"{w:8.2e}"

    for row in range(n_edges):
        idx = int(g["edge_ids"][row])
        if fm["conductance"]:
            # The generator's own offset is zero BY CONSTRUCTION, not by fit:
            # W * relu(v_j) * (E - v_i) vanishes whenever the sender is silent.
            # Printing the zero is the point -- it is what the fitted C is to be
            # read against.
            txt = [f"generator   W = {fmt_W(fm['W'][idx])}   E = {fm['E'][idx]:+8.3f}"
                   f"   offset =   0.0000"]
        else:
            txt = [f"generator   W = {fmt_W(fm['W'][idx])}"]
        if fm["conductance"]:
            teq = sr.get("tmpl", {}).get(idx)
            E = sr.get("tmpl_E", {}).get(idx)
            C = sr.get("tmpl_C", {}).get(idx)
            W = sr.get("tmpl_W", {}).get(idx)
            if teq:
                Wc = None if (W is None or kW is None) else W * kW
                Cc = None if (C is None or kW is None) else C * kW
                wtxt = ("W = " + (fmt_W(Wc) if Wc is not None else
                                  (fmt_W(W) if W is not None else "     n/a")))
                etxt = f"E = {E:+8.3f}" if E is not None else "E =      n/a"
                ctxt = ("offset = " + (fmt_W(Cc) if Cc is not None else "     n/a"))
                txt.append(f"template    {wtxt}   {etxt}   {ctxt}"
                           f"{fmt_r2(sr.get('tmpl_r2', {}).get(idx))}")
            else:
                txt.append(f"template    [{sr.get('tmpl_notes', {}).get(idx) or 'not fitted'}]")
        eq = sr["edges"].get(idx)
        txt.append(f"free        {eq}{fmt_r2(sr.get('edge_r2', {}).get(idx))}" if eq
                   else f"free        [{sr['edge_notes'].get(idx) or 'not fitted'}]")
        axf.text(0.0, offs[row], "\n".join(txt), transform=axf.get_yaxis_transform(),
                 va="center", ha="left", fontsize=7, family="monospace")

    out_dir = os.path.join(log_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"neuron{neuron}_panels.png")
    fig.subplots_adjust(left=0.055, right=0.995, top=0.965, bottom=0.035)
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
    # The rollout bundle is written by `-o test` from the TEST split, so the
    # panels are computed there too and every one of them shows the same frames.
    # If that split cannot be loaded the panels fall back to whatever was loaded
    # for training and the free run is dropped rather than shown misaligned.
    x_ts_panels, aligned = _test_split(config, data, bundle)
    if bundle is not None and not aligned:
        _say(logger, "test split unavailable; drawing panels without the free run")
        bundle = None
    written = []
    for neuron in cfg.neurons:
        if not (0 <= int(neuron) < n):
            _say(logger, f"neuron {neuron} out of range (0..{n - 1}), skipped")
            continue
        try:
            g = gather(model, data, int(neuron), cfg.sr_start_frame, cfg.sr_frames,
                       device, x_ts=x_ts_panels)
            sr = symbolic_forms(g, cfg)
            roll = _rollout_slice(bundle, int(neuron), g["frames"])
            path = plot_neuron_panels(g, sr, int(neuron), log_dir, rollout=roll, dt=dt,
                                      label=os.path.basename(log_dir.rstrip("/")))
            written.append(path)
            _say(logger, f"neuron {neuron}: panels -> {path}")
        except Exception as exc:
            _say(logger, f"neuron {neuron}: readout failed: {type(exc).__name__}: {exc}")
    return written


def _test_split(config, data, bundle):
    """The test-split time series, when it matches the rollout bundle's length.

    Returns (x_ts, aligned). `aligned` is False when the bundle cannot be
    indexed by the same frame numbers, in which case panel a is dropped instead
    of being drawn from a different trajectory than panels b to d.
    """
    if bundle is None:
        return data.x_ts, False
    try:
        from connectome_gnn.models.training_utils import (
            determine_load_fields, load_flyvis_data)
        x_ts, _, _ = load_flyvis_data(config.dataset, split="test",
                                      fields=determine_load_fields(config))
        n_bundle = int(bundle["activity_true"].shape[1])
        if abs(int(x_ts.n_frames) - n_bundle) <= 1:
            return x_ts, True
        return data.x_ts, False
    except Exception:
        return data.x_ts, False


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
