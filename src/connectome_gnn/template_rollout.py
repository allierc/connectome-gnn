"""Run the recovered parameters as a generator, on noise-free data.

WHAT THIS ADDS. The template readout proves the parameters fit a regression:
per edge, the model's own message against the generator's form; per neuron, its
update against the generator's. A high R2 there says the numbers are consistent
with what the trained network computes. It does not say they ARE the circuit --
a readout can be an excellent regression of a model that does not reproduce the
dynamics when you integrate it. Loading the recovered W, tau and V_rest into the
known-ODE and rolling IT out is the test that closes that gap, and the number it
returns is the one a paper can carry: not "the fit is good" but "the recovered
circuit runs".

NO NEW MODEL, NO SECOND TESTER. `FlyvisKnownODE` already computes exactly the
template's form -- dv/dt = (-v + msg + I + V_rest)/tau with msg = sum W*relu(v_j)
-- with W, raw_tau and V_rest as its parameters, so the "template-fit model" is
that class with parameters SET instead of trained. Written to a checkpoint,
`models/template_fit.pt`, it goes through `data_test_gnn` unchanged: the tester
builds the model from a config whose signal_model_name is the known-ODE and
loads the checkpoint the way it loads any other.

ALWAYS NOISE-FREE, AND NEVER OVER THE RUN'S OWN OUTPUTS. The test data is the
noise-free sibling of the run's dataset, handed in through the tester's existing
`test_config` argument -- which also makes every output it writes carry an
`_on_<dataset>_template` suffix, so `results_rollout*.log`, the bundle and the
figures of the GNN's own test are untouched.

THE ONE THING THAT WILL BITE is parameter conventions. `raw_tau` is pre-softplus,
so writing tau into it directly gives softplus(0.02) = 0.703, a 35x error in
every time constant, and the rollout would look like the readout failed. Every
field is therefore written through its inverse transform and READ BACK before
anything runs: `_verify_round_trip` compares the model's own accessors to the
arrays that went in, and raises rather than roll out a model that is not the fit.
"""

import os
import re

import numpy as np
import torch

from connectome_gnn.utils import to_numpy

# The known-ODE that computes each family's generator form. Keyed by the
# signal_model_name of the trained model the parameters came out of.
_KNOWN_ODE_FOR = {
    "flyvis_current": "flyvis_known_ode",
    "flyvis": "flyvis_known_ode",
    "flyvis_conductance": "flyvis_conductance_known_ode",
}

# Absolute tolerance on the read-back. Loose enough for float32 storage of
# values of order 10 -- V_rest came back 1.41e-05 off and failed a 1e-5 check,
# which is the storage, not the parameters -- and tight enough that a real
# mistake, an un-inverted softplus or a misaligned array, is orders away.
_ROUND_TRIP_TOL = 1e-3

# AND A SECOND, RELATIVE BAND BEFORE IT REFUSES TO ROLL OUT AT ALL. Six Block B
# runs produced no rollout of either form because V_rest read back 0.0265 away
# from what was written -- 0.3% of a resting potential of order 10, which is a
# parameter the known-ODE stores through a transform, not a misassignment. What
# the check exists to catch is an array written to the wrong edges, and that is
# orders of magnitude, not percent. Beyond the absolute tolerance the deviation
# is now RECORDED and the rollout proceeds; beyond this relative band it still
# raises, because at that point the rolled-out model is no longer the fit.
_ROUND_TRIP_REL_ABORT = 0.05


def known_ode_name(config):
    """The known-ODE that matches this run's model, or None if there is none."""
    return _KNOWN_ODE_FOR.get(getattr(config.graph_model, "signal_model_name", ""))


def other_known_ode_name(config):
    """The OTHER family's known-ODE: current for a conductance run, and back.

    What the alternative-form fit is rolled out in. The point of running it is
    that R2 cannot separate the families on this data -- the conductance form
    contains the current one -- while a rollout can: constants that reproduce
    the message but not the trajectory were describing the sample, not the
    dynamics.
    """
    own = known_ode_name(config)
    if own is None:
        return None
    return ("flyvis_known_ode" if own == "flyvis_conductance_known_ode"
            else "flyvis_conductance_known_ode")


def noise_free_dataset(dataset: str) -> str:
    """The noise-free sibling of a dataset name.

    `flyvis_noise_005_blank50_cv00` -> `flyvis_noise_free_blank50_cv00`, and the
    same for the 010/020 measurement-noise ladder and the conductance twins. A
    name that is already noise-free comes back unchanged.
    """
    if "noise_free" in dataset:
        return dataset
    # `noise_\d+(_\d+)*` in one go: the measurement-noise ladder names carry two
    # levels, `noise_005_020` being process 0.05 with measurement 0.20, and
    # replacing only the first leaves `noise_free_020`, a dataset that does not
    # exist. Noise-free here means no noise of either kind.
    return re.sub(r"noise_\d+(?:_\d+)*", "noise_free", dataset)


def _inverse_softplus(x):
    """The raw value whose softplus is x. log(expm1(x)), which stays accurate
    for the small taus this data has -- exp(0.02) - 1 computed naively loses
    most of its significant digits."""
    x = np.asarray(x, dtype=np.float64)
    return np.log(np.expm1(np.clip(x, 1e-12, None)))


def build_state_dict(rec, n_neurons, n_edges, model):
    """A known-ODE state_dict carrying the recovered parameters.

    Only the fields the readout recovers are written; everything else keeps the
    freshly built model's own value, which is why `model` is passed in rather
    than a bare dict. Returns (state_dict, what_was_written).
    """
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    written = {}

    # THE FULL PER-EDGE ARRAY, NOT THE PAIR. `rec.pairs["W"]` is compacted --
    # `_pair` drops every edge whose fit was not finite, so on this run it is
    # 304,108 long against 434,112 edges -- and writing it into W[:n_edges]
    # silently assigns each weight to the wrong edge. The message then comes out
    # near zero (std 0.005 against the generator's 0.59) while W's median
    # magnitude still looks right, which is the hardest kind of wrong to see.
    # `_W_learned_full` is the uncompacted array the extractor keeps for exactly
    # this.
    _w_full = rec.diagnostics.get("_W_learned_full")
    W = rec.pairs.get("W")
    if _w_full is not None and np.asarray(_w_full).size >= n_edges:
        learned = np.asarray(_w_full, dtype=np.float32).ravel()
    elif W is not None and np.asarray(W[1]).size >= n_edges:
        learned = np.asarray(W[1], dtype=np.float32).ravel()
    else:
        learned = None
    if learned is not None:
        if True:
            # The readout's W is already in the generator's units -- the per-edge
            # fit, scaled by the per-neuron gauge k_i -- so it goes in as is. The
            # known-ODE does not square W and does not carry the GNN's sign
            # convention; both live behind the readout.
            # AN UNFITTED EDGE CONTRIBUTES NOTHING, it does not contribute NaN.
            # The template fit leaves W non-finite wherever the per-edge fit had
            # too few usable frames -- 133,377 of 434,112 edges on this run, the
            # ones whose presynaptic cell is rarely above the v_j floor. Writing
            # those through makes every message NaN and the free rollout diverges
            # at frame 0, while the one-step score still reads 0.999 because the
            # pooling skips non-finite neurons. Zero is the neutral value: no
            # evidence for a synapse, so no current through it.
            _col = np.asarray(learned[:n_edges], dtype=np.float32)
            _unfitted = int((~np.isfinite(_col)).sum())
            _col = np.nan_to_num(_col, nan=0.0, posinf=0.0, neginf=0.0)
            # THE CONDUCTANCE CLASS SQUARES W: its message is W^2 * act(v_j) *
            # (E - v_i), so the parameter is the square root of the conductance
            # the readout recovered. A negative recovered value has no square
            # root and no meaning on a conductance edge -- the sign there lives
            # in the driving force -- so it enters as zero and is counted.
            if getattr(model, "model", "") == "flyvis_conductance_known_ode" or \
                    hasattr(model, "edge_is_inh"):
                _neg = int((_col < 0).sum())
                _col = np.sqrt(np.clip(_col, 0.0, None))
                written["W_negative_set_to_zero"] = _neg
            sd["W"] = torch.zeros_like(sd["W"])
            sd["W"][:n_edges] = torch.as_tensor(_col, dtype=torch.float32)[:, None]
            written["W"] = n_edges
            written["W_unfitted_set_to_zero"] = _unfitted

    tau = rec.pairs.get("tau")
    if tau is not None and "raw_tau" in sd:
        t = np.asarray(tau[1], dtype=np.float64).ravel()[:n_neurons]
        if t.size == n_neurons:
            sd["raw_tau"] = torch.as_tensor(_inverse_softplus(t), dtype=torch.float32)
            written["tau"] = n_neurons

    vr = rec.pairs.get("V_rest")
    if vr is not None and "V_rest" in sd:
        v = np.asarray(vr[1], dtype=np.float32).ravel()[:n_neurons]
        if v.size == n_neurons:
            sd["V_rest"] = torch.as_tensor(v, dtype=torch.float32)
            written["V_rest"] = n_neurons

    return sd, written


def _verify_round_trip(model, rec, n_neurons, n_edges, written):
    """Read the parameters back out of the model and compare to what went in.

    Returns the worst relative deviation seen, and raises only when it exceeds
    `_ROUND_TRIP_REL_ABORT`: a rollout of a model that is not the fit produces a
    number that looks like a result and is not one, but a deviation of a fraction
    of a percent is the storage transform and refusing to roll out over it
    costs the run both of its rollouts.
    """
    bad = []
    rel = 0.0
    if "tau" in written and hasattr(model, "get_learned_tau"):
        got = np.asarray(model.get_learned_tau().cpu(), dtype=np.float64).ravel()[:n_neurons]
        want = np.asarray(rec.pairs["tau"][1], dtype=np.float64).ravel()[:n_neurons]
        err = float(np.nanmax(np.abs(got - want)))
        _scale = float(np.nanmax(np.abs(want))) or 1.0
        rel = max(rel, err / _scale)
        if not (err < _ROUND_TRIP_TOL):
            bad.append(f"tau: max |read - written| = {err:.3g} "
                       f"({100.0 * err / _scale:.2f}% of its range)")
    if "V_rest" in written and hasattr(model, "get_learned_vrest"):
        got = np.asarray(model.get_learned_vrest().cpu(), dtype=np.float64).ravel()[:n_neurons]
        want = np.asarray(rec.pairs["V_rest"][1], dtype=np.float64).ravel()[:n_neurons]
        err = float(np.nanmax(np.abs(got - want)))
        _scale = float(np.nanmax(np.abs(want))) or 1.0
        rel = max(rel, err / _scale)
        if not (err < _ROUND_TRIP_TOL):
            bad.append(f"V_rest: max |read - written| = {err:.3g} "
                       f"({100.0 * err / _scale:.2f}% of its range)")
    if "W" in written:
        got = np.asarray(model.W.detach().cpu(), dtype=np.float64).ravel()[:n_edges]
        # The conductance class holds the square root, so compare what its
        # message actually uses, and only over the edges that had a value to
        # write: a negative recovered conductance was clipped to zero on purpose.
        if "W_negative_set_to_zero" in written:
            got = got ** 2
        _src = rec.diagnostics.get("_W_learned_full")
        if _src is None or np.asarray(_src).size < n_edges:
            _src = rec.pairs["W"][1]
        want = np.nan_to_num(np.asarray(_src, dtype=np.float64).ravel()[:n_edges],
                             nan=0.0, posinf=0.0, neginf=0.0)
        if "W_negative_set_to_zero" in written:
            want = np.clip(want, 0.0, None)
        err = float(np.nanmax(np.abs(got - want)))
        _scale = float(np.nanmax(np.abs(want))) or 1.0
        rel = max(rel, err / _scale)
        if not (err < _ROUND_TRIP_TOL):
            bad.append(f"W: max |read - written| = {err:.3g} "
                       f"({100.0 * err / _scale:.2f}% of its range)")
    if bad and rel > _ROUND_TRIP_REL_ABORT:
        raise ValueError("template checkpoint does not read back as written -- "
                         + "; ".join(bad))
    if bad:
        written["roundtrip_max_rel_dev"] = round(rel, 6)
        print(f"\033[93mtemplate checkpoint read back {100.0 * rel:.2f}% off "
              f"({'; '.join(bad)}); rolling out anyway\033[0m")
    return rel


def _prepare_conductance(model, rec, edges, x_ts, n_neurons, n_edges, notes):
    """Configure the conductance known-ODE from the readout, and say what was lost.

    THE CLASS DOES NOT TAKE PER-EDGE REVERSALS. It carries E_exc and E_inh per
    postsynaptic neuron (or per type) with `edge_is_inh` choosing between them,
    while the readout produces one E per edge. Collapsing the second into the
    first is the only way to roll these parameters out at all, and it is lossy:
    the per-edge spread is replaced by a median per neuron per channel. That is
    recorded rather than hidden, because a rollout of the median is a weaker
    statement than a rollout of what was measured.

      sign     an edge is inhibitory when its recovered reversal sits below the
               recovered resting potential of the cell it drives -- the readout's
               own numbers, never the generator's.
      E rows   median recovered E over each neuron's excitatory and inhibitory
               incoming edges, over the edges whose reversal cleared the t-gate.
      range    set_teacher_voltage_range from the trajectory's own extremes,
               which the class requires before any forward pass.
    """
    E_full = rec.diagnostics.get("_tmpl_E_full")
    if E_full is None:
        raise ValueError("no per-edge reversals in the readout: nothing to set E from")
    E_e = np.asarray(E_full, dtype=np.float64).ravel()[:n_edges]
    e_np = to_numpy(edges).reshape(2, -1)[:, :n_edges]
    dst = e_np[1].astype(np.int64)

    vr = rec.pairs.get("V_rest")
    v_rest = (np.asarray(vr[1], dtype=np.float64).ravel() if vr is not None
              else np.zeros(n_neurons))
    if v_rest.size != n_neurons:
        v_rest = np.zeros(n_neurons)

    finite = np.isfinite(E_e)
    is_inh = np.zeros(n_edges, dtype=bool)
    is_inh[finite] = E_e[finite] < v_rest[dst[finite]]
    notes["E_identified"] = int(finite.sum())
    notes["edges_inhibitory"] = int(is_inh.sum())

    # The voltage range the class needs, from the trajectory itself.
    v_lo, v_hi = float("inf"), float("-inf")
    for k in range(0, int(x_ts.n_frames), max(1, int(x_ts.n_frames) // 64)):
        fr = x_ts.frame(int(k))
        v = to_numpy(getattr(fr, "voltage", fr)).ravel()[:n_neurons]
        v_lo, v_hi = min(v_lo, float(v.min())), max(v_hi, float(v.max()))
    model.set_teacher_voltage_range(v_lo, v_hi)
    model.set_presynaptic_sign(torch.as_tensor(
        np.where(is_inh, -1.0, 1.0), dtype=torch.float32))
    notes["voltage_range"] = (round(v_lo, 3), round(v_hi, 3))

    # Per-neuron medians, one per channel, over the identified edges only.
    def _median_by_dst(mask):
        out = np.full(n_neurons, np.nan)
        sel = finite & mask
        if sel.any():
            order = np.argsort(dst[sel])
            d_s, e_s = dst[sel][order], E_e[sel][order]
            bounds = np.searchsorted(d_s, np.arange(n_neurons + 1))
            for i in range(n_neurons):
                lo, hi = bounds[i], bounds[i + 1]
                if hi > lo:
                    out[i] = np.median(e_s[lo:hi])
        return out

    E_exc_n, E_inh_n = _median_by_dst(~is_inh), _median_by_dst(is_inh)
    with torch.no_grad():
        for name, vals in (("E_exc", E_exc_n), ("E_inh", E_inh_n)):
            par = getattr(model, name, None)
            if par is None:
                continue
            cur = par.detach().cpu().numpy().ravel()
            if cur.size == n_neurons:            # per-neuron rows: write directly
                filled = np.where(np.isfinite(vals), vals, cur)
            else:                                # a coarser granularity: one median
                med = np.nanmedian(vals) if np.isfinite(vals).any() else cur.mean()
                filled = np.full(cur.size, med)
            par.copy_(torch.as_tensor(filled, dtype=torch.float32, device=par.device))
        notes["E_exc_median"] = float(np.nanmedian(E_exc_n))
        notes["E_inh_median"] = float(np.nanmedian(E_inh_n))
    notes["E_collapsed_to"] = "per-neuron median per channel"
    return notes


def write_checkpoint(rec, config, log_dir, device, logger=None, edges=None,
                     x_ts=None, alt=False, filename="template_fit.pt"):
    """models/template_fit.pt: the recovered parameters in a known-ODE.

    With `alt`, the OTHER family's known-ODE carrying the other form's
    constants: the same message read as W*relu(v_j) + C when the model is
    conductance, and as W*relu(v_j)*(E - v_i) + C when it is current. tau and
    V_rest are unchanged -- the update fit does not depend on which form the
    per-edge message was read with -- so only W, and the reversal where the
    alternative has one, come from the other fit.

    Returns (path, known_ode_name, what_was_written), or None when this run has
    no matching known-ODE or the readout recovered nothing to write.
    """
    name = other_known_ode_name(config) if alt else known_ode_name(config)
    if name is None:
        return None
    if rec is None or not any(q in rec.pairs for q in ("W", "tau", "V_rest")):
        return None
    if alt:
        _w = rec.diagnostics.get("_W_alt_full")
        if _w is None:
            return None
        rec = _alt_view(rec)

    from connectome_gnn.models.registry import create_model
    model = create_model(name, aggr_type=config.graph_model.aggr_type,
                         config=config, device=device).to(device)
    n_neurons = int(config.simulation.n_neurons)
    n_edges = int(config.simulation.n_edges)

    sd, written = build_state_dict(rec, n_neurons, n_edges, model)
    if not written:
        return None
    model.load_state_dict(sd, strict=False)
    if name == "flyvis_conductance_known_ode":
        _prepare_conductance(model, rec, edges, x_ts, n_neurons, n_edges, written)
    _rel = _verify_round_trip(model, rec, n_neurons, n_edges, written)
    written["roundtrip_max_rel_dev"] = round(float(_rel), 6)

    out_dir = os.path.join(log_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save({"model_state_dict": model.state_dict()}, path)
    if logger:
        logger.info(f"template checkpoint -> {path} ({written})")
    return path, name, written


class _AltRec:
    """`rec` with the alternative form's W and E in place of the own form's.

    A view, not a copy of the arrays: everything the builder reads -- tau,
    V_rest, the estimators -- is the same object, and only the two entries that
    differ between the families are replaced. `pairs["W"]` is replaced too so
    the round-trip check compares the checkpoint against what was written into
    it rather than against the other fit.
    """

    def __init__(self, rec):
        self.pairs = dict(rec.pairs)
        self.diagnostics = dict(rec.diagnostics)
        self.estimator = getattr(rec, "estimator", {})
        self.correction = getattr(rec, "correction", {})
        self.valid = getattr(rec, "valid", {})
        w_alt = np.asarray(self.diagnostics["_W_alt_full"], dtype=np.float64)
        self.diagnostics["_W_learned_full"] = w_alt
        self.diagnostics["_tmpl_E_full"] = self.diagnostics.get("_E_alt_full")
        if "W" in self.pairs:
            gt = np.asarray(self.pairs["W"][0], dtype=np.float64)
            keep = np.isfinite(w_alt)
            n = min(gt.size, int(keep.sum()))
            self.pairs["W"] = (gt[:n], w_alt[keep][:n])

    def get(self, key):
        return self.pairs.get(key)


def _alt_view(rec):
    return _AltRec(rec)


def _parse_rollout_log(path):
    """The Fisher-z pooled r, the RMSE and the clamp share the tester wrote."""
    out = {}
    if not os.path.isfile(path):
        return out
    text = open(path).read()
    m = re.search(r"^RMSE:\s*([-\d.eE]+)", text, re.M)
    if m:
        out["rmse"] = float(m.group(1))
    # CARRIED INTO metrics.txt BESIDE THE r IT QUALIFIES. A template rollout
    # that saturates the divergence clamp still reports an r -- exp02's
    # conductance arm reported 0.112-0.120 with 67-71% of neuron-frames on the
    # rail -- and that r ranks nothing. `pct_clamped` is what says so.
    m = re.search(r"^Clamped at \+/-[\d.]+ V:\s*([-\d.eE]+)%", text, re.M)
    if m:
        out["pct_clamped"] = float(m.group(1))
    m = re.search(r"Pearson r \(Fisher-z pooled over neurons\):\s*([-\d.eE]+)", text)
    if m:
        out["r"] = float(m.group(1))
    else:                                   # older line, same number
        m = re.search(r"Pearson r.*?:\s*([-\d.eE]+)", text)
        if m:
            out["r"] = float(m.group(1))
    return out


def run(rec, config, log_dir, device, logger=None, test_mode="template",
        edges=None, x_ts=None, alt=False):
    """Roll the recovered parameters out on noise-free data.

    Returns the metrics as a dict with `template_` names, ready to be merged into
    the `scored` dict results/metrics.txt is written from. Never raises: this is
    an extra analysis at the end of a pass, and a failure in it must not cost the
    pass its figures.
    """
    prefix = f"{test_mode}_rollout" if not alt else f"{test_mode}_alt_rollout"
    _ckpt = "template_fit.pt" if not alt else "template_fit_alt.pt"
    test_mode = test_mode if not alt else f"{test_mode}_alt"
    try:
        made = write_checkpoint(rec, config, log_dir, device, logger=logger,
                                edges=edges, x_ts=x_ts, alt=alt,
                                filename=_ckpt)
        if made is None:
            return {}
        _path, name, written = made

        # Two config copies, deep so nothing leaks back into the caller's: one
        # naming the known-ODE as the model to build, one naming the noise-free
        # dataset as the data to test on.
        cfg = config.model_copy(deep=True)
        cfg.graph_model.signal_model_name = name
        nf = config.model_copy(deep=True)
        nf.dataset = noise_free_dataset(config.dataset)

        # A dataset that is not on disk would surface as a load error from deep
        # inside the tester; say it here instead, in one line, naming the path.
        from connectome_gnn.utils import graphs_data_path
        _dir = graphs_data_path(nf.dataset)
        if not os.path.isdir(_dir):
            msg = f"noise-free dataset not found: {_dir}"
            if logger:
                logger.warning(f"template rollout skipped: {msg}")
            print(f"\033[93mtemplate rollout skipped: {msg}\033[0m")
            return {f"{prefix}_error": msg}

        _what = ("the template fit" if not alt
                 else f"the same message read as {name.replace('flyvis_', '').replace('_known_ode', '')}")
        print(f"\033[93mrolling {_what} out on {nf.dataset} ...\033[0m")
        from connectome_gnn.models.graph_tester import data_test_gnn
        data_test_gnn(cfg, best_model=_ckpt, device=device,
                      test_config=nf, test_mode=test_mode)

        _short = nf.dataset.split("/")[-1].replace("flyvis_", "")
        log = os.path.join(log_dir, f"results_rollout_on_{_short}_{test_mode}.log")
        got = _parse_rollout_log(log)
        out = {}
        if "W_unfitted_set_to_zero" in written:
            out[f"{prefix}_W_unfitted"] = int(written["W_unfitted_set_to_zero"])
        if "roundtrip_max_rel_dev" in written:
            out[f"{prefix}_roundtrip_rel_dev"] = float(written["roundtrip_max_rel_dev"])
        if "r" in got:
            out[f"{prefix}_r"] = got["r"]
        if "rmse" in got:
            out[f"{prefix}_rmse"] = got["rmse"]
        if "pct_clamped" in got:
            out[f"{prefix}_pct_clamped"] = got["pct_clamped"]
        out[f"{prefix}_dataset"] = nf.dataset
        out[f"{prefix}_model"] = name
        return out
    except Exception as exc:
        if logger:
            logger.warning(f"template rollout skipped: {type(exc).__name__}: {exc}")
        print(f"\033[91mtemplate rollout skipped: {type(exc).__name__}: {exc}\033[0m")
        return {f"{prefix}_error": f"{type(exc).__name__}: {exc}"}
