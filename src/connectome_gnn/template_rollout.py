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

# The known-ODE that computes each family's generator form. Keyed by the
# signal_model_name of the trained model the parameters came out of.
_KNOWN_ODE_FOR = {
    "flyvis_current": "flyvis_known_ode",
    "flyvis": "flyvis_known_ode",
    "flyvis_conductance": "flyvis_conductance_known_ode",
}

_ROUND_TRIP_TOL = 1e-5


def known_ode_name(config):
    """The known-ODE that matches this run's model, or None if there is none."""
    return _KNOWN_ODE_FOR.get(getattr(config.graph_model, "signal_model_name", ""))


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

    Raises rather than returns: a rollout of a model that is not the fit produces
    a number that looks like a result and is not one.
    """
    bad = []
    if "tau" in written and hasattr(model, "get_learned_tau"):
        got = np.asarray(model.get_learned_tau().cpu(), dtype=np.float64).ravel()[:n_neurons]
        want = np.asarray(rec.pairs["tau"][1], dtype=np.float64).ravel()[:n_neurons]
        err = float(np.nanmax(np.abs(got - want)))
        if not (err < _ROUND_TRIP_TOL):
            bad.append(f"tau: max |read - written| = {err:.3g}")
    if "V_rest" in written and hasattr(model, "get_learned_vrest"):
        got = np.asarray(model.get_learned_vrest().cpu(), dtype=np.float64).ravel()[:n_neurons]
        want = np.asarray(rec.pairs["V_rest"][1], dtype=np.float64).ravel()[:n_neurons]
        err = float(np.nanmax(np.abs(got - want)))
        if not (err < _ROUND_TRIP_TOL):
            bad.append(f"V_rest: max |read - written| = {err:.3g}")
    if "W" in written:
        got = np.asarray(model.W.detach().cpu(), dtype=np.float64).ravel()[:n_edges]
        _src = rec.diagnostics.get("_W_learned_full")
        if _src is None or np.asarray(_src).size < n_edges:
            _src = rec.pairs["W"][1]
        want = np.nan_to_num(np.asarray(_src, dtype=np.float64).ravel()[:n_edges],
                             nan=0.0, posinf=0.0, neginf=0.0)
        err = float(np.nanmax(np.abs(got - want)))
        if not (err < _ROUND_TRIP_TOL):
            bad.append(f"W: max |read - written| = {err:.3g}")
    if bad:
        raise ValueError("template checkpoint does not read back as written -- "
                         + "; ".join(bad))


def write_checkpoint(rec, config, log_dir, device, logger=None):
    """models/template_fit.pt: the recovered parameters in a known-ODE.

    Returns (path, known_ode_name, what_was_written), or None when this run has
    no matching known-ODE or the readout recovered nothing to write.
    """
    name = known_ode_name(config)
    if name is None:
        return None
    if rec is None or not any(q in rec.pairs for q in ("W", "tau", "V_rest")):
        return None

    from connectome_gnn.models.registry import create_model
    model = create_model(name, aggr_type=config.graph_model.aggr_type,
                         config=config, device=device).to(device)
    n_neurons = int(config.simulation.n_neurons)
    n_edges = int(config.simulation.n_edges)

    sd, written = build_state_dict(rec, n_neurons, n_edges, model)
    if not written:
        return None
    model.load_state_dict(sd, strict=False)
    _verify_round_trip(model, rec, n_neurons, n_edges, written)

    out_dir = os.path.join(log_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "template_fit.pt")
    torch.save({"model_state_dict": model.state_dict()}, path)
    if logger:
        logger.info(f"template checkpoint -> {path} ({written})")
    return path, name, written


def _parse_rollout_log(path):
    """The Fisher-z pooled r and the RMSE the tester wrote, as a dict."""
    out = {}
    if not os.path.isfile(path):
        return out
    text = open(path).read()
    m = re.search(r"^RMSE:\s*([-\d.eE]+)", text, re.M)
    if m:
        out["rmse"] = float(m.group(1))
    m = re.search(r"Pearson r \(Fisher-z pooled over neurons\):\s*([-\d.eE]+)", text)
    if m:
        out["r"] = float(m.group(1))
    else:                                   # older line, same number
        m = re.search(r"Pearson r.*?:\s*([-\d.eE]+)", text)
        if m:
            out["r"] = float(m.group(1))
    return out


def run(rec, config, log_dir, device, logger=None, test_mode="template"):
    """Roll the recovered parameters out on noise-free data.

    Returns the metrics as a dict with `template_` names, ready to be merged into
    the `scored` dict results/metrics.txt is written from. Never raises: this is
    an extra analysis at the end of a pass, and a failure in it must not cost the
    pass its figures.
    """
    prefix = f"{test_mode}_rollout"
    try:
        made = write_checkpoint(rec, config, log_dir, device, logger=logger)
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

        print(f"\033[93mrolling the template fit out on {nf.dataset} ...\033[0m")
        from connectome_gnn.models.graph_tester import data_test_gnn
        data_test_gnn(cfg, best_model="template_fit.pt", device=device,
                      test_config=nf, test_mode=test_mode)

        _short = nf.dataset.split("/")[-1].replace("flyvis_", "")
        log = os.path.join(log_dir, f"results_rollout_on_{_short}_{test_mode}.log")
        got = _parse_rollout_log(log)
        out = {}
        if "W_unfitted_set_to_zero" in written:
            out[f"{prefix}_W_unfitted"] = int(written["W_unfitted_set_to_zero"])
        if "r" in got:
            out[f"{prefix}_r"] = got["r"]
        if "rmse" in got:
            out[f"{prefix}_rmse"] = got["rmse"]
        out[f"{prefix}_dataset"] = nf.dataset
        out[f"{prefix}_model"] = name
        return out
    except Exception as exc:
        if logger:
            logger.warning(f"template rollout skipped: {type(exc).__name__}: {exc}")
        print(f"\033[91mtemplate rollout skipped: {type(exc).__name__}: {exc}\033[0m")
        return {f"{prefix}_error": f"{type(exc).__name__}: {exc}"}
