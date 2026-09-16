"""Is the reversal error carried by the synapses that barely exist?

Eij_rmse is 7.76 V over every fitted edge, while the eight strong synapses onto
neuron 2895 come back within 0.2-0.8 V of the truth. Both cannot be typical. An
edge whose true conductance is 1e-8 carries no information about its reversal --
the driving force is multiplied by nothing -- yet it enters the RMSE with the
same weight as the strongest synapse in the connectome.

This bins the per-edge reversal error by the DECILE OF THE TRUE CONDUCTANCE and
prints both the unweighted and the conductance-weighted error, which is the
number a paper should report if the claim is "the reversals are recovered".

    python tools/eij_by_weight.py <config> [<config> ...]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.metrics import extract_template_params          # noqa: E402
from connectome_gnn.utils import to_numpy                           # noqa: E402

from pysr_recovery import load_run                                  # noqa: E402


def report(name, device):
    cfg, data, model, _log_dir = load_run(name, device)
    rec = extract_template_params(model, data.ode_params, config=cfg,
                                  edges=data.edges.to(device), x_ts=data.x_ts,
                                  device=device, n_neurons=int(data.n_neurons))
    pair = rec.pairs.get("E_ij")
    if pair is None:
        print(f"{name}: no reversal pair")
        return
    # _pair drops non-finite entries from BOTH sides, so it cannot be indexed
    # against the full edge array; refit the alignment from the arrays the
    # extractor kept.
    e_true = np.asarray(to_numpy(data.ode_params.reversal_per_edge())).ravel()
    w_true = np.asarray(data.ode_params.effective_true_weights(
        to_numpy(data.ode_params.W), to_numpy(data.edges), int(data.n_neurons))).ravel()
    e_learn = rec.diagnostics.get("_tmpl_E_full")
    if e_learn is None:
        print(f"{name}: extractor did not keep the per-edge reversal; "
              "add _tmpl_E_full to its diagnostics")
        return
    ok = np.isfinite(e_learn) & np.isfinite(e_true) & np.isfinite(w_true)
    e_t, e_l, w = e_true[ok], e_learn[ok], np.abs(w_true[ok])
    err = np.abs(e_l - e_t)

    print(f"\n=== {name}")
    print(f"edges fitted: {ok.sum()} of {ok.size}")
    print(f"{'decile of true W':<20}{'n':>8}{'median |W|':>12}"
          f"{'median |dE|':>13}{'rmse dE':>10}")
    q = np.quantile(w, np.linspace(0, 1, 11))
    for i in range(10):
        lo, hi = q[i], q[i + 1]
        m = (w >= lo) & (w <= hi if i == 9 else w < hi)
        if not m.any():
            continue
        print(f"{f'{i + 1}':<20}{int(m.sum()):>8}{np.median(w[m]):>12.4g}"
              f"{np.median(err[m]):>13.3f}{np.sqrt(np.mean(err[m] ** 2)):>10.3f}")

    tot = np.sqrt(np.mean(err ** 2))
    wgt = np.sqrt(np.sum(w * err ** 2) / np.sum(w))
    top = w >= np.quantile(w, 0.9)
    print(f"\nrmse, every fitted edge            {tot:8.3f} V")
    print(f"rmse, weighted by true conductance {wgt:8.3f} V")
    print(f"rmse, top decile of true W         {np.sqrt(np.mean(err[top] ** 2)):8.3f} V")
    print(f"median |dE|, top decile            {np.median(err[top]):8.3f} V")


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    for n in sys.argv[1:]:
        try:
            report(n, dev)
        except Exception as exc:
            print(f"\n=== {n}\n  failed: {type(exc).__name__}: {exc}")
