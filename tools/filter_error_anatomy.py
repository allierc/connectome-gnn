"""Where the parameter fit's W error comes from, with and without the activity filter.

    PYTHONPATH=src python tools/filter_error_anatomy.py [run ...] [--device cuda:0]

Runs extract_template_params twice on the same checkpoint and frames, recovery.template_frame_filter
True (keep a frame on an edge only when the sender's relu(v_j) clears the 0.52 floor) and False (every
sampled frame), and splits the unfiltered fit's edges by how often the sender is active:
p_j = the share of sampled frames in which relu(v_j) is above the floor (every 64th training frame).
Prints R2_W, the median absolute error |W_fit - W_true| and the share of edges off by more than 1
(the outlier band of W) for: the edges fitted both ways (filtered fit, unfiltered fit), the edges only
the unfiltered fit reaches, and by quartile of p_j. Nothing is written to the run.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "tools")
from clustering_template_vs_chain import ROOT, load                    # noqa: E402
from connectome_gnn.generators.ode_params import load_ode_params_for_run  # noqa: E402
from connectome_gnn.metrics import extract_template_params             # noqa: E402
from connectome_gnn.utils import set_data_root                         # noqa: E402

FLOOR = 0.5247


def r2(t, l):
    return 1.0 - np.mean((l - t) ** 2) / np.var(t)


def report(name, t, l):
    if len(t) == 0:
        print(f"  {name:38s} none"); return
    e = np.abs(l - t)
    print(f"  {name:38s} n {len(t):7,d}  R2_W {r2(t, l):6.3f}  median |err| {np.median(e):.3f}  |err|>1: {100 * np.mean(e > 1):5.2f}%")


def main(run, device="cuda:0"):
    set_data_root(ROOT)
    dev = torch.device(device)
    log_dir, config, x_ts, edges, model = load(run, dev)
    n = int(x_ts.n_neurons)
    op = load_ode_params_for_run(config, device="cpu")
    W_true = np.asarray(op.W.detach().cpu()).ravel().astype(np.float64)
    src = np.asarray(edges[0].cpu()).astype(np.int64)
    v = np.asarray(x_ts.voltage[::64].cpu() if torch.is_tensor(x_ts.voltage) else x_ts.voltage[::64], dtype=np.float64)
    p_node = (np.maximum(v, 0.0) > FLOOR).mean(0)
    p = p_node[src]
    fit = {}
    for flt in (True, False):
        config.recovery.template_frame_filter = flt
        rec = extract_template_params(model, op, config=config, edges=edges, x_ts=x_ts, device=dev, n_neurons=n)
        fit[flt] = np.asarray(rec.diagnostics["_W_learned_full"], dtype=np.float64).ravel()
    ok_f, ok_u = np.isfinite(fit[True]), np.isfinite(fit[False])
    nz = W_true != 0
    both = ok_f & ok_u & nz
    only_u = ~ok_f & ok_u & nz
    print(f"\n{run}: {len(W_true):,} edges; fitted with the filter {int(ok_f.sum()):,}, without {int(ok_u.sum()):,}")
    print(f"sender active in a sampled frame: edges fitted both ways {p[both].mean():.3f} on average, edges only the unfiltered fit reaches {p[only_u].mean():.3f}")
    report("both: filtered fit", W_true[both], fit[True][both])
    report("both: unfiltered fit", W_true[both], fit[False][both])
    report("only the unfiltered fit reaches", W_true[only_u], fit[False][only_u])
    report("all unfiltered-fitted edges", W_true[ok_u & nz], fit[False][ok_u & nz])
    sel = ok_u & nz
    qs = np.quantile(p[sel], [0, 0.25, 0.5, 0.75, 1.0])
    for k in range(4):
        m = sel & (p >= qs[k]) & (p <= qs[k + 1])
        report(f"unfiltered, p_j quartile {k + 1} ({qs[k]:.3f}-{qs[k + 1]:.3f})", W_true[m], fit[False][m])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    args = [a for a in args if a != dev] or ["flyvis_noise_005_010_condl25rc20_cv00", "flyvis_noise_005_blank50_condl25_cv00"]
    for r in args:
        main(r, dev)
