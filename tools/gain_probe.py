"""What is coeff_f_theta_msg_gain's residual actually worth on a trained model?

The term penalises || (df/dmsg + df/dv) / rms(df/dv) ||_2, and the identity it
enforces says that ratio is 1 -- so on a model whose update template reports
G = 0.0404, the per-neuron residual should be |G - 1| ~ 0.96. Measured inside
training it came to 7.9e-05, which is not a small coefficient, it is a different
number. This prints the pieces side by side: the two finite-difference
derivatives the regulariser uses, the same two by autograd, and the G the update
template fits, all on the same features.

    python tools/gain_probe.py <config>
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from connectome_gnn.metrics import _update_template_fit, compute_grad_msg   # noqa: E402
from connectome_gnn.utils import to_numpy                                   # noqa: E402

from pysr_recovery import load_run                                          # noqa: E402


def main(name, device):
    cfg, data, model, _ = load_run(name, device)
    core = getattr(model, "_orig_mod", model)
    n = int(data.n_neurons)
    emb = int(core.a.shape[1])
    edges = data.edges.to(device)
    did = torch.zeros((n, 1), dtype=torch.int, device=device)

    with torch.no_grad():
        st = data.x_ts.frame(int(data.x_ts.n_frames) // 2).to(device)
        _pred, feats, _msg = core(st, edges, data_id=did, return_all=True)

    base = feats.clone().detach()
    m_col = emb + 1
    print(f"features: {base.shape}, embedding_dim {emb}, msg column {m_col}")
    for c, nm in ((0, "v"), (m_col, "msg"), (m_col + 1, "stim")):
        col = base[:, c]
        print(f"  col {c:2d} ({nm:4s}): mean {float(col.mean()):+.4g}  "
              f"std {float(col.std()):.4g}  min {float(col.min()):+.4g}  "
              f"max {float(col.max()):+.4g}")

    for d in (1e-6, 1e-3, 0.05 * float(base[:, 0].std()), 0.1, 1.0):
        fv = base.clone(); fv[:, 0] = fv[:, 0] + d
        fm = base.clone(); fm[:, m_col] = fm[:, m_col] + d
        with torch.no_grad():
            f0 = core.f_theta(base)
            dv = (core.f_theta(fv) - f0) / d
            dm = (core.f_theta(fm) - f0) / d
        r = (dm + dv).ravel()
        scale = float((dv ** 2).mean().sqrt())
        print(f"  step {d:>10.4g}:  df/dv median {float(dv.median()):+10.4f}   "
              f"df/dmsg median {float(dm.median()):+10.4f}   "
              f"residual/rms median {float((r / max(scale, 1e-12)).abs().median()):.4g}")

    g = to_numpy(compute_grad_msg(core, feats, cfg)).ravel()
    print(f"\nautograd df/dmsg: median {np.median(g):+.4f}  "
          f"absmedian {np.median(np.abs(g)):.4f}")
    T, V, G, r2, _sl, _off = _update_template_fit(core, cfg, edges, data.x_ts, n,
                                                  device, n_frames=32)
    print(f"update template : T median {np.median(T):+.4f}   G median "
          f"{np.median(G):+.6f}   R2 median {np.median(r2):.4f}")
    print(f"the term's target is G = 1, so the per-neuron residual should be "
          f"~{abs(np.median(G) - 1.0):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], "cuda:0" if torch.cuda.is_available() else "cpu")
