#!/usr/bin/env python
"""make_eye_GL -- eye G with a wider lateral span, as a derived eye spec.

    python make_eye_GL.py                       # writes eye_fit_GL.json
    python make_eye_GL.py --gain 2.0 --out eye_fit_GL2.json

WHY. Eye G reaches ±6.9° horizontally, a 14.2° span, against the 15° the
tracking task asks for -- `train_eyeG.GATE_H` -- so every run so far has
printed a gate warning and the world has been scaled down to fit an eye that
is slightly too small for it. eye_GL is the same eye with the horizontal
channel amplified, so the span question can be separated from the circuit
question instead of confounded with it.

WHAT IS CHANGED, EXACTLY. `EyeG.equilibrium` is

    g^k(m) = sum_i a^k_i m_i + sum_{i<=j} b^k_ij m_i m_j,   k in (theta, phi, psi)

with the 6 linear, 6 square and 15 cross coefficients packed as `beta`, a
(27, 3) matrix -- one column per gaze angle. This scales COLUMN 0, the theta
column, by `gain`, and touches nothing else:

    beta_GL[:, 0] = gain * beta_G[:, 0]

That is a pure gain on horizontal gaze. Because it multiplies all 27 rows of
the column together, the SHAPE of the horizontal response is preserved
exactly -- which muscle pulls where, and every interaction between them -- and
only its size changes. The dynamics (C, K) are untouched, so the eye's time
constants and damping are eye G's.

WHAT THIS IS NOT. It is not a re-fit, not a measurement, and not a claim about
any real fish. eye G is a fit to the MPM soft-body eye's own step responses;
eye_GL is that fit with one column rescaled, and a real eye with twice the
lateral reach would differ in its dynamics too, not only in its gain. Use it
to ask whether the circuit's tracking error is limited by the plant's span,
and say which eye a number came from.

ONE SIDE EFFECT WORTH STATING. phi is untouched while theta grows, so the
phi/theta ratio FALLS: at gain 1.8 the medial rectus drags 3.69° of phi per
6.92° of theta at eye G, and per 12.5° at eye_GL. The vertical excursion is
identical in degrees; it is simply a smaller fraction of the horizontal one.
So eye_GL makes the gaze look more horizontal without any change to the
vertical problem, which is anatomical (see test_zebra_eyeG's docstring) and
still needs OMN in the pool.

eye G is a LEFT eye.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def derive(spec, gain):
    """Return a copy of `spec` with the theta column of beta scaled."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v)
           for k, v in spec.items()}
    beta = np.asarray(spec["beta"], float).copy()
    if beta.shape[1] != 3:
        raise ValueError(f"beta is {beta.shape}, expected (27, 3)")
    beta[:, 0] *= float(gain)
    out["beta"] = beta
    out["derived_from"] = "eye_G"
    out["lateral_gain"] = float(gain)
    out["side"] = "left"
    return out


def main():
    import sys
    sys.path.insert(0, HERE)
    import torch
    import train_eyeG as TG

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gain", type=float, default=1.8,
                   help="multiplier on the theta column of beta. 1.8 takes the "
                        "span from 14.2 to ~25 deg, comfortably past the 15 "
                        "deg GATE_H the task needs.")
    p.add_argument("--src", default=TG.EYE_FIT_PATH)
    p.add_argument("--out", default=os.path.join(HERE, "eye_fit_GL.json"))
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    a = p.parse_args()

    src = TG.fit_eye(cache=a.src, verbose=False)
    new = derive(src, a.gain)

    for nm, sp in (("eye_G ", src), ("eye_GL", new)):
        eye = TG.EyeG({k: (np.asarray(v) if isinstance(v, list) else v)
                       for k, v in sp.items()}, a.dt)
        reach, pos, neg = eye.reach_deg()
        m = torch.zeros(1, 1, 6)
        m[0, 0, 2] = 1.0                                   # MR alone
        with torch.no_grad():
            u = eye.equilibrium(m)[0, 0].numpy()
        print(f"[{nm}] reach h {reach[0]:5.2f}  v {reach[1]:5.2f}  "
              f"torsion {reach[2]:5.2f} deg   span_h {pos[0] + neg[0]:5.2f}   "
              f"MR alone: theta {u[0]:+6.2f} phi {u[1]:+6.2f}  "
              f"|phi/theta| {abs(u[1] / u[0]):.2f}")

    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in new.items()}, open(a.out, "w"), indent=2)
    print(f"[eye] wrote {a.out}  (gain {a.gain} on the theta column)")


if __name__ == "__main__":
    main()
