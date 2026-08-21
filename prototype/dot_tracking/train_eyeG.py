#!/usr/bin/env python
"""train_eyeG -- fit the eye G reduced model, then train a controller through it.

    python train_eyeG.py --fit light                  # runs today
    python train_eyeG.py --fit deep                   # needs the 6-D sweep
    python train_eyeG.py --fit light --epochs 250 --tag long

Two characterisations, one script, because the controller is identical in both and
only the eye it is trained through changes. Which one you get is `--fit`:

  LIGHT -- stage 0-lite of PROTOCOL_eye_characterisation.md. Four cardinal synergies
    (SR+SO up, IR+IO down, LR temporal, MR nasal) driven in one simulation. The
    readout emits four non-negative SYNERGY drives and the static map is linear,
    `x_inf = A a`, because one amplitude per synergy measures a direction and a gain
    and cannot measure a curvature. Built from what already exists in
    `archive/eye_G/`: the plateaus come from `pairs_long_diag.json`, the mechanics
    are fitted here from the gaze trace in `pairs_long_curves.npz`.

  DEEP -- stages 1 to 3. Six non-negative MUSCLE drives, the static map additive in
    the six marginals plus whatever pairs the screen flagged, the mechanics 3x3.
    Reads `<eye>/charac/holds.npz`, the table `characterise_eye.py --collect`
    re-assembles over every stage that has landed, so it works the moment stage 1
    finishes and sharpens by itself as stage 2 arrives.

BOTH eyes are 6-in-3-out in the sense that matters: the output is (theta, phi, psi),
horizontal, vertical and TORSION. Eye G's synergies leak hard into torsion -- SR+SO
puts 6.8 deg of it against 11.7 deg of elevation -- so a two-angle eye is not an
option here, and the loss carries a torsion penalty for the reason section 5 of the
oculomotor note gives: six drives against two supervised angles leaves the network
free to wander in a null space nothing scores.

The eye is FROZEN. Its coefficients are buffers, not parameters; only the circuit
learns. Training reuses `learn.load_split` for the corpus, so the trajectories, the
seeds and the disjointness are the same ones every other controller was scored on.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import learn                                            # noqa: E402

EYE_DIR = "/workspace/Plexus/prototype/eye/archive/eye_G"
MODELS = os.path.join(HERE, "models")
BOUND = 0.95                       # arena half-width, as in openloop
GATE_H, GATE_V = 15.0, 10.0        # the span the task needs; 25 was unreachable

# Which muscles make each synergy, and the order the light readout emits them in.
# Indices are eye_anatomy.MUSCLE_KEYS = LR 0, SR 1, MR 2, IR 3, SO 4, IO 5, which is
# also the order `probe_groups` uses, so `groups: [[1,4],[3,5],[0],[2]]` is this.
SYNERGIES = [("up", "SR+SO", [1, 4]), ("down", "IR+IO", [3, 5]),
             ("temporal", "LR", [0]), ("nasal", "MR", [2])]
MUSCLES = ["LR", "SR", "MR", "IR", "SO", "IO"]


# ---------------------------------------------------------------------------
# the eye
# ---------------------------------------------------------------------------
def _spd(M):
    """Cholesky factor of a symmetric positive definite 3x3, so the fitted mechanics
    are stable by construction whatever the optimiser reaches."""
    return torch.linalg.cholesky(torch.as_tensor(M, dtype=torch.float64)
                                 + 1e-6 * torch.eye(3, dtype=torch.float64))


def damp_inv(C, dt, backend=torch):
    """(I + dt C)^-1, so the damping term is taken IMPLICITLY.

    With explicit damping the rollout is stable only while `dt * max(eig C) < 2`, and a
    fitter minimising error at one dt will happily walk C right up to that edge --
    this one reached 1.980 at dt = 0.009 s -- and then diverge when the controller
    rolls the same eye out at 1/60 s. Taking the damping implicitly removes the limit
    and, more importantly, makes the fitted C mean the same thing at both timesteps.
    """
    I = backend.eye(3, dtype=C.dtype, device=getattr(C, "device", None)) \
        if backend is torch else np.eye(3)
    return backend.linalg.inv(I + dt * C)


def rollout(xi, K, C, dt, Minv=None, backend=torch):
    """Semi-implicit in the spring, implicit in the damping:

        v <- (I + dt C)^-1 (v + dt K (x_inf - x)) ;   x <- x + dt v
    """
    Minv = damp_inv(C, dt, backend) if Minv is None else Minv
    lead = xi.shape[:-2]
    x = backend.zeros(*lead, 3, dtype=xi.dtype, device=getattr(xi, "device", None)) \
        if backend is torch else np.zeros(lead + (3,))
    v = x * 0
    out = []
    for k in range(xi.shape[-2]):
        out.append(x)
        v = (v + dt * ((xi[..., k, :] - x) @ K.T)) @ Minv.T
        x = x + dt * v
    return backend.stack(out, -2)


def fit_mechanics(traces, iters=3000, verbose=True):
    """Fit 3x3 C and K to `xdd + C xd + K x = K x_inf` over a list of recorded runs.

    Each trace is (dt, x_inf (T,3), gaze (T,3)). Nothing here re-runs a simulation:
    every hold in the protocol is a step from rest followed by a release, recorded at
    full rate, so the transients the mechanics need are already on disk. That is stage
    3 of the protocol costing three runs instead of twenty-five.
    """
    LK = torch.tensor(_spd(np.diag([400.0] * 3)), requires_grad=True)
    LC = torch.tensor(_spd(np.diag([20.0] * 3)), requires_grad=True)
    opt = torch.optim.Adam([LK, LC], lr=0.05)
    T = [(dt, torch.tensor(np.asarray(xi, np.float64)),
          torch.tensor(np.asarray(g, np.float64))) for dt, xi, g in traces]
    for it in range(iters):
        K = LK @ LK.T + 1e-6 * torch.eye(3, dtype=torch.float64)
        C = LC @ LC.T
        loss = 0.0
        for dt, xi, gi in T:
            loss = loss + ((rollout(xi, K, C, dt) - gi) ** 2).mean()
        loss = loss / len(T)
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and it % 1000 == 0:
            print(f"  [mech] {it:5d}  rms {float(loss.detach()) ** 0.5:.3f} deg")
    K = (LK @ LK.T + 1e-6 * torch.eye(3, dtype=torch.float64)).detach().numpy()
    C = (LC @ LC.T).detach().numpy()
    rms = float(loss.detach()) ** 0.5
    if verbose:
        wn = np.sqrt(np.linalg.eigvalsh(K))
        zeta = np.linalg.eigvalsh(C) / (2 * wn)
        print(f"  [mech] rms {rms:.3f} deg over {len(T)} runs;  "
              f"wn = {np.round(wn, 2)} rad/s   zeta = {np.round(zeta, 2)}"
              + ("   (overdamped: the fit prefers a first-order eye)"
                 if zeta.min() > 1 else ""))
    return C, K, rms


def fit_light(eye_dir=EYE_DIR, verbose=True):
    """The four-synergy eye, from stage 0-lite.

    A comes off the plateaus: each synergy's settled excursion IS its column, in
    degrees per unit of drive, because the probe holds every synergy at a_hi = 1. One
    amplitude measures a direction and a gain and cannot measure a curvature, so the
    static map is linear -- that is the honest limit of a light characterisation, not
    a modelling choice.
    """
    diag = json.load(open(os.path.join(eye_dir, "pairs_long_diag.json")))
    A = np.array([diag["synergies"][key]["gaze_excursion_deg"]
                  for _, key, _ in SYNERGIES], dtype=np.float64)      # (4, 3)
    z = np.load(os.path.join(eye_dir, "pairs_long_curves.npz"), allow_pickle=True)
    frame, gaze, act = z["frame"], np.asarray(z["gaze"], np.float64), z["act"]
    dt_rec = float(np.median(np.diff(frame))) * 0.003
    tonic = float(np.percentile(act, 5))
    a = np.stack([np.clip((act[:, idx].mean(1) - tonic) / max(1e-6, 1 - tonic), 0, 1)
                  for _, _, idx in SYNERGIES], -1)                    # (T, 4)
    C, K, rms = fit_mechanics([(dt_rec, a @ A, gaze)], verbose=verbose)
    return dict(kind="light", act_names=[n for n, _, _ in SYNERGIES],
                A=A, C=C, K=K, fit_rms_deg=rms, dt_fit=dt_rec)


def _charac(eye_dir, name):
    p = os.path.join(eye_dir, "charac", name)
    return json.load(open(p)) if os.path.isfile(p) else []


PAIRS = [(i, j) for i in range(6) for j in range(i + 1, 6)]      # 15


def quad_design(U):
    """(n, 6) drives -> (n, 27): the six linear terms, six squares, fifteen crosses.

    The full quadratic, not an additive model with a few interactions bolted on. Eye
    G's screen found 15 of 15 muscle pairs non-additive, residuals 0.09 to 1.03 deg
    against a 0.20 tolerance and a 0.03 noise floor, which triggered the protocol's
    own stop rule: an eye that far from additive needs a different sampling plan, not
    a longer one. The 64-point Sobol sweep is that plan, and this is the model it
    supports.
    """
    U = np.atleast_2d(np.asarray(U, np.float64))
    return np.concatenate(
        [U, U ** 2, np.stack([U[:, i] * U[:, j] for i, j in PAIRS], -1)], -1)


def fit_deep(eye_dir=EYE_DIR, verbose=True):
    """The six-muscle eye: one joint quadratic per axis over every settled hold.

    Reads every stage the characterisation has written -- 0, 1, 2a and the 6-D Sobol
    sweep -- rather than `holds.npz`, which `--collect` assembles from stages 0/1/2a
    only and so leaves the 64 Sobol rows out of the table.

    Fitted JOINTLY on all of them. The earlier version of this function fitted the
    marginals from single-muscle holds and then read each cross term off a pair
    residual; that is the decomposition stage 2a rejected, and it also conditions
    badly, since every cross term inherits the marginals' error.

    Nothing is constrained to be monotone, and on this eye that matters: LR is
    NEGATIVE at u = 0.10 and only turns over by 0.25. A monotone parameterisation
    would be unable to express the plant's own low-drive behaviour. What follows from
    it -- that the inverse is not unique down there -- is the controller's problem,
    not the fit's, and it is reported below rather than hidden.
    """
    rows = []
    for st in ("stage0.json", "stage1.json", "stage2a.json", "stage2b.json",
               "stage6d.json"):
        rows += [r for r in _charac(eye_dir, st) if r.get("settled")]
    if len(rows) < 40:
        sys.exit(f"[deep] only {len(rows)} settled holds under {eye_dir}/charac/.\n"
                 "       The 6-D sweep is what this model needs; until it lands, "
                 "train against the light eye:  --fit light")
    # Stages 0-2a name only the muscles they drive; the Sobol sweep names all six.
    # Scatter every row into the same 6-vector, with undriven muscles at 0 -- pose is
    # recorded relative to the tonic rest pose, so 0 IS "this muscle adds nothing".
    muscles = MUSCLES
    idx = {m: k for k, m in enumerate(muscles)}
    U = np.zeros((len(rows), 6))
    for k, r in enumerate(rows):
        for nm, lv in zip(r["muscles"], r["level"]):
            U[k, idx[str(nm)]] = float(lv)
    P = np.array([r["pose_deg"] for r in rows], float)
    A = quad_design(U)
    beta, *_ = np.linalg.lstsq(A, P, rcond=None)                 # (27, 3)
    res = P - A @ beta
    rms = np.sqrt((res ** 2).mean(0))
    lin, *_ = np.linalg.lstsq(U, P, rcond=None)
    rms_lin = np.sqrt(((P - U @ lin) ** 2).mean(0))
    if verbose:
        rng = P.max(0) - P.min(0)
        print(f"  [quad] {len(rows)} settled holds, 27 coefficients per axis")
        print(f"  [quad] rms  h {rms[0]:.3f}  v {rms[1]:.3f}  t {rms[2]:.3f} deg "
              f"against ranges {np.round(rng, 1)}")
        print(f"  [quad] linear-only leaves {np.round(rms_lin, 2)} deg -- the squares "
              "and crosses are load-bearing")
        # where the map folds: the drive at which a muscle's own slope changes sign
        for mi, m in enumerate(muscles):
            u = np.zeros((41, 6)); u[:, mi] = np.linspace(0, 1, 41)
            y = quad_design(u) @ beta
            d = np.diff(y[:, np.argmax(np.abs(y[-1]))])
            if (d[0] * d[-1]) < 0:
                turn = float(np.linspace(0, 1, 40)[np.argmax(np.sign(d) != np.sign(d[0]))])
                print(f"  [quad] {m}: slope reverses at u = {turn:.2f} -- "
                      "NOT invertible below it")
    traces = []
    for f in sorted(glob.glob(os.path.join(eye_dir, "charac", "runs", "*_curves.npz"))):
        z = np.load(f)
        if "act" not in z.files or "gaze" not in z.files:
            continue
        traces.append((float(np.median(np.diff(z["t"]))),
                       quad_design(np.asarray(z["act"], np.float64)) @ beta,
                       np.asarray(z["gaze"], np.float64)))
    if not traces:
        sys.exit("[deep] no charac/runs/*_curves.npz to fit the mechanics from")
    C, K, mrms = fit_mechanics(traces[:8], verbose=verbose)
    return dict(kind="quad", act_names=list(muscles), beta=beta, C=C, K=K,
                fit_rms_deg=mrms, static_rms_deg=rms.tolist(), n_holds=len(rows))


class EyeG(nn.Module):
    """Non-negative drives in, three gaze angles out. Frozen: every coefficient is
    a buffer, so `.parameters()` is empty and the optimiser cannot retune the body.

        x_inf = g(m)                            static, measured
        xdd + C xd + K x = K x_inf              linear, fitted
    """

    def __init__(self, spec, dt):
        super().__init__()
        self.kind = spec["kind"]
        self.act_names = list(spec["act_names"])
        self.dt = float(dt)
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.register_buffer("C", f(spec["C"]))
        self.register_buffer("K", f(spec["K"]))
        self.register_buffer("Minv", torch.linalg.inv(
            torch.eye(3, dtype=torch.float32) + float(dt) * f(spec["C"])))
        if self.kind == "light":
            self.register_buffer("A", f(spec["A"]))              # (4, 3)
        elif self.kind == "quad":
            self.register_buffer("beta", f(spec["beta"]))        # (27, 3)
            self.register_buffer("pidx", torch.as_tensor(np.asarray(PAIRS),
                                                         dtype=torch.long))
        else:
            self.register_buffer("phi", f(spec["phi"]))          # (6, 3, 2)
            self.register_buffer("pairs", torch.as_tensor(
                np.asarray(spec["pairs"]), dtype=torch.long))    # (P, 2)
            self.register_buffer("pair_coef", f(spec["pair_coef"]))  # (P, 3)

    @property
    def n_act(self):
        return len(self.act_names)

    def equilibrium(self, m):
        """(B, T, n_act) drives -> (B, T, 3) the pose the eye would settle at."""
        if self.kind == "light":
            return m @ self.A
        if self.kind == "quad":
            cross = m[..., self.pidx[:, 0]] * m[..., self.pidx[:, 1]]
            return torch.cat([m, m ** 2, cross], -1) @ self.beta
        x = torch.einsum("btm,ma->bta", m, self.phi[:, :, 0]) \
            + torch.einsum("btm,ma->bta", m ** 2, self.phi[:, :, 1])
        if len(self.pairs):
            prod = m[..., self.pairs[:, 0]] * m[..., self.pairs[:, 1]]
            x = x + prod @ self.pair_coef
        return x

    def forward(self, m):
        return rollout(self.equilibrium(m), self.K, self.C, self.dt, self.Minv)

    def reach_deg(self):
        """Per-axis reachable travel, the smaller of the two directions, which is
        what the world can be scaled to. Same quantity as panel (c) of the eye-model
        figure, computed on whichever eye is loaded."""
        with torch.no_grad():
            dev = self.C.device
            eye = torch.eye(self.n_act, device=dev)[None]         # (1, n_act, n_act)
            x = self.equilibrium(eye)[0]                          # (n_act, 3)
        pos = np.array([max(0.0, float(x[:, k].max())) for k in range(3)])
        neg = np.array([max(0.0, float(-x[:, k].min())) for k in range(3)])
        return np.minimum(pos, neg), pos, neg


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------
class CTRNNEyeG(nn.Module):
    """The same core as `learn_eye.CTRNNEye` -- continuous-time rate, learned tau,
    recurrent matrix initialised at ZERO -- with the readout widened to whatever the
    eye takes, four synergies or six muscles, and the eye appended."""

    def __init__(self, eye, hidden=64, tau0=0.5, dt=1 / 60):
        super().__init__()
        self.eye, self.dt = eye, dt
        self.log_tau = nn.Parameter(torch.full((hidden,), float(np.log(tau0))))
        self.W = nn.Parameter(torch.zeros(hidden, hidden))
        self.enc = nn.Linear(2, hidden)                 # (xdot, ydot)
        self.mot = nn.Linear(hidden, eye.n_act)
        with torch.no_grad():
            self.enc.weight.mul_(0.5)

    def forward(self, u, want_states=False):
        alpha = (self.dt / self.log_tau.exp()).clamp(1e-4, 1.0)
        B, T, _ = u.shape
        h = torch.zeros(B, self.W.shape[0], device=u.device, dtype=u.dtype)
        drive = self.enc(u)
        rates = []
        for t in range(T):
            r = torch.tanh(h)
            rates.append(r)
            h = h + alpha * (-h + r @ self.W.T + drive[:, t])
        R = torch.stack(rates, 1)
        m = nn.functional.softplus(self.mot(R))         # non-negative drives
        x = self.eye(m)                                 # (B, T, 3) degrees
        return (x, m, R) if want_states else (x, m)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", default="light", choices=["light", "deep"])
    p.add_argument("--eye-dir", default=EYE_DIR)
    p.add_argument("--tag", default=None, help="checkpoint name; default eyeG_<fit>")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dt", type=float, default=1.0 / 60.0)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--lam-psi", type=float, default=0.05,
                   help="torsion penalty; Donders' law in its simplest form")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    tag = a.tag or f"eyeG_{a.fit}"
    dev = torch.device(a.device)
    os.makedirs(MODELS, exist_ok=True)

    # --- the eye ---------------------------------------------------------
    print(f"[eye] fitting the {a.fit} eye from {a.eye_dir}")
    spec = fit_light(a.eye_dir) if a.fit == "light" else fit_deep(a.eye_dir)
    eye = EyeG(spec, a.dt).to(dev)
    reach, pos, neg = eye.reach_deg()
    print(f"[eye] {eye.n_act} drives: {', '.join(eye.act_names)}")
    print(f"[eye] reach  h {reach[0]:.1f}  v {reach[1]:.1f}  torsion {reach[2]:.1f} deg"
          f"   (+{pos[0]:.1f}/-{neg[0]:.1f} h, +{pos[1]:.1f}/-{neg[1]:.1f} v)")
    span_h, span_v = pos[0] + neg[0], pos[1] + neg[1]
    rp = os.path.join(a.eye_dir, "charac", "report.json")
    if os.path.isfile(rp):
        r = json.load(open(rp))
        print(f"[gate] the eye's own report: span_h {r.get('span_h')}, "
              f"span_v {r.get('span_v')}, gate_pass {r.get('gate_pass')}")
        if abs(float(r.get("span_h", 0)) - span_h) > 1.0:
            print(f"[gate] NOTE: the eye's report reads the SYNERGY workspace "
                  f"({r.get('span_h'):.1f} deg) while this eye model reaches "
                  f"{span_h:.1f} on its own drives. The larger one is what a "
                  "controller can command only if it may spend the torsion that "
                  "comes with the recruitment.")
    if span_h < GATE_H or span_v < GATE_V:
        print(f"[gate] WARNING: span {span_h:.1f} deg h / {span_v:.1f} deg v against "
              f"the {GATE_H:.0f}/{GATE_V:.0f} the task needs.")
        print("[gate] Training proceeds, but the world below is scaled to what this "
              "eye can actually reach, so the task is EASIER than specified and the "
              "error is not comparable with an eye that passes the gate.")

    # --- data: the shared corpus, scaled to this eye's own reach ---------
    scale = np.array([reach[0], reach[1]], np.float32) / BOUND    # deg per grid unit
    print(f"[data] anisotropic world: h x{scale[0]:.2f}, v x{scale[1]:.2f} deg/unit "
          f"(+-{reach[0]:.1f} / +-{reach[1]:.1f} deg)")
    sp = {}
    for nm, n, s0 in (("train", 150, 0), ("val", 25, 5_000_000),
                      ("test", 40, 9_000_000)):
        u, y, _ = learn.load_split(nm, n, a.duration, a.dt, s0)
        sp[nm] = (torch.as_tensor(u).to(dev),
                  torch.as_tensor(y * scale).to(dev))             # target in degrees
    (Utr, Ytr), (Uva, Yva), (Ute, Yte) = sp["train"], sp["val"], sp["test"]

    torch.manual_seed(0)
    model = CTRNNEyeG(eye, hidden=a.hidden, dt=a.dt).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    T = Utr.shape[1]
    sched = [max(60, int(T * f)) for f in (0.25, 0.5, 0.75, 1.0)]   # horizon curriculum

    def evaluate(U, Y):
        model.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, U.shape[0], a.batch):
                x, _ = model(U[i:i + a.batch])
                errs.append((x[..., :2] - Y[i:i + a.batch]).norm(dim=-1))
            return float(torch.cat(errs).mean())

    best, best_state = np.inf, None
    for ep in range(a.epochs):
        model.train()
        h = sched[min(3, int(4 * ep / max(a.epochs - 1, 1)))]
        perm = torch.randperm(Utr.shape[0], device=dev)
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            j = perm[i:i + a.batch]
            x, _ = model(Utr[j, :h])
            loss = ((x[..., :2] - Ytr[j, :h]) ** 2).mean() \
                + a.lam_psi * (x[..., 2] ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss) * len(j)
        sch.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            e = evaluate(Uva, Yva)
            print(f"  ep {ep:3d}  horizon {h:3d}  train {tot / len(perm):.5f}  "
                  f"val |err| {e:.3f} deg")
            if e < best:
                best, best_state = e, {k: v.detach().clone()
                                       for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(Ute, Yte)
    with torch.no_grad():
        x, m = model(Ute[:64])
        psi = float(x[..., 2].abs().mean())
        occ = float((m > 1e-3).float().mean())
    print(f"[done] test |err| {test:.3f} deg   mean |torsion| {psi:.2f} deg   "
          f"drives active {occ * 100:.0f}% of the time")

    ck = os.path.join(MODELS, f"{tag}.pt")
    torch.save({"state": model.state_dict(),
                "eye": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in spec.items()},
                "eye_shapes": {k: list(v.shape) for k, v in spec.items()
                               if isinstance(v, np.ndarray)},
                "hidden": a.hidden, "dt": a.dt, "scale": scale.tolist(),
                "act_names": eye.act_names, "kind": eye.kind}, ck)
    rep = os.path.join(MODELS, f"{tag}.json")
    json.dump({"tag": tag, "fit": a.fit, "n_act": eye.n_act,
               "act_names": eye.act_names,
               "gaze_err_mean_deg": test, "val_err_deg": best,
               "mean_abs_torsion_deg": psi,
               "deg_per_unit_h": float(scale[0]), "deg_per_unit_v": float(scale[1]),
               "span_h_deg": float(span_h), "span_v_deg": float(span_v),
               "gate_pass": bool(span_h >= GATE_H and span_v >= GATE_V),
               "mech_fit_rms_deg": spec.get("fit_rms_deg"),
               "epochs": a.epochs, "hidden": a.hidden, "lam_psi": a.lam_psi},
              open(rep, "w"), indent=2)
    print(f"[done] wrote {ck}\n       wrote {rep}")


if __name__ == "__main__":
    main()
