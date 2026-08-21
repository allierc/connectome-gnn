#!/usr/bin/env python
"""train_eyeG -- fit the eye G reduced model, then train a controller through it.

    python train_eyeG.py
    python train_eyeG.py --epochs 250 --tag long
    python train_eyeG.py --refit                      # ignore the cached eye fit

Six non-negative MUSCLE drives in, three gaze angles out -- (theta, phi, psi),
horizontal, vertical and TORSION. Eye G's synergies leak hard into torsion -- SR+SO
puts 6.8 deg of it against 11.7 deg of elevation -- so a two-angle eye is not an
option here, and the loss carries a torsion penalty for the reason section 5 of the
oculomotor note gives: six drives against two supervised angles leaves the network
free to wander in a null space nothing scores.

The eye is FROZEN. Its coefficients are buffers, not parameters; only the circuit
learns. Training reuses `learn.load_split` for the corpus, so the trajectories, the
seeds and the disjointness are the same ones every other controller was scored on.

THE EYE FIT IS CACHED. Fitting it needs `Plexus/prototype/eye/archive/eye_G/`, which
is ~440 MB and lives outside this repo -- fine for whoever ran the characterisation,
not for a colleague who just wants to repeat the training. `fit_eye` writes its
result to `EYE_FIT_PATH` (small enough to commit) the first time it runs; every run
after that, here or on someone else's clone, loads the cached numbers and never
touches the Plexus archive at all. Delete the file, or pass `--refit`, to redo it.
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
EYE_FIT_PATH = os.path.join(HERE, "eye_fit.json")
MODELS = os.path.join(HERE, "models")
BOUND = 0.95                       # arena half-width, as in openloop
GATE_H, GATE_V = 15.0, 10.0        # the span the task needs; 25 was unreachable
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


def rollout(u_inf, K, C, dt, Minv=None, backend=torch):
    """The note's eq:euler-eye, the discretisation of the mechanics eq:mechanics,
    u-double-dot + C u-dot + K u = K u_inf, semi-implicit in the spring and
    implicit in the damping:

        u_dot <- (I + dt C)^-1 (u_dot + dt K (u_inf - u)) ;   u <- u + dt u_dot

    `u_inf` is the commanded equilibrium g(m) of eq:quadratic; `u` is the gaze
    (theta, phi, psi) of eq:gaze-vector, NOT the target position (x, y) -- the
    note picks the letter u for exactly this reason (see step 6 of section 4).
    """
    Minv = damp_inv(C, dt, backend) if Minv is None else Minv
    lead = u_inf.shape[:-2]
    u = backend.zeros(*lead, 3, dtype=u_inf.dtype, device=getattr(u_inf, "device", None)) \
        if backend is torch else np.zeros(lead + (3,))
    u_dot = u * 0
    out = []
    for k in range(u_inf.shape[-2]):
        out.append(u)
        u_dot = (u_dot + dt * ((u_inf[..., k, :] - u) @ K.T)) @ Minv.T
        u = u + dt * u_dot
    return backend.stack(out, -2)


def fit_mechanics(traces, iters=3000, verbose=True):
    """Fit C and K of the note's eq:mechanics, u-double-dot + C u-dot + K u =
    K u_inf, over a list of recorded runs -- eq:mech-fit of the note, minimising
    the summed squared rollout error by gradient descent rather than a
    closed-form solve, since u at frame t depends on every earlier frame.

    Each trace is (dt, u_inf (T,3), u_true (T,3), the recorded gaze). Nothing here
    re-runs a simulation: every hold in the protocol is a step from rest followed
    by a release, recorded at full rate, so the transients the mechanics need are
    already on disk. That is stage 3 of the protocol costing three runs instead of
    twenty-five.
    """
    LK = _spd(np.diag([400.0] * 3)).clone().detach().requires_grad_(True)
    LC = _spd(np.diag([20.0] * 3)).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([LK, LC], lr=0.05)
    T = [(dt, torch.tensor(np.asarray(u_inf, np.float64)),
          torch.tensor(np.asarray(u_true, np.float64))) for dt, u_inf, u_true in traces]
    for it in range(iters):
        K = LK @ LK.T + 1e-6 * torch.eye(3, dtype=torch.float64)
        C = LC @ LC.T
        loss = 0.0
        for dt, u_inf, u_true in T:
            loss = loss + ((rollout(u_inf, K, C, dt) - u_true) ** 2).mean()
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


def _fit_eye_from_archive(eye_dir, verbose=True):
    """The six-muscle eye: one joint quadratic per axis over every settled hold.

    Reads every stage the characterisation has written -- 0, 1, 2a and the 6-D Sobol
    sweep -- rather than `holds.npz`, which `--collect` assembles from stages 0/1/2a
    only and so leaves the 64 Sobol rows out of the table.

    Fitted JOINTLY on all of them. An earlier version of this function fitted the
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
        sys.exit(f"[eye] only {len(rows)} settled holds under {eye_dir}/charac/.\n"
                 "      The 6-D sweep is what this model needs.")
    # Stages 0-2a name only the muscles they drive; the Sobol sweep names all six.
    # Scatter every row into the same 6-vector, with undriven muscles at 0 -- pose is
    # recorded relative to the tonic rest pose, so 0 IS "this muscle adds nothing".
    idx = {m: k for k, m in enumerate(MUSCLES)}
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
        for mi, m in enumerate(MUSCLES):
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
        sys.exit("[eye] no charac/runs/*_curves.npz to fit the mechanics from")
    C, K, mrms = fit_mechanics(traces[:8], verbose=verbose)
    return dict(kind="quad", act_names=list(MUSCLES), beta=beta, C=C, K=K,
                fit_rms_deg=mrms, static_rms_deg=rms.tolist(), n_holds=len(rows))


def fit_eye(eye_dir=EYE_DIR, cache=EYE_FIT_PATH, refit=False, verbose=True):
    """Load the cached eye fit if there is one; otherwise fit it and cache it.

    A fresh fit needs `eye_dir` -- the ~440 MB Plexus characterisation archive.
    A cached fit needs nothing but `cache`, which is small enough to commit and
    push, so this is the one function a colleague repeating the training actually
    calls: as long as `eye_fit.json` made the trip, `eye_dir` never has to.
    """
    if not refit and os.path.isfile(cache):
        spec = json.load(open(cache))
        spec["beta"] = np.array(spec["beta"], np.float64)
        spec["C"] = np.array(spec["C"], np.float64)
        spec["K"] = np.array(spec["K"], np.float64)
        print(f"[eye] loaded cached fit from {cache}  "
              f"(delete it, or pass --refit, to redo it from {eye_dir})")
        return spec
    spec = _fit_eye_from_archive(eye_dir, verbose=verbose)
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in spec.items()}, open(cache, "w"), indent=2)
    print(f"[eye] fitted from {eye_dir}, wrote {cache}")
    return spec


class EyeG(nn.Module):
    """Non-negative drives in, three gaze angles out -- the note's eq:hammerstein
    and eq:eye-io. Frozen: every coefficient is a buffer, so `.parameters()` is
    empty and the optimiser cannot retune the body.

        u_inf = g(m)                     static, measured (steady state; m = the
                                          six non-negative muscle drives, (..., 6))
        u_ddot + C u_dot + K u = K u_inf  linear, fitted (from the MPM soft-body
                                          eye's own step responses; u = the gaze
                                          (theta, phi, psi), (..., 3))

    u, not x: (x, y) is already the target's position in the world (eq:world-scale),
    whose derivative is the circuit's only input, so the note reserves u for the
    gaze specifically to keep the two from colliding.
    """

    def __init__(self, spec, dt):
        super().__init__()
        self.kind = spec.get("kind", "quad")
        self.act_names = list(spec["act_names"])
        self.dt = float(dt)
        f = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.register_buffer("C", f(spec["C"]))
        self.register_buffer("K", f(spec["K"]))
        self.register_buffer("Minv", torch.linalg.inv(
            torch.eye(3, dtype=torch.float32) + float(dt) * f(spec["C"])))
        self.register_buffer("beta", f(spec["beta"]))             # (27, 3)
        self.register_buffer("pidx", torch.as_tensor(np.asarray(PAIRS),
                                                     dtype=torch.long))

    @property
    def n_act(self):
        return len(self.act_names)

    def equilibrium(self, m):
        """(B, T, 6) drives -> (B, T, 3) u_inf = g(m), the note's eq:quadratic:
        g^k(m) = sum_i a^k_i m_i + sum_{i<=j} b^k_ij m_i m_j, k in {theta, phi, psi}.
        `self.beta` is (27, 3): the six a^k_i, six b^k_ii and fifteen b^k_ij packed
        into one matrix per k, so the three sums become one matmul."""
        cross = m[..., self.pidx[:, 0]] * m[..., self.pidx[:, 1]]
        return torch.cat([m, m ** 2, cross], -1) @ self.beta

    def forward(self, m):
        u_inf = self.equilibrium(m)                      # eq:quadratic
        return rollout(u_inf, self.K, self.C, self.dt, self.Minv)   # eq:euler-eye

    def reach_deg(self):
        """Per-axis reachable travel, the smaller of the two directions, which is
        what the world can be scaled to. Same quantity as panel (c) of the eye-model
        figure, computed on whichever eye is loaded."""
        with torch.no_grad():
            dev = self.C.device
            eye = torch.eye(self.n_act, device=dev)[None]         # (1, n_act, n_act)
            pose = self.equilibrium(eye)[0]                       # (n_act, 3), u_inf
        pos = np.array([max(0.0, float(pose[:, k].max())) for k in range(3)])
        neg = np.array([max(0.0, float(-pose[:, k].min())) for k in range(3)])
        return np.minimum(pos, neg), pos, neg


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------
class CTRNNEyeG(nn.Module):
    """The circuit of the note's section 4 (steps 1-5), same core as
    `learn_eye.CTRNNEye` -- continuous-time rate, learned tau, recurrent matrix
    initialised at ZERO -- with the readout widened to the six muscle drives the
    eye takes, and the eye (EyeG, steps 6-7) appended.

    THIS IS THE FREE PROTOTYPE, NOT THE CONNECTOME-CONSTRAINED CIRCUIT: section
    5.2 of the note is explicit about it ("what the eye does when coupled to a
    ctRNN, not the zebrafish circuit"). `self.W` here is a free, unconstrained
    (hidden, hidden) matrix -- it stands in for eq:circuit's W_hat, the sign-locked
    recurrent weights of eq:sign-lock, but is not itself sign-locked. Likewise
    `self.enc`/`self.mot` stand in for eq:input-map's W_in and eq:output-map's
    W_out, dense here rather than masked to AF5 / AMN+AIN. Swap this class for
    the constrained one and every equation below still applies unchanged.
    """

    def __init__(self, eye, hidden=64, tau0=0.5, dt=1 / 60):
        super().__init__()
        self.eye, self.dt = eye, dt
        self.log_tau = nn.Parameter(torch.full((hidden,), float(np.log(tau0))))
        self.W = nn.Parameter(torch.zeros(hidden, hidden))   # stands in for W_hat
        self.enc = nn.Linear(2, hidden)          # stands in for W_in, eq:input-map
        self.mot = nn.Linear(hidden, eye.n_act)  # stands in for W_out, eq:output-map
        with torch.no_grad():
            self.enc.weight.mul_(0.5)

    def forward(self, pdot, want_states=False):
        """pdot = p_dot = (x_dot, y_dot), the target velocity of eq:input-vector --
        the circuit's only input. Returns (u, m) -- u the gaze of eq:gaze-vector,
        m the six muscle drives of eq:output-map -- or (u, m, R) with the firing
        rates too if `want_states`."""
        alpha = (self.dt / self.log_tau.exp()).clamp(1e-4, 1.0)   # dt / tau_i, eq:euler-circuit
        B, T, _ = pdot.shape                              # batch, time
        v = torch.zeros(B, self.W.shape[0], device=pdot.device, dtype=pdot.dtype)  # v_i(0)=0
        I = self.enc(pdot)                # I(t) = W_in p_dot(t), eq:input-map
        rates = []
        for t in range(T):
            r = torch.tanh(v)             # r_j = rho(v_j), rho = tanh, eq:circuit
            rates.append(r)
            v = v + alpha * (-v + r @ self.W.T + I[:, t])       # eq:euler-circuit
        R = torch.stack(rates, 1)                         # (B, T, hidden), r_j(t)
        m = nn.functional.softplus(self.mot(R))           # m(t) = [W_out r(t)]_+, eq:output-map
        u = self.eye(m)                                    # (B, T, 3) gaze, degrees
        return (u, m, R) if want_states else (u, m)


def _err_color(deg):
    """ANSI-colour a gaze error in degrees, so a scrollback full of epochs reads at
    a glance. Thresholds are set against the eye's own precision floor rather than
    picked arbitrarily: 0.05 deg is g's static-map residual (eye_fit.json's own
    static_rms_deg), so nothing trained THROUGH that eye can mean much below it;
    0.1 deg is where the previous run's val/test error already sits (see --status);
    0.5 deg is "still clearly training", not yet "broken". Anything at or above 0.5,
    or non-finite, is red -- one bin is enough once a value is that far off, finer
    tiering there would not change what to do about it.
    """
    if not np.isfinite(deg):
        c = "\033[31m"           # red
    elif deg < 0.05:
        c = "\033[32m"           # green
    elif deg < 0.1:
        c = "\033[33m"           # yellow
    elif deg < 0.5:
        c = "\033[38;5;208m"     # orange (no plain-ANSI orange; 256-colour code)
    else:
        c = "\033[31m"           # red
    return f"{c}{deg:.3f}\033[0m"


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eye-dir", default=EYE_DIR)
    p.add_argument("--eye-fit", default=EYE_FIT_PATH,
                   help="cached eye fit; loaded if present, else fitted and written")
    p.add_argument("--refit", action="store_true",
                   help="ignore --eye-fit even if present and refit from --eye-dir")
    p.add_argument("--tag", default="eyeG", help="checkpoint name")
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
    tag = a.tag
    dev = torch.device(a.device)
    os.makedirs(MODELS, exist_ok=True)

    # --- the eye ---------------------------------------------------------
    spec = fit_eye(a.eye_dir, cache=a.eye_fit, refit=a.refit)
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
        pdot, star, _ = learn.load_split(nm, n, a.duration, a.dt, s0)
        sp[nm] = (torch.as_tensor(pdot).to(dev),                  # p_dot, eq:input-vector
                  torch.as_tensor(star * scale).to(dev))          # (theta*, phi*), degrees
    (Pdot_tr, Star_tr), (Pdot_va, Star_va), (Pdot_te, Star_te) = \
        sp["train"], sp["val"], sp["test"]

    torch.manual_seed(0)
    model = CTRNNEyeG(eye, hidden=a.hidden, dt=a.dt).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    T = Pdot_tr.shape[1]
    sched = [max(60, int(T * f)) for f in (0.25, 0.5, 0.75, 1.0)]   # horizon curriculum

    def evaluate(Pdot, Star):
        model.eval()
        with torch.no_grad():
            errs = []
            for i in range(0, Pdot.shape[0], a.batch):
                u, _ = model(Pdot[i:i + a.batch])
                errs.append((u[..., :2] - Star[i:i + a.batch]).norm(dim=-1))
            return float(torch.cat(errs).mean())

    best, best_state = np.inf, None
    for ep in range(a.epochs):
        model.train()
        h = sched[min(3, int(4 * ep / max(a.epochs - 1, 1)))]
        perm = torch.randperm(Pdot_tr.shape[0], device=dev)
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            j = perm[i:i + a.batch]
            u, _ = model(Pdot_tr[j, :h])
            # eq:loss / eq:loss-again: (theta, phi) tracked, psi penalised to zero
            loss = ((u[..., :2] - Star_tr[j, :h]) ** 2).mean() \
                + a.lam_psi * (u[..., 2] ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss.detach()) * len(j)
        sch.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            e = evaluate(Pdot_va, Star_va)
            print(f"  ep {ep:3d}  horizon {h:3d}  train {tot / len(perm):.5f}  "
                  f"val |err| {_err_color(e)} deg")
            if e < best:
                best, best_state = e, {k: v.detach().clone()
                                       for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(Pdot_te, Star_te)
    with torch.no_grad():
        u, m = model(Pdot_te[:64])
        psi = float(u[..., 2].abs().mean())
        occ = float((m > 1e-3).float().mean())
    print(f"[done] test |err| {_err_color(test)} deg   mean |torsion| {psi:.2f} deg   "
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
    json.dump({"tag": tag, "n_act": eye.n_act,
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

# python test_eyeG.py --tag eyeG                  # models/eyeG_test.mp4    — corpus regimes
# python test_eyeG.py --tag eyeG --saccade        # models/eyeG_saccade.mp4 — six L/R saccade rates
