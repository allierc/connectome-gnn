"""How the model's own ODE is advanced by one observed frame.

ONE DISPATCHER, TWO INTEGRATORS, EVERY CALLER. The step is taken in three places
-- the tester (`data_test_gnn`), the template-fit rollout (which goes through the
tester, so it inherits this for free) and the trainer's sub-step training branch.
They must agree, or a run's training loss and its reported rollout describe
different dynamical systems.

WHAT THIS FIXES. The conductance generator is integrated with exponential Euler:
exact at frozen coefficients, a contraction for any synaptic conductance. The
model is integrated with forward Euler at the same `delta_t`, which contracts
only while

    z = (delta_t / tau_i) * (1 + G_i) < 2,        G_i = sum_j W_ij relu(v_j)

and the generating network was measured at z = 4.4. A GNN that learned the
generator EXACTLY would therefore diverge in rollout, and the tester's
clamp(+-100) was hiding it. Splitting the frame into M sub-steps of
h = delta_t / M divides z by M: 4.42 at M=1, 2.21 at M=2 (still diverges), 1.47
at M=3, 0.88 at M=5.

WHY SUB-STEPPING RATHER THAN A BETTER TARGET. `f_theta` predicts an
instantaneous derivative, which carries no step size, so it can be integrated at
any h without retraining and without touching the template readout. Training on
the one-step finite difference instead would bake `delta_t` into the target and
inflate the recovered tau by z / (1 - e^-z) -- a factor of 4.5 at z = 4.4.
"""

def euler_step(x, dvdt, dt):
    """One forward-Euler step: v <- v + dt * dv/dt.

    THE BASE PATH, and `multi_substep` at M=1 is not routed here by accident --
    `integrate_frame` dispatches to it explicitly, so a run with
    n_rollout_substeps=1 takes bit-for-bit the same arithmetic it always did and
    no sub-step machinery is on the path at all.
    """
    x.voltage = x.voltage + dt * dvdt.squeeze(-1)
    return x


def multi_substep(x, forward, dt, n_substeps):
    """Advance one observed frame as `n_substeps` forward-Euler steps of dt/M.

    THE MESSAGE IS RECOMPUTED EVERY SUB-STEP -- that is the whole point. The
    shunting term -v_i * sum_j g_ij relu(v_j) is what makes the system stiff, so
    freezing the message across the frame would reintroduce exactly the
    instability this exists to remove. The STIMULUS is not recomputed: `delta_t`
    is the observed cadence, so it is constant within the frame, which is the
    same convention the generator's own sub-step loop used.

    Args:
        x: NeuronState, mutated in place.
        forward: callable `x -> dv/dt`, shape (N, 1). Re-evaluated M times.
        dt: the OBSERVED frame interval, not the sub-step.
        n_substeps: M >= 2.

    Returns:
        dv/dt at the LAST sub-step -- the derivative at the end of the frame, not
        its average across it. Callers that log `y` should read it as such.
    """
    h = dt / n_substeps
    dvdt = None
    for _ in range(int(n_substeps)):
        dvdt = forward(x)
        euler_step(x, dvdt, h)
    return dvdt


def integrate_frame(x, forward, dt, method="euler", n_substeps=5, dvdt=None):
    """Advance `x` by one observed frame, dispatching on `method`.

    The choice is explicit rather than inferred from `n_substeps`, so a config
    reading `integration_method: euler` takes the Euler path whatever
    `n_rollout_substeps` happens to say, and a run cannot change integrator by
    someone editing a count.

    `dvdt` is the derivative the caller has ALREADY computed for this frame. The
    Euler path uses it as-is and never calls `forward`, so a caller that needs
    the derivative for its own logging does not pay for a second forward pass.
    """
    if method == "euler":
        if dvdt is None:
            dvdt = forward(x)
        euler_step(x, dvdt, dt)
        return dvdt
    if method == "multi_substeps":
        m = int(n_substeps)
        if m < 1:
            raise ValueError(f"n_rollout_substeps must be >= 1, got {n_substeps}")
        if m == 1:
            # Not silently an error -- M=1 IS Euler -- but the config is then
            # saying two different things, so take the Euler path and let the
            # stability line in the log report z at M=1.
            return integrate_frame(x, forward, dt, "euler", dvdt=dvdt)
        return multi_substep(x, forward, dt, m)
    raise ValueError(
        f"unknown integration_method {method!r}; expected 'euler' or "
        f"'multi_substeps'")


def substep_report(method, n_substeps, delta_t, tau=0.019, a_max=4.2):
    """One line naming the stability the chosen integrator buys, for the run log.

    `a_max` is 1 + G_i at the stiffest neuron measured on the conductance
    generator (G_i = 3.2); `tau` its time constant. Forward Euler contracts
    while z < 2.
    """
    m = max(int(n_substeps or 1), 1) if method == "multi_substeps" else 1
    z = a_max * (delta_t / m) / tau
    verdict = "contracts" if z < 2 else "DIVERGES"
    return (f"integration: {method}, {m} x {1000 * delta_t / m:.1f} ms per "
            f"{1000 * delta_t:.0f} ms frame, z = {z:.2f} at the stiffest neuron "
            f"(1 + G_i = {a_max}, tau = {1000 * tau:.0f} ms) -- {verdict}")
