"""The exponential-Euler step of `ConductanceSynapses`, on a four-neuron network.

WHAT IS BEING TESTED IS THE DISCRETISATION, NOT THE MODEL, so the fixture holds
the presynaptic activity FIXED rather than letting the network drive itself: the
step is then exactly linear in the postsynaptic voltage and has a closed-form
fixed point, which is what makes both claims checkable rather than merely
plausible.

    v(t+dt) = v_inf + (v(t) - v_inf) e^(-z),
    z = (1 + G) dt / tau,   v_inf = (bias + S + x) / (1 + G)

with G the total synaptic conductance onto the neuron in units of its own leak,
S that conductance weighted by each edge's reversal, and x the external input.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flyvis")

from types import SimpleNamespace  # noqa: E402

from flyvis_conductance_optical_flow.dynamics import ConductanceSynapses  # noqa: E402

N_NODES = 4
DT = 0.02  # seconds, flyvis's own integration step for the optic-flow task


def _rig(conductance, tau, drive=1.0, bias=0.5, reversal=4.0):
    """A ring of four neurons, each receiving one edge from a FIXED driver.

    Args:
        conductance: g of every edge, in units of the leak conductance. The total
            onto each neuron is g * relu(drive), which is what sets the stiffness.
        tau: membrane time constant of every neuron, in seconds.
        drive: the presynaptic voltage, held constant through the run.
        bias: resting potential of every neuron.
        reversal: the reversal potential every edge drives toward.

    Returns (dynamics, params, target_sum, state_factory).
    """
    dyn = ConductanceSynapses()
    src = torch.arange(N_NODES)
    dst = (src + 1) % N_NODES

    params = SimpleNamespace(
        edges=SimpleNamespace(
            conductance=torch.full((N_NODES,), float(conductance)),
            sign=torch.ones(N_NODES),
        ),
        nodes=SimpleNamespace(
            bias=torch.full((N_NODES,), float(bias)),
            time_const=torch.full((N_NODES,), float(tau)),
        ),
        targets=SimpleNamespace(
            E_exc=torch.full((N_NODES,), float(reversal)),
            E_inh=torch.full((N_NODES,), -float(reversal)),
        ),
    )

    def target_sum(x):
        out = torch.zeros((*x.shape[:-1], N_NODES))
        return out.scatter_add_(-1, dst.expand(*x.shape), x)

    def state_of(v):
        return SimpleNamespace(
            nodes=SimpleNamespace(activity=v),
            sources=SimpleNamespace(activity=torch.full((N_NODES,), float(drive))),
            targets=SimpleNamespace(activity=v[dst]),
        )

    return dyn, params, target_sum, state_of


def _velocity(dyn, params, target_sum, state, x_t, dt):
    vel = SimpleNamespace(nodes=SimpleNamespace(activity=None))
    dyn.write_state_velocity(vel, state, params, target_sum, x_t, dt)
    return vel.nodes.activity


def _forward_euler_velocity(params, target_sum, state, x_t, dt):
    """The step this replaced, kept here as the thing to agree with and to beat."""
    driving = params.targets.E_exc - state.targets.activity
    chem = target_sum(
        params.edges.conductance * torch.relu(state.sources.activity) * driving
    )
    tau = torch.clamp(params.nodes.time_const, min=dt)
    return (-state.nodes.activity + params.nodes.bias + chem + x_t) / tau


def test_agrees_with_forward_euler_in_the_small_step_limit():
    """As dt -> 0 the two velocities are the same equation, so they must converge.

    Checked at dt = 20 microseconds, a thousandth of the real step, where the
    amplification factor (dt/tau)(1 + G) is 0.0024 and forward Euler is still an
    accurate integrator -- so a disagreement there is a wrong ODE, not a different
    discretisation of the right one.
    """
    dyn, params, target_sum, state_of = _rig(conductance=1.0, tau=0.05)
    v = torch.tensor([0.4, 0.5, 0.6, 0.7])
    x_t = torch.full((N_NODES,), 0.1)
    small = 2e-5

    got = _velocity(dyn, params, target_sum, state_of(v), x_t, small)
    want = _forward_euler_velocity(params, target_sum, state_of(v), x_t, small)
    assert torch.allclose(got, want, rtol=2e-3), (got, want)


def test_stable_where_forward_euler_diverges():
    """Total conductance 8x the leak: forward Euler blows up, this converges.

    The amplification factor is (dt/tau)(1 + G) = 3.6 against the contraction
    bound of 2, which is the regime flow/2000/000 wandered into at iteration
    16,368. Both integrators are run for 400 steps of the SAME map; the assertion
    is that one leaves the voltage range entirely and the other settles on the
    analytic fixed point (bias + S + x)/(1 + G).
    """
    g, tau = 8.0, 0.05
    dyn, params, target_sum, state_of = _rig(conductance=g, tau=tau)
    x_t = torch.zeros(N_NODES)
    factor = (DT / tau) * (1.0 + g)
    assert factor > 2.0, f"fixture is not stiff: factor {factor}"

    v_inf = (0.5 + g * 4.0) / (1.0 + g)

    v = torch.full((N_NODES,), 0.5)
    for _ in range(400):
        v = v + DT * _velocity(dyn, params, target_sum, state_of(v), x_t, DT)
    assert torch.isfinite(v).all()
    assert torch.allclose(v, torch.full((N_NODES,), v_inf), atol=1e-4), v

    u = torch.full((N_NODES,), 0.5)
    for _ in range(400):
        u = u + DT * _forward_euler_velocity(
            params, target_sum, state_of(u), x_t, DT
        )
    assert not torch.isfinite(u).all() or u.abs().max() > 1e3, (
        f"forward Euler stayed bounded at factor {factor}; the fixture no longer "
        "reproduces the instability this step exists to remove"
    )


def test_fixed_point_is_unchanged_by_the_new_step():
    """Both integrators, where both are stable, settle on the SAME voltage.

    The point of an exponential-Euler step is that it changes only how the
    trajectory is approximated between steps, never where it ends up. With a
    conductance of 0.2 the factor is 0.48, well inside forward Euler's bound, so
    the two can be run side by side and compared.
    """
    g, tau = 0.2, 0.05
    dyn, params, target_sum, state_of = _rig(conductance=g, tau=tau)
    x_t = torch.full((N_NODES,), 0.05)

    v = torch.full((N_NODES,), 0.5)
    u = torch.full((N_NODES,), 0.5)
    for _ in range(2000):
        v = v + DT * _velocity(dyn, params, target_sum, state_of(v), x_t, DT)
        u = u + DT * _forward_euler_velocity(
            params, target_sum, state_of(u), x_t, DT
        )
    assert torch.allclose(v, u, atol=1e-5), (v, u)
