"""Tests for control.pid.

Five tests, each pinning down one behaviour you should be able to predict
before you run it. If a test fails after you change pid.py, the test is
telling you the physics changed -- read it before you edit it.
"""

import numpy as np

from control.pid import PID, simulate, overshoot_percent, settling_time


def test_p_only_oscillates_forever():
    """A double integrator has no damping. P alone cannot settle it.

    This is the whole reason D exists.
    """
    t, x, _ = simulate(PID(kp=5.0), setpoint=1.0, t_final=20.0)
    ts = settling_time(t, x, 1.0)
    assert np.isinf(ts), "P-only should never settle on a double integrator"
    assert overshoot_percent(x, 1.0) > 50.0


def test_derivative_settles_the_response():
    """Adding D damps the oscillation and the response settles."""
    t, x, _ = simulate(PID(kp=5.0, kd=4.0), setpoint=1.0, t_final=20.0)
    ts = settling_time(t, x, 1.0)
    assert np.isfinite(ts), "PD should settle"
    assert ts < 5.0


def test_integral_kills_steady_state_error():
    """With a constant disturbance, PD leaves an offset. PID does not."""
    _, x_pd, _ = simulate(PID(kp=5.0, kd=4.0), setpoint=1.0,
                          disturbance=-2.0, t_final=30.0)
    _, x_pid, _ = simulate(PID(kp=5.0, ki=3.0, kd=4.0), setpoint=1.0,
                           disturbance=-2.0, t_final=30.0)

    err_pd = abs(1.0 - x_pd[-1])
    err_pid = abs(1.0 - x_pid[-1])

    assert err_pd > 0.1, "PD should leave a steady-state offset"
    assert err_pid < 0.01, "PID should drive the offset to zero"


def test_output_respects_saturation():
    """The controller must never command beyond the actuator limits."""
    ctrl = PID(kp=100.0, kd=10.0, u_min=-2.0, u_max=2.0)
    _, _, u = simulate(ctrl, setpoint=1.0, t_final=5.0)
    assert np.all(u <= 2.0 + 1e-9)
    assert np.all(u >= -2.0 - 1e-9)


def test_anti_windup_reduces_overshoot():
    """Integrating while saturated stores a debt paid back as overshoot."""
    common = dict(kp=5.0, ki=8.0, kd=4.0, u_min=-2.0, u_max=2.0)

    _, x_on, _ = simulate(PID(anti_windup=True, **common),
                          setpoint=1.0, t_final=20.0)
    _, x_off, _ = simulate(PID(anti_windup=False, **common),
                           setpoint=1.0, t_final=20.0)

    assert overshoot_percent(x_on, 1.0) < overshoot_percent(x_off, 1.0)
