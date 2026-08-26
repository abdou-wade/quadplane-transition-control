"""
PID control on a 1-DOF mass.

Level 0.1 of the curriculum. The plant is a mass on a frictionless line:

    m * xddot = u

Nothing aerospace about it. Every controller you write this semester --
attitude, altitude, the transition blend -- is this loop, nested.

Run directly to produce results/week01_pid_step.png:

    python -m control.pid
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PID:
    """A PID controller with output saturation and optional anti-windup.

    kp, ki, kd    gains
    u_min, u_max  actuator limits (the real world always has these)
    anti_windup   if True, stop integrating while the output is saturated
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    u_min: float = -np.inf
    u_max: float = np.inf
    anti_windup: bool = True

    integral: float = field(default=0.0, init=False)
    prev_error: float = field(default=None, init=False)

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def step(self, error: float, dt: float) -> float:
        """Advance one timestep. Returns the saturated control output."""
        if dt <= 0:
            raise ValueError("dt must be positive")

        # Proportional: push harder the further off you are.
        p_term = self.kp * error

        # Derivative: react to how fast the error is changing. A brake.
        # First call has no history, so derivative is zero.
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Integral: accumulate past error to kill steady-state offset.
        # Tentatively integrate, then undo it if we saturate (anti-windup).
        self.integral += error * dt
        i_term = self.ki * self.integral

        u_raw = p_term + i_term + d_term
        u = float(np.clip(u_raw, self.u_min, self.u_max))

        # Anti-windup: if the actuator is pinned, integrating further does
        # nothing but store a debt that must be unwound later as overshoot.
        if self.anti_windup and u != u_raw:
            self.integral -= error * dt

        self.prev_error = error
        return u


def simulate(controller, setpoint=1.0, mass=1.0, dt=0.01, t_final=10.0,
             disturbance=0.0, x0=0.0, v0=0.0):
    """Closed-loop step response of a mass on a frictionless line.

    Returns (t, x, u) as numpy arrays.

    Integration is semi-implicit Euler: update velocity first, then position
    with the new velocity. Cheap and stable enough for a double integrator.
    """
    controller.reset()
    n = int(round(t_final / dt))

    t = np.zeros(n)
    x = np.zeros(n)
    u_log = np.zeros(n)

    pos, vel = x0, v0
    for k in range(n):
        error = setpoint - pos
        u = controller.step(error, dt)

        accel = (u + disturbance) / mass
        vel += accel * dt
        pos += vel * dt

        t[k] = k * dt
        x[k] = pos
        u_log[k] = u

    return t, x, u_log


def overshoot_percent(x, setpoint):
    """Peak overshoot as a percentage of the setpoint."""
    return 100.0 * (np.max(x) - setpoint) / setpoint


def settling_time(t, x, setpoint, tol=0.02):
    """Time after which the response stays within tol of setpoint.

    Returns np.inf if it never settles.
    """
    outside = np.abs(x - setpoint) > tol * abs(setpoint)
    if not np.any(outside):
        return 0.0
    last = np.where(outside)[0][-1]
    if last == len(t) - 1:
        return np.inf
    return t[last + 1]


def main():
    import os

    import matplotlib
    matplotlib.use("Agg")  # no display needed; write straight to file
    import matplotlib.pyplot as plt

    setpoint = 1.0
    cases = [
        ("P only  (kp=5)", PID(kp=5.0)),
        ("PD      (kp=5, kd=4)", PID(kp=5.0, kd=4.0)),
        ("PID     (kp=5, ki=3, kd=4)", PID(kp=5.0, ki=3.0, kd=4.0)),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    print(f"{'case':<28} {'overshoot':>10} {'settling':>10}")
    print("-" * 50)

    for label, ctrl in cases:
        t, x, u = simulate(ctrl, setpoint=setpoint)
        os_pct = overshoot_percent(x, setpoint)
        ts = settling_time(t, x, setpoint)
        ts_str = "never" if np.isinf(ts) else f"{ts:.2f} s"
        print(f"{label:<28} {os_pct:>9.1f}% {ts_str:>10}")

        ax1.plot(t, x, label=label)
        ax2.plot(t, u, label=label)

    ax1.axhline(setpoint, color="k", ls="--", lw=0.8, label="setpoint")
    ax1.set_ylabel("position [m]")
    ax1.set_title("Step response, mass on a frictionless line")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_ylabel("control u [N]")
    ax2.set_xlabel("time [s]")
    ax2.grid(alpha=0.3)

    os.makedirs("results", exist_ok=True)
    out = "results/week01_pid_step.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
