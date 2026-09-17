"""PID controller and a 1-DOF mass simulation."""

import numpy as np


class PID:

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = None

    def step(self, error, dt):
        p = self.kp * error

        self.integral += error * dt
        i = self.ki * self.integral

        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt
        d = self.kd * derivative

        self.prev_error = error
        return p + i + d


def simulate(controller, setpoint=1.0, mass=1.0, dt=0.01, t_final=10.0):
    n = int(t_final / dt)

    t = np.zeros(n)
    x = np.zeros(n)

    pos = 0.0
    vel = 0.0

    for k in range(n):
        error = setpoint - pos
        u = controller.step(error, dt)

        accel = u / mass
        vel = vel + accel * dt
        pos = pos + vel * dt

        t[k] = k * dt
        x[k] = pos

    return t, x


def main():
    import matplotlib.pyplot as plt

    controller = PID(kp=5, ki=0, kd=4)
    t, x = simulate(controller)

    plt.plot(t, x)
    plt.axhline(1.0, color="red", linestyle="--")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("PID step response")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
