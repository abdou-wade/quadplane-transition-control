"""PID controller and a 1-DOF mass simulation."""

# Import libraries
import numpy as np


## Setting up PID controller
class PID:

    def __init__(self, kp, ki, kd):
        self.kp = kp                # proportional gain
        self.ki = ki                # integral gain
        self.kd = kd                # derivative gain
        self.integral = 0.0         # running sum for the integral term
        self.prev_error = None      # no previous error exists yet

    def step(self, error, dt):
        # Proportional: react to the present error
        p = self.kp * error

        # Integral: accumulate error over time
        self.integral += error * dt
        i = self.ki * self.integral

        # Derivative: rate of change of error.
        # First call has no previous error, so slope is undefined -> use 0.
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt
        d = self.kd * derivative

        # store this error so the NEXT call can use it as the previous one
        self.prev_error = error

        return p + i + d


## Define the simulation: a mass on a frictionless line
def simulate(controller, setpoint=1.0, mass=1.0, dt=0.01, t_final=10.0):
    n = int(t_final / dt)           # number of steps: 10 / 0.01 = 1000

    t = np.zeros(n)                 # time at each step
    x = np.zeros(n)                 # position at each step

    pos = 0.0                       # block starts at origin
    vel = 0.0                       # and at rest

    for k in range(n):
        error = setpoint - pos      # how far from target
        u = controller.step(error, dt)   # ask PID for a force

        accel = u / mass            # Newton: a = u/m
        vel = vel + accel * dt      # velocity update (first)
        pos = pos + vel * dt        # position update (uses new velocity)

        t[k] = k * dt               # record time
        x[k] = pos                  # record position

    return t, x


## Running and plotting
def main():
    import matplotlib.pyplot as plt

    controller = PID(kp=9, ki=0, kd=6)   # zeta = 1, critically damped
    t, x = simulate(controller)

    # measure the actual overshoot to compare with the prediction
    peak = x.max()
    overshoot = 100 * (peak - 1.0) / 1.0
    print(f"peak = {peak:.4f}, overshoot = {overshoot:.2f}%")

    plt.plot(t, x)
    plt.axhline(1.0, color="red", linestyle="--")   # the target line
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("PID step response")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
## Quick lessons:
## P reacts to present error but oscillates forever alone
## D is a brake that adds damping, kills oscillation
## I cancels steady-state error but can cause overshoot / oscillation
## Saturation + windup: real actuators have limits. If the integral keeps
## growing while maxed out, it builds a debt that overshoots later.]
