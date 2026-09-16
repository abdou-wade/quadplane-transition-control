"""PID controller and a 1-DOF mass simulation."""
##import libaraies 
import numpy as np
## PID CONTROLLER
class PID:

    def __init__(self, kp, ki, kd):   ## 'self' just directs the value where I want it to go.
        self.kp = kp              # proportional gain
        self.ki = ki              # integral gain
        self.kd = kd              # derivative gain

        self.integral = 0.0       # running sum for the integral term
        self.prev_error = None    # no previous error exists yet

    def step(self, error, dt):

        #  Proportional: focus on the present error
        p = self.kp * error

        #  Integral: accumulate error over time
        self.integral += error * dt
        i = self.ki * self.integral

        #  Derivative: rate of change of error
        # On the first call there is no previous error, so slope is undefined.
        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt
        d = self.kd * derivative

        # store this error so the NEXT call can use it as the previous one
        self.prev_error = error

        # the control output is the sum of the three terms
        return p + i + d


## Simulate a simple mass on a frictionless line
def simulate(controller, setpoint=1.0, mass=1.0, dt=0.01, t_final=10.0):
    """Run the PID against a mass on a frictionless line. Return t, x arrays."""
    n = int(t_final / dt)          # number of steps: 10 / 0.01 = 1000

    t = np.zeros(n)                # time at each step
    x = np.zeros(n)                # position at each step

    pos = 0.0                      # block starts at origin
    vel = 0.0                      # and at rest

    for k in range(n):
        error = setpoint - pos     # how far from target
        u = controller.step(error, dt)   # ask PID for a force

        accel = u / mass           # Newton: a = u/m
        vel = vel + accel * dt     # velocity update (first)
        pos = pos + vel * dt       # position update (uses new velocity)

        t[k] = k * dt              # record time
        x[k] = pos                 # record position

    return t, x
