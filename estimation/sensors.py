"""
 fake sensors that lie the way real sensors lie.

Every sensor in this file follows one pattern (Concept 1):
    reading = truth + (known-cause error) + bias + noise

"""
#import libraries
import numpy as np

# Gravity in NED frame: 9.81 m/s^2 
GRAVITY_NED = np.array([0.0, 0.0, 9.81])

# Earth's magnetic field near Cincinnati, NED, in microtesla (approximate).
# NOAA magnetic field calculator.
MAG_FIELD_NED = np.array([20.9, -1.8, 49.7])


class _Sensor:
    

    def __init__(self, rng, size, bias_init_std, bias_walk_std):
        self.rng = rng                      # seeded generator -> reproducible runs
        self.size = size                    # 3 for vector sensors, None for scalar
        self.bias_walk_std = bias_walk_std  # how fast the bias wanders (drift)
        self.bias = rng.normal(0.0, bias_init_std, size)  # starting offset

    def _drift(self, dt):
        """Random walk: nudge the bias by a small random step (Concept 1: drift).

        Step size scales with sqrt(dt) so the drift rate is the same
        whether you sample at 100 Hz or 1000 Hz.
        """
        step = self.rng.normal(0.0, self.bias_walk_std * np.sqrt(dt), self.size)
        self.bias = self.bias + step

    def _noise(self, std):
        """Fresh random jitter, different on every reading."""
        return self.rng.normal(0.0, std, self.size)


class Gyro(_Sensor):


    def __init__(self, rng, noise_density=4.9e-5, bias_init_std=0.01, bias_walk_std=1e-4):
        super().__init__(rng, 3, bias_init_std, bias_walk_std)
        # Datasheets give noise as a "density" (rad/s per sqrt(Hz)).
        # ~0.0028 deg/s/sqrt(Hz), ICM-42688-P class IMU (verify on datasheet).
        self.noise_density = noise_density

    def measure(self, omega_body, dt):
        self._drift(dt)
        # Density -> per-reading std: faster sampling = noisier single readings,
        # but more of them to average. That's why we divide by sqrt(dt).
        noise = self._noise(self.noise_density / np.sqrt(dt))
        return omega_body + self.bias + noise


class Accel(_Sensor):


    def __init__(self, rng, noise_density=6.9e-4, bias_init_std=0.05, bias_walk_std=1e-3):
        super().__init__(rng, 3, bias_init_std, bias_walk_std)
        self.noise_density = noise_density  # ~70 micro-g/sqrt(Hz) (verify on datasheet)

    def measure(self, R, accel_ned, dt):
        self._drift(dt)
        # 1. What the springs feel (world frame), 2. rotate into the drone's view.
        specific_force = R.T @ (accel_ned - GRAVITY_NED)
        noise = self._noise(self.noise_density / np.sqrt(dt))
        return specific_force + self.bias + noise


class Mag(_Sensor):


    def __init__(self, rng, noise_std=0.3, hard_iron_std=2.0,
                 current_coeff=(0.15, 0.05, 0.20)):
        # Hard iron is constant, so bias_walk_std = 0 -> no drift.
        super().__init__(rng, 3, hard_iron_std, 0.0)
        self.noise_std = noise_std                       # microtesla per reading
        self.current_coeff = np.asarray(current_coeff)   # microtesla per amp

    def measure(self, R, motor_current):
        earth_field = R.T @ MAG_FIELD_NED                # Concept 2: rotate into body
        motor_error = self.current_coeff * motor_current  # more amps -> bigger lie
        return earth_field + motor_error + self.bias + self._noise(self.noise_std)


class Baro(_Sensor):


    def __init__(self, rng, noise_std=0.1, bias_init_std=0.5, bias_walk_std=0.01,
                 thrust_coeff=0.02):
        super().__init__(rng, None, bias_init_std, bias_walk_std)  # scalar sensor
        self.noise_std = noise_std        # m per reading
        self.thrust_coeff = thrust_coeff  # m of error per newton of lift thrust

    def measure(self, altitude, lift_thrust, dt):
        self._drift(dt)
        wash_error = self.thrust_coeff * lift_thrust     # more thrust -> bigger lie
        return altitude + wash_error + self.bias + self._noise(self.noise_std)


# Quick demo: -m estimation.sensors

if __name__ == "__main__":
    rng = np.random.default_rng(42)   # same seed -> same output every run
    dt = 0.01                         # 100 Hz
    level = np.eye(3)                 # R for a level drone facing north

    # Gyro: sit still for 60 s and integrate 
    gyro = Gyro(rng)
    angle = np.zeros(3)
    for _ in range(6000):
        angle += gyro.measure(np.zeros(3), dt) * dt
    print("Gyro  | still 60 s, believed angle (deg):", np.degrees(angle).round(2))

    # Accel: rest reads ~[0, 0, -9.81]; free fall reads ~[0, 0, 0] (Concept 3).
    accel = Accel(rng)
    print("Accel | at rest   :", accel.measure(level, np.zeros(3), dt).round(2))
    print("Accel | free fall :", accel.measure(level, GRAVITY_NED, dt).round(2))

    # Mag: error should grow with motor current (Concept 5).
    mag = Mag(rng)
    for amps in (0, 20, 40):
        err = np.linalg.norm(mag.measure(level, amps) - MAG_FIELD_NED)
        print(f"Mag   | {amps:>2} A -> error {err:5.2f} uT")

    # Baro: true altitude 10 m; error should grow with thrust (Concept 5).
    baro = Baro(rng)
    for thrust in (0, 30, 60):
        reading = baro.measure(10.0, thrust, dt)
        print(f"Baro  | {thrust:>2} N -> reads {reading:5.2f} m (true 10.00 m)")



## NOTES
## I learned that  a controller only knows what its sensors tell it.
##   Real sensors never give the truth, so a sim that uses perfect state
##   is lying to you. This file makes the sim lie the way reality does which is the goal .


## THE ONE PATTERN: reading = truth + known-cause error + bias + noise
##   noise -> random, zero-mean     -> fixed by AVERAGING 
##   bias  -> constant offset       -> fixed by CALIBRATION
##   drift -> bias that wanders     -> fixed by continuous ESTIMATION (Kalman filters)

## FRAMES: world = NED (gravity [0,0,+9.81]); body frame = nose/right wing/belly.
##   Sensors are bolted to the body -> they see world vectors.
##   Tilt is detected by which body axis gravity appears along.
## GYRO: measures rotation rate. Angle = integral of rate.
##   Bias integrates into angle error that grows forever
##   (0.5 deg/s bias x 120 s = 60 deg). Good short-term, bad long-term.
##
## ACCEL: measures proper acceleration = accel - gravity.
##   Gyro + accel cover each other's weakness -> sensor fusion (L2).
##

## BARO: pressure -> altitude. Rotor wash corrupts it (error grows with
##   thrust); weather makes it drift. Hardware (foam, placement) reduces
##   it; software handles the rest.
##
## KNOWN-CAUSE ERRORS: current and thrust are measurable -> you know WHEN a
##   sensor lies hardest -> trust it less then (regime-dependent R, L4).
##   Matters most in transition: thrust changes most exactly then.
##
## DESIGN CHOICES:
##   - Seeded rng passed in -> every run reproducible (Monte Carlo, L8).
##   - Noise given as datasheet DENSITY; per-reading std = density/sqrt(dt).
##   - Bias walk step = walk_std*sqrt(dt) -> drift independent of sample rate.
##   - Shared _Sensor base class -> bias/noise logic written once.
##
## VERIFIED (demo): still gyro drifts 13-34 deg in 60 s | accel -9.8 at
##   rest, 0 in free fall | mag error 1.7->11.2 uT for 0->40 A |
##   baro 9.34->10.65 m for 0->60 N at true 10 m.
