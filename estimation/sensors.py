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