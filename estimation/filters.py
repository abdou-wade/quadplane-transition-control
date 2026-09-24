"""
estimation/filters.py smoothing filters, and they create : lag.

"""
from collections import deque

import numpy as np


class MovingAverage:
    """Average of the last N readings.

    Noise shrinks by sqrt(N).  Lag = (N - 1) / 2 samples
    (the average sits in the MIDDLE of the window, so it describes the past).
    """

    def __init__(self, n):
        self.n = n
        self.window = deque(maxlen=n)   # old readings fall off automatically

    def update(self, reading):
        self.window.append(reading)
        return sum(self.window) / len(self.window)

    def predicted_lag(self, dt):
        """Delay in seconds, from theory."""
        return (self.n - 1) / 2 * dt


class LowPass:
    """First-order low-pass:  new = old + alpha * (reading - old)

    Each step, move a fraction alpha toward the new reading.
        small alpha -> heavy smoothing, big lag
        big alpha   -> light smoothing, small lag
    Lag = (1 - alpha) / alpha samples.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None               # no memory until the first reading

    def update(self, reading):
        if self.value is None:          # first reading: start there, not at 0
            self.value = reading
        else:
            self.value = self.value + self.alpha * (reading - self.value)
        return self.value

    def predicted_lag(self, dt):
        """Delay in seconds, from theory."""
        return (1 - self.alpha) / self.alpha * dt


# Demo: run with  python -m estimation.filters
# Noisy baro during hover -> climb -> hover. Light vs heavy smoothing.
# Prints noise vs lag for each filter 
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from estimation.sensors import Baro

    rng = np.random.default_rng(42)
    dt = 0.02                                   # baro at 50 Hz
    t = np.arange(0.0, 12.0, dt)

    # True altitude: hover at 10 m, climb at 2 m/s from t=3 to t=8, hover at 20 m.
    climb_rate = 2.0
    true_alt = np.clip(10.0 + climb_rate * (t - 3.0), 10.0, 20.0)

    baro = Baro(rng)
    raw = np.array([baro.measure(h, 0.0, dt) for h in true_alt])

    filters = {
        "MA  N=5     (light)": MovingAverage(5),
        "MA  N=25    (heavy)": MovingAverage(25),
        "LPF a=0.30  (light)": LowPass(0.30),
        "LPF a=0.05  (heavy)": LowPass(0.05),
    }
    outputs = {name: np.array([f.update(r) for r in raw]) for name, f in filters.items()}

    # Measure noise: std during the first hover (after filters warm up).
    hover = (t > 1.0) & (t < 3.0)
    # Measure lag: during steady climb, lag = (raw - filtered) / climb rate.
    # (Comparing to raw, not truth, so the baro bias cancels out.)
    climb = (t > 5.0) & (t < 8.0)

    print(f"{'filter':<22}{'noise std (m)':>14}{'lag theory (s)':>16}{'lag measured (s)':>18}")
    print(f"{'raw baro':<22}{raw[hover].std():>14.3f}{0.0:>16.3f}{0.0:>18.3f}")
    for name, f in filters.items():
        y = outputs[name]
        lag = np.mean(raw[climb] - y[climb]) / climb_rate
        print(f"{name:<22}{y[hover].std():>14.3f}{f.predicted_lag(dt):>16.3f}{lag:>18.3f}")

    # Plot: one panel per filter type.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, kind in zip(axes, ("MA", "LPF")):
        ax.plot(t, raw, color="lightgray", label="raw baro")
        ax.plot(t, true_alt, "k--", label="true altitude")
        for name, y in outputs.items():
            if name.startswith(kind):
                ax.plot(t, y, label=name.strip())
        ax.set_ylabel("altitude (m)")
        ax.set_title("Moving average" if kind == "MA" else "Low-pass filter")
        ax.grid(True)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Smoothing vs lag: cleaner signal = later signal")
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    fig.savefig("results/l03_filter_lag.png", dpi=120)
    print("saved results/l03_filter_lag.png")


## NOTES
## So sensors are noisy (sensors.py). Smoothing removes
##   noise, but smoothing is built from OLD readings, so the output always
##   describes the past. Every filter trades noise for delay. 
##
## MOVING AVERAGE: mean of the last N readings.
##   noise / sqrt(N)          lag = (N-1)/2 samples
##   ex: 50 Hz, N=25 -> 0.5 s window -> ~0.24 s behind.
##
## LOW-PASS (first order): new = old + alpha*(reading - old)
##   small alpha -> smooth + slow     big alpha -> jumpy + fast
##   lag = (1-alpha)/alpha samples.  Needs 1 stored number, not N -> the
##   default on embedded hardware (PX4 uses low-pass on gyro/accel).
##
## WHY LAG IS DANGEROUS: a controller acting on old data overcorrects
##   (the slow-shower effect) -> oscillation. Delay T at crossover
##   frequency w costs w*T radians of phase margin.
##   ex: 20 ms delay at 10 rad/s -> 0.2 rad = 11.5 deg of margin gone.
##
## THE REAL LESSON: you cannot beat the tradeoff with ONE sensor.
##   The fix is a second sensor that is fast where the first is slow:
##   gyro (fast) + accel (slow but no drift) -> complementary filter (L2)
##   -> Kalman filter (L4), which picks the blend optimally.
##
## DESIGN CHOICES:
##   - Streaming update() one sample at a time -> same shape as flight code.
##   - LowPass starts at the first reading, not 0 -> no fake startup ramp.
##   - Lag measured vs RAW (not truth) during steady climb -> baro bias cancels.
##   - predicted_lag() from theory, printed next to measured -> proves model.