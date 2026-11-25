"""Sensor replay simulator.

The full 20 Hz recording is the ground truth of the world.  The simulator is
the only component that can touch it, and it only hands out samples that the
current sensing policy has paid for.  That is what makes the three strategies
comparable: identical data, identical models, different *access* to data.

Accounting per run:
* ``samples``      - IMU samples actually acquired
* ``sensor_on_s``  - IMU on-time (energy proxy input)
* ``energy_mj``    - relative energy proxy from the power model
* ``gps_on_s``     - GPS on-time
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_POWER, FS, LATENT_SPEED, PowerModel
from ..data.sessions import Session


@dataclass
class SensingCounters:
    samples: int = 0
    sensor_on_s: float = 0.0
    energy_mj: float = 0.0
    gps_on_s: float = 0.0
    gps_reads: int = 0


class SensorSimulator:
    def __init__(self, session: Session, power: PowerModel | None = None, seed: int = 0):
        self.session = session
        self.power = power or DEFAULT_POWER
        self.acc = session.acc
        self.gyro = session.gyro
        self.n = len(session.acc)
        self.acquired = np.zeros(self.n, dtype=bool)
        self.mode = "off"                      # "active" | "burst" | "off"
        self.burst_s = 1.0
        self.period_s = 4.0
        self.last_t = 0.0
        self.counters = SensingCounters()
        self.gps_on = False
        self.gps_until = 0.0
        self.rng = np.random.default_rng(seed)

    # ----------------------------------------------------------- control ----
    def configure(self, mode: str, burst_s: float = 1.0, period_s: float = 4.0) -> None:
        self.mode = mode
        self.burst_s = burst_s
        self.period_s = period_s

    def gps_activate(self, t: float, duty_s: float) -> None:
        self.gps_on = True
        self.gps_until = max(self.gps_until, t + duty_s)

    # --------------------------------------------------------- simulation ----
    def _in_burst(self, ts: float) -> bool:
        """True for samples in [t - burst_s, t) where t is the next period edge."""
        if ts <= 0:
            return False
        r = ts % self.period_s
        if r <= 1e-9:
            return False
        return (self.period_s - r) <= self.burst_s + 1e-9

    def advance_to(self, t: float) -> None:
        dt = t - self.last_t
        if dt <= 0:
            return
        i0 = int(round(self.last_t * FS))
        i1 = min(int(round(t * FS)), self.n)
        idx = np.arange(i0, i1)
        if len(idx):
            ts = idx / FS
            if self.mode == "active":
                take = np.ones(len(idx), dtype=bool)
            elif self.mode == "burst":
                take = np.array([self._in_burst(x) for x in ts])
            else:
                take = np.zeros(len(idx), dtype=bool)
            if take.any():
                self.acquired[idx[take]] = True
                n_new = int(take.sum())
                self.counters.samples += n_new
                on_s = n_new / FS
                self.counters.sensor_on_s += on_s
                self.counters.energy_mj += self.power.imu_on_mw * on_s
        # sleep current always present
        self.counters.energy_mj += self.power.imu_sleep_mw * dt
        if self.gps_on:
            gps_dt = max(0.0, min(t, self.gps_until) - self.last_t)
            self.counters.gps_on_s += gps_dt
            self.counters.energy_mj += self.power.gps_mw * gps_dt
            self.counters.gps_reads += 1
            if t >= self.gps_until:
                self.gps_on = False
        self.last_t = t

    # ------------------------------------------------------------ readout ----
    def window(self, t_end: float, seconds: float = 3.0):
        """Return a fully-acquired contiguous window, else ``None``."""
        i1 = int(round(t_end * FS))
        i0 = i1 - int(round(seconds * FS))
        if i0 < 0 or i1 > self.n:
            return None
        if not self.acquired[i0:i1].all():
            return None
        return self.acc[i0:i1], self.gyro[i0:i1]

    def burst(self, t_end: float, burst_s: float | None = None):
        i1 = int(round(t_end * FS))
        i0 = i1 - int(round((burst_s or self.burst_s) * FS))
        if i0 < 0 or i1 > self.n:
            return None
        if not self.acquired[i0:i1].all():
            return None
        return self.acc[i0:i1]

    def burst_pair(self, t_end: float, burst_s: float | None = None):
        """Like :meth:`burst` but returns both accelerometer and gyroscope."""
        i1 = int(round(t_end * FS))
        i0 = i1 - int(round((burst_s or self.burst_s) * FS))
        if i0 < 0 or i1 > self.n:
            return None
        if not self.acquired[i0:i1].all():
            return None
        return self.acc[i0:i1], self.gyro[i0:i1]

    def gps_context(self, t: float) -> dict:
        """Synthetic GPS fix (WISDM has no GPS; see REPORT.md)."""
        label = self.session.label_at(t)
        speed = LATENT_SPEED[label] * float(1.0 + 0.15 * self.rng.standard_normal())
        speed = max(0.0, speed)
        return {
            "speed_mps": round(speed, 2),
            "moving": bool(speed > 0.5),
            "source": "simulated_gps",
        }
