from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallState:
    position_mm: np.ndarray
    velocity_mm_s: np.ndarray
    timestamp_s: float


class LowPassVelocityEstimator:
    """Smoothed board-frame velocity from successive position fixes.

    Two guards keep the velocity estimate from exploding on bad samples, which
    is critical because downstream logic (the stabilize state, the emergency
    brake) reacts to speed and a phantom 400 mm/s spike triggers a needless
    freak-out response:

    * ``min_dt_s`` - the camera (esp. MSMF) sometimes delivers frames in a
      burst, so two reads land ~0 ms apart. Dividing a sub-millimetre detection
      jitter by ~0 s manufactures a huge velocity. When dt is below this floor
      we treat the sample as a duplicate: keep the fresh POSITION but do not
      update the velocity or advance the reference, so the next well-spaced
      frame computes velocity over a real time interval.
    * ``max_speed_mm_s`` - even at a normal dt, an occasional tracker mislock
      jumps the position several mm in one frame. The measured velocity is
      clamped in magnitude before it enters the low-pass filter, so one bad fix
      cannot inject a spike. Set generously above any real ball speed.
    """

    def __init__(self, alpha: float = 0.35, min_dt_s: float = 0.006,
                 max_speed_mm_s: float = 250.0):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.min_dt_s = float(min_dt_s)
        self.max_speed_mm_s = float(max_speed_mm_s)
        self.previous_position: np.ndarray | None = None
        self.previous_timestamp_s: float | None = None
        self.velocity_mm_s = np.zeros(2, dtype=float)

    def reset(self) -> None:
        self.previous_position = None
        self.previous_timestamp_s = None
        self.velocity_mm_s = np.zeros(2, dtype=float)

    def update(self, position_mm: np.ndarray, timestamp_s: float) -> BallState:
        if self.previous_position is not None and self.previous_timestamp_s is not None:
            dt = timestamp_s - self.previous_timestamp_s
            if dt < self.min_dt_s:
                # Burst/duplicate frame: no real time elapsed. Serve the fresh
                # position with the last velocity, and do NOT advance the
                # reference - the next well-spaced frame will measure velocity
                # over the true interval instead of this near-zero one.
                return BallState(
                    position_mm=position_mm,
                    velocity_mm_s=self.velocity_mm_s.copy(),
                    timestamp_s=timestamp_s,
                )
            measured_velocity = (position_mm - self.previous_position) / dt
            speed = float(np.linalg.norm(measured_velocity))
            if speed > self.max_speed_mm_s:
                # Detection jump, not real motion: clamp magnitude before it
                # enters the filter so one mislocated fix cannot spike the speed.
                measured_velocity = measured_velocity * (self.max_speed_mm_s / speed)
            self.velocity_mm_s = (
                self.alpha * measured_velocity + (1.0 - self.alpha) * self.velocity_mm_s
            )

        self.previous_position = position_mm
        self.previous_timestamp_s = timestamp_s
        return BallState(
            position_mm=position_mm,
            velocity_mm_s=self.velocity_mm_s.copy(),
            timestamp_s=timestamp_s,
        )
