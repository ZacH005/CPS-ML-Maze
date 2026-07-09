from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PathFollowerConfig:
    kp: float
    kd: float
    max_command: float
    # Integral action: accumulates while error persists, which walks the
    # command up until the ball un-sticks; also absorbs a non-level neutral.
    ki: float = 0.0
    integral_limit: float = 0.25  # max |command| contribution of the I term
    # Stiction kick: below some tilt the ball simply does not move (static
    # friction). If the ball is stalled away from the target, scale the
    # command up to at least this magnitude so it always breaks free.
    # Set from axis_check observations (~the amplitude that reliably moved
    # the ball). 0 disables.
    stall_kick: float = 0.0
    stall_speed_mm_s: float = 8.0   # "stalled" when slower than this
    stall_dist_mm: float = 8.0      # ...and further than this from target


@dataclass
class VelocityFollowerConfig:
    """Velocity-tracking path follower.

    Board tilt commands the ball's ACCELERATION, so controlling velocity
    error with tilt is the natural (first-order) loop: it brakes
    automatically when the ball carries too much speed into a corner,
    which position-PD cannot do.
    """
    v_max_mm_s: float = 45.0       # cruise speed on straights
    min_speed_frac: float = 0.25   # never slow below this fraction of v_max
    corner_slow_deg: float = 110.0 # heading change over the span that maps to full slowdown
    k_lat: float = 2.5             # lateral velocity per mm of cross-track error (1/s)
    lat_v_max_mm_s: float = 30.0   # cap on the corrective lateral velocity
    k_vel: float = 0.010           # tilt command per mm/s of velocity error
    max_command: float = 0.45
    stall_kick: float = 0.30       # min |command| when stuck but asked to move
    stall_speed_mm_s: float = 8.0


class VelocityPathFollower:
    def __init__(self, config: VelocityFollowerConfig):
        self.config = config

    def command(
        self,
        position_mm: np.ndarray,
        velocity_mm_s: np.ndarray,
        path_point_mm: np.ndarray,
        tangent: np.ndarray,
        heading_change_deg: float,
        dt_s: float = 0.0,
        extra_speed_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (board tilt command, desired velocity) for logging/overlay.

        extra_speed_scale lets the caller impose additional slowdowns
        (e.g. wall proximity); the tighter of the two constraints wins."""
        cfg = self.config

        # slow down in proportion to how sharply the path turns ahead
        corner_scale = 1.0 - heading_change_deg / cfg.corner_slow_deg
        speed_scale = max(cfg.min_speed_frac, min(corner_scale, extra_speed_scale))
        v_forward = cfg.v_max_mm_s * speed_scale * tangent

        # corrective lateral velocity toward the path centerline, capped
        lateral_error = path_point_mm - position_mm
        v_lat = cfg.k_lat * lateral_error
        lat_norm = float(np.linalg.norm(v_lat))
        if lat_norm > cfg.lat_v_max_mm_s:
            v_lat *= cfg.lat_v_max_mm_s / lat_norm

        v_desired = v_forward + v_lat
        raw = cfg.k_vel * (v_desired - velocity_mm_s)

        # stiction: ball parked but asked to move -> guarantee breakaway tilt
        if (cfg.stall_kick > 0.0
                and float(np.linalg.norm(velocity_mm_s)) < cfg.stall_speed_mm_s
                and float(np.linalg.norm(v_desired)) > 5.0):
            magnitude = float(np.linalg.norm(raw))
            if 1e-9 < magnitude < cfg.stall_kick:
                raw = raw * (cfg.stall_kick / magnitude)

        return np.clip(raw, -cfg.max_command, cfg.max_command), v_desired


class PathFollower:
    def __init__(self, config: PathFollowerConfig):
        self.config = config
        self.integral = np.zeros(2)

    def reset(self) -> None:
        self.integral = np.zeros(2)

    def command(
        self,
        position_mm: np.ndarray,
        velocity_mm_s: np.ndarray,
        target_mm: np.ndarray,
        dt_s: float = 0.0,
    ) -> np.ndarray:
        cfg = self.config
        error = target_mm - position_mm
        err_dist = float(np.linalg.norm(error))
        speed = float(np.linalg.norm(velocity_mm_s))

        if cfg.ki > 0.0 and dt_s > 0.0:
            self.integral += error * dt_s
            # anti-windup: clamp the I contribution, and bleed it off once
            # the ball is at the target so it cannot cause overshoot later
            limit = cfg.integral_limit / cfg.ki
            self.integral = np.clip(self.integral, -limit, limit)
            if err_dist < cfg.stall_dist_mm:
                self.integral *= 0.90

        raw = cfg.kp * error + cfg.ki * self.integral - cfg.kd * velocity_mm_s

        if (cfg.stall_kick > 0.0 and speed < cfg.stall_speed_mm_s
                and err_dist > cfg.stall_dist_mm):
            magnitude = float(np.linalg.norm(raw))
            if 1e-9 < magnitude < cfg.stall_kick:
                raw = raw * (cfg.stall_kick / magnitude)

        return np.clip(raw, -cfg.max_command, cfg.max_command)
