from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class StallKicker:
    """Decides when and how hard to apply the anti-stiction breakaway kick.

    Fixes two failure modes of a naive per-frame kick:

    1. Timer jitter: the velocity ESTIMATE flickers above the stall threshold
       from measurement noise while the ball is physically parked. A naive
       timer resets on every flicker, toggling the kick on/off at a few Hz -
       the board visibly jitters while the average tilt never exceeds
       breakaway, so the ball just sits there "balanced". Hysteresis: the
       timer only resets once the ball is CLEARLY moving (release_factor x
       threshold); inside the noise band it holds its value.

    2. Insufficient kick: a fixed magnitude below the local breakaway tilt
       stalls forever. The kick escalates (ramp_per_s) the longer the stall
       persists, until the ball physically breaks free (the caller's command
       cap still bounds it).
    """

    def __init__(self, kick: float, speed_mm_s: float, min_duration_s: float,
                 ramp_per_s: float = 0.0, release_factor: float = 2.0):
        self.kick = kick
        self.speed_mm_s = speed_mm_s
        self.min_duration_s = min_duration_s
        self.ramp_per_s = ramp_per_s
        self.release_factor = release_factor
        self.low_speed_time_s = 0.0
        self.last_kick = 0.0  # most recent kick magnitude (0 = not kicking)

    def reset(self) -> None:
        self.low_speed_time_s = 0.0
        self.last_kick = 0.0

    def update(self, speed_mm_s: float, dt_s: float) -> float:
        """Returns 0.0 (no kick) or the kick magnitude to enforce."""
        if speed_mm_s >= self.release_factor * self.speed_mm_s:
            self.low_speed_time_s = 0.0  # clearly rolling: real release
        elif speed_mm_s < self.speed_mm_s:
            self.low_speed_time_s += max(dt_s, 0.0)
        else:
            # noise band: accumulate at half rate instead of holding. A
            # parked ball whose velocity-estimate noise floor sits INSIDE
            # the band would otherwise never build stall time and never get
            # kicked (observed: 22 s stalls at 0.1 commands).
            self.low_speed_time_s += 0.5 * max(dt_s, 0.0)

        if self.kick <= 0.0 or self.low_speed_time_s < self.min_duration_s:
            self.last_kick = 0.0
        else:
            stalled_for = self.low_speed_time_s - self.min_duration_s
            self.last_kick = self.kick + self.ramp_per_s * stalled_for
        return self.last_kick


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
    stall_kick: float = 0.30
    stall_speed_mm_s: float = 8.0   # "stalled" when slower than this
    stall_dist_mm: float = 8.0      # ...and further than this from target
    # Require the low-speed condition to PERSIST this long before kicking.
    # A single slow frame also happens during ordinary deceleration (e.g.
    # easing toward a target); only a sustained stop is real static friction.
    stall_min_duration_s: float = 0.3
    stall_kick_ramp_per_s: float = 0.0  # escalate kick while stall persists


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
    max_command: float = 1
    stall_kick: float = 1       # min |command| when stuck but asked to move
    stall_speed_mm_s: float = 8.00
    stall_request_speed_mm_s: float = 8.00
    # See PathFollowerConfig.stall_min_duration_s: a slow instant is normal
    # while braking into a corner; only a sustained stop is real stiction.
    stall_min_duration_s: float = 0.03
    stall_kick_ramp_per_s: float = 0.0  # escalate kick while stall persists
    # Braking may exceed max_command up to this value (0 = same as
    # max_command). max_command is a gentleness cap for DRIVING; stopping a
    # fast ball needs the full tilt authority (firmware still clamps).
    brake_max_command: float = 0.0
    # Speed-proportional brake ceiling (ABS); see CarrotVelocityFollowerConfig.
    brake_cmd_per_mm_s: float = 0.0
    brake_cmd_floor: float = 0.06


class VelocityPathFollower:
    def __init__(self, config: VelocityFollowerConfig):
        self.config = config
        self.kicker = StallKicker(
            kick=config.stall_kick,
            speed_mm_s=config.stall_speed_mm_s,
            min_duration_s=config.stall_min_duration_s,
            ramp_per_s=config.stall_kick_ramp_per_s,
        )

    def reset(self) -> None:
        self.kicker.reset()

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

        # Stiction: kick only after low speed PERSISTS (a slow instant also
        # happens during intentional corner braking), with hysteresis against
        # velocity-estimate noise and escalation while the stall lasts.
        speed = float(np.linalg.norm(velocity_mm_s))
        kick = self.kicker.update(speed, dt_s)
        v_des_norm = float(np.linalg.norm(v_desired))
        if kick > 0.0 and v_des_norm > cfg.stall_request_speed_mm_s:
            magnitude = float(np.linalg.norm(raw))
            if magnitude < kick:
                raw = (kick / v_des_norm) * v_desired  # stable direction

        # asymmetric authority: a command opposing the current motion is a
        # brake and may use the full tilt range; driving stays gentle
        cap = cfg.max_command
        is_braking = float(np.dot(raw, velocity_mm_s)) < 0.0
        if cfg.brake_max_command > cfg.max_command and speed > 20.0 and is_braking:
            cap = cfg.brake_max_command
        if cfg.brake_cmd_per_mm_s > 0.0 and is_braking:
            limit = cfg.brake_cmd_floor + cfg.brake_cmd_per_mm_s * speed
            magnitude = float(np.linalg.norm(raw))
            if magnitude > limit:
                raw = raw * (limit / magnitude)
        return np.clip(raw, -cap, cap), v_desired


@dataclass
class CarrotVelocityFollowerConfig:
    """Velocity controller that aims at a lookahead point on the path."""

    v_max_mm_s: float = 45.0
    min_speed_frac: float = 0.25
    corner_slow_deg: float = 110.0
    k_vel: float = 0.010
    max_command: float = 0.45
    stall_kick: float = 0.30
    stall_speed_mm_s: float = 8.0
    stall_request_speed_mm_s: float = 1.0
    stall_min_duration_s: float = 0.3
    # Escalate the kick while the stall persists (per second of stall), so a
    # spot whose breakaway tilt exceeds stall_kick still gets un-stuck.
    stall_kick_ramp_per_s: float = 0.0
    # Braking may exceed max_command up to this value (0 = same as
    # max_command). max_command is a gentleness cap for DRIVING; stopping a
    # fast ball needs the full tilt authority (firmware still clamps).
    brake_max_command: float = 0.0
    # Speed-proportional brake ceiling (like ABS): tilt is FORCE, so any
    # brake tilt still applied when the ball reaches zero speed launches it
    # backward - observed as a violent forward/backward oscillation around
    # every hard stop. Cap |brake| <= floor + per_mm_s * speed so the brake
    # melts away as the ball slows. 0 disables.
    brake_cmd_per_mm_s: float = 0.0
    brake_cmd_floor: float = 0.06


class CarrotVelocityPathFollower:
    def __init__(self, config: CarrotVelocityFollowerConfig):
        self.config = config
        self.kicker = StallKicker(
            kick=config.stall_kick,
            speed_mm_s=config.stall_speed_mm_s,
            min_duration_s=config.stall_min_duration_s,
            ramp_per_s=config.stall_kick_ramp_per_s,
        )

    def reset(self) -> None:
        self.kicker.reset()

    def command(
        self,
        position_mm: np.ndarray,
        velocity_mm_s: np.ndarray,
        carrot_mm: np.ndarray,
        heading_change_deg: float,
        dt_s: float = 0.0,
        extra_speed_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (board tilt command, desired velocity).

        Unlike the local-tangent follower, this controller points desired
        motion at a real lookahead point so the target marker stays ahead of
        the ball and the board tilts decisively toward it.
        """
        cfg = self.config
        to_carrot = carrot_mm - position_mm
        distance = float(np.linalg.norm(to_carrot))
        if distance < 1e-9:
            v_desired = np.zeros(2)
        else:
            direction = to_carrot / distance
            corner_scale = 1.0 - heading_change_deg / max(cfg.corner_slow_deg, 1e-9)
            speed_scale = max(cfg.min_speed_frac, min(corner_scale, extra_speed_scale))
            v_desired = cfg.v_max_mm_s * speed_scale * direction

        raw = cfg.k_vel * (v_desired - velocity_mm_s)

        speed = float(np.linalg.norm(velocity_mm_s))
        kick = self.kicker.update(speed, dt_s)
        v_des_norm = float(np.linalg.norm(v_desired))
        if kick > 0.0 and v_des_norm > cfg.stall_request_speed_mm_s:
            magnitude = float(np.linalg.norm(raw))
            if magnitude < kick:
                # kick along the DESIRED direction (stable, toward the
                # carrot), not along the raw command whose direction jitters
                # with velocity-estimate noise while the ball is parked
                raw = (kick / v_des_norm) * v_desired

        # asymmetric authority: a command opposing the current motion is a
        # brake and may use the full tilt range; driving stays gentle
        cap = cfg.max_command
        is_braking = float(np.dot(raw, velocity_mm_s)) < 0.0
        if cfg.brake_max_command > cfg.max_command and speed > 20.0 and is_braking:
            cap = cfg.brake_max_command
        if cfg.brake_cmd_per_mm_s > 0.0 and is_braking:
            # ABS: the brake ceiling shrinks with speed, reaching ~flat at
            # the stop so the residual tilt cannot launch the ball backward
            limit = cfg.brake_cmd_floor + cfg.brake_cmd_per_mm_s * speed
            magnitude = float(np.linalg.norm(raw))
            if magnitude > limit:
                raw = raw * (limit / magnitude)
        return np.clip(raw, -cap, cap), v_desired


class PathFollower:
    def __init__(self, config: PathFollowerConfig):
        self.config = config
        self.integral = np.zeros(2)
        self.kicker = StallKicker(
            kick=config.stall_kick,
            speed_mm_s=config.stall_speed_mm_s,
            min_duration_s=config.stall_min_duration_s,
            ramp_per_s=config.stall_kick_ramp_per_s,
        )

    def reset(self) -> None:
        self.integral = np.zeros(2)
        self.kicker.reset()

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

        kick = self.kicker.update(speed, dt_s)
        if kick > 0.0 and err_dist > cfg.stall_dist_mm:
            magnitude = float(np.linalg.norm(raw))
            if magnitude < kick:
                raw = (kick / err_dist) * error  # stable direction: at target

        return np.clip(raw, -cfg.max_command, cfg.max_command)
