"""Phone-orientation -> board-tilt mapping. Pure math, no I/O, no hardware.

Holding convention: portrait, phone flat like a tray, top edge pointing away
from the operator. In that pose `beta` (front/back tilt) drives yaw and
`gamma` (left/right tilt) drives pitch - the same dy->yaw, dx->pitch
convention scripts/touchpad_teleop.py uses for cursor offset.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhoneTiltConfig:
    deadzone_deg: float = 3.0
    max_tilt_deg: float = 25.0
    max_tilt: float = 0.9
    yaw_sign: float = 1.0
    pitch_sign: float = 1.0
    swap_axes: bool = False


def map_orientation(beta_deg: float, gamma_deg: float, cfg: PhoneTiltConfig) -> tuple[float, float]:
    """Map phone beta/gamma (degrees, already recentered) to (yaw, pitch).

    Both outputs are clamped to [-cfg.max_tilt, +cfg.max_tilt].
    """
    raw_beta, raw_gamma = beta_deg, gamma_deg
    if cfg.swap_axes:
        raw_beta, raw_gamma = raw_gamma, raw_beta

    yaw_deg = _apply_deadzone(raw_beta, cfg.deadzone_deg)
    pitch_deg = _apply_deadzone(raw_gamma, cfg.deadzone_deg)

    yaw = _clamp(yaw_deg / cfg.max_tilt_deg, -1.0, 1.0) * cfg.max_tilt * cfg.yaw_sign
    pitch = _clamp(pitch_deg / cfg.max_tilt_deg, -1.0, 1.0) * cfg.max_tilt * cfg.pitch_sign
    return yaw, pitch


def is_stale(age_s: float, timeout_s: float) -> bool:
    return age_s > timeout_s


def _apply_deadzone(value_deg: float, deadzone_deg: float) -> float:
    return 0.0 if abs(value_deg) < deadzone_deg else value_deg


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
