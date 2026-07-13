#!/usr/bin/env python3
"""Closed-loop path following: camera -> ball -> PD controller -> servos.

Prerequisites (each has its own script, test them in order):
  1. calibration/board_homography.npz   (scripts/calibrate_homography.py)
  2. a real path CSV                    (scripts/annotate_path.py)
  3. calibration/axis_map.npz           (scripts/axis_check.py)

First closed-loop test: start with --dry-run (no servos, overlay only), then
run for real with a low cap, e.g. --max-command 0.2, on a short straight
segment before attempting the full maze.

Keys in the preview window: q/Esc = stop (returns to neutral).
"""
from __future__ import annotations

import argparse
import contextlib
from collections import deque
from ctypes import ArgumentError
import time
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.control.axis_map import AxisMap
from cps_maze.control.pid import (
    CarrotVelocityFollowerConfig,
    CarrotVelocityPathFollower,
    PathFollower,
    PathFollowerConfig,
    VelocityFollowerConfig,
    VelocityPathFollower,
)
from cps_maze.control.trim import NeutralTrim
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand
from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.planning.hazards import HoleMap, should_emergency_brake
from cps_maze.planning.path import WaypointPath
from cps_maze.planning.recovery_astar import (
    RecoveryAStarConfig,
    RecoveryAStarPlanner,
    progress_limited_point_along_polyline,
)
from cps_maze.planning.speed_profile import build_speed_profile
from cps_maze.planning.walls import WallMap
from cps_maze.vision.ball_pipeline import make_tracker
from cps_maze.vision.state_estimator import LowPassVelocityEstimator

WINDOW = "autonomous run"

RUN_LOG_FIELDS = [
    "timestamp_s", "found", "x_mm", "y_mm", "vx_mm_s", "vy_mm_s",
    "target_x_mm", "target_y_mm", "progress_mm",
    "carrot_x_mm", "carrot_y_mm", "desired_vx_mm_s", "desired_vy_mm_s",
    "cross_track_mm", "turn_deg", "wall_speed_scale", "hole_brake",
    "wall_distance_mm", "target_speed_mm_s",
    "hole_hazard_distance_mm", "hole_speed_cap_mm_s",
    "wall_escape_x", "wall_escape_y",
    "board_cmd_x", "board_cmd_y", "yaw_command", "pitch_command",
    "stall_kick",
]


def slew_limit_command(
    target: np.ndarray,
    prev: np.ndarray,
    dt_s: float,
    braking: bool,
    slow_per_s: float,
    fast_per_s: float,
    fast_reduce: bool = True,
) -> np.ndarray:
    """Per-axis slew limiting with an asymmetric rule: increasing drive is
    limited gently, while braking and (when ``fast_reduce``) reducing a
    command's magnitude use the fast rate.

    Fast reduction unwinds a large stall-kick command quickly when the ball is
    already overspeeding in the same direction (dot(cmd, v) > 0, so the braking
    fast lane never engages) - observed as the ball being pushed at +0.65 while
    at 4x the planned speed, straight into a hole. But when the ball is STALLED
    (fast_reduce=False) that same fast unwind collapses a breakaway kick tilt
    before the servo can hold it, so the board only twitches and the ball never
    accelerates. So the caller enables fast reduction only while the ball is
    actually moving.
    """
    out = target.copy()
    for i in range(len(out)):
        reducing = abs(target[i]) < abs(prev[i])
        use_fast = braking or (reducing and fast_reduce)
        rate = fast_per_s if use_fast else slow_per_s
        step = rate * dt_s
        out[i] = prev[i] + float(np.clip(target[i] - prev[i], -step, step))
    return out


def unstick_is_stuck(
    net_disp_mm: float,
    span_s: float,
    window_s: float,
    target_speed_mm_s: float,
    dist_mm: float,
    progress_frac: float,
) -> bool:
    """True if the ball has genuinely failed to make progress over the window.

    Stuck is judged against how far the PLAN wanted the ball to travel
    (target_speed * window), not a fixed distance. A fixed 6 mm / 1 s threshold
    falsely fired on a ball rolling slowly-but-steadily past a hole (where the
    planned speed is intentionally low) and then launched it into the hole. The
    absolute dist_mm is kept only as an UPPER bound so a fast zone does not
    trigger unstick too eagerly.
    """
    if target_speed_mm_s <= 2.0 or window_s <= 0.0:
        return False
    if span_s < 0.8 * window_s:  # window not filled yet
        return False
    threshold = min(dist_mm, progress_frac * target_speed_mm_s * window_s)
    return net_disp_mm < threshold


def unstick_bias_command(
    magnitude: float,
    to_target_mm: np.ndarray,
    velocity_mm_s: np.ndarray,
    damping: float,
    cap: float,
) -> np.ndarray:
    """A bounded, velocity-damped push toward the target, to be ADDED to the
    follower command - never to replace it.

    Adding it (instead of overwriting) preserves the follower's velocity
    feedback, and the explicit ``-damping * velocity`` term makes the push shrink
    (and reverse to a brake) as the ball accelerates in the push direction. So
    unstick can break stiction from rest without launching the ball open-loop
    the way ``board_cmd = magnitude * direction`` did. The magnitude is capped.
    """
    to_target_mm = np.asarray(to_target_mm, dtype=float)
    n = float(np.linalg.norm(to_target_mm))
    if n < 1e-6 or magnitude <= 0.0:
        return np.zeros(2)
    bias = (magnitude * (to_target_mm / n)
            - damping * np.asarray(velocity_mm_s, dtype=float))
    m = float(np.linalg.norm(bias))
    if cap > 0.0 and m > cap:
        bias = bias * (cap / m)
    return bias


def load_holes(path: Path) -> np.ndarray:
    """Returns (N, 3) array of x_mm, y_mm, radius_mm; empty if file missing."""
    if not path.exists():
        return np.zeros((0, 3))
    rows = np.genfromtxt(path, delimiter=",", names=True)
    rows = np.atleast_1d(rows)
    return np.column_stack([rows["x_mm"], rows["y_mm"], rows["radius_mm"]]).astype(float)


def choose_carrot_point(
    path: WaypointPath,
    position_mm: np.ndarray,
    progress_mm: float,
    lookahead_mm: float,
    min_lookahead_mm: float,
    wall_map: WallMap | None = None,
    step_mm: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Pick the furthest line-of-sight lookahead point allowed by walls."""
    lookahead = max(float(lookahead_mm), 0.0)
    min_lookahead = max(0.0, min(float(min_lookahead_mm), lookahead))
    if wall_map is None:
        return path.point_at_progress_mm(progress_mm + lookahead), lookahead

    step = max(float(step_mm), 1e-6)
    current = lookahead
    while current > min_lookahead:
        point = path.point_at_progress_mm(progress_mm + current)
        if not wall_map.line_blocked(position_mm, point):
            return point, current
        current = max(min_lookahead, current - step)

    point = path.point_at_progress_mm(progress_mm + min_lookahead)
    return point, min_lookahead


def draw_overlay(
    image: np.ndarray,
    homography: Homography,
    path: WaypointPath,
    holes: np.ndarray,
    ball_px: tuple[float, float] | None,
    target_mm: np.ndarray | None,
    servo_cmd: np.ndarray,
    status: str,
) -> np.ndarray:
    out = image.copy()
    path_px = homography.board_points_to_image_px(path.points_mm).astype(np.int32)
    cv2.polylines(out, [path_px], False, (0, 255, 0), 2)
    cv2.circle(out, tuple(path_px[-1]), 8, (255, 0, 255), 2)  # goal
    for x_mm, y_mm, r_mm in holes:
        center = homography.board_point_to_image_px(x_mm, y_mm)
        edge = homography.board_point_to_image_px(x_mm + r_mm, y_mm)
        radius = int(np.hypot(edge[0] - center[0], edge[1] - center[1]))
        cv2.circle(out, (int(center[0]), int(center[1])), max(radius, 3), (0, 0, 255), 2)
    if target_mm is not None:
        tx, ty = homography.board_point_to_image_px(float(target_mm[0]), float(target_mm[1]))
        cv2.circle(out, (int(tx), int(ty)), 6, (0, 255, 255), -1)
    if ball_px is not None:
        cv2.circle(out, (int(ball_px[0]), int(ball_px[1])), 6, (255, 0, 0), 2)
    cv2.putText(out, f"yaw={servo_cmd[0]:+.2f} pitch={servo_cmd[1]:+.2f}  {status}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--path", default=None, help="Path CSV override")
    parser.add_argument("--holes", default="configs/maze_holes.csv")
    parser.add_argument("--axis-map", default="calibration/axis_map.npz")
    parser.add_argument("--wall-mask", default="calibration/wall_mask.npz",
                        help="Wall mask from build_wall_mask.py; enables "
                             "cross-wall association rejection and wall slowdown")
    parser.add_argument("--port", default=None, help="Serial port override, e.g. COM10")
    parser.add_argument("--log", default="data/raw/autonomous_run.csv")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="No serial output; visualize what the controller would do")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--ki", type=float, default=None)
    parser.add_argument("--stall-kick", type=float, default=None,
                        help="Min command magnitude when the ball is stalled off-target "
                             "(the tilt that breaks static friction, ~0.3)")
    parser.add_argument("--controller", choices=["carrot", "velocity", "position"], default=None,
                        help="carrot (default): chase a lookahead path point with "
                             "velocity feedback; velocity: track local path tangent; "
                             "position: legacy lookahead PD")
    parser.add_argument("--v-max", type=float, default=None,
                        help="Cruise speed mm/s for the velocity controller")
    parser.add_argument("--command-slew", type=float, default=None,
                        help="Max |servo command| change per second sent to hardware "
                             "(0 disables). Smooths sudden jumps from stiction kicks "
                             "or re-association so corners don't get punched through.")
    parser.add_argument("--max-command", type=float, default=None,
                        help="Cap |servo command|; start low (e.g. 0.2)")
    parser.add_argument("--lookahead", type=float, default=None, help="Lookahead mm")
    parser.add_argument("--goal-tolerance-mm", type=float, default=15.0)
    parser.add_argument("--lost-timeout-s", type=float, default=2.0,
                        help="Stop if the ball is undetected this long")
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)
    tracker = make_tracker(config.vision)
    path_file = Path(args.path) if args.path else config.resolve_path(config.maze["path_file"])
    path = WaypointPath.from_csv(path_file)
    holes = load_holes(Path(args.holes))

    axis_map_file = Path(args.axis_map)
    if axis_map_file.exists():
        axis_map = AxisMap.load(axis_map_file)
        print(f"axis map loaded from {axis_map_file}:\n{axis_map.matrix}")
    else:
        axis_map = AxisMap.identity()
        print("WARNING: no axis map found - using identity. Run scripts/axis_check.py "
              "first, or the controller may push the ball the wrong way.")

    wall_map = None
    wall_mask_file = Path(args.wall_mask)
    if wall_mask_file.exists():
        wall_map = WallMap.load(wall_mask_file)
        print(f"wall mask loaded from {wall_mask_file} "
              "(cross-wall rejection + wall slowdown active)")
        # Consistency self-check: the route centerline must lie in free
        # space. A stale mask (recalibrated homography, old mask) draws
        # phantom walls that block the association's line-of-sight checks -
        # observed as huge cross-track errors and the ball stuck at the
        # very start of the route.
        total_mm = float(path.cumulative_lengths[-1])
        probe = np.arange(0.0, total_mm, 4.0)
        on_wall = sum(
            1 for s in probe if wall_map.is_wall(path.point_at_progress_mm(s)))
        frac = on_wall / max(len(probe), 1)
        if frac > 0.02:
            print(f"WARNING: {100 * frac:.0f}% of the route reads as INSIDE "
                  "a wall - the wall mask is stale for the current "
                  "homography/path. Rebuild it now: python "
                  "scripts/build_wall_mask.py  (running without it is "
                  "better than running with a wrong one)")
            wall_map = None
            print("wall mask DISABLED for this run.")
    else:
        print("note: no wall mask - run scripts/build_wall_mask.py to enable "
              "cross-wall association rejection and wall-proximity slowdown")

    kp = args.kp if args.kp is not None else float(config.control["kp"])
    kd = args.kd if args.kd is not None else float(config.control["kd"])
    ki = args.ki if args.ki is not None else float(config.control.get("ki", 0.0))
    stall_kick = (args.stall_kick if args.stall_kick is not None
                  else float(config.control.get("stall_kick", 0.0)))
    max_command = (args.max_command if args.max_command is not None
                   else float(config.control["max_command"]))
    lookahead_mm = (args.lookahead if args.lookahead is not None
                    else float(config.control["lookahead_mm"]))
    carrot_lookahead_mm = (args.lookahead if args.lookahead is not None
                           else float(config.control.get(
                               "carrot_lookahead_mm", lookahead_mm)))
    carrot_min_lookahead_mm = float(config.control.get(
        "carrot_min_lookahead_mm", min(12.0, carrot_lookahead_mm)))

    if 0.0 < max_command < stall_kick:
        print(f"WARNING: --max-command {max_command} is BELOW stall_kick "
              f"{stall_kick}. The anti-stiction kick gets clipped away and the "
              f"ball may never move. Use max-command >= {stall_kick}, or lower "
              f"stall_kick deliberately.")

    mode = args.controller or str(config.control.get("mode", "carrot")).lower()
    v_max = (args.v_max if args.v_max is not None
             else float(config.control.get("v_max_mm_s", 45.0)))
    command_slew_per_s = (args.command_slew if args.command_slew is not None
                          else float(config.control.get("command_slew_per_s", 3.0)))
    stall_min_duration_s = float(config.control.get("stall_min_duration_s", 0.3))
    stall_speed_mm_s = float(config.control.get("stall_speed_mm_s", 8.0))
    stall_kick_ramp_per_s = float(config.control.get("stall_kick_ramp_per_s", 0.15))
    stall_kick_max = float(config.control.get("stall_kick_max", 0.7))
    corner_noise_deg = float(config.control.get("corner_noise_deg", 6.0))
    corner_span_mm = float(config.control.get("corner_span_mm", 30.0))

    # Hole awareness: anticipatory speed cap from braking physics, plus a
    # last-resort emergency brake when the trajectory enters a hole and the
    # stopping distance exceeds the distance to it.
    hole_map = HoleMap(
        holes,
        ball_radius_mm=float(config.control.get("ball_radius_mm", 6.0)),
        margin_mm=float(config.control.get("hole_margin_mm", 4.0)),
    )
    hole_brake_accel = float(config.control.get("hole_brake_accel_mm_s2", 250.0))
    hole_emergency = bool(config.control.get("hole_emergency_brake", True))
    hole_emergency_offroute_mm = float(config.control.get(
        "hole_emergency_offroute_mm", 12.0))
    hole_emergency_align_deg = float(config.control.get(
        "hole_emergency_align_deg", 40.0))
    wall_escape_distance_mm = float(config.control.get("wall_escape_distance_mm", 6.0))
    wall_escape_speed_mm_s = float(config.control.get("wall_escape_speed_mm_s", 8.0))
    wall_escape_command = float(config.control.get("wall_escape_command", 0.20))
    wall_escape_min_cmd = float(config.control.get("wall_escape_min_command", 0.08))
    recovery_enabled = bool(config.control.get("recovery_astar_enabled", True))
    recovery_cross_track_mm = float(config.control.get("recovery_astar_cross_track_mm", 15.0))
    recovery_wall_distance_mm = float(config.control.get("recovery_astar_wall_mm", 4.0))
    recovery_stall_speed_mm_s = float(config.control.get("recovery_astar_stall_speed_mm_s", 5.0))
    recovery_stall_duration_s = float(config.control.get("recovery_astar_stall_duration_s", 0.8))
    recovery_follow_mm = float(config.control.get("recovery_astar_follow_mm", 20.0))
    recovery_goal_lookahead_mm = float(config.control.get(
        "recovery_astar_goal_lookahead_mm", 35.0))
    recovery_max_backtrack_mm = float(config.control.get(
        "recovery_astar_max_backtrack_mm", 4.0))
    recovery_planner = None
    if recovery_enabled and wall_map is not None:
        recovery_planner = RecoveryAStarPlanner(
            wall_map,
            hole_map,
            RecoveryAStarConfig(
                grid_mm=float(config.control.get("recovery_astar_grid_mm", 4.0)),
                wall_clearance_mm=float(config.control.get(
                    "recovery_astar_wall_clearance_mm", 2.0)),
                hole_clearance_mm=float(config.control.get(
                    "recovery_astar_hole_clearance_mm", 0.0)),
                max_snap_mm=float(config.control.get("recovery_astar_max_snap_mm", 18.0)),
                max_expansions=int(config.control.get(
                    "recovery_astar_max_expansions", 8000)),
            ),
        )
    # Braking authority: max_command is a DRIVING gentleness cap; stopping a
    # fast ball needs the full tilt range (the firmware still clamps).
    brake_max_command = float(config.control.get("brake_max_command", 1.0))
    brake_max_command = max(brake_max_command, max_command)
    brake_slew_per_s = float(config.control.get("brake_slew_per_s", 10.0))
    brake_cmd_per_mm_s = float(config.control.get("brake_cmd_per_mm_s", 0.012))
    brake_cmd_floor = float(config.control.get("brake_cmd_floor", 0.06))
    plan_latency_s = float(config.control.get("plan_latency_s", 0.35))
    slowzone_max_command = float(config.control.get("slowzone_max_command", 0.55))
    # Composure: when control gets violent (emergency, or speed far above
    # plan), stop pursuing progress - hold position, damp the ball, then
    # resume the moment control is regained. Prevents the freak-out cascade:
    # rough brake -> too fast -> unstable -> pushed forward anyway -> hole.
    # Trigger only on a GENUINE runaway: fire when speed exceeds BOTH a large
    # multiple of the planned speed AND an absolute floor, so ordinary
    # overshoot during following never trips it (that over-triggered before).
    stabilize_enabled = bool(config.control.get("stabilize_enabled", True))
    stabilize_margin = float(config.control.get("stabilize_overspeed_margin_mm_s", 40.0))
    stabilize_trigger_mult = float(config.control.get("stabilize_trigger_mult", 3.0))
    stabilize_trigger_floor = float(config.control.get("stabilize_trigger_speed_mm_s", 55.0))
    # Exit as soon as the ball is back under control (speed below this) for a
    # brief settle - NOT a dead stop, which held the ball frozen for the full
    # timeout every time and blocked all progress.
    stabilize_exit_speed = float(config.control.get("stabilize_exit_speed_mm_s", 18.0))
    stabilize_settle_s = float(config.control.get("stabilize_settle_s", 0.2))
    stabilize_max_s = float(config.control.get("stabilize_max_s", 2.0))
    stabilize_kp = float(config.control.get("stabilize_kp", 0.010))
    stabilize_kd = float(config.control.get("stabilize_kd", 0.012))
    # Displacement-based unstick. The velocity-based stall kick cannot break the
    # ball out of a tight corner: the ball twitches to ~10 mm/s in place, which
    # resets the kick before it ramps, so the ball jitters forever. This detects
    # stuck by ACTUAL net displacement over a window and ramps a sustained push
    # toward the carrot until the ball genuinely moves.
    unstick_enabled = bool(config.control.get("unstick_enabled", True))
    unstick_window_s = float(config.control.get("unstick_window_s", 1.0))
    unstick_dist_mm = float(config.control.get("unstick_dist_mm", 6.0))
    unstick_base = float(config.control.get("unstick_base", 0.5))
    unstick_ramp_per_s = float(config.control.get("unstick_ramp_per_s", 0.6))
    unstick_max = float(config.control.get("unstick_max", 0.65))
    # Unstick is a DAMPED BIAS added to the follower command, not an open-loop
    # overwrite: this velocity-damping term keeps it from launching the ball,
    # and the progress fraction judges "stuck" against the planned travel so a
    # slow-but-progressing ball near a hole is not mistaken for stuck.
    unstick_kd = float(config.control.get("unstick_kd", 0.02))
    unstick_progress_frac = float(config.control.get("unstick_progress_frac", 0.35))
    # Reuse the hole slow-band as the "near a hole -> cap unstick gently" radius.
    unstick_hole_band_mm = float(config.control.get("hole_slow_band_mm", 20.0))
    # The runtime wall speed-scale is OFF-route protection only; the speed
    # PROFILE already folds planned wall clearance into the on-route target.
    # Applying the runtime scale on-route double-counts it, and a dense mask
    # then crushes the on-route speed to a crawl (observed: 0.35 scale parked
    # the ball at the start). Only let it act once the ball is this far off the
    # centerline.
    wall_scale_cross_track_mm = float(
        config.control.get("wall_scale_cross_track_mm", 8.0))

    # One coherent speed PLAN for the whole route, computed once: hole
    # passes get a committed moderate speed (not a crawl), every slowdown
    # is reachable by braking (backward pass) and exits ramp smoothly
    # (forward pass). Replaces the per-frame reactive hole cap, whose
    # interaction with the stall kick caused the ball to "spazz" at
    # overlapping capture zones.
    profile = build_speed_profile(
        path, hole_map, wall_map,
        v_max_mm_s=v_max,
        hole_pass_mm_s=float(config.control.get("hole_pass_mm_s", 16.0)),
        hole_slow_band_mm=float(config.control.get("hole_slow_band_mm", 20.0)),
        floor_mm_s=float(config.control.get("profile_floor_mm_s", 12.0)),
        corner_slow_deg=float(config.control.get("corner_slow_deg", 110.0)),
        corner_span_mm=corner_span_mm,
        corner_noise_deg=corner_noise_deg,
        corner_min_frac=float(config.control.get("min_speed_frac", 0.25)),
        accel_mm_s2=hole_brake_accel,
        end_speed_mm_s=float(config.control.get("end_speed_mm_s", 10.0)),
    )
    print(profile.summary())
    # Planned slow rolling must NEVER be mistaken for a stall, or the kick
    # launches the ball right where the plan wants it careful.
    safe_stall_speed = 0.5 * profile.min_speed()
    if stall_speed_mm_s > safe_stall_speed:
        print(f"stall_speed_mm_s {stall_speed_mm_s:.1f} clamped to "
              f"{safe_stall_speed:.1f} (must stay below the profile minimum "
              f"{profile.min_speed():.1f} mm/s)")
        stall_speed_mm_s = safe_stall_speed

    follower = PathFollower(PathFollowerConfig(
        kp=kp, kd=kd, ki=ki, max_command=max_command,
        stall_kick=stall_kick,
        integral_limit=float(config.control.get("integral_limit", 0.25)),
        stall_speed_mm_s=stall_speed_mm_s,
        stall_dist_mm=float(config.control.get("stall_dist_mm", 8.0)),
        stall_min_duration_s=stall_min_duration_s,
        stall_kick_ramp_per_s=stall_kick_ramp_per_s,
        stall_kick_max=stall_kick_max,
    ))
    velocity_follower = VelocityPathFollower(VelocityFollowerConfig(
        v_max_mm_s=v_max,
        min_speed_frac=float(config.control.get("min_speed_frac", 0.25)),
        corner_slow_deg=float(config.control.get("corner_slow_deg", 110.0)),
        k_lat=float(config.control.get("k_lat", 2.5)),
        lat_v_max_mm_s=float(config.control.get("lat_v_max_mm_s", 30.0)),
        k_vel=float(config.control.get("k_vel", 0.010)),
        max_command=max_command,
        stall_kick=stall_kick,
        stall_speed_mm_s=stall_speed_mm_s,
        stall_request_speed_mm_s=float(config.control.get(
            "stall_request_speed_mm_s", 1.0)),
        stall_min_duration_s=stall_min_duration_s,
        stall_kick_ramp_per_s=stall_kick_ramp_per_s,
        stall_kick_max=stall_kick_max,
        brake_max_command=brake_max_command,
        brake_cmd_per_mm_s=brake_cmd_per_mm_s,
        brake_cmd_floor=brake_cmd_floor,
    ))
    carrot_follower = CarrotVelocityPathFollower(CarrotVelocityFollowerConfig(
        v_max_mm_s=v_max,
        min_speed_frac=float(config.control.get("min_speed_frac", 0.25)),
        corner_slow_deg=float(config.control.get("corner_slow_deg", 110.0)),
        k_vel=float(config.control.get("k_vel", 0.010)),
        max_command=max_command,
        stall_kick=stall_kick,
        stall_speed_mm_s=stall_speed_mm_s,
        stall_request_speed_mm_s=float(config.control.get(
            "stall_request_speed_mm_s", 1.0)),
        stall_min_duration_s=stall_min_duration_s,
        stall_kick_ramp_per_s=stall_kick_ramp_per_s,
        stall_kick_max=stall_kick_max,
        brake_max_command=brake_max_command,
        brake_cmd_per_mm_s=brake_cmd_per_mm_s,
        brake_cmd_floor=brake_cmd_floor,
    ))
    print(f"controller: {mode}" + (
        f" (v_max {v_max:.0f} mm/s)" if mode in ("carrot", "velocity") else ""))
    estimator = LowPassVelocityEstimator(
        tau_s=float(config.control.get("velocity_tau_s", 0.10)),
        min_dt_s=float(config.control.get("velocity_min_dt_s", 0.006)),
        max_speed_mm_s=float(config.control.get("velocity_max_speed_mm_s", 250.0)),
    )
    total_length = float(path.cumulative_lengths[-1])

    trim = NeutralTrim.load_if_exists()
    if trim.yaw or trim.pitch:
        print(f"neutral trim loaded: yaw={trim.yaw:+.3f} pitch={trim.pitch:+.3f} "
              "(command 0,0 = level board)")

    if args.dry_run:
        serial_ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
        print("DRY RUN: servos disabled, visualization only")
    else:
        serial_ctx = ArduinoServoLink(
            port=args.port or config.serial["port"],
            baudrate=int(config.serial["baudrate"]),
            timeout_s=float(config.serial["timeout_s"]),
            trim_yaw=trim.yaw, trim_pitch=trim.pitch,
        )

    start_time = monotonic()
    last_seen = monotonic()
    prev_timestamp_s = None
    progress_est = None  # last known path progress; keeps association local
    prev_servo_cmd = np.zeros(2)  # for command-slew limiting
    recovery_low_speed_s = 0.0
    stabilize_active = False
    stabilize_still = 0.0
    stabilize_entered = 0.0
    freeze_point = np.zeros(2)
    pos_history: deque = deque()  # (timestamp, position) for unstick detection
    unstick_time = 0.0
    outcome = "stopped by user"

    mouse_state: dict = {}
    if not args.no_preview:
        cv2.namedWindow(WINDOW)

        def on_mouse(event: int, x: int, y: int, *_rest) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_state["seed"] = (x, y)

        cv2.setMouseCallback(WINDOW, on_mouse)

    with CameraCapture(config.camera) as camera, serial_ctx as link, \
            CsvRunLogger(Path(args.log), RUN_LOG_FIELDS) as logger:
        if link is not None:
            time.sleep(2.0)  # Arduino reset after port open
            link.neutral()

        # Arming phase: the run (and the ball-lost timeout) must not start
        # until the ball is actually acquired and the operator says go.
        armed = args.no_preview  # headless mode: start immediately as before
        if not args.no_preview:
            print("ARMING: click the ball to seed, then SPACE to start (q quits)")
            while True:
                frame = camera.read()
                seed = mouse_state.pop("seed", None)
                if seed is not None and hasattr(tracker, "seed"):
                    tracker.seed(*seed)
                detection = tracker.detect(frame.image)
                ball_px = ((detection.x_px, detection.y_px)
                           if detection.found else None)
                status = ("ball locked - SPACE to start" if detection.found
                          else "CLICK THE BALL to seed")
                view = draw_overlay(frame.image, homography, path, holes,
                                    ball_px, None, np.zeros(2), status)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and detection.found:
                    armed = True
                    break
                if key in (27, ord("q")):
                    break
            start_time = monotonic()
            last_seen = monotonic()

        if not armed:
            outcome = "aborted before start"
        try:
            while armed:
                if args.max_seconds > 0 and monotonic() - start_time >= args.max_seconds:
                    outcome = "time limit"
                    break

                frame = camera.read()
                seed = mouse_state.pop("seed", None)
                if seed is not None and hasattr(tracker, "seed"):
                    tracker.seed(*seed)  # click the ball to (re)seed the track
                    progress_est = None  # ball may have been moved: re-associate
                    prev_timestamp_s = None
                    prev_servo_cmd = np.zeros(2)
                    recovery_low_speed_s = 0.0
                    stabilize_active = False
                    unstick_time = 0.0
                    pos_history.clear()
                    estimator.reset()
                    follower.reset()
                    velocity_follower.reset()
                    carrot_follower.reset()
                detection = tracker.detect(frame.image)
                servo_cmd = np.zeros(2)
                target = None
                ball_px = None
                status = "ball lost"

                if detection.found and detection.x_px is not None and detection.y_px is not None:
                    last_seen = monotonic()
                    ball_px = (detection.x_px, detection.y_px)
                    board_xy = np.array(
                        homography.image_point_to_board_mm(detection.x_px, detection.y_px),
                        dtype=float,
                    )
                    state = estimator.update(board_xy, frame.timestamp_s)
                    # Windowed projection: the ball cannot jump along the path
                    # between frames, so never associate it with a corridor far
                    # away in path order. With a wall mask, additionally reject
                    # candidates whose line of sight from the ball crosses a
                    # wall (adjacent chicane corridors sit inside the window).
                    if wall_map is not None:
                        cands = path.candidate_projections(board_xy, progress_est)
                        clear = [c for c in cands
                                 if not wall_map.line_blocked(board_xy, c[2])]
                        if not clear:  # wedged against a wall: nearest anyway
                            clear = cands
                        progress, cross, _ = clear[0]
                        if progress_est is not None and cross > 35.0:
                            cands = path.candidate_projections(board_xy, None)
                            clear = [c for c in cands
                                     if not wall_map.line_blocked(board_xy, c[2])]
                            if clear:
                                progress, cross, _ = clear[0]
                    else:
                        progress, cross = path.nearest_progress_and_distance_mm(
                            board_xy, progress_est
                        )
                        if progress_est is not None and cross > 35.0:
                            progress, cross = path.nearest_progress_and_distance_mm(
                                board_xy)
                    progress_est = progress

                    dt_s = (frame.timestamp_s - prev_timestamp_s
                            if prev_timestamp_s is not None else 0.0)
                    prev_timestamp_s = frame.timestamp_s

                    # The route speed PLAN, computed once at startup: hole
                    # passes, walls, corners and braking feasibility are all
                    # already folded into one smooth profile. The controller
                    # just tracks the planned speed at this progress.
                    hole_brake = ""
                    speed_now = float(np.linalg.norm(state.velocity_mm_s))
                    # read the plan where the ball WILL be (control latency),
                    # so slow zones are entered at plan speed, not discovered
                    target_speed = profile.speed_at(
                        progress + speed_now * plan_latency_s)
                    profile_scale = min(1.0, target_speed / max(v_max, 1e-6))
                    if profile_scale < 0.8:
                        hole_brake = "slow"
                    wall_distance = (wall_map.wall_distance_mm(board_xy)
                                     if wall_map is not None else float("inf"))
                    if speed_now >= 2.0 * recovery_stall_speed_mm_s:
                        recovery_low_speed_s = 0.0
                    elif speed_now < recovery_stall_speed_mm_s and dt_s > 0.0:
                        recovery_low_speed_s += dt_s

                    # Runtime wall scale is OFF-route protection ONLY. The
                    # profile already handles planned wall clearance on-route;
                    # applying the runtime scale on-route too double-counts it,
                    # and a dense/over-marked mask then crushes the on-route
                    # speed to a crawl (observed: 0.35 scale parked the ball at
                    # the start). On-route trust the profile; slow near walls
                    # only once the ball has drifted off the centerline.
                    wall_scale = 1.0
                    if (wall_map is not None
                            and float(cross) > wall_scale_cross_track_mm):
                        wall_scale = wall_map.speed_scale(board_xy)
                    speed_scale = min(wall_scale, profile_scale)

                    if mode == "velocity":
                        recovery_reason = ""
                        path_point = path.point_at_progress_mm(progress)
                        tangent = path.tangent_at_progress_mm(progress)
                        turn_deg = path.heading_change_deg(
                            progress, span_mm=corner_span_mm,
                            noise_deg=corner_noise_deg)
                        board_cmd, v_des = velocity_follower.command(
                            state.position_mm, state.velocity_mm_s,
                            path_point, tangent, 0.0, dt_s,
                            extra_speed_scale=speed_scale,
                        )
                        # overlay: aim marker a half-second of travel ahead
                        target = state.position_mm + 0.5 * v_des
                        carrot_x = ""
                        carrot_y = ""
                    elif mode == "carrot":
                        turn_deg = path.heading_change_deg(
                            progress, span_mm=corner_span_mm,
                            noise_deg=corner_noise_deg)
                        target, _carrot_lookahead = choose_carrot_point(
                            path, state.position_mm, progress,
                            carrot_lookahead_mm, carrot_min_lookahead_mm,
                            wall_map,
                        )
                        recovery_goal = path.point_at_progress_mm(
                            progress + max(recovery_goal_lookahead_mm,
                                           carrot_lookahead_mm))
                        recovery_reason = ""
                        if recovery_planner is not None:
                            if float(cross) >= recovery_cross_track_mm:
                                recovery_reason = "offroute"
                            elif (wall_map is not None
                                  and speed_now < recovery_stall_speed_mm_s
                                  and wall_map.line_blocked(
                                      state.position_mm, recovery_goal)):
                                recovery_reason = "wall"
                            elif (wall_distance <= recovery_wall_distance_mm
                                  and speed_now < recovery_stall_speed_mm_s):
                                recovery_reason = "wall"
                            elif recovery_low_speed_s >= recovery_stall_duration_s:
                                recovery_reason = "stall"

                            if recovery_reason:
                                recovery_path = recovery_planner.plan(
                                    state.position_mm, recovery_goal)
                                if recovery_path is not None and len(recovery_path) >= 2:
                                    candidate = progress_limited_point_along_polyline(
                                        recovery_path, path, progress,
                                        recovery_follow_mm,
                                        recovery_max_backtrack_mm)
                                    _, cand_off = path.nearest_progress_and_distance_mm(
                                        candidate, progress)
                                    # reject detours that increase distance
                                    # to the route (observed: carrot sent to
                                    # mid-board around a hole capture zone,
                                    # driving the ball into hole 2)
                                    if cand_off <= max(float(cross), 8.0) + 5.0:
                                        target = candidate
                        board_cmd, v_des = carrot_follower.command(
                            state.position_mm, state.velocity_mm_s,
                            target, 0.0, dt_s,
                            extra_speed_scale=speed_scale,
                        )
                        carrot_x = target[0]
                        carrot_y = target[1]
                    else:
                        turn_deg = path.heading_change_deg(
                            progress, span_mm=corner_span_mm,
                            noise_deg=corner_noise_deg)
                        v_des = np.zeros(2)
                        # position mode: speed emerges from the pull toward
                        # the lookahead target, so the plan's slowdown is a
                        # nearer target
                        lookahead_eff = max(6.0, lookahead_mm * speed_scale)
                        target = path.point_at_progress_mm(progress + lookahead_eff)
                        recovery_goal = path.point_at_progress_mm(
                            progress + max(recovery_goal_lookahead_mm,
                                           lookahead_mm))
                        recovery_reason = ""
                        if recovery_planner is not None:
                            if float(cross) >= recovery_cross_track_mm:
                                recovery_reason = "offroute"
                            elif (wall_map is not None
                                  and speed_now < recovery_stall_speed_mm_s
                                  and wall_map.line_blocked(
                                      state.position_mm, recovery_goal)):
                                recovery_reason = "wall"
                            elif (wall_distance <= recovery_wall_distance_mm
                                  and speed_now < recovery_stall_speed_mm_s):
                                recovery_reason = "wall"
                            elif recovery_low_speed_s >= recovery_stall_duration_s:
                                recovery_reason = "stall"

                            if recovery_reason:
                                recovery_path = recovery_planner.plan(
                                    state.position_mm, recovery_goal)
                                if recovery_path is not None and len(recovery_path) >= 2:
                                    candidate = progress_limited_point_along_polyline(
                                        recovery_path, path, progress,
                                        recovery_follow_mm,
                                        recovery_max_backtrack_mm)
                                    _, cand_off = path.nearest_progress_and_distance_mm(
                                        candidate, progress)
                                    # reject detours that increase distance
                                    # to the route (observed: carrot sent to
                                    # mid-board around a hole capture zone,
                                    # driving the ball into hole 2)
                                    if cand_off <= max(float(cross), 8.0) + 5.0:
                                        target = candidate
                        carrot_x = ""
                        carrot_y = ""
                        board_cmd = follower.command(state.position_mm,
                                                     state.velocity_mm_s,
                                                     target, dt_s)
                    wall_escape = np.zeros(2)
                    if (wall_map is not None
                            and wall_escape_command > 0.0
                            and wall_distance <= wall_escape_distance_mm
                            and speed_now < wall_escape_speed_mm_s
                            and float(np.linalg.norm(board_cmd)) > wall_escape_min_cmd):
                        escape_dir = wall_map.escape_direction_mm(state.position_mm)
                        if float(np.linalg.norm(escape_dir)) > 1e-9:
                            away_component = float(np.dot(board_cmd, escape_dir))
                            needed = wall_escape_command - away_component
                            if needed > 0.0:
                                wall_escape = needed * escape_dir
                                board_cmd = board_cmd + wall_escape
                    # Reactive layer: the trajectory enters a hole and the
                    # stopping distance exceeds the distance to it - normal
                    # control can no longer prevent the fall. Full brake
                    # opposite to the velocity, bypassing the slew limiter
                    # (an emergency cannot wait for a ramp).
                    emergency = False
                    if (hole_emergency and speed_now > 15.0
                            and should_emergency_brake(
                                hole_map, state.position_mm,
                                state.velocity_mm_s, hole_brake_accel,
                                path_tangent=path.tangent_at_progress_mm(progress),
                                cross_track_mm=float(cross),
                                offroute_mm=hole_emergency_offroute_mm,
                                align_deg=hole_emergency_align_deg)):
                        emergency = True
                        hole_brake = "emergency"
                        board_cmd = ((-brake_max_command / speed_now)
                                     * state.velocity_mm_s)

                    # Composure state machine: enter on a genuine runaway,
                    # hold position and damp, resume the moment control is back.
                    if stabilize_enabled:
                        overspeed = speed_now > max(
                            stabilize_trigger_mult * target_speed,
                            target_speed + stabilize_margin,
                            stabilize_trigger_floor)
                        if (emergency or overspeed) and not stabilize_active:
                            stabilize_active = True
                            freeze_point = state.position_mm.copy()
                            stabilize_entered = monotonic()
                            stabilize_still = 0.0
                        if stabilize_active and not emergency:
                            # "settled" = speed recovered below exit_speed, held
                            # briefly. Not a dead stop - once the ball is back
                            # under control there is no reason to keep holding.
                            if speed_now < stabilize_exit_speed:
                                stabilize_still += max(dt_s, 0.0)
                            else:
                                stabilize_still = 0.0
                            if (stabilize_still >= stabilize_settle_s
                                    or monotonic() - stabilize_entered
                                    > stabilize_max_s):
                                stabilize_active = False  # calm: resume path
                                follower.reset()
                                velocity_follower.reset()
                                carrot_follower.reset()
                            else:
                                # hold position with damping; no progress
                                hold_err = freeze_point - state.position_mm
                                board_cmd = (stabilize_kp * hold_err
                                             - stabilize_kd * state.velocity_mm_s)
                                m = float(np.linalg.norm(board_cmd))
                                hold_cap = 0.21 + 0.012 * speed_now
                                if m > hold_cap:
                                    board_cmd = board_cmd * (hold_cap / m)
                                hole_brake = "stabilize"

                    # Displacement-based UNSTICK: at a tight corner the ball
                    # twitches in place, which the velocity-based stall kick
                    # reads as "moving" and keeps resetting, so it never frees
                    # the ball. Detect stuck by net displacement vs PLANNED
                    # travel (not a fixed distance, which mis-fired on a slow
                    # hole pass), then ADD a damped, capped bias toward the
                    # carrot - never an open-loop overwrite - so the follower's
                    # velocity feedback stays active and the push cannot launch
                    # the ball. Suppressed/capped near holes and in slow zones.
                    unstick_now = 0.0
                    if unstick_enabled and not emergency and not stabilize_active:
                        pos_history.append((frame.timestamp_s,
                                            state.position_mm.copy()))
                        while (len(pos_history) >= 2
                               and frame.timestamp_s - pos_history[0][0]
                               > unstick_window_s):
                            pos_history.popleft()
                        span = frame.timestamp_s - pos_history[0][0]
                        net_disp = float(np.linalg.norm(
                            state.position_mm - pos_history[0][1]))
                        clearance = (hole_map.clearance_mm(state.position_mm)
                                     if hole_map is not None else float("inf"))
                        stuck = unstick_is_stuck(
                            net_disp, span, unstick_window_s, target_speed,
                            unstick_dist_mm, unstick_progress_frac)
                        # Suppress entirely inside a hole's capture zone - pushing
                        # a ball that is already in the danger zone risks the fall.
                        if stuck and clearance > 0.0:
                            unstick_time += max(dt_s, 0.0)
                            unstick_now = min(
                                unstick_base + unstick_ramp_per_s * unstick_time,
                                unstick_max)
                            # Cap the push hard in slow zones and near holes so a
                            # breakaway there stays gentle, never a launch.
                            u_cap = unstick_max
                            if (profile_scale < 0.75
                                    or clearance < unstick_hole_band_mm):
                                u_cap = min(u_cap, slowzone_max_command)
                            # ADD a damped bias to the follower command (keeps the
                            # follower's velocity feedback/braking active), never
                            # an open-loop overwrite.
                            board_cmd = board_cmd + unstick_bias_command(
                                unstick_now, target - state.position_mm,
                                state.velocity_mm_s, unstick_kd, u_cap)
                            hole_brake = "unstick"
                        else:
                            unstick_time = 0.0
                    else:
                        unstick_time = 0.0
                        pos_history.clear()

                    # a command opposing the motion is a brake: allow the
                    # full tilt range and a faster slew (stopping a fast
                    # ball cannot wait for the gentle driving ramp)
                    braking = emergency or (
                        speed_now > 20.0
                        and float(np.dot(board_cmd, state.velocity_mm_s)) < 0.0)
                    cap = brake_max_command if braking else max_command
                    if profile_scale < 0.75 and not emergency:
                        # calm hands near holes: follow the path with small,
                        # steady corrections; never slam inside a hole pass
                        cap = min(cap, slowzone_max_command)
                    servo_cmd = np.clip(axis_map.apply(board_cmd), -cap, cap)
                    if not emergency and command_slew_per_s > 0.0:
                        if dt_s > 0.0:
                            # Fast-unwind a command only when the ball is
                            # actually moving (could overspeed). While stalled,
                            # let a breakaway kick HOLD its tilt so the servo
                            # has time to move the ball instead of twitching.
                            servo_cmd = slew_limit_command(
                                servo_cmd, prev_servo_cmd, dt_s, braking,
                                command_slew_per_s, brake_slew_per_s,
                                fast_reduce=speed_now > 15.0)
                        else:
                            # Burst/duplicate frame (dt=0): no real time has
                            # elapsed, so the command cannot legitimately change.
                            # Hold the previous one - sending the full un-slewed
                            # target here spiked the servo to ~0.5-0.7 for a
                            # single frame, which the servo cannot follow.
                            servo_cmd = prev_servo_cmd.copy()
                    prev_servo_cmd = servo_cmd.copy()
                    # Stall-kick indicator: how hard the breakaway kick is
                    # currently pushing a stuck ball (0 = not kicking). Watching
                    # this tells a stall being actively worked (KICK ramping,
                    # ball then moves) apart from a dead loop (KICK pinned high
                    # while progress never advances).
                    if mode == "carrot":
                        stall_kick_now = carrot_follower.kicker.last_kick
                    elif mode == "velocity":
                        stall_kick_now = velocity_follower.kicker.last_kick
                    else:
                        stall_kick_now = follower.kicker.last_kick
                    status = f"progress {progress:.0f}/{total_length:.0f} mm"
                    if hole_brake == "stabilize":
                        status += "  STABILIZING"
                    elif hole_brake == "emergency":
                        status += "  EMERGENCY BRAKE"
                    elif hole_brake == "unstick":
                        status += f"  UNSTICK {unstick_now:.2f}"
                    elif hole_brake == "slow":
                        status += "  hole ahead"
                    if stall_kick_now > 0.0 and hole_brake != "unstick":
                        status += f"  KICK {stall_kick_now:.2f}"
                    if float(np.linalg.norm(wall_escape)) > 1e-9:
                        status += "  wall escape"
                    if recovery_reason:
                        status += f"  A* recovery:{recovery_reason}"

                    if link is not None:
                        link.send(ServoCommand(yaw=float(servo_cmd[0]),
                                               pitch=float(servo_cmd[1])))
                    logger.write({
                        "timestamp_s": frame.timestamp_s, "found": True,
                        "x_mm": state.position_mm[0], "y_mm": state.position_mm[1],
                        "vx_mm_s": state.velocity_mm_s[0], "vy_mm_s": state.velocity_mm_s[1],
                        "target_x_mm": target[0], "target_y_mm": target[1],
                        "progress_mm": progress,
                        "carrot_x_mm": carrot_x, "carrot_y_mm": carrot_y,
                        "desired_vx_mm_s": v_des[0], "desired_vy_mm_s": v_des[1],
                        "cross_track_mm": cross, "turn_deg": turn_deg,
                        "wall_speed_scale": wall_scale,
                        "hole_brake": hole_brake,
                        "wall_distance_mm": wall_distance,
                        "target_speed_mm_s": target_speed,
                        "wall_escape_x": wall_escape[0], "wall_escape_y": wall_escape[1],
                        "board_cmd_x": board_cmd[0], "board_cmd_y": board_cmd[1],
                        "yaw_command": servo_cmd[0], "pitch_command": servo_cmd[1],
                        "stall_kick": stall_kick_now,
                    })

                    if progress >= total_length - args.goal_tolerance_mm:
                        outcome = "GOAL REACHED"
                        break
                else:
                    if link is not None:
                        link.neutral()
                    # Ball gone: don't let a stale timestamp/command survive
                    # the gap. A leftover big dt_s on reacquire would spike
                    # the stiction timer straight past its threshold and the
                    # slew limiter would jump from a stale nonzero command.
                    prev_timestamp_s = None
                    prev_servo_cmd = np.zeros(2)
                    recovery_low_speed_s = 0.0
                    stabilize_active = False
                    unstick_time = 0.0
                    pos_history.clear()
                    estimator.reset()
                    follower.reset()
                    velocity_follower.reset()
                    carrot_follower.reset()
                    logger.write({
                        "timestamp_s": frame.timestamp_s, "found": False,
                        "x_mm": "", "y_mm": "", "vx_mm_s": "", "vy_mm_s": "",
                        "target_x_mm": "", "target_y_mm": "", "progress_mm": "",
                        "carrot_x_mm": "", "carrot_y_mm": "",
                        "desired_vx_mm_s": "", "desired_vy_mm_s": "",
                        "cross_track_mm": "", "turn_deg": "",
                        "wall_speed_scale": "",
                        "hole_brake": "",
                        "wall_distance_mm": "",
                        "hole_hazard_distance_mm": "",
                        "hole_speed_cap_mm_s": "",
                        "wall_escape_x": "", "wall_escape_y": "",
                        "board_cmd_x": "", "board_cmd_y": "",
                        "yaw_command": 0.0, "pitch_command": 0.0,
                    })
                    if monotonic() - last_seen > args.lost_timeout_s:
                        outcome = "ball lost (fell in a hole?)"
                        break

                if not args.no_preview:
                    view = draw_overlay(frame.image, homography, path, holes,
                                        ball_px, target, servo_cmd, status)
                    cv2.imshow(WINDOW, view)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        outcome = "stopped by user"
                        break
        finally:
            if link is not None:
                link.neutral()
            cv2.destroyAllWindows()

    elapsed = monotonic() - start_time
    print(f"\nrun finished: {outcome} after {elapsed:.1f}s  (log: {args.log})")


if __name__ == "__main__":
    main()
