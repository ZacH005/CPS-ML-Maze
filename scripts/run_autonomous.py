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
from dataclasses import dataclass
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
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand
from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.planning.hazards import HoleMap, should_emergency_brake
from cps_maze.planning.path import WaypointPath
from cps_maze.planning.recovery_astar import (
    RecoveryAStarConfig,
    RecoveryAStarPlanner,
    bounded_recovery_point_along_polyline,
)
from cps_maze.planning.speed_profile import build_speed_profile
from cps_maze.planning.walls import WallMap
from cps_maze.vision.ball_pipeline import make_tracker
from cps_maze.vision.state_estimator import LowPassVelocityEstimator

WINDOW = "autonomous run"

RUN_LOG_FIELDS = [
    "timestamp_s", "found", "x_mm", "y_mm", "vx_mm_s", "vy_mm_s",
    "target_x_mm", "target_y_mm", "progress_mm",
    "loop_dt_ms", "association_mode", "progress_delta_mm",
    "carrot_x_mm", "carrot_y_mm", "desired_vx_mm_s", "desired_vy_mm_s",
    "cross_track_mm", "turn_deg", "wall_speed_scale", "hole_brake",
    "wall_distance_mm", "target_speed_mm_s",
    "hole_hazard_distance_mm", "hole_speed_cap_mm_s",
    "hole_clearance_mm",
    "recovery_reason", "recovery_active", "recovery_plan_ms",
    "recovery_target_distance_mm", "recovery_cache_age_ms",
    "wall_escape_x", "wall_escape_y",
    "board_cmd_x", "board_cmd_y", "yaw_command", "pitch_command",
]


@dataclass(frozen=True)
class AssociationResult:
    progress_mm: float
    cross_track_mm: float
    progress_delta_mm: float
    mode: str
    high_cross_frames: int


@dataclass(frozen=True)
class RecoveryRuntimeResult:
    target_mm: np.ndarray | None
    active: bool
    plan_ms: float
    target_distance_mm: float | None
    cache_age_ms: float | None


class RecoveryAStarRuntime:
    """Throttles A* and reuses only bounded local recovery targets."""

    def __init__(
        self,
        planner: RecoveryAStarPlanner,
        nominal_path: WaypointPath,
        *,
        min_interval_s: float,
        cache_max_age_s: float,
        follow_mm: float,
        max_backtrack_mm: float,
        max_forward_mm: float,
        max_target_distance_mm: float,
    ):
        self.planner = planner
        self.nominal_path = nominal_path
        self.min_interval_s = max(float(min_interval_s), 0.0)
        self.cache_max_age_s = max(float(cache_max_age_s), 0.0)
        self.follow_mm = float(follow_mm)
        self.max_backtrack_mm = float(max_backtrack_mm)
        self.max_forward_mm = float(max_forward_mm)
        self.max_target_distance_mm = float(max_target_distance_mm)
        self._last_plan_s: float | None = None
        self._cached_at_s: float | None = None
        self._cached_target: np.ndarray | None = None

    def reset(self) -> None:
        self._last_plan_s = None
        self._cached_at_s = None
        self._cached_target = None

    def target_for(
        self,
        now_s: float,
        ball_position_mm: np.ndarray,
        current_progress_mm: float,
        goal_mm: np.ndarray,
        reason: str,
    ) -> RecoveryRuntimeResult:
        if not reason:
            self._cached_target = None
            self._cached_at_s = None
            return RecoveryRuntimeResult(None, False, 0.0, None, None)

        cached_age_s = (
            float(now_s) - self._cached_at_s
            if self._cached_at_s is not None else None
        )
        cache_valid = (
            self._cached_target is not None
            and cached_age_s is not None
            and cached_age_s <= self.cache_max_age_s
            and float(np.linalg.norm(self._cached_target - ball_position_mm))
            <= self.max_target_distance_mm
        )
        plan_age_s = (
            float(now_s) - self._last_plan_s
            if self._last_plan_s is not None else None
        )
        if cache_valid and plan_age_s is not None and plan_age_s < self.min_interval_s:
            return RecoveryRuntimeResult(
                self._cached_target.copy(),
                True,
                0.0,
                float(np.linalg.norm(self._cached_target - ball_position_mm)),
                cached_age_s * 1000.0,
            )

        start = monotonic()
        recovery_path = self.planner.plan(ball_position_mm, goal_mm)
        plan_ms = (monotonic() - start) * 1000.0
        self._last_plan_s = float(now_s)

        target = None
        if recovery_path is not None and len(recovery_path) >= 2:
            target = bounded_recovery_point_along_polyline(
                recovery_path,
                self.nominal_path,
                ball_position_mm,
                current_progress_mm,
                self.follow_mm,
                self.max_backtrack_mm,
                self.max_forward_mm,
                self.max_target_distance_mm,
            )
        if target is not None:
            self._cached_target = target
            self._cached_at_s = float(now_s)
            return RecoveryRuntimeResult(
                target.copy(),
                True,
                plan_ms,
                float(np.linalg.norm(target - ball_position_mm)),
                0.0,
            )

        if cache_valid:
            return RecoveryRuntimeResult(
                self._cached_target.copy(),
                True,
                plan_ms,
                float(np.linalg.norm(self._cached_target - ball_position_mm)),
                cached_age_s * 1000.0,
            )
        return RecoveryRuntimeResult(None, False, plan_ms, None, None)


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


def _best_path_projection(
    path: WaypointPath,
    position_mm: np.ndarray,
    progress_hint_mm: float | None,
    wall_map: WallMap | None,
) -> tuple[float, float]:
    if wall_map is None:
        return path.nearest_progress_and_distance_mm(position_mm, progress_hint_mm)

    cands = path.candidate_projections(position_mm, progress_hint_mm)
    clear = [c for c in cands if not wall_map.line_blocked(position_mm, c[2])]
    if not clear:
        clear = cands
    progress, cross, _projection = clear[0]
    return progress, cross


def associate_progress_guarded(
    path: WaypointPath,
    position_mm: np.ndarray,
    progress_est_mm: float | None,
    wall_map: WallMap | None,
    speed_mm_s: float,
    dt_s: float,
    high_cross_frames: int,
    *,
    max_progress_jump_mm: float,
    global_reacquire_cross_mm: float,
    global_reacquire_frames: int = 3,
) -> AssociationResult:
    """Associate ball position to route progress without one-frame corridor jumps."""
    if progress_est_mm is None:
        progress, cross = _best_path_projection(path, position_mm, None, wall_map)
        return AssociationResult(progress, cross, 0.0, "init", 0)

    progress, cross = _best_path_projection(path, position_mm, progress_est_mm, wall_map)
    next_high_cross = (
        high_cross_frames + 1
        if cross >= global_reacquire_cross_mm else 0
    )
    mode = "local"
    if next_high_cross >= max(int(global_reacquire_frames), 1):
        global_progress, global_cross = _best_path_projection(path, position_mm, None, wall_map)
        progress, cross = global_progress, global_cross
        next_high_cross = 0
        mode = "global_reacquire"

    delta = progress - float(progress_est_mm)
    max_jump = max(
        float(max_progress_jump_mm),
        2.5 * max(float(speed_mm_s), 0.0) * max(float(dt_s), 0.0) + 5.0,
    )
    if mode != "global_reacquire" and abs(delta) > max_jump:
        progress = float(progress_est_mm)
        cross = float(np.linalg.norm(
            np.asarray(position_mm, dtype=float)
            - path.point_at_progress_mm(progress)
        ))
        delta = 0.0
        mode = "jump_guard"

    return AssociationResult(progress, cross, delta, mode, next_high_cross)


def recovery_reason_for_state(
    cross_track_mm: float,
    wall_distance_mm: float,
    speed_mm_s: float,
    low_speed_duration_s: float,
    recovery_goal_mm: np.ndarray,
    state_position_mm: np.ndarray,
    wall_map: WallMap | None,
    *,
    cross_track_trigger_mm: float,
    wall_distance_trigger_mm: float,
    stall_speed_mm_s: float,
    stall_duration_s: float,
) -> str:
    if cross_track_mm >= cross_track_trigger_mm:
        return "offroute"
    if wall_map is not None and wall_map.line_blocked(state_position_mm, recovery_goal_mm):
        return "wall"
    offcenter_for_wall = max(8.0, 0.5 * cross_track_trigger_mm)
    if (wall_distance_mm <= wall_distance_trigger_mm
            and speed_mm_s < stall_speed_mm_s
            and low_speed_duration_s >= stall_duration_s
            and cross_track_mm >= offcenter_for_wall):
        return "wall"
    return ""


def should_apply_hole_emergency(
    hole_map: HoleMap,
    position_mm: np.ndarray,
    velocity_mm_s: np.ndarray,
    brake_accel_mm_s2: float,
    path_tangent: np.ndarray,
    cross_track_mm: float,
    *,
    offroute_mm: float,
    align_deg: float,
    clearance_mm: float,
    clearance_trigger_mm: float,
    clearance_speed_mm_s: float,
    min_trajectory_speed_mm_s: float = 15.0,
) -> bool:
    speed = float(np.linalg.norm(velocity_mm_s))
    if clearance_mm <= clearance_trigger_mm and speed > clearance_speed_mm_s:
        return True
    return (
        speed > min_trajectory_speed_mm_s
        and should_emergency_brake(
            hole_map,
            position_mm,
            velocity_mm_s,
            brake_accel_mm_s2,
            path_tangent=path_tangent,
            cross_track_mm=float(cross_track_mm),
            offroute_mm=offroute_mm,
            align_deg=align_deg,
        )
    )


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
    hole_brake_accel = float(config.control.get("hole_brake_accel_mm_s2", 150.0))
    hole_emergency = bool(config.control.get("hole_emergency_brake", True))
    hole_emergency_offroute_mm = float(config.control.get(
        "hole_emergency_offroute_mm", 8.0))
    hole_emergency_align_deg = float(config.control.get(
        "hole_emergency_align_deg", 35.0))
    hole_emergency_clearance_mm = float(config.control.get(
        "hole_emergency_clearance_mm", 2.0))
    hole_emergency_clearance_speed_mm_s = float(config.control.get(
        "hole_emergency_clearance_speed_mm_s", 25.0))
    wall_escape_distance_mm = float(config.control.get("wall_escape_distance_mm", 6.0))
    wall_escape_speed_mm_s = float(config.control.get("wall_escape_speed_mm_s", 8.0))
    wall_escape_command = float(config.control.get("wall_escape_command", 0.20))
    wall_escape_min_cmd = float(config.control.get("wall_escape_min_command", 0.08))
    recovery_enabled = bool(config.control.get("recovery_astar_enabled", True))
    recovery_cross_track_mm = float(config.control.get("recovery_astar_cross_track_mm", 25.0))
    recovery_wall_distance_mm = float(config.control.get("recovery_astar_wall_mm", 3.0))
    recovery_stall_speed_mm_s = float(config.control.get("recovery_astar_stall_speed_mm_s", 5.0))
    recovery_stall_duration_s = float(config.control.get("recovery_astar_stall_duration_s", 1.2))
    recovery_follow_mm = float(config.control.get("recovery_astar_follow_mm", 20.0))
    recovery_min_interval_s = float(config.control.get("recovery_astar_min_interval_s", 0.25))
    recovery_cache_max_age_s = float(config.control.get("recovery_astar_cache_max_age_s", 0.75))
    recovery_max_target_distance_mm = float(config.control.get(
        "recovery_astar_max_target_distance_mm", 28.0))
    recovery_max_forward_mm = float(config.control.get(
        "recovery_astar_max_target_progress_ahead_mm", 45.0))
    recovery_goal_lookahead_mm = float(config.control.get(
        "recovery_astar_goal_lookahead_mm", 35.0))
    recovery_max_backtrack_mm = float(config.control.get(
        "recovery_astar_max_backtrack_mm", 4.0))
    association_max_progress_jump_mm = float(config.control.get(
        "association_max_progress_jump_mm", 12.0))
    association_global_reacquire_cross_mm = float(config.control.get(
        "association_global_reacquire_cross_mm", 30.0))
    association_global_reacquire_frames = int(config.control.get(
        "association_global_reacquire_frames", 3))
    recovery_planner = None
    recovery_runtime = None
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
                    "recovery_astar_max_expansions", 3000)),
            ),
        )
        recovery_runtime = RecoveryAStarRuntime(
            recovery_planner,
            path,
            min_interval_s=recovery_min_interval_s,
            cache_max_age_s=recovery_cache_max_age_s,
            follow_mm=recovery_follow_mm,
            max_backtrack_mm=recovery_max_backtrack_mm,
            max_forward_mm=recovery_max_forward_mm,
            max_target_distance_mm=recovery_max_target_distance_mm,
        )
    # Braking authority: max_command is a DRIVING gentleness cap; stopping a
    # fast ball needs the full tilt range (the firmware still clamps).
    brake_max_command = float(config.control.get("brake_max_command", 1.0))
    brake_max_command = max(brake_max_command, max_command)
    brake_slew_per_s = float(config.control.get("brake_slew_per_s", 10.0))

    # One coherent speed PLAN for the whole route, computed once: hole
    # passes get a committed moderate speed (not a crawl), every slowdown
    # is reachable by braking (backward pass) and exits ramp smoothly
    # (forward pass). Replaces the per-frame reactive hole cap, whose
    # interaction with the stall kick caused the ball to "spazz" at
    # overlapping capture zones.
    profile = build_speed_profile(
        path, hole_map, wall_map,
        v_max_mm_s=v_max,
        hole_pass_mm_s=float(config.control.get("hole_pass_mm_s", 15.0)),
        hole_slow_band_mm=float(config.control.get("hole_slow_band_mm", 25.0)),
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
        brake_max_command=brake_max_command,
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
        brake_max_command=brake_max_command,
    ))
    print(f"controller: {mode}" + (
        f" (v_max {v_max:.0f} mm/s)" if mode in ("carrot", "velocity") else ""))
    estimator = LowPassVelocityEstimator()
    total_length = float(path.cumulative_lengths[-1])

    if args.dry_run:
        serial_ctx: contextlib.AbstractContextManager = contextlib.nullcontext()
        print("DRY RUN: servos disabled, visualization only")
    else:
        serial_ctx = ArduinoServoLink(
            port=args.port or config.serial["port"],
            baudrate=int(config.serial["baudrate"]),
            timeout_s=float(config.serial["timeout_s"]),
        )

    start_time = monotonic()
    last_seen = monotonic()
    prev_timestamp_s = None
    progress_est = None  # last known path progress; keeps association local
    association_high_cross_frames = 0
    prev_servo_cmd = np.zeros(2)  # for command-slew limiting
    recovery_low_speed_s = 0.0
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
                    association_high_cross_frames = 0
                    prev_timestamp_s = None
                    prev_servo_cmd = np.zeros(2)
                    recovery_low_speed_s = 0.0
                    if recovery_runtime is not None:
                        recovery_runtime.reset()
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
                    dt_s = (frame.timestamp_s - prev_timestamp_s
                            if prev_timestamp_s is not None else 0.0)
                    prev_timestamp_s = frame.timestamp_s
                    speed_now = float(np.linalg.norm(state.velocity_mm_s))
                    association = associate_progress_guarded(
                        path,
                        board_xy,
                        progress_est,
                        wall_map,
                        speed_now,
                        dt_s,
                        association_high_cross_frames,
                        max_progress_jump_mm=association_max_progress_jump_mm,
                        global_reacquire_cross_mm=association_global_reacquire_cross_mm,
                        global_reacquire_frames=association_global_reacquire_frames,
                    )
                    progress = association.progress_mm
                    cross = association.cross_track_mm
                    association_high_cross_frames = association.high_cross_frames
                    progress_delta = association.progress_delta_mm
                    association_mode = association.mode
                    progress_est = progress

                    # The route speed PLAN, computed once at startup: hole
                    # passes, walls, corners and braking feasibility are all
                    # already folded into one smooth profile. The controller
                    # just tracks the planned speed at this progress.
                    hole_brake = ""
                    target_speed = profile.speed_at(progress)
                    profile_scale = min(1.0, target_speed / max(v_max, 1e-6))
                    if profile_scale < 0.8:
                        hole_brake = "slow"
                    hole_clearance = hole_map.clearance_mm(state.position_mm)
                    wall_distance = (wall_map.wall_distance_mm(board_xy)
                                     if wall_map is not None else float("inf"))
                    if speed_now >= 2.0 * recovery_stall_speed_mm_s:
                        recovery_low_speed_s = 0.0
                    elif speed_now < recovery_stall_speed_mm_s and dt_s > 0.0:
                        recovery_low_speed_s += dt_s

                    # Runtime wall scale stays as OFF-route protection (the
                    # profile only knows centerline clearances); on-route the
                    # two agree, so min() introduces no discontinuity.
                    wall_scale = (wall_map.speed_scale(board_xy)
                                  if wall_map is not None else 1.0)
                    speed_scale = min(wall_scale, profile_scale)
                    recovery_reason = ""
                    recovery_active = False
                    recovery_plan_ms = 0.0
                    recovery_target_distance = None
                    recovery_cache_age_ms = None

                    if mode == "velocity":
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
                        if recovery_runtime is not None:
                            recovery_reason = recovery_reason_for_state(
                                float(cross), wall_distance, speed_now,
                                recovery_low_speed_s,
                                recovery_goal, state.position_mm, wall_map,
                                cross_track_trigger_mm=recovery_cross_track_mm,
                                wall_distance_trigger_mm=recovery_wall_distance_mm,
                                stall_speed_mm_s=recovery_stall_speed_mm_s,
                                stall_duration_s=recovery_stall_duration_s,
                            )
                            recovery_result = recovery_runtime.target_for(
                                frame.timestamp_s,
                                state.position_mm,
                                progress,
                                recovery_goal,
                                recovery_reason,
                            )
                            recovery_active = recovery_result.active
                            recovery_plan_ms = recovery_result.plan_ms
                            recovery_target_distance = recovery_result.target_distance_mm
                            recovery_cache_age_ms = recovery_result.cache_age_ms
                            if recovery_result.target_mm is not None:
                                target = recovery_result.target_mm
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
                        if recovery_runtime is not None:
                            recovery_reason = recovery_reason_for_state(
                                float(cross), wall_distance, speed_now,
                                recovery_low_speed_s,
                                recovery_goal, state.position_mm, wall_map,
                                cross_track_trigger_mm=recovery_cross_track_mm,
                                wall_distance_trigger_mm=recovery_wall_distance_mm,
                                stall_speed_mm_s=recovery_stall_speed_mm_s,
                                stall_duration_s=recovery_stall_duration_s,
                            )
                            recovery_result = recovery_runtime.target_for(
                                frame.timestamp_s,
                                state.position_mm,
                                progress,
                                recovery_goal,
                                recovery_reason,
                            )
                            recovery_active = recovery_result.active
                            recovery_plan_ms = recovery_result.plan_ms
                            recovery_target_distance = recovery_result.target_distance_mm
                            recovery_cache_age_ms = recovery_result.cache_age_ms
                            if recovery_result.target_mm is not None:
                                target = recovery_result.target_mm
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
                    if hole_emergency and should_apply_hole_emergency(
                        hole_map,
                        state.position_mm,
                        state.velocity_mm_s,
                        hole_brake_accel,
                        path_tangent=path.tangent_at_progress_mm(progress),
                        cross_track_mm=float(cross),
                        offroute_mm=hole_emergency_offroute_mm,
                        align_deg=hole_emergency_align_deg,
                        clearance_mm=hole_clearance,
                        clearance_trigger_mm=hole_emergency_clearance_mm,
                        clearance_speed_mm_s=hole_emergency_clearance_speed_mm_s,
                    ):
                        emergency = True
                        hole_brake = "emergency"
                        board_cmd = ((-brake_max_command / speed_now)
                                     * state.velocity_mm_s)

                    # a command opposing the motion is a brake: allow the
                    # full tilt range and a faster slew (stopping a fast
                    # ball cannot wait for the gentle driving ramp)
                    braking = emergency or (
                        speed_now > 20.0
                        and float(np.dot(board_cmd, state.velocity_mm_s)) < 0.0)
                    cap = brake_max_command if braking else max_command
                    servo_cmd = np.clip(axis_map.apply(board_cmd), -cap, cap)
                    slew = brake_slew_per_s if braking else command_slew_per_s
                    if slew > 0.0 and dt_s > 0.0 and not emergency:
                        max_step = slew * dt_s
                        servo_cmd = prev_servo_cmd + np.clip(
                            servo_cmd - prev_servo_cmd, -max_step, max_step)
                    prev_servo_cmd = servo_cmd.copy()
                    status = f"progress {progress:.0f}/{total_length:.0f} mm"
                    if hole_brake == "emergency":
                        status += "  EMERGENCY BRAKE"
                    elif hole_brake == "slow":
                        status += "  hole ahead"
                    if float(np.linalg.norm(wall_escape)) > 1e-9:
                        status += "  wall escape"
                    if recovery_reason:
                        recovery_state = "active" if recovery_active else "rejected"
                        status += f"  A* recovery:{recovery_reason}/{recovery_state}"

                    if link is not None:
                        link.send(ServoCommand(yaw=float(servo_cmd[0]),
                                               pitch=float(servo_cmd[1])))
                    logger.write({
                        "timestamp_s": frame.timestamp_s, "found": True,
                        "x_mm": state.position_mm[0], "y_mm": state.position_mm[1],
                        "vx_mm_s": state.velocity_mm_s[0], "vy_mm_s": state.velocity_mm_s[1],
                        "target_x_mm": target[0], "target_y_mm": target[1],
                        "progress_mm": progress,
                        "loop_dt_ms": 1000.0 * dt_s,
                        "association_mode": association_mode,
                        "progress_delta_mm": progress_delta,
                        "carrot_x_mm": carrot_x, "carrot_y_mm": carrot_y,
                        "desired_vx_mm_s": v_des[0], "desired_vy_mm_s": v_des[1],
                        "cross_track_mm": cross, "turn_deg": turn_deg,
                        "wall_speed_scale": wall_scale,
                        "hole_brake": hole_brake,
                        "wall_distance_mm": wall_distance,
                        "target_speed_mm_s": target_speed,
                        "hole_clearance_mm": hole_clearance,
                        "recovery_reason": recovery_reason,
                        "recovery_active": recovery_active,
                        "recovery_plan_ms": recovery_plan_ms,
                        "recovery_target_distance_mm": (
                            "" if recovery_target_distance is None
                            else recovery_target_distance
                        ),
                        "recovery_cache_age_ms": (
                            "" if recovery_cache_age_ms is None
                            else recovery_cache_age_ms
                        ),
                        "wall_escape_x": wall_escape[0], "wall_escape_y": wall_escape[1],
                        "board_cmd_x": board_cmd[0], "board_cmd_y": board_cmd[1],
                        "yaw_command": servo_cmd[0], "pitch_command": servo_cmd[1],
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
                    association_high_cross_frames = 0
                    if recovery_runtime is not None:
                        recovery_runtime.reset()
                    estimator.reset()
                    follower.reset()
                    velocity_follower.reset()
                    carrot_follower.reset()
                    logger.write({
                        "timestamp_s": frame.timestamp_s, "found": False,
                        "x_mm": "", "y_mm": "", "vx_mm_s": "", "vy_mm_s": "",
                        "target_x_mm": "", "target_y_mm": "", "progress_mm": "",
                        "loop_dt_ms": "", "association_mode": "",
                        "progress_delta_mm": "",
                        "carrot_x_mm": "", "carrot_y_mm": "",
                        "desired_vx_mm_s": "", "desired_vy_mm_s": "",
                        "cross_track_mm": "", "turn_deg": "",
                        "wall_speed_scale": "",
                        "hole_brake": "",
                        "wall_distance_mm": "",
                        "target_speed_mm_s": "",
                        "hole_hazard_distance_mm": "",
                        "hole_speed_cap_mm_s": "",
                        "hole_clearance_mm": "",
                        "recovery_reason": "", "recovery_active": "",
                        "recovery_plan_ms": "",
                        "recovery_target_distance_mm": "",
                        "recovery_cache_age_ms": "",
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
