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
from cps_maze.hardware.serial_link import ArduinoServoLink, ServoCommand
from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.planning.hazards import HoleMap
from cps_maze.planning.path import WaypointPath
from cps_maze.planning.walls import WallMap
from cps_maze.vision.ball_pipeline import make_tracker
from cps_maze.vision.state_estimator import LowPassVelocityEstimator

WINDOW = "autonomous run"


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
    hole_horizon_mm = float(config.control.get("hole_horizon_mm", 80.0))
    hole_standoff_mm = float(config.control.get("hole_standoff_mm", 10.0))
    hole_brake_accel = float(config.control.get("hole_brake_accel_mm_s2", 250.0))
    hole_emergency = bool(config.control.get("hole_emergency_brake", True))
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
    ))
    print(f"controller: {mode}" + (
        f" (v_max {v_max:.0f} mm/s)" if mode in ("carrot", "velocity") else ""))
    estimator = LowPassVelocityEstimator()
    total_length = float(path.cumulative_lengths[-1])

    log_fields = [
        "timestamp_s", "found", "x_mm", "y_mm", "vx_mm_s", "vy_mm_s",
        "target_x_mm", "target_y_mm", "progress_mm",
        "carrot_x_mm", "carrot_y_mm", "desired_vx_mm_s", "desired_vy_mm_s",
        "cross_track_mm", "turn_deg", "wall_speed_scale", "hole_brake",
        "board_cmd_x", "board_cmd_y", "yaw_command", "pitch_command",
    ]

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
    prev_servo_cmd = np.zeros(2)  # for command-slew limiting
    outcome = "stopped by user"

    mouse_state: dict = {}
    if not args.no_preview:
        cv2.namedWindow(WINDOW)

        def on_mouse(event: int, x: int, y: int, *_rest) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_state["seed"] = (x, y)

        cv2.setMouseCallback(WINDOW, on_mouse)

    with CameraCapture(config.camera) as camera, serial_ctx as link, \
            CsvRunLogger(Path(args.log), log_fields) as logger:
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

                    # hole-aware speed cap: braking starts early enough by
                    # construction (v_allowed = sqrt(2 a d) toward the pass)
                    hole_brake = ""
                    hazard_d = hole_map.path_hazard_distance_mm(
                        path, progress, horizon_mm=hole_horizon_mm)
                    speed_cap = hole_map.speed_cap_mm_s(
                        hazard_d, hole_brake_accel, standoff_mm=hole_standoff_mm)
                    hole_scale = 1.0
                    if speed_cap is not None:
                        hole_scale = min(1.0, speed_cap / max(v_max, 1e-6))
                        if hole_scale < 0.999:
                            hole_brake = "slow"

                    if mode == "velocity":
                        path_point = path.point_at_progress_mm(progress)
                        tangent = path.tangent_at_progress_mm(progress)
                        turn_deg = path.heading_change_deg(
                            progress, span_mm=corner_span_mm,
                            noise_deg=corner_noise_deg)
                        wall_scale = (wall_map.speed_scale(board_xy)
                                      if wall_map is not None else 1.0)
                        board_cmd, v_des = velocity_follower.command(
                            state.position_mm, state.velocity_mm_s,
                            path_point, tangent, turn_deg, dt_s,
                            extra_speed_scale=min(wall_scale, hole_scale),
                        )
                        # overlay: aim marker a half-second of travel ahead
                        target = state.position_mm + 0.5 * v_des
                        carrot_x = ""
                        carrot_y = ""
                    elif mode == "carrot":
                        turn_deg = path.heading_change_deg(
                            progress, span_mm=corner_span_mm,
                            noise_deg=corner_noise_deg)
                        wall_scale = (wall_map.speed_scale(board_xy)
                                      if wall_map is not None else 1.0)
                        target, _carrot_lookahead = choose_carrot_point(
                            path, state.position_mm, progress,
                            carrot_lookahead_mm, carrot_min_lookahead_mm,
                            wall_map,
                        )
                        board_cmd, v_des = carrot_follower.command(
                            state.position_mm, state.velocity_mm_s,
                            target, turn_deg, dt_s,
                            extra_speed_scale=min(wall_scale, hole_scale),
                        )
                        carrot_x = target[0]
                        carrot_y = target[1]
                    else:
                        turn_deg = 0.0
                        wall_scale = 1.0
                        v_des = np.zeros(2)
                        target = path.point_at_progress_mm(progress + lookahead_mm)
                        carrot_x = ""
                        carrot_y = ""
                        board_cmd = follower.command(state.position_mm,
                                                     state.velocity_mm_s,
                                                     target, dt_s)
                    # Reactive layer: the trajectory enters a hole and the
                    # stopping distance exceeds the distance to it - normal
                    # control can no longer prevent the fall. Full brake
                    # opposite to the velocity, bypassing the slew limiter
                    # (an emergency cannot wait for a ramp).
                    emergency = False
                    speed_now = float(np.linalg.norm(state.velocity_mm_s))
                    if (hole_emergency and speed_now > 15.0
                            and hole_map.must_emergency_brake(
                                state.position_mm, state.velocity_mm_s,
                                hole_brake_accel)):
                        emergency = True
                        hole_brake = "emergency"
                        board_cmd = (-max_command / speed_now) * state.velocity_mm_s

                    servo_cmd = np.clip(axis_map.apply(board_cmd), -max_command, max_command)
                    if command_slew_per_s > 0.0 and dt_s > 0.0 and not emergency:
                        max_step = command_slew_per_s * dt_s
                        servo_cmd = prev_servo_cmd + np.clip(
                            servo_cmd - prev_servo_cmd, -max_step, max_step)
                    prev_servo_cmd = servo_cmd.copy()
                    status = f"progress {progress:.0f}/{total_length:.0f} mm"
                    if hole_brake == "emergency":
                        status += "  EMERGENCY BRAKE"
                    elif hole_brake == "slow":
                        status += "  hole ahead"

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
