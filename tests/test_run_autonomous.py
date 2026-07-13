import numpy as np

from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.planning.hazards import HoleMap
from cps_maze.planning.path import WaypointPath
from scripts.run_autonomous import (
    RUN_LOG_FIELDS,
    RecoveryAStarRuntime,
    associate_progress_guarded,
    choose_carrot_point,
    recovery_reason_for_state,
    should_apply_hole_emergency,
)


class XLimitWallMap:
    def __init__(self, max_clear_x: float):
        self.max_clear_x = max_clear_x

    def line_blocked(self, _a_mm: np.ndarray, b_mm: np.ndarray) -> bool:
        return bool(b_mm[0] > self.max_clear_x)


def test_choose_carrot_point_without_wall_map_uses_full_lookahead():
    path = WaypointPath(np.array([[0.0, 0.0], [50.0, 0.0]]))

    carrot, lookahead = choose_carrot_point(
        path=path,
        position_mm=np.array([0.0, 0.0]),
        progress_mm=0.0,
        lookahead_mm=30.0,
        min_lookahead_mm=10.0,
        wall_map=None,
    )

    assert np.allclose(carrot, [30.0, 0.0])
    assert np.isclose(lookahead, 30.0)


def test_choose_carrot_point_backs_down_to_clear_line_of_sight():
    path = WaypointPath(np.array([[0.0, 0.0], [50.0, 0.0]]))

    carrot, lookahead = choose_carrot_point(
        path=path,
        position_mm=np.array([0.0, 0.0]),
        progress_mm=0.0,
        lookahead_mm=30.0,
        min_lookahead_mm=10.0,
        wall_map=XLimitWallMap(max_clear_x=12.0),
        step_mm=5.0,
    )

    assert np.allclose(carrot, [10.0, 0.0])
    assert np.isclose(lookahead, 10.0)


def test_run_logger_accepts_lost_ball_row_with_hole_fields(tmp_path):
    log_path = tmp_path / "run.csv"
    row = {
        "timestamp_s": 1.25, "found": False,
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
    }

    with CsvRunLogger(log_path, RUN_LOG_FIELDS) as logger:
        logger.write(row)

    text = log_path.read_text(encoding="utf-8")
    assert "hole_hazard_distance_mm" in text
    assert "hole_speed_cap_mm_s" in text


def test_progress_jump_guard_rejects_implausible_one_frame_jump():
    path = WaypointPath(np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 10.0],
        [0.0, 10.0],
    ]))

    result = associate_progress_guarded(
        path,
        position_mm=np.array([80.0, 10.0]),
        progress_est_mm=50.0,
        wall_map=None,
        speed_mm_s=0.0,
        dt_s=0.024,
        high_cross_frames=0,
        max_progress_jump_mm=12.0,
        global_reacquire_cross_mm=30.0,
    )

    assert result.mode == "jump_guard"
    assert result.progress_mm == 50.0
    assert result.progress_delta_mm == 0.0


def test_recovery_reason_ignores_ordinary_on_route_stall():
    reason = recovery_reason_for_state(
        cross_track_mm=2.0,
        wall_distance_mm=20.0,
        speed_mm_s=0.0,
        low_speed_duration_s=5.0,
        recovery_goal_mm=np.array([20.0, 0.0]),
        state_position_mm=np.array([0.0, 0.0]),
        wall_map=None,
        cross_track_trigger_mm=25.0,
        wall_distance_trigger_mm=3.0,
        stall_speed_mm_s=5.0,
        stall_duration_s=1.2,
    )

    assert reason == ""


class FakeRecoveryPlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, start_mm: np.ndarray, _goal_mm: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.array([start_mm, start_mm + np.array([10.0, 0.0])])


def test_recovery_runtime_throttles_and_reuses_cached_target():
    planner = FakeRecoveryPlanner()
    nominal = WaypointPath(np.array([[0.0, 0.0], [100.0, 0.0]]))
    runtime = RecoveryAStarRuntime(
        planner,
        nominal,
        min_interval_s=0.25,
        cache_max_age_s=0.75,
        follow_mm=20.0,
        max_backtrack_mm=4.0,
        max_forward_mm=45.0,
        max_target_distance_mm=28.0,
    )
    ball = np.array([0.0, 0.0])
    goal = np.array([20.0, 0.0])

    first = runtime.target_for(0.0, ball, 0.0, goal, "offroute")
    second = runtime.target_for(0.10, ball, 0.0, goal, "offroute")
    third = runtime.target_for(0.30, ball, 0.0, goal, "offroute")

    assert first.active
    assert second.active
    assert third.active
    assert planner.calls == 2
    assert np.allclose(first.target_mm, second.target_mm)
    assert second.plan_ms == 0.0
    assert second.cache_age_ms == 100.0


def test_hole_clearance_override_brakes_even_when_route_aligned():
    holes = HoleMap(np.array([[60.0, 0.0, 8.0]]),
                    ball_radius_mm=6.0, margin_mm=4.0)
    pos = np.array([42.0, 0.0])       # capture edge for this hole map
    vel = np.array([30.0, 0.0])       # aligned with the route
    tangent = np.array([1.0, 0.0])

    assert should_apply_hole_emergency(
        holes,
        pos,
        vel,
        brake_accel_mm_s2=150.0,
        path_tangent=tangent,
        cross_track_mm=0.0,
        offroute_mm=8.0,
        align_deg=35.0,
        clearance_mm=holes.clearance_mm(pos),
        clearance_trigger_mm=2.0,
        clearance_speed_mm_s=25.0,
    )


def test_planned_pass_outside_clearance_override_still_avoids_emergency():
    holes = HoleMap(np.array([[60.0, -14.0, 8.0], [60.0, 14.0, 8.0]]),
                    ball_radius_mm=6.0, margin_mm=4.0)
    pos = np.array([30.0, 0.0])
    vel = np.array([50.0, 0.0])
    tangent = np.array([1.0, 0.0])

    assert not should_apply_hole_emergency(
        holes,
        pos,
        vel,
        brake_accel_mm_s2=150.0,
        path_tangent=tangent,
        cross_track_mm=2.0,
        offroute_mm=8.0,
        align_deg=35.0,
        clearance_mm=holes.clearance_mm(pos),
        clearance_trigger_mm=2.0,
        clearance_speed_mm_s=25.0,
    )
