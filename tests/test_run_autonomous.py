import numpy as np

from cps_maze.logging.run_logger import CsvRunLogger
from cps_maze.planning.path import WaypointPath
from scripts.run_autonomous import (
    RUN_LOG_FIELDS,
    choose_carrot_point,
    slew_limit_command,
    unstick_bias_command,
    unstick_is_stuck,
)


def test_unstick_bias_is_not_full_open_loop_push():
    # A pure open-loop push would be exactly magnitude*direction regardless of
    # velocity. With the ball already moving toward the target, the damping term
    # must REDUCE the forward push (so it can't keep accelerating the ball).
    to_target = np.array([10.0, 0.0])
    still = unstick_bias_command(0.6, to_target, np.array([0.0, 0.0]),
                                 damping=0.02, cap=1.0)
    moving = unstick_bias_command(0.6, to_target, np.array([20.0, 0.0]),
                                  damping=0.02, cap=1.0)
    assert np.isclose(still[0], 0.6)              # from rest: full push
    assert moving[0] < still[0]                    # moving: damped, less push
    # and at high speed in the push direction it reverses to a brake
    fast = unstick_bias_command(0.6, to_target, np.array([60.0, 0.0]),
                                damping=0.02, cap=1.0)
    assert fast[0] < 0.0


def test_unstick_bias_is_magnitude_capped():
    to_target = np.array([10.0, 0.0])
    bias = unstick_bias_command(0.6, to_target, np.array([0.0, 100.0]),
                                damping=0.02, cap=0.55)
    assert np.linalg.norm(bias) <= 0.55 + 1e-9


def test_unstick_bias_zero_when_no_direction_or_magnitude():
    z = unstick_bias_command(0.6, np.array([0.0, 0.0]), np.array([1.0, 1.0]),
                             damping=0.02, cap=1.0)
    assert np.allclose(z, [0.0, 0.0])
    z2 = unstick_bias_command(0.0, np.array([10.0, 0.0]), np.array([0.0, 0.0]),
                              damping=0.02, cap=1.0)
    assert np.allclose(z2, [0.0, 0.0])


def test_unstick_not_triggered_by_slow_but_steady_progress():
    # Near a hole the plan is slow (12 mm/s), so expected travel is ~12 mm and
    # the progress threshold is 0.35*12 = 4.2 mm. A ball making steady net
    # progress of 5 mm/window is progressing, NOT stuck - the old fixed 6 mm
    # threshold wrongly flagged this and launched the ball into the hole.
    assert not unstick_is_stuck(net_disp_mm=5.0, span_s=1.0, window_s=1.0,
                                target_speed_mm_s=12.0, dist_mm=6.0,
                                progress_frac=0.35)
    # A ball that barely moved (2 mm) at the same plan IS stuck.
    assert unstick_is_stuck(net_disp_mm=2.0, span_s=1.0, window_s=1.0,
                            target_speed_mm_s=12.0, dist_mm=6.0,
                            progress_frac=0.35)


def test_unstick_requires_filled_window_and_moving_plan():
    # window not yet filled -> not stuck
    assert not unstick_is_stuck(0.0, span_s=0.3, window_s=1.0,
                                target_speed_mm_s=12.0, dist_mm=6.0,
                                progress_frac=0.35)
    # plan wants ~no motion -> unstick must not fire
    assert not unstick_is_stuck(0.0, span_s=1.0, window_s=1.0,
                                target_speed_mm_s=1.0, dist_mm=6.0,
                                progress_frac=0.35)


def test_unstick_threshold_capped_by_dist_mm_in_fast_zones():
    # In a fast zone (25 mm/s) the progress threshold (0.35*25 = 8.75 mm) is
    # capped at dist_mm=6, so a ball that moved 7 mm is NOT stuck.
    assert not unstick_is_stuck(net_disp_mm=7.0, span_s=1.0, window_s=1.0,
                                target_speed_mm_s=25.0, dist_mm=6.0,
                                progress_frac=0.35)


def test_slew_limits_increasing_drive_to_slow_rate():
    # increasing magnitude uses the slow rate: 0 -> at most slow*dt in one step
    out = slew_limit_command(
        np.array([0.5, 0.0]), np.array([0.0, 0.0]), dt_s=0.016,
        braking=False, slow_per_s=1.5, fast_per_s=12.0)
    assert np.isclose(out[0], 1.5 * 0.016)  # 0.024, not the full 0.5


def test_stalled_kick_tilt_is_not_fast_collapsed():
    # A breakaway kick that drops (reducing magnitude) must HOLD when the ball
    # is stalled (fast_reduce=False): it unwinds at the slow rate, not fast.
    prev = np.array([0.55, 0.0])
    target = np.array([0.08, 0.0])  # kick toggled off
    slow = slew_limit_command(target, prev, dt_s=0.016, braking=False,
                              slow_per_s=1.5, fast_per_s=12.0, fast_reduce=False)
    fast = slew_limit_command(target, prev, dt_s=0.016, braking=False,
                              slow_per_s=1.5, fast_per_s=12.0, fast_reduce=True)
    # stalled: barely drops (holds the tilt); moving: collapses much faster
    assert np.isclose(slow[0], 0.55 - 1.5 * 0.016)
    assert fast[0] < slow[0] - 0.1


def test_braking_always_uses_fast_rate():
    # a command opposing motion (braking) still unwinds fast regardless
    out = slew_limit_command(
        np.array([0.0, 0.0]), np.array([0.5, 0.0]), dt_s=0.016,
        braking=True, slow_per_s=1.5, fast_per_s=12.0, fast_reduce=False)
    assert np.isclose(out[0], 0.5 - 12.0 * 0.016)


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
