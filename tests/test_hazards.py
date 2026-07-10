import numpy as np

from cps_maze.planning.hazards import HoleMap
from cps_maze.planning.path import WaypointPath


def _hole_map():
    # one hole of radius 8 at (60, 0); capture = 8 + 6 + 4 = 18 mm
    return HoleMap(np.array([[60.0, 0.0, 8.0]]), ball_radius_mm=6.0, margin_mm=4.0)


def test_path_hazard_distance_finds_hole_on_route():
    holes = _hole_map()
    path = WaypointPath(points_mm=np.array([[0.0, 0.0], [200.0, 0.0]]))

    d = holes.path_hazard_distance_mm(path, progress_mm=0.0, horizon_mm=80.0)

    assert d is not None
    assert 38.0 <= d <= 46.0  # capture zone starts at x = 60 - 18 = 42


def test_path_hazard_distance_clear_when_route_avoids_hole():
    holes = HoleMap(np.array([[60.0, 50.0, 8.0]]))  # far off the route
    path = WaypointPath(points_mm=np.array([[0.0, 0.0], [200.0, 0.0]]))

    assert holes.path_hazard_distance_mm(path, 0.0, horizon_mm=80.0) is None


def test_speed_cap_is_braking_physics():
    holes = _hole_map()
    # 50mm of usable distance at 250 mm/s^2: v = sqrt(2*250*50) = 158 mm/s
    cap = holes.speed_cap_mm_s(60.0, brake_accel_mm_s2=250.0, standoff_mm=10.0)
    assert np.isclose(cap, np.sqrt(2 * 250.0 * 50.0), rtol=0.01)
    # inside the standoff: capped at the crawl floor
    cap_close = holes.speed_cap_mm_s(5.0, brake_accel_mm_s2=250.0, standoff_mm=10.0)
    assert cap_close == 8.0


def test_trajectory_hazard_detects_incoming_ball():
    holes = _hole_map()
    hit = holes.trajectory_hazard(np.array([0.0, 0.0]), np.array([100.0, 0.0]))
    assert hit is not None
    t_entry, d_entry = hit
    assert np.isclose(d_entry, 42.0, atol=2.0)   # entry at capture edge
    assert np.isclose(t_entry, 0.42, atol=0.03)

    # moving away: no hazard
    assert holes.trajectory_hazard(np.array([0.0, 0.0]),
                                   np.array([-100.0, 0.0])) is None
    # passing wide of the capture zone: no hazard
    assert holes.trajectory_hazard(np.array([0.0, 25.0]),
                                   np.array([100.0, 0.0])) is None


def test_emergency_brake_only_when_stopping_distance_insufficient():
    holes = _hole_map()
    pos = np.array([20.0, 0.0])  # 22mm before the capture edge at x=42

    # slow ball: stopping distance 8^2/(2*250) = 0.13mm - controllable
    assert not holes.must_emergency_brake(pos, np.array([8.0, 0.0]), 250.0)

    # fast ball: stopping distance 150^2/500 = 45mm > 22mm - cannot stop
    assert holes.must_emergency_brake(pos, np.array([150.0, 0.0]), 250.0)
