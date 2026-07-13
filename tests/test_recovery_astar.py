import numpy as np

from cps_maze.planning.hazards import HoleMap
from cps_maze.planning.recovery_astar import (
    RecoveryAStarConfig,
    RecoveryAStarPlanner,
    bounded_recovery_point_along_polyline,
    point_along_polyline,
    progress_limited_point_along_polyline,
)
from cps_maze.planning.path import WaypointPath
from cps_maze.planning.walls import WallMap


def test_recovery_astar_routes_through_wall_gap():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[:, 30] = 1
    mask[25:36, 30] = 0
    wall_map = WallMap(mask, origin_mm=np.array([0.0, 0.0]), scale_px_per_mm=1.0)
    planner = RecoveryAStarPlanner(
        wall_map,
        config=RecoveryAStarConfig(grid_mm=2.0, wall_clearance_mm=1.0),
    )

    path = planner.plan(np.array([10.0, 10.0]), np.array([50.0, 10.0]))

    assert path is not None
    near_gap = path[np.abs(path[:, 0] - 30.0) <= 2.0]
    assert len(near_gap) > 0
    assert np.any((25.0 <= near_gap[:, 1]) & (near_gap[:, 1] <= 35.0))


def test_recovery_astar_avoids_inflated_hole_zone():
    mask = np.zeros((60, 60), dtype=np.uint8)
    wall_map = WallMap(mask, origin_mm=np.array([0.0, -30.0]), scale_px_per_mm=1.0)
    holes = HoleMap(np.array([[30.0, 0.0, 3.0]]), ball_radius_mm=3.0, margin_mm=2.0)
    planner = RecoveryAStarPlanner(
        wall_map,
        holes,
        RecoveryAStarConfig(grid_mm=2.0, wall_clearance_mm=0.0, hole_clearance_mm=0.0),
    )

    path = planner.plan(np.array([5.0, 0.0]), np.array([55.0, 0.0]))

    assert path is not None
    clearances = [holes.clearance_mm(p) for p in path]
    assert min(clearances) >= 0.0
    assert np.max(np.abs(path[:, 1])) > 7.0


def test_point_along_polyline_clamps_to_end():
    path = np.array([[0.0, 0.0], [3.0, 4.0], [13.0, 4.0]])

    assert np.allclose(point_along_polyline(path, 2.5), [1.5, 2.0])
    assert np.allclose(point_along_polyline(path, 99.0), [13.0, 4.0])


def test_progress_limited_recovery_target_skips_backtracking_waypoint():
    nominal = WaypointPath(points_mm=np.array([[0.0, 0.0], [100.0, 0.0]]))
    recovery = np.array([
        [50.0, 0.0],
        [35.0, 8.0],
        [45.0, 12.0],
        [70.0, 0.0],
    ])

    target = progress_limited_point_along_polyline(
        recovery,
        nominal,
        current_progress_mm=50.0,
        distance_mm=16.0,
        max_backtrack_mm=4.0,
        sample_mm=2.0,
    )
    progress, _cross = nominal.nearest_progress_and_distance_mm(target)

    assert progress >= 46.0
    assert target[0] > 45.0


def test_bounded_recovery_rejects_far_corridor_jump():
    nominal = WaypointPath(points_mm=np.array([
        [72.9, 52.0],
        [74.4, 29.5],
        [69.9, 22.5],
        [35.4, 24.0],
        [24.4, 74.5],
        [50.4, 76.5],
    ]))
    ball = np.array([47.6, 22.4])
    recovery = np.array([
        ball,
        [82.1, 84.0],
    ])

    target = bounded_recovery_point_along_polyline(
        recovery,
        nominal,
        ball,
        current_progress_mm=80.0,
        distance_mm=20.0,
        max_backtrack_mm=4.0,
        max_forward_mm=45.0,
        max_target_distance_mm=28.0,
    )

    assert target is None


def test_bounded_recovery_rejects_when_all_candidates_are_too_far():
    nominal = WaypointPath(points_mm=np.array([[0.0, 0.0], [100.0, 0.0]]))
    ball = np.array([10.0, 0.0])
    recovery = np.array([[10.0, 0.0], [80.0, 0.0]])

    target = bounded_recovery_point_along_polyline(
        recovery,
        nominal,
        ball,
        current_progress_mm=10.0,
        distance_mm=35.0,
        max_target_distance_mm=20.0,
    )

    assert target is None


def test_bounded_recovery_accepts_nearby_sideways_target():
    nominal = WaypointPath(points_mm=np.array([[0.0, 0.0], [100.0, 0.0]]))
    ball = np.array([50.0, 4.0])
    recovery = np.array([[50.0, 4.0], [58.0, 9.0], [70.0, 0.0]])

    target = bounded_recovery_point_along_polyline(
        recovery,
        nominal,
        ball,
        current_progress_mm=50.0,
        distance_mm=9.0,
        max_backtrack_mm=4.0,
        max_forward_mm=45.0,
        max_target_distance_mm=28.0,
    )

    assert target is not None
    assert float(np.linalg.norm(target - ball)) <= 28.0
