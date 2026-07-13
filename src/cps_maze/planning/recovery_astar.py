from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from cps_maze.planning.hazards import HoleMap
from cps_maze.planning.path import WaypointPath
from cps_maze.planning.walls import WallMap


@dataclass(frozen=True)
class RecoveryAStarConfig:
    grid_mm: float = 4.0
    wall_clearance_mm: float = 2.0
    hole_clearance_mm: float = 0.0
    max_snap_mm: float = 18.0
    max_expansions: int = 8000


class RecoveryAStarPlanner:
    """Local grid A* planner for off-route/stuck recovery.

    This is intentionally a short-horizon recovery tool, not the main route
    planner. It plans from the measured ball position to the current route
    marker while treating the static walls and inflated hole capture zones as
    blocked space.
    """

    def __init__(
        self,
        wall_map: WallMap,
        hole_map: HoleMap | None = None,
        config: RecoveryAStarConfig = RecoveryAStarConfig(),
    ):
        self.wall_map = wall_map
        self.hole_map = hole_map
        self.config = config
        h, w = wall_map.mask.shape
        size_mm = np.array([w, h], dtype=float) / wall_map.scale
        self._min_cell = (0, 0)
        self._max_cell = tuple(np.floor(size_mm / config.grid_mm).astype(int))

    def plan(self, start_mm: np.ndarray, goal_mm: np.ndarray) -> np.ndarray | None:
        start = self._nearest_free_cell(self._to_cell(start_mm))
        goal = self._nearest_free_cell(self._to_cell(goal_mm))
        if start is None or goal is None:
            return None
        if start == goal:
            return np.array([self._to_mm(start), np.asarray(goal_mm, dtype=float)])

        frontier: list[tuple[float, float, tuple[int, int]]] = []
        heapq.heappush(frontier, (self._heuristic(start, goal), 0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
        expansions = 0

        while frontier and expansions < self.config.max_expansions:
            _priority, _cost, current = heapq.heappop(frontier)
            if current == goal:
                return self._smooth(self._reconstruct(came_from, current), goal_mm)
            expansions += 1

            for nxt, step_cost in self._neighbors(current):
                new_cost = cost_so_far[current] + step_cost
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self._heuristic(nxt, goal)
                    heapq.heappush(frontier, (priority, new_cost, nxt))
                    came_from[nxt] = current
        return None

    def _to_cell(self, p_mm: np.ndarray) -> tuple[int, int]:
        q = (np.asarray(p_mm, dtype=float) - self.wall_map.origin_mm) / self.config.grid_mm
        return int(round(q[0])), int(round(q[1]))

    def _to_mm(self, cell: tuple[int, int]) -> np.ndarray:
        return self.wall_map.origin_mm + self.config.grid_mm * np.asarray(cell, dtype=float)

    def _in_bounds(self, cell: tuple[int, int]) -> bool:
        return (self._min_cell[0] <= cell[0] <= self._max_cell[0]
                and self._min_cell[1] <= cell[1] <= self._max_cell[1])

    def _blocked(self, cell: tuple[int, int]) -> bool:
        if not self._in_bounds(cell):
            return True
        p = self._to_mm(cell)
        if self.wall_map.is_wall(p):
            return True
        if self.wall_map.wall_distance_mm(p) < self.config.wall_clearance_mm:
            return True
        if (self.hole_map is not None
                and self.hole_map.clearance_mm(p) < self.config.hole_clearance_mm):
            return True
        return False

    def _nearest_free_cell(self, origin: tuple[int, int]) -> tuple[int, int] | None:
        if not self._blocked(origin):
            return origin
        max_radius = max(int(np.ceil(self.config.max_snap_mm / self.config.grid_mm)), 1)
        best: tuple[float, tuple[int, int]] | None = None
        ox, oy = origin
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    cell = (ox + dx, oy + dy)
                    if self._blocked(cell):
                        continue
                    dist = float(np.hypot(dx, dy))
                    if best is None or dist < best[0]:
                        best = (dist, cell)
            if best is not None:
                return best[1]
        return None

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        out: list[tuple[tuple[int, int], float]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nxt = (cell[0] + dx, cell[1] + dy)
                if self._blocked(nxt):
                    continue
                # Do not cut diagonally through a blocked corner.
                if dx != 0 and dy != 0:
                    if self._blocked((cell[0] + dx, cell[1])) or self._blocked((cell[0], cell[1] + dy)):
                        continue
                out.append((nxt, self.config.grid_mm * float(np.hypot(dx, dy))))
        return out

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return self.config.grid_mm * float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        current: tuple[int, int],
    ) -> np.ndarray:
        cells = [current]
        while came_from[current] is not None:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        return np.array([self._to_mm(cell) for cell in cells], dtype=float)

    def _line_free(self, a_mm: np.ndarray, b_mm: np.ndarray) -> bool:
        a = np.asarray(a_mm, dtype=float)
        b = np.asarray(b_mm, dtype=float)
        length = float(np.linalg.norm(b - a))
        steps = max(int(length / max(self.config.grid_mm * 0.5, 1e-6)), 1)
        for i in range(steps + 1):
            p = a + (b - a) * (i / steps)
            if self._blocked(self._to_cell(p)):
                return False
        return True

    def _smooth(self, path_mm: np.ndarray, goal_mm: np.ndarray) -> np.ndarray:
        if len(path_mm) <= 2:
            return path_mm
        out = [path_mm[0]]
        i = 0
        while i < len(path_mm) - 1:
            j = len(path_mm) - 1
            while j > i + 1 and not self._line_free(path_mm[i], path_mm[j]):
                j -= 1
            out.append(path_mm[j])
            i = j
        goal = np.asarray(goal_mm, dtype=float)
        if self._line_free(out[-1], goal):
            out[-1] = goal
        return np.array(out, dtype=float)


def point_along_polyline(points_mm: np.ndarray, distance_mm: float) -> np.ndarray:
    """Point ``distance_mm`` along a polyline, clamped to its end."""
    points = np.asarray(points_mm, dtype=float)
    if len(points) == 0:
        raise ValueError("points_mm must not be empty")
    remaining = max(float(distance_mm), 0.0)
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1e-9:
            continue
        if remaining <= length:
            return start + (remaining / length) * segment
        remaining -= length
    return points[-1]


def progress_limited_point_along_polyline(
    points_mm: np.ndarray,
    nominal_path: WaypointPath,
    current_progress_mm: float,
    distance_mm: float,
    max_backtrack_mm: float = 4.0,
    sample_mm: float = 2.0,
    projection_window_mm: float = 120.0,
) -> np.ndarray:
    """Pick a recovery waypoint without sending the ball far backward.

    A* may need an initial sideways/backward move around a wall pixel, but the
    visible target should not jump to an earlier route section. Sample forward
    along the A* polyline until the point projects no farther back than
    ``current_progress_mm - max_backtrack_mm`` on the nominal route.
    """
    points = np.asarray(points_mm, dtype=float)
    if len(points) == 0:
        raise ValueError("points_mm must not be empty")
    if len(points) == 1:
        return points[0]

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(segment_lengths))
    start_d = min(max(float(distance_mm), 0.0), total)
    step = max(float(sample_mm), 1e-6)
    min_progress = float(current_progress_mm) - max(float(max_backtrack_mm), 0.0)

    distances = list(np.arange(start_d, total + step, step))
    if not distances or distances[-1] < total:
        distances.append(total)
    for d in distances:
        point = point_along_polyline(points, min(float(d), total))
        progress, _cross = nominal_path.nearest_progress_and_distance_mm(
            point,
            near_progress_mm=current_progress_mm,
            window_mm=projection_window_mm,
        )
        if progress >= min_progress:
            return point
    return points[-1]


def bounded_recovery_point_along_polyline(
    points_mm: np.ndarray,
    nominal_path: WaypointPath,
    ball_position_mm: np.ndarray,
    current_progress_mm: float,
    distance_mm: float,
    max_backtrack_mm: float = 4.0,
    max_forward_mm: float = 45.0,
    max_target_distance_mm: float = 28.0,
    sample_mm: float = 2.0,
    projection_window_mm: float = 120.0,
) -> np.ndarray | None:
    """Pick a local recovery waypoint, or None if A* only offers bad targets.

    Recovery A* is allowed to find a path through nearby free space, but the
    visible controller target must remain local to the ball and local in route
    progress. Returning None is deliberate: normal path following is safer than
    chasing a recovery point on a different corridor.
    """
    points = np.asarray(points_mm, dtype=float)
    if len(points) == 0:
        raise ValueError("points_mm must not be empty")

    ball = np.asarray(ball_position_mm, dtype=float)
    if len(points) == 1:
        candidate = points[0]
        if float(np.linalg.norm(candidate - ball)) > max_target_distance_mm:
            return None
        progress, _cross = nominal_path.nearest_progress_and_distance_mm(
            candidate,
            near_progress_mm=current_progress_mm,
            window_mm=projection_window_mm,
        )
        min_progress = float(current_progress_mm) - max(float(max_backtrack_mm), 0.0)
        max_progress = float(current_progress_mm) + max(float(max_forward_mm), 0.0)
        return candidate if min_progress <= progress <= max_progress else None

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(np.sum(segment_lengths))
    start_d = min(max(float(distance_mm), 0.0), total)
    step = max(float(sample_mm), 1e-6)
    min_progress = float(current_progress_mm) - max(float(max_backtrack_mm), 0.0)
    max_progress = float(current_progress_mm) + max(float(max_forward_mm), 0.0)
    max_distance = max(float(max_target_distance_mm), 0.0)

    distances = list(np.arange(start_d, total + step, step))
    if not distances or distances[-1] < total:
        distances.append(total)
    for d in distances:
        point = point_along_polyline(points, min(float(d), total))
        if float(np.linalg.norm(point - ball)) > max_distance:
            continue
        progress, _cross = nominal_path.nearest_progress_and_distance_mm(
            point,
            near_progress_mm=current_progress_mm,
            window_mm=projection_window_mm,
        )
        if min_progress <= progress <= max_progress:
            return point
    return None
