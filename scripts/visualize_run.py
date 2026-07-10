#!/usr/bin/env python3
"""Visualize why an autonomous run slowed, braked, or stalled.

Offline diagnostic overlays for `data/raw/autonomous_run.csv`:

- path progress and recent trail
- holes: physical radius and expanded capture radius
- wall mask contours and wall-speed slowdown
- target/carrot, desired velocity, measured velocity, and board command
- active decisions: hole slowdown/emergency, wall slowdown, corner slowdown,
  braking, and commanded-but-stalled state

By default it writes PNG stills for the longest stall episodes. Pass
`--out-video ...mp4` to render a time-compressed replay too.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.config import load_config
from cps_maze.planning.hazards import HoleMap
from cps_maze.planning.path import WaypointPath
from cps_maze.planning.walls import WallMap


def load_holes(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3))
    rows = np.genfromtxt(path, delimiter=",", names=True)
    rows = np.atleast_1d(rows)
    return np.column_stack([rows["x_mm"], rows["y_mm"], rows["radius_mm"]]).astype(float)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [row for row in csv.DictReader(f)]


def f(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def found(row: dict) -> bool:
    return str(row.get("found", "")).lower() in ("true", "1")


def put_text(img: np.ndarray, text: str, org: tuple[int, int],
             color: tuple[int, int, int] = (255, 255, 255),
             scale: float = 0.55) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_arrow_mm(
    img: np.ndarray,
    homography: Homography,
    start_mm: np.ndarray,
    vector_mm: np.ndarray,
    color: tuple[int, int, int],
    label: str,
    scale: float = 1.0,
) -> None:
    if float(np.linalg.norm(vector_mm)) < 1e-9:
        return
    a = homography.board_point_to_image_px(float(start_mm[0]), float(start_mm[1]))
    b_mm = start_mm + scale * vector_mm
    b = homography.board_point_to_image_px(float(b_mm[0]), float(b_mm[1]))
    a_i = tuple(np.round(a).astype(int))
    b_i = tuple(np.round(b).astype(int))
    cv2.arrowedLine(img, a_i, b_i, color, 2, cv2.LINE_AA, tipLength=0.22)
    put_text(img, label, (b_i[0] + 5, b_i[1] - 5), color, 0.45)


def draw_translucent_circle(
    img: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = img.copy()
    cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img)


@dataclass
class StallEpisode:
    start_idx: int
    end_idx: int
    duration_s: float
    start_t_s: float
    progress_mm: float
    cross_track_mm: float


def stall_episodes(
    rows: list[dict],
    stall_speed_mm_s: float,
    stall_cmd: float,
) -> list[StallEpisode]:
    tracked = [(i, r) for i, r in enumerate(rows) if found(r)]
    if not tracked:
        return []
    t0 = f(tracked[0][1], "timestamp_s")
    stalled_flags = []
    for _idx, row in tracked:
        speed = float(np.hypot(f(row, "vx_mm_s"), f(row, "vy_mm_s")))
        cmd = float(np.hypot(f(row, "yaw_command"), f(row, "pitch_command")))
        stalled_flags.append(speed < stall_speed_mm_s and cmd > stall_cmd)

    out: list[StallEpisode] = []
    start = None
    for j, is_stalled in enumerate(stalled_flags):
        if is_stalled and start is None:
            start = j
        elif not is_stalled and start is not None:
            start_idx, start_row = tracked[start]
            end_idx, end_row = tracked[j - 1]
            duration = f(end_row, "timestamp_s") - f(start_row, "timestamp_s")
            if duration >= 0.4:
                out.append(StallEpisode(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    duration_s=duration,
                    start_t_s=f(start_row, "timestamp_s") - t0,
                    progress_mm=f(start_row, "progress_mm"),
                    cross_track_mm=f(start_row, "cross_track_mm"),
                ))
            start = None
    if start is not None:
        start_idx, start_row = tracked[start]
        end_idx, end_row = tracked[-1]
        duration = f(end_row, "timestamp_s") - f(start_row, "timestamp_s")
        if duration >= 0.4:
            out.append(StallEpisode(
                start_idx=start_idx,
                end_idx=end_idx,
                duration_s=duration,
                start_t_s=f(start_row, "timestamp_s") - t0,
                progress_mm=f(start_row, "progress_mm"),
                cross_track_mm=f(start_row, "cross_track_mm"),
            ))
    return sorted(out, key=lambda ep: -ep.duration_s)


class RunVisualizer:
    def __init__(
        self,
        config_path: Path,
        homography_path: Path,
        path_override: Path | None,
        holes_path: Path,
        wall_mask_path: Path,
        background_path: Path | None,
    ):
        self.config = load_config(config_path)
        self.homography = Homography.load(homography_path)
        path_file = path_override if path_override else self.config.resolve_path(
            self.config.maze["path_file"])
        self.path = WaypointPath.from_csv(path_file)
        self.holes = load_holes(holes_path)
        self.hole_map = HoleMap(
            self.holes,
            ball_radius_mm=float(self.config.control.get("ball_radius_mm", 6.0)),
            margin_mm=float(self.config.control.get("hole_margin_mm", 4.0)),
        )
        self.wall_map = WallMap.load(wall_mask_path) if wall_mask_path.exists() else None
        self.background = self._load_background(background_path)
        self.total_mm = float(self.path.cumulative_lengths[-1])
        self.v_max = float(self.config.control.get("v_max_mm_s", 45.0))
        self.min_speed_frac = float(self.config.control.get("min_speed_frac", 0.25))
        self.corner_slow_deg = float(self.config.control.get("corner_slow_deg", 110.0))
        self.hole_horizon_mm = float(self.config.control.get("hole_horizon_mm", 80.0))
        self.hole_standoff_mm = float(self.config.control.get("hole_standoff_mm", 10.0))
        self.hole_brake_accel = float(self.config.control.get("hole_brake_accel_mm_s2", 250.0))
        self.static_base = self.background.copy()
        self.draw_static(self.static_base)

    def _load_background(self, explicit: Path | None) -> np.ndarray:
        candidates = []
        if explicit is not None:
            candidates.append(explicit)
        candidates.extend([
            Path("calibration/live_roi_source.png"),
            Path("calibration/CURRENT_FIXED_CAMERA_VIEW.png"),
        ])
        for candidate in candidates:
            if candidate.exists():
                img = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        return np.full((800, 1280, 3), 235, dtype=np.uint8)

    def mm_to_px(self, point: np.ndarray) -> tuple[int, int]:
        px = self.homography.board_point_to_image_px(float(point[0]), float(point[1]))
        return tuple(np.round(px).astype(int))

    def path_poly_px(self, start_mm: float | None = None,
                     end_mm: float | None = None,
                     step_mm: float = 3.0) -> np.ndarray:
        if start_mm is None:
            points = self.path.points_mm
        else:
            end = self.total_mm if end_mm is None else min(float(end_mm), self.total_mm)
            start = max(0.0, float(start_mm))
            if end < start:
                end = start
            samples = np.arange(start, end + step_mm, step_mm)
            if len(samples) == 0 or samples[-1] < end:
                samples = np.append(samples, end)
            points = np.array([self.path.point_at_progress_mm(s) for s in samples])
        return self.homography.board_points_to_image_px(points).astype(np.int32)

    def draw_static(self, img: np.ndarray) -> None:
        path_px = self.path_poly_px()
        cv2.polylines(img, [path_px], False, (0, 210, 0), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(path_px[-1]), 7, (255, 0, 255), 2, cv2.LINE_AA)

        if self.wall_map is not None:
            mask = self.wall_map.mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 30:
                    continue
                approx = cv2.approxPolyDP(contour, 2.0, True).reshape(-1, 2)
                board = self.wall_map.origin_mm + approx.astype(float) / self.wall_map.scale
                px = self.homography.board_points_to_image_px(board).astype(np.int32)
                cv2.polylines(img, [px], True, (60, 60, 60), 1, cv2.LINE_AA)

        for idx, (x_mm, y_mm, r_mm) in enumerate(self.holes, start=1):
            center = self.mm_to_px(np.array([x_mm, y_mm]))
            edge = self.mm_to_px(np.array([x_mm + r_mm, y_mm]))
            r_px = int(np.hypot(edge[0] - center[0], edge[1] - center[1]))
            cap_mm = self.hole_map.capture_mm[idx - 1]
            cap_edge = self.mm_to_px(np.array([x_mm + cap_mm, y_mm]))
            cap_px = int(np.hypot(cap_edge[0] - center[0], cap_edge[1] - center[1]))
            draw_translucent_circle(img, center, max(cap_px, 3), (0, 120, 255), 0.12)
            cv2.circle(img, center, max(r_px, 3), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(img, center, max(cap_px, 3), (0, 120, 255), 1, cv2.LINE_AA)
            put_text(img, str(idx), (center[0] + 6, center[1] - 6), (0, 0, 255), 0.42)

    def render(self, rows: list[dict], idx: int, title: str = "") -> np.ndarray:
        row = rows[idx]
        img = self.static_base.copy()

        rel_t = f(row, "timestamp_s") - f(next(r for r in rows if found(r)), "timestamp_s")
        progress = f(row, "progress_mm")
        ball = np.array([f(row, "x_mm"), f(row, "y_mm")])
        vel = np.array([f(row, "vx_mm_s"), f(row, "vy_mm_s")])
        target = np.array([f(row, "target_x_mm"), f(row, "target_y_mm")])
        desired = np.array([f(row, "desired_vx_mm_s"), f(row, "desired_vy_mm_s")])
        board_cmd = np.array([f(row, "board_cmd_x"), f(row, "board_cmd_y")])
        speed = float(np.linalg.norm(vel))
        cmd_mag = float(np.hypot(f(row, "yaw_command"), f(row, "pitch_command")))
        wall_scale = f(row, "wall_speed_scale", 1.0)
        turn_deg = f(row, "turn_deg")
        cross = f(row, "cross_track_mm")
        hole_brake = row.get("hole_brake", "")

        # Recent trail.
        trail_points = []
        for prev in rows[max(0, idx - 120):idx + 1]:
            if found(prev):
                trail_points.append([f(prev, "x_mm"), f(prev, "y_mm")])
        if len(trail_points) >= 2:
            trail_px = self.homography.board_points_to_image_px(
                np.array(trail_points, dtype=float)).astype(np.int32)
            cv2.polylines(img, [trail_px], False, (255, 140, 0), 2, cv2.LINE_AA)

        # Hazard scan from current progress.
        stop_d = speed * speed / max(2.0 * self.hole_brake_accel, 1e-9)
        horizon = max(self.hole_horizon_mm, 1.3 * stop_d + self.hole_standoff_mm + 20.0)
        hazard_d = self.hole_map.path_hazard_distance_mm(self.path, progress, horizon)
        speed_cap = self.hole_map.speed_cap_mm_s(
            hazard_d, self.hole_brake_accel, self.hole_standoff_mm)
        horizon_px = self.path_poly_px(progress, progress + horizon)
        cv2.polylines(img, [horizon_px], False, (0, 190, 255), 3, cv2.LINE_AA)
        if hazard_d is not None:
            hp = self.path.point_at_progress_mm(progress + hazard_d)
            cv2.circle(img, self.mm_to_px(hp), 8, (0, 0, 255), -1, cv2.LINE_AA)
            put_text(img, f"first hole hazard {hazard_d:.0f}mm ahead",
                     (self.mm_to_px(hp)[0] + 8, self.mm_to_px(hp)[1] + 16),
                     (0, 0, 255), 0.48)

        if found(row):
            cv2.circle(img, self.mm_to_px(ball), 7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(img, self.mm_to_px(target), 6, (0, 255, 255), -1, cv2.LINE_AA)
            draw_arrow_mm(img, self.homography, ball, vel, (255, 0, 0), "vel", 0.18)
            draw_arrow_mm(img, self.homography, ball, desired, (0, 255, 255), "desired", 0.30)
            draw_arrow_mm(img, self.homography, ball, board_cmd, (0, 140, 255), "cmd", 20.0)

        if self.wall_map is not None and found(row):
            wall_d = self.wall_map.wall_distance_mm(ball)
        else:
            wall_d = float("nan")

        corner_scale = max(self.min_speed_frac,
                           1.0 - turn_deg / max(self.corner_slow_deg, 1e-9))
        braking = speed > 20.0 and float(np.dot(board_cmd, vel)) < 0.0
        stalled = speed < 8.0 and cmd_mag > 0.05

        reasons = []
        if hole_brake:
            reasons.append(f"hole_brake={hole_brake}")
        if speed_cap is not None:
            reasons.append(f"hole speed cap {speed_cap:.0f} mm/s")
        if wall_scale < 0.999:
            reasons.append(f"wall slowdown {wall_scale:.2f} (dist {wall_d:.1f}mm)")
        if corner_scale < 0.999:
            reasons.append(f"corner slowdown {corner_scale:.2f} (turn {turn_deg:.0f}deg)")
        if braking:
            reasons.append("braking: command opposes velocity")
        if stalled:
            reasons.append("STALLED: command present but speed < 8mm/s")
        if not reasons:
            reasons.append("no active slowdown/brake flag")

        panel = img.copy()
        cv2.rectangle(panel, (8, 8), (560, 176 + 22 * len(reasons)), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.58, img, 0.42, 0.0, img)
        put_text(img, title or f"run frame {idx}", (18, 32), (255, 255, 255), 0.68)
        put_text(img, f"t={rel_t:.1f}s  progress={progress:.0f}/{self.total_mm:.0f}mm"
                 f"  cross={cross:.1f}mm", (18, 60), (255, 255, 255), 0.55)
        put_text(img, f"speed={speed:.0f}mm/s  cmd_mag={cmd_mag:.2f}"
                 f"  yaw={f(row, 'yaw_command'):+.2f} pitch={f(row, 'pitch_command'):+.2f}",
                 (18, 86), (255, 255, 255), 0.55)
        put_text(img, f"wall_dist={wall_d:.1f}mm  wall_scale={wall_scale:.2f}"
                 f"  turn={turn_deg:.0f}deg  hazard_d={hazard_d if hazard_d is not None else 'none'}",
                 (18, 112), (255, 255, 255), 0.55)
        put_text(img, "why:", (18, 144), (0, 255, 255), 0.55)
        for i, reason in enumerate(reasons):
            color = (0, 0, 255) if "STALLED" in reason or "emergency" in reason else (0, 255, 255)
            put_text(img, f"- {reason}", (34, 170 + 22 * i), color, 0.52)

        legend_y = img.shape[0] - 78
        cv2.rectangle(img, (8, legend_y - 20), (720, img.shape[0] - 8), (0, 0, 0), -1)
        put_text(img, "green=path  orange=hazard lookahead  red=hole  orange fill=capture zone",
                 (18, legend_y), (255, 255, 255), 0.48)
        put_text(img, "blue=ball/velocity  yellow=target/desired velocity  orange arrow=board command",
                 (18, legend_y + 24), (255, 255, 255), 0.48)
        return img


def make_contact_sheet(paths: list[Path], output: Path, thumb_w: int = 420) -> None:
    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return
    thumbs = []
    for im in imgs:
        scale = thumb_w / im.shape[1]
        thumbs.append(cv2.resize(im, (thumb_w, int(im.shape[0] * scale))))
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    h = max(im.shape[0] for im in thumbs)
    sheet = np.full((rows * h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for i, im in enumerate(thumbs):
        y = (i // cols) * h
        x = (i % cols) * thumb_w
        sheet[y:y + im.shape[0], x:x + thumb_w] = im
    cv2.imwrite(str(output), sheet)


def first_found_time(rows: list[dict]) -> float:
    for row in rows:
        if found(row):
            return f(row, "timestamp_s")
    return f(rows[0], "timestamp_s")


def nearest_found_index_by_progress(rows: list[dict], progress_mm: float) -> int | None:
    best: tuple[float, int] | None = None
    for idx, row in enumerate(rows):
        if not found(row) or row.get("progress_mm", "") == "":
            continue
        distance = abs(f(row, "progress_mm") - progress_mm)
        if best is None or distance < best[0]:
            best = (distance, idx)
    return None if best is None else best[1]


def nearest_index_by_relative_time(rows: list[dict], t_s: float) -> int:
    t0 = first_found_time(rows)
    best_idx = 0
    best_distance = float("inf")
    for idx, row in enumerate(rows):
        distance = abs((f(row, "timestamp_s") - t0) - t_s)
        if distance < best_distance:
            best_distance = distance
            best_idx = idx
    return best_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/raw/autonomous_run.csv")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--path", default=None)
    parser.add_argument("--holes", default="configs/maze_holes.csv")
    parser.add_argument("--wall-mask", default="calibration/wall_mask.npz")
    parser.add_argument("--background", default=None,
                        help="image to draw on; defaults to live_roi_source.png then CURRENT_FIXED_CAMERA_VIEW.png")
    parser.add_argument("--out-dir", default="data/processed/run_visualizer")
    parser.add_argument("--stills", type=int, default=12,
                        help="number of longest stall episodes to render as PNGs")
    parser.add_argument("--out-video", default=None,
                        help="optional MP4 replay output")
    parser.add_argument("--stride", type=int, default=8,
                        help="render every Nth log row for --out-video")
    parser.add_argument("--video-fps", type=float, default=20.0)
    parser.add_argument("--progress", type=float, action="append", default=[],
                        help="also render the log frame nearest this path progress in mm")
    parser.add_argument("--time", type=float, action="append", default=[],
                        help="also render the log frame nearest this relative run time in seconds")
    parser.add_argument("--stall-speed", type=float, default=8.0)
    parser.add_argument("--stall-cmd", type=float, default=0.05)
    args = parser.parse_args()

    rows = read_rows(Path(args.log))
    if not rows:
        raise SystemExit("empty run log")
    visualizer = RunVisualizer(
        Path(args.config),
        Path(args.homography),
        Path(args.path) if args.path else None,
        Path(args.holes),
        Path(args.wall_mask),
        Path(args.background) if args.background else None,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = stall_episodes(rows, args.stall_speed, args.stall_cmd)
    still_paths: list[Path] = []
    for n, ep in enumerate(episodes[:args.stills], start=1):
        idx = (ep.start_idx + ep.end_idx) // 2
        title = (f"stall {n}: t={ep.start_t_s:.1f}s dur={ep.duration_s:.1f}s "
                 f"progress={ep.progress_mm:.0f}mm")
        img = visualizer.render(rows, idx, title)
        output = out_dir / (
            f"stall_{n:02d}_t{ep.start_t_s:05.1f}_p{ep.progress_mm:04.0f}.png"
        )
        cv2.imwrite(str(output), img)
        still_paths.append(output)
    if still_paths:
        make_contact_sheet(still_paths, out_dir / "stall_contact_sheet.jpg")

    targeted_paths: list[Path] = []
    for progress in args.progress:
        idx = nearest_found_index_by_progress(rows, progress)
        if idx is None:
            continue
        actual = f(rows[idx], "progress_mm")
        output = out_dir / f"progress_{progress:04.0f}_actual_{actual:04.0f}.png"
        img = visualizer.render(rows, idx, f"nearest progress {progress:.0f}mm")
        cv2.imwrite(str(output), img)
        targeted_paths.append(output)

    for t_s in args.time:
        idx = nearest_index_by_relative_time(rows, t_s)
        actual = f(rows[idx], "timestamp_s") - first_found_time(rows)
        output = out_dir / f"time_{t_s:05.1f}_actual_{actual:05.1f}.png"
        img = visualizer.render(rows, idx, f"nearest t={t_s:.1f}s")
        cv2.imwrite(str(output), img)
        targeted_paths.append(output)

    if args.out_video:
        first = visualizer.render(rows, 0, "run replay")
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(
            str(args.out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.video_fps,
            (w, h),
        )
        for idx in range(0, len(rows), max(args.stride, 1)):
            writer.write(visualizer.render(rows, idx, "run replay"))
        writer.release()
        print(f"saved video: {args.out_video}")

    print(f"stall episodes found: {len(episodes)}")
    if still_paths:
        print(f"saved stills: {out_dir}")
        print(f"saved contact sheet: {out_dir / 'stall_contact_sheet.jpg'}")
    else:
        print("no stall stills written")
    if targeted_paths:
        print("saved targeted snapshots:")
        for path in targeted_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
