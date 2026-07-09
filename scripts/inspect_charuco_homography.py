#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.calibration.intrinsics import CameraIntrinsics, undistort_image
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config


@dataclass
class InspectorState:
    undistort_enabled: bool = True
    warp_enabled: bool = True
    grid_enabled: bool = True
    cursor_x: int | None = None
    cursor_y: int | None = None


def build_board_display_transform(
    image_to_board_mm: np.ndarray,
    mm_per_px: float,
    width_mm: float,
    height_mm: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    board_to_display = np.array(
        [
            [1.0 / mm_per_px, 0.0, 0.0],
            [0.0, -1.0 / mm_per_px, height_mm / mm_per_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    display_to_source = np.linalg.inv(board_to_display @ image_to_board_mm)
    display_size = (int(round(width_mm / mm_per_px)), int(round(height_mm / mm_per_px)))
    return board_to_display @ image_to_board_mm, display_to_source, display_size


def draw_text_block(image: np.ndarray, lines: list[str], origin: tuple[int, int] = (12, 24)) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        y += 20


def draw_mm_grid(image: np.ndarray, mm_per_px: float, width_mm: float, height_mm: float, step_mm: int = 100) -> None:
    step_px = max(1, int(round(step_mm / mm_per_px)))
    width_px = image.shape[1]
    height_px = image.shape[0]

    for x in range(0, width_px, step_px):
        color = (80, 80, 80) if x % (step_px * 5) else (120, 120, 120)
        cv2.line(image, (x, 0), (x, height_px - 1), color, 1, cv2.LINE_AA)
    for y in range(0, height_px, step_px):
        color = (80, 80, 80) if y % (step_px * 5) else (120, 120, 120)
        cv2.line(image, (0, y), (width_px - 1, y), color, 1, cv2.LINE_AA)


def display_to_board_mm(
    x_px: float,
    y_px: float,
    display_to_source: np.ndarray,
    homography: Homography,
) -> tuple[float, float] | None:
    point = np.array([x_px, y_px, 1.0], dtype=np.float32)
    source_point = display_to_source @ point
    if abs(float(source_point[2])) < 1e-9:
        return None
    source_x = float(source_point[0] / source_point[2])
    source_y = float(source_point[1] / source_point[2])
    return homography.image_point_to_board_mm(source_x, source_y)


def make_preview_frame(
    source_image: np.ndarray,
    homography: Homography,
    intrinsics: CameraIntrinsics | None,
    state: InspectorState,
    mm_per_px: float,
    width_mm: float,
    height_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_display = source_image
    if state.undistort_enabled and intrinsics is not None:
        source_display = undistort_image(source_image, intrinsics)

    if state.warp_enabled:
        display_H, display_to_source, display_size = build_board_display_transform(
            homography.image_to_board,
            mm_per_px,
            width_mm,
            height_mm,
        )
        preview = cv2.warpPerspective(source_display, display_H, display_size)
        if state.grid_enabled:
            draw_mm_grid(preview, mm_per_px, width_mm, height_mm)
    else:
        preview = source_display.copy()
        display_to_source = np.eye(3, dtype=np.float32)

    if preview.ndim == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    lines = [
        f"undistort: {'on' if state.undistort_enabled else 'off'}",
        f"warp: {'on' if state.warp_enabled else 'off'}",
        f"grid: {'on' if state.grid_enabled else 'off'}",
    ]
    if intrinsics is None:
        lines.append("intrinsics: missing")
    else:
        lines.append(f"intrinsics reproj: {intrinsics.reprojection_error:.4f}")
    draw_text_block(preview, lines)

    if state.cursor_x is not None and state.cursor_y is not None:
        cv2.circle(preview, (state.cursor_x, state.cursor_y), 5, (0, 255, 255), 2)

    return preview, display_to_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Charuco homography inspector with toggles for undistortion and warp.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--intrinsics", default="calibration/camera_intrinsics.npz")
    parser.add_argument("--mm-per-px", type=float, default=2.0)
    parser.add_argument("--board-width-mm", type=float, default=322.00)
    parser.add_argument("--board-height-mm", type=float, default=282.00)
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)
    intrinsics = CameraIntrinsics.load(args.intrinsics) if Path(args.intrinsics).exists() else None
    state = InspectorState()

    window_name = "Charuco Homography Inspector"
    raw_window_name = "Raw Camera"

    cursor_board_mm: tuple[float, float] | None = None
    display_to_source = np.eye(3, dtype=np.float32)

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        nonlocal cursor_board_mm
        if event == cv2.EVENT_MOUSEMOVE:
            state.cursor_x = x
            state.cursor_y = y
            cursor_board_mm = display_to_board_mm(x, y, display_to_source, homography)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(raw_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    with CameraCapture(config.camera) as camera:
        while True:
            frame = camera.read()
            raw_image = frame.image
            preview, display_to_source = make_preview_frame(
                raw_image,
                homography,
                intrinsics,
                state,
                args.mm_per_px,
                args.board_width_mm,
                args.board_height_mm,
            )

            raw_display = raw_image.copy()
            if raw_display.ndim == 2:
                raw_display = cv2.cvtColor(raw_display, cv2.COLOR_GRAY2BGR)

            raw_lines = ["raw camera", "keys: u=undistort w=warp g=grid q=quit"]
            if intrinsics is not None:
                raw_lines.append(f"intrinsics reproj: {intrinsics.reprojection_error:.4f}")
            draw_text_block(raw_display, raw_lines)

            if cursor_board_mm is not None:
                mm_text = f"cursor mm: x={cursor_board_mm[0]:.1f}, y={cursor_board_mm[1]:.1f}"
            else:
                mm_text = "cursor mm: n/a"
            cv2.putText(preview, mm_text, (12, preview.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(raw_display, mm_text, (12, raw_display.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(raw_window_name, raw_display)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("u"):
                state.undistort_enabled = not state.undistort_enabled
            elif key == ord("w"):
                state.warp_enabled = not state.warp_enabled
            elif key == ord("g"):
                state.grid_enabled = not state.grid_enabled

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()