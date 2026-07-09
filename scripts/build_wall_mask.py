#!/usr/bin/env python3
"""Build the static wall/obstacle mask used for control-time wall awareness.

Rectifies the camera view into board-mm space and thresholds the dark,
THICK structures (walls; the thin printed guide line is removed by
morphological opening). Holes stay in the mask - they are obstacles too.
Everything outside the board rectangle is marked blocked.

Run once per board setup; regenerate whenever the homography changes.

Controls:
  threshold trackbar : adjust until walls are solid red, floor clean
  SPACE              : grab a fresh frame
  s                  : save calibration/wall_mask.npz
  q/Esc              : quit
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.planning.walls import WallMap

from auto_detect_holes import warp_topdown

WINDOW = "build wall mask"


def build_mask(
    topdown_bgr: np.ndarray,
    threshold: int,
    scale: float,
    origin_mm: np.ndarray,
    board_mm: tuple[float, float] | None,
    min_wall_width_mm: float = 3.0,
) -> np.ndarray:
    gray = cv2.cvtColor(topdown_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    # opening removes structures thinner than a real wall (the printed line)
    k = max(int(min_wall_width_mm * scale) | 1, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    walls = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    if board_mm is not None:
        h, w = walls.shape
        xs = (np.arange(w) / scale) + origin_mm[0]
        ys = (np.arange(h) / scale) + origin_mm[1]
        outside_x = (xs < 0) | (xs > board_mm[0])
        outside_y = (ys < 0) | (ys > board_mm[1])
        walls[:, outside_x] = 255
        walls[outside_y, :] = 255
    return walls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--homography", default="calibration/board_homography.npz")
    parser.add_argument("--output", default="calibration/wall_mask.npz")
    parser.add_argument("--threshold", type=int, default=110)
    parser.add_argument("--min-wall-width-mm", type=float, default=3.0,
                        help="Thinner dark structures (the printed line) are ignored")
    args = parser.parse_args()

    config = load_config(args.config)
    homography = Homography.load(args.homography)
    try:
        board_mm = (float(config.maze["width_mm"]), float(config.maze["height_mm"]))
    except (KeyError, TypeError, ValueError):
        board_mm = None

    cv2.namedWindow(WINDOW)
    cv2.createTrackbar("threshold", WINDOW, args.threshold, 255, lambda _v: None)
    print(__doc__)

    with CameraCapture(config.camera) as camera:
        frame = camera.read().image
        topdown, origin, scale = warp_topdown(frame, homography)
        last_threshold = -1
        mask = None

        while True:
            threshold = cv2.getTrackbarPos("threshold", WINDOW)
            if threshold != last_threshold:
                mask = build_mask(topdown, threshold, scale, origin, board_mm,
                                  args.min_wall_width_mm)
                last_threshold = threshold

            view = topdown.copy()
            view[mask > 0] = (view[mask > 0] * 0.4 + np.array([0, 0, 153])).astype(np.uint8)
            cv2.putText(view, "walls tinted red - s=save  SPACE=refresh", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(WINDOW, view)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                frame = camera.read().image
                topdown, origin, scale = warp_topdown(frame, homography)
                last_threshold = -1
            elif key == ord("s") and mask is not None:
                WallMap(mask > 0, origin, scale).save(args.output)
                coverage = 100.0 * (mask > 0).mean()
                print(f"saved {args.output} ({coverage:.0f}% blocked incl. off-board)")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
