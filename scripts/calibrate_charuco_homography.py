#!/usr/bin/env python3
from __future__ import annotations

import argparse

import cv2
import numpy as np

from cps_maze.calibration.charuco import charuco_ids_to_maze_points_mm
from cps_maze.calibration.homography import estimate_homography
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.vision.aruco import CharucoDetector


def _extract_charuco_points(detection) -> tuple[np.ndarray, np.ndarray] | None:
    if detection.charuco_corners is None or detection.charuco_ids is None:
        return None
    image_points = np.asarray(detection.charuco_corners, dtype=np.float32).reshape(-1, 2)
    maze_points = charuco_ids_to_maze_points_mm(detection.charuco_ids)
    if image_points.shape[0] != maze_points.shape[0] or image_points.shape[0] < 4:
        return None
    return image_points, maze_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate an image-to-maze homography from a CharUco board.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="calibration/board_homography.npz")
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--min-corners", type=int, default=8)
    parser.add_argument("--preview", action="store_true", help="show detections while searching for the board")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = CharucoDetector()

    best_detection: tuple[np.ndarray, np.ndarray] | None = None
    best_count = 0
    with CameraCapture(config.camera) as camera:
        for _ in range(args.max_frames):
            frame = camera.read()
            detection = detector.detect(frame.image)
            extracted = _extract_charuco_points(detection)
            if extracted is None:
                if args.preview:
                    preview = detector.draw_detection(frame.image, detection)
                    cv2.imshow("charuco calibration", preview)
                    cv2.waitKey(1)
                continue

            image_points, maze_points = extracted
            count = image_points.shape[0]
            if count > best_count:
                best_detection = (image_points, maze_points)
                best_count = count

            if args.preview:
                preview = detector.draw_detection(frame.image, detection)
                cv2.imshow("charuco calibration", preview)
                cv2.waitKey(1)

            if count >= args.min_corners:
                break

    if args.preview:
        cv2.destroyAllWindows()

    if best_detection is None:
        raise RuntimeError("Could not detect enough CharUco corners to calibrate")

    image_points, maze_points = best_detection
    homography = estimate_homography(image_points, maze_points)
    homography.save(args.output)
    print(f"Saved CharUco homography to {args.output}")
    print(f"Used {best_count} CharUco corners")


if __name__ == "__main__":
    main()