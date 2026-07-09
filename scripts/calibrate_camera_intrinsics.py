#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cps_maze.calibration.charuco import board_charuco_corner_points_mm
from cps_maze.calibration.intrinsics import CameraIntrinsics
from cps_maze.vision.aruco import CharucoDetector


def _detect_charuco_frames(
    image_paths: list[Path],
    preview: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    detector = CharucoDetector()
    charuco_corners: list[np.ndarray] = []
    charuco_ids: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"skip {image_path}: could not read image")
            continue
        if image_size is None:
            height, width = image.shape[:2]
            image_size = (width, height)

        detection = detector.detect(image)
        if not detection.found or detection.charuco_corners is None or detection.charuco_ids is None:
            print(f"skip {image_path.name}: no CharUco corners")
            continue

        corners = np.asarray(detection.charuco_corners, dtype=np.float32)
        ids = np.asarray(detection.charuco_ids, dtype=np.int32)
        if corners.shape[0] < 4:
            print(f"skip {image_path.name}: only {corners.shape[0]} corners")
            continue

        charuco_corners.append(corners)
        charuco_ids.append(ids)
        print(f"use {image_path.name}: {corners.shape[0]} corners")

        if preview:
            preview_image = detector.draw_detection(image, detection)
            cv2.imshow("charuco intrinsics calibration", preview_image)
            cv2.waitKey(1)

    if image_size is None:
        raise RuntimeError("Could not read any calibration images")
    return charuco_corners, charuco_ids, image_size


def _build_calibration_points(
    charuco_corners: list[np.ndarray],
    charuco_ids: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    board_points = board_charuco_corner_points_mm()
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    for corners, ids in zip(charuco_corners, charuco_ids):
        flat_ids = np.asarray(ids, dtype=int).reshape(-1)
        if flat_ids.size < 4:
            continue
        planar_points = board_points[flat_ids].astype(np.float32)
        object_points_3d = np.column_stack([planar_points, np.zeros((planar_points.shape[0], 1), dtype=np.float32)])
        object_points.append(object_points_3d.reshape(-1, 1, 3))
        image_points.append(np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2))

    if len(object_points) < 3:
        raise RuntimeError("Need at least 3 valid CharUco frames with enough matched points")

    return object_points, image_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate camera intrinsics from CharUco calibration images.")
    parser.add_argument("--images-dir", default="calibration")
    parser.add_argument("--pattern", default="CharUco_*.png")
    parser.add_argument("--output", default="calibration/camera_intrinsics.npz")
    parser.add_argument("--min-images", type=int, default=3)
    parser.add_argument("--save-new-camera-matrix", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.0, help="used with --save-new-camera-matrix")
    parser.add_argument("--preview", action="store_true", help="show detections while scanning images")
    args = parser.parse_args()

    image_dir = Path(args.images_dir)
    image_paths = sorted(image_dir.glob(args.pattern))
    if not image_paths:
        raise RuntimeError(f"No images matched {args.pattern} in {image_dir}")

    charuco_corners, charuco_ids, image_size = _detect_charuco_frames(image_paths, preview=args.preview)
    if len(charuco_corners) < args.min_images:
        raise RuntimeError(f"Need at least {args.min_images} valid CharUco images, found {len(charuco_corners)}")

    object_points, image_points = _build_calibration_points(charuco_corners, charuco_ids)
    reprojection_error, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    reprojection_error = float(reprojection_error)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64)

    new_camera_matrix = None
    if args.save_new_camera_matrix:
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            image_size,
            args.alpha,
            image_size,
        )

    intrinsics = CameraIntrinsics(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        reprojection_error=reprojection_error,
        new_camera_matrix=None if new_camera_matrix is None else np.asarray(new_camera_matrix, dtype=np.float64),
    )
    intrinsics.save(args.output)

    print(f"Saved camera intrinsics to {args.output}")
    print(f"  images used: {len(charuco_corners)}")
    print(f"  image size: {image_size[0]}x{image_size[1]}")
    print(f"  reprojection error: {reprojection_error:.6f}")
    if intrinsics.new_camera_matrix is not None:
        print(f"  saved new_camera_matrix with alpha={args.alpha}")

    if args.preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()