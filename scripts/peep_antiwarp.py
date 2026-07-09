from pathlib import Path

import cv2
import numpy as np

from cps_maze.calibration.homography import Homography
from cps_maze.calibration.intrinsics import CameraIntrinsics, undistort_image
from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.vision.ball_tracker import BrightBlobBallTracker

def build_display_warp(image_to_board_mm, mm_per_px=2.0, width_mm=322.0, height_mm=282.0):
    board_to_display = np.array([
        [1.0 / mm_per_px, 0, 0],
        [0, -1.0 / mm_per_px, height_mm / mm_per_px],
        [0, 0, 1],
    ], dtype=np.float32)

    display_H = board_to_display @ image_to_board_mm
    out_w = int(width_mm / mm_per_px)
    out_h = int(height_mm / mm_per_px)
    return display_H, (out_w, out_h)

def main():
    config = load_config("configs/default.yaml")
    homography = Homography.load("calibration/board_homography.npz")
    tracker = BrightBlobBallTracker(config.vision)
    intrinsics_path = Path("calibration/camera_intrinsics.npz")
    intrinsics = CameraIntrinsics.load(intrinsics_path) if intrinsics_path.exists() else None

    display_H, out_size = build_display_warp(homography.image_to_board, mm_per_px=2.0)

    with CameraCapture(config.camera) as camera:
        while True:
            frame = camera.read()
            image = undistort_image(frame.image, intrinsics) if intrinsics is not None else frame.image

            warped = cv2.warpPerspective(image, display_H, out_size)
            detection = tracker.detect(warped)

            cv2.imshow("rectified board", warped)
            if detection.found and detection.x_px is not None and detection.y_px is not None:
                overlay = tracker.draw_detection(warped, detection)
                cv2.imshow("rectified board overlay", overlay)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

if __name__ == "__main__":
    main()