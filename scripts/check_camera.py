#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

from cps_maze.camera import CameraCapture
from cps_maze.config import load_config
from cps_maze.vision.aruco import ArucoDetector
from cps_maze.vision.ball_tracker import BrightBlobBallTracker


def _format_bool(value: bool) -> str:
    return str(value).lower()


def print_stage0_convention(camera_config: dict[str, Any]) -> None:
    print("Stage 0 fixed-camera convention:")
    print("  reference_image: calibration/CURRENT_FIXED_CAMERA_VIEW.png")
    print("  image_frame: OpenCV pixels, origin top-left, x right, y down")
    print("  accepted_orientation: Start near top-right; Finish near bottom-left")
    print(
        "  invalidates_later_data: changing device index, resolution, FPS, flips, "
        "crop, or camera pose"
    )
    print("Requested camera settings:")
    print(f"  device_index={camera_config['device_index']}")
    print(f"  width={camera_config['width']}")
    print(f"  height={camera_config['height']}")
    print(f"  fps={camera_config['fps']}")
    print(f"  flip_horizontal={_format_bool(bool(camera_config.get('flip_horizontal', False)))}")
    print(f"  flip_vertical={_format_bool(bool(camera_config.get('flip_vertical', False)))}")


def print_camera_runtime(camera: CameraCapture) -> None:
    requested = camera.requested_settings()
    observed = camera.observed_settings()
    print("Camera runtime settings:")
    print(
        "  requested: "
        f"device_index={requested['device_index']} "
        f"{requested['width']}x{requested['height']}@{requested['fps']}fps "
        f"fourcc={requested['fourcc']} backend={requested['backend']} "
        f"flip_horizontal={_format_bool(requested['flip_horizontal'])} "
        f"flip_vertical={_format_bool(requested['flip_vertical'])}"
    )
    print(
        "  observed: "
        f"{observed['width']}x{observed['height']}@{observed['fps']:.1f}fps "
        f"fourcc={observed['fourcc']} backend={observed['backend']}"
    )


def save_current_config(config_path: Path, config_data: dict[str, Any], params: dict[str, Any]) -> None:
    config_data.setdefault("vision", {})
    for key in (
        "min_blob_area_px",
        "max_blob_area_px",
        "threshold_value",
        "blur_kernel",
        "morph_kernel",
        "clahe_clip",
        "smoothing_alpha",
        "use_otsu",
        "debug_overlay",
        "debug_overlay_every_n",
        "debug_start_stage",
    ):
        config_data["vision"][key] = params[key]
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_data, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    print_stage0_convention(config.camera)
    tracker = BrightBlobBallTracker(config.vision)
    aruco_detector = ArucoDetector()

    stages = ["gray", "clahe", "blurred", "binary", "adaptive_mean", "adaptive_gauss", "morph"]
    stage_idx = stages.index(config.vision.get("debug_start_stage", "binary")) if config.vision.get("debug_start_stage") in stages else 3
    show_aruco = bool(config.vision.get("aruco_show_by_default", False))
    save_debug = bool(config.vision.get("save_debug", False))

    live_params: dict[str, Any] = {
        "min_blob_area_px": int(config.vision.get("min_blob_area_px", 25)),
        "max_blob_area_px": int(config.vision.get("max_blob_area_px", 5000)),
        "threshold_value": int(config.vision.get("threshold_value", 220)),
        "blur_kernel": int(config.vision.get("blur_kernel", 5)),
        "morph_kernel": int(config.vision.get("morph_kernel", 5)),
        "clahe_clip": float(config.vision.get("clahe_clip", 2.0)),
        "smoothing_alpha": float(config.vision.get("smoothing_alpha", 0.6)),
        "use_otsu": bool(config.vision.get("use_otsu", False)),
        "debug_overlay": bool(config.vision.get("debug_overlay", False)),
        "debug_overlay_every_n": int(config.vision.get("debug_overlay_every_n", 1)),
        "debug_start_stage": config.vision.get("debug_start_stage", "binary"),
    }

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 500, 320)
    cv2.createTrackbar("min_area", "Controls", int(live_params["min_blob_area_px"]), 1000, lambda _value: None)
    cv2.createTrackbar("max_area", "Controls", int(live_params["max_blob_area_px"]), 10000, lambda _value: None)
    cv2.createTrackbar("threshold", "Controls", int(live_params["threshold_value"]), 255, lambda _value: None)
    cv2.createTrackbar("blur", "Controls", int(live_params["blur_kernel"]), 21, lambda _value: None)
    cv2.createTrackbar("morph", "Controls", int(live_params["morph_kernel"]), 21, lambda _value: None)
    cv2.createTrackbar("clahe", "Controls", int(live_params["clahe_clip"] * 10), 40, lambda _value: None)
    cv2.createTrackbar("smooth", "Controls", int(live_params["smoothing_alpha"] * 100), 100, lambda _value: None)
    cv2.createTrackbar("otsu", "Controls", int(live_params["use_otsu"]), 1, lambda _value: None)
    cv2.createTrackbar("debug", "Controls", int(live_params["debug_overlay"]), 1, lambda _value: None)
    cv2.createTrackbar("every_n", "Controls", int(live_params["debug_overlay_every_n"]), 10, lambda _value: None)

    with CameraCapture(config.camera) as camera:
        print_camera_runtime(camera)
        while True:
            live_params["min_blob_area_px"] = cv2.getTrackbarPos("min_area", "Controls")
            live_params["max_blob_area_px"] = cv2.getTrackbarPos("max_area", "Controls")
            live_params["threshold_value"] = cv2.getTrackbarPos("threshold", "Controls")
            live_params["blur_kernel"] = cv2.getTrackbarPos("blur", "Controls")
            live_params["morph_kernel"] = cv2.getTrackbarPos("morph", "Controls")
            live_params["clahe_clip"] = cv2.getTrackbarPos("clahe", "Controls") / 10.0
            live_params["smoothing_alpha"] = cv2.getTrackbarPos("smooth", "Controls") / 100.0
            live_params["use_otsu"] = bool(cv2.getTrackbarPos("otsu", "Controls"))
            live_params["debug_overlay"] = bool(cv2.getTrackbarPos("debug", "Controls"))
            live_params["debug_overlay_every_n"] = cv2.getTrackbarPos("every_n", "Controls")
            tracker.update_from_config(live_params)

            frame = camera.read()
            ball_detection = tracker.detect(frame.image)
            ball_output = tracker.draw_detection(frame.image, ball_detection)
            aruco_detection = aruco_detector.detect(frame.image)
            aruco_output = aruco_detector.draw_detection(frame.image, aruco_detection) if show_aruco else frame.image

            cv2.imshow("Ball Detection", ball_output)
            if show_aruco:
                cv2.imshow("ArUco Detection", aruco_output)
            else:
                try:
                    cv2.destroyWindow("ArUco Detection")
                except Exception:
                    pass

            if tracker.debug_enabled and (tracker.frame_count % tracker.debug_every_n == 0):
                overlay = tracker.draw_debug_overlay(frame.image, stages[stage_idx])
                cv2.imshow("Debug Overlay", overlay)
            else:
                try:
                    cv2.destroyWindow("Debug Overlay")
                except Exception:
                    pass

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("d"):
                live_params["debug_overlay"] = not bool(live_params["debug_overlay"])
                print("Debug overlay:", live_params["debug_overlay"])
            elif key == ord("s"):
                stage_idx = (stage_idx + 1) % len(stages)
                live_params["debug_start_stage"] = stages[stage_idx]
                print("Debug stage:", stages[stage_idx])
            elif key == ord("a"):
                show_aruco = not show_aruco
                print("Show ArUco:", show_aruco)
            elif key == ord("p"):
                if tracker.debug_enabled:
                    out = tracker.draw_debug_overlay(frame.image, stages[stage_idx])
                    os.makedirs("logs/agent", exist_ok=True)
                    fname = f"logs/agent/debug_{int(time.time())}_{tracker.frame_count:06d}.png"
                    cv2.imwrite(fname, out)
                    print("Saved debug frame:", fname)
            elif key == ord("r"):
                live_params = {
                    "min_blob_area_px": int(config.vision.get("min_blob_area_px", 25)),
                    "max_blob_area_px": int(config.vision.get("max_blob_area_px", 5000)),
                    "threshold_value": int(config.vision.get("threshold_value", 220)),
                    "blur_kernel": int(config.vision.get("blur_kernel", 5)),
                    "morph_kernel": int(config.vision.get("morph_kernel", 5)),
                    "clahe_clip": float(config.vision.get("clahe_clip", 2.0)),
                    "smoothing_alpha": float(config.vision.get("smoothing_alpha", 0.6)),
                    "use_otsu": bool(config.vision.get("use_otsu", False)),
                    "debug_overlay": bool(config.vision.get("debug_overlay", False)),
                    "debug_overlay_every_n": int(config.vision.get("debug_overlay_every_n", 1)),
                    "debug_start_stage": config.vision.get("debug_start_stage", "binary"),
                }
                tracker.update_from_config(live_params)
                print("Reset to defaults")
            elif key == ord("c"):
                save_current_config(config_path, config.raw, live_params)
                print("Saved current tuning values to", config_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
