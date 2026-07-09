#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cps_maze.camera import CameraCapture
from cps_maze.config import load_config


def default_codec_for_path(path: Path) -> str:
    if path.suffix.lower() in {".mp4", ".m4v"}:
        return "mp4v"
    return "MJPG"


def default_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"live_camera_{timestamp}.avi"


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return value


def draw_preview(frame, elapsed_s: float, frame_count: int, duration_s: float):
    preview = frame.copy()
    text = f"REC {elapsed_s:5.1f}/{duration_s:.1f}s  frames={frame_count}  q/esc stop"
    cv2.putText(
        preview,
        text,
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record unscaled fixed-camera video through CameraCapture."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument(
        "--output",
        default=None,
        help="video output path; defaults to data/raw/live_camera_YYYYMMDD_HHMMSS.avi",
    )
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument(
        "--codec",
        default=None,
        help="fourcc codec for VideoWriter; defaults to MJPG for .avi, mp4v for .mp4",
    )
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be positive")

    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = Path(args.output_dir)
    output_path = Path(args.output) if args.output else default_output_path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")
    codec = args.codec or default_codec_for_path(output_path)

    started_at = datetime.now(timezone.utc)
    start_monotonic = monotonic()
    frame_count = 0
    stopped_by_user = False
    first_frame_size: dict[str, int] | None = None
    requested_settings: dict[str, Any] | None = None
    observed_settings: dict[str, Any] | None = None

    with CameraCapture(config.camera) as camera:
        requested_settings = camera.requested_settings()
        observed_settings = camera.observed_settings()
        first_frame = camera.read().image
        height, width = first_frame.shape[:2]
        first_frame_size = {"width": int(width), "height": int(height)}
        writer_fps = float(requested_settings["fps"])

        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, writer_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")

        print("Recording fixed-camera video")
        print(f"  output: {output_path}")
        print(f"  metadata: {metadata_path}")
        print(f"  duration_s: {args.duration_s}")
        print(f"  writer: {width}x{height}@{writer_fps:.1f}fps codec={codec}")
        print(
            "  camera: "
            f"device_index={requested_settings['device_index']} "
            f"{requested_settings['width']}x{requested_settings['height']}"
            f"@{requested_settings['fps']}fps "
            f"flip_horizontal={str(requested_settings['flip_horizontal']).lower()} "
            f"flip_vertical={str(requested_settings['flip_vertical']).lower()}"
        )

        try:
            frame = first_frame
            while True:
                elapsed_s = monotonic() - start_monotonic
                if elapsed_s >= args.duration_s:
                    break

                writer.write(frame)
                frame_count += 1

                if not args.no_preview:
                    cv2.imshow(
                        "Fixed Camera Recording",
                        draw_preview(frame, elapsed_s, frame_count, args.duration_s),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        stopped_by_user = True
                        break

                frame = camera.read().image
        finally:
            writer.release()
            if not args.no_preview:
                try:
                    cv2.destroyWindow("Fixed Camera Recording")
                except cv2.error:
                    pass

    ended_at = datetime.now(timezone.utc)
    elapsed_s = monotonic() - start_monotonic
    achieved_fps = frame_count / elapsed_s if elapsed_s > 0 else 0.0
    metadata = {
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "config_path": str(config_path),
        "output_video": str(output_path),
        "requested_duration_s": args.duration_s,
        "elapsed_s": elapsed_s,
        "frame_count": frame_count,
        "achieved_fps": achieved_fps,
        "writer_fps": float(requested_settings["fps"]) if requested_settings else None,
        "writer_codec": codec,
        "first_frame_size": first_frame_size,
        "requested_camera_settings": requested_settings,
        "observed_camera_settings": observed_settings,
        "camera_convention": config.camera.get("convention"),
        "reference_image": config.camera.get("reference_image"),
        "stopped_by_user": stopped_by_user,
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
    }
    metadata_path.write_text(json.dumps(make_json_safe(metadata), indent=2) + "\n")

    print("Recording complete")
    print(f"  frames_written: {frame_count}")
    print(f"  elapsed_s: {elapsed_s:.2f}")
    print(f"  achieved_fps: {achieved_fps:.2f}")
    print(f"  video: {output_path}")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
