#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WINDOW = "Select Maze ROI"


def roi_to_arg(roi: list[list[int]]) -> str:
    return ";".join(f"{x},{y}" for x, y in roi)


def load_frame(source: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(source))
    if cap.isOpened():
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return frame

    frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image/video frame from {source}")
    return frame


def load_existing_roi(path: Path | None) -> list[list[int]]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text())
    roi = data if isinstance(data, list) else data.get("roi", [])
    return [[int(round(x)), int(round(y))] for x, y in roi]


def draw_overlay(frame: np.ndarray, roi: list[list[int]], message: str) -> np.ndarray:
    out = frame.copy()
    if len(roi) >= 3:
        poly = np.array(roi, dtype=np.int32)
        fill = out.copy()
        cv2.fillPoly(fill, [poly], (0, 180, 0))
        out = cv2.addWeighted(fill, 0.22, out, 0.78, 0)
        cv2.polylines(out, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
    elif len(roi) >= 2:
        cv2.polylines(out, [np.array(roi, dtype=np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)

    for idx, (x, y) in enumerate(roi):
        cv2.circle(out, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            str(idx + 1),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    lines = [
        "Left click: add point | right click/u: undo | r: reset | s/enter: save | q/esc: quit",
        message,
    ]
    for idx, line in enumerate(lines):
        y = 28 + idx * 28
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1)
    return out


def save_roi(
    output: Path,
    overlay_output: Path,
    source: Path,
    frame_index: int,
    frame: np.ndarray,
    roi: list[list[int]],
) -> None:
    height, width = frame.shape[:2]
    payload = {
        "roi": roi,
        "roi_arg": roi_to_arg(roi),
        "source": str(source),
        "frame_index": frame_index,
        "frame_size": {"width": width, "height": height},
        "coordinate_frame": "opencv_pixels_top_left_x_right_y_down",
        "intent": "playable maze surface only; excludes external CharUco board and outside frame",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    cv2.imwrite(str(overlay_output), draw_overlay(frame, roi, "Saved ROI overlay"))

    print(f"saved ROI: {output}")
    print(f"saved overlay: {overlay_output}")
    print(f"roi arg: {payload['roi_arg']}")
    print("calibrate with:")
    print(
        "  python3 scripts/pipeline.py "
        f"--calibrate {source} "
        "--confusers-file calibration/live_confusers.json "
        f"--roi-file {output}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click a playable-maze ROI polygon on a fixed-camera frame."
    )
    parser.add_argument("--source", default="data/raw/live_camera_20260707_234339.avi")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--output", default="calibration/live_roi.json")
    parser.add_argument("--overlay-output", default="calibration/live_roi_overlay.png")
    parser.add_argument(
        "--load",
        default=None,
        help="optional existing ROI JSON to load before editing; defaults to --output if it exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    overlay_output = Path(args.overlay_output)
    frame = load_frame(source, args.frame_index)
    roi = load_existing_roi(Path(args.load) if args.load else output)
    message = "Click the inside playable maze boundary. Exclude CharUco and outer frame."
    saved = False

    def on_mouse(event, x, y, _flags, _param):
        nonlocal roi
        if event == cv2.EVENT_LBUTTONDOWN:
            roi.append([int(x), int(y)])
        elif event == cv2.EVENT_RBUTTONDOWN and roi:
            roi.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            cv2.imshow(WINDOW, draw_overlay(frame, roi, message))
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("u"), 8, 127) and roi:
                roi.pop()
            elif key == ord("r"):
                roi = []
            elif key in (ord("s"), 13, 10):
                if len(roi) < 3:
                    message = "Need at least 3 points before saving."
                    continue
                save_roi(output, overlay_output, source, args.frame_index, frame, roi)
                saved = True
                message = f"Saved {len(roi)} ROI points. Press q to close."
    finally:
        try:
            cv2.destroyWindow(WINDOW)
        except cv2.error:
            pass

    if not saved:
        print("ROI was not saved.")


if __name__ == "__main__":
    main()
