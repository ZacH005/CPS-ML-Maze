from __future__ import annotations

import sys
from dataclasses import dataclass
from time import monotonic
from typing import Any

import cv2
import numpy as np

_BACKENDS = {
    "auto": None,
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "v4l2": cv2.CAP_V4L2,
}


def _decode_fourcc(value: float) -> str:
    code = int(value)
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    if all(32 <= ord(char) <= 126 for char in chars):
        return chars
    return str(code)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    timestamp_s: float


class CameraCapture:
    def __init__(self, config: dict):
        self.config = config
        self.cap: cv2.VideoCapture | None = None

    def requested_settings(self) -> dict[str, Any]:
        return {
            "device_index": int(self.config["device_index"]),
            "backend": str(self.config.get("backend", "auto")).lower(),
            "fourcc": str(self.config.get("fourcc", "MJPG")),
            "width": int(self.config["width"]),
            "height": int(self.config["height"]),
            "fps": int(self.config["fps"]),
            "flip_horizontal": bool(self.config.get("flip_horizontal", False)),
            "flip_vertical": bool(self.config.get("flip_vertical", False)),
        }

    def observed_settings(self) -> dict[str, Any]:
        if self.cap is None:
            raise RuntimeError("Camera is not open")
        return {
            "backend": int(self.cap.get(cv2.CAP_PROP_BACKEND)),
            "fourcc": _decode_fourcc(self.cap.get(cv2.CAP_PROP_FOURCC)),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
        }

    def open(self) -> None:
        device_index = int(self.config["device_index"])
        # Windows: the default MSMF backend can take 10-30 s to probe/open a
        # UVC camera; DirectShow opens near-instantly. Other platforms keep
        # OpenCV's default. Override with camera.backend in the config.
        backend_name = str(self.config.get("backend", "auto")).lower()
        backend = _BACKENDS.get(backend_name)
        if backend is None:  # "auto"
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(device_index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera device {device_index}")

        # MJPG lets UVC cameras deliver high FPS over USB2 (default YUY2 often
        # caps at ~5-10 FPS at 640x480+). Set fourcc before the mode.
        fourcc = str(self.config.get("fourcc", "MJPG"))
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config["width"]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config["height"]))
        self.cap.set(cv2.CAP_PROP_FPS, int(self.config["fps"]))
        # Keep the internal frame queue at 1 so the control loop always acts
        # on the newest frame instead of stale buffered ones.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Frame:
        if self.cap is None:
            raise RuntimeError("Camera is not open")
        ok, image = self.cap.read()
        if not ok or image is None:
            raise RuntimeError("Could not read camera frame")
        if self.config.get("flip_horizontal", False):
            image = cv2.flip(image, 1)
        if self.config.get("flip_vertical", False):
            image = cv2.flip(image, 0)
        return Frame(image=image, timestamp_s=monotonic())

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
