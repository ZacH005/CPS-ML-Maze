from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    reprojection_error: float
    new_camera_matrix: np.ndarray | None = None

    def save(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray] = {
            "camera_matrix": np.asarray(self.camera_matrix, dtype=np.float64),
            "dist_coeffs": np.asarray(self.dist_coeffs, dtype=np.float64),
            "image_size": np.asarray(self.image_size, dtype=np.int32),
            "reprojection_error": np.asarray(self.reprojection_error, dtype=np.float64),
        }
        if self.new_camera_matrix is not None:
            payload["new_camera_matrix"] = np.asarray(self.new_camera_matrix, dtype=np.float64)
        np.savez(Path(path), **payload)

    @classmethod
    def load(cls, path: str | Path) -> "CameraIntrinsics":
        data = np.load(Path(path), allow_pickle=False)
        new_camera_matrix = data["new_camera_matrix"] if "new_camera_matrix" in data.files else None
        image_size = tuple(int(v) for v in np.asarray(data["image_size"]).tolist())
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64),
            image_size=image_size,
            reprojection_error=float(np.asarray(data["reprojection_error"]).item()),
            new_camera_matrix=None if new_camera_matrix is None else np.asarray(new_camera_matrix, dtype=np.float64),
        )

    @property
    def rectification_camera_matrix(self) -> np.ndarray:
        return self.new_camera_matrix if self.new_camera_matrix is not None else self.camera_matrix


def undistort_image(image: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    return cv2.undistort(
        image,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        None,
        intrinsics.rectification_camera_matrix,
    )


def undistort_points(points_px: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float32)
    if points.size == 0:
        return points.reshape(-1, 1, 2)
    if points.ndim == 2:
        points = points[:, None, :]
    return cv2.undistortPoints(
        points,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        None,
        intrinsics.rectification_camera_matrix,
    )