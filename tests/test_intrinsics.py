import numpy as np

from cps_maze.calibration.intrinsics import CameraIntrinsics, undistort_image


def test_camera_intrinsics_round_trip_and_undistort_identity(tmp_path):
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.zeros((1, 5), dtype=float),
        image_size=(8, 6),
        reprojection_error=0.123,
    )

    path = tmp_path / "camera_intrinsics.npz"
    intrinsics.save(path)

    loaded = CameraIntrinsics.load(path)
    image = np.arange(24, dtype=np.uint8).reshape(6, 4)

    assert np.allclose(loaded.camera_matrix, intrinsics.camera_matrix)
    assert np.allclose(loaded.dist_coeffs, intrinsics.dist_coeffs)
    assert loaded.image_size == intrinsics.image_size
    assert loaded.reprojection_error == intrinsics.reprojection_error
    assert np.array_equal(undistort_image(image, loaded), image)