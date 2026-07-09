# Calibration Files

Store generated calibration files here.

Expected files later:

- `camera_intrinsics.npz` - camera matrix and distortion coefficients.
- `board_homography.npz` - image-to-board coordinate transform.
- `marker_points.csv` - selected image and board reference points.

Useful scripts:

- `scripts/calibrate_camera_intrinsics.py` - estimate camera intrinsics from `CharUco_*.png` images.
- `scripts/calibrate_charuco_homography.py` - estimate the board homography, optionally using saved intrinsics to undistort CharUco points first.

These generated files are ignored by Git by default because they are machine/setup specific.

