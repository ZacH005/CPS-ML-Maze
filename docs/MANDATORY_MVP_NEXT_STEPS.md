# Mandatory MVP Features And Next Steps

## Goal

Move the ball from the start area toward the finish using camera feedback.

The MVP does not require a metric maze homography, CharUco-based board calibration, reinforcement learning, or a perfect full-board coordinate model. It requires a reliable closed loop on the real hardware.

## Mandatory Features

### 1. Stable Camera Convention

The camera image orientation must be fixed before collecting waypoints or tuning control.

Required:

- Set `camera.flip_vertical` correctly in `configs/default.yaml`.
- Use the same camera settings for waypoint collection, live ball tracking, and autonomous runs.
- Do not reuse calibration or waypoint files after changing flip, resolution, crop, or camera position.

Done when:

- The live image has a documented orientation.
- Start, finish, left, and right are unambiguous in the displayed camera frame.

### 2. Reliable Ball Detection

The ball must be detected frame by frame in the current lighting and camera position.

Required:

- Tune `BrightBlobBallTracker` thresholds on the actual live feed.
- Confirm the detector ignores holes, walls, glare, printed numbers, and the CharUco board.
- Log missed detections and false detections.

Done when:

- The ball center is stable enough for control while the board is moving.
- Missed detections cause neutral servo commands, not runaway motion.

### 3. Pixel-Space Waypoints

Use manually clicked waypoints in camera/image coordinates for the first working solver.

Required:

- Create a waypoint collection script or simple CSV format using `x_px,y_px`.
- Record waypoints from the same camera orientation used at runtime.
- Place waypoints along the playable route, not just at maze corners.

Done when:

- A saved waypoint path overlays correctly on the live camera image.
- The path is dense enough that each local target is reachable with simple feedback.

### 4. Servo Direction And Limits

Before autonomous control, prove which command direction moves the ball in the image.

Required:

- Keep firmware limits conservative.
- Test yaw and pitch independently.
- Record command sign mapping: positive yaw moves ball image direction, positive pitch moves ball image direction.
- Add config flags for invert or swap behavior if needed.

Done when:

- A small positive command produces a known repeatable ball acceleration direction.
- Neutral command stops active pushing and does not drive into mechanical stops.

### 5. Basic Feedback Controller

The controller should initially operate in image coordinates.

Required:

- Compute `error_px = target_px - ball_px`.
- Convert image-space error to yaw and pitch commands using the measured axis mapping.
- Clamp commands hard.
- Advance to the next waypoint when the ball is within a configured pixel radius.
- Send neutral when the ball is lost or the run is complete.

Done when:

- The ball can move through a short sequence of 3 to 5 waypoints without manual command input.

### 6. Run Logging

Every autonomous run must leave enough data to diagnose failure.

Required log fields:

- timestamp
- ball found
- ball `x_px,y_px`
- target `x_px,y_px`
- waypoint index
- yaw command
- pitch command
- run state

Done when:

- A failed run can be replayed mentally from the CSV without guessing what the controller saw.

## Deferred Work

### CharUco Homography

CharUco is not mandatory for the MVP.

The CharUco board is glued onto the side/outside of the board. That makes it a weak primary calibration target for the rolling maze plane. A homography maps one plane; if the CharUco board is not coplanar with the ball surface, it does not directly define the ball-plane transform.

Keep CharUco only for:

- optional camera intrinsic calibration
- detector debugging
- future metric calibration if the target is proven coplanar or replaced with board-plane markers

Do not block MVP progress on CharUco.

### Metric Millimeter Coordinates

Metric coordinates are useful later, but not required to prove closed-loop control.

Defer until:

- ball detection works
- servo direction mapping is known
- pixel-space waypoint following works on a short route

### Full Maze Optimization

Do not tune for the full maze first.

Work in stages:

1. Hold neutral safely.
2. Move the ball in one intended direction.
3. Reach one target.
4. Reach 3 to 5 waypoints.
5. Run one maze section.
6. Extend to the full path.

## Top Implementation Changes Needed

1. Add a pixel waypoint format and loader.
2. Add a waypoint collection or overlay script for the live camera image.
3. Add an autonomous pixel-space runner separate from the millimeter homography runner.
4. Add config for camera-axis to servo-axis mapping.
5. Update logging to include pixel coordinates and waypoint state.
6. Fix stale `3220 x 2820 mm` references to `322 x 282 mm` where metric calibration remains.
7. Quarantine old `calibration/board_homography.npz` unless it is regenerated from the current fixed camera setup.

## Immediate Next Steps

1. Set `camera.flip_vertical` correctly.
2. Run the live camera check and confirm the displayed orientation.
3. Tune ball detection on the current physical setup.
4. Collect a short pixel waypoint path for the first maze segment.
5. Test servo direction with tiny commands and record axis mapping.
6. Implement the pixel-space autonomous runner.
7. Run a 3 to 5 waypoint closed-loop test.

## Non-Negotiable Safety Rules

- If the ball is not detected, command neutral.
- If the camera frame fails, command neutral.
- If the path is complete, command neutral.
- Keep servo limits conservative until the board has proven repeatable behavior.
- Do not widen servo limits to compensate for bad tracking or bad control signs.
