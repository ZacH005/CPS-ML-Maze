# Software Stack: What Is Set Up and Fixed

Reference for the paper. Everything below is implemented, validated on the
real rig, and will not change. Tuning values (controller gains, speeds,
brightness thresholds) are intentionally NOT listed here because they are
still being adjusted.

Repository: https://github.com/ZacH005/CPS-ML-Maze

## 1. System architecture

The system is a classical closed-loop cyber-physical stack. No machine
learning and no reinforcement learning are used; the camera is the only
feedback sensor (the RC servos provide no position feedback).

```
fixed overhead camera (OV9281, global shutter, 1280x800)
  -> ball detection (motion + specular highlight tracker)
  -> homography: image pixels -> board millimetres
  -> path association (windowed, wall-aware)
  -> controller (board-frame command)
  -> axis map (board frame -> servo yaw/pitch)
  -> serial link -> Arduino UNO R4 -> PCA9685 -> 2 servos -> board tilt
```

The control loop runs on the host PC in Python. One iteration per camera
frame: detect the ball, convert to board coordinates, find where it is along
the annotated route, compute a tilt command, send it to the Arduino.

## 2. Coordinate system and calibration

- Board frame: millimetres, origin at the play-area top-left corner, x to
  the right, y downward. Play area measured at 263 mm x 222 mm inside the
  walls.
- The image-to-board mapping is a planar homography stored in
  `calibration/board_homography.npz`. It is calibrated by clicking the four
  play-area corners in a live view (`scripts/calibrate_homography.py`) and
  verified with a reprojected grid overlay. A ChArUco-based calibration
  script exists as an alternative and writes the same file format; it is
  only geometrically valid when the printed pattern lies flat on the play
  surface (a homography maps exactly one plane).
- All derived artifacts (route, holes, wall mask) are stored in board
  millimetres, so they survive camera moves; only the homography must be
  recalibrated when the camera pose changes. Fixed rule: a new homography
  requires regenerating the derived artifacts.

## 3. One-time board annotation (all implemented and in use)

The maze is static, so board knowledge is captured once per setup:

- Route: `scripts/auto_trace_path.py` automatically traces the printed
  guide line on the board. It isolates thin dark structures by
  morphological thickness separation (the line is thinner than walls),
  orders the points with a gap-jumping greedy trace (the line disappears
  under walls), and simplifies to waypoints. Saved as
  `configs/maze_path_auto.csv` (x_mm, y_mm). A manual click-based
  annotation tool exists as fallback and writes the same format.
- Holes: `scripts/auto_detect_holes.py` finds the holes by thresholding a
  rectified top-down view and filtering blobs by size and circularity, with
  manual click correction. Saved as `configs/maze_holes.csv`
  (x_mm, y_mm, radius_mm).
- Walls: `scripts/build_wall_mask.py` rasterizes the walls once into an
  obstacle mask in board space (`calibration/wall_mask.npz`). Thin printed
  lines are excluded by morphological opening; everything outside the play
  area counts as blocked.

## 4. Perception

Ball detection is the motion + specular-highlight tracker (package module
`cps_maze/vision/ball_pipeline.py`, shared by the offline video pipeline and
all live tools):

- Two complementary cues: frame-to-frame motion (works while the ball
  moves; holes do not move) and near-saturated specular glint (works while
  the ball is stationary; the metal ball glints brighter than holes or
  printed text).
- Static confusers: an offline calibration pass over a recorded video finds
  board locations that are bright suspiciously often (hole rims, glare) and
  permanently excludes them. An ROI polygon excludes everything outside the
  playable surface.
- Track state machine with statuses seed / detected / predicted / lost;
  short gaps are bridged by constant-velocity prediction.
- Seeding policy for demos: click-to-seed. The operator clicks the ball in
  the live window; automatic seeding exists but is not relied on.
- Ball velocity is estimated with a low-pass filtered finite difference.

## 5. Planning and path following (structure)

- The route is a waypoint polyline in board millimetres. The ball's
  position is projected onto it to obtain path progress.
- Association is windowed: the projection may only move within a bounded
  progress window per frame, because the ball cannot teleport along the
  route between frames. This prevents locking onto physically adjacent but
  topologically distant corridors (the maze snakes, so corridors metres
  apart in path order sit millimetres apart behind one wall).
- With the wall mask loaded, candidate projections whose straight line of
  sight from the ball crosses a wall are rejected outright.
- Path curvature ahead of the ball is measured as accumulated absolute
  turning (not endpoint tangent difference, which cancels in chicanes) and
  is used to slow the ball before corners. Wall proximity, from a
  precomputed distance transform, imposes an additional slowdown near
  walls.

## 6. Control and safety (structure)

- The controller works in the board frame; a measured 2x2 axis map
  (`calibration/axis_map.npz`, produced by `scripts/axis_check.py`, which
  pulses each servo axis and measures the ball's response with the camera)
  converts board-frame commands to servo yaw/pitch. This absorbs any
  channel swaps or sign flips in the physical build.
- Static friction is compensated explicitly: below a measurable tilt the
  ball does not move at all, so a sustained commanded-but-not-moving state
  triggers a breakaway command floor.
- Safety layers, all active in every run:
  - commands are clamped to a configurable cap and rate-limited (slew)
    before reaching hardware;
  - runs start through an arming phase (operator clicks the ball, then
    explicitly starts; the run cannot begin without a tracked ball);
  - the board is commanded to neutral whenever the ball is not confidently
    detected, on finish, on timeout, and on operator abort;
  - independently of the host, the firmware returns both axes to neutral
    within 500 ms if commands stop arriving (watchdog), clamps pulse widths
    to a safe range, and ramps rather than steps between targets.

## 7. Hardware interface and firmware (fixed)

- Host to Arduino: USB serial at 500000 baud, line-oriented ASCII protocol:
  `SET <yaw> <pitch>` with normalized values in [-1, 1], `NEUTRAL`, and
  `PING`/`PONG`. The firmware maps normalized commands to servo pulse
  widths around a 1500 us neutral.
- Firmware: non-blocking serial reader, 200 Hz servo update schedule,
  I2C at 400 kHz to a PCA9685 PWM driver (address 0x40, 50 Hz servo frame),
  channel 0 and channel 1 driving the two tilt axes.
- Firmware safety (independent of all host software): hard pulse-width
  clamp, slew-rate ramping between targets, 500 ms neutral watchdog.

## 8. Run instrumentation and evaluation method

- Every autonomous run writes a CSV log: timestamp, detection flag, ball
  position and velocity (mm, mm/s), path progress, target, board-frame
  command, and the final servo commands.
- `scripts/analyze_run.py` computes the evaluation metrics from a log:
  detection rate, share of the route reached, ball speed statistics,
  cross-track error statistics (median / p90 / max distance from the route
  centerline), and stall episodes located by path position. These are the
  quantitative metrics used for tuning and will be the basis of the
  evaluation section.

## 9. Software engineering setup

- Python package `cps_maze` under `src/` (camera, vision, calibration,
  planning, control, hardware, logging), with operator scripts under
  `scripts/` and a pytest suite under `tests/` covering the pure logic
  (controllers, path association, curvature, calibration mappings, wall
  map, tracker behavior).
- Camera capture is cross-platform (DirectShow backend on Windows for fast
  open, MJPG mode for full frame rate, single-frame buffer so the control
  loop always acts on the newest frame).
- Machine-specific settings (serial port, camera device index) live in a
  gitignored `configs/local.yaml` overlay; shared configuration lives in
  `configs/default.yaml`.
- Development followed a staged bring-up: electronics smoke test, servo
  direction checks, calibration, perception, teleoperation, then closed
  loop, with each stage validated on hardware before the next. Manual
  teleoperation tools (keyboard and touchpad) exist for testing and for
  driving the ball during calibration recordings.
